"""
Модель «что предложить» (category propensity).

Задача: для каждой пары `клиент × категория × дата_среза T` предсказать
вероятность покупки этой категории в окне [T, T + horizon] дней.
Метрика — Precision@N (сколько из N предложенных категорий реально куплено).

Ключевые архитектурные решения:
- Формат обучающей таблицы: длинный (client × category × T).
- Много срезов T (начало каждого месяца в trained-периоде) — так из редких
  покупок получается богатый обучающий набор.
- Все признаки считаются строго по истории до T (антиутечка).
- Валидация — временной сплит: обучение на срезах T < T_valid, валидация
  на срезе T_valid.
- Оценщик: LightAutoML (TabularAutoML). Если LAMA не установлен —
  прозрачный fallback на LightGBM с тем же интерфейсом.
- Выход: offers.csv (client_id, category, score, rank).

Запуск:
    python propensity_model.py --receipts receipts.csv \
        --train-start 2024-06-01 --train-end 2025-09-01 \
        --valid-cut 2025-10-01 --score-at 2026-01-01 \
        --horizon-days 90 --top-n 3 \
        --out-offers offers.csv --out-metrics metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from features import build_client_features, build_pair_features, make_labels

logger = logging.getLogger("propensity")
warnings.filterwarnings("ignore", category=UserWarning)


# ------------------- Слайсинг: генерация срезов T -------------------

def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Список первых чисел месяца в [start, end]."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if start.day != 1:
        start = (start + pd.offsets.MonthBegin(0)).normalize()
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur)
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()
    return dates


# ------------------- Сборка обучающих строк -------------------

def build_dataset_for_cut(
    receipts: pd.DataFrame, T: pd.Timestamp, horizon_days: int,
    categories: Sequence[str],
) -> pd.DataFrame:
    """
    Для одного среза T собрать длинную таблицу:
        client_id × category × T → features + label y.
    """
    client_feats = build_client_features(receipts, T)
    if client_feats.empty:
        return pd.DataFrame()

    pair_feats = build_pair_features(receipts, T, categories)
    y = make_labels(receipts, T, horizon_days, categories)

    df = pair_feats.merge(client_feats, on="client_id", how="left")
    df = df.merge(y, on=["client_id", "category"], how="inner")

    # Календарные фичи среза — помогают модели ловить сезонность спроса.
    df["cut_month"] = int(T.month)
    df["cut_quarter"] = int((T.month - 1) // 3 + 1)
    df["cut_year"] = int(T.year)
    df["cut_date"] = T
    return df


def build_train_valid(
    receipts: pd.DataFrame,
    train_cuts: Sequence[pd.Timestamp],
    valid_cut: pd.Timestamp,
    horizon_days: int,
    categories: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Building train from %d cuts, valid on %s", len(train_cuts), valid_cut.date())
    train_parts: list[pd.DataFrame] = []
    for T in train_cuts:
        part = build_dataset_for_cut(receipts, T, horizon_days, categories)
        if not part.empty:
            train_parts.append(part)
            logger.info("  cut %s -> %d rows, pos rate=%.3f",
                        T.date(), len(part), part["y"].mean())
    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    valid = build_dataset_for_cut(receipts, valid_cut, horizon_days, categories)
    return train, valid


# ------------------- Обёртки для моделей -------------------

FEATURE_EXCLUDE = {"client_id", "y", "cut_date"}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


class BasePropensityModel:
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def feature_importance(self) -> pd.DataFrame:
        raise NotImplementedError


class LamaPropensityModel(BasePropensityModel):
    """Обёртка над LightAutoML TabularAutoML."""

    def __init__(self, timeout: int = 600, cpu_limit: int = 4) -> None:
        from lightautoml.automl.presets.tabular_presets import TabularAutoML  # noqa: F401
        from lightautoml.tasks import Task
        self._Task = Task
        self._TabularAutoML = TabularAutoML
        self.timeout = timeout
        self.cpu_limit = cpu_limit
        self.automl = None
        self.features: list[str] = []

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> None:
        self.features = _feature_columns(train)
        task = self._Task("binary", metric="auc")
        self.automl = self._TabularAutoML(
            task=task,
            timeout=self.timeout,
            cpu_limit=self.cpu_limit,
            reader_params={"n_jobs": self.cpu_limit, "cv": 5, "random_state": 42},
        )
        roles = {"target": "y", "drop": ["client_id", "cut_date"]}
        self.automl.fit_predict(train, roles=roles, verbose=1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = self.automl.predict(df)
        return np.asarray(preds.data).ravel()

    def feature_importance(self) -> pd.DataFrame:
        try:
            fi = self.automl.get_feature_scores("fast")
            return pd.DataFrame(fi).rename(columns={0: "feature", 1: "importance"})
        except Exception as e:  # pragma: no cover
            logger.warning("LAMA feature importance failed: %s", e)
            return pd.DataFrame(columns=["feature", "importance"])


class LgbmPropensityModel(BasePropensityModel):
    """Fallback на LightGBM с автоматической кодировкой категорий."""

    def __init__(self, num_leaves: int = 63, n_estimators: int = 400,
                 learning_rate: float = 0.05, random_state: int = 42) -> None:
        import lightgbm as lgb
        self._lgb = lgb
        self.params = dict(
            num_leaves=num_leaves, n_estimators=n_estimators,
            learning_rate=learning_rate, random_state=random_state,
            objective="binary", metric="auc",
        )
        self.model = None
        self.features: list[str] = []
        self.cat_features: list[str] = []

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c in X.columns:
            col = X[c]
            if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
                X[c] = col.astype("category")
        return X

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> None:
        self.features = _feature_columns(train)
        X_tr, y_tr = self._prepare(train), train["y"]
        X_va, y_va = self._prepare(valid), valid["y"]
        self.cat_features = [c for c in X_tr.columns
                             if isinstance(X_tr[c].dtype, pd.CategoricalDtype)]
        self.model = self._lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=self.cat_features,
            callbacks=[self._lgb.early_stopping(30), self._lgb.log_evaluation(0)],
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare(df)
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return (pd.DataFrame({"feature": self.features, "importance": imp})
                  .sort_values("importance", ascending=False)
                  .reset_index(drop=True))


def make_model(prefer_lama: bool = True) -> BasePropensityModel:
    if prefer_lama:
        try:
            return LamaPropensityModel()
        except Exception as e:
            logger.warning("LightAutoML unavailable (%s) — falling back to LightGBM.", e)
    return LgbmPropensityModel()


# ------------------- Оценка: Precision@N -------------------

def precision_at_n(preds: pd.DataFrame, n: int = 3) -> float:
    """
    preds — DataFrame с колонками client_id, category, score, y.
    Считаем: доля топ-N категорий на клиента, у которых y == 1.
    Усредняем по клиентам, у которых был хотя бы один y == 1
    (иначе метрика тривиально ноль и разбавляет сигнал).
    """
    preds = preds.sort_values(["client_id", "score"], ascending=[True, False])
    topn = preds.groupby("client_id").head(n)
    hit = topn.groupby("client_id")["y"].sum()
    total_pos = preds.groupby("client_id")["y"].sum()
    mask = total_pos > 0
    if mask.sum() == 0:
        return float("nan")
    p = (hit[mask] / n).mean()
    return float(p)


def recall_at_n(preds: pd.DataFrame, n: int = 3) -> float:
    preds = preds.sort_values(["client_id", "score"], ascending=[True, False])
    topn = preds.groupby("client_id").head(n)
    hit = topn.groupby("client_id")["y"].sum()
    total_pos = preds.groupby("client_id")["y"].sum()
    mask = total_pos > 0
    if mask.sum() == 0:
        return float("nan")
    r = (hit[mask] / total_pos[mask]).mean()
    return float(r)


# ------------------- Оркестрация -------------------

@dataclass
class RunConfig:
    receipts_path: Path
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_cut: pd.Timestamp
    score_at: pd.Timestamp | None
    horizon_days: int = 90
    top_n: int = 3
    out_offers: Path = Path("offers.csv")
    out_metrics: Path = Path("metrics.json")
    out_importance: Path = Path("feature_importance.csv")
    prefer_lama: bool = True
    categories: list[str] = field(default_factory=list)


def run(cfg: RunConfig) -> dict:
    logger.info("Loading receipts from %s", cfg.receipts_path)
    receipts = pd.read_csv(cfg.receipts_path, parse_dates=["date"])
    required = {"client_id", "date", "category", "brand", "amount",
                "discount", "channel", "size"}
    missing = required - set(receipts.columns)
    if missing:
        raise ValueError(f"receipts.csv missing columns: {missing}")

    categories = cfg.categories or sorted(receipts["category"].unique().tolist())
    logger.info("Using %d categories: %s", len(categories), categories)

    train_cuts = month_starts(cfg.train_start, cfg.train_end)
    train, valid = build_train_valid(
        receipts, train_cuts, cfg.valid_cut, cfg.horizon_days, categories,
    )
    if train.empty or valid.empty:
        raise RuntimeError("Empty train or valid — check date ranges vs. receipts.")

    logger.info("Train: %d rows, pos rate=%.3f", len(train), train["y"].mean())
    logger.info("Valid: %d rows, pos rate=%.3f", len(valid), valid["y"].mean())

    model = make_model(prefer_lama=cfg.prefer_lama)
    model.fit(train, valid)

    valid = valid.copy()
    valid["score"] = model.predict(valid)

    metrics = {
        "n_train_rows": int(len(train)),
        "n_valid_rows": int(len(valid)),
        "train_pos_rate": float(train["y"].mean()),
        "valid_pos_rate": float(valid["y"].mean()),
        "precision_at_1": precision_at_n(valid, 1),
        "precision_at_2": precision_at_n(valid, 2),
        "precision_at_3": precision_at_n(valid, 3),
        "recall_at_3": recall_at_n(valid, 3),
        "model": type(model).__name__,
        "valid_cut": str(cfg.valid_cut.date()),
        "horizon_days": cfg.horizon_days,
    }
    logger.info("Metrics: %s", metrics)
    cfg.out_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    fi = model.feature_importance()
    if not fi.empty:
        fi.to_csv(cfg.out_importance, index=False)
        logger.info("Feature importance -> %s (top 10):\n%s",
                    cfg.out_importance, fi.head(10).to_string(index=False))

    # Финальный скоринг для рассылки — на дате score_at (по умолчанию: valid_cut).
    score_T = cfg.score_at or cfg.valid_cut
    logger.info("Scoring at T=%s for offers.csv", score_T.date())
    score_df = build_dataset_for_cut(receipts, score_T, cfg.horizon_days, categories)
    if score_df.empty:
        logger.warning("Nothing to score at %s", score_T.date())
    else:
        score_df["score"] = model.predict(score_df)
        offers = _pick_top_offers(score_df, cfg.top_n)
        offers.to_csv(cfg.out_offers, index=False)
        logger.info("Wrote %d offers -> %s", len(offers), cfg.out_offers)

    return metrics


def _pick_top_offers(scored: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Из скора client×category выбрать топ-N категорий на клиента."""
    scored = scored.sort_values(["client_id", "score"], ascending=[True, False])
    scored["rank"] = scored.groupby("client_id").cumcount() + 1
    out = scored.loc[scored["rank"] <= top_n, ["client_id", "category", "score", "rank"]]
    return out.reset_index(drop=True)


# ------------------- CLI -------------------

def _parse_ts(x: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(x, "%Y-%m-%d"))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--receipts", required=True)
    p.add_argument("--train-start", required=True, type=_parse_ts)
    p.add_argument("--train-end", required=True, type=_parse_ts)
    p.add_argument("--valid-cut", required=True, type=_parse_ts)
    p.add_argument("--score-at", default=None, type=_parse_ts)
    p.add_argument("--horizon-days", type=int, default=90)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--out-offers", type=Path, default=Path("offers.csv"))
    p.add_argument("--out-metrics", type=Path, default=Path("metrics.json"))
    p.add_argument("--out-importance", type=Path, default=Path("feature_importance.csv"))
    p.add_argument("--no-lama", action="store_true", help="Force LightGBM fallback")
    p.add_argument("--categories", nargs="*", default=None)
    args = p.parse_args()

    cfg = RunConfig(
        receipts_path=Path(args.receipts),
        train_start=args.train_start,
        train_end=args.train_end,
        valid_cut=args.valid_cut,
        score_at=args.score_at,
        horizon_days=args.horizon_days,
        top_n=args.top_n,
        out_offers=args.out_offers,
        out_metrics=args.out_metrics,
        out_importance=args.out_importance,
        prefer_lama=not args.no_lama,
        categories=args.categories or [],
    )
    run(cfg)


if __name__ == "__main__":
    main()
