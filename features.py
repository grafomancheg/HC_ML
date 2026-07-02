"""
Стартовая четвёрка признаков + дополнительные "дешёвые" фичи для модели
propensity. Все функции принимают DataFrame чеков и дату среза T
и считают признак **строго по истории до T** (антиутечка).

Работает и как источник фичей для модели, и как основа для правил без ML.

Использование:
    from features import build_client_features, build_pair_features

    Xc = build_client_features(receipts, T=pd.Timestamp("2025-06-01"))
    # -> DataFrame по client_id: RFM, fav_brand, fav_size, discount_share, ...

    Xp = build_pair_features(receipts, T=pd.Timestamp("2025-06-01"),
                             categories=["dresses", "jeans", ...])
    # -> DataFrame client_id × category с историческими признаками пары.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLS = ["client_id", "date", "category", "brand", "amount",
                 "discount", "channel", "size", "color"]


def _hist(receipts: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    if not np.issubdtype(receipts["date"].dtype, np.datetime64):
        receipts = receipts.assign(date=pd.to_datetime(receipts["date"]))
    return receipts.loc[receipts["date"] < T]


def _mode_or_na(s: pd.Series) -> object:
    if s.empty:
        return np.nan
    vc = s.value_counts()
    return vc.index[0] if len(vc) else np.nan


def build_client_features(receipts: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    """Признаки уровня клиента, по истории до T."""
    h = _hist(receipts, T)
    if h.empty:
        return pd.DataFrame(columns=[
            "client_id", "recency_days", "frequency", "monetary",
            "avg_check", "n_receipts", "fav_brand", "fav_size", "fav_color",
            "discount_share", "online_share", "days_since_first",
            "distinct_categories", "avg_gap_days",
        ])

    # RFM-компоненты
    per_client_last = h.groupby("client_id")["date"].max()
    recency = (T - per_client_last).dt.days.rename("recency_days")

    # frequency = число чеков (сгруппированных покупок)
    receipt_key = h.groupby(["client_id", "date"]).ngroup()
    n_receipts = (
        h.assign(_rk=receipt_key)
         .groupby("client_id")["_rk"].nunique().rename("n_receipts")
    )

    monetary = h.groupby("client_id")["amount"].sum().rename("monetary")
    frequency = h.groupby("client_id").size().rename("frequency")
    avg_check = (monetary / n_receipts).rename("avg_check")

    fav_brand = h.groupby("client_id")["brand"].agg(_mode_or_na).rename("fav_brand")
    fav_size = h.groupby("client_id")["size"].agg(_mode_or_na).rename("fav_size")
    fav_color = h.groupby("client_id")["color"].agg(_mode_or_na).rename("fav_color")

    disc_share = h.groupby("client_id")["discount"].apply(
        lambda s: float((s > 0).mean())
    ).rename("discount_share")
    online_share = h.groupby("client_id")["channel"].apply(
        lambda s: float((s == "online").mean())
    ).rename("online_share")

    first_dt = h.groupby("client_id")["date"].min()
    days_since_first = (T - first_dt).dt.days.rename("days_since_first")

    distinct_cats = h.groupby("client_id")["category"].nunique().rename("distinct_categories")

    # Средний интервал между чеками (для «когда слать» пригодится)
    def _avg_gap(dates: pd.Series) -> float:
        d = np.sort(dates.unique())
        if len(d) < 2:
            return np.nan
        return float(np.mean(np.diff(d).astype("timedelta64[D]").astype(int)))

    avg_gap = h.groupby("client_id")["date"].apply(_avg_gap).rename("avg_gap_days")

    out = pd.concat([
        recency, frequency, monetary, avg_check, n_receipts,
        fav_brand, fav_size, fav_color,
        disc_share, online_share, days_since_first, distinct_cats, avg_gap,
    ], axis=1).reset_index()
    out = _add_rfm_segment(out)
    return out


def _add_rfm_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Простой сегмент RFM: лучшие / новички / спящие / уходящие / прочее."""
    if df.empty:
        df["rfm_segment"] = pd.Series(dtype="object")
        return df

    r = df["recency_days"]
    f = df["frequency"]
    m = df["monetary"]

    r_score = pd.qcut(-r.fillna(-r.max() - 1), 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    f_score = pd.qcut(f.rank(method="first"), 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    m_score = pd.qcut(m.rank(method="first"), 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)

    seg = pd.Series("other", index=df.index, dtype="object")
    seg[(r_score >= 3) & (f_score >= 3) & (m_score >= 3)] = "best"
    seg[(r_score >= 3) & (f_score <= 2)] = "new"
    seg[(r_score <= 2) & (f_score >= 3)] = "sleeping"
    seg[(r_score == 1) & (f_score <= 2)] = "churn_risk"
    df["rfm_segment"] = seg
    df["r_score"] = r_score
    df["f_score"] = f_score
    df["m_score"] = m_score
    return df


def build_pair_features(
    receipts: pd.DataFrame, T: pd.Timestamp, categories: Iterable[str],
) -> pd.DataFrame:
    """
    Признаки пары client × category по истории до T.
    Возвращает long-таблицу: одна строка = client × category.
    """
    h = _hist(receipts, T)
    categories = list(categories)
    clients = sorted(h["client_id"].unique().tolist())
    if not clients or not categories:
        return pd.DataFrame(columns=[
            "client_id", "category", "cat_buys", "cat_share",
            "cat_last_days", "cat_amount",
        ])

    grid = pd.MultiIndex.from_product([clients, categories],
                                     names=["client_id", "category"]).to_frame(index=False)

    per = h.groupby(["client_id", "category"]).agg(
        cat_buys=("amount", "count"),
        cat_amount=("amount", "sum"),
        cat_last=("date", "max"),
    ).reset_index()

    per["cat_last_days"] = (T - per["cat_last"]).dt.days
    per = per.drop(columns=["cat_last"])

    out = grid.merge(per, on=["client_id", "category"], how="left")
    out[["cat_buys", "cat_amount"]] = out[["cat_buys", "cat_amount"]].fillna(0)

    total_buys = h.groupby("client_id").size().rename("total_buys")
    out = out.merge(total_buys.reset_index(), on="client_id", how="left")
    out["cat_share"] = np.where(out["total_buys"] > 0,
                                out["cat_buys"] / out["total_buys"], 0.0)
    out = out.drop(columns=["total_buys"])
    return out


def make_labels(
    receipts: pd.DataFrame, T: pd.Timestamp, horizon_days: int,
    categories: Iterable[str],
) -> pd.DataFrame:
    """
    Таргет: купил ли клиент категорию в окне [T, T+horizon_days).
    Возвращает long-таблицу client × category → y.
    """
    if not np.issubdtype(receipts["date"].dtype, np.datetime64):
        receipts = receipts.assign(date=pd.to_datetime(receipts["date"]))
    T_end = T + pd.Timedelta(days=horizon_days)
    window = receipts[(receipts["date"] >= T) & (receipts["date"] < T_end)]
    bought = window.groupby(["client_id", "category"]).size().rename("y").reset_index()
    bought["y"] = 1

    # Кандидаты — все клиенты, у кого была хоть одна покупка до T.
    hist_clients = receipts.loc[receipts["date"] < T, "client_id"].unique()
    categories = list(categories)
    grid = pd.MultiIndex.from_product([sorted(hist_clients), categories],
                                     names=["client_id", "category"]).to_frame(index=False)
    out = grid.merge(bought, on=["client_id", "category"], how="left")
    out["y"] = out["y"].fillna(0).astype(int)
    return out
