"""
Генератор синтетического receipts.csv для отладки пайплайна.

Схема — как в CLAUDE.md: одна строка = одна позиция в чеке.
Ключевые свойства, которые синтетика воспроизводит:
  - Редкие покупки: в среднем раз в ~3 месяца на клиента.
  - Пол клиента влияет на категорию/бренд/размер.
  - Сезонность: пуховики зимой, футболки летом и т.д.
  - Скидочники: часть клиентов покупает почти всегда по акции.
  - Каналы: магазин / сайт.
  - Клиенты и покупки распределены по 2 годам.

Запуск:
    python generate_synthetic_receipts.py --clients 5000 --out receipts.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORIES_W = [
    "dresses", "tops", "jeans", "outerwear", "shoes",
    "bags", "accessories", "lingerie", "sportswear", "knitwear",
]
CATEGORIES_M = [
    "shirts", "tshirts", "jeans", "outerwear", "shoes",
    "bags", "accessories", "underwear", "sportswear", "knitwear",
]

BRANDS = ["Aster", "Nord", "Vela", "Rowan", "Kite", "Puma", "Nike", "Zara-alike"]
COLORS = ["black", "white", "beige", "blue", "red", "green", "grey", "pink"]
SIZES_W = ["XS", "S", "M", "L", "XL"]
SIZES_M = ["S", "M", "L", "XL", "XXL"]
CHANNELS = ["store", "online"]

# Сезонные множители спроса на категорию (месяц 1..12 → множитель)
# Примитивно: outerwear зимой в 3 раза чаще; sportswear летом; dresses весна/лето.
SEASONALITY = {
    "outerwear": np.array([3, 2.5, 1.5, 0.6, 0.3, 0.2, 0.2, 0.3, 0.8, 1.5, 2.5, 3.5]),
    "knitwear":  np.array([2.5, 2, 1.5, 0.8, 0.4, 0.3, 0.3, 0.4, 1.0, 1.8, 2.3, 2.5]),
    "dresses":   np.array([0.5, 0.6, 1.0, 1.5, 2.0, 2.2, 2.0, 1.8, 1.2, 0.9, 0.6, 0.5]),
    "shoes":     np.array([1.0, 1.0, 1.2, 1.3, 1.3, 1.1, 1.0, 1.0, 1.2, 1.3, 1.1, 1.0]),
    "sportswear":np.array([0.8, 0.9, 1.1, 1.3, 1.5, 1.6, 1.6, 1.5, 1.3, 1.1, 0.9, 0.8]),
    "tshirts":   np.array([0.4, 0.5, 0.8, 1.2, 1.8, 2.2, 2.3, 2.0, 1.4, 0.9, 0.6, 0.4]),
    "tops":      np.array([0.7, 0.7, 1.0, 1.3, 1.6, 1.7, 1.6, 1.5, 1.3, 1.0, 0.8, 0.7]),
    "shirts":    np.array([0.9, 0.9, 1.1, 1.2, 1.2, 1.1, 1.0, 1.0, 1.2, 1.2, 1.0, 0.9]),
    "jeans":     np.array([1.1, 1.0, 1.1, 1.1, 1.0, 0.9, 0.8, 0.9, 1.2, 1.3, 1.2, 1.1]),
    "bags":      np.array([1.0] * 12),
    "accessories": np.array([1.0] * 12),
    "lingerie":  np.array([1.0] * 12),
    "underwear": np.array([1.0] * 12),
}

# Базовая средняя цена по категории
BASE_PRICE = {
    "outerwear": 12000, "knitwear": 6500, "dresses": 5500, "shoes": 7000,
    "sportswear": 4500, "tshirts": 1800, "tops": 3200, "shirts": 3800,
    "jeans": 4800, "bags": 5000, "accessories": 1500,
    "lingerie": 2200, "underwear": 1200,
}


@dataclass
class GenConfig:
    n_clients: int = 5000
    start: str = "2024-01-01"
    end: str = "2025-12-31"
    mean_days_between_purchases: float = 90.0  # ~ раз в 3 месяца
    seed: int = 42
    out: Path = Path("receipts.csv")


def _make_clients(cfg: GenConfig, rng: np.random.Generator) -> pd.DataFrame:
    ids = np.arange(1, cfg.n_clients + 1)
    genders = rng.choice(["F", "M"], size=cfg.n_clients, p=[0.62, 0.38])
    # Скидочник: 25% клиентов охотятся за акциями.
    discount_lover = rng.random(cfg.n_clients) < 0.25
    # Каждому — «сила» (лямбда покупок): часть клиентов активнее.
    activity = rng.gamma(shape=2.0, scale=0.5, size=cfg.n_clients).clip(0.2, 5.0)
    # Любимый бренд — с вероятностью 0.7 у клиента есть предпочтение.
    fav_brand_idx = rng.integers(0, len(BRANDS), size=cfg.n_clients)
    has_fav = rng.random(cfg.n_clients) < 0.7
    # Любимый размер (сдвиг по полу).
    size_idx = rng.integers(0, len(SIZES_W), size=cfg.n_clients)
    return pd.DataFrame({
        "client_id": ids,
        "gender": genders,
        "discount_lover": discount_lover,
        "activity": activity,
        "fav_brand": np.where(has_fav, np.array(BRANDS)[fav_brand_idx], None),
        "fav_size_idx": size_idx,
    })


def _sample_purchase_dates(
    start: pd.Timestamp, end: pd.Timestamp, activity: float, mean_gap: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Пуассоновский процесс с индивидуальной интенсивностью."""
    # Средний интервал = mean_gap / activity дней.
    scale = mean_gap / max(activity, 0.05)
    dates = []
    t = start + pd.Timedelta(days=float(rng.exponential(scale)))
    while t <= end:
        dates.append(t)
        t = t + pd.Timedelta(days=float(rng.exponential(scale)))
    return np.array(dates, dtype="datetime64[ns]")


def _pick_category(
    gender: str, month: int, rng: np.random.Generator,
) -> str:
    pool = CATEGORIES_W if gender == "F" else CATEGORIES_M
    weights = np.array([SEASONALITY.get(c, np.ones(12))[month - 1] for c in pool])
    weights = weights / weights.sum()
    return str(rng.choice(pool, p=weights))


def generate(cfg: GenConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)

    clients = _make_clients(cfg, rng)
    rows = []

    for _, c in clients.iterrows():
        dates = _sample_purchase_dates(start, end, float(c["activity"]),
                                       cfg.mean_days_between_purchases, rng)
        if len(dates) == 0:
            continue
        for d in dates:
            d_ts = pd.Timestamp(d)
            month = d_ts.month
            # 1..4 позиции в чеке
            n_items = int(rng.choice([1, 2, 3, 4], p=[0.55, 0.28, 0.12, 0.05]))
            channel = str(rng.choice(CHANNELS, p=[0.7, 0.3]))
            # Скидка: у скидочника чаще, распродажные месяцы (январь, июль).
            base_disc_prob = 0.7 if c["discount_lover"] else 0.25
            if month in (1, 7):
                base_disc_prob = min(1.0, base_disc_prob + 0.2)
            for _ in range(n_items):
                cat = _pick_category(c["gender"], month, rng)
                # Бренд: с вероятностью 0.6 берём любимый (если есть).
                if c["fav_brand"] is not None and rng.random() < 0.6:
                    brand = c["fav_brand"]
                else:
                    brand = str(rng.choice(BRANDS))
                color = str(rng.choice(COLORS))
                sizes = SIZES_W if c["gender"] == "F" else SIZES_M
                # Любимый размер + шум ±1.
                si = int(np.clip(
                    c["fav_size_idx"] + rng.integers(-1, 2), 0, len(sizes) - 1
                ))
                size = sizes[si]
                base = BASE_PRICE.get(cat, 3000)
                amount = float(rng.normal(base, base * 0.15))
                amount = max(299.0, round(amount, -1))
                is_disc = bool(rng.random() < base_disc_prob)
                disc_pct = float(rng.choice([0.1, 0.15, 0.2, 0.3, 0.4])) if is_disc else 0.0
                paid = round(amount * (1 - disc_pct), -1)
                rows.append((
                    int(c["client_id"]), d_ts, cat, brand, paid, disc_pct,
                    channel, size, color, c["gender"],
                ))

    df = pd.DataFrame(rows, columns=[
        "client_id", "date", "category", "brand", "amount", "discount",
        "channel", "size", "color", "gender",
    ])
    df = df.sort_values(["client_id", "date"]).reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=5000)
    p.add_argument("--start", type=str, default="2024-01-01")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--mean-gap", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="receipts.csv")
    args = p.parse_args()

    cfg = GenConfig(
        n_clients=args.clients, start=args.start, end=args.end,
        mean_days_between_purchases=args.mean_gap, seed=args.seed,
        out=Path(args.out),
    )
    df = generate(cfg)
    df.to_csv(cfg.out, index=False)
    print(f"Wrote {len(df):,} rows for {df['client_id'].nunique():,} clients -> {cfg.out}")
    print(df.head())
    print("\nrows per client (describe):")
    print(df.groupby("client_id").size().describe())


if __name__ == "__main__":
    main()
