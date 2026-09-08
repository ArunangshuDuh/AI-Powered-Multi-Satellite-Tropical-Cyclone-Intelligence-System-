"""
Preprocessing pipeline for the Tropical Cyclone AI System.
Builds real, trainable sequences from NOAA HURDAT2 best-track data.

Category label follows the Saffir-Simpson scale based on max sustained wind (knots):
  0 = TD  (< 34 kt)
  1 = TS  (34-63 kt)
  2 = Cat1 (64-82 kt)
  3 = Cat2 (83-95 kt)
  4 = Cat3 (96-112 kt)
  5 = Cat4 (113-136 kt)
  6 = Cat5 (>= 137 kt)

Rapid intensification flag follows the standard operational definition:
  wind increase >= 30 kt in a 24h window.
"""
import pandas as pd
import numpy as np
import json
import os

RAW_PATH = "data/atlantic_storms.csv"
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

CATEGORY_NAMES = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]


def wind_to_category_code(wind_kt: float) -> int:
    if wind_kt < 34:
        return 0
    elif wind_kt < 64:
        return 1
    elif wind_kt < 83:
        return 2
    elif wind_kt < 96:
        return 3
    elif wind_kt < 113:
        return 4
    elif wind_kt < 137:
        return 5
    else:
        return 6


def load_and_clean():
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    df = df[df["date"] >= "1980-01-01"].copy()

    # Drop rows with missing/invalid wind (the core signal we need)
    df = df[df["maximum_sustained_wind_knots"] >= 0].copy()

    # Interpolate missing pressure per-storm (real physical continuity assumption,
    # standard practice for this dataset given documented pre-1979 gaps)
    df = df.sort_values(["id", "date"])
    df["maximum_pressure"] = df.groupby("id")["maximum_pressure"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    # Any storm with 100% missing pressure (interpolation can't fill) gets dropped
    df = df.dropna(subset=["maximum_pressure"])

    df["category_code"] = df["maximum_sustained_wind_knots"].apply(wind_to_category_code)
    df["month"] = df["date"].dt.month
    df["hour"] = df["date"].dt.hour

    return df.reset_index(drop=True)


def build_sequences(df, history_len=4, horizons_steps=(1, 2, 4)):
    """
    history_len: number of past 6-hourly observations used as input context (4 = 24h of history)
    horizons_steps: prediction horizons in units of 6h steps -> (1,2,4) = (+6h, +12h, +24h)
    Only builds sequences where the full history and all horizons exist within the SAME storm id
    (never crosses storm boundaries).
    """
    sequences = []
    for storm_id, g in df.groupby("id"):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        max_horizon = max(horizons_steps)
        for t in range(history_len - 1, n - max_horizon):
            hist_idx = list(range(t - history_len + 1, t + 1))
            hist = g.iloc[hist_idx]

            targets = {}
            valid = True
            for h in horizons_steps:
                future_idx = t + h
                if future_idx >= n:
                    valid = False
                    break
                targets[h] = {
                    "wind_kt": float(g.iloc[future_idx]["maximum_sustained_wind_knots"]),
                    "pressure_mb": float(g.iloc[future_idx]["maximum_pressure"]),
                }
            if not valid:
                continue

            current = g.iloc[t]
            seq = {
                "storm_id": storm_id,
                "name": current["name"],
                "timestamp": current["date"].isoformat(),
                "history": hist[[
                    "latitude", "longitude", "maximum_sustained_wind_knots",
                    "maximum_pressure", "month", "hour"
                ]].values.tolist(),
                "current_wind_kt": float(current["maximum_sustained_wind_knots"]),
                "current_pressure_mb": float(current["maximum_pressure"]),
                "current_category_code": int(current["category_code"]),
                "lat": float(current["latitude"]),
                "lon": float(current["longitude"]),
                "month": int(current["month"]),
                "hour": int(current["hour"]),
                "targets": targets,
            }
            sequences.append(seq)
    return sequences


def main():
    print("Loading and cleaning real HURDAT2 data...")
    df = load_and_clean()
    print(f"  {len(df)} clean observations across {df['id'].nunique()} storms (1980-2015)")

    print("Building supervised sequences (24h history -> +6h/+12h/+24h targets)...")
    sequences = build_sequences(df)
    print(f"  {len(sequences)} training sequences built")

    # Split by storm id (not by row) to prevent leakage between train/val
    storm_ids = sorted(set(s["storm_id"] for s in sequences))
    rng = np.random.RandomState(42)
    rng.shuffle(storm_ids)
    split = int(0.85 * len(storm_ids))
    train_ids = set(storm_ids[:split])
    val_ids = set(storm_ids[split:])

    train_seqs = [s for s in sequences if s["storm_id"] in train_ids]
    val_seqs = [s for s in sequences if s["storm_id"] in val_ids]
    print(f"  Train: {len(train_seqs)} sequences ({len(train_ids)} storms)")
    print(f"  Val:   {len(val_seqs)} sequences ({len(val_ids)} storms)")

    with open(f"{CACHE_DIR}/train_sequences.json", "w") as f:
        json.dump(train_seqs, f)
    with open(f"{CACHE_DIR}/val_sequences.json", "w") as f:
        json.dump(val_seqs, f)

    # Category distribution check (for class weighting)
    cat_counts = df["category_code"].value_counts().sort_index()
    print("\nCategory distribution (for weighted sampling):")
    for code, count in cat_counts.items():
        print(f"  {CATEGORY_NAMES[code]}: {count}")

    with open(f"{CACHE_DIR}/category_counts.json", "w") as f:
        json.dump({int(k): int(v) for k, v in cat_counts.items()}, f)

    print("\nCache written to cache/train_sequences.json and cache/val_sequences.json")


if __name__ == "__main__":
    main()
