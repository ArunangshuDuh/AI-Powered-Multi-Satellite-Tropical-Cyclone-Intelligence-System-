"""
Precomputes track + prediction JSON for real, well-known demo storms using the
trained model. This is what the API serves -- no live inference per request,
so the demo can't break on stage.
"""
import json
import math
import numpy as np
import pandas as pd
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.model import TCModel

CATEGORY_NAMES = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]
HORIZONS = [1, 2, 4]  # -> +6h, +12h, +24h steps

# Real, well-known storms with dramatic/clean tracks, present in this HURDAT2 subset (1980-2015).
# SANDY (2012) includes a clear late-stage intensity swing; KATRINA (2005) includes a real
# rapid-intensification stretch before landfall.
DEMO_STORM_NAMES = ["SANDY", "KATRINA", "WILMA", "ANDREW"]


def cyclical_encode(month, hour):
    return [
        math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12),
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
    ]


def wind_to_category(wind_kt):
    if wind_kt < 34: return 0
    if wind_kt < 64: return 1
    if wind_kt < 83: return 2
    if wind_kt < 96: return 3
    if wind_kt < 113: return 4
    if wind_kt < 137: return 5
    return 6


def main():
    df = pd.read_csv("data/atlantic_storms.csv", parse_dates=["date"])
    df = df[df["date"] >= "1980-01-01"].copy()
    df = df[df["maximum_sustained_wind_knots"] >= 0].copy()
    df = df.sort_values(["id", "date"])
    df["maximum_pressure"] = df.groupby("id")["maximum_pressure"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )

    ckpt = torch.load("outputs/best_model.pt", map_location="cpu", weights_only=False)
    model = TCModel()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = ckpt["norm_stats"]
    hist_mean = np.array(norm["mean"], dtype=np.float32)
    hist_std = np.array(norm["std"], dtype=np.float32)
    wm, ws = norm["wind_mean"], norm["wind_std"]
    pm, ps = norm["pressure_mean"], norm["pressure_std"]

    os.makedirs("outputs/demo_storms", exist_ok=True)
    storms_index = []

    for storm_name in DEMO_STORM_NAMES:
        matches = df[df["name"] == storm_name]
        if matches.empty:
            print(f"  SKIP {storm_name}: not found in this HURDAT2 subset")
            continue
        # pick the storm id with the most records if the name repeats across years
        storm_id = matches["id"].value_counts().idxmax()
        g = df[df["id"] == storm_id].sort_values("date").reset_index(drop=True)
        if len(g) < 5:
            print(f"  SKIP {storm_name}: too few records ({len(g)})")
            continue

        frames = []
        for i, row in g.iterrows():
            cat_code = wind_to_category(row["maximum_sustained_wind_knots"])
            frames.append({
                "timestamp": row["date"].isoformat(),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "wind_kt": float(row["maximum_sustained_wind_knots"]),
                "pressure_mb": float(row["maximum_pressure"]) if not pd.isna(row["maximum_pressure"]) else None,
                "category": CATEGORY_NAMES[cat_code],
                "category_code": cat_code,
                "image_url": f"/api/storms/{storm_id}/frame/{len(frames)}/image",
            })

        track = {"storm_id": storm_id, "name": storm_name, "frames": frames}
        with open(f"outputs/demo_storms/{storm_id}_track.json", "w") as f:
            json.dump(track, f, indent=2)

        # Predictions per frame (from index 3 onward, since we need 4 steps of history)
        predictions = {}
        history_len = 4
        for t in range(history_len - 1, len(g)):
            hist_rows = g.iloc[t - history_len + 1: t + 1].copy()
            hist_rows["month"] = hist_rows["date"].dt.month
            hist_rows["hour"] = hist_rows["date"].dt.hour
            hist = hist_rows[["latitude", "longitude", "maximum_sustained_wind_knots",
                               "maximum_pressure", "month", "hour"]]
            hist_arr = hist.values.astype(np.float32)
            hist_norm = (hist_arr - hist_mean) / hist_std

            current = g.iloc[t]
            meta = np.array(cyclical_encode(current["date"].month, current["date"].hour), dtype=np.float32)

            with torch.no_grad():
                out = model(
                    torch.from_numpy(hist_norm).unsqueeze(0),
                    torch.from_numpy(meta).unsqueeze(0),
                )
            probs = torch.softmax(out["category_logits"], dim=1)[0]
            pred_cat = int(probs.argmax())
            confidence = float(probs[pred_cat])

            trend = out["trend"][0].numpy()
            trend_denorm = []
            for i in range(3):
                wind = trend[i * 2] * ws + wm
                pressure = trend[i * 2 + 1] * ps + pm
                trend_denorm.append({"wind_kt": round(float(wind), 1), "pressure_mb": round(float(pressure), 1)})

            # rapid intensification: predicted +24h wind vs current, >= 30kt threshold
            ri_flag = (trend_denorm[2]["wind_kt"] - float(current["maximum_sustained_wind_knots"])) >= 30

            predictions[str(t)] = {
                "frame_index": t,
                "predicted_category": CATEGORY_NAMES[pred_cat],
                "predicted_category_code": pred_cat,
                "predicted_wind_kt": round(float(out["regression"][0, 0].item() * ws + wm), 1),
                "predicted_pressure_mb": round(float(out["regression"][0, 1].item() * ps + pm), 1),
                "confidence": round(confidence, 3),
                "trend": {
                    "plus_6h": trend_denorm[0],
                    "plus_12h": trend_denorm[1],
                    "plus_24h": trend_denorm[2],
                },
                "rapid_intensification_flag": bool(ri_flag),
            }

        with open(f"outputs/demo_storms/{storm_id}_predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

        storms_index.append({
            "storm_id": storm_id, "name": storm_name, "basin": "ATL",
            "year": int(g.iloc[0]["date"].year),
            "max_category": max(f["category_code"] for f in frames),
        })
        print(f"  OK {storm_name} ({storm_id}): {len(frames)} frames, {len(predictions)} predictions")

    with open("outputs/demo_storms/index.json", "w") as f:
        json.dump({"storms": storms_index}, f, indent=2)
    print(f"\n{len(storms_index)} real demo storms precomputed -> outputs/demo_storms/")


if __name__ == "__main__":
    main()
