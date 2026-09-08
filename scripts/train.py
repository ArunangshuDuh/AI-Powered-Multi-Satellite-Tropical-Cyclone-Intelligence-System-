"""
Training script for the TC intensity + trend model, trained on real HURDAT2 sequences.
"""
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.model import TCModel, NUM_CATEGORIES

CACHE_DIR = "cache"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [1, 2, 4]  # steps -> +6h, +12h, +24h


def cyclical_encode(month, hour):
    return [
        math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12),
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
    ]


class TCDataset(Dataset):
    def __init__(self, sequences, norm_stats=None):
        self.sequences = sequences
        if norm_stats is None:
            self.norm_stats = self._compute_norm_stats()
        else:
            self.norm_stats = norm_stats

    def _compute_norm_stats(self):
        all_hist = np.array([s["history"] for s in self.sequences]).reshape(-1, 6)
        mean = all_hist.mean(axis=0)
        std = all_hist.std(axis=0) + 1e-6

        # Separately normalize wind/pressure targets (regression + trend heads),
        # since raw-scale wind (~kt) and pressure (~mb) losses otherwise dwarf the
        # classification loss and destabilize multi-task training.
        all_wind = np.array([s["current_wind_kt"] for s in self.sequences])
        all_pressure = np.array([s["current_pressure_mb"] for s in self.sequences])
        wind_mean, wind_std = float(all_wind.mean()), float(all_wind.std() + 1e-6)
        pressure_mean, pressure_std = float(all_pressure.mean()), float(all_pressure.std() + 1e-6)

        return {
            "mean": mean.tolist(), "std": std.tolist(),
            "wind_mean": wind_mean, "wind_std": wind_std,
            "pressure_mean": pressure_mean, "pressure_std": pressure_std,
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        hist = np.array(s["history"], dtype=np.float32)
        mean = np.array(self.norm_stats["mean"], dtype=np.float32)
        std = np.array(self.norm_stats["std"], dtype=np.float32)
        hist_norm = (hist - mean) / std

        meta = np.array(cyclical_encode(s["month"], s["hour"]), dtype=np.float32)

        wm, ws = self.norm_stats["wind_mean"], self.norm_stats["wind_std"]
        pm, ps = self.norm_stats["pressure_mean"], self.norm_stats["pressure_std"]

        category = s["current_category_code"]
        regression_target = np.array([
            (s["current_wind_kt"] - wm) / ws,
            (s["current_pressure_mb"] - pm) / ps,
        ], dtype=np.float32)

        trend_target = []
        for h in HORIZONS:
            t = s["targets"][str(h)] if str(h) in s["targets"] else s["targets"][h]
            trend_target.extend([(t["wind_kt"] - wm) / ws, (t["pressure_mb"] - pm) / ps])
        trend_target = np.array(trend_target, dtype=np.float32)

        return {
            "history": torch.from_numpy(hist_norm),
            "meta": torch.from_numpy(meta),
            "category": torch.tensor(category, dtype=torch.long),
            "regression_target": torch.from_numpy(regression_target),
            "trend_target": torch.from_numpy(trend_target),
            "current_wind": torch.tensor(s["current_wind_kt"], dtype=torch.float32),
            "current_pressure": torch.tensor(s["current_pressure_mb"], dtype=torch.float32),
        }


def compute_persistence_mae(dataset):
    """Compute the persistence baseline MAE on wind for the given dataset, for reporting."""
    errors = []
    for s in dataset.sequences:
        for h in HORIZONS:
            t = s["targets"][str(h)] if str(h) in s["targets"] else s["targets"][h]
            errors.append(abs(t["wind_kt"] - s["current_wind_kt"]))
    return float(np.mean(errors))


def run_epoch(model, loader, optimizer, device, lam=0.3, train=True):
    model.train() if train else model.eval()
    total_loss, total_cls_loss, total_reg_loss, total_trend_loss = 0, 0, 0, 0
    correct, total = 0, 0
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            history = batch["history"].to(device)
            meta = batch["meta"].to(device)
            category = batch["category"].to(device)
            regression_target = batch["regression_target"].to(device)
            trend_target = batch["trend_target"].to(device)

            out = model(history, meta)

            cls_loss = ce_loss(out["category_logits"], category)
            reg_loss = mse_loss(out["regression"], regression_target)
            trend_loss = mse_loss(out["trend"], trend_target)
            loss = cls_loss + lam * reg_loss + lam * trend_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * history.size(0)
            total_cls_loss += cls_loss.item() * history.size(0)
            total_reg_loss += reg_loss.item() * history.size(0)
            total_trend_loss += trend_loss.item() * history.size(0)
            preds = out["category_logits"].argmax(dim=1)
            correct += (preds == category).sum().item()
            total += history.size(0)

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "reg_loss": total_reg_loss / n,
        "trend_loss": total_trend_loss / n,
        "accuracy": correct / total,
    }


def sanity_check():
    print("=" * 60)
    print("SANITY CHECK: training on 100 samples, confirming loss drops")
    print("=" * 60)
    with open(f"{CACHE_DIR}/train_sequences.json") as f:
        train_seqs = json.load(f)

    small = train_seqs[:100]
    ds = TCDataset(small)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TCModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(5):
        metrics = run_epoch(model, loader, optimizer, device, train=True)
        losses.append(metrics["loss"])
        print(f"  epoch {epoch+1}: loss={metrics['loss']:.4f} (cls={metrics['cls_loss']:.4f} "
              f"reg={metrics['reg_loss']:.1f} trend={metrics['trend_loss']:.1f}) acc={metrics['accuracy']:.2%}")

    assert losses[-1] < losses[0], "Loss did not decrease — something is wrong before full training"
    assert not any(math.isnan(l) for l in losses), "NaN loss detected"
    print("PASSED: loss decreased, no NaNs. Safe to proceed to full training.\n")


def full_train(epochs=15, batch_size=32, lr=1e-3, lam=0.3):
    print("=" * 60)
    print("FULL TRAINING on real HURDAT2 sequences")
    print("=" * 60)
    with open(f"{CACHE_DIR}/train_sequences.json") as f:
        train_seqs = json.load(f)
    with open(f"{CACHE_DIR}/val_sequences.json") as f:
        val_seqs = json.load(f)
    with open(f"{CACHE_DIR}/category_counts.json") as f:
        cat_counts = json.load(f)

    train_ds = TCDataset(train_seqs)
    val_ds = TCDataset(val_seqs, norm_stats=train_ds.norm_stats)  # use train stats on val, no leakage

    # Weighted sampling to balance rare categories (Cat3/4/5 are rare in real data)
    weights_per_class = {int(k): 1.0 / v for k, v in cat_counts.items()}
    sample_weights = [weights_per_class[s["current_category_code"]] for s in train_seqs]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = TCModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    persistence_mae = compute_persistence_mae(val_ds)
    print(f"Persistence baseline wind MAE (val): {persistence_mae:.2f} kt\n")

    best_val_loss = float("inf")
    history_log = []
    start = time.time()

    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device, lam=lam, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, lam=lam, train=False)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"train_loss={train_metrics['loss']:.3f} acc={train_metrics['accuracy']:.2%} | "
              f"val_loss={val_metrics['loss']:.3f} acc={val_metrics['accuracy']:.2%}")

        history_log.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_state": model.state_dict(),
                "norm_stats": train_ds.norm_stats,
                "val_metrics": val_metrics,
            }, f"{OUT_DIR}/best_model.pt")

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s. Best val loss: {best_val_loss:.3f}")

    # Compute model wind MAE on val for the baseline comparison claim (denormalized to real kt)
    wm, ws = train_ds.norm_stats["wind_mean"], train_ds.norm_stats["wind_std"]
    model.eval()
    all_pred_wind, all_true_wind = [], []
    with torch.no_grad():
        for batch in val_loader:
            history = batch["history"].to(device)
            meta = batch["meta"].to(device)
            out = model(history, meta)
            trend = out["trend"].cpu().numpy()  # (batch, 6) -> [+6h_wind, +6h_p, +12h_wind, +12h_p, +24h_wind, +24h_p]
            pred_wind = trend[:, [0, 2, 4]] * ws + wm
            true_wind = batch["trend_target"].numpy()[:, [0, 2, 4]] * ws + wm
            all_pred_wind.append(pred_wind)
            all_true_wind.append(true_wind)
    all_pred_wind = np.concatenate(all_pred_wind)
    all_true_wind = np.concatenate(all_true_wind)
    model_mae = float(np.mean(np.abs(all_pred_wind - all_true_wind)))

    print(f"\nFINAL REPORTABLE METRICS:")
    print(f"  Persistence baseline wind MAE: {persistence_mae:.2f} kt")
    print(f"  Model wind MAE (avg across +6h/+12h/+24h): {model_mae:.2f} kt")
    improvement = (persistence_mae - model_mae) / persistence_mae * 100
    print(f"  Improvement over baseline: {improvement:.1f}%")

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({
            "persistence_wind_mae": persistence_mae,
            "model_wind_mae": model_mae,
            "improvement_pct": improvement,
            "best_val_loss": best_val_loss,
            "training_time_sec": elapsed,
            "history": history_log,
        }, f, indent=2)

    print(f"\nModel checkpoint: {OUT_DIR}/best_model.pt")
    print(f"Metrics report: {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    sanity_check()
    full_train(epochs=15)
