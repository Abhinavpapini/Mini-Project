"""F1: Multi-label stutter classification with Focal Loss + Class-Balanced Sampling.

Architecture:
  HuBERT-large layer 21 (PCA-32) + MFCC-21 → concat [B, 53] → CNN-1D → Sigmoid × 5

Key design:
  - Multi-label: all 5 stutter types classified simultaneously
  - Focal Loss: gamma=2, down-weights easy negatives (fluent clips)
  - Per-class alpha weights: inverse class frequency, applied in focal loss
  - WeightedRandomSampler: sample-level weights based on label rarity
  - Metrics: per-class F1, Macro-F1, AUPRC (primary = Macro-F1)

Run command:
    python experiments/F1_focal_multilabel.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --mfcc-cache artifacts/features/mfcc/fold0/mfcc_stats.npy \
        --mfcc-dim 21 \
        --ssl-pca-dim 32 \
        --epochs 30 \
        --batch-size 256 \
        --lr 3e-4 \
        --focal-gamma 2.0 \
        --out-dir results/tables \
        --fig-dir results/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F1 Multi-label Focal Loss + Class-Balanced Sampling")
    p.add_argument("--features-root",  type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",   type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",   type=int,  default=21)
    p.add_argument("--fold",           type=str,  default="fold0")
    p.add_argument("--mfcc-cache",     type=Path, default=Path("artifacts/features/mfcc/fold0/mfcc_stats.npy"))
    p.add_argument("--mfcc-dim",       type=int,  default=21,
                   help="Number of MFCC dims to use from the 78-dim stats cache (first N cols)")
    p.add_argument("--ssl-pca-dim",    type=int,  default=32)
    p.add_argument("--clips-root",     type=Path, default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",     type=Path, default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path, default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--focal-gamma",    type=float,default=2.0)
    p.add_argument("--focal-alpha-strategy", choices=["inv_freq", "uniform"], default="inv_freq",
                   help="How to compute per-class alpha for focal loss")
    p.add_argument("--test-size",      type=float,default=0.20)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--epochs",         type=int,  default=30)
    p.add_argument("--batch-size",     type=int,  default=256)
    p.add_argument("--lr",             type=float,default=3e-4)
    p.add_argument("--weight-decay",   type=float,default=1e-4)
    p.add_argument("--dropout",        type=float,default=0.3)
    p.add_argument("--threshold",      type=float,default=0.5,
                   help="Decision threshold for converting sigmoid to binary prediction")
    p.add_argument("--out-dir",  type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",  type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/F1"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def norm_text(x: object) -> str:
    return str(x).strip()


def load_multilabel_map(
    csv_path: Path,
) -> Dict[Tuple[str, str, str], np.ndarray]:
    """Load all 5 stutter type labels simultaneously. Returns {key: [5] binary array}."""
    out: Dict[Tuple[str, str, str], np.ndarray] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (norm_text(row["Show"]), norm_text(row["EpId"]), norm_text(row["ClipId"]))
            labels = np.array(
                [1 if float(norm_text(row[t])) >= 1 else 0 for t in STUTTER_TYPES],
                dtype=np.float32,
            )
            out[key] = labels
    return out


def sorted_clip_keys(clips_root: Path) -> List[Tuple[str, str, str]]:
    keys: List[Tuple[str, str, str]] = []
    for w in sorted(clips_root.rglob("*.wav")):
        parts = w.stem.split("_")
        if len(parts) >= 3:
            keys.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return keys


def load_ssl_cache(features_root: Path, alias: str, fold: str, layer: int) -> np.ndarray:
    direct = features_root / alias / fold / f"layer_{layer}.npy"
    if direct.exists():
        return np.load(direct)
    alias_root = features_root / alias
    if alias_root.is_dir():
        for child in sorted(alias_root.iterdir()):
            cand = child / f"layer_{layer}.npy"
            if cand.exists():
                print(f"  [WARN] Using fallback fold '{child.name}' for '{alias}'")
                return np.load(cand)
    raise FileNotFoundError(f"Cannot find layer_{layer}.npy for alias '{alias}'")


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Per-sample weight for WeightedRandomSampler in multi-label setting.
    Weight = max inverse class frequency across positive labels in sample.
    Purely fluent samples (all zeros) get weight = min weight (already covered).
    """
    n, c = y.shape
    class_pos = y.sum(axis=0)  # [C] positives per class
    class_freq = class_pos / n  # [C]
    class_freq = np.where(class_freq == 0, 1.0, class_freq)  # avoid div/0
    inv_freq = 1.0 / class_freq  # [C]

    weights = np.zeros(n, dtype=np.float32)
    for i in range(n):
        pos_mask = y[i] > 0
        if pos_mask.any():
            weights[i] = inv_freq[pos_mask].max()
        else:
            weights[i] = inv_freq.min()  # fluent clips get minimum weight

    # Normalize to [1, max_weight]
    weights = weights / weights.min()
    return weights


# ---------------------------------------------------------------------------
# Focal Loss (multi-label, per-class alpha)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary focal loss for multi-label classification.
    L = -alpha * (1 - pt)^gamma * log(pt)
    where pt = sigmoid(logit) for positive class, 1-sigmoid(logit) for negative.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)  # [C] or None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B, C] float in {0,1}
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B,C]
        pt  = torch.exp(-bce)  # probability of correct class
        focal = ((1 - pt) ** self.gamma) * bce  # [B,C]

        if self.alpha is not None:
            # alpha shape [C] → broadcast to [B,C]
            # alpha_t = alpha for positive, (1-alpha) for negative
            alpha_t = self.alpha.unsqueeze(0) * targets + (1 - self.alpha.unsqueeze(0)) * (1 - targets)
            focal = alpha_t * focal

        return focal.mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class F1MultiLabelCNN(nn.Module):
    """
    1D CNN over concatenated [SSL-PCA | MFCC] features → 5-class sigmoid head.
    Treats the feature vector as a 1D sequence of length F with 1 channel.
    """

    def __init__(self, in_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # [B, 256, 1]
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),  # no sigmoid here — applied in loss/eval
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(1)        # [B, 1, F]
        z = self.encoder(z)        # [B, 256, 1]
        return self.head(z)        # [B, num_classes]


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_multilabel(
    y_true: np.ndarray, y_pred_logits: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Compute per-class + macro metrics for multi-label classification."""
    probs  = 1 / (1 + np.exp(-y_pred_logits))   # sigmoid
    y_pred = (probs >= threshold).astype(int)

    metrics: Dict[str, float] = {}

    per_class_f1  = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_pre = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)

    for i, t in enumerate(STUTTER_TYPES):
        metrics[f"f1_{t}"]        = float(per_class_f1[i])
        metrics[f"precision_{t}"] = float(per_class_pre[i])
        metrics[f"recall_{t}"]    = float(per_class_rec[i])

    metrics["macro_f1"]  = float(f1_score(y_true, y_pred, average="macro",  zero_division=0))
    metrics["micro_f1"]  = float(f1_score(y_true, y_pred, average="micro",  zero_division=0))
    metrics["macro_pre"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_rec"] = float(recall_score(y_true, y_pred, average="macro",    zero_division=0))

    # AUPRC per class + macro
    auprc_list = []
    for i, t in enumerate(STUTTER_TYPES):
        if y_true[:, i].sum() > 0:
            ap = float(average_precision_score(y_true[:, i], probs[:, i]))
        else:
            ap = 0.0
        metrics[f"auprc_{t}"] = ap
        auprc_list.append(ap)
    metrics["macro_auprc"] = float(np.mean(auprc_list))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load HuBERT-large SSL features
    # ------------------------------------------------------------------
    print(f"\nLoading HuBERT-large layer {args.hubert_layer} cache ...")
    ssl_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    if ssl_feats.ndim != 2:
        ssl_feats = ssl_feats.reshape(ssl_feats.shape[0], -1)
    n = ssl_feats.shape[0]
    print(f"  SSL features: {ssl_feats.shape}")

    # ------------------------------------------------------------------
    # 2. Load MFCC stats cache (reuse from C1)
    # ------------------------------------------------------------------
    print(f"Loading MFCC cache: {args.mfcc_cache}")
    mfcc_all = np.load(args.mfcc_cache)
    if mfcc_all.shape[0] < n:
        raise RuntimeError(
            f"MFCC cache has {mfcc_all.shape[0]} rows but SSL has {n}. "
            "Delete MFCC cache and rebuild."
        )
    mfcc_all = mfcc_all[:n, :args.mfcc_dim].astype(np.float32)
    print(f"  MFCC features (first {args.mfcc_dim} dims): {mfcc_all.shape}")

    # ------------------------------------------------------------------
    # 3. Multi-label targets (all 5 stutter types)
    # ------------------------------------------------------------------
    label_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    label_map.update(load_multilabel_map(args.sep_labels))
    label_map.update(load_multilabel_map(args.fluency_labels))

    clip_keys = sorted_clip_keys(args.clips_root)[:n]
    y = np.array(
        [label_map.get(k, np.zeros(len(STUTTER_TYPES), dtype=np.float32)) for k in clip_keys],
        dtype=np.float32,
    )  # [N, 5]

    print(f"\nDataset: {n} samples | Label distribution:")
    for i, t in enumerate(STUTTER_TYPES):
        pos = int(y[:, i].sum())
        print(f"  {t:15s}: {pos:5d} pos  ({pos/n:.2%})")

    fluent_only = int((y.sum(axis=1) == 0).sum())
    print(f"  {'Fluent (all 0)':15s}: {fluent_only:5d}  ({fluent_only/n:.2%})")

    # ------------------------------------------------------------------
    # 4. Train / test split
    # ------------------------------------------------------------------
    idx = np.arange(n)
    # Stratify on most common class (Block) for reproducibility
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y[:, 0].astype(int)
    )

    # ------------------------------------------------------------------
    # 5. Preprocessing — PCA on SSL (train only), scaler on MFCC (train only)
    # ------------------------------------------------------------------
    pca_dim = min(args.ssl_pca_dim, ssl_feats.shape[1])

    ssl_sc = StandardScaler()
    x_ssl_tr = ssl_sc.fit_transform(ssl_feats[train_idx])
    x_ssl_te = ssl_sc.transform(ssl_feats[test_idx])

    pca = PCA(n_components=pca_dim, random_state=args.seed)
    x_ssl_tr = pca.fit_transform(x_ssl_tr).astype(np.float32)
    x_ssl_te = pca.transform(x_ssl_te).astype(np.float32)
    print(f"\nSSL PCA-{pca_dim} explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    mfcc_sc = StandardScaler()
    x_mfcc_tr = mfcc_sc.fit_transform(mfcc_all[train_idx])
    x_mfcc_te = mfcc_sc.transform(mfcc_all[test_idx])

    # Fuse: [SSL-PCA | MFCC]
    x_tr = np.hstack([x_ssl_tr, x_mfcc_tr]).astype(np.float32)
    x_te = np.hstack([x_ssl_te, x_mfcc_te]).astype(np.float32)
    y_tr = y[train_idx]
    y_te = y[test_idx]
    in_dim = x_tr.shape[1]
    print(f"Fused feature dim: {in_dim}  (SSL {pca_dim} + MFCC {args.mfcc_dim})")

    # ------------------------------------------------------------------
    # 6. Focal loss: per-class alpha = inverse frequency
    # ------------------------------------------------------------------
    class_pos_rate = y_tr.mean(axis=0)  # [5] fraction positive per class (train only)
    if args.focal_alpha_strategy == "inv_freq":
        alpha_vals = 1.0 - class_pos_rate  # high alpha for rare classes
    else:
        alpha_vals = np.ones(len(STUTTER_TYPES), dtype=np.float32) * 0.5

    alpha_tensor = torch.tensor(alpha_vals, dtype=torch.float32).to(device)
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha_tensor)

    print(f"\nFocal loss alpha per class (gamma={args.focal_gamma}):")
    for t, a, r in zip(STUTTER_TYPES, alpha_vals, class_pos_rate):
        print(f"  {t:15s}: alpha={a:.4f}  pos_rate={r:.4f}")

    # ------------------------------------------------------------------
    # 7. Class-balanced weighted sampler (train only)
    # ------------------------------------------------------------------
    sample_weights = compute_sample_weights(y_tr)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # ------------------------------------------------------------------
    # 8. DataLoaders
    # ------------------------------------------------------------------
    train_ds = TensorDataset(
        torch.from_numpy(x_tr),
        torch.from_numpy(y_tr),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_te),
        torch.from_numpy(y_te),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    # ------------------------------------------------------------------
    # 9. Model, optimiser, scheduler
    # ------------------------------------------------------------------
    model = F1MultiLabelCNN(
        in_dim=in_dim, num_classes=len(STUTTER_TYPES), dropout=args.dropout
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nF1 model parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    # ------------------------------------------------------------------
    # 10. Training loop
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs ...")
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "f1_best.pt"

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)
        scheduler.step()

        # Quick val
        model.eval()
        val_logits_list, val_labels_list = [], []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                val_logits_list.append(logits.cpu().numpy())
                val_labels_list.append(yb.cpu().numpy())

        val_loss /= len(test_ds)
        val_logits = np.concatenate(val_logits_list, axis=0)
        val_labels = np.concatenate(val_labels_list, axis=0)
        val_metrics = evaluate_multilabel(val_labels, val_logits, args.threshold)
        macro_f1 = val_metrics["macro_f1"]

        row = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "val_loss":   round(val_loss, 6),
            "macro_f1":   round(macro_f1, 6),
            "lr":         scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d}  train_loss={tr_loss:.5f}  "
            f"val_loss={val_loss:.5f}  macro_f1={macro_f1:.5f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), best_ckpt)

    # ------------------------------------------------------------------
    # 11. Final evaluation (best checkpoint)
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            logits = model(xb.to(device))
            test_logits_list.append(logits.cpu().numpy())
            test_labels_list.append(yb.numpy())

    test_logits = np.concatenate(test_logits_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)
    test_metrics = evaluate_multilabel(test_labels, test_logits, args.threshold)

    print(f"\n--- Test Results (best ckpt, macro_f1={best_macro_f1:.5f}) ---")
    print(f"  Macro-F1 : {test_metrics['macro_f1']:.5f}")
    print(f"  Micro-F1 : {test_metrics['micro_f1']:.5f}")
    print(f"  Macro-Pre: {test_metrics['macro_pre']:.5f}")
    print(f"  Macro-Rec: {test_metrics['macro_rec']:.5f}")
    print(f"  Macro-AUPRC: {test_metrics['macro_auprc']:.5f}")
    print(f"\n  Per-class F1:")
    for t in STUTTER_TYPES:
        print(
            f"    {t:15s}: F1={test_metrics[f'f1_{t}']:.5f}  "
            f"P={test_metrics[f'precision_{t}']:.5f}  "
            f"R={test_metrics[f'recall_{t}']:.5f}  "
            f"AUPRC={test_metrics[f'auprc_{t}']:.5f}"
        )

    # ------------------------------------------------------------------
    # 12. Save outputs
    # ------------------------------------------------------------------
    # Per-class results CSV
    perclass_csv = args.out_dir / "f1_perclass_results.csv"
    with perclass_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type", "f1", "precision", "recall", "auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({
                "stutter_type": t,
                "f1":        test_metrics[f"f1_{t}"],
                "precision": test_metrics[f"precision_{t}"],
                "recall":    test_metrics[f"recall_{t}"],
                "auprc":     test_metrics[f"auprc_{t}"],
            })

    # Train history CSV
    hist_csv = args.out_dir / "f1_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "macro_f1", "lr"])
        w.writeheader()
        w.writerows(history)

    # Run report JSON
    best_ep = history[[r["macro_f1"] for r in history].index(max(r["macro_f1"] for r in history))]["epoch"]
    run_report = {
        "experiment":        "F1",
        "title":             "Multi-label stutter: Focal Loss + Class-Balanced Sampling",
        "device":            str(device),
        "hubert_alias":      args.hubert_alias,
        "hubert_layer":      args.hubert_layer,
        "ssl_pca_dim":       int(pca_dim),
        "mfcc_dim":          args.mfcc_dim,
        "fused_dim":         int(in_dim),
        "fold":              args.fold,
        "num_samples":       int(n),
        "focal_gamma":       args.focal_gamma,
        "focal_alpha":       {t: float(a) for t, a in zip(STUTTER_TYPES, alpha_vals)},
        "threshold":         args.threshold,
        "epochs":            args.epochs,
        "best_epoch":        int(best_ep),
        "batch_size":        args.batch_size,
        "lr":                args.lr,
        "macro_f1":          test_metrics["macro_f1"],
        "micro_f1":          test_metrics["micro_f1"],
        "macro_precision":   test_metrics["macro_pre"],
        "macro_recall":      test_metrics["macro_rec"],
        "macro_auprc":       test_metrics["macro_auprc"],
        "per_class":         {t: {
                "f1":        test_metrics[f"f1_{t}"],
                "precision": test_metrics[f"precision_{t}"],
                "recall":    test_metrics[f"recall_{t}"],
                "auprc":     test_metrics[f"auprc_{t}"],
            } for t in STUTTER_TYPES},
        "best_checkpoint":   str(best_ckpt),
        "perclass_csv":      str(perclass_csv),
        "history_csv":       str(hist_csv),
    }
    report_json = args.out_dir / "f1_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 13. Figures
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        # Training curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs_x, [r["train_loss"] for r in history], label="train loss")
        axes[0].plot(epochs_x, [r["val_loss"]   for r in history], label="val loss")
        axes[0].set_title("F1 Training Loss (multi-label focal)")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Focal Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs_x, [r["macro_f1"] for r in history], color="green", marker="o", markersize=3)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], linestyle="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("F1 Validation Macro-F1")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        curve_path = args.fig_dir / "f1_train_curves.png"
        fig.savefig(curve_path, dpi=160); plt.close(fig)
        print(f"  Saved: {curve_path}")

        # Per-class F1 bar chart
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        per_f1 = [test_metrics[f"f1_{t}"] for t in STUTTER_TYPES]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        bars = ax2.bar(STUTTER_TYPES, per_f1, color=colors, width=0.55)
        ax2.axhline(test_metrics["macro_f1"], linestyle="--", color="black", alpha=0.5,
                    label=f"Macro-F1={test_metrics['macro_f1']:.4f}")
        for bar, val in zip(bars, per_f1):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        ax2.set_ylim(0, 1.0)
        ax2.set_title("F1 Per-class F1 (Focal Loss, Multi-label)")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1 Score")
        ax2.legend(); fig2.tight_layout()
        bar_path = args.fig_dir / "f1_perclass_f1.png"
        fig2.savefig(bar_path, dpi=160); plt.close(fig2)
        print(f"  Saved: {bar_path}")

    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  F1 complete.")
    print(f"   Report    : {report_json}")
    print(f"   Per-class : {perclass_csv}")
    print(f"   History   : {hist_csv}")
    print(f"   Ckpt      : {best_ckpt}")


if __name__ == "__main__":
    main()
