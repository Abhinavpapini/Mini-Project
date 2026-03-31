"""F4: Multi-label stutter classification with Conformer encoder + BCE loss.

Architecture:
  HuBERT-large PCA-32 + MFCC-21 → [B, 53] → token embedding [B, 53, 64]
  → 2× ConformerBlock (MHSA + Conv + FFN) → AdaptiveAvgPool → Linear(5) → sigmoid

Key differences from F1:
  - Architecture: Conformer (MHSA + Convolution + FFN) instead of CNN-1D
  - Loss: Binary Cross Entropy (BCEWithLogitsLoss) instead of Focal Loss
  - Conformer can capture long-range dependencies across feature dimensions
  - Same class-balanced WeightedRandomSampler and multi-label setup as F1

Run command:
    python experiments/F4_conformer_bce.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --mfcc-cache artifacts/features/mfcc/fold0/mfcc_stats.npy \
        --mfcc-dim 21 \
        --ssl-pca-dim 32 \
        --d-model 64 \
        --nhead 4 \
        --num-conformer-blocks 2 \
        --conv-kernel 7 \
        --epochs 30 \
        --batch-size 256 \
        --lr 3e-4 \
        --out-dir results/tables \
        --fig-dir results/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    p = argparse.ArgumentParser(description="F4 Multi-label Conformer + BCE")
    p.add_argument("--features-root",  type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",   type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",   type=int,  default=21)
    p.add_argument("--fold",           type=str,  default="fold0")
    p.add_argument("--mfcc-cache",     type=Path, default=Path("artifacts/features/mfcc/fold0/mfcc_stats.npy"))
    p.add_argument("--mfcc-dim",       type=int,  default=21)
    p.add_argument("--ssl-pca-dim",    type=int,  default=32)
    p.add_argument("--clips-root",     type=Path, default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",     type=Path, default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path, default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    # Conformer params
    p.add_argument("--d-model",             type=int,  default=64,
                   help="Conformer model dimension")
    p.add_argument("--nhead",               type=int,  default=4,
                   help="Number of attention heads")
    p.add_argument("--num-conformer-blocks", type=int, default=2)
    p.add_argument("--conv-kernel",         type=int,  default=7,
                   help="Kernel size for depthwise conv in Conformer")
    p.add_argument("--ffn-expansion",       type=int,  default=4,
                   help="FFN inner expansion factor")
    p.add_argument("--dropout",             type=float,default=0.1)
    # Training
    p.add_argument("--test-size",      type=float,default=0.20)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--epochs",         type=int,  default=30)
    p.add_argument("--batch-size",     type=int,  default=256)
    p.add_argument("--lr",             type=float,default=3e-4)
    p.add_argument("--weight-decay",   type=float,default=1e-4)
    p.add_argument("--threshold",      type=float,default=0.5)
    # Output
    p.add_argument("--out-dir",  type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",  type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/F4"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data utilities (reused from F1)
# ---------------------------------------------------------------------------

def norm_text(x: object) -> str:
    return str(x).strip()


def load_multilabel_map(csv_path: Path) -> Dict[Tuple[str, str, str], np.ndarray]:
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
    keys = []
    for w in sorted(clips_root.rglob("*.wav")):
        parts = w.stem.split("_")
        if len(parts) >= 3:
            keys.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return keys


def load_ssl_cache(features_root: Path, alias: str, fold: str, layer: int) -> np.ndarray:
    direct = features_root / alias / fold / f"layer_{layer}.npy"
    if direct.exists():
        return np.load(direct)
    for child in sorted((features_root / alias).iterdir()):
        cand = child / f"layer_{layer}.npy"
        if cand.exists():
            return np.load(cand)
    raise FileNotFoundError(f"layer_{layer}.npy not found for alias '{alias}'")


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    n, c = y.shape
    class_freq = np.where(y.mean(axis=0) == 0, 1.0, y.mean(axis=0))
    inv_freq   = 1.0 / class_freq
    weights    = np.zeros(n, dtype=np.float32)
    for i in range(n):
        pos_mask = y[i] > 0
        weights[i] = inv_freq[pos_mask].max() if pos_mask.any() else inv_freq.min()
    return weights / weights.min()


# ---------------------------------------------------------------------------
# Conformer Building Blocks
# ---------------------------------------------------------------------------

class FeedForwardModule(nn.Module):
    """Half-step feed-forward module (used twice per ConformerBlock)."""

    def __init__(self, d_model: int, expansion: int, dropout: float) -> None:
        super().__init__()
        inner = d_model * expansion
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, inner),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.ff(x)


class ConvolutionModule(nn.Module):
    """Conformer convolution module with depthwise conv + pointwise conv."""

    def __init__(self, d_model: int, kernel: int, dropout: float) -> None:
        super().__init__()
        assert kernel % 2 == 1, "kernel must be odd for same-padding"
        self.norm   = nn.LayerNorm(d_model)
        self.pw1    = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)  # gating
        self.dw     = nn.Conv1d(d_model, d_model, kernel_size=kernel,
                                padding=kernel // 2, groups=d_model)
        self.bn     = nn.BatchNorm1d(d_model)
        self.act    = nn.SiLU()
        self.pw2    = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x = self.norm(x).transpose(1, 2)   # [B, D, T]
        x = self.pw1(x)                     # [B, 2D, T]
        x = F.glu(x, dim=1)                 # [B, D, T]   (gated linear unit)
        x = self.dw(x)                      # [B, D, T]
        x = self.bn(x)
        x = self.act(x)
        x = self.pw2(x)                     # [B, D, T]
        x = self.drop(x).transpose(1, 2)    # [B, T, D]
        return residual + x


class ConformerBlock(nn.Module):
    """
    Full Conformer block:
    x → FF(0.5) → MHSA → Conv → FF(0.5) → LayerNorm
    """

    def __init__(self, d_model: int, nhead: int, conv_kernel: int,
                 ffn_expansion: int, dropout: float) -> None:
        super().__init__()
        self.ff1   = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.conv  = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ff2   = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff1(x)                              # FF half-step

        residual = x                                 # MHSA
        xn = self.norm1(x)
        attn_out, _ = self.attn(xn, xn, xn)
        x = residual + self.drop1(attn_out)

        x = self.conv(x)                             # Conv module
        x = self.ff2(x)                              # FF half-step
        x = self.norm2(x)
        return x


# ---------------------------------------------------------------------------
# Full F4 Model
# ---------------------------------------------------------------------------

class F4ConformerMultiLabel(nn.Module):
    """
    Treats each feature dimension as a "token" in a sequence.
    Input: [B, F]  (F = ssl_pca_dim + mfcc_dim)
    Step 1: Linear projection to [B, F, d_model]  (each of F features → d_model)
    Step 2: N× ConformerBlock          → [B, F, d_model]
    Step 3: AdaptiveAvgPool over F     → [B, d_model]
    Step 4: Linear(d_model → 5)        → [B, 5]  (logits, sigmoid at inference)
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        nhead: int,
        num_blocks: int,
        conv_kernel: int,
        ffn_expansion: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)   # each scalar feature → d_model

        self.conformer = nn.Sequential(
            *[ConformerBlock(d_model, nhead, conv_kernel, ffn_expansion, dropout)
              for _ in range(num_blocks)]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)   # pool over sequence (feature) dim

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        x = x.unsqueeze(-1)                   # [B, F, 1]
        x = self.input_proj(x)                # [B, F, d_model]
        x = self.conformer(x)                 # [B, F, d_model]
        x = x.transpose(1, 2)                 # [B, d_model, F]
        x = self.pool(x).squeeze(-1)          # [B, d_model]
        return self.head(x)                   # [B, num_classes]


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_multilabel(
    y_true: np.ndarray, y_logits: np.ndarray, threshold: float
) -> Dict[str, float]:
    probs  = 1 / (1 + np.exp(-y_logits))
    y_pred = (probs >= threshold).astype(int)
    metrics: Dict[str, float] = {}

    per_f1  = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_pre = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)

    for i, t in enumerate(STUTTER_TYPES):
        metrics[f"f1_{t}"]        = float(per_f1[i])
        metrics[f"precision_{t}"] = float(per_pre[i])
        metrics[f"recall_{t}"]    = float(per_rec[i])

    metrics["macro_f1"]  = float(f1_score(y_true, y_pred, average="macro",  zero_division=0))
    metrics["micro_f1"]  = float(f1_score(y_true, y_pred, average="micro",  zero_division=0))
    metrics["macro_pre"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_rec"] = float(recall_score(y_true, y_pred, average="macro",    zero_division=0))

    auprc = []
    for i, t in enumerate(STUTTER_TYPES):
        ap = float(average_precision_score(y_true[:, i], probs[:, i])) if y_true[:, i].sum() > 0 else 0.0
        metrics[f"auprc_{t}"] = ap
        auprc.append(ap)
    metrics["macro_auprc"] = float(np.mean(auprc))
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
    # 1. Load features
    # ------------------------------------------------------------------
    print(f"\nLoading HuBERT-large layer {args.hubert_layer} cache ...")
    ssl_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    if ssl_feats.ndim != 2:
        ssl_feats = ssl_feats.reshape(ssl_feats.shape[0], -1)
    n = ssl_feats.shape[0]
    print(f"  SSL features : {ssl_feats.shape}")

    print(f"Loading MFCC cache: {args.mfcc_cache}")
    mfcc_all = np.load(args.mfcc_cache)[:n, :args.mfcc_dim].astype(np.float32)
    print(f"  MFCC features: {mfcc_all.shape}")

    # ------------------------------------------------------------------
    # 2. Multi-label targets
    # ------------------------------------------------------------------
    label_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    label_map.update(load_multilabel_map(args.sep_labels))
    label_map.update(load_multilabel_map(args.fluency_labels))

    clip_keys = sorted_clip_keys(args.clips_root)[:n]
    y = np.array(
        [label_map.get(k, np.zeros(len(STUTTER_TYPES), dtype=np.float32)) for k in clip_keys],
        dtype=np.float32,
    )

    print(f"\nDataset: {n} samples | Label distribution:")
    for i, t in enumerate(STUTTER_TYPES):
        pos = int(y[:, i].sum())
        print(f"  {t:15s}: {pos:5d} pos  ({pos/n:.2%})")
    fluent = int((y.sum(axis=1) == 0).sum())
    print(f"  {'Fluent (all 0)':15s}: {fluent:5d}  ({fluent/n:.2%})")

    # ------------------------------------------------------------------
    # 3. Train / test split
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y[:, 0].astype(int)
    )

    # ------------------------------------------------------------------
    # 4. Preprocessing (PCA + Scaler, train only)
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

    x_tr = np.hstack([x_ssl_tr, x_mfcc_tr]).astype(np.float32)
    x_te = np.hstack([x_ssl_te, x_mfcc_te]).astype(np.float32)
    y_tr, y_te = y[train_idx], y[test_idx]
    in_dim = x_tr.shape[1]
    print(f"Fused feature dim: {in_dim}  (SSL {pca_dim} + MFCC {args.mfcc_dim})")

    # ------------------------------------------------------------------
    # 5. BCE loss with pos_weight (class imbalance correction)
    # ------------------------------------------------------------------
    pos_counts = y_tr.sum(axis=0)
    neg_counts = len(y_tr) - pos_counts
    pos_weight  = torch.tensor(neg_counts / np.where(pos_counts == 0, 1, pos_counts),
                               dtype=torch.float32).to(device)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"\nBCE pos_weight per class:")
    for t, pw in zip(STUTTER_TYPES, pos_weight.cpu().tolist()):
        print(f"  {t:15s}: {pw:.4f}")

    # ------------------------------------------------------------------
    # 6. Class-balanced sampler (same as F1 for fair comparison)
    # ------------------------------------------------------------------
    sample_weights = compute_sample_weights(y_tr)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # ------------------------------------------------------------------
    # 7. DataLoaders
    # ------------------------------------------------------------------
    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    test_ds  = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    # ------------------------------------------------------------------
    # 8. Model, optimiser, scheduler
    # ------------------------------------------------------------------
    model = F4ConformerMultiLabel(
        in_dim=in_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_blocks=args.num_conformer_blocks,
        conv_kernel=args.conv_kernel,
        ffn_expansion=args.ffn_expansion,
        dropout=args.dropout,
        num_classes=len(STUTTER_TYPES),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nF4 Conformer parameters: {n_params:,}")
    print(f"  d_model={args.d_model} | nhead={args.nhead} | "
          f"blocks={args.num_conformer_blocks} | conv_k={args.conv_kernel}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    # ------------------------------------------------------------------
    # 9. Training loop
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs ...")
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "f4_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)
        scheduler.step()

        # Validation
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
    # 10. Final evaluation
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
    print(f"  Macro-F1   : {test_metrics['macro_f1']:.5f}")
    print(f"  Micro-F1   : {test_metrics['micro_f1']:.5f}")
    print(f"  Macro-Pre  : {test_metrics['macro_pre']:.5f}")
    print(f"  Macro-Rec  : {test_metrics['macro_rec']:.5f}")
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
    # 11. Save outputs
    # ------------------------------------------------------------------
    perclass_csv = args.out_dir / "f4_perclass_results.csv"
    with perclass_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type", "f1", "precision", "recall", "auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_metrics[f"f1_{t}"],
                        "precision": test_metrics[f"precision_{t}"],
                        "recall":    test_metrics[f"recall_{t}"],
                        "auprc":     test_metrics[f"auprc_{t}"]})

    hist_csv = args.out_dir / "f4_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "macro_f1", "lr"])
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment":       "F4",
        "title":            "Multi-label stutter: Conformer encoder + BCE loss",
        "device":           str(device),
        "hubert_alias":     args.hubert_alias,
        "hubert_layer":     args.hubert_layer,
        "ssl_pca_dim":      int(pca_dim),
        "mfcc_dim":         args.mfcc_dim,
        "fused_dim":        int(in_dim),
        "d_model":          args.d_model,
        "nhead":            args.nhead,
        "num_blocks":       args.num_conformer_blocks,
        "conv_kernel":      args.conv_kernel,
        "ffn_expansion":    args.ffn_expansion,
        "n_params":         int(n_params),
        "fold":             args.fold,
        "num_samples":      int(n),
        "threshold":        args.threshold,
        "epochs":           args.epochs,
        "best_epoch":       int(best_ep),
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "macro_f1":         test_metrics["macro_f1"],
        "micro_f1":         test_metrics["micro_f1"],
        "macro_precision":  test_metrics["macro_pre"],
        "macro_recall":     test_metrics["macro_rec"],
        "macro_auprc":      test_metrics["macro_auprc"],
        "per_class":        {t: {"f1": test_metrics[f"f1_{t}"],
                                 "precision": test_metrics[f"precision_{t}"],
                                 "recall":    test_metrics[f"recall_{t}"],
                                 "auprc":     test_metrics[f"auprc_{t}"]}
                             for t in STUTTER_TYPES},
        "best_checkpoint":  str(best_ckpt),
        "perclass_csv":     str(perclass_csv),
        "history_csv":      str(hist_csv),
    }
    report_json = args.out_dir / "f4_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 12. Figures
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs_x, [r["train_loss"] for r in history], label="train loss")
        axes[0].plot(epochs_x, [r["val_loss"]   for r in history], label="val loss")
        axes[0].set_title("F4 Training Loss (BCE, Conformer)")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs_x, [r["macro_f1"] for r in history],
                     color="green", marker="o", markersize=3)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], linestyle="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("F4 Validation Macro-F1")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "f4_train_curves.png", dpi=160)
        plt.close(fig)
        print(f"  Saved: {args.fig_dir / 'f4_train_curves.png'}")

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
        ax2.set_title("F4 Per-class F1 (Conformer + BCE)")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1 Score")
        ax2.legend(); fig2.tight_layout()
        fig2.savefig(args.fig_dir / "f4_perclass_f1.png", dpi=160)
        plt.close(fig2)
        print(f"  Saved: {args.fig_dir / 'f4_perclass_f1.png'}")

    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  F4 complete.")
    print(f"   Report    : {report_json}")
    print(f"   Per-class : {perclass_csv}")
    print(f"   History   : {hist_csv}")
    print(f"   Ckpt      : {best_ckpt}")


if __name__ == "__main__":
    main()
