"""B2: Kernel PCA (RBF) dimensionality reduction benchmark.

Pipeline:
    HuBERT-large layer 21 (1024-dim)
    → StandardScaler
    → KernelPCA(kernel='rbf', n_components=32)
    → CNN-1D multi-label classifier
    → MultiLabelSoftMarginLoss

KernelPCA with RBF maps features to a high-dimensional Hilbert space,
capturing non-linear structure that linear PCA cannot.

Practical note: kernel matrix is (N_fit × N_fit). We fit on a subsample
(default 5000 training points) to keep memory and compute tractable.
Transform for all N samples uses (N × N_fit) kernel evaluation.

Run command:
    python experiments/B2_kpca_cnn.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --pre-pca-dim 128 \
        --kpca-dim 32 \
        --kpca-gamma auto \
        --kpca-fit-samples 5000 \
        --epochs 30 \
        --batch-size 256 \
        --lr 3e-4
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
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B2: Kernel PCA (RBF) + CNN-1D, multi-label")
    p.add_argument("--features-root",   type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",    type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",    type=int,  default=21)
    p.add_argument("--fold",            type=str,  default="fold0")
    p.add_argument("--clips-root",      type=Path,
                   default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",      type=Path,
                   default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels",  type=Path,
                   default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--pre-pca-dim",      type=int,  default=128,
                   help="Linear PCA dim before KPCA (reduces 1024→pre-pca-dim first)")
    # KernelPCA params
    p.add_argument("--kpca-dim",         type=int,  default=32,
                   help="Number of KPCA components")
    p.add_argument("--kpca-gamma",       type=str,  default="auto",
                   help="RBF gamma: float or 'auto' = 1/(n_features*X.var())")
    p.add_argument("--kpca-fit-samples", type=int,  default=5000,
                   help="Subsample of training data used to fit KPCA")
    # CNN params
    p.add_argument("--cnn-channels",    type=str,  default="64,128,256")
    p.add_argument("--dropout",         type=float,default=0.3)
    # Training
    p.add_argument("--test-size",       type=float,default=0.20)
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--epochs",          type=int,  default=30)
    p.add_argument("--batch-size",      type=int,  default=256)
    p.add_argument("--lr",              type=float,default=3e-4)
    p.add_argument("--weight-decay",    type=float,default=1e-4)
    p.add_argument("--threshold",       type=float,default=0.5)
    p.add_argument("--out-dir",   type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",   type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir",  type=Path, default=Path("artifacts/checkpoints/B2"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data utilities
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
    class_freq = np.where(y.mean(axis=0) == 0, 1.0, y.mean(axis=0))
    inv_freq   = 1.0 / class_freq
    weights    = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        pos_mask   = y[i] > 0
        weights[i] = inv_freq[pos_mask].max() if pos_mask.any() else inv_freq.min()
    return weights / weights.min()


# ---------------------------------------------------------------------------
# CNN-1D classifier
# ---------------------------------------------------------------------------

class CNN1DMultiLabel(nn.Module):
    """
    1D-CNN treating the KPCA-32 dims as a 1D sequence.
    """

    def __init__(self, in_dim: int, channels: List[int],
                 num_classes: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        c_in = 1
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm1d(c_out), nn.GELU(),
            ]
            c_in = c_out
        self.cnn  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x.unsqueeze(1))   # [B, 1, D] → [B, C, D]
        h = self.pool(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_multilabel(y_true: np.ndarray, y_logits: np.ndarray,
                        threshold: float) -> Dict[str, float]:
    probs  = 1 / (1 + np.exp(-y_logits))
    y_pred = (probs >= threshold).astype(int)
    m: Dict[str, float] = {}
    pf1  = f1_score(y_true, y_pred, average=None, zero_division=0)
    ppre = precision_score(y_true, y_pred, average=None, zero_division=0)
    prec = recall_score(y_true, y_pred, average=None, zero_division=0)
    for i, t in enumerate(STUTTER_TYPES):
        m[f"f1_{t}"] = float(pf1[i]); m[f"pre_{t}"] = float(ppre[i])
        m[f"rec_{t}"] = float(prec[i])
    m["macro_f1"]  = float(f1_score(y_true, y_pred, average="macro",  zero_division=0))
    m["micro_f1"]  = float(f1_score(y_true, y_pred, average="micro",  zero_division=0))
    m["macro_pre"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    m["macro_rec"] = float(recall_score(y_true, y_pred, average="macro",    zero_division=0))
    auprc = []
    for i, t in enumerate(STUTTER_TYPES):
        ap = float(average_precision_score(y_true[:, i], probs[:, i])) if y_true[:, i].sum() > 0 else 0.0
        m[f"auprc_{t}"] = ap; auprc.append(ap)
    m["macro_auprc"] = float(np.mean(auprc))
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    for d in (args.out_dir, args.fig_dir, args.ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Features + labels
    # ------------------------------------------------------------------
    ssl_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    if ssl_feats.ndim != 2:
        ssl_feats = ssl_feats.reshape(ssl_feats.shape[0], -1)
    n = ssl_feats.shape[0]

    label_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    label_map.update(load_multilabel_map(args.sep_labels))
    label_map.update(load_multilabel_map(args.fluency_labels))
    clip_keys = sorted_clip_keys(args.clips_root)[:n]
    y = np.array(
        [label_map.get(k, np.zeros(len(STUTTER_TYPES), dtype=np.float32)) for k in clip_keys],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y[:, 0].astype(int)
    )

    # ------------------------------------------------------------------
    # 3. StandardScaler + optional pre-PCA (reduce 1024 → pre_pca_dim)
    # ------------------------------------------------------------------
    sc = StandardScaler()
    x_tr_sc = sc.fit_transform(ssl_feats[train_idx])
    x_te_sc = sc.transform(ssl_feats[test_idx])

    if args.pre_pca_dim and args.pre_pca_dim < ssl_feats.shape[1]:
        pre_pca = PCA(n_components=args.pre_pca_dim, random_state=args.seed)
        x_tr_sc = pre_pca.fit_transform(x_tr_sc)
        x_te_sc = pre_pca.transform(x_te_sc)
        n_kpca_features = args.pre_pca_dim
    else:
        n_kpca_features = ssl_feats.shape[1]

    # ------------------------------------------------------------------
    # 4. Kernel PCA (RBF) — fit on subsample of train for speed
    # ------------------------------------------------------------------
    fit_n = min(args.kpca_fit_samples, len(train_idx))
    rng   = np.random.default_rng(args.seed)
    fit_subidx = rng.choice(len(train_idx), size=fit_n, replace=False)
    x_fit  = x_tr_sc[fit_subidx]

    # Auto-compute gamma from data if requested
    if args.kpca_gamma == "auto":
        # gamma = 1 / (n_features * X.var())  — sklearn 'scale' heuristic
        gamma_val = 1.0 / (n_kpca_features * float(x_fit.var()))
    else:
        gamma_val = float(args.kpca_gamma)

    # Sanity check: expected kernel value at mean pairwise distance
    mean_sq_dist = 2.0 * n_kpca_features * float(x_fit.var())
    expected_k   = np.exp(-gamma_val * mean_sq_dist)
    # Warn if kernel values are likely near zero (gamma too large).
    if expected_k < 0.01:
        print(f"  [WARN] Kernel values near zero — gamma may be too large!")
    kpca = KernelPCA(
        n_components=args.kpca_dim,
        kernel="rbf",
        gamma=gamma_val,
        fit_inverse_transform=False,
        eigen_solver="dense",
        random_state=args.seed,
    )
    kpca.fit(x_fit)
    x_tr = kpca.transform(x_tr_sc).astype(np.float32)
    x_te = kpca.transform(x_te_sc).astype(np.float32)

    # Normalise KPCA output
    kpca_sc = StandardScaler()
    x_tr = kpca_sc.fit_transform(x_tr).astype(np.float32)
    x_te = kpca_sc.transform(x_te).astype(np.float32)

    # Reference linear PCA
    pca_ref = PCA(n_components=args.kpca_dim, random_state=args.seed)
    pca_ref.fit(x_fit)

    y_tr, y_te = y[train_idx], y[test_idx]

    # ------------------------------------------------------------------
    # 5. DataLoaders
    # ------------------------------------------------------------------
    sw = compute_sample_weights(y_tr)
    sampler  = WeightedRandomSampler(torch.from_numpy(sw), len(sw), replacement=True)
    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    test_ds  = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    channels = [int(c) for c in args.cnn_channels.split(",")]
    model    = CNN1DMultiLabel(in_dim=args.kpca_dim, channels=channels,
                               num_classes=len(STUTTER_TYPES), dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.MultiLabelSoftMarginLoss()
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                            eta_min=args.lr * 0.05)

    # ------------------------------------------------------------------
    # 7. Training
    # ------------------------------------------------------------------
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "b2_best.pt"

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

        model.eval()
        vl, vlog, vlab = 0.0, [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                lg = model(xb)
                vl += criterion(lg, yb).item() * xb.size(0)
                vlog.append(lg.cpu().numpy()); vlab.append(yb.cpu().numpy())
        vl /= len(test_ds)
        vm = evaluate_multilabel(np.concatenate(vlab), np.concatenate(vlog), args.threshold)
        macro_f1 = vm["macro_f1"]
        row = {"epoch": epoch, "train_loss": round(tr_loss, 6),
               "val_loss": round(vl, 6), "macro_f1": round(macro_f1, 6)}
        history.append(row)
        print(f"epoch={epoch:02d}  train_loss={tr_loss:.5f}  val_loss={vl:.5f}  macro_f1={macro_f1:.5f}")
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), best_ckpt)

    # ------------------------------------------------------------------
    # 8. Final evaluation
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    tlog, tlab = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            tlog.append(model(xb.to(device)).cpu().numpy())
            tlab.append(yb.numpy())
    test_m = evaluate_multilabel(np.concatenate(tlab), np.concatenate(tlog), args.threshold)

    print(f"\n--- Test Results (best ckpt, macro_f1={best_macro_f1:.5f}) ---")
    print(f"  Macro-F1   : {test_m['macro_f1']:.5f}")
    print(f"  Micro-F1   : {test_m['micro_f1']:.5f}")
    print(f"  Macro-Pre  : {test_m['macro_pre']:.5f}")
    print(f"  Macro-Rec  : {test_m['macro_rec']:.5f}")
    print(f"  Macro-AUPRC: {test_m['macro_auprc']:.5f}")
    print(f"\n  Per-class F1:")
    for t in STUTTER_TYPES:
        print(f"    {t:15s}: F1={test_m[f'f1_{t}']:.5f}  "
              f"P={test_m[f'pre_{t}']:.5f}  R={test_m[f'rec_{t}']:.5f}  "
              f"AUPRC={test_m[f'auprc_{t}']:.5f}")

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    perclass_csv = args.out_dir / "b2_perclass_results.csv"
    with perclass_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    hist_csv = args.out_dir / "b2_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "B2",
        "title": "Kernel PCA (RBF) + CNN-1D (HuBERT-large, non-linear reduction benchmark)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "kpca_dim": args.kpca_dim,
        "pre_pca_dim": args.pre_pca_dim, "kpca_gamma": gamma_val,
        "kpca_fit_samples": fit_n,
        "n_params": int(n_params), "loss": "MultiLabelSoftMarginLoss",
        "pca_ref_expl_var": float(pca_ref.explained_variance_ratio_.sum()),
        "epochs": args.epochs, "best_epoch": int(best_ep),
        "macro_f1": test_m["macro_f1"], "micro_f1": test_m["micro_f1"],
        "macro_precision": test_m["macro_pre"], "macro_recall": test_m["macro_rec"],
        "macro_auprc": test_m["macro_auprc"],
        "per_class": {t: {"f1": test_m[f"f1_{t}"], "precision": test_m[f"pre_{t}"],
                          "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]}
                      for t in STUTTER_TYPES},
        "best_checkpoint": str(best_ckpt),
    }
    (args.out_dir / "b2_run_report.json").write_text(json.dumps(run_report, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val")
        axes[0].set_title(f"B2 KPCA-RBF Loss (γ={args.kpca_gamma})")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="darkorange", marker="o", ms=4)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("B2 Validation Macro-F1"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "b2_train_curves.png", dpi=160); plt.close(fig)
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        pf1  = [test_m[f"f1_{t}"] for t in STUTTER_TYPES]
        bars = ax2.bar(STUTTER_TYPES, pf1,
                       color=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"], width=0.55)
        ax2.axhline(test_m["macro_f1"], ls="--", color="black", alpha=0.5,
                    label=f"Macro-F1={test_m['macro_f1']:.4f}")
        for b, v in zip(bars, pf1):
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_title(f"B2 Per-class F1 (KPCA-RBF-{args.kpca_dim}, γ={args.kpca_gamma})")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "b2_perclass_f1.png", dpi=160); plt.close(fig2)
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")


if __name__ == "__main__":
    main()
