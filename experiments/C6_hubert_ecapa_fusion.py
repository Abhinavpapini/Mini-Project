"""C6: HuBERT-large + WavLM-SV speaker embedding fusion (CNN-1D, binary per stutter type).

Architecture:
  HuBERT-large (layer 21, PCA-32) ──┐
                                      ├─► concat [B, 64] → CNN-1D → binary classifier
  WavLM-SV (768-dim, PCA-32) ────────┘

Speaker embeddings: microsoft/wavlm-base-plus-sv (WavLM fine-tuned for speaker
verification on VoxCeleb). Mean-pooled last hidden state → 768-dim x-vector.
Captures speaker identity / vocal tract characteristics — same concept as ECAPA-TDNN
but using HuggingFace transformers (no SpeechBrain dependency).

Key idea: Does knowing WHO is speaking help classify stutter type?

Run command:
    python experiments/C6_hubert_ecapa_fusion.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --ecapa-cache artifacts/features/speaker-embed/fold0/embeddings.npy \
        --target Block \
        --ssl-pca-dim 32 \
        --ecapa-pca-dim 32 \
        --epochs 20 \
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
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C6 HuBERT+ECAPA-TDNN fusion (CNN-1D, binary)")
    p.add_argument("--features-root",  type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",   type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",   type=int,  default=21)
    p.add_argument("--fold",           type=str,  default="fold0")
    p.add_argument("--ecapa-cache",    type=Path,
                   default=Path("artifacts/features/speaker-embed/fold0/embeddings.npy"),
                   help="Path to speaker embedding cache (WavLM-SV or ECAPA-TDNN)")
    p.add_argument("--clips-root",     type=Path,
                   default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",     type=Path,
                   default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path,
                   default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--target", choices=STUTTER_TYPES, default="Block")
    p.add_argument("--ssl-pca-dim",    type=int,  default=32)
    p.add_argument("--ecapa-pca-dim",  type=int,  default=32)
    p.add_argument("--test-size",      type=float,default=0.20)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--epochs",         type=int,  default=20)
    p.add_argument("--batch-size",     type=int,  default=256)
    p.add_argument("--lr",             type=float,default=3e-4)
    p.add_argument("--weight-decay",   type=float,default=1e-4)
    p.add_argument("--dropout",        type=float,default=0.3)
    p.add_argument("--out-dir",  type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",  type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/C6"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def norm_text(x: object) -> str:
    return str(x).strip()


def load_label_map(csv_path: Path, target_col: str) -> Dict[Tuple[str, str, str], int]:
    out: Dict[Tuple[str, str, str], int] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (norm_text(row["Show"]), norm_text(row["EpId"]), norm_text(row["ClipId"]))
            out[key] = 1 if float(norm_text(row[target_col])) >= 1 else 0
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
    for child in sorted((features_root / alias).iterdir()):
        cand = child / f"layer_{layer}.npy"
        if cand.exists():
            return np.load(cand)
    raise FileNotFoundError(f"layer_{layer}.npy not found for alias '{alias}'")


# ---------------------------------------------------------------------------
# Model — CNN-1D binary classifier on fused features
# ---------------------------------------------------------------------------

class C6FusionCNN(nn.Module):
    """
    Binary classifier for fused [HuBERT-PCA | ECAPA-PCA] features.
    Treats the concatenated vector as a 1D sequence and applies 1D convolutions.
    """

    def __init__(self, in_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x.unsqueeze(1)))  # [B, 2]


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

    print(f"Loading ECAPA-TDNN cache: {args.ecapa_cache}")
    if not args.ecapa_cache.exists():
        raise FileNotFoundError(
            f"Speaker embed cache not found: {args.ecapa_cache}\n"
            "Run build_speaker_embed_cache.py first:\n"
            "  python experiments/build_speaker_embed_cache.py --batch-size 32 --fp16 --resume"
        )
    ecapa_feats = np.load(args.ecapa_cache)
    if ecapa_feats.ndim != 2:
        ecapa_feats = ecapa_feats.reshape(ecapa_feats.shape[0], -1)

    n = min(ssl_feats.shape[0], ecapa_feats.shape[0])
    ssl_feats   = ssl_feats[:n]
    ecapa_feats = ecapa_feats[:n]
    print(f"Samples (aligned): {n} | HuBERT={ssl_feats.shape[1]}-dim | SpeakerEmb={ecapa_feats.shape[1]}-dim")

    # ------------------------------------------------------------------
    # 2. Labels (binary, one target)
    # ------------------------------------------------------------------
    label_map: Dict[Tuple[str, str, str], int] = {}
    label_map.update(load_label_map(args.sep_labels,     args.target))
    label_map.update(load_label_map(args.fluency_labels, args.target))

    clip_keys = sorted_clip_keys(args.clips_root)[:n]
    y = np.array([label_map.get(k, 0) for k in clip_keys], dtype=np.int64)
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    print(f"Positives ({args.target}): {n_pos} | Negatives: {n_neg} | Pos rate: {n_pos/n:.2%}")

    # ------------------------------------------------------------------
    # 3. Train/test split
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # ------------------------------------------------------------------
    # 4. Per-stream PCA (train only, no leakage)
    # ------------------------------------------------------------------
    ssl_pca_dim   = min(args.ssl_pca_dim,   ssl_feats.shape[1])
    ecapa_pca_dim = min(args.ecapa_pca_dim, ecapa_feats.shape[1])

    ssl_sc = StandardScaler()
    x_ssl_tr = pca_ssl = None
    x_ssl_tr = ssl_sc.fit_transform(ssl_feats[train_idx])
    x_ssl_te = ssl_sc.transform(ssl_feats[test_idx])
    pca_ssl  = PCA(n_components=ssl_pca_dim, random_state=args.seed)
    x_ssl_tr = pca_ssl.fit_transform(x_ssl_tr).astype(np.float32)
    x_ssl_te = pca_ssl.transform(x_ssl_te).astype(np.float32)

    ecapa_sc = StandardScaler()
    x_ecapa_tr = ecapa_sc.fit_transform(ecapa_feats[train_idx])
    x_ecapa_te = ecapa_sc.transform(ecapa_feats[test_idx])
    pca_ecapa  = PCA(n_components=ecapa_pca_dim, random_state=args.seed)
    x_ecapa_tr = pca_ecapa.fit_transform(x_ecapa_tr).astype(np.float32)
    x_ecapa_te = pca_ecapa.transform(x_ecapa_te).astype(np.float32)

    print(f"HuBERT PCA-{ssl_pca_dim} expl. var : {pca_ssl.explained_variance_ratio_.sum():.3f}")
    print(f"ECAPA  PCA-{ecapa_pca_dim} expl. var : {pca_ecapa.explained_variance_ratio_.sum():.3f}")

    # Fuse by concatenation
    x_tr = np.hstack([x_ssl_tr, x_ecapa_tr])
    x_te = np.hstack([x_ssl_te, x_ecapa_te])
    y_tr = y[train_idx]
    y_te = y[test_idx]
    in_dim = x_tr.shape[1]
    print(f"Fused dim: {in_dim}  (HuBERT {ssl_pca_dim} + ECAPA {ecapa_pca_dim})")

    # ------------------------------------------------------------------
    # 5. DataLoaders
    # ------------------------------------------------------------------
    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).long())
    test_ds  = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te).long())
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    # ------------------------------------------------------------------
    # 6. Model, loss, optimiser
    # ------------------------------------------------------------------
    model = C6FusionCNN(in_dim=in_dim, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nC6 model parameters: {n_params:,}")

    pos_weight_val = n_neg / max(n_pos, 1)
    weight_tensor  = torch.tensor([1.0, pos_weight_val], device=device)
    criterion      = nn.CrossEntropyLoss(weight=weight_tensor)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs ...")
    history: List[Dict] = []
    best_f1   = -1.0
    best_ckpt = args.ckpt_dir / "c6_best.pt"

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
        val_preds, val_true = [], []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                val_preds.extend(logits.argmax(dim=1).cpu().tolist())
                val_true.extend(yb.cpu().tolist())
        val_loss /= len(test_ds)
        val_f1 = f1_score(val_true, val_preds, zero_division=0)

        history.append({
            "epoch": epoch, "train_loss": round(tr_loss, 6),
            "val_loss": round(val_loss, 6), "val_f1": round(val_f1, 6),
            "lr": scheduler.get_last_lr()[0],
        })
        print(f"epoch={epoch:02d}  train_loss={tr_loss:.5f}  val_loss={val_loss:.5f}  val_f1={val_f1:.5f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)

    # ------------------------------------------------------------------
    # 8. Final evaluation
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            logits = model(xb.to(device))
            test_preds.extend(logits.argmax(dim=1).cpu().tolist())
            test_true.extend(yb.tolist())

    acc  = accuracy_score(test_true, test_preds)
    f1   = f1_score(test_true, test_preds,       zero_division=0)
    prec = precision_score(test_true, test_preds, zero_division=0)
    rec  = recall_score(test_true, test_preds,    zero_division=0)
    try:
        auc = roc_auc_score(test_true, test_preds)
    except Exception:
        auc = float("nan")

    print(f"\n--- Test Results (best ckpt, val_f1={best_f1:.5f}) ---")
    print(f"  Accuracy : {acc:.5f}")
    print(f"  F1       : {f1:.5f}")
    print(f"  Precision: {prec:.5f}")
    print(f"  Recall   : {rec:.5f}")
    print(f"  AUROC    : {auc:.5f}")

    # ------------------------------------------------------------------
    # 9. Save outputs
    # ------------------------------------------------------------------
    hist_csv = args.out_dir / "c6_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_f1", "lr"])
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["val_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment":    "C6",
        "title":         "HuBERT-large + ECAPA-TDNN speaker embedding fusion (CNN-1D)",
        "target":        args.target,
        "device":        str(device),
        "hubert_alias":  args.hubert_alias,
        "hubert_layer":  args.hubert_layer,
        "ecapa_cache":   str(args.ecapa_cache),
        "ssl_pca_dim":   int(ssl_pca_dim),
        "ecapa_pca_dim": int(ecapa_pca_dim),
        "fused_dim":     int(in_dim),
        "num_samples":   int(n),
        "num_pos":       int(n_pos),
        "num_neg":       int(n_neg),
        "epochs":        args.epochs,
        "best_epoch":    int(best_ep),
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "accuracy":      float(acc),
        "f1":            float(f1),
        "precision":     float(prec),
        "recall":        float(rec),
        "auroc":         float(auc),
        "best_val_f1":   float(best_f1),
        "history_csv":   str(hist_csv),
        "best_checkpoint": str(best_ckpt),
    }
    report_json = args.out_dir / "c6_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 10. Figures
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_x   = [r["epoch"]     for r in history]
        val_f1s    = [r["val_f1"]    for r in history]
        best_ep_idx = int(np.argmax(val_f1s))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs_x, [r["train_loss"] for r in history], label="train loss")
        axes[0].plot(epochs_x, [r["val_loss"]   for r in history], label="val loss")
        axes[0].set_title(f"C6 Training Curves ({args.target})")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs_x, val_f1s, color="green", marker="o", markersize=3)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], linestyle="--",
                        color="red", alpha=0.6, label=f"best F1={best_f1:.4f}")
        axes[1].set_title(f"C6 Validation F1 ({args.target})")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val F1")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "c6_train_curves.png", dpi=160)
        plt.close(fig)
        print(f"  Saved: {args.fig_dir / 'c6_train_curves.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  C6 complete.")
    print(f"   Report : {report_json}")
    print(f"   History: {hist_csv}")
    print(f"   Ckpt   : {best_ckpt}")


if __name__ == "__main__":
    main()
