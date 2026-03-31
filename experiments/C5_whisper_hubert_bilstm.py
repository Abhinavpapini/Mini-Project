"""C5: Whisper encoder + HuBERT-large cross-modal ASR+SSL fusion.

Architecture:
  HuBERT-large (layer 21, PCA-32) ──┐
                                      ├─► 2-step BiLSTM → Stream Attention → Classifier
  Whisper-large (layer 24, PCA-32) ──┘

Workflow:
  1) Load cached HuBERT-large and Whisper-large layer features
  2) Build binary targets from SEP-28k / FluencyBank CSVs
  3) Train/test split (stratified, seed-fixed)
  4) Fit independent PCA-32 on each stream (train fold only, no leakage)
  5) Stack streams as 2-step sequence → BiLSTM + Attention
  6) Train with AdamW + CosineAnnealingLR, CrossEntropyLoss
  7) Evaluate: Accuracy, F1, Precision, Recall
  8) Save: run_report.json, stream_attention.csv, train_history.csv, figures

Run command (after both caches are built):
    python experiments/C5_whisper_hubert_bilstm.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --whisper-alias whisper-large \
        --whisper-layer 24 \
        --target Block \
        --fold fold0 \
        --ssl-pca-dim 32 \
        --proj-dim 64 \
        --lstm-hidden 128 \
        --lstm-layers 2 \
        --epochs 20 \
        --batch-size 256 \
        --lr 3e-4 \
        --dropout 0.3 \
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

LABEL_COLUMNS = {
    "Block":        "Block",
    "Prolongation": "Prolongation",
    "SoundRep":     "SoundRep",
    "WordRep":      "WordRep",
    "Interjection": "Interjection",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C5 Whisper+HuBERT cross-modal BiLSTM fusion")

    # Feature cache paths
    p.add_argument("--features-root",  type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",   type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",   type=int,  default=21,
                   help="HuBERT-large transformer layer index (1-24)")
    p.add_argument("--whisper-alias",  type=str,  default="whisper-large")
    p.add_argument("--whisper-layer",  type=int,  default=24,
                   help="Whisper-large encoder layer index (1-32)")
    p.add_argument("--fold",           type=str,  default="fold0")

    # Dataset
    p.add_argument("--clips-root",     type=Path, default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",     type=Path, default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path, default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--target",         choices=list(LABEL_COLUMNS.keys()), default="Block")

    # Preprocessing
    p.add_argument("--ssl-pca-dim",    type=int,  default=32,
                   help="PCA output dimension for each stream independently")
    p.add_argument("--test-size",      type=float,default=0.20)
    p.add_argument("--seed",           type=int,  default=42)

    # Model
    p.add_argument("--proj-dim",       type=int,  default=64,
                   help="Linear projection dim applied to each PCA stream before BiLSTM")
    p.add_argument("--lstm-hidden",    type=int,  default=128,
                   help="BiLSTM hidden size per direction (output = 2x this)")
    p.add_argument("--lstm-layers",    type=int,  default=2)
    p.add_argument("--dropout",        type=float,default=0.30)

    # Training
    p.add_argument("--epochs",       type=int,  default=20)
    p.add_argument("--batch-size",   type=int,  default=256)
    p.add_argument("--lr",           type=float,default=3e-4)
    p.add_argument("--weight-decay", type=float,default=1e-4)

    # Output
    p.add_argument("--out-dir",  type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",  type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/C5"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def norm_text(x: object) -> str:
    return str(x).strip()


def load_label_map(
    csv_path: Path, target_col: str
) -> Dict[Tuple[str, str, str], int]:
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


def load_cache(
    features_root: Path, alias: str, fold: str, layer: int
) -> np.ndarray:
    """Load a cached layer .npy file, with fold-fallback handling."""
    direct = features_root / alias / fold / f"layer_{layer}.npy"
    if direct.exists():
        return np.load(direct)
    # Fallback: look in any sub-folder under the alias root
    alias_root = features_root / alias
    if alias_root.is_dir():
        for child in sorted(alias_root.iterdir()):
            cand = child / f"layer_{layer}.npy"
            if cand.exists():
                print(f"  [WARN] Using fallback fold '{child.name}' for alias '{alias}'")
                return np.load(cand)
    raise FileNotFoundError(
        f"Cannot find layer_{layer}.npy for alias '{alias}' fold '{fold}' under {features_root}."
        f"\nRun build_ssl_cache_full.py first for this model."
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class C5CrossModalBiLSTM(nn.Module):
    """
    Two-stream cross-modal fusion model.

    Streams:
        stream 0 → HuBERT features (acoustic structure)
        stream 1 → Whisper features (linguistic context)

    Forward returns (logits [B,2], attn_weights [B,2])
    where attn_weights[:, 0] = HuBERT importance,
          attn_weights[:, 1] = Whisper importance.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Independent per-stream projections (allows each stream its own transform)
        self.proj_hubert = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.proj_whisper = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # BiLSTM over stream sequence [B, 2, proj_dim]
        self.bilstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Scaled dot-product attention over 2 stream positions
        self.attn_q = nn.Linear(lstm_out_dim, lstm_out_dim, bias=False)
        self.attn_k = nn.Linear(lstm_out_dim, lstm_out_dim, bias=False)
        self.scale   = lstm_out_dim ** -0.5

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        x_hubert: torch.Tensor,   # [B, in_dim]
        x_whisper: torch.Tensor,  # [B, in_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h = self.proj_hubert(x_hubert)   # [B, proj_dim]
        w = self.proj_whisper(x_whisper) # [B, proj_dim]

        seq = torch.stack([h, w], dim=1)  # [B, 2, proj_dim]
        lstm_out, _ = self.bilstm(seq)    # [B, 2, lstm_out_dim]

        # Scaled dot-product attention: each position attends to itself + other
        Q = self.attn_q(lstm_out)  # [B, 2, D]
        K = self.attn_k(lstm_out)  # [B, 2, D]
        scores = (Q * K).sum(dim=-1) * self.scale  # [B, 2]  (element-wise dot)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, 2]

        context = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, lstm_out_dim]
        logits  = self.classifier(context)  # [B, 2]
        return logits, attn_weights


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    opt: torch.optim.Optimizer | None,
    device: torch.device,
    training: bool,
) -> Tuple[float, List[int], List[int], np.ndarray]:
    model.train() if training else model.eval()
    total_loss = 0.0
    all_pred: List[int] = []
    all_true: List[int] = []
    all_attn: List[np.ndarray] = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xh, xw, yb in loader:
            xh, xw, yb = xh.to(device), xw.to(device), yb.to(device)
            logits, attn = model(xh, xw)
            loss = criterion(logits, yb)

            if training and opt is not None:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total_loss   += loss.item() * xh.size(0)
            all_pred.extend(logits.argmax(dim=1).cpu().tolist())
            all_true.extend(yb.cpu().tolist())
            all_attn.append(attn.detach().cpu().numpy())

    attn_arr = np.concatenate(all_attn, axis=0)  # [N, 2]
    return total_loss / len(loader.dataset), all_pred, all_true, attn_arr


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
    # 1. Load cached features
    # ------------------------------------------------------------------
    print(f"\nLoading HuBERT-large layer {args.hubert_layer} cache ...")
    feats_h = load_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    if feats_h.ndim != 2:
        feats_h = feats_h.reshape(feats_h.shape[0], -1)

    print(f"Loading Whisper-large layer {args.whisper_layer} cache ...")
    feats_w = load_cache(args.features_root, args.whisper_alias, args.fold, args.whisper_layer)
    if feats_w.ndim != 2:
        feats_w = feats_w.reshape(feats_w.shape[0], -1)

    n = min(feats_h.shape[0], feats_w.shape[0])
    feats_h = feats_h[:n]
    feats_w = feats_w[:n]
    print(f"Samples (aligned): {n} | HuBERT dim={feats_h.shape[1]} | Whisper dim={feats_w.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Labels
    # ------------------------------------------------------------------
    target_col = LABEL_COLUMNS[args.target]
    label_map: Dict[Tuple[str, str, str], int] = {}
    label_map.update(load_label_map(args.sep_labels, target_col))
    label_map.update(load_label_map(args.fluency_labels, target_col))

    clip_keys = sorted_clip_keys(args.clips_root)
    if len(clip_keys) < n:
        raise RuntimeError(
            f"Clip count ({len(clip_keys)}) < cached rows ({n}). "
            "Rebuild cache from clips_root."
        )
    clip_keys = clip_keys[:n]
    y = np.array([label_map.get(k, 0) for k in clip_keys], dtype=np.int64)

    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    print(f"Positives: {n_pos} | Negatives: {n_neg} | Pos rate: {n_pos/n:.2%}")

    # ------------------------------------------------------------------
    # 3. Train / test split
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # ------------------------------------------------------------------
    # 4. Per-stream PCA (fit on train only — no leakage)
    # ------------------------------------------------------------------
    pca_dim = min(args.ssl_pca_dim, feats_h.shape[1], feats_w.shape[1])

    # HuBERT stream
    sc_h = StandardScaler()
    xh_tr = sc_h.fit_transform(feats_h[train_idx])
    xh_te = sc_h.transform(feats_h[test_idx])
    pca_h = PCA(n_components=pca_dim, random_state=args.seed)
    xh_tr = pca_h.fit_transform(xh_tr).astype(np.float32)
    xh_te = pca_h.transform(xh_te).astype(np.float32)

    # Whisper stream
    sc_w = StandardScaler()
    xw_tr = sc_w.fit_transform(feats_w[train_idx])
    xw_te = sc_w.transform(feats_w[test_idx])
    pca_w = PCA(n_components=pca_dim, random_state=args.seed)
    xw_tr = pca_w.fit_transform(xw_tr).astype(np.float32)
    xw_te = pca_w.transform(xw_te).astype(np.float32)

    y_tr = y[train_idx]
    y_te = y[test_idx]

    print(f"PCA dim per stream: {pca_dim}")
    print(
        f"HuBERT explained variance (top-{pca_dim}): "
        f"{pca_h.explained_variance_ratio_.sum():.3f}"
    )
    print(
        f"Whisper explained variance (top-{pca_dim}): "
        f"{pca_w.explained_variance_ratio_.sum():.3f}"
    )

    # ------------------------------------------------------------------
    # 5. DataLoaders
    # ------------------------------------------------------------------
    def make_loader(xh, xw, yy, shuffle):
        ds = TensorDataset(
            torch.from_numpy(xh),
            torch.from_numpy(xw),
            torch.from_numpy(yy).long(),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_dl = make_loader(xh_tr, xw_tr, y_tr, shuffle=True)
    test_dl  = make_loader(xh_te, xw_te, y_te, shuffle=False)

    # ------------------------------------------------------------------
    # 6. Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    model = C5CrossModalBiLSTM(
        in_dim=pca_dim,
        proj_dim=args.proj_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nC5 model parameters: {n_params:,}")

    # Class weights for imbalanced data
    pos_weight_val = n_neg / max(n_pos, 1)
    weight_tensor  = torch.tensor([1.0, pos_weight_val], device=device)
    criterion      = nn.CrossEntropyLoss(weight=weight_tensor)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs ...")
    history: List[Dict[str, float]] = []
    best_f1    = -1.0
    best_ckpt  = args.ckpt_dir / "c5_best.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _, _ = run_epoch(model, train_dl, criterion, opt, device, training=True)
        scheduler.step()

        # Quick validation F1 each epoch
        val_loss, val_pred, val_true, _ = run_epoch(
            model, test_dl, criterion, None, device, training=False
        )
        val_f1 = f1_score(val_true, val_pred, zero_division=0)

        row = {
            "epoch":    epoch,
            "train_loss": round(tr_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_f1":     round(val_f1, 6),
            "lr":         scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d}  train_loss={tr_loss:.5f}  "
            f"val_loss={val_loss:.5f}  val_f1={val_f1:.5f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)

    # ------------------------------------------------------------------
    # 8. Final evaluation (best checkpoint)
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, test_pred, test_true, test_attn = run_epoch(
        model, test_dl, criterion, None, device, training=False
    )

    acc  = accuracy_score(test_true, test_pred)
    f1   = f1_score(test_true, test_pred, zero_division=0)
    prec = precision_score(test_true, test_pred, zero_division=0)
    rec  = recall_score(test_true, test_pred, zero_division=0)
    try:
        auc = roc_auc_score(test_true, test_pred)
    except Exception:
        auc = float("nan")

    print(f"\n--- Test Results (best ckpt at epoch with val_f1={best_f1:.5f}) ---")
    print(f"  Accuracy : {acc:.5f}")
    print(f"  F1       : {f1:.5f}")
    print(f"  Precision: {prec:.5f}")
    print(f"  Recall   : {rec:.5f}")
    print(f"  AUROC    : {auc:.5f}")

    # Stream attention summary
    attn_mean = test_attn.mean(axis=0)  # [2]
    stream_names = ["hubert", "whisper"]
    print(f"\n  Stream attention (mean over test set):")
    print(f"    HuBERT  : {attn_mean[0]:.4f}")
    print(f"    Whisper : {attn_mean[1]:.4f}")
    dominant = stream_names[int(np.argmax(attn_mean))]
    print(f"  Dominant stream: {dominant}")

    # ------------------------------------------------------------------
    # 9. Save outputs
    # ------------------------------------------------------------------

    # Attention CSV
    attn_csv = args.out_dir / "c5_stream_attention.csv"
    with attn_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stream", "mean_attn_weight"])
        w.writeheader()
        for name, val in zip(stream_names, attn_mean):
            w.writerow({"stream": name, "mean_attn_weight": float(val)})

    # Train history CSV
    hist_csv = args.out_dir / "c5_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_f1", "lr"])
        w.writeheader()
        w.writerows(history)

    # Run report JSON
    run_report = {
        "experiment":     "C5",
        "title":          "Whisper-large + HuBERT-large cross-modal BiLSTM+Attention fusion",
        "target":         args.target,
        "device":         str(device),
        "hubert_alias":   args.hubert_alias,
        "hubert_layer":   args.hubert_layer,
        "whisper_alias":  args.whisper_alias,
        "whisper_layer":  args.whisper_layer,
        "fold":           args.fold,
        "num_samples":    int(n),
        "num_pos":        int(n_pos),
        "num_neg":        int(n_neg),
        "pca_dim":        int(pca_dim),
        "proj_dim":       args.proj_dim,
        "lstm_hidden":    args.lstm_hidden,
        "lstm_layers":    args.lstm_layers,
        "dropout":        args.dropout,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "accuracy":       float(acc),
        "f1":             float(f1),
        "precision":      float(prec),
        "recall":         float(rec),
        "auroc":          float(auc),
        "best_val_f1":    float(best_f1),
        "stream_attention": {name: float(val) for name, val in zip(stream_names, attn_mean)},
        "dominant_stream":  dominant,
        "best_checkpoint":  str(best_ckpt),
        "attn_csv":         str(attn_csv),
        "history_csv":      str(hist_csv),
    }
    report_json = args.out_dir / "c5_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 10. Figures
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- Training curves ---
        epochs = [r["epoch"]     for r in history]
        tr_l   = [r["train_loss"]for r in history]
        va_l   = [r["val_loss"]  for r in history]
        va_f1  = [r["val_f1"]    for r in history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs, tr_l, label="train loss")
        axes[0].plot(epochs, va_l, label="val loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"C5 Training Curves ({args.target})")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        val_f1s   = [r["val_f1"] for r in history]
        best_epoch = history[int(np.argmax(val_f1s))]["epoch"]

        axes[1].plot(epochs, val_f1s, color="green", marker="o", markersize=3)
        axes[1].axvline(
            x=best_epoch,
            linestyle="--", color="red", alpha=0.6, label=f"best F1={best_f1:.4f}"
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val F1")
        axes[1].set_title(f"C5 Validation F1 ({args.target})")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        curve_fig = args.fig_dir / "c5_train_curves.png"
        fig.savefig(curve_fig, dpi=160)
        plt.close(fig)
        print(f"  Saved: {curve_fig}")

        # --- Stream attention bar chart ---
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = ["#4C72B0", "#DD8452"]
        bars = ax2.bar(["HuBERT-large\n(acoustic)", "Whisper-large\n(linguistic)"],
                       attn_mean, color=colors, width=0.5)
        for bar, val in zip(bars, attn_mean):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=11,
            )
        ax2.set_ylim(0, 1.0)
        ax2.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Equal split")
        ax2.set_ylabel("Mean Attention Weight", fontsize=11)
        ax2.set_title(f"C5 Stream Attention ({args.target}, layer H={args.hubert_layer}/W={args.whisper_layer})",
                      fontsize=11)
        ax2.legend()
        fig2.tight_layout()
        attn_fig = args.fig_dir / "c5_stream_attention.png"
        fig2.savefig(attn_fig, dpi=160)
        plt.close(fig2)
        print(f"  Saved: {attn_fig}")

    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  C5 complete.")
    print(f"   Report : {report_json}")
    print(f"   Attn   : {attn_csv}")
    print(f"   History: {hist_csv}")
    print(f"   Ckpt   : {best_ckpt}")


if __name__ == "__main__":
    main()
