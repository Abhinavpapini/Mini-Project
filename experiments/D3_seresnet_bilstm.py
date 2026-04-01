"""D3: SE-ResNet1D + BiLSTM — Squeeze-Excitation deep temporal model.

ARCHITECTURE:
  Since frame-level sequences are not available (mean-pooled cache),
  we treat the 1024-dim HuBERT vector as a pseudo-temporal sequence:
  reshape [B, 1024] → [B, 32, 32] (32 "time steps" × 32-dim features)

  This allows the SE-ResNet to learn channel-wise attention over
  feature subspaces, and the BiLSTM to capture dependencies between
  adjacent sub-groups of the HuBERT representation.

  SE-ResNet1D:
    Input [B, 32, 32] (channels=32, length=32)
    → 3 × SEResBlock(32→64→128, kernel=3)
    → Each block: Conv1d → BN → GELU → Conv1d → BN → SE(16:1 ratio) → residual

  BiLSTM:
    [B, 128, 32] → transpose → [B, 32, 128]
    → BiLSTM(hidden=128, 2 layers) → [B, 32, 256]
    → Attention pooling → [B, 256]

  Classifier:
    → FC(256→128) → GELU → Dropout → FC(128→5) → MLSM

Run command:
    python experiments/D3_seresnet_bilstm.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --seq-len 32 \
        --lstm-hidden 128 \
        --lstm-layers 2 \
        --epochs 50 \
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
    p = argparse.ArgumentParser(description="D3: SE-ResNet1D + BiLSTM, multi-label")
    p.add_argument("--features-root",  type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",   type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",   type=int,  default=21)
    p.add_argument("--fold",           type=str,  default="fold0")
    p.add_argument("--clips-root",     type=Path,
                   default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",     type=Path,
                   default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path,
                   default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    # Architecture
    p.add_argument("--seq-len",        type=int,  default=32,
                   help="Reshape 1024=seq_len×feat_size. Must divide 1024.")
    p.add_argument("--se-ratio",       type=int,  default=4,
                   help="SE reduction ratio (channel // se_ratio)")
    p.add_argument("--se-channels",    type=str,  default="64,128,256")
    p.add_argument("--lstm-hidden",    type=int,  default=128)
    p.add_argument("--lstm-layers",    type=int,  default=2)
    p.add_argument("--dropout",        type=float,default=0.3)
    # Training
    p.add_argument("--test-size",      type=float,default=0.20)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--epochs",         type=int,  default=50)
    p.add_argument("--batch-size",     type=int,  default=256)
    p.add_argument("--lr",             type=float,default=3e-4)
    p.add_argument("--weight-decay",   type=float,default=1e-4)
    p.add_argument("--threshold",      type=float,default=0.5)
    p.add_argument("--out-dir",  type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",  type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/D3"))
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
        pm = y[i] > 0
        weights[i] = inv_freq[pm].max() if pm.any() else inv_freq.min()
    return weights / weights.min()


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation on 1D channel axis."""
    def __init__(self, channels: int, se_ratio: int = 4) -> None:
        super().__init__()
        reduced = max(1, channels // se_ratio)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced), nn.ReLU(inplace=True),
            nn.Linear(reduced, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        s = self.excite(self.squeeze(x))          # [B, C]
        return x * s.unsqueeze(-1)                # scale channels


class SEResBlock1D(nn.Module):
    """SE-ResNet block: Conv→BN→GELU→Conv→BN→SE + residual."""
    def __init__(self, in_ch: int, out_ch: int,
                 se_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.se      = SEBlock1D(out_ch, se_ratio)
        self.act     = nn.GELU()
        # Projection if channels change
        self.proj = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.conv(x)) + self.proj(x))


class AttentionPool1D(nn.Module):
    """Soft attention pooling over sequence length."""
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        w = torch.softmax(self.attn(x), dim=1)   # [B, L, 1]
        return (w * x).sum(dim=1)                 # [B, D]


class SEResNetBiLSTM(nn.Module):
    def __init__(self, in_dim: int, seq_len: int, se_channels: List[int],
                 se_ratio: int, lstm_hidden: int, lstm_layers: int,
                 n_classes: int, dropout: float) -> None:
        super().__init__()
        # Reshape 1024 → [seq_len, feat_size]
        feat_size = in_dim // seq_len
        assert in_dim == seq_len * feat_size, f"{in_dim} must be divisible by {seq_len}"

        self.seq_len  = seq_len
        self.feat_size = feat_size

        # Input projection: feat_size → se_channels[0] (channel dim)
        self.input_proj = nn.Sequential(
            nn.Conv1d(feat_size, se_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(se_channels[0]), nn.GELU(),
        )

        # SE-ResNet stack
        se_blocks: List[nn.Module] = []
        in_ch = se_channels[0]
        for out_ch in se_channels[1:]:
            se_blocks.append(SEResBlock1D(in_ch, out_ch, se_ratio, dropout * 0.5))
            in_ch = out_ch
        self.se_blocks = nn.Sequential(*se_blocks)

        # BiLSTM over sequence
        self.bilstm = nn.LSTM(
            input_size=in_ch, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Attention pooling
        self.attn_pool = AttentionPool1D(lstm_out_dim)

        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1024] → [B, seq_len, feat_size] → [B, feat_size, seq_len]
        B = x.size(0)
        x = x.view(B, self.seq_len, self.feat_size).transpose(1, 2)  # [B, F, L]
        x = self.input_proj(x)   # [B, se_ch0, L]
        x = self.se_blocks(x)    # [B, se_chN, L]
        # BiLSTM: [B, L, C]
        x = x.transpose(1, 2)    # [B, L, C]
        x, _ = self.bilstm(x)    # [B, L, 2*lstm_hidden]
        x = self.attn_pool(x)    # [B, 2*lstm_hidden]
        return self.head(x)      # [B, n_classes]


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
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Features + labels
    # ------------------------------------------------------------------
    print(f"\nLoading {args.hubert_alias} layer {args.hubert_layer} ...")
    ssl_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    if ssl_feats.ndim != 2:
        ssl_feats = ssl_feats.reshape(ssl_feats.shape[0], -1)
    n, in_dim = ssl_feats.shape
    feat_size = in_dim // args.seq_len
    print(f"  SSL features: {ssl_feats.shape}")
    print(f"  Pseudo-sequence: {args.seq_len} steps × {feat_size}-dim features")

    label_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    label_map.update(load_multilabel_map(args.sep_labels))
    label_map.update(load_multilabel_map(args.fluency_labels))
    clip_keys = sorted_clip_keys(args.clips_root)[:n]
    y = np.array(
        [label_map.get(k, np.zeros(len(STUTTER_TYPES), np.float32)) for k in clip_keys],
        dtype=np.float32,
    )
    print(f"\nDataset: {n} samples | Label distribution:")
    for i, t in enumerate(STUTTER_TYPES):
        pos = int(y[:, i].sum())
        print(f"  {t:15s}: {pos:5d} pos  ({pos/n:.2%})")

    # ------------------------------------------------------------------
    # 2. Split + StandardScaler
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y[:, 0].astype(int)
    )
    sc = StandardScaler()
    x_tr = sc.fit_transform(ssl_feats[train_idx]).astype(np.float32)
    x_te = sc.transform(ssl_feats[test_idx]).astype(np.float32)
    y_tr, y_te = y[train_idx], y[test_idx]

    # ------------------------------------------------------------------
    # 3. DataLoaders
    # ------------------------------------------------------------------
    sw = compute_sample_weights(y_tr)
    sampler  = WeightedRandomSampler(torch.from_numpy(sw), len(sw), replacement=True)
    train_dl = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                          batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    test_dl  = DataLoader(TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te)),
                          batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    se_channels = [int(c) for c in args.se_channels.split(",")]
    model = SEResNetBiLSTM(
        in_dim=in_dim, seq_len=args.seq_len, se_channels=se_channels,
        se_ratio=args.se_ratio, lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers, n_classes=len(STUTTER_TYPES),
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    lstm_out = args.lstm_hidden * 2
    print(f"\nD3 model parameters: {n_params:,}")
    print(f"  Input reshape: {in_dim} → {args.seq_len}×{feat_size}")
    print(f"  SE-ResNet: ch={se_channels}, se_ratio={args.se_ratio}")
    print(f"  BiLSTM: hidden={args.lstm_hidden}×2={lstm_out}, layers={args.lstm_layers}")
    print(f"  AttnPool → MLP({lstm_out}→{lstm_out//2}→5)")

    criterion = nn.MultiLabelSoftMarginLoss()
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                             eta_min=args.lr * 0.05)

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs ...")
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "d3_best.pt"

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
        tr_loss /= len(y_tr)
        scheduler.step()

        model.eval()
        vl, vlog, vlab = 0.0, [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                lg = model(xb)
                vl += criterion(lg, yb).item() * xb.size(0)
                vlog.append(lg.cpu().numpy()); vlab.append(yb.cpu().numpy())
        vl /= len(y_te)
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
    # 6. Final evaluation
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
    # 7. Save
    # ------------------------------------------------------------------
    with (args.out_dir / "d3_perclass_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    with (args.out_dir / "d3_train_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "D3",
        "title": "SE-ResNet1D + BiLSTM + Attn-Pool (HuBERT-large pseudo-sequence)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "in_dim": int(in_dim),
        "seq_len": args.seq_len, "feat_size": int(feat_size),
        "se_channels": se_channels, "se_ratio": args.se_ratio,
        "lstm_hidden": args.lstm_hidden, "lstm_layers": args.lstm_layers,
        "n_params": int(n_params), "loss": "MultiLabelSoftMarginLoss",
        "epochs": args.epochs, "best_epoch": int(best_ep),
        "macro_f1": test_m["macro_f1"], "micro_f1": test_m["micro_f1"],
        "macro_precision": test_m["macro_pre"], "macro_recall": test_m["macro_rec"],
        "macro_auprc": test_m["macro_auprc"],
        "per_class": {t: {"f1": test_m[f"f1_{t}"], "precision": test_m[f"pre_{t}"],
                          "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]}
                      for t in STUTTER_TYPES},
        "best_checkpoint": str(best_ckpt),
    }
    (args.out_dir / "d3_run_report.json").write_text(json.dumps(run_report, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val")
        axes[0].set_title(f"D3 Loss (SE-ResNet1D + BiLSTM, seq={args.seq_len})")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="teal", marker="o", ms=4)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("D3 Validation Macro-F1 (SE-ResNet + BiLSTM)")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "d3_train_curves.png", dpi=160); plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        pf1  = [test_m[f"f1_{t}"] for t in STUTTER_TYPES]
        bars = ax2.bar(STUTTER_TYPES, pf1,
                       color=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"], width=0.55)
        ax2.axhline(test_m["macro_f1"], ls="--", color="black", alpha=0.5,
                    label=f"Macro-F1={test_m['macro_f1']:.4f}")
        for bar, v in zip(bars, pf1):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_title(f"D3 Per-class F1 (SE-ResNet1D + BiLSTM)")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "d3_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"\n  Saved: {args.fig_dir / 'd3_train_curves.png'}")
        print(f"  Saved: {args.fig_dir / 'd3_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  D3 complete.")
    print(f"   Report : {args.out_dir / 'd3_run_report.json'}")
    print(f"   Ckpt   : {best_ckpt}")


if __name__ == "__main__":
    main()
