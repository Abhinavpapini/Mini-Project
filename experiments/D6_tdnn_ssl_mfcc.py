"""D6: TDNN (Time Delay Neural Network) with SSL + MFCC fusion.

TDNN processes features with explicit temporal context windows at each layer,
using dilated receptive fields. Originally designed for speaker verification
(x-vector), here applied to HuBERT + MFCC multi-modal fusion for stutter detection.

ARCHITECTURE:
  Two input streams:
    Stream H: HuBERT-large L21 (1024-dim) → FC proj → 256-dim
    Stream M: MFCC-78 (39 MFCCs + deltas) → FC proj → 64-dim

  Fusion: Concatenate [H_proj; M_proj] → 320-dim

  TDNN stack (5 layers with dilation):
    Layer 1: context [-2,2]        dilation=1,  320→256
    Layer 2: context [-2,0,2]      dilation=1,  256→256
    Layer 3: context [-3,0,3]      dilation=2,  256→256
    Layer 4: context {0}           dilation=1,  256→256  (bottleneck)
    Layer 5: context {0}           dilation=1,  256→256  (pre-output)

  Since our input is a fixed feature vector (not a frame sequence),
  TDNN "temporal context" is simulated over feature sub-groups
  (reshaped to pseudo-timesteps), consistent with D3 approach.

  Stats pooling: mean + std pooling → 512-dim
  Classifier MLP: 512→256→5 (MLSM)

Run command:
    python experiments/D6_tdnn_ssl_mfcc.py \
        --hubert-alias hubert-large --hubert-layer 21 \
        --mfcc-cache artifacts/features/mfcc/fold0/mfcc_stats.npy \
        --epochs 50 --batch-size 256 --lr 3e-4
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
    p = argparse.ArgumentParser(description="D6: TDNN SSL+MFCC fusion, multi-label")
    p.add_argument("--features-root",  type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",   type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",   type=int,  default=21)
    p.add_argument("--mfcc-cache",     type=Path,
                   default=Path("artifacts/features/mfcc/fold0/mfcc_stats.npy"))
    p.add_argument("--fold",           type=str,  default="fold0")
    p.add_argument("--clips-root",     type=Path,
                   default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",     type=Path,
                   default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path,
                   default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    # Architecture
    p.add_argument("--ssl-proj-dim",   type=int,  default=256)
    p.add_argument("--mfcc-proj-dim",  type=int,  default=64)
    p.add_argument("--tdnn-channels",  type=str,  default="256,256,256,256,256")
    p.add_argument("--seq-len",        type=int,  default=32,
                   help="Reshape fused features to seq_len pseudo-steps")
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
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/D6"))
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
# TDNN Model
# ---------------------------------------------------------------------------

class TDNNLayer(nn.Module):
    """Single TDNN layer: Conv1d with dilation over pseudo-sequence."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                      dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class StatsPooling(nn.Module):
    """Compute mean and std over temporal dimension → concat → 2×C."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        mean = x.mean(dim=2)
        std  = x.std(dim=2, unbiased=False).clamp(min=1e-5)
        return torch.cat([mean, std], dim=1)   # [B, 2C]


class TDNNSSLMFCCModel(nn.Module):
    def __init__(self, ssl_dim: int, mfcc_dim: int,
                 ssl_proj: int, mfcc_proj: int,
                 tdnn_channels: List[int], seq_len: int,
                 n_classes: int, dropout: float) -> None:
        super().__init__()
        fused_dim = ssl_proj + mfcc_proj

        # Input projections
        self.ssl_proj  = nn.Sequential(
            nn.Linear(ssl_dim,  ssl_proj),  nn.LayerNorm(ssl_proj), nn.GELU(),
        )
        self.mfcc_proj = nn.Sequential(
            nn.Linear(mfcc_dim, mfcc_proj), nn.LayerNorm(mfcc_proj), nn.GELU(),
        )
        self.seq_len   = seq_len
        self.feat_step = fused_dim // seq_len  # features per pseudo-step

        # Input conv to first TDNN channels
        self.input_conv = nn.Sequential(
            nn.Conv1d(self.feat_step, tdnn_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(tdnn_channels[0]), nn.ReLU(inplace=True),
        )

        # TDNN stack with increasing dilation
        dilations = [1, 1, 2, 2, 1]
        kernels   = [3, 3, 3, 1, 1]
        layers: List[nn.Module] = []
        for i, (out_ch, k, d) in enumerate(zip(tdnn_channels[1:], kernels, dilations)):
            layers.append(TDNNLayer(tdnn_channels[i], out_ch, k, d, dropout * 0.5))
        self.tdnn = nn.Sequential(*layers)

        # Stats pooling + classifier
        pool_dim = tdnn_channels[-1] * 2  # mean + std
        self.stats_pool = StatsPooling()
        self.head = nn.Sequential(
            nn.Linear(pool_dim, pool_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(pool_dim // 2, n_classes),
        )

    def forward(self, ssl: torch.Tensor, mfcc: torch.Tensor) -> torch.Tensor:
        B = ssl.size(0)
        h = self.ssl_proj(ssl)         # [B, ssl_proj]
        m = self.mfcc_proj(mfcc)       # [B, mfcc_proj]
        x = torch.cat([h, m], dim=1)   # [B, fused_dim]
        # Reshape to pseudo-sequence: [B, seq_len, feat_step] → [B, feat_step, seq_len]
        x = x.view(B, self.seq_len, self.feat_step).transpose(1, 2)
        x = self.input_conv(x)         # [B, ch0, seq_len]
        x = self.tdnn(x)               # [B, chN, seq_len]
        x = self.stats_pool(x)         # [B, 2*chN]
        return self.head(x)            # [B, n_classes]


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
    # 1. SSL features
    # ------------------------------------------------------------------
    print(f"\nLoading {args.hubert_alias} layer {args.hubert_layer} ...")
    ssl_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    if ssl_feats.ndim != 2:
        ssl_feats = ssl_feats.reshape(ssl_feats.shape[0], -1)
    n, ssl_dim = ssl_feats.shape
    print(f"  SSL features: {ssl_feats.shape}")

    # ------------------------------------------------------------------
    # 2. MFCC features
    # ------------------------------------------------------------------
    print(f"\nLoading MFCC cache: {args.mfcc_cache} ...")
    mfcc_feats = np.load(args.mfcc_cache)
    if mfcc_feats.ndim != 2:
        mfcc_feats = mfcc_feats.reshape(mfcc_feats.shape[0], -1)
    # Align to SSL count
    mfcc_feats = mfcc_feats[:n]
    mfcc_dim   = mfcc_feats.shape[1]
    print(f"  MFCC features: {mfcc_feats.shape}")

    # Compute fused dim and pad if needed for seq_len divisibility
    fused_dim = args.ssl_proj_dim + args.mfcc_proj_dim
    assert fused_dim % args.seq_len == 0, (
        f"ssl_proj_dim+mfcc_proj_dim ({fused_dim}) must be divisible by seq_len ({args.seq_len})"
    )
    feat_step = fused_dim // args.seq_len
    print(f"  Fused proj dim: {fused_dim} → reshape [{args.seq_len} steps × {feat_step}]")

    # ------------------------------------------------------------------
    # 3. Labels
    # ------------------------------------------------------------------
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
    # 4. Split + Scale
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y[:, 0].astype(int)
    )
    sc_s = StandardScaler(); sc_m = StandardScaler()
    s_tr = sc_s.fit_transform(ssl_feats[train_idx]).astype(np.float32)
    s_te = sc_s.transform(ssl_feats[test_idx]).astype(np.float32)
    m_tr = sc_m.fit_transform(mfcc_feats[train_idx]).astype(np.float32)
    m_te = sc_m.transform(mfcc_feats[test_idx]).astype(np.float32)
    y_tr, y_te = y[train_idx], y[test_idx]

    # ------------------------------------------------------------------
    # 5. DataLoaders
    # ------------------------------------------------------------------
    sw = compute_sample_weights(y_tr)
    sampler  = WeightedRandomSampler(torch.from_numpy(sw), len(sw), replacement=True)
    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(s_tr), torch.from_numpy(m_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True,
    )
    test_dl  = DataLoader(
        TensorDataset(torch.from_numpy(s_te), torch.from_numpy(m_te), torch.from_numpy(y_te)),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    tdnn_ch = [int(c) for c in args.tdnn_channels.split(",")]
    model   = TDNNSSLMFCCModel(
        ssl_dim=ssl_dim, mfcc_dim=mfcc_dim,
        ssl_proj=args.ssl_proj_dim, mfcc_proj=args.mfcc_proj_dim,
        tdnn_channels=tdnn_ch, seq_len=args.seq_len,
        n_classes=len(STUTTER_TYPES), dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nD6 model parameters: {n_params:,}")
    print(f"  SSL: {ssl_dim}→{args.ssl_proj_dim} | MFCC: {mfcc_dim}→{args.mfcc_proj_dim}")
    print(f"  Fused: {fused_dim}-dim → {args.seq_len}×{feat_step} pseudo-seq")
    print(f"  TDNN: ch={tdnn_ch}, dilations=[1,1,2,2,1]")
    print(f"  StatsPool({tdnn_ch[-1]}×2={tdnn_ch[-1]*2}) → MLP→5")

    criterion = nn.MultiLabelSoftMarginLoss()
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                             eta_min=args.lr * 0.05)

    # ------------------------------------------------------------------
    # 7. Training
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs ...")
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "d6_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for sb, mb, yb in train_dl:
            sb, mb, yb = sb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(sb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * sb.size(0)
        tr_loss /= len(y_tr)
        scheduler.step()

        model.eval()
        vl, vlog, vlab = 0.0, [], []
        with torch.no_grad():
            for sb, mb, yb in test_dl:
                sb, mb, yb = sb.to(device), mb.to(device), yb.to(device)
                lg = model(sb, mb)
                vl += criterion(lg, yb).item() * sb.size(0)
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
    # 8. Final evaluation
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    tlog, tlab = [], []
    with torch.no_grad():
        for sb, mb, yb in test_dl:
            tlog.append(model(sb.to(device), mb.to(device)).cpu().numpy())
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
    with (args.out_dir / "d6_perclass_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    with (args.out_dir / "d6_train_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "D6",
        "title": "TDNN SSL+MFCC Fusion (HuBERT-large + MFCC-78, dilated temporal)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "ssl_dim": int(ssl_dim),
        "mfcc_dim": int(mfcc_dim), "fused_dim": int(fused_dim),
        "tdnn_channels": tdnn_ch, "seq_len": args.seq_len,
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
    (args.out_dir / "d6_run_report.json").write_text(json.dumps(run_report, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val")
        axes[0].set_title(f"D6 Loss (TDNN, SSL+MFCC)")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="darkorange", marker="o", ms=4)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("D6 Validation Macro-F1 (TDNN SSL+MFCC)")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "d6_train_curves.png", dpi=160); plt.close(fig)

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
        ax2.set_title("D6 Per-class F1 (TDNN SSL+MFCC)")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "d6_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"\n  Saved: {args.fig_dir / 'd6_train_curves.png'}")
        print(f"  Saved: {args.fig_dir / 'd6_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  D6 complete.")
    print(f"   Report : {args.out_dir / 'd6_run_report.json'}")
    print(f"   Ckpt   : {best_ckpt}")


if __name__ == "__main__":
    main()
