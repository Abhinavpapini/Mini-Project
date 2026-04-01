"""D7: Atrous-CNN (Dilated Convolutional Network) — multi-scale feature extraction.

Atrous (dilated) convolutions capture multi-scale receptive fields WITHOUT
pooling, preserving feature resolution while growing context windows.
Each dilation doubles the effective receptive field:
  dilation=[1,2,4,8,16] → receptive fields=[3,5,9,17,33] on a 32-step sequence

This is inspired by WaveNet/DeepSpeech architectures where dilated stacking
efficiently models long-range dependencies with O(log N) depth.

ARCHITECTURE:
  Input: HuBERT-large L21 (1024-dim) + Whisper-large L28 (1280-dim)
  → concat [2304-dim] → Project to d_model (256)
  → Reshape → [B, 32, 8] pseudo-seq → [B, 8, 32]

  Atrous-CNN stack (5 layers, exponential dilation):
    Each layer: Conv1d(dilation=2^i) → BN → GELU → Residual add
    Dilations: [1, 2, 4, 8, 16] → effective receptive fields [3,5,9,17,33]

  Global pooling: Concat [max-pool; mean-pool] → 2×C
  Classifier: FC(2C → C) → GELU → Dropout → FC(C → 5)

Using BOTH SSL streams (HuBERT + Whisper) as input — building on C4's
finding that dual-SSL is very powerful.

Run command:
    python experiments/D7_atrous_cnn.py \
        --hubert-alias hubert-large --hubert-layer 21 \
        --whisper-alias whisper-large --whisper-layer 28 \
        --d-model 256 \
        --n-dilation-layers 5 \
        --epochs 40 --batch-size 256 --lr 3e-4
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
    p = argparse.ArgumentParser(description="D7: Atrous-CNN dual-SSL, multi-label")
    p.add_argument("--features-root",   type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",    type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",    type=int,  default=21)
    p.add_argument("--whisper-alias",   type=str,  default="whisper-large")
    p.add_argument("--whisper-layer",   type=int,  default=28)
    p.add_argument("--fold",            type=str,  default="fold0")
    p.add_argument("--clips-root",      type=Path,
                   default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",      type=Path,
                   default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels",  type=Path,
                   default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    # Architecture
    p.add_argument("--d-model",         type=int,  default=256,
                   help="Internal channel dim (fused input projected to d_model)")
    p.add_argument("--n-dilation-layers",type=int, default=5,
                   help="Number of atrous layers (dilations = 2^i for i in range(N))")
    p.add_argument("--kernel-size",     type=int,  default=3)
    p.add_argument("--dropout",         type=float,default=0.3)
    # Training
    p.add_argument("--test-size",       type=float,default=0.20)
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--epochs",          type=int,  default=40)
    p.add_argument("--batch-size",      type=int,  default=256)
    p.add_argument("--lr",              type=float,default=3e-4)
    p.add_argument("--weight-decay",    type=float,default=1e-4)
    p.add_argument("--threshold",       type=float,default=0.5)
    p.add_argument("--out-dir",   type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",   type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir",  type=Path, default=Path("artifacts/checkpoints/D7"))
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
# Atrous-CNN Model
# ---------------------------------------------------------------------------

class AtrousResBlock(nn.Module):
    """Dilated Conv1d residual block."""
    def __init__(self, channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size,
                      dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv(x))


class AtrousCNNModel(nn.Module):
    def __init__(self, hubert_dim: int, whisper_dim: int, d_model: int,
                 n_dilation_layers: int, kernel_size: int,
                 n_classes: int, dropout: float) -> None:
        super().__init__()
        fused_dim = hubert_dim + whisper_dim
        # Find seq_len such that d_model // seq_len is integer ≥ 1
        # We project fused input to d_model first, then reshape
        self.input_proj = nn.Sequential(
            nn.Linear(fused_dim, d_model * 4), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(d_model * 4, d_model),   nn.LayerNorm(d_model),
        )
        # Treat d_model features as seq of length 32, each step has d_model//32 channels
        # → [B, d_model] → [B, 32, d_model//32] → [B, d_model//32, 32]
        seq_len = 32
        feat_step = d_model // seq_len   # 256//32 = 8
        self.seq_len   = seq_len
        self.feat_step = feat_step

        # Project feat_step → d_model channels for conv
        self.conv_in = nn.Sequential(
            nn.Conv1d(feat_step, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )

        # Atrous stack: dilations 1, 2, 4, 8, 16, ...
        self.atrous = nn.Sequential(*[
            AtrousResBlock(d_model, kernel_size, dilation=2**i, dropout=dropout * 0.5)
            for i in range(n_dilation_layers)
        ])

        # Dual pooling: max + mean → 2 × d_model
        pool_dim = d_model * 2
        self.head = nn.Sequential(
            nn.Linear(pool_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B = h.size(0)
        x = torch.cat([h, w], dim=-1)           # [B, hubert+whisper]
        x = self.input_proj(x)                  # [B, d_model]
        # Reshape: [B, d_model] → [B, seq_len, feat_step] → [B, feat_step, seq_len]
        x = x.view(B, self.seq_len, self.feat_step).transpose(1, 2)
        x = self.conv_in(x)                     # [B, d_model, seq_len]
        x = self.atrous(x)                      # [B, d_model, seq_len]
        # Dual pooling
        x_max  = x.max(dim=2).values            # [B, d_model]
        x_mean = x.mean(dim=2)                  # [B, d_model]
        x = torch.cat([x_max, x_mean], dim=1)  # [B, 2*d_model]
        return self.head(x)                     # [B, n_classes]


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
    # 1. Load both SSL caches
    # ------------------------------------------------------------------
    print(f"\nLoading {args.hubert_alias} layer {args.hubert_layer} ...")
    h_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    if h_feats.ndim != 2:
        h_feats = h_feats.reshape(h_feats.shape[0], -1)
    print(f"  HuBERT features: {h_feats.shape}")

    print(f"\nLoading {args.whisper_alias} layer {args.whisper_layer} ...")
    w_feats = load_ssl_cache(args.features_root, args.whisper_alias, args.fold, args.whisper_layer)
    if w_feats.ndim != 2:
        w_feats = w_feats.reshape(w_feats.shape[0], -1)
    print(f"  Whisper features: {w_feats.shape}")

    assert h_feats.shape[0] == w_feats.shape[0]
    n = h_feats.shape[0]
    hubert_dim = h_feats.shape[1]; whisper_dim = w_feats.shape[1]
    fused_dim  = hubert_dim + whisper_dim
    feat_step  = args.d_model // 32
    print(f"  Fused: {fused_dim}-dim → proj {args.d_model} → reshape [32×{feat_step}]")
    dils = [2**i for i in range(args.n_dilation_layers)]
    rfs  = [args.kernel_size + (args.kernel_size-1)*(d-1) for d in dils]
    print(f"  Atrous dilations: {dils}")
    print(f"  Effective receptive fields: {rfs}")

    # ------------------------------------------------------------------
    # 2. Labels
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
    # 3. Split + Scale
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y[:, 0].astype(int)
    )
    sc_h = StandardScaler(); sc_w = StandardScaler()
    h_tr = sc_h.fit_transform(h_feats[train_idx]).astype(np.float32)
    h_te = sc_h.transform(h_feats[test_idx]).astype(np.float32)
    w_tr = sc_w.fit_transform(w_feats[train_idx]).astype(np.float32)
    w_te = sc_w.transform(w_feats[test_idx]).astype(np.float32)
    y_tr, y_te = y[train_idx], y[test_idx]

    # ------------------------------------------------------------------
    # 4. DataLoaders
    # ------------------------------------------------------------------
    sw = compute_sample_weights(y_tr)
    sampler  = WeightedRandomSampler(torch.from_numpy(sw), len(sw), replacement=True)
    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(h_tr), torch.from_numpy(w_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True,
    )
    test_dl  = DataLoader(
        TensorDataset(torch.from_numpy(h_te), torch.from_numpy(w_te), torch.from_numpy(y_te)),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 5. Model
    # ------------------------------------------------------------------
    model = AtrousCNNModel(
        hubert_dim=hubert_dim, whisper_dim=whisper_dim,
        d_model=args.d_model, n_dilation_layers=args.n_dilation_layers,
        kernel_size=args.kernel_size, n_classes=len(STUTTER_TYPES), dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nD7 model parameters: {n_params:,}")
    print(f"  Input: HuBERT({hubert_dim})+Whisper({whisper_dim}) → proj({args.d_model})")
    print(f"  Atrous stack: {args.n_dilation_layers} layers, k={args.kernel_size}")
    print(f"  DualPool({args.d_model*2}) → MLP → 5")

    criterion = nn.MultiLabelSoftMarginLoss()
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                             eta_min=args.lr * 0.05)

    # ------------------------------------------------------------------
    # 6. Training
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs ...")
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "d7_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for hb, wb, yb in train_dl:
            hb, wb, yb = hb.to(device), wb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(hb, wb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * hb.size(0)
        tr_loss /= len(y_tr)
        scheduler.step()

        model.eval()
        vl, vlog, vlab = 0.0, [], []
        with torch.no_grad():
            for hb, wb, yb in test_dl:
                hb, wb, yb = hb.to(device), wb.to(device), yb.to(device)
                lg = model(hb, wb)
                vl += criterion(lg, yb).item() * hb.size(0)
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
    # 7. Final evaluation
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    tlog, tlab = [], []
    with torch.no_grad():
        for hb, wb, yb in test_dl:
            tlog.append(model(hb.to(device), wb.to(device)).cpu().numpy())
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
    # 8. Save
    # ------------------------------------------------------------------
    with (args.out_dir / "d7_perclass_results.csv").open("w", newline="", encoding="utf-8") as f:
        wrt = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        wrt.writeheader()
        for t in STUTTER_TYPES:
            wrt.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                          "precision": test_m[f"pre_{t}"],
                          "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    with (args.out_dir / "d7_train_history.csv").open("w", newline="", encoding="utf-8") as f:
        wrt = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        wrt.writeheader(); wrt.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "D7",
        "title": "Atrous-CNN Dual-SSL Fusion (HuBERT+Whisper, dilated multi-scale)",
        "device": str(device),
        "hubert_alias": args.hubert_alias, "hubert_layer": args.hubert_layer, "hubert_dim": int(hubert_dim),
        "whisper_alias": args.whisper_alias, "whisper_layer": args.whisper_layer, "whisper_dim": int(whisper_dim),
        "d_model": args.d_model, "n_dilation_layers": args.n_dilation_layers,
        "dilations": [2**i for i in range(args.n_dilation_layers)],
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
    (args.out_dir / "d7_run_report.json").write_text(json.dumps(run_report, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val")
        axes[0].set_title(f"D7 Loss (Atrous-CNN, HuBERT+Whisper, {args.n_dilation_layers} layers)")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="purple", marker="o", ms=4)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title(f"D7 Validation Macro-F1 (Atrous-CNN d={args.d_model})")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "d7_train_curves.png", dpi=160); plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        pf1  = [test_m[f"f1_{t}"] for t in STUTTER_TYPES]
        bars = ax2.bar(STUTTER_TYPES, pf1,
                       color=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"], width=0.55)
        ax2.axhline(test_m["macro_f1"], ls="--", color="black", alpha=0.5,
                    label=f"Macro-F1={test_m['macro_f1']:.4f}")
        for bar, v in zip(bars, pf1):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax2.set_ylim(0, 1); ax2.set_title("D7 Per-class F1 (Atrous-CNN HuBERT+Whisper)")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "d7_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"\n  Saved: {args.fig_dir / 'd7_train_curves.png'}")
        print(f"  Saved: {args.fig_dir / 'd7_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  D7 complete.")
    print(f"   Report : {args.out_dir / 'd7_run_report.json'}")
    print(f"   Ckpt   : {best_ckpt}")


if __name__ == "__main__":
    main()
