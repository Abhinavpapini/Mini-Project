"""C4: Cross-Attention SSL Fusion — HuBERT-large × Whisper-large.

Two complementary SSL models are fused via multi-head cross-attention:
  - HuBERT-large L21  (1024-dim): phonetic/acoustic representation
  - Whisper-large L28 (1280-dim): ASR-trained, text-aligned representation

FUSION STRATEGY — Chunked Multi-Head Cross-Attention:
  1. Project both to shared d_model (256-dim)
  2. Reshape 256-dim → N_chunks × d_head (8 × 32) — treats feature
     sub-spaces as a "sequence" of tokens for attention
  3. Bidirectional cross-attention:
       H→W: Q=HuBERT chunks,  K=V=Whisper chunks  → attended_H
       W→H: Q=Whisper chunks, K=V=HuBERT chunks   → attended_W
  4. Concatenate & pool → 256-dim → MLP → 5-class MLSM

The chunked approach enables meaningful cross-attention on fixed-size
SSL vectors without requiring raw temporal frame sequences.

Run command:
    python experiments/C4_xattn_cnn.py \
        --hubert-alias hubert-large --hubert-layer 21 \
        --whisper-alias whisper-large --whisper-layer 28 \
        --d-model 256 --n-heads 8 \
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
import torch.nn.functional as F
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
    p = argparse.ArgumentParser(description="C4: Cross-Attention HuBERT×Whisper Fusion")
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
    # Fusion params
    p.add_argument("--d-model",         type=int,  default=256,
                   help="Shared projection dimension (chunked into n_heads × d_head)")
    p.add_argument("--n-heads",         type=int,  default=8,
                   help="Number of attention heads (d_head = d_model // n_heads)")
    p.add_argument("--attn-dropout",    type=float,default=0.1)
    # MLP classifier
    p.add_argument("--mlp-hidden",      type=int,  default=256)
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
    p.add_argument("--ckpt-dir",  type=Path, default=Path("artifacts/checkpoints/C4"))
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
# Cross-Attention Fusion Model
# ---------------------------------------------------------------------------

class ChunkedCrossAttention(nn.Module):
    """
    Single-direction cross-attention over chunked feature vectors.
    Query: x_q  [B, d_model]  → reshaped to [B, n_heads, d_head]
    Key/Value: x_kv [B, d_model] → same reshape
    Output: attended [B, d_model]
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.Wq  = nn.Linear(d_model, d_model, bias=False)
        self.Wk  = nn.Linear(d_model, d_model, bias=False)
        self.Wv  = nn.Linear(d_model, d_model, bias=False)
        self.Wo  = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        B = x_q.size(0)
        # Project & reshape: [B, d_model] → [B, n_heads, d_head]
        Q = self.Wq(x_q).view(B, self.n_heads, self.d_head)   # [B, H, dh]
        K = self.Wk(x_kv).view(B, self.n_heads, self.d_head)  # [B, H, dh]
        V = self.Wv(x_kv).view(B, self.n_heads, self.d_head)  # [B, H, dh]
        # Per-head dot-product attention: [B, H, dh] × [B, H, dh] → [B, H]
        attn = (Q * K).sum(dim=-1) * self.scale                # [B, H]
        attn = torch.softmax(attn.unsqueeze(-1), dim=1)        # [B, H, 1]
        out  = (attn * V).view(B, -1)                          # [B, d_model]
        out  = self.drop(self.Wo(out))
        return self.norm(x_q + out)                            # residual


class CrossAttnFusionModel(nn.Module):
    def __init__(self, hubert_dim: int, whisper_dim: int, d_model: int,
                 n_heads: int, mlp_hidden: int, n_classes: int,
                 dropout: float, attn_dropout: float) -> None:
        super().__init__()
        # Input projections
        self.proj_h = nn.Sequential(
            nn.Linear(hubert_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )
        self.proj_w = nn.Sequential(
            nn.Linear(whisper_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )
        # Bidirectional cross-attention
        self.xattn_hw = ChunkedCrossAttention(d_model, n_heads, attn_dropout)  # H→W
        self.xattn_wh = ChunkedCrossAttention(d_model, n_heads, attn_dropout)  # W→H
        # MLP classifier
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, n_classes),
        )

    def forward(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ph = self.proj_h(h)          # [B, d_model]
        pw = self.proj_w(w)          # [B, d_model]
        ah = self.xattn_hw(ph, pw)   # H attends to W: [B, d_model]
        aw = self.xattn_wh(pw, ph)   # W attends to H: [B, d_model]
        fused = torch.cat([ah, aw], dim=-1)  # [B, 2*d_model]
        return self.head(fused)


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

    assert h_feats.shape[0] == w_feats.shape[0], "Feature count mismatch!"
    n = h_feats.shape[0]
    hubert_dim  = h_feats.shape[1]
    whisper_dim = w_feats.shape[1]

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
    # 3. Split + StandardScaler (per-model)
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
    model = CrossAttnFusionModel(
        hubert_dim=hubert_dim, whisper_dim=whisper_dim,
        d_model=args.d_model, n_heads=args.n_heads,
        mlp_hidden=args.mlp_hidden, n_classes=len(STUTTER_TYPES),
        dropout=args.dropout, attn_dropout=args.attn_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    d_head   = args.d_model // args.n_heads
    print(f"\nC4 model parameters: {n_params:,}")
    print(f"  proj_H: {hubert_dim}→{args.d_model}")
    print(f"  proj_W: {whisper_dim}→{args.d_model}")
    print(f"  CrossAttn H→W + W→H: {args.n_heads} heads × {d_head}-dim")
    print(f"  MLP: {args.d_model*2}→{args.mlp_hidden}→{args.mlp_hidden//2}→5")

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
    best_ckpt = args.ckpt_dir / "c4_best.pt"

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
    with (args.out_dir / "c4_perclass_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    with (args.out_dir / "c4_train_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "C4",
        "title": "Cross-Attention SSL Fusion: HuBERT-large × Whisper-large",
        "device": str(device),
        "hubert_alias": args.hubert_alias, "hubert_layer": args.hubert_layer,
        "hubert_dim": int(hubert_dim),
        "whisper_alias": args.whisper_alias, "whisper_layer": args.whisper_layer,
        "whisper_dim": int(whisper_dim),
        "d_model": args.d_model, "n_heads": args.n_heads,
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
    (args.out_dir / "c4_run_report.json").write_text(json.dumps(run_report, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val")
        axes[0].set_title(f"C4 Loss (HuBERT-L{args.hubert_layer} × Whisper-L{args.whisper_layer})")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="royalblue", marker="o", ms=4)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("C4 Validation Macro-F1 (Cross-Attn Fusion)")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "c4_train_curves.png", dpi=160); plt.close(fig)

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
        ax2.set_title("C4 Per-class F1 (HuBERT × Whisper Cross-Attention)")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "c4_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"\n  Saved: {args.fig_dir / 'c4_train_curves.png'}")
        print(f"  Saved: {args.fig_dir / 'c4_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  C4 complete.")
    print(f"   Report : {args.out_dir / 'c4_run_report.json'}")
    print(f"   Ckpt   : {best_ckpt}")


if __name__ == "__main__":
    main()
