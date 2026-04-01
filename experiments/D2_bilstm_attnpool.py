"""D2: Multi-label stutter classification with BiLSTM + Attention Pooling.

Architecture:
  HuBERT-large (layer 21, PCA-64) → [B, 64] → token embed [B, 64, d_hidden]
  → 2-layer BiLSTM → [B, 64, 2*d_hidden]
  → Attention pooling (learned weights) → [B, 2*d_hidden]
  → Linear classifier → [B, 5]

Key value: interpretable attention over feature positions — which PCA
components the model attends to when predicting each stutter type.

Ablation vs D1: BiLSTM recurrent vs Conformer (MHSA + conv)
  - D2 uses 64-dim PCA (more features than D1's 32)
  - D2 BiLSTM processes feature positions sequentially
  - D2 provides per-position attention weights for interpretability

Run command:
    python experiments/D2_bilstm_attnpool.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --ssl-pca-dim 64 \
        --lstm-hidden 128 \
        --lstm-layers 2 \
        --dropout 0.2 \
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


STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D2: BiLSTM + Attention Pooling, multi-label")
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
    p.add_argument("--ssl-pca-dim",    type=int,  default=64,
                   help="PCA dims (64 for D2, broader feature set than D1's 32)")
    p.add_argument("--lstm-hidden",    type=int,  default=128,
                   help="Hidden dim per direction in BiLSTM")
    p.add_argument("--lstm-layers",    type=int,  default=2,
                   help="Number of BiLSTM layers")
    p.add_argument("--attn-dim",       type=int,  default=64,
                   help="Attention projection dimension")
    p.add_argument("--dropout",        type=float,default=0.2)
    p.add_argument("--test-size",      type=float,default=0.20)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--epochs",         type=int,  default=30)
    p.add_argument("--batch-size",     type=int,  default=256)
    p.add_argument("--lr",             type=float,default=3e-4)
    p.add_argument("--weight-decay",   type=float,default=1e-4)
    p.add_argument("--threshold",      type=float,default=0.5)
    p.add_argument("--out-dir",  type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",  type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/D2"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data utilities (same pattern as D-series)
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
# D2 Model: BiLSTM + Attention Pooling
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """
    Additive (Bahdanau) attention over a sequence of hidden states.
    Returns pooled context vector and attention weights (for interpretability).

    Input : [B, T, H]  (T positions, H hidden dim)
    Output: [B, H], [B, T] attention weights
    """

    def __init__(self, hidden_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v    = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: [B, T, H]
        e       = self.v(torch.tanh(self.proj(h))).squeeze(-1)  # [B, T]
        weights = F.softmax(e, dim=-1)                           # [B, T]
        context = (h * weights.unsqueeze(-1)).sum(dim=1)         # [B, H]
        return context, weights


class D2BiLSTMAttnPool(nn.Module):
    """
    Treats each PCA dimension as a token (sequence position).
    BiLSTM processes left-to-right and right-to-left across feature positions.
    Attention pooling aggregates bidirectional states → classification head.

    This provides interpretable attention showing which PCA components
    drive each stutter type prediction.
    """

    def __init__(self, in_dim: int, lstm_hidden: int, lstm_layers: int,
                 attn_dim: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        self.token_embed = nn.Sequential(
            nn.Linear(1, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU(),
        )
        self.bilstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attn = AttentionPool(hidden_dim=lstm_hidden * 2, attn_dim=attn_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, F]  (F = PCA dim)
        tokens = self.token_embed(x.unsqueeze(-1))    # [B, F, lstm_hidden]
        h, _   = self.bilstm(tokens)                   # [B, F, 2*lstm_hidden]
        ctx, attn_w = self.attn(h)                     # [B, 2*lstm_hidden], [B, F]
        logits = self.head(ctx)                         # [B, num_classes]
        return logits, attn_w


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
        m[f"f1_{t}"]  = float(pf1[i]);  m[f"pre_{t}"] = float(ppre[i])
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
    # 1. Features
    # ------------------------------------------------------------------
    print(f"\nLoading HuBERT-large layer {args.hubert_layer} cache ...")
    ssl_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    if ssl_feats.ndim != 2:
        ssl_feats = ssl_feats.reshape(ssl_feats.shape[0], -1)
    n = ssl_feats.shape[0]
    print(f"  SSL features: {ssl_feats.shape}")

    # ------------------------------------------------------------------
    # 2. Labels
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

    # ------------------------------------------------------------------
    # 3. Split + PCA (64 dims for D2)
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y[:, 0].astype(int)
    )
    pca_dim = min(args.ssl_pca_dim, ssl_feats.shape[1])
    sc  = StandardScaler()
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    x_tr = pca.fit_transform(sc.fit_transform(ssl_feats[train_idx])).astype(np.float32)
    x_te = pca.transform(sc.transform(ssl_feats[test_idx])).astype(np.float32)
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f"\nSSL PCA-{pca_dim} explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"BiLSTM processes {pca_dim} feature tokens sequentially")

    # ------------------------------------------------------------------
    # 4. Sampler + DataLoaders
    # ------------------------------------------------------------------
    sample_weights = compute_sample_weights(y_tr)
    sampler  = WeightedRandomSampler(torch.from_numpy(sample_weights),
                                     len(sample_weights), replacement=True)
    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    test_ds  = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    # ------------------------------------------------------------------
    # 5. Model
    # ------------------------------------------------------------------
    model = D2BiLSTMAttnPool(
        in_dim=pca_dim, lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers, attn_dim=args.attn_dim,
        dropout=args.dropout, num_classes=len(STUTTER_TYPES),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nD2 model parameters: {n_params:,}")
    print(f"  BiLSTM: {pca_dim} tokens → hidden={args.lstm_hidden}×2 "
          f"× {args.lstm_layers} layers → AttnPool → head")

    criterion = nn.MultiLabelSoftMarginLoss()
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    # ------------------------------------------------------------------
    # 6. Training
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs ...")
    history: List[Dict] = []
    best_macro_f1, best_attn_weights = -1.0, None
    best_ckpt = args.ckpt_dir / "d2_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        vl, vlog, vlab, vattn = 0.0, [], [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits, attn_w = model(xb)
                vl += criterion(logits, yb).item() * xb.size(0)
                vlog.append(logits.cpu().numpy()); vlab.append(yb.cpu().numpy())
                vattn.append(attn_w.cpu().numpy())
        vl /= len(test_ds)
        vm = evaluate_multilabel(np.concatenate(vlab), np.concatenate(vlog), args.threshold)
        macro_f1 = vm["macro_f1"]

        row = {"epoch": epoch, "train_loss": round(tr_loss, 6),
               "val_loss": round(vl, 6), "macro_f1": round(macro_f1, 6),
               "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        print(f"epoch={epoch:02d}  train_loss={tr_loss:.5f}  val_loss={vl:.5f}  macro_f1={macro_f1:.5f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_attn_weights = np.concatenate(vattn)
            torch.save(model.state_dict(), best_ckpt)

    # ------------------------------------------------------------------
    # 7. Final evaluation
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    tlog, tlab, tattn = [], [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            lg, aw = model(xb.to(device))
            tlog.append(lg.cpu().numpy()); tlab.append(yb.numpy())
            tattn.append(aw.cpu().numpy())
    test_m = evaluate_multilabel(np.concatenate(tlab), np.concatenate(tlog), args.threshold)
    attn_weights = np.concatenate(tattn)   # [N_test, pca_dim]

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

    # Attention summary: mean attention over top-5 PCA dims
    mean_attn = attn_weights.mean(axis=0)
    top5_idx  = np.argsort(mean_attn)[::-1][:5]
    print(f"\n  Top-5 attended PCA dims (interpretability):")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"    #{rank}: PCA-{idx:02d}  attn={mean_attn[idx]:.4f}  "
              f"(expl_var={pca.explained_variance_ratio_[idx]:.4f})")

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    perclass_csv = args.out_dir / "d2_perclass_results.csv"
    with perclass_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    hist_csv = args.out_dir / "d2_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1","lr"])
        w.writeheader(); w.writerows(history)

    # Save attention weights for external analysis
    attn_npy = args.out_dir / "d2_attn_weights.npy"
    np.save(attn_npy, attn_weights)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "D2", "title": "BiLSTM + Attention Pooling (HuBERT-large PCA-64)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "ssl_pca_dim": int(pca_dim),
        "lstm_hidden": args.lstm_hidden, "lstm_layers": args.lstm_layers,
        "attn_dim": args.attn_dim, "n_params": int(n_params),
        "loss": "MultiLabelSoftMarginLoss",
        "epochs": args.epochs, "best_epoch": int(best_ep),
        "batch_size": args.batch_size, "lr": args.lr, "threshold": args.threshold,
        "macro_f1": test_m["macro_f1"], "micro_f1": test_m["micro_f1"],
        "macro_precision": test_m["macro_pre"], "macro_recall": test_m["macro_rec"],
        "macro_auprc": test_m["macro_auprc"],
        "top5_attn_pca_dims": top5_idx.tolist(),
        "mean_attn_weights": mean_attn.tolist(),
        "per_class": {t: {"f1": test_m[f"f1_{t}"], "precision": test_m[f"pre_{t}"],
                          "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]}
                      for t in STUTTER_TYPES},
        "best_checkpoint": str(best_ckpt),
    }
    report_json = args.out_dir / "d2_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 9. Figures
    # ------------------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_x    = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs_x, [r["train_loss"] for r in history], label="train")
        axes[0].plot(epochs_x, [r["val_loss"]   for r in history], label="val")
        axes[0].set_title("D2 Training Loss (BiLSTM + AttnPool)")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs_x, [r["macro_f1"] for r in history], color="green", marker="o", ms=3)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("D2 Validation Macro-F1"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "d2_train_curves.png", dpi=160); plt.close(fig)
        print(f"  Saved: {args.fig_dir / 'd2_train_curves.png'}")

        # Attention heatmap (mean over test set)
        fig2, ax2 = plt.subplots(figsize=(14, 3.5))
        im = ax2.bar(range(pca_dim), mean_attn,
                     color=["#e63946" if i in top5_idx else "#457b9d" for i in range(pca_dim)])
        ax2.set_title(f"D2 Mean Attention Weights over {pca_dim} PCA Dimensions")
        ax2.set_xlabel("PCA Component Index"); ax2.set_ylabel("Mean Attention Weight")
        for idx in top5_idx:
            ax2.text(idx, mean_attn[idx] + 0.0005, f"PC{idx}", ha="center",
                     va="bottom", fontsize=7, color="#e63946", fontweight="bold")
        ax2.grid(alpha=0.2)
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "d2_attn_heatmap.png", dpi=160); plt.close(fig2)
        print(f"  Saved: {args.fig_dir / 'd2_attn_heatmap.png'}")

        # Per-class F1 bar
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        pf1  = [test_m[f"f1_{t}"] for t in STUTTER_TYPES]
        bars = ax3.bar(STUTTER_TYPES, pf1,
                       color=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"], width=0.55)
        ax3.axhline(test_m["macro_f1"], ls="--", color="black", alpha=0.5,
                    label=f"Macro-F1={test_m['macro_f1']:.4f}")
        for b, v in zip(bars, pf1):
            ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax3.set_ylim(0, 1); ax3.set_title("D2 Per-class F1 (BiLSTM + AttnPool)")
        ax3.set_xlabel("Stutter Type"); ax3.set_ylabel("F1"); ax3.legend()
        fig3.tight_layout()
        fig3.savefig(args.fig_dir / "d2_perclass_f1.png", dpi=160); plt.close(fig3)
        print(f"  Saved: {args.fig_dir / 'd2_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  D2 complete.")
    print(f"   Report    : {report_json}")
    print(f"   Per-class : {perclass_csv}")
    print(f"   History   : {hist_csv}")
    print(f"   Attn wts  : {attn_npy}")
    print(f"   Ckpt      : {best_ckpt}")


if __name__ == "__main__":
    main()
