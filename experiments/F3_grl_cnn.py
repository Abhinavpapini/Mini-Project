"""F3: Adversarial Speaker Disentanglement via Gradient Reversal Layer (GRL).

MOTIVATION: HuBERT features carry both stutter-type information AND
speaker-identity information. Speaker-specific features can overfit to
speakers seen during training, hurting generalisation on new speakers.
GRL forces the shared encoder to produce SPEAKER-INVARIANT representations.

ARCHITECTURE:
  PCA-32 features → Shared Encoder (FC layers) → Bottleneck-32
  ├─→ Stutter Classifier: bottleneck → FC → 5-class MLSM
  └─→ Speaker Classifier: bottleneck → GRL(λ) → FC → N-show softmax

  GRL(λ): passes forward unchanged, multiplies gradient by -λ during backprop
  → encoder is penalised for making speaker information extractable

SPEAKER PROXY: We use "Show" (TV programme) from the SEP-28k metadata as a
speaker group proxy. Different shows = different speakers/recording conditions.
This is a coarse but available approximation (no speaker-ID metadata in SEP-28k).

LOSS:
  L_total = L_stutter + λ_grl * L_speaker
  (L_speaker backpropagated through GRL → encoder trained to confuse it)

Run command:
    python experiments/F3_grl_cnn.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --ssl-pca-dim 32 \
        --bottleneck-dim 32 \
        --grl-lambda 0.5 \
        --lambda-warmup 10 \
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
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F3: GRL speaker disentanglement + CNN-1D")
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
    # Feature params
    p.add_argument("--ssl-pca-dim",     type=int,  default=32)
    p.add_argument("--bottleneck-dim",  type=int,  default=32)
    # GRL params
    p.add_argument("--grl-lambda",      type=float,default=0.5,
                   help="GRL reversal strength (gradient multiplied by -lambda)")
    p.add_argument("--lambda-warmup",   type=int,  default=10,
                   help="Epochs to linearly ramp GRL lambda from 0 to grl-lambda")
    # CNN params
    p.add_argument("--cnn-channels",    type=str,  default="64,128,256")
    p.add_argument("--dropout",         type=float,default=0.3)
    # Training
    p.add_argument("--test-size",       type=float,default=0.20)
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--epochs",          type=int,  default=50)
    p.add_argument("--batch-size",      type=int,  default=256)
    p.add_argument("--lr",              type=float,default=3e-4)
    p.add_argument("--weight-decay",    type=float,default=1e-4)
    p.add_argument("--threshold",       type=float,default=0.5)
    p.add_argument("--out-dir",   type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",   type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir",  type=Path, default=Path("artifacts/checkpoints/F3"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GRLFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


class GRL(nn.Module):
    def __init__(self, lam: float = 1.0) -> None:
        super().__init__()
        self.lam = lam

    def set_lambda(self, lam: float) -> None:
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GRLFunction.apply(x, self.lam)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def norm_text(x: object) -> str:
    return str(x).strip()


def load_multilabel_map(csv_path: Path) -> Tuple[
        Dict[Tuple[str, str, str], np.ndarray],
        Dict[Tuple[str, str, str], str]]:
    labels_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    show_map:   Dict[Tuple[str, str, str], str] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (norm_text(row["Show"]), norm_text(row["EpId"]), norm_text(row["ClipId"]))
            labels_map[key] = np.array(
                [1 if float(norm_text(row[t])) >= 1 else 0 for t in STUTTER_TYPES],
                dtype=np.float32,
            )
            show_map[key] = norm_text(row["Show"])
    return labels_map, show_map


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
    raise FileNotFoundError(f"layer_{layer}.npy not found")


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    class_freq = np.where(y.mean(axis=0) == 0, 1.0, y.mean(axis=0))
    inv_freq   = 1.0 / class_freq
    weights    = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        pm = y[i] > 0
        weights[i] = inv_freq[pm].max() if pm.any() else inv_freq.min()
    return weights / weights.min()


# ---------------------------------------------------------------------------
# Model: Shared Encoder + Stutter Head + Speaker Adversary (GRL)
# ---------------------------------------------------------------------------

class GRLModel(nn.Module):
    def __init__(self, in_dim: int, bottleneck: int, channels: List[int],
                 n_classes: int, n_speakers: int, dropout: float) -> None:
        super().__init__()
        # Shared encoder: PCA-32 → 64 → bottleneck-32
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, bottleneck), nn.BatchNorm1d(bottleneck), nn.GELU(),
        )
        # Stutter classifier head (CNN-1D, same as F1 baseline)
        cnn_layers: List[nn.Module] = []
        c_in = 1
        for c_out in channels:
            cnn_layers += [nn.Conv1d(c_in, c_out, kernel_size=3, padding=1),
                           nn.BatchNorm1d(c_out), nn.GELU()]
            c_in = c_out
        self.cnn  = nn.Sequential(*cnn_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.stutter_head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(channels[-1], channels[-1] // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(channels[-1] // 2, n_classes),
        )
        # Speaker adversary: bottleneck → GRL → FC → speaker softmax
        self.grl = GRL(lam=0.0)
        self.speaker_head = nn.Sequential(
            nn.Linear(bottleneck, 32), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, n_speakers),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)                             # [B, bottleneck]
        # Stutter branch (CNN-1D over bottleneck sequence)
        stutter_logits = self.stutter_head(self.pool(self.cnn(z.unsqueeze(1))))
        # Speaker adversary branch (through GRL)
        speaker_logits = self.speaker_head(self.grl(z))
        return stutter_logits, speaker_logits


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
    # 1. Features + labels + show IDs
    # ------------------------------------------------------------------
    print(f"\nLoading HuBERT-large layer {args.hubert_layer} cache ...")
    ssl_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    if ssl_feats.ndim != 2:
        ssl_feats = ssl_feats.reshape(ssl_feats.shape[0], -1)
    n = ssl_feats.shape[0]
    print(f"  SSL features: {ssl_feats.shape}")

    label_map:  Dict[Tuple[str, str, str], np.ndarray] = {}
    show_map:   Dict[Tuple[str, str, str], str]        = {}
    lm1, sm1 = load_multilabel_map(args.sep_labels)
    lm2, sm2 = load_multilabel_map(args.fluency_labels)
    label_map.update(lm1); label_map.update(lm2)
    show_map.update(sm1);  show_map.update(sm2)

    clip_keys = sorted_clip_keys(args.clips_root)[:n]
    y_labels  = np.array(
        [label_map.get(k, np.zeros(len(STUTTER_TYPES), np.float32)) for k in clip_keys],
        dtype=np.float32,
    )
    show_ids = [show_map.get(k, "unknown") for k in clip_keys]

    # Encode show IDs as integer speaker-group labels
    le = LabelEncoder()
    show_enc = le.fit_transform(show_ids).astype(np.int64)
    n_speakers = len(le.classes_)
    print(f"\nDataset: {n} samples | Label distribution:")
    for i, t in enumerate(STUTTER_TYPES):
        pos = int(y_labels[:, i].sum())
        print(f"  {t:15s}: {pos:5d} pos  ({pos/n:.2%})")
    print(f"\n  Speaker groups (shows): {n_speakers}")
    for cls in le.classes_:
        cnt = int((np.array(show_ids) == cls).sum())
        print(f"    {cls:25s}: {cnt:5d} clips")

    # ------------------------------------------------------------------
    # 2. Split + PCA-32
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y_labels[:, 0].astype(int)
    )
    sc = StandardScaler()
    x_tr_sc = sc.fit_transform(ssl_feats[train_idx])
    x_te_sc = sc.transform(ssl_feats[test_idx])

    print(f"\nSSL PCA-{args.ssl_pca_dim} ...")
    pca = PCA(n_components=args.ssl_pca_dim, random_state=args.seed)
    x_tr = pca.fit_transform(x_tr_sc).astype(np.float32)
    x_te = pca.transform(x_te_sc).astype(np.float32)
    print(f"  expl_var={pca.explained_variance_ratio_.sum():.3f}")

    y_tr = y_labels[train_idx];   y_te  = y_labels[test_idx]
    s_tr = show_enc[train_idx];   s_te  = show_enc[test_idx]

    # ------------------------------------------------------------------
    # 3. DataLoaders (features, stutter labels, speaker labels)
    # ------------------------------------------------------------------
    sw = compute_sample_weights(y_tr)
    sampler  = WeightedRandomSampler(torch.from_numpy(sw), len(sw), replacement=True)
    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr),
                      torch.from_numpy(s_tr)),
        batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True,
    )
    test_dl  = DataLoader(
        TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te),
                      torch.from_numpy(s_te)),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    channels = [int(c) for c in args.cnn_channels.split(",")]
    model    = GRLModel(in_dim=args.ssl_pca_dim, bottleneck=args.bottleneck_dim,
                        channels=channels, n_classes=len(STUTTER_TYPES),
                        n_speakers=n_speakers, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nF3 model parameters: {n_params:,}")
    print(f"  Encoder: {args.ssl_pca_dim}→64→{args.bottleneck_dim}")
    print(f"  Stutter head: CNN-{channels}→5 (MLSM)")
    print(f"  Speaker adversary: GRL(λ={args.grl_lambda})→{n_speakers} (CE)")
    print(f"  GRL λ warm-up: 0 → {args.grl_lambda} over {args.lambda_warmup} epochs")

    stutter_crit = nn.MultiLabelSoftMarginLoss()
    speaker_crit = nn.CrossEntropyLoss()
    opt          = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                               eta_min=args.lr * 0.05)

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs ...")
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "f3_best.pt"

    for epoch in range(1, args.epochs + 1):
        # GRL lambda warm-up
        lam = min(args.grl_lambda, args.grl_lambda * epoch / max(args.lambda_warmup, 1))
        model.grl.set_lambda(lam)

        model.train()
        tr_sl = tr_spl = 0.0
        for xb, yb, sb in train_dl:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            opt.zero_grad()
            stutter_logits, speaker_logits = model(xb)
            loss = stutter_crit(stutter_logits, yb) + speaker_crit(speaker_logits, sb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sl  += stutter_crit(stutter_logits.detach(), yb).item() * xb.size(0)
            tr_spl += speaker_crit(speaker_logits.detach(), sb).item() * xb.size(0)
        tr_sl /= len(y_tr); tr_spl /= len(y_tr)
        scheduler.step()

        model.eval()
        vl, vlog, vlab = 0.0, [], []
        with torch.no_grad():
            for xb, yb, sb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                sl, _ = model(xb)
                vl += stutter_crit(sl, yb).item() * xb.size(0)
                vlog.append(sl.cpu().numpy()); vlab.append(yb.cpu().numpy())
        vl /= len(y_te)
        vm = evaluate_multilabel(np.concatenate(vlab), np.concatenate(vlog), args.threshold)
        macro_f1 = vm["macro_f1"]
        row = {"epoch": epoch, "stutter_loss": round(tr_sl, 6), "speaker_loss": round(tr_spl, 6),
               "val_loss": round(vl, 6), "macro_f1": round(macro_f1, 6), "grl_lambda": round(lam, 4)}
        history.append(row)
        print(f"epoch={epoch:02d}  stutter={tr_sl:.5f}  speaker={tr_spl:.5f}  "
              f"val={vl:.5f}  macro_f1={macro_f1:.5f}  λ={lam:.3f}")
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
        for xb, yb, _ in test_dl:
            sl, _ = model(xb.to(device))
            tlog.append(sl.cpu().numpy()); tlab.append(yb.numpy())
    test_m = evaluate_multilabel(np.concatenate(tlab), np.concatenate(tlog), args.threshold)

    print(f"\n--- Test Results (best ckpt, macro_f1={best_macro_f1:.5f}) ---")
    print(f"  Macro-F1   : {test_m['macro_f1']:.5f}")
    print(f"  Micro-F1   : {test_m['micro_f1']:.5f}")
    print(f"  Macro-Pre  : {test_m['macro_pre']:.5f}")
    print(f"  Macro-Rec  : {test_m['macro_rec']:.5f}")
    print(f"  Macro-AUPRC: {test_m['macro_auprc']:.5f}")
    print(f"  Speaker groups: {n_speakers} shows")
    print(f"\n  Per-class F1:")
    for t in STUTTER_TYPES:
        print(f"    {t:15s}: F1={test_m[f'f1_{t}']:.5f}  "
              f"P={test_m[f'pre_{t}']:.5f}  R={test_m[f'rec_{t}']:.5f}  "
              f"AUPRC={test_m[f'auprc_{t}']:.5f}")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    with (args.out_dir / "f3_perclass_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    with (args.out_dir / "f3_train_history.csv").open("w", newline="", encoding="utf-8") as f:
        fnames = ["epoch","stutter_loss","speaker_loss","val_loss","macro_f1","grl_lambda"]
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "F3",
        "title": "GRL Speaker Disentanglement + CNN-1D (HuBERT-large, adversarial robustness)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "ssl_pca_dim": args.ssl_pca_dim,
        "bottleneck_dim": args.bottleneck_dim, "grl_lambda": args.grl_lambda,
        "lambda_warmup": args.lambda_warmup, "n_speakers": int(n_speakers),
        "speaker_groups": list(le.classes_),
        "n_params": int(n_params), "epochs": args.epochs, "best_epoch": int(best_ep),
        "macro_f1": test_m["macro_f1"], "micro_f1": test_m["micro_f1"],
        "macro_precision": test_m["macro_pre"], "macro_recall": test_m["macro_rec"],
        "macro_auprc": test_m["macro_auprc"],
        "per_class": {t: {"f1": test_m[f"f1_{t}"], "precision": test_m[f"pre_{t}"],
                          "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]}
                      for t in STUTTER_TYPES},
        "best_checkpoint": str(best_ckpt),
    }
    (args.out_dir / "f3_run_report.json").write_text(json.dumps(run_report, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["stutter_loss"] for r in history], label="stutter (MLSM)")
        axes[0].plot(ep_x, [r["speaker_loss"] for r in history], label="speaker (CE)", ls="--")
        axes[0].plot(ep_x, [r["val_loss"]     for r in history], label="val stutter")
        axes[0].set_title(f"F3 GRL Losses (λ={args.grl_lambda}, {n_speakers} groups)")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="crimson", marker="o", ms=4)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("F3 Validation Macro-F1 (Speaker-disentangled)")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "f3_train_curves.png", dpi=160); plt.close(fig)

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
        ax2.set_title(f"F3 Per-class F1 (GRL λ={args.grl_lambda}, {n_speakers} speaker groups)")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "f3_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"\n  Saved: {args.fig_dir / 'f3_train_curves.png'}")
        print(f"  Saved: {args.fig_dir / 'f3_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  F3 complete.")
    print(f"   Report : {args.out_dir / 'f3_run_report.json'}")
    print(f"   Ckpt   : {best_ckpt}")


if __name__ == "__main__":
    main()
