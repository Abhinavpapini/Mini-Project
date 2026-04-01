"""B4: Autoencoder Bottleneck Compression benchmark.

Unlike B2 (KPCA) and B3 (MDS) which are task-agnostic transformations,
the autoencoder LEARNS the 32-dim compression from data via reconstruction.

Two-stage pipeline:
  Stage 1 — AE pre-training (unsupervised, reconstruction):
    Encoder: 1024 → 512 → 256 → 128 → 32  (deep bottleneck)
    Decoder: 32 → 128 → 256 → 512 → 1024
    Loss: MSE reconstruction

  Stage 2 — Classification on frozen bottleneck:
    Freeze encoder, extract 32-dim features for all clips
    CNN-1D (same arch as F1/B1/B2/B3) → MultiLabelSoftMarginLoss

Key hypothesis: A LEARNED 32-dim representation may preserve more
task-relevant structure than PCA (which maximises variance, not discriminability).

Run command:
    python experiments/B4_ae_cnn.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --bottleneck-dim 32 \
        --ae-epochs 50 \
        --ae-lr 5e-4 \
        --clf-epochs 30 \
        --clf-lr 3e-4 \
        --batch-size 256
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
    p = argparse.ArgumentParser(description="B4: AE Bottleneck + CNN-1D, multi-label")
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
    # AE params
    p.add_argument("--bottleneck-dim", type=int,  default=32)
    p.add_argument("--ae-hidden",      type=str,  default="512,256,128",
                   help="Encoder hidden dims (comma-sep, decoder mirrors these)")
    p.add_argument("--ae-dropout",     type=float,default=0.1)
    p.add_argument("--ae-epochs",      type=int,  default=50)
    p.add_argument("--ae-lr",          type=float,default=5e-4)
    # CNN classifier params (same as B1/B2/B3)
    p.add_argument("--cnn-channels",   type=str,  default="64,128,256")
    p.add_argument("--clf-dropout",    type=float,default=0.3)
    p.add_argument("--clf-epochs",     type=int,  default=30)
    p.add_argument("--clf-lr",         type=float,default=3e-4)
    # Shared
    p.add_argument("--batch-size",     type=int,  default=256)
    p.add_argument("--weight-decay",   type=float,default=1e-4)
    p.add_argument("--test-size",      type=float,default=0.20)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--threshold",      type=float,default=0.5)
    p.add_argument("--out-dir",  type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",  type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/B4"))
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
    inv_freq = 1.0 / class_freq
    weights  = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        pos_mask = y[i] > 0
        weights[i] = inv_freq[pos_mask].max() if pos_mask.any() else inv_freq.min()
    return weights / weights.min()


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """
    Deep autoencoder: 1024 → [512,256,128] → 32 → [128,256,512] → 1024
    Symmetric encoder/decoder with BatchNorm + GELU.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 bottleneck: int, dropout: float) -> None:
        super().__init__()
        # Encoder
        enc_layers: List[nn.Module] = []
        d_in = input_dim
        for d_out in hidden_dims:
            enc_layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out),
                           nn.GELU(), nn.Dropout(dropout)]
            d_in = d_out
        enc_layers += [nn.Linear(d_in, bottleneck)]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirrors encoder)
        dec_layers: List[nn.Module] = []
        d_in = bottleneck
        for d_out in reversed(hidden_dims):
            dec_layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out),
                           nn.GELU(), nn.Dropout(dropout)]
            d_in = d_out
        dec_layers += [nn.Linear(d_in, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z    = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ---------------------------------------------------------------------------
# CNN-1D Classifier (identical to B1/B2/B3)
# ---------------------------------------------------------------------------

class CNN1DMultiLabel(nn.Module):
    def __init__(self, in_dim: int, channels: List[int],
                 num_classes: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        c_in = 1
        for c_out in channels:
            layers += [nn.Conv1d(c_in, c_out, kernel_size=3, padding=1),
                       nn.BatchNorm1d(c_out), nn.GELU()]
            c_in = c_out
        self.cnn  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(channels[-1], channels[-1] // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(channels[-1] // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.cnn(x.unsqueeze(1))))


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
    print(f"\nLoading HuBERT-large layer {args.hubert_layer} cache ...")
    ssl_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    if ssl_feats.ndim != 2:
        ssl_feats = ssl_feats.reshape(ssl_feats.shape[0], -1)
    n = ssl_feats.shape[0]
    print(f"  SSL features: {ssl_feats.shape}")

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
    # 2. Train/test split + StandardScaler
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y[:, 0].astype(int)
    )
    sc = StandardScaler()
    x_all_sc = sc.fit_transform(ssl_feats[train_idx])
    # Refit on full train for final transform
    sc2 = StandardScaler()
    x_tr_raw = sc2.fit_transform(ssl_feats[train_idx]).astype(np.float32)
    x_te_raw = sc2.transform(ssl_feats[test_idx]).astype(np.float32)
    x_all_raw = sc2.transform(ssl_feats).astype(np.float32)

    # ------------------------------------------------------------------
    # 3. AE pre-training (Stage 1 — unsupervised reconstruction)
    # ------------------------------------------------------------------
    hidden_dims = [int(h) for h in args.ae_hidden.split(",")]
    in_dim = ssl_feats.shape[1]
    ae = Autoencoder(in_dim, hidden_dims, args.bottleneck_dim, args.ae_dropout).to(device)
    ae_params = sum(p.numel() for p in ae.parameters())

    print(f"\n{'='*60}")
    print(f"STAGE 1: Autoencoder Pre-training ({args.ae_epochs} epochs)")
    print(f"{'='*60}")
    print(f"  Architecture: {in_dim}→{hidden_dims}→{args.bottleneck_dim}→{list(reversed(hidden_dims))}→{in_dim}")
    print(f"  AE parameters: {ae_params:,}")

    ae_dl = DataLoader(TensorDataset(torch.from_numpy(x_tr_raw)),
                       batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    ae_opt = torch.optim.AdamW(ae.parameters(), lr=args.ae_lr, weight_decay=1e-5)
    ae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ae_opt, T_max=args.ae_epochs,
                                                           eta_min=args.ae_lr * 0.05)
    best_ae_loss = float("inf")
    ae_ckpt = args.ckpt_dir / "b4_ae.pt"

    for epoch in range(1, args.ae_epochs + 1):
        ae.train()
        tr_loss = 0.0
        for (xb,) in ae_dl:
            xb = xb.to(device)
            ae_opt.zero_grad()
            recon, _ = ae(xb)
            loss = F.mse_loss(recon, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            ae_opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_idx)
        ae_sched.step()
        if epoch % 10 == 0:
            print(f"  ae_epoch={epoch:02d}  recon_loss={tr_loss:.6f}")
        if tr_loss < best_ae_loss:
            best_ae_loss = tr_loss
            torch.save(ae.state_dict(), ae_ckpt)

    print(f"  AE training done. Best recon_loss={best_ae_loss:.6f}")

    # ------------------------------------------------------------------
    # 4. Extract 32-dim bottleneck features (freeze AE encoder)
    # ------------------------------------------------------------------
    print(f"\nExtracting {args.bottleneck_dim}-dim bottleneck features ...")
    ae.load_state_dict(torch.load(ae_ckpt, map_location=device))
    ae.eval()

    def extract(x_np: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for s in range(0, len(x_np), 512):
                z = ae.encode(torch.from_numpy(x_np[s:s+512]).to(device))
                out.append(z.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    z_tr = extract(x_tr_raw)
    z_te = extract(x_te_raw)

    # Normalise bottleneck (important for CNN stability)
    z_sc = StandardScaler()
    z_tr = z_sc.fit_transform(z_tr).astype(np.float32)
    z_te = z_sc.transform(z_te).astype(np.float32)

    print(f"  Bottleneck features: train={z_tr.shape}, test={z_te.shape}")
    print(f"  Bottleneck stats: mean={z_tr.mean():.4f}, std={z_tr.std():.4f}")

    y_tr, y_te = y[train_idx], y[test_idx]

    # ------------------------------------------------------------------
    # 5. CNN-1D Classifier (Stage 2 — same as B1/B2/B3)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STAGE 2: CNN-1D Classifier ({args.clf_epochs} epochs)")
    print(f"{'='*60}")

    channels = [int(c) for c in args.cnn_channels.split(",")]
    model  = CNN1DMultiLabel(in_dim=args.bottleneck_dim, channels=channels,
                              num_classes=len(STUTTER_TYPES), dropout=args.clf_dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CNN-1D parameters: {n_params:,}")
    print(f"  Input: {args.bottleneck_dim}-dim AE bottleneck")

    sw = compute_sample_weights(y_tr)
    sampler  = WeightedRandomSampler(torch.from_numpy(sw), len(sw), replacement=True)
    train_dl = DataLoader(TensorDataset(torch.from_numpy(z_tr), torch.from_numpy(y_tr)),
                          batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    test_dl  = DataLoader(TensorDataset(torch.from_numpy(z_te), torch.from_numpy(y_te)),
                          batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.MultiLabelSoftMarginLoss()
    opt       = torch.optim.AdamW(model.parameters(), lr=args.clf_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.clf_epochs,
                                                             eta_min=args.clf_lr * 0.05)
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "b4_clf_best.pt"

    for epoch in range(1, args.clf_epochs + 1):
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
    print(f"\n  [B4 vs B-series comparison]")
    print(f"    B1 linear PCA-32 : expected macro_f1 ≈ 0.48-0.50")
    print(f"    B2 KPCA-RBF-32   : 0.394 ⛔")
    print(f"    B3 MDS-32        : 0.464 ⚠️")
    print(f"    B4 AE-32         : {test_m['macro_f1']:.5f}")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    perclass_csv = args.out_dir / "b4_perclass_results.csv"
    with perclass_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    hist_csv = args.out_dir / "b4_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "B4",
        "title": "AE Bottleneck-32 + CNN-1D (HuBERT-large, learned compression)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "bottleneck_dim": args.bottleneck_dim,
        "ae_hidden_dims": hidden_dims, "ae_params": int(ae_params),
        "best_ae_loss": float(best_ae_loss), "ae_epochs": args.ae_epochs,
        "clf_params": int(n_params), "clf_epochs": args.clf_epochs,
        "loss": "MSE (AE) + MultiLabelSoftMarginLoss (clf)",
        "best_clf_epoch": int(best_ep),
        "macro_f1": test_m["macro_f1"], "micro_f1": test_m["micro_f1"],
        "macro_precision": test_m["macro_pre"], "macro_recall": test_m["macro_rec"],
        "macro_auprc": test_m["macro_auprc"],
        "per_class": {t: {"f1": test_m[f"f1_{t}"], "precision": test_m[f"pre_{t}"],
                          "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]}
                      for t in STUTTER_TYPES},
        "ae_checkpoint": str(ae_ckpt), "clf_checkpoint": str(best_ckpt),
    }
    (args.out_dir / "b4_run_report.json").write_text(json.dumps(run_report, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train (clf)")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val (clf)")
        axes[0].set_title(f"B4 AE-32 Classifier Loss (AE recon={best_ae_loss:.5f})")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (MLSM)")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="teal", marker="o", ms=4)
        axes[1].axhline(0.49, ls="--", color="steelblue", alpha=0.5, label="B1 PCA baseline")
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("B4 Validation Macro-F1"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "b4_train_curves.png", dpi=160); plt.close(fig)

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
        ax2.set_title(f"B4 Per-class F1 (AE bottleneck={args.bottleneck_dim})")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "b4_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"  Saved: {args.fig_dir / 'b4_train_curves.png'}")
        print(f"  Saved: {args.fig_dir / 'b4_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  B4 complete.")
    print(f"   Report    : {args.out_dir / 'b4_run_report.json'}")
    print(f"   AE ckpt   : {ae_ckpt}")
    print(f"   Clf ckpt  : {best_ckpt}")


if __name__ == "__main__":
    main()
