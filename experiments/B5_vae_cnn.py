"""B5: VAE (Variational Autoencoder) latent space compression benchmark.

Probabilistic bottleneck details:
  - Encoder outputs μ and log-σ² (mean and log-variance) of the latent Gaussian
  - Reparameterisation trick: z = μ + σ * ε, ε ~ N(0,1)
  - Loss = MSE reconstruction + β * KL divergence
    KL(q(z|x) || p(z)) = -½ Σ(1 + log-σ² - μ² - σ²)
  - β annealing: starts at 0, linearly ramps to β_max over first 20 epochs
    (prevents posterior collapse in early training)

Two-stage pipeline:
  Stage 1: VAE pre-training (unsupervised, recon + KL)
  Stage 2: CNN-1D classifier on frozen μ (mean of latent posterior)

The probabilistic latent space imposes a structured N(0,1) prior which may
disentangle stutter-type factors.

Run command:
    python experiments/B5_vae_cnn.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --latent-dim 32 \
        --beta-max 0.5 \
        --vae-epochs 60 \
        --vae-lr 5e-4 \
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
    p = argparse.ArgumentParser(description="B5: VAE latent space + CNN-1D, multi-label")
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
    # VAE params
    p.add_argument("--latent-dim",    type=int,  default=32)
    p.add_argument("--vae-hidden",    type=str,  default="512,256,128")
    p.add_argument("--vae-dropout",   type=float,default=0.1)
    p.add_argument("--beta-max",      type=float,default=0.5,
                   help="Max KL weight (β-VAE). Anneals from 0 to β-max over first 20 epochs.")
    p.add_argument("--beta-warmup",   type=int,  default=20,
                   help="Epochs over which β ramps from 0 to beta-max")
    p.add_argument("--vae-epochs",    type=int,  default=60)
    p.add_argument("--vae-lr",        type=float,default=5e-4)
    # CNN classifier params
    p.add_argument("--cnn-channels",  type=str,  default="64,128,256")
    p.add_argument("--clf-dropout",   type=float,default=0.3)
    p.add_argument("--clf-epochs",    type=int,  default=30)
    p.add_argument("--clf-lr",        type=float,default=3e-4)
    # Shared
    p.add_argument("--batch-size",    type=int,  default=256)
    p.add_argument("--weight-decay",  type=float,default=1e-4)
    p.add_argument("--test-size",     type=float,default=0.20)
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--threshold",     type=float,default=0.5)
    p.add_argument("--out-dir",  type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",  type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/B5"))
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
    raise FileNotFoundError(f"layer_{layer}.npy not found")


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    class_freq = np.where(y.mean(axis=0) == 0, 1.0, y.mean(axis=0))
    inv_freq = 1.0 / class_freq
    weights  = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        pm = y[i] > 0
        weights[i] = inv_freq[pm].max() if pm.any() else inv_freq.min()
    return weights / weights.min()


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """
    β-VAE: encoder outputs (μ, log-σ²), reparameterise to sample z,
    decoder reconstructs input. KL(q||p) regulates the latent space.
    β annealing prevents posterior collapse in early training.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 latent_dim: int, dropout: float) -> None:
        super().__init__()
        # Shared encoder backbone
        enc_layers: List[nn.Module] = []
        d_in = input_dim
        for d_out in hidden_dims:
            enc_layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out),
                           nn.GELU(), nn.Dropout(dropout)]
            d_in = d_out
        self.encoder_backbone = nn.Sequential(*enc_layers)
        # μ and log-σ² heads
        self.fc_mu     = nn.Linear(d_in, latent_dim)
        self.fc_logvar = nn.Linear(d_in, latent_dim)
        # Decoder
        dec_layers: List[nn.Module] = []
        d_in = latent_dim
        for d_out in reversed(hidden_dims):
            dec_layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out),
                           nn.GELU(), nn.Dropout(dropout)]
            d_in = d_out
        dec_layers += [nn.Linear(d_in, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu   # at eval, use mean only

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decode(z)
        return recon, mu, logvar

    def vae_loss(self, x: torch.Tensor, recon: torch.Tensor,
                 mu: torch.Tensor, logvar: torch.Tensor,
                 beta: float) -> Tuple[torch.Tensor, float, float]:
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total      = recon_loss + beta * kl_loss
        return total, recon_loss.item(), kl_loss.item()


# ---------------------------------------------------------------------------
# CNN-1D Classifier (identical to B1/B2/B3/B4)
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
    # 2. Split + StandardScaler
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y[:, 0].astype(int)
    )
    sc = StandardScaler()
    x_tr_raw = sc.fit_transform(ssl_feats[train_idx]).astype(np.float32)
    x_te_raw = sc.transform(ssl_feats[test_idx]).astype(np.float32)

    # ------------------------------------------------------------------
    # 3. VAE pre-training (Stage 1)
    # ------------------------------------------------------------------
    hidden_dims = [int(h) for h in args.vae_hidden.split(",")]
    in_dim = ssl_feats.shape[1]
    vae = VAE(in_dim, hidden_dims, args.latent_dim, args.vae_dropout).to(device)
    vae_params = sum(p.numel() for p in vae.parameters())


    vae_dl = DataLoader(TensorDataset(torch.from_numpy(x_tr_raw)),
                        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    vae_opt   = torch.optim.AdamW(vae.parameters(), lr=args.vae_lr, weight_decay=1e-5)
    vae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(vae_opt, T_max=args.vae_epochs,
                                                             eta_min=args.vae_lr * 0.05)
    best_vae_loss = float("inf")
    vae_ckpt = args.ckpt_dir / "b5_vae.pt"

    for epoch in range(1, args.vae_epochs + 1):
        # Beta annealing schedule (0 → beta_max over warmup epochs)
        beta = min(args.beta_max, args.beta_max * epoch / args.beta_warmup)
        vae.train()
        tr_loss = tr_recon = tr_kl = 0.0
        for (xb,) in vae_dl:
            xb = xb.to(device)
            vae_opt.zero_grad()
            recon, mu, logvar = vae(xb)
            loss, rl, kl = vae.vae_loss(xb, recon, mu, logvar, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            vae_opt.step()
            bs = xb.size(0)
            tr_loss  += loss.item() * bs
            tr_recon += rl * bs
            tr_kl    += kl * bs
        n_tr = len(train_idx)
        tr_loss /= n_tr; tr_recon /= n_tr; tr_kl /= n_tr
        vae_sched.step()
        if epoch % 10 == 0:
            print(f"  vae_ep={epoch:02d}  total={tr_loss:.5f}  "
                  f"recon={tr_recon:.5f}  kl={tr_kl:.5f}  β={beta:.3f}")
        if tr_loss < best_vae_loss:
            best_vae_loss = tr_loss
            torch.save(vae.state_dict(), vae_ckpt)


    # ------------------------------------------------------------------
    # 4. Extract μ (mean of posterior = deterministic latent)
    # ------------------------------------------------------------------
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae.eval()

    def extract_mu(x_np: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for s in range(0, len(x_np), 512):
                mu, _ = vae.encode(torch.from_numpy(x_np[s:s+512]).to(device))
                out.append(mu.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    z_tr = extract_mu(x_tr_raw)
    z_te = extract_mu(x_te_raw)

    z_sc = StandardScaler()
    z_tr = z_sc.fit_transform(z_tr).astype(np.float32)
    z_te = z_sc.transform(z_te).astype(np.float32)

    y_tr, y_te = y[train_idx], y[test_idx]

    # ------------------------------------------------------------------
    # 5. CNN-1D Classifier (Stage 2)
    # ------------------------------------------------------------------

    channels = [int(c) for c in args.cnn_channels.split(",")]
    model  = CNN1DMultiLabel(in_dim=args.latent_dim, channels=channels,
                              num_classes=len(STUTTER_TYPES), dropout=args.clf_dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

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
    best_ckpt = args.ckpt_dir / "b5_clf_best.pt"

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

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    perclass_csv = args.out_dir / "b5_perclass_results.csv"
    with perclass_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    hist_csv = args.out_dir / "b5_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "B5",
        "title": "β-VAE latent-32 (μ) + CNN-1D (HuBERT-large, probabilistic latent space)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "latent_dim": args.latent_dim,
        "vae_hidden": hidden_dims, "beta_max": args.beta_max,
        "beta_warmup": args.beta_warmup, "vae_params": int(vae_params),
        "best_vae_loss": float(best_vae_loss), "vae_epochs": args.vae_epochs,
        "clf_params": int(n_params), "clf_epochs": args.clf_epochs,
        "loss": f"MSE+β*KL (β_max={args.beta_max}) + MultiLabelSoftMarginLoss",
        "best_clf_epoch": int(best_ep),
        "macro_f1": test_m["macro_f1"], "micro_f1": test_m["micro_f1"],
        "macro_precision": test_m["macro_pre"], "macro_recall": test_m["macro_rec"],
        "macro_auprc": test_m["macro_auprc"],
        "per_class": {t: {"f1": test_m[f"f1_{t}"], "precision": test_m[f"pre_{t}"],
                          "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]}
                      for t in STUTTER_TYPES},
        "vae_checkpoint": str(vae_ckpt), "clf_checkpoint": str(best_ckpt),
    }
    (args.out_dir / "b5_run_report.json").write_text(json.dumps(run_report, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train (clf)")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val (clf)")
        axes[0].set_title(f"B5 VAE-32 Classifier Loss (β_max={args.beta_max})")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (MLSM)")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="mediumvioletred", marker="o", ms=4)
        
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("B5 Validation Macro-F1"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "b5_train_curves.png", dpi=160); plt.close(fig)

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
        ax2.set_title(f"B5 Per-class F1 (VAE μ-{args.latent_dim}, β={args.beta_max})")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "b5_perclass_f1.png", dpi=160); plt.close(fig2)
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")


if __name__ == "__main__":
    main()
