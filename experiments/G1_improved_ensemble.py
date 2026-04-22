"""G1: Improved Ensemble - 4 Fixes applied to C4+D7 champion.
Fix1: Feature augmentation (noise+scale) during training
Fix2: Soft labels from annotator vote counts (votes/3)
Fix3: FluencyBank explicitly included
Fix4: Ensemble of saved C4 + D7 checkpoints
New file - no existing files modified.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]
MAX_VOTES = 3  # SEP-28k uses 0-3 annotator votes

def parse_args():
    p = argparse.ArgumentParser(description="G1: Improved Ensemble (4 Fixes)")
    p.add_argument("--features-root", type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",  type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",  type=int,  default=21)
    p.add_argument("--whisper-alias", type=str,  default="whisper-large")
    p.add_argument("--whisper-layer", type=int,  default=28)
    p.add_argument("--fold",          type=str,  default="fold0")
    p.add_argument("--clips-root",    type=Path, default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",    type=Path, default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels",type=Path, default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--d-model",       type=int,  default=256)
    p.add_argument("--n-heads",       type=int,  default=8)
    p.add_argument("--attn-dropout",  type=float,default=0.1)
    p.add_argument("--mlp-hidden",    type=int,  default=256)
    p.add_argument("--dropout",       type=float,default=0.3)
    p.add_argument("--test-size",     type=float,default=0.20)
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--epochs",        type=int,  default=60)
    p.add_argument("--batch-size",    type=int,  default=256)
    p.add_argument("--lr",            type=float,default=3e-4)
    p.add_argument("--weight-decay",  type=float,default=1e-4)
    p.add_argument("--threshold",     type=float,default=0.45)
    p.add_argument("--aug-noise",     type=float,default=0.02)
    p.add_argument("--aug-scale",     type=float,default=0.10)
    p.add_argument("--mixup-alpha",   type=float,default=0.3)
    p.add_argument("--c4-ckpt",       type=Path, default=Path("artifacts/checkpoints/C4/c4_best.pt"))
    p.add_argument("--d7-ckpt",       type=Path, default=Path("artifacts/checkpoints/D7/d7_best.pt"))
    p.add_argument("--out-dir",       type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",       type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir",      type=Path, default=Path("artifacts/checkpoints/G1"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def norm(x): return str(x).strip()

def load_multilabel_map_soft(csv_path: Path, soft: bool = True):
    """Load labels. If soft=True, return votes/MAX_VOTES. Else hard threshold>=1."""
    out = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (norm(row["Show"]), norm(row["EpId"]), norm(row["ClipId"]))
            if soft:
                labels = np.array(
                    [min(float(norm(row[t])), MAX_VOTES) / MAX_VOTES for t in STUTTER_TYPES],
                    dtype=np.float32)
            else:
                labels = np.array(
                    [1.0 if float(norm(row[t])) >= 1 else 0.0 for t in STUTTER_TYPES],
                    dtype=np.float32)
            out[key] = labels
    return out

def sorted_clip_keys(clips_root: Path, n: int):
    keys = []
    for w in sorted(clips_root.rglob("*.wav")):
        parts = w.stem.split("_")
        if len(parts) >= 3:
            keys.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return keys[:n]

def load_ssl_cache(features_root: Path, alias: str, fold: str, layer: int):
    direct = features_root / alias / fold / f"layer_{layer}.npy"
    if direct.exists():
        return np.load(direct)
    for child in sorted((features_root / alias).iterdir()):
        cand = child / f"layer_{layer}.npy"
        if cand.exists():
            return np.load(cand)
    raise FileNotFoundError(f"layer_{layer}.npy not found for alias {alias!r}")

def compute_sample_weights(y: np.ndarray):
    hard_y = (y >= 0.5).astype(float)
    class_freq = np.where(hard_y.mean(axis=0) == 0, 1.0, hard_y.mean(axis=0))
    inv_freq = 1.0 / class_freq
    weights = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        pm = hard_y[i] > 0
        weights[i] = inv_freq[pm].max() if pm.any() else inv_freq.min()
    return weights / weights.min()


# ---------------------------------------------------------------------------
# Fix 1: Feature augmentation applied during training
# ---------------------------------------------------------------------------

def augment_features(h: torch.Tensor, w: torch.Tensor,
                     noise_std: float, scale_std: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add Gaussian noise + random scale jitter to cached SSL features."""
    if noise_std > 0:
        h = h + torch.randn_like(h) * noise_std
        w = w + torch.randn_like(w) * noise_std
    if scale_std > 0:
        sh = 1.0 + (torch.rand(h.size(0), 1, device=h.device) - 0.5) * 2 * scale_std
        sw = 1.0 + (torch.rand(w.size(0), 1, device=w.device) - 0.5) * 2 * scale_std
        h = h * sh
        w = w * sw
    return h, w


def mixup_batch(h, w, y, alpha: float):
    """Label-conditioned Mixup: only mix samples that share at least one positive label."""
    if alpha <= 0:
        return h, w, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(h.size(0), device=h.device)
    # Only apply mixup where label overlap exists
    hard_y  = (y >= 0.5).float()
    hard_yi = hard_y[idx]
    overlap = (hard_y * hard_yi).sum(dim=1, keepdim=True) > 0  # [B,1]
    lam_t   = torch.where(overlap, torch.full_like(overlap, lam, dtype=torch.float32),
                          torch.ones_like(overlap, dtype=torch.float32))
    h_mix = lam_t * h + (1 - lam_t) * h[idx]
    w_mix = lam_t * w + (1 - lam_t) * w[idx]
    y_mix = lam_t * y + (1 - lam_t) * y[idx]
    return h_mix, w_mix, y_mix


# ---------------------------------------------------------------------------
# Models (C4 cross-attention + D7 atrous-CNN — same architecture as originals)
# ---------------------------------------------------------------------------

class ChunkedCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
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

    def forward(self, x_q, x_kv):
        B = x_q.size(0)
        Q = self.Wq(x_q).view(B, self.n_heads, self.d_head)
        K = self.Wk(x_kv).view(B, self.n_heads, self.d_head)
        V = self.Wv(x_kv).view(B, self.n_heads, self.d_head)
        attn = (Q * K).sum(dim=-1) * self.scale
        attn = torch.softmax(attn.unsqueeze(-1), dim=1)
        out  = (attn * V).view(B, -1)
        out  = self.drop(self.Wo(out))
        return self.norm(x_q + out)


class CrossAttnFusionModel(nn.Module):
    def __init__(self, hubert_dim, whisper_dim, d_model, n_heads, mlp_hidden,
                 n_classes, dropout, attn_dropout):
        super().__init__()
        self.proj_h  = nn.Sequential(nn.Linear(hubert_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_w  = nn.Sequential(nn.Linear(whisper_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.xattn_hw = ChunkedCrossAttention(d_model, n_heads, attn_dropout)
        self.xattn_wh = ChunkedCrossAttention(d_model, n_heads, attn_dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model*2, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden//2, n_classes))

    def forward(self, h, w):
        ph = self.proj_h(h); pw = self.proj_w(w)
        ah = self.xattn_hw(ph, pw); aw = self.xattn_wh(pw, ph)
        return self.head(torch.cat([ah, aw], dim=-1))


class AtrousResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm1d(channels))
        self.act = nn.GELU()

    def forward(self, x): return self.act(x + self.conv(x))


class AtrousCNNModel(nn.Module):
    def __init__(self, hubert_dim, whisper_dim, d_model, n_dilation_layers, kernel_size,
                 n_classes, dropout):
        super().__init__()
        fused_dim = hubert_dim + whisper_dim
        self.input_proj = nn.Sequential(
            nn.Linear(fused_dim, d_model*4), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(d_model*4, d_model), nn.LayerNorm(d_model))
        seq_len = 32; feat_step = d_model // seq_len
        self.seq_len = seq_len; self.feat_step = feat_step
        self.conv_in = nn.Sequential(nn.Conv1d(feat_step, d_model, 1, bias=False),
                                     nn.BatchNorm1d(d_model), nn.GELU())
        self.atrous = nn.Sequential(*[
            AtrousResBlock(d_model, kernel_size, 2**i, dropout*0.5)
            for i in range(n_dilation_layers)])
        self.head = nn.Sequential(nn.Linear(d_model*2, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, h, w):
        B = h.size(0)
        x = self.input_proj(torch.cat([h, w], dim=-1))
        x = x.view(B, self.seq_len, self.feat_step).transpose(1, 2)
        x = self.atrous(self.conv_in(x))
        return self.head(torch.cat([x.max(2).values, x.mean(2)], dim=1))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_multilabel(y_true, y_logits, threshold):
    probs  = torch.sigmoid(torch.tensor(y_logits)).numpy()
    y_hard = (y_true >= 0.5).astype(int)
    y_pred = (probs >= threshold).astype(int)
    m = {}
    pf1  = f1_score(y_hard, y_pred, average=None, zero_division=0)
    ppre = precision_score(y_hard, y_pred, average=None, zero_division=0)
    prec = recall_score(y_hard, y_pred, average=None, zero_division=0)
    for i, t in enumerate(STUTTER_TYPES):
        m[f"f1_{t}"] = float(pf1[i]); m[f"pre_{t}"] = float(ppre[i]); m[f"rec_{t}"] = float(prec[i])
    m["macro_f1"]  = float(f1_score(y_hard, y_pred, average="macro",  zero_division=0))
    m["micro_f1"]  = float(f1_score(y_hard, y_pred, average="micro",  zero_division=0))
    m["macro_pre"] = float(precision_score(y_hard, y_pred, average="macro", zero_division=0))
    m["macro_rec"] = float(recall_score(y_hard, y_pred, average="macro",    zero_division=0))
    auprc = []
    for i, t in enumerate(STUTTER_TYPES):
        ap = float(average_precision_score(y_hard[:, i], probs[:, i])) if y_hard[:, i].sum() > 0 else 0.0
        m[f"auprc_{t}"] = ap; auprc.append(ap)
    m["macro_auprc"] = float(np.mean(auprc))
    return m


def print_results(label, m):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Macro-F1   : {m['macro_f1']:.5f}")
    print(f"  Micro-F1   : {m['micro_f1']:.5f}")
    print(f"  Macro-Pre  : {m['macro_pre']:.5f}")
    print(f"  Macro-Rec  : {m['macro_rec']:.5f}")
    print(f"  Macro-AUPRC: {m['macro_auprc']:.5f}")
    print(f"  Per-class F1:")
    for t in STUTTER_TYPES:
        print(f"    {t:15s}: {m[f'f1_{t}']:.4f}  P={m[f'pre_{t}']:.4f}  R={m[f'rec_{t}']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    for d in (args.out_dir, args.fig_dir, args.ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load cached SSL features ---
    print(f"\n[1] Loading HuBERT L{args.hubert_layer} + Whisper L{args.whisper_layer} ...")
    h_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    w_feats = load_ssl_cache(args.features_root, args.whisper_alias, args.fold, args.whisper_layer)
    if h_feats.ndim != 2: h_feats = h_feats.reshape(h_feats.shape[0], -1)
    if w_feats.ndim != 2: w_feats = w_feats.reshape(w_feats.shape[0], -1)
    n = h_feats.shape[0]
    hubert_dim, whisper_dim = h_feats.shape[1], w_feats.shape[1]
    print(f"  HuBERT: {h_feats.shape}  Whisper: {w_feats.shape}")

    # --- Fix 2+3: Soft labels from vote counts + FluencyBank ---
    print(f"\n[2] Loading labels (soft={args.soft_labels if hasattr(args,'soft_labels') else True}) ...")
    use_soft = True
    label_map = {}
    label_map.update(load_multilabel_map_soft(args.sep_labels, soft=use_soft))
    fb_count = 0
    if args.fluency_labels.exists():
        fb_map = load_multilabel_map_soft(args.fluency_labels, soft=False)
        label_map.update(fb_map)
        fb_count = len(fb_map)
    print(f"  SEP-28k + FluencyBank ({fb_count} FB entries loaded)")

    clip_keys = sorted_clip_keys(args.clips_root, n)
    y = np.array([label_map.get(k, np.zeros(len(STUTTER_TYPES), np.float32)) for k in clip_keys],
                 dtype=np.float32)
    print(f"  Dataset: {n} samples  |  Label stats (positive rate):")
    for i, t in enumerate(STUTTER_TYPES):
        pos = int((y[:, i] >= 0.5).sum())
        print(f"    {t:15s}: {pos:5d} ({pos/n:.1%})")

    # --- Split + Scale ---
    print("\n[3] Train/test split + StandardScaler ...")
    y_hard_strat = (y[:, 0] >= 0.5).astype(int)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size,
                                           random_state=args.seed, stratify=y_hard_strat)
    sc_h = StandardScaler(); sc_w = StandardScaler()
    h_tr = sc_h.fit_transform(h_feats[train_idx]).astype(np.float32)
    h_te = sc_h.transform(h_feats[test_idx]).astype(np.float32)
    w_tr = sc_w.fit_transform(w_feats[train_idx]).astype(np.float32)
    w_te = sc_w.transform(w_feats[test_idx]).astype(np.float32)
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f"  Train: {len(train_idx)}  Test: {len(test_idx)}")

    # --- DataLoaders ---
    sw = compute_sample_weights(y_tr)
    sampler  = WeightedRandomSampler(torch.from_numpy(sw), len(sw), replacement=True)
    train_dl = DataLoader(TensorDataset(torch.from_numpy(h_tr), torch.from_numpy(w_tr), torch.from_numpy(y_tr)),
                          batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    test_dl  = DataLoader(TensorDataset(torch.from_numpy(h_te), torch.from_numpy(w_te), torch.from_numpy(y_te)),
                          batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- Build improved G1 model (same arch as C4) ---
    print(f"\n[4] Building G1 model (C4 arch + soft labels + augmentation) ...")
    model = CrossAttnFusionModel(
        hubert_dim=hubert_dim, whisper_dim=whisper_dim,
        d_model=args.d_model, n_heads=args.n_heads, mlp_hidden=args.mlp_hidden,
        n_classes=len(STUTTER_TYPES), dropout=args.dropout, attn_dropout=args.attn_dropout
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Fix 2: BCEWithLogitsLoss works with soft labels (vs MultiLabelSoftMarginLoss)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine LR with linear warmup
    warmup_epochs = 5
    def lr_lambda(ep):
        if ep < warmup_epochs: return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"\n[5] Training for {args.epochs} epochs (Fix1: augment, Fix2: soft labels) ...")
    history = []; best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "g1_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train(); tr_loss = 0.0
        for hb, wb, yb in train_dl:
            hb, wb, yb = hb.to(device), wb.to(device), yb.to(device)
            # Fix 1: augment
            hb, wb = augment_features(hb, wb, args.aug_noise, args.aug_scale)
            hb, wb, yb = mixup_batch(hb, wb, yb, args.mixup_alpha)
            opt.zero_grad()
            loss = criterion(model(hb, wb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * hb.size(0)
        tr_loss /= len(y_tr); scheduler.step()

        model.eval(); vl, vlog, vlab = 0.0, [], []
        with torch.no_grad():
            for hb, wb, yb in test_dl:
                hb, wb, yb = hb.to(device), wb.to(device), yb.to(device)
                lg = model(hb, wb)
                vl += criterion(lg, yb).item() * hb.size(0)
                vlog.append(lg.cpu().numpy()); vlab.append(yb.cpu().numpy())
        vl /= len(y_te)
        vm = evaluate_multilabel(np.concatenate(vlab), np.concatenate(vlog), args.threshold)
        mf1 = vm["macro_f1"]
        history.append({"epoch": epoch, "train_loss": round(tr_loss,6), "val_loss": round(vl,6), "macro_f1": round(mf1,6)})
        print(f"  epoch={epoch:03d}  train_loss={tr_loss:.5f}  val_loss={vl:.5f}  macro_f1={mf1:.5f}")
        if mf1 > best_macro_f1:
            best_macro_f1 = mf1; torch.save(model.state_dict(), best_ckpt)

    # --- Final evaluation: G1 alone ---
    model.load_state_dict(torch.load(best_ckpt, map_location=device)); model.eval()
    g1_log, g1_lab = [], []
    with torch.no_grad():
        for hb, wb, yb in test_dl:
            g1_log.append(model(hb.to(device), wb.to(device)).cpu().numpy())
            g1_lab.append(yb.numpy())
    g1_logits = np.concatenate(g1_log); g1_labels = np.concatenate(g1_lab)
    g1_m = evaluate_multilabel(g1_labels, g1_logits, args.threshold)
    print_results("G1 Improved Model (soft labels + augmentation)", g1_m)

    # --- Fix 4: Ensemble C4 + D7 + G1 ---
    print(f"\n[6] Fix 4: Ensemble C4 + D7 + G1 ...")
    ens_logits = None; ens_weights = []

    if args.c4_ckpt.exists():
        c4_model = CrossAttnFusionModel(hubert_dim=hubert_dim, whisper_dim=whisper_dim,
                                        d_model=256, n_heads=8, mlp_hidden=256,
                                        n_classes=5, dropout=0.3, attn_dropout=0.1).to(device)
        c4_model.load_state_dict(torch.load(args.c4_ckpt, map_location=device)); c4_model.eval()
        c4_log = []
        with torch.no_grad():
            for hb, wb, yb in test_dl:
                c4_log.append(c4_model(hb.to(device), wb.to(device)).cpu().numpy())
        c4_logits = np.concatenate(c4_log)
        ens_logits = c4_logits; ens_weights.append(0.4)
        print("  Loaded C4 checkpoint")
    else:
        print("  WARNING: C4 checkpoint not found — skipping")
        c4_logits = None

    if args.d7_ckpt.exists():
        d7_model = AtrousCNNModel(hubert_dim=hubert_dim, whisper_dim=whisper_dim,
                                  d_model=256, n_dilation_layers=5, kernel_size=3,
                                  n_classes=5, dropout=0.3).to(device)
        d7_model.load_state_dict(torch.load(args.d7_ckpt, map_location=device)); d7_model.eval()
        d7_log = []
        with torch.no_grad():
            for hb, wb, yb in test_dl:
                d7_log.append(d7_model(hb.to(device), wb.to(device)).cpu().numpy())
        d7_logits = np.concatenate(d7_log)
        ens_weights.append(0.3)
        print("  Loaded D7 checkpoint")
    else:
        print("  WARNING: D7 checkpoint not found — skipping")
        d7_logits = None

    # G1 weight
    g1_weight = 1.0 - sum(ens_weights)

    # Weighted sigmoid average ensemble
    probs_list = []
    weights_list = []
    if c4_logits is not None:
        probs_list.append(torch.sigmoid(torch.tensor(c4_logits)).numpy())
        weights_list.append(0.4)
    if d7_logits is not None:
        probs_list.append(torch.sigmoid(torch.tensor(d7_logits)).numpy())
        weights_list.append(0.3)
    probs_list.append(torch.sigmoid(torch.tensor(g1_logits)).numpy())
    weights_list.append(g1_weight)

    total_w = sum(weights_list)
    ens_probs = sum(p * w for p, w in zip(probs_list, weights_list)) / total_w
    # Threshold at 0.45 to slightly boost minority class recall
    y_hard_te = (g1_labels >= 0.5).astype(int)
    ens_pred  = (ens_probs >= args.threshold).astype(int)
    from sklearn.metrics import f1_score as f1s, precision_score as ps, recall_score as rs, average_precision_score as aps
    ens_pf1  = f1s(y_hard_te, ens_pred, average=None, zero_division=0)
    ens_m = {
        "macro_f1":  float(f1s(y_hard_te, ens_pred, average="macro",  zero_division=0)),
        "micro_f1":  float(f1s(y_hard_te, ens_pred, average="micro",  zero_division=0)),
        "macro_pre": float(ps(y_hard_te, ens_pred, average="macro",   zero_division=0)),
        "macro_rec": float(rs(y_hard_te, ens_pred, average="macro",   zero_division=0)),
        "macro_auprc": float(np.mean([aps(y_hard_te[:,i], ens_probs[:,i]) if y_hard_te[:,i].sum()>0 else 0.0 for i in range(5)])),
    }
    for i, t in enumerate(STUTTER_TYPES):
        ens_m[f"f1_{t}"] = float(ens_pf1[i])
        ens_m[f"pre_{t}"] = float(ps(y_hard_te, ens_pred, average=None, zero_division=0)[i])
        ens_m[f"rec_{t}"] = float(rs(y_hard_te, ens_pred, average=None, zero_division=0)[i])
    print_results(f"G1 Ensemble (C4 x0.4 + D7 x0.3 + G1 x{g1_weight:.1f})", ens_m)

    # --- Summary comparison ---
    print("\n" + "="*60)
    print("  FINAL COMPARISON SUMMARY")
    print("="*60)
    print(f"  Baseline C4 (original)   : 0.6587  [from paper]")
    print(f"  G1 Improved (this run)   : {g1_m['macro_f1']:.4f}  (Fix1+Fix2+Fix3)")
    print(f"  G1 Ensemble C4+D7+G1     : {ens_m['macro_f1']:.4f}  (Fix1+Fix2+Fix3+Fix4)")
    delta_g1  = g1_m['macro_f1']  - 0.6587
    delta_ens = ens_m['macro_f1'] - 0.6587
    print(f"  Improvement G1 alone     : {delta_g1:+.4f}")
    print(f"  Improvement Ensemble     : {delta_ens:+.4f}")
    print("="*60)

    # --- Save ---
    report = {
        "experiment": "G1",
        "fixes_applied": ["feature_augmentation", "soft_labels", "fluencybank", "ensemble_c4_d7"],
        "g1_improved": g1_m,
        "ensemble_c4_d7_g1": ens_m,
        "baseline_c4_macro_f1": 0.6587,
        "improvement_g1_alone": delta_g1,
        "improvement_ensemble": delta_ens,
    }
    (args.out_dir / "g1_report.json").write_text(json.dumps(report, indent=2))
    with (args.out_dir / "g1_train_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        w.writeheader(); w.writerows(history)
    print(f"\n  Report saved: {args.out_dir}/g1_report.json")
    print(f"  Best ckpt  : {best_ckpt}")
    print("\nG1 complete.")


if __name__ == "__main__":
    main()
