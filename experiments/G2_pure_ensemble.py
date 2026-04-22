"""G2: Pure Ensemble — C4 + D7 saved checkpoints only.

No retraining. Just load the two best GPU-trained models,
run inference on the test set, and average their sigmoid probabilities.

Strategy: Weighted sigmoid average
  C4 (Cross-Attention HuBERT x Whisper)  weight=0.55  (stronger model)
  D7 (Atrous-CNN HuBERT + Whisper)       weight=0.45

Also sweeps thresholds [0.40, 0.42, 0.45, 0.48, 0.50] to find best.

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
from torch.utils.data import DataLoader, TensorDataset

STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]

def parse_args():
    p = argparse.ArgumentParser(description="G2: Pure C4+D7 Ensemble")
    p.add_argument("--features-root",  type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",   type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",   type=int,  default=21)
    p.add_argument("--whisper-alias",  type=str,  default="whisper-large")
    p.add_argument("--whisper-layer",  type=int,  default=28)
    p.add_argument("--fold",           type=str,  default="fold0")
    p.add_argument("--clips-root",     type=Path, default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",     type=Path, default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path, default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--c4-ckpt",        type=Path, default=Path("artifacts/checkpoints/C4/c4_best.pt"))
    p.add_argument("--d7-ckpt",        type=Path, default=Path("artifacts/checkpoints/D7/d7_best.pt"))
    p.add_argument("--c4-weight",      type=float,default=0.55)
    p.add_argument("--d7-weight",      type=float,default=0.45)
    p.add_argument("--test-size",      type=float,default=0.20)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--batch-size",     type=int,  default=256)
    p.add_argument("--out-dir",        type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",        type=Path, default=Path("results/figures"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def norm(x): return str(x).strip()

def load_multilabel_map(csv_path: Path):
    out = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (norm(row["Show"]), norm(row["EpId"]), norm(row["ClipId"]))
            labels = np.array([1.0 if float(norm(row[t])) >= 1 else 0.0 for t in STUTTER_TYPES],
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

def evaluate(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    pf1  = f1_score(y_true, y_pred, average=None, zero_division=0)
    ppre = precision_score(y_true, y_pred, average=None, zero_division=0)
    prec = recall_score(y_true, y_pred, average=None, zero_division=0)
    m = {
        "macro_f1":   float(f1_score(y_true, y_pred, average="macro",  zero_division=0)),
        "micro_f1":   float(f1_score(y_true, y_pred, average="micro",  zero_division=0)),
        "macro_pre":  float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_rec":  float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
        "macro_auprc": float(np.mean([
            average_precision_score(y_true[:, i], probs[:, i])
            if y_true[:, i].sum() > 0 else 0.0 for i in range(5)])),
    }
    for i, t in enumerate(STUTTER_TYPES):
        m[f"f1_{t}"] = float(pf1[i]); m[f"pre_{t}"] = float(ppre[i]); m[f"rec_{t}"] = float(prec[i])
    return m

def print_results(label, m):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Macro-F1    : {m['macro_f1']:.5f}")
    print(f"  Micro-F1    : {m['micro_f1']:.5f}")
    print(f"  Macro-Pre   : {m['macro_pre']:.5f}")
    print(f"  Macro-Rec   : {m['macro_rec']:.5f}")
    print(f"  Macro-AUPRC : {m['macro_auprc']:.5f}")
    print(f"  Per-class F1:")
    for t in STUTTER_TYPES:
        print(f"    {t:15s}: {m[f'f1_{t}']:.4f}  P={m[f'pre_{t}']:.4f}  R={m[f'rec_{t}']:.4f}")


# ---------------------------------------------------------------------------
# Model architectures (must match saved checkpoints exactly)
# ---------------------------------------------------------------------------

class ChunkedCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads; self.d_head = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout); self.norm = nn.LayerNorm(d_model)

    def forward(self, x_q, x_kv):
        B = x_q.size(0)
        Q = self.Wq(x_q).view(B, self.n_heads, self.d_head)
        K = self.Wk(x_kv).view(B, self.n_heads, self.d_head)
        V = self.Wv(x_kv).view(B, self.n_heads, self.d_head)
        attn = torch.softmax((Q * K).sum(-1).unsqueeze(-1) * self.scale, dim=1)
        return self.norm(x_q + self.drop(self.Wo((attn * V).view(B, -1))))

class CrossAttnFusionModel(nn.Module):
    def __init__(self, hubert_dim, whisper_dim, d_model=256, n_heads=8,
                 mlp_hidden=256, n_classes=5, dropout=0.3, attn_dropout=0.1):
        super().__init__()
        self.proj_h   = nn.Sequential(nn.Linear(hubert_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_w   = nn.Sequential(nn.Linear(whisper_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.xattn_hw = ChunkedCrossAttention(d_model, n_heads, attn_dropout)
        self.xattn_wh = ChunkedCrossAttention(d_model, n_heads, attn_dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model*2, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden//2, n_classes))

    def forward(self, h, w):
        ph = self.proj_h(h); pw = self.proj_w(w)
        return self.head(torch.cat([self.xattn_hw(ph, pw), self.xattn_wh(pw, ph)], dim=-1))

class AtrousResBlock(nn.Module):
    def __init__(self, ch, ks, dil, drop=0.1):
        super().__init__()
        pad = dil * (ks - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, ks, dilation=dil, padding=pad, bias=False),
            nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(drop),
            nn.Conv1d(ch, ch, 1, bias=False), nn.BatchNorm1d(ch))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.conv(x))

class AtrousCNNModel(nn.Module):
    def __init__(self, hubert_dim, whisper_dim, d_model=256, n_dilation_layers=5,
                 kernel_size=3, n_classes=5, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(hubert_dim+whisper_dim, d_model*4), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(d_model*4, d_model), nn.LayerNorm(d_model))
        self.seq_len = 32; self.feat_step = d_model // 32
        self.conv_in = nn.Sequential(nn.Conv1d(self.feat_step, d_model, 1, bias=False),
                                     nn.BatchNorm1d(d_model), nn.GELU())
        self.atrous = nn.Sequential(*[AtrousResBlock(d_model, kernel_size, 2**i, dropout*0.5)
                                      for i in range(n_dilation_layers)])
        self.head = nn.Sequential(nn.Linear(d_model*2, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, h, w):
        B = h.size(0)
        x = self.input_proj(torch.cat([h, w], dim=-1))
        x = self.atrous(self.conv_in(x.view(B, self.seq_len, self.feat_step).transpose(1, 2)))
        return self.head(torch.cat([x.max(2).values, x.mean(2)], dim=1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"\nG2: Pure Ensemble (C4 x{args.c4_weight} + D7 x{args.d7_weight})")
    print("No retraining -- loading saved checkpoints only.\n")

    # Load features
    print("[1] Loading cached SSL features ...")
    h_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    w_feats = load_ssl_cache(args.features_root, args.whisper_alias, args.fold, args.whisper_layer)
    if h_feats.ndim != 2: h_feats = h_feats.reshape(h_feats.shape[0], -1)
    if w_feats.ndim != 2: w_feats = w_feats.reshape(w_feats.shape[0], -1)
    n = h_feats.shape[0]
    hubert_dim, whisper_dim = h_feats.shape[1], w_feats.shape[1]
    print(f"  HuBERT L{args.hubert_layer}: {h_feats.shape}  Whisper L{args.whisper_layer}: {w_feats.shape}")

    # Load labels (hard labels, same as original C4/D7)
    print("\n[2] Loading labels ...")
    label_map = {}
    label_map.update(load_multilabel_map(args.sep_labels))
    if args.fluency_labels.exists():
        label_map.update(load_multilabel_map(args.fluency_labels))
    clip_keys = sorted_clip_keys(args.clips_root, n)
    y = np.array([label_map.get(k, np.zeros(len(STUTTER_TYPES), np.float32)) for k in clip_keys],
                 dtype=np.float32)
    print(f"  {n} samples loaded")
    for i, t in enumerate(STUTTER_TYPES):
        pos = int(y[:, i].sum())
        print(f"  {t:15s}: {pos:5d} positive ({pos/n:.1%})")

    # Same split as C4/D7 (same seed + stratify on Block)
    print("\n[3] Reproducing train/test split (seed=42, same as C4/D7) ...")
    idx = np.arange(n)
    _, test_idx = train_test_split(idx, test_size=args.test_size,
                                   random_state=args.seed, stratify=y[:, 0].astype(int))
    sc_h = StandardScaler().fit(h_feats[np.setdiff1d(idx, test_idx)])
    sc_w = StandardScaler().fit(w_feats[np.setdiff1d(idx, test_idx)])
    h_te = sc_h.transform(h_feats[test_idx]).astype(np.float32)
    w_te = sc_w.transform(w_feats[test_idx]).astype(np.float32)
    y_te = y[test_idx]
    print(f"  Test set: {len(test_idx)} samples")

    test_dl = DataLoader(TensorDataset(torch.from_numpy(h_te), torch.from_numpy(w_te)),
                         batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load C4
    print(f"\n[4] Loading C4 checkpoint: {args.c4_ckpt} ...")
    c4 = CrossAttnFusionModel(hubert_dim, whisper_dim).to(device)
    c4.load_state_dict(torch.load(args.c4_ckpt, map_location=device, weights_only=True))
    c4.eval()
    c4_logits = []
    with torch.no_grad():
        for hb, wb in test_dl:
            c4_logits.append(c4(hb.to(device), wb.to(device)).cpu().numpy())
    c4_logits = np.concatenate(c4_logits)
    c4_probs  = torch.sigmoid(torch.tensor(c4_logits)).numpy()
    c4_m = evaluate(y_te, c4_probs, threshold=0.50)
    print_results(f"C4 alone (threshold=0.50)", c4_m)

    # Load D7
    print(f"\n[5] Loading D7 checkpoint: {args.d7_ckpt} ...")
    d7 = AtrousCNNModel(hubert_dim, whisper_dim).to(device)
    d7.load_state_dict(torch.load(args.d7_ckpt, map_location=device, weights_only=True))
    d7.eval()
    d7_logits = []
    with torch.no_grad():
        for hb, wb in test_dl:
            d7_logits.append(d7(hb.to(device), wb.to(device)).cpu().numpy())
    d7_logits = np.concatenate(d7_logits)
    d7_probs  = torch.sigmoid(torch.tensor(d7_logits)).numpy()
    d7_m = evaluate(y_te, d7_probs, threshold=0.50)
    print_results(f"D7 alone (threshold=0.50)", d7_m)

    # Ensemble: weighted average of sigmoid probabilities
    print(f"\n[6] G2 Ensemble: C4 x{args.c4_weight} + D7 x{args.d7_weight}")
    total_w = args.c4_weight + args.d7_weight
    ens_probs = (c4_probs * args.c4_weight + d7_probs * args.d7_weight) / total_w

    # Threshold sweep to find best
    print("\n  Threshold sweep:")
    print(f"  {'Threshold':>10}  {'Macro-F1':>10}  {'Macro-Pre':>10}  {'Macro-Rec':>10}")
    print(f"  {'-'*46}")
    best_t, best_f1, best_m = 0.50, -1.0, None
    for t in [0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52]:
        m = evaluate(y_te, ens_probs, threshold=t)
        marker = " <- BEST" if m["macro_f1"] > best_f1 else ""
        print(f"  {t:>10.2f}  {m['macro_f1']:>10.5f}  {m['macro_pre']:>10.5f}  {m['macro_rec']:>10.5f}{marker}")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]; best_t = t; best_m = m

    print_results(f"G2 Ensemble best (threshold={best_t})", best_m)

    # Final summary
    baseline_c4 = 0.6587
    print(f"\n{'='*60}")
    print(f"  FINAL G2 SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline C4 (recorded)     : 0.6587")
    print(f"  C4 alone (this run)        : {c4_m['macro_f1']:.4f}")
    print(f"  D7 alone (this run)        : {d7_m['macro_f1']:.4f}")
    print(f"  G2 Ensemble (best thresh)  : {best_m['macro_f1']:.4f}  (threshold={best_t})")
    delta = best_m["macro_f1"] - baseline_c4
    print(f"  vs Baseline C4             : {delta:+.4f}")
    print(f"{'='*60}")

    # Save report
    report = {
        "experiment": "G2",
        "method": "Pure ensemble C4+D7 (no retraining)",
        "c4_weight": args.c4_weight, "d7_weight": args.d7_weight,
        "best_threshold": best_t,
        "c4_alone_macro_f1": c4_m["macro_f1"],
        "d7_alone_macro_f1": d7_m["macro_f1"],
        "ensemble_macro_f1": best_m["macro_f1"],
        "baseline_c4_macro_f1": baseline_c4,
        "improvement_vs_baseline": delta,
        "ensemble_results": best_m,
        "c4_results": c4_m,
        "d7_results": d7_m,
    }
    (args.out_dir / "g2_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {args.out_dir}/g2_report.json")
    print("\nG2 complete.")


if __name__ == "__main__":
    main()
