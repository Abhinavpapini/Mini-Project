"""E5: Graph-Transformer — batch-graph k-NN + Transformer attention (final experiment).

MOTIVATION: Graph methods (E2-GAT) showed relational reasoning helps (0.515 F1).
Transformers (C4 cross-attention) showed attention is powerful (0.659 F1).
E5 combines both: build a k-NN similarity graph over batch samples,
then apply graph-aware Transformer attention using both node features AND
graph structure via adjacency-biased attention.

ARCHITECTURE:
  Inputs: HuBERT-large L21 (1024) + Whisper-large L28 (1280)
  → Concat [2304] → Linear proj → node embeddings (256-dim)

  Batch-level k-NN graph (k=8, cosine similarity, per-batch):
    adj[i,j] = 1 if j in top-k neighbors of i, else 0

  Graph-Transformer layer (Graphormer-inspired):
    QKV attention with adjacency bias:
      attn_logit[i,j] = (Q_i · K_j) / sqrt(d_h) + β * adj[i,j]
    Where β is a learnable scalar bias for connected edges
    → Aggregated node features [B, d_model]

  Classifier MLP: d_model → d_model//2 → 5 (MLSM)

At inference: graph is built over the test BATCH (transductive-style),
consistent with training. k-NN graph changes every batch.

Run command:
    python experiments/E5_graph_transformer.py \
        --hubert-alias hubert-large --hubert-layer 21 \
        --whisper-alias whisper-large --whisper-layer 28 \
        --d-model 256 --n-heads 8 --n-gt-layers 2 \
        --knn-k 8 --adj-bias-init 1.0 \
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
    p = argparse.ArgumentParser(description="E5: Graph-Transformer, multi-label")
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
    # Graph-Transformer params
    p.add_argument("--d-model",         type=int,  default=256)
    p.add_argument("--n-heads",         type=int,  default=8)
    p.add_argument("--n-gt-layers",     type=int,  default=2,
                   help="Number of Graph-Transformer layers")
    p.add_argument("--knn-k",           type=int,  default=8,
                   help="k for batch-level k-NN graph construction")
    p.add_argument("--adj-bias-init",   type=float,default=1.0,
                   help="Initial value of learnable adjacency bias β")
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
    p.add_argument("--ckpt-dir",  type=Path, default=Path("artifacts/checkpoints/E5"))
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


def build_knn_adj(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build binary k-NN adjacency matrix from cosine similarity. [B, B]"""
    x_norm = F.normalize(x, dim=-1)               # [B, D]
    sim    = x_norm @ x_norm.T                     # [B, B]
    sim.fill_diagonal_(-1e9)                       # exclude self
    _, idx = sim.topk(k, dim=-1)                   # [B, k]
    adj    = torch.zeros_like(sim)
    adj.scatter_(-1, idx, 1.0)
    # Symmetrise
    adj = ((adj + adj.T) > 0).float()
    return adj                                     # [B, B]


# ---------------------------------------------------------------------------
# Graph-Transformer Layer
# ---------------------------------------------------------------------------

class GraphTransformerLayer(nn.Module):
    """
    Multi-head self-attention with adjacency bias.
    attn[i,j] += β * adj[i,j]  (β is a learnable parameter per head)
    """
    def __init__(self, d_model: int, n_heads: int,
                 adj_bias_init: float = 1.0, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.Wq      = nn.Linear(d_model, d_model, bias=False)
        self.Wk      = nn.Linear(d_model, d_model, bias=False)
        self.Wv      = nn.Linear(d_model, d_model, bias=False)
        self.Wo      = nn.Linear(d_model, d_model)
        self.drop    = nn.Dropout(dropout)
        # Learnable adjacency bias (one per head)
        self.adj_bias = nn.Parameter(torch.full((n_heads,), adj_bias_init))
        # FFN
        self.ffn  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, d_model], adj: [B, B]
        B = x.size(0)
        Q = self.Wq(x).view(B, self.n_heads, self.d_head)   # [B, H, dh]
        K = self.Wk(x).view(B, self.n_heads, self.d_head)   # [B, H, dh]
        V = self.Wv(x).view(B, self.n_heads, self.d_head)   # [B, H, dh]

        # Attention scores: [B, H, B] (queries × keys)
        attn = torch.einsum("bhd,nhd->bhn", Q, K) * self.scale   # [B, H, B]
        # Add adjacency bias: adj [B, B] → broadcast over H
        adj_bias = self.adj_bias.view(1, self.n_heads, 1) * adj.unsqueeze(1)  # [B, H, B]
        attn = attn + adj_bias
        attn = torch.softmax(attn, dim=-1)                        # [B, H, B]
        attn = self.drop(attn)

        # Aggregate: [B, H, B] × [B, H, dh] → [B, H, dh]
        out  = torch.einsum("bhn,nhd->bhd", attn, V)              # [B, H, dh]
        out  = self.Wo(out.reshape(B, -1))                        # [B, d_model]
        x    = self.norm(x + out)                                 # residual
        x    = x + self.ffn(x)                                    # FFN residual
        return x


class GraphTransformerModel(nn.Module):
    def __init__(self, hubert_dim: int, whisper_dim: int, d_model: int,
                 n_heads: int, n_layers: int, knn_k: int,
                 adj_bias_init: float, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.knn_k = knn_k
        self.input_proj = nn.Sequential(
            nn.Linear(hubert_dim + whisper_dim, d_model * 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
        )
        self.gt_layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, adj_bias_init, dropout * 0.5)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x   = torch.cat([h, w], dim=-1)   # [B, H+W]
        x   = self.input_proj(x)          # [B, d_model]
        adj = build_knn_adj(x.detach(), self.knn_k)  # [B, B] (no grad through graph)
        for layer in self.gt_layers:
            x = layer(x, adj)             # [B, d_model]
        return self.head(x)               # [B, n_classes]


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
    print(f"  k-NN graph: k={args.knn_k}, built per-batch on projected embeddings")

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
    model = GraphTransformerModel(
        hubert_dim=h_feats.shape[1], whisper_dim=w_feats.shape[1],
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_gt_layers,
        knn_k=args.knn_k, adj_bias_init=args.adj_bias_init,
        n_classes=len(STUTTER_TYPES), dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nE5 model parameters: {n_params:,}")
    print(f"  Input: HuBERT({h_feats.shape[1]})+Whisper({w_feats.shape[1]}) → {args.d_model}")
    print(f"  GT layers: {args.n_gt_layers}, heads: {args.n_heads}, adj_bias β={args.adj_bias_init}")
    print(f"  k-NN: k={args.knn_k} (batch-level cosine graph, per-forward)")

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
    best_ckpt = args.ckpt_dir / "e5_best.pt"

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
    with (args.out_dir / "e5_perclass_results.csv").open("w", newline="", encoding="utf-8") as f:
        wrt = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        wrt.writeheader()
        for t in STUTTER_TYPES:
            wrt.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                          "precision": test_m[f"pre_{t}"],
                          "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    with (args.out_dir / "e5_train_history.csv").open("w", newline="", encoding="utf-8") as f:
        wrt = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        wrt.writeheader(); wrt.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "E5",
        "title": "Graph-Transformer: batch k-NN + adjacency-biased attention (HuBERT+Whisper)",
        "device": str(device),
        "hubert_alias": args.hubert_alias, "hubert_layer": args.hubert_layer,
        "whisper_alias": args.whisper_alias, "whisper_layer": args.whisper_layer,
        "d_model": args.d_model, "n_heads": args.n_heads, "n_gt_layers": args.n_gt_layers,
        "knn_k": args.knn_k, "adj_bias_init": args.adj_bias_init,
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
    (args.out_dir / "e5_run_report.json").write_text(json.dumps(run_report, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val")
        axes[0].set_title(f"E5 Loss (GraphTransformer, k={args.knn_k}, L={args.n_gt_layers})")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="darkred", marker="o", ms=4)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("E5 Validation Macro-F1 (Graph-Transformer)")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "e5_train_curves.png", dpi=160); plt.close(fig)

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
        ax2.set_title(f"E5 Per-class F1 (GraphTransformer k={args.knn_k})")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "e5_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"\n  Saved: {args.fig_dir / 'e5_train_curves.png'}")
        print(f"  Saved: {args.fig_dir / 'e5_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  E5 complete.")
    print(f"   🏁 ALL POST-35 EXPERIMENTS DONE! Full benchmark complete.")
    print(f"   Report : {args.out_dir / 'e5_run_report.json'}")
    print(f"   Ckpt   : {best_ckpt}")


if __name__ == "__main__":
    main()
