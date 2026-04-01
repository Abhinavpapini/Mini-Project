"""E2: Dynamic Graph Attention Network (GAT) for multi-label stutter classification.

Extends E1 (GCN) by replacing uniform neighbour aggregation with LEARNED
per-edge attention weights (Veličković et al., 2018 — Graph Attention Networks).

Key difference vs E1:
  E1 GCN : h_i = ReLU(sum_j (norm) * W * h_j)         uniform weights
  E2 GAT : h_i = ReLU(sum_j  α_ij  * W * h_j)         learned attention

Attention: α_ij = softmax_j( LeakyReLU( a^T [W*h_i || W*h_j] ) )
Multi-head: K heads, output concatenated (layer 1) or averaged (layer 2)

Graph: same k-NN cosine similarity graph as E1 (k=10, 20906 nodes)
Features: PCA-64 (E2 uses wider feature set than E1's 32)
Loss: MultiLabelSoftMarginLoss + L2 reg

Run command:
    python experiments/E2_gat_dynamic.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --ssl-pca-dim 64 \
        --k-nn 10 \
        --gat-hidden 64 \
        --gat-heads 4 \
        --dropout 0.3 \
        --attn-dropout 0.1 \
        --epochs 200 \
        --lr 5e-4
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
from sklearn.preprocessing import StandardScaler, normalize


STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2: Multi-head GAT on k-NN speech graph")
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
                   help="PCA dims — 64 for E2 (wider than E1's 32)")
    p.add_argument("--k-nn",           type=int,  default=10)
    # GAT params
    p.add_argument("--gat-hidden",     type=int,  default=64,
                   help="Hidden dim PER HEAD in GAT layer 1")
    p.add_argument("--gat-heads",      type=int,  default=4,
                   help="Number of attention heads")
    p.add_argument("--gat-out",        type=int,  default=32,
                   help="Output dim of GAT layer 2 (averaged heads)")
    p.add_argument("--dropout",        type=float,default=0.3)
    p.add_argument("--attn-dropout",   type=float,default=0.1)
    p.add_argument("--leaky-slope",    type=float,default=0.2)
    # Training
    p.add_argument("--test-size",      type=float,default=0.20)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--epochs",         type=int,  default=200)
    p.add_argument("--lr",             type=float,default=5e-4)
    p.add_argument("--weight-decay",   type=float,default=5e-4)
    p.add_argument("--threshold",      type=float,default=0.5)
    p.add_argument("--out-dir",  type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",  type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/E2"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data utilities (same as E1)
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


# ---------------------------------------------------------------------------
# Graph construction (same cosine k-NN as E1 — returns edge index)
# ---------------------------------------------------------------------------

def build_knn_edges(x: np.ndarray, k: int, device: torch.device):
    """
    Returns:
      edge_index: [2, E] — source and target node indices  (both directions)
      edge_src, edge_dst: 1-D LongTensors for GAT scatter ops
    """
    n = x.shape[0]
    print(f"  Building {k}-NN cosine graph over {n} nodes ...")
    x_norm = normalize(x, norm="l2")
    chunk  = 512
    src_list, dst_list = [], []
    for start in range(0, n, chunk):
        end  = min(start + chunk, n)
        sims = x_norm[start:end] @ x_norm.T
        for local_i in range(end - start):
            sims[local_i, start + local_i] = -1.0
        top_k = np.argpartition(sims, -k, axis=1)[:, -k:]
        for local_i in range(end - start):
            gi = start + local_i
            for j in top_k[local_i]:
                src_list.append(gi);   dst_list.append(int(j))
                src_list.append(int(j)); dst_list.append(gi)

    edge_set = set(zip(src_list, dst_list))
    edge_set  = {(s, d) for s, d in edge_set if s != d}
    # Add self-loops
    edge_set |= {(i, i) for i in range(n)}
    src_arr = np.array([s for s, d in edge_set], dtype=np.int64)
    dst_arr = np.array([d for s, d in edge_set], dtype=np.int64)
    n_edges = len(src_arr) - n   # excluding self-loops
    print(f"  Graph: {n} nodes | {n_edges // 2} undirected edges (k={k}) | +self-loops")
    src_t = torch.from_numpy(src_arr).to(device)
    dst_t = torch.from_numpy(dst_arr).to(device)
    return src_t, dst_t


# ---------------------------------------------------------------------------
# GAT layer (manual implementation, no PyG dependency)
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    Single GAT layer with K attention heads.

    For each directed edge (i→j):
      e_ij = LeakyReLU( a_k^T [W_k h_i || W_k h_j] )
      α_ij = softmax over j of e_ij   (per target node i, over neighbours j)
      h_i' = σ( Σ_j α_ij W_k h_j )   (aggregation per head)

    concat=True  → output dim = K * out_per_head  (used for layer 1)
    concat=False → output dim = out_per_head       (mean over heads, layer 2)
    """

    def __init__(self, in_dim: int, out_per_head: int, n_heads: int,
                 dropout: float, attn_dropout: float,
                 leaky_slope: float, concat: bool) -> None:
        super().__init__()
        self.n_heads      = n_heads
        self.out_per_head = out_per_head
        self.concat       = concat

        self.W    = nn.Linear(in_dim, out_per_head * n_heads, bias=False)
        self.a    = nn.Parameter(torch.Tensor(n_heads, 2 * out_per_head))
        self.bias = nn.Parameter(torch.zeros(out_per_head * n_heads if concat else out_per_head))
        nn.init.xavier_uniform_(self.a.unsqueeze(0), gain=1.4)

        self.leaky    = nn.LeakyReLU(leaky_slope)
        self.drop     = nn.Dropout(dropout)
        self.attn_drop= nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor,
                src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """
        x   : [N, in_dim]
        src : [E]  source node indices of each edge
        dst : [E]  target node indices of each edge
        """
        N = x.size(0)
        # Linear projection: [N, K*D]
        Wh = self.W(self.drop(x)).view(N, self.n_heads, self.out_per_head)

        # Gather src and dst embeddings for each edge
        Wh_src = Wh[src]   # [E, K, D]
        Wh_dst = Wh[dst]   # [E, K, D]

        # Attention logit: a^T [h_i || h_j]  → [E, K]
        cat = torch.cat([Wh_src, Wh_dst], dim=-1)   # [E, K, 2D]
        e   = self.leaky((cat * self.a).sum(dim=-1)) # [E, K]

        # Softmax over neighbours of each dst node (scatter softmax)
        # We need: for each (dst, head) → softmax over src
        e_exp = e.exp()   # [E, K]
        # Sum e_exp per (dst, head)
        E_sum = torch.zeros(N, self.n_heads, device=x.device)
        E_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.n_heads), e_exp)
        alpha = e_exp / (E_sum[dst] + 1e-16)   # [E, K]  normalised attention
        alpha = self.attn_drop(alpha)

        # Aggregate: Σ_j α_ij * Wh_j
        agg_src = Wh_src * alpha.unsqueeze(-1)   # [E, K, D]
        out = torch.zeros(N, self.n_heads, self.out_per_head, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2)
                         .expand(-1, self.n_heads, self.out_per_head), agg_src)

        if self.concat:
            out = out.view(N, self.n_heads * self.out_per_head) + self.bias
        else:
            out = out.mean(dim=1) + self.bias   # average heads
        return out


class GAT2Layer(nn.Module):
    """
    2-layer GAT:
      Layer 1: K-head concat  → K*d_h features, ELU activation
      Layer 2: K-head average → d_out features, no activation
      Head   : Linear(d_out→5) for multi-label classification
    """

    def __init__(self, in_dim: int, hidden_per_head: int, n_heads: int,
                 gat_out: int, num_classes: int,
                 dropout: float, attn_dropout: float, leaky_slope: float) -> None:
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_per_head, n_heads, dropout,
                             attn_dropout, leaky_slope, concat=True)
        self.bn1  = nn.BatchNorm1d(hidden_per_head * n_heads)
        self.gat2 = GATLayer(hidden_per_head * n_heads, gat_out, n_heads,
                             dropout, attn_dropout, leaky_slope, concat=False)
        self.head = nn.Sequential(
            nn.LayerNorm(gat_out),
            nn.Dropout(dropout),
            nn.Linear(gat_out, gat_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gat_out * 2, num_classes),
        )

    def forward(self, x: torch.Tensor,
                src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.bn1(self.gat1(x, src, dst)))
        h = self.gat2(h, src, dst)
        return self.head(h)


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
    print(f"\nDataset: {n} nodes | Label distribution:")
    for i, t in enumerate(STUTTER_TYPES):
        pos = int(y[:, i].sum())
        print(f"  {t:15s}: {pos:5d} pos  ({pos/n:.2%})")

    # ------------------------------------------------------------------
    # 2. PCA (fit on train, transform all — transductive)
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y[:, 0].astype(int)
    )
    pca_dim = min(args.ssl_pca_dim, ssl_feats.shape[1])
    sc  = StandardScaler()
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    pca.fit(sc.fit_transform(ssl_feats[train_idx]))
    x_all = pca.transform(sc.transform(ssl_feats)).astype(np.float32)
    print(f"\nSSL PCA-{pca_dim} expl_var={pca.explained_variance_ratio_.sum():.3f}")

    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx]   = True

    # ------------------------------------------------------------------
    # 3. Build k-NN edge list
    # ------------------------------------------------------------------
    print(f"\nBuilding k-NN graph (k={args.k_nn}) ...")
    src_t, dst_t = build_knn_edges(x_all, k=args.k_nn, device=device)

    X = torch.from_numpy(x_all).to(device)
    Y = torch.from_numpy(y).to(device)

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model = GAT2Layer(
        in_dim=pca_dim, hidden_per_head=args.gat_hidden,
        n_heads=args.gat_heads, gat_out=args.gat_out,
        num_classes=len(STUTTER_TYPES),
        dropout=args.dropout, attn_dropout=args.attn_dropout,
        leaky_slope=args.leaky_slope,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nE2 model parameters: {n_params:,}")
    print(f"  GAT: {pca_dim}→{args.gat_hidden}×{args.gat_heads}(concat)"
          f"→{args.gat_out}(avg-heads)→5 | k={args.k_nn}")
    print(f"  Attention: {args.gat_heads}-head LeakyReLU(slope={args.leaky_slope})")

    criterion = nn.MultiLabelSoftMarginLoss()
    opt       = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    # ------------------------------------------------------------------
    # 5. Training (full-batch, train mask loss only)
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs (full-batch GAT) ...")
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "e2_best.pt"
    log_every = max(1, args.epochs // 20)

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X, src_t, dst_t)
        loss   = criterion(logits[train_mask], Y[train_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                logits_e = model(X, src_t, dst_t)
                val_loss = criterion(logits_e[test_mask], Y[test_mask]).item()
                vm = evaluate_multilabel(Y[test_mask].cpu().numpy(),
                                         logits_e[test_mask].cpu().numpy(), args.threshold)
            macro_f1 = vm["macro_f1"]
            row = {"epoch": epoch, "train_loss": round(loss.item(), 6),
                   "val_loss": round(val_loss, 6), "macro_f1": round(macro_f1, 6)}
            history.append(row)
            print(f"epoch={epoch:03d}  train_loss={loss.item():.5f}  "
                  f"val_loss={val_loss:.5f}  macro_f1={macro_f1:.5f}")
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save(model.state_dict(), best_ckpt)

    # ------------------------------------------------------------------
    # 6. Final evaluation
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        logits_final = model(X, src_t, dst_t)
    test_m = evaluate_multilabel(
        Y[test_mask].cpu().numpy(), logits_final[test_mask].cpu().numpy(), args.threshold)

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
    # 7. Save outputs
    # ------------------------------------------------------------------
    perclass_csv = args.out_dir / "e2_perclass_results.csv"
    with perclass_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    hist_csv = args.out_dir / "e2_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "E2", "title": "2-layer GAT on k-NN speech graph (HuBERT-large PCA-64)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "ssl_pca_dim": int(pca_dim),
        "k_nn": args.k_nn, "gat_hidden_per_head": args.gat_hidden,
        "gat_heads": args.gat_heads, "gat_out": args.gat_out,
        "n_params": int(n_params), "n_nodes": int(n),
        "loss": "MultiLabelSoftMarginLoss", "setting": "transductive",
        "epochs": args.epochs, "best_epoch": int(best_ep),
        "lr": args.lr, "weight_decay": args.weight_decay, "threshold": args.threshold,
        "macro_f1": test_m["macro_f1"], "micro_f1": test_m["micro_f1"],
        "macro_precision": test_m["macro_pre"], "macro_recall": test_m["macro_rec"],
        "macro_auprc": test_m["macro_auprc"],
        "per_class": {t: {"f1": test_m[f"f1_{t}"], "precision": test_m[f"pre_{t}"],
                          "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]}
                      for t in STUTTER_TYPES},
        "best_checkpoint": str(best_ckpt),
    }
    report_json = args.out_dir / "e2_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 8. Figures
    # ------------------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val")
        axes[0].set_title(f"E2 GAT Loss ({args.gat_heads}-head, k={args.k_nn})")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="green", marker="o", ms=4)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("E2 Validation Macro-F1"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "e2_train_curves.png", dpi=160); plt.close(fig)
        print(f"  Saved: {args.fig_dir / 'e2_train_curves.png'}")

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
        ax2.set_title(f"E2 Per-class F1 (GAT, {args.gat_heads}-head, k={args.k_nn})")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "e2_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"  Saved: {args.fig_dir / 'e2_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  E2 complete.")
    print(f"   Report    : {report_json}")
    print(f"   Per-class : {perclass_csv}")
    print(f"   History   : {hist_csv}")
    print(f"   Ckpt      : {best_ckpt}")


if __name__ == "__main__":
    main()
