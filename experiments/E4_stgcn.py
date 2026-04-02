"""E4: Spatial-Temporal GCN (ST-GCN) for multi-label stutter classification.

Combines TWO types of graph edges:
    1. SPATIAL edges  — k-NN cosine similarity (captures acoustic similarity)
    2. TEMPORAL edges — sequential clip ordering WITHIN same show+episode
                                            (captures conversational context: neighbouring clips influence each other)

Combined adjacency: A = A_spatial + lambda * A_temporal
Then apply a 2-layer GCN on the combined graph.

Rationale for temporal edges:
  - Stuttering events are contextual: a speaker block often follows or precedes other events
  - Adjacent clips in the same episode share conversational context
  - If clip_i is Prolongation, clip_{i+1} from the same speaker is more likely stutter too

Temporal graph construction:
  - Sort clips within same (show, episode) by clip_id numerically
  - Connect clip_k to clip_{k±1} within the same episode
  - Self-loops remain, directed edges made symmetric

Run command:
    python experiments/E4_stgcn.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --ssl-pca-dim 32 \
        --k-nn 10 \
        --temporal-lambda 0.5 \
        --gcn-hidden 64 \
        --gcn-out 32 \
        --dropout 0.3 \
        --epochs 200 \
        --lr 1e-3
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
    p = argparse.ArgumentParser(description="E4: ST-GCN spatial+temporal graph")
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
    p.add_argument("--ssl-pca-dim",     type=int,  default=32)
    p.add_argument("--k-nn",            type=int,  default=10,
                   help="Spatial k-NN edges")
    p.add_argument("--temporal-lambda", type=float,default=0.5,
                   help="Weight scaling for temporal edges relative to spatial edges")
    # GCN params
    p.add_argument("--gcn-hidden",      type=int,  default=64)
    p.add_argument("--gcn-out",         type=int,  default=32)
    p.add_argument("--dropout",         type=float,default=0.3)
    # Training
    p.add_argument("--test-size",       type=float,default=0.20)
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--epochs",          type=int,  default=200)
    p.add_argument("--lr",              type=float,default=1e-3)
    p.add_argument("--weight-decay",    type=float,default=5e-4)
    p.add_argument("--threshold",       type=float,default=0.5)
    p.add_argument("--out-dir",   type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",   type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir",  type=Path, default=Path("artifacts/checkpoints/E4"))
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


# ---------------------------------------------------------------------------
# Spatial graph: k-NN cosine
# ---------------------------------------------------------------------------

def build_spatial_edges(x: np.ndarray, k: int) -> Tuple[List[int], List[int]]:
    n = x.shape[0]
    x_norm = normalize(x, norm="l2")
    chunk  = 512
    rows, cols = [], []
    for start in range(0, n, chunk):
        end  = min(start + chunk, n)
        sims = x_norm[start:end] @ x_norm.T
        for li in range(end - start):
            sims[li, start + li] = -1.0
        top_k = np.argpartition(sims, -k, axis=1)[:, -k:]
        for li in range(end - start):
            gi = start + li
            for j in top_k[li]:
                rows.append(gi); cols.append(int(j))
                rows.append(int(j)); cols.append(gi)
    # Deduplicate and drop self
    edge_set = {(r, c) for r, c in zip(rows, cols) if r != c}
    return [s for s, d in edge_set], [d for s, d in edge_set]


# ---------------------------------------------------------------------------
# Temporal graph: sequential clip ordering within same show+episode
# ---------------------------------------------------------------------------

def build_temporal_edges(clip_keys: List[Tuple[str, str, str]],
                         n: int) -> Tuple[List[int], List[int], int]:
    """
    Groups clips by (show, episode) and connects consecutive clips.
    Returns src_list, dst_list (symmetric), n_temporal_edges (undirected).
    """
    # Group by (show, episode)
    episode_clips: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for idx, (show, ep, clip_id) in enumerate(clip_keys[:n]):
        key = (show, ep)
        if key not in episode_clips:
            episode_clips[key] = []
        episode_clips[key].append((idx, clip_id))

    rows, cols = [], []
    for (show, ep), clips in episode_clips.items():
        # Sort by clip_id numerically where possible
        def sort_key(x: Tuple[int, str]) -> int:
            try:
                return int(x[1])
            except ValueError:
                return hash(x[1])
        clips_sorted = sorted(clips, key=sort_key)
        # Connect consecutive clips
        for i in range(len(clips_sorted) - 1):
            a, b = clips_sorted[i][0], clips_sorted[i + 1][0]
            rows.append(a); cols.append(b)
            rows.append(b); cols.append(a)

    n_temporal = len({(min(r, c), max(r, c)) for r, c in zip(rows, cols)})
    return rows, cols, n_temporal


# ---------------------------------------------------------------------------
# Build combined normalised A_hat (spatial + lambda * temporal)
# ---------------------------------------------------------------------------

def build_stgcn_adjacency(
    x: np.ndarray, clip_keys: List[Tuple[str, str, str]],
    k: int, lam: float, device: torch.device
) -> torch.Tensor:
    n = x.shape[0]
    print(f"  Building spatial k-NN (k={k}) edges ...")
    sp_src, sp_dst = build_spatial_edges(x, k)
    n_spatial = len({(min(r, c), max(r, c)) for r, c in zip(sp_src, sp_dst)})

    print(f"  Building temporal (within-episode) edges ...")
    tm_src, tm_dst, n_temporal = build_temporal_edges(clip_keys, n)

    print(f"  Spatial edges: {n_spatial} undirected | Temporal edges: {n_temporal} undirected")
    print(f"  Temporal lambda: {lam}")

    # Build sparse adjacency with weights
    all_rows = sp_src + tm_src + list(range(n))    # + self-loops
    all_cols = sp_dst + tm_dst + list(range(n))
    all_vals = ([1.0] * len(sp_src) +
                [lam] * len(tm_src) +
                [1.0] * n)

    # Acccumulate (some edges may be both spatial and temporal — add weights)
    edge_dict: Dict[Tuple[int, int], float] = {}
    for r, c, v in zip(all_rows, all_cols, all_vals):
        edge_dict[(r, c)] = edge_dict.get((r, c), 0.0) + v

    rows_ = [r for r, c in edge_dict]; cols_ = [c for r, c in edge_dict]
    vals_ = [v for v in edge_dict.values()]

    idx = torch.LongTensor([rows_, cols_])
    val = torch.FloatTensor(vals_)
    A   = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
    d_inv_sqrt = deg.pow(-0.5)
    r_idx = A.coalesce().indices()[0]; c_idx = A.coalesce().indices()[1]
    scale = d_inv_sqrt[r_idx] * d_inv_sqrt[c_idx]
    A_hat = torch.sparse_coo_tensor(A.coalesce().indices(),
                                    A.coalesce().values() * scale, (n, n)).coalesce()
    n_total_edges = A.coalesce().indices().shape[1] - n
    print(f"  Combined A_hat: {n} nodes | {n_total_edges} weighted edges | normalised")
    return A_hat.to(device)


# ---------------------------------------------------------------------------
# GCN model (2-layer) on combined ST adjacency
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, A_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.W(torch.sparse.mm(A_hat, x))


class STGCN(nn.Module):
    """
    Spatial-Temporal GCN: 2-layer GCN on combined spatial (k-NN)
    + temporal (episode-sequential) adjacency.

    The combined adjacency encodes BOTH acoustic similarity AND
    conversational temporal context in one propagation kernel.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 num_classes: int, dropout: float) -> None:
        super().__init__()
        self.drop  = nn.Dropout(dropout)
        self.gcn1  = GCNLayer(in_dim,  hidden)
        self.gcn2  = GCNLayer(hidden,  out_dim)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.head  = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, num_classes),
        )

    def forward(self, A_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.gcn1(A_hat, self.drop(x))))
        h = self.gcn2(A_hat, self.drop(h))
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
    # 2. PCA (transductive)
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

    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[test_idx]   = True

    # ------------------------------------------------------------------
    # 3. Build ST-GCN combined adjacency
    # ------------------------------------------------------------------
    print(f"\nBuilding Spatial-Temporal graph ...")
    A_hat = build_stgcn_adjacency(x_all, clip_keys, args.k_nn,
                                   args.temporal_lambda, device)

    X = torch.from_numpy(x_all).to(device)
    Y = torch.from_numpy(y).to(device)

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model = STGCN(
        in_dim=pca_dim, hidden=args.gcn_hidden,
        out_dim=args.gcn_out, num_classes=len(STUTTER_TYPES),
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nE4 model parameters: {n_params:,}")
    print(f"  ST-GCN: {pca_dim}→{args.gcn_hidden}→{args.gcn_out}→5")
    print(f"  Spatial: k-NN={args.k_nn} | Temporal: λ={args.temporal_lambda}")

    criterion = nn.MultiLabelSoftMarginLoss()
    opt       = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs (full-batch ST-GCN) ...")
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "e4_best.pt"
    log_every = max(1, args.epochs // 20)

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(A_hat, X)
        loss   = criterion(logits[train_mask], Y[train_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                logits_e = model(A_hat, X)
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
        logits_final = model(A_hat, X)
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
    print(f"\n  Temporal edge weight λ={args.temporal_lambda}")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    perclass_csv = args.out_dir / "e4_perclass_results.csv"
    with perclass_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})
    hist_csv = args.out_dir / "e4_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "E4", "title": "ST-GCN: spatial k-NN + temporal episode edges (HuBERT-large PCA-32)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "ssl_pca_dim": int(pca_dim),
        "k_nn": args.k_nn, "temporal_lambda": args.temporal_lambda,
        "gcn_hidden": args.gcn_hidden, "gcn_out": args.gcn_out,
        "n_params": int(n_params), "n_nodes": int(n),
        "loss": "MultiLabelSoftMarginLoss", "setting": "transductive",
        "epochs": args.epochs, "best_epoch": int(best_ep),
        "macro_f1": test_m["macro_f1"], "micro_f1": test_m["micro_f1"],
        "macro_precision": test_m["macro_pre"], "macro_recall": test_m["macro_rec"],
        "macro_auprc": test_m["macro_auprc"],
        "per_class": {t: {"f1": test_m[f"f1_{t}"], "precision": test_m[f"pre_{t}"],
                          "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]}
                      for t in STUTTER_TYPES},
        "best_checkpoint": str(best_ckpt),
    }
    (args.out_dir / "e4_run_report.json").write_text(json.dumps(run_report, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val")
        axes[0].set_title(f"E4 ST-GCN Loss (spatial+temporal, λ={args.temporal_lambda})")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="darkorange", marker="o", ms=4)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("E4 Validation Macro-F1"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "e4_train_curves.png", dpi=160); plt.close(fig)
        print(f"  Saved: {args.fig_dir / 'e4_train_curves.png'}")

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
        ax2.set_title(f"E4 Per-class F1 (ST-GCN, λ={args.temporal_lambda})")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "e4_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"  Saved: {args.fig_dir / 'e4_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  E4 complete.")
    print(f"   Report : {args.out_dir / 'e4_run_report.json'}")
    print(f"   Ckpt   : {best_ckpt}")


if __name__ == "__main__":
    main()
