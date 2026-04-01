"""E3: Hypergraph Neural Network (HGNN) for multi-label stutter classification.

A hypergraph generalises regular graphs: each HYPEREDGE can connect
MORE than 2 nodes simultaneously (a whole cluster of similar clips).

Hyperedge construction (multi-relational):
  Type-A: k-means clusters on PCA-32 SSL features        (n_A=100 hyperedges)
  Type-B: k-means clusters on MFCC-21 handcrafted feats  (n_B=50  hyperedges)
  Combined incidence matrix H: [N, n_A + n_B]
    H[i, c] = 1 iff clip i belongs to cluster c

HGNN propagation (Feng et al. 2019, HGNN):
  A_hyper = D_v^(-1/2) * H * B * D_e^(-1) * H^T * D_v^(-1/2)
  H^(l+1) = sigma( A_hyper * H^(l) * Theta_l )

Where:
  D_v = diag(node degrees)       [N, N]
  D_e = diag(hyperedge degrees)  [M, M]
  B   = diag(hyperedge weights)  [M, M]  (uniform = I here)
  Theta_l = learnable weight matrix

Value over E1/E2:
  - Each hyperedge aggregates an ENTIRE CLUSTER of clips at once
  - Multi-relational: SSL clusters capture acoustic similarity,
    MFCC clusters capture prosodic/spectral similarity separately
  - Richer information flow than pairwise edges

Run command:
    python experiments/E3_hgnn.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --ssl-pca-dim 32 \
        --mfcc-cache artifacts/features/mfcc/fold0/mfcc_stats.npy \
        --mfcc-dim 21 \
        --n-ssl-clusters 100 \
        --n-mfcc-clusters 50 \
        --hgnn-hidden 64 \
        --hgnn-out 32 \
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F


STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3: HGNN with multi-relational hyperedges")
    p.add_argument("--features-root",    type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",     type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",     type=int,  default=21)
    p.add_argument("--fold",             type=str,  default="fold0")
    p.add_argument("--clips-root",       type=Path,
                   default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",       type=Path,
                   default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels",   type=Path,
                   default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--mfcc-cache",       type=Path,
                   default=Path("artifacts/features/mfcc/fold0/mfcc_stats.npy"))
    p.add_argument("--ssl-pca-dim",      type=int,  default=32)
    p.add_argument("--mfcc-dim",         type=int,  default=21)
    # Hyperedge params
    p.add_argument("--n-ssl-clusters",   type=int,  default=100,
                   help="k-means clusters on SSL features (Type-A hyperedges)")
    p.add_argument("--n-mfcc-clusters",  type=int,  default=50,
                   help="k-means clusters on MFCC features (Type-B hyperedges)")
    # HGNN params
    p.add_argument("--hgnn-hidden",      type=int,  default=64)
    p.add_argument("--hgnn-out",         type=int,  default=32)
    p.add_argument("--dropout",          type=float,default=0.3)
    # Training
    p.add_argument("--test-size",        type=float,default=0.20)
    p.add_argument("--seed",             type=int,  default=42)
    p.add_argument("--epochs",           type=int,  default=200)
    p.add_argument("--lr",               type=float,default=1e-3)
    p.add_argument("--weight-decay",     type=float,default=5e-4)
    p.add_argument("--threshold",        type=float,default=0.5)
    p.add_argument("--out-dir",    type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",    type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir",   type=Path, default=Path("artifacts/checkpoints/E3"))
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
# Hypergraph construction
# ---------------------------------------------------------------------------

def build_hyperedge_incidence(
    x_ssl: np.ndarray, x_mfcc: np.ndarray,
    n_ssl: int, n_mfcc: int, seed: int
) -> Tuple[np.ndarray, int, int]:
    """
    Build multi-relational incidence matrix H: [N, n_ssl + n_mfcc]
    H[i, c] = 1 iff clip i belongs to hyperedge c.

    Type-A (SSL):  k-means on PCA-32 features    → n_ssl  hyperedges
    Type-B (MFCC): k-means on MFCC-21 features   → n_mfcc hyperedges
    """
    N = x_ssl.shape[0]
    H = np.zeros((N, n_ssl + n_mfcc), dtype=np.float32)

    print(f"  Type-A: k-means (k={n_ssl}) on SSL PCA-32 ...")
    km_ssl = KMeans(n_clusters=n_ssl, random_state=seed, n_init=5, max_iter=100)
    lbls_ssl = km_ssl.fit_predict(x_ssl)
    for i, c in enumerate(lbls_ssl):
        H[i, c] = 1.0
    sizes_ssl = [int((lbls_ssl == c).sum()) for c in range(n_ssl)]
    print(f"    Cluster sizes: min={min(sizes_ssl)}, mean={np.mean(sizes_ssl):.1f}, max={max(sizes_ssl)}")

    print(f"  Type-B: k-means (k={n_mfcc}) on MFCC-21 ...")
    km_mfcc = KMeans(n_clusters=n_mfcc, random_state=seed+1, n_init=5, max_iter=100)
    lbls_mfcc = km_mfcc.fit_predict(x_mfcc)
    for i, c in enumerate(lbls_mfcc):
        H[i, n_ssl + c] = 1.0
    sizes_mfcc = [int((lbls_mfcc == c).sum()) for c in range(n_mfcc)]
    print(f"    Cluster sizes: min={min(sizes_mfcc)}, mean={np.mean(sizes_mfcc):.1f}, max={max(sizes_mfcc)}")

    print(f"  Incidence matrix H: [{N} × {n_ssl + n_mfcc}] | "
          f"{int(H.sum())} non-zeros | sparsity={(1-H.mean()):.3f}")
    return H, n_ssl, n_mfcc


def compute_hypergraph_laplacian(H: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute: Theta = D_v^(-1/2) * H * D_e^(-1) * H^T * D_v^(-1/2)
    Returns dense [N, N] propagation matrix.
    NOTE: For 20K nodes this is large — computed once and stored as sparse.
    """
    N, M = H.shape
    H_t = torch.from_numpy(H).to(device)

    D_v = H_t.sum(dim=1)                    # [N] — node degrees
    D_e = H_t.sum(dim=0)                    # [M] — hyperedge degrees

    D_v_inv_sqrt = (D_v + 1e-8).pow(-0.5)  # [N]
    D_e_inv      = (D_e + 1e-8).pow(-1.0)  # [M]

    # Theta = D_v^(-1/2) H D_e^(-1) H^T D_v^(-1/2)
    # Compute step by step to avoid N×N intermediate when possible
    # H_norm_left  = D_v^(-1/2) [N,1] * H [N,M]
    H_left  = H_t * D_v_inv_sqrt.unsqueeze(1)   # [N, M]
    # H_norm_right = H [M,N] * D_e^(-1) [M,1] → D_e^(-1)[M,1] * H^T[M,N]
    H_right = (H_t * D_e_inv.unsqueeze(0)).T     # [M, N] → transposed: use H_t^T
    # Actually: H_right = H^T * D_v^(-1/2) component
    H_right2 = H_t.T * D_v_inv_sqrt.unsqueeze(0) # [M, N]

    # Theta = H_left @ (D_e^(-1) * H_right2)
    # = H_left [N,M] @ diag(D_e_inv)[M,M] @ H_right2[M,N]
    D_e_H = D_e_inv.unsqueeze(1) * H_right2      # [M, N]   D_e^(-1) * H^T * D_v^(-1/2)

    # Theta stored as [N, M] @ [M, N] product — do as sparse for memory efficiency
    # For N=20906, M=150: [N,M]@[M,N] = [20906,20906] — large!
    # Instead, keep factored form (Theta_left, Theta_right) and apply lazily
    return H_left, D_e_H   # [N, M], [M, N]


# ---------------------------------------------------------------------------
# HGNN model
# ---------------------------------------------------------------------------

class HGNNLayer(nn.Module):
    """
    Single HGNN layer (Feng et al. 2019).
    H^(l+1) = sigma( Theta * X * W )
    where Theta = D_v^(-1/2) H B D_e^(-1) H^T D_v^(-1/2)
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, Theta_L: torch.Tensor, Theta_R: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        """
        Theta_L: [N, M]  left factor of Theta
        Theta_R: [M, N]  right factor of Theta
        x      : [N, F]
        """
        Xw   = self.W(x)                       # [N, out_dim]
        prop = Theta_L @ (Theta_R @ Xw)        # [N, M] @ [M, N] @ [N, out] → [N, out]
        return prop


class HGNN2Layer(nn.Module):
    """
    2-layer HGNN:
      X^1 = ReLU( Theta * X^0 * W_1 )
      X^2 = Theta * X^1 * W_2
      logits = head(X^2)
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 num_classes: int, dropout: float) -> None:
        super().__init__()
        self.drop  = nn.Dropout(dropout)
        self.hgnn1 = HGNNLayer(in_dim,  hidden)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.hgnn2 = HGNNLayer(hidden,  out_dim)
        self.head  = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, num_classes),
        )

    def forward(self, TL: torch.Tensor, TR: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.hgnn1(TL, TR, self.drop(x))))
        h = self.hgnn2(TL, TR, self.drop(h))
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
    # 1. Load features
    # ------------------------------------------------------------------
    print(f"\nLoading HuBERT-large layer {args.hubert_layer} cache ...")
    ssl_feats = load_ssl_cache(args.features_root, args.hubert_alias, args.fold, args.hubert_layer)
    if ssl_feats.ndim != 2:
        ssl_feats = ssl_feats.reshape(ssl_feats.shape[0], -1)
    n = ssl_feats.shape[0]
    print(f"  SSL features: {ssl_feats.shape}")

    if args.mfcc_cache.exists():
        mfcc_all = np.load(args.mfcc_cache)
        if mfcc_all.ndim != 2:
            mfcc_all = mfcc_all.reshape(mfcc_all.shape[0], -1)
        mfcc_feats = mfcc_all[:n, :args.mfcc_dim]
        print(f"  MFCC features: {mfcc_feats.shape}")
        use_mfcc = True
    else:
        print(f"  [WARN] MFCC cache not found at {args.mfcc_cache}. Using SSL only.")
        mfcc_feats = None; use_mfcc = False
        args.n_mfcc_clusters = 0

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
    print(f"\nDataset: {n} nodes | Label distribution:")
    for i, t in enumerate(STUTTER_TYPES):
        pos = int(y[:, i].sum())
        print(f"  {t:15s}: {pos:5d} pos  ({pos/n:.2%})")

    # ------------------------------------------------------------------
    # 3. PCA (fit on train, transform all — transductive)
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y[:, 0].astype(int)
    )
    pca_dim = min(args.ssl_pca_dim, ssl_feats.shape[1])
    sc  = StandardScaler()
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    pca.fit(sc.fit_transform(ssl_feats[train_idx]))
    x_ssl = pca.transform(sc.transform(ssl_feats)).astype(np.float32)
    print(f"\nSSL PCA-{pca_dim} expl_var={pca.explained_variance_ratio_.sum():.3f}")

    if use_mfcc:
        mfcc_sc = StandardScaler()
        mfcc_sc.fit(mfcc_feats[train_idx])
        x_mfcc = mfcc_sc.transform(mfcc_feats).astype(np.float32)
    else:
        x_mfcc = x_ssl   # fallback

    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[test_idx]   = True

    # ------------------------------------------------------------------
    # 4. Build hyperedge incidence matrix
    # ------------------------------------------------------------------
    n_mfcc = args.n_mfcc_clusters if use_mfcc else 0
    print(f"\nBuilding multi-relational hyperedges ...")
    print(f"  SSL clusters: {args.n_ssl_clusters}  |  MFCC clusters: {n_mfcc}")
    H, n_ssl_he, n_mfcc_he = build_hyperedge_incidence(
        x_ssl, x_mfcc, args.n_ssl_clusters, n_mfcc, args.seed)
    n_hyperedges = n_ssl_he + n_mfcc_he
    print(f"  Total hyperedges: {n_hyperedges}")

    # ------------------------------------------------------------------
    # 5. Compute HGNN Laplacian (factored form)
    # ------------------------------------------------------------------
    print(f"\nComputing HGNN propagation matrix ...")
    Theta_L, Theta_R = compute_hypergraph_laplacian(H, device)   # [N,M], [M,N]
    print(f"  Theta_L: {Theta_L.shape}  Theta_R: {Theta_R.shape}")

    X = torch.from_numpy(x_ssl).to(device)
    Y = torch.from_numpy(y).to(device)

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    node_in_dim = pca_dim
    model = HGNN2Layer(
        in_dim=node_in_dim, hidden=args.hgnn_hidden,
        out_dim=args.hgnn_out, num_classes=len(STUTTER_TYPES),
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nE3 model parameters: {n_params:,}")
    print(f"  HGNN: {node_in_dim}→{args.hgnn_hidden}→{args.hgnn_out}→5")
    print(f"  Hyperedges: {args.n_ssl_clusters} SSL + {n_mfcc} MFCC = {n_hyperedges} total")

    criterion = nn.MultiLabelSoftMarginLoss()
    opt       = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    # ------------------------------------------------------------------
    # 7. Training (full-batch, train mask only)
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs (full-batch HGNN) ...")
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "e3_best.pt"
    log_every = max(1, args.epochs // 20)

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(Theta_L, Theta_R, X)
        loss   = criterion(logits[train_mask], Y[train_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                logits_e = model(Theta_L, Theta_R, X)
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
    # 8. Final evaluation
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        logits_final = model(Theta_L, Theta_R, X)
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
    # 9. Save outputs
    # ------------------------------------------------------------------
    perclass_csv = args.out_dir / "e3_perclass_results.csv"
    with perclass_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    hist_csv = args.out_dir / "e3_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","macro_f1"])
        w.writeheader(); w.writerows(history)

    best_ep = history[int(np.argmax([r["macro_f1"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "E3",
        "title": "HGNN multi-relational hyperedges (SSL + MFCC clusters, HuBERT-large PCA-32)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "ssl_pca_dim": int(pca_dim),
        "n_ssl_clusters": args.n_ssl_clusters, "n_mfcc_clusters": n_mfcc,
        "n_hyperedges": int(n_hyperedges), "n_params": int(n_params), "n_nodes": int(n),
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
    report_json = args.out_dir / "e3_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 10. Figures
    # ------------------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep_x = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val")
        axes[0].set_title(f"E3 HGNN Loss (SSL={args.n_ssl_clusters} + MFCC={n_mfcc} hyperedges)")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(ep_x, [r["macro_f1"] for r in history], color="purple", marker="o", ms=4)
        axes[1].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("E3 Validation Macro-F1"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "e3_train_curves.png", dpi=160); plt.close(fig)
        print(f"  Saved: {args.fig_dir / 'e3_train_curves.png'}")

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
        ax2.set_title(f"E3 Per-class F1 (HGNN, {n_hyperedges} hyperedges)")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "e3_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"  Saved: {args.fig_dir / 'e3_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  E3 complete.")
    print(f"   Report    : {report_json}")
    print(f"   Per-class : {perclass_csv}")
    print(f"   History   : {hist_csv}")
    print(f"   Ckpt      : {best_ckpt}")


if __name__ == "__main__":
    main()
