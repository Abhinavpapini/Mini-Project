"""D8: Prototypical Network for few-shot-style stutter classification.

Prototypical Networks learn an embedding space where each class is
represented by its prototype (centroid of support examples). At inference,
queries are classified by nearest prototype (Euclidean distance).

Here we apply it to the *full* SEP-28k dataset with episodic training:
  - Each episode: sample N-way K-shot from class subsets
  - Support set → class prototypes (mean embedding)
  - Query set → nearest prototype classification

Benefits for stuttering:
  - Naturally handles class imbalance (each class gets K shots per episode)
  - Improves minority class (WordRep, SoundRep) by episodic balancing
  - Produces interpretable embedding space (inspectable via t-SNE/UMAP)
  - Does not require softmax over fixed classes — generalises to new types

Design choices:
  - 5-way (one class per stutter type) K=8 shot episodes
  - Shared CNN embedding network → 128-dim embedding space
  - Euclidean distance to prototypes → softmax → cross-entropy
  - Evaluation: standard multi-label by running each class as 1-way vs rest

Run command:
    python experiments/D8_prototypical_net.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --ssl-pca-dim 32 \
        --emb-dim 128 \
        --n-way 5 \
        --k-shot 8 \
        --q-queries 16 \
        --episodes 2000 \
        --eval-episodes 400 \
        --lr 1e-3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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


STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D8: Prototypical Network for stutter classification")
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
    p.add_argument("--ssl-pca-dim",    type=int, default=32)
    # Proto-net params
    p.add_argument("--emb-dim",        type=int, default=128,
                   help="Embedding space dimensionality")
    p.add_argument("--n-way",          type=int, default=5,
                   help="Number of classes per episode (5 = all stutter types)")
    p.add_argument("--k-shot",         type=int, default=8,
                   help="Support examples per class per episode")
    p.add_argument("--q-queries",      type=int, default=16,
                   help="Query examples per class per episode")
    p.add_argument("--episodes",       type=int, default=2000,
                   help="Training episodes")
    p.add_argument("--eval-episodes",  type=int, default=400,
                   help="Evaluation episodes")
    p.add_argument("--dropout",        type=float, default=0.2)
    p.add_argument("--test-size",      type=float, default=0.20)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--threshold",      type=float, default=0.5)
    p.add_argument("--out-dir",  type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",  type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/D8"))
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
# Embedding network (shared across all episodes)
# ---------------------------------------------------------------------------

class EmbeddingNet(nn.Module):
    """
    Projects PCA features into a metric space where stutter types cluster.
    Uses same CNN-1D style as the best-performing F1/D1 backbone.
    """

    def __init__(self, in_dim: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=7, padding=3), nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x.unsqueeze(1)))   # [B, emb_dim]


# ---------------------------------------------------------------------------
# Episodic sampler
# ---------------------------------------------------------------------------

def sample_episode(
    per_class_idx: List[np.ndarray],
    selected_classes: List[int],
    k_shot: int,
    q_queries: int,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Samples one episode.
    Returns:
      support_x [N*K, F], support_y [N*K] (class indices)
      query_x   [N*Q, F], query_y   [N*Q] (class indices)
    """
    sx, sy, qx, qy = [], [], [], []
    for cls_idx, cls in enumerate(selected_classes):
        pool = per_class_idx[cls]
        if len(pool) < k_shot + q_queries:
            chosen = np.random.choice(pool, k_shot + q_queries, replace=True)
        else:
            chosen = np.random.choice(pool, k_shot + q_queries, replace=False)
        sx.append(x[chosen[:k_shot]])
        sy.extend([cls_idx] * k_shot)
        qx.append(x[chosen[k_shot:k_shot + q_queries]])
        qy.extend([cls_idx] * q_queries)
    return (np.vstack(sx).astype(np.float32), np.array(sy),
            np.vstack(qx).astype(np.float32), np.array(qy))


# ---------------------------------------------------------------------------
# Prototypical loss
# ---------------------------------------------------------------------------

def prototypical_loss(
    emb_net: EmbeddingNet,
    support_x: torch.Tensor, support_y: torch.Tensor,
    query_x: torch.Tensor,   query_y: torch.Tensor,
    n_way: int,
) -> Tuple[torch.Tensor, float]:
    support_emb = emb_net(support_x)   # [N*K, D]
    query_emb   = emb_net(query_x)     # [N*Q, D]

    # Compute prototypes (mean embedding per class)
    prototypes = torch.stack([
        support_emb[support_y == c].mean(dim=0) for c in range(n_way)
    ])  # [N, D]

    # Euclidean distance from each query to each prototype
    dists = torch.cdist(query_emb, prototypes)   # [N*Q, N]
    log_p = F.log_softmax(-dists, dim=1)          # [N*Q, N]
    loss  = F.nll_loss(log_p, query_y)

    # Episode accuracy
    preds  = log_p.argmax(dim=1)
    acc    = (preds == query_y).float().mean().item()
    return loss, acc


# ---------------------------------------------------------------------------
# Standard multi-label evaluation (after proto-net training)
# ---------------------------------------------------------------------------

def evaluate_multilabel_from_embeddings(
    emb_net: EmbeddingNet,
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_te: np.ndarray, y_te: np.ndarray,
    device: torch.device,
    threshold: float,
    batch_size: int = 512,
) -> Dict[str, float]:
    """
    Use the trained embedding net + class prototypes from train set pooled mean.
    For each stutter type, compute prototype of positive vs negative class,
    then classify test samples by distance ratio → binary F1.
    """
    emb_net.eval()
    def get_embs(x: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                xb = torch.from_numpy(x[i:i+batch_size]).to(device)
                out.append(emb_net(xb).cpu().numpy())
        return np.vstack(out)

    tr_emb = get_embs(x_tr)
    te_emb = get_embs(x_te)

    m: Dict[str, float] = {}
    pf1, ppre, prec = [], [], []
    auprc_list = []

    for i, t in enumerate(STUTTER_TYPES):
        pos_mask = y_tr[:, i] > 0
        neg_mask = ~pos_mask
        if pos_mask.sum() < 2 or neg_mask.sum() < 2:
            pf1.append(0.0); ppre.append(0.0); prec.append(0.0); auprc_list.append(0.0)
            continue

        proto_pos = tr_emb[pos_mask].mean(axis=0)   # [D]
        proto_neg = tr_emb[neg_mask].mean(axis=0)   # [D]

        # Distance to pos vs neg prototype → score
        d_pos = np.linalg.norm(te_emb - proto_pos, axis=1)
        d_neg = np.linalg.norm(te_emb - proto_neg, axis=1)
        # Score: higher = more likely positive (closer to pos prototype)
        score = d_neg - d_pos   # positive if closer to pos than neg
        # Normalise to [0,1]
        score_norm = (score - score.min()) / (score.max() - score.min() + 1e-8)

        y_pred = (score_norm >= threshold).astype(int)
        y_true_i = y_te[:, i].astype(int)

        pf1.append(float(f1_score(y_true_i, y_pred, zero_division=0)))
        ppre.append(float(precision_score(y_true_i, y_pred, zero_division=0)))
        prec.append(float(recall_score(y_true_i, y_pred, zero_division=0)))
        ap = float(average_precision_score(y_true_i, score_norm)) if y_true_i.sum() > 0 else 0.0
        auprc_list.append(ap)

        m[f"f1_{t}"] = pf1[-1]; m[f"pre_{t}"] = ppre[-1]
        m[f"rec_{t}"] = prec[-1]; m[f"auprc_{t}"] = auprc_list[-1]

    m["macro_f1"]    = float(np.mean(pf1))
    m["macro_pre"]   = float(np.mean(ppre))
    m["macro_rec"]   = float(np.mean(prec))
    m["macro_auprc"] = float(np.mean(auprc_list))

    # Also compute micro-F1 using binary prediction matrix
    y_pred_mat = np.zeros_like(y_te, dtype=int)
    for i, t in enumerate(STUTTER_TYPES):
        pos_mask = y_tr[:, i] > 0; neg_mask = ~pos_mask
        if pos_mask.sum() < 2 or neg_mask.sum() < 2:
            continue
        proto_pos = tr_emb[pos_mask].mean(axis=0)
        proto_neg = tr_emb[neg_mask].mean(axis=0)
        d_pos = np.linalg.norm(te_emb - proto_pos, axis=1)
        d_neg = np.linalg.norm(te_emb - proto_neg, axis=1)
        score = (d_neg - d_pos)
        score_norm = (score - score.min()) / (score.max() - score.min() + 1e-8)
        y_pred_mat[:, i] = (score_norm >= threshold).astype(int)
    m["micro_f1"] = float(f1_score(y_te.astype(int), y_pred_mat, average="micro", zero_division=0))
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    for d in (args.out_dir, args.fig_dir, args.ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
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
    # 2. Split + PCA
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

    # Per-class sample pools for episodic sampling
    per_class_tr = [np.where(y_tr[:, i] > 0)[0] for i in range(len(STUTTER_TYPES))]
    print(f"\nEpisodic training: {args.n_way}-way {args.k_shot}-shot, "
          f"{args.q_queries} queries/class")
    for i, t in enumerate(STUTTER_TYPES):
        print(f"  {t:15s}: {len(per_class_tr[i]):5d} training support examples")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    emb_net = EmbeddingNet(in_dim=pca_dim, emb_dim=args.emb_dim, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in emb_net.parameters())
    print(f"\nD8 embedding net parameters: {n_params:,}")
    print(f"  CNN → {args.emb_dim}-dim embedding space | {args.n_way}-way {args.k_shot}-shot")

    opt = torch.optim.Adam(emb_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)

    # ------------------------------------------------------------------
    # 4. Episodic training
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.episodes} episodes ...")
    all_classes = list(range(args.n_way))  # 0-4 for 5 stutter types
    history: List[Dict] = []
    best_acc   = -1.0
    best_ckpt  = args.ckpt_dir / "d8_best.pt"
    log_every  = max(1, args.episodes // 20)

    ep_losses, ep_accs = [], []
    for ep in range(1, args.episodes + 1):
        emb_net.train()
        sx, sy, qx, qy = sample_episode(per_class_tr, all_classes,
                                         args.k_shot, args.q_queries, x_tr)
        sx_t = torch.from_numpy(sx).to(device)
        sy_t = torch.from_numpy(sy).to(device)
        qx_t = torch.from_numpy(qx).to(device)
        qy_t = torch.from_numpy(qy).to(device)

        opt.zero_grad()
        loss, acc = prototypical_loss(emb_net, sx_t, sy_t, qx_t, qy_t, args.n_way)
        loss.backward()
        nn.utils.clip_grad_norm_(emb_net.parameters(), 1.0)
        opt.step()
        scheduler.step()

        ep_losses.append(loss.item()); ep_accs.append(acc)

        if ep % log_every == 0:
            mean_loss = np.mean(ep_losses[-log_every:])
            mean_acc  = np.mean(ep_accs[-log_every:])
            row = {"episode": ep, "mean_loss": round(mean_loss, 6),
                   "mean_acc": round(mean_acc, 6)}
            history.append(row)
            print(f"ep={ep:5d}  loss={mean_loss:.5f}  ep_acc={mean_acc:.5f}")
            if mean_acc > best_acc:
                best_acc = mean_acc
                torch.save(emb_net.state_dict(), best_ckpt)

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------
    print(f"\nEvaluating with prototype-based classification ...")
    emb_net.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_m = evaluate_multilabel_from_embeddings(
        emb_net, x_tr, y_tr, x_te, y_te, device, args.threshold
    )

    print(f"\n--- Test Results (prototype-based, best ep_acc={best_acc:.5f}) ---")
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
    # 6. Save outputs
    # ------------------------------------------------------------------
    perclass_csv = args.out_dir / "d8_perclass_results.csv"
    with perclass_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": test_m[f"f1_{t}"],
                        "precision": test_m[f"pre_{t}"],
                        "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]})

    hist_csv = args.out_dir / "d8_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode","mean_loss","mean_acc"])
        w.writeheader(); w.writerows(history)

    run_report = {
        "experiment": "D8", "title": "Prototypical Network (HuBERT-large PCA-32, 5-way K-shot)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "ssl_pca_dim": int(pca_dim),
        "emb_dim": args.emb_dim, "n_way": args.n_way, "k_shot": args.k_shot,
        "q_queries": args.q_queries, "episodes": args.episodes, "n_params": int(n_params),
        "best_episode_acc": round(best_acc, 6), "threshold": args.threshold,
        "macro_f1": test_m["macro_f1"], "micro_f1": test_m["micro_f1"],
        "macro_precision": test_m["macro_pre"], "macro_recall": test_m["macro_rec"],
        "macro_auprc": test_m["macro_auprc"],
        "per_class": {t: {"f1": test_m[f"f1_{t}"], "precision": test_m[f"pre_{t}"],
                          "recall": test_m[f"rec_{t}"], "auprc": test_m[f"auprc_{t}"]}
                      for t in STUTTER_TYPES},
        "best_checkpoint": str(best_ckpt),
    }
    report_json = args.out_dir / "d8_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 7. Figures
    # ------------------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep_x   = [r["episode"]  for r in history]
        acc_y  = [r["mean_acc"] for r in history]
        loss_y = [r["mean_loss"] for r in history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ep_x, loss_y, color="steelblue")
        axes[0].set_title("D8 Episode Loss (Prototypical)"); axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Prototypical Loss"); axes[0].grid(alpha=0.3)

        axes[1].plot(ep_x, acc_y, color="green", marker="o", ms=4)
        axes[1].axhline(best_acc, ls="--", color="red", alpha=0.6, label=f"best={best_acc:.4f}")
        axes[1].set_title("D8 Episode Accuracy (5-way)"); axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "d8_train_curves.png", dpi=160); plt.close(fig)
        print(f"  Saved: {args.fig_dir / 'd8_train_curves.png'}")

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        pf1  = [test_m[f"f1_{t}"] for t in STUTTER_TYPES]
        bars = ax2.bar(STUTTER_TYPES, pf1,
                       color=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"], width=0.55)
        ax2.axhline(test_m["macro_f1"], ls="--", color="black", alpha=0.5,
                    label=f"Macro-F1={test_m['macro_f1']:.4f}")
        for b, v in zip(bars, pf1):
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax2.set_ylim(0, 1); ax2.set_title("D8 Per-class F1 (Prototype-based)")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "d8_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"  Saved: {args.fig_dir / 'd8_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  D8 complete.")
    print(f"   Report    : {report_json}")
    print(f"   Per-class : {perclass_csv}")
    print(f"   History   : {hist_csv}")
    print(f"   Ckpt      : {best_ckpt}")


if __name__ == "__main__":
    main()
