"""D5: Multi-task stutter detection + type classification.

Architecture:
  HuBERT-large (PCA-32) + MFCC-21 → [B, 53]
  → Shared CNN-1D backbone → [B, 128]
  → Head 1: Linear(128→1)   BCE loss  → fluent/stutter detection
  → Head 2: Linear(128→5)   MLSM loss → stutter type (5-class multi-label)

Key idea: jointly training detection + classification regularises the shared
encoder, forcing it to learn general stutter-discriminative features rather
than type-specific shortcuts.

Loss: L_total = λ₁·L_detect + λ₂·L_types
  L_detect computed on all samples (fluent + stutter)
  L_types  computed only on stutter samples (mask out fluent clips)

Run command:
    python experiments/D5_multitask_cnn.py \
        --hubert-alias hubert-large \
        --hubert-layer 21 \
        --mfcc-cache artifacts/features/mfcc/fold0/mfcc_stats.npy \
        --mfcc-dim 21 \
        --ssl-pca-dim 32 \
        --epochs 30 \
        --batch-size 256 \
        --lr 3e-4 \
        --lambda-detect 1.0 \
        --lambda-type 1.0
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
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D5: Multi-task detection + type CNN")
    p.add_argument("--features-root",  type=Path, default=Path("artifacts/features"))
    p.add_argument("--hubert-alias",   type=str,  default="hubert-large")
    p.add_argument("--hubert-layer",   type=int,  default=21)
    p.add_argument("--fold",           type=str,  default="fold0")
    p.add_argument("--mfcc-cache",     type=Path,
                   default=Path("artifacts/features/mfcc/fold0/mfcc_stats.npy"))
    p.add_argument("--mfcc-dim",       type=int,  default=21)
    p.add_argument("--clips-root",     type=Path,
                   default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",     type=Path,
                   default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path,
                   default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--ssl-pca-dim",    type=int,  default=32)
    p.add_argument("--lambda-detect",  type=float,default=1.0,
                   help="Weight for detection loss (head 1)")
    p.add_argument("--lambda-type",    type=float,default=1.0,
                   help="Weight for type loss (head 2)")
    p.add_argument("--test-size",      type=float,default=0.20)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--epochs",         type=int,  default=30)
    p.add_argument("--batch-size",     type=int,  default=256)
    p.add_argument("--lr",             type=float,default=3e-4)
    p.add_argument("--weight-decay",   type=float,default=1e-4)
    p.add_argument("--dropout",        type=float,default=0.2)
    p.add_argument("--threshold",      type=float,default=0.5)
    p.add_argument("--out-dir",  type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",  type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/checkpoints/D5"))
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


def compute_sample_weights(y_detect: np.ndarray) -> np.ndarray:
    """Weight by fluent/stutter ratio — oversample stutter clips."""
    n_stutter = y_detect.sum()
    n_fluent  = len(y_detect) - n_stutter
    w = np.where(y_detect == 1, n_fluent / max(n_stutter, 1),
                 n_stutter / max(n_fluent, 1)).astype(np.float32)
    return w / w.min()


# ---------------------------------------------------------------------------
# D5 Model: Shared CNN backbone + 2 heads
# ---------------------------------------------------------------------------

class D5MultiTaskCNN(nn.Module):
    """
    Shared CNN-1D encoder → 2 task heads:
      Head 1 (detect) : [B, 128] → Linear → [B, 1]  binary fluent/stutter
      Head 2 (type)   : [B, 128] → Linear → [B, 5]  multi-label stutter type
    """

    def __init__(self, in_dim: int, dropout: float) -> None:
        super().__init__()
        # Shared backbone (3-layer CNN)
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        # Head 1: binary detection (fluent=0, stutter=1)
        self.head_detect = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Head 2: 5-class stutter type (multi-label)
        self.head_type = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, len(STUTTER_TYPES)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_mlp(self.backbone(x.unsqueeze(1)))
        return self.head_detect(shared), self.head_type(shared)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_detection(y_true: np.ndarray, logits: np.ndarray, threshold: float) -> Dict:
    probs  = 1 / (1 + np.exp(-logits.squeeze()))
    preds  = (probs >= threshold).astype(int)
    f1  = float(f1_score(y_true, preds, zero_division=0))
    pre = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    try: auc = float(roc_auc_score(y_true, probs))
    except: auc = float("nan")
    return {"detect_f1": f1, "detect_precision": pre, "detect_recall": rec, "detect_auroc": auc}


def eval_multilabel(y_true: np.ndarray, logits: np.ndarray,
                    threshold: float) -> Dict[str, float]:
    probs  = 1 / (1 + np.exp(-logits))
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
    print(f"  SSL features: {ssl_feats.shape}")

    print(f"Loading MFCC cache: {args.mfcc_cache}")
    mfcc_all = np.load(args.mfcc_cache)
    if mfcc_all.ndim != 2:
        mfcc_all = mfcc_all.reshape(mfcc_all.shape[0], -1)
    mfcc_feats = mfcc_all[:, :args.mfcc_dim]
    print(f"  MFCC features (first {args.mfcc_dim} dims): {mfcc_feats.shape}")

    n = min(ssl_feats.shape[0], mfcc_feats.shape[0])
    ssl_feats = ssl_feats[:n]; mfcc_feats = mfcc_feats[:n]

    # ------------------------------------------------------------------
    # 2. Labels
    # ------------------------------------------------------------------
    label_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    label_map.update(load_multilabel_map(args.sep_labels))
    label_map.update(load_multilabel_map(args.fluency_labels))
    clip_keys = sorted_clip_keys(args.clips_root)[:n]
    y_types = np.array(
        [label_map.get(k, np.zeros(len(STUTTER_TYPES), dtype=np.float32)) for k in clip_keys],
        dtype=np.float32,
    )
    # Detection label: 1 if any stutter type present, 0 if all zeros (fluent)
    y_detect = (y_types.sum(axis=1) > 0).astype(np.float32)
    n_stutter = int(y_detect.sum()); n_fluent = int(n - n_stutter)

    print(f"\nDataset: {n} samples | Label distribution:")
    for i, t in enumerate(STUTTER_TYPES):
        pos = int(y_types[:, i].sum())
        print(f"  {t:15s}: {pos:5d} pos  ({pos/n:.2%})")
    print(f"  Fluent (all 0) : {n_fluent:5d}  ({n_fluent/n:.2%})")
    print(f"\nDetection labels: {n_stutter} stutter | {n_fluent} fluent")

    # ------------------------------------------------------------------
    # 3. Split + PCA
    # ------------------------------------------------------------------
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed,
        stratify=y_detect.astype(int)
    )
    pca_dim = min(args.ssl_pca_dim, ssl_feats.shape[1])
    ssl_sc  = StandardScaler(); pca = PCA(n_components=pca_dim, random_state=args.seed)
    x_ssl_tr = pca.fit_transform(ssl_sc.fit_transform(ssl_feats[train_idx])).astype(np.float32)
    x_ssl_te = pca.transform(ssl_sc.transform(ssl_feats[test_idx])).astype(np.float32)
    mfcc_sc  = StandardScaler()
    x_mfcc_tr = mfcc_sc.fit_transform(mfcc_feats[train_idx]).astype(np.float32)
    x_mfcc_te = mfcc_sc.transform(mfcc_feats[test_idx]).astype(np.float32)
    x_tr = np.hstack([x_ssl_tr, x_mfcc_tr])
    x_te = np.hstack([x_ssl_te, x_mfcc_te])
    in_dim = x_tr.shape[1]
    print(f"\nSSL PCA-{pca_dim} explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"Fused feature dim: {in_dim}  (SSL {pca_dim} + MFCC {args.mfcc_dim})")

    yd_tr = y_detect[train_idx]; yt_tr = y_types[train_idx]
    yd_te = y_detect[test_idx];  yt_te = y_types[test_idx]

    # ------------------------------------------------------------------
    # 4. Sampler + DataLoaders
    # ------------------------------------------------------------------
    weights  = compute_sample_weights(yd_tr)
    sampler  = WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True)
    train_ds = TensorDataset(
        torch.from_numpy(x_tr),
        torch.from_numpy(yd_tr).unsqueeze(1),
        torch.from_numpy(yt_tr),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_te),
        torch.from_numpy(yd_te).unsqueeze(1),
        torch.from_numpy(yt_te),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

    # ------------------------------------------------------------------
    # 5. Model + losses
    # ------------------------------------------------------------------
    model = D5MultiTaskCNN(in_dim=in_dim, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nD5 model parameters: {n_params:,}")
    print(f"  Shared CNN → Head1 (detect BCE) + Head2 (type MLSM)")
    print(f"  λ_detect={args.lambda_detect}  λ_type={args.lambda_type}")

    # Pos weight for detection: balance fluent/stutter
    n_stutter_tr = int(yd_tr.sum()); n_fluent_tr = int(len(yd_tr) - n_stutter_tr)
    detect_pos_w = torch.tensor([n_fluent_tr / max(n_stutter_tr, 1)], device=device)
    crit_detect = nn.BCEWithLogitsLoss(pos_weight=detect_pos_w)
    crit_type   = nn.MultiLabelSoftMarginLoss()

    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    # ------------------------------------------------------------------
    # 6. Training
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs ...")
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "d5_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_det = tr_typ = 0.0
        for xb, yd_b, yt_b in train_dl:
            xb  = xb.to(device)
            yd_b = yd_b.to(device)
            yt_b = yt_b.to(device)
            opt.zero_grad()
            logit_det, logit_typ = model(xb)

            l_det = crit_detect(logit_det, yd_b)

            # Type loss only on stutter samples
            stutter_mask = (yd_b.squeeze(1) > 0)
            if stutter_mask.sum() > 0:
                l_typ = crit_type(logit_typ[stutter_mask], yt_b[stutter_mask])
            else:
                l_typ = torch.tensor(0.0, device=device)

            loss = args.lambda_detect * l_det + args.lambda_type * l_typ
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_det  += l_det.item() * bs
            tr_typ  += l_typ.item() * bs
        tr_loss /= len(train_ds); tr_det /= len(train_ds); tr_typ /= len(train_ds)
        scheduler.step()

        model.eval()
        vl = 0.0
        v_det_log, v_det_true, v_typ_log, v_typ_true = [], [], [], []
        with torch.no_grad():
            for xb, yd_b, yt_b in test_dl:
                xb, yd_b, yt_b = xb.to(device), yd_b.to(device), yt_b.to(device)
                ld, lt = model(xb)
                l_d = crit_detect(ld, yd_b)
                sm  = (yd_b.squeeze(1) > 0)
                l_t = crit_type(lt[sm], yt_b[sm]) if sm.sum() > 0 else torch.tensor(0.0, device=device)
                vl += (args.lambda_detect * l_d + args.lambda_type * l_t).item() * xb.size(0)
                v_det_log.append(ld.cpu().numpy()); v_det_true.append(yd_b.cpu().numpy())
                v_typ_log.append(lt.cpu().numpy()); v_typ_true.append(yt_b.cpu().numpy())
        vl /= len(test_ds)

        det_m = eval_detection(np.concatenate(v_det_true).squeeze(),
                               np.concatenate(v_det_log), args.threshold)
        typ_m = eval_multilabel(np.concatenate(v_typ_true),
                                np.concatenate(v_typ_log), args.threshold)
        macro_f1 = typ_m["macro_f1"]

        row = {"epoch": epoch, "train_loss": round(tr_loss, 6),
               "loss_detect": round(tr_det, 6), "loss_type": round(tr_typ, 6),
               "val_loss": round(vl, 6), "macro_f1_type": round(macro_f1, 6),
               "detect_f1": round(det_m["detect_f1"], 6)}
        history.append(row)
        print(f"epoch={epoch:02d}  tr_loss={tr_loss:.5f}  "
              f"det={tr_det:.4f}  typ={tr_typ:.4f}  "
              f"val_loss={vl:.5f}  macro_f1={macro_f1:.5f}  det_f1={det_m['detect_f1']:.5f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), best_ckpt)

    # ------------------------------------------------------------------
    # 7. Final evaluation
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    t_det_log, t_det_true, t_typ_log, t_typ_true = [], [], [], []
    with torch.no_grad():
        for xb, yd_b, yt_b in test_dl:
            ld, lt = model(xb.to(device))
            t_det_log.append(ld.cpu().numpy()); t_det_true.append(yd_b.cpu().numpy())
            t_typ_log.append(lt.cpu().numpy()); t_typ_true.append(yt_b.cpu().numpy())

    det_m = eval_detection(np.concatenate(t_det_true).squeeze(),
                           np.concatenate(t_det_log), args.threshold)
    typ_m = eval_multilabel(np.concatenate(t_typ_true),
                            np.concatenate(t_typ_log), args.threshold)

    print(f"\n--- Test Results (best ckpt, type macro_f1={best_macro_f1:.5f}) ---")
    print(f"\n  [Head 1 — Detection]")
    print(f"  Detect-F1   : {det_m['detect_f1']:.5f}")
    print(f"  Detect-Pre  : {det_m['detect_precision']:.5f}")
    print(f"  Detect-Rec  : {det_m['detect_recall']:.5f}")
    print(f"  Detect-AUROC: {det_m['detect_auroc']:.5f}")
    print(f"\n  [Head 2 — Type Classification]")
    print(f"  Macro-F1   : {typ_m['macro_f1']:.5f}")
    print(f"  Micro-F1   : {typ_m['micro_f1']:.5f}")
    print(f"  Macro-Pre  : {typ_m['macro_pre']:.5f}")
    print(f"  Macro-Rec  : {typ_m['macro_rec']:.5f}")
    print(f"  Macro-AUPRC: {typ_m['macro_auprc']:.5f}")
    print(f"\n  Per-class F1 (type head):")
    for t in STUTTER_TYPES:
        print(f"    {t:15s}: F1={typ_m[f'f1_{t}']:.5f}  "
              f"P={typ_m[f'pre_{t}']:.5f}  R={typ_m[f'rec_{t}']:.5f}  "
              f"AUPRC={typ_m[f'auprc_{t}']:.5f}")

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    hist_csv = args.out_dir / "d5_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)

    perclass_csv = args.out_dir / "d5_perclass_results.csv"
    with perclass_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stutter_type","f1","precision","recall","auprc"])
        w.writeheader()
        for t in STUTTER_TYPES:
            w.writerow({"stutter_type": t, "f1": typ_m[f"f1_{t}"],
                        "precision": typ_m[f"pre_{t}"],
                        "recall": typ_m[f"rec_{t}"], "auprc": typ_m[f"auprc_{t}"]})

    best_ep = history[int(np.argmax([r["macro_f1_type"] for r in history]))]["epoch"]
    run_report = {
        "experiment": "D5",
        "title": "Multi-task CNN: detection + type classification (HuBERT+MFCC)",
        "device": str(device), "hubert_alias": args.hubert_alias,
        "hubert_layer": args.hubert_layer, "ssl_pca_dim": int(pca_dim),
        "mfcc_dim": args.mfcc_dim, "fused_dim": int(in_dim),
        "n_params": int(n_params), "lambda_detect": args.lambda_detect,
        "lambda_type": args.lambda_type, "n_samples": int(n),
        "n_stutter": int(n_stutter), "n_fluent": int(n_fluent),
        "epochs": args.epochs, "best_epoch": int(best_ep),
        "loss_detect": "BCEWithLogitsLoss", "loss_type": "MultiLabelSoftMarginLoss",
        "head1_detection": det_m,
        "head2_type": {
            "macro_f1": typ_m["macro_f1"], "micro_f1": typ_m["micro_f1"],
            "macro_precision": typ_m["macro_pre"], "macro_recall": typ_m["macro_rec"],
            "macro_auprc": typ_m["macro_auprc"],
            "per_class": {t: {"f1": typ_m[f"f1_{t}"], "precision": typ_m[f"pre_{t}"],
                              "recall": typ_m[f"rec_{t}"], "auprc": typ_m[f"auprc_{t}"]}
                          for t in STUTTER_TYPES},
        },
        "best_checkpoint": str(best_ckpt),
    }
    report_json = args.out_dir / "d5_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 9. Figures
    # ------------------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_x    = [r["epoch"] for r in history]
        best_ep_idx = int(np.argmax([r["macro_f1_type"] for r in history]))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].plot(epochs_x, [r["loss_detect"] for r in history], label="detect loss")
        axes[0].plot(epochs_x, [r["loss_type"]   for r in history], label="type loss")
        axes[0].set_title("D5 Task Losses"); axes[0].set_xlabel("Epoch")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs_x, [r["detect_f1"] for r in history], color="orange", marker="o", ms=3)
        axes[1].set_title("D5 Detection F1 (Head 1)"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1"); axes[1].grid(alpha=0.3)

        axes[2].plot(epochs_x, [r["macro_f1_type"] for r in history], color="green", marker="o", ms=3)
        axes[2].axvline(x=history[best_ep_idx]["epoch"], ls="--", color="red", alpha=0.6,
                        label=f"best={best_macro_f1:.4f}")
        axes[2].set_title("D5 Type Macro-F1 (Head 2)"); axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Macro-F1"); axes[2].legend(); axes[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "d5_train_curves.png", dpi=160); plt.close(fig)
        print(f"  Saved: {args.fig_dir / 'd5_train_curves.png'}")

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        pf1  = [typ_m[f"f1_{t}"] for t in STUTTER_TYPES]
        bars = ax2.bar(STUTTER_TYPES, pf1,
                       color=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"], width=0.55)
        ax2.axhline(typ_m["macro_f1"], ls="--", color="black", alpha=0.5,
                    label=f"Macro-F1={typ_m['macro_f1']:.4f}")
        for b, v in zip(bars, pf1):
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax2.set_ylim(0, 1); ax2.set_title("D5 Per-class F1 (Type Head)")
        ax2.set_xlabel("Stutter Type"); ax2.set_ylabel("F1"); ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "d5_perclass_f1.png", dpi=160); plt.close(fig2)
        print(f"  Saved: {args.fig_dir / 'd5_perclass_f1.png'}")
    except Exception as exc:
        print(f"  [WARN] Figure generation failed: {exc}")

    print("\n✅  D5 complete.")
    print(f"   Report    : {report_json}")
    print(f"   Per-class : {perclass_csv}")
    print(f"   History   : {hist_csv}")
    print(f"   Ckpt      : {best_ckpt}")


if __name__ == "__main__":
    main()
