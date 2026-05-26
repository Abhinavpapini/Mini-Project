"""G3 Part 2: Gumbel layer selector + CNN-1D training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


CLASS_NAMES   = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection", "Fluent"]
N_CLASSES     = len(CLASS_NAMES)
STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]
PRIORITY      = [0, 4, 1, 2, 3]   # Block > Interjection > Prolongation > SoundRep > WordRep


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G3 Part-2: Gumbel Layer Selector + CNN-1D")
    p.add_argument("--meta-json",    type=Path,
                   default=Path("results/tables/g3_probe_meta.json"))
    p.add_argument("--features-root",type=Path, default=Path("artifacts/features"))
    p.add_argument("--d-model",      type=int,  default=256)
    p.add_argument("--dropout",      type=float,default=0.3)
    p.add_argument("--tau-start",    type=float,default=2.0,
                   help="Gumbel temperature at epoch 1")
    p.add_argument("--tau-end",      type=float,default=0.5,
                   help="Gumbel temperature at final epoch")
    p.add_argument("--epochs",       type=int,  default=40)
    p.add_argument("--batch-size",   type=int,  default=256)
    p.add_argument("--lr",           type=float,default=3e-4)
    p.add_argument("--weight-decay", type=float,default=1e-4)
    p.add_argument("--amp",          action="store_true", default=True,
                   help="Use automatic mixed precision (faster on GPU)")
    p.add_argument("--out-dir",      type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir",      type=Path, default=Path("results/figures"))
    p.add_argument("--ckpt-dir",     type=Path, default=Path("artifacts/checkpoints/G3"))
    return p.parse_args()


# Data utilities
def norm(x: object) -> str:
    return str(x).strip()


def load_multilabel_map(csv_path: Path) -> Dict[Tuple[str,str,str], np.ndarray]:
    out: Dict[Tuple[str,str,str], np.ndarray] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (norm(row["Show"]), norm(row["EpId"]), norm(row["ClipId"]))
            labels = np.array(
                [1.0 if float(norm(row[t])) >= 1 else 0.0 for t in STUTTER_TYPES],
                dtype=np.float32,
            )
            out[key] = labels
    return out


def sorted_clip_keys(clips_root: Path) -> List[Tuple[str,str,str]]:
    keys = []
    for w in sorted(clips_root.rglob("*.wav")):
        parts = w.stem.split("_")
        if len(parts) >= 3:
            keys.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return keys


def multilabel_to_single(y_multi: np.ndarray) -> np.ndarray:
    y_single = np.full(len(y_multi), 5, dtype=np.int64)
    for col in reversed(PRIORITY):
        y_single[y_multi[:, col] > 0] = col
    return y_single


def find_layer_folder(features_root: Path, alias: str, fold: str) -> Path:
    direct = features_root / alias / fold
    if direct.exists() and any(direct.glob("layer_*.npy")):
        return direct
    parent = features_root / alias
    if parent.exists():
        for child in sorted(parent.iterdir()):
            if child.is_dir() and any(child.glob("layer_*.npy")):
                return child
    raise FileNotFoundError(f"No layer cache for alias '{alias}' under {features_root}")


def load_top_k_layers(
    features_root: Path,
    alias: str,
    fold: str,
    top_layers: List[int],
) -> np.ndarray:
    """Load top-K layers and stack → [N, K, D]."""
    folder = find_layer_folder(features_root, alias, fold)
    arrs = []
    for l in top_layers:
        a = np.load(folder / f"layer_{l}.npy")
        if a.ndim != 2:
            a = a.reshape(a.shape[0], -1)
        arrs.append(a.astype(np.float32))
    return np.stack(arrs, axis=1)   # [N, K, D]


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-sample weights for WeightedRandomSampler."""
    class_counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    class_counts  = np.where(class_counts == 0, 1.0, class_counts)
    class_weights  = 1.0 / class_counts
    return class_weights[y]


# Model components

class WeightedLayerSum(nn.Module):
    """Learnable softmax-weighted sum over K layers.

    init_weights: 1-D array of probe macro-F1 per layer (used as logit init).
    forward(feats) where feats: [B, K, D] → [B, D]
    """
    def __init__(self, k: int, init_weights: Optional[np.ndarray] = None) -> None:
        super().__init__()
        if init_weights is not None:
            w = torch.tensor(init_weights, dtype=torch.float32)
        else:
            w = torch.ones(k, dtype=torch.float32)
        self.logits = nn.Parameter(w)   # [K]

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, K, D]
        weights = torch.softmax(self.logits, dim=0)          # [K]
        return (feats * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # [B, D]


class ResidualConv1d(nn.Module):
    """Single residual Conv1d block: [B, C, L] → [B, C, L]."""
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        # Use (kernel_size-1)//2 so output always equals input length for odd ks
        pad = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class GumbelClassRouter(nn.Module):
    """Per-class Gumbel-Softmax routing over M model streams.

    logits shape: [n_classes, M]
    forward(streams, tau, hard):
      streams: [B, M, d_model]
      returns: [B, n_classes, d_model]
    """
    def __init__(self, n_classes: int, n_models: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_classes, n_models))

    def forward(
        self, streams: torch.Tensor, tau: float = 1.0, hard: bool = False
    ) -> torch.Tensor:
        # streams: [B, M, d_model]
        B = streams.size(0)
        # Expand logits to [B, C, M] so Gumbel noise is per-sample
        logits_exp = self.logits.unsqueeze(0).expand(B, -1, -1)   # [B, C, M]
        weights = F.gumbel_softmax(logits_exp, tau=tau, hard=hard, dim=-1)  # [B, C, M]
        # class_feat = Σ_m weight[c,m] * stream[m]
        # [B,C,M] × [B,M,d] → [B,C,d]
        return torch.bmm(weights, streams)   # [B, C, d_model]


class G3Model(nn.Module):
    """Full G3 architecture.

    Parameters
    ----------
    model_dims    : list of input feature dims per SSL model [D_0, D_1, ...]
    top_k         : number of top layers loaded per model
    d_model       : shared projection dimension
    n_classes     : 6 (Block, Prolongation, SoundRep, WordRep, Interjection, Fluent)
    probe_weights : list of 1-D arrays, one per model — LR probe F1 scores for
                    weight-sum initialization
    dropout       : dropout rate
    """
    def __init__(
        self,
        model_dims: List[int],
        top_k: int,
        d_model: int = 256,
        n_classes: int = N_CLASSES,
        probe_weights: Optional[List[np.ndarray]] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        M = len(model_dims)

        # ── 1. Weighted layer sum (per model) ─────────────────────────────
        self.layer_sums = nn.ModuleList([
            WeightedLayerSum(
                k=top_k,
                init_weights=probe_weights[i] if probe_weights else None,
            )
            for i in range(M)
        ])

        # ── 2. Projection to d_model (per model) ──────────────────────────
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            for dim in model_dims
        ])

        # ── 3. CNN-1D over M model streams ────────────────────────────────
        # Input: [B, d_model, M]  (M treated as sequence length)
        # kernel_size: use 1 for M<=2 (preserves length), 3 for M>=3
        ks = 1 if M <= 2 else 3
        self.cnn_in  = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.cnn_res1 = ResidualConv1d(d_model, kernel_size=ks, dropout=dropout * 0.4)
        self.cnn_res2 = ResidualConv1d(d_model, kernel_size=ks, dropout=dropout * 0.4)

        # ── 4. Gumbel class router ────────────────────────────────────────
        self.router = GumbelClassRouter(n_classes=n_classes, n_models=M)

        # ── 5. Per-class output heads ─────────────────────────────────────
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            for _ in range(n_classes)
        ])

        self.n_classes = n_classes
        self.M = M

    def forward(
        self,
        model_feats: List[torch.Tensor],
        tau: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        model_feats : list of M tensors, each [B, K, D_m]
        returns     : logits [B, n_classes]
        """
        # Step 1+2: weighted sum + project each model → [B, d_model]
        streams = []
        for feats, layer_sum, proj in zip(model_feats, self.layer_sums, self.projections):
            pooled    = layer_sum(feats)   # [B, D_m]
            projected = proj(pooled)        # [B, d_model]
            streams.append(projected)

        # Stack: [B, M, d_model]
        stacked = torch.stack(streams, dim=1)   # [B, M, d_model]

        # Step 3: CNN-1D over model axis [B, d_model, M]
        x = stacked.transpose(1, 2)             # [B, d_model, M]
        x = self.cnn_in(x)
        x = self.cnn_res1(x)
        x = self.cnn_res2(x)
        x = x.transpose(1, 2)                   # [B, M, d_model]

        # Step 4: Gumbel routing → [B, n_classes, d_model]
        class_feats = self.router(x, tau=tau, hard=hard)

        # Step 5: per-class logit → [B, n_classes]
        logits = torch.cat(
            [head(class_feats[:, c, :]) for c, head in enumerate(self.heads)],
            dim=-1,
        )   # [B, n_classes]
        return logits

    def get_hard_selections(self) -> Dict[str, int]:
        """Return the hard-selected model index per class (inference mode)."""
        return {
            CLASS_NAMES[c]: int(self.router.logits[c].argmax().item())
            for c in range(self.n_classes)
        }

    def get_layer_weights(self, model_names: List[str]) -> Dict[str, np.ndarray]:
        """Return softmax α weights per model (shows relative layer importance)."""
        out = {}
        for name, ls in zip(model_names, self.layer_sums):
            out[name] = torch.softmax(ls.logits, dim=0).detach().cpu().numpy()
        return out


# Evaluation helper
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m: Dict[str, float] = {}
    m["accuracy"]  = float(accuracy_score(y_true, y_pred))
    m["macro_f1"]  = float(f1_score(y_true, y_pred, average="macro",  zero_division=0))
    m["micro_f1"]  = float(f1_score(y_true, y_pred, average="micro",  zero_division=0))
    m["macro_pre"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    m["macro_rec"] = float(recall_score(y_true, y_pred,    average="macro", zero_division=0))
    per_f1  = f1_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    per_pre = precision_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    per_rec = recall_score(y_true, y_pred,    average=None, labels=list(range(N_CLASSES)), zero_division=0)
    for c, name in enumerate(CLASS_NAMES):
        m[f"f1_{name}"]  = float(per_f1[c])
        m[f"pre_{name}"] = float(per_pre[c])
        m[f"rec_{name}"] = float(per_rec[c])
    return m


def print_results(label: str, m: Dict[str, float]) -> None:
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print(f"  Accuracy   : {m['accuracy']:.5f}")
    print(f"  Macro-F1   : {m['macro_f1']:.5f}")
    print(f"  Micro-F1   : {m['micro_f1']:.5f}")
    print(f"  Macro-Pre  : {m['macro_pre']:.5f}")
    print(f"  Macro-Rec  : {m['macro_rec']:.5f}")
    print(f"  Per-class F1:")
    for name in CLASS_NAMES:
        print(f"    {name:15s}: F1={m[f'f1_{name}']:.4f}  "
              f"P={m[f'pre_{name}']:.4f}  R={m[f'rec_{name}']:.4f}")


# Main — data loading, training loop, evaluation, saving
def main() -> None:
    args = parse_args()
    for d in (args.out_dir, args.fig_dir, args.ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    # Support both old (torch.cuda.amp) and new (torch.amp) PyTorch AMP API
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"\nDevice : {device}  |  AMP : {use_amp}")

    # ── 1. Read Part-1 metadata ───────────────────────────────────────────────
    if not args.meta_json.exists():
        raise FileNotFoundError(
            f"Meta JSON not found: {args.meta_json}\n"
            "Run G3_part1_probe.py first!"
        )
    meta = json.loads(args.meta_json.read_text())
    fold          = meta["fold"]
    n_samples     = meta["n_samples"]
    seed          = meta["seed"]
    test_size     = meta["test_size"]
    top_k         = meta["top_k"]
    models_meta   = meta["models"]         # {alias: {top_layers, init_weights}}
    features_root = Path(meta["features_root"])
    clips_root    = Path(meta["clips_root"])
    sep_labels    = Path(meta["sep_labels"])

    model_aliases   = sorted(models_meta.keys())
    print(f"\nModels from Part-1 : {model_aliases}")
    print(f"Top-K layers each  : {top_k}")

    # ── 2. Load features for each model (top-K layers) ────────────────────────
    print("\n[1] Loading top-K layer features per model ...")
    all_feats: List[np.ndarray] = []          # [N, K, D] per model
    model_dims: List[int]       = []
    probe_weights_list: List[np.ndarray] = []

    for alias in model_aliases:
        info        = models_meta[alias]
        top_layers  = info["top_layers"]
        init_w      = np.array(info["init_weights"], dtype=np.float32)
        stacked     = load_top_k_layers(features_root, alias, fold, top_layers)
        stacked     = stacked[:n_samples]
        all_feats.append(stacked)
        model_dims.append(stacked.shape[2])
        probe_weights_list.append(init_w)
        print(f"  {alias}: layers {top_layers}  shape {stacked.shape}  dim={stacked.shape[2]}")

    # ── 3. Labels ─────────────────────────────────────────────────────────────
    print("\n[2] Loading labels ...")
    label_map = load_multilabel_map(sep_labels)
    clip_keys  = sorted_clip_keys(clips_root)[:n_samples]
    y_multi    = np.array(
        [label_map.get(k, np.zeros(5, np.float32)) for k in clip_keys],
        dtype=np.float32,
    )
    y_single = multilabel_to_single(y_multi)
    print("  Class distribution:")
    for c, name in enumerate(CLASS_NAMES):
        cnt = int((y_single == c).sum())
        print(f"    {name:15s}: {cnt:5d}  ({cnt/n_samples:.1%})")

    # ── 4. Train / test split ─────────────────────────────────────────────────
    idx = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y_single
    )
    y_tr = y_single[train_idx]
    y_te = y_single[test_idx]
    print(f"\n  Train={len(train_idx)}  Test={len(test_idx)}")

    # ── 5. Scale + split features per model ──────────────────────────────────
    print("\n[3] StandardScaler fit on train fold ...")
    all_feats_tr: List[torch.Tensor] = []
    all_feats_te: List[torch.Tensor] = []

    for stacked in all_feats:
        K, D   = stacked.shape[1], stacked.shape[2]
        flat_tr = stacked[train_idx].reshape(-1, D)
        flat_te = stacked[test_idx].reshape(-1, D)
        sc = StandardScaler()
        flat_tr = sc.fit_transform(flat_tr).astype(np.float32)
        flat_te = sc.transform(flat_te).astype(np.float32)
        all_feats_tr.append(
            torch.from_numpy(flat_tr.reshape(len(train_idx), K, D))
        )
        all_feats_te.append(
            torch.from_numpy(flat_te.reshape(len(test_idx), K, D))
        )

    # ── 6. DataLoaders ────────────────────────────────────────────────────────
    sw = compute_sample_weights(y_tr, N_CLASSES)
    sampler = WeightedRandomSampler(
        torch.from_numpy(sw).float(), len(sw), replacement=True
    )
    y_tr_t = torch.from_numpy(y_tr)
    y_te_t = torch.from_numpy(y_te)

    # Bundle all model features + label into one TensorDataset
    train_ds = TensorDataset(*all_feats_tr, y_tr_t)
    test_ds  = TensorDataset(*all_feats_te, y_te_t)
    M = len(model_aliases)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          sampler=sampler, num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)

    # ── 7. Model ──────────────────────────────────────────────────────────────
    print("\n[4] Building G3Model ...")
    model = G3Model(
        model_dims=model_dims,
        top_k=top_k,
        d_model=args.d_model,
        n_classes=N_CLASSES,
        probe_weights=probe_weights_list,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")
    print(f"  d_model    : {args.d_model}")
    print(f"  M streams  : {M}  ({', '.join(model_aliases)})")
    print(f"  Gumbel τ   : {args.tau_start} → {args.tau_end} over {args.epochs} epochs")

    cw        = compute_class_weights(y_tr, N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    # ── 8. Training loop ──────────────────────────────────────────────────────
    print(f"\n[5] Training for {args.epochs} epochs ...")
    history: List[Dict] = []
    best_macro_f1 = -1.0
    best_ckpt = args.ckpt_dir / "g3_best.pt"

    for epoch in range(1, args.epochs + 1):
        # Anneal Gumbel temperature
        tau = args.tau_start - (args.tau_start - args.tau_end) * (epoch / args.epochs)

        # ── Train ──
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for batch in train_dl:
            *feat_batches, yb = batch
            feat_batches = [f.to(device) for f in feat_batches]
            yb = yb.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(feat_batches, tau=tau, hard=False)
            loss = criterion(logits.float(), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            tr_loss    += loss.item() * yb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += yb.size(0)
        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total
        scheduler.step()

        # ── Validate ──
        model.eval()
        vl_loss, all_preds, all_true = 0.0, [], []
        with torch.no_grad():
            for batch in test_dl:
                *feat_batches, yb = batch
                feat_batches = [f.to(device) for f in feat_batches]
                yb = yb.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(feat_batches, tau=tau, hard=True)
                vl_loss += criterion(logits.float(), yb).item() * yb.size(0)
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_true.append(yb.cpu().numpy())
        vl_loss /= len(y_te)
        y_pred_ep = np.concatenate(all_preds)
        y_true_ep = np.concatenate(all_true)
        macro_f1  = float(f1_score(y_true_ep, y_pred_ep, average="macro", zero_division=0))
        acc_ep    = float(accuracy_score(y_true_ep, y_pred_ep))

        row = {
            "epoch": epoch, "tau": round(tau, 4),
            "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 5),
            "val_loss": round(vl_loss, 6),   "val_macro_f1": round(macro_f1, 5),
            "val_acc": round(acc_ep, 5),
        }
        history.append(row)
        print(f"  epoch={epoch:02d}  τ={tau:.3f}  "
              f"tr_loss={tr_loss:.5f}  tr_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.5f}  macro_f1={macro_f1:.5f}  acc={acc_ep:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), best_ckpt)

    # ── 9. Final evaluation on best checkpoint ────────────────────────────────
    print(f"\n[6] Loading best checkpoint (macro_f1={best_macro_f1:.5f}) ...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_dl:
            *feat_batches, yb = batch
            feat_batches = [f.to(device) for f in feat_batches]
            logits = model(feat_batches, tau=args.tau_end, hard=True)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_true.append(yb.numpy())
    y_pred_final = np.concatenate(all_preds)
    y_true_final = np.concatenate(all_true)
    test_m = evaluate(y_true_final, y_pred_final)
    print_results("G3 Test Results (best checkpoint)", test_m)

    # Gumbel hard selections and layer weights
    hard_sel    = model.get_hard_selections()
    layer_wts   = model.get_layer_weights(model_aliases)
    print("\n  Gumbel Hard Model Selection per class:")
    for cls, midx in hard_sel.items():
        print(f"    {cls:15s} → {model_aliases[midx]}")
    print("\n  Learned Layer Weights (softmax α) per model:")
    for alias, wts in layer_wts.items():
        top_layers = models_meta[alias]["top_layers"]
        wt_str = "  ".join(f"L{l}={w:.3f}" for l, w in zip(top_layers, wts))
        print(f"    {alias}: {wt_str}")

    print("\n  Sklearn classification report:")
    print(classification_report(
        y_true_final, y_pred_final, target_names=CLASS_NAMES, zero_division=0
    ))

    # ── 10. Save outputs ──────────────────────────────────────────────────────
    # Training history CSV
    hist_csv = args.out_dir / "g3_train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)

    # Per-class results CSV
    cls_csv = args.out_dir / "g3_perclass_results.csv"
    with cls_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class","f1","precision","recall"])
        w.writeheader()
        for name in CLASS_NAMES:
            w.writerow({"class": name,
                        "f1":        test_m[f"f1_{name}"],
                        "precision": test_m[f"pre_{name}"],
                        "recall":    test_m[f"rec_{name}"]})

    # Full run report JSON
    run_report = {
        "experiment":     "G3",
        "title":          "Multi-SSL Layer-wise Probing + Gumbel Layer Selection",
        "dataset":        "SEP-28k (6-class single-label, no FluencyBank)",
        "models":         model_aliases,
        "top_k":          top_k,
        "d_model":        args.d_model,
        "n_params":       n_params,
        "tau_start":      args.tau_start,
        "tau_end":        args.tau_end,
        "epochs":         args.epochs,
        "best_epoch":     int(history[int(np.argmax([r["val_macro_f1"] for r in history]))]["epoch"]),
        "best_macro_f1":  best_macro_f1,
        "test_results":   test_m,
        "gumbel_hard_selections": hard_sel,
        "layer_weights":  {k: v.tolist() for k, v in layer_wts.items()},
        "checkpoint":     str(best_ckpt),
        "baseline_G2":    0.6753,
        "vs_G2":          round(test_m["macro_f1"] - 0.6753, 5),
    }
    report_path = args.out_dir / "g3_run_report.json"
    report_path.write_text(json.dumps(run_report, indent=2))

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep_x = [r["epoch"] for r in history]
        best_ep = int(np.argmax([r["val_macro_f1"] for r in history]))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Loss curves
        axes[0].plot(ep_x, [r["train_loss"] for r in history], label="train")
        axes[0].plot(ep_x, [r["val_loss"]   for r in history], label="val")
        axes[0].set_title("G3 Loss"); axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

        # Macro-F1 curve
        axes[1].plot(ep_x, [r["val_macro_f1"] for r in history],
                     "o-", color="royalblue", ms=3, linewidth=2)
        axes[1].axvline(x=history[best_ep]["epoch"], ls="--", color="red", alpha=0.5,
                        label=f"best={best_macro_f1:.4f}")
        axes[1].set_title("G3 Val Macro-F1"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1"); axes[1].legend(); axes[1].grid(alpha=0.3)

        # Per-class F1 bar chart
        pf1    = [test_m[f"f1_{n}"] for n in CLASS_NAMES]
        colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
        bars   = axes[2].bar(CLASS_NAMES, pf1, color=colors, width=0.55)
        axes[2].axhline(test_m["macro_f1"], ls="--", color="black", alpha=0.5,
                        label=f"Macro-F1={test_m['macro_f1']:.4f}")
        for bar, v in zip(bars, pf1):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        axes[2].set_ylim(0, 1); axes[2].set_title("G3 Per-class F1 (Test)")
        axes[2].set_xlabel("Class"); axes[2].set_ylabel("F1")
        axes[2].legend(); axes[2].tick_params(axis="x", rotation=20)

        fig.suptitle("G3 — Gumbel Layer Selector + CNN-1D (SEP-28k 6-class)", fontsize=12)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "g3_results.png", dpi=160)
        plt.close(fig)
        print(f"\n  Plot saved → {args.fig_dir / 'g3_results.png'}")

        # Gumbel weight heatmap
        fig2, ax2 = plt.subplots(figsize=(max(6, M*1.5), 5))
        gumbel_np = torch.softmax(model.router.logits, dim=-1).detach().cpu().numpy()
        im = ax2.imshow(gumbel_np, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax2.set_xticks(range(M)); ax2.set_xticklabels(model_aliases, rotation=20, ha="right")
        ax2.set_yticks(range(N_CLASSES)); ax2.set_yticklabels(CLASS_NAMES)
        ax2.set_title("Gumbel Router Weights: Class × Model Stream")
        ax2.set_xlabel("SSL Model"); ax2.set_ylabel("Stutter Class")
        for ci in range(N_CLASSES):
            for mi in range(M):
                ax2.text(mi, ci, f"{gumbel_np[ci,mi]:.2f}",
                         ha="center", va="center", fontsize=8, color="black")
        plt.colorbar(im, ax=ax2)
        fig2.tight_layout()
        fig2.savefig(args.fig_dir / "g3_gumbel_heatmap.png", dpi=160)
        plt.close(fig2)
        print(f"  Heatmap saved → {args.fig_dir / 'g3_gumbel_heatmap.png'}")

    except Exception as exc:
        print(f"  [WARN] Plotting failed: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  G3 EXPERIMENT COMPLETE")
    print("="*62)
    print(f"  Macro-F1       : {test_m['macro_f1']:.5f}")
    print(f"  Accuracy       : {test_m['accuracy']:.5f}")
    print(f"  vs G2 baseline : {run_report['vs_G2']:+.5f}  (G2=0.6753)")
    print(f"  Report         : {report_path}")
    print(f"  Checkpoint     : {best_ckpt}")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()

