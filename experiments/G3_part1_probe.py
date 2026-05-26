"""G3 Part 1: Layer-wise LR probing (SEP-28k)."""
Outputs are consumed by G3_part2_model.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
STUTTER_TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]
CLASS_NAMES   = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection", "Fluent"]
# Priority order for single-label conversion (index into STUTTER_TYPES)
PRIORITY = [0, 4, 1, 2, 3]   # Block=0 > Interjection=4 > Prolongation=1 > SoundRep=2 > WordRep=3

# Default SSL model aliases + expected layer counts
DEFAULT_MODELS: List[Tuple[str, int]] = [
    ("hubert-large",   24),
    ("whisper-large",  32),
    ("wav2vec2-large", 24),
    ("wavlm-large",    24),
]


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G3 Part-1: Multi-SSL Layer Probing")
    p.add_argument("--features-root", type=Path,
                   default=Path("artifacts/features"),
                   help="Root of cached SSL features (alias/fold/layer_N.npy)")
    p.add_argument("--fold",          type=str,  default="fold0")
    p.add_argument("--clips-root",    type=Path,
                   default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels",    type=Path,
                   default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--model-aliases", type=str, nargs="+",
                   default=[m for m, _ in DEFAULT_MODELS],
                   help="SSL model cache aliases to probe")
    p.add_argument("--top-k",         type=int, default=5,
                   help="Top-K layers per model to keep for Part-2 training")
    p.add_argument("--test-size",     type=float, default=0.20)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--lr-max-iter",   type=int,   default=1000)
    p.add_argument("--out-dir",       type=Path,  default=Path("results/tables"))
    p.add_argument("--fig-dir",       type=Path,  default=Path("results/figures"))
    return p.parse_args()


# Label loading — multi-label → 6-class single-label
def norm(x: object) -> str:
    return str(x).strip()


def load_multilabel_map(csv_path: Path) -> Dict[Tuple[str, str, str], np.ndarray]:
    """Load SEP-28k CSV → dict{(Show, EpId, ClipId): float32 array [5]}."""
    out: Dict[Tuple[str, str, str], np.ndarray] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (norm(row["Show"]), norm(row["EpId"]), norm(row["ClipId"]))
            labels = np.array(
                [1.0 if float(norm(row[t])) >= 1 else 0.0 for t in STUTTER_TYPES],
                dtype=np.float32,
            )
            out[key] = labels
    return out


def multilabel_to_single(y_multi: np.ndarray) -> np.ndarray:
    """Convert [N,5] multi-label → [N] single-label (0-5) using PRIORITY.

    Returns class index:
      0 Block | 1 Prolongation | 2 SoundRep | 3 WordRep | 4 Interjection | 5 Fluent
    """
    y_single = np.full(len(y_multi), 5, dtype=np.int64)  # default = Fluent
    # Apply in REVERSE priority so highest-priority wins last
    for col in reversed(PRIORITY):
        mask = y_multi[:, col] > 0
        y_single[mask] = col
    return y_single


def sorted_clip_keys(clips_root: Path) -> List[Tuple[str, str, str]]:
    keys = []
    for w in sorted(clips_root.rglob("*.wav")):
        parts = w.stem.split("_")
        if len(parts) >= 3:
            keys.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return keys


# SSL cache utilities
def find_layer_folder(features_root: Path, alias: str, fold: str) -> Optional[Path]:
    """Find the folder that contains layer_N.npy files for a given alias."""
    direct = features_root / alias / fold
    if direct.exists() and any(direct.glob("layer_*.npy")):
        return direct
    parent = features_root / alias
    if parent.exists():
        for child in sorted(parent.iterdir()):
            if child.is_dir() and any(child.glob("layer_*.npy")):
                return child
    return None


def discover_layers(layer_folder: Path) -> List[int]:
    """Return sorted list of available layer indices from layer_N.npy files."""
    layers = []
    for f in layer_folder.glob("layer_*.npy"):
        try:
            layers.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(layers)


def load_layer(layer_folder: Path, layer_idx: int) -> np.ndarray:
    """Load [N, D] feature matrix for one layer."""
    arr = np.load(layer_folder / f"layer_{layer_idx}.npy")
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float32)


# LR Probing
def probe_layer(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    max_iter: int = 300,
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """Train LR probe, return (macro_f1, per_class_f1[6])."""
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    lr = LogisticRegression(
        max_iter=max_iter, random_state=seed,
        solver="lbfgs", C=1.0, n_jobs=-1
    )
    lr.fit(Xtr, y_tr)
    y_pred = lr.predict(Xte)
    macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    per_class = f1_score(y_te, y_pred, average=None, zero_division=0,
                         labels=list(range(len(CLASS_NAMES))))
    # Pad to len(CLASS_NAMES) in case some classes missing in test
    out = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    out[:len(per_class)] = per_class
    return macro_f1, out


# Main
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    # ── 1. Load labels ────────────────────────────────────────────────────────
    print("\n[1] Loading SEP-28k labels ...")
    if not args.sep_labels.exists():
        raise FileNotFoundError(f"Labels not found: {args.sep_labels}")
    label_map = load_multilabel_map(args.sep_labels)
    clip_keys  = sorted_clip_keys(args.clips_root)
    print(f"    Found {len(clip_keys)} .wav clips")

    # We build labels aligned to feature N later (after knowing N from first cache)
    # For now store multi-label per clip
    y_multi_all = np.array(
        [label_map.get(k, np.zeros(5, np.float32)) for k in clip_keys],
        dtype=np.float32,
    )
    print(f"    Multi-label matrix: {y_multi_all.shape}")

    # ── 2. Discover available SSL caches ─────────────────────────────────────
    print("\n[2] Discovering cached SSL layers ...")
    model_layer_folders: Dict[str, Path] = {}
    model_layers: Dict[str, List[int]] = {}
    for alias in args.model_aliases:
        folder = find_layer_folder(args.features_root, alias, args.fold)
        if folder is None:
            print(f"    ⚠  {alias}: NO CACHE FOUND — skipping")
            continue
        layers = discover_layers(folder)
        if not layers:
            print(f"    ⚠  {alias}: folder found but no layer_*.npy files — skipping")
            continue
        model_layer_folders[alias] = folder
        model_layers[alias] = layers
        print(f"    ✓  {alias}: {len(layers)} layers found  ({layers[0]}–{layers[-1]})  @ {folder}")

    if not model_layer_folders:
        raise RuntimeError(
            "No SSL caches found! Run build_ssl_cache_full.py first for at least one model."
        )

    # ── 3. Align label length to feature N (use first available model/layer) ──
    first_alias  = next(iter(model_layer_folders))
    first_layer  = model_layers[first_alias][0]
    sample_feats = load_layer(model_layer_folders[first_alias], first_layer)
    N = sample_feats.shape[0]
    print(f"\n[3] Feature N = {N}  (from {first_alias} layer {first_layer})")

    # Align labels
    n_clips = min(N, len(clip_keys))
    y_multi  = y_multi_all[:n_clips]
    y_single = multilabel_to_single(y_multi)

    print("    Single-label class distribution:")
    for c, name in enumerate(CLASS_NAMES):
        cnt = int((y_single == c).sum())
        print(f"      {name:15s}: {cnt:5d}  ({cnt/n_clips:.1%})")

    # ── 4. Train/test split (same seed as other experiments) ─────────────────
    idx = np.arange(n_clips)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed,
        stratify=y_single
    )
    y_tr = y_single[train_idx]
    y_te = y_single[test_idx]
    print(f"\n[4] Split: train={len(train_idx)}  test={len(test_idx)}")

    # ── 5. Layer-wise LR probing ──────────────────────────────────────────────
    print("\n[5] LR probing — layer by layer per model ...")
    all_rows: List[Dict] = []
    model_probe_results: Dict[str, List[Dict]] = {}

    for alias in sorted(model_layer_folders.keys()):
        folder  = model_layer_folders[alias]
        layers  = model_layers[alias]
        rows_m: List[Dict] = []

        print(f"\n  ── {alias}  ({len(layers)} layers) ──")
        for li, layer_idx in enumerate(layers):
            feats = load_layer(folder, layer_idx)[:n_clips]
            X_tr  = feats[train_idx]
            X_te  = feats[test_idx]
            macro_f1, per_class = probe_layer(
                X_tr, X_te, y_tr, y_te,
                max_iter=args.lr_max_iter, seed=args.seed
            )
            row = {
                "model":   alias,
                "layer":   layer_idx,
                "macro_f1": round(macro_f1, 5),
            }
            for c, name in enumerate(CLASS_NAMES):
                row[f"f1_{name}"] = round(float(per_class[c]), 5)
            rows_m.append(row)
            all_rows.append(row)
            print(f"    layer {layer_idx:3d}  macro_f1={macro_f1:.5f}  "
                  f"  [{' | '.join(f'{CLASS_NAMES[c]}={per_class[c]:.3f}' for c in range(6))}]")

        model_probe_results[alias] = rows_m

    # ── 6. Save full probe CSV ────────────────────────────────────────────────
    probe_csv = args.out_dir / "g3_probe_scores.csv"
    fieldnames = ["model", "layer", "macro_f1"] + [f"f1_{n}" for n in CLASS_NAMES]
    with probe_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(all_rows)
    print(f"\n[6] Probe scores saved → {probe_csv}")

    # ── 7. Select top-K layers per model ─────────────────────────────────────
    print(f"\n[7] Selecting top-{args.top_k} layers per model ...")
    top_layers_out: List[Dict] = []
    top_layers_meta: Dict[str, Dict] = {}

    for alias, rows_m in model_probe_results.items():
        sorted_rows = sorted(rows_m, key=lambda r: r["macro_f1"], reverse=True)
        top_k_rows  = sorted_rows[:args.top_k]
        top_layer_indices = sorted([r["layer"] for r in top_k_rows])  # keep sorted
        init_weights = [r["macro_f1"] for r in sorted(top_k_rows, key=lambda r: r["layer"])]

        top_layers_meta[alias] = {
            "top_layers":   top_layer_indices,
            "init_weights": init_weights,  # probe macro-F1 for each top layer (sorted by layer idx)
        }
        print(f"  {alias}: top layers = {top_layer_indices}")
        for r in top_k_rows:
            top_layers_out.append({
                "model": alias, "rank": sorted_rows.index(r) + 1,
                "layer": r["layer"], "macro_f1": r["macro_f1"],
            })

    top_csv = args.out_dir / "g3_top_layers.csv"
    with top_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "rank", "layer", "macro_f1"])
        w.writeheader(); w.writerows(top_layers_out)
    print(f"    Top-K CSV saved → {top_csv}")

    # ── 8. Save metadata JSON for Part-2 ─────────────────────────────────────
    meta = {
        "n_samples":        n_clips,
        "n_train":          len(train_idx),
        "n_test":           len(test_idx),
        "seed":             args.seed,
        "test_size":        args.test_size,
        "top_k":            args.top_k,
        "class_names":      CLASS_NAMES,
        "models":           top_layers_meta,
        "features_root":    str(args.features_root),
        "fold":             args.fold,
        "clips_root":       str(args.clips_root),
        "sep_labels":       str(args.sep_labels),
    }
    meta_path = args.out_dir / "g3_probe_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"    Meta JSON saved  → {meta_path}")

    # ── 9. Plot layer curves ──────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap

        n_models = len(model_probe_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
        cmap = get_cmap("tab10")

        for ax_idx, (alias, rows_m) in enumerate(model_probe_results.items()):
            ax = axes[0][ax_idx]
            layers_x     = [r["layer"]    for r in rows_m]
            macro_f1_y   = [r["macro_f1"] for r in rows_m]
            ax.plot(layers_x, macro_f1_y, "o-", color=cmap(ax_idx),
                    linewidth=2, markersize=4, label="Macro-F1")
            # Highlight top-K
            top_l = top_layers_meta[alias]["top_layers"]
            for tl in top_l:
                tidx = layers_x.index(tl) if tl in layers_x else None
                if tidx is not None:
                    ax.axvline(x=tl, ls="--", color="red", alpha=0.4, linewidth=1)
            best_row = max(rows_m, key=lambda r: r["macro_f1"])
            ax.set_title(f"{alias}\nbest=layer {best_row['layer']} ({best_row['macro_f1']:.4f})",
                         fontsize=9)
            ax.set_xlabel("Layer Index"); ax.set_ylabel("LR Probe Macro-F1")
            ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=7)

        fig.suptitle("G3 — Layer-wise LR Probe Macro-F1 (SEP-28k 6-class)", fontsize=11)
        fig.tight_layout()
        out_fig = args.fig_dir / "g3_layer_curves.png"
        fig.savefig(out_fig, dpi=160); plt.close(fig)
        print(f"    Layer curves saved → {out_fig}")
    except Exception as exc:
        print(f"    [WARN] Plotting failed: {exc}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  G3 Part-1 complete.")
    print(f"  Probed {len(model_probe_results)} SSL model(s).")
    print(f"  Top-{args.top_k} layers per model selected.")
    print(f"  → Run G3_part2_model.py next (reads g3_probe_meta.json).")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
