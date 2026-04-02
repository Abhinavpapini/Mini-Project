"""A7: CKA/SVCCA cross-model probing.

Expected feature layout (recommended):
artifacts/features/<ssl_model>/<fold>/layer_<k>.<npy|npz|pt|parquet|csv>
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


LAYER_RE = re.compile(r"layer[_-]?(\d+)", re.IGNORECASE)
SUPPORTED_EXTS = {".npy", ".npz", ".pt", ".parquet", ".csv"}


@dataclass(frozen=True)
class FeatureEntry:
    model: str
    fold: str
    layer: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A7 CKA/SVCCA probing")
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("artifacts/features"),
        help="Root directory with cached SSL features.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory where A7 output tables will be written.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory where optional heatmaps will be written.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.99,
        help="Explained variance threshold used by SVCCA PCA pre-step.",
    )
    parser.add_argument(
        "--max-layer-gaps",
        type=int,
        default=999,
        help="Intra-model layer-gap limit (default uses all discovered pairs).",
    )
    return parser.parse_args()


def discover_feature_files(features_root: Path) -> List[FeatureEntry]:
    entries: List[FeatureEntry] = []
    if not features_root.exists():
        return entries

    for model_dir in sorted(p for p in features_root.iterdir() if p.is_dir()):
        model_name = model_dir.name
        for fold_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            fold_name = fold_dir.name
            for f in fold_dir.rglob("*"):
                if not f.is_file() or f.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                match = LAYER_RE.search(f.stem)
                if not match:
                    continue
                entries.append(
                    FeatureEntry(
                        model=model_name,
                        fold=fold_name,
                        layer=int(match.group(1)),
                        path=f,
                    )
                )
    return entries


def _to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    # For (n, t, d) or higher rank, collapse all non-batch dims.
    return arr.reshape(arr.shape[0], -1)


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return _to_2d(np.asarray(arr, dtype=np.float64))

    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        key = data.files[0]
        return _to_2d(np.asarray(data[key], dtype=np.float64))

    if suffix == ".pt":
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required to read .pt feature files") from exc
        tensor = torch.load(path, map_location="cpu")
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)
        return _to_2d(np.asarray(arr, dtype=np.float64))

    if suffix == ".parquet":
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas+pyarrow are required to read .parquet") from exc
        df = pd.read_parquet(path)
        return _to_2d(df.to_numpy(dtype=np.float64))

    if suffix == ".csv":
        arr = np.genfromtxt(path, delimiter=",", skip_header=1)
        return _to_2d(np.asarray(arr, dtype=np.float64))

    raise ValueError(f"Unsupported feature file: {path}")


def center(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    # Feature-space formulation avoids building n x n Gram matrices.
    # CKA = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
    x = center(x)
    y = center(y)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    hsic = np.sum(xty * xty)
    norm = np.sqrt(np.sum(xtx * xtx) * np.sum(yty * yty))
    if norm == 0:
        return 0.0
    return float(hsic / norm)


def pca_by_variance(x: np.ndarray, threshold: float) -> np.ndarray:
    x = center(x)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    var = s**2
    ratio = np.cumsum(var) / np.sum(var)
    keep = int(np.searchsorted(ratio, threshold) + 1)
    keep = max(1, min(keep, x.shape[1]))
    return u[:, :keep] * s[:keep]


def svcca(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    from sklearn.cross_decomposition import CCA
    from sklearn.exceptions import ConvergenceWarning

    x_p = pca_by_variance(x, threshold)
    y_p = pca_by_variance(y, threshold)
    n = min(x_p.shape[0], y_p.shape[0])
    x_p = x_p[:n]
    y_p = y_p[:n]

    k = min(x_p.shape[1], y_p.shape[1], 64)
    if k < 1:
        return 0.0

    cca = CCA(n_components=k, max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        x_c, y_c = cca.fit_transform(x_p, y_p)

    corrs: List[float] = []
    for i in range(k):
        c = np.corrcoef(x_c[:, i], y_c[:, i])[0, 1]
        if np.isnan(c):
            c = 0.0
        corrs.append(float(c))
    return float(np.mean(corrs)) if corrs else 0.0


def align_samples(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def aggregate_by_model_layer(entries: List[FeatureEntry]) -> Dict[str, Dict[int, np.ndarray]]:
    grouped: Dict[str, Dict[int, List[np.ndarray]]] = {}
    for e in entries:
        grouped.setdefault(e.model, {}).setdefault(e.layer, []).append(load_array(e.path))

    out: Dict[str, Dict[int, np.ndarray]] = {}
    for model, layer_dict in grouped.items():
        out[model] = {}
        for layer, arrays in layer_dict.items():
            min_n = min(arr.shape[0] for arr in arrays)
            aligned = [arr[:min_n] for arr in arrays]
            merged = np.concatenate(aligned, axis=0)
            out[model][layer] = merged
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for row in rows:
        vals = []
        for h in header:
            v = row[h]
            if isinstance(v, str) and ("," in v or '"' in v):
                vals.append('"' + v.replace('"', '""') + '"')
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_plot_heatmap(matrix: np.ndarray, labels: List[str], out_path: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    entries = discover_feature_files(args.features_root)
    if not entries:
        raise FileNotFoundError(
            "No layer feature files discovered. "
            f"Expected files under {args.features_root} matching layer_<k>.<ext>. "
            "Generate caches first, for example: "
            "python experiments/A7_build_feature_cache.py --clips-root ml-stuttering-events-dataset/clips "
            "--out-root artifacts/features --model-name facebook/wav2vec2-base --fold fold0"
        )

    feature_map = aggregate_by_model_layer(entries)
    models = sorted(feature_map.keys())

    rows: List[Dict[str, object]] = []

    # Intra-model adjacent layer similarity.
    for model in models:
        layers = sorted(feature_map[model].keys())
        for i, la in enumerate(layers):
            for lb in layers[i + 1 :]:
                if lb - la > args.max_layer_gaps:
                    continue
                xa, xb = align_samples(feature_map[model][la], feature_map[model][lb])
                rows.append(
                    {
                        "pair_type": "intra_model",
                        "model_a": model,
                        "model_b": model,
                        "layer_a": la,
                        "layer_b": lb,
                        "n_samples": xa.shape[0],
                        "dim_a": xa.shape[1],
                        "dim_b": xb.shape[1],
                        "cka": round(linear_cka(xa, xb), 6),
                        "svcca": round(svcca(xa, xb, args.variance_threshold), 6),
                    }
                )

    # Cross-model same-layer similarity over shared layer ids.
    for i, ma in enumerate(models):
        for mb in models[i + 1 :]:
            shared_layers = sorted(set(feature_map[ma]).intersection(feature_map[mb]))
            for layer in shared_layers:
                xa, xb = align_samples(feature_map[ma][layer], feature_map[mb][layer])
                rows.append(
                    {
                        "pair_type": "cross_model_same_layer",
                        "model_a": ma,
                        "model_b": mb,
                        "layer_a": layer,
                        "layer_b": layer,
                        "n_samples": xa.shape[0],
                        "dim_a": xa.shape[1],
                        "dim_b": xb.shape[1],
                        "cka": round(linear_cka(xa, xb), 6),
                        "svcca": round(svcca(xa, xb, args.variance_threshold), 6),
                    }
                )

    if not rows:
        raise RuntimeError(
            "No valid model/layer pairs were available for A7 similarity computation."
        )

    summary_csv = args.out_dir / "a7_similarity_summary.csv"
    write_csv(summary_csv, rows)

    sorted_rows = sorted(rows, key=lambda r: (r["cka"] + r["svcca"]), reverse=True)
    top_rows = sorted_rows[:20]
    top_csv = args.out_dir / "a7_top_pairs.csv"
    write_csv(top_csv, top_rows)

    # Optional within-model heatmaps for CKA values.
    heatmap_written = 0
    for model in models:
        layers = sorted(feature_map[model].keys())
        if len(layers) < 2:
            continue
        mat = np.zeros((len(layers), len(layers)), dtype=np.float64)
        for i, la in enumerate(layers):
            for j, lb in enumerate(layers):
                xa, xb = align_samples(feature_map[model][la], feature_map[model][lb])
                mat[i, j] = linear_cka(xa, xb)
        wrote = try_plot_heatmap(
            matrix=mat,
            labels=[str(x) for x in layers],
            out_path=args.fig_dir / f"a7_cka_heatmap_{model}.png",
            title=f"A7 CKA Heatmap - {model}",
        )
        if wrote:
            heatmap_written += 1

    report = {
        "experiment": "A7",
        "features_root": str(args.features_root),
        "models_found": models,
        "num_feature_files": len(entries),
        "num_similarity_rows": len(rows),
        "summary_csv": str(summary_csv),
        "top_pairs_csv": str(top_csv),
        "heatmaps_written": heatmap_written,
    }
    report_path = args.out_dir / "a7_run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")




if __name__ == "__main__":
    main()
