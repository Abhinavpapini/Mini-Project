"""B7: Sparse PCA analysis on cached SSL layer features.

Current implementation runs unsupervised SparsePCA diagnostics on a cached layer file
and produces result tables/plots for dimensionality and sparsity behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.decomposition import SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B7 Sparse PCA")
    parser.add_argument("--features-root", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--model-alias", type=str, default="wav2vec2-base")
    parser.add_argument("--fold", type=str, default="fold0")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[256, 128, 64, 32, 16, 8],
        help="SparsePCA component counts to evaluate",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 1.0],
        help="Sparse penalty strengths to evaluate",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--fig-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def reconstruction_mse(x: np.ndarray, z: np.ndarray, components: np.ndarray, mean: np.ndarray) -> float:
    x_hat = z @ components + mean
    return float(np.mean((x - x_hat) ** 2))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    feature_file = args.features_root / args.model_alias / args.fold / f"layer_{args.layer}.npy"
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    x = np.load(feature_file)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)

    x_train, x_test = train_test_split(x, test_size=args.test_size, random_state=args.seed)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    rows: List[dict] = []

    for alpha in args.alphas:
        for dim in args.dims:
            dim_eff = min(dim, x_train_s.shape[1], x_train_s.shape[0] - 1)
            if dim_eff < 2:
                continue

            model = SparsePCA(
                n_components=dim_eff,
                alpha=alpha,
                random_state=args.seed,
                max_iter=400,
                n_jobs=-1,
            )
            z_train = model.fit_transform(x_train_s)
            z_test = model.transform(x_test_s)

            train_mse = reconstruction_mse(x_train_s, z_train, model.components_, model.mean_)
            test_mse = reconstruction_mse(x_test_s, z_test, model.components_, model.mean_)

            abs_comp = np.abs(model.components_)
            sparsity = float(np.mean(abs_comp < 1e-6))
            nnz_per_component = float(np.mean(np.sum(abs_comp >= 1e-6, axis=1)))

            rows.append(
                {
                    "model_alias": args.model_alias,
                    "fold": args.fold,
                    "layer": args.layer,
                    "alpha": alpha,
                    "dim": dim_eff,
                    "train_recon_mse": round(train_mse, 6),
                    "test_recon_mse": round(test_mse, 6),
                    "sparsity_ratio": round(sparsity, 6),
                    "avg_nonzero_per_component": round(nnz_per_component, 3),
                }
            )
            print(
                f"alpha={alpha} dim={dim_eff} test_mse={test_mse:.6f} "
                f"sparsity={sparsity:.4f}"
            )

    if not rows:
        raise RuntimeError("No B7 rows were produced. Check inputs/dimensions.")

    summary_csv = args.out_dir / "b7_sparse_pca_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = sorted(rows, key=lambda r: (r["test_recon_mse"], -r["sparsity_ratio"]))[0]
    best_json = args.out_dir / "b7_best_config.json"
    best_json.write_text(json.dumps(best, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        # Plot test MSE for each alpha across dims.
        dims_sorted = sorted(set(int(r["dim"]) for r in rows))
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        for alpha in args.alphas:
            vals = []
            for d in dims_sorted:
                match = [r for r in rows if float(r["alpha"]) == float(alpha) and int(r["dim"]) == d]
                vals.append(match[0]["test_recon_mse"] if match else np.nan)
            ax.plot(dims_sorted, vals, marker="o", label=f"alpha={alpha}")
        ax.set_title(f"B7 SparsePCA Test Reconstruction MSE ({args.model_alias}, L{args.layer})")
        ax.set_xlabel("Components")
        ax.set_ylabel("Test Reconstruction MSE")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig_path = args.fig_dir / "b7_sparse_pca_curve.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    except Exception:
        fig_path = None

    report = {
        "experiment": "B7",
        "feature_file": str(feature_file),
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "rows": len(rows),
        "summary_csv": str(summary_csv),
        "best_json": str(best_json),
        "figure": str(fig_path) if fig_path else "not_generated",
        "best": best,
    }
    report_json = args.out_dir / "b7_run_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")




if __name__ == "__main__":
    main()
