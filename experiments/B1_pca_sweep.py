"""B1: PCA logarithmic sweep on cached SSL embeddings.

Produces:
- results/tables/b1_pca_sweep_summary.csv
- results/tables/b1_best_dims.json
- results/tables/b1_run_report.json
- results/figures/b1_pca_sweep.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SWEEP_DIMS = [1024, 512, 256, 128, 64, 32, 16, 8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 PCA sweep")
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("artifacts/features"),
        help="Root folder for cached features.",
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default="wav2vec2-base",
        help="Model alias folder under features-root.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold0",
        help="Fold folder under model-alias.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=9,
        help="Layer id to use for B1 sweep.",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=SWEEP_DIMS,
        help="Requested PCA dimensions.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio for reconstruction diagnostics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--expected-min-samples",
        type=int,
        default=1000,
        help="Minimum sample count expected for full-run mode.",
    )
    parser.add_argument(
        "--allow-small-sample",
        action="store_true",
        help="Allow running even if sample count is below expected-min-samples.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory where B1 outputs will be written.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory where B1 figure will be written.",
    )
    return parser.parse_args()


def reconstruction_mse(x: np.ndarray, z: np.ndarray, pca: PCA) -> float:
    x_hat = pca.inverse_transform(z)
    return float(np.mean((x - x_hat) ** 2))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    features_file = args.features_root / args.model_alias / args.fold / f"layer_{args.layer}.npy"
    if not features_file.exists():
        raise FileNotFoundError(f"Feature file not found: {features_file}")

    x = np.load(features_file)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)

    n_samples, n_features = x.shape
    if n_samples < args.expected_min_samples and not args.allow_small_sample:
        raise RuntimeError(
            f"Sample count ({n_samples}) is below expected-min-samples ({args.expected_min_samples}). "
            "For full dataset runs, rebuild cache with all clips (max-files 0). "
            "If intentional, re-run with --allow-small-sample."
        )

    x_train, x_test = train_test_split(x, test_size=args.test_size, random_state=args.seed)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    max_dim = min(x_train_s.shape[1], x_train_s.shape[0] - 1)
    dims = sorted({max(2, min(int(d), max_dim)) for d in args.dims}, reverse=True)

    rows = []
    for dim in dims:
        pca = PCA(n_components=dim, random_state=args.seed)
        z_train = pca.fit_transform(x_train_s)
        z_test = pca.transform(x_test_s)

        train_mse = reconstruction_mse(x_train_s, z_train, pca)
        test_mse = reconstruction_mse(x_test_s, z_test, pca)
        evr_sum = float(np.sum(pca.explained_variance_ratio_))

        row = {
            "model_alias": args.model_alias,
            "fold": args.fold,
            "layer": args.layer,
            "dim": dim,
            "train_recon_mse": round(train_mse, 6),
            "test_recon_mse": round(test_mse, 6),
            "explained_variance_sum": round(evr_sum, 6),
        }
        rows.append(row)
        print(
            f"dim={dim} test_mse={test_mse:.6f} explained_var={evr_sum:.6f}"
        )

    if not rows:
        raise RuntimeError("No PCA sweep rows were produced.")

    summary_csv = args.out_dir / "b1_pca_sweep_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_by_mse = sorted(rows, key=lambda r: (r["test_recon_mse"], -r["explained_variance_sum"]))[0]
    best_by_evr = sorted(rows, key=lambda r: (r["explained_variance_sum"], -r["test_recon_mse"]), reverse=True)[0]

    best_json = args.out_dir / "b1_best_dims.json"
    best_json.write_text(
        json.dumps({"best_by_test_mse": best_by_mse, "best_by_explained_variance": best_by_evr}, indent=2),
        encoding="utf-8",
    )

    fig_path = None
    try:
        import matplotlib.pyplot as plt

        dims_plot = [int(r["dim"]) for r in rows]
        mse_plot = [float(r["test_recon_mse"]) for r in rows]
        evr_plot = [float(r["explained_variance_sum"]) for r in rows]

        fig = plt.figure(figsize=(8.5, 5.2))
        ax1 = fig.add_subplot(111)
        ax1.plot(dims_plot, mse_plot, marker="o", color="#1f77b4", label="Test recon MSE")
        ax1.set_xlabel("PCA dimensions")
        ax1.set_ylabel("Test reconstruction MSE", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(dims_plot, evr_plot, marker="s", color="#ff7f0e", label="Explained variance sum")
        ax2.set_ylabel("Explained variance sum", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        fig.suptitle(f"B1 PCA Sweep ({args.model_alias}, fold={args.fold}, layer={args.layer})")
        fig.tight_layout()
        fig_path = args.fig_dir / "b1_pca_sweep.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    except Exception:
        fig_path = None

    report = {
        "experiment": "B1",
        "feature_file": str(features_file),
        "num_samples": int(n_samples),
        "num_features": int(n_features),
        "rows": len(rows),
        "summary_csv": str(summary_csv),
        "best_json": str(best_json),
        "figure": str(fig_path) if fig_path else "not_generated",
        "best_by_test_mse": best_by_mse,
        "best_by_explained_variance": best_by_evr,
    }
    report_json = args.out_dir / "b1_run_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("B1 completed.")
    print(f"Summary: {summary_csv}")
    print(f"Best dims: {best_json}")
    print(f"Run report: {report_json}")
    if fig_path:
        print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
