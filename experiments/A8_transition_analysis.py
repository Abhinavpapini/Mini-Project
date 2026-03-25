"""A8: Transition phenomena detection across layers.

Computes representation drift between adjacent available layers using:
- L2 norm of mean embedding shift
- Cosine similarity of mean embeddings
- Mean per-sample L2 drift

Input layout:
artifacts/features/<model_alias>/<fold>/layer_<k>.npy
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A8 transition analysis")
    parser.add_argument("--features-root", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--model-alias", type=str, default="wav2vec2-base")
    parser.add_argument("--fold", type=str, default="fold0")
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 3, 6, 9, 12])
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--fig-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    layer_arrays: List[Tuple[int, np.ndarray]] = []
    for layer in sorted(args.layers):
        p = args.features_root / args.model_alias / args.fold / f"layer_{layer}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.ndim != 2:
                arr = arr.reshape(arr.shape[0], -1)
            layer_arrays.append((layer, arr))

    if len(layer_arrays) < 2:
        raise RuntimeError("Need at least two discovered layers for A8 transition analysis")

    rows = []
    for idx in range(len(layer_arrays) - 1):
        la, xa = layer_arrays[idx]
        lb, xb = layer_arrays[idx + 1]

        n = min(xa.shape[0], xb.shape[0])
        xa = xa[:n]
        xb = xb[:n]

        mean_a = xa.mean(axis=0)
        mean_b = xb.mean(axis=0)

        l2_mean_shift = float(np.linalg.norm(mean_b - mean_a))
        mean_cos = cosine_similarity(mean_a, mean_b)
        per_sample_l2 = np.linalg.norm(xb - xa, axis=1)
        mean_per_sample_l2 = float(np.mean(per_sample_l2))
        std_per_sample_l2 = float(np.std(per_sample_l2))

        rows.append(
            {
                "model_alias": args.model_alias,
                "fold": args.fold,
                "layer_from": la,
                "layer_to": lb,
                "num_samples": n,
                "l2_mean_shift": round(l2_mean_shift, 6),
                "mean_cosine_similarity": round(mean_cos, 6),
                "mean_per_sample_l2": round(mean_per_sample_l2, 6),
                "std_per_sample_l2": round(std_per_sample_l2, 6),
            }
        )

    summary_csv = args.out_dir / "a8_transition_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Largest transition by mean per-sample drift.
    best_row = sorted(rows, key=lambda r: r["mean_per_sample_l2"], reverse=True)[0]

    try:
        import matplotlib.pyplot as plt

        x_labels = [f"{r['layer_from']}->{r['layer_to']}" for r in rows]
        y_vals = [r["mean_per_sample_l2"] for r in rows]
        fig = plt.figure(figsize=(8, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(x_labels, y_vals, marker="o")
        ax.set_title(f"A8 Layer Transition Drift ({args.model_alias}, {args.fold})")
        ax.set_xlabel("Layer Transition")
        ax.set_ylabel("Mean per-sample L2 drift")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = args.fig_dir / "a8_transition_drift_curve.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    except Exception:
        fig_path = None

    run_report = {
        "experiment": "A8",
        "model_alias": args.model_alias,
        "fold": args.fold,
        "num_pairs": len(rows),
        "summary_csv": str(summary_csv),
        "figure": str(fig_path) if fig_path else "not_generated",
        "largest_transition": best_row,
    }
    report_json = args.out_dir / "a8_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    print("A8 completed.")
    print(f"Summary: {summary_csv}")
    print(f"Run report: {report_json}")
    if fig_path:
        print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
