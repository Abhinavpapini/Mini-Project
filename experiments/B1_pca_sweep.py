"""
B1: PCA logarithmic sweep entrypoint.

This script is intentionally minimal and code-first.
It validates inputs and can be extended to execute the full sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SWEEP_DIMS = [1024, 512, 256, 128, 64, 32, 16, 8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 PCA sweep")
    parser.add_argument(
        "--features-file",
        type=Path,
        required=False,
        help="Path to serialized best-layer embeddings.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory where B1 outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("B1 script scaffold is ready.")
    print(f"Sweep dimensions: {SWEEP_DIMS}")
    if args.features_file:
        print(f"Input features file: {args.features_file}")
    else:
        print("No features file passed yet. Use --features-file to run full sweep.")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
