"""A5: Hierarchical convolution across layers with MLP classifier.

Workflow:
1) Load cached layer features from artifacts/features/<model>/<fold>/layer_<k>.npy
2) Build binary target labels from SEP-28k/FluencyBank CSVs
3) Learn cross-layer interactions using 1D convolutions over layer axis
4) Classify using an MLP head
5) Save metrics, per-layer salience, and training history
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


LABEL_COLUMNS = {
    "Block": "Block",
    "Prolongation": "Prolongation",
    "SoundRep": "SoundRep",
    "WordRep": "WordRep",
    "Interjection": "Interjection",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A5 hierarchical conv across layers")
    p.add_argument("--features-root", type=Path, default=Path("artifacts/features"))
    p.add_argument("--model-alias", type=str, default="wav2vec2-base")
    p.add_argument("--fold", type=str, default="fold0")
    p.add_argument("--layers", nargs="+", type=int, default=[1, 3, 6, 9, 12])
    p.add_argument("--clips-root", type=Path, default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels", type=Path, default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path, default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--target", choices=list(LABEL_COLUMNS.keys()), default="Block")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected-min-samples", type=int, default=1000)
    p.add_argument("--allow-small-sample", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    p.add_argument("--fig-dir", type=Path, default=Path("results/figures"))
    return p.parse_args()


def norm_text(x: str) -> str:
    return str(x).strip()


def load_label_map(csv_path: Path, target_col: str) -> Dict[Tuple[str, str, str], int]:
    out: Dict[Tuple[str, str, str], int] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            show = norm_text(row["Show"])
            ep = norm_text(row["EpId"])
            clip = norm_text(row["ClipId"])
            val = float(norm_text(row[target_col]))
            out[(show, ep, clip)] = 1 if val >= 1 else 0
    return out


def sorted_clip_keys(clips_root: Path) -> List[Tuple[str, str, str]]:
    wavs = sorted(clips_root.rglob("*.wav"))
    keys: List[Tuple[str, str, str]] = []
    for w in wavs:
        parts = w.stem.split("_")
        if len(parts) < 3:
            continue
        keys.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return keys


class A5Model(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layer_proj = nn.Linear(feature_dim, hidden_dim)

        # Convolve over the layer axis (sequence length = number of layers).
        self.conv_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, D]
        h = self.layer_proj(x)  # [B, L, H]
        h = torch.relu(h)

        # Rearrange to channels-first for Conv1d over layers.
        z = h.transpose(1, 2)  # [B, H, L]
        z = self.conv_block(z)  # [B, H, L]

        # Global average pool over layer dimension.
        pooled = z.mean(dim=2)  # [B, H]
        logits = self.classifier(pooled)

        # Per-layer salience proxy from projected activations.
        salience = h.abs().mean(dim=2)  # [B, L]
        return logits, salience


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    layer_arrays: List[np.ndarray] = []
    for layer in args.layers:
        p = args.features_root / args.model_alias / args.fold / f"layer_{layer}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing cached layer file: {p}")
        arr = np.load(p)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        layer_arrays.append(arr)

    n = min(a.shape[0] for a in layer_arrays)
    d = layer_arrays[0].shape[1]
    if n < args.expected_min_samples and not args.allow_small_sample:
        raise RuntimeError(
            f"Sample count ({n}) is below expected-min-samples ({args.expected_min_samples}). "
            "Use full cache or pass --allow-small-sample intentionally."
        )

    x = np.stack([a[:n] for a in layer_arrays], axis=1)  # [N, L, D]

    target_col = LABEL_COLUMNS[args.target]
    label_map = {}
    label_map.update(load_label_map(args.sep_labels, target_col))
    label_map.update(load_label_map(args.fluency_labels, target_col))

    clip_keys = sorted_clip_keys(args.clips_root)
    if len(clip_keys) < n:
        raise RuntimeError(f"Clip count ({len(clip_keys)}) lower than cached rows ({n}).")

    y = np.array([label_map.get(k, 0) for k in clip_keys[:n]], dtype=np.int64)

    x_flat = x.reshape(n, -1)
    x_train, x_test, y_train, y_test = train_test_split(
        x_flat, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).reshape(-1, len(args.layers), d)
    x_test = scaler.transform(x_test).reshape(-1, len(args.layers), d)

    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = A5Model(feature_dim=d, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)

        train_loss = running / len(train_ds)
        history.append({"epoch": epoch, "train_loss": train_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f}")

    model.eval()
    all_pred: List[int] = []
    all_true: List[int] = []
    salience_accum = []

    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            logits, sal = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_pred.extend(pred)
            all_true.extend(yb.numpy().tolist())
            salience_accum.append(sal.cpu().numpy())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    prec = precision_score(all_true, all_pred, zero_division=0)
    rec = recall_score(all_true, all_pred, zero_division=0)

    salience_matrix = np.concatenate(salience_accum, axis=0)  # [Ntest, L]
    layer_salience = salience_matrix.mean(axis=0)
    layer_salience = layer_salience / np.sum(layer_salience)

    salience_csv = args.out_dir / "a5_layer_salience.csv"
    with salience_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "salience_share"])
        writer.writeheader()
        for layer, s in zip(args.layers, layer_salience):
            writer.writerow({"layer": int(layer), "salience_share": float(s)})

    run_report = {
        "experiment": "A5",
        "target": args.target,
        "device": str(device),
        "model_alias": args.model_alias,
        "fold": args.fold,
        "num_samples": int(n),
        "num_pos": int(np.sum(y)),
        "num_neg": int(n - np.sum(y)),
        "layers": args.layers,
        "feature_dim": int(d),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "layer_salience_share": {str(layer): float(s) for layer, s in zip(args.layers, layer_salience)},
        "dominant_salience_layer": int(args.layers[int(np.argmax(layer_salience))]),
        "salience_csv": str(salience_csv),
    }

    report_json = args.out_dir / "a5_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    history_csv = args.out_dir / "a5_train_history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(history)

    fig_path = None
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 4.6))
        ax = fig.add_subplot(111)
        ax.bar([str(l) for l in args.layers], layer_salience)
        ax.set_title(f"A5 Layer Salience ({args.target}, {args.model_alias})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized salience")
        fig.tight_layout()
        fig_path = args.fig_dir / "a5_layer_salience.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    except Exception:
        fig_path = None

    print("A5 completed.")
    print(f"Report: {report_json}")
    print(f"Salience: {salience_csv}")
    print(f"History: {history_csv}")
    if fig_path:
        print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
