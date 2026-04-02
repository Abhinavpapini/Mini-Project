"""A2: Learnable weighted-sum over SSL layers + linear classifier.

Full-dataset workflow:
1) Requires cached layer features from A7_build_feature_cache.py
2) Reconstructs clip order from clips directory (sorted)
3) Maps each clip to binary label from SEP-28k/FluencyBank label CSVs
4) Trains learnable softmax layer weights end-to-end
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
    p = argparse.ArgumentParser(description="A2 weighted-sum layer training")
    p.add_argument("--features-root", type=Path, default=Path("artifacts/features"))
    p.add_argument("--model-alias", type=str, default="wav2vec2-base")
    p.add_argument("--fold", type=str, default="fold0")
    p.add_argument("--layers", nargs="+", type=int, default=[1, 3, 6, 9, 12])
    p.add_argument("--clips-root", type=Path, default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels", type=Path, default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path, default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--target", choices=list(LABEL_COLUMNS.keys()), default="Block")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
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
        stem = w.stem  # Show_Ep_Clip
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        show = parts[0].strip()
        ep = parts[1].strip()
        clip = parts[2].strip()
        keys.append((show, ep, clip))
    return keys


class WeightedSumClassifier(nn.Module):
    def __init__(self, num_layers: int, feature_dim: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_layers))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        w = torch.softmax(self.logits, dim=0)
        pooled = torch.einsum("bld,l->bd", x, w)
        return self.classifier(pooled)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load cached feature arrays per layer.
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
    x = np.stack([a[:n] for a in layer_arrays], axis=1)  # [N, L, D]

    target_col = LABEL_COLUMNS[args.target]
    label_map = {}
    label_map.update(load_label_map(args.sep_labels, target_col))
    label_map.update(load_label_map(args.fluency_labels, target_col))

    clip_keys = sorted_clip_keys(args.clips_root)
    if len(clip_keys) < n:
        raise RuntimeError(
            f"Clip count ({len(clip_keys)}) is lower than cached feature rows ({n})."
        )

    clip_keys = clip_keys[:n]
    y = np.array([label_map.get(k, 0) for k in clip_keys], dtype=np.int64)

    # Scale per feature dimension after flattening layer axis.
    x_flat = x.reshape(n, -1)
    x_train, x_test, y_train, y_test = train_test_split(
        x_flat, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    l = len(args.layers)
    x_train = x_train.reshape(x_train.shape[0], l, d)
    x_test = x_test.reshape(x_test.shape[0], l, d)

    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeightedSumClassifier(num_layers=l, feature_dim=d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)

        avg_loss = running / len(train_ds)
        print(f"epoch={epoch} train_loss={avg_loss:.6f}")

    # Evaluate
    model.eval()
    all_pred: List[int] = []
    all_true: List[int] = []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_pred.extend(pred)
            all_true.extend(yb.numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    prec = precision_score(all_true, all_pred, zero_division=0)
    rec = recall_score(all_true, all_pred, zero_division=0)

    with torch.no_grad():
        w = torch.softmax(model.logits, dim=0).cpu().numpy()

    weight_rows = []
    for layer, wt in zip(args.layers, w):
        weight_rows.append({"layer": int(layer), "weight": float(wt)})

    weight_csv = args.out_dir / "a2_layer_weights.csv"
    with weight_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "weight"])
        writer.writeheader()
        writer.writerows(weight_rows)

    metrics = {
        "experiment": "A2",
        "target": args.target,
        "device": str(device),
        "model_alias": args.model_alias,
        "fold": args.fold,
        "num_samples": int(n),
        "num_pos": int(np.sum(y)),
        "num_neg": int(n - np.sum(y)),
        "layers": args.layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "layer_weights": {str(layer): float(wt) for layer, wt in zip(args.layers, w)},
        "weights_csv": str(weight_csv),
    }

    out_json = args.out_dir / "a2_run_report.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 4.2))
        ax = fig.add_subplot(111)
        ax.bar([str(lr) for lr in args.layers], w)
        ax.set_title(f"A2 Layer Weights ({args.target}, {args.model_alias})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Softmax weight")
        fig.tight_layout()
        fig_path = args.fig_dir / "a2_layer_weight_heatmap.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    except Exception:
        fig_path = None




if __name__ == "__main__":
    main()
