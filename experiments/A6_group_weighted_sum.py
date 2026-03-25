"""A6: Early/Mid/Late grouped weighted-sum with PCA + CNN classifier.

Workflow:
1) Load cached layer features from artifacts/features/<model>/<fold>/layer_<k>.npy
2) Build binary target labels from SEP-28k/FluencyBank CSVs
3) Group layers into early/mid/late buckets and compute per-group means
4) Learn softmax group weights, fuse grouped features, then classify with CNN
5) Save metrics, learned group weights, and training history
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


GROUP_NAMES = ["early", "mid", "late"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A6 grouped weighted sum + PCA + CNN")
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
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pca-dim", type=int, default=32)
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


def make_group_indices(num_layers: int) -> List[np.ndarray]:
    # Keep contiguous ordering and split into 3 buckets: early, mid, late.
    idx = np.arange(num_layers)
    splits = np.array_split(idx, 3)
    # Ensure 3 groups even for small layer counts.
    while len(splits) < 3:
        splits.append(np.array([], dtype=int))
    return [splits[0], splits[1], splits[2]]


def grouped_mean_features(x: np.ndarray, group_indices: List[np.ndarray]) -> np.ndarray:
    # x: [N, L, D] -> out: [N, G, D]
    out = []
    for g_idx in group_indices:
        if g_idx.size == 0:
            out.append(np.zeros((x.shape[0], x.shape[2]), dtype=x.dtype))
        else:
            out.append(x[:, g_idx, :].mean(axis=1))
    return np.stack(out, axis=1)


class A6Model(nn.Module):
    def __init__(self, pca_dim: int, num_groups: int, dropout: float):
        super().__init__()
        self.group_logits = nn.Parameter(torch.zeros(num_groups))

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, G, P]
        w = torch.softmax(self.group_logits, dim=0)  # [G]
        fused = torch.sum(x * w.view(1, -1, 1), dim=1)  # [B, P]

        z = fused.unsqueeze(1)  # [B, 1, P]
        z = self.cnn(z).squeeze(-1)  # [B, 64]
        logits = self.head(z)
        return logits, w


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

    group_indices = make_group_indices(len(args.layers))
    x_group = grouped_mean_features(x, group_indices)  # [N, 3, D]

    # Fit PCA on train split only, using all groups stacked.
    x_tmp = x_group.reshape(n, -1)
    x_train_tmp, x_test_tmp, y_train, y_test = train_test_split(
        x_tmp, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    x_train_group = x_train_tmp.reshape(-1, len(GROUP_NAMES), d)
    x_test_group = x_test_tmp.reshape(-1, len(GROUP_NAMES), d)

    pca_fit_data = x_train_group.reshape(-1, d)
    pca_dim = min(args.pca_dim, d)
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    pca.fit(pca_fit_data)

    x_train_p = pca.transform(x_train_group.reshape(-1, d)).reshape(-1, len(GROUP_NAMES), pca_dim)
    x_test_p = pca.transform(x_test_group.reshape(-1, d)).reshape(-1, len(GROUP_NAMES), pca_dim)

    scaler = StandardScaler()
    x_train_p = scaler.fit_transform(x_train_p.reshape(x_train_p.shape[0], -1)).reshape(-1, len(GROUP_NAMES), pca_dim)
    x_test_p = scaler.transform(x_test_p.reshape(x_test_p.shape[0], -1)).reshape(-1, len(GROUP_NAMES), pca_dim)

    train_ds = TensorDataset(torch.from_numpy(x_train_p).float(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(x_test_p).float(), torch.from_numpy(y_test).long())
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = A6Model(pca_dim=pca_dim, num_groups=len(GROUP_NAMES), dropout=args.dropout).to(device)

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
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            logits, _ = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_pred.extend(pred)
            all_true.extend(yb.numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    prec = precision_score(all_true, all_pred, zero_division=0)
    rec = recall_score(all_true, all_pred, zero_division=0)

    with torch.no_grad():
        group_w = torch.softmax(model.group_logits, dim=0).cpu().numpy()

    layer_group_csv = args.out_dir / "a6_layer_group_assignment.csv"
    with layer_group_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "group"])
        writer.writeheader()
        for group_name, g_idx in zip(GROUP_NAMES, group_indices):
            for li in g_idx.tolist():
                writer.writerow({"layer": int(args.layers[li]), "group": group_name})

    weights_csv = args.out_dir / "a6_group_weights.csv"
    with weights_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "weight"])
        writer.writeheader()
        for g, w in zip(GROUP_NAMES, group_w):
            writer.writerow({"group": g, "weight": float(w)})

    run_report = {
        "experiment": "A6",
        "target": args.target,
        "device": str(device),
        "model_alias": args.model_alias,
        "fold": args.fold,
        "num_samples": int(n),
        "num_pos": int(np.sum(y)),
        "num_neg": int(n - np.sum(y)),
        "layers": args.layers,
        "layer_groups": {
            GROUP_NAMES[i]: [int(args.layers[j]) for j in group_indices[i].tolist()]
            for i in range(len(GROUP_NAMES))
        },
        "feature_dim": int(d),
        "pca_dim": int(pca_dim),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "group_weights": {g: float(w) for g, w in zip(GROUP_NAMES, group_w)},
        "dominant_group": GROUP_NAMES[int(np.argmax(group_w))],
        "group_weights_csv": str(weights_csv),
        "layer_group_assignment_csv": str(layer_group_csv),
    }

    report_json = args.out_dir / "a6_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    history_csv = args.out_dir / "a6_train_history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(history)

    fig_path = None
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7.5, 4.2))
        ax = fig.add_subplot(111)
        ax.bar(GROUP_NAMES, group_w)
        ax.set_title(f"A6 Group Weights ({args.target}, {args.model_alias})")
        ax.set_xlabel("Group")
        ax.set_ylabel("Softmax weight")
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()
        fig_path = args.fig_dir / "a6_group_weights.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    except Exception:
        fig_path = None

    print("A6 completed.")
    print(f"Report: {report_json}")
    print(f"Group weights: {weights_csv}")
    print(f"Layer groups: {layer_group_csv}")
    print(f"History: {history_csv}")
    if fig_path:
        print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
