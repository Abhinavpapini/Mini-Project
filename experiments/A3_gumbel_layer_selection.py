"""A3: Gumbel-Softmax hard layer selection with BiLSTM + attention classifier.

Workflow:
1) Use cached layer features from artifacts/features/<model>/<fold>/layer_<k>.npy
2) Build binary target labels from SEP-28k/FluencyBank CSVs
3) Train with straight-through Gumbel-Softmax over layer logits
4) Save metrics, selected-layer stats, and training curve
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
import torch.nn.functional as F
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
    p = argparse.ArgumentParser(description="A3 gumbel hard layer selection")
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
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.1)
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


class A3Model(nn.Module):
    def __init__(self, num_layers: int, feature_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.bilstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, D]
        g = F.gumbel_softmax(self.layer_logits, tau=tau, hard=True)
        xg = x * g.view(1, -1, 1)
        out, _ = self.bilstm(xg)
        alpha = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        pooled = torch.sum(out * alpha.unsqueeze(-1), dim=1)
        logits = self.head(pooled)
        return logits, g


def linear_tau(epoch: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 1:
        return end
    frac = (epoch - 1) / (total_epochs - 1)
    return float(start + (end - start) * frac)


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

    x = np.stack([a[:n] for a in layer_arrays], axis=1)  # [N,L,D]

    target_col = LABEL_COLUMNS[args.target]
    label_map = {}
    label_map.update(load_label_map(args.sep_labels, target_col))
    label_map.update(load_label_map(args.fluency_labels, target_col))

    clip_keys = sorted_clip_keys(args.clips_root)
    if len(clip_keys) < n:
        raise RuntimeError(f"Clip count ({len(clip_keys)}) lower than cached rows ({n}).")

    y = np.array([label_map.get(k, 0) for k in clip_keys[:n]], dtype=np.int64)

    # Standardize on flattened representation, then restore [N,L,D].
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
    model = A3Model(
        num_layers=len(args.layers),
        feature_dim=d,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        tau = linear_tau(epoch, args.epochs, args.tau_start, args.tau_end)
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb, tau=tau)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)

        train_loss = running / len(train_ds)
        history.append({"epoch": epoch, "tau": tau, "train_loss": train_loss})
        print(f"epoch={epoch} tau={tau:.4f} train_loss={train_loss:.6f}")

    # Evaluation
    model.eval()
    all_pred: List[int] = []
    all_true: List[int] = []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            logits, _ = model(xb, tau=args.tau_end)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_pred.extend(pred)
            all_true.extend(yb.numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    prec = precision_score(all_true, all_pred, zero_division=0)
    rec = recall_score(all_true, all_pred, zero_division=0)

    with torch.no_grad():
        soft_w = torch.softmax(model.layer_logits, dim=0).cpu().numpy()
        hard_w = F.gumbel_softmax(model.layer_logits, tau=args.tau_end, hard=True).cpu().numpy()

    weights_csv = args.out_dir / "a3_layer_selection_weights.csv"
    with weights_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "soft_weight", "hard_selected"])
        writer.writeheader()
        for layer, sw, hw in zip(args.layers, soft_w, hard_w):
            writer.writerow(
                {
                    "layer": int(layer),
                    "soft_weight": float(sw),
                    "hard_selected": int(hw),
                }
            )

    run_report = {
        "experiment": "A3",
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
        "tau_start": args.tau_start,
        "tau_end": args.tau_end,
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "soft_layer_weights": {str(layer): float(w) for layer, w in zip(args.layers, soft_w)},
        "hard_selected_layer": int(args.layers[int(np.argmax(hard_w))]),
        "weights_csv": str(weights_csv),
    }

    report_json = args.out_dir / "a3_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    history_csv = args.out_dir / "a3_train_history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "tau", "train_loss"])
        writer.writeheader()
        writer.writerows(history)

    fig_path = None
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 4.6))
        ax = fig.add_subplot(111)
        ax.bar([str(l) for l in args.layers], soft_w)
        ax.set_title(f"A3 Soft Layer Weights ({args.target}, {args.model_alias})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Weight")
        fig.tight_layout()
        fig_path = args.fig_dir / "a3_layer_selection_weights.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    except Exception:
        fig_path = None




if __name__ == "__main__":
    main()
