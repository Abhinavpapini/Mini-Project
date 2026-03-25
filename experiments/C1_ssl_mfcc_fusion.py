"""C1: SSL + MFCC fusion baseline with MFCC-dimension sweep.

Workflow:
1) Load cached SSL features from artifacts/features/<model>/<fold>/layer_<k>.npy
2) Build or load MFCC stats cache aligned to clip ordering
3) Reduce SSL to PCA-32, concatenate with MFCC subsets (dimension sweep)
4) Train CNN-1D classifier for each MFCC subset and record metrics
5) Save sweep table, best-run report, training history, and figure
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C1 SSL+MFCC fusion baseline")
    p.add_argument("--features-root", type=Path, default=Path("artifacts/features"))
    p.add_argument("--model-alias", type=str, default="wav2vec2-base")
    p.add_argument("--fold", type=str, default="fold0")
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--clips-root", type=Path, default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels", type=Path, default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path, default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--target", choices=list(LABEL_COLUMNS.keys()), default="Block")
    p.add_argument("--ssl-pca-dim", type=int, default=32)
    p.add_argument("--mfcc-dims", nargs="+", type=int, default=[13, 26, 39, 52, 65, 78])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--n-mfcc", type=int, default=14)
    p.add_argument("--n-fft", type=int, default=256)
    p.add_argument("--win-length", type=int, default=200)
    p.add_argument("--hop-length", type=int, default=80)
    p.add_argument("--pre-emph", type=float, default=0.97)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected-min-samples", type=int, default=1000)
    p.add_argument("--allow-small-sample", action="store_true")
    p.add_argument("--mfcc-cache", type=Path, default=Path("artifacts/features/mfcc/fold0/mfcc_stats.npy"))
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


def clip_path_from_key(clips_root: Path, key: Tuple[str, str, str]) -> Path:
    show, ep, clip = key
    return clips_root / show / ep / f"{show}_{ep}_{clip}.wav"


def extract_mfcc_stats(
    wav_path: Path,
    sr: int,
    n_mfcc: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    pre_emph: float,
) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr)
    if y is None or len(y) < 2:
        return np.zeros((78,), dtype=np.float32)

    y = librosa.effects.preemphasis(y, coef=pre_emph)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window="hann",
    )

    mfcc = mfcc[1:, :]  # drop C0 => 13 coefficients
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.hstack(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta.mean(axis=1),
            delta.std(axis=1),
            delta2.mean(axis=1),
            delta2.std(axis=1),
        ]
    )
    return feat.astype(np.float32)


class FusionCNN(nn.Module):
    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
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
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(1)  # [B,1,F]
        z = self.net(z).squeeze(-1)
        return self.head(z)


def train_eval_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionCNN(in_dim=x_train.shape[1], dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
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

        train_loss = running / len(train_ds)
        history.append({"epoch": epoch, "train_loss": train_loss})

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

    metrics = {
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "f1": float(f1_score(all_true, all_pred)),
        "precision": float(precision_score(all_true, all_pred, zero_division=0)),
        "recall": float(recall_score(all_true, all_pred, zero_division=0)),
        "device": str(device),
    }
    return metrics, history


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ssl_path = args.features_root / args.model_alias / args.fold / f"layer_{args.layer}.npy"
    if not ssl_path.exists():
        raise FileNotFoundError(f"Missing SSL cached layer file: {ssl_path}")

    ssl = np.load(ssl_path)
    if ssl.ndim != 2:
        ssl = ssl.reshape(ssl.shape[0], -1)

    n = ssl.shape[0]
    if n < args.expected_min_samples and not args.allow_small_sample:
        raise RuntimeError(
            f"Sample count ({n}) is below expected-min-samples ({args.expected_min_samples}). "
            "Use full cache or pass --allow-small-sample intentionally."
        )

    target_col = LABEL_COLUMNS[args.target]
    label_map = {}
    label_map.update(load_label_map(args.sep_labels, target_col))
    label_map.update(load_label_map(args.fluency_labels, target_col))

    clip_keys = sorted_clip_keys(args.clips_root)
    if len(clip_keys) < n:
        raise RuntimeError(f"Clip count ({len(clip_keys)}) lower than cached rows ({n}).")

    clip_keys = clip_keys[:n]
    y = np.array([label_map.get(k, 0) for k in clip_keys], dtype=np.int64)

    # Build/load MFCC cache aligned with sorted clip order.
    mfcc_cache = args.mfcc_cache
    if mfcc_cache.exists():
        mfcc = np.load(mfcc_cache)
        if mfcc.shape[0] < n:
            raise RuntimeError(
                f"MFCC cache rows ({mfcc.shape[0]}) lower than needed ({n}). Delete cache to rebuild."
            )
        mfcc = mfcc[:n]
    else:
        mfcc_cache.parent.mkdir(parents=True, exist_ok=True)
        feats = []
        for i, key in enumerate(clip_keys, start=1):
            wav_path = clip_path_from_key(args.clips_root, key)
            if not wav_path.exists():
                feats.append(np.zeros((78,), dtype=np.float32))
            else:
                feats.append(
                    extract_mfcc_stats(
                        wav_path,
                        sr=args.sr,
                        n_mfcc=args.n_mfcc,
                        n_fft=args.n_fft,
                        win_length=args.win_length,
                        hop_length=args.hop_length,
                        pre_emph=args.pre_emph,
                    )
                )
            if i % 2000 == 0:
                print(f"mfcc_cache_progress={i}/{n}")
        mfcc = np.stack(feats, axis=0)
        np.save(mfcc_cache, mfcc)

    # Shared train/test split for fair MFCC-dim comparison.
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    x_ssl_train = ssl[train_idx]
    x_ssl_test = ssl[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # SSL branch: scale + PCA on train only.
    ssl_scaler = StandardScaler()
    x_ssl_train = ssl_scaler.fit_transform(x_ssl_train)
    x_ssl_test = ssl_scaler.transform(x_ssl_test)

    pca_dim = min(args.ssl_pca_dim, x_ssl_train.shape[1])
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    x_ssl_train_p = pca.fit_transform(x_ssl_train)
    x_ssl_test_p = pca.transform(x_ssl_test)

    mfcc_dims = sorted({d for d in args.mfcc_dims if 1 <= d <= mfcc.shape[1]})
    if not mfcc_dims:
        raise RuntimeError("No valid MFCC dimensions provided in --mfcc-dims.")

    x_mfcc_train = mfcc[train_idx]
    x_mfcc_test = mfcc[test_idx]

    rows = []
    histories: Dict[int, List[Dict[str, float]]] = {}

    for d_mfcc in mfcc_dims:
        mfcc_scaler = StandardScaler()
        x_m_train = mfcc_scaler.fit_transform(x_mfcc_train[:, :d_mfcc])
        x_m_test = mfcc_scaler.transform(x_mfcc_test[:, :d_mfcc])

        x_train = np.hstack([x_ssl_train_p, x_m_train]).astype(np.float32)
        x_test = np.hstack([x_ssl_test_p, x_m_test]).astype(np.float32)

        metrics, history = train_eval_one(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            seed=args.seed,
        )

        histories[d_mfcc] = history
        row = {
            "mfcc_dim": int(d_mfcc),
            "fusion_dim": int(x_train.shape[1]),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "device": metrics["device"],
        }
        rows.append(row)
        print(
            f"mfcc_dim={d_mfcc} fusion_dim={row['fusion_dim']} "
            f"f1={row['f1']:.6f} acc={row['accuracy']:.6f}"
        )

    rows = sorted(rows, key=lambda r: r["mfcc_dim"])
    best = max(rows, key=lambda r: r["f1"])
    best_dim = int(best["mfcc_dim"])

    sweep_csv = args.out_dir / "c1_mfcc_sweep_results.csv"
    with sweep_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mfcc_dim", "fusion_dim", "accuracy", "f1", "precision", "recall", "device"],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_hist_csv = args.out_dir / "c1_best_train_history.csv"
    with best_hist_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(histories[best_dim])

    run_report = {
        "experiment": "C1",
        "target": args.target,
        "model_alias": args.model_alias,
        "fold": args.fold,
        "layer": args.layer,
        "num_samples": int(n),
        "num_pos": int(np.sum(y)),
        "num_neg": int(n - np.sum(y)),
        "ssl_pca_dim": int(pca_dim),
        "mfcc_dims": mfcc_dims,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "best_mfcc_dim": best_dim,
        "best_accuracy": float(best["accuracy"]),
        "best_f1": float(best["f1"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "device": best["device"],
        "mfcc_cache": str(mfcc_cache),
        "sweep_csv": str(sweep_csv),
        "best_history_csv": str(best_hist_csv),
    }

    report_json = args.out_dir / "c1_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    fig_path = None
    try:
        import matplotlib.pyplot as plt

        dims = [r["mfcc_dim"] for r in rows]
        f1s = [r["f1"] for r in rows]
        fig = plt.figure(figsize=(8, 4.6))
        ax = fig.add_subplot(111)
        ax.plot(dims, f1s, marker="o")
        ax.set_title(f"C1 MFCC Sweep ({args.target}, {args.model_alias}, layer {args.layer})")
        ax.set_xlabel("MFCC feature dimension")
        ax.set_ylabel("F1")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig_path = args.fig_dir / "c1_mfcc_sweep_f1.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    except Exception:
        fig_path = None

    print("C1 completed.")
    print(f"Report: {report_json}")
    print(f"Sweep: {sweep_csv}")
    print(f"Best history: {best_hist_csv}")
    if fig_path:
        print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
