"""C2: SSL + handcrafted groups ablation baseline.

Workflow:
1) Load cached SSL features from artifacts/features/<model>/<fold>/layer_<k>.npy
2) Build or load handcrafted feature cache aligned to clip ordering
3) Reduce SSL to PCA-32 and concatenate selected handcrafted groups
4) Train CNN-1D classifier for each ablation setting
5) Save ablation table, best-run report, history, and figure
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


GROUPS = {
    "energy": [0, 1],
    "zcr": [2, 3],
    "jitter": [4, 5, 6],
    "formant": [7, 8, 9],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C2 SSL+handcrafted ablation baseline")
    p.add_argument("--features-root", type=Path, default=Path("artifacts/features"))
    p.add_argument("--model-alias", type=str, default="wav2vec2-base")
    p.add_argument("--fold", type=str, default="fold0")
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--clips-root", type=Path, default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels", type=Path, default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path, default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--target", choices=list(LABEL_COLUMNS.keys()), default="Block")
    p.add_argument("--ssl-pca-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--sr", type=int, default=8000)
    p.add_argument("--frame-length", type=int, default=200)
    p.add_argument("--hop-length", type=int, default=80)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected-min-samples", type=int, default=1000)
    p.add_argument("--allow-small-sample", action="store_true")
    p.add_argument(
        "--handcrafted-cache",
        type=Path,
        default=Path("artifacts/features/handcrafted/fold0/c2_handcrafted.npy"),
    )
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


def lpc_formant_proxy(y: np.ndarray, sr: int) -> np.ndarray:
    if y.size < 256:
        return np.zeros((3,), dtype=np.float32)
    try:
        a = librosa.lpc(y, order=12)
        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]
        angz = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angz * (sr / (2 * np.pi))
        freqs = np.sort(freqs[(freqs > 90) & (freqs < 5000)])
        if freqs.size == 0:
            return np.zeros((3,), dtype=np.float32)
        out = np.zeros((3,), dtype=np.float32)
        k = min(3, freqs.size)
        out[:k] = freqs[:k].astype(np.float32)
        return out
    except Exception:
        return np.zeros((3,), dtype=np.float32)


def extract_handcrafted(wav_path: Path, sr: int, frame_length: int, hop_length: int) -> np.ndarray:
    try:
        y, _ = librosa.load(str(wav_path), sr=sr)
    except Exception:
        return np.zeros((10,), dtype=np.float32)

    if y is None or len(y) < 2:
        return np.zeros((10,), dtype=np.float32)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).flatten()
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length).flatten()

    energy_mean = float(np.mean(rms)) if rms.size else 0.0
    energy_std = float(np.std(rms)) if rms.size else 0.0
    zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0
    zcr_std = float(np.std(zcr)) if zcr.size else 0.0

    # Jitter proxy from frame-level F0 variation.
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        f0 = f0[f0 > 0]
    except Exception:
        f0 = np.array([], dtype=np.float32)

    if f0.size >= 2:
        d = np.abs(np.diff(f0))
        f0_mean = float(np.mean(f0))
        jitter_local = float(np.mean(d) / (f0_mean + 1e-8))
        jitter_rap = float(np.mean(np.abs(f0[2:] - (f0[1:-1] + f0[:-2]) / 2.0)) / (f0_mean + 1e-8)) if f0.size >= 3 else 0.0
        jitter_ppq5 = (
            float(np.mean(np.abs(f0[4:] - np.mean(np.stack([f0[3:-1], f0[2:-2], f0[1:-3], f0[:-4]], axis=0), axis=0))) / (f0_mean + 1e-8))
            if f0.size >= 5
            else 0.0
        )
    else:
        jitter_local, jitter_rap, jitter_ppq5 = 0.0, 0.0, 0.0

    formants = lpc_formant_proxy(y, sr)

    feat = np.array(
        [
            energy_mean,
            energy_std,
            zcr_mean,
            zcr_std,
            jitter_local,
            jitter_rap,
            jitter_ppq5,
            float(formants[0]),
            float(formants[1]),
            float(formants[2]),
        ],
        dtype=np.float32,
    )
    return feat


class FusionCNN(nn.Module):
    def __init__(self, dropout: float):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(1)
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
    model = FusionCNN(dropout=dropout).to(device)
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

    hc_cache = args.handcrafted_cache
    if hc_cache.exists():
        handcrafted = np.load(hc_cache)
        if handcrafted.shape[0] < n:
            raise RuntimeError(
                f"Handcrafted cache rows ({handcrafted.shape[0]}) lower than needed ({n}). Delete cache to rebuild."
            )
        handcrafted = handcrafted[:n]
    else:
        hc_cache.parent.mkdir(parents=True, exist_ok=True)
        feats = []
        for i, key in enumerate(clip_keys, start=1):
            wav_path = clip_path_from_key(args.clips_root, key)
            if not wav_path.exists():
                feats.append(np.zeros((10,), dtype=np.float32))
            else:
                feats.append(
                    extract_handcrafted(
                        wav_path,
                        sr=args.sr,
                        frame_length=args.frame_length,
                        hop_length=args.hop_length,
                    )
                )
            if i % 2000 == 0:
                print(f"handcrafted_cache_progress={i}/{n}")
        handcrafted = np.stack(feats, axis=0)
        np.save(hc_cache, handcrafted)

    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    x_ssl_train = ssl[train_idx]
    x_ssl_test = ssl[test_idx]
    x_h_train = handcrafted[train_idx]
    x_h_test = handcrafted[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    ssl_scaler = StandardScaler()
    x_ssl_train = ssl_scaler.fit_transform(x_ssl_train)
    x_ssl_test = ssl_scaler.transform(x_ssl_test)

    pca_dim = min(args.ssl_pca_dim, x_ssl_train.shape[1])
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    x_ssl_train_p = pca.fit_transform(x_ssl_train)
    x_ssl_test_p = pca.transform(x_ssl_test)

    setting_to_cols = {
        "ssl_only": [],
        "ssl_plus_energy": GROUPS["energy"],
        "ssl_plus_zcr": GROUPS["zcr"],
        "ssl_plus_jitter": GROUPS["jitter"],
        "ssl_plus_formant": GROUPS["formant"],
        "ssl_plus_all": sorted([i for v in GROUPS.values() for i in v]),
    }

    rows = []
    histories: Dict[str, List[Dict[str, float]]] = {}

    for setting, cols in setting_to_cols.items():
        if cols:
            h_scaler = StandardScaler()
            x_h_train_s = h_scaler.fit_transform(x_h_train[:, cols])
            x_h_test_s = h_scaler.transform(x_h_test[:, cols])
            x_train = np.hstack([x_ssl_train_p, x_h_train_s]).astype(np.float32)
            x_test = np.hstack([x_ssl_test_p, x_h_test_s]).astype(np.float32)
        else:
            x_train = x_ssl_train_p.astype(np.float32)
            x_test = x_ssl_test_p.astype(np.float32)

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

        histories[setting] = history
        row = {
            "setting": setting,
            "handcrafted_dim": int(len(cols)),
            "fusion_dim": int(x_train.shape[1]),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "device": metrics["device"],
        }
        rows.append(row)
        print(
            f"setting={setting} fusion_dim={row['fusion_dim']} "
            f"f1={row['f1']:.6f} acc={row['accuracy']:.6f}"
        )

    best = max(rows, key=lambda r: r["f1"])
    best_setting = best["setting"]

    ablation_csv = args.out_dir / "c2_group_ablation_results.csv"
    with ablation_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["setting", "handcrafted_dim", "fusion_dim", "accuracy", "f1", "precision", "recall", "device"],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_hist_csv = args.out_dir / "c2_best_train_history.csv"
    with best_hist_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(histories[best_setting])

    group_meta_json = args.out_dir / "c2_handcrafted_groups.json"
    group_meta_json.write_text(json.dumps(GROUPS, indent=2), encoding="utf-8")

    run_report = {
        "experiment": "C2",
        "target": args.target,
        "model_alias": args.model_alias,
        "fold": args.fold,
        "layer": args.layer,
        "num_samples": int(n),
        "num_pos": int(np.sum(y)),
        "num_neg": int(n - np.sum(y)),
        "ssl_pca_dim": int(pca_dim),
        "groups": GROUPS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "best_setting": best_setting,
        "best_accuracy": float(best["accuracy"]),
        "best_f1": float(best["f1"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "device": best["device"],
        "handcrafted_cache": str(hc_cache),
        "ablation_csv": str(ablation_csv),
        "best_history_csv": str(best_hist_csv),
        "groups_json": str(group_meta_json),
    }

    report_json = args.out_dir / "c2_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    fig_path = None
    try:
        import matplotlib.pyplot as plt

        labels = [r["setting"] for r in rows]
        f1s = [r["f1"] for r in rows]
        fig = plt.figure(figsize=(10, 4.8))
        ax = fig.add_subplot(111)
        ax.bar(labels, f1s)
        ax.set_title(f"C2 Handcrafted Group Ablation ({args.target}, {args.model_alias}, layer {args.layer})")
        ax.set_xlabel("Setting")
        ax.set_ylabel("F1")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig_path = args.fig_dir / "c2_group_ablation_f1.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    except Exception:
        fig_path = None

    print("C2 completed.")
    print(f"Report: {report_json}")
    print(f"Ablation: {ablation_csv}")
    print(f"Best history: {best_hist_csv}")
    print(f"Groups: {group_meta_json}")
    if fig_path:
        print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
