"""C3: Multi-SSL temporal concatenation with linear projection + Transformer.

Workflow:
1) Load cached best-layer features from multiple SSL aliases
2) Build binary target labels from SEP-28k/FluencyBank CSVs
3) Project each SSL stream to shared dimension and treat streams as sequence
4) Fuse with Transformer encoder and classify with MLP head
5) Save metrics, per-stream salience proxy, history, and figure
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
    p = argparse.ArgumentParser(description="C3 multi-SSL temporal concat transformer")
    p.add_argument("--features-root", type=Path, default=Path("artifacts/features"))
    p.add_argument("--model-aliases", nargs="+", default=["wav2vec2-base", "benchmark-w2v2-base"])
    p.add_argument("--fold", type=str, default="fold0")
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--clips-root", type=Path, default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--sep-labels", type=Path, default=Path("ml-stuttering-events-dataset/SEP-28k_labels.csv"))
    p.add_argument("--fluency-labels", type=Path, default=Path("ml-stuttering-events-dataset/fluencybank_labels.csv"))
    p.add_argument("--target", choices=list(LABEL_COLUMNS.keys()), default="Block")
    p.add_argument("--proj-dim", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
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


def resolve_layer_cache_path(features_root: Path, alias: str, fold: str, layer: int) -> Path | None:
    direct = features_root / alias / fold / f"layer_{layer}.npy"
    if direct.exists():
        return direct

    alias_root = features_root / alias
    if not alias_root.exists() or not alias_root.is_dir():
        return None

    # Fallback: pick the first subdirectory containing the requested layer cache.
    for child in sorted(alias_root.iterdir()):
        if child.is_dir():
            candidate = child / f"layer_{layer}.npy"
            if candidate.exists():
                print(f"Using fallback fold '{child.name}' for alias '{alias}'")
                return candidate
    return None


class C3Model(nn.Module):
    def __init__(
        self,
        num_streams: int,
        in_dim: int,
        proj_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.stream_proj = nn.Linear(in_dim, proj_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        self.num_streams = num_streams

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, S, D]
        z = self.stream_proj(x)  # [B, S, P]
        z = self.encoder(z)  # [B, S, P]

        # Salience proxy from encoded stream magnitudes.
        salience = z.abs().mean(dim=2)  # [B, S]

        pooled = z.mean(dim=1)  # [B, P]
        logits = self.classifier(pooled)
        return logits, salience


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if len(args.model_aliases) < 2:
        raise RuntimeError("C3 requires at least two model aliases in --model-aliases.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arrays = []
    used_aliases = []
    for alias in args.model_aliases:
        p = resolve_layer_cache_path(args.features_root, alias, args.fold, args.layer)
        if p is None:
            print(
                f"Skipping missing cache for alias '{alias}' at requested fold '{args.fold}' "
                f"(no fallback folder with layer_{args.layer}.npy found)."
            )
            continue
        arr = np.load(p)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        arrays.append(arr)
        used_aliases.append(alias)

    if len(arrays) < 2:
        raise RuntimeError(
            "Need at least two available cached aliases for C3. "
            "Populate more caches or pass available aliases via --model-aliases."
        )

    n = min(a.shape[0] for a in arrays)
    d = arrays[0].shape[1]
    if n < args.expected_min_samples and not args.allow_small_sample:
        raise RuntimeError(
            f"Sample count ({n}) is below expected-min-samples ({args.expected_min_samples}). "
            "Use full cache or pass --allow-small-sample intentionally."
        )

    x = np.stack([a[:n] for a in arrays], axis=1)  # [N,S,D]

    target_col = LABEL_COLUMNS[args.target]
    label_map = {}
    label_map.update(load_label_map(args.sep_labels, target_col))
    label_map.update(load_label_map(args.fluency_labels, target_col))

    clip_keys = sorted_clip_keys(args.clips_root)
    if len(clip_keys) < n:
        raise RuntimeError(f"Clip count ({len(clip_keys)}) lower than cached rows ({n}).")

    y = np.array([label_map.get(k, 0) for k in clip_keys[:n]], dtype=np.int64)

    # Standardize stream-wise.
    x_flat = x.reshape(n, -1)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    x_train = x_flat[train_idx]
    x_test = x_flat[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).reshape(-1, len(used_aliases), d)
    x_test = scaler.transform(x_test).reshape(-1, len(used_aliases), d)

    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C3Model(
        num_streams=len(used_aliases),
        in_dim=d,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
    salience_batches = []

    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            logits, sal = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_pred.extend(pred)
            all_true.extend(yb.numpy().tolist())
            salience_batches.append(sal.cpu().numpy())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    prec = precision_score(all_true, all_pred, zero_division=0)
    rec = recall_score(all_true, all_pred, zero_division=0)

    sal = np.concatenate(salience_batches, axis=0)  # [Ntest,S]
    sal_mean = sal.mean(axis=0)
    sal_norm = sal_mean / (np.sum(sal_mean) + 1e-8)

    sal_csv = args.out_dir / "c3_stream_salience.csv"
    with sal_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_alias", "salience_share"])
        writer.writeheader()
        for alias, s in zip(used_aliases, sal_norm):
            writer.writerow({"model_alias": alias, "salience_share": float(s)})

    history_csv = args.out_dir / "c3_train_history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(history)

    run_report = {
        "experiment": "C3",
        "target": args.target,
        "device": str(device),
        "model_aliases_requested": args.model_aliases,
        "model_aliases_used": used_aliases,
        "fold": args.fold,
        "layer": args.layer,
        "num_samples": int(n),
        "num_pos": int(np.sum(y)),
        "num_neg": int(n - np.sum(y)),
        "feature_dim": int(d),
        "proj_dim": args.proj_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "stream_salience": {a: float(s) for a, s in zip(used_aliases, sal_norm)},
        "dominant_stream": used_aliases[int(np.argmax(sal_norm))],
        "salience_csv": str(sal_csv),
        "history_csv": str(history_csv),
    }

    report_json = args.out_dir / "c3_run_report.json"
    report_json.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    fig_path = None
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8.2, 4.6))
        ax = fig.add_subplot(111)
        ax.bar(used_aliases, sal_norm)
        ax.set_title(f"C3 Stream Salience ({args.target}, layer {args.layer})")
        ax.set_xlabel("SSL stream")
        ax.set_ylabel("Normalized salience")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig_path = args.fig_dir / "c3_stream_salience.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    except Exception:
        fig_path = None

    print("C3 completed.")
    print(f"Report: {report_json}")
    print(f"Salience: {sal_csv}")
    print(f"History: {history_csv}")
    if fig_path:
        print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
