"""Build layer-wise SSL feature cache for A7 CKA/SVCCA probing.

Output layout:
artifacts/features/<model_alias>/<fold>/layer_<k>.npy
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build A7 feature cache")
    parser.add_argument(
        "--clips-root",
        type=Path,
        default=Path("ml-stuttering-events-dataset/clips"),
        help="Root directory containing .wav clips.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("artifacts/features"),
        help="Feature cache root.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/wav2vec2-base",
        help="HF model name used for feature extraction.",
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default="wav2vec2-base",
        help="Directory alias under out-root.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold0",
        help="Fold name used in cache path.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 3, 6, 9, 12],
        help="Transformer layer indices to cache.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1000,
        help="Maximum number of audio files to process.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sampling rate used by wav2vec2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic file sampling.",
    )
    return parser.parse_args()


def list_wavs(clips_root: Path) -> List[Path]:
    return sorted(clips_root.rglob("*.wav"))


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=sample_rate, mono=True)
    return wav.astype(np.float32)


def main() -> None:
    args = parse_args()

    if not args.clips_root.exists():
        raise FileNotFoundError(f"Clips root not found: {args.clips_root}")

    wav_files = list_wavs(args.clips_root)
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found under: {args.clips_root}")

    random.seed(args.seed)
    if args.max_files > 0 and len(wav_files) > args.max_files:
        wav_files = random.sample(wav_files, args.max_files)
        wav_files = sorted(wav_files)

    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2Model.from_pretrained(args.model_name).to(device)
    model.eval()

    # Collect one pooled vector per clip per layer.
    layer_vectors: Dict[int, List[np.ndarray]] = {k: [] for k in args.layers}

    for idx, wav_path in enumerate(wav_files, start=1):
        audio = load_audio(wav_path, args.sample_rate)
        inputs = processor(
            audio,
            sampling_rate=args.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            out = model(input_values, output_hidden_states=True)

        for layer in args.layers:
            hidden = out.hidden_states[layer]  # [1, T, D]
            vec = hidden.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
            layer_vectors[layer].append(vec)

        if idx % 50 == 0 or idx == len(wav_files):
            print(f"Processed {idx}/{len(wav_files)} files")

    save_dir = args.out_root / args.model_alias / args.fold
    save_dir.mkdir(parents=True, exist_ok=True)

    for layer in args.layers:
        arr = np.stack(layer_vectors[layer], axis=0)
        np.save(save_dir / f"layer_{layer}.npy", arr)

    print("A7 feature cache build completed.")
    print(f"Model: {args.model_name}")
    print(f"Alias: {args.model_alias}")
    print(f"Fold: {args.fold}")
    print(f"Saved to: {save_dir}")
    print(f"Layers: {args.layers}")
    print(f"Num files: {len(wav_files)}")


if __name__ == "__main__":
    main()
