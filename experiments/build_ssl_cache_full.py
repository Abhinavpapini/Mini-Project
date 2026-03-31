"""build_ssl_cache_full.py — Full-dataset SSL layer feature cache builder.

Processes ALL wav clips in canonical sorted order (same ordering used by all experiments).
Saves per-layer mean-pooled vectors to:
    artifacts/features/<model-alias>/<fold>/layer_<k>.npy

Supported model families: wav2vec2, hubert, whisper

Run commands:
    # HuBERT-large (layers 1,6,12,18,21,23)
    python experiments/build_ssl_cache_full.py \
        --model-name facebook/hubert-large-ll60k \
        --model-alias hubert-large \
        --model-type hubert \
        --layers 1 6 12 18 21 23 \
        --batch-size 16 --fold fold0

    # Whisper-large-v3 (layers 8,16,24,28,31)
    python experiments/build_ssl_cache_full.py \
        --model-name openai/whisper-large-v3 \
        --model-alias whisper-large \
        --model-type whisper \
        --layers 8 16 24 28 31 \
        --batch-size 8 --fold fold0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-dataset SSL feature cache builder")
    p.add_argument("--clips-root", type=Path, default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--out-root",   type=Path, default=Path("artifacts/features"))
    p.add_argument("--model-name", type=str,  required=True, help="HuggingFace model name or local path")
    p.add_argument("--model-alias",type=str,  required=True, help="Directory alias under --out-root")
    p.add_argument("--model-type", choices=["wav2vec2", "hubert", "whisper"], required=True)
    p.add_argument("--fold",       type=str,  default="fold0")
    p.add_argument("--layers",     type=int,  nargs="+", required=True,
                   help="1-indexed transformer layer indices to cache")
    p.add_argument("--batch-size", type=int,  default=16)
    p.add_argument("--sample-rate",type=int,  default=16000)
    p.add_argument("--max-dur",    type=float,default=5.0,   help="Truncate clips longer than this (sec)")
    p.add_argument("--fp16",       action="store_true",      help="Use AMP fp16 (saves VRAM, ~2× faster)")
    p.add_argument("--resume",     action="store_true",      help="Skip layers already saved on disk")
    p.add_argument("--seed",       type=int,  default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def list_wavs(clips_root: Path) -> List[Path]:
    """Canonical sorted order — must match every other experiment's ordering."""
    return sorted(clips_root.rglob("*.wav"))


def load_audio_safe(path: Path, sr: int, max_samples: int) -> Tuple[np.ndarray, bool]:
    """Load mono audio; return (zeros, False) on any error."""
    try:
        import librosa
        wav, _ = librosa.load(str(path), sr=sr, mono=True)
        if wav is None or len(wav) == 0:
            return np.zeros(sr * 3, dtype=np.float32), False
        return wav[:max_samples].astype(np.float32), True
    except Exception:
        return np.zeros(sr * 3, dtype=np.float32), False


# ---------------------------------------------------------------------------
# Model extractors
# ---------------------------------------------------------------------------

class Wav2Vec2HubertExtractor:
    """Works for facebook/wav2vec2-* and facebook/hubert-* checkpoints."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        from transformers import AutoFeatureExtractor, AutoModel
        print(f"[LOAD] {model_name}")
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  → {n_params:.0f}M parameters | device={device}")

    @torch.no_grad()
    def extract(
        self,
        audio_batch: List[np.ndarray],
        layers: List[int],
        fp16: bool,
    ) -> Dict[int, np.ndarray]:
        inputs = self.processor(
            audio_batch,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(16000 * 5.0),
        )
        input_values = inputs.input_values.to(self.device)
        attn_mask = inputs.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        with torch.autocast("cuda", dtype=torch.float16, enabled=(fp16 and self.device.type == "cuda")):
            out = self.model(
                input_values,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        hidden_states = out.hidden_states  # tuple len = num_layers + 1

        result: Dict[int, np.ndarray] = {}
        for layer in layers:
            if layer >= len(hidden_states):
                raise ValueError(
                    f"Layer {layer} requested but model has only {len(hidden_states)-1} transformer layers."
                )
            hs = hidden_states[layer]  # [B, T, D]
            result[layer] = hs.mean(dim=1).float().cpu().numpy()
        return result


class WhisperExtractor:
    """Works for openai/whisper-* checkpoints (encoder hidden states only)."""

    HOP_LENGTH = 160  # Whisper at 16 kHz → 1 mel frame = 10 ms

    def __init__(self, model_name: str, device: torch.device) -> None:
        from transformers import WhisperFeatureExtractor, WhisperModel
        print(f"[LOAD] {model_name}")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        n_params = sum(p.numel() for p in self.model.encoder.parameters()) / 1e6
        print(f"  → Encoder {n_params:.0f}M parameters | device={device}")

    @torch.no_grad()
    def extract(
        self,
        audio_batch: List[np.ndarray],
        audio_lengths: List[int],
        layers: List[int],
        fp16: bool,
    ) -> Dict[int, np.ndarray]:
        inputs = self.feature_extractor(
            audio_batch,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",       # must pad to exactly 3000 mel frames
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(self.device)  # [B, 80, 3000]

        with torch.autocast("cuda", dtype=torch.float16, enabled=(fp16 and self.device.type == "cuda")):
            enc_out = self.model.encoder(
                input_features=input_features,
                output_hidden_states=True,
            )
        hidden_states = enc_out.hidden_states  # tuple len = num_layers + 1

        result: Dict[int, np.ndarray] = {}
        for layer in layers:
            if layer >= len(hidden_states):
                raise ValueError(
                    f"Layer {layer} requested but Whisper encoder has only {len(hidden_states)-1} layers."
                )
            hs = hidden_states[layer].float()  # [B, T, D]
            # Mean-pool only over frames that correspond to the actual audio duration.
            pooled_list = []
            for b, audio_len in enumerate(audio_lengths):
                n_frames = min(math.ceil(audio_len / self.HOP_LENGTH), hs.shape[1])
                n_frames = max(n_frames, 1)
                pooled_list.append(hs[b, :n_frames, :].mean(dim=0))
            result[layer] = torch.stack(pooled_list, dim=0).cpu().numpy()
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Discover all wav files in canonical order ---
    wav_files = list_wavs(args.clips_root)
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found under: {args.clips_root}")
    print(f"Total wav files (full dataset): {len(wav_files)}")

    save_dir = args.out_root / args.model_alias / args.fold
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume: skip layers already on disk ---
    layers_to_run = []
    for layer in args.layers:
        out_path = save_dir / f"layer_{layer}.npy"
        if args.resume and out_path.exists():
            print(f"  [SKIP] layer_{layer}.npy already exists")
        else:
            layers_to_run.append(layer)

    if not layers_to_run:
        print("All requested layers cached. Done.")
        return

    print(f"Layers to build: {layers_to_run}")

    # --- Load model ---
    max_samples = int(args.sample_rate * args.max_dur)
    is_whisper = args.model_type == "whisper"
    if is_whisper:
        extractor = WhisperExtractor(args.model_name, device)
    else:
        extractor = Wav2Vec2HubertExtractor(args.model_name, device)

    # --- Feature accumulation ---
    layer_vectors: Dict[int, List[np.ndarray]] = {k: [] for k in layers_to_run}
    error_count = 0
    n_total = len(wav_files)
    batch_size = args.batch_size

    i = 0
    while i < n_total:
        batch_paths = wav_files[i: i + batch_size]
        audio_batch: List[np.ndarray] = []
        audio_lengths: List[int] = []
        for p in batch_paths:
            wav, ok = load_audio_safe(p, args.sample_rate, max_samples)
            if not ok:
                error_count += 1
            audio_batch.append(wav)
            audio_lengths.append(len(wav))

        try:
            if is_whisper:
                batch_result = extractor.extract(audio_batch, audio_lengths, layers_to_run, args.fp16)
            else:
                batch_result = extractor.extract(audio_batch, layers_to_run, args.fp16)

            for layer in layers_to_run:
                layer_vectors[layer].append(batch_result[layer])

            i += len(batch_paths)

            if i % max(batch_size * 10, 1) == 0 or i >= n_total:
                print(f"  Progress: {min(i, n_total)}/{n_total} | errors={error_count}")

        except RuntimeError as exc:
            if "memory" in str(exc).lower() or "cuda" in str(exc).lower():
                torch.cuda.empty_cache()
                new_bs = max(1, batch_size // 2)
                print(f"  [WARN] CUDA OOM — reducing batch_size {batch_size} → {new_bs}")
                batch_size = new_bs
                # Do NOT advance i — retry same batch with smaller batch_size
            else:
                raise

    # --- Save ---
    print(f"\nSaving to: {save_dir}")
    for layer in layers_to_run:
        arr = np.concatenate(layer_vectors[layer], axis=0)
        out_path = save_dir / f"layer_{layer}.npy"
        np.save(out_path, arr)
        print(f"  layer_{layer}.npy  shape={arr.shape}  dtype={arr.dtype}")

    print("\n✅  Cache build complete.")
    print(f"   Model   : {args.model_name}")
    print(f"   Alias   : {args.model_alias}")
    print(f"   Clips   : {n_total}  (errors/zeros: {error_count})")
    print(f"   Layers  : {layers_to_run}")
    print(f"   Out dir : {save_dir}")


if __name__ == "__main__":
    main()
