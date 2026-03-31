"""ECAPA-TDNN speaker embedding cache builder for C6.

Extracts 192-dim ECAPA-TDNN x-vector speaker embeddings for all clips
using SpeechBrain's pretrained model (speechbrain/spkrec-ecapa-voxceleb).

Prerequisites:
    pip install speechbrain

Run command:
    python experiments/build_ecapa_cache.py \
        --clips-root ml-stuttering-events-dataset/clips \
        --out-dir artifacts/features/ecapa-tdnn/fold0 \
        --batch-size 64 \
        --resume
"""

from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ECAPA-TDNN speaker embedding cache")
    p.add_argument("--clips-root", type=Path,
                   default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--out-dir",    type=Path,
                   default=Path("artifacts/features/ecapa-tdnn/fold0"))
    p.add_argument("--model-source", type=str,
                   default="speechbrain/spkrec-ecapa-voxceleb",
                   help="SpeechBrain Hub source for ECAPA-TDNN model")
    p.add_argument("--batch-size", type=int,  default=64)
    p.add_argument("--sample-rate",type=int,  default=16000)
    p.add_argument("--resume",     action="store_true",
                   help="Skip extraction if embeddings.npy already exists")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    """Load a .wav to mono float32 tensor at target_sr. Returns [T]."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)  # [T]


# ---------------------------------------------------------------------------
# ECAPA-TDNN extractor wrapper
# ---------------------------------------------------------------------------

class ECAPAExtractor:
    """
    Wraps SpeechBrain's ECAPA-TDNN speaker encoder.
    Outputs 192-dim embeddings per clip.
    """

    def __init__(self, model_source: str, device: torch.device) -> None:
        self.device = device
        print(f"[LOAD] SpeechBrain ECAPA-TDNN: {model_source}")

        # Try new SpeechBrain >= 1.0 API first, fall back to old API
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            try:
                from speechbrain.pretrained import EncoderClassifier
            except ImportError:
                raise ImportError(
                    "SpeechBrain is not installed.\n"
                    "Install with: pip install speechbrain"
                )

        self.model = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=f"pretrained_models/{model_source.replace('/', '_')}",
            run_opts={"device": str(device)},
        )
        self.model.eval()

        # Determine embedding dim by running a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 16000, device=device)
            emb = self.model.encode_batch(dummy)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            self.emb_dim = int(emb.reshape(1, -1).shape[1])
        print(f"  → ECAPA embedding dim: {self.emb_dim} | device={device}")

    @torch.no_grad()
    def extract(self, waveforms: list[torch.Tensor]) -> np.ndarray:
        """
        waveforms: list of [T] 1D tensors (may have different lengths)
        Returns: np.ndarray [B, emb_dim]
        """
        # Pad waveforms to same length for batched inference
        max_len = max(w.shape[0] for w in waveforms)
        padded = torch.zeros(len(waveforms), max_len, device=self.device)
        for i, w in enumerate(waveforms):
            padded[i, :w.shape[0]] = w.to(self.device)

        emb = self.model.encode_batch(padded)   # [B, 1, D] or [B, D]
        emb = emb.reshape(len(waveforms), -1)   # [B, D]
        return emb.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "embeddings.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.resume and out_path.exists():
        existing = np.load(out_path)
        print(
            f"[RESUME] {out_path} already exists — "
            f"shape={existing.shape}. Nothing to do.\n"
            f"✅  Cache already complete."
        )
        return

    # Collect all clips (canonical sorted order — must match other caches)
    wav_files = sorted(args.clips_root.rglob("*.wav"))
    n_total   = len(wav_files)
    print(f"Total wav files (full dataset): {n_total}")

    extractor = ECAPAExtractor(args.model_source, device)
    emb_dim   = extractor.emb_dim

    # Pre-allocate output array
    all_embeddings = np.zeros((n_total, emb_dim), dtype=np.float32)
    n_errors = 0

    batch_wavs: list[torch.Tensor] = []
    batch_idxs: list[int] = []

    def flush_batch() -> None:
        nonlocal n_errors
        if not batch_wavs:
            return
        try:
            embs = extractor.extract(batch_wavs)
            for rel_idx, global_idx in enumerate(batch_idxs):
                all_embeddings[global_idx] = embs[rel_idx]
        except Exception as exc:
            print(f"  [BATCH ERROR] idx={batch_idxs[0]}: {exc}")
            n_errors += len(batch_idxs)
        batch_wavs.clear()
        batch_idxs.clear()

    for i, wav_path in enumerate(wav_files):
        try:
            wav = load_audio(wav_path, args.sample_rate)
            batch_wavs.append(wav)
            batch_idxs.append(i)
        except Exception as exc:
            print(f"  [LOAD ERROR] {wav_path.name}: {exc}")
            n_errors += 1
            continue

        if len(batch_wavs) >= args.batch_size:
            flush_batch()
            if (i + 1) % (args.batch_size * 10) == 0 or (i + 1) == n_total:
                print(f"  Progress: {i+1}/{n_total} | errors={n_errors}")

    flush_batch()  # remainder
    print(f"  Progress: {n_total}/{n_total} | errors={n_errors}")

    # Save
    np.save(out_path, all_embeddings)
    print(f"\nSaving to: {args.out_dir}")
    print(f"  embeddings.npy  shape={all_embeddings.shape}  dtype=float32")
    print(f"\n✅  ECAPA-TDNN cache complete.")
    print(f"   Clips   : {n_total}  (errors: {n_errors})")
    print(f"   Emb dim : {emb_dim}")
    print(f"   Out     : {out_path}")


if __name__ == "__main__":
    main()
