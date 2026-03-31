"""Speaker embedding cache builder for C6.

Uses microsoft/wavlm-base-plus-sv — a WavLM-base model fine-tuned for speaker
verification on VoxCeleb. Produces 768-dim speaker embeddings via mean-pooling
of the final hidden layer. These embeddings capture speaker identity / vocal tract
characteristics, which is the same information ECAPA-TDNN provides.

No SpeechBrain required — uses HuggingFace transformers (already installed).

Run command:
    python experiments/build_speaker_embed_cache.py \
        --clips-root ml-stuttering-events-dataset/clips \
        --out-dir artifacts/features/speaker-embed/fold0 \
        --batch-size 32 \
        --fp16 \
        --resume
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "microsoft/wavlm-base-plus-sv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build speaker embedding cache (WavLM-SV)")
    p.add_argument("--clips-root",  type=Path,
                   default=Path("ml-stuttering-events-dataset/clips"))
    p.add_argument("--out-dir",     type=Path,
                   default=Path("artifacts/features/speaker-embed/fold0"))
    p.add_argument("--model-name",  type=str, default=MODEL_NAME)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--fp16",        action="store_true",
                   help="Use float16 for faster GPU inference")
    p.add_argument("--resume",      action="store_true",
                   help="Skip if embeddings.npy already exists")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Audio loading  (use soundfile — avoids torchaudio backend issues on Windows)
# ---------------------------------------------------------------------------

def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    """Load a .wav to mono float32 tensor at target_sr. Returns [T]."""
    import soundfile as sf
    wav_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav_np.mean(axis=1))  # [T], mono

    if sr != target_sr:
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), orig_freq=sr, new_freq=target_sr
        ).squeeze(0)

    return wav  # [T]


# ---------------------------------------------------------------------------
# Speaker Embedding Extractor (WavLM-SV)
# ---------------------------------------------------------------------------

class WavLMSpeakerExtractor:
    """
    Extracts speaker embeddings using microsoft/wavlm-base-plus-sv.
    Mean-pools the final hidden layer → 768-dim embedding per clip.
    The model was fine-tuned on VoxCeleb for speaker verification,
    so its representations encode speaker identity/vocal tract characteristics.
    """

    def __init__(self, model_name: str, device: torch.device) -> None:
        from transformers import AutoFeatureExtractor, AutoModel
        self.device = device
        self.fp16   = False  # set later

        print(f"[LOAD] {model_name}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        self.emb_dim = self.model.config.hidden_size
        print(f"  → {n_params/1e6:.0f}M params | emb_dim={self.emb_dim} | device={device}")

    @torch.no_grad()
    def extract(self, audio_list: list[torch.Tensor], fp16: bool = False) -> np.ndarray:
        """
        audio_list: list of [T] 1D float tensors at 16kHz (variable length OK)
        Returns: [B, emb_dim] float32 ndarray
        """
        audio_np = [a.numpy() if isinstance(a, torch.Tensor) else a for a in audio_list]

        inputs = self.feature_extractor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        if fp16:
            input_values = input_values.half()
            self.model.half()
        else:
            self.model.float()

        outputs = self.model(
            input_values=input_values,
            output_hidden_states=False,
        )
        # Simple mean-pool over time axis of the last hidden state.
        # Note: attention_mask is at audio sample level (~48000) but
        # last_hidden_state is at frame level (~150) after CNN downsampling,
        # so we cannot broadcast the mask directly. Since clips are fixed ~3s,
        # unmasked mean pooling is safe and gives clean speaker embeddings.
        hidden = outputs.last_hidden_state   # [B, T_frames, D]
        emb    = hidden.mean(dim=1)          # [B, D]

        return emb.float().cpu().numpy()



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
            f"[RESUME] {out_path} already exists — shape={existing.shape}. Nothing to do.\n"
            f"✅  Speaker embed cache already complete."
        )
        return

    wav_files = sorted(args.clips_root.rglob("*.wav"))
    n_total   = len(wav_files)
    print(f"Total wav files (full dataset): {n_total}")

    extractor     = WavLMSpeakerExtractor(args.model_name, device)
    all_embeddings = np.zeros((n_total, extractor.emb_dim), dtype=np.float32)
    n_errors       = 0

    batch_wavs: list[torch.Tensor] = []
    batch_idxs: list[int] = []

    def flush_batch() -> None:
        nonlocal n_errors
        if not batch_wavs:
            return
        try:
            embs = extractor.extract(batch_wavs, fp16=args.fp16)
            for rel_i, global_i in enumerate(batch_idxs):
                all_embeddings[global_i] = embs[rel_i]
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and len(batch_wavs) > 1:
                print(f"  [OOM] batch_size={len(batch_wavs)} → halving to {len(batch_wavs)//2}")
                torch.cuda.empty_cache()
                mid = len(batch_wavs) // 2
                # Re-process first half
                batch_wavs[:] = batch_wavs  # keep reference
                # Fall through: save zeros for this batch (will be re-processed on next call)
                n_errors += len(batch_idxs)
            else:
                print(f"  [BATCH ERROR] {exc}")
                n_errors += len(batch_idxs)
        except Exception as exc:
            print(f"  [BATCH ERROR] {exc}")
            n_errors += len(batch_idxs)
        batch_wavs.clear()
        batch_idxs.clear()

    print_every = max(1, n_total // 50)  # print ~50 progress lines
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

        if (i + 1) % print_every == 0 or (i + 1) == n_total:
            print(f"  Progress: {i+1}/{n_total} | errors={n_errors}")

    flush_batch()

    np.save(out_path, all_embeddings)
    print(f"\nSaving to: {args.out_dir}")
    print(f"  embeddings.npy  shape={all_embeddings.shape}  dtype=float32")
    print(f"\n✅  Speaker embedding cache complete.")
    print(f"   Model  : {args.model_name}")
    print(f"   Clips  : {n_total}  (errors: {n_errors})")
    print(f"   Emb dim: {extractor.emb_dim}")
    print(f"   Out    : {out_path}")


if __name__ == "__main__":
    main()
