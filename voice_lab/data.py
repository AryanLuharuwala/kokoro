"""Dataset for (audio_clip, transcript) training pairs.

Expects a manifest file: one JSON object per line with at least::

    {"audio": "clips/0001.wav", "text": "..."}

Audio paths are resolved relative to the manifest file. Audio is resampled
to the model's sample rate (24 kHz) and returned as a float32 mono tensor
in [-1, 1].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, NamedTuple

import torch


SAMPLE_RATE = 24000


class Clip(NamedTuple):
    audio: torch.Tensor  # [num_samples], float32, mono, in [-1, 1]
    text: str
    source: str  # original audio path, for logs


def _load_audio(path: Path, target_sr: int) -> torch.Tensor:
    """Load mono audio at ``target_sr`` Hz. Uses soundfile + resampy if needed."""
    try:
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(
            "voice_lab needs `soundfile` to load audio. "
            "Install with: pip install 'kokoro[training]'"
        ) from e
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        try:
            import resampy
        except ImportError as e:
            raise RuntimeError(
                f"Audio at {path} is {sr} Hz but model needs {target_sr} Hz. "
                "Install resampy: pip install 'kokoro[training]'"
            ) from e
        wav = resampy.resample(wav, sr, target_sr)
    return torch.from_numpy(wav).float()


class VoiceDataset:
    """Iterable of ``Clip`` objects loaded from a JSONL manifest."""

    def __init__(self, manifest: str | Path, sample_rate: int = SAMPLE_RATE):
        self.manifest = Path(manifest)
        if not self.manifest.exists():
            raise FileNotFoundError(self.manifest)
        self.root = self.manifest.parent
        self.sample_rate = sample_rate
        self._entries: List[dict] = []
        with self.manifest.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "audio" not in obj or "text" not in obj:
                    raise ValueError(
                        f"{self.manifest}:{i}: missing 'audio' or 'text' key"
                    )
                self._entries.append(obj)
        if not self._entries:
            raise ValueError(f"{self.manifest} has no entries")

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Clip]:
        for e in self._entries:
            audio_path = (self.root / e["audio"]).resolve()
            wav = _load_audio(audio_path, self.sample_rate)
            yield Clip(audio=wav, text=e["text"], source=str(audio_path))

    def __getitem__(self, idx: int) -> Clip:
        e = self._entries[idx]
        audio_path = (self.root / e["audio"]).resolve()
        wav = _load_audio(audio_path, self.sample_rate)
        return Clip(audio=wav, text=e["text"], source=str(audio_path))
