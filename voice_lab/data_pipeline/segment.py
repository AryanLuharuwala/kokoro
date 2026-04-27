"""Segment downloaded audio into Shinchan-only speech clips.

Two-stage filter:

1. Voice Activity Detection (Silero VAD) splits the audio into speech
   regions, dropping silence and music.
2. Pitch filter: Shinchan's voice sits roughly in [220, 450] Hz. Adult
   speakers sit well below (males ~80-180 Hz, females ~165-255 Hz). For
   each VAD region we estimate F0 with librosa.pyin, and keep regions
   where most voiced frames land in the Shinchan range.

This is heuristic, not perfect. It will keep some adult-female clips with
high pitch and reject some quieter Shinchan moments. For the target task
(fitting one voice to a few minutes of clean clips) the bias toward
high-pitch clips is exactly what we want.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import numpy as np

from .config import SegmentationConfig
from .download import DownloadedClip, SAMPLE_RATE


@dataclass
class Segment:
    video_id: str
    index: int
    start: float  # seconds within source
    end: float
    audio: np.ndarray  # float32 mono in [-1, 1] at SAMPLE_RATE
    median_f0: float


def _load_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import resampy

        wav = resampy.resample(wav, sr, SAMPLE_RATE)
    return wav


_silero_cache = {}


def _silero_model():
    if "model" in _silero_cache:
        return _silero_cache["model"], _silero_cache["utils"]
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("voice_lab dataset stage needs torch") from e
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        onnx=False,
    )
    _silero_cache["model"] = model
    _silero_cache["utils"] = utils
    return model, utils


def _vad_regions(
    wav: np.ndarray,
    threshold: float,
    min_duration: float,
    max_duration: float,
) -> List[tuple[float, float]]:
    """Return a list of (start_s, end_s) speech regions using Silero VAD."""
    import torch

    model, utils = _silero_model()
    get_speech_timestamps = utils[0]
    # Silero wants 16 kHz internally; we feed float32 tensor and let it
    # do the resampling. We pass our 24 kHz audio with the matching sr.
    audio_t = torch.from_numpy(wav).float()
    raw = get_speech_timestamps(
        audio_t,
        model,
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=int(min_duration * 1000),
        max_speech_duration_s=max_duration,
    )
    return [(r["start"] / SAMPLE_RATE, r["end"] / SAMPLE_RATE) for r in raw]


def _pitch_stats(
    chunk: np.ndarray,
    fmin: float,
    fmax: float,
) -> tuple[float, float]:
    """Return (median voiced F0, fraction of voiced frames in [fmin, fmax])."""
    import librosa

    # pyin works well on speech. Use a generous detection range so frames
    # outside [fmin, fmax] are still measured (they just won't count toward
    # the in-range ratio).
    f0, voiced_flag, _ = librosa.pyin(
        chunk,
        fmin=80.0,
        fmax=600.0,
        sr=SAMPLE_RATE,
        frame_length=2048,
    )
    voiced = f0[voiced_flag.astype(bool)]
    voiced = voiced[~np.isnan(voiced)]
    if voiced.size == 0:
        return 0.0, 0.0
    in_range = (voiced >= fmin) & (voiced <= fmax)
    return float(np.median(voiced)), float(in_range.mean())


def segment_clip(
    clip: DownloadedClip, cfg: SegmentationConfig
) -> Iterator[Segment]:
    wav = _load_audio(clip.audio_path)

    # Trim per-source offsets.
    start_off = int(clip.source.start_offset * SAMPLE_RATE)
    end_off = int(clip.source.end_offset * SAMPLE_RATE) if clip.source.end_offset else 0
    if end_off:
        wav = wav[start_off:-end_off] if end_off else wav[start_off:]
    elif start_off:
        wav = wav[start_off:]

    regions = _vad_regions(
        wav, cfg.vad_threshold, cfg.min_duration, cfg.max_duration
    )

    kept = 0
    for i, (s, e) in enumerate(regions):
        dur = e - s
        if dur < cfg.min_duration or dur > cfg.max_duration:
            continue
        chunk = wav[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)]
        if chunk.size == 0:
            continue
        if cfg.method == "pitch":
            median_f0, in_range = _pitch_stats(chunk, cfg.min_f0, cfg.max_f0)
            if (
                in_range < cfg.min_in_range_ratio
                or not (cfg.min_f0 <= median_f0 <= cfg.max_f0)
            ):
                continue
        elif cfg.method == "all":
            median_f0 = 0.0
        else:
            raise ValueError(f"unknown segmentation method: {cfg.method}")
        yield Segment(
            video_id=clip.video_id,
            index=kept,
            start=s + clip.source.start_offset,
            end=e + clip.source.start_offset,
            audio=chunk,
            median_f0=median_f0,
        )
        kept += 1


def write_segments(segments: Iterator[Segment], out_dir: Path) -> List[Path]:
    """Write each segment as a .wav and return the paths."""
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for seg in segments:
        sub = out_dir / seg.video_id
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{seg.index:05d}_{int(seg.start*100):07d}.wav"
        sf.write(str(path), seg.audio, SAMPLE_RATE)
        paths.append(path)
    return paths
