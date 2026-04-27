"""Per-segment ASR using faster-whisper.

Transcripts are cached as JSON next to the audio so re-runs are cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import AsrConfig

_model_cache: dict = {}


def _load_model(cfg: AsrConfig):
    key = (cfg.model, cfg.device, cfg.compute_type)
    if key in _model_cache:
        return _model_cache[key]
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper not installed. Install with: pip install 'kokoro[dataset]'"
        ) from e
    compute = cfg.compute_type if cfg.device == "cuda" else "int8"
    model = WhisperModel(cfg.model, device=cfg.device, compute_type=compute)
    _model_cache[key] = model
    return model


@dataclass
class Transcript:
    audio_path: Path
    text: str
    language: str


def transcribe_one(audio_path: Path, cfg: AsrConfig) -> Optional[Transcript]:
    cache = audio_path.with_suffix(".json")
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            text = data.get("text", "").strip()
            if len(text) >= cfg.min_chars:
                return Transcript(audio_path, text, data.get("language", cfg.language))
            return None
        except Exception:
            pass  # fall through and re-transcribe

    model = _load_model(cfg)
    segments, info = model.transcribe(
        str(audio_path),
        language=cfg.language,
        beam_size=cfg.beam_size,
        vad_filter=False,  # already VAD-segmented upstream
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    cache.write_text(
        json.dumps({"text": text, "language": info.language}, ensure_ascii=False),
        encoding="utf-8",
    )
    if len(text) < cfg.min_chars:
        return None
    return Transcript(audio_path, text, info.language)


def transcribe_all(audio_paths: List[Path], cfg: AsrConfig) -> List[Transcript]:
    out: List[Transcript] = []
    for p in audio_paths:
        try:
            t = transcribe_one(p, cfg)
        except Exception as e:
            print(f"  ! ASR failed for {p}: {e}")
            continue
        if t is not None:
            out.append(t)
    return out
