"""TOML config for the Shinchan dataset pipeline.

Schema (see ``voice_lab/examples/shinchan_hindi.toml`` for a full file)::

    output_dir = "data/shinchan"
    language   = "hi"

    [[sources]]
    url = "https://www.youtube.com/watch?v=..."
    # optional per-source trimming:
    # start_offset = 30  # seconds, skip intro
    # end_offset   = 30  # seconds, skip outro

    [segmentation]
    method        = "pitch"   # currently only "pitch"
    min_f0        = 220       # Hz, Shinchan's pitch range
    max_f0        = 450
    min_duration  = 1.5
    max_duration  = 10.0
    vad_threshold = 0.5

    [asr]
    model    = "small"
    language = "hi"
    device   = "cuda"

    [fit]
    enabled    = true
    init_voice = "hm_psi"
    epochs     = 50
    lr         = 1e-2
    out        = "voices/shinchan.pt"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass
class Source:
    url: str
    start_offset: float = 0.0
    end_offset: float = 0.0
    title: Optional[str] = None  # cosmetic only


@dataclass
class SegmentationConfig:
    method: str = "pitch"
    min_f0: float = 220.0
    max_f0: float = 450.0
    min_duration: float = 1.5
    max_duration: float = 10.0
    vad_threshold: float = 0.5
    # Fraction of voiced frames whose F0 must fall in [min_f0, max_f0]
    # for a segment to be kept.
    min_in_range_ratio: float = 0.6


@dataclass
class AsrConfig:
    model: str = "small"
    language: str = "hi"
    device: str = "cuda"
    compute_type: str = "float16"  # ignored on CPU
    beam_size: int = 5
    min_chars: int = 3  # drop transcripts shorter than this


@dataclass
class FitConfig:
    enabled: bool = True
    init_voice: Optional[str] = "hm_psi"
    init_blend: Optional[dict] = None
    parameterization: str = "shared"
    epochs: int = 50
    lr: float = 1e-2
    out: str = "voices/shinchan.pt"


@dataclass
class FinetuneConfig:
    enabled: bool = False
    preset: str = "voice"
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    warmup_steps: int = 100
    mel_weight: float = 1.0
    stft_weight: float = 0.5
    train_style: bool = True
    style_lr_scale: float = 10.0
    parameterization: str = "shared"
    init_voice: Optional[str] = "hm_psi"
    init_blend: Optional[dict] = None
    val_every_epochs: int = 1
    save_every_epochs: int = 5
    val_sentence: Optional[str] = None
    out_dir: str = "checkpoints/shinchan"
    final_pth: str = "checkpoints/shinchan/kokoro-shinchan.pth"
    final_voice: str = "voices/shinchan.pt"


@dataclass
class PipelineConfig:
    output_dir: Path
    language: str
    sources: List[Source]
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)

    @property
    def raw_dir(self) -> Path:
        return self.output_dir / "raw"

    @property
    def segments_dir(self) -> Path:
        return self.output_dir / "segments"

    @property
    def transcripts_dir(self) -> Path:
        return self.output_dir / "transcripts"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.jsonl"


def _coerce(cls, data: dict):
    """Drop keys not in the dataclass and instantiate."""
    if not data:
        return cls()
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    extra = set(data) - valid
    if extra:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(extra)}")
    return cls(**data)


def load_config(path: str | Path) -> PipelineConfig:
    p = Path(path)
    with p.open("rb") as f:
        raw = tomllib.load(f)
    if "output_dir" not in raw:
        raise ValueError(f"{p}: missing 'output_dir'")
    if "sources" not in raw or not raw["sources"]:
        raise ValueError(f"{p}: missing 'sources' (list of {{url=...}} entries)")
    sources = [_coerce(Source, s) for s in raw["sources"]]
    segmentation = _coerce(SegmentationConfig, raw.get("segmentation", {}))
    asr = _coerce(AsrConfig, raw.get("asr", {}))
    fit = _coerce(FitConfig, raw.get("fit", {}))
    finetune = _coerce(FinetuneConfig, raw.get("finetune", {}))
    return PipelineConfig(
        output_dir=Path(raw["output_dir"]),
        language=raw.get("language", "hi"),
        sources=sources,
        segmentation=segmentation,
        asr=asr,
        fit=fit,
        finetune=finetune,
    )
