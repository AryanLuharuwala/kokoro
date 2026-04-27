"""End-to-end pipeline: YouTube URLs -> manifest -> (optionally) fit.

Stages and where they cache:

  download   -> output_dir/raw/<video_id>.wav
  segment    -> output_dir/segments/<video_id>/NNNNN_TTTTTTT.wav
  transcribe -> output_dir/segments/<video_id>/NNNNN_TTTTTTT.json
  manifest   -> output_dir/manifest.jsonl
  fit (opt)  -> fit.out (.pt voice pack)

Re-running with the same config is incremental: anything already on disk
is reused, anything missing is produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .config import PipelineConfig
from .download import DownloadedClip, download_all
from .segment import segment_clip, write_segments
from .transcribe import Transcript, transcribe_all


def _segments_already_done(seg_dir: Path, video_id: str) -> List[Path]:
    sub = seg_dir / video_id
    if not sub.exists():
        return []
    return sorted(sub.glob("*.wav"))


def _write_manifest(transcripts: List[Transcript], cfg: PipelineConfig) -> Path:
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    root = cfg.manifest_path.parent
    with cfg.manifest_path.open("w", encoding="utf-8") as f:
        for t in transcripts:
            try:
                rel = t.audio_path.relative_to(root)
            except ValueError:
                rel = t.audio_path
            f.write(
                json.dumps({"audio": str(rel), "text": t.text}, ensure_ascii=False)
                + "\n"
            )
    return cfg.manifest_path


def build_dataset(cfg: PipelineConfig) -> Path:
    """Run download -> segment -> transcribe -> manifest. Return manifest path."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] downloading {len(cfg.sources)} source(s) -> {cfg.raw_dir}")
    clips = download_all(cfg.sources, cfg.raw_dir)
    if not clips:
        raise RuntimeError("no audio downloaded; aborting")

    print(f"[2/4] segmenting (method={cfg.segmentation.method})")
    all_segment_paths: List[Path] = []
    for clip in clips:
        cached = _segments_already_done(cfg.segments_dir, clip.video_id)
        if cached:
            print(f"  - {clip.video_id}: {len(cached)} cached segments")
            all_segment_paths.extend(cached)
            continue
        segs = list(segment_clip(clip, cfg.segmentation))
        paths = write_segments(iter(segs), cfg.segments_dir)
        print(f"  - {clip.video_id}: kept {len(paths)} Shinchan-pitch segments")
        all_segment_paths.extend(paths)

    if not all_segment_paths:
        raise RuntimeError(
            "no segments survived filtering; relax segmentation.* in the config"
        )

    print(f"[3/4] transcribing {len(all_segment_paths)} segments via Whisper")
    transcripts = transcribe_all(all_segment_paths, cfg.asr)
    print(f"  - kept {len(transcripts)}/{len(all_segment_paths)} after ASR filtering")

    if not transcripts:
        raise RuntimeError("no transcripts produced; check ASR config / language")

    print(f"[4/4] writing manifest -> {cfg.manifest_path}")
    return _write_manifest(transcripts, cfg)


def fit_voice(cfg: PipelineConfig) -> Path:
    """Run the StyleOptimizer using the manifest produced by build_dataset."""
    if not cfg.manifest_path.exists():
        raise RuntimeError(f"manifest not found: {cfg.manifest_path}")

    # Imported here so users who only want to build a dataset don't need
    # the full kokoro stack at import time.
    from kokoro import KPipeline

    from ..data import VoiceDataset
    from ..optimize import OptimizerConfig, StyleOptimizer

    pipeline = KPipeline(lang_code=cfg.language, device=cfg.asr.device)
    dataset = VoiceDataset(cfg.manifest_path)
    opt = StyleOptimizer(
        pipeline,
        dataset,
        OptimizerConfig(
            lr=cfg.fit.lr,
            epochs=cfg.fit.epochs,
            parameterization=cfg.fit.parameterization,
            init_voice=cfg.fit.init_voice,
            init_blend=cfg.fit.init_blend,
        ),
    )
    opt.fit()
    return opt.export(cfg.fit.out)


def run(cfg: PipelineConfig) -> dict:
    """Build the dataset and (if enabled) fit the voice."""
    manifest = build_dataset(cfg)
    out = {"manifest": manifest}
    if cfg.fit.enabled:
        print(f"[fit] training voice -> {cfg.fit.out}")
        out["voice_pack"] = fit_voice(cfg)
    else:
        print("[fit] disabled in config; stopping after manifest")
    return out
