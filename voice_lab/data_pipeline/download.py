"""Download YouTube audio as 24 kHz mono WAV via yt-dlp.

We standardise on WAV (not MP3) here because the rest of the pipeline
(VAD, ASR, training) wants raw PCM at 24 kHz mono anyway. yt-dlp's
post-processing step does the format + sample-rate + channel conversion in
one ffmpeg invocation.

A simple cache: we name files by the YouTube video ID, so re-running the
pipeline doesn't re-download.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Source

SAMPLE_RATE = 24000


@dataclass
class DownloadedClip:
    source: Source
    video_id: str
    audio_path: Path


_VIDEO_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|youtube\.com/(?:embed|shorts)/)([A-Za-z0-9_-]{11})"
)


def _video_id(url: str) -> str:
    m = _VIDEO_ID_RE.search(url)
    if not m:
        # Fall back to a slug of the URL so unusual links still cache.
        return re.sub(r"[^A-Za-z0-9_-]", "_", url)[-32:]
    return m.group(1)


def _ensure_yt_dlp() -> str:
    from shutil import which

    p = which("yt-dlp")
    if not p:
        raise RuntimeError(
            "yt-dlp not found on PATH. Install with: pip install 'kokoro[dataset]'"
        )
    return p


def _ensure_ffmpeg() -> None:
    from shutil import which

    if not which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install with your package manager "
            "(apt-get install ffmpeg / brew install ffmpeg)."
        )


def download(source: Source, raw_dir: Path) -> DownloadedClip:
    """Download ``source`` to ``raw_dir/<video_id>.wav`` (cached)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    vid = _video_id(source.url)
    out = raw_dir / f"{vid}.wav"
    if out.exists() and out.stat().st_size > 0:
        return DownloadedClip(source=source, video_id=vid, audio_path=out)

    _ensure_yt_dlp()
    _ensure_ffmpeg()

    # We let yt-dlp drive ffmpeg via --postprocessor-args. The audio is
    # extracted, downsampled to 24 kHz mono PCM16, and written to .wav.
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--postprocessor-args",
        f"ffmpeg:-ac 1 -ar {SAMPLE_RATE}",
        "-o",
        str(raw_dir / f"{vid}.%(ext)s"),
        source.url,
    ]
    subprocess.run(cmd, check=True)
    if not out.exists():
        raise RuntimeError(f"yt-dlp did not produce {out}")
    return DownloadedClip(source=source, video_id=vid, audio_path=out)


def download_all(sources: list[Source], raw_dir: Path) -> list[DownloadedClip]:
    out: list[DownloadedClip] = []
    for s in sources:
        try:
            out.append(download(s, raw_dir))
        except subprocess.CalledProcessError as e:
            print(f"  ! download failed for {s.url}: {e}")
        except RuntimeError as e:
            print(f"  ! {e}")
    return out
