"""Blend Kokoro voices into a new voice pack.

A Kokoro voice pack is a tensor of shape ``[510, 256]``: one 256-d style
vector per phoneme length 1..510. Inference picks ``pack[len(ps) - 1]``.

This module provides utilities to:
  * load packs by name from the HF hub (or from a local .pt path)
  * blend several packs by weighted average
  * save the result as a .pt file Kokoro can load
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Union

import torch
from huggingface_hub import hf_hub_download


VOICE_PACK_SHAPE = (510, 256)
DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"


def load_pack(name_or_path: str, repo_id: str = DEFAULT_REPO_ID) -> torch.Tensor:
    """Load a voice pack by name (e.g. ``hm_psi``) or local .pt path."""
    if name_or_path.endswith(".pt") and os.path.exists(name_or_path):
        path = name_or_path
    else:
        path = hf_hub_download(repo_id=repo_id, filename=f"voices/{name_or_path}.pt")
    pack = torch.load(path, weights_only=True, map_location="cpu")
    if pack.shape != VOICE_PACK_SHAPE:
        raise ValueError(
            f"Expected voice pack shape {VOICE_PACK_SHAPE}, got {tuple(pack.shape)}"
        )
    return pack.float()


def blend_voices(
    weights: Mapping[str, float],
    repo_id: str = DEFAULT_REPO_ID,
    normalize: bool = True,
) -> torch.Tensor:
    """Weighted blend of named voices into a single ``[510, 256]`` pack.

    ``weights`` is a mapping like ``{"hm_psi": 1.0, "hf_alpha": 0.7}``. When
    ``normalize`` is true (default), weights are rescaled to sum to 1.
    """
    if not weights:
        raise ValueError("weights must be non-empty")
    items = list(weights.items())
    if normalize:
        total = sum(items[i][1] for i in range(len(items)))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        items = [(n, w / total) for n, w in items]
    blended = torch.zeros(VOICE_PACK_SHAPE)
    for name, w in items:
        blended.add_(load_pack(name, repo_id=repo_id), alpha=float(w))
    return blended


def save_voice_pack(pack: torch.Tensor, out_path: Union[str, Path]) -> Path:
    """Save a voice pack as a .pt file Kokoro can load.

    The file is written so that ``KPipeline.load_single_voice("/path/to.pt")``
    returns the same tensor (Kokoro keys voices by ``.endswith('.pt')``).
    """
    if pack.shape != VOICE_PACK_SHAPE:
        raise ValueError(
            f"Expected pack shape {VOICE_PACK_SHAPE}, got {tuple(pack.shape)}"
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack.detach().cpu().float(), out)
    return out


SHINCHAN_HINDI_PRESET: Dict[str, float] = {
    # Male Hindi voices supply the timbre baseline; female voices lift the
    # pitch and add the nasal/childlike quality. Tune from here.
    "hm_psi": 0.5,
    "hf_alpha": 0.4,
    "hf_beta": 0.1,
}
