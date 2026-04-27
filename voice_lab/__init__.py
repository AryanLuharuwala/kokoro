"""Voice embedding tools for Kokoro: blend, fit, save.

Submodules are imported lazily so that lightweight uses (just blending or
just computing losses) don't drag in the full ``kokoro`` + ``misaki`` stack.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "blend_voices",
    "save_voice_pack",
    "MultiResMelLoss",
    "MultiResSTFTLoss",
    "VoiceDataset",
    "StyleOptimizer",
    "Finetuner",
    "FinetuneConfig",
]

_LAZY = {
    "blend_voices": ("voice_lab.blend", "blend_voices"),
    "save_voice_pack": ("voice_lab.blend", "save_voice_pack"),
    "MultiResMelLoss": ("voice_lab.losses", "MultiResMelLoss"),
    "MultiResSTFTLoss": ("voice_lab.losses", "MultiResSTFTLoss"),
    "VoiceDataset": ("voice_lab.data", "VoiceDataset"),
    "StyleOptimizer": ("voice_lab.optimize", "StyleOptimizer"),
    "Finetuner": ("voice_lab.finetune", "Finetuner"),
    "FinetuneConfig": ("voice_lab.finetune", "FinetuneConfig"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module, attr = _LAZY[name]
        return getattr(import_module(module), attr)
    raise AttributeError(f"module 'voice_lab' has no attribute {name!r}")
