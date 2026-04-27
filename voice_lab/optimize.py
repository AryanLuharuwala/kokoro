"""Fit a Kokoro voice embedding via gradient descent.

The Kokoro model is frozen. Only the 256-d style vector(s) are learnable.
For each (text, target_audio) pair we synthesise audio with the current
style, compare to the target via a multi-resolution mel loss, and step.

Two parameterisations are supported:

* ``shared`` (default): one 256-d vector, broadcast to all 510 length
  buckets. Tiny (1 KB), fast to fit, and it's how a single-speaker voice
  should naturally behave.
* ``per-length``: a full ``[510, 256]`` tensor, matching the on-disk pack
  shape exactly. Higher capacity but harder to fit with little data.

Most of the model's forward pass is reproduced here (instead of calling
``KModel.forward``) because the model decorates its forward with
``@torch.no_grad()`` for inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from kokoro import KModel, KPipeline

from .blend import VOICE_PACK_SHAPE, blend_voices, load_pack, save_voice_pack
from .data import VoiceDataset, Clip
from .losses import MultiResMelLoss


@dataclass
class OptimizerConfig:
    lr: float = 1e-2
    epochs: int = 50
    parameterization: str = "shared"  # "shared" or "per-length"
    grad_clip: float = 1.0
    log_every: int = 1
    init_voice: Optional[str] = None  # name of an existing voice to seed from
    init_blend: Optional[dict] = None  # alternative: weights dict for blend


class _StyleParam(nn.Module):
    """Learnable style tensor with the same shape conventions as a voice pack."""

    def __init__(
        self,
        parameterization: str,
        init_pack: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if parameterization not in ("shared", "per-length"):
            raise ValueError(parameterization)
        self.parameterization = parameterization
        if init_pack is None:
            init_pack = torch.zeros(VOICE_PACK_SHAPE)
        if init_pack.shape != VOICE_PACK_SHAPE:
            raise ValueError(init_pack.shape)
        if parameterization == "shared":
            # Average across length buckets to get a single 256-d seed.
            self.style = nn.Parameter(init_pack.mean(dim=0).clone())
        else:
            self.style = nn.Parameter(init_pack.clone())

    def pack(self) -> torch.Tensor:
        if self.parameterization == "shared":
            return self.style.unsqueeze(0).expand(*VOICE_PACK_SHAPE).contiguous()
        return self.style

    def select(self, ps_len: int) -> torch.Tensor:
        """Style row to use for an utterance whose phoneme length is ``ps_len``.

        Returns a ``[1, 256]`` tensor (matches inference indexing
        ``pack[len(ps)-1]``).
        """
        if self.parameterization == "shared":
            return self.style.unsqueeze(0)
        idx = max(0, min(ps_len - 1, VOICE_PACK_SHAPE[0] - 1))
        return self.style[idx].unsqueeze(0)


def _forward_with_grad(
    model: KModel,
    input_ids: torch.LongTensor,
    ref_s: torch.FloatTensor,
    speed: float = 1.0,
) -> torch.FloatTensor:
    """Reproduces ``KModel.forward_with_tokens`` without ``@torch.no_grad``.

    Mirrors kokoro/model.py:87-119. Kept in this module so we don't have to
    edit the upstream model file.
    """
    device = model.device
    input_lengths = torch.full(
        (input_ids.shape[0],),
        input_ids.shape[-1],
        device=input_ids.device,
        dtype=torch.long,
    )
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
        input_lengths.shape[0], -1
    ).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)
    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(dim=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
    indices = torch.repeat_interleave(
        torch.arange(input_ids.shape[1], device=device), pred_dur
    )
    pred_aln_trg = torch.zeros(
        (input_ids.shape[1], indices.shape[0]), device=device
    )
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0).to(device)
    en = d.transpose(-1, -2) @ pred_aln_trg
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg
    audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
    return audio


class StyleOptimizer:
    """Coordinates dataset + model + loss + optimizer to fit a voice pack."""

    def __init__(
        self,
        pipeline: KPipeline,
        dataset: VoiceDataset,
        config: OptimizerConfig = OptimizerConfig(),
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ):
        if pipeline.model is None:
            raise ValueError("pipeline.model is required for training")
        self.pipeline = pipeline
        self.model: KModel = pipeline.model
        self.dataset = dataset
        self.config = config
        self.device = device or str(self.model.device)
        self.loss_fn = (loss_fn or MultiResMelLoss(sample_rate=dataset.sample_rate)).to(
            self.device
        )
        # Freeze the entire model.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        init_pack = self._init_pack()
        self.style = _StyleParam(config.parameterization, init_pack=init_pack).to(
            self.device
        )
        self.optim = torch.optim.AdamW([self.style.style], lr=config.lr)

    def _init_pack(self) -> Optional[torch.Tensor]:
        if self.config.init_blend:
            return blend_voices(self.config.init_blend, repo_id=self.model.repo_id)
        if self.config.init_voice:
            return load_pack(self.config.init_voice, repo_id=self.model.repo_id)
        return None

    def _phonemize(self, text: str) -> str:
        # KPipeline.g2p returns (phonemes, tokens) for English / Japanese /
        # Mandarin and (phonemes, None) for espeak languages.
        out = self.pipeline.g2p(text)
        ps = out[0] if isinstance(out, tuple) else out
        return ps

    def _ids_for(self, ps: str) -> torch.LongTensor:
        ids = [
            self.model.vocab[p] for p in ps if self.model.vocab.get(p) is not None
        ]
        if not ids:
            raise ValueError("phonemes produced no in-vocab tokens")
        ids = [0, *ids, 0]
        return torch.LongTensor([ids]).to(self.device)

    def step(self, clip: Clip) -> float:
        ps = self._phonemize(clip.text)
        # ps_len matches inference: pack indexed by len(ps) - 1.
        input_ids = self._ids_for(ps)
        ref_s = self.style.select(len(ps)).to(self.device)
        target = clip.audio.to(self.device)

        audio = _forward_with_grad(self.model, input_ids, ref_s)
        loss = self.loss_fn(audio.unsqueeze(0), target.unsqueeze(0))

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [self.style.style], self.config.grad_clip
            )
        self.optim.step()
        return float(loss.detach().cpu())

    def fit(self, on_epoch: Optional[Callable[[int, float], None]] = None) -> List[float]:
        history: List[float] = []
        for epoch in range(1, self.config.epochs + 1):
            losses: List[float] = []
            for clip in self.dataset:
                losses.append(self.step(clip))
            mean_loss = sum(losses) / len(losses)
            history.append(mean_loss)
            if on_epoch is not None:
                on_epoch(epoch, mean_loss)
            elif epoch % self.config.log_every == 0:
                print(f"[epoch {epoch:3d}/{self.config.epochs}] loss={mean_loss:.4f}")
        return history

    def export(self, out_path: str | Path) -> Path:
        return save_voice_pack(self.style.pack().detach().cpu(), out_path)
