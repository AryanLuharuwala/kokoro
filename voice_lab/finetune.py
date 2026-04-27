"""Fine-tune Kokoro-82M model weights on a single voice.

This is a *minimal* fine-tuner that uses only what the Kokoro repo ships:
the model itself, our (audio, transcript) dataset, and reconstruction
losses. It does **not** rely on the auxiliary models from the original
StyleTTS2 training stack (HiFi-GAN discriminator, WavLM SLM, JDC pitch
extractor, separate ASR aligner) -- those weights aren't open-sourced for
Kokoro. See the README for higher-quality alternatives that do.

What works:
  * Mel reconstruction + multi-resolution STFT magnitude loss on raw audio.
  * Selective unfreezing via presets (``style_only`` / ``voice`` / ``full``).
  * Periodic validation: synthesize a fixed sentence to a WAV so you can
    listen to drift between checkpoints.
  * Saves checkpoints in Kokoro's native state-dict format (``{'bert': ...,
    'predictor': ..., ...}``) so ``KModel(config=..., model=ckpt.pth)``
    loads them with no glue code.

Known limitations:
  * Duration is constructed via ``torch.round`` + ``.long()`` in the
    forward, which blocks gradient flow into the duration LSTM and proj.
    Even in the ``full`` preset those parameters get zero gradient. This
    means cadence/timing won't change much from the pre-trained model.
  * Without a discriminator the decoder may drift toward muffled audio if
    you over-train. Keep epochs low (~10-30) and watch the validation
    samples.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from kokoro import KModel, KPipeline

from .blend import VOICE_PACK_SHAPE, blend_voices, load_pack, save_voice_pack
from .data import VoiceDataset, Clip
from .losses import MultiResMelLoss, MultiResSTFTLoss
from .optimize import _StyleParam, _forward_with_grad


# ----- Freeze presets ---------------------------------------------------

# Each preset maps parameter-name prefixes to requires_grad. Longest prefix
# wins, so e.g. ``predictor.duration_proj: False`` overrides ``predictor: True``.
PRESETS: Dict[str, Dict[str, bool]] = {
    # All weights frozen; only the style vector trains. Equivalent to
    # voice_lab.optimize.StyleOptimizer; included so a single ``finetune``
    # command can run the same recipe.
    "style_only": {
        "bert": False,
        "bert_encoder": False,
        "predictor": False,
        "text_encoder": False,
        "decoder": False,
    },
    # Recommended for new-voice fine-tuning. Keeps BERT (linguistic) and
    # the duration head frozen; trains the parts that shape timbre and
    # acoustic detail.
    "voice": {
        "bert": False,
        "bert_encoder": True,
        "predictor": True,
        "predictor.lstm": False,
        "predictor.duration_proj": False,
        "text_encoder": True,
        "decoder": True,
    },
    # Everything trainable. Higher capacity but easier to forget the
    # pre-training. Note duration params still don't receive gradient
    # because torch.round/.long block it in the forward.
    "full": {
        "bert": True,
        "bert_encoder": True,
        "predictor": True,
        "text_encoder": True,
        "decoder": True,
    },
}


def apply_preset(model: KModel, preset: str) -> Dict[str, int]:
    """Set ``requires_grad`` on each parameter according to ``preset``.

    Returns a small summary dict ``{trainable_params, frozen_params}``.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"unknown preset {preset!r}; choose from {sorted(PRESETS)}"
        )
    spec = PRESETS[preset]
    # Sort by descending prefix length so more specific keys override.
    keys = sorted(spec.keys(), key=lambda k: -len(k))
    n_train, n_frozen = 0, 0
    for name, p in model.named_parameters():
        flag = False
        for prefix in keys:
            if name == prefix or name.startswith(prefix + "."):
                flag = spec[prefix]
                break
        p.requires_grad_(flag)
        if flag:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    return {"trainable_params": n_train, "frozen_params": n_frozen}


# ----- Config -----------------------------------------------------------


@dataclass
class FinetuneConfig:
    preset: str = "voice"
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # Loss weights. Mel is dimensionless, STFT is too; keep mel dominant.
    mel_weight: float = 1.0
    stft_weight: float = 0.5

    # Style vector co-training (initialised from init_voice / init_blend).
    train_style: bool = True
    style_lr_scale: float = 10.0  # style param trains with lr * this
    parameterization: str = "shared"
    init_voice: Optional[str] = "hm_psi"
    init_blend: Optional[dict] = None

    # Validation / checkpointing.
    val_every_epochs: int = 1
    save_every_epochs: int = 5
    val_sentence: Optional[str] = None  # synthesise + save WAV after each val epoch
    out_dir: str = "checkpoints/shinchan"
    final_pth: str = "checkpoints/shinchan/kokoro-shinchan.pth"
    final_voice: str = "voices/shinchan.pt"


# ----- Trainer ----------------------------------------------------------


class Finetuner:
    def __init__(
        self,
        pipeline: KPipeline,
        dataset: VoiceDataset,
        cfg: FinetuneConfig = FinetuneConfig(),
        loss_mel: Optional[nn.Module] = None,
        loss_stft: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ):
        if pipeline.model is None:
            raise ValueError("pipeline.model is required")
        self.pipeline = pipeline
        self.model: KModel = pipeline.model
        self.dataset = dataset
        self.cfg = cfg
        self.device = device or str(self.model.device)

        # Even at train time we want the decoder's hardcoded-eval dropouts
        # to stay off (consistent with how the model was published) but the
        # TextEncoder's nn.Dropout etc. to follow .train(). Using .train()
        # is the right call.
        self.model.train()
        summary = apply_preset(self.model, cfg.preset)
        print(
            f"[preset={cfg.preset}] trainable={summary['trainable_params']:,}"
            f" frozen={summary['frozen_params']:,}"
        )

        self.loss_mel = (loss_mel or MultiResMelLoss(sample_rate=dataset.sample_rate)).to(
            self.device
        )
        self.loss_stft = (loss_stft or MultiResSTFTLoss()).to(self.device)

        # Style param (always allocated; only trained if cfg.train_style).
        init_pack = self._init_pack()
        self.style = _StyleParam(cfg.parameterization, init_pack=init_pack).to(
            self.device
        )
        if not cfg.train_style:
            self.style.style.requires_grad_(False)

        # Optimizer: model params at cfg.lr, style param at cfg.lr * scale.
        param_groups = []
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        if model_params:
            param_groups.append({"params": model_params, "lr": cfg.lr})
        if cfg.train_style:
            param_groups.append(
                {"params": [self.style.style], "lr": cfg.lr * cfg.style_lr_scale}
            )
        if not param_groups:
            raise ValueError(
                "preset froze all model params and train_style=False; nothing to train"
            )
        self.optim = torch.optim.AdamW(
            param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.99)
        )

        self._step = 0
        self._total_steps = max(1, cfg.epochs * len(dataset))

    # ----- helpers --------------------------------------------------------

    def _init_pack(self) -> Optional[torch.Tensor]:
        if self.cfg.init_blend:
            return blend_voices(self.cfg.init_blend, repo_id=self.model.repo_id)
        if self.cfg.init_voice:
            return load_pack(self.cfg.init_voice, repo_id=self.model.repo_id)
        return None

    def _phonemize(self, text: str) -> str:
        out = self.pipeline.g2p(text)
        return out[0] if isinstance(out, tuple) else out

    def _ids_for(self, ps: str) -> torch.LongTensor:
        ids = [self.model.vocab[p] for p in ps if self.model.vocab.get(p) is not None]
        if not ids:
            raise ValueError("phonemes produced no in-vocab tokens")
        ids = [0, *ids, 0]
        return torch.LongTensor([ids]).to(self.device)

    def _lr_scale(self) -> float:
        """Linear warmup, then cosine decay to zero."""
        if self._step < self.cfg.warmup_steps:
            return (self._step + 1) / max(1, self.cfg.warmup_steps)
        progress = (self._step - self.cfg.warmup_steps) / max(
            1, self._total_steps - self.cfg.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _apply_lr(self, base_lrs: List[float]) -> None:
        scale = self._lr_scale()
        for g, base in zip(self.optim.param_groups, base_lrs):
            g["lr"] = base * scale

    # ----- training -------------------------------------------------------

    def step(self, clip: Clip) -> Dict[str, float]:
        ps = self._phonemize(clip.text)
        input_ids = self._ids_for(ps)
        ref_s = self.style.select(len(ps)).to(self.device)
        target = clip.audio.to(self.device)

        audio = _forward_with_grad(self.model, input_ids, ref_s)
        gen = audio.unsqueeze(0)
        tgt = target.unsqueeze(0)
        loss_mel = self.loss_mel(gen, tgt)
        loss_stft = self.loss_stft(gen, tgt)
        loss = self.cfg.mel_weight * loss_mel + self.cfg.stft_weight * loss_stft

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optim.param_groups for p in g["params"]],
                self.cfg.grad_clip,
            )
        self.optim.step()
        self._step += 1
        return {
            "loss": float(loss.detach().cpu()),
            "loss_mel": float(loss_mel.detach().cpu()),
            "loss_stft": float(loss_stft.detach().cpu()),
        }

    def fit(self, on_epoch: Optional[Callable[[int, Dict[str, float]], None]] = None) -> List[Dict[str, float]]:
        out_dir = Path(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base_lrs = [g["lr"] for g in self.optim.param_groups]
        history: List[Dict[str, float]] = []

        for epoch in range(1, self.cfg.epochs + 1):
            sums: Dict[str, float] = {}
            count = 0
            for clip in self.dataset:
                self._apply_lr(base_lrs)
                metrics = self.step(clip)
                for k, v in metrics.items():
                    sums[k] = sums.get(k, 0.0) + v
                count += 1
            avg = {k: v / count for k, v in sums.items()}
            history.append(avg)
            print(
                f"[epoch {epoch:3d}/{self.cfg.epochs}] "
                f"loss={avg['loss']:.4f} mel={avg['loss_mel']:.4f}"
                f" stft={avg['loss_stft']:.4f} lr_scale={self._lr_scale():.3f}"
            )

            if (
                self.cfg.val_sentence
                and self.cfg.val_every_epochs
                and epoch % self.cfg.val_every_epochs == 0
            ):
                wav_path = out_dir / f"sample_epoch{epoch:03d}.wav"
                self.synth_sample(self.cfg.val_sentence, wav_path)
            if self.cfg.save_every_epochs and epoch % self.cfg.save_every_epochs == 0:
                self.save_checkpoint(out_dir / f"ckpt_epoch{epoch:03d}.pth")
            if on_epoch is not None:
                on_epoch(epoch, avg)
        return history

    # ----- evaluation -----------------------------------------------------

    @torch.no_grad()
    def synth_sample(self, text: str, wav_path: Path) -> Path:
        was_training = self.model.training
        self.model.eval()
        try:
            ps = self._phonemize(text)
            input_ids = self._ids_for(ps)
            ref_s = self.style.select(len(ps)).to(self.device)
            audio = _forward_with_grad(self.model, input_ids, ref_s)
            import soundfile as sf

            sf.write(str(wav_path), audio.detach().cpu().float().numpy(), self.dataset.sample_rate)
            print(f"  ↳ sample @ {wav_path}")
        finally:
            if was_training:
                self.model.train()
        return wav_path

    # ----- checkpointing --------------------------------------------------

    def save_checkpoint(self, path: Path) -> Path:
        """Write a Kokoro-format ``.pth`` (dict of component state_dicts).

        Loadable via ``KModel(config=..., model=str(path))``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "bert": self.model.bert.state_dict(),
            "bert_encoder": self.model.bert_encoder.state_dict(),
            "predictor": self.model.predictor.state_dict(),
            "text_encoder": self.model.text_encoder.state_dict(),
            "decoder": self.model.decoder.state_dict(),
        }
        torch.save(state, path)
        print(f"  ↳ checkpoint @ {path}")
        return path

    def export_final(self) -> Dict[str, Path]:
        out_pth = Path(self.cfg.final_pth)
        self.save_checkpoint(out_pth)
        # Copy the model's config.json next to the .pth so a user can do
        # KModel(config=<config>, model=<pth>) without surprises.
        from huggingface_hub import hf_hub_download

        cfg_src = Path(hf_hub_download(repo_id=self.model.repo_id, filename="config.json"))
        cfg_dst = out_pth.with_name("config.json")
        shutil.copy(cfg_src, cfg_dst)

        out_voice = Path(self.cfg.final_voice)
        save_voice_pack(self.style.pack().detach().cpu(), out_voice)
        return {"pth": out_pth, "config": cfg_dst, "voice": out_voice}
