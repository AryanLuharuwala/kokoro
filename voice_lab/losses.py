"""Audio losses for fitting a voice embedding to target Shinchan clips.

We use a multi-resolution mel loss: STFT magnitudes -> mel filterbank ->
log -> L1, summed across several FFT sizes. This is robust to small phase
mismatches (the model and target won't be aligned sample-for-sample) while
still capturing timbre.

Note on length mismatch: Kokoro predicts its own duration, so generated and
target clips will not have the same length even for identical text. Compare
the two over their shared minimum length; the loss focuses on local spectral
content rather than duration.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Slaney-normalised mel filterbank, shape ``[n_mels, n_fft // 2 + 1]``."""
    if f_max is None:
        f_max = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0, sample_rate / 2.0, n_freqs, dtype=dtype)
    m_min, m_max = _hz_to_mel(torch.tensor(f_min)), _hz_to_mel(torch.tensor(f_max))
    m_pts = torch.linspace(m_min.item(), m_max.item(), n_mels + 2, dtype=dtype)
    f_pts = _mel_to_hz(m_pts)
    fb = torch.zeros(n_mels, n_freqs, dtype=dtype)
    for m in range(n_mels):
        f_left, f_center, f_right = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        left = (all_freqs - f_left) / (f_center - f_left)
        right = (f_right - all_freqs) / (f_right - f_center)
        fb[m] = torch.clamp(torch.minimum(left, right), min=0.0)
        # Slaney norm: equal area per filter.
        enorm = 2.0 / (f_right - f_left)
        fb[m] *= enorm
    return fb


class MelSpec(nn.Module):
    """Differentiable log-mel spectrogram (no torchaudio dep)."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int | None = None,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = None,
        log_eps: float = 1e-5,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.log_eps = log_eps
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer(
            "fb",
            mel_filterbank(sample_rate, n_fft, n_mels, f_min=f_min, f_max=f_max),
            persistent=False,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        mag = spec.abs()
        mel = self.fb @ mag  # [B, n_mels, T]
        return torch.log(mel.clamp_min(self.log_eps))


class MultiResMelLoss(nn.Module):
    """L1 loss on log-mel spectrograms at several FFT resolutions.

    Compares only the overlapping prefix of the two waveforms, so generated
    audio whose length differs from the target is still useful.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_ffts: Sequence[int] = (512, 1024, 2048),
        n_mels: int = 80,
    ):
        super().__init__()
        self.mels = nn.ModuleList(
            [MelSpec(sample_rate=sample_rate, n_fft=n, n_mels=n_mels) for n in n_ffts]
        )

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Crop to the shorter length so we don't penalise mere duration drift.
        n = min(generated.shape[-1], target.shape[-1])
        g = generated[..., :n]
        t = target[..., :n]
        loss = generated.new_tensor(0.0)
        for mel in self.mels:
            loss = loss + F.l1_loss(mel(g), mel(t))
        return loss / len(self.mels)
