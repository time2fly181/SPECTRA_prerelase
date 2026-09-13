"""Small waveform perturbations used only for inference-time averaging."""

from __future__ import annotations

import torch


def augment_batch(
    wave: torch.Tensor,
    *,
    noise_std: float = 0.01,
    amp_scale: float = 0.05,
    shift_sec: float = 0.0,
) -> torch.Tensor:
    """Apply the requested gain, noise, and circular shift to an inference copy.

    Zero-filled unavailable channel epochs remain zero. The caller owns the
    copy; the return value is also modified in place for runtime compatibility.
    """
    if wave.ndim not in (3, 4):
        raise ValueError("TTA waveforms must have shape [B,C,T] or [B,L,C,T]")
    if min(noise_std, amp_scale, shift_sec) < 0:
        raise ValueError("TTA strengths must be nonnegative")
    present = wave.ne(0).any(dim=-1, keepdim=True)
    if amp_scale:
        shape = (wave.shape[0],) + (1,) * (wave.ndim - 3) + (wave.shape[-2], 1)
        gain = torch.empty(shape, device=wave.device, dtype=wave.dtype).uniform_(
            1 - amp_scale, 1 + amp_scale
        )
        wave.mul_(gain)
    if noise_std:
        wave.add_(torch.randn_like(wave) * noise_std * present)
    shift = int(round(shift_sec * 128))
    if shift:
        amount = int(torch.randint(-shift, shift + 1, ()).item())
        wave.copy_(wave.roll(amount, dims=-1))
    return wave
