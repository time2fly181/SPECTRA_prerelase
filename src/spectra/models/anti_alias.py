"""Fixed Kaiser-window anti-aliasing filters for one-dimensional signals."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_kaiser_lowpass(
    cutoff_ratio: float = 0.85,
    num_taps: int = 9,
    beta: float = 6.0,
    *,
    stride: int = 2,
    legacy_cutoff: bool = False,
) -> torch.Tensor:
    """Return a unit-DC-gain low-pass FIR for decimation by ``stride``.

    Args:
        cutoff_ratio: Cutoff as a fraction of the reference Nyquist.
        num_taps: Odd FIR length.
        beta: Kaiser window parameter.
        stride: Decimation factor the filter precedes.
        legacy_cutoff: Measure ``cutoff_ratio`` against the *input* Nyquist
            instead of the post-decimation Nyquist. This reproduces the filter
            design used before the decimation cutoff was corrected, which passed
            ``stride`` times the intended band. For ``stride=1`` the two designs
            coincide. Enable only to reconstruct pre-fix checkpoints.

    Returns:
        ``[num_taps]`` FIR coefficients normalized to unit DC gain.
    """
    if not isinstance(stride, int) or stride < 1:
        raise ValueError(f"stride must be an integer >= 1, got {stride!r}")
    if not 0.0 < cutoff_ratio <= 1.0:
        raise ValueError(f"cutoff_ratio must be in (0, 1], got {cutoff_ratio!r}")
    if not isinstance(num_taps, int) or num_taps < 1 or num_taps % 2 == 0:
        raise ValueError(f"num_taps must be a positive odd integer, got {num_taps!r}")
    if beta < 0:
        raise ValueError(f"beta must be nonnegative, got {beta!r}")
    n = torch.arange(num_taps, dtype=torch.float32) - (num_taps - 1) / 2
    reference_nyquist = 0.5 if legacy_cutoff else 0.5 / stride
    cutoff = cutoff_ratio * reference_nyquist
    kernel = torch.sinc(2.0 * cutoff * n)
    if not legacy_cutoff:
        # The prefactor is removed again by the unit-DC-gain normalization below.
        # Skipping it on the legacy path keeps that path bit-identical to the
        # pre-fix implementation rather than merely equal to within fp32 epsilon.
        kernel = 2.0 * cutoff * kernel
    kernel = kernel * torch.kaiser_window(num_taps, periodic=False, beta=beta)
    return kernel / kernel.sum()


def _safe_symmetric_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return x
    mode = "reflect" if x.size(-1) > pad else "replicate"
    return F.pad(x, (pad, pad), mode=mode)


class KaiserAntiAliasDownsample1D(nn.Module):
    """Fixed depthwise Kaiser filtering followed by integer decimation."""

    kernel: torch.Tensor

    def __init__(
        self,
        channels: int,
        cutoff_ratio: float = 0.85,
        num_taps: int = 9,
        beta: float = 6.0,
        stride: int = 2,
        legacy_cutoff: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError(f"channels must be a positive integer, got {channels!r}")
        self.channels = channels
        self.stride = stride
        self.cutoff_ratio = float(cutoff_ratio)
        self.num_taps = num_taps
        self.beta = float(beta)
        self.legacy_cutoff = bool(legacy_cutoff)
        self.pad = (num_taps - 1) // 2
        kernel = create_kaiser_lowpass(
            cutoff_ratio, num_taps, beta, stride=stride, legacy_cutoff=legacy_cutoff
        ).view(1, 1, -1)
        # This buffer is deterministically rebuilt from the constructor config.
        # Keeping it out of state_dict lets checkpoints survive filter-tap upgrades.
        self.register_buffer(
            "kernel", kernel.expand(channels, 1, -1).contiguous(), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.channels:
            raise ValueError(f"expected [B, {self.channels}, T], got {tuple(x.shape)}")
        x = _safe_symmetric_pad(x, self.pad)
        return F.conv1d(
            x, self.kernel.to(dtype=x.dtype), stride=self.stride, groups=self.channels
        )

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, stride={self.stride}, "
            f"cutoff_ratio={self.cutoff_ratio}, num_taps={self.num_taps}, "
            f"beta={self.beta}, legacy_cutoff={self.legacy_cutoff}"
        )


class KaiserAntiAliasUpsample1D(nn.Module):
    """Linear upsampling followed by a fixed anti-imaging low-pass FIR."""

    kernel: torch.Tensor

    def __init__(
        self,
        channels: int,
        cutoff_ratio: float = 0.85,
        num_taps: int = 9,
        beta: float = 6.0,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError(f"channels must be a positive integer, got {channels!r}")
        if not isinstance(scale_factor, int) or scale_factor < 1:
            raise ValueError(
                f"scale_factor must be an integer >= 1, got {scale_factor!r}"
            )
        self.channels = channels
        self.scale_factor = scale_factor
        self.cutoff_ratio = float(cutoff_ratio)
        self.num_taps = num_taps
        self.beta = float(beta)
        self.pad = (num_taps - 1) // 2
        kernel = create_kaiser_lowpass(
            cutoff_ratio, num_taps, beta, stride=scale_factor
        ).view(1, 1, -1)
        self.register_buffer(
            "kernel", kernel.expand(channels, 1, -1).contiguous(), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.channels:
            raise ValueError(f"expected [B, {self.channels}, T], got {tuple(x.shape)}")
        if x.size(-1) < 1:
            raise ValueError("input temporal length must be at least 1")
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode="linear", align_corners=False
        )
        x = _safe_symmetric_pad(x, self.pad)
        return F.conv1d(x, self.kernel.to(dtype=x.dtype), groups=self.channels)
