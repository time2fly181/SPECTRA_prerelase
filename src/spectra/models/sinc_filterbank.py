"""Constrained sinc filters used by the multirate EEG branch."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value.clamp_min(1e-4)))


def _overlapping_passbands(
    num_filters: int,
    low_hz: float,
    high_hz: float,
) -> list[tuple[float, float]]:
    """Create deterministic logarithmic, overlapping passbands."""
    if num_filters < 1:
        return []
    if not (0.0 < low_hz < high_hz):
        raise ValueError(f"Invalid frequency range ({low_hz}, {high_hz})")
    anchors = torch.logspace(
        math.log10(low_hz), math.log10(high_hz), steps=num_filters + 2
    )
    return [
        (float(anchors[idx]), float(anchors[idx + 2])) for idx in range(num_filters)
    ]


_SIGMA_BAND_HZ: tuple[float, float] = (11.0, 16.0)

_SIGMA_CLUSTER_HZ: tuple[float, float] = (8.0, 18.0)


class ConstrainedSincFilterBank(nn.Module):
    """Optional constrained sinc filter bank with trainable passband edges."""

    def __init__(
        self,
        num_filters: int,
        *,
        fs: int,
        kernel_size: int,
        low_hz: float,
        high_hz: float,
        band_limits_hz: Sequence[tuple[float, float]] | None = None,
        initial_bands_hz: Sequence[tuple[float, float]] | None = None,
        minimum_bandwidth_hz: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_filters = int(num_filters)
        self.fs = int(fs)
        self.kernel_size = int(kernel_size)
        self.minimum_bandwidth_hz = float(minimum_bandwidth_hz)
        bands = list(
            initial_bands_hz
            if initial_bands_hz is not None
            else _overlapping_passbands(num_filters, low_hz, high_hz)
        )
        if len(bands) != self.num_filters:
            raise ValueError(
                f"initial_bands_hz must have {self.num_filters} entries, got {len(bands)}"
            )
        limits = None
        if band_limits_hz is not None:
            limits = torch.tensor(tuple(band_limits_hz), dtype=torch.float32)
            if limits.shape != (self.num_filters, 2):
                raise ValueError(
                    "band_limits_hz must have shape "
                    f"({self.num_filters}, 2), got {tuple(limits.shape)}"
                )
            if self.minimum_bandwidth_hz <= 0.0:
                raise ValueError("minimum_bandwidth_hz must be positive")
            if bool((limits[:, 0] <= 0).any()) or bool(
                (limits[:, 1] > self.fs / 2).any()
            ):
                raise ValueError("band limits must lie in (0, fs/2]")
            if bool((limits[:, 1] - limits[:, 0] <= self.minimum_bandwidth_hz).any()):
                raise ValueError("every band limit must exceed the minimum bandwidth")
            for index, ((low, high), (limit_low, limit_high)) in enumerate(
                zip(bands, limits.tolist(), strict=True)
            ):
                tolerance = 1e-5
                if not (limit_low - tolerance <= low < high <= limit_high + tolerance):
                    raise ValueError(
                        f"initial band {index} {(low, high)} lies outside "
                        f"its limits {(limit_low, limit_high)}"
                    )
                bands[index] = (max(low, limit_low), min(high, limit_high))
        self.register_buffer("_band_limits_hz", limits, persistent=False)
        self.raw_low_hz = nn.Parameter(torch.empty(self.num_filters))
        self.raw_band_hz = nn.Parameter(torch.empty(self.num_filters))
        self.reset_parameters(bands)

    @staticmethod
    def _logit(value: torch.Tensor) -> torch.Tensor:
        """Return a finite inverse sigmoid for boundary-valued initializers."""
        eps = torch.finfo(value.dtype).eps * 16
        value = value.clamp(eps, 1.0 - eps)
        return torch.logit(value)

    def reset_parameters(self, bands: Sequence[tuple[float, float]]) -> None:
        """Reset learned edges to the supplied passbands."""
        lows = torch.tensor([band[0] for band in bands], dtype=torch.float32)
        highs = torch.tensor([band[1] for band in bands], dtype=torch.float32)
        with torch.no_grad():
            if self._band_limits_hz is None:
                self.raw_low_hz.copy_(_inverse_softplus(lows))
                self.raw_band_hz.copy_(_inverse_softplus(highs - lows))
                return
            limits = cast(torch.Tensor, self._band_limits_hz).cpu()
            available_low = limits[:, 1] - limits[:, 0] - self.minimum_bandwidth_hz
            low_fraction = (lows - limits[:, 0]) / available_low
            remaining = limits[:, 1] - lows - self.minimum_bandwidth_hz
            width_fraction = (highs - lows - self.minimum_bandwidth_hz) / remaining
            self.raw_low_hz.copy_(self._logit(low_fraction))
            self.raw_band_hz.copy_(self._logit(width_fraction))

    def band_edges_hz(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return differentiable low and high cutoff frequencies in Hz."""
        if self._band_limits_hz is None:
            low = F.softplus(self.raw_low_hz) + 1e-4
            high = (low + F.softplus(self.raw_band_hz)).clamp(max=self.fs / 2.0 - 1e-3)
            return low, high
        limits = cast(torch.Tensor, self._band_limits_hz).to(self.raw_low_hz.device)
        available_low = limits[:, 1] - limits[:, 0] - self.minimum_bandwidth_hz
        low = limits[:, 0] + available_low * torch.sigmoid(self.raw_low_hz)
        remaining = limits[:, 1] - low - self.minimum_bandwidth_hz
        high = (
            low
            + self.minimum_bandwidth_hz
            + remaining * torch.sigmoid(self.raw_band_hz)
        )
        return low, high

    def kernels(self) -> torch.Tensor:
        low, high = self.band_edges_hz()
        n = (
            torch.arange(self.kernel_size, device=low.device, dtype=torch.float32)
            - (self.kernel_size - 1) / 2
        )
        low_n = low.float().unsqueeze(1)
        high_n = high.float().unsqueeze(1)
        high_lp = (
            2.0 * high_n / self.fs * torch.sinc(2.0 * high_n / self.fs * n.unsqueeze(0))
        )
        low_lp = (
            2.0 * low_n / self.fs * torch.sinc(2.0 * low_n / self.fs * n.unsqueeze(0))
        )
        window = torch.hann_window(
            self.kernel_size, periodic=False, device=low.device, dtype=torch.float32
        )
        kernels = (high_lp - low_lp) * window.unsqueeze(0)
        kernels = kernels - kernels.mean(dim=-1, keepdim=True)
        kernels = F.normalize(kernels, dim=-1)
        return kernels.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernels = self.kernels().to(dtype=x.dtype)
        return F.conv1d(x, kernels, padding=self.kernel_size // 2)
