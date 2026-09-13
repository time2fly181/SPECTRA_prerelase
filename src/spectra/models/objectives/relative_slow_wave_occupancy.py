"""Relative slow-wave occupancy targets and a structured retention head.

The canonical PSGStage waveform is stored in per-recording IQR units, so an
absolute microvolt threshold cannot be recovered reliably for every existing
store.  This module instead measures a versioned *relative* occupancy curve:
the fraction of an epoch whose 0.5--2 Hz analytic amplitude exceeds several
fixed peak-to-peak thresholds in IQR units.

The curve is a training target, not a clinical AASM score.  It is deliberately
computed from the clean primary SupCon view and never from an amplitude-altered
augmentation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "RELATIVE_SLOW_WAVE_TARGET_VERSION",
    "RELATIVE_SLOW_WAVE_THRESHOLDS",
    "RelativeSlowWaveOccupancyHead",
]

RELATIVE_SLOW_WAVE_TARGET_VERSION: int = 1

RELATIVE_SLOW_WAVE_THRESHOLDS: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0)

_SLOW_WAVE_BAND_HZ: tuple[float, float] = (0.5, 2.0)

_DEFAULT_SOFTNESS: float = 0.10


class RelativeSlowWaveOccupancyHead(nn.Module):
    """Linearly expose a monotonic relative occupancy curve from an embedding.

    The first threshold logit is unconstrained. Remaining outputs are positive
    decrements from the preceding logit, guaranteeing that predicted occupancy
    cannot increase as the amplitude threshold rises.
    """

    def __init__(
        self,
        in_dim: int,
        *,
        eeg_channels: int,
        thresholds: Sequence[float] = RELATIVE_SLOW_WAVE_THRESHOLDS,
    ) -> None:
        super().__init__()
        threshold_values = tuple(float(value) for value in thresholds)
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if eeg_channels <= 0:
            raise ValueError(f"eeg_channels must be positive, got {eeg_channels}")
        if not threshold_values:
            raise ValueError("thresholds must not be empty")
        if any(
            right <= left
            for left, right in zip(
                threshold_values[:-1], threshold_values[1:], strict=True
            )
        ):
            raise ValueError(
                f"thresholds must be strictly increasing, got {threshold_values}"
            )

        self.in_dim = int(in_dim)
        self.eeg_channels = int(eeg_channels)
        self.thresholds = threshold_values
        self.linear = nn.Linear(self.in_dim, self.eeg_channels * len(self.thresholds))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(
            self.linear.weight, mode="fan_out", nonlinearity="linear"
        )
        with torch.no_grad():
            bias = self.linear.bias.view(self.eeg_channels, len(self.thresholds))
            bias[:, 0] = math.log(0.2 / 0.8)
            if bias.size(1) > 1:
                bias[:, 1:] = math.log(math.expm1(0.25))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2 or embedding.size(1) != self.in_dim:
            raise ValueError(
                f"embedding must have shape [B, {self.in_dim}], got "
                f"{tuple(embedding.shape)}"
            )
        raw = self.linear(embedding).view(
            embedding.size(0), self.eeg_channels, len(self.thresholds)
        )
        first = raw[..., :1]
        if raw.size(-1) == 1:
            logits = first
        else:
            decrements = F.softplus(raw[..., 1:])
            logits = torch.cat([first, first - decrements.cumsum(dim=-1)], dim=-1)
        return torch.sigmoid(logits)

    def get_config(self) -> dict[str, Any]:
        """Return checkpoint metadata required to reconstruct the head."""
        return {
            "version": RELATIVE_SLOW_WAVE_TARGET_VERSION,
            "in_dim": self.in_dim,
            "eeg_channels": self.eeg_channels,
            "thresholds": self.thresholds,
        }
