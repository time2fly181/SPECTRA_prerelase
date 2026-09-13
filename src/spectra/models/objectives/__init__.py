"""Compatibility heads retained for strict checkpoint reconstruction."""

from __future__ import annotations

from .relative_slow_wave_occupancy import (
    RELATIVE_SLOW_WAVE_TARGET_VERSION,
    RELATIVE_SLOW_WAVE_THRESHOLDS,
    RelativeSlowWaveOccupancyHead,
)

__all__ = [
    "RELATIVE_SLOW_WAVE_TARGET_VERSION",
    "RELATIVE_SLOW_WAVE_THRESHOLDS",
    "RelativeSlowWaveOccupancyHead",
]
