"""Shared NumPy robust normalization for conversion and inference."""

from __future__ import annotations

import numpy as np

from .signal_quality import epoch_signal_validity

__all__ = [
    "normalize_channel",
    "normalize_channel_masked",
    "normalize_channel_masked_with_validity",
]


def normalize_channel(
    signal: np.ndarray,
    *,
    clip_threshold: float = 20.0,
) -> tuple[np.ndarray, dict]:
    """Normalize a single channel using robust scaling.

    Args:
        signal: One-dimensional channel samples.
        clip_threshold: Symmetric normalized clipping threshold.

    Returns:
        Tuple of the float32 normalized samples and normalization statistics.
    """
    median = np.median(signal)
    p25 = np.percentile(signal, 25.0)
    p75 = np.percentile(signal, 75.0)
    iqr = p75 - p25

    stats = {
        "median": float(median),
        "q1": float(p25),
        "q3": float(p75),
        "iqr": float(iqr),
        "clip_threshold": float(clip_threshold),
        "mode": "iqr",
    }

    if abs(iqr) < 1e-8:
        standard_deviation = np.std(signal, dtype=np.float64)
        if standard_deviation > 1e-8:
            scale = standard_deviation
            stats["iqr"] = float(standard_deviation)
            stats["mode"] = "std_fallback"
        else:
            scale = 1.0
            stats["iqr"] = 1.0
            stats["mode"] = "flat_signal"

        signal_normalized = (signal - median) / scale
        signal_normalized = np.clip(
            signal_normalized,
            -float(clip_threshold),
            float(clip_threshold),
        )
        return signal_normalized.astype(np.float32), stats

    signal_normalized = (signal - median) / iqr
    signal_normalized = np.clip(
        signal_normalized,
        -float(clip_threshold),
        float(clip_threshold),
    )
    return signal_normalized.astype(np.float32), stats


def normalize_channel_masked(
    signal_epochs: np.ndarray,
    stat_mask: np.ndarray,
    *,
    min_stat_epochs: int = 10,
    clip_threshold: float = 20.0,
) -> tuple[np.ndarray, dict]:
    """Robust-scale a channel using statistics from selected valid epochs.

    Args:
        signal_epochs: Channel signal shaped ``[n_epochs, samples]``.
        stat_mask: Boolean mask selecting epochs used for statistics.
        min_stat_epochs: Minimum selected epochs required for normalization.
        clip_threshold: Symmetric normalized clipping threshold.

    Returns:
        Tuple of normalized float32 epochs and normalization statistics.

    Raises:
        ValueError: If shapes are invalid, too few epochs are selected, or
            normalization produces invalid statistics or output.
    """
    epochs = np.asarray(signal_epochs)
    mask = np.asarray(stat_mask, dtype=bool)
    if epochs.ndim != 2:
        raise ValueError(
            f"signal_epochs must have shape [n_epochs, samples], got {epochs.shape}"
        )
    if mask.ndim != 1 or mask.shape[0] != epochs.shape[0]:
        raise ValueError(
            f"stat_mask must have shape {(epochs.shape[0],)}, got {mask.shape}"
        )
    if min_stat_epochs <= 0:
        raise ValueError(f"min_stat_epochs must be positive, got {min_stat_epochs}")
    n_selected = int(mask.sum())
    if n_selected < min_stat_epochs:
        raise ValueError(
            f"Only {n_selected} signal-valid epochs are available; "
            f"at least {min_stat_epochs} are required"
        )

    stat_signal = epochs[mask].ravel()
    if not np.all(np.isfinite(stat_signal)):
        raise ValueError("Selected normalization samples must all be finite")
    _, stats = normalize_channel(stat_signal, clip_threshold=clip_threshold)
    stats["n_stat_epochs"] = n_selected

    median = stats["median"]
    scale = stats["iqr"]
    if not np.isfinite(median) or not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"Invalid normalization statistics: median={median}, scale={scale}"
        )
    normalized = (epochs - median) / scale
    normalized = np.clip(
        normalized,
        -float(clip_threshold),
        float(clip_threshold),
    )
    normalized[~mask] = 0.0
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Normalization produced non-finite output")
    return normalized.astype(np.float32), stats


def normalize_channel_masked_with_validity(
    signal_epochs: np.ndarray,
    stat_mask: np.ndarray,
    *,
    min_stat_epochs: int = 10,
    clip_threshold: float = 20.0,
    max_refinement_steps: int = 8,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Normalize a channel and reconcile validity with the stored waveform.

    Normalization and clipping can turn a previously varying epoch into a flat
    waveform. This function removes such epochs from the statistics mask,
    recomputes normalization, and repeats until the mask matches the normalized
    output.

    Args:
        signal_epochs: Channel signal shaped ``[n_epochs, samples]``.
        stat_mask: Initial pre-normalization validity mask.
        min_stat_epochs: Minimum valid epochs required for normalization.
        clip_threshold: Symmetric normalized clipping threshold.
        max_refinement_steps: Maximum mask/statistics refinement iterations.

    Returns:
        Tuple of normalized float32 epochs, normalization statistics, and the
        final post-normalization validity mask.

    Raises:
        ValueError: If inputs are invalid, too few usable epochs remain, or
            validity refinement does not converge.
    """
    if max_refinement_steps <= 0:
        raise ValueError(
            f"max_refinement_steps must be positive, got {max_refinement_steps}"
        )

    validity = np.asarray(stat_mask, dtype=bool).copy()
    for _ in range(max_refinement_steps):
        normalized, stats = normalize_channel_masked(
            signal_epochs,
            validity,
            min_stat_epochs=min_stat_epochs,
            clip_threshold=clip_threshold,
        )
        refined_validity = validity & epoch_signal_validity(normalized)
        if np.array_equal(refined_validity, validity):
            return normalized, stats, validity
        validity = refined_validity

    raise ValueError(
        "Post-normalization signal validity did not converge after "
        f"{max_refinement_steps} refinement steps"
    )
