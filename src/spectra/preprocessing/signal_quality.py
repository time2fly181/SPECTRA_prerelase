"""Shared epoch-level signal quality checks for conversion and inference."""

from __future__ import annotations

import numpy as np

__all__ = ["epoch_signal_validity", "recording_epoch_signal_validity"]


def epoch_signal_validity(
    channel_epochs: np.ndarray, *, flat_eps: float = 1e-8
) -> np.ndarray:
    """Return usability flags for epochs from one physiological channel.

    Args:
        channel_epochs: Signal shaped ``[n_epochs, samples_per_epoch]``.
        flat_eps: Minimum within-epoch standard deviation.

    Returns:
        Boolean array shaped ``[n_epochs]``. An epoch is valid only when all
        samples are finite and the waveform is neither zero-filled nor flat.

    Raises:
        ValueError: If the input is not two-dimensional or ``flat_eps`` is
            negative.
    """
    epochs = np.asarray(channel_epochs)
    if epochs.ndim != 2:
        raise ValueError(
            "channel_epochs must have shape [n_epochs, samples_per_epoch], "
            f"got {epochs.shape}"
        )
    if flat_eps < 0:
        raise ValueError(f"flat_eps must be non-negative, got {flat_eps}")

    finite = np.all(np.isfinite(epochs), axis=1)
    standard_deviation = np.zeros(epochs.shape[0], dtype=np.float64)
    if finite.any():
        standard_deviation[finite] = np.std(
            epochs[finite],
            axis=1,
            dtype=np.float64,
        )
    return finite & np.any(epochs != 0.0, axis=1) & (standard_deviation >= flat_eps)


def recording_epoch_signal_validity(
    recording: np.ndarray,
    samples_per_epoch: int,
    *,
    presence_mask: np.ndarray | None = None,
    flat_eps: float = 1e-8,
) -> np.ndarray:
    """Return per-epoch, per-channel validity for a complete recording.

    Args:
        recording: Signal shaped ``[channels, samples]``.
        samples_per_epoch: Number of samples in each epoch.
        presence_mask: Optional recording-level channel availability ``[C]``.
        flat_eps: Minimum within-epoch standard deviation.

    Returns:
        Boolean array shaped ``[n_epochs, channels]``. Incomplete trailing
        samples are ignored and absent channels are always invalid.

    Raises:
        ValueError: If shapes or ``samples_per_epoch`` are invalid.
    """
    signal = np.asarray(recording)
    if signal.ndim != 2:
        raise ValueError(
            f"recording must have shape [channels, samples], got {signal.shape}"
        )
    if samples_per_epoch <= 0:
        raise ValueError(f"samples_per_epoch must be positive, got {samples_per_epoch}")

    channels, samples = signal.shape
    n_epochs = samples // samples_per_epoch
    validity = np.zeros((n_epochs, channels), dtype=bool)
    if n_epochs == 0:
        return validity

    if presence_mask is None:
        present = np.ones(channels, dtype=bool)
    else:
        present = np.asarray(presence_mask, dtype=bool)
        if present.shape != (channels,):
            raise ValueError(
                f"presence_mask must have shape {(channels,)}, got {present.shape}"
            )

    complete = signal[:, : n_epochs * samples_per_epoch].reshape(
        channels, n_epochs, samples_per_epoch
    )
    for channel_index in np.flatnonzero(present):
        validity[:, channel_index] = epoch_signal_validity(
            complete[channel_index], flat_eps=flat_eps
        )
    return validity
