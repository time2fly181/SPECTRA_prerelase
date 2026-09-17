"""EDF preprocessing matching the canonical normalized-float32 converter.

Channels are selected before direct 128 Hz resampling. No bandpass, notch, or
algebraic rereferencing is applied. Statistics use signal-valid epochs within
the requested interval; only invalid exterior epochs are removed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyedflib

from spectra.data.channel import build_substitution_map, select_5ch_for_model

from .edf_reader import (
    _EDF_REPAIRABLE_HEADER_ERRORS,
    _labels_for_channel_mapping,
    _open_edf_with_mne,
    _repair_edf_header,
    resample_signal,
)
from .robust_normalization import normalize_channel_masked_with_validity
from .signal_quality import epoch_signal_validity

logger = logging.getLogger(__name__)
CHANNEL_NAMES = ["EEG1", "EEG2", "EOG1", "EOG2", "EMG"]
SAMPLE_RATE = 128
EPOCH_SAMPLES = 3840


@contextmanager
def open_edf_reader(path: str) -> Iterator[tuple[Any, frozenset[int]]]:
    """Open an EDF with the converter's temporary repair and MNE fallback."""
    reader = None
    repair = None
    try:
        try:
            reader = pyedflib.EdfReader(path)
        except Exception as exc:
            if any(message in str(exc) for message in _EDF_REPAIRABLE_HEADER_ERRORS):
                repair = _repair_edf_header(path)
            if repair is not None:
                logger.warning("EDF header repaired temporarily: %s", repair.changes)
                reader = pyedflib.EdfReader(repair.path)
            else:
                try:
                    reader = _open_edf_with_mne(path)
                except Exception as fallback_exc:
                    raise ValueError(
                        f"Cannot open EDF: pyedflib failed ({exc}); "
                        f"MNE failed ({fallback_exc})"
                    ) from fallback_exc
                logger.warning("Using MNE EDF reader for %s", Path(path).name)
        yield reader, repair.unusable_signal_indices if repair else frozenset()
    finally:
        try:
            if reader is not None:
                reader.close()
        finally:
            if repair is not None:
                Path(repair.path).unlink(missing_ok=True)


def select_channel_plan(
    labels: list[str], canonical_channels: list[str] | None = None
) -> list[tuple[str, str | None]]:
    """Resolve five slots using converter priorities and optional overrides.

    Generic slot names represent automatic selection. Explicit names use the
    converter's direct/substitution matcher, with rereferencing disabled.

    Args:
        labels: Recorded EDF channel labels in header order.
        canonical_channels: Five requested names in EEG/EEG/EOG/EOG/EMG order.
            ``None`` selects all slots automatically. Generic names in
            ``CHANNEL_NAMES`` request automatic selection for that slot only.

    Returns:
        Five ``(requested_name, recorded_name)`` pairs. Missing sources are
        ``None``. Automatic selection does not enforce frontal/central anatomy.

    Raises:
        ValueError: If the request has the wrong length or repeats a source.
    """
    requested = CHANNEL_NAMES if canonical_channels is None else canonical_channels
    if len(requested) != 5:
        raise ValueError("SPECTRA requires five ordered EEG/EEG/EOG/EOG/EMG slots")
    explicit = [
        name
        for name, slot in zip(requested, CHANNEL_NAMES, strict=True)
        if name != slot
    ]
    matches = build_substitution_map(explicit, labels) if explicit else {}
    used = {actual for actual, _ in matches.values() if actual is not None}
    remaining = [label for label in labels if label not in used]
    automatic, present = select_5ch_for_model(remaining)
    # Exact converter fallback for mentalis/ment/chin labels.
    if not present[4]:
        source = next(
            (
                label
                for label in remaining
                if re.search(r"(ment|mentalis|chin|submental|emg)", label, re.I)
            ),
            None,
        )
        automatic[4] = ("EMG", source)
    queues = {
        "EEG": [source for _, source in automatic[:2] if source is not None],
        "EOG": [source for _, source in automatic[2:4] if source is not None],
        "EMG": [source for _, source in automatic[4:] if source is not None],
    }
    plan = []
    for name, slot in zip(requested, CHANNEL_NAMES, strict=True):
        if name != slot:
            source = matches.get(name, (None, None))[0]
        else:
            candidates = queues[slot[:3]]
            source = candidates.pop(0) if candidates else None
        plan.append((name, source))
    chosen = [source for _, source in plan if source is not None]
    if len(set(chosen)) != len(chosen):
        raise ValueError(
            "Channel mapping repeats a recorded channel; choose distinct slot overrides"
        )
    return plan


@dataclass
class RecordingSignals:
    """Selected unnormalized waveforms on the original 30-second epoch grid.

    Attributes:
        signals: Float32 ``[5, original_n_epochs * 3840]`` at 128 Hz in reader
            physical units; unavailable channels and incomplete epochs are zero.
        presence_mask: Uint8 ``[5]`` selection mask; not an epoch-quality mask.
        channel_names: Requested slot names in EEG/EEG/EOG/EOG/EMG order.
        channel_mapping: Requested names mapped to recorded names or ``None``.
        sample_rates: Native sampling rates in Hz keyed by recorded channel.
        reader_backend: Reader identifier for provenance.
    """

    signals: np.ndarray
    presence_mask: np.ndarray
    channel_names: list[str]
    channel_mapping: dict[str, str | None]
    sample_rates: dict[str, float]
    reader_backend: str


@dataclass
class PreparedRecording:
    """Normalized recording and provenance, without requiring stage labels.

    Attributes:
        signals_stacked: Contiguous float32 ``[N, 5, 3840]`` dimensionless epochs,
            where ``N = end_epoch - start_epoch``. Invalid channel epochs are zero.
        presence_mask: Uint8 ``[5]``; one for channels that can be normalized.
        epoch_signal_valid: Boolean ``[N, 5]``; true for usable channel epochs.
        channel_names: Requested names in EEG/EEG/EOG/EOG/EMG order.
        channel_mapping: Requested names mapped to recorded names or ``None``.
        normalization: Per-channel scaling statistics over the analysis interval.
        start_epoch: Inclusive offset into the original EDF epoch grid.
        end_epoch: Exclusive offset after optional exterior trimming.
        original_n_epochs: Complete 30-second epochs before cropping.
        reader_backend: Reader identifier for provenance.
    """

    signals_stacked: np.ndarray
    presence_mask: np.ndarray
    epoch_signal_valid: np.ndarray
    channel_names: list[str]
    channel_mapping: dict[str, str | None]
    normalization: dict[str, dict[str, Any]]
    start_epoch: int
    end_epoch: int
    original_n_epochs: int
    reader_backend: str


def load_edf_signals(
    path: str, canonical_channels: list[str] | None = None
) -> RecordingSignals:
    """Select recorded channels and resample each directly to 128 Hz.

    Missing channels and incomplete channel epochs are zero-filled. Recording
    duration defines the epoch grid, as in the canonical converter. The primary
    reader uses EDF header physical units; the MNE adapter returns MNE values
    (volts for voltage channels) at MNE's common sampling rate. Neither path
    applies model normalization here.

    Args:
        path: EDF file to read; the source file is never modified.
        canonical_channels: Optional five-slot request for ``select_channel_plan``.

    Returns:
        Unnormalized signals, channel availability, and reader provenance.

    Raises:
        ValueError: If the recording has no complete epoch, no selected channel,
            an invalid sampling rate, or an unsupported channel request.
    """
    with open_edf_reader(path) as (reader, unusable):
        labels = _labels_for_channel_mapping(list(reader.getSignalLabels()), unusable)
        indices = {label: i for i, label in enumerate(labels) if i not in unusable}
        plan = select_channel_plan(labels, canonical_channels)
        n_epochs = int(np.floor(float(reader.getFileDuration()) / 30.0))
        if n_epochs < 1:
            raise ValueError("Recording is shorter than one complete 30-second epoch")
        signals = np.zeros((5, n_epochs * EPOCH_SAMPLES), dtype=np.float32)
        presence = np.zeros(5, dtype=np.uint8)
        rates = {}
        for slot, (_, source) in enumerate(plan):
            if source is None or source not in indices:
                continue
            index = indices[source]
            fs = float(reader.getSampleFrequency(index))
            if not np.isfinite(fs) or fs <= 0:
                raise ValueError(f"Invalid sampling frequency for {source}: {fs}")
            signal = reader.readSignal(index)
            if abs(fs - SAMPLE_RATE) > 1e-6:
                signal = resample_signal(signal, fs, SAMPLE_RATE)
            count = min(n_epochs, len(signal) // EPOCH_SAMPLES) * EPOCH_SAMPLES
            signals[slot, :count] = signal[:count]
            presence[slot] = 1
            rates[source] = fs
        if not presence.any():
            raise ValueError("No usable model channels matched the EDF recording")
        return RecordingSignals(
            signals,
            presence,
            [name for name, _ in plan],
            dict(plan),
            rates,
            getattr(reader, "backend", "pyedflib"),
        )


def preprocess_edf(
    path: str,
    canonical_channels: list[str] | None = None,
    *,
    start_epoch: int = 0,
    end_epoch: int = -1,
    auto_signal_window: bool = True,
) -> PreparedRecording:
    """Read and robust-normalize an EDF analysis interval for model inference.

    Use the converter's annotation crop bounds as ``start_epoch`` and
    ``end_epoch`` when comparing against an annotated Zarr. Without bounds,
    inference uses all complete epochs; no labels are invented or required.

    Args:
        path: EDF file to read.
        canonical_channels: Optional five-slot request for ``select_channel_plan``.
        start_epoch: Inclusive zero-based analysis offset; must be nonnegative.
        end_epoch: Exclusive analysis offset, clipped to recording length.
            ``-1`` uses the recording end.
        auto_signal_window: Trim fully invalid exterior epochs after normalization.
            Interior gaps remain in place with false validity masks.

    Returns:
        PreparedRecording with normalized epochs and offsets into the source EDF.

    Raises:
        ValueError: If bounds are invalid or no channel has enough usable epochs.

    Notes:
        A channel that cannot be normalized is disabled with a warning. Scaling
        uses only valid epochs, clips the result, then reconciles validity with
        the stored waveform. See ``normalize_channel_masked_with_validity`` for
        thresholds and convergence limits. No bandpass or notch filter is applied.
    """
    recording = load_edf_signals(path, canonical_channels)
    original_n_epochs = recording.signals.shape[1] // EPOCH_SAMPLES
    start = int(start_epoch)
    end = (
        original_n_epochs if end_epoch == -1 else min(int(end_epoch), original_n_epochs)
    )
    if start < 0 or start >= end:
        raise ValueError(f"Invalid analysis interval [{start}, {end})")
    raw = recording.signals.reshape(5, original_n_epochs, EPOCH_SAMPLES)[:, start:end]
    output = np.zeros((end - start, 5, EPOCH_SAMPLES), dtype=np.float32)
    valid = np.zeros((end - start, 5), dtype=bool)
    presence = recording.presence_mask.copy()
    statistics = {}
    for c, name in enumerate(recording.channel_names):
        if not presence[c]:
            continue
        try:
            normalized, stats, mask = normalize_channel_masked_with_validity(
                raw[c], epoch_signal_validity(raw[c])
            )
        except ValueError as exc:
            logger.warning("Channel %s unavailable: %s", name, exc)
            presence[c] = 0
            continue
        output[:, c] = normalized
        valid[:, c] = mask
        statistics[name] = stats
    usable = np.flatnonzero(valid.any(axis=1))
    if not presence.any() or usable.size == 0:
        raise ValueError(
            "No channel has enough usable signal for normalization (minimum 10 epochs)"
        )
    left, right = (
        (int(usable[0]), int(usable[-1]) + 1)
        if auto_signal_window
        else (0, end - start)
    )
    return PreparedRecording(
        np.ascontiguousarray(output[left:right]),
        presence,
        valid[left:right],
        recording.channel_names,
        recording.channel_mapping,
        statistics,
        start + left,
        start + right,
        original_n_epochs,
        recording.reader_backend,
    )
