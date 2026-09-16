"""Complete inference runtime for PSG sleep staging.

This module provides end-to-end inference on EDF files with:
- EDF loading and channel extraction
- Channel name normalization and substitution
- Canonical EDF resampling and signal-valid IQR normalization before inference
- Batched inference with context windows
- Post-processing and output saving
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly

from spectra.utils.device_utils import clear_device_cache, resolve_device

if TYPE_CHECKING:
    from spectra.models.multirate_asymmetric_epoch_cnn import (
        PerRecordingBandNorm,
    )

logger = logging.getLogger(__name__)

# Stage names
STAGE_NAMES_5 = ["W", "N1", "N2", "N3", "REM"]
DEFAULT_5CH_NAMES = ["EEG1", "EEG2", "EOG1", "EOG2", "EMG"]


class InferencePreprocessingError(Exception):
    """Raised when preprocessing fails during inference."""

    pass


def _resolve_inference_device(
    preference: str = "auto",
) -> tuple[torch.device, str | None, str]:
    """Resolve an inference device with fallback warning and display label."""

    resolved = resolve_device(preference)
    return resolved.torch_device, resolved.warning, resolved.description


def _should_use_cuda_prefetch(
    options: ScoreOptions,
    device: torch.device,
    n_batches: int,
) -> bool:
    """Return True only when CUDA stream prefetching is valid for this run."""

    if not options.cuda_prefetch:
        return False

    if device.type != "cuda":
        if n_batches > 0:
            logger.info(
                "CUDA prefetch requested but backend '%s' does not support CUDA streams; continuing without prefetch.",
                device.type,
            )
        return False

    return n_batches > 1


def _cleanup_inference_device_memory(device: torch.device | None) -> bool:
    """Best-effort cleanup for inference backends that cache accelerator memory."""

    if not isinstance(device, torch.device):
        return False
    resolved_device = cast(torch.device, device)
    if resolved_device.type not in {"cuda", "mps"}:
        return False
    return clear_device_cache(resolved_device, synchronize=True)


def _repair_edf_physical_dimensions(edf_path: str) -> str | None:
    """Repair EDFs with empty Physical Dimension fields into a temp copy.

    This mirrors the batch EDF->Zarr pipeline so inference can open the same
    non-compliant files that pyedflib would otherwise reject.
    """
    with open(edf_path, "rb") as f:
        data = bytearray(f.read())

    try:
        n_signals = int(bytes(data[252:256]).decode("ascii").strip())
    except (ValueError, UnicodeDecodeError):
        return None

    if n_signals <= 0:
        return None

    phys_dim_offset = 256 + (16 + 80) * n_signals
    field_len = 8
    needs_repair = False

    for i in range(n_signals):
        start = phys_dim_offset + i * field_len
        end = start + field_len
        field = bytes(data[start:end]).decode("ascii", errors="replace").strip()
        if not field:
            data[start:end] = b"uV" + b" " * 6
            needs_repair = True

    if not needs_repair:
        return None

    fd, tmp_path = tempfile.mkstemp(suffix=".edf")
    try:
        os.write(fd, bytes(data))
    finally:
        os.close(fd)
    return tmp_path


def extract_checkpoint_channel_names(
    checkpoint_dict: Mapping[str, Any] | None,
) -> list[str] | None:
    """Extract saved channel names from checkpoint model_config when present."""
    if not isinstance(checkpoint_dict, Mapping):
        return None

    model_config = checkpoint_dict.get("model_config")
    if not isinstance(model_config, Mapping):
        return None

    model_kwargs = model_config.get("model_kwargs")
    if not isinstance(model_kwargs, Mapping):
        return None

    saved_channel_names = model_kwargs.get("channel_names")
    if isinstance(saved_channel_names, str):
        parsed = [
            name.strip() for name in saved_channel_names.split(",") if name.strip()
        ]
        return parsed or None
    if isinstance(saved_channel_names, (list, tuple)):
        parsed = [str(name).strip() for name in saved_channel_names]
        parsed = [name for name in parsed if name]
        return parsed or None
    return None


def resolve_model_channel_names(
    checkpoint_dict: Mapping[str, Any] | None,
    *,
    canonical_channels: list[str] | None = None,
    default_channel_names: list[str] | None = None,
) -> tuple[list[str] | None, str]:
    """Resolve model-construction channel names with stable precedence."""
    if canonical_channels is not None:
        return list(canonical_channels), "canonical"

    checkpoint_channel_names = extract_checkpoint_channel_names(checkpoint_dict)
    if checkpoint_channel_names is not None:
        return checkpoint_channel_names, "checkpoint"

    fallback = (
        DEFAULT_5CH_NAMES if default_channel_names is None else default_channel_names
    )
    if fallback is None:
        return None, "none"
    return list(fallback), "default_5ch"


def _strip_legacy_router_context_keys(
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Remove legacy router-context keys that no longer exist in the model."""
    legacy_prefixes = (
        "classifier.router_context_norm.",
        "classifier.router_context_gate.",
        "moe_router_context_proj.",
    )
    legacy_scalars = {"classifier.router_context_scale"}

    filtered: dict[str, Any] = {}
    removed = 0
    for key, value in state_dict.items():
        clean_key = key[6:] if key.startswith("model.") else key
        if clean_key in legacy_scalars or any(
            clean_key.startswith(prefix) for prefix in legacy_prefixes
        ):
            removed += 1
            continue
        filtered[key] = value
    return filtered, removed


class InferenceModelLoadError(RuntimeError):
    """Raised when checkpoint weights cannot be loaded into the model."""

    pass


@dataclass
class ScoreOptions:
    """Options for inference scoring.

    Default preprocessing matches the training pipeline's sampling rate:
    - Resampling: 128 Hz
    - Channel filtering is intentionally disabled (signals are assumed to have
      been filtered during offline preprocessing).
    """

    # Signal processing
    epoch_sec: int = 30  # Epoch duration in seconds

    # Analysis window. Only this interval is normalized and run through the
    # model; full-length returned/saved arrays mark outside epochs unscored.
    start_epoch: int = 0  # First epoch of the analysis window (inclusive)
    end_epoch: int = -1  # Last+1 epoch (exclusive); -1 = to end of recording
    auto_signal_window: bool = True  # Trim exterior epochs with no usable channel

    # Inference
    amp_mode: str = "fp32"  # Mixed precision: fp16, bf16, fp32
    batch_size: int = 32
    sequential_loading: bool = (
        True  # Use sequential epoch loading (reduces memory by ~10-100x)
    )
    cuda_prefetch: bool = True  # CUDA-only: overlap CPU→GPU transfer with computation

    # Preprocessing calibration
    calibration_mode: str = "per_recording"  # checkpoint, per_recording, warmup
    prenormalized: bool = (
        False  # If True, zarr files contain prenormalized data (skip calibration)
    )

    # Monte Carlo dropout
    use_mc_dropout: bool = False
    mc_samples: int = 10
    mc_dropout_rate: float | None = None
    # How the MC samples are pooled into the returned tensor.
    #   "prob"  -> log(mean(softmax(logits))): the arithmetic mean of the
    #              per-sample posteriors, which is the MC-dropout predictive
    #              distribution. A downstream ``softmax`` recovers it exactly.
    #   "logit" -> mean(logits): softmax yields a normalized geometric mean.
    mc_pooling: str = "prob"
    # "legacy" retains the historical MC/overlap pooling rules.
    mc_aggregation: str = "consistent"
    # Include standard MHA and custom functional SDPA attention dropout.
    mc_include_attention: bool = True
    # Match qualified module names in the unwrapped model.
    # None selects existing sites throughout encoder, transformer, and classifier.
    # Use r"^classifier\." to restrict sampling to a dropout-bearing classifier.
    mc_module_pattern: str | None = None

    # Overlap-averaging (predict_all): when True, each epoch is scored by averaging
    # the LOGITS produced for it across every overlapping context window (train/test
    # parity with --predict_all eval) instead of using only the center prediction.
    # Off by default, retaining center-only predictions.
    overlap_average: bool = False

    # Averaged (EMA/SWA) weights: when True, load ``ema.module`` from the
    # checkpoint instead of the raw ``model`` weights. Training evaluates and
    # selects checkpoints on the averaged weights whenever averaging is enabled
    # (``best.ckpt.metrics.json`` records ``"weights": "ema"``), so scoring the
    # raw weights measures a different model than the one validation ranked.
    # Off by default to preserve historical numbers for existing checkpoints.
    prefer_averaged: bool = False

    @property
    def fs(self) -> int:
        """Fixed EDF/model sampling rate; not configurable."""
        return 128

    @property
    def context_half(self) -> int:
        """Fixed ten-epoch context on each side; not configurable."""
        return 10

    def __post_init__(self) -> None:
        """Validate inference and MC dropout settings."""
        _validate_mc_options(self.mc_samples, self.mc_dropout_rate)
        if self.mc_aggregation not in {"consistent", "legacy"}:
            raise ValueError("mc_aggregation must be 'consistent' or 'legacy'")

        mc_pooling = str(self.mc_pooling or "prob").strip().lower()
        if mc_pooling not in {"prob", "logit"}:
            raise ValueError(
                f"mc_pooling must be 'prob' or 'logit', got {self.mc_pooling!r}"
            )
        self.mc_pooling = mc_pooling

        pattern = self.mc_module_pattern
        if pattern is not None:
            pattern = str(pattern).strip() or None
            if pattern is not None:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"mc_module_pattern is not a valid regex: {pattern!r}"
                    ) from exc
        self.mc_module_pattern = pattern

        # Analysis window: clamp start non-negative and validate ordering.
        self.start_epoch = int(max(0, self.start_epoch))
        if self.end_epoch >= 0 and self.end_epoch <= self.start_epoch:
            raise ValueError(
                "end_epoch must be greater than start_epoch (or -1 for 'to end'); "
                f"got start_epoch={self.start_epoch}, end_epoch={self.end_epoch}"
            )


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    denom = np.sum(exp_x, axis=axis, keepdims=True)
    denom = np.clip(denom, 1e-12, np.inf)
    return (exp_x / denom).astype(np.float32, copy=False)


def compute_flag_scores(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-epoch uncertainty-flag scores from a posterior array.

    These are the cheap-pass flag metrics for uncertainty-gated adaptive compute.
    All three are derived directly from the softmax posterior with zero extra
    forward passes.

    Args:
        probabilities: Posterior array of shape ``(n_epochs, n_classes)``.

    Returns:
        Dict with three per-epoch ``float32`` arrays of shape ``(n_epochs,)``:

        - ``"flag_margin"``: top1 minus top2 probability (higher = more confident).
        - ``"flag_entropy"``: Shannon entropy in nats (higher = more uncertain).
        - ``"flag_maxprob"``: maximum class probability (higher = more confident).
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(
            f"Expected probabilities shape (n_epochs, n_classes), got {probs.shape}"
        )

    max_prob = np.max(probs, axis=-1)
    if probs.shape[-1] >= 2:
        # Partition is enough to get the two largest values per row.
        top2 = np.partition(probs, -2, axis=-1)[:, -2:]
        margin = top2[:, -1] - top2[:, -2]
    else:
        margin = max_prob

    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)

    return {
        "flag_margin": margin.astype(np.float32, copy=False),
        "flag_entropy": entropy.astype(np.float32, copy=False),
        "flag_maxprob": max_prob.astype(np.float32, copy=False),
    }


# Optional per-epoch uncertainty arrays that ``postprocess_logits``
# attaches to its result dict. ``FLAG_SCORE_KEYS`` are always present (computed
# free on the deterministic pass). ``score_recording`` propagates whichever of
# these are present.
FLAG_SCORE_KEYS: tuple[str, ...] = ("flag_margin", "flag_entropy", "flag_maxprob")
MC_SCORE_KEYS: tuple[str, ...] = (
    "mc_probability_variance",
    "mc_expected_entropy",
    "mc_predictive_entropy",
)
# These moments describe individual sampled predictions.
# With overlap they include augmentation/context variation, not only dropout.


def _validate_checkpoint_geometry(checkpoint: Mapping[str, Any]) -> int:
    """Reject metadata that conflicts with the fixed inference geometry."""
    config = checkpoint.get("model_config") or {}
    kwargs = config.get("model_kwargs") or {}
    encoder = kwargs.get("multirate_asymmetric_encoder_kwargs") or {}
    metadata = config.get("inference_metadata") or {}
    resampling = (checkpoint.get("preprocessing_config") or {}).get("resampling") or {}
    checks = (
        (kwargs, "fs", 128),
        (encoder, "fs", 128),
        (resampling, "target_fs", 128),
        (kwargs, "time_len", 3840),
        (encoder, "time_len", 3840),
        (kwargs, "context_epochs", 21),
        (kwargs, "context_half", 10),
        (metadata, "context_half", 10),
    )
    for values, key, expected in checks:
        if key in values and values[key] != expected:
            raise InferenceModelLoadError(
                f"SPECTRA requires fixed {key}={expected}; checkpoint declares {values[key]!r}"
            )
    return 3840


def _get_model_expected_channels(model: nn.Module) -> int | None:
    """Read the expected number of channels from the unwrapped model."""
    expected_channels = getattr(model, "expected_input_channels", None)
    if isinstance(expected_channels, (int, np.integer)):
        return int(expected_channels)

    in_ch = getattr(model, "in_ch", None)
    if isinstance(in_ch, (int, np.integer)):
        return int(in_ch)

    # Compatibility fallback for older model classes that expose channel count
    # only through the retired serialized channel-offset adapter.
    channel_emb = getattr(model, "channel_embedding", None)
    num_channels = getattr(channel_emb, "num_channels", None)
    if isinstance(num_channels, (int, np.integer)):
        return int(num_channels)

    return None


def load_edf(
    edf_path: str,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, float]]:
    """Read EDF channels using the same repair/fallback policy as preprocessing."""
    from spectra.preprocessing.edf import open_edf_reader

    with open_edf_reader(edf_path) as (reader, unusable):
        labels = list(reader.getSignalLabels())
        data = {
            name: reader.readSignal(i)
            for i, name in enumerate(labels)
            if i not in unusable
        }
        rates = {
            name: float(reader.getSampleFrequency(i))
            for i, name in enumerate(labels)
            if i not in unusable
        }
        return list(data), data, rates


def load_canonical_channels(canon_json: str) -> list[str]:
    """Load canonical channel list from JSON file.

    Args:
        canon_json: Path to JSON file with canonical channel list

    Returns:
        List of canonical channel names
    """
    with open(canon_json) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "channels" in data:
        return data["channels"]
    else:
        raise ValueError(
            "Canonical JSON must be a list of channel names or dict with 'channels' key"
        )


def align_channels(
    channel_data: dict[str, np.ndarray],
    sample_rates: dict[str, float],
    canonical_channels: list[str],
    enable_rereferencing: bool = True,
    enable_substitutions: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align recorded channels to canonical channel list.

    This function performs:
    1. Channel name normalization
    2. Channel substitution (e.g., A1/A2 <-> M1/M2)
    3. Channel rereferencing (algebraic derivation)
    4. Zero-padding for missing channels

    Args:
        channel_data: Dict of channel_name -> signal array
        sample_rates: Dict of channel_name -> sampling rate
        canonical_channels: List of desired canonical channel names
        enable_rereferencing: Enable algebraic channel derivation
        enable_substitutions: Enable channel label substitutions
        verbose: Print detailed channel mapping information

    Returns:
        Tuple of (aligned_data, presence_mask, channel_types)
        - aligned_data: Array of shape (n_canonical, n_samples) with zeros for missing
        - presence_mask: Binary array (n_canonical,) indicating present channels
        - channel_types: List of channel types for normalization
    """
    from spectra.data.channel.normalization import infer_channel_type
    from spectra.data.channel.substitutions import (
        build_substitution_map_with_rereferencing,
    )

    # Validate that all channels have the same length (required for rereferencing)
    channel_lengths = {name: sig.shape[0] for name, sig in channel_data.items()}
    if not channel_lengths:
        raise InferencePreprocessingError("No channel data provided to align_channels")

    unique_lengths = set(channel_lengths.values())
    if len(unique_lengths) > 1:
        # Channels have different lengths - this will cause crashes during rereferencing
        length_details = "\n".join(
            [
                f"  {name}: {length} samples"
                for name, length in sorted(channel_lengths.items())
            ]
        )
        raise InferencePreprocessingError(
            f"Channel length mismatch detected - all channels must have the same length for rereferencing.\n"
            f"Found {len(unique_lengths)} different lengths:\n{length_details}\n\n"
            f"This usually indicates that harmonize_channel_sample_rates() was not called before align_channels(), "
            f"or that the EDF file has corrupted channel data."
        )

    # Get the signal length (now validated to be the same for all channels)
    n_samples = next(iter(channel_data.values())).shape[0]
    n_canonical = len(canonical_channels)

    # Build channel mapping
    available_channels = list(channel_data.keys())

    if enable_substitutions or enable_rereferencing:
        mapping = build_substitution_map_with_rereferencing(
            canonical_channels,
            available_channels,
            enable_rereferencing=enable_rereferencing,
            verbose=verbose,
        )
    else:
        # Simple direct matching
        mapping = {}
        for canon in canonical_channels:
            if canon in channel_data:
                mapping[canon] = (canon, None, None)
            else:
                mapping[canon] = (None, f"Channel {canon} not available", None)

    # Allocate output arrays
    aligned_data = np.zeros((n_canonical, n_samples), dtype=np.float32)
    presence_mask = np.zeros(n_canonical, dtype=np.float32)
    channel_types = []

    # Fill aligned data
    for i, canon_ch in enumerate(canonical_channels):
        actual_ch, note, transform_row = mapping[canon_ch]

        if verbose:
            if note:
                logger.info(f"  {canon_ch}: {note}")
            elif actual_ch:
                logger.info(f"  {canon_ch}: {actual_ch} (direct)")

        # Infer channel type for normalization
        channel_types.append(infer_channel_type(canon_ch))

        if transform_row is not None:
            # Linear combination of channels (rereferencing)
            combined_signal = np.zeros(n_samples, dtype=np.float32)
            for j, avail_ch in enumerate(available_channels):
                coeff = transform_row[j]
                if coeff != 0:
                    source_signal = channel_data[avail_ch].astype(np.float32)
                    # Safety check: ensure signal has expected length
                    if source_signal.shape[0] != n_samples:
                        raise InferencePreprocessingError(
                            f"Channel length mismatch during rereferencing: "
                            f"channel '{avail_ch}' has {source_signal.shape[0]} samples "
                            f"but expected {n_samples} samples. "
                            f"This should not happen after harmonization."
                        )
                    combined_signal += coeff * source_signal

            aligned_data[i, :] = combined_signal
            presence_mask[i] = 1.0

        elif actual_ch is not None:
            # Direct match or substitution
            source_signal = channel_data[actual_ch].astype(np.float32)
            # Safety check: ensure signal has expected length
            if source_signal.shape[0] != n_samples:
                raise InferencePreprocessingError(
                    f"Channel length mismatch: channel '{actual_ch}' has {source_signal.shape[0]} samples "
                    f"but expected {n_samples} samples."
                )
            aligned_data[i, :] = source_signal
            presence_mask[i] = 1.0
        else:
            # Missing channel - leave as zeros
            pass

    return aligned_data, presence_mask, channel_types


def resample_signal(signal: np.ndarray, fs_orig: float, fs_target: float) -> np.ndarray:
    """Resample signal to target frequency using polyphase filtering.

    Args:
        signal: Input signal
        fs_orig: Original sampling frequency
        fs_target: Target sampling frequency

    Returns:
        Resampled signal
    """
    if abs(fs_orig - fs_target) < 0.1:
        return signal

    # Use polyphase filtering for high-quality resampling
    from fractions import Fraction

    if fs_orig <= 0 or fs_target <= 0:
        raise InferencePreprocessingError(
            f"Invalid sampling rates for resampling (fs_orig={fs_orig}, fs_target={fs_target})"
        )
    ratio = Fraction.from_float(float(fs_target) / float(fs_orig)).limit_denominator(
        512
    )
    up = ratio.numerator
    down = ratio.denominator

    return resample_poly(signal, up, down)


def harmonize_channel_sample_rates(
    channel_data: dict[str, np.ndarray],
    sample_rates: dict[str, float],
) -> tuple[dict[str, np.ndarray], float]:
    """Ensure all channels share a common sampling rate/length before alignment."""
    if not channel_data:
        raise InferencePreprocessingError("EDF file does not contain any channels.")

    # Gather sampling rates for available channels
    rates: list[float] = []
    for name in channel_data.keys():
        if name in sample_rates:
            rates.append(float(sample_rates[name]))
    if not rates:
        raise InferencePreprocessingError(
            "Failed to determine sampling rates for EDF channels."
        )

    rounded_unique = sorted({round(rate, 6) for rate in rates})

    def _trim_to_shortest(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        lengths = [sig.shape[0] for sig in data.values()]
        min_len = min(lengths)
        max_len = max(lengths)
        if min_len == 0:
            raise InferencePreprocessingError(
                "EDF channels contain zero-length recordings."
            )
        if min_len == max_len:
            return data
        logger.warning(
            "Channel sample counts differ (min=%d, max=%d). Truncating all to %d samples "
            "so alignment does not crash.",
            min_len,
            max_len,
            min_len,
        )
        return {name: sig[:min_len] for name, sig in data.items()}

    if len(rounded_unique) == 1:
        fs_value = float(rates[0])
        normalized = _trim_to_shortest(dict(channel_data))
        return normalized, fs_value

    target_fs = max(rates)
    logger.warning(
        "Detected mixed channel sample rates %s Hz. Resampling every channel to %.2f Hz "
        "to keep inference stable.",
        rounded_unique,
        target_fs,
    )
    normalized: dict[str, np.ndarray] = {}
    for name, signal in channel_data.items():
        orig_fs = float(sample_rates.get(name, target_fs))
        if abs(orig_fs - target_fs) > 1e-6:
            logger.info(
                "  - Resampling %s from %.2f Hz to %.2f Hz", name, orig_fs, target_fs
            )
            normalized[name] = resample_signal(signal, orig_fs, target_fs)
        else:
            normalized[name] = signal

    normalized = _trim_to_shortest(normalized)
    return normalized, target_fs


def create_epoch_batches(
    data: np.ndarray,
    presence_mask: np.ndarray,
    fs: int,
    epoch_sec: int,
    context_half: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create epoch batches with context windows.

    Args:
        data: Preprocessed data (n_channels, n_samples)
        presence_mask: Channel presence mask (n_channels,)
        fs: Sampling frequency
        epoch_sec: Epoch duration in seconds
        context_half: Context window half-width

    Returns:
        Tuple of (epoch_windows, presence_mask_tensor)
        - epoch_windows: Tensor of shape (n_epochs, context_len, n_channels, samples_per_epoch)
        - presence_mask_tensor: Tensor of shape (n_channels,)
    """
    if (fs, epoch_sec, context_half) != (128, 30, 10):
        raise ValueError(
            "SPECTRA requires 128 Hz, 30-second epochs, and context_half=10"
        )
    n_channels, n_samples = data.shape
    samples_per_epoch = fs * epoch_sec
    n_epochs = n_samples // samples_per_epoch

    # Truncate to complete epochs
    data = data[:, : n_epochs * samples_per_epoch]

    # Reshape to epochs: (n_channels, n_epochs, samples_per_epoch)
    data_epochs = data.reshape(n_channels, n_epochs, samples_per_epoch)
    # Transpose to (n_epochs, n_channels, samples_per_epoch)
    data_epochs = data_epochs.transpose(1, 0, 2)

    # Create context windows
    context_len = 2 * context_half + 1
    epoch_windows = []

    for i in range(n_epochs):
        # Get context range
        start_idx = max(0, i - context_half)
        end_idx = min(n_epochs, i + context_half + 1)

        # Extract window
        window = data_epochs[
            start_idx:end_idx
        ]  # (actual_len, n_channels, samples_per_epoch)

        # Pad if needed
        actual_len = window.shape[0]
        if actual_len < context_len:
            # Calculate how many epochs we're missing before and after
            missing_before = max(0, context_half - i)
            missing_after = max(0, (i + context_half + 1) - n_epochs)

            window = np.pad(
                window,
                ((missing_before, missing_after), (0, 0), (0, 0)),
                mode="constant",
            )

        epoch_windows.append(window)

    # Stack to (n_epochs, context_len, n_channels, samples_per_epoch)
    epoch_windows = np.stack(epoch_windows, axis=0)

    # Convert to tensors
    epoch_windows_tensor = torch.from_numpy(epoch_windows).float()
    presence_mask_tensor = torch.from_numpy(presence_mask).float()

    return epoch_windows_tensor, presence_mask_tensor


def _create_epoch_channel_mask_windows(
    epoch_channel_valid: np.ndarray,
    context_half: int,
    pad_mode: str = "constant",
) -> torch.Tensor:
    """Expand ``[N,C]`` epoch validity into centered ``[N,L,C]`` windows.

    Args:
        epoch_channel_valid: Per-epoch, per-channel validity shaped ``[N, C]``.
        context_half: Context window half-width; the output context length is
            ``2 * context_half + 1``.
        pad_mode: How to fill window positions that fall outside the recording.
            ``"constant"`` marks them invalid (zero-fill), matching the EDF
            pipeline's zero-padded waveform windows. ``"edge"`` replicates the
            nearest in-range row, matching callers that pad the waveform with
            ``np.pad(..., mode="edge")`` so a duplicated epoch keeps its true
            validity.

    Returns:
        Float tensor shaped ``[N, 2 * context_half + 1, C]``.

    Raises:
        ValueError: If the input is not 2-D or ``pad_mode`` is unsupported.
    """
    if pad_mode not in ("constant", "edge"):
        raise ValueError(f"pad_mode must be 'constant' or 'edge', got {pad_mode!r}")
    validity = np.asarray(epoch_channel_valid, dtype=bool)
    if validity.ndim != 2:
        raise ValueError(
            f"epoch_channel_valid must have shape [epochs, channels], got {validity.shape}"
        )
    n_epochs, channels = validity.shape
    context_len = 2 * context_half + 1
    windows = np.zeros((n_epochs, context_len, channels), dtype=np.float32)
    for center in range(n_epochs):
        source_start = max(0, center - context_half)
        source_end = min(n_epochs, center + context_half + 1)
        target_start = max(0, context_half - center)
        target_end = target_start + source_end - source_start
        windows[center, target_start:target_end] = validity[source_start:source_end]
        if pad_mode == "edge":
            windows[center, :target_start] = validity[source_start]
            windows[center, target_end:] = validity[source_end - 1]
    return torch.from_numpy(windows)


def create_epoch_batch_generator(
    data: np.ndarray,
    presence_mask: np.ndarray,
    fs: int,
    epoch_sec: int,
    context_half: int,
    batch_size: int,
    epoch_channel_valid: np.ndarray | None = None,
):
    """Generator that yields epoch batches sequentially to reduce memory usage.

    This generator creates context windows on-the-fly, processing only one batch
    at a time. This dramatically reduces memory usage compared to pre-allocating
    all epoch windows, especially beneficial for long recordings on systems with
    limited RAM (like macOS).

    Args:
        data: Preprocessed data (n_channels, n_samples)
        presence_mask: Channel presence mask (n_channels,)
        fs: Sampling frequency
        epoch_sec: Epoch duration in seconds
        context_half: Context window half-width
        batch_size: Number of epochs per batch

    Yields:
        Tuple of (batch_tensor, presence_mask_tensor) for each batch:
        - batch_tensor: Tensor of shape (batch_size, context_len, n_channels, samples_per_epoch)
        - presence_mask_tensor: Tensor of shape (n_channels,)
    """
    if (fs, epoch_sec, context_half) != (128, 30, 10):
        raise ValueError(
            "SPECTRA requires 128 Hz, 30-second epochs, and context_half=10"
        )
    n_channels, n_samples = data.shape
    samples_per_epoch = fs * epoch_sec
    n_epochs = n_samples // samples_per_epoch

    # Truncate to complete epochs
    data = data[:, : n_epochs * samples_per_epoch]

    # Reshape to epochs: (n_channels, n_epochs, samples_per_epoch)
    data_epochs = data.reshape(n_channels, n_epochs, samples_per_epoch)
    # Transpose to (n_epochs, n_channels, samples_per_epoch)
    data_epochs = data_epochs.transpose(1, 0, 2)

    context_len = 2 * context_half + 1
    presence_mask_tensor = torch.from_numpy(presence_mask).float()
    resolved_validity: np.ndarray | None = None
    if epoch_channel_valid is not None:
        resolved_validity = np.asarray(epoch_channel_valid, dtype=bool)
        if resolved_validity.shape != (n_epochs, n_channels):
            raise ValueError(
                "epoch_channel_valid must have shape "
                f"{(n_epochs, n_channels)}, got {resolved_validity.shape}"
            )

    # Process epochs in batches
    for batch_start in range(0, n_epochs, batch_size):
        batch_end = min(batch_start + batch_size, n_epochs)
        batch_windows = []
        batch_masks: list[np.ndarray] = []

        # Create context windows for this batch only
        for i in range(batch_start, batch_end):
            # Get context range
            start_idx = max(0, i - context_half)
            end_idx = min(n_epochs, i + context_half + 1)

            # Extract window
            window = data_epochs[
                start_idx:end_idx
            ]  # (actual_len, n_channels, samples_per_epoch)

            # Pad if needed
            actual_len = window.shape[0]
            missing_before = max(0, context_half - i)
            missing_after = max(0, (i + context_half + 1) - n_epochs)
            if actual_len < context_len:
                window = np.pad(
                    window,
                    ((missing_before, missing_after), (0, 0), (0, 0)),
                    mode="constant",  # Zero-fill to match training behavior
                )

            batch_windows.append(window)
            if resolved_validity is not None:
                mask_window = resolved_validity[start_idx:end_idx]
                if mask_window.shape[0] < context_len:
                    mask_window = np.pad(
                        mask_window,
                        ((missing_before, missing_after), (0, 0)),
                        mode="constant",
                    )
                batch_masks.append(mask_window)

        # Stack this batch and convert to tensor
        batch_array = np.stack(batch_windows, axis=0)
        batch_tensor = torch.from_numpy(batch_array).float()
        batch_presence = (
            torch.from_numpy(np.stack(batch_masks, axis=0)).float()
            if batch_masks
            else presence_mask_tensor
        )

        yield batch_tensor, batch_presence, batch_start, batch_end


def _strip_model_prefix(key: str) -> str:
    """Drop the optional ``model.`` wrapper prefix from a state-dict key."""
    return key[6:] if key.startswith("model.") else key


def _collect_enc_layer_indices(keys: list[str], marker: str) -> set[int]:
    """Return transformer encoder layer indices whose keys contain `marker`."""
    indices: set[int] = set()
    for key in keys:
        if marker not in key:
            continue
        parts = key.split(".")
        if key.startswith("enc_layers.") and len(parts) >= 2:
            try:
                indices.add(int(parts[1]))
            except ValueError:
                continue
        elif key.startswith("transformer.layers.") and len(parts) >= 3:
            try:
                indices.add(int(parts[2]))
            except ValueError:
                continue
    return indices


def _infer_transformer_num_layers_from_keys(keys: list[str]) -> int | None:
    """Infer TransformerContextNet layer count from MHA self-attention keys."""
    layer_indices = _collect_enc_layer_indices(keys, ".self_attn.in_proj_weight")
    if not layer_indices:
        return None
    return max(layer_indices) + 1


def _build_clean_key_map(
    state_dict: Mapping[str, Any],
) -> tuple[list[str], dict[str, str]]:
    """Return normalized checkpoint keys plus a map to original keys."""
    key_map: dict[str, str] = {}
    keys: list[str] = []
    for orig_key in state_dict.keys():
        if not isinstance(orig_key, str):
            continue
        clean_key = orig_key[6:] if orig_key.startswith("model.") else orig_key
        keys.append(clean_key)
        key_map[clean_key] = orig_key
    return keys, key_map


def _get_tensor_from_clean_key(
    state_dict: Mapping[str, Any],
    key_map: Mapping[str, str],
    clean_key: str,
) -> torch.Tensor | None:
    """Return a checkpoint tensor by its normalized key."""
    raw_key = key_map.get(clean_key)
    if raw_key is None:
        return None
    value = state_dict.get(raw_key)
    return value if isinstance(value, torch.Tensor) else None


def _extract_checkpoint_d_model(
    state_dict: Mapping[str, Any],
    *,
    keys: list[str] | None = None,
    key_map: Mapping[str, str] | None = None,
) -> int | None:
    """Infer and cross-check checkpoint d_model from inter-epoch tensors."""
    if keys is None or key_map is None:
        keys, key_map = _build_clean_key_map(state_dict)

    candidates: list[tuple[str, int]] = []

    def add(source: str, value: int | None) -> None:
        if value is None:
            return
        candidates.append((source, int(value)))

    proj = _get_tensor_from_clean_key(state_dict, key_map, "proj.weight")
    if proj is not None and proj.ndim == 2:
        add("proj.weight", int(proj.shape[0]))

    pos_encoding = _get_tensor_from_clean_key(state_dict, key_map, "pos_encoding")
    if pos_encoding is not None and pos_encoding.ndim == 3:
        add("pos_encoding", int(pos_encoding.shape[2]))

    output_norm = _get_tensor_from_clean_key(state_dict, key_map, "output_norm.weight")
    if output_norm is not None and output_norm.ndim == 1:
        add("output_norm.weight", int(output_norm.shape[0]))

    for attn_proj_key in (
        "enc_layers.0.self_attn.in_proj_weight",
        "transformer.layers.0.self_attn.in_proj_weight",
    ):
        attn_tensor = _get_tensor_from_clean_key(state_dict, key_map, attn_proj_key)
        if attn_tensor is not None and attn_tensor.ndim == 2:
            add(attn_proj_key, int(attn_tensor.shape[1]))
            break

    if not candidates:
        return None
    unique_values = sorted({value for _, value in candidates})
    if len(unique_values) > 1:
        detail = ", ".join(f"{source}={value}" for source, value in candidates)
        raise ValueError(
            f"Checkpoint has inconsistent inter transformer d_model values: {detail}"
        )

    return candidates[0][1]


def _infer_context_mode_metadata(
    state_dict: Mapping[str, Any],
    keys: list[str],
    key_map: Mapping[str, str],
) -> dict[str, Any]:
    """Recover relative-attention geometry, defaulting old states to legacy."""
    encoder_bias_key = next(
        (
            key
            for key in keys
            if key.endswith("relative_position_bias")
            and ("transformer.layers." in key or "enc_layers." in key)
            and "center_context_readout." not in key
        ),
        None,
    )
    readout_bias_key = next(
        (
            key
            for key in keys
            if key.endswith("center_context_readout.relative_position_bias")
        ),
        None,
    )
    swiglu_key = next(
        (
            key
            for key in keys
            if key.endswith("w_gate.weight")
            and ("transformer.layers." in key or "enc_layers." in key)
        ),
        None,
    )
    gelu_key = next(
        (
            key
            for key in keys
            if key.endswith("linear1.weight")
            and ("transformer.layers." in key or "enc_layers." in key)
        ),
        None,
    )
    metadata: dict[str, Any] = {
        "context_attention_mode": (
            "relative_full" if encoder_bias_key is not None else "legacy_absolute"
        ),
        "context_readout_mode": (
            "relative_multihead" if readout_bias_key is not None else "legacy_single"
        ),
    }
    if swiglu_key is not None:
        metadata["ffn_activation"] = "swiglu"
    elif gelu_key is not None:
        metadata["ffn_activation"] = "gelu"
    geometry_key = encoder_bias_key or readout_bias_key
    if geometry_key is not None:
        bias = state_dict[key_map[geometry_key]]
        if isinstance(bias, torch.Tensor) and bias.ndim == 2:
            lag_count = int(bias.shape[1])
            if lag_count > 0 and lag_count % 2 == 1:
                metadata["context_epochs"] = (lag_count + 1) // 2
            metadata["nhead"] = int(bias.shape[0])
    return metadata


def _reconcile_context_mode_metadata(
    model_kwargs: dict[str, Any],
    inferred_context: Mapping[str, Any],
) -> None:
    """Make relative-attention state geometry authoritative during loading.

    Relative attention and readout modes add learned parameters that cannot be
    loaded into their legacy counterparts.  Some early checkpoints containing
    those parameters carried missing or stale mode metadata, so the state-dict
    signatures must win whenever either side says a relative mode is in use.
    """
    relative_modes = {
        "context_attention_mode": "relative_full",
        "context_readout_mode": "relative_multihead",
    }
    state_uses_relative = False

    for key, relative_mode in relative_modes.items():
        inferred_value = inferred_context[key]
        configured_value = model_kwargs.get(key)
        should_reconcile = (
            inferred_value == relative_mode or configured_value == relative_mode
        )
        if should_reconcile:
            state_uses_relative = state_uses_relative or inferred_value == relative_mode
            if configured_value != inferred_value:
                logger.warning(
                    "Checkpoint %s mismatch: config=%r, state_dict=%r. "
                    "Using state-dict architecture so learned parameters load.",
                    key,
                    configured_value,
                    inferred_value,
                )
            model_kwargs[key] = inferred_value
        else:
            model_kwargs.setdefault(key, inferred_value)

    # Relative-position tables encode both the trained head count and maximum
    # context geometry.  Preserve those exact shapes even when config metadata
    # is stale, otherwise attention behavior can differ despite a successful
    # load of the ordinary projection weights.
    for key in ("context_epochs", "nhead"):
        inferred_value = inferred_context.get(key)
        if not state_uses_relative or inferred_value is None:
            continue
        configured_value = model_kwargs.get(key)
        if configured_value != inferred_value:
            logger.warning(
                "Checkpoint %s mismatch: config=%r, relative-bias state_dict=%r. "
                "Using state-dict geometry.",
                key,
                configured_value,
                inferred_value,
            )
        model_kwargs[key] = inferred_value

    inferred_ffn = inferred_context.get("ffn_activation")
    if inferred_ffn is not None:
        configured_ffn = model_kwargs.get("ffn_activation")
        if configured_ffn is not None and configured_ffn != inferred_ffn:
            logger.warning(
                "Checkpoint ffn_activation mismatch: config=%r, state_dict=%r. "
                "Using state-dict architecture so learned parameters load.",
                configured_ffn,
                inferred_ffn,
            )
        model_kwargs["ffn_activation"] = inferred_ffn
    else:
        model_kwargs.setdefault("ffn_activation", "gelu")


def _detect_classifier_head_family(keys: Iterable[str]) -> str | None:
    """Identify which ``TransformerContextNet`` classifier head a state dict holds.

    All three heads can carry ``classifier.norm.*`` or ``classifier.head.*``, so
    identification uses the markers unique to each:

    * residual MLP -- ``classifier.input_proj.0.*`` / ``classifier.res_linear1.*``
    * per-position -- ``classifier.mlp.0.*``
    * linear probe -- ``classifier.head.*`` + ``classifier.norm.*`` and neither
      of the above

    The rule is key-based rather than shape-based (the linear head's
    ``classifier.head.weight`` is ``[C, d_model]`` and the residual head's is
    ``[C, d_model // 2]``) so it can run before ``d_model`` has been recovered.

    Args:
        keys: State-dict keys, with or without a wrapper prefix.

    Returns:
        ``"residual_mlp"``, ``"per_position"``, ``"linear"``, or None when no
        classifier head is recognizable.
    """
    key_list = list(keys)
    if any(k.endswith("classifier.mlp.0.weight") for k in key_list) and any(
        k.endswith("classifier.norm.weight") for k in key_list
    ):
        return "per_position"
    if any(
        k.endswith("classifier.input_proj.0.weight")
        or k.endswith("classifier.res_linear1.weight")
        for k in key_list
    ):
        return "residual_mlp"
    if any(k.endswith("classifier.head.weight") for k in key_list) and any(
        k.endswith("classifier.norm.weight") for k in key_list
    ):
        return "linear"
    return None


def _validate_state_dict_load(
    load_result: torch.nn.modules.module._IncompatibleKeys,
    *,
    checkpoint_label: str,
    model_label: str,
    consumed_unexpected_keys: set[str] | None = None,
) -> None:
    """Ensure checkpoint weights fully match the instantiated model.

    Args:
        load_result: Result returned by ``load_state_dict``.
        checkpoint_label: Checkpoint path, for error messages.
        model_label: Model description, for error messages.
        consumed_unexpected_keys: Keys that ``load_state_dict`` reported as
            unexpected but that a later adapter actually applied. Reporting them
            as ignored would misrepresent what was loaded.
    """

    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    mismatched_keys = list(getattr(load_result, "mismatched_keys", []))

    if consumed_unexpected_keys:
        unexpected_keys = [
            key
            for key in unexpected_keys
            if _strip_model_prefix(key) not in consumed_unexpected_keys
        ]

    # Filter out safely-skippable missing keys
    # These are buffers that are computed at initialization, not learned weights:
    # - .num_batches_tracked: BatchNorm running statistics tracker
    safely_skippable_suffixes = (".num_batches_tracked",)

    def _is_safely_skippable(key: str) -> bool:
        return key.endswith(safely_skippable_suffixes) or key.endswith("._extra_state")

    critical_missing_keys = [k for k in missing_keys if not _is_safely_skippable(k)]
    skipped_keys = [k for k in missing_keys if _is_safely_skippable(k)]

    if skipped_keys:
        logger.info(
            f"Skipping {len(skipped_keys)} non-critical missing keys (computed at init or training-only): "
            f"{', '.join(skipped_keys[:5])}"
            + (f" ... (+{len(skipped_keys) - 5} more)" if len(skipped_keys) > 5 else "")
        )

    missing_keys = critical_missing_keys

    if not missing_keys and not unexpected_keys and not mismatched_keys:
        return

    def _summarize(keys: list[str]) -> str:
        if not keys:
            return "None"
        preview = keys[:5]
        suffix = "" if len(keys) <= 5 else f" … (+{len(keys) - 5} more)"
        return ", ".join(preview) + suffix

    details = [
        (
            f"Failed to load checkpoint '{checkpoint_label}' into {model_label} "
            "due to key mismatch."
        ),
        (
            "This usually happens when the saved model architecture does not match "
            "the current model configuration (e.g., different channel count or "
            "transformer geometry)."
        ),
    ]

    if missing_keys or mismatched_keys:
        if missing_keys:
            details.append(
                f"Missing keys ({len(missing_keys)}): {_summarize(missing_keys)}"
            )
        if mismatched_keys:
            details.append(
                f"Mismatched keys ({len(mismatched_keys)}): "
                f"{_summarize(mismatched_keys)}"
            )
        raise InferenceModelLoadError("\n".join(details))

    if unexpected_keys:
        raise InferenceModelLoadError(
            f"Checkpoint '{checkpoint_label}' contains unexpected model parameters: "
            f"{_summarize(unexpected_keys)}"
        )


def _infer_pe_mode_in_state_dict(state_dict: Mapping[str, Any]) -> str:
    """Infer positional encoding mode from checkpoint state dict keys."""
    has_learnable_pe = any(
        isinstance(k, str) and (k == "pos_encoding" or k.endswith(".pos_encoding"))
        for k in state_dict.keys()
    )
    has_sinusoidal_pe = any(
        isinstance(k, str) and (k == "pos.pe" or k.endswith(".pos.pe"))
        for k in state_dict.keys()
    )

    if has_learnable_pe:
        return "learnable"
    if has_sinusoidal_pe:
        return "sinusoidal"
    return "none"


def _extract_learnable_pe_tensor(
    state_dict: Mapping[str, Any],
) -> torch.Tensor | None:
    """Return learnable positional encoding tensor from state dict, if present."""
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        if key == "pos_encoding" or key.endswith(".pos_encoding"):
            if torch.is_tensor(value):
                return value
    return None


def _verify_loaded_learnable_pe(
    model: nn.Module,
    checkpoint_pe: torch.Tensor | None,
    *,
    checkpoint_label: str,
) -> bool:
    """Verify that checkpoint learnable PE was loaded exactly into model weights."""
    if checkpoint_pe is None:
        return False

    base_model = model
    maybe_inner_model = getattr(model, "model", None)
    if isinstance(maybe_inner_model, nn.Module):
        base_model = maybe_inner_model

    model_pe = getattr(base_model, "pos_encoding", None)
    if not torch.is_tensor(model_pe):
        raise InferenceModelLoadError(
            "Checkpoint contains learnable 'pos_encoding' but instantiated model has no "
            f"learnable PE parameter. Checkpoint '{checkpoint_label}' was not fully loaded."
        )

    model_pe_cpu = cast(torch.Tensor, model_pe).detach().cpu()
    ckpt_pe_cpu = checkpoint_pe.detach().cpu()
    if tuple(model_pe_cpu.shape) != tuple(ckpt_pe_cpu.shape):
        raise InferenceModelLoadError(
            "Learnable PE shape mismatch after load: "
            f"model={tuple(model_pe_cpu.shape)} vs checkpoint={tuple(ckpt_pe_cpu.shape)} "
            f"for checkpoint '{checkpoint_label}'."
        )

    compare_model = (
        model_pe_cpu.to(dtype=ckpt_pe_cpu.dtype)
        if model_pe_cpu.dtype != ckpt_pe_cpu.dtype
        else model_pe_cpu
    )
    if not torch.equal(compare_model, ckpt_pe_cpu):
        max_abs_diff = float((compare_model - ckpt_pe_cpu).abs().max().item())
        raise InferenceModelLoadError(
            "Learnable PE values do not match checkpoint after load "
            f"(max_abs_diff={max_abs_diff:.6g}) for checkpoint '{checkpoint_label}'."
        )

    return True


def _infer_multirate_structure(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Recover a ``multirate_asymmetric`` encoder's structure from weight shapes.

    Used only when a checkpoint carries no ``model_config``. Everything returned
    is unambiguous from a tensor shape (widths, branch widths and filter counts,
    kernel sizes, pooling head geometry). Options that leave no trace in the
    weights -- dilation *values*, stride schedule, decimation factor, low-pass
    cutoffs, sinc kernel length -- fall back to the encoder defaults, which is
    why ``extract_transformer_config`` persists the whole ``get_config()`` for
    this variant.

    Returns:
        Constructor kwargs plus a ``"widths"`` entry the caller routes to
        ``cnn_widths``. Empty when the stem cannot be read.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not hasattr(value, "shape"):
            continue
        marker = key.split("epoch_encoder.", 1)
        if len(marker) != 2:
            continue
        shapes[marker[1]] = tuple(int(dim) for dim in value.shape)

    stem_fusion = shapes.get("multirate_stem.fusion.0.weight")
    trunk_fusions = [shapes.get(f"trunk.{i}.0.fusion.0.weight") for i in range(4)]
    if stem_fusion is None or any(shape is None for shape in trunk_fusions):
        return {}
    widths = (stem_fusion[0], *[cast(tuple[int, ...], f)[0] for f in trunk_fusions])
    inferred: dict[str, Any] = {"widths": widths}

    def _count(prefix: str, suffix: str) -> int:
        idx = 0
        while f"{prefix}{idx}{suffix}" in shapes:
            idx += 1
        return idx

    # --- stem branches -----------------------------------------------------
    eeg_short = shapes.get("multirate_stem.eeg_branch.short.fusion.0.weight")
    eeg_band = shapes.get("multirate_stem.eeg_branch.band_project.0.weight")
    eeg_first = shapes.get(
        "multirate_stem.eeg_branch.short.dilated_branches.0.0.weight"
    )
    eog_mix = shapes.get("multirate_stem.eog_branch.mix.2.weight")
    eog_first = shapes.get("multirate_stem.eog_branch.dilated.0.weight")
    emg_proj = shapes.get("multirate_stem.emg_branch.project.0.weight")
    emg_filters = shapes.get("multirate_stem.emg_branch.filters.weight")

    eeg_out = (eeg_short[0] if eeg_short else 0) + (eeg_band[0] if eeg_band else 0)
    eog_out = eog_mix[0] if eog_mix else 0
    emg_out = emg_proj[0] if emg_proj else 0
    total = eeg_out + eog_out + emg_out
    if total:
        inferred["modality_split"] = (
            eeg_out / total,
            eog_out / total,
            emg_out / total,
        )
    n_eeg = eeg_first[1] if eeg_first else 0
    n_eog = eog_first[1] if eog_first else 0
    n_emg = emg_filters[1] if emg_filters else 0
    cursor = 0
    inferred["eeg_indices"] = tuple(range(cursor, cursor + n_eeg))
    cursor += n_eeg
    inferred["eog_indices"] = tuple(range(cursor, cursor + n_eog))
    cursor += n_eog
    inferred["emg_indices"] = tuple(range(cursor, cursor + n_emg))
    if eeg_first is not None:
        inferred["stem_kernel_size"] = eeg_first[2]
        n_short = _count(
            "multirate_stem.eeg_branch.short.dilated_branches.", ".0.weight"
        )
        if n_short:
            inferred["eeg_short_dilations"] = tuple(2**i for i in range(n_short))
    if eeg_out:
        band_out = eeg_band[0] if eeg_band else 0
        inferred["eeg_band_fraction"] = band_out / eeg_out
    band_low = shapes.get("multirate_stem.eeg_branch.band_filters.raw_low_hz")
    if band_low is not None:
        inferred["eeg_band_filters"] = band_low[0]
    if eog_mix is not None:
        inferred["eog_kernel"] = eog_mix[2]
        n_dil = _count("multirate_stem.eog_branch.dilated.", ".weight")
        if n_dil:
            inferred["eog_dilations"] = tuple(2**i for i in range(n_dil))
    if emg_filters is not None:
        inferred["emg_filters"] = emg_filters[0]
        inferred["emg_kernel"] = emg_filters[2]

    # --- trunk -------------------------------------------------------------
    trunk_first = shapes.get("trunk.0.0.branches.0.0.weight")
    if trunk_first is not None:
        inferred["trunk_kernel_size"] = trunk_first[2]
        dilations = []
        for stage in range(4):
            n_branches = _count(f"trunk.{stage}.0.branches.", ".0.weight")
            base = (1, 2, 4) if stage == 0 else (1, 4, 8)
            if n_branches == len(base):
                dilations.append(base)
            else:
                dilations.append(tuple(2**i for i in range(max(n_branches, 1))))
        inferred["trunk_dilations"] = tuple(dilations)

    # --- pooling -----------------------------------------------------------
    inferred["pool_mfa"] = "aggregate.mix.0.weight" in shapes
    project = shapes.get("pool.project.weight")
    if "pool.latent_queries" in shapes:
        inferred["pooling_mode"] = "learned"
    else:
        inferred["pooling_mode"] = "attentive_stats"
        inferred["pool_occupancy"] = "pool.tau" in shapes
        attn_in = shapes.get("pool.attention.0.weight")
        attn_out = shapes.get("pool.attention.3.weight")
        if attn_in is not None:
            inferred["pool_bottleneck"] = attn_in[0]
        if attn_out is not None and widths[4]:
            inferred["pool_heads"] = max(1, attn_out[0] // widths[4])
    if project is not None and project[0] != widths[4]:
        inferred["pool_out_dim"] = project[0]
    return inferred


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    checkpoint_data: dict | None = None,
    *,
    prefer_averaged: bool = False,
    options: ScoreOptions | None = None,
    channel_names: list[str] | None = None,
) -> tuple[nn.Module, dict]:
    """Reconstruct the supported CNN + transformer from saved inference metadata."""
    from spectra.models import TransformerContextNet
    from spectra.utils.checkpoint import (
        normalize_state_dict_keys,
        restore_fixed_filter_buffers,
    )

    checkpoint = (
        dict(checkpoint_data)
        if checkpoint_data is not None
        else torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    _validate_checkpoint_geometry(checkpoint)
    model_config = dict(checkpoint.get("model_config") or {})
    model_kwargs = dict(model_config.get("model_kwargs") or {})
    if (
        model_config.get("model_type", "TransformerContextNet")
        != "TransformerContextNet"
        or model_kwargs.get("epoch_encoder_variant") != "multirate_asymmetric"
    ):
        raise InferenceModelLoadError(
            "SPECTRA supports only multirate_asymmetric CNN + TransformerContextNet "
            "checkpoints with explicit model_config metadata."
        )
    encoder_kwargs = dict(model_kwargs.get("multirate_asymmetric_encoder_kwargs") or {})
    if model_kwargs.get("recording_conditioning") or encoder_kwargs.get(
        "recording_conditioning"
    ):
        raise InferenceModelLoadError(
            "Recording-conditioned checkpoints are unsupported; supply an unconditioned "
            "multirate_asymmetric checkpoint."
        )
    # Older unconditioned exports included these inactive encoder defaults.
    for key in (
        "recording_conditioning",
        "recording_conditioning_samples",
        "recording_conditioning_dim",
    ):
        encoder_kwargs.pop(key, None)
    model_kwargs["multirate_asymmetric_encoder_kwargs"] = encoder_kwargs
    excluded_flags = (
        "use_feature_extraction",
        "use_engineered_features",
        "use_sleepfm_fusion",
        "use_n1_attention",
        "cnn_feature_guidance",
        "use_patch_transformer",
        "use_patch_tokens",
        "use_patch_features",
        "use_patch_cross_attention",
    )
    if (
        any(model_kwargs.get(name) for name in excluded_flags)
        or model_kwargs.get("epoch_tokenization", "epoch") != "epoch"
    ):
        raise InferenceModelLoadError(
            "Feature fusion, engineered/YASA features, and patch models are not "
            "supported; supply a waveform-only multirate_asymmetric checkpoint."
        )

    state_dict = checkpoint.get(
        "model", checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
    )
    if prefer_averaged:
        averaged = checkpoint.get("ema", {})
        if isinstance(averaged, Mapping) and isinstance(
            averaged.get("module"), Mapping
        ):
            state_dict = averaged["module"]
        else:
            logger.info("No averaged weights found; using model weights")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise InferenceModelLoadError("Checkpoint contains no model state dictionary")
    state_dict = dict(normalize_state_dict_keys(state_dict))
    removed_state = [
        key
        for key in state_dict
        if _strip_model_prefix(key).startswith("preprocessor.")
        or "recording_conditioner." in key
    ]
    if removed_state:
        raise InferenceModelLoadError(
            "Recording conditioning and embedded model preprocessing are unsupported: "
            f"{removed_state[:5]}. Preprocess EDFs externally and use an unwrapped model."
        )
    excluded_prefixes = (
        "feature_extractor.",
        "feat_projection.",
        "sequential_fusion.",
        "epoch_feature_fusion.",
        "sleepfm_fusion.",
        "n1_feature_extractor.",
        "n1_attention.",
        "cross_attn_fusion_layers.",
        "eng_token_projection.",
        "epoch_encoder.cnn_extractor.",
        "epoch_encoder.sleep_features.",
        "axial_backbone.",
        "axial_v2_",
        "epoch_grid.",
        "patch_frontend.",
        "patch_tokenizer.",
        "patch_feature_extractor.",
        "center_patch_readout.",
    )
    unsupported = [
        key
        for key in state_dict
        if _strip_model_prefix(key).startswith(excluded_prefixes)
    ]
    if unsupported:
        raise InferenceModelLoadError(
            f"Checkpoint contains unsupported feature fusion or other model state: {unsupported[:5]}"
        )
    # These parameters never participate in inference. Keep only an audited
    # allowlist; unknown learned state still fails checkpoint validation.
    auxiliary_prefixes = (
        "source_convention.",
        "slow_wave_occupancy_head.",
        "neighbor_head.",
        "transition_head.",
        "cnn_reconstruction_head.",
        "eng_reconstruction_head.",
    )
    ignored = [
        key
        for key in state_dict
        if _strip_model_prefix(key).startswith(auxiliary_prefixes)
        or _strip_model_prefix(key) == "mask_token"
    ]
    state_dict = {key: value for key, value in state_dict.items() if key not in ignored}
    state_dict, _ = _strip_legacy_router_context_keys(state_dict)
    keys = [_strip_model_prefix(key) for key in state_dict]
    key_map = {_strip_model_prefix(key): key for key in state_dict}
    if not any(key.startswith("epoch_encoder.multirate_stem.") for key in keys):
        raise InferenceModelLoadError(
            "Checkpoint is missing the multirate asymmetric encoder"
        )
    _reconcile_context_mode_metadata(
        model_kwargs, _infer_context_mode_metadata(state_dict, keys, key_map)
    )
    if model_kwargs.get("recurrent_refinement_steps", 0) or any(
        key.startswith("recurrent_refiner.") for key in keys
    ):
        raise InferenceModelLoadError(
            "Recurrent refinement checkpoints are unsupported"
        )
    model_kwargs.pop("recurrent_refinement_steps", None)
    d_model = _extract_checkpoint_d_model(state_dict)
    if d_model is not None:
        model_kwargs["d_model"] = d_model
    num_layers = _infer_transformer_num_layers_from_keys(keys)
    if num_layers is not None:
        model_kwargs["num_layers"] = num_layers
    head_family = _detect_classifier_head_family(keys)
    if head_family == "per_position":
        model_kwargs["use_per_position_head"] = True
        model_kwargs["classifier_head"] = "residual_mlp"
        model_kwargs["head_use_local_mix"] = "classifier.local.weight" in key_map
        model_kwargs["head_norm"] = (
            "layernorm" if "classifier.norm.bias" in key_map else "rmsnorm"
        )
        local = _get_tensor_from_clean_key(
            state_dict, key_map, "classifier.local.weight"
        )
        if local is not None:
            model_kwargs["head_local_kernel"] = int(local.shape[-1])
    elif head_family in {"linear", "residual_mlp"}:
        model_kwargs["classifier_head"] = head_family
        model_kwargs["use_per_position_head"] = False
    model_kwargs["learnable_temperature"] = "log_temperature" in key_map
    model_kwargs["use_confidence_head"] = any(
        key.startswith("confidence_head.") for key in keys
    )
    confidence_weight = _get_tensor_from_clean_key(
        state_dict, key_map, "confidence_head.net.1.weight"
    )
    if confidence_weight is not None:
        model_kwargs["confidence_head_hidden_dim"] = int(confidence_weight.shape[0])
    inferred_encoder = _infer_multirate_structure(state_dict)
    widths = inferred_encoder.pop("widths", None)
    if widths is not None:
        model_kwargs.setdefault("cnn_widths", widths)
    inferred_encoder.update(
        model_kwargs.get("multirate_asymmetric_encoder_kwargs") or {}
    )
    model_kwargs["multirate_asymmetric_encoder_kwargs"] = inferred_encoder
    model_kwargs.update(time_len=3840, fs=128, context_epochs=21)
    if channel_names is not None:
        model_kwargs["channel_names"] = channel_names
    signature = inspect.signature(TransformerContextNet.__init__)
    constructor_kwargs = {
        key: value for key, value in model_kwargs.items() if key in signature.parameters
    }
    model = TransformerContextNet(**constructor_kwargs)
    state_dict = {_strip_model_prefix(key): value for key, value in state_dict.items()}
    pe_mode = _infer_pe_mode_in_state_dict(state_dict)
    pe_tensor = _extract_learnable_pe_tensor(state_dict)
    try:
        load_result = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise InferenceModelLoadError(f"Checkpoint tensor mismatch: {exc}") from exc
    restored = restore_fixed_filter_buffers(model, state_dict)
    _validate_state_dict_load(
        load_result,
        checkpoint_label=str(checkpoint_path),
        model_label="TransformerContextNet",
        consumed_unexpected_keys={
            _strip_model_prefix(f"{name}.kernel") for name in restored
        },
    )
    pe_verified = False
    if pe_mode == "learnable":
        pe_verified = _verify_loaded_learnable_pe(
            model, pe_tensor, checkpoint_label=str(checkpoint_path)
        )
    model_config["model_kwargs"] = {
        **constructor_kwargs,
        **{key: model_kwargs[key] for key in ("context_half",) if key in model_kwargs},
    }
    checkpoint["model_config"] = model_config
    checkpoint["_inference_load_audit"] = {
        "missing_keys_count": len(load_result.missing_keys),
        "unexpected_keys_count": len(load_result.unexpected_keys),
        "mismatched_keys_count": 0,
        "pe_mode_in_state": pe_mode,
        "pe_verified": pe_verified,
        "ignored_training_keys": ignored,
    }
    model.to(device)
    model.eval()
    return model, checkpoint


def _finalize_overlap_average(
    overlap_sum: torch.Tensor,
    overlap_count: torch.Tensor,
    overlap_center: torch.Tensor,
) -> np.ndarray:
    """Divide an overlap accumulator, falling back to centre-window values.

    An epoch legitimately receives zero contributions when every window that
    covers it was excluded -- a run of unlabeled centres longer than the context
    window under ``center_valid_mask``, or a stretch with no present channel.
    Dividing by a clamped count would hand those epochs an all-zero logit row
    whose ``argmax`` is class 0 (Wake), which is indistinguishable from a real
    Wake prediction. Fall back to the epoch's own centre-window logits instead.

    Args:
        overlap_sum: ``[n_epochs, width]`` sums (logits, probabilities, or moments).
        overlap_count: ``[n_epochs]`` contribution counts.
        overlap_center: ``[n_epochs, width]`` centre-window values in the same space.

    Returns:
        ``[n_epochs, n_classes]`` averaged logits as a numpy array.
    """
    covered = overlap_count > 0
    averaged = overlap_sum / overlap_count.clamp_min(1.0).unsqueeze(1)
    resolved = torch.where(covered.unsqueeze(1), averaged, overlap_center)
    n_uncovered = int((~covered).sum().item())
    if n_uncovered:
        logger.info(
            "Overlap-averaging: %d/%d epochs had no valid overlapping window; "
            "used their own center-window logits instead.",
            n_uncovered,
            int(overlap_count.numel()),
        )
    return resolved.cpu().numpy()


def _build_epoch_indices(
    start_idx: int,
    end_idx: int,
    n_epochs: int,
    context_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute epoch_indices and recording_length for an inference batch.

    Used by the overlap-averaging path to scatter per-position predictions back
    into absolute epoch positions.

    Args:
        start_idx: First center-epoch index in this batch.
        end_idx: One past the last center-epoch index.
        n_epochs: Total epochs in the recording.
        context_len: Context window length (2 * context_half + 1).
        device: Target device.

    Returns:
        epoch_indices  [batch_size, context_len]  absolute epoch positions
        recording_length  [batch_size]  total epochs (constant across batch)
    """
    half = context_len // 2
    centers = torch.arange(start_idx, end_idx, device=device, dtype=torch.long)
    offsets = torch.arange(context_len, device=device, dtype=torch.long) - half
    epoch_indices = centers.unsqueeze(1) + offsets.unsqueeze(0)
    epoch_indices = epoch_indices.clamp(min=0, max=max(0, n_epochs - 1))
    recording_length = torch.full(
        (end_idx - start_idx,), n_epochs, device=device, dtype=torch.long
    )
    return epoch_indices, recording_length


def _build_epoch_valid_mask(
    start_idx: int,
    end_idx: int,
    n_epochs: int,
    context_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``[B,L]`` validity for zero-padded context epochs."""
    half = context_len // 2
    centers = torch.arange(start_idx, end_idx, device=device, dtype=torch.long)
    offsets = torch.arange(context_len, device=device, dtype=torch.long) - half
    positions = centers.unsqueeze(1) + offsets.unsqueeze(0)
    return (positions >= 0) & (positions < n_epochs)


_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


def _validate_mc_options(samples: int, rate: float | None) -> None:
    if (
        isinstance(samples, bool)
        or not isinstance(samples, (int, np.integer))
        or samples < 1
    ):
        raise ValueError("mc_samples must be a positive integer")
    if rate is not None and not (np.isfinite(rate) and 0.0 <= rate < 1.0):
        raise ValueError("mc_dropout_rate must be finite and in [0, 1)")


def _mc_component(name: str) -> str:
    parts = name.split(".")
    if "classifier" in parts:
        return "classifier"
    if any(part in {"epoch_encoder", "cnn", "encoder"} for part in parts):
        return "encoder"
    return "transformer"


@dataclass
class MCDropoutState:
    """Reversible MC activation, including effective per-module coverage.

    Attributes:
        dropout_layers: Number of selected dropout modules, including zero rates.
        attention_layers: Number of selected functional attention modules.
        sites: Qualified name, component, original rate, and effective rate.
    """

    dropout_layers: int = 0
    attention_layers: int = 0
    sites: list[tuple[str, str, float, float]] = field(default_factory=list)
    _saved: list[tuple[nn.Module, bool, str | None, float | None]] = field(
        default_factory=list
    )

    def restore(self) -> None:
        """Restore flags nonrecursively and surface failed rate restoration."""
        errors: list[Exception] = []
        for mod, was_training, rate_attr, saved_rate in reversed(self._saved):
            mod.training = was_training
            if rate_attr is not None and saved_rate is not None:
                try:
                    setattr(mod, rate_attr, saved_rate)
                except Exception as exc:
                    errors.append(exc)
        self._saved.clear()
        if errors:
            raise RuntimeError("Could not restore an MC dropout rate") from errors[0]


def _enable_mc_dropout(
    m: nn.Module,
    p_override: float | None = None,
    *,
    include_attention: bool = False,
    module_pattern: str | None = None,
) -> MCDropoutState:
    """Activate selected trained dropout sites without recursively training modules.

    Args:
        m: Model already in evaluation mode.
        p_override: Optional rate for existing sites, including zero-rate controls.
        include_attention: Include standard and repository functional attention.
        module_pattern: Regex on qualified names in the unwrapped model.

    Returns:
        State to restore after sampling. Setup failures restore it automatically.
    """
    _validate_mc_options(1, p_override)
    state = MCDropoutState()
    matcher = re.compile(module_pattern) if module_pattern else None
    attention_types = (nn.MultiheadAttention,)
    modules = dict(m.named_modules())
    selected_names: list[str] = []
    try:
        for name, mod in modules.items():
            is_dropout = isinstance(mod, _DROPOUT_TYPES)
            is_attention = include_attention and isinstance(mod, attention_types)
            if not (is_dropout or is_attention):
                continue
            if matcher is not None and not matcher.search(name):
                continue
            rate_attr = "p" if is_dropout else "dropout"
            saved_rate = float(getattr(mod, rate_attr))
            effective_rate = saved_rate if p_override is None else float(p_override)
            state._saved.append(
                (
                    mod,
                    mod.training,
                    rate_attr if p_override is not None else None,
                    saved_rate,
                )
            )
            # Never recurse: MHA can have children, and norm modes must stay fixed.
            mod.training = True
            if p_override is not None:
                setattr(mod, rate_attr, effective_rate)
            state.sites.append((name, _mc_component(name), saved_rate, effective_rate))
            selected_names.append(name)
            if is_dropout:
                state.dropout_layers += 1
            else:
                state.attention_layers += 1

        for name, mod in modules.items():
            # Only the standard inherited forward can bypass its dropout children.
            # Relative/grid subclasses already use explicit evaluation forwards.
            if not isinstance(mod, nn.TransformerEncoderLayer):
                continue
            if type(mod).forward is not nn.TransformerEncoderLayer.forward:
                continue
            prefix = name + "." if name else ""
            if any(site.startswith(prefix) for site in selected_names):
                state._saved.append((mod, mod.training, None, None))
                mod.training = True
    except Exception:
        state.restore()
        raise
    return state


def _configure_mc_dropout(
    model: nn.Module, options: ScoreOptions
) -> tuple[int, bool, MCDropoutState | None]:
    """Activate MC sites and report effective coverage in all model components."""
    _validate_mc_options(options.mc_samples, options.mc_dropout_rate)
    if not options.use_mc_dropout:
        return 1, False, None
    state = _enable_mc_dropout(
        model,
        options.mc_dropout_rate,
        include_attention=bool(options.mc_include_attention),
        module_pattern=options.mc_module_pattern,
    )
    try:
        if not state.sites:
            raise ValueError(
                "MC dropout was requested but no module was activated "
                f"(mc_module_pattern={options.mc_module_pattern!r})"
            )
        for component in ("encoder", "transformer", "classifier"):
            sites = [s for s in state.sites if s[1] == component]
            nonzero = sum(rate > 0 for _, _, _, rate in sites)
            logger.info(
                "MC dropout %s: selected=%d, nonzero=%d",
                component,
                len(sites),
                nonzero,
            )
            if not nonzero:
                logger.warning(
                    "MC dropout %s has no selected nonzero dropout; "
                    "no new dropout is inserted.",
                    component,
                )
        for name, component, original, effective in state.sites:
            logger.info(
                "MC site %s (%s): trained_p=%.4f, effective_p=%.4f",
                name or "<root>",
                component,
                original,
                effective,
            )
        logger.info(
            "MC dropout: samples=%d, pooling=%s, aggregation=%s. "
            "Diagnostics include overlapping-context variation when enabled.",
            options.mc_samples,
            options.mc_pooling,
            options.mc_aggregation,
        )
    except Exception:
        state.restore()
        raise
    # In the consistent path probabilities must survive through overlap pooling.
    prob_pool = options.mc_pooling == "prob" and (
        options.mc_samples > 1
        or (options.mc_aggregation == "consistent" and options.overlap_average)
    )
    return options.mc_samples, prob_pool, state


@contextmanager
def _mc_dropout_session(
    model: nn.Module, options: ScoreOptions
) -> Iterator[tuple[int, bool]]:
    state: MCDropoutState | None = None
    try:
        samples, prob_pool, state = _configure_mc_dropout(model, options)
        yield samples, prob_pool
    finally:
        if state is not None:
            state.restore()


def _extract_inference_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        return cast(torch.Tensor, outputs["logits"])
    if hasattr(outputs, "logits"):
        return cast(torch.Tensor, outputs.logits)
    return cast(torch.Tensor, outputs)


@dataclass
class _SamplingMoments:
    """Online moments of individual forward posteriors, never pooled logits."""

    total: torch.Tensor | None = None
    count: int = 0

    def add(self, logits: torch.Tensor) -> None:
        probs = logits.float().softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(
            dim=-1, keepdim=True
        )
        packed = torch.cat((probs, probs.square(), entropy), dim=-1)
        if self.total is None:
            self.total = packed
        else:
            self.total.add_(packed)
        self.count += 1

    def mean(self) -> torch.Tensor:
        if self.total is None or self.count == 0:
            raise RuntimeError("No sampling moments were collected")
        return self.total / self.count


def _sample_inference_batch(
    model_input: dict[str, Any],
    options: ScoreOptions,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    mc_samples: int,
    prob_pool: bool,
    forward: Callable[[dict[str, Any]], Any],
    *,
    collect_diagnostics: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample fresh full-model forwards while reusing invariant batch inputs."""
    from spectra.utils.device import autocast_context

    moments = _SamplingMoments() if collect_diagnostics else None
    consistent_prob = prob_pool and options.mc_aggregation == "consistent"
    accum: torch.Tensor | None = None
    for _ in range(mc_samples):
        with autocast_context(device, dtype=amp_dtype):
            logits = _extract_inference_logits(forward(model_input)).float()
            if moments is not None:
                moments.add(logits)
        value = logits.float()
        if prob_pool:
            value = value.softmax(dim=-1)
        # Own the first output: a compiled forward can reuse its output storage.
        if accum is None:
            accum = value.clone()
        else:
            accum.add_(value)
    if accum is None:
        raise RuntimeError("Inference produced no logits")
    batch_out = accum / mc_samples
    if prob_pool and not consistent_prob:
        batch_out = batch_out.clamp_min(torch.finfo(batch_out.dtype).tiny).log()
    return batch_out, moments.mean() if moments is not None else None


@dataclass
class _InferenceAccumulator:
    """Shared masked overlap pooling for predictions and sampling moments."""

    n_epochs: int
    overlap: bool
    probability_space: bool
    stream_to_cpu: bool = False
    chunks: list[torch.Tensor] = field(default_factory=list)
    total: torch.Tensor | None = None
    count: torch.Tensor | None = None
    center: torch.Tensor | None = None
    n_classes: int = 0
    has_moments: bool = False

    def add(
        self,
        prediction: torch.Tensor,
        moments: torch.Tensor | None,
        start: int,
        end: int,
        channel_mask: torch.Tensor,
        center_valid: torch.Tensor | None,
    ) -> None:
        self.n_classes = prediction.shape[-1]
        self.has_moments = moments is not None
        packed = (
            torch.cat((prediction, moments), dim=-1)
            if moments is not None
            else prediction
        )
        if not self.overlap:
            self.chunks.append(packed.cpu() if self.stream_to_cpu else packed)
            return
        if prediction.ndim != 3:
            raise InferencePreprocessingError(
                "overlap_average=True requires a predict_all-capable model "
                f"returning [B, L, C]; got shape {tuple(prediction.shape)}"
            )
        length = prediction.shape[1]
        device = prediction.device
        indices, _ = _build_epoch_indices(start, end, self.n_epochs, length, device)
        valid = _build_epoch_valid_mask(start, end, self.n_epochs, length, device)
        if channel_mask.ndim == 3:
            valid &= channel_mask.bool().any(dim=-1)
        if center_valid is not None:
            valid &= center_valid[start:end].unsqueeze(1)
        width = packed.shape[-1]
        if self.total is None:
            self.total = torch.zeros(
                self.n_epochs, width, device=device, dtype=torch.float32
            )
            self.center = torch.zeros_like(self.total)
            self.count = torch.zeros(self.n_epochs, device=device, dtype=torch.float32)
        assert self.center is not None and self.count is not None
        self.center[start:end] = packed[:, length // 2]
        keep = valid.reshape(-1)
        ids = indices.reshape(-1)[keep]
        self.total.index_add_(0, ids, packed.reshape(-1, width)[keep])
        self.count.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    def finish(self, diagnostics_out: dict[str, np.ndarray] | None) -> np.ndarray:
        if self.overlap:
            if self.total is None or self.count is None or self.center is None:
                raise RuntimeError("Overlap-averaging produced no predictions")
            packed = _finalize_overlap_average(self.total, self.count, self.center)
        else:
            packed = torch.cat(self.chunks, dim=0).cpu().numpy()
        prediction = packed[..., : self.n_classes]
        if self.has_moments and diagnostics_out is not None:
            mean = packed[..., self.n_classes : 2 * self.n_classes]
            second = packed[..., 2 * self.n_classes : 3 * self.n_classes]
            diagnostics_out.update(
                {
                    "mc_probability_variance": np.maximum(second - mean * mean, 0.0),
                    "mc_expected_entropy": packed[..., -1].copy(),
                    "mc_predictive_entropy": -np.sum(
                        mean * np.log(np.maximum(mean, np.finfo(np.float32).tiny)),
                        axis=-1,
                    ),
                }
            )
        if self.probability_space:
            prediction = np.log(np.maximum(prediction, np.finfo(np.float32).tiny))
        return np.ascontiguousarray(prediction)


def _with_recording_band_statistics[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Keep whole-recording band statistics through scoring."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        from spectra.model.band_norm_access import recording_norm_modules

        values = signature.bind(*args, **kwargs).arguments
        model = values["model"]
        modules = recording_norm_modules(model)
        if not modules or all(
            norm._inference_mean is not None and norm._inference_std is not None
            for norm in modules.values()
        ):
            return function(*args, **kwargs)
        model.eval()
        presence = torch.as_tensor(values["presence_mask"], dtype=torch.bool)
        if "epoch_windows" in values:
            windows = values["epoch_windows"]
            n, length, channels = windows.shape[:3]
            mask = presence.reshape(1, channels).expand(n, -1)
            channel_mask = values.get("epoch_channel_mask")
            if channel_mask is not None:
                mask = (
                    mask
                    & torch.as_tensor(
                        channel_mask[:, length // 2], device=mask.device
                    ).bool()
                )
        else:
            samples = int(values["fs"]) * int(values["epoch_sec"])
            data = torch.as_tensor(values["data"])
            channels, total = data.shape
            n = total // samples
            windows = (
                data[:, : n * samples]
                .reshape(channels, n, samples)
                .permute(1, 0, 2)[:, None]
            )
            mask = presence.reshape(1, channels).expand(n, -1)
            channel_mask = values.get("epoch_channel_valid")
            if channel_mask is not None:
                mask = mask & torch.as_tensor(channel_mask, device=mask.device).bool()
        with pinned_recording_band_statistics(
            model,
            windows,
            values["device"],
            epoch_valid=mask.any(dim=1),
            presence_mask=mask,
        ):
            return function(*args, **kwargs)

    return wrapped


@_with_recording_band_statistics
def run_inference(
    model: nn.Module,
    epoch_windows: torch.Tensor,
    presence_mask: torch.Tensor,
    device: torch.device,
    options: ScoreOptions,
    *,
    center_valid_mask: np.ndarray | None = None,
    epoch_channel_mask: torch.Tensor | None = None,
    diagnostics_out: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Run batched inference on epoch windows.

    Args:
        model: Model to use for inference
        epoch_windows: Epoch windows (n_epochs, context_len, n_channels, samples_per_epoch)
        presence_mask: Channel presence mask (n_channels,)
        device: Device for inference
        options: Scoring options

        center_valid_mask: Optional boolean array ``(n_epochs,)`` marking which
            center epochs are labeled/valid. Only consulted when
            ``options.overlap_average`` is True: windows whose center is invalid
            are excluded from the overlap accumulation (train/test parity with the
            ``drop_unlabeled_center`` eval default). None means "all centers valid".
        epoch_channel_mask: Optional availability windows shaped
            ``[n_epochs, context_len, channels]``.

        diagnostics_out: Optional output dictionary populated with per-epoch sampling
            moments. Includes overlapping-context variation when enabled.

    Returns:
        Logits array ``(n_epochs, n_classes)``. When MC dropout is on the
        samples are pooled per ``options.mc_pooling``: ``"prob"`` returns
        ``log(mean(softmax(logits)))`` so a downstream ``softmax`` recovers the
        mean posterior exactly, while ``"logit"`` returns the raw logit mean.
    """

    if epoch_windows.ndim != 4 or epoch_windows.shape[1] != 21:
        raise ValueError("SPECTRA requires context_half=10 (21 context epochs)")
    if epoch_windows.shape[-1] != 3840:
        raise ValueError("SPECTRA requires 3840 samples per epoch at 128 Hz")
    model.eval()
    n_epochs = epoch_windows.shape[0]
    n_batches = (n_epochs + options.batch_size - 1) // options.batch_size

    overlap_average = bool(getattr(options, "overlap_average", False))
    center_valid_t: torch.Tensor | None = None
    if overlap_average and center_valid_mask is not None:
        center_valid_t = torch.as_tensor(
            np.asarray(center_valid_mask), dtype=torch.bool, device=device
        )

    # Setup AMP dtype (match training approach)
    if options.amp_mode == "fp16":
        amp_dtype = torch.float16
    elif options.amp_mode == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None  # No AMP for fp32

    # Move presence mask to device once
    presence_mask_device = presence_mask.to(device)
    if epoch_channel_mask is not None and epoch_channel_mask.shape[:1] != (n_epochs,):
        raise ValueError(
            "epoch_channel_mask must have n_epochs as its first dimension, got "
            f"{tuple(epoch_channel_mask.shape)}"
        )

    if diagnostics_out is not None:
        diagnostics_out.clear()
    with _mc_dropout_session(model, options) as (mc_samples, mc_prob_pool):
        accumulator = _InferenceAccumulator(
            n_epochs,
            overlap_average,
            mc_prob_pool and options.mc_aggregation == "consistent",
            stream_to_cpu=False,
        )
        with torch.inference_mode():
            # Request all-position logits [B, L, C] only in overlap-averaging mode; an
            # empty kwargs dict keeps the center-only call byte-identical when disabled.
            extra_fwd: dict[str, Any] = {"predict_all": True} if overlap_average else {}

            # Setup CUDA prefetching to overlap CPU→GPU transfer with computation
            use_prefetch = _should_use_cuda_prefetch(options, device, n_batches)
            prefetch_stream = torch.cuda.Stream(device=device) if use_prefetch else None
            next_batch_windows: torch.Tensor | None = None

            # Prefetch first batch
            if use_prefetch and n_batches > 0:
                with torch.cuda.stream(prefetch_stream):
                    next_batch_windows = epoch_windows[
                        0 : min(options.batch_size, n_epochs)
                    ].to(device, non_blocking=True)

            for batch_idx in range(n_batches):
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                start_idx = batch_idx * options.batch_size
                end_idx = min(start_idx + options.batch_size, n_epochs)

                # Get current batch (either prefetched or load now)
                if use_prefetch and next_batch_windows is not None:
                    # Wait for prefetch to complete
                    torch.cuda.current_stream().wait_stream(prefetch_stream)
                    batch_windows = next_batch_windows

                    # The batch was allocated on ``prefetch_stream`` but is consumed
                    # by the forward pass on the current stream. Without this the
                    # caching allocator returns the block to the prefetch stream's
                    # pool as soon as this Python reference is dropped, and a later
                    # prefetch copy can reuse it while the forward pass is still
                    # reading it -- silently corrupting the input. The loop keeps
                    # logits on-device precisely to avoid a per-batch sync, so there
                    # is nothing else serializing the two streams.
                    batch_windows.record_stream(torch.cuda.current_stream())

                    # Start prefetching next batch while we compute
                    if batch_idx + 1 < n_batches:
                        next_start = (batch_idx + 1) * options.batch_size
                        next_end = min(next_start + options.batch_size, n_epochs)
                        with torch.cuda.stream(prefetch_stream):
                            next_batch_windows = epoch_windows[next_start:next_end].to(
                                device, non_blocking=True
                            )
                    else:
                        next_batch_windows = None
                else:
                    batch_windows = epoch_windows[start_idx:end_idx].to(device)

                batch_mask = (
                    epoch_channel_mask[start_idx:end_idx].to(device)
                    if epoch_channel_mask is not None
                    else presence_mask_device
                )

                model_input = {
                    "wave": batch_windows,
                    "presence_mask": batch_mask,
                }

                # Inject the epoch-validity mask (loop-boundary protection)
                if isinstance(model_input, dict):
                    epoch_valid_mask = _build_epoch_valid_mask(
                        start_idx,
                        end_idx,
                        n_epochs,
                        batch_windows.shape[1],
                        device,
                    )
                    if batch_mask.ndim == 3:
                        epoch_valid_mask &= batch_mask.bool().any(dim=-1)
                    model_input["epoch_valid_mask"] = epoch_valid_mask

                # Forward pass
                batch_out, moments = _sample_inference_batch(
                    model_input,
                    options,
                    device,
                    amp_dtype,
                    mc_samples,
                    mc_prob_pool,
                    lambda inp: model(inp, **extra_fwd),
                    collect_diagnostics=options.use_mc_dropout
                    and diagnostics_out is not None,
                )
                accumulator.add(
                    batch_out, moments, start_idx, end_idx, batch_mask, center_valid_t
                )

        return accumulator.finish(diagnostics_out)


@_with_recording_band_statistics
def run_inference_sequential(
    model: nn.Module,
    data: np.ndarray,
    presence_mask: np.ndarray,
    fs: int,
    epoch_sec: int,
    context_half: int,
    device: torch.device,
    options: ScoreOptions,
    progress_callback: Callable[[str, int], None] | None = None,
    *,
    center_valid_mask: np.ndarray | None = None,
    epoch_channel_valid: np.ndarray | None = None,
    diagnostics_out: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Run batched inference with sequential epoch loading (memory-efficient).

    This function uses a generator to create epoch batches on-the-fly, which
    dramatically reduces memory usage compared to pre-allocating all epochs.
    Recommended for long recordings or systems with limited RAM.

    Args:
        model: Model to use for inference
        data: Preprocessed data (n_channels, n_samples)
        presence_mask: Channel presence mask (n_channels,)
        fs: Sampling frequency
        epoch_sec: Epoch duration in seconds
        context_half: Context window half-width
        device: Device for inference
        options: Scoring options
        progress_callback: Optional callback for progress updates

        center_valid_mask: Optional boolean array ``(n_epochs,)`` marking valid
            center epochs; only consulted under ``options.overlap_average`` to
            exclude unlabeled-center windows from the overlap accumulation. None
            means "all centers valid" (the pure-scoring default).
        epoch_channel_valid: Optional signal availability shaped
            ``[n_epochs, channels]``. It is expanded into each context window.

        diagnostics_out: Optional output dictionary populated with per-epoch sampling
            moments. Includes overlapping-context variation when enabled.

    Returns:
        Logits array ``(n_epochs, n_classes)``. When MC dropout is on the
        samples are pooled per ``options.mc_pooling``: ``"prob"`` returns
        ``log(mean(softmax(logits)))`` so a downstream ``softmax`` recovers the
        mean posterior exactly, while ``"logit"`` returns the raw logit mean.
    """
    if (fs, epoch_sec, context_half) != (128, 30, 10):
        raise ValueError(
            "SPECTRA requires 128 Hz, 30-second epochs, and context_half=10"
        )

    model.eval()

    # Calculate total epochs
    n_channels, n_samples = data.shape
    samples_per_epoch = fs * epoch_sec
    n_epochs = n_samples // samples_per_epoch

    logger.info(
        f"Sequential inference mode: processing {n_epochs} epochs in batches of {options.batch_size}"
    )
    logger.info(
        f"Memory savings: ~{(n_epochs * (2 * context_half + 1) * n_channels * samples_per_epoch * 4) / (1024**3):.2f} GB not pre-allocated"
    )

    overlap_average = bool(getattr(options, "overlap_average", False))
    center_valid_t: torch.Tensor | None = None
    if overlap_average and center_valid_mask is not None:
        center_valid_t = torch.as_tensor(
            np.asarray(center_valid_mask), dtype=torch.bool, device=device
        )

    # Setup AMP dtype
    if options.amp_mode == "fp16":
        amp_dtype = torch.float16
    elif options.amp_mode == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None

    if diagnostics_out is not None:
        diagnostics_out.clear()
    with _mc_dropout_session(model, options) as (mc_samples, mc_prob_pool):
        accumulator = _InferenceAccumulator(
            n_epochs,
            overlap_average,
            mc_prob_pool and options.mc_aggregation == "consistent",
            stream_to_cpu=True,
        )
        with torch.inference_mode():
            # Create generator for sequential batch processing
            batch_generator = create_epoch_batch_generator(
                data,
                presence_mask,
                fs,
                epoch_sec,
                context_half,
                options.batch_size,
                epoch_channel_valid=epoch_channel_valid,
            )

            batch_idx = 0

            for batch_tensor, batch_presence, batch_start, batch_end in batch_generator:
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                batch = batch_tensor.to(device)
                batch_presence_mask = batch_presence.to(device)

                # Report progress
                if progress_callback:
                    percent = 65 + int(20 * (batch_end / n_epochs))  # 65-85% range
                    progress_callback(
                        f"Processing epochs {batch_start}-{batch_end}/{n_epochs}",
                        percent,
                    )

                model_input: dict[str, Any] = {
                    "wave": batch,
                    "presence_mask": batch_presence_mask,
                }

                # Inject the epoch-validity mask (loop-boundary protection)
                if isinstance(model_input, dict):
                    model_input["epoch_valid_mask"] = _build_epoch_valid_mask(
                        batch_start,
                        batch_end,
                        n_epochs,
                        batch.shape[1],
                        device,
                    ) & batch_presence_mask.bool().any(dim=-1)

                forward_kwargs: dict[str, Any] = (
                    {"predict_all": True} if overlap_average else {}
                )

                batch_out, moments = _sample_inference_batch(
                    model_input,
                    options,
                    device,
                    amp_dtype,
                    mc_samples,
                    mc_prob_pool,
                    lambda inp, kwargs=forward_kwargs: model(inp, **kwargs),
                    collect_diagnostics=options.use_mc_dropout
                    and diagnostics_out is not None,
                )
                accumulator.add(
                    batch_out,
                    moments,
                    batch_start,
                    batch_end,
                    batch_presence_mask,
                    center_valid_t,
                )
                batch_idx += 1

        return accumulator.finish(diagnostics_out)


def postprocess_logits(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Convert logits to predictions, confidence, and uncertainty flags."""
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits shape (n_epochs, n_classes), got {logits.shape}"
        )
    probabilities = _softmax_np(logits, axis=-1)
    predictions = np.argmax(probabilities, axis=-1).astype(np.int64)
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "raw_predictions": predictions,
        "raw_probabilities": probabilities,
        "confidences": np.max(probabilities, axis=-1).astype(np.float32, copy=False),
        **compute_flag_scores(probabilities),
    }


@torch.no_grad()
def _pin_recording_band_statistics(
    model: nn.Module,
    epoch_windows: torch.Tensor,
    device: torch.device,
    *,
    epoch_valid: torch.Tensor | None = None,
    presence_mask: torch.Tensor | None = None,
) -> list[PerRecordingBandNorm]:
    """Compute this recording's envelope statistics and pin them for inference.

    A model trained with ``--band_norm_per_recording`` learned on envelopes
    normalised per recording, so inference must apply the same correction or
    the encoder sees a distribution it never trained on. Each enabled norm
    (EEG band path, EMG envelope) is pinned from the recording's centre epochs
    through the encoder's own envelope function and the reducer named by the
    module's ``statistic`` -- the same two functions the training-time table
    was built with, so train and inference agree by construction.

    Args:
        model: The unwrapped inference model.
        epoch_windows: ``[n_windows, context_len, channels, samples]``.
        device: Device to compute the envelopes on.
        epoch_valid: Optional bool ``[n_windows]`` restricting the statistics
            to signal-valid centre epochs (the training population).
        presence_mask: Optional ``[C]`` or ``[n_windows, C]`` centre-channel mask.

    Returns:
        The pinned modules, so the caller can clear them; empty when the model
        has no enabled per-recording norm.
    """
    from spectra.model.band_norm_access import find_multirate_encoder
    from spectra.models.multirate_asymmetric_epoch_cnn import (
        reduce_recording_statistics,
    )

    encoder = find_multirate_encoder(model)
    if encoder is None:
        return []
    modules = encoder.recording_norm_modules()
    if not modules:
        return []
    count, length, channels = epoch_windows.shape[:3]
    mask = torch.ones(count, channels, dtype=torch.bool)
    if presence_mask is not None:
        available = torch.as_tensor(presence_mask, dtype=torch.bool, device="cpu")
        if available.shape not in ((channels,), (count, channels)):
            raise ValueError("Band normalization presence mask must be [C] or [N,C]")
        mask &= available
    if epoch_valid is not None:
        keep = torch.as_tensor(epoch_valid, dtype=torch.bool, device="cpu").reshape(-1)
        if keep.numel() != count:
            raise ValueError("Band normalization epoch_valid must have N entries")
        mask &= keep[:, None]
    summaries: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {
        modality: ([], []) for modality in modules
    }
    for start in range(0, count, 64):
        # A bounded slice also supports lazy context-window providers.
        chunk = torch.as_tensor(epoch_windows[start : start + 64])[:, length // 2].to(
            device, torch.float32
        )
        chunk_mask = mask[start : start + 64].to(device)
        chunk = torch.where(chunk_mask[..., None], chunk, 0.0)
        chunk = torch.nan_to_num(chunk, nan=0.0, posinf=10.0, neginf=-10.0)
        for modality, norm in modules.items():
            idx = list(encoder.modality_channel_indices(modality))
            keep = chunk_mask[:, idx].any(dim=1)
            if not bool(keep.any()):
                continue
            with torch.autocast(device_type=device.type, enabled=False):
                env = encoder.modality_envelope(modality, chunk[keep][:, idx]).double()
            per_epoch_mean, per_epoch_var = norm.per_sample_statistics(env)
            summaries[modality][0].append(per_epoch_mean.cpu())
            summaries[modality][1].append(per_epoch_var.cpu())
    computed: list[tuple[PerRecordingBandNorm, torch.Tensor, torch.Tensor]] = []
    for modality, norm in modules.items():
        means, variances = summaries[modality]
        if means:
            loc, scale = reduce_recording_statistics(
                norm.statistic, torch.cat(means), torch.cat(variances), eps=norm.eps
            )
        else:
            # No signal exists in this modality. Pin an explicit neutral row;
            # never let an unrelated saved recording's row provide its stats.
            loc, scale = torch.zeros(norm.num_channels), torch.ones(norm.num_channels)
        computed.append((norm, loc, scale))
        logger.info(
            "Per-recording band norm: prepared %s statistics for %s from %d epochs",
            norm.statistic,
            modality,
            sum(m.shape[0] for m in means),
        )
    # Install only after all modalities have been computed successfully.
    for norm, loc, scale in computed:
        norm.set_inference_statistics(mean=loc.float(), std=scale.float())
    return [norm for norm, _, _ in computed]


@contextmanager
def pinned_recording_band_statistics(
    model: nn.Module,
    epoch_windows: torch.Tensor,
    device: torch.device,
    *,
    epoch_valid: torch.Tensor | None = None,
    presence_mask: torch.Tensor | None = None,
) -> Iterator[None]:
    """Pin this recording's envelope statistics for the duration of the block.

    Any caller that runs a model trained with ``--band_norm_per_recording``
    outside :func:`infer_recording` must wrap the forward passes in this,
    or the per-recording normalisation silently degrades to identity and the
    encoder sees an envelope distribution it never trained on.

    Args:
        model: The unwrapped inference model.
        epoch_windows: ``[n_windows, context_len, channels, samples]`` for this
            recording. A single-epoch recording may be passed as
            ``[n_epochs, 1, channels, samples]``.
        device: Device to compute the statistics on.
        epoch_valid: Optional bool ``[n_windows]`` restricting the statistics
            to signal-valid centre epochs, as :func:`infer_recording` does.
    """
    from spectra.model.band_norm_access import recording_norm_modules

    mask = (
        presence_mask
        if presence_mask is not None
        else torch.ones(
            epoch_windows.shape[0], epoch_windows.shape[2], dtype=torch.bool
        )
    )
    previous = [
        (norm, norm._inference_mean, norm._inference_std)
        for norm in recording_norm_modules(model).values()
    ]
    try:
        _pin_recording_band_statistics(
            model, epoch_windows, device, epoch_valid=epoch_valid, presence_mask=mask
        )
        yield
    finally:
        for norm, mean, std in previous:
            norm._inference_mean, norm._inference_std = mean, std


@_with_recording_band_statistics
def infer_recording(
    model: nn.Module,
    epoch_windows: torch.Tensor,
    presence_mask: torch.Tensor,
    device: torch.device,
    options: ScoreOptions,
    *,
    center_valid_mask: np.ndarray | None = None,
    epoch_channel_mask: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Run inference and compute uncertainty diagnostics for an already-windowed recording.

    ``center_valid_mask`` (bool ``(n_epochs,)``) is forwarded to ``run_inference``
    and only consulted under ``options.overlap_average`` to drop unlabeled-center
    windows from the overlap accumulation (see ``run_inference``).

    ``epoch_channel_mask`` (``[n_epochs, context_len, channels]``) is likewise
    forwarded: it replaces the recording-level presence mask per window and
    supplies the model's epoch-validity mask, so per-epoch channel dropout is
    handled the same way as in supervised training (see ``run_inference``).
    """
    diagnostics: dict[str, np.ndarray] = {}
    logits = run_inference(
        model,
        epoch_windows,
        presence_mask,
        device,
        options,
        center_valid_mask=center_valid_mask,
        epoch_channel_mask=epoch_channel_mask,
        diagnostics_out=diagnostics,
    )

    result = postprocess_logits(logits)
    result.update(diagnostics)
    return result


def save_results(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_dir: str,
    edf_basename: str,
    stage_names: list[str] = STAGE_NAMES_5,
    artifact_stats: dict | None = None,
    canonical_channels: list[str] | None = None,
    *,
    flag_scores: dict[str, np.ndarray] | None = None,
) -> dict[str, str]:
    """Save predictions and probabilities to files.

    Args:
        predictions: Predicted classes (n_epochs,)
        probabilities: Class probabilities (n_epochs, n_classes)
        output_dir: Output directory
        edf_basename: Base name for output files
        stage_names: Stage name labels
        artifact_stats: Optional artifact statistics dict
        canonical_channels: Optional list of canonical channel names
        flag_scores: Optional per-epoch uncertainty arrays (e.g.
            ``flag_margin``/``flag_entropy``/``flag_maxprob`` and the MC-dropout
            ``mc_*`` arrays). When provided, they are written to a single
            ``{edf_basename}_flags.npz`` archive.

    Returns:
        Dict of output file paths
    """
    # Ensure output_dir is valid (handle empty strings)
    if not output_dir or str(output_dir).strip() == "":
        output_dir = "."
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions as CSV
    predictions_csv = resolved_output_dir / f"{edf_basename}_predictions.csv"
    with open(predictions_csv, "w") as f:
        f.write("Epoch,Stage\n")
        for i, pred in enumerate(predictions):
            pred_int = int(pred)
            stage = "Unscored" if pred_int < 0 else stage_names[pred_int]
            f.write(f"{i},{stage}\n")

    # Save probabilities as NPY
    probabilities_npy = resolved_output_dir / f"{edf_basename}_probabilities.npy"
    np.save(probabilities_npy, probabilities)

    logger.info(f"Saved predictions to: {predictions_csv}")
    logger.info(f"Saved probabilities to: {probabilities_npy}")

    output_paths = {
        "predictions_csv": str(predictions_csv),
        "probabilities_npy": str(probabilities_npy),
    }

    # Save optional per-epoch uncertainty arrays as a single compressed archive.
    if flag_scores:
        flags_npz = resolved_output_dir / f"{edf_basename}_flags.npz"
        flag_arrays = {k: np.asarray(v) for k, v in flag_scores.items()}
        # ``**`` unpack of a str->ndarray dict trips pyright's overload match
        # against savez_compressed's ``allow_pickle: bool`` keyword.
        np.savez_compressed(flags_npz, **flag_arrays)  # type: ignore[reportArgumentType]
        logger.info(f"Saved uncertainty flags to: {flags_npz}")
        output_paths["flags_npz"] = str(flags_npz)

    return output_paths


def score_recording(
    edf_path: str,
    checkpoint: str,
    canon_json: str | None,
    output_dir: str,
    device: str = "auto",
    options: ScoreOptions | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
) -> dict:
    """Score a PSG recording from EDF file.

    This is the main entry point for inference.

    Args:
        edf_path: Path to EDF file
        checkpoint: Path to model checkpoint
        canon_json: Optional path to canonical channels JSON
        output_dir: Output directory for results
        device: Device to use ("auto", "cuda", "mps", "cpu")
        options: Scoring options (uses defaults if None)
        progress_callback: Optional callback for progress updates

    Returns:
        Dict with results and output paths
    """
    if options is None:
        options = ScoreOptions()

    def report_progress(message: str, percent: int = 0):
        logger.info(message)
        if progress_callback:
            progress_callback(message, percent)

    # Setup device
    resolved_device, device_warning, device_description = _resolve_inference_device(
        device
    )
    if device_warning:
        logger.warning(device_warning)
        report_progress(device_warning, 1)

    logger.info("Using device: %s (%s)", resolved_device, device_description)

    # Load checkpoint metadata early so we can align preprocessing parameters
    report_progress("Reading checkpoint metadata", 5)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_time_len = _validate_checkpoint_geometry(checkpoint_data)
    logger.info(
        "Inference sampling parameters: fs=%d Hz, epoch_sec=%d (checkpoint time_len=%s)",
        options.fs,
        options.epoch_sec,
        expected_time_len if expected_time_len is not None else "unknown",
    )

    from spectra.preprocessing.edf import preprocess_edf

    if (options.fs, options.epoch_sec) != (128, 30):
        raise InferencePreprocessingError(
            "SPECTRA requires 128 Hz and 30-second epochs"
        )
    canonical_channels = load_canonical_channels(canon_json) if canon_json else None
    report_progress("Selecting and preprocessing EDF channels", 10)
    try:
        prepared = preprocess_edf(
            edf_path,
            canonical_channels,
            start_epoch=options.start_epoch,
            end_epoch=options.end_epoch,
            auto_signal_window=options.auto_signal_window,
        )
    except ValueError as exc:
        raise InferencePreprocessingError(str(exc)) from exc
    inference_channel_names = prepared.channel_names
    canonical_channels = list(inference_channel_names)
    presence_mask = prepared.presence_mask.astype(np.float32)
    presence_mask_tensor = torch.from_numpy(presence_mask)
    analysis_epoch_signal_valid = prepared.epoch_signal_valid
    analysis_start, analysis_end = prepared.start_epoch, prepared.end_epoch
    original_n_epochs = prepared.original_n_epochs
    n_epochs_available = analysis_end - analysis_start
    n_epochs = n_epochs_available
    n_channels = n_channels_in_data = len(inference_channel_names)
    preprocessed_data = np.ascontiguousarray(
        prepared.signals_stacked.transpose(1, 0, 2).reshape(n_channels, -1)
    )

    # These arrays are already normalized by the converter's shared helper.
    # Pass them directly to the unwrapped model without further preprocessing.
    report_progress("Loading checkpoint", 40)
    model_channel_names, _ = resolve_model_channel_names(
        checkpoint_data, canonical_channels=canonical_channels
    )
    model, checkpoint_dict = load_model_from_checkpoint(
        checkpoint,
        resolved_device,
        checkpoint_data=checkpoint_data,
        options=options,
        channel_names=model_channel_names,
    )
    epoch_windows = None
    epoch_channel_mask_windows = None
    if not options.sequential_loading:
        epoch_windows, presence_mask_tensor = create_epoch_batches(
            preprocessed_data,
            presence_mask,
            options.fs,
            options.epoch_sec,
            options.context_half,
        )
        epoch_channel_mask_windows = _create_epoch_channel_mask_windows(
            analysis_epoch_signal_valid, options.context_half
        )

    # Validate channel count matches model expectations
    # NOTE: This validation checks the number of INPUT CHANNELS (e.g., EEG, EOG, EMG channels).
    # Engineered sleep features do NOT affect this count - they are additional features
    # concatenated in the feature dimension AFTER epoch encoding, not additional channels.
    # This error indicates a mismatch between the canonical channel list used during
    # training vs. inference.
    model_in_ch = _get_model_expected_channels(model)

    logger.info(f"Model expects {model_in_ch} input channels")
    if canonical_channels is not None:
        logger.info(f"Canonical channels file has {len(canonical_channels)} channels")
    else:
        logger.info("No canonical channels file provided; using raw EDF channel order")
    logger.info(f"Data tensor has {n_channels_in_data} channels")

    if model_in_ch is not None and model_in_ch != n_channels_in_data:
        # Try to get expected channels from checkpoint metadata
        expected_channels_hint = ""
        if "canonical_channels" in checkpoint_dict:
            expected_channels = checkpoint_dict["canonical_channels"]
            expected_channels_hint = (
                f"\n\nExpected channels from checkpoint metadata:\n{expected_channels}"
            )
        elif (
            "config" in checkpoint_dict
            and "canonical_channels" in checkpoint_dict["config"]
        ):
            expected_channels = checkpoint_dict["config"]["canonical_channels"]
            expected_channels_hint = (
                f"\n\nExpected channels from checkpoint metadata:\n{expected_channels}"
            )

        if canonical_channels is not None:
            raise InferencePreprocessingError(
                f"Channel count mismatch detected:\n"
                f"  - Model expects: {model_in_ch} input channels\n"
                f"  - Canonical channels JSON provides: {len(canonical_channels)} channels\n"
                f"  - Data tensor has: {n_channels_in_data} channels\n\n"
                "Canonical channels from "
                f"{Path(canon_json).name if canon_json is not None else 'runtime configuration'}:\n"
                f"{canonical_channels}\n"
                f"{expected_channels_hint}\n\n"
                f"Possible causes:\n"
                f"1. Different canonical_channels.json than used during training\n"
                f"2. Model was trained with different number of channels\n"
                f"3. Checkpoint file is from a different model configuration\n\n"
                f"To fix: Use the canonical_channels.json that matches the model's training config"
            )
        raise InferencePreprocessingError(
            f"Channel count mismatch detected:\n"
            f"  - Model expects: {model_in_ch} input channels\n"
            f"  - Raw EDF channel order provides: {n_channels_in_data} channels\n\n"
            f"Raw EDF channels:\n{inference_channel_names}\n"
            f"{expected_channels_hint}\n\n"
            f"Possible causes:\n"
            f"1. The model was trained with a different channel count/order\n"
            f"2. This recording needs a matching canonical_channels.json for alignment\n"
            f"3. Checkpoint file is from a different model configuration"
        )

    logger.info(
        f"✓ Channel validation passed: {n_channels_in_data} channels match model expectations"
    )

    # Keep recording statistics available throughout scoring.
    recording_context = ExitStack()
    # Use try/finally to ensure cleanup happens even if there's an error
    try:
        report_progress("Preparing recording statistics", 55)

        support_wave = torch.from_numpy(prepared.signals_stacked).unsqueeze(1)
        support_mask = torch.as_tensor(
            analysis_epoch_signal_valid, dtype=torch.bool
        ) & torch.as_tensor(presence_mask, dtype=torch.bool)
        recording_context.enter_context(
            pinned_recording_band_statistics(
                model,
                support_wave,
                resolved_device,
                epoch_valid=support_mask.any(dim=1),
                presence_mask=support_mask,
            )
        )
        del support_wave, support_mask

        # Run inference
        report_progress("Running inference", 65)

        diagnostics: dict[str, np.ndarray] = {}
        if options.sequential_loading:
            # Use memory-efficient sequential inference
            logits = run_inference_sequential(
                model,
                preprocessed_data,
                presence_mask,
                options.fs,
                options.epoch_sec,
                options.context_half,
                resolved_device,
                options,
                progress_callback=progress_callback,
                epoch_channel_valid=analysis_epoch_signal_valid,
                diagnostics_out=diagnostics,
            )
        else:
            # Use original batch inference
            if epoch_windows is None:
                raise RuntimeError(
                    "epoch_windows must be materialized for batch inference"
                )
            logits = run_inference(
                model,
                epoch_windows,
                presence_mask_tensor,
                resolved_device,
                options,
                epoch_channel_mask=epoch_channel_mask_windows,
                diagnostics_out=diagnostics,
            )

        post = postprocess_logits(logits)

        probabilities = post["probabilities"]
        post.update(diagnostics)
        predictions = post["predictions"]
        raw_probabilities = post.get("raw_probabilities")
        raw_predictions = post.get("raw_predictions")
        confidences = post.get("confidences")

        # Per-epoch uncertainty arrays computed by postprocessing (flag_* are
        # always present). Collect the subset that exists so we can both persist
        # and return them (additive, so existing callers that only read the base
        # keys are unaffected).
        optional_arrays: dict[str, np.ndarray] = {
            k: post[k] for k in (*FLAG_SCORE_KEYS, *MC_SCORE_KEYS) if k in post
        }

        report_progress("Post-processing predictions", 85)

        def _expand_epoch_array(
            value: np.ndarray | None, *, fill_value: float | int
        ) -> np.ndarray | None:
            """Restore analysis-local output to the original EDF epoch grid."""
            if value is None:
                return None
            local = np.asarray(value)
            if local.shape[0] != n_epochs:
                raise RuntimeError(
                    "Inference output length does not match the analysis interval: "
                    f"{local.shape[0]} vs {n_epochs}"
                )
            expanded = np.full(
                (original_n_epochs, *local.shape[1:]),
                fill_value,
                dtype=local.dtype,
            )
            expanded[analysis_start:analysis_end] = local
            return expanded

        predictions = cast(np.ndarray, _expand_epoch_array(predictions, fill_value=-1))
        probabilities = cast(
            np.ndarray, _expand_epoch_array(probabilities, fill_value=0.0)
        )
        raw_predictions = _expand_epoch_array(raw_predictions, fill_value=-1)
        raw_probabilities = _expand_epoch_array(raw_probabilities, fill_value=0.0)
        confidences = _expand_epoch_array(confidences, fill_value=0.0)
        optional_arrays = {
            key: cast(np.ndarray, _expand_epoch_array(value, fill_value=0.0))
            for key, value in optional_arrays.items()
        }

        full_signal_valid = np.zeros((original_n_epochs, n_channels), dtype=bool)
        full_signal_valid[analysis_start:analysis_end] = analysis_epoch_signal_valid
        invalid = ~full_signal_valid.any(axis=1)
        predictions[invalid] = -1
        probabilities[invalid] = 0.0
        if raw_predictions is not None:
            raw_predictions[invalid] = -1
        if raw_probabilities is not None:
            raw_probabilities[invalid] = 0.0
        if confidences is not None:
            confidences[invalid] = 0.0
        for value in optional_arrays.values():
            value[invalid] = 0.0

        # Save results
        report_progress("Saving results", 95)
        edf_basename = Path(edf_path).stem
        output_paths = save_results(
            predictions,
            probabilities,
            output_dir,
            edf_basename,
            canonical_channels=canonical_channels,
            flag_scores=optional_arrays or None,
        )

        validity_path = Path(output_dir) / f"{edf_basename}_epoch_signal_valid.npy"
        np.save(validity_path, full_signal_valid)
        provenance_path = Path(output_dir) / f"{edf_basename}_preprocessing.json"
        provenance = {
            "sampling_rate": 128,
            "epoch_seconds": 30,
            "channel_names": inference_channel_names,
            "channel_mapping": prepared.channel_mapping,
            "presence_mask": prepared.presence_mask.tolist(),
            "normalization": prepared.normalization,
            "analysis_start_epoch": analysis_start,
            "analysis_end_epoch": analysis_end,
            "original_n_epochs": original_n_epochs,
            "reader_backend": prepared.reader_backend,
            "rereferencing": False,
            "filtering": False,
        }
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        output_paths["epoch_signal_valid"] = str(validity_path)
        output_paths["preprocessing"] = str(provenance_path)

        report_progress("Complete", 100)

        result = {
            "n_epochs": original_n_epochs,
            "predictions": predictions,
            "probabilities": probabilities,
            "raw_predictions": raw_predictions,
            "raw_probabilities": raw_probabilities,
            "confidences": confidences,
            "score_window": (analysis_start, analysis_end),
            "epoch_signal_valid": full_signal_valid,
            "presence_mask": prepared.presence_mask,
            "channel_mapping": prepared.channel_mapping,
            "output_paths": output_paths,
            # Additive per-epoch uncertainty arrays (flag_* always, mc_* when MC
            # dropout ran). Consumers that only read the base keys are unaffected.
            **optional_arrays,
        }

        return result

    finally:
        recording_context.close()
        # Critical: Clean up ALL memory to prevent crashes on subsequent runs
        logger.info("Cleaning up model and memory...")

        import gc

        # Explicitly delete heavy objects in order of importance
        # Model and checkpoint data (largest memory consumers)
        try:
            del model
        except (NameError, UnboundLocalError):
            pass
        try:
            del checkpoint_data
        except (NameError, UnboundLocalError):
            pass
        try:
            del checkpoint_dict
        except (NameError, UnboundLocalError):
            pass

        # Data tensors and arrays
        try:
            del epoch_windows
        except (NameError, UnboundLocalError):
            pass
        try:
            del presence_mask_tensor
        except (NameError, UnboundLocalError):
            pass
        try:
            del preprocessed_data
        except (NameError, UnboundLocalError):
            pass
        try:
            del aligned_data
        except (NameError, UnboundLocalError):
            pass
        try:
            del channel_data
        except (NameError, UnboundLocalError):
            pass

        # Local outputs are released when this function exits. Avoid referring
        # to conditionally-created names here; that obscures static guarantees
        # and does not materially improve cleanup inside a finally block.

        # Force multiple garbage collection passes for thorough cleanup
        gc.collect()
        gc.collect()
        gc.collect()

        # Clear backend cache if using GPU. Use the resolved torch.device rather
        # than the raw ``device`` preference string (which may be "auto"/"cuda"
        # and has no ``.type`` attribute).
        cleanup_device = resolved_device
        device_type = cleanup_device.type

        if device_type in {"cuda", "mps"}:
            try:
                cleared = _cleanup_inference_device_memory(cleanup_device)
                if cleared:
                    logger.info(
                        "✓ %s memory cache cleared successfully", device_type.upper()
                    )
            except Exception as e:
                logger.warning(f"Failed to clear accelerator cache: {e}")

        # Final garbage collection
        gc.collect()
        logger.info("✓ Memory cleanup complete")
