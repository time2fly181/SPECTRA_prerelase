"""Complete inference runtime for PSG sleep staging.

This module provides end-to-end inference on EDF files with:
- EDF loading and channel extraction
- Channel name normalization and substitution
- Channel rereferencing (algebraic derivation)
- Resampling to match training sampling rate (bandpass filtering handled offline)
- Data normalization using wrapped preprocessor
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
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly

from spectra.model.recording_conditioning import with_recording_observations
from spectra.preprocessing.robust_normalization import (
    normalize_channel_masked_with_validity,
)
from spectra.utils.device_utils import clear_device_cache, resolve_device

if TYPE_CHECKING:
    from spectra.models.multirate_asymmetric_epoch_cnn import (
        PerRecordingBandNorm,
    )
    from spectra.preprocessing.config import ProcCfg

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


def _model_has_embedded_preprocessor(model: nn.Module) -> bool:
    """Return True if a model exposes an embedded waveform preprocessor."""
    if hasattr(model, "preprocessor"):
        return True
    if hasattr(model, "model") and hasattr(model.model, "preprocessor"):
        return True
    return False


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


def get_attention_capture_status(model: nn.Module) -> tuple[bool, str | None]:
    """Return whether a loaded inference model supports attention capture."""
    from spectra.models import TransformerContextNet

    inner_model = model.model if hasattr(model, "model") else model
    if not isinstance(inner_model, TransformerContextNet):
        return (
            False,
            "Attention visualization is only supported for TransformerContextNet models.",
        )
    if not hasattr(inner_model, "_last_attention_weights"):
        return (
            False,
            "This model version does not support attention weight capture.",
        )
    return True, None


# Legacy flexible-stem schema markers. ``(?:block\.)?`` covers both the wrapped
# blocks used by flexible_asymmetric and the bare blocks used elsewhere.
_GLOBAL_BRANCH_WEIGHT_RE = re.compile(
    r"^epoch_encoder\.stem\.(?:eeg|eog|emg)_branch\.global_branch\.1\.weight$"
)
_LEGACY_BRANCH_AA_RE = re.compile(
    r"^epoch_encoder\.stage\d+\.\d+\.(?:block\.)?branches\.\d+\.0\.kernel$"
)
_STEM_POOL_KERNEL_RE = re.compile(
    r"^epoch_encoder\.stem\.(?:eeg|eog|emg)_branch\.pool\.kernel$"
)
_STAGE_POOL_KERNEL_RE = re.compile(
    r"^epoch_encoder\.stage(\d+)\.0\.(?:block\.)?pool\.kernel$"
)
_STAGE_SHORTCUT_KERNEL_RE = re.compile(
    r"^epoch_encoder\.stage(\d+)\.0\.(?:block\.)?shortcut\.\d+\.kernel$"
)


def _fixed_filter_buffer_names(model: nn.Module | None) -> set[str]:
    """Return state-dict keys for fixed FIR buffers the model does not persist."""
    if model is None:
        return set()

    from spectra.models.anti_alias import (
        KaiserAntiAliasDownsample1D,
        KaiserAntiAliasUpsample1D,
    )

    return {
        f"{name}.kernel"
        for name, module in model.named_modules()
        if isinstance(module, KaiserAntiAliasDownsample1D | KaiserAntiAliasUpsample1D)
    }


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


def _robust_iqr_normalize_np(
    x: np.ndarray,
    presence_mask: np.ndarray,
    *,
    stats_source: np.ndarray | None = None,
    epoch_signal_valid: np.ndarray | None = None,
    samples_per_epoch: int | None = None,
    clip_threshold: float = 20.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    """Per-recording per-channel robust scaling: (x - median) / IQR, clipped.

    This matches ``batch_edf_to_zarr_fp32.py``: statistics come only from
    signal-valid epochs in the selected analysis interval and invalid
    channel-epochs are zero-filled after scaling.

    Args:
        x: Array shaped (C, T) to normalize.
        presence_mask: Array shaped (C,) with 1 for present channels, 0 for missing.
        stats_source: Optional array shaped (C, T_src) to compute stats from.
            If None, compute stats from x.
        epoch_signal_valid: Optional validity mask shaped ``[n_epochs, C]``.
        samples_per_epoch: Required with ``epoch_signal_valid``.
        clip_threshold: Clamp value after scaling (default ±20).
        eps: Minimum IQR to avoid division by zero.

    Returns:
        Tuple of (normalized_data, norm_stats, refined_epoch_signal_valid):
        - normalized_data: Array shaped (C, T) as float32.
        - norm_stats: Dict with 'median' and 'iqr' arrays of shape (C,)
          for potential denormalization / wave_raw reconstruction.
        - refined_epoch_signal_valid: Post-normalization validity shaped
          ``[n_epochs, C]``, or ``None`` when no epoch mask was supplied.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (C, T), got {x.shape}")
    if presence_mask.ndim != 1 or presence_mask.shape[0] != x.shape[0]:
        raise ValueError(
            f"Expected presence_mask shape (C,) matching x, got {presence_mask.shape} vs x {x.shape}"
        )
    resolved_stats_source = x if stats_source is None else stats_source
    if resolved_stats_source.ndim != 2 or resolved_stats_source.shape[0] != x.shape[0]:
        raise ValueError(
            "Expected stats_source shape (C, T_src) matching x channels, got "
            f"{resolved_stats_source.shape} vs x {x.shape}"
        )

    x_fp32 = x.astype(np.float32, copy=False)
    stats_fp32 = resolved_stats_source.astype(np.float32, copy=False)
    mask = presence_mask.astype(bool, copy=False)
    n_channels = x_fp32.shape[0]
    validity: np.ndarray | None = None
    resolved_samples_per_epoch = 0
    expected_epochs = 0
    if epoch_signal_valid is not None:
        validity = np.asarray(epoch_signal_valid, dtype=bool)
        if samples_per_epoch is None or samples_per_epoch <= 0:
            raise ValueError(
                "samples_per_epoch must be positive when epoch_signal_valid is used"
            )
        resolved_samples_per_epoch = int(samples_per_epoch)
        expected_epochs = stats_fp32.shape[1] // resolved_samples_per_epoch
        if validity.shape != (expected_epochs, n_channels):
            raise ValueError(
                "epoch_signal_valid must have shape "
                f"{(expected_epochs, n_channels)}, got {validity.shape}"
            )
    # Default stats for missing channels: keep zeros as zeros.
    med = np.zeros((n_channels, 1), dtype=np.float32)
    iqr = np.ones((n_channels, 1), dtype=np.float32)

    present_idx = np.where(mask)[0]
    if validity is not None and stats_source is None:
        complete_samples = expected_epochs * resolved_samples_per_epoch
        normalized = np.zeros_like(x_fp32)
        refined_validity = np.zeros_like(validity)
        for c in present_idx:
            channel_epochs = x_fp32[c, :complete_samples].reshape(
                expected_epochs,
                resolved_samples_per_epoch,
            )
            try:
                normalized_epochs, channel_stats, channel_validity = (
                    normalize_channel_masked_with_validity(
                        channel_epochs,
                        validity[:, c],
                        min_stat_epochs=10,
                        clip_threshold=clip_threshold,
                    )
                )
            except ValueError as exc:
                logger.warning(
                    "Channel %d became unavailable during post-normalization "
                    "validity refinement: %s",
                    c,
                    exc,
                )
                continue

            normalized[c, :complete_samples] = normalized_epochs.reshape(-1)
            refined_validity[:, c] = channel_validity
            med[c, 0] = float(channel_stats["median"])
            iqr[c, 0] = float(channel_stats["iqr"])
            if complete_samples < x_fp32.shape[1]:
                tail = (x_fp32[c, complete_samples:] - med[c, 0]) / iqr[c, 0]
                normalized[c, complete_samples:] = np.clip(
                    tail,
                    -float(clip_threshold),
                    float(clip_threshold),
                )

        norm_stats = {
            "median": med[:, 0].copy(),
            "iqr": iqr[:, 0].copy(),
        }
        return normalized, norm_stats, refined_validity

    for c in present_idx:
        if validity is None:
            ch = stats_fp32[c]
        else:
            source_epochs = stats_fp32[
                c, : expected_epochs * resolved_samples_per_epoch
            ]
            source_epochs = source_epochs.reshape(
                expected_epochs, resolved_samples_per_epoch
            )
            ch = source_epochs[validity[:, c]].reshape(-1)
        ch = ch[np.isfinite(ch)]
        if ch.size == 0:
            continue

        m = np.median(ch)
        p25 = np.percentile(ch, 25.0)
        p75 = np.percentile(ch, 75.0)
        q = float(p75 - p25)
        if abs(q) < eps:
            standard_deviation = float(np.std(ch, dtype=np.float64))
            q = standard_deviation if standard_deviation > eps else 1.0
        med[c, 0] = float(m)
        iqr[c, 0] = float(q)

    z = (x_fp32 - med) / iqr
    z = np.clip(z, -float(clip_threshold), float(clip_threshold))
    z[~mask] = 0.0
    if validity is not None:
        output_epochs = x_fp32.shape[1] // resolved_samples_per_epoch
        if output_epochs != validity.shape[0]:
            raise ValueError(
                "epoch_signal_valid does not align with the normalized recording"
            )
        z_epochs = z[:, : output_epochs * resolved_samples_per_epoch].reshape(
            n_channels, output_epochs, resolved_samples_per_epoch
        )
        z_epochs *= validity.T[:, :, None]
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    # Return 1-D per-channel stats for wave_raw reconstruction
    norm_stats = {
        "median": med[:, 0].copy(),  # (C,)
        "iqr": iqr[:, 0].copy(),  # (C,)
    }

    refined_validity = None if validity is None else validity.copy()
    return z.astype(np.float32, copy=False), norm_stats, refined_validity


def _robust_iqr_normalize_per_epoch_np(
    data: np.ndarray,
    presence_mask: np.ndarray,
    samples_per_epoch: int,
    *,
    clip_threshold: float = 20.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-epoch per-channel robust scaling: (x - median) / IQR, clipped.

    NOTE: This does NOT match batch_edf_to_zarr_fp32.py, which uses
    per-RECORDING normalization (single median/IQR from all epochs combined).
    Use _robust_iqr_normalize_np() for training-consistent normalization.

    This function normalizes each epoch independently:
    - Each epoch (30-second window) is normalized independently
    - Each channel uses its own per-epoch statistics
    - Method: x' = (x - median) / IQR, clipped to ±clip_threshold

    Args:
        data: Array shaped (C, T) where T = n_epochs * samples_per_epoch.
        presence_mask: Array shaped (C,) with 1 for present channels, 0 for missing.
        samples_per_epoch: Number of samples per epoch (e.g., fs * epoch_sec).
        clip_threshold: Clamp value after scaling (default ±20).
        eps: Minimum IQR to avoid division by zero.

    Returns:
        Tuple of (normalized_data, norm_stats):
        - normalized_data: Array shaped (C, T) as float32.
        - norm_stats: Dict with 'median' and 'iqr' arrays of shape (C, n_epochs)
          for potential denormalization / feature reconstruction.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected data with shape (C, T), got {data.shape}")
    if presence_mask.ndim != 1 or presence_mask.shape[0] != data.shape[0]:
        raise ValueError(
            f"Expected presence_mask shape (C,) matching data, got {presence_mask.shape} vs data {data.shape}"
        )

    n_channels, n_samples = data.shape
    n_epochs = n_samples // samples_per_epoch

    if n_epochs == 0:
        raise ValueError(
            f"Recording too short for per-epoch normalization: {n_samples} samples, "
            f"need at least {samples_per_epoch} samples per epoch"
        )

    # Truncate to complete epochs
    n_samples_truncated = n_epochs * samples_per_epoch
    data_fp32 = data[:, :n_samples_truncated].astype(np.float32, copy=True)
    mask = presence_mask.astype(bool, copy=False)

    # Reshape to (C, n_epochs, samples_per_epoch) for per-epoch processing
    data_epochs = data_fp32.reshape(n_channels, n_epochs, samples_per_epoch)

    # Store per-epoch stats for potential denormalization
    epoch_medians = np.zeros((n_channels, n_epochs), dtype=np.float32)
    epoch_iqrs = np.ones((n_channels, n_epochs), dtype=np.float32)

    present_idx = np.where(mask)[0]

    for c in present_idx:
        for e in range(n_epochs):
            epoch_data = data_epochs[c, e, :]

            # Handle NaN values
            if np.isnan(epoch_data).any():
                valid_data = epoch_data[~np.isnan(epoch_data)]
                if valid_data.size == 0:
                    # All NaN - leave as zeros
                    data_epochs[c, e, :] = 0.0
                    continue
            else:
                valid_data = epoch_data

            # Compute per-epoch statistics
            median = np.median(valid_data)
            p25 = np.percentile(valid_data, 25.0)
            p75 = np.percentile(valid_data, 75.0)
            iqr_val = float(p75 - p25)

            if abs(iqr_val) < eps:
                iqr_val = eps

            # Store stats
            epoch_medians[c, e] = float(median)
            epoch_iqrs[c, e] = float(iqr_val)

            # Normalize this epoch
            normalized = (epoch_data - median) / iqr_val
            normalized = np.clip(normalized, -clip_threshold, clip_threshold)
            data_epochs[c, e, :] = normalized.astype(np.float32)

    # Reshape back to (C, T)
    normalized_data = data_epochs.reshape(n_channels, n_samples_truncated)

    # If original data was longer, pad with zeros (shouldn't happen in practice)
    if n_samples > n_samples_truncated:
        padded = np.zeros((n_channels, n_samples), dtype=np.float32)
        padded[:, :n_samples_truncated] = normalized_data
        normalized_data = padded

    norm_stats = {
        "median": epoch_medians,  # (C, n_epochs)
        "iqr": epoch_iqrs,  # (C, n_epochs)
    }

    return normalized_data, norm_stats


def _extract_proc_cfg(checkpoint_dict: dict) -> ProcCfg | None:
    """Best-effort extraction of ProcCfg from checkpoint metadata."""
    proc_cfg_dict = checkpoint_dict.get("preprocessing_config")
    if not isinstance(proc_cfg_dict, dict):
        return None

    try:
        from spectra.preprocessing.config import (
            ProcCfg,  # Local import to avoid mandatory dependency
        )
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        logger.warning(
            "pydantic not available; cannot parse preprocessing_config (%s)", exc
        )
        return None

    try:
        return ProcCfg(**proc_cfg_dict)
    except Exception as exc:
        logger.warning("Failed to parse preprocessing_config from checkpoint: %s", exc)
        return None


def _extract_linear_out_features_from_state_key(
    state_dict: dict[str, Any],
    key_map: dict[str, str],
    key: str,
) -> int | None:
    """Return linear out_features from a checkpoint tensor when possible."""
    tensor = state_dict.get(key_map.get(key, ""))
    if isinstance(tensor, torch.Tensor):
        resolved_tensor = cast(torch.Tensor, tensor)
        if resolved_tensor.ndim >= 1:
            return int(resolved_tensor.shape[0])
    return None


def _infer_num_classes_from_classifier_keys(
    state_dict: dict[str, Any],
    keys: list[str],
    key_map: dict[str, str],
) -> tuple[int | None, str | None]:
    """Infer num_classes from the terminal classifier projection."""
    preferred_keys = (
        "classifier.head.weight",
        "classifier.3.weight",
        "classifier.head.3.weight",
        "classifier.direct_head.3.weight",
        "classifier.classifier.3.weight",
        "classifier.classifier.weight",
    )
    for key in preferred_keys:
        if key in key_map:
            out_features = _extract_linear_out_features_from_state_key(
                state_dict, key_map, key
            )
            if out_features is not None:
                return out_features, key

    suffix_priority = (
        ".head.weight",
        ".head.3.weight",
        ".direct_head.3.weight",
        ".classifier.3.weight",
        ".classifier.weight",
        ".out.weight",
        ".output.weight",
    )
    candidate_keys = [
        key
        for key in keys
        if key.endswith("weight")
        and "classifier" in key
        and not key.startswith("confidence_head.")
        and ".confidence_head." not in key
    ]
    for suffix in suffix_priority:
        matches = [key for key in candidate_keys if key.endswith(suffix)]
        if matches:
            key = sorted(matches)[-1]
            out_features = _extract_linear_out_features_from_state_key(
                state_dict, key_map, key
            )
            if out_features is not None:
                return out_features, key

    fallback_keys = [
        key
        for key in candidate_keys
        if all(
            excluded not in key
            for excluded in (
                "input_proj",
                "res_linear",
                "context_attn",
                "cross_attn",
                "transition",
                "router",
                "eng_proj",
                "norm",
            )
        )
    ]
    for key in sorted(fallback_keys, reverse=True):
        out_features = _extract_linear_out_features_from_state_key(
            state_dict, key_map, key
        )
        if out_features is not None:
            return out_features, key

    return None, None


def _validate_engineered_feature_statistics(model: nn.Module) -> None:
    """Validate that engineered feature statistics are properly loaded.

    This function checks that:
    1. The model has engineered features enabled
    2. Statistics buffers (global_mean, global_std, stats_initialized) exist
    3. Statistics were initialized during training (stats_initialized=True)
    4. Statistics have reasonable values (not default zeros/ones)

    Args:
        model: Model to validate (may be wrapped)

    Raises:
        RuntimeError: If statistics are missing or uninitialized

    Logs:
        Warning if statistics look suspicious but allow inference to continue
    """
    # Recursively gather SleepFeatureExtractor modules – models with N1 attention
    # can contain both the main epoch encoder extractor and an auxiliary N1-only one.
    sleep_feature_candidates: list[tuple[str, nn.Module]] = []

    def collect_sleep_features(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            child_path = f"{prefix}.{name}" if prefix else name
            if type(child).__name__ == "SleepFeatureExtractor":
                sleep_feature_candidates.append((child_path, child))
                logger.debug(f"Found SleepFeatureExtractor at: {child_path}")
            collect_sleep_features(child, child_path)

    collect_sleep_features(model)

    if not sleep_feature_candidates:
        logger.debug("No engineered features detected - skipping statistics validation")
        return

    def _required_tensor_attr(module: nn.Module, attr_name: str) -> torch.Tensor:
        value = getattr(module, attr_name, None)
        if not torch.is_tensor(value):
            raise RuntimeError(
                f"Expected tensor attribute '{attr_name}' on {type(module).__name__}"
            )
        return cast(torch.Tensor, value)

    def _scalar_attr_as_float(module: nn.Module, attr_name: str) -> float:
        value = getattr(module, attr_name, None)
        if torch.is_tensor(value):
            return float(cast(torch.Tensor, value).item())
        if isinstance(value, (int, float)):
            return float(value)
        raise RuntimeError(
            f"Expected scalar tensor attribute '{attr_name}' on {type(module).__name__}"
        )

    n1_candidates: list[tuple[str, nn.Module]] = [
        (path, module)
        for path, module in sleep_feature_candidates
        if hasattr(module, "n1_stats_initialized")
    ]

    def _module_priority(path: str) -> tuple[int, str]:
        lowered = path.lower()
        if "epoch_encoder" in lowered and "sleep_features" in lowered:
            return (0, path)
        if "sleep_features" in lowered:
            return (1, path)
        if "n1_feature_extractor" in lowered:
            return (3, path)
        return (2, path)

    selected_path, sleep_features_module = min(
        sleep_feature_candidates, key=lambda item: _module_priority(item[0])
    )
    logger.debug(f"Using SleepFeatureExtractor at: {selected_path}")

    logger.info("Engineered features detected - validating statistics...")

    # Check for required buffers (support both old and new format)
    # New format: separate N1 statistics (n1_global_mean, n1_global_std, n1_stats_initialized)
    # Old format: unified statistics (global_mean, global_std, stats_initialized)
    selected_has_n1_buffers = hasattr(sleep_features_module, "n1_stats_initialized")
    has_n1_buffers = bool(n1_candidates)

    if selected_has_n1_buffers:
        required_buffers = [
            "global_mean",
            "global_std",
            "stats_initialized",
            "n1_global_mean",
            "n1_global_std",
            "n1_stats_initialized",
            "alpha_global_baseline",
            "alpha_global_count",
        ]
    else:
        required_buffers = [
            "global_mean",
            "global_std",
            "stats_initialized",
            "alpha_global_baseline",
            "alpha_global_count",
        ]

    missing_buffers = []
    for buffer_name in required_buffers:
        if not hasattr(sleep_features_module, buffer_name):
            missing_buffers.append(buffer_name)

    if missing_buffers:
        raise RuntimeError(
            f"Checkpoint missing required buffers for engineered features: {missing_buffers}. "
            "This checkpoint may be from an incompatible version or was not properly saved."
        )

    # Check stats_initialized flag
    stats_initialized_tensor = _required_tensor_attr(
        sleep_features_module, "stats_initialized"
    )
    stats_initialized = bool(stats_initialized_tensor.item())
    if not stats_initialized:
        # Auto-initialize with safe defaults instead of failing
        # This allows loading checkpoints saved before first training batch
        logger.warning(
            "⚠️  Checkpoint has uninitialized engineered feature statistics. "
            "Auto-initializing to safe defaults (mean=0, std=1). "
            "The checkpoint was likely saved before any training batches were processed. "
            "For best results, retrain and ensure statistics are initialized during training."
        )
        # Initialize to defaults that won't change the values (zero mean, unit std = no normalization)
        _required_tensor_attr(sleep_features_module, "global_mean").fill_(0.0)
        _required_tensor_attr(sleep_features_module, "global_std").fill_(1.0)
        stats_initialized_tensor.fill_(1)

    # Get statistics
    global_mean = _required_tensor_attr(sleep_features_module, "global_mean")
    global_std = _required_tensor_attr(sleep_features_module, "global_std")

    # Check for suspicious values
    mean_min, mean_max = global_mean.min().item(), global_mean.max().item()
    std_min, std_max = global_std.min().item(), global_std.max().item()

    # Warn if mean is all zeros (likely not initialized properly)
    if abs(mean_min) < 1e-6 and abs(mean_max) < 1e-6:
        logger.warning(
            "⚠️  Engineered feature global_mean is all zeros! "
            "Statistics may not have been properly computed during training. "
            "Inference will continue but predictions may be incorrect."
        )

    # Warn if std is all ones (likely not initialized properly)
    if abs(std_min - 1.0) < 1e-6 and abs(std_max - 1.0) < 1e-6:
        logger.warning(
            "⚠️  Engineered feature global_std is all ones! "
            "Statistics may not have been properly computed during training. "
            "Inference will continue but predictions may be incorrect."
        )

    # Success - log statistics for verification
    logger.info(
        f"✓ Engineered feature statistics validated and loaded from checkpoint:\n"
        f"  mean range=[{mean_min:.3f}, {mean_max:.3f}], "
        f"  std range=[{std_min:.3f}, {std_max:.3f}]\n"
        f"  First 5 mean values: {global_mean[:5].tolist()}\n"
        f"  First 5 std values: {global_std[:5].tolist()}\n"
        f"  These statistics will be used to standardize engineered features during inference.\n"
        f"  They should match the statistics computed during training."
    )

    # Check alpha baseline
    if hasattr(sleep_features_module, "alpha_global_baseline"):
        alpha_baseline = _required_tensor_attr(
            sleep_features_module, "alpha_global_baseline"
        )
        alpha_count = _required_tensor_attr(sleep_features_module, "alpha_global_count")

        baseline_min, baseline_max = (
            alpha_baseline.min().item(),
            alpha_baseline.max().item(),
        )
        count_val = float(alpha_count.item())

        if (
            abs(baseline_min - 1.0) < 1e-6
            and abs(baseline_max - 1.0) < 1e-6
            and count_val <= 1.0
        ):
            logger.warning(
                "⚠️  Alpha global baseline is all ones (default)! "
                "Alpha attenuation features may not be properly calibrated."
            )
        else:
            logger.info(
                f"✓ Alpha global baseline validated:\n"
                f"  range=[{baseline_min:.3f}, {baseline_max:.3f}], count={count_val:.1f}"
            )

    # Check N1-specific statistics if present (new checkpoint format)
    if has_n1_buffers:

        def _n1_module_priority(path: str) -> tuple[int, str]:
            lowered = path.lower()
            if "n1_feature_extractor" in lowered:
                return (0, path)
            if "n1" in lowered:
                return (1, path)
            if "epoch_encoder" in lowered:
                return (2, path)
            return (3, path)

        n1_candidates_sorted = sorted(
            n1_candidates,
            key=lambda item: _n1_module_priority(item[0]),
        )
        initialized_candidate = next(
            (
                (path, module)
                for path, module in n1_candidates_sorted
                if bool(_required_tensor_attr(module, "n1_stats_initialized").item())
            ),
            None,
        )
        n1_path, n1_module = initialized_candidate or n1_candidates_sorted[0]
        n1_stats_initialized = _required_tensor_attr(n1_module, "n1_stats_initialized")

        if not bool(n1_stats_initialized.item()):
            # Auto-initialize N1 stats with safe defaults
            logger.warning(
                f"⚠️  N1-specific feature statistics are not initialized on '{n1_path}'! "
                "Auto-initializing to safe defaults (mean=0, std=1). "
                "If this model uses N1 attention, predictions may be suboptimal. "
                "Consider retraining to properly initialize N1 statistics."
            )
            _required_tensor_attr(n1_module, "n1_global_mean").fill_(0.0)
            _required_tensor_attr(n1_module, "n1_global_std").fill_(1.0)
            n1_stats_initialized.fill_(1)
        else:
            n1_mean = _required_tensor_attr(n1_module, "n1_global_mean")
            n1_std = _required_tensor_attr(n1_module, "n1_global_std")
            n1_mean_min, n1_mean_max = n1_mean.min().item(), n1_mean.max().item()
            n1_std_min, n1_std_max = n1_std.min().item(), n1_std.max().item()

            # Warn if N1 stats look uninitialized
            if abs(n1_mean_min) < 1e-6 and abs(n1_mean_max) < 1e-6:
                logger.warning(
                    f"⚠️  N1 feature global_mean is all zeros for '{n1_path}'! "
                    "N1 statistics may not have been properly computed during training."
                )
            elif abs(n1_std_min - 1.0) < 1e-6 and abs(n1_std_max - 1.0) < 1e-6:
                logger.warning(
                    f"⚠️  N1 feature global_std is all ones for '{n1_path}'! "
                    "N1 statistics may not have been properly computed during training."
                )
            else:
                logger.info(
                    f"✓ N1-specific feature statistics validated for '{n1_path}':\n"
                    f"  mean range=[{n1_mean_min:.3f}, {n1_mean_max:.3f}], "
                    f"  std range=[{n1_std_min:.3f}, {n1_std_max:.3f}]"
                )

        # Check N1 learnable parameters (theta/alpha ratio detection)
        # These are nn.Parameters that are optimized during training
        if hasattr(n1_module, "n1_theta_alpha_center"):
            center_val = _scalar_attr_as_float(n1_module, "n1_theta_alpha_center")
            bandwidth_val = _scalar_attr_as_float(n1_module, "n1_ratio_bandwidth")

            # Check for reasonable values
            # center: typical N1 theta/alpha ratio is 1.0-2.0, default is 1.5
            # bandwidth: Gaussian width for soft N1 detection, typical 0.3-1.0, default is 0.5
            if not (0.5 <= center_val <= 3.0):
                logger.warning(
                    f"⚠️  N1 theta/alpha center value seems unusual: {center_val:.3f} "
                    f"(expected range: 0.5-3.0, typical: 1.0-2.0)"
                )

            if not (0.1 <= bandwidth_val <= 2.0):
                logger.warning(
                    f"⚠️  N1 ratio bandwidth value seems unusual: {bandwidth_val:.3f} "
                    f"(expected range: 0.1-2.0, typical: 0.3-1.0)"
                )

            logger.info(
                f"✓ N1 learnable parameters validated for '{n1_path}':\n"
                f"  n1_theta_alpha_center={center_val:.3f} (optimal theta/alpha ratio for N1)\n"
                f"  n1_ratio_bandwidth={bandwidth_val:.3f} (Gaussian bandwidth for N1 zone)"
            )


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
    fs: int = 128  # Target sampling frequency (default: 128 Hz)
    epoch_sec: int = 30  # Epoch duration in seconds
    context_half: int = 15  # Context window half-width (epochs before/after)

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

    # Test-time augmentation
    tta: int = 0  # Number of TTA passes (0 = disabled) - legacy name
    tta_passes: int = 0  # Preferred name (kept in sync with tta for compatibility)
    tta_noise: float = 0.01
    tta_amp: float = 0.05
    tta_shift: float = 0.0
    tta_mode: str = (
        "legacy"  # legacy (mean logits) | enhanced (geometric mean of probs)
    )

    # Preprocessing calibration
    calibration_mode: str = "per_recording"  # checkpoint, per_recording, warmup
    warmup_epochs: int = 10  # Number of epochs to use for warmup calibration
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
    # "legacy" retains the historical nested TTA/MC/overlap pooling rules.
    mc_aggregation: str = "consistent"
    # Include standard MHA and custom functional SDPA attention dropout.
    mc_include_attention: bool = True
    # Match qualified names and names relative to known inference wrappers.
    # None selects existing sites throughout encoder, transformer, and classifier.
    # Use r"^classifier\." to restrict sampling to a dropout-bearing classifier.
    mc_module_pattern: str | None = None

    # Recurrent-depth test-time compute. None uses the trained step count.
    recurrent_refinement_steps: int | None = None

    # Backend / compilation performance toggles
    allow_tf32: bool = False  # TF32 matmul + cuDNN (Ampere+); tiny accuracy cost
    cudnn_benchmark: bool = False  # autotune cuDNN convs (fixed-shape conv nets)
    compile_model: bool = False  # compile via torch.compile
    compile_mode: str = "reduce-overhead"  # default|reduce-overhead|max-autotune

    # Overlap-averaging (predict_all): when True, each epoch is scored by averaging
    # the LOGITS produced for it across every overlapping context window (train/test
    # parity with --predict_all eval) instead of using only the center prediction.
    # Off by default -> byte-for-byte identical to the center-only path. Note: with
    # enhanced TTA the per-window tensor is in log-prob space, so the cross-window
    # average becomes a geometric mean of probabilities rather than a raw-logit
    # mean (functionally fine; not identical to training's logit average).
    overlap_average: bool = False

    # Averaged (EMA/SWA) weights: when True, load ``ema.module`` from the
    # checkpoint instead of the raw ``model`` weights. Training evaluates and
    # selects checkpoints on the averaged weights whenever averaging is enabled
    # (``best.ckpt.metrics.json`` records ``"weights": "ema"``), so scoring the
    # raw weights measures a different model than the one validation ranked.
    # Off by default to preserve historical numbers for existing checkpoints.
    prefer_averaged: bool = False

    def __post_init__(self) -> None:
        """Keep legacy TTA fields in sync and ensure non-negative integers."""
        tta_mode = str(getattr(self, "tta_mode", "legacy") or "legacy").strip().lower()
        if tta_mode not in {"legacy", "enhanced"}:
            logger.warning("Unknown tta_mode=%r; falling back to 'legacy'", tta_mode)
            tta_mode = "legacy"
        self.tta_mode = tta_mode

        # Prefer explicit tta_passes if provided; otherwise mirror tta
        tta_val = int(max(0, self.tta))
        tta_passes_val = int(max(0, getattr(self, "tta_passes", tta_val)))

        if tta_passes_val == 0 and tta_val > 0:
            tta_passes_val = tta_val
        elif tta_val == 0 and tta_passes_val > 0:
            tta_val = tta_passes_val

        self.tta = tta_val
        self.tta_passes = tta_passes_val
        if (
            self.recurrent_refinement_steps is not None
            and self.recurrent_refinement_steps < 1
        ):
            raise ValueError("recurrent_refinement_steps must be positive or None")

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


@dataclass
class ReasoningOptions:
    """Options for optional inference-time reasoning/post-processing."""

    # Iterative refinement
    use_iterative_refinement: bool = False
    refinement_passes: int = 3
    refinement_threshold: float = 0.7
    context_expansion: int = 2


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


# Optional per-epoch uncertainty arrays that ``postprocess_logits_with_reasoning``
# attaches to its result dict. ``FLAG_SCORE_KEYS`` are always present (computed
# free on the deterministic pass). ``score_recording`` propagates whichever of
# these are present.
FLAG_SCORE_KEYS: tuple[str, ...] = ("flag_margin", "flag_entropy", "flag_maxprob")
MC_SCORE_KEYS: tuple[str, ...] = (
    "mc_probability_variance",
    "mc_expected_entropy",
    "mc_predictive_entropy",
)
# These moments describe individual sampled predictions, before reasoning.
# With TTA/overlap they include augmentation/context variation, not only dropout.
INFERENCE_ALGORITHM_REVISION = "mc-coverage-aggregation-v2"


def _get_saved_context_half(model_config: Mapping[str, object]) -> int | None:
    """Read inference context width from current or legacy checkpoint metadata."""
    inference_metadata = model_config.get("inference_metadata")
    if isinstance(inference_metadata, Mapping):
        context_half = inference_metadata.get("context_half")
        if isinstance(context_half, int) and context_half >= 0:
            return context_half

    model_kwargs = model_config.get("model_kwargs")
    if isinstance(model_kwargs, Mapping):
        context_half = model_kwargs.get("context_half")
        if isinstance(context_half, int) and context_half >= 0:
            return context_half
        context_epochs = model_kwargs.get("context_epochs")
        if (
            isinstance(context_epochs, int)
            and context_epochs > 0
            and context_epochs % 2
        ):
            return (context_epochs - 1) // 2

    return None


def _sync_options_with_checkpoint_metadata(
    options: ScoreOptions,
    checkpoint_dict: dict,
) -> int | None:
    """Adjust sampling-related options to match checkpoint metadata.

    Returns the expected samples-per-epoch from the checkpoint if available.
    """
    target_fs: float | None = None
    target_source = None
    expected_time_len: int | None = None

    # Prefer explicit preprocessing config if present
    preproc_cfg = checkpoint_dict.get("preprocessing_config")
    if isinstance(preproc_cfg, dict):
        resampling_cfg = preproc_cfg.get("resampling", {})
        if isinstance(resampling_cfg, dict):
            target_fs = resampling_cfg.get("target_fs")
            if target_fs is not None:
                target_source = "checkpoint preprocessing_config.resampling.target_fs"

    # Fall back to model_config -> model_kwargs
    model_config = checkpoint_dict.get("model_config")
    if isinstance(model_config, dict):
        model_kwargs = model_config.get("model_kwargs", {})
        if isinstance(model_kwargs, dict):
            time_len = model_kwargs.get("time_len")
            if isinstance(time_len, (int, float)):
                expected_time_len = int(time_len)

            # Prefer explicit fs from model_kwargs (always saved after fix)
            ckpt_fs = model_kwargs.get("fs")
            if target_fs is None and isinstance(ckpt_fs, (int, float)) and ckpt_fs > 0:
                target_fs = float(ckpt_fs)
                target_source = "checkpoint model_config.model_kwargs.fs"
            # Fall back to deriving from time_len / epoch_sec
            elif (
                target_fs is None
                and expected_time_len is not None
                and options.epoch_sec
            ):
                target_fs = float(expected_time_len) / float(options.epoch_sec)
                target_source = "checkpoint model_config.model_kwargs.time_len"

    # Apply target sampling frequency if we found one
    if target_fs is not None:
        try:
            target_fs_float = float(target_fs)
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring non-numeric target sampling rate %r from checkpoint",
                target_fs,
            )
        else:
            target_fs_int = int(round(target_fs_float))
            if target_fs_int <= 0:
                logger.warning(
                    "Ignoring invalid target sampling rate %.3f Hz from checkpoint",
                    target_fs_float,
                )
            else:
                if abs(target_fs_float - target_fs_int) > 1e-3:
                    logger.warning(
                        "Checkpoint target sampling rate %.3f Hz is non-integer; rounding to %d Hz",
                        target_fs_float,
                        target_fs_int,
                    )
                if options.fs != target_fs_int:
                    logger.info(
                        "Adjusting inference sampling rate from %d Hz to %d Hz based on %s",
                        options.fs,
                        target_fs_int,
                        target_source,
                    )
                    options.fs = target_fs_int

    # Ensure epoch duration matches checkpoint expectation when possible
    samples_per_epoch = options.fs * options.epoch_sec
    if expected_time_len is not None:
        if samples_per_epoch != expected_time_len:
            if expected_time_len % options.fs == 0:
                new_epoch_sec = expected_time_len // options.fs
                if new_epoch_sec != options.epoch_sec:
                    logger.info(
                        "Adjusting epoch duration from %d s to %d s to match checkpoint configuration",
                        options.epoch_sec,
                        new_epoch_sec,
                    )
                    options.epoch_sec = new_epoch_sec
                    samples_per_epoch = options.fs * options.epoch_sec
            elif target_source is not None and "time_len" not in target_source:
                # fs came from an explicit source (model_kwargs.fs or
                # preprocessing_config) but time_len is inconsistent —
                # stale checkpoint metadata.  Override with fs * epoch_sec.
                logger.warning(
                    "Checkpoint time_len=%d is inconsistent with fs=%d x "
                    "epoch_sec=%d = %d; ignoring stale time_len (fs sourced "
                    "from %s).",
                    expected_time_len,
                    options.fs,
                    options.epoch_sec,
                    samples_per_epoch,
                    target_source,
                )
                expected_time_len = samples_per_epoch
            else:
                raise InferencePreprocessingError(
                    "Checkpoint expects "
                    f"{expected_time_len} samples per epoch but current settings "
                    f"produce {samples_per_epoch} (fs={options.fs}, epoch_sec={options.epoch_sec})."
                )

    # Sync context_half from checkpoint if available
    if isinstance(model_config, dict):
        model_kwargs = model_config.get("model_kwargs", {})
        if isinstance(model_kwargs, dict):
            if model_kwargs.get("epoch_encoder_variant") in {
                "learned_feature_axial",
                "learned_feature_axial_v2",
            }:
                context_epochs = model_kwargs.get("context_epochs")
                if (
                    not isinstance(context_epochs, int)
                    or context_epochs < 1
                    or context_epochs % 2 == 0
                ):
                    raise InferencePreprocessingError(
                        "learned_feature_axial requires a positive odd context_epochs"
                    )
                ckpt_context_half = (context_epochs - 1) // 2
            else:
                ckpt_context_half = _get_saved_context_half(model_config)
            if isinstance(ckpt_context_half, int) and ckpt_context_half >= 0:
                if options.context_half != ckpt_context_half:
                    logger.info(
                        "Adjusting inference context_half from %d to %d based on "
                        "checkpoint model_config (training value)",
                        options.context_half,
                        ckpt_context_half,
                    )
                    options.context_half = ckpt_context_half

    return expected_time_len


def _get_model_expected_channels(model: nn.Module) -> int | None:
    """Recursively extract the expected number of input channels from a model."""
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

    # Handle wrappers (e.g., ModelWithPreproc -> .model)
    inner_model = getattr(model, "model", None)
    if isinstance(inner_model, nn.Module):
        return _get_model_expected_channels(inner_model)
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


def preprocess_signals(
    aligned_data: np.ndarray,
    channel_types: list[str],
    fs_orig: float,
    options: ScoreOptions,
    proc_cfg: ProcCfg | None = None,
) -> np.ndarray:
    """Preprocess signals: resample to the model’s target frequency.

    Filtering is intentionally disabled for inference/testing because most input
    data has already been filtered during offline preprocessing. This avoids
    double-filtering (and phase distortions) while still matching the training
    sampling rate.
    """
    n_channels = aligned_data.shape[0]

    # Resample to target frequency when necessary
    if abs(fs_orig - options.fs) > 0.1:
        logger.info(f"Resampling from {fs_orig} Hz to {options.fs} Hz")
        aligned_data = np.array(
            [
                resample_signal(aligned_data[i], fs_orig, options.fs)
                for i in range(n_channels)
            ]
        )
    return aligned_data


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


def create_epoch_windows_for_indices(
    data: np.ndarray,
    indices: np.ndarray,
    fs: int,
    epoch_sec: int,
    context_half: int,
) -> torch.Tensor:
    """Create epoch context windows only for selected indices.

    Returns a tensor shaped (n_sel, context_len, n_channels, samples_per_epoch).
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(
            f"Expected data shape (n_channels, n_samples), got {data.shape}"
        )

    indices_arr = np.asarray(indices, dtype=np.int64)
    samples_per_epoch = int(fs) * int(epoch_sec)
    if samples_per_epoch <= 0:
        raise ValueError(f"Invalid sampling parameters: fs={fs}, epoch_sec={epoch_sec}")

    n_channels, n_samples = data.shape
    n_epochs = n_samples // samples_per_epoch
    if n_epochs <= 0:
        raise ValueError("Recording is shorter than one complete epoch")

    context_len = 2 * int(context_half) + 1
    if indices_arr.size == 0:
        return torch.empty(
            (0, context_len, n_channels, samples_per_epoch), dtype=torch.float32
        )

    # Truncate to complete epochs
    data = data[:, : n_epochs * samples_per_epoch]

    # (n_epochs, n_channels, samples_per_epoch)
    data_epochs = data.reshape(n_channels, n_epochs, samples_per_epoch).transpose(
        1, 0, 2
    )

    windows = []
    for idx in indices_arr.tolist():
        if idx < 0 or idx >= n_epochs:
            raise IndexError(f"Epoch index {idx} out of range for n_epochs={n_epochs}")

        start_idx = max(0, idx - context_half)
        end_idx = min(n_epochs, idx + context_half + 1)
        window = data_epochs[start_idx:end_idx]

        actual_len = window.shape[0]
        if actual_len < context_len:
            missing_before = max(0, context_half - idx)
            missing_after = max(0, (idx + context_half + 1) - n_epochs)
            window = np.pad(
                window,
                ((missing_before, missing_after), (0, 0), (0, 0)),
                mode="constant",  # Zero-fill to match training behavior
            )
        windows.append(window)

    batch_array = np.stack(windows, axis=0)
    return torch.from_numpy(batch_array).float()


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


def _find_patch_checkpoint_keys(keys: list[str]) -> list[str]:
    """Return normalized keys that indicate removed patch-token support.

    The supported epoch-patch grid tokenizer (``epoch_tokenization="grid"``)
    nests its parameters under ``epoch_grid.`` (e.g.
    ``epoch_grid.patch_tokenizer.*``), and the order-aware epoch pooling head
    lives under the epoch encoder's ``pool.``. Both are excluded here so
    legitimate checkpoints load, while genuinely removed features -- the old
    ``patch_frontend.`` duplicate transformer stack, the legacy top-level
    tokenizer, the dual-stream variants -- are still rejected with a clear
    message.
    """
    patch_markers = (
        # The removed flat patch front-end, which owned a duplicate transformer.
        "patch_frontend.",
        "center_patch_readout.",
        # Legacy pre-grid patch features.
        "patch_tokenizer.patch_pe.",
        "subepoch_classifier.",
        "epoch_pool_attn.",
        "patch_feature_extractor.",
        "dual_stream_tokenizer.",
        "dual_stream_classifier.",
        "dual_patch_feature_extractor.",
        "intra_epoch_patch_transformer.",
    )
    return [
        key
        for key in keys
        if "epoch_grid." not in key and any(marker in key for marker in patch_markers)
    ]


def _raise_if_patch_checkpoint_keys(keys: list[str]) -> None:
    """Reject checkpoints that depend on removed patch-token features."""
    matches = _find_patch_checkpoint_keys(keys)
    if not matches:
        return
    sample = ", ".join(matches[:5])
    hint = (
        "The flat 'patch' tokenization mode was replaced by the epoch-patch grid "
        "(--epoch_tokenization grid), whose parameters live under 'epoch_grid.'. "
        if any(
            "patch_frontend." in key or "center_patch_readout." in key
            for key in matches
        )
        else ""
    )
    raise ValueError(
        "Patch-based checkpoints are no longer supported in this repository. "
        "Detected removed patch-tokenization or patch-feature parameters "
        f"({sample}). {hint}"
        "Use an epoch- or grid-tokenization checkpoint, or retrain."
    )


def _has_learned_feature_axial_keys(keys: list[str]) -> bool:
    return any(
        "axial_backbone.blocks." in key
        or "center_feature_time_readout." in key
        or "axial_output_projection." in key
        for key in keys
    )


def _has_learned_feature_axial_v2_keys(keys: list[str]) -> bool:
    """Detect v2 only to validate explicit metadata, never to infer geometry."""
    return any("axial_v2_summary." in key or "axial_v2_readout." in key for key in keys)


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


def _infer_recurrent_refinement_steps(
    state_dict: Mapping[str, Any],
    key_map: Mapping[str, str],
) -> int:
    """Infer recurrent depth from its persistent architecture buffer."""
    tensor = _get_tensor_from_clean_key(
        state_dict,
        key_map,
        "recurrent_refiner.configured_steps",
    )
    if tensor is None:
        return 0
    if tensor.numel() != 1:
        raise ValueError(
            "recurrent_refiner.configured_steps must contain exactly one value"
        )
    steps = int(tensor.detach().cpu().item())
    if steps < 1:
        raise ValueError(
            "recurrent_refiner.configured_steps must be positive when present"
        )
    return steps


def _reconcile_recurrent_refinement_metadata(
    model_kwargs: dict[str, Any],
    state_dict: Mapping[str, Any],
    key_map: Mapping[str, str],
) -> None:
    """Make recurrent-refiner state authoritative over stale metadata."""
    inferred_steps = _infer_recurrent_refinement_steps(state_dict, key_map)
    configured_steps = int(model_kwargs.get("recurrent_refinement_steps", 0))
    if configured_steps != inferred_steps:
        logger.warning(
            "Checkpoint recurrent_refinement_steps mismatch: config=%r, "
            "state_dict=%r. Using state-dict architecture.",
            configured_steps,
            inferred_steps,
        )
    model_kwargs["recurrent_refinement_steps"] = inferred_steps


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


def _infer_model_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model configuration from state dict keys (legacy checkpoint support).

    Args:
        state_dict: Model state dictionary

    Returns:
        Model configuration dictionary
    """
    keys, key_map = _build_clean_key_map(state_dict)

    # Debug logging: show sample keys
    logger.info(f"State dict has {len(keys)} keys")
    logger.info(f"Sample keys (first 5): {keys[:5]}")
    _raise_if_patch_checkpoint_keys(keys)

    # Log specific checks for transformer keys
    transformer_keys = [
        k for k in keys if "enc_layers" in k or "transformer.layers" in k
    ]
    logger.info(
        "Found %d keys containing transformer layer markers", len(transformer_keys)
    )
    if transformer_keys:
        logger.info(f"Sample transformer keys: {transformer_keys[:3]}")

    # Infer model type from architecture signatures
    has_transformer = any("enc_layers" in k or "transformer.layers" in k for k in keys)
    has_tcn = any("tcn" in k.lower() for k in keys)

    logger.info(
        f"Architecture signatures: " f"transformer={has_transformer}, tcn={has_tcn}"
    )

    if has_transformer:
        model_type = "TransformerContextNet"
    else:
        raise ValueError(
            "Could not infer model type from checkpoint. "
            "No recognized architecture signatures found in keys. "
            "Supported model types: TransformerContextNet"
        )

    logger.info(f"Inferred model type: {model_type}")

    # Extract dimensions from state dict
    config: dict[str, Any] = {"model_type": model_type}

    # Try to extract in_ch from channel_embedding or first conv layer
    for key in keys:
        if "channel_embedding.channel_embed" in key:
            config["in_ch"] = state_dict[key_map[key]].shape[2]
            break
        if "channel_embedding.weight" in key:
            config["in_ch"] = state_dict[key_map[key]].shape[0]
            break
        elif "epoch_encoder.stem.0.conv.weight" in key or "stem.0.conv.weight" in key:
            config["in_ch"] = state_dict[key_map[key]].shape[1]
            break

    # Detect epoch-patch grid tokenization (epoch_tokenization="grid").
    # Reconstruct its geometry so a checkpoint lacking model_config metadata still
    # rebuilds an identical grid. slot_pos is [1, P+1, d_model] (slot 0 is the
    # summary token); patch_proj.weight is [d_model, d_token]. The presence of a
    # per-head epoch-lag bias identifies the relative attention mode.
    if any("epoch_grid.patch_tokenizer." in k for k in keys):
        config["epoch_tokenization"] = "grid"
        for key in keys:
            if key.endswith("epoch_grid.slot_pos"):
                config["patches_per_epoch"] = int(state_dict[key_map[key]].shape[1]) - 1
                break
        for key in keys:
            if key.endswith("epoch_grid.patch_proj.weight"):
                config["patch_token_dim"] = int(state_dict[key_map[key]].shape[1])
                break
        config["grid_attention_mode"] = (
            "relative_factorized"
            if any("epoch_lag_bias" in k for k in keys)
            else "absolute"
        )
        logger.info(
            "Detected epoch-patch grid tokenization: patches_per_epoch=%s, "
            "patch_token_dim=%s, grid_attention_mode=%s",
            config.get("patches_per_epoch"),
            config.get("patch_token_dim"),
            config.get("grid_attention_mode"),
        )

    has_confidence_head = any(k.startswith("confidence_head.") for k in keys)

    # PerPositionSleepHead (predict-all / all-positions training) uses a weight-
    # shared MLP (classifier.mlp.{0,3}) + norm (classifier.norm), distinct from the
    # default head (classifier.head/input_proj/res_*).
    # extract_transformer_config does not record use_per_position_head, so a
    # checkpoint trained with it must be detected here or its head weights are
    # silently dropped -> untrained classifier -> single stage collapse.
    head_family = _detect_classifier_head_family(keys)
    has_per_position_head = head_family == "per_position"
    if has_per_position_head:
        config["use_per_position_head"] = True
        # Optional depthwise mix adds classifier.local.* only when enabled.
        config["head_use_local_mix"] = any("classifier.local." in k for k in keys)
        # LayerNorm carries a bias; VariancePreservingRMSNorm (rmsnorm) does not.
        config["head_norm"] = (
            "layernorm"
            if any(k.endswith("classifier.norm.bias") for k in keys)
            else "rmsnorm"
        )
        for key in keys:
            if key.endswith("classifier.mlp.3.weight"):
                config["num_classes"] = int(state_dict[key_map[key]].shape[0])
                break
        logger.info(
            "Detected PerPositionSleepHead: local_mix=%s, norm=%s, num_classes=%s",
            config["head_use_local_mix"],
            config["head_norm"],
            config.get("num_classes"),
        )
    elif head_family is not None:
        # LinearSleepHead vs the default residual MLP. Recorded either way so the
        # rebuilt model matches the stored weights; the linear head's
        # classifier.head.weight is [C, d_model] and would otherwise shape-clash
        # with the default head's [C, d_model // 2].
        config["classifier_head"] = head_family
        if head_family == "linear":
            for key in keys:
                if key.endswith("classifier.head.weight"):
                    weight = state_dict[key_map[key]]
                    config["num_classes"] = int(weight.shape[0])
                    logger.info(
                        "Detected LinearSleepHead: classifier.head.weight=%s "
                        "(num_classes=%d, d_model=%d)",
                        tuple(weight.shape),
                        int(weight.shape[0]),
                        int(weight.shape[1]),
                    )
                    break

    if "eng_feature_dim" not in config:
        for key in keys:
            if key.endswith("classifier.eng_router_proj.0.weight"):
                config["eng_feature_dim"] = int(state_dict[key_map[key]].shape[1])
                logger.info(
                    "Extracted eng_feature_dim=%d from %s",
                    config["eng_feature_dim"],
                    key,
                )
                break

    if has_confidence_head:
        config["use_confidence_head"] = True
        logger.info("Detected confidence head in checkpoint")
        hidden_key = "confidence_head.net.1.weight"
        if hidden_key in key_map:
            config["confidence_head_hidden_dim"] = int(
                state_dict[key_map[hidden_key]].shape[0]
            )
            logger.info(
                "Extracted confidence_head_hidden_dim=%d from %s",
                config["confidence_head_hidden_dim"],
                hidden_key,
            )
    else:
        config["use_confidence_head"] = False

    # Extract num_classes from final classifier layer
    if "num_classes" not in config:
        inferred_num_classes, source_key = _infer_num_classes_from_classifier_keys(
            state_dict, keys, key_map
        )
        if inferred_num_classes is not None and source_key is not None:
            config["num_classes"] = inferred_num_classes
            logger.info(
                "Extracted num_classes=%d from %s",
                config["num_classes"],
                source_key,
            )

    # Set reasonable defaults for other parameters
    config["time_len"] = 3840  # Standard 30s epoch at 128Hz

    # Model-specific defaults
    if model_type == "TransformerContextNet":
        config.update(_infer_context_mode_metadata(state_dict, keys, key_map))
        config["recurrent_refinement_steps"] = _infer_recurrent_refinement_steps(
            state_dict,
            key_map,
        )
        proj_key = "proj.weight"
        checkpoint_d_model = _extract_checkpoint_d_model(
            state_dict, keys=keys, key_map=key_map
        )
        if checkpoint_d_model is not None:
            config["d_model"] = int(checkpoint_d_model)
            logger.info(
                "Extracted d_model=%d from checkpoint tensor shapes", config["d_model"]
            )

        # Extract dim_ff from either a GELU or gated SwiGLU feedforward layer.
        for key in keys:
            if (
                "enc_layers.0.linear1.weight" in key
                or "transformer.layers.0.linear1.weight" in key
                or "enc_layers.0.w_gate.weight" in key
                or "transformer.layers.0.w_gate.weight" in key
            ):
                # linear1/w_gate weight shape is [dim_ff, d_model].
                config["dim_ff"] = state_dict[key_map[key]].shape[0]
                logger.info(f"Extracted dim_ff={config['dim_ff']} from {key}")
                break

        # Count transformer layers from standard MHA self-attention keys.
        inferred_num_layers = _infer_transformer_num_layers_from_keys(keys)
        config["num_layers"] = (
            inferred_num_layers if inferred_num_layers is not None else 4
        )
        logger.info(f"Detected {config['num_layers']} transformer layers")

        # Extract attention heads from first layer
        for key in keys:
            if (
                "enc_layers.0.self_attn.in_proj_weight" in key
                or "transformer.layers.0.self_attn.in_proj_weight" in key
            ):
                # in_proj_weight shape is [3 * d_model, d_model] for QKV
                # nhead = d_model / head_dim, but head_dim not directly available
                # Use the relative-bias head axis when available; otherwise the
                # historical state-only fallback remains eight heads.
                config.setdefault("nhead", 8)
                break

        # Check for engineered features using multiple detection methods
        # Method 1: Check for fusion-specific parameters (most reliable)
        has_cnn_weight = any("epoch_encoder.cnn_weight" in k for k in keys)
        has_eng_weight = any("epoch_encoder.eng_weight" in k for k in keys)
        has_gate = any("epoch_encoder.gate" in k for k in keys)

        # Method 2: Check for sequential fusion mode
        has_feature_extractor = any("feature_extractor." in k for k in keys)
        has_feat_projection = any("feat_projection." in k for k in keys)
        has_sequential_fusion = any("sequential_fusion." in k for k in keys)

        # Method 3: Check transformer projection input dimension
        proj_input_dim = None
        if proj_key in key_map:
            proj_tensor = state_dict[key_map[proj_key]]
            if isinstance(proj_tensor, torch.Tensor) and proj_tensor.ndim == 2:
                proj_input_dim = int(proj_tensor.shape[1])
                logger.info(f"Found proj.weight with input dim: {proj_input_dim}")

        # Engineered features are present if EITHER:
        # 1. Fusion parameters exist (cnn_weight, eng_weight, or gate)
        # 2. Sequential fusion parameters exist (feature_extractor, feat_projection, sequential_fusion)
        # 3. Proj input dimension is larger than expected CNN output (256)
        has_fusion_params = has_cnn_weight or has_eng_weight or has_gate
        has_sequential_params = (
            has_feature_extractor or has_feat_projection or has_sequential_fusion
        )
        has_large_proj = proj_input_dim is not None and proj_input_dim > 256

        if has_fusion_params or has_sequential_params or has_large_proj:
            config["use_feature_extraction"] = True
            if has_large_proj:
                logger.info(
                    f"Detected engineered features: proj input dim = {proj_input_dim} (> 256)"
                )
            if has_fusion_params:
                logger.info(
                    f"Detected engineered features: fusion parameters present (cnn_weight={has_cnn_weight}, eng_weight={has_eng_weight}, gate={has_gate})"
                )
            if has_sequential_params:
                logger.info(
                    f"Detected engineered features: sequential parameters present (feature_extractor={has_feature_extractor}, feat_projection={has_feat_projection}, sequential_fusion={has_sequential_fusion})"
                )

            # Detect fusion mode from presence of specific parameters
            # Sequential mode takes priority over other fusion modes.
            if has_sequential_params:
                config["fusion_mode"] = "sequential"
                logger.info("Detected sequential fusion mode")
            elif has_cnn_weight:
                config["fusion_mode"] = "learned_weight"
                logger.info("Detected learned_weight fusion mode")
            elif has_gate:
                config["fusion_mode"] = "gated"
                logger.info("Detected gated fusion mode")
            else:
                config["fusion_mode"] = "concat"
                logger.info("Detected concat fusion mode (default)")

            # Detect simple_feature_projection (Linear vs BottleneckProjection)
            # BottleneckProjection has keys like: feat_projection.bottleneck.0.weight
            # Simple Linear projection has keys like: feat_projection.weight
            has_bottleneck_projection = any(
                "feat_projection.bottleneck" in k for k in keys
            )
            has_simple_projection = any(
                k == "feat_projection.weight" or k.endswith(".feat_projection.weight")
                for k in keys
            )

            if has_bottleneck_projection:
                config["simple_feature_projection"] = False
                logger.info(
                    "Detected BottleneckProjection (simple_feature_projection=False)"
                )
            elif has_simple_projection:
                config["simple_feature_projection"] = True
                logger.info(
                    "Detected simple Linear projection (simple_feature_projection=True)"
                )
            else:
                # Default to False (BottleneckProjection) when engineered features are used
                config["simple_feature_projection"] = False
                logger.info(
                    "Defaulting to BottleneckProjection (simple_feature_projection=False)"
                )

        # CNN feature guidance (FiLM-style injection) changes epoch encoder parameters.
        config["cnn_feature_guidance"] = any("feature_inject_stage" in k for k in keys)

        # Defaults for other params (only set if not already extracted)
        config.setdefault("d_model", 256)
        config.setdefault("dim_ff", 256)
        config.setdefault("nhead", 8)
        config.setdefault("num_layers", 4)
        config.setdefault("cnn_dropout", 0.1)
        config.setdefault("head_dropout", 0.1)
        config.setdefault("classifier_dropout", 0.2)
        config.setdefault("norm", "bn")
        config.setdefault("sdp_backend", "auto")
        config.setdefault("intra_num_layers", 2)
        config.setdefault("intra_nhead", 4)
        config.setdefault("intra_ff_mult", 2.0)

        # === CENTER MASK TOKEN DETECTION ===
        # mask_token is a learnable parameter created when center_mask_prob > 0
        has_mask_token = any(
            k == "mask_token" or k.endswith(".mask_token") for k in keys
        )
        if has_mask_token:
            # Check if it's an actual tensor (not a None buffer)
            mask_token_key = next(
                (k for k in keys if k == "mask_token" or k.endswith(".mask_token")),
                None,
            )
            if (
                mask_token_key
                and state_dict.get(key_map.get(mask_token_key, mask_token_key))
                is not None
            ):
                config["center_mask_prob"] = 0.15
                logger.info(
                    "Detected mask_token in checkpoint, setting center_mask_prob=0.15"
                )

        # === NEIGHBOR PREDICTION HEAD DETECTION ===
        # neighbor_head.neighbor_heads is a ModuleList with 2*n_neighbors heads
        neighbor_head_keys = [
            k for k in keys if "neighbor_head.neighbor_heads." in k and ".weight" in k
        ]
        if neighbor_head_keys:
            # Count unique head indices to determine n_neighbors
            head_indices = set()
            for key in neighbor_head_keys:
                # Keys: neighbor_head.neighbor_heads.0.weight, neighbor_head.neighbor_heads.1.weight, etc.
                parts = key.split(".")
                try:
                    idx = parts.index("neighbor_heads")
                    if idx + 1 < len(parts):
                        head_indices.add(int(parts[idx + 1]))
                except (ValueError, IndexError):
                    pass

            num_heads = len(head_indices)
            n_neighbors = num_heads // 2 if num_heads >= 2 else 2
            config["use_neighbor_prediction"] = True
            config["neighbor_prediction_n"] = n_neighbors
            logger.info(
                f"Detected neighbor_head with {num_heads} heads, setting neighbor_prediction_n={n_neighbors}"
            )

        # === SLEEPFM FUSION DETECTION ===
        has_epoch_feature_fusion = any("epoch_feature_fusion." in k for k in keys)
        has_sleepfm_feature_fusion = any(
            "sleepfm_fusion.feature_fusion" in k for k in keys
        )
        has_sleepfm_temporal_pool = any(
            "sleepfm_fusion.temporal_pool" in k for k in keys
        )
        has_sleepfm_pool_fusion = (
            has_sleepfm_feature_fusion and has_sleepfm_temporal_pool
        )
        has_sleepfm_fusion = has_epoch_feature_fusion or has_sleepfm_pool_fusion

        if has_sleepfm_fusion:
            config["use_sleepfm_fusion"] = True
            if has_epoch_feature_fusion and has_sleepfm_pool_fusion:
                logger.warning(
                    "Detected both epoch_feature_fusion and sleepfm_fusion keys in checkpoint; "
                    "preferring sleepfm_mode='fuse_only'."
                )
            if has_epoch_feature_fusion:
                config["sleepfm_mode"] = "fuse_only"
                logger.info(
                    "Detected epoch_feature_fusion module in checkpoint (sleepfm_mode='fuse_only')"
                )
            else:
                config["sleepfm_mode"] = "pool"
                logger.info(
                    "Detected sleepfm_fusion module in checkpoint (sleepfm_mode='pool')"
                )

                # Detect temporal pool type from submodule structure
                has_hierarchical_pool = any(
                    "sleepfm_fusion.temporal_pool.local_pool" in k for k in keys
                )
                if has_hierarchical_pool:
                    config["sleepfm_temporal_pool_type"] = "hierarchical"
                    logger.info("  Detected hierarchical temporal pooling")
                else:
                    config["sleepfm_temporal_pool_type"] = "attention"
                    logger.info("  Detected attention temporal pooling")

                # Detect center bias type from temporal pool parameters
                has_learned_bias = any(
                    "sleepfm_fusion.temporal_pool.position_bias" in k for k in keys
                )
                if has_learned_bias:
                    config["sleepfm_center_bias"] = "learned"
                    logger.info(
                        "  Detected learned center bias (position_bias parameter)"
                    )
                else:
                    config.setdefault("sleepfm_center_bias", "gaussian")

            # Set reasonable defaults for other SleepFM params
            config.setdefault("sleepfm_fusion_num_heads", 4)
            config.setdefault("sleepfm_center_bias_strength", 2.0)
            config.setdefault("sleepfm_temperature", 1.0)

        # === LEARNABLE POSITIONAL ENCODING DETECTION ===
        has_learnable_pe = any(
            k == "pos_encoding" or k.endswith(".pos_encoding") for k in keys
        )
        if has_learnable_pe:
            config["use_learnable_pe"] = True
            logger.info(
                "Detected learnable pos_encoding in checkpoint, "
                "setting use_learnable_pe=True"
            )

    # Set default in_ch and num_classes if not found
    config.setdefault("in_ch", 8)  # Match current 8-channel canonical list
    config.setdefault("num_classes", 5)

    # Wrap in model_kwargs for consistency
    model_kwargs = {}
    for key, value in config.items():
        if key != "model_type":
            model_kwargs[key] = value

    return {
        "model_type": config["model_type"],
        "model_kwargs": model_kwargs,
    }


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
    # - .filt: BlurPool1D low-pass filter coefficients (fixed [1,2,1] or [1,2,4,2,1])
    # - .num_batches_tracked: BatchNorm running statistics tracker
    safely_skippable_suffixes = (".filt", ".num_batches_tracked")

    # Components that are optional at inference time (training-only auxiliary heads):
    # - transition_head: Binary transition prediction head for explicit boundary supervision
    # - cnn_reconstruction_head: CNN feature reconstruction (pretraining objective)
    # - eng_reconstruction_head: Engineered feature reconstruction (pretraining objective)
    #   Only used during training for auxiliary losses, not needed at inference
    safely_skippable_prefixes = (
        "transition_head.",
        "cnn_reconstruction_head.",
        "eng_reconstruction_head.",
    )

    def _is_safely_skippable(key: str) -> bool:
        """Check if a missing key can be safely ignored."""
        if "source_convention." in key:
            return False
        if key.endswith(safely_skippable_suffixes):
            return True
        if key.endswith("._extra_state"):
            return True
        # Check for training-only components (handle both raw keys and prefixed keys)
        # e.g., "transition_head.0.weight" or "model.transition_head.0.weight"
        key_parts = key.split(".")
        for prefix in safely_skippable_prefixes:
            prefix_name = prefix.rstrip(".")
            if prefix_name in key_parts:
                return True
        return False

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
            "engineered feature settings)."
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
        warning = [
            (
                f"Checkpoint '{checkpoint_label}' contains {len(unexpected_keys)} "
                f"parameters not present in {model_label}."
            ),
            "These weights were ignored when loading the checkpoint. "
            "Ensure the current model configuration matches the training "
            "architecture if results look suspicious.",
            f"Unexpected keys: {_summarize(unexpected_keys)}",
        ]
        logger.warning("=" * 80)
        for line in warning:
            logger.warning(line)
        logger.warning("=" * 80)


def _slow_wave_occupancy_head_tensors(
    state_dict: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Return SWS-head tensors independent of compile/DDP/model prefixes."""
    marker = "slow_wave_occupancy_head."
    tensors: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and marker in key and isinstance(value, torch.Tensor):
            tensors[key.split(marker, 1)[1]] = value
    return tensors


def _validate_slow_wave_occupancy_checkpoint_contract(
    state_dict: Mapping[str, Any],
    model_kwargs: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> bool:
    """Validate that SWS-head tensors and reconstruction metadata are complete."""
    head_tensors = _slow_wave_occupancy_head_tensors(state_dict)
    saved_config = model_kwargs.get("slow_wave_occupancy_config")
    saved_weight = model_kwargs.get("slow_wave_occupancy_loss_weight")
    carries_head = bool(head_tensors)
    carries_metadata = saved_config is not None or saved_weight is not None
    if not carries_head and not carries_metadata:
        return False
    if not carries_head:
        raise InferenceModelLoadError(
            f"Checkpoint '{checkpoint_label}' carries slow-wave occupancy "
            "metadata but no learned head tensors."
        )
    if not isinstance(saved_config, Mapping) or saved_weight is None:
        raise InferenceModelLoadError(
            f"Checkpoint '{checkpoint_label}' carries slow-wave occupancy head "
            "tensors without complete reconstruction config and loss weight."
        )

    linear_weight = head_tensors.get("linear.weight")
    linear_bias = head_tensors.get("linear.bias")
    if linear_weight is None or linear_bias is None:
        raise InferenceModelLoadError(
            f"Checkpoint '{checkpoint_label}' has an incomplete slow-wave "
            "occupancy head (linear.weight and linear.bias are required)."
        )
    in_dim = int(saved_config.get("in_dim", -1))
    eeg_channels = int(saved_config.get("eeg_channels", 0))
    thresholds = tuple(saved_config.get("thresholds", ()))
    expected_outputs = eeg_channels * len(thresholds)
    if tuple(linear_weight.shape) != (expected_outputs, in_dim) or tuple(
        linear_bias.shape
    ) != (expected_outputs,):
        raise InferenceModelLoadError(
            f"Checkpoint '{checkpoint_label}' slow-wave occupancy metadata does "
            "not match its learned head tensor geometry."
        )
    if float(saved_weight) < 0.0:
        raise InferenceModelLoadError(
            f"Checkpoint '{checkpoint_label}' has a negative slow-wave occupancy "
            "loss weight."
        )
    return True


def _verify_loaded_slow_wave_occupancy_head(
    model: nn.Module,
    checkpoint_state: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> bool:
    """Verify bit-exact restoration of every learned SWS-head tensor."""
    expected = _slow_wave_occupancy_head_tensors(checkpoint_state)
    if not expected:
        return False
    actual = _slow_wave_occupancy_head_tensors(model.state_dict())
    if actual.keys() != expected.keys():
        raise InferenceModelLoadError(
            f"Checkpoint '{checkpoint_label}' slow-wave occupancy head keys were "
            "not reconstructed exactly."
        )
    for key, expected_tensor in expected.items():
        actual_tensor = actual[key]
        comparable = actual_tensor.detach().cpu().to(dtype=expected_tensor.dtype)
        reference = expected_tensor.detach().cpu()
        if not torch.equal(comparable, reference):
            max_abs_diff = float((comparable - reference).abs().max().item())
            raise InferenceModelLoadError(
                f"Checkpoint '{checkpoint_label}' slow-wave occupancy tensor "
                f"'{key}' was not restored exactly (max_abs_diff={max_abs_diff:.6g})."
            )
    return True


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


def _infer_modern_tcn_structure(
    state_dict: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    """Recover a ModernTCN-family encoder's structure from its weight shapes.

    Used only when a checkpoint carries no ``model_config``. Everything returned
    here is unambiguous from a tensor shape; behaviour-only options
    (``linear_stem``, and ``magnitude_mode`` within the ``abs``/``none`` pair)
    leave no trace in the weights and are deliberately not guessed.

    Args:
        state_dict: Checkpoint weights, optionally prefixed (``model.``,
            ``_orig_mod.``, ``module.``).
        variant: ``"modern_sleep_tcn"`` or ``"channel_indep_patch"``.

    Returns:
        Constructor kwargs, plus a ``"widths"`` entry the caller routes to
        ``cnn_widths``. Empty when the trunk cannot be read.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    trunk: dict[int, dict[str, tuple[int, ...]]] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not hasattr(value, "shape"):
            continue
        shape = tuple(int(dim) for dim in value.shape)
        marker = key.split("epoch_encoder.", 1)
        if len(marker) != 2:
            continue
        suffix = marker[1]
        shapes[suffix] = shape
        match = re.fullmatch(r"trunk\.(\d+)\.(dwconv|pw_expand)\.weight", suffix)
        if match:
            trunk.setdefault(int(match.group(1)), {})[match.group(2)] = shape

    if not trunk:
        return {}

    order = sorted(trunk)
    kernels = [trunk[i]["dwconv"][2] for i in order if "dwconv" in trunk[i]]
    block_widths = [trunk[i]["dwconv"][0] for i in order if "dwconv" in trunk[i]]
    if len(kernels) != len(order):
        return {}

    # Run-length encode (kernel, width) pairs back into trunk groups.
    groups: list[list[int]] = []  # [kernel, width, count]
    for kernel, width in zip(kernels, block_widths, strict=True):
        if groups and groups[-1][0] == kernel and groups[-1][1] == width:
            groups[-1][2] += 1
        else:
            groups.append([kernel, width, 1])

    if variant == "channel_indep_patch":
        # This encoder pins three trunk groups (one per trunk width slot). Pad
        # with zero-count groups, which reproduce an identical module tree.
        if len(groups) > 3:
            return {}
        while len(groups) < 3:
            groups.append([groups[-1][0], groups[-1][1], 0])

    inferred: dict[str, Any] = {
        "block_kernels": tuple(group[0] for group in groups),
        "blocks_per_stage": tuple(group[2] for group in groups),
    }

    first = order[0]
    if "pw_expand" in trunk[first]:
        expand = trunk[first]["pw_expand"]
        if expand[1]:
            inferred["expand_ratio"] = expand[0] // expand[1]

    project = shapes.get("pool.project.weight")
    trunk_width = block_widths[-1]
    if project is not None and trunk_width:
        heads = project[1] // trunk_width - 3
        if heads >= 1:
            inferred["pool_heads"] = heads

    if variant == "channel_indep_patch":
        proj = shapes.get("patch_embed.proj.weight")
        if proj is not None:
            inferred["patch_size"] = proj[2] // 2
        # A mix convolution exists only in signed_abs_concat mode.
        if "patch_embed.mix.weight" in shapes:
            inferred["magnitude_mode"] = "signed_abs_concat"
        embed_dim = proj[0] if proj is not None else trunk_width
        widths = (
            embed_dim,
            groups[0][1],
            groups[1][1],
            groups[2][1],
            project[0] if project is not None else trunk_width,
        )
    else:
        stem = shapes.get("input_stem.pw.weight")
        reduce_widths = [shapes.get(f"reduce.{idx}.pw.weight") for idx in range(3)]
        if stem is None or any(shape is None for shape in reduce_widths):
            return {}
        widths = (
            stem[0],
            cast(tuple[int, ...], reduce_widths[0])[0],
            cast(tuple[int, ...], reduce_widths[1])[0],
            cast(tuple[int, ...], reduce_widths[2])[0],
            project[0] if project is not None else trunk_width,
        )
        stem_dw = shapes.get("input_stem.dw.weight")
        if stem_dw is not None:
            inferred["stem_kernel"] = stem_dw[2]
        down_kernels = [shapes.get(f"reduce.{idx}.dw.weight") for idx in range(3)]
        if all(shape is not None for shape in down_kernels):
            inferred["down_kernels"] = tuple(
                cast(tuple[int, ...], shape)[2] for shape in down_kernels
            )

    inferred["widths"] = widths
    return inferred


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    *,
    checkpoint_data: dict | None = None,
    options: ScoreOptions | None = None,
    proc_cfg: ProcCfg | None = None,
    channel_names: list[str] | None = None,
    prefer_averaged: bool = False,
) -> tuple[nn.Module, dict]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        checkpoint_data: Optional preloaded checkpoint dict
        prefer_averaged: Load the averaged (EMA/SWA) weights stored under
            ``ema.module`` instead of the raw ``model`` weights, when present.
            Training evaluates and selects checkpoints on the averaged weights
            whenever averaging is enabled, so without this the model that gets
            deployed is not the one that was validated. Defaults to False to
            preserve historical behavior for existing checkpoints.
        options: Optional scoring options for artifact masking config
        proc_cfg: Optional preprocessing config to reuse for embedded preprocessor
        channel_names: Optional list of canonical channel names to pass into the model

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    if checkpoint_data is None:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:
        logger.info(f"Using preloaded checkpoint data: {checkpoint_path}")
        checkpoint = checkpoint_data

    # === CHECKPOINT STRUCTURE INSPECTION ===
    logger.info("=" * 80)
    logger.info("CHECKPOINT STRUCTURE INSPECTION")
    logger.info("=" * 80)
    logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Log important metadata
    if "epoch" in checkpoint:
        logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if "best_val_acc" in checkpoint:
        logger.info(f"  Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    if "train_config" in checkpoint:
        train_cfg = checkpoint["train_config"]
        logger.info(f"  Training task: {train_cfg.get('task', 'unknown')}")
        logger.info(
            f"  Training epochs completed: {train_cfg.get('epochs', 'unknown')}"
        )

    # Surface optional checkpoint metadata for debugging.
    class_priors = checkpoint.get("class_priors", None)
    if class_priors is not None:
        if isinstance(class_priors, list):
            class_priors = torch.tensor(class_priors, dtype=torch.float32)
        logger.info(
            f"  Class priors found in checkpoint: {class_priors.tolist() if torch.is_tensor(class_priors) else class_priors}"
        )
    else:
        logger.info("  No class priors found in checkpoint metadata")

    # Extract model architecture from checkpoint
    model_config = checkpoint.get("model_config", {})

    if not model_config:
        logger.warning(
            "Checkpoint does not contain model_config. "
            "Attempting to infer model architecture from state dict."
        )
        # Try to infer from state dict keys
        # Priority: 'model' > 'model_state_dict' > 'state_dict' > checkpoint itself
        state_dict = checkpoint.get(
            "model",
            checkpoint.get(
                "model_state_dict", checkpoint.get("state_dict", checkpoint)
            ),
        )
        # Normalize common wrapper prefixes (DDP/torch.compile) for robust loading.
        try:
            from spectra.utils.checkpoint import normalize_state_dict_keys

            state_dict = normalize_state_dict_keys(state_dict)
        except Exception:
            pass
        _raise_if_patch_checkpoint_keys(
            [k[6:] if k.startswith("model.") else k for k in state_dict.keys()]
        )
        normalized_keys = [
            k[6:] if k.startswith("model.") else k for k in state_dict.keys()
        ]
        if any(k.startswith("source_convention.") for k in normalized_keys):
            raise ValueError(
                "Source convention checkpoints require explicit model_config metadata"
            )
        if _has_learned_feature_axial_keys(normalized_keys):
            raise ValueError(
                "learned_feature_axial checkpoints require explicit model_config "
                "metadata and cannot be reconstructed from state-dictionary keys"
            )
        model_config = _infer_model_config_from_state_dict(dict(state_dict))
        logger.info(f"  Inferred model config: {model_config}")
        model_kwargs = model_config.get("model_kwargs", {})
    else:
        logger.info("  Model config found in checkpoint:")
        logger.info(f"    Model type: {model_config.get('model_type', 'unknown')}")
        logger.info(
            f"    Model kwargs keys: {list(model_config.get('model_kwargs', {}).keys())}"
        )

        # Even if model_config exists, we need to validate it matches the state_dict
        # This handles cases where domain modules exist in state_dict but model_config
        # has them set to 0 (or missing entirely)
        state_dict = checkpoint.get(
            "model",
            checkpoint.get(
                "model_state_dict", checkpoint.get("state_dict", checkpoint)
            ),
        )
        # Normalize common wrapper prefixes (DDP/torch.compile) for robust loading.
        try:
            from spectra.utils.checkpoint import normalize_state_dict_keys

            state_dict = normalize_state_dict_keys(state_dict)
        except Exception:
            pass

        model_kwargs = model_config.get("model_kwargs", {})

        # Check for engineered features and feature fusion mode
        keys_for_detection = [
            k[6:] if k.startswith("model.") else k for k in state_dict.keys()
        ]
        convention_keys = {
            key for key in keys_for_detection if key.startswith("source_convention.")
        }
        if convention_keys or model_kwargs.get("source_convention_names") is not None:
            if (
                model_kwargs.get("source_convention_names") is None
                or "source_convention_penalty" not in model_kwargs
                or convention_keys
                != {"source_convention.raw_offsets", "source_convention._extra_state"}
            ):
                raise ValueError(
                    "Source convention checkpoint requires complete learned state, "
                    "ordered source names and penalty metadata"
                )
        _raise_if_patch_checkpoint_keys(keys_for_detection)
        has_axial_v2 = _has_learned_feature_axial_v2_keys(keys_for_detection)
        has_axial = _has_learned_feature_axial_keys(keys_for_detection)
        configured_axial = model_kwargs.get("epoch_encoder_variant")
        if has_axial_v2 and configured_axial != "learned_feature_axial_v2":
            raise ValueError(
                "Checkpoint contains learned-feature axial v2 parameters but its "
                "model_config does not explicitly select learned_feature_axial_v2"
            )
        if (
            has_axial
            and not has_axial_v2
            and configured_axial != "learned_feature_axial"
        ):
            raise ValueError(
                "Checkpoint contains learned-feature axial parameters but its "
                "model_config does not explicitly select learned_feature_axial"
            )
        key_map_for_detection = {
            (k[6:] if k.startswith("model.") else k): k for k in state_dict.keys()
        }

        if model_config.get("model_type") == "TransformerContextNet":
            inferred_context = _infer_context_mode_metadata(
                state_dict,
                keys_for_detection,
                key_map_for_detection,
            )
            _reconcile_context_mode_metadata(model_kwargs, inferred_context)
            _reconcile_recurrent_refinement_metadata(
                model_kwargs,
                state_dict,
                key_map_for_detection,
            )
            logger.info("=" * 80)
            logger.info("D_MODEL RECONCILIATION")
            logger.info("=" * 80)
            checkpoint_d_model = _extract_checkpoint_d_model(state_dict)
            if checkpoint_d_model is not None:
                current_d_model = model_kwargs.get("d_model")
                if current_d_model is None:
                    model_kwargs["d_model"] = int(checkpoint_d_model)
                elif int(current_d_model) != int(checkpoint_d_model):
                    logger.warning(
                        "Checkpoint d_model mismatch: config=%r, checkpoint=%r. "
                        "Using checkpoint value.",
                        current_d_model,
                        checkpoint_d_model,
                    )
                    model_kwargs["d_model"] = int(checkpoint_d_model)

            model_config["model_kwargs"] = model_kwargs

        # Reconcile classifier / feature architecture from checkpoint state dict.
        if model_config.get("model_type") == "TransformerContextNet":
            # PerPositionSleepHead reconciliation: extract_transformer_config does
            # not record use_per_position_head, so a metadata checkpoint trained
            # with it still lacks the flag -> default head is built and the trained
            # classifier.mlp/norm weights are dropped -> single-stage collapse.
            head_family = _detect_classifier_head_family(keys_for_detection)
            has_per_position_head = head_family == "per_position"
            if has_per_position_head and not model_kwargs.get(
                "use_per_position_head", False
            ):
                logger.info(
                    "Detected PerPositionSleepHead in checkpoint state_dict; "
                    "enabling use_per_position_head=True"
                )
                model_kwargs["use_per_position_head"] = True
                model_kwargs["head_use_local_mix"] = any(
                    "classifier.local." in k for k in keys_for_detection
                )
                model_kwargs["head_norm"] = (
                    "layernorm"
                    if any(
                        k.endswith("classifier.norm.bias") for k in keys_for_detection
                    )
                    else "rmsnorm"
                )
                model_config["model_kwargs"] = model_kwargs

            # Classifier-head reconciliation. Checkpoints written before
            # classifier_head existed carry no such key, so a state dict with the
            # linear head's signature must override the absent/stale metadata or
            # the default residual head is built and every trained classifier
            # weight is dropped or shape-mismatched. State dict wins in both
            # directions, matching the d_model reconciliation above.
            configured_head = model_kwargs.get("classifier_head", "residual_mlp")
            if head_family == "linear" and configured_head != "linear":
                logger.info(
                    "Detected LinearSleepHead in checkpoint state_dict; setting "
                    "classifier_head='linear' (config said %r)",
                    configured_head,
                )
                model_kwargs["classifier_head"] = "linear"
                # A stale config carrying both would trip the constructor's
                # mutual-exclusion check from deep inside inference.
                model_kwargs["use_per_position_head"] = False
                model_config["model_kwargs"] = model_kwargs
            elif head_family == "residual_mlp" and configured_head == "linear":
                logger.warning(
                    "Checkpoint classifier_head mismatch: config='linear', "
                    "state_dict=residual MLP. Using the state-dict architecture."
                )
                model_kwargs["classifier_head"] = "residual_mlp"
                model_config["model_kwargs"] = model_kwargs

        has_cnn_weight_in_state = any(
            "epoch_encoder.cnn_weight" in k for k in keys_for_detection
        )
        has_eng_weight_in_state = any(
            "epoch_encoder.eng_weight" in k for k in keys_for_detection
        )
        has_gate_in_state = any("epoch_encoder.gate" in k for k in keys_for_detection)
        has_feature_extractor_in_state = any(
            "feature_extractor." in k for k in keys_for_detection
        )
        has_feat_projection_in_state = any(
            "feat_projection." in k for k in keys_for_detection
        )
        has_sequential_fusion_in_state = any(
            "sequential_fusion." in k for k in keys_for_detection
        )

        has_fusion_in_state = (
            has_cnn_weight_in_state or has_eng_weight_in_state or has_gate_in_state
        )
        has_sequential_params_in_state = (
            has_feature_extractor_in_state
            or has_feat_projection_in_state
            or has_sequential_fusion_in_state
        )
        has_engineered_in_state = has_fusion_in_state or has_sequential_params_in_state

        logger.info("=" * 80)
        logger.info("ENGINEERED FEATURES DETECTION")
        logger.info("=" * 80)
        logger.info("State dict checks:")
        logger.info(f"  has_cnn_weight_in_state: {has_cnn_weight_in_state}")
        logger.info(f"  has_eng_weight_in_state: {has_eng_weight_in_state}")
        logger.info(f"  has_gate_in_state: {has_gate_in_state}")
        logger.info(f"  has_fusion_in_state: {has_fusion_in_state}")
        logger.info(
            f"  has_feature_extractor_in_state: {has_feature_extractor_in_state}"
        )
        logger.info(f"  has_feat_projection_in_state: {has_feat_projection_in_state}")
        logger.info(
            f"  has_sequential_fusion_in_state: {has_sequential_fusion_in_state}"
        )
        logger.info(
            f"  has_sequential_params_in_state: {has_sequential_params_in_state}"
        )

        has_engineered_in_config = model_kwargs.get("use_feature_extraction", False)
        logger.info("Config check:")
        logger.info(f"  has_engineered_in_config: {has_engineered_in_config}")

        if has_engineered_in_state and not has_engineered_in_config:
            logger.warning(
                "Checkpoint state_dict contains feature fusion parameters but "
                "model_config does not specify use_feature_extraction=True. "
                "Enabling feature extraction and inferring fusion mode from state_dict..."
            )
            model_kwargs["use_feature_extraction"] = True

            # Detect fusion mode
            if has_sequential_params_in_state:
                model_kwargs["fusion_mode"] = "sequential"
                logger.info(
                    "Detected sequential fusion mode from feature_extractor parameters"
                )
            elif has_cnn_weight_in_state:
                model_kwargs["fusion_mode"] = "learned_weight"
                logger.info(
                    "Detected learned_weight fusion mode from cnn_weight parameter"
                )
            elif has_gate_in_state:
                model_kwargs["fusion_mode"] = "gated"
                logger.info("Detected gated fusion mode from gate parameter")
            else:
                model_kwargs["fusion_mode"] = "concat"
                logger.info("Using concat fusion mode (default)")

            model_config["model_kwargs"] = model_kwargs
        elif has_engineered_in_config:
            # Validate fusion mode matches state dict
            config_fusion_mode = model_kwargs.get("fusion_mode", "concat")
            detected_fusion_mode = None

            if has_sequential_params_in_state:
                detected_fusion_mode = "sequential"
            elif has_cnn_weight_in_state:
                detected_fusion_mode = "learned_weight"
            elif has_gate_in_state:
                detected_fusion_mode = "gated"
            else:
                detected_fusion_mode = "concat"

            if detected_fusion_mode != config_fusion_mode:
                logger.warning(
                    f"Fusion mode mismatch: config says '{config_fusion_mode}' but "
                    f"state_dict indicates '{detected_fusion_mode}'. Using state_dict value."
                )
                model_kwargs["fusion_mode"] = detected_fusion_mode
                model_config["model_kwargs"] = model_kwargs

        # Check for simple_feature_projection (Linear vs BottleneckProjection)
        # BottleneckProjection has keys like: feat_projection.bottleneck.0.weight
        # Simple Linear projection has keys like: feat_projection.weight
        has_bottleneck_projection_in_state = any(
            "feat_projection.bottleneck" in k for k in keys_for_detection
        )
        has_simple_projection_in_state = any(
            k == "feat_projection.weight" or k.endswith(".feat_projection.weight")
            for k in keys_for_detection
        )
        has_simple_projection_in_config = model_kwargs.get(
            "simple_feature_projection", False
        )

        logger.info("=" * 80)
        logger.info("FEATURE PROJECTION TYPE DETECTION")
        logger.info("=" * 80)
        logger.info("State dict checks:")
        logger.info(
            "  has_bottleneck_projection_in_state: %s",
            has_bottleneck_projection_in_state,
        )
        logger.info(
            "  has_simple_projection_in_state: %s", has_simple_projection_in_state
        )
        logger.info("Config check:")
        logger.info(
            "  simple_feature_projection in config: %s", has_simple_projection_in_config
        )

        if has_bottleneck_projection_in_state and has_simple_projection_in_config:
            logger.warning(
                "Checkpoint state_dict contains BottleneckProjection parameters but "
                "model_config specifies simple_feature_projection=True. "
                "Disabling simple_feature_projection to use BottleneckProjection..."
            )
            model_kwargs["simple_feature_projection"] = False
            model_config["model_kwargs"] = model_kwargs
        elif has_simple_projection_in_state and not has_simple_projection_in_config:
            logger.warning(
                "Checkpoint state_dict contains simple Linear projection but "
                "model_config specifies simple_feature_projection=False. "
                "Enabling simple_feature_projection to match checkpoint..."
            )
            model_kwargs["simple_feature_projection"] = True
            model_config["model_kwargs"] = model_kwargs
        elif has_bottleneck_projection_in_state:
            logger.info(
                "✓ BottleneckProjection detected in checkpoint (simple_feature_projection=False)"
            )
            # Ensure config matches
            if model_kwargs.get("simple_feature_projection", False):
                model_kwargs["simple_feature_projection"] = False
                model_config["model_kwargs"] = model_kwargs
        elif has_simple_projection_in_state:
            logger.info(
                "✓ Simple Linear projection detected in checkpoint (simple_feature_projection=True)"
            )
            # Ensure config matches
            if not model_kwargs.get("simple_feature_projection", False):
                model_kwargs["simple_feature_projection"] = True
                model_config["model_kwargs"] = model_kwargs
        else:
            logger.info(
                "✓ No feat_projection detected (engineered features may not be used)"
            )

        # Check for CNN feature guidance (FiLM-style injection into epoch encoder).
        # When enabled, epoch encoders include feature_inject_stage*_* linear layers.
        has_feature_guidance_in_state = any(
            "feature_inject_stage" in k for k in keys_for_detection
        )
        has_feature_guidance_in_config = model_kwargs.get("cnn_feature_guidance", False)
        logger.info("=" * 80)
        logger.info("CNN FEATURE GUIDANCE DETECTION")
        logger.info("=" * 80)
        logger.info("State dict checks:")
        logger.info(
            "  has_feature_guidance_in_state: %s", has_feature_guidance_in_state
        )
        logger.info("Config check:")
        logger.info(
            "  has_feature_guidance_in_config: %s", has_feature_guidance_in_config
        )

        if has_feature_guidance_in_state and not has_feature_guidance_in_config:
            logger.warning(
                "Checkpoint state_dict contains feature guidance parameters but "
                "model_config does not specify cnn_feature_guidance=True. "
                "Enabling cnn_feature_guidance to match checkpoint..."
            )
            model_kwargs["cnn_feature_guidance"] = True
            model_config["model_kwargs"] = model_kwargs
        elif not has_feature_guidance_in_state and has_feature_guidance_in_config:
            logger.warning(
                "Model config specifies cnn_feature_guidance=True but "
                "checkpoint state_dict does not contain feature guidance parameters. "
                "Disabling cnn_feature_guidance to match checkpoint..."
            )
            model_kwargs["cnn_feature_guidance"] = False
            model_config["model_kwargs"] = model_kwargs

        # Check for N1 attention
        # N1 attention adds n1_feature_extractor and n1_attention modules
        has_n1_feature_extractor = any(
            "n1_feature_extractor" in k for k in state_dict.keys()
        )
        has_n1_attention = any("n1_attention" in k for k in state_dict.keys())
        has_n1_in_state = has_n1_feature_extractor or has_n1_attention

        logger.info("=" * 80)
        logger.info("N1 ATTENTION DETECTION")
        logger.info("=" * 80)
        logger.info("State dict checks:")
        logger.info(f"  has_n1_feature_extractor: {has_n1_feature_extractor}")
        logger.info(f"  has_n1_attention: {has_n1_attention}")
        logger.info(f"  has_n1_in_state: {has_n1_in_state}")

        has_n1_in_config = model_kwargs.get("use_n1_attention", False)
        logger.info("Config check:")
        logger.info(f"  has_n1_in_config: {has_n1_in_config}")

        if has_n1_in_state and not has_n1_in_config:
            logger.warning(
                "Checkpoint state_dict contains N1 attention parameters but "
                "model_config does not specify use_n1_attention=True. "
                "Enabling N1 attention to match checkpoint..."
            )
            model_kwargs["use_n1_attention"] = True
            model_config["model_kwargs"] = model_kwargs
        elif not has_n1_in_state and has_n1_in_config:
            logger.warning(
                "Model config specifies use_n1_attention=True but "
                "checkpoint state_dict does not contain N1 attention parameters. "
                "Disabling N1 attention to match checkpoint..."
            )
            model_kwargs["use_n1_attention"] = False
            model_config["model_kwargs"] = model_kwargs
        elif has_n1_in_state and has_n1_in_config:
            logger.info("✓ N1 attention enabled in both config and checkpoint")
        else:
            logger.info("✓ N1 attention not used (config and checkpoint agree)")

        # Check for learnable temperature
        # Learnable temperature adds log_temperature parameter
        has_log_temperature_in_state = any(
            "log_temperature" in k for k in state_dict.keys()
        )
        has_learnable_temp_in_config = model_kwargs.get("learnable_temperature", False)

        logger.info("=" * 80)
        logger.info("LEARNABLE TEMPERATURE DETECTION")
        logger.info("=" * 80)
        logger.info("State dict checks:")
        logger.info(f"  has_log_temperature_in_state: {has_log_temperature_in_state}")
        logger.info("Config check:")
        logger.info(f"  has_learnable_temp_in_config: {has_learnable_temp_in_config}")

        if has_log_temperature_in_state and not has_learnable_temp_in_config:
            logger.warning(
                "Checkpoint state_dict contains log_temperature parameters but "
                "model_config does not specify learnable_temperature=True. "
                "Enabling learnable temperature to match checkpoint..."
            )
            model_kwargs["learnable_temperature"] = True
            model_config["model_kwargs"] = model_kwargs
        elif not has_log_temperature_in_state and has_learnable_temp_in_config:
            logger.warning(
                "Model config specifies learnable_temperature=True but "
                "checkpoint state_dict does not contain log_temperature parameters. "
                "Disabling learnable temperature to match checkpoint..."
            )
            model_kwargs["learnable_temperature"] = False
            model_config["model_kwargs"] = model_kwargs
        elif has_log_temperature_in_state and has_learnable_temp_in_config:
            logger.info("✓ Learnable temperature enabled in both config and checkpoint")
        else:
            logger.info(
                "✓ Learnable temperature not used (config and checkpoint agree)"
            )

        keys_for_detection = [
            k[6:] if k.startswith("model.") else k for k in state_dict.keys()
        ]
        key_map_for_detection = {
            (k[6:] if k.startswith("model.") else k): k for k in state_dict.keys()
        }

        has_confidence_head_in_state = any(
            "confidence_head." in k for k in state_dict.keys()
        )
        has_confidence_head_in_config = model_kwargs.get("use_confidence_head", False)

        logger.info("=" * 80)
        logger.info("CONFIDENCE HEAD DETECTION")
        logger.info("=" * 80)
        logger.info("State dict checks:")
        logger.info("  has_confidence_head_in_state: %s", has_confidence_head_in_state)
        logger.info("Config check:")
        logger.info(
            "  has_confidence_head_in_config: %s", has_confidence_head_in_config
        )

        if has_confidence_head_in_state and not has_confidence_head_in_config:
            logger.warning(
                "Checkpoint state_dict contains confidence_head parameters but "
                "model_config does not specify use_confidence_head=True. "
                "Enabling confidence head to match checkpoint..."
            )
            model_kwargs["use_confidence_head"] = True
            hidden_key = next(
                (
                    k
                    for k in keys_for_detection
                    if k.endswith("confidence_head.net.1.weight")
                ),
                None,
            )
            if hidden_key is not None:
                model_kwargs["confidence_head_hidden_dim"] = int(
                    state_dict[key_map_for_detection[hidden_key]].shape[0]
                )
            model_config["model_kwargs"] = model_kwargs
        elif not has_confidence_head_in_state and has_confidence_head_in_config:
            logger.warning(
                "Model config specifies use_confidence_head=True but checkpoint "
                "state_dict does not contain confidence_head parameters. "
                "Disabling confidence head to match checkpoint..."
            )
            model_kwargs["use_confidence_head"] = False
            model_config["model_kwargs"] = model_kwargs
        elif has_confidence_head_in_state and has_confidence_head_in_config:
            logger.info("✓ Confidence head enabled in both config and checkpoint")
        else:
            logger.info("✓ Confidence head not used (config and checkpoint agree)")

    # =========================================================================
    # EPOCH ENCODER VARIANT DETECTION
    # =========================================================================
    # Detect epoch_encoder_variant from state_dict structure and kernel sizes.
    # The legacy 'resepochcnn' (ResEpochCNN, single stem, small kernels) and
    # 'improved' (ImprovedResEpochCNN, MultiScaleStem fast/medium/slow branches)
    # encoders have been removed; checkpoints matching them now raise a clear
    # error instead of being reconstructed.
    logger.info("=" * 80)
    logger.info("EPOCH ENCODER VARIANT DETECTION")
    logger.info("=" * 80)

    variant_keys = [
        k[6:] if isinstance(k, str) and k.startswith("model.") else k
        for k in state_dict.keys()
        if isinstance(k, str)
    ]

    # Check for ImprovedResEpochCNN markers (MultiScaleStem branches)
    has_multiscale_stem = any("epoch_encoder.stem.fast" in k for k in variant_keys)
    has_stem_medium = any("epoch_encoder.stem.medium" in k for k in variant_keys)
    has_stem_slow = any("epoch_encoder.stem.slow" in k for k in variant_keys)

    # Check for DilatedAsymmetricEpochCNN markers (MultiDilatedBlock with branches and res_scale)
    has_multidilated_block = any(
        (
            "epoch_encoder.stage1.0.branches." in k
            or "epoch_encoder.stage2.0.branches." in k
            or "epoch_encoder.stage1.0.block.branches." in k
            or "epoch_encoder.stage2.0.block.branches." in k
        )
        for k in variant_keys
    )
    has_learnable_res_scale = any(
        "epoch_encoder.stage1.0.res_scale" in k
        or "epoch_encoder.stage2.0.res_scale" in k
        or "epoch_encoder.stage1.0.block.res_scale" in k
        or "epoch_encoder.stage2.0.block.res_scale" in k
        for k in variant_keys
    )

    # Check for FlexibleAsymmetricEpochCNN markers (FlexibleModalityAwareStem with FlexiblePhysiologicalStem)
    # FlexiblePhysiologicalStem has dilated_branches and scale_attention instead of fast/spindle/kcomplex/slow
    has_flexible_stem = any(
        "epoch_encoder.stem.eeg_branch.dilated_branches" in k
        or "epoch_encoder.stem.eeg_branch.scale_attention" in k
        for k in variant_keys
    )

    has_sleep_staging_hybrid = has_flexible_stem and any(
        "epoch_encoder.stage3.0.bn1.weight" in k
        or "epoch_encoder.stage4.0.bn1.weight" in k
        for k in variant_keys
    )

    # DeepAsymmetricEpochCNN shares the flexible stem + MultiDilatedBlock stages of
    # flexible_asymmetric but stacks a configurable number of blocks per stage
    # (flexible is always exactly 2). Recover the block count per stage from keys of
    # the form "epoch_encoder.stage{1..4}.{idx}." so we can (a) tell the two variants
    # apart and (b) reconstruct the correct depth for checkpoints whose model_kwargs
    # lack deep_blocks_per_stage.
    stage_block_counts: dict[int, int] = {}
    for k in variant_keys:
        parts = k.split(".")
        if (
            len(parts) >= 3
            and parts[0] == "epoch_encoder"
            and parts[1] in ("stage1", "stage2", "stage3", "stage4")
            and parts[2].isdigit()
        ):
            stage_num = int(parts[1][-1])
            block_idx = int(parts[2])
            stage_block_counts[stage_num] = max(
                stage_block_counts.get(stage_num, 0), block_idx + 1
            )
    # A deep model built with (2,2,2,2) is byte-identical to flexible and correctly
    # falls through to the flexible branch; only >2 blocks in some stage mark it deep.
    has_deep_blocks = (
        has_flexible_stem
        and has_multidilated_block
        and max(stage_block_counts.values(), default=0) > 2
    )

    # PreactAsymmetricEpochCNN also uses the flexible stem + parallel dilated
    # branches, so it collides with flexible_asymmetric/deep_asymmetric on the
    # markers above. Its full-preactivation blocks carry two submodules the other
    # variants never emit: a per-block ``pre_norm`` BatchNorm and a ``layer_scale``
    # parameter. Detect on those so Preact is never misread as deep/flexible.
    has_preact = any(
        (".pre_norm." in k and k.startswith("epoch_encoder.stage"))
        or (k.startswith("epoch_encoder.stage") and k.endswith(".layer_scale"))
        for k in variant_keys
    )

    # ModernTCN family (modern_sleep_tcn / channel_indep_patch). Global Response
    # Normalization exists nowhere else in the repository, so a ``.grn.gamma``
    # key is an unambiguous family marker. Neither encoder registers a submodule
    # under a name starting with "stage", so none of the markers above can fire
    # for them -- but detect positively anyway, because a checkpoint saved
    # without model_config would otherwise fall through to the config default.
    has_grn_trunk = any(k.endswith(".grn.gamma") for k in variant_keys)
    # The shared per-channel patch embed is what separates the two: architecture
    # A mixes channels in its stem and has ``input_stem`` instead.
    has_channel_indep_embed = any(
        "epoch_encoder.patch_embed.proj." in k for k in variant_keys
    )
    has_modern_tcn_stem = any("epoch_encoder.input_stem.dw." in k for k in variant_keys)
    # multirate_asymmetric registers its stem as ``multirate_stem``; the name
    # exists nowhere else in the repository.
    has_multirate_stem = any("epoch_encoder.multirate_stem." in k for k in variant_keys)

    logger.info(f"  has_multiscale_stem (stem.fast): {has_multiscale_stem}")
    logger.info(f"  has_grn_trunk (ModernTCN family): {has_grn_trunk}")
    logger.info(f"  has_multirate_stem (multirate_asymmetric): {has_multirate_stem}")
    logger.info(f"  has_channel_indep_embed: {has_channel_indep_embed}")
    logger.info(f"  has_stem_medium: {has_stem_medium}")
    logger.info(f"  has_stem_slow: {has_stem_slow}")
    logger.info(f"  has_multidilated_block: {has_multidilated_block}")
    logger.info(f"  has_learnable_res_scale: {has_learnable_res_scale}")
    logger.info(f"  has_flexible_stem: {has_flexible_stem}")
    logger.info(
        "  has_sleep_staging_hybrid (flexible stem + explicit residual late stages): %s",
        has_sleep_staging_hybrid,
    )
    logger.info(
        "  has_deep_blocks (flexible stem + >2 blocks/stage): %s (stage counts: %s)",
        has_deep_blocks,
        {s: stage_block_counts[s] for s in sorted(stage_block_counts)},
    )
    logger.info("  has_preact (per-block pre_norm / layer_scale): %s", has_preact)

    # Check kernel sizes to confirm architecture
    stage2_kernel_size = None
    for key, value in state_dict.items():
        clean_key = key[6:] if key.startswith("model.") else key
        if (
            "epoch_encoder.stage2.0.conv1.weight" in clean_key
            or "epoch_encoder.stage2.0.block.conv1.weight" in clean_key
        ) and hasattr(value, "shape"):
            stage2_kernel_size = value.shape[-1]  # kernel size is last dimension
            break

    logger.info(f"  stage2 conv1 kernel size: {stage2_kernel_size}")

    config_epoch_encoder_variant = model_kwargs.get("epoch_encoder_variant")
    if config_epoch_encoder_variant in {"resepochcnn", "improved"}:
        raise ValueError(
            f"This checkpoint declares the removed '{config_epoch_encoder_variant}' "
            "epoch encoder (ResEpochCNN/ImprovedResEpochCNN), which is no longer "
            "available. Re-train with a supported epoch_encoder_variant "
            "(e.g. flexible_asymmetric)."
        )
    detected_variant = None

    # Check distinctive Conformer and ModernTCN markers before legacy CNNs.
    if any(
        "epoch_encoder.blocks.0.attention.in_proj_weight"
        in key.replace("epoch_encoder.cnn_extractor.", "epoch_encoder.")
        for key in variant_keys
    ):
        detected_variant = "conformer"
        if "conformer_encoder_kwargs" not in model_kwargs:
            raise ValueError(
                "Conformer inference requires model_config.model_kwargs.conformer_encoder_kwargs; attention heads cannot be inferred from weights"
            )
        required = {
            "stem_type",
            "d_model",
            "num_blocks",
            "heads",
            "ffn_expansion",
            "conv_kernel_size",
            "pooling_mode",
        }
        if model_kwargs["conformer_encoder_kwargs"].get("stem_type") == "multirate":
            required.update({"eeg_indices", "eog_indices", "emg_indices"})
        missing_config = required.difference(model_kwargs["conformer_encoder_kwargs"])
        if missing_config:
            raise ValueError(
                f"Conformer inference configuration is missing {sorted(missing_config)}"
            )
        logger.info("  Detected conformer (Conformer attention marker)")
    elif has_multirate_stem:
        detected_variant = "multirate_asymmetric"
        logger.info("  Detected multirate_asymmetric (multirate_stem marker)")
    elif has_grn_trunk and has_channel_indep_embed:
        detected_variant = "channel_indep_patch"
        logger.info(
            "  Detected channel_indep_patch (GRN trunk + shared per-channel embed)"
        )
    elif has_grn_trunk and has_modern_tcn_stem:
        detected_variant = "modern_sleep_tcn"
        logger.info("  Detected modern_sleep_tcn (GRN trunk + depthwise input stem)")
    elif has_grn_trunk:
        # A GRN trunk with neither stem marker: trust the config if it names a
        # family member rather than guessing between them.
        if config_epoch_encoder_variant in ("modern_sleep_tcn", "channel_indep_patch"):
            detected_variant = config_epoch_encoder_variant
            logger.info(
                "  GRN trunk present; keeping variant from config: %s",
                config_epoch_encoder_variant,
            )
        else:
            logger.warning(
                "  Checkpoint has a GRN trunk (ModernTCN family) but no recognizable "
                "stem marker and config says %r. Leaving the config value alone.",
                config_epoch_encoder_variant,
            )
    elif (
        config_epoch_encoder_variant == "sleep_staging_hybrid"
        and has_sleep_staging_hybrid
    ):
        logger.info(
            "  Keeping sleep-staging hybrid variant from config: %s",
            config_epoch_encoder_variant,
        )
        detected_variant = config_epoch_encoder_variant
    elif has_sleep_staging_hybrid:
        detected_variant = "sleep_staging_hybrid"
        logger.info(
            "  Detected SleepStagingHybridEpochCNN from flexible stem + late residual stage markers"
        )
    elif config_epoch_encoder_variant == "preact_asymmetric" and has_preact:
        # PreactAsymmetricEpochCNN shares flexible_asymmetric's stem + parallel
        # dilated branches, so trust an explicit preact_asymmetric config. The
        # per-block pre_norm/layer_scale markers confirm the state_dict matches.
        detected_variant = "preact_asymmetric"
        logger.info(
            "  Keeping preact_asymmetric variant from config "
            "(flexible stem + full-preactivation blocks)"
        )
    elif has_preact:
        # Full-preactivation residual trunk: flexible stem + per-block
        # pre_norm/layer_scale. Must check BEFORE deep/flexible because those
        # markers (flexible stem + >2 blocks/stage) also fire for Preact.
        detected_variant = "preact_asymmetric"
        logger.info(
            "  Detected PreactAsymmetricEpochCNN from per-block pre_norm/layer_scale markers"
        )
    elif (
        config_epoch_encoder_variant == "deep_asymmetric"
        and has_flexible_stem
        and has_multidilated_block
    ):
        # DeepAsymmetricEpochCNN shares flexible_asymmetric's stem + MultiDilatedBlock
        # markers, so trust an explicit deep_asymmetric config and never downgrade it to
        # flexible. This also covers a deep model built with (2,2,2,2) blocks, where
        # has_deep_blocks is False but the config is still authoritative.
        detected_variant = "deep_asymmetric"
        logger.info(
            "  Keeping deep_asymmetric variant from config "
            "(flexible stem + MultiDilatedBlock trunk)"
        )
    elif has_deep_blocks:
        # DeepAsymmetricEpochCNN: flexible stem + MultiDilatedBlock stages with a
        # configurable (>2) block count per stage. Must check BEFORE flexible_asymmetric.
        detected_variant = "deep_asymmetric"
        logger.info(
            "  Detected DeepAsymmetricEpochCNN from flexible stem + >2 blocks/stage"
        )
    elif has_flexible_stem and has_multidilated_block:
        # FlexibleAsymmetricEpochCNN uses FlexibleModalityAwareStem (with dilated_branches/scale_attention)
        # and MultiDilatedBlock stages - must check BEFORE dilated_asymmetric
        detected_variant = "flexible_asymmetric"
        logger.info(
            "  Detected FlexibleAsymmetricEpochCNN from FlexibleModalityAwareStem structure"
        )
    elif has_multidilated_block:
        # DilatedAsymmetricEpochCNN uses MultiDilatedBlock with learnable res_scale
        # Uses ModalityAwareStem with fast/spindle/kcomplex/slow branches (not dilated_branches)
        detected_variant = "dilated_asymmetric"
        logger.info(
            "  Detected DilatedAsymmetricEpochCNN from MultiDilatedBlock/res_scale structure"
        )
    elif has_multiscale_stem and has_stem_medium and has_stem_slow:
        # MultiScaleStem (fast/medium/slow) marks the removed ImprovedResEpochCNN.
        raise ValueError(
            "This checkpoint uses the removed 'improved' (ImprovedResEpochCNN) epoch "
            "encoder, which is no longer available. Re-train with a supported "
            "epoch_encoder_variant (e.g. flexible_asymmetric)."
        )
    elif stage2_kernel_size is not None:
        # A plain residual stem with small kernels marks the removed encoders:
        # kernel >= 9 was ImprovedResEpochCNN, otherwise ResEpochCNN.
        removed = "improved" if stage2_kernel_size >= 9 else "resepochcnn"
        raise ValueError(
            f"This checkpoint uses the removed '{removed}' epoch encoder "
            "(ResEpochCNN/ImprovedResEpochCNN), which is no longer available. "
            "Re-train with a supported epoch_encoder_variant (e.g. flexible_asymmetric)."
        )

    if config_epoch_encoder_variant in {
        "learned_feature_axial",
        "learned_feature_axial_v2",
    }:
        # This architecture is metadata-only by design. Never reinterpret it
        # from state-dictionary key heuristics.
        logger.info(
            f"  Using explicit {config_epoch_encoder_variant} checkpoint metadata; "
            "state-key architecture inference is disabled"
        )
    elif detected_variant is not None:
        if config_epoch_encoder_variant is None:
            logger.info(
                f"  epoch_encoder_variant not in config, using detected: {detected_variant}"
            )
            model_kwargs["epoch_encoder_variant"] = detected_variant
            model_config["model_kwargs"] = model_kwargs
        elif config_epoch_encoder_variant != detected_variant:
            logger.warning(
                f"  Config says epoch_encoder_variant='{config_epoch_encoder_variant}' but "
                f"state_dict indicates '{detected_variant}'. Using state_dict value."
            )
            model_kwargs["epoch_encoder_variant"] = detected_variant
            model_config["model_kwargs"] = model_kwargs
        else:
            logger.info(f"  ✓ epoch_encoder_variant matches: {detected_variant}")
    else:
        logger.info("  Could not detect epoch_encoder_variant, using config default")

    # deep_asymmetric: blocks_per_stage changes the state_dict, so the constructor
    # must be given the matching depth. Prefer the value persisted in model_kwargs
    # (authoritative - saved from the trained model); fall back to the depth inferred
    # from the state dict for older/partial checkpoints. drop_path is inert at eval,
    # so a default keeps the constructor call clean when it is absent.
    if model_kwargs.get("epoch_encoder_variant") == "deep_asymmetric":
        if "deep_blocks_per_stage" not in model_kwargs and stage_block_counts:
            inferred_blocks = tuple(stage_block_counts.get(s, 2) for s in (1, 2, 3, 4))
            model_kwargs["deep_blocks_per_stage"] = inferred_blocks
            logger.info(
                "  deep_blocks_per_stage missing from config; inferred %s from state_dict",
                inferred_blocks,
            )
        model_kwargs.setdefault("deep_drop_path_rate", 0.1)
        model_config["model_kwargs"] = model_kwargs

    # preact_asymmetric: like deep_asymmetric, blocks_per_stage changes the
    # state_dict, so the constructor must receive the matching depth. Prefer the
    # value persisted in model_kwargs; otherwise infer it from the state dict.
    if model_kwargs.get("epoch_encoder_variant") == "preact_asymmetric":
        if "preact_blocks_per_stage" not in model_kwargs and stage_block_counts:
            inferred_blocks = tuple(stage_block_counts.get(s, 2) for s in (1, 2, 3, 4))
            model_kwargs["preact_blocks_per_stage"] = inferred_blocks
            logger.info(
                "  preact_blocks_per_stage missing from config; inferred %s from state_dict",
                inferred_blocks,
            )
        model_kwargs.setdefault("preact_drop_path_rate", 0.1)
        model_config["model_kwargs"] = model_kwargs

    # ModernTCN family: recover the shape-bearing structure from the weights when
    # model_config is absent. Everything inferred here is unambiguous from a
    # tensor shape. Behaviour-only options (linear_stem, and magnitude_mode
    # within the abs/none pair) leave no trace in the weights and fall back to
    # the encoder defaults -- which is exactly why extract_transformer_config
    # persists the whole get_config() dict for these variants.
    _modern_tcn_variant = model_kwargs.get("epoch_encoder_variant")
    _passthrough_keys = {
        "modern_sleep_tcn": "modern_tcn_encoder_kwargs",
        "channel_indep_patch": "channel_indep_encoder_kwargs",
        "multirate_asymmetric": "multirate_asymmetric_encoder_kwargs",
    }
    if _modern_tcn_variant in _passthrough_keys:
        passthrough_key = _passthrough_keys[_modern_tcn_variant]
        if passthrough_key not in model_kwargs:
            if _modern_tcn_variant == "multirate_asymmetric":
                inferred = _infer_multirate_structure(state_dict)
            else:
                inferred = _infer_modern_tcn_structure(state_dict, _modern_tcn_variant)
            if inferred:
                widths = inferred.pop("widths", None)
                if widths is not None and "cnn_widths" not in model_kwargs:
                    model_kwargs["cnn_widths"] = widths
                if inferred:
                    model_kwargs[passthrough_key] = inferred
                logger.info(
                    "  %s missing from config; inferred %s (widths=%s) from state_dict",
                    passthrough_key,
                    inferred,
                    widths,
                )
            model_config["model_kwargs"] = model_kwargs

    # =========================================================================
    # POOLING MODE DETECTION
    # =========================================================================
    # When pooling_mode="both", output dim = 256 + 128 = 384
    # When pooling_mode="attention" or "statistics", output dim = 256
    # We can detect this from:
    #   - proj.weight shape (input dim should match CNN output)
    #   - sequential_fusion dimensions
    #   - feat_projection dimensions
    if model_kwargs.get("epoch_encoder_variant") == "dilated_asymmetric":
        logger.info("=" * 80)
        logger.info("POOLING MODE DETECTION")
        logger.info("=" * 80)

        # Check proj.weight input dimension (should match CNN output)
        proj_weight_shape = None
        for key, value in state_dict.items():
            clean_key = key[6:] if key.startswith("model.") else key
            if clean_key.endswith("proj.weight") and "feat_projection" not in clean_key:
                if hasattr(value, "shape") and len(value.shape) == 2:
                    proj_weight_shape = value.shape
                    break

        # Check feat_projection output dimension (should match d_model for sequential fusion)
        feat_proj_shape = None
        for key, value in state_dict.items():
            clean_key = key[6:] if key.startswith("model.") else key
            if "feat_projection.bottleneck.4.weight" in clean_key and hasattr(
                value, "shape"
            ):
                feat_proj_shape = value.shape
                break

        logger.info(f"  proj.weight shape: {proj_weight_shape}")
        logger.info(f"  feat_projection.bottleneck.4.weight shape: {feat_proj_shape}")

        config_pooling_mode = model_kwargs.get("pooling_mode")
        detected_pooling_mode = None

        # Pooling-mode-dependent CNN output dim:
        #   pooling_mode="both" -> out_dim = 256 + 128 = 384
        #   pooling_mode="attention" or "statistics" -> out_dim = 256
        # The proj.weight input dim or feat_projection output should indicate this
        if feat_proj_shape is not None:
            # feat_projection.bottleneck.4.weight is [d_model, hidden_dim]
            # For sequential fusion, this projects to d_model
            d_model = model_kwargs.get("d_model", 384)
            if feat_proj_shape[0] == d_model:
                # The hidden_dim in bottleneck is 2x input (768 for 384 output)
                # This suggests pooling_mode="both" which has larger CNN output
                detected_pooling_mode = "both"
                logger.info(
                    "  Detected pooling_mode='both' from feat_projection dimensions"
                )

        if proj_weight_shape is not None and detected_pooling_mode is None:
            # proj.weight is [d_model, cnn_output_dim]
            cnn_output_dim = proj_weight_shape[1]
            if cnn_output_dim == 384:
                detected_pooling_mode = "both"
            elif cnn_output_dim == 256:
                detected_pooling_mode = (
                    "attention"  # or 'statistics', can't distinguish
                )
            logger.info(
                f"  Detected pooling_mode from proj.weight input dim {cnn_output_dim}"
            )

        if detected_pooling_mode is not None:
            if config_pooling_mode is None:
                logger.info(
                    f"  pooling_mode not in config, using detected: {detected_pooling_mode}"
                )
                model_kwargs["pooling_mode"] = detected_pooling_mode
                model_config["model_kwargs"] = model_kwargs
            elif config_pooling_mode != detected_pooling_mode:
                logger.warning(
                    f"  Config says pooling_mode='{config_pooling_mode}' but "
                    f"state_dict indicates '{detected_pooling_mode}'. Using state_dict value."
                )
                model_kwargs["pooling_mode"] = detected_pooling_mode
                model_config["model_kwargs"] = model_kwargs
            else:
                logger.info(f"  ✓ pooling_mode matches: {detected_pooling_mode}")
        else:
            logger.info("  Could not detect pooling_mode, using config default")
    elif (
        model_kwargs.get("epoch_encoder_variant") == "flexible_asymmetric"
        or has_flexible_stem
    ):
        logger.info("=" * 80)
        logger.info("POOLING MODE DETECTION (FLEXIBLE ASYMMETRIC)")
        logger.info("=" * 80)

        config_pooling_mode = model_kwargs.get("pooling_mode")

        has_pool_head_scorers = any(
            "epoch_encoder.pool.head_scorers." in k or "cnn.pool.head_scorers." in k
            for k in variant_keys
        )
        has_pool_latent_queries = any(
            "epoch_encoder.pool.latent_queries" in k or "cnn.pool.latent_queries" in k
            for k in variant_keys
        )
        has_pool_key_proj = any(
            "epoch_encoder.pool.key_proj." in k or "cnn.pool.key_proj." in k
            for k in variant_keys
        )
        has_pool_value_proj = any(
            "epoch_encoder.pool.value_proj." in k or "cnn.pool.value_proj." in k
            for k in variant_keys
        )
        has_pool_gate_conv = any(
            "epoch_encoder.pool.gate_conv." in k or "cnn.pool.gate_conv." in k
            for k in variant_keys
        )
        has_pool_gate_proj = any(
            "epoch_encoder.pool.gate_proj." in k or "cnn.pool.gate_proj." in k
            for k in variant_keys
        )
        has_pool_sustained = any(
            "epoch_encoder.pool.sustained_attn." in k or "cnn.pool.sustained_attn." in k
            for k in variant_keys
        )
        has_pool_transient = any(
            "epoch_encoder.pool.transient_attn." in k or "cnn.pool.transient_attn." in k
            for k in variant_keys
        )
        has_pool_project = any(
            "epoch_encoder.pool.project." in k or "cnn.pool.project." in k
            for k in variant_keys
        )

        logger.info("State dict checks:")
        logger.info("  has_pool_head_scorers: %s", has_pool_head_scorers)
        logger.info("  has_pool_latent_queries: %s", has_pool_latent_queries)
        logger.info("  has_pool_key_proj: %s", has_pool_key_proj)
        logger.info("  has_pool_value_proj: %s", has_pool_value_proj)
        logger.info("  has_pool_gate_conv: %s", has_pool_gate_conv)
        logger.info("  has_pool_gate_proj: %s", has_pool_gate_proj)
        logger.info("  has_pool_sustained: %s", has_pool_sustained)
        logger.info("  has_pool_transient: %s", has_pool_transient)
        logger.info("  has_pool_project: %s", has_pool_project)
        logger.info("Config check:")
        logger.info("  pooling_mode in config: %s", config_pooling_mode)

        has_pool_intra_epoch = any(
            "epoch_encoder.pool.tokenizer.depthwise." in k
            or "cnn.pool.tokenizer.depthwise." in k
            or k.endswith("epoch_encoder.pool.patch_pos")
            or k.endswith("cnn.pool.patch_pos")
            for k in variant_keys
        )
        logger.info("  has_pool_intra_epoch: %s", has_pool_intra_epoch)

        detected_pooling_mode = None
        # Checked first: IntraEpochAttentionPool also owns a ``pool.project.*``,
        # which would otherwise be misread as statistics pooling below.
        if has_pool_intra_epoch:
            detected_pooling_mode = "intra_epoch"
        elif has_pool_latent_queries and has_pool_key_proj and has_pool_value_proj:
            detected_pooling_mode = "learned"
        elif has_pool_head_scorers:
            detected_pooling_mode = "learned"
        elif has_pool_gate_conv or has_pool_gate_proj:
            detected_pooling_mode = "attentive_fusion"
        elif has_pool_sustained or has_pool_transient:
            detected_pooling_mode = "transient"
        elif has_pool_project:
            # FlexibleAsymmetricEpochCNN uses the same module for
            # pooling_mode='both' and pooling_mode='statistics'.
            detected_pooling_mode = "statistics"

        if detected_pooling_mode is not None:
            both_statistics_compatible = (
                detected_pooling_mode == "statistics" and config_pooling_mode == "both"
            )
            if config_pooling_mode is None:
                logger.info(
                    "  pooling_mode not in config, using detected: %s",
                    detected_pooling_mode,
                )
                model_kwargs["pooling_mode"] = detected_pooling_mode
                model_config["model_kwargs"] = model_kwargs
            elif both_statistics_compatible:
                logger.info(
                    "  ✓ pooling_mode is compatible (detected '%s', config '%s')",
                    detected_pooling_mode,
                    config_pooling_mode,
                )
            elif config_pooling_mode != detected_pooling_mode:
                logger.warning(
                    "  Config says pooling_mode='%s' but state_dict indicates '%s'. "
                    "Using state_dict value.",
                    config_pooling_mode,
                    detected_pooling_mode,
                )
                model_kwargs["pooling_mode"] = detected_pooling_mode
                model_config["model_kwargs"] = model_kwargs
            else:
                logger.info("  ✓ pooling_mode matches: %s", detected_pooling_mode)
        else:
            logger.info("  Could not detect pooling_mode, using config default")

        # -------------------------------------------------------------------
        # EPOCH-POOL WIDTH / INTRA-EPOCH GEOMETRY
        # -------------------------------------------------------------------
        # ``epoch_pool_output_dim`` decides whether the model rebuilds a learned
        # ``proj`` or an ``nn.Identity``. Newer checkpoints persist it; recover it
        # from the state dict when an older or hand-assembled config omits it.
        if "epoch_pool_output_dim" not in model_kwargs:
            has_top_level_proj = any(
                _strip_model_prefix(k) == "proj.weight" for k in state_dict.keys()
            )
            pool_project_shape = None
            for key, value in state_dict.items():
                ck = _strip_model_prefix(key)
                if ck in (
                    "epoch_encoder.pool.project.weight",
                    "cnn.pool.project.weight",
                ) and hasattr(value, "shape"):
                    pool_project_shape = tuple(value.shape)
                    break
            if not has_top_level_proj and pool_project_shape is not None:
                # No projection weights at all -> the epoch vector already had
                # d_model width and proj was nn.Identity.
                recovered = int(pool_project_shape[0])
                logger.info(
                    "  No top-level proj.weight in checkpoint; recovering "
                    "epoch_pool_output_dim=%d from pool.project",
                    recovered,
                )
                model_kwargs["epoch_pool_output_dim"] = recovered
                model_config["model_kwargs"] = model_kwargs

        if model_kwargs.get("pooling_mode") == "intra_epoch":
            patch_pos_shape = None
            block_indices: set[int] = set()
            for key, value in state_dict.items():
                ck = _strip_model_prefix(key)
                if ck in (
                    "epoch_encoder.pool.patch_pos",
                    "cnn.pool.patch_pos",
                ) and hasattr(value, "shape"):
                    patch_pos_shape = tuple(value.shape)
                marker = ".pool.blocks."
                if marker in ck:
                    tail = ck.split(marker, 1)[1].split(".", 1)[0]
                    if tail.isdigit():
                        block_indices.add(int(tail))
            if patch_pos_shape is not None and len(patch_pos_shape) == 3:
                model_kwargs["intra_epoch_patches"] = int(patch_pos_shape[1])
                model_kwargs["intra_epoch_dim"] = int(patch_pos_shape[2])
            if block_indices:
                model_kwargs["intra_epoch_layers"] = max(block_indices) + 1

            # Readout. The concat readout has no pool.aggregation_head, and its
            # pool.project is P times wider, so the two are not interchangeable.
            if "intra_epoch_aggregation" not in model_kwargs:
                has_agg_head = any(
                    ".pool.aggregation_head." in _strip_model_prefix(k)
                    for k in state_dict.keys()
                )
                detected_agg = "attention" if has_agg_head else "concat"
                logger.info(
                    "  intra_epoch_aggregation not in config; detected %r "
                    "(pool.aggregation_head %s)",
                    detected_agg,
                    "present" if has_agg_head else "absent",
                )
                model_kwargs["intra_epoch_aggregation"] = detected_agg

            # Pool topology. Version 2 owns pool.token_norm.* and
            # pool.log_temperature; a checkpoint written before those existed
            # must be rebuilt as version 1 or it gains two randomly initialized
            # LayerNorm tensors and a temperature that were never trained.
            if "intra_epoch_pool_version" not in model_kwargs:
                has_v2_keys = any(
                    ".pool.token_norm." in _strip_model_prefix(k)
                    or _strip_model_prefix(k).endswith(".pool.log_temperature")
                    for k in state_dict.keys()
                )
                detected_version = 2 if has_v2_keys else 1
                logger.info(
                    "  intra_epoch_pool_version not in config; detected %d from "
                    "the state dict (token_norm/log_temperature %s)",
                    detected_version,
                    "present" if has_v2_keys else "absent",
                )
                model_kwargs["intra_epoch_pool_version"] = detected_version
            if "intra_epoch_heads" not in model_kwargs:
                # Head count is invisible in the weights (MultiheadAttention packs
                # in_proj as [3d, d] for every head count), so a missing value
                # would silently change the attention pattern. Say so loudly.
                logger.warning(
                    "  Checkpoint uses pooling_mode='intra_epoch' but does not record "
                    "intra_epoch_heads; falling back to the default of 8. If the "
                    "model was trained with a different head count its attention "
                    "will not match, and state_dict loading will NOT report it."
                )
            model_config["model_kwargs"] = model_kwargs
            logger.info(
                "  intra_epoch geometry: patches=%s dim=%s layers=%s heads=%s",
                model_kwargs.get("intra_epoch_patches"),
                model_kwargs.get("intra_epoch_dim"),
                model_kwargs.get("intra_epoch_layers"),
                model_kwargs.get("intra_epoch_heads", 8),
            )

    # =========================================================================
    # VARIANCE PRESERVING NORM DETECTION
    # =========================================================================
    # VariancePreservingNorm has a unique 'var_weight' parameter that
    # differentiates it from standard LayerNorm. This is used in sequential
    # fusion for better N2 detection by preserving delta variance information.
    logger.info("=" * 80)
    logger.info("VARIANCE PRESERVING NORM DETECTION")
    logger.info("=" * 80)

    has_variance_preserving_norm = any("var_weight" in k for k in state_dict.keys())
    logger.info(
        f"  has_variance_preserving_norm (var_weight in keys): {has_variance_preserving_norm}"
    )

    config_variance_preserving = model_kwargs.get("use_variance_preserving_norm", False)

    if has_variance_preserving_norm and not config_variance_preserving:
        logger.info(
            "  Checkpoint uses VariancePreservingNorm but config does not specify "
            "use_variance_preserving_norm=True. Enabling to match checkpoint..."
        )
        model_kwargs["use_variance_preserving_norm"] = True
        model_config["model_kwargs"] = model_kwargs
    elif not has_variance_preserving_norm and config_variance_preserving:
        logger.info(
            "  Config specifies use_variance_preserving_norm=True but "
            "checkpoint does not use it. Disabling to match checkpoint..."
        )
        model_kwargs["use_variance_preserving_norm"] = False
        model_config["model_kwargs"] = model_kwargs
    elif has_variance_preserving_norm:
        logger.info("  ✓ VariancePreservingNorm enabled (config and checkpoint agree)")
    else:
        logger.info("  ✓ Standard LayerNorm used (config and checkpoint agree)")

    # =========================================================================
    # CENTER MASK TOKEN DETECTION
    # =========================================================================
    # mask_token is a learnable parameter created when center_mask_prob > 0.
    # During training, this masks the center epoch to force context learning.
    logger.info("=" * 80)
    logger.info("CENTER MASK TOKEN DETECTION")
    logger.info("=" * 80)

    has_mask_token_in_state = any(
        k == "mask_token" or k.endswith(".mask_token") for k in state_dict.keys()
    )
    # Check if mask_token is a real tensor (not None buffer)
    mask_token_is_tensor = False
    for k in state_dict.keys():
        if k == "mask_token" or k.endswith(".mask_token"):
            val = state_dict[k]
            if val is not None and hasattr(val, "shape"):
                mask_token_is_tensor = True
                break

    logger.info(f"  has_mask_token_in_state: {has_mask_token_in_state}")
    logger.info(f"  mask_token_is_tensor: {mask_token_is_tensor}")

    has_mask_token_in_config = model_kwargs.get("center_mask_prob", 0.0) > 0
    logger.info(
        f"  has_mask_token_in_config (center_mask_prob > 0): {has_mask_token_in_config}"
    )

    if mask_token_is_tensor and not has_mask_token_in_config:
        logger.warning(
            "Checkpoint state_dict contains mask_token parameter but "
            "model_config does not specify center_mask_prob > 0. "
            "Setting center_mask_prob=0.15 to match checkpoint..."
        )
        model_kwargs["center_mask_prob"] = 0.15
        model_config["model_kwargs"] = model_kwargs
    elif not mask_token_is_tensor and has_mask_token_in_config:
        logger.warning(
            "Model config specifies center_mask_prob > 0 but "
            "checkpoint state_dict does not contain mask_token parameter. "
            "Setting center_mask_prob=0.0 to match checkpoint..."
        )
        model_kwargs["center_mask_prob"] = 0.0
        model_config["model_kwargs"] = model_kwargs
    elif mask_token_is_tensor and has_mask_token_in_config:
        logger.info("✓ Center mask token enabled in both config and checkpoint")
    else:
        logger.info("✓ Center mask token not used (config and checkpoint agree)")

    # =========================================================================
    # NEIGHBOR PREDICTION HEAD DETECTION
    # =========================================================================
    # TemporalContextHead (neighbor_head) predicts neighboring epochs' stages
    # from the center representation. It has 2*n_neighbors Linear heads.
    logger.info("=" * 80)
    logger.info("NEIGHBOR PREDICTION HEAD DETECTION")
    logger.info("=" * 80)

    neighbor_head_keys = [
        k
        for k in state_dict.keys()
        if "neighbor_head.neighbor_heads." in k and ".weight" in k
    ]
    has_neighbor_head_in_state = len(neighbor_head_keys) > 0

    logger.info(f"  has_neighbor_head_in_state: {has_neighbor_head_in_state}")
    if neighbor_head_keys:
        logger.info(f"  Sample neighbor_head keys: {neighbor_head_keys[:3]}")

    has_neighbor_head_in_config = model_kwargs.get("use_neighbor_prediction", False)
    logger.info(f"  has_neighbor_head_in_config: {has_neighbor_head_in_config}")

    if has_neighbor_head_in_state and not has_neighbor_head_in_config:
        logger.warning(
            "Checkpoint state_dict contains neighbor_head parameters but "
            "model_config does not specify use_neighbor_prediction=True. "
            "Enabling neighbor prediction to match checkpoint..."
        )
        model_kwargs["use_neighbor_prediction"] = True

        # Infer neighbor_prediction_n from number of heads
        head_indices = set()
        for key in neighbor_head_keys:
            parts = key.split(".")
            try:
                idx = parts.index("neighbor_heads")
                if idx + 1 < len(parts):
                    head_indices.add(int(parts[idx + 1]))
            except (ValueError, IndexError):
                pass

        num_heads = len(head_indices)
        n_neighbors = num_heads // 2 if num_heads >= 2 else 2
        model_kwargs["neighbor_prediction_n"] = n_neighbors
        logger.info(
            f"  Extracted neighbor_prediction_n={n_neighbors} from {num_heads} heads"
        )
        model_config["model_kwargs"] = model_kwargs
    elif not has_neighbor_head_in_state and has_neighbor_head_in_config:
        logger.warning(
            "Model config specifies use_neighbor_prediction=True but "
            "checkpoint state_dict does not contain neighbor_head parameters. "
            "Disabling neighbor prediction to match checkpoint..."
        )
        model_kwargs["use_neighbor_prediction"] = False
        model_config["model_kwargs"] = model_kwargs
    elif has_neighbor_head_in_state and has_neighbor_head_in_config:
        logger.info("✓ Neighbor prediction enabled in both config and checkpoint")

        # Validate n_neighbors matches
        head_indices = set()
        for key in neighbor_head_keys:
            parts = key.split(".")
            try:
                idx = parts.index("neighbor_heads")
                if idx + 1 < len(parts):
                    head_indices.add(int(parts[idx + 1]))
            except (ValueError, IndexError):
                pass

        state_n_neighbors = len(head_indices) // 2 if len(head_indices) >= 2 else 2
        config_n_neighbors = model_kwargs.get("neighbor_prediction_n", 2)

        if state_n_neighbors != config_n_neighbors:
            logger.warning(
                f"  neighbor_prediction_n mismatch: config={config_n_neighbors}, "
                f"state_dict={state_n_neighbors}. Using state_dict value."
            )
            model_kwargs["neighbor_prediction_n"] = state_n_neighbors
            model_config["model_kwargs"] = model_kwargs
        else:
            logger.info(f"  ✓ neighbor_prediction_n matches: {state_n_neighbors}")
    else:
        logger.info("✓ Neighbor prediction not used (config and checkpoint agree)")

    # =========================================================================
    # SLEEPFM FUSION DETECTION
    # =========================================================================
    logger.info("=" * 80)
    logger.info("SLEEPFM FUSION DETECTION")
    logger.info("=" * 80)

    has_epoch_feature_fusion_in_state = any(
        "epoch_feature_fusion." in k for k in state_dict.keys()
    )
    has_sleepfm_feature_fusion = any(
        "sleepfm_fusion.feature_fusion" in k for k in state_dict.keys()
    )
    has_sleepfm_temporal_pool = any(
        "sleepfm_fusion.temporal_pool" in k for k in state_dict.keys()
    )
    has_sleepfm_output_proj = any(
        "sleepfm_fusion.output_proj" in k for k in state_dict.keys()
    )
    has_sleepfm_pool_in_state = has_sleepfm_feature_fusion and has_sleepfm_temporal_pool
    has_sleepfm_in_state = (
        has_epoch_feature_fusion_in_state or has_sleepfm_pool_in_state
    )

    detected_sleepfm_mode: str | None
    if has_epoch_feature_fusion_in_state:
        detected_sleepfm_mode = "fuse_only"
    elif has_sleepfm_pool_in_state:
        detected_sleepfm_mode = "pool"
    else:
        detected_sleepfm_mode = None

    logger.info(
        f"  has_epoch_feature_fusion_in_state: {has_epoch_feature_fusion_in_state}"
    )
    logger.info(f"  has_sleepfm_feature_fusion: {has_sleepfm_feature_fusion}")
    logger.info(f"  has_sleepfm_temporal_pool: {has_sleepfm_temporal_pool}")
    logger.info(f"  has_sleepfm_output_proj: {has_sleepfm_output_proj}")
    logger.info(f"  has_sleepfm_in_state: {has_sleepfm_in_state}")
    logger.info(f"  detected_sleepfm_mode: {detected_sleepfm_mode}")

    supports_sleepfm_mode = model_config.get("model_type") == "TransformerContextNet"
    has_sleepfm_in_config = model_kwargs.get("use_sleepfm_fusion", False)
    config_sleepfm_mode = str(model_kwargs.get("sleepfm_mode", "pool"))
    logger.info(f"  has_sleepfm_in_config: {has_sleepfm_in_config}")
    if supports_sleepfm_mode:
        logger.info(f"  config_sleepfm_mode: {config_sleepfm_mode}")

    if (
        supports_sleepfm_mode
        and has_sleepfm_in_state
        and "sleepfm_mode" not in model_kwargs
    ):
        model_kwargs["sleepfm_mode"] = detected_sleepfm_mode or "pool"
        config_sleepfm_mode = str(model_kwargs["sleepfm_mode"])
        model_config["model_kwargs"] = model_kwargs
        logger.info(
            "  Added missing sleepfm_mode=%s from checkpoint", config_sleepfm_mode
        )

    if has_epoch_feature_fusion_in_state and has_sleepfm_pool_in_state:
        logger.warning(
            "Checkpoint contains both epoch_feature_fusion and sleepfm_fusion keys. "
            "Preferring sleepfm_mode='fuse_only'."
        )

    if has_sleepfm_in_state and not has_sleepfm_in_config:
        logger.warning(
            "Checkpoint state_dict contains SleepFM fusion parameters but model_config "
            "does not specify use_sleepfm_fusion=True. Enabling SleepFM fusion."
        )
        model_kwargs["use_sleepfm_fusion"] = True
        if supports_sleepfm_mode:
            model_kwargs["sleepfm_mode"] = detected_sleepfm_mode or "pool"
            config_sleepfm_mode = str(model_kwargs["sleepfm_mode"])

        if (not supports_sleepfm_mode) or config_sleepfm_mode == "pool":
            has_hierarchical_pool = any(
                "sleepfm_fusion.temporal_pool.local_pool" in k
                for k in state_dict.keys()
            )
            model_kwargs["sleepfm_temporal_pool_type"] = (
                "hierarchical" if has_hierarchical_pool else "attention"
            )
            has_learned_bias = any(
                "sleepfm_fusion.temporal_pool.position_bias" in k
                for k in state_dict.keys()
            )
            if has_learned_bias:
                model_kwargs["sleepfm_center_bias"] = "learned"
            else:
                model_kwargs.setdefault("sleepfm_center_bias", "gaussian")
            model_kwargs.setdefault("sleepfm_fusion_num_heads", 4)
            model_kwargs.setdefault("sleepfm_center_bias_strength", 2.0)
            model_kwargs.setdefault("sleepfm_temperature", 1.0)

        # SleepFM fusion requires engineered features
        if not model_kwargs.get("use_feature_extraction", False):
            logger.warning(
                "  SleepFM fusion requires use_feature_extraction=True. Enabling it."
            )
            model_kwargs["use_feature_extraction"] = True

        model_config["model_kwargs"] = model_kwargs
    elif not has_sleepfm_in_state and has_sleepfm_in_config:
        logger.warning(
            "Model config specifies use_sleepfm_fusion=True but checkpoint state_dict "
            "does not contain SleepFM fusion parameters. Disabling SleepFM fusion."
        )
        model_kwargs["use_sleepfm_fusion"] = False
        if supports_sleepfm_mode:
            model_kwargs["sleepfm_mode"] = "pool"
        model_config["model_kwargs"] = model_kwargs
    elif has_sleepfm_in_state and has_sleepfm_in_config:
        logger.info("✓ SleepFM fusion enabled in both config and checkpoint")
        if (
            supports_sleepfm_mode
            and detected_sleepfm_mode is not None
            and config_sleepfm_mode != detected_sleepfm_mode
        ):
            logger.warning(
                "  sleepfm_mode mismatch: config=%s, state_dict=%s. Using state_dict value.",
                config_sleepfm_mode,
                detected_sleepfm_mode,
            )
            model_kwargs["sleepfm_mode"] = detected_sleepfm_mode
            config_sleepfm_mode = detected_sleepfm_mode
            model_config["model_kwargs"] = model_kwargs

        if (not supports_sleepfm_mode) or config_sleepfm_mode == "pool":
            has_hierarchical_pool = any(
                "sleepfm_fusion.temporal_pool.local_pool" in k
                for k in state_dict.keys()
            )
            detected_pool_type = (
                "hierarchical" if has_hierarchical_pool else "attention"
            )
            config_pool_type = model_kwargs.get(
                "sleepfm_temporal_pool_type", "attention"
            )
            if detected_pool_type != config_pool_type:
                logger.warning(
                    "  sleepfm_temporal_pool_type mismatch: config=%s, state_dict=%s. "
                    "Using state_dict value.",
                    config_pool_type,
                    detected_pool_type,
                )
                model_kwargs["sleepfm_temporal_pool_type"] = detected_pool_type
                model_config["model_kwargs"] = model_kwargs
            else:
                logger.info(
                    "  ✓ sleepfm_temporal_pool_type matches: %s", detected_pool_type
                )

            has_learned_bias_in_state = any(
                "sleepfm_fusion.temporal_pool.position_bias" in k
                for k in state_dict.keys()
            )
            config_center_bias = model_kwargs.get("sleepfm_center_bias", "gaussian")
            detected_center_bias = (
                "learned" if has_learned_bias_in_state else "gaussian"
            )
            if detected_center_bias != config_center_bias:
                logger.warning(
                    "  sleepfm_center_bias mismatch: config='%s', state_dict='%s'. "
                    "Using state_dict value.",
                    config_center_bias,
                    detected_center_bias,
                )
                model_kwargs["sleepfm_center_bias"] = detected_center_bias
                model_config["model_kwargs"] = model_kwargs
            else:
                logger.info("  ✓ sleepfm_center_bias matches: %s", detected_center_bias)
    else:
        logger.info("✓ SleepFM fusion not used (config and checkpoint agree)")

    # TransformerContextNet now treats SleepFM as a post-transformer fusion path
    # that is mutually exclusive with pre-transformer sequential fusion.
    if (
        model_config.get("model_type") == "TransformerContextNet"
        and model_kwargs.get("use_sleepfm_fusion", False)
        and model_kwargs.get("fusion_mode") == "sequential"
    ):
        logger.warning(
            "TransformerContextNet no longer allows use_sleepfm_fusion with "
            "fusion_mode='sequential'. Switching fusion_mode to 'learned_weight' "
            "for checkpoint compatibility."
        )
        model_kwargs["fusion_mode"] = "learned_weight"
        model_config["model_kwargs"] = model_kwargs

    removed_patch_flags = (
        "use_subepoch_patches",
        "use_patch_features",
        "use_dual_stream_patches",
        "use_intra_epoch_patches",
    )
    if any(bool(model_kwargs.get(flag, False)) for flag in removed_patch_flags):
        raise ValueError(
            "Patch-based checkpoints are no longer supported in this repository. "
            "The checkpoint model_config still requests removed patch-tokenization "
            "or patch-feature options."
        )

    # =========================================================================
    # LEARNABLE POSITIONAL ENCODING DETECTION
    # =========================================================================
    # pos_encoding is a learnable nn.Parameter [1, max_len, d_model] created when
    # use_learnable_pe=True. When False, the model uses sinusoidal PositionalEncoding
    # module with a registered buffer named 'pe' (accessed as pos.pe).
    logger.info("=" * 80)
    logger.info("LEARNABLE POSITIONAL ENCODING DETECTION")
    logger.info("=" * 80)

    has_learnable_pe_in_state = any(
        k == "pos_encoding" or k.endswith(".pos_encoding") for k in state_dict.keys()
    )
    has_sinusoidal_pe_in_state = any(
        k == "pos.pe" or k.endswith(".pos.pe") for k in state_dict.keys()
    )

    logger.info(
        f"  has_learnable_pe_in_state (pos_encoding): {has_learnable_pe_in_state}"
    )
    logger.info(f"  has_sinusoidal_pe_in_state (pos.pe): {has_sinusoidal_pe_in_state}")

    has_learnable_pe_in_config = model_kwargs.get("use_learnable_pe", False)
    logger.info(f"  has_learnable_pe_in_config: {has_learnable_pe_in_config}")

    if has_learnable_pe_in_state and not has_learnable_pe_in_config:
        logger.warning(
            "Checkpoint contains learnable pos_encoding parameter but "
            "model_config does not specify use_learnable_pe=True. "
            "Setting use_learnable_pe=True to match checkpoint..."
        )
        model_kwargs["use_learnable_pe"] = True
        model_config["model_kwargs"] = model_kwargs
    elif not has_learnable_pe_in_state and has_learnable_pe_in_config:
        logger.warning(
            "Model config specifies use_learnable_pe=True but "
            "checkpoint does not contain learnable pos_encoding parameter. "
            "Setting use_learnable_pe=False to match checkpoint..."
        )
        model_kwargs["use_learnable_pe"] = False
        model_config["model_kwargs"] = model_kwargs
    elif has_learnable_pe_in_state and has_learnable_pe_in_config:
        logger.info("✓ Learnable PE enabled in both config and checkpoint")
    else:
        logger.info("✓ Sinusoidal PE used (config and checkpoint agree)")

    # Log pos_scale effective value if present in checkpoint
    for k in state_dict.keys():
        clean_k = k[6:] if k.startswith("model.") else k
        if clean_k == "pos_scale" or clean_k.endswith(".pos_scale"):
            raw_val = float(state_dict[k])
            import math as _math

            effective_scale = _math.log1p(_math.exp(raw_val))  # softplus
            logger.info(
                "  pos_scale: raw=%.4f, effective=%.4f (position/content balance)",
                raw_val,
                effective_scale,
            )
            break

    # =========================================================================
    # MODALITY-AWARE STEM RATIO DETECTION
    # =========================================================================
    # The ModalityAwareStem computes filter dimensions dynamically based on
    # eeg_ratio, eog_ratio, emg_ratio. These must match between training and
    # inference or the state_dict won't load. Detect from checkpoint shapes.
    logger.info("=" * 80)
    logger.info("MODALITY-AWARE STEM RATIO DETECTION")
    logger.info("=" * 80)

    # Check if checkpoint has modality-aware stem
    has_modality_stem = any(
        "epoch_encoder.stem.eeg_branch.fusion" in k or "cnn.stem.eeg_branch.fusion" in k
        for k in state_dict.keys()
    )
    # Check if it's FlexibleModalityAwareStem (has dilated_branches) vs ModalityAwareStem
    is_flexible_stem = any(
        "epoch_encoder.stem.eeg_branch.dilated_branches" in k
        or "cnn.stem.eeg_branch.dilated_branches" in k
        for k in state_dict.keys()
    )
    logger.info(f"  has_modality_aware_stem: {has_modality_stem}")
    logger.info(f"  is_flexible_stem: {is_flexible_stem}")

    # Initialize checkpoint channel variables (used later for width detection)
    checkpoint_eeg_out_ch = None
    checkpoint_eog_out_ch = None
    checkpoint_emg_out_ch = None

    if has_modality_stem:
        # Extract branch output channels from checkpoint state dict shapes
        # EEG branch: fusion layer is [eeg_out_ch, eeg_out_ch, 1]
        eeg_fusion_key = None
        for k in state_dict.keys():
            if (
                "epoch_encoder.stem.eeg_branch.fusion.0.weight" in k
                or "cnn.stem.eeg_branch.fusion.0.weight" in k
            ):
                eeg_fusion_key = k
                break

        if is_flexible_stem:
            # FlexibleModalityAwareStem: each branch uses FlexiblePhysiologicalStem
            # EMG/EOG out_ch can be extracted from their fusion layers
            emg_fusion_key = None
            eog_fusion_key = None
            for k in state_dict.keys():
                if (
                    "epoch_encoder.stem.emg_branch.fusion.0.weight" in k
                    or "cnn.stem.emg_branch.fusion.0.weight" in k
                ):
                    emg_fusion_key = k
                elif (
                    "epoch_encoder.stem.eog_branch.fusion.0.weight" in k
                    or "cnn.stem.eog_branch.fusion.0.weight" in k
                ):
                    eog_fusion_key = k
            # Set these to None so the standard pattern detection is skipped
            emg_burst_key = None
            emg_envelope_key = None
            eog_rem_key = None
            eog_sem_key = None
        else:
            # ModalityAwareStem: EMG has burst + envelope, EOG has rem + sem
            emg_burst_key = None
            emg_envelope_key = None
            for k in state_dict.keys():
                if (
                    "epoch_encoder.stem.emg_branch.burst.0.weight" in k
                    or "cnn.stem.emg_branch.burst.0.weight" in k
                ):
                    emg_burst_key = k
                elif (
                    "epoch_encoder.stem.emg_branch.envelope.0.weight" in k
                    or "cnn.stem.emg_branch.envelope.0.weight" in k
                ):
                    emg_envelope_key = k

            # EOG branch: rem + sem output channels (if present)
            eog_rem_key = None
            eog_sem_key = None
            for k in state_dict.keys():
                if (
                    "epoch_encoder.stem.eog_branch.rem.0.weight" in k
                    or "cnn.stem.eog_branch.rem.0.weight" in k
                ):
                    eog_rem_key = k
                elif (
                    "epoch_encoder.stem.eog_branch.sem.0.weight" in k
                    or "cnn.stem.eog_branch.sem.0.weight" in k
                ):
                    eog_sem_key = k
            emg_fusion_key = None
            eog_fusion_key = None

        if eeg_fusion_key:
            # Fusion layer weight shape: [out_ch, out_ch, 1] or [out_ch, in_ch, 1]
            eeg_fusion_shape = state_dict[eeg_fusion_key].shape
            checkpoint_eeg_out_ch = eeg_fusion_shape[0]  # Output channels
            logger.info(
                f"  EEG fusion weight shape: {list(eeg_fusion_shape)} -> eeg_out_ch={checkpoint_eeg_out_ch}"
            )

        if emg_burst_key and emg_envelope_key:
            # ModalityAwareStem pattern: burst + envelope
            burst_shape = state_dict[emg_burst_key].shape
            envelope_shape = state_dict[emg_envelope_key].shape
            burst_ch = burst_shape[0]
            envelope_ch = envelope_shape[0]
            checkpoint_emg_out_ch = burst_ch + envelope_ch
            logger.info(
                f"  EMG burst shape: {list(burst_shape)} -> burst_ch={burst_ch}"
            )
            logger.info(
                f"  EMG envelope shape: {list(envelope_shape)} -> envelope_ch={envelope_ch}"
            )
            logger.info(f"  EMG total: emg_out_ch={checkpoint_emg_out_ch}")
        elif emg_fusion_key:
            # FlexibleModalityAwareStem pattern: single fusion layer
            emg_fusion_shape = state_dict[emg_fusion_key].shape
            checkpoint_emg_out_ch = emg_fusion_shape[0]
            logger.info(
                f"  EMG fusion shape: {list(emg_fusion_shape)} -> emg_out_ch={checkpoint_emg_out_ch}"
            )

        if eog_rem_key and eog_sem_key:
            # ModalityAwareStem pattern: rem + sem
            rem_shape = state_dict[eog_rem_key].shape
            sem_shape = state_dict[eog_sem_key].shape
            rem_ch = rem_shape[0]
            sem_ch = sem_shape[0]
            checkpoint_eog_out_ch = rem_ch + sem_ch
            logger.info(f"  EOG rem shape: {list(rem_shape)} -> rem_ch={rem_ch}")
            logger.info(f"  EOG sem shape: {list(sem_shape)} -> sem_ch={sem_ch}")
            logger.info(f"  EOG total: eog_out_ch={checkpoint_eog_out_ch}")
        elif eog_fusion_key:
            # FlexibleModalityAwareStem pattern: single fusion layer
            eog_fusion_shape = state_dict[eog_fusion_key].shape
            checkpoint_eog_out_ch = eog_fusion_shape[0]
            logger.info(
                f"  EOG fusion shape: {list(eog_fusion_shape)} -> eog_out_ch={checkpoint_eog_out_ch}"
            )

        # Calculate total stem output and infer ratios
        if checkpoint_eeg_out_ch is not None:
            # Stem total = sum of all branch outputs
            total_out_ch = checkpoint_eeg_out_ch
            if checkpoint_emg_out_ch is not None:
                total_out_ch += checkpoint_emg_out_ch
            if checkpoint_eog_out_ch is not None:
                total_out_ch += checkpoint_eog_out_ch

            # If no EOG detected but we have EEG and EMG, EOG is the remainder
            if checkpoint_eog_out_ch is None and checkpoint_emg_out_ch is not None:
                # Standard base size is 64; try to detect from pattern
                # Common stem sizes: 48 (small), 64 (base), 96 (large)
                possible_stem_sizes = [48, 64, 96, 128]
                detected_stem_size = None
                for size in possible_stem_sizes:
                    if checkpoint_eeg_out_ch + checkpoint_emg_out_ch <= size:
                        detected_stem_size = size
                        break
                if detected_stem_size:
                    checkpoint_eog_out_ch = (
                        detected_stem_size
                        - checkpoint_eeg_out_ch
                        - checkpoint_emg_out_ch
                    )
                    total_out_ch = detected_stem_size
                    logger.info(f"  Inferred stem total out_ch: {detected_stem_size}")
                    logger.info(f"  Inferred EOG out_ch: {checkpoint_eog_out_ch}")

            # Calculate ratios
            if total_out_ch > 0:
                detected_eeg_ratio = checkpoint_eeg_out_ch / total_out_ch
                detected_emg_ratio = (checkpoint_emg_out_ch or 0) / total_out_ch
                detected_eog_ratio = (checkpoint_eog_out_ch or 0) / total_out_ch

                logger.info("  Detected modality ratios from checkpoint:")
                logger.info(f"    eeg_ratio: {detected_eeg_ratio:.4f}")
                logger.info(f"    eog_ratio: {detected_eog_ratio:.4f}")
                logger.info(f"    emg_ratio: {detected_emg_ratio:.4f}")

                # Compare with config defaults and override if different
                config_eeg_ratio = model_kwargs.get("eeg_ratio", 0.60)
                config_eog_ratio = model_kwargs.get("eog_ratio", 0.25)
                config_emg_ratio = model_kwargs.get("emg_ratio", 0.15)

                # Check if ratios differ significantly (threshold: 0.02)
                ratio_threshold = 0.02
                needs_override = (
                    abs(detected_eeg_ratio - config_eeg_ratio) > ratio_threshold
                    or abs(detected_eog_ratio - config_eog_ratio) > ratio_threshold
                    or abs(detected_emg_ratio - config_emg_ratio) > ratio_threshold
                )

                if needs_override:
                    logger.info(
                        "  ⚠️  Checkpoint modality ratios differ from config defaults!"
                    )
                    logger.info(
                        f"    Config: eeg={config_eeg_ratio}, eog={config_eog_ratio}, emg={config_emg_ratio}"
                    )
                    logger.info(
                        f"    Checkpoint: eeg={detected_eeg_ratio:.4f}, eog={detected_eog_ratio:.4f}, emg={detected_emg_ratio:.4f}"
                    )
                    logger.info("    Overriding model_kwargs with checkpoint ratios...")

                    model_kwargs["eeg_ratio"] = detected_eeg_ratio
                    model_kwargs["eog_ratio"] = detected_eog_ratio
                    model_kwargs["emg_ratio"] = detected_emg_ratio
                    model_config["model_kwargs"] = model_kwargs
                else:
                    logger.info(
                        "  ✓ Modality ratios match between checkpoint and config"
                    )

                # For FlexibleModalityAwareStem, also pass explicit channel counts
                # to reproduce the checkpoint architecture exactly.
                if is_flexible_stem:
                    logger.info(
                        "  Setting explicit stem channel counts for FlexibleModalityAwareStem:"
                    )
                    if checkpoint_eeg_out_ch is not None:
                        model_kwargs["stem_eeg_out_ch"] = checkpoint_eeg_out_ch
                        logger.info(f"    stem_eeg_out_ch: {checkpoint_eeg_out_ch}")
                    if checkpoint_eog_out_ch is not None:
                        model_kwargs["stem_eog_out_ch"] = checkpoint_eog_out_ch
                        logger.info(f"    stem_eog_out_ch: {checkpoint_eog_out_ch}")
                    if checkpoint_emg_out_ch is not None:
                        model_kwargs["stem_emg_out_ch"] = checkpoint_emg_out_ch
                        logger.info(f"    stem_emg_out_ch: {checkpoint_emg_out_ch}")
                    model_config["model_kwargs"] = model_kwargs
    else:
        logger.info("  ✓ No modality-aware stem detected (using standard stem)")

    # =========================================================================
    # WIDTH DETECTION FOR FLEXIBLE-STEM ENCODERS
    # =========================================================================
    detected_encoder_variant = model_kwargs.get("epoch_encoder_variant")
    if detected_encoder_variant == "sleep_staging_hybrid":
        logger.info("=" * 80)
        logger.info("WIDTH DETECTION FOR SLEEP STAGING HYBRID")
        logger.info("=" * 80)

        detected_widths: list[int] = []
        stem_width = 0
        if checkpoint_eeg_out_ch is not None:
            stem_width += checkpoint_eeg_out_ch
        if checkpoint_eog_out_ch is not None:
            stem_width += checkpoint_eog_out_ch
        if checkpoint_emg_out_ch is not None:
            stem_width += checkpoint_emg_out_ch
        if stem_width > 0:
            detected_widths.append(stem_width)
            logger.info(f"  Detected stem width (widths[0]): {stem_width}")

        def _find_key(patterns: tuple[str, ...]) -> str | None:
            for raw_key in state_dict.keys():
                clean_key = raw_key[6:] if raw_key.startswith("model.") else raw_key
                if any(pattern in clean_key for pattern in patterns):
                    return raw_key
            return None

        stage_patterns = (
            (
                "epoch_encoder.stage1.1.block.fusion.0.weight",
                "epoch_encoder.stage1.0.block.fusion.0.weight",
            ),
            (
                "epoch_encoder.stage2.0.block.fusion.0.weight",
                "epoch_encoder.stage2.1.conv1.weight",
            ),
            ("epoch_encoder.stage3.0.conv1.weight",),
            ("epoch_encoder.stage4.0.conv1.weight",),
        )
        for stage_idx, patterns in enumerate(stage_patterns, start=1):
            found_key = _find_key(patterns)
            if found_key is None:
                continue
            stage_width = state_dict[found_key].shape[0]
            detected_widths.append(stage_width)
            logger.info(f"  Detected stage{stage_idx} width: {stage_width}")

        if len(detected_widths) == 5:
            detected_widths_tuple = tuple(detected_widths)
            config_widths = model_kwargs.get("cnn_widths")
            if config_widths != detected_widths_tuple:
                logger.info("  ⚠️  Checkpoint widths differ from config!")
                logger.info(f"    Config widths: {config_widths}")
                logger.info(f"    Detected widths: {detected_widths_tuple}")
                logger.info("    Overriding model_kwargs with checkpoint widths...")
                model_kwargs["cnn_widths"] = detected_widths_tuple
                model_config["model_kwargs"] = model_kwargs
            else:
                logger.info(f"  ✓ Widths match: {detected_widths_tuple}")
        else:
            logger.info(
                "  Could not detect all 5 width values for sleep_staging_hybrid "
                "(found %d)",
                len(detected_widths),
            )

        logger.info("-" * 80)
        logger.info("STAGE STRIDE DETECTION FOR SLEEP STAGING HYBRID")
        logger.info("-" * 80)

        def _hybrid_stage_stride(stage_name: str) -> int | None:
            downsample_markers = (
                f"epoch_encoder.{stage_name}.0.downsample.",
                f"epoch_encoder.{stage_name}.0.skip.1.",
                f"epoch_encoder.{stage_name}.0.block.pool.",
                f"epoch_encoder.{stage_name}.0.block.shortcut.2.",
            )
            has_downsample = any(
                any(marker in clean_key for marker in downsample_markers)
                for clean_key in variant_keys
            )
            if has_downsample:
                return 2

            conv_marker = f"epoch_encoder.{stage_name}.0.conv1.weight"
            if any(conv_marker in clean_key for clean_key in variant_keys):
                return 1
            return None

        detected_stage_strides = tuple(
            stride if stride is not None else 1
            for stride in (
                _hybrid_stage_stride("stage1"),
                _hybrid_stage_stride("stage2"),
                _hybrid_stage_stride("stage3"),
                _hybrid_stage_stride("stage4"),
            )
        )
        if any(
            _hybrid_stage_stride(stage_name) is not None
            for stage_name in ("stage1", "stage2", "stage3", "stage4")
        ):
            config_stage_strides = model_kwargs.get("cnn_stage_strides")
            if config_stage_strides is None:
                config_stage_strides = model_kwargs.get("stage_strides")
            normalized_config_stage_strides = (
                tuple(config_stage_strides)
                if config_stage_strides is not None
                else None
            )
            if normalized_config_stage_strides != detected_stage_strides:
                logger.info(
                    "  Overriding hybrid stage strides from checkpoint state_dict"
                )
                logger.info(
                    "    Config stage strides: %s", normalized_config_stage_strides
                )
                logger.info("    Detected stage strides: %s", detected_stage_strides)
                model_kwargs["cnn_stage_strides"] = detected_stage_strides
                model_kwargs.pop("stage_strides", None)
                model_config["model_kwargs"] = model_kwargs
            else:
                logger.info(
                    "  ✓ Hybrid stage strides match: %s", detected_stage_strides
                )
    elif detected_encoder_variant == "flexible_asymmetric" or is_flexible_stem:
        logger.info("=" * 80)
        logger.info("WIDTH DETECTION FOR FLEXIBLE ASYMMETRIC")
        logger.info("=" * 80)

        # Detect widths from stage output shapes
        # Each stage ends with a fusion layer: [out_ch, out_ch, 1]
        detected_widths = []

        # Stem width (widths[0]): sum of modality branch fusion outputs
        stem_width = 0
        if checkpoint_eeg_out_ch is not None:
            stem_width += checkpoint_eeg_out_ch
        if checkpoint_eog_out_ch is not None:
            stem_width += checkpoint_eog_out_ch
        if checkpoint_emg_out_ch is not None:
            stem_width += checkpoint_emg_out_ch
        if stem_width > 0:
            detected_widths.append(stem_width)
            logger.info(f"  Detected stem width (widths[0]): {stem_width}")

        # Stage widths from fusion layers
        # Check both epoch_encoder.stage* and cnn.stage* key prefixes.
        # FlexibleAsymmetric uses wrapped keys like stageX.Y.block.fusion.0.weight.
        def _find_stage_fusion_key(stage_idx: int, block_idx: int) -> str | None:
            key_patterns = [
                f"epoch_encoder.stage{stage_idx}.{block_idx}.block.fusion.0.weight",
                f"epoch_encoder.stage{stage_idx}.{block_idx}.block.fusion.weight",
                f"epoch_encoder.stage{stage_idx}.{block_idx}.fusion.0.weight",
                f"epoch_encoder.stage{stage_idx}.{block_idx}.fusion.weight",
                f"cnn.stage{stage_idx}.{block_idx}.block.fusion.0.weight",
                f"cnn.stage{stage_idx}.{block_idx}.block.fusion.weight",
                f"cnn.stage{stage_idx}.{block_idx}.fusion.0.weight",
                f"cnn.stage{stage_idx}.{block_idx}.fusion.weight",
            ]
            for raw_key in state_dict.keys():
                clean_key = raw_key[6:] if raw_key.startswith("model.") else raw_key
                if any(pattern in clean_key for pattern in key_patterns):
                    return raw_key
            return None

        for stage_idx in range(1, 5):
            # Prefer the second block, then fallback to first block.
            found_key = _find_stage_fusion_key(stage_idx, 1)
            if found_key:
                stage_width = state_dict[found_key].shape[0]
                detected_widths.append(stage_width)
                logger.info(f"  Detected stage{stage_idx} width: {stage_width}")
            else:
                found_key = _find_stage_fusion_key(stage_idx, 0)
                if found_key:
                    stage_width = state_dict[found_key].shape[0]
                    detected_widths.append(stage_width)
                    logger.info(
                        f"  Detected stage{stage_idx} width (from block 0): {stage_width}"
                    )

        if len(detected_widths) >= 3:
            # Pad to 5 elements if needed (repeat last)
            while len(detected_widths) < 5:
                detected_widths.append(detected_widths[-1])
            resolved_widths = tuple(detected_widths[:5])

            # TransformerContextNet uses 'cnn_widths'.
            detected_model_type = model_config.get(
                "model_type", "TransformerContextNet"
            )
            widths_key = (
                "cnn_widths"
                if detected_model_type == "TransformerContextNet"
                else "widths"
            )
            config_widths = model_kwargs.get(widths_key)
            if config_widths is None and widths_key == "cnn_widths":
                config_widths = model_kwargs.get("widths")
            if config_widths != resolved_widths:
                logger.info("  ⚠️  Checkpoint widths differ from config!")
                logger.info(f"    Config widths: {config_widths}")
                logger.info(f"    Detected widths: {resolved_widths}")
                logger.info("    Overriding model_kwargs with checkpoint widths...")
                model_kwargs[widths_key] = resolved_widths
                if widths_key == "cnn_widths":
                    model_kwargs.pop("widths", None)
                model_config["model_kwargs"] = model_kwargs
            else:
                logger.info(f"  ✓ Widths match: {resolved_widths}")
        else:
            logger.info(
                f"  Could not detect enough width values ({len(detected_widths)} found)"
            )

    # =========================================================================
    # LEGACY FLEXIBLE-STEM SCHEMA DETECTION
    # =========================================================================
    # Two committed architecture changes shifted module indices inside the
    # flexible stem and its dilated blocks, and a third raised the anti-alias
    # tap counts. Checkpoints predating them cannot be loaded into the current
    # layout, so the original layout has to be reconstructed rather than the
    # state dict rewritten: the orderings are not function-preserving and the
    # removed filters are real computation, not bookkeeping.
    logger.info("=" * 80)
    logger.info("LEGACY FLEXIBLE-STEM SCHEMA DETECTION")
    logger.info("=" * 80)

    legacy_keys = {
        _strip_model_prefix(k): v for k, v in state_dict.items() if isinstance(k, str)
    }

    # (1) global_branch ordering. Index 1 is a 1x1 Conv1d weight (rank 3) in the
    # legacy layout and a norm weight (rank 1) in the current one.
    global_branch_ranks: dict[str, int] = {
        k: int(v.ndim)
        for k, v in legacy_keys.items()
        if _GLOBAL_BRANCH_WEIGHT_RE.match(k) and hasattr(v, "ndim")
    }
    distinct_ranks = sorted(set(global_branch_ranks.values()))
    logger.info(f"  global_branch.1 tensor ranks: {distinct_ranks}")

    if len(distinct_ranks) > 1:
        logger.error(
            "Checkpoint mixes global_branch layouts across modality branches: %s. "
            "This cannot be reconstructed without silently mismatching normalization "
            "statistics on part of the stem.",
            global_branch_ranks,
        )
        raise InferenceModelLoadError(
            "Checkpoint contains inconsistent global_branch layouts across modality "
            f"branches (tensor ranks {distinct_ranks}); refusing to guess a layout."
        )

    has_legacy_global_branch = bool(distinct_ranks) and distinct_ranks[0] == 3
    logger.info(
        f"  legacy global_branch ordering (GAP->conv->norm->act): {has_legacy_global_branch}"
    )

    # (2) Per-branch anti-alias modules. Index 0 under ``branches.N.`` can only
    # be the anti-alias filter that used to precede high-dilation convolutions.
    legacy_branch_aa_keys = [k for k in legacy_keys if _LEGACY_BRANCH_AA_RE.match(k)]
    has_legacy_branch_aa = bool(legacy_branch_aa_keys)
    logger.info(
        f"  legacy per-branch anti-alias modules: {has_legacy_branch_aa} "
        f"({len(legacy_branch_aa_keys)} branches)"
    )

    # (3) Anti-alias tap counts. These buffers are non-persistent now, so a
    # mismatch never surfaces as a load error -- the model just rebuilds a
    # different filter. Read the lengths the checkpoint actually stored.
    stem_taps_seen: list[int] = [
        int(v.shape[-1])
        for k, v in legacy_keys.items()
        if _STEM_POOL_KERNEL_RE.match(k) and hasattr(v, "shape")
    ]
    stage_taps: dict[int, int] = {}
    for key, value in legacy_keys.items():
        match = _STAGE_POOL_KERNEL_RE.match(key)
        if match is not None and hasattr(value, "shape"):
            stage_taps[int(match.group(1))] = int(value.shape[-1])

    detected_stem_taps: int | None = None
    if stem_taps_seen:
        if len(set(stem_taps_seen)) > 1:
            detected_stem_taps = max(set(stem_taps_seen), key=stem_taps_seen.count)
            logger.warning(
                "Stem anti-alias tap counts disagree across modality branches (%s); "
                "using the most common value %d.",
                sorted(set(stem_taps_seen)),
                detected_stem_taps,
            )
        else:
            detected_stem_taps = stem_taps_seen[0]
    logger.info(
        f"  checkpoint stem AA taps: {detected_stem_taps} (current default: 47)"
    )
    logger.info(
        f"  checkpoint stage AA taps by stage: {dict(sorted(stage_taps.items()))} "
        "(current default: (31, 23, 23, 23))"
    )

    # Cross-check against the residual shortcut filters, which are built from
    # the same per-stage tap count. A disagreement means the derivation is wrong.
    for key, value in legacy_keys.items():
        match = _STAGE_SHORTCUT_KERNEL_RE.match(key)
        if match is None or not hasattr(value, "shape"):
            continue
        stage_idx = int(match.group(1))
        shortcut_taps = int(value.shape[-1])
        if stage_idx in stage_taps and stage_taps[stage_idx] != shortcut_taps:
            logger.warning(
                "Stage %d anti-alias tap counts disagree between pool (%d) and "
                "shortcut (%d); tap detection may be unreliable.",
                stage_idx,
                stage_taps[stage_idx],
                shortcut_taps,
            )

    detected_variant_for_aa = model_kwargs.get("epoch_encoder_variant")
    supports_aa_kwargs = detected_variant_for_aa == "flexible_asymmetric"

    if has_legacy_global_branch:
        logger.warning(
            "Checkpoint uses the pre-reorder global_branch layout "
            "(GAP -> 1x1 conv with bias -> norm -> activation). "
            "Setting cnn_legacy_global_branch=True to match checkpoint..."
        )
        model_kwargs["cnn_legacy_global_branch"] = True
        model_config["model_kwargs"] = model_kwargs

    if has_legacy_branch_aa:
        if supports_aa_kwargs:
            logger.warning(
                "Checkpoint carries anti-alias filters on %d dilated branches "
                "(pre-removal layout). Setting "
                "cnn_anti_alias_dilated_branches=True to match checkpoint...",
                len(legacy_branch_aa_keys),
            )
            model_kwargs["cnn_anti_alias_dilated_branches"] = True
            model_config["model_kwargs"] = model_kwargs
        else:
            logger.warning(
                "Checkpoint carries legacy per-branch anti-alias filters but "
                "encoder variant %r does not accept the reconstruction flag; "
                "the load will likely fail.",
                detected_variant_for_aa,
            )

    if supports_aa_kwargs:
        if detected_stem_taps is not None:
            model_kwargs["cnn_stem_aa_num_taps"] = detected_stem_taps
            model_config["model_kwargs"] = model_kwargs
        if stage_taps:
            # Only stages that actually downsample build a pool, and the encoder
            # consumes its tap tuple in exactly that order. Pad to four so the
            # constructor's per-downsampling-stage validation is satisfied.
            ordered_taps = [stage_taps[stage] for stage in sorted(stage_taps)]
            padded_taps = ordered_taps + [ordered_taps[-1]] * (4 - len(ordered_taps))
            model_kwargs["cnn_stage_aa_num_taps"] = tuple(padded_taps[:4])
            model_config["model_kwargs"] = model_kwargs
            logger.info(
                f"  Applying checkpoint AA taps: stem={detected_stem_taps}, "
                f"stages={model_kwargs['cnn_stage_aa_num_taps']}"
            )
    elif detected_stem_taps is not None or stage_taps:
        logger.info(
            "  Encoder variant %r does not accept anti-alias tap kwargs; the "
            "checkpoint's stored filters will be restored after load instead.",
            detected_variant_for_aa,
        )

    if not (
        has_legacy_global_branch
        or has_legacy_branch_aa
        or model_kwargs.get("cnn_stem_aa_num_taps") is not None
    ):
        logger.info("  ✓ Current flexible-stem schema (no legacy adaptation needed)")

    # Import model classes
    from spectra.models import TransformerContextNet

    # Get model class
    model_type = model_config.get("model_type", "TransformerContextNet")
    model_classes = {"TransformerContextNet": TransformerContextNet}

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")

    ModelClass = model_classes[model_type]

    # Extract model kwargs
    model_kwargs = model_config.get("model_kwargs", {})

    if channel_names is not None:
        inferred_in_ch = model_kwargs.get("in_ch")
        if inferred_in_ch is not None and len(channel_names) != inferred_in_ch:
            logger.warning(
                "Channel name count (%d) does not match inferred in_ch (%d). "
                "Continuing but engineered feature extractor may misbehave.",
                len(channel_names),
                inferred_in_ch,
            )

        # Check if the checkpoint saved channel_names for the feature extractor.
        # If so, verify EEG channel count consistency.  If the caller-provided
        # channel_names would produce a different num_eeg_channels (and thus
        # different engineered feature dimension), keep the checkpoint's
        # channel_names to preserve weight compatibility.
        saved_channel_names = model_kwargs.get("channel_names")
        uses_engineered = model_kwargs.get("use_feature_extraction", False)

        if uses_engineered and saved_channel_names is not None:
            from spectra.data.channel import infer_channel_type

            saved_eeg_count = sum(
                1 for name in saved_channel_names if infer_channel_type(name) == "eeg"
            )
            caller_eeg_count = sum(
                1 for name in channel_names if infer_channel_type(name) == "eeg"
            )

            # Check if saved channel names are all generic/unrecognizable
            # (e.g., ch_0, ch_1, ... from checkpoints that didn't save real names).
            # Generic names can't be classified into modalities (EEG/EOG/EMG),
            # which causes FlexibleModalityAwareStem to fail. In this case,
            # always prefer the caller's channel names.
            saved_recognized_count = sum(
                1
                for name in saved_channel_names
                if infer_channel_type(name) != "unknown"
            )

            if saved_recognized_count == 0:
                logger.info(
                    "Saved channel_names are all generic/unrecognizable (%s). "
                    "Using caller's channel_names (%s) for proper modality routing.",
                    saved_channel_names,
                    channel_names,
                )
                model_kwargs["channel_names"] = list(channel_names)
            elif saved_eeg_count != caller_eeg_count:
                logger.warning(
                    "Caller channel_names have %d EEG channels but checkpoint was "
                    "trained with %d EEG channels. Using checkpoint's channel_names "
                    "for feature extractor to preserve weight dimensions. "
                    "Caller channels: %s, Saved channels: %s",
                    caller_eeg_count,
                    saved_eeg_count,
                    channel_names,
                    saved_channel_names,
                )
                # Keep saved channel_names for model construction (feature extractor)
                # Do NOT override model_kwargs["channel_names"]
            else:
                # Same EEG count, safe to override
                model_kwargs["channel_names"] = list(channel_names)
        else:
            model_kwargs["channel_names"] = list(channel_names)

    # Filter out obsolete parameters from older model versions
    obsolete_params: set[str] = set()
    obsolete_found = {k: v for k, v in model_kwargs.items() if k in obsolete_params}
    if obsolete_found:
        logger.info(f"Filtering out obsolete parameters: {list(obsolete_found.keys())}")
        model_kwargs = {
            k: v for k, v in model_kwargs.items() if k not in obsolete_params
        }

    # Handle parameter renames for backward compatibility
    # Migrate use_engineered_features → use_feature_extraction
    if (
        "use_engineered_features" in model_kwargs
        and "use_feature_extraction" not in model_kwargs
    ):
        model_kwargs["use_feature_extraction"] = model_kwargs.pop(
            "use_engineered_features"
        )
        logger.info(
            "Migrated parameter: use_engineered_features -> use_feature_extraction"
        )

    # Migrate feature_fusion_mode → fusion_mode
    if "feature_fusion_mode" in model_kwargs and "fusion_mode" not in model_kwargs:
        model_kwargs["fusion_mode"] = model_kwargs.pop("feature_fusion_mode")
        logger.info("Migrated parameter: feature_fusion_mode -> fusion_mode")

    # Map deprecated pooling modes to current values
    if "pooling_mode" in model_kwargs:
        pooling_mode = model_kwargs["pooling_mode"]
        pooling_mode_mapping = {
            "weighted_fusion": "attentive_fusion",  # Deprecated name -> current name
        }
        if pooling_mode in pooling_mode_mapping:
            new_pooling_mode = pooling_mode_mapping[pooling_mode]
            model_kwargs["pooling_mode"] = new_pooling_mode
            logger.info(
                f"Mapped deprecated pooling_mode '{pooling_mode}' -> '{new_pooling_mode}'"
            )

    # Normalize hybrid encoder legacy config keys to current constructor kwargs.
    if "stage_strides" in model_kwargs and "cnn_stage_strides" not in model_kwargs:
        model_kwargs["cnn_stage_strides"] = tuple(model_kwargs.pop("stage_strides"))
        logger.info("Migrated parameter: stage_strides -> cnn_stage_strides")
    if (
        "stage1_dilations" in model_kwargs
        and "cnn_stage1_dilations" not in model_kwargs
    ):
        model_kwargs["cnn_stage1_dilations"] = tuple(
            model_kwargs.pop("stage1_dilations")
        )
        logger.info("Migrated parameter: stage1_dilations -> cnn_stage1_dilations")
    if (
        "stage2_dilations" in model_kwargs
        and "cnn_stage2_dilations" not in model_kwargs
    ):
        model_kwargs["cnn_stage2_dilations"] = tuple(
            model_kwargs.pop("stage2_dilations")
        )
        logger.info("Migrated parameter: stage2_dilations -> cnn_stage2_dilations")

    # Override time_len and fs based on options if provided
    # This ensures engineered features use the correct sampling rate for the input data
    if options is not None:
        new_time_len = options.fs * options.epoch_sec
        old_time_len = model_kwargs.get("time_len")
        if old_time_len is not None and old_time_len != new_time_len:
            logger.info(
                "Overriding model time_len from %d to %d based on options (fs=%d Hz, epoch_sec=%d)",
                old_time_len,
                new_time_len,
                options.fs,
                options.epoch_sec,
            )
        model_kwargs["time_len"] = new_time_len
        # Also set explicit fs parameter if model supports it
        model_kwargs["fs"] = options.fs

    # Drop stale config keys that the current constructor no longer accepts,
    # while preserving metadata-only keys used elsewhere in the runtime.
    metadata_only_keys = {"context_half", "eng_feature_dim"}
    constructor_signature = inspect.signature(ModelClass.__init__)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in constructor_signature.parameters.values()
    )
    valid_constructor_keys = {
        name for name in constructor_signature.parameters if name not in {"self"}
    }
    unsupported_constructor_keys = sorted(
        key
        for key in model_kwargs
        if key not in metadata_only_keys
        and key not in valid_constructor_keys
        and not accepts_var_kwargs
    )
    if unsupported_constructor_keys:
        logger.info(
            "Filtering unsupported constructor kwargs for %s: %s",
            model_type,
            unsupported_constructor_keys,
        )
        model_kwargs = {
            key: value
            for key, value in model_kwargs.items()
            if key not in unsupported_constructor_keys
        }

    # Persist the fully resolved kwargs back into the checkpoint audit payload
    # before constructor-only filtering so downstream callers don't re-derive.
    model_config["model_kwargs"] = model_kwargs

    # Remove metadata-only keys that aren't model constructor parameters
    constructor_kwargs = dict(model_kwargs)
    constructor_kwargs.pop("context_half", None)
    constructor_kwargs.pop("eng_feature_dim", None)
    if model_type != "TransformerContextNet":
        constructor_kwargs.pop("sleepfm_mode", None)

    expects_slow_wave_occupancy = _validate_slow_wave_occupancy_checkpoint_contract(
        state_dict,
        constructor_kwargs,
        checkpoint_label=str(checkpoint_path),
    )

    # Create model
    logger.info(f"Creating {model_type} model")
    logger.info(f"Model kwargs: {constructor_kwargs}")
    base_model = ModelClass(**constructor_kwargs)

    from spectra.model.recording_conditioning import get_recording_conditioner

    saved_conditioning = any("recording_conditioner." in key for key in state_dict)
    if saved_conditioning != (get_recording_conditioner(base_model) is not None):
        raise InferenceModelLoadError(
            "Recording conditioning metadata and learned checkpoint state disagree"
        )

    if saved_conditioning:
        saved_observer = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if "recording_conditioner." in key
        }
        expected_observer = {
            key: value
            for key, value in base_model.state_dict().items()
            if "recording_conditioner." in key
        }
        missing = sorted(expected_observer.keys() - saved_observer.keys())
        unexpected = sorted(saved_observer.keys() - expected_observer.keys())
        mismatched = sorted(
            key
            for key in expected_observer.keys() & saved_observer.keys()
            if isinstance(expected_observer[key], torch.Tensor)
            and (
                not isinstance(saved_observer[key], torch.Tensor)
                or expected_observer[key].shape != saved_observer[key].shape
            )
        )
        if missing or unexpected or mismatched:
            raise InferenceModelLoadError(
                f"Incomplete recording conditioner: missing={missing}, "
                f"unexpected={unexpected}, shape_mismatch={mismatched}"
            )

    # === CHECKPOINT-MODEL COMPATIBILITY CHECK ===
    logger.info("=" * 80)
    logger.info("CHECKPOINT-MODEL COMPATIBILITY CHECK")
    logger.info("=" * 80)

    # Compare expected config with actual model
    actual_in_ch = _get_model_expected_channels(base_model)
    config_in_ch = model_kwargs.get("in_ch")

    if actual_in_ch != config_in_ch:
        logger.warning(
            f"⚠️  Model in_ch mismatch: config says {config_in_ch}, model has {actual_in_ch}"
        )
    else:
        logger.info(f"✓ Input channels match: {actual_in_ch}")

    # Compare model/checkpoint sizes using like-for-like metrics:
    # - Parameters only
    # - Full state tensors (parameters + buffers)
    # Previous logic compared model parameters against checkpoint tensors,
    # which can over-report false differences when large buffers are present
    # (e.g., sinusoidal pos.pe).
    model_param_count = sum(p.numel() for p in base_model.parameters())
    model_state_tensor_count = sum(
        v.numel()
        for v in base_model.state_dict().values()
        if isinstance(v, torch.Tensor)
    )
    checkpoint_state_tensor_count = sum(
        v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)
    )

    # Fixed FIR filter buffers are non-persistent in the current schema, so a
    # checkpoint that stored them legitimately carries tensors the model's
    # state_dict never will. Account for them separately rather than reporting a
    # permanent size discrepancy that reads like missing learned state.
    fixed_filter_names = _fixed_filter_buffer_names(base_model)
    fixed_filter_elems = sum(
        v.numel()
        for k, v in state_dict.items()
        if isinstance(v, torch.Tensor)
        and isinstance(k, str)
        and _strip_model_prefix(k) in fixed_filter_names
    )

    logger.info("Size comparison:")
    logger.info(f"  Instantiated model parameters: {model_param_count:,}")
    logger.info(f"  Instantiated model state tensors: {model_state_tensor_count:,}")
    logger.info(f"  Checkpoint state tensors: {checkpoint_state_tensor_count:,}")
    if fixed_filter_elems:
        logger.info(
            f"  ... of which non-persistent fixed filter buffers: {fixed_filter_elems:,}"
        )

    state_tensor_delta = abs(
        model_state_tensor_count - checkpoint_state_tensor_count + fixed_filter_elems
    )
    if state_tensor_delta > 100:
        logger.warning(
            "⚠️  Significant state-tensor count difference: %s elements",
            f"{state_tensor_delta:,}",
        )
    elif fixed_filter_elems:
        logger.info(
            "  ✓ State-tensor difference fully explained by non-persistent "
            "fixed filter buffers"
        )

    # Load state dict (handle multiple checkpoint formats)
    # Priority: 'model' > 'model_state_dict' > 'state_dict' > checkpoint itself
    state_dict = checkpoint.get(
        "model",
        checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)),
    )
    if prefer_averaged:
        ema_state = checkpoint.get("ema")
        averaged = ema_state.get("module") if isinstance(ema_state, dict) else None
        if isinstance(averaged, dict) and averaged:
            logger.info(
                "Using averaged (EMA/SWA) weights from checkpoint['ema']['module'] "
                "(%d tensors)",
                len(averaged),
            )
            state_dict = averaged
        else:
            logger.warning(
                "prefer_averaged=True but this checkpoint stores no averaged "
                "weights; falling back to the raw 'model' weights."
            )

    # === STATE DICT INSPECTION ===
    logger.info("=" * 80)
    logger.info("STATE DICT INSPECTION")
    logger.info("=" * 80)
    logger.info(f"Total parameters in state dict: {len(state_dict)}")

    # Show first 10 keys
    sample_keys = list(state_dict.keys())[:10]
    logger.info(f"Sample keys (first 10): {sample_keys}")

    # Count parameters by prefix
    prefix_counts = {}
    for key in state_dict.keys():
        prefix = key.split(".")[0] if "." in key else key
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    logger.info(f"Parameter counts by prefix: {prefix_counts}")

    # Check for custom N1-specific modules that may have been removed
    n1_modules = [
        "n1_position_bias",
        "multi_scale_conv",
        "theta_gate_proj",
        "sem_gate_proj",
        "n1_amplifier",
    ]
    found_n1_modules = [
        mod for mod in n1_modules if any(mod in k for k in state_dict.keys())
    ]
    if found_n1_modules:
        logger.warning("=" * 80)
        logger.warning("⚠️  CHECKPOINT CONTAINS REMOVED N1-SPECIFIC MODULES")
        logger.warning("=" * 80)
        logger.warning(
            "This checkpoint was trained with custom N1 detection modules that have been "
            "REMOVED from the current model code:"
        )
        for mod in found_n1_modules:
            mod_keys = [k for k in state_dict.keys() if mod in k]
            logger.warning(f"  - {mod}: {len(mod_keys)} parameters")
        logger.warning(
            "\nThese modules will be SKIPPED during loading (strict=False). "
            "This is expected behavior if you've updated to a version without N1 modules."
        )
        logger.warning("=" * 80)

    # Strip _orig_mod. prefix from all keys (from torch.compile) before checking
    cleaned_state_dict = {}
    orig_mod_count = 0
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[10:]  # len('_orig_mod.') = 10
            orig_mod_count += 1
        cleaned_state_dict[clean_key] = value

    if orig_mod_count > 0:
        logger.info(
            f"Stripped _orig_mod. prefix from {orig_mod_count} keys (torch.compile checkpoint)"
        )

    state_dict = cleaned_state_dict
    state_dict, legacy_router_removed = _strip_legacy_router_context_keys(state_dict)
    if legacy_router_removed > 0:
        logger.warning(
            "Ignoring legacy router_context parameters from checkpoint "
            "(%d keys removed)",
            legacy_router_removed,
        )

    # Migrate old feat_projection keys (nn.Sequential) to new BottleneckProjection structure
    from spectra.utils.checkpoint import migrate_feat_projection_keys

    state_dict = migrate_feat_projection_keys(state_dict)
    pe_mode_in_state = _infer_pe_mode_in_state_dict(state_dict)
    checkpoint_learnable_pe_tensor = _extract_learnable_pe_tensor(state_dict)

    preproc_stats_snapshot = checkpoint.get("preprocessor_stats")
    if preproc_stats_snapshot:
        norm_cfg = preproc_stats_snapshot.get("normalization_cfg", {})
        logger.info(
            "Checkpoint preprocessor stats snapshot: calibrated=%s, mode=%s, channels=%s",
            preproc_stats_snapshot.get("calibrated"),
            norm_cfg.get("mode"),
            preproc_stats_snapshot.get("n_channels"),
        )
        q1_snapshot = preproc_stats_snapshot.get("q1")
        q3_snapshot = preproc_stats_snapshot.get("q3")
        if isinstance(q1_snapshot, torch.Tensor) and isinstance(
            q3_snapshot, torch.Tensor
        ):
            logger.info(
                "  q1 range=[%.3f, %.3f], q3 range=[%.3f, %.3f]",
                float(q1_snapshot.min()),
                float(q1_snapshot.max()),
                float(q3_snapshot.min()),
                float(q3_snapshot.max()),
            )

    # Check if state dict has preprocessor (indicates ModelWithPreproc wrapper)
    # Look for any preprocessor keys - even just calibration buffers indicate the model was trained with preprocessing
    preprocessor_keys = [k for k in state_dict.keys() if k.startswith("preprocessor.")]
    has_preprocessor = len(preprocessor_keys) > 0
    loaded_learnable_pe_tensor: torch.Tensor | None = None

    if has_preprocessor:
        logger.info(
            f"Found {len(preprocessor_keys)} preprocessor parameters in checkpoint:"
        )
        for key in preprocessor_keys[:10]:  # Show first 10
            logger.info(f"  {key}")
        logger.info(
            "Checkpoint contains embedded preprocessor - creating ModelWithPreproc wrapper"
        )
        from spectra.model.wrapped import ModelWithPreproc
        from spectra.preprocessing.config import ProcCfg
        from spectra.preprocessing.normalizer import EmbeddedPreproc

        proc_cfg_for_model: ProcCfg | None = proc_cfg or _extract_proc_cfg(checkpoint)
        if proc_cfg_for_model is None:
            logger.info(
                "No preprocessing config supplied or stored in checkpoint; using defaults"
            )
            proc_cfg_for_model = ProcCfg(version="1.0.0", dataset_name="inference")
        elif proc_cfg is None:
            logger.info(
                "Using preprocessing config from checkpoint for EmbeddedPreproc"
            )
        else:
            logger.info(
                "Using preprocessing config supplied by caller for EmbeddedPreproc"
            )

        # Get number of channels from model
        in_ch = model_kwargs.get("in_ch", 8)

        # Infer channel types (need these for EmbeddedPreproc)
        # Use generic 'eeg' for all channels as a reasonable default
        channel_types = ["eeg"] * in_ch

        # Create EmbeddedPreproc with config from checkpoint
        preprocessor = EmbeddedPreproc(
            cfg=proc_cfg_for_model,
            channel_types=channel_types,
        )

        # Create wrapped model
        model = ModelWithPreproc(
            model=base_model,
            preprocessor=preprocessor,
        )

        # Align state dict keys to model format (handles compiled model case)
        # Then resize feat_projection weights if engineered feature dimension changed
        from spectra.utils.checkpoint import (
            align_state_dict_to_model,
            resize_feat_projection_weights,
            restore_fixed_filter_buffers,
        )

        state_dict = align_state_dict_to_model(state_dict, model)
        state_dict = resize_feat_projection_weights(state_dict, model)
        loaded_learnable_pe_tensor = _extract_learnable_pe_tensor(state_dict)

        # Load full state dict (includes both model and preprocessor)
        load_result = model.load_state_dict(state_dict, strict=False)
        restored_filters = restore_fixed_filter_buffers(model, state_dict)
        if restored_filters:
            logger.info(
                "Restored %d fixed anti-alias filter buffers from checkpoint",
                len(restored_filters),
            )
        _validate_state_dict_load(
            load_result,
            checkpoint_label=str(checkpoint_path),
            model_label=f"{model_type} (with EmbeddedPreproc)",
            consumed_unexpected_keys={f"{name}.kernel" for name in restored_filters},
        )
        logger.info("Loaded model with embedded preprocessor")
    else:
        logger.info(
            "Checkpoint does not contain preprocessor - loading base model only"
        )
        # Handle wrapped model (remove "model." prefix if present)
        # Also filter out orphaned preprocessor buffers (calibration stats without actual preprocessor layers)
        new_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            # Strip model. prefix (from wrapper)
            if clean_key.startswith("model."):
                clean_key = clean_key[6:]  # len('model.') = 6
            # Skip orphaned preprocessor buffers (these are from compiled models that had
            # preprocessor statistics but we're loading into a base model without preprocessor)
            if clean_key.startswith("preprocessor."):
                logger.debug(f"Skipping orphaned preprocessor buffer: {clean_key}")
                continue
            new_state_dict[clean_key] = value

        # Align state dict keys to model format (handles compiled model case)
        # Then resize feat_projection weights if engineered feature dimension changed
        from spectra.utils.checkpoint import (
            align_state_dict_to_model,
            resize_feat_projection_weights,
            restore_fixed_filter_buffers,
        )

        new_state_dict = align_state_dict_to_model(new_state_dict, base_model)
        new_state_dict = resize_feat_projection_weights(new_state_dict, base_model)
        loaded_learnable_pe_tensor = _extract_learnable_pe_tensor(new_state_dict)

        load_result = base_model.load_state_dict(new_state_dict, strict=False)

        # Fixed FIR buffers are non-persistent, so the checkpoint's copies arrive
        # as unexpected keys. Consume them before reporting: they are the filters
        # training actually used, and rebuilding them from current defaults would
        # silently change the low-pass response.
        restored_filters = restore_fixed_filter_buffers(base_model, new_state_dict)
        if restored_filters:
            logger.info(
                "Restored %d fixed anti-alias filter buffers from checkpoint",
                len(restored_filters),
            )
        restored_filter_keys = {f"{name}.kernel" for name in restored_filters}

        # Log any unexpected keys (parameters in checkpoint but not in model)
        unexpected_keys = [
            key
            for key in getattr(load_result, "unexpected_keys", [])
            if _strip_model_prefix(key) not in restored_filter_keys
        ]
        if unexpected_keys:
            logger.warning("=" * 80)
            logger.warning("⚠️  UNEXPECTED KEYS IN CHECKPOINT (NOT LOADED)")
            logger.warning("=" * 80)
            logger.warning(
                f"Found {len(unexpected_keys)} parameters in checkpoint that don't exist in current model:"
            )
            for key in unexpected_keys[:20]:  # Show first 20
                logger.warning(f"  - {key}")
            if len(unexpected_keys) > 20:
                logger.warning(f"  ... and {len(unexpected_keys) - 20} more")
            logger.warning(
                "\n⚠️  This means the checkpoint was trained with a different model architecture!"
            )
            logger.warning("=" * 80)

        _validate_state_dict_load(
            load_result,
            checkpoint_label=str(checkpoint_path),
            model_label=model_type,
            consumed_unexpected_keys=restored_filter_keys,
        )
        model = base_model

    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    mismatched_keys = list(getattr(load_result, "mismatched_keys", []))
    logger.info(
        "State-dict load audit: missing_keys=%d, unexpected_keys=%d, mismatched_keys=%d",
        len(missing_keys),
        len(unexpected_keys),
        len(mismatched_keys),
    )
    pe_verified = False
    if pe_mode_in_state == "learnable":
        pe_tensor_to_verify = loaded_learnable_pe_tensor
        if pe_tensor_to_verify is None:
            pe_tensor_to_verify = checkpoint_learnable_pe_tensor
        pe_verified = _verify_loaded_learnable_pe(
            model,
            pe_tensor_to_verify,
            checkpoint_label=str(checkpoint_path),
        )
    slow_wave_occupancy_verified = _verify_loaded_slow_wave_occupancy_head(
        model,
        state_dict,
        checkpoint_label=str(checkpoint_path),
    )
    if expects_slow_wave_occupancy and not slow_wave_occupancy_verified:
        raise InferenceModelLoadError(
            f"Checkpoint '{checkpoint_path}' declared a slow-wave occupancy head "
            "but runtime did not verify its learned tensors."
        )
    # Persist resolved config and load-audit into checkpoint output.
    checkpoint["model_config"] = model_config
    checkpoint["_inference_load_audit"] = {
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
        "mismatched_keys_count": len(mismatched_keys),
        "pe_mode_in_state": pe_mode_in_state,
        "pe_verified": pe_verified,
        "slow_wave_occupancy_verified": slow_wave_occupancy_verified,
    }

    model.to(device)
    model.eval()

    # === MODEL ARCHITECTURE VERIFICATION ===
    logger.info("=" * 80)
    logger.info("MODEL ARCHITECTURE VERIFICATION")
    logger.info("=" * 80)

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Print model structure (first level)
    logger.info("\nModel structure:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        logger.info(f"  {name}: {module.__class__.__name__} ({num_params:,} params)")

    # Check for specific architectural components
    has_preprocessor = hasattr(model, "preprocessor") or (
        hasattr(model, "model") and hasattr(model.model, "preprocessor")
    )
    expected_in_ch = _get_model_expected_channels(model)
    has_epoch_encoder = any(
        "epoch_encoder" in name for name, _ in model.named_modules()
    )
    has_context_encoder = any(
        "enc_layers" in name or "tcn" in name.lower()
        for name, _ in model.named_modules()
    )

    logger.info("\nArchitectural components detected:")
    logger.info(f"  Preprocessor: {has_preprocessor}")
    logger.info(f"  Expected input channels: {expected_in_ch}")
    logger.info(f"  Epoch encoder (CNN): {has_epoch_encoder}")
    logger.info(f"  Context encoder: {has_context_encoder}")

    # Get expected input/output shapes
    expected_out_ch = None
    for name, module in model.named_modules():
        if "classifier" in name and hasattr(module, "out_features"):
            # Skip cross_attn internal projections (e.g.,
            # classifier.cross_attn.out_proj has out_features=d_model,
            # not num_classes).
            if "cross_attn" in name:
                continue
            expected_out_ch = module.out_features
            break

    logger.info("\nExpected I/O:")
    logger.info(f"  Input channels: {expected_in_ch}")
    logger.info(f"  Output classes: {expected_out_ch}")

    # Validate engineered feature statistics if present
    _validate_engineered_feature_statistics(model)

    # === WEIGHT STATISTICS VALIDATION ===
    logger.info("=" * 80)
    logger.info("WEIGHT STATISTICS VALIDATION")
    logger.info("=" * 80)

    # Check a few key layers for reasonable weight ranges
    weight_stats = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            # Compute std only if there are enough elements (avoid degrees of freedom warning)
            numel = param.data.numel()
            std_val = param.data.std().item() if numel > 1 else 0.0

            weight_stats[name] = {
                "mean": param.data.mean().item(),
                "std": std_val,
                "min": param.data.min().item(),
                "max": param.data.max().item(),
                "has_nan": torch.isnan(param.data).any().item(),
                "has_inf": torch.isinf(param.data).any().item(),
            }

    # Check for problematic weights
    problematic_layers = []
    for name, stats in weight_stats.items():
        if stats["has_nan"]:
            problematic_layers.append(f"{name}: contains NaN values")
        elif stats["has_inf"]:
            problematic_layers.append(f"{name}: contains Inf values")
        elif abs(stats["mean"]) < 1e-8 and stats["std"] < 1e-8:
            problematic_layers.append(
                f"{name}: all zeros (mean={stats['mean']:.2e}, std={stats['std']:.2e})"
            )
        elif stats["std"] > 100:
            problematic_layers.append(f"{name}: unusually large std={stats['std']:.2f}")

    if problematic_layers:
        logger.warning(
            f"⚠️  Found {len(problematic_layers)} layers with potentially problematic weights:"
        )
        for issue in problematic_layers[:5]:  # Show first 5
            logger.warning(f"  - {issue}")
        if len(problematic_layers) > 5:
            logger.warning(f"  ... and {len(problematic_layers) - 5} more")
    else:
        logger.info("✓ All weight statistics look reasonable")

    # Show sample statistics from key layers
    sample_layers = [
        name
        for name in weight_stats.keys()
        if any(x in name for x in ["classifier", "proj", "conv", "linear"])
    ][:3]
    if sample_layers:
        logger.info("\nSample weight statistics from key layers:")
        for name in sample_layers:
            stats = weight_stats[name]
            logger.info(f"  {name}:")
            logger.info(
                f"    mean={stats['mean']:.4f}, std={stats['std']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]"
            )

    # === TEST FORWARD PASS ===
    logger.info("=" * 80)
    logger.info("TEST FORWARD PASS")
    logger.info("=" * 80)

    try:
        # Create dummy input matching expected shape
        expected_in_ch = _get_model_expected_channels(model) or 8
        model_kwargs = model_config.get("model_kwargs", {})
        time_len = model_kwargs.get("time_len", 3840)
        if model_kwargs.get("epoch_encoder_variant") in {
            "learned_feature_axial",
            "learned_feature_axial_v2",
        }:
            context_len = int(model_kwargs.get("context_epochs", 21))
        else:
            context_half = _get_saved_context_half(model_config)
            context_len = 2 * (context_half if context_half is not None else 16) + 1

        dummy_input = torch.randn(
            1, context_len, expected_in_ch, time_len, device=device
        )
        logger.info(f"Dummy input shape: {tuple(dummy_input.shape)} [B, L, C, T]")

        # Build dict input to exercise the ModelWithPreproc passthrough path
        # during the test; plain tensor input otherwise.
        from spectra.model.wrapped import ModelWithPreproc as _MWP

        if isinstance(model, _MWP):
            dummy_model_input = {"wave": dummy_input}
            logger.info(
                "Using dict input for test forward pass (ModelWithPreproc=True)"
            )
        else:
            dummy_model_input = dummy_input

        from spectra.model.recording_conditioning import (
            pinned_recording_observations,
        )

        with (
            torch.inference_mode(),
            pinned_recording_observations(
                model,
                dummy_input[0],
                torch.ones(expected_in_ch, dtype=torch.bool, device=device),
            ),
        ):
            dummy_output = model(dummy_model_input)

        # Handle both tensor output and ForwardOutput (NamedTuple)
        if hasattr(dummy_output, "logits"):
            output_tensor = dummy_output.logits
        elif hasattr(dummy_output, "shape"):
            output_tensor = dummy_output
        else:
            # Try to get first element if it's a tuple/list
            output_tensor = (
                dummy_output[0]
                if isinstance(dummy_output, (tuple, list))
                else dummy_output
            )

        logger.info(f"Dummy output shape: {tuple(output_tensor.shape)}")

        # Check output statistics
        output_mean = output_tensor.mean().item()
        output_std = output_tensor.std().item()
        output_min = output_tensor.min().item()
        output_max = output_tensor.max().item()
        has_nan = torch.isnan(output_tensor).any().item()
        has_inf = torch.isinf(output_tensor).any().item()

        logger.info("Output statistics:")
        logger.info(f"  mean={output_mean:.4f}, std={output_std:.4f}")
        logger.info(f"  range=[{output_min:.4f}, {output_max:.4f}]")
        logger.info(f"  has_nan={has_nan}, has_inf={has_inf}")

        # Check output distribution (logits should have reasonable range)
        if has_nan or has_inf:
            logger.error("❌ Model produces NaN or Inf outputs - model may be broken!")
        elif abs(output_mean) > 100 or output_std > 100:
            logger.warning(
                "⚠️  Unusual output statistics detected - predictions may be unreliable"
            )
        else:
            # Apply softmax and check class probabilities
            probs = F.softmax(output_tensor, dim=-1)
            prob_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            max_prob = probs.max(dim=-1)[0].mean().item()

            logger.info("Softmax probabilities:")
            logger.info(f"  average entropy={prob_entropy:.4f} (higher=more uncertain)")
            logger.info(f"  average max_prob={max_prob:.4f} (lower=more uncertain)")

            if max_prob < 0.3:
                logger.warning(
                    "⚠️  Model is very uncertain (low confidence) - may collapse to uniform predictions"
                )
            elif max_prob > 0.95:
                logger.warning(
                    "⚠️  Model is overconfident - may collapse to single class"
                )
            else:
                logger.info("✓ Output distribution looks reasonable")

    except Exception as e:
        logger.error(f"❌ Test forward pass failed: {e}")
        logger.error("Model may not be properly loaded or configured")
        import traceback

        logger.error(traceback.format_exc())

    logger.info("=" * 80)
    logger.info("Model loaded and verified successfully")
    logger.info("=" * 80)
    return model, checkpoint


def calibrate_preprocessor(
    model: nn.Module,
    calibration_data: torch.Tensor | np.ndarray,
    presence_mask: torch.Tensor | np.ndarray,
    calibration_mode: str,
) -> None:
    """Calibrate embedded preprocessor.

    Args:
        model: Model with embedded preprocessor
        calibration_data: Calibration data tensor
        presence_mask: Channel presence mask
        calibration_mode: Calibration strategy
    """
    # Check if model has preprocessor
    direct_preprocessor = getattr(model, "preprocessor", None)
    inner_model = getattr(model, "model", None)
    if direct_preprocessor is not None:
        preprocessor: Any = direct_preprocessor
    elif isinstance(inner_model, nn.Module) and hasattr(inner_model, "preprocessor"):
        preprocessor = cast(Any, inner_model).preprocessor
    else:
        logger.warning(
            "Model does not have embedded preprocessor, skipping calibration"
        )
        return

    # Convert calibration data to tensor and collapse context dimension if present
    if isinstance(calibration_data, np.ndarray):
        calibration_tensor = torch.from_numpy(calibration_data).float()
    else:
        calibration_tensor = calibration_data.float()

    if calibration_tensor.ndim == 2:
        calibration_tensor = calibration_tensor.unsqueeze(0)
    elif calibration_tensor.ndim == 4:
        # Flatten context windows so quartiles see the entire recording
        B, L, C, T = calibration_tensor.shape
        calibration_tensor = calibration_tensor.reshape(B * L, C, T)
    elif calibration_tensor.ndim != 3:
        raise ValueError(
            f"Unsupported calibration tensor shape {calibration_tensor.shape}; expected 2D, 3D, or 4D"
        )

    if isinstance(presence_mask, np.ndarray):
        presence_mask_tensor = torch.from_numpy(presence_mask).float()
    else:
        presence_mask_tensor = presence_mask.float()

    if calibration_mode == "checkpoint":
        # Use statistics from checkpoint (already loaded)
        if preprocessor.calibrated.item():
            logger.info("Using preprocessor calibration from checkpoint")
        else:
            logger.warning(
                "Checkpoint preprocessor not calibrated, calibrating on sample data"
            )
            preprocessor.calibrate(
                calibration_tensor, channel_mask=presence_mask_tensor
            )
    elif calibration_mode == "per_recording":
        logger.info("Calibrating preprocessor on full recording")
        # Use all available data for calibration
        preprocessor.calibrate(calibration_tensor, channel_mask=presence_mask_tensor)
    elif calibration_mode == "warmup":
        logger.info("Calibrating preprocessor on warmup epochs")
        # Use subset for calibration
        preprocessor.calibrate(calibration_tensor, channel_mask=presence_mask_tensor)


def _build_enhanced_tta_configs(
    options: ScoreOptions, *, n_views: int
) -> list[dict[str, float]]:
    """Build deterministic TTA configs (excluding the base view)."""
    n = int(max(0, n_views))
    if n == 0:
        return []

    noise = float(getattr(options, "tta_noise", 0.0) or 0.0)
    amp = float(getattr(options, "tta_amp", 0.0) or 0.0)
    shift = float(getattr(options, "tta_shift", 0.0) or 0.0)

    candidates: list[dict[str, float]] = []
    if noise > 0:
        candidates.append({"noise_std": noise, "amp_mult": 1.0, "shift_sec": 0.0})
    if amp > 0:
        candidates.append({"noise_std": 0.0, "amp_mult": 1.0 - amp, "shift_sec": 0.0})
        candidates.append({"noise_std": 0.0, "amp_mult": 1.0 + amp, "shift_sec": 0.0})
        candidates.append(
            {
                "noise_std": noise * 0.5,
                "amp_mult": 1.0 - 0.4 * amp,
                "shift_sec": 0.0,
            }
        )
    if shift > 0:
        candidates.append({"noise_std": 0.0, "amp_mult": 1.0, "shift_sec": shift})
        candidates.append({"noise_std": 0.0, "amp_mult": 1.0, "shift_sec": -shift})

    if not candidates:
        candidates.append({"noise_std": 0.0, "amp_mult": 1.0, "shift_sec": 0.0})

    return [candidates[i % len(candidates)] for i in range(n)]


def _apply_enhanced_tta_inplace(
    xb: torch.Tensor,
    cfg: dict[str, float],
    *,
    epoch_sec: int,
) -> None:
    """Apply deterministic augmentation in-place to a [B, L, C, T] tensor."""
    if xb.ndim != 4:
        raise ValueError(f"Expected xb with shape [B, L, C, T], got {tuple(xb.shape)}")

    noise_std = float(cfg.get("noise_std", 0.0) or 0.0)
    amp_mult = float(cfg.get("amp_mult", 1.0) or 1.0)
    shift_sec = float(cfg.get("shift_sec", 0.0) or 0.0)

    if amp_mult != 1.0:
        xb.mul_(amp_mult)
    if noise_std > 0.0:
        xb.add_(torch.randn_like(xb) * noise_std)
    if shift_sec != 0.0 and epoch_sec > 0:
        # samples/sec ~= samples_per_epoch / epoch_sec
        samples_per_sec = xb.shape[-1] / float(epoch_sec)
        shift_samples = int(round(shift_sec * samples_per_sec))
        if shift_samples != 0:
            xb.copy_(torch.roll(xb, shifts=shift_samples, dims=-1))


def _run_tta_inference(
    model: nn.Module,
    inputs: torch.Tensor | dict[str, Any],
    channel_mask: torch.Tensor,
    options: ScoreOptions,
    *,
    probability_pool: bool = False,
    on_sample: Callable[[torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Pool TTA forwards and optionally observe each unpooled sample.

    Args:
        model: Model with its inference/MC modes already configured.
        inputs: Prepared batch inputs, reused without mutation.
        channel_mask: Channel availability passed to every forward.
        options: TTA configuration and optional all-position prediction settings.
        probability_pool: Average individual posteriors instead of legacy logits.
        on_sample: Observer receiving each forward's logits before aggregation.

    Returns:
        Mean probabilities when requested, otherwise legacy TTA logits/log-probs.
    """
    # Unpack input if it's a dict. ``norm_stats`` (if present) stays inside the
    # dict and is forwarded to the model as-is.
    if isinstance(inputs, dict):
        x = inputs["wave"]
        # Ensure presence mask travels with dict inputs (for models without kwarg support)
        if "presence_mask" not in inputs:
            inputs = dict(inputs)
            inputs["presence_mask"] = channel_mask
    else:
        x = inputs

    B, L, C, T = x.shape

    tta_mode = str(getattr(options, "tta_mode", "legacy") or "legacy").strip().lower()

    # In overlap-averaging mode each TTA view must also return all-position logits
    # so the per-window aggregation happens per epoch position, not center-only.
    extra_fwd: dict[str, Any] = {}
    if bool(getattr(options, "overlap_average", False)):
        extra_fwd["predict_all"] = True
    if options.recurrent_refinement_steps is not None:
        extra_fwd["recurrent_refinement_steps"] = options.recurrent_refinement_steps

    def _forward_with_optional_mask(inp: torch.Tensor | dict[str, Any]) -> Any:
        try:
            return model(inp, presence_mask=channel_mask, **extra_fwd)
        except TypeError as e:
            if "presence_mask" in str(e):
                return model(inp, **extra_fwd)
            raise

    # Base (unaugmented) view
    if isinstance(inputs, dict):
        base_outputs = _forward_with_optional_mask(inputs)
    else:
        base_outputs = _forward_with_optional_mask(x)

    # Handle both dict and ForwardOutput dataclass
    if hasattr(base_outputs, "logits"):
        base_logits = (
            base_outputs.logits
            if hasattr(base_outputs, "__dataclass_fields__")
            else base_outputs["logits"]
        )
    elif isinstance(base_outputs, dict) and "logits" in base_outputs:
        base_logits = base_outputs["logits"]
    else:
        base_logits = base_outputs
    base_logits = cast(torch.Tensor, base_logits)
    if on_sample is not None:
        on_sample(base_logits)

    if tta_mode == "enhanced":
        accum = (
            base_logits.float().softmax(dim=-1)
            if probability_pool
            else F.log_softmax(base_logits.float(), dim=-1)
        )
        total_views = 1
        for cfg in _build_enhanced_tta_configs(
            options, n_views=int(options.tta_passes)
        ):
            aug_x = x.clone()
            _apply_enhanced_tta_inplace(aug_x, cfg, epoch_sec=int(options.epoch_sec))

            if isinstance(inputs, dict):
                aug_inputs = inputs.copy()
                aug_inputs["wave"] = aug_x

                if "wave_raw" in inputs:
                    aug_x_raw = inputs["wave_raw"].clone()
                    _apply_enhanced_tta_inplace(
                        aug_x_raw, cfg, epoch_sec=int(options.epoch_sec)
                    )
                    aug_inputs["wave_raw"] = aug_x_raw

                if "presence_mask" not in aug_inputs:
                    aug_inputs["presence_mask"] = channel_mask
                view_outputs = _forward_with_optional_mask(aug_inputs)
            else:
                view_outputs = _forward_with_optional_mask(aug_x)

            # Handle both dict and ForwardOutput dataclass
            if hasattr(view_outputs, "logits"):
                view_logits = (
                    view_outputs.logits
                    if hasattr(view_outputs, "__dataclass_fields__")
                    else view_outputs["logits"]
                )
            elif isinstance(view_outputs, dict) and "logits" in view_outputs:
                view_logits = view_outputs["logits"]
            else:
                view_logits = view_outputs
            view_logits = cast(torch.Tensor, view_logits)
            if on_sample is not None:
                on_sample(view_logits)
            accum = accum + (
                view_logits.float().softmax(dim=-1)
                if probability_pool
                else F.log_softmax(view_logits.float(), dim=-1)
            )
            total_views += 1

        return accum / total_views

    # Legacy stochastic TTA: arithmetic mean of logits
    accum_logits = (
        base_logits.float().softmax(dim=-1)
        if probability_pool
        else base_logits.float().clone()
    )
    total_tta_views = 1

    # Import augment here to keep runtime dependencies minimal unless TTA is used.
    from spectra.inference.tta import augment_batch as _augment_batch

    for _v in range(int(max(0, options.tta_passes))):
        aug_x = x.clone()
        _augment_batch(
            aug_x,
            noise_std=float(options.tta_noise),
            amp_scale=float(options.tta_amp),
            shift_sec=float(options.tta_shift),
        )

        if isinstance(inputs, dict):
            aug_inputs = inputs.copy()
            aug_inputs["wave"] = aug_x

            if "wave_raw" in inputs:
                aug_x_raw = inputs["wave_raw"].clone()
                _augment_batch(
                    aug_x_raw,
                    noise_std=float(options.tta_noise),
                    amp_scale=float(options.tta_amp),
                    shift_sec=float(options.tta_shift),
                )
                aug_inputs["wave_raw"] = aug_x_raw

            if "presence_mask" not in aug_inputs:
                aug_inputs["presence_mask"] = channel_mask
            view_outputs = _forward_with_optional_mask(aug_inputs)
        else:
            view_outputs = _forward_with_optional_mask(aug_x)

        # Handle both dict and ForwardOutput dataclass
        if hasattr(view_outputs, "logits"):
            view_logits = (
                view_outputs.logits
                if hasattr(view_outputs, "__dataclass_fields__")
                else view_outputs["logits"]
            )
        elif isinstance(view_outputs, dict) and "logits" in view_outputs:
            view_logits = view_outputs["logits"]
        else:
            view_logits = view_outputs
        view_logits = cast(torch.Tensor, view_logits)
        if on_sample is not None:
            on_sample(view_logits)
        accum_logits = accum_logits + (
            view_logits.float().softmax(dim=-1)
            if probability_pool
            else view_logits.float()
        )
        total_tta_views += 1

    return accum_logits.div_(max(1, total_tta_views))


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


def apply_inference_backend_settings(
    *,
    allow_tf32: bool,
    cudnn_benchmark: bool,
    device: torch.device,
) -> None:
    """Configure global Torch backend flags for faster inference.

    These are process-global toggles; calling repeatedly is idempotent. The
    CUDA-specific flags are no-ops on non-CUDA devices.

    Args:
        allow_tf32: Enable TF32 tensor-core matmul and cuDNN TF32. Recommended
            on Ampere+ GPUs; accuracy impact is negligible.
        cudnn_benchmark: Enable cuDNN autotuning of convolution algorithms.
            Helps conv-heavy models with stable input shapes.
        device: Target inference device.
    """
    if allow_tf32:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception as exc:  # pragma: no cover - backend-specific
            logger.warning("Could not set float32 matmul precision: %s", exc)
        if device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - backend-specific
                logger.warning("Could not enable matmul TF32: %s", exc)
            try:
                torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - backend-specific
                logger.warning("Could not enable cuDNN TF32: %s", exc)
        logger.info("TF32 enabled (matmul precision=high, device=%s)", device.type)

    if cudnn_benchmark and device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
            logger.info("cuDNN benchmark autotuning enabled")
        except Exception as exc:  # pragma: no cover - backend-specific
            logger.warning("Could not enable cuDNN benchmark: %s", exc)


def maybe_compile_inference_model(
    model: nn.Module,
    *,
    compile_model: bool,
    compile_mode: str,
    device: torch.device,
) -> None:
    """Compile ``model.forward`` in place via :meth:`torch.nn.Module.compile`.

    In-place compilation is used deliberately so the model keeps its concrete
    type and attributes. The inference loops rely on
    ``isinstance(model, ModelWithPreproc)`` and set ``model.bypass_preprocessing``;
    wrapping with ``torch.compile(model)`` (which returns an ``OptimizedModule``)
    would break both. ``reduce-overhead``/``max-autotune`` require CUDA and are
    downgraded to ``default`` on other devices.

    Args:
        model: Model to compile in place. No-op if ``compile_model`` is False.
        compile_model: Whether to compile at all.
        compile_mode: One of ``"default"``, ``"reduce-overhead"``, ``"max-autotune"``.
        device: Target inference device.
    """
    if not compile_model:
        return
    if not callable(getattr(torch, "compile", None)):
        logger.warning("torch.compile is unavailable in this build; skipping compile.")
        return

    effective_mode = compile_mode
    if device.type != "cuda" and compile_mode in ("reduce-overhead", "max-autotune"):
        logger.warning(
            "compile_mode=%s requires CUDA; falling back to 'default' on %s",
            compile_mode,
            device.type,
        )
        effective_mode = "default"

    # Avoid graph breaks from scalar extraction (e.g. preprocessor .item() guards).
    try:
        torch._dynamo.config.capture_scalar_outputs = True  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - dynamo-version-specific
        logger.warning("Could not enable capture_scalar_outputs: %s", exc)

    from spectra.model.recording_conditioning import get_recording_conditioner

    if get_recording_conditioner(model) is not None:
        from spectra.model.compile import configure_inductor_for_mode

        effective_mode = {
            "reduce-overhead": "default",
            "max-autotune": "max-autotune-no-cudagraphs",
        }.get(effective_mode, effective_mode)
        policy = configure_inductor_for_mode(
            effective_mode, fullgraph=False, logger=logger
        )
        model.compile(**policy.compile_kwargs)
        return

    logger.info("Compiling model in place with torch.compile (mode=%s)", effective_mode)
    model.compile(mode=effective_mode)


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


def _mc_relative_name(model: nn.Module, name: str) -> str:
    """Strip only known preprocessing/compile wrapper prefixes."""
    from spectra.model.wrapped import ModelWithPreproc

    current = model
    while True:
        if isinstance(current, ModelWithPreproc):
            prefix = "model."
            child = current.model
        elif isinstance(current, torch._dynamo.OptimizedModule):
            prefix = "_orig_mod."
            child = current._orig_mod
        else:
            break
        if not name.startswith(prefix):
            break
        name = name[len(prefix) :]
        current = child
    return name


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
        module_pattern: Regex on qualified names or names relative to known wrappers.

    Returns:
        State to restore after sampling. Setup failures restore it automatically.
    """
    from spectra.models.attention import (
        CrossAttention,
        SpecializedTemporalAttention,
        TemporalAttention,
    )
    from spectra.models.axial_feature_time_transformer import (
        SDPACrossAttention,
        SDPAMultiheadAttention,
    )

    _validate_mc_options(1, p_override)
    state = MCDropoutState()
    matcher = re.compile(module_pattern) if module_pattern else None
    attention_types = (
        nn.MultiheadAttention,
        SDPAMultiheadAttention,
        SDPACrossAttention,
        TemporalAttention,
        SpecializedTemporalAttention,
        CrossAttention,
    )
    modules = dict(m.named_modules())
    selected_names: list[str] = []
    try:
        for name, mod in modules.items():
            is_dropout = isinstance(mod, _DROPOUT_TYPES)
            is_attention = include_attention and isinstance(mod, attention_types)
            if not (is_dropout or is_attention):
                continue
            relative_name = _mc_relative_name(m, name)
            if matcher is not None and not (
                matcher.search(name) or matcher.search(relative_name)
            ):
                continue
            rate_attr = (
                "p"
                if is_dropout
                else (
                    "dropout_p"
                    if isinstance(
                        mod,
                        (
                            TemporalAttention,
                            SpecializedTemporalAttention,
                            CrossAttention,
                        ),
                    )
                    else "dropout"
                )
            )
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
            state.sites.append(
                (name, _mc_component(relative_name), saved_rate, effective_rate)
            )
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


def _pool_mc_samples(
    accum: torch.Tensor, n_samples: int, *, prob_space: bool
) -> torch.Tensor:
    """Reduce an MC accumulator to a single logit-like tensor.

    Args:
        accum: Running sum over MC samples. Holds raw logits when
            ``prob_space`` is False and softmax posteriors when it is True.
        n_samples: Number of samples accumulated.
        prob_space: Whether ``accum`` holds posteriors.

    Returns:
        ``mean(logits)`` for logit pooling, or ``log(mean(prob))`` for
        probability pooling so a downstream ``softmax`` recovers the mean
        posterior exactly.
    """
    mean = accum / max(1, n_samples)
    if not prob_space:
        return mean
    return mean.clamp_min(torch.finfo(mean.dtype).tiny).log()


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
            "Diagnostics include TTA/context variation when those are enabled.",
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
        or (
            options.mc_aggregation == "consistent"
            and (options.tta_passes > 0 or options.overlap_average)
        )
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
    model: nn.Module,
    model_input: dict[str, Any],
    mask: torch.Tensor,
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
            if options.tta_passes > 0:
                logits = _run_tta_inference(
                    model,
                    model_input,
                    mask,
                    options,
                    probability_pool=consistent_prob,
                    on_sample=moments.add if moments is not None else None,
                )
            else:
                logits = _extract_inference_logits(forward(model_input)).float()
                if moments is not None:
                    moments.add(logits)
        value = logits.float()
        if prob_pool and not (consistent_prob and options.tta_passes > 0):
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
    """Keep whole-recording band statistics through scoring and refinement."""
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
@with_recording_observations
def run_inference(
    model: nn.Module,
    epoch_windows: torch.Tensor,
    presence_mask: torch.Tensor,
    device: torch.device,
    options: ScoreOptions,
    norm_stats: dict[str, np.ndarray] | None = None,
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
        norm_stats: Optional normalization statistics for pre-normalized data.
            Can be either:
            - Per-recording (legacy): {'median': (C,), 'iqr': (C,)}
            - Per-epoch (matches batch_edf_to_zarr_fp32.py): {'median': (C, n_epochs), 'iqr': (C, n_epochs)}
            Per-epoch stats are used to reconstruct wave_raw for each epoch independently.

        center_valid_mask: Optional boolean array ``(n_epochs,)`` marking which
            center epochs are labeled/valid. Only consulted when
            ``options.overlap_average`` is True: windows whose center is invalid
            are excluded from the overlap accumulation (train/test parity with the
            ``drop_unlabeled_center`` eval default). None means "all centers valid".
        epoch_channel_mask: Optional availability windows shaped
            ``[n_epochs, context_len, channels]``.

        diagnostics_out: Optional output dictionary populated with per-epoch sampling
            moments. Includes TTA/context variation when enabled, before refinement.

    Returns:
        Logits array ``(n_epochs, n_classes)``. When MC dropout is on the
        samples are pooled per ``options.mc_pooling``: ``"prob"`` returns
        ``log(mean(softmax(logits)))`` so a downstream ``softmax`` recovers the
        mean posterior exactly, while ``"logit"`` returns the raw logit mean.
    """

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

    # Prepare normalization stats tensor if provided
    # norm_stats can be either:
    # - Per-recording: (C,) arrays for median/iqr (legacy)
    # - Per-epoch: (C, n_epochs) arrays for median/iqr (matches batch_edf_to_zarr_fp32.py)
    norm_stats_tensors = None
    per_epoch_norm = False
    if norm_stats is not None:
        norm_stats_tensors = {}
        for k, v in norm_stats.items():
            t = torch.from_numpy(v).to(device=device, dtype=torch.float32)
            norm_stats_tensors[k] = t
        # Check if per-epoch format (C, n_epochs)
        if norm_stats_tensors["median"].ndim == 2:
            per_epoch_norm = True
            logger.debug(
                "Using per-epoch normalization stats for wave_raw reconstruction"
            )
        else:
            # Legacy per-recording format: (C,) -> (1, C, 1) for broadcasting
            for k, t in norm_stats_tensors.items():
                if t.ndim == 1:
                    norm_stats_tensors[k] = t.view(1, -1, 1)  # (C,) -> (1, C, 1)

    # Configure Test-Time Augmentation (TTA) if requested
    tta_views = int(max(0, options.tta_passes))
    if tta_views > 0:
        logger.info(
            "TTA enabled: mode=%s, views=%d (noise=%.3f, amp=%.3f, shift=%.3f sec)",
            getattr(options, "tta_mode", "legacy"),
            tta_views,
            float(options.tta_noise),
            float(options.tta_amp),
            float(options.tta_shift),
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
            # Track whether the model accepts presence_mask kwarg; cache result after first call
            presence_kw_supported: bool | None = None

            # Request all-position logits [B, L, C] only in overlap-averaging mode; an
            # empty kwargs dict keeps the center-only call byte-identical when disabled.
            extra_fwd: dict[str, Any] = {"predict_all": True} if overlap_average else {}
            if options.recurrent_refinement_steps is not None:
                extra_fwd["recurrent_refinement_steps"] = (
                    options.recurrent_refinement_steps
                )

            def _forward_with_optional_mask(
                inp: torch.Tensor | dict[str, Any], mask: torch.Tensor
            ) -> Any:
                nonlocal presence_kw_supported
                if presence_kw_supported is True:
                    return model(inp, presence_mask=mask, **extra_fwd)
                if presence_kw_supported is False:
                    return model(inp, **extra_fwd)
                try:
                    out = model(inp, presence_mask=mask, **extra_fwd)
                    presence_kw_supported = True
                    return out
                except TypeError as e:
                    if "presence_mask" in str(e):
                        presence_kw_supported = False
                        return model(inp, **extra_fwd)
                    raise

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

                # Prepare input
                if norm_stats_tensors is not None:
                    if per_epoch_norm:
                        # Per-epoch stats: extract stats for epochs in this batch
                        # norm_stats_tensors has shape (C, n_epochs_total)
                        # We need stats for epochs [start_idx:end_idx] -> shape (B, C, 1)
                        batch_median = norm_stats_tensors["median"][
                            :, start_idx:end_idx
                        ].T.unsqueeze(
                            -1
                        )  # (B, C, 1)
                        batch_iqr = norm_stats_tensors["iqr"][
                            :, start_idx:end_idx
                        ].T.unsqueeze(
                            -1
                        )  # (B, C, 1)
                        batch_stats = {
                            "median": batch_median,
                            "iqr": batch_iqr,
                        }
                    else:
                        # Legacy per-recording stats: expand to batch size
                        # norm_stats_tensors already has shape (1, C, 1)
                        batch_stats = {
                            k: v.expand(batch_windows.shape[0], -1, -1)
                            for k, v in norm_stats_tensors.items()
                        }

                    # CRITICAL: Unnormalize data for engineered features
                    # Prenormalized data uses: normalized = (raw - median) / iqr
                    # To reverse: raw = normalized * iqr + median
                    # Shape: batch_windows is [B, L, C, T], stats are [B, C, 1]
                    # Need to broadcast: [B, 1, C, 1] for proper broadcasting
                    batch_median_expanded = batch_stats["median"].unsqueeze(
                        1
                    )  # [B, 1, C, 1]
                    batch_iqr_expanded = batch_stats["iqr"].unsqueeze(1)  # [B, 1, C, 1]
                    # Unnormalize: wave_raw = wave * iqr + median
                    wave_raw = (
                        batch_windows * batch_iqr_expanded + batch_median_expanded
                    )

                    # Pass as dict to trigger ModelWithPreproc logic and carry presence mask for base models
                    model_input = {
                        "wave": batch_windows,
                        "wave_raw": wave_raw,  # Unnormalized for engineered features
                        "norm_stats": batch_stats,
                        "presence_mask": batch_mask,
                    }
                else:
                    # Build dict input so ModelWithPreproc logic runs and the
                    # epoch-validity mask can be injected below.
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
                    model,
                    model_input,
                    batch_mask,
                    options,
                    device,
                    amp_dtype,
                    mc_samples,
                    mc_prob_pool,
                    lambda inp, mask=batch_mask: _forward_with_optional_mask(inp, mask),
                    collect_diagnostics=options.use_mc_dropout
                    and diagnostics_out is not None,
                )
                accumulator.add(
                    batch_out, moments, start_idx, end_idx, batch_mask, center_valid_t
                )

        return accumulator.finish(diagnostics_out)


@_with_recording_band_statistics
@with_recording_observations
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
    norm_stats: dict[str, np.ndarray] | None = None,
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
        norm_stats: Optional normalization statistics for wave_raw reconstruction.
            Can be either:
            - Per-recording (legacy): {'median': (C,), 'iqr': (C,)}
            - Per-epoch (matches batch_edf_to_zarr_fp32.py): {'median': (C, n_epochs), 'iqr': (C, n_epochs)}
            Per-epoch stats are used to reconstruct wave_raw for each epoch independently.

        center_valid_mask: Optional boolean array ``(n_epochs,)`` marking valid
            center epochs; only consulted under ``options.overlap_average`` to
            exclude unlabeled-center windows from the overlap accumulation. None
            means "all centers valid" (the pure-scoring default).
        epoch_channel_valid: Optional signal availability shaped
            ``[n_epochs, channels]``. It is expanded into each context window.

        diagnostics_out: Optional output dictionary populated with per-epoch sampling
            moments. Includes TTA/context variation when enabled, before refinement.

    Returns:
        Logits array ``(n_epochs, n_classes)``. When MC dropout is on the
        samples are pooled per ``options.mc_pooling``: ``"prob"`` returns
        ``log(mean(softmax(logits)))`` so a downstream ``softmax`` recovers the
        mean posterior exactly, while ``"logit"`` returns the raw logit mean.
    """
    from spectra.model.wrapped import ModelWithPreproc

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

    # Prepare normalization stats tensor if provided (for wave_raw reconstruction)
    # norm_stats can be either:
    # - Per-recording: (C,) arrays for median/iqr (legacy)
    # - Per-epoch: (C, n_epochs) arrays for median/iqr (matches batch_edf_to_zarr_fp32.py)
    norm_stats_tensors: dict[str, torch.Tensor] | None = None
    per_epoch_norm = False
    if norm_stats is not None:
        norm_stats_tensors = {}
        for k, v in norm_stats.items():
            t = torch.from_numpy(v).to(device=device, dtype=torch.float32)
            norm_stats_tensors[k] = t
        # Check if per-epoch format (C, n_epochs)
        if norm_stats_tensors["median"].ndim == 2:
            per_epoch_norm = True
            logger.info(
                "Using per-epoch normalization stats for wave_raw reconstruction (sequential mode)"
            )
        else:
            # Legacy per-recording format: (C,) -> (1, C, 1) for broadcasting
            for k, t in norm_stats_tensors.items():
                if t.ndim == 1:
                    norm_stats_tensors[k] = t.view(1, -1, 1)  # (C,) -> (1, C, 1)
            logger.info(
                "Prepared per-recording norm_stats tensors for wave_raw reconstruction (sequential mode)"
            )

    # Configure TTA
    tta_views = int(max(0, options.tta_passes))
    if tta_views > 0:
        logger.info(
            "TTA enabled: mode=%s, views=%d (noise=%.3f, amp=%.3f, shift=%.3f sec)",
            getattr(options, "tta_mode", "legacy"),
            tta_views,
            float(options.tta_noise),
            float(options.tta_amp),
            float(options.tta_shift),
        )

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

                if isinstance(model, ModelWithPreproc):
                    # Build dict so ModelWithPreproc logic runs and the
                    # epoch-validity mask can be injected below.
                    model_input: dict[str, Any] = {
                        "wave": batch,
                        "presence_mask": batch_presence_mask,
                    }
                elif norm_stats_tensors is not None:
                    # Reconstruct wave_raw for engineered features
                    if per_epoch_norm:
                        # Per-epoch stats: extract stats for epochs in this batch
                        # norm_stats_tensors has shape (C, n_epochs_total)
                        # We need stats for epochs [batch_start:batch_end] -> shape (B, C, 1)
                        batch_median = norm_stats_tensors["median"][
                            :, batch_start:batch_end
                        ].T.unsqueeze(
                            -1
                        )  # (B, C, 1)
                        batch_iqr = norm_stats_tensors["iqr"][
                            :, batch_start:batch_end
                        ].T.unsqueeze(
                            -1
                        )  # (B, C, 1)
                        # Add context dimension for broadcasting: (B, C, 1) -> (B, 1, C, 1)
                        batch_median = batch_median.unsqueeze(1)
                        batch_iqr = batch_iqr.unsqueeze(1)
                    else:
                        # Legacy per-recording stats: stats already have shape (1, C, 1)
                        batch_median = norm_stats_tensors["median"].unsqueeze(
                            1
                        )  # [1, 1, C, 1]
                        batch_iqr = norm_stats_tensors["iqr"].unsqueeze(
                            1
                        )  # [1, 1, C, 1]
                    # Unnormalize: wave_raw = wave * iqr + median
                    wave_raw = batch * batch_iqr + batch_median
                    model_input = {
                        "wave": batch,
                        "wave_raw": wave_raw,
                        "presence_mask": batch_presence_mask,
                    }
                else:
                    model_input = {
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
                if options.recurrent_refinement_steps is not None:
                    forward_kwargs["recurrent_refinement_steps"] = (
                        options.recurrent_refinement_steps
                    )

                batch_out, moments = _sample_inference_batch(
                    model,
                    model_input,
                    batch_presence_mask,
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


def _iterative_refinement_update(
    *,
    model: nn.Module,
    window_provider: Callable[[np.ndarray], torch.Tensor],
    presence_mask: torch.Tensor,
    device: torch.device,
    options: ScoreOptions,
    reasoning: ReasoningOptions,
    probabilities: np.ndarray,
    norm_stats: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_passes = int(max(1, reasoning.refinement_passes))
    threshold = float(reasoning.refinement_threshold)
    context_expansion = int(max(0, reasoning.context_expansion))

    if n_passes <= 1:
        confidences = np.max(probabilities, axis=-1).astype(np.float32, copy=False)
        return probabilities, confidences

    # Disable expensive uncertainty/augmentation for refinement; keep AMP + batch size.
    refine_options = replace(options, tta=0, tta_passes=0, use_mc_dropout=False)

    n_epochs = probabilities.shape[0]
    for pass_idx in range(1, n_passes):
        confidences = np.max(probabilities, axis=-1)
        uncertain_indices = np.where(confidences < threshold)[0]
        if uncertain_indices.size == 0:
            break

        windows = window_provider(uncertain_indices)
        logits_uncertain = run_inference(
            model,
            windows,
            presence_mask,
            device,
            refine_options,
            norm_stats=norm_stats,
        )
        probs_uncertain = _softmax_np(logits_uncertain, axis=-1)

        for local_i, idx in enumerate(uncertain_indices.tolist()):
            new_probs = probs_uncertain[local_i].astype(np.float64, copy=False)

            if context_expansion > 0:
                start_ctx = max(0, idx - context_expansion)
                end_ctx = min(n_epochs, idx + context_expansion + 1)
                neighbor_probs = probabilities[start_ctx:end_ctx].astype(
                    np.float64, copy=False
                )
                if neighbor_probs.shape[0] > 1:
                    distances = np.abs(np.arange(start_ctx, end_ctx) - idx).astype(
                        np.float64
                    )
                    weights = np.exp(-distances / 2.0)
                    weights = weights / np.sum(weights)
                    context_prior = np.sum(neighbor_probs * weights[:, None], axis=0)

                    blended = new_probs * context_prior
                    denom = float(np.sum(blended))
                    if denom > 0:
                        new_probs = blended / denom

            probabilities[idx] = new_probs.astype(np.float32, copy=False)

        logger.info(
            "Iterative refinement pass %d/%d: updated %d uncertain epochs (threshold=%.3f)",
            pass_idx + 1,
            n_passes,
            int(uncertain_indices.size),
            threshold,
        )

    confidences = np.max(probabilities, axis=-1).astype(np.float32, copy=False)
    return probabilities, confidences


def postprocess_logits_with_reasoning(
    logits: np.ndarray,
    *,
    reasoning: ReasoningOptions | None = None,
    model: nn.Module | None = None,
    window_provider: Callable[[np.ndarray], torch.Tensor] | None = None,
    presence_mask: torch.Tensor | None = None,
    device: torch.device | None = None,
    options: ScoreOptions | None = None,
    norm_stats: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Apply optional reasoning steps on top of base inference logits."""
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits shape (n_epochs, n_classes), got {logits.shape}"
        )

    raw_probabilities = _softmax_np(logits, axis=-1)
    raw_predictions = np.argmax(raw_probabilities, axis=-1).astype(np.int64)

    # Cheap-pass uncertainty-flag scores (computed on the deterministic pass;
    # zero extra forward passes). These do not affect predictions.
    flag_scores = compute_flag_scores(raw_probabilities)

    if reasoning is None:
        confidences = np.max(raw_probabilities, axis=-1).astype(np.float32, copy=False)
        return {
            "predictions": raw_predictions,
            "probabilities": raw_probabilities,
            "raw_predictions": raw_predictions,
            "raw_probabilities": raw_probabilities,
            "confidences": confidences,
            **flag_scores,
        }

    probabilities = raw_probabilities

    # Iterative refinement requires access to the model + epoch windows.
    if bool(reasoning.use_iterative_refinement):
        if (
            model is None
            or window_provider is None
            or presence_mask is None
            or device is None
            or options is None
        ):
            raise ValueError(
                "Iterative refinement requires model, window_provider, presence_mask, device, and options"
            )
        probabilities, confidences = _iterative_refinement_update(
            model=model,
            window_provider=window_provider,
            presence_mask=presence_mask,
            device=device,
            options=options,
            reasoning=reasoning,
            probabilities=probabilities.copy(),
            norm_stats=norm_stats,
        )
    else:
        confidences = np.max(probabilities, axis=-1).astype(np.float32, copy=False)

    predictions = np.argmax(probabilities, axis=-1).astype(np.int64)

    return {
        "predictions": predictions.astype(np.int64, copy=False),
        "probabilities": probabilities.astype(np.float32, copy=False),
        "raw_predictions": raw_predictions,
        "raw_probabilities": raw_probabilities,
        "confidences": confidences,
        **flag_scores,
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
        model: The inference model, wrapped or compiled.
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
    outside :func:`infer_with_reasoning` must wrap the forward passes in this,
    or the per-recording normalisation silently degrades to identity and the
    encoder sees an envelope distribution it never trained on.

    Args:
        model: The inference model, wrapped or compiled.
        epoch_windows: ``[n_windows, context_len, channels, samples]`` for this
            recording. A single-epoch recording may be passed as
            ``[n_epochs, 1, channels, samples]``.
        device: Device to compute the statistics on.
        epoch_valid: Optional bool ``[n_windows]`` restricting the statistics
            to signal-valid centre epochs, as :func:`infer_with_reasoning` does.
    """
    from spectra.model.band_norm_access import recording_norm_modules
    from spectra.model.recording_conditioning import (
        get_recording_conditioner,
        pinned_recording_observations,
    )

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
        with ExitStack() as context:
            if get_recording_conditioner(model) is not None:
                centre = epoch_windows[:, epoch_windows.shape[1] // 2]
                context.enter_context(
                    pinned_recording_observations(
                        model, centre, mask, epoch_valid=epoch_valid
                    )
                )
            yield
    finally:
        for norm, mean, std in previous:
            norm._inference_mean, norm._inference_std = mean, std


@_with_recording_band_statistics
@with_recording_observations
def infer_with_reasoning(
    model: nn.Module,
    epoch_windows: torch.Tensor,
    presence_mask: torch.Tensor,
    device: torch.device,
    options: ScoreOptions,
    *,
    reasoning: ReasoningOptions | None = None,
    norm_stats: dict[str, np.ndarray] | None = None,
    center_valid_mask: np.ndarray | None = None,
    epoch_channel_mask: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Run inference + optional reasoning for an already-windowed recording.

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
        norm_stats=norm_stats,
        center_valid_mask=center_valid_mask,
        epoch_channel_mask=epoch_channel_mask,
        diagnostics_out=diagnostics,
    )

    def _provider(idxs: np.ndarray) -> torch.Tensor:
        idxs = np.asarray(idxs, dtype=np.int64)
        return epoch_windows[idxs]

    result = postprocess_logits_with_reasoning(
        logits,
        reasoning=reasoning,
        model=model,
        window_provider=_provider,
        presence_mask=presence_mask,
        device=device,
        options=options,
        norm_stats=norm_stats,
    )
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
    reasoning: ReasoningOptions | None = None,
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
        reasoning: Optional inference-time reasoning options
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
    proc_cfg = _extract_proc_cfg(checkpoint_data)
    if proc_cfg is not None:
        logger.info(
            "Detected preprocessing_config in checkpoint (version=%s, dataset=%s)",
            getattr(proc_cfg, "version", "unknown"),
            getattr(proc_cfg, "dataset_name", "unknown"),
        )
    else:
        logger.info(
            "Checkpoint missing preprocessing_config; using runtime defaults for preprocessing"
        )
    expected_time_len = _sync_options_with_checkpoint_metadata(options, checkpoint_data)
    logger.info(
        "Inference sampling parameters: fs=%d Hz, epoch_sec=%d (checkpoint time_len=%s)",
        options.fs,
        options.epoch_sec,
        expected_time_len if expected_time_len is not None else "unknown",
    )

    from spectra.model.wrapped import ModelWithPreproc
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
    external_norm_stats = {
        key: np.asarray(
            [
                prepared.normalization.get(name, {}).get(key, default)
                for name in inference_channel_names
            ],
            dtype=np.float32,
        )
        for key, default in (("median", 0.0), ("iqr", 1.0))
    }
    # These arrays are already normalized by the converter's shared helper.
    # Reconstruct wrappers for strict state loading, then score the underlying
    # model directly to avoid applying embedded normalization a second time.
    report_progress("Loading checkpoint", 40)
    model_channel_names, _ = resolve_model_channel_names(
        checkpoint_data, canonical_channels=canonical_channels
    )
    model, checkpoint_dict = load_model_from_checkpoint(
        checkpoint,
        resolved_device,
        checkpoint_data=checkpoint_data,
        options=options,
        proc_cfg=proc_cfg,
        channel_names=model_channel_names,
    )
    if isinstance(model, ModelWithPreproc):
        model = model.model
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

    # Keep recording support available through both scoring and refinement.
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
                norm_stats=external_norm_stats,  # For wave_raw reconstruction in engineered features
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
                norm_stats=external_norm_stats,  # For wave_raw reconstruction in engineered features
                epoch_channel_mask=epoch_channel_mask_windows,
                diagnostics_out=diagnostics,
            )

        # Optional reasoning pipeline on top of base logits.
        #
        # NOTE: ``epoch_windows`` and ``preprocessed_data`` are enclosing-scope
        # locals captured by this closure. They are ``del``-ed in the ``finally``
        # cleanup below, so static analyzers (ruff F821 / pyright possibly-unbound)
        # flag the references here. At runtime this closure is only invoked by
        # ``postprocess_logits_with_reasoning`` *inside this try block*, before the
        # cleanup runs, so the captured values are always bound. Suppress the
        # false positives for both tools.
        def _window_provider(idxs: np.ndarray) -> torch.Tensor:
            idxs = np.asarray(idxs, dtype=np.int64)
            if not options.sequential_loading and epoch_windows is not None:  # type: ignore[name-defined]  # noqa: F821
                return epoch_windows[idxs]  # type: ignore[name-defined]  # noqa: F821
            return create_epoch_windows_for_indices(
                preprocessed_data,  # type: ignore[name-defined]  # noqa: F821
                idxs,
                options.fs,
                options.epoch_sec,
                options.context_half,
            )

        post = postprocess_logits_with_reasoning(
            logits,
            reasoning=reasoning,
            model=model,
            window_provider=_window_provider,
            presence_mask=presence_mask_tensor,
            device=resolved_device,
            options=options,
            norm_stats=external_norm_stats,  # For wave_raw in reasoning re-inference
        )

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
