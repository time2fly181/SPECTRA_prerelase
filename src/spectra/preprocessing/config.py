"""Preprocessing configuration with versioning and hashing.

This module defines a Pydantic model for all preprocessing parameters,
including channel mapping, rereferencing, resampling, filtering, and
normalization. The config can be serialized to YAML and hashed for
versioning to ensure reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Literal

import yaml

# Suppress Pydantic v2.11+ warnings about field attributes in nested models
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic._internal._generate_schema",
    message=".*'repr' attribute.*has no effect.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic._internal._generate_schema",
    message=".*'frozen' attribute.*has no effect.*",
)

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as e:
    raise ImportError(
        "pydantic is required for preprocessing config. "
        "Install with: pip install pydantic"
    ) from e

__all__ = ["ProcCfg", "cfg_hash", "load_proc_config", "save_proc_config"]


class ChannelMappingCfg(BaseModel):
    """Channel mapping and rereferencing configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    canonical_json: str | None = Field(
        None, description="Path to canonical channel list JSON file"
    )
    enable_rereferencing: bool = Field(
        True, description="Enable algebraic channel derivation via linear rereferencing"
    )
    enable_substitutions: bool = Field(
        True, description="Enable channel label substitutions (e.g., A1<->M1, A2<->M2)"
    )


class ResamplingCfg(BaseModel):
    """Resampling configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_fs: float = Field(128.0, description="Target sampling frequency in Hz")


class FilterCfg(BaseModel):
    """Per-channel-type filtering configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    eeg_lowcut: float | None = Field(
        0.3, description="EEG high-pass cutoff in Hz (None to disable)"
    )
    eeg_highcut: float | None = Field(
        35.0, description="EEG low-pass cutoff in Hz (None to disable)"
    )
    eog_lowcut: float | None = Field(
        0.3, description="EOG high-pass cutoff in Hz (None to disable)"
    )
    eog_highcut: float | None = Field(
        35.0, description="EOG low-pass cutoff in Hz (None to disable)"
    )
    emg_lowcut: float | None = Field(
        10.0, description="EMG high-pass cutoff in Hz (None to disable)"
    )
    emg_highcut: float | None = Field(
        45.0, description="EMG low-pass cutoff in Hz (None to disable)"
    )
    ecg_lowcut: float | None = Field(
        0.5, description="ECG high-pass cutoff in Hz (None to disable)"
    )
    ecg_highcut: float | None = Field(
        40.0, description="ECG low-pass cutoff in Hz (None to disable)"
    )
    filter_order: int = Field(4, description="Butterworth filter order")


class NormalizationCfg(BaseModel):
    """Normalization configuration matching normalization.py logic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Literal["iqr", "sigma", "keep", "percentile_1_99"] = Field(
        "iqr",
        description="Normalization mode: 'iqr' (recommended, matches batch_edf_to_zarr_fp32.py), 'sigma', 'keep', or 'percentile_1_99' (P1/P99 clip-and-scale to [-1, 1])",
    )
    clip_threshold: float = Field(
        20.0, description="Hard clipping threshold in IQR units (±clip_threshold)"
    )
    q1_percentile: float = Field(25.0, description="Lower percentile for IQR mode")
    q3_percentile: float = Field(75.0, description="Upper percentile for IQR mode")
    # Percentile_1_99 mode parameters
    p_low_percentile: float = Field(
        1.0, description="Lower percentile for percentile_1_99 mode (default: 1.0)"
    )
    p_high_percentile: float = Field(
        99.0, description="Upper percentile for percentile_1_99 mode (default: 99.0)"
    )
    # Legacy sigma mode parameters
    zspan: float = Field(
        5.152,
        description="Z-span for sigma estimation from percentile range (P0.5-P99.5 = 5.152)",
    )
    clip_method: Literal["hard", "soft"] = Field(
        "hard", description="Sigma-mode clipping: 'hard' for clamp, 'soft' for tanh"
    )
    clamp: dict[str, float | str] = Field(
        default_factory=lambda: {
            "k_eeg": 10.0,
            "k_emg": 12.0,
            "tanh_c": 10.0,
        },
        description="Sigma-mode clamp thresholds (must match PSGNormalizer constants)",
    )


class StorageCfg(BaseModel):
    """Zarr storage configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dtype: Literal["int16", "float16", "float32"] = Field(
        "int16", description="Storage data type"
    )
    chunk_seconds: float = Field(
        30.0, description="Chunk size in seconds (~epoch length)"
    )
    compression: Literal["blosc", "zstd"] = Field(
        "blosc", description="Compression codec"
    )
    compression_level: int = Field(5, description="Compression level (1-9)")
    shuffle: Literal["bitshuffle", "shuffle", "none"] = Field(
        "bitshuffle", description="Shuffle filter (bitshuffle recommended for int16)"
    )


class ProcCfg(BaseModel):
    """Complete preprocessing configuration.

    This model captures all preprocessing parameters for reproducibility.
    The configuration can be hashed to create a version identifier that
    is saved with model checkpoints and scoring outputs.

    Example:
        >>> cfg = ProcCfg(
        ...     version="1.0.0",
        ...     channel_mapping=ChannelMappingCfg(canonical_json="channels.json"),
        ...     resampling=ResamplingCfg(target_fs=100.0),
        ...     filtering=FilterCfg(),
        ...     normalization=NormalizationCfg(),
        ...     storage=StorageCfg()
        ... )
        >>> hash_id = cfg_hash(cfg)
        >>> save_proc_config(cfg, "preprocessing_config.yaml")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    version: str = Field("1.0.0", description="Preprocessing pipeline version")
    channel_mapping: ChannelMappingCfg = Field(
        default_factory=ChannelMappingCfg,  # type: ignore[reportArgumentType]
        description="Channel mapping and rereferencing configuration",
    )
    resampling: ResamplingCfg = Field(
        default_factory=ResamplingCfg,  # type: ignore[reportArgumentType]
        description="Resampling configuration",
    )
    filtering: FilterCfg = Field(
        default_factory=FilterCfg,  # type: ignore[reportArgumentType]
        description="Per-channel-type filtering configuration",
    )
    normalization: NormalizationCfg = Field(
        default_factory=NormalizationCfg,  # type: ignore[reportArgumentType]
        description="Normalization configuration",
    )
    storage: StorageCfg = Field(
        default_factory=StorageCfg,  # type: ignore[reportArgumentType]
        description="Zarr storage configuration",
    )
    dataset_name: str | None = Field(None, description="Dataset name for attribution")


def cfg_hash(cfg: ProcCfg) -> str:
    """Compute a deterministic hash of the preprocessing configuration.

    Args:
        cfg: Preprocessing configuration

    Returns:
        12-character hex hash string

    Example:
        >>> cfg = ProcCfg()
        >>> hash_id = cfg_hash(cfg)
        >>> print(hash_id)  # e.g., "a3f2e1d9c8b7"
    """
    blob = json.dumps(cfg.model_dump(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def save_proc_config(cfg: ProcCfg, path: str | Path) -> str:
    """Save preprocessing configuration to YAML file with hash.

    Args:
        cfg: Preprocessing configuration
        path: Output YAML file path

    Returns:
        Configuration hash string

    Example:
        >>> cfg = ProcCfg()
        >>> hash_id = save_proc_config(cfg, "preprocessing.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add hash to config
    config_dict = cfg.model_dump()
    config_dict["config_hash"] = cfg_hash(cfg)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    return config_dict["config_hash"]


def load_proc_config(path: str | Path) -> tuple[ProcCfg, str]:
    """Load preprocessing configuration from YAML file.

    Args:
        path: Input YAML file path

    Returns:
        Tuple of (config, hash_string)

    Example:
        >>> cfg, hash_id = load_proc_config("preprocessing.yaml")
        >>> print(f"Loaded config with hash: {hash_id}")
    """
    path = Path(path)

    with open(path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Extract and verify hash
    stored_hash = config_dict.pop("config_hash", None)

    cfg = ProcCfg(**config_dict)
    computed_hash = cfg_hash(cfg)

    if stored_hash and stored_hash != computed_hash:
        warnings.warn(
            f"Config hash mismatch! Stored: {stored_hash}, Computed: {computed_hash}. "
            "Config may have been modified.",
            stacklevel=2,
        )

    return cfg, computed_hash
