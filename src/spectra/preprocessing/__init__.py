"""Versioned preprocessing configuration and embedded normalization."""

from __future__ import annotations

from .config import ProcCfg, cfg_hash, load_proc_config, save_proc_config
from .normalizer import EmbeddedPreproc
from .robust_normalization import (
    normalize_channel,
    normalize_channel_masked,
    normalize_channel_masked_with_validity,
)
from .signal_quality import epoch_signal_validity, recording_epoch_signal_validity

__all__ = [
    "ProcCfg",
    "cfg_hash",
    "load_proc_config",
    "save_proc_config",
    "EmbeddedPreproc",
    "normalize_channel",
    "normalize_channel_masked",
    "normalize_channel_masked_with_validity",
    "epoch_signal_validity",
    "recording_epoch_signal_validity",
]
