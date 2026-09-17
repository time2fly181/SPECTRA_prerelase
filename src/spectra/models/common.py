"""Attention backend selection, initialization, and normalization factories."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

try:
    from torch.nn.attention import (
        SDPBackend as _TorchSDPBackend,
    )
    from torch.nn.attention import (
        sdpa_kernel as _torch_sdpa_kernel_ctx,
    )
except (
    ImportError,
    AttributeError,
):  # pragma: no cover - fallback when torch.nn.attention is missing
    _torch_sdpa_kernel_ctx = None  # type: ignore[assignment]
    _TorchSDPBackend = None  # type: ignore[assignment]

_SDP_BACKEND_CHOICES = ("auto", "flash", "mem_efficient", "math")


def _sdp_kernel_context(policy: str):
    """Return a context manager for selecting SDPA backend."""
    if policy not in _SDP_BACKEND_CHOICES or policy == "auto":
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    if _torch_sdpa_kernel_ctx is None or _TorchSDPBackend is None:
        return nullcontext()
    if policy == "flash":
        backends = [
            _TorchSDPBackend.FLASH_ATTENTION,
            _TorchSDPBackend.EFFICIENT_ATTENTION,
            _TorchSDPBackend.MATH,
        ]
    elif policy == "mem_efficient":
        backends = [
            _TorchSDPBackend.EFFICIENT_ATTENTION,
            _TorchSDPBackend.MATH,
        ]
    elif policy == "math":
        backends = [_TorchSDPBackend.MATH]
    else:
        return nullcontext()
    try:
        return _torch_sdpa_kernel_ctx(backends)
    except RuntimeError:
        return nullcontext()


def transformer_init_(module: nn.Module) -> None:
    """Initialize supported layers in place for model construction.

    Linear weights use Xavier uniform initialization; Conv1d weights use Kaiming
    normal with ReLU gain. Normalization scales start at one and biases at zero.
    Checkpoint loading subsequently replaces saved parameters.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _default_group_count(channels: int) -> int:
    """Select a reasonable group count for GroupNorm."""
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _make_norm1d(norm: str, num_channels: int) -> nn.Module:
    """Create a 1D normalization layer (bn or gn)."""
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm1d(num_channels)
    if norm == "gn":
        return nn.GroupNorm(_default_group_count(num_channels), num_channels)
    raise ValueError(f"Unsupported norm '{norm}'. Expected 'bn' or 'gn'.")
