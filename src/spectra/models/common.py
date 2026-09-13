# spectra.models/common.py
"""
Common utilities, constants, and base components for PSG models.
"""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from typing import cast

import torch
import torch.nn as nn

from spectra.data.channel import (
    POSITION_PAD_IDX,
    POSITION_TO_INDEX,
    REFERENCE_PAD_IDX,
    REFERENCE_TO_INDEX,
    TYPE_PAD_IDX,
    TYPE_TO_INDEX,
)

# ---------------------------
# DropPath (Stochastic Depth)
# ---------------------------

_TimmDropPathImpl: type[nn.Module]

try:
    from timm.layers.drop import DropPath as _TimmDropPathImported

    _TimmDropPathImpl = cast(type[nn.Module], _TimmDropPathImported)
except ImportError:
    try:
        from timm.layers import (
            DropPath as _TimmDropPathImported,  # type: ignore[attr-defined]
        )

        _TimmDropPathImpl = cast(type[nn.Module], _TimmDropPathImported)
    except ImportError:  # pragma: no cover - fallback if timm is unavailable

        class _FallbackDropPath(nn.Module):
            """Stochastic depth per sample (when timm is unavailable)."""

            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.drop_prob == 0.0 or (not self.training):
                    return x
                keep_prob = 1.0 - self.drop_prob
                if keep_prob <= 0.0:
                    return torch.zeros_like(x)
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(
                    shape, dtype=x.dtype, device=x.device
                )
                random_tensor.floor_()
                return x.div(keep_prob) * random_tensor

        _TimmDropPathImpl = _FallbackDropPath


DropPath = _TimmDropPathImpl

# ---------------------------
# FusedLayerNorm from apex
# ---------------------------

_FUSED_NORM_IMPORT_ERROR = ""
_FusedLayerNormImpl: type[nn.LayerNorm] | None = None


def _load_apex_fused_layer_norm(module_name: str) -> type[nn.LayerNorm]:
    module = importlib.import_module(module_name)
    fused = module.FusedLayerNorm
    if not isinstance(fused, type):
        raise TypeError(
            f"Expected FusedLayerNorm to be a class in {module_name}, got {type(fused)!r}"
        )
    return cast(type[nn.LayerNorm], fused)


try:  # Preferred public re-export used by most apex wheels
    _FusedLayerNormImpl = _load_apex_fused_layer_norm("apex.normalization")
except ImportError as exc:
    _FUSED_NORM_IMPORT_ERROR = f"{exc.__class__.__name__}: {exc}"
    try:  # Some builds only expose the implementation via the submodule path
        _FusedLayerNormImpl = _load_apex_fused_layer_norm(
            "apex.normalization.fused_layer_norm"
        )
    except ImportError as sub_exc:
        _FUSED_NORM_IMPORT_ERROR = (
            f"{_FUSED_NORM_IMPORT_ERROR}; {sub_exc.__class__.__name__}: {sub_exc}"
        )
        _FusedLayerNormImpl = None
    except (AttributeError, TypeError) as sub_exc:
        _FUSED_NORM_IMPORT_ERROR = (
            f"{_FUSED_NORM_IMPORT_ERROR}; {sub_exc.__class__.__name__}: {sub_exc}"
        )
        _FusedLayerNormImpl = None
except (AttributeError, TypeError) as exc:
    _FUSED_NORM_IMPORT_ERROR = f"{exc.__class__.__name__}: {exc}"
    _FusedLayerNormImpl = None

if _FusedLayerNormImpl is not None:
    FusedLayerNorm = _FusedLayerNormImpl
    _FUSED_NORM_AVAILABLE = True
else:
    FusedLayerNorm = nn.LayerNorm
    _FUSED_NORM_AVAILABLE = False

# ---------------------------
# Channel metadata constants
# ---------------------------

TYPE_VOCAB_SIZE = max(TYPE_TO_INDEX.values(), default=TYPE_PAD_IDX) + 1
POSITION_VOCAB_SIZE = max(POSITION_TO_INDEX.values(), default=POSITION_PAD_IDX) + 1
REFERENCE_VOCAB_SIZE = max(REFERENCE_TO_INDEX.values(), default=REFERENCE_PAD_IDX) + 1

# ---------------------------
# SDPA backend context
# ---------------------------

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


# ---------------------------
# Common utilities
# ---------------------------


def kaiming_init_(module: nn.Module) -> None:
    """Kaiming init for Conv/Linear, zeros for bias; leave norm layers default."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def transformer_init_(module: nn.Module) -> None:
    """Xavier init for Transformer layers with GELU activation.

    Applies Xavier (Glorot) initialization to linear layers, which is more
    appropriate for GELU activations than Kaiming (which assumes ReLU-like
    half-rectifier behavior). GELU is smoother and closer to linear near zero,
    making Xavier's fan_avg variance scaling more suitable for deep transformer
    stacks with pre-LayerNorm.

    For Conv1d layers (in CNN encoders), keeps Kaiming initialization as it
    works well for feature extraction even with GELU.
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
    # Keep Kaiming for Conv1d in CNN encoder (works well for feature extraction)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def resolve_sampling_params(
    time_len: int,
    fs: int | None,
    *,
    default_epoch_sec: int = 30,
) -> tuple[int, int]:
    """
    Derive a positive sampling frequency and epoch duration.

    Args:
        time_len: Samples per epoch in the checkpoint/config.
        fs: Explicit sampling frequency if provided (may be ``None``).
        default_epoch_sec: Epoch duration used when no explicit fs is given.

    Returns:
        (resolved_fs, resolved_epoch_sec)

    Raises:
        ValueError: If parameters are invalid or a reasonable fs cannot be inferred.
    """
    if time_len <= 0:
        raise ValueError(f"time_len must be positive, got {time_len}")

    if fs is not None:
        if fs <= 0:
            raise ValueError(f"fs must be positive when provided, got {fs}")
        epoch_sec = max(1, int(round(time_len / float(fs))))
        return fs, epoch_sec

    if default_epoch_sec <= 0:
        raise ValueError(f"default_epoch_sec must be positive, got {default_epoch_sec}")

    if time_len % default_epoch_sec == 0:
        inferred_fs = time_len // default_epoch_sec
        if inferred_fs > 0:
            return inferred_fs, default_epoch_sec

    approx_fs = int(round(time_len / float(default_epoch_sec)))
    if approx_fs > 0:
        return approx_fs, default_epoch_sec

    raise ValueError(
        "Unable to infer sampling frequency. Provide fs explicitly in the model "
        f"configuration or ensure time_len ({time_len}) reflects samples per epoch."
    )


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
