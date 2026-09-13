"""Shared helpers for selecting and managing PyTorch devices across scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch


def is_mps_available() -> bool:
    """Return True if the Metal Performance Shaders (MPS) backend is usable."""

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False

    is_built = getattr(mps_backend, "is_built", None)
    if callable(is_built):
        try:
            if not is_built():
                return False
        except RuntimeError:
            return False

    is_available = getattr(mps_backend, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except RuntimeError:
            return False
    return False


def _normalize_device_name(device: torch.device | str | None) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        normalized = device.strip().lower()
        return normalized or "cpu"
    return "cpu"


@dataclass(frozen=True)
class ResolvedDevice:
    """Resolved runtime device with warning and description metadata."""

    requested: str
    resolved: str
    warning: str | None
    description: str

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.resolved)


def resolve_device(preference: str = "auto") -> ResolvedDevice:
    """Choose the best device given a preference and availability."""

    pref = _normalize_device_name(preference)

    warning: str | None = None
    if pref == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif is_mps_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    elif pref == "cuda":
        if torch.cuda.is_available():
            resolved = "cuda"
        else:
            resolved = "cpu"
            warning = "CUDA was requested but is not available; falling back to CPU."
    elif pref == "mps":
        if is_mps_available():
            resolved = "mps"
        else:
            resolved = "cpu"
            warning = "MPS was requested but is not available; falling back to CPU."
    else:
        resolved = "cpu"

    return ResolvedDevice(
        requested=pref,
        resolved=resolved,
        warning=warning,
        description=describe_device(resolved),
    )


def select_device(preference: str = "auto") -> tuple[str, str | None]:
    """Backward-compatible device selector returning `(device, warning)`."""

    resolved = resolve_device(preference)
    return resolved.resolved, resolved.warning


def describe_device(device: str | torch.device) -> str:
    """Return a human-readable description of the selected device."""

    device_name = _normalize_device_name(device)

    if device_name == "cuda":
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                return f"{name} (CUDA)"
            except Exception:  # pragma: no cover - defensive
                pass
        return "CUDA GPU"

    if device_name == "mps":
        return "Apple Silicon GPU (MPS backend)"

    return "CPU"


def clear_device_cache(
    device: torch.device | str | None,
    *,
    synchronize: bool = True,
    reset_peak_memory_stats: bool = False,
) -> bool:
    """Best-effort cache cleanup for CUDA and MPS backends."""

    device_name = _normalize_device_name(device)

    if device_name == "cuda":
        if not torch.cuda.is_available():
            return False
        torch.cuda.empty_cache()
        if reset_peak_memory_stats:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        if synchronize:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        torch.cuda.empty_cache()
        return True

    if device_name == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is None or not is_mps_available():
            return False
        if synchronize and hasattr(mps_module, "synchronize"):
            try:
                mps_module.synchronize()
            except Exception:
                pass
        if hasattr(mps_module, "empty_cache"):
            try:
                mps_module.empty_cache()
                return True
            except Exception:
                return False
        return False

    return False


def get_device_memory_stats(
    device: torch.device | str | None = "auto",
) -> dict[str, object] | None:
    """Return backend memory statistics when the runtime exposes them."""

    device_name = _normalize_device_name(device)
    if device_name == "auto":
        device_name = resolve_device("auto").resolved

    if device_name == "cuda":
        if not torch.cuda.is_available():
            return None
        try:
            allocated = int(torch.cuda.memory_allocated(0))
            total = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            return None
        return {
            "device": "cuda",
            "allocated_bytes": allocated,
            "total_bytes": total,
            "description": describe_device("cuda"),
        }

    if device_name == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is None or not is_mps_available():
            return None

        allocated_bytes: int | None = None
        total_bytes: int | None = None

        current_allocated = getattr(mps_module, "current_allocated_memory", None)
        if callable(current_allocated):
            try:
                allocated_bytes = int(cast(Any, current_allocated()))
            except Exception:
                allocated_bytes = None

        recommended_max = getattr(mps_module, "recommended_max_memory", None)
        if callable(recommended_max):
            try:
                total_bytes = int(cast(Any, recommended_max()))
            except Exception:
                total_bytes = None

        if allocated_bytes is None and total_bytes is None:
            return {
                "device": "mps",
                "allocated_bytes": None,
                "total_bytes": None,
                "description": describe_device("mps"),
            }

        return {
            "device": "mps",
            "allocated_bytes": allocated_bytes,
            "total_bytes": total_bytes,
            "description": describe_device("mps"),
        }

    return None
