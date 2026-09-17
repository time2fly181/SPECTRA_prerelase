"""Autocast contexts with reported backend fallback and exception propagation."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Generator
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def _is_autocast_available(device_type: str) -> bool:
    amp_module = getattr(torch, "amp", None)
    autocast_mode = getattr(amp_module, "autocast_mode", None)
    checker = getattr(autocast_mode, "is_autocast_available", None)
    if callable(checker):
        try:
            return bool(checker(device_type))
        except Exception:
            return False
    return device_type == "cuda"


@contextmanager
def _autocast_or_fallback(
    device_type: str,
    autocast_dtype: torch.dtype,
    warning_message: str,
) -> Generator[None, None, None]:
    if autocast_dtype not in (torch.float16, torch.bfloat16):
        warnings.warn(warning_message, RuntimeWarning, stacklevel=3)
        yield
        return

    if not _is_autocast_available(device_type):
        warnings.warn(
            f"autocast_context: autocast is not available for device_type={device_type}. "
            "Disabling autocast.",
            RuntimeWarning,
            stacklevel=3,
        )
        yield
        return

    try:
        autocast_cm = torch.autocast(device_type=device_type, dtype=autocast_dtype)
        autocast_cm.__enter__()
    except Exception as exc:
        warnings.warn(
            f"autocast_context: failed to enable autocast for device_type={device_type}, "
            f"dtype={autocast_dtype}: {exc}. Disabling autocast.",
            RuntimeWarning,
            stacklevel=3,
        )
        logger.debug(
            "Disabling autocast after backend rejected device_type=%s dtype=%s",
            device_type,
            autocast_dtype,
            exc_info=True,
        )
        yield
        return

    try:
        yield
    except BaseException as exc:
        suppress = autocast_cm.__exit__(type(exc), exc, exc.__traceback__)
        if not suppress:
            raise
    else:
        autocast_cm.__exit__(None, None, None)


@contextmanager
def autocast_context(
    device_or_str: torch.device | str,
    maybe_use_amp_or_dtype: object | None = None,
    dtype_arg: torch.dtype | None = None,
    *,
    dtype: torch.dtype | None = None,
    use_amp: bool | None = None,
) -> Generator[None, None, None]:
    """Return an autocast context manager.

    Supports these calling conventions for compatibility:
    - autocast_context(device: torch.device, dtype: Optional[torch.dtype])
    - autocast_context(device: str, use_amp: bool, amp_dtype: torch.dtype)
    - autocast_context(..., dtype=..., use_amp=...) (keyword-friendly)

    When AMP is enabled, supports both fp16 (torch.float16) and bf16 (torch.bfloat16).
    Unsupported autocast setup emits a warning and continues without autocast.
    Exceptions raised by the wrapped model body are propagated.
    """
    # Honor keyword overrides if provided
    kw_dtype = dtype
    kw_use_amp = use_amp

    # Legacy signature: (device: str, use_amp: bool, dtype)
    if isinstance(device_or_str, str):
        device_str = device_or_str
        use_amp_val = bool(
            kw_use_amp if kw_use_amp is not None else maybe_use_amp_or_dtype
        )
        amp_dtype = kw_dtype if kw_dtype is not None else dtype_arg
        if use_amp_val and amp_dtype is not None:
            with _autocast_or_fallback(
                device_str,
                amp_dtype,
                f"autocast_context: unsupported amp_dtype={amp_dtype}, "
                "expected torch.float16 or torch.bfloat16. Disabling autocast.",
            ):
                yield
            return
        yield
        return

    # New signature: (device: torch.device, dtype)
    device = device_or_str
    chosen_dtype = None
    # Prefer explicit keyword dtype, else positional dtype_arg, else legacy positional torch.dtype
    if kw_dtype is not None:
        chosen_dtype = kw_dtype
    elif dtype_arg is not None:
        chosen_dtype = dtype_arg
    elif isinstance(maybe_use_amp_or_dtype, torch.dtype):
        chosen_dtype = maybe_use_amp_or_dtype

    if isinstance(device, torch.device) and isinstance(chosen_dtype, torch.dtype):
        with _autocast_or_fallback(
            device.type,
            chosen_dtype,
            f"autocast_context: unsupported dtype={chosen_dtype}, "
            "expected torch.float16 or torch.bfloat16. Disabling autocast.",
        ):
            yield
        return
    yield
