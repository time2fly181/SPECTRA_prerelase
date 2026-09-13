"""Lifecycle and whole-recording context for optional conditioned epoch encoders."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from spectra.models.recording_conditioning import RecordingConditioner

logger = logging.getLogger(__name__)


def conditioning_encoder(model: nn.Module) -> nn.Module | None:
    """Find an enabled encoder through the supported context/compile wrappers."""
    pending = [model]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if getattr(current, "recording_conditioner", None) is not None:
            return current
        for name in (
            "model",
            "module",
            "_orig_mod",
            "epoch_encoder",
            "cnn",
            "cnn_extractor",
        ):
            child = getattr(current, name, None)
            if isinstance(child, nn.Module):
                pending.append(child)
    return None


def get_recording_conditioner(model: nn.Module) -> RecordingConditioner | None:
    """Return the optional conditioner without changing any model state."""
    encoder = conditioning_encoder(model)
    return getattr(encoder, "recording_conditioner", None)


@contextmanager
def pinned_recording_observations(
    model: nn.Module,
    wave: torch.Tensor,
    presence_mask: torch.Tensor,
    *,
    epoch_valid: torch.Tensor | None = None,
) -> Iterator[None]:
    """Compute and pin one complete recording's observations, restoring nesting.

    ``wave`` is normalized ``[E,C,T]``. No labels, class probabilities, site IDs
    or cohort metadata enter this pass. A cache provider is ignored while pinned.
    """
    conditioner = get_recording_conditioner(model)
    if conditioner is None:
        yield
        return
    previous: Any = conditioner.pinned_observations
    tokens, valid = conditioner.compute_observations(wave, presence_mask, epoch_valid)
    conditioner.set_observations(tokens, valid)
    try:
        yield
    finally:
        if previous is None:
            conditioner.clear_observations()
        else:
            conditioner.set_observations(*previous)


def with_recording_observations[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Pin complete-recording support around a public scoring operation.

    Works for windowed and sequential scoring. Selection uses signal masks only;
    the label-dependent center_valid_mask never enters recording conditioning.
    Nested scoring/reasoning calls reuse the outer operation's observations.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = signature.bind(*args, **kwargs)
        values = bound.arguments
        model = values["model"]
        conditioner = get_recording_conditioner(model)
        if conditioner is None or conditioner.pinned_observations is not None:
            return function(*args, **kwargs)
        from spectra.models.recording_conditioning import select_recording_epochs

        presence = torch.as_tensor(values["presence_mask"], dtype=torch.bool)
        if "epoch_windows" in values:
            windows = values["epoch_windows"]
            count, length, channels = windows.shape[:3]
            mask = presence.reshape(1, channels).expand(count, -1)
            channel_mask = values.get("epoch_channel_mask")
            if channel_mask is not None:
                mask = (
                    mask
                    & torch.as_tensor(
                        channel_mask[:, length // 2], device=mask.device
                    ).bool()
                )
            picks = select_recording_epochs(mask.any(dim=1), conditioner.samples)
            # Read only selected support, including for lazy window providers.
            wave = (
                torch.stack(
                    [
                        torch.as_tensor(windows[int(i)])[length // 2]
                        for i in picks.tolist()
                    ]
                )
                if picks.numel()
                else torch.empty(0, channels, 3840)
            )
        else:
            data = values["data"]
            samples = int(values["fs"]) * int(values["epoch_sec"])
            if samples != 3840:
                raise ValueError(
                    "Recording conditioning requires 128 Hz, 30-second epochs"
                )
            channels, total = data.shape
            count = total // samples
            mask = presence.reshape(1, channels).expand(count, -1)
            channel_mask = values.get("epoch_channel_valid")
            if channel_mask is not None:
                mask = mask & torch.as_tensor(channel_mask, device=mask.device).bool()
            picks = select_recording_epochs(mask.any(dim=1), conditioner.samples)
            wave = (
                torch.stack(
                    [
                        torch.as_tensor(
                            data[:, int(i) * samples : (int(i) + 1) * samples]
                        )
                        for i in picks.tolist()
                    ]
                )
                if picks.numel()
                else torch.empty(0, channels, samples)
            )
        with pinned_recording_observations(model, wave, mask.index_select(0, picks)):
            return function(*args, **kwargs)

    return wrapped
