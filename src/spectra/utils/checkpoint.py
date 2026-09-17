"""Checkpoint key normalization and fixed-filter reconstruction adapters."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import torch

_STATE_KEY_NORMALIZE_PREFIXES = ("module.", "_orig_mod.", "orig_mod.")


def normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """Strip common wrapper prefixes (module./_orig_mod./orig_mod.) from state_dict keys.

    Torch DDP and torch.compile add wrapper prefixes to parameter/buffer names.
    When resuming on an unwrapped model these prefixes prevent successful loading.
    """
    if not isinstance(state_dict, Mapping):
        return state_dict

    needs_normalization = False
    for key in state_dict.keys():
        if not isinstance(key, str):
            continue
        if any(key.startswith(prefix) for prefix in _STATE_KEY_NORMALIZE_PREFIXES):
            needs_normalization = True
            break

    if not needs_normalization:
        return state_dict

    # Preserve OrderedDict when present; cast because the runtime class is mutable.
    normalized = cast("dict[str, Any]", state_dict.__class__())
    for key, value in state_dict.items():
        if not isinstance(key, str):
            normalized[key] = value
            continue
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in _STATE_KEY_NORMALIZE_PREFIXES:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    changed = True
        normalized[clean_key] = value
    return normalized


def restore_fixed_filter_buffers(
    model: Any, state_dict: Mapping[str, Any]
) -> list[str]:
    """Seed non-persistent Kaiser FIR buffers from a checkpoint that stored them.

    Checkpoints written before these filters became non-persistent buffers carry
    the exact coefficients used during training. Those keys otherwise land in
    ``unexpected_keys`` and are discarded, leaving the reconstructed model with
    filters rebuilt from the *current* tap-count defaults and the *current*
    cutoff convention. Both have changed, so discarding them silently swaps in a
    different low-pass response. Copying the stored coefficients back makes the
    reconstructed model numerically identical to the trained one, and does so
    without depending on how the coefficients are derived today.

    The restored buffers remain non-persistent, so a model that is re-saved after
    restoration falls back to formula-derived kernels. This is an inference-time
    reconstruction adapter, not a training-state migration.

    Args:
        model: Module to restore buffers into, modified in place.
        state_dict: Checkpoint state dict, with prefixes already normalized.

    Returns:
        Sorted module names whose ``kernel`` buffer was restored.
    """
    if model is None or not isinstance(state_dict, Mapping):
        return []

    from spectra.models.anti_alias import (
        KaiserAntiAliasDownsample1D,
        KaiserAntiAliasUpsample1D,
    )

    logger = logging.getLogger(__name__)
    restored: list[str] = []
    max_delta = 0.0

    for name, module in model.named_modules():
        if not isinstance(
            module, KaiserAntiAliasDownsample1D | KaiserAntiAliasUpsample1D
        ):
            continue

        current = getattr(module, "kernel", None)
        if not isinstance(current, torch.Tensor):
            continue

        key = f"{name}.kernel"
        saved = state_dict.get(key)
        if saved is None:
            saved = state_dict.get(f"model.{key}")
        if not isinstance(saved, torch.Tensor):
            continue

        if saved.ndim != current.ndim or saved.shape[0] != current.shape[0]:
            logger.warning(
                "Skipping fixed-filter restore for %s: checkpoint shape %s is not "
                "compatible with model shape %s",
                key,
                tuple(saved.shape),
                tuple(current.shape),
            )
            continue
        if not bool(torch.isfinite(saved).all()):
            logger.warning(
                "Skipping fixed-filter restore for %s: checkpoint kernel is not finite",
                key,
            )
            continue

        if saved.shape == current.shape:
            delta = float((saved.to(current.dtype) - current).abs().max())
            max_delta = max(max_delta, delta)
            with torch.no_grad():
                current.copy_(saved.to(dtype=current.dtype, device=current.device))
        else:
            # Tap count differs, so the buffer must be replaced rather than
            # copied into. Padding is derived from the tap count and has to move
            # with it or the filter would shift the signal.
            logger.warning(
                "Fixed-filter tap count mismatch for %s: checkpoint has %d taps, "
                "model built %d. Rebuilding the buffer from the checkpoint; the "
                "reconstructed tap count was likely mis-detected.",
                key,
                int(saved.shape[-1]),
                int(current.shape[-1]),
            )
            num_taps = int(saved.shape[-1])
            module.kernel = (
                saved.detach().clone().to(dtype=current.dtype, device=current.device)
            )
            module.num_taps = num_taps
            module.pad = (num_taps - 1) // 2
            max_delta = float("inf")

        restored.append(name)

    if restored:
        if max_delta == 0.0:
            logger.debug(
                "Restored %d fixed anti-alias filter buffers from checkpoint "
                "(identical to rebuilt filters)",
                len(restored),
            )
        else:
            logger.info(
                "Restored %d fixed anti-alias filter buffers from checkpoint "
                "(max coefficient delta vs rebuilt filters: %s). The checkpoint's "
                "own filters are authoritative.",
                len(restored),
                "tap-count change" if max_delta == float("inf") else f"{max_delta:.6f}",
            )

    return sorted(restored)
