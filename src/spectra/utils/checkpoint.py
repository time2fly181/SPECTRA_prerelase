from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import torch

_ENGINEERED_REQUIRED_KEYS = (
    "global_mean",
    "global_std",
    "stats_initialized",
    "band_weights",
    # DEPRECATED (removed, kept for backward compat documentation):
    # "alpha_global_baseline" - replaced with context-window relative detection
    # "alpha_global_count" - replaced with context-window relative detection
    # "n1_theta_alpha_center" - removed in orthogonal N1 redesign
    # "n1_ratio_bandwidth" - removed in orthogonal N1 redesign
)

_N1_REQUIRED_KEYS = (
    "n1_global_mean",
    "n1_global_std",
    "n1_stats_initialized",
    # DEPRECATED (removed in orthogonal N1 redesign, kept for backward compat loading):
    # "n1_theta_alpha_center",
    # "n1_ratio_bandwidth",
)

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


def detect_model_prefix(model: Any) -> str | None:
    """Detect if model uses a wrapper prefix (e.g., '_orig_mod.' from torch.compile).

    Args:
        model: The model to check

    Returns:
        The detected prefix string, or None if no prefix is detected.
    """
    if model is None or not hasattr(model, "state_dict"):
        return None

    try:
        model_state = model.state_dict()
    except Exception:
        return None

    if not model_state:
        return None

    # Check first few keys for prefix patterns
    for key in list(model_state.keys())[:10]:
        if not isinstance(key, str):
            continue
        for prefix in _STATE_KEY_NORMALIZE_PREFIXES:
            if key.startswith(prefix):
                return prefix

    return None


def add_prefix_to_state_dict_keys(
    state_dict: Mapping[str, Any], prefix: str
) -> Mapping[str, Any]:
    """Add a prefix to all keys in a state dict.

    This is the inverse of normalize_state_dict_keys() - used when loading
    a normalized checkpoint into a compiled model that expects prefixed keys.

    Args:
        state_dict: State dict with unprefixed keys
        prefix: Prefix to add (e.g., '_orig_mod.')

    Returns:
        New state dict with prefixed keys
    """
    if not isinstance(state_dict, Mapping) or not prefix:
        return state_dict

    prefixed = cast(
        "dict[str, Any]",
        state_dict.__class__() if hasattr(state_dict, "__class__") else {},
    )
    for key, value in state_dict.items():
        if isinstance(key, str):
            prefixed[prefix + key] = value
        else:
            prefixed[key] = value
    return prefixed


def align_state_dict_to_model(
    state_dict: Mapping[str, Any], model: Any
) -> Mapping[str, Any]:
    """Align checkpoint state dict keys to match model's expected format.

    Handles the case where checkpoint was saved from a non-compiled model
    (no prefix) but is being loaded into a compiled model (has _orig_mod. prefix),
    or vice versa.

    Args:
        state_dict: Normalized state dict (prefixes already stripped)
        model: Target model to load into

    Returns:
        State dict with keys aligned to match model's state dict format
    """
    if not isinstance(state_dict, Mapping) or model is None:
        return state_dict

    model_prefix = detect_model_prefix(model)

    if model_prefix is None:
        # Model has no prefix, state dict is already normalized - good to go
        return state_dict

    # Model has a prefix (e.g., compiled model) - add prefix to state dict keys
    logger = logging.getLogger(__name__)
    logger.info(
        f"[CHECKPOINT LOAD] Target model uses '{model_prefix}' prefix (likely torch.compile). "
        "Adding prefix to checkpoint keys for compatibility."
    )
    return add_prefix_to_state_dict_keys(state_dict, model_prefix)


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


def migrate_feat_projection_keys(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """Migrate old nn.Sequential feat_projection keys to new BottleneckProjection structure.

    Old structure (nn.Sequential):
        feat_projection.0.weight  -> Linear(in, out)
        feat_projection.0.bias
        feat_projection.1.weight  -> LayerNorm
        feat_projection.1.bias

    New structure (BottleneckProjection):
        feat_projection.bottleneck.0.weight  -> Linear(in, hidden)
        feat_projection.bottleneck.0.bias
        feat_projection.bottleneck.1.weight  -> LayerNorm(hidden)
        feat_projection.bottleneck.1.bias
        feat_projection.bottleneck.4.weight  -> Linear(hidden, out)
        feat_projection.bottleneck.4.bias
        feat_projection.residual_proj.weight -> Linear(in, out) if in != out
        feat_projection.norm.weight          -> LayerNorm(out)
        feat_projection.norm.bias

    Since the architectures are fundamentally different (old has 1 linear, new has 2),
    we cannot directly migrate weights. Instead, we:
    1. Detect old-style keys
    2. Remove them (they will be re-initialized)
    3. Log a warning about the architecture change

    Returns:
        Modified state_dict with old feat_projection keys removed for clean re-init.
    """
    if not isinstance(state_dict, Mapping):
        return state_dict

    logger = logging.getLogger(__name__)

    # Detect old-style feat_projection keys (nn.Sequential pattern: .0., .1., .2., .3.)
    old_style_keys = []
    new_style_keys = []

    for key in state_dict.keys():
        if not isinstance(key, str):
            continue
        if "feat_projection." not in key:
            continue

        # Extract the part after feat_projection.
        suffix = key.split("feat_projection.")[-1]

        # Old style: starts with digit (e.g., "0.weight", "1.bias")
        # New style: starts with named submodule (e.g., "bottleneck.", "residual_proj.", "norm.")
        if suffix and suffix[0].isdigit():
            old_style_keys.append(key)
        elif suffix.startswith(("bottleneck.", "residual_proj.", "norm.")):
            new_style_keys.append(key)

    # If we have old-style keys but no new-style keys, this is an old checkpoint
    if old_style_keys and not new_style_keys:
        logger.warning("=" * 80)
        logger.warning("⚠️  CHECKPOINT MIGRATION: feat_projection architecture changed")
        logger.warning("=" * 80)
        logger.warning(
            "This checkpoint uses the old nn.Sequential feat_projection structure.\n"
            "The current model uses BottleneckProjection with a different architecture:\n"
            "  - Old: Single Linear layer (in_dim → out_dim)\n"
            "  - New: Bottleneck with expansion (in_dim → hidden → out_dim) + residual\n"
            "\nThe old feat_projection weights will be DISCARDED and re-initialized.\n"
            "This is safe but means the projection layer needs to be re-trained."
        )
        logger.warning(
            f"Removing {len(old_style_keys)} old-style feat_projection keys:"
        )
        for key in old_style_keys[:5]:
            logger.warning(f"  - {key}")
        if len(old_style_keys) > 5:
            logger.warning(f"  ... and {len(old_style_keys) - 5} more")
        logger.warning("=" * 80)

        # Create new state dict without old keys
        migrated = cast(
            "dict[str, Any]",
            state_dict.__class__() if hasattr(state_dict, "__class__") else {},
        )
        for key, value in state_dict.items():
            if key not in old_style_keys:
                migrated[key] = value
        return migrated

    return state_dict


def resize_feat_projection_weights(
    state_dict: Mapping[str, Any],
    model: Any,
    allow_padding: bool = True,
) -> Mapping[str, Any]:
    """Resize feat_projection weight matrices when engineered feature dimension changes.

    When a checkpoint was trained with N engineered features but the current model
    expects M features (M > N), we can either pad or fail depending on allow_padding.

    Affected layers:
        feat_projection.bottleneck.0.weight  -> Linear(in_features, hidden)
        feat_projection.residual_proj.weight -> Linear(in_features, out_features)
        sleepfm_fusion.feature_fusion.input_projections.engineered.0.weight -> Linear(eng_dim, output_dim)
        n1_attention.n1_query.weight -> Linear(n1_feature_dim, d_model)

    For weight matrices of shape (out, in), we pad the 'in' dimension (dim=1)
    with small random values (Xavier-like initialization).

    Args:
        state_dict: Checkpoint state dict (already normalized and migrated)
        model: Target model to get expected shapes from
        allow_padding: If False, raise error instead of padding (default: True)

    Returns:
        Modified state_dict with resized feat_projection weights.

    Raises:
        ValueError: If allow_padding=False and feature count mismatch detected
    """
    if not isinstance(state_dict, Mapping) or model is None:
        return state_dict

    logger = logging.getLogger(__name__)

    # Get model state dict to compare shapes
    try:
        model_state = model.state_dict()
    except Exception:
        return state_dict

    # Suffixes for input-dimension sensitive Linear weights.
    # For weight matrices with shape (out_features, in_features), we resize
    # the input dimension (dim=1) when engineered/patch/N1 feature dims change.
    resize_suffixes = [
        # Existing engineered feature projection paths.
        "feat_projection.bottleneck.0.weight",
        "feat_projection.residual_proj.weight",
        "sleepfm_fusion.feature_fusion.input_projections.engineered.0.weight",
        # N1 feature projection path.
        "n1_attention.n1_query.weight",
        # MoE gate input projection (router input width can change with eng_router_proj).
        "classifier.gate.0.weight",
    ]

    # Resolve concrete keys from the current model/state dict using suffix matching.
    resize_keys = [
        key
        for key in state_dict.keys()
        if key in model_state
        and any(key.endswith(suffix) for suffix in resize_suffixes)
    ]

    new_state_dict: dict[str, Any] | None = None

    for key in resize_keys:
        if key not in state_dict or key not in model_state:
            continue

        ckpt_tensor = state_dict[key]
        model_tensor = model_state[key]

        if not isinstance(ckpt_tensor, torch.Tensor) or not isinstance(
            model_tensor, torch.Tensor
        ):
            continue

        if ckpt_tensor.shape == model_tensor.shape:
            continue

        # Check if this is an input dimension mismatch (dim=1 for weight matrices)
        # Weight shape: (out_features, in_features)
        if ckpt_tensor.ndim != 2 or model_tensor.ndim != 2:
            continue

        ckpt_out, ckpt_in = ckpt_tensor.shape
        model_out, model_in = model_tensor.shape

        # Output dimension must match; only input dimension can differ
        if ckpt_out != model_out:
            logger.warning(
                f"[CHECKPOINT] Cannot resize {key}: output dimension mismatch "
                f"(checkpoint={ckpt_out}, model={model_out})"
            )
            continue

        if ckpt_in == model_in:
            continue

        if new_state_dict is None:
            # Create mutable copy of state dict
            new_state_dict = dict(state_dict)

        if ckpt_in < model_in:
            # Checkpoint has fewer features -> pad or error
            pad_size = model_in - ckpt_in

            if not allow_padding:
                raise ValueError(
                    f"Feature count mismatch for {key}: checkpoint has {ckpt_in} features "
                    f"but model expects {model_in} features. Padding is disabled. "
                    f"Either:\n"
                    f"  1. Configure the model to use {ckpt_in} features (recommended)\n"
                    f"  2. Enable padding with allow_padding=True (may cause degradation)"
                )

            # Xavier uniform initialization for the padded weights
            # fan_in = model_in, fan_out = model_out
            # std = sqrt(2 / (fan_in + fan_out))
            std = (2.0 / (model_in + model_out)) ** 0.5
            padding = (
                torch.randn(
                    model_out,
                    pad_size,
                    device=ckpt_tensor.device,
                    dtype=ckpt_tensor.dtype,
                )
                * std
            )

            new_tensor = torch.cat([ckpt_tensor, padding], dim=1)
            new_state_dict[key] = new_tensor

            logger.warning(
                f"[CHECKPOINT RESIZE] Padded {key} from shape {tuple(ckpt_tensor.shape)} "
                f"to {tuple(new_tensor.shape)} (added {pad_size} input features with Xavier init)"
            )
        else:
            # Checkpoint has MORE features -> truncate (unusual but handle gracefully)
            new_tensor = ckpt_tensor[:, :model_in]
            new_state_dict[key] = new_tensor

            logger.warning(
                f"[CHECKPOINT RESIZE] Truncated {key} from shape {tuple(ckpt_tensor.shape)} "
                f"to {tuple(new_tensor.shape)} (removed {ckpt_in - model_in} input features)"
            )

    # Also handle bias if present (though bias doesn't depend on input features)
    # The bias corresponds to out_features, so it shouldn't need resizing
    # But let's verify no bias mismatches exist

    return new_state_dict if new_state_dict is not None else state_dict
