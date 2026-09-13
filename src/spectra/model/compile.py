"""torch.compile wrapper for PSG models.

Handles all edge cases around torch.compile: compile-incompatible modules,
reduce-overhead + grad accumulation conflict, dynamic shapes, fullgraph +
N1 attention conflict, CUDA-only enforcement, learned fusion tuning, etc.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CompileResult:
    """Return value from :func:`maybe_compile_model`."""

    model: nn.Module
    compiled: bool
    compile_mode: str | None


@dataclass(frozen=True)
class InductorConfig:
    """Resolved TorchInductor configuration for ``torch.compile``.

    Returned by :func:`configure_inductor_for_mode` with compile kwargs
    ready to be unpacked into ``torch.compile(**cfg.compile_kwargs)``.
    """

    compile_kwargs: dict[str, Any] = field(default_factory=dict)


def _env_enabled(name: str) -> bool:
    """Interpret common environment variable truthy values."""
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _set_inductor_flag(path: str, value: bool | int) -> bool:
    """Best-effort setter for TorchInductor flags with version-safe fallbacks."""
    inductor_mod = getattr(torch, "_inductor", None)
    if inductor_mod is None:
        return False
    config = getattr(inductor_mod, "config", None)
    if config is None:
        return False

    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            return False
        target = getattr(target, part)

    leaf = parts[-1]
    if not hasattr(target, leaf):
        return False

    try:
        setattr(target, leaf, value)
        return True
    except Exception:
        return False


def _configure_learned_fusion(log_fn: Callable[[str], None]) -> None:
    """Enable coordinate descent tuning for learned kernel fusion decisions.

    Sets inductor flags that allow the autotuner to profile different
    fusion strategies rather than relying on heuristics alone.

    Args:
        log_fn: Callable for logging messages.
    """
    ok = _set_inductor_flag("coordinate_descent_tuning", True)
    if ok:
        log_fn("[INFO] Enabled coordinate_descent_tuning for learned fusion.")
    else:
        log_fn(
            "[WARN] coordinate_descent_tuning not available " "in this PyTorch version."
        )

    ok_bf = _set_inductor_flag("benchmark_fusion", True)
    if ok_bf:
        log_fn("[INFO] Enabled benchmark_fusion for empirical fusion selection.")


def configure_inductor_for_mode(
    compile_mode: str,
    *,
    fullgraph: bool = False,
    accum_steps: int = 1,
    enable_learned_fusion: bool = False,
    multi_forward_per_step: bool = False,
    scheduled_trainability_changes: bool = False,
    logger: Any = None,
) -> InductorConfig:
    """Configure TorchInductor flags and build compile kwargs for a mode.

    This is the shared configuration entry point used by both the main
    training compile path (:func:`maybe_compile_model`) and pretrain
    scripts (e.g. supervised contrastive pretraining).

    Args:
        compile_mode: One of ``"default"``, ``"reduce-overhead"``,
            ``"max-autotune"``, ``"max-autotune-no-cudagraphs"``.
        fullgraph: Whether to enforce full graph compilation.
        accum_steps: Gradient accumulation steps (for reduce-overhead
            conflict detection).
        enable_learned_fusion: When ``True`` and mode is
            ``max-autotune*``, enable coordinate descent tuning and
            benchmark fusion for learned kernel fusion decisions.
        multi_forward_per_step: Set ``True`` when the compiled module is
            replayed more than once before each ``backward()`` (e.g.
            two-view contrastive pretraining encodes ``x`` and ``x_aug``
            through the same compiled encoder). CUDA graphs store
            activations in static buffers that are only valid until the
            next replay, so a second forward overwrites saved-for-backward
            activations from the first, tripping autograd's version
            counter. When set, cudagraph-using modes are remapped to their
            cudagraph-free equivalents (``reduce-overhead`` -> ``default``,
            ``max-autotune`` -> ``max-autotune-no-cudagraphs``).
        scheduled_trainability_changes: Set ``True`` when parameters will be
            frozen or unfrozen during training. Retracing after a
            ``requires_grad`` transition can leave earlier CUDA-graph pools
            reserved, so cudagraph-using modes are remapped to their
            cudagraph-free equivalents.
        logger: Optional logger with ``.info()`` / ``.warning()``
            methods; falls back to ``print()`` if ``None``.

    Returns:
        An :class:`InductorConfig` with resolved compile kwargs ready
        to be unpacked into ``torch.compile(**cfg.compile_kwargs)``.
    """

    def _log(msg: str) -> None:
        if logger is not None:
            # Strip bracket prefix for logger (it adds its own level)
            stripped = msg
            for prefix in ("[INFO] ", "[WARN] "):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    break
            if msg.startswith("[WARN]"):
                logger.warning(stripped)
            else:
                logger.info(stripped)
        else:
            print(msg)

    # Modules replayed multiple times before a single backward (e.g. two-view
    # contrastive pretraining) cannot use CUDA graphs: the static replay buffers
    # get overwritten by the second forward, corrupting the first forward's
    # saved-for-backward activations. Remap cudagraph-using modes to their
    # cudagraph-free equivalents before the mode dispatch below.
    if multi_forward_per_step:
        if compile_mode == "reduce-overhead":
            _log(
                "[WARN] reduce-overhead mode uses CUDA graphs which conflict "
                "with modules replayed multiple times per backward "
                "(e.g. two-view contrastive pretraining)"
            )
            _log(
                "[INFO] Automatically switching to 'default' compile mode "
                "for stability"
            )
            compile_mode = "default"
        elif compile_mode == "max-autotune":
            _log(
                "[WARN] max-autotune uses CUDA graphs which conflict with "
                "modules replayed multiple times per backward; switching to "
                "'max-autotune-no-cudagraphs'"
            )
            compile_mode = "max-autotune-no-cudagraphs"

    if scheduled_trainability_changes:
        if compile_mode == "reduce-overhead":
            _log(
                "[WARN] reduce-overhead uses CUDA graphs whose memory pools can "
                "remain reserved across scheduled freeze/unfreeze retracing"
            )
            _log(
                "[INFO] Automatically switching to 'default' compile mode for "
                "scheduled trainability changes"
            )
            compile_mode = "default"
        elif compile_mode == "max-autotune":
            _log(
                "[WARN] max-autotune uses CUDA graphs whose memory pools can "
                "remain reserved across scheduled freeze/unfreeze retracing; "
                "switching to 'max-autotune-no-cudagraphs'"
            )
            compile_mode = "max-autotune-no-cudagraphs"

    compile_kwargs: dict[str, Any] = {"mode": compile_mode}
    disable_cudagraph_trees = False
    disable_cudagraphs = False

    if fullgraph:
        compile_kwargs["fullgraph"] = True

    # ---- mode-specific constraints ----
    if compile_mode == "max-autotune":
        _log("[WARN] max-autotune mode selected: expect SLOW first evaluation!")
        _log(
            "[WARN] Triton will exhaustively search for optimal kernels "
            "(can take 5-10 minutes)"
        )
        compile_kwargs["dynamic"] = False
        if enable_learned_fusion:
            _configure_learned_fusion(_log)

    elif compile_mode == "max-autotune-no-cudagraphs":
        _log(
            "[INFO] max-autotune-no-cudagraphs selected: "
            "disabling Inductor CUDA graphs for stability."
        )
        # Preserve the native no-cudagraphs mode. Rewriting this to
        # ``max-autotune`` is unsafe: that mode explicitly enables
        # ``triton.cudagraphs`` and can override the global flag below when
        # torch.compile applies its mode options. This matters for modules
        # invoked multiple times before backward, such as two-view SupCon.
        compile_kwargs["mode"] = "max-autotune-no-cudagraphs"
        compile_kwargs["dynamic"] = False
        disable_cudagraph_trees = True
        disable_cudagraphs = True
        if enable_learned_fusion:
            _configure_learned_fusion(_log)

    elif compile_mode == "reduce-overhead":
        if accum_steps > 1:
            _log(
                f"[WARN] reduce-overhead mode uses CUDA graphs which conflict "
                f"with gradient accumulation (accum={accum_steps})"
            )
            _log(
                "[INFO] Automatically switching to 'default' compile mode "
                "for stability"
            )
            compile_kwargs["mode"] = "default"
        compile_kwargs["dynamic"] = False
        # Work around runtime failures like:
        # "graph recording observed an input tensor deallocate ... replay".
        # Set PSGSTAGE_ENABLE_CUDAGRAPH_TREES=1 to opt back in.
        disable_cudagraph_trees = not _env_enabled("PSGSTAGE_ENABLE_CUDAGRAPH_TREES")

    else:  # default mode
        compile_kwargs["dynamic"] = False

    # ---- apply CUDA graph flags ----
    if disable_cudagraphs:
        updated = _set_inductor_flag("triton.cudagraphs", False) or _set_inductor_flag(
            "cudagraphs", False
        )
        if updated:
            _log("[INFO] Disabled TorchInductor CUDA graphs.")

    if disable_cudagraph_trees:
        updated = _set_inductor_flag(
            "triton.cudagraph_trees", False
        ) or _set_inductor_flag("cudagraph_trees", False)
        if updated:
            _log(
                "[INFO] Disabled TorchInductor CUDA graph trees "
                "for training stability."
            )

    return InductorConfig(compile_kwargs=compile_kwargs)


def _module_supports_torch_compile(module: nn.Module) -> bool:
    """Best-effort check that traverses submodules for compile opt-out flags."""
    attr = getattr(module, "supports_torch_compile", None)
    if attr is not None:
        return bool(attr)

    checker = getattr(module, "is_torch_compile_safe", None)
    if callable(checker):
        try:
            result = checker()
        except TypeError:
            result = checker(module)  # legacy signature support
        if result is not None:
            return bool(result)

    for child in module.children():
        if not _module_supports_torch_compile(child):
            return False
    return True


def maybe_compile_model(
    model: nn.Module,
    *,
    torch_compile: bool,
    compile_mode: str = "default",
    compile_fullgraph: bool = False,
    accum_steps: int = 1,
    device: str = "cuda",
    uses_n1_attention: bool = False,
    scheduled_trainability_changes: bool = False,
) -> CompileResult:
    """Apply ``torch.compile`` to *model* if requested and supported.

    Returns a :class:`CompileResult` with the (possibly compiled) model and
    metadata about what happened.
    """
    if not torch_compile:
        return CompileResult(model=model, compiled=False, compile_mode=None)

    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        print("[WARN] torch.compile not available in this PyTorch version; skipping.")
        return CompileResult(model=model, compiled=False, compile_mode=None)

    if not _module_supports_torch_compile(model):
        print(
            "[WARN] torch.compile requested but model contains "
            "compile-incompatible modules; skipping compile."
        )
        return CompileResult(model=model, compiled=False, compile_mode=None)

    # Enable capture of scalar outputs to avoid graph breaks from .item()
    try:
        torch._dynamo.config.capture_scalar_outputs = True
    except Exception as exc:
        print(f"[WARN] Failed to enable capture_scalar_outputs: {exc}")

    # Handle fullgraph + N1 attention conflict
    resolved_fullgraph = compile_fullgraph
    if compile_fullgraph:
        if uses_n1_attention:
            print("[WARN] --compile_fullgraph is incompatible with --use_n1_attention")
            print(
                "[WARN] N1 feature extraction uses STFT operations that "
                "cannot be compiled"
            )
            print("[WARN] Disabling fullgraph mode; using partial compilation instead")
            resolved_fullgraph = False
        else:
            print("[INFO] fullgraph=True enabled: no graph breaks allowed")

    from spectra.model.recording_conditioning import get_recording_conditioner

    conditioned = get_recording_conditioner(model) is not None
    if conditioned:
        # Cache lookup and varying recording groups use an eager boundary.
        resolved_fullgraph = False
        compile_mode = {
            "reduce-overhead": "default",
            "max-autotune": "max-autotune-no-cudagraphs",
        }.get(compile_mode, compile_mode)

    # Configure inductor via shared helper
    inductor_cfg = configure_inductor_for_mode(
        compile_mode,
        fullgraph=resolved_fullgraph,
        accum_steps=accum_steps,
        scheduled_trainability_changes=scheduled_trainability_changes,
    )
    compile_kwargs = inductor_cfg.compile_kwargs

    # Only compile on CUDA
    if isinstance(device, str) and device.startswith("cuda"):
        print(
            f"[INFO] Compiling model with torch.compile "
            f"(mode={compile_kwargs.get('mode', compile_mode)})..."
        )
        print("[INFO] First epoch will be slower due to compilation overhead")
        try:
            model = torch.compile(model, **compile_kwargs)  # type: ignore[assignment]
            print("[INFO] Model compilation successful")
            return CompileResult(
                model=model,
                compiled=True,
                compile_mode=compile_kwargs.get("mode", compile_mode),
            )
        except Exception as exc:
            print(
                f"[WARN] torch.compile failed: {exc}. "
                "Continuing without compilation."
            )
            return CompileResult(model=model, compiled=False, compile_mode=None)
    else:
        print("[INFO] torch.compile is only beneficial on CUDA; skipping on CPU")
        return CompileResult(model=model, compiled=False, compile_mode=None)
