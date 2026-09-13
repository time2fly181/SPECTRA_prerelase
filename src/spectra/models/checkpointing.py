"""Gradient-checkpointing helpers for stateful model layers."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_BATCH_NORM_OPS = frozenset(
    {
        torch.ops.aten.batch_norm.default,
        torch.ops.aten.cudnn_batch_norm.default,
        torch.ops.aten.miopen_batch_norm.default,
        torch.ops.aten.native_batch_norm.default,
        torch.ops.aten._native_batch_norm_legit.default,
        torch.ops.aten._native_batch_norm_legit_functional.default,
    }
)
_INPLACE_TENSOR_ADD = torch.ops.aten.add_.Tensor


class _BatchNormSafeCheckpointMode(TorchDispatchMode):
    """Prevent BatchNorm buffer mutations during checkpoint recomputation.

    Training-mode BatchNorm must still use minibatch statistics during the
    recomputed forward so gradients match the original forward. Only its state
    mutations are redirected: the batch counter increment becomes out-of-place,
    and running mean/variance updates target temporary clones.
    """

    def __init__(self, *, recomputing: bool) -> None:
        super().__init__()
        self.recomputing = recomputing

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        call_kwargs = kwargs or {}
        if not self.recomputing:
            return func(*args, **call_kwargs)

        if (
            func is _INPLACE_TENSOR_ADD
            and len(args) >= 2
            and isinstance(args[0], torch.Tensor)
            and args[0].ndim == 0
            and args[0].dtype == torch.long
        ):
            return torch.add(args[0], args[1], alpha=call_kwargs.get("alpha", 1))

        training = bool(args[5]) if len(args) >= 6 else False
        if func in _BATCH_NORM_OPS and training:
            safe_args = list(args)
            if safe_args[3] is not None:
                safe_args[3] = safe_args[3].clone()
            if safe_args[4] is not None:
                safe_args[4] = safe_args[4].clone()
            return func(*safe_args, **call_kwargs)

        return func(*args, **call_kwargs)


def batchnorm_safe_checkpoint_context() -> tuple[TorchDispatchMode, TorchDispatchMode]:
    """Return compile-compatible forward and recomputation contexts."""
    return (
        _BatchNormSafeCheckpointMode(recomputing=False),
        _BatchNormSafeCheckpointMode(recomputing=True),
    )
