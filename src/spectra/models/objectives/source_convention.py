"""Checkpoint-only storage of legacy source offsets; never applied to predictions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class SourceConvention(nn.Module):
    """Allocate N2/N3 probability mass according to a source scoring offset.

    Offsets are centered with equal weight per source. This defines a source
    barycenter, not a clinically anchored reference policy. The observation
    model must be called explicitly by training code; inference uses the base
    classifier without a source offset.

    Args:
        source_names: Stable, ordered names of at least two annotation sources.
        penalty_weight: Nonnegative weight on the mean squared raw offsets.
    """

    def __init__(
        self, source_names: Sequence[str], penalty_weight: float = 0.01
    ) -> None:
        super().__init__()
        if isinstance(source_names, (str, bytes)):
            raise ValueError("source_names must be a sequence of source names")
        names = tuple(source_names)
        if (
            len(names) < 2
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "source_names must contain at least two unique nonempty names"
            )
        if not math.isfinite(penalty_weight) or penalty_weight < 0:
            raise ValueError("penalty_weight must be finite and nonnegative")
        self.source_names = names
        self.penalty_weight = float(penalty_weight)
        self.raw_offsets = nn.Parameter(torch.zeros(len(names)))

    def get_extra_state(self) -> dict[str, object]:
        """Serialize the ordered source vocabulary and objective configuration."""
        return {
            "source_names": self.source_names,
            "penalty_weight": self.penalty_weight,
        }

    def set_extra_state(self, state: object) -> None:
        """Reject checkpoint loads that silently change source semantics."""
        if (
            not isinstance(state, dict)
            or state.get("source_names") != self.source_names
        ):
            raise ValueError(
                "Checkpoint source_names do not match ordered source_names"
            )
        if state.get("penalty_weight") != self.penalty_weight:
            raise ValueError("Checkpoint penalty_weight does not match penalty_weight")
