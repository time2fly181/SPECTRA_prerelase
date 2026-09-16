"""Normalization layers used by the context classifier."""

import torch
from torch import nn


class VariancePreservingRMSNorm(nn.Module):
    """RMSNorm without centering + explicit variance re-injection.

    Unlike standard LayerNorm which removes variance information during
    normalization, this module preserves variance as an explicit feature.
    This is critical for N2/N3 discrimination where delta power variance
    is a key distinguishing characteristic.

    Why this combination:
    1. No centering -> preserves relative feature magnitudes (N2 transients vs N3 sustained)
    2. RMS scaling -> numerical stability without destroying variance ratio
    3. Variance side channel -> explicit signal the model can learn to use

    Args:
        dim: Feature dimension to normalize
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        # Variance pathway - projects scalar variance back to feature space
        self.var_proj = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
        )
        # Initialize gate small for conservative initial contribution
        self.var_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute variance BEFORE normalization (this is what we want to preserve)
        var = x.var(dim=-1, keepdim=True)  # [B, ..., 1]

        # RMSNorm (no mean centering)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = (x / rms) * self.weight

        # Re-inject variance as learned embedding
        var_embed = self.var_proj(var)  # [B, ..., dim]

        return normalized + self.var_gate * var_embed
