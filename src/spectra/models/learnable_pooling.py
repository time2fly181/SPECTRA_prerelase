"""Learned temporal pooling for the multirate encoder."""

import math

import torch
from torch import nn


class LatentQueryAttentionPool(nn.Module):
    """Legacy latent-query pool used by existing CNN checkpoints.

    Both keys and values are projected from the normalized sequence, and the
    output projection consumes only the concatenated query summaries. This
    topology and behavior are intentionally retained for checkpoint fidelity.
    """

    def __init__(
        self,
        channels: int,
        out_dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channels < 1 or num_heads < 1:
            raise ValueError("channels and num_heads must be positive")
        self.channels = channels
        self.out_dim = out_dim or channels
        self.num_heads = num_heads

        self.sequence_norm = nn.LayerNorm(channels)
        self.latent_queries = nn.Parameter(torch.randn(num_heads, channels) * 0.02)
        self.key_proj = nn.Linear(channels, channels, bias=False)
        self.value_proj = nn.Linear(channels, channels, bias=False)
        self.project = nn.Linear(num_heads * channels, self.out_dim)
        self.norm = nn.LayerNorm(self.out_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._last_attn_weights: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.project.weight)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)

    def _attention(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = keys.size(0)
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        scores = torch.einsum("bkc,btc->bkt", queries, keys) / math.sqrt(self.channels)
        weights = torch.softmax(scores.clamp(min=-50.0, max=50.0), dim=-1)
        self._last_attn_weights = weights.detach()
        pooled = torch.einsum("bkt,btc->bkc", weights, values)
        return pooled, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        x_norm = self.sequence_norm(x_t)
        keys = self.key_proj(x_norm)
        values = self.value_proj(x_norm)
        pooled, _ = self._attention(keys, values)
        pooled = pooled.reshape(x.size(0), self.num_heads * self.channels)
        return self.drop(self.norm(self.project(pooled)))

    def get_attention_stats(self) -> dict[str, float] | None:
        """Return the legacy per-query entropy and maximum attention metrics."""
        if self._last_attn_weights is None:
            return None
        stats: dict[str, float] = {}
        for head_idx in range(self.num_heads):
            weights = self._last_attn_weights[:, head_idx, :]
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
            stats[f"head_{head_idx}_entropy"] = float(entropy.item())
            stats[f"head_{head_idx}_max_attn"] = float(
                weights.max(dim=-1).values.mean().item()
            )
        return stats
