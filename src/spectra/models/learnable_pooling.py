"""Legacy-compatible and enhanced learnable temporal pooling modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.compiler.disable decorator for operations incompatible with Triton
# (e.g., the sort() used for median in TemporalStatisticsPooling).
try:
    torch_compile_disable = torch.compiler.disable
except AttributeError:
    # Fallback for older PyTorch versions
    try:
        torch_compile_disable = torch._dynamo.disable
    except AttributeError:
        # No-op decorator if neither available
        def torch_compile_disable(fn):
            return fn


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


class EnhancedLatentQueryAttentionPool(LatentQueryAttentionPool):
    """Latent-query pool with raw values and appended temporal statistics.

    Normalized feature directions determine attention through the keys. Values
    retain raw feature magnitudes, and temporal mean and standard deviation are
    appended before the output projection. This is a distinct checkpoint
    topology from :class:`LatentQueryAttentionPool`.
    """

    def __init__(
        self,
        channels: int,
        out_dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(channels, out_dim, num_heads, dropout)
        self.project = nn.Linear((num_heads + 2) * channels, self.out_dim)
        nn.init.xavier_uniform_(self.project.weight)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        x_norm = self.sequence_norm(x_t)
        # key_proj and value_proj are bias-free linear maps and nothing
        # nonlinear sits between projection and aggregation, so both matmuls
        # reassociate exactly: q·(x_norm·Wk^T)^T == (q·Wk)·x_norm^T and
        # w·(x_t·Wv^T) == (w·x_t)·Wv^T. Projecting the num_heads latent
        # queries and pooled summaries instead of every temporal position
        # avoids materializing the full [B, T, C] key and value tensors.
        effective_queries = self.latent_queries @ self.key_proj.weight
        scores = torch.einsum("kc,btc->bkt", effective_queries, x_norm) / math.sqrt(
            self.channels
        )
        weights = torch.softmax(scores.clamp(min=-50.0, max=50.0), dim=-1)
        self._last_attn_weights = weights.detach()
        raw_pooled = torch.einsum("bkt,btc->bkc", weights, x_t)
        pooled = self.value_proj(raw_pooled)
        pooled = pooled.reshape(x.size(0), self.num_heads * self.channels)
        mean = x.mean(dim=-1)
        std = x.var(dim=-1, unbiased=False).add(1e-5).sqrt()
        pooled = torch.cat((pooled, mean, std), dim=1)
        return self.drop(self.norm(self.project(pooled)))

    def get_attention_stats(self) -> dict[str, float] | None:
        """Return concentration, normalized entropy, and collapse diagnostics."""
        if self._last_attn_weights is None:
            return None
        weights = self._last_attn_weights
        stats: dict[str, float] = {}
        for head_idx in range(self.num_heads):
            head_weights = weights[:, head_idx, :]
            entropy = (
                -(head_weights * torch.log(head_weights + 1e-8)).sum(dim=-1).mean()
            )
            entropy_value = float(entropy.item())
            time_len = head_weights.size(-1)
            normalized_entropy = (
                entropy_value / math.log(time_len) if time_len > 1 else 0.0
            )
            stats[f"head_{head_idx}_entropy"] = entropy_value
            stats[f"head_{head_idx}_normalized_entropy"] = normalized_entropy
            stats[f"head_{head_idx}_max_attn"] = float(
                head_weights.max(dim=-1).values.mean().item()
            )
        normalized = F.normalize(weights, p=2, dim=-1)
        similarity = torch.einsum("bqt,brt->bqr", normalized, normalized)
        off_diagonal = ~torch.eye(
            self.num_heads, dtype=torch.bool, device=weights.device
        )
        mean_similarity = (
            similarity[:, off_diagonal].mean().item() if self.num_heads > 1 else 0.0
        )
        stats["mean_off_diagonal_attention_cosine_similarity"] = float(mean_similarity)
        return stats


class TemporalStatisticsPooling(nn.Module):
    """
    Multi-statistic temporal pooling that preserves temporal information.

    Instead of just mean pooling, compute:
    - Mean: Overall activity level
    - Std: Variability/burstiness (spindles vs continuous slow waves)
    - Max: Peak amplitude (K-complexes, artifacts)
    - Quantiles: Distribution shape (skewness proxy)

    This captures temporal dynamics that get lost in standard average pooling.
    """

    def __init__(self, in_features: int, out_features: int | None = None):
        super().__init__()
        self.in_features = in_features
        # 4 statistics: mean, std, max, median
        self.num_stats = 4
        self.out_features = out_features or in_features

        # Project the concatenated statistics to desired dimension
        self.project = nn.Sequential(
            nn.Linear(in_features * self.num_stats, self.out_features),
            nn.LayerNorm(self.out_features),
            nn.GELU(),
        )

    @torch_compile_disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input features
        Returns:
            [B, out_features] pooled features

        Note: This method is excluded from torch.compile because the sort()
        operation for median computation causes Triton compilation failures.
        """
        # Compute statistics along temporal dimension
        mean = x.mean(dim=-1)  # [B, C]
        std = x.std(dim=-1).clamp(min=1e-6)  # [B, C]
        max_val = x.max(dim=-1)[0]  # [B, C]

        # Median via sorting (differentiable approximation)
        # NOTE: sort() is incompatible with Triton, hence @torch_compile_disable
        sorted_x, _ = x.sort(dim=-1)
        T = x.size(-1)
        median = sorted_x[:, :, T // 2]  # [B, C]

        # Concatenate statistics: [B, C*4]
        stats = torch.cat([mean, std, max_val, median], dim=-1)

        return self.project(stats)
