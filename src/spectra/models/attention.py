# spectra.models/attention.py
"""
Attention mechanisms with Flash Attention support.

This module uses PyTorch's F.scaled_dot_product_attention which automatically
selects the best backend:
- Flash Attention 2 (when available, ~2-4x faster, less memory)
- xFormers memory-efficient attention
- Standard math implementation (fallback)

The attention backend can be controlled via the torch.nn.attention.sdpa_kernel context manager.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Scaled dot-product attention with Flash Attention support.

    This uses PyTorch's F.scaled_dot_product_attention which automatically
    selects the best backend (Flash Attention, xFormers, or math).

    Benefits:
    - 2-4x faster than manual implementation
    - 50-70% less memory usage
    - Automatic kernel selection based on hardware
    - Optional attention-weight inspection without changing the forward path

    Args:
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads (must divide hidden_dim evenly)
        dropout: Dropout probability for attention weights
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """
        Args:
            x: [B, L, H] sequence features
            mask: [B, L] optional padding mask (True/1 = valid, False/0 = masked out)
            return_attention: Whether to return attention weights

        Returns:
            output: [B, L, H] attended features
            attention_weights: [B, num_heads, L, L] (if return_attention=True)
        """
        B, L, H = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [B, num_heads, L, head_dim]

        # Prepare attention mask for SDPA
        # SDPA expects: values where True/finite = attend, False/-inf = mask out
        attn_mask = None
        if mask is not None:
            # Convert boolean/binary mask to attention mask format
            # Input mask: True/1 = valid, False/0 = masked
            # SDPA mask: True/1 = valid, False/0 = masked (same convention)
            if mask.dtype == torch.bool:
                # Boolean mask: expand to [B, 1, 1, L] for broadcasting
                attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            else:
                # Numeric mask (0/1): convert to boolean
                attn_mask = (mask > 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            # SDPA will broadcast this to [B, num_heads, L, L]

        # Always use optimized SDPA (Flash Attention / xFormers / efficient kernels)
        # for the actual context computation so requesting attention weights does not
        # change the train/eval forward path.
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,  # Bidirectional attention for sleep staging
        )

        attn_weights = None
        if return_attention:
            # Compute weights separately for inspection/visualization only.
            with torch.no_grad():
                scale = math.sqrt(self.head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale

                if attn_mask is not None:
                    # Invert mask for scores: True = keep, False = mask out (-inf)
                    scores = scores.masked_fill(~attn_mask, float("-inf"))

                # CRITICAL FP16 FIX: Clamp scores before softmax to prevent overflow
                # FP16 exp() overflows at ~11.09 (exp(11.09) ≈ 65504 = FP16 max)
                # Clamp to [-10, 10] is safe for both FP16 and BF16
                scores = torch.clamp(scores, min=-10.0, max=10.0)

                attn_weights = F.softmax(scores, dim=-1)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(B, L, H)
        output = self.out_proj(context)

        if return_attention:
            return output, attn_weights
        return output


class SpecializedTemporalAttention(nn.Module):
    """
    Multi-head temporal attention with head specialization priors for sleep staging.

    Encourages each head to focus on stage-specific frequency patterns via:
    - Learnable frequency preference biases for queries
    - Auxiliary specialization loss that aligns head activation with sleep stages
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if num_heads != 4:
            raise ValueError("SpecializedTemporalAttention expects exactly 4 heads")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout_p = dropout

        # Head-specific frequency preference vectors (learnable priors)
        # Each head gets a bias vector of size head_dim
        self.head_freq_preference = nn.Parameter(
            torch.randn(num_heads, self.head_dim) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_head_specialization: bool = False,
    ):
        """
        Args:
            x: [B, L, H] sequence features
            mask: [B, L] optional padding mask
            return_attention: Whether to return attention weights
            return_head_specialization: Whether to return specialization metrics

        Returns:
            output: [B, L, H] attended features
            attention_weights: [B, num_heads, L, L] (optional)
            head_spec: Dict with specialization metrics (optional)
        """
        B, L, H = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Bias queries with head-specific frequency preferences
        freq_bias = self.head_freq_preference.view(1, self.num_heads, 1, self.head_dim)
        q = q + 0.1 * freq_bias

        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
            else:
                attn_mask = (mask > 0).unsqueeze(1).unsqueeze(2)

        head_spec: dict[str, torch.Tensor] | None = None

        # CRITICAL FIX: Always use Flash Attention for consistent train/eval behavior
        # The context computation must be identical in both modes to avoid train/eval gap
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        # Compute attention weights separately (under no_grad) only when needed
        # for visualization or head specialization metrics - this doesn't affect gradients
        attn_weights = None
        if return_attention or return_head_specialization:
            with torch.no_grad():
                scale = math.sqrt(self.head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale

                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))

                # CRITICAL FP16 FIX: Clamp scores before softmax to prevent overflow
                scores = torch.clamp(scores, min=-10.0, max=10.0)

                attn_weights = F.softmax(scores, dim=-1)

            if return_head_specialization:
                head_spec = self._compute_head_specialization(attn_weights)

        context = context.transpose(1, 2).contiguous().view(B, L, H)
        output = self.out_proj(context)

        if return_head_specialization:
            return output, attn_weights, head_spec
        if return_attention:
            return output, attn_weights
        return output

    def _compute_head_specialization(
        self, attn_weights: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Compute diversity metrics that describe how specialized each head is.

        Args:
            attn_weights: [B, num_heads, L, L]

        Returns:
            Dict with entropy and concentration tensors (both [num_heads])
        """
        avg_attn = attn_weights.mean(dim=(0, 2))  # [num_heads, L]
        eps = 1e-8
        entropy = -(avg_attn * torch.log(avg_attn + eps)).sum(dim=-1)

        head_std = attn_weights.std(dim=-1)  # [B, num_heads, L]
        concentration = head_std.mean(dim=0).mean(dim=-1)  # [num_heads]

        return {
            "entropy": entropy.detach(),
            "concentration": concentration.detach(),
        }


class CrossAttention(nn.Module):
    """
    Cross-attention with Flash Attention support.
    Fuses CNN features (query) with engineered features (key/value).

    This allows the model to selectively attend to relevant domain-knowledge
    features based on the learned CNN representations.

    Args:
        query_dim: Dimension of query features (CNN features)
        key_value_dim: Dimension of key/value features (engineered features)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError(
                f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_value_dim, query_dim)
        self.v_proj = nn.Linear(key_value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.dropout_p = dropout

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Args:
            query: [B, L, H_q] - CNN features
            key_value: [B, L, H_kv] - Engineered features
            mask: [B, L] - optional mask (True/1 = valid, False/0 = masked)

        Returns:
            output: [B, L, H_q] - Enhanced CNN features
        """
        B, L, _ = query.shape

        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(key_value)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(key_value)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Prepare mask
        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
            else:
                attn_mask = (mask > 0).unsqueeze(1).unsqueeze(2)

        # Use Flash Attention
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        output = self.out_proj(context)

        return output


class SelfAttentionPooling(nn.Module):
    """
    Attention-based pooling to aggregate sequence into fixed-size representation.

    This is a simple attention mechanism that learns which positions in the
    sequence are most important for the task, producing a weighted average.

    No Flash Attention needed here - this is a simple, efficient operation.

    Args:
        hidden_dim: Hidden dimension of input features
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x: [B, L, H] sequence
            mask: [B, L] optional mask (True/1 = valid, False/0 = masked)

        Returns:
            pooled: [B, H] attention-weighted pooling
            weights: [B, L] attention weights
        """
        scores = self.attention(x).squeeze(-1)  # [B, L]

        if mask is not None:
            # Mask out invalid positions
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # [B, L]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, H]

        return pooled, weights
