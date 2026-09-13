"""
SleepFM-Inspired Modules for Sleep Staging
==========================================

Implements temporal attention pooling and attention-based feature fusion
based on the SleepFM paper (Thapa et al., Nature Medicine 2026).

Key adaptations for epoch-level sleep staging:
- Temporal pooling biased toward center epoch (classification target)
- Multi-source feature fusion via attention pooling
- Optional learned positional weighting for context windows
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Temporal Attention Pooling
# =============================================================================


class TemporalAttentionPool(nn.Module):
    """
    Attention-based temporal pooling for context windows.

    Instead of mean pooling or just using the center token, this learns
    which epochs in the context window are most informative for classifying
    the center epoch.

    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads (if using multi-head variant)
        dropout: Dropout probability
        center_bias: How to bias attention toward center epoch
            - None: No bias, pure learned attention
            - 'learned': Learnable position-dependent bias
            - 'gaussian': Fixed Gaussian centered on middle position
            - 'linear': Linear decay from center
        center_bias_strength: Initial/fixed strength of center bias
        temperature: Softmax temperature (lower = sharper attention)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        center_bias: Literal["learned", "gaussian", "linear"] | None = "gaussian",
        center_bias_strength: float = 2.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.center_bias = center_bias
        self.center_bias_strength = center_bias_strength
        self.temperature = temperature

        # Attention score computation
        if num_heads == 1:
            # Simple single-head attention
            self.attention_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.LayerNorm(embed_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 4, 1),
            )
        else:
            # Multi-head attention pooling with learned query
            self.query = nn.Parameter(torch.randn(1, num_heads, embed_dim // num_heads))
            self.key_proj = nn.Linear(embed_dim, embed_dim)
            self.value_proj = nn.Linear(embed_dim, embed_dim)
            self.head_dim = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout)

        # Learnable center bias (if using learned variant)
        if center_bias == "learned":
            # Modulates Gaussian sigma: exp(position_bias) * (seq_len/4)
            self.position_bias = nn.Parameter(torch.zeros(1))
            self.bias_scale = nn.Parameter(torch.tensor(center_bias_strength))

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

    def _compute_position_bias(
        self,
        seq_len: int,
        center_idx: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute position-dependent attention bias."""
        if self.center_bias is None:
            return torch.zeros(seq_len, device=device)

        if center_idx is None:
            center_idx = seq_len // 2

        positions = torch.arange(seq_len, device=device, dtype=torch.float)
        distances = (positions - center_idx).abs()

        bias: torch.Tensor
        if self.center_bias == "gaussian":
            # Gaussian decay from center
            sigma = seq_len / 4  # ~95% weight within half the window
            bias = torch.exp(-0.5 * (distances / sigma) ** 2)
            bias = bias * self.center_bias_strength

        elif self.center_bias == "linear":
            # Linear decay from center
            max_dist = distances.max()
            bias = 1 - (distances / max_dist)
            bias = bias * self.center_bias_strength

        elif self.center_bias == "learned":
            # Learnable Gaussian-like bias with learnable width.
            # position_bias modulates sigma: exp(0)=1 preserves default width,
            # positive values widen the window, negative values narrow it.
            sigma = (seq_len / 4) * torch.exp(self.position_bias)
            base_bias = torch.exp(-0.5 * (distances / sigma) ** 2)
            bias = base_bias * self.bias_scale
        else:
            bias = torch.zeros(seq_len, device=device)

        return bias

    def forward(
        self,
        x: torch.Tensor,
        center_idx: int | None = None,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: Input tensor of shape (B, seq_len, embed_dim)
            center_idx: Index of center epoch (default: seq_len // 2)
            mask: Optional boolean mask, True for positions to ignore
            return_weights: Whether to return attention weights

        Returns:
            pooled: Pooled representation (B, embed_dim)
            weights: Attention weights (B, seq_len) if return_weights=True
        """
        B, S, D = x.shape

        # Initialize values for multi-head path
        values: torch.Tensor | None = None

        if self.num_heads == 1:
            # Single-head attention pooling
            attn_logits = self.attention_net(x).squeeze(-1)  # (B, S)
        else:
            # Multi-head attention pooling
            # Project to keys and values
            keys = self.key_proj(x).view(B, S, self.num_heads, self.head_dim)
            values = self.value_proj(x).view(B, S, self.num_heads, self.head_dim)

            # Expand query for batch
            query = self.query.expand(B, -1, -1)  # (B, num_heads, head_dim)

            # Compute attention scores
            attn_logits = torch.einsum(
                "bhd,bshd->bhs", query, keys
            )  # (B, num_heads, S)
            attn_logits = attn_logits / math.sqrt(self.head_dim)
            attn_logits = attn_logits.mean(dim=1)  # Average over heads: (B, S)

        # Apply temperature
        attn_logits = attn_logits / self.temperature

        # Add position bias
        position_bias = self._compute_position_bias(S, center_idx, device=x.device)
        attn_logits = attn_logits + position_bias.unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        # Compute attention weights
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, S)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to get pooled representation
        if self.num_heads == 1:
            pooled = torch.einsum("bs,bsd->bd", attn_weights, x)
        else:
            # For multi-head, apply to values then project
            assert values is not None, "values not computed for multi-head attention"
            values_flat = values.view(B, S, D)
            pooled = torch.einsum("bs,bsd->bd", attn_weights, values_flat)

        # Output projection with residual from center token
        pooled = self.output_proj(pooled)

        # Optional: add residual connection from center token
        if center_idx is not None:
            center_token = x[:, center_idx, :]
            pooled = self.output_norm(pooled + center_token)
        else:
            pooled = self.output_norm(pooled)

        if return_weights:
            return pooled, attn_weights
        return pooled, None


class HierarchicalTemporalPool(nn.Module):
    """
    Two-stage temporal pooling for very long context windows.

    First pools local neighborhoods, then pools across the coarser sequence.
    Useful if you want to extend context beyond 21-31 epochs without
    quadratic attention cost.

    Args:
        embed_dim: Dimension of embeddings
        local_window: Size of local pooling window
        num_heads: Attention heads for global pooling
    """

    def __init__(
        self,
        embed_dim: int,
        local_window: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.local_window = local_window

        # Local pooling (within neighborhoods)
        self.local_pool = TemporalAttentionPool(
            embed_dim=embed_dim,
            num_heads=1,
            dropout=dropout,
            center_bias="gaussian",
            center_bias_strength=1.0,
        )

        # Global pooling (across neighborhoods)
        self.global_pool = TemporalAttentionPool(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            center_bias="gaussian",
            center_bias_strength=2.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        center_idx: int | None = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            x: (B, seq_len, embed_dim)
            center_idx: Index of center epoch

        Returns:
            pooled: (B, embed_dim)
            weights: Dict with 'local' and 'global' weights if requested
        """
        B, S, D = x.shape

        if center_idx is None:
            center_idx = S // 2

        # Pad sequence to be divisible by local_window
        pad_len = (self.local_window - S % self.local_window) % self.local_window
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), value=0)

        S_padded = x.shape[1]
        num_windows = S_padded // self.local_window

        # Reshape into local windows
        x_windowed = x.view(B, num_windows, self.local_window, D)

        # Pool each local window
        local_pooled = []
        local_weights = []
        for i in range(num_windows):
            window = x_windowed[:, i, :, :]  # (B, local_window, D)
            pooled, weights = self.local_pool(
                window, center_idx=self.local_window // 2, return_weights=return_weights
            )
            local_pooled.append(pooled)
            if return_weights:
                local_weights.append(weights)

        # Stack local pooled representations
        local_pooled = torch.stack(local_pooled, dim=1)  # (B, num_windows, D)

        # Global pooling across windows
        global_center = center_idx // self.local_window
        pooled, global_weights = self.global_pool(
            local_pooled, center_idx=global_center, return_weights=return_weights
        )

        if return_weights:
            weights = {
                "local": torch.stack(local_weights, dim=1) if local_weights else None,
                "global": global_weights,
            }
            return pooled, weights

        return pooled, None


# =============================================================================
# Attention-Based Feature Fusion
# =============================================================================


class AttentionFeatureFusion(nn.Module):
    """
    Fuses multiple feature sources using attention pooling.

    Instead of concatenation or cross-attention, treats each feature source
    as a "token" and uses attention to learn optimal weighting.

    This is inspired by SleepFM's multimodal fusion but adapted for
    CNN features + engineered features fusion.

    Args:
        feature_dims: Dict mapping feature source names to their dimensions
        output_dim: Dimension of fused output
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_learned_query: If True, use a learned query vector for pooling
                          If False, use mean of inputs as query
    """

    def __init__(
        self,
        feature_dims: dict,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_learned_query: bool = True,
    ):
        super().__init__()
        self.feature_names = list(feature_dims.keys())
        self.num_sources = len(feature_dims)
        self.output_dim = output_dim
        self.use_learned_query = use_learned_query

        # Project each feature source to common dimension
        self.input_projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for name, dim in feature_dims.items()
            }
        )

        # Learnable source embeddings (like positional embeddings but for feature sources)
        self.source_embeddings = nn.Parameter(
            torch.randn(self.num_sources, output_dim) * 0.02
        )
        self.source_prior_logits = nn.Parameter(torch.zeros(self.num_sources))

        # Attention pooling
        if use_learned_query:
            self.query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

        # Gate for residual connection (optional)
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid(),
        )

        # Cached source weights from the most recent forward pass.
        # These are consumed by entropy regularization/logging.
        self._last_source_weights: torch.Tensor | None = None

    def _resize_source_tensor(
        self,
        ckpt_tensor: torch.Tensor,
        current_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Pad or truncate source-sized tensors for checkpoint compatibility."""
        if ckpt_tensor.shape == current_tensor.shape:
            return ckpt_tensor
        if ckpt_tensor.ndim != current_tensor.ndim:
            return current_tensor.detach().clone()
        if ckpt_tensor.shape[1:] != current_tensor.shape[1:]:
            return current_tensor.detach().clone()

        ckpt_sources = ckpt_tensor.shape[0]
        current_sources = current_tensor.shape[0]
        if ckpt_sources < current_sources:
            pad = current_tensor[ckpt_sources:].detach().clone()
            return torch.cat([ckpt_tensor, pad], dim=0)
        return ckpt_tensor[:current_sources]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Handle 2-source->3-source checkpoint migration and new parameters."""
        source_keys = ("source_embeddings", "source_prior_logits")
        current_state = self.state_dict()

        for key in source_keys:
            full_key = f"{prefix}{key}"
            if full_key in state_dict:
                state_dict[full_key] = self._resize_source_tensor(
                    state_dict[full_key],
                    current_state[key],
                )

        # Fill missing projection weights for newly added sources (e.g., YASA).
        for key, value in current_state.items():
            full_key = f"{prefix}{key}"
            if full_key in state_dict:
                continue
            if key.startswith("input_projections.yasa."):
                fallback_key = key.replace(
                    "input_projections.yasa.",
                    "input_projections.engineered.",
                    1,
                )
                fallback_full_key = f"{prefix}{fallback_key}"
                if (
                    fallback_full_key in state_dict
                    and state_dict[fallback_full_key].shape == value.shape
                ):
                    state_dict[full_key] = (
                        state_dict[fallback_full_key].detach().clone()
                    )
                    continue
            state_dict[full_key] = value.detach().clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def get_source_prior_weights(self) -> torch.Tensor:
        """Return normalized global source priors."""
        return torch.softmax(self.source_prior_logits, dim=-1)

    def compute_source_entropy_floor_loss(
        self,
        min_entropy_ratio: float,
    ) -> torch.Tensor:
        """Penalize source-collapse when entropy falls below a floor."""
        if self._last_source_weights is None:
            return self.source_prior_logits.new_zeros(())

        if min_entropy_ratio <= 0:
            return self.source_prior_logits.new_zeros(())

        weights = torch.clamp(self._last_source_weights.float(), min=1e-8)
        entropy = -torch.sum(weights * torch.log(weights), dim=-1)
        max_entropy = math.log(max(self.num_sources, 1))
        target_entropy = max_entropy * float(min_entropy_ratio)
        penalty = torch.relu(entropy.new_tensor(target_entropy) - entropy)
        return penalty.mean().to(self.source_prior_logits.dtype)

    def forward(
        self,
        features: dict,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            features: Dict mapping feature source names to tensors
                     Each tensor should be (B, seq_len, feature_dim) or (B, feature_dim)
            return_weights: Whether to return attention weights

        Returns:
            fused: Fused features (B, seq_len, output_dim) or (B, output_dim)
            weights: Attention weights over feature sources if requested
        """
        # Determine if we have sequence dimension
        sample_feat = features[self.feature_names[0]]
        has_sequence = sample_feat.dim() == 3

        if has_sequence:
            B, S, _ = sample_feat.shape
        else:
            B = sample_feat.shape[0]
            S = 1
            # Add sequence dimension for uniform processing
            features = {k: v.unsqueeze(1) for k, v in features.items()}

        # Project all features to common dimension and stack
        projected = []
        for i, name in enumerate(self.feature_names):
            feat = features[name]  # (B, S, feat_dim)
            proj = self.input_projections[name](feat)  # (B, S, output_dim)
            # Add source embedding
            proj = proj + self.source_embeddings[i].unsqueeze(0).unsqueeze(0)
            projected.append(proj)

        # Stack along a new "source" dimension
        # (B, S, num_sources, output_dim)
        stacked = torch.stack(projected, dim=2)

        # Reshape for attention: treat each (sample, timestep) independently
        # (B * S, num_sources, output_dim)
        stacked_flat = stacked.view(B * S, self.num_sources, self.output_dim)

        # Compute attention query
        if self.use_learned_query:
            query = self.query.expand(B * S, -1, -1)  # (B*S, 1, output_dim)
        else:
            # Use mean as query
            query = stacked_flat.mean(dim=1, keepdim=True)  # (B*S, 1, output_dim)

        # Dynamic attention from query-key interaction
        attn_output, attn_weights = self.attention(
            query=query,
            key=stacked_flat,
            value=stacked_flat,
            need_weights=True,
        )
        if attn_weights is None:
            raise RuntimeError("Expected attention weights from MultiheadAttention")
        if attn_weights.dim() == 4:
            # [B*S, H, 1, N] -> [B*S, N]
            attn_weights = attn_weights.mean(dim=1).squeeze(1)
        elif attn_weights.dim() == 3:
            # [B*S, 1, N] -> [B*S, N]
            attn_weights = attn_weights.squeeze(1)
        elif attn_weights.dim() != 2:
            raise RuntimeError(
                f"Unexpected attention weight shape: {tuple(attn_weights.shape)}"
            )

        # Combine dynamic attention with a learnable global prior.
        prior = self.get_source_prior_weights().to(attn_weights.dtype).unsqueeze(0)
        combined_logits = torch.log(attn_weights + 1e-8) + torch.log(prior + 1e-8)
        source_weights_flat = torch.softmax(combined_logits, dim=-1)
        fused_prior = torch.einsum("bn,bnd->bd", source_weights_flat, stacked_flat)
        fused_dynamic = attn_output.squeeze(1)
        fused = 0.5 * (fused_dynamic + fused_prior)

        # Output projection
        fused = self.output_proj(fused)

        # Gated residual: blend attention-fused output with primary (CNN) features.
        # gate ≈ 1 → use fused representation; gate ≈ 0 → fall back to CNN only.
        cnn_residual = projected[0].reshape(B * S, self.output_dim)
        gate = self.fusion_gate(torch.cat([fused, cnn_residual], dim=-1))
        fused = gate * fused + (1 - gate) * cnn_residual

        # Reshape back
        if has_sequence:
            fused = fused.view(B, S, self.output_dim)
            source_weights = source_weights_flat.view(B, S, self.num_sources)
        else:
            fused = fused.view(B, self.output_dim)
            source_weights = source_weights_flat.view(B, self.num_sources)

        self._last_source_weights = source_weights

        if return_weights:
            return fused, source_weights
        return fused, None


class GatedFeatureFusion(nn.Module):
    """
    Alternative fusion using gating mechanism.

    Learns element-wise gates for each feature source, allowing
    fine-grained control over which features contribute to each
    dimension of the output.

    Args:
        feature_dims: Dict mapping feature source names to their dimensions
        output_dim: Dimension of fused output
        dropout: Dropout probability
    """

    def __init__(
        self,
        feature_dims: dict,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_names = list(feature_dims.keys())
        self.num_sources = len(feature_dims)
        self.output_dim = output_dim

        # Project each feature source to common dimension
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.LayerNorm(output_dim),
                )
                for name, dim in feature_dims.items()
            }
        )

        # Gate networks for each source
        total_dim = sum(feature_dims.values())
        self.gate_networks = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(total_dim, output_dim),
                    nn.Sigmoid(),
                )
                for name in feature_dims.keys()
            }
        )

        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: dict,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            features: Dict mapping feature source names to tensors
            return_weights: Whether to return gate values

        Returns:
            fused: Fused features
            gates: Dict of gate values for each source if requested
        """
        # Concatenate all features for gate computation
        all_features = torch.cat(
            [features[name] for name in self.feature_names], dim=-1
        )

        # Compute gates and weighted projections
        fused = 0
        gates = {}

        for name in self.feature_names:
            # Project features
            proj = self.projections[name](features[name])

            # Compute gate
            gate = self.gate_networks[name](all_features)
            gates[name] = gate

            # Apply gate
            fused = fused + gate * proj

        fused = self.output_norm(fused)
        fused = self.dropout(fused)

        if return_weights:
            return fused, gates
        return fused, None


# =============================================================================
# Combined Module: Temporal + Feature Fusion
# =============================================================================


class EpochFeatureFusion(nn.Module):
    """Per-epoch feature fusion without temporal pooling.

    Applies attention-based fusion of CNN/transformer features with engineered
    sleep features at each epoch independently, returning the full sequence.
    Unlike SleepFMInspiredFusion, does NOT pool across the context window -
    the downstream model handles center extraction or full-sequence processing.

    This is designed for use with sleepfm_mode='fuse_only', where the fused
    [B, L, d_model] sequence flows directly to the classifier.

    Args:
        cnn_dim: Dimension of CNN/transformer epoch embeddings.
        engineered_dim: Dimension of engineered features (core, excluding YASA).
        output_dim: Final output dimension (typically d_model).
        yasa_dim: Dimension of optional YASA feature branch (0 to disable).
        num_heads: Attention heads for fusion.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        cnn_dim: int,
        engineered_dim: int,
        output_dim: int,
        yasa_dim: int = 0,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.yasa_dim = max(0, int(yasa_dim))

        feature_dims: dict[str, int] = {
            "cnn": cnn_dim,
            "engineered": engineered_dim,
        }
        if self.yasa_dim > 0:
            feature_dims["yasa"] = self.yasa_dim

        self.feature_fusion = AttentionFeatureFusion(
            feature_dims=feature_dims,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
        )

        self._last_feature_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        cnn_features: torch.Tensor,
        engineered_features: torch.Tensor,
        yasa_features: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            cnn_features: (B, seq_len, cnn_dim) transformer output.
            engineered_features: (B, seq_len, engineered_dim) core eng features.
            yasa_features: (B, seq_len, yasa_dim) optional YASA features.
            return_weights: Whether to return fusion attention weights.

        Returns:
            fused: (B, seq_len, output_dim) fused features, full sequence.
            weights: Dict with 'feature' weights if requested, else None.
        """
        source_features: dict[str, torch.Tensor] = {
            "cnn": torch.nan_to_num(cnn_features, nan=0.0, posinf=0.0, neginf=0.0),
            "engineered": torch.nan_to_num(
                engineered_features,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
        }
        if self.yasa_dim > 0:
            if yasa_features is None:
                raise ValueError(
                    "EpochFeatureFusion initialized with yasa_dim > 0 "
                    "but no yasa_features provided."
                )
            source_features["yasa"] = torch.nan_to_num(
                yasa_features, nan=0.0, posinf=0.0, neginf=0.0
            )

        fused, feature_weights = self.feature_fusion(
            source_features,
            return_weights=True,
        )  # (B, seq_len, output_dim)
        self._last_feature_attention_weights = feature_weights

        fused = self.output_proj(fused)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

        if return_weights:
            return fused, {"feature": feature_weights}
        return fused, None


class SleepFMInspiredFusion(nn.Module):
    """
    Complete fusion module combining temporal attention pooling
    and multi-source feature fusion.

    Designed to replace sequential cross-attention with SleepFM-style
    attention pooling for sleep staging.

    Args:
        cnn_dim: Dimension of CNN epoch embeddings
        engineered_dim: Dimension of engineered features
        output_dim: Final output dimension
        yasa_dim: Dimension of optional YASA feature branch
        num_heads: Attention heads for fusion
        dropout: Dropout probability
        temporal_pool_type: 'attention' or 'hierarchical'
        center_bias: Bias type for temporal pooling ('gaussian', 'linear', 'learned', None)
        center_bias_strength: Strength of center bias (higher = more focus on center)
        temperature: Softmax temperature for attention (lower = sharper)
    """

    def __init__(
        self,
        cnn_dim: int,
        engineered_dim: int,
        output_dim: int,
        yasa_dim: int = 0,
        num_heads: int = 4,
        dropout: float = 0.1,
        temporal_pool_type: str = "attention",
        center_bias: Literal["learned", "gaussian", "linear"] | None = "gaussian",
        center_bias_strength: float = 2.0,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.yasa_dim = max(0, int(yasa_dim))

        # Feature fusion (per-timestep)
        feature_dims = {
            "cnn": cnn_dim,
            "engineered": engineered_dim,
        }
        if self.yasa_dim > 0:
            feature_dims["yasa"] = self.yasa_dim

        self.feature_fusion = AttentionFeatureFusion(
            feature_dims=feature_dims,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Temporal pooling (across context window)
        if temporal_pool_type == "attention":
            self.temporal_pool = TemporalAttentionPool(
                embed_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout,
                center_bias=center_bias,
                center_bias_strength=center_bias_strength,
                temperature=temperature,
            )
        elif temporal_pool_type == "hierarchical":
            self.temporal_pool = HierarchicalTemporalPool(
                embed_dim=output_dim,
                local_window=5,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown temporal_pool_type: {temporal_pool_type}")

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._last_feature_attention_weights: torch.Tensor | None = None
        self._last_temporal_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        cnn_features: torch.Tensor,
        engineered_features: torch.Tensor,
        yasa_features: torch.Tensor | None = None,
        center_idx: int | None = None,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            cnn_features: (B, seq_len, cnn_dim)
            engineered_features: (B, seq_len, engineered_dim)
            center_idx: Index of center epoch for classification
            return_weights: Whether to return attention weights

        Returns:
            output: (B, output_dim) - representation for center epoch
            weights: Dict with 'feature' and 'temporal' weights if requested
        """
        source_features = {
            "cnn": cnn_features,
            "engineered": engineered_features,
        }
        if self.yasa_dim > 0:
            if yasa_features is None:
                raise ValueError(
                    "SleepFMInspiredFusion was initialized with yasa_dim > 0 but "
                    "no yasa_features were provided."
                )
            source_features["yasa"] = yasa_features

        # Fuse features at each timestep
        fused, feature_weights = self.feature_fusion(
            source_features,
            return_weights=True,
        )  # (B, seq_len, output_dim)
        self._last_feature_attention_weights = feature_weights

        # Pool temporally with center bias
        pooled, temporal_weights = self.temporal_pool(
            fused,
            center_idx=center_idx,
            return_weights=True,
        )  # (B, output_dim)
        self._last_temporal_attention_weights = temporal_weights

        # Final projection
        output = self.output_proj(pooled)

        if return_weights:
            weights = {
                "feature": feature_weights,
                "temporal": temporal_weights,
            }
            return output, weights

        return output, None


# =============================================================================
# Integration Example
# =============================================================================


class ExampleSleepStager(nn.Module):
    """
    Example of how to integrate these modules into a sleep staging model.

    This is a simplified example - you would replace your existing
    sequential feature fusion with SleepFMInspiredFusion.
    """

    def __init__(
        self,
        cnn_encoder: nn.Module,
        feature_extractor: nn.Module,
        cnn_dim: int = 256,
        engineered_dim: int = 28,
        hidden_dim: int = 256,
        num_classes: int = 5,
        context_size: int = 21,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_size = context_size
        self.center_idx = context_size // 2

        # Epoch encoder (your existing CNN)
        self.cnn_encoder = cnn_encoder

        # Engineered feature extractor (your existing YASA-style features)
        self.feature_extractor = feature_extractor

        # Transformer for temporal context
        self.input_proj = nn.Linear(cnn_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # SleepFM-inspired fusion
        center_bias_typed: Literal["learned", "gaussian", "linear"] = "gaussian"
        self.fusion = SleepFMInspiredFusion(
            cnn_dim=hidden_dim,
            engineered_dim=engineered_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            temporal_pool_type="attention",
            center_bias=center_bias_typed,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            x: Raw PSG signals (B, context_size, channels, samples_per_epoch)
            return_attention: Whether to return attention weights

        Returns:
            logits: Class logits (B, num_classes)
            attention: Attention weights dict if requested
        """
        B, S, C, T = x.shape

        # Encode each epoch with CNN
        x_flat = x.view(B * S, C, T)
        cnn_features = self.cnn_encoder(x_flat)  # (B*S, cnn_dim)
        cnn_features = cnn_features.view(B, S, -1)  # (B, S, cnn_dim)

        # Project and apply transformer
        cnn_features = self.input_proj(cnn_features)
        cnn_features = self.transformer(cnn_features)  # (B, S, hidden_dim)

        # Extract engineered features
        engineered = self.feature_extractor(x)  # (B, S, engineered_dim)

        # Fuse with temporal attention pooling
        fused, weights = self.fusion(
            cnn_features,
            engineered,
            center_idx=self.center_idx,
            return_weights=return_attention,
        )  # (B, hidden_dim)

        # Classify
        logits = self.classifier(fused)  # (B, num_classes)

        if return_attention:
            return logits, weights
        return logits, None


# =============================================================================
# Utility: Visualize Attention Weights
# =============================================================================


def visualize_attention(
    temporal_weights: torch.Tensor,
    feature_weights: torch.Tensor | None = None,
    feature_names: list | None = None,
    epoch_labels: list | None = None,
    figsize: tuple = (12, 4),
):
    """
    Visualize attention weights from the fusion module.

    Args:
        temporal_weights: (seq_len,) or (B, seq_len) attention over epochs
        feature_weights: (num_sources,) or (B, num_sources) attention over features
        feature_names: Names of feature sources
        epoch_labels: Labels for epochs (e.g., sleep stages)
    """
    import matplotlib.pyplot as plt

    # Handle batch dimension
    if temporal_weights.dim() > 1:
        temporal_weights = temporal_weights[0]  # Take first sample
    if feature_weights is not None and feature_weights.dim() > 1:
        feature_weights = feature_weights[0]

    temporal_weights_np = temporal_weights.detach().cpu().numpy()

    num_plots = 2 if feature_weights is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    # Temporal attention
    seq_len = len(temporal_weights_np)
    center = seq_len // 2
    x = range(seq_len)

    axes[0].bar(x, temporal_weights_np, color="steelblue", alpha=0.7)
    axes[0].axvline(x=center, color="red", linestyle="--", label="Center epoch")
    axes[0].set_xlabel("Epoch in context window")
    axes[0].set_ylabel("Attention weight")
    axes[0].set_title("Temporal Attention")
    axes[0].legend()

    if epoch_labels is not None:
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(epoch_labels, rotation=45, ha="right")

    # Feature attention
    if feature_weights is not None:
        feature_weights_np = feature_weights.detach().cpu().numpy()
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(feature_weights_np))]

        axes[1].bar(feature_names, feature_weights_np, color="coral", alpha=0.7)
        axes[1].set_ylabel("Attention weight")
        axes[1].set_title("Feature Source Attention")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Quick test
    B, S, D_cnn, D_eng = 4, 21, 256, 28

    # Test temporal pooling
    print("Testing TemporalAttentionPool...")
    pool = TemporalAttentionPool(embed_dim=D_cnn, center_bias="gaussian")
    x = torch.randn(B, S, D_cnn)
    pooled, weights = pool(x, center_idx=S // 2, return_weights=True)
    print(f"  Input: {x.shape} -> Pooled: {pooled.shape}")
    print(f"  Attention weights shape: {weights.shape}")
    print(
        f"  Center weight: {weights[0, S // 2]:.3f}, Edge weights: {weights[0, 0]:.3f}, {weights[0, -1]:.3f}"
    )

    # Test feature fusion
    print("\nTesting AttentionFeatureFusion...")
    fusion = AttentionFeatureFusion(
        feature_dims={"cnn": D_cnn, "engineered": D_eng},
        output_dim=D_cnn,
    )
    cnn_feat = torch.randn(B, S, D_cnn)
    eng_feat = torch.randn(B, S, D_eng)
    fused, weights = fusion(
        {"cnn": cnn_feat, "engineered": eng_feat}, return_weights=True
    )
    print(
        f"  CNN: {cnn_feat.shape}, Engineered: {eng_feat.shape} -> Fused: {fused.shape}"
    )
    print(f"  Feature attention shape: {weights.shape}")

    # Test combined module
    print("\nTesting SleepFMInspiredFusion...")
    combined = SleepFMInspiredFusion(
        cnn_dim=D_cnn,
        engineered_dim=D_eng,
        output_dim=D_cnn,
    )
    output, weights = combined(
        cnn_feat, eng_feat, center_idx=S // 2, return_weights=True
    )
    print(f"  Output shape: {output.shape}")
    print(f"  Temporal attention: {weights['temporal'].shape}")

    print("\nAll tests passed!")
