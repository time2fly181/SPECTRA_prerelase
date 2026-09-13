"""
Practical Integration: Add Epoch Attention to Your Sequential Feature Fusion

This shows the REALISTIC integration path that works with a pooled epoch
encoder (which outputs pooled [B, D] features) by applying epoch
attention to the ENGINEERED FEATURES which have natural temporal/spectral
structure.

The key insight is that your engineered features (band powers, temporal
statistics) are computed from spectral frames - we can apply attention
over these frames before the final aggregation.

Position Encoding Support:
    When using epoch attention alongside ALiBi or other positional encodings
    in the transformer encoder, it's important to maintain consistency.
    This module supports matching position encoding strategies:
    - "alibi": Use ALiBi-style linear distance penalties (matches transformer)
    - "learned": Learn position biases (flexible but more parameters)
    - "none": Position-agnostic attention (original behavior)
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariancePreservingNorm(nn.Module):
    """
    Normalize without removing variance information.

    Standard LayerNorm removes variance information which can be problematic
    for N2 detection where delta variance is a key discriminative feature.
    This normalization re-injects variance information via a learned pathway.

    Args:
        dim: Feature dimension to normalize
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        # Learn to preserve variance as a feature (N2 has high delta variance)
        self.var_weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply variance-preserving normalization.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor with variance information preserved
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = (var + self.eps).sqrt()

        # Standard normalization
        x_norm = (x - mean) / std

        # Re-inject variance information (N2 has high delta variance)
        var_signal = torch.log1p(var) * self.var_weight  # [..., dim]

        return self.gamma * x_norm + self.beta + var_signal


class SpectralFrameAttention(nn.Module):
    """
    Attention over spectral frames for engineered feature extraction.

    Instead of simple averaging over STFT frames, this module learns
    which time-frequency frames are most relevant for each epoch.

    This is applied INSIDE your SleepFeatureExtractor to make the
    band power computation attention-weighted.

    Key benefits:
    - K-complexes: High attention on frames with sharp transients
    - Spindles: High attention on frames with 11-16 Hz bursts
    - Alpha dropout: Learn to weight frames where alpha disappears

    Args:
        num_frames: Expected number of spectral frames (T in STFT output)
        num_bands: Number of frequency bands
        hidden_dim: Hidden dimension for attention (default: 32)
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_frames: int = 60,  # ~60 frames for 30s epoch with 0.5s hop
        num_bands: int = 5,  # delta, theta, alpha, sigma, beta
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input: [B, num_frames, num_bands] - band powers per frame
        # Learn attention weights per frame
        self.frame_attention = nn.Sequential(
            nn.Linear(num_bands, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, band_powers_per_frame: torch.Tensor) -> torch.Tensor:
        """
        Attention-weighted aggregation over spectral frames.

        Args:
            band_powers_per_frame: [B, T, num_bands] band powers per frame

        Returns:
            weighted_powers: [B, num_bands] attention-weighted band powers
        """
        # Compute attention scores: [B, T, 1]
        attn_scores = self.frame_attention(band_powers_per_frame)
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T, 1]
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: [B, num_bands]
        weighted_powers = (band_powers_per_frame * attn_weights).sum(dim=1)

        return weighted_powers


class EnhancedSequentialFeatureFusion(nn.Module):
    """
    Your existing SequentialFeatureFusion + lightweight epoch attention.

    This version applies attention in a computationally efficient way:

    1. CNN features: Passed through as-is (already pooled by CNN)
    2. Engineered features: Enhanced with a learnable "epoch context"
       that modulates their importance before cross-attention

    The "epoch context" learns what combination of engineered features
    is most discriminative for each sleep stage, similar to how
    SleepTransformer's epoch attention learns what time-frequency
    patterns matter.

    Position Encoding:
        To maintain consistency with the transformer encoder's positional
        encoding (especially ALiBi), this module supports matching strategies:
        - "alibi": ALiBi-style linear distance penalties
        - "learned": Learnable position biases
        - "none": Position-agnostic (original behavior)

    DROP-IN REPLACEMENT for SequentialFeatureFusion.
    """

    def __init__(
        self,
        cnn_dim: int,
        feat_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_epoch_context: bool = True,
        num_context_heads: int = 4,
        position_encoding: str = "none",
        max_seq_len: int = 64,
        n2_feature_indices: list[int] | None = None,
        use_variance_preserving_norm: bool = True,
    ):
        super().__init__()

        self.cnn_dim = cnn_dim
        self.feat_dim = feat_dim
        self.use_epoch_context = use_epoch_context
        self.position_encoding = position_encoding
        self.max_seq_len = max_seq_len

        # === N2-aware attention bias ===
        # These indices should correspond to N2-discriminative features:
        # spindle_density, delta_continuity_var, sigma_power, delta_sigma_ratio
        self.n2_feature_indices = n2_feature_indices or []
        if self.n2_feature_indices:
            self.n2_attention_boost = nn.Parameter(
                torch.zeros(len(self.n2_feature_indices))
            )

        # === Epoch context attention for engineered features ===
        if use_epoch_context:
            # Multi-head attention where queries are learnable "stage detectors"
            # This learns what feature patterns indicate each stage
            self.num_context_heads = num_context_heads
            head_dim = feat_dim // num_context_heads
            # Store actual multi-head dimension (may be < feat_dim if not evenly divisible)
            self.head_dim = head_dim
            self.mh_dim = (
                num_context_heads * head_dim
            )  # Actual dimension used by attention

            # Learnable stage-specific queries (like SleepTransformer's context vectors)
            self.stage_queries = nn.Parameter(
                torch.randn(num_context_heads, head_dim) * 0.02
            )

            # Position embeddings for queries - makes each position's query unique
            # Critical for learning temporal context (otherwise all positions get identical queries)
            self.query_pos_embed = nn.Parameter(
                torch.randn(max_seq_len, num_context_heads, head_dim) * 0.02
            )

            # Center epoch marker - identifies which epoch we're classifying
            # This helps attention focus on the target epoch
            self.center_marker = nn.Parameter(torch.zeros(1, 1, feat_dim))
            nn.init.normal_(self.center_marker, std=0.02)

            # Project engineered features for attention (use mh_dim * 2 for k,v)
            self.feat_to_kv = nn.Linear(feat_dim, self.mh_dim * 2)

            # Position encoding components
            if position_encoding == "alibi":
                # ALiBi: fixed slopes per head (matches transformer encoder)
                slopes = self._get_alibi_slopes(num_context_heads)
                self.register_buffer("alibi_slopes", slopes, persistent=False)
            elif position_encoding == "learned":
                # Learned position biases (more flexible, more parameters)
                self.rel_pos_bias = nn.Parameter(
                    torch.zeros(num_context_heads, max_seq_len, max_seq_len) * 0.01
                )

            # Combine attention outputs (project from mh_dim back to feat_dim)
            self.context_out = nn.Linear(self.mh_dim, feat_dim)
            self.context_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.Sigmoid(),
            )
            self.context_norm = nn.LayerNorm(feat_dim)

        # === Sequential fusion components ===

        # Project engineered features to CNN dimension
        self.feat_proj = (
            nn.Linear(feat_dim, cnn_dim) if feat_dim != cnn_dim else nn.Identity()
        )

        # Cross-attention: CNN queries engineered features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cnn_dim,
            num_heads=num_heads,
            kdim=cnn_dim,
            vdim=cnn_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Gated residual
        # Bottleneck design to prevent parameter explosion while allowing
        # engineered features to directly influence gating decision
        bottleneck = cnn_dim // 2
        self.gate = nn.Sequential(
            nn.Linear(cnn_dim * 3, bottleneck),  # cnn + eng_proj + attn_out
            nn.GELU(),
            nn.Linear(bottleneck, cnn_dim),
            nn.Sigmoid(),
        )

        # Final normalization - use variance-preserving for N2 discrimination
        if use_variance_preserving_norm:
            self.norm = VariancePreservingNorm(cnn_dim)
        else:
            self.norm = nn.LayerNorm(cnn_dim)

    @staticmethod
    def _get_alibi_slopes(num_heads: int) -> torch.Tensor:
        """
        Compute ALiBi slopes matching the transformer encoder implementation.

        From the ALiBi paper: slopes are powers of 2^(-8/n) for n heads.
        """

        def get_slopes(n: int) -> list[float]:
            def get_power_of_2_slopes(k: int) -> list[float]:
                start = 2.0 ** (-(2.0 ** -(math.log2(k) - 3)))
                ratio = start
                return [start * (ratio**i) for i in range(k)]

            if math.log2(n).is_integer():
                return get_power_of_2_slopes(n)
            closest = 2 ** math.floor(math.log2(n))
            slopes = get_power_of_2_slopes(closest)
            extra = get_power_of_2_slopes(2 * closest)[0::2][: n - closest]
            return slopes + extra

        return torch.tensor(get_slopes(num_heads), dtype=torch.float32)

    def _apply_epoch_context(self, eng_feats: torch.Tensor) -> torch.Tensor:
        """
        Apply epoch context attention to engineered features.

        This learns which engineered features are most relevant,
        similar to SleepTransformer's intra-epoch attention.

        Now includes position encoding support to maintain consistency with
        the transformer encoder's ALiBi or other positional biases.

        Args:
            eng_feats: [B, L, F] or [B, F] engineered features

        Returns:
            enhanced: [B, L, F] or [B, F] context-enhanced features
        """
        squeeze = eng_feats.dim() == 2
        if squeeze:
            eng_feats = eng_feats.unsqueeze(1)  # [B, 1, F]

        B, L, _ = eng_feats.shape
        center_idx = L // 2
        H = self.num_context_heads
        head_dim = self.head_dim  # Use stored head_dim (handles non-divisible feat_dim)

        # Add center epoch marker to identify the target epoch
        # This helps the attention mechanism focus on the epoch being classified
        eng_feats = eng_feats.clone()
        eng_feats[:, center_idx, :] = eng_feats[:, center_idx, :] + self.center_marker

        # Project to keys and values (outputs mh_dim * 2)
        kv = self.feat_to_kv(eng_feats)  # [B, L, mh_dim * 2]
        kv = kv.reshape(B, L, 2, H, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # Each: [B, H, L, head_dim]

        # Learnable queries with position-specific embeddings
        # stage_queries: [H, head_dim] provides base query (shared patterns)
        # query_pos_embed: [L, H, head_dim] provides position-specific modulation
        # Result: each position has a unique query enabling temporal differentiation
        base_q = self.stage_queries.unsqueeze(0).unsqueeze(2)  # [1, H, 1, head_dim]
        pos_q = (
            self.query_pos_embed[:L].permute(1, 0, 2).unsqueeze(0)
        )  # [1, H, L, head_dim]
        q = (base_q + pos_q).expand(B, -1, -1, -1)  # [B, H, L, head_dim]

        # Scaled dot-product attention
        scale = head_dim**0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, L, L]

        # Add position bias to attention scores (before softmax)
        if self.position_encoding == "alibi":
            # ALiBi: linear distance penalty matching transformer encoder
            pos = torch.arange(L, device=attn.device)
            rel_dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()  # [L, L]
            # slopes: [H] -> [H, 1, 1] for broadcasting
            bias = -rel_dist * cast(torch.Tensor, self.alibi_slopes).view(
                -1, 1, 1
            )  # [H, L, L]
            attn = attn + bias.unsqueeze(0)  # [B, H, L, L]
        elif self.position_encoding == "learned":
            # Learned position biases
            attn = attn + self.rel_pos_bias[:, :L, :L].unsqueeze(0)  # [B, H, L, L]

        attn = F.softmax(attn, dim=-1)

        # Aggregate: [B, H, L, head_dim] -> [B, L, mh_dim]
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).reshape(
            B, L, self.mh_dim
        )  # [B, L, mh_dim]

        # Output projection (mh_dim -> feat_dim)
        context = self.context_out(context)

        # Gated residual: original features + gated context
        gate = self.context_gate(torch.cat([eng_feats, context], dim=-1))
        enhanced = self.context_norm(eng_feats + gate * context)

        if squeeze:
            enhanced = enhanced.squeeze(1)

        return enhanced

    def forward(
        self,
        cnn_feats: torch.Tensor,
        eng_feats: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with epoch context attention.

        Args:
            cnn_feats: [B, L, D] or [B, D] CNN features
            eng_feats: [B, L, F] or [B, F] engineered features
            return_attention: Return cross-attention weights

        Returns:
            fused: [B, L, D] or [B, D] fused features
        """
        squeeze_output = cnn_feats.dim() == 2
        if squeeze_output:
            cnn_feats = cnn_feats.unsqueeze(1)
            eng_feats = eng_feats.unsqueeze(1)

        # === NEW: Apply epoch context to engineered features ===
        if self.use_epoch_context:
            eng_feats = self._apply_epoch_context(eng_feats)

        # === EXISTING: Your sequential fusion ===

        # Project engineered features
        eng_proj = self.feat_proj(eng_feats)

        # === N2-aware attention boost ===
        # Boost attention to N2-specific features in the key before cross-attention
        if self.n2_feature_indices:
            eng_proj_boosted = eng_proj.clone()
            for i, idx in enumerate(self.n2_feature_indices):
                if idx < eng_proj_boosted.size(-1):
                    eng_proj_boosted[..., idx] = (
                        eng_proj_boosted[..., idx] + self.n2_attention_boost[i]
                    )
            # Use boosted features as key, original as value
            attn_out, attn_weights = self.cross_attn(
                query=cnn_feats,
                key=eng_proj_boosted,
                value=eng_proj,
            )
        else:
            # Cross-attention without N2 boost
            attn_out, attn_weights = self.cross_attn(
                query=cnn_feats,
                key=eng_proj,
                value=eng_proj,
            )

        # Gated fusion
        # Include eng_proj directly so engineered features have independent influence
        gate_input = torch.cat([cnn_feats, eng_proj, attn_out], dim=-1)
        gate = self.gate(gate_input)
        fused = cnn_feats + gate * attn_out
        fused = self.norm(fused)

        if squeeze_output:
            fused = fused.squeeze(1)

        if return_attention:
            return fused, attn_weights
        return fused


class DualPathwayFusion(nn.Module):
    """
    Separate pathways for transient (N2) and sustained (N3) features.

    N2 sleep is characterized by transient events (spindles, K-complexes, delta
    variance) while N3 is characterized by sustained features (high delta power,
    slow wave count). This module processes them separately and learns to
    combine them.

    Args:
        cnn_dim: CNN feature dimension
        feat_dim: Engineered feature dimension
        num_heads: Number of attention heads per pathway
        dropout: Dropout probability
        use_variance_preserving_norm: Use VariancePreservingNorm instead of LayerNorm
    """

    def __init__(
        self,
        cnn_dim: int,
        feat_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_variance_preserving_norm: bool = True,
    ):
        super().__init__()

        self.cnn_dim = cnn_dim
        self.feat_dim = feat_dim

        # Project features to cnn_dim
        self.feat_proj = (
            nn.Linear(feat_dim, cnn_dim) if feat_dim != cnn_dim else nn.Identity()
        )

        # Transient pathway (spindles, K-complexes, delta variance)
        self.transient_attn = nn.MultiheadAttention(
            embed_dim=cnn_dim,
            num_heads=num_heads,
            kdim=cnn_dim,
            vdim=cnn_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Sustained pathway (total delta power, slow wave count)
        self.sustained_attn = nn.MultiheadAttention(
            embed_dim=cnn_dim,
            num_heads=num_heads,
            kdim=cnn_dim,
            vdim=cnn_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Feature routing (learn which features are transient vs sustained)
        # Output: 0 = sustained, 1 = transient
        self.feature_router = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )

        # Pathway combination gate
        self.pathway_gate = nn.Sequential(
            nn.Linear(cnn_dim * 2, cnn_dim),
            nn.Sigmoid(),
        )

        # Final normalization
        if use_variance_preserving_norm:
            self.norm = VariancePreservingNorm(cnn_dim)
        else:
            self.norm = nn.LayerNorm(cnn_dim)

    def forward(
        self,
        cnn_feats: torch.Tensor,
        eng_feats: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dual-pathway fusion.

        Args:
            cnn_feats: [B, L, D] or [B, D] CNN features
            eng_feats: [B, L, F] or [B, F] engineered features
            return_attention: Return pathway routing weights

        Returns:
            fused: [B, L, D] or [B, D] fused features
        """
        squeeze_output = cnn_feats.dim() == 2
        if squeeze_output:
            cnn_feats = cnn_feats.unsqueeze(1)
            eng_feats = eng_feats.unsqueeze(1)

        # Route features to pathways
        route_weights = self.feature_router(eng_feats)  # [B, L, feat_dim]

        transient_feats = eng_feats * route_weights
        sustained_feats = eng_feats * (1 - route_weights)

        # Project both pathways
        transient_proj = self.feat_proj(transient_feats)
        sustained_proj = self.feat_proj(sustained_feats)

        # Separate attention pathways
        transient_out, transient_attn = self.transient_attn(
            cnn_feats, transient_proj, transient_proj
        )
        sustained_out, sustained_attn = self.sustained_attn(
            cnn_feats, sustained_proj, sustained_proj
        )

        # Combine pathways with learned gate
        gate = self.pathway_gate(torch.cat([transient_out, sustained_out], dim=-1))
        combined = gate * transient_out + (1 - gate) * sustained_out

        # Residual connection and normalization
        fused = self.norm(cnn_feats + combined)

        if squeeze_output:
            fused = fused.squeeze(1)

        if return_attention:
            # Return the routing weights as "attention" for interpretability
            return fused, route_weights
        return fused


# =============================================================================
# INTEGRATION INTO YOUR EXISTING CODE
# =============================================================================


def patch_sequential_fusion_in_transformer_context_net():
    """
    Example code showing how to patch TransformerContextNet to use
    epoch attention. Add this to your model initialization.
    """
    example_code = """
    # In TransformerContextNet.__init__, find where you create sequential_fusion:
    
    # BEFORE (your current code):
    if feature_fusion_mode == "sequential":
        self.sequential_fusion = SequentialFeatureFusion(
            cnn_dim=enc_dim,
            feat_dim=enc_dim,
            num_heads=nhead // 2,
            dropout=cnn_dropout,
        )
    
    # AFTER (with epoch attention):
    if feature_fusion_mode == "sequential":
        # Import the enhanced version
        from .enhanced_sequential_fusion import EnhancedSequentialFeatureFusion
        
        self.sequential_fusion = EnhancedSequentialFeatureFusion(
            cnn_dim=enc_dim,
            feat_dim=enc_dim,
            num_heads=nhead // 2,
            dropout=cnn_dropout,
            use_epoch_context=use_epoch_attention,  # New CLI flag
            num_context_heads=nhead // 2,
        )
    """
    return example_code


def create_enhanced_sequential_fusion(
    cnn_dim: int,
    feat_dim: int,
    num_heads: int = 4,
    dropout: float = 0.1,
    use_epoch_attention: bool = True,
    epoch_attention_type: str = "context",  # 'context', 'dual_pathway', or 'none'
    position_encoding: str = "none",  # 'alibi', 'learned', 'none'
    n2_feature_indices: list[int] | None = None,
    use_variance_preserving_norm: bool = True,
) -> nn.Module:
    """
    Factory function to create sequential fusion with optional epoch attention.

    Use this in your model builder instead of directly instantiating
    SequentialFeatureFusion.

    Args:
        cnn_dim: CNN feature dimension
        feat_dim: Engineered feature dimension (before projection)
        num_heads: Number of cross-attention heads
        dropout: Dropout probability
        use_epoch_attention: Whether to use epoch context attention
        epoch_attention_type: Type of epoch attention:
            - 'context': Enhanced sequential fusion with epoch context (recommended)
            - 'dual_pathway': Separate transient/sustained pathways for N2/N3
            - 'none': Original sequential fusion
        position_encoding: Position encoding type ('alibi', 'learned', 'none').
            Use 'alibi' to match transformer encoder's ALiBi setting.
        n2_feature_indices: Indices of N2-discriminative features for attention boost.
            Typically: spindle_density, delta_continuity_var, sigma_power, delta_sigma_ratio
        use_variance_preserving_norm: Use VariancePreservingNorm instead of LayerNorm.
            Helps preserve delta variance information important for N2 detection.

    Returns:
        Sequential fusion module (drop-in compatible)
    """
    if use_epoch_attention and epoch_attention_type == "dual_pathway":
        return DualPathwayFusion(
            cnn_dim=cnn_dim,
            feat_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_variance_preserving_norm=use_variance_preserving_norm,
        )
    elif use_epoch_attention and epoch_attention_type == "context":
        return EnhancedSequentialFeatureFusion(
            cnn_dim=cnn_dim,
            feat_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_epoch_context=True,
            num_context_heads=num_heads,
            position_encoding=position_encoding,
            n2_feature_indices=n2_feature_indices,
            use_variance_preserving_norm=use_variance_preserving_norm,
        )
    else:
        # Fall back to your existing implementation
        from spectra.models.feature_extraction import SequentialFeatureFusion

        return SequentialFeatureFusion(
            cnn_dim=cnn_dim,
            feat_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
        )


# =============================================================================
# CLI INTEGRATION
# =============================================================================

CLI_ADDITIONS = """
# Add to cli.py after the feature_fusion_mode argument:

ap.add_argument(
    "--use_epoch_attention",
    action="store_true",
    help=(
        "Enable epoch-level context attention in sequential feature fusion. "
        "Learns which engineered feature patterns are most discriminative "
        "for each sleep stage (inspired by SleepTransformer). "
        "Only used when --feature_fusion_mode sequential."
    ),
)

ap.add_argument(
    "--epoch_attention_type",
    type=str,
    default="context",
    choices=["context", "dual_pathway", "none"],
    help=(
        "Type of epoch attention for sequential fusion. "
        "'context': Enhanced fusion with epoch context attention (recommended). "
        "'dual_pathway': Separate transient (N2) and sustained (N3) feature pathways. "
        "'none': Original sequential fusion without epoch attention. "
        "Only used when --use_epoch_attention is enabled."
    ),
)

ap.add_argument(
    "--epoch_attention_position",
    type=str,
    default="match",
    choices=["match", "alibi", "learned", "none"],
    help=(
        "Position encoding for epoch attention. "
        "'match': No extra positional bias (resolves to 'none'). "
        "'alibi': Always use ALiBi distance penalties. "
        "'learned': Learn position biases. "
        "'none': No position encoding (position-agnostic). "
        "Only used when --use_epoch_attention is enabled."
    ),
)

ap.add_argument(
    "--n2_feature_indices",
    type=str,
    default=None,
    help=(
        "Comma-separated indices of N2-discriminative features for attention boost. "
        "These should correspond to spindle_density, delta_continuity_var, "
        "sigma_power, delta_sigma_ratio in your engineered feature vector. "
        "Example: '3,7,12,15'. Only used with --use_epoch_attention."
    ),
)

ap.add_argument(
    "--use_variance_preserving_norm",
    action="store_true",
    default=True,
    help=(
        "Use VariancePreservingNorm instead of LayerNorm in fusion layers. "
        "Helps preserve delta variance information important for N2 detection. "
        "Enabled by default."
    ),
)
"""


# =============================================================================
# ALTERNATIVE: Simpler integration via feature weighting
# =============================================================================


class LearnableFeatureWeighting(nn.Module):
    """
    Simplest form of "epoch attention": learn fixed weights for each
    engineered feature based on its importance for sleep staging.

    This is a lightweight alternative that adds minimal parameters
    but still captures the key idea of learning feature importance.

    Can be added as a preprocessing step before your existing
    SequentialFeatureFusion.
    """

    def __init__(self, feat_dim: int, num_groups: int = 5):
        """
        Args:
            feat_dim: Total dimension of engineered features
            num_groups: Number of feature groups (e.g., 5 for band powers)
        """
        super().__init__()

        # Learnable importance weights per feature
        self.feature_weights = nn.Parameter(torch.ones(feat_dim))

        # Optional: Group-wise attention (e.g., for band powers)
        self.group_attention = nn.Sequential(
            nn.Linear(feat_dim, num_groups),
            nn.Softmax(dim=-1),
        )
        self.group_proj = nn.Linear(num_groups, feat_dim)

    def forward(self, eng_feats: torch.Tensor) -> torch.Tensor:
        """
        Apply learned feature weighting.

        Args:
            eng_feats: [..., feat_dim] engineered features

        Returns:
            weighted: [..., feat_dim] weighted features
        """
        # Apply per-feature weights
        weighted = eng_feats * F.softmax(self.feature_weights, dim=-1)

        # Optional group attention
        group_attn = self.group_attention(eng_feats)  # [..., num_groups]
        group_bias = self.group_proj(group_attn)  # [..., feat_dim]

        return weighted + 0.1 * group_bias  # Small residual from group attention
