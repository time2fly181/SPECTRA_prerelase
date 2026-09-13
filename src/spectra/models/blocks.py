# spectra.models/blocks.py
"""
Reusable building blocks for PSG models.

FREQUENCY PRESERVATION FIX (v2.0):
- Added dilation support to ResidualConvBlock for receptive field without downsampling
- Optional Kaiser anti-aliasing filter for proper frequency cutoff
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anti_alias import KaiserAntiAliasDownsample1D
from .blur_pool import BlurPool1D
from .common import DropPath, FusedLayerNorm, _make_norm1d


class ConvBlock(nn.Module):
    """
    1D Convolution block with normalization, activation, pooling, and dropout.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        k: Kernel size.
        p: Padding.
        pool: Pooling size (1 or None disables pooling).
        dropout: Dropout probability.
        norm: Normalization type ('bn' or 'gn').
        res_scale_init: Initial value for learnable residual scaling γ.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
        p: int,
        pool: int = 2,
        dropout: float = 0.0,
        *,
        norm: str = "bn",
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p)
        self.bn = _make_norm1d(norm, out_ch)
        self.act = nn.GELU(approximate="tanh")  # Fast approximation for AMP (fp16/bf16)
        self.pool = (
            BlurPool1D(channels=out_ch, stride=pool)
            if pool and pool > 1
            else nn.Identity()
        )
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.do(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Standard sine/cosine positional encoding.

    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length.
    """

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        # Handle odd d_model: sin gets ceiling(d_model/2), cos gets floor(d_model/2)
        pe[:, 0::2] = torch.sin(pos * div[: min(div.size(0), (d_model + 1) // 2)])
        pe[:, 1::2] = torch.cos(pos * div[: min(div.size(0), d_model // 2)])
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape [B, L, D].

        Returns:
            Tensor of shape [B, L, D] with positional encoding added.
        """
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)  # [B, L, D]


class DropPathTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder block with stochastic depth, LayerScale, and locality bias support.

    LayerScale (from CaiT/DeiT-III) adds learnable per-dimension scaling to residual
    connections, helping preserve original features when needed (e.g., for transient
    sleep stage detection like N1).

    Locality bias adds a distance-based penalty to attention, making nearby positions
    attend more strongly than distant ones. This helps transient stage detection by
    preventing distant epochs from overwhelming local signals.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: Feedforward dimension.
        dropout: Dropout probability.
        activation: Activation function ('relu' or 'gelu').
        drop_path: Stochastic depth probability.
        layer_scale_init: Initial value for LayerScale parameters. Use 1.0 for
            backward compatibility with old checkpoints. Lower values (e.g., 0.1)
            help preserve original epoch features when context is added.
        use_locality_bias: Whether to add distance-based locality bias to attention.
        locality_strength_init: Initial decay strength for locality bias. Higher values
            mean stronger preference for nearby positions (0.1 = gentle, 0.5 = strong).
        locality_learnable: Whether the locality decay is learnable or fixed.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        activation: str = "relu",
        drop_path: float = 0.0,
        layer_scale_init: float = 1.0,
        use_locality_bias: bool = False,
        locality_strength_init: float = 0.1,
        locality_learnable: bool = True,
    ):
        super().__init__()
        # Store for checkpoint compatibility hooks
        self.d_model = d_model
        self.nhead = nhead
        # Validate that the model width is compatible with the requested number of heads.
        # torch.nn.MultiheadAttention asserts this internally, but we raise a clearer
        # error here so callers get helpful guidance when model config is invalid.
        if nhead <= 0:
            raise ValueError(f"nhead must be > 0; received {nhead}")
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead}) for MultiheadAttention. "
                f"Choose a d_model that is a multiple of nhead or adjust nhead accordingly."
            )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # Use fused LayerNorm if available (5-10% speedup)
        self.norm1 = FusedLayerNorm(d_model)
        self.norm2 = FusedLayerNorm(d_model)

        if activation == "relu":
            self.activation = nn.GELU(
                approximate="tanh"
            )  # Fast approximation for AMP (fp16/bf16)
        elif activation == "gelu":
            self.activation = nn.GELU(
                approximate="tanh"
            )  # Fast approximation for AMP (fp16/bf16)
        else:
            raise ValueError(f"Unsupported activation '{activation}'")

        drop_prob = max(0.0, float(drop_path))
        self.drop_path1 = DropPath(drop_prob) if drop_prob > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob) if drop_prob > 0 else nn.Identity()

        # LayerScale: learnable per-dimension scaling for residual connections
        # Helps preserve original epoch features when context transformer is added
        self.layer_scale1 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.layer_scale2 = nn.Parameter(torch.ones(d_model) * layer_scale_init)

        # Locality bias: distance-based attention penalty for local focus
        # Helps transient stage detection by favoring nearby epochs
        self.use_locality_bias = use_locality_bias
        if use_locality_bias:
            if locality_learnable:
                # Learnable decay rate per head (softplus ensures positive)
                self.locality_decay = nn.Parameter(
                    torch.ones(nhead, 1, 1) * locality_strength_init
                )
            else:
                # Fixed decay rate
                self.register_buffer(
                    "locality_decay",
                    torch.ones(nhead, 1, 1) * locality_strength_init,
                )
            # Cache for distance matrix (computed once per sequence length)
            self._cached_distance: torch.Tensor | None = None
            self._cached_seq_len: int = 0

        # Register hook for backward compatibility with checkpoints missing layer_scale/locality
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _load_state_dict_pre_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Initialize missing layer_scale/locality params for backward compatibility."""
        # Handle missing LayerScale params — use current init values (not hardcoded 1.0)
        # so that layer_scale_init configured at construction time is preserved.
        for name in ["layer_scale1", "layer_scale2"]:
            key = f"{prefix}{name}"
            if key not in state_dict:
                current_param = getattr(self, name, None)
                if current_param is not None:
                    state_dict[key] = current_param.data.clone()
                else:
                    state_dict[key] = torch.ones(self.d_model)

        # Handle missing locality_decay param
        locality_key = f"{prefix}locality_decay"
        if locality_key not in state_dict and self.use_locality_bias:
            current_decay = getattr(self, "locality_decay", None)
            if current_decay is not None:
                state_dict[locality_key] = current_decay.data.clone()
            else:
                state_dict[locality_key] = torch.ones(self.nhead, 1, 1) * 0.1

    def _get_distance_matrix(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute or retrieve cached distance matrix."""
        if self._cached_seq_len != seq_len or self._cached_distance is None:
            positions = torch.arange(seq_len, device=device, dtype=torch.float)
            # [L, L] distance matrix: |i - j|
            self._cached_distance = (
                positions.unsqueeze(0) - positions.unsqueeze(1)
            ).abs()
            self._cached_seq_len = seq_len
        return self._cached_distance.to(device)

    def _compute_locality_bias(
        self, seq_len: int, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Compute locality bias for attention.

        Returns additive bias of shape [B * nhead, L, L] for nn.MultiheadAttention.
        """
        distance = self._get_distance_matrix(seq_len, device)  # [L, L]
        # Apply learnable decay (softplus ensures positive)
        decay = F.softplus(self.locality_decay)  # [nhead, 1, 1]
        # Negative bias: larger distance = more negative = lower attention
        bias = -decay * distance.unsqueeze(0)  # [nhead, L, L]
        # Expand for batch: [nhead, L, L] -> [B * nhead, L, L]
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, nhead, L, L]
        bias = bias.reshape(
            batch_size * self.nhead, seq_len, seq_len
        )  # [B*nhead, L, L]
        return bias.to(dtype)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with pre-norm architecture.

        Args:
            src: Input tensor of shape [B, L, D].
            src_mask: Optional attention mask (additive, shape [L, L] or [B*nhead, L, L]).
            src_key_padding_mask: Optional key padding mask.

        Returns:
            Output tensor of shape [B, L, D].
        """
        # Pre-norm self-attention block
        x = src
        B, L, _ = x.shape

        q = self.norm1(x)

        # Standard self-attention (nn.MultiheadAttention).
        # Note: PyTorch 2.0+ uses scaled_dot_product_attention (SDPA) by default
        # which is more numerically stable when need_weights=False
        attn_mask = src_mask
        if self.use_locality_bias:
            locality_bias = self._compute_locality_bias(L, B, x.device, x.dtype)
            if attn_mask is None:
                attn_mask = locality_bias
            else:
                # Combine existing mask with locality bias
                # Expand src_mask to match locality_bias shape if needed
                if attn_mask.dim() == 2:
                    # [L, L] -> [B*nhead, L, L]
                    attn_mask = attn_mask.unsqueeze(0).expand(B * self.nhead, -1, -1)
                attn_mask = attn_mask + locality_bias

        attn_out, attn_weights = self.self_attn(
            q,
            q,
            q,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False if return_attention else True,
        )

        x = x + self.layer_scale1 * self.drop_path1(self.dropout1(attn_out))

        # Pre-norm feed-forward block
        y = self.norm2(x)
        y = self.linear2(self.dropout(self.activation(self.linear1(y))))

        x = x + self.layer_scale2 * self.drop_path2(self.dropout2(y))

        # Single sanitization at layer output boundary
        # This handles edge cases without masking root causes in attention/FFN
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if return_attention and attn_weights is not None:
            return x, attn_weights
        return x


class ResidualConvBlock(nn.Module):
    """
    Residual 1D Convolution block with optional downsampling and dilation.

    FREQUENCY PRESERVATION FIX (v2.0):
    - Added dilation parameter for receptive field growth without downsampling
    - Optional configurable Kaiser anti-aliasing
    - Default still uses BlurPool1D for backward compatibility

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        k: Kernel size.
        pool: Pooling size (1 or None disables pooling).
        dropout: Dropout probability.
        norm: Normalization type ('bn' or 'gn').
        dilation: Dilation factor for convolutions (default 1, no dilation).
        use_kaiser: If True, use Kaiser anti-aliasing instead of BlurPool1D.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
        pool: int = 1,
        dropout: float = 0.0,
        *,
        norm: str = "bn",
        res_scale_init: float = 0.1,
        dilation: int = 1,
        use_kaiser: bool = False,
    ):
        super().__init__()

        # Compute padding for same-size output with dilation
        # For dilated conv: effective_kernel = k + (k - 1) * (dilation - 1)
        effective_k = k + (k - 1) * (dilation - 1)
        p = effective_k // 2

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=k, padding=p, dilation=dilation
        )
        self.bn1 = _make_norm1d(norm, out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=k, padding=p, dilation=dilation
        )
        self.bn2 = _make_norm1d(norm, out_ch)
        self.act = nn.GELU(approximate="tanh")  # Fast approximation for AMP (fp16/bf16)

        # Choose anti-aliasing filter
        if pool and pool > 1:
            if use_kaiser:
                self.pool = KaiserAntiAliasDownsample1D(channels=out_ch, stride=pool)
            else:
                self.pool = BlurPool1D(channels=out_ch, stride=pool)
        else:
            self.pool = nn.Identity()

        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        init_scale = float(res_scale_init)
        if init_scale <= 0:
            raise ValueError(
                f"res_scale_init must be positive; received {res_scale_init}."
            )
        self.res_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

        # Shortcut connection
        if in_ch != out_ch or (pool and pool > 1):
            if use_kaiser and pool and pool > 1:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=1),
                    _make_norm1d(norm, out_ch),
                    KaiserAntiAliasDownsample1D(channels=out_ch, stride=pool),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=1),
                    _make_norm1d(norm, out_ch),
                    (
                        BlurPool1D(channels=out_ch, stride=pool)
                        if pool and pool > 1
                        else nn.Identity()
                    ),
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.do(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.pool(out)

        out = identity + self.res_scale * out
        out = self.act(out)
        return out


class TemporalConvBlock(nn.Module):
    """
    Dilated Conv1d block operating on [B, F, L] with optional residual.

    Args:
        channels: Number of channels.
        kernel_size: Kernel size (must be odd for same padding).
        dilation: Dilation factor.
        dropout: Dropout probability.
        residual: Whether to add residual connection.
        norm: Normalization type ('bn' or 'gn').
        res_scale_init: Initial value for learnable residual scaling γ.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        residual: bool = True,
        *,
        norm: str = "bn",
        res_scale_init: float = 0.1,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                "TemporalConvBlock expects an odd kernel_size for same padding."
            )
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = _make_norm1d(norm, channels)
        self.act = nn.GELU(approximate="tanh")  # Fast approximation for AMP (fp16/bf16)
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual
        if self.residual:
            init_scale = float(res_scale_init)
            if init_scale <= 0:
                raise ValueError(
                    f"res_scale_init must be positive; received {res_scale_init}."
                )
            self.res_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        else:
            self.register_parameter("res_scale", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, F, L].

        Returns:
            Output tensor of shape [B, F, L].
        """
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.do(out)
        if self.residual:
            out = x + self.res_scale * out
        return out


class FlexiblePhysiologicalStem(nn.Module):
    """
    Flexible multi-dilation stem with sigmoid gating and global context.

    Combines U-Sleep's uniform kernel approach with ASPP's multi-dilation strategy
    and SE-style attention for learned scale selection. Designed for physiological
    signals where multiple timescales must be active simultaneously.

    Architecture:
        Input [B, in_ch, T]
          ├─ Multi-dilation branches (same kernel k, different d)
          │   ├─ Branch d=1: k samples (fast transients)
          │   ├─ Branch d=2: (k + k-1) samples (medium)
          │   ├─ Branch d=4: (k + 3(k-1)) samples (slow)
          │   └─ Branch d=8: (k + 7(k-1)) samples (very slow)
          ├─ Global Context Branch
          │   └─ 1x1 conv → normalization → activation → GAP → Upsample to T
          └─ SE Attention (sigmoid gating)
              └─ GAP → FC → Sigmoid → [B, num_scales]

        Weighted Fusion → [B, out_ch, T]
        Anti-aliased Downsample 2x → [B, out_ch, T//2]

    Key Features:
    - Same kernel size across branches (simplifies hyperparameter tuning)
    - Dilation-based multi-scale (not varying kernel sizes)
    - Sigmoid gating allows multiple scales active simultaneously (not competitive)
    - Global context via ASPP-style branch for slow/context cues
    - SE-style attention for adaptive scale weighting
    - Proper anti-aliasing with Kaiser filter

    Timescale Examples (k=9, fs=128Hz):
        d=1:  9 samples  = 70ms  (spindles, beta, fast transients)
        d=2: 17 samples  = 133ms (alpha, theta)
        d=4: 33 samples  = 258ms (K-complexes, slow theta)
        d=8: 65 samples  = 508ms (delta, slow waves)
        global: entire window (context)

    Args:
        in_ch: Input channels.
        out_ch: Output channels. Widths smaller than the number of scales use as
            many scales as channels, while the output width remains exact.
        kernel_size: Base kernel size for all dilated branches (default 9).
        dilations: Dilation rates for multi-scale branches (default (1,2,4,8)).
        fs: Sampling frequency in Hz (for documentation, not used in computation).
        norm: Normalization type ('bn' or 'gn'). Use 'gn' for small batch sizes.
        activation: Activation function ('gelu', 'silu').
        dropout: Dropout probability.
        use_kaiser: Use configurable Kaiser anti-aliasing instead of BlurPool.
        se_reduction: SE attention reduction ratio (default 4).
        return_weights: If True, forward() returns (output, scale_weights) for interpretability.
        legacy_global_branch: Rebuild the global-context branch with the ordering
            used before the branch was reordered (GAP -> 1x1 conv with bias ->
            norm -> activation). This exists solely to reconstruct checkpoints
            trained against that layout; the current ordering normalizes before
            pooling so a one-sample modality subset still supplies T values per
            channel to training-mode BatchNorm. Do not enable for new training.
        aa_legacy_cutoff: Measure ``aa_cutoff_ratio`` against the pre-decimation
            Nyquist in the 2x downsampling filter, reproducing the design used
            before the decimation cutoff was corrected. Do not enable for new
            training.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 9,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        fs: int = 128,
        norm: str = "bn",
        activation: str = "gelu",
        dropout: float = 0.1,
        use_kaiser: bool = True,
        se_reduction: int = 4,
        return_weights: bool = False,
        aa_cutoff_ratio: float = 0.85,
        aa_num_taps: int = 9,
        aa_beta: float = 6.0,
        legacy_global_branch: bool = False,
        aa_legacy_cutoff: bool = False,
    ):
        super().__init__()

        if not isinstance(in_ch, int) or in_ch < 1:
            raise ValueError(f"in_ch must be a positive integer, got {in_ch!r}")
        if not isinstance(out_ch, int) or out_ch < 1:
            raise ValueError(f"out_ch must be a positive integer, got {out_ch!r}")
        if not dilations:
            raise ValueError("dilations must contain at least one value")
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd for same padding, got {kernel_size}"
            )

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.fs = fs
        self.return_weights = return_weights
        self.legacy_global_branch = bool(legacy_global_branch)

        # Number of scales = dilated branches + 1 global branch
        num_scales = len(dilations) + 1
        self.num_scales = num_scales

        # Preserve the established allocation for normal widths. For widths
        # smaller than the number of scales, activate one scale per channel
        # instead of constructing invalid zero-output convolutions.
        if out_ch < num_scales:
            scale_channels = [1 if i < out_ch else 0 for i in range(num_scales)]
        else:
            branch_ch = out_ch // num_scales
            remainder = out_ch - branch_ch * num_scales
            scale_channels = [branch_ch] * num_scales
            scale_channels[len(dilations) - 1] += remainder
        self.scale_channels = tuple(scale_channels)

        # Activation function
        if activation == "gelu":
            act_fn = nn.GELU(approximate="tanh")
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Multi-dilation branches
        self.dilated_branches = nn.ModuleList()
        for i, d in enumerate(dilations):
            ch = self.scale_channels[i]

            if ch == 0:
                self.dilated_branches.append(nn.Identity())
                continue

            # Compute padding for same-size output
            effective_kernel = kernel_size + (kernel_size - 1) * (d - 1)
            padding = effective_kernel // 2

            branch = nn.Sequential(
                nn.Conv1d(
                    in_ch, ch, kernel_size=kernel_size, dilation=d, padding=padding
                ),
                _make_norm1d(norm, ch),
                (
                    act_fn if isinstance(act_fn, nn.Module) else type(act_fn)()
                ),  # Clone activation
            )
            self.dilated_branches.append(branch)

        # Normalize before global pooling so a one-sample modality subset still
        # supplies T values per channel to training-mode BatchNorm.
        global_ch = self.scale_channels[-1]
        self.global_branch: nn.Module
        if global_ch <= 0:
            self.global_branch = nn.Identity()
        elif self.legacy_global_branch:
            # Checkpoint-reconstruction ordering only: pooling first means the
            # norm sees a length-1 sequence, which is the pathology the current
            # ordering was introduced to fix. Both layouts emit [B, global_ch, 1].
            self.global_branch = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(in_ch, global_ch, kernel_size=1),
                _make_norm1d(norm, global_ch),
                act_fn if isinstance(act_fn, nn.Module) else type(act_fn)(),
            )
        else:
            self.global_branch = nn.Sequential(
                nn.Conv1d(in_ch, global_ch, kernel_size=1, bias=False),
                _make_norm1d(norm, global_ch),
                act_fn if isinstance(act_fn, nn.Module) else type(act_fn)(),
                nn.AdaptiveAvgPool1d(1),
            )

        # SE-style scale attention (sigmoid gating allows multiple scales active)
        total_ch = out_ch  # Sum of all branch channels
        se_hidden = max(num_scales, total_ch // se_reduction)
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(total_ch, se_hidden),
            nn.SiLU(),
            nn.Linear(se_hidden, num_scales),
            nn.Sigmoid(),  # Sigmoid allows multiple scales to be active simultaneously
        )

        # Channel fusion after weighting
        self.fusion = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1),
            _make_norm1d(norm, out_ch),
            act_fn if isinstance(act_fn, nn.Module) else type(act_fn)(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Anti-aliased 2x downsampling
        if use_kaiser:
            self.pool = KaiserAntiAliasDownsample1D(
                channels=out_ch,
                stride=2,
                cutoff_ratio=aa_cutoff_ratio,
                num_taps=aa_num_taps,
                beta=aa_beta,
                legacy_cutoff=aa_legacy_cutoff,
            )
        else:
            self.pool = BlurPool1D(channels=out_ch, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, in_ch, T].

        Returns:
            If return_weights=False: [B, out_ch, T//2]
            If return_weights=True: ([B, out_ch, T//2], [B, num_scales])
        """
        B, _, T = x.shape

        # Process dilated branches
        branch_outputs: list[tuple[int, torch.Tensor]] = []
        for scale_index, (branch, channels) in enumerate(
            zip(self.dilated_branches, self.scale_channels[:-1], strict=True)
        ):
            if channels > 0:
                branch_outputs.append(
                    (scale_index, branch(x))
                )  # Each: [B, scale_ch, T]

        # Process global context branch
        if self.scale_channels[-1] > 0:
            global_out = self.global_branch(x)  # [B, global_ch, 1]
            # Upsample to match temporal dimension
            global_out = F.interpolate(
                global_out, size=T, mode="nearest"
            )  # [B, global_ch, T]
            branch_outputs.append((self.num_scales - 1, global_out))

        # Concatenate all branches
        concat = torch.cat(
            [branch_out for _, branch_out in branch_outputs], dim=1
        )  # [B, out_ch, T]

        # Compute scale attention weights (sigmoid gating)
        scale_weights = self.scale_attention(concat)  # [B, num_scales]

        # Apply scale weights to each branch
        weighted_outputs = []
        for scale_index, branch_out in branch_outputs:
            w = scale_weights[:, scale_index].view(B, 1, 1)  # [B, 1, 1]
            weighted_outputs.append(branch_out * w)

        # Concatenate weighted branches
        weighted_concat = torch.cat(weighted_outputs, dim=1)  # [B, out_ch, T]

        # Fusion
        fused = self.fusion(weighted_concat)  # [B, out_ch, T]

        # Downsample
        output = self.pool(fused)  # [B, out_ch, T//2]

        if self.return_weights:
            return output, scale_weights
        return output
