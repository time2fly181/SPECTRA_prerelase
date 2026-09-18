"""Physiological stem retained for compatible multirate checkpoints."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .anti_alias import KaiserAntiAliasDownsample1D
from .common import _make_norm1d


class FlexiblePhysiologicalStem(nn.Module):
    """
    Flexible multi-dilation stem with sigmoid gating and global context.
    Designed for physiological
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
        self.pool = KaiserAntiAliasDownsample1D(
            channels=out_ch,
            stride=2,
            cutoff_ratio=aa_cutoff_ratio,
            num_taps=aa_num_taps,
            beta=aa_beta,
            legacy_cutoff=aa_legacy_cutoff,
        )

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
