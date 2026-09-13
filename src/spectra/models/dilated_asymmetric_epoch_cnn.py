"""
DilatedAsymmetricEpochCNN - Improved architecture for N2 classification.

This architecture improves upon PhysiologicalImprovedEpochCNN with two key changes:

1. MULTI-DILATED BLOCKS IN EARLY STAGES:
   Replace aggressive stride-2 downsampling in stages 1-2 with multi-dilated
   convolutional blocks that expand receptive field WITHOUT reducing temporal
   resolution. This preserves the fine temporal detail needed for spindle/K-complex
   detection (spindles oscillate at ~13Hz = 75ms per cycle, K-complexes have
   biphasic shape requiring ~50ms resolution to see).

2. ASYMMETRIC STEM BRANCH ALLOCATION:
   Allocate stem channels based on detection difficulty, not equally:
   - Fast branch: 20% (transients, artifacts, sharp waves)
   - Spindle branch: 35% (hardest to detect, defines N2)
   - K-complex branch: 30% (intermittent, shape-sensitive, defines N2)
   - Slow branch: 15% (reduced - high amplitude, continuous)

   The spindle branch also gets an extra conv layer to better capture the
   spindle envelope (waxing-waning amplitude modulation).

Resolution Analysis (why this matters):
- Current: Stem->T/2, Stage1->T/4, Stage2->T/8 => 62ms/sample at stage 2
- New: Stem->T/2, Stage1->T/2 (no downsample), Stage2->T/4 => 31ms/sample at stage 2
- 31ms resolution can see individual spindle oscillations (75ms period)
- 62ms resolution loses spindle oscillatory structure

Author: Michael (PSGStage project)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .anti_alias import KaiserAntiAliasDownsample1D
from .blur_pool import BlurPool1D

try:
    torch_compile_disable = torch.compiler.disable
except AttributeError:
    try:
        torch_compile_disable = torch._dynamo.disable
    except AttributeError:

        def torch_compile_disable(fn):
            return fn


MIN_INTERMEDIATE_CHANNELS = 1

SPINDLE_EXPANSION_FACTOR = 1.25

TEMPORAL_WINDOW_FRACTION = 8

MIN_FILTER_ORDER = 51

RATIO_TOLERANCE = 1e-4


def _make_norm1d(norm: str, num_features: int) -> nn.Module:
    """Create normalization layer."""
    if norm == "bn":
        return nn.BatchNorm1d(num_features)
    elif norm == "gn":
        num_groups = min(32, max(1, num_features // 4))
        while num_features % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_features)
    elif norm == "ln":
        return nn.GroupNorm(1, num_features)  # LayerNorm equivalent
    else:
        raise ValueError(f"Unknown norm: {norm}")


def _get_activation(activation: str) -> nn.Module:
    """Get activation function by name."""
    if activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation: {activation}")


class MultiDilatedBlock(nn.Module):
    """
    Multi-dilated convolutional block that expands receptive field without downsampling.

    Instead of using stride-2 convolutions that lose temporal resolution, this block
    uses parallel branches with different dilation rates to capture multi-scale
    patterns while preserving the original resolution.

    Architecture:
        Input -> [Branch d=1, Branch d=2, Branch d=4, ...] -> Concat -> 1x1 Conv -> Residual Add

    Receptive field math: RF = dilation * (kernel_size - 1) + 1
        - k=7, d=1: RF=7 samples (55ms @ 128Hz)
        - k=7, d=2: RF=13 samples (102ms @ 128Hz)
        - k=7, d=4: RF=25 samples (195ms @ 128Hz)

    Combined RF covers from fast transients to K-complex timescales.

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        kernel_size: Base kernel size for all branches (default: 7)
        dilations: Tuple of dilation rates (default: (1, 2, 4))
        stride: Output stride for optional pooling (default: 1 = no pooling)
        dropout: Dropout probability
        norm: Normalization type ('bn', 'gn', 'ln')
        activation: Activation function ('gelu', 'silu')
        res_scale_init: Initial value for learnable residual scaling
        use_kaiser: Use configurable Kaiser filtering instead of BlurPool1D.
        aa_cutoff_ratio: Cutoff relative to the post-decimation Nyquist.
        aa_num_taps: Odd FIR length used by downsampling filters.
        aa_beta: Kaiser window beta.
        anti_alias_dilated_branches: Preserve the legacy pre-dilation smoothing.
            Dilation alone does not decimate; new callers should disable this.
        aa_legacy_cutoff: Measure ``aa_cutoff_ratio`` against the pre-decimation
            Nyquist, reproducing the filter design used before the decimation
            cutoff was corrected. Affects the stride>1 filters only; the
            stride-1 pre-dilation filters are identical either way. Enable only
            to reconstruct pre-fix checkpoints.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4),
        stride: int = 1,
        dropout: float = 0.1,
        norm: str = "bn",
        activation: str = "gelu",
        res_scale_init: float = 0.1,
        use_kaiser: bool = False,
        aa_cutoff_ratio: float = 0.85,
        aa_num_taps: int = 9,
        aa_beta: float = 6.0,
        anti_alias_dilated_branches: bool = True,
        aa_legacy_cutoff: bool = False,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilations = dilations
        self.stride = stride

        n_branches = len(dilations)
        branch_ch = out_ch // n_branches
        remainder = out_ch - n_branches * branch_ch

        act_fn = _get_activation(activation)

        # Parallel dilated convolution branches
        self.branches = nn.ModuleList()
        for i, d in enumerate(dilations):
            # Last branch gets remainder channels
            ch = branch_ch + (remainder if i == n_branches - 1 else 0)
            # Explicit effective kernel formula for same-padding (generalizes to any kernel_size)
            # effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
            # This is mathematically equivalent to (kernel_size - 1) * d // 2 for odd kernels,
            # but more explicit about the dilated convolution mechanics
            effective_kernel = kernel_size + (kernel_size - 1) * (d - 1)
            padding = effective_kernel // 2

            branch_layers: list[nn.Module] = []

            # Optional legacy pre-dilation smoothing. A dilated convolution retains
            # every output timestep, so this is not required for anti-aliasing.
            if d >= 4 and use_kaiser and anti_alias_dilated_branches:
                # cutoff_ratio = 0.5/d ensures filter cuts off before effective Nyquist
                branch_layers.append(
                    KaiserAntiAliasDownsample1D(
                        channels=in_ch,
                        cutoff_ratio=0.5 / d,  # e.g., 0.125 for d=4
                        num_taps=aa_num_taps,
                        beta=aa_beta,
                        stride=1,  # No downsampling, just filtering
                        # No-op at stride 1, where both cutoff conventions agree.
                        legacy_cutoff=aa_legacy_cutoff,
                    )
                )

            branch_layers.extend(
                [
                    nn.Conv1d(
                        in_ch, ch, kernel_size=kernel_size, dilation=d, padding=padding
                    ),
                    _make_norm1d(norm, ch),
                    act_fn,
                ]
            )

            self.branches.append(nn.Sequential(*branch_layers))

        # 1x1 fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1),
            _make_norm1d(norm, out_ch),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def downsample(channels: int) -> nn.Module:
            if use_kaiser:
                return KaiserAntiAliasDownsample1D(
                    channels=channels,
                    cutoff_ratio=aa_cutoff_ratio,
                    num_taps=aa_num_taps,
                    beta=aa_beta,
                    stride=stride,
                    legacy_cutoff=aa_legacy_cutoff,
                )
            return BlurPool1D(channels, stride=stride)

        # Optional pooling for stride > 1 with proper anti-aliasing
        if stride > 1:
            self.pool = downsample(out_ch)
        else:
            self.pool = nn.Identity()

        # Shortcut connection with matched anti-aliasing
        if in_ch != out_ch:
            shortcut_layers = [
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                _make_norm1d(norm, out_ch),
            ]
            if stride > 1:
                shortcut_layers.append(downsample(out_ch))
            self.shortcut = nn.Sequential(*shortcut_layers)
        elif stride > 1:
            self.shortcut = downsample(in_ch)
        else:
            self.shortcut = nn.Identity()

        # Learnable residual scaling
        self.res_scale = nn.Parameter(torch.tensor(res_scale_init))
        self.final_act = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_ch, T] input tensor
        Returns:
            [B, out_ch, T//stride] output tensor
        """
        identity = self.shortcut(x)

        # Parallel dilated branches
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)

        # Fuse
        out = self.fusion(out)
        out = self.dropout(out)
        out = self.pool(out)

        # Scaled residual connection
        return self.final_act(identity + self.res_scale * out)
