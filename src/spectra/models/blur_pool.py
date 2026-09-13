"""SPECTRA inference support."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool1D(nn.Module):
    """
    Anti-aliased downsampling using a low-pass filter followed by subsampling.

    This preserves shift invariance and reduces aliasing compared to standard MaxPool.
    Based on: https://arxiv.org/abs/1904.11486
    """

    filt: torch.Tensor

    def __init__(self, channels: int, filt_size: int = 3, stride: int = 2):
        super().__init__()

        if filt_size == 3:
            filt = [1, 2, 1]
        elif filt_size == 5:
            filt = [1, 2, 4, 2, 1]
        else:
            raise ValueError(f"Unsupported filter size: {filt_size}")

        f = torch.tensor(filt, dtype=torch.float32)
        f = (f / f.sum()).view(1, 1, -1)  # (1,1,K)

        # Register buffer so it's part of state_dict but not a learnable parameter
        self.register_buffer("filt", f.repeat(channels, 1, 1))  # (C,1,K)

        self.stride = stride
        self.pad = (len(filt) - 1) // 2
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # Apply low-pass filter (depthwise convolution)
        x_blurred = F.conv1d(
            x, self.filt, stride=1, padding=self.pad, groups=self.channels
        )

        # Subsample
        return x_blurred[:, :, :: self.stride]
