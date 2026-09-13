"""Serialized-state adapter for retired raw-waveform channel offsets."""

from __future__ import annotations

import torch
import torch.nn as nn


class LegacyChannelEmbeddingState(nn.Module):
    """Preserve the historical channel-offset checkpoint schema.

    Older checkpoints contain ``channel_embedding.channel_embed`` as a learned
    DC offset. A constant raw-signal offset is not a useful channel identity
    embedding for modality-aware CNNs and creates a pretrain/fine-tune mismatch.
    This adapter remains registered solely so those checkpoints load without a
    state-dict migration. Runtime input semantics live in
    :class:`spectra.models.context_input.ContextWaveformPreparer`.

    Args:
        num_channels: Number of input channels.
        embed_dim: Retained for API compatibility; unused.
    """

    channel_embed: torch.Tensor

    def __init__(self, num_channels: int, embed_dim: int = 1) -> None:
        super().__init__()
        self.num_channels = num_channels
        del embed_dim
        self.register_buffer(
            "channel_embed", torch.zeros(1, 1, num_channels, 1), persistent=True
        )

    def reset_parameters(self) -> None:
        """Keep the compatibility buffer at the identity value."""
        with torch.no_grad():
            self.channel_embed.zero_()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Validate and return an unchanged contiguous waveform.

        Args:
            waveform: [B, L, C, T] raw waveform input

        Returns:
            [B, L, C, T] unmodified waveform
        """
        if waveform.ndim != 4:
            raise ValueError(
                f"Expected waveform [B, L, C, T], got {tuple(waveform.shape)}"
            )
        if waveform.size(2) != self.num_channels:
            raise RuntimeError(
                f"Waveform channel dimension ({waveform.size(2)}) must match the size "
                f"expected by the model ({self.num_channels})"
            )
        return waveform.contiguous()


# Public import compatibility for callers that historically constructed the
# identity module directly. New model code should use the explicit legacy name.
ChannelEmbedding = LegacyChannelEmbeddingState
