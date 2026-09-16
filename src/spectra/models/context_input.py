"""Canonical waveform preparation for context-model encoder paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import torch
import torch.nn as nn

type WaveformInputs = torch.Tensor | Mapping[str, object]


class PreparedContextWaveforms(NamedTuple):
    """Canonical tensors shared by every context-model encoder path.

    Attributes:
        waveform: Contiguous physical waveform with shape ``[B, L, C, T]``.
            Values are unchanged.
        cnn_waveform: Finite, presence-masked waveform with shape
            ``[B * L, C, T]``.
        presence_mask: Actual channel presence with shape ``[B * L, C]``, or
            ``None`` when the caller supplied no mask.
        cnn_presence_mask: CNN-safe channel presence. All-missing rows receive a
            nominal first-channel placeholder while remaining invalid epochs.
        epoch_valid_mask: Boolean epoch validity with shape ``[B, L]``.
        recording_index: Optional ``[B]`` recording ids, one per context window.
            Encoders that normalise by per-recording statistics expand this
            ``L``-fold to match ``cnn_waveform``.
    """

    waveform: torch.Tensor
    cnn_waveform: torch.Tensor
    presence_mask: torch.Tensor | None
    cnn_presence_mask: torch.Tensor | None
    epoch_valid_mask: torch.Tensor
    recording_index: torch.Tensor | None = None


class ContextWaveformPreparer(nn.Module):
    """Validate and canonicalize waveform inputs for context models.

    This module is the single seam for waveform shape validation, channel-mask
    expansion, epoch-validity derivation, CNN sanitation, and all-missing epoch
    handling. It intentionally does not own resampling or normalization, which
    happen before the model.

    Args:
        expected_channels: Number of channels required by the model.
        samples_per_epoch: Number of waveform samples in each epoch.
    """

    def __init__(self, expected_channels: int, samples_per_epoch: int) -> None:
        super().__init__()
        if expected_channels < 1:
            raise ValueError("expected_channels must be positive")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        self.expected_channels = int(expected_channels)
        self.samples_per_epoch = int(samples_per_epoch)

    def forward(self, inputs: WaveformInputs) -> PreparedContextWaveforms:
        """Prepare a waveform tensor or mapping for all encoder variants.

        Args:
            inputs: A ``[B, L, C, T]`` tensor, or a mapping containing ``wave``
                and optional ``presence_mask`` (legacy ``mask``),
                and ``epoch_valid_mask`` tensors.

        Returns:
            Canonical waveform, mask, and validity tensors.

        Raises:
            TypeError: If an input field has the wrong type.
            ValueError: If a tensor has an unsupported shape or device.
            RuntimeError: If the waveform channel count does not match the model.
        """
        mapping: Mapping[str, object] | None
        if isinstance(inputs, torch.Tensor):
            waveform = inputs
            mapping = None
        elif isinstance(inputs, Mapping):
            mapping = inputs
            waveform_value = inputs.get("wave")
            if waveform_value is None:
                raise ValueError("Input mapping must contain a 'wave' tensor")
            if not isinstance(waveform_value, torch.Tensor):
                raise TypeError("Input mapping 'wave' must be a torch.Tensor")
            waveform = waveform_value
        else:
            raise TypeError("Inputs must be a tensor or mapping with a 'wave' entry")

        if waveform.ndim != 4:
            raise ValueError(
                "Expected waveform [B, L, C, T], got " f"{tuple(waveform.shape)}"
            )
        batch, epochs, channels, samples = waveform.shape
        if epochs != 21:
            raise ValueError("SPECTRA requires context_half=10 (21 context epochs)")
        if batch < 1 or epochs < 1:
            raise ValueError(
                "Waveform batch and context dimensions must be positive, got "
                f"{tuple(waveform.shape)}"
            )
        if channels != self.expected_channels:
            raise RuntimeError(
                f"Waveform channel dimension ({channels}) must match the size "
                f"expected by the model ({self.expected_channels})"
            )
        if samples != self.samples_per_epoch:
            raise ValueError(
                f"Expected {self.samples_per_epoch} samples per epoch, got {samples}"
            )

        waveform = waveform.contiguous()
        flat_rows = batch * epochs

        presence_value: object | None = None
        if mapping is not None:
            presence_value = mapping.get("presence_mask")
            if presence_value is None:
                presence_value = mapping.get("mask")
        presence_mask = self.normalize_presence_mask(
            presence_value,
            batch=batch,
            epochs=epochs,
            channels=channels,
            device=waveform.device,
        )

        epoch_valid_value = (
            mapping.get("epoch_valid_mask") if mapping is not None else None
        )
        epoch_valid_mask = self._normalize_epoch_valid_mask(
            epoch_valid_value,
            batch=batch,
            epochs=epochs,
            device=waveform.device,
        )

        cnn_waveform = torch.nan_to_num(
            waveform.reshape(flat_rows, channels, samples),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        cnn_presence_mask: torch.Tensor | None = None
        if presence_mask is not None:
            present = presence_mask.unsqueeze(-1)
            cnn_waveform = cnn_waveform * present.to(cnn_waveform.dtype)
            observed_epoch = presence_mask.any(dim=1).view(batch, epochs)
            epoch_valid_mask = epoch_valid_mask & observed_epoch

            fallback = (
                torch.arange(channels, device=waveform.device)
                .eq(0)
                .unsqueeze(0)
                .expand(flat_rows, -1)
            )
            cnn_presence_mask = torch.where(
                presence_mask.any(dim=1, keepdim=True),
                presence_mask,
                fallback,
            )

        recording_value = (
            mapping.get("recording_index") if mapping is not None else None
        )
        recording_index = (
            recording_value.reshape(-1).to(waveform.device, torch.long)
            if isinstance(recording_value, torch.Tensor)
            else None
        )

        return PreparedContextWaveforms(
            waveform=waveform,
            cnn_waveform=cnn_waveform,
            presence_mask=presence_mask,
            cnn_presence_mask=cnn_presence_mask,
            epoch_valid_mask=epoch_valid_mask,
            recording_index=recording_index,
        )

    @staticmethod
    def normalize_presence_mask(
        value: object | None,
        *,
        batch: int,
        epochs: int,
        channels: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Expand a supported channel-presence mask to ``[B * L, C]``."""
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            raise TypeError("presence_mask must be a torch.Tensor")
        if value.shape == (channels,):
            resolved = value.view(1, 1, channels).expand(batch, epochs, -1)
        elif value.shape == (batch, channels):
            resolved = value.view(batch, 1, channels).expand(-1, epochs, -1)
        elif value.shape == (batch, epochs, channels):
            resolved = value
        else:
            raise ValueError(
                "presence_mask must have shape "
                f"{(channels,)}, {(batch, channels)}, or "
                f"{(batch, epochs, channels)}; got {tuple(value.shape)}"
            )
        return resolved.to(device=device, dtype=torch.bool).reshape(
            batch * epochs, channels
        )

    @staticmethod
    def _normalize_epoch_valid_mask(
        value: object | None,
        *,
        batch: int,
        epochs: int,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            return torch.ones(batch, epochs, device=device, dtype=torch.bool)
        if not isinstance(value, torch.Tensor):
            raise TypeError("epoch_valid_mask must be a torch.Tensor")
        if value.shape != (batch, epochs):
            raise ValueError(
                f"epoch_valid_mask must have shape {(batch, epochs)}, got "
                f"{tuple(value.shape)}"
            )
        return value.to(device=device, dtype=torch.bool)
