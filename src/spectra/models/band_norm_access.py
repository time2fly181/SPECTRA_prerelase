"""Access per-recording band normalization in the unwrapped inference model."""

from __future__ import annotations

import torch.nn as nn

from spectra.models.multirate_asymmetric_epoch_cnn import (
    MultiRateAsymmetricEpochCNN,
    PerRecordingBandNorm,
)

__all__ = ["find_multirate_encoder", "recording_norm_modules"]


def find_multirate_encoder(model: nn.Module) -> MultiRateAsymmetricEpochCNN | None:
    """Return the direct epoch encoder, or an encoder passed on its own."""
    if isinstance(model, MultiRateAsymmetricEpochCNN):
        return model
    candidate = getattr(model, "epoch_encoder", None)
    return candidate if isinstance(candidate, MultiRateAsymmetricEpochCNN) else None


def recording_norm_modules(model: nn.Module) -> dict[str, PerRecordingBandNorm]:
    """Return the enabled per-recording norms of ``model`` keyed by modality."""
    encoder = find_multirate_encoder(model)
    if encoder is None:
        return {}
    return encoder.recording_norm_modules()
