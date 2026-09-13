"""Locate per-recording band-norm modules behind training/inference wrappers.

Every stage wraps the multirate encoder differently: supervised training and
inference hold it as ``model.epoch_encoder``, temporal pretraining as
``model.cnn``, SupCon as the bare encoder, and ``torch.compile`` inserts an
``OptimizedModule`` (``_orig_mod``) at either level. The statistics table,
the inference pin and the loop-time guards must all reach the same modules, so
this is the single place that knows how.
"""

from __future__ import annotations

import torch.nn as nn

from spectra.models.multirate_asymmetric_epoch_cnn import (
    MultiRateAsymmetricEpochCNN,
    PerRecordingBandNorm,
)

__all__ = ["find_multirate_encoder", "recording_norm_modules"]

_ENCODER_ATTR_PREFIXES = ("epoch_encoder.", "cnn.")


def _unwrap_compiled(module: nn.Module) -> nn.Module:
    return getattr(module, "_orig_mod", module)


def find_multirate_encoder(model: nn.Module) -> MultiRateAsymmetricEpochCNN | None:
    """Return the ``MultiRateAsymmetricEpochCNN`` inside ``model``, if any.

    Looks through ``torch.compile`` wrappers and the ``epoch_encoder`` /
    ``cnn`` attributes used by the context model and the temporal pretrainer.
    """
    base = _unwrap_compiled(model)
    if isinstance(base, MultiRateAsymmetricEpochCNN):
        return base
    for attr in ("epoch_encoder", "cnn"):
        candidate = getattr(base, attr, None)
        if candidate is None:
            continue
        candidate = _unwrap_compiled(candidate)
        if isinstance(candidate, MultiRateAsymmetricEpochCNN):
            return candidate
    return None


def recording_norm_modules(model: nn.Module) -> dict[str, PerRecordingBandNorm]:
    """Return the enabled per-recording norms of ``model`` keyed by modality."""
    encoder = find_multirate_encoder(model)
    if encoder is None:
        return {}
    return encoder.recording_norm_modules()
