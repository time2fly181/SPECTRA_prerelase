"""Band normalization must use recording statistics and restore inference state."""

from __future__ import annotations

import pytest
import torch

from spectra.inference.runtime import pinned_recording_band_statistics
from spectra.models import TransformerContextNet


@pytest.mark.parametrize("statistic", ["ema", "robust"])
def test_pinning_changes_logits_and_cleans_up(
    model_kwargs: dict, statistic: str
) -> None:
    kwargs = dict(model_kwargs)
    kwargs["multirate_asymmetric_encoder_kwargs"] = dict(
        kwargs["multirate_asymmetric_encoder_kwargs"],
        band_norm_statistic=statistic,
        band_norm_modalities="eeg,emg",
    )
    model = TransformerContextNet(**kwargs).eval()
    wave = torch.randn(3, 3, 5, 3840)
    before = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    with torch.no_grad():
        unpinned = model(wave).logits
        with pinned_recording_band_statistics(model, wave, torch.device("cpu")):
            pinned = model(wave).logits
            for norm in model.epoch_encoder.recording_norm_modules().values():
                assert norm._inference_mean is not None
                assert torch.isfinite(norm._inference_mean).all()
        assert not torch.equal(unpinned, pinned)
        torch.testing.assert_close(model(wave).logits, unpinned, rtol=0, atol=0)
    for norm in model.epoch_encoder.recording_norm_modules().values():
        assert norm._inference_mean is None
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="forced"):
        with pinned_recording_band_statistics(model, wave, torch.device("cpu")):
            raise RuntimeError("forced")
    assert all(
        norm._inference_mean is None
        for norm in model.epoch_encoder.recording_norm_modules().values()
    )
