"""External EDF preprocessing feeds the unwrapped, unconditioned model directly."""

from __future__ import annotations

from importlib.util import find_spec
from inspect import signature
from pathlib import Path

import numpy as np
import pytest
import torch

from spectra.inference import runtime
from spectra.models import MultiRateAsymmetricEpochCNN, TransformerContextNet
from spectra.preprocessing import edf


@pytest.mark.parametrize("nested", [False, True])
def test_conditioning_metadata_is_rejected(checkpoint_path: Path, nested: bool) -> None:
    data = torch.load(checkpoint_path, weights_only=False)
    kwargs = data["model_config"]["model_kwargs"]
    if nested:
        kwargs = kwargs["multirate_asymmetric_encoder_kwargs"]
    kwargs["recording_conditioning"] = True
    with pytest.raises(runtime.InferenceModelLoadError, match="Recording-conditioned"):
        runtime.load_model_from_checkpoint("unused", torch.device("cpu"), data)


@pytest.mark.parametrize(
    "key",
    [
        "preprocessor.median_",
        "model.preprocessor.iqr_",
        "epoch_encoder.recording_conditioner.attention.weight",
    ],
)
def test_removed_model_state_is_rejected(checkpoint_path: Path, key: str) -> None:
    data = torch.load(checkpoint_path, weights_only=False)
    data["model"][key] = torch.ones(1)
    with pytest.raises(runtime.InferenceModelLoadError, match="unsupported"):
        runtime.load_model_from_checkpoint("unused", torch.device("cpu"), data)


def test_inactive_legacy_metadata_loads_an_unwrapped_model(
    checkpoint_path: Path,
) -> None:
    data = torch.load(checkpoint_path, weights_only=False)
    kwargs = data["model_config"]["model_kwargs"]
    for config in (kwargs, kwargs["multirate_asymmetric_encoder_kwargs"]):
        config.update(
            recording_conditioning=False,
            recording_conditioning_samples=64,
            recording_conditioning_dim=64,
        )
    model, resolved = runtime.load_model_from_checkpoint(
        "unused", torch.device("cpu"), data
    )
    assert type(model) is TransformerContextNet
    assert type(model.epoch_encoder) is MultiRateAsymmetricEpochCNN
    for module in model.modules():
        assert not hasattr(module, "recording_conditioner")
        assert not hasattr(module, "preprocessor")
    assert not hasattr(model, "model")
    assert "recording_conditioning" not in signature(TransformerContextNet).parameters
    assert (
        "recording_conditioning"
        not in signature(MultiRateAsymmetricEpochCNN).parameters
    )
    assert "proc_cfg" not in signature(runtime.load_model_from_checkpoint).parameters
    resolved_kwargs = resolved["model_config"]["model_kwargs"]
    assert "recording_conditioning" not in resolved_kwargs
    assert (
        "recording_conditioning"
        not in resolved_kwargs["multirate_asymmetric_encoder_kwargs"]
    )


@pytest.mark.parametrize("sequential", [False, True])
def test_edf_waveforms_are_preprocessed_once_before_model_inference(
    edf_path: Path,
    checkpoint_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sequential: bool,
) -> None:
    expected = edf.preprocess_edf(str(edf_path))
    data = expected.signals_stacked.transpose(1, 0, 2).reshape(5, -1)
    expected_windows, _ = runtime.create_epoch_batches(
        data, expected.presence_mask, 128, 30, 10
    )
    preprocess_calls = []
    model_inputs = []
    original_preprocess = edf.preprocess_edf
    original_load = runtime.load_model_from_checkpoint

    def capture_preprocess(*args: object, **kwargs: object):
        preprocess_calls.append(args)
        return original_preprocess(*args, **kwargs)

    def capture_load(*args: object, **kwargs: object):
        model, metadata = original_load(*args, **kwargs)
        assert type(model) is TransformerContextNet
        assert not hasattr(model, "preprocessor")
        model.register_forward_pre_hook(
            lambda module, inputs: model_inputs.append(
                inputs[0]["wave"].detach().cpu().clone()
            )
        )
        return model, metadata

    monkeypatch.setattr(edf, "preprocess_edf", capture_preprocess)
    monkeypatch.setattr(runtime, "load_model_from_checkpoint", capture_load)
    result = runtime.score_recording(
        str(edf_path),
        str(checkpoint_path),
        None,
        str(tmp_path),
        device="cpu",
        options=runtime.ScoreOptions(batch_size=2, sequential_loading=sequential),
    )
    assert len(preprocess_calls) == 1
    assert model_inputs
    torch.testing.assert_close(
        torch.cat(model_inputs), expected_windows, rtol=0, atol=0
    )
    assert result["score_window"] == (expected.start_epoch, expected.end_epoch)
    valid = result["epoch_signal_valid"].any(axis=1)
    assert np.isfinite(result["probabilities"]).all()
    np.testing.assert_allclose(
        result["probabilities"][valid].sum(axis=1), 1.0, atol=1e-6
    )


@pytest.mark.parametrize(
    "module",
    [
        "spectra.model.wrapped",
        "spectra.model.recording_conditioning",
        "spectra.models.recording_conditioning",
        "spectra.preprocessing.normalizer",
        "spectra.preprocessing.config",
    ],
)
def test_removed_components_are_not_importable(module: str) -> None:
    assert find_spec(module) is None
