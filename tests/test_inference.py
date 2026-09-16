"""Checkpoint reconstruction and complete EDF-to-export inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from spectra.inference import ScoreOptions, score_recording
from spectra.inference.runtime import load_model_from_checkpoint
from spectra.models import MultiRateAsymmetricEpochCNN, TransformerContextNet


def test_checkpoint_round_trip(checkpoint_path: Path, model_kwargs: dict) -> None:
    saved = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    original = TransformerContextNet(**model_kwargs).eval()
    original.load_state_dict(saved["model"], strict=True)
    reloaded, _ = load_model_from_checkpoint(str(checkpoint_path), torch.device("cpu"))
    assert reloaded.state_dict().keys() == original.state_dict().keys()
    for key, value in original.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(
                reloaded.state_dict()[key], value, rtol=0, atol=0
            )
        else:
            assert reloaded.state_dict()[key] == value
    wave = torch.randn(2, 21, 5, 3840)
    mask = torch.tensor([[1, 1, 1, 0, 1], [1, 0, 1, 1, 1]], dtype=torch.float32)
    with torch.no_grad():
        expected = original({"wave": wave, "presence_mask": mask}).logits
        actual = reloaded({"wave": wave, "presence_mask": mask}).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("strides", [(1, 2, 2, 2), (2, 2, 2, 1)])
def test_encoder_geometry_and_reload(strides: tuple[int, ...]) -> None:
    model = MultiRateAsymmetricEpochCNN(
        widths=(32, 32, 48, 64, 64),
        trunk_strides=strides,
        eeg_band_filters=4,
        emg_filters=6,
        pool_bottleneck=8,
        band_norm_per_recording=True,
        band_norm_max_recordings=8,
    ).eval()
    assert model.final_temporal_len == 240
    rebuilt = MultiRateAsymmetricEpochCNN(**model.get_config()).eval()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    wave = torch.randn(1, 5, 3840)
    with torch.no_grad():
        torch.testing.assert_close(model(wave), rebuilt(wave), rtol=0, atol=0)


@pytest.mark.parametrize("sequential", [True, False])
def test_edf_to_export(
    edf_path: Path,
    checkpoint_path: Path,
    tmp_path: Path,
    sequential: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spectra.inference import runtime

    pinned = []
    original_pin = runtime._pin_recording_band_statistics

    def capture_pin(*args: object, **kwargs: object) -> list:
        modules = original_pin(*args, **kwargs)
        assert modules and all(norm._inference_mean is not None for norm in modules)
        pinned.extend(modules)
        return modules

    monkeypatch.setattr(runtime, "_pin_recording_band_statistics", capture_pin)
    result = score_recording(
        str(edf_path),
        str(checkpoint_path),
        None,
        str(tmp_path),
        device="cpu",
        options=ScoreOptions(batch_size=2, sequential_loading=sequential),
    )
    assert result["n_epochs"] == 14
    assert pinned and all(norm._inference_mean is None for norm in pinned)
    assert result["score_window"] == (1, 13)
    invalid = ~result["epoch_signal_valid"].any(axis=1)
    assert np.flatnonzero(invalid).tolist() == [0, 6, 13]
    assert (result["predictions"][invalid] == -1).all()
    assert not result["probabilities"][invalid].any()
    np.testing.assert_allclose(
        result["probabilities"][~invalid].sum(axis=1), 1.0, atol=1e-6
    )
    provenance = json.loads(Path(result["output_paths"]["preprocessing"]).read_text())
    assert provenance["sampling_rate"] == 128
    assert provenance["rereferencing"] is False
    np.testing.assert_array_equal(
        np.load(result["output_paths"]["epoch_signal_valid"]),
        result["epoch_signal_valid"],
    )


def test_other_encoder_is_rejected(model_kwargs: dict) -> None:
    kwargs = dict(model_kwargs, epoch_encoder_variant="conformer")
    with pytest.raises(ValueError, match="only multirate_asymmetric"):
        TransformerContextNet(**kwargs)


def test_mc_dropout_edf_to_export(
    edf_path: Path, checkpoint_path: Path, tmp_path: Path
) -> None:
    result = score_recording(
        str(edf_path),
        str(checkpoint_path),
        None,
        str(tmp_path),
        device="cpu",
        options=ScoreOptions(batch_size=2, use_mc_dropout=True, mc_samples=3),
    )
    variance = result["mc_probability_variance"]
    assert variance.shape == (14, 5)
    assert np.isfinite(variance).all()
    assert variance.max() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("amp", ["fp32", "bf16"])
def test_cuda_edf_to_export(
    edf_path: Path, checkpoint_path: Path, tmp_path: Path, amp: str
) -> None:
    result = score_recording(
        str(edf_path),
        str(checkpoint_path),
        None,
        str(tmp_path),
        device="cuda",
        options=ScoreOptions(batch_size=2, amp_mode=amp),
    )
    valid = result["epoch_signal_valid"].any(axis=1)
    assert np.isfinite(result["probabilities"]).all()
    np.testing.assert_allclose(
        result["probabilities"][valid].sum(axis=1), 1.0, atol=1e-6
    )
