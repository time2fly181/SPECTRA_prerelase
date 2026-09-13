"""Regression coverage for public recording-normalized inference entry points."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from spectra.inference import runtime
from spectra.models.multirate_asymmetric_epoch_cnn import MultiRateAsymmetricEpochCNN


@pytest.fixture
def recording() -> tuple[None, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    wave = (
        torch.randn(8, 5, 3840)
        * torch.tensor([1.0, 2.0, 3.0, 8.0, 2.0, 4.0, 1.0, 2.0])[:, None, None]
    )
    mask = torch.ones(8, 5, dtype=torch.bool)
    mask[4, 0] = False
    mask[6] = False
    wave[4, 0] = 500
    wave[6] = 1000
    return None, wave, mask


class EnvelopeProbe(nn.Module):
    """Expose normalization through logits while using the real EEG envelope."""

    def __init__(self, statistic: str = "ema") -> None:
        super().__init__()
        self.epoch_encoder = MultiRateAsymmetricEpochCNN(
            in_ch=5,
            band_norm_per_recording=True,
            band_norm_statistic=statistic,
            band_norm_max_recordings=16,
        )
        self.outputs: list[torch.Tensor] = []

    def forward(self, inputs: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        wave = inputs["wave"]
        if wave.ndim == 4:
            wave = wave[:, wave.shape[1] // 2]
        branch = self.epoch_encoder.multirate_stem.eeg_branch
        assert branch is not None and branch.band_recording_norm is not None
        envelope = branch.band_envelope(wave[:, :2])
        normalized = branch.band_recording_norm(envelope, inputs.get("recording_index"))
        score = normalized.mean(dim=(1, 2))
        logits = torch.stack([score * 0, score * 0, -score, score, score * 0], dim=1)
        self.outputs.append(logits.detach().cpu())
        return logits


@pytest.mark.parametrize("sequential", [False, True])
def test_public_runtime_pins_normalization(recording: Any, sequential: bool) -> None:
    _, wave, mask = recording
    model = EnvelopeProbe().eval()
    options = runtime.ScoreOptions(batch_size=2, context_half=0, amp_mode="off")
    windows = wave[:, None]
    with runtime.pinned_recording_band_statistics(
        model,
        windows,
        torch.device("cpu"),
        presence_mask=mask,
        epoch_valid=mask.any(dim=1),
    ):
        expected = runtime.run_inference(
            model,
            windows,
            torch.ones(5),
            torch.device("cpu"),
            options,
            epoch_channel_mask=mask[:, None],
        )
    if sequential:
        actual = runtime.run_inference_sequential(
            model,
            wave.permute(1, 0, 2).reshape(5, -1).numpy(),
            np.ones(5),
            128,
            30,
            0,
            torch.device("cpu"),
            options,
            epoch_channel_valid=mask.numpy(),
        )
    else:
        actual = runtime.run_inference(
            model,
            windows,
            torch.ones(5),
            torch.device("cpu"),
            options,
            epoch_channel_mask=mask[:, None],
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    assert model.epoch_encoder.recording_norm_modules()["eeg"]._inference_mean is None


def test_reasoning_keeps_whole_recording_statistics(
    recording: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, wave, mask = recording
    model = EnvelopeProbe().eval()
    norm = model.epoch_encoder.recording_norm_modules()["eeg"]

    def postprocess(logits: np.ndarray, **kwargs: Any) -> dict[str, np.ndarray]:
        assert norm._inference_mean is not None, "refinement lost recording statistics"
        return {"logits": logits}

    monkeypatch.setattr(runtime, "postprocess_logits_with_reasoning", postprocess)
    runtime.infer_with_reasoning(
        model,
        wave[:, None],
        torch.ones(5),
        torch.device("cpu"),
        runtime.ScoreOptions(batch_size=2, context_half=0, amp_mode="off"),
        epoch_channel_mask=mask[:, None],
    )
    assert norm._inference_mean is None


def test_nested_pin_restores_previous_statistics(recording: Any) -> None:
    _, wave, mask = recording
    model = EnvelopeProbe().eval()
    norm = model.epoch_encoder.recording_norm_modules()["eeg"]
    with runtime.pinned_recording_band_statistics(
        model, wave[:, None], torch.device("cpu"), presence_mask=mask
    ):
        outer = norm._inference_mean
        with pytest.raises(RuntimeError, match="probe failure"):
            with runtime.pinned_recording_band_statistics(
                model, wave[:, None] * 2, torch.device("cpu"), presence_mask=mask
            ):
                raise RuntimeError("probe failure")
        assert norm._inference_mean is outer
