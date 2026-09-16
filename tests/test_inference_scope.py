"""The release reconstructs only waveform-based multirate context models."""

from __future__ import annotations

from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

from spectra.inference.runtime import (
    InferenceModelLoadError,
    load_model_from_checkpoint,
)
from spectra.models import TransformerContextNet


def load(data: dict, **kwargs: object) -> tuple:
    return load_model_from_checkpoint(
        "synthetic.ckpt", torch.device("cpu"), checkpoint_data=data, **kwargs
    )


@pytest.mark.parametrize(
    "setting,value",
    [
        ("epoch_encoder_variant", "conformer"),
        ("epoch_encoder_variant", "learned_feature_axial"),
        ("use_feature_extraction", True),
        ("use_engineered_features", True),
        ("use_sleepfm_fusion", True),
        ("use_n1_attention", True),
        ("epoch_tokenization", "grid"),
    ],
)
def test_rejects_other_inference_paths(
    checkpoint_path: Path, setting: str, value: object
) -> None:
    data = torch.load(checkpoint_path, weights_only=False)
    data["model_config"]["model_kwargs"][setting] = value
    with pytest.raises(InferenceModelLoadError):
        load(data)


@pytest.mark.parametrize(
    "key", ["feature_extractor.weight", "axial_backbone.weight", "unknown_head.weight"]
)
def test_rejects_unsupported_state_even_with_supported_metadata(
    checkpoint_path: Path, key: str
) -> None:
    data = torch.load(checkpoint_path, weights_only=False)
    data["model"][key] = torch.ones(1)
    with pytest.raises(InferenceModelLoadError):
        load(data)


def test_requires_explicit_architecture(checkpoint_path: Path) -> None:
    data = torch.load(checkpoint_path, weights_only=False)
    data.pop("model_config")
    with pytest.raises(InferenceModelLoadError, match="explicit model_config"):
        load(data)


def test_missing_encoder_weights_fail(checkpoint_path: Path) -> None:
    data = torch.load(checkpoint_path, weights_only=False)
    data["model"].pop("epoch_encoder.multirate_stem.fusion.0.weight")
    with pytest.raises(InferenceModelLoadError, match="Missing keys"):
        load(data)


def test_training_auxiliaries_are_audited_and_do_not_affect_predictions(
    checkpoint_path: Path,
) -> None:
    data = torch.load(checkpoint_path, weights_only=False)
    expected, _ = load(deepcopy(data))
    auxiliary = {
        "source_convention.raw_offsets": torch.randn(2, 5),
        "source_convention._extra_state": {"names": ["a", "b"]},
        "slow_wave_occupancy_head.weight": torch.randn(2, 32),
        "neighbor_head.weight": torch.randn(5, 32),
        "cnn_reconstruction_head.weight": torch.randn(32, 32),
        "mask_token": torch.randn(1, 1, 32),
    }
    data["model"].update(auxiliary)
    data["model"] = {"module._orig_mod.model." + k: v for k, v in data["model"].items()}
    actual, audit = load(data)
    assert len(audit["_inference_load_audit"]["ignored_training_keys"]) == len(
        auxiliary
    )
    assert not hasattr(actual, "source_convention")
    assert not hasattr(actual, "slow_wave_occupancy_head")
    wave = torch.randn(1, 21, 5, 3840)
    with torch.inference_mode():
        torch.testing.assert_close(
            actual(wave).logits, expected(wave).logits, rtol=0, atol=0
        )


def test_averaged_weights_selected_before_reconstruction(checkpoint_path: Path) -> None:
    data = torch.load(checkpoint_path, weights_only=False)
    averaged = deepcopy(data["model"])
    averaged["classifier.head.bias"] = torch.arange(5, dtype=torch.float32)
    data["ema"] = {"module": averaged}
    model, _ = load(data, prefer_averaged=True)
    torch.testing.assert_close(
        model.classifier.head.bias, averaged["classifier.head.bias"], rtol=0, atol=0
    )


def test_saved_fixed_filter_is_restored(checkpoint_path: Path) -> None:
    from spectra.models.anti_alias import KaiserAntiAliasDownsample1D

    data = torch.load(checkpoint_path, weights_only=False)
    original, _ = load(deepcopy(data))
    name, module = next(
        (name, module)
        for name, module in original.named_modules()
        if isinstance(module, KaiserAntiAliasDownsample1D)
    )
    saved_filter = module.kernel.clone() * 0.95
    data["model"][name + ".kernel"] = saved_filter
    actual, _ = load(data)
    torch.testing.assert_close(
        actual.get_submodule(name).kernel, saved_filter, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "extra",
    [
        {
            "context_attention_mode": "relative_full",
            "context_readout_mode": "relative_multihead",
            "ffn_activation": "swiglu",
        },
        {"classifier_head": "linear"},
        {
            "use_per_position_head": True,
            "head_norm": "rmsnorm",
            "head_use_local_mix": True,
        },
        {"use_confidence_head": True, "learnable_temperature": True},
    ],
)
def test_supported_context_configurations_round_trip(
    model_kwargs: dict, extra: dict
) -> None:
    kwargs = dict(model_kwargs, **extra)
    expected = TransformerContextNet(**kwargs).eval()
    actual, _ = load(
        {
            "model": expected.state_dict(),
            "model_config": {
                "model_type": "TransformerContextNet",
                "model_kwargs": kwargs,
            },
        }
    )
    wave = torch.randn(1, 21, 5, 3840)
    with torch.inference_mode():
        for predict_all in (False, True):
            original = expected(wave, predict_all=predict_all, return_attention=True)
            reloaded = actual(wave, predict_all=predict_all, return_attention=True)
            torch.testing.assert_close(reloaded.logits, original.logits, rtol=0, atol=0)
            torch.testing.assert_close(
                reloaded.attention_weights, original.attention_weights, rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "name",
    [
        "feature_extraction",
        "yasa_features",
        "enhanced_sequential_fusion",
        "sleepfm_inspired_modules",
        "axial_feature_time_transformer",
        "epoch_patch_grid",
        "patch_tokenization",
        "attention",
        "channel_config",
        "dilated_asymmetric_epoch_cnn",
        "learned_feature_bank_cnn",
        "checkpointing",
    ],
)
def test_removed_modules_are_not_shipped(name: str) -> None:
    assert find_spec(f"spectra.models.{name}") is None
