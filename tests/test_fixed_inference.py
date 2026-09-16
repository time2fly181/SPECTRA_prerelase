"""The release exposes only the fixed EDF geometry and supported sampling path."""

from dataclasses import fields
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

from spectra.cli import main
from spectra.inference import runtime
from spectra.models import TransformerContextNet
from spectra.models.anti_alias import KaiserAntiAliasDownsample1D


@pytest.mark.parametrize("name,value", [("fs", 256), ("context_half", 15)])
def test_geometry_cannot_be_configured(name: str, value: int) -> None:
    options = runtime.ScoreOptions()
    assert (options.fs, options.context_half) == (128, 10)
    assert name not in {field.name for field in fields(options)}
    with pytest.raises(TypeError):
        runtime.ScoreOptions(**{name: value})
    with pytest.raises(AttributeError):
        setattr(options, name, value)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "fs", 256),
        ("model", "context_half", 15),
        ("model", "context_epochs", 31),
        ("model", "time_len", 7680),
        ("encoder", "fs", 256),
        ("metadata", "context_half", 15),
        ("preprocessing", "target_fs", 256),
    ],
)
def test_conflicting_checkpoint_geometry_is_rejected(
    checkpoint_path: Path, section: str, key: str, value: int
) -> None:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint["model_config"]
    sections = {
        "model": config["model_kwargs"],
        "encoder": config["model_kwargs"]["multirate_asymmetric_encoder_kwargs"],
        "metadata": config["inference_metadata"],
        "preprocessing": checkpoint.setdefault("preprocessing_config", {}).setdefault(
            "resampling", {}
        ),
    }
    sections[section][key] = value
    with pytest.raises(runtime.InferenceModelLoadError, match="requires fixed"):
        runtime.load_model_from_checkpoint("unused", torch.device("cpu"), checkpoint)


@pytest.mark.parametrize("active_state", [False, True])
def test_refinement_checkpoint_is_rejected(
    checkpoint_path: Path, active_state: bool
) -> None:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if active_state:
        checkpoint["model"]["recurrent_refiner.configured_steps"] = torch.tensor(2)
    else:
        checkpoint["model_config"]["model_kwargs"]["recurrent_refinement_steps"] = 2
    with pytest.raises(runtime.InferenceModelLoadError, match="refinement"):
        runtime.load_model_from_checkpoint("unused", torch.device("cpu"), checkpoint)


def test_inactive_refinement_metadata_and_kaiser_filters(checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["model_config"]["model_kwargs"]["recurrent_refinement_steps"] = 0
    model, _ = runtime.load_model_from_checkpoint(
        "unused", torch.device("cpu"), checkpoint
    )
    assert not hasattr(model, "recurrent_refiner")
    assert any(
        isinstance(module, KaiserAntiAliasDownsample1D) for module in model.modules()
    )


@pytest.mark.parametrize(
    "module", ["spectra.models.blur_pool", "spectra.inference.tta"]
)
def test_removed_modules_are_not_importable(module: str) -> None:
    assert find_spec(module) is None


@pytest.mark.parametrize("option", ["tta_passes", "tta", "recurrent_refinement_steps"])
def test_removed_options_are_rejected(option: str) -> None:
    with pytest.raises(TypeError):
        runtime.ScoreOptions(**{option: 2})
    assert not hasattr(runtime, "ReasoningOptions")


@pytest.mark.parametrize("flag", ["--context-half", "--fs", "--tta-passes"])
def test_cli_has_no_override(flag: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main([flag, "15"])
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("key,value", [("context_epochs", 3), ("fs", 256)])
def test_direct_model_rejects_geometry_override(
    model_kwargs: dict, key: str, value: int
) -> None:
    with pytest.raises(ValueError, match="SPECTRA requires"):
        TransformerContextNet(**{**model_kwargs, key: value})


def test_direct_forward_rejects_short_context(model_kwargs: dict) -> None:
    model = TransformerContextNet(**model_kwargs).eval()
    with pytest.raises(ValueError, match="21 context epochs"):
        model(torch.zeros(1, 3, 5, 3840))
