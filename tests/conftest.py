"""Synthetic EDF and checkpoint fixtures; no patient data or pretrained weights."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyedflib
import pytest
import torch

from spectra.models import TransformerContextNet


@pytest.fixture(scope="session", autouse=True)
def limit_cpu_threads() -> None:
    """Keep tiny CPU inference checks fast and repeatable."""
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def edf_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create mixed-rate signals, invalid exterior epochs, and an interior gap."""
    path = tmp_path_factory.mktemp("edf") / "synthetic.edf"
    names = ["C4-A1", "C3-A2", "LOC-A2", "ROC-A1", "Chin EMG", "ECG"]
    rates = [256, 128, 200, 100, 512, 32]
    rng = np.random.default_rng(19)
    signals = []
    headers = []
    for c, (name, rate) in enumerate(zip(names, rates, strict=True)):
        t = np.arange(14 * 30 * rate) / rate
        signal = 20 * np.sin(2 * np.pi * (0.7 + c) * t) + rng.normal(0, 2, t.size)
        signal += 3 * np.sin(2 * np.pi * 0.01 * t)
        for epoch in (0, 6, 13):
            signal[
                max(0, epoch * 30 * rate - rate) : min(
                    signal.size, (epoch + 1) * 30 * rate + rate
                )
            ] = 0
        signals.append(signal)
        headers.append(
            {
                "label": name,
                "dimension": "uV",
                "sample_frequency": rate,
                "physical_min": -500,
                "physical_max": 500,
                "digital_min": -32767,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            }
        )
    with pyedflib.EdfWriter(
        str(path), len(names), file_type=pyedflib.FILETYPE_EDFPLUS
    ) as writer:
        writer.setSignalHeaders(headers)
        writer.writeSamples(signals)
    return path


@pytest.fixture(scope="session")
def model_kwargs() -> dict:
    """Use the production 128 Hz geometry with a small model width."""
    return {
        "in_ch": 5,
        "time_len": 3840,
        "num_classes": 5,
        "epoch_encoder_variant": "multirate_asymmetric",
        "context_epochs": 3,
        "d_model": 32,
        "nhead": 2,
        "num_layers": 1,
        "dim_ff": 64,
        "cnn_dropout": 0.0,
        "head_dropout": 0.0,
        "classifier_dropout": 0.0,
        "cnn_widths": (32, 32, 48, 64, 64),
        "multirate_asymmetric_encoder_kwargs": {
            "eeg_band_filters": 4,
            "emg_filters": 6,
            "pool_bottleneck": 8,
            "band_norm_per_recording": True,
            "band_norm_max_recordings": 8,
            "band_norm_warmup_updates": 1,
            "band_norm_statistic": "robust",
        },
    }


@pytest.fixture(scope="session")
def checkpoint_path(
    tmp_path_factory: pytest.TempPathFactory, model_kwargs: dict
) -> Path:
    """Serialize the same model_config/state schema used by PSGStage."""
    path = tmp_path_factory.mktemp("checkpoints") / "synthetic.ckpt"
    torch.manual_seed(22)
    model = TransformerContextNet(**model_kwargs).eval()
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": {
                "model_type": "TransformerContextNet",
                "model_kwargs": model_kwargs,
                "inference_metadata": {"context_half": 1},
            },
        },
        path,
    )
    return path
