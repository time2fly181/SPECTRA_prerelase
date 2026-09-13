"""Signal geometry, direct resampling, masks, and GUI channel-plan parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyedflib
import pytest
from scipy.signal import resample_poly

from spectra.preprocessing.edf import (
    load_edf_signals,
    preprocess_edf,
    select_channel_plan,
)


def test_direct_resampling_and_normalization(edf_path: Path) -> None:
    recording = load_edf_signals(str(edf_path))
    prepared = preprocess_edf(str(edf_path))
    assert prepared.signals_stacked.shape == (12, 5, 3840)
    assert prepared.signals_stacked.dtype == np.float32
    assert (prepared.start_epoch, prepared.end_epoch) == (1, 13)
    assert prepared.original_n_epochs == 14
    assert prepared.presence_mask.tolist() == [1] * 5
    assert not prepared.epoch_signal_valid[5].any()
    assert not prepared.signals_stacked[5].any()
    assert np.isfinite(prepared.signals_stacked).all()
    with pyedflib.EdfReader(str(edf_path)) as reader:
        index = reader.getSignalLabels().index("C4-A1")
        expected_raw = resample_poly(reader.readSignal(index), 1, 2).astype(np.float32)
    c = next(
        i
        for i, source in enumerate(recording.channel_mapping.values())
        if source == "C4-A1"
    )
    np.testing.assert_array_equal(recording.signals[c], expected_raw)
    raw_epochs = expected_raw.reshape(14, 3840)[1:13]
    valid = prepared.epoch_signal_valid[:, c]
    stats_data = raw_epochs[valid].reshape(-1)
    median = np.median(stats_data)
    q1, q3 = np.percentile(stats_data, [25, 75])
    expected = np.clip((raw_epochs - median) / (q3 - q1), -20, 20).astype(np.float32)
    expected[~valid] = 0
    np.testing.assert_allclose(
        prepared.signals_stacked[:, c], expected, rtol=1e-6, atol=1e-6
    )


def test_missing_channels_and_too_short_interval(edf_path: Path) -> None:
    prepared = preprocess_edf(
        str(edf_path), ["NO_CHANNEL", "EEG2", "EOG1", "EOG2", "EMG"]
    )
    assert prepared.presence_mask[0] == 0
    assert not prepared.signals_stacked[:, 0].any()
    assert not prepared.epoch_signal_valid[:, 0].any()
    with pytest.raises(ValueError, match="minimum 10 epochs"):
        preprocess_edf(str(edf_path), start_epoch=1, end_epoch=6)
    with pytest.raises(ValueError, match="Invalid analysis interval"):
        preprocess_edf(str(edf_path), start_epoch=-1)


def test_auto_and_explicit_slots_do_not_duplicate_sources() -> None:
    labels = ["C4-A1", "C3-A2", "LOC-A2", "ROC-A1", "Chin EMG"]
    plan = select_channel_plan(labels, ["C3-A2", "EEG2", "EOG1", "EOG2", "EMG"])
    assert plan[0][1] == "C3-A2"
    assert plan[1][1] == "C4-A1"
    assert len({source for _, source in plan}) == 5
    # Bare electrodes do not become algebraic bipolar derivations.
    plan = select_channel_plan(
        ["F4", "M1", "C4", "M2", "Chin EMG"], ["F4-M1", "C4-M1", "EOG1", "EOG2", "EMG"]
    )
    assert all(
        source is None or source in {"F4", "M1", "C4", "M2", "Chin EMG"}
        for _, source in plan
    )


def test_gui_preview_uses_scoring_plan() -> None:
    from spectra.gui import _build_channel_preview_plan

    labels = ["C4-A1", "C3-A2", "LOC-A2", "ROC-A1", "Chin EMG"]
    plan = select_channel_plan(labels)
    preview = _build_channel_preview_plan(labels, None)
    assert [(slot, detail) for slot, _, detail in preview] == plan


def test_review_waveforms_match_selected_channels(edf_path: Path) -> None:
    from spectra.gui import _load_review_signal_payload

    for first in ("EEG1", "C3-A2"):
        payload = _load_review_signal_payload(
            str(edf_path), [first, "EEG2", "EOG1", "EOG2", "EMG"]
        )
        presets = payload["display_presets"]
        np.testing.assert_array_equal(
            presets["Scoring 5ch"]["signals"], presets["Raw Matched"]["signals"]
        )


@pytest.mark.parametrize("header_error", ["dimension", "start_time"])
def test_edf_header_repair_preserves_source_and_signals(
    edf_path: Path,
    tmp_path: Path,
    header_error: str,
) -> None:
    payload = bytearray(edf_path.read_bytes())
    if header_error == "dimension":
        n_signals = int(payload[252:256])
        offset = 256 + n_signals * 96
        payload[offset : offset + 8] = b"\x80" * 8
    else:
        payload[176:184] = b"99.99.99"
    damaged = tmp_path / "damaged.edf"
    damaged.write_bytes(payload)
    prepared = preprocess_edf(str(damaged))
    expected = preprocess_edf(str(edf_path))
    np.testing.assert_array_equal(prepared.signals_stacked, expected.signals_stacked)
    np.testing.assert_array_equal(
        prepared.epoch_signal_valid, expected.epoch_signal_valid
    )
    assert damaged.read_bytes() == payload
