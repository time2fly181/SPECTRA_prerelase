"""Headless desktop startup and inference-only installation checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication


def test_gui_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from spectra.gui import InferenceGUI, initialize_matplotlib_backend

    monkeypatch.setattr(
        InferenceGUI, "get_settings_path", lambda self: tmp_path / "settings.json"
    )
    (tmp_path / "settings.json").write_text(
        json.dumps({"fs": 256, "context_half": 15, "use_iterative_refinement": True})
    )
    app = QApplication.instance() or QApplication([])
    initialize_matplotlib_backend()
    window = InferenceGUI()
    assert "SPECTRA" in window.windowTitle()
    assert (window.get_score_options().fs, window.get_score_options().context_half) == (
        128,
        10,
    )
    assert not hasattr(window, "fs_spin")
    assert not hasattr(window, "context_half_spin")
    assert not hasattr(window, "iter_refine_check")
    window.show()
    app.processEvents()
    window.close()
    app.processEvents()


def test_training_packages_are_not_installed() -> None:
    assert importlib.util.find_spec("psgstage_train") is None
    assert importlib.util.find_spec("psg_models") is None
    assert importlib.util.find_spec("spectra.train") is None
    assert importlib.util.find_spec("spectra.pretrain") is None
