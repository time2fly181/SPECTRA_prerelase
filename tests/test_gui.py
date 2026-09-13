"""Headless desktop startup and inference-only installation checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication


def test_gui_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from spectra.gui import InferenceGUI, initialize_matplotlib_backend

    monkeypatch.setattr(
        InferenceGUI, "get_settings_path", lambda self: tmp_path / "settings.json"
    )
    app = QApplication.instance() or QApplication([])
    initialize_matplotlib_backend()
    window = InferenceGUI()
    assert "SPECTRA" in window.windowTitle()
    window.show()
    app.processEvents()
    window.close()
    app.processEvents()


def test_training_packages_are_not_installed() -> None:
    assert importlib.util.find_spec("psgstage_train") is None
    assert importlib.util.find_spec("psg_models") is None
    assert importlib.util.find_spec("spectra.train") is None
    assert importlib.util.find_spec("spectra.pretrain") is None
