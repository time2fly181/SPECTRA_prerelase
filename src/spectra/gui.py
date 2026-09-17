#!/usr/bin/env python3
"""Desktop EDF scoring, waveform review, reference comparison, and export.

Scoring delegates to ``spectra.inference.runtime``. Display filters affect only
waveform viewing. Import configures Qt rendering; launch with ``spectra-gui``
or the repository's ``inference_gui.py`` entry point.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

if TYPE_CHECKING:
    from spectra.inference import ScoreOptions

# Set environment variables BEFORE importing Qt to prevent graphics conflicts
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""  # Let Qt auto-detect
os.environ["QT_OPENGL"] = "software"  # Use software OpenGL to avoid GPU conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
    "CUDA_VISIBLE_DEVICES", "0"
)  # Ensure CUDA device is set

from spectra.inference_gui.state import (  # noqa: E402
    InferenceRunState,
    ProjectState,
    ReviewSessionState,
)
from spectra.inference_gui.tabs import (  # noqa: E402
    ExportTab,
    ReviewTab,
    RunTab,
    SetupTab,
)

try:
    import matplotlib  # pyright: ignore[reportMissingModuleSource]

    matplotlib.use("Agg")  # Use non-GUI backend first to prevent conflicts
    import matplotlib.figure  # pyright: ignore[reportMissingModuleSource]

    # CRITICAL: Do NOT import Qt backends here - they cause SIGSEGV on macOS before QApplication init
except ImportError:
    print("Error: matplotlib is required for hypnogram visualization.")
    print("Install with: pip install matplotlib")
    sys.exit(1)

try:
    from PySide6.QtCore import Qt, QThread, QTimer, Signal  # noqa: I001
    from PySide6.QtGui import (  # noqa: I001
        QColor,
        QFont,
        QKeySequence,
        QShortcut,
        QTextCursor,
        QUndoCommand,
        QUndoStack,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMenu,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSlider,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStyle,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWhatsThis,
        QWidget,
    )  # noqa: I001
except ImportError:
    print("Error: PySide6 is required for the GUI.")
    print("Install with: pip install PySide6")
    sys.exit(1)

try:
    import pyqtgraph as pg  # pyright: ignore[reportMissingModuleSource]
except ImportError:
    pg = None

# These will be set in main() after QApplication init
FigureCanvas = None
NavigationToolbar = None
HypnogramCanvas = None

DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.65
HIGH_CONFIDENCE_THRESHOLD = 0.80
MIN_CONFIDENCE_THRESHOLD = 0.30
MAX_CONFIDENCE_THRESHOLD = HIGH_CONFIDENCE_THRESHOLD

# Per-epoch uncertainty arrays the backend attaches to a scoring result. The
# flag_* arrays are always present.
_UNCERTAINTY_FLAG_KEYS: tuple[str, ...] = (
    "flag_margin",
    "flag_entropy",
    "flag_maxprob",
)

STAGE_LABELS = ["W", "N1", "N2", "N3", "REM"]

# AASM manual-scoring hotkeys -> stage index. W/1/2/3/R is the clinical scoring
# layout; 0 and 4 are digit aliases for Wake and REM.
SCORING_KEY_TO_STAGE: dict[str, int] = {
    "W": 0,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "R": 4,
    "4": 4,
}
STAGE_DISPLAY_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}
STAGE_DETAILS_NAMES = {
    "W": "Wake",
    "N1": "N1 (Light)",
    "N2": "N2 (Light)",
    "N3": "N3 (Deep)",
    "REM": "REM",
}
CONFIDENCE_BAND_COLORS = {
    "high": "#43A047",
    "medium": "#FFA726",
    "low": "#E53935",
}
CONFIDENCE_BAND_LABELS = {
    "high": "High confidence",
    "medium": "Medium confidence",
    "low": "Low confidence",
}
CHANNEL_LAYOUT_SLOTS: tuple[tuple[str, str], ...] = (
    ("EEG1", "eeg"),
    ("EEG2", "eeg"),
    ("EOG1", "eog"),
    ("EOG2", "eog"),
    ("EMG", "emg"),
)
CHANNEL_LAYOUT_SLOT_NAMES = [slot for slot, _ in CHANNEL_LAYOUT_SLOTS]
CHANNEL_LAYOUT_MODALITY_BY_SLOT = {
    slot: modality for slot, modality in CHANNEL_LAYOUT_SLOTS
}


def _slot_modality(label: str) -> str:
    """Map a channel-plan slot label to a modality bucket."""
    up = label.strip().upper()
    if up.startswith("EEG"):
        return "eeg"
    if up.startswith("EOG"):
        return "eog"
    if up.startswith("EMG"):
        return "emg"
    if up.startswith("ECG"):
        return "ecg"
    return "eeg"


SIGNAL_REVIEW_PAGE_DURATIONS = [30, 60, 120, 300]
# Inter-channel spacing multipliers. Smaller = channels packed tighter, so normal
# waves look larger and big deflections overlap (cross over) their neighbours, like
# Compumedics Profusion. Larger = more separation, less overlap.
SIGNAL_REVIEW_SPACING_PRESETS = {
    "Tight": 0.8,
    "Normal": 1.0,
    "Loose": 1.4,
}
DEFAULT_SIGNAL_REVIEW_PAGE_SEC = 30
# Clinical display sensitivities (µV per division). Tuned for sleep morphology so
# spindles/K-complexes/delta are legible rather than swamped by coarse gain.
DEFAULT_SIGNAL_REVIEW_EEG_EOG_UV_PER_DIV = 50.0
DEFAULT_SIGNAL_REVIEW_EMG_UV_PER_DIV = 75.0
# 1 mV/div. Channels are converted to µV on load, so a normal 1–2 mV QRS complex
# fills ~1–2 divisions — matching clinical ECG display rather than the old 5 mV/div
# that rendered the trace almost flat.
SIGNAL_REVIEW_ECG_UV_PER_DIV = 1000.0
SIGNAL_REVIEW_UV_PER_DIV_MIN = 5.0
SIGNAL_REVIEW_UV_PER_DIV_MAX = 1000.0
SIGNAL_REVIEW_UV_PER_DIV_STEP = 5.0
LEGACY_SIGNAL_REVIEW_GAIN_TO_UV_PER_DIV = {
    "Low": 100.0,
    "Medium": 50.0,
    "High": 30.0,
}
DEFAULT_SIGNAL_REVIEW_SPACING = "Normal"
DEFAULT_SIGNAL_REVIEW_PRESET = "Scoring 5ch"
DEFAULT_SIGNAL_REVIEW_PLOT_MIN_HEIGHT = 760
DEFAULT_SIGNAL_REVIEW_QUEUE_VISIBLE = False
DEFAULT_SIGNAL_REVIEW_PLOT_BASE_HEIGHT = 60
DEFAULT_SIGNAL_REVIEW_PLOT_HEIGHT_PER_CHANNEL = 110
# Baseline vertical gap between channel traces, in display divisions. A feature whose
# peak equals the µV/div sensitivity fills one division; with a gap of ~1.3 a normal
# wave nearly fills its row, and large deflections overlap (cross over) neighbours
# because traces share one un-clipped plotting field.
SIGNAL_REVIEW_CHANNEL_GAP_DIVS = 1.3

# Classic-light (Profusion-style) palette.
SIGNAL_REVIEW_BACKGROUND_COLOR = "#FFFFFF"
SIGNAL_REVIEW_PANEL_COLOR = "#F4F4F2"
SIGNAL_REVIEW_AXIS_TEXT_COLOR = "#1F2933"
SIGNAL_REVIEW_AXIS_LINE_COLOR = "#8A9099"
SIGNAL_REVIEW_GRID_COLOR = "#DCDCDC"
SIGNAL_REVIEW_GRID_MAJOR_COLOR = "#9AA0A6"
SIGNAL_REVIEW_TRACE_COLOR = "#10161C"
SIGNAL_REVIEW_TRACE_ACCENT_COLOR = "#15407A"
SIGNAL_REVIEW_MISSING_TRACE_COLOR = "#AEB4BB"
SIGNAL_REVIEW_CALIBRATION_COLOR = "#444444"
SIGNAL_REVIEW_EPOCH_TINT = (76, 175, 80, 40)
StageInput = int | str | np.integer | None


class InferenceCancelled(RuntimeError):
    """Raised inside the worker thread when the user requests cancellation."""


@dataclass(frozen=True)
class DisplayFilterSpec:
    """Per-modality display-filter settings for the PSG signal reader.

    Attributes:
        low_cut: High-pass corner in Hz (removes DC/drift), or ``None``.
        high_cut: Low-pass corner in Hz (removes HF noise), or ``None``.
        notch_hz: Mains-notch frequency in Hz (50/60), or ``None`` to disable.
    """

    low_cut: float | None = None
    high_cut: float | None = None
    notch_hz: float | None = None


SIGNAL_REVIEW_FILTER_ORDER = 4
SIGNAL_REVIEW_NOTCH_Q = 30.0
SIGNAL_REVIEW_NOTCH_OPTIONS: tuple[float | None, ...] = (None, 50.0, 60.0)
DEFAULT_SIGNAL_REVIEW_NOTCH_HZ = 60.0

# AASM-inspired defaults applied on load so morphology is immediately legible.
DEFAULT_DISPLAY_FILTERS: dict[str, DisplayFilterSpec] = {
    "eeg": DisplayFilterSpec(0.3, 35.0, DEFAULT_SIGNAL_REVIEW_NOTCH_HZ),
    "eog": DisplayFilterSpec(0.3, 35.0, DEFAULT_SIGNAL_REVIEW_NOTCH_HZ),
    "emg": DisplayFilterSpec(10.0, 100.0, DEFAULT_SIGNAL_REVIEW_NOTCH_HZ),
    "ecg": DisplayFilterSpec(0.3, 70.0, DEFAULT_SIGNAL_REVIEW_NOTCH_HZ),
}
# Selectable low-cut / high-cut menu values (Hz) for the reader filter controls.
SIGNAL_REVIEW_LOW_CUT_OPTIONS: tuple[float | None, ...] = (
    None,
    0.1,
    0.3,
    0.5,
    1.0,
    5.0,
    10.0,
)
SIGNAL_REVIEW_HIGH_CUT_OPTIONS: tuple[float | None, ...] = (
    None,
    15.0,
    35.0,
    50.0,
    70.0,
    100.0,
)


def _stage_to_int(stage: StageInput) -> int | None:
    """Convert numeric/textual stage identifiers to a stage index."""
    if isinstance(stage, np.integer):
        stage = int(cast(np.integer, stage))

    if isinstance(stage, int):
        if 0 <= stage < len(STAGE_LABELS):
            return stage
        return None

    normalized = _normalize_stage_label(stage)
    try:
        return STAGE_LABELS.index(normalized)
    except ValueError:
        return None


def _coerce_manual_score_overrides(
    raw_overrides: Any,
    *,
    n_epochs: int | None = None,
) -> dict[int, int]:
    """Normalize persisted/manual epoch overrides into a stable dict."""
    if not isinstance(raw_overrides, dict):
        return {}

    overrides: dict[int, int] = {}
    for raw_epoch, raw_stage in raw_overrides.items():
        try:
            epoch_idx = int(raw_epoch)
        except (TypeError, ValueError):
            continue
        stage_idx = _stage_to_int(cast(StageInput, raw_stage))
        if stage_idx is None or epoch_idx < 0:
            continue
        if n_epochs is not None and epoch_idx >= n_epochs:
            continue
        overrides[epoch_idx] = stage_idx
    return overrides


def _apply_manual_score_overrides(
    predictions: np.ndarray | None,
    overrides: dict[int, int] | None,
) -> np.ndarray | None:
    """Return predictions with manual stage overrides applied."""
    if predictions is None:
        return None

    effective = np.asarray(predictions, dtype=np.int64).copy()
    if not overrides:
        return effective

    for epoch_idx, stage_idx in overrides.items():
        if 0 <= int(epoch_idx) < len(effective) and 0 <= int(stage_idx) < len(
            STAGE_LABELS
        ):
            effective[int(epoch_idx)] = int(stage_idx)
    return effective


def _normalize_stage_label(stage: StageInput) -> str:
    """Normalize numeric or textual stage identifiers to GUI stage labels."""
    if isinstance(stage, np.integer):
        stage = int(cast(np.integer, stage))

    if isinstance(stage, int):
        if 0 <= stage < len(STAGE_LABELS):
            return STAGE_LABELS[stage]
        return str(stage)

    if stage is None:
        return "?"

    stage_text = str(stage).strip()
    if not stage_text:
        return "?"

    normalized = stage_text.upper()
    aliases = {
        "WAKE": "W",
        "S0": "W",
        "S1": "N1",
        "S2": "N2",
        "S3": "N3",
        "R": "REM",
    }
    return aliases.get(normalized, normalized)


def _format_elapsed_time_hms(time_sec: int) -> str:
    """Format elapsed time in HH:MM:SS."""
    hours = int(time_sec // 3600)
    minutes = int((time_sec % 3600) // 60)
    seconds = int(time_sec % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_elapsed_time_hms_ms(time_sec: float) -> str:
    """Format elapsed time with tenths precision for cursor readouts."""
    time_sec = max(0.0, float(time_sec))
    whole_seconds = int(time_sec)
    tenths = int(round((time_sec - whole_seconds) * 10.0))
    if tenths >= 10:
        whole_seconds += 1
        tenths = 0
    return f"{_format_elapsed_time_hms(whole_seconds)}.{tenths}"


def _resolve_confidences(
    probabilities: np.ndarray | None,
    confidences: np.ndarray | None,
) -> np.ndarray | None:
    """Use provided confidences when available, otherwise derive from probabilities."""
    if confidences is not None:
        return np.asarray(confidences, dtype=np.float32)
    if probabilities is None:
        return None
    return np.max(np.asarray(probabilities), axis=-1).astype(np.float32, copy=False)


def _get_confidence_band(
    confidence: float,
    threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> str:
    """Classify a confidence value into high / medium / low bands."""
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if confidence >= threshold:
        return "medium"
    return "low"


def _summarize_confidences(
    confidences: np.ndarray | None,
    threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> dict[str, float | int] | None:
    """Compute confidence summary statistics using the shared GUI policy."""
    if confidences is None or len(confidences) == 0:
        return None

    confidence_array = np.asarray(confidences, dtype=np.float32)
    high_mask = confidence_array >= HIGH_CONFIDENCE_THRESHOLD
    medium_mask = (confidence_array >= threshold) & ~high_mask
    low_mask = confidence_array < threshold

    n_epochs = len(confidence_array)
    return {
        "mean": float(np.mean(confidence_array)),
        "median": float(np.median(confidence_array)),
        "min": float(np.min(confidence_array)),
        "max": float(np.max(confidence_array)),
        "std": float(np.std(confidence_array)),
        "high_count": int(np.sum(high_mask)),
        "medium_count": int(np.sum(medium_mask)),
        "low_count": int(np.sum(low_mask)),
        "pct_high": float(np.mean(high_mask) * 100.0),
        "pct_medium": float(np.mean(medium_mask) * 100.0),
        "pct_low": float(np.mean(low_mask) * 100.0),
        "n_below_threshold": int(np.sum(low_mask)),
        "n_epochs": n_epochs,
        "threshold": float(threshold),
    }


def _build_epoch_confidence_records(
    predictions: np.ndarray,
    *,
    final_predictions: np.ndarray | None = None,
    manual_overrides: dict[int, int] | None = None,
    epoch_sec: int = 30,
    probabilities: np.ndarray | None = None,
    confidences: np.ndarray | None = None,
    threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    flag_scores: dict[str, np.ndarray] | None = None,
    uncertainty_metric: str = "entropy",
) -> list[dict[str, Any]]:
    """Build one confidence-review record per epoch.

    When ``flag_scores`` carries the backend uncertainty-flag arrays
    (``flag_margin``/``flag_entropy``/``flag_maxprob``) each record gains an
    ``uncertainty_score`` (higher = more uncertain) derived via
    :func:`spectra.review.uncertainty.combine_flag_score`. When the flags
    are absent the score falls back to ``1 - confidence`` and ``flag_source`` is
    set to ``"confidence"`` so callers can note the degraded mode.
    """
    prediction_values = np.asarray(predictions, dtype=np.int64)
    final_prediction_values = (
        np.asarray(final_predictions, dtype=np.int64)
        if final_predictions is not None
        and len(final_predictions) == len(prediction_values)
        else prediction_values
    )
    normalized_overrides = _coerce_manual_score_overrides(
        manual_overrides,
        n_epochs=len(prediction_values),
    )
    confidence_values = _resolve_confidences(probabilities, confidences)
    probability_values = (
        np.asarray(probabilities, dtype=np.float32)
        if probabilities is not None
        else None
    )

    # Per-epoch uncertainty score for the triage worklist. Prefer the backend
    # flag arrays; otherwise fall back to (1 - confidence).
    uncertainty_values: np.ndarray | None = None
    flag_source = "confidence"
    if flag_scores:
        from spectra.review.uncertainty import combine_flag_score

        uncertainty_values = combine_flag_score(
            margin=flag_scores.get("flag_margin"),
            entropy=flag_scores.get("flag_entropy"),
            maxprob=flag_scores.get("flag_maxprob"),
            metric=uncertainty_metric,
        )
        if uncertainty_values is not None:
            flag_source = "flags"
    if uncertainty_values is None and confidence_values is not None:
        uncertainty_values = 1.0 - np.asarray(confidence_values, dtype=np.float64)

    records: list[dict[str, Any]] = []
    for index, pred_idx in enumerate(prediction_values):
        final_stage_idx = int(final_prediction_values[index])
        time_sec = index * epoch_sec
        confidence = (
            float(confidence_values[index])
            if confidence_values is not None and index < len(confidence_values)
            else None
        )
        uncertainty = (
            float(uncertainty_values[index])
            if uncertainty_values is not None and index < len(uncertainty_values)
            else None
        )
        probs = (
            probability_values[index]
            if probability_values is not None and index < len(probability_values)
            else None
        )
        band = (
            _get_confidence_band(confidence, threshold)
            if confidence is not None
            else None
        )
        next_stage = None
        next_stage_probability = None
        if probs is not None and len(probs) == len(STAGE_LABELS):
            alt_probs = np.asarray(probs, dtype=np.float32).copy()
            if 0 <= pred_idx < len(alt_probs):
                alt_probs[pred_idx] = -np.inf
            next_stage_idx = int(np.argmax(alt_probs))
            if np.isfinite(float(alt_probs[next_stage_idx])):
                next_stage = STAGE_LABELS[next_stage_idx]
                next_stage_probability = float(probs[next_stage_idx])
        records.append(
            {
                "epoch": index + 1,
                "epoch_index": index,
                "time_sec": time_sec,
                "time_hms": _format_elapsed_time_hms(time_sec),
                "stage": STAGE_LABELS[int(pred_idx)],
                "stage_numeric": int(pred_idx),
                "model_stage": STAGE_LABELS[int(pred_idx)],
                "model_stage_numeric": int(pred_idx),
                "final_stage": STAGE_LABELS[final_stage_idx],
                "final_stage_numeric": final_stage_idx,
                "confidence": confidence,
                "uncertainty_score": uncertainty,
                "flag_source": flag_source,
                "band": band,
                "low_confidence": band == "low",
                "probabilities": probs,
                "next_stage": next_stage,
                "next_stage_probability": next_stage_probability,
                "manual_override": index in normalized_overrides,
                "override_stage": (
                    STAGE_LABELS[normalized_overrides[index]]
                    if index in normalized_overrides
                    else None
                ),
            }
        )
    return records


def _repair_edf_physical_dimensions(edf_path: str) -> str | None:
    """Repair EDF files with empty Physical Dimension fields into a temp copy."""
    with open(edf_path, "rb") as f:
        data = bytearray(f.read())

    try:
        n_signals = int(bytes(data[252:256]).decode("ascii").strip())
    except (ValueError, UnicodeDecodeError):
        return None

    if n_signals <= 0:
        return None

    phys_dim_offset = 256 + (16 + 80) * n_signals
    field_len = 8
    needs_repair = False

    for i in range(n_signals):
        start = phys_dim_offset + i * field_len
        end = start + field_len
        field = bytes(data[start:end]).decode("ascii", errors="replace").strip()
        if not field:
            data[start:end] = b"uV" + b" " * 6
            needs_repair = True

    if not needs_repair:
        return None

    fd, tmp_path = tempfile.mkstemp(suffix=".edf")
    try:
        os.write(fd, bytes(data))
    finally:
        os.close(fd)
    return tmp_path


def _read_edf_header_with_repair(
    edf_path: str,
) -> tuple[int, list[str], list[float], list[int], bool]:
    """Read EDF header metadata, repairing invalid physical-dimension fields if needed."""
    import pyedflib

    repaired_path: str | None = None
    repaired = False

    try:
        try:
            reader = pyedflib.EdfReader(edf_path)
        except Exception as open_err:
            if "Physical Dimension" not in str(open_err):
                raise
            repaired_path = _repair_edf_physical_dimensions(edf_path)
            if repaired_path is None:
                raise
            repaired = True
            reader = pyedflib.EdfReader(repaired_path)

        with reader as f:
            n_signals = f.signals_in_file
            labels = list(f.getSignalLabels())
            sample_rates = [float(f.getSampleFrequency(i)) for i in range(n_signals)]
            n_samples = list(f.getNSamples())
        return n_signals, labels, sample_rates, n_samples, repaired
    finally:
        if repaired_path is not None:
            try:
                Path(repaired_path).unlink(missing_ok=True)
            except Exception:
                pass


# Factors converting a channel's declared EDF physical dimension to microvolts.
# Keys are lower-cased dimension strings; anything unlisted is assumed to already
# be in µV (matching the header-repair default that stamps blank dimensions "uV").
_UV_DIMENSION_SCALES: dict[str, float] = {
    "uv": 1.0,
    "µv": 1.0,  # micro sign U+00B5
    "μv": 1.0,  # greek small mu U+03BC
    "microvolt": 1.0,
    "microvolts": 1.0,
    "mv": 1000.0,
    "millivolt": 1000.0,
    "millivolts": 1000.0,
    "v": 1_000_000.0,
    "volt": 1_000_000.0,
    "volts": 1_000_000.0,
}


def _edf_uv_scale_by_label(edf_path: str) -> dict[str, float]:
    """Map each EDF channel label to a factor converting its samples to µV.

    ``pyedflib`` returns signals in their declared physical dimension — µV for most
    EEG/EOG/EMG channels, but frequently mV for ECG. The signal-review display
    assumes µV, so channels declared in mV/V must be rescaled or they render far too
    small (and any derived/bipolar montage mixing units would be wrong). Unknown or
    blank dimensions are assumed to already be µV.

    Args:
        edf_path: Path to the EDF recording.

    Returns:
        Mapping of channel label to a multiplicative µV scale factor.
    """
    import pyedflib

    repaired_path: str | None = None
    scales: dict[str, float] = {}
    try:
        try:
            reader = pyedflib.EdfReader(edf_path)
        except Exception as open_err:
            if "Physical Dimension" not in str(open_err):
                raise
            repaired_path = _repair_edf_physical_dimensions(edf_path)
            if repaired_path is None:
                raise
            reader = pyedflib.EdfReader(repaired_path)
        with reader as f:
            labels = list(f.getSignalLabels())
            for i, label in enumerate(labels):
                dim = str(f.getPhysicalDimension(i) or "").strip().lower()
                scales[label] = _UV_DIMENSION_SCALES.get(dim, 1.0)
    except Exception:
        # Physical-dimension probing is a display nicety; if the header can't be
        # read we assume the signals are already in µV rather than failing the load.
        return {}
    finally:
        if repaired_path is not None:
            try:
                Path(repaired_path).unlink(missing_ok=True)
            except Exception:
                pass
    return scales


def _load_canonical_channels_for_gui(canon_json_path: str) -> list[str]:
    """Load canonical channels using the same rules as inference runtime."""
    with open(canon_json_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(ch) for ch in data]
    if (
        isinstance(data, dict)
        and "channels" in data
        and isinstance(data["channels"], list)
    ):
        return [str(ch) for ch in data["channels"]]
    raise ValueError(
        "Canonical JSON must be a list of channels or a dict with a 'channels' list"
    )


def _default_channel_slot_overrides() -> dict[str, str | None]:
    """Return the default fixed-slot channel override state."""
    return {slot: None for slot in CHANNEL_LAYOUT_SLOT_NAMES}


def _normalize_channel_slot_overrides(
    overrides: dict[str, str | None] | None,
    *,
    available_labels: list[str] | None = None,
) -> tuple[dict[str, str | None], list[str]]:
    """Normalize persisted/manual slot overrides and drop invalid selections."""
    from spectra.data.channel.normalization import infer_channel_type

    normalized = _default_channel_slot_overrides()
    warnings: list[str] = []
    available_set = set(available_labels) if available_labels is not None else None

    if not overrides:
        return normalized, warnings

    for slot, modality in CHANNEL_LAYOUT_SLOTS:
        raw_value = overrides.get(slot)
        if raw_value is None:
            continue

        value = str(raw_value).strip()
        if not value or value.lower() == "auto":
            continue

        if available_set is not None and value not in available_set:
            warnings.append(f"{slot}: '{value}' is not present in the EDF. Using Auto.")
            continue

        inferred_type = infer_channel_type(value)
        if inferred_type != modality:
            warnings.append(
                f"{slot}: '{value}' is {inferred_type!r}, expected {modality!r}. Using Auto."
            )
            continue

        normalized[slot] = value

    return normalized, warnings


def _build_effective_channel_layout(
    overrides: dict[str, str | None] | None,
) -> list[str]:
    """Build the ordered 5-channel runtime layout from slot overrides."""
    overrides = overrides or {}
    return [overrides.get(slot) or slot for slot in CHANNEL_LAYOUT_SLOT_NAMES]


def _find_duplicate_channel_slot_overrides(
    overrides: dict[str, str | None] | None,
) -> dict[str, list[str]]:
    """Return duplicate explicit EDF labels and the slots that use them."""
    duplicates: dict[str, list[str]] = {}
    if not overrides:
        return duplicates

    for slot in CHANNEL_LAYOUT_SLOT_NAMES:
        value = overrides.get(slot)
        if not value:
            continue
        duplicates.setdefault(value, []).append(slot)

    return {label: slots for label, slots in duplicates.items() if len(slots) > 1}


def _legacy_canonical_channels_to_slot_overrides(
    canonical_channels: list[str],
) -> dict[str, str | None] | None:
    """Convert a legacy 5-channel canonical list into slot overrides."""
    from spectra.data.channel.normalization import infer_channel_type

    if len(canonical_channels) != len(CHANNEL_LAYOUT_SLOT_NAMES):
        return None

    migrated = _default_channel_slot_overrides()
    for (slot, modality), channel_name in zip(
        CHANNEL_LAYOUT_SLOTS, canonical_channels, strict=True
    ):
        value = str(channel_name).strip()
        if not value or infer_channel_type(value) != modality:
            return None
        if value != slot:
            migrated[slot] = value

    return migrated


def _resolve_gui_device(preference: str = "auto") -> Any:
    from spectra.utils.device_utils import resolve_device

    return resolve_device(preference)


def _clear_gui_device_cache(
    device: Any,
    *,
    synchronize: bool = True,
    reset_peak_memory_stats: bool = False,
) -> bool:
    from spectra.utils.device_utils import clear_device_cache

    return clear_device_cache(
        device,
        synchronize=synchronize,
        reset_peak_memory_stats=reset_peak_memory_stats,
    )


def _describe_gui_device(device: Any) -> str:
    from spectra.utils.device_utils import describe_device

    return describe_device(device)


def _get_gui_device_memory_stats(device: Any = "auto"):
    from spectra.utils.device_utils import get_device_memory_stats

    return get_device_memory_stats(device)


def _format_transform_sources(labels: list[str], transform_row) -> str:
    """Format rereferencing/source coefficients for channel preview."""
    sources: list[str] = []
    for coeff, label in zip(transform_row, labels, strict=True):
        coeff_value = float(coeff)
        if abs(coeff_value) < 1e-9:
            continue
        if coeff_value == 1.0:
            sources.append(label)
        elif coeff_value == -1.0:
            sources.append(f"-{label}")
        else:
            sources.append(f"{coeff_value:g}*{label}")
    return " + ".join(sources) if sources else "(derived)"


def _compact_transform_label(detail: str) -> str:
    """Compact a derived-source expression for narrow trace labels."""
    compact = detail.replace(" ", "").replace("1*", "")
    compact = compact.replace("+-", "-")
    return compact


def _build_review_channel_display_label(
    *,
    slot: str,
    status: str,
    detail: str,
    source_label: str | None = None,
) -> str:
    """Build the short trace label shown on the waveform plot."""
    if status == "missing":
        return str(slot)
    if source_label:
        return str(source_label)

    detail_text = str(detail).strip()
    if not detail_text:
        return str(slot)
    compact = _compact_transform_label(detail_text)
    return compact or str(slot)


def _downsample_trace_for_display(
    time_axis: np.ndarray,
    y_values: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce plotted points while preserving local extrema within each bucket."""
    x = np.asarray(time_axis, dtype=np.float32)
    y = np.asarray(y_values, dtype=np.float32)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size <= 2:
        return x, y

    max_points = max(2, int(max_points))
    if x.size <= max_points:
        return x, y

    target_buckets = max(1, max_points // 2)
    bucket_size = max(1, int(np.ceil(x.size / target_buckets)))
    sampled_x: list[float] = []
    sampled_y: list[float] = []

    for start in range(0, x.size, bucket_size):
        end = min(x.size, start + bucket_size)
        segment_y = y[start:end]
        if segment_y.size == 0:
            continue
        min_idx = start + int(np.argmin(segment_y))
        max_idx = start + int(np.argmax(segment_y))
        if min_idx == max_idx:
            sampled_x.append(float(x[min_idx]))
            sampled_y.append(float(y[min_idx]))
            continue
        first_idx, second_idx = sorted((min_idx, max_idx))
        sampled_x.extend((float(x[first_idx]), float(x[second_idx])))
        sampled_y.extend((float(y[first_idx]), float(y[second_idx])))

    return (
        np.asarray(sampled_x, dtype=np.float32),
        np.asarray(sampled_y, dtype=np.float32),
    )


_DISPLAY_SOS_CACHE: dict[tuple[Any, ...], np.ndarray | None] = {}


def _format_display_filter_label(spec: DisplayFilterSpec) -> str:
    """Return a compact human-readable summary of a display-filter spec."""
    if spec.low_cut is not None and spec.high_cut is not None:
        band = f"{spec.low_cut:g}-{spec.high_cut:g} Hz"
    elif spec.low_cut is not None:
        band = f"HP {spec.low_cut:g} Hz"
    elif spec.high_cut is not None:
        band = f"LP {spec.high_cut:g} Hz"
    else:
        band = "raw"
    if spec.notch_hz is not None:
        band += f", notch {spec.notch_hz:g}"
    return band


def _design_display_sos(spec: DisplayFilterSpec, fs: float) -> np.ndarray | None:
    """Build cached second-order sections for a display-filter spec at ``fs``.

    Mirrors the Butterworth/notch design used in the offline preprocessing path
    (``src/spectra/preprocessing/raw_conversion.py``) but for live display.

    Args:
        spec: Per-modality filter spec (low/high cut plus optional mains notch).
        fs: Sampling rate in Hz.

    Returns:
        Stacked SOS array, or ``None`` when the spec implies no filtering.
    """
    if fs <= 0:
        return None
    cache_key = (spec.low_cut, spec.high_cut, spec.notch_hz, round(float(fs), 6))
    if cache_key in _DISPLAY_SOS_CACHE:
        return _DISPLAY_SOS_CACHE[cache_key]

    nyq = 0.5 * fs
    sections: list[np.ndarray] = []
    low = spec.low_cut
    high = spec.high_cut
    if high is not None and high >= nyq:
        high = None  # Requested low-pass is above Nyquist; skip it.
    order = SIGNAL_REVIEW_FILTER_ORDER
    if low is not None and low > 0 and high is not None:
        sections.append(
            butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        )
    elif low is not None and low > 0:
        sections.append(butter(order, low / nyq, btype="high", output="sos"))
    elif high is not None:
        sections.append(butter(order, high / nyq, btype="low", output="sos"))

    if spec.notch_hz is not None and 0.0 < spec.notch_hz < nyq:
        notch_b, notch_a = iirnotch(spec.notch_hz, SIGNAL_REVIEW_NOTCH_Q, fs)
        sections.append(tf2sos(notch_b, notch_a))

    sos = np.concatenate(sections, axis=0) if sections else None
    _DISPLAY_SOS_CACHE[cache_key] = sos
    return sos


def _apply_display_filter(signal: np.ndarray, sos: np.ndarray | None) -> np.ndarray:
    """Apply a zero-phase display filter, falling back to median removal.

    Args:
        signal: One-dimensional signal array.
        sos: Second-order sections from :func:`_design_display_sos`, or ``None``.

    Returns:
        Filtered signal as ``float32``. If ``sos`` is ``None`` or the segment is too
        short for the filter, the per-channel median is removed instead so the trace
        is at least baseline-centered.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        return np.asarray(signal, dtype=np.float32)
    if sos is None:
        return (x - float(np.median(x))).astype(np.float32)
    padlen = 6 * int(sos.shape[0]) + 12
    if x.size <= padlen:
        return (x - float(np.median(x))).astype(np.float32)
    try:
        return sosfiltfilt(sos, x).astype(np.float32)
    except ValueError:
        return (x - float(np.median(x))).astype(np.float32)


def _filter_display_signals(
    signals: np.ndarray,
    sample_rate: float,
    channel_plan: list[dict[str, Any]],
    filter_specs: dict[str, DisplayFilterSpec],
) -> np.ndarray:
    """Filter every channel of a display payload using its per-modality spec.

    Args:
        signals: Array of shape ``[n_channels, n_samples]`` in microvolts.
        sample_rate: Sampling rate in Hz.
        channel_plan: Per-channel metadata; ``label`` selects the modality.
        filter_specs: Mapping of modality bucket to :class:`DisplayFilterSpec`.

    Returns:
        Filtered copy with the same shape as ``signals``.
    """
    out = np.empty_like(signals, dtype=np.float32)
    for idx in range(signals.shape[0]):
        row = channel_plan[idx] if idx < len(channel_plan) else {}
        label = str(row.get("label", f"Ch {idx + 1}"))
        modality = _slot_modality(label)
        spec = filter_specs.get(modality, DisplayFilterSpec())
        sos = _design_display_sos(spec, sample_rate)
        out[idx] = _apply_display_filter(signals[idx], sos)
    return out


def _build_channel_preview_plan(
    signal_labels: list[str],
    canonical_channels: list[str] | None,
) -> list[tuple[str, str, str]]:
    """Preview the exact converter-compatible, direct recorded channel plan."""
    from spectra.preprocessing.edf import select_channel_plan

    return [
        (slot, "matched" if source else "missing", source or "zero-filled")
        for slot, source in select_channel_plan(signal_labels, canonical_channels)
    ]


def _load_review_signal_payload(
    edf_path: str,
    channel_layout: list[str],
) -> dict[str, Any]:
    """Load EDF review signals and precompute display presets."""
    from spectra.data.channel.normalization import infer_channel_type
    from spectra.inference.runtime import (
        harmonize_channel_sample_rates,
        load_edf,
    )
    from spectra.preprocessing.edf import select_channel_plan

    signal_labels, channel_data, sample_rates = load_edf(edf_path)
    harmonized_data, harmonized_fs = harmonize_channel_sample_rates(
        channel_data,
        sample_rates,
    )
    # Convert every channel to µV before deriving montages so the display's µV/div
    # sensitivities are accurate and bipolar derivations never mix units (ECG is
    # commonly stored in mV). Inference is unaffected — it IQR-normalizes per
    # channel, which is scale-invariant.
    uv_scales = _edf_uv_scale_by_label(edf_path)
    if any(abs(scale - 1.0) > 1e-9 for scale in uv_scales.values()):
        harmonized_data = {
            name: np.asarray(sig, dtype=np.float32) * float(uv_scales.get(name, 1.0))
            for name, sig in harmonized_data.items()
        }
    raw_channel_labels = [label for label in signal_labels if label in harmonized_data]
    if len(raw_channel_labels) != len(harmonized_data):
        raw_channel_labels = list(harmonized_data.keys())
    raw_signals = np.stack(
        [
            np.asarray(harmonized_data[label], dtype=np.float32)
            for label in raw_channel_labels
        ],
        axis=0,
    )
    raw_label_to_index = {label: idx for idx, label in enumerate(raw_channel_labels)}

    selected = select_channel_plan(signal_labels, channel_layout)
    aligned_signals = np.stack(
        [
            (
                harmonized_data[source]
                if source is not None
                else np.zeros(raw_signals.shape[1], dtype=np.float32)
            )
            for _, source in selected
        ]
    )
    presence_mask = np.asarray([source is not None for _, source in selected])
    preview_rows = _build_channel_preview_plan(signal_labels, channel_layout)
    channel_plan = [
        {
            "slot": slot,
            "status": status,
            "detail": detail,
            "present": bool(index < len(presence_mask) and presence_mask[index] > 0),
        }
        for index, (slot, status, detail) in enumerate(preview_rows)
    ]

    def _empty_signal() -> np.ndarray:
        return np.zeros(raw_signals.shape[1], dtype=np.float32)

    scoring_plan = [
        {
            "label": str(row["slot"]),
            "status": str(row["status"]),
            "detail": str(row["detail"]),
            "present": bool(row["present"]),
            "source_label": (
                str(row["detail"]).split(" (", 1)[0].strip()
                if row["status"] == "matched"
                else None
            ),
            "display_label": _build_review_channel_display_label(
                slot=str(row["slot"]),
                status=str(row["status"]),
                detail=str(row["detail"]),
                source_label=(
                    str(row["detail"]).split(" (", 1)[0].strip()
                    if row["status"] == "matched"
                    else None
                ),
            ),
        }
        for row in channel_plan
    ]

    raw_matched_signals: list[np.ndarray] = []
    raw_matched_plan: list[dict[str, Any]] = []
    used_raw_labels: set[str] = set()
    for scoring_row in scoring_plan:
        source_label = scoring_row.get("source_label")
        if source_label and source_label in raw_label_to_index:
            raw_matched_signals.append(raw_signals[raw_label_to_index[source_label]])
            raw_matched_plan.append(
                {
                    "label": str(scoring_row["label"]),
                    "status": "matched",
                    "detail": str(source_label),
                    "present": True,
                    "source_label": str(source_label),
                    "display_label": _build_review_channel_display_label(
                        slot=str(scoring_row["label"]),
                        status="matched",
                        detail=str(source_label),
                        source_label=str(source_label),
                    ),
                }
            )
            used_raw_labels.add(str(source_label))
        else:
            raw_matched_signals.append(_empty_signal())
            raw_matched_plan.append(
                {
                    "label": str(scoring_row["label"]),
                    "status": "missing",
                    "detail": "Unavailable without derivation",
                    "present": False,
                    "source_label": None,
                    "display_label": _build_review_channel_display_label(
                        slot=str(scoring_row["label"]),
                        status="missing",
                        detail="Unavailable without derivation",
                        source_label=None,
                    ),
                }
            )

    def _pick_first_available(
        allowed_types: set[str],
        *,
        exclude: set[str],
    ) -> str | None:
        for label in raw_channel_labels:
            if label in exclude:
                continue
            if infer_channel_type(label) in allowed_types:
                return label
        return None

    expanded_signals: list[np.ndarray] = [
        np.asarray(sig, dtype=np.float32) for sig in aligned_signals
    ]
    expanded_plan: list[dict[str, Any]] = [dict(row) for row in scoring_plan]
    extra_specs = [
        ("ECG", {"ecg"}),
        ("EEG Aux", {"eeg"}),
        ("EOG Aux", {"eog"}),
    ]
    expanded_excludes = set(used_raw_labels)
    for label_name, allowed_types in extra_specs:
        extra_label = _pick_first_available(allowed_types, exclude=expanded_excludes)
        if extra_label is not None and extra_label in raw_label_to_index:
            expanded_signals.append(raw_signals[raw_label_to_index[extra_label]])
            expanded_plan.append(
                {
                    "label": label_name,
                    "status": "matched",
                    "detail": extra_label,
                    "present": True,
                    "source_label": extra_label,
                    "display_label": _build_review_channel_display_label(
                        slot=label_name,
                        status="matched",
                        detail=extra_label,
                        source_label=extra_label,
                    ),
                }
            )
            expanded_excludes.add(extra_label)
        else:
            expanded_signals.append(_empty_signal())
            expanded_plan.append(
                {
                    "label": label_name,
                    "status": "missing",
                    "detail": "Unavailable",
                    "present": False,
                    "source_label": None,
                    "display_label": _build_review_channel_display_label(
                        slot=label_name,
                        status="missing",
                        detail="Unavailable",
                        source_label=None,
                    ),
                }
            )

    display_presets = {
        "Scoring 5ch": {
            "signals": np.asarray(aligned_signals, dtype=np.float32),
            "channel_plan": scoring_plan,
        },
        "Expanded Sleep Review": {
            "signals": np.asarray(expanded_signals, dtype=np.float32),
            "channel_plan": expanded_plan,
        },
        "Raw Matched": {
            "signals": np.asarray(raw_matched_signals, dtype=np.float32),
            "channel_plan": raw_matched_plan,
        },
    }
    return {
        "edf_path": edf_path,
        "sample_rate": float(harmonized_fs),
        "signals": np.asarray(aligned_signals, dtype=np.float32),
        "channel_layout": list(channel_layout),
        "channel_plan": channel_plan,
        "raw_channel_labels": list(raw_channel_labels),
        "raw_signals": np.asarray(raw_signals, dtype=np.float32),
        "display_presets": display_presets,
        "n_samples": int(raw_signals.shape[1]) if raw_signals.ndim == 2 else 0,
    }


def create_hypnogram_canvas_class():
    """Factory function to create HypnogramCanvas class after FigureCanvas is available."""
    global FigureCanvas, NavigationToolbar
    if FigureCanvas is None:
        raise RuntimeError("FigureCanvas is not initialized")
    figure_canvas_base = cast(type, FigureCanvas)

    class _HypnogramCanvas(figure_canvas_base):
        """Canvas for displaying hypnogram visualization with confidence overlay.

        Features:
        - Zoom/pan with navigation toolbar
        - Click-to-inspect epochs
        - Sleep cycle markers
        - Confidence heatmap
        - Low-confidence highlighting
        - Reference hypnogram comparison
        """

        # Signal for epoch click (emitted when user clicks on hypnogram)
        # We'll handle this via callback since Qt signals need special setup

        def __init__(self, parent=None):
            # Create figure with two subplots - hypnogram and confidence
            self.fig = matplotlib.figure.Figure(
                figsize=(12, 5), dpi=100, layout="constrained"
            )
            self.gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)
            self.ax = self.fig.add_subplot(self.gs[0])
            self.ax_conf = self.fig.add_subplot(self.gs[1], sharex=self.ax)
            super().__init__(self.fig)
            self.setParent(parent)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self.setMinimumSize(0, 0)

            # Data storage
            self.predictions = None
            self.probabilities = None
            self.epoch_sec = 30
            self.reference_predictions = None  # For comparison mode
            self.time_hours = None
            self.confidence = None
            self.score_window = None  # (start, end) analysis window, or None for all

            # Visualization options
            self.show_sleep_cycles = False
            self.show_confidence_heatmap = False
            self.confidence_threshold = DEFAULT_LOW_CONFIDENCE_THRESHOLD
            self.highlight_low_confidence = False

            # Epoch inspection callback
            self.epoch_click_callback = None

            # Connect mouse events for click-to-inspect
            self.mpl_connect("button_press_event", self._on_click)
            self.mpl_connect("motion_notify_event", self._on_hover)

            # Annotation for hover tooltip
            self._hover_annotation = None

            # Configure initial plots
            for ax in [self.ax, self.ax_conf]:
                ax.set_facecolor("#2b2b2b")
                ax.tick_params(axis="both", colors="#dddddd")
                for spine in ax.spines.values():
                    spine.set_color("#555555")

            self.fig.set_facecolor("#2b2b2b")
            self.ax.grid(True, alpha=0.2, linestyle="--", color="#555555")
            self.ax.set_ylabel(
                "Sleep Stage", fontsize=10, fontweight="bold", color="#dddddd"
            )
            self.ax.set_title(
                "Hypnogram", fontsize=12, fontweight="bold", color="#ffffff"
            )
            self.ax_conf.set_xlabel(
                "Time (hours)", fontsize=10, fontweight="bold", color="#dddddd"
            )
            self.ax_conf.set_ylabel(
                "Confidence", fontsize=9, fontweight="bold", color="#dddddd"
            )
            # Hide x-axis labels on hypnogram (shared with confidence plot)
            self.ax.tick_params(labelbottom=False)

        def set_epoch_click_callback(self, callback):
            """Set callback for when user clicks on an epoch.

            Args:
                callback: Function that takes (epoch_index, stage, confidence, probabilities)
            """
            self.epoch_click_callback = callback

        def _get_epoch_at_time(self, time_hours):
            """Get epoch index at given time in hours."""
            if self.predictions is None or time_hours is None:
                return None
            epoch_index = int(time_hours * 3600 / self.epoch_sec)
            if 0 <= epoch_index < len(self.predictions):
                return epoch_index
            return None

        def _on_click(self, event):
            """Handle mouse click on hypnogram."""
            if event.inaxes not in [self.ax, self.ax_conf]:
                return
            if self.predictions is None:
                return

            epoch_idx = self._get_epoch_at_time(event.xdata)
            if epoch_idx is not None and self.epoch_click_callback:
                stage = self.predictions[epoch_idx]
                conf = (
                    self.confidence[epoch_idx] if self.confidence is not None else None
                )
                probs = (
                    self.probabilities[epoch_idx]
                    if self.probabilities is not None
                    else None
                )
                self.epoch_click_callback(epoch_idx, stage, conf, probs)

        def _on_hover(self, event):
            """Handle mouse hover for tooltip."""
            if event.inaxes not in [self.ax, self.ax_conf]:
                if self._hover_annotation:
                    self._hover_annotation.set_visible(False)
                    self.draw_idle()
                return

            if self.predictions is None:
                return

            epoch_idx = self._get_epoch_at_time(event.xdata)
            if epoch_idx is None:
                if self._hover_annotation:
                    self._hover_annotation.set_visible(False)
                    self.draw_idle()
                return

            # Build tooltip text
            stage = self.predictions[epoch_idx]
            stage_name = STAGE_DISPLAY_NAMES.get(stage, "Unknown")

            time_sec = epoch_idx * self.epoch_sec
            time_str = _format_elapsed_time_hms(time_sec)

            tooltip = f"Epoch {epoch_idx + 1}\nTime: {time_str}\nStage: {stage_name}"

            if self.confidence is not None:
                conf = self.confidence[epoch_idx]
                tooltip += f"\nConf: {conf:.1%}"

            # Create or update annotation
            if self._hover_annotation is None:
                self._hover_annotation = self.ax.annotate(
                    tooltip,
                    xy=(event.xdata, event.ydata),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="#3b3b3b",
                        edgecolor="#4CAF50",
                        alpha=0.9,
                    ),
                    fontsize=9,
                    color="#dddddd",
                    zorder=100,
                )
            else:
                self._hover_annotation.remove()
                self._hover_annotation = self.ax.annotate(
                    tooltip,
                    xy=(event.xdata, event.ydata),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="#3b3b3b",
                        edgecolor="#4CAF50",
                        alpha=0.9,
                    ),
                    fontsize=9,
                    color="#dddddd",
                    zorder=100,
                )

            self.draw_idle()

        def set_reference_hypnogram(self, reference_predictions):
            """Set a reference hypnogram for comparison mode.

            Args:
                reference_predictions: Array of stage predictions to compare against
            """
            self.reference_predictions = reference_predictions

        def clear_reference_hypnogram(self):
            """Clear the reference hypnogram."""
            self.reference_predictions = None

        def get_reference_agreement(self):
            """Calculate agreement percentage between predictions and reference.

            Returns:
                float: Agreement percentage (0.0 to 1.0), or None if no reference is set
            """
            if self.reference_predictions is None or self.predictions is None:
                return None

            # Handle length mismatch
            min_len = min(len(self.predictions), len(self.reference_predictions))
            if min_len == 0:
                return None

            preds = np.asarray(self.predictions[:min_len])
            ref = np.asarray(self.reference_predictions[:min_len])

            # Only score epochs that are labeled in the reference (-1 = unscored).
            scored = ref >= 0
            if not np.any(scored):
                return None

            agreements = preds[scored] == ref[scored]
            return np.mean(agreements)

        def set_confidence_threshold(self, threshold):
            """Set the confidence threshold for low-confidence highlighting."""
            self.confidence_threshold = threshold

        def toggle_sleep_cycles(self, show: bool):
            """Toggle sleep cycle markers display."""
            self.show_sleep_cycles = show
            if self.predictions is not None:
                self.plot_hypnogram(
                    self.predictions,
                    self.epoch_sec,
                    self.probabilities,
                    self.confidence,
                )

        def toggle_low_confidence_highlight(self, show: bool):
            """Toggle low-confidence epoch highlighting."""
            self.highlight_low_confidence = show
            if self.predictions is not None:
                self.plot_hypnogram(
                    self.predictions,
                    self.epoch_sec,
                    self.probabilities,
                    self.confidence,
                )

        def _detect_sleep_cycles(self, predictions):
            """Detect NREM-REM sleep cycles.

            Returns list of (cycle_start_epoch, rem_start_epoch, cycle_end_epoch) tuples.
            """
            cycles = []
            n_epochs = len(predictions)

            # Find sleep onset
            sleep_onset = None
            for i, p in enumerate(predictions):
                if p in [1, 2, 3, 4]:  # Any sleep stage
                    sleep_onset = i
                    break

            if sleep_onset is None:
                return cycles

            # Track cycles: NREM -> REM -> (NREM or Wake)
            in_nrem = False
            in_rem = False
            cycle_start = None
            rem_start = None

            for i in range(sleep_onset, n_epochs):
                stage = predictions[i]

                if not in_nrem and not in_rem:
                    # Looking for NREM to start a cycle
                    if stage in [1, 2, 3]:  # N1, N2, N3
                        in_nrem = True
                        cycle_start = i
                elif in_nrem and not in_rem:
                    # In NREM, looking for REM
                    if stage == 4:  # REM
                        in_rem = True
                        rem_start = i
                    elif stage == 0:  # Wake - reset if prolonged
                        # Allow brief wake, but reset if > 5 min
                        wake_count = 0
                        for j in range(i, min(i + 10, n_epochs)):  # 5 min = 10 epochs
                            if predictions[j] == 0:
                                wake_count += 1
                        if wake_count >= 10:
                            in_nrem = False
                            cycle_start = None
                elif in_rem:
                    # In REM, looking for NREM/Wake to end cycle
                    if stage in [0, 1, 2, 3]:  # Not REM
                        cycles.append((cycle_start, rem_start, i))
                        in_nrem = False
                        in_rem = False
                        cycle_start = None
                        rem_start = None

                        # Check if this starts a new cycle
                        if stage in [1, 2, 3]:
                            in_nrem = True
                            cycle_start = i

            # Handle incomplete final cycle
            if in_rem and cycle_start is not None and rem_start is not None:
                cycles.append((cycle_start, rem_start, n_epochs - 1))

            return cycles

        def plot_hypnogram(
            self,
            predictions: np.ndarray,
            epoch_sec: int = 30,
            probabilities: np.ndarray | None = None,
            confidences: np.ndarray | None = None,
            score_window: tuple[int, int] | None = None,
        ):
            """Plot hypnogram from predictions array with optional confidence overlay.

            Args:
                predictions: Array of stage predictions (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
                epoch_sec: Duration of each epoch in seconds
                probabilities: Optional (n_epochs, 5) array of class probabilities
                confidences: Optional confidence values aligned to epochs
                score_window: Optional ``(start, end)`` epoch range to treat as scored.
                    Epochs outside it are rendered blank (greyed) instead of as a
                    colored stage line. ``None`` shows the whole recording.
            """
            self.predictions = predictions
            self.probabilities = probabilities
            self.epoch_sec = epoch_sec

            # Clear previous plots and reset hover annotation
            self.ax.clear()
            self.ax_conf.clear()
            self._hover_annotation = None

            # Define stage mapping and colors
            stage_names = ["W", "REM", "N1", "N2", "N3"]
            stage_values = [
                4,
                3,
                2,
                1,
                0,
            ]  # Y-axis positions (inverted for traditional hypnogram)
            stage_colors = {
                0: "#8B4513",  # W (Wake) - brown
                1: "#FFD700",  # N1 - gold
                2: "#4CAF50",  # N2 - green
                3: "#2196F3",  # N3 - blue
                4: "#FF4500",  # REM - orange-red
            }

            # Create mapping from prediction values to y-axis positions
            pred_to_y = {0: 4, 1: 2, 2: 1, 3: 0, 4: 3}

            # Convert predictions to y-axis positions
            y_values = np.array([pred_to_y[p] for p in predictions])

            # Time axis in hours
            n_epochs = len(predictions)
            time_hours = np.arange(n_epochs) * epoch_sec / 3600
            self.time_hours = time_hours

            # Resolve the analysis window. Epochs outside it are rendered blank.
            if score_window is None:
                w_start, w_end = 0, n_epochs
            else:
                w_start = max(0, min(int(score_window[0]), n_epochs))
                w_end = max(w_start, min(int(score_window[1]), n_epochs))
            # Persist the resolved window so exports can mark unscored epochs.
            self.score_window = (
                None
                if (w_start, w_end) == (0, n_epochs)
                else (
                    w_start,
                    w_end,
                )
            )

            def _in_window(i: int) -> bool:
                return w_start <= i < w_end

            # Grey-shade the unscored spans before / after the window.
            if (w_start, w_end) != (0, n_epochs):
                if w_start > 0:
                    self.ax.axvspan(
                        0.0,
                        w_start * epoch_sec / 3600,
                        alpha=0.12,
                        color="#9E9E9E",
                        zorder=0,
                    )
                if w_end < n_epochs:
                    self.ax.axvspan(
                        w_end * epoch_sec / 3600,
                        n_epochs * epoch_sec / 3600,
                        alpha=0.12,
                        color="#9E9E9E",
                        zorder=0,
                    )

            # Calculate confidence for later use
            self.confidence = _resolve_confidences(probabilities, confidences)
            if self.confidence is not None and len(self.confidence) != n_epochs:
                self.confidence = None

            # Plot sleep cycle markers first (behind hypnogram)
            if self.show_sleep_cycles:
                cycles = self._detect_sleep_cycles(predictions)
                cycle_colors = [
                    "#3F51B5",
                    "#9C27B0",
                    "#00BCD4",
                    "#FF5722",
                    "#607D8B",
                    "#795548",
                ]
                for idx, (start, _rem_start, end) in enumerate(cycles):
                    cycle_color = cycle_colors[idx % len(cycle_colors)]
                    t_start = start * epoch_sec / 3600
                    t_end = end * epoch_sec / 3600

                    # Shade the cycle region
                    self.ax.axvspan(
                        t_start, t_end, alpha=0.15, color=cycle_color, zorder=0
                    )

                    # Add cycle label
                    self.ax.text(
                        (t_start + t_end) / 2,
                        4.3,
                        f"Cycle {idx + 1}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=cycle_color,
                        fontweight="bold",
                        alpha=0.8,
                    )

            # Plot reference hypnogram if available (comparison mode)
            if (
                self.reference_predictions is not None
                and len(self.reference_predictions) == n_epochs
            ):
                # Map to y positions; unscored epochs (-1) become NaN and are
                # rendered as gaps rather than crashing on a missing key.
                ref_y = np.array(
                    [pred_to_y.get(int(p), np.nan) for p in self.reference_predictions],
                    dtype=float,
                )
                for i in range(n_epochs - 1):
                    if np.isnan(ref_y[i]) or np.isnan(ref_y[i + 1]):
                        continue
                    self.ax.plot(
                        [time_hours[i], time_hours[i + 1]],
                        [ref_y[i] + 0.15, ref_y[i] + 0.15],  # Offset slightly
                        color="#888888",
                        linewidth=1.5,
                        linestyle="--",
                        alpha=0.7,
                        solid_capstyle="butt",
                    )

            # Plot main hypnogram as step function with colors
            for i in range(n_epochs - 1):
                # Leave epochs outside the analysis window blank.
                if not _in_window(i):
                    continue
                stage = predictions[i]
                y_start = y_values[i]
                y_end = y_values[i + 1]

                # Check if this is a low-confidence epoch
                line_alpha = 1.0
                line_width = 2
                if self.highlight_low_confidence and self.confidence is not None:
                    if self.confidence[i] < self.confidence_threshold:
                        line_alpha = 0.4
                        # Add highlight marker
                        self.ax.axvspan(
                            time_hours[i],
                            time_hours[i + 1],
                            alpha=0.3,
                            color="#F44336",
                            zorder=1,
                        )

                # Horizontal line
                self.ax.plot(
                    [time_hours[i], time_hours[i + 1]],
                    [y_start, y_start],
                    color=stage_colors[stage],
                    linewidth=line_width,
                    alpha=line_alpha,
                    solid_capstyle="butt",
                )

                # Vertical transition line (if stage changes within the window)
                if y_start != y_end and _in_window(i + 1):
                    self.ax.plot(
                        [time_hours[i + 1], time_hours[i + 1]],
                        [y_start, y_end],
                        color="gray",
                        linewidth=1,
                        alpha=0.5,
                    )

            # Plot last epoch
            if n_epochs > 0 and _in_window(n_epochs - 1):
                stage = predictions[-1]
                line_alpha = 1.0
                if self.highlight_low_confidence and self.confidence is not None:
                    if self.confidence[-1] < self.confidence_threshold:
                        line_alpha = 0.4
                        self.ax.axvspan(
                            time_hours[-1],
                            time_hours[-1] + epoch_sec / 3600,
                            alpha=0.3,
                            color="#F44336",
                            zorder=1,
                        )
                self.ax.plot(
                    [time_hours[-1], time_hours[-1] + epoch_sec / 3600],
                    [y_values[-1], y_values[-1]],
                    color=stage_colors[stage],
                    linewidth=2,
                    alpha=line_alpha,
                    solid_capstyle="butt",
                )

            # Configure axes
            self.ax.set_yticks(stage_values)
            self.ax.set_yticklabels(stage_names, fontsize=9)
            self.ax.set_xlabel("Time (hours)", fontsize=10, fontweight="bold")
            self.ax.set_ylabel("Sleep Stage", fontsize=10, fontweight="bold")

            # Title with optional comparison indicator
            title = "Sleep Hypnogram"
            if self.reference_predictions is not None:
                title += " (with Reference)"
            if self.show_sleep_cycles:
                cycles = self._detect_sleep_cycles(predictions)
                title += f" - {len(cycles)} Sleep Cycles"
            self.ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

            # Set x-axis limits and format
            self.ax.set_xlim(0, max(time_hours[-1] + epoch_sec / 3600, 1))
            self.ax.set_ylim(-0.5, 5)  # Extra space for cycle labels
            self.ax.grid(True, alpha=0.3, linestyle="--", which="both")

            # Style
            self.ax.set_facecolor("#2b2b2b")
            self.fig.set_facecolor("#2b2b2b")
            self.ax.tick_params(axis="both", colors="#dddddd")
            for spine in self.ax.spines.values():
                spine.set_color("#555555")

            # Add legend
            from matplotlib.patches import (  # pyright: ignore[reportMissingModuleSource]
                Patch,  # pyright: ignore[reportMissingModuleSource]
            )

            legend_elements = [
                Patch(facecolor=stage_colors[0], label="Wake"),
                Patch(facecolor=stage_colors[4], label="REM"),
                Patch(facecolor=stage_colors[1], label="N1"),
                Patch(facecolor=stage_colors[2], label="N2"),
                Patch(facecolor=stage_colors[3], label="N3"),
            ]
            if self.reference_predictions is not None:
                legend_elements.append(Patch(facecolor="#888888", label="Reference"))
            self.ax.legend(
                handles=legend_elements,
                loc="upper right",
                ncol=min(6, len(legend_elements)),
                framealpha=0.9,
                fontsize=9,
                facecolor="#3b3b3b",
                edgecolor="#555555",
                labelcolor="#dddddd",
            )

            # Plot confidence overlay
            self.ax_conf.set_facecolor("#2b2b2b")
            self.ax_conf.tick_params(axis="both", colors="#dddddd")
            for spine in self.ax_conf.spines.values():
                spine.set_color("#555555")

            if self.confidence is not None:
                # Color-code by confidence level: red (low) -> yellow -> green (high)
                colors = []
                for conf in self.confidence:
                    colors.append(
                        CONFIDENCE_BAND_COLORS[
                            _get_confidence_band(conf, self.confidence_threshold)
                        ]
                    )

                # Plot as bar chart
                self.ax_conf.bar(
                    time_hours,
                    self.confidence,
                    width=epoch_sec / 3600 * 0.9,
                    color=colors,
                    alpha=0.8,
                    edgecolor="none",
                )

                # Add threshold lines
                self.ax_conf.axhline(
                    y=HIGH_CONFIDENCE_THRESHOLD,
                    color=CONFIDENCE_BAND_COLORS["high"],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                )
                self.ax_conf.axhline(
                    y=self.confidence_threshold,
                    color=CONFIDENCE_BAND_COLORS["medium"],
                    linestyle="--",
                    alpha=0.6,
                    linewidth=1,
                )

                # Add user-defined threshold line if highlighting is enabled
                if self.highlight_low_confidence:
                    self.ax_conf.axhline(
                        y=self.confidence_threshold,
                        color=CONFIDENCE_BAND_COLORS["low"],
                        linestyle="-",
                        alpha=0.8,
                        linewidth=2,
                        label=f"Flag threshold ({self.confidence_threshold:.0%})",
                    )

                self.ax_conf.set_ylim(0, 1)
                self.ax_conf.set_ylabel(
                    "Confidence", fontsize=9, fontweight="bold", color="#dddddd"
                )

                # Add confidence legend
                from matplotlib.patches import (  # pyright: ignore[reportMissingModuleSource]
                    Patch,  # pyright: ignore[reportMissingModuleSource]
                )

                conf_legend = [
                    Patch(
                        facecolor=CONFIDENCE_BAND_COLORS["high"],
                        label=f"High (≥{HIGH_CONFIDENCE_THRESHOLD:.0%})",
                    ),
                    Patch(
                        facecolor=CONFIDENCE_BAND_COLORS["medium"],
                        label=(
                            f"Medium ({self.confidence_threshold:.0%}-"
                            f"{HIGH_CONFIDENCE_THRESHOLD:.0%})"
                        ),
                    ),
                    Patch(
                        facecolor=CONFIDENCE_BAND_COLORS["low"],
                        label=f"Low (<{self.confidence_threshold:.0%})",
                    ),
                ]
                self.ax_conf.legend(
                    handles=conf_legend,
                    loc="upper right",
                    ncol=3,
                    framealpha=0.9,
                    fontsize=8,
                    facecolor="#3b3b3b",
                    edgecolor="#555555",
                    labelcolor="#dddddd",
                )
            else:
                # No probabilities available - show placeholder
                self.ax_conf.text(
                    0.5,
                    0.5,
                    "Confidence data not available",
                    transform=self.ax_conf.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#888888",
                    style="italic",
                )
                self.ax_conf.set_ylim(0, 1)

            self.ax_conf.set_xlabel(
                "Time (hours)", fontsize=10, fontweight="bold", color="#dddddd"
            )
            self.ax_conf.grid(True, alpha=0.2, linestyle="--", color="#555555")

            self.draw()

        def get_agreement_with_reference(self):
            """Calculate agreement statistics with reference hypnogram.

            Returns:
                dict with agreement metrics, or None if no reference set
            """
            if self.reference_predictions is None or self.predictions is None:
                return None

            if len(self.reference_predictions) != len(self.predictions):
                return None

            preds = np.asarray(self.predictions)
            ref = np.asarray(self.reference_predictions)
            agreements = preds == ref

            # Only score epochs labeled in the reference (-1 = unscored).
            scored = ref >= 0
            n_epochs = int(np.sum(scored))
            if n_epochs == 0:
                return None
            overall_agreement = np.mean(agreements[scored])

            # Per-stage agreement
            stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
            stage_agreement = {}
            for stage in range(5):
                mask = ref == stage
                if np.sum(mask) > 0:
                    stage_agreement[stage_names[stage]] = np.mean(agreements[mask])
                else:
                    stage_agreement[stage_names[stage]] = None

            return {
                "overall": overall_agreement,
                "by_stage": stage_agreement,
                "n_epochs": n_epochs,
                "n_disagreements": int(n_epochs - np.sum(agreements[scored])),
            }

        def export_hypnogram(self, file_path: str, dpi: int = 300):
            """Export hypnogram to file.

            Args:
                file_path: Output file path (.png, .pdf, .svg)
                dpi: Resolution for raster formats
            """
            if self.predictions is None:
                raise ValueError("No hypnogram to export")

            self.fig.savefig(file_path, dpi=dpi, bbox_inches="tight", facecolor="white")

        def export_edf_annotations(self, file_path: str, recording_start_time=None):
            """Export predictions as EDF+ annotations file.

            Creates an EDF+ compatible annotations file that can be loaded in
            EDFbrowser, Polyman, and other EDF viewers alongside the original recording.

            Args:
                file_path: Output file path (.edf)
                recording_start_time: datetime object for recording start, or None for midnight
            """
            if self.predictions is None:
                raise ValueError("No predictions to export")

            try:
                import pyedflib
            except ImportError as import_err:
                raise ImportError(
                    "pyedflib is required for EDF+ export. Install with: pip install pyedflib"
                ) from import_err

            from datetime import datetime

            if recording_start_time is None:
                recording_start_time = datetime(2000, 1, 1, 0, 0, 0)

            # Stage name mapping for annotations
            stage_names = {
                0: "Sleep stage W",
                1: "Sleep stage N1",
                2: "Sleep stage N2",
                3: "Sleep stage N3",
                4: "Sleep stage R",
            }

            n_epochs = len(self.predictions)

            # Only annotate epochs inside the analysis window; out-of-window epochs
            # are left unannotated (i.e. unscored) so the file matches the report.
            if self.score_window is None:
                w_start, w_end = 0, n_epochs
            else:
                w_start = max(0, min(int(self.score_window[0]), n_epochs))
                w_end = max(w_start, min(int(self.score_window[1]), n_epochs))

            # Build annotation list: (onset_seconds, duration_seconds, annotation_text)
            annotations = []
            if w_end > w_start:
                current_stage = self.predictions[w_start]
                segment_start = w_start

                for i in range(w_start + 1, w_end):
                    if self.predictions[i] != current_stage:
                        # End of segment
                        onset = segment_start * self.epoch_sec
                        duration = (i - segment_start) * self.epoch_sec
                        annotations.append(
                            (onset, duration, stage_names[current_stage])
                        )
                        current_stage = self.predictions[i]
                        segment_start = i

                # Add final segment
                onset = segment_start * self.epoch_sec
                duration = (w_end - segment_start) * self.epoch_sec
                annotations.append((onset, duration, stage_names[current_stage]))

            # Create EDF+ annotations file
            with pyedflib.EdfWriter(
                file_path, 0, file_type=pyedflib.FILETYPE_EDFPLUS
            ) as f:
                f.setStartdatetime(recording_start_time)
                f.setPatientName("Anonymous")
                f.setPatientCode("")
                f.setTechnician("SPECTRA")
                f.setRecordingAdditional("Sleep staging annotations")

                # Write annotations
                for onset, duration, text in annotations:
                    f.writeAnnotation(onset, duration, text)

            return len(annotations)

        def export_epoch_csv(
            self,
            file_path: str,
            threshold: float | None = None,
            model_predictions: np.ndarray | None = None,
            manual_overrides: dict[int, int] | None = None,
        ):
            """Export detailed epoch-by-epoch results to CSV.

            Includes epoch number, time, stage, confidence, low-confidence flag,
            strongest alternate stage for flagged epochs, and per-class probabilities.

            Args:
                file_path: Output file path (.csv)
                threshold: Low-confidence threshold override
            """
            if self.predictions is None:
                raise ValueError("No predictions to export")

            import csv

            active_threshold = (
                self.confidence_threshold if threshold is None else float(threshold)
            )
            base_predictions = (
                np.asarray(model_predictions, dtype=np.int64)
                if model_predictions is not None
                and len(model_predictions) == len(self.predictions)
                else np.asarray(self.predictions, dtype=np.int64)
            )
            records = _build_epoch_confidence_records(
                base_predictions,
                final_predictions=self.predictions,
                manual_overrides=manual_overrides,
                epoch_sec=self.epoch_sec,
                probabilities=self.probabilities,
                confidences=self.confidence,
                threshold=active_threshold,
            )

            # Out-of-window epochs are reported as Unscored (final stage only; the
            # raw model stage is preserved in the ModelStage columns).
            n_epochs = len(self.predictions)
            if self.score_window is None:
                w_start, w_end = 0, n_epochs
            else:
                w_start = max(0, min(int(self.score_window[0]), n_epochs))
                w_end = max(w_start, min(int(self.score_window[1]), n_epochs))

            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                header = [
                    "Epoch",
                    "Time_sec",
                    "Time_hms",
                    "Stage",
                    "Stage_numeric",
                    "ModelStage",
                    "ModelStageNumeric",
                    "FinalStage",
                    "FinalStageNumeric",
                    "ManualOverride",
                ]
                if records and records[0]["confidence"] is not None:
                    header.extend(
                        [
                            "Confidence",
                            "LowConfidenceFlag",
                            "NextMostProbableStage",
                            "NextMostProbableProbability",
                        ]
                    )
                if records and records[0]["probabilities"] is not None:
                    header.extend(["P_Wake", "P_N1", "P_N2", "P_N3", "P_REM"])
                writer.writerow(header)

                # Data
                for record in records:
                    in_window = w_start <= int(record["epoch_index"]) < w_end
                    final_stage = record["final_stage"] if in_window else "Unscored"
                    final_stage_numeric = (
                        record["final_stage_numeric"] if in_window else -1
                    )
                    row = [
                        record["epoch"],
                        record["time_sec"],
                        record["time_hms"],
                        final_stage,
                        final_stage_numeric,
                        record["model_stage"],
                        record["model_stage_numeric"],
                        final_stage,
                        final_stage_numeric,
                        "Yes" if record["manual_override"] else "No",
                    ]

                    confidence = record["confidence"]
                    if confidence is not None:
                        row.extend(
                            [
                                f"{float(confidence):.4f}",
                                "Yes" if record["low_confidence"] else "No",
                                (
                                    str(record["next_stage"])
                                    if record["low_confidence"]
                                    and record["next_stage"] is not None
                                    else ""
                                ),
                                (
                                    f"{float(record['next_stage_probability']):.4f}"
                                    if record["low_confidence"]
                                    and record["next_stage_probability"] is not None
                                    else ""
                                ),
                            ]
                        )

                    probabilities = record["probabilities"]
                    if probabilities is not None:
                        for p in probabilities:
                            row.append(f"{p:.4f}")

                    writer.writerow(row)

            return len(self.predictions)

        def get_low_confidence_epochs(self, threshold=None):
            """Get list of epochs with confidence below threshold.

            Args:
                threshold: Confidence threshold (default: self.confidence_threshold)

            Returns:
                List of (epoch_index, stage, confidence) tuples
            """
            if self.confidence is None:
                return []
            if self.predictions is None:
                return []

            if threshold is None:
                threshold = self.confidence_threshold

            low_conf = []

            for i, conf in enumerate(self.confidence):
                if conf < threshold:
                    stage = self.predictions[i]
                    low_conf.append(
                        {
                            "epoch": i + 1,
                            "time_sec": i * self.epoch_sec,
                            "stage": STAGE_DISPLAY_NAMES[stage],
                            "confidence": conf,
                        }
                    )

            return low_conf

        def get_confidence_summary(self):
            """Get summary statistics about prediction confidence.

            Returns:
                dict with confidence statistics
            """
            if self.confidence is None:
                return None

            return _summarize_confidences(self.confidence, self.confidence_threshold)

    return _HypnogramCanvas


class SleepStatistics:
    """Calculate comprehensive sleep statistics from predictions."""

    def __init__(self, predictions: np.ndarray, epoch_sec: int = 30):
        """Initialize with predictions array.

        Args:
            predictions: Array of stage predictions (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
            epoch_sec: Duration of each epoch in seconds
        """
        self.predictions = predictions
        self.epoch_sec = epoch_sec
        self.epoch_min = epoch_sec / 60.0

        # Calculate all statistics
        self.stats = self._calculate_statistics()

    def _calculate_statistics(self) -> dict:
        """Calculate all sleep statistics."""
        n_epochs = len(self.predictions)
        total_recording_min = n_epochs * self.epoch_min

        # Count epochs per stage
        n_wake = (self.predictions == 0).sum()
        n_n1 = (self.predictions == 1).sum()
        n_n2 = (self.predictions == 2).sum()
        n_n3 = (self.predictions == 3).sum()
        n_rem = (self.predictions == 4).sum()

        # Total sleep time (TST) = all non-wake epochs
        n_sleep = n_n1 + n_n2 + n_n3 + n_rem
        tst_min = n_sleep * self.epoch_min

        # Total NREM
        n_nrem = n_n1 + n_n2 + n_n3

        # Sleep Onset Latency (SOL) - time to first sleep epoch
        sol_min = 0
        first_sleep_idx = np.where(self.predictions != 0)[0]
        if len(first_sleep_idx) > 0:
            sol_min = first_sleep_idx[0] * self.epoch_min

        # REM Onset Latency (time from first sleep to first REM)
        rem_latency_min = 0
        first_rem_idx = np.where(self.predictions == 4)[0]
        if len(first_sleep_idx) > 0 and len(first_rem_idx) > 0:
            if first_rem_idx[0] > first_sleep_idx[0]:
                rem_latency_min = (
                    first_rem_idx[0] - first_sleep_idx[0]
                ) * self.epoch_min

        # Wake After Sleep Onset (WASO) - wake epochs after first sleep epoch
        waso_min = 0
        if len(first_sleep_idx) > 0:
            after_sleep_onset = self.predictions[first_sleep_idx[0] :]
            n_waso = (after_sleep_onset == 0).sum()
            waso_min = n_waso * self.epoch_min

        # Time in Bed (TIB) - total recording time
        tib_min = total_recording_min

        # Sleep Period Time (SPT) - time from first sleep to last sleep epoch
        spt_min = 0
        last_sleep_idx = np.where(self.predictions != 0)[0]
        if len(first_sleep_idx) > 0 and len(last_sleep_idx) > 0:
            spt_min = (last_sleep_idx[-1] - first_sleep_idx[0] + 1) * self.epoch_min

        # Sleep Efficiency - (TST / TIB) * 100
        sleep_efficiency = (tst_min / tib_min * 100) if tib_min > 0 else 0

        # Sleep Maintenance Efficiency - (TST / SPT) * 100
        sleep_maintenance = (tst_min / spt_min * 100) if spt_min > 0 else 0

        # Number of awakenings (transitions from sleep to wake)
        n_awakenings = 0
        for i in range(1, len(self.predictions)):
            if self.predictions[i] == 0 and self.predictions[i - 1] != 0:
                n_awakenings += 1

        # Number of stage shifts
        n_stage_shifts = 0
        for i in range(1, len(self.predictions)):
            if self.predictions[i] != self.predictions[i - 1]:
                n_stage_shifts += 1

        # Sleep fragmentation index (stage shifts per hour of sleep)
        fragmentation_index = (n_stage_shifts / (tst_min / 60)) if tst_min > 0 else 0

        # Arousal index (awakenings per hour of sleep)
        arousal_index = (n_awakenings / (tst_min / 60)) if tst_min > 0 else 0

        # Stage percentages (of TST)
        pct_n1 = (n_n1 / n_sleep * 100) if n_sleep > 0 else 0
        pct_n2 = (n_n2 / n_sleep * 100) if n_sleep > 0 else 0
        pct_n3 = (n_n3 / n_sleep * 100) if n_sleep > 0 else 0
        pct_rem = (n_rem / n_sleep * 100) if n_sleep > 0 else 0
        pct_nrem = (n_nrem / n_sleep * 100) if n_sleep > 0 else 0

        # Stage percentages (of TIB)
        pct_n1_tib = (n_n1 / n_epochs * 100) if n_epochs > 0 else 0
        pct_n2_tib = (n_n2 / n_epochs * 100) if n_epochs > 0 else 0
        pct_n3_tib = (n_n3 / n_epochs * 100) if n_epochs > 0 else 0
        pct_rem_tib = (n_rem / n_epochs * 100) if n_epochs > 0 else 0
        pct_wake_tib = (n_wake / n_epochs * 100) if n_epochs > 0 else 0

        return {
            # Recording info
            "total_epochs": n_epochs,
            "epoch_duration_sec": self.epoch_sec,
            "total_recording_min": total_recording_min,
            "total_recording_hr": total_recording_min / 60,
            # Time in each stage (minutes)
            "wake_min": n_wake * self.epoch_min,
            "n1_min": n_n1 * self.epoch_min,
            "n2_min": n_n2 * self.epoch_min,
            "n3_min": n_n3 * self.epoch_min,
            "rem_min": n_rem * self.epoch_min,
            "nrem_min": n_nrem * self.epoch_min,
            "tst_min": tst_min,
            # Time in each stage (hours)
            "wake_hr": n_wake * self.epoch_min / 60,
            "n1_hr": n_n1 * self.epoch_min / 60,
            "n2_hr": n_n2 * self.epoch_min / 60,
            "n3_hr": n_n3 * self.epoch_min / 60,
            "rem_hr": n_rem * self.epoch_min / 60,
            "nrem_hr": n_nrem * self.epoch_min / 60,
            "tst_hr": tst_min / 60,
            # Epoch counts
            "n_wake": n_wake,
            "n_n1": n_n1,
            "n_n2": n_n2,
            "n_n3": n_n3,
            "n_rem": n_rem,
            "n_sleep": n_sleep,
            # Latencies
            "sol_min": sol_min,
            "rem_latency_min": rem_latency_min,
            # Sleep architecture
            "waso_min": waso_min,
            "tib_min": tib_min,
            "spt_min": spt_min,
            "sleep_efficiency": sleep_efficiency,
            "sleep_maintenance": sleep_maintenance,
            # Percentages (of TST)
            "pct_n1": pct_n1,
            "pct_n2": pct_n2,
            "pct_n3": pct_n3,
            "pct_rem": pct_rem,
            "pct_nrem": pct_nrem,
            # Percentages (of TIB)
            "pct_wake_tib": pct_wake_tib,
            "pct_n1_tib": pct_n1_tib,
            "pct_n2_tib": pct_n2_tib,
            "pct_n3_tib": pct_n3_tib,
            "pct_rem_tib": pct_rem_tib,
            # Fragmentation
            "n_awakenings": n_awakenings,
            "n_stage_shifts": n_stage_shifts,
            "fragmentation_index": fragmentation_index,
            "arousal_index": arousal_index,
        }

    def get_stat(self, key: str):
        """Get a specific statistic."""
        return self.stats.get(key, None)


class SleepStatisticsWidget(QWidget):
    """Widget for displaying comprehensive sleep statistics."""

    # Clinical reference ranges for healthy adults (based on AASM guidelines)
    # Format: (min_normal, max_normal, unit, description)
    NORMAL_RANGES = {
        "sleep_efficiency": (85, 95, "%", "Normal: ≥85%"),
        "sol_min": (0, 30, "min", "Normal: <30 min"),
        "rem_latency_min": (60, 150, "min", "Normal: 60-150 min"),
        "waso_min": (0, 30, "min", "Normal: <30 min"),
        "arousal_index": (0, 15, "/hr", "Normal: <15/hr"),
        "fragmentation_index": (0, 20, "/hr", "Normal: <20/hr"),
        "pct_n1": (2, 10, "%", "Normal: 2-10% of TST"),
        "pct_n2": (40, 60, "%", "Normal: 40-60% of TST"),
        "pct_n3": (10, 25, "%", "Normal: 10-25% of TST"),
        "pct_rem": (15, 30, "%", "Normal: 15-30% of TST"),
        "tst_hr": (6, 9, "hr", "Recommended: 7-9 hours"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.statistics = None
        self.confidence_threshold = DEFAULT_LOW_CONFIDENCE_THRESHOLD
        self._current_predictions: np.ndarray | None = None
        self._current_confidences: np.ndarray | None = None
        self._current_epoch_sec = 30
        self.init_ui()

    def _format_with_range(
        self, value: float, key: str, format_str: str = "{:.1f}"
    ) -> str:
        """Format a value with color coding based on normal range.

        Args:
            value: The value to format
            key: Key to look up in NORMAL_RANGES
            format_str: Format string for the value

        Returns:
            HTML formatted string with color coding
        """
        if key not in self.NORMAL_RANGES:
            return format_str.format(value)

        min_normal, max_normal, unit, desc = self.NORMAL_RANGES[key]

        # Determine color based on whether value is in normal range
        if min_normal <= value <= max_normal:
            color = "#4CAF50"  # Green - normal
            status = "✓"
        elif key in ["sol_min", "waso_min", "arousal_index", "fragmentation_index"]:
            # For these, lower is better
            if value < min_normal:
                color = "#4CAF50"  # Green - good
                status = "✓"
            else:
                color = "#F44336"  # Red - elevated
                status = "↑"
        elif key == "sleep_efficiency":
            if value >= min_normal:
                color = "#4CAF50"  # Green
                status = "✓"
            elif value >= 70:
                color = "#FF9800"  # Orange
                status = "↓"
            else:
                color = "#F44336"  # Red
                status = "↓↓"
        else:
            # Outside range - determine if too high or too low
            if value < min_normal:
                color = "#FF9800"  # Orange - low
                status = "↓"
            else:
                color = "#FF9800"  # Orange - high
                status = "↑"

        formatted_value = format_str.format(value)
        return f"<font color='{color}'><b>{formatted_value}</b></font> <small style='color:#888;'>{status}</small>"

    def _get_range_tooltip(self, key: str) -> str:
        """Get tooltip text for a statistic with its normal range."""
        if key not in self.NORMAL_RANGES:
            return ""
        min_normal, max_normal, unit, desc = self.NORMAL_RANGES[key]
        return desc

    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(0, 0)

        # Title
        title = QLabel("Sleep Statistics Summary")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Create scroll area for statistics
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setMinimumSize(0, 0)
        scroll_widget = QWidget()
        scroll_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        scroll_widget.setMinimumSize(0, 0)
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)

        # Recording Information
        self.recording_group = self._create_recording_info_group()
        scroll_layout.addWidget(self.recording_group)

        # Sleep Architecture
        self.architecture_group = self._create_architecture_group()
        scroll_layout.addWidget(self.architecture_group)

        # Stage Duration
        self.duration_group = self._create_stage_duration_group()
        scroll_layout.addWidget(self.duration_group)

        # Stage Percentages
        self.percentage_group = self._create_stage_percentage_group()
        scroll_layout.addWidget(self.percentage_group)

        # Sleep Fragmentation
        self.fragmentation_group = self._create_fragmentation_group()
        scroll_layout.addWidget(self.fragmentation_group)

        # Prediction Confidence Summary
        self.confidence_group = self._create_confidence_group()
        scroll_layout.addWidget(self.confidence_group)

        scroll_layout.addStretch()

        # Export button
        export_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Statistics to CSV")
        self.export_button.clicked.connect(self.export_statistics)
        self.export_button.setEnabled(False)
        export_layout.addWidget(self.export_button)
        export_layout.addStretch()
        layout.addLayout(export_layout)

    def _create_recording_info_group(self) -> QGroupBox:
        """Create recording information group."""
        group = QGroupBox("Recording Information")
        group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout = QGridLayout()
        group.setLayout(layout)

        self.total_epochs_label = QLabel("--")
        self.epoch_duration_label = QLabel("--")
        self.recording_duration_label = QLabel("--")

        layout.addWidget(QLabel("Total Epochs:"), 0, 0)
        layout.addWidget(self.total_epochs_label, 0, 1)
        layout.addWidget(QLabel("Epoch Duration:"), 1, 0)
        layout.addWidget(self.epoch_duration_label, 1, 1)
        layout.addWidget(QLabel("Recording Duration:"), 2, 0)
        layout.addWidget(self.recording_duration_label, 2, 1)

        layout.setColumnStretch(2, 1)
        return group

    def _create_architecture_group(self) -> QGroupBox:
        """Create sleep architecture group."""
        group = QGroupBox("Sleep Architecture")
        group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout = QGridLayout()
        group.setLayout(layout)

        # Create labels
        self.tst_label = QLabel("--")
        self.tib_label = QLabel("--")
        self.spt_label = QLabel("--")
        self.sol_label = QLabel("--")
        self.rem_latency_label = QLabel("--")
        self.waso_label = QLabel("--")
        self.sleep_efficiency_label = QLabel("--")
        self.sleep_maintenance_label = QLabel("--")

        # Bold font for labels
        bold_font = QFont("Arial", 10, QFont.Weight.Bold)

        # Add to layout with descriptions
        row = 0
        layout.addWidget(QLabel("Total Sleep Time (TST):"), row, 0)
        layout.addWidget(self.tst_label, row, 1)
        layout.addWidget(QLabel("(All non-wake sleep)"), row, 2)

        row += 1
        layout.addWidget(QLabel("Time in Bed (TIB):"), row, 0)
        layout.addWidget(self.tib_label, row, 1)
        layout.addWidget(QLabel("(Total recording time)"), row, 2)

        row += 1
        layout.addWidget(QLabel("Sleep Period Time (SPT):"), row, 0)
        layout.addWidget(self.spt_label, row, 1)
        layout.addWidget(QLabel("(First to last sleep epoch)"), row, 2)

        row += 1
        layout.addWidget(QLabel("Sleep Onset Latency (SOL):"), row, 0)
        layout.addWidget(self.sol_label, row, 1)
        layout.addWidget(QLabel("(Time to first sleep epoch)"), row, 2)

        row += 1
        layout.addWidget(QLabel("REM Latency:"), row, 0)
        layout.addWidget(self.rem_latency_label, row, 1)
        layout.addWidget(QLabel("(From sleep onset to first REM)"), row, 2)

        row += 1
        layout.addWidget(QLabel("Wake After Sleep Onset (WASO):"), row, 0)
        layout.addWidget(self.waso_label, row, 1)
        layout.addWidget(QLabel("(Wake time after first sleep)"), row, 2)

        row += 1
        separator = QLabel("─" * 60)
        separator.setStyleSheet("color: #888;")
        layout.addWidget(separator, row, 0, 1, 3)

        row += 1
        se_label = QLabel("Sleep Efficiency:")
        se_label.setFont(bold_font)
        self.sleep_efficiency_label.setFont(bold_font)
        layout.addWidget(se_label, row, 0)
        layout.addWidget(self.sleep_efficiency_label, row, 1)
        layout.addWidget(QLabel("(TST / TIB × 100)"), row, 2)

        row += 1
        sm_label = QLabel("Sleep Maintenance:")
        sm_label.setFont(bold_font)
        self.sleep_maintenance_label.setFont(bold_font)
        layout.addWidget(sm_label, row, 0)
        layout.addWidget(self.sleep_maintenance_label, row, 1)
        layout.addWidget(QLabel("(TST / SPT × 100)"), row, 2)

        layout.setColumnStretch(3, 1)
        return group

    def _create_stage_duration_group(self) -> QGroupBox:
        """Create stage duration group."""
        group = QGroupBox("Time in Each Stage")
        group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout = QGridLayout()
        group.setLayout(layout)

        # Headers
        layout.addWidget(QLabel("Stage"), 0, 0)
        layout.addWidget(QLabel("Minutes"), 0, 1)
        layout.addWidget(QLabel("Hours"), 0, 2)
        layout.addWidget(QLabel("Epochs"), 0, 3)

        # Stage labels
        self.wake_duration_label = QLabel("--")
        self.n1_duration_label = QLabel("--")
        self.n2_duration_label = QLabel("--")
        self.n3_duration_label = QLabel("--")
        self.rem_duration_label = QLabel("--")
        self.nrem_duration_label = QLabel("--")

        self.wake_hours_label = QLabel("--")
        self.n1_hours_label = QLabel("--")
        self.n2_hours_label = QLabel("--")
        self.n3_hours_label = QLabel("--")
        self.rem_hours_label = QLabel("--")
        self.nrem_hours_label = QLabel("--")

        self.wake_epochs_label = QLabel("--")
        self.n1_epochs_label = QLabel("--")
        self.n2_epochs_label = QLabel("--")
        self.n3_epochs_label = QLabel("--")
        self.rem_epochs_label = QLabel("--")

        # Add to layout
        row = 1
        layout.addWidget(QLabel("Wake"), row, 0)
        layout.addWidget(self.wake_duration_label, row, 1)
        layout.addWidget(self.wake_hours_label, row, 2)
        layout.addWidget(self.wake_epochs_label, row, 3)

        row += 1
        layout.addWidget(QLabel("N1"), row, 0)
        layout.addWidget(self.n1_duration_label, row, 1)
        layout.addWidget(self.n1_hours_label, row, 2)
        layout.addWidget(self.n1_epochs_label, row, 3)

        row += 1
        layout.addWidget(QLabel("N2"), row, 0)
        layout.addWidget(self.n2_duration_label, row, 1)
        layout.addWidget(self.n2_hours_label, row, 2)
        layout.addWidget(self.n2_epochs_label, row, 3)

        row += 1
        layout.addWidget(QLabel("N3"), row, 0)
        layout.addWidget(self.n3_duration_label, row, 1)
        layout.addWidget(self.n3_hours_label, row, 2)
        layout.addWidget(self.n3_epochs_label, row, 3)

        row += 1
        layout.addWidget(QLabel("REM"), row, 0)
        layout.addWidget(self.rem_duration_label, row, 1)
        layout.addWidget(self.rem_hours_label, row, 2)
        layout.addWidget(self.rem_epochs_label, row, 3)

        row += 1
        separator = QLabel("─" * 60)
        separator.setStyleSheet("color: #888;")
        layout.addWidget(separator, row, 0, 1, 4)

        row += 1
        nrem_label = QLabel("Total NREM")
        nrem_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(nrem_label, row, 0)
        layout.addWidget(self.nrem_duration_label, row, 1)
        layout.addWidget(self.nrem_hours_label, row, 2)

        layout.setColumnStretch(4, 1)
        return group

    def _create_stage_percentage_group(self) -> QGroupBox:
        """Create stage percentage group."""
        group = QGroupBox("Stage Composition")
        group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout = QGridLayout()
        group.setLayout(layout)

        # Labels
        self.pct_n1_label = QLabel("--")
        self.pct_n2_label = QLabel("--")
        self.pct_n3_label = QLabel("--")
        self.pct_rem_label = QLabel("--")
        self.pct_nrem_label = QLabel("--")

        self.pct_wake_tib_label = QLabel("--")
        self.pct_n1_tib_label = QLabel("--")
        self.pct_n2_tib_label = QLabel("--")
        self.pct_n3_tib_label = QLabel("--")
        self.pct_rem_tib_label = QLabel("--")

        # Layout
        layout.addWidget(QLabel("Stage"), 0, 0)
        layout.addWidget(QLabel("% of TST"), 0, 1)
        layout.addWidget(QLabel("% of TIB"), 0, 2)

        row = 1
        layout.addWidget(QLabel("Wake"), row, 0)
        layout.addWidget(QLabel("--"), row, 1)
        layout.addWidget(self.pct_wake_tib_label, row, 2)

        row += 1
        layout.addWidget(QLabel("N1"), row, 0)
        layout.addWidget(self.pct_n1_label, row, 1)
        layout.addWidget(self.pct_n1_tib_label, row, 2)

        row += 1
        layout.addWidget(QLabel("N2"), row, 0)
        layout.addWidget(self.pct_n2_label, row, 1)
        layout.addWidget(self.pct_n2_tib_label, row, 2)

        row += 1
        layout.addWidget(QLabel("N3"), row, 0)
        layout.addWidget(self.pct_n3_label, row, 1)
        layout.addWidget(self.pct_n3_tib_label, row, 2)

        row += 1
        layout.addWidget(QLabel("REM"), row, 0)
        layout.addWidget(self.pct_rem_label, row, 1)
        layout.addWidget(self.pct_rem_tib_label, row, 2)

        row += 1
        separator = QLabel("─" * 60)
        separator.setStyleSheet("color: #888;")
        layout.addWidget(separator, row, 0, 1, 3)

        row += 1
        nrem_label = QLabel("Total NREM")
        nrem_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(nrem_label, row, 0)
        layout.addWidget(self.pct_nrem_label, row, 1)

        layout.setColumnStretch(3, 1)
        return group

    def _create_fragmentation_group(self) -> QGroupBox:
        """Create sleep fragmentation group."""
        group = QGroupBox("Sleep Fragmentation")
        group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout = QGridLayout()
        group.setLayout(layout)

        self.n_awakenings_label = QLabel("--")
        self.arousal_index_label = QLabel("--")
        self.n_stage_shifts_label = QLabel("--")
        self.fragmentation_index_label = QLabel("--")

        layout.addWidget(QLabel("Number of Awakenings:"), 0, 0)
        layout.addWidget(self.n_awakenings_label, 0, 1)
        layout.addWidget(QLabel("(Sleep → Wake transitions)"), 0, 2)

        layout.addWidget(QLabel("Arousal Index:"), 1, 0)
        layout.addWidget(self.arousal_index_label, 1, 1)
        layout.addWidget(QLabel("(Awakenings per hour of sleep)"), 1, 2)

        layout.addWidget(QLabel("Number of Stage Shifts:"), 2, 0)
        layout.addWidget(self.n_stage_shifts_label, 2, 1)
        layout.addWidget(QLabel("(All stage transitions)"), 2, 2)

        layout.addWidget(QLabel("Fragmentation Index:"), 3, 0)
        layout.addWidget(self.fragmentation_index_label, 3, 1)
        layout.addWidget(QLabel("(Stage shifts per hour of sleep)"), 3, 2)

        layout.setColumnStretch(3, 1)
        return group

    def _create_confidence_group(self) -> QGroupBox:
        """Create prediction confidence summary group."""
        group = QGroupBox("Prediction Confidence")
        group.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout = QGridLayout()
        group.setLayout(layout)

        # Summary stats
        self.mean_confidence_label = QLabel("--")
        self.median_confidence_label = QLabel("--")
        self.min_confidence_label = QLabel("--")

        layout.addWidget(QLabel("Mean Confidence:"), 0, 0)
        layout.addWidget(self.mean_confidence_label, 0, 1)
        layout.addWidget(QLabel("(Average prediction certainty)"), 0, 2)

        layout.addWidget(QLabel("Median Confidence:"), 1, 0)
        layout.addWidget(self.median_confidence_label, 1, 1)
        layout.addWidget(QLabel("(Middle value)"), 1, 2)

        layout.addWidget(QLabel("Minimum Confidence:"), 2, 0)
        layout.addWidget(self.min_confidence_label, 2, 1)
        layout.addWidget(QLabel("(Least certain epoch)"), 2, 2)

        # Distribution by confidence level
        self.high_conf_label = QLabel("--")
        self.medium_conf_label = QLabel("--")
        self.low_conf_label = QLabel("--")

        self.high_conf_title_label = QLabel(
            f"High Confidence (≥{HIGH_CONFIDENCE_THRESHOLD:.0%}):"
        )
        layout.addWidget(self.high_conf_title_label, 3, 0)
        layout.addWidget(self.high_conf_label, 3, 1)
        layout.addWidget(
            QLabel(
                f"<span style='color:{CONFIDENCE_BAND_COLORS['high']}'>●</span> Very reliable"
            ),
            3,
            2,
        )

        self.medium_conf_title_label = QLabel()
        layout.addWidget(self.medium_conf_title_label, 4, 0)
        layout.addWidget(self.medium_conf_label, 4, 1)
        layout.addWidget(
            QLabel(
                f"<span style='color:{CONFIDENCE_BAND_COLORS['medium']}'>●</span> Review recommended"
            ),
            4,
            2,
        )

        self.low_conf_title_label = QLabel()
        layout.addWidget(self.low_conf_title_label, 5, 0)
        layout.addWidget(self.low_conf_label, 5, 1)
        layout.addWidget(
            QLabel(
                f"<span style='color:{CONFIDENCE_BAND_COLORS['low']}'>●</span> Manual review needed"
            ),
            5,
            2,
        )

        # Uncertain epochs list
        self.uncertain_epochs_label = QLabel("--")
        self.uncertain_epochs_label.setWordWrap(True)
        self.uncertain_epochs_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(QLabel("Uncertain Epochs:"), 6, 0)
        layout.addWidget(self.uncertain_epochs_label, 6, 1, 1, 2)

        layout.setColumnStretch(3, 1)
        self._update_confidence_group_labels()
        return group

    def _update_confidence_group_labels(self):
        """Refresh confidence group captions to match the active threshold."""
        self.medium_conf_title_label.setText(
            f"Medium Confidence ({self.confidence_threshold:.0%}-{HIGH_CONFIDENCE_THRESHOLD:.0%}):"
        )
        self.low_conf_title_label.setText(
            f"Low Confidence (<{self.confidence_threshold:.0%}):"
        )

    def update_statistics(
        self,
        predictions: np.ndarray,
        epoch_sec: int = 30,
        confidences: np.ndarray | None = None,
    ):
        """Update statistics display with new predictions.

        Args:
            predictions: Array of stage predictions (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
            epoch_sec: Duration of each epoch in seconds
        """
        self._current_predictions = predictions
        self._current_confidences = confidences
        self._current_epoch_sec = epoch_sec
        self.statistics = SleepStatistics(predictions, epoch_sec)
        stats = self.statistics.stats

        # Update recording info
        self.total_epochs_label.setText(f"{stats['total_epochs']}")
        self.epoch_duration_label.setText(f"{stats['epoch_duration_sec']} seconds")
        self.recording_duration_label.setText(
            f"{stats['total_recording_min']:.1f} min ({stats['total_recording_hr']:.2f} hr)"
        )

        # Update architecture with normal range indicators
        tst_formatted = self._format_with_range(stats["tst_hr"], "tst_hr", "{:.2f}")
        self.tst_label.setText(f"{stats['tst_min']:.1f} min ({tst_formatted} hr)")
        self.tst_label.setToolTip(self._get_range_tooltip("tst_hr"))

        self.tib_label.setText(
            f"{stats['tib_min']:.1f} min ({stats['tib_min'] / 60:.2f} hr)"
        )
        self.spt_label.setText(
            f"{stats['spt_min']:.1f} min ({stats['spt_min'] / 60:.2f} hr)"
        )

        # SOL with range indicator
        sol_formatted = self._format_with_range(stats["sol_min"], "sol_min")
        self.sol_label.setText(f"{sol_formatted} min")
        self.sol_label.setToolTip(self._get_range_tooltip("sol_min"))

        # REM latency with range indicator
        rem_lat_formatted = self._format_with_range(
            stats["rem_latency_min"], "rem_latency_min"
        )
        self.rem_latency_label.setText(f"{rem_lat_formatted} min")
        self.rem_latency_label.setToolTip(self._get_range_tooltip("rem_latency_min"))

        # WASO with range indicator
        waso_formatted = self._format_with_range(stats["waso_min"], "waso_min")
        self.waso_label.setText(f"{waso_formatted} min")
        self.waso_label.setToolTip(self._get_range_tooltip("waso_min"))

        # Sleep efficiency with color coding
        se_formatted = self._format_with_range(
            stats["sleep_efficiency"], "sleep_efficiency"
        )
        self.sleep_efficiency_label.setText(f"{se_formatted}%")
        self.sleep_efficiency_label.setToolTip(
            self._get_range_tooltip("sleep_efficiency")
        )

        self.sleep_maintenance_label.setText(
            f"<b>{stats['sleep_maintenance']:.1f}%</b>"
        )

        # Update durations
        self.wake_duration_label.setText(f"{stats['wake_min']:.1f}")
        self.n1_duration_label.setText(f"{stats['n1_min']:.1f}")
        self.n2_duration_label.setText(f"{stats['n2_min']:.1f}")
        self.n3_duration_label.setText(f"{stats['n3_min']:.1f}")
        self.rem_duration_label.setText(f"{stats['rem_min']:.1f}")
        self.nrem_duration_label.setText(f"<b>{stats['nrem_min']:.1f}</b>")

        self.wake_hours_label.setText(f"{stats['wake_hr']:.2f}")
        self.n1_hours_label.setText(f"{stats['n1_hr']:.2f}")
        self.n2_hours_label.setText(f"{stats['n2_hr']:.2f}")
        self.n3_hours_label.setText(f"{stats['n3_hr']:.2f}")
        self.rem_hours_label.setText(f"{stats['rem_hr']:.2f}")
        self.nrem_hours_label.setText(f"<b>{stats['nrem_hr']:.2f}</b>")

        self.wake_epochs_label.setText(f"{stats['n_wake']}")
        self.n1_epochs_label.setText(f"{stats['n_n1']}")
        self.n2_epochs_label.setText(f"{stats['n_n2']}")
        self.n3_epochs_label.setText(f"{stats['n_n3']}")
        self.rem_epochs_label.setText(f"{stats['n_rem']}")

        # Update percentages with color coding for normal ranges
        self.pct_n1_label.setText(
            self._format_with_range(stats["pct_n1"], "pct_n1") + "%"
        )
        self.pct_n1_label.setToolTip(self._get_range_tooltip("pct_n1"))

        self.pct_n2_label.setText(
            self._format_with_range(stats["pct_n2"], "pct_n2") + "%"
        )
        self.pct_n2_label.setToolTip(self._get_range_tooltip("pct_n2"))

        self.pct_n3_label.setText(
            self._format_with_range(stats["pct_n3"], "pct_n3") + "%"
        )
        self.pct_n3_label.setToolTip(self._get_range_tooltip("pct_n3"))

        self.pct_rem_label.setText(
            self._format_with_range(stats["pct_rem"], "pct_rem") + "%"
        )
        self.pct_rem_label.setToolTip(self._get_range_tooltip("pct_rem"))

        self.pct_nrem_label.setText(f"<b>{stats['pct_nrem']:.1f}%</b>")

        self.pct_wake_tib_label.setText(f"{stats['pct_wake_tib']:.1f}%")
        self.pct_n1_tib_label.setText(f"{stats['pct_n1_tib']:.1f}%")
        self.pct_n2_tib_label.setText(f"{stats['pct_n2_tib']:.1f}%")
        self.pct_n3_tib_label.setText(f"{stats['pct_n3_tib']:.1f}%")
        self.pct_rem_tib_label.setText(f"{stats['pct_rem_tib']:.1f}%")

        # Update fragmentation with color coding
        self.n_awakenings_label.setText(f"{stats['n_awakenings']}")

        arousal_formatted = self._format_with_range(
            stats["arousal_index"], "arousal_index", "{:.2f}"
        )
        self.arousal_index_label.setText(f"{arousal_formatted}/hr")
        self.arousal_index_label.setToolTip(self._get_range_tooltip("arousal_index"))

        self.n_stage_shifts_label.setText(f"{stats['n_stage_shifts']}")

        frag_formatted = self._format_with_range(
            stats["fragmentation_index"], "fragmentation_index", "{:.2f}"
        )
        self.fragmentation_index_label.setText(f"{frag_formatted}/hr")
        self.fragmentation_index_label.setToolTip(
            self._get_range_tooltip("fragmentation_index")
        )

        # Update confidence summary if provided
        if confidences is not None and len(confidences) > 0:
            self._update_confidence_summary(confidences, predictions, epoch_sec)

        # Enable export button
        self.export_button.setEnabled(True)

    def _update_confidence_summary(
        self, confidences: np.ndarray, predictions: np.ndarray, epoch_sec: int
    ):
        """Update confidence summary display."""
        summary = _summarize_confidences(confidences, self.confidence_threshold)
        if summary is None:
            return

        mean_conf = float(summary["mean"])
        median_conf = float(summary["median"])
        min_conf = float(summary["min"])
        min_conf_idx = int(np.argmin(confidences))
        mean_color = CONFIDENCE_BAND_COLORS[
            _get_confidence_band(mean_conf, self.confidence_threshold)
        ]

        self.mean_confidence_label.setText(
            f"<font color='{mean_color}'><b>{mean_conf:.1%}</b></font>"
        )
        self.median_confidence_label.setText(f"{median_conf:.1%}")

        # Minimum confidence with epoch info
        min_epoch_time = min_conf_idx * epoch_sec
        min_epoch_minutes = min_epoch_time // 60
        min_epoch_seconds = min_epoch_time % 60
        min_epoch_stage = (
            STAGE_LABELS[predictions[min_conf_idx]]
            if min_conf_idx < len(predictions)
            else "?"
        )
        self.min_confidence_label.setText(
            f"<font color='{CONFIDENCE_BAND_COLORS['low']}'><b>{min_conf:.1%}</b></font> "
            f"(Epoch {min_conf_idx + 1}, {min_epoch_minutes}:{min_epoch_seconds:02d}, {min_epoch_stage})"
        )

        # Confidence distribution
        high_conf = int(summary["high_count"])
        medium_conf = int(summary["medium_count"])
        low_conf = int(summary["low_count"])
        total = int(summary["n_epochs"])

        self.high_conf_label.setText(
            f"<font color='{CONFIDENCE_BAND_COLORS['high']}'><b>{high_conf}</b></font> epochs ({100 * high_conf / total:.1f}%)"
        )
        self.medium_conf_label.setText(
            f"<font color='{CONFIDENCE_BAND_COLORS['medium']}'><b>{medium_conf}</b></font> epochs ({100 * medium_conf / total:.1f}%)"
        )
        self.low_conf_label.setText(
            f"<font color='{CONFIDENCE_BAND_COLORS['low']}'><b>{low_conf}</b></font> epochs ({100 * low_conf / total:.1f}%)"
        )

        # List uncertain epochs (first 10)
        uncertain_indices = np.where(confidences < self.confidence_threshold)[0]
        if len(uncertain_indices) > 0:
            # Show first 10 uncertain epochs with times
            uncertain_info = []
            for idx in uncertain_indices[:10]:
                time_sec = idx * epoch_sec
                time_min = time_sec // 60
                time_s = time_sec % 60
                stage = (
                    STAGE_LABELS[predictions[idx]] if idx < len(predictions) else "?"
                )
                conf = confidences[idx]
                uncertain_info.append(
                    f"#{idx + 1} ({time_min}:{time_s:02d}, {stage}, {conf:.0%})"
                )

            uncertain_text = ", ".join(uncertain_info)
            if len(uncertain_indices) > 10:
                uncertain_text += f" ... and {len(uncertain_indices) - 10} more"
            self.uncertain_epochs_label.setText(
                f"<span style='color:{CONFIDENCE_BAND_COLORS['low']}'>{uncertain_text}</span>"
            )
        else:
            self.uncertain_epochs_label.setText(
                f"<span style='color:{CONFIDENCE_BAND_COLORS['high']}'>"
                f"None - all epochs meet the {self.confidence_threshold:.0%} threshold."
                "</span>"
            )

    def set_confidence_threshold(self, threshold: float):
        """Update the shared low-confidence threshold and refresh confidence stats."""
        self.confidence_threshold = float(threshold)
        self._update_confidence_group_labels()
        if (
            self._current_confidences is not None
            and self._current_predictions is not None
        ):
            self._update_confidence_summary(
                self._current_confidences,
                self._current_predictions,
                self._current_epoch_sec,
            )

    def _get_efficiency_color(self, efficiency: float) -> str:
        """Get color based on sleep efficiency value."""
        if efficiency >= 85:
            return "#2e7d32"  # Green - good
        elif efficiency >= 70:
            return "#f57c00"  # Orange - fair
        else:
            return "#c62828"  # Red - poor

    def export_statistics(self):
        """Export statistics to CSV file."""
        if self.statistics is None:
            QMessageBox.warning(self, "No Data", "No statistics to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Sleep Statistics",
            "sleep_statistics.csv",
            "CSV Files (*.csv);;All Files (*)",
        )

        if filename:
            try:
                import csv

                stats = self.statistics.stats

                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "Value", "Unit"])

                    # Recording info
                    writer.writerow(["Total Epochs", stats["total_epochs"], "epochs"])
                    writer.writerow(
                        ["Epoch Duration", stats["epoch_duration_sec"], "seconds"]
                    )
                    writer.writerow(
                        [
                            "Total Recording Time",
                            stats["total_recording_min"],
                            "minutes",
                        ]
                    )
                    writer.writerow(["", stats["total_recording_hr"], "hours"])
                    writer.writerow([])

                    # Sleep architecture
                    writer.writerow(
                        ["Total Sleep Time (TST)", stats["tst_min"], "minutes"]
                    )
                    writer.writerow(["", stats["tst_hr"], "hours"])
                    writer.writerow(["Time in Bed (TIB)", stats["tib_min"], "minutes"])
                    writer.writerow(
                        ["Sleep Period Time (SPT)", stats["spt_min"], "minutes"]
                    )
                    writer.writerow(
                        ["Sleep Onset Latency (SOL)", stats["sol_min"], "minutes"]
                    )
                    writer.writerow(
                        ["REM Latency", stats["rem_latency_min"], "minutes"]
                    )
                    writer.writerow(
                        ["Wake After Sleep Onset (WASO)", stats["waso_min"], "minutes"]
                    )
                    writer.writerow(
                        ["Sleep Efficiency", stats["sleep_efficiency"], "%"]
                    )
                    writer.writerow(
                        ["Sleep Maintenance", stats["sleep_maintenance"], "%"]
                    )
                    writer.writerow([])

                    # Stage durations
                    writer.writerow(["Wake Time", stats["wake_min"], "minutes"])
                    writer.writerow(["N1 Time", stats["n1_min"], "minutes"])
                    writer.writerow(["N2 Time", stats["n2_min"], "minutes"])
                    writer.writerow(["N3 Time", stats["n3_min"], "minutes"])
                    writer.writerow(["REM Time", stats["rem_min"], "minutes"])
                    writer.writerow(["Total NREM Time", stats["nrem_min"], "minutes"])
                    writer.writerow([])

                    # Stage percentages
                    writer.writerow(["N1 % of TST", stats["pct_n1"], "%"])
                    writer.writerow(["N2 % of TST", stats["pct_n2"], "%"])
                    writer.writerow(["N3 % of TST", stats["pct_n3"], "%"])
                    writer.writerow(["REM % of TST", stats["pct_rem"], "%"])
                    writer.writerow(["NREM % of TST", stats["pct_nrem"], "%"])
                    writer.writerow([])

                    # Fragmentation
                    writer.writerow(
                        ["Number of Awakenings", stats["n_awakenings"], "count"]
                    )
                    writer.writerow(
                        ["Arousal Index", stats["arousal_index"], "per hour"]
                    )
                    writer.writerow(
                        ["Number of Stage Shifts", stats["n_stage_shifts"], "count"]
                    )
                    writer.writerow(
                        [
                            "Fragmentation Index",
                            stats["fragmentation_index"],
                            "per hour",
                        ]
                    )

                QMessageBox.information(
                    self, "Export Successful", f"Statistics exported to:\n{filename}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Export Failed", f"Failed to export statistics:\n{str(e)}"
                )


class ConfidenceReviewWidget(QWidget):
    """Widget for reviewing per-epoch confidence and stage probabilities."""

    epoch_selected = Signal(int)

    TABLE_HEADERS = [
        "Epoch",
        "Time",
        "Predicted Stage",
        "Confidence",
        "Flag",
        "Next Stage",
        "Next Prob",
        "P_W",
        "P_N1",
        "P_N2",
        "P_N3",
        "P_REM",
        "Final Stage",
        "Override",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictions: np.ndarray | None = None
        self.final_predictions: np.ndarray | None = None
        self.probabilities: np.ndarray | None = None
        self.confidences: np.ndarray | None = None
        self.flag_scores: dict[str, np.ndarray] | None = None
        self.manual_overrides: dict[int, int] = {}
        self.epoch_sec = 30
        self.confidence_threshold = DEFAULT_LOW_CONFIDENCE_THRESHOLD
        # Filter state — populated by the filter toolbar.
        self._stage_filter: str = "All"
        self._low_conf_only: bool = False
        self._search_text: str = ""
        self._row_metadata: list[dict[str, Any]] = []
        self.init_ui()

    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(0, 0)

        title = QLabel("Confidence Review")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "Review epoch-level stage confidence and per-stage probabilities. "
            "Rows below the active threshold are flagged for manual review and show "
            "the strongest alternate stage."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; margin-bottom: 8px;")
        layout.addWidget(desc)

        summary_group = QGroupBox("Summary")
        summary_layout = QGridLayout(summary_group)
        self.total_epochs_value = QLabel("--")
        self.mean_conf_value = QLabel("--")
        self.min_conf_value = QLabel("--")
        self.flagged_epochs_value = QLabel("--")
        self.threshold_value = QLabel("--")

        summary_layout.addWidget(QLabel("Total Epochs:"), 0, 0)
        summary_layout.addWidget(self.total_epochs_value, 0, 1)
        summary_layout.addWidget(QLabel("Mean Confidence:"), 0, 2)
        summary_layout.addWidget(self.mean_conf_value, 0, 3)
        summary_layout.addWidget(QLabel("Minimum Confidence:"), 1, 0)
        summary_layout.addWidget(self.min_conf_value, 1, 1)
        summary_layout.addWidget(QLabel("Flagged Epochs:"), 1, 2)
        summary_layout.addWidget(self.flagged_epochs_value, 1, 3)
        summary_layout.addWidget(QLabel("Threshold:"), 2, 0)
        summary_layout.addWidget(self.threshold_value, 2, 1)
        self.flag_source_value = QLabel("--")
        summary_layout.addWidget(QLabel("Uncertainty source:"), 2, 2)
        summary_layout.addWidget(self.flag_source_value, 2, 3)
        layout.addWidget(summary_group)

        filter_group = QGroupBox("Filter")
        filter_layout = QHBoxLayout(filter_group)
        filter_layout.setContentsMargins(8, 6, 8, 6)

        filter_layout.addWidget(QLabel("Stage:"))
        self.stage_filter_combo = QComboBox()
        self.stage_filter_combo.addItems(
            ["All", "W", "N1", "N2", "N3", "REM", "Overridden"]
        )
        self.stage_filter_combo.setToolTip(
            "Show only rows for the chosen stage, or 'Overridden' to list epochs "
            "that have been manually rescored."
        )
        self.stage_filter_combo.currentTextChanged.connect(
            self._on_stage_filter_changed
        )
        filter_layout.addWidget(self.stage_filter_combo)

        self.low_conf_only_checkbox = QCheckBox("Low confidence only")
        self.low_conf_only_checkbox.setToolTip(
            "Show only epochs scoring below the current confidence threshold."
        )
        self.low_conf_only_checkbox.toggled.connect(self._on_low_conf_only_toggled)
        filter_layout.addWidget(self.low_conf_only_checkbox)

        filter_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Epoch # or HH:MM:SS")
        self.search_edit.setToolTip(
            "Filter rows by epoch number (e.g. 412) or timestamp (e.g. 01:23)."
        )
        self.search_edit.textChanged.connect(self._on_search_text_changed)
        self.search_edit.setMaximumWidth(220)
        filter_layout.addWidget(self.search_edit)

        self.clear_filters_button = QPushButton("Clear")
        self.clear_filters_button.setToolTip("Reset all filters.")
        self.clear_filters_button.clicked.connect(self._reset_filters)
        filter_layout.addWidget(self.clear_filters_button)

        self.prev_uncertain_button = QPushButton("◀ Uncertain")
        self.prev_uncertain_button.setToolTip(
            "Select the previous epoch in uncertainty-ranked order "
            "(most-uncertain first)."
        )
        self.prev_uncertain_button.clicked.connect(lambda: self._jump_uncertain(-1))
        filter_layout.addWidget(self.prev_uncertain_button)

        self.next_uncertain_button = QPushButton("Uncertain ▶")
        self.next_uncertain_button.setToolTip(
            "Select the next most-uncertain epoch (uses per-epoch flag scores "
            "when available, otherwise confidence)."
        )
        self.next_uncertain_button.clicked.connect(lambda: self._jump_uncertain(1))
        filter_layout.addWidget(self.next_uncertain_button)

        self.visible_count_label = QLabel("")
        self.visible_count_label.setStyleSheet("color: #888;")
        filter_layout.addStretch()
        filter_layout.addWidget(self.visible_count_label)

        layout.addWidget(filter_group)

        self.table = QTableWidget(0, len(self.TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(self.TABLE_HEADERS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(self.table.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(self.table.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.table.setMinimumSize(0, 0)
        self.table.itemSelectionChanged.connect(self._emit_selected_epoch)
        layout.addWidget(self.table, 1)

    def set_confidence_threshold(self, threshold: float):
        """Update the threshold and refresh current records."""
        self.confidence_threshold = float(threshold)
        self._refresh()

    def update_review(
        self,
        predictions: np.ndarray | None,
        probabilities: np.ndarray | None,
        confidences: np.ndarray | None,
        epoch_sec: int = 30,
        *,
        final_predictions: np.ndarray | None = None,
        manual_overrides: dict[int, int] | None = None,
        flag_scores: dict[str, np.ndarray] | None = None,
    ):
        """Update the review table with the latest inference outputs."""
        self.predictions = predictions
        self.final_predictions = final_predictions
        self.probabilities = probabilities
        self.confidences = _resolve_confidences(probabilities, confidences)
        self.flag_scores = flag_scores
        self.manual_overrides = _coerce_manual_score_overrides(
            manual_overrides,
            n_epochs=(len(predictions) if predictions is not None else None),
        )
        self.epoch_sec = epoch_sec
        self._refresh()

    def _refresh(self):
        """Rebuild summary labels and table rows from the current state."""
        self.threshold_value.setText(f"{self.confidence_threshold:.0%}")

        if self.predictions is None or len(self.predictions) == 0:
            self.total_epochs_value.setText("--")
            self.mean_conf_value.setText("--")
            self.min_conf_value.setText("--")
            self.flagged_epochs_value.setText("--")
            self.table.setRowCount(0)
            return

        records = _build_epoch_confidence_records(
            self.predictions,
            final_predictions=self.final_predictions,
            manual_overrides=self.manual_overrides,
            epoch_sec=self.epoch_sec,
            probabilities=self.probabilities,
            confidences=self.confidences,
            threshold=self.confidence_threshold,
            flag_scores=self.flag_scores,
        )
        summary = _summarize_confidences(self.confidences, self.confidence_threshold)

        self.total_epochs_value.setText(str(len(records)))
        if summary is not None:
            mean_band = _get_confidence_band(
                float(summary["mean"]), self.confidence_threshold
            )
            self.mean_conf_value.setText(
                f"<font color='{CONFIDENCE_BAND_COLORS[mean_band]}'><b>{float(summary['mean']):.1%}</b></font>"
            )
            self.min_conf_value.setText(
                f"<font color='{CONFIDENCE_BAND_COLORS['low']}'><b>{float(summary['min']):.1%}</b></font>"
            )
            self.flagged_epochs_value.setText(
                f"<font color='{CONFIDENCE_BAND_COLORS['low']}'><b>{int(summary['low_count'])}</b></font>"
            )
        else:
            self.mean_conf_value.setText("--")
            self.min_conf_value.setText("--")
            self.flagged_epochs_value.setText("--")

        flag_source = str(records[0]["flag_source"]) if records else "confidence"
        self.flag_source_value.setText(
            "flag entropy"
            if flag_source == "flags"
            else "confidence (flags unavailable)"
        )

        self.table.setRowCount(len(records))
        self._row_metadata = [
            {
                "epoch": int(rec["epoch"]),
                "stage": str(rec["stage"]),
                "final_stage": str(rec["final_stage"]),
                "low_confidence": bool(rec["low_confidence"]),
                "manual_override": bool(rec["manual_override"]),
                "time_hms": str(rec["time_hms"]),
                "uncertainty_score": rec["uncertainty_score"],
            }
            for rec in records
        ]
        for row, record in enumerate(records):
            confidence = record["confidence"]
            band = record["band"]
            probs = record["probabilities"]
            uncertainty = record["uncertainty_score"]
            if uncertainty is not None:
                flag_text = f"{float(uncertainty):.2f}"
                if record["low_confidence"]:
                    flag_text += " ⚑"
            else:
                flag_text = "LOW" if record["low_confidence"] else ""
            row_values = [
                str(record["epoch"]),
                str(record["time_hms"]),
                str(record["stage"]),
                f"{float(confidence):.1%}" if confidence is not None else "—",
                flag_text,
                (
                    str(record["next_stage"])
                    if record["low_confidence"] and record["next_stage"] is not None
                    else ""
                ),
                (
                    f"{float(record['next_stage_probability']):.1%}"
                    if record["low_confidence"]
                    and record["next_stage_probability"] is not None
                    else ""
                ),
                *(
                    [f"{float(prob):.1%}" for prob in probs]
                    if probs is not None
                    else ["—"] * 5
                ),
                str(record["final_stage"]),
                "Yes" if record["manual_override"] else "",
            ]

            for col, value in enumerate(row_values):
                item = QTableWidgetItem(value)
                if col == 4 and record["low_confidence"]:
                    item.setForeground(QColor(CONFIDENCE_BAND_COLORS["low"]))
                if col in {5, 6} and record["low_confidence"]:
                    item.setForeground(QColor(CONFIDENCE_BAND_COLORS["medium"]))
                if col == 3 and band is not None:
                    item.setForeground(QColor(CONFIDENCE_BAND_COLORS[band]))
                if col == len(self.TABLE_HEADERS) - 2 and record["manual_override"]:
                    item.setForeground(QColor("#4FC3F7"))
                if col == len(self.TABLE_HEADERS) - 1 and record["manual_override"]:
                    item.setForeground(QColor("#4FC3F7"))
                self.table.setItem(row, col, item)

            if record["low_confidence"]:
                for col in range(len(self.TABLE_HEADERS)):
                    cell_item = self.table.item(row, col)
                    if cell_item is not None:
                        cell_item.setBackground(QColor("#4d1f1f"))
            if record["manual_override"]:
                for col in range(len(self.TABLE_HEADERS)):
                    cell_item = self.table.item(row, col)
                    if cell_item is not None:
                        cell_item.setBackground(QColor("#13334a"))

        self.table.resizeColumnsToContents()
        self._apply_filters()

    def _on_stage_filter_changed(self, text: str) -> None:
        """Stage filter combo callback."""
        self._stage_filter = text or "All"
        self._apply_filters()

    def _on_low_conf_only_toggled(self, checked: bool) -> None:
        """Low-confidence-only checkbox callback."""
        self._low_conf_only = bool(checked)
        self._apply_filters()

    def _on_search_text_changed(self, text: str) -> None:
        """Free-text search box callback."""
        self._search_text = text.strip().lower()
        self._apply_filters()

    def _reset_filters(self) -> None:
        """Reset every filter back to its default."""
        self.stage_filter_combo.blockSignals(True)
        self.stage_filter_combo.setCurrentText("All")
        self.stage_filter_combo.blockSignals(False)
        self.low_conf_only_checkbox.blockSignals(True)
        self.low_conf_only_checkbox.setChecked(False)
        self.low_conf_only_checkbox.blockSignals(False)
        self.search_edit.blockSignals(True)
        self.search_edit.clear()
        self.search_edit.blockSignals(False)
        self._stage_filter = "All"
        self._low_conf_only = False
        self._search_text = ""
        self._apply_filters()

    def _apply_filters(self) -> None:
        """Hide/show table rows based on the current filter widgets."""
        if not self._row_metadata:
            self.visible_count_label.setText("")
            return
        visible = 0
        stage_filter = self._stage_filter
        search = self._search_text
        for row, meta in enumerate(self._row_metadata):
            keep = True

            # Stage filter
            if stage_filter == "Overridden":
                if not meta["manual_override"]:
                    keep = False
            elif stage_filter != "All":
                # Match against the final (post-override) stage if available,
                # otherwise the model stage.
                shown_stage = meta["final_stage"] or meta["stage"]
                if shown_stage != stage_filter:
                    keep = False

            # Low-confidence filter
            if keep and self._low_conf_only and not meta["low_confidence"]:
                keep = False

            # Free-text search across epoch number or timestamp
            if keep and search:
                if (
                    search not in str(meta["epoch"]).lower()
                    and search not in meta["time_hms"].lower()
                ):
                    keep = False

            self.table.setRowHidden(row, not keep)
            if keep:
                visible += 1

        total = len(self._row_metadata)
        if visible == total:
            self.visible_count_label.setText(f"{total} epoch(s)")
        else:
            self.visible_count_label.setText(f"{visible} of {total} epoch(s)")

    def _jump_uncertain(self, direction: int) -> None:
        """Move the selection through visible epochs in uncertainty rank order.

        Ranks the currently-visible rows by their per-epoch uncertainty score
        (most-uncertain first) and steps the selection forward/backward within
        that ranking. Falls back to the confidence-derived score when the backend
        flags are unavailable.
        """
        from spectra.review.uncertainty import rank_epochs_by_uncertainty

        if not self._row_metadata:
            return
        visible_rows = [
            r for r in range(len(self._row_metadata)) if not self.table.isRowHidden(r)
        ]
        if not visible_rows:
            return
        scores = np.array(
            [
                (
                    self._row_metadata[r].get("uncertainty_score")
                    if self._row_metadata[r].get("uncertainty_score") is not None
                    else -np.inf
                )
                for r in visible_rows
            ],
            dtype=np.float64,
        )
        order = rank_epochs_by_uncertainty(scores, descending=True)
        ranked_rows = [visible_rows[i] for i in order]

        current = self.table.currentRow()
        if current in ranked_rows:
            # Wrap around the ranked worklist so stepping past either end cycles.
            pos = (ranked_rows.index(current) + direction) % len(ranked_rows)
        else:
            pos = 0 if direction > 0 else len(ranked_rows) - 1
        target = ranked_rows[pos]
        self.select_epoch(target)
        self.epoch_selected.emit(target)

    def select_epoch(self, epoch_idx: int):
        """Select a specific epoch row without emitting duplicate logic."""
        if self.table.rowCount() == 0:
            return
        if not 0 <= epoch_idx < self.table.rowCount():
            return
        self.table.blockSignals(True)
        self.table.selectRow(epoch_idx)
        self.table.blockSignals(False)

    def _emit_selected_epoch(self):
        """Emit the selected epoch index when the table selection changes."""
        selected_items = self.table.selectedItems()
        if not selected_items:
            return
        row = selected_items[0].row()
        self.epoch_selected.emit(row)


class ReviewSignalLoader(QThread):
    """Background EDF loader for the signal review tab."""

    loaded = Signal(dict)
    error = Signal(str)

    def __init__(self, edf_path: str, channel_layout: list[str]):
        super().__init__()
        self.edf_path = edf_path
        self.channel_layout = list(channel_layout)

    def run(self):
        """Load aligned review signals off the GUI thread."""
        try:
            payload = _load_review_signal_payload(self.edf_path, self.channel_layout)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        if not self.isInterruptionRequested():
            self.loaded.emit(payload)


class SignalReviewPlotWidget(QWidget):
    """Profusion-style single-field page viewer for EDF signal review.

    All channels share one un-clipped plotting field with stacked vertical
    offsets, so a normal wave fills most of its row and large deflections
    (K-complexes, movements, slow waves) overlap — cross over — their neighbours,
    just like Compumedics Profusion. Clinical display filtering is applied to the
    whole recording once and cached; paging only slices the cached array. Each
    channel keeps its own modality sensitivity (µV/div) and calibration bar.
    """

    cursor_changed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals: np.ndarray | None = None
        self.sample_rate: float = 1.0
        self.channel_plan: list[dict[str, Any]] = []
        self.page_start_sec: float = 0.0
        self.page_duration_sec: float = float(DEFAULT_SIGNAL_REVIEW_PAGE_SEC)
        self.selected_epoch: int = 0
        self.epoch_sec: int = 30
        self.confidences: np.ndarray | None = None
        self.manual_overrides: dict[int, int] = {}
        self.confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD
        self.eeg_eog_uv_per_div: float = DEFAULT_SIGNAL_REVIEW_EEG_EOG_UV_PER_DIV
        self.emg_uv_per_div: float = DEFAULT_SIGNAL_REVIEW_EMG_UV_PER_DIV
        self.spacing_multiplier: float = SIGNAL_REVIEW_SPACING_PRESETS[
            DEFAULT_SIGNAL_REVIEW_SPACING
        ]
        self.filter_specs: dict[str, DisplayFilterSpec] = dict(DEFAULT_DISPLAY_FILTERS)
        self._current_total_duration_sec: float = 0.0
        self._page_end_sec: float = 0.0
        self._plot_widget: Any = None
        self._plot_item: Any = None
        self._cursor_line: Any = None
        self._signal_proxy = None
        self._filtered_signals: np.ndarray | None = None
        self._filter_cache_key: tuple[Any, ...] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if pg is None:
            self._fallback_label = QLabel(
                "pyqtgraph is not installed, so waveform review is unavailable."
            )
            self._fallback_label.setWordWrap(True)
            self._fallback_label.setStyleSheet("color: #FFA726; padding: 8px;")
            layout.addWidget(self._fallback_label)
            self._plot_widget = None
            self._plot_item = None
            return

        self._plot_widget = pg.PlotWidget(background=SIGNAL_REVIEW_BACKGROUND_COLOR)
        self._plot_item = self._plot_widget.getPlotItem()
        self._plot_item.setMenuEnabled(False)
        self._plot_item.hideButtons()
        self._plot_item.showGrid(x=True, y=False, alpha=0.3)
        self._plot_item.setMouseEnabled(x=True, y=False)
        left_axis = self._plot_item.getAxis("left")
        left_axis.setTextPen(pg.mkPen(SIGNAL_REVIEW_AXIS_TEXT_COLOR))
        left_axis.setPen(pg.mkPen(SIGNAL_REVIEW_AXIS_LINE_COLOR))
        left_axis.setWidth(86)
        bottom_axis = self._plot_item.getAxis("bottom")
        bottom_axis.setTextPen(pg.mkPen(SIGNAL_REVIEW_AXIS_TEXT_COLOR))
        bottom_axis.setPen(pg.mkPen(SIGNAL_REVIEW_AXIS_LINE_COLOR))
        bottom_axis.setLabel("Page Time (s)", color=SIGNAL_REVIEW_AXIS_TEXT_COLOR)
        self._plot_item.getViewBox().setBackgroundColor(SIGNAL_REVIEW_BACKGROUND_COLOR)
        layout.addWidget(self._plot_widget, 1)

        self._cursor_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen("#15407A", width=1.0, style=Qt.PenStyle.DashLine),
        )
        self._plot_item.addItem(self._cursor_line)
        self._signal_proxy = pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )

    def set_filter_specs(self, filter_specs: dict[str, DisplayFilterSpec]) -> None:
        """Replace the per-modality display filters and re-render the page."""
        self.filter_specs = dict(filter_specs)
        self._filtered_signals = None
        self._filter_cache_key = None
        self.render_page(
            page_start_sec=self.page_start_sec,
            page_duration_sec=self.page_duration_sec,
            selected_epoch=self.selected_epoch,
            epoch_sec=self.epoch_sec,
            confidences=self.confidences,
            manual_overrides=self.manual_overrides,
            threshold=self.confidence_threshold,
        )

    def preferred_height(self) -> int:
        """Return the preferred widget height for the current channel count."""
        n_channels = (
            int(self.signals.shape[0])
            if self.signals is not None and self.signals.ndim == 2
            else max(1, len(self.channel_plan))
        )
        per_channel = int(
            round(
                DEFAULT_SIGNAL_REVIEW_PLOT_HEIGHT_PER_CHANNEL * self.spacing_multiplier
            )
        )
        return max(
            DEFAULT_SIGNAL_REVIEW_PLOT_MIN_HEIGHT,
            DEFAULT_SIGNAL_REVIEW_PLOT_BASE_HEIGHT + n_channels * per_channel,
        )

    def _voltage_scale_for(self, label: str) -> float:
        """Return the µV/div scale to apply for a given channel-plan slot."""
        modality = _slot_modality(label)
        if modality == "emg":
            return max(1e-6, float(self.emg_uv_per_div))
        if modality == "ecg":
            return SIGNAL_REVIEW_ECG_UV_PER_DIV
        return max(1e-6, float(self.eeg_eog_uv_per_div))

    def _ensure_filtered(self) -> np.ndarray | None:
        """Return the filtered full-recording signals, recomputing only when stale."""
        if self.signals is None or self.signals.ndim != 2 or self.sample_rate <= 0:
            self._filtered_signals = None
            return None
        spec_key = tuple(
            (modality, spec.low_cut, spec.high_cut, spec.notch_hz)
            for modality, spec in sorted(self.filter_specs.items())
        )
        cache_key = (id(self.signals), round(self.sample_rate, 6), spec_key)
        if self._filtered_signals is not None and self._filter_cache_key == cache_key:
            return self._filtered_signals
        self._filtered_signals = _filter_display_signals(
            self.signals,
            self.sample_rate,
            self.channel_plan,
            self.filter_specs,
        )
        self._filter_cache_key = cache_key
        return self._filtered_signals

    def resizeEvent(self, event):
        """Rerender using the current viewport width after widget resizes."""
        super().resizeEvent(event)
        if pg is None or self._plot_item is None:
            return
        self.render_page(
            page_start_sec=self.page_start_sec,
            page_duration_sec=self.page_duration_sec,
            selected_epoch=self.selected_epoch,
            epoch_sec=self.epoch_sec,
            confidences=self.confidences,
            manual_overrides=self.manual_overrides,
            threshold=self.confidence_threshold,
        )

    def set_display_data(
        self,
        signals: np.ndarray | None,
        sample_rate: float,
        channel_plan: list[dict[str, Any]] | None,
    ):
        """Attach signals for the active display preset."""
        self.signals = (
            None if signals is None else np.asarray(signals, dtype=np.float32)
        )
        self.sample_rate = float(sample_rate) if sample_rate else 1.0
        self.channel_plan = list(channel_plan or [])
        self._filtered_signals = None
        self._filter_cache_key = None
        if self.signals is not None and self.signals.ndim == 2 and self.sample_rate > 0:
            self._current_total_duration_sec = self.signals.shape[1] / self.sample_rate
        else:
            self._current_total_duration_sec = 0.0
        self.render_page(
            page_start_sec=self.page_start_sec,
            page_duration_sec=self.page_duration_sec,
            selected_epoch=self.selected_epoch,
            epoch_sec=self.epoch_sec,
            confidences=self.confidences,
            manual_overrides=self.manual_overrides,
            threshold=self.confidence_threshold,
        )
        self.updateGeometry()

    def render_page(
        self,
        *,
        page_start_sec: float,
        page_duration_sec: float,
        selected_epoch: int,
        epoch_sec: int,
        confidences: np.ndarray | None,
        manual_overrides: dict[int, int] | None,
        threshold: float,
    ):
        """Render a page-based reader view as a single stacked-offset montage."""
        if pg is None or self._plot_item is None:
            return

        self.page_start_sec = max(0.0, float(page_start_sec))
        self.page_duration_sec = max(1.0, float(page_duration_sec))
        self.selected_epoch = max(0, int(selected_epoch))
        self.epoch_sec = max(1, int(epoch_sec))
        self.confidences = confidences
        self.manual_overrides = _coerce_manual_score_overrides(manual_overrides)
        self.confidence_threshold = float(threshold)

        self._plot_item.clear()
        self._plot_item.addItem(self._cursor_line)
        self._cursor_line.setPos(0.0)

        filtered = self._ensure_filtered()
        if filtered is None:
            self._plot_item.getAxis("left").setTicks([[]])
            self._plot_item.setXRange(0.0, self.page_duration_sec, padding=0.0)
            return

        n_traces, n_samples_total = filtered.shape
        start_idx = max(0, int(round(self.page_start_sec * self.sample_rate)))
        end_idx = min(
            n_samples_total,
            int(
                round((self.page_start_sec + self.page_duration_sec) * self.sample_rate)
            ),
        )
        if end_idx <= start_idx:
            end_idx = min(n_samples_total, start_idx + int(self.sample_rate))
        window = filtered[:, start_idx:end_idx]
        self._page_end_sec = self.page_start_sec + (window.shape[1] / self.sample_rate)
        time_axis = np.arange(window.shape[1], dtype=np.float32) / self.sample_rate

        device_pixel_ratio = max(1.0, float(self._plot_widget.devicePixelRatioF()))
        viewport_width = max(
            200,
            int(round(self._plot_widget.viewport().width() * device_pixel_ratio)),
        )
        max_plot_points = max(800, viewport_width * 4)

        channel_gap = SIGNAL_REVIEW_CHANNEL_GAP_DIVS * self.spacing_multiplier
        tick_pairs: list[tuple[float, str]] = []
        max_offset = 0.0
        cal_pen = pg.mkPen(SIGNAL_REVIEW_CALIBRATION_COLOR, width=1.4)
        cal_x = self.page_duration_sec * 0.008
        cap = self.page_duration_sec * 0.004

        for trace_idx in range(n_traces):
            channel_row = (
                self.channel_plan[trace_idx]
                if trace_idx < len(self.channel_plan)
                else {
                    "label": f"Ch {trace_idx + 1}",
                    "status": "missing",
                    "detail": "Unavailable",
                    "present": False,
                }
            )
            offset = float((n_traces - 1 - trace_idx) * channel_gap)
            max_offset = max(max_offset, offset)

            present = bool(channel_row.get("present", False))
            trace_label = str(channel_row.get("label", f"Ch {trace_idx + 1}"))
            voltage_scale = self._voltage_scale_for(trace_label)
            signal = window[trace_idx] if trace_idx < len(window) else np.zeros(1)
            if signal.size == 0:
                signal = np.zeros(1, dtype=np.float32)
            baseline = float(np.median(signal))
            y_values = (
                signal.astype(np.float32, copy=False) - baseline
            ) / voltage_scale + offset
            color = (
                SIGNAL_REVIEW_TRACE_COLOR
                if present
                else SIGNAL_REVIEW_MISSING_TRACE_COLOR
            )
            plot_x, plot_y = _downsample_trace_for_display(
                time_axis, y_values, max_points=max_plot_points
            )
            self._plot_item.plot(plot_x, plot_y, pen=pg.mkPen(color, width=1.0))

            # Faint separator below this channel (except under the bottom trace).
            if trace_idx < n_traces - 1:
                self._plot_item.addItem(
                    pg.InfiniteLine(
                        pos=offset - channel_gap / 2.0,
                        angle=0,
                        pen=pg.mkPen(SIGNAL_REVIEW_GRID_COLOR, width=0.8),
                    )
                )

            # Per-channel amplitude calibration bar (one division tall).
            self._plot_item.addItem(
                pg.PlotDataItem(
                    [cal_x, cal_x], [offset - 0.5, offset + 0.5], pen=cal_pen
                )
            )
            self._plot_item.addItem(
                pg.PlotDataItem(
                    [cal_x - cap, cal_x + cap],
                    [offset + 0.5, offset + 0.5],
                    pen=cal_pen,
                )
            )
            self._plot_item.addItem(
                pg.PlotDataItem(
                    [cal_x - cap, cal_x + cap],
                    [offset - 0.5, offset - 0.5],
                    pen=cal_pen,
                )
            )
            # Absolute amplitude of the one-division calibration bar, so the reader
            # can read off scale at a glance (Profusion-style).
            cal_text = pg.TextItem(
                f"{voltage_scale:g} µV",
                color=SIGNAL_REVIEW_CALIBRATION_COLOR,
                anchor=(0.0, 0.5),
            )
            cal_text.setPos(cal_x + cap * 2.0, offset)
            self._plot_item.addItem(cal_text)

            display_label = str(channel_row.get("display_label", trace_label))
            tick_pairs.append((offset, f"{display_label}\n{voltage_scale:g} µV/div"))

        self._draw_epoch_overlays(max_offset, channel_gap)
        self._plot_item.getAxis("left").setTicks([tick_pairs])
        self._plot_item.getAxis("bottom").setLabel(
            f"Page Time (s)   Study Range: "
            f"{_format_elapsed_time_hms(int(self.page_start_sec))}"
            f" - {_format_elapsed_time_hms(int(self._page_end_sec))}",
            color=SIGNAL_REVIEW_AXIS_TEXT_COLOR,
        )
        self._plot_item.setYRange(-channel_gap, max_offset + channel_gap, padding=0.0)
        self._plot_item.setXRange(0.0, self.page_duration_sec, padding=0.0)

    def _draw_epoch_overlays(self, max_offset: float, channel_gap: float) -> None:
        """Add the active-epoch tint, 30 s boundaries, and low-conf/override flags."""
        if pg is None or self._plot_item is None:
            return
        # Active-epoch tint across the full field.
        selected_start = self.selected_epoch * self.epoch_sec
        active_left = max(0.0, selected_start - self.page_start_sec)
        active_right = min(
            self.page_duration_sec,
            (selected_start + self.epoch_sec) - self.page_start_sec,
        )
        if active_right > active_left:
            self._plot_item.addItem(
                pg.LinearRegionItem(
                    values=(active_left, active_right),
                    brush=pg.mkBrush(*SIGNAL_REVIEW_EPOCH_TINT),
                    pen=pg.mkPen("#4CAF50", width=1.0),
                    movable=False,
                )
            )

        low_x: list[float] = []
        override_x: list[float] = []
        low_level = max_offset + channel_gap * 0.55
        override_level = max_offset + channel_gap * 0.8
        visible_epoch_start = int(self.page_start_sec // self.epoch_sec)
        visible_epoch_end = int(np.ceil(self._page_end_sec / self.epoch_sec))
        for epoch_idx in range(visible_epoch_start, visible_epoch_end + 1):
            epoch_left = epoch_idx * self.epoch_sec - self.page_start_sec
            if 0.0 <= epoch_left <= self.page_duration_sec:
                is_selected = epoch_idx == self.selected_epoch
                self._plot_item.addItem(
                    pg.InfiniteLine(
                        pos=epoch_left,
                        angle=90,
                        pen=pg.mkPen(
                            SIGNAL_REVIEW_GRID_MAJOR_COLOR,
                            width=1.4 if is_selected else 0.8,
                            style=(
                                Qt.PenStyle.SolidLine
                                if is_selected
                                else Qt.PenStyle.DashLine
                            ),
                        ),
                    )
                )
            if epoch_idx < 0:
                continue
            epoch_center = epoch_left + self.epoch_sec / 2.0
            if not (0.0 <= epoch_center <= self.page_duration_sec):
                continue
            if (
                self.confidences is not None
                and epoch_idx < len(self.confidences)
                and float(self.confidences[epoch_idx]) < self.confidence_threshold
            ):
                low_x.append(epoch_center)
            if epoch_idx in self.manual_overrides:
                override_x.append(epoch_center)
        if low_x:
            self._plot_item.addItem(
                pg.ScatterPlotItem(
                    x=low_x,
                    y=[low_level] * len(low_x),
                    symbol="t",
                    size=10,
                    brush=pg.mkBrush("#E53935"),
                    pen=pg.mkPen("#E53935"),
                )
            )
        if override_x:
            self._plot_item.addItem(
                pg.ScatterPlotItem(
                    x=override_x,
                    y=[override_level] * len(override_x),
                    symbol="d",
                    size=9,
                    brush=pg.mkBrush("#42A5F5"),
                    pen=pg.mkPen("#42A5F5"),
                )
            )

    def _on_mouse_moved(self, event):
        """Update the crosshair and cursor labels from mouse motion."""
        if pg is None or self._plot_item is None or self._plot_widget is None:
            return
        if self.signals is None:
            return

        pos = event[0]
        if not self._plot_widget.sceneBoundingRect().contains(pos):
            return
        mouse_point = self._plot_item.vb.mapSceneToView(pos)
        x_pos = float(mouse_point.x())
        if x_pos < 0.0 or x_pos > self.page_duration_sec:
            return
        self._cursor_line.setPos(x_pos)
        absolute_sec = self.page_start_sec + x_pos
        self.cursor_changed.emit(
            _format_elapsed_time_hms_ms(x_pos),
            _format_elapsed_time_hms_ms(absolute_sec),
        )


class HypnogramNavigatorWidget(QWidget):
    """Compact whole-night hypnogram strip with a draggable page marker.

    Emits :attr:`page_requested` (page-start in seconds) when the user clicks the
    strip or drags the highlighted page region, providing Profusion-style
    whole-night context and one-click navigation below the signal lanes.
    """

    page_requested = Signal(float)

    _STAGE_Y: dict[int, float] = {0: 4.0, 1: 2.0, 2: 1.0, 3: 0.0, 4: 3.0}
    _STAGE_TICKS: list[tuple[float, str]] = [
        (4.0, "W"),
        (3.0, "REM"),
        (2.0, "N1"),
        (1.0, "N2"),
        (0.0, "N3"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_duration_sec: float = 0.0
        self.page_duration_sec: float = float(DEFAULT_SIGNAL_REVIEW_PAGE_SEC)
        self._plot_widget: Any = None
        self._plot_item: Any = None
        self._curve: Any = None
        self._region: Any = None
        self._markers: Any = None
        self._suppress_region: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if pg is None:
            self.setVisible(False)
            return

        self._plot_widget = pg.PlotWidget(background=SIGNAL_REVIEW_PANEL_COLOR)
        self._plot_widget.setFixedHeight(104)
        plot = self._plot_widget.getPlotItem()
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setMouseEnabled(x=False, y=False)
        plot.getViewBox().setBackgroundColor(SIGNAL_REVIEW_PANEL_COLOR)
        left_axis = plot.getAxis("left")
        left_axis.setTicks([self._STAGE_TICKS])
        left_axis.setTextPen(pg.mkPen(SIGNAL_REVIEW_AXIS_TEXT_COLOR))
        left_axis.setPen(pg.mkPen(SIGNAL_REVIEW_AXIS_LINE_COLOR))
        left_axis.setWidth(34)
        bottom_axis = plot.getAxis("bottom")
        bottom_axis.setTextPen(pg.mkPen(SIGNAL_REVIEW_AXIS_TEXT_COLOR))
        bottom_axis.setPen(pg.mkPen(SIGNAL_REVIEW_AXIS_LINE_COLOR))
        bottom_axis.setLabel(
            "Whole-night hypnogram — click or drag to navigate",
            color=SIGNAL_REVIEW_AXIS_TEXT_COLOR,
        )
        plot.setYRange(-0.4, 4.4, padding=0.0)
        self._curve = plot.plot(
            [], [], pen=pg.mkPen(SIGNAL_REVIEW_TRACE_ACCENT_COLOR, width=1.2)
        )
        self._region = pg.LinearRegionItem(
            values=(0.0, self.page_duration_sec),
            brush=pg.mkBrush(76, 175, 80, 60),
            pen=pg.mkPen("#4CAF50", width=1.2),
        )
        self._region.setZValue(10)
        self._region.sigRegionChangeFinished.connect(self._on_region_changed)
        plot.addItem(self._region)
        # Flagged/uncertain-epoch markers along the top of the strip.
        self._markers = pg.ScatterPlotItem(
            size=6,
            symbol="t",
            brush=pg.mkBrush(255, 152, 0, 200),
            pen=None,
        )
        self._markers.setZValue(9)
        plot.addItem(self._markers)
        self._plot_item = plot
        self._plot_widget.scene().sigMouseClicked.connect(self._on_clicked)
        layout.addWidget(self._plot_widget)

    def set_hypnogram(
        self,
        stages: np.ndarray | None,
        epoch_sec: int,
        total_duration_sec: float,
    ) -> None:
        """Draw the whole-night staged step function.

        Args:
            stages: Per-epoch stage indices (0-4), or ``None`` to clear.
            epoch_sec: Epoch length in seconds.
            total_duration_sec: Study duration in seconds (for the x-range).
        """
        if pg is None or self._plot_widget is None:
            return
        self.total_duration_sec = float(total_duration_sec)
        if stages is None or len(stages) == 0:
            self._curve.setData([], [])
            self._plot_item.setXRange(
                0.0, max(1.0, self.total_duration_sec), padding=0.0
            )
            return
        stages_arr = np.asarray(stages)
        n_epochs = len(stages_arr)
        x = (np.arange(n_epochs, dtype=np.float64) + 0.5) * float(epoch_sec)
        y = np.array(
            [self._STAGE_Y.get(int(s), np.nan) for s in stages_arr],
            dtype=np.float64,
        )
        self._curve.setData(x, y, connect="finite")
        span = max(self.total_duration_sec, n_epochs * float(epoch_sec))
        self._plot_item.setXRange(0.0, span, padding=0.0)

    def set_page(
        self,
        page_start_sec: float,
        page_duration_sec: float,
        total_duration_sec: float,
    ) -> None:
        """Move the highlighted page marker without emitting a navigation signal."""
        if pg is None or self._region is None:
            return
        self.page_duration_sec = float(page_duration_sec)
        self.total_duration_sec = float(total_duration_sec)
        self._suppress_region = True
        self._region.setRegion(
            (float(page_start_sec), float(page_start_sec) + float(page_duration_sec))
        )
        self._suppress_region = False

    def set_markers(self, marker_secs: np.ndarray | None) -> None:
        """Overlay flagged/uncertain-epoch markers along the top of the strip.

        Args:
            marker_secs: Epoch-center times (seconds) to mark, or ``None``/empty
                to clear the overlay.
        """
        if pg is None or self._markers is None:
            return
        if marker_secs is None or len(marker_secs) == 0:
            self._markers.setData([], [])
            return
        xs = np.asarray(marker_secs, dtype=np.float64)
        ys = np.full(xs.shape, 4.3, dtype=np.float64)
        self._markers.setData(xs, ys)

    def _on_region_changed(self) -> None:
        """Emit the requested page-start when the user drags the page marker."""
        if self._suppress_region or self._region is None:
            return
        start = float(self._region.getRegion()[0])
        self.page_requested.emit(max(0.0, start))

    def _on_clicked(self, event) -> None:
        """Center the page on a clicked position in the hypnogram strip."""
        if pg is None or self._plot_item is None:
            return
        pos = event.scenePos()
        if not self._plot_item.sceneBoundingRect().contains(pos):
            return
        x_pos = float(self._plot_item.vb.mapSceneToView(pos).x())
        self.page_requested.emit(max(0.0, x_pos - self.page_duration_sec / 2.0))


class SignalReviewWidget(QWidget):
    """Reader-first workspace for page-based EDF signal review and rescoring."""

    epoch_selected = Signal(int)
    manual_override_requested = Signal(int, int)
    clear_override_requested = Signal(int)
    clear_all_overrides_requested = Signal()

    TABLE_HEADERS = ["Epoch", "Time", "Model", "Final", "Conf", "Flags"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictions: np.ndarray | None = None
        self.final_predictions: np.ndarray | None = None
        self.probabilities: np.ndarray | None = None
        self.confidences: np.ndarray | None = None
        self.flag_scores: dict[str, np.ndarray] | None = None
        self.rank_by_uncertainty: bool = False
        self.keep_epoch_centered: bool = False
        self.manual_overrides: dict[int, int] = {}
        self.epoch_sec: int = 30
        self.confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD
        self.selected_epoch: int = 0
        self.records: list[dict[str, Any]] = []
        self.filtered_records: list[dict[str, Any]] = []
        self._signal_payload: dict[str, Any] | None = None
        self.page_duration_sec: int = DEFAULT_SIGNAL_REVIEW_PAGE_SEC
        self.display_preset_name: str = DEFAULT_SIGNAL_REVIEW_PRESET
        self.eeg_eog_uv_per_div: float = DEFAULT_SIGNAL_REVIEW_EEG_EOG_UV_PER_DIV
        self.emg_uv_per_div: float = DEFAULT_SIGNAL_REVIEW_EMG_UV_PER_DIV
        self.spacing_preset_name: str = DEFAULT_SIGNAL_REVIEW_SPACING
        # Display-filter state (EEG/EOG share one band; EMG is separate; notch global).
        self.eeg_low_cut: float | None = DEFAULT_DISPLAY_FILTERS["eeg"].low_cut
        self.eeg_high_cut: float | None = DEFAULT_DISPLAY_FILTERS["eeg"].high_cut
        self.emg_low_cut: float | None = DEFAULT_DISPLAY_FILTERS["emg"].low_cut
        self.emg_high_cut: float | None = DEFAULT_DISPLAY_FILTERS["emg"].high_cut
        self.notch_hz: float | None = DEFAULT_SIGNAL_REVIEW_NOTCH_HZ
        self.page_start_sec: float = 0.0
        self.queue_visible: bool = DEFAULT_SIGNAL_REVIEW_QUEUE_VISIBLE
        self._setup_ui()

    def _current_filter_specs(self) -> dict[str, DisplayFilterSpec]:
        """Build the per-modality filter specs from the current control state."""
        return {
            "eeg": DisplayFilterSpec(
                self.eeg_low_cut, self.eeg_high_cut, self.notch_hz
            ),
            "eog": DisplayFilterSpec(
                self.eeg_low_cut, self.eeg_high_cut, self.notch_hz
            ),
            "emg": DisplayFilterSpec(
                self.emg_low_cut, self.emg_high_cut, self.notch_hz
            ),
            "ecg": DisplayFilterSpec(0.3, 70.0, self.notch_hz),
        }

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Signal Review")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        description = QLabel(
            "Clinical-style page reader for reviewing PSG pages, navigating low-confidence epochs, "
            "and applying stage overrides."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #aaa; margin-bottom: 2px; font-size: 11px;")
        layout.addWidget(description)

        self.controls_frame = QFrame()
        controls_layout = QGridLayout(self.controls_frame)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setHorizontalSpacing(6)
        controls_layout.setVerticalSpacing(4)
        controls_font = QFont()
        controls_font.setPointSize(10)

        self.page_duration_combo = QComboBox()
        self.page_duration_combo.setFont(controls_font)
        for seconds in SIGNAL_REVIEW_PAGE_DURATIONS:
            self.page_duration_combo.addItem(f"{seconds} s", seconds)
        self.page_duration_combo.currentIndexChanged.connect(
            self._on_page_duration_changed
        )
        timebase_label = QLabel("Timebase:")
        timebase_label.setFont(controls_font)
        controls_layout.addWidget(timebase_label, 0, 0)
        controls_layout.addWidget(self.page_duration_combo, 0, 1)

        self.eeg_eog_scale_spin = QDoubleSpinBox()
        self.eeg_eog_scale_spin.setFont(controls_font)
        self.eeg_eog_scale_spin.setRange(
            SIGNAL_REVIEW_UV_PER_DIV_MIN, SIGNAL_REVIEW_UV_PER_DIV_MAX
        )
        self.eeg_eog_scale_spin.setSingleStep(SIGNAL_REVIEW_UV_PER_DIV_STEP)
        self.eeg_eog_scale_spin.setDecimals(0)
        self.eeg_eog_scale_spin.setSuffix(" µV/div")
        self.eeg_eog_scale_spin.valueChanged.connect(self._on_eeg_eog_scale_changed)
        eeg_eog_label = QLabel("EEG/EOG:")
        eeg_eog_label.setFont(controls_font)
        controls_layout.addWidget(eeg_eog_label, 0, 2)
        controls_layout.addWidget(self.eeg_eog_scale_spin, 0, 3)

        self.emg_scale_spin = QDoubleSpinBox()
        self.emg_scale_spin.setFont(controls_font)
        self.emg_scale_spin.setRange(
            SIGNAL_REVIEW_UV_PER_DIV_MIN, SIGNAL_REVIEW_UV_PER_DIV_MAX
        )
        self.emg_scale_spin.setSingleStep(SIGNAL_REVIEW_UV_PER_DIV_STEP)
        self.emg_scale_spin.setDecimals(0)
        self.emg_scale_spin.setSuffix(" µV/div")
        self.emg_scale_spin.valueChanged.connect(self._on_emg_scale_changed)
        emg_label = QLabel("EMG:")
        emg_label.setFont(controls_font)
        controls_layout.addWidget(emg_label, 0, 4)
        controls_layout.addWidget(self.emg_scale_spin, 0, 5)

        self.spacing_combo = QComboBox()
        self.spacing_combo.setFont(controls_font)
        for label_name in SIGNAL_REVIEW_SPACING_PRESETS:
            self.spacing_combo.addItem(label_name, label_name)
        self.spacing_combo.setToolTip(
            "Vertical gap between channels — smaller shows larger waves and lets "
            "big deflections overlap (cross over) neighbours."
        )
        self.spacing_combo.currentIndexChanged.connect(self._on_spacing_changed)
        spacing_label = QLabel("Spacing:")
        spacing_label.setFont(controls_font)
        controls_layout.addWidget(spacing_label, 0, 6)
        controls_layout.addWidget(self.spacing_combo, 0, 7)

        self.display_preset_combo = QComboBox()
        self.display_preset_combo.setFont(controls_font)
        self.display_preset_combo.currentIndexChanged.connect(
            self._on_display_preset_changed
        )
        montage_label = QLabel("Montage:")
        montage_label.setFont(controls_font)
        controls_layout.addWidget(montage_label, 0, 8)
        controls_layout.addWidget(self.display_preset_combo, 0, 9)

        self.fit_button = QPushButton("Fit")
        self.fit_button.setFont(controls_font)
        self.fit_button.setMinimumHeight(26)
        self.fit_button.clicked.connect(self._reset_reader_view)
        controls_layout.addWidget(self.fit_button, 0, 10)

        # --- Display filter controls (row 3) --------------------------------
        filter_label = QLabel("Filters:")
        filter_label.setFont(controls_font)
        controls_layout.addWidget(filter_label, 3, 0)

        self.eeg_low_cut_combo = self._build_filter_combo(
            SIGNAL_REVIEW_LOW_CUT_OPTIONS, self.eeg_low_cut, controls_font
        )
        self.eeg_low_cut_combo.setToolTip("EEG/EOG low-cut (high-pass) corner.")
        self.eeg_low_cut_combo.currentIndexChanged.connect(self._on_filter_changed)
        eeg_lf_label = QLabel("EEG LF:")
        eeg_lf_label.setFont(controls_font)
        controls_layout.addWidget(eeg_lf_label, 3, 1)
        controls_layout.addWidget(self.eeg_low_cut_combo, 3, 2)

        self.eeg_high_cut_combo = self._build_filter_combo(
            SIGNAL_REVIEW_HIGH_CUT_OPTIONS, self.eeg_high_cut, controls_font
        )
        self.eeg_high_cut_combo.setToolTip("EEG/EOG high-cut (low-pass) corner.")
        self.eeg_high_cut_combo.currentIndexChanged.connect(self._on_filter_changed)
        eeg_hf_label = QLabel("EEG HF:")
        eeg_hf_label.setFont(controls_font)
        controls_layout.addWidget(eeg_hf_label, 3, 3)
        controls_layout.addWidget(self.eeg_high_cut_combo, 3, 4)

        self.emg_low_cut_combo = self._build_filter_combo(
            SIGNAL_REVIEW_LOW_CUT_OPTIONS, self.emg_low_cut, controls_font
        )
        self.emg_low_cut_combo.setToolTip("EMG low-cut (high-pass) corner.")
        self.emg_low_cut_combo.currentIndexChanged.connect(self._on_filter_changed)
        emg_lf_label = QLabel("EMG LF:")
        emg_lf_label.setFont(controls_font)
        controls_layout.addWidget(emg_lf_label, 3, 5)
        controls_layout.addWidget(self.emg_low_cut_combo, 3, 6)

        self.emg_high_cut_combo = self._build_filter_combo(
            SIGNAL_REVIEW_HIGH_CUT_OPTIONS, self.emg_high_cut, controls_font
        )
        self.emg_high_cut_combo.setToolTip(
            "EMG high-cut (low-pass) corner. Skipped automatically when it "
            "exceeds the display Nyquist frequency."
        )
        self.emg_high_cut_combo.currentIndexChanged.connect(self._on_filter_changed)
        emg_hf_label = QLabel("EMG HF:")
        emg_hf_label.setFont(controls_font)
        controls_layout.addWidget(emg_hf_label, 3, 7)
        controls_layout.addWidget(self.emg_high_cut_combo, 3, 8)

        self.notch_combo = QComboBox()
        self.notch_combo.setFont(controls_font)
        for value in SIGNAL_REVIEW_NOTCH_OPTIONS:
            self.notch_combo.addItem("Off" if value is None else f"{value:g} Hz", value)
        self._set_combo_by_data(self.notch_combo, self.notch_hz)
        self.notch_combo.setToolTip("Mains line-noise notch filter.")
        self.notch_combo.currentIndexChanged.connect(self._on_filter_changed)
        notch_label = QLabel("Notch:")
        notch_label.setFont(controls_font)
        controls_layout.addWidget(notch_label, 3, 9)
        controls_layout.addWidget(self.notch_combo, 3, 10)

        self.prev_page_button = QPushButton("◀ Page")
        self.prev_page_button.setFont(controls_font)
        self.prev_page_button.setMinimumHeight(26)
        self.prev_page_button.clicked.connect(lambda: self.step_page(-1))
        controls_layout.addWidget(self.prev_page_button, 1, 0)

        self.next_page_button = QPushButton("Page ▶")
        self.next_page_button.setFont(controls_font)
        self.next_page_button.setMinimumHeight(26)
        self.next_page_button.clicked.connect(lambda: self.step_page(1))
        controls_layout.addWidget(self.next_page_button, 1, 1)

        self.prev_epoch_button = QPushButton("◀ Epoch")
        self.prev_epoch_button.setFont(controls_font)
        self.prev_epoch_button.setMinimumHeight(26)
        self.prev_epoch_button.clicked.connect(lambda: self.step_epoch(-1))
        controls_layout.addWidget(self.prev_epoch_button, 1, 2)

        self.next_epoch_button = QPushButton("Epoch ▶")
        self.next_epoch_button.setFont(controls_font)
        self.next_epoch_button.setMinimumHeight(26)
        self.next_epoch_button.clicked.connect(lambda: self.step_epoch(1))
        controls_layout.addWidget(self.next_epoch_button, 1, 3)

        self.jump_epoch_spin = QSpinBox()
        self.jump_epoch_spin.setFont(controls_font)
        self.jump_epoch_spin.setMinimum(1)
        self.jump_epoch_spin.setMaximum(1)
        self.jump_epoch_spin.valueChanged.connect(self._on_jump_epoch_changed)
        jump_epoch_label = QLabel("Jump Epoch:")
        jump_epoch_label.setFont(controls_font)
        controls_layout.addWidget(jump_epoch_label, 1, 4)
        controls_layout.addWidget(self.jump_epoch_spin, 1, 5)

        self.follow_queue_checkbox = QCheckBox("Follow Queue")
        self.follow_queue_checkbox.setFont(controls_font)
        self.follow_queue_checkbox.setChecked(True)
        controls_layout.addWidget(self.follow_queue_checkbox, 1, 6)

        self.show_queue_checkbox = QCheckBox("Show Queue")
        self.show_queue_checkbox.setFont(controls_font)
        self.show_queue_checkbox.setChecked(True)
        self.show_queue_checkbox.toggled.connect(self._toggle_queue_visibility)
        controls_layout.addWidget(self.show_queue_checkbox, 1, 7)

        self.keep_centered_checkbox = QCheckBox("Keep Centered")
        self.keep_centered_checkbox.setFont(controls_font)
        self.keep_centered_checkbox.setToolTip(
            "Recenter the page on the active epoch when stepping with the arrow "
            "keys (instead of only scrolling it into view)."
        )
        self.keep_centered_checkbox.toggled.connect(self._on_keep_centered_toggled)
        controls_layout.addWidget(self.keep_centered_checkbox, 1, 8)

        self.page_status_label = QLabel("Page: --")
        self.page_status_label.setFont(controls_font)
        self.page_status_label.setStyleSheet("color: #aaa;")
        controls_layout.addWidget(self.page_status_label, 2, 0, 1, 4)

        self.cursor_status_label = QLabel("Cursor: --")
        self.cursor_status_label.setFont(controls_font)
        self.cursor_status_label.setStyleSheet("color: #aaa;")
        controls_layout.addWidget(self.cursor_status_label, 2, 4, 1, 2)

        self.active_epoch_label = QLabel("Active: --")
        self.active_epoch_label.setFont(controls_font)
        self.active_epoch_label.setStyleSheet("color: #4FC3F7; font-weight: bold;")
        controls_layout.addWidget(self.active_epoch_label, 2, 6, 1, 3)
        layout.addWidget(self.controls_frame)

        self.reader_splitter = QSplitter(Qt.Orientation.Vertical)
        self.reader_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.reader_splitter, 1)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        self.reader_splitter.addWidget(main_splitter)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.signal_status_label = QLabel("Signals not loaded.")
        self.signal_status_label.setWordWrap(True)
        self.signal_status_label.setStyleSheet("color: #888; font-size: 11px;")
        center_layout.addWidget(self.signal_status_label)

        self.plot_scroll_area = QScrollArea()
        self.plot_scroll_area.setWidgetResizable(True)
        self.plot_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.plot_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.plot_scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.plot_widget = SignalReviewPlotWidget()
        self.plot_widget.setMinimumHeight(DEFAULT_SIGNAL_REVIEW_PLOT_MIN_HEIGHT)
        self.plot_widget.cursor_changed.connect(self._on_cursor_changed)
        self.plot_scroll_area.setWidget(self.plot_widget)
        center_layout.addWidget(self.plot_scroll_area, 1)

        self.navigator = HypnogramNavigatorWidget()
        self.navigator.page_requested.connect(self._on_navigator_page_requested)
        center_layout.addWidget(self.navigator)
        main_splitter.addWidget(center_panel)

        side_panel = QWidget()
        # Allow the splitter to shrink the side panel to 0 on small screens, but
        # keep a generous default width when there's space.
        side_panel.setMinimumWidth(0)
        side_panel.setMaximumWidth(16777215)
        side_layout = QVBoxLayout(side_panel)

        self.queue_summary_label = QLabel("No epochs loaded.")
        self.queue_summary_label.setWordWrap(True)
        self.queue_summary_label.setStyleSheet("color: #aaa;")
        side_layout.addWidget(self.queue_summary_label)

        flagged_nav_layout = QHBoxLayout()
        self.prev_flagged_button = QPushButton("◀ Prev Flagged")
        self.prev_flagged_button.clicked.connect(self._select_previous_filtered)
        flagged_nav_layout.addWidget(self.prev_flagged_button)
        self.next_flagged_button = QPushButton("Next Flagged ▶")
        self.next_flagged_button.clicked.connect(self._select_next_filtered)
        flagged_nav_layout.addWidget(self.next_flagged_button)
        side_layout.addLayout(flagged_nav_layout)

        metadata_group = QGroupBox("Current Epoch")
        metadata_layout = QFormLayout(metadata_group)
        self.epoch_header_label = QLabel("—")
        self.model_stage_label = QLabel("—")
        self.final_stage_label = QLabel("—")
        self.override_badge_label = QLabel("—")
        self.confidence_label = QLabel("—")
        self.channel_plan_label = QLabel("—")
        self.channel_plan_label.setWordWrap(True)
        metadata_layout.addRow("Epoch:", self.epoch_header_label)
        metadata_layout.addRow("Model Stage:", self.model_stage_label)
        metadata_layout.addRow("Final Stage:", self.final_stage_label)
        metadata_layout.addRow("Override:", self.override_badge_label)
        metadata_layout.addRow("Confidence:", self.confidence_label)
        metadata_layout.addRow("Preset:", self.channel_plan_label)
        side_layout.addWidget(metadata_group)

        prob_group = QGroupBox("Class Probabilities")
        prob_layout = QVBoxLayout(prob_group)
        self.prob_table = QTableWidget(5, 2)
        self.prob_table.setHorizontalHeaderLabels(["Stage", "Probability"])
        self.prob_table.verticalHeader().setVisible(False)
        self.prob_table.horizontalHeader().setStretchLastSection(True)
        self.prob_table.setMaximumHeight(180)
        prob_layout.addWidget(self.prob_table)
        side_layout.addWidget(prob_group)

        score_group = QGroupBox("Rescore")
        score_layout = QVBoxLayout(score_group)
        rescore_font = QFont()
        rescore_font.setPointSize(10)
        stage_layout = QHBoxLayout()
        stage_layout.setSpacing(4)
        self.stage_buttons: dict[int, QPushButton] = {}
        for stage_idx, stage_name in enumerate(STAGE_LABELS):
            button = QPushButton(stage_name)
            button.setFont(rescore_font)
            button.setMinimumHeight(28)
            button.setStyleSheet("padding: 3px 6px;")
            button.clicked.connect(
                lambda checked=False, idx=stage_idx: self._request_manual_override(idx)
            )
            self.stage_buttons[stage_idx] = button
            stage_layout.addWidget(button)
        score_layout.addLayout(stage_layout)

        clear_layout = QHBoxLayout()
        clear_layout.setSpacing(4)
        self.clear_override_button = QPushButton("Clear Override")
        self.clear_override_button.setFont(rescore_font)
        self.clear_override_button.setMinimumHeight(28)
        self.clear_override_button.setStyleSheet("padding: 3px 6px;")
        self.clear_override_button.clicked.connect(self._request_clear_override)
        clear_layout.addWidget(self.clear_override_button)
        self.clear_all_overrides_button = QPushButton("Clear All Overrides")
        self.clear_all_overrides_button.setFont(rescore_font)
        self.clear_all_overrides_button.setMinimumHeight(28)
        self.clear_all_overrides_button.setStyleSheet("padding: 3px 6px;")
        self.clear_all_overrides_button.clicked.connect(
            self.clear_all_overrides_requested.emit
        )
        clear_layout.addWidget(self.clear_all_overrides_button)
        score_layout.addLayout(clear_layout)
        side_layout.addWidget(score_group)
        side_layout.addStretch()
        main_splitter.addWidget(side_panel)
        main_splitter.setStretchFactor(0, 5)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([1440, 280])

        self.queue_group = QGroupBox("Review Queue")
        queue_layout = QVBoxLayout(self.queue_group)
        filter_layout = QHBoxLayout()
        self.show_all_checkbox = QCheckBox("Show all epochs")
        self.show_all_checkbox.toggled.connect(self._refresh_queue)
        filter_layout.addWidget(self.show_all_checkbox)
        self.show_overridden_only_checkbox = QCheckBox("Show overridden only")
        self.show_overridden_only_checkbox.toggled.connect(self._refresh_queue)
        filter_layout.addWidget(self.show_overridden_only_checkbox)
        self.rank_by_uncertainty_checkbox = QCheckBox("Rank by uncertainty")
        self.rank_by_uncertainty_checkbox.setToolTip(
            "Order the queue by per-epoch uncertainty (most-uncertain first). "
            "Uses backend flag scores when available, otherwise confidence."
        )
        self.rank_by_uncertainty_checkbox.toggled.connect(
            self._on_rank_by_uncertainty_toggled
        )
        filter_layout.addWidget(self.rank_by_uncertainty_checkbox)
        filter_layout.addStretch()
        queue_layout.addLayout(filter_layout)

        self.queue_table = QTableWidget(0, len(self.TABLE_HEADERS))
        self.queue_table.setHorizontalHeaderLabels(self.TABLE_HEADERS)
        self.queue_table.setSelectionBehavior(
            self.queue_table.SelectionBehavior.SelectRows
        )
        self.queue_table.setSelectionMode(
            self.queue_table.SelectionMode.SingleSelection
        )
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        queue_layout.addWidget(self.queue_table, 1)
        self.reader_splitter.addWidget(self.queue_group)
        self.reader_splitter.setStretchFactor(0, 8)
        self.reader_splitter.setStretchFactor(1, 1)
        self.reader_splitter.setSizes([1180, 120])

        # Initialize reader controls.
        self._set_combo_by_data(self.page_duration_combo, self.page_duration_sec)
        self.eeg_eog_scale_spin.setValue(self.eeg_eog_uv_per_div)
        self.emg_scale_spin.setValue(self.emg_uv_per_div)
        self._set_combo_by_data(self.spacing_combo, self.spacing_preset_name)
        self.show_queue_checkbox.setChecked(self.queue_visible)
        self._toggle_queue_visibility(self.queue_visible)

    def _build_filter_combo(
        self,
        options: tuple[float | None, ...],
        current: float | None,
        font: QFont,
    ) -> QComboBox:
        """Build a low-cut/high-cut filter combo from a list of Hz options."""
        combo = QComboBox()
        combo.setFont(font)
        for value in options:
            combo.addItem("Off" if value is None else f"{value:g} Hz", value)
        self._set_combo_by_data(combo, current)
        return combo

    def _on_filter_changed(self):
        """Rebuild filter specs from the controls and push them to the plot."""
        self.eeg_low_cut = cast("float | None", self.eeg_low_cut_combo.currentData())
        self.eeg_high_cut = cast("float | None", self.eeg_high_cut_combo.currentData())
        self.emg_low_cut = cast("float | None", self.emg_low_cut_combo.currentData())
        self.emg_high_cut = cast("float | None", self.emg_high_cut_combo.currentData())
        self.notch_hz = cast("float | None", self.notch_combo.currentData())
        self.plot_widget.set_filter_specs(self._current_filter_specs())
        self._refresh_detail()

    def _set_combo_by_data(self, combo: QComboBox, value: Any):
        """Select a combo box entry by its item data when available."""
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
            return
        index = combo.findText(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _reset_reader_view(self):
        """Reset timebase, sensitivity, filters, and lanes to Profusion defaults."""
        self.page_duration_sec = DEFAULT_SIGNAL_REVIEW_PAGE_SEC
        self.eeg_eog_uv_per_div = DEFAULT_SIGNAL_REVIEW_EEG_EOG_UV_PER_DIV
        self.emg_uv_per_div = DEFAULT_SIGNAL_REVIEW_EMG_UV_PER_DIV
        self.spacing_preset_name = DEFAULT_SIGNAL_REVIEW_SPACING
        self.eeg_low_cut = DEFAULT_DISPLAY_FILTERS["eeg"].low_cut
        self.eeg_high_cut = DEFAULT_DISPLAY_FILTERS["eeg"].high_cut
        self.emg_low_cut = DEFAULT_DISPLAY_FILTERS["emg"].low_cut
        self.emg_high_cut = DEFAULT_DISPLAY_FILTERS["emg"].high_cut
        self.notch_hz = DEFAULT_SIGNAL_REVIEW_NOTCH_HZ
        self._set_combo_by_data(self.page_duration_combo, self.page_duration_sec)
        self.eeg_eog_scale_spin.setValue(self.eeg_eog_uv_per_div)
        self.emg_scale_spin.setValue(self.emg_uv_per_div)
        self._set_combo_by_data(self.spacing_combo, self.spacing_preset_name)
        for combo, value in (
            (self.eeg_low_cut_combo, self.eeg_low_cut),
            (self.eeg_high_cut_combo, self.eeg_high_cut),
            (self.emg_low_cut_combo, self.emg_low_cut),
            (self.emg_high_cut_combo, self.emg_high_cut),
            (self.notch_combo, self.notch_hz),
        ):
            combo.blockSignals(True)
            self._set_combo_by_data(combo, value)
            combo.blockSignals(False)
        self.plot_widget.set_filter_specs(self._current_filter_specs())
        self._center_selected_epoch()
        self._refresh_detail()

    def get_reader_preferences(self) -> dict[str, Any]:
        """Return stable reader preferences for project persistence."""
        return {
            "page_duration_sec": int(self.page_duration_sec),
            "eeg_eog_uv_per_div": float(self.eeg_eog_uv_per_div),
            "emg_uv_per_div": float(self.emg_uv_per_div),
            "display_preset": str(self.display_preset_name),
            "queue_visible": bool(self.queue_visible),
            "spacing_preset": str(self.spacing_preset_name),
            "keep_epoch_centered": bool(self.keep_epoch_centered),
        }

    def set_reader_preferences(self, preferences: dict[str, Any] | None):
        """Restore stable reader preferences from a project file."""
        if not isinstance(preferences, dict):
            return
        page_duration_sec = preferences.get("page_duration_sec")
        if (
            isinstance(page_duration_sec, (int, float))
            and int(page_duration_sec) in SIGNAL_REVIEW_PAGE_DURATIONS
        ):
            self.page_duration_sec = int(page_duration_sec)

        eeg_eog_scale = preferences.get("eeg_eog_uv_per_div")
        emg_scale = preferences.get("emg_uv_per_div")
        legacy_voltage_scale = preferences.get("voltage_scale_uv_per_div")
        legacy_gain_preset = preferences.get("gain_preset")
        legacy_value: float | None = None
        if isinstance(legacy_voltage_scale, (int, float)):
            legacy_value = float(legacy_voltage_scale)
        elif isinstance(legacy_gain_preset, str):
            legacy_value = LEGACY_SIGNAL_REVIEW_GAIN_TO_UV_PER_DIV.get(
                legacy_gain_preset
            )

        if isinstance(eeg_eog_scale, (int, float)):
            self.eeg_eog_uv_per_div = float(eeg_eog_scale)
        elif legacy_value is not None:
            self.eeg_eog_uv_per_div = legacy_value
        if isinstance(emg_scale, (int, float)):
            self.emg_uv_per_div = float(emg_scale)
        elif legacy_value is not None:
            self.emg_uv_per_div = legacy_value

        spacing_preset = preferences.get("spacing_preset")
        if (
            isinstance(spacing_preset, str)
            and spacing_preset in SIGNAL_REVIEW_SPACING_PRESETS
        ):
            self.spacing_preset_name = spacing_preset
        display_preset = preferences.get("display_preset")
        if isinstance(display_preset, str):
            self.display_preset_name = display_preset
        queue_visible = preferences.get("queue_visible")
        if isinstance(queue_visible, bool):
            self.queue_visible = queue_visible
        keep_centered = preferences.get("keep_epoch_centered")
        if isinstance(keep_centered, bool):
            self.keep_epoch_centered = keep_centered

        self._set_combo_by_data(self.page_duration_combo, self.page_duration_sec)
        self.eeg_eog_scale_spin.setValue(self.eeg_eog_uv_per_div)
        self.emg_scale_spin.setValue(self.emg_uv_per_div)
        self._set_combo_by_data(self.spacing_combo, self.spacing_preset_name)
        if self.display_preset_combo.count() > 0:
            self._set_combo_by_data(self.display_preset_combo, self.display_preset_name)
        self.show_queue_checkbox.setChecked(self.queue_visible)
        self._toggle_queue_visibility(self.queue_visible)
        self.keep_centered_checkbox.blockSignals(True)
        self.keep_centered_checkbox.setChecked(self.keep_epoch_centered)
        self.keep_centered_checkbox.blockSignals(False)

    def set_confidence_threshold(self, threshold: float):
        """Update the active threshold used by the review queue and timeline markers."""
        self.confidence_threshold = float(threshold)
        self._refresh_queue()
        self._refresh_detail()

    def update_review(
        self,
        predictions: np.ndarray | None,
        final_predictions: np.ndarray | None,
        probabilities: np.ndarray | None,
        confidences: np.ndarray | None,
        manual_overrides: dict[int, int] | None,
        epoch_sec: int = 30,
        *,
        flag_scores: dict[str, np.ndarray] | None = None,
    ):
        """Update the full review state."""
        self.predictions = predictions
        self.final_predictions = final_predictions
        self.probabilities = probabilities
        self.confidences = _resolve_confidences(probabilities, confidences)
        self.flag_scores = flag_scores
        self.manual_overrides = _coerce_manual_score_overrides(
            manual_overrides,
            n_epochs=(len(predictions) if predictions is not None else None),
        )
        self.epoch_sec = epoch_sec
        self.records = (
            _build_epoch_confidence_records(
                predictions,
                final_predictions=final_predictions,
                manual_overrides=self.manual_overrides,
                epoch_sec=epoch_sec,
                probabilities=probabilities,
                confidences=self.confidences,
                threshold=self.confidence_threshold,
                flag_scores=flag_scores,
            )
            if predictions is not None and len(predictions) > 0
            else []
        )
        if self.records:
            self.jump_epoch_spin.setMaximum(len(self.records))
            self.selected_epoch = max(
                0, min(self.selected_epoch, len(self.records) - 1)
            )
        else:
            self.jump_epoch_spin.setMaximum(1)
            self.selected_epoch = 0
        self._update_navigator_hypnogram()
        self._refresh_queue()
        self._refresh_detail()

    def _update_navigator_hypnogram(self):
        """Refresh the whole-night hypnogram strip from the current predictions."""
        stages = (
            self.final_predictions
            if self.final_predictions is not None
            else self.predictions
        )
        n_epochs = len(self.records)
        self.navigator.set_hypnogram(
            stages,
            self.epoch_sec,
            float(n_epochs * self.epoch_sec),
        )
        # Mark flagged / overridden epochs on the whole-night strip.
        marker_secs = [
            float(rec["time_sec"]) + self.epoch_sec / 2.0
            for rec in self.records
            if rec["low_confidence"] or rec["manual_override"]
        ]
        self.navigator.set_markers(
            np.asarray(marker_secs, dtype=np.float64) if marker_secs else None
        )

    def set_signal_payload(self, payload: dict[str, Any] | None):
        """Attach a cached study-level review payload and populate display presets."""
        self._signal_payload = payload
        self.display_preset_combo.blockSignals(True)
        self.display_preset_combo.clear()
        if payload is not None:
            for preset_name in payload.get("display_presets", {}).keys():
                self.display_preset_combo.addItem(preset_name, preset_name)
        self.display_preset_combo.blockSignals(False)

        if payload is None:
            self.signal_status_label.setText("Signals not loaded.")
            self.plot_widget.set_display_data(None, 1.0, None)
            self._refresh_detail()
            return

        if self.display_preset_combo.findData(self.display_preset_name) < 0:
            self.display_preset_name = DEFAULT_SIGNAL_REVIEW_PRESET
        self._set_combo_by_data(self.display_preset_combo, self.display_preset_name)
        sample_rate = float(payload.get("sample_rate", 1.0))
        n_samples = int(payload.get("n_samples", 0))
        duration_sec = n_samples / sample_rate if sample_rate > 0 else 0.0
        self.signal_status_label.setText(
            f"Study loaded at {sample_rate:.2f} Hz "
            f"({duration_sec / 60.0:.1f} minutes)."
        )
        self._apply_display_preset()
        self._refresh_detail()

    def set_signal_loading_message(self, message: str, *, is_error: bool = False):
        """Update the signal loading status line."""
        color = "#E53935" if is_error else "#888"
        self.signal_status_label.setStyleSheet(f"color: {color};")
        self.signal_status_label.setText(message)

    def select_epoch(self, epoch_idx: int, *, center_page: bool = True):
        """Select an epoch and refresh all review surfaces."""
        if not self.records:
            self.selected_epoch = max(0, int(epoch_idx))
            return
        epoch_idx = max(0, min(int(epoch_idx), len(self.records) - 1))
        self.selected_epoch = epoch_idx
        self.jump_epoch_spin.blockSignals(True)
        self.jump_epoch_spin.setValue(epoch_idx + 1)
        self.jump_epoch_spin.blockSignals(False)
        self._ensure_epoch_visible(epoch_idx, center=center_page)
        self._select_table_row_for_epoch(epoch_idx)
        self._refresh_detail()

    def _on_keep_centered_toggled(self, checked: bool) -> None:
        """Toggle whether arrow-key stepping recenters the page on the epoch."""
        self.keep_epoch_centered = bool(checked)

    def step_epoch(self, delta: int):
        """Move the active epoch, recentering only if 'Keep Centered' is on."""
        if not self.records:
            return
        self.select_epoch(
            self.selected_epoch + int(delta),
            center_page=self.keep_epoch_centered,
        )
        self.epoch_selected.emit(self.selected_epoch)

    def step_page(self, delta: int):
        """Move the current reader page by one full page."""
        if not self.records:
            return
        page_shift = float(self.page_duration_sec * int(delta))
        self.page_start_sec = self._clamp_page_start(self.page_start_sec + page_shift)
        self._refresh_detail()

    def jump_to_edge(self, which: str):
        """Jump to the start or end of the study."""
        if not self.records:
            return
        if which == "start":
            self.select_epoch(0, center_page=True)
            self.epoch_selected.emit(self.selected_epoch)
            return
        self.select_epoch(len(self.records) - 1, center_page=True)
        self.epoch_selected.emit(self.selected_epoch)

    def clear_selected_override(self):
        """Request clearing the override for the active epoch."""
        self._request_clear_override()

    def _toggle_queue_visibility(self, visible: bool):
        """Show or hide the bottom review queue strip."""
        self.queue_visible = bool(visible)
        self.queue_group.setVisible(self.queue_visible)
        if self.queue_visible:
            self.reader_splitter.setSizes([1180, 120])
        else:
            self.reader_splitter.setSizes([1300, 0])
        self._sync_plot_scroll_height()

    def _sync_plot_scroll_height(self):
        """Resize the plot widget so larger channel stacks can scroll vertically."""
        preferred_height = self.plot_widget.preferred_height()
        self.plot_widget.setMinimumHeight(preferred_height)
        self.plot_widget.resize(
            self.plot_scroll_area.viewport().width(), preferred_height
        )

    def _on_page_duration_changed(self):
        """Apply a new page duration and preserve focus on the active epoch."""
        selected = self.page_duration_combo.currentData()
        if selected is None:
            return
        self.page_duration_sec = int(selected)
        self._center_selected_epoch()
        self._refresh_detail()

    def _on_eeg_eog_scale_changed(self, value: float):
        """Apply a new EEG/EOG µV/div sensitivity."""
        self.eeg_eog_uv_per_div = float(value)
        self._refresh_detail()

    def _on_emg_scale_changed(self, value: float):
        """Apply a new EMG µV/div sensitivity."""
        self.emg_uv_per_div = float(value)
        self._refresh_detail()

    def _on_spacing_changed(self):
        """Apply a new lane-height preset."""
        selected = self.spacing_combo.currentData()
        if selected is None:
            return
        self.spacing_preset_name = str(selected)
        self.plot_widget.spacing_multiplier = SIGNAL_REVIEW_SPACING_PRESETS[
            self.spacing_preset_name
        ]
        self._sync_plot_scroll_height()
        self._refresh_detail()

    def _on_navigator_page_requested(self, start_sec: float):
        """Navigate to a page selected from the whole-night hypnogram strip."""
        if not self.records:
            return
        self.page_start_sec = self._clamp_page_start(float(start_sec))
        center_epoch = int(
            (self.page_start_sec + self.page_duration_sec / 2.0) // self.epoch_sec
        )
        center_epoch = max(0, min(center_epoch, len(self.records) - 1))
        self.selected_epoch = center_epoch
        self.jump_epoch_spin.blockSignals(True)
        self.jump_epoch_spin.setValue(center_epoch + 1)
        self.jump_epoch_spin.blockSignals(False)
        self._select_table_row_for_epoch(center_epoch)
        self._refresh_detail()
        self.epoch_selected.emit(self.selected_epoch)

    def _on_display_preset_changed(self):
        """Switch display presets without affecting inference or rescoring state."""
        selected = self.display_preset_combo.currentData()
        if selected is None:
            return
        self.display_preset_name = str(selected)
        self._apply_display_preset()
        self._refresh_detail()

    def _on_jump_epoch_changed(self, value: int):
        """Jump directly to an epoch from the reader control bar."""
        if not self.records:
            return
        epoch_idx = max(0, min(int(value) - 1, len(self.records) - 1))
        self.select_epoch(epoch_idx, center_page=True)
        self.epoch_selected.emit(self.selected_epoch)

    def _on_cursor_changed(self, page_time: str, absolute_time: str):
        """Refresh the cursor readout from the plot crosshair."""
        self.cursor_status_label.setText(
            f"Cursor: {page_time} page / {absolute_time} study"
        )

    def _apply_display_preset(self):
        """Apply the current display preset from the cached payload."""
        if self._signal_payload is None:
            self.plot_widget.set_display_data(None, 1.0, None)
            self._sync_plot_scroll_height()
            return
        presets = cast(dict[str, Any], self._signal_payload.get("display_presets", {}))
        preset = presets.get(self.display_preset_name)
        if preset is None and presets:
            self.display_preset_name = next(iter(presets.keys()))
            self._set_combo_by_data(self.display_preset_combo, self.display_preset_name)
            preset = presets.get(self.display_preset_name)
        if preset is None:
            self.plot_widget.set_display_data(None, 1.0, None)
            self._sync_plot_scroll_height()
            return
        self.plot_widget.eeg_eog_uv_per_div = self.eeg_eog_uv_per_div
        self.plot_widget.emg_uv_per_div = self.emg_uv_per_div
        self.plot_widget.spacing_multiplier = SIGNAL_REVIEW_SPACING_PRESETS[
            self.spacing_preset_name
        ]
        self.plot_widget.filter_specs = self._current_filter_specs()
        self.plot_widget.set_display_data(
            cast(np.ndarray | None, preset.get("signals")),
            float(self._signal_payload.get("sample_rate", 1.0)),
            cast(list[dict[str, Any]] | None, preset.get("channel_plan")),
        )
        self._sync_plot_scroll_height()

    def _total_duration_sec(self) -> float:
        """Return total study duration in seconds for the active display payload."""
        if self._signal_payload is None:
            return 0.0
        sample_rate = float(self._signal_payload.get("sample_rate", 0.0))
        n_samples = int(self._signal_payload.get("n_samples", 0))
        return (n_samples / sample_rate) if sample_rate > 0 else 0.0

    def _clamp_page_start(self, start_sec: float) -> float:
        """Clamp a page start time to the valid study range."""
        total_duration = self._total_duration_sec()
        if total_duration <= self.page_duration_sec:
            return 0.0
        return max(0.0, min(float(start_sec), total_duration - self.page_duration_sec))

    def _center_selected_epoch(self):
        """Center the active epoch in the current page when possible."""
        epoch_center = self.selected_epoch * self.epoch_sec + self.epoch_sec / 2.0
        self.page_start_sec = self._clamp_page_start(
            epoch_center - self.page_duration_sec / 2.0
        )

    def _ensure_epoch_visible(self, epoch_idx: int, *, center: bool):
        """Ensure an epoch is visible inside the current reader page."""
        if center or self.page_duration_sec <= self.epoch_sec:
            self.selected_epoch = epoch_idx
            self._center_selected_epoch()
            return
        epoch_start = epoch_idx * self.epoch_sec
        epoch_end = epoch_start + self.epoch_sec
        if epoch_start < self.page_start_sec:
            self.page_start_sec = self._clamp_page_start(epoch_start)
        elif epoch_end > self.page_start_sec + self.page_duration_sec:
            self.page_start_sec = self._clamp_page_start(
                epoch_end - self.page_duration_sec
            )

    def _on_rank_by_uncertainty_toggled(self, checked: bool) -> None:
        """Toggle uncertainty-ranked queue ordering and rebuild the queue."""
        self.rank_by_uncertainty = bool(checked)
        self._refresh_queue()

    def _refresh_queue(self):
        """Rebuild the review queue strip."""
        if not self.records:
            self.filtered_records = []
            self.queue_table.setRowCount(0)
            self.queue_summary_label.setText("No epochs loaded.")
            return

        if self.show_overridden_only_checkbox.isChecked():
            filtered = [record for record in self.records if record["manual_override"]]
        elif self.show_all_checkbox.isChecked():
            filtered = list(self.records)
        else:
            filtered = [
                record
                for record in self.records
                if record["low_confidence"] or record["manual_override"]
            ]

        if self.rank_by_uncertainty:
            # Most-uncertain first; epochs without a score sort to the bottom.
            filtered = sorted(
                filtered,
                key=lambda rec: (
                    rec["uncertainty_score"]
                    if rec["uncertainty_score"] is not None
                    else float("-inf")
                ),
                reverse=True,
            )

        self.filtered_records = filtered
        low_count = sum(1 for record in self.records if record["low_confidence"])
        override_count = sum(1 for record in self.records if record["manual_override"])
        self.queue_summary_label.setText(
            f"{len(filtered)} queue rows shown. {low_count} low-confidence, {override_count} overridden."
        )

        self.queue_table.setRowCount(len(filtered))
        for row, record in enumerate(filtered):
            flags = []
            if record["low_confidence"]:
                flags.append("LOW")
            if record["manual_override"]:
                flags.append("OVERRIDE")
            row_values = [
                str(record["epoch"]),
                str(record["time_hms"]),
                str(record["model_stage"]),
                str(record["final_stage"]),
                (
                    f"{float(record['confidence']):.1%}"
                    if record["confidence"] is not None
                    else "—"
                ),
                ", ".join(flags),
            ]
            for col, value in enumerate(row_values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, int(record["epoch_index"]))
                if record["manual_override"]:
                    item.setBackground(QColor("#13334a"))
                elif record["low_confidence"]:
                    item.setBackground(QColor("#4d1f1f"))
                self.queue_table.setItem(row, col, item)
        self.queue_table.resizeColumnsToContents()
        self._select_table_row_for_epoch(self.selected_epoch)

    def _refresh_detail(self):
        """Refresh the signal reader, metadata, and plot markers."""
        if not self.records:
            self.epoch_header_label.setText("—")
            self.model_stage_label.setText("—")
            self.final_stage_label.setText("—")
            self.override_badge_label.setText("—")
            self.confidence_label.setText("—")
            self.channel_plan_label.setText("—")
            self.page_status_label.setText("Page: --")
            self.active_epoch_label.setText("Active: --")
            self.plot_widget.render_page(
                page_start_sec=self.page_start_sec,
                page_duration_sec=self.page_duration_sec,
                selected_epoch=self.selected_epoch,
                epoch_sec=self.epoch_sec,
                confidences=self.confidences,
                manual_overrides=self.manual_overrides,
                threshold=self.confidence_threshold,
            )
            self.navigator.set_page(self.page_start_sec, self.page_duration_sec, 0.0)
            return

        # Guard against a stale selection index outliving a shorter reload.
        self.selected_epoch = min(self.selected_epoch, len(self.records) - 1)
        record = self.records[self.selected_epoch]
        active_start = self.selected_epoch * self.epoch_sec
        active_end = active_start + self.epoch_sec
        self.epoch_header_label.setText(
            f"Epoch {record['epoch']} ({record['time_hms']})"
        )
        self.model_stage_label.setText(str(record["model_stage"]))
        self.final_stage_label.setText(str(record["final_stage"]))
        self.override_badge_label.setText(
            str(record["override_stage"]) if record["manual_override"] else "No"
        )
        if record["confidence"] is None:
            self.confidence_label.setText("Unavailable")
        else:
            band = _get_confidence_band(
                float(record["confidence"]),
                self.confidence_threshold,
            )
            self.confidence_label.setText(
                f"{float(record['confidence']):.1%} ({CONFIDENCE_BAND_LABELS[band]})"
            )

        self.channel_plan_label.setText(self.display_preset_name)
        eeg_filter = _format_display_filter_label(
            DisplayFilterSpec(self.eeg_low_cut, self.eeg_high_cut, self.notch_hz)
        )
        self.page_status_label.setText(
            f"Page: {_format_elapsed_time_hms(int(self.page_start_sec))} - "
            f"{_format_elapsed_time_hms(int(self.page_start_sec + self.page_duration_sec))}   "
            f"Sensitivity: EEG/EOG {self.eeg_eog_uv_per_div:.0f} µV/div · "
            f"EMG {self.emg_uv_per_div:.0f} µV/div   "
            f"EEG filter: {eeg_filter}"
        )
        self.active_epoch_label.setText(
            f"Active: Epoch {record['epoch']} / {_format_elapsed_time_hms(active_start)} - "
            f"{_format_elapsed_time_hms(active_end)}"
        )

        stage_order = ["W", "N1", "N2", "N3", "REM"]
        probs = (
            self.probabilities[self.selected_epoch]
            if self.probabilities is not None
            and self.selected_epoch < len(self.probabilities)
            else None
        )
        for row, stage_name in enumerate(stage_order):
            stage_item = QTableWidgetItem(stage_name)
            self.prob_table.setItem(row, 0, stage_item)
            value = "—"
            if probs is not None and row < len(probs):
                value = f"{float(probs[row]):.1%}"
            prob_item = QTableWidgetItem(value)
            if stage_name == record["model_stage"]:
                prob_item.setBackground(QColor("#1f4d1f"))
            if stage_name == record["final_stage"] and record["manual_override"]:
                prob_item.setBackground(QColor("#13334a"))
            self.prob_table.setItem(row, 1, prob_item)

        self.plot_widget.eeg_eog_uv_per_div = self.eeg_eog_uv_per_div
        self.plot_widget.emg_uv_per_div = self.emg_uv_per_div
        self.plot_widget.spacing_multiplier = SIGNAL_REVIEW_SPACING_PRESETS[
            self.spacing_preset_name
        ]
        self.plot_widget.render_page(
            page_start_sec=self.page_start_sec,
            page_duration_sec=self.page_duration_sec,
            selected_epoch=self.selected_epoch,
            epoch_sec=self.epoch_sec,
            confidences=self.confidences,
            manual_overrides=self.manual_overrides,
            threshold=self.confidence_threshold,
        )
        self.navigator.set_page(
            self.page_start_sec,
            self.page_duration_sec,
            float(len(self.records) * self.epoch_sec),
        )

    def _select_table_row_for_epoch(self, epoch_idx: int):
        """Select the first visible queue row that matches the epoch."""
        self.queue_table.blockSignals(True)
        self.queue_table.clearSelection()
        for row, record in enumerate(self.filtered_records):
            if int(record["epoch_index"]) == int(epoch_idx):
                self.queue_table.selectRow(row)
                break
        self.queue_table.blockSignals(False)

    def _on_table_selection_changed(self):
        """Handle queue table selection changes."""
        selected_items = self.queue_table.selectedItems()
        if not selected_items:
            return
        epoch_idx = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if epoch_idx is None:
            return
        self.select_epoch(int(epoch_idx), center_page=True)
        self.epoch_selected.emit(self.selected_epoch)

    def _emit_epoch_selection(self, epoch_idx: int, *, center_page: bool = False):
        """Emit a validated epoch-selection request."""
        if self.predictions is None or len(self.predictions) == 0:
            return
        epoch_idx = max(0, min(int(epoch_idx), len(self.predictions) - 1))
        self.select_epoch(epoch_idx, center_page=center_page)
        self.epoch_selected.emit(epoch_idx)

    def _select_previous_filtered(self):
        """Jump to the previous epoch in the queue's current order.

        Walks position-within-queue so uncertainty-ranked ordering is honored;
        in chronological order this matches the previous flagged epoch.
        """
        if not self.filtered_records:
            return
        epoch_indices = [int(record["epoch_index"]) for record in self.filtered_records]
        if self.selected_epoch in epoch_indices:
            pos = epoch_indices.index(self.selected_epoch)
            target = epoch_indices[pos - 1] if pos > 0 else epoch_indices[-1]
        elif self.rank_by_uncertainty:
            target = epoch_indices[0]
        else:
            previous = [idx for idx in epoch_indices if idx < self.selected_epoch]
            target = previous[-1] if previous else epoch_indices[-1]
        self._emit_epoch_selection(
            target,
            center_page=self.follow_queue_checkbox.isChecked(),
        )

    def _select_next_filtered(self):
        """Jump to the next epoch in the queue's current order.

        Walks position-within-queue so uncertainty-ranked ordering is honored;
        in chronological order this matches the next flagged epoch.
        """
        if not self.filtered_records:
            return
        epoch_indices = [int(record["epoch_index"]) for record in self.filtered_records]
        if self.selected_epoch in epoch_indices:
            pos = epoch_indices.index(self.selected_epoch)
            target = (
                epoch_indices[pos + 1]
                if pos + 1 < len(epoch_indices)
                else epoch_indices[0]
            )
        elif self.rank_by_uncertainty:
            target = epoch_indices[0]
        else:
            following = [idx for idx in epoch_indices if idx > self.selected_epoch]
            target = following[0] if following else epoch_indices[0]
        self._emit_epoch_selection(
            target,
            center_page=self.follow_queue_checkbox.isChecked(),
        )

    def _request_manual_override(self, stage_idx: int):
        """Request a manual override for the selected epoch."""
        if self.predictions is None or len(self.predictions) == 0:
            return
        self.manual_override_requested.emit(self.selected_epoch, int(stage_idx))

    def _request_clear_override(self):
        """Request clearing the override for the selected epoch."""
        if self.predictions is None or len(self.predictions) == 0:
            return
        self.clear_override_requested.emit(self.selected_epoch)


class InferenceWorker(QThread):
    """Worker thread for running inference without blocking the GUI."""

    # Signals
    progress = Signal(str, int)  # (message, percent)
    result_ready = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        edf_path: str,
        checkpoint: str,
        canon_json: str | None,
        output_dir: str,
        device: str,
        options: ScoreOptions,
        cleanup_paths: list[str] | None = None,
    ):
        super().__init__()
        self.edf_path = edf_path
        self.checkpoint = checkpoint
        self.canon_json = canon_json
        self.output_dir = output_dir
        self.device = device
        self.options = options
        self.cleanup_paths = list(cleanup_paths or [])
        self._is_running = True

    def run(self):
        """Run inference in background thread."""
        try:
            from spectra.inference import score_recording

            result = score_recording(
                edf_path=self.edf_path,
                checkpoint=self.checkpoint,
                canon_json=self.canon_json,
                output_dir=self.output_dir,
                device=self.device,
                options=self.options,
                progress_callback=self._progress_callback,
            )
            if self._is_running:
                self.result_ready.emit(result)
            else:
                self.cancelled.emit()
        except InferenceCancelled:
            self.cancelled.emit()
        except Exception as e:
            if self._is_running:
                error_msg = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
                self.failed.emit(error_msg)
            else:
                self.cancelled.emit()
        finally:
            for cleanup_path in self.cleanup_paths:
                try:
                    Path(cleanup_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _progress_callback(self, message: str, percent: int):
        """Callback for progress updates."""
        if not self._is_running:
            raise InferenceCancelled("Inference cancelled by user")
        self.progress.emit(message, percent)

    def stop(self):
        """Stop the worker."""
        self._is_running = False


class EpochDetailsDialog(QDialog):
    """Dialog showing detailed information about a specific sleep epoch."""

    STAGE_COLORS = {
        "W": "#E53935",  # Red for Wake
        "N1": "#42A5F5",  # Light blue for N1
        "N2": "#1976D2",  # Medium blue for N2
        "N3": "#0D47A1",  # Dark blue for N3
        "REM": "#43A047",  # Green for REM
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Epoch Details")
        self.setMinimumWidth(400)
        self.setup_ui()

    def setup_ui(self):
        """Create the dialog UI."""
        layout = QVBoxLayout(self)

        # Header with epoch number and time
        self.header_label = QLabel()
        self.header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.header_label)

        # Stage display with color
        stage_frame = QFrame()
        stage_frame.setStyleSheet(
            "background-color: #2d2d2d; border-radius: 8px; padding: 10px;"
        )
        stage_layout = QVBoxLayout(stage_frame)

        self.stage_label = QLabel()
        self.stage_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stage_layout.addWidget(self.stage_label)

        self.final_stage_note_label = QLabel()
        self.final_stage_note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.final_stage_note_label.setStyleSheet("font-size: 13px; color: #4FC3F7;")
        stage_layout.addWidget(self.final_stage_note_label)

        self.confidence_label = QLabel()
        self.confidence_label.setStyleSheet("font-size: 14px;")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stage_layout.addWidget(self.confidence_label)

        layout.addWidget(stage_frame)

        # Probabilities table
        prob_group = QGroupBox("Class Probabilities")
        prob_layout = QVBoxLayout(prob_group)

        self.prob_table = QTableWidget(5, 2)
        self.prob_table.setHorizontalHeaderLabels(["Stage", "Probability"])
        self.prob_table.horizontalHeader().setStretchLastSection(True)
        self.prob_table.verticalHeader().setVisible(False)
        self.prob_table.setMaximumHeight(180)
        prob_layout.addWidget(self.prob_table)

        layout.addWidget(prob_group)

        # Context info
        context_group = QGroupBox("Epoch Context")
        context_layout = QFormLayout(context_group)

        self.prev_stage_label = QLabel()
        self.next_stage_label = QLabel()
        self.transition_label = QLabel()

        context_layout.addRow("Previous Stage:", self.prev_stage_label)
        context_layout.addRow("Next Stage:", self.next_stage_label)
        context_layout.addRow("Transition:", self.transition_label)

        layout.addWidget(context_group)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.prev_button = QPushButton("◀ Previous Epoch")
        self.prev_button.clicked.connect(self.go_previous)
        nav_layout.addWidget(self.prev_button)

        nav_layout.addStretch()

        self.next_button = QPushButton("Next Epoch ▶")
        self.next_button.clicked.connect(self.go_next)
        nav_layout.addWidget(self.next_button)

        layout.addLayout(nav_layout)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)

        # Store data
        self.current_epoch = 0
        self.predictions = None
        self.final_predictions = None
        self.confidence = None
        self.probabilities = None
        self.manual_overrides: dict[int, int] = {}
        self.epoch_callback = None
        self.confidence_threshold = DEFAULT_LOW_CONFIDENCE_THRESHOLD

    def set_data(
        self,
        predictions: np.ndarray | None,
        confidence: np.ndarray | None,
        probabilities: np.ndarray | None = None,
        *,
        final_predictions: np.ndarray | None = None,
        manual_overrides: dict[int, int] | None = None,
        epoch_callback=None,
    ):
        """Set the full prediction data for navigation."""
        self.predictions = predictions
        self.final_predictions = final_predictions
        self.confidence = confidence
        self.probabilities = probabilities
        self.manual_overrides = _coerce_manual_score_overrides(
            manual_overrides,
            n_epochs=(len(predictions) if predictions is not None else None),
        )
        self.epoch_callback = epoch_callback

    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold used for band labels."""
        self.confidence_threshold = float(threshold)

    def show_epoch(
        self,
        epoch_idx: int,
        stage: StageInput,
        confidence: float | np.floating | None,
        probs: np.ndarray | None = None,
        recording_start_time: str | None = None,
        final_stage: StageInput | None = None,
    ):
        """Display details for a specific epoch."""
        self.current_epoch = epoch_idx
        stage_label = _normalize_stage_label(stage)
        final_stage_label = (
            _normalize_stage_label(final_stage)
            if final_stage is not None
            else stage_label
        )
        probability_values = (
            np.asarray(probs, dtype=np.float32).reshape(-1)
            if probs is not None
            else None
        )

        # Format time
        epoch_start_sec = epoch_idx * 30
        hours = epoch_start_sec // 3600
        minutes = (epoch_start_sec % 3600) // 60
        seconds = epoch_start_sec % 60

        if recording_start_time:
            self.header_label.setText(
                f"Epoch {epoch_idx + 1} — {recording_start_time} + {hours:02d}:{minutes:02d}:{seconds:02d}"
            )
        else:
            self.header_label.setText(
                f"Epoch {epoch_idx + 1} — Time: {hours:02d}:{minutes:02d}:{seconds:02d}"
            )

        # Stage with color
        color = self.STAGE_COLORS.get(stage_label, "#888888")
        self.stage_label.setText(
            STAGE_DETAILS_NAMES.get(stage_label, stage_label) or stage_label
        )
        self.stage_label.setStyleSheet(
            f"font-size: 24px; font-weight: bold; color: {color};"
        )
        if final_stage_label != stage_label:
            final_detail = STAGE_DETAILS_NAMES.get(final_stage_label, final_stage_label)
            self.final_stage_note_label.setText(
                f"Final rescored stage: {final_detail} (model: {stage_label})"
            )
        else:
            self.final_stage_note_label.setText("")

        # Confidence with color coding
        if confidence is None:
            self.confidence_label.setText("Confidence unavailable")
            self.confidence_label.setStyleSheet("font-size: 14px; color: #888888;")
        else:
            band = _get_confidence_band(float(confidence), self.confidence_threshold)
            conf_color = CONFIDENCE_BAND_COLORS[band]
            conf_text = CONFIDENCE_BAND_LABELS[band]
            self.confidence_label.setText(f"{float(confidence):.1%} — {conf_text}")
            self.confidence_label.setStyleSheet(
                f"font-size: 14px; color: {conf_color};"
            )

        # Probabilities
        stage_order = ["W", "N1", "N2", "N3", "REM"]
        for row, s in enumerate(stage_order):
            stage_item = QTableWidgetItem(s)
            stage_item.setForeground(QColor(self.STAGE_COLORS.get(s, "#888888")))
            self.prob_table.setItem(row, 0, stage_item)

            if probability_values is not None and len(probability_values) > row:
                prob_val = float(probability_values[row])
                prob_item = QTableWidgetItem(f"{prob_val:.1%}")
                # Highlight the predicted stage
                if s == stage_label:
                    prob_item.setBackground(QColor("#3d5c3d"))
                self.prob_table.setItem(row, 1, prob_item)
            else:
                self.prob_table.setItem(row, 1, QTableWidgetItem("—"))

        # Context info
        if self.predictions is not None:
            stages = ["W", "N1", "N2", "N3", "REM"]

            # Previous stage
            if epoch_idx > 0:
                prev_stage = stages[self.predictions[epoch_idx - 1]]
                self.prev_stage_label.setText(prev_stage)
                self.prev_stage_label.setStyleSheet(
                    f"color: {self.STAGE_COLORS.get(prev_stage, '#888')};"
                )
            else:
                self.prev_stage_label.setText("—")

            # Next stage
            if epoch_idx < len(self.predictions) - 1:
                next_stage = stages[self.predictions[epoch_idx + 1]]
                self.next_stage_label.setText(next_stage)
                self.next_stage_label.setStyleSheet(
                    f"color: {self.STAGE_COLORS.get(next_stage, '#888')};"
                )
            else:
                self.next_stage_label.setText("—")

            # Transition detection
            is_transition = False
            prev_stage = None
            if epoch_idx > 0:
                prev_stage = stages[self.predictions[epoch_idx - 1]]
                if prev_stage != stage_label:
                    is_transition = True
                    self.transition_label.setText(f"← Transition from {prev_stage}")
                    self.transition_label.setStyleSheet(
                        "color: #FFA726; font-weight: bold;"
                    )
            if epoch_idx < len(self.predictions) - 1:
                next_stage = stages[self.predictions[epoch_idx + 1]]
                if next_stage != stage_label:
                    if is_transition:
                        transition_from = (
                            prev_stage if prev_stage is not None else stage_label
                        )
                        self.transition_label.setText(
                            f"Transition: {transition_from} → {stage_label} → {next_stage}"
                        )
                    else:
                        self.transition_label.setText(f"Transition to {next_stage} →")
                        self.transition_label.setStyleSheet(
                            "color: #FFA726; font-weight: bold;"
                        )
                    is_transition = True
            if not is_transition:
                self.transition_label.setText("No transition")
                self.transition_label.setStyleSheet("color: #888;")

        # Update navigation buttons
        self.prev_button.setEnabled(epoch_idx > 0)
        if self.predictions is not None:
            self.next_button.setEnabled(epoch_idx < len(self.predictions) - 1)

    def go_previous(self):
        """Navigate to previous epoch."""
        if self.current_epoch > 0:
            self.navigate_to_epoch(self.current_epoch - 1)

    def go_next(self):
        """Navigate to next epoch."""
        if (
            self.predictions is not None
            and self.current_epoch < len(self.predictions) - 1
        ):
            self.navigate_to_epoch(self.current_epoch + 1)

    def navigate_to_epoch(self, epoch_idx):
        """Navigate to a specific epoch."""
        if self.predictions is None:
            return

        stages = ["W", "N1", "N2", "N3", "REM"]
        stage = stages[self.predictions[epoch_idx]]
        final_stage = (
            stages[self.final_predictions[epoch_idx]]
            if self.final_predictions is not None
            and epoch_idx < len(self.final_predictions)
            else stage
        )
        conf = (
            float(self.confidence[epoch_idx]) if self.confidence is not None else None
        )
        probs = (
            self.probabilities[epoch_idx] if self.probabilities is not None else None
        )

        self.show_epoch(epoch_idx, stage, conf, probs, final_stage=final_stage)

        # Callback to update canvas highlight if provided
        if self.epoch_callback:
            self.epoch_callback(epoch_idx)


class SleepArchitectureWidget(QWidget):
    """Widget showing detailed sleep architecture charts and analysis."""

    STAGE_COLORS = {
        "W": "#E53935",  # Red for Wake
        "N1": "#42A5F5",  # Light blue for N1
        "N2": "#1976D2",  # Medium blue for N2
        "N3": "#0D47A1",  # Dark blue for N3
        "REM": "#43A047",  # Green for REM
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictions: np.ndarray | None = None
        self.epoch_sec = 30
        self.setup_ui()

    def setup_ui(self):
        """Create the widget UI."""
        layout = QVBoxLayout(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(0, 0)
        if FigureCanvas is None:
            raise RuntimeError("FigureCanvas is not initialized")
        figure_canvas_cls = cast(type, FigureCanvas)

        # Create scroll area for all charts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setMinimumSize(0, 0)

        scroll_content = QWidget()
        scroll_content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        scroll_content.setMinimumSize(0, 0)
        scroll_layout = QVBoxLayout(scroll_content)

        # Stage Distribution Bar Chart
        dist_group = QGroupBox("Stage Distribution")
        dist_layout = QVBoxLayout(dist_group)

        self.dist_figure = matplotlib.figure.Figure(figsize=(8, 3), facecolor="#1e1e1e")
        self.dist_canvas = figure_canvas_cls(self.dist_figure)
        self.dist_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.dist_canvas.setMinimumSize(0, 0)
        self.dist_canvas.setMinimumHeight(200)
        dist_layout.addWidget(self.dist_canvas)

        scroll_layout.addWidget(dist_group)

        # Stage Duration Over Time (Stacked Area)
        time_group = QGroupBox("Sleep Architecture Over Time")
        time_layout = QVBoxLayout(time_group)

        self.time_figure = matplotlib.figure.Figure(figsize=(8, 3), facecolor="#1e1e1e")
        self.time_canvas = figure_canvas_cls(self.time_figure)
        self.time_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.time_canvas.setMinimumSize(0, 0)
        self.time_canvas.setMinimumHeight(200)
        time_layout.addWidget(self.time_canvas)

        scroll_layout.addWidget(time_group)

        # Transition Matrix Heatmap
        trans_group = QGroupBox("Stage Transition Matrix")
        trans_layout = QVBoxLayout(trans_group)

        self.trans_figure = matplotlib.figure.Figure(
            figsize=(6, 5), facecolor="#1e1e1e"
        )
        self.trans_canvas = figure_canvas_cls(self.trans_figure)
        self.trans_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.trans_canvas.setMinimumSize(0, 0)
        self.trans_canvas.setMinimumHeight(350)
        trans_layout.addWidget(self.trans_canvas)

        # Transition stats
        self.trans_stats_label = QLabel()
        self.trans_stats_label.setWordWrap(True)
        trans_layout.addWidget(self.trans_stats_label)

        scroll_layout.addWidget(trans_group)

        # Sleep Fragmentation Analysis
        frag_group = QGroupBox("Sleep Fragmentation Analysis")
        frag_layout = QVBoxLayout(frag_group)

        self.frag_table = QTableWidget(5, 4)
        self.frag_table.setHorizontalHeaderLabels(
            ["Stage", "Bout Count", "Avg Duration", "Total Time"]
        )
        self.frag_table.horizontalHeader().setStretchLastSection(True)
        self.frag_table.verticalHeader().setVisible(False)
        self.frag_table.setMaximumHeight(180)
        frag_layout.addWidget(self.frag_table)

        scroll_layout.addWidget(frag_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Export button
        export_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Architecture Report")
        self.export_button.clicked.connect(self.export_report)
        self.export_button.setEnabled(False)
        export_layout.addWidget(self.export_button)
        export_layout.addStretch()
        layout.addLayout(export_layout)

    def update_analysis(self, predictions, epoch_sec=30):
        """Update all charts with new prediction data."""
        self.predictions = predictions
        self.epoch_sec = epoch_sec

        if predictions is None or len(predictions) == 0:
            self.export_button.setEnabled(False)
            return

        self.export_button.setEnabled(True)

        # Update all charts
        self._update_distribution_chart()
        self._update_time_chart()
        self._update_transition_matrix()
        self._update_fragmentation_table()

    def _update_distribution_chart(self):
        """Update the stage distribution bar chart."""
        if self.predictions is None:
            return
        self.dist_figure.clear()
        ax = self.dist_figure.add_subplot(111)
        ax.set_facecolor("#2b2b2b")

        stages = ["W", "N1", "N2", "N3", "REM"]
        colors = [self.STAGE_COLORS[s] for s in stages]

        # Calculate counts and percentages
        counts = [(self.predictions == i).sum() for i in range(5)]
        total = sum(counts)
        percentages = [100 * c / total if total > 0 else 0 for c in counts]
        minutes = [c * self.epoch_sec / 60 for c in counts]

        # Create bar chart
        bars = ax.bar(stages, minutes, color=colors, edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bar, pct, mins in zip(bars, percentages, minutes, strict=True):
            height = bar.get_height()
            ax.annotate(
                f"{mins:.1f}m\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="white",
                fontsize=9,
            )

        ax.set_ylabel("Duration (minutes)", color="white")
        ax.set_xlabel("Sleep Stage", color="white")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_color("#444")

        self.dist_figure.tight_layout()
        self.dist_canvas.draw()

    def _update_time_chart(self):
        """Update the sleep architecture over time chart."""
        if self.predictions is None:
            return
        self.time_figure.clear()
        ax = self.time_figure.add_subplot(111)
        ax.set_facecolor("#2b2b2b")

        stages = ["W", "N1", "N2", "N3", "REM"]
        colors = [self.STAGE_COLORS[s] for s in stages]

        # Create hourly bins
        epochs_per_hour = 3600 // self.epoch_sec
        n_hours = int(np.ceil(len(self.predictions) / epochs_per_hour))

        hourly_data = np.zeros((5, n_hours))
        for hour in range(n_hours):
            start_idx = hour * epochs_per_hour
            end_idx = min((hour + 1) * epochs_per_hour, len(self.predictions))
            hour_preds = self.predictions[start_idx:end_idx]
            for stage in range(5):
                hourly_data[stage, hour] = (
                    (hour_preds == stage).sum() * self.epoch_sec / 60
                )

        # Stacked area chart
        hours = np.arange(n_hours)
        ax.stackplot(hours, hourly_data, labels=stages, colors=colors, alpha=0.8)

        ax.set_xlabel("Recording Hour", color="white")
        ax.set_ylabel("Minutes per Hour", color="white")
        ax.tick_params(colors="white")
        ax.legend(loc="upper right", fontsize=8)
        x_max = max(n_hours - 1, 1)
        ax.set_xlim(0, x_max)

        for spine in ax.spines.values():
            spine.set_color("#444")

        self.time_figure.tight_layout()
        self.time_canvas.draw()

    def _update_transition_matrix(self):
        """Update the stage transition matrix heatmap."""
        if self.predictions is None:
            return
        self.trans_figure.clear()
        ax = self.trans_figure.add_subplot(111)
        ax.set_facecolor("#2b2b2b")

        stages = ["W", "N1", "N2", "N3", "REM"]

        # Calculate transition counts
        trans_matrix = np.zeros((5, 5), dtype=int)
        for i in range(len(self.predictions) - 1):
            from_stage = self.predictions[i]
            to_stage = self.predictions[i + 1]
            trans_matrix[from_stage, to_stage] += 1

        # Normalize to percentages (row-wise)
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_pct = np.divide(
            trans_matrix * 100.0,
            row_sums,
            out=np.zeros_like(trans_matrix, dtype=np.float64),
            where=row_sums > 0,
        )

        # Create heatmap
        im = ax.imshow(trans_pct, cmap="YlOrRd", aspect="auto")

        # Add text annotations
        for i in range(5):
            for j in range(5):
                text_color = "white" if trans_pct[i, j] > 50 else "black"
                ax.text(
                    j,
                    i,
                    f"{trans_pct[i, j]:.1f}%\n({trans_matrix[i, j]})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(stages, color="white")
        ax.set_yticklabels(stages, color="white")
        ax.set_xlabel("To Stage", color="white")
        ax.set_ylabel("From Stage", color="white")

        # Colorbar
        cbar = self.trans_figure.colorbar(im, ax=ax)
        cbar.set_label("Transition %", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        outline = getattr(cbar, "outline", None)
        if outline is not None:
            outline.set_edgecolor("#444")
        plt = cbar.ax.yaxis.get_ticklabels()
        for t in plt:
            t.set_color("white")

        # Calculate transition statistics
        total_transitions = trans_matrix.sum() - np.trace(trans_matrix)
        n_arousals = (
            trans_matrix[1:, 0].sum() + trans_matrix[2:, 1].sum()
        )  # Transitions to lighter sleep
        n_deepening = (
            trans_matrix[0, 1:].sum() + trans_matrix[1, 2:].sum()
        )  # Transitions to deeper sleep

        self.trans_stats_label.setText(
            f"<b>Transition Statistics:</b><br>"
            f"Total stage transitions: {total_transitions}<br>"
            f"Transitions to lighter sleep (arousals): {n_arousals}<br>"
            f"Transitions to deeper sleep: {n_deepening}<br>"
            f"Average transitions per hour: {total_transitions / (len(self.predictions) * self.epoch_sec / 3600):.1f}"
        )
        self.trans_stats_label.setStyleSheet("color: #ddd;")

        self.trans_figure.tight_layout()
        self.trans_canvas.draw()

    def _update_fragmentation_table(self):
        """Update the sleep fragmentation analysis table."""
        if self.predictions is None:
            return
        stages = ["W", "N1", "N2", "N3", "REM"]

        for stage_idx, stage_name in enumerate(stages):
            # Find bouts (consecutive epochs of same stage)
            is_stage = np.asarray(self.predictions == stage_idx, dtype=np.int8)
            padded_mask = np.pad(is_stage, (1, 1), constant_values=0)
            stage_edges = np.diff(padded_mask)
            bout_starts = np.where(stage_edges == 1)[0]
            bout_ends = np.where(stage_edges == -1)[0]

            n_bouts = len(bout_starts)
            if n_bouts > 0:
                bout_durations = (
                    (bout_ends - bout_starts) * self.epoch_sec / 60
                )  # minutes
                avg_duration = np.mean(bout_durations)
                total_time = np.sum(bout_durations)
            else:
                avg_duration = 0
                total_time = 0

            # Update table
            stage_item = QTableWidgetItem(stage_name)
            stage_item.setForeground(QColor(self.STAGE_COLORS[stage_name]))
            self.frag_table.setItem(stage_idx, 0, stage_item)
            self.frag_table.setItem(stage_idx, 1, QTableWidgetItem(str(n_bouts)))
            self.frag_table.setItem(
                stage_idx, 2, QTableWidgetItem(f"{avg_duration:.1f} min")
            )
            self.frag_table.setItem(
                stage_idx, 3, QTableWidgetItem(f"{total_time:.1f} min")
            )

    def export_report(self):
        """Export sleep architecture analysis to a file."""
        from datetime import datetime

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Sleep Architecture Report",
            f"sleep_architecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Report (*.html);;CSV Data (*.csv)",
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".html"):
                self._export_html_report(file_path)
            else:
                self._export_csv_data(file_path)

            QMessageBox.information(
                self, "Export Complete", f"Report saved to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Export Error", f"Failed to export report:\n{str(e)}"
            )

    def _export_html_report(self, file_path):
        """Export as HTML report with embedded charts."""
        import base64
        from io import BytesIO

        stages = ["W", "N1", "N2", "N3", "REM"]

        # Convert figures to base64
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(
                buf, format="png", dpi=100, facecolor="#1e1e1e", bbox_inches="tight"
            )
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        dist_img = fig_to_base64(self.dist_figure)
        time_img = fig_to_base64(self.time_figure)
        trans_img = fig_to_base64(self.trans_figure)

        # Calculate statistics
        counts = [(self.predictions == i).sum() for i in range(5)]
        total = sum(counts)
        minutes = [c * self.epoch_sec / 60 for c in counts]

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sleep Architecture Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #1e1e1e; color: #ddd; padding: 20px; }}
                h1, h2 {{ color: #fff; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                .chart img {{ max-width: 100%; border: 1px solid #444; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #444; padding: 8px; text-align: left; }}
                th {{ background-color: #333; }}
                .stage-W {{ color: #E53935; }}
                .stage-N1 {{ color: #42A5F5; }}
                .stage-N2 {{ color: #1976D2; }}
                .stage-N3 {{ color: #0D47A1; }}
                .stage-REM {{ color: #43A047; }}
            </style>
        </head>
        <body>
            <h1>Sleep Architecture Report</h1>
            <p>Generated: {Path(file_path).stem}</p>

            <h2>Stage Distribution</h2>
            <div class="chart"><img src="data:image/png;base64,{dist_img}"></div>
            <table>
                <tr><th>Stage</th><th>Epochs</th><th>Duration (min)</th><th>Percentage</th></tr>
                {"".join(f'<tr><td class="stage-{s}">{s}</td><td>{counts[i]}</td><td>{minutes[i]:.1f}</td><td>{100 * counts[i] / total:.1f}%</td></tr>' for i, s in enumerate(stages))}
            </table>

            <h2>Sleep Architecture Over Time</h2>
            <div class="chart"><img src="data:image/png;base64,{time_img}"></div>

            <h2>Stage Transition Matrix</h2>
            <div class="chart"><img src="data:image/png;base64,{trans_img}"></div>

            <p>{self.trans_stats_label.text()}</p>
        </body>
        </html>
        """

        with open(file_path, "w") as f:
            f.write(html)

    def _export_csv_data(self, file_path):
        """Export raw data as CSV."""
        if self.predictions is None:
            raise ValueError("No predictions available for CSV export")
        stages = ["W", "N1", "N2", "N3", "REM"]
        counts = [(self.predictions == i).sum() for i in range(5)]
        total = sum(counts)
        minutes = [c * self.epoch_sec / 60 for c in counts]

        lines = [
            "Sleep Architecture Analysis",
            "",
            "Stage Distribution",
            "Stage,Epochs,Duration_min,Percentage",
        ]
        for i, s in enumerate(stages):
            lines.append(
                f"{s},{counts[i]},{minutes[i]:.2f},{100 * counts[i] / total:.2f}"
            )

        lines.extend(["", "Transition Matrix (counts)", "From\\To," + ",".join(stages)])

        trans_matrix = np.zeros((5, 5), dtype=int)
        for i in range(len(self.predictions) - 1):
            trans_matrix[self.predictions[i], self.predictions[i + 1]] += 1

        for i, s in enumerate(stages):
            lines.append(f"{s}," + ",".join(str(trans_matrix[i, j]) for j in range(5)))

        with open(file_path, "w") as f:
            f.write("\n".join(lines))


class _OverrideCommand(QUndoCommand):
    """Undoable single-epoch manual override change.

    ``new_stage`` of ``None`` means "remove the override for this epoch", which
    is how the GUI represents reverting to the model's base prediction.
    """

    def __init__(
        self,
        gui: InferenceGUI,
        *,
        epoch: int,
        old_stage: int | None,
        new_stage: int | None,
        text: str,
    ) -> None:
        super().__init__(text)
        self._gui = gui
        self._epoch = int(epoch)
        self._old_stage = old_stage
        self._new_stage = new_stage

    def redo(self) -> None:  # type: ignore[override]
        self._gui._set_override_state(
            self._epoch,
            self._new_stage,
            log_message=self.text(),
        )

    def undo(self) -> None:  # type: ignore[override]
        self._gui._set_override_state(
            self._epoch,
            self._old_stage,
            log_message=f"Undo: {self.text()}",
        )


class _BulkOverrideCommand(QUndoCommand):
    """Undoable bulk replacement of the entire manual-override dict."""

    def __init__(
        self,
        gui: InferenceGUI,
        *,
        previous: dict[int, int],
        new: dict[int, int],
        text: str,
    ) -> None:
        super().__init__(text)
        self._gui = gui
        self._previous = {int(k): int(v) for k, v in previous.items()}
        self._new = {int(k): int(v) for k, v in new.items()}

    def redo(self) -> None:  # type: ignore[override]
        self._gui._replace_all_overrides(self._new, log_message=self.text())

    def undo(self) -> None:  # type: ignore[override]
        self._gui._replace_all_overrides(
            self._previous, log_message=f"Undo: {self.text()}"
        )


@dataclass(frozen=True)
class ThemePalette:
    """Color palette for one GUI theme.

    Backs :func:`_build_stylesheet` so the light and dark themes share a single
    Qt stylesheet template and differ only by these values.
    """

    window_bg: str
    text: str
    accent: str
    groupbox_bg: str
    pane_bg: str
    input_bg: str
    border_soft: str
    input_border: str
    input_focus_bg: str
    input_extra: str
    disabled_bg: str
    input_disabled_text: str
    input_disabled_border: str
    button_bg: str
    button_border: str
    button_text: str
    button_hover_bg: str
    button_hover_border: str
    button_pressed_bg: str
    button_disabled_text: str
    button_disabled_border: str
    tab_bg: str
    tab_text: str
    tab_selected_bg: str
    tab_hover_bg: str
    tab_hover_text: str
    table_grid: str
    table_text: str
    textedit_text: str
    header_bg: str
    header_text: str


_DARK_PALETTE = ThemePalette(
    window_bg="#2b2b2b",
    text="#ffffff",
    accent="#4CAF50",
    groupbox_bg="#333333",
    pane_bg="#2b2b2b",
    input_bg="#1e1e1e",
    border_soft="#444",
    input_border="#555",
    input_focus_bg="#252525",
    input_extra="",
    disabled_bg="#2b2b2b",
    input_disabled_text="#777",
    input_disabled_border="#444",
    button_bg="#444",
    button_border="#555",
    button_text="white",
    button_hover_bg="#555",
    button_hover_border="#666",
    button_pressed_bg="#333",
    button_disabled_text="#555",
    button_disabled_border="#333",
    tab_bg="#1e1e1e",
    tab_text="#aaa",
    tab_selected_bg="#2b2b2b",
    tab_hover_bg="#252525",
    tab_hover_text="#fff",
    table_grid="#444",
    table_text="#ddd",
    textedit_text="#d4d4d4",
    header_bg="#333",
    header_text="white",
)


_LIGHT_PALETTE = ThemePalette(
    window_bg="#f5f5f5",
    text="#1a1a1a",
    accent="#2E7D32",
    groupbox_bg="#ffffff",
    pane_bg="#ffffff",
    input_bg="#ffffff",
    border_soft="#c8c8c8",
    input_border="#c0c0c0",
    input_focus_bg="#f7fbf8",
    input_extra="selection-color: white;",
    disabled_bg="#ececec",
    input_disabled_text="#888",
    input_disabled_border="#d0d0d0",
    button_bg="#eaeaea",
    button_border="#bbbbbb",
    button_text="#1a1a1a",
    button_hover_bg="#dddddd",
    button_hover_border="#aaaaaa",
    button_pressed_bg="#cccccc",
    button_disabled_text="#aaaaaa",
    button_disabled_border="#d0d0d0",
    tab_bg="#eaeaea",
    tab_text="#555",
    tab_selected_bg="#ffffff",
    tab_hover_bg="#f5f5f5",
    tab_hover_text="#1a1a1a",
    table_grid="#d8d8d8",
    table_text="#1a1a1a",
    textedit_text="#1a1a1a",
    header_bg="#ececec",
    header_text="#1a1a1a",
)


def _build_stylesheet(p: ThemePalette) -> str:
    """Render the shared Qt stylesheet template for a :class:`ThemePalette`."""
    return f"""
        QMainWindow, QWidget {{
            background-color: {p.window_bg};
            color: {p.text};
            font-family: "Segoe UI", "Roboto", "Helvetica Neue", sans-serif;
        }}
        QWidget#workflowPage {{
            background-color: {p.window_bg};
        }}
        QWidget#pageHeader {{
            background-color: transparent;
        }}
        QLabel#stepBadge {{
            color: {p.accent};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
        }}
        QLabel#pageTitle {{
            color: {p.text};
            font-size: 22px;
            font-weight: 700;
        }}
        QLabel#pageSubtitle {{
            color: {p.tab_text};
            font-size: 12px;
        }}
        QLabel#emptyState {{
            color: {p.tab_text};
            background-color: {p.input_bg};
            border: 1px dashed {p.input_border};
            border-radius: 7px;
            padding: 12px 16px;
        }}
        QLabel#emptyState[state="ready"] {{
            color: {p.accent};
            border: 1px solid {p.accent};
        }}
        QLabel#emptyState[state="active"] {{
            color: {p.text};
            border: 1px solid {p.accent};
        }}

        QGroupBox {{
            font-weight: bold;
            border: 1px solid {p.border_soft};
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 10px;
            background-color: {p.groupbox_bg};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {p.accent};
        }}

        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {p.input_bg};
            border: 1px solid {p.input_border};
            border-radius: 4px;
            padding: 6px;
            color: {p.text};
            selection-background-color: {p.accent};
            {p.input_extra}
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border: 1px solid {p.accent};
            background-color: {p.input_focus_bg};
        }}
        QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
            background-color: {p.disabled_bg};
            color: {p.input_disabled_text};
            border: 1px solid {p.input_disabled_border};
        }}

        QPushButton {{
            background-color: {p.button_bg};
            border: 1px solid {p.button_border};
            border-radius: 4px;
            padding: 8px 16px;
            color: {p.button_text};
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {p.button_hover_bg};
            border: 1px solid {p.button_hover_border};
        }}
        QPushButton:pressed {{
            background-color: {p.button_pressed_bg};
        }}
        QPushButton:disabled {{
            background-color: {p.disabled_bg};
            color: {p.button_disabled_text};
            border: 1px solid {p.button_disabled_border};
        }}
        QPushButton[role="primary"] {{
            background-color: #2E7D32;
            border-color: #1B5E20;
            color: white;
        }}
        QPushButton[role="primary"]:hover {{
            background-color: #388E3C;
            border-color: #2E7D32;
        }}
        QPushButton[role="accent"] {{
            background-color: #1565C0;
            border-color: #0D47A1;
            color: white;
        }}
        QPushButton[role="accent"]:hover {{
            background-color: #1976D2;
        }}
        QPushButton[role="danger"] {{
            background-color: #C62828;
            border-color: #B71C1C;
            color: white;
        }}
        QPushButton[role="danger"]:hover {{
            background-color: #D32F2F;
        }}
        QPushButton[role="primary"]:disabled,
        QPushButton[role="accent"]:disabled,
        QPushButton[role="danger"]:disabled {{
            background-color: {p.disabled_bg};
            color: {p.button_disabled_text};
            border-color: {p.button_disabled_border};
        }}

        QTabWidget::pane {{
            border: 1px solid {p.border_soft};
            background-color: {p.pane_bg};
        }}
        QTabWidget::tab-bar {{
            left: 5px;
        }}
        QTabBar::tab {{
            background-color: {p.tab_bg};
            color: {p.tab_text};
            border: 1px solid {p.border_soft};
            border-bottom-color: {p.border_soft};
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 8px 16px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{
            background-color: {p.tab_selected_bg};
            color: {p.accent};
            border-bottom-color: {p.tab_selected_bg};
            font-weight: bold;
        }}
        QTabBar::tab:hover {{
            background-color: {p.tab_hover_bg};
            color: {p.tab_hover_text};
        }}

        QScrollArea {{
            border: none;
            background-color: transparent;
        }}
        QScrollBar:vertical {{
            border: none;
            background: {p.tab_bg};
            width: 10px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: {p.button_border};
            min-height: 20px;
            border-radius: 5px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}

        QProgressBar {{
            border: 1px solid {p.button_border};
            border-radius: 4px;
            text-align: center;
            background-color: {p.input_bg};
            color: {p.button_text};
            font-weight: bold;
        }}
        QProgressBar::chunk {{
            background-color: {p.accent};
            border-radius: 3px;
        }}

        QTextEdit {{
            background-color: {p.input_bg};
            color: {p.textedit_text};
            border: 1px solid {p.border_soft};
            border-radius: 4px;
        }}

        QSplitter::handle {{
            background-color: {p.border_soft};
        }}

        QTableWidget {{
            background-color: {p.input_bg};
            gridline-color: {p.table_grid};
            color: {p.table_text};
            border: 1px solid {p.border_soft};
        }}
        QHeaderView::section {{
            background-color: {p.header_bg};
            color: {p.header_text};
            padding: 4px;
            border: 1px solid {p.border_soft};
        }}
    """


class InferenceGUI(QMainWindow):
    """Main GUI window for PSG inference."""

    # Maximum number of recent files to remember per category
    MAX_RECENT_FILES = 10

    def __init__(self):
        super().__init__()
        self.worker: InferenceWorker | None = None
        self._logger: logging.Logger | None = None
        self._qt_log_handler: logging.Handler | None = None
        self.run_state = InferenceRunState()
        self.review_state = ReviewSessionState()
        self.project_state = ProjectState()
        self._advanced_mode: bool = True
        self._easy_mode: bool = True  # Start in easy mode by default

        self._override_undo_stack: QUndoStack = QUndoStack(self)
        self._override_undo_stack.setUndoLimit(500)
        self._review_signal_loader: ReviewSignalLoader | None = None
        self._close_pending: bool = False

        # Recent files storage
        self._recent_edf_files: list = []
        self._recent_checkpoint_files: list = []
        self._available_edf_channel_labels: list[str] = []
        self._channel_slot_overrides: dict[str, str | None] = (
            _default_channel_slot_overrides()
        )
        self._channel_slot_combos: dict[str, QComboBox] = {}
        self._channel_slot_signal_block = False

        # Batch processing queue. Each entry is
        # ``{"path": str, "status": str, "percent": int, "error": str}``.
        self._batch_queue: list[dict[str, Any]] = []
        self._batch_current_index: int = 0
        self._batch_mode: bool = False
        self._batch_paused: bool = False
        # Live reference to the dock widget so progress updates can reach it.
        self._batch_queue_dialog: QDialog | None = None
        self._batch_queue_table: QTableWidget | None = None

        # Settings dictionary for View menu and other persistent options
        self.settings: dict = {
            "font_size": 12,
            "high_contrast": False,
            "recent_projects": [],
        }

        self.initUI()
        self.apply_stylesheet()
        self.load_settings()
        self._update_setup_readiness()
        self._refresh_device_controls()
        self._update_status_bar()
        self.setup_keyboard_shortcuts()
        self._setup_project_menu()
        self._setup_view_menu()

        # Enable drag and drop
        self.setAcceptDrops(True)

    def _resolve_selected_device(self):
        if not hasattr(self, "device_combo"):
            return _resolve_gui_device("auto")
        return _resolve_gui_device(self.device_combo.currentText())

    def _refresh_device_controls(self):
        if not hasattr(self, "device_combo"):
            return

        device_info = self._resolve_selected_device()
        if hasattr(self, "device_combo"):
            self.device_combo.setToolTip(
                "Device for inference: auto (prefer CUDA, then MPS, then CPU),\n"
                "cuda (force NVIDIA GPU), mps (force Apple Silicon GPU), cpu (force CPU)"
            )

        if hasattr(self, "amp_mode_combo"):
            self.amp_mode_combo.setToolTip(
                "fp32: highest accuracy, fp16/bf16: best-effort mixed precision on supported accelerators"
            )

        if hasattr(self, "_version_label"):
            self._version_label.setToolTip(
                f"Current auto-selected backend: {device_info.description}"
            )

    def _clear_selected_device_cache(
        self,
        *,
        success_message: str | None = None,
        warning_prefix: str = "Could not clear accelerator cache",
        synchronize: bool = True,
        reset_peak_memory_stats: bool = False,
    ) -> bool:
        device_info = self._resolve_selected_device()
        try:
            cleared = _clear_gui_device_cache(
                device_info.torch_device,
                synchronize=synchronize,
                reset_peak_memory_stats=reset_peak_memory_stats,
            )
        except Exception as exc:
            self.log(f"{warning_prefix}: {exc}", logging.WARNING)
            return False

        if cleared and success_message:
            self.log(success_message, logging.INFO)
        return cleared

    def apply_stylesheet(self):
        """Apply the theme stylesheet stored in settings (defaults to dark)."""
        theme = self.settings.get("theme", "dark")
        if theme == "light":
            self.setStyleSheet(self._light_stylesheet())
        else:
            self.setStyleSheet(self._dark_stylesheet())

    @staticmethod
    def _dark_stylesheet() -> str:
        """Return the dark-theme Qt stylesheet."""
        return _build_stylesheet(_DARK_PALETTE)

    @staticmethod
    def _light_stylesheet() -> str:
        """Return the light-theme Qt stylesheet."""
        return _build_stylesheet(_LIGHT_PALETTE)

    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle("SPECTRA - Sleep Staging")

        app = cast(QApplication, QApplication.instance())
        screen = app.primaryScreen() if app is not None else None
        available_geometry = screen.availableGeometry() if screen is not None else None

        # 1024×720 is the smallest practical size on a 13" laptop. The Configuration
        # tab is wrapped in a QScrollArea, so users can still reach every control.
        min_width = 1024
        min_height = 720
        initial_width = 1400
        initial_height = 1000
        if available_geometry is not None:
            min_width = min(min_width, max(800, available_geometry.width() - 120))
            min_height = min(min_height, max(600, available_geometry.height() - 180))
            initial_width = min(
                initial_width,
                max(min_width, available_geometry.width() - 80),
            )
            initial_height = min(
                initial_height,
                max(min_height, available_geometry.height() - 80),
            )

        self.setMinimumSize(min_width, min_height)
        self.resize(initial_width, initial_height)

        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create status bar
        self._setup_status_bar()

        # Four workflow-level pages. Specialist analysis tools live inside Review.
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.tab_widget.setMinimumSize(0, 0)
        main_layout.addWidget(self.tab_widget, 1)

        self.setup_page = SetupTab()
        self.run_page = RunTab()
        self.review_page = ReviewTab()
        self.export_page = ExportTab()
        page_specs = (
            (self.setup_page, "Setup", QStyle.StandardPixmap.SP_DirOpenIcon),
            (self.run_page, "Run", QStyle.StandardPixmap.SP_MediaPlay),
            (
                self.review_page,
                "Review",
                QStyle.StandardPixmap.SP_FileDialogContentsView,
            ),
            (self.export_page, "Export", QStyle.StandardPixmap.SP_DialogSaveButton),
        )
        for page, label, icon_type in page_specs:
            self.tab_widget.addTab(page, self.style().standardIcon(icon_type), label)

        params_layout = self.setup_page.content_layout

        # Mode toggle section with Easy Mode / Advanced Mode
        mode_layout = QHBoxLayout()

        # Easy Mode checkbox (checked by default)
        self.easy_mode_toggle = QCheckBox("Essential options only")
        self.easy_mode_toggle.setToolTip(
            "Simplified interface: shows only essential options (EDF, Model, Channels)\n"
            "Uncheck to see all configuration options"
        )
        self.easy_mode_toggle.setChecked(True)
        self.easy_mode_toggle.stateChanged.connect(self.toggle_easy_mode)
        self.easy_mode_toggle.setStyleSheet(
            "QCheckBox { color: #4CAF50; font-weight: bold; }"
        )
        mode_layout.addWidget(self.easy_mode_toggle)

        mode_layout.addSpacing(20)

        # Advanced Mode checkbox
        self.mode_toggle = QCheckBox("Show expert options")
        self.mode_toggle.setToolTip("Show advanced options (MC Dropout, Calibration)")
        self.mode_toggle.setChecked(True)
        self.mode_toggle.stateChanged.connect(self.toggle_advanced_mode)
        self.mode_toggle.setEnabled(False)  # Disabled until Easy Mode is unchecked
        self.mode_toggle.setVisible(False)
        mode_layout.addWidget(self.mode_toggle)
        mode_layout.addStretch()

        # Keyboard shortcuts hint
        shortcuts_hint = QLabel(
            "<small style='color: #666;'>Press F1 for shortcuts</small>"
        )
        mode_layout.addWidget(shortcuts_hint)

        mode_layout.addSpacing(10)

        # Preview channels button
        self.preview_channels_btn = QPushButton("Preview EDF Channels")
        self.preview_channels_btn.setToolTip(
            "Check available channels in the EDF file before scoring"
        )
        self.preview_channels_btn.clicked.connect(self.preview_edf_channels)
        mode_layout.addWidget(self.preview_channels_btn)

        params_layout.addLayout(mode_layout)

        # Add parameter sections
        # Essential section (always visible)
        self.files_group = self.create_files_section()
        params_layout.addWidget(self.files_group)

        # Standard sections (hidden in Easy Mode)
        self.signal_processing_group = self.create_signal_processing_section()
        self.inference_group = self.create_inference_section()
        params_layout.addWidget(self.signal_processing_group)
        params_layout.addWidget(self.inference_group)

        # Advanced sections (hidden in Simple mode, require Advanced checkbox)
        self.calibration_group = self.create_calibration_section()
        self.advanced_group = self.create_advanced_section()

        params_layout.addWidget(self.calibration_group)
        params_layout.addWidget(self.advanced_group)

        # Initially hide non-essential sections (Easy Mode is ON by default)
        self.signal_processing_group.setVisible(False)
        self.inference_group.setVisible(False)
        self.calibration_group.setVisible(False)
        self.advanced_group.setVisible(False)

        params_layout.addStretch()

        # Run page: progress, phase indicators, controls, and logs.
        log_widget = QWidget()
        log_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        log_widget.setMinimumSize(0, 0)
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(10)
        self.run_page.content_layout.addWidget(log_widget)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        log_layout.addWidget(self.progress_bar)

        # Progress phase indicators
        phase_layout = QHBoxLayout()
        phase_layout.setSpacing(5)

        self.phase_indicators = []
        phase_names = [
            "Loading",
            "Preprocessing",
            "Inference",
            "Post-processing",
            "Saving",
        ]
        for i, name in enumerate(phase_names):
            phase_widget = QWidget()
            phase_inner = QHBoxLayout(phase_widget)
            phase_inner.setContentsMargins(0, 0, 0, 0)
            phase_inner.setSpacing(2)

            # Phase indicator circle
            indicator = QLabel("○")
            indicator.setStyleSheet("color: #555; font-size: 12px;")
            indicator.setFixedWidth(16)
            phase_inner.addWidget(indicator)

            # Phase name
            label = QLabel(name)
            label.setStyleSheet("color: #666; font-size: 11px;")
            phase_inner.addWidget(label)

            self.phase_indicators.append(
                {
                    "widget": phase_widget,
                    "indicator": indicator,
                    "label": label,
                    "name": name,
                }
            )
            phase_layout.addWidget(phase_widget)

            # Add arrow between phases (except last)
            if i < len(phase_names) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet("color: #444; font-size: 10px;")
                phase_layout.addWidget(arrow)

        phase_layout.addStretch()
        log_layout.addLayout(phase_layout)

        # Status and ETA layout
        status_eta_layout = QHBoxLayout()

        # Status label
        self.status_label = QLabel("Ready")
        status_eta_layout.addWidget(self.status_label)

        status_eta_layout.addStretch()

        # ETA label
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet("color: #888; font-style: italic;")
        status_eta_layout.addWidget(self.eta_label)

        log_layout.addLayout(status_eta_layout)

        # Log output
        log_group = QGroupBox("Log Output")
        log_group_layout = QVBoxLayout()
        log_group.setLayout(log_group_layout)

        log_toolbar = QHBoxLayout()
        log_toolbar.setContentsMargins(0, 0, 0, 0)
        self.copy_log_button = QPushButton("Copy Log")
        self.copy_log_button.setToolTip(
            "Copy the entire log to the clipboard for sharing in a bug report."
        )
        self.copy_log_button.clicked.connect(self._copy_log_to_clipboard)
        log_toolbar.addWidget(self.copy_log_button)

        self.save_log_button = QPushButton("Save Log…")
        self.save_log_button.setToolTip(
            "Save the log to a timestamped .txt file for archiving or sharing."
        )
        self.save_log_button.clicked.connect(self._save_log_to_file)
        log_toolbar.addWidget(self.save_log_button)

        self.clear_log_button = QPushButton("Clear")
        self.clear_log_button.setToolTip("Clear the log window (Ctrl+L).")
        self.clear_log_button.clicked.connect(self.clear_log)
        log_toolbar.addWidget(self.clear_log_button)

        log_toolbar.addStretch()
        log_group_layout.addLayout(log_toolbar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        self.log_text.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.log_text.setMinimumSize(0, 0)
        log_group_layout.addWidget(self.log_text)

        log_layout.addWidget(log_group)

        # Control buttons
        button_layout = self.run_page.actions_layout

        self.start_button = QPushButton("Start Inference")
        self.start_button.clicked.connect(self.start_inference)
        self.start_button.setProperty("role", "primary")
        self.start_button.setMinimumHeight(42)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_inference)
        self.stop_button.setEnabled(False)
        self.stop_button.setProperty("role", "danger")
        self.stop_button.setMinimumHeight(42)
        button_layout.addWidget(self.stop_button)

        button_layout.addStretch()

        # Review workspace: the whole-night hypnogram and specialist views.
        hypno_tab = QWidget()
        hypno_tab.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        hypno_tab.setMinimumSize(0, 0)
        hypno_layout = QVBoxLayout(hypno_tab)
        self.review_page.add_view(hypno_tab, "Hypnogram")
        if HypnogramCanvas is None:
            raise RuntimeError("HypnogramCanvas is not initialized")
        if NavigationToolbar is None:
            raise RuntimeError("NavigationToolbar is not initialized")

        # Hypnogram canvas
        self.hypnogram_canvas = HypnogramCanvas()
        hypno_layout.addWidget(self.hypnogram_canvas, 1)

        # Hypnogram toolbar
        toolbar = NavigationToolbar(self.hypnogram_canvas, hypno_tab)
        hypno_layout.addWidget(toolbar)

        # Hypnogram control panel
        control_panel = QFrame()
        control_panel.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 6px;
                padding: 8px;
                margin: 4px 0;
            }
        """)
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 6, 10, 6)

        # Sleep cycle markers toggle
        self.show_cycles_checkbox = QCheckBox("Show Sleep Cycles")
        self.show_cycles_checkbox.setToolTip(
            "Display NREM-REM sleep cycle markers on the hypnogram"
        )
        self.show_cycles_checkbox.setEnabled(False)
        self.show_cycles_checkbox.stateChanged.connect(self._toggle_sleep_cycles)
        control_layout.addWidget(self.show_cycles_checkbox)

        control_layout.addWidget(QLabel("|"))

        # Low confidence highlighting toggle
        self.highlight_low_conf_checkbox = QCheckBox("Highlight Low Confidence")
        self.highlight_low_conf_checkbox.setToolTip(
            "Highlight epochs with confidence below the threshold"
        )
        self.highlight_low_conf_checkbox.setEnabled(False)
        self.highlight_low_conf_checkbox.setChecked(True)
        self.highlight_low_conf_checkbox.stateChanged.connect(
            self._toggle_low_confidence_highlight
        )
        self.hypnogram_canvas.highlight_low_confidence = (
            self.highlight_low_conf_checkbox.isChecked()
        )
        control_layout.addWidget(self.highlight_low_conf_checkbox)

        # Confidence threshold slider
        control_layout.addWidget(QLabel("Threshold:"))
        self.conf_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_threshold_slider.setMinimum(int(MIN_CONFIDENCE_THRESHOLD * 100))
        self.conf_threshold_slider.setMaximum(int(MAX_CONFIDENCE_THRESHOLD * 100))
        self.conf_threshold_slider.setValue(int(DEFAULT_LOW_CONFIDENCE_THRESHOLD * 100))
        self.conf_threshold_slider.setFixedWidth(100)
        self.conf_threshold_slider.setToolTip(
            f"Confidence threshold for highlighting (currently {DEFAULT_LOW_CONFIDENCE_THRESHOLD:.0%})"
        )
        self.conf_threshold_slider.setEnabled(False)
        self.conf_threshold_slider.valueChanged.connect(
            self._update_confidence_threshold
        )
        control_layout.addWidget(self.conf_threshold_slider)

        self.conf_threshold_label = QLabel(f"{DEFAULT_LOW_CONFIDENCE_THRESHOLD:.0%}")
        self.conf_threshold_label.setFixedWidth(35)
        control_layout.addWidget(self.conf_threshold_label)

        control_layout.addWidget(QLabel("|"))

        # Compare mode button
        self.compare_mode_button = QPushButton("Load Reference...")
        self.compare_mode_button.setToolTip(
            "Load a reference hypnogram for comparison "
            "(CSV, NPY/NPZ, TXT, or Profusion XML file)"
        )
        self.compare_mode_button.setEnabled(False)
        self.compare_mode_button.clicked.connect(self._load_reference_hypnogram)
        control_layout.addWidget(self.compare_mode_button)

        self.clear_reference_button = QPushButton("Clear Reference")
        self.clear_reference_button.setToolTip(
            "Remove the reference hypnogram comparison"
        )
        self.clear_reference_button.setEnabled(False)
        self.clear_reference_button.setVisible(False)
        self.clear_reference_button.clicked.connect(self._clear_reference_hypnogram)
        control_layout.addWidget(self.clear_reference_button)

        control_layout.addStretch()

        # Click-to-inspect hint
        self.click_hint_label = QLabel("💡 Click on hypnogram to inspect epoch")
        self.click_hint_label.setStyleSheet("color: #888; font-style: italic;")
        control_layout.addWidget(self.click_hint_label)

        hypno_layout.addWidget(control_panel)

        # Connect epoch click callback
        self.hypnogram_canvas.set_epoch_click_callback(self._show_epoch_details)

        # Initialize epoch details dialog
        self.epoch_details_dialog = EpochDetailsDialog(self)

        self.export_png_button = QPushButton("Export as PNG")
        self.export_png_button.clicked.connect(lambda: self.export_hypnogram("png"))
        self.export_png_button.setEnabled(False)
        self.export_page.add_action(
            self.export_page.figure_actions, self.export_png_button
        )

        self.export_pdf_button = QPushButton("Export as PDF")
        self.export_pdf_button.clicked.connect(lambda: self.export_hypnogram("pdf"))
        self.export_pdf_button.setEnabled(False)
        self.export_page.add_action(
            self.export_page.figure_actions, self.export_pdf_button
        )

        self.export_svg_button = QPushButton("Export as SVG")
        self.export_svg_button.clicked.connect(lambda: self.export_hypnogram("svg"))
        self.export_svg_button.setEnabled(False)
        self.export_page.add_action(
            self.export_page.figure_actions, self.export_svg_button
        )

        # EDF+ Annotations export (clinical standard format)
        self.export_edfplus_button = QPushButton("Export EDF+ Annotations")
        self.export_edfplus_button.setToolTip(
            "Export sleep stages as EDF+ annotations (.edf)\n"
            "Standard clinical format compatible with most sleep software"
        )
        self.export_edfplus_button.clicked.connect(self.export_edf_annotations)
        self.export_edfplus_button.setEnabled(False)
        self.export_edfplus_button.setProperty("role", "primary")
        self.export_page.add_action(
            self.export_page.clinical_actions, self.export_edfplus_button
        )

        # Detailed epoch CSV export
        self.export_csv_button = QPushButton("Export Epoch CSV")
        self.export_csv_button.setToolTip(
            "Export detailed epoch-by-epoch data (.csv)\n"
            "Includes timestamps, stages, confidence scores, and flags"
        )
        self.export_csv_button.clicked.connect(self.export_detailed_csv)
        self.export_csv_button.setEnabled(False)
        self.export_page.add_action(
            self.export_page.clinical_actions, self.export_csv_button
        )

        # PDF Report button (styled differently)
        self.export_report_button = QPushButton("Generate PDF Report")
        self.export_report_button.setToolTip(
            "Generate a comprehensive PDF report with hypnogram and sleep statistics"
        )
        self.export_report_button.clicked.connect(self.generate_pdf_report)
        self.export_report_button.setEnabled(False)
        self.export_report_button.setProperty("role", "accent")
        self.export_page.add_action(
            self.export_page.report_actions, self.export_report_button
        )

        # Tab 3: Sleep Statistics
        self.statistics_widget = SleepStatisticsWidget()
        self.review_page.add_view(self.statistics_widget, "Sleep Statistics")

        # Tab 4: Confidence Review
        self.confidence_review_widget = ConfidenceReviewWidget()
        self.review_page.add_view(self.confidence_review_widget, "Confidence Review")

        # Tab 5: Signal Review
        self.signal_review_widget = SignalReviewWidget()
        self.review_page.add_view(self.signal_review_widget, "Signal Review")

        # Tab 6: Sleep Architecture Charts
        self.architecture_widget = SleepArchitectureWidget()
        self.review_page.add_view(self.architecture_widget, "Architecture")

        # Tab 7: Validation (model-vs-reference agreement + calibration)
        from spectra.review_gui import (
            CalibrationPanel,
            ReferenceAgreementPanel,
        )

        validation_tab = QWidget()
        validation_layout = QVBoxLayout(validation_tab)
        validation_splitter = QSplitter(Qt.Orientation.Vertical)
        self.reference_agreement_panel = ReferenceAgreementPanel()
        self.calibration_panel = CalibrationPanel()
        validation_splitter.addWidget(self.reference_agreement_panel)
        validation_splitter.addWidget(self.calibration_panel)
        validation_splitter.setStretchFactor(0, 1)
        validation_splitter.setStretchFactor(1, 1)
        validation_layout.addWidget(validation_splitter)
        self.review_page.add_view(validation_tab, "Validation")

        self.confidence_review_widget.epoch_selected.connect(self._select_epoch)
        self.signal_review_widget.epoch_selected.connect(self._select_epoch)
        self.signal_review_widget.manual_override_requested.connect(
            self.apply_manual_override
        )
        self.signal_review_widget.clear_override_requested.connect(
            self.clear_manual_override
        )
        self.signal_review_widget.clear_all_overrides_requested.connect(
            self.clear_all_manual_overrides
        )
        self.review_page.current_changed.connect(self._on_review_tab_changed)

        # Setup logging
        self.setup_logging()

    def _setup_status_bar(self):
        """Setup status bar with GPU memory and system info."""
        status_bar = self.statusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #1e1e1e;
                border-top: 1px solid #333;
            }
            QStatusBar::item {
                border: none;
            }
        """)

        # GPU Memory indicator
        self._gpu_label = QLabel("GPU: --")
        self._gpu_label.setStyleSheet("color: #888; padding: 0 10px;")
        status_bar.addPermanentWidget(self._gpu_label)

        # Separator
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #444;")
        status_bar.addPermanentWidget(sep1)

        # Processing speed indicator
        self._speed_label = QLabel("Speed: --")
        self._speed_label.setStyleSheet("color: #888; padding: 0 10px;")
        status_bar.addPermanentWidget(self._speed_label)

        # Separator
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #444;")
        status_bar.addPermanentWidget(sep2)

        # Accelerator/backend summary
        try:
            import torch

            resolved = self._resolve_selected_device()
            if resolved.resolved == "cuda" and torch.cuda.is_available():
                cuda_ver = torch.version.cuda
                device_name = torch.cuda.get_device_name(0)
                self._version_label = QLabel(f"CUDA {cuda_ver} • {device_name[:30]}")
            elif resolved.resolved == "mps":
                self._version_label = QLabel("MPS • Apple Silicon GPU")
            else:
                self._version_label = QLabel("CPU Mode")
        except Exception:
            self._version_label = QLabel("PyTorch not loaded")
        self._version_label.setStyleSheet("color: #666; padding: 0 10px;")
        status_bar.addPermanentWidget(self._version_label)

        # Timer for updating GPU stats
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status_bar)
        self._status_timer.start(2000)  # Update every 2 seconds

        # Track processing metrics
        self.run_state.last_progress_at = None
        self._last_progress_value = 0
        self._epochs_processed = 0

    def _update_status_bar(self):
        """Update status bar with current GPU memory and stats."""
        try:
            import torch

            resolved = self._resolve_selected_device()
            stats = _get_gui_device_memory_stats(resolved.torch_device)
            self._version_label.setToolTip(
                f"Requested: {resolved.requested} | Using: {resolved.description}"
            )

            if resolved.resolved == "cuda" and torch.cuda.is_available():
                cuda_ver = torch.version.cuda
                device_name = torch.cuda.get_device_name(0)
                self._version_label.setText(f"CUDA {cuda_ver} • {device_name[:30]}")
            elif resolved.resolved == "mps":
                self._version_label.setText("MPS • Apple Silicon GPU")
            else:
                self._version_label.setText("CPU Mode")

            if stats is not None and stats.get("device") in {"cuda", "mps"}:
                allocated_bytes = stats.get("allocated_bytes")
                total_bytes = stats.get("total_bytes")
                allocated_gb = (
                    float(allocated_bytes) / (1024**3)
                    if isinstance(allocated_bytes, int)
                    else None
                )
                total_gb = (
                    float(total_bytes) / (1024**3)
                    if isinstance(total_bytes, int)
                    else None
                )

                if allocated_gb is not None and total_gb is not None and total_gb > 0:
                    usage_pct = (allocated_gb / total_gb) * 100
                    if usage_pct > 80:
                        color = "#E53935"
                    elif usage_pct > 60:
                        color = "#FFA726"
                    else:
                        color = "#43A047"
                    self._gpu_label.setText(
                        f"GPU: {allocated_gb:.1f}/{total_gb:.1f} GB ({usage_pct:.0f}%)"
                    )
                    self._gpu_label.setStyleSheet(f"color: {color}; padding: 0 10px;")
                elif allocated_gb is not None:
                    self._gpu_label.setText(f"GPU: {allocated_gb:.1f} GB in use")
                    self._gpu_label.setStyleSheet("color: #43A047; padding: 0 10px;")
                else:
                    self._gpu_label.setText(
                        f"GPU: {_describe_gui_device(resolved.resolved)}"
                    )
                    self._gpu_label.setStyleSheet("color: #43A047; padding: 0 10px;")
            else:
                self._gpu_label.setText("GPU: N/A")
                self._gpu_label.setStyleSheet("color: #888; padding: 0 10px;")
        except Exception:
            self._gpu_label.setText("GPU: --")
            self._gpu_label.setStyleSheet("color: #888; padding: 0 10px;")

    def _update_processing_speed(self, epochs_per_sec: float):
        """Update the processing speed indicator."""
        if epochs_per_sec > 0:
            self._speed_label.setText(f"Speed: {epochs_per_sec:.1f} epochs/s")
            self._speed_label.setStyleSheet("color: #43A047; padding: 0 10px;")
        else:
            self._speed_label.setText("Speed: --")
            self._speed_label.setStyleSheet("color: #888; padding: 0 10px;")

    def create_files_section(self) -> QGroupBox:
        """Create file selection section."""
        group = QGroupBox("Input Files")
        layout = QGridLayout()
        group.setLayout(layout)

        # EDF file with recent files dropdown
        layout.addWidget(QLabel("EDF File:"), 0, 0)
        self.edf_path_edit = QLineEdit()
        self.edf_path_edit.setPlaceholderText(
            "Select EDF file (e.g., patient001.edf)..."
        )
        layout.addWidget(self.edf_path_edit, 0, 1)

        edf_btn_layout = QHBoxLayout()
        edf_btn_layout.setSpacing(2)
        edf_browse_btn = QPushButton("Browse...")
        edf_browse_btn.clicked.connect(self.browse_edf)
        edf_btn_layout.addWidget(edf_browse_btn)

        # Recent EDF files button with dropdown menu
        self.edf_recent_btn = QPushButton("▼")
        self.edf_recent_btn.setFixedWidth(30)
        self.edf_recent_btn.setToolTip("Recent EDF files")
        self.edf_recent_menu = QMenu(self)
        self.edf_recent_btn.setMenu(self.edf_recent_menu)
        self._update_recent_menu("edf")
        edf_btn_layout.addWidget(self.edf_recent_btn)

        edf_btn_widget = QWidget()
        edf_btn_widget.setLayout(edf_btn_layout)
        layout.addWidget(edf_btn_widget, 0, 2)

        # Checkpoint with recent files dropdown
        layout.addWidget(QLabel("Model Checkpoint:"), 1, 0)
        self.checkpoint_edit = QLineEdit()
        self.checkpoint_edit.setPlaceholderText(
            "Select model checkpoint (.pth, .pt, .ckpt)..."
        )
        self.checkpoint_edit.editingFinished.connect(self._validate_checkpoint_path)
        layout.addWidget(self.checkpoint_edit, 1, 1)

        ckpt_btn_layout = QHBoxLayout()
        ckpt_btn_layout.setSpacing(2)
        checkpoint_browse_btn = QPushButton("Browse...")
        checkpoint_browse_btn.clicked.connect(self.browse_checkpoint)
        ckpt_btn_layout.addWidget(checkpoint_browse_btn)

        self.checkpoint_recent_btn = QPushButton("▼")
        self.checkpoint_recent_btn.setFixedWidth(30)
        self.checkpoint_recent_btn.setToolTip("Recent checkpoints")
        self.checkpoint_recent_menu = QMenu(self)
        self.checkpoint_recent_btn.setMenu(self.checkpoint_recent_menu)
        self._update_recent_menu("checkpoint")
        ckpt_btn_layout.addWidget(self.checkpoint_recent_btn)

        ckpt_btn_widget = QWidget()
        ckpt_btn_widget.setLayout(ckpt_btn_layout)
        layout.addWidget(ckpt_btn_widget, 1, 2)

        # Model info display (shown after checkpoint is loaded)
        self.model_info_label = QLabel("")
        self.model_info_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 11px;
                padding: 2px 5px;
                background-color: #252525;
                border-radius: 3px;
            }
        """)
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setVisible(False)
        layout.addWidget(self.model_info_label, 2, 1, 1, 2)

        # Fixed 5-slot channel layout
        channel_layout_label = QLabel("Channel Layout:")
        channel_layout_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        layout.addWidget(channel_layout_label, 3, 0)

        channel_layout_widget = QWidget()
        channel_layout_outer = QVBoxLayout(channel_layout_widget)
        channel_layout_outer.setContentsMargins(0, 0, 0, 0)
        channel_layout_outer.setSpacing(6)

        channel_layout_help = QLabel(
            "Fixed inference order: EEG1, EEG2, EOG1, EOG2, EMG. "
            "Leave a slot on Auto to let inference choose or derive it."
        )
        channel_layout_help.setWordWrap(True)
        channel_layout_help.setStyleSheet("color: #888; font-size: 11px;")
        channel_layout_outer.addWidget(channel_layout_help)

        channel_layout_grid = QGridLayout()
        channel_layout_grid.setContentsMargins(0, 0, 0, 0)
        channel_layout_grid.setHorizontalSpacing(8)
        channel_layout_grid.setVerticalSpacing(4)

        self._channel_slot_combos = {}
        for row, (slot, modality) in enumerate(CHANNEL_LAYOUT_SLOTS):
            slot_label = QLabel(f"{slot} ({modality.upper()})")
            channel_layout_grid.addWidget(slot_label, row, 0)

            combo = QComboBox()
            combo.setToolTip(
                f"Select a {modality.upper()} channel for {slot}, or leave Auto."
            )
            combo.currentIndexChanged.connect(
                lambda _index, slot_name=slot: self._on_channel_slot_changed(slot_name)
            )
            self._channel_slot_combos[slot] = combo
            channel_layout_grid.addWidget(combo, row, 1)

        channel_layout_outer.addLayout(channel_layout_grid)

        channel_layout_btns = QHBoxLayout()
        channel_layout_btns.setContentsMargins(0, 0, 0, 0)
        channel_layout_btns.setSpacing(6)

        self.reload_channel_layout_btn = QPushButton("Reload From EDF")
        self.reload_channel_layout_btn.setToolTip(
            "Read the selected EDF header and refresh available EEG/EOG/EMG channels"
        )
        self.reload_channel_layout_btn.clicked.connect(
            self.reload_channel_layout_from_edf
        )
        channel_layout_btns.addWidget(self.reload_channel_layout_btn)

        self.reset_channel_layout_btn = QPushButton("Reset All To Auto")
        self.reset_channel_layout_btn.setToolTip(
            "Clear all manual channel assignments and use Auto for every slot"
        )
        self.reset_channel_layout_btn.clicked.connect(self.reset_channel_layout_to_auto)
        channel_layout_btns.addWidget(self.reset_channel_layout_btn)
        channel_layout_btns.addStretch()

        channel_layout_outer.addLayout(channel_layout_btns)
        layout.addWidget(channel_layout_widget, 3, 1, 1, 2)

        self._sync_channel_slot_comboboxes()

        # Output directory
        layout.addWidget(QLabel("Output Directory:"), 4, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText("./output")
        self.output_dir_edit.editingFinished.connect(self._validate_output_dir_path)
        layout.addWidget(self.output_dir_edit, 4, 1)

        output_btn_layout = QHBoxLayout()
        output_btn_layout.setSpacing(6)
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self.browse_output)
        output_btn_layout.addWidget(output_browse_btn)

        self.open_output_btn = QPushButton("Open")
        self.open_output_btn.setMinimumWidth(64)
        self.open_output_btn.setToolTip("Open output directory in file manager")
        self.open_output_btn.clicked.connect(self.open_output_directory)
        output_btn_layout.addWidget(self.open_output_btn)

        output_btn_widget = QWidget()
        output_btn_widget.setLayout(output_btn_layout)
        layout.addWidget(output_btn_widget, 4, 2)

        # Output dir validity indicator (writability check)
        self.output_dir_status_label = QLabel("")
        self.output_dir_status_label.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        self.output_dir_status_label.setVisible(False)
        layout.addWidget(self.output_dir_status_label, 5, 1, 1, 2)

        # Device selection (hidden in easy mode)
        self.device_label = QLabel("Device:")
        layout.addWidget(self.device_label, 6, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "mps", "cpu"])
        self.device_combo.setToolTip(
            "Device for inference: auto (prefer CUDA, then MPS, then CPU),\n"
            "cuda (force NVIDIA GPU), mps (force Apple Silicon GPU), cpu (force CPU)"
        )
        self.device_combo.currentTextChanged.connect(self._refresh_device_controls)
        self.device_combo.currentTextChanged.connect(
            lambda _text: self._update_status_bar()
        )
        layout.addWidget(self.device_combo, 6, 1, 1, 2)

        # Batch processing section
        batch_frame = QFrame()
        batch_frame.setFrameShape(QFrame.Shape.StyledPanel)
        batch_layout = QHBoxLayout(batch_frame)
        batch_layout.setContentsMargins(5, 5, 5, 5)

        self.batch_status_label = QLabel("Batch: No files queued")
        self.batch_status_label.setStyleSheet("color: #888;")
        batch_layout.addWidget(self.batch_status_label)

        batch_layout.addStretch()

        self.add_batch_btn = QPushButton("Add Files")
        self.add_batch_btn.setToolTip("Add EDF files to batch queue (Ctrl+B)")
        self.add_batch_btn.clicked.connect(self.add_to_batch_queue)
        batch_layout.addWidget(self.add_batch_btn)

        self.show_batch_btn = QPushButton("View Queue")
        self.show_batch_btn.setToolTip("View files in batch queue")
        self.show_batch_btn.clicked.connect(self.show_batch_queue)
        batch_layout.addWidget(self.show_batch_btn)

        self.start_batch_button = QPushButton("Start Batch")
        self.start_batch_button.setToolTip("Process all files in batch queue")
        self.start_batch_button.clicked.connect(self.start_batch_processing)
        self.start_batch_button.setEnabled(False)
        self.start_batch_button.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1E88E5; }
            QPushButton:disabled { background-color: #1e1e1e; color: #555; }
        """)
        batch_layout.addWidget(self.start_batch_button)

        self.pause_batch_button = QPushButton("Pause")
        self.pause_batch_button.setToolTip(
            "Pause / resume batch processing after the current file completes."
        )
        self.pause_batch_button.clicked.connect(self.toggle_batch_pause)
        self.pause_batch_button.setEnabled(False)
        batch_layout.addWidget(self.pause_batch_button)

        self.clear_batch_button = QPushButton("Clear")
        self.clear_batch_button.setToolTip("Clear batch queue")
        self.clear_batch_button.clicked.connect(self.clear_batch_queue)
        self.clear_batch_button.setEnabled(False)
        batch_layout.addWidget(self.clear_batch_button)

        layout.addWidget(batch_frame, 7, 0, 1, 3)

        for edit in (self.edf_path_edit, self.checkpoint_edit, self.output_dir_edit):
            edit.textChanged.connect(self._update_setup_readiness)

        return group

    def _update_setup_readiness(self) -> None:
        """Summarize required input readiness at the top of the Setup page."""
        edf_path = Path(self.edf_path_edit.text().strip()).expanduser()
        checkpoint_path = Path(self.checkpoint_edit.text().strip()).expanduser()
        output_text = self.output_dir_edit.text().strip()
        missing: list[str] = []
        if not self.edf_path_edit.text().strip():
            missing.append("EDF recording")
        elif not edf_path.is_file():
            missing.append("valid EDF recording")
        if not self.checkpoint_edit.text().strip():
            missing.append("model checkpoint")
        elif not checkpoint_path.is_file():
            missing.append("valid model checkpoint")
        if not output_text:
            missing.append("output directory")

        if missing:
            self.setup_page.set_readiness(
                "Still needed: " + ", ".join(missing) + ".",
                ready=False,
            )
            return
        self.setup_page.set_readiness(
            "Required inputs are present. Continue to Run when the channel mapping looks correct.",
            ready=True,
        )

    def open_output_directory(self):
        """Open the output directory in the system file manager."""
        output_dir = self.output_dir_edit.text()
        if not output_dir:
            output_dir = "./output"

        output_path = Path(output_dir)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(
                    self, "Error", f"Could not create output directory:\n{e}"
                )
                return

        import platform
        import subprocess

        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(output_path)], check=False)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(output_path)], check=False)
            else:  # Linux
                subprocess.run(["xdg-open", str(output_path)], check=False)
        except Exception as e:
            self.log(f"Could not open directory: {e}", logging.WARNING)

    def create_signal_processing_section(self) -> QGroupBox:
        """Create signal processing parameters section."""
        group = QGroupBox("Signal Processing")
        layout = QGridLayout()
        group.setLayout(layout)

        # Sampling frequency
        layout.addWidget(QLabel("Target Sampling Frequency (Hz):"), 0, 0)
        layout.addWidget(QLabel("128 Hz (fixed)"), 0, 1)

        # Epoch duration
        layout.addWidget(QLabel("Epoch Duration (seconds):"), 1, 0)
        self.epoch_sec_spin = QSpinBox()
        self.epoch_sec_spin.setRange(10, 60)
        self.epoch_sec_spin.setValue(30)
        self.epoch_sec_spin.setToolTip(
            "Length of one scoring epoch. AASM standard is 30 s — almost never change this."
        )
        layout.addWidget(self.epoch_sec_spin, 1, 1)

        # Context half-width
        layout.addWidget(QLabel("Context Half-Width (epochs):"), 2, 0)
        layout.addWidget(QLabel("10 epochs (fixed)"), 2, 1)

        # Analysis window (start inclusive, end exclusive). Only this interval is
        # normalized and scored, matching the annotated-span crop used for training.
        layout.addWidget(QLabel("Start Epoch:"), 3, 0)
        self.start_epoch_spin = QSpinBox()
        self.start_epoch_spin.setRange(0, 0)
        self.start_epoch_spin.setValue(0)
        self.start_epoch_spin.setToolTip(
            "First trustworthy epoch (0-based) of the analysis window. Include "
            "valid wake before sleep onset; exclude setup, calibration, or disconnected "
            "signal. Epochs before this are not normalized or scored."
        )
        layout.addWidget(self.start_epoch_spin, 3, 1)

        layout.addWidget(QLabel("End Epoch:"), 4, 0)
        self.end_epoch_spin = QSpinBox()
        self.end_epoch_spin.setRange(0, 0)
        self.end_epoch_spin.setValue(0)
        self.end_epoch_spin.setToolTip(
            "End of the trustworthy analysis window (exclusive). Include valid wake "
            "after final awakening; exclude teardown or disconnected signal. Epochs "
            "at or after this index are not normalized or scored."
        )
        layout.addWidget(self.end_epoch_spin, 4, 1)

        self.auto_signal_window_check = QCheckBox(
            "Auto-trim exterior flat/disconnected epochs"
        )
        self.auto_signal_window_check.setChecked(True)
        self.auto_signal_window_check.setToolTip(
            "Intersect the selected interval with the first and last epochs that "
            "contain at least one finite, non-flat channel. Manual bounds can still "
            "exclude non-flat setup or teardown artifact."
        )
        layout.addWidget(self.auto_signal_window_check, 5, 0, 1, 2)

        self.total_epochs_hint_label = QLabel("Total: — epochs")
        self.total_epochs_hint_label.setStyleSheet("color: gray; font-size: 10px;")
        self.total_epochs_hint_label.setToolTip(
            "Total complete epochs in the selected recording (≈ duration / epoch length)."
        )
        layout.addWidget(self.total_epochs_hint_label, 6, 0, 1, 2)

        layout.setColumnStretch(2, 1)
        return group

    def create_inference_section(self) -> QGroupBox:
        """Create inference settings section."""
        group = QGroupBox("Inference Settings")
        layout = QGridLayout()
        group.setLayout(layout)

        # Memory preset selector
        layout.addWidget(QLabel("Memory Preset:"), 0, 0)
        self.memory_preset_combo = QComboBox()
        self.memory_preset_combo.addItems(
            [
                "Default (High Memory)",
                "Balanced",
                "Memory-Constrained (<8 GB GPU)",
                "Extreme Savings (<4 GB)",
                "Custom",
            ]
        )
        self.memory_preset_combo.setCurrentText("Balanced")
        self.memory_preset_combo.currentTextChanged.connect(self.apply_memory_preset)
        self.memory_preset_combo.setToolTip(
            "Quick presets for memory optimization:\n"
            "• Default: Standard settings (32 batch, fp32)\n"
            "• Balanced: 50% memory reduction (16 batch, fp16)\n"
            "• Memory-Constrained: 75% reduction for <8GB GPU\n"
            "• Extreme Savings: 90% reduction for large files\n"
            "• Custom: Manual configuration"
        )
        layout.addWidget(self.memory_preset_combo, 0, 1)

        # Batch size
        layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(32)
        self.batch_size_spin.setToolTip(
            "How many epochs the GPU processes in parallel. Larger = faster but uses "
            "more VRAM. Typical: 32 on a 16 GB GPU, 16 on 8 GB, 8 on 6 GB. Halve this "
            "if you see CUDA out-of-memory errors."
        )
        self.batch_size_spin.valueChanged.connect(self.on_manual_parameter_change)
        layout.addWidget(self.batch_size_spin, 1, 1)

        # AMP mode
        layout.addWidget(QLabel("Precision Mode:"), 2, 0)
        self.amp_mode_combo = QComboBox()
        self.amp_mode_combo.addItems(["fp32", "fp16", "bf16"])
        self.amp_mode_combo.setToolTip(
            "Numerical precision used for inference:\n"
            "• fp32 — full precision; slowest and most VRAM-hungry, most reproducible.\n"
            "• fp16 — half precision; ~2× faster and ~50% less VRAM on most NVIDIA GPUs.\n"
            "• bf16 — half precision with wider dynamic range; best on RTX 30/40 and "
            "Apple Silicon. Recommended when supported."
        )
        self.amp_mode_combo.currentTextChanged.connect(self.on_manual_parameter_change)
        layout.addWidget(self.amp_mode_combo, 2, 1)

        layout.setColumnStretch(2, 1)
        return group

    def create_calibration_section(self) -> QGroupBox:
        """Create preprocessing calibration section."""
        group = QGroupBox("Preprocessing Calibration")
        layout = QGridLayout()
        group.setLayout(layout)

        info_label = QLabel(
            "Signals are median/IQR-normalized per recording (per channel, clipped to ±20) "
            "before inference — matching batch_edf_to_zarr_fp32.py. "
            "All bandpass/notch filtering must be handled offline."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #555555; font-size: 11px;")
        layout.addWidget(info_label, 0, 0, 1, 2)

        layout.setColumnStretch(1, 1)
        return group

    def create_advanced_section(self) -> QGroupBox:
        """Create advanced options section."""
        group = QGroupBox("Advanced Options")
        layout = QGridLayout()
        group.setLayout(layout)

        # Monte Carlo dropout
        self.mc_dropout_check = QCheckBox("Enable MC Dropout")
        self.mc_dropout_check.setToolTip(
            "Run the model multiple times with dropout active to estimate prediction "
            "uncertainty. Slows inference by ~Nx where N is the sample count. Useful "
            "for flagging epochs the model is unsure about — most users can leave off."
        )
        self.mc_dropout_check.stateChanged.connect(self._sync_advanced_option_widgets)
        layout.addWidget(self.mc_dropout_check, 0, 0, 1, 2)

        layout.addWidget(QLabel("MC Samples:"), 1, 0)
        self.mc_samples_spin = QSpinBox()
        self.mc_samples_spin.setRange(2, 50)
        self.mc_samples_spin.setValue(10)
        self.mc_samples_spin.setToolTip(
            "Number of stochastic forward passes used to estimate uncertainty. "
            "More samples = smoother estimate, but linearly longer runtime. Typical: 10."
        )
        layout.addWidget(self.mc_samples_spin, 1, 1)

        layout.addWidget(QLabel("MC Dropout Rate:"), 2, 0)
        self.mc_dropout_rate_spin = QDoubleSpinBox()
        self.mc_dropout_rate_spin.setRange(0.0, 0.5)
        self.mc_dropout_rate_spin.setSingleStep(0.05)
        self.mc_dropout_rate_spin.setValue(0.2)
        self.mc_dropout_rate_spin.setDecimals(2)
        self.mc_dropout_rate_spin.setToolTip(
            "Probability of dropping each unit during MC sampling. Typical: 0.1–0.3. "
            "Set to 0.0 to use the dropout rate the model was trained with."
        )
        layout.addWidget(self.mc_dropout_rate_spin, 2, 1)

        layout.addWidget(QLabel("MC Pooling:"), 3, 0)
        self.mc_pooling_combo = QComboBox()
        self.mc_pooling_combo.addItems(["prob", "logit"])
        self.mc_pooling_combo.setToolTip(
            "How the MC samples are combined.\n"
            "prob (default): average the softmax posteriors — the MC-dropout\n"
            "  predictive distribution, and the pooling that actually smooths.\n"
            "logit: average raw logits — the geometric mean of posteriors, which\n"
            "  largely reproduces the deterministic pass. Use only to reproduce\n"
            "  runs scored before this option existed."
        )
        layout.addWidget(self.mc_pooling_combo, 3, 1)

        self.mc_attention_check = QCheckBox("Sample attention dropout")
        self.mc_attention_check.setChecked(True)
        self.mc_attention_check.setToolTip(
            "Also sample nn.MultiheadAttention dropout. MHA applies dropout off\n"
            "its own training flag rather than through a child nn.Dropout, so\n"
            "leaving this off keeps a large share of the model's trained\n"
            "stochasticity out of the MC ensemble."
        )
        layout.addWidget(self.mc_attention_check, 4, 0, 1, 2)

        layout.setColumnStretch(2, 1)
        self._sync_advanced_option_widgets()
        return group

    def _sync_advanced_option_widgets(self):
        """Enable advanced widgets only when their parent option is active."""
        mc_enabled = self.mc_dropout_check.isChecked()
        self.mc_samples_spin.setEnabled(mc_enabled)
        self.mc_dropout_rate_spin.setEnabled(mc_enabled)
        self.mc_pooling_combo.setEnabled(mc_enabled)
        self.mc_attention_check.setEnabled(mc_enabled)

    def setup_logging(self):
        """Setup logging to display in GUI."""
        # Use a dedicated logger so we don't attach GUI handlers to the root logger.
        logger = logging.getLogger("psgstage.inference_gui")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Remove any stale GUI handlers (e.g., after hot-reload / re-init).
        for h in list(logger.handlers):
            if isinstance(h, QTextEditLogger):
                logger.removeHandler(h)

        handler = QTextEditLogger(self.log_text)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

        self._logger = logger
        self._qt_log_handler = handler

    def apply_memory_preset(self, preset_name: str):
        """Apply memory optimization preset.

        Args:
            preset_name: Name of preset to apply
        """
        # Block signals to prevent triggering "Custom" mode during preset application.
        # context_half is locked at 10 project-wide, so presets never touch it.
        self.batch_size_spin.blockSignals(True)
        self.amp_mode_combo.blockSignals(True)

        try:
            if preset_name == "Default (High Memory)":
                self.batch_size_spin.setValue(32)
                self.amp_mode_combo.setCurrentText("fp32")
                self.log("Applied Default preset: 32 batch, fp32", logging.INFO)

            elif preset_name == "Balanced":
                self.batch_size_spin.setValue(16)
                self.amp_mode_combo.setCurrentText("fp16")
                self.log(
                    "Applied Balanced preset: 16 batch, fp16 (~50% memory)",
                    logging.INFO,
                )

            elif preset_name == "Memory-Constrained (<8 GB GPU)":
                self.batch_size_spin.setValue(8)
                self.amp_mode_combo.setCurrentText("fp16")
                self.log(
                    "Applied Memory-Constrained preset: 8 batch, fp16 (~75% memory reduction)",
                    logging.INFO,
                )

            elif preset_name == "Extreme Savings (<4 GB)":
                self.batch_size_spin.setValue(4)
                self.amp_mode_combo.setCurrentText("fp16")
                self.device_combo.setCurrentText("cpu")
                self.log(
                    "Applied Extreme Savings preset: 4 batch, fp16, CPU (~90% memory reduction)",
                    logging.INFO,
                )

            # Custom mode - do nothing, user is manually configuring

        finally:
            # Re-enable signals
            self.batch_size_spin.blockSignals(False)
            self.amp_mode_combo.blockSignals(False)

    def on_manual_parameter_change(self):
        """Called when user manually changes a parameter - switch to Custom mode."""
        # Don't switch to Custom if we're currently applying a preset
        if not self.batch_size_spin.signalsBlocked():
            self.memory_preset_combo.blockSignals(True)
            self.memory_preset_combo.setCurrentText("Custom")
            self.memory_preset_combo.blockSignals(False)

    def browse_edf(self):
        """Browse for EDF file."""
        # Use last directory if available
        start_dir = ""
        if self._recent_edf_files:
            start_dir = str(Path(self._recent_edf_files[0]).parent)

        filename, _ = QFileDialog.getOpenFileName(
            self, "Select EDF File", start_dir, "EDF Files (*.edf);;All Files (*)"
        )
        if filename:
            self.edf_path_edit.setText(filename)
            self.add_to_recent_files(filename, "edf")
            self._refresh_channel_layout_from_edf(log_missing=True)
            self._offer_override_sidecar_restore()

    def browse_checkpoint(self):
        """Browse for checkpoint file."""
        start_dir = ""
        if self._recent_checkpoint_files:
            start_dir = str(Path(self._recent_checkpoint_files[0]).parent)

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Checkpoint",
            start_dir,
            "PyTorch Checkpoints (*.pth *.pt *.ckpt);;All Files (*)",
        )
        if filename:
            self.checkpoint_edit.setText(filename)
            self.add_to_recent_files(filename, "checkpoint")
            # Auto-detect and display model info
            self._load_checkpoint_info(filename)

    def _load_checkpoint_info(self, checkpoint_path: str):
        """Load and display model information from checkpoint file."""
        try:
            import torch

            # Load checkpoint with weights_only=False to allow config loading
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

            info_parts = []

            # Extract model architecture
            if "config" in checkpoint:
                config = checkpoint["config"]
                if hasattr(config, "model"):
                    model_type = getattr(config, "model", "unknown")
                    info_parts.append(f"<b>Model:</b> {model_type}")

                # Context size
                if hasattr(config, "context_half"):
                    ctx = getattr(config, "context_half", 10)
                    info_parts.append(f"<b>Context:</b> {ctx * 2 + 1} epochs")

                # Task type
                if hasattr(config, "task"):
                    task = getattr(config, "task", "five")
                    task_display = {
                        "five": "5-class",
                        "wake_sleep": "2-class (Wake/Sleep)",
                    }.get(task, task)
                    info_parts.append(f"<b>Task:</b> {task_display}")

                # Channels
                if hasattr(config, "n_channels"):
                    info_parts.append(f"<b>Channels:</b> {config.n_channels}")

            # Extract training epoch
            if "epoch" in checkpoint:
                info_parts.append(f"<b>Trained:</b> {checkpoint['epoch']} epochs")

            # Get model state dict size for architecture inference
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                n_params = sum(
                    p.numel()
                    for p in state_dict.values()
                    if isinstance(p, torch.Tensor)
                )
                if n_params > 1e6:
                    info_parts.append(f"<b>Params:</b> {n_params / 1e6:.1f}M")
                else:
                    info_parts.append(f"<b>Params:</b> {n_params / 1e3:.1f}K")

                # Detect model type from state dict keys
                keys = list(state_dict.keys())
                if any("transformer" in k.lower() for k in keys):
                    if "Model:" not in str(info_parts):
                        info_parts.insert(0, "<b>Model:</b> Transformer")
                elif any("lstm" in k.lower() or "rnn" in k.lower() for k in keys):
                    if "Model:" not in str(info_parts):
                        info_parts.insert(0, "<b>Model:</b> LSTM/RNN")
                elif any("tcn" in k.lower() or "conv" in k.lower() for k in keys):
                    if "Model:" not in str(info_parts):
                        info_parts.insert(0, "<b>Model:</b> TCN/Conv")

            # Check for MoE components
            if "model_state_dict" in checkpoint:
                if any(
                    "expert" in k.lower() or "moe" in k.lower()
                    for k in checkpoint["model_state_dict"].keys()
                ):
                    info_parts.append("<b>MoE:</b> Yes")

            if info_parts:
                self.model_info_label.setText(" • ".join(info_parts))
                self.model_info_label.setVisible(True)
                self.model_info_label.setStyleSheet("""
                    QLabel {
                        color: #43A047;
                        font-size: 11px;
                        padding: 4px 8px;
                        background-color: #1a2e1a;
                        border: 1px solid #2E7D32;
                        border-radius: 4px;
                    }
                """)
                self.log(
                    f"Loaded checkpoint info: {', '.join(p.replace('<b>', '').replace('</b>', '') for p in info_parts)}",
                    logging.INFO,
                )
            else:
                self.model_info_label.setText(
                    "Checkpoint loaded (no config info available)"
                )
                self.model_info_label.setVisible(True)

            # Clean up
            del checkpoint

        except Exception as e:
            self.model_info_label.setText(f"⚠ Could not read checkpoint: {str(e)[:50]}")
            self.model_info_label.setVisible(True)
            self.model_info_label.setStyleSheet("""
                QLabel {
                    color: #FFA726;
                    font-size: 11px;
                    padding: 4px 8px;
                    background-color: #2e2a1a;
                    border: 1px solid #FF9800;
                    border-radius: 4px;
                }
            """)
            self.log(f"Warning: Could not read checkpoint info: {e}", logging.WARNING)

    def _validate_checkpoint_path(self) -> None:
        """Re-run checkpoint introspection when the user types/edits the path."""
        path = self.checkpoint_edit.text().strip()
        if not path:
            self.model_info_label.setVisible(False)
            return
        if not Path(path).exists():
            self.model_info_label.setText(
                f"⚠ Checkpoint file not found: {Path(path).name}"
            )
            self.model_info_label.setVisible(True)
            self.model_info_label.setStyleSheet("""
                QLabel {
                    color: #ef5350;
                    font-size: 11px;
                    padding: 4px 8px;
                    background-color: #2e1a1a;
                    border: 1px solid #c62828;
                    border-radius: 4px;
                }
            """)
            return
        self._load_checkpoint_info(path)

    def _validate_output_dir_path(self) -> None:
        """Check that the configured output directory can receive files."""
        path = self.output_dir_edit.text().strip()
        if not path:
            self.output_dir_status_label.setVisible(False)
            return
        target = Path(path).expanduser()
        if target.exists():
            if not target.is_dir():
                self._set_output_dir_status(
                    "⚠ Path exists but is not a directory.", ok=False
                )
                return
            writable = os.access(target, os.W_OK)
            if writable:
                self._set_output_dir_status(
                    f"✓ Writable. Exports will be saved to {target}.", ok=True
                )
            else:
                self._set_output_dir_status(
                    f"⚠ Not writable: {target}. Choose another directory.",
                    ok=False,
                )
            return
        # Parent must exist + be writable so we can create the directory at run time.
        parent = target.parent if target.parent != target else target
        if not parent.exists():
            self._set_output_dir_status(
                f"⚠ Parent directory does not exist: {parent}.", ok=False
            )
            return
        if not os.access(parent, os.W_OK):
            self._set_output_dir_status(
                f"⚠ Cannot create here — parent is not writable: {parent}.",
                ok=False,
            )
            return
        self._set_output_dir_status(
            f"ℹ Will be created at run time: {target}.", ok=True
        )

    def _set_output_dir_status(self, message: str, *, ok: bool) -> None:
        """Update the inline output-directory status label."""
        self.output_dir_status_label.setText(message)
        self.output_dir_status_label.setVisible(True)
        colour = "#43A047" if ok else "#ef5350"
        bg = "#1a2e1a" if ok else "#2e1a1a"
        border = "#2E7D32" if ok else "#c62828"
        self.output_dir_status_label.setStyleSheet(
            f"QLabel {{ color: {colour}; font-size: 11px; padding: 2px 6px; "
            f"background-color: {bg}; border: 1px solid {border}; border-radius: 3px; }}"
        )

    def _load_legacy_channel_slot_overrides(
        self,
        canon_json_path: str,
        *,
        source_label: str,
    ) -> dict[str, str | None] | None:
        """Migrate a legacy canonical_channels.json path into slot overrides."""
        canon_json_path = canon_json_path.strip()
        if not canon_json_path:
            return None

        if not Path(canon_json_path).exists():
            self.log(
                f"Ignoring legacy {source_label} canonical channels file because it no longer exists: {canon_json_path}",
                logging.WARNING,
            )
            return None

        try:
            canonical_channels = _load_canonical_channels_for_gui(canon_json_path)
        except Exception as exc:
            self.log(
                f"Ignoring legacy {source_label} canonical channels file '{canon_json_path}': {exc}",
                logging.WARNING,
            )
            return None

        migrated = _legacy_canonical_channels_to_slot_overrides(canonical_channels)
        if migrated is None:
            self.log(
                f"Ignoring legacy {source_label} canonical channels file '{canon_json_path}' because it is not a 5-channel EEG/EEG/EOG/EOG/EMG layout.",
                logging.WARNING,
            )
            return None

        self.log(
            f"Migrated legacy {source_label} channel layout from {Path(canon_json_path).name}",
            logging.INFO,
        )
        return migrated

    def _sync_channel_slot_comboboxes(self):
        """Populate the fixed slot combo boxes from available EDF channels."""
        from spectra.data.channel.normalization import infer_channel_type

        self._channel_slot_signal_block = True
        try:
            for slot, modality in CHANNEL_LAYOUT_SLOTS:
                combo = self._channel_slot_combos.get(slot)
                if combo is None:
                    continue

                current_override = self._channel_slot_overrides.get(slot)
                combo.blockSignals(True)
                combo.clear()
                combo.addItem("Auto", None)

                for label in self._available_edf_channel_labels:
                    if infer_channel_type(label) == modality:
                        combo.addItem(label, label)

                if current_override:
                    index = combo.findData(current_override)
                    combo.setCurrentIndex(index if index >= 0 else 0)
                else:
                    combo.setCurrentIndex(0)
                combo.blockSignals(False)
        finally:
            self._channel_slot_signal_block = False

    def _on_channel_slot_changed(self, slot_name: str):
        """Update slot override state when a combo box changes."""
        if self._channel_slot_signal_block:
            return

        combo = self._channel_slot_combos.get(slot_name)
        if combo is None:
            return

        selected_label = combo.currentData()
        self._channel_slot_overrides[slot_name] = (
            str(selected_label) if selected_label else None
        )
        self._mark_project_modified()

    def reset_channel_layout_to_auto(self):
        """Reset all channel layout slots to Auto."""
        self._channel_slot_overrides = _default_channel_slot_overrides()
        self._sync_channel_slot_comboboxes()
        self._mark_project_modified()
        self.log("Reset channel layout to Auto for all 5 slots", logging.INFO)

    def reload_channel_layout_from_edf(self):
        """Reload channel options from the selected EDF header."""
        signal_labels = self._refresh_channel_layout_from_edf(
            show_dialog_errors=True,
            log_missing=True,
        )
        if signal_labels is not None:
            self.log(
                f"Loaded {len(signal_labels)} EDF channels into the 5-slot mapper",
                logging.INFO,
            )

    def _update_epoch_window_bounds(
        self,
        sample_rates: list[float] | None,
        n_samples: list[int] | None,
    ) -> None:
        """Update the start/end epoch spin boxes from EDF header info.

        Computes the number of complete epochs from the recording duration and the
        configured epoch length, then sets the spin-box ranges and the total-epochs
        hint label. When called with ``None`` (no/invalid EDF), the bounds reset.

        Args:
            sample_rates: Per-channel sampling rates from the EDF header, or ``None``.
            n_samples: Per-channel sample counts from the EDF header, or ``None``.
        """
        total_epochs = 0
        if sample_rates and n_samples:
            epoch_sec = max(1, self.epoch_sec_spin.value())
            durations = [
                n / rate
                for n, rate in zip(n_samples, sample_rates, strict=False)
                if rate and rate > 0
            ]
            if durations:
                total_epochs = int(min(durations) // epoch_sec)

        if total_epochs <= 0:
            self.start_epoch_spin.setRange(0, 0)
            self.start_epoch_spin.setValue(0)
            self.end_epoch_spin.setRange(0, 0)
            self.end_epoch_spin.setValue(0)
            self.total_epochs_hint_label.setText("Total: — epochs")
            return

        hours = total_epochs * max(1, self.epoch_sec_spin.value()) / 3600.0
        self.start_epoch_spin.setRange(0, max(0, total_epochs - 1))
        self.start_epoch_spin.setValue(0)
        self.end_epoch_spin.setRange(1, total_epochs)
        self.end_epoch_spin.setValue(total_epochs)
        self.total_epochs_hint_label.setText(
            f"Total: {total_epochs} epochs / {hours:.1f} h"
        )

    def _refresh_channel_layout_from_edf(
        self,
        *,
        edf_path: str | None = None,
        show_dialog_errors: bool = False,
        log_missing: bool = False,
    ) -> list[str] | None:
        """Refresh available EDF channels and reapply any persisted slot overrides."""
        path = (edf_path if edf_path is not None else self.edf_path_edit.text()).strip()
        if not path:
            self._available_edf_channel_labels = []
            self._update_epoch_window_bounds(None, None)
            self._sync_channel_slot_comboboxes()
            return None

        if not Path(path).exists():
            self._available_edf_channel_labels = []
            self._update_epoch_window_bounds(None, None)
            self._sync_channel_slot_comboboxes()
            if show_dialog_errors:
                QMessageBox.warning(
                    self, "File Not Found", f"EDF file not found:\n{path}"
                )
            return None

        try:
            _, signal_labels, sample_rates, n_samples, repaired = (
                _read_edf_header_with_repair(path)
            )
        except ImportError:
            if show_dialog_errors:
                QMessageBox.warning(
                    self,
                    "Missing Dependency",
                    "pyedflib is required to inspect EDF channels.\nInstall with: pip install pyedflib",
                )
            return None
        except Exception as exc:
            if show_dialog_errors:
                QMessageBox.warning(
                    self,
                    "EDF Read Error",
                    f"Could not read the EDF header:\n\n{str(exc)[:300]}",
                )
            else:
                self.log(f"Failed to refresh EDF channels: {exc}", logging.WARNING)
            return None

        self._available_edf_channel_labels = list(signal_labels)
        self._update_epoch_window_bounds(sample_rates, n_samples)
        normalized_overrides, warnings = _normalize_channel_slot_overrides(
            self._channel_slot_overrides,
            available_labels=self._available_edf_channel_labels,
        )
        self._channel_slot_overrides = normalized_overrides
        self._sync_channel_slot_comboboxes()

        if repaired and log_missing:
            self.log(
                "EDF header contained invalid Physical Dimension fields and was repaired for channel inspection.",
                logging.INFO,
            )
        if log_missing:
            for warning in warnings:
                self.log(warning, logging.WARNING)

        return list(signal_labels)

    def _get_effective_channel_layout(self) -> list[str]:
        """Return the ordered runtime layout for inference."""
        return _build_effective_channel_layout(self._channel_slot_overrides)

    def _validate_channel_layout(self) -> bool:
        """Validate explicit slot selections for the currently selected EDF."""
        duplicates = _find_duplicate_channel_slot_overrides(
            self._channel_slot_overrides
        )
        if duplicates:
            duplicate_lines = [
                f"• {label} is assigned to {', '.join(slots)}"
                for label, slots in sorted(duplicates.items())
            ]
            QMessageBox.warning(
                self,
                "Duplicate Channel Assignments",
                "Each manually selected EDF channel can only be used once.\n\n"
                + "\n".join(duplicate_lines),
            )
            return False
        return True

    def _write_temp_channel_layout_json(self, channel_layout: list[str]) -> str:
        """Write the effective 5-slot layout to a temporary JSON file."""
        fd, temp_path = tempfile.mkstemp(
            prefix="psgstage_inference_channels_",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(channel_layout, handle, indent=2)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            Path(temp_path).unlink(missing_ok=True)
            raise
        return temp_path

    def browse_output(self):
        """Browse for output directory."""
        dirname = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if dirname:
            self.output_dir_edit.setText(dirname)
            self._validate_output_dir_path()

    def validate_inputs(self) -> bool:
        """Validate user inputs before starting inference with actionable error messages."""
        errors = []
        suggestions = []

        # Check required files
        if not self.edf_path_edit.text():
            errors.append("• EDF file not selected")
            suggestions.append("Click 'Browse...' next to EDF File or use Ctrl+O")
        elif not Path(self.edf_path_edit.text()).exists():
            errors.append(
                f"• EDF file not found: {Path(self.edf_path_edit.text()).name}"
            )
            suggestions.append("Check the file path or select a different file")

        if not self.checkpoint_edit.text():
            errors.append("• Model checkpoint not selected")
            suggestions.append(
                "Click 'Browse...' next to Model Checkpoint or use Ctrl+M"
            )
        elif not Path(self.checkpoint_edit.text()).exists():
            errors.append(
                f"• Checkpoint file not found: {Path(self.checkpoint_edit.text()).name}"
            )
            suggestions.append("Verify the model file exists (.pth, .pt, or .ckpt)")

        # Check output directory is specified
        if not self.output_dir_edit.text() or self.output_dir_edit.text().strip() == "":
            errors.append("• Output directory not specified")
            suggestions.append("Enter an output path (default: ./output)")

        # Check the analysis window is ordered (end of 0 means "to end of recording")
        start_epoch = self.start_epoch_spin.value()
        end_epoch = self.end_epoch_spin.value()
        if end_epoch > 0 and end_epoch <= start_epoch:
            errors.append(
                f"• Analysis window is empty: start epoch ({start_epoch}) must be "
                f"below end epoch ({end_epoch})"
            )
            suggestions.append(
                "Set Start Epoch below End Epoch, or set End Epoch to the total epoch "
                "count to score to the end of the recording"
            )

        # If there are errors, show a comprehensive message
        if errors:
            error_text = "The following issues need to be resolved:\n\n"
            error_text += "\n".join(errors)
            error_text += "\n\n" + "─" * 40 + "\n\n"
            error_text += "How to fix:\n"
            error_text += "\n".join(f"  → {s}" for s in suggestions)

            self._show_error_dialog(
                title="Cannot Start Inference",
                summary="Some required inputs are missing or invalid.",
                detail=error_text,
                suggestion="Resolve the issues listed in the details and try again.",
                icon=QMessageBox.Icon.Warning,
            )
            return False

        # Additional validation: try to read the EDF header to check file validity
        edf_path = self.edf_path_edit.text()
        try:
            n_signals, signal_labels, _, _, repaired = _read_edf_header_with_repair(
                edf_path
            )
            if n_signals == 0:
                QMessageBox.warning(
                    self,
                    "Invalid EDF File",
                    f"The EDF file appears to have no signals.\n\n"
                    f"File: {Path(edf_path).name}\n\n"
                    "Please select a valid polysomnography EDF file.",
                )
                return False
            if repaired:
                self.log(
                    "EDF header contained invalid Physical Dimension fields and was "
                    "repaired with the same fallback used by batch EDF->Zarr conversion.",
                    logging.INFO,
                )
            self._available_edf_channel_labels = list(signal_labels)
            normalized_overrides, warnings = _normalize_channel_slot_overrides(
                self._channel_slot_overrides,
                available_labels=self._available_edf_channel_labels,
            )
            self._channel_slot_overrides = normalized_overrides
            self._sync_channel_slot_comboboxes()
            for warning in warnings:
                self.log(warning, logging.WARNING)
        except ImportError:
            pass  # pyedflib not available, skip validation
        except Exception as e:
            QMessageBox.warning(
                self,
                "EDF Read Error",
                f"Could not read the EDF file:\n\n{str(e)[:200]}\n\n"
                f"File: {Path(edf_path).name}\n\n"
                "The file may be corrupted or in an unsupported format.",
            )
            return False

        if not self._validate_channel_layout():
            return False

        return True

    def get_score_options(self) -> ScoreOptions:
        """Get ScoreOptions from GUI inputs."""
        from spectra.inference import ScoreOptions

        mc_dropout_rate = self.mc_dropout_rate_spin.value()
        if mc_dropout_rate == 0.0:
            mc_dropout_rate = None

        # End epoch of 0 means "no window set" (e.g. no EDF loaded yet) -> score to end.
        end_epoch = self.end_epoch_spin.value()
        if end_epoch <= 0:
            end_epoch = -1

        return ScoreOptions(
            # Signal processing
            epoch_sec=self.epoch_sec_spin.value(),
            # Analysis window (start inclusive, end exclusive)
            start_epoch=self.start_epoch_spin.value(),
            end_epoch=end_epoch,
            auto_signal_window=self.auto_signal_window_check.isChecked(),
            # Inference
            amp_mode=self.amp_mode_combo.currentText(),
            batch_size=self.batch_size_spin.value(),
            # Calibration
            calibration_mode="per_recording",
            # MC Dropout
            use_mc_dropout=self.mc_dropout_check.isChecked(),
            mc_samples=self.mc_samples_spin.value(),
            mc_dropout_rate=mc_dropout_rate,
            mc_pooling=self.mc_pooling_combo.currentText(),
            mc_include_attention=self.mc_attention_check.isChecked(),
        )

    def start_inference(self):
        """Start inference in background thread."""
        if not self.validate_inputs():
            return

        # CRITICAL: Clean up previous worker if it exists to prevent memory leaks
        if self.worker is not None:
            if self.worker.isRunning():
                self.log(
                    "Inference is already running; wait for it to finish or cancel it.",
                    logging.WARNING,
                )
                self.tab_widget.setCurrentWidget(self.run_page)
                return

            # A finished signal can be queued briefly behind a second Start action.
            self.worker.deleteLater()
            self.worker = None

        # AGGRESSIVE cleanup before starting new inference
        import gc

        # Multiple GC passes to ensure everything is collected
        for _ in range(3):
            gc.collect()

        # Disable start button, enable stop button
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Clear previous log and reset progress indicators
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting inference...")
        self.eta_label.setText("")
        self.reset_phase_indicators()
        self.set_phase("Loading", "active")
        self.run_state.begin(time.time())
        self.tab_widget.setCurrentWidget(self.run_page)
        self.run_page.set_activity("Inference is starting…", active=True)

        device_info = self._resolve_selected_device()
        if device_info.warning:
            self.log(device_info.warning, logging.WARNING)
        self._clear_selected_device_cache(
            success_message=f"Prepared {device_info.description} for inference",
            warning_prefix="Could not clear accelerator cache before inference",
            synchronize=True,
            reset_peak_memory_stats=True,
        )

        channel_layout = self._get_effective_channel_layout()
        temp_channel_layout_path = self._write_temp_channel_layout_json(channel_layout)
        self.log(
            "Using fixed 5-slot channel layout: "
            + ", ".join(
                f"{slot}={value}"
                for slot, value in zip(
                    CHANNEL_LAYOUT_SLOT_NAMES, channel_layout, strict=True
                )
            ),
            logging.INFO,
        )

        # Create worker
        try:
            self.worker = InferenceWorker(
                edf_path=self.edf_path_edit.text(),
                checkpoint=self.checkpoint_edit.text(),
                canon_json=temp_channel_layout_path,
                output_dir=self.output_dir_edit.text(),
                device=self.device_combo.currentText(),
                options=self.get_score_options(),
                cleanup_paths=[temp_channel_layout_path],
            )
        except Exception:
            Path(temp_channel_layout_path).unlink(missing_ok=True)
            raise

        # Connect signals
        self.worker.progress.connect(self.on_progress)
        self.worker.result_ready.connect(self.on_finished)
        self.worker.failed.connect(self.on_error)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.finished.connect(self._on_inference_thread_finished)

        # Start worker
        self.worker.start()

    def stop_inference(self):
        """Request cancellation without blocking the Qt event loop."""
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Cancelling inference…")
            self.log(
                "Cancellation requested; finishing the active operation",
                logging.WARNING,
            )
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def on_cancelled(self):
        """Restore the controls after cooperative worker cancellation."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Inference cancelled")
        self.eta_label.setText("")
        self.run_state.finish()
        self.run_page.set_activity("Inference was cancelled safely.", active=False)
        if self._batch_mode:
            self._batch_mode = False
            if 0 <= self._batch_current_index < len(self._batch_queue):
                self._batch_queue[self._batch_current_index]["status"] = "cancelled"
                self._refresh_batch_queue_dialog()

    def _on_inference_thread_finished(self) -> None:
        """Release a worker only after its native QThread has actually exited."""
        worker = self.sender()
        current_worker = self.worker
        if current_worker is not None and worker is current_worker:
            current_worker.deleteLater()
            self.worker = None
            import gc

            gc.collect()
            self._clear_selected_device_cache(
                success_message="✓ Cleaned up accelerator memory after inference",
                warning_prefix="Could not clear accelerator cache after inference",
                synchronize=True,
            )
        self._try_finish_deferred_close()

    def on_progress(self, message: str, percent: int):
        """Handle progress update."""
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)
        self.run_page.set_activity(message, active=True)

        # Update phase indicators based on progress
        self.update_phase_from_progress(percent)

        # Forward to the live batch queue dialog when applicable.
        if self._batch_mode and 0 <= self._batch_current_index < len(self._batch_queue):
            self._batch_queue[self._batch_current_index]["percent"] = int(percent)
            self._refresh_batch_queue_dialog()

        # Calculate and display ETA
        if percent > 0 and self.run_state.started_at is not None:
            elapsed = time.time() - self.run_state.started_at
            if percent < 100:
                # Estimate remaining time based on progress
                estimated_total = elapsed / (percent / 100.0)
                remaining = estimated_total - elapsed
                self.eta_label.setText(self.format_eta(remaining))
            else:
                self.eta_label.setText(
                    f"Completed in {self.format_eta(elapsed).replace(' remaining', '')}"
                )

        # Calculate processing speed (epochs per second)
        # Parse epoch count from message if available (e.g., "Processing epoch 100/1000")
        try:
            import re

            match = re.search(r"epoch\s+(\d+)/(\d+)", message, re.IGNORECASE)
            if match:
                current_epoch = int(match.group(1))
                now = time.time()

                # Calculate speed if we have previous data
                if (
                    self.run_state.last_progress_at is not None
                    and self.run_state.last_epoch_count is not None
                ):
                    time_diff = now - self.run_state.last_progress_at
                    epoch_diff = current_epoch - self.run_state.last_epoch_count
                    if time_diff > 0 and epoch_diff > 0:
                        epochs_per_sec = epoch_diff / time_diff
                        self._update_processing_speed(epochs_per_sec)

                self.run_state.last_progress_at = now
                self.run_state.last_epoch_count = current_epoch
        except Exception:
            pass

    def on_finished(self, result: dict):
        """Handle successful completion."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Inference complete!")
        self.run_page.set_activity(
            "Inference complete. Results are ready to review.", active=False
        )

        # Show total elapsed time
        if self.run_state.started_at is not None:
            elapsed = time.time() - self.run_state.started_at
            if elapsed < 60:
                self.eta_label.setText(f"Completed in {elapsed:.1f}s")
            elif elapsed < 3600:
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                self.eta_label.setText(f"Completed in {mins}m {secs}s")
            else:
                hours = int(elapsed // 3600)
                mins = int((elapsed % 3600) // 60)
                self.eta_label.setText(f"Completed in {hours}h {mins}m")
            self.run_state.finish()

        # Display results summary (extract only what we need to avoid keeping large arrays)
        n_epochs = result.get("n_epochs", 0)
        predictions = result.get("predictions", None)
        probabilities = result.get("probabilities", None)
        output_paths = result.get("output_paths", {})
        confidences = result.get("confidences", None)

        # Copy only essential data for display
        if predictions is not None:
            # Keep predictions for display
            predictions_copy = predictions.copy()
        else:
            predictions_copy = None

        # Keep confidences for confidence review and exports
        if confidences is not None:
            confidences_copy = confidences.copy()
        else:
            confidences_copy = _resolve_confidences(probabilities, None)

        # Keep the small per-epoch uncertainty-flag arrays for the triage
        # worklist (float32 copies; each is (n_epochs,), not the big posterior).
        flag_scores_copy: dict[str, np.ndarray] = {}
        for key in _UNCERTAINTY_FLAG_KEYS:
            arr = result.get(key, None)
            if arr is not None:
                flag_scores_copy[key] = np.asarray(arr, dtype=np.float32).copy()

        # Immediately delete the full result dict to free memory
        del result

        # Display with copied data
        if predictions_copy is not None:
            self.display_results(
                {
                    "n_epochs": n_epochs,
                    "predictions": predictions_copy,
                    "probabilities": probabilities,
                    "output_paths": output_paths,
                    "confidences": confidences_copy,
                    "flag_scores": flag_scores_copy or None,
                }
            )

        # Handle batch processing continuation
        if self._batch_mode:
            if 0 <= self._batch_current_index < len(self._batch_queue):
                self._batch_queue[self._batch_current_index]["status"] = "done"
                self._batch_queue[self._batch_current_index]["percent"] = 100
                self._refresh_batch_queue_dialog()
            self._batch_current_index += 1
            if self._batch_paused:
                self.log(
                    "Batch paused after completing the current file. "
                    "Click Resume to continue.",
                    logging.INFO,
                )
                self._refresh_batch_queue_dialog()
                return
            # Use a short timer to allow GUI update before starting next file
            QTimer.singleShot(500, self._process_next_in_batch)

    def on_error(self, error_msg: str):
        """Handle error."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Inference failed - see log for details")
        self.run_page.set_activity(
            "Inference failed. Review the log below, then return to Setup to correct the inputs.",
            active=False,
        )

        # Log error
        self.log(f"ERROR: {error_msg}", logging.ERROR)

        # Show error dialog with full traceback available in Details
        self._show_error_dialog(
            title="Inference Error",
            summary="Inference failed.",
            detail=error_msg,
            suggestion=(
                "Check the log window for context. If this is reproducible, copy "
                "the details below and include them in your bug report."
            ),
        )

        # Mark current batch entry as failed and advance, so a bad file doesn't
        # take the whole queue down.
        if self._batch_mode:
            if 0 <= self._batch_current_index < len(self._batch_queue):
                entry = self._batch_queue[self._batch_current_index]
                entry["status"] = "error"
                entry["error"] = error_msg.splitlines()[-1][:120] if error_msg else ""
                self._refresh_batch_queue_dialog()
            self._batch_current_index += 1
            if not self._batch_paused:
                QTimer.singleShot(500, self._process_next_in_batch)

    def _show_error_dialog(
        self,
        title: str,
        summary: str,
        detail: str,
        *,
        suggestion: str | None = None,
        icon: QMessageBox.Icon = QMessageBox.Icon.Critical,
    ) -> None:
        """Show a standardized error dialog with a collapsible detail pane.

        Args:
            title: Window title shown in the title bar.
            summary: One-line headline displayed in bold.
            detail: Full text (typically a traceback) accessible via "Show Details".
            suggestion: Optional recovery hint shown beneath the summary.
            icon: Icon to display; defaults to Critical.
        """
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(summary)
        if suggestion:
            box.setInformativeText(suggestion)
        if detail:
            box.setDetailedText(detail)
        # Standard OK button plus a Copy-to-clipboard button.
        copy_button = box.addButton(
            "Copy to Clipboard", QMessageBox.ButtonRole.ActionRole
        )
        box.addButton(QMessageBox.StandardButton.Ok)
        box.setDefaultButton(QMessageBox.StandardButton.Ok)

        def _copy_to_clipboard() -> None:
            clipboard = QApplication.clipboard()
            if clipboard is None:
                return
            parts = [f"{title}: {summary}"]
            if suggestion:
                parts.append(suggestion)
            if detail:
                parts.append("")
                parts.append(detail)
            clipboard.setText("\n".join(parts))

        copy_button.clicked.connect(_copy_to_clipboard)
        box.exec()

    def display_results(self, result: dict):
        """Display results summary."""
        from spectra.inference import STAGE_NAMES_5

        n_epochs = result["n_epochs"]
        predictions = np.asarray(result["predictions"], dtype=np.int64)
        probabilities = result.get("probabilities", None)
        confidences = result.get("confidences", None)
        if confidences is None:
            confidences = _resolve_confidences(probabilities, None)
        output_paths = result.get("output_paths", {})
        score_window = result.get("score_window", None)
        flag_scores = result.get("flag_scores", None)

        # Build summary message
        summary_lines = [
            "=" * 80,
            "INFERENCE RESULTS",
            "=" * 80,
            f"Total epochs: {n_epochs}",
            "",
            "Stage distribution:",
        ]

        epoch_min = self.epoch_sec_spin.value() / 60.0
        for stage_idx, stage_name in enumerate(STAGE_NAMES_5):
            count = (predictions == stage_idx).sum()
            percent = 100 * count / len(predictions)
            minutes = count * epoch_min
            summary_lines.append(
                f"  {stage_name:5s}: {count:5d} epochs ({percent:5.1f}%) = {minutes:6.1f} minutes"
            )

        summary_lines.extend(
            [
                "",
                "Output files:",
            ]
        )

        for key, path in output_paths.items():
            summary_lines.append(f"  {key}: {path}")

        summary_lines.extend(
            [
                "",
                "=" * 80,
            ]
        )

        summary = "\n".join(summary_lines)
        self.log(summary, logging.INFO)

        self._populate_results_from_arrays(
            predictions=predictions,
            probabilities=probabilities,
            confidences=confidences,
            output_paths={str(k): str(v) for k, v in output_paths.items()},
            channel_layout=self._get_effective_channel_layout(),
            score_window=score_window,
            flag_scores=flag_scores,
            reset_overrides=True,
            switch_to_hypnogram=True,
        )

        # Show success dialog
        QMessageBox.information(
            self,
            "Inference Complete",
            f"Inference completed successfully!\n\n"
            f"Scored {n_epochs} epochs.\n"
            f"Results saved to: {result['output_paths']['predictions_csv']}\n\n"
            "Review the hypnogram in the 'Hypnogram' tab and flagged epochs in the "
            "'Confidence Review' or 'Signal Review' tab.",
        )

    def _clear_result_views(self) -> None:
        """Clear recording-derived state and return result pages to empty states."""
        self.review_state.clear_results()
        self._score_window = None
        self._override_undo_stack.clear()

        self.hypnogram_canvas.predictions = None
        self.hypnogram_canvas.probabilities = None
        self.hypnogram_canvas.confidence = None
        self.hypnogram_canvas.reference_predictions = None
        self.hypnogram_canvas.ax.clear()
        self.hypnogram_canvas.ax_conf.clear()
        self.hypnogram_canvas.ax.text(
            0.5,
            0.5,
            "Run inference to display the hypnogram",
            transform=self.hypnogram_canvas.ax.transAxes,
            ha="center",
            va="center",
            color="#888888",
        )
        self.hypnogram_canvas.draw_idle()

        self.confidence_review_widget.update_review(None, None, None)
        self.signal_review_widget.update_review(None, None, None, None, None)
        self.signal_review_widget.set_signal_payload(None)
        self.architecture_widget.update_analysis(None)
        self.reference_agreement_panel.set_data(None, None)
        self.calibration_panel.set_data(None, None)

        for button in (
            self.export_png_button,
            self.export_pdf_button,
            self.export_svg_button,
            self.export_edfplus_button,
            self.export_csv_button,
            self.export_report_button,
        ):
            button.setEnabled(False)
        self.show_cycles_checkbox.setEnabled(False)
        self.highlight_low_conf_checkbox.setEnabled(False)
        self.conf_threshold_slider.setEnabled(False)
        self.compare_mode_button.setEnabled(False)
        self.clear_reference_button.setVisible(False)
        self.review_page.set_has_results(False)
        self.export_page.set_has_results(False)

    def _populate_results_from_arrays(
        self,
        *,
        predictions: np.ndarray,
        probabilities: np.ndarray | None,
        confidences: np.ndarray | None,
        output_paths: dict[str, str] | None = None,
        channel_layout: list[str] | None = None,
        score_window: tuple[int, int] | None = None,
        flag_scores: dict[str, np.ndarray] | None = None,
        reset_overrides: bool = True,
        switch_to_hypnogram: bool = True,
    ) -> None:
        """Populate every result widget from already-computed prediction arrays.

        Shared between the fresh-inference path (``display_results``) and the
        project-restore path so reopening a saved project doesn't re-run the model.

        Args:
            score_window: Optional ``(start, end)`` epoch range that was scored.
                Epochs outside it are blanked in the hypnogram and excluded from
                sleep statistics. ``None`` means the whole recording.
            flag_scores: Optional per-epoch uncertainty-flag arrays for the triage
                worklist. ``None`` (e.g. older projects) falls back to confidence.
        """
        if confidences is None:
            confidences = _resolve_confidences(probabilities, None)

        self._score_window = score_window

        self.review_state.base_predictions = predictions
        self.review_state.probabilities = probabilities
        self.review_state.confidences = confidences
        self.review_state.flag_scores = flag_scores
        self.review_state.output_paths = dict(output_paths or {})
        if channel_layout is not None:
            self.review_state.channel_layout = list(channel_layout)
        if reset_overrides:
            self.review_state.manual_overrides.clear()
        # Fresh predictions invalidate the override history — there's no sensible
        # epoch alignment between a prior recording's overrides and these.
        self._override_undo_stack.clear()
        self.review_state.selected_epoch = 0
        self.review_state.signal_payload = None
        self.signal_review_widget.set_signal_payload(None)
        self.signal_review_widget.set_signal_loading_message(
            "Waiting to load aligned EDF review signals..."
        )
        self._refresh_result_views()

        # Enable export buttons
        self.export_png_button.setEnabled(True)
        self.export_pdf_button.setEnabled(True)
        self.export_svg_button.setEnabled(True)
        self.export_edfplus_button.setEnabled(True)
        self.export_csv_button.setEnabled(True)
        self.export_report_button.setEnabled(True)

        # Enable hypnogram control panel
        self.show_cycles_checkbox.setEnabled(True)
        self.highlight_low_conf_checkbox.setEnabled(True)
        self.conf_threshold_slider.setEnabled(True)
        self.compare_mode_button.setEnabled(True)
        self.review_page.set_has_results(True)
        self.export_page.set_has_results(True)
        self._start_review_signal_load()

        if switch_to_hypnogram:
            self._show_review_view("Hypnogram")

    def _get_effective_predictions(self) -> np.ndarray | None:
        """Return the active predictions including manual overrides."""
        return self.review_state.refresh_effective_predictions()

    def _has_manual_overrides(self) -> bool:
        """Return True when any manual rescoring is active."""
        return bool(self.review_state.manual_overrides)

    def _get_export_base_name(self, default_stem: str) -> str:
        """Append a rescored suffix to exports when manual overrides exist."""
        return (
            f"{default_stem}_rescored" if self._has_manual_overrides() else default_stem
        )

    def _resolve_score_window(self, n_epochs: int) -> tuple[int, int]:
        """Clamp the active analysis window to ``[0, n_epochs]``.

        Returns ``(0, n_epochs)`` (the whole recording) when no window is set.
        """
        if not self._score_window:
            return 0, n_epochs
        start, end = self._score_window
        start = max(0, min(int(start), n_epochs))
        end = max(start, min(int(end), n_epochs))
        return start, end

    def _refresh_result_views(self):
        """Refresh all result widgets from the current review state."""
        effective_predictions = self._get_effective_predictions()
        if effective_predictions is None:
            self.review_state.predictions = None
            return

        epoch_sec = self.epoch_sec_spin.value()
        active_threshold = self.conf_threshold_slider.value() / 100.0
        self.review_state.predictions = effective_predictions
        win_start, win_end = self._resolve_score_window(len(effective_predictions))

        # Aggregate-stats widgets report only the scored window; confidence/signal
        # review keep the full recording (they are per-epoch inspection tools).
        windowed_predictions = effective_predictions[win_start:win_end]
        windowed_confidences = (
            self.review_state.confidences[win_start:win_end]
            if self.review_state.confidences is not None
            else None
        )

        self.hypnogram_canvas.set_confidence_threshold(active_threshold)
        self.hypnogram_canvas.highlight_low_confidence = (
            self.highlight_low_conf_checkbox.isChecked()
        )
        self.statistics_widget.set_confidence_threshold(active_threshold)
        self.confidence_review_widget.set_confidence_threshold(active_threshold)
        self.signal_review_widget.set_confidence_threshold(active_threshold)
        self.epoch_details_dialog.set_confidence_threshold(active_threshold)

        self.hypnogram_canvas.plot_hypnogram(
            effective_predictions,
            epoch_sec,
            self.review_state.probabilities,
            self.review_state.confidences,
            score_window=(win_start, win_end),
        )

        self.epoch_details_dialog.set_data(
            self.review_state.base_predictions,
            self.review_state.confidences,
            self.review_state.probabilities,
            final_predictions=effective_predictions,
            manual_overrides=self.review_state.manual_overrides,
            epoch_callback=self._highlight_epoch_on_canvas,
        )

        widget_updates = [
            (
                "sleep statistics",
                lambda: self.statistics_widget.update_statistics(
                    windowed_predictions,
                    epoch_sec,
                    windowed_confidences,
                ),
            ),
            (
                "confidence review",
                lambda: self.confidence_review_widget.update_review(
                    self.review_state.base_predictions,
                    self.review_state.probabilities,
                    self.review_state.confidences,
                    epoch_sec,
                    final_predictions=effective_predictions,
                    manual_overrides=self.review_state.manual_overrides,
                    flag_scores=self.review_state.flag_scores,
                ),
            ),
            (
                "signal review",
                lambda: self.signal_review_widget.update_review(
                    self.review_state.base_predictions,
                    effective_predictions,
                    self.review_state.probabilities,
                    self.review_state.confidences,
                    self.review_state.manual_overrides,
                    epoch_sec,
                    flag_scores=self.review_state.flag_scores,
                ),
            ),
            (
                "sleep architecture",
                lambda: self.architecture_widget.update_analysis(
                    windowed_predictions,
                    epoch_sec,
                ),
            ),
        ]
        for widget_name, update_fn in widget_updates:
            try:
                update_fn()
            except Exception as exc:
                self.log(
                    f"Failed to refresh {widget_name} panel: {exc}",
                    logging.WARNING,
                )

        if len(effective_predictions) > 0:
            self._select_epoch(self.review_state.selected_epoch)

        # Keep the Validation tab in sync with the current predictions when a
        # reference is loaded (no-op otherwise).
        self._refresh_validation_panels()

    def _select_epoch(self, epoch_idx: int):
        """Synchronize the selected epoch across review widgets."""
        if (
            self.review_state.base_predictions is None
            or len(self.review_state.base_predictions) == 0
        ):
            return
        epoch_idx = max(
            0, min(int(epoch_idx), len(self.review_state.base_predictions) - 1)
        )
        self.review_state.selected_epoch = epoch_idx
        self.confidence_review_widget.select_epoch(epoch_idx)
        self.signal_review_widget.select_epoch(epoch_idx)

    def apply_manual_override(self, epoch_idx: int, stage_idx: int):
        """Apply or replace a manual stage override for an epoch."""
        if (
            self.review_state.base_predictions is None
            or len(self.review_state.base_predictions) == 0
        ):
            return
        if not 0 <= int(epoch_idx) < len(self.review_state.base_predictions):
            return
        if not 0 <= int(stage_idx) < len(STAGE_LABELS):
            return

        epoch = int(epoch_idx)
        base_stage = int(self.review_state.base_predictions[epoch])
        new_stage: int | None = int(stage_idx)
        if new_stage == base_stage:
            new_stage = None  # rescore back to base ⇒ remove override
        old_stage = self.review_state.manual_overrides.get(epoch)
        if new_stage == old_stage:
            return

        label = (
            f"Override epoch {epoch + 1} → {STAGE_LABELS[int(stage_idx)]}"
            if new_stage is not None
            else f"Clear override for epoch {epoch + 1}"
        )
        cmd = _OverrideCommand(
            self,
            epoch=epoch,
            old_stage=old_stage,
            new_stage=new_stage,
            text=label,
        )
        self._override_undo_stack.push(cmd)

    def clear_manual_override(self, epoch_idx: int):
        """Clear the manual override for one epoch."""
        epoch = int(epoch_idx)
        if epoch not in self.review_state.manual_overrides:
            return
        cmd = _OverrideCommand(
            self,
            epoch=epoch,
            old_stage=self.review_state.manual_overrides[epoch],
            new_stage=None,
            text=f"Clear override for epoch {epoch + 1}",
        )
        self._override_undo_stack.push(cmd)

    def clear_all_manual_overrides(self):
        """Clear all manual rescoring overrides."""
        if not self.review_state.manual_overrides:
            return
        cmd = _BulkOverrideCommand(
            self,
            previous=dict(self.review_state.manual_overrides),
            new={},
            text=(
                f"Clear {len(self.review_state.manual_overrides)} manual override(s)"
            ),
        )
        self._override_undo_stack.push(cmd)

    # ------------------------------------------------------------------
    # Low-level override mutators (called by undo commands)
    # ------------------------------------------------------------------

    def _set_override_state(
        self,
        epoch: int,
        new_stage: int | None,
        *,
        log_message: str | None = None,
    ) -> None:
        """Mutate `_manual_score_overrides` for a single epoch without history."""
        if not self.review_state.set_override(epoch, new_stage):
            return
        self._mark_project_modified()
        self._write_override_sidecar()
        if log_message:
            self.log(log_message, logging.INFO)
        self._refresh_result_views()
        if hasattr(self, "status_label"):
            self.status_label.setText(
                "Ctrl+Z to undo last override • Ctrl+Shift+Z to redo"
            )

    def _replace_all_overrides(
        self, new_overrides: dict[int, int], *, log_message: str | None = None
    ) -> None:
        """Replace the entire override dict without history."""
        if not self.review_state.replace_overrides(new_overrides):
            return
        self._mark_project_modified()
        self._write_override_sidecar()
        if log_message:
            self.log(log_message, logging.INFO)
        self._refresh_result_views()

    def _undo_override(self) -> None:
        """Trigger one step of override undo (bound to Ctrl+Z)."""
        if self._override_undo_stack.canUndo():
            self._override_undo_stack.undo()
        elif hasattr(self, "status_label"):
            self.status_label.setText("Nothing to undo.")

    def _redo_override(self) -> None:
        """Trigger one step of override redo (bound to Ctrl+Shift+Z)."""
        if self._override_undo_stack.canRedo():
            self._override_undo_stack.redo()
        elif hasattr(self, "status_label"):
            self.status_label.setText("Nothing to redo.")

    # ------------------------------------------------------------------
    # Crash-safe sidecar autosave of manual overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _override_sidecar_path(edf_path: str | Path) -> Path:
        """Compute the sidecar JSON path for a given EDF file."""
        edf = Path(edf_path)
        return edf.with_name(f"{edf.stem}.psgstage_overrides.json")

    def _write_override_sidecar(self) -> None:
        """Persist `_manual_score_overrides` to a sidecar JSON next to the EDF.

        Best-effort: failures are logged at WARNING level but do not interrupt
        the user. When the override dict is empty, any existing sidecar is removed
        so stale state doesn't haunt later sessions.
        """
        edf_path = self.edf_path_edit.text().strip()
        if not edf_path:
            return
        sidecar = self._override_sidecar_path(edf_path)
        try:
            if not self.review_state.manual_overrides:
                if sidecar.exists():
                    sidecar.unlink()
                return
            payload = {
                "format": "psgstage_overrides",
                "version": 1,
                "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                "edf_basename": Path(edf_path).name,
                "overrides": {
                    str(k): int(v)
                    for k, v in self.review_state.manual_overrides.items()
                },
            }
            sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            self.log(
                f"Could not write override sidecar {sidecar.name}: {e}",
                logging.WARNING,
            )

    def _load_override_sidecar(self) -> tuple[dict[int, int], str] | None:
        """Load a sidecar override file for the current EDF if one exists.

        Returns ``(overrides, saved_timestamp)`` or ``None`` when no recoverable
        sidecar is present.
        """
        edf_path = self.edf_path_edit.text().strip()
        if not edf_path:
            return None
        sidecar = self._override_sidecar_path(edf_path)
        if not sidecar.exists():
            return None
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as e:
            self.log(
                f"Could not read override sidecar {sidecar.name}: {e}",
                logging.WARNING,
            )
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("format") != "psgstage_overrides"
        ):
            return None
        raw_overrides = payload.get("overrides") or {}
        overrides = _coerce_manual_score_overrides(raw_overrides)
        if not overrides:
            return None
        saved = str(payload.get("saved", "an earlier session"))
        return overrides, saved

    def _offer_override_sidecar_restore(self) -> None:
        """Prompt the user to restore manual overrides from a sidecar file.

        Called after an EDF is selected. Skips when there is no sidecar, when
        the in-memory overrides already match, or when no results are loaded
        yet — in the latter case the overrides are kept pending and applied
        when results become available.
        """
        loaded = self._load_override_sidecar()
        if loaded is None:
            return
        overrides, saved = loaded
        if overrides == self.review_state.manual_overrides:
            return
        reply = QMessageBox.question(
            self,
            "Restore manual overrides?",
            (
                f"Found {len(overrides)} unsaved manual override(s) from "
                f"{saved} next to this EDF.\n\nRestore them now?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.review_state.manual_overrides = overrides
        self.log(
            f"Restored {len(overrides)} manual override(s) from sidecar.",
            logging.INFO,
        )
        if (
            self.review_state.base_predictions is not None
            and len(self.review_state.base_predictions) > 0
        ):
            self._refresh_result_views()

    def _get_review_signal_cache_key(self) -> tuple[str, tuple[str, ...]] | None:
        """Build the stable review-signal cache key for the current file/layout."""
        edf_path = self.edf_path_edit.text().strip()
        if not edf_path or not Path(edf_path).exists():
            return None
        layout = (
            self.review_state.channel_layout
            if self.review_state.channel_layout is not None
            else self._get_effective_channel_layout()
        )
        return edf_path, tuple(layout)

    def _start_review_signal_load(self):
        """Load or reuse aligned EDF review signals for the current file/layout."""
        cache_key = self._get_review_signal_cache_key()
        if cache_key is None:
            self.signal_review_widget.setEnabled(True)
            self.signal_review_widget.set_signal_loading_message(
                "Select a valid EDF file to enable waveform review.",
                is_error=True,
            )
            return

        cached_payload = self.review_state.signal_cache.get(cache_key)
        if cached_payload is not None:
            self.review_state.signal_payload = cached_payload
            self.signal_review_widget.setEnabled(True)
            self.signal_review_widget.set_signal_payload(cached_payload)
            return

        if (
            self._review_signal_loader is not None
            and self._review_signal_loader.isRunning()
        ):
            return

        edf_path, layout = cache_key
        self.signal_review_widget.setEnabled(False)
        self.signal_review_widget.set_signal_payload(None)
        self.signal_review_widget.set_signal_loading_message(
            "Loading aligned EDF review signals..."
        )
        self._review_signal_loader = ReviewSignalLoader(edf_path, list(layout))
        self._review_signal_loader.loaded.connect(self._on_review_signal_loaded)
        self._review_signal_loader.error.connect(self._on_review_signal_load_error)
        self._review_signal_loader.finished.connect(
            self._on_review_signal_loader_finished
        )
        self._review_signal_loader.start()

    def _on_review_signal_loaded(self, payload: dict):
        """Attach loaded review signals and cache them."""
        loader = self._review_signal_loader
        if loader is None:
            return
        origin_key = (str(loader.edf_path), tuple(loader.channel_layout))
        self.review_state.signal_cache[origin_key] = payload
        current_key = self._get_review_signal_cache_key()
        loader.deleteLater()
        self._review_signal_loader = None

        if origin_key == current_key and not self._close_pending:
            self.review_state.signal_payload = payload
            self.signal_review_widget.setEnabled(True)
            self.signal_review_widget.set_signal_payload(payload)
            self.log("Aligned EDF review signals loaded", logging.INFO)
            return

        if not self._close_pending:
            self.review_state.signal_payload = None
            self._start_review_signal_load()

    def _on_review_signal_load_error(self, message: str):
        """Handle waveform review loading errors."""
        loader = self._review_signal_loader
        origin_key = (
            (str(loader.edf_path), tuple(loader.channel_layout))
            if loader is not None
            else None
        )
        current_key = self._get_review_signal_cache_key()
        if loader is not None:
            loader.deleteLater()
            self._review_signal_loader = None
        if origin_key != current_key and not self._close_pending:
            self._start_review_signal_load()
            return
        self.review_state.signal_payload = None
        self.signal_review_widget.setEnabled(True)
        self.signal_review_widget.set_signal_payload(None)
        self.signal_review_widget.set_signal_loading_message(
            f"Failed to load review signals: {message}",
            is_error=True,
        )
        self.log(f"Failed to load review signals: {message}", logging.WARNING)

    def _on_review_signal_loader_finished(self) -> None:
        """Release an interrupted loader after its QThread exits."""
        loader = self.sender()
        current_loader = self._review_signal_loader
        if current_loader is not None and loader is current_loader:
            current_loader.deleteLater()
            self._review_signal_loader = None
        self._try_finish_deferred_close()

    def _on_review_tab_changed(self, index: int):
        """Kick off waveform loading when the Signal Review tab is opened."""
        tab_text = self.review_page.tab_text(index)
        if tab_text == "Signal Review":
            self._start_review_signal_load()

    def export_hypnogram(self, format_type: str):
        """Export hypnogram to file.

        Args:
            format_type: Export format ('png', 'pdf', 'svg')
        """
        # Get base name from EDF file
        edf_path = Path(self.edf_path_edit.text())
        base_name = (
            self._get_export_base_name(edf_path.stem)
            if edf_path.exists()
            else self._get_export_base_name("hypnogram")
        )

        # File dialog
        file_filters = {
            "png": "PNG Image (*.png)",
            "pdf": "PDF Document (*.pdf)",
            "svg": "SVG Vector (*.svg)",
        }

        default_name = f"{base_name}_hypnogram.{format_type}"
        output_dir = self.output_dir_edit.text() if self.output_dir_edit.text() else "."

        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Hypnogram as {format_type.upper()}",
            str(Path(output_dir) / default_name),
            file_filters[format_type],
        )

        if filename:
            try:
                self.hypnogram_canvas.export_hypnogram(filename, dpi=300)
                self.log(f"Hypnogram exported to: {filename}", logging.INFO)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Hypnogram exported successfully to:\n{filename}",
                )
            except Exception as e:
                self.log(f"Failed to export hypnogram: {e}", logging.ERROR)
                self._show_error_dialog(
                    title="Export Failed",
                    summary="Failed to export hypnogram.",
                    detail=traceback.format_exc(),
                    suggestion=(
                        "Check that the target directory is writable and that the "
                        "selected format is supported."
                    ),
                )

    def export_edf_annotations(self):
        """Export sleep stages as EDF+ annotations file.

        Creates a standard EDF+ annotations file compatible with most sleep software.
        """
        if self.hypnogram_canvas.predictions is None:
            QMessageBox.warning(
                self, "No Data", "No results to export. Run inference first."
            )
            return

        # Get base name from EDF file
        edf_path = Path(self.edf_path_edit.text())
        base_name = (
            self._get_export_base_name(edf_path.stem)
            if edf_path.exists()
            else self._get_export_base_name("sleep_stages")
        )
        output_dir = self.output_dir_edit.text() if self.output_dir_edit.text() else "."

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export EDF+ Annotations",
            str(Path(output_dir) / f"{base_name}_annotations.edf"),
            "EDF+ Files (*.edf)",
        )

        if not filename:
            return

        try:
            # Try to get recording start time from original EDF
            recording_start = None
            try:
                import mne

                raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
                recording_start = raw.info.get("meas_date")
                if recording_start is not None:
                    # Convert to datetime if needed
                    if hasattr(recording_start, "replace"):
                        recording_start = recording_start.replace(tzinfo=None)
            except Exception:
                pass  # Use None if we can't get the recording start time

            self.hypnogram_canvas.export_edf_annotations(filename, recording_start)
            self.log(f"EDF+ annotations exported to: {filename}", logging.INFO)

            # Show success with file info
            n_epochs = len(self.hypnogram_canvas.predictions)
            duration_hours = (n_epochs * self.hypnogram_canvas.epoch_sec) / 3600

            QMessageBox.information(
                self,
                "Export Successful",
                f"EDF+ annotations exported successfully!\n\n"
                f"File: {filename}\n"
                f"Epochs: {n_epochs}\n"
                f"Duration: {duration_hours:.1f} hours\n\n"
                f"This file can be loaded into sleep scoring software\n"
                f"such as RemLogic, Noxturnal, or EDFBROWSER.",
            )
        except Exception as e:
            self.log(f"Failed to export EDF+ annotations: {e}", logging.ERROR)
            self._show_error_dialog(
                title="Export Failed",
                summary="Failed to export EDF+ annotations.",
                detail=traceback.format_exc(),
                suggestion=(
                    "Make sure pyedflib is installed (uv sync) and the "
                    "target directory is writable."
                ),
            )

    def export_detailed_csv(self):
        """Export detailed epoch-by-epoch data to CSV.

        Includes timestamps, predicted stages, confidence scores, and flags.
        """
        if self.hypnogram_canvas.predictions is None:
            QMessageBox.warning(
                self, "No Data", "No results to export. Run inference first."
            )
            return

        # Get base name from EDF file
        edf_path = Path(self.edf_path_edit.text())
        base_name = (
            self._get_export_base_name(edf_path.stem)
            if edf_path.exists()
            else self._get_export_base_name("sleep_epochs")
        )
        output_dir = self.output_dir_edit.text() if self.output_dir_edit.text() else "."

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Detailed Epoch CSV",
            str(Path(output_dir) / f"{base_name}_epochs_detailed.csv"),
            "CSV Files (*.csv)",
        )

        if not filename:
            return

        try:
            self.hypnogram_canvas.export_epoch_csv(
                filename,
                threshold=self.hypnogram_canvas.confidence_threshold,
                model_predictions=self.review_state.base_predictions,
                manual_overrides=self.review_state.manual_overrides,
            )
            self.log(f"Detailed epoch CSV exported to: {filename}", logging.INFO)

            # Get summary info
            n_epochs = len(self.hypnogram_canvas.predictions)
            low_conf_epochs = self.hypnogram_canvas.get_low_confidence_epochs()
            n_low_conf = len(low_conf_epochs) if low_conf_epochs else 0

            summary = self.hypnogram_canvas.get_confidence_summary()
            mean_conf = float(summary.get("mean", 0)) * 100 if summary else 0
            threshold_pct = int(self.hypnogram_canvas.confidence_threshold * 100)

            QMessageBox.information(
                self,
                "Export Successful",
                f"Detailed epoch CSV exported successfully!\n\n"
                f"File: {filename}\n"
                f"Total epochs: {n_epochs}\n"
                f"Mean confidence: {mean_conf:.1f}%\n"
                f"Low-confidence epochs (<{threshold_pct}%): {n_low_conf}\n\n"
                f"Columns include:\n"
                f"• Epoch number, timestamp, elapsed time\n"
                f"• Predicted stage (numeric and label)\n"
                f"• Confidence score and LowConfidenceFlag\n"
                f"• Next-most-probable stage for flagged epochs\n"
                f"• Per-stage probabilities for Wake, N1, N2, N3, and REM",
            )
        except Exception as e:
            self.log(f"Failed to export detailed CSV: {e}", logging.ERROR)
            self._show_error_dialog(
                title="Export Failed",
                summary="Failed to export detailed epoch CSV.",
                detail=traceback.format_exc(),
                suggestion="Verify the target directory is writable and try again.",
            )

    def generate_pdf_report(self):
        """Generate a PDF report (delegates to the configurable enhanced path).

        Kept as a stable entry point for the toolbar button and the Ctrl+R
        shortcut; the actual rendering lives in
        :meth:`_generate_enhanced_pdf_report`, which is a superset (page size,
        color scheme, section toggles) and degrades gracefully when sleep
        statistics have not been computed.
        """
        self._generate_enhanced_pdf_report(None)

    def log(self, message: str, level=logging.INFO):
        """Add message to log."""
        logger = self._logger if self._logger is not None else logging.getLogger()
        logger.log(level, message)

    def clear_log(self):
        """Clear log output."""
        self.log_text.clear()

    def _copy_log_to_clipboard(self) -> None:
        """Copy the full log contents to the system clipboard."""
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        text = self.log_text.toPlainText()
        if not text:
            self.status_label.setText("Log is empty — nothing to copy.")
            return
        clipboard.setText(text)
        self.status_label.setText("Log copied to clipboard.")

    def _save_log_to_file(self) -> None:
        """Save the log contents to a timestamped .txt file."""
        text = self.log_text.toPlainText()
        if not text:
            QMessageBox.information(self, "Empty Log", "There is nothing to save yet.")
            return
        default_name = f"psgstage_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        # Prefer ~/Downloads if present, otherwise home dir.
        downloads = Path.home() / "Downloads"
        default_dir = downloads if downloads.is_dir() else Path.home()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Log As",
            str(default_dir / default_name),
            "Text Files (*.txt);;All Files (*)",
        )
        if not filename:
            return
        try:
            Path(filename).write_text(text, encoding="utf-8")
            self.log(f"Log saved to: {filename}", logging.INFO)
            self.status_label.setText(f"Log saved to {Path(filename).name}.")
        except Exception:
            self._show_error_dialog(
                title="Save Log Failed",
                summary="Could not save the log to disk.",
                detail=traceback.format_exc(),
                suggestion="Choose a writable location and try again.",
            )

    def toggle_advanced_mode(self, state: int):
        """Toggle visibility of advanced sections."""
        self._advanced_mode = state == Qt.CheckState.Checked.value

        # Show advanced sections only if both Easy Mode is OFF and Advanced Mode is ON
        show_advanced = self._advanced_mode and not self._easy_mode
        self.calibration_group.setVisible(show_advanced)
        self.advanced_group.setVisible(show_advanced)

    def toggle_easy_mode(self, state: int):
        """Toggle Easy Mode - shows/hides most configuration options."""
        self._easy_mode = state == Qt.CheckState.Checked.value

        # In Easy Mode, hide everything except files section
        self.signal_processing_group.setVisible(not self._easy_mode)
        self.inference_group.setVisible(not self._easy_mode)

        # Device selection within files section (keep visible only in non-easy mode)
        if hasattr(self, "device_label"):
            self.device_label.setVisible(not self._easy_mode)
            self.device_combo.setVisible(not self._easy_mode)

        # Update advanced sections visibility
        show_advanced = self._advanced_mode and not self._easy_mode
        self.calibration_group.setVisible(show_advanced)
        self.advanced_group.setVisible(show_advanced)

        # Disable Advanced Mode checkbox in Easy Mode
        self.mode_toggle.setEnabled(not self._easy_mode)
        self.mode_toggle.setVisible(not self._easy_mode)
        if not self._easy_mode and not self.mode_toggle.isChecked():
            self.mode_toggle.setChecked(True)

        if self._easy_mode:
            self.log("Easy Mode enabled - showing essential options only", logging.INFO)
        else:
            self.log(
                "Easy Mode disabled - showing all configuration options", logging.INFO
            )

    # =========================================================================
    # Progress Phase Indicators
    # =========================================================================

    def reset_phase_indicators(self):
        """Reset all phase indicators to initial state."""
        for phase in self.phase_indicators:
            phase["indicator"].setText("○")
            phase["indicator"].setStyleSheet("color: #555; font-size: 12px;")
            phase["label"].setStyleSheet("color: #666; font-size: 11px;")

    def set_phase(self, phase_name: str, status: str = "active"):
        """Set the status of a specific phase.

        Args:
            phase_name: One of 'Loading', 'Preprocessing', 'Inference', 'Post-processing', 'Saving'
            status: One of 'pending', 'active', 'completed'
        """
        self.run_state.phase = phase_name

        for phase in self.phase_indicators:
            name = phase["name"]

            if name == phase_name:
                if status == "active":
                    phase["indicator"].setText("◉")
                    phase["indicator"].setStyleSheet(
                        "color: #4CAF50; font-size: 12px; font-weight: bold;"
                    )
                    phase["label"].setStyleSheet(
                        "color: #4CAF50; font-size: 11px; font-weight: bold;"
                    )
                elif status == "completed":
                    phase["indicator"].setText("✓")
                    phase["indicator"].setStyleSheet("color: #4CAF50; font-size: 12px;")
                    phase["label"].setStyleSheet("color: #888; font-size: 11px;")
            else:
                # Check if this phase should be marked completed (comes before current phase)
                phase_order = [
                    "Loading",
                    "Preprocessing",
                    "Inference",
                    "Post-processing",
                    "Saving",
                ]
                current_idx = (
                    phase_order.index(phase_name) if phase_name in phase_order else -1
                )
                this_idx = phase_order.index(name) if name in phase_order else -1

                if this_idx < current_idx:
                    # This phase is completed
                    phase["indicator"].setText("✓")
                    phase["indicator"].setStyleSheet("color: #4CAF50; font-size: 12px;")
                    phase["label"].setStyleSheet("color: #888; font-size: 11px;")

    def update_phase_from_progress(self, percent: int):
        """Update phase indicators based on progress percentage."""
        # Map progress percentage to phases
        if percent < 5:
            self.set_phase("Loading", "active")
        elif percent < 15:
            self.set_phase("Preprocessing", "active")
        elif percent < 90:
            self.set_phase("Inference", "active")
        elif percent < 98:
            self.set_phase("Post-processing", "active")
        else:
            self.set_phase("Saving", "active")

    def dragEnterEvent(self, event):
        """Handle drag enter event for EDF files."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(".edf"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Handle drop event for EDF files."""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(".edf"):
                self.edf_path_edit.setText(file_path)
                self.log(f"EDF file loaded via drag-drop: {file_path}", logging.INFO)
                self._refresh_channel_layout_from_edf(log_missing=True)
                self._offer_override_sidecar_restore()
                event.acceptProposedAction()
                return
        event.ignore()

    def preview_edf_channels(self):
        """Preview available channels in the selected EDF file."""
        edf_path = self.edf_path_edit.text()
        if not edf_path:
            QMessageBox.warning(self, "No File", "Please select an EDF file first.")
            return

        if not Path(edf_path).exists():
            QMessageBox.warning(
                self, "File Not Found", "The selected EDF file does not exist."
            )
            return

        try:
            n_signals, signal_labels, sample_rates, n_samples, repaired = (
                _read_edf_header_with_repair(edf_path)
            )
            durations = [
                n_samples[i] / sample_rates[i] if sample_rates[i] > 0 else 0.0
                for i in range(n_signals)
            ]
            self._available_edf_channel_labels = list(signal_labels)
            normalized_overrides, warnings = _normalize_channel_slot_overrides(
                self._channel_slot_overrides,
                available_labels=self._available_edf_channel_labels,
            )
            self._channel_slot_overrides = normalized_overrides
            self._sync_channel_slot_comboboxes()
            for warning in warnings:
                self.log(warning, logging.WARNING)

            effective_layout = self._get_effective_channel_layout()
            channel_plan = _build_channel_preview_plan(signal_labels, effective_layout)

            # Build preview message
            msg = f"EDF File: {Path(edf_path).name}\n"
            msg += f"Total Signals: {n_signals}\n"
            if durations:
                msg += f"Duration: {max(durations) / 3600:.2f} hours\n"
            if repaired:
                msg += (
                    "Header Repair: Applied batch-compatible Physical Dimension fix "
                    "(empty unit fields -> uV)\n"
                )
            msg += "\n" + "=" * 50 + "\n"
            msg += f"{'Channel':<30} {'Rate (Hz)':<12} {'Status'}\n"
            msg += "=" * 50 + "\n"

            for i, label in enumerate(signal_labels):
                rate = f"{sample_rates[i]:.0f}"
                status = ""
                for requested, (_, plan_status, detail) in zip(
                    effective_layout, channel_plan, strict=True
                ):
                    if plan_status == "matched" and (
                        detail == label
                        or detail.startswith(f"{label} (")
                        or requested == label
                    ):
                        status = " [Matched]"
                        break
                msg += f"{label:<30} {rate:<12}{status}\n"

            msg += "\n" + "=" * 50 + "\n"
            msg += "Fixed 5-Slot Mapping Plan:\n"
            for (slot, _), requested, (_, plan_status, detail) in zip(
                CHANNEL_LAYOUT_SLOTS,
                effective_layout,
                channel_plan,
                strict=True,
            ):
                if plan_status == "derived":
                    display_status = "derived"
                elif plan_status == "missing":
                    display_status = "missing"
                elif requested == slot:
                    display_status = "auto match"
                else:
                    display_status = "manual direct"
                msg += (
                    f"  - {slot:<5} [{display_status:<12}] "
                    f"request={requested:<12} result={detail}\n"
                )

            # Show in a scrollable dialog
            dialog = QMessageBox(self)
            dialog.setWindowTitle("EDF Channel Preview")
            dialog.setIcon(QMessageBox.Icon.Information)
            dialog.setText(f"Found {n_signals} channels in the EDF file.")
            dialog.setDetailedText(msg)
            dialog.exec()

        except ImportError:
            QMessageBox.warning(
                self,
                "Missing Dependency",
                "pyedflib is required for EDF preview.\nInstall with: pip install pyedflib",
            )
        except Exception:
            self._show_error_dialog(
                title="Preview Error",
                summary="Failed to read EDF file.",
                detail=traceback.format_exc(),
                suggestion=(
                    "The file may be corrupted or in an unsupported format. Try "
                    "re-exporting it from your recording software."
                ),
            )

    def format_eta(self, seconds: float) -> str:
        """Format seconds into human-readable ETA string."""
        if seconds < 60:
            return f"{int(seconds)}s remaining"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s remaining"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m remaining"

    def get_settings_path(self) -> Path:
        """Get path to settings file."""
        # Store settings in user's home directory
        settings_dir = Path.home() / ".spectra"
        settings_dir.mkdir(exist_ok=True)
        return settings_dir / "inference_gui_settings.json"

    def load_settings(self):
        """Load saved settings from JSON file."""
        settings_path = self.get_settings_path()

        if not settings_path.exists():
            return

        try:
            with open(settings_path) as f:
                settings = json.load(f)

            # Load file paths
            if "edf_path" in settings:
                self.edf_path_edit.setText(settings["edf_path"])
            if "checkpoint" in settings:
                self.checkpoint_edit.setText(settings["checkpoint"])
            if "output_dir" in settings:
                self.output_dir_edit.setText(settings["output_dir"])
            if isinstance(settings.get("channel_slot_overrides"), dict):
                normalized_overrides, _ = _normalize_channel_slot_overrides(
                    settings["channel_slot_overrides"]
                )
                self._channel_slot_overrides = normalized_overrides
            elif isinstance(settings.get("canon_json"), str):
                migrated = self._load_legacy_channel_slot_overrides(
                    settings["canon_json"],
                    source_label="settings",
                )
                if migrated is not None:
                    self._channel_slot_overrides = migrated
            if "device" in settings:
                idx = self.device_combo.findText(settings["device"])
                if idx >= 0:
                    self.device_combo.setCurrentIndex(idx)
            if "easy_mode" in settings and isinstance(settings["easy_mode"], bool):
                self.easy_mode_toggle.setChecked(settings["easy_mode"])
            if "advanced_mode" in settings and isinstance(
                settings["advanced_mode"], bool
            ):
                self.mode_toggle.setChecked(settings["advanced_mode"])

            # Load recent files (filter out non-existent files)
            if "recent_edf_files" in settings and isinstance(
                settings["recent_edf_files"], list
            ):
                self._recent_edf_files = [
                    f for f in settings["recent_edf_files"] if Path(f).exists()
                ]
                self._update_recent_menu("edf")
            if "recent_checkpoint_files" in settings and isinstance(
                settings["recent_checkpoint_files"], list
            ):
                self._recent_checkpoint_files = [
                    f for f in settings["recent_checkpoint_files"] if Path(f).exists()
                ]
                self._update_recent_menu("checkpoint")

            # Load signal processing settings with validation
            if "epoch_sec" in settings:
                val = settings["epoch_sec"]
                if (
                    isinstance(val, (int, float))
                    and self.epoch_sec_spin.minimum()
                    <= val
                    <= self.epoch_sec_spin.maximum()
                ):
                    self.epoch_sec_spin.setValue(int(val))
            if "auto_signal_window" in settings and isinstance(
                settings["auto_signal_window"], bool
            ):
                self.auto_signal_window_check.setChecked(settings["auto_signal_window"])

            # Load inference settings with validation
            if "batch_size" in settings:
                val = settings["batch_size"]
                if (
                    isinstance(val, (int, float))
                    and self.batch_size_spin.minimum()
                    <= val
                    <= self.batch_size_spin.maximum()
                ):
                    self.batch_size_spin.setValue(int(val))
            if "amp_mode" in settings:
                idx = self.amp_mode_combo.findText(settings["amp_mode"])
                if idx >= 0:
                    self.amp_mode_combo.setCurrentIndex(idx)

            # Load advanced settings with validation
            if "use_mc_dropout" in settings:
                if isinstance(settings["use_mc_dropout"], bool):
                    self.mc_dropout_check.setChecked(settings["use_mc_dropout"])
            if "mc_samples" in settings:
                val = settings["mc_samples"]
                if (
                    isinstance(val, (int, float))
                    and self.mc_samples_spin.minimum()
                    <= val
                    <= self.mc_samples_spin.maximum()
                ):
                    self.mc_samples_spin.setValue(int(val))
            if "mc_dropout_rate" in settings:
                val = settings["mc_dropout_rate"]
                if (
                    isinstance(val, (int, float))
                    and self.mc_dropout_rate_spin.minimum()
                    <= val
                    <= self.mc_dropout_rate_spin.maximum()
                ):
                    self.mc_dropout_rate_spin.setValue(float(val))
            if "mc_pooling" in settings:
                idx = self.mc_pooling_combo.findText(str(settings["mc_pooling"]))
                if idx >= 0:
                    self.mc_pooling_combo.setCurrentIndex(idx)
            if "mc_include_attention" in settings:
                self.mc_attention_check.setChecked(
                    bool(settings["mc_include_attention"])
                )
            if "confidence_threshold" in settings:
                val = settings["confidence_threshold"]
                min_threshold = int(MIN_CONFIDENCE_THRESHOLD * 100)
                max_threshold = int(MAX_CONFIDENCE_THRESHOLD * 100)
                if (
                    isinstance(val, (int, float))
                    and min_threshold <= val <= max_threshold
                ):
                    self.conf_threshold_slider.setValue(int(val))

            # Load View menu settings
            if "font_size" in settings:
                self.settings["font_size"] = settings["font_size"]
            if "high_contrast" in settings:
                self.settings["high_contrast"] = settings["high_contrast"]
            if settings.get("theme") in ("dark", "light"):
                self.settings["theme"] = settings["theme"]
            if "recent_projects" in settings and isinstance(
                settings["recent_projects"], list
            ):
                self.settings["recent_projects"] = [
                    p for p in settings["recent_projects"] if Path(p).exists()
                ]

            # Load output template
            if "output_template" in settings:
                self.project_state.output_template = settings["output_template"]

            # Load PDF settings
            if "pdf_settings" in settings and isinstance(
                settings["pdf_settings"], dict
            ):
                self._pdf_settings = settings["pdf_settings"]

            self._refresh_channel_layout_from_edf(log_missing=True)
            if (
                self.checkpoint_edit.text().strip()
                and Path(self.checkpoint_edit.text().strip()).exists()
            ):
                self._load_checkpoint_info(self.checkpoint_edit.text().strip())
            self._sync_advanced_option_widgets()
            # Re-apply theme stylesheet in case the user previously chose light mode.
            self.apply_stylesheet()
            self.log("Settings loaded successfully", logging.INFO)

        except Exception as e:
            self.log(f"Failed to load settings: {e}", logging.WARNING)

    def save_settings(self):
        """Save current settings to JSON file."""
        settings_path = self.get_settings_path()

        try:
            settings = {
                # Settings version for future compatibility
                "version": 4,
                # File paths
                "edf_path": self.edf_path_edit.text(),
                "checkpoint": self.checkpoint_edit.text(),
                "output_dir": self.output_dir_edit.text(),
                "channel_slot_overrides": self._channel_slot_overrides,
                "device": self.device_combo.currentText(),
                "easy_mode": self.easy_mode_toggle.isChecked(),
                "advanced_mode": self.mode_toggle.isChecked(),
                # Recent files
                "recent_edf_files": self._recent_edf_files[: self.MAX_RECENT_FILES],
                "recent_checkpoint_files": self._recent_checkpoint_files[
                    : self.MAX_RECENT_FILES
                ],
                # Signal processing
                "epoch_sec": self.epoch_sec_spin.value(),
                "auto_signal_window": self.auto_signal_window_check.isChecked(),
                # Inference
                "batch_size": self.batch_size_spin.value(),
                "amp_mode": self.amp_mode_combo.currentText(),
                # Advanced
                "use_mc_dropout": self.mc_dropout_check.isChecked(),
                "mc_samples": self.mc_samples_spin.value(),
                "mc_dropout_rate": self.mc_dropout_rate_spin.value(),
                "mc_pooling": self.mc_pooling_combo.currentText(),
                "mc_include_attention": self.mc_attention_check.isChecked(),
                "confidence_threshold": self.conf_threshold_slider.value(),
                # View settings
                "font_size": self.settings.get("font_size", 12),
                "high_contrast": self.settings.get("high_contrast", False),
                "theme": self.settings.get("theme", "dark"),
                "recent_projects": self.settings.get("recent_projects", [])[:10],
                # Output template
                "output_template": self.project_state.output_template,
                # PDF settings
                "pdf_settings": getattr(self, "_pdf_settings", {}),
            }

            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=2)

            self.log(f"Settings saved to {settings_path}", logging.INFO)

        except Exception as e:
            self.log(f"Failed to save settings: {e}", logging.WARNING)

    def _try_finish_deferred_close(self) -> None:
        """Close once every background QThread has exited cleanly."""
        if not self._close_pending:
            return
        inference_running = self.worker is not None and self.worker.isRunning()
        review_running = (
            self._review_signal_loader is not None
            and self._review_signal_loader.isRunning()
        )
        if inference_running or review_running:
            return
        self._close_pending = False
        QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        """Handle window close without destroying active worker threads."""
        current_worker = self.worker
        inference_running = current_worker is not None and current_worker.isRunning()
        if inference_running and not self._close_pending:
            reply = QMessageBox.question(
                self,
                "Inference Running",
                "Inference is still running. Do you want to stop it and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                assert current_worker is not None
                current_worker.stop()
            else:
                event.ignore()
                return  # user cancelled exit; keep window + state untouched

        current_loader = self._review_signal_loader
        review_running = current_loader is not None and current_loader.isRunning()
        if review_running:
            assert current_loader is not None
            current_loader.requestInterruption()

        if inference_running or review_running:
            self._close_pending = True
            if inference_running:
                self.status_label.setText("Cancelling inference before closing…")
            else:
                self.status_label.setText("Finishing waveform load before closing…")
            event.ignore()
            return

        event.accept()

        # Persist checkpoint, EDF, and all options so they restore next launch.
        self.save_settings()

        # Detach GUI log handler to avoid logging into deleted widgets.
        try:
            if self._logger is not None and self._qt_log_handler is not None:
                self._logger.removeHandler(self._qt_log_handler)
        except Exception:
            pass

    # =========================================================================
    # Project/Workspace Management
    # =========================================================================

    def _setup_project_menu(self):
        """Setup Project menu for workspace management."""
        menu_bar = self.menuBar()

        # Project menu
        project_menu = menu_bar.addMenu("Project")

        # New Project
        new_action = project_menu.addAction("New Project")
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_project)

        project_menu.addSeparator()

        # Open Project
        open_action = project_menu.addAction("Open Project...")
        open_action.setShortcut("Ctrl+Shift+O")
        open_action.triggered.connect(self._open_project)

        # Recent Projects submenu
        self._recent_projects_menu = project_menu.addMenu("Recent Projects")
        self._update_recent_projects_menu()

        project_menu.addSeparator()

        # Save Project
        save_action = project_menu.addAction("Save Project")
        save_action.setShortcut("Ctrl+Shift+S")
        save_action.triggered.connect(self._save_project)

        # Save Project As
        save_as_action = project_menu.addAction("Save Project As...")
        save_as_action.triggered.connect(self._save_project_as)

        project_menu.addSeparator()

        # Output Template Settings
        template_action = project_menu.addAction("Output Naming Template...")
        template_action.triggered.connect(self._show_template_dialog)

        project_menu.addSeparator()

        # Enhanced PDF Report
        pdf_config_action = project_menu.addAction("Configure PDF Report...")
        pdf_config_action.triggered.connect(self._show_pdf_config_dialog)

    def _update_recent_projects_menu(self):
        """Update the recent projects submenu."""
        self._recent_projects_menu.clear()

        recent = self.settings.get("recent_projects", [])
        if not recent:
            no_recent = self._recent_projects_menu.addAction("(No recent projects)")
            no_recent.setEnabled(False)
            return

        for path in recent[:10]:
            if Path(path).exists():
                action = self._recent_projects_menu.addAction(Path(path).name)
                action.setToolTip(path)
                action.triggered.connect(
                    lambda checked, p=path: self._open_project_file(p)
                )

        self._recent_projects_menu.addSeparator()
        clear_action = self._recent_projects_menu.addAction("Clear Recent")
        clear_action.triggered.connect(self._clear_recent_projects)

    def _clear_recent_projects(self):
        """Clear recent projects list."""
        self.settings["recent_projects"] = []
        self._update_recent_projects_menu()
        self.save_settings()

    def _new_project(self):
        """Create a new project (reset workspace)."""
        if self.project_state.modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Current project has unsaved changes. Save before creating new project?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                self._save_project()

        # Reset workspace
        self.edf_path_edit.clear()
        self.checkpoint_edit.clear()
        self.output_dir_edit.clear()
        self._available_edf_channel_labels = []
        self._channel_slot_overrides = _default_channel_slot_overrides()
        self._sync_channel_slot_comboboxes()
        self._clear_result_views()
        self.signal_review_widget.set_reader_preferences(
            {
                "page_duration_sec": DEFAULT_SIGNAL_REVIEW_PAGE_SEC,
                "eeg_eog_uv_per_div": DEFAULT_SIGNAL_REVIEW_EEG_EOG_UV_PER_DIV,
                "emg_uv_per_div": DEFAULT_SIGNAL_REVIEW_EMG_UV_PER_DIV,
                "display_preset": DEFAULT_SIGNAL_REVIEW_PRESET,
                "queue_visible": DEFAULT_SIGNAL_REVIEW_QUEUE_VISIBLE,
                "spacing_preset": DEFAULT_SIGNAL_REVIEW_SPACING,
            }
        )

        # Reset to defaults
        self.batch_size_spin.setValue(32)
        self.epoch_sec_spin.setValue(30)

        self.project_state.reset()
        self._update_window_title()

        self.log("New project created", logging.INFO)

    def _open_project(self):
        """Open a project file."""
        if self.project_state.modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Current project has unsaved changes. Save before opening?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                self._save_project()

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(Path.home()),
            "PSG Stage Projects (*.psgproj);;JSON Files (*.json);;All Files (*)",
        )

        if filename:
            self._open_project_file(filename)

    def _open_project_file(self, filepath: str):
        """Open a project from a file path."""
        try:
            with open(filepath) as f:
                project = json.load(f)

            # Validate project format
            if project.get("format") != "psgstage_project":
                QMessageBox.warning(
                    self, "Invalid File", "This file is not a valid PSG Stage project."
                )
                return

            # Never leak predictions or export state from the previously open study.
            # A valid results sidecar below will repopulate these views explicitly.
            self._clear_result_views()

            # Load file paths
            if "edf_path" in project:
                self.edf_path_edit.setText(project["edf_path"])
            if "checkpoint" in project:
                self.checkpoint_edit.setText(project["checkpoint"])
            if "output_dir" in project:
                self.output_dir_edit.setText(project["output_dir"])
            if isinstance(project.get("channel_slot_overrides"), dict):
                normalized_overrides, _ = _normalize_channel_slot_overrides(
                    project["channel_slot_overrides"]
                )
                self._channel_slot_overrides = normalized_overrides
            elif isinstance(project.get("canon_json"), str):
                migrated = self._load_legacy_channel_slot_overrides(
                    project["canon_json"],
                    source_label="project",
                )
                if migrated is not None:
                    self._channel_slot_overrides = migrated
                else:
                    self._channel_slot_overrides = _default_channel_slot_overrides()
            else:
                self._channel_slot_overrides = _default_channel_slot_overrides()
            self.review_state.manual_overrides = _coerce_manual_score_overrides(
                project.get("manual_score_overrides")
            )

            # Load parameters
            params = project.get("parameters", {})
            if "batch_size" in params:
                self.batch_size_spin.setValue(params["batch_size"])
            if "epoch_sec" in params:
                self.epoch_sec_spin.setValue(params["epoch_sec"])
            if "device" in params:
                idx = self.device_combo.findText(params["device"])
                if idx >= 0:
                    self.device_combo.setCurrentIndex(idx)
            if "amp_mode" in params:
                idx = self.amp_mode_combo.findText(params["amp_mode"])
                if idx >= 0:
                    self.amp_mode_combo.setCurrentIndex(idx)

            # Load output template
            if "output_template" in project:
                self.project_state.output_template = project["output_template"]

            # Load PDF report settings if present
            if "pdf_settings" in project:
                self._pdf_settings = project["pdf_settings"]
            self.signal_review_widget.set_reader_preferences(
                project.get("reader_preferences")
            )

            self._refresh_channel_layout_from_edf(log_missing=True)
            if (
                self.checkpoint_edit.text().strip()
                and Path(self.checkpoint_edit.text().strip()).exists()
            ):
                self._load_checkpoint_info(self.checkpoint_edit.text().strip())
            self.project_state.mark_saved(filepath)
            self._update_window_title()

            # Add to recent projects
            self._add_to_recent_projects(filepath)

            # Restore cached predictions, if a sidecar exists and is current.
            sidecar_name = project.get("results_sidecar")
            if sidecar_name:
                self._restore_results_from_sidecar(filepath, sidecar_name)

            self.log(f"Project loaded: {filepath}", logging.INFO)

        except Exception as e:
            self.log(f"Failed to load project: {e}", logging.ERROR)
            self._show_error_dialog(
                title="Load Failed",
                summary="Failed to load project.",
                detail=traceback.format_exc(),
                suggestion="Verify the .psgproj file is readable and not corrupted.",
            )

    def _save_project(self):
        """Save current project."""
        if self.project_state.path is None:
            self._save_project_as()
            return

        self._save_project_to_file(str(self.project_state.path))

    def _save_project_as(self):
        """Save project to a new file."""
        default_name = "untitled.psgproj"
        if self.project_state.path:
            default_name = self.project_state.path.name

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(Path.home() / default_name),
            "PSG Stage Projects (*.psgproj);;JSON Files (*.json)",
        )

        if filename:
            if not filename.endswith((".psgproj", ".json")):
                filename += ".psgproj"
            self._save_project_to_file(filename)

    def _save_project_to_file(self, filepath: str):
        """Save the project to a file."""
        try:
            results_sidecar_name: str | None = None
            if self.review_state.base_predictions is not None:
                results_sidecar_name = self._write_results_sidecar(filepath)

            project = {
                "format": "psgstage_project",
                "version": 4,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                # File paths
                "edf_path": self.edf_path_edit.text(),
                "checkpoint": self.checkpoint_edit.text(),
                "output_dir": self.output_dir_edit.text(),
                "channel_slot_overrides": self._channel_slot_overrides,
                "manual_score_overrides": {
                    str(epoch_idx): int(stage_idx)
                    for epoch_idx, stage_idx in sorted(
                        self.review_state.manual_overrides.items()
                    )
                },
                # Parameters
                "parameters": {
                    "batch_size": self.batch_size_spin.value(),
                    "epoch_sec": self.epoch_sec_spin.value(),
                    "device": self.device_combo.currentText(),
                    "amp_mode": self.amp_mode_combo.currentText(),
                },
                # Output template
                "output_template": self.project_state.output_template,
                # PDF settings
                "pdf_settings": getattr(self, "_pdf_settings", {}),
                # Reader preferences
                "reader_preferences": self.signal_review_widget.get_reader_preferences(),
                # Optional results sidecar (.npz next to the .psgproj)
                "results_sidecar": results_sidecar_name,
            }

            with open(filepath, "w") as f:
                json.dump(project, f, indent=2)

            self.project_state.mark_saved(filepath)
            self._update_window_title()

            # Add to recent projects
            self._add_to_recent_projects(filepath)

            if results_sidecar_name:
                self.log(
                    f"Project saved: {filepath} (with cached predictions in "
                    f"{results_sidecar_name})",
                    logging.INFO,
                )
            else:
                self.log(f"Project saved: {filepath}", logging.INFO)

        except Exception as e:
            self.log(f"Failed to save project: {e}", logging.ERROR)
            self._show_error_dialog(
                title="Save Failed",
                summary="Failed to save project.",
                detail=traceback.format_exc(),
                suggestion="Check that the target directory is writable.",
            )

    def _restore_results_from_sidecar(
        self, project_path: str, sidecar_name: str
    ) -> None:
        """Restore cached predictions from a project sidecar, with staleness check.

        Compares the cached EDF mtime/size against the current file. If they
        diverge, the cache is treated as stale and a warning is shown — the user
        can choose to re-run inference rather than view potentially mismatched
        results.
        """
        sidecar = self._load_results_sidecar(project_path, sidecar_name)
        if sidecar is None:
            self.log(
                f"Results sidecar '{sidecar_name}' is missing; predictions not restored.",
                logging.WARNING,
            )
            return

        # Staleness check: compare the cached EDF size/mtime against the on-disk file.
        edf_path = self.edf_path_edit.text().strip()
        cached_mtime = sidecar.get("edf_mtime", 0.0)
        cached_size = sidecar.get("edf_size", 0)
        stale = False
        if edf_path and Path(edf_path).exists():
            try:
                stat = Path(edf_path).stat()
                if cached_size and stat.st_size != cached_size:
                    stale = True
                elif cached_mtime and abs(stat.st_mtime - cached_mtime) > 1.0:
                    stale = True
            except OSError:
                stale = True
        if stale:
            reply = QMessageBox.question(
                self,
                "EDF file has changed",
                (
                    "The EDF file referenced by this project appears to have "
                    "been modified since predictions were cached.\n\n"
                    "Load the cached predictions anyway? (Choose No to clear "
                    "them so you can re-run inference.)"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.log(
                    "Cached predictions discarded — EDF has changed since save.",
                    logging.INFO,
                )
                return

        self._populate_results_from_arrays(
            predictions=sidecar["predictions"],
            probabilities=sidecar.get("probabilities"),
            confidences=sidecar.get("confidences"),
            channel_layout=sidecar.get("channel_layout"),
            flag_scores=sidecar.get("flag_scores"),
            reset_overrides=False,  # keep overrides loaded from .psgproj JSON
            switch_to_hypnogram=True,
        )
        self.log(
            f"Restored {len(sidecar['predictions'])} cached predictions from "
            f"{sidecar_name}.",
            logging.INFO,
        )

    def _write_results_sidecar(self, project_path: str) -> str | None:
        """Save current prediction arrays to an .npz next to the project file.

        Returns the relative filename so the project JSON can reference it. On
        failure, logs a warning and returns ``None`` (the project itself still
        saves successfully — predictions just won't be cached).
        """
        if self.review_state.base_predictions is None:
            return None
        sidecar = Path(project_path).with_suffix(".psgproj.npz")
        edf_path = self.edf_path_edit.text().strip()
        edf_mtime = 0.0
        edf_size = 0
        if edf_path and Path(edf_path).exists():
            try:
                stat = Path(edf_path).stat()
                edf_mtime = stat.st_mtime
                edf_size = stat.st_size
            except OSError:
                pass
        try:
            arrays: dict[str, np.ndarray] = {
                "predictions": np.asarray(
                    self.review_state.base_predictions, dtype=np.int64
                ),
            }
            if self.review_state.probabilities is not None:
                arrays["probabilities"] = np.asarray(
                    self.review_state.probabilities, dtype=np.float32
                )
            if self.review_state.confidences is not None:
                arrays["confidences"] = np.asarray(
                    self.review_state.confidences, dtype=np.float32
                )
            # Per-epoch uncertainty-flag arrays, namespaced to avoid colliding
            # with the reserved sidecar keys above.
            if self.review_state.flag_scores:
                for key, arr in self.review_state.flag_scores.items():
                    arrays[f"uflag__{key}"] = np.asarray(arr, dtype=np.float32)
            arrays["edf_mtime"] = np.asarray([edf_mtime], dtype=np.float64)
            arrays["edf_size"] = np.asarray([edf_size], dtype=np.int64)
            if self.review_state.channel_layout is not None:
                arrays["channel_layout"] = np.asarray(
                    self.review_state.channel_layout, dtype=object
                )
            np.savez_compressed(sidecar, **arrays)  # type: ignore[reportArgumentType]
            return sidecar.name
        except Exception as e:
            self.log(
                f"Could not write results sidecar {sidecar.name}: {e}",
                logging.WARNING,
            )
            return None

    def _load_results_sidecar(
        self, project_path: str, sidecar_name: str
    ) -> dict[str, Any] | None:
        """Load cached prediction arrays from a project sidecar .npz.

        Returns a dict with keys ``predictions``, ``probabilities``,
        ``confidences``, ``channel_layout``, ``edf_mtime``, ``edf_size``; or
        ``None`` if the sidecar is missing or unreadable.
        """
        sidecar = Path(project_path).parent / sidecar_name
        if not sidecar.exists():
            return None
        try:
            data = np.load(sidecar, allow_pickle=True)
        except Exception as e:
            self.log(
                f"Could not read results sidecar {sidecar.name}: {e}",
                logging.WARNING,
            )
            return None
        result: dict[str, Any] = {
            "predictions": np.asarray(data["predictions"], dtype=np.int64),
        }
        if "probabilities" in data.files:
            result["probabilities"] = np.asarray(
                data["probabilities"], dtype=np.float32
            )
        else:
            result["probabilities"] = None
        if "confidences" in data.files:
            result["confidences"] = np.asarray(data["confidences"], dtype=np.float32)
        else:
            result["confidences"] = None
        if "channel_layout" in data.files:
            result["channel_layout"] = [str(x) for x in data["channel_layout"].tolist()]
        else:
            result["channel_layout"] = None
        flag_scores: dict[str, np.ndarray] = {}
        for fname in data.files:
            if fname.startswith("uflag__"):
                flag_scores[fname[len("uflag__") :]] = np.asarray(
                    data[fname], dtype=np.float32
                )
        result["flag_scores"] = flag_scores or None
        if "edf_mtime" in data.files:
            result["edf_mtime"] = float(data["edf_mtime"][0])
        else:
            result["edf_mtime"] = 0.0
        if "edf_size" in data.files:
            result["edf_size"] = int(data["edf_size"][0])
        else:
            result["edf_size"] = 0
        return result

    def _add_to_recent_projects(self, filepath: str):
        """Add a project to the recent projects list."""
        recent = self.settings.get("recent_projects", [])

        # Remove if already exists, then add to front
        if filepath in recent:
            recent.remove(filepath)
        recent.insert(0, filepath)

        # Keep only last 10
        self.settings["recent_projects"] = recent[:10]
        self._update_recent_projects_menu()
        self.save_settings()

    def _update_window_title(self):
        """Update window title with project name."""
        title = "SPECTRA - Sleep Staging"
        if self.project_state.path:
            title = f"{self.project_state.path.name} - {title}"
            if self.project_state.modified:
                title = f"* {title}"
        self.setWindowTitle(title)

    def _mark_project_modified(self):
        """Mark the project as modified."""
        if not self.project_state.modified:
            self.project_state.mark_modified()
            self._update_window_title()

    # =========================================================================
    # Output Template Naming
    # =========================================================================

    def _show_template_dialog(self):
        """Show dialog for configuring output naming template."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Output Naming Template")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout(dialog)

        # Help text
        help_label = QLabel(
            "Configure the output file naming pattern using placeholders:\n\n"
            "Available placeholders:\n"
            "  {basename}  - Original file name (without extension)\n"
            "  {date}      - Current date (YYYY-MM-DD)\n"
            "  {time}      - Current time (HH-MM-SS)\n"
            "  {model}     - Model name from checkpoint\n"
            "  {subject}   - Subject ID (from filename or first part)\n"
            "  {session}   - Session number (if present in filename)\n"
            "  {n}         - Counter for multiple outputs"
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(help_label)

        # Template input
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Template:"))
        self._template_edit = QLineEdit(self.project_state.output_template)
        self._template_edit.setPlaceholderText("{basename}")
        template_layout.addWidget(self._template_edit)
        layout.addLayout(template_layout)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self._template_preview = QLabel()
        self._template_preview.setStyleSheet("color: #4CAF50; font-family: monospace;")
        preview_layout.addWidget(self._template_preview)
        layout.addWidget(preview_group)

        # Update preview when template changes
        self._template_edit.textChanged.connect(self._update_template_preview)
        self._update_template_preview()

        # Preset buttons
        presets_layout = QHBoxLayout()
        presets_layout.addWidget(QLabel("Presets:"))

        presets = [
            ("Simple", "{basename}"),
            ("With Date", "{basename}_{date}"),
            ("Full", "{subject}_{date}_{model}"),
            ("Timestamped", "{basename}_{date}_{time}"),
        ]
        for name, template in presets:
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda checked, t=template: self._template_edit.setText(t)
            )
            presets_layout.addWidget(btn)
        presets_layout.addStretch()
        layout.addLayout(presets_layout)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.project_state.output_template = (
                self._template_edit.text() or "{basename}"
            )
            self._mark_project_modified()
            self.log(
                f"Output template set to: {self.project_state.output_template}",
                logging.INFO,
            )

    def _update_template_preview(self):
        """Update the template preview label."""
        template = self._template_edit.text() or "{basename}"

        # Get sample values
        edf_path = (
            Path(self.edf_path_edit.text())
            if self.edf_path_edit.text()
            else Path("sample_recording.edf")
        )
        basename = edf_path.stem

        # Extract subject and session from filename
        parts = basename.split("_")
        subject = parts[0] if parts else "subject"
        session = "1"
        for part in parts:
            if part.lower().startswith("s") and part[1:].isdigit():
                session = part[1:]
                break

        # Get model name
        model = "transformer"
        ckpt = self.checkpoint_edit.text()
        if ckpt:
            model = Path(ckpt).stem.split("_")[0]

        # Apply template
        from datetime import datetime

        now = datetime.now()
        preview = template.format(
            basename=basename,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H-%M-%S"),
            model=model,
            subject=subject,
            session=session,
            n=1,
        )

        self._template_preview.setText(f"Example: {preview}_hypnogram.png")

    def apply_output_template(self, base_name: str, suffix: str = "") -> str:
        """Apply the output template to generate a filename.

        Args:
            base_name: Original base name (without extension)
            suffix: Suffix to add (e.g., '_hypnogram', '_report')

        Returns:
            Formatted filename (without extension)
        """
        template = self.project_state.output_template or "{basename}"

        # Extract subject and session from base_name
        parts = base_name.split("_")
        subject = parts[0] if parts else "subject"
        session = "1"
        for part in parts:
            if part.lower().startswith("s") and part[1:].isdigit():
                session = part[1:]
                break

        # Get model name
        model = "model"
        ckpt = self.checkpoint_edit.text()
        if ckpt:
            model = Path(ckpt).stem.split("_")[0]

        # Apply template
        from datetime import datetime

        now = datetime.now()

        try:
            result = template.format(
                basename=base_name,
                date=now.strftime("%Y-%m-%d"),
                time=now.strftime("%H-%M-%S"),
                model=model,
                subject=subject,
                session=session,
                n=1,
            )
        except KeyError as e:
            self.log(
                f"Invalid template placeholder: {e}, using basename", logging.WARNING
            )
            result = base_name

        return result + suffix

    # =========================================================================
    # Enhanced PDF Report Configuration
    # =========================================================================

    def _show_pdf_config_dialog(self):
        """Show dialog for configuring PDF report settings."""
        dialog = QDialog(self)
        dialog.setWindowTitle("PDF Report Configuration")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout(dialog)

        # Get current settings
        pdf_settings = getattr(self, "_pdf_settings", {})

        # Header section
        header_group = QGroupBox("Header")
        header_layout = QFormLayout(header_group)

        self._pdf_title_edit = QLineEdit(
            pdf_settings.get("title", "Sleep Staging Report")
        )
        header_layout.addRow("Report Title:", self._pdf_title_edit)

        self._pdf_clinic_edit = QLineEdit(pdf_settings.get("clinic_name", ""))
        self._pdf_clinic_edit.setPlaceholderText("Enter clinic/lab name (optional)")
        header_layout.addRow("Clinic/Lab Name:", self._pdf_clinic_edit)

        self._pdf_technician_edit = QLineEdit(pdf_settings.get("technician", ""))
        self._pdf_technician_edit.setPlaceholderText("Enter technician name (optional)")
        header_layout.addRow("Technician:", self._pdf_technician_edit)

        layout.addWidget(header_group)

        # Content options
        content_group = QGroupBox("Content Options")
        content_layout = QVBoxLayout(content_group)

        self._pdf_include_hypnogram = QCheckBox("Include Hypnogram")
        self._pdf_include_hypnogram.setChecked(
            pdf_settings.get("include_hypnogram", True)
        )
        content_layout.addWidget(self._pdf_include_hypnogram)

        self._pdf_include_statistics = QCheckBox("Include Sleep Statistics")
        self._pdf_include_statistics.setChecked(
            pdf_settings.get("include_statistics", True)
        )
        content_layout.addWidget(self._pdf_include_statistics)

        self._pdf_include_pie_chart = QCheckBox("Include Stage Distribution Pie Chart")
        self._pdf_include_pie_chart.setChecked(
            pdf_settings.get("include_pie_chart", True)
        )
        content_layout.addWidget(self._pdf_include_pie_chart)

        self._pdf_include_architecture = QCheckBox("Include Sleep Architecture Details")
        self._pdf_include_architecture.setChecked(
            pdf_settings.get("include_architecture", True)
        )
        content_layout.addWidget(self._pdf_include_architecture)

        self._pdf_include_confidence = QCheckBox("Include Confidence Summary")
        self._pdf_include_confidence.setChecked(
            pdf_settings.get("include_confidence", False)
        )
        content_layout.addWidget(self._pdf_include_confidence)

        self._pdf_include_cycles = QCheckBox("Include Sleep Cycles Information")
        self._pdf_include_cycles.setChecked(pdf_settings.get("include_cycles", False))
        content_layout.addWidget(self._pdf_include_cycles)

        layout.addWidget(content_group)

        # Style options
        style_group = QGroupBox("Style Options")
        style_layout = QFormLayout(style_group)

        self._pdf_color_scheme = QComboBox()
        self._pdf_color_scheme.addItems(
            ["Standard (Color)", "Print-friendly (Grayscale)", "High Contrast"]
        )
        style_layout.addRow("Color Scheme:", self._pdf_color_scheme)

        self._pdf_page_size = QComboBox()
        self._pdf_page_size.addItems(["Letter (US)", "A4 (International)"])
        current_size = pdf_settings.get("page_size", "Letter")
        self._pdf_page_size.setCurrentText(
            "A4 (International)" if "A4" in current_size else "Letter (US)"
        )
        style_layout.addRow("Page Size:", self._pdf_page_size)

        self._pdf_orientation = QComboBox()
        self._pdf_orientation.addItems(["Portrait", "Landscape"])
        self._pdf_orientation.setCurrentText(
            pdf_settings.get("orientation", "Portrait")
        )
        style_layout.addRow("Orientation:", self._pdf_orientation)

        layout.addWidget(style_group)

        # Notes
        notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout(notes_group)
        self._pdf_notes_edit = QTextEdit()
        self._pdf_notes_edit.setPlaceholderText(
            "Add any notes to include in the report..."
        )
        self._pdf_notes_edit.setPlainText(pdf_settings.get("notes", ""))
        self._pdf_notes_edit.setMaximumHeight(80)
        notes_layout.addWidget(self._pdf_notes_edit)
        layout.addWidget(notes_group)

        # Buttons
        button_layout = QHBoxLayout()

        generate_btn = QPushButton("Generate Report Now")
        generate_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        generate_btn.clicked.connect(lambda: self._generate_enhanced_pdf_report(dialog))

        button_layout.addWidget(generate_btn)
        button_layout.addStretch()

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(lambda: self._save_pdf_settings(dialog))
        button_box.rejected.connect(dialog.reject)
        button_layout.addWidget(button_box)

        layout.addLayout(button_layout)

        dialog.exec()

    def _save_pdf_settings(self, dialog: QDialog):
        """Save PDF settings from the dialog."""
        self._pdf_settings = {
            "title": self._pdf_title_edit.text(),
            "clinic_name": self._pdf_clinic_edit.text(),
            "technician": self._pdf_technician_edit.text(),
            "include_hypnogram": self._pdf_include_hypnogram.isChecked(),
            "include_statistics": self._pdf_include_statistics.isChecked(),
            "include_pie_chart": self._pdf_include_pie_chart.isChecked(),
            "include_architecture": self._pdf_include_architecture.isChecked(),
            "include_confidence": self._pdf_include_confidence.isChecked(),
            "include_cycles": self._pdf_include_cycles.isChecked(),
            "color_scheme": self._pdf_color_scheme.currentText(),
            "page_size": self._pdf_page_size.currentText(),
            "orientation": self._pdf_orientation.currentText(),
            "notes": self._pdf_notes_edit.toPlainText(),
        }
        self._mark_project_modified()
        self.log("PDF report settings saved", logging.INFO)
        dialog.accept()

    def _generate_enhanced_pdf_report(self, config_dialog: QDialog | None = None):
        """Generate an enhanced PDF report with customization options."""
        if self.hypnogram_canvas.predictions is None:
            QMessageBox.warning(
                self, "No Data", "No results to export. Run inference first."
            )
            return

        # Save settings first if dialog is open
        if config_dialog:
            self._save_pdf_settings(config_dialog)

        # Get settings
        pdf_settings = getattr(self, "_pdf_settings", {})

        # Get output filename
        edf_path = Path(self.edf_path_edit.text())
        base_name = edf_path.stem if edf_path.exists() else "sleep_report"
        output_name = self.apply_output_template(base_name, "_report")
        output_dir = self.output_dir_edit.text() if self.output_dir_edit.text() else "."

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Enhanced PDF Report",
            str(Path(output_dir) / f"{output_name}.pdf"),
            "PDF Files (*.pdf)",
        )

        if not filename:
            return

        try:
            from datetime import datetime

            import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
            from matplotlib.backends.backend_pdf import (  # pyright: ignore[reportMissingModuleSource]
                PdfPages,  # pyright: ignore[reportMissingModuleSource]
            )

            # Get page size
            page_size = (11, 8.5)  # Letter landscape by default
            if "A4" in pdf_settings.get("page_size", ""):
                page_size = (11.69, 8.27)
            if pdf_settings.get("orientation", "Portrait") == "Portrait":
                page_size = (page_size[1], page_size[0])

            # Get color scheme
            grayscale = "Grayscale" in pdf_settings.get("color_scheme", "")
            high_contrast = "High Contrast" in pdf_settings.get("color_scheme", "")

            if grayscale:
                stage_colors = {
                    0: "#888888",
                    1: "#666666",
                    2: "#444444",
                    3: "#222222",
                    4: "#555555",
                }
            elif high_contrast:
                stage_colors = {
                    0: "#FF0000",
                    1: "#FFFF00",
                    2: "#00FF00",
                    3: "#0000FF",
                    4: "#FF00FF",
                }
            else:
                stage_colors = {
                    0: "#8B4513",
                    1: "#FFD700",
                    2: "#4CAF50",
                    3: "#2196F3",
                    4: "#FF4500",
                }

            with PdfPages(filename) as pdf:
                # Page 1: Title and Hypnogram
                fig1 = plt.figure(figsize=page_size)
                fig1.set_facecolor("white")

                # Title section with customization
                ax_title = fig1.add_axes((0.05, 0.85, 0.9, 0.12))
                ax_title.axis("off")

                title = pdf_settings.get("title", "Sleep Staging Report")
                ax_title.text(
                    0.5,
                    0.85,
                    title,
                    fontsize=22,
                    fontweight="bold",
                    ha="center",
                    va="top",
                    transform=ax_title.transAxes,
                )

                # Clinic and file info
                clinic = pdf_settings.get("clinic_name", "")
                if clinic:
                    ax_title.text(
                        0.5,
                        0.55,
                        clinic,
                        fontsize=12,
                        color="#666",
                        ha="center",
                        va="top",
                        transform=ax_title.transAxes,
                    )

                ax_title.text(
                    0.5,
                    0.35,
                    f"File: {edf_path.name}",
                    fontsize=11,
                    ha="center",
                    va="top",
                    transform=ax_title.transAxes,
                )

                info_line = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                tech = pdf_settings.get("technician", "")
                if tech:
                    info_line += f"  |  Technician: {tech}"
                ax_title.text(
                    0.5,
                    0.15,
                    info_line,
                    fontsize=10,
                    color="gray",
                    ha="center",
                    va="top",
                    transform=ax_title.transAxes,
                )

                # Hypnogram (if included)
                if pdf_settings.get("include_hypnogram", True):
                    ax_hypno = fig1.add_axes((0.08, 0.35, 0.88, 0.45))
                    predictions = self.hypnogram_canvas.predictions
                    epoch_sec = self.hypnogram_canvas.epoch_sec
                    n_epochs = len(predictions)
                    time_hours = np.arange(n_epochs) * epoch_sec / 3600

                    pred_to_y = {0: 4, 1: 2, 2: 1, 3: 0, 4: 3}
                    y_values = np.array([pred_to_y[p] for p in predictions])

                    for i in range(n_epochs - 1):
                        stage = predictions[i]
                        ax_hypno.plot(
                            [time_hours[i], time_hours[i + 1]],
                            [y_values[i], y_values[i]],
                            color=stage_colors[stage],
                            linewidth=2,
                            solid_capstyle="butt",
                        )
                        if y_values[i] != y_values[i + 1]:
                            ax_hypno.plot(
                                [time_hours[i + 1], time_hours[i + 1]],
                                [y_values[i], y_values[i + 1]],
                                color="gray",
                                linewidth=0.5,
                                alpha=0.5,
                            )

                    ax_hypno.set_yticks([4, 3, 2, 1, 0])
                    ax_hypno.set_yticklabels(["W", "REM", "N1", "N2", "N3"])
                    ax_hypno.set_xlabel("Time (hours)", fontsize=11)
                    ax_hypno.set_ylabel("Sleep Stage", fontsize=11)
                    ax_hypno.set_title("Hypnogram", fontsize=14, fontweight="bold")
                    ax_hypno.set_xlim(0, max(time_hours[-1] + epoch_sec / 3600, 1))
                    ax_hypno.grid(True, alpha=0.3, linestyle="--")

                # Summary box
                if (
                    pdf_settings.get("include_statistics", True)
                    and self.statistics_widget.statistics is not None
                ):
                    stats = self.statistics_widget.statistics.stats
                    ax_summary = fig1.add_axes((0.08, 0.05, 0.88, 0.25))
                    ax_summary.axis("off")

                    summary_text = (
                        f"Recording Duration: {stats['total_recording_hr']:.2f} hours ({stats['total_epochs']} epochs)\n"
                        f"Total Sleep Time (TST): {stats['tst_hr']:.2f} hours ({stats['tst_min']:.1f} min)\n"
                        f"Sleep Efficiency: {stats['sleep_efficiency']:.1f}%\n"
                        f"Sleep Onset Latency: {stats['sol_min']:.1f} min    |    "
                        f"REM Latency: {stats['rem_latency_min']:.1f} min    |    "
                        f"WASO: {stats['waso_min']:.1f} min"
                    )
                    ax_summary.text(
                        0.02,
                        0.95,
                        summary_text,
                        fontsize=11,
                        va="top",
                        transform=ax_summary.transAxes,
                        family="monospace",
                    )

                pdf.savefig(fig1, dpi=150)
                plt.close(fig1)

                # Page 2: Detailed Statistics (if enabled)
                if (
                    pdf_settings.get("include_pie_chart", True)
                    or pdf_settings.get("include_architecture", True)
                ) and self.statistics_widget.statistics is not None:
                    stats = self.statistics_widget.statistics.stats
                    fig2 = plt.figure(figsize=page_size)
                    fig2.set_facecolor("white")

                    if pdf_settings.get("include_pie_chart", True):
                        ax_pie = fig2.add_axes((0.05, 0.55, 0.4, 0.4))
                        stage_times = [
                            stats["wake_min"],
                            stats["n1_min"],
                            stats["n2_min"],
                            stats["n3_min"],
                            stats["rem_min"],
                        ]
                        stage_labels = ["Wake", "N1", "N2", "N3", "REM"]
                        colors = [stage_colors[i] for i in range(5)]
                        ax_pie.pie(
                            stage_times,
                            labels=stage_labels,
                            colors=colors,
                            autopct="%1.1f%%",
                            startangle=90,
                            textprops={"fontsize": 10},
                        )
                        ax_pie.set_title(
                            "Sleep Stage Distribution", fontsize=12, fontweight="bold"
                        )

                        # Stage Duration Table
                        ax_table = fig2.add_axes((0.55, 0.55, 0.4, 0.4))
                        ax_table.axis("off")
                        table_data = [
                            ["Stage", "Minutes", "Hours", "% of TST"],
                            [
                                "Wake",
                                f"{stats['wake_min']:.1f}",
                                f"{stats['wake_hr']:.2f}",
                                "--",
                            ],
                            [
                                "N1",
                                f"{stats['n1_min']:.1f}",
                                f"{stats['n1_hr']:.2f}",
                                f"{stats['pct_n1']:.1f}%",
                            ],
                            [
                                "N2",
                                f"{stats['n2_min']:.1f}",
                                f"{stats['n2_hr']:.2f}",
                                f"{stats['pct_n2']:.1f}%",
                            ],
                            [
                                "N3",
                                f"{stats['n3_min']:.1f}",
                                f"{stats['n3_hr']:.2f}",
                                f"{stats['pct_n3']:.1f}%",
                            ],
                            [
                                "REM",
                                f"{stats['rem_min']:.1f}",
                                f"{stats['rem_hr']:.2f}",
                                f"{stats['pct_rem']:.1f}%",
                            ],
                            [
                                "Total NREM",
                                f"{stats['nrem_min']:.1f}",
                                f"{stats['nrem_hr']:.2f}",
                                f"{stats['pct_nrem']:.1f}%",
                            ],
                        ]
                        table = ax_table.table(
                            cellText=table_data, loc="center", cellLoc="center"
                        )
                        table.auto_set_font_size(False)
                        table.set_fontsize(10)
                        table.scale(1.2, 1.5)
                        for i in range(len(table_data[0])):
                            table[(0, i)].set_facecolor("#4CAF50")
                            table[(0, i)].set_text_props(
                                color="white", fontweight="bold"
                            )

                    if pdf_settings.get("include_architecture", True):
                        ax_arch = fig2.add_axes((0.05, 0.08, 0.9, 0.4))
                        ax_arch.axis("off")
                        ax_arch.text(
                            0.5,
                            0.95,
                            "Sleep Architecture Details",
                            fontsize=14,
                            fontweight="bold",
                            ha="center",
                            transform=ax_arch.transAxes,
                        )

                        arch_text = (
                            f"Time in Bed (TIB):           {stats['tib_min']:.1f} min ({stats['tib_min'] / 60:.2f} hr)\n"
                            f"Sleep Period Time (SPT):     {stats['spt_min']:.1f} min ({stats['spt_min'] / 60:.2f} hr)\n"
                            f"Total Sleep Time (TST):      {stats['tst_min']:.1f} min ({stats['tst_hr']:.2f} hr)\n"
                            f"\n"
                            f"Sleep Efficiency:            {stats['sleep_efficiency']:.1f}%  (TST / TIB)\n"
                            f"Sleep Maintenance:           {stats['sleep_maintenance']:.1f}%  (TST / SPT)\n"
                            f"\n"
                            f"Sleep Onset Latency (SOL):   {stats['sol_min']:.1f} min\n"
                            f"REM Latency:                 {stats['rem_latency_min']:.1f} min\n"
                            f"Wake After Sleep Onset:      {stats['waso_min']:.1f} min\n"
                            f"\n"
                            f"Number of Awakenings:        {stats['n_awakenings']}\n"
                            f"Arousal Index:               {stats['arousal_index']:.2f} /hr\n"
                            f"Stage Shifts:                {stats['n_stage_shifts']}\n"
                            f"Fragmentation Index:         {stats['fragmentation_index']:.2f} /hr"
                        )
                        ax_arch.text(
                            0.1,
                            0.8,
                            arch_text,
                            fontsize=11,
                            va="top",
                            transform=ax_arch.transAxes,
                            family="monospace",
                        )

                    pdf.savefig(fig2, dpi=150)
                    plt.close(fig2)

                # Page 3: Confidence Summary (if enabled)
                if (
                    pdf_settings.get("include_confidence", False)
                    and self.hypnogram_canvas.confidence is not None
                ):
                    fig3 = plt.figure(figsize=page_size)
                    fig3.set_facecolor("white")

                    ax_conf = fig3.add_axes((0.1, 0.5, 0.8, 0.4))
                    confidence = self.hypnogram_canvas.confidence
                    confidence_threshold = self.hypnogram_canvas.confidence_threshold
                    time_hours = (
                        np.arange(len(confidence))
                        * self.hypnogram_canvas.epoch_sec
                        / 3600
                    )

                    ax_conf.fill_between(
                        time_hours, confidence, alpha=0.3, color="#4CAF50"
                    )
                    ax_conf.plot(time_hours, confidence, color="#4CAF50", linewidth=1)
                    ax_conf.axhline(
                        y=confidence_threshold,
                        color="#FFC107",
                        linestyle="--",
                        alpha=0.7,
                        label=f"Flag threshold ({confidence_threshold:.0%})",
                    )
                    ax_conf.set_xlabel("Time (hours)", fontsize=11)
                    ax_conf.set_ylabel("Confidence", fontsize=11)
                    ax_conf.set_title(
                        "Prediction Confidence Over Time",
                        fontsize=14,
                        fontweight="bold",
                    )
                    ax_conf.set_ylim(0, 1)
                    ax_conf.grid(True, alpha=0.3)
                    ax_conf.legend()

                    # Confidence statistics
                    ax_stats = fig3.add_axes((0.1, 0.1, 0.8, 0.3))
                    ax_stats.axis("off")
                    confidence_summary = _summarize_confidences(
                        confidence, confidence_threshold
                    )
                    if confidence_summary is None:
                        raise ValueError(
                            "Confidence summary unavailable for PDF export"
                        )
                    conf_text = (
                        f"Mean Confidence: {float(confidence_summary['mean']):.1%}\n"
                        f"Median Confidence: {float(confidence_summary['median']):.1%}\n"
                        f"Min Confidence: {float(confidence_summary['min']):.1%}\n"
                        f"High Confidence (≥{HIGH_CONFIDENCE_THRESHOLD:.0%}): {int(confidence_summary['high_count'])} epochs ({float(confidence_summary['pct_high']):.1f}%)\n"
                        f"Low Confidence (<{confidence_threshold:.0%}): {int(confidence_summary['low_count'])} epochs ({float(confidence_summary['pct_low']):.1f}%)"
                    )
                    ax_stats.text(
                        0.1,
                        0.9,
                        conf_text,
                        fontsize=12,
                        va="top",
                        transform=ax_stats.transAxes,
                        family="monospace",
                    )

                    pdf.savefig(fig3, dpi=150)
                    plt.close(fig3)

                # Page 4: Notes (if provided)
                notes = pdf_settings.get("notes", "")
                if notes.strip():
                    fig4 = plt.figure(figsize=page_size)
                    fig4.set_facecolor("white")

                    ax_notes = fig4.add_axes((0.1, 0.1, 0.8, 0.8))
                    ax_notes.axis("off")
                    ax_notes.text(
                        0.5,
                        0.95,
                        "Notes",
                        fontsize=16,
                        fontweight="bold",
                        ha="center",
                        transform=ax_notes.transAxes,
                    )
                    ax_notes.text(
                        0.05,
                        0.85,
                        notes,
                        fontsize=11,
                        va="top",
                        wrap=True,
                        transform=ax_notes.transAxes,
                    )

                    pdf.savefig(fig4, dpi=150)
                    plt.close(fig4)

            self.log(f"Enhanced PDF report generated: {filename}", logging.INFO)
            QMessageBox.information(
                self,
                "Report Generated",
                f"Enhanced PDF report saved successfully to:\n{filename}",
            )

            if config_dialog:
                config_dialog.accept()

        except Exception as e:
            self.log(f"Failed to generate PDF report: {e}", logging.ERROR)
            self._show_error_dialog(
                title="Report Generation Failed",
                summary="Failed to generate PDF report.",
                detail=traceback.format_exc(),
                suggestion=(
                    "Verify the target directory is writable and matplotlib's PDF "
                    "backend is installed."
                ),
            )

    # =========================================================================
    # Keyboard Shortcuts
    # =========================================================================

    def _setup_view_menu(self):
        """Setup View menu with font size and accessibility options."""
        menu_bar = self.menuBar()

        # View menu
        view_menu = menu_bar.addMenu("View")

        # Font size submenu
        font_menu = view_menu.addMenu("Font Size")

        self._font_sizes = {"Small": 10, "Normal": 12, "Large": 14, "Extra Large": 16}
        self._current_font_size = self.settings.get("font_size", 12)

        for name, size in self._font_sizes.items():
            action = font_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(size == self._current_font_size)
            action.triggered.connect(
                lambda checked, s=size, a=action: self._set_font_size(s)
            )

        view_menu.addSeparator()

        # High contrast mode
        self._high_contrast_action = view_menu.addAction("High Contrast Mode")
        self._high_contrast_action.setCheckable(True)
        self._high_contrast_action.setChecked(self.settings.get("high_contrast", False))
        self._high_contrast_action.triggered.connect(self._toggle_high_contrast)

        # Theme submenu (Dark / Light)
        theme_menu = view_menu.addMenu("Theme")
        current_theme = self.settings.get("theme", "dark")
        self._theme_dark_action = theme_menu.addAction("Dark")
        self._theme_dark_action.setCheckable(True)
        self._theme_dark_action.setChecked(current_theme == "dark")
        self._theme_dark_action.triggered.connect(lambda: self._set_theme("dark"))
        self._theme_light_action = theme_menu.addAction("Light")
        self._theme_light_action.setCheckable(True)
        self._theme_light_action.setChecked(current_theme == "light")
        self._theme_light_action.triggered.connect(lambda: self._set_theme("light"))

        view_menu.addSeparator()

        # Reset to defaults
        reset_action = view_menu.addAction("Reset to Defaults")
        reset_action.triggered.connect(self._reset_to_defaults)

        view_menu.addSeparator()

        # Help-overlay (Shift+F1) reachable from the menu too
        whats_this_action = view_menu.addAction("Show Field Help…")
        whats_this_action.setShortcut(QKeySequence("Shift+F1"))
        whats_this_action.triggered.connect(self._enter_whats_this_mode)

        # Apply saved settings
        self._apply_font_size(self._current_font_size)
        if self.settings.get("high_contrast", False):
            self._apply_high_contrast(True)

    def _set_font_size(self, size: int):
        """Set application font size."""
        self._current_font_size = size
        self.settings["font_size"] = size
        self.save_settings()
        self._apply_font_size(size)

    def _set_theme(self, theme: str) -> None:
        """Switch between 'dark' and 'light' themes."""
        if theme not in ("dark", "light"):
            return
        self.settings["theme"] = theme
        self.save_settings()
        self.apply_stylesheet()
        if hasattr(self, "_theme_dark_action"):
            self._theme_dark_action.setChecked(theme == "dark")
        if hasattr(self, "_theme_light_action"):
            self._theme_light_action.setChecked(theme == "light")

    def _apply_font_size(self, size: int):
        """Apply font size to the application."""
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)

        # Update specific widgets that may need larger fonts
        self.log_text.setStyleSheet(
            f"font-size: {size}px; background-color: #1e1e1e; color: #dcdcdc;"
        )

    def _toggle_high_contrast(self, checked: bool):
        """Toggle high contrast mode."""
        self.settings["high_contrast"] = checked
        self.save_settings()
        self._apply_high_contrast(checked)

    def _apply_high_contrast(self, enabled: bool):
        """Apply high contrast styling."""
        if enabled:
            # High contrast: black background, white text, bold borders
            base = self._dark_stylesheet()
            self.setStyleSheet(base + """
                QMainWindow, QWidget {
                    background-color: #000000;
                    color: #FFFFFF;
                }
                QGroupBox {
                    border: 2px solid #FFFFFF;
                    margin-top: 10px;
                    padding-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title {
                    color: #FFFFFF;
                }
                QPushButton {
                    background-color: #333333;
                    color: #FFFFFF;
                    border: 2px solid #FFFFFF;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    background-color: #1a1a1a;
                    color: #FFFFFF;
                    border: 2px solid #FFFFFF;
                }
                QTabWidget::pane {
                    border: 2px solid #FFFFFF;
                }
                QTabBar::tab {
                    background-color: #333333;
                    color: #FFFFFF;
                    border: 1px solid #FFFFFF;
                    padding: 8px;
                }
                QTabBar::tab:selected {
                    background-color: #555555;
                }
                QTableWidget {
                    background-color: #000000;
                    color: #FFFFFF;
                    gridline-color: #FFFFFF;
                }
                QHeaderView::section {
                    background-color: #333333;
                    color: #FFFFFF;
                    border: 1px solid #FFFFFF;
                }
                QCheckBox, QLabel {
                    color: #FFFFFF;
                }
            """)
        else:
            self.apply_stylesheet()

    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        response = QMessageBox.question(
            self,
            "Reset to Defaults",
            "This will reset all settings to their default values.\n\nAre you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if response == QMessageBox.StandardButton.Yes:
            # Clear settings
            self.settings = {
                "recent_edf_files": [],
                "recent_checkpoint_files": [],
                "font_size": 12,
                "high_contrast": False,
            }
            self.save_settings()

            # Reset UI
            self._apply_font_size(12)
            self._apply_high_contrast(False)
            self._high_contrast_action.setChecked(False)

            # Reset spinbox/checkbox values to defaults
            self.batch_size_spin.setValue(32)
            self.epoch_sec_spin.setValue(30)
            self.amp_mode_combo.setCurrentText("fp32")
            self._channel_slot_overrides = _default_channel_slot_overrides()
            self._available_edf_channel_labels = []
            self._sync_channel_slot_comboboxes()

            self.log("Settings reset to defaults", logging.INFO)
            QMessageBox.information(
                self, "Reset Complete", "All settings have been reset to defaults."
            )

    def _scoring_shortcuts_active(self) -> bool:
        """Whether single-key scoring shortcuts should act right now.

        Scoring keys apply to the selected epoch and are only meaningful on the
        review tabs. On the Confidence Review tab they are suppressed while the
        search box has focus so typing an epoch number doesn't rescore an epoch.
        """
        if self.tab_widget.currentWidget() is not self.review_page:
            return False
        current = self.review_page.current_widget()
        if current is self.signal_review_widget:
            return True
        if current is self.confidence_review_widget:
            return (
                QApplication.focusWidget()
                is not self.confidence_review_widget.search_edit
            )
        return False

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for power users."""
        # Ctrl+O - Open EDF file
        shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        shortcut_open.activated.connect(self.browse_edf)

        # Ctrl+M - Open Model checkpoint
        shortcut_model = QShortcut(QKeySequence("Ctrl+M"), self)
        shortcut_model.activated.connect(self.browse_checkpoint)

        # Ctrl+Enter - Start inference
        shortcut_start = QShortcut(QKeySequence("Ctrl+Return"), self)
        shortcut_start.activated.connect(self.start_inference)

        # Ctrl+Shift+Enter - Also start inference (alternative)
        shortcut_start2 = QShortcut(QKeySequence("Ctrl+Shift+Return"), self)
        shortcut_start2.activated.connect(self.start_inference)

        # Ctrl+S - Save settings
        shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut_save.activated.connect(self.save_settings)

        # Ctrl+E - Export current hypnogram as PNG
        shortcut_export = QShortcut(QKeySequence("Ctrl+E"), self)
        shortcut_export.activated.connect(
            lambda: (
                self.export_hypnogram("png")
                if self.hypnogram_canvas.predictions is not None
                else None
            )
        )

        # Ctrl+R - Generate PDF report
        shortcut_report = QShortcut(QKeySequence("Ctrl+R"), self)
        shortcut_report.activated.connect(self.generate_pdf_report)

        # Escape - Stop inference
        shortcut_stop = QShortcut(QKeySequence("Escape"), self)
        shortcut_stop.activated.connect(self.stop_inference)

        # Ctrl+1..9 - Switch tabs
        for i in range(min(9, self.tab_widget.count())):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            shortcut.activated.connect(
                lambda idx=i: self.tab_widget.setCurrentIndex(idx)
            )

        # Ctrl+L - Clear log
        shortcut_clear = QShortcut(QKeySequence("Ctrl+L"), self)
        shortcut_clear.activated.connect(self.clear_log)

        # Ctrl+B - Add files to batch queue
        shortcut_batch = QShortcut(QKeySequence("Ctrl+B"), self)
        shortcut_batch.activated.connect(self.add_to_batch_queue)

        # F1 - Show keyboard shortcuts help
        shortcut_help = QShortcut(QKeySequence("F1"), self)
        shortcut_help.activated.connect(self.show_shortcuts_help)

        # Left/Right - move across epochs when review data is available
        shortcut_prev_epoch = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        shortcut_prev_epoch.activated.connect(
            lambda: self.signal_review_widget.step_epoch(-1)
        )
        shortcut_next_epoch = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        shortcut_next_epoch.activated.connect(
            lambda: self.signal_review_widget.step_epoch(1)
        )

        # Shift+Left/Right - move by page
        shortcut_prev_page = QShortcut(QKeySequence("Shift+Left"), self)
        shortcut_prev_page.activated.connect(
            lambda: self.signal_review_widget.step_page(-1)
        )
        shortcut_next_page = QShortcut(QKeySequence("Shift+Right"), self)
        shortcut_next_page.activated.connect(
            lambda: self.signal_review_widget.step_page(1)
        )

        # Page Up/Down - larger page jumps
        shortcut_page_up = QShortcut(QKeySequence(Qt.Key.Key_PageUp), self)
        shortcut_page_up.activated.connect(
            lambda: self.signal_review_widget.step_page(-1)
        )
        shortcut_page_down = QShortcut(QKeySequence(Qt.Key.Key_PageDown), self)
        shortcut_page_down.activated.connect(
            lambda: self.signal_review_widget.step_page(1)
        )

        # Home/End - jump to start/end of study
        shortcut_home = QShortcut(QKeySequence(Qt.Key.Key_Home), self)
        shortcut_home.activated.connect(
            lambda: self.signal_review_widget.jump_to_edge("start")
        )
        shortcut_end = QShortcut(QKeySequence(Qt.Key.Key_End), self)
        shortcut_end.activated.connect(
            lambda: self.signal_review_widget.jump_to_edge("end")
        )

        # AASM manual scoring for the selected epoch: W/1/2/3/R (with 0/4 digit
        # aliases for Wake/REM). Gated to the review tabs so the keys don't fire
        # while typing in the Confidence Review search box.
        for key, stage_idx in SCORING_KEY_TO_STAGE.items():
            stage_shortcut = QShortcut(QKeySequence(key), self)
            stage_shortcut.activated.connect(
                lambda idx=stage_idx: (
                    self.apply_manual_override(self.review_state.selected_epoch, idx)
                    if self._scoring_shortcuts_active()
                    else None
                )
            )

        # Delete/Backspace - clear selected override (0 now scores Wake).
        for clear_key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            shortcut_clear_override = QShortcut(QKeySequence(clear_key), self)
            shortcut_clear_override.activated.connect(
                lambda: (
                    self.signal_review_widget.clear_selected_override()
                    if self._scoring_shortcuts_active()
                    else None
                )
            )

        # f / b - next or previous flagged epoch (global, switches to Signal Review)
        shortcut_next_flagged = QShortcut(QKeySequence("F"), self)
        shortcut_next_flagged.activated.connect(self._jump_to_next_flagged)
        shortcut_prev_flagged = QShortcut(QKeySequence("B"), self)
        shortcut_prev_flagged.activated.connect(self._jump_to_prev_flagged)

        # Ctrl+Z / Ctrl+Shift+Z - undo / redo manual stage overrides
        shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        shortcut_undo.activated.connect(self._undo_override)
        shortcut_redo = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        shortcut_redo.activated.connect(self._redo_override)

        # Shift+F1 - "What's this?" mode: click any control to read its tooltip.
        shortcut_whatsthis = QShortcut(QKeySequence("Shift+F1"), self)
        shortcut_whatsthis.activated.connect(self._enter_whats_this_mode)

        self.log("Keyboard shortcuts enabled (press F1 for help)", logging.INFO)

    def _enter_whats_this_mode(self) -> None:
        """Enter Qt's "What's this?" mode and propagate tooltips first.

        Qt renders the widget's ``whatsThis`` text (not the tooltip) in that
        mode. To make tooltips usable from this overlay, we copy every empty
        ``whatsThis`` from the matching tooltip on demand.
        """
        self._propagate_tooltips_to_whats_this()
        QWhatsThis.enterWhatsThisMode()
        self.status_label.setText(
            "Field help mode: click any control to see its description."
        )

    def _propagate_tooltips_to_whats_this(self) -> None:
        """Copy each widget's tooltip into its whatsThis when whatsThis is unset."""
        for widget in self.findChildren(QWidget):
            tooltip = widget.toolTip()
            if tooltip and not widget.whatsThis():
                widget.setWhatsThis(tooltip)

    def _jump_to_next_flagged(self) -> None:
        """Switch to Signal Review and jump to the next flagged epoch."""
        self._show_review_view("Signal Review")
        self.signal_review_widget._select_next_filtered()

    def _jump_to_prev_flagged(self) -> None:
        """Switch to Signal Review and jump to the previous flagged epoch."""
        self._show_review_view("Signal Review")
        self.signal_review_widget._select_previous_filtered()

    def _show_review_view(self, label: str) -> bool:
        """Open the Review workflow and select one specialist workspace."""
        if not self.review_page.show_view(label):
            return False
        self.tab_widget.setCurrentWidget(self.review_page)
        return True

    def _tab_index_by_label(self, label: str) -> int | None:
        """Return the nested Review index for ``label``, or ``None``."""
        return self.review_page.index_of(label)

    def show_shortcuts_help(self):
        """Show keyboard shortcuts help dialog."""
        shortcuts = """
<h3>Keyboard Shortcuts</h3>
<table style='margin: 10px;'>
<tr><td colspan='2'><b>File Operations</b></td></tr>
<tr><td><b>Ctrl+O</b></td><td>Open EDF file</td></tr>
<tr><td><b>Ctrl+M</b></td><td>Open model checkpoint</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save settings</td></tr>
<tr><td colspan='2'><b>Project</b></td></tr>
<tr><td><b>Ctrl+N</b></td><td>New project</td></tr>
<tr><td><b>Ctrl+Shift+O</b></td><td>Open project</td></tr>
<tr><td><b>Ctrl+Shift+S</b></td><td>Save project</td></tr>
<tr><td colspan='2'><b>Inference</b></td></tr>
<tr><td><b>Ctrl+Enter</b></td><td>Start inference</td></tr>
<tr><td><b>Escape</b></td><td>Stop inference</td></tr>
<tr><td><b>Ctrl+B</b></td><td>Add files to batch queue</td></tr>
<tr><td colspan='2'><b>Export</b></td></tr>
<tr><td><b>Ctrl+E</b></td><td>Export hypnogram as PNG</td></tr>
<tr><td><b>Ctrl+R</b></td><td>Generate PDF report</td></tr>
<tr><td colspan='2'><b>Navigation</b></td></tr>
<tr><td><b>Ctrl+1..4</b></td><td>Switch Setup / Run / Review / Export</td></tr>
<tr><td><b>Left / Right</b></td><td>Move selected epoch</td></tr>
<tr><td><b>Shift+Left / Shift+Right</b></td><td>Move one full signal page</td></tr>
<tr><td><b>Page Up / Page Down</b></td><td>Page backward / forward</td></tr>
<tr><td><b>Home / End</b></td><td>Jump to start / end of study</td></tr>
<tr><td><b>F / B</b></td><td>Next / previous flagged epoch</td></tr>
<tr><td colspan='2'><b>Manual Scoring (Signal / Confidence Review tabs)</b></td></tr>
<tr><td><b>W / 1 / 2 / 3 / R</b></td><td>Rescore selected epoch as Wake / N1 / N2 / N3 / REM</td></tr>
<tr><td><b>0 / 4</b></td><td>Digit aliases for Wake / REM</td></tr>
<tr><td><b>Delete / Backspace</b></td><td>Clear selected manual override</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>Undo last manual override</td></tr>
<tr><td><b>Ctrl+Shift+Z</b></td><td>Redo manual override</td></tr>
<tr><td><b>Ctrl+L</b></td><td>Clear log</td></tr>
<tr><td><b>Shift+F1</b></td><td>Field help mode (click any control for a description)</td></tr>
<tr><td><b>F1</b></td><td>Show this help</td></tr>
</table>
<p><i>Tip: Drag and drop EDF files directly onto the window!</i></p>
"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Keyboard Shortcuts")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(shortcuts)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    # =========================================================================
    # Recent Files Management
    # =========================================================================

    def add_to_recent_files(self, file_path: str, file_type: str):
        """Add a file to the recent files list.

        Args:
            file_path: Path to the file
            file_type: One of 'edf', 'checkpoint'
        """
        if not file_path or not Path(file_path).exists():
            return

        recent_list = {
            "edf": self._recent_edf_files,
            "checkpoint": self._recent_checkpoint_files,
        }.get(file_type)

        if recent_list is None:
            return

        # Remove if already exists (to move to top)
        if file_path in recent_list:
            recent_list.remove(file_path)

        # Add to beginning
        recent_list.insert(0, file_path)

        # Keep only MAX_RECENT_FILES
        while len(recent_list) > self.MAX_RECENT_FILES:
            recent_list.pop()

        # Update the corresponding dropdown
        self._update_recent_menu(file_type)

    def _update_recent_menu(self, file_type: str):
        """Update the recent files menu for a file type."""
        menu_map = {
            "edf": (self._recent_edf_files, getattr(self, "edf_recent_menu", None)),
            "checkpoint": (
                self._recent_checkpoint_files,
                getattr(self, "checkpoint_recent_menu", None),
            ),
        }

        recent_list, menu = menu_map.get(file_type, (None, None))
        if menu is None or recent_list is None:
            return

        menu.clear()
        if not recent_list:
            action = menu.addAction("No recent files")
            action.setEnabled(False)
        else:
            for path in recent_list:
                # Show just filename with tooltip for full path
                filename = Path(path).name
                action = menu.addAction(filename)
                action.setToolTip(path)
                action.setData(path)
                action.triggered.connect(
                    lambda checked, p=path, ft=file_type: self._select_recent_file(
                        p, ft
                    )
                )

    def _select_recent_file(self, file_path: str, file_type: str):
        """Select a file from recent files."""
        if not Path(file_path).exists():
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The file no longer exists:\n{file_path}\n\nIt will be removed from recent files.",
            )
            # Remove from list
            recent_list = {
                "edf": self._recent_edf_files,
                "checkpoint": self._recent_checkpoint_files,
            }.get(file_type)
            if recent_list and file_path in recent_list:
                recent_list.remove(file_path)
                self._update_recent_menu(file_type)
            return

        # Set the path in the appropriate field
        if file_type == "edf":
            self.edf_path_edit.setText(file_path)
            self._refresh_channel_layout_from_edf(log_missing=True)
            self._offer_override_sidecar_restore()
        elif file_type == "checkpoint":
            self.checkpoint_edit.setText(file_path)
            self._load_checkpoint_info(file_path)

        self.log(
            f"Loaded recent {file_type} file: {Path(file_path).name}", logging.INFO
        )

    # =========================================================================
    # Batch Processing
    # =========================================================================

    def add_to_batch_queue(self):
        """Add multiple EDF files to the batch processing queue."""
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select EDF Files for Batch Processing",
            "",
            "EDF Files (*.edf);;All Files (*)",
        )

        if not filenames:
            return

        existing = {entry["path"] for entry in self._batch_queue}
        added = 0
        for f in filenames:
            if f in existing:
                continue
            self._batch_queue.append(
                {"path": f, "status": "pending", "percent": 0, "error": ""}
            )
            existing.add(f)
            added += 1

        self.log(
            f"Added {added} files to batch queue (total: {len(self._batch_queue)})",
            logging.INFO,
        )

        # Show batch queue status
        self._update_batch_status()
        self._refresh_batch_queue_dialog()

    def _update_batch_status(self):
        """Update the batch processing status display."""
        if hasattr(self, "batch_status_label"):
            if self._batch_queue:
                self.batch_status_label.setText(
                    f"Batch: {len(self._batch_queue)} files queued"
                )
                self.batch_status_label.setStyleSheet(
                    "color: #4CAF50; font-weight: bold;"
                )
                self.start_batch_button.setEnabled(True)
                self.clear_batch_button.setEnabled(True)
            else:
                self.batch_status_label.setText("Batch: No files queued")
                self.batch_status_label.setStyleSheet("color: #888;")
                self.start_batch_button.setEnabled(False)
                self.clear_batch_button.setEnabled(False)
        if hasattr(self, "pause_batch_button"):
            self.pause_batch_button.setEnabled(self._batch_mode)
            self.pause_batch_button.setText("Resume" if self._batch_paused else "Pause")

    def start_batch_processing(self):
        """Start processing all files in the batch queue."""
        if not self._batch_queue:
            QMessageBox.warning(
                self, "No Files", "No files in batch queue. Use Ctrl+B to add files."
            )
            return

        if not self.checkpoint_edit.text():
            QMessageBox.warning(
                self,
                "Missing Configuration",
                "Please select a model checkpoint before starting batch processing.",
            )
            return

        # Reset any prior status flags but keep the queue order.
        for entry in self._batch_queue:
            entry["status"] = "pending"
            entry["percent"] = 0
            entry["error"] = ""

        self._batch_mode = True
        self._batch_paused = False
        self._batch_current_index = 0
        self.log(
            f"Starting batch processing of {len(self._batch_queue)} files...",
            logging.INFO,
        )
        self._update_batch_status()
        self._refresh_batch_queue_dialog()
        self._process_next_in_batch()

    def _process_next_in_batch(self):
        """Process the next file in the batch queue."""
        if self._batch_current_index >= len(self._batch_queue):
            self._batch_mode = False
            success = sum(1 for e in self._batch_queue if e["status"] == "done")
            errors = sum(1 for e in self._batch_queue if e["status"] == "error")
            self.log(
                f"Batch processing complete: {success} succeeded, {errors} failed.",
                logging.INFO,
            )
            QMessageBox.information(
                self,
                "Batch Complete",
                (
                    f"Batch complete.\n\n"
                    f"Succeeded: {success}\n"
                    f"Failed: {errors}\n\n"
                    f"Results saved to: {self.output_dir_edit.text()}"
                ),
            )
            self._update_batch_status()
            self._refresh_batch_queue_dialog()
            return

        if self._batch_paused:
            self.log("Batch paused. Press 'Resume' to continue.", logging.INFO)
            return

        entry = self._batch_queue[self._batch_current_index]
        current_file = entry["path"]
        entry["status"] = "running"
        entry["percent"] = 0
        entry["error"] = ""
        self.edf_path_edit.setText(current_file)
        self._refresh_channel_layout_from_edf(edf_path=current_file, log_missing=True)
        self.log(
            f"Batch [{self._batch_current_index + 1}/{len(self._batch_queue)}]: "
            f"{Path(current_file).name}",
            logging.INFO,
        )
        self._refresh_batch_queue_dialog()
        self.start_inference()

    def clear_batch_queue(self):
        """Clear the batch processing queue."""
        self._batch_queue.clear()
        self._batch_current_index = 0
        self._batch_mode = False
        self._batch_paused = False
        self._update_batch_status()
        self._refresh_batch_queue_dialog()
        self.log("Batch queue cleared", logging.INFO)

    def toggle_batch_pause(self) -> None:
        """Pause the batch between files or resume after being paused."""
        if not self._batch_mode:
            return
        self._batch_paused = not self._batch_paused
        if self._batch_paused:
            self.log("Batch will pause after the current file finishes.", logging.INFO)
        else:
            self.log("Batch resumed.", logging.INFO)
            # Kick the loop in case we were idle.
            QTimer.singleShot(100, self._process_next_in_batch)
        self._update_batch_status()
        self._refresh_batch_queue_dialog()

    def show_batch_queue(self):
        """Open a resizable dialog listing all queued files with live status."""
        if not self._batch_queue and self._batch_queue_dialog is None:
            QMessageBox.information(
                self,
                "Batch Queue",
                "No files in batch queue.\n\nUse Ctrl+B to add files.",
            )
            return

        if (
            self._batch_queue_dialog is not None
            and self._batch_queue_dialog.isVisible()
        ):
            self._batch_queue_dialog.raise_()
            self._batch_queue_dialog.activateWindow()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Queue")
        dialog.resize(640, 400)

        layout = QVBoxLayout(dialog)

        helper = QLabel(
            "Drag pending rows to reorder. Right-click a row for remove/retry. "
            "Use Pause to halt after the current file finishes."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(helper)

        table = QTableWidget(0, 4, dialog)
        table.setHorizontalHeaderLabels(["File", "Status", "Progress", "Notes"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(table.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(table.SelectionBehavior.SelectRows)
        table.setSelectionMode(table.SelectionMode.SingleSelection)
        table.setDragDropMode(table.DragDropMode.InternalMove)
        table.setDragDropOverwriteMode(False)
        table.setDefaultDropAction(Qt.DropAction.MoveAction)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(
            lambda pos, t=table: self._on_batch_queue_context_menu(t, pos)
        )
        # Persist new order on drop.
        table.model().rowsMoved.connect(
            lambda *_args, t=table: self._apply_batch_queue_table_order(t)
        )
        layout.addWidget(table, 1)

        button_layout = QHBoxLayout()
        pause_button = QPushButton("Resume" if self._batch_paused else "Pause")
        pause_button.setToolTip("Pause / resume the batch after the current file.")
        pause_button.clicked.connect(self.toggle_batch_pause)
        button_layout.addWidget(pause_button)

        remove_button = QPushButton("Remove Selected")
        remove_button.setToolTip("Remove the selected pending file from the queue.")
        remove_button.clicked.connect(lambda: self._remove_selected_batch_entry(table))
        button_layout.addWidget(remove_button)

        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        # Hold a reference for live updates from the inference progress callbacks.
        self._batch_queue_dialog = dialog
        self._batch_queue_table = table
        self._batch_pause_button = pause_button

        def _on_close() -> None:
            self._batch_queue_dialog = None
            self._batch_queue_table = None
            self._batch_pause_button = None

        dialog.finished.connect(lambda *_args: _on_close())
        self._refresh_batch_queue_dialog()
        dialog.show()

    def _on_batch_queue_context_menu(self, table: QTableWidget, position: Any) -> None:
        """Right-click menu on a queue row."""
        item = table.itemAt(position)
        if item is None:
            return
        row = item.row()
        if not 0 <= row < len(self._batch_queue):
            return
        entry = self._batch_queue[row]
        menu = QMenu(table)
        remove_action = menu.addAction("Remove from queue")
        retry_action = menu.addAction("Retry") if entry["status"] == "error" else None
        open_action = menu.addAction("Show in Finder/Explorer")
        chosen = menu.exec(table.viewport().mapToGlobal(position))
        if chosen is remove_action:
            self._remove_batch_entry_at(row)
        elif retry_action is not None and chosen is retry_action:
            entry["status"] = "pending"
            entry["percent"] = 0
            entry["error"] = ""
            self._refresh_batch_queue_dialog()
        elif chosen is open_action:
            self._reveal_in_file_manager(entry["path"])

    def _remove_selected_batch_entry(self, table: QTableWidget) -> None:
        """Remove the currently selected row from the queue."""
        rows = table.selectionModel().selectedRows() if table.selectionModel() else []
        if not rows:
            return
        self._remove_batch_entry_at(rows[0].row())

    def _remove_batch_entry_at(self, row: int) -> None:
        if not 0 <= row < len(self._batch_queue):
            return
        entry = self._batch_queue[row]
        if entry["status"] == "running":
            QMessageBox.information(
                self,
                "Cannot Remove",
                "This file is currently being processed. Stop the run first.",
            )
            return
        del self._batch_queue[row]
        if row < self._batch_current_index:
            self._batch_current_index = max(0, self._batch_current_index - 1)
        self._update_batch_status()
        self._refresh_batch_queue_dialog()

    def _reveal_in_file_manager(self, file_path: str) -> None:
        """Open the OS file manager and select the file."""
        path = Path(file_path)
        target = str(path.parent if path.exists() else path)
        try:
            if sys.platform == "darwin":
                import subprocess

                subprocess.run(["open", "-R", str(path)], check=False)
            elif sys.platform == "win32":
                import subprocess

                subprocess.run(["explorer", "/select,", str(path)], check=False)
            else:
                import subprocess

                subprocess.run(["xdg-open", target], check=False)
        except Exception as e:
            self.log(f"Could not reveal file {path}: {e}", logging.WARNING)

    def _apply_batch_queue_table_order(self, table: QTableWidget) -> None:
        """After a drag-reorder, re-sequence ``_batch_queue`` to match the table.

        Reordering of the currently running file is disallowed to avoid race
        conditions; we restore the prior order in that case.
        """
        path_to_entry = {entry["path"]: entry for entry in self._batch_queue}
        new_order: list[dict[str, Any]] = []
        for row in range(table.rowCount()):
            path_item = table.item(row, 0)
            if path_item is None:
                continue
            entry = path_to_entry.get(path_item.data(Qt.ItemDataRole.UserRole))
            if entry is not None:
                new_order.append(entry)
        if len(new_order) != len(self._batch_queue):
            return  # ambiguous; ignore
        if self._batch_mode:
            running_path = self._batch_queue[self._batch_current_index]["path"]
            try:
                self._batch_current_index = next(
                    i for i, e in enumerate(new_order) if e["path"] == running_path
                )
            except StopIteration:
                # Running entry vanished — bail out without applying the move.
                self._refresh_batch_queue_dialog()
                return
        self._batch_queue = new_order

    def _refresh_batch_queue_dialog(self) -> None:
        """Repopulate the live queue table with current statuses."""
        if self._batch_queue_table is None:
            return
        table = self._batch_queue_table
        # Block the rowsMoved signal while rebuilding programmatically.
        table.model().blockSignals(True)
        try:
            table.setRowCount(len(self._batch_queue))
            for row, entry in enumerate(self._batch_queue):
                name_item = QTableWidgetItem(Path(entry["path"]).name)
                name_item.setToolTip(entry["path"])
                name_item.setData(Qt.ItemDataRole.UserRole, entry["path"])
                table.setItem(row, 0, name_item)

                status = entry["status"]
                status_item = QTableWidgetItem(status)
                colour = {
                    "pending": "#888",
                    "running": "#42A5F5",
                    "done": "#66BB6A",
                    "error": "#EF5350",
                }.get(status, "#888")
                status_item.setForeground(QColor(colour))
                table.setItem(row, 1, status_item)

                pct = int(entry.get("percent", 0))
                progress_text = f"{pct}%" if status in ("running", "done") else "—"
                table.setItem(row, 2, QTableWidgetItem(progress_text))

                table.setItem(row, 3, QTableWidgetItem(str(entry.get("error", ""))))

                # Lock running/done rows so drag can't reorder them.
                if status in ("running", "done"):
                    name_item.setFlags(
                        name_item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled
                    )
        finally:
            table.model().blockSignals(False)
        if self._batch_pause_button is not None:
            self._batch_pause_button.setText(
                "Resume" if self._batch_paused else "Pause"
            )
            self._batch_pause_button.setEnabled(self._batch_mode)

    # -------------------------------------------------------------------------
    # Hypnogram Control Panel Handlers
    # -------------------------------------------------------------------------

    def _toggle_sleep_cycles(self, state):
        """Toggle sleep cycle markers on the hypnogram."""
        show_cycles = state == Qt.CheckState.Checked.value
        self.hypnogram_canvas.show_sleep_cycles = show_cycles
        # Re-plot to update display
        if self.review_state.predictions is not None:
            epoch_sec = self.epoch_sec_spin.value()
            self.hypnogram_canvas.plot_hypnogram(
                self.review_state.predictions,
                epoch_sec,
                self.review_state.probabilities,
                self.review_state.confidences,
            )
            self.log(
                f"Sleep cycle markers {'enabled' if show_cycles else 'disabled'}",
                logging.INFO,
            )

    def _toggle_low_confidence_highlight(self, state):
        """Toggle low confidence epoch highlighting."""
        highlight = state == Qt.CheckState.Checked.value
        self.hypnogram_canvas.highlight_low_confidence = highlight
        # Re-plot to update display
        if self.review_state.predictions is not None:
            epoch_sec = self.epoch_sec_spin.value()
            self.hypnogram_canvas.plot_hypnogram(
                self.review_state.predictions,
                epoch_sec,
                self.review_state.probabilities,
                self.review_state.confidences,
            )
            self.log(
                f"Low confidence highlighting {'enabled' if highlight else 'disabled'}",
                logging.INFO,
            )

    def _update_confidence_threshold(self, value):
        """Update confidence threshold for highlighting."""
        threshold = value / 100.0
        self.conf_threshold_label.setText(f"{value}%")
        self.conf_threshold_slider.setToolTip(
            f"Confidence threshold for highlighting (currently {value}%)"
        )
        self.hypnogram_canvas.set_confidence_threshold(threshold)
        self.statistics_widget.set_confidence_threshold(threshold)
        self.confidence_review_widget.set_confidence_threshold(threshold)
        self.signal_review_widget.set_confidence_threshold(threshold)
        self.epoch_details_dialog.set_confidence_threshold(threshold)

        if self.review_state.predictions is not None:
            epoch_sec = self.epoch_sec_spin.value()
            self.hypnogram_canvas.plot_hypnogram(
                self.review_state.predictions,
                epoch_sec,
                self.review_state.probabilities,
                self.review_state.confidences,
            )

    def _load_reference_hypnogram(self):
        """Load a reference hypnogram for comparison."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Reference Hypnogram",
            "",
            "All Supported (*.csv *.npy *.npz *.txt *.xml);;CSV Files (*.csv);;"
            "NumPy Files (*.npy *.npz);;Text Files (*.txt);;XML Files (*.xml)",
        )

        if not file_path:
            return

        try:
            from spectra.review.reference import parse_reference_hypnogram

            epoch_len = (
                float(self.epoch_sec_spin.value())
                if hasattr(self, "epoch_sec_spin")
                else 30.0
            )
            reference = parse_reference_hypnogram(file_path, epoch_sec=epoch_len)

            # Check if lengths match
            if self.review_state.predictions is not None:
                if len(reference) != len(self.review_state.predictions):
                    response = QMessageBox.question(
                        self,
                        "Length Mismatch",
                        f"Reference hypnogram has {len(reference)} epochs but current predictions have {len(self.review_state.predictions)} epochs.\n\n"
                        "Do you want to continue anyway? (Shorter will be padded, longer will be truncated)",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if response != QMessageBox.StandardButton.Yes:
                        return

            # Set reference on canvas
            self.hypnogram_canvas.set_reference_hypnogram(reference)

            # Re-plot
            if self.review_state.predictions is not None:
                epoch_sec = self.epoch_sec_spin.value()
                self.hypnogram_canvas.plot_hypnogram(
                    self.review_state.predictions,
                    epoch_sec,
                    self.review_state.probabilities,
                    self.review_state.confidences,
                )

            # Store the reference for the Validation tab and update UI.
            self.review_state.reference_predictions = np.asarray(reference)
            self.clear_reference_button.setVisible(True)
            self.clear_reference_button.setEnabled(True)
            self.compare_mode_button.setText("Reference Loaded ✓")
            self.compare_mode_button.setStyleSheet(
                "background-color: #2E7D32; color: white;"
            )
            self._refresh_validation_panels()

            # Calculate and display agreement
            agreement = self.hypnogram_canvas.get_reference_agreement()
            if agreement is not None:
                self.log(
                    f"Reference hypnogram loaded: {len(reference)} epochs, {agreement:.1%} agreement",
                    logging.INFO,
                )
            else:
                self.log(
                    f"Reference hypnogram loaded: {len(reference)} epochs", logging.INFO
                )

        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
            self.log(f"Failed to load reference: {e}", logging.ERROR)
        except Exception as e:
            QMessageBox.warning(
                self, "Error", f"Failed to load reference hypnogram:\n{str(e)}"
            )
            self.log(f"Failed to load reference: {e}", logging.ERROR)

    def _parse_xml_reference(self, file_path: str) -> np.ndarray:
        """Parse a Profusion/NSRR XML hypnogram into per-epoch stage indices.

        Thin wrapper around
        :func:`spectra.review.reference.parse_xml_reference` that supplies
        the configured epoch length.

        Args:
            file_path: Path to the XML annotation file.

        Returns:
            A 1-D ``np.ndarray`` of AASM stage indices (0=Wake..4=REM, -1 for
            unscored epochs).
        """
        from spectra.review.reference import parse_xml_reference

        epoch_len = 30.0
        if hasattr(self, "epoch_sec_spin"):
            epoch_len = float(self.epoch_sec_spin.value()) or 30.0
        return parse_xml_reference(file_path, epoch_sec=epoch_len)

    def _refresh_validation_panels(self) -> None:
        """Push current predictions/probabilities + reference into the panels.

        Safe to call any time: when no reference is loaded the panels reset to
        their empty state.
        """
        if not hasattr(self, "reference_agreement_panel"):
            return
        reference = self.review_state.reference_predictions
        self.reference_agreement_panel.set_data(
            self.review_state.predictions, reference
        )
        self.calibration_panel.set_data(self.review_state.probabilities, reference)

    def _clear_reference_hypnogram(self):
        """Clear the reference hypnogram comparison."""
        self.hypnogram_canvas.set_reference_hypnogram(None)
        self.review_state.reference_predictions = None
        self._refresh_validation_panels()

        # Re-plot
        if self.review_state.predictions is not None:
            epoch_sec = self.epoch_sec_spin.value()
            self.hypnogram_canvas.plot_hypnogram(
                self.review_state.predictions,
                epoch_sec,
                self.review_state.probabilities,
                self.review_state.confidences,
            )

        # Update UI
        self.clear_reference_button.setVisible(False)
        self.compare_mode_button.setText("Load Reference...")
        self.compare_mode_button.setStyleSheet("")

        self.log("Reference hypnogram cleared", logging.INFO)

    def _show_epoch_details(
        self,
        epoch_idx: int,
        stage: StageInput,
        confidence: float | np.floating | None,
        probs: np.ndarray | None = None,
    ):
        """Show detailed information about a clicked epoch."""
        self._select_epoch(epoch_idx)
        model_stage = (
            int(self.review_state.base_predictions[epoch_idx])
            if self.review_state.base_predictions is not None
            and epoch_idx < len(self.review_state.base_predictions)
            else _stage_to_int(stage)
        )
        final_stage = (
            int(self.review_state.predictions[epoch_idx])
            if self.review_state.predictions is not None
            and epoch_idx < len(self.review_state.predictions)
            else model_stage
        )
        # Show the dialog with epoch details
        self.epoch_details_dialog.show_epoch(
            epoch_idx,
            _normalize_stage_label(model_stage),
            confidence,
            probs,
            final_stage=_normalize_stage_label(final_stage),
        )
        self.epoch_details_dialog.show()
        self.epoch_details_dialog.raise_()
        self.epoch_details_dialog.activateWindow()

    def _highlight_epoch_on_canvas(self, epoch_idx):
        """Highlight a specific epoch on the hypnogram canvas (callback from dialog navigation)."""
        # This could be used to visually highlight the current epoch on the canvas
        # For now, we just ensure the canvas is visible
        self._select_epoch(epoch_idx)
        self._show_review_view("Hypnogram")


class QTextEditLogger(logging.Handler):
    """Custom logging handler that writes to QTextEdit."""

    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        """Emit log record to text edit."""
        msg = self.format(record)
        try:
            self.text_edit.append(msg)
            # Auto-scroll to bottom
            self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        except RuntimeError:
            # The QTextEdit can be deleted during shutdown/re-init; ignore late logs.
            return


def initialize_matplotlib_backend():
    """Initialize matplotlib with Qt backend after QApplication is created."""
    global FigureCanvas, NavigationToolbar, HypnogramCanvas
    import matplotlib  # pyright: ignore[reportMissingModuleSource]

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qt import (  # pyright: ignore[reportMissingModuleSource]
        NavigationToolbar2QT,  # pyright: ignore[reportMissingModuleSource]
    )
    from matplotlib.backends.backend_qtagg import (  # pyright: ignore[reportMissingModuleSource]
        FigureCanvasQTAgg,  # pyright: ignore[reportMissingModuleSource]
    )

    FigureCanvas = FigureCanvasQTAgg
    NavigationToolbar = NavigationToolbar2QT
    HypnogramCanvas = create_hypnogram_canvas_class()


def main():
    """Main entry point for GUI application."""
    # CRITICAL: Required for Windows PyInstaller to prevent recursive spawning
    multiprocessing.freeze_support()

    # Check if QApplication already exists (e.g., from bootstrap launcher)
    app = QApplication.instance()
    if app is None:
        # No existing instance, create new one
        app = QApplication(sys.argv)
        app.setApplicationName("SPECTRA")
        should_exec = True  # We created it, so we'll manage the event loop
    else:
        # Reuse existing QApplication from bootstrap
        should_exec = False  # Bootstrap will manage the event loop

    # Initialize matplotlib backend AFTER QApplication is created (critical for macOS)
    initialize_matplotlib_backend()

    # Set application style
    qt_app = cast(QApplication, app)
    qt_app.setStyle("Fusion")

    # Create and show main window
    window = InferenceGUI()

    window.show()

    # Only call exec() if we created the QApplication
    if should_exec:
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
