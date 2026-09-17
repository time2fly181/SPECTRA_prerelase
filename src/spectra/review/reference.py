"""Parse reference hypnograms into per-epoch AASM stage indices.

Supports the formats the inference GUI accepts: ``.npy``/``.npz`` arrays,
``.csv``/``.txt`` stage columns, and Compumedics Profusion / NSRR ``.xml``
annotations (both the time-based ``ScoredEvent`` shape and the sequential
``SleepStage`` shape).

The public parsers return a 1-D ``int64`` array in Wake/N1/N2/N3/REM order
(``0`` through ``4``), with ``-1`` for unscored epochs. XML numeric NSRR codes
map N4 to N3 and REM code 5 to index 4. Array inputs use the five-class indices
directly. Callers must align the reference with the EDF epoch grid.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

__all__ = ["parse_reference_hypnogram", "parse_xml_reference"]

# Small stage-name map used for CSV/TXT columns.
_CSV_STAGE_MAP: dict[str, int] = {
    "W": 0,
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    "R": 4,
}

# NSRR numeric stage codes -> AASM 5-class index (N4 -> N3, unscored -> -1).
_NSRR_TO_AASM: dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4,
    6: -1,
    7: -1,
    9: -1,
}

# Free-form stage strings -> AASM index (covers Profusion EventConcepts).
_XML_STAGE_STR_MAP: dict[str, int] = {
    "W": 0,
    "WAKE": 0,
    "N1": 1,
    "S1": 1,
    "STAGE 1": 1,
    "STAGE 1 SLEEP": 1,
    "N2": 2,
    "S2": 2,
    "STAGE 2": 2,
    "STAGE 2 SLEEP": 2,
    "N3": 3,
    "S3": 3,
    "S4": 3,
    "N4": 3,
    "STAGE 3": 3,
    "STAGE 4": 3,
    "STAGE 3 SLEEP": 3,
    "STAGE 4 SLEEP": 3,
    "REM": 4,
    "R": 4,
    "REM SLEEP": 4,
}


def parse_reference_hypnogram(
    path: str | Path, *, epoch_sec: float = 30.0
) -> np.ndarray:
    """Parse a reference hypnogram file into AASM stage indices.

    Args:
        path: Path to a ``.npy``/``.npz``/``.csv``/``.txt``/``.xml`` file.
        epoch_sec: Epoch length in seconds, used only for time-based XML.

    Returns:
        1-D ``int64`` array of stage indices (``-1`` for unscored epochs).

    Raises:
        ValueError: If the extension is unsupported or no stage data is found.
    """
    file_path = Path(path)
    ext = file_path.suffix.lower()

    if ext == ".npy":
        reference = np.load(file_path)
    elif ext == ".npz":
        reference = _parse_npz(file_path)
    elif ext in (".csv", ".txt"):
        reference = _parse_csv_or_text(file_path)
    elif ext == ".xml":
        reference = parse_xml_reference(file_path, epoch_sec=epoch_sec)
    else:
        raise ValueError(f"Unsupported reference format: {ext}")

    reference = np.asarray(reference).ravel()
    if reference.size == 0:
        raise ValueError("No valid stage data found in reference file.")
    return reference.astype(np.int64, copy=False)


def _parse_npz(file_path: Path) -> np.ndarray:
    """Extract the stage array from a ``.npz`` archive by common key names."""
    data = np.load(file_path)
    for key in ("predictions", "hypnogram", "stages", "y", "labels"):
        if key in data:
            return np.asarray(data[key])
    return np.asarray(data[list(data.keys())[0]])


def _parse_csv_or_text(file_path: Path) -> np.ndarray:
    """Parse a stage column from a CSV/TXT file (pandas, text fallback)."""
    try:
        import pandas as pd  # pyright: ignore[reportMissingModuleSource]

        df = pd.read_csv(file_path)
        stage_col = None
        for col in (
            "stage",
            "Stage",
            "prediction",
            "Prediction",
            "label",
            "Label",
            "sleep_stage",
        ):
            if col in df.columns:
                stage_col = df[col]
                break
        if stage_col is None:
            stage_col = df.iloc[:, 0]

        if stage_col.dtype == object:
            return np.array(
                [
                    _CSV_STAGE_MAP.get(
                        str(s).strip(),
                        int(s) if str(s).strip().isdigit() else 0,
                    )
                    for s in stage_col
                ]
            )
        return stage_col.values.astype(int)
    except Exception:
        return _parse_text_fallback(file_path)


def _parse_text_fallback(file_path: Path) -> np.ndarray:
    """Simple line-based fallback when pandas cannot read the file."""
    with open(file_path) as f:
        lines = f.readlines()
    reference: list[int] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        stage_str = parts[-1].strip() if len(parts) > 1 else line
        if stage_str in _CSV_STAGE_MAP:
            reference.append(_CSV_STAGE_MAP[stage_str])
        elif stage_str.isdigit():
            reference.append(int(stage_str))
    return np.array(reference, dtype=np.int64)


def parse_xml_reference(path: str | Path, *, epoch_sec: float = 30.0) -> np.ndarray:
    """Parse a Profusion/NSRR XML hypnogram into per-epoch stage indices.

    Supports the two XML shapes Compumedics Profusion exports:

    1. NSRR ``ScoredEvent`` format (time-based): elements whose ``EventType``
       contains "Stages", carrying an ``EventConcept`` (e.g. ``Stage 1 sleep|1``)
       plus ``Start``/``Duration`` in seconds.
    2. Native sequential format (epoch-based): a ``SleepStages`` container of
       ``SleepStage`` codes, one per epoch.

    Args:
        path: Path to the XML annotation file.
        epoch_sec: Epoch length in seconds (defaults to 30.0 when falsy).

    Returns:
        1-D ``int64`` array of AASM stage indices (``-1`` for unscored epochs).
    """
    epoch_len = float(epoch_sec) or 30.0

    def stage_from_concept(concept: str | None) -> int:
        if not concept:
            return -1
        code = concept.split("|")[-1].strip() if "|" in concept else concept.strip()
        try:
            return _NSRR_TO_AASM.get(int(code), -1)
        except ValueError:
            return _XML_STAGE_STR_MAP.get(concept.strip().upper(), -1)

    tree = ET.parse(str(path))
    root = tree.getroot()

    # --- Format 1: NSRR ScoredEvent (time-based) ---
    events: list[tuple[float, float, int]] = []
    stage_events_found = False
    for event in root.findall(".//ScoredEvent"):
        event_type = event.findtext("EventType")
        if event_type is None or "Stages" not in event_type:
            continue
        concept = event.findtext("EventConcept")
        start_text = event.findtext("Start")
        duration_text = event.findtext("Duration")
        if concept is None or start_text is None or duration_text is None:
            continue
        try:
            start_sec = float(start_text)
            duration_sec = float(duration_text)
        except (TypeError, ValueError):
            continue
        stage_events_found = True
        events.append((start_sec, duration_sec, stage_from_concept(concept)))

    if stage_events_found:
        n_epochs = max(
            int(math.ceil((start + dur) / epoch_len)) for start, dur, _ in events
        )
        labels = np.full(n_epochs, -1, dtype=np.int64)
        for start_sec, duration_sec, stage in events:
            if stage < 0:
                continue
            start_epoch = max(0, int(math.floor(start_sec / epoch_len)))
            end_epoch = min(
                n_epochs, int(math.ceil((start_sec + duration_sec) / epoch_len))
            )
            if start_epoch < end_epoch:
                labels[start_epoch:end_epoch] = stage
        return labels

    # --- Format 2: native sequential SleepStage (epoch-based) ---
    container = root.find(".//SleepStages")
    if container is not None:
        stage_elements = container.findall("SleepStage")
    else:
        stage_elements = root.findall(".//SleepStage")

    seq_labels: list[int] = []
    for stage_elem in stage_elements:
        text = (stage_elem.text or "").strip()
        if not text:
            seq_labels.append(-1)
            continue
        try:
            seq_labels.append(_NSRR_TO_AASM.get(int(text), -1))
        except ValueError:
            seq_labels.append(_XML_STAGE_STR_MAP.get(text.upper(), -1))

    return np.array(seq_labels, dtype=np.int64)
