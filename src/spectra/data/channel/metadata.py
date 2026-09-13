"""Utilities for parsing and encoding channel metadata.

This module extracts coarse metadata from channel labels so that models can
reason about sensor type, approximate 10-20 position, and reference montage.
The encoder maps free-form channel names to compact integer identifiers that
can be embedded by the neural network.  ``0`` is reserved as the padding index
for every vocabulary so that absent channels can be masked cleanly.
"""

from __future__ import annotations

import re

__all__ = [
    "TYPE_PAD_IDX",
    "TYPE_UNKNOWN_IDX",
    "TYPE_TO_INDEX",
    "POSITION_PAD_IDX",
    "POSITION_UNKNOWN_IDX",
    "POSITION_TO_INDEX",
    "REFERENCE_PAD_IDX",
    "REFERENCE_UNKNOWN_IDX",
    "REFERENCE_TO_INDEX",
    "encode_channel_metadata_tokens",
]

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

TYPE_PAD_IDX = 0
TYPE_UNKNOWN_IDX = 1
TYPE_TO_INDEX = {
    "eeg": 2,
    "eog": 3,
    "emg": 4,
    "ecg": 5,
    "ekg": 5,
}

POSITION_PAD_IDX = 0
POSITION_UNKNOWN_IDX = 1
POSITION_OTHER_IDX = 2

# Common 10-20 locations and auxiliary sensor aliases.
_POSITION_TOKENS = {
    "fp1": 3,
    "fp2": 4,
    "fpz": 5,
    "f1": 6,
    "f2": 7,
    "f3": 8,
    "f4": 9,
    "f7": 10,
    "f8": 11,
    "fz": 12,
    "fc1": 13,
    "fc2": 14,
    "fc5": 15,
    "fc6": 16,
    "c1": 17,
    "c2": 18,
    "c3": 19,
    "c4": 20,
    "c5": 21,
    "c6": 22,
    "cz": 23,
    "cp1": 24,
    "cp2": 25,
    "cp5": 26,
    "cp6": 27,
    "t1": 28,
    "t2": 29,
    "t3": 30,
    "t4": 31,
    "t5": 32,
    "t6": 33,
    "tp7": 34,
    "tp8": 35,
    "p1": 36,
    "p2": 37,
    "p3": 38,
    "p4": 39,
    "p5": 40,
    "p6": 41,
    "pz": 42,
    "poz": 43,
    "o1": 44,
    "o2": 45,
    "oz": 46,
    "io": 47,
    "m1": 48,
    "m2": 49,
    "a1": 50,
    "a2": 51,
    "loc": 52,
    "roc": 53,
    "le": 54,
    "re": 55,
    "lh": 56,
    "rh": 57,
    "chin": 58,
    "chin1": 59,
    "chin2": 60,
    "leg": 61,
    "abd": 62,
    "thor": 63,
    "belly": 64,
    "airflow": 65,
    "snore": 66,
    "spo2": 67,
    "pleth": 68,
    "nasal": 69,
    "mask": 70,
    "pulse": 71,
    "therm": 72,
    "ecg": 73,
    "ekg": 73,
}

POSITION_TO_INDEX = {token: idx for token, idx in _POSITION_TOKENS.items()}

REFERENCE_PAD_IDX = 0
REFERENCE_UNKNOWN_IDX = 1
REFERENCE_OTHER_IDX = 2
REFERENCE_TO_INDEX = POSITION_TO_INDEX.copy()
REFERENCE_TO_INDEX.update(
    {
        "ref": 74,
        "avg": 75,
        "average": 75,
        "linked": 76,
    }
)

# Aliases to improve matching for common channel descriptions.
_POSITION_ALIASES = {
    "l": "loc",
    "left": "loc",
    "r": "roc",
    "right": "roc",
    "lo": "loc",
    "ro": "roc",
    "locc": "loc",
    "rocc": "roc",
    "lheog": "loc",
    "rheog": "roc",
    "eog-l": "loc",
    "eog-r": "roc",
    "chin3": "chin",
    "chin4": "chin",
}

# Regex for alphanumeric tokens (keeps digits together with letters).
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalise_token(token: str) -> str:
    token = token.lower()
    return _POSITION_ALIASES.get(token, token)


def _infer_type(label: str) -> str:
    norm = label.lower()
    # Check for type/ prefix notation first (e.g., "eeg/F4_M1", "eog/EOG1")
    if norm.startswith("eeg/"):
        return "eeg"
    if norm.startswith("eog/"):
        return "eog"
    if norm.startswith("emg/"):
        return "emg"
    if norm.startswith("ecg/") or norm.startswith("ekg/"):
        return "ecg"
    # Fall back to substring matching
    for key in ("eeg", "eog", "emg", "ecg", "ekg"):
        if key in norm:
            return key
    return "unknown"


def _extract_positions(label: str) -> tuple[str | None, str | None]:
    tokens = [_normalise_token(tok) for tok in _TOKEN_RE.findall(label.lower())]
    positions = [tok for tok in tokens if tok in POSITION_TO_INDEX]
    if not positions:
        return None, None
    if len(positions) == 1:
        return positions[0], None
    return positions[0], positions[1]


def encode_channel_metadata_tokens(label: str) -> tuple[int, int, int]:
    """Return ``(type_idx, position_idx, reference_idx)`` tokens for ``label``."""

    chan_type = _infer_type(label)
    type_idx = TYPE_TO_INDEX.get(chan_type, TYPE_UNKNOWN_IDX)

    position, reference = _extract_positions(label)
    if position is None:
        pos_idx = POSITION_UNKNOWN_IDX
    else:
        pos_idx = POSITION_TO_INDEX.get(position, POSITION_OTHER_IDX)

    if reference is None:
        ref_idx = REFERENCE_UNKNOWN_IDX
    else:
        ref_idx = REFERENCE_TO_INDEX.get(reference, REFERENCE_OTHER_IDX)

    return type_idx, pos_idx, ref_idx
