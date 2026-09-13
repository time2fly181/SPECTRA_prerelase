"""Utilities for deriving montage signatures from channel lists.

This module provides functions to identify and hash channel montages for
domain identification and tracking.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

__all__ = ["derive_montage_signature"]


_EOG_LEFT_TOKENS = {"L", "LEFT", "LEOG", "LOC"}
_EOG_RIGHT_TOKENS = {"R", "RIGHT", "REOG", "ROC"}
_EOG_HORIZONTAL_TOKENS = {"H", "HOR", "HORIZONTAL"}
_EMG_CHIN_TOKENS = {"CHIN", "SUBMENTAL", "MENTAL", "SUB"}
_REFERENTIAL_TOKENS = {"A1", "A2", "M1", "M2", "REF", "REF1", "REF2", "CZ"}


def _tokens_from_label(label: str) -> list[str]:
    """Extract alphanumeric tokens from a channel label."""
    cleaned = re.sub(r"[^A-Z0-9/\-]+", "", label.upper())
    if not cleaned:
        return [""]
    pieces = re.split(r"[\-/]", cleaned)
    tokens = []
    for piece in pieces:
        if not piece:
            continue
        parts = re.findall(r"[A-Z]+[0-9]*", piece)
        if not parts:
            tokens.append(piece)
            continue
        for part in parts:
            if part in {"EEG", "PSG"}:
                continue
            tokens.append(part)
    return tokens or [cleaned]


def _canonical_montage_name(label: str) -> str:
    """Derive a canonical name for a channel to identify montage type.

    This function collapses channel labels into coarse categories for montage
    identification (e.g., "C4-A1" -> "C4:REF", "EOG horizontal" -> "EOG:REF").

    Args:
        label: Channel label string

    Returns:
        Canonical name in format "BASE:REFERENCE_TYPE"
    """
    tokens = _tokens_from_label(label)
    if not tokens:
        return "UNK:REF"

    base = "".join(tokens)
    token_set = set(tokens)

    eog_hint = (
        "EOG" in token_set
        or token_set & _EOG_LEFT_TOKENS
        or token_set & _EOG_RIGHT_TOKENS
        or token_set & _EOG_HORIZONTAL_TOKENS
        or any(tok.startswith("EOG") for tok in tokens)
        or any("EOG" in tok for tok in tokens)
        or (tokens and tokens[0] in {"LOC", "ROC"})
    )

    if eog_hint:
        base = "EOG"
    elif "ECG" in token_set or "EKG" in token_set:
        base = "ECG"
    elif any("EMG" in tok for tok in tokens) and (
        token_set & _EMG_CHIN_TOKENS
        or any(tok.startswith("CHEMG") for tok in tokens)
        or any(tok.startswith("CHIN") for tok in tokens)
        or any("CHIN" in tok for tok in tokens)
        or any("SUB" in tok for tok in tokens)
        or any("MENTAL" in tok for tok in tokens)
    ):
        base = "EMG"
    elif tokens:
        first = tokens[0]
        if re.match(r"^[A-Z]+[0-9]+Z?$", first):
            base = first

    ref_hint = "REF"
    if len(tokens) > 1:
        second = tokens[1]
        if second not in _REFERENTIAL_TOKENS:
            ref_hint = "BIP"
    if any(tok.startswith("REF") for tok in tokens):
        ref_hint = "REF"

    return f"{base}:{ref_hint}"


def derive_montage_signature(channels: Iterable[str]) -> str:
    """Derive a unique signature for a channel montage.

    This creates a deterministic hash based on the canonical names of all
    channels in the montage, allowing datasets with identical montages to
    be grouped together for domain adaptation.

    Args:
        channels: Iterable of channel names

    Returns:
        Montage signature string in format "CANONICAL_LIST|HASH"

    Example:
        >>> channels = ["C3-A2", "C4-A1", "EOG-L", "EMG chin"]
        >>> derive_montage_signature(channels)
        'C3:REF;C4:REF;EMG:REF;EOG:REF|a1b2c3d4e5f6'
    """
    canonical = [_canonical_montage_name(ch) for ch in channels]
    canonical.sort()
    if not canonical:
        return "NONE"
    signature = ";".join(canonical)
    digest = hashlib.blake2s(signature.encode("utf-8"), digest_size=6).hexdigest()
    return f"{signature}|{digest}"
