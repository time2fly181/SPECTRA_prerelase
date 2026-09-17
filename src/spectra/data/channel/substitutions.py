"""Channel substitution and aliasing for PSG recordings.

This module handles common channel naming variations and provides
reference substitution (e.g., A1/A2 ↔ M1/M2) for missing channels.

Explicit derivation helpers also support algebraic rereferencing. The EDF
scoring path uses direct/substitution matching with rereferencing disabled.
"""

from __future__ import annotations

import re
from typing import Any

from .normalization import infer_channel_type as _infer_channel_type

try:
    from .rereferencing import (
        build_montage_transform,
    )

    HAS_REREFERENCING = True
except ImportError:
    HAS_REREFERENCING = False
    build_montage_transform = None  # type: ignore[assignment]
    np = None  # type: ignore


# Channel reference substitutions
# Format: (canonical_name, possible_aliases)
REFERENCE_SUBSTITUTIONS = {
    # M1/M2 ↔ A1/A2 (mastoid references - same location, different naming)
    "C3-M2": ["C3-A2", "C3_A2", "C3_M2", "EEG C3-M2", "EEG C3-A2"],
    "C4-M1": ["C4-A1", "C4_A1", "C4_M1", "EEG C4-M1", "EEG C4-A1"],
    "F3-M2": ["F3-A2", "F3_A2", "F3_M2", "EEG F3-M2", "EEG F3-A2"],
    "F4-M1": ["F4-A1", "F4_A1", "F4_M1", "EEG F4-M1", "EEG F4-A1"],
    "O1-M2": ["O1-A2", "O1_A2", "O1_M2", "EEG O1-M2", "EEG O1-A2"],
    "O2-M1": ["O2-A1", "O2_A1", "O2_M1", "EEG O2-M1", "EEG O2-A1"],
    "Fp1-M2": [
        "Fp1-A2",
        "Fp1_A2",
        "Fp1_M2",
        "FP1-M2",
        "FP1-A2",
        "FP1_M2",
        "FP1_A2",
        "EEG Fp1-M2",
        "EEG Fp1-A2",
    ],
    "Fp2-M1": [
        "Fp2-A1",
        "Fp2_A1",
        "Fp2_M1",
        "FP2-M1",
        "FP2-A1",
        "FP2_M1",
        "FP2_A1",
        "EEG Fp2-M1",
        "EEG Fp2-A1",
    ],
    "T3-M2": [
        "T3-A2",
        "T3_A2",
        "T3_M2",
        "T7-M2",
        "T7-A2",
        "T7_M2",
        "T7_A2",
        "EEG T3-M2",
        "EEG T3-A2",
    ],
    "T4-M1": [
        "T4-A1",
        "T4_A1",
        "T4_M1",
        "T8-M1",
        "T8-A1",
        "T8_M1",
        "T8_A1",
        "EEG T4-M1",
        "EEG T4-A1",
    ],
    "P3-M2": ["P3-A2", "P3_A2", "P3_M2", "EEG P3-M2", "EEG P3-A2"],
    "P4-M1": ["P4-A1", "P4_A1", "P4_M1", "EEG P4-M1", "EEG P4-A1"],
    "Cz-M1": [
        "Cz-A1",
        "Cz_A1",
        "Cz_M1",
        "CZ-M1",
        "CZ-A1",
        "CZ_M1",
        "CZ_A1",
        "EEG Cz-M1",
        "EEG Cz-A1",
    ],
    "Cz-M2": [
        "Cz-A2",
        "Cz_A2",
        "Cz_M2",
        "CZ-M2",
        "CZ-A2",
        "CZ_M2",
        "CZ_A2",
        "EEG Cz-M2",
        "EEG Cz-A2",
    ],
    # EOG variations
    # LOC (Left Outer Canthus) and ROC (Right Outer Canthus) are the primary canonical forms
    # Both EOG1 and EOG2 map to LOC/ROC respectively, all referenced to A2
    # Generic "EOG" or "PSG_EOG" (single channel) maps to both LOC-A2 and ROC-A2 (duplicated)
    "LOC-A2": [
        "LOC-M2",
        "LOC",
        "LOC-A2",
        "EOG1-A2",
        "EOG1-M2",
        "EOG1",
        "E1-A2",
        "E1-M2",
        "E1M2",
        "E1",
        "EOG E1-M2",
        "EOG_E1-M2",
        "EOG-E1-M2",
        "EOG E1-A2",
        "EOG_E1-A2",
        "EOG-E1-A2",
        "EOG LOC-M2",
        "EOG LOC",
        "EOG LOC-A2",
        "EOG-H",
        "EOGH",
        "EOG_H",
        "EOG L",
        "EOG_L",
        "EOGL",
        "EOG left",
        "EOG(L)",
        "EOG (L)",  # SHHS dataset
        "lefteye",
        "LeftEye",
        "LEFTEYE",
        "Left Eye",
        "LEFT EYE",
        "EOG-L",
        "PSG_EOG",
        "PSG EOG",
        "EOG",
    ],
    "ROC-A2": [
        "ROC-M2",
        "ROC",
        "ROC-A2",
        "EOG2-A2",
        "EOG2-M2",
        "EOG2",
        "E2-A2",
        "E2-M2",
        "E2M2",
        "E2",
        "EOG E2-M2",
        "EOG_E2-M2",
        "EOG-E2-M2",
        "EOG E2-M1",
        "EOG_E2-M1",
        "EOG-E2-M1",
        "EOG E2-A2",
        "EOG_E2-A2",
        "EOG-E2-A2",
        "EOG E2-A1",
        "EOG_E2-A1",
        "EOG-E2-A1",
        "ROC-M1",
        "ROC-A1",
        "EOG ROC-M2",
        "EOG ROC",
        "EOG ROC-A2",
        "EOG ROC-M1",
        "EOG-V",
        "EOGV",
        "EOG_V",
        "EOG R",
        "EOG_R",
        "EOGR",
        "EOG right",
        "EOG(R)",
        "EOG (R)",  # SHHS dataset
        "righteye",
        "RightEye",
        "RIGHTEYE",
        "Right Eye",
        "RIGHT EYE",
        "EOG-R",
        "PSG_EOG",
        "PSG EOG",
        "EOG",
    ],
    # Backward compatibility: Keep separate E1-M2, E2-M1, E2-M2 for datasets using E# nomenclature
    "E1-M2": ["EOG1-M2", "E1-A2", "E1M2", "LOC-M2", "EOG LOC-M2"],
    "E2-M1": ["EOG2-M1", "E2-A1", "ROC-M1", "EOG ROC-M1"],
    "E2-M2": ["EOG2-M2", "E2-A2", "E2M2", "ROC-M2", "EOG ROC-M2"],
    # Backward compatibility: EOG2-A1 (ROC referenced to A1 instead of A2)
    "EOG2-A1": [
        "ROC-A1",
        "EOG2-M1",
        "E2-M1",
        "E2-A1",
        "EOG E2-M1",
        "EOG_E2-M1",
        "EOG-E2-M1",
        "EOG E2-A1",
        "EOG_E2-A1",
        "EOG-E2-A1",
        "EOG ROC-M1",
    ],
    # EEG channel variations - bidirectional M1/M2 ↔ A1/A2 equivalents
    "C3-A2": [
        "C3A2",
        "C3_A2",
        "C3 A2",
        "EEG C3-A2",
        "EEG C3A2",
        "C3-M2",
        "C3_M2",
        "C3M2",
        "EEG C3-M2",
        "EEG C3_M2",
        "EEG(sec)",
        "EEG (sec)",
    ],  # SHHS dataset: EEG(sec) is C3-A2
    "C4-A1": [
        "C4A1",
        "C4_A1",
        "C4 A1",
        "EEG C4-A1",
        "EEG C4A1",
        "C4-M1",
        "C4_M1",
        "C4M1",
        "EEG C4-M1",
        "EEG C4_M1",
        "EEG3",  # MESA dataset: EEG3 is C4-M1
        "EEG",
    ],  # SHHS dataset: generic EEG is C4-A1
    "F3-A2": [
        "F3A2",
        "F3_A2",
        "F3 A2",
        "EEG F3-A2",
        "EEG F3A2",
        "F3-M2",
        "F3_M2",
        "F3M2",
        "EEG F3-M2",
        "EEG F3_M2",
    ],
    "F4-A1": [
        "F4A1",
        "F4_A1",
        "F4 A1",
        "EEG F4-A1",
        "EEG F4A1",
        "F4-M1",
        "F4_M1",
        "F4M1",
        "EEG F4-M1",
        "EEG F4_M1",
    ],
    "O1-A2": [
        "O1A2",
        "O1_A2",
        "O1 A2",
        "EEG O1-A2",
        "EEG O1A2",
        "O1-M2",
        "O1_M2",
        "O1M2",
        "EEG O1-M2",
        "EEG O1_M2",
    ],
    "O2-A1": [
        "O2A1",
        "O2_A1",
        "O2 A1",
        "EEG O2-A1",
        "EEG O2A1",
        "O2-M1",
        "O2_M1",
        "O2M1",
        "EEG O2-M1",
        "EEG O2_M1",
    ],
    "P3-A2": [
        "P3A2",
        "P3_A2",
        "P3 A2",
        "EEG P3-A2",
        "EEG P3A2",
        "P3-M2",
        "P3_M2",
        "P3M2",
        "EEG P3-M2",
        "EEG P3_M2",
    ],
    "P4-A1": [
        "P4A1",
        "P4_A1",
        "P4 A1",
        "EEG P4-A1",
        "EEG P4A1",
        "P4-M1",
        "P4_M1",
        "P4M1",
        "EEG P4-M1",
        "EEG P4_M1",
    ],
    # Midline EEG channels (central/frontal/occipital)
    # MESA dataset uses EEG1, EEG2 for midline channels
    "Fz-Cz": [
        "FZ-CZ",
        "Fz-Cz",
        "FzCz",
        "Fz_Cz",
        "FZ_CZ",
        "EEG1",
    ],  # MESA: EEG1 is Fz-Cz
    "Cz-Oz": [
        "CZ-OZ",
        "Cz-Oz",
        "CzOz",
        "Cz_Oz",
        "CZ_OZ",
        "EEG2",
    ],  # MESA: EEG2 is Cz-Oz
    # EMG variations
    # Note: EMG1 and EMG2 standalone are NOT aliases for EMG1-EMG2 bipolar
    # They should be rereferenced: EMG1-EMG2 = ChEMG1 - ChEMG2
    "EMG1-EMG2": [
        "Chin1-Chin2",
        "Chin",
        "EMG Chin",
        "Chin EMG",
        "L Chin",
        "R Chin",
        "LChin",
        "RChin",
        "L_Chin",
        "R_Chin",
        "Left Chin",
        "Right Chin",
        "LeftChin",
        "RightChin",
        "Submental",
        "EMG submental",
        "EMG_submental",
        "EMG subment",
        "EMG_subment",
        "EMG_chin",
        "EMG chin",
        "EMG",
        "emg",
        "Emg",
        "Ment1-Ment2",
        "MENT1-MENT2",
        "MENTALIS1-MENTALIS2",
    ],
    # ECG variations
    # Note: ECG-II (Lead II) is the most common single-lead ECG configuration for sleep studies
    "ECG1-ECG2": [
        "ECG",
        "EKG",
        "ECG1",
        "ECG2",
        "ECG I",
        "ECG II",
        "ECG-II",
        "ECGII",
        "ECG 1",
        "ECG 2",
        "ECG_1",
        "ECG_2",
        "ecg",
        "Ecg",
        "ekg",
        "Ekg",
    ],
}


def normalize_for_substitution(ch_name: str) -> str:
    """Normalize channel name for substitution matching.

    Args:
        ch_name: Original channel name (e.g., 'C3-A2', 'eeg/F4_M1', 'eog/EOG1')

    Returns:
        Normalized name (uppercase, spaces removed), or empty string if should be dropped
    """
    # Remove common prefixes (apply multiple passes to handle "PSG_EEG F3" -> "EEG F3" -> "F3")
    name = ch_name.strip()

    # Handle type/ or type_ prefix notation (e.g., "eeg/F4_M1", "eeg_F4_M1", "eog/EOG1", "eog_EOG1", "emg/EMG", "emg_EMG")
    type_prefix_match = re.match(
        r"^(eeg|eog|emg|ecg|ekg)[/_](.+)$", name, flags=re.IGNORECASE
    )
    if type_prefix_match:
        # Extract the channel name after the type prefix
        type_prefix = type_prefix_match.group(1).lower()
        name = type_prefix_match.group(2)

        # Special case: Handle mislabeled channels like "emg/ECG" (should be ecg/ECG)
        # Detect if the channel name contradicts the type prefix
        name_upper = name.upper()
        if type_prefix == "emg" and (
            name_upper.startswith("ECG") or name_upper.startswith("EKG")
        ):
            # Channel is actually ECG despite emg/ prefix - this is correct, just strip prefix
            pass
        elif type_prefix == "emg" and name_upper.startswith("EOG"):
            # Channel is actually EOG despite emg/ prefix
            pass
        elif type_prefix in ("ecg", "ekg") and name_upper.startswith("EMG"):
            # Channel is actually EMG despite ecg/ekg prefix
            pass

        # Convert underscores to hyphens for bipolar notation if it looks like a bipolar channel
        if "_" in name and "-" not in name:
            parts = name.split("_", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                # Check if both parts look like electrode names
                if (
                    any(c.isalpha() for c in parts[0])
                    and any(c.isalpha() for c in parts[1])
                    and not parts[0].upper().startswith("EOG")
                    and not parts[0].upper().startswith("EMG")
                    and not parts[0].upper().startswith("ECG")
                ):
                    name = name.replace("_", "-", 1)

    # Check if this is "EOG vertical" or "EOG horizontal" which should be dropped entirely
    eog_drop_pattern = re.match(
        r"^(PSG_)?(EEG_)?EOG[\s_]+(VERTICAL|HORIZONTAL)\s*$", name, flags=re.IGNORECASE
    )
    if eog_drop_pattern:
        return ""  # Signal this channel should be dropped

    # Special handling: preserve EOG1, EOG2, LOC, ROC
    # Remove PSG/EEG prefixes first
    name = re.sub(r"^PSG[\s_-]+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^EEG[\s_-]+", "", name, flags=re.IGNORECASE)

    # Normalize EOG channel patterns:
    # "EOG E1-M2", "EOG E1-A2" -> "E1-M2", "E1-A2"
    # "EOG 1", "EOG_1", "EOG-1" -> "EOG1"
    # "EOG LOC", "EOG_LOC" -> "LOC"
    # "EOG 2", "EOG_2", "EOG-2" -> "EOG2"
    # "EOG ROC", "EOG_ROC" -> "ROC"

    # First, handle "EOG E1-M2" and similar patterns (keep the full derivation)
    if re.match(r"^EOG[\s_-]+E[12][\s_-]", name, flags=re.IGNORECASE):
        # "EOG E1-M2" -> "E1-M2", "EOG_E2-A1" -> "E2-A1"
        name = re.sub(r"^EOG[\s_-]+", "", name, flags=re.IGNORECASE)
    elif re.match(r"^EOG[\s_-]*(1|2)\b", name, flags=re.IGNORECASE):
        # EOG with number -> EOG1 or EOG2
        name = re.sub(
            r"^EOG[\s_-]*(1|2)\b",
            lambda m: "EOG" + m.group(1),
            name,
            flags=re.IGNORECASE,
        )
    elif re.match(r"^EOG[\s_-]*(LOC|ROC|L|R)\b", name, flags=re.IGNORECASE):
        # EOG with LOC/ROC/L/R -> just LOC/ROC
        match = re.search(r"^EOG[\s_-]*(LOC|ROC|L|R)\b", name, flags=re.IGNORECASE)
        if match:
            suffix = match.group(1).upper()
            # Normalize L->LOC, R->ROC
            if suffix == "L":
                name = "LOC" + name[match.end() :]
            elif suffix == "R":
                name = "ROC" + name[match.end() :]
            else:
                name = suffix + name[match.end() :]

    prev_name = ""
    while prev_name != name:
        prev_name = name
        # Remove common prefixes (space-separated)
        name = re.sub(r"^EEG\s+", "", name, flags=re.IGNORECASE)
        # Skip EOG prefix removal - handled above
        # name = re.sub(r'^EOG\s+', '', name, flags=re.IGNORECASE)
        name = re.sub(r"^EMG\s+", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^ECG\s+", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^PSG\s+", "", name, flags=re.IGNORECASE)

        # Remove common prefixes (underscore-separated)
        name = re.sub(r"^EEG_", "", name, flags=re.IGNORECASE)
        # Skip EOG prefix removal - handled above
        # name = re.sub(r'^EOG_', '', name, flags=re.IGNORECASE)
        name = re.sub(r"^EMG_", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^ECG_", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^PSG_", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^EEG-", "", name, flags=re.IGNORECASE)
        # Skip EOG prefix removal - handled above
        # name = re.sub(r'^EOG-', '', name, flags=re.IGNORECASE)
        name = re.sub(r"^EMG-", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^ECG-", "", name, flags=re.IGNORECASE)
        name = re.sub(r"^PSG-", "", name, flags=re.IGNORECASE)

    # Strip descriptive terms for EOG: "vertical", "horizontal"
    name = re.sub(r"\s*(VERTICAL|HORIZONTAL)\s*", "", name, flags=re.IGNORECASE)

    # Normalize spaces and case
    name = name.replace(" ", "").upper()
    # Handle channels recorded as "ChEMG1", "ChEMG2", etc.
    if name.startswith("CHEMG"):
        name = name[2:]

    # Convert forward slash to hyphen for bipolar notation (e.g., "F4/M1" -> "F4-M1")
    # This handles datasets like HMC that use forward slash separators
    if "/" in name and "-" not in name:
        parts = name.split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            # Check if both parts look like electrode names (contain letters)
            if any(c.isalpha() for c in parts[0]) and any(
                c.isalpha() for c in parts[1]
            ):
                name = name.replace("/", "-", 1)

    # Keep normalized name including generic "EOG"
    return name


# Build reverse lookup (alias -> canonical) using NORMALIZED aliases
_ALIAS_TO_CANONICAL = {}
for canonical, aliases in REFERENCE_SUBSTITUTIONS.items():
    for alias in aliases:
        normalized_alias = normalize_for_substitution(alias)
        if normalized_alias:  # Skip empty strings
            _ALIAS_TO_CANONICAL[normalized_alias] = canonical


def find_canonical_name(ch_name: str) -> str:
    """Find canonical name for a channel, handling aliases.

    Args:
        ch_name: Channel name (possibly an alias)

    Returns:
        Canonical name, or original if no match found
    """
    normalized = normalize_for_substitution(ch_name)

    # Check aliases first (handles cases where a key is also an alias for another key)
    # For example, C3-M2 is a key, but it's also an alias for C3-A2
    if normalized in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[normalized]

    # Check if already canonical
    if ch_name in REFERENCE_SUBSTITUTIONS:
        return ch_name

    # Not found - return original
    return ch_name


def find_substitute_channel(
    wanted_ch: str, available_channels: list[str]
) -> tuple[str, str] | None:
    """Find a substitute channel from available channels.

    Args:
        wanted_ch: Desired channel name
        available_channels: List of available channel names

    Returns:
        Tuple of (available_channel_name, substitution_note) if found, None otherwise
    """
    # Normalize wanted channel to canonical form
    canonical = find_canonical_name(wanted_ch)

    # Build normalized lookup
    available_normalized = {
        normalize_for_substitution(ch): ch for ch in available_channels
    }

    # Check if canonical version exists
    if canonical in REFERENCE_SUBSTITUTIONS:
        # Get all possible aliases for this canonical channel
        aliases = REFERENCE_SUBSTITUTIONS[canonical]

        for alias in aliases:
            norm_alias = normalize_for_substitution(alias)
            if norm_alias in available_normalized:
                actual_ch = available_normalized[norm_alias]
                note = f"Using {actual_ch} as proxy for {wanted_ch} (same location, different reference)"
                return actual_ch, note

    # Check if it's an alias and the canonical exists
    if canonical != wanted_ch:
        norm_canonical = normalize_for_substitution(canonical)
        if norm_canonical in available_normalized:
            actual_ch = available_normalized[norm_canonical]
            note = f"Using {actual_ch} (canonical form) for {wanted_ch}"
            return actual_ch, note

    return None


def build_substitution_map(
    canonical_channels: list[str], available_channels: list[str]
) -> dict[str, tuple[str, str | None]]:
    """Build a mapping from canonical channels to available channels.

    Args:
        canonical_channels: List of desired canonical channel names
        available_channels: List of channels actually present in the recording

    Returns:
        Dictionary mapping canonical_name -> (actual_channel_name, substitution_note)
        substitution_note is None if it's a direct match, string describing substitution otherwise
    """
    mapping = {}

    # Build normalized lookup for available channels
    available_normalized = {
        normalize_for_substitution(ch): ch for ch in available_channels
    }

    for canonical_ch in canonical_channels:
        norm_canonical = normalize_for_substitution(canonical_ch)

        # Direct match?
        if norm_canonical in available_normalized:
            actual_ch = available_normalized[norm_canonical]
            mapping[canonical_ch] = (actual_ch, None)
            continue

        # Try to find substitute
        substitute = find_substitute_channel(canonical_ch, available_channels)
        if substitute:
            actual_ch, note = substitute
            mapping[canonical_ch] = (actual_ch, note)
        else:
            # No match - will be zero-padded
            mapping[canonical_ch] = (
                None,
                f"Channel {canonical_ch} not available (will be zero-padded)",
            )

    return mapping


def build_substitution_map_with_rereferencing(
    canonical_channels: list[str],
    available_channels: list[str],
    enable_rereferencing: bool = True,
    enable_modality_fallback: bool = True,
    verbose: bool = False,
) -> dict[str, tuple[str | None, str | None, Any | None]]:
    """Build a mapping from canonical channels to available channels with re-referencing.

    This enhanced version first attempts direct matching and alias substitution,
    then optionally uses linear re-referencing to derive missing channels from
    available bipolar pairs. Finally, if modality fallback is enabled, it tries
    alternative channels from the same modality (e.g., frontal if central unavailable).

    Args:
        canonical_channels: List of desired canonical channel names
        available_channels: List of channels actually present in the recording
        enable_rereferencing: If True and numpy is available, attempt algebraic derivation
        enable_modality_fallback: If True, fall back to other channels of same modality
        verbose: If True, print derivation information

    Returns:
        Dictionary mapping canonical_name -> (actual_channel_name, substitution_note, transform_row)
        - actual_channel_name: Direct match or substitute channel name, None if not available
        - substitution_note: None for direct match, string describing substitution/derivation
        - transform_row: None for direct match, 1D numpy array for linear combinations

    Example:
        >>> canonical = ['C4-A1', 'C3-A2']
        >>> available = ['C4-P4', 'P4-A1', 'C3-A2']
        >>> mapping = build_substitution_map_with_rereferencing(canonical, available)
        >>> # C4-A1 can be derived from C4-P4 + P4-A1
        >>> # C3-A2 is directly available
    """
    mapping: dict[str, tuple[str | None, str | None, Any | None]] = {}

    # Build normalized lookup for available channels
    available_normalized = {
        normalize_for_substitution(ch): ch for ch in available_channels
    }

    # Track which source channels have been used (by normalized name)
    used_sources: set = set()

    # First pass: try direct matching and simple substitution
    missing_channels: list[str] = []

    for canonical_ch in canonical_channels:
        norm_canonical = normalize_for_substitution(canonical_ch)

        # Direct match?
        if norm_canonical in available_normalized:
            actual_ch = available_normalized[norm_canonical]
            mapping[canonical_ch] = (actual_ch, None, None)
            used_sources.add(normalize_for_substitution(actual_ch))
            continue

        # Try to find simple substitute (same location, different reference)
        substitute = find_substitute_channel(canonical_ch, available_channels)
        if substitute:
            actual_ch, note = substitute
            # If find_substitute_channel found a match, it means the actual channel is a known
            # alias in REFERENCE_SUBSTITUTIONS, so it's a valid bipolar equivalent.
            # We should ALWAYS use it - don't defer to rereferencing.
            # The substitute is already a properly referenced channel.
            mapping[canonical_ch] = (actual_ch, note, None)
            used_sources.add(normalize_for_substitution(actual_ch))
        else:
            # Mark for re-referencing attempt
            missing_channels.append(canonical_ch)

    # Second pass: try re-referencing for missing channels
    still_missing: list[str] = []
    if (
        missing_channels
        and enable_rereferencing
        and HAS_REREFERENCING
        and build_montage_transform is not None
    ):
        if verbose:
            print(
                f"Attempting re-referencing for {len(missing_channels)} missing channels..."
            )

        try:
            transform, info_list, mask = build_montage_transform(
                missing_channels, available_channels, verbose=verbose
            )

            for i, canonical_ch in enumerate(missing_channels):
                if mask[i] == 1:  # Derivable
                    info = info_list[i]
                    transform_row = transform[i, :]

                    if info.method == "direct":
                        # Direct match via re-referencing parser
                        if info.sources:
                            mapping[canonical_ch] = (
                                info.sources[0],
                                info.description,
                                None,
                            )
                            used_sources.add(
                                normalize_for_substitution(info.sources[0])
                            )
                    else:
                        # Telescoped derivation
                        mapping[canonical_ch] = (
                            None,  # No single source channel
                            info.description,
                            transform_row,
                        )
                else:
                    # Not derivable - mark for modality fallback
                    still_missing.append(canonical_ch)

        except Exception as e:
            if verbose:
                print(f"Re-referencing failed: {e}")
            still_missing = missing_channels[:]
    else:
        still_missing = missing_channels[:]

    # Third pass: modality fallback for still missing channels
    if still_missing and enable_modality_fallback:
        # Determine modality priority lists
        modality_priorities = {
            "eeg": _EEG_PRIORITY,
            "eog": _EOG_PRIORITY,
            "emg": _EMG_PRIORITY,
        }

        for canonical_ch in still_missing:
            ch_type = _infer_channel_type(canonical_ch)
            priority_list = modality_priorities.get(ch_type, [])

            found_fallback = False
            for fallback_ch in priority_list:
                norm_fallback = normalize_for_substitution(fallback_ch)

                # Skip if already used
                if norm_fallback in used_sources:
                    continue

                # Try direct match
                if norm_fallback in available_normalized:
                    actual_ch = available_normalized[norm_fallback]
                    norm_actual = normalize_for_substitution(actual_ch)
                    if norm_actual not in used_sources:
                        mapping[canonical_ch] = (
                            actual_ch,
                            f"Fallback: using {actual_ch} instead of {canonical_ch}",
                            None,
                        )
                        used_sources.add(norm_actual)
                        found_fallback = True
                        break

                # Try substitution for fallback channel
                substitute = find_substitute_channel(fallback_ch, available_channels)
                if substitute:
                    actual_ch, _ = substitute
                    norm_actual = normalize_for_substitution(actual_ch)
                    if norm_actual not in used_sources:
                        mapping[canonical_ch] = (
                            actual_ch,
                            f"Fallback: using {actual_ch} instead of {canonical_ch}",
                            None,
                        )
                        used_sources.add(norm_actual)
                        found_fallback = True
                        break

            if not found_fallback:
                mapping[canonical_ch] = (
                    None,
                    f"Channel {canonical_ch} not available (will be zero-padded)",
                    None,
                )
    else:
        # No modality fallback - mark remaining as unavailable
        for canonical_ch in still_missing:
            mapping[canonical_ch] = (
                None,
                f"Channel {canonical_ch} not available (will be zero-padded)",
                None,
            )

    return mapping


# Preferred channel priority for each modality
# Higher priority channels are selected first
_EEG_PRIORITY = [
    # Central channels are best for sleep staging (bipolar then unipolar, plus MESA/SHHS generic names)
    "C3-M2",
    "C4-M1",
    "C3-A2",
    "C4-A1",
    "EEG3",
    "EEG(sec)",
    "C3",
    "C4",  # EEG3=C4-M1 (MESA), EEG(sec)=C3-A2 (SHHS)
    # Frontal channels (bipolar then unipolar)
    "F3-M2",
    "F4-M1",
    "F3-A2",
    "F4-A1",
    "F3",
    "F4",
    # Occipital channels (bipolar then unipolar)
    "O1-M2",
    "O2-M1",
    "O1-A2",
    "O2-A1",
    "O1",
    "O2",
    # Midline derivations (bipolar then unipolar, plus MESA generic names)
    "Fz-Cz",
    "Cz-Oz",
    "EEG1",
    "EEG2",
    "Cz",
    "Fz",
    "Oz",  # EEG1=Fz-Cz, EEG2=Cz-Oz in MESA
    # Parietal channels (bipolar then unipolar)
    "P3-M2",
    "P4-M1",
    "P3-A2",
    "P4-A1",
    "P3",
    "P4",
    "Pz",
    # Generic fallback (SHHS uses generic "EEG" for C4-A1)
    "EEG",
]
_EOG_PRIORITY = [
    # Left and right EOG
    "LOC-A2",
    "ROC-A2",
    "LOC-M2",
    "ROC-M2",
    "E1-M2",
    "E2-M1",
    "E2-M2",
    "EOG1",
    "EOG2",
]
_EMG_PRIORITY = [
    "EMG1-EMG2",
    "Chin1-Chin2",
    "Chin",
    "EMG",
]


def select_channels_by_modality(
    available_channels: list[str],
    n_eeg: int = 2,
    n_eog: int = 2,
    n_emg: int = 1,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Select channels by modality from available channels.

    Uses existing channel matching logic to find the best channels
    for each modality type.

    Args:
        available_channels: List of channel names available in the recording
        n_eeg: Number of EEG channels to select (default: 2)
        n_eog: Number of EOG channels to select (default: 2)
        n_emg: Number of EMG channels to select (default: 1)

    Returns:
        Tuple of (eeg_channels, eog_channels, emg_channels)
        Each is a list of tuples: (canonical_name, actual_channel_name)
        Canonical names are generic: 'EEG1', 'EEG2', 'EOG1', 'EOG2', 'EMG'
    """
    # Build normalized lookup
    available_normalized = {
        normalize_for_substitution(ch): ch for ch in available_channels
    }

    # Classify all available channels by type
    eeg_available = []
    eog_available = []
    emg_available = []

    for ch in available_channels:
        ch_type = _infer_channel_type(ch)
        if ch_type == "eeg":
            eeg_available.append(ch)
        elif ch_type == "eog":
            eog_available.append(ch)
        elif ch_type == "emg":
            emg_available.append(ch)

    # Select best channels using priority lists
    def select_best(available: list[str], priority: list[str], n: int) -> list[str]:
        """Select best n channels from available using priority ordering."""
        selected = []
        used_normalized = set()

        # First, try to match priority channels
        for prio_ch in priority:
            if len(selected) >= n:
                break
            norm_prio = normalize_for_substitution(prio_ch)

            # Direct match
            if norm_prio in available_normalized:
                actual = available_normalized[norm_prio]
                if actual in available and norm_prio not in used_normalized:
                    selected.append(actual)
                    used_normalized.add(norm_prio)
                    continue

            # Try substitution
            substitute = find_substitute_channel(prio_ch, available)
            if substitute:
                actual, _ = substitute
                norm_actual = normalize_for_substitution(actual)
                if norm_actual not in used_normalized:
                    selected.append(actual)
                    used_normalized.add(norm_actual)

        # If not enough, add remaining channels of this type
        for ch in available:
            if len(selected) >= n:
                break
            norm_ch = normalize_for_substitution(ch)
            if norm_ch not in used_normalized:
                selected.append(ch)
                used_normalized.add(norm_ch)

        return selected

    eeg_selected = select_best(eeg_available, _EEG_PRIORITY, n_eeg)
    eog_selected = select_best(eog_available, _EOG_PRIORITY, n_eog)
    emg_selected = select_best(emg_available, _EMG_PRIORITY, n_emg)

    # Build output with canonical names
    eeg_out = [(f"EEG{i + 1}", ch) for i, ch in enumerate(eeg_selected)]
    eog_out = [(f"EOG{i + 1}", ch) for i, ch in enumerate(eog_selected)]
    emg_out = [("EMG", ch) for ch in emg_selected]

    return eeg_out, eog_out, emg_out


def select_5ch_for_model(
    available_channels: list[str],
) -> tuple[list[tuple[str, str | None]], list[bool]]:
    """Select 5 channels (2 EEG, 2 EOG, 1 EMG) for model input.

    Returns channels in fixed order: [EEG1, EEG2, EOG1, EOG2, EMG]
    with a presence mask indicating which channels were found.

    Args:
        available_channels: List of channel names available in the recording

    Returns:
        Tuple of:
            - channel_map: List of 5 tuples (canonical_name, actual_channel_name or None)
            - presence_mask: List of 5 booleans indicating channel presence
    """
    eeg_out, eog_out, emg_out = select_channels_by_modality(
        available_channels, n_eeg=2, n_eog=2, n_emg=1
    )

    # Build fixed 5-channel output
    # Order: [EEG1, EEG2, EOG1, EOG2, EMG]
    channel_map: list[tuple[str, str | None]] = []
    presence_mask: list[bool] = []

    # EEG1
    if len(eeg_out) > 0:
        channel_map.append(("EEG1", eeg_out[0][1]))
        presence_mask.append(True)
    else:
        channel_map.append(("EEG1", None))
        presence_mask.append(False)

    # EEG2
    if len(eeg_out) > 1:
        channel_map.append(("EEG2", eeg_out[1][1]))
        presence_mask.append(True)
    else:
        channel_map.append(("EEG2", None))
        presence_mask.append(False)

    # EOG1
    if len(eog_out) > 0:
        channel_map.append(("EOG1", eog_out[0][1]))
        presence_mask.append(True)
    else:
        channel_map.append(("EOG1", None))
        presence_mask.append(False)

    # EOG2
    if len(eog_out) > 1:
        channel_map.append(("EOG2", eog_out[1][1]))
        presence_mask.append(True)
    else:
        channel_map.append(("EOG2", None))
        presence_mask.append(False)

    # EMG
    if len(emg_out) > 0:
        channel_map.append(("EMG", emg_out[0][1]))
        presence_mask.append(True)
    else:
        channel_map.append(("EMG", None))
        presence_mask.append(False)

    return channel_map, presence_mask


# Example usage
if __name__ == "__main__":
    # Test substitution
    canonical = ["C3-M2", "C4-M1", "F3-M2", "E1-M2"]
    available = ["C3-A2", "C4-M1", "F3-A2", "LOC-M2", "ExtraChannel"]

    mapping = build_substitution_map(canonical, available)

    print("Channel Mapping:")
    for canon, (actual, note) in mapping.items():
        if actual:
            if note:
                print(f"  {canon} -> {actual} ({note})")
            else:
                print(f"  {canon} -> {actual} (direct match)")
        else:
            print(f"  {canon} -> MISSING ({note})")
