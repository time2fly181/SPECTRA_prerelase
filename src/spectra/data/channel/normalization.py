"""Utilities for normalising PSG channel labels and signals.

This module provides:
- Channel label normalization for matching and aliasing
- Robust percentile-based signal normalization
"""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "normalize_channel_label",
    "normalize_robust",
    "infer_channel_type",
    "PSGNormalizer",
]


_HORIZONTAL_TOKENS = {"horizontal", "horiz", "hor"}
_EOG_LEFT_HINTS = {
    "l",
    "left",
    "loc",
    "lo",
    "le",
    "lheog",
    "leog",
    "eogl",
    "eog1",
    "e1",
}
_EOG_RIGHT_HINTS = {
    "r",
    "right",
    "roc",
    "ro",
    "re",
    "rheog",
    "reog",
    "eogr",
    "eog2",
    "e2",
}
_EOG_STANDALONE_LEFT = {"loc", "lheog", "leog", "e1", "eogl"}
_EOG_STANDALONE_RIGHT = {"roc", "rheog", "reog", "e2", "eogr"}
_EOG_GENERIC_HINTS = _EOG_LEFT_HINTS | _EOG_RIGHT_HINTS | _HORIZONTAL_TOKENS
_SUBMENTAL_TOKENS = {
    "submental",
    "submentalis",
    "mentalis",
    "ment1",
    "ment2",
    "chin",
    "chin1",
    "chin2",
    "chin3",
    "chin4",
    "chemg",
    "chemg1",
    "chemg2",
    "emg",
    "emg1",
    "emg2",
    "emg3",
}

_CARDIAC_TOKENS = {"ecg", "ekg"}
_CARDIAC_PREFIXES = ("ecg", "ekg")
_PREFIX_DROPS = {"psg", "eeg"}

_SIDE_LEFT_HINTS = {"l", "left", "le", "lt"}
_SIDE_RIGHT_HINTS = {"r", "right", "re", "rt"}
_SIDE_LEFT_DIGITS = {"1", "01"}
_SIDE_RIGHT_DIGITS = {"2", "02"}

_EMG_LEFT_HINTS = {tok for tok in _SUBMENTAL_TOKENS if tok.endswith("1")}
_EMG_RIGHT_HINTS = {tok for tok in _SUBMENTAL_TOKENS if tok.endswith("2")}

_BASE_SIDE_HINTS = {
    "eog": {
        "left": (_EOG_LEFT_HINTS | _EOG_STANDALONE_LEFT | {"e1"}),
        "right": (_EOG_RIGHT_HINTS | _EOG_STANDALONE_RIGHT | {"e2"}),
    },
    "emg": {
        "left": (_EMG_LEFT_HINTS | {"emg_left", "emg_l", "chin_left", "leftchin"}),
        "right": (_EMG_RIGHT_HINTS | {"emg_right", "emg_r", "chin_right", "rightchin"}),
    },
    "ecg": {
        "left": {"ecg_left", "ekg_left"},
        "right": {"ecg_right", "ekg_right"},
    },
}

_BASE_STRINGS = {
    "eog": ["eog"],
    "emg": ["emg", "chemg", "chin", "mentalis", "ment", "submental"],
    "ecg": ["ecg", "ekg"],
}

_DROPPED_LABELS = {"resp_oro_nasal", "temp_rectal", "spo2", "event_marker"}


def _clean_label(label: str) -> str:
    """Return a lowercase label with non-alphanumeric characters collapsed."""

    cleaned = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return cleaned


_ELECTRODE_BASES = {
    "a",
    "af",
    "c",
    "cp",
    "cz",
    "f",
    "fc",
    "fp",
    "ft",
    "i",
    "m",
    "o",
    "op",
    "oz",
    "p",
    "po",
    "pz",
    "t",
    "tp",
}


def _looks_like_electrode(token: str) -> bool:
    """Heuristically determine if ``token`` is a 10-20 style electrode name."""

    if not token:
        return False

    has_digit = any(ch.isdigit() for ch in token)
    ends_with_z = token.endswith("z")

    if not (has_digit or ends_with_z):
        return False

    base = re.sub(r"\d+$", "", token)
    if base.endswith("z"):
        base = base[:-1]

    if not base:
        return False

    return base in _ELECTRODE_BASES


def _side_from_suffix(s: str) -> str | None:
    s = s.strip("_-")
    if not s:
        return None
    if s in _SIDE_LEFT_HINTS or s in _SIDE_LEFT_DIGITS:
        return "l"
    if s in _SIDE_RIGHT_HINTS or s in _SIDE_RIGHT_DIGITS:
        return "r"
    if re.fullmatch(r"0*1", s):
        return "l"
    if re.fullmatch(r"0*2", s):
        return "r"
    return None


def _side_from_token(token: str, base: str) -> str | None:
    if not token:
        return None
    if token in _SIDE_LEFT_HINTS or token in _SIDE_LEFT_DIGITS:
        return "l"
    if token in _SIDE_RIGHT_HINTS or token in _SIDE_RIGHT_DIGITS:
        return "r"

    base_hints = _BASE_SIDE_HINTS.get(base)
    if base_hints:
        if token in base_hints["left"]:
            return "l"
        if token in base_hints["right"]:
            return "r"

    for base_token in _BASE_STRINGS.get(base, ()):
        if not base_token:
            continue
        if token.startswith(base_token):
            suffix = token[len(base_token) :]
            side = _side_from_suffix(suffix)
            if side:
                return side
        if token.endswith(base_token) and len(base_token) < len(token):
            prefix = token[: -len(base_token)]
            side = _side_from_suffix(prefix)
            if side:
                return side
        if base_token in token:
            idx = token.find(base_token)
            suffix = token[idx + len(base_token) :]
            side = _side_from_suffix(suffix)
            if side:
                return side
    return None


def _detect_side_for_base(
    base: str, tokens: list[str], original_tokens: list[str]
) -> str | None:
    combined: list[str] = []
    seen = set()
    for tok in tokens + original_tokens:
        if tok and tok not in seen:
            combined.append(tok)
            seen.add(tok)
    side: str | None = None
    for tok in combined:
        candidate = _side_from_token(tok, base)
        if candidate:
            if side is None:
                side = candidate
            elif side != candidate:
                return None
    return side


def normalize_channel_label(label: str) -> str:
    """Normalise ``label`` so that common aliases map to a shared key.

    The normalisation intentionally collapses a few known aliases:

    * ``psg_EOG`` and other EOG variations map to ``"eog"``.
    * ``PSG_EMG``, ``EMG submental``, and ``ChEMG`` labels map to ``"emg"``.
    * ``eeg/F4_M1`` and similar type/ prefix notation are handled.
    """

    # Handle type/ or type_ prefix notation first (e.g., "eeg/F4_M1", "eeg_F4_M1", "eog/EOG1", "eog_EOG1", "emg/ECG", "emg_ECG")
    type_prefix_match = re.match(
        r"^(eeg|eog|emg|ecg|ekg)[/_](.+)$", label, flags=re.IGNORECASE
    )
    if type_prefix_match:
        # Extract the channel name after the type prefix
        channel_name = type_prefix_match.group(2)

        # Special case: Handle mislabeled channels like "emg/ECG" (should be ecg/ECG)
        # The type prefix might not match the actual channel type - that's okay, we just strip it
        # The actual type will be inferred from the channel name itself

        # Convert underscores to hyphens for bipolar notation if applicable
        if "_" in channel_name and "-" not in channel_name:
            parts = channel_name.split("_", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                # Check if both parts look like electrode names
                if (
                    any(c.isalpha() for c in parts[0])
                    and any(c.isalpha() for c in parts[1])
                    and not parts[0].upper().startswith("EOG")
                    and not parts[0].upper().startswith("EMG")
                    and not parts[0].upper().startswith("ECG")
                ):
                    channel_name = channel_name.replace("_", "-", 1)
        # Process the channel name after stripping the type prefix
        label = channel_name

    cleaned = _clean_label(label)
    if not cleaned:
        return ""

    if cleaned in _DROPPED_LABELS:
        return ""

    tokens = [tok for tok in cleaned.split("_") if tok]
    if not tokens:
        return ""

    original_tokens = tokens[:]

    while len(tokens) > 1 and tokens[0] in _PREFIX_DROPS:
        tokens = tokens[1:]

    if not tokens:
        return ""

    token_set = set(tokens)
    combined_tokens = tokens + original_tokens

    has_eog_indicator = any(
        ("eog" in tok)
        or tok in _EOG_STANDALONE_LEFT
        or tok in _EOG_STANDALONE_RIGHT
        or tok in _HORIZONTAL_TOKENS
        for tok in combined_tokens
    )
    if has_eog_indicator:
        side = _detect_side_for_base("eog", tokens, original_tokens)
        return "eog" if side is None else f"eog_{side}"

    if (
        "emg" in token_set
        or any("emg" in tok for tok in combined_tokens)
        or token_set & _SUBMENTAL_TOKENS
        or original_tokens == ["psg", "emg"]
        or any("chin" in tok for tok in combined_tokens)
        or any("mentalis" in tok for tok in combined_tokens)
        or any("submental" in tok for tok in combined_tokens)
    ):
        side = _detect_side_for_base("emg", tokens, original_tokens)
        return "emg" if side is None else f"emg_{side}"

    if any(
        (tok in _CARDIAC_TOKENS) or tok.startswith(_CARDIAC_PREFIXES)
        for tok in combined_tokens
    ):
        side = _detect_side_for_base("ecg", tokens, original_tokens)
        return "ecg" if side is None else f"ecg_{side}"

    if tokens:
        first = tokens[0]
        if _looks_like_electrode(first):
            return first

    return "_".join(tokens)


def infer_channel_type(label: str) -> str:
    """Infer channel type (eeg, eog, emg, ecg) from channel label.

    Args:
        label: Channel label string (e.g., 'C3-A2', 'eeg/F4_M1', 'eog/EOG1')

    Returns:
        Channel type string: "eeg", "eog", "emg", "ecg", or "unknown"
    """
    norm = label.lower()

    # Check for type prefix notation (e.g., "eeg/F4_M1", "eeg_F4_M1", "EEG C4-M1", "eog/EOG1")
    if norm.startswith("eeg/") or norm.startswith("eeg_") or norm.startswith("eeg "):
        return "eeg"
    if norm.startswith("eog/") or norm.startswith("eog_") or norm.startswith("eog "):
        return "eog"
    if norm.startswith("emg/") or norm.startswith("emg_") or norm.startswith("emg "):
        return "emg"
    if (
        norm.startswith("ecg/")
        or norm.startswith("ecg_")
        or norm.startswith("ecg ")
        or norm.startswith("ekg/")
        or norm.startswith("ekg_")
        or norm.startswith("ekg ")
    ):
        return "ecg"

    # Check for explicit type indicators
    if (
        "eog" in norm
        or "loc" in norm
        or "roc" in norm
        or "heog" in norm
        or "veog" in norm
    ):
        return "eog"
    if "emg" in norm or "chin" in norm or "mentalis" in norm or "submental" in norm:
        return "emg"
    # "Ment1-Ment2" pattern: mentalis electrodes without the full "mentalis" string
    if re.match(r"ment\d", norm):
        return "emg"
    if "ecg" in norm or "ekg" in norm:
        return "ecg"

    # Check for E1/E2 pattern (common EOG naming: E1-M2, E2-M1, E2-M2).
    # Some STAGES sites omit the derivation separator (for example, E1M2).
    # Must check before electrode detection since E1/E2 look like electrodes
    if (
        re.match(r"^e[12][-_]", norm)
        or re.fullmatch(r"e[12][am][12]", norm)
        or norm in ("e1", "e2")
    ):
        return "eog"

    # Check for generic EEG channel names (EEG, EEG1, EEG2, EEG3, EEG(sec), etc.)
    # Common in MESA and SHHS datasets
    # Matches: "eeg", "eeg1", "eeg2", "eeg3", "eeg(sec)", "eeg(primary)", etc.
    if re.match(r"^eeg(\d+|\([^)]+\))?$", norm):
        return "eeg"

    # Compact bipolar 10-20 derivations used by some STAGES sites (C3M2,
    # C4M1, F3M2, and similar) omit the separator between electrodes.
    if re.fullmatch(r"(?:fp|af|f|fc|c|cp|p|po|o|t|tp)\d[am]\d", norm):
        return "eeg"

    # If no explicit indicator and looks like electrode, assume EEG
    cleaned = _clean_label(label)
    tokens = [tok for tok in cleaned.split("_") if tok]
    if tokens and _looks_like_electrode(tokens[0]):
        return "eeg"

    return "unknown"


def normalize_robust(
    signal: np.ndarray,
    *,
    return_metadata: bool = False,
    mode: str = "iqr",
    channel_type: str = "eeg",
    clip_threshold: float = 20.0,
    q1_percentile: float = 25.0,
    q3_percentile: float = 75.0,
    # Legacy sigma mode params
    k_eeg: float = 10.0,
    k_emg: float = 12.0,
    lo: float = 0.5,
    hi: float = 99.5,
    clip_method: str = "hard",
    tanh_c: float = 10.0,
    return_outlier_mask: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, dict[str, float]]
    | tuple[np.ndarray, np.ndarray, dict[str, float]]
):
    """Robust normalization with IQR or sigma-based scaling.

    IQR mode (recommended):
        - Scales by interquartile range for robustness
        - Hard clips at ±clip_threshold IQR units (default ±20)
        - No compression - preserves linear relationships

    Args:
        signal: 1D array of waveform samples
        mode: "iqr" (recommended), "sigma", or "keep"
        clip_threshold: Hard clipping threshold for IQR mode (default 20.0)
        q1_percentile: Lower quartile percentile (default 25.0)
        q3_percentile: Upper quartile percentile (default 75.0)
    """

    if mode == "iqr":
        # IQR-based normalization (Nature paper method)
        q1, q3 = np.percentile(signal, [q1_percentile, q3_percentile]).astype(
            np.float32
        )
        med = np.median(signal).astype(np.float32)

        iqr = max(q3 - q1, 1e-6)
        y = (signal - med) / iqr

        # Track outliers before clipping
        outlier_mask = (np.abs(y) > clip_threshold).astype(np.float32)

        # Hard clip
        y = np.clip(y, -clip_threshold, clip_threshold)

        metadata = {
            "q1": float(q1),
            "q3": float(q3),
            "median": float(med),
            "iqr": float(iqr),
            "clip_threshold": float(clip_threshold),
            "mode": "iqr",
        }

    elif mode == "sigma":
        # Legacy sigma mode - your existing implementation
        ZSPAN_LOOKUP = {
            (1.0, 99.0): 4.6527,
            (0.5, 99.5): 5.152,
            (0.1, 99.9): 6.1805,
            (5.0, 95.0): 3.2897,
        }
        zspan = ZSPAN_LOOKUP.get((lo, hi), 4.6527)

        p_lo, p_hi = np.percentile(signal, [lo, hi]).astype(np.float32)
        med = np.median(signal).astype(np.float32)
        sigma_hat = max((p_hi - p_lo) / zspan, 1e-6)

        raw_z = (signal - med) / sigma_hat
        k = k_emg if channel_type in ("emg",) else k_eeg
        outlier_mask = (np.abs(raw_z) > k).astype(np.float32)

        if clip_method == "hard":
            y = np.clip(raw_z, -k, k)
        else:  # soft
            y = np.tanh(raw_z / tanh_c)

        metadata = {
            "p_lo": float(p_lo),
            "p_hi": float(p_hi),
            "median": float(med),
            "sigma_hat": float(sigma_hat),
            "zspan": float(zspan),
            "k": float(k),
            "mode": "sigma",
            "clip_method": clip_method,
            "tanh_c": float(tanh_c) if clip_method == "soft" else None,
        }

    elif mode == "keep":
        p_lo, p_hi = np.percentile(signal, [1.0, 99.0]).astype(np.float32)
        med = np.median(signal).astype(np.float32)
        outlier_mask = ((signal < p_lo) | (signal > p_hi)).astype(np.float32)
        x = np.clip(signal, p_lo, p_hi)
        span = max(p_hi - p_lo, 1e-6)
        y = (x - med) / span
        metadata = {
            "p_lo": float(p_lo),
            "p_hi": float(p_hi),
            "median": float(med),
            "scale": float(span),
            "mode": "keep",
        }
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    y = y.astype(np.float32)

    # Build return tuple
    if return_outlier_mask and return_metadata:
        return y, outlier_mask, metadata
    elif return_outlier_mask:
        return y, outlier_mask
    elif return_metadata:
        return y, metadata
    else:
        return y


class PSGNormalizer:
    """Single source of truth for PSG normalization.

    This class centralizes the normalization logic so training, offline
    preprocessing, and inference stay in sync. The recommended strategy is the
    IQR-based scaling described in the Nature paper:

        - Subtract per-channel median
        - Divide by interquartile range (Q3-Q1)
        - Clip hard at ±20 IQR units

    Legacy sigma and "keep" modes are kept for backward compatibility but are no
    longer the default. Any changes to the constants below must be reflected in:
        - preprocessing/config.py (NormalizationCfg defaults)
        - preprocessing/normalizer.py (EmbeddedPreproc buffers)
        - data/channel/normalization.py (normalize_robust defaults)
    """

    # IQR-based normalization defaults
    MODE = "iqr"  # "iqr", "sigma", or "keep"
    LO_PERCENTILE = 25.0  # Q1 for IQR mode
    HI_PERCENTILE = 75.0  # Q3 for IQR mode
    CLIP_THRESHOLD = 20.0  # Hard clip at ±20 IQR units

    # Legacy sigma mode parameters
    SIGMA_LO_PERCENTILE = 0.5
    SIGMA_HI_PERCENTILE = 99.5
    K_EEG = 10.0  # Sigma clipping for EEG/EOG
    K_EMG = 12.0  # Sigma clipping for EMG
    ZSPAN = 5.152  # Sigma equivalence for P0.5-P99.5 span
    CLIP_METHOD = "hard"  # "hard" or "soft" (tanh)
    TANH_C = 10.0  # Divisor for tanh soft clipping

    @classmethod
    def normalize(cls, signal: np.ndarray, channel_type: str = "eeg") -> np.ndarray:
        """Normalize a PSG signal using the configured strategy."""
        if cls.MODE == "iqr":
            q1, q3 = np.percentile(signal, [cls.LO_PERCENTILE, cls.HI_PERCENTILE])
            median = np.median(signal)
            iqr = max(q3 - q1, 1e-6)
            normalized = (signal - median) / iqr
            normalized = np.clip(normalized, -cls.CLIP_THRESHOLD, cls.CLIP_THRESHOLD)
            return normalized.astype(np.float32)

        if cls.MODE == "sigma":
            p_lo, p_hi = np.percentile(
                signal, [cls.SIGMA_LO_PERCENTILE, cls.SIGMA_HI_PERCENTILE]
            )
            median = np.median(signal)
            sigma_hat = max((p_hi - p_lo) / cls.ZSPAN, 1e-6)
            raw_z = (signal - median) / sigma_hat
            k = cls.K_EMG if channel_type == "emg" else cls.K_EEG
            if cls.CLIP_METHOD == "soft":
                normalized = np.tanh(raw_z / cls.TANH_C)
            else:
                normalized = np.clip(raw_z, -k, k)
            return normalized.astype(np.float32)

        # Keep / legacy percentile clipping
        p_lo, p_hi = np.percentile(signal, [1.0, 99.0])
        median = np.median(signal)
        x_clipped = np.clip(signal, p_lo, p_hi)
        span = max(p_hi - p_lo, 1e-6)
        normalized = (x_clipped - median) / span
        return normalized.astype(np.float32)

    @classmethod
    def normalize_with_metadata(
        cls, signal: np.ndarray, channel_type: str = "eeg"
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Normalize and return metadata for reproducibility."""
        if cls.MODE == "iqr":
            q1, q3 = np.percentile(signal, [cls.LO_PERCENTILE, cls.HI_PERCENTILE])
            median = np.median(signal)
            iqr = max(q3 - q1, 1e-6)
            normalized = (signal - median) / iqr
            normalized = np.clip(normalized, -cls.CLIP_THRESHOLD, cls.CLIP_THRESHOLD)
            metadata = {
                "q1": float(q1),
                "q3": float(q3),
                "median": float(median),
                "iqr": float(iqr),
                "clip_threshold": float(cls.CLIP_THRESHOLD),
                "mode": "iqr",
            }
            return normalized.astype(np.float32), metadata

        if cls.MODE == "sigma":
            p_lo, p_hi = np.percentile(
                signal, [cls.SIGMA_LO_PERCENTILE, cls.SIGMA_HI_PERCENTILE]
            )
            median = np.median(signal)
            sigma_hat = max((p_hi - p_lo) / cls.ZSPAN, 1e-6)
            raw_z = (signal - median) / sigma_hat
            k = cls.K_EMG if channel_type == "emg" else cls.K_EEG
            if cls.CLIP_METHOD == "soft":
                normalized = np.tanh(raw_z / cls.TANH_C)
            else:
                normalized = np.clip(raw_z, -k, k)
            metadata = {
                "p_lo": float(p_lo),
                "p_hi": float(p_hi),
                "median": float(median),
                "sigma_hat": float(sigma_hat),
                "zspan": float(cls.ZSPAN),
                "k": float(k),
                "mode": "sigma",
                "clip_method": cls.CLIP_METHOD,
                "tanh_c": float(cls.TANH_C) if cls.CLIP_METHOD == "soft" else None,
            }
            return normalized.astype(np.float32), metadata

        p_lo, p_hi = np.percentile(signal, [1.0, 99.0])
        median = np.median(signal)
        x_clipped = np.clip(signal, p_lo, p_hi)
        span = max(p_hi - p_lo, 1e-6)
        normalized = (x_clipped - median) / span
        metadata = {
            "p_lo": float(p_lo),
            "p_hi": float(p_hi),
            "median": float(median),
            "scale": float(span),
            "mode": "keep",
        }
        return normalized.astype(np.float32), metadata
