"""EDF reader and resampler extracted from the canonical fp32 converter."""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
from typing import Any, NoReturn

import numpy as np


@dataclass(frozen=True)
class EdfHeaderRepair:
    """Description of a temporary, minimally repaired EDF header."""

    path: str
    unusable_signal_indices: frozenset[int]
    changes: tuple[str, ...]


_EDF_REPAIRABLE_HEADER_ERRORS = (
    "Physical Dimension",
    "Physical Minimum",
    "Physical Maximum",
    "starttime is incorrect",
)


def _edf_ascii_field(value: str, width: int) -> bytes:
    """Encode a printable EDF header field with fixed-width space padding."""
    encoded = value.encode("ascii")
    if len(encoded) > width:
        raise ValueError(f"EDF header value {value!r} exceeds {width} bytes")
    return encoded.ljust(width, b" ")


def _edf_number_field(value: float) -> bytes:
    """Encode a finite number into an EDF's eight-byte numeric field."""
    if not math.isfinite(value):
        raise ValueError(f"Cannot encode non-finite EDF header value {value!r}")

    # EDF permits decimal and scientific notation. Use the highest precision
    # representation that fits the fixed-width field.
    for precision in range(7, 0, -1):
        text = format(value, f".{precision}g")
        if len(text) <= 8:
            return _edf_ascii_field(text, 8)
    raise ValueError(f"Cannot fit EDF header value {value!r} in eight bytes")


def _parse_edf_number(field: bytes) -> float | None:
    """Return a finite EDF numeric value, or ``None`` for malformed input."""
    try:
        text = field.decode("ascii").strip()
        value = float(text)
    except (UnicodeDecodeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _repair_edf_start_time(field: bytes) -> bytes | None:
    """Return a compliant EDF start time for a malformed eight-byte field."""
    try:
        text = field.decode("ascii").strip()
    except UnicodeDecodeError:
        text = ""

    match = re.fullmatch(r"(\d{2})[.:](\d{2})[.:](\d{2})", text)
    if match is not None:
        hour, minute, second = (int(part) for part in match.groups())
        if hour < 24 and minute < 60 and second < 60:
            normalized = f"{hour:02d}.{minute:02d}.{second:02d}".encode("ascii")
            return None if normalized == field else normalized

    # EDF requires a valid time even when the source clock time is unknown.
    # The converter uses elapsed sample/annotation timing, so a midnight
    # placeholder satisfies the strict reader without changing alignment.
    return b"00.00.00"


def _expanded_physical_range(center: float) -> tuple[bytes, bytes]:
    """Build a representable, non-degenerate range around ``center``."""
    span = max(1.0, abs(center) * 0.01)
    try:
        minimum = _edf_number_field(center - span)
        maximum = _edf_number_field(center + span)
        if _parse_edf_number(minimum) != _parse_edf_number(maximum):
            return minimum, maximum
    except ValueError:
        pass

    # The original calibration is unusable and the channel will be excluded.
    # This fallback exists only to let a strict EDF reader open the other signals.
    return _edf_number_field(-1.0), _edf_number_field(1.0)


def _repair_edf_header(edf_path: str) -> EdfHeaderRepair | None:
    """Create a temporary copy with safely repairable EDF header defects fixed.

    Only defects that can be identified unambiguously are changed:

    * missing/malformed start times become ``00.00.00``;
    * blank/non-printable Physical Dimension fields become ``"unknown"``;
    * equal finite physical minimum/maximum values are expanded.

    A signal with a degenerate physical range has lost its calibration, so its
    index is returned in ``unusable_signal_indices``. Callers must not consume
    that signal after opening the repaired file. The source EDF is never changed.
    """
    with open(edf_path, "rb") as source:
        fixed_header = source.read(256)
        if len(fixed_header) != 256:
            return None

        try:
            n_signals = int(fixed_header[252:256].decode("ascii").strip())
        except (ValueError, UnicodeDecodeError):
            return None
        if n_signals <= 0:
            return None

        signal_header = source.read(256 * n_signals)
        if len(signal_header) != 256 * n_signals:
            return None

    header = bytearray(fixed_header + signal_header)
    changes: list[str] = []
    unusable_signal_indices: set[int] = set()

    label_offset = 256
    physical_dimension_offset = 256 + 96 * n_signals
    physical_minimum_offset = 256 + 104 * n_signals
    physical_maximum_offset = 256 + 112 * n_signals

    start_time_start = 176
    start_time_end = 184
    start_time_field = bytes(header[start_time_start:start_time_end])
    repaired_start_time = _repair_edf_start_time(start_time_field)
    if repaired_start_time is not None:
        header[start_time_start:start_time_end] = repaired_start_time
        original_start_time = start_time_field.decode("ascii", errors="replace").strip()
        changes.append(
            f"start time {original_start_time!r} -> "
            f"{repaired_start_time.decode('ascii')}"
        )

    for index in range(n_signals):
        label_start = label_offset + 16 * index
        label = (
            bytes(header[label_start : label_start + 16])
            .decode("ascii", errors="replace")
            .strip()
        )
        display_label = label or f"signal {index}"

        dimension_start = physical_dimension_offset + 8 * index
        dimension = bytes(header[dimension_start : dimension_start + 8])
        dimension_is_printable = all(32 <= byte <= 126 for byte in dimension)
        if not dimension_is_printable or not dimension.strip():
            header[dimension_start : dimension_start + 8] = _edf_ascii_field(
                "unknown", 8
            )
            changes.append(
                f"{display_label} (index {index}): physical dimension -> unknown"
            )

        minimum_start = physical_minimum_offset + 8 * index
        maximum_start = physical_maximum_offset + 8 * index
        minimum_field = bytes(header[minimum_start : minimum_start + 8])
        maximum_field = bytes(header[maximum_start : maximum_start + 8])
        minimum = _parse_edf_number(minimum_field)
        maximum = _parse_edf_number(maximum_field)

        if minimum is not None and maximum is not None and minimum == maximum:
            repaired_minimum, repaired_maximum = _expanded_physical_range(minimum)
            header[minimum_start : minimum_start + 8] = repaired_minimum
            header[maximum_start : maximum_start + 8] = repaired_maximum
            unusable_signal_indices.add(index)
            changes.append(
                f"{display_label} (index {index}): degenerate physical range "
                f"[{minimum:g}, {maximum:g}] expanded; signal excluded"
            )

    if not changes:
        return None

    # Copy only after finding a repairable defect. shutil.copyfile uses bounded
    # memory/in-kernel copying, unlike loading multi-gigabyte EDFs into a bytearray.
    fd, temporary_path = tempfile.mkstemp(
        suffix=os.path.splitext(edf_path)[1] or ".edf"
    )
    os.close(fd)
    try:
        shutil.copyfile(edf_path, temporary_path)
        with open(temporary_path, "r+b") as repaired_file:
            repaired_file.write(header)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise

    return EdfHeaderRepair(
        path=temporary_path,
        unusable_signal_indices=frozenset(unusable_signal_indices),
        changes=tuple(changes),
    )


class _MneEdfReader:
    """``mne``-backed reader exposing the ``pyedflib.EdfReader`` methods used here.

    Some EDFs are rejected outright by ``pyedflib`` even though their signal data
    is intact. The common case in this repository is a header whose ``reserved``
    field declares ``EDF+C`` while the file carries no ``EDF Annotations`` signal
    (external sidecars hold the staging). ``pyedflib`` reports that as
    ``"the file is not EDF(+) or BDF(+) compliant the label is incorrect"``,
    which is not a label defect at all and is not repairable through
    :func:`_repair_edf_header`. Rewriting the ``reserved`` field would require
    copying every multi-hundred-megabyte recording, so ``mne`` reads them in
    place instead.

    The method names deliberately mirror ``pyedflib`` so the conversion body does
    not need to know which backend is open.

    Note:
        ``mne`` returns SI units (volts) where ``pyedflib`` returns the EDF
        physical unit (typically microvolts). The stored waveform is unaffected
        because normalization is a per-recording ``(x - median) / IQR``, but the
        recorded ``robust_median``/``robust_iqr`` attributes are scaled
        accordingly. The ``reader_backend`` store attribute records which backend
        produced them.
    """

    backend = "mne"

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self._sfreq = float(raw.info["sfreq"])

    @property
    def signals_in_file(self) -> int:
        return len(self._raw.ch_names)

    def getSignalLabels(self) -> list[str]:  # noqa: N802  # pyedflib API name
        return [str(name) for name in self._raw.ch_names]

    def getSampleFrequency(self, index: int) -> float:  # noqa: N802  # pyedflib API
        # mne unifies every channel to a single (maximum) rate when reading.
        del index
        return self._sfreq

    def getFileDuration(self) -> float:  # noqa: N802  # pyedflib API name
        return float(self._raw.n_times) / self._sfreq

    def getStartdatetime(self) -> datetime:  # noqa: N802  # pyedflib API name
        measurement_date = self._raw.info.get("meas_date")
        if measurement_date is None:
            return datetime(2000, 1, 1)
        return datetime(
            measurement_date.year,
            measurement_date.month,
            measurement_date.day,
            measurement_date.hour,
            measurement_date.minute,
            measurement_date.second,
            measurement_date.microsecond,
        )

    def readSignal(self, index: int) -> np.ndarray:  # noqa: N802  # pyedflib API
        return np.asarray(self._raw.get_data(picks=[index])[0], dtype=np.float64)

    def readAnnotations(self) -> NoReturn:  # noqa: N802  # pyedflib API name
        raise ValueError(
            "Embedded EDF+ annotations are unavailable through the mne fallback "
            "reader. Provide an annotation sidecar for this recording."
        )

    def close(self) -> None:
        try:
            self._raw.close()
        except Exception:  # noqa: BLE001  # mne Raw.close is best-effort
            pass


def _open_edf_with_mne(edf_path: str) -> _MneEdfReader:
    """Open an EDF that ``pyedflib`` refused, using ``mne`` as the backend.

    Raises:
        ModuleNotFoundError: If ``mne`` is not installed.
    """
    import mne  # imported lazily: only non-compliant EDFs need this backend

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    return _MneEdfReader(raw)


def _labels_for_channel_mapping(
    labels: list[str], unusable_signal_indices: frozenset[int]
) -> list[str]:
    """Mask uncalibrated signals while preserving transform-matrix positions."""
    return [
        (
            label
            if index not in unusable_signal_indices
            else f"__UNUSABLE_SIGNAL_{index}__"
        )
        for index, label in enumerate(labels)
    ]


@lru_cache(maxsize=64)
def _up_down(fin: float, fout: float):
    """Calculate resampling ratio."""
    if abs(fin - fout) < 1e-8:
        return 1, 1
    frac = Fraction(fout / fin).limit_denominator(512)
    return frac.numerator, frac.denominator


def resample_signal(x: np.ndarray, fin: float, fout: float) -> np.ndarray:
    """Resample signal from fin to fout Hz."""
    try:
        from scipy.signal import resample_poly  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "scipy is required for EDF resampling. Install with: pip install scipy"
        ) from exc

    up, down = _up_down(fin, fout)
    if up == 1 and down == 1:
        return x.astype(np.float32, copy=False)
    xr = resample_poly(x, up, down)
    return xr.astype(np.float32, copy=False)
