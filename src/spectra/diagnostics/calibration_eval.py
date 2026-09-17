"""Epoch-level probability calibration and reliability curves for GUI and API use.

These functions compare confidence with reference-label accuracy. Inputs must
already be aligned on the same epoch grid. Filter unscored model rows before
calling: label masking does not remove zero probability rows. No CLI is provided.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectra.logging.tensorboard_metrics import compute_calibration_metrics

_STAGE_TO_INDEX = {
    "0": 0,
    "w": 0,
    "wake": 0,
    "1": 1,
    "n1": 1,
    "2": 2,
    "n2": 2,
    "3": 3,
    "n3": 3,
    "n4": 3,
    "4": 4,
    "r": 4,
    "rem": 4,
}


def multiclass_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Return mean per-epoch squared probability error summed over classes.

    Args:
        probs: Finite normalized probabilities ``[N, K]`` in stage order.
        labels: Integer class indices ``[N]`` in ``[0, K)``; no ignored labels.

    Returns:
        Float64-derived scalar, or zero for no epochs. The class sum is not
        divided by ``K``. Probability normalization is the caller's responsibility.

    Raises:
        ValueError: If shapes disagree or labels are outside the class range.
    """
    if probs.ndim != 2:
        raise ValueError(f"Expected probs shape [N, K], got {probs.shape}")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Probability rows ({probs.shape[0]}) do not match labels ({labels.shape[0]})"
        )
    if labels.size == 0:
        return 0.0
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError(
            f"Labels must be in [0, {probs.shape[1] - 1}] after filtering; got {labels.min()}..{labels.max()}"
        )

    one_hot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    return float(np.mean(np.sum((probs.astype(np.float64) - one_hot) ** 2, axis=1)))


def evaluate_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
    drop_invalid: bool = True,
) -> dict[str, float | int]:
    """Compare epoch-level confidence with aligned reference-label accuracy.

    Args:
        probabilities: Finite normalized ``[N, K]`` probabilities. For SPECTRA,
            columns are Wake, N1, N2, N3, REM. Exclude unscored rows first.
        labels: Integer reference labels ``[N]`` on the same epoch grid.
        n_bins: Positive number of equal-width confidence bins over ``[0, 1]``.
        drop_invalid: Exclude labels outside ``[0, K)`` (including ``-1``).
            This does not validate or filter probability rows.

    Returns:
        Sample/class counts, accuracy, average confidence, ECE, MCE, signed
        overconfidence (confidence minus accuracy), and multiclass Brier score.
        ECE weights bin gaps by epoch fraction. Bins are left-open/right-closed;
        zero confidence is not binned. Empty inputs return zero-valued metrics.

    Raises:
        ValueError: If shapes disagree, or invalid labels remain for Brier scoring.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)

    if probs.ndim != 2:
        raise ValueError(f"Expected probabilities shape [N, K], got {probs.shape}")
    if probs.shape[0] != labels_arr.shape[0]:
        raise ValueError(
            f"Probability rows ({probs.shape[0]}) do not match labels ({labels_arr.shape[0]})"
        )

    if drop_invalid:
        valid = (labels_arr >= 0) & (labels_arr < probs.shape[1])
        probs = probs[valid]
        labels_arr = labels_arr[valid]

    if labels_arr.size == 0:
        return {
            "num_samples": 0,
            "num_classes": int(probabilities.shape[1]),
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "ece": 0.0,
            "mce": 0.0,
            "overconfidence": 0.0,
            "brier": 0.0,
        }

    predictions = probs.argmax(axis=1).astype(np.int64, copy=False)
    confidences = probs.max(axis=1).astype(np.float64, copy=False)

    calibration = compute_calibration_metrics(
        confidences=confidences,
        predictions=predictions,
        labels=labels_arr,
        n_bins=n_bins,
    )

    accuracy = float((predictions == labels_arr).mean())
    brier = multiclass_brier_score(probs, labels_arr)

    return {
        "num_samples": int(labels_arr.size),
        "num_classes": int(probs.shape[1]),
        "accuracy": accuracy,
        "avg_confidence": float(calibration["avg_confidence"]),
        "ece": float(calibration["ece"]),
        "mce": float(calibration["mce"]),
        "overconfidence": float(calibration["overconfidence"]),
        "brier": brier,
    }


@dataclass(frozen=True)
class ReliabilityCurve:
    """Per-bin confidence/accuracy for a reliability diagram.

    Attributes:
        bin_confidence: ``(n_bins,)`` mean predicted confidence per bin
            (``0.0`` for empty bins).
        bin_accuracy: ``(n_bins,)`` empirical accuracy per bin
            (``0.0`` for empty bins).
        bin_count: ``(n_bins,)`` int number of samples per bin.
        n_bins: Number of equal-width confidence bins over ``[0, 1]``.
    """

    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_count: np.ndarray
    n_bins: int


def reliability_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
    drop_invalid: bool = True,
) -> ReliabilityCurve:
    """Compute per-bin confidence/accuracy for a reliability diagram.

    Uses the same top-class confidence, label masking, and equal-width binning
    as :func:`evaluate_calibration` / ``compute_calibration_metrics`` so the
    diagram and the scalar ECE agree.

    Args:
        probabilities: Posterior array of shape ``(n_epochs, n_classes)``.
        labels: Integer reference labels of shape ``(n_epochs,)``.
        n_bins: Number of equal-width confidence bins over ``[0, 1]``.
        drop_invalid: Drop epochs whose label is outside ``[0, n_classes)``
            (e.g. ``-1`` unscored) before binning.

    Returns:
        A :class:`ReliabilityCurve`.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)

    if probs.ndim != 2:
        raise ValueError(f"Expected probabilities shape [N, K], got {probs.shape}")
    if probs.shape[0] != labels_arr.shape[0]:
        raise ValueError(
            f"Probability rows ({probs.shape[0]}) do not match labels "
            f"({labels_arr.shape[0]})"
        )

    if drop_invalid:
        valid = (labels_arr >= 0) & (labels_arr < probs.shape[1])
        probs = probs[valid]
        labels_arr = labels_arr[valid]

    bin_confidence = np.zeros(n_bins, dtype=np.float64)
    bin_accuracy = np.zeros(n_bins, dtype=np.float64)
    bin_count = np.zeros(n_bins, dtype=np.int64)

    if labels_arr.size == 0:
        return ReliabilityCurve(bin_confidence, bin_accuracy, bin_count, n_bins)

    predictions = probs.argmax(axis=1).astype(np.int64, copy=False)
    confidences = probs.max(axis=1).astype(np.float64, copy=False)
    accuracies = (predictions == labels_arr).astype(np.float64)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        count = int(np.count_nonzero(in_bin))
        bin_count[i] = count
        if count > 0:
            bin_confidence[i] = float(confidences[in_bin].mean())
            bin_accuracy[i] = float(accuracies[in_bin].mean())

    return ReliabilityCurve(bin_confidence, bin_accuracy, bin_count, n_bins)
