"""NumPy calibration metrics used by reference evaluation and GUI review."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def compute_calibration_metrics(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute calibration metrics including ECE (Expected Calibration Error).

    Args:
        confidences: Finite top-class probabilities ``[N]`` in ``[0, 1]``.
        predictions: Predicted integer class indices ``[N]``.
        labels: Aligned reference indices ``[N]``; filter ignored labels first.
        n_bins: Positive number of equal-width confidence bins.

    Returns:
        ECE (epoch-fraction-weighted absolute bin gap), MCE (maximum bin gap),
        average confidence, and signed overconfidence (confidence minus accuracy).
        Bins are left-open/right-closed, so zero confidence is not binned.
        Empty inputs return zeros. Inputs are not filtered or normalized here.
    """
    if len(confidences) == 0:
        return {
            "ece": 0.0,
            "mce": 0.0,
            "avg_confidence": 0.0,
            "overconfidence": 0.0,
        }

    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            bin_error = np.abs(avg_acc - avg_conf)

            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)

    avg_confidence = float(confidences.mean())

    overconfidence = avg_confidence - float(accuracies.mean())

    return {
        "ece": float(ece),
        "mce": float(mce),
        "avg_confidence": avg_confidence,
        "overconfidence": float(overconfidence),
    }
