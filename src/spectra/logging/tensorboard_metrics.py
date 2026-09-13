"""Comprehensive TensorBoard metrics logging for PSGStage training.

This module provides 15 categories of metrics logging for debugging sleep staging
models, with particular focus on N1 classification issues.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_HAS_MATPLOTLIB = False

plt: Any = None

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for headless servers
    import matplotlib.pyplot as _plt

    plt = _plt
    _HAS_MATPLOTLIB = True
except ImportError:
    pass

DEFAULT_STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def compute_calibration_metrics(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute calibration metrics including ECE (Expected Calibration Error).

    Args:
        confidences: Model confidence (max softmax prob) [N]
        predictions: Predicted class indices [N]
        labels: True class indices [N]
        n_bins: Number of bins for calibration

    Returns:
        Dict with calibration metrics
    """
    if len(confidences) == 0:
        return {
            "ece": 0.0,
            "mce": 0.0,
            "avg_confidence": 0.0,
            "overconfidence": 0.0,
        }

    accuracies = (predictions == labels).astype(float)

    # ECE calculation
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0  # Maximum Calibration Error

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

    # Average confidence and accuracy
    avg_confidence = float(confidences.mean())

    # Overconfidence: how much higher is confidence than accuracy on average
    overconfidence = avg_confidence - float(accuracies.mean())

    return {
        "ece": float(ece),
        "mce": float(mce),
        "avg_confidence": avg_confidence,
        "overconfidence": float(overconfidence),
    }
