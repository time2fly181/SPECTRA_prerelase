"""Model-vs-reference hypnogram agreement metrics.

Given a model's per-epoch predictions and a reference (expert) hypnogram, this
module computes the standard inter-scorer agreement summary: overall accuracy,
Cohen's kappa, a 5x5 confusion matrix (rows = reference, cols = model) and
per-stage recall. It mirrors the scored-epoch masking already used by the GUI's
hypnogram canvas (only epochs the reference actually labels are scored) but
returns a full breakdown instead of a single accuracy scalar.

Pure NumPy, Qt-free, and self-contained (kappa is computed from the confusion
matrix) so it can be unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AgreementResult", "compute_agreement"]


@dataclass(frozen=True)
class AgreementResult:
    """Summary of agreement between model predictions and a reference hypnogram.

    Attributes:
        n_scored: Number of epochs scored (reference label in ``[0, n_classes)``).
        accuracy: Fraction of scored epochs where model equals reference. Model
            epochs marked ``-1`` (out-of-window/Unscored) count as disagreements.
        cohen_kappa: Chance-corrected agreement over epochs where both model and
            reference carry a valid class label.
        confusion: ``(n_classes, n_classes)`` int64 matrix; rows are reference
            stages, columns are model stages. Tallied over epochs where both
            labels are valid.
        per_stage_recall: ``(n_classes,)`` float array; for each reference stage,
            the fraction of its epochs the model labeled correctly (denominator
            is every scored epoch of that reference stage). ``0.0`` for stages
            absent from the reference.
    """

    n_scored: int
    accuracy: float
    cohen_kappa: float
    confusion: np.ndarray
    per_stage_recall: np.ndarray


def _cohen_kappa_from_confusion(confusion: np.ndarray) -> float:
    """Compute Cohen's kappa from a confusion matrix.

    Args:
        confusion: Square count matrix of shape ``(n, n)``.

    Returns:
        Kappa in ``[-1, 1]``; ``0.0`` when the matrix is empty or expected
        agreement is degenerate (``p_e == 1``).
    """
    total = float(confusion.sum())
    if total <= 0.0:
        return 0.0
    observed = float(np.trace(confusion)) / total
    row_marginals = confusion.sum(axis=1).astype(np.float64)
    col_marginals = confusion.sum(axis=0).astype(np.float64)
    expected = float(np.sum(row_marginals * col_marginals)) / (total * total)
    if abs(1.0 - expected) < 1e-12:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def compute_agreement(
    model_pred: np.ndarray,
    reference: np.ndarray,
    *,
    n_classes: int = 5,
    ignore_index: int = -1,
) -> AgreementResult:
    """Compare model predictions against a reference hypnogram.

    Epochs where the reference equals ``ignore_index`` (or is otherwise outside
    ``[0, n_classes)``) are not scored. The two arrays are truncated to their
    common length before comparison, matching the GUI's existing behavior.

    Args:
        model_pred: Per-epoch model predictions, shape ``(n_epochs,)``.
        reference: Per-epoch reference stages, shape ``(n_epochs,)``.
        n_classes: Number of AASM classes (default 5).
        ignore_index: Sentinel marking unscored reference epochs (default ``-1``).

    Returns:
        An :class:`AgreementResult`.
    """
    model_arr = np.asarray(model_pred).ravel().astype(np.int64, copy=False)
    ref_arr = np.asarray(reference).ravel().astype(np.int64, copy=False)

    min_len = int(min(model_arr.shape[0], ref_arr.shape[0]))
    model_arr = model_arr[:min_len]
    ref_arr = ref_arr[:min_len]

    empty_confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    empty_recall = np.zeros(n_classes, dtype=np.float64)

    scored = (ref_arr != ignore_index) & (ref_arr >= 0) & (ref_arr < n_classes)
    n_scored = int(np.count_nonzero(scored))
    if n_scored == 0:
        return AgreementResult(0, 0.0, 0.0, empty_confusion, empty_recall)

    m = model_arr[scored]
    r = ref_arr[scored]

    accuracy = float(np.mean(m == r))

    both_valid = (m >= 0) & (m < n_classes)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    if np.any(both_valid):
        np.add.at(confusion, (r[both_valid], m[both_valid]), 1)

    per_stage_recall = np.zeros(n_classes, dtype=np.float64)
    for cls in range(n_classes):
        denom = int(np.count_nonzero(r == cls))
        if denom > 0:
            correct = int(np.count_nonzero((r == cls) & (m == cls)))
            per_stage_recall[cls] = correct / denom

    kappa = _cohen_kappa_from_confusion(confusion)

    return AgreementResult(
        n_scored=n_scored,
        accuracy=accuracy,
        cohen_kappa=kappa,
        confusion=confusion,
        per_stage_recall=per_stage_recall,
    )
