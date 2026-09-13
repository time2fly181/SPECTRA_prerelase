"""Per-epoch uncertainty scoring and ranking for the triage worklist.

The inference backend attaches cheap per-epoch flag arrays to every scoring run
(``flag_margin``/``flag_entropy``/``flag_maxprob``; see
:func:`spectra.inference.runtime.compute_flag_scores`). This module turns
those into a single direction-normalized *uncertainty* score (higher = more
uncertain) and ranks epochs by it, so the GUI can present a "most uncertain
first" review worklist.

All functions are pure NumPy and Qt-free so they can be unit-tested directly::

    >>> from spectra.review.uncertainty import combine_flag_score
    >>> combine_flag_score(margin=None, entropy=None, maxprob=None)
    (None returned when the requested array is absent)
"""

from __future__ import annotations

import numpy as np

__all__ = ["VALID_METRICS", "combine_flag_score", "rank_epochs_by_uncertainty"]

VALID_METRICS: tuple[str, ...] = ("entropy", "margin", "maxprob")


def combine_flag_score(
    *,
    margin: np.ndarray | None,
    entropy: np.ndarray | None,
    maxprob: np.ndarray | None,
    metric: str = "entropy",
) -> np.ndarray | None:
    """Derive a per-epoch uncertainty score (higher = more uncertain).

    The three inputs come straight from the backend flag dict and point in
    different directions: ``entropy`` already increases with uncertainty, while
    ``margin`` (top1 minus top2) and ``maxprob`` increase with *confidence*. The
    latter two are inverted to ``1 - x`` so every returned score is
    "higher = more uncertain".

    Args:
        margin: ``flag_margin`` array of shape ``(n_epochs,)`` or ``None``.
        entropy: ``flag_entropy`` array of shape ``(n_epochs,)`` or ``None``.
        maxprob: ``flag_maxprob`` array of shape ``(n_epochs,)`` or ``None``.
        metric: Which flag to base the score on. One of ``"entropy"``,
            ``"margin"``, ``"maxprob"``.

    Returns:
        A ``float32`` array of shape ``(n_epochs,)``, or ``None`` when the array
        required by ``metric`` was not provided (the caller then falls back to a
        confidence-derived score).

    Raises:
        ValueError: If ``metric`` is not one of :data:`VALID_METRICS`.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {VALID_METRICS}")

    source = {"entropy": entropy, "margin": margin, "maxprob": maxprob}[metric]
    if source is None:
        return None

    values = np.asarray(source, dtype=np.float64).ravel()
    if metric == "entropy":
        score = values
    else:
        # margin and maxprob increase with confidence -> invert to uncertainty.
        score = 1.0 - values
    return score.astype(np.float32, copy=False)


def rank_epochs_by_uncertainty(
    scores: np.ndarray,
    *,
    descending: bool = True,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return epoch indices ordered by uncertainty score.

    Ties keep their original epoch order (stable sort), so equal-uncertainty
    epochs stay in chronological order.

    Args:
        scores: Per-epoch uncertainty scores of shape ``(n_epochs,)`` (higher =
            more uncertain, e.g. the output of :func:`combine_flag_score`).
        descending: If ``True`` (default) the most uncertain epoch comes first.
        mask: Optional boolean array of shape ``(n_epochs,)``; when given, only
            epochs where the mask is ``True`` are ranked and returned.

    Returns:
        An ``int64`` array of epoch indices into ``scores``. Empty when
        ``scores`` is empty or ``mask`` excludes everything.

    Raises:
        ValueError: If ``scores`` is not 1-D, or ``mask`` shape mismatches.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    indices = np.arange(scores.shape[0])

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool).ravel()
        if mask_arr.shape != scores.shape:
            raise ValueError(
                f"mask shape {mask_arr.shape} does not match scores shape "
                f"{scores.shape}"
            )
        indices = indices[mask_arr]
        subset = scores[mask_arr]
    else:
        subset = scores

    if indices.size == 0:
        return np.empty(0, dtype=np.int64)

    # Negate for descending so ties stay in ascending-index (chronological)
    # order under the stable sort.
    keys = -subset if descending else subset
    order = np.argsort(keys, kind="stable")
    return indices[order].astype(np.int64, copy=False)
