"""Pure, Qt-free logic for the uncertainty-aware review workflow.

This subpackage holds the numerical helpers that back the inference GUI's
review features so they can be unit-tested without a Qt event loop:

- :mod:`spectra.review.uncertainty` — rank epochs by per-epoch
  uncertainty-flag scores for the triage worklist.
- :mod:`spectra.review.agreement` — compare model predictions against a
  reference hypnogram (accuracy, Cohen's kappa, confusion matrix).
- :mod:`spectra.review.reference` — parse reference hypnograms from
  CSV/NPY/NPZ/TXT/Profusion-XML into AASM stage indices.
"""

from __future__ import annotations

from spectra.review.agreement import AgreementResult, compute_agreement
from spectra.review.reference import (
    parse_reference_hypnogram,
    parse_xml_reference,
)
from spectra.review.uncertainty import (
    combine_flag_score,
    rank_epochs_by_uncertainty,
)

__all__ = [
    "AgreementResult",
    "combine_flag_score",
    "compute_agreement",
    "parse_reference_hypnogram",
    "parse_xml_reference",
    "rank_epochs_by_uncertainty",
]
