"""Thin Qt view widgets for the uncertainty-aware review workflow.

These panels are pure views: all numerical work lives in
:mod:`spectra.review` and :mod:`spectra.diagnostics.calibration_eval`.
They are imported lazily so that ``import spectra`` (and importing this
package) does not pull in PySide6 unless a panel is actually requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["CalibrationPanel", "ReferenceAgreementPanel"]

if TYPE_CHECKING:
    from spectra.review_gui.calibration_panel import CalibrationPanel
    from spectra.review_gui.reference_agreement_panel import (
        ReferenceAgreementPanel,
    )


def __getattr__(name: str) -> Any:
    """Lazily import the panel classes to keep Qt out of package import."""
    if name == "CalibrationPanel":
        from spectra.review_gui.calibration_panel import CalibrationPanel

        return CalibrationPanel
    if name == "ReferenceAgreementPanel":
        from spectra.review_gui.reference_agreement_panel import (
            ReferenceAgreementPanel,
        )

        return ReferenceAgreementPanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
