"""Qt panel showing model-vs-reference hypnogram agreement.

Renders overall accuracy, Cohen's kappa and a 5x5 confusion heatmap from
:func:`spectra.review.agreement.compute_agreement`. All math is done in
the pure ``review`` package; this module only draws the result.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from spectra.review.agreement import compute_agreement

_STAGE_NAMES: tuple[str, ...] = ("Wake", "N1", "N2", "N3", "REM")
_FACE_COLOR = "#1e1e1e"
_TEXT_COLOR = "#e0e0e0"


class ReferenceAgreementPanel(QWidget):
    """Panel comparing model predictions against a reference hypnogram.

    Call :meth:`set_data` with the model predictions and the parsed reference
    array to populate; :meth:`clear` resets to the empty state.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the panel UI (metrics header + confusion heatmap)."""
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)

        self._placeholder = QLabel(
            "Load a reference hypnogram to compare it against the model's scoring."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        layout.addWidget(self._placeholder)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        metrics_group = QGroupBox("Agreement")
        metrics_layout = QGridLayout(metrics_group)
        self._accuracy_label = QLabel("-")
        self._kappa_label = QLabel("-")
        self._scored_label = QLabel("-")
        metrics_layout.addWidget(QLabel("Overall accuracy:"), 0, 0)
        metrics_layout.addWidget(self._accuracy_label, 0, 1)
        metrics_layout.addWidget(QLabel("Cohen's kappa:"), 1, 0)
        metrics_layout.addWidget(self._kappa_label, 1, 1)
        metrics_layout.addWidget(QLabel("Scored epochs:"), 2, 0)
        metrics_layout.addWidget(self._scored_label, 2, 1)
        content_layout.addWidget(metrics_group)

        heatmap_group = QGroupBox("Confusion (rows = reference, cols = model)")
        heatmap_layout = QVBoxLayout(heatmap_group)
        # Defer the Qt matplotlib backend import to construction time (after the
        # QApplication exists), matching the GUI's SIGSEGV-avoidance strategy.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        self._figure = Figure(figsize=(5, 4), facecolor=_FACE_COLOR)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._canvas.setMinimumHeight(320)
        heatmap_layout.addWidget(self._canvas)
        content_layout.addWidget(heatmap_group)

        layout.addWidget(self._content)
        self._content.setVisible(False)

    def clear(self) -> None:
        """Reset the panel to its empty (no-reference) state."""
        self._content.setVisible(False)
        self._placeholder.setVisible(True)
        self._figure.clear()
        self._canvas.draw_idle()

    def set_data(
        self, model_pred: np.ndarray | None, reference: np.ndarray | None
    ) -> None:
        """Compute and render agreement between predictions and reference.

        Args:
            model_pred: Per-epoch model predictions, or ``None``.
            reference: Parsed reference stage array, or ``None``.
        """
        if model_pred is None or reference is None or len(reference) == 0:
            self.clear()
            return

        result = compute_agreement(np.asarray(model_pred), np.asarray(reference))

        self._placeholder.setVisible(False)
        self._content.setVisible(True)

        self._accuracy_label.setText(f"{result.accuracy:.1%}")
        self._kappa_label.setText(f"{result.cohen_kappa:.3f}")
        self._scored_label.setText(str(result.n_scored))

        self._draw_confusion(result.confusion)

    def _draw_confusion(self, confusion: np.ndarray) -> None:
        """Render the confusion matrix as a heatmap with count annotations."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor(_FACE_COLOR)

        # Row-normalize for color so stage prevalence doesn't wash out the map.
        row_sums = confusion.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized = np.where(row_sums > 0, confusion / row_sums, 0.0)

        im = ax.imshow(normalized, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")

        n = len(_STAGE_NAMES)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(_STAGE_NAMES, color=_TEXT_COLOR)
        ax.set_yticklabels(_STAGE_NAMES, color=_TEXT_COLOR)
        ax.set_xlabel("Model", color=_TEXT_COLOR)
        ax.set_ylabel("Reference", color=_TEXT_COLOR)

        for r in range(n):
            for c in range(n):
                count = int(confusion[r, c])
                ax.text(
                    c,
                    r,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if normalized[r, c] < 0.6 else "black",
                    fontsize=9,
                )

        cbar = self._figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=_TEXT_COLOR)
        self._figure.tight_layout()
        self._canvas.draw_idle()
