"""Qt panel showing model calibration against a reference hypnogram.

Displays ECE / MCE / Brier / accuracy / over-confidence from
:func:`spectra.diagnostics.calibration_eval.evaluate_calibration` plus a
reliability diagram from
:func:`spectra.diagnostics.calibration_eval.reliability_curve`. All math
lives in ``calibration_eval``; this module only draws the result and lets the
user change the bin count.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spectra.diagnostics.calibration_eval import (
    evaluate_calibration,
    reliability_curve,
)

_FACE_COLOR = "#1e1e1e"
_TEXT_COLOR = "#e0e0e0"


class CalibrationPanel(QWidget):
    """Panel reporting probability calibration against a reference hypnogram.

    Call :meth:`set_data` with the per-epoch probabilities and the parsed
    reference labels; :meth:`clear` resets to the empty state.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the panel UI (bin control + metrics + reliability diagram)."""
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._probabilities: np.ndarray | None = None
        self._labels: np.ndarray | None = None

        layout = QVBoxLayout(self)

        self._placeholder = QLabel(
            "Load a reference hypnogram to compute calibration (ECE, Brier, "
            "reliability diagram)."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        layout.addWidget(self._placeholder)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Bins:"))
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(2, 50)
        self._bins_spin.setValue(15)
        self._bins_spin.valueChanged.connect(self._on_bins_changed)
        controls.addWidget(self._bins_spin)
        controls.addStretch(1)
        content_layout.addLayout(controls)

        metrics_group = QGroupBox("Calibration metrics")
        metrics_layout = QGridLayout(metrics_group)
        self._metric_labels: dict[str, QLabel] = {}
        metric_specs = [
            ("ece", "Expected calibration error (ECE):"),
            ("mce", "Maximum calibration error (MCE):"),
            ("brier", "Brier score:"),
            ("accuracy", "Accuracy:"),
            ("avg_confidence", "Average confidence:"),
            ("overconfidence", "Over-confidence:"),
            ("num_samples", "Scored epochs:"),
        ]
        for row, (key, text) in enumerate(metric_specs):
            metrics_layout.addWidget(QLabel(text), row, 0)
            value_label = QLabel("-")
            self._metric_labels[key] = value_label
            metrics_layout.addWidget(value_label, row, 1)
        content_layout.addWidget(metrics_group)

        diagram_group = QGroupBox("Reliability diagram")
        diagram_layout = QVBoxLayout(diagram_group)
        # Defer the Qt matplotlib backend import to construction time.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        self._figure = Figure(figsize=(5, 4), facecolor=_FACE_COLOR)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._canvas.setMinimumHeight(320)
        diagram_layout.addWidget(self._canvas)
        content_layout.addWidget(diagram_group)

        layout.addWidget(self._content)
        self._content.setVisible(False)

    def clear(self) -> None:
        """Reset the panel to its empty (no-reference) state."""
        self._probabilities = None
        self._labels = None
        self._content.setVisible(False)
        self._placeholder.setVisible(True)
        self._figure.clear()
        self._canvas.draw_idle()

    def set_data(
        self, probabilities: np.ndarray | None, labels: np.ndarray | None
    ) -> None:
        """Store data and render calibration metrics + reliability diagram.

        Args:
            probabilities: Per-epoch posterior of shape ``(n_epochs, n_classes)``.
            labels: Parsed reference stage array of shape ``(n_epochs,)``.
        """
        if probabilities is None or labels is None or len(labels) == 0:
            self.clear()
            return

        probs = np.asarray(probabilities)
        labels_arr = np.asarray(labels)
        # Align lengths (reference and predictions may differ by a few epochs).
        min_len = min(probs.shape[0], labels_arr.shape[0])
        probs = probs[:min_len]
        labels_arr = labels_arr[:min_len]
        # The runtime writes zero rows for unscored epochs and signal gaps.
        # They are not posteriors and must not contribute to calibration.
        scored = (
            np.isfinite(probs).all(axis=1)
            & (probs >= 0).all(axis=1)
            & np.isclose(probs.sum(axis=1), 1.0)
            & (labels_arr >= 0)
            & (labels_arr < probs.shape[1])
        )
        if not np.any(scored):
            self.clear()
            return
        self._probabilities = probs[scored]
        self._labels = labels_arr[scored]

        self._placeholder.setVisible(False)
        self._content.setVisible(True)
        self._recompute()

    def _on_bins_changed(self, _value: int) -> None:
        """Re-render when the user changes the bin count."""
        if self._probabilities is not None and self._labels is not None:
            self._recompute()

    def _recompute(self) -> None:
        """Compute metrics + reliability curve for the current data and bins."""
        assert self._probabilities is not None and self._labels is not None
        n_bins = int(self._bins_spin.value())

        metrics = evaluate_calibration(self._probabilities, self._labels, n_bins=n_bins)
        self._metric_labels["ece"].setText(f"{float(metrics['ece']):.4f}")
        self._metric_labels["mce"].setText(f"{float(metrics['mce']):.4f}")
        self._metric_labels["brier"].setText(f"{float(metrics['brier']):.4f}")
        self._metric_labels["accuracy"].setText(f"{float(metrics['accuracy']):.1%}")
        self._metric_labels["avg_confidence"].setText(
            f"{float(metrics['avg_confidence']):.1%}"
        )
        self._metric_labels["overconfidence"].setText(
            f"{float(metrics['overconfidence']):+.4f}"
        )
        self._metric_labels["num_samples"].setText(str(int(metrics["num_samples"])))

        curve = reliability_curve(self._probabilities, self._labels, n_bins=n_bins)
        self._draw_reliability(
            curve.bin_confidence, curve.bin_accuracy, curve.bin_count
        )

    def _draw_reliability(
        self,
        bin_confidence: np.ndarray,
        bin_accuracy: np.ndarray,
        bin_count: np.ndarray,
    ) -> None:
        """Plot accuracy vs confidence per bin against the y=x diagonal."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor(_FACE_COLOR)

        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Perfect")

        populated = bin_count > 0
        if np.any(populated):
            ax.plot(
                bin_confidence[populated],
                bin_accuracy[populated],
                marker="o",
                color="#42A5F5",
                label="Model",
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Confidence", color=_TEXT_COLOR)
        ax.set_ylabel("Accuracy", color=_TEXT_COLOR)
        ax.tick_params(colors=_TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#555555")
        legend = ax.legend(facecolor=_FACE_COLOR, edgecolor="#555555")
        for text in legend.get_texts():
            text.set_color(_TEXT_COLOR)

        self._figure.tight_layout()
        self._canvas.draw_idle()
