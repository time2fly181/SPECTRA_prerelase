"""Export workflow page for clinical, data, and reporting outputs."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QVBoxLayout, QWidget

from .common import add_page_header, make_empty_state


class ExportTab(QWidget):
    """Group export actions by output intent and manage their empty state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workflowPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        add_page_header(
            layout,
            step="Step 4",
            title="Export results",
            description=(
                "Save clinical annotations, epoch-level data, publication-ready figures, "
                "or a complete PDF report from the reviewed scores."
            ),
        )
        self.empty_state = make_empty_state(
            "No results are available yet. Complete inference before exporting."
        )
        layout.addWidget(self.empty_state)

        self.figure_group, self.figure_actions = self._make_group("Figures")
        self.clinical_group, self.clinical_actions = self._make_group(
            "Clinical and epoch data"
        )
        self.report_group, self.report_actions = self._make_group("Reports")
        layout.addWidget(self.figure_group)
        layout.addWidget(self.clinical_group)
        layout.addWidget(self.report_group)
        layout.addStretch()
        self.set_has_results(False)

    @staticmethod
    def _make_group(title: str) -> tuple[QGroupBox, QHBoxLayout]:
        group = QGroupBox(title)
        actions = QHBoxLayout(group)
        actions.setSpacing(10)
        actions.addStretch()
        return group, actions

    @staticmethod
    def add_action(layout: QHBoxLayout, widget: QWidget) -> None:
        """Insert an action before the group's trailing stretch."""
        layout.insertWidget(max(0, layout.count() - 1), widget)

    def set_has_results(self, has_results: bool) -> None:
        """Switch between guidance and available export groups."""
        self.empty_state.setVisible(not has_results)
        self.figure_group.setVisible(has_results)
        self.clinical_group.setVisible(has_results)
        self.report_group.setVisible(has_results)
