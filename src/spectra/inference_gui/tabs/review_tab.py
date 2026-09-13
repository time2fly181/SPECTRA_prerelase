"""Review workflow page containing all specialist sleep-analysis workspaces."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .common import add_page_header, make_empty_state


class ReviewTab(QWidget):
    """Own nested specialist views while presenting one workflow-level page."""

    current_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workflowPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)
        add_page_header(
            layout,
            step="Step 3",
            title="Review and refine",
            description=(
                "Inspect the whole-night hypnogram, triage uncertain epochs, and use "
                "the specialist signal reader for keyboard-first manual rescoring."
            ),
        )
        self.empty_state = make_empty_state(
            "Review workspaces are ready. Run inference to populate predictions and signals."
        )
        layout.addWidget(self.empty_state, 1)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self.current_changed)
        layout.addWidget(self.tabs, 1)
        self.set_has_results(False)

    def add_view(self, widget: QWidget, label: str) -> int:
        """Add a specialist review view and return its nested tab index."""
        return self.tabs.addTab(widget, label)

    def show_view(self, label: str) -> bool:
        """Select a specialist view by its stable user-facing label."""
        index = self.index_of(label)
        if index is None:
            return False
        self.tabs.setCurrentIndex(index)
        return True

    def index_of(self, label: str) -> int | None:
        """Return the nested index for ``label`` when present."""
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == label:
                return index
        return None

    def current_widget(self) -> QWidget | None:
        """Return the active specialist view."""
        return self.tabs.currentWidget()

    def tab_text(self, index: int) -> str:
        """Return the label at a nested index."""
        return self.tabs.tabText(index)

    def set_has_results(self, has_results: bool) -> None:
        """Show guidance only while no predictions are available."""
        self.empty_state.setVisible(not has_results)
        self.tabs.setVisible(has_results)
        self.tabs.setEnabled(has_results)
