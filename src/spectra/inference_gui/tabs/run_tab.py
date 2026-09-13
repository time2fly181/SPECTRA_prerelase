"""Run workflow page for inference progress, controls, and logs."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .common import add_page_header, make_empty_state


class RunTab(QWidget):
    """Host long-running inference feedback behind a small layout interface."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workflowPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        add_page_header(
            layout,
            step="Step 2",
            title="Run inference",
            description=(
                "Monitor preprocessing, model inference, post-processing, and saved "
                "outputs without blocking the application."
            ),
        )
        self.empty_state = make_empty_state(
            "Ready to run. Return to Setup if the recording or model still needs attention."
        )
        layout.addWidget(self.empty_state)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(12)
        layout.addLayout(self.content_layout, 1)

        self.actions_layout = QHBoxLayout()
        self.actions_layout.setSpacing(10)
        layout.addLayout(self.actions_layout)

    def set_activity(self, message: str, *, active: bool) -> None:
        """Present the current run state in the page callout."""
        self.empty_state.setText(message)
        self.empty_state.setProperty("state", "active" if active else "empty")
        self.empty_state.style().unpolish(self.empty_state)
        self.empty_state.style().polish(self.empty_state)
