"""Setup workflow page for recording, model, and inference options."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .common import add_page_header, make_empty_state


class SetupTab(QWidget):
    """Scrollable setup page with progressive-disclosure content."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workflowPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        add_page_header(
            layout,
            step="Step 1",
            title="Set up the recording",
            description=(
                "Choose an EDF recording and compatible checkpoint. Essential options "
                "stay visible; processing and uncertainty controls expand only when needed."
            ),
        )

        self.readiness_label = make_empty_state(
            "Select an EDF recording and model checkpoint to begin."
        )
        layout.addWidget(self.readiness_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content = QWidget()
        content.setObjectName("setupContent")
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(2, 2, 10, 2)
        self.content_layout.setSpacing(14)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

    def set_readiness(self, message: str, *, ready: bool) -> None:
        """Update the setup callout without exposing label styling to callers."""
        self.readiness_label.setText(message)
        self.readiness_label.setProperty("state", "ready" if ready else "empty")
        self.readiness_label.style().unpolish(self.readiness_label)
        self.readiness_label.style().polish(self.readiness_label)
