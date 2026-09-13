"""Shared presentation helpers for inference workflow pages."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


def add_page_header(
    layout: QVBoxLayout,
    *,
    step: str,
    title: str,
    description: str,
) -> None:
    """Add a consistent workflow step, title, and description to a page."""
    header = QWidget()
    header.setObjectName("pageHeader")
    header.setSizePolicy(
        QSizePolicy.Policy.Preferred,
        QSizePolicy.Policy.Maximum,
    )
    header_layout = QVBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(4)

    step_label = QLabel(step.upper())
    step_label.setObjectName("stepBadge")
    title_label = QLabel(title)
    title_label.setObjectName("pageTitle")
    description_label = QLabel(description)
    description_label.setObjectName("pageSubtitle")
    description_label.setWordWrap(True)

    header_layout.addWidget(step_label)
    header_layout.addWidget(title_label)
    header_layout.addWidget(description_label)
    layout.addWidget(header, 0)


def make_empty_state(text: str) -> QLabel:
    """Create a reusable, accessible empty-state callout."""
    label = QLabel(text)
    label.setObjectName("emptyState")
    label.setWordWrap(True)
    label.setMinimumHeight(58)
    return label
