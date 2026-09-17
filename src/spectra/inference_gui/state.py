"""Toolkit-independent state for the PSG inference application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class InferenceRunState:
    """Mutable state for one foreground inference run."""

    started_at: float | None = None
    progress_percent: int = 0
    phase: str = ""
    last_progress_at: float | None = None
    last_epoch_count: int | None = None

    def begin(self, started_at: float) -> None:
        """Reset transient measurements for a newly started run."""
        self.started_at = float(started_at)
        self.progress_percent = 0
        self.phase = "Loading"
        self.last_progress_at = None
        self.last_epoch_count = None

    def finish(self) -> None:
        """Clear timing data once the run reaches a terminal state."""
        self.started_at = None
        self.last_progress_at = None
        self.last_epoch_count = None


@dataclass(slots=True)
class ReviewSessionState:
    """Prediction, review, and waveform state for the selected recording.

    Prediction arrays use the original EDF epoch grid with labels 0-4 in
    Wake/N1/N2/N3/REM order and -1 for unscored epochs. ``base_predictions``
    holds model output; ``predictions`` includes manual overrides. Overrides
    do not recompute probabilities, confidence, or uncertainty arrays.
    """

    base_predictions: np.ndarray | None = None
    predictions: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    confidences: np.ndarray | None = None
    flag_scores: dict[str, np.ndarray] | None = None
    reference_predictions: np.ndarray | None = None
    output_paths: dict[str, str] = field(default_factory=dict)
    manual_overrides: dict[int, int] = field(default_factory=dict)
    selected_epoch: int = 0
    channel_layout: list[str] | None = None
    signal_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = field(
        default_factory=dict
    )
    signal_payload: dict[str, Any] | None = None

    def set_override(self, epoch: int, stage: int | None) -> bool:
        """Set or clear an epoch override and refresh effective predictions.

        Returns:
            ``True`` when the effective session state changed.
        """
        epoch_idx = int(epoch)
        if epoch_idx < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch_idx}")
        if self.base_predictions is not None and epoch_idx >= len(
            self.base_predictions
        ):
            raise ValueError(
                f"epoch {epoch_idx} is outside {len(self.base_predictions)} predictions"
            )
        if stage is not None and not 0 <= int(stage) <= 4:
            raise ValueError(f"stage must be in [0, 4], got {stage}")

        previous = self.manual_overrides.get(epoch_idx)
        if stage is None:
            if previous is None:
                return False
            self.manual_overrides.pop(epoch_idx)
        else:
            stage_idx = int(stage)
            if previous == stage_idx:
                return False
            self.manual_overrides[epoch_idx] = stage_idx
        self.refresh_effective_predictions()
        return True

    def replace_overrides(self, overrides: dict[int, int]) -> bool:
        """Replace all manual scores after validating their epoch and stage values."""
        normalized: dict[int, int] = {}
        for epoch, stage in overrides.items():
            epoch_idx = int(epoch)
            stage_idx = int(stage)
            if epoch_idx < 0:
                raise ValueError(f"epoch must be non-negative, got {epoch_idx}")
            if self.base_predictions is not None and epoch_idx >= len(
                self.base_predictions
            ):
                raise ValueError(
                    f"epoch {epoch_idx} is outside {len(self.base_predictions)} predictions"
                )
            if not 0 <= stage_idx <= 4:
                raise ValueError(f"stage must be in [0, 4], got {stage_idx}")
            normalized[epoch_idx] = stage_idx
        if normalized == self.manual_overrides:
            return False
        self.manual_overrides = normalized
        self.refresh_effective_predictions()
        return True

    def refresh_effective_predictions(self) -> np.ndarray | None:
        """Recompute final predictions from the immutable model output plus overrides."""
        if self.base_predictions is None:
            self.predictions = None
            return None
        effective = np.asarray(self.base_predictions, dtype=np.int64).copy()
        for epoch, stage in self.manual_overrides.items():
            if 0 <= epoch < len(effective):
                effective[epoch] = stage
        self.predictions = effective
        return effective

    def clear_results(self, *, clear_overrides: bool = True) -> None:
        """Clear recording-derived results while retaining reusable signal cache."""
        self.base_predictions = None
        self.predictions = None
        self.probabilities = None
        self.confidences = None
        self.flag_scores = None
        self.reference_predictions = None
        self.output_paths.clear()
        self.selected_epoch = 0
        self.channel_layout = None
        self.signal_payload = None
        if clear_overrides:
            self.manual_overrides.clear()


@dataclass(slots=True)
class ProjectState:
    """Persistence state for the currently open PSGStage project."""

    path: Path | None = None
    modified: bool = False
    output_template: str = "{basename}"

    @property
    def display_name(self) -> str:
        """Return a stable human-facing project name."""
        return self.path.stem if self.path is not None else "Untitled"

    def reset(self) -> None:
        """Reset to a new unsaved project."""
        self.path = None
        self.modified = False
        self.output_template = "{basename}"

    def mark_modified(self) -> None:
        """Mark the project dirty after a user-visible state change."""
        self.modified = True

    def mark_saved(self, path: str | Path) -> None:
        """Record the saved path and clear the dirty flag."""
        self.path = Path(path)
        self.modified = False
