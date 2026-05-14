from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TimeState:
    """Temporal primitives for a single agent session."""

    session_started_at: float = field(default_factory=time.time)
    task_started_at: float = field(default_factory=time.time)
    last_user_activity_at: float = field(default_factory=time.time)
    last_progress_at: float = field(default_factory=time.time)

    estimated_remaining_steps: int = 0
    estimated_remaining_seconds: float = 0.0

    deadline: Optional[float] = None
    urgency: str = "normal"  # "low" | "normal" | "high" | "critical"

    def update_activity(self) -> None:
        self.last_user_activity_at = time.time()

    def update_progress(self) -> None:
        self.last_progress_at = time.time()

    def compute_idle_seconds(self) -> float:
        return round(time.time() - self.last_progress_at, 3)

    def compute_task_age(self) -> float:
        return round(time.time() - self.task_started_at, 3)

    def should_checkpoint(self, interval: float = 30.0) -> bool:
        """True when idle long enough to warrant a checkpoint."""
        return self.compute_idle_seconds() >= interval

    def should_replan_due_to_stall(self, threshold: float = 60.0) -> bool:
        """True when idle long enough to suggest a stall."""
        return self.compute_idle_seconds() >= threshold

    def update_estimates(
        self,
        steps_taken: int,
        steps_total: int,
        avg_step_s: float,
    ) -> None:
        remaining = max(0, steps_total - steps_taken)
        self.estimated_remaining_steps = remaining
        self.estimated_remaining_seconds = round(remaining * avg_step_s, 3)
        if self.deadline is not None:
            time_left = self.deadline - time.time()
            if time_left < self.estimated_remaining_seconds:
                self.urgency = "critical"
            elif time_left < self.estimated_remaining_seconds * 2:
                self.urgency = "high"
            else:
                self.urgency = "normal"

    def to_dict(self) -> dict:
        return {
            "session_started_at": self.session_started_at,
            "task_started_at": self.task_started_at,
            "task_age_s": self.compute_task_age(),
            "idle_s": self.compute_idle_seconds(),
            "estimated_remaining_steps": self.estimated_remaining_steps,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "urgency": self.urgency,
        }
