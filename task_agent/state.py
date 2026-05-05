from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class TraceEventType:
    GOAL_RECEIVED = "GOAL_RECEIVED"
    PLAN_CREATED = "PLAN_CREATED"
    REASONING = "REASONING"
    ACTION_SELECTED = "ACTION_SELECTED"
    TOOL_CALLED = "TOOL_CALLED"
    OBSERVATION_RECORDED = "OBSERVATION_RECORDED"
    REFLECTION_LOGGED = "REFLECTION_LOGGED"
    EVIDENCE_GAP_DETECTED = "EVIDENCE_GAP_DETECTED"
    REPLAN_TRIGGERED = "REPLAN_TRIGGERED"
    CONFIDENCE_UPDATED = "CONFIDENCE_UPDATED"
    TERMINATED_SUCCESS = "TERMINATED_SUCCESS"
    TERMINATED_PARTIAL = "TERMINATED_PARTIAL"
    TERMINATED_FAILED = "TERMINATED_FAILED"
    TERMINATED_MAX_STEPS = "TERMINATED_MAX_STEPS"


@dataclass
class PlanStep:
    id: int
    action_type: str
    description: str
    target: str
    expected_evidence: str
    rationale: str = ""


@dataclass
class ReflectionNote:
    step: int
    assessment: str
    evidence_gaps: List[str]
    confidence_delta: float
    should_replan: bool
    evidence_quality: float = 0.0


@dataclass
class AgentPlan:
    steps: List[PlanStep]
    stopping_criteria: str
    notes: Optional[str] = None


@dataclass
class AgentState:
    goal: str
    plan: AgentPlan
    start_time: float = field(default_factory=time.time)
    completed_steps: List[PlanStep] = field(default_factory=list)
    pending_steps: List[PlanStep] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    reflections: List[ReflectionNote] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_log: List[str] = field(default_factory=list)
    failures: int = 0
    replan_count: int = 0
    confidence: str = "unknown"
    sufficiency_status: str = "unknown"
    step_count: int = 0
    final_answer: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)
    unsupported: bool = False
    no_progress_count: int = 0
    evidence_quality: float = 0.0

    def log_event(
        self,
        event_type: str,
        reason: str | None = None,
        tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.trace.append(
            {
                "step": len(self.trace) + 1,
                "event_type": event_type,
                "tool_name": tool_name,
                "reason": reason,
                "metadata": metadata or {},
                "elapsed_s": round(time.time() - self.start_time, 3),
            }
        )

    @property
    def elapsed_seconds(self) -> float:
        return round(time.time() - self.start_time, 3)

    @property
    def confidence_numeric(self) -> float:
        return self.evidence_quality

    @property
    def source_diversity(self) -> int:
        return len({obs.get("source", "") for obs in self.observations})
