"""
Termination conditions for the agent loop.

The agent stops as soon as the first matching condition is true.  Conditions
are evaluated in priority order: success > unsupported > max-steps > failures >
no-progress.  Each condition maps to a distinct trace event type so the reason
is always auditable.
"""
from __future__ import annotations

from .config import EVIDENCE_QUALITY_HIGH, FAILURE_BUDGET, NO_PROGRESS_BUDGET
from .state import AgentState, TraceEventType


def should_terminate(state: AgentState, max_steps: int = 10) -> tuple[bool, str]:
    """Return (stop, trace_event_type).  Called after every step."""

    # ── 1. Explicit success ────────────────────────────────────────────────────
    if state.sufficiency_status == "sufficient":
        return True, TraceEventType.TERMINATED_SUCCESS

    # ── 2. Goal unsupported by corpus ─────────────────────────────────────────
    if state.unsupported:
        return True, TraceEventType.TERMINATED_FAILED

    # ── 3. Step budget exhausted ───────────────────────────────────────────────
    if state.step_count >= max_steps:
        return True, TraceEventType.TERMINATED_MAX_STEPS

    # ── 4. Failure budget exhausted ───────────────────────────────────────────
    if state.failures >= FAILURE_BUDGET:
        return True, TraceEventType.TERMINATED_PARTIAL

    # ── 5. No progress for two consecutive steps ──────────────────────────────
    if state.no_progress_count >= NO_PROGRESS_BUDGET:
        return True, TraceEventType.TERMINATED_PARTIAL

    return False, ""


def termination_summary(state: AgentState) -> dict:
    """Return a human-readable termination summary for the trace."""
    terminal_events = [
        e for e in state.trace
        if e.get("event_type", "").startswith("TERMINATED_")
    ]
    last = terminal_events[-1] if terminal_events else {}
    return {
        "status": last.get("event_type", "UNKNOWN"),
        "steps_taken": state.step_count,
        "observations_collected": len(state.observations),
        "source_diversity": state.source_diversity,
        "evidence_quality": round(state.evidence_quality, 3),
        "confidence": state.confidence,
        "replans": state.replan_count,
        "failures": state.failures,
        "elapsed_s": state.elapsed_seconds,
    }
