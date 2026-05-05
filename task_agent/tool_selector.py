"""
State-conditional tool selection.

The selector enforces two invariants before handing a step to the executor:
  1. A 'compare' step requires at least one prior observation.
  2. A 'summarize' step requires at least one prior observation.

When either invariant is violated the selector inserts a corrective search step
at the front of the pending queue and logs an ACTION_SELECTED event explaining
why the original step was deferred.

This is equivalent to a lightweight constraint-satisfaction layer: the planner
may generate an optimistic plan (compare before searching), and the selector
silently repairs the ordering based on current state.
"""
from __future__ import annotations

from .state import AgentState, PlanStep, TraceEventType


def _make_corrective_search(step_id: int, target: str, reason: str) -> PlanStep:
    return PlanStep(
        id=step_id,
        action_type="search",
        description=f"Prerequisite search: {reason}",
        target=target,
        expected_evidence="evidence required before next planned step",
        rationale=reason,
    )


def choose_next_action(state: AgentState) -> PlanStep:
    """
    Return the next PlanStep to execute, inserting corrective search steps
    when the current plan ordering would violate evidence prerequisites.
    """
    if not state.pending_steps:
        raise ValueError("choose_next_action called with no pending steps.")

    next_step = state.pending_steps[0]
    next_id = len(state.completed_steps) + len(state.pending_steps) + 100

    # ── Guard: compare needs prior observations ────────────────────────────────
    if next_step.action_type == "compare" and not state.observations:
        corrective = _make_corrective_search(
            step_id=next_id,
            target=next_step.target,
            reason="no observations available for comparison",
        )
        state.pending_steps.insert(0, corrective)
        state.log_event(
            TraceEventType.ACTION_SELECTED,
            tool_name="local_search",
            reason="Deferred compare — inserting prerequisite search",
            metadata={"deferred_step": next_step.action_type, "target": next_step.target},
        )
        return corrective

    # ── Guard: summarize needs prior observations ──────────────────────────────
    if next_step.action_type == "summarize" and not state.observations:
        corrective = _make_corrective_search(
            step_id=next_id,
            target=state.goal,
            reason="no observations available for summarization",
        )
        state.pending_steps.insert(0, corrective)
        state.log_event(
            TraceEventType.ACTION_SELECTED,
            tool_name="local_search",
            reason="Deferred summarize — inserting prerequisite search",
            metadata={"deferred_step": next_step.action_type},
        )
        return corrective

    return next_step
