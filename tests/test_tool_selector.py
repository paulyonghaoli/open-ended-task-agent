"""Tests for state-conditional tool selection."""
from __future__ import annotations

import pytest

from task_agent.state import AgentPlan, AgentState, PlanStep
from task_agent.tool_selector import choose_next_action


def _state_with_steps(*steps: PlanStep, observations=None) -> AgentState:
    plan = AgentPlan(steps=list(steps), stopping_criteria="test")
    state = AgentState(
        goal="test goal",
        plan=plan,
        pending_steps=list(steps),
        observations=observations or [],
    )
    return state


def _step(action_type: str, target: str = "target") -> PlanStep:
    return PlanStep(
        id=1,
        action_type=action_type,
        description=f"Step: {action_type}",
        target=target,
        expected_evidence="evidence",
    )


# ── Prerequisite guards ────────────────────────────────────────────────────────

def test_choose_next_action_inserts_search_before_compare():
    state = _state_with_steps(_step("compare", "A,B"))
    next_action = choose_next_action(state)
    assert next_action.action_type == "search"
    assert state.pending_steps[0].action_type == "search"


def test_compare_not_deferred_when_observations_present():
    obs = [{"source": "doc", "summary": "Some evidence."}]
    state = _state_with_steps(_step("compare", "A,B"), observations=obs)
    next_action = choose_next_action(state)
    assert next_action.action_type == "compare"


def test_choose_next_action_inserts_search_before_summarize_when_empty():
    state = _state_with_steps(_step("summarize"))
    next_action = choose_next_action(state)
    assert next_action.action_type == "search"


def test_choose_next_action_allows_summarize_when_evidence_present():
    obs = [{"source": "doc", "summary": "Evidence."}]
    state = _state_with_steps(_step("summarize"), observations=obs)
    next_action = choose_next_action(state)
    assert next_action.action_type == "summarize"


def test_choose_next_action_returns_search_directly():
    state = _state_with_steps(_step("search"))
    next_action = choose_next_action(state)
    assert next_action.action_type == "search"


def test_choose_next_action_raises_on_empty_pending():
    plan = AgentPlan(steps=[], stopping_criteria="test")
    state = AgentState(goal="test", plan=plan)
    with pytest.raises(ValueError):
        choose_next_action(state)


def test_corrective_search_uses_original_target():
    state = _state_with_steps(_step("compare", "DuckDB,SQLite"))
    corrective = choose_next_action(state)
    assert "DuckDB" in corrective.target or "SQLite" in corrective.target


def test_corrective_search_has_rationale():
    state = _state_with_steps(_step("compare", "A,B"))
    corrective = choose_next_action(state)
    assert len(corrective.rationale) > 0


def test_deferred_compare_stays_in_pending():
    state = _state_with_steps(_step("compare", "A,B"), _step("summarize"))
    choose_next_action(state)
    # compare step should still be in pending (not removed)
    action_types = [s.action_type for s in state.pending_steps]
    assert "compare" in action_types


def test_trace_event_logged_for_corrective_insert():
    state = _state_with_steps(_step("compare", "A,B"))
    choose_next_action(state)
    types = [e["event_type"] for e in state.trace]
    assert "ACTION_SELECTED" in types
