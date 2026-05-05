"""Tests for termination conditions and termination_summary."""
from __future__ import annotations

import pytest

from task_agent.state import AgentPlan, AgentState, TraceEventType
from task_agent.termination import should_terminate, termination_summary


def _make_state(**kwargs) -> AgentState:
    plan = AgentPlan(steps=[], stopping_criteria="test")
    state = AgentState(goal="test goal", plan=plan)
    for k, v in kwargs.items():
        setattr(state, k, v)
    return state


# ── should_terminate ───────────────────────────────────────────────────────────

def test_terminates_on_success():
    state = _make_state(sufficiency_status="sufficient")
    stop, reason = should_terminate(state)
    assert stop is True
    assert reason == TraceEventType.TERMINATED_SUCCESS


def test_terminates_on_max_steps():
    state = _make_state(step_count=8)
    stop, reason = should_terminate(state, max_steps=8)
    assert stop is True
    assert reason == TraceEventType.TERMINATED_MAX_STEPS


def test_does_not_terminate_below_max_steps():
    state = _make_state(step_count=5)
    stop, _ = should_terminate(state, max_steps=10)
    assert stop is False


def test_terminates_on_repeated_failures():
    state = _make_state(failures=3)
    stop, reason = should_terminate(state)
    assert stop is True
    assert reason == TraceEventType.TERMINATED_PARTIAL


def test_terminates_on_unsupported():
    state = _make_state(unsupported=True)
    stop, reason = should_terminate(state)
    assert stop is True
    assert reason == TraceEventType.TERMINATED_FAILED


def test_terminates_on_no_progress():
    state = _make_state(no_progress_count=2)
    stop, reason = should_terminate(state)
    assert stop is True
    assert reason == TraceEventType.TERMINATED_PARTIAL


def test_does_not_terminate_when_healthy():
    state = _make_state(step_count=3, failures=1, no_progress_count=0)
    stop, _ = should_terminate(state, max_steps=10)
    assert stop is False


def test_success_takes_priority_over_max_steps():
    """Sufficient evidence should terminate as SUCCESS even if step budget is hit."""
    state = _make_state(sufficiency_status="sufficient", step_count=10)
    stop, reason = should_terminate(state, max_steps=10)
    assert reason == TraceEventType.TERMINATED_SUCCESS


# ── termination_summary ────────────────────────────────────────────────────────

def test_termination_summary_keys():
    state = _make_state(step_count=4, failures=1, evidence_quality=0.6, confidence="medium")
    state.observations = [{"source": "a", "summary": "x"}, {"source": "b", "summary": "y"}]
    state.log_event(TraceEventType.TERMINATED_SUCCESS, reason="done")
    summary = termination_summary(state)
    assert summary["status"] == TraceEventType.TERMINATED_SUCCESS
    assert summary["steps_taken"] == 4
    assert summary["observations_collected"] == 2
    assert summary["evidence_quality"] == pytest.approx(0.6, abs=0.01)
    assert summary["confidence"] == "medium"


def test_termination_summary_source_diversity():
    state = _make_state()
    state.observations = [
        {"source": "doc_a", "summary": "x"},
        {"source": "doc_b", "summary": "y"},
        {"source": "doc_a", "summary": "z"},  # duplicate
    ]
    state.log_event(TraceEventType.TERMINATED_SUCCESS)
    summary = termination_summary(state)
    assert summary["source_diversity"] == 2
