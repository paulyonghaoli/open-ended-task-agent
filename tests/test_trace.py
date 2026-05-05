"""Tests for trace event structure and the format_trace_line helper."""
from __future__ import annotations

from task_agent.state import AgentPlan, AgentState, TraceEventType
from task_agent.trace import format_trace_line, make_event


def test_make_event_creates_structured_trace():
    event = make_event(1, "GOAL_RECEIVED", tool_name=None, reason="Accepted goal")
    assert event["step"] == 1
    assert event["event_type"] == "GOAL_RECEIVED"
    assert event["reason"] == "Accepted goal"
    assert event["metadata"] == {}


def test_make_event_required_fields():
    event = make_event(step=1, event_type="GOAL_RECEIVED")
    assert event["step"] == 1
    assert event["event_type"] == "GOAL_RECEIVED"
    assert isinstance(event["metadata"], dict)


def test_make_event_optional_fields():
    event = make_event(
        step=2,
        event_type="TOOL_CALLED",
        tool_name="local_search",
        reason="search triggered",
        metadata={"hits": 3},
    )
    assert event["tool_name"] == "local_search"
    assert event["reason"] == "search triggered"
    assert event["metadata"]["hits"] == 3


def test_format_trace_line_contains_event_type():
    event = make_event(step=1, event_type="GOAL_RECEIVED", reason="accepted")
    line = format_trace_line(event)
    assert "GOAL_RECEIVED" in line


def test_format_trace_line_contains_tool_name():
    event = make_event(step=2, event_type="TOOL_CALLED", tool_name="local_search")
    line = format_trace_line(event)
    assert "local_search" in line


def test_format_trace_line_contains_step_number():
    event = make_event(step=7, event_type="OBSERVATION_RECORDED")
    line = format_trace_line(event)
    assert "7" in line


def test_agent_state_log_event_increments():
    plan = AgentPlan(steps=[], stopping_criteria="test")
    state = AgentState(goal="test", plan=plan)
    state.log_event("A")
    state.log_event("B")
    state.log_event("C")
    assert len(state.trace) == 3
    assert state.trace[0]["step"] == 1
    assert state.trace[1]["step"] == 2
    assert state.trace[2]["step"] == 3


def test_trace_events_have_elapsed_s():
    plan = AgentPlan(steps=[], stopping_criteria="test")
    state = AgentState(goal="test", plan=plan)
    state.log_event("GOAL_RECEIVED")
    assert "elapsed_s" in state.trace[0]
    assert isinstance(state.trace[0]["elapsed_s"], (int, float))
    assert state.trace[0]["elapsed_s"] >= 0
