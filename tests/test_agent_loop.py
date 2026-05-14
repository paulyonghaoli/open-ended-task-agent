"""Tests for the full agent loop — success, replanning, unsupported goals, and termination."""
from __future__ import annotations

import pytest

from task_agent import run_agent
from task_agent.agent import search_local_corpus
from task_agent.state import TraceEventType
from task_agent.tools import load_corpus


# ── Helpers ────────────────────────────────────────────────────────────────────

def event_types(state) -> list[str]:
    return [e["event_type"] for e in state.trace]


# ── Core loop behaviour ────────────────────────────────────────────────────────

def test_agent_completes_known_comparison():
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    assert state.final_answer is not None
    assert state.sufficiency_status in {"sufficient", "insufficient"}
    types = event_types(state)
    assert TraceEventType.PLAN_CREATED in types
    assert any(t.startswith("TERMINATED_") for t in types)


def test_agent_completes_study_plan_goal():
    state = run_agent("Create a study plan for learning ROS2 basics in 2 weeks")
    assert state.final_answer is not None
    types = event_types(state)
    assert TraceEventType.PLAN_CREATED in types


def test_agent_completes_web_framework_comparison():
    state = run_agent("Compare FastAPI and Flask for a beginner Python web app")
    assert state.final_answer is not None
    assert state.step_count >= 1


def test_agent_reasoning_events_present():
    """ReAct: every action should be preceded by a REASONING event."""
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    types = event_types(state)
    assert TraceEventType.REASONING in types


def test_agent_reflection_events_present():
    """Reflexion: at least one REFLECTION_LOGGED event per non-summarize step."""
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    types = event_types(state)
    assert TraceEventType.REFLECTION_LOGGED in types


def test_agent_state_has_evidence_quality():
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    assert 0.0 <= state.evidence_quality <= 1.0


def test_agent_state_has_reflections():
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    assert isinstance(state.reflections, list)


def test_agent_trace_has_elapsed_s():
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    for event in state.trace:
        assert "elapsed_s" in event
        assert isinstance(event["elapsed_s"], (int, float))


# ── Replanning behaviour ───────────────────────────────────────────────────────

def test_agent_replans_after_empty_search(monkeypatch):
    corpus = load_corpus()
    original_search = search_local_corpus
    calls = {"count": 0}

    def failing_once(query, corpus_data, max_results=3):
        calls["count"] += 1
        if calls["count"] == 1:
            return []
        return original_search(query, corpus_data, max_results)

    monkeypatch.setattr("task_agent.agent.search_local_corpus", failing_once)
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    assert state.replan_count >= 1
    assert state.final_answer is not None


def test_agent_replan_event_logged_on_empty_search(monkeypatch):
    original_search = search_local_corpus

    calls = {"n": 0}

    def always_fail(query, corpus_data, max_results=3):
        calls["n"] += 1
        if calls["n"] <= 2:
            return []
        return original_search(query, corpus_data, max_results)

    monkeypatch.setattr("task_agent.agent.search_local_corpus", always_fail)
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    types = event_types(state)
    assert TraceEventType.REPLAN_TRIGGERED in types


# ── Failure and unsupported goals ─────────────────────────────────────────────

def test_agent_handles_unsupported_goal():
    # For goals with no corpus coverage the agent should either set
    # unsupported=True or produce a low-confidence answer — both are
    # valid out-of-scope signals.  High or medium confidence on an
    # unrelated goal would be a false positive.
    state = run_agent("Write a haiku about quantum mechanics and superconductors")
    out_of_scope = state.unsupported or state.confidence in ("low", "unknown")
    assert out_of_scope, (
        f"Expected out-of-scope signal, got unsupported={state.unsupported} "
        f"confidence={state.confidence}"
    )
    assert state.confidence not in ("high", "medium")


def test_agent_unsupported_produces_answer():
    """Even for unsupported goals the agent should produce some answer string."""
    state = run_agent("Solve the P vs NP problem")
    assert state.final_answer is not None
    assert len(state.final_answer) > 10


# ── Max-step termination ───────────────────────────────────────────────────────

def test_agent_terminates_by_max_steps():
    state = run_agent("Compare DuckDB and SQLite for a local analytics app", max_steps=1)
    assert state.step_count == 1
    types = event_types(state)
    terminal = [t for t in types if t.startswith("TERMINATED_")]
    assert terminal[-1] == TraceEventType.TERMINATED_MAX_STEPS


def test_agent_respects_max_steps_budget():
    for budget in (2, 4):
        state = run_agent("Compare LangGraph and AutoGen", max_steps=budget)
        assert state.step_count <= budget


# ── Trace structural integrity ─────────────────────────────────────────────────

def test_trace_first_event_is_goal_received():
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    assert state.trace[0]["event_type"] == TraceEventType.GOAL_RECEIVED


def test_trace_second_event_is_routing_decided():
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    assert state.trace[1]["event_type"] == TraceEventType.ROUTING_DECIDED


def test_trace_third_event_is_plan_created():
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    assert state.trace[2]["event_type"] == TraceEventType.PLAN_CREATED


def test_trace_steps_are_sequential():
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    for i, event in enumerate(state.trace, 1):
        assert event["step"] == i
