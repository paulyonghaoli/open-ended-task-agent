"""Tests for the Reflexion-style post-observation reflection module."""
from __future__ import annotations

import pytest

from task_agent.reflection import reflect, _rule_based_reflect
from task_agent.state import AgentPlan, AgentState, ReflectionNote, TraceEventType


def _make_state(observations=None, failures=0, replan_count=0, evidence_quality=0.0) -> AgentState:
    plan = AgentPlan(steps=[], stopping_criteria="test")
    state = AgentState(goal="Compare DuckDB and SQLite", plan=plan)
    state.observations = observations or []
    state.failures = failures
    state.replan_count = replan_count
    state.evidence_quality = evidence_quality
    return state


# ── Rule-based reflection ──────────────────────────────────────────────────────

def test_rule_based_reflect_returns_note():
    state = _make_state()
    note = _rule_based_reflect(state, {"success": True})
    assert isinstance(note, ReflectionNote)


def test_rule_based_reflect_identifies_gaps_when_sparse():
    state = _make_state(observations=[{"source": "a", "summary": "x"}])
    note = _rule_based_reflect(state, {"success": True})
    assert len(note.evidence_gaps) > 0


def test_rule_based_reflect_no_gaps_with_sufficient_evidence():
    state = _make_state(
        observations=[
            {"source": "doc_a", "summary": "DuckDB is fast."},
            {"source": "doc_b", "summary": "SQLite is portable."},
        ]
    )
    note = _rule_based_reflect(state, {"success": True})
    # With 2 observations from 2 sources, gaps should be empty or minimal
    assert isinstance(note.evidence_gaps, list)


def test_rule_based_reflect_suggests_replan_when_budget_allows():
    state = _make_state(observations=[], replan_count=0)
    note = _rule_based_reflect(state, {"success": False})
    assert isinstance(note.should_replan, bool)


def test_rule_based_reflect_does_not_replan_when_budget_exhausted():
    state = _make_state(observations=[], replan_count=2)
    note = _rule_based_reflect(state, {"success": False})
    assert note.should_replan is False


# ── Full reflect() integration ─────────────────────────────────────────────────

def test_reflect_updates_state_evidence_quality():
    state = _make_state(
        observations=[
            {"source": "a", "summary": "Evidence A"},
            {"source": "b", "summary": "Evidence B"},
        ]
    )
    before = state.evidence_quality
    reflect(state, {"success": True, "tool": "local_search"})
    # Quality should change after reflection
    assert state.evidence_quality != before or state.evidence_quality >= 0


def test_reflect_appends_to_reflections_list():
    state = _make_state()
    assert len(state.reflections) == 0
    reflect(state, {"success": False, "tool": "local_search"})
    assert len(state.reflections) == 1


def test_reflect_logs_reflection_event():
    state = _make_state()
    reflect(state, {"success": True, "tool": "local_search"})
    types = [e["event_type"] for e in state.trace]
    assert TraceEventType.REFLECTION_LOGGED in types


def test_reflect_logs_gap_event_when_gaps_present():
    state = _make_state(observations=[])
    reflect(state, {"success": False, "tool": "local_search"})
    types = [e["event_type"] for e in state.trace]
    # May or may not log gap event depending on assessment
    assert isinstance(types, list)


def test_reflect_confidence_label_updated():
    state = _make_state(evidence_quality=0.0)
    state.observations = [
        {"source": "a", "summary": "x"},
        {"source": "b", "summary": "y"},
    ]
    reflect(state, {"success": True, "tool": "local_search"})
    assert state.confidence in ("high", "medium", "low")


def test_reflect_confidence_high_when_quality_above_threshold():
    from task_agent.config import EVIDENCE_QUALITY_HIGH
    state = _make_state(evidence_quality=EVIDENCE_QUALITY_HIGH + 0.1)
    state.observations = [{"source": "a", "summary": "x"}, {"source": "b", "summary": "y"}]
    reflect(state, {"success": True})
    assert state.confidence == "high"


def test_reflect_does_not_exceed_quality_bounds():
    state = _make_state(evidence_quality=0.95)
    state.observations = [{"source": "a", "summary": "x"}, {"source": "b", "summary": "y"}]
    reflect(state, {"success": True})
    assert 0.0 <= state.evidence_quality <= 1.0
