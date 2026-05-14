"""Tests for the verification report module."""
from __future__ import annotations

import pytest

from task_agent import run_agent
from task_agent.verifier import VerificationReport, verify_state


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_state(goal="Compare A and B", observations=None, confidence="medium",
                evidence_quality=0.4, unsupported=False, tool_calls=None):
    from task_agent.state import AgentPlan, AgentState, PlanStep
    plan = AgentPlan(steps=[], stopping_criteria="test", notes="test")
    state = AgentState(goal=goal, plan=plan)
    state.observations = observations or []
    state.confidence = confidence
    state.evidence_quality = evidence_quality
    state.unsupported = unsupported
    state.tool_calls = tool_calls or []
    return state


# ── Basic structure ────────────────────────────────────────────────────────────

def test_verifier_report_is_dataclass():
    r = VerificationReport(passed=True)
    assert r.passed is True
    assert r.issues == []
    assert r.missing_evidence == []
    assert r.contradiction_risk == 0.0


def test_to_dict_has_required_keys():
    r = VerificationReport(passed=False, issues=["a"], contradiction_risk=0.3)
    d = r.to_dict()
    for key in ("passed", "issues", "missing_evidence", "contradiction_risk", "confidence_adjustment"):
        assert key in d


# ── No evidence ────────────────────────────────────────────────────────────────

def test_no_evidence_fails():
    state = _make_state(observations=[])
    r = verify_state(state)
    assert r.passed is False
    assert any("No evidence" in i for i in r.issues)


def test_no_evidence_has_negative_confidence_adjustment():
    state = _make_state(observations=[])
    r = verify_state(state)
    assert r.confidence_adjustment < 0


# ── One-sided comparison ───────────────────────────────────────────────────────

def test_one_sided_comparison_flagged():
    obs = [{"source": "duckdb_vs_sqlite", "summary": "DuckDB is fast"}]
    state = _make_state(
        goal="Compare DuckDB and SQLite",
        observations=obs,
    )
    r = verify_state(state)
    assert not r.passed
    assert any("one source" in i for i in r.issues)


def test_two_source_comparison_passes_this_check():
    obs = [
        {"source": "duckdb_doc", "summary": "DuckDB is fast"},
        {"source": "sqlite_doc", "summary": "SQLite is lightweight"},
    ]
    state = _make_state(goal="Compare DuckDB and SQLite", observations=obs)
    r = verify_state(state)
    # Should not flag one-sided comparison
    assert not any("one source" in i for i in r.issues)


# ── High confidence with low evidence ─────────────────────────────────────────

def test_high_confidence_low_quality_flagged():
    obs = [{"source": "x", "summary": "some text"}]
    state = _make_state(observations=obs, confidence="high", evidence_quality=0.05)
    r = verify_state(state)
    assert not r.passed
    assert any("high" in i.lower() for i in r.issues)


# ── Repeated failed searches ───────────────────────────────────────────────────

def test_repeated_failed_searches_flagged():
    tool_calls = [
        {"tool": "local_search", "target": "foo", "success": False},
        {"tool": "local_search", "target": "bar", "success": False},
    ]
    state = _make_state(tool_calls=tool_calls)
    r = verify_state(state)
    assert any("search" in i.lower() or "failed" in i.lower() for i in r.issues)


# ── Clean state passes ─────────────────────────────────────────────────────────

def test_clean_state_passes():
    obs = [
        {"source": "doc_a", "summary": "Evidence A"},
        {"source": "doc_b", "summary": "Evidence B"},
    ]
    state = _make_state(observations=obs, confidence="medium", evidence_quality=0.4)
    r = verify_state(state)
    assert r.passed


# ── Integration: verifier runs after agent ────────────────────────────────────

def test_agent_state_has_verifier_report():
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    assert state.verifier_report is not None


def test_verifier_report_is_structured():
    state = run_agent("Compare DuckDB and SQLite for a local analytics app")
    vr = state.verifier_report
    assert isinstance(vr.passed, bool)
    assert isinstance(vr.issues, list)
    assert 0.0 <= vr.contradiction_risk <= 1.0
