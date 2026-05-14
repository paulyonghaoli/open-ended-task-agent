"""Tests for the export_state function."""
from __future__ import annotations

import json

import pytest

from task_agent import run_agent
from task_agent.export import export_state


@pytest.fixture(scope="module")
def state():
    return run_agent("Compare DuckDB and SQLite for a local analytics app")


def test_export_returns_dict(state):
    d = export_state(state)
    assert isinstance(d, dict)


def test_export_contains_required_keys(state):
    d = export_state(state)
    for key in ("goal", "status", "final_answer", "confidence",
                 "evidence_quality", "stats", "observations"):
        assert key in d, f"Missing key: {key}"


def test_export_stats_contains_steps(state):
    d = export_state(state)
    assert "steps_taken" in d["stats"]
    assert d["stats"]["steps_taken"] >= 1


def test_export_routing_present(state):
    d = export_state(state)
    assert "routing" in d
    assert "selected_mode" in d["routing"]
    assert "confidence" in d["routing"]


def test_export_verifier_present(state):
    d = export_state(state)
    assert "verifier" in d
    assert "passed" in d["verifier"]


def test_export_ledger_present(state):
    d = export_state(state)
    assert "ledger" in d
    assert "goal" in d["ledger"]


def test_export_temporal_present(state):
    d = export_state(state)
    assert "temporal" in d
    assert "task_age_s" in d["temporal"]


def test_export_reflections_present(state):
    d = export_state(state)
    assert "reflections" in d
    assert isinstance(d["reflections"], list)


def test_export_trace_present(state):
    d = export_state(state)
    assert "trace" in d
    assert isinstance(d["trace"], list)
    assert len(d["trace"]) > 0


def test_export_is_json_serialisable(state):
    d = export_state(state)
    serialised = json.dumps(d, default=str)
    parsed = json.loads(serialised)
    assert parsed["goal"] == state.goal


def test_export_without_trace(state):
    d = export_state(state, include_trace=False)
    assert "trace" not in d


def test_export_without_reflections(state):
    d = export_state(state, include_reflections=False)
    assert "reflections" not in d


def test_export_evidence_quality_bounded(state):
    d = export_state(state)
    eq = d["evidence_quality"]
    assert 0.0 <= eq <= 1.0
