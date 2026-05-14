"""Tests for the episode memory store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from task_agent.memory import Episode, EpisodeMemoryStore


# ── Episode ────────────────────────────────────────────────────────────────────

def test_episode_to_dict():
    ep = Episode(
        goal="test goal",
        status="TERMINATED_SUCCESS",
        confidence="medium",
        evidence_quality=0.4,
        steps_taken=5,
        observations_count=3,
        source_diversity=2,
        final_answer_preview="answer preview",
    )
    d = ep.to_dict()
    assert d["goal"] == "test goal"
    assert d["status"] == "TERMINATED_SUCCESS"
    assert d["steps_taken"] == 5


def test_episode_roundtrip():
    ep = Episode(
        goal="roundtrip goal",
        status="TERMINATED_SUCCESS",
        confidence="high",
        evidence_quality=0.7,
        steps_taken=3,
        observations_count=2,
        source_diversity=1,
        final_answer_preview="some answer",
    )
    ep2 = Episode.from_dict(ep.to_dict())
    assert ep2.goal == ep.goal
    assert ep2.status == ep.status
    assert ep2.evidence_quality == ep.evidence_quality


# ── EpisodeMemoryStore ─────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    return EpisodeMemoryStore(memory_dir=tmp_path)


def test_empty_store_returns_empty_list(tmp_store):
    assert tmp_store.load_recent() == []


def test_append_and_load(tmp_store):
    from task_agent import run_agent
    state = run_agent("Compare DuckDB and SQLite for analytics")
    path = tmp_store.append_episode(state)
    assert path.exists()
    loaded = tmp_store.load_recent(n=10)
    assert len(loaded) == 1
    assert loaded[0].goal == state.goal


def test_load_recent_ordered_newest_first(tmp_store):
    from task_agent import run_agent
    s1 = run_agent("Compare DuckDB and SQLite for analytics")
    s2 = run_agent("Compare LangGraph and AutoGen for a small agent project")
    tmp_store.append_episode(s1)
    tmp_store.append_episode(s2)
    loaded = tmp_store.load_recent(n=2)
    # Newest is s2 (appended last)
    assert loaded[0].goal == s2.goal


def test_search_by_goal_terms(tmp_store):
    from task_agent import run_agent
    state = run_agent("Compare DuckDB and SQLite for analytics")
    tmp_store.append_episode(state)
    results = tmp_store.search_by_goal_terms("DuckDB analytics")
    assert len(results) >= 1
    assert any("DuckDB" in r.goal for r in results)


def test_search_returns_empty_for_no_match(tmp_store):
    from task_agent import run_agent
    state = run_agent("Compare DuckDB and SQLite for analytics")
    tmp_store.append_episode(state)
    results = tmp_store.search_by_goal_terms("quantum entanglement xenon")
    assert results == []


def test_malformed_json_skipped(tmp_store):
    bad = tmp_store.memory_dir / "episode_00000.json"
    bad.write_text("{invalid json", encoding="utf-8")
    # Should not raise
    loaded = tmp_store.load_recent()
    assert isinstance(loaded, list)


def test_enable_memory_flag():
    """run_agent with enable_memory=True should complete without error."""
    from task_agent import run_agent
    state = run_agent(
        "Compare FastAPI and Flask for a beginner Python web app",
        enable_memory=True,
    )
    assert state.final_answer is not None
    assert state.step_count >= 1
