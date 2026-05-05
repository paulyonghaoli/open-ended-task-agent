"""Tests for the Planner module — LLM plan parsing and deterministic fallbacks."""
from __future__ import annotations

import pytest

from task_agent.planner import Planner, _create_fallback_plan, _parse_llm_plan


# ── LLM plan parsing ───────────────────────────────────────────────────────────

def test_parse_llm_plan_identifies_search_compare_summarize():
    plan_text = (
        '1. Search for "DuckDB and SQLite" in the local corpus.\n'
        '2. Compare DuckDB and SQLite using the evidence.\n'
        '3. Summarize the findings with a recommendation.\n'
    )
    steps = _parse_llm_plan("Compare DuckDB and SQLite", plan_text)
    assert [s.action_type for s in steps] == ["search", "compare", "summarize"]
    assert steps[0].target == "DuckDB and SQLite"


def test_parse_llm_plan_extracts_targets_from_quotes():
    plan_text = '1. Search for "FastAPI vs Flask comparison"\n2. Summarize "FastAPI vs Flask"\n'
    steps = _parse_llm_plan("Compare FastAPI and Flask", plan_text)
    assert steps[0].target == "FastAPI vs Flask comparison"


def test_parse_llm_plan_falls_back_to_goal_when_no_quotes():
    plan_text = "1. Search for relevant documents\n2. Summarize the findings\n"
    goal = "My test goal"
    steps = _parse_llm_plan(goal, plan_text)
    assert all(s.target == goal for s in steps)


def test_parse_llm_plan_skips_non_action_lines():
    plan_text = "Here is my plan:\n1. Think about the goal\n2. Search for evidence\n3. Summarize\n"
    steps = _parse_llm_plan("goal", plan_text)
    action_types = {s.action_type for s in steps}
    assert "search" in action_types
    assert "summarize" in action_types


# ── Deterministic fallback ─────────────────────────────────────────────────────

def test_generate_plan_uses_fallback_when_no_openai(monkeypatch):
    monkeypatch.setattr("task_agent.planner.is_openai_available", lambda: False)
    planner = Planner()
    plan = planner.generate_plan(
        "Compare DuckDB and SQLite for analytics",
        ["duckdb_vs_sqlite", "fastapi_vs_flask"],
    )
    assert any(step.action_type == "search" for step in plan.steps)
    assert any(step.action_type == "compare" for step in plan.steps)
    assert plan.notes is not None
    assert "Deterministic" in plan.notes


def test_fallback_plan_compare_goal():
    plan = _create_fallback_plan(
        "Compare DuckDB and SQLite for analytics",
        ["duckdb_vs_sqlite"],
    )
    action_types = [s.action_type for s in plan.steps]
    assert "search" in action_types
    assert "compare" in action_types
    assert "summarize" in action_types


def test_fallback_plan_study_goal():
    plan = _create_fallback_plan(
        "Create a study plan for learning ROS2",
        ["ros2_basics"],
    )
    action_types = [s.action_type for s in plan.steps]
    assert "search" in action_types
    assert "summarize" in action_types
    assert "compare" not in action_types


def test_fallback_plan_always_ends_with_summarize():
    for goal in [
        "Compare DuckDB and SQLite",
        "Learn ROS2 basics",
        "What is the best Python web framework?",
    ]:
        plan = _create_fallback_plan(goal, [])
        assert plan.steps[-1].action_type == "summarize"


def test_fallback_plan_always_starts_with_search():
    for goal in [
        "Compare DuckDB and SQLite",
        "Learn ROS2 basics",
        "Generic question",
    ]:
        plan = _create_fallback_plan(goal, [])
        assert plan.steps[0].action_type == "search"


def test_fallback_plan_has_stopping_criteria():
    plan = _create_fallback_plan("Compare A and B", [])
    assert isinstance(plan.stopping_criteria, str)
    assert len(plan.stopping_criteria) > 10


def test_plan_steps_have_rationale():
    plan = _create_fallback_plan("Compare DuckDB and SQLite", [])
    for step in plan.steps:
        assert isinstance(step.rationale, str)
