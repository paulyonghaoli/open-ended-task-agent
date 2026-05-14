"""Tests for the TaskLedger."""
from __future__ import annotations

import pytest

from task_agent.ledger import TaskLedger


def test_ledger_initialises_with_goal():
    l = TaskLedger(goal="Compare A and B")
    assert l.goal == "Compare A and B"
    assert l.known_facts == []
    assert l.open_questions == []
    assert l.stop_reason is None


def test_record_fact_adds_entry():
    l = TaskLedger(goal="test")
    l.record_fact("DuckDB is columnar")
    assert "DuckDB is columnar" in l.known_facts


def test_record_fact_no_duplicates():
    l = TaskLedger(goal="test")
    l.record_fact("same fact")
    l.record_fact("same fact")
    assert l.known_facts.count("same fact") == 1


def test_record_failure_adds_entry():
    l = TaskLedger(goal="test")
    l.record_failure("Search for X returned nothing")
    assert len(l.failed_attempts) == 1


def test_add_question_adds_entry():
    l = TaskLedger(goal="test")
    l.add_question("What is the performance of DuckDB?")
    assert len(l.open_questions) == 1


def test_add_question_no_duplicates():
    l = TaskLedger(goal="test")
    l.add_question("Q?")
    l.add_question("Q?")
    assert l.open_questions.count("Q?") == 1


def test_resolve_question_removes_entry():
    l = TaskLedger(goal="test")
    l.add_question("Q?")
    l.resolve_question("Q?")
    assert "Q?" not in l.open_questions


def test_close_sets_stop_reason():
    l = TaskLedger(goal="test")
    l.close("TERMINATED_SUCCESS")
    assert l.stop_reason == "TERMINATED_SUCCESS"


def test_to_dict_has_all_keys():
    l = TaskLedger(goal="test", subgoals=["step 1", "step 2"])
    d = l.to_dict()
    for key in ("goal", "subgoals", "assumptions", "known_facts",
                "open_questions", "failed_attempts", "next_best_action", "stop_reason"):
        assert key in d


def test_agent_attaches_ledger():
    from task_agent import run_agent
    state = run_agent("Compare DuckDB and SQLite for analytics")
    assert state.ledger is not None
    assert state.ledger.goal == "Compare DuckDB and SQLite for analytics"


def test_ledger_records_facts_during_search():
    from task_agent import run_agent
    state = run_agent("Compare DuckDB and SQLite for analytics")
    # Ledger should have at least one known fact from successful searches
    assert len(state.ledger.known_facts) >= 1


def test_ledger_has_stop_reason_after_run():
    from task_agent import run_agent
    state = run_agent("Compare DuckDB and SQLite for analytics")
    assert state.ledger.stop_reason is not None
    assert "TERMINATED_" in state.ledger.stop_reason
