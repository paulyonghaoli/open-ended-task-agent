"""Tests for tool implementations — search, compare, summarize, calculator."""
from __future__ import annotations

import pytest

from task_agent.tools import (
    calculator,
    compare_items,
    load_corpus,
    search_local_corpus,
    summarize_findings,
    _build_evidence_matrix,
    _render_comparison_table,
)


# ── load_corpus ────────────────────────────────────────────────────────────────

def test_load_corpus_returns_docs():
    corpus = load_corpus()
    assert "duckdb_vs_sqlite" in corpus
    assert "fastapi_vs_flask" in corpus
    assert "langgraph_vs_autogen" in corpus
    assert "ros2_basics" in corpus
    assert "python_web_frameworks" in corpus


def test_load_corpus_docs_are_non_empty():
    corpus = load_corpus()
    for key, text in corpus.items():
        assert len(text) > 100, f"Doc '{key}' is suspiciously short"


# ── search_local_corpus ────────────────────────────────────────────────────────

def test_search_local_corpus_matches_expected_doc():
    corpus = load_corpus()
    results = search_local_corpus("Compare DuckDB and SQLite", corpus)
    assert any(item["source"] == "duckdb_vs_sqlite" for item in results)


def test_search_returns_source_and_summary():
    corpus = load_corpus()
    results = search_local_corpus("DuckDB analytics", corpus)
    assert len(results) > 0
    assert "source" in results[0]
    assert "summary" in results[0]


def test_search_empty_query_returns_empty():
    corpus = load_corpus()
    results = search_local_corpus("", corpus)
    assert results == []


def test_search_unknown_topic_returns_empty_or_low_results():
    corpus = load_corpus()
    results = search_local_corpus("blockchain NFT metaverse", corpus)
    # Either no results or very low-scoring results
    assert len(results) == 0 or all(r.get("score", 999) < 1 for r in results)


# ── compare_items ──────────────────────────────────────────────────────────────

def _sample_observations():
    return [
        {"source": "duckdb_vs_sqlite", "summary": "DuckDB is analytic and SQLite is portable."},
        {"source": "duckdb_vs_sqlite", "summary": "SQLite is lighter for file-based workflows."},
        {"source": "duckdb_vs_sqlite", "summary": "DuckDB performance is superior for analytics."},
    ]


def test_compare_items_returns_success_with_observations():
    result = compare_items(["DuckDB", "SQLite"], _sample_observations())
    assert result["success"] is True
    assert "DuckDB" in result["comparison"]


def test_compare_items_returns_winner():
    result = compare_items(["DuckDB", "SQLite"], _sample_observations())
    assert "winner" in result
    assert result["winner"] in ("DuckDB", "SQLite")


def test_compare_items_returns_confidence():
    result = compare_items(["DuckDB", "SQLite"], _sample_observations())
    assert result["confidence"] in ("high", "medium", "low")


def test_compare_items_fails_without_observations():
    result = compare_items(["DuckDB", "SQLite"], [])
    assert result["success"] is False


def test_compare_items_fails_when_no_evidence_matches():
    observations = [{"source": "unrelated", "summary": "Totally unrelated content."}]
    result = compare_items(["DuckDB", "SQLite"], observations)
    assert result["success"] is False


def test_evidence_matrix_built_correctly():
    matrix = _build_evidence_matrix(
        ["DuckDB", "SQLite"],
        [{"source": "doc", "summary": "DuckDB performance is great for analytics."}],
    )
    assert "DuckDB" in matrix
    assert "SQLite" in matrix
    assert any(matrix["DuckDB"].values())


def test_comparison_table_contains_candidates():
    matrix = _build_evidence_matrix(
        ["FastAPI", "Flask"],
        [{"source": "doc", "summary": "FastAPI async and Flask synchronous."}],
    )
    table = _render_comparison_table(["FastAPI", "Flask"], matrix)
    assert "FastAPI" in table
    assert "Flask" in table


# ── summarize_findings ─────────────────────────────────────────────────────────

def test_summarize_findings_includes_goal_and_confidence():
    summary = summarize_findings(
        "Compare DuckDB and SQLite",
        [{"source": "duckdb_vs_sqlite", "summary": "One is analytic."}],
        "medium",
    )
    assert "Compare DuckDB and SQLite" in summary["final"]
    assert "Confidence: moderate" in summary["final"]


def test_summarize_findings_high_confidence():
    summary = summarize_findings("Test", [{"source": "s", "summary": "evidence"}], "high")
    assert "Confidence: high" in summary["final"]


def test_summarize_findings_no_observations():
    summary = summarize_findings("Test goal", [], "low")
    assert summary["confidence"] == "low"
    assert "Test goal" in summary["final"]


def test_summarize_findings_cites_sources():
    observations = [
        {"source": "duckdb_vs_sqlite", "summary": "DuckDB is fast."},
        {"source": "fastapi_vs_flask", "summary": "FastAPI is typed."},
    ]
    summary = summarize_findings("Multi-source test", observations, "high")
    assert "duckdb_vs_sqlite" in summary["final"] or "fastapi_vs_flask" in summary["final"]


# ── calculator ─────────────────────────────────────────────────────────────────

def test_calculator_handles_numbers():
    result = calculator("Which is larger: 5 or 10?")
    assert result["success"] is True
    assert "5 vs 10" in result["result"]


def test_calculator_identifies_relation():
    result = calculator("Compare 100 and 50")
    assert result["success"] is True
    assert "greater" in result["result"]


def test_calculator_fails_with_one_number():
    result = calculator("I have 42 apples")
    assert result["success"] is False


def test_calculator_fails_with_no_numbers():
    result = calculator("Which framework is better?")
    assert result["success"] is False
