"""Tests for the TF-IDF KnowledgeGraph retrieval layer."""
from __future__ import annotations

import pytest

from task_agent.knowledge import KnowledgeGraph, _tokenize
from task_agent.tools import load_corpus


# ── Tokeniser ──────────────────────────────────────────────────────────────────

def test_tokenize_removes_stop_words():
    tokens = _tokenize("the quick and the brown fox")
    assert "the" not in tokens
    assert "and" not in tokens
    assert "quick" in tokens or "brown" in tokens or "fox" in tokens


def test_tokenize_lowercases():
    tokens = _tokenize("DuckDB SQLite Analytics")
    assert all(t == t.lower() for t in tokens)


def test_tokenize_filters_short_tokens():
    tokens = _tokenize("a ab abc abcd")
    assert "a" not in tokens
    assert "ab" not in tokens
    # "abc" has length 3 — depends on filter (> 2), so should be included
    assert "abcd" in tokens


# ── KnowledgeGraph construction ────────────────────────────────────────────────

def test_knowledge_graph_builds_and_searches():
    docs = load_corpus()
    graph = KnowledgeGraph.from_documents(docs)
    assert "duckdb_vs_sqlite" in graph.document_keys()
    results = graph.search("DuckDB analytics")
    assert isinstance(results, list)
    assert results and results[0]["source"] == "duckdb_vs_sqlite"


def test_knowledge_graph_related_documents():
    docs = load_corpus()
    graph = KnowledgeGraph.from_documents(docs)
    neighbors = graph.related("duckdb_vs_sqlite")
    assert isinstance(neighbors, list)
    assert all(isinstance(key, str) for key in neighbors)


def test_knowledge_graph_all_docs_indexed():
    docs = load_corpus()
    graph = KnowledgeGraph.from_documents(docs)
    assert len(graph.document_keys()) == len(docs)


def test_search_result_structure():
    docs = load_corpus()
    graph = KnowledgeGraph.from_documents(docs)
    results = graph.search("DuckDB SQLite")
    assert len(results) > 0
    for r in results:
        assert "source" in r
        assert "summary" in r
        assert "score" in r
        assert isinstance(r["score"], float)


def test_search_returns_non_empty_snippets():
    docs = load_corpus()
    graph = KnowledgeGraph.from_documents(docs)
    results = graph.search("analytics database")
    for r in results:
        assert len(r["summary"]) > 20


def test_search_returns_most_relevant_first():
    docs = load_corpus()
    graph = KnowledgeGraph.from_documents(docs)
    results = graph.search("DuckDB vectorized columnar analytics")
    assert results[0]["source"] == "duckdb_vs_sqlite"


def test_search_ros2_topic():
    docs = load_corpus()
    graph = KnowledgeGraph.from_documents(docs)
    results = graph.search("ROS2 nodes topics publisher subscriber")
    assert len(results) > 0
    assert results[0]["source"] == "ros2_basics"


def test_empty_query_returns_empty():
    docs = load_corpus()
    graph = KnowledgeGraph.from_documents(docs)
    results = graph.search("")
    assert results == []


def test_unknown_topic_low_or_no_results():
    docs = load_corpus()
    graph = KnowledgeGraph.from_documents(docs)
    results = graph.search("blockchain crypto NFT defi")
    assert len(results) == 0 or all(r["score"] < 0.01 for r in results)


# ── TF-IDF scoring properties ──────────────────────────────────────────────────

def test_tfidf_scores_are_positive():
    docs = load_corpus()
    graph = KnowledgeGraph.from_documents(docs)
    results = graph.search("DuckDB analytics performance")
    for r in results:
        assert r["score"] > 0


def test_idf_populated():
    docs = load_corpus()
    graph = KnowledgeGraph.from_documents(docs)
    assert len(graph._idf) > 0
    for term, idf_val in graph._idf.items():
        assert idf_val > 0
