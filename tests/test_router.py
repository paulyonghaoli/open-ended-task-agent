"""Tests for the cognitive substrate router."""
from __future__ import annotations

import pytest

from task_agent.router import CognitiveMode, RoutingDecision, route_reasoning_mode


def test_count_goal_routes_to_symbolic():
    d = route_reasoning_mode("Count the number of r in strawberry")
    assert d.selected_mode == CognitiveMode.SYMBOLIC


def test_arithmetic_routes_to_symbolic():
    d = route_reasoning_mode("Calculate 5 * 12 + 3")
    assert d.selected_mode == CognitiveMode.SYMBOLIC


def test_how_many_routes_to_symbolic():
    d = route_reasoning_mode("How many combinations are there?")
    assert d.selected_mode == CognitiveMode.SYMBOLIC


def test_geometry_routes_to_geometric():
    d = route_reasoning_mode("Find the area of a circle with radius 5")
    assert d.selected_mode == CognitiveMode.GEOMETRIC


def test_debug_routes_to_execution():
    d = route_reasoning_mode("Debug this Python function")
    assert d.selected_mode == CognitiveMode.EXECUTION


def test_compare_routes_to_retrieval():
    d = route_reasoning_mode("Compare DuckDB and SQLite for analytics")
    assert d.selected_mode == CognitiveMode.RETRIEVAL


def test_explain_routes_to_retrieval():
    d = route_reasoning_mode("Explain the differences between FastAPI and Flask")
    assert d.selected_mode == CognitiveMode.RETRIEVAL


def test_graph_routes_to_graph():
    d = route_reasoning_mode("Find the shortest path between two nodes in a graph")
    assert d.selected_mode == CognitiveMode.GRAPH


def test_default_routes_to_language():
    d = route_reasoning_mode("Tell me something interesting")
    assert d.selected_mode == CognitiveMode.LANGUAGE


def test_routing_decision_has_confidence():
    d = route_reasoning_mode("Compare LangGraph and AutoGen")
    assert 0.0 < d.confidence <= 1.0


def test_routing_decision_has_rationale():
    d = route_reasoning_mode("Compare LangGraph and AutoGen")
    assert isinstance(d.rationale, str)
    assert len(d.rationale) > 5


def test_to_dict_contains_required_keys():
    d = route_reasoning_mode("Count letters in word")
    result = d.to_dict()
    assert "selected_mode" in result
    assert "confidence" in result
    assert "rationale" in result
    assert isinstance(result["selected_mode"], str)
