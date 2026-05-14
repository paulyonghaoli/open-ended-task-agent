from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class CognitiveMode(Enum):
    """Broad classes of reasoning substrate required by a goal."""
    LANGUAGE = "language"
    SYMBOLIC = "symbolic"
    RETRIEVAL = "retrieval"
    GRAPH = "graph"
    EXECUTION = "execution"
    GEOMETRIC = "geometric"


@dataclass
class RoutingDecision:
    """Result of routing a goal to a cognitive mode."""
    selected_mode: CognitiveMode
    confidence: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "selected_mode": self.selected_mode.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


# Ordered list of (mode, pattern, rationale) — first match wins.
_ROUTES: List[Tuple[CognitiveMode, re.Pattern, str]] = [
    (
        CognitiveMode.SYMBOLIC,
        re.compile(
            r"\b(count|how many|number of|occurrences? of|"
            r"factorial|fibonacci|combinations?|permutations?)\b",
            re.I,
        ),
        "counting or combinatorics task detected",
    ),
    (
        CognitiveMode.SYMBOLIC,
        re.compile(
            r"\b(calculate|compute|evaluate|arithmetic|"
            r"sum|total|add|subtract|multiply|divide)\b",
            re.I,
        ),
        "arithmetic computation task detected",
    ),
    (
        CognitiveMode.GEOMETRIC,
        re.compile(
            r"\b(angle|triangle|circle|rectangle|polygon|geometry|"
            r"area|perimeter|radius|hypotenuse|coordinate|vector)\b",
            re.I,
        ),
        "geometric reasoning task detected",
    ),
    (
        CognitiveMode.EXECUTION,
        re.compile(
            r"\b(debug|implement|write code|script|function|"
            r"algorithm|program|execute|refactor|bug)\b",
            re.I,
        ),
        "code generation or debugging task detected",
    ),
    (
        CognitiveMode.GRAPH,
        re.compile(
            r"\b(graph|network|path|traversal|shortest path|"
            r"connected|node|edge|relationship|topology)\b",
            re.I,
        ),
        "graph reasoning task detected",
    ),
    (
        CognitiveMode.RETRIEVAL,
        re.compile(
            r"\b(compare|research|find|explain|overview|study plan|"
            r"learn|what is|how does|survey|review|describe)\b",
            re.I,
        ),
        "research or comparison task detected",
    ),
]


def route_reasoning_mode(goal: str) -> RoutingDecision:
    """
    Map a natural-language goal to a CognitiveMode using lightweight heuristics.
    Returns a RoutingDecision with the selected mode, confidence, and rationale.
    """
    for mode, pattern, rationale in _ROUTES:
        if pattern.search(goal):
            return RoutingDecision(
                selected_mode=mode,
                confidence=0.80,
                rationale=rationale,
            )
    return RoutingDecision(
        selected_mode=CognitiveMode.LANGUAGE,
        confidence=0.55,
        rationale="no specialized pattern matched; defaulting to language reasoning",
    )
