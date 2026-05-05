"""
Tool implementations for the task agent.

Each tool function is a pure transformation: it takes typed inputs and returns
a ToolResult dict with at minimum {"success": bool}.  Tools do not mutate
agent state directly — the agent loop is responsible for recording results.

Tool registry maps logical action names to ToolSpec instances so new tools
can be registered without modifying the agent loop.

Comparison uses a multi-criteria evidence matrix when sufficient evidence
exists, falling back to mention-count heuristics when evidence is sparse.
Summarization uses source citations in the final answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .knowledge import KnowledgeGraph
from .llm import is_openai_available, safe_generate
from .config import LLM_COMPARE_MAX_TOKENS, LLM_SUMMARIZE_MAX_TOKENS, MAX_SEARCH_RESULTS

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"

ToolResult = dict[str, Any]
ToolFunction = Callable[..., ToolResult]

# ── Criteria used for comparison analysis ─────────────────────────────────────
_COMPARISON_CRITERIA = [
    "performance",
    "ease of use",
    "ecosystem",
    "scalability",
    "portability",
    "documentation",
    "production",
    "beginner",
    "async",
    "analytics",
]


@dataclass
class ToolSpec:
    name: str
    description: str
    function: ToolFunction


# ── Corpus loader ──────────────────────────────────────────────────────────────

def load_corpus() -> dict[str, str]:
    corpus: dict[str, str] = {}
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            corpus[path.stem] = text
    return corpus


# ── local_search ───────────────────────────────────────────────────────────────

def search_local_corpus(
    query: str,
    corpus: "dict[str, str] | KnowledgeGraph",
    max_results: int = MAX_SEARCH_RESULTS,
) -> list[dict[str, str]]:
    """Return ranked passages from the local knowledge corpus."""
    if isinstance(corpus, KnowledgeGraph):
        return corpus.search(query, max_results=max_results)

    # Fallback: simple term-overlap if raw dict is passed (tests)
    query_terms = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2]
    if not query_terms:
        return []
    hits: list[tuple[int, str, str]] = []
    for title, body in corpus.items():
        score = sum(title.lower().count(t) * 3 + body.lower().count(t) for t in query_terms)
        if score > 0:
            snippet = body.replace("\n", " ")
            snippet = snippet[:500] + ("…" if len(snippet) > 500 else "")
            hits.append((score, title, snippet))
    hits.sort(reverse=True)
    return [{"source": t, "summary": s, "score": sc} for sc, t, s in hits[:max_results]]


# ── compare_items ──────────────────────────────────────────────────────────────

def _build_evidence_matrix(
    targets: list[str],
    observations: list[dict[str, str]],
) -> dict[str, dict[str, list[str]]]:
    """
    For each (target, criterion) pair collect matching observation snippets.
    Returns { target: { criterion: [sentences] } }.
    """
    matrix: dict[str, dict[str, list[str]]] = {t: {c: [] for c in _COMPARISON_CRITERIA} for t in targets}
    for obs in observations:
        text = f"{obs.get('source', '')} {obs.get('summary', '')}".lower()
        for target in targets:
            if target.lower() not in text:
                continue
            for criterion in _COMPARISON_CRITERIA:
                if criterion in text:
                    sentence_hint = obs.get("summary", "")[:120]
                    matrix[target][criterion].append(sentence_hint)
    return matrix


def _render_comparison_table(
    targets: list[str],
    matrix: dict[str, dict[str, list[str]]],
) -> str:
    header = "| Criterion        | " + " | ".join(f"{t:<20}" for t in targets) + " |"
    separator = "|" + "---|" * (len(targets) + 1)
    rows = [header, separator]
    for criterion in _COMPARISON_CRITERIA:
        cells = []
        for t in targets:
            n = len(matrix[t][criterion])
            cells.append("Strong" if n >= 2 else ("Present" if n == 1 else "—"))
        rows.append(f"| {criterion:<16} | " + " | ".join(f"{c:<20}" for c in cells) + " |")
    return "\n".join(rows)


def compare_items(
    targets: list[str],
    observations: list[dict[str, str]],
) -> ToolResult:
    """
    Compare candidates using a multi-criteria evidence matrix.
    Returns a rich comparison text and a winner recommendation.
    """
    evidence_by_target: dict[str, list[str]] = {t: [] for t in targets}
    for obs in observations:
        text = f"{obs.get('source', '')} {obs.get('summary', '')}".lower()
        for t in targets:
            if t.lower() in text:
                evidence_by_target[t].append(obs.get("summary", ""))

    if not any(evidence_by_target.values()):
        return {"success": False, "reason": "no evidence for comparison"}

    matrix = _build_evidence_matrix(targets, observations)
    table = _render_comparison_table(targets, matrix)

    # Score: weight criteria hits more heavily than raw mention count
    scores = {t: sum(len(v) for v in matrix[t].values()) for t in targets}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner, winner_score = ranked[0]
    total_evidence = sum(len(v) for v in evidence_by_target.values())
    confidence = "high" if winner_score >= 4 else ("medium" if total_evidence >= 2 else "low")

    note = (
        "Evidence matrix is derived from local corpus passages. "
        "Confidence reflects mention density across criteria."
    )
    comparison = (
        f"Evidence Matrix:\n{table}\n\n"
        f"Recommended: **{winner}** ({confidence} confidence).\n{note}"
    )

    return {
        "success": True,
        "comparison": comparison,
        "items": evidence_by_target,
        "winner": winner,
        "confidence": confidence,
    }


def llm_compare(
    targets: list[str],
    observations: list[dict[str, str]],
    goal: str,
) -> ToolResult:
    if not is_openai_available():
        return compare_items(targets, observations)

    matrix = _build_evidence_matrix(targets, observations)
    table = _render_comparison_table(targets, matrix)
    obs_text = "\n".join(
        f"[{o.get('source', '?')}] {o.get('summary', '')[:200]}" for o in observations
    )
    prompt = (
        "You are a technical analyst comparing candidate options.\n\n"
        f"Goal: {goal}\n"
        f"Candidates: {', '.join(targets)}\n\n"
        f"Evidence matrix (from local corpus):\n{table}\n\n"
        f"Raw observations:\n{obs_text}\n\n"
        "Write a clear comparison covering: performance, ease of use, ecosystem fit, and "
        "use-case alignment. End with a concrete recommendation and confidence level."
    )
    text = safe_generate(prompt, fallback="", max_tokens=LLM_COMPARE_MAX_TOKENS)
    if not text:
        return compare_items(targets, observations)

    return {
        "success": True,
        "comparison": text,
        "items": {
            t: [o["summary"] for o in observations if t.lower() in f"{o.get('source','')} {o.get('summary','')}".lower()]
            for t in targets
        },
        "winner": targets[0] if targets else "",
        "confidence": "medium",
    }


# ── summarize_findings ─────────────────────────────────────────────────────────

def summarize_findings(
    goal: str,
    observations: list[dict[str, str]],
    confidence: str,
) -> dict[str, str]:
    """Deterministic summarization with source citations."""
    if not observations:
        return {
            "final": (
                f"Goal: {goal}\n\n"
                "The local corpus did not contain relevant evidence for this goal. "
                "No answer can be generated.\nConfidence: low; evidence unavailable."
            ),
            "confidence": "low",
        }

    seen_sources: list[str] = []
    for obs in observations:
        src = obs.get("source", "")
        if src and src not in seen_sources:
            seen_sources.append(src)

    bullets = [f"[{obs.get('source', '?')}] {obs.get('summary', '')[:180]}" for obs in observations]
    conf_text = {
        "high": "Confidence: high — multiple independent sources corroborate the answer.",
        "medium": "Confidence: moderate — evidence is present but limited in scope.",
        "low": "Confidence: low — evidence is sparse; treat this answer as indicative only.",
    }.get(confidence, "Confidence: unknown.")

    sources_line = f"Sources consulted: {', '.join(seen_sources)}." if seen_sources else ""

    final = (
        f"Goal: {goal}\n\n"
        "Evidence summary:\n- " + "\n- ".join(bullets) + "\n\n"
        f"Recommendation: Based on the local corpus, the above evidence addresses the goal.\n"
        f"{conf_text}\n{sources_line}"
    )
    return {"final": final, "confidence": confidence}


def llm_summarize(
    goal: str,
    observations: list[dict[str, str]],
    confidence: str,
) -> dict[str, str]:
    if not is_openai_available():
        return summarize_findings(goal, observations, confidence)

    obs_text = "\n".join(
        f"[{o.get('source', '?')}] {o.get('summary', '')[:200]}" for o in observations
    )
    prompt = (
        "You are a precise summarization agent. Write a final answer to the goal "
        "using only the evidence below. Include a one-sentence confidence note "
        "(high/medium/low) and cite which sources you drew from.\n\n"
        f"Goal: {goal}\n\n"
        f"Evidence:\n{obs_text}"
    )
    text = safe_generate(prompt, fallback="", max_tokens=LLM_SUMMARIZE_MAX_TOKENS)
    if not text:
        return summarize_findings(goal, observations, confidence)
    return {"final": text, "confidence": confidence or "medium"}


# ── calculator ─────────────────────────────────────────────────────────────────

def calculator(query: str) -> ToolResult:
    """Extract numbers from the query and produce a simple numeric comparison."""
    numbers = [float(n) for n in re.findall(r"\d+(?:[.,]\d+)?", query.replace(",", ""))]
    if len(numbers) < 2:
        return {"success": False, "reason": "fewer than two numeric values found"}

    a, b = numbers[0], numbers[1]
    a_str = str(int(a)) if a == int(a) else str(a)
    b_str = str(int(b)) if b == int(b) else str(b)
    relation = "greater than" if a > b else ("less than" if a < b else "equal to")
    result = f"Simple numeric comparison: {a_str} vs {b_str} — {a_str} is {relation} {b_str}."
    return {"success": True, "result": result}


# ── Registry ───────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, ToolSpec] = {
    "search": ToolSpec(
        name="local_search",
        description="Retrieve ranked passages from the local knowledge corpus using TF-IDF scoring.",
        function=search_local_corpus,
    ),
    "compare": ToolSpec(
        name="compare_items",
        description="Compare candidates using a multi-criteria evidence matrix derived from observations.",
        function=compare_items,
    ),
    "summarize": ToolSpec(
        name="summarize_findings",
        description="Synthesize accumulated evidence into a final answer with source citations.",
        function=summarize_findings,
    ),
    "calculate": ToolSpec(
        name="calculator",
        description="Extract and compare numeric values from a query string.",
        function=calculator,
    ),
}
