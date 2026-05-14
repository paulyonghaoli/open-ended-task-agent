"""
Goal decomposition and plan generation.

Two strategies are supported:
  LLM-guided  — uses chain-of-thought prompting to reason about the goal,
                 identify candidate items, and produce a numbered action plan.
  Deterministic — rule-based pattern matching for comparison goals, study-plan
                  goals, and generic open-ended questions.  Runs with zero
                  external dependencies as a safe fallback.

The LLM strategy follows a ReAct-style framing: the prompt asks the model to
(1) think about what information is needed, then (2) produce a concrete plan
using only the registered tool names.
"""
from __future__ import annotations

import re
from typing import Iterable

from .config import LLM_MODEL, LLM_PLAN_MAX_TOKENS
from .llm import is_openai_available, safe_generate
from .state import AgentPlan, PlanStep

TOOL_ACTIONS = ["search", "compare", "summarize", "calculate", "symbolic"]

_PLAN_SYSTEM = (
    "You are a task-planning agent that decomposes user goals into a short, "
    "ordered action plan. You have access to exactly five tools: "
    "search (retrieves local documents), compare (compares candidates using "
    "gathered evidence), summarize (synthesizes evidence into a final answer), "
    "calculate (numeric comparisons), symbolic (deterministic character counting "
    "or arithmetic — use for 'count X in Y' or math expression goals). "
    "Think step by step before writing the plan."
)

_PLAN_PROMPT = """\
Goal: {goal}
Available local documents: {titles}

Step 1 — Think: What information is needed to answer this goal?
Step 2 — Plan: Write a numbered action plan using ONLY the tools: search, compare, summarize, calculate, symbolic.
         Format each line as:  N. <action> "<target or query>"
         Include 2–4 steps. End with a summarize step.

Rules:
- Always start with at least one search step.
- For comparison goals (X vs Y), include: search, compare, summarize.
- For study/learning goals, include: search, summarize.
- For numeric goals, include: search, calculate, summarize.
- If the goal is unrelated to the local documents, output only:
  1. search "<goal>"
  2. summarize "<goal>"
"""


# ── LLM plan parsing ───────────────────────────────────────────────────────────

def _parse_llm_plan(goal: str, plan_text: str) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for raw in plan_text.splitlines():
        raw = raw.strip()
        line = re.sub(r"^\d+[\.)]\s*", "", raw)
        action = next(
            (a for a in TOOL_ACTIONS if re.search(rf"\b{re.escape(a)}\b", line, re.I)),
            None,
        )
        if not action:
            continue
        target_match = re.search(r'["“‘](.+?)["”’]', line)
        target = target_match.group(1).strip() if target_match else goal
        steps.append(
            PlanStep(
                id=len(steps) + 1,
                action_type=action,
                description=line,
                target=target,
                expected_evidence=_expected_evidence(action),
                rationale="LLM-generated step",
            )
        )
    return steps


def _expected_evidence(action: str) -> str:
    return {
        "search": "relevant passages from the local knowledge corpus",
        "compare": "structured multi-criteria comparison of candidates",
        "summarize": "final answer with confidence note and source citations",
        "calculate": "numeric result supporting the comparison",
        "symbolic": "exact deterministic result (character count or arithmetic value)",
    }.get(action, "supporting evidence")


# ── Deterministic fallback ─────────────────────────────────────────────────────

_SYMBOLIC_RE = re.compile(
    r"(?:"
    r"count\s+(?:the\s+)?(?:number\s+of\s+)?(\w+)(?:\s+in\s+(\w+))?"
    r"|how\s+many\s+\w+\s+(?:are\s+(?:in|there)|in)\s+\w+"
    r"|(?:what\s+is\s+)?(?:\d+[\+\-\*/\^]\d+)"
    r")",
    re.I,
)

_COMPARE_RE = re.compile(
    r"compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)(?:\s+for\s+.+)?(?:[.?!]|$)",
    re.I,
)
_STUDY_RE = re.compile(
    r"study\s+plan|learn(?:ing)?|basics|fundamentals|course|tutorial|how\s+to\s+(?:start|learn)",
    re.I,
)
_NUMERIC_RE = re.compile(r"\d[\d,.]*\s*(?:vs\.?|versus|compared)", re.I)


def _parse_compare_goal(goal: str) -> tuple[list[str], bool]:
    m = _COMPARE_RE.search(goal)
    if m:
        return [m.group(1).strip(), m.group(2).strip()], True
    # Also detect "X or Y" and "X and Y" when "compare" appears nearby
    if re.search(r"\bcompare\b", goal, re.I):
        parts = re.split(r"\s+(?:and|or|vs\.?|versus)\s+", goal, flags=re.I)
        if len(parts) == 2:
            left = re.sub(r"compare\s+", "", parts[0], flags=re.I).strip()
            right = parts[1].strip().rstrip("?.")
            if left and right:
                return [left, right], True
    return [], False


def _create_fallback_plan(goal: str, corpus_keys: Iterable[str]) -> AgentPlan:
    corpus_list = list(corpus_keys)

    # Symbolic tasks (counting, arithmetic) — no corpus needed
    if _SYMBOLIC_RE.search(goal):
        return AgentPlan(
            steps=[
                PlanStep(
                    id=1,
                    action_type="symbolic",
                    description="Execute deterministic symbolic computation",
                    target=goal,
                    expected_evidence="exact numeric or character-count result",
                    rationale="Goal requires deterministic computation, not retrieval",
                ),
                PlanStep(
                    id=2,
                    action_type="summarize",
                    description="Wrap symbolic result in a final answer",
                    target=goal,
                    expected_evidence="concise answer with computed value",
                    rationale="Deliver the exact result to the user",
                ),
            ],
            stopping_criteria="Compute the exact symbolic answer and deliver it.",
            notes="Deterministic symbolic-computation plan.",
        )

    items, is_compare = _parse_compare_goal(goal)

    if is_compare and len(items) == 2:
        a, b = items
        return AgentPlan(
            steps=[
                PlanStep(
                    id=1,
                    action_type="search",
                    description=f"Search for evidence on {a}",
                    target=a,
                    expected_evidence=f"characteristics, trade-offs, and use cases of {a}",
                    rationale="Gather evidence for first candidate before comparing",
                ),
                PlanStep(
                    id=2,
                    action_type="search",
                    description=f"Search for evidence on {b}",
                    target=b,
                    expected_evidence=f"characteristics, trade-offs, and use cases of {b}",
                    rationale="Gather evidence for second candidate before comparing",
                ),
                PlanStep(
                    id=3,
                    action_type="compare",
                    description=f"Compare {a} and {b} based on accumulated evidence",
                    target=f"{a},{b}",
                    expected_evidence="structured comparison with winner and confidence",
                    rationale="Synthesize evidence into a side-by-side comparison",
                ),
                PlanStep(
                    id=4,
                    action_type="summarize",
                    description="Produce final recommendation with confidence and citations",
                    target=goal,
                    expected_evidence="final answer with limitation note",
                    rationale="Deliver the answer and explain confidence level",
                ),
            ],
            stopping_criteria=(
                f"Collect evidence on both {a} and {b}, compare them, "
                "and deliver a grounded recommendation."
            ),
            notes="Deterministic two-candidate comparison plan.",
        )

    if _STUDY_RE.search(goal):
        topic = next(
            (k for k in corpus_list if any(w in goal.lower() for w in k.split("_"))),
            "learning path",
        )
        return AgentPlan(
            steps=[
                PlanStep(
                    id=1,
                    action_type="search",
                    description=f"Search for foundational content on {topic}",
                    target=topic,
                    expected_evidence="background concepts and structured learning notes",
                    rationale="Retrieve course-relevant documents before composing a plan",
                ),
                PlanStep(
                    id=2,
                    action_type="summarize",
                    description="Compose a structured study plan from retrieved content",
                    target=goal,
                    expected_evidence="week-by-week or phase-based study plan",
                    rationale="Synthesize evidence into actionable learning steps",
                ),
            ],
            stopping_criteria="Retrieve enough background content to produce a practical study plan.",
            notes="Deterministic study-plan fallback.",
        )

    if _NUMERIC_RE.search(goal):
        return AgentPlan(
            steps=[
                PlanStep(id=1, action_type="search", description="Retrieve numeric data", target=goal,
                         expected_evidence="numeric values for comparison", rationale=""),
                PlanStep(id=2, action_type="calculate", description="Compare numeric values", target=goal,
                         expected_evidence="quantitative comparison result", rationale=""),
                PlanStep(id=3, action_type="summarize", description="Summarize with numeric context", target=goal,
                         expected_evidence="final numeric recommendation", rationale=""),
            ],
            stopping_criteria="Extract and compare numeric values, then summarize.",
            notes="Deterministic numeric-comparison fallback.",
        )

    # Generic fallback
    return AgentPlan(
        steps=[
            PlanStep(
                id=1,
                action_type="search",
                description="Search the local corpus for relevant evidence",
                target=goal,
                expected_evidence="relevant passages from the knowledge base",
                rationale="Open-ended goal — start with broad retrieval",
            ),
            PlanStep(
                id=2,
                action_type="summarize",
                description="Synthesize the best available evidence into an answer",
                target=goal,
                expected_evidence="concise answer with a limitation statement",
                rationale="Deliver the best available answer given corpus coverage",
            ),
        ],
        stopping_criteria="Locate relevant evidence and produce a grounded answer.",
        notes="Deterministic generic fallback.",
    )


# ── Public API ─────────────────────────────────────────────────────────────────

class Planner:
    def __init__(self) -> None:
        self.use_llm = is_openai_available()

    def generate_plan(self, goal: str, corpus_keys: list[str]) -> AgentPlan:
        if self.use_llm:
            prompt = _PLAN_PROMPT.format(
                goal=goal,
                titles=", ".join(corpus_keys) or "(empty corpus)",
            )
            plan_text = safe_generate(
                prompt,
                fallback="",
                system=_PLAN_SYSTEM,
                model=LLM_MODEL,
                max_tokens=LLM_PLAN_MAX_TOKENS,
            )
            steps = _parse_llm_plan(goal, plan_text) if plan_text else []
            if len(steps) >= 2:
                return AgentPlan(
                    steps=steps,
                    stopping_criteria=(
                        "Follow the LLM-generated plan until the goal is answered "
                        "or evidence is exhausted."
                    ),
                    notes=plan_text,
                )

        plan = _create_fallback_plan(goal, corpus_keys)
        return plan


def generate_plan(goal: str, corpus_keys: list[str]) -> AgentPlan:
    return Planner().generate_plan(goal, corpus_keys)
