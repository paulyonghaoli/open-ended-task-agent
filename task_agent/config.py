from __future__ import annotations

# ── Execution budget ───────────────────────────────────────────────────────────
DEFAULT_MAX_STEPS: int = 10
REPLAN_LIMIT: int = 2
MAX_SEARCH_RESULTS: int = 4
MIN_TFIDF_SCORE: float = 0.015
SNIPPET_SENTENCES: int = 4

# ── Evidence quality thresholds ────────────────────────────────────────────────
MINIMUM_EVIDENCE_FOR_CONFIDENCE: int = 2
EVIDENCE_QUALITY_HIGH: float = 0.55
EVIDENCE_QUALITY_MEDIUM: float = 0.25

# ── LLM settings ──────────────────────────────────────────────────────────────
LLM_MODEL: str = "gpt-4o-mini"
LLM_PLAN_MAX_TOKENS: int = 500
LLM_COMPARE_MAX_TOKENS: int = 700
LLM_SUMMARIZE_MAX_TOKENS: int = 700
LLM_REFLECT_MAX_TOKENS: int = 200
LLM_REASON_MAX_TOKENS: int = 150

# ── Termination ────────────────────────────────────────────────────────────────
FAILURE_BUDGET: int = 3
NO_PROGRESS_BUDGET: int = 2
