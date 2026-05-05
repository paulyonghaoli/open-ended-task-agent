"""
Post-observation reflection implementing the Reflexion pattern (Shinn et al., 2023).

After each tool execution the agent calls `reflect()` to:
  1. Assess whether the latest observation moved the goal forward.
  2. Identify evidence gaps that remain.
  3. Produce a signed confidence delta that updates AgentState.evidence_quality.
  4. Signal whether re-planning is warranted.

When an LLM is available the assessment uses a structured prompt constrained to
three lines so the result is deterministically parseable.  When no LLM is
available a rule-based fallback evaluates observation count, source diversity,
and failure rate.
"""
from __future__ import annotations

import re

from .config import (
    EVIDENCE_QUALITY_HIGH,
    EVIDENCE_QUALITY_MEDIUM,
    MINIMUM_EVIDENCE_FOR_CONFIDENCE,
)
from .llm import is_openai_available, safe_generate
from .state import AgentState, ReflectionNote, TraceEventType

_REFLECT_SYSTEM = (
    "You are a metacognitive evaluator for a task-execution agent. "
    "Respond with exactly three lines — no other text."
)

_REFLECT_PROMPT = """\
Goal: {goal}

Observations so far ({n_obs} total):
{obs_text}

Latest result (tool={tool}, success={success}):
{latest_summary}

Respond in EXACTLY this format:
Assessment: <one sentence on whether evidence is sufficient to answer the goal>
Gaps: <comma-separated missing pieces, or "none">
Replan: <yes or no>
"""


# ── Deterministic fallback ─────────────────────────────────────────────────────

def _rule_based_reflect(state: AgentState, result: dict) -> ReflectionNote:
    n_obs = len(state.observations)
    unique_sources = {obs.get("source", "") for obs in state.observations}
    gaps: list[str] = []

    if n_obs < MINIMUM_EVIDENCE_FOR_CONFIDENCE:
        gaps.append("more supporting evidence")
    if len(unique_sources) < 2:
        gaps.append("evidence from additional sources")
    if state.failures > 0 and state.replan_count == 0:
        gaps.append("successful retrieval on alternative query")

    sufficient = n_obs >= MINIMUM_EVIDENCE_FOR_CONFIDENCE and len(unique_sources) >= 2
    assessment = (
        "Evidence is sufficient to draft a grounded answer."
        if sufficient
        else f"Evidence is sparse ({n_obs} observations, {len(unique_sources)} source(s)); further retrieval recommended."
    )
    confidence_delta = 0.15 if not gaps else (-0.05 if result.get("success") else -0.12)
    quality_now = max(0.0, min(1.0, state.evidence_quality + confidence_delta))
    should_replan = bool(gaps) and state.replan_count == 0 and not sufficient

    return ReflectionNote(
        step=state.step_count,
        assessment=assessment,
        evidence_gaps=gaps,
        confidence_delta=confidence_delta,
        should_replan=should_replan,
        evidence_quality=quality_now,
    )


# ── LLM-based reflection ───────────────────────────────────────────────────────

def _parse_llm_reflection(text: str, step: int, current_quality: float) -> ReflectionNote:
    lines = {
        k.strip().lower(): v.strip()
        for line in text.splitlines()
        if ":" in line
        for k, v in [line.split(":", 1)]
    }
    assessment = lines.get("assessment", "Unable to assess evidence quality.")
    gaps_raw = lines.get("gaps", "")
    replan_raw = lines.get("replan", "no").lower()

    gaps = (
        []
        if not gaps_raw or gaps_raw.lower() == "none"
        else [g.strip() for g in re.split(r",\s*", gaps_raw) if g.strip()]
    )
    should_replan = "yes" in replan_raw
    confidence_delta = -0.05 * len(gaps) if gaps else 0.15
    quality_now = max(0.0, min(1.0, current_quality + confidence_delta))

    return ReflectionNote(
        step=step,
        assessment=assessment,
        evidence_gaps=gaps,
        confidence_delta=confidence_delta,
        should_replan=should_replan,
        evidence_quality=quality_now,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def reflect(state: AgentState, result: dict) -> ReflectionNote:
    """Assess the latest observation and update AgentState.evidence_quality."""
    from .config import LLM_REFLECT_MAX_TOKENS  # avoid circular at module load

    note: ReflectionNote

    if is_openai_available():
        obs_lines = "\n".join(
            f"  [{i+1}] ({o.get('source','?')}) {str(o.get('summary',''))[:120]}"
            for i, o in enumerate(state.observations[-5:])
        )
        latest_summary = str(result.get("output", result.get("reason", "")))[:200]
        prompt = _REFLECT_PROMPT.format(
            goal=state.goal,
            n_obs=len(state.observations),
            obs_text=obs_lines or "  (none yet)",
            tool=result.get("tool", "?"),
            success=result.get("success", False),
            latest_summary=latest_summary,
        )
        text = safe_generate(
            prompt,
            fallback="",
            system=_REFLECT_SYSTEM,
            max_tokens=LLM_REFLECT_MAX_TOKENS,
        )
        if text and "Assessment:" in text:
            note = _parse_llm_reflection(text, state.step_count, state.evidence_quality)
        else:
            note = _rule_based_reflect(state, result)
    else:
        note = _rule_based_reflect(state, result)

    # Commit to state
    state.reflections.append(note)
    state.evidence_quality = note.evidence_quality

    # Update string confidence label
    if state.evidence_quality >= EVIDENCE_QUALITY_HIGH:
        state.confidence = "high"
    elif state.evidence_quality >= EVIDENCE_QUALITY_MEDIUM:
        state.confidence = "medium"
    else:
        state.confidence = "low"

    # Log to trace
    state.log_event(
        TraceEventType.REFLECTION_LOGGED,
        reason=note.assessment,
        metadata={
            "evidence_gaps": note.evidence_gaps,
            "should_replan": note.should_replan,
            "evidence_quality": round(note.evidence_quality, 3),
            "confidence": state.confidence,
        },
    )

    if note.evidence_gaps:
        state.log_event(
            TraceEventType.EVIDENCE_GAP_DETECTED,
            reason=f"Gaps identified: {', '.join(note.evidence_gaps)}",
            metadata={"gap_count": len(note.evidence_gaps)},
        )

    return note
