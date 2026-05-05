"""
Agent execution loop implementing ReAct + Reflexion.

Loop structure (per step):
  1. REASON  — generate a Thought explaining why the next action was chosen
               and what evidence it is expected to produce.
  2. ACT     — execute the selected tool.
  3. OBSERVE — record the tool result as a structured observation.
  4. REFLECT — assess observation quality, identify evidence gaps, and
               update the AgentState.evidence_quality score.
  5. REPLAN  — if the reflection signals a gap and the replan budget allows,
               insert a corrective step at the front of the pending queue.
  6. CHECK   — evaluate termination conditions; stop if any are met.

References:
  ReAct:     Yao et al. 2022 (arXiv:2210.03629)
  Reflexion: Shinn et al. 2023 (arXiv:2303.11366)
"""
from __future__ import annotations

from typing import Any

from .config import DEFAULT_MAX_STEPS, LLM_REASON_MAX_TOKENS, REPLAN_LIMIT
from .knowledge import KnowledgeGraph
from .llm import is_openai_available, safe_generate
from .planner import generate_plan
from .reflection import reflect
from .state import AgentPlan, AgentState, PlanStep, TraceEventType
from .termination import should_terminate, termination_summary
from .tool_selector import choose_next_action
from .tools import (
    KnowledgeGraph,
    calculator,
    llm_compare,
    llm_summarize,
    load_corpus,
    search_local_corpus,
    summarize_findings,
    compare_items,
)

_REASON_SYSTEM = "You are a reasoning agent. Think step by step. Be concise."
_REASON_PROMPT = """\
Goal: {goal}
Completed steps: {done}
Pending steps: {pending}
Observations so far: {n_obs}
Evidence quality: {quality:.0%}

Next action: {action} on "{target}"

In one sentence, explain WHY this action is the right next step and WHAT evidence you expect it to produce.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_targets(target: str) -> list[str]:
    return [t.strip() for t in target.split(",") if t.strip()]


def _make_search_step(step_id: int, target: str, rationale: str = "") -> PlanStep:
    return PlanStep(
        id=step_id,
        action_type="search",
        description=f"Refined search: {target}",
        target=target,
        expected_evidence="additional corpus evidence",
        rationale=rationale or "Inserted by re-planner to fill evidence gap",
    )


def _make_summarize_step(step_id: int, goal: str) -> PlanStep:
    return PlanStep(
        id=step_id,
        action_type="summarize",
        description="Synthesize accumulated evidence into a final answer",
        target=goal,
        expected_evidence="final answer with confidence and citations",
        rationale="All planned steps exhausted; producing best available answer",
    )


# ── Reasoning step (ReAct: Thought) ───────────────────────────────────────────

def _reason(current: PlanStep, state: AgentState) -> str:
    """Generate a one-sentence Thought before acting."""
    if not is_openai_available():
        return (
            f"Taking action '{current.action_type}' on '{current.target}' — "
            f"expecting: {current.expected_evidence}."
        )
    prompt = _REASON_PROMPT.format(
        goal=state.goal,
        done=len(state.completed_steps),
        pending=len(state.pending_steps),
        n_obs=len(state.observations),
        quality=state.evidence_quality,
        action=current.action_type,
        target=current.target,
    )
    return safe_generate(
        prompt,
        fallback=f"Executing {current.action_type} on '{current.target}'.",
        system=_REASON_SYSTEM,
        max_tokens=LLM_REASON_MAX_TOKENS,
    )


# ── Tool execution (ReAct: Act + Observe) ─────────────────────────────────────

def _execute_step(
    current: PlanStep, state: AgentState, corpus: KnowledgeGraph
) -> dict[str, Any]:
    # ── search ──────────────────────────────────────────────────────────────────
    if current.action_type == "search":
        results = search_local_corpus(current.target, corpus)
        state.tool_calls.append(
            {"tool": "local_search", "target": current.target, "hits": len(results)}
        )

        if results:
            for item in results:
                state.observations.append(
                    {
                        "source": item["source"],
                        "summary": item["summary"],
                        "score": item.get("score", 0),
                    }
                )
            state.log_event(
                TraceEventType.OBSERVATION_RECORDED,
                reason=f"{len(results)} passage(s) retrieved",
                metadata={
                    "result_count": len(results),
                    "sources": [r["source"] for r in results],
                    "top_score": results[0].get("score", 0) if results else 0,
                },
            )
            return {"success": True, "tool": "local_search", "output": results}

        # No hits — attempt one refinement
        if state.replan_count < REPLAN_LIMIT:
            state.replan_count += 1
            refined = f"{current.target} overview basics"
            state.pending_steps.insert(
                0, _make_search_step(current.id + 1, refined, "Refined query after empty result")
            )
            state.log_event(
                TraceEventType.REPLAN_TRIGGERED,
                reason="search returned no hits",
                metadata={"refined_target": refined, "replan_count": state.replan_count},
            )
            return {"success": False, "tool": "local_search", "reason": "no_hits", "replan": True}

        state.unsupported = True
        return {"success": False, "tool": "local_search", "reason": "no_hits_exhausted"}

    # ── compare ─────────────────────────────────────────────────────────────────
    if current.action_type == "compare":
        targets = _parse_targets(current.target)
        result = llm_compare(targets, state.observations, state.goal)
        state.tool_calls.append(
            {"tool": "compare_items", "targets": targets, "success": result.get("success", False)}
        )

        if result.get("success"):
            state.observations.append(
                {"source": "comparison", "summary": result["comparison"]}
            )
            state.log_event(
                TraceEventType.OBSERVATION_RECORDED,
                reason="Comparison matrix recorded",
                metadata={"winner": result.get("winner"), "confidence": result.get("confidence")},
            )
            return {"success": True, "tool": "compare_items", "output": result}

        # Comparison failed — replan with fallback search
        if state.replan_count < REPLAN_LIMIT:
            state.replan_count += 1
            fallback_target = " and ".join(targets)
            state.pending_steps.insert(
                0,
                _make_search_step(
                    current.id + 1,
                    fallback_target,
                    "Comparison lacked evidence; re-searching",
                ),
            )
            state.log_event(
                TraceEventType.REPLAN_TRIGGERED,
                reason="comparison lacked sufficient evidence",
                metadata={"fallback_target": fallback_target},
            )
            return {
                "success": False,
                "tool": "compare_items",
                "reason": "insufficient_evidence",
                "replan": True,
            }
        return {"success": False, "tool": "compare_items", "reason": "insufficient_evidence"}

    # ── summarize ───────────────────────────────────────────────────────────────
    if current.action_type == "summarize":
        summary = llm_summarize(state.goal, state.observations, state.confidence)
        state.final_answer = summary["final"]
        state.confidence = summary["confidence"]
        state.sufficiency_status = "sufficient" if state.observations else "insufficient"
        state.tool_calls.append(
            {"tool": "summarize_findings", "confidence": state.confidence}
        )
        return {"success": True, "tool": "summarize_findings", "output": summary}

    # ── calculate ────────────────────────────────────────────────────────────────
    if current.action_type == "calculate":
        result = calculator(current.target)
        state.tool_calls.append(
            {"tool": "calculator", "target": current.target, "success": result.get("success", False)}
        )
        if result.get("success"):
            state.observations.append(
                {"source": "calculator", "summary": result["result"]}
            )
            return {"success": True, "tool": "calculator", "output": result}
        return {"success": False, "tool": "calculator", "reason": result.get("reason")}

    return {"success": False, "tool": current.action_type, "reason": "unknown_action_type"}


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_agent(goal: str, max_steps: int = DEFAULT_MAX_STEPS) -> AgentState:
    """
    Execute the ReAct+Reflexion agent loop for a given goal.

    Returns the final AgentState containing the answer, trace, observations,
    reflections, and termination summary.
    """
    # ── Initialisation ──────────────────────────────────────────────────────────
    raw_docs = load_corpus()
    corpus = KnowledgeGraph.from_documents(raw_docs)
    plan = generate_plan(goal, corpus.document_keys())

    state = AgentState(goal=goal, plan=plan, pending_steps=list(plan.steps))
    state.log_event(
        TraceEventType.GOAL_RECEIVED,
        reason="Goal accepted for planning",
        metadata={"goal": goal},
    )
    state.log_event(
        TraceEventType.PLAN_CREATED,
        reason=plan.stopping_criteria,
        metadata={
            "steps": len(plan.steps),
            "step_types": [s.action_type for s in plan.steps],
            "llm_guided": bool(plan.notes and "Deterministic" not in plan.notes),
        },
    )

    # ── Main loop ───────────────────────────────────────────────────────────────
    while True:
        # Ensure there are pending steps; synthesize a summary step if exhausted
        if not state.pending_steps:
            if state.final_answer:
                state.sufficiency_status = "sufficient"
                break
            if state.observations:
                state.pending_steps.append(
                    _make_summarize_step(
                        len(state.completed_steps) + 1, goal
                    )
                )
            else:
                state.unsupported = True
                break

        # ── 1. Select action ─────────────────────────────────────────────────────
        current = choose_next_action(state)
        if current in state.pending_steps:
            state.pending_steps.remove(current)
        state.step_count += 1

        # ── 2. Reason (ReAct: Thought) ──────────────────────────────────────────
        thought = _reason(current, state)
        state.reasoning_log.append(thought)
        state.log_event(
            TraceEventType.REASONING,
            tool_name=current.action_type,
            reason=thought,
            metadata={"step": state.step_count, "target": current.target},
        )

        # Log the action selection
        state.log_event(
            TraceEventType.ACTION_SELECTED,
            tool_name=current.action_type,
            reason=current.description,
            metadata={"target": current.target, "rationale": current.rationale},
        )

        # ── 3. Execute tool (ReAct: Act + Observe) ──────────────────────────────
        result = _execute_step(current, state, corpus)
        state.log_event(
            TraceEventType.TOOL_CALLED,
            tool_name=result.get("tool"),
            reason=result.get("reason") or ("success" if result.get("success") else "failed"),
            metadata={
                "success": result.get("success", False),
                "replan": result.get("replan", False),
            },
        )

        # ── 4. Reflect (Reflexion: assess + gap-detect) ─────────────────────────
        if current.action_type != "summarize":
            reflection = reflect(state, result)
            # Only trigger gap-driven replan when:
            #  (a) reflection signals a gap
            #  (b) budget allows
            #  (c) last result succeeded (no point re-searching after failure)
            #  (d) there are NO more search steps already in the pending queue
            #      (the plan may already address the gap)
            pending_searches = [
                s for s in state.pending_steps if s.action_type == "search"
            ]
            if (
                reflection.should_replan
                and state.replan_count < REPLAN_LIMIT
                and result.get("success")
                and not pending_searches
            ):
                # Use the goal as the gap query, not the generic gap description
                gap_query = state.goal
                state.replan_count += 1
                state.pending_steps.insert(
                    0,
                    _make_search_step(
                        state.step_count + 1,
                        gap_query,
                        "Gap-driven replan: additional evidence needed",
                    ),
                )
                state.log_event(
                    TraceEventType.REPLAN_TRIGGERED,
                    reason="Reflection gap: additional evidence needed",
                    metadata={"replan_count": state.replan_count},
                )

        # ── 5. Update progress counters ─────────────────────────────────────────
        if result.get("success"):
            state.completed_steps.append(current)
            state.no_progress_count = 0
        else:
            state.failures += 1
            state.no_progress_count += 1

        # ── 6. Check termination ────────────────────────────────────────────────
        stop, event_type = should_terminate(state, max_steps=max_steps)
        if stop:
            state.log_event(
                event_type,
                reason=event_type,
                metadata=termination_summary(state),
            )
            break

    # ── Fallback answer ─────────────────────────────────────────────────────────
    if not state.final_answer:
        if state.observations:
            fallback = summarize_findings(goal, state.observations, state.confidence)
            state.final_answer = fallback["final"]
        else:
            state.final_answer = (
                "The agent could not produce an answer. "
                "The goal may be outside the scope of the local knowledge corpus."
            )
        state.confidence = state.confidence if state.confidence != "unknown" else "low"
        if state.sufficiency_status == "unknown":
            state.sufficiency_status = "insufficient"

    return state
