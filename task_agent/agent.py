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

import re
from typing import Any

from .config import DEFAULT_MAX_STEPS, LLM_REASON_MAX_TOKENS, REPLAN_LIMIT
from .knowledge import KnowledgeGraph
from .ledger import TaskLedger
from .llm import is_openai_available, safe_generate
from .planner import generate_plan
from .reflection import reflect
from .router import route_reasoning_mode
from .state import AgentPlan, AgentState, PlanStep, TraceEventType
from .symbolic import count_characters, factorial, n_choose_k, safe_arithmetic_eval
from .termination import should_terminate, termination_summary
from .time_state import TimeState
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
from .verifier import verify_state

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


# ── Symbolic dispatch ──────────────────────────────────────────────────────────

def _dispatch_symbolic(goal: str) -> dict[str, Any]:
    """Route a natural-language symbolic goal to the appropriate pure function."""
    text = goal.strip()

    # 1. Character count: "count <char> in <word>"
    m = re.search(
        r"count\s+(?:the\s+)?(?:number\s+of\s+)?([a-zA-Z])\s+in\s+(\w+)",
        text, re.I,
    )
    if m:
        return count_characters(m.group(2), m.group(1))

    # 2. "how many <char>(s) in <word>"
    m = re.search(
        r"how\s+many\s+([a-zA-Z])[s']?\s+(?:are\s+(?:in|there\s+in)|in)\s+(\w+)",
        text, re.I,
    )
    if m:
        return count_characters(m.group(2), m.group(1))

    # 3. Factorial: "5!" or "factorial of 5"
    m = re.search(r"(\d+)\s*!", text) or re.search(
        r"factorial\s+(?:of\s+)?(\d+)", text, re.I
    )
    if m:
        return factorial(int(m.group(1)))

    # 4. C(n, k) combinations
    m = re.search(r"[Cc]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if m:
        return n_choose_k(int(m.group(1)), int(m.group(2)))

    # 5. Arithmetic expression
    m = re.search(r"[\d][\d\+\-\*\/\^\(\)\.\s]+[\d\)]", text)
    if m:
        expr = m.group(0).strip().replace("^", "**")
        if any(c in expr for c in "+-*/"):
            return safe_arithmetic_eval(expr)

    return {"success": False, "reason": "could not parse symbolic task from goal"}


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
    # ── symbolic ─────────────────────────────────────────────────────────────
    if current.action_type == "symbolic":
        result_data = _dispatch_symbolic(current.target)
        state.tool_calls.append(
            {
                "tool": "symbolic",
                "target": current.target,
                "success": result_data.get("success", False),
            }
        )
        if result_data.get("success"):
            summary = result_data.get("result") or str(result_data.get("expression", ""))
            state.observations.append({"source": "symbolic", "summary": summary})
            # Symbolic results are deterministic — always high confidence
            state.evidence_quality = 1.0
            state.confidence = "high"
            state.log_event(
                TraceEventType.SYMBOLIC_COMPUTED,
                tool_name="symbolic",
                reason=summary,
                metadata=result_data,
            )
            state.log_event(
                TraceEventType.OBSERVATION_RECORDED,
                reason="Symbolic computation result recorded",
                metadata={"result": summary},
            )
            return {"success": True, "tool": "symbolic", "output": result_data}
        return {"success": False, "tool": "symbolic", "reason": result_data.get("reason")}

    # ── search ────────────────────────────────────────────────────────────────
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
            if state.ledger is not None:
                state.ledger.record_fact(
                    f"Found {len(results)} passage(s) about '{current.target}'"
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
            if state.ledger is not None:
                state.ledger.record_failure(
                    f"Search for '{current.target}' returned no results"
                )
            return {"success": False, "tool": "local_search", "reason": "no_hits", "replan": True}

        state.unsupported = True
        return {"success": False, "tool": "local_search", "reason": "no_hits_exhausted"}

    # ── compare ───────────────────────────────────────────────────────────────
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
            if state.ledger is not None:
                state.ledger.record_fact(
                    f"Comparison complete; recommended: {result.get('winner', 'unknown')}"
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

    # ── summarize ─────────────────────────────────────────────────────────────
    if current.action_type == "summarize":
        summary = llm_summarize(state.goal, state.observations, state.confidence)
        state.final_answer = summary["final"]
        state.confidence = summary["confidence"]
        state.sufficiency_status = "sufficient" if state.observations else "insufficient"
        state.tool_calls.append(
            {"tool": "summarize_findings", "confidence": state.confidence}
        )
        return {"success": True, "tool": "summarize_findings", "output": summary}

    # ── calculate ─────────────────────────────────────────────────────────────
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

def run_agent(
    goal: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    enable_memory: bool = False,
) -> AgentState:
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

    # Attach cognitive subsystems
    routing = route_reasoning_mode(goal)
    state.routing = routing
    state.time_state = TimeState()
    state.ledger = TaskLedger(
        goal=goal,
        subgoals=[s.description for s in plan.steps],
    )

    state.log_event(
        TraceEventType.GOAL_RECEIVED,
        reason="Goal accepted for planning",
        metadata={"goal": goal},
    )
    state.log_event(
        TraceEventType.ROUTING_DECIDED,
        reason=routing.rationale,
        metadata=routing.to_dict(),
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

        # ── 1. Select action ──────────────────────────────────────────────────
        current = choose_next_action(state)
        if current in state.pending_steps:
            state.pending_steps.remove(current)
        state.step_count += 1

        if state.time_state is not None:
            state.time_state.update_progress()

        # ── 2. Reason (ReAct: Thought) ────────────────────────────────────────
        thought = _reason(current, state)
        state.reasoning_log.append(thought)
        state.log_event(
            TraceEventType.REASONING,
            tool_name=current.action_type,
            reason=thought,
            metadata={"step": state.step_count, "target": current.target},
        )

        state.log_event(
            TraceEventType.ACTION_SELECTED,
            tool_name=current.action_type,
            reason=current.description,
            metadata={"target": current.target, "rationale": current.rationale},
        )

        # ── 3. Execute tool (ReAct: Act + Observe) ───────────────────────────
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

        # ── 4. Reflect (Reflexion: assess + gap-detect) ───────────────────────
        if current.action_type not in ("summarize", "symbolic"):
            reflection = reflect(state, result)
            pending_searches = [
                s for s in state.pending_steps if s.action_type == "search"
            ]
            if (
                reflection.should_replan
                and state.replan_count < REPLAN_LIMIT
                and result.get("success")
                and not pending_searches
            ):
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

        # ── 5. Update progress counters ───────────────────────────────────────
        if result.get("success"):
            state.completed_steps.append(current)
            state.no_progress_count = 0
        else:
            state.failures += 1
            state.no_progress_count += 1

        # ── 6. Check termination ──────────────────────────────────────────────
        stop, event_type = should_terminate(state, max_steps=max_steps)
        if stop:
            state.log_event(
                event_type,
                reason=event_type,
                metadata=termination_summary(state),
            )
            if state.ledger is not None:
                state.ledger.close(reason=event_type)
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

    # ── Post-loop: verifier ─────────────────────────────────────────────────────
    verifier_report = verify_state(state)
    state.verifier_report = verifier_report
    state.log_event(
        TraceEventType.VERIFICATION_LOGGED,
        reason="passed" if verifier_report.passed else f"{len(verifier_report.issues)} issue(s) found",
        metadata=verifier_report.to_dict(),
    )
    if verifier_report.confidence_adjustment < 0 and state.confidence == "high":
        state.confidence = "medium"

    # ── Optional memory persistence ─────────────────────────────────────────────
    if enable_memory:
        from .memory import EpisodeMemoryStore
        EpisodeMemoryStore().append_episode(state)

    return state
