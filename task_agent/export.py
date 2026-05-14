from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .state import AgentState


def export_state(
    state: AgentState,
    *,
    include_trace: bool = True,
    include_reflections: bool = True,
) -> dict[str, Any]:
    """
    Serialize a complete AgentState into a JSON-serialisable dict.

    Includes temporal state, routing decision, verifier report,
    ledger, observations, trace, and run statistics.
    """
    terminal_events = [
        e["event_type"]
        for e in state.trace
        if e["event_type"].startswith("TERMINATED_")
    ]
    status = terminal_events[-1] if terminal_events else "UNKNOWN"

    out: dict[str, Any] = {
        "goal": state.goal,
        "status": status,
        "final_answer": state.final_answer,
        "confidence": state.confidence,
        "evidence_quality": round(state.evidence_quality, 3),
        "sufficiency_status": state.sufficiency_status,
        "stats": {
            "steps_taken": state.step_count,
            "observations": len(state.observations),
            "source_diversity": state.source_diversity,
            "replans": state.replan_count,
            "failures": state.failures,
            "elapsed_s": state.elapsed_seconds,
        },
        "observations": state.observations,
    }

    # Optional structured subsystems
    if state.routing is not None:
        out["routing"] = state.routing.to_dict()  # type: ignore[attr-defined]

    if state.verifier_report is not None:
        out["verifier"] = state.verifier_report.to_dict()  # type: ignore[attr-defined]

    if state.ledger is not None:
        out["ledger"] = state.ledger.to_dict()  # type: ignore[attr-defined]

    if state.time_state is not None:
        out["temporal"] = state.time_state.to_dict()  # type: ignore[attr-defined]

    if include_reflections:
        out["reflections"] = [
            {
                "step": r.step,
                "assessment": r.assessment,
                "evidence_gaps": r.evidence_gaps,
                "confidence_delta": r.confidence_delta,
                "should_replan": r.should_replan,
                "evidence_quality": r.evidence_quality,
            }
            for r in state.reflections
        ]

    if include_trace:
        out["trace"] = state.trace

    return out
