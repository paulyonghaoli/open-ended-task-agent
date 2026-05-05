"""
Optional FastAPI server for the task agent.

Install extras first:
    pip install -e .[web]

Start the server:
    uvicorn task_agent.api:app --reload

Endpoints:
    POST /run     — execute a goal and return the full agent response
    GET  /health  — liveness check
"""
from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, field_validator
    _fastapi_available = True
except ImportError:
    _fastapi_available = False

from .agent import run_agent
from .termination import termination_summary

if not _fastapi_available:
    raise ImportError(
        "FastAPI is not installed. Run: pip install -e .[web]"
    )


# ── Request / Response models ──────────────────────────────────────────────────

class RunRequest(BaseModel):
    goal: str
    debug: bool = False
    max_steps: int = 10

    @field_validator("goal")
    @classmethod
    def goal_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("goal must not be empty")
        return v.strip()

    @field_validator("max_steps")
    @classmethod
    def steps_in_range(cls, v: int) -> int:
        if not (1 <= v <= 30):
            raise ValueError("max_steps must be between 1 and 30")
        return v


class EvidenceItem(BaseModel):
    source: str
    summary: str
    score: Any = None


class ReflectionItem(BaseModel):
    step: int
    assessment: str
    evidence_gaps: list[str]
    confidence_delta: float
    should_replan: bool
    evidence_quality: float


class RunResponse(BaseModel):
    final_answer: str
    confidence: str
    evidence_quality: float
    status: str
    evidence: list[dict[str, Any]]
    stats: dict[str, Any]
    trace: list[dict[str, Any]]
    reflections: list[dict[str, Any]]


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Open-Ended Task Agent API",
    description=(
        "A compact higher-agency task agent that plans, retrieves evidence, "
        "reflects, and terminates safely. Implements ReAct + Reflexion patterns."
    ),
    version="0.2.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.2.0"}


@app.post("/run", response_model=RunResponse)
def run_task(request: RunRequest) -> RunResponse:
    """Execute the agent loop for the given goal and return the full response."""
    try:
        state = run_agent(request.goal, max_steps=request.max_steps)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    status = "success" if state.sufficiency_status == "sufficient" else "partial"
    stats = termination_summary(state)

    reflections_out = [
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

    trace_out = state.trace if request.debug else [
        e for e in state.trace
        if e.get("event_type") not in ("REASONING", "REFLECTION_LOGGED", "EVIDENCE_GAP_DETECTED")
    ]

    return RunResponse(
        final_answer=state.final_answer or "",
        confidence=state.confidence,
        evidence_quality=round(state.evidence_quality, 3),
        status=status,
        evidence=state.observations,
        stats=stats,
        trace=trace_out,
        reflections=reflections_out,
    )
