from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from .config import EVIDENCE_QUALITY_MEDIUM

if TYPE_CHECKING:
    from .state import AgentState


@dataclass
class VerificationReport:
    """
    Post-execution sanity check on an agent state.

    Inspects observations, confidence labels, and tool outcomes to detect
    common failure patterns: missing evidence, one-sided comparisons,
    confidence inflation, and repeated search failures.
    """

    passed: bool
    issues: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    contradiction_risk: float = 0.0
    confidence_adjustment: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "missing_evidence": self.missing_evidence,
            "contradiction_risk": round(self.contradiction_risk, 3),
            "confidence_adjustment": round(self.confidence_adjustment, 3),
        }


def verify_state(state: AgentState) -> VerificationReport:
    """Run deterministic checks against the final agent state."""
    issues: List[str] = []
    missing: List[str] = []
    contradiction_risk = 0.0
    confidence_adj = 0.0

    # 1. No evidence at all
    if not state.observations:
        issues.append("No evidence collected")
        missing.append("Any supporting evidence for the goal")
        confidence_adj -= 0.20

    # 2. One-sided comparison
    goal_lower = state.goal.lower()
    is_comparison = (
        "compare" in goal_lower
        or " vs " in goal_lower
        or (" and " in goal_lower and "compare" not in goal_lower and len(goal_lower) < 120)
    )
    if is_comparison and state.observations:
        sources = {obs.get("source", "") for obs in state.observations}
        if len(sources) < 2:
            issues.append("Comparison goal but evidence from only one source")
            missing.append("Evidence about the second comparison candidate")
            confidence_adj -= 0.10
            contradiction_risk += 0.30

    # 3. Unsupported goal but detailed answer generated
    if state.unsupported and state.final_answer and len(state.final_answer) > 60:
        issues.append("Goal flagged unsupported yet a detailed answer was produced")
        contradiction_risk += 0.40

    # 4. High confidence label with low evidence quality
    if state.confidence == "high" and state.evidence_quality < EVIDENCE_QUALITY_MEDIUM:
        issues.append(
            f"Confidence label 'high' but evidence_quality={state.evidence_quality:.2f}"
        )
        confidence_adj -= 0.15

    # 5. Multiple failed searches
    failed_searches = sum(
        1
        for tc in state.tool_calls
        if tc.get("tool") == "local_search" and not tc.get("success", True)
    )
    if failed_searches >= 2:
        issues.append(f"{failed_searches} search attempts returned no results")
        missing.append("Corpus coverage for this goal")

    passed = len(issues) == 0
    return VerificationReport(
        passed=passed,
        issues=issues,
        missing_evidence=missing,
        contradiction_risk=min(1.0, contradiction_risk),
        confidence_adjustment=max(-0.5, confidence_adj),
    )
