from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TaskLedger:
    """
    Running record of what the agent knows, assumes, and still needs to find.

    Updated incrementally during planning, retrieval, comparison, and
    termination so that observers can read the agent's epistemic state at
    any point in the trace.
    """

    goal: str
    subgoals: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    known_facts: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    failed_attempts: List[str] = field(default_factory=list)
    next_best_action: Optional[str] = None
    stop_reason: Optional[str] = None

    def record_fact(self, fact: str) -> None:
        if fact not in self.known_facts:
            self.known_facts.append(fact)

    def record_failure(self, attempt: str) -> None:
        self.failed_attempts.append(attempt)

    def add_question(self, question: str) -> None:
        if question not in self.open_questions:
            self.open_questions.append(question)

    def resolve_question(self, question: str) -> None:
        self.open_questions = [q for q in self.open_questions if q != question]

    def close(self, reason: str) -> None:
        self.stop_reason = reason

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "subgoals": self.subgoals,
            "assumptions": self.assumptions,
            "known_facts": self.known_facts,
            "open_questions": self.open_questions,
            "failed_attempts": self.failed_attempts,
            "next_best_action": self.next_best_action,
            "stop_reason": self.stop_reason,
        }
