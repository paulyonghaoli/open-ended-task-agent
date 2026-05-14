from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .state import AgentState

MEMORY_DIR = Path(".oeta_memory")


@dataclass
class Episode:
    """Compressed record of one agent run, suitable for long-term storage."""

    goal: str
    status: str
    confidence: str
    evidence_quality: float
    steps_taken: int
    observations_count: int
    source_diversity: int
    final_answer_preview: str
    timestamp: float = field(default_factory=time.time)
    replans: int = 0
    failures: int = 0
    routing_mode: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class EpisodeMemoryStore:
    """
    Flat-file episode store backed by .oeta_memory/.

    Each episode is written as a single JSON file named by timestamp.
    This is intentionally simple — no database, no embeddings.
    """

    def __init__(self, memory_dir: Path = MEMORY_DIR) -> None:
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def append_episode(self, state: AgentState) -> Path:
        """Persist an AgentState as a compressed Episode record."""
        terminal_events = [
            e["event_type"]
            for e in state.trace
            if e["event_type"].startswith("TERMINATED_")
        ]
        status = terminal_events[-1] if terminal_events else "UNKNOWN"

        routing_mode: Optional[str] = None
        if state.routing is not None:
            routing_mode = state.routing.selected_mode.value  # type: ignore[attr-defined]

        episode = Episode(
            goal=state.goal,
            status=status,
            confidence=state.confidence,
            evidence_quality=round(state.evidence_quality, 3),
            steps_taken=state.step_count,
            observations_count=len(state.observations),
            source_diversity=state.source_diversity,
            final_answer_preview=(state.final_answer or "")[:200],
            replans=state.replan_count,
            failures=state.failures,
            routing_mode=routing_mode,
        )
        ts = int(episode.timestamp * 1000)
        path = self.memory_dir / f"episode_{ts}.json"
        path.write_text(json.dumps(episode.to_dict(), indent=2), encoding="utf-8")
        return path

    def load_recent(self, n: int = 10) -> List[Episode]:
        """Load the n most recent episodes, newest first."""
        files = sorted(self.memory_dir.glob("episode_*.json"), reverse=True)
        episodes: List[Episode] = []
        for f in files[:n]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                episodes.append(Episode.from_dict(data))
            except (json.JSONDecodeError, TypeError):
                continue
        return episodes

    def search_by_goal_terms(self, query: str) -> List[Episode]:
        """Return episodes whose goals overlap with query terms."""
        terms = {t.lower() for t in query.split() if len(t) > 2}
        matches: List[Episode] = []
        for ep in self.load_recent(n=100):
            goal_terms = {t.lower() for t in ep.goal.split()}
            if terms & goal_terms:
                matches.append(ep)
        return matches
