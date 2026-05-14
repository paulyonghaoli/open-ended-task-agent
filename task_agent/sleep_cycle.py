"""
Offline replay and skill extraction (the 'sleep cycle').

Operates exclusively on stored episode traces — no live agent calls,
no network access, no self-modification. Reads .oeta_memory/, writes
structured reports to .oeta_sleep/.

Analogous to offline experience replay in RL: distils past runs into
candidate skills and failure summaries that can inform future planning.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import List

from .memory import Episode, EpisodeMemoryStore

SLEEP_DIR = Path(".oeta_sleep")
MEMORY_DIR = Path(".oeta_memory")


def replay_recent_episodes(store: EpisodeMemoryStore, n: int = 20) -> dict:
    """Load recent episodes and compute aggregate statistics."""
    episodes = store.load_recent(n)
    if not episodes:
        return {"count": 0, "episodes": []}

    statuses = Counter(ep.status for ep in episodes)
    routing_modes = Counter(ep.routing_mode for ep in episodes if ep.routing_mode)
    avg_quality = sum(ep.evidence_quality for ep in episodes) / len(episodes)
    avg_steps = sum(ep.steps_taken for ep in episodes) / len(episodes)

    return {
        "count": len(episodes),
        "status_distribution": dict(statuses),
        "routing_distribution": dict(routing_modes),
        "avg_evidence_quality": round(avg_quality, 3),
        "avg_steps": round(avg_steps, 2),
        "episodes": [ep.to_dict() for ep in episodes],
    }


def extract_candidate_skills(episodes: List[Episode]) -> List[dict]:
    """
    Identify patterns that reliably led to SUCCESS.

    A 'candidate skill' is a (routing_mode, goal_word) pair observed in
    multiple successful runs.  These are candidates for future plan templates.
    """
    success_eps = [ep for ep in episodes if "SUCCESS" in ep.status]
    if not success_eps:
        return []

    pattern_counts: Counter = Counter()
    for ep in success_eps:
        mode = ep.routing_mode or "language"
        for word in ep.goal.lower().split():
            if len(word) > 3:
                pattern_counts[(mode, word)] += 1

    skills = []
    for (mode, word), count in pattern_counts.most_common(10):
        if count >= 2:
            skills.append(
                {
                    "routing_mode": mode,
                    "trigger_word": word,
                    "observed_successes": count,
                    "candidate_action": "retrieval" if mode == "retrieval" else "symbolic",
                }
            )
    return skills


def summarize_failures(episodes: List[Episode]) -> List[dict]:
    """Group failure episodes by status and routing mode."""
    fail_eps = [ep for ep in episodes if "SUCCESS" not in ep.status]
    if not fail_eps:
        return []

    by_status: dict = {}
    for ep in fail_eps:
        key = ep.status
        by_status.setdefault(key, []).append(
            {
                "goal": ep.goal,
                "routing_mode": ep.routing_mode,
                "evidence_quality": ep.evidence_quality,
                "steps_taken": ep.steps_taken,
            }
        )
    return [{"status": k, "count": len(v), "examples": v[:3]} for k, v in by_status.items()]


def generate_synthetic_tasks(episodes: List[Episode]) -> List[str]:
    """
    Produce simple goal variations from successful episodes.

    These synthetic tasks can seed future evaluation runs without
    requiring any LLM calls.
    """
    success_eps = [ep for ep in episodes if "SUCCESS" in ep.status]
    tasks: List[str] = []
    seen: set = set()

    for ep in success_eps:
        words = [w for w in ep.goal.split() if len(w) > 4]
        for i, word in enumerate(words):
            if i + 1 < len(words):
                variant = f"Explain {word} and {words[i + 1]}"
                if variant not in seen:
                    tasks.append(variant)
                    seen.add(variant)
        if len(tasks) >= 5:
            break

    return tasks


def run_sleep_cycle(memory_dir: Path = MEMORY_DIR, sleep_dir: Path = SLEEP_DIR) -> dict:
    """
    Execute a full sleep cycle: replay, extract, summarize, generate.

    Writes three artefacts to sleep_dir and returns a summary dict.
    """
    sleep_dir.mkdir(parents=True, exist_ok=True)
    store = EpisodeMemoryStore(memory_dir)
    ts = int(time.time())

    replay = replay_recent_episodes(store, n=50)
    episodes = [Episode.from_dict(e) for e in replay.get("episodes", [])]

    skills = extract_candidate_skills(episodes)
    failures = summarize_failures(episodes)
    synthetic = generate_synthetic_tasks(episodes)

    report = {
        "timestamp": ts,
        "episodes_processed": replay["count"],
        "replay_stats": {
            k: v for k, v in replay.items() if k != "episodes"
        },
        "candidate_skills": skills,
        "failure_summary": failures,
        "synthetic_tasks": synthetic,
    }

    report_path = sleep_dir / f"sleep_report_{ts}.json"
    skills_path = sleep_dir / "candidate_skills.json"
    failures_path = sleep_dir / "failure_summary.json"

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    skills_path.write_text(json.dumps(skills, indent=2), encoding="utf-8")
    failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    return report
