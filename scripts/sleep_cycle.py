"""
Sleep cycle CLI — offline replay and skill extraction.

Reads stored episodes from .oeta_memory/ and writes analysis
artefacts to .oeta_sleep/:
  - sleep_report_<timestamp>.json
  - candidate_skills.json
  - failure_summary.json

Usage:
    python scripts/sleep_cycle.py
    python scripts/sleep_cycle.py --memory-dir /path/to/memory
    python scripts/sleep_cycle.py --sleep-dir /path/to/output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_agent.sleep_cycle import run_sleep_cycle

_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_R = "\033[0m"


def _c(col: str, txt: str) -> str:
    return f"{col}{txt}{_R}"


def main() -> None:
    parser = argparse.ArgumentParser(description="OETA sleep cycle: offline replay and skill extraction.")
    parser.add_argument(
        "--memory-dir", default=".oeta_memory", help="Path to episode memory directory."
    )
    parser.add_argument(
        "--sleep-dir", default=".oeta_sleep", help="Path for sleep cycle output artefacts."
    )
    parser.add_argument("--json", action="store_true", help="Print full report as JSON.")
    args = parser.parse_args()

    memory_dir = Path(args.memory_dir)
    sleep_dir = Path(args.sleep_dir)

    print(f"\n  {_c(_BOLD, 'OETA Sleep Cycle')}")
    print(f"  Memory : {memory_dir}")
    print(f"  Output : {sleep_dir}\n")

    report = run_sleep_cycle(memory_dir=memory_dir, sleep_dir=sleep_dir)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    n = report["episodes_processed"]
    print(f"  Episodes processed : {n}")

    if n == 0:
        print(f"\n  {_c(_YELLOW, 'No episodes found in memory directory.')}")
        print(f"  Run `python -m task_agent --memory \"...\"` first to populate memory.\n")
        return

    stats = report.get("replay_stats", {})
    status_dist = stats.get("status_distribution", {})
    routing_dist = stats.get("routing_distribution", {})
    print(f"  Avg evidence quality : {stats.get('avg_evidence_quality', 0):.3f}")
    print(f"  Avg steps            : {stats.get('avg_steps', 0):.1f}")

    if status_dist:
        print(f"\n  Status distribution:")
        for status, count in status_dist.items():
            print(f"    {status:<30} {count}")

    if routing_dist:
        print(f"\n  Routing distribution:")
        for mode, count in routing_dist.items():
            print(f"    {mode:<20} {count}")

    skills = report.get("candidate_skills", [])
    print(f"\n  Candidate skills extracted : {len(skills)}")
    for sk in skills[:5]:
        print(f"    [{sk['routing_mode']}] trigger='{sk['trigger_word']}' successes={sk['observed_successes']}")

    failures = report.get("failure_summary", [])
    print(f"\n  Failure groups : {len(failures)}")
    for f in failures:
        print(f"    {f['status']}: {f['count']} episode(s)")

    synthetic = report.get("synthetic_tasks", [])
    if synthetic:
        print(f"\n  Synthetic tasks generated:")
        for task in synthetic[:5]:
            print(f"    - {task}")

    print(f"\n  Artefacts written to: {sleep_dir}/\n")


if __name__ == "__main__":
    main()
