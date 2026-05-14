"""
Evaluation harness for OETA-Lab.

Runs a battery of test goals across four categories and reports:
  - final answer (truncated)
  - termination status
  - evidence quality score
  - confidence label
  - source diversity
  - steps taken
  - elapsed time
  - trace event histogram
  - routing mode
  - verifier issues

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --verbose
    python scripts/run_eval.py --save-json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_agent import run_agent

# ── Evaluation scenarios ───────────────────────────────────────────────────────

SCENARIOS = [
    # (label, goal, expected_status_prefix)
    ("comparison:duckdb-sqlite",
     "Compare DuckDB and SQLite for a local analytics app",
     "TERMINATED_SUCCESS"),
    ("comparison:langgraph-autogen",
     "Compare LangGraph and AutoGen for a small agent project",
     "TERMINATED_SUCCESS"),
    ("comparison:fastapi-flask",
     "Compare FastAPI and Flask for a beginner-friendly Python web app",
     "TERMINATED_SUCCESS"),
    ("study-plan:ros2",
     "Create a study plan for learning ROS2 basics in 2 weeks",
     "TERMINATED_SUCCESS"),
    ("symbolic:count-chars",
     "Count the number of r in strawberry",
     "TERMINATED_SUCCESS"),
    ("unsupported:out-of-scope",
     "Write a sonnet about Schrödinger equation and quantum entanglement",
     "TERMINATED_"),
    ("max-steps:budget",
     "Compare DuckDB and SQLite for a local analytics app",
     "TERMINATED_MAX_STEPS"),
    ("partial:vague",
     "Tell me about the best technology",
     "TERMINATED_"),
]

MAX_STEPS_FOR_BUDGET_TEST = 2

# ── ANSI ───────────────────────────────────────────────────────────────────────
_R = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_DIM = "\033[2m"


def _c(col, txt): return f"{col}{txt}{_R}"


def _status_colour(status: str) -> str:
    if "SUCCESS" in status:
        return _GREEN
    if "PARTIAL" in status or "MAX_STEPS" in status:
        return _YELLOW
    return _RED


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_scenario(label: str, goal: str, expected_prefix: str, verbose: bool) -> dict:
    kwargs = {}
    if label == "max-steps:budget":
        kwargs["max_steps"] = MAX_STEPS_FOR_BUDGET_TEST

    t0 = time.time()
    state = run_agent(goal, **kwargs)
    elapsed = round(time.time() - t0, 3)

    terminal_events = [
        e["event_type"] for e in state.trace if e["event_type"].startswith("TERMINATED_")
    ]
    status = terminal_events[-1] if terminal_events else "UNKNOWN"
    passed = status.startswith(expected_prefix)

    event_hist = Counter(e["event_type"] for e in state.trace)

    verifier_issues = 0
    verifier_passed = True
    if state.verifier_report is not None:
        verifier_issues = len(state.verifier_report.issues)
        verifier_passed = state.verifier_report.passed

    routing_mode = "unknown"
    if state.routing is not None:
        routing_mode = state.routing.selected_mode.value

    result = {
        "label": label,
        "goal": goal,
        "passed": passed,
        "status": status,
        "expected": expected_prefix,
        "steps": state.step_count,
        "evidence_quality": round(state.evidence_quality, 3),
        "confidence": state.confidence,
        "source_diversity": state.source_diversity,
        "observations": len(state.observations),
        "reflections": len(state.reflections),
        "replans": state.replan_count,
        "elapsed_s": elapsed,
        "routing_mode": routing_mode,
        "verifier_passed": verifier_passed,
        "verifier_issues": verifier_issues,
        "answer_preview": (state.final_answer or "")[:120].replace("\n", " "),
        "event_hist": dict(event_hist),
    }

    if verbose:
        _print_verbose(result)

    return result


def _print_verbose(r: dict) -> None:
    print(f"\n{'-' * 70}")
    print(f"  {_c(_BOLD, r['label'])}")
    print(f"  Goal    : {r['goal']}")
    print(f"  Status  : {_c(_status_colour(r['status']), r['status'])}  (expected: {r['expected']})")
    print(f"  Routing : {r['routing_mode']}")
    print(f"  Steps   : {r['steps']}  |  Observations: {r['observations']}  |  Reflections: {r['reflections']}")
    print(f"  Quality : {r['evidence_quality']:.3f}  Confidence: {r['confidence']}  Sources: {r['source_diversity']}")
    print(f"  Elapsed : {r['elapsed_s']}s  |  Replans: {r['replans']}")
    vok = "OK" if r["verifier_passed"] else f"WARN ({r['verifier_issues']} issues)"
    print(f"  Verifier: {_c(_GREEN if r['verifier_passed'] else _YELLOW, vok)}")
    print(f"  Answer  : {_c(_DIM, r['answer_preview'])}")


def _build_aggregate(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {}
    passed = sum(1 for r in results if r["passed"])
    routing_dist = Counter(r["routing_mode"] for r in results)
    verifier_issues = sum(r["verifier_issues"] for r in results)
    unsupported = sum(1 for r in results if r["status"] == "TERMINATED_FAILED")
    return {
        "total_scenarios": total,
        "passed": passed,
        "task_success_rate": round(passed / total, 3),
        "unsupported_claim_rate": round(unsupported / total, 3),
        "verifier_issue_rate": round(sum(1 for r in results if not r["verifier_passed"]) / total, 3),
        "avg_evidence_quality": round(sum(r["evidence_quality"] for r in results) / total, 3),
        "avg_steps": round(sum(r["steps"] for r in results) / total, 2),
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results) / total, 3),
        "routing_distribution": dict(routing_dist),
        "total_verifier_issues": verifier_issues,
    }


def _print_summary(results: list[dict]) -> None:
    agg = _build_aggregate(results)
    passed = agg["passed"]
    total = agg["total_scenarios"]
    rate = agg["task_success_rate"]
    colour = _GREEN if rate >= 0.8 else (_YELLOW if rate >= 0.5 else _RED)

    print(f"\n{'=' * 70}")
    print(f"  {_c(_BOLD, 'EVALUATION SUMMARY')}")
    print(f"{'=' * 70}")
    print(f"  Scenarios             : {total}")
    print(f"  Passed                : {_c(colour, f'{passed}/{total}')}  ({rate:.0%})")
    print(f"  Task success rate     : {agg['task_success_rate']:.1%}")
    print(f"  Unsupported rate      : {agg['unsupported_claim_rate']:.1%}")
    print(f"  Verifier issue rate   : {agg['verifier_issue_rate']:.1%}")
    print(f"  Avg evidence quality  : {agg['avg_evidence_quality']:.3f}")
    print(f"  Avg steps             : {agg['avg_steps']:.1f}")
    print(f"  Avg elapsed           : {agg['avg_elapsed_s']:.2f}s")
    if agg["routing_distribution"]:
        print(f"  Routing distribution  : {dict(agg['routing_distribution'])}")
    print()
    print(f"  {'Label':<30}  {'Status':<25}  {'Pass':<6}  {'Quality'}")
    print(f"  {'-' * 30}  {'-' * 25}  {'-' * 6}  {'-' * 7}")
    for r in results:
        col = _status_colour(r["status"])
        tick = _c(_GREEN, "OK  ") if r["passed"] else _c(_RED, "FAIL")
        print(f"  {r['label']:<30}  {_c(col, r['status']):<35}  {tick}    {r['evidence_quality']:.3f}")
    print()
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OETA evaluation harness.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--save-json", action="store_true", help="Write eval_results.json to disk."
    )
    args = parser.parse_args()

    print(f"\n{_c(_BOLD, '  OETA-Lab - Evaluation Harness')}")
    print(f"  Running {len(SCENARIOS)} scenarios...\n")

    results = []
    for label, goal, expected in SCENARIOS:
        print(f"  Running {_c(_BOLD, label)}...", end=" ", flush=True)
        r = run_scenario(label, goal, expected, args.verbose)
        tick = _c(_GREEN, "OK") if r["passed"] else _c(_RED, "FAIL")
        print(f"{tick}  ({r['elapsed_s']}s)")
        results.append(r)

    agg = _print_summary(results)

    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"  {_c(_YELLOW, 'Failed scenarios:')}")
        for r in failed:
            print(f"    - {r['label']}: got {r['status']}, expected {r['expected']}")
        print()

    if args.save_json:
        out = {
            "aggregate": agg,
            "scenarios": [
                {k: v for k, v in r.items() if k != "event_hist"} for r in results
            ],
        }
        out_path = Path("eval_results.json")
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  Results saved to {out_path}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
