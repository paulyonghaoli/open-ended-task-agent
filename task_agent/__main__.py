"""
CLI entry point for the Open-Ended Task Agent.

Usage:
    python -m task_agent "Compare DuckDB and SQLite for a local analytics app"
    python -m task_agent --steps 12 "Research LangGraph vs AutoGen"
    python -m task_agent --no-trace "Create a study plan for learning ROS2"

Colour output is enabled automatically when the terminal supports it.
Falls back to plain ASCII when stdout is not a TTY or encoding is limited.
"""
from __future__ import annotations

import argparse
import sys
import textwrap

from .agent import run_agent
from .termination import termination_summary
from .trace import format_trace_line

# ── ANSI / colour detection ────────────────────────────────────────────────────
_USE_COLOUR = (
    sys.stdout.isatty()
    and sys.stdout.encoding is not None
    and sys.stdout.encoding.lower().replace("-", "") in ("utf8", "utf-8", "utf8bom")
)

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"

_CONF_COLOUR = {"high": _GREEN, "medium": _YELLOW, "low": _RED, "unknown": _DIM}


def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{_RESET}" if _USE_COLOUR else text


def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    if _USE_COLOUR:
        return _c(_GREEN, "#" * filled) + _c(_DIM, "." * (width - filled))
    return "#" * filled + "." * (width - filled)


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(
        text, width=88, initial_indent=prefix, subsequent_indent=prefix
    )


def _print_header(title: str) -> None:
    bar = "-" * 72
    print(f"\n{_c(_CYAN, bar)}")
    print(f"{_c(_BOLD, f'  {title}')}")
    print(_c(_CYAN, bar))


def _print_section(label: str, colour: str = _BOLD) -> None:
    print(f"\n{_c(colour, f'>> {label}')}")


def _print_final_answer(state) -> None:
    _print_header("FINAL ANSWER")
    conf_colour = _CONF_COLOUR.get(state.confidence, _DIM)
    conf_bar = _bar(state.evidence_quality)
    print(
        f"\n  Confidence : {_c(conf_colour, state.confidence.upper())}  "
        f"{conf_bar}  {state.evidence_quality:.0%}"
    )
    print(f"  Status     : {state.sufficiency_status}")
    print(f"  Sources    : {state.source_diversity} unique")
    print(f"  Elapsed    : {state.elapsed_seconds}s\n")
    answer = state.final_answer or "(no answer generated)"
    for para in answer.split("\n"):
        if para.strip():
            print(_wrap(para.strip()))
        else:
            print()


def _print_evidence(state) -> None:
    if not state.observations:
        return
    _print_section("EVIDENCE COLLECTED", _CYAN)
    for i, obs in enumerate(state.observations, 1):
        src = obs.get("source", "?")
        summary = obs.get("summary", "")[:160]
        score = obs.get("score", "")
        score_str = f" [score={score:.3f}]" if isinstance(score, float) else ""
        print(f"\n  {_c(_BOLD, f'[{i}] {src}')}{_c(_DIM, score_str)}")
        print(
            _wrap(
                summary + ("..." if len(obs.get("summary", "")) > 160 else ""),
                indent=6,
            )
        )


def _print_reflections(state) -> None:
    if not state.reflections:
        return
    _print_section("REFLECTIONS", _MAGENTA)
    for ref in state.reflections:
        gaps = ", ".join(ref.evidence_gaps) if ref.evidence_gaps else "none"
        replan_flag = _c(_YELLOW, " [replan]") if ref.should_replan else ""
        print(f"\n  Step {ref.step}: {ref.assessment}")
        print(f"    Gaps: {_c(_DIM, gaps)}{replan_flag}")


def _print_trace(state) -> None:
    _print_section("EXECUTION TRACE", _BLUE)
    for event in state.trace:
        etype = event.get("event_type", "")
        if etype.startswith("TERMINATED_"):
            colour = (
                _GREEN
                if "SUCCESS" in etype
                else (_YELLOW if "PARTIAL" in etype or "MAX_STEPS" in etype else _RED)
            )
            print(_c(colour, format_trace_line(event)))
        elif etype == "REASONING":
            print(_c(_DIM, format_trace_line(event)))
        else:
            print(format_trace_line(event))


def _print_stats(state) -> None:
    summary = termination_summary(state)
    _print_section("RUN STATISTICS", _DIM)
    rows = [
        ("Steps taken", summary["steps_taken"]),
        ("Observations", summary["observations_collected"]),
        ("Source diversity", summary["source_diversity"]),
        ("Evidence quality", f"{summary['evidence_quality']:.3f}"),
        ("Re-plans triggered", summary["replans"]),
        ("Failures", summary["failures"]),
        ("Elapsed", f"{summary['elapsed_s']}s"),
    ]
    for label, value in rows:
        print(f"  {label + ':':<30} {value}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="task-agent",
        description="Open-Ended Task Agent -- plan, act, observe, reflect, terminate.",
    )
    parser.add_argument("goal", nargs="*", help="The goal to pursue.")
    parser.add_argument(
        "--steps", type=int, default=None, help="Max execution steps (default: 10)."
    )
    parser.add_argument("--no-trace", action="store_true", help="Suppress trace output.")
    parser.add_argument(
        "--no-reflect", action="store_true", help="Suppress reflection output."
    )
    args = parser.parse_args()

    if not args.goal:
        parser.print_help()
        raise SystemExit(1)

    goal = " ".join(args.goal).strip()
    kwargs: dict = {}
    if args.steps:
        kwargs["max_steps"] = args.steps

    _print_header("OPEN-ENDED TASK AGENT")
    print(f"\n  {_c(_BOLD, 'Goal:')} {goal}\n")

    state = run_agent(goal, **kwargs)

    _print_final_answer(state)
    _print_evidence(state)

    if not args.no_reflect:
        _print_reflections(state)

    if not args.no_trace:
        _print_trace(state)

    _print_stats(state)
    print()


if __name__ == "__main__":
    main()
