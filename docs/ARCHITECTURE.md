# OETA-Lab Architecture

## Overview

OETA-Lab is a modular cognitive runtime for goal-driven agents. It implements
ReAct+Reflexion as a stable foundation and adds independent cognitive subsystems
that can each be understood, tested, and replaced in isolation.

## Module Map

```
task_agent/
├── agent.py          Main loop: integrates all subsystems
├── state.py          AgentState dataclass; TraceEventType constants
├── config.py         Numeric budgets and thresholds

── Retrieval ──────────────────────────────────────────────────────
├── knowledge.py      TF-IDF KnowledgeGraph with sentence-level snippets
├── tools.py          Tool implementations: search, compare, summarize, calc

── Planning ───────────────────────────────────────────────────────
├── planner.py        Goal decomposition; LLM + deterministic fallbacks
├── tool_selector.py  Prerequisite-enforcing action selection

── ReAct ──────────────────────────────────────────────────────────
├── llm.py            OpenAI v1 client wrapper with deterministic fallback
├── trace.py          Trace event formatting

── Reflexion ──────────────────────────────────────────────────────
├── reflection.py     ReflectionNote generation; evidence quality update

── Termination ────────────────────────────────────────────────────
├── termination.py    5 priority-ordered termination conditions

── Cognitive Substrate (OETA-Lab Phase 1) ─────────────────────────
├── router.py         CognitiveMode routing: SYMBOLIC, RETRIEVAL, LANGUAGE, ...
├── symbolic.py       Deterministic tools: count_characters, safe_arithmetic_eval
├── time_state.py     Temporal cognition: idle, stall, checkpoint, deadline
├── ledger.py         TaskLedger: known_facts, open_questions, failed_attempts
├── verifier.py       Post-execution sanity check; VerificationReport
├── memory.py         EpisodeMemoryStore: flat-file episode persistence
├── sleep_cycle.py    Offline replay: skill extraction, failure summary
├── export.py         Serialise full AgentState to JSON-serialisable dict

── APIs ────────────────────────────────────────────────────────────
├── api.py            FastAPI server
└── __main__.py       CLI: --json, --memory, --steps, --no-trace
```

## Data Flow

```
User goal
    │
    ▼
route_reasoning_mode()  →  RoutingDecision (SYMBOLIC / RETRIEVAL / ...)
    │
    ▼
generate_plan()         →  AgentPlan [PlanStep, ...]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  while pending_steps:                                           │
│    choose_next_action(state)   # prerequisite-safe selection    │
│    _reason(step, state)        # ReAct: Thought                 │
│    _execute_step(step, state)  # ReAct: Act + Observe           │
│    reflect(state, result)      # Reflexion: gap detection        │
│    should_terminate(state)     # 5-condition priority check      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
verify_state(state)     →  VerificationReport
    │
    ▼
(optional) EpisodeMemoryStore.append_episode(state)
    │
    ▼
AgentState  →  export_state()  →  JSON
```

## Cognitive Substrate Routing

Goals are classified into one of six cognitive modes before planning:

| Mode | Trigger words | Planned action |
|---|---|---|
| SYMBOLIC | count, how many, calculate, factorial | `symbolic` tool |
| RETRIEVAL | compare, explain, research, study plan | search + compare |
| GEOMETRIC | angle, area, triangle, coordinate | (future) |
| EXECUTION | debug, implement, script, algorithm | (future) |
| GRAPH | graph, path, node, edge | (future) |
| LANGUAGE | fallback | search + summarize |

Routing is deterministic (regex-based). The decision is logged as
`ROUTING_DECIDED` in the trace.

## Evidence Quality Lifecycle

```
Initial: evidence_quality = 0.0, confidence = "unknown"

After successful search (new source): +0.15
After detected evidence gap:          -0.05 × gap_count
After failed tool call:               -0.12
After successful symbolic step:       → 1.0 (deterministic, exact)

Thresholds (config.py):
  EVIDENCE_QUALITY_HIGH   = 0.55  →  confidence = "high"
  EVIDENCE_QUALITY_MEDIUM = 0.25  →  confidence = "medium"
  Below medium            →        confidence = "low"
```

## Trace Event Sequence

A typical comparison run produces events in this order:

```
GOAL_RECEIVED
ROUTING_DECIDED
PLAN_CREATED
REASONING          ← before step 1
ACTION_SELECTED    ← step 1 action
OBSERVATION_RECORDED
REFLECTION_LOGGED
REASONING          ← before step 2
ACTION_SELECTED    ← step 2 action
OBSERVATION_RECORDED
REFLECTION_LOGGED
REASONING          ← before compare step
ACTION_SELECTED
OBSERVATION_RECORDED
REASONING          ← before summarize
ACTION_SELECTED
TOOL_CALLED
TERMINATED_SUCCESS
VERIFICATION_LOGGED
```

## Termination Priority

```
1. sufficiency_status == "sufficient"  →  TERMINATED_SUCCESS
2. unsupported == True                 →  TERMINATED_FAILED
3. step_count >= max_steps             →  TERMINATED_MAX_STEPS
4. failures >= 3                       →  TERMINATED_PARTIAL
5. no_progress_count >= 2              →  TERMINATED_PARTIAL
```

## Sleep Cycle

The sleep cycle runs offline against stored episode files:

```
.oeta_memory/           # Episode JSON files (written by --memory flag)
    episode_<ts>.json

scripts/sleep_cycle.py  # CLI entry point

Output → .oeta_sleep/
    sleep_report_<ts>.json    # Full analysis
    candidate_skills.json     # (routing_mode, trigger_word) patterns
    failure_summary.json      # Grouped failure analysis
```

No live agent calls, no network access, no self-modification.
