# Open-Ended Task Agent (OETA)

**A Compact Implementation of ReAct–Reflexion Planning for Higher-Agency Task Execution Under Uncertainty**

> This repository is a deliberately compact, interview-friendly demonstration of higher-agency agent architecture. It is not a production framework. The focus is architectural clarity: explicit planning, traceable reasoning, bounded autonomy, and structured termination.

---

## Abstract

We present **OETA** (Open-Ended Task Agent), a compact implementation of a plan–act–observe–reflect loop for open-ended question answering over a local knowledge corpus. OETA integrates three research-grounded patterns: **ReAct** (interleaved reasoning and acting), **Reflexion** (post-observation self-assessment), and **retrieval-augmented generation** (evidence-grounded answers). The system decomposes a user goal into a structured plan, selects tools based on current evidence state, executes retrieval and comparison actions, reflects on observation quality after each step, and terminates under explicit conditions with a confidence-annotated final answer. All execution state, reasoning steps, reflections, and termination events are captured in a structured trace. The system operates without mandatory external dependencies and degrades gracefully to deterministic behavior when no LLM is available.

---

## 1. Introduction

Building agents that reliably accomplish open-ended goals requires solving three coupled problems simultaneously: *what to do next* (planning), *whether the result was useful* (observation assessment), and *when to stop* (safe termination). Most agent frameworks delegate these decisions entirely to an LLM, which produces agents that are brittle under distribution shift, unauditable, and prone to looping or hallucinating when evidence is sparse.

OETA approaches these problems differently. Planning is explicit and can be inspected before execution. Observation assessment is structured (a `ReflectionNote` with numeric quality delta and gap inventory). Termination is rule-based with five distinct conditions, each producing a named trace event. When an LLM is available it augments each of these with natural-language reasoning; when it is not, the system falls back to deterministic heuristics that maintain the same behavioral contract.

**Contributions:**

1. A readable, test-covered implementation of ReAct+Reflexion in under 800 lines of Python with zero mandatory dependencies.
2. A TF-IDF retrieval layer that replaces keyword counting with principled document ranking.
3. A multi-criteria evidence matrix for structured comparison tasks.
4. A numeric evidence-quality score (`evidence_quality ∈ [0, 1]`) that drives confidence calibration and re-planning decisions.
5. A structured execution trace with typed events, timestamps, and per-step reasoning.

**Positioning relative to other repos in this portfolio:**

| Repo | Domain | Core Demonstration |
|---|---|---|
| `analyst-agent` | Bounded analytics | Routing, validation, retry, observability |
| `FixFriend` | Structured product AI | Slot filling, safety, product UX |
| **`open-ended-task-agent`** | **Higher-agency tasks** | **Planning, acting, reflecting, terminating** |

---

## 2. Background and Related Work

### 2.1 ReAct: Synergizing Reasoning and Acting

Yao et al. (2022) introduced ReAct, showing that interleaving a "Thought" (chain-of-thought reasoning) with an "Action" (tool call) and "Observation" (result) outperforms either alone on knowledge-intensive tasks. OETA implements this pattern explicitly: before each tool call the agent generates a `REASONING` trace event with a natural-language justification; after each result an `OBSERVATION_RECORDED` event captures structured evidence.

*Reference:* Yao, S., Zhao, J., Yu, D., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023. arXiv:2210.03629.

### 2.2 Reflexion: Verbal Reinforcement Learning

Shinn et al. (2023) proposed Reflexion, in which an agent reflects on its own trajectory to produce a verbal self-evaluation that informs future attempts. OETA applies a single-episode variant: after each non-summarization step the `reflect()` function produces a `ReflectionNote` containing an assessment sentence, a list of evidence gaps, a signed confidence delta, and a replanning recommendation. This drives the `evidence_quality` score and triggers corrective search steps when gaps are detected.

*Reference:* Shinn, N., Labash, B., Gopinath, A. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023. arXiv:2303.11366.

### 2.3 MRKL: Modular Reasoning, Knowledge and Language

Karpas et al. (2022) described MRKL systems, in which an LLM router dispatches queries to specialized modules (calculators, search engines, databases). OETA's `TOOL_REGISTRY` is a compact MRKL-style module registry: four named tools (`local_search`, `compare_items`, `summarize_findings`, `calculator`) each with a typed spec and independent implementation.

*Reference:* Karpas, E., Abend, O., Belinkov, Y., et al. "MRKL Systems: A Modular, Neuro-Symbolic Architecture that Combines Language Models, External Knowledge Sources and Discrete Reasoning." arXiv:2205.00445.

### 2.4 Retrieval-Augmented Generation (RAG)

Lewis et al. (2020) formalized retrieval-augmented generation, showing that grounding LLM outputs in retrieved documents dramatically reduces hallucination. OETA uses a local TF-IDF knowledge graph as the retrieval layer, with sentence-level snippet extraction to maximize information density per context token.

*Reference:* Lewis, P., Perez, E., Piktus, A., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020. arXiv:2005.11401.

### 2.5 Self-RAG: Adaptive Retrieval and Critique

Asai et al. (2023) proposed Self-RAG, in which a model learns to retrieve on demand and to critique retrieved passages for relevance. OETA implements a lightweight analogue: the `reflect()` function assesses retrieval usefulness after each search step, and re-planning is only triggered when the reflection signals a gap and the replan budget has not been exhausted.

*Reference:* Asai, A., Wu, Z., Wang, B., et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." arXiv:2310.11511.

---

## 3. System Architecture

### 3.1 Formal Problem Statement

Let `G` denote a natural-language goal, `K` a local knowledge corpus of documents `{d_1, …, d_n}`, and `T` a set of tools `{local_search, compare_items, summarize_findings, calculator}`. The agent must produce a final answer `A` along with a confidence score `q ∈ [0, 1]` and a termination reason `r ∈ R` where `R = {SUCCESS, PARTIAL, FAILED, MAX_STEPS}`.

Formally, the agent seeks:

```
A*, q*, r* = argmax_{A, q} P(A correctly answers G | evidence E ⊆ K)
             subject to: step_count ≤ max_steps
                         failures ≤ failure_budget
                         no_progress_count ≤ no_progress_budget
```

### 3.2 State Space

The complete execution state `S` is a `AgentState` dataclass:

```
S = {
  goal:               str                    # original goal
  plan:               AgentPlan              # decomposed action plan
  pending_steps:      List[PlanStep]         # remaining actions
  completed_steps:    List[PlanStep]         # executed actions
  observations:       List[Observation]      # retrieved evidence
  reflections:        List[ReflectionNote]   # post-step assessments
  tool_calls:         List[ToolCall]         # execution log
  reasoning_log:      List[str]              # Thought strings (ReAct)
  evidence_quality:   float ∈ [0, 1]        # cumulative quality score
  confidence:         {low, medium, high}    # calibrated label
  step_count:         int                    # steps consumed
  failures:           int                    # consecutive failures
  replan_count:       int                    # replanning events triggered
  sufficiency_status: {unknown, sufficient, insufficient}
  unsupported:        bool                   # goal outside corpus scope
  final_answer:       Optional[str]          # answer string
  trace:              List[TraceEvent]       # structured audit log
}
```

### 3.3 Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AGENT LOOP (agent.py)                       │
│                                                                     │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌────────────┐  │
│  │ Planner  │ →  │   Tool     │ →  │  Tool    │ →  │ Reflection │  │
│  │(planner) │    │ Selector   │    │ Executor │    │(reflection)│  │
│  └──────────┘    │(tool_sele- │    │(tools.py)│    └────────────┘  │
│                  │ ctor.py)   │    └──────────┘           │        │
│                  └────────────┘                           ↓        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 AgentState (state.py)                        │  │
│  │  observations · reflections · trace · evidence_quality       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                         │
│                  ┌─────────────────┐                               │
│                  │  Termination    │                               │
│                  │ (termination.py)│                               │
│                  └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
         ↑                                        ↓
   KnowledgeGraph                            Final Answer
   (knowledge.py)                       + Confidence + Trace
```

---

## 4. The Agent Loop

### 4.1 Algorithm

```
OETA(goal G, max_steps M):
  corpus  ← load_corpus()
  graph   ← KnowledgeGraph.from_documents(corpus)   # TF-IDF index
  plan    ← Planner.generate_plan(G, graph.keys())  # goal decomposition
  S       ← AgentState(G, plan)
  emit GOAL_RECEIVED, PLAN_CREATED

  while S.step_count < M and not terminate(S):
    step ← choose_next_action(S)           # prerequisite-safe selection
    S.step_count += 1

    thought ← Reason(step, S)             # ReAct: Thought
    emit REASONING(thought)

    result ← Execute(step, graph, S)      # ReAct: Act
    emit TOOL_CALLED, OBSERVATION_RECORDED

    if step.action_type ≠ summarize:
      note ← Reflect(S, result)           # Reflexion
      S.evidence_quality += note.Δq
      emit REFLECTION_LOGGED
      if note.should_replan and S.replan_count < REPLAN_LIMIT:
        insert corrective search step      # gap-driven replan
        emit REPLAN_TRIGGERED

    update_progress_counters(S, result)
    check_terminate(S, M)

  if not S.final_answer:
    S.final_answer ← summarize_findings(G, S.observations)

  emit TERMINATED_{SUCCESS|PARTIAL|FAILED|MAX_STEPS}
  return S
```

### 4.2 Trace Events

Every state transition emits a structured trace event:

| Event | Trigger |
|---|---|
| `GOAL_RECEIVED` | Agent initialized with a goal |
| `PLAN_CREATED` | Planner produced a step sequence |
| `REASONING` | Thought generated before each action |
| `ACTION_SELECTED` | Next step chosen (or corrective step inserted) |
| `TOOL_CALLED` | Tool executed; success/failure recorded |
| `OBSERVATION_RECORDED` | Evidence saved to state.observations |
| `REFLECTION_LOGGED` | Post-step quality assessment completed |
| `EVIDENCE_GAP_DETECTED` | Reflection identified missing evidence |
| `REPLAN_TRIGGERED` | Corrective step inserted into pending queue |
| `CONFIDENCE_UPDATED` | evidence_quality crossed a threshold |
| `TERMINATED_SUCCESS` | Answer marked sufficient |
| `TERMINATED_PARTIAL` | Step/failure budget exhausted, partial answer |
| `TERMINATED_FAILED` | Goal unsupported or no evidence found |
| `TERMINATED_MAX_STEPS` | Step count reached max_steps |

Each event includes: `step`, `event_type`, `tool_name`, `reason`, `metadata`, `elapsed_s`.

---

## 5. Planning Module

The `Planner` decomposes the user goal into an ordered `AgentPlan` containing 2–4 `PlanStep` instances. Two strategies are supported:

**LLM-guided planning** uses a ReAct-style chain-of-thought prompt that first asks the model to reason about what information is needed, then produces a numbered action plan constrained to the registered tool names. Output is parsed by extracting numbered lines and matching against `{search, compare, summarize, calculate}`. If parsing yields fewer than 2 valid steps, the system falls back to deterministic planning.

**Deterministic fallback** uses regular-expression pattern matching to classify goals into three categories:
- *Comparison* (`compare … and/vs …`): produces `search → search → compare → summarize`
- *Study plan* (`learn`, `study plan`, `basics`): produces `search → summarize`
- *Generic*: produces `search → summarize`

The deterministic path executes in constant time with no external calls, ensuring the system is always demonstrable regardless of API availability.

---

## 6. Retrieval and Evidence Accumulation

### 6.1 TF-IDF Knowledge Graph

The `KnowledgeGraph` builds a TF-IDF inverted index over the local document corpus at initialization time:

```
TF(t, d)  = count(t, d) / |d|
IDF(t)    = log((N + 1) / (df(t) + 1)) + 1   # Laplace-smoothed
Score(q, d) = Σ_{t ∈ q} TF(t, d) · IDF(t)
```

This replaces the original keyword-count approach with a standard retrieval model. Rare terms that appear in few documents receive higher IDF weights, improving discrimination between topically similar documents.

### 6.2 Sentence-Level Snippet Extraction

Rather than returning fixed-length character windows, the search method selects the top-N sentences by query-term overlap (preserving their original document order). This produces coherent, information-dense snippets that contain the actual answer-relevant content rather than arbitrary character prefixes.

### 6.3 Document Adjacency (Graph Edges)

Beyond retrieval, the `KnowledgeGraph` maintains an adjacency structure: documents that share many vocabulary terms are connected as neighbors. This enables future extensions such as graph traversal for multi-hop reasoning without architectural changes.

---

## 7. Tool Registry and Selection

### 7.1 Tool Specification

```python
@dataclass
class ToolSpec:
    name: str           # human-readable name
    description: str    # for LLM-based tool selection (future)
    function: Callable  # pure function: inputs → ToolResult
```

The `TOOL_REGISTRY` maps logical action names to `ToolSpec` instances. Adding a new tool requires only: implementing the function, adding it to the registry. The agent loop does not need modification.

### 7.2 State-Conditional Selection

The `tool_selector` enforces two invariants before dispatching a step:

1. A `compare` step must have at least one prior observation. If not, a corrective `search` step is inserted.
2. A `summarize` step must have at least one prior observation. If not, a corrective `search` step is inserted.

This makes the planner optimistic (it can generate compare-before-search plans) while the selector silently repairs ordering violations based on actual runtime state.

### 7.3 Multi-Criteria Comparison Matrix

The `compare_items` tool scores each candidate against ten criteria derived from corpus text:
`performance`, `ease of use`, `ecosystem`, `scalability`, `portability`, `documentation`, `production`, `beginner`, `async`, `analytics`.

For each `(candidate, criterion)` pair it counts supporting observation sentences. Results are rendered as a markdown table, providing a structured, auditable comparison rather than a free-form narrative.

---

## 8. Reflection and Re-planning

After each non-summarization step, `reflect(state, result)` is called. It produces a `ReflectionNote`:

```python
@dataclass
class ReflectionNote:
    step:               int          # which step triggered this
    assessment:         str          # one-sentence evaluation
    evidence_gaps:      List[str]    # what is still missing
    confidence_delta:   float        # Δ to apply to evidence_quality
    should_replan:      bool         # insert corrective step?
    evidence_quality:   float        # updated quality score
```

The `evidence_quality` score evolves over the episode:
- Successful retrieval from a new source: `+0.15`
- Gap detected: `−0.05 × gap_count`
- Failed tool call: `−0.12`
- Evidence quality is clamped to `[0, 1]`

Re-planning is triggered when `should_replan=True` AND `replan_count < REPLAN_LIMIT` AND the last result succeeded. This prevents re-planning from becoming an infinite loop: the budget is explicit and logged.

---

## 9. Termination Logic

The agent evaluates five termination conditions in priority order after every step:

| Priority | Condition | Trace Event |
|---|---|---|
| 1 | `sufficiency_status == "sufficient"` | `TERMINATED_SUCCESS` |
| 2 | `unsupported == True` | `TERMINATED_FAILED` |
| 3 | `step_count >= max_steps` | `TERMINATED_MAX_STEPS` |
| 4 | `failures >= 3` | `TERMINATED_PARTIAL` |
| 5 | `no_progress_count >= 2` | `TERMINATED_PARTIAL` |

The final response always includes: an answer (possibly partial), a confidence label (`low/medium/high`), the numeric `evidence_quality` score, evidence observations with source citations, the full trace, and the termination summary.

---

## 10. Evaluation

### 10.1 Test Suite

The test suite covers 7 categories across 8 files:

| File | Coverage |
|---|---|
| `test_agent_loop.py` | Full loop: success, replan, unsupported, max-steps, trace structure |
| `test_tools.py` | Search, compare matrix, summarize citations, calculator |
| `test_knowledge.py` | TF-IDF scoring, snippet quality, adjacency, empty/unknown queries |
| `test_planner.py` | LLM plan parsing, fallback patterns, stopping criteria |
| `test_tool_selector.py` | Prerequisite guards, corrective insertion, trace events |
| `test_termination.py` | All five conditions, priority ordering, termination_summary |
| `test_reflection.py` | Quality scoring, gap detection, replan signalling, bounds |
| `test_trace.py` | Event structure, elapsed_s, step sequence, formatting |

Run the suite:

```bash
pip install -e .[dev]
pytest -v
```

### 10.2 Evaluation Harness

```bash
python scripts/run_eval.py          # summary output
python scripts/run_eval.py --verbose  # per-scenario detail
```

Scenarios: comparison tasks, study-plan tasks, unsupported goals, budget-limited runs, and partial-evidence cases. The harness reports: pass/fail per scenario, average evidence quality, average steps, and average elapsed time.

---

## 11. Example Runs

### CLI

```bash
python -m task_agent "Compare DuckDB and SQLite for a local analytics app"
python -m task_agent "Compare LangGraph and AutoGen for a small agent project"
python -m task_agent "Create a study plan for learning ROS2 basics in 2 weeks"
python -m task_agent --steps 5 "Compare FastAPI and Flask"
python -m task_agent --no-trace "What Python web framework is best for beginners?"
```

### Sample Output (deterministic mode, no API key)

```
────────────────────────────────────────────────────────────────────────
  OPEN-ENDED TASK AGENT
────────────────────────────────────────────────────────────────────────

  Goal: Compare DuckDB and SQLite for a local analytics app

  Confidence : MEDIUM  ████████████░░░░░░░░  58%
  Status     : sufficient
  Sources    : 2 unique
  Elapsed    : 0.041s

  Goal: Compare DuckDB and SQLite for a local analytics app

  Evidence summary:
  - [duckdb_vs_sqlite] DuckDB uses columnar storage and vectorized query
    execution, processing data in batches using SIMD instructions…
  - [comparison] Evidence Matrix: | Criterion | DuckDB | SQLite |…

  Recommended: DuckDB (medium confidence).
  Confidence: moderate — evidence is present but limited in scope.
  Sources consulted: duckdb_vs_sqlite, comparison.
```

### FastAPI Server

```bash
pip install -e .[web]
uvicorn task_agent.api:app --reload
```

```bash
curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "Compare LangGraph and AutoGen", "debug": true, "max_steps": 8}' \
  | python -m json.tool
```

Response includes: `final_answer`, `confidence`, `evidence_quality`, `status`, `evidence`, `reflections`, `stats`, `trace`.

---

## 12. Failure Analysis

| Failure Mode | Detection | Response |
|---|---|---|
| Goal outside corpus scope | Empty search after REPLAN_LIMIT exhausted | `unsupported=True` → `TERMINATED_FAILED` |
| Repeated tool failures | `failures >= FAILURE_BUDGET` | `TERMINATED_PARTIAL` + partial answer |
| No forward progress | `no_progress_count >= 2` | `TERMINATED_PARTIAL` |
| Max steps reached | `step_count >= max_steps` | `TERMINATED_MAX_STEPS` + summarize available evidence |
| Compare with no evidence | `tool_selector` prerequisite guard | Insert corrective search, log `ACTION_SELECTED` |
| LLM plan parse failure | `len(steps) < 2` | Fall through to deterministic planner |
| LLM API unavailable | `is_openai_available() == False` | All modules use deterministic fallbacks |

---

## 13. Setup

```bash
git clone https://github.com/paullimale/open-ended-task-agent
cd open-ended-task-agent
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -e .[dev]          # core + tests
pip install -e .[dev,llm]      # + OpenAI integration
pip install -e .[dev,llm,web]  # + FastAPI server
```

Create `.env` with your OpenAI API key (optional — the system is fully functional without it):

```
OPENAI_API_KEY=sk-...
```

Run tests:

```bash
pytest -v
```

Run evaluation harness:

```bash
python scripts/run_eval.py --verbose
```

---

## 14. Scope and Non-Goals

This repository is a compact demonstration of higher-agency task execution. It is intentionally bounded:

**In scope:**
- Explicit, auditable planning and tool selection
- Evidence-quality-driven re-planning
- Principled termination with typed events
- Deterministic operation without API keys
- Readable, testable, interview-demonstrable code

**Out of scope (by design):**
- Multi-agent coordination or role-based conversation
- Large memory systems or persistent vector databases
- Browser automation or live web retrieval
- Production-grade infrastructure, authentication, or scaling
- Autonomous code execution or OS-level actions
- Comparison with frontier models on open benchmarks

---

## 15. Limitations and Future Work

**Current limitations:**
- The local corpus is synthetic and small (5 documents). Real deployment would require a vector store (FAISS, Chroma, Weaviate) with dense retrieval.
- The evidence matrix uses term-overlap heuristics. A fine-tuned NLI model would produce higher-quality criterion matching.
- Reflection quality degrades when no LLM is available; the rule-based fallback uses observation counts as a proxy for evidence quality.
- The confidence score is calibrated heuristically, not via a proper calibration procedure.

**Future directions:**
- Replace TF-IDF with dense retrieval (sentence-transformers, all-MiniLM-L6-v2)
- Add a proper NLI-based relevance scorer for Self-RAG-style filtering
- Implement multi-hop reasoning via knowledge graph traversal
- Add episodic memory so the agent can incorporate information from prior runs
- Expose per-step streaming via Server-Sent Events on the FastAPI endpoint

---

## References

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y. (2022). **ReAct: Synergizing Reasoning and Acting in Language Models.** *ICLR 2023.* arXiv:2210.03629.
2. Shinn, N., Labash, B., Gopinath, A. (2023). **Reflexion: Language Agents with Verbal Reinforcement Learning.** *NeurIPS 2023.* arXiv:2303.11366.
3. Karpas, E., Abend, O., Belinkov, Y., et al. (2022). **MRKL Systems: A Modular, Neuro-Symbolic Architecture.** arXiv:2205.00445.
4. Lewis, P., Perez, E., Piktus, A., et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** *NeurIPS 2020.* arXiv:2005.11401.
5. Asai, A., Wu, Z., Wang, B., et al. (2023). **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.** arXiv:2310.11511.
6. Wei, J., Wang, X., Schuurmans, D., et al. (2022). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.** *NeurIPS 2022.* arXiv:2201.11903.
7. Schick, T., Dwivedi-Yu, J., Dessì, R., et al. (2023). **Toolformer: Language Models Can Teach Themselves to Use Tools.** *NeurIPS 2023.* arXiv:2302.04761.
