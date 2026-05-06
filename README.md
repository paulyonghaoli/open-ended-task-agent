# Open-Ended Task Agent (OETA)

**A Compact Implementation of ReAct–Reflexion Planning for Higher-Agency Task Execution Under Uncertainty**

> This repository is a deliberately compact, interview-friendly demonstration of higher-agency agent architecture. It is not a production framework. The focus is architectural clarity: explicit planning, traceable reasoning, bounded autonomy, and structured termination.

---

## Abstract

**OETA** (Open-Ended Task Agent) is a compact implementation of a plan–act–observe–reflect loop for open-ended question answering over a local knowledge corpus. It integrates three research-grounded patterns — **ReAct** (interleaved reasoning and acting), **Reflexion** (post-observation self-assessment), and **retrieval-augmented generation** (evidence-grounded answers) — into a single, readable codebase with no mandatory external dependencies. The goal is not to maximize capability, but to demonstrate a bounded, debuggable agent architecture that can be reasoned about in production settings. The system decomposes a user goal into a structured plan, selects tools based on current evidence state, executes retrieval and comparison actions, reflects on observation quality after each step, and terminates under explicit conditions with a confidence-annotated final answer. All execution state, reasoning steps, reflections, and termination events are captured in a structured trace.

---

## 1. Introduction

Building agents that reliably accomplish open-ended goals requires solving three coupled problems simultaneously: *what to do next* (planning), *whether the result was useful* (observation assessment), and *when to stop* (safe termination). Most agent frameworks delegate these decisions entirely to an LLM, which produces agents that are brittle under distribution shift, unauditable, and prone to looping or hallucinating when evidence is sparse.

OETA approaches these problems differently. Planning is explicit and can be inspected before execution. Observation assessment is structured — a `ReflectionNote` with a numeric quality delta and gap inventory. Termination is rule-based with five distinct conditions, each producing a named trace event. When an LLM is available it augments each of these with natural-language reasoning; when it is not, the system falls back to deterministic heuristics that maintain the same behavioral contract.

What this repo demonstrates in under 800 lines of Python:

- A readable, test-covered implementation of ReAct+Reflexion with zero mandatory dependencies
- A TF-IDF retrieval layer that replaces keyword counting with principled document ranking
- A multi-criteria evidence matrix for structured comparison tasks
- A numeric `evidence_quality` score (`∈ [0, 1]`) that drives confidence calibration and re-planning decisions
- A structured execution trace with typed events, timestamps, and per-step reasoning

**Positioning relative to other repos in this portfolio:**

| Repo | Domain | Core Demonstration |
|---|---|---|
| `analyst-agent` | Bounded analytics | Routing, validation, retry, observability |
| `FixFriend` | Structured product AI | Slot filling, safety, product UX |
| **`open-ended-task-agent`** | **Higher-agency tasks** | **Planning, acting, reflecting, terminating** |

---

## 2. System Architecture

### 2.1 Formal Problem Statement

Let `G` denote a natural-language goal, `K` a local knowledge corpus of documents `{d_1, …, d_n}`, and `T` a set of tools `{local_search, compare_items, summarize_findings, calculator}`. The agent must produce a final answer `A` along with a confidence score `q ∈ [0, 1]` and a termination reason `r ∈ R` where `R = {SUCCESS, PARTIAL, FAILED, MAX_STEPS}`.

Formally, the agent seeks:

```
A*, q*, r* = argmax_{A, q} P(A correctly answers G | evidence E ⊆ K)
             subject to: step_count ≤ max_steps
                         failures ≤ failure_budget
                         no_progress_count ≤ no_progress_budget
```

### 2.2 State Space

The complete execution state `S` is an `AgentState` dataclass:

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

### 2.3 Component Overview

```
+---------------------------------------------------------------------+
|                         AGENT LOOP (agent.py)                       |
|                                                                     |
|  +----------+    +------------+    +----------+    +------------+  |
|  | Planner  | -> |   Tool     | -> |  Tool    | -> | Reflection |  |
|  |(planner) |    | Selector   |    | Executor |    |(reflection)|  |
|  +----------+    |(tool_sele- |    |(tools.py)|    +------------+  |
|                  | ctor.py)   |    +----------+           |        |
|                  +------------+                           v        |
|  +----------------------------------------------------------+      |
|  |                 AgentState (state.py)                    |      |
|  |  observations . reflections . trace . evidence_quality   |      |
|  +----------------------------------------------------------+      |
|                           |                                         |
|                  +-----------------+                               |
|                  |  Termination    |                               |
|                  | (termination.py)|                               |
|                  +-----------------+                               |
+---------------------------------------------------------------------+
         ^                                        v
   KnowledgeGraph                            Final Answer
   (knowledge.py)                       + Confidence + Trace
```

---

## 3. Engineering Interpretation

This section explains why each research pattern was implemented the way it was — the practical tradeoffs behind the design choices.

### ReAct: bounded and traceable, not open-ended

The paper's version of ReAct runs in an LLM-generated loop with no explicit step budget or termination contract. This implementation constrains the loop: `max_steps` is a hard parameter (default 10), and every Thought is a named `REASONING` trace event rather than a string concatenated into a prompt. This makes the reasoning log inspectable after the fact and gives operators a clear knob for controlling execution cost.

### Reflexion: structured note, not vague self-critique

Shinn et al. use natural-language verbal feedback accumulated across episodes. Here, a single-episode variant runs after every non-summarization step and produces a typed `ReflectionNote` — not a blob of text. The note has explicit fields: `evidence_gaps` (a list of strings), `confidence_delta` (a signed float), and `should_replan` (a bool). This makes the reflection machine-readable: the agent loop can branch on `should_replan` without parsing prose, and tests can assert on specific fields.

### TF-IDF over local corpus: determinism and testability

A vector database (FAISS, Chroma) would offer better semantic recall. TF-IDF was chosen because it is deterministic, requires no GPU or embedding API, and produces reproducible scores that make tests meaningful. The retrieval layer can be swapped without touching the agent loop — `search_local_corpus` is a pure function and the `KnowledgeGraph` interface is stable.

### Evidence quality as a first-class float

Rather than letting the LLM assert whether it has enough information (which it cannot reliably do), `evidence_quality` is maintained as a numeric score that the agent loop can read deterministically. Successful retrievals increment it (+0.15), detected gaps decrement it (−0.05 × gap count), and failed tool calls penalize it (−0.12). Confidence labels (`low/medium/high`) are derived from thresholds on this float. This means confidence is checkable in tests and observable in logs.

---

## 4. Design Tradeoffs

| Decision | What was chosen | What was rejected | Why |
|---|---|---|---|
| Retrieval | TF-IDF over local corpus | Vector DB (FAISS/Chroma) | Deterministic, zero infra, reproducible in tests |
| Loop bound | Hard `max_steps` budget | Open-ended LLM loop | Prevents runaway cost; operators can tune |
| Evidence quality | Numeric float updated by rules | LLM self-assessment | Machine-readable, testable, not subject to hallucination |
| Reflection output | Typed `ReflectionNote` dataclass | Free-text verbal feedback | Branching logic can read fields without parsing |
| Termination | 5 explicit rule-based conditions | Threshold on LLM confidence string | Auditable, consistent, observable in trace |
| Planning fallback | Regex pattern matching | Always calling OpenAI | System is fully functional with no API key |
| Comparison | Multi-criteria evidence matrix | Free-form LLM narrative | Structured, auditable, extensible to new criteria |

The recurring theme: prefer decisions that keep behavior observable and testable over decisions that maximize single-run output quality.

---

## 5. The Agent Loop

### 5.1 Algorithm

```
OETA(goal G, max_steps M):
  corpus  <- load_corpus()
  graph   <- KnowledgeGraph.from_documents(corpus)   # TF-IDF index
  plan    <- Planner.generate_plan(G, graph.keys())  # goal decomposition
  S       <- AgentState(G, plan)
  emit GOAL_RECEIVED, PLAN_CREATED

  while S.step_count < M and not terminate(S):
    step <- choose_next_action(S)           # prerequisite-safe selection
    S.step_count += 1

    thought <- Reason(step, S)             # ReAct: Thought
    emit REASONING(thought)

    result <- Execute(step, graph, S)      # ReAct: Act
    emit TOOL_CALLED, OBSERVATION_RECORDED

    if step.action_type != summarize:
      note <- Reflect(S, result)           # Reflexion
      S.evidence_quality += note.delta_q
      emit REFLECTION_LOGGED
      if note.should_replan and S.replan_count < REPLAN_LIMIT:
        insert corrective search step      # gap-driven replan
        emit REPLAN_TRIGGERED

    update_progress_counters(S, result)
    check_terminate(S, M)

  if not S.final_answer:
    S.final_answer <- summarize_findings(G, S.observations)

  emit TERMINATED_{SUCCESS|PARTIAL|FAILED|MAX_STEPS}
  return S
```

### 5.2 Trace Events

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

## 6. Planning Module

The `Planner` decomposes the user goal into an ordered `AgentPlan` containing 2–4 `PlanStep` instances. Two strategies are supported:

**LLM-guided planning** uses a ReAct-style chain-of-thought prompt that first asks the model to reason about what information is needed, then produces a numbered action plan constrained to the registered tool names. Output is parsed by extracting numbered lines and matching against `{search, compare, summarize, calculate}`. If parsing yields fewer than 2 valid steps, the system falls back to deterministic planning.

**Deterministic fallback** uses regular-expression pattern matching to classify goals into three categories:
- *Comparison* (`compare … and/vs …`): produces `search → search → compare → summarize`
- *Study plan* (`learn`, `study plan`, `basics`): produces `search → summarize`
- *Generic*: produces `search → summarize`

The deterministic path executes in constant time with no external calls, ensuring the system is always demonstrable regardless of API availability.

---

## 7. Retrieval and Evidence Accumulation

### 7.1 TF-IDF Knowledge Graph

The `KnowledgeGraph` builds a TF-IDF inverted index over the local document corpus at initialization time:

```
TF(t, d)  = count(t, d) / |d|
IDF(t)    = log((N + 1) / (df(t) + 1)) + 1   # Laplace-smoothed
Score(q, d) = sum_{t in q} TF(t, d) * IDF(t)
```

This replaces the original keyword-count approach with a standard retrieval model. Rare terms that appear in few documents receive higher IDF weights, improving discrimination between topically similar documents. A minimum score threshold (`MIN_TFIDF_SCORE = 0.015`) filters spurious matches from common query terms that happen to overlap incidentally.

### 7.2 Sentence-Level Snippet Extraction

Rather than returning fixed-length character windows, the search method selects the top-N sentences by query-term overlap (preserving their original document order). This produces coherent, information-dense snippets that contain the actual answer-relevant content rather than arbitrary character prefixes.

### 7.3 Document Adjacency (Graph Edges)

Beyond retrieval, the `KnowledgeGraph` maintains an adjacency structure: documents that share many vocabulary terms are connected as neighbors. This enables future extensions such as graph traversal for multi-hop reasoning without architectural changes.

---

## 8. Tool Registry and Selection

### 8.1 Tool Specification

```python
@dataclass
class ToolSpec:
    name: str           # human-readable name
    description: str    # for LLM-based tool selection (future)
    function: Callable  # pure function: inputs -> ToolResult
```

The `TOOL_REGISTRY` maps logical action names to `ToolSpec` instances. Adding a new tool requires only: implementing the function, adding it to the registry. The agent loop does not need modification.

### 8.2 State-Conditional Selection

The `tool_selector` enforces two invariants before dispatching a step:

1. A `compare` step must have at least one prior observation. If not, a corrective `search` step is inserted.
2. A `summarize` step must have at least one prior observation. If not, a corrective `search` step is inserted.

This makes the planner optimistic (it can generate compare-before-search plans) while the selector silently repairs ordering violations based on actual runtime state.

### 8.3 Multi-Criteria Comparison Matrix

The `compare_items` tool scores each candidate against ten criteria derived from corpus text:
`performance`, `ease of use`, `ecosystem`, `scalability`, `portability`, `documentation`, `production`, `beginner`, `async`, `analytics`.

For each `(candidate, criterion)` pair it counts supporting observation sentences. Results are rendered as a markdown table, providing a structured, auditable comparison rather than a free-form narrative.

---

## 9. Reflection and Re-planning

After each non-summarization step, `reflect(state, result)` is called. It produces a `ReflectionNote`:

```python
@dataclass
class ReflectionNote:
    step:               int          # which step triggered this
    assessment:         str          # one-sentence evaluation
    evidence_gaps:      List[str]    # what is still missing
    confidence_delta:   float        # delta to apply to evidence_quality
    should_replan:      bool         # insert corrective step?
    evidence_quality:   float        # updated quality score
```

The `evidence_quality` score evolves over the episode:
- Successful retrieval from a new source: `+0.15`
- Gap detected: `-0.05 * gap_count`
- Failed tool call: `-0.12`
- Evidence quality is clamped to `[0, 1]`

Re-planning is triggered when `should_replan=True` AND `replan_count < REPLAN_LIMIT` AND the last result succeeded AND no search steps are already pending. This prevents re-planning from becoming an infinite loop: the budget is explicit and logged.

---

## 10. Termination Logic

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

## 11. Failure Walkthroughs

These two scenarios illustrate how the agent handles realistic failure modes, step by step.

### Walkthrough A: Comparison with a gap-driven replan

**Goal:** `"Compare LangGraph and AutoGen for a small agent project"`

The planner generates: `search("LangGraph") → search("AutoGen") → compare → summarize`.

Step 1 — search "LangGraph": returns 3 passages from `langgraph_vs_autogen.md`. Reflection notes strong coverage of orchestration patterns. `evidence_quality` → 0.15. No gaps flagged; no replan.

Step 2 — search "AutoGen": returns 2 passages. Reflection detects a gap: *"AutoGen multi-agent configuration details missing."* `evidence_quality` → 0.25 (new source +0.15, gap -0.05). `should_replan=True`.

Replan guard check: `replan_count=0 < REPLAN_LIMIT=2`, last result succeeded, no pending searches. A corrective search step is inserted for `state.goal` (the full goal, not the gap text). `REPLAN_TRIGGERED` logged.

Step 3 — corrective search: returns 2 more passages from a different section of the document. `evidence_quality` → 0.40.

Step 4 — compare: runs evidence matrix against 10 criteria. `OBSERVATION_RECORDED` with comparison table.

Step 5 — summarize: produces final answer with citations. `sufficiency_status = "sufficient"`. `TERMINATED_SUCCESS`.

**Total: 5 steps, 1 replan, 7 observations, confidence: medium.**

---

### Walkthrough B: Out-of-scope goal that terminates cleanly

**Goal:** `"Write a sonnet about the Schrödinger equation and quantum entanglement"`

The planner generates: `search("Schrödinger equation quantum entanglement") → summarize`.

Step 1 — search: query tokens `["schrodinger", "quantum", "entanglement"]` produce zero TF-IDF hits above the `MIN_TFIDF_SCORE=0.015` threshold across all 5 corpus documents. Replan budget allows one refinement: a corrective search for `"schrodinger equation quantum entanglement overview basics"` is inserted. `REPLAN_TRIGGERED` logged, `replan_count=1`.

Step 2 — refined search: still zero hits above threshold. `replan_count >= REPLAN_LIMIT`. `state.unsupported = True`.

Termination check: `unsupported=True` matches condition 2 (priority 2). `TERMINATED_FAILED` logged.

Fallback answer: *"The agent could not produce an answer. The goal may be outside the scope of the local knowledge corpus."*

**Total: 2 steps, 1 replan, 0 observations, confidence: low.**

The trace shows exactly what happened at each step. No hallucinated answer, no silent failure.

---

## 12. When This System Fails

This section is a candid inventory of conditions where OETA will produce degraded or wrong output — not failures of implementation, but limits of the design.

**Corpus gaps produce confident-sounding partial answers.** If a goal partially overlaps the corpus (e.g., "Compare DuckDB and PostgreSQL"), the agent will find DuckDB documents, build a comparison matrix, and produce an answer — but with `confidence: low` and only one side of the comparison populated. A user who doesn't read the confidence label will receive a misleading result.

**Rule-based reflection cannot detect semantic gaps.** The `ReflectionNote` uses observation count and source diversity as proxies for evidence quality. If the corpus returns 4 passages that are all about the same narrow subtopic, reflection will not flag this — the gap is semantic, not structural.

**Deterministic comparison matrix uses term overlap, not meaning.** A document that mentions "performance" in any context will be counted as evidence for the performance criterion. Negations ("DuckDB does not support X") are treated the same as affirmations.

**The replan budget (2) can be exhausted by unrelated failures.** If two replans fire early for search refinements, a later genuine gap will not trigger a replan — the budget is already gone.

**Max-steps termination does not guarantee a usable answer.** If `max_steps=2` is passed and the plan has 4 steps, `TERMINATED_MAX_STEPS` fires after step 2. Whatever evidence exists at that point becomes the answer, even if it covers only one of two comparison candidates.

These failure modes are observable in the trace. An operator who reads `TERMINATED_PARTIAL` with `evidence_quality < 0.3` should treat the answer as a starting point, not a conclusion.

---

## 13. Evaluation

### 13.1 Test Suite

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

### 13.2 Evaluation Harness

```bash
python scripts/run_eval.py          # summary output
python scripts/run_eval.py --verbose  # per-scenario detail
```

Scenarios: comparison tasks, study-plan tasks, unsupported goals, budget-limited runs, and partial-evidence cases. The harness reports: pass/fail per scenario, average evidence quality, average steps, and average elapsed time.

---

## 14. Example Runs

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
------------------------------------------------------------------------
  OPEN-ENDED TASK AGENT
------------------------------------------------------------------------

  Goal: Compare DuckDB and SQLite for a local analytics app

  Confidence : MEDIUM  ########............  58%
  Status     : sufficient
  Sources    : 2 unique
  Elapsed    : 0.041s

  Evidence summary:
  - [duckdb_vs_sqlite] DuckDB uses columnar storage and vectorized query
    execution, processing data in batches using SIMD instructions...
  - [comparison] Evidence Matrix: | Criterion | DuckDB | SQLite |...

  Recommended: DuckDB (medium confidence).
  Confidence: moderate -- evidence is present but limited in scope.
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

## 15. Setup

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

## 16. Scope and Non-Goals

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

## 17. Limitations and Future Work

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

## 18. Background and Related Work

### 18.1 ReAct: Synergizing Reasoning and Acting

Yao et al. (2022) introduced ReAct, showing that interleaving a "Thought" (chain-of-thought reasoning) with an "Action" (tool call) and "Observation" (result) outperforms either alone on knowledge-intensive tasks. OETA implements this pattern explicitly: before each tool call the agent generates a `REASONING` trace event with a natural-language justification; after each result an `OBSERVATION_RECORDED` event captures structured evidence.

*Reference:* Yao, S., Zhao, J., Yu, D., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023. arXiv:2210.03629.

### 18.2 Reflexion: Verbal Reinforcement Learning

Shinn et al. (2023) proposed Reflexion, in which an agent reflects on its own trajectory to produce a verbal self-evaluation that informs future attempts. OETA applies a single-episode variant: after each non-summarization step the `reflect()` function produces a `ReflectionNote` containing an assessment sentence, a list of evidence gaps, a signed confidence delta, and a replanning recommendation. This drives the `evidence_quality` score and triggers corrective search steps when gaps are detected.

*Reference:* Shinn, N., Labash, B., Gopinath, A. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023. arXiv:2303.11366.

### 18.3 MRKL: Modular Reasoning, Knowledge and Language

Karpas et al. (2022) described MRKL systems, in which an LLM router dispatches queries to specialized modules (calculators, search engines, databases). OETA's `TOOL_REGISTRY` is a compact MRKL-style module registry: four named tools (`local_search`, `compare_items`, `summarize_findings`, `calculator`) each with a typed spec and independent implementation.

*Reference:* Karpas, E., Abend, O., Belinkov, Y., et al. "MRKL Systems: A Modular, Neuro-Symbolic Architecture that Combines Language Models, External Knowledge Sources and Discrete Reasoning." arXiv:2205.00445.

### 18.4 Retrieval-Augmented Generation (RAG)

Lewis et al. (2020) formalized retrieval-augmented generation, showing that grounding LLM outputs in retrieved documents dramatically reduces hallucination. OETA uses a local TF-IDF knowledge graph as the retrieval layer, with sentence-level snippet extraction to maximize information density per context token.

*Reference:* Lewis, P., Perez, E., Piktus, A., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020. arXiv:2005.11401.

### 18.5 Self-RAG: Adaptive Retrieval and Critique

Asai et al. (2023) proposed Self-RAG, in which a model learns to retrieve on demand and to critique retrieved passages for relevance. OETA implements a lightweight analogue: the `reflect()` function assesses retrieval usefulness after each search step, and re-planning is only triggered when the reflection signals a gap and the replan budget has not been exhausted.

*Reference:* Asai, A., Wu, Z., Wang, B., et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." arXiv:2310.11511.

---

## References

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y. (2022). **ReAct: Synergizing Reasoning and Acting in Language Models.** *ICLR 2023.* arXiv:2210.03629.
2. Shinn, N., Labash, B., Gopinath, A. (2023). **Reflexion: Language Agents with Verbal Reinforcement Learning.** *NeurIPS 2023.* arXiv:2303.11366.
3. Karpas, E., Abend, O., Belinkov, Y., et al. (2022). **MRKL Systems: A Modular, Neuro-Symbolic Architecture.** arXiv:2205.00445.
4. Lewis, P., Perez, E., Piktus, A., et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** *NeurIPS 2020.* arXiv:2005.11401.
5. Asai, A., Wu, Z., Wang, B., et al. (2023). **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.** arXiv:2310.11511.
6. Wei, J., Wang, X., Schuurmans, D., et al. (2022). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.** *NeurIPS 2022.* arXiv:2201.11903.
7. Schick, T., Dwivedi-Yu, J., Dessì, R., et al. (2023). **Toolformer: Language Models Can Teach Themselves to Use Tools.** *NeurIPS 2023.* arXiv:2302.04761.
