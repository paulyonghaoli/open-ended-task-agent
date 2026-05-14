# OETA-Lab Open Research Questions

These are questions that the OETA-Lab codebase is designed to explore or
expose — not questions with known answers, but ones where the architecture
creates useful experimental footholds.

---

## 1. Cognitive Substrate Routing

**Question:** Can lightweight heuristic routing (regex-based) produce the same
task decomposition quality as LLM-based routing, and under what conditions
does it fail?

**Observable in OETA-Lab:** The `router.py` uses regex patterns. The trace
logs `ROUTING_DECIDED` with the selected mode and rationale. You can
compare runs with and without the routing decision influencing the plan.

**Hypothesis:** Heuristic routing works for clearly-typed goals (arithmetic,
counting, comparison) but degrades for ambiguous compound goals
("compare and calculate X").

---

## 2. Verifier-Guided Stopping

**Question:** Does adding a post-execution verifier that can detect one-sided
comparisons, confidence inflation, and corpus gaps reduce the rate of
misleading answers — and at what cost in additional steps?

**Observable in OETA-Lab:** `verifier.py` produces a `VerificationReport`
with `passed`, `issues`, and `contradiction_risk`. The eval harness reports
`verifier_issue_rate`. Current version: verifier runs after termination (no
loop influence). Phase 3 will give it a vote.

**Hypothesis:** Verifier-triggered replans are most valuable for comparison
goals where only one candidate has corpus coverage.

---

## 3. Temporal Cognition: Does Task Age Predict Failure?

**Question:** Is there a correlation between `time_state.task_age_s` at
termination and the final termination status? Do stalled agents
(long idle periods) correlate with TERMINATED_PARTIAL or TERMINATED_FAILED?

**Observable in OETA-Lab:** `TimeState.compute_task_age()` and
`compute_idle_seconds()` are logged. The `export_state()` output includes
`temporal.task_age_s`. Running the sleep cycle aggregates across episodes.

**Hypothesis:** In the current synchronous implementation, task age is
dominated by step count × average step duration, so it's a proxy for
complexity rather than an independent signal. This changes in async settings.

---

## 4. Evidence Quality as a Calibration Signal

**Question:** Does the numeric `evidence_quality` score correlate with
answer accuracy? Is it a useful proxy for when to trust the final answer?

**Observable in OETA-Lab:** Every run produces an `evidence_quality` float.
The `eval_results.json` output includes it per scenario. The verifier checks
`confidence == "high" and evidence_quality < 0.25`.

**Known limitations:** The score uses observation count and source diversity
as proxies. A single long passage from one source can inflate it; a corpus
gap will deflate it even when the answer is correct.

---

## 5. Sleep Cycle: Does Offline Replay Produce Useful Skills?

**Question:** Can the `extract_candidate_skills()` function produce plan
templates that improve future runs — i.e., does running similar goals
repeatedly converge on a stable skill representation?

**Observable in OETA-Lab:** Run `python scripts/sleep_cycle.py` after
accumulating episodes. `candidate_skills.json` shows (routing_mode,
trigger_word, success_count) tuples. Phase 5 will close the loop by
feeding these back into the planner.

**Hypothesis:** After 10+ successful comparison runs, "compare" + "retrieval"
becomes a high-confidence skill candidate. The skill's plan template
(search-A → search-B → compare → summarize) can be retrieved without
LLM planning.

---

## 6. Symbolic vs Language Reasoning: When Does Determinism Win?

**Question:** For what class of goals does replacing LLM calls with
deterministic symbolic tools produce higher accuracy? What is the boundary
between tasks that are symbolic (count letters, evaluate arithmetic) and
tasks that require language reasoning (explain, summarize, compare)?

**Observable in OETA-Lab:** The router classifies goals as SYMBOLIC or
non-SYMBOLIC. The `symbolic.py` module produces exact answers. You can
compare `evidence_quality` (1.0 for symbolic) vs retrieval-based runs.

**Hypothesis:** The boundary is sharp for well-formed goals ("count r in
strawberry" vs "how does DuckDB perform on analytics") but blurry for
compound goals ("count the databases that support SQL"). Compound goals
need both symbolic and retrieval modes.

---

## 7. Task Ledger Utility

**Question:** Does maintaining a `TaskLedger` (known_facts, open_questions,
failed_attempts) during execution produce measurably better replanning
decisions compared to relying solely on the observation list?

**Observable in OETA-Lab:** The ledger is attached to every `AgentState` and
exported via `export_state()`. It currently updates opportunistically but
does not influence planning decisions.

**Hypothesis:** The ledger's `open_questions` list would be most useful as
input to the re-planner: if a question remains unresolved after all planned
searches, trigger a targeted refinement search specifically for that question
rather than re-searching the original goal.
