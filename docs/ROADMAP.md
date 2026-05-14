# OETA-Lab Roadmap

## Current: Phase 1 — Stable Cognitive Runtime Foundation

**Status:** Complete

All Phase 1 modules are implemented, tested (204 tests), and integrated into
the main agent loop.

| Module | Status | Description |
|---|---|---|
| `router.py` | Done | 6-mode cognitive substrate routing |
| `symbolic.py` | Done | Deterministic char-count, arithmetic, combinatorics |
| `time_state.py` | Done | Temporal primitives: idle, stall, checkpoint, deadline |
| `ledger.py` | Done | TaskLedger: known facts, open questions, failure log |
| `verifier.py` | Done | Post-execution sanity check; contradiction detection |
| `memory.py` | Done | Flat-file EpisodeMemoryStore |
| `sleep_cycle.py` | Done | Offline replay: skill extraction, failure summary |
| `export.py` | Done | Full AgentState → JSON serialisation |

---

## Phase 2 — Evidence and Retrieval Upgrade

**Goal:** Replace TF-IDF with dense retrieval; improve evidence quality signals.

- [ ] Add `retrieval/dense.py`: sentence-transformers `all-MiniLM-L6-v2` retrieval
- [ ] Add `retrieval/hybrid.py`: BM25 + dense fusion (RRF)
- [ ] Add NLI-based relevance scorer for Self-RAG-style filtering
- [ ] Extend `KnowledgeGraph` with dense index alongside TF-IDF
- [ ] Add `--retrieval-mode [tfidf|dense|hybrid]` CLI flag

---

## Phase 3 — Verifier-Guided Stopping

**Goal:** Give the verifier real influence over replanning and termination.

- [ ] Extend `VerificationReport` with `suggested_action` field
- [ ] Add `VERIFICATION_TRIGGERED_REPLAN` trace event
- [ ] Give verifier a vote in `should_terminate()` when contradiction_risk > 0.6
- [ ] Add configurable verifier thresholds to `config.py`

---

## Phase 4 — GraphRAG-Lite

**Goal:** Multi-hop reasoning via knowledge graph traversal.

- [ ] Implement `graph_traversal.py`: BFS/DFS over `KnowledgeGraph` adjacency
- [ ] Add `graph_search` action type in planner and agent loop
- [ ] Route goals with GRAPH cognitive mode to graph_traversal tool
- [ ] Add `graph_query()` for "what is connected to X?" style questions

---

## Phase 5 — Skill Learning

**Goal:** Turn candidate skills from the sleep cycle into reusable plan templates.

- [ ] Formalise `Skill` dataclass: trigger, plan_template, success_rate
- [ ] Add `skill_store.py`: persist and load skills from `.oeta_sleep/`
- [ ] Integrate skill lookup into `planner.py` as highest-priority strategy
- [ ] Add `skill_match_score()` to rank applicable skills for a given goal

---

## Phase 6 — Episodic Memory with Semantic Search

**Goal:** Replace term-overlap search with sentence embedding similarity.

- [ ] Extend `EpisodeMemoryStore` with vector index (faiss or hnswlib)
- [ ] Add `search_by_embedding(query)` method
- [ ] Add `recall_relevant(goal)` that surfaces the 3 most similar past episodes
- [ ] Surface recalled episodes in `AgentState` for plan initialisation

---

## Phase 7 — Streaming and Observability

**Goal:** Real-time trace visibility for long-running agent sessions.

- [ ] Add SSE streaming to FastAPI `/run` endpoint (yield trace events as they happen)
- [ ] Add `--stream` CLI flag that prints trace events live
- [ ] Expose `evidence_quality` and `step_count` as Prometheus metrics (optional)

---

## Out of Scope (by design)

- Multi-agent coordination or role-based conversation
- Autonomous code execution or OS-level actions
- Browser automation or live web retrieval
- Production authentication or multi-tenancy
- Comparison with frontier models on open benchmarks
