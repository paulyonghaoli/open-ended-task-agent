# Architecture Overview

`open-ended-task-agent` is a compact demonstration of a higher-agency agent built with a layered, production-oriented design.

## Core principles

- explicit state and traceability
- separation of concerns across planning, retrieval, action selection, execution, and termination
- LLM-first behavior with deterministic fallback for safety
- bounded iteration and safe fail-stop logic
- evidence-driven decisions rather than opaque chain-of-thought

## Layers

### Planner

`task_agent/planner.py` creates a structured plan from a vague goal. When an OpenAI key is available, the planner uses the LLM to generate a goal-driven step plan with local corpus context, while still providing deterministic fallback behavior.

### Retrieval

`task_agent/knowledge.py` implements a prototype knowledge graph over local documents. It extracts keywords, builds document relations, and returns search results with related-document context. This is intentionally small but conceptually aligned with RAG and graph retrieval systems.

### Tool selection

`task_agent/tool_selector.py` decides which step to execute next. It ensures the agent does not compare or summarize without first collecting evidence. This is a simple example of goal-driven routing.

### Tools

`task_agent/tools.py` contains concrete actions and LLM-assisted tools:

- `local_search` for retrieval over the corpus
- `compare_items` and `llm_compare` for evidence-based comparison
- `summarize_findings` and `llm_summarize` for final synthesis
- `calculator` for light numeric reasoning

This registry-based design makes it easy to add new tools or swap local actions for remote services.

### Execution and termination

`task_agent/agent.py` orchestrates the loop. It updates `AgentState`, logs trace events, executes tools, and triggers replanning when evidence is insufficient.

`task_agent/termination.py` encodes safe stop conditions, including max-step budgets, repeated failures, unsupported goals, and no-progress scenarios.

## Scalability

The current implementation uses a small local document corpus, but it is designed to scale:

- replace `KnowledgeGraph` with a vector store or graph database for larger document sets
- extend the planner with task decomposition policies or LLM proposals
- add tool wrappers for external APIs, search services, or domain data
- preserve the same agent loop and trace model for auditability

## Why this design is useful

This repository is not trying to be a complete agent platform. Instead, it demonstrates an architectural approach that can grow into a more capable system:

- plan first
- retrieve evidence next
- act with tools
- observe and update state
- repeat until done

That sequence is the heart of open-ended task execution, and it is visible in the code rather than hidden inside a monolithic model.
