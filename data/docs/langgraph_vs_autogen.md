# LangGraph vs AutoGen for Agentic Workflows

## Overview

LangGraph and AutoGen represent two distinct design philosophies for building multi-step, tool-using AI agent systems. LangGraph (from LangChain / LangChain Inc.) models agent behavior as a stateful directed graph where nodes represent computation steps and edges represent conditional transitions. AutoGen (from Microsoft Research) structures multi-agent coordination as a conversational exchange between specialized agent personas. Both are actively maintained open-source projects targeting production-grade agentic applications.

## LangGraph

LangGraph is a library for building stateful, cyclical graph-structured workflows. Developers define workflows as Python functions attached to graph nodes; LangGraph handles state threading, checkpointing, and conditional routing. Core properties:

**Architecture**: Nodes are arbitrary Python callables (LLM chains, tool calls, routing logic). Edges are unconditional (always go from A to B) or conditional (route based on a function of the current state). The graph is compiled once and then invoked with an initial state dict.

**State management**: The framework owns a typed state schema (a TypedDict or Pydantic model). Every node receives the full current state and returns a partial update. This makes the control flow auditable before execution.

**Checkpointing**: Built-in persistence adapters (SQLite, PostgreSQL, Redis) let workflows pause, resume, and recover from failures. This is critical for long-horizon tasks that may span multiple LLM calls across minutes.

**Streaming**: LangGraph supports token-level streaming across multi-node graphs, useful for UI applications where partial results should appear progressively.

**Human-in-the-loop**: The graph can pause at a designated node and wait for human input before continuing. This is architecturally cleaner than polling in a loop.

**Subgraphs**: Complex workflows can be composed by embedding one compiled graph as a node inside another, enabling hierarchical agent structures.

**Limitations**: The graph abstraction adds cognitive overhead — developers must model their agent in terms of nodes and edges before writing any logic. The LangChain ecosystem dependency can be heavy, and debugging across async graph execution requires familiarity with the framework's state threading model.

## AutoGen

AutoGen (Microsoft Research) structures agent systems as conversations between specialized agent personas. Each agent has a system prompt, optional tool access, and communicates through a shared message history. Core properties:

**Architecture**: A `ConversableAgent` has a name, system prompt, and optionally a code executor or set of function tools. Conversations are initiated by calling `agent_a.initiate_chat(agent_b, message="...")`. The framework handles turn-taking, termination detection, and message routing.

**Code execution**: AutoGen has first-class support for Python code execution in a sandboxed Docker container or subprocess. An assistant agent generates code; a user-proxy agent executes it and reports back. This is one of AutoGen's most powerful features and requires essentially no boilerplate.

**Ease of prototyping**: A working two-agent system (planner + executor) can be running in under 30 lines of Python. The conversational abstraction is intuitive for developers already comfortable thinking in terms of chat.

**Model agnosticism**: Works with OpenAI, Azure OpenAI, Google Gemini, Anthropic Claude (via LiteLLM), and local Ollama models with the same interface.

**GroupChat**: AutoGen supports multi-agent group chats with a manager that selects which agent speaks next. This enables dynamic, emergent coordination patterns without manually coding routing logic.

**Limitations**: The conversational abstraction can obscure control flow — reasoning is embedded in natural-language messages rather than explicit graph edges, making it harder to enforce invariants. Long multi-agent conversations accumulate context, increasing cost. Termination conditions based on keyword matching can be fragile. State persistence requires external tooling.

## Comparison Matrix

| Dimension | LangGraph | AutoGen |
|---|---|---|
| Primary abstraction | Stateful directed graph | Multi-agent chat |
| Control flow | Explicit graph edges | Conversation turns |
| Observability | High (node-level state) | Medium (message log) |
| Setup complexity | Medium (graph compilation) | Low (initiate_chat) |
| Code execution | Via tool nodes | Built-in (Docker / subprocess) |
| Checkpointing / persistence | Built-in adapters | Requires external tooling |
| Human-in-the-loop | Interrupt at graph node | User proxy agent |
| Streaming | Token-level across nodes | Per-message |
| Multi-agent coordination | Subgraphs + conditional edges | GroupChat manager |
| Best for | Structured, auditable pipelines | Conversational task automation |

## Decision Guidance

**Choose LangGraph when:**
- Control flow has conditional branches that must be deterministic and auditable
- The workflow needs state persistence and resumability across failures
- You are building production systems where you need to inspect and modify state at each step
- Token-level streaming is important for user-facing interfaces
- Your organization already uses the LangChain ecosystem

**Choose AutoGen when:**
- You want to prototype a working agent system in minimal time
- The task naturally involves code generation and execution (data analysis, debugging, automation)
- Multi-agent coordination is emergent (you do not know the exact interaction pattern ahead of time)
- The interaction is inherently conversational (negotiation, review, brainstorming)
- You need to mix multiple LLM providers in one system

## For a Small Single-Agent Project

For a single-agent project (plan → retrieve → synthesize), both frameworks introduce unnecessary abstraction. LangGraph adds graph compilation overhead; AutoGen adds conversational message management. A hand-written loop with explicit state (as implemented in this repository) is more readable, more testable, and more instructive for understanding agent mechanics. Use a framework when your agent grows to multi-step coordination, human-in-the-loop requirements, or production persistence needs.
