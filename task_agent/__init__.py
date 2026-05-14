"""
open-ended-task-agent — OETA-Lab cognitive runtime.

Public API:
    run_agent(goal, max_steps=10, enable_memory=False) -> AgentState
"""
from .agent import run_agent
from .export import export_state
from .ledger import TaskLedger
from .memory import Episode, EpisodeMemoryStore
from .router import CognitiveMode, RoutingDecision, route_reasoning_mode
from .sleep_cycle import run_sleep_cycle
from .state import AgentPlan, AgentState, PlanStep, ReflectionNote, TraceEventType
from .symbolic import count_characters, factorial, n_choose_k, safe_arithmetic_eval
from .time_state import TimeState
from .verifier import VerificationReport, verify_state

__all__ = [
    # Core runtime
    "run_agent",
    "export_state",
    # State types
    "AgentState",
    "AgentPlan",
    "PlanStep",
    "ReflectionNote",
    "TraceEventType",
    # Cognitive router
    "CognitiveMode",
    "RoutingDecision",
    "route_reasoning_mode",
    # Symbolic tools
    "count_characters",
    "safe_arithmetic_eval",
    "factorial",
    "n_choose_k",
    # Temporal cognition
    "TimeState",
    # Ledger
    "TaskLedger",
    # Verifier
    "VerificationReport",
    "verify_state",
    # Memory
    "Episode",
    "EpisodeMemoryStore",
    # Sleep cycle
    "run_sleep_cycle",
]
