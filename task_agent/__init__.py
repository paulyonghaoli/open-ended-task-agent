"""
open-ended-task-agent — a compact implementation of a higher-agency task agent.

Public API:
    run_agent(goal, max_steps=10) -> AgentState
    AgentState  — full execution state including trace, reflections, and answer
"""
from .agent import run_agent
from .state import AgentState, AgentPlan, PlanStep, ReflectionNote, TraceEventType

__all__ = [
    "run_agent",
    "AgentState",
    "AgentPlan",
    "PlanStep",
    "ReflectionNote",
    "TraceEventType",
]
