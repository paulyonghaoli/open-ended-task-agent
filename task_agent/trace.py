from __future__ import annotations

import time
from typing import Any, Dict, Optional


def make_event(
    step: int,
    event_type: str,
    tool_name: Optional[str] = None,
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    elapsed_s: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "step": step,
        "event_type": event_type,
        "tool_name": tool_name,
        "reason": reason,
        "metadata": metadata or {},
        "elapsed_s": elapsed_s if elapsed_s is not None else round(time.time(), 3),
    }


def format_trace_line(event: Dict[str, Any]) -> str:
    step = event.get("step", "?")
    etype = event.get("event_type", "UNKNOWN")
    tool = event.get("tool_name")
    reason = event.get("reason", "")
    elapsed = event.get("elapsed_s", "")
    tool_str = f"[{tool}] " if tool else ""
    elapsed_str = f" +{elapsed}s" if elapsed else ""
    reason_str = f" — {reason}" if reason else ""
    return f"  {step:>3}. {etype:<28} {tool_str}{reason_str}{elapsed_str}"
