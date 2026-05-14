"""Tests for temporal cognition primitives."""
from __future__ import annotations

import time

import pytest

from task_agent.time_state import TimeState


def test_time_state_initialises():
    ts = TimeState()
    assert ts.urgency == "normal"
    assert ts.estimated_remaining_steps == 0
    assert ts.deadline is None


def test_compute_task_age_non_negative():
    ts = TimeState()
    age = ts.compute_task_age()
    assert age >= 0.0


def test_compute_idle_seconds_non_negative():
    ts = TimeState()
    idle = ts.compute_idle_seconds()
    assert idle >= 0.0


def test_update_progress_resets_idle():
    ts = TimeState()
    time.sleep(0.01)
    before = ts.compute_idle_seconds()
    ts.update_progress()
    after = ts.compute_idle_seconds()
    assert after <= before


def test_should_checkpoint_false_when_recent():
    ts = TimeState()
    ts.update_progress()
    assert ts.should_checkpoint(interval=60.0) is False


def test_should_checkpoint_true_when_old(monkeypatch):
    ts = TimeState()
    # Fake last_progress_at to 100 seconds ago
    monkeypatch.setattr(ts, "last_progress_at", time.time() - 100)
    assert ts.should_checkpoint(interval=30.0) is True


def test_should_replan_false_when_active():
    ts = TimeState()
    ts.update_progress()
    assert ts.should_replan_due_to_stall(threshold=120.0) is False


def test_should_replan_true_when_stalled(monkeypatch):
    ts = TimeState()
    monkeypatch.setattr(ts, "last_progress_at", time.time() - 200)
    assert ts.should_replan_due_to_stall(threshold=60.0) is True


def test_update_estimates_sets_fields():
    ts = TimeState()
    ts.update_estimates(steps_taken=3, steps_total=10, avg_step_s=2.0)
    assert ts.estimated_remaining_steps == 7
    assert ts.estimated_remaining_seconds == pytest.approx(14.0, abs=0.1)


def test_urgency_becomes_critical_with_tight_deadline(monkeypatch):
    ts = TimeState()
    ts.deadline = time.time() + 1.0  # 1 second from now
    ts.update_estimates(steps_taken=0, steps_total=10, avg_step_s=2.0)
    assert ts.urgency == "critical"


def test_to_dict_contains_required_keys():
    ts = TimeState()
    d = ts.to_dict()
    for key in ("task_age_s", "idle_s", "estimated_remaining_steps", "urgency"):
        assert key in d
