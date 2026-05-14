"""Tests for deterministic symbolic tools."""
from __future__ import annotations

import pytest

from task_agent.symbolic import (
    count_characters,
    factorial,
    n_choose_k,
    safe_arithmetic_eval,
)


# ── count_characters ───────────────────────────────────────────────────────────

def test_count_r_in_strawberry():
    r = count_characters("strawberry", "r")
    assert r["success"] is True
    assert r["count"] == 3


def test_count_case_insensitive():
    r = count_characters("Strawberry", "r")
    assert r["count"] == 3


def test_count_zero_occurrences():
    r = count_characters("python", "z")
    assert r["success"] is True
    assert r["count"] == 0


def test_count_single_char_text():
    r = count_characters("aaaa", "a")
    assert r["count"] == 4


def test_count_invalid_char_empty():
    r = count_characters("hello", "")
    assert r["success"] is False


def test_count_invalid_char_multi():
    r = count_characters("hello", "ab")
    assert r["success"] is False


def test_count_result_string():
    r = count_characters("strawberry", "r")
    assert "3" in r["result"]
    assert "strawberry" in r["result"]


# ── safe_arithmetic_eval ───────────────────────────────────────────────────────

def test_arithmetic_addition():
    r = safe_arithmetic_eval("2 + 3")
    assert r["success"] is True
    assert r["result"] == 5


def test_arithmetic_multiplication():
    r = safe_arithmetic_eval("6 * 7")
    assert r["result"] == 42


def test_arithmetic_division():
    r = safe_arithmetic_eval("10 / 4")
    assert r["success"] is True
    assert r["result"] == 2.5


def test_arithmetic_power():
    r = safe_arithmetic_eval("2 ** 10")
    assert r["result"] == 1024


def test_arithmetic_complex_expression():
    r = safe_arithmetic_eval("3 + 4 * 2")
    assert r["success"] is True
    assert r["result"] == 11  # operator precedence: 3 + (4*2)


def test_arithmetic_zero_division():
    r = safe_arithmetic_eval("1 / 0")
    assert r["success"] is False


def test_arithmetic_invalid_expression():
    r = safe_arithmetic_eval("import os")
    assert r["success"] is False


def test_arithmetic_modulo():
    r = safe_arithmetic_eval("17 % 5")
    assert r["result"] == 2


def test_arithmetic_floor_division():
    r = safe_arithmetic_eval("17 // 5")
    assert r["result"] == 3


def test_arithmetic_negative():
    r = safe_arithmetic_eval("-5 + 3")
    assert r["result"] == -2


# ── n_choose_k ─────────────────────────────────────────────────────────────────

def test_combinations_basic():
    r = n_choose_k(5, 2)
    assert r["success"] is True
    assert r["result"] == 10


def test_combinations_k_zero():
    r = n_choose_k(10, 0)
    assert r["result"] == 1


def test_combinations_k_equals_n():
    r = n_choose_k(7, 7)
    assert r["result"] == 1


def test_combinations_invalid_k_greater():
    r = n_choose_k(3, 5)
    assert r["success"] is False


def test_combinations_negative_n():
    r = n_choose_k(-1, 0)
    assert r["success"] is False


# ── factorial ──────────────────────────────────────────────────────────────────

def test_factorial_zero():
    r = factorial(0)
    assert r["success"] is True
    assert r["result"] == 1


def test_factorial_five():
    r = factorial(5)
    assert r["result"] == 120


def test_factorial_ten():
    r = factorial(10)
    assert r["result"] == 3628800


def test_factorial_negative():
    r = factorial(-1)
    assert r["success"] is False
