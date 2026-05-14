from __future__ import annotations

import ast
import operator
from math import comb, factorial as _factorial
from typing import Union


# Allowed AST node types for safe arithmetic evaluation
_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_SAFE_UNOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.expr) -> Union[int, float]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported literal type: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        fn = _SAFE_BINOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return fn(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        fn = _SAFE_UNOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return fn(_eval_node(node.operand))
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def count_characters(text: str, char: str) -> dict:
    """Count occurrences of a single character in text (case-insensitive)."""
    if not char or len(char) != 1:
        return {"success": False, "reason": "char must be exactly one character"}
    count = text.lower().count(char.lower())
    return {
        "success": True,
        "text": text,
        "char": char,
        "count": count,
        "result": f"'{char}' appears {count} time(s) in '{text}'",
    }


def safe_arithmetic_eval(expression: str) -> dict:
    """Evaluate a mathematical expression without using Python's built-in eval."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return {
            "success": True,
            "expression": expression,
            "result": result,
        }
    except ZeroDivisionError:
        return {"success": False, "reason": "division by zero", "expression": expression}
    except (ValueError, SyntaxError, TypeError) as exc:
        return {"success": False, "reason": str(exc), "expression": expression}


def n_choose_k(n: int, k: int) -> dict:
    """Compute binomial coefficient C(n, k)."""
    if n < 0 or k < 0 or k > n:
        return {"success": False, "reason": f"invalid arguments: n={n}, k={k}"}
    result = comb(n, k)
    return {
        "success": True,
        "n": n,
        "k": k,
        "result": result,
        "expression": f"C({n},{k}) = {result}",
    }


def factorial(n: int) -> dict:
    """Compute n!"""
    if not isinstance(n, int) or n < 0:
        return {"success": False, "reason": "factorial requires a non-negative integer"}
    result = _factorial(n)
    return {
        "success": True,
        "n": n,
        "result": result,
        "expression": f"{n}! = {result}",
    }
