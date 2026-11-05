"""Tools the agent can call. Execution happens in app.graph (it needs the store)."""

import ast
import math
import operator

from langchain_core.tools import tool


@tool
def search_docs(query: str) -> str:
    """Search the documentation again with a new, more specific query. Use when the
    current passages are related but miss part of the answer."""
    raise NotImplementedError("executed by the graph")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(2 * 4) + 1'."""
    return safe_eval(expression)


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


MAX_BITS = 512  # ~150 digits: plenty for arithmetic, too small to burn CPU or memory


def _check_size(value):
    if isinstance(value, int) and value.bit_length() > MAX_BITS:
        raise ValueError("result too large")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result too large")
    return value


def safe_eval(expression: str) -> str:
    """Arithmetic only: no names, calls or attributes, so nothing can be executed.

    Every intermediate result is size-checked, and powers are checked *before*
    computing, so nested exponents like ((9**100)**100)**100 can't hang the server.
    """

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return _check_size(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 100:
                    raise ValueError("exponent too large")
                if abs(left) > 1 and right > 0 and right * math.log2(abs(left)) > MAX_BITS:
                    raise ValueError("result too large")
            return _check_size(_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _check_size(_OPS[type(node.op)](ev(node.operand)))
        raise ValueError("unsupported expression")

    if len(expression) > 200:
        return "error: expression too long"
    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError) as exc:
        return f"error: {exc}"


TOOLS = [search_docs, calculator]
