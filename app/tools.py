"""Tools the agent can call. Execution happens in app.graph (it needs the store)."""

import ast
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


def safe_eval(expression: str) -> str:
    """Arithmetic only: no names, calls or attributes, so nothing can be executed."""

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    if len(expression) > 200:
        return "error: expression too long"
    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"error: {exc}"


TOOLS = [search_docs, calculator]
