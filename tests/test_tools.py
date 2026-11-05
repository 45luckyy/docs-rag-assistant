import pytest

from app.tools import safe_eval


@pytest.mark.parametrize("expr,expected", [("(2 * 4) + 1", "9"), ("10 / 4", "2.5"), ("-3 ** 2", "-9")])
def test_safe_eval_arithmetic(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "a + 1", "2 ** 10000", "1/0", "x" * 300])
def test_safe_eval_rejects_unsafe_or_bad(expr):
    assert safe_eval(expr).startswith("error")
