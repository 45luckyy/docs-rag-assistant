import pytest

from app.tools import safe_eval


@pytest.mark.parametrize("expr,expected", [("(2 * 4) + 1", "9"), ("10 / 4", "2.5"), ("-3 ** 2", "-9")])
def test_safe_eval_arithmetic(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "a + 1", "2 ** 10000", "1/0", "x" * 300])
def test_safe_eval_rejects_unsafe_or_bad(expr):
    assert safe_eval(expr).startswith("error")


@pytest.mark.parametrize("expr", [
    "(((9**100)**100)**100)**100",   # hung the server before the fix
    "((9**100)**100)**100",
    "9**100*9**100*9**100*9**100",
    "1e308*10",
    "2.0**1000*2.0**1000",
])
def test_safe_eval_blocks_huge_results_quickly(expr):
    import time

    start = time.perf_counter()
    assert safe_eval(expr).startswith("error")
    assert time.perf_counter() - start < 0.5


def test_safe_eval_still_handles_normal_powers():
    assert safe_eval("2 ** 64") == str(2**64)
    assert safe_eval("(2 * 4 + 1) ** 3") == "729"
    assert safe_eval("0.5 ** 2") == "0.25"
