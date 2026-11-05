import pytest

from eval import run_eval


class FlakySvc:
    def __init__(self, errors):
        self.errors = list(errors)

    def ask(self, question):
        if self.errors:
            raise RuntimeError(self.errors.pop(0))
        return {"answer": "ok [1]"}


def test_ask_with_retry_waits_on_rate_limit(monkeypatch):
    sleeps = []
    monkeypatch.setattr(run_eval.time, "sleep", sleeps.append)
    svc = FlakySvc(["429 RESOURCE_EXHAUSTED per minute. Please retry in 47.7s."])
    assert run_eval.ask_with_retry(svc, "q")["answer"] == "ok [1]"
    assert sleeps == [pytest.approx(49.7)]


def test_ask_with_retry_stops_on_daily_quota(monkeypatch):
    monkeypatch.setattr(run_eval.time, "sleep", lambda s: None)
    svc = FlakySvc(["429 RESOURCE_EXHAUSTED requests per day exceeded. Please retry in 10h2m3s."])
    with pytest.raises(SystemExit, match="quota exhausted"):
        run_eval.ask_with_retry(svc, "q")


def test_ask_with_retry_reraises_other_errors():
    with pytest.raises(RuntimeError, match="boom"):
        run_eval.ask_with_retry(FlakySvc(["boom"]), "q")


def test_ask_with_retry_waits_on_busy_server(monkeypatch):
    sleeps = []
    monkeypatch.setattr(run_eval.time, "sleep", sleeps.append)
    svc = FlakySvc(["503 UNAVAILABLE. This model is currently experiencing high demand."])
    assert run_eval.ask_with_retry(svc, "q")["answer"] == "ok [1]"
    assert sleeps == [20.0]
