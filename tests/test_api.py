import pytest
from fastapi.testclient import TestClient

from app import config


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr("app.api.get_settings", lambda: settings)
    from app.api import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["provider"] == "offline"


def test_ask_answer_and_refusal(client):
    ok = client.post("/ask", json={"question": "How do I run background tasks after a response?"}).json()
    assert not ok["refused"] and ok["citations"][0]["url"].endswith("/tutorial/background-tasks/")
    no = client.post("/ask", json={"question": "Best pizza in Naples?"}).json()
    assert no["refused"] and no["citations"] == []


def test_validation(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422


def test_rate_limit(client):
    codes = [client.post("/ask", json={"question": "background tasks?"}).status_code for _ in range(11)]
    assert codes[:10] == [200] * 10 and codes[10] == 429


def test_index_page(client):
    assert "FastAPI Docs Assistant" in client.get("/").text


def test_limiter_daily_cap_resets_next_day():
    from datetime import date

    from fastapi import HTTPException

    from app.api import Limiter

    day = [date(2026, 11, 5)]
    lim = Limiter(per_minute=100, per_day=3, clock=lambda: 0.0, today=lambda: day[0])
    for ip in ("a", "b", "c"):
        lim.check(ip)
    with pytest.raises(HTTPException) as exc:
        lim.check("d")
    assert exc.value.status_code == 503
    day[0] = date(2026, 11, 6)
    lim.check("d")  # new day, cap reset


def test_limiter_prunes_idle_ips():
    from app.api import Limiter

    t = [0.0]
    lim = Limiter(per_minute=10, per_day=10_000, clock=lambda: t[0])
    for i in range(500):
        lim.check(f"10.0.{i // 250}.{i % 250}")
    assert lim.tracked_ips == 500
    t[0] = 400.0  # idle for > 5 minutes
    lim.check("1.2.3.4")
    assert lim.tracked_ips == 1


def test_limiter_is_thread_safe():
    import threading

    from fastapi import HTTPException

    from app.api import Limiter

    lim = Limiter(per_minute=50, per_day=10_000)
    ok = []

    def hit():
        try:
            lim.check("same-ip")
            ok.append(1)
        except HTTPException:
            pass

    threads = [threading.Thread(target=hit) for _ in range(200)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(ok) == 50  # exactly the limit, no races
