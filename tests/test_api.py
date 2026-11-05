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
