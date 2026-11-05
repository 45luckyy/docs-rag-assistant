import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph import RagService
from app.providers import get_embeddings, get_llm
from app.store import VectorStore

log = logging.getLogger("app")
STATIC = Path(__file__).parent / "static"


class Limiter:
    """Per-IP sliding-window limit plus a global daily cap (resets at UTC midnight).

    Thread-safe (sync endpoints run in a thread pool) and prunes idle IPs so memory
    doesn't grow with every client ever seen.
    """

    def __init__(self, per_minute: int, per_day: int, clock=time.monotonic, today=None):
        self.per_minute, self.per_day = per_minute, per_day
        self._clock = clock
        self._today = today or (lambda: datetime.now(UTC).date())
        self._lock = threading.Lock()
        self._windows: dict[str, deque] = {}
        self._day, self._day_count = self._today(), 0
        self._last_prune = clock()

    def check(self, ip: str) -> None:
        with self._lock:
            now, today = self._clock(), self._today()
            if today != self._day:
                self._day, self._day_count = today, 0
            if now - self._last_prune > 300:
                self._windows = {k: w for k, w in self._windows.items() if w and now - w[-1] <= 60}
                self._last_prune = now
            window = self._windows.setdefault(ip, deque())
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.per_minute:
                raise HTTPException(429, "Too many requests, try again in a minute.")
            if self._day_count >= self.per_day:
                raise HTTPException(503, "The demo has reached its daily limit. Please try again tomorrow.")
            window.append(now)
            self._day_count += 1

    @property
    def tracked_ips(self) -> int:
        return len(self._windows)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Citation(BaseModel):
    id: int
    title: str
    section: str
    url: str
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None = None
    citations: list[Citation]
    latency_ms: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.rag = RagService(s, VectorStore.load(s.index_dir), get_embeddings(s), get_llm(s))
    app.state.limiter = Limiter(s.rate_limit_per_minute, s.daily_ask_limit)
    log.info("Loaded index with %d chunks (provider=%s)", app.state.rag.store.meta["count"], s.provider)
    yield


app = FastAPI(title="Docs RAG Assistant", lifespan=lifespan)


def check_rate_limit(request: Request) -> None:
    request.app.state.limiter.check(request.client.host if request.client else "unknown")


@app.get("/health")
def health(request: Request) -> dict:
    rag = request.app.state.rag
    return {"status": "ok", "provider": rag.settings.provider, "chunks": rag.store.meta["count"]}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    check_rate_limit(request)
    start = time.perf_counter()
    try:
        result = request.app.state.rag.ask(body.question.strip())
    except Exception as exc:
        log.exception("ask failed")
        raise HTTPException(502, "The language model is unavailable right now. Try again shortly.") from exc
    latency = int((time.perf_counter() - start) * 1000)
    log.info("ask refused=%s citations=%d latency_ms=%d",
             result["refused"], len(result["citations"]), latency)
    return AskResponse(**{k: result[k] for k in ("answer", "refused", "refusal_reason", "citations")},
                       latency_ms=latency)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
