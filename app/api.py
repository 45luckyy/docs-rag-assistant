import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph import RagService
from app.providers import get_embeddings, get_llm
from app.store import VectorStore

log = logging.getLogger("app")
RATE_LIMIT = 10  # requests per minute per client IP; protects the free-tier API quota
STATIC = Path(__file__).parent / "static"


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
    app.state.hits = defaultdict(deque)
    log.info("Loaded index with %d chunks (provider=%s)", app.state.rag.store.meta["count"], s.provider)
    yield


app = FastAPI(title="Docs RAG Assistant", lifespan=lifespan)


def check_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    window, now = request.app.state.hits[ip], time.monotonic()
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests, try again in a minute.")
    window.append(now)


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
