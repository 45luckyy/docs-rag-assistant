from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import Settings
from app.ingest import build_index

DOCS = {
    "tutorial/cors.md": """# CORS (Cross-Origin Resource Sharing) { #cors }

You can configure CORS in your FastAPI application using the `CORSMiddleware`.
Import `CORSMiddleware`, create a list of allowed origins and add it as a middleware
to your FastAPI application with allow_origins, allow_credentials, allow_methods and allow_headers.

/// tip

You could also use `from starlette.middleware.cors import CORSMiddleware`.

///
""",
    "tutorial/background-tasks.md": """# Background Tasks

You can define background tasks to be run after returning a response. This is useful for
operations that need to happen after a request, but that the client doesn't really have to be
waiting for the operation to complete before receiving the response, like email notifications.

{* ../../docs_src/background_tasks/tutorial001_py310.py hl[1,13] *}

Import `BackgroundTasks` and define a parameter in your path operation function with a type
declaration of `BackgroundTasks`.
""",
    "deployment/server-workers.md": """# Server Workers - Uvicorn with Workers

You can use the `--workers` command line option to start multiple worker processes when
running FastAPI in production with the fastapi run command. Each worker process handles requests,
so multiple workers let you use multiple CPU cores on the server machine.
""",
    "release-notes.md": "# Release Notes\n\n" + "Changelog entry about version bumps. " * 20,
}


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    root: Path = tmp_path_factory.mktemp("rag")
    for rel, text in DOCS.items():
        p = root / "docs" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    s = Settings(provider="offline", docs_dir=root / "docs", index_dir=root / "index",
                 chunk_size=400, chunk_overlap=40, top_k=2, min_score=0.15, _env_file=None)
    build_index(s)
    return s


class ScriptedChatModel(BaseChatModel):
    """Returns pre-set AIMessages in order; records every prompt it sees."""

    responses: list[AIMessage]
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])
