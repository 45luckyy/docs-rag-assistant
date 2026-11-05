"""Embedding and chat-model providers.

`gemini` is the real setup. `offline` needs no key or network: hashing-vectorizer
embeddings plus an extractive "LLM" that quotes the best-matching sentence. It
exists so tests, CI and local demos run anywhere; its answers are not generative.
"""

import re
from functools import lru_cache
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import Settings

EMBED_DIM_GEMINI = 768
REFUSAL = "I don't know based on the provided documents."


class HashingEmbeddings(Embeddings):
    def __init__(self, n_features: int = 2**13):
        # Imported here: scikit-learn is only needed offline and is slow to import.
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            stop_words="english",
            norm="l2",
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._vec.transform(texts).toarray().astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


_WORD = re.compile(r"[a-z0-9_]+")


@lru_cache
def _stop_words() -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return ENGLISH_STOP_WORDS


def _terms(text: str) -> set[str]:
    stop = _stop_words()
    return {w for w in _WORD.findall(text.lower()) if w not in stop and len(w) > 2}


class ExtractiveChatModel(BaseChatModel):
    """Answers by quoting the context sentence with most question-term overlap.

    Expects the prompt format built in app.graph: a "Question:" line and
    numbered "[n]" context blocks. Never calls tools.
    """

    @property
    def _llm_type(self) -> str:
        return "extractive-offline"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ExtractiveChatModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw) -> ChatResult:
        prompt = str(messages[-1].content)
        q_match = re.search(r"^Question:\s*(.+)$", prompt, re.M)
        q_terms = _terms(q_match.group(1)) if q_match else set()
        blocks = re.findall(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z)", prompt, re.M | re.S)

        best, best_overlap = None, 0
        for n, body in blocks:
            for sent in re.split(r"(?<=[.!?])\s+|\n{2,}", body):
                sent = " ".join(sent.split())
                if len(sent) < 25 or sent.startswith("```"):
                    continue
                overlap = len(q_terms & _terms(sent))
                if overlap > best_overlap:
                    best, best_overlap = f"{sent} [{n}]", overlap

        answer = best if best and best_overlap >= 2 else REFUSAL
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])


def get_embeddings(s: Settings) -> Embeddings:
    if s.provider == "offline":
        return HashingEmbeddings()
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=s.embed_model,
        google_api_key=s.google_api_key,
        output_dimensionality=EMBED_DIM_GEMINI,
    )


def get_llm(s: Settings) -> BaseChatModel:
    if s.provider == "offline":
        return ExtractiveChatModel()
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = {"model": s.llm_model, "google_api_key": s.google_api_key, "max_retries": 3}
    if "flash-lite" not in s.llm_model:  # Flash-Lite uses fixed sampling and warns on temperature
        kwargs["temperature"] = 0
    return ChatGoogleGenerativeAI(**kwargs)
