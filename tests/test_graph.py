import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.graph import RagService
from app.providers import REFUSAL, ExtractiveChatModel, HashingEmbeddings
from app.store import VectorStore
from tests.conftest import ScriptedChatModel


def service(settings, llm):
    return RagService(settings, VectorStore.load(settings.index_dir), HashingEmbeddings(), llm)


def test_offline_answer_is_cited(settings):
    r = service(settings, ExtractiveChatModel()).ask("How do I configure CORS middleware?")
    assert not r["refused"]
    assert r["citations"][0]["source"] == "tutorial/cors.md"


def test_out_of_scope_refused_before_llm(settings):
    llm = ScriptedChatModel(responses=[])  # would raise if called
    r = service(settings, llm).ask("What is the capital of France?")
    assert r["refused"] and r["answer"] == REFUSAL
    assert r["refusal_reason"] == "no passage above similarity threshold"


def test_model_refusal_is_respected(settings):
    llm = ScriptedChatModel(responses=[AIMessage(content=REFUSAL)])
    r = service(settings, llm).ask("How do I configure CORS middleware for websockets?")
    assert r["refused"] and r["citations"] == []


@pytest.mark.parametrize("reply", ["Use CORSMiddleware.", "Use CORSMiddleware [99]."])
def test_uncited_or_invalid_citation_becomes_refusal(settings, reply):
    llm = ScriptedChatModel(responses=[AIMessage(content=reply)])
    r = service(settings, llm).ask("How do I configure CORS middleware?")
    assert r["refused"] and r["refusal_reason"] == "answer had no valid citation"


def test_search_tool_adds_passages_and_citation_maps(settings):
    llm = ScriptedChatModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "search_docs", "id": "t1",
                                           "args": {"query": "uvicorn workers multiple CPU cores"}}]),
        AIMessage(content="Use the --workers option [3]."),
    ])
    r = service(settings, llm).ask("How do I configure CORS middleware?")
    assert not r["refused"]
    assert r["citations"][0]["source"] == "deployment/server-workers.md"
    tool_msg = llm.seen[1][-1]
    assert isinstance(tool_msg, ToolMessage) and "[3]" in tool_msg.content


def test_calculator_tool_and_hop_limit(settings):
    call = {"name": "calculator", "id": "c", "args": {"expression": "(2 * 4) + 1"}}
    llm = ScriptedChatModel(responses=[
        AIMessage(content="", tool_calls=[call]),
        AIMessage(content="", tool_calls=[{**call, "id": "c2"}]),
        AIMessage(content="Start 9 workers [1]."),
    ])
    r = service(settings, llm).ask("How many workers should I run with the workers option?")
    assert llm.seen[1][-1].content == "9"
    assert not r["refused"] and len(llm.seen) == 3  # hop limit (2) forces a final answer


def test_provider_mismatch_raises(settings):
    store = VectorStore.load(settings.index_dir)
    with pytest.raises(ValueError, match="Re-run"):
        RagService(settings.model_copy(update={"provider": "gemini"}), store, HashingEmbeddings(),
                   ExtractiveChatModel())
