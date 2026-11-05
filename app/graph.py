"""LangGraph RAG agent.

retrieve -> (top score below threshold?) -> refuse
         -> generate <-> tools (search_docs / calculator, bounded hops)
         -> finalize (keep only valid citations; no citation means refuse)
"""

import re
from typing import Annotated, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import Settings
from app.providers import REFUSAL
from app.store import Hit, VectorStore
from app.tools import TOOLS, safe_eval

SYSTEM_PROMPT = f"""You answer questions about the FastAPI documentation.
Use ONLY the numbered passages you are given. Rules:
- Cite every claim with passage numbers in square brackets, e.g. [1] or [2][3].
- If the passages do not contain the answer, reply exactly: "{REFUSAL}"
- Do not use outside knowledge, even if you know the answer.
- If passages are related but incomplete, you may call search_docs with a better query.
- Use calculator for any arithmetic.
- Be concise. Include short code only when the passages contain it."""

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class State(TypedDict, total=False):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    hits: list[Hit]  # citation [n] refers to hits[n-1]
    hops: int
    answer: str
    citations: list[dict]
    refused: bool
    refusal_reason: str


def format_passages(hits: list[Hit], start: int) -> str:
    return "\n\n".join(
        f"[{start + i}] {h.chunk.title} > {h.chunk.section} ({h.chunk.url})\n{h.chunk.text}"
        for i, h in enumerate(hits)
    )


def is_refusal(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith("i don't know") or REFUSAL.lower() in t


def build_graph(store: VectorStore, embeddings: Embeddings, llm: BaseChatModel, s: Settings):
    llm_with_tools = llm.bind_tools(TOOLS)

    def search(query: str) -> list[Hit]:
        return store.search(embeddings.embed_query(query), s.top_k)

    def retrieve(state: State) -> State:
        hits = search(state["question"])
        prompt = f"Question: {state['question']}\n\nPassages:\n{format_passages(hits, 1)}"
        return {
            "hits": hits, "hops": 0,
            "messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)],
        }

    def after_retrieve(state: State) -> str:
        hits = state["hits"]
        return "generate" if hits and hits[0].score >= s.min_score else "refuse"

    def generate(state: State) -> State:
        model = llm_with_tools if state.get("hops", 0) < s.max_search_hops else llm
        return {"messages": [model.invoke(state["messages"])]}

    def after_generate(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "finalize"

    def run_tools(state: State) -> State:
        hits = list(state["hits"])
        seen = {h.chunk.id for h in hits}
        out: list[ToolMessage] = []
        for call in state["messages"][-1].tool_calls:
            if call["name"] == "search_docs":
                new = [h for h in search(str(call["args"].get("query", ""))) if h.chunk.id not in seen]
                new = [h for h in new if h.score >= s.min_score]
                seen.update(h.chunk.id for h in new)
                content = format_passages(new, len(hits) + 1) if new else "No new relevant passages."
                hits.extend(new)
            elif call["name"] == "calculator":
                content = safe_eval(str(call["args"].get("expression", "")))
            else:
                content = f"Unknown tool {call['name']}"
            out.append(ToolMessage(content=content, tool_call_id=call["id"]))
        return {"messages": out, "hits": hits, "hops": state.get("hops", 0) + 1}

    def finalize(state: State) -> State:
        text = state["messages"][-1].text.strip()
        if is_refusal(text):
            return refuse_with("model found no answer in passages")
        hits = state["hits"]
        nums = sorted({int(n) for grp in _CITE.findall(text) for n in grp.split(",")})
        valid = [n for n in nums if 1 <= n <= len(hits)]
        if not valid:
            return refuse_with("answer had no valid citation")
        citations = [
            {"id": n, "title": hits[n - 1].chunk.title, "section": hits[n - 1].chunk.section,
             "url": hits[n - 1].chunk.url, "source": hits[n - 1].chunk.source,
             "score": round(hits[n - 1].score, 3)}
            for n in valid
        ]
        return {"answer": text, "citations": citations, "refused": False}

    def refuse_with(reason: str) -> State:
        return {"answer": REFUSAL, "citations": [], "refused": True, "refusal_reason": reason}

    def refuse(state: State) -> State:
        return refuse_with("no passage above similarity threshold")

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("tools", run_tools)
    g.add_node("finalize", finalize)
    g.add_node("refuse", refuse)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, ["generate", "refuse"])
    g.add_conditional_edges("generate", after_generate, ["tools", "finalize"])
    g.add_edge("tools", "generate")
    g.add_edge("finalize", END)
    g.add_edge("refuse", END)
    return g.compile()


class RagService:
    """Holds the loaded index and compiled graph; one per process."""

    def __init__(self, s: Settings, store: VectorStore, embeddings: Embeddings, llm: BaseChatModel):
        if store.meta.get("provider") != s.provider:
            raise ValueError(
                f"Index was built with provider={store.meta.get('provider')!r} but "
                f"PROVIDER={s.provider!r}. Re-run `python -m app.ingest`."
            )
        self.settings, self.store = s, store
        self.graph = build_graph(store, embeddings, llm, s)

    def ask(self, question: str) -> dict:
        result = self.graph.invoke({"question": question}, {"recursion_limit": 12})
        return {
            "answer": result["answer"],
            "refused": result["refused"],
            "refusal_reason": result.get("refusal_reason"),
            "citations": result["citations"],
            "contexts": [h.chunk.text for h in result.get("hits", [])],
            "retrieved_sources": [h.chunk.source for h in result.get("hits", [])],
            "top_score": round(result["hits"][0].score, 3) if result.get("hits") else None,
        }
