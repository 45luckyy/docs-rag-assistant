"""Evaluate the assistant on the golden set.

  python -m eval.run_eval                 # retrieval hit@k, citation accuracy, refusal accuracy
  python -m eval.run_eval --sweep         # pick the refusal threshold (embeddings only, cheap)
  python -m eval.run_eval --ragas         # + RAGAS faithfulness / answer relevancy / context recall
                                          #   (needs requirements-eval.txt and PROVIDER=gemini)
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

from app.config import get_settings
from app.graph import RagService
from app.ingest import retry_wait
from app.providers import get_embeddings, get_llm
from app.store import VectorStore

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score(golden: list[dict], outputs: list[dict]) -> dict:
    ins = [(g, o) for g, o in zip(golden, outputs, strict=True) if g["in_scope"]]
    outs = [(g, o) for g, o in zip(golden, outputs, strict=True) if not g["in_scope"]]
    answered = [(g, o) for g, o in ins if not o["refused"]]

    def hit(g, o):
        return any(src in g["expected_sources"] for src in o["retrieved_sources"])

    def cited_right(g, o):
        return any(c["source"] in g["expected_sources"] for c in o["citations"])

    def pct(n, d):
        return round(100 * n / d, 1) if d else None

    return {
        "n_questions": len(golden),
        "n_in_scope": len(ins),
        "n_out_of_scope": len(outs),
        "retrieval_hit_at_k_pct": pct(sum(hit(g, o) for g, o in ins), len(ins)),
        "in_scope_answered_pct": pct(len(answered), len(ins)),
        "citation_accuracy_pct": pct(sum(cited_right(g, o) for g, o in answered), len(answered)),
        "correct_refusal_pct": pct(sum(o["refused"] for _, o in outs), len(outs)),
        "false_refusal_pct": pct(len(ins) - len(answered), len(ins)),
    }


def sweep(golden: list[dict], store: VectorStore, embeddings) -> list[dict]:
    """Refusal quality at each threshold, using only the top retrieval score."""
    tops = [(g["in_scope"], store.search(embeddings.embed_query(g["question"]), 1)[0].score) for g in golden]
    n_in = sum(1 for i, _ in tops if i)
    n_out = len(tops) - n_in
    rows = []
    for t in [x / 100 for x in range(5, 95, 5)]:
        correct_ref = sum(1 for i, sc in tops if not i and sc < t)
        false_ref = sum(1 for i, sc in tops if i and sc < t)
        rows.append({"threshold": t, "correct_refusal_pct": round(100 * correct_ref / n_out, 1),
                     "false_refusal_pct": round(100 * false_ref / n_in, 1)})
    return rows


def run_ragas(golden: list[dict], outputs: list[dict], s) -> dict:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
    from ragas.run_config import RunConfig

    rows = [
        {"user_input": g["question"], "response": o["answer"],
         "retrieved_contexts": o["contexts"], "reference": g["reference"]}
        for g, o in zip(golden, outputs, strict=True)
        if g["in_scope"] and not o["refused"]
    ]
    if not rows:
        return {}
    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=s.llm_model, google_api_key=s.google_api_key,
                                                       temperature=0))
    emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model=s.embed_model,
                                                                  google_api_key=s.google_api_key))
    result = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextRecall()],
        llm=judge, embeddings=emb,
        run_config=RunConfig(max_workers=2, max_retries=6, timeout=120),  # gentle on free-tier quota
    )
    df = result.to_pandas()
    return {col: round(float(df[col].mean()), 3) for col in df.columns if df[col].dtype.kind == "f"}


def ask_with_retry(svc: RagService, question: str, attempts: int = 4) -> dict:
    """Retry on rate limits (429) and busy servers (503); stop on daily-quota errors."""
    for attempt in range(attempts):
        try:
            return svc.ask(question)
        except Exception as exc:
            msg = str(exc)
            busy = "503" in msg or "UNAVAILABLE" in msg
            if not busy and "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                raise
            wait = 20.0 * (attempt + 1) if busy else retry_wait(exc, attempt)
            if "per day" in msg.lower() or wait > 300 or attempt == attempts - 1:
                raise SystemExit(
                    f"Gemini quota exhausted ({msg[:160]}...). Try later or change LLM_MODEL."
                ) from None
            print(f"  rate limited, waiting {wait:.0f}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--delay", type=float, default=None, help="seconds between questions")
    args = ap.parse_args()

    s = get_settings()
    golden = load_golden(args.golden)
    store, embeddings = VectorStore.load(s.index_dir), get_embeddings(s)

    if args.sweep:
        print(f"{'threshold':>9}  correct_refusal%  false_refusal%")
        for r in sweep(golden, store, embeddings):
            print(f"{r['threshold']:>9.2f}  {r['correct_refusal_pct']:>16}  {r['false_refusal_pct']:>14}")
        return

    svc = RagService(s, store, embeddings, get_llm(s))
    # Free-tier Flash-Lite allows ~10 LLM requests/min; a question can take 2 calls (tool hop).
    delay = args.delay if args.delay is not None else (0 if s.provider == "offline" else 7.0)
    outputs = []
    for i, g in enumerate(golden, 1):
        o = ask_with_retry(svc, g["question"])
        outputs.append(o)
        print(f"[{i}/{len(golden)}] {g['id']} refused={o['refused']} top={o['top_score']}")
        if delay:
            time.sleep(delay)

    report = {"date": date.today().isoformat(), "provider": s.provider,
              "llm_model": s.llm_model if s.provider != "offline" else "extractive-offline",
              "min_score": s.min_score, "top_k": s.top_k, "metrics": score(golden, outputs)}
    if args.ragas:
        report["ragas"] = run_ragas(golden, outputs, s)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{s.provider}.json"
    out.write_text(json.dumps({**report, "per_question": [
        {"id": g["id"], "refused": o["refused"], "top_score": o["top_score"],
         "cited": [c["source"] for c in o["citations"]], "answer": o["answer"]}
        for g, o in zip(golden, outputs, strict=True)]}, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
