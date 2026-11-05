import json

import pytest

from app.ingest import clean_markdown, page_url


def test_clean_markdown_strips_site_markup():
    raw = (
        "# First Steps { #first-steps }\n\n{* ../../docs_src/x.py *}\n\n/// tip\n\nUse it.\n\n///\n\n"
        '<div class="termy">\n\n<font color="#fff">run</font>\n\n</div>\n\n![img](a.png) &apos;ok&apos;'
    )
    out = clean_markdown(raw)
    assert out.startswith("# First Steps\n")
    for junk in ("{*", "///", "<div", "<font", "![img]", "&apos;", "{ #"):
        assert junk not in out
    assert "Tip:" in out and "run" in out and "'ok'" in out


def test_page_url():
    base = "https://fastapi.tiangolo.com"
    assert page_url("index.md", base) == base + "/"
    assert page_url("tutorial/index.md", base) == base + "/tutorial/"
    assert page_url("tutorial/first-steps.md", base) == base + "/tutorial/first-steps/"


def test_index_excludes_changelog_and_has_metadata(settings):
    rows = [json.loads(line) for line in open(settings.index_dir / "chunks.jsonl")]
    sources = {r["source"] for r in rows}
    assert "release-notes.md" not in sources
    assert "tutorial/cors.md" in sources
    cors = next(r for r in rows if r["source"] == "tutorial/cors.md")
    assert cors["title"].startswith("CORS")
    assert cors["url"].endswith("/tutorial/cors/")
    meta = json.loads((settings.index_dir / "meta.json").read_text())
    assert meta["provider"] == "offline" and meta["count"] == len(rows)


def test_retry_wait_uses_server_hint():
    from app.ingest import retry_wait

    assert retry_wait(Exception("429 ... Please retry in 52.13s."), 0) == 54.13
    assert retry_wait(Exception("boom"), 1) == 10.0
    assert retry_wait(Exception("Please retry in 14h33m26.3s."), 0) == 14 * 3600 + 33 * 60 + 28.3


def test_daily_quota_stops_immediately():
    from app.ingest import DailyQuotaError, embed_in_batches

    class DailyLimited:
        calls = 0

        def embed_documents(self, texts):
            DailyLimited.calls += 1
            raise RuntimeError("Embed content requests per day limit exceeded. Please retry in 14h33m26s.")

    with pytest.raises(DailyQuotaError, match="resumes"):
        embed_in_batches(DailyLimited(), ["a", "b"], 1, 0)
    assert DailyLimited.calls == 1


def test_gemini_ingest_paces_retries_and_resumes(settings, monkeypatch):
    import app.ingest as ingest
    from app.providers import HashingEmbeddings

    class FlakyEmbeddings(HashingEmbeddings):
        calls = 0

        def embed_documents(self, texts):
            FlakyEmbeddings.calls += 1
            if FlakyEmbeddings.calls == 2:
                raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 0.01s.")
            if FlakyEmbeddings.calls == 4:
                raise KeyboardInterrupt  # simulate the user stopping the run
            return super().embed_documents(texts)

    sleeps = []
    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)
    monkeypatch.setattr(ingest, "get_embeddings", lambda s: FlakyEmbeddings())
    s = settings.model_copy(update={"provider": "gemini", "index_dir": settings.index_dir.parent / "g",
                                    "embed_batch_size": 1, "embed_per_minute": 60})
    with pytest.raises(KeyboardInterrupt):
        ingest.build_index(s)
    cached = len((s.index_dir / f".cache-{s.embed_model}.jsonl").read_text().splitlines())
    assert cached == 2  # two good batches saved before the interruption
    assert 2.01 in [round(x, 2) for x in sleeps]  # waited as the server asked

    FlakyEmbeddings.calls = 10  # no more failures
    store = ingest.build_index(s)
    total = store.meta["count"]
    assert total > cached and FlakyEmbeddings.calls == 10 + total - cached  # only uncached embedded
    assert store.meta["provider"] == "gemini"
