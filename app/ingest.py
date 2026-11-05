"""Load markdown docs -> clean -> chunk -> embed -> save FAISS index.

Usage: python -m app.ingest
"""

import hashlib
import html
import json
import logging
import re
import time

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.providers import get_embeddings
from app.store import Chunk, VectorStore

log = logging.getLogger(__name__)

# Pages that are changelogs, people lists or site plumbing, not documentation.
EXCLUDE = {
    "release-notes.md", "fastapi-people.md", "management.md", "management-tasks.md",
    "translations.md", "translation-banner.md", "_llm-test.md", "contributing.md",
    "external-links.md", "newsletter.md", "help-fastapi.md",
}

_RULES = [
    (re.compile(r"^\{\*.*?\*\}\s*$", re.M), ""),            # code-file includes
    (re.compile(r"^[ \t]*/{3,4}\s*(tip|note|warning|info|check|danger)\b.*$", re.M | re.I),
     lambda m: m.group(1).capitalize() + ":"),               # admonition openers
    (re.compile(r"^[ \t]*/{3,4}.*$", re.M), ""),                 # admonition closers / tabs
    (re.compile(r"<br\s*/?>"), " "),
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),               # images
    (re.compile(r"</?[a-zA-Z][^>]*>"), ""),                  # html tags
    (re.compile(r"\s*\{\s*#[\w-]+\s*\}"), ""),               # header anchors { #id }
    (re.compile(r"\n{3,}"), "\n\n"),
]


def clean_markdown(text: str) -> str:
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    return html.unescape(text).strip()


def page_url(rel_path: str, base_url: str) -> str:
    slug = rel_path.removesuffix(".md")
    if slug == "index":
        return base_url + "/"
    slug = slug.removesuffix("/index")
    return f"{base_url}/{slug}/"


def load_chunks(s: Settings) -> list[Chunk]:
    header_split = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
    )
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap,
        separators=["\n## ", "\n### ", "\n```\n", "\n\n", "\n", " "],
    )
    chunks: list[Chunk] = []
    files = sorted(p for p in s.docs_dir.rglob("*.md") if p.name not in EXCLUDE)
    if not files:
        raise FileNotFoundError(f"No markdown in {s.docs_dir}. Run scripts/fetch_corpus.sh first.")

    for path in files:
        rel = path.relative_to(s.docs_dir).as_posix()
        text = clean_markdown(path.read_text(encoding="utf-8"))
        if len(text) < 200:  # section index pages with no content
            continue
        title_match = re.search(r"^# (.+)$", text, re.M)
        title = title_match.group(1).strip() if title_match else rel
        for section_doc in header_split.split_text(text):
            section = section_doc.metadata.get("h3") or section_doc.metadata.get("h2") or title
            for piece in text_split.split_text(section_doc.page_content):
                if len(piece) < 80:
                    continue
                cid = hashlib.sha1(f"{rel}:{piece}".encode()).hexdigest()[:12]
                chunks.append(Chunk(cid, piece, rel, title, section, page_url(rel, s.docs_base_url)))
    return chunks


_RETRY_IN = re.compile(r"retry in ((?:[\d.]+[hms])+)", re.I)
_UNIT_S = {"h": 3600, "m": 60, "s": 1}


class DailyQuotaError(RuntimeError):
    """The provider's per-day quota is used up; retrying today won't help."""


def retry_wait(exc: Exception, attempt: int) -> float:
    """Use the server's suggested delay when it gives one (Gemini 429s do)."""
    match = _RETRY_IN.search(str(exc))
    if not match:
        return min(60.0, 5.0 * 2**attempt)
    parts = re.findall(r"([\d.]+)([hms])", match.group(1))
    return sum(float(n) * _UNIT_S[u] for n, u in parts) + 2


def embed_in_batches(embeddings, texts: list[str], batch_size: int, per_minute: int,
                     on_batch=None) -> list[list[float]]:
    """Embed in batches, paced to `per_minute` texts (0 = no pacing)."""
    vectors: list[list[float]] = []
    interval = 60 * batch_size / per_minute if per_minute else 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        began = time.monotonic()
        for attempt in range(8):
            try:
                out = embeddings.embed_documents(batch)
                break
            except Exception as exc:  # rate limits (429) are the usual cause
                wait = retry_wait(exc, attempt)
                if "per day" in str(exc).lower() or wait > 300:
                    raise DailyQuotaError(
                        f"Daily embedding quota used up; retry in about {wait / 3600:.1f} h. "
                        f"Progress so far is cached, so rerunning resumes from here."
                    ) from exc
                if attempt == 7:
                    raise
                log.warning("Embedding batch failed (%s...); retrying in %.0fs", str(exc)[:120], wait)
                time.sleep(wait)
        vectors.extend(out)
        if on_batch:
            on_batch(start, out)
        log.info("Embedded %d/%d chunks", len(vectors), len(texts))
        if interval and start + batch_size < len(texts):
            time.sleep(max(0.0, interval - (time.monotonic() - began)))
    return vectors


def build_index(s: Settings) -> VectorStore:
    chunks = load_chunks(s)
    log.info("Loaded %d chunks from %s", len(chunks), s.docs_dir)
    embeddings = get_embeddings(s)
    offline = s.provider == "offline"
    texts = [f"{c.title} > {c.section}\n{c.text}" for c in chunks]  # headings help retrieval

    # Cache vectors per model so an interrupted run resumes instead of starting over.
    cache: dict[str, list[float]] = {}
    safe_model = re.sub(r"[^\w.-]", "_", s.embed_model)
    cache_file = s.index_dir / f".cache-{safe_model}.jsonl"
    if not offline and cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[row["id"]] = row["v"]
    todo = [i for i, c in enumerate(chunks) if c.id not in cache]
    if not offline:
        log.info("%d cached, %d to embed (~%.0f min at %d/min)", len(chunks) - len(todo), len(todo),
                 len(todo) / s.embed_per_minute, s.embed_per_minute)

    def save_batch(start: int, out: list[list[float]]) -> None:
        if offline:
            return
        s.index_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "a", encoding="utf-8") as f:
            for i, vec in zip(todo[start : start + len(out)], out, strict=True):
                cache[chunks[i].id] = vec
                f.write(json.dumps({"id": chunks[i].id, "v": vec}) + "\n")

    new = embed_in_batches(embeddings, [texts[i] for i in todo], s.embed_batch_size,
                           0 if offline else s.embed_per_minute, save_batch)
    if offline:
        cache.update({chunks[i].id: v for i, v in zip(todo, new, strict=True)})
    vectors = [cache[c.id] for c in chunks]

    meta = {"provider": s.provider, "embed_model": "hashing" if offline else s.embed_model}
    store = VectorStore.build(chunks, vectors, meta)
    store.save(s.index_dir)
    log.info("Saved index (%d chunks, dim %d) to %s", len(chunks), store.meta["dim"], s.index_dir)
    return store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HTTP call is noise
    try:
        build_index(get_settings())
    except DailyQuotaError as exc:
        log.error("%s", exc)
        raise SystemExit(1) from None
