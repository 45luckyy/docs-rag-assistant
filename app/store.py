"""Minimal FAISS store: cosine similarity over L2-normalised vectors."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # path relative to the corpus, e.g. tutorial/first-steps.md
    title: str
    section: str
    url: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


def _normalise(vectors: list[list[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], meta: dict):
        self.index, self.chunks, self.meta = index, chunks, meta

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: list[list[float]], meta: dict) -> "VectorStore":
        arr = _normalise(vectors)
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        return cls(index, chunks, {**meta, "dim": arr.shape[1], "count": len(chunks)})

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        if not (directory / "index.faiss").exists():
            raise FileNotFoundError(f"No index in {directory}. Run `python -m app.ingest` first.")
        index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        meta = json.loads((directory / "meta.json").read_text())
        return cls(index, chunks, meta)

    def search(self, query_vector: list[float], k: int) -> list[Hit]:
        scores, ids = self.index.search(_normalise([query_vector]), k)
        return [Hit(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i != -1]
