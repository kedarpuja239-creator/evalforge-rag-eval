"""Corpus -> chunks -> embeddings -> index, and the Retriever that searches it.

    python -m evalkit.index        # build and save the index (data/index/)

Everything here is the code from steps 1 to 4, tidied: load_documents = step1, build_chunks = step2,
embed = step3, Retriever.search = step4.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from . import chunking, config


@dataclass
class Chunk:
    source_id: str  # corpus file name without .txt: the retrieval ground-truth unit
    text: str


def load_documents() -> list[dict]:
    docs = []
    for path in sorted(config.CORPUS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        first_line, _, body = text.partition("\n")
        docs.append({"id": path.stem, "source": first_line.replace("SOURCE:", "").strip(), "text": body.strip()})
    return docs


def build_chunks(strategy: str | None = None, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    strategy = strategy or config.CHUNK_STRATEGY
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    chunks: list[Chunk] = []
    for doc in load_documents():
        for piece in chunking.chunk(doc["text"], size, overlap, strategy):
            chunks.append(Chunk(doc["id"], piece))
    return chunks


_model = None


def embed(texts: list[str]) -> np.ndarray:
    """Local embedding model: no API key. Vectors have length 1, so dot product == cosine similarity."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBED_MODEL)
    return np.asarray(_model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


class Retriever:
    """Holds the vectors and chunks. `self` is how search() reaches them without reloading each time."""

    def __init__(self, vectors: np.ndarray, chunks: list[Chunk], embed_fn: Callable = embed) -> None:
        self.vectors = vectors
        self.chunks = chunks
        self.embed_fn = embed_fn

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], embed_fn: Callable = embed) -> "Retriever":
        return cls(embed_fn([c.text for c in chunks]), chunks, embed_fn)

    @classmethod
    def load(cls) -> "Retriever":
        vectors = np.load(config.INDEX_DIR / "vectors.npy")
        chunks = [Chunk(**c) for c in json.loads((config.INDEX_DIR / "chunks.json").read_text("utf-8"))]
        return cls(vectors, chunks)

    def save(self) -> None:
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(config.INDEX_DIR / "vectors.npy", self.vectors)
        (config.INDEX_DIR / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks]), encoding="utf-8")

    def search(self, query: str, k: int | None = None) -> list[tuple[float, Chunk]]:
        k = k or config.TOP_K
        scores = self.vectors @ self.embed_fn([query])[0]
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in top]


if __name__ == "__main__":
    chunks = build_chunks()
    Retriever.from_chunks(chunks).save()
    print(f"index built: {len(chunks)} chunks ({config.CHUNK_STRATEGY}, size {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP})")
