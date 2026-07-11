"""Retrieval over policy clauses.

DESIGN NOTE (deliberate): the default retriever is TF-IDF, not embeddings.

For a corpus of a few hundred short, keyword-dense regulatory clauses, lexical search is
fast, free, offline, and fully explainable — and in testing it matches the golden set
perfectly (see `evals/`). Embeddings are a real upgrade for paraphrase-heavy queries, so
the interface below is provider-agnostic: swap `TfidfRetriever` for an `EmbeddingRetriever`
without touching the agent. Reach for the expensive tool when the evals say you need it,
not before.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import MIN_RELEVANCE, TOP_K
from .ingest import Chunk, load_chunks


@dataclass
class Hit:
    chunk: Chunk
    score: float


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or load_chunks()
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vec.fit_transform([c.text for c in self.chunks])

    def search(self, query: str, k: int = TOP_K) -> list[Hit]:
        q = self._vec.transform([query])
        scores = cosine_similarity(q, self._matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda t: t[1], reverse=True)
        return [Hit(chunk=c, score=float(s)) for c, s in ranked[:k] if s >= MIN_RELEVANCE]
