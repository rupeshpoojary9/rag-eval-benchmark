"""Retrieval strategies: BM25, dense, hybrid (RRF + weighted), and cross-encoder rerank.

Every first-stage retriever exposes ``scores(query) -> np.ndarray`` giving a score per
chunk (higher is better), and ``rank(query, top_k) -> list[int]`` returning chunk
indices best-first. Hybrid retrievers fuse two first-stage rankings; the reranker wraps
any base retriever and reorders its top candidates with a cross-encoder.

Dense and reranker models are local sentence-transformers, loaded lazily so the BM25
path needs no torch. All scoring is deterministic (models run in eval mode, no sampling).
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .corpus import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _rank_from_scores(scores: np.ndarray, top_k: int | None) -> list[int]:
    """Indices sorted by score desc; ties broken by index asc for determinism."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    order = order.tolist()
    return order if top_k is None else order[:top_k]


# --- first-stage retrievers ----------------------------------------------------

class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(_tokenize(query)), dtype=float)

    def rank(self, query: str, top_k: int | None = None) -> list[int]:
        return _rank_from_scores(self.scores(query), top_k)


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: Sequence[Chunk], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.chunks = list(chunks)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # normalized embeddings -> cosine similarity is a dot product
        self._emb = self._model.encode(
            [c.text for c in self.chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def scores(self, query: str) -> np.ndarray:
        q = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)[0]
        return self._emb @ q

    def rank(self, query: str, top_k: int | None = None) -> list[int]:
        return _rank_from_scores(self.scores(query), top_k)


# --- fusion --------------------------------------------------------------------

def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridRetriever:
    """Fuse two first-stage retrievers.

    method='rrf'      : Reciprocal Rank Fusion, score-scale-free, robust default.
    method='weighted' : min-max normalise each score vector, combine as
                        alpha*dense + (1-alpha)*sparse. alpha=1 is pure dense.
    """

    def __init__(self, sparse, dense, method: str = "rrf", rrf_k: int = 60, alpha: float = 0.5):
        assert method in {"rrf", "weighted"}
        self.sparse = sparse
        self.dense = dense
        self.method = method
        self.rrf_k = rrf_k
        self.alpha = alpha
        self.name = f"hybrid_{method}" + (f"_a{alpha}" if method == "weighted" else "")
        self.chunks = sparse.chunks

    def rank(self, query: str, top_k: int | None = None) -> list[int]:
        s_scores = self.sparse.scores(query)
        d_scores = self.dense.scores(query)
        n = len(s_scores)
        if self.method == "weighted":
            fused = self.alpha * _minmax(d_scores) + (1.0 - self.alpha) * _minmax(s_scores)
            return _rank_from_scores(fused, top_k)
        # RRF: sum of 1/(k + rank) across both ranked lists
        fused = np.zeros(n, dtype=float)
        for scores in (s_scores, d_scores):
            for rank, idx in enumerate(_rank_from_scores(scores, None)):
                fused[idx] += 1.0 / (self.rrf_k + rank + 1)
        return _rank_from_scores(fused, top_k)


# --- reranking -----------------------------------------------------------------

class RerankRetriever:
    """Wrap a base retriever: take its top ``candidates``, reorder with a
    cross-encoder, and return the reranked list."""

    def __init__(
        self,
        base,
        chunks: Sequence[Chunk],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidates: int = 30,
    ):
        from sentence_transformers import CrossEncoder

        self.base = base
        self.chunks = list(chunks)
        self.candidates = candidates
        self.model_name = model_name
        self._ce = CrossEncoder(model_name)
        self.name = f"{base.name}+rerank"

    def rank(self, query: str, top_k: int | None = None) -> list[int]:
        cand = self.base.rank(query, top_k=self.candidates)
        if not cand:
            return []
        pairs = [(query, self.chunks[i].text) for i in cand]
        ce_scores = np.asarray(self._ce.predict(pairs, show_progress_bar=False), dtype=float)
        order = np.lexsort((np.arange(len(cand)), -ce_scores)).tolist()
        reranked = [cand[i] for i in order]
        return reranked if top_k is None else reranked[:top_k]
