"""BM25 and RRF fusion behaviour — deterministic, no dense models needed."""

from rag_eval.corpus import Chunk
from rag_eval.retrievers import BM25Retriever, HybridRetriever


def _chunks(texts):
    return [Chunk(f"d::{i}", "d", i, t, 0, len(t)) for i, t in enumerate(texts)]


CHUNKS = _chunks([
    "reciprocal rank fusion combines two ranked lists",
    "bm25 scores passages by exact term frequency",
    "cross encoder rerankers reorder a shortlist of candidates",
    "dense retrieval embeds text into vectors for semantic search",
])


def test_bm25_finds_lexical_match():
    bm25 = BM25Retriever(CHUNKS)
    ranked = bm25.rank("term frequency", top_k=1)
    assert ranked[0] == 1  # the bm25 chunk


def test_bm25_ranking_is_deterministic():
    bm25 = BM25Retriever(CHUNKS)
    assert bm25.rank("dense vectors semantic") == bm25.rank("dense vectors semantic")


class _FakeScorer:
    """Stand-in first-stage retriever with fixed per-chunk scores."""

    def __init__(self, chunks, scores):
        self.chunks = chunks
        self._scores = scores
        self.name = "fake"

    def scores(self, query):
        import numpy as np

        return np.asarray(self._scores, dtype=float)

    def rank(self, query, top_k=None):
        from rag_eval.retrievers import _rank_from_scores

        return _rank_from_scores(self.scores(query), top_k)


def test_rrf_promotes_consensus():
    # sparse ranks chunk 0 top; dense ranks chunk 3 top; chunk 2 is second in both
    sparse = _FakeScorer(CHUNKS, [9, 1, 5, 0])
    dense = _FakeScorer(CHUNKS, [0, 1, 5, 9])
    hybrid = HybridRetriever(sparse, dense, method="rrf", rrf_k=60)
    ranked = hybrid.rank("q", top_k=4)
    # chunk 2 (second in both lists) should outrank chunks each list ranks last
    assert ranked.index(2) < ranked.index(1)


def test_weighted_alpha_extremes_match_components():
    sparse = _FakeScorer(CHUNKS, [9, 1, 5, 0])
    dense = _FakeScorer(CHUNKS, [0, 1, 5, 9])
    pure_sparse = HybridRetriever(sparse, dense, method="weighted", alpha=0.0)
    pure_dense = HybridRetriever(sparse, dense, method="weighted", alpha=1.0)
    assert pure_sparse.rank("q", top_k=1)[0] == 0   # follows sparse
    assert pure_dense.rank("q", top_k=1)[0] == 3    # follows dense
