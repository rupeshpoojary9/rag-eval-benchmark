"""Retrieval metrics computed against span-level gold labels.

Gold labels are (doc_id, char-span) pairs, not chunk ids, so the same labels stay
valid when the chunking configuration changes (see chunking.py). A retrieved chunk
counts as relevant to a question if it lies in the same document and its character
span overlaps a gold span by at least ``min_overlap`` characters.

Because relevance is defined on gold *spans* rather than chunks, we report three
complementary retrieval metrics that answer genuinely different questions:

* ``hit@k`` (a.k.a. success@k): did *at least one* relevant chunk make the top k?
  This is what matters when a generator is handed the top-k chunks and only needs
  the answer to be present somewhere in that set.
* ``recall@k``: of the distinct gold spans for a question, what fraction have a
  covering chunk in the top k? This is stricter when a question's answer is spread
  across several passages.
* ``mrr@k``: reciprocal rank of the *first* relevant chunk. This is what matters
  when only the single top passage is used (one citation, "feeling lucky").

We also report ``ndcg@k`` with binary per-chunk relevance for ordering quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    """A character span [start, end) inside a document."""

    doc_id: str
    start: int
    end: int

    def overlaps(self, doc_id: str, start: int, end: int, min_overlap: int = 1) -> bool:
        if doc_id != self.doc_id:
            return False
        inter = min(self.end, end) - max(self.start, start)
        return inter >= min_overlap


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk as returned by a retriever, carrying its source span."""

    doc_id: str
    start: int
    end: int


def _relevant_flags(
    ranked: list[RetrievedChunk],
    gold: list[Span],
    min_overlap: int = 1,
) -> list[bool]:
    """For each ranked chunk, True if it overlaps any gold span."""
    return [
        any(g.overlaps(c.doc_id, c.start, c.end, min_overlap) for g in gold)
        for c in ranked
    ]


def hit_at_k(ranked: list[RetrievedChunk], gold: list[Span], k: int, min_overlap: int = 1) -> float:
    """1.0 if any relevant chunk is in the top k, else 0.0."""
    if not gold:
        return 0.0
    flags = _relevant_flags(ranked[:k], gold, min_overlap)
    return 1.0 if any(flags) else 0.0


def recall_at_k(ranked: list[RetrievedChunk], gold: list[Span], k: int, min_overlap: int = 1) -> float:
    """Fraction of distinct gold spans that have a covering chunk in the top k."""
    if not gold:
        return 0.0
    topk = ranked[:k]
    covered = 0
    for g in gold:
        if any(g.overlaps(c.doc_id, c.start, c.end, min_overlap) for c in topk):
            covered += 1
    return covered / len(gold)


def mrr_at_k(ranked: list[RetrievedChunk], gold: list[Span], k: int, min_overlap: int = 1) -> float:
    """Reciprocal rank of the first relevant chunk within the top k (0 if none)."""
    if not gold:
        return 0.0
    flags = _relevant_flags(ranked[:k], gold, min_overlap)
    for i, is_rel in enumerate(flags):
        if is_rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[RetrievedChunk], gold: list[Span], k: int, min_overlap: int = 1) -> float:
    """Binary-relevance nDCG@k. DCG uses log2(rank+1) discounting; IDCG assumes as
    many relevant chunks as we could ideally place, capped at k."""
    if not gold:
        return 0.0
    flags = _relevant_flags(ranked[:k], gold, min_overlap)
    dcg = sum(1.0 / math.log2(i + 2) for i, is_rel in enumerate(flags) if is_rel)
    n_rel = sum(flags)
    ideal_n = min(max(n_rel, 1), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


# Registry so the evaluator can iterate metric-name -> function uniformly.
METRICS = {
    "hit": hit_at_k,
    "recall": recall_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}
