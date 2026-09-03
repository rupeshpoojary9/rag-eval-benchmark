"""Metrics on hand-checked toy inputs — no models, fully deterministic."""

from rag_eval.metrics import (
    RetrievedChunk,
    Span,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def rc(doc, start, end):
    return RetrievedChunk(doc, start, end)


# One gold span in doc "d" covering chars [10, 20).
GOLD = [Span("d", 10, 20)]


def test_span_overlap():
    assert Span("d", 10, 20).overlaps("d", 15, 25)          # partial overlap
    assert not Span("d", 10, 20).overlaps("d", 20, 30)      # touching, no overlap
    assert not Span("d", 10, 20).overlaps("other", 10, 20)  # wrong doc


def test_hit_at_k():
    # relevant chunk sits at rank 3
    ranked = [rc("d", 0, 5), rc("x", 0, 5), rc("d", 12, 18)]
    assert hit_at_k(ranked, GOLD, 3) == 1.0
    assert hit_at_k(ranked, GOLD, 2) == 0.0          # not in top 2
    assert hit_at_k([], GOLD, 5) == 0.0


def test_mrr_at_k():
    ranked = [rc("d", 0, 5), rc("d", 12, 18)]        # first relevant at rank 2
    assert mrr_at_k(ranked, GOLD, 10) == 0.5
    ranked2 = [rc("d", 12, 18), rc("d", 0, 5)]       # relevant at rank 1
    assert mrr_at_k(ranked2, GOLD, 10) == 1.0
    assert mrr_at_k([rc("x", 0, 5)], GOLD, 10) == 0.0


def test_recall_counts_distinct_spans():
    gold = [Span("d", 10, 20), Span("d", 40, 50)]    # two relevant spans
    ranked = [rc("d", 12, 18)]                        # covers only the first
    assert recall_at_k(ranked, gold, 5) == 0.5
    ranked_both = [rc("d", 12, 18), rc("d", 42, 48)]
    assert recall_at_k(ranked_both, gold, 5) == 1.0


def test_ndcg_prefers_higher_rank():
    top = [rc("d", 12, 18), rc("x", 0, 5)]           # relevant at rank 1
    low = [rc("x", 0, 5), rc("d", 12, 18)]           # relevant at rank 2
    assert ndcg_at_k(top, GOLD, 5) == 1.0            # ideal placement
    assert ndcg_at_k(low, GOLD, 5) < 1.0


def test_empty_gold_is_zero():
    assert recall_at_k([rc("d", 12, 18)], [], 5) == 0.0
