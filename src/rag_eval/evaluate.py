"""Run the experiment matrix and aggregate metrics.

Ties corpus + chunking + retrievers + gold + metrics together, runs every strategy
over every gold question, and returns a results structure that the CLI serialises to
JSON / Markdown and the plotter renders.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .corpus import Chunk, Document, chunk_corpus, load_corpus
from .faithfulness import answer_coverage, faithfulness_score, generate_answer
from .gold import GoldQuestion, load_gold
from .metrics import METRICS, RetrievedChunk
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever

DEFAULT_KS = (1, 3, 5, 10)


def _to_retrieved(chunks: list[Chunk], ranked_idx: list[int]) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunks[i].doc_id, chunks[i].start, chunks[i].end) for i in ranked_idx]


@dataclass
class StrategyResult:
    name: str
    metrics: dict[str, float] = field(default_factory=dict)          # "recall@5" -> mean
    by_phrasing: dict[str, float] = field(default_factory=dict)      # phrasing -> recall@5
    faithfulness: float | None = None          # answer grounded in retrieved context
    answer_correctness: float | None = None    # answer matches the gold answer


def build_retrievers(
    chunks: list[Chunk],
    *,
    include_dense: bool = True,
    include_rerank: bool = True,
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rrf_k: int = 60,
    rerank_candidates: int = 30,
) -> list:
    """Construct the strategy list in reporting order. Dense/rerank are optional so
    the BM25-only path needs no torch."""
    bm25 = BM25Retriever(chunks)
    retrievers = [bm25]
    if include_dense:
        dense = DenseRetriever(chunks, model_name=dense_model)
        hybrid = HybridRetriever(bm25, dense, method="rrf", rrf_k=rrf_k)
        retrievers += [dense, hybrid]
        if include_rerank:
            retrievers.append(
                RerankRetriever(hybrid, chunks, model_name=rerank_model, candidates=rerank_candidates)
            )
    return retrievers


def evaluate_strategy(
    retriever,
    questions: list[GoldQuestion],
    chunks: list[Chunk],
    ks: tuple[int, ...] = DEFAULT_KS,
    min_overlap: int = 1,
    with_faithfulness: bool = False,
    faithfulness_k: int = 5,
) -> StrategyResult:
    max_k = max(ks)
    # accumulate per-metric-per-k lists, plus per-phrasing recall@5
    acc: dict[str, list[float]] = {f"{m}@{k}": [] for m in METRICS for k in ks}
    phrasing_acc: dict[str, list[float]] = {}
    faith_scores: list[float] = []
    correctness_scores: list[float] = []

    for q in questions:
        ranked_idx = retriever.rank(q.question, top_k=max_k)
        ranked = _to_retrieved(chunks, ranked_idx)
        for m_name, m_fn in METRICS.items():
            for k in ks:
                acc[f"{m_name}@{k}"].append(m_fn(ranked, q.gold, k, min_overlap))
        # recall@5 broken out by question phrasing, for the analysis section
        r5 = METRICS["recall"](ranked, q.gold, 5, min_overlap)
        phrasing_acc.setdefault(q.phrasing, []).append(r5)

        if with_faithfulness:
            contexts = [chunks[i].text for i in ranked_idx[:faithfulness_k]]
            answer = generate_answer(q.question, contexts)
            faith_scores.append(faithfulness_score(answer, contexts))
            correctness_scores.append(answer_coverage(answer, q.gold_texts))

    result = StrategyResult(name=retriever.name)
    result.metrics = {key: statistics.mean(vals) for key, vals in acc.items()}
    result.by_phrasing = {p: statistics.mean(v) for p, v in phrasing_acc.items()}
    if with_faithfulness and faith_scores:
        result.faithfulness = statistics.mean(faith_scores)
        result.answer_correctness = statistics.mean(correctness_scores)
    return result


@dataclass
class RunConfig:
    corpus_dir: str
    gold_path: str
    strategy: str = "paragraph"       # chunking strategy
    chunk_size: int = 120
    overlap: int = 20
    ks: tuple[int, ...] = DEFAULT_KS
    include_dense: bool = True
    include_rerank: bool = True
    with_faithfulness: bool = False


@dataclass
class RunResult:
    config: RunConfig
    n_docs: int
    n_chunks: int
    n_questions: int
    strategies: list[StrategyResult]


def run(config: RunConfig) -> RunResult:
    docs: list[Document] = load_corpus(config.corpus_dir)
    chunks = chunk_corpus(docs, strategy=config.strategy, size=config.chunk_size, overlap=config.overlap)
    questions = load_gold(config.gold_path, docs)
    retrievers = build_retrievers(
        chunks, include_dense=config.include_dense, include_rerank=config.include_rerank
    )
    strategies = [
        evaluate_strategy(
            r, questions, chunks, ks=config.ks, with_faithfulness=config.with_faithfulness
        )
        for r in retrievers
    ]
    return RunResult(
        config=config,
        n_docs=len(docs),
        n_chunks=len(chunks),
        n_questions=len(questions),
        strategies=strategies,
    )


def chunk_size_ablation(
    corpus_dir: str,
    gold_path: str,
    sizes: list[int],
    overlap: int = 20,
    k: int = 5,
    include_dense: bool = True,
    include_rerank: bool = False,
) -> dict[int, dict[str, float]]:
    """recall@k per strategy across chunk sizes (fixed-window chunking).
    Returns {chunk_size: {strategy_name: recall@k}}."""
    docs = load_corpus(corpus_dir)
    questions = load_gold(gold_path, docs)
    out: dict[int, dict[str, float]] = {}
    for size in sizes:
        chunks = chunk_corpus(docs, strategy="fixed", size=size, overlap=overlap)
        retrievers = build_retrievers(
            chunks, include_dense=include_dense, include_rerank=include_rerank
        )
        row: dict[str, float] = {}
        for r in retrievers:
            res = evaluate_strategy(r, questions, chunks, ks=(k,))
            row[r.name] = res.metrics[f"recall@{k}"]
        out[size] = row
    return out
