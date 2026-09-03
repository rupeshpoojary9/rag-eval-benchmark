"""A reproducible RAG retrieval benchmark.

Compares BM25, dense, hybrid (RRF), and hybrid+rerank retrieval over one
hand-labelled corpus and reports recall@k, MRR, nDCG, plus generation-side
faithfulness. Runs on local models with no API key.
"""

__version__ = "0.1.0"
