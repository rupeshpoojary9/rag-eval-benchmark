# RAG Eval Benchmark — results

- corpus: 17 docs, 95 chunks (chunking: paragraph)
- gold questions: 40

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 | answer_corr | faithfulness |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.375 | 0.637 | 0.738 | 0.838 | 0.538 | 0.614 | 0.418 | 1.000 |
| dense | 0.537 | 0.700 | 0.863 | 0.912 | 0.668 | 0.728 | 0.442 | 1.000 |
| hybrid_rrf | 0.537 | 0.812 | 0.925 | 0.925 | 0.698 | 0.761 | 0.457 | 1.000 |
| hybrid_rrf+rerank | 0.600 | 0.838 | 0.912 | 0.912 | 0.730 | 0.781 | 0.451 | 1.000 |

**Best config:** `hybrid_rrf` — recall@5 0.925 vs BM25 baseline 0.738 (+0.188, 25% relative).

## recall@5 by question phrasing

| strategy | paraphrase | synonym | verbatim |
|---|---|---|---|
| bm25 | 0.750 | 0.556 | 1.000 |
| dense | 0.885 | 0.722 | 1.000 |
| hybrid_rrf | 0.942 | 0.833 | 1.000 |
| hybrid_rrf+rerank | 0.923 | 0.833 | 1.000 |
