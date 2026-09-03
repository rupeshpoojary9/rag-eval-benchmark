# RAG Eval Benchmark

A small, **reproducible** benchmark that compares retrieval strategies over one
hand-labelled corpus and reports real retrieval and answer-quality metrics. It exists
to answer a question most RAG write-ups hand-wave: *does hybrid + reranking actually
retrieve better, and by how much?* — with numbers, not vibes.

It compares four strategies:

```
                      BM25  (lexical / sparse)
corpus ─▶ chunking ─▶ dense (all-MiniLM embeddings)          ─▶ recall@k · MRR · nDCG
                      hybrid (Reciprocal Rank Fusion of both)     └▶ answer ─▶ faithfulness
                      hybrid + rerank (cross-encoder over the top candidates)
```

and reports `recall@k`, `MRR`, `nDCG`, per-question-phrasing recall, and generation-side
**faithfulness**, plus a **chunk-size ablation**.

Everything runs on **local models with no API key** — `sentence-transformers` for dense
retrieval and reranking, a deterministic claim-overlap judge for faithfulness. A real
Anthropic LLM-judge and grounded generator swap in with one environment variable.

## Why the numbers are trustworthy

Two design choices do the heavy lifting, and they're the parts worth defending in an
interview:

1. **A real, hand-labelled gold set.** 40 questions, each mapped to the passage(s) that
   answer it (`gold/questions.jsonl`). Building this is the hard, valuable part — most
   teams skip it and then can't say whether a change helped. Many questions are phrased
   in *user language* (synonyms and paraphrases that don't share words with the source),
   which is exactly where lexical search fails and semantic search earns its keep.

2. **Span-level gold labels, not chunk-id labels.** Each label is a character span in a
   document. A retrieved chunk counts as a hit if it *overlaps* a gold span. Because the
   labels don't reference chunk boundaries, the **same gold set stays valid when chunk
   size changes** — which is what makes the chunk-size ablation honest rather than a
   re-labelling exercise.

`recall@k` and `MRR` are deterministic given the retrieval output, and the retrieval
output is deterministic given the (local, eval-mode) models — so re-running reproduces
the table.

## Quickstart

```bash
uv venv --python 3.12 && source .venv/bin/activate     # or: python -m venv .venv
uv pip install -e ".[dense,plots,dev]"                 # dense+rerank models, plots, tests

rag-eval run --faithfulness --plots                    # full matrix → results/
rag-eval ablation --sizes 60,120,240 --plots           # chunk-size ablation
rag-eval run --no-dense                                # BM25-only path (no torch needed)
```

First run downloads two small models (~80 MB each: `all-MiniLM-L6-v2` and the
`ms-marco-MiniLM` cross-encoder), then runs fully offline.

## What each metric answers

| Metric | Question it answers | When it's the one to watch |
|---|---|---|
| `hit@k` | Is *any* relevant passage in the top k? | You hand the generator the top-k chunks; the answer just needs to be present. |
| `recall@k` | What fraction of the gold spans are covered in the top k? | The answer is spread across several passages. |
| `MRR` | How high is the *first* relevant passage? | Only the single top passage is used (one citation, "feeling lucky"). |
| `nDCG@k` | Ordering quality with positional discount. | You care about the whole ranking, not just presence. |
| `faithfulness` | Fraction of the generated answer's claims grounded in retrieved context. | End-to-end answer quality — retrieval can be good while the answer drifts. |

Reporting `recall@k` *and* `MRR` matters because they can disagree: widening the net can
raise recall@5 while pushing the single best passage down, lowering MRR.

## Layout

```
corpus/           15 original, license-clean AI-engineering explainer docs
gold/             questions.jsonl — 40 hand-labelled questions (span-level gold)
src/rag_eval/     corpus/chunking · retrievers · metrics · gold · faithfulness · evaluate · plots · cli
results/          generated: results.json/.md, ablation.md, *.png
tests/            deterministic tests for metrics, chunking, gold resolution, fusion
RESULTS.md        the written analysis (generated table + what it means)
```

## Swapping in a real LLM (optional)

The offline defaults keep the benchmark reproducible for anyone who clones it. To use a
real model for the generation-side numbers:

```bash
export RAG_EVAL_LLM=anthropic      # grounded answer generation with claude-opus-4-8
export RAG_EVAL_JUDGE=anthropic    # LLM-as-judge faithfulness scoring
export ANTHROPIC_API_KEY=...       # (or an `ant auth login` profile)
rag-eval run --faithfulness
```

Retrieval metrics (`recall@k`, `MRR`, `nDCG`) are unaffected by this switch — it only
changes the models behind the faithfulness column.

## Tests

```bash
pytest            # metrics on known toy inputs, chunk-span invariants, gold resolution
```

## License

MIT.
