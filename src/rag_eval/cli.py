"""Command-line entry point: run the benchmark, write a results table + JSON, and
optionally render plots and the chunk-size ablation.

    rag-eval run --faithfulness --plots
    rag-eval ablation --sizes 60,120,240
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .evaluate import DEFAULT_KS, RunConfig, chunk_size_ablation, run

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]  # repo root (…/rag-eval-benchmark)
DEFAULT_CORPUS = str(_ROOT / "corpus")
DEFAULT_GOLD = str(_ROOT / "gold" / "questions.jsonl")
DEFAULT_RESULTS = str(_ROOT / "results")


def _results_markdown(result) -> str:
    ks = list(result.config.ks)
    lines = []
    lines.append(f"# RAG Eval Benchmark — results\n")
    lines.append(
        f"- corpus: {result.n_docs} docs, {result.n_chunks} chunks "
        f"(chunking: {result.config.strategy}"
        + (f", size={result.config.chunk_size}, overlap={result.config.overlap}"
           if result.config.strategy == "fixed" else "")
        + ")"
    )
    lines.append(f"- gold questions: {result.n_questions}\n")

    # main table: recall@k + mrr@10 + ndcg@10 (+ faithfulness if present)
    header = ["strategy"] + [f"recall@{k}" for k in ks] + ["mrr@10", "ndcg@10"]
    has_faith = any(s.faithfulness is not None for s in result.strategies)
    if has_faith:
        header += ["answer_corr", "faithfulness"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for s in result.strategies:
        row = [s.name] + [f"{s.metrics[f'recall@{k}']:.3f}" for k in ks]
        row += [f"{s.metrics['mrr@10']:.3f}", f"{s.metrics['ndcg@10']:.3f}"]
        if has_faith:
            row.append(f"{s.answer_correctness:.3f}" if s.answer_correctness is not None else "-")
            row.append(f"{s.faithfulness:.3f}" if s.faithfulness is not None else "-")
        lines.append("| " + " | ".join(row) + " |")

    # lift over BM25 baseline
    base = next((s for s in result.strategies if s.name == "bm25"), None)
    best = max(result.strategies, key=lambda s: s.metrics["recall@5"])
    if base and best.name != "bm25":
        b, x = base.metrics["recall@5"], best.metrics["recall@5"]
        lift = (x - b)
        lines.append(
            f"\n**Best config:** `{best.name}` — recall@5 {x:.3f} vs BM25 baseline "
            f"{b:.3f} (+{lift:.3f}, {lift / b * 100:.0f}% relative)." if b else ""
        )

    # recall@5 broken out by question phrasing (the discrimination story)
    phrasings = sorted({p for s in result.strategies for p in s.by_phrasing})
    if phrasings:
        lines.append("\n## recall@5 by question phrasing\n")
        lines.append("| strategy | " + " | ".join(phrasings) + " |")
        lines.append("|" + "|".join(["---"] * (len(phrasings) + 1)) + "|")
        for s in result.strategies:
            row = [s.name] + [
                f"{s.by_phrasing.get(p, float('nan')):.3f}" for p in phrasings
            ]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _serialise(result) -> dict:
    d = {
        "config": asdict(result.config),
        "n_docs": result.n_docs,
        "n_chunks": result.n_chunks,
        "n_questions": result.n_questions,
        "strategies": [asdict(s) for s in result.strategies],
    }
    d["config"]["ks"] = list(result.config.ks)
    return d


def _cmd_run(args):
    config = RunConfig(
        corpus_dir=args.corpus,
        gold_path=args.gold,
        strategy=args.chunking,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        ks=tuple(int(k) for k in args.ks.split(",")),
        include_dense=not args.no_dense,
        include_rerank=not args.no_rerank,
        with_faithfulness=args.faithfulness,
    )
    print(f"Running benchmark (chunking={config.strategy}, dense={config.include_dense}, "
          f"rerank={config.include_rerank}, faithfulness={config.with_faithfulness}) …")
    result = run(config)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(_serialise(result), indent=2))
    md = _results_markdown(result)
    (out / "results.md").write_text(md)
    print("\n" + md)
    print(f"Wrote {out/'results.json'} and {out/'results.md'}")

    if args.plots:
        from . import plots

        plots.plot_recall_curves(result, out / "recall_at_k.png")
        plots.plot_mrr_bars(result, out / "mrr.png")
        if config.with_faithfulness:
            plots.plot_answer_quality(result, out / "answer_quality.png")
        print(f"Wrote plots to {out}/")


def _cmd_ablation(args):
    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"Running chunk-size ablation over sizes {sizes} (recall@{args.k}) …")
    ablation = chunk_size_ablation(
        args.corpus, args.gold, sizes=sizes, overlap=args.overlap, k=args.k,
        include_dense=not args.no_dense, include_rerank=not args.no_rerank,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(ablation, indent=2))
    # markdown table
    strategies = list(next(iter(ablation.values())).keys())
    lines = [f"# Chunk-size ablation (recall@{args.k})\n",
             "| chunk size | " + " | ".join(strategies) + " |",
             "|" + "|".join(["---"] * (len(strategies) + 1)) + "|"]
    for size in sizes:
        lines.append("| " + str(size) + " | "
                     + " | ".join(f"{ablation[size][s]:.3f}" for s in strategies) + " |")
    md = "\n".join(lines) + "\n"
    (out / "ablation.md").write_text(md)
    print("\n" + md)
    if args.plots:
        from . import plots

        plots.plot_chunk_ablation(ablation, out / "chunk_ablation.png", k=args.k)
        print(f"Wrote plot to {out}/chunk_ablation.png")


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag-eval", description="Reproducible RAG retrieval benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the strategy matrix and write results")
    r.add_argument("--corpus", default=DEFAULT_CORPUS)
    r.add_argument("--gold", default=DEFAULT_GOLD)
    r.add_argument("--out", default=DEFAULT_RESULTS)
    r.add_argument("--chunking", default="paragraph", choices=["paragraph", "fixed"])
    r.add_argument("--chunk-size", type=int, default=120)
    r.add_argument("--overlap", type=int, default=20)
    r.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    r.add_argument("--no-dense", action="store_true", help="BM25 only (no torch)")
    r.add_argument("--no-rerank", action="store_true")
    r.add_argument("--faithfulness", action="store_true", help="also score answer faithfulness")
    r.add_argument("--plots", action="store_true", help="render matplotlib plots")
    r.set_defaults(func=_cmd_run)

    a = sub.add_parser("ablation", help="chunk-size ablation (recall@k vs chunk size)")
    a.add_argument("--corpus", default=DEFAULT_CORPUS)
    a.add_argument("--gold", default=DEFAULT_GOLD)
    a.add_argument("--out", default=DEFAULT_RESULTS)
    a.add_argument("--sizes", default="60,120,240")
    a.add_argument("--overlap", type=int, default=20)
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--no-dense", action="store_true")
    a.add_argument("--no-rerank", action="store_true")
    a.add_argument("--plots", action="store_true")
    a.set_defaults(func=_cmd_ablation)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
