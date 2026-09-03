"""Matplotlib plots for the benchmark. Imported lazily by the CLI so the core has no
hard matplotlib dependency."""

from __future__ import annotations

from pathlib import Path

from .evaluate import RunResult


def _savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")


def plot_recall_curves(result: RunResult, out_path: str | Path):
    import matplotlib.pyplot as plt

    ks = list(result.config.ks)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in result.strategies:
        ys = [s.metrics[f"recall@{k}"] for k in ks]
        ax.plot(ks, ys, marker="o", label=s.name)
    ax.set_xlabel("k")
    ax.set_ylabel("recall@k")
    ax.set_title("Recall@k by retrieval strategy")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _savefig(fig, Path(out_path))
    plt.close(fig)


def plot_mrr_bars(result: RunResult, out_path: str | Path, k: int = 10):
    import matplotlib.pyplot as plt

    names = [s.name for s in result.strategies]
    vals = [s.metrics[f"mrr@{k}"] for s in result.strategies]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, vals, color="#4C78A8")
    ax.set_ylabel(f"MRR@{k}")
    ax.set_title(f"Mean Reciprocal Rank (MRR@{k}) by strategy")
    ax.set_ylim(0, 1.02)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, Path(out_path))
    plt.close(fig)


def plot_answer_quality(result: RunResult, out_path: str | Path):
    """Answer correctness (vs gold) and faithfulness (vs context) side by side.

    The point is the contrast: correctness rises with retrieval quality, while
    faithfulness-to-context stays ~1.0 for an extractive generator that can only
    quote what it was given."""
    import matplotlib.pyplot as plt
    import numpy as np

    strat = [s for s in result.strategies if s.faithfulness is not None]
    if not strat:
        return
    names = [s.name for s in strat]
    corr = [s.answer_correctness or 0.0 for s in strat]
    faith = [s.faithfulness or 0.0 for s in strat]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b1 = ax.bar(x - w / 2, corr, w, label="answer correctness (vs gold)", color="#4C78A8")
    b2 = ax.bar(x + w / 2, faith, w, label="faithfulness (vs context)", color="#E45756")
    ax.set_ylabel("score")
    ax.set_title("Answer quality by retrieval strategy")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _savefig(fig, Path(out_path))
    plt.close(fig)


def plot_chunk_ablation(ablation: dict[int, dict[str, float]], out_path: str | Path, k: int = 5):
    import matplotlib.pyplot as plt

    sizes = sorted(ablation)
    strategies = list(next(iter(ablation.values())).keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in strategies:
        ys = [ablation[s][name] for s in sizes]
        ax.plot(sizes, ys, marker="o", label=name)
    ax.set_xlabel("chunk size (words, fixed-window chunking)")
    ax.set_ylabel(f"recall@{k}")
    ax.set_title(f"Chunk-size ablation: recall@{k} vs chunk size")
    ax.set_xticks(sizes)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _savefig(fig, Path(out_path))
    plt.close(fig)
