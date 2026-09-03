"""Integrity check: every gold snippet in the shipped gold set resolves against the
shipped corpus. Guards against a mislabel or a corpus edit that orphans a label."""

from pathlib import Path

from rag_eval.corpus import load_corpus
from rag_eval.gold import load_gold

ROOT = Path(__file__).resolve().parents[1]


def test_all_gold_snippets_resolve():
    docs = load_corpus(ROOT / "corpus")
    questions = load_gold(ROOT / "gold" / "questions.jsonl", docs)
    assert len(questions) >= 40
    for q in questions:
        assert q.gold, f"{q.id} has no resolved gold spans"
        for span, text in zip(q.gold, q.gold_texts):
            assert span.end > span.start
            assert text.strip()
