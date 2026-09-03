"""Gold evaluation set.

Each question is authored as JSON with the *exact answer snippet(s)* copied from a
source document, rather than raw character offsets, because snippets are what a human
can actually label reliably. At load time each snippet is resolved to a character
span by locating it in the document text. A snippet that cannot be found (or is
ambiguous) raises, which turns a mislabel into a loud failure instead of a silent one.

questions.jsonl schema (one JSON object per line):
    {
      "id": "q001",
      "question": "how do I stop the model from making things up?",
      "answers": [
        {"doc": "faithfulness-groundedness", "snippet": "The common failure mode is hallucination"}
      ],
      "phrasing": "paraphrase",     # "verbatim" | "paraphrase" | "synonym"
      "difficulty": "medium"        # "easy" | "medium" | "hard"
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import Document
from .metrics import Span


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    gold: list[Span]
    gold_texts: list[str] = field(default_factory=list)  # resolved answer snippet strings
    phrasing: str = "paraphrase"
    difficulty: str = "medium"
    meta: dict = field(default_factory=dict)


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to a single space; return the normalised string and
    a map from each normalised-char index to its raw index in ``text``."""
    out: list[str] = []
    idx_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx_map.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx_map.append(i)
            prev_space = False
    return "".join(out), idx_map


def _resolve_snippet(doc_text: str, snippet: str) -> tuple[int, int]:
    """Locate ``snippet`` in ``doc_text`` ignoring whitespace differences (line
    wrapping, double spaces), and return its raw character span. Raises if the snippet
    is missing or ambiguous, so a mislabel fails loudly instead of scoring nothing."""
    doc_norm, idx_map = _normalize_with_map(doc_text)
    snip_norm = " ".join(snippet.split())
    if not snip_norm:
        raise ValueError("empty snippet")
    pos = doc_norm.find(snip_norm)
    if pos == -1:
        raise ValueError(f"snippet not found in document: {snippet!r}")
    if doc_norm.find(snip_norm, pos + 1) != -1:
        raise ValueError(f"snippet is ambiguous (appears more than once): {snippet!r}")
    raw_start = idx_map[pos]
    raw_end = idx_map[pos + len(snip_norm) - 1] + 1
    return raw_start, raw_end


def load_gold(gold_path: str | Path, docs: list[Document]) -> list[GoldQuestion]:
    """Parse questions.jsonl and resolve every answer snippet to a char span."""
    by_id = {d.doc_id: d for d in docs}
    questions: list[GoldQuestion] = []
    with open(gold_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            spans: list[Span] = []
            gold_texts: list[str] = []
            for ans in obj["answers"]:
                doc_id = ans["doc"]
                if doc_id not in by_id:
                    raise ValueError(
                        f"[{gold_path}:{line_no}] unknown doc id {doc_id!r} in question {obj['id']}"
                    )
                start, end = _resolve_snippet(by_id[doc_id].text, ans["snippet"])
                spans.append(Span(doc_id=doc_id, start=start, end=end))
                gold_texts.append(by_id[doc_id].text[start:end])
            questions.append(
                GoldQuestion(
                    id=obj["id"],
                    question=obj["question"],
                    gold=spans,
                    gold_texts=gold_texts,
                    phrasing=obj.get("phrasing", "paraphrase"),
                    difficulty=obj.get("difficulty", "medium"),
                    meta={k: v for k, v in obj.items()
                          if k not in {"id", "question", "answers", "phrasing", "difficulty"}},
                )
            )
    if not questions:
        raise ValueError(f"no questions parsed from {gold_path}")
    return questions
