"""Generation-side faithfulness (groundedness).

Two stages, each with an offline-deterministic default and an optional Anthropic
swap (one env var, no code change):

1. generate_answer(query, contexts): produce an answer from the retrieved passages.
   - default: an extractive generator that returns the context sentences most
     relevant to the query. Deterministic, no API key.
   - RAG_EVAL_LLM=anthropic: a real grounded generation with claude-opus-4-8.

2. faithfulness_score(answer, contexts): fraction of the answer's claims that are
   supported by the retrieved context.
   - default: a deterministic claim-overlap judge — split the answer into claims,
     mark a claim supported if its content words are sufficiently covered by the
     context. A cheap proxy, good for fast iteration.
   - RAG_EVAL_JUDGE=anthropic: an LLM-as-judge with claude-opus-4-8.

The point the benchmark makes is the *link*: as retrieval recall rises, the answer's
context contains the supporting passage more often, so faithfulness rises too.
"""

from __future__ import annotations

import os
import re

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "be", "by", "it", "as", "at", "that", "this", "with", "from", "which", "not",
    "but", "can", "will", "its", "into", "than", "so", "if", "you", "your", "how",
    "what", "when", "why", "do", "does", "they", "them", "their", "more", "most",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# --- generation ---------------------------------------------------------------

def generate_answer(query: str, contexts: list[str], max_sentences: int = 3) -> str:
    """Answer the query from the retrieved contexts."""
    if os.environ.get("RAG_EVAL_LLM") == "anthropic":
        return _generate_answer_anthropic(query, contexts)
    return _generate_answer_extractive(query, contexts, max_sentences)


def _generate_answer_extractive(query: str, contexts: list[str], max_sentences: int) -> str:
    q = _content_words(query)
    scored: list[tuple[float, int, str]] = []
    for sent in (s for c in contexts for s in _sentences(c)):
        sw = _content_words(sent)
        if not sw:
            continue
        overlap = len(q & sw) / (len(q) or 1)
        scored.append((overlap, -len(sent), sent))
    scored.sort(reverse=True)
    chosen = [s for _, _, s in scored[:max_sentences] if _ > 0] or [
        s for _, _, s in scored[:1]
    ]
    return " ".join(chosen)


def _generate_answer_anthropic(query: str, contexts: list[str]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=400,
        system=(
            "Answer the question using ONLY the provided context passages. "
            "If the answer is not in the context, say you cannot answer from the "
            "provided context. Do not use outside knowledge."
        ),
        messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "").strip()


# --- answer correctness vs gold ----------------------------------------------

def answer_coverage(answer: str, gold_texts: list[str]) -> float:
    """How much of the gold answer the generated answer actually contains (0..1).

    Unlike faithfulness (grounded *in the retrieved context*), this compares the
    answer to the *truth*. It is the offline proxy that tracks retrieval quality:
    when retrieval misses the gold passage, an extractive answer cannot contain the
    gold content, so coverage drops. For multiple gold spans we take the best-covered
    one (the answer only needs to capture one correct passage)."""
    if not gold_texts or not answer.strip():
        return 0.0
    aw = _content_words(answer)
    best = 0.0
    for gt in gold_texts:
        gw = _content_words(gt)
        if not gw:
            continue
        best = max(best, len(aw & gw) / len(gw))
    return best


# --- faithfulness scoring -----------------------------------------------------

def faithfulness_score(answer: str, contexts: list[str], support_threshold: float = 0.6) -> float:
    """Fraction of the answer's claims supported by the context (0..1)."""
    if not answer.strip():
        return 0.0
    if os.environ.get("RAG_EVAL_JUDGE") == "anthropic":
        return _faithfulness_anthropic(answer, contexts)
    return _faithfulness_overlap(answer, contexts, support_threshold)


def _faithfulness_overlap(answer: str, contexts: list[str], support_threshold: float) -> float:
    context_words = set()
    for c in contexts:
        context_words |= _content_words(c)
    claims = _sentences(answer)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        cw = _content_words(claim)
        if not cw:
            supported += 1  # no checkable content -> not a hallucination
            continue
        coverage = len(cw & context_words) / len(cw)
        if coverage >= support_threshold:
            supported += 1
    return supported / len(claims)


def _faithfulness_anthropic(answer: str, contexts: list[str]) -> float:
    import anthropic

    client = anthropic.Anthropic()
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=500,
        system=(
            "You are a strict faithfulness judge for a RAG system. Decompose the "
            "ANSWER into individual factual claims. For each claim, decide whether it "
            "is supported by the CONTEXT. Respond with a JSON object of the exact form "
            '{\"supported\": <int>, \"total\": <int>} and nothing else.'
        ),
        messages=[{"role": "user", "content": f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}"}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "supported": {"type": "integer"},
                        "total": {"type": "integer"},
                    },
                    "required": ["supported", "total"],
                    "additionalProperties": False,
                },
            }
        },
    )
    import json

    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    obj = json.loads(text)
    total = obj.get("total", 0)
    return (obj.get("supported", 0) / total) if total else 0.0
