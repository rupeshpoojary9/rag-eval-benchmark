"""Corpus and chunking.

A document's id is its filename stem. Every chunk records the exact character span
``[start, end)`` it occupies in the original document text, so gold spans (defined
once, independent of chunking) can be matched against chunks under any chunk config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str          # f"{doc_id}::{index}"
    doc_id: str
    index: int
    text: str
    start: int             # char offset into the source document
    end: int


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    """Load every .md file in ``corpus_dir`` as a Document, sorted by id for
    determinism."""
    corpus_dir = Path(corpus_dir)
    docs: list[Document] = []
    for path in sorted(corpus_dir.glob("*.md")):
        docs.append(Document(doc_id=path.stem, text=path.read_text(encoding="utf-8")))
    if not docs:
        raise FileNotFoundError(f"No .md documents found in {corpus_dir}")
    return docs


# --- chunking strategies -------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Split on blank lines; return (start, end) char spans of each paragraph."""
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"[^\n].*?(?=\n\s*\n|\Z)", text, re.DOTALL):
        start, end = m.start(), m.end()
        # trim trailing whitespace from the span
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans


def _fixed_word_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Sliding window of ``size`` words with ``overlap`` words of stride overlap.
    Returns char spans covering each window."""
    words = list(_WORD_RE.finditer(text))
    if not words:
        return []
    spans: list[tuple[int, int]] = []
    step = max(1, size - overlap)
    i = 0
    n = len(words)
    while i < n:
        window = words[i : i + size]
        start = window[0].start()
        end = window[-1].end()
        spans.append((start, end))
        if i + size >= n:
            break
        i += step
    return spans


def chunk_document(
    doc: Document,
    strategy: str = "paragraph",
    size: int = 120,
    overlap: int = 20,
) -> list[Chunk]:
    """Chunk one document. ``strategy`` is 'paragraph' (structure-aware) or 'fixed'
    (sliding word window of ``size`` words with ``overlap``)."""
    if strategy == "paragraph":
        spans = _paragraph_spans(doc.text)
    elif strategy == "fixed":
        spans = _fixed_word_spans(doc.text, size=size, overlap=overlap)
    else:
        raise ValueError(f"unknown chunking strategy: {strategy}")

    chunks: list[Chunk] = []
    for idx, (start, end) in enumerate(spans):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{idx}",
                doc_id=doc.doc_id,
                index=idx,
                text=doc.text[start:end],
                start=start,
                end=end,
            )
        )
    return chunks


def chunk_corpus(
    docs: list[Document],
    strategy: str = "paragraph",
    size: int = 120,
    overlap: int = 20,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, strategy=strategy, size=size, overlap=overlap))
    return chunks
