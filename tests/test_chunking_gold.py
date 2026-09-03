"""Chunking-span invariants and gold-snippet resolution."""

import pytest

from rag_eval.corpus import Document, chunk_document
from rag_eval.gold import _resolve_snippet

DOC = Document(
    doc_id="d",
    text="# Title\n\nFirst paragraph with several words here.\n\n"
    "Second paragraph that is a bit longer and wraps\nacross a line for testing.",
)


def test_chunk_spans_reconstruct_source():
    for strategy in ("paragraph", "fixed"):
        for c in chunk_document(DOC, strategy=strategy, size=6, overlap=2):
            # every chunk's text is exactly the slice of the source at its span
            assert DOC.text[c.start:c.end] == c.text
            assert c.start < c.end


def test_paragraph_chunking_counts_blocks():
    chunks = chunk_document(DOC, strategy="paragraph")
    # title + 2 paragraphs = 3 blocks
    assert len(chunks) == 3
    assert chunks[0].text == "# Title"


def test_fixed_chunking_overlaps():
    chunks = chunk_document(DOC, strategy="fixed", size=5, overlap=2)
    assert len(chunks) >= 2
    # consecutive windows advance but overlap (start strictly increases)
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)


def test_snippet_resolution_is_whitespace_insensitive():
    # snippet spans a line break in the source; author writes it as one line
    start, end = _resolve_snippet(DOC.text, "a bit longer and wraps across a line")
    assert "wraps" in DOC.text[start:end]
    assert DOC.text[start:end].startswith("a bit longer")


def test_missing_snippet_raises():
    with pytest.raises(ValueError, match="not found"):
        _resolve_snippet(DOC.text, "this phrase is not in the document")


def test_ambiguous_snippet_raises():
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_snippet(DOC.text, "paragraph")  # appears twice
