# Chunking Strategies

Chunking is the step that splits source documents into the passages that get embedded,
indexed, and retrieved. It is one of the highest-leverage and most underrated decisions
in a retrieval system, because a chunk is the unit of everything downstream: if the
answer is split across two chunks, no single retrieval can contain it, and if a chunk is
bloated with unrelated text, its embedding is diluted and its relevance signal weakens.

The simplest approach is fixed-size chunking, which cuts text every N tokens or
characters, usually with a small overlap between consecutive chunks so a sentence that
straddles a boundary is not lost. It is trivial to implement and predictable, but it
cuts blindly through the middle of sentences, tables, and ideas. Structure-aware
chunking instead splits on natural boundaries such as headings, paragraphs, list items,
or sentences, which keeps each chunk semantically coherent at the cost of variable chunk
sizes.

Chunk size is a genuine trade-off with no universal answer. Small chunks give precise,
tightly-scoped matches and a high relevance-to-noise ratio, but they fragment context and
may omit the surrounding detail needed to actually answer. Large chunks preserve context
and reduce the risk of splitting an answer, but they dilute the embedding, retrieve more
irrelevant text alongside the relevant part, and consume more of the generation model's
context budget and cost. Overlap softens boundary effects but inflates the index with
near-duplicate content.

A useful pattern that sidesteps part of the trade-off is to decouple the retrieval unit
from the generation unit. Small child chunks are embedded and retrieved for precision,
but when one is matched the system returns its larger parent section to the generator so
the model still sees full context. This small-to-big or parent-document strategy captures
the precision of small chunks and the context of large ones.

Because chunking changes which passages exist at all, it should be evaluated, not
guessed. Varying chunk size while holding the retriever and the gold labels fixed shows
directly how recall and answer quality move, and the right size is usually found
empirically for a given corpus rather than copied from a blog post. Gold labels defined
by character spans rather than chunk ids stay valid across chunk-size changes, which is
what makes such an ablation possible.
