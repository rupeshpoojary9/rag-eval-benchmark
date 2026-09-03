# Query Transformation

Query transformation improves retrieval by rewriting the user's question before it is sent
to the retriever, rather than changing the retriever itself. The motivation is that raw
user queries are often short, ambiguous, or phrased differently from the documents, and a
better-formed query retrieves better passages. These techniques sit in front of any
retriever, sparse or dense, and are cheap ways to lift recall.

Query expansion adds related terms to the query so it matches more relevant passages. For
sparse retrieval this directly attacks the vocabulary-mismatch problem by injecting
synonyms and morphological variants that BM25 would otherwise miss. Expansion can be drawn
from a thesaurus, from pseudo-relevance feedback that harvests terms from the first pass
of results, or from a language model asked to list alternative phrasings.

Multi-query retrieval generates several reformulations of the same question, retrieves for
each, and merges the results, often with reciprocal rank fusion. Because different
phrasings surface different passages, the union has higher recall than any single query.
The cost is more retrieval calls and some added latency, and a risk of drifting off topic
if the reformulations are poor.

A distinctive technique is hypothetical document embedding, or HyDE, which asks a language
model to draft a hypothetical answer to the question and then embeds that draft instead of,
or alongside, the question. The intuition is that a passage answering the question looks
more like an answer than like a question, so an embedded hypothetical answer lands closer
to the real answer passages in vector space. HyDE can help notably in zero-shot settings,
though it depends on the model producing a plausible draft and adds a generation step.

For complex questions, query decomposition breaks a multi-part question into simpler
sub-questions, retrieves for each, and composes the answer, which suits multi-hop
questions that no single passage answers. All of these transformations trade extra
computation and latency for recall, and like every retrieval change they should be
measured on a gold set rather than assumed to help, since an aggressive rewrite can just
as easily pull the query away from the answer.
