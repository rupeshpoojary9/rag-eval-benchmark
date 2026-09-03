# Reranking

Reranking is a second stage that reorders a shortlist of candidate passages to push the
most relevant ones to the very top. First-stage retrieval, whether BM25, dense, or
hybrid, is optimised to be fast over the whole corpus and so it casts a wide but coarse
net. A reranker then spends more compute on a small set, typically the top twenty to one
hundred candidates, to get the ordering right where it matters most.

The key architectural distinction is between bi-encoders and cross-encoders. A
bi-encoder, which is what dense retrieval uses, embeds the query and each passage
separately and compares the two vectors; this is cheap because passage vectors can be
precomputed and indexed, but the query and passage never directly interact. A
cross-encoder instead feeds the query and a candidate passage together into a single
transformer, which attends across both at once and outputs a single relevance score.
That joint attention lets it catch fine distinctions a bi-encoder misses, which is why
cross-encoder rerankers consistently improve precision at the top ranks.

The cost is latency and compute. Because a cross-encoder must run one forward pass per
query-passage pair and cannot precompute anything, it is far too slow to score an entire
corpus. That is exactly why it is confined to reranking a shortlist: retrieval narrows
thousands of passages down to a handful of candidates, and the reranker is applied only
to those. Choosing how many candidates to rerank is a direct latency-versus-quality
trade; reranking one hundred candidates is more thorough but slower than reranking
twenty.

Rerankers matter most when the first stage has decent recall but poor ordering, meaning
the right passage is somewhere in the top twenty but not at rank one. If the right
passage is not retrieved at all, a reranker cannot help, because it only reorders what it
is given. This is the practical reason to widen first-stage recall with hybrid search
before reranking: the reranker can only promote a good passage that the retriever
actually surfaced. Popular open cross-encoders include the MS MARCO MiniLM models and the
BGE reranker family.
