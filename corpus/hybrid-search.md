# Hybrid Search

Hybrid search combines a sparse ranker like BM25 with a dense embedding ranker so the
system benefits from both exact-term precision and semantic recall. The premise is that
the two methods make uncorrelated mistakes: BM25 misses paraphrases, dense retrieval
misses rare exact tokens, and a passage that either method ranks highly is worth
surfacing. Fusing them typically lifts recall above either component alone.

There are two common ways to fuse the two result lists. The first is weighted score
fusion, where the BM25 score and the dense similarity are each normalised to a
comparable range and then combined as a weighted sum. A weight parameter, often called
alpha, controls how much the dense signal counts relative to the sparse signal; alpha of
zero is pure BM25 and alpha of one is pure dense. The difficulty is that BM25 scores and
cosine similarities live on different, unbounded scales, so the normalisation choice
(min-max, z-score, or otherwise) strongly affects the outcome and can be brittle across
queries.

The second method is Reciprocal Rank Fusion (RRF), which ignores the raw scores and uses
only the rank position of each passage in each list. Every passage receives a
contribution of one divided by a small constant k plus its rank, summed across the
lists, so a passage ranked highly by either retriever floats to the top. RRF is popular
because it needs no score normalisation, is robust to scale mismatches, and works well
with sensible defaults, with the constant k commonly set around 60. It sacrifices the
fine-grained score information that weighted fusion keeps, but in exchange it is far
harder to break.

In practice teams often start with RRF for its robustness, then move to tuned weighted
fusion only if they have a labelled evaluation set to tune the weight against. Whichever
fusion is used, hybrid retrieval is usually run as the candidate-generation stage feeding
a reranker, because fusion widens recall while a cross-encoder reranker then sharpens
precision at the very top of the list.
