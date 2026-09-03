# Dense Retrieval

Dense retrieval finds relevant passages by comparing meaning rather than words. Each
passage in the corpus is passed through an embedding model that maps text to a
fixed-length vector, typically a few hundred to a couple of thousand dimensions. A
query is embedded with the same model, and the passages whose vectors sit closest to
the query vector are returned. Closeness is usually measured with cosine similarity or
an inner product over L2-normalised vectors, where the two are equivalent.

The defining strength of dense retrieval is that it can match a question to an answer
even when they share no words. A user who asks "how do I stop the model from making
things up" can be matched to a passage about grounded generation and hallucination,
because the embedding model has learned that those phrasings live in the same region of
vector space. This robustness to vocabulary mismatch is exactly where keyword search
tends to fail.

Dense retrieval also has weaknesses. It can miss exact-match signals that matter, such
as a specific error code, a product SKU, a function name, or a rare proper noun, because
those tokens carry little semantic weight and get smoothed away in the embedding. It is
sensitive to the domain the embedding model was trained on; an off-the-shelf model may
underperform on legal, medical, or code corpora without adaptation. And embeddings are
opaque, so a bad retrieval is harder to debug than a keyword miss.

Because comparing a query against every vector is expensive at scale, production systems
use approximate nearest neighbour (ANN) search. Index structures such as HNSW graphs or
IVF partitions trade a small amount of recall for a large speed-up, returning the
approximate top matches in sublinear time. For small corpora of a few thousand passages,
an exact brute-force scan over all vectors is simpler and fast enough, and it avoids the
recall loss that an approximate index introduces.

A practical detail that changes results more than people expect is normalisation.
Embeddings should generally be L2-normalised before comparison so that cosine similarity
behaves consistently, and the same pooling and normalisation used at index time must be
used at query time. Mixing a normalised index with unnormalised queries silently
degrades every downstream metric.
