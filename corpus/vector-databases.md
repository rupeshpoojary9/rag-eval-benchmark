# Vector Databases and ANN Indexes

A vector database stores the embeddings produced for a corpus and answers nearest
neighbour queries: given a query vector, return the stored vectors closest to it. For a
handful of thousand passages a plain array scanned with a brute-force similarity
computation is exact and fast enough, and it is the right choice when correctness matters
more than latency. The need for a specialised index arises at scale, when scanning every
vector for every query becomes too slow.

Approximate nearest neighbour (ANN) indexes make search sublinear by accepting a small,
tunable loss of recall. The most widely used structure is HNSW, a layered proximity graph
that navigates from coarse to fine, hopping between neighbours until it converges on the
closest vectors. It offers excellent speed and recall and supports incremental inserts,
at the cost of higher memory use. An alternative family, IVF, partitions the space into
clusters and searches only the clusters nearest the query, trading some recall for lower
memory; it is often combined with product quantisation to compress vectors and shrink the
index further.

Every ANN index exposes a knob that trades recall for speed. In HNSW the size of the
candidate list explored at query time controls how thorough the search is; in IVF it is
the number of clusters probed. Turning the knob up recovers recall lost to
approximation but costs latency, and the correct setting depends on how much recall the
application can afford to lose. This is why an ANN system should be evaluated against an
exact brute-force baseline, to measure the recall actually given up.

Beyond the raw index, production vector stores add metadata filtering, so results can be
restricted by attributes like tenant, date, or source; hybrid support, so sparse and
dense scores can be combined server-side; and persistence, replication, and updates.
Options range from libraries embedded in the application, such as FAISS, to the pgvector
extension that adds vector types and indexes to PostgreSQL, to dedicated services.
Choosing among them is mostly about operational fit: existing infrastructure, corpus
size, update frequency, and filtering needs, rather than raw benchmark speed alone.
