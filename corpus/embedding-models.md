# Embedding Models

An embedding model turns a piece of text into a vector of numbers that positions it in a
semantic space, where texts with similar meaning end up near each other. These models are
the engine of dense retrieval, and the choice of model sets a ceiling on how good
semantic search can be, so it deserves deliberate evaluation rather than defaulting to
whatever a tutorial used.

Several properties distinguish embedding models. Dimensionality, the length of the
vector, ranges from a few hundred to several thousand; higher dimensions can capture more
nuance but cost more memory and compute per comparison. The maximum input length caps how
much text a single embedding can represent, which interacts directly with chunk size.
Whether the model was trained symmetrically, for comparing two similar-length texts, or
asymmetrically, for matching short queries to longer passages, affects how you should use
it. Many modern models expect a task-specific instruction or prefix on the query, and
omitting it quietly hurts quality.

Model quality is commonly compared on public leaderboards such as MTEB, which aggregates
performance across retrieval, classification, clustering, and other tasks. Leaderboards
are a starting point, not an answer, because a model that tops a general benchmark can
still underperform on a specific domain like code, law, or medicine. The reliable way to
choose is to run the candidate models through your own labelled retrieval evaluation on
your own corpus and compare recall and MRR directly.

Two operational details matter disproportionately. First, the exact same model,
preprocessing, and pooling must be used to embed the corpus at index time and the query
at search time; any mismatch corrupts the comparison. Second, vectors are usually
L2-normalised so that cosine similarity and dot product agree and distances behave
predictably. Compact general-purpose models such as the MiniLM sentence-transformers
family are a strong, fast default that runs comfortably on a laptop CPU, while larger
models trade speed for a modest quality gain.
