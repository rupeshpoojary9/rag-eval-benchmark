# Building a RAG Evaluation Set

The hardest and most valuable part of evaluating a retrieval system is the evaluation set
itself: a collection of questions paired with the passages that genuinely answer them.
Most teams skip this and judge their system by eyeballing a few queries, which is why so
few can actually say whether a change helped. A trustworthy gold set turns retrieval from
guesswork into measurement.

Good questions reflect how real users ask, not how the documents are written. If every
question simply reuses the wording of its answer passage, the evaluation flatters keyword
search and hides the vocabulary-mismatch problem that dense retrieval exists to solve.
Deliberately phrasing some questions in user language, with synonyms and paraphrases that
do not lexically overlap the source, is what makes the benchmark discriminate between
methods. A good set also mixes easy factual lookups with harder questions whose answer is
spread across passages or requires connecting two ideas.

Labelling relevance is a judgement call that must be made consistently. For each question
the author records which passages contain the answer, and a written guideline for what
counts as relevant keeps decisions stable across the set. A subtle but important choice is
the granularity of the label. Labelling by character span within a document, rather than
by chunk id, keeps the gold set valid when the chunking configuration changes, so the same
labels can be reused to ablate chunk size. A retrieved chunk is then scored as a hit when
it overlaps a labelled span.

Evaluation comes in two modes that answer different questions. Offline evaluation runs the
fixed gold set through the pipeline and reports metrics like recall and MRR; it is
reproducible, cheap to rerun on every change, and ideal for comparing retrieval
strategies. Online evaluation observes real traffic through signals like click-through,
thumbs, or task success; it captures real user value but is noisy, slow, and cannot be
run before shipping. A mature team uses offline evaluation to gate changes and online
signals to validate them in production.

Two hazards deserve care. Incomplete labels punish a retriever for surfacing a genuinely
relevant passage the author forgot to mark, so pooling candidates from several systems
before labelling improves coverage. And a static gold set can drift out of date as the
corpus changes, so it needs occasional maintenance. Even a small set of forty
well-labelled questions is far more useful than none, and it pays for itself the first
time it catches a change that looked good but lowered recall.
