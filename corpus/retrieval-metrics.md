# Retrieval Metrics

Retrieval metrics measure how well a search system surfaces the passages that are
actually relevant to a query, using a gold set that records which passages count as
relevant for each question. They evaluate the retriever in isolation, before any answer
is generated, which is essential because a generator can only work with what retrieval
gives it. The metrics differ in what they reward, so the right one depends on how the
retrieved passages are used.

Recall at k asks a simple question: of the passages that are truly relevant, how many
appear in the top k results? It cares only about presence within the cutoff, not
position, so a relevant passage at rank one and at rank k count the same. Recall at k is
the metric to watch when the downstream consumer reads several passages, such as a
generator that is given the top five chunks as context, because there what matters is
whether the answer is present anywhere in that set.

Mean Reciprocal Rank (MRR) rewards putting the first relevant passage as high as
possible. For each query it takes the reciprocal of the rank of the first relevant
result, so rank one scores one, rank two scores one half, rank three one third, and a
query with no relevant result in the list scores zero; MRR is the average of that across
all queries. MRR is the metric to watch when only the top result really matters, such as
a "feeling lucky" answer or a single-passage citation, because it punishes burying the
right passage even one position too low.

Two further metrics round out the picture. Precision at k measures how many of the top k
are relevant, which matters when irrelevant passages are costly, for example because they
crowd out the context window or mislead the generator. Normalised Discounted Cumulative
Gain (nDCG) accounts for graded relevance and applies a smooth positional discount, so it
captures ordering quality more finely than recall or MRR when relevance is not simply
binary.

The core discipline is to report the metric that matches the actual use of retrieval, and
usually more than one, because they can disagree. A change that raises recall at five by
widening the net can lower MRR by pushing the single best passage down, and only looking
at both reveals the trade. Retrieval metrics are also distinct from answer metrics like
faithfulness; good retrieval is necessary but not sufficient for a good answer.
