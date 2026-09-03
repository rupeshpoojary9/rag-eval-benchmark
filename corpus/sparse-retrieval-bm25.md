# Sparse Retrieval and BM25

Sparse retrieval ranks passages by the words they literally contain. The classic
algorithm is BM25, a probabilistic ranking function that scores a passage against a
query using three ingredients: how often each query term appears in the passage (term
frequency), how rare that term is across the whole collection (inverse document
frequency), and how long the passage is relative to the average.

Term frequency rewards passages that mention a query word many times, but with
diminishing returns, so a word appearing ten times is not ten times as convincing as
appearing once. Inverse document frequency down-weights common words like "the" or
"model" that appear everywhere and carry little discriminating power, while boosting rare
words that strongly identify a topic. Length normalisation prevents long passages from
winning simply because they contain more words and therefore more chances to match.

The great advantage of BM25 is precision on exact terms. If a user searches for a
specific function name, an error code, a part number, or an unusual proper noun, BM25
finds the passage that contains that exact token immediately, with no training and no
embedding model. It is fast, cheap, interpretable, and a remarkably strong baseline that
many teams underestimate.

Its weakness is the vocabulary mismatch problem. BM25 has no notion of meaning, so a
query and a relevant passage that use different words for the same idea will not match. A
question phrased as "keep answers tied to the source" will not retrieve a passage that
only ever says "grounded generation" and "faithfulness", because there is no lexical
overlap. Synonyms, paraphrases, morphological variants, and cross-lingual queries all
expose this gap.

Practical BM25 quality depends heavily on tokenisation and preprocessing. Lower-casing,
removing punctuation, optional stop-word removal, and stemming or lemmatisation all
change which terms match. The two tunable constants, k1 and b, control term-frequency
saturation and the strength of length normalisation respectively; sensible defaults are
around k1 between 1.2 and 2.0 and b around 0.75, and they rarely need heavy tuning for a
first cut. Because BM25 and dense retrieval fail in different ways, they are natural
partners in a hybrid system.
