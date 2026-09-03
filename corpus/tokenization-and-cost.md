# Tokenization, Latency, and Cost

Language models do not read characters or words directly; they read tokens, the sub-word
units a tokenizer splits text into. A token is often a common word or a fragment of a
longer or rarer word, and as a rough rule one token is about four characters or
three-quarters of a word in English. Everything about model cost, speed, and capacity is
denominated in tokens, so understanding them is essential to running a retrieval system
economically.

Token count drives cost directly, because providers bill per input and output token. In a
RAG pipeline the retrieved context is usually the largest contributor to input tokens, so
the retrieval cutoff k, the chunk size, and any overlap translate straight into money on
every request. Doubling k or doubling chunk size roughly doubles the context tokens and
therefore the input cost, which is why a retriever that concentrates relevance and lets you
use a smaller k is not only more accurate but cheaper.

Latency has several components that behave differently. Time to first token depends largely
on how much input the model must read, so a bloated context slows the start of the response;
generation time then scales with how many output tokens are produced. A cross-encoder
reranker adds its own latency proportional to the number of candidates it scores, and each
step of an agent loop adds a full round trip. Budgeting latency means accounting for
retrieval, reranking, and generation together, not just the final model call.

Because tokens couple accuracy, cost, and speed, retrieval design is always a
three-way trade rather than a pure quality optimisation. A larger k or chunk size can lift
recall while raising both cost and latency; a reranker can improve precision enough to
justify a smaller k, recovering cost and speed at the top of the pipeline. The only way to
navigate the trade honestly is to measure all three together on a representative workload,
so that a change sold as a quality win is checked against what it does to the bill and the
response time.
