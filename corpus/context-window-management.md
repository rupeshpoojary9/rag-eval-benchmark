# Context Window Management

Once passages are retrieved they must be packed into the generation model's context
window alongside the prompt and the question. How many passages to include, in what
order, and how to fit them within the token budget is its own design problem, separate
from retrieval quality, and getting it wrong can waste good retrieval.

The number of passages, the retrieval cutoff k, is the central lever. A larger k raises
the chance that the answer is somewhere in the context, improving recall, but it also
pulls in more irrelevant text that can distract the model, and it costs more tokens and
therefore more money and latency. A smaller k is cheaper and cleaner but risks omitting
the passage that holds the answer. The right k is found empirically against an evaluation
set and depends on retrieval quality: a strong reranker that concentrates relevance at
the top lets you use a smaller k, while a noisy retriever needs a larger one to be safe.

Order matters more than intuition suggests. Language models attend unevenly across a long
context, tending to use information at the very beginning and the very end more reliably
than material buried in the middle, an effect often described as "lost in the middle."
The practical response is to place the highest-ranked passages at the edges of the
context rather than the centre, and to keep the context no longer than it needs to be,
since padding it with marginal passages can actively degrade the answer.

More context is therefore not automatically better. Beyond a point, adding passages
lowers answer quality by diluting the signal and inviting the model to wander, even
though the raw retrieval recall keeps rising. This is why retrieval recall at k and final
answer quality must both be measured: the k that maximises recall is not always the k
that maximises a faithful, correct answer, and only evaluating end to end reveals the
sweet spot.

Techniques that stretch the budget include compressing or summarising retrieved passages
before insertion, deduplicating near-identical chunks, and using the decouple-retrieval
-from-generation pattern where small chunks are matched but their larger parent sections
are what get sent to the model. Each trades some fidelity or compute for a better
signal-to-token ratio in the final context.
