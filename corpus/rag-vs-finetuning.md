# RAG versus Fine-Tuning

Retrieval-augmented generation and fine-tuning are often framed as competitors, but they
solve different problems and are frequently used together. The clearest way to decide is
to separate what the model needs to know from how the model needs to behave. RAG supplies
knowledge at inference time by retrieving relevant passages into the prompt; fine-tuning
adjusts the model's weights to change its behaviour, format, or style.

RAG is the right tool when the challenge is knowledge: facts that are large, changing, or
private. Because the knowledge lives in an external store, it can be updated by editing
documents rather than retraining, which makes RAG ideal for information that changes
often, for corpora too large to fit in any context or memorise reliably, and for
per-tenant or access-controlled data that must not be baked into shared weights. RAG also
supports attribution, since answers can cite the passage they came from, and it reduces
hallucination by grounding the model in real sources.

Fine-tuning is the right tool when the challenge is behaviour: a consistent output format,
a domain tone, a specialised classification or extraction skill, adherence to a rigid
schema, or reliably following an unusual instruction pattern. It teaches the model a way
of responding that is hard to elicit with prompting alone, and it can also make a smaller,
cheaper model match a larger one on a narrow task. What fine-tuning does poorly is inject
fresh facts: teaching new knowledge through weights is data-hungry, expensive to keep
current, and prone to the model stating outdated information confidently.

The two are complementary rather than exclusive. A common production shape is a fine-tuned
model that reliably follows the house format and grounding instructions, wrapped in a RAG
pipeline that feeds it current, cited knowledge at query time. Prompt engineering is the
cheap first move to try before either, and RAG is usually the next step because it is
faster to build and to update than a training pipeline. Fine-tuning is reached for when
prompting and retrieval have been exhausted on a behavioural gap, not as a way to make the
model "know more."

A useful rule of thumb: if the fix is "the model should have known this fact," reach for
retrieval; if the fix is "the model should have responded like this," reach for
fine-tuning; and if you are unsure, improve the prompt and the retrieval first, because
they are cheaper to iterate and easier to evaluate.
