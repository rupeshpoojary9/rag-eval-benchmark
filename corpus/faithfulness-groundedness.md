# Faithfulness and Groundedness

Faithfulness measures whether a generated answer is supported by the passages the system
retrieved, rather than by the model's parametric memory or invention. An answer is
faithful, or grounded, when every claim it makes can be traced back to the provided
context. It is an answer-side metric, distinct from retrieval metrics: retrieval can be
excellent while the answer still drifts, and retrieval can be poor while the model
faithfully reports that it cannot answer.

Faithfulness is not the same as correctness. An answer can be faithful to the context yet
wrong because the retrieved passage was itself wrong or outdated, and an answer can be
factually correct yet unfaithful because the model supplied the fact from memory rather
than from the context. In a retrieval-augmented system faithfulness is often the property
you can actually control and audit, since it depends on the pipeline you built rather
than on the model's training data, which is why it is a primary target for evaluation.

The common failure mode is hallucination: the model asserts something the context does
not support, filling gaps with plausible but unsupported detail. Hallucination rises when
retrieval fails to surface the answer and the prompt still pressures the model to produce
one, so weak retrieval and a demanding prompt together are the classic recipe. This is
the practical link between the two halves of the system: improving retrieval so the
answer is actually present in the context is often the most effective way to reduce
hallucination.

Faithfulness is typically scored by decomposing the answer into individual claims and
checking each against the retrieved context, then reporting the fraction of claims that
are supported. The check can be done heuristically, by measuring overlap between the
answer's content and the context, or with an LLM-as-judge that reads the context and the
answer and labels each claim as supported or not. LLM judges align better with human
judgement but cost more and add variance, so a cheap deterministic proxy is useful for
fast iteration and a stronger judge for final numbers.

Mitigations that improve faithfulness include instructing the model to answer only from
the provided context and to abstain when the answer is absent, requiring inline citations
to specific passages so claims are auditable, and raising retrieval recall so the
supporting passage is present in the first place. Faithfulness should be tracked alongside
retrieval recall, because the two together explain end-to-end answer quality in a way
neither does alone.
