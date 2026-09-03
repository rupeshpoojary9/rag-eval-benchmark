# Prompting for Retrieval-Augmented Generation

Once relevant passages are retrieved, the prompt decides how faithfully the model uses
them. Even perfect retrieval produces a bad answer if the prompt lets the model ignore the
context, invent detail, or answer when it should decline. Prompting for RAG is therefore
about constraining the model to reason from the supplied passages and to be honest about
their limits.

The foundational instruction is to answer only from the provided context. Telling the
model explicitly to base its answer on the passages, and not on prior knowledge, reduces
the tendency to supply remembered facts that the retrieval did not support. Paired with
this is an abstention instruction: the model should say it does not know, or that the
context does not contain the answer, when the passages are insufficient. Without an
explicit licence to abstain, a model pressured to always answer will hallucinate to fill
the gap, which is one of the most common causes of unfaithful output.

Citations make an answer auditable. Asking the model to attach the identifier of the
passage that supports each claim lets a reader, or an automated check, verify grounding
and makes faithfulness measurable rather than a matter of trust. Citations also nudge the
model toward using the context, because a claim it must cite is a claim it must find in
the passages.

How the context is presented matters as much as what it says. Clearly delimiting each
passage, labelling it with an identifier, and separating the context block from the
question and the instructions helps the model tell sources apart and reference them.
Ordering passages so the strongest sit where the model attends best, and keeping the
context focused rather than padded, both improve how reliably the answer draws on the
right material.

Prompting is the cheapest lever in the whole pipeline and the first one to pull, but it
has a ceiling. If the answer simply is not in the retrieved passages, no instruction can
conjure a faithful answer, because there is nothing to ground it in. That ceiling is why
prompting, retrieval, and evaluation are a single loop: the prompt maximises how well the
model uses what it is given, retrieval determines what it is given, and evaluation tells
you which half to fix next.
