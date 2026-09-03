# LLM Agents and Tool Use

An LLM agent is a language model given the ability to take actions, not just produce text.
Instead of answering in one shot, the model runs in a loop: it reasons about the goal,
chooses a tool to call, observes the result, and repeats until the task is done. Tools are
typed functions the model can invoke, such as a web search, a database query, a calculator,
a code runner, or a retrieval call, each described to the model by a name, a purpose, and a
schema of arguments.

The dominant pattern interleaves reasoning and acting, often summarised as reason-then-act.
The model writes a short thought about what to do next, emits a structured tool call, the
runtime executes it and returns an observation, and that observation is fed back into the
context for the next step. This loop lets the model gather information it did not have,
recover from a failed step by trying another approach, and break a large task into a
sequence of tool calls rather than solving everything in a single generation.

Retrieval fits naturally as one tool among several. Rather than always stuffing retrieved
passages into the prompt, an agent can decide when a question needs a lookup and issue a
retrieval call only then, which is sometimes called agentic retrieval. This makes retrieval
conditional and iterative: the agent can retrieve, inspect what came back, and retrieve
again with a refined query, effectively performing query transformation and multi-step
search on its own.

Agents introduce their own failure modes and costs. Each loop step is another model call,
so latency and token spend grow with the number of steps, and a poorly bounded agent can
loop indefinitely or wander off task. Reliability suffers when tool schemas are ambiguous
or the model hallucinates arguments, so clear typed contracts, step limits, and validation
of tool inputs and outputs are essential. Governance concerns such as which actions require
human approval, an audit trail of every call, and a budget on steps and cost become
first-class design requirements once a model can act on the world rather than only describe
it.
