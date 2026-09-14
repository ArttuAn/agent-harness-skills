# Choosing a pattern

Start here before invoking a skill. Picking the wrong pattern costs more than
writing the right one slightly worse, and the most common mistake is reaching
for an elaborate pattern when a plain loop would do.

## Default: you probably want ReAct

A single tool-calling loop — model reasons, calls a tool, reads the result,
repeats, answers — solves most agent problems. Every other pattern here is
ReAct plus a specific piece of machinery, bought to solve a specific problem.
Do not buy the machinery until you have the problem.

Reach for something else only when one of these is true:

| Symptom you actually have | Pattern | What it adds |
| --- | --- | --- |
| The task has many steps, and the model loses the thread halfway | **plan-and-execute** | An explicit plan object, executed step by step, replanned on failure |
| The agent fails in ways it could notice itself, and you can score attempts | **reflexion** | Evaluate → reflect → retry, with the lesson kept in episodic memory |
| Work splits cleanly into roles that do not need each other's context | **multi-agent** | Orchestrator + workers with separate context windows |
| The agent edits files and runs commands | **code-agent** | Workspace confinement, command blocklist, diff review |
| Answers must come from a specific corpus, with citations | **rag-memory** | Embedding store, retrieval, prompt injection |
| You want the prompts and tools to improve across runs | **self-evolving** | Run → evaluate → reflect → apply, gated by tests |

## When each pattern is the wrong choice

Worth reading before you commit a weekend.

**plan-and-execute** is wrong when the task is short or the environment is
unpredictable. A plan made before any observation is a guess; if step 2 usually
invalidates the plan, you have paid for planning and gained replanning
overhead. Use ReAct.

**reflexion** is wrong without a real evaluator. Reflexion's whole value is the
score — if the only judge is the same model that produced the answer, you get
confident self-approval and burn tokens per retry. No ground truth, no tests,
no verifier? Use ReAct.

**multi-agent** is wrong for anything under roughly five distinct subtasks, and
wrong whenever workers need to see each other's intermediate state. The cost is
real: N× tokens, N× latency, plus synthesis that loses detail. A single agent
with good tools beats three agents with a coordination problem. The genuine win
is context isolation — when one worker's 50k tokens of logs must not pollute
another's window.

**code-agent** is wrong when a plain script would do. If you know the edit, do
not hire an agent to discover it. The pattern earns its keep on diffuse tasks
("make the tests pass", "migrate this API") where the file set is unknown up
front.

**rag-memory** is wrong when the corpus fits in context. Below roughly 50k
tokens of source material, putting it in the prompt beats retrieving it — no
chunking loss, no embedding drift, no retrieval misses. It is also wrong when
answers need aggregation across every document ("how many of these mention X")
— top-k retrieval structurally cannot see documents it did not rank.

**self-evolving** is wrong without a test suite it cannot edit. A harness that
improves itself against a metric it also controls will optimize the metric. The
gate must be external and immutable. Without that, this pattern degrades
quietly and looks like progress.

## Combinations that work

- **rag-memory + ReAct** — the common case: retrieval as a tool in a plain loop.
- **plan-and-execute + code-agent** — plan the migration, execute each file edit
  in a confined workspace.
- **reflexion + anything with tests** — the test suite is the evaluator, which
  is the one evaluator that does not lie.

## Combinations that do not

- **multi-agent + self-evolving** — two sources of nondeterminism, no way to
  attribute a regression to either.
- **reflexion + multi-agent** — retries multiply across workers; cost grows as
  the product, and the failure you are reflecting on is usually coordination,
  which the reflection cannot see.
