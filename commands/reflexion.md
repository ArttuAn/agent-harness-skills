---
description: Build a Reflexion agent harness from a spec (act, evaluate, reflect, retry with episodic memory)
argument-hint: your spec — domain, tools, endpoint, constraints
---

Build a Reflexion agent harness for this spec:

$ARGUMENTS

If the spec does not say what the agent is for, which tools it needs, or which
endpoint and model it targets, ask — briefly — then build. Do not ask for
permission to begin.

**Read `~/.claude/skills/reflexion/SKILL.md` first** (the `harness-reflexion` skill from
agent-harness-skills). It carries the design decisions, the failure modes and
the required tests. If it is not installed, the essentials are below.

## Non-negotiables for this pattern

- **Establish the evaluator first.** If the only judge is the same model that wrote the answer, it will approve its own work — stop and recommend plain ReAct instead. Prefer tests, ground truth, or a schema check.
- A reflection must be a **transferable lesson** ("search_users takes an email, not an ID — look it up with find_account first"), not a retelling of what happened. Constrain to 1-3 imperative sentences.
- Default to **episodic** memory. Persistent memory is the exciting option that fails quietly: one wrong lesson contaminates every later run.
- `max_attempts=3`, and stop early when the score stops improving.
- Return the **best** attempt, not the last.
- Pass the 2-3 most recent deduplicated lessons, not every past transcript.
- Classify transport and rate-limit errors as non-reflectable — retry with backoff, don't spend an attempt.

## Non-negotiables for every pattern

Whatever the pattern, these are not optional — they are what separates this
from a scaffold written from memory:

- **Take the LLM client as a constructor argument.** Define your own
  `ChatClient` ABC, `ToolCall(id, name, arguments: dict)` and `ModelResponse`,
  and ship a `FakeChat` that replays scripted responses. The loop must never
  construct a provider SDK. Without this seam none of the guards below can be
  tested, which in practice means they will not be written.
- **Normalize provider differences inside the client.** OpenAI sends tool
  arguments as a JSON string, Ollama as a dict. OpenAI keys tool results by
  `tool_call_id`, Ollama by `tool_name`. Ollama defaults to a 4096-token
  context regardless of the model — set `num_ctx` explicitly.
- **Tool errors return as text, never raise.** `f"Error: {type(exc).__name__}: {exc}"`
  goes back to the model, which reads it and retries.
- **One result message per tool call.** A model can emit several in one turn;
  a missing result breaks the *next* request, not the one that caused it.
- **Coerce arguments against the declared schema.** Models send `"5"` for an
  integer, invent parameters, and echo the schema fragment back as the value
  (`limit={"type": "integer"}`). Repair or drop; never let it reach the tool.
- **On budget exhaustion, ask once more with no tools offered** so the model
  has nothing to emit but prose. Do not raise at the user.
- **`temperature=0`** for any turn that selects a tool.
- **Write the offline tests before declaring done** — tool error recovery,
  argument coercion, repeat-call handling, and the forced final answer. They
  need no API key. Then run them.

Verify in tiers: (0) install + import + `--help`, (1) `pytest -q` offline,
(2) one real run against the endpoint. Report exactly which tiers ran. If
there was no API key and tier 2 was skipped, say so — do not imply an
end-to-end run happened.
