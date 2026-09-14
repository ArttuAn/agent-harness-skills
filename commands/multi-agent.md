---
description: Build a Multi-agent agent harness from a spec (an orchestrator with role workers)
argument-hint: your spec — domain, tools, endpoint, constraints
---

Build a Multi-agent agent harness for this spec:

$ARGUMENTS

If the spec does not say what the agent is for, which tools it needs, or which
endpoint and model it targets, ask — briefly — then build. Do not ask for
permission to begin.

**Read `~/.claude/skills/multi-agent/SKILL.md` first** (the `harness-multi-agent` skill from
agent-harness-skills). It carries the design decisions, the failure modes and
the required tests. If it is not installed, the essentials are below.

## Non-negotiables for this pattern

- **Challenge the pattern first.** Under ~5 distinct subtasks, or workers needing shared state, a single agent with good tools wins. The one strong reason to build this is context isolation. Say the cost out loud: 5 workers is roughly 6-8x a single agent.
- Workers get their **own** message list, system prompt and tool subset — never the orchestrator's history.
- Bound fan-out (`max_workers`, default 5) **and** depth (default 1 — workers do not spawn workers).
- A failed worker is data, not an exception: `WorkerResult(ok=False, error=...)`, and the synthesis prompt must be told which workers failed.
- Return structured results with an `evidence` field (ids, URLs, paths). Synthesis compresses prose; evidence survives.
- Sequential by default. Parallelize only after checking for shared SQLite connections, rate limits and shared workspaces.
- If every worker failed, report that — do not synthesize an answer from nothing.

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
