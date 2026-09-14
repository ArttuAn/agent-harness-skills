---
description: Build a RAG / memory agent harness from a spec (chunking, embeddings, hybrid retrieval and cited answers)
argument-hint: your spec — domain, tools, endpoint, constraints
---

Build a RAG / memory agent harness for this spec:

$ARGUMENTS

If the spec does not say what the agent is for, which tools it needs, or which
endpoint and model it targets, ask — briefly — then build. Do not ask for
permission to begin.

**Read `~/.claude/skills/rag-memory/SKILL.md` first** (the `harness-rag-memory` skill from
agent-harness-skills). It carries the design decisions, the failure modes and
the required tests. If it is not installed, the essentials are below.

## Non-negotiables for this pattern

- **Check the corpus size first.** Below roughly 50k tokens it fits in context and RAG makes it worse. Saying so is more valuable than building it.
- Chunk ~1200 chars with ~200 overlap at natural boundaries, and store each chunk's **character offset** so the agent can read past the snippet.
- `nomic-embed-text` needs `search_document: ` / `search_query: ` prefixes. Store the embedding model name with the vectors and refuse to search a store built by a different one — otherwise the change silently returns noise.
- Semantic search alone misses exact strings (version numbers, error codes, `4271`). Add an FTS5 keyword pass. Their scores are **not comparable** — cosine is 0-1, bm25 is an unbounded rank — so keep and show which retriever found each hit. Quote FTS terms so user punctuation can't be read as syntax.
- **Retrieve before the first model call** and put results in the prompt; do not leave grounding to the model's discretion. Put instructions above the question — a small model parrots what it read last.
- Brute-force cosine over float32 blobs in SQLite is right at laptop scale. No vector DB until you have measured a problem.
- Give sources stable `[S1]` ids in every snippet the model sees, and require inline citations plus a Sources list.
- Strip `nav`/`header`/`footer`/`aside`/`script`/`style` **and `math`** when extracting HTML — MathML serializes to one character per line and poisons chunks.
- `executemany` does not report `lastrowid`, so inserting chunks in bulk silently leaves the FTS index empty while vector-only tests still pass. Insert one at a time.

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
