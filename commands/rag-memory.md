# Build a RAG / Long-term Memory Agent Harness

Given the user's spec, build a complete runnable Python project for a
retrieval-augmented agent with persistent long-term memory. Ask for a spec if
they didn't give one (what the agent does + what it should remember).

## What to build

Scaffold a `src/`-layout project with `pyproject.toml` into the requested
directory (default: a folder named after the project). **Stdlib + `openai`
only** — no numpy, chroma, faiss, or langchain.

```
<project>/                     # e.g. remember-bot → package remember_bot
├── pyproject.toml
├── .env.example
├── README.md
└── src/<package>/
    ├── __init__.py
    ├── config.py     # env/.env config: AGENT_MODEL, AGENT_BASE_URL, AGENT_API_KEY, MEMORY_TOP_K
    ├── storage.py    # SQLite memory store (id, text, metadata, embedding BLOB)
    ├── embeddings.py # pluggable providers: ollama (nomic-embed-text) + hash fallback
    ├── retriever.py  # cosine top-k retrieval; injects "Relevant memories:" block
    ├── writer.py     # LLM fact extraction + naive sentence-split fallback; upsert
    ├── agent.py      # retrieve → complete → store → re-retrieve loop
    └── cli.py        # <cmd> ask|chat|search, all with --memory-dir
```

## Pattern

Before each LLM call, embed the prompt, cosine-rank every stored memory, take
the top-k, and prepend them as a `Relevant memories:` block in the system prompt.
After the turn, extract facts from the user turn / tool results / answer and
upsert them as new memories. Keep them in SQLite (float32 BLOBs via
`array("f")`), so memory persists across sessions with no vector DB.

## Core pieces

1. **Storage** (`storage.py`): `memories(id, text, metadata JSON, embedding BLOB)`.
   `upsert(text, metadata, embedding)` inserts or updates on identical text so
   repeated turns don't duplicate facts. Embeddings stored via
   `array("f", vec).tobytes()` and read back with `array("f").frombytes(...)`.

2. **Embeddings** (`embeddings.py`): an `EmbeddingProvider` ABC with two backends,
   chosen by `EMBEDDING_BACKEND` (default `ollama`):
   - `OllamaEmbeddingProvider` — POST `{model, prompt}` to
     `http://127.0.0.1:11434/api/embeddings` with stdlib `urllib`, model
     `nomic-embed-text`; returns `data["embedding"]`.
   - `HashEmbeddingProvider` — deterministic feature-hash: tokenize into words +
     char 3-grams, `blake2b` each into signed buckets of a 512-dim vector, L2
     normalize. Offline, so tests and `search` work without a server.
   Both must produce `list[float]`.

3. **Retrieval** (`retriever.py`):

   ```python
   def cosine_similarity(a, b):
       if not a or not b or len(a) != len(b):
           return 0.0          # mismatched dims = mixed backends
       dot = sum(x * y for x, y in zip(a, b))
       na = math.sqrt(sum(x*x for x in a)) or 1.0
       nb = math.sqrt(sum(y*y for y in b)) or 1.0
       return dot / (na * nb)

   def as_context_block(self, query, top_k=None):
       hits = self.retrieve(query, top_k=top_k)
       if not hits:
           return ""
       lines = ["Relevant memories:"]
       for score, mem in hits:
           lines.append(f"- ({mem.metadata.get('source','memory')}, {score:.3f}) {mem.text}")
       return "\n".join(lines)
   ```

   `MemoryAgent` builds its system prompt as `base + "\n\n" + as_context_block(...)`
   when the block is non-empty.

4. **Writer** (`writer.py`): `extract(text)` tries
   `client.chat.completions.create(...)` asking for a JSON string array (strip
   ``` fences), falls back to a naive sentence split (drop sentences < 12 chars /
   < 3 words). Each fact, embedded, goes through `store.upsert(...)`;
   `write(text, source)` returns the stored facts.

5. **Agent loop** (`agent.py`):

   ```python
   def run(self, user_input, *, write_memory=True, refine=True):
       memories = self.retriever.as_context_block(user_input, self.top_k)
       system = build_system_prompt(self.system_prompt, memories)
       answer = self._complete([
           {"role": "system", "content": system},
           {"role": "user", "content": user_input},
       ])
       if write_memory:
           stored = self.writer.write(user_input, source="user")
           stored += self.writer.write(answer, source="answer")
           if refine and stored:                      # re-retrieve: new facts qualify
               refreshed = self.retriever.as_context_block(user_input, self.top_k)
               if refreshed != memories:
                   answer = self._complete([
                       {"role": "system", "content": build_system_prompt(self.system_prompt, refreshed)},
                       {"role": "user", "content": "Taking the newly stored memories into account, reconsider: " + user_input},
                   ])
       return answer
   ```

   Also a `chat_loop(agent)` REPL — cross-turn memory is automatic since the
   store persists.

6. **CLI** (`cli.py`): subcommands `ask "<prompt>"`, `chat`, and `search "query"`
   (query memory directly, prints `score [source] text`, **no API key needed**).
   Add `--memory-dir` and `--top-k` to every subcommand via a small
   `_add_common(p)` helper (flags must live on each subparser for argparse).
   `search` returns before the API-key check.

## pyproject.toml

```toml
[project]
name = "<package-kebab>"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["openai>=1.0.0"]

[project.scripts]
<command> = "<package>:cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/<package>"]
```

.env.example:

```
AGENT_API_KEY=sk-...
# AGENT_BASE_URL=https://api.openai.com/v1
# AGENT_MODEL=gpt-4o-mini
# MEMORY_TOP_K=5
# EMBEDDING_BACKEND=ollama     # or "hash" for offline/deterministic
# OLLAMA_EMBED_URL=http://127.0.0.1:11434/api/embeddings
# OLLAMA_EMBED_MODEL=nomic-embed-text
```

## Conventions

- Keep one embedding backend per store (hash=512 dims, nomic=768); mixing them
  scores 0 and silences retrieval. Change backend → fresh store.
- Write memories after completing the turn; temp=0 throughout; upsert by text to
  dedupe. Never hardcode/print keys.

## Verify before finishing

1. `pip install -e .` (or `uv sync`); `python -m <package>.cli --help`.
2. Offline smoke test (no LLM, no network):

   ```bash
   EMBEDDING_BACKEND=hash python -c "
   from <package>.storage import MemoryStore
   from <package>.embeddings import HashEmbeddingProvider
   from <package>.writer import MemoryWriter
   MemoryWriter(MemoryStore('/tmp/mem/memories.db'), HashEmbeddingProvider()).write(
       'The user prefers short answers.', 'user')"
   EMBEDDING_BACKEND=hash python -m <package>.cli search "how should I answer" --memory-dir /tmp/mem
   ```

3. With a key: `ask "remember my coffee is black"` → `ask "what do I drink?"`;
   repeat in a fresh shell to show cross-session persistence.

## Example

User: "Build a remember-bot that remembers my preferences across sessions and
lets me query memory from the CLI." You: scaffold `remember-bot/` with the six
modules, wire `ask|chat|search --memory-dir`, verify the offline hash `search`
smoke test + `--help`, report structure and run commands.