---
name: harness-rag-memory
description: "Build a RAG / long-term memory pattern agent harness from a spec"
---

# RAG / Long-term Memory Agent Harness

## What is this pattern?

A RAG / long-term memory agent keeps a **persistent memory** of what it has seen
and been told. Facts and episodes are stored as plain-text chunks, each with an
**embedding vector**. Before every LLM call the agent:

1. **Retrieves** the top-k memories most similar to the current prompt
   (cosine similarity over the embeddings);
2. **Injects** them into the system prompt as a `Relevant memories:` block;
3. **Completes** the turn with that context.

After the turn it optionally **stores** new facts it observed along the way —
from the user's turn, tool results, and its own final answer — so the next run
starts with more knowledge. This turns a stateless LLM call into something that
"remembers" across sessions without retraining.

```
        ┌─────────►   retrieve top-k memories for prompt
        │                    │  (embed the prompt, cosine search the store)
        │                    ▼
  store new facts   inject "Relevant memories:" into system prompt
  (user turn,              │
  tool result,             ▼
  answer)   ◄──  complete the LLM call with memory context
        │
        └── optionally re-retrieve (new memories can now qualify) and answer again
```

Persistent, cross-session, embed-what-you-know — cheap enough to run as a small
service and library, no vector DB required.

## Workflow

When the user gives you a spec (or says "build me an agent with memory"):

1. **Ask for a spec if none given.** What does the agent Do? What should it
   remember? If vague, propose a domain (e.g. a personal-facts assistant: "remembers
   user preferences and past tasks, answers from memory") and confirm.

2. **Scaffold the project** into the directory the user specifies (default:
   `<folder-name>/` next to where they're working). Same conventions as the ReAct
   skill: kebab-case folder, snake_case package, `hatchling` + `pyproject.toml`.

3. **Write the five core modules** (see inline code below): `storage.py`,
   `embeddings.py`, `retriever.py`, `writer.py`, `agent.py`, plus `cli.py`.

4. **Wire the CLI**: `<command> ask "..."`, `<command> chat`, and —
   importantly — `<command> search "..."` to query memory directly, all with a
   `--memory-dir` flag. Verify `--help` and an offline `search` smoke test.

## Project Layout

```
<project>/
├── pyproject.toml
├── .env.example
├── README.md
└── src/<package>/
    ├── __init__.py
    ├── config.py        # env-based config (model, base_url, api_key, top_k)
    ├── storage.py       # SQLite-backed memory store (id, text, metadata, embedding BLOB)
    ├── embeddings.py    # pluggable embedding providers (ollama + hash fallback)
    ├── retriever.py     # cosine top-k retrieval + "Relevant memories:" prompt injection
    ├── writer.py        # fact extraction (LLM or naive sentence split) + upsert
    ├── agent.py         # retrieve → complete → maybe store + re-retrieve loop
    └── cli.py           # ask / chat / search subcommands, --memory-dir flag
```

Use the kebab-case project name for the folder/CLI and the snake_case form as the
Python package (e.g. `memory-bot` → package `memory_bot`).

## Dependencies

Use only:

- `openai` (LLM calls)
- stdlib (`sqlite3`, `urllib`, `hashlib`, `array`, `json`, `argparse`, `math`,
  `re`, `abc`)

Deliberately **no** numpy, no pandas, no chroma/faiss, no langchain. Embeddings
are stored as raw `float32` bytes in a SQLite BLOB (`array("f")`) and similarity
is hand-rolled cosine on plain Python lists. The `hash` embedding backend is
deterministic and network-free, so tests and the `search` command work offline.

> Same-presence note: the two backends produce different vector dimensions
> (hash = 512, nomic-embed-text = 768). Never mix backends against one persisted
> store; cosine on mismatched dims returns 0.0. Pick a backend per store and keep
> it (env var `EMBEDDING_BACKEND`).

## The Memory Store (implement this in `storage.py`)

```python
# src/<package>/storage.py
"""SQLite-backed memory store.

Schema:
    memories(id       INTEGER PRIMARY KEY AUTOINCREMENT,
             text     TEXT NOT NULL,
             metadata TEXT NOT NULL DEFAULT '{}',   -- JSON blob
             embedding BLOB)                         -- float32 bytes
"""

from __future__ import annotations

import json
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path


def vec_to_blob(vec) -> bytes:
    """float32 BLOB round-trip (no numpy needed)."""
    return array("f", [float(x) for x in vec]).tobytes()


def blob_to_vec(blob) -> list[float]:
    if not blob:
        return []
    a = array("f")
    a.frombytes(blob)
    return [float(x) for x in a]


@dataclass
class Memory:
    id: int
    text: str
    metadata: dict
    embedding: list[float]

    @classmethod
    def from_row(cls, row) -> "Memory":
        mid, text, meta_json, emb_blob = row
        return cls(
            id=mid,
            text=text,
            metadata=json.loads(meta_json or "{}"),
            embedding=blob_to_vec(emb_blob),
        )


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS memories(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding BLOB
            )"""
        )
        self.conn.commit()

    def add(self, text: str, metadata: dict | None = None,
            embedding: list[float] | None = None) -> Memory:
        cur = self.conn.execute(
            "INSERT INTO memories(text, metadata, embedding) VALUES (?, ?, ?)",
            (text, json.dumps(metadata or {}),
             vec_to_blob(embedding) if embedding else None),
        )
        self.conn.commit()
        return Memory(id=cur.lastrowid, text=text,
                      metadata=metadata or {}, embedding=list(embedding or []))

    def upsert(self, text: str, metadata: dict | None = None,
               embedding: list[float] | None = None) -> Memory:
        """Insert, or update the row whose text is identical (avoids dup facts)."""
        row = self.conn.execute(
            "SELECT id FROM memories WHERE text = ?", (text,)
        ).fetchone()
        if row:
            self.conn.execute(
                "UPDATE memories SET metadata = ?, embedding = ? WHERE id = ?",
                (json.dumps(metadata or {}),
                 vec_to_blob(embedding) if embedding else None, row[0]),
            )
            self.conn.commit()
            return Memory(id=row[0], text=text,
                          metadata=metadata or {}, embedding=list(embedding or []))
        return self.add(text, metadata, embedding)

    def all(self) -> list[Memory]:
        rows = self.conn.execute(
            "SELECT id, text, metadata, embedding FROM memories ORDER BY id"
        ).fetchall()
        return [Memory.from_row(r) for r in rows]

    def get(self, memory_id: int) -> Memory | None:
        row = self.conn.execute(
            "SELECT id, text, metadata, embedding FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return Memory.from_row(row) if row else None

    def delete(self, memory_id: int) -> int:
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cur.rowcount

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
```

## The Embedding Provider (implement this in `embeddings.py`)

Two pluggable backends behind one interface:

- **`ollama`** — `nomic-embed-text` via `POST http://127.0.0.1:11434/api/embeddings`,
  plain stdlib `urllib`.
- **`hash`** — deterministic built-in feature-hash (bag of word + char-ngram
  tokens, signed buckets into a 512-dim vector, L2-normalized). No network, so
  offline tests and poor-man's semantics both work.

Selection via `EMBEDDING_BACKEND` env var.

```python
# src/<package>/embeddings.py
"""Pluggable embedding providers: ollama (nomic-embed-text) or deterministic hash."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from abc import ABC, abstractmethod
from urllib import error, request

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"


class EmbeddingProvider(ABC):
    name = "base"

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed one text into a fixed-dimension float vector."""

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @classmethod
    def from_env(cls) -> "EmbeddingProvider":
        backend = os.getenv("EMBEDDING_BACKEND", "ollama").lower()
        if backend in ("hash", "builtin", "offline"):
            return HashEmbeddingProvider()
        if backend in ("ollama", "nomic"):
            return OllamaEmbeddingProvider(
                url=os.getenv("OLLAMA_EMBED_URL", DEFAULT_OLLAMA_URL),
                model=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_MODEL),
            )
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}")


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    chars = re.sub(r"\s+", "", text.lower())
    return [chars[i:i + n] for i in range(max(0, len(chars) - n + 1))]


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic feature-hash bag-of-ngrams embedding (offline / tests)."""
    name = "hash"
    DIM = 512

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for tok in _tokenize(text) + _char_ngrams(text, n=3):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.DIM
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


class OllamaEmbeddingProvider(EmbeddingProvider):
    name = "ollama"

    def __init__(self, url: str = DEFAULT_OLLAMA_URL,
                 model: str = DEFAULT_OLLAMA_MODEL):
        self.url = url
        self.model = model

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                f"Ollama embedding failed (is ollama up at {self.url}?): {exc}"
            ) from exc
        embedding = data.get("embedding") or data.get("embeddings")
        if not embedding:
            raise RuntimeError(f"Ollama returned no embedding: {data}")
        return [float(x) for x in embedding]
```

## Retrieval + Prompt Injection (implement this in `retriever.py`)

```python
# src/<package>/retriever.py
"""Cosine top-k retrieval over the memory store + system-prompt injection."""

from __future__ import annotations

import math

from .embeddings import EmbeddingProvider
from .storage import Memory, MemoryStore


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0  # mismatched dims (mixed backends) score nothing
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class Retriever:
    def __init__(self, store: MemoryStore, provider: EmbeddingProvider,
                 top_k: int = 5):
        self.store = store
        self.provider = provider
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None,
                 threshold: float | None = None) -> list[tuple[float, Memory]]:
        """Embed the query, cosine-rank every memory, return top-k (score, memory)."""
        q_vec = self.provider.embed(query)
        scored = [(cosine_similarity(q_vec, m.embedding), m)
                  for m in self.store.all() if m.embedding]
        scored.sort(key=lambda sm: sm[0], reverse=True)
        hits = scored[: top_k if top_k is not None else self.top_k]
        if threshold is not None:
            hits = [(s, m) for s, m in hits if s >= threshold]
        return hits

    def as_context_block(self, query: str, top_k: int | None = None) -> str:
        """Render retrieved memories as the 'Relevant memories:' system block."""
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""
        lines = ["Relevant memories:"]
        for score, mem in hits:
            src = mem.metadata.get("source", "memory")
            lines.append(f"- ({src}, score {score:.3f}) {mem.text}")
        return "\n".join(lines)


def build_system_prompt(base_system: str, memories_block: str) -> str:
    """Append the memories block to a base system prompt (no-op when empty)."""
    if not memories_block:
        return base_system
    return f"{base_system.rstrip()}\n\n{memories_block}"
```

## The Memory Writer (implement this in `writer.py`)

Extracts standalone facts from three kinds of text — user turns, tool results,
and the final answer — then upserts each as a memory. Preferred path is LLM
extraction (`client.chat.completions` asking for a JSON string array); the naive
sentence-splitter is the offline/test fallback.

```python
# src/<package>/writer.py
"""Fact extraction (LLM or naive sentence split) + upsert into the store."""

from __future__ import annotations

import json
import re

from openai import OpenAI

from .embeddings import EmbeddingProvider
from .storage import MemoryStore


class MemoryWriter:
    def __init__(self, store: MemoryStore, provider: EmbeddingProvider,
                 client: OpenAI | None = None, model: str = "gpt-4o-mini",
                 use_llm: bool = True):
        self.store = store
        self.provider = provider
        self.client = client
        self.model = model
        self.use_llm = bool(client) and use_llm

    def write(self, text: str, source: str = "answer") -> list[str]:
        """Extract facts from `text` and upsert each. Returns the stored facts."""
        facts = self.extract(text, source)
        stored = []
        for fact in facts:
            embedding = self.provider.embed(fact)
            self.store.upsert(fact, metadata={"source": source}, embedding=embedding)
            stored.append(fact)
        return stored

    def extract(self, text: str, source: str) -> list[str]:
        if not text or not text.strip():
            return []
        if self.use_llm:
            try:
                facts = self._extract_llm(text)
                if facts:
                    return facts
            except Exception:
                pass  # ollama/key hiccups fall back to naive extraction
        return self._extract_naive(text)

    def _extract_llm(self, text: str) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "Extract standalone factual statements worth remembering "
                    "long-term. Return ONLY a JSON array of strings; empty array "
                    "if nothing memorable."
                )},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
        fence = chr(96) * 3  # a literal triple backtick fence
        if content.startswith(fence):
            content = content[len(fence):]
            if content.endswith(fence):
                content = content[:-len(fence)]
            content = content.strip()
            if content.startswith("json"):
                content = content[4:].strip()
        facts = json.loads(content)
        return [str(f).strip() for f in facts if str(f).strip()]

    def _extract_naive(self, text: str) -> list[str]:
        """Fallback: split on sentence boundaries, drop trivial/oversized ones."""
        facts = []
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            sentence = " ".join(sentence.split())
            words = sentence.split()
            if len(sentence) < 12 or len(words) < 3:
                continue
            if len(words) > 45:
                sentence = " ".join(words[:45]) + "..."
            facts.append(sentence)
        return facts
```

## The Agent Loop (implement this in `agent.py`)

```python
# src/<package>/agent.py
"""retrieve -> complete -> maybe store + re-retrieve (and answer again)."""

from __future__ import annotations

from openai import OpenAI

from .retriever import Retriever, build_system_prompt
from .storage import MemoryStore
from .writer import MemoryWriter

SYSTEM_PROMPT = (
    "You are a helpful assistant with persistent long-term memory.\n"
    "Relevant memories appear below under 'Relevant memories:'.\n"
    "Use them to ground and personalize your answers; if none appear,\n"
    "just answer from your own knowledge."
)


class MemoryAgent:
    def __init__(self, client: OpenAI, retriever: Retriever, writer: MemoryWriter,
                 model: str, top_k: int = 5, system_prompt: str | None = None):
        self.client = client
        self.retriever = retriever
        self.writer = writer
        self.model = model
        self.top_k = top_k
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def _complete(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def run(self, user_input: str, *, write_memory: bool = True,
            refine: bool = True) -> str:
        """Retrieve memories, answer; then store new facts and, if memories
        changed, re-retrieve and answer once more with the richer context."""
        # 1) retrieve + inject
        memories = self.retriever.as_context_block(user_input, self.top_k)
        system = build_system_prompt(self.system_prompt, memories)
        # 2) complete
        answer = self._complete([
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ])
        # 3) store what this turn revealed (user turn + tool results + answer)
        if write_memory:
            stored = self.writer.write(user_input, source="user")
            stored += self.writer.write(answer, source="answer")
            # 4) re-retrieve: freshly stored facts can now qualify, then refine
            if refine and stored:
                refreshed = self.retriever.as_context_block(
                    user_input, self.top_k
                )
                if refreshed != memories:
                    answer = self._complete([
                        {"role": "system", "content":
                            build_system_prompt(self.system_prompt, refreshed)},
                        {"role": "user", "content":
                            "Taking the newly stored memories into account, "
                            "reconsider your answer: " + user_input},
                    ])
        return answer


def chat_loop(agent: MemoryAgent, *, write_memory: bool = True) -> None:
    """Interactive REPL. Cross-turn memory works automatically via the store."""
    print("Memory agent ready. Type 'exit' or Ctrl-D to quit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user_input.lower() in {"exit", "quit", "q"}:
            return
        if not user_input:
            continue
        print(f"\nAgent: {agent.run(user_input, write_memory=write_memory)}")
```

Note the loop is exactly the pattern from the skill title:
**retrieve → complete → maybe store → re-retrieve → answer again**. In `chat_loop`
the store persists between turns at no extra cost, so the agent naturally
"remembers" what it was told earlier in the session and in prior sessions.

## CLI (implement this in `cli.py`)

`--memory-dir` flag on every subcommand; `search` needs no API key (works fully
offline with the hash backend).

```python
# src/<package>/cli.py
"""<command> ask|chat|search, each with --memory-dir."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .embeddings import EmbeddingProvider
from .storage import MemoryStore
from .writer import MemoryWriter


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--memory-dir", default=".memory",
                   help="Directory holding memories.db (default: %(default)s).")
    p.add_argument("--top-k", type=int, default=None,
                   help="How many memories to retrieve/inject.")


def store_for(args) -> MemoryStore:
    return MemoryStore(Path(args.memory_dir) / "memories.db")


def _cmd_search(cfg: Config, args) -> int:
    store = store_for(args)
    provider = EmbeddingProvider.from_env()
    retriever = Retriever(store, provider, top_k=args.top_k or cfg.top_k)
    query = " ".join(args.query)
    hits = retriever.retrieve(query, top_k=args.top_k or cfg.top_k)
    if not hits:
        print("No matching memories.")
        return 0
    for score, mem in hits:
        src = mem.metadata.get("source", "memory")
        print(f"{score:.3f}\t[{src}] {mem.text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="<command>",
        description="RAG / long-term memory agent harness",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Answer a question; facts are stored.")
    ask.add_argument("prompt", nargs="+")
    ask.add_argument("--no-write", action="store_true",
                     help="Do not store new memories this turn.")
    _add_common(ask)

    chat = sub.add_parser("chat", help="Interactive chat with persistent memory.")
    chat.add_argument("--no-write", action="store_true")
    _add_common(chat)

    search = sub.add_parser("search", help="Query the memory store directly.")
    search.add_argument("query", nargs="+")
    _add_common(search)

    args = parser.parse_args(argv)
    cfg = Config.from_env()

    if args.command == "search":
        return _cmd_search(cfg, args)

    if not cfg.api_key:
        print("No AGENT_API_KEY set (add to .env or export it).", file=sys.stderr)
        return 2

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    store = store_for(args)
    provider = EmbeddingProvider.from_env()
    retriever = Retriever(store, provider, top_k=args.top_k or cfg.top_k)
    writer = MemoryWriter(store, provider, client=client, model=cfg.model,
                          use_llm=cfg.use_llm_extraction)
    agent = MemoryAgent(client=client, retriever=retriever, writer=writer,
                        model=cfg.model, top_k=args.top_k or cfg.top_k)

    if args.command == "chat":
        from .agent import chat_loop
        chat_loop(agent, write_memory=not args.no_write)
        return 0

    answer = agent.run(" ".join(args.prompt), write_memory=not args.no_write)
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Fix the imports in `cli.py` to match your package (`.agent`, `.retriever`,
`.config`), and remember `ask`/`chat` need `AGENT_API_KEY` while `search` does not.

## Config (implement this in `config.py`)

```python
# src/<package>/config.py
"""Config from env vars / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no external dep)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


@dataclass
class Config:
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    top_k: int = 5
    use_llm_extraction: bool = True

    @classmethod
    def from_env(cls, dotenv_path: str = ".env") -> "Config":
        load_dotenv(dotenv_path)
        return cls(
            model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("AGENT_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("AGENT_API_KEY", ""),
            top_k=int(os.getenv("MEMORY_TOP_K", "5")),
            use_llm_extraction=os.getenv("MEMORY_USE_LLM", "1").lower()
            not in ("0", "false", "no"),
        )
```

## pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "<package-kebab>"
version = "0.1.0"
description = "A RAG / long-term memory agent harness for <spec>"
requires-python = ">=3.10"
dependencies = [
  "openai>=1.0.0",
]

[project.scripts]
<command> = "<package>:cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/<package>"]
```

## .env.example

```
AGENT_API_KEY=sk-...
# AGENT_BASE_URL=https://api.openai.com/v1   # or any OpenAI-compatible endpoint
# AGENT_MODEL=gpt-4o-mini
# MEMORY_TOP_K=5
# MEMORY_USE_LLM=1                            # 0 = naive sentence-split extraction only
# EMBEDDING_BACKEND=ollama                    # or "hash" for offline/deterministic
# OLLAMA_EMBED_URL=http://127.0.0.1:11434/api/embeddings
# OLLAMA_EMBED_MODEL=nomic-embed-text
```

## Verification

1. `python -m pip install -e .` (or `uv sync`).

2. **Offline smoke test** (hash backend, no LLM, no network):
   ```bash
   EMBEDDING_BACKEND=hash python -c "
   from <package>.storage import MemoryStore
   from <package>.embeddings import HashEmbeddingProvider
   from <package>.writer import MemoryWriter
   s = MemoryStore('/tmp/mem/memories.db')
   MemoryWriter(s, HashEmbeddingProvider()).write(
       'The user prefers short answers over long explanations.', 'user')
   "
   EMBEDDING_BACKEND=hash python -m <package>.cli search "how should I answer" --memory-dir /tmp/mem
   ```
   Expect the stored sentence to rank at the top with a score > 0.

3. `python -m <package>.cli --help` — confirm subcommands + flags render, no
   ImportError.

4. With a real key: `python -m <package>.cli ask "remember my coffee is black"`,
   then `ask "what do I drink?"` — the second run should cite the stored memory.
   Repeat in a fresh shell to prove persistence across sessions.

5. `python -m <package>.cli chat` for the REPL; `search` works even with
   `AGENT_API_KEY` unset.

## Conventions

- Same embedding backend per store: hash = 512 dims, nomic = 768; mixing them
  silently scores 0 (see `cosine_similarity`). Changing backends = fresh store,
  or `search`/retrieval go cold.
- Upsert by exact text so repeated turns don't duplicate facts.
- Write memories *after* completing the turn, so the answer's own facts persist.
- temp=0 keeps completion and extraction deterministic; the hash backend keeps
  retrieval deterministic too.
- `search` must never require an API key — it's your debugging/auditing surface.
- Never hardcode or print API keys; ship `.env.example`, never `.env`.

## Example Output

User: "Build me a memory agent that remembers my preferences and past tasks, so
it can ground answers across sessions. Call it `remember-bot` and make memory
queryable from the CLI."

You: scaffold `remember-bot/` with `storage.py`, `embeddings.py` (ollama + hash
backends), `retriever.py`, `writer.py`, `agent.py`, `cli.py`; wire
`remember-bot ask|chat|search` with `--memory-dir`; verify the offline hash
`search` smoke test and `--help`; report the structure and how to run it.