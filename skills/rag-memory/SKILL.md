---
name: harness-rag-memory
description: "Build a RAG / long-term-memory agent harness from a spec — chunking, embeddings, hybrid retrieval and cited answers that stay grounded"
---

# RAG / Memory Agent Harness

An agent whose answers come from a corpus you control: documents are chunked,
embedded and stored; a question retrieves the relevant passages; the model
answers from those and cites them.

A frontier model will write the chunker, the SQLite schema and the cosine
search from a spec. What it will not do is **stop the model answering from its
own weights, make exact names and numbers findable, keep the store valid when
the embedding model changes, or let the agent read past the snippet it was
given**. Those four decide whether the answers are trustworthy.

## Use this when

- Answers must come from a specific corpus, and be attributable to it.
- The corpus is too big for the context window, or changes over time.
- You need memory that survives across sessions.

**Don't use this when:**

- **The corpus fits in context.** Below roughly 50k tokens, put it in the
  prompt. No chunk boundaries, no retrieval misses, no embedding drift. This is
  the most common mistake this pattern invites.
- The question aggregates over everything ("how many documents mention X",
  "what changed between all versions"). Top-k retrieval structurally cannot see
  what it did not rank. Use a query over structured data instead.
- You need exact lookup by key → that is a database, not a retriever.

## Workflow

1. **Get the spec** — what goes in the corpus, how it arrives (URLs, files,
   notes), whether answers need citations, and the embedding endpoint.
2. **Check the corpus size first.** If it fits in context, say so and recommend
   against RAG. Being right here is worth more than building what was asked.
3. **Scaffold** the layout below.
4. **Write ingestion, retrieval and the loop**, with grounding non-optional.
5. **Write the Tier 1 tests** with a deterministic stub embedder.
6. **Verify** — including a retrieval-quality check, not just a passing test.

## The decisions that matter

### 1. Chunk size, and why offsets matter more

Start at **1200 characters with 200 overlap** for prose, and say why in the
code: smaller chunks (≈400) fragment an idea across boundaries and the model
answers from half of it; larger chunks dilute the embedding so retrieval gets
vaguer. Tune per corpus — code and tables want different numbers than prose.

Split at natural boundaries, preferring paragraph breaks, then sentence ends,
then whitespace. And store each chunk's **character offset in its source**:

```python
def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[tuple[int, str]]:
    """Return (start_offset, chunk) pairs. The offset is what lets the agent
    read past the snippet later — without it, a chunk is an orphan."""
```

The offset is what makes `read_source(id, start=...)` possible. An agent that
can only ever see the retrieved snippet will hedge on anything that continues
past the chunk boundary.

### 2. Embeddings are model-specific, and some need prefixes

Two traps, both silent:

- **Task prefixes.** `nomic-embed-text` is trained with `search_document: ` and
  `search_query: ` prefixes and scores measurably worse without them. Other
  models ignore an unknown prefix harmlessly, so applying them is safe and
  omitting them is not.
- **Changing the model invalidates the store.** Vectors from a different model
  are not comparable — search silently returns nonsense rather than failing.
  Store the embedding model name alongside the vectors and refuse to search a
  store built with a different one.

```python
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
```

### 3. Semantic search alone will miss exact strings

Embeddings blur precisely what users search for: version numbers, error codes,
surnames, `4271`. Run a keyword pass (SQLite **FTS5**, which is compiled in
almost everywhere) and merge it with the vector results.

Two things people get wrong when merging:

- **The scores are not comparable.** Cosine similarity is 0–1; FTS5 `bm25()` is
  an unbounded rank where lower is better. Do not sort them in one column as if
  they were one metric. Keep which retriever produced each hit and show it.
- **FTS5 syntax is not user input.** A query containing `"` or `AND` or `(` is
  an FTS expression, and a malformed one raises. Quote each term:

```python
def fts_query(query: str) -> str:
    """Quote every term so user punctuation cannot be read as FTS syntax."""
    terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]
    return " OR ".join(f'"{t}"' for t in terms) or '""'
```

### 4. Retrieval is not the model's decision

The failure this pattern exists to prevent is the model answering from its own
weights. Left to its discretion, it sometimes will — especially on questions
that sound like general knowledge, and especially on small models.

**Run the retrieval before the first model call** and put the results in the
prompt. Keep the search tool available for follow-ups. Grounding becomes a
property of the loop instead of a request the model may ignore, and it saves a
round trip.

Order the prompt so the instructions sit **above** the question: a small model
parrots whatever it read most recently, and a trailing instruction ends up in
the answer.

```python
def grounded_prompt(retriever, question: str, limit: int = 5) -> str:
    hits = retriever.search(question, limit=limit)
    if not hits:
        return (f"Question: {question}\n\nThe corpus returned nothing for this "
                "question. Say so plainly and say what is missing. Do not "
                "answer from memory.")
    return (
        f'Search results for "{question}":\n\n{format_hits(hits)}\n\n---\n'
        "Answer from the results above. Cite each claim with its [S<id>] label "
        "and end with a Sources: line. If they do not answer it, search again "
        "with different wording, or read further into a promising source.\n\n"
        f"Question: {question}"
    )
```

### 5. Brute force is the right default

Cosine similarity over every stored vector is a `numpy` matrix multiply. Tens
of thousands of chunks rank in milliseconds, with no index to build, tune, or
keep in sync with deletes. Reach for FAISS or a vector database when you have
measured a problem, not before. Store vectors as `float32` blobs in the same
SQLite file as the text — one file to back up, copy, or delete.

### 6. Citations need stable ids

Give each source a short id (`S1`, `S4`) and put it in every snippet the model
sees. Require inline `[S1]` citations and a closing `Sources:` list. Without an
id in the snippet, the model invents plausible attributions.

## Build it

```text
<project>/
├── pyproject.toml            # httpx + numpy; no vector DB, no framework
├── src/<package>/
│   ├── config.py             # chunk size/overlap, models, db path, context
│   ├── llm.py                # ChatClient + Embedder + FakeChat (references/llm-seam.md)
│   ├── store.py              # SQLite: sources, chunks, embeddings, FTS5
│   ├── chunking.py           # offset-preserving splitter
│   ├── ingest.py             # fetch/read -> extract -> chunk -> embed -> store
│   ├── retriever.py          # vector + keyword, merged
│   ├── tools.py              # search, read_source, list_sources, save_note
│   ├── agent.py              # the ReAct loop (harness-react) + grounded_prompt
│   └── cli.py                # add, list, search, ask, chat, stats, rm
└── tests/
```

Schema worth getting right the first time:

```sql
CREATE TABLE sources (
    id INTEGER PRIMARY KEY, uri TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
    kind TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '',
    added_at TEXT NOT NULL, text TEXT NOT NULL
);
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL, start INTEGER NOT NULL,
    text TEXT NOT NULL, embedding BLOB
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');
```

`uri UNIQUE` makes re-ingesting a source an update instead of a duplicate.
Keeping the full source text lets `read_source` window into it.

One insertion detail that silently breaks the keyword index: **`executemany`
does not report `lastrowid`**, so the FTS rows cannot be linked. Insert chunks
one at a time and use each `cur.lastrowid`, or the FTS table ends up empty
while every test that only exercises vector search still passes.

The tools the agent gets: `search_library(query, limit, tag)`,
`read_source(source_id, start, length)` — returning a window plus "continues;
read again with start=N" — `list_sources(tag)`, and `save_note(title, body)` so
a synthesis becomes searchable evidence later.

## Failure modes

`references/failure-modes.md` covers the loop. These are retrieval-specific:

- **The model answers from memory.** The headline failure. Guarded by
  pre-retrieval (decision 4), not by asking nicely in the system prompt.
- **Embedding model changed, store not rebuilt.** Search degrades to noise with
  no error. Guard: store the model name with the vectors; refuse mismatches.
- **Boilerplate outranks content.** Nav menus, cookie banners and "Related
  articles" embed well and crowd out the article. Guard: drop `nav`, `header`,
  `footer`, `aside`, `script`, `style` — and `math`, whose MathML serializes to
  one character per line and poisons chunks. On one Wikipedia page this alone
  cut 23.7k characters of extracted text to 18.7k of real prose.
- **Exact strings unfindable.** Guarded by the keyword pass (decision 3).
- **Top-k cannot aggregate.** "How many sources say X" is unanswerable by
  retrieval. Guard: detect aggregate questions and answer them with a query
  over `sources`, or say the corpus cannot answer it.
- **Snippet truncation read as the whole document.** The model concludes "the
  source does not say" when the answer is 200 characters past the cut. Guard:
  offsets plus `read_source`, and say so in the system prompt.
- **Deleting a source leaves orphan chunks.** Guard: `ON DELETE CASCADE` plus
  `PRAGMA foreign_keys = ON` — SQLite ignores the constraint without it — and
  delete the FTS rows explicitly, since an external-content FTS table does not
  cascade.
- **Re-ingest duplicates a source.** Guard: `uri UNIQUE` and replace on
  conflict.

## Required tests

Use a **deterministic stub embedder** — a bag-of-characters vector is enough
for similar text to score higher than unrelated text, and it needs no network:

```python
class StubEmbedder:
    dimension = 16

    def embed_documents(self, texts, batch_size=16):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)

    def _vector(self, text):
        vector = [0.0] * self.dimension
        for char in text.lower():
            if char.isalpha():
                vector[(ord(char) - 97) % self.dimension] += 1.0
        return vector
```

Mandatory:

```python
def test_chunk_offsets_point_back_into_the_source():
    text = "alpha beta gamma. " * 40
    for start, piece in chunk_text(text, size=120, overlap=20):
        assert text[start : start + len(piece)].strip().startswith(piece[:10])


def test_vector_search_ranks_the_matching_source_first(librarian):
    librarian.ingest(Document("u://1", "Vectors", "embeddings and cosine similarity"))
    librarian.ingest(Document("u://2", "Baking", "flour butter sugar oven"))
    assert librarian.search("cosine similarity", limit=1)[0].source.title == "Vectors"


def test_keyword_search_finds_exact_numbers(library, librarian):
    librarian.ingest(Document("u://1", "Specs", "throughput was 4271 tokens per second"))
    assert "4271" in library.search_keyword("4271", limit=1)[0].chunk.text


def test_keyword_search_survives_punctuation(library):
    library.search_keyword('"quoted" AND (weird)', limit=5)      # must not raise


def test_reingesting_the_same_uri_replaces_it(librarian, library):
    librarian.ingest(Document("u://1", "v1", "first version"))
    librarian.ingest(Document("u://1", "v2", "second version"))
    assert len(library.sources()) == 1
    assert library.stats()["chunks"] == 1


def test_deleting_a_source_removes_its_chunks(librarian, library):
    result = librarian.ingest(Document("u://1", "T", "alpha beta gamma. " * 50))
    library.delete_source(result.source_id)
    assert library.stats() == {"sources": 0, "chunks": 0, "characters": 0}


def test_grounded_prompt_is_explicit_when_nothing_matches(librarian):
    prompt = grounded_prompt(librarian, "anything")
    assert "Do not answer from memory" in prompt
```

Plus the ReAct loop tests from `harness-react`.

## Verify

1. **Tier 0** — install, import, `--help`.
2. **Tier 1** — `pytest -q`, offline with the stub embedder.
3. **Retrieval quality, by hand.** Tests prove plumbing, not relevance:

   ```bash
   <command> add <a real document>
   <command> search "<a question you know the answer to>"
   ```

   Read the top 3 hits. If boilerplate is outranking content, fix extraction
   before touching the prompt — no prompt recovers from bad retrieval.
4. **Tier 2** — one real `ask`, checking the citations point at sources that
   actually contain the claim. An uncited or mis-cited answer is a failure even
   when the content is right.

Report what ran, and say plainly if retrieval quality was not hand-checked.
