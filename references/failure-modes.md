# Agent loop failure modes

Every failure here has been observed against a real endpoint. None of them is
prevented by writing a *correct* agent loop — a loop can be textbook-correct
and still hit all fifteen. They are the difference between a harness that
demos and one that runs unattended.

Each entry gives the **symptom** (what you actually see), the **cause**, and
the **guard** (what to put in the loop). Guards are cheap. Add them when you
scaffold, not after the first incident.

---

## A. The model sends malformed tool arguments

Small models (3B–8B) do this constantly; frontier models do it occasionally
under load or with deep schemas. All three are repaired the same way — by
validating arguments against the schema you already declared.

### A1. The schema is echoed back as the value

**Symptom.** `TypeError: slice indices must be integers` deep inside your
tool, or a nonsense result. Logging the call shows:

```text
search_library(query='...', limit={'type': 'integer'})
```

**Cause.** The model copies the parameter's schema fragment instead of
producing a value. It is not confused about intent — only about serialization.

### A2. Numbers and booleans arrive as strings

**Symptom.** `limit="5"`, `enabled="true"`. Python happily accepts them and
then fails somewhere unrelated (`range("5")`, or a string used as a flag —
note `bool("false")` is `True`).

### A3. Invented parameters

**Symptom.** `TypeError: f() got an unexpected keyword argument 'context'`.
The model adds a plausible-sounding argument your function does not take.

### Guard: coerce against the declared schema

One function fixes all three. Drop what cannot be repaired so the parameter
falls back to its default, rather than raising:

```python
class _Unusable:
    """Sentinel: an argument that cannot be repaired into its declared type."""


UNUSABLE = _Unusable()


def coerce_value(value, expected: str):
    if isinstance(value, dict) and expected != "object":
        return UNUSABLE  # A1: the model echoed the schema.
    if expected in ("integer", "number"):
        kind = int if expected == "integer" else float
        if isinstance(value, bool):
            return UNUSABLE
        if isinstance(value, (int, float)):
            return kind(value)
        if isinstance(value, str):
            try:
                return kind(float(value.strip()))
            except ValueError:
                return UNUSABLE
        return UNUSABLE
    if expected == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return UNUSABLE
    if expected == "string":
        return value if isinstance(value, str) else str(value)
    if expected == "array":
        if isinstance(value, list):
            return value
        return [value] if isinstance(value, str) else UNUSABLE
    return value


def coerce_arguments(arguments: dict, properties: dict) -> dict:
    """Keep only declared parameters, each repaired into its declared type."""
    cleaned = {}
    for key, value in (arguments or {}).items():
        schema = properties.get(key)
        if schema is None:
            continue  # A3: hallucinated parameter.
        converted = coerce_value(value, schema.get("type", "string"))
        if converted is not UNUSABLE:
            cleaned[key] = converted
    return cleaned
```

Never invent a *missing required* argument. Let the call fail and return the
error to the model as text (see C1) — it will retry with the argument.

---

## B. The loop does not terminate usefully

### B1. The same tool call, over and over

**Symptom.** Identical search three times, context growing, then a timeout.

**Cause.** The result did not obviously answer the question, so the model
tries again — with the same query, because the question has not changed.

**Guard.** Key each call by name + arguments. On a repeat, return the cached
result with an instruction not to repeat it. This costs one dict and removes
the most common spiral:

```python
key = f"{name}:{sorted((arguments or {}).items(), key=str)}"
if key in seen:
    result = (
        "You already ran this exact call and got the result below. Do not run "
        "it again — either answer from what you have, or try a materially "
        "different query.\n\n" + seen[key]
    )
else:
    result = execute(name, arguments)
    seen[key] = result
```

### B2. The budget is spent and there is no answer

**Symptom.** `RuntimeError: max steps exceeded` — the user gets an exception
instead of the answer the model was close to writing.

**Guard.** Spend the last turn differently: ask once more **with no tools
offered**. With no tools in the request, the model has nothing to emit but
prose. Raise only if that also comes back empty.

```python
def force_answer(client, messages):
    messages.append({"role": "user", "content": (
        "Stop searching and answer now using only the results above. "
        "If they do not answer the question, say exactly what is missing."
    )})
    return client.complete(messages, tools=None)
```

### B3. Empty content with no tool calls

**Symptom.** The loop returns `""`. The user sees nothing and no error.

**Cause.** The model emitted only whitespace, or a stop token immediately.
`response.content or ""` silently converts this into a successful empty
answer.

**Guard.** Treat empty content with no tool calls as a failed turn: retry
once, then fall back to B2's forced answer. Never return `""` as success.

---

## C. Tool execution

### C1. A tool exception kills the run

**Symptom.** A 404 inside one tool ends the whole session.

**Guard.** The executor catches everything and returns the error *as the tool
result string*. The model reads it and adapts — wrong argument, wrong tool,
retry. This is the single highest-value line in a tool executor:

```python
try:
    return tool.run(**arguments)
except Exception as exc:  # noqa: BLE001 - errors are for the model to read
    return f"Error: {type(exc).__name__}: {exc}"
```

Return unknown-tool as text too, listing the available names — the model
usually picks the right one on the retry.

### C2. Parallel tool calls with missing results

**Symptom.** With OpenAI: `400 — 'tool_call_id' did not have response
messages`. The run dies on the *next* request, not the one that caused it.

**Cause.** The model emitted three tool calls in one turn and the loop
appended only one result.

**Guard.** Iterate over **every** call in `response.tool_calls` and append one
result message per call, in order, before the next request. Even for a failed
or skipped call, append something.

---

## D. Context and transport

### D1. Silent context truncation

**Symptom.** Mid-run, the model "forgets" its instructions — stops citing,
stops using tools, answers from memory.

**Cause.** The context window filled and the front of the conversation (your
system prompt) was dropped. Ollama defaults to **4096 tokens** regardless of
what the model supports; a few tool results overflow it.

**Guard.** Set the window explicitly — `options.num_ctx` on Ollama,
and track message length yourself. Budget backwards: system prompt + tool
results × steps must fit with room to answer.

### D2. "Server disconnected without sending a response"

**Symptom.** An `httpx.RemoteProtocolError` about 40 seconds into the first
request. No error in the API response, because there is no response.

**Cause.** The model did not fit in RAM and the server was OOM-killed while
loading. Confirm with `journalctl --user -u ollama | grep -i oom`.

**Guard.** Check free memory before choosing a default model. Roughly: a
quantized model needs its file size plus ~25% for context and overhead. A 9 GB
14B model needs ~12 GB free. Fail loudly with that arithmetic rather than
letting the user read a transport error.

### D3. Unbounded message growth

**Symptom.** Each step is slower and more expensive than the last; step 8
costs 10× step 1.

**Cause.** Every step appends an assistant message and a tool result, and the
whole list is resent. Cost grows quadratically with steps.

**Guard.** Cap tool-result size before appending (truncate with an explicit
`... truncated, N chars total` marker so the model knows to read more), and
keep step budgets small. Prefer better retrieval over more steps.

---

## E. Determinism and provider divergence

### E1. Flaky tool selection

**Guard.** Use `temperature=0` for any turn that chooses a tool. Sampling
buys nothing when the task is "pick the right function". Raise temperature
only for turns that write prose, if at all.

### E2. Providers disagree about shapes

The same logical conversation is spelled differently per provider. Normalize
at the client boundary — never leak these differences into the loop:

| Concern | OpenAI-compatible | Ollama native (`/api/chat`) |
| --- | --- | --- |
| Tool call arguments | JSON **string**, needs `json.loads` | already a **dict** |
| Tool call id | `call.id`, required on the result | often absent; synthesize one |
| Tool result message | `{"role":"tool","tool_call_id":...}` | `{"role":"tool","tool_name":...}` |
| Context window | model default | `options.num_ctx`, defaults to 4096 |
| Sampling | `temperature` top level | inside `options` |

**Guard.** Your loop should speak one internal shape (`ModelResponse`,
`ToolCall(id, name, arguments: dict)`). Each client adapts to it. This is also
what makes the loop testable — see `llm-seam.md`.

### E3. The model answers from memory instead of using a tool

**Symptom.** A retrieval agent answers a question about the user's private
data with general knowledge, confidently and without citation.

**Guard.** Do not leave the first retrieval to the model's discretion. Run it
before the first model call and put the results in the prompt. Tools stay
available for follow-ups. Grounding becomes a property of the loop rather than
a request the model may ignore.

---

## Minimum guard set

If you take nothing else, a loop is not production-shaped without these five:

1. Tool errors returned as text, never raised (C1).
2. One result message per tool call (C2).
3. Arguments coerced against the declared schema (A1–A3).
4. A forced, toolless final turn when the budget runs out (B2).
5. `temperature=0` on tool-selecting turns (E1).
