---
name: harness-react
description: "Build a ReAct (reason-act-observe) agent harness from a spec — a tool-calling loop with the guards that keep it terminating, testable, and recoverable"
---

# ReAct Agent Harness

A ReAct agent interleaves reasoning and tool use: the model thinks, calls a
tool, reads the result, and repeats until it answers or runs out of budget.
It is the default agent shape and the foundation the other patterns extend.

Writing the loop is easy and a frontier model will do it unprompted. What it
will not do unprompted is make the loop **terminate under stress, survive a
tool that raises, stay testable without an API key, and normalize provider
differences**. That is what this skill is for. Build the loop quickly; spend
your effort on the guards.

## Use this when

- The agent needs tools and the number of steps is not known in advance.
- Each step's result should inform the next decision.
- You want the simplest thing that can still be called an agent.

**Don't use this when:**

- The task has a fixed sequence of steps → write a script, not an agent.
- A long task loses coherence halfway → `harness-plan-and-execute`.
- The work splits into roles with unrelated context → `harness-multi-agent`.
- Every answer must be grounded in a corpus → `harness-rag-memory` (which is
  this loop plus retrieval).

See `references/choosing-a-pattern.md` before committing.

## Workflow

1. **Get the spec.** Domain, the 3–5 tools that matter, the endpoint (OpenAI,
   Ollama, vLLM…), and the model. If the spec is vague, propose tools and
   confirm before writing code.
2. **Check the endpoint supports tool calling.** This decides the protocol
   (below) and is the one question that changes the whole build.
3. **Scaffold** — layout below. Standard packaging; do not overthink it.
4. **Write the loop with every guard** in "Failure modes". Not afterwards.
5. **Write the Tier 1 tests** from "Required tests". They run without a key.
6. **Verify** and report honestly what ran.

## The decisions that matter

### 1. Native tool calling, or JSON in the content?

The old ReAct papers predate tool-calling APIs and ask the model to emit
`{"thought": ..., "action": ...}` as text. That is still the right choice in
one specific case and the wrong choice otherwise.

| | Native tool calls | JSON in content |
| --- | --- | --- |
| Reliability | Constrained decoding, provider-validated | Model may emit prose, fences, trailing commas |
| Parallel calls | Supported | Awkward |
| Works with | Most models since 2024 | Anything, including base models |
| Code cost | Provider adapter | Fence stripping + JSON repair + retry ladder |

**Default to native tool calls.** Choose JSON-in-content only when the target
model genuinely cannot do tool calls — then budget for the repair path:

```python
def parse_json_reply(content: str) -> dict | None:
    """Tolerate fenced and prose-wrapped JSON. Returns None if unrecoverable."""
    import json

    text = content.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text[len(fence):]
        if text.startswith("json"):
            text = text[4:]
        text = text.split(fence)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
```

Retry at most twice with an explicit "reply with only a JSON object" message,
then give up with a clear error. An unbounded retry loop here is its own bug.

### 2. What the step budget counts

Count **model calls**, not tool calls. A turn with three parallel tool calls is
one step. This is the number that bounds cost and latency.

Pick the budget from the work: 4–6 for a focused lookup agent, 8–12 for
research. Higher is rarely better — a loop that needs 20 steps usually has a
retrieval problem, not a budget problem.

### 3. What happens when the budget runs out

Do not raise. Ask once more with `tools=None` so the model has nothing to emit
but prose (see `failure-modes.md` B2). Raise only if that is also empty.

### 4. Where the reasoning goes

With native tool calls, many models return **empty content plus a tool call** —
the thought is gone. If you want a visible trace, either ask for a one-line
rationale in the system prompt, or emit it through hooks. Do not parse it out
of content; it is often not there.

### 5. Observability is a constructor argument

Add event hooks from the start — `on_llm_call`, `on_tool_call`,
`on_tool_result`, `on_error`. A `-v` flag that prints the tool calls is the
difference between debugging in a minute and guessing for an hour. Hooks keep
the loop free of `print` statements and let tests observe behaviour.

## Build it

```text
<project>/
├── pyproject.toml            # hatchling, src layout, one script entry point
├── .env.example
├── README.md
├── src/<package>/
│   ├── config.py             # env-driven dataclass; temperature defaults to 0
│   ├── llm.py                # ChatClient ABC, ModelResponse, ToolCall, FakeChat, adapter
│   ├── messages.py           # system/user/assistant/tool_result builders
│   ├── tools.py              # @tool decorator, schema inference, coercion, executor
│   ├── agent.py              # the loop
│   ├── cli.py                # ask + chat, -v for hooks
│   └── tools/examples.py     # spec-specific tools
└── tests/                    # Tier 1, offline
```

Packaging, config dataclass, argparse CLI and type-hint-to-JSON-schema
inference are standard — write them in your usual style. Two modules are not
standard and carry the value: `llm.py` and `agent.py`.

**`llm.py`** — copy the seam from `references/llm-seam.md` verbatim: the
`ChatClient` ABC, `ToolCall`, `ModelResponse`, `FakeChat`, and the adapter for
the target provider. `FakeChat` ships in the package, not in tests.

**`agent.py`** — the loop, with the guards inline:

```python
from __future__ import annotations

from typing import Any, Callable

from .llm import ChatClient, ModelResponse
from .messages import assistant, system, tool_result, user
from .tools import Tool, ToolExecutor

FORCE_ANSWER = (
    "Stop calling tools and answer now, using only the results above. "
    "If they do not answer the question, say exactly what is missing."
)

REPEAT_NUDGE = (
    "You already ran this exact call and got the result below. Do not run it "
    "again — either answer from what you have, or try a different approach.\n\n"
)


class StepBudgetExceeded(RuntimeError):
    """The model would not stop calling tools, even when asked directly."""


class Agent:
    def __init__(self, client, tools=None, config=None, system_prompt=None, hooks=None):
        self.client = client
        self.config = config
        self.executor = ToolExecutor(tools or [])
        self.system_prompt = system_prompt
        self.hooks: dict[str, Callable[..., Any]] = hooks or {}
        self.messages: list[dict[str, Any]] = []
        self._seen: dict[str, str] = {}

    def _emit(self, event: str, **data: Any) -> None:
        hook = self.hooks.get(event)
        if hook:
            hook(**data)

    def _record(self, response: ModelResponse) -> None:
        calls = [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": c.arguments}}
            for c in response.tool_calls
        ]
        self.messages.append(assistant(response.content, calls))

    def _run_tools(self, response: ModelResponse, step: int) -> None:
        # One result message per call, always — a missing one breaks the
        # NEXT request, not this one.
        for call in response.tool_calls:
            self._emit("on_tool_call", step=step, name=call.name, arguments=call.arguments)

            key = f"{call.name}:{sorted((call.arguments or {}).items(), key=str)}"
            if key in self._seen:
                result = REPEAT_NUDGE + self._seen[key]
            else:
                result = self.executor.execute(call.name, call.arguments)
                self._seen[key] = result

            self._emit("on_tool_result", step=step, name=call.name, result=result)
            self.messages.append(tool_result(call.id, call.name, result))

    def run(self, prompt: str) -> str:
        if not self.messages and self.system_prompt:
            self.messages.append(system(self.system_prompt))
        self.messages.append(user(prompt))

        schemas = self.executor.schemas() if self.executor else None

        for step in range(self.config.max_steps):
            self._emit("on_llm_call", step=step)
            try:
                response = self.client.complete(self.messages, tools=schemas)
            except Exception as exc:  # noqa: BLE001 - surface transport failures
                self._emit("on_error", step=step, error=exc)
                raise

            self._emit("on_llm_response", step=step,
                       content=response.content, tool_calls=response.tool_calls)
            self._record(response)

            if response.tool_calls:
                self._run_tools(response, step)
                continue

            if response.content and response.content.strip():
                return response.content

            # Empty content and no tool calls is a failed turn, not an answer.
            self.messages.append(user("That reply was empty. Answer the question."))

        return self._force_answer()

    def _force_answer(self) -> str:
        """Last turn: no tools offered, so prose is the only thing to emit."""
        self._emit("on_force_answer", step=self.config.max_steps)
        self.messages.append(user(FORCE_ANSWER))
        response = self.client.complete(self.messages, tools=None)
        self._record(response)
        if response.content and response.content.strip():
            return response.content
        raise StepBudgetExceeded(
            f"The model called tools for {self.config.max_steps} steps and produced "
            "no answer even when asked directly. Narrow the task or raise the budget."
        )

    def reset(self) -> None:
        self.messages = []
        self._seen = {}
```

**`tools.py`** — the executor must never raise into the loop, and must repair
arguments before calling through. Both from `references/failure-modes.md`:

```python
class ToolExecutor:
    def __init__(self, tools=None):
        self._tools = {t.name: t for t in (tools or [])}

    def __bool__(self) -> bool:
        return bool(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self.names()) or "none"
            return f"Error: unknown tool '{name}'. Available tools: {available}"
        try:
            return tool.run(**tool.coerce(arguments))
        except TypeError as exc:
            return f"Error: invalid arguments for '{name}': {exc}"
        except Exception as exc:  # noqa: BLE001 - the model reads and retries
            return f"Error: {type(exc).__name__}: {exc}"
```

## Failure modes

Read `references/failure-modes.md` in full — all fifteen apply to this loop,
because every other pattern inherits it. The five that bite a ReAct agent
first, and where each is handled above:

| Failure | Guard | Where |
| --- | --- | --- |
| A tool raises and kills the run | Return the error as the tool result | `ToolExecutor.execute` |
| Model repeats the same call until timeout | Cache by name+args, nudge | `_run_tools` |
| Budget spent, user gets an exception | Toolless final turn | `_force_answer` |
| Empty reply returned as a successful answer | Treat as a failed turn | `run` |
| Parallel calls, one result appended | Loop over every call | `_run_tools` |

Pattern-specific, beyond the shared list:

- **The model answers without using tools at all.** Common on small models and
  on questions that *sound* like general knowledge. If the agent's whole value
  is its tools, do not leave the first call to discretion — run the obvious
  tool yourself and put the result in the prompt.
- **Tool descriptions are the real prompt.** Most "the agent picked the wrong
  tool" bugs are a docstring problem, not a model problem. Say what the tool is
  *for* and when not to use it.
- **`max_steps=1` is a valid configuration** and a good smoke test: it proves
  the forced-answer path works without waiting for a spiral.

## Required tests

All offline, using `FakeChat`. See `references/verification.md` for the full
table; these are mandatory for this pattern:

```python
def test_plain_answer_returns_immediately(config):
    agent = Agent(client=FakeChat([ModelResponse(content="done")]), config=config)
    assert agent.run("q") == "done"
    assert agent.client.calls == 1


def test_tool_error_is_reported_not_raised(config):
    responses = [
        ModelResponse(tool_calls=[ToolCall("c0", "boom", {})]),
        ModelResponse(content="recovered"),
    ]
    agent = Agent(client=FakeChat(responses), tools=[boom], config=config)
    assert agent.run("q") == "recovered"
    assert "Error: ValueError" in [m for m in agent.messages if m["role"] == "tool"][0]["content"]


def test_budget_forces_a_toolless_answer(config):
    looping = ModelResponse(tool_calls=[ToolCall("c0", "search", {})])
    responses = [looping] * config.max_steps + [ModelResponse(content="forced")]
    agent = Agent(client=FakeChat(responses), tools=[search], config=config)
    assert agent.run("q") == "forced"
    assert agent.client.last_tools is None


def test_repeated_call_is_served_from_cache(config):
    call = ToolCall("c0", "search", {"q": "x"})
    responses = [
        ModelResponse(tool_calls=[call]),
        ModelResponse(tool_calls=[call]),
        ModelResponse(content="done"),
    ]
    agent = Agent(client=FakeChat(responses), tools=[search], config=config)
    agent.run("q")
    results = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert "You already ran this exact call" in results[1]


def test_every_parallel_call_gets_a_result(config):
    calls = [ToolCall("c0", "search", {"q": "a"}), ToolCall("c1", "search", {"q": "b"})]
    responses = [ModelResponse(tool_calls=calls), ModelResponse(content="done")]
    agent = Agent(client=FakeChat(responses), tools=[search], config=config)
    agent.run("q")
    assert len([m for m in agent.messages if m["role"] == "tool"]) == 2
```

## Verify

Follow `references/verification.md`. For this pattern specifically:

1. **Tier 0** — `pip install -e ".[dev]"`, import, `--help`.
2. **Tier 1** — `pytest -q`. All tests above pass, no network, no key.
3. **Tier 2** — one real run with `-v`, against the endpoint from the spec:

   ```bash
   <command> "<question needing at least one tool>" -v
   ```

   Confirm by eye: a tool was called with sensible arguments, the result was
   used, and the run ended on prose rather than the budget.

Then tell the user what actually ran. If there was no API key and Tier 2 was
skipped, say so plainly rather than implying an end-to-end run happened.
