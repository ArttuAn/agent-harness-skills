# The LLM seam

The single architectural decision that determines whether an agent harness can
be tested. Everything else — layout, packaging, CLI — is recoverable later.
This is not.

## The problem

The natural way to write an agent loop is the untestable way:

```python
class Agent:
    def __init__(self, model="gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()          # constructed inside; nothing can replace it

    def run(self, prompt):
        response = self.client.chat.completions.create(...)
```

Now every test of the loop needs a network, an API key, money, and patience —
and still cannot reproduce the cases you actually care about, because you
cannot make a real model reliably emit a malformed argument or refuse to stop
calling tools. So the guards in `failure-modes.md` go untested, which means in
practice they go unwritten.

## The seam

Define your own tiny vocabulary, accept a client that speaks it, and let each
provider adapt to it. Three dataclasses and one method:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]   # always a dict, whatever the provider sent


@dataclass
class ModelResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatClient(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """Send the conversation and return the model's reply."""
```

The loop takes a `ChatClient` and never imports a provider SDK. Two rules keep
the seam honest:

1. **Normalize inside the client.** Arguments are a dict by the time the loop
   sees them, whether the provider sent a dict or a JSON string. Ids exist even
   when the provider omits them.
2. **Nothing provider-shaped crosses the boundary.** No `openai` types, no raw
   response objects.

## The fake

Twelve lines that make the whole loop testable offline:

```python
class FakeChat(ChatClient):
    """Replays scripted responses. An entry may be a ModelResponse, or a
    callable taking the current messages — for asserting on what was sent."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = 0
        self.last_tools = None

    def complete(self, messages, tools=None):
        self.last_tools = tools
        self.calls += 1
        entry = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return entry(messages) if callable(entry) else entry
```

Clamping the index to the last entry is deliberate: a one-element script that
returns a tool call replays forever, which is exactly how you test a spiral.

## What this buys you

Every guard becomes a fast, deterministic test — no network, no key:

```python
def test_tool_error_is_reported_not_raised():
    responses = [
        ModelResponse(tool_calls=[ToolCall("c0", "explodes", {})]),
        ModelResponse(content="recovered"),
    ]
    agent = Agent(client=FakeChat(responses), tools=[explodes])

    assert agent.run("go") == "recovered"
    tool_messages = [m for m in agent.messages if m["role"] == "tool"]
    assert "Error: ValueError" in tool_messages[0]["content"]


def test_spent_budget_forces_a_toolless_answer():
    looping = ModelResponse(tool_calls=[ToolCall("c0", "search", {})])
    agent = Agent(
        client=FakeChat([looping] * agent_config.max_steps + [ModelResponse(content="forced")]),
        tools=[search],
        config=agent_config,
    )

    assert agent.run("go") == "forced"
    assert agent.client.last_tools is None   # the final turn offered no tools
```

The second test is the point. You cannot write it against a real endpoint,
because you cannot make a real model loop on demand — and it is the test that
proves your harness terminates.

## Provider adapters

Two adapters cover almost everything. Both return the same `ModelResponse`.

**OpenAI-compatible** (OpenAI, vLLM, LM Studio, LiteLLM, Ollama's `/v1`):

```python
class OpenAIChat(ChatClient):
    def __init__(self, config):
        from openai import OpenAI
        self._client = OpenAI(api_key=config.api_key or "local", base_url=config.base_url)
        self._config = config

    def complete(self, messages, tools=None):
        import json
        request = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
        }
        if tools:
            request["tools"] = tools
        message = self._client.chat.completions.create(**request).choices[0].message

        calls = []
        for call in message.tool_calls or []:
            # OpenAI sends arguments as a JSON *string*.
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(ToolCall(call.id, call.function.name, arguments))
        return ModelResponse(content=message.content, tool_calls=calls)
```

**Ollama native** (`/api/chat`) — worth using directly when you want
`num_ctx` control and no SDK dependency:

```python
class OllamaChat(ChatClient):
    def __init__(self, config):
        import httpx
        self._client = httpx.Client(base_url=config.ollama_url, timeout=config.timeout)
        self._config = config

    def complete(self, messages, tools=None):
        payload = {
            "model": self._config.model,
            "messages": messages,
            "stream": False,
            # Ollama defaults to 4096 regardless of what the model supports.
            "options": {
                "temperature": self._config.temperature,
                "num_ctx": self._config.context,
            },
        }
        if tools:
            payload["tools"] = tools

        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        message = response.json().get("message", {})

        calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function", {})
            # Ollama sends arguments already parsed, and often omits the id.
            calls.append(ToolCall(
                id=call.get("id") or f"call_{index}",
                name=function.get("name", ""),
                arguments=function.get("arguments") or {},
            ))
        return ModelResponse(content=message.get("content") or None, tool_calls=calls)
```

Tool-result messages differ too — OpenAI keys them by `tool_call_id`, Ollama by
`tool_name`. Put that in a `tool_result()` helper next to your other message
builders so the loop never has to know.

## Rules

- The loop's constructor takes a client. It never constructs one.
- The client is the only module that imports a provider SDK.
- `FakeChat` ships in the package, not the tests — it is part of the contract,
  and downstream users testing their own tools need it.
- Every guard in `failure-modes.md` gets a test that uses `FakeChat`.
