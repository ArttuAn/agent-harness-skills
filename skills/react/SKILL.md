---
name: harness-react
description: "Build a ReAct (Reason-Act-Observe) pattern agent harness from a spec"
---

# ReAct Agent Harness

## What is ReAct?

ReAct (Reason + Act) is a pattern where an agent interleaves reasoning and tool use
in a loop:

```
Thought:  the LLM reasons about the current state and picks a next step
Action:   the LLM emits a tool call to act on the world
Observation: the tool result is fed back into the LLM prompt
```

The loop repeats until the agent emits a final answer or the step budget is
exhausted. This is the generic loop behind agents like WebGPT and many modern
tool-using assistants.

## Workflow

When the user gives you a spec (or says "build me a ReAct agent"), do the following:

1. **Ask for a spec if none given.** The user should roughly describe the domain:
   what the agent should be able to do, and what example tools make sense. If the
   spec is vague, propose 3-5 example tools yourself and confirm.

2. **Scaffold the project** into the directory the user specifies (default: create
   `<folder-name>/` next to where they're working). Use a clean `src/` layout with
   `pyproject.toml`.

3. **Write the full ReAct loop** (see the implementation below — put it in your
   own words as needed, but it must be complete and runnable).

4. **Write the tool registry** with `@tool` decorator + type-hint-to-JSON-schema
   inference.

5. **Write config handling** via env vars / `.env`, plus a **CLI entry point** and
   **chat mode**.

6. **Add a handful of example tools** relevant to the spec. Then verify the project
   imports and the CLI `--help` runs.

## Project Layout

Create this structure in the target directory:

```
<project>/
├── pyproject.toml
├── .env.example
├── README.md
└── src/<package>/
    ├── __init__.py
    ├── config.py        # env-based config (model, base_url, api_key, max_steps)
    ├── loop.py          # the ReAct loop + chat loop
    ├── tools.py         # @tool decorator, registry, schema inference
    ├── cli.py           # entry point
    └── tools/           # example tools
        ├── __init__.py
        └── examples.py
```

Use the kebab-case project name the user picked for the folder, and use the
snake_case version as the Python package name (e.g. folder `weather-bot` →
package `weather_bot`). The CLI command name matches the kebab-case name as is.

## Dependencies

Use only:

- `openai` (for LLM calls)
- `python-dotenv` (optional, for `.env` loading — it's tiny and standard; if the
  user objects, read `os.getenv` only)
- stdlib (`json`, `typing`, `inspect`, `abc`, `collections`, etc.)

Do **not** pull in langchain, instructor, instructor-style validators, or any other
heavy dependencies. Schema inference is hand-rolled in ~30 lines.

## The Agent Loop (implement this in `loop.py`)

```python
# src/<package>/loop.py
"""The generic ReAct loop: Thought -> Action -> Observation -> repeat."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .tools import ToolRegistry

SYSTEM_PROMPT = """You are a ReAct agent. You reason and act in a loop.

For each turn, respond with EXACTLY one JSON object of one of these two shapes:

1. To take an action:
{"thought": "<your reasoning>", "action": {"name": "<tool>", "args": {"<param>": <value>}}}

2. To finish:
{"thought": "<your final reasoning>", "answer": "<final answer to the user>"}

Available tools:
{tools}
"""


class ReActLoop:
    def __init__(
        self,
        client: OpenAI,
        registry: ToolRegistry,
        model: str,
        max_steps: int = 10,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def _build_system_prompt(self) -> str:
        tools = self.registry.describe()
        return (self.system_prompt or SYSTEM_PROMPT).format(tools=tools)

    def run(self, user_input: str, *, verbose: bool = True) -> str:
        """Run the ReAct loop until an answer is produced or max_steps hits.

        Returns the final answer string.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        for step in range(1, self.max_steps + 1):
            # 1) Thought + Action/Answer decision from the LLM
            if verbose:
                print(f"\n[step {step}/{self.max_steps}] asking model...")
            decision = self._ask(messages)

            thought = decision.get("thought", "")
            if verbose:
                print(f"[thought] {thought}")

            if "answer" in decision:
                answer = decision["answer"]
                if verbose:
                    print(f"[answer] {answer}")
                return str(answer)

            action = decision.get("action")
            if not isinstance(action, dict) or "name" not in action:
                messages.append({
                    "role": "user",
                    "content": (
                        "Invalid response. You must output a JSON object with "
                        "either an 'answer' field or an 'action' field."
                    ),
                })
                continue

            name, args = action["name"], action.get("args", {})
            if verbose:
                print(f"[action] {name}({args})")

            # 2) Execute the tool
            observation = self.registry.execute(name, args)
            if verbose:
                print(f"[observation] {observation}")

            # 3) Feed the observation back and repeat
            messages.append({
                "role": "assistant",
                "content": json.dumps(decision, ensure_ascii=False),
            })
            messages.append({
                "role": "user",
                "content": f"Observation from {name}: {observation}\nContinue.",
            })

        return f"Step budget ({self.max_steps}) exceeded. Last observation: {observation}"

    def _ask(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Send the message list, get a JSON object back. Retries on bad JSON."""
        for _ in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
            # Tolerate fenced JSON responses (fence = 3 x chr(96))
            fence = chr(96) * 3
            if content.startswith(fence):
                content = content[len(fence):]
                if content.endswith(fence):
                    content = content[: -len(fence)]
                content = content.strip()
                if content.startswith("json"):
                    content = content[4:]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                messages.append({
                    "role": "user",
                    "content": (
                        "You did not output valid JSON. Reply with only a JSON object."
                    ),
                })
        raise RuntimeError("Model repeatedly failed to produce valid JSON.")


def chat_loop(loop: ReActLoop) -> None:
    """Interactive REPL. The loop object's config is reused, but you may want
    to thread `max_steps`/fresh system prompt per turn."""
    print("ReAct agent ready. Type 'exit' or Ctrl-D to quit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user_input in {"exit", "quit", "q"}:
            return
        if not user_input:
            continue
        answer = loop.run(user_input)
        print(f"\nAgent: {answer}")
```

## The Tool Registry (implement this in `tools.py`)

```python
# src/<package>/tools.py
"""Registry + @tool decorator with type-hint-to-JSON-schema inference."""

from __future__ import annotations

import inspect
import types
import typing

import json

def type_to_schema(t: type) -> dict:
    """Map a Python type annotation to a JSON-schema-ish type descriptor."""
    origin = typing.get_origin(t)
    if origin is None:
        if t is int:
            return {"type": "integer"}
        if t is float:
            return {"type": "number"}
        if t is bool:
            return {"type": "boolean"}
        if t is str:
            return {"type": "string"}
        if t in (list, typing.List):
            return {"type": "array", "items": {"type": "string"}}
        if t in (dict, typing.Dict):
            return {"type": "object"}
        return {"type": "string"}  # fallback: treat anything else as string
    if origin is typing.Union:
        # Optional[...] -> pick the first non-None branch
        for arg in typing.get_args(t):
            if arg is not type(None):
                return type_to_schema(arg)
    if origin is list or origin is typing.List:
        (item,) = typing.get_args(t)
        return {"type": "array", "items": type_to_schema(item)}
    if origin is dict or origin is typing.Dict:
        values = typing.get_args(t)
        if values:
            return {"type": "object", "additionalProperties": type_to_schema(values[1])}
        return {"type": "object"}
    if origin is typing.Literal:
        choices = [str(a) for a in typing.get_args(t)]
        return {"type": "string", "enum": choices}
    # fallback
    return {"type": "string"}


class Tool:
    def __init__(self, fn, name: str | None = None, description: str | None = None):
        self.fn = fn
        self.name = name or fn.__name__
        sig = inspect.signature(fn)
        self.description = (
            description
            if description is not None
            else (fn.__doc__ or f"Call the {self.name} tool.").strip()
        )
        self.parameters = {}
        for pname, param in sig.parameters.items():
            if pname == "return":
                continue
            annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
            pschema = type_to_schema(annotation)
            if param.default is not inspect.Parameter.empty:
                pschema["default"] = param.default
                pschema["required"] = False
            else:
                pschema["required"] = True
            self.parameters[pname] = pschema
        self.required = [
            p for p, ps in self.parameters.items() if ps.get("required")
        ]

    def to_openai_schema(self) -> dict:
        """Format for the tools list you pass to openai if you use native tool-calls."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        p: {k: v for k, v in s.items() if k != "required"}
                        for p, s in self.parameters.items()
                    },
                    "required": self.required,
                },
            },
        }

    def __call__(self, **kwargs: typing.Any) -> typing.Any:
        return self.fn(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, fn=None, *, name: str | None = None, description: str | None = None):
        """Usage: @registry.register or @registry.register(name="x")."""
        def wrap(f):
            tool = Tool(f, name=name, description=description)
            self._tools[tool.name] = tool
            return tool
        if fn is not None:
            return wrap(fn)
        return wrap

    def execute(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'. Available: {', '.join(self._tools)}"
        try:
            result = self._tools[name](**args)
            return json.dumps(result, ensure_ascii=False, default=str)
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:  # tools should never crash the loop
            return f"Error in {name}: {type(e).__name__}: {e}"

    def describe(self) -> str:
        """Human-readable tool list for the system prompt."""
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- {name}: {tool.description}")
            lines.append(f"  args: {tool.parameters}")
        return "\n".join(lines)

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())


# Convenience decorator usable without a registry first (registers on the global one)
_global_registry = ToolRegistry()


def tool(fn=None, *, name=None, description=None):
    return _global_registry.register(fn, name=name, description=description)
```

Note: `tool` registers into a module-global registry; if the user wants multiple
registries, prefer `registry = ToolRegistry(); @registry.register`.

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
    max_steps: int = 10

    @classmethod
    def from_env(cls, dotenv_path: str = ".env") -> "Config":
        load_dotenv(dotenv_path)
        return cls(
            model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("AGENT_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("AGENT_API_KEY", ""),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "10")),
        )
```

## CLI (implement this in `cli.py`)

```python
# src/<package>/cli.py
"""Entry point: <command> "ask something" | --chat | --help"""

import argparse
import sys

from openai import OpenAI

from . import tools as tools_module
from .config import Config
from .loop import ReActLoop, chat_loop

def build_agent(cfg: Config) -> ReActLoop:
    # Import the example tools package so @tool regustrations run.
    from .tools import examples  # noqa: F401
    return ReActLoop(
        client=OpenAI(api_key=cfg.api_key, base_url=cfg.base_url),
        registry=tools_module._global_registry,
        model=cfg.model,
        max_steps=cfg.max_steps,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="<command>", description="ReAct agent harness")
    parser.add_argument("prompt", nargs="*", help="Question to run the agent on.")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode.")
    parser.add_argument("--model", default=None, help="Override model.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override step budget.")
    args = parser.parse_args(argv)

    cfg = Config.from_env()
    if args.model:
        cfg.model = args.model
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if not cfg.api_key:
        print("No AGENT_API_KEY set. Add it to .env or export it.", file=sys.stderr)
        return 2

    agent = build_agent(cfg)

    if args.chat:
        chat_loop(agent)
        return 0

    prompt = " ".join(args.prompt)
    if not prompt:
        parser.print_help()
        return 1

    print(agent.run(prompt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> Fix the `from .tools import examples` line to point at wherever you put the
> example tools (e.g. `..tools.examples`), and remember to import the tools module
> that actually calls `@tool`.

## Example Tools

Give the spec 3-5 small tools. Good generic defaults:

- `current_time()` — returns `datetime.now().isoformat()`
- `calculate(expression)` — safely evaluates arithmetic with `ast` or `operator`
- `search_web(query)` — stub that returns a note it is unimplemented, or calls a free API
- `read_file(path)` / `list_files(path)` — small filesystem helpers (be careful with
  arbitrary paths; restrict to a whitelisted dir if the spec allows)

Each tool is just a plain function plus `@tool`:

```python
# src/<package>/tools/examples.py
import ast
import operator
from datetime import datetime

from ..tools import tool

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
}

@tool
def current_time() -> str:
    """Return the current date and time as an ISO-8601 string."""
    return datetime.now().isoformat()

@tool
def calculate(expression: str) -> float:
    """Evaluate a simple arithmetic expression (no functions, no vars)."""
    node = ast.parse(expression, mode="eval").body
    if not isinstance(node, ast.Expression):
        raise ValueError("not an expression")
    def eval_node(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp):
            op = _BIN_OPS.get(type(n.op))
            if op is None:
                raise ValueError(f"unsupported op: {type(n.op).__name__}")
            return op(eval_node(n.left), eval_node(n.right))
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                return -eval_node(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +eval_node(n.operand)
        raise ValueError(f"unsupported node: {type(n).__name__}")
    return eval_node(node)
```

## pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "<package-kebab>"
version = "0.1.0"
description = "A ReAct agent harness for <spec>"
requires-python = ">=3.10"
dependencies = [
  "openai>=1.0.0",
  "python-dotenv>=1.0.0",
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
# AGENT_MAX_STEPS=10
```

## Verification

1. `python -m pip install -e .` in the project (or `uv sync` / `pip install -e ".[dev]"`).
2. `python -m <package>.cli --help` — confirm help renders, no import errors.
3. `python -m <package>.cli "today's date is when?"` against a real key, or
   `python -m <package>.cli --chat` for the REPL.
4. Run a 1-step smoke test (a tool with no key needed, e.g. `calculate("2+2")`) to
   prove the loop glue works before wiring the LLM.

## Conventions

- Never print the raw API key; keep keys out of source and git.
- Observation errors return as strings; the loop should never crash on a tool error.
- Keep the loop's state in an explicit `messages` list; don't use hidden globals.
- Prefer temp=0 so the loop is deterministic about tool calls.
- `default=str` in `json.dumps` when serializing observations so non-serializable
  results don't break the loop.

## Example Output

User: "I want an agent that answers questions about the current date, time zone, and simple arithmetic — rename the command `time-bot`."

You: scaffold `~/projects/time-bot`, tools `current_time`, `time_in_zone(zone)`,
`calculate(expr)`, a `time-bot` CLI, `.env.example`, and verify the module imports
and prints `--help`.