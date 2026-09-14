---
name: harness-multi-agent
description: "Build a Multi-Agent Orchestrator pattern agent harness from a spec"
---

# Multi-Agent Orchestrator Harness

## What is this pattern?

An **orchestrator** LLM receives a high-level task, decomposes it into subtasks,
picks which **worker role** handles each one, dispatches them (sequentially or in
a thread pool), collects structured deliverables, and synthesizes a final answer.

Each worker is a lightweight agent: it gets a task spec (role system prompt +
user subtask), has access to a filtered subset of tools, and returns a
deliverable with markdown content plus a result JSON. The orchestrator never
executes tools itself — it delegates and aggregates.

```
User task
  → Orchestrator (decompose + route)
     → Worker A (role: research, tools: search, read)
     → Worker B (role: analysis, tools: calculate, format)
     → Worker C (role: writing, tools: summarize)
  → Orchestrator (synthesize final answer)
```

## Workflow

When the user gives a spec (or says "build me a multi-agent orchestrator"):

1. **Ask for a spec if none given.** The user should describe:
   - What domain the system covers
   - What worker roles make sense (e.g. research, analysis, writing, code-gen)
   - What tools each role needs
   - Whether parallel dispatch matters or sequential is fine

   If the spec is vague, propose 3-4 roles with tool assignments and confirm.

2. **Scaffold the project** into the directory the user specifies (default:
   `<folder-name>/`). Use a clean `src/` layout with `pyproject.toml`.

3. **Write roles.py** — role definitions: name, system prompt, allowed tools.

4. **Write worker.py** — a single agent that performs a delegated subtask using
   its own tools and returns a structured deliverable.

5. **Write orchestrator.py** — the decomposition, routing, dispatch (sequential
   or thread pool), collection, and synthesis logic.

6. **Write tools.py** — shared `@tool` registry with role-based filtering.

7. **Add example tools** for the spec, a CLI entry point, and `.env.example`.
   Verify imports and `--help`.

## Project Layout

```
<project>/
├── pyproject.toml
├── .env.example
├── README.md
└── src/<package>/
    ├── __init__.py
    ├── config.py         # env-based config (model, base_url, api_key, max_workers)
    ├── roles.py          # role definitions (name, system_prompt, allowed_tools)
    ├── worker.py         # single-agent subtask executor + deliverable format
    ├── orchestrator.py   # decompose → route → dispatch → collect → synthesize
    ├── tools.py          # @tool decorator, registry, role-filtered tool access
    ├── cli.py            # entry point
    └── tools/
        ├── __init__.py
        └── examples.py   # domain-relevant example tools
```

Use the kebab-case project name for the folder, snake_case for the package.

## Dependencies

Only:

- `openai` (for LLM calls)
- `python-dotenv` (optional, for `.env`)
- stdlib (`json`, `typing`, `inspect`, `concurrent.futures`, `dataclasses`, etc.)

No langchain, no instructor, no heavy frameworks.

## Role Definitions (implement in `roles.py`)

```python
# src/<package>/roles.py
"""Worker role definitions: name, system prompt, and allowed tool names."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Role:
    name: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)


# ── Example roles (adapt to the spec) ──────────────────────────────────

RESEARCHER = Role(
    name="researcher",
    system_prompt=(
        "You are a research specialist. Given a task, use your tools to gather "
        "relevant information. Return your findings as structured JSON with a "
        "'content' field (markdown summary) and a 'result' field (data)."
    ),
    allowed_tools=["search_web", "read_file", "list_files"],
)

ANALYST = Role(
    name="analyst",
    system_prompt=(
        "You are a data analyst. Given research findings, perform calculations, "
        "identify patterns, and produce insights. Return structured JSON with a "
        "'content' field (markdown analysis) and a 'result' field (data)."
    ),
    allowed_tools=["calculate", "format_table", "summarize"],
)

WRITER = Role(
    name="writer",
    system_prompt=(
        "You are a technical writer. Given analysis results, produce a clear, "
        "well-structured markdown document. Return structured JSON with a "
        "'content' field (final markdown) and a 'result' field (metadata)."
    ),
    allowed_tools=["summarize", "format_table"],
)

ALL_ROLES: list[Role] = [RESEARCHER, ANALYST, WRITER]
```

## Worker Agent (implement in `worker.py`)

```python
# src/<package>/worker.py
"""A single worker agent: receives a task spec, uses its tools, returns a deliverable."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .roles import Role
from .tools import ToolRegistry


DELIVERABLE_SCHEMA = """\
You are the "{role}" worker. Your task:

{task}

Respond with a JSON object:
{{
  "content": "<markdown deliverable>",
  "result": {{ ... any structured data you produced ... }}
}}
"""

@dataclass
class Deliverable:
    role: str
    content: str  # markdown
    result: dict[str, Any]  # structured data


def run_worker(
    client: OpenAI,
    model: str,
    role: Role,
    task: str,
    registry: ToolRegistry,
    *,
    max_steps: int = 6,
    verbose: bool = False,
) -> Deliverable:
    """Execute a subtask as the given role, using only its allowed tools."""
    allowed = registry.for_role(role)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": role.system_prompt},
        {"role": "user", "content": DELIVERABLE_SCHEMA.format(role=role.name, task=task)},
    ]

    for step in range(1, max_steps + 1):
        # Build the tools list for this call
        tools = [t.to_openai_schema() for t in allowed.tools]
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools if tools else None,
            temperature=0,
        )
        msg = response.choices[0].message

        # If the model called a tool, execute it and loop
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                if verbose:
                    print(f"  [{role.name}] tool: {fn_name}({fn_args})")
                observation = allowed.execute(fn_name, fn_args)
                messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})
            continue

        # Otherwise parse the final deliverable JSON
        content = msg.content or ""
        try:
            parsed = _extract_json(content)
            return Deliverable(
                role=role.name,
                content=parsed.get("content", content),
                result=parsed.get("result", {}),
            )
        except (json.JSONDecodeError, KeyError):
            # If JSON parse fails, treat the whole response as content
            return Deliverable(role=role.name, content=content, result={})

    return Deliverable(role=role.name, content=f"[{role.name}] step budget exceeded", result={})


def _extract_json(text: str) -> dict:
    """Parse JSON from model output, tolerating markdown fences."""
    text = text.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text[len(fence):]
        if text.endswith(fence):
            text = text[:-len(fence)]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
```

## Orchestrator (implement in `orchestrator.py`)

```python
# src/<package>/orchestrator.py
"""Orchestrator: decompose → route → dispatch → collect → synthesize."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .roles import Role
from .tools import ToolRegistry
from .worker import Deliverable, run_worker


DECOMPOSE_PROMPT = """\
You are an orchestrator. Given the user's task, decompose it into subtasks and
assign each to a worker role.

Available roles and their tools:
{roles}

Respond with a JSON array of subtasks:
[
  {{
    "role": "<role name>",
    "task": "<specific subtask description>"
  }}
]

If the task is simple and needs only one role, return a single-element array.
If no tools are needed, pick the role whose system prompt best fits.
"""

SYNTHESIZE_PROMPT = """\
You are an orchestrator synthesizing worker results into a final answer.

Original task: {task}

Worker deliverables:
{deliverables}

Produce a single coherent response that combines all worker outputs.
"""


@dataclass
class Subtask:
    role: str
    task: str


class Orchestrator:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        roles: list[Role],
        registry: ToolRegistry,
        *,
        max_workers: int = 4,
        worker_max_steps: int = 6,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.roles = {r.name: r for r in roles}
        self.registry = registry
        self.max_workers = max_workers
        self.worker_max_steps = worker_max_steps
        self.verbose = verbose

    def run(self, user_input: str) -> str:
        """Full pipeline: decompose → dispatch → synthesize."""
        # 1) Decompose
        subtasks = self._decompose(user_input)
        if self.verbose:
            print(f"[orchestrator] decomposed into {len(subtasks)} subtasks")

        # 2) Dispatch (parallel or sequential)
        deliverables = self._dispatch(subtasks)

        # 3) Synthesize
        return self._synthesize(user_input, deliverables)

    def _decompose(self, task: str) -> list[Subtask]:
        """Ask the LLM to split the task into role-assignable subtasks."""
        roles_desc = "\n".join(
            f"- {r.name}: {r.system_prompt[:80]}... | tools: {r.allowed_tools}"
            for r in self.roles.values()
        )
        messages = [
            {"role": "system", "content": "You decompose tasks for a multi-agent system."},
            {"role": "user", "content": DECOMPOSE_PROMPT.format(roles=roles_desc) + f"\n\nTask: {task}"},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0,
        )
        content = response.choices[0].message.content or "[]"
        items = self._parse_json(content)
        return [Subtask(role=s["role"], task=s["task"]) for s in items]

    def _dispatch(self, subtasks: list[Subtask]) -> list[Deliverable]:
        """Run workers — parallel via ThreadPoolExecutor, or sequential."""
        def _run(st: Subtask) -> Deliverable:
            role = self.roles.get(st.role)
            if role is None:
                # fallback: pick the first role
                role = next(iter(self.roles.values()))
            if self.verbose:
                print(f"[dispatch] {role.name} ← {st.task[:60]}...")
            return run_worker(
                self.client, self.model, role, st.task, self.registry,
                max_steps=self.worker_max_steps, verbose=self.verbose,
            )

        if len(subtasks) <= 1:
            return [_run(st) for st in subtasks]

        results: list[Deliverable] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_run, st): st for st in subtasks}
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def _synthesize(self, task: str, deliverables: list[Deliverable]) -> str:
        """Combine all worker outputs into a final answer."""
        parts = []
        for d in deliverables:
            parts.append(f"### [{d.role}]\n{d.content}\n")
        deliverables_text = "\n".join(parts)

        messages = [
            {"role": "system", "content": "You synthesize multi-agent outputs into a final answer."},
            {"role": "user", "content": SYNTHESIZE_PROMPT.format(
                task=task, deliverables=deliverables_text
            )},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def _parse_json(self, text: str) -> Any:
        text = text.strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            text = text[len(fence):]
            if text.endswith(fence):
                text = text[:-len(fence)]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
```

## Tool Registry with Role Filtering (implement in `tools.py`)

```python
# src/<package>/tools.py
"""@tool registry with role-based tool filtering."""

from __future__ import annotations

import inspect
import json
import typing


def type_to_schema(t: type) -> dict:
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
        return {"type": "string"}
    if origin is typing.Union:
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
        return {"type": "string", "enum": [str(a) for a in typing.get_args(t)]}
    return {"type": "string"}


class Tool:
    def __init__(self, fn, name: str | None = None, description: str | None = None):
        self.fn = fn
        self.name = name or fn.__name__
        sig = inspect.signature(fn)
        self.description = (description if description else (fn.__doc__ or f"Call {self.name}")).strip()
        self.parameters = {}
        for pname, param in sig.parameters.items():
            if pname == "return":
                continue
            ann = param.annotation if param.annotation is not inspect.Parameter.empty else str
            pschema = type_to_schema(ann)
            if param.default is not inspect.Parameter.empty:
                pschema["default"] = param.default
                pschema["required"] = False
            else:
                pschema["required"] = True
            self.parameters[pname] = pschema

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {p: {k: v for k, v in s.items() if k != "required"} for p, s in self.parameters.items()},
                    "required": [p for p, s in self.parameters.items() if s.get("required")],
                },
            },
        }

    def __call__(self, **kwargs: typing.Any) -> typing.Any:
        return self.fn(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, fn=None, *, name: str | None = None, description: str | None = None):
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
        except Exception as e:
            return f"Error in {name}: {type(e).__name__}: {e}"

    def for_role(self, role) -> ToolRegistry:
        """Return a new registry containing only tools the role is allowed to use."""
        filtered = ToolRegistry()
        allowed = set(role.allowed_tools) if role.allowed_tools else set(self._tools.keys())
        for name, tool in self._tools.items():
            if name in allowed:
                filtered._tools[name] = tool
        return filtered

    def describe(self) -> str:
        return "\n".join(f"- {n}: {t.description}" for n, t in self._tools.items())

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())


_global_registry = ToolRegistry()


def tool(fn=None, *, name=None, description=None):
    return _global_registry.register(fn, name=name, description=description)
```

## Example Tools (implement in `tools/examples.py`)

```python
# src/<package>/tools/examples.py
"""Example tools — adapt to the spec."""

import ast
import json
import operator
from datetime import datetime
from pathlib import Path

from ..tools import tool

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
}


@tool
def calculate(expression: str) -> float:
    """Evaluate a safe arithmetic expression."""
    node = ast.parse(expression, mode="eval").body
    def _eval(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp):
            return _BIN_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError(f"unsupported: {type(n).__name__}")
    return _eval(node)


@tool
def current_time() -> str:
    """Return the current ISO-8601 timestamp."""
    return datetime.now().isoformat()


@tool
def search_web(query: str) -> str:
    """Stub web search — returns a placeholder."""
    return json.dumps({"query": query, "results": [], "note": "search_web is a stub — wire up an API"})


@tool
def read_file(path: str) -> str:
    """Read a file and return its contents."""
    return Path(path).read_text()


@tool
def list_files(directory: str = ".") -> list[str]:
    """List files in a directory."""
    return [str(p) for p in Path(directory).iterdir()]


@tool
def summarize(text: str) -> str:
    """Return a basic summary (first 500 chars). Truncation only — for real use, wire an LLM call."""
    return text[:500]


@tool
def format_table(rows: list[dict]) -> str:
    """Format a list of dicts as a markdown table."""
    if not rows:
        return "(empty table)"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)
```

## Config (implement in `config.py`)

```python
# src/<package>/config.py
"""Config from env vars / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Config:
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 6
    max_workers: int = 4

    @classmethod
    def from_env(cls, dotenv_path: str = ".env") -> "Config":
        load_dotenv(dotenv_path)
        return cls(
            model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("AGENT_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("AGENT_API_KEY", ""),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")),
            max_workers=int(os.getenv("AGENT_MAX_WORKERS", "4")),
        )
```

## CLI (implement in `cli.py`)

```python
# src/<package>/cli.py
"""Entry point: <command> "task" | --chat | --help"""

import argparse
import sys

from openai import OpenAI

from . import tools as tools_module
from .config import Config
from .orchestrator import Orchestrator
from .roles import ALL_ROLES


def build_orchestrator(cfg: Config) -> Orchestrator:
    from .tools import examples  # noqa: F401 — register example tools
    return Orchestrator(
        client=OpenAI(api_key=cfg.api_key, base_url=cfg.base_url),
        model=cfg.model,
        roles=ALL_ROLES,
        registry=tools_module._global_registry,
        max_workers=cfg.max_workers,
        worker_max_steps=cfg.max_steps,
        verbose=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="<command>", description="Multi-agent orchestrator")
    parser.add_argument("prompt", nargs="*", help="High-level task for the agent team.")
    parser.add_argument("--chat", action="store_true", help="Interactive mode.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = Config.from_env()
    if args.model:
        cfg.model = args.model
    if args.max_workers:
        cfg.max_workers = args.max_workers
    if not cfg.api_key:
        print("No AGENT_API_KEY set. Add it to .env or export it.", file=sys.stderr)
        return 2

    orch = build_orchestrator(cfg)

    if args.chat:
        print("Multi-agent orchestrator ready. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if user_input in {"exit", "quit", "q"}:
                return 0
            if not user_input:
                continue
            print(f"\n{orch.run(user_input)}")
        return 0

    prompt = " ".join(args.prompt)
    if not prompt:
        parser.print_help()
        return 1

    print(orch.run(prompt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "<package-kebab>"
version = "0.1.0"
description = "A multi-agent orchestrator harness for <spec>"
requires-python = ">=3.10"
dependencies = [
  "openai>=1.0.0",
  "python-dotenv>=1.0.0",
]

[project.scripts]
<command> = "<package>.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/<package>"]
```

## .env.example

```
AGENT_API_KEY=sk-...
# AGENT_BASE_URL=https://api.openai.com/v1
# AGENT_MODEL=gpt-4o-mini
# AGENT_MAX_STEPS=6
# AGENT_MAX_WORKERS=4
```

## Verification

1. `pip install -e .` in the project directory.
2. `python -m <package>.cli --help` — confirm no ImportError, help renders.
3. `python -m <package>.cli "calculate 2+2 and tell me the time"` — verify
   orchestrator decomposes, routes to analyst + (any role with current_time),
   workers execute, and a synthesized answer appears.
4. `python -m <package>.cli --chat` for the interactive REPL.
5. Run a single-tool smoke test (e.g. `calculate("2+2")`) to confirm tool
   glue works before wiring the LLM.

## Conventions

- Never print or log API keys.
- Workers return `Deliverable` dataclass (content markdown + result dict); if
  JSON parse fails, the raw text becomes content.
- `json.dumps(..., default=str)` when serializing tool results.
- Orchestrator uses `concurrent.futures.ThreadPoolExecutor` for parallel dispatch;
  falls back to sequential for a single subtask.
- Prefer temp=0 for deterministic routing and tool calls.
- Keep messages lists explicit — no hidden globals.

## Customization Guide

To adapt to a new spec:

1. **Edit `roles.py`**: define new `Role` entries with domain-appropriate system
   prompts and tool allowlists.
2. **Edit `tools/examples.py`**: add or remove tools; the registry auto-discovers
   anything decorated with `@tool`.
3. **Tune `DECOMPOSE_PROMPT`** in orchestrator.py if you want more control over
   how the LLM splits tasks (e.g. enforce exactly N subtasks, or require
   dependencies between them).
4. **Adjust `max_workers`** in env or CLI for parallelism.
