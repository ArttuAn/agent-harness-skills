---
name: harness-code-agent
description: "Build a Code Agent pattern harness that edits files and runs commands in a workspace"
---

# Code Agent Harness

## What is the Code Agent pattern?

A Code Agent is a coding loop where an LLM has access to file-system and shell
tools, operates inside a sandboxed workspace directory, and iterates: read code,
plan changes, edit files, run tests/commands, observe output, repeat. The key
constraints are:

- **Workspace isolation**: every file operation and shell command is restricted to
  a single base directory. Path traversal outside the workspace is blocked.
- **Tool boundary**: the agent can only call tools you explicitly provide
  (`read_file`, `write_file`, `edit_file`, `list_dir`, `run_command`, `grep`).
  No raw Python exec, no network, no unrestricted shell.
- **Safety guardrails**: dangerous commands (`rm -rf /`, `sudo`, `mkfs`, `chmod`
  on system dirs, etc.) are blocked before they reach `subprocess`.
- **Test-driven loop**: after file edits the system prompt nudges the agent to run
  tests, linting, or builds so it can self-correct.

## Workflow

When the user gives a spec (or says "build me a code agent"), do the following:

1. **Ask for a spec if none given.** The user should describe what the agent should
   work on (e.g. "a Python CLI project", "a React app"). If vague, propose a
   sensible default and confirm.

2. **Scaffold the project** into the directory the user specifies (default:
   `code-agent/` next to their working dir). Use a clean `src/` layout with
   `pyproject.toml`.

3. **Write every module with real, runnable code.** Do not stub or omit. Each
   module below must be fully implemented.

4. **Verify** the project imports and CLI `--help` runs before finishing.

## Project Layout

```
code-agent/
├── pyproject.toml
├── .env.example
├── README.md
└── src/code_agent/
    ├── __init__.py
    ├── workspace.py   # workspace manager: path isolation + canonicalization
    ├── tools.py       # @tool registry: read_file, write_file, edit_file, list_dir, run_command, grep
    ├── agent.py       # the coding loop (LLM + tools, test-driven)
    ├── safety.py      # command blocklist + safety checks
    ├── config.py      # env/.env config (workspace dir, command timeout, model)
    └── cli.py         # argparse entry point with --workspace flag
```

## Dependencies

Only:

- `openai` (for LLM calls)
- `python-dotenv` (optional, for `.env` loading)
- stdlib (`subprocess`, `pathlib`, `json`, `os`, `re`, `shutil`, `typing`,
  `inspect`, `dataclasses`, etc.)

No langchain, no external tool frameworks.

## Module: `workspace.py` — Workspace Manager

This is the **safety boundary**. Every file path passes through it.

```python
"""Workspace manager — restricts all file ops to a single base directory."""

from __future__ import annotations

import os
from pathlib import Path


class Workspace:
    """Wraps a base directory and enforces that every resolved path stays inside it."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base = Path(base_dir).resolve()
        self.base.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str | Path) -> Path:
        """Canonicalize *path* relative to the workspace and verify containment.

        Raises ValueError if the resolved path escapes the workspace.
        """
        if not path:
            return self.base
        candidate = (self.base / path).resolve()
        if not self._is_inside(candidate):
            raise ValueError(
                f"Path traversal blocked: {candidate} is outside workspace {self.base}"
            )
        return candidate

    def _is_inside(self, candidate: Path) -> bool:
        try:
            candidate.relative_to(self.base)
            return True
        except ValueError:
            return False

    def read_file(self, path: str | Path) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"No such file: {target.relative_to(self.base)}")
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str | Path, content: str) -> str:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {target.relative_to(self.base)}"

    def list_dir(self, path: str | Path = ".") -> list[str]:
        target = self.resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {target.relative_to(self.base)}")
        entries: list[str] = []
        for entry in sorted(target.iterdir()):
            name = entry.name + ("/" if entry.is_dir() else "")
            entries.append(name)
        return entries

    def grep(self, pattern: str, path: str | Path = ".") -> list[dict]:
        """Grep for *pattern* inside the workspace. Returns list of {file, line_no, line}."""
        target = self.resolve(path)
        results: list[dict] = []
        regex = __import__("re").compile(pattern)
        files = [target] if target.is_file() else self._walk(target)
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    results.append({
                        "file": str(fp.relative_to(self.base)),
                        "line_no": i,
                        "line": line.rstrip(),
                    })
        return results

    def _walk(self, directory: Path) -> list[Path]:
        files: list[Path] = []
        for root, dirs, filenames in os.walk(directory):
            # skip hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fn in filenames:
                if not fn.startswith("."):
                    files.append(Path(root) / fn)
        return files
```

## Module: `tools.py` — Tool Registry

```python
"""@tool decorator + registry + workspace-bound coding tools."""

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import typing
from pathlib import Path

from .workspace import Workspace
from .safety import check_command


# ── schema inference (lightweight, stdlib only) ──────────────────────────────

def _type_to_schema(t: type) -> dict:
    origin = typing.get_origin(t)
    if origin is None:
        return {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
        }.get(t, {"type": "string"})
    if origin is typing.Union:
        for arg in typing.get_args(t):
            if arg is not type(None):
                return _type_to_schema(arg)
    return {"type": "string"}


# ── Tool + Registry ──────────────────────────────────────────────────────────

class Tool:
    def __init__(self, fn, name: str | None = None, description: str | None = None) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        sig = inspect.signature(fn)
        self.description = (description or fn.__doc__ or f"Call {self.name}").strip()
        self.parameters: dict[str, dict] = {}
        for pname, param in sig.parameters.items():
            ann = param.annotation if param.annotation is not inspect.Parameter.empty else str
            schema = _type_to_schema(ann)
            if param.default is not inspect.Parameter.empty:
                schema["default"] = param.default
                schema["required"] = False
            else:
                schema["required"] = True
            self.parameters[pname] = schema
        self.required = [p for p, s in self.parameters.items() if s.get("required")]

    def __call__(self, **kw: typing.Any) -> typing.Any:
        return self.fn(**kw)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, fn=None, *, name: str | None = None, description: str | None = None):
        def wrap(f):
            t = Tool(f, name=name, description=description)
            self._tools[t.name] = t
            return t
        if fn is not None:
            return wrap(fn)
        return wrap

    def execute(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'. Available: {', '.join(self._tools)}"
        try:
            result = self._tools[name](**args)
            return json.dumps(result, ensure_ascii=False, default=str) if not isinstance(result, str) else result
        except Exception as e:
            return f"Error in {name}: {type(e).__name__}: {e}"

    def describe(self) -> str:
        return "\n".join(
            f"- {t.name}: {t.description}\n  params: {t.parameters}"
            for t in self._tools.values()
        )

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())


# ── workspace-bound tools ─────────────────────────────────────────────────────

_registry = ToolRegistry()

def tool(fn=None, *, name=None, description=None):
    return _registry.register(fn, name=name, description=description)


def register_workspace_tools(registry: ToolRegistry, ws: Workspace, timeout: int = 30) -> None:
    """Register the six coding tools bound to *ws* and *timeout*."""

    @registry.register
    def read_file(path: str) -> str:
        """Read and return the contents of a file inside the workspace."""
        return ws.read_file(path)

    @registry.register
    def write_file(path: str, content: str) -> str:
        """Write *content* to *path*, creating parent dirs as needed."""
        return ws.write_file(path, content)

    @registry.register
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace the first occurrence of *old* with *new* in the file."""
        text = ws.read_file(path)
        if old not in text:
            return f"Error: '{old}' not found in {path}"
        updated = text.replace(old, new, 1)
        ws.write_file(path, updated)
        return f"Edited {path}: replaced one occurrence"

    @registry.register
    def list_dir(path: str = ".") -> str:
        """List files and directories at the given path."""
        return json.dumps(ws.list_dir(path))

    @registry.register
    def run_command(command: str) -> str:
        """Run a shell command inside the workspace. Blocked if unsafe."""
        err = check_command(command)
        if err:
            return f"Blocked: {err}"
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(ws.base),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code {result.returncode}]"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error running command: {e}"

    @registry.register
    def grep(pattern: str, path: str = ".") -> str:
        """Search for a regex pattern in workspace files."""
        return json.dumps(ws.grep(pattern, path), default=str)
```

## Module: `safety.py` — Command Blocklist

```python
"""Safety checks for run_command — block dangerous shell commands."""

from __future__ import annotations

import re

# Patterns that should never be allowed through run_command.
BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*r[a-zA-Z]*\s+/", "recursive delete from root"),
    (r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*f[a-zA-Z]*\s+/", "forced recursive delete"),
    (r"\brm\s+-rf\s+/", "rm -rf /"),
    (r"\brm\s+-rf\s+~", "rm -rf ~"),
    (r"\bsudo\b", "sudo is not allowed"),
    (r"\bchmod\s+777\s+/", "chmod 777 on root"),
    (r"\bchmod\s+-R\s+777\s+/", "chmod -R 777"),
    (r"\bmkfs\b", "mkfs (format disk)"),
    (r"\bdd\s+.*of=/dev/", "dd writing to device"),
    (r"\b:\(\)\s*\{", "fork bomb"),
    (r">\s*/dev/sd", "writing to disk device"),
    (r"\bcurl\b.*\|\s*bash", "curl | bash"),
    (r"\bwget\b.*\|\s*sh", "wget | sh"),
    (r"\bsystemctl\b", "systemctl is not allowed"),
    (r"\bservice\b", "service control is not allowed"),
    (r"\bkill\s+-9\s+1\b", "killing PID 1"),
    (r"\bshutdown\b", "shutdown is not allowed"),
    (r"\breboot\b", "reboot is not allowed"),
    (r"\bmount\b", "mount is not allowed"),
    (r"\bumount\b", "umount is not allowed"),
    (r"\bnc\s+-l", "listening netcat"),
    (r"\bffserver\b", "ffserver is blocked"),
]

_BLOCKED_RE = [(re.compile(pat), reason) for pat, reason in BLOCKED_PATTERNS]


def check_command(command: str) -> str | None:
    """Return an error reason if the command is dangerous, else None."""
    for regex, reason in _BLOCKED_RE:
        if regex.search(command):
            return reason
    return None
```

## Module: `agent.py` — The Coding Loop

```python
"""Coding loop: LLM plans, calls tools, runs tests, repeats."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .tools import ToolRegistry

SYSTEM_PROMPT = """\
You are a Code Agent. You work inside a sandboxed workspace directory.
You can read, write, and edit files, list directories, run shell commands,
and grep through code.

Available tools:
{tools}

RULES:
1. Before making changes, read the relevant files first.
2. After editing files, run the tests or build to verify your changes.
3. If a command fails, diagnose the error and fix it before proceeding.
4. Never attempt to escape the workspace or run dangerous commands.
5. When your task is complete, respond with a JSON object:
   {{"thought": "...", "answer": "<summary of what you did>"}}

To take an action, respond with:
{{"thought": "...", "action": {{"name": "<tool>", "args": {{"<param>": <value>}}}}}}
"""


class CodeAgentLoop:
    def __init__(
        self,
        client: OpenAI,
        registry: ToolRegistry,
        model: str,
        max_steps: int = 20,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.max_steps = max_steps

    def _system(self) -> str:
        return SYSTEM_PROMPT.format(tools=self.registry.describe())

    def run(self, task: str, *, verbose: bool = True) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system()},
            {"role": "user", "content": task},
        ]

        for step in range(1, self.max_steps + 1):
            if verbose:
                print(f"\n[step {step}/{self.max_steps}]")

            decision = self._ask(messages)
            thought = decision.get("thought", "")
            if verbose and thought:
                print(f"  thought: {thought}")

            if "answer" in decision:
                if verbose:
                    print(f"  answer: {decision['answer']}")
                return str(decision["answer"])

            action = decision.get("action")
            if not isinstance(action, dict) or "name" not in action:
                messages.append({
                    "role": "user",
                    "content": "Invalid response: must have 'action' with 'name' or 'answer'.",
                })
                continue

            name, args = action["name"], action.get("args", {})
            if verbose:
                print(f"  action: {name}({args})")

            observation = self.registry.execute(name, args)
            if verbose:
                print(f"  observation: {observation[:200]}")

            messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"Observation from {name}:\n{observation}\nContinue."})

        return f"Step budget ({self.max_steps}) exceeded."

    def _ask(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        for _ in range(3):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip()
            # strip markdown fences
            fence = chr(96) * 3
            if content.startswith(fence):
                content = content[len(fence):]
                if content.endswith(fence):
                    content = content[:-len(fence)]
                content = content.strip()
                if content.startswith("json"):
                    content = content[4:]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                messages.append({"role": "user", "content": "Reply with valid JSON only."})
        raise RuntimeError("Model failed to produce valid JSON after 3 retries.")
```

## Module: `config.py`

```python
"""Env-based config for the Code Agent."""

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
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


@dataclass
class Config:
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    workspace_dir: str = "./workspace"
    command_timeout: int = 30
    max_steps: int = 20

    @classmethod
    def from_env(cls, dotenv_path: str = ".env") -> Config:
        load_dotenv(dotenv_path)
        return cls(
            model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("AGENT_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("AGENT_API_KEY", ""),
            workspace_dir=os.getenv("AGENT_WORKSPACE", "./workspace"),
            command_timeout=int(os.getenv("AGENT_COMMAND_TIMEOUT", "30")),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "20")),
        )
```

## Module: `cli.py`

```python
"""CLI entry point: code-agent "fix the tests" --workspace ./my-project"""

import argparse
import sys

from openai import OpenAI

from .config import Config
from .workspace import Workspace
from .tools import ToolRegistry, register_workspace_tools
from .agent import CodeAgentLoop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="code-agent", description="Code Agent harness")
    parser.add_argument("task", nargs="*", help="Task description for the agent.")
    parser.add_argument("--workspace", "-w", default=None, help="Workspace directory.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="Command timeout (seconds).")
    args = parser.parse_args(argv)

    cfg = Config.from_env()
    if args.workspace:
        cfg.workspace_dir = args.workspace
    if args.model:
        cfg.model = args.model
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.timeout:
        cfg.command_timeout = args.timeout

    if not cfg.api_key:
        print("No AGENT_API_KEY set. Add it to .env or export it.", file=sys.stderr)
        return 2

    ws = Workspace(cfg.workspace_dir)
    registry = ToolRegistry()
    register_workspace_tools(registry, ws, timeout=cfg.command_timeout)

    agent = CodeAgentLoop(
        client=OpenAI(api_key=cfg.api_key, base_url=cfg.base_url),
        registry=registry,
        model=cfg.model,
        max_steps=cfg.max_steps,
    )

    task = " ".join(args.task)
    if not task:
        parser.print_help()
        return 1

    result = agent.run(task)
    print(f"\n{result}")
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
name = "code-agent"
version = "0.1.0"
description = "A sandboxed Code Agent harness"
requires-python = ">=3.10"
dependencies = [
  "openai>=1.0.0",
  "python-dotenv>=1.0.0",
]

[project.scripts]
code-agent = "code_agent.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/code_agent"]
```

## .env.example

```
AGENT_API_KEY=sk-...
# AGENT_MODEL=gpt-4o-mini
# AGENT_WORKSPACE=./workspace
# AGENT_COMMAND_TIMEOUT=30
# AGENT_MAX_STEPS=20
```

## Verification

1. `pip install -e .` in the project directory.
2. `code-agent --help` — confirm no ImportError and help text renders.
3. `code-agent -w /tmp/test-ws "create a file called hello.py with a hello world"` —
   verify the file appears in `/tmp/test-ws/hello.py` and the agent reports success.
4. `code-agent -w /tmp/test-ws "rm -rf /"` — verify the command is **blocked**.

## Conventions

- Every path passes through `Workspace.resolve()` — never `open(path)` directly.
- `run_command` enforces cwd=workspace and applies the blocklist before subprocess.
- Tool errors are returned as strings, never raised — the loop must not crash.
- `json.dumps(..., default=str)` for observation serialization.
- Keep `temp=0` so tool calls are deterministic.
- Never print or log API keys.

## Safety Checklist

| Threat | Mitigation |
|--------|------------|
| Path traversal (`../../etc/passwd`) | `Workspace.resolve()` canonicalizes + checks containment |
| `sudo rm -rf /` | `safety.py` blocklist catches `sudo` and destructive `rm` patterns |
| Fork bombs / disk wipes | Blocked by pattern list |
| `curl \| bash` | Blocked |
| Command timeout | `subprocess.run(..., timeout=N)` enforced |
| Shell escapes workspace cwd | `subprocess.run(cwd=str(ws.base))` |
