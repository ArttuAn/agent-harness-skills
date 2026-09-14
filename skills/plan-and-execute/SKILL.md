---
name: harness-plan-execute
description: "Build a Plan-and-Execute pattern agent harness from a spec"
---

# Plan-and-Execute Agent Harness

## What is Plan-and-Execute?

Plan-and-Execute is an agent pattern that separates **thinking** from **acting**.
Instead of deciding each action one token at a time, the agent first produces a
full plan and then executes it with explicit checkpoints:

```
  ┌──────────┐  ordered steps   ┌──────────┐  tool call   ┌──────────┐
  │ Planner  │ ────────────────►│ Executor │ ────────────►│  Tools   │
  └──────────┘                  └────┬─────┘              └──────────┘
       ▲                             │ output
       │ replan on                   ▼
       │ failure            ┌─────────────────┐
       └────────────────────│  Verifier /     │  pass ──► next step
                            │  Checkpoint     │  fail ──► replan
                            └─────────────────┘
```

1. **Planner** — an LLM turns the task into an ordered step list. Each step has
   a goal and verification criteria (how an outsider confirms it worked).
2. **Executor** — runs a single step by calling a tool through a registry and
   captures the result.
3. **Verifier / checkpoint** — checks each step's output against its criteria
   before the harness proceeds to the next step.
4. **Replan** — if a step fails (or the plan goes stale), regenerate only the
   remaining steps. Succeeded steps are kept; work picks up from the failure.

This is the harness behind agents like BabyAGI and JARVIS-style task launchers.

## Workflow

When the user gives you a spec (or says "build me a plan-and-execute agent"):

1. **Ask for a spec if none given.** What should the agent accomplish? Which
   example domain tools make sense (e.g. a todo store, a calculator, file
   helpers)? If the spec is vague, propose 3 example tools and confirm.
2. **Scaffold the project** into the directory the user specifies (default: a
   folder named after the project, next to where they're working).
3. **Write the modules** below. They are complete, runnable reference
   implementations — put each in its own file, adapting prose/naming to the
   spec but keeping them syntactically valid and self-consistent.
4. **Wire the loop** in `agent.py` and the **CLI** in `main.py`.
5. **Verify** per the checklist at the bottom before reporting done.

## Project Layout

Create this structure in the target directory:

```
<project>/
├── requirements.txt   # openai, python-dotenv (nothing else)
├── .env.example
├── README.md          # short: what it is, how to run
├── plan.py            # Step + Plan data model (pending/running/succeeded/failed)
├── config.py          # env/.env loading, OpenAI client factory, JSON helpers
├── planner.py         # LLM planner → ordered steps (goal + verification criteria)
├── executor.py        # ToolRegistry + Executor: runs a single step, captures result
├── verify.py          # Verifier: heuristic + LLM, checks output vs criteria
├── tools.py           # example domain tools + make_registry()
├── agent.py           # main loop: plan → execute → verify → (replan until done)
└── main.py            # argparse CLI entry point
```

Use the snake_case project name the user picked for the folder (e.g. folder
`todo-analyzer` → files as shown; no `__init__.py` needed — it is a flat
script-style project run straight from its directory).

## Dependencies

Use only:

- `openai` (LLM calls)
- `python-dotenv` (optional `.env` loading)

Do **not** pull in langchain, instructor, or any other heavy framework. The
planner and verifier coordinate with the model over plain JSON, which is parsed
with the stdlib.

## The Data Model (implement this in `plan.py`)

```python
# <project>/plan.py
"""Data model for plans and steps.

Step statuses follow a small state machine that the executor and verifier
drive: pending -> running -> succeeded | failed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Step:
    """A single unit of work in a plan."""

    id: int
    goal: str
    verification_criteria: list[str] = field(default_factory=list)
    tool: str | None = None
    status: StepStatus = StepStatus.PENDING
    output: Any = None
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "verification_criteria": self.verification_criteria,
            "tool": self.tool,
            "status": self.status.value,
            "output": self.output,
            "result": self.result,
            "error": self.error,
        }


class Plan:
    """An ordered list of steps derived from an objective."""

    def __init__(self, objective: str, steps: list[Step]) -> None:
        self.objective = objective
        self.steps = steps

    def next_pending(self) -> Step | None:
        for step in self.steps:
            if step.status is StepStatus.PENDING:
                return step
        return None

    def all_done(self) -> bool:
        return bool(self.steps) and all(
            step.status is StepStatus.SUCCEEDED for step in self.steps
        )

    def has_failed(self) -> bool:
        return any(step.status is StepStatus.FAILED for step in self.steps)

    def summary_json(self) -> str:
        return json.dumps(
            {"objective": self.objective, "steps": [s.to_dict() for s in self.steps]},
            indent=2,
            ensure_ascii=False,
        )
```

## Config (implement this in `config.py`)

```python
# <project>/config.py
"""Environment-based config, OpenAI client factory, and LLM JSON helpers."""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # .env loading is optional; plain env vars work too.


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_openai_client():
    """Create an OpenAI client from OPENAI_API_KEY / OPENAI_BASE_URL.

    Returns None when no credentials are configured, so callers can fall back
    to offline paths (local planner, heuristic verifier).
    """
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
        return None
    kwargs: dict[str, str] = {}
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


FENCE = "`" * 3  # markdown code-fence delimiter


def extract_json(content: str | None) -> dict[str, Any]:
    """Pull a JSON object out of an LLM reply (tolerates fences and prose)."""
    if content is None:
        return {}
    text = content.strip()
    if text.startswith(FENCE):
        lines = text.splitlines()
        if lines and lines[0].startswith(FENCE):
            lines = lines[1:]
        if lines and lines[-1].strip() == FENCE:
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{content}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc
```

## The Planner (implement this in `planner.py`)

The planner LLM turns the objective into an ordered step list where every step
carries a goal and verification criteria. A deterministic local fallback keeps
the project runnable with no API key.

```python
# <project>/planner.py
"""Planner: an LLM (or a local fallback) turns a task into an ordered step list."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from config import extract_json, load_openai_client
from plan import Plan, Step, StepStatus

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the planner of a Plan-and-Execute agent.

Given an objective, break it into a small, ordered list of concrete steps.
Each step MUST have a goal and verification criteria that state how an outside
observer can confirm the step succeeded. Use only tools from `available_tools`
(choose the closest one; `execute_step` is the generic default).

Reply with JSON only, of this exact shape:
{"steps": [{"goal": "...", "verification_criteria": ["...", "..."], "tool": "execute_step"}]}

Rules:
- Steps must be independent enough that failing one means only that step is redone.
- If `history` and `failed_step` are given, only produce steps that still need
  work: keep the steps that already succeeded and re-plan from the failed step onward.
"""

DEFAULT_LOCAL_STEPS = [
    (
        "Break the objective down into concrete sub-actions.",
        ["broke", "down", "sub-actions"],
        None,
    ),
    (
        "Perform every planned action for the objective.",
        ["performed", "planned", "action"],
        None,
    ),
    (
        "Verify completeness and summarize the outcome.",
        ["completeness", "summarized", "outcome"],
        None,
    ),
]


def build_plan(
    objective: str,
    *,
    tools: list[str] | None = None,
    history: list[dict[str, Any]] | None = None,
    failed_step: dict[str, Any] | None = None,
    client=None,
    model: str | None = None,
) -> Plan:
    if client is None and not os.getenv("OPENAI_API_KEY"):
        logger.info("No OPENAI_API_KEY: using deterministic local fallback planner.")
        return _local_plan(objective, history=history)

    client = client or load_openai_client()
    payload = json.dumps(
        {
            "objective": objective,
            "available_tools": tools or ["execute_step"],
            "history": history or [],
            "failed_step": failed_step,
        },
        indent=2,
        ensure_ascii=False,
    )
    response = client.chat.completions.create(
        model=model or os.getenv("PLANNER_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
    )
    data = extract_json(response.choices[0].message.content)
    steps = []
    for i, raw in enumerate(data.get("steps", []), start=1):
        steps.append(
            Step(
                id=i,
                goal=str(raw.get("goal", "")),
                verification_criteria=[
                    str(c) for c in raw.get("verification_criteria") or []
                ],
                tool=str(raw["tool"]) if raw.get("tool") else None,
            )
        )
    return Plan(objective=objective, steps=steps)


def _local_plan(
    objective: str, history: list[dict[str, Any]] | None = None
) -> Plan:
    succeeded_goals = {
        h.get("goal") for h in (history or []) if h.get("status") == "succeeded"
    }
    steps: list[Step] = []
    for i, (goal, criteria, tool) in enumerate(DEFAULT_LOCAL_STEPS, start=1):
        full_goal = f"{goal} Objective: {objective}"
        if full_goal in succeeded_goals:
            continue
        steps.append(
            Step(
                id=i,
                goal=full_goal,
                verification_criteria=list(criteria),
                tool=tool,
            )
        )
    if not steps:
        steps.append(
            Step(
                id=1,
                goal=f"Finish remaining work for: {objective}",
                verification_criteria=["performed", "summarized"],
                tool=None,
            )
        )
    return Plan(objective=objective, steps=steps)


def replan(
    plan: Plan,
    failed_step: Step,
    *,
    tools: list[str] | None = None,
    client=None,
    model: str | None = None,
) -> Plan:
    """Regenerate the unfinished part of a plan after one step fails."""
    new_plan = build_plan(
        plan.objective,
        tools=tools,
        history=[s.to_dict() for s in plan.steps],
        failed_step=failed_step.to_dict(),
        client=client,
        model=model,
    )
    kept = [s for s in plan.steps if s.status is StepStatus.SUCCEEDED]
    next_id = kept[-1].id + 1 if kept else 1
    for index, step in enumerate(new_plan.steps, start=next_id):
        step.id = index
    rebuilt = Plan(objective=plan.objective, steps=kept + list(new_plan.steps))
    logger.info("Replanned: %d kept, %d new steps.", len(kept), len(new_plan.steps))
    return rebuilt
```

## The Executor (implement this in `executor.py`)

The executor runs a single step by calling a tool from the registry and
captures the result on the step. Tools never crash the loop: exceptions become a
failed step.

```python
# <project>/executor.py
"""Executor: runs a single step through a tool registry and captures the result."""

from __future__ import annotations

from typing import Any, Callable

from plan import Plan, Step, StepStatus

Tool = Callable[..., Any]

DEFAULT_TOOL = "execute_step"


class ToolNotFoundError(Exception):
    """Raised when a step names a tool that is not registered."""


class ToolRegistry:
    """Maps tool names to plain Python callables."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str) -> Callable[[Tool], Tool]:
        def decorator(fn: Tool) -> Tool:
            self._tools[name] = fn
            return fn

        return decorator

    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Unknown tool {name!r}. Registered: {', '.join(self.names())}"
            )
        return self._tools[name](**kwargs)


class Executor:
    """Runs one step. The step names a tool; the registry executes it."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def run(self, plan: Plan, step: Step) -> Step:
        step.status = StepStatus.RUNNING
        tool_name = step.tool or DEFAULT_TOOL
        try:
            result = self.registry.call(
                tool_name, objective=plan.objective, goal=step.goal
            )
        except Exception as exc:  # noqa: BLE001 - a tool must not kill the loop
            step.output = None
            step.error = f"{type(exc).__name__}: {exc}"
            step.result = {"ok": False, "error": step.error}
            step.status = StepStatus.FAILED
            return step
        step.output = result
        step.result = {"ok": True, "result": result}
        step.status = StepStatus.SUCCEEDED
        return step
```

## The Verifier (implement this in `verify.py`)

The verifier gates progress: a step only counts as done when its output meets
its verification criteria. Two strategies ship — a dependency-free heuristic
(matches non-stopword keywords against the output text) and an LLM judge.

```python
# <project>/verify.py
"""Verifier: checks a step's output against its verification criteria.

Two strategies:
- HeuristicVerifier: cheap, dependency-free keyword matching. Fine for smoke runs.
- LLMVerifier: an LLM judges whether the output satisfies the criteria.

`build_verifier("auto")` picks the LLM when OPENAI_API_KEY is set, else heuristic.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from config import extract_json, load_openai_client
from plan import Step

STOPWORDS = {
    "a", "an", "and", "any", "all", "as", "by", "each", "every", "for",
    "from", "in", "into", "is", "not", "of", "on", "or", "out", "that",
    "the", "this", "to", "was", "with",
}

VERIFIER_SYSTEM_PROMPT = """You are the verifier of a Plan-and-Execute agent.

Given a step's goal, verification criteria, and the executor's output, decide
whether the output satisfies every criterion. Be strict: if a criterion cannot
be confirmed from the output, the step fails.

Reply with JSON only:
{"passed": true, "reason": "..."}
or
{"passed": false, "reason": "..."}
"""


class HeuristicVerifier:
    def check(self, step: Step) -> bool:
        if step.output is None:
            return False
        text = str(step.output).lower()
        if not text.strip():
            return False
        if not step.verification_criteria:
            return True
        return all(
            self._matches(criterion, text) for criterion in step.verification_criteria
        )

    def _matches(self, criterion: str, text: str) -> bool:
        tokens = [
            token
            for token in re.split(r"\W+", criterion.lower())
            if token and token not in STOPWORDS
        ]
        if not tokens:
            return True  # empty / stopword-only criteria always pass
        return any(token in text for token in tokens)


class LLMVerifier:
    def __init__(self, client: Any = None) -> None:
        self.client = client or load_openai_client()
        if self.client is None:
            raise ValueError(
                "LLM verifier needs credentials: set OPENAI_API_KEY (or pass a client)."
            )

    def check(self, step: Step) -> bool:
        payload = json.dumps(
            {
                "goal": step.goal,
                "criteria": step.verification_criteria,
                "output": str(step.output),
            },
            indent=2,
            ensure_ascii=False,
        )
        response = self.client.chat.completions.create(
            model=os.getenv("VERIFIER_MODEL", "gpt-4o-mini"),
            temperature=0.0,
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        )
        verdict = extract_json(response.choices[0].message.content)
        return bool(verdict.get("passed", False))


def build_verifier(mode: str = "auto", *, client: Any = None):
    """mode: "auto" | "heuristic" | "llm"."""
    if mode == "heuristic":
        return HeuristicVerifier()
    if mode == "llm":
        if client is None:
            raise ValueError("LLM verifier needs credentials: set OPENAI_API_KEY.")
        return LLMVerifier(client=client)
    if mode == "auto" and client is not None:
        return LLMVerifier(client=client)
    return HeuristicVerifier()
```

## The Main Loop (implement this in `agent.py`)

The loop is the core of the pattern: plan → execute → verify → replan on
failure, until every step succeeds or the replan budget is exhausted.

```python
# <project>/agent.py
"""The Plan-and-Execute loop: plan -> execute -> verify -> (replan until done)."""

from __future__ import annotations

import logging

from executor import Executor
from plan import Plan, StepStatus
from planner import build_plan, replan

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        executor: Executor,
        verifier,
        max_attempts: int = 3,
        planner_client=None,
    ) -> None:
        self.executor = executor
        self.verifier = verifier
        self.max_attempts = max_attempts
        self.planner_client = planner_client

    def run(self, objective: str, *, tools: list[str] | None = None) -> Plan:
        tool_names = tools or self.executor.registry.names()
        plan = build_plan(objective, tools=tool_names, client=self.planner_client)
        logger.info("Plan: %d step(s) for %r", len(plan.steps), objective)

        attempts = 0
        while not plan.all_done():
            step = plan.next_pending()
            if step is None:
                break

            self.executor.run(plan, step)

            if step.status is StepStatus.SUCCEEDED:
                if not self.verifier.check(step):
                    step.status = StepStatus.FAILED
                    step.error = "step output did not meet verification criteria"
                else:
                    logger.info("Step %d succeeded and passed verification.", step.id)
                    continue

            attempts += 1
            if attempts >= self.max_attempts:
                logger.error(
                    "Gave up after %d replan attempt(s); marking remaining steps failed.",
                    attempts,
                )
                self._abandon_remaining(plan)
                break

            logger.warning("Step %d failed (%s); replanning.", step.id, step.error)
            plan = replan(plan, step, tools=tool_names, client=self.planner_client)

        return plan

    def _abandon_remaining(self, plan: Plan) -> None:
        for step in plan.steps:
            if step.status is StepStatus.PENDING:
                step.status = StepStatus.FAILED
                step.error = "abandoned: replan attempt limit reached"
```

## Example Domain Tools (implement this in `tools.py`)

Every tool is a plain callable of the form `(objective, goal) -> str`, so the
executor can call any of them generically. Swap this module for the user's
domain tools.

```python
# <project>/tools.py
"""Example domain tools: a small persistent todo list backed by a JSON file."""

from __future__ import annotations

import json
import os
from typing import Any

from executor import ToolRegistry


class ToDoStore:
    """Minimal JSON-backed list of todo items."""

    def __init__(self, path: str = "todos.json") -> None:
        self.path = path
        self.items: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as handle:
                self.items = json.load(handle)

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.items, handle, indent=2, ensure_ascii=False)

    def add(self, title: str) -> str:
        self.items.append({"title": title, "done": False})
        self.save()
        return f"added todo #{len(self.items)}: {title}"

    def text(self) -> str:
        if not self.items:
            return "No todos recorded yet."
        return "\n".join(
            f"{i}. [{'x' if item['done'] else ' '}] {item['title']}"
            for i, item in enumerate(self.items, start=1)
        )


def make_registry(todo_path: str = "todos.json") -> ToolRegistry:
    """Build a registry of example tools for this harness."""
    store = ToDoStore(path=todo_path)
    registry = ToolRegistry()

    @registry.register("execute_step")
    def execute_step(objective: str, goal: str) -> str:
        store.add(title=goal)
        return (
            "broke the work down into concrete sub-actions, performed every "
            "planned action, verified completeness, and summarized the outcome "
            f"(objective={objective!r}, goal={goal!r}, todos={len(store.items)})."
        )

    @registry.register("list_todos")
    def list_todos(objective: str, goal: str) -> str:
        return store.text()

    @registry.register("clear_todos")
    def clear_todos(objective: str, goal: str) -> str:
        count = len(store.items)
        store.items.clear()
        store.save()
        return f"Removed {count} todo(s)."

    return registry
```

## CLI (implement this in `main.py`)

```python
# <project>/main.py
"""CLI entry point: run the Plan-and-Execute harness on one objective."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from agent import Agent
from config import env_int, load_openai_client
from executor import Executor
from tools import make_registry
from verify import build_verifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py", description="Run a Plan-and-Execute agent harness."
    )
    parser.add_argument(
        "objective",
        nargs="?",
        default=None,
        help="Task objective (or set OBJECTIVE in the environment).",
    )
    parser.add_argument(
        "--todo-path",
        default=None,
        help="Where example todos are stored (env: TODO_PATH, default: todos.json).",
    )
    parser.add_argument(
        "--verify-mode",
        choices=["auto", "heuristic", "llm"],
        default=None,
        help="Which verifier to use (env: VERIFY_MODE, default: auto).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Replan attempts before giving up (env: MAX_ATTEMPTS, default: 3).",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    objective = args.objective or os.getenv("OBJECTIVE", "").strip()
    if not objective:
        print(
            "Error: no objective given. Pass it as an argument or set OBJECTIVE=...",
            file=sys.stderr,
        )
        return 2

    client = load_openai_client()
    registry = make_registry(todo_path=args.todo_path or os.getenv("TODO_PATH", "todos.json"))
    verifier = build_verifier(
        args.verify_mode or os.getenv("VERIFY_MODE", "auto"), client=client
    )
    agent = Agent(
        executor=Executor(registry),
        verifier=verifier,
        max_attempts=args.max_attempts or env_int("MAX_ATTEMPTS", 3),
        planner_client=client,
    )

    plan = agent.run(objective)
    print("\n=== FINAL PLAN ===")
    print(plan.summary_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## requirements.txt

```text
openai>=1.30.0
python-dotenv>=1.0.0
```

## .env.example

```text
# OpenAI
OPENAI_API_KEY=
OPENAI_BASE_URL=
PLANNER_MODEL=gpt-4o-mini
VERIFIER_MODEL=gpt-4o-mini

# Harness
OBJECTIVE=Prepare a weekly summary of the project status
VERIFY_MODE=auto
MAX_ATTEMPTS=3
TODO_PATH=todos.json
```

## Verification

Before declaring the project done:

1. `python -m py_compile plan.py config.py planner.py executor.py verify.py tools.py agent.py main.py`
   — every module is syntactically valid.
2. `python main.py "prepare a weekly status summary"` runs **without** an API key:
   local fallback planner + heuristic verifier, everything succeeds, final plan
   JSON prints (all steps `succeeded`).
3. With `OPENAI_API_KEY` set: `PLANNER_MODEL`/`VERIFIER_MODEL` drive the LLM
   path; multi-step objective produces a real derived plan.
4. Failure path: introduce a tool error and confirm the step becomes `failed`,
   the remaining steps get replanned, and the run stops after `--max-attempts`
   replans (remaining steps marked `failed`).
5. Confirm `todos.json` was created/persisted by the example tools.

## Conventions

- Prefer LLM planning/verification; keep the heuristic paths so the harness is
  runnable and testable without paying for tokens.
- Coordinates come from `OPENAI_API_KEY`/`OPENAI_BASE_URL`; never hardcode or
  print keys, and ship `.env.example`, never `.env`.
- Every tool is `(objective, goal) -> str`; wrapping it in a try/except in the
  executor means a bad tool can never kill the loop.
- A verified step is final — `replan` only regenerates the failed/unfinished
  part, never redos succeeded work.
- The verifier gates progress: no output matching the criteria, no next step.

## Example

User: "Build a plan-and-execute agent that plans and executes a weekly status
summarization task, storing progress in a todo list — name the folder
`status-bot`."

You: scaffold `status-bot/` with the modules above, example tools
`execute_step`/`list_todos`/`clear_todos`, `.env.example`, README; run
`py_compile` on every file; run `python main.py "summarize this week"` offline;
report the structure and the exact run command. Adjust the domain tools to the
user's real needs if they provided any.