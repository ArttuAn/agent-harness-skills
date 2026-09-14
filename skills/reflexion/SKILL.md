---
name: harness-reflexion
description: "Build a Reflexion pattern agent harness from a spec"
---

# Reflexion Agent Harness

## What is Reflexion?

Reflexion (Shinn et al. 2023) is a pattern where an agent improves across
retries by writing self-reflection notes after each failure. Unlike ReAct's
single-shot loop, Reflexion runs multiple **trials**: the agent acts, the result
is evaluated, and if it fails, a reflector LLM call produces a structured
self-critique that gets stored in episodic memory. On the next trial, those
reflections are injected into the system prompt so the agent knows what went
wrong and what to change.

```
Trial loop:
  1. Act      — run the ReAct-style action loop to produce a final answer
  2. Evaluate — check the answer against the task (pass/fail + reason)
  3. Reflect  — if failed: LLM reads transcript + failure reason, writes a
                self-reflection note (key mistake, corrected strategy)
  4. Store    — append reflection to episodic memory
  5. Retry    — loop back to step 1 with updated memory in the prompt
  ...repeat up to max_trials
```

The core insight: memory grows across trials, so the agent gets progressively
better context about its own failure modes without any weight updates.

## Workflow

When the user gives a spec (or says "build me a Reflexion agent"), do the
following:

1. **Ask for a spec if none given.** The user should describe:
   - What the agent should solve (coding task, math problem, QA, etc.)
   - How to judge success (unit tests pass? correct answer? specific format?)
   - What tools the agent needs (if any)

2. **Scaffold the project** into the requested directory. Use a clean `src/`
   layout with `pyproject.toml`.

3. **Write all five modules** with complete, runnable code:
   - `agent.py` — ReAct-style action loop that produces a final answer
   - `reflector.py` — given transcript + failure, produces a self-reflection
   - `memory.py` — episodic memory store (list of reflection strings)
   - `evaluator.py` — checks if the final answer passes (pass/fail + reason)
   - `reflexion_loop.py` — orchestrates trials with memory injection
   - `config.py` — env-based config
   - `cli.py` — CLI entry point with `--max-trials`

4. **Emphasize memory injection** in code comments. This is the key mechanism
   that differentiates Reflexion from plain retry loops.

5. **Verify** the project imports and `--help` runs.

## Project Layout

```
<project>/
├── pyproject.toml
├── .env.example
├── README.md
└── src/<package>/
    ├── __init__.py
    ├── config.py            # env/.env config: model, base_url, api_key, max_trials
    ├── agent.py             # single-trial ReAct loop (thought → action → observation)
    ├── reflector.py         # LLM call: transcript + failure → self-reflection note
    ├── memory.py            # episodic memory: list of reflection strings
    ├── evaluator.py         # is the final answer acceptable? pass/fail + reason
    ├── reflexion_loop.py    # orchestrator: trial loop with memory injection
    ├── cli.py               # CLI entry point with --max-trials flag
    └── tools.py             # (optional) tools for the agent to call
```

## Dependencies

Use only:

- `openai` (for LLM calls)
- `python-dotenv` (optional, for `.env` loading)
- stdlib (`json`, `typing`, `dataclasses`, etc.)

No langchain, no instructor, no heavy frameworks.

## Module: `config.py`

```python
# src/<package>/config.py
"""Configuration from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader — no external dependency."""
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
    max_trials: int = 3
    max_steps_per_trial: int = 10

    @classmethod
    def from_env(cls, dotenv_path: str = ".env") -> Config:
        load_dotenv(dotenv_path)
        return cls(
            model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("AGENT_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("AGENT_API_KEY", ""),
            max_trials=int(os.getenv("AGENT_MAX_TRIALS", "3")),
            max_steps_per_trial=int(os.getenv("AGENT_MAX_STEPS", "10")),
        )
```

## Module: `memory.py`

```python
# src/<package>/memory.py
"""Episodic memory — stores self-reflection notes across trials.

This is the core Reflexion mechanism. Each failed trial produces a reflection
string (what went wrong, what to change). These accumulate in memory and are
injected into the agent's system prompt on subsequent trials, giving it
progressively better context about its own failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodicMemory:
    """Ordered list of reflection notes from past failed trials.

    On each new trial, the agent's system prompt includes all prior
    reflections so it can learn from its mistakes.
    """

    reflections: list[str] = field(default_factory=list)

    def add(self, reflection: str) -> None:
        """Append a reflection note from a failed trial."""
        self.reflections.append(reflection)

    def format_for_prompt(self) -> str:
        """Format all reflections for injection into the system prompt.

        This is the KEY MEMORY INJECTION mechanism: every reflection from
        every past trial is included, giving the agent explicit knowledge
        of what went wrong before and what strategies to try instead.
        """
        if not self.reflections:
            return ""
        lines = [
            "## Lessons from past failed attempts",
            "",
            "Read these carefully — they describe mistakes you made and "
            "strategies you should try instead:",
            "",
        ]
        for i, ref in enumerate(self.reflections, 1):
            lines.append(f"### Attempt {i} reflection")
            lines.append(ref)
            lines.append("")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.reflections)

    def __bool__(self) -> bool:
        return bool(self.reflections)
```

## Module: `reflector.py`

```python
# src/<package>/reflector.py
"""Self-reflection module — produces structured critique after a failed trial.

The reflector takes the full transcript of a failed agent run plus the
evaluator's failure reason, and asks the LLM to produce a concise self-critique
identifying:
  1. The key mistake(s) that caused the failure
  2. A corrected strategy to try on the next attempt
"""

from __future__ import annotations

from openai import OpenAI

REFLECTION_PROMPT = """\
You are a self-reflection assistant. The following agent just FAILED a task.

## Task
{task}

## Transcript of the failed attempt
{transcript}

## Why it failed
{failure_reason}

---

Write a concise self-reflection note (3-6 sentences) that covers:
1. The key mistake(s) — what specifically went wrong
2. A corrected strategy — what the agent should do differently next time

Be specific and actionable. This note will be shown to the agent before its
next attempt so it can learn from this failure.
"""


class Reflector:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def reflect(
        self,
        task: str,
        transcript: str,
        failure_reason: str,
    ) -> str:
        """Produce a self-reflection note for a failed trial.

        Args:
            task: the original user task
            transcript: full conversation transcript from the failed trial
            failure_reason: why the evaluator rejected the answer

        Returns:
            A reflection string to be stored in episodic memory.
        """
        prompt = REFLECTION_PROMPT.format(
            task=task,
            transcript=transcript,
            failure_reason=failure_reason,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise self-reflection assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or "No reflection produced."
```

## Module: `evaluator.py`

```python
# src/<package>/evaluator.py
"""Evaluator — checks if the agent's final answer is acceptable.

Unlike plan-execute (which has per-step checks), Reflexion evaluates the
whole turn's final answer as pass/fail. The evaluator returns a structured
result with the pass/fail decision and a reason string that feeds into the
reflector.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI


@dataclass
class EvalResult:
    passed: bool
    reason: str


# Default evaluation prompt — customize per task domain.
EVAL_PROMPT = """\
You are an answer evaluator. Judge whether the agent's answer correctly
completes the task.

## Task
{task}

## Agent's answer
{answer}

---

Respond with EXACTLY one JSON object:
{{"passed": true/false, "reason": "brief explanation of why it passed or failed"}}

Be strict: "passed" should only be true if the answer actually solves the task.
"""


class Evaluator:
    def __init__(self, client: OpenAI, model: str, eval_prompt: str | None = None) -> None:
        self.client = client
        self.model = model
        self.eval_prompt = eval_prompt or EVAL_PROMPT

    def evaluate(self, task: str, answer: str) -> EvalResult:
        """Evaluate the agent's final answer against the task.

        Returns EvalResult(passed=bool, reason=str).
        """
        prompt = self.eval_prompt.format(task=task, answer=answer)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strict answer evaluator. Reply with only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        content = content.strip()
        # Tolerate fenced JSON
        fence = chr(96) * 3
        if content.startswith(fence):
            content = content[len(fence):]
            if content.endswith(fence):
                content = content[:-len(fence)]
            content = content.strip()
            if content.startswith("json"):
                content = content[4:]
        import json
        try:
            data = json.loads(content)
            return EvalResult(
                passed=bool(data.get("passed", False)),
                reason=str(data.get("reason", "No reason given.")),
            )
        except (json.JSONDecodeError, KeyError):
            return EvalResult(passed=False, reason=f"Evaluator returned invalid JSON: {content}")
```

## Module: `agent.py`

```python
# src/<package>/agent.py
"""Single-trial agent — a ReAct-style action loop that produces a final answer.

This is called once per trial. The reflexion_loop orchestrator passes in
the episodic memory, which gets injected into the system prompt so the agent
can learn from past failures.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


AGENT_SYSTEM_PROMPT = """\
You are a problem-solving agent. You reason and act in a loop.

{memory_section}

For each turn, respond with EXACTLY one JSON object of one of these shapes:

1. To take an action:
{{"thought": "<your reasoning>", "action": {{"name": "<tool>", "args": {{"<param>": <value>}}}}}}

2. To finish with your answer:
{{"thought": "<your final reasoning>", "answer": "<your final answer>"}}

Available tools:
{tools}
"""


class Agent:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools: str = "No tools available.",
        max_steps: int = 10,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools
        self.max_steps = max_steps

    def run(
        self,
        task: str,
        memory_text: str = "",
        tool_executor=None,
        verbose: bool = True,
    ) -> tuple[str, str]:
        """Run one trial of the agent.

        Args:
            task: the user's task
            memory_text: formatted reflections from EpisodicMemory.format_for_prompt()
            tool_executor: optional callable(name, args) -> str for tool execution
            verbose: print progress

        Returns:
            (final_answer, transcript) — the answer string and full message history
        """
        memory_section = memory_text if memory_text else "(No past failures — this is your first attempt.)"

        system = AGENT_SYSTEM_PROMPT.format(
            memory_section=memory_section,
            tools=self.tools,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        transcript_lines: list[str] = [f"TASK: {task}", ""]

        for step in range(1, self.max_steps + 1):
            if verbose:
                print(f"  [step {step}/{self.max_steps}] asking model...")

            decision = self._ask(messages)
            thought = decision.get("thought", "")
            transcript_lines.append(f"[step {step}] thought: {thought}")

            if verbose and thought:
                print(f"  [thought] {thought}")

            if "answer" in decision:
                answer = str(decision["answer"])
                transcript_lines.append(f"[answer] {answer}")
                if verbose:
                    print(f"  [answer] {answer}")
                return answer, "\n".join(transcript_lines)

            action = decision.get("action")
            if not isinstance(action, dict) or "name" not in action:
                messages.append({
                    "role": "user",
                    "content": "Invalid response. Output JSON with 'answer' or 'action'.",
                })
                transcript_lines.append("[error] invalid response format")
                continue

            name, args = action["name"], action.get("args", {})
            transcript_lines.append(f"[action] {name}({args})")

            if verbose:
                print(f"  [action] {name}({args})")

            if tool_executor:
                observation = tool_executor(name, args)
            else:
                observation = f"No tool executor configured. Cannot run '{name}'."

            transcript_lines.append(f"[observation] {observation}")
            if verbose:
                print(f"  [observation] {observation}")

            messages.append({
                "role": "assistant",
                "content": json.dumps(decision, ensure_ascii=False),
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}\nContinue.",
            })

        answer = f"Step budget ({self.max_steps}) exceeded."
        transcript_lines.append(answer)
        return answer, "\n".join(transcript_lines)

    def _ask(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Send messages, get JSON back. Retries on parse failure."""
        for _ in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
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
                messages.append({
                    "role": "user",
                    "content": "You did not output valid JSON. Reply with only a JSON object.",
                })
        raise RuntimeError("Model repeatedly failed to produce valid JSON.")
```

## Module: `reflexion_loop.py` — The Orchestrator

This is the central module. It ties everything together and implements the
trial loop with memory injection.

```python
# src/<package>/reflexion_loop.py
"""Reflexion orchestrator — runs multiple trials with memory injection.

This is the heart of the Reflexion pattern:

  for trial in range(max_trials):
      1. INJECT memory into agent prompt  ← this is the key mechanism
      2. Run agent → get answer + transcript
      3. Evaluate answer → pass or fail
      4. If pass → return success
      5. If fail  → reflect on failure, store reflection in memory, retry

Memory grows each trial, giving the agent progressively better context.
"""

from __future__ import annotations

from openai import OpenAI

from .agent import Agent
from .evaluator import Evaluator, EvalResult
from .memory import EpisodicMemory
from .reflector import Reflector


class ReflexionLoop:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_trials: int = 3,
        max_steps_per_trial: int = 10,
        tools: str = "No tools available.",
        tool_executor=None,
        eval_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.max_trials = max_trials
        self.tool_executor = tool_executor

        self.agent = Agent(
            client=client,
            model=model,
            tools=tools,
            max_steps=max_steps_per_trial,
        )
        self.evaluator = Evaluator(client=client, model=model, eval_prompt=eval_prompt)
        self.reflector = Reflector(client=client, model=model)
        self.memory = EpisodicMemory()

    def run(self, task: str, verbose: bool = True) -> tuple[bool, str, EpisodicMemory]:
        """Run the full Reflexion loop.

        Args:
            task: the user's task description
            verbose: print trial progress

        Returns:
            (success, final_answer, memory) — whether any trial passed,
            the best answer found, and the accumulated memory.
        """
        for trial in range(1, self.max_trials + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"TRIAL {trial}/{self.max_trials}")
                print(f"{'='*60}")
                if self.memory:
                    print(f"[memory] {len(self.memory)} reflection(s) from past failures")

            # ─── KEY MEMORY INJECTION STEP ───────────────────────────
            # On trial 1, memory is empty → agent gets no extra context.
            # On trial 2+, memory contains reflections from all past
            # failures. format_for_prompt() builds a string like:
            #   "## Lessons from past failed attempts
            #    ### Attempt 1 reflection
            #    I failed because I used wrong formula..."
            # This string is passed to agent.run() as memory_text, which
            # gets injected into the system prompt under a dedicated
            # section. The agent sees its own past mistakes explicitly.
            # ─────────────────────────────────────────────────────────
            memory_text = self.memory.format_for_prompt()

            answer, transcript = self.agent.run(
                task=task,
                memory_text=memory_text,
                tool_executor=self.tool_executor,
                verbose=verbose,
            )

            if verbose:
                print(f"\n[eval] evaluating answer...")

            eval_result = self.evaluator.evaluate(task=answer)

            if eval_result.passed:
                if verbose:
                    print(f"[eval] PASSED — {eval_result.reason}")
                return True, answer, self.memory

            if verbose:
                print(f"[eval] FAILED — {eval_result.reason}")

            # ─── REFLECTION + MEMORY UPDATE ──────────────────────────
            # The reflector reads the full transcript and failure reason,
            # then produces a structured self-critique. This critique is
            # appended to episodic memory, so on the NEXT trial it will
            # be injected into the agent's prompt.
            # ─────────────────────────────────────────────────────────
            if verbose:
                print(f"[reflect] generating self-reflection...")

            reflection = self.reflector.reflect(
                task=task,
                transcript=transcript,
                failure_reason=eval_result.reason,
            )
            self.memory.add(reflection)

            if verbose:
                print(f"[reflect] {reflection[:200]}...")

        if verbose:
            print(f"\n[done] all {self.max_trials} trials exhausted without success.")
        return False, answer, self.memory
```

## Module: `cli.py`

```python
# src/<package>/cli.py
"""CLI entry point — run the Reflexion agent from the command line."""

import argparse
import sys

from openai import OpenAI

from .config import Config
from .reflexion_loop import ReflexionLoop


def build_loop(cfg: Config, tools: str = "No tools available.", tool_executor=None) -> ReflexionLoop:
    return ReflexionLoop(
        client=OpenAI(api_key=cfg.api_key, base_url=cfg.base_url),
        model=cfg.model,
        max_trials=cfg.max_trials,
        max_steps_per_trial=cfg.max_steps_per_trial,
        tools=tools,
        tool_executor=tool_executor,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="<command>",
        description="Reflexion agent — improves through self-reflection across trials",
    )
    parser.add_argument("prompt", nargs="*", help="Task for the agent to solve.")
    parser.add_argument("--model", default=None, help="Override model.")
    parser.add_argument("--max-trials", type=int, default=None, help="Max retry trials (default: 3).")
    parser.add_argument("--max-steps", type=int, default=None, help="Max action steps per trial.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output.")
    args = parser.parse_args(argv)

    cfg = Config.from_env()
    if args.model:
        cfg.model = args.model
    if args.max_trials:
        cfg.max_trials = args.max_trials
    if args.max_steps:
        cfg.max_steps_per_trial = args.max_steps

    if not cfg.api_key:
        print("No AGENT_API_KEY set. Add it to .env or export it.", file=sys.stderr)
        return 2

    task = " ".join(args.prompt)
    if not task:
        parser.print_help()
        return 1

    loop = build_loop(cfg)
    success, answer, memory = loop.run(task, verbose=not args.quiet)

    print(f"\n{'='*60}")
    if success:
        print(f"SUCCESS: {answer}")
    else:
        print(f"FAILED after {len(memory)} trial(s).")
        print(f"Last answer: {answer}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
```

## `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "<package-kebab>"
version = "0.1.0"
description = "A Reflexion agent harness for <spec>"
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

## `.env.example`

```
AGENT_API_KEY=sk-...
# AGENT_BASE_URL=https://api.openai.com/v1
# AGENT_MODEL=gpt-4o-mini
# AGENT_MAX_TRIALS=3
# AGENT_MAX_STEPS=10
```

## Key Design Decisions

1. **Memory injection is explicit.** `EpisodicMemory.format_for_prompt()` builds
   a clearly labeled section that gets spliced into the system prompt. The agent
   sees "Lessons from past failed attempts" with numbered reflection notes.

2. **Evaluator is pluggable.** Pass a custom `eval_prompt` to `ReflexionLoop` to
   change how answers are judged. The default uses an LLM judge; you could swap in
   a code-execution checker, unit-test runner, etc.

3. **Transcript flows: agent → reflector → memory → agent.** The agent produces
   a full transcript (not just the answer). The reflector reads the whole thing to
   understand what went wrong. The reflection goes into memory. On the next trial,
   memory is in the prompt, so the agent sees its own past reasoning.

4. **Each trial is independent.** The agent gets a fresh message list each trial —
   it doesn't carry forward the old conversation. Only the memory reflections
   persist, which keeps context clean.

5. **No weight updates.** Reflexion achieves improvement purely through prompt
   engineering and episodic memory. No fine-tuning required.

## Verification

1. `python -m pip install -e .` in the project directory.
2. `python -m <package>.cli --help` — confirm help renders, no ImportError.
3. `python -m <package>.cli --max-trials 1 "what is 2+2?"` — single trial smoke test.
4. `python -m <package>.cli --max-trials 3 "write a function that..."` — full
   multi-trial test. Watch memory grow across trials.

## Conventions

- Never print or commit API keys.
- Tool errors return as strings; the agent loop should never crash.
- `json.dumps(..., default=str)` for serializing any observation data.
- temp=0 for the agent and evaluator (determinism); temp=0.3 for reflector
  (slightly creative reflection, but still focused).
- Keep memory as plain strings — no embedding storage, no vector DB.
