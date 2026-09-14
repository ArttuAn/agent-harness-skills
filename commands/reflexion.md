# Build a Reflexion Agent Harness

Given the user's spec, build a complete Reflexion-pattern agent harness:
a runnable Python project where the agent acts, gets evaluated, and on
failure writes a self-reflection stored in episodic memory to retry smarter.
Ask for a spec if none given (domain + how success is judged).

## What to build

Scaffold a `src/`-layout project with `pyproject.toml` into the requested
directory (default: a folder named after the project). Use the `openai`
library for LLM calls; no other heavy deps (dotenv is fine).

Structure:

```
<project>/                     # e.g. math-solver → package math_solver
├── pyproject.toml
├── .env.example
├── README.md
└── src/<package>/
    ├── __init__.py
    ├── config.py        # env/.env config: model, base_url, api_key, max_trials
    ├── agent.py         # single-trial action loop (LLM → answer, ReAct-style)
    ├── reflector.py     # transcript + failure → self-reflection note
    ├── memory.py        # episodic memory: list of reflections, injected into prompt
    ├── evaluator.py     # is the final answer acceptable? pass/fail + reason
    ├── reflexion_loop.py# orchestrates trials: act → eval → reflect → retry
    └── cli.py           # CLI with --max-trials
```

## Core pieces

1. **Loop** (`reflexion_loop.py`): for `trial in range(max_trials)`: inject
   memory → run agent → evaluate answer → if pass, return; if fail, call
   reflector, append to memory, retry. This is Shinn et al. 2023: no weight
   updates, improvement comes from episodic memory in the prompt.

2. **Memory injection goes in `agent.py`** — its system prompt has a
   `{memory_section}` slot filled from `EpisodicMemory.format_for_prompt()`,
   which renders "Lessons from past failed attempts" with numbered reflection
   notes. On trial 1 the slot says "(No past failures...)". This explicit
   injection is the defining Reflexion mechanism — call it out in comments.

3. **Agent** (`agent.py`): ReAct-style loop, JSON `{thought, action}` / 
   `{thought, answer}`, returns `(final_answer, transcript)`. Tolerate fenced
   JSON; never crash on a tool error. Tools optional — pass a
   `tool_executor=callable(name, args) -> str`.

4. **Reflector** (`reflector.py`): prompt template with Task + Transcript +
   failure reason; asks LLM for 3-6 sentence note: (1) key mistake, (2)
   corrected strategy. temp=0.3.

5. **Evaluator** (`evaluator.py`): whole-turn pass/fail (unlike plan-execute).
   LLM judge returns `{"passed": bool, "reason": str}`; make it pluggable via
   `eval_prompt=` (swap for a test-runner if the spec wants).

6. **Config** (`config.py`): env vars `AGENT_MODEL`, `AGENT_BASE_URL`,
   `AGENT_API_KEY`, `AGENT_MAX_TRIALS`, `AGENT_MAX_STEPS`; tiny `.env` loader.

7. **Memory** (`memory.py`): just a list of reflection strings. No vector DB.

## Conventions

- temp=0 for agent + evaluator; reflector 0.3.
- Each trial gets a fresh message list; only memory persists.
- Never print or commit API keys; ship `.env.example` only.
- Plain strings for reflections — no embeddings.

## Example code (implement with your own words)

```python
# reflexion_loop.py — the heart of it
def run(self, task: str, verbose: bool = True):
    for trial in range(1, self.max_trials + 1):
        memory_text = self.memory.format_for_prompt()   # ← inject memory
        answer, transcript = self.agent.run(task, memory_text=memory_text)
        result = self.evaluator.evaluate(task, answer)
        if result.passed:
            return True, answer, self.memory
        reflection = self.reflector.reflect(task, transcript, result.reason)
        self.memory.add(reflection)                     # ← episodic memory
    return False, answer, self.memory
```

```python
# memory.py — the injection format
def format_for_prompt(self) -> str:
    if not self.reflections:
        return ""
    parts = ["## Lessons from past failed attempts", "",
             "Read these carefully — they describe mistakes you made "
             "and strategies you should try instead:", ""]
    for i, ref in enumerate(self.reflections, 1):
        parts += [f"### Attempt {i} reflection", ref, ""]
    return "\n".join(parts)
```

```python
# agent.py — system prompt with the memory slot
AGENT_SYSTEM_PROMPT = """You are a problem-solving agent.
{memory_section}

Respond with EXACTLY one JSON object:
{{"thought": "...", "action": {{"name": "...", "args": {{}}}}}}
or {{"thought": "...", "answer": "..."}}
"""
```

## Verify before finishing

1. `pip install -e .` (or `uv sync`).
2. `python -m <package>.cli --help` — no ImportError.
3. `python -m <package>.cli --max-trials 1 "2+2?"` smoke test; then
   `--max-trials 3` on a real task — watch memory grow in verbose output.

## Example

User: "Build a math-solver agent that evaluates expressions, retrying with
self-reflection when the answer is wrong."

You: scaffold `math-solver/` with calculator tool, LLM evaluator, reflector,
memory, `--max-trials` CLI, `.env.example`, README; verify imports + `--help`
runs. Report structure and run instructions.