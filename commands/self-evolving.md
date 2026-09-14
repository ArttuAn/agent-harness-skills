# Build a Self-Evolving Agent Harness

Given the user's spec, build a self-evolving agent harness: a Python project that
runs an LLM agent, evaluates its performance, reflects on failures, and applies
improvements to its own prompt/tools/config between generations. Ask for a spec
if missing (domain, task type, evaluation criteria).

## What to build

Scaffold a `src/`-layout project into the requested directory. Use the `openai`
library; minimal deps (dotenv is fine). The core loop is:

```
Run → Evaluate → Reflect → Apply → Repeat × N generations
```

Structure:

```
<project>/
├── pyproject.toml
├── .env.example
├── README.md
└── src/<package>/
    ├── __init__.py
    ├── config.py        # env config: model, base_url, api_key, max_steps
    ├── prompt.py        # SYSTEM_PROMPT constant (edited by applier)
    ├── evaluator.py     # scores transcript: completeness, correctness, efficiency
    ├── reflector.py     # LLM reflection → improvement proposals (JSON)
    ├── applier.py       # writes prompt edits, new tools, config changes
    ├── evolve.py        # main loop: run → eval → reflect → apply × N
    ├── tools/
    │   ├── __init__.py  # @tool registry, list_tools(), execute()
    │   └── examples.py  # 3-5 domain-relevant starter tools
    └── cli.py           # argparse: task arg, --generations, --model
```

## Core modules

### evaluator.py

Score a run transcript against weighted criteria. Return `{score, breakdown, issues}`:

```python
def evaluate(transcript: list[dict], criteria: dict | None = None) -> dict:
    criteria = criteria or {"completeness": 4, "correctness": 4, "efficiency": 2}
    max_weight = sum(criteria.values())
    has_answer = any(
        "answer" in (json.loads(m["content"]) if isinstance(m["content"], str) else {})
        for m in transcript if m["role"] == "assistant"
    )
    tool_calls = sum(
        1 for m in transcript
        if m["role"] == "assistant" and isinstance(m.get("content"), str) and '"action"' in m["content"]
    )
    errors = sum(
        1 for m in transcript
        if m["role"] == "tool" and "error" in (m.get("content") or "").lower()
    )
    breakdown = {
        "completeness": 10 if has_answer else 3,
        "correctness": max(0, 10 - errors * 3),
        "efficiency": max(0, 10 - max(0, tool_calls - 3)),
    }
    weighted = sum(breakdown[k] * criteria.get(k, 1) for k in criteria)
    issues = []
    if not has_answer:
        issues.append("No final answer produced.")
    if errors:
        issues.append(f"{errors} tool error(s).")
    return {"score": int(weighted / max_weight * 10), "breakdown": breakdown, "issues": issues}
```

### reflector.py

Send the transcript + score to the LLM; get back structured improvement proposals:

```python
def reflect(transcript: list[dict], evaluation: dict) -> dict:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if BASE_URL else OpenAI(api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.3,
        messages=[
            {"role": "system", "content": REFLECT_SYSTEM},
            {"role": "user", "content": f"Transcript:\n{json.dumps(transcript[-10:], default=str)}\nEval:\n{json.dumps(evaluation)}"},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content or "{}")
```

The reflector returns: `{prompt_edits: [...], new_tools: [...], tool_fixes: [...], config_tuning: {...}, reasoning: "..."}`.

### applier.py

Apply approved changes — append prompt edits, insert new tool functions, replace broken tools, update `.env` config. Keep changes append-only.

### evolve.py

The main loop:
1. `run_agent(task)` — ReAct-style loop using `SYSTEM_PROMPT` + tool registry
2. `evaluate(transcript)` — score it
3. `reflect(transcript, eval)` — get proposals
4. `apply_improvements(proposals)` — write changes
5. `run_tests()` — execute `pytest` if present; halt on failure
6. Log everything to `generations/` directory
7. Repeat for N generations

## Safety rules

- **Append-only**: applier never deletes existing code, only appends/replaces.
- **Test gate**: run `pytest` after every apply. If tests fail, halt and print revert instructions.
- **Git-friendly**: each generation should be its own git commit so changes are reversible.
- **No secrets**: load API keys from `.env`; ship `.env.example` with dummies.
- **Transcript truncation**: reflector only sees last 10 messages to control token cost.

## Conventions

- temp=0 for agent runs; 0.3 for reflection.
- `json.dumps(..., default=str)` for serializing observations.
- Register tools by importing examples module in the package `__init__.py`.

## Verify

1. `pip install -e .` — clean imports.
2. `python -m <package>.cli "What is 2+2?" -g 2` — loop runs, `generations/` has log files.
3. Inspect `generations/gen_001_proposals.json` — structured proposals present.
