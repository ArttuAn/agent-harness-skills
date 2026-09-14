# Build a ReAct Agent Harness

Given the user's spec, build a complete ReAct-pattern agent harness: a runnable
Python project with a Thought → Action → Observation loop. Ask for a spec if they
didn't give one (domain + what example tools make sense); propose 3-5 tools if
vague.

## What to build

Scaffold a `src/`-layout project with `pyproject.toml` into the requested
directory (default: a folder named after the project). Use the `openai` library
for LLM calls; no other heavy deps (dotenv is fine).

Structure:

```
<project>/                     # e.g. weather-bot → package weather_bot
├── pyproject.toml
├── .env.example
├── README.md
└── src/<package>/
    ├── __init__.py
    ├── config.py     # env/.env config: model, base_url, api_key, max_steps
    ├── loop.py       # the ReAct loop + chat REPL
    ├── tools.py      # @tool decorator, registry, type-hint→JSON-schema inference
    ├── cli.py        # argparse entry: prompt arg, --chat, --model, --max-steps
    └── tools/examples.py  # 3-5 domain-relevant example tools
```

## Core pieces

1. **ReAct loop** (`loop.py`): system prompt telling the model to reply with a JSON
   object containing either `{thought, action:{name,args}}` or `{thought, answer}`.
   Loop: call LLM → execute tool → append observation → repeat until an `answer`
   appears or `max_steps` is hit. Tolerate fenced/wrapped JSON; retry on parse
   failure; never crash on a tool error (return error as observation string).
2. **Registry** (`tools.py`): `@tool`/`@registry.register` decorator that infers
   JSON-schema params from type hints (`str`→string, `int`→integer, `Optional`→
   drop None, `Literal`→enum, `list[X]`→array). Keep a module-global registry so
   example tools self-register; `execute(name, args)` returns JSON string.
3. **Config** (`config.py`): env vars `AGENT_MODEL`, `AGENT_BASE_URL`,
   `AGENT_MAX_STEPS`, `AGENT_API_KEY`; load `.env` if present.
4. **CLI** (`cli.py`): `<command> "prompt"`, `--chat` REPL, `--model`,
   `--max-steps`. Exit 2 with a message if no API key.
5. **Example tools**: match the spec (e.g. calculator, date/time, filesystem
   helpers, a stub web search). Each is a plain function + `@tool` with a docstring.

## Conventions

- temp=0 for deterministic tool calls; keep loop state in a `messages` list.
- `json.dumps(..., default=str)` when serializing observations.
- Never hardcode or print API keys; ship `.env.example`, never `.env`.
- Register tools by importing the examples module in `build_agent()`.

## Verify before finishing

1. `pip install -e .` (or `uv sync`).
2. `python -m <package>.cli --help` — no ImportError.
3. `python -m <package>.cli --chat` smoke test, and test a no-API-key tool call
   (e.g. `calculate("2+2")`) to prove tool glue works.

## Example

User: "Build a time-bot that answers date/time and calculator questions."

You: scaffold `time-bot/` with tools `current_time`, `time_in_zone(zone, iso)`,
`calculate(expr)`, a `time-bot` CLI, `.env.example`, README; verify imports +
`--help` runs. Report the structure and how to run it.