# Build a Multi-Agent Orchestrator Harness

Given the user's spec, build a multi-agent orchestrator: an LLM decomposes a
task, routes subtasks to specialized worker roles (each with its own tools),
workers run in parallel or sequentially, and the orchestrator synthesizes a final
answer. Ask for a spec if they didn't give one (domain + what roles + what tools
per role).

## What to build

Scaffold a `src/`-layout project with `pyproject.toml` into the requested
directory. Use the `openai` library; no other heavy deps.

Structure:

```
<project>/
├── pyproject.toml
├── .env.example
├── README.md
└── src/<package>/
    ├── __init__.py
    ├── config.py          # env config: model, base_url, api_key, max_workers
    ├── roles.py           # Role dataclass: name, system_prompt, allowed_tools
    ├── worker.py          # single worker: runs subtask with filtered tools → Deliverable
    ├── orchestrator.py    # decompose → route → dispatch (ThreadPoolExecutor) → synthesize
    ├── tools.py           # @tool registry + for_role() filtering
    ├── cli.py             # argparse entry: prompt arg, --chat, --model, --max-workers
    └── tools/
        ├── __init__.py
        └── examples.py    # 4-5 domain tools
```

## Core pieces

1. **roles.py**: frozen `@dataclass Role(name, system_prompt, allowed_tools)`.
   3-4 example roles relevant to the spec.
2. **worker.py**: `run_worker(client, model, role, task, registry)` — builds
   messages with role's system prompt, calls LLM with filtered tools (OpenAI
   tool-calling API), loops until the model emits a final JSON deliverable
   `{content: markdown, result: {…}}` or step budget hits. Tolerate fenced JSON.
3. **orchestrator.py**: `Orchestrator` class with `run(task)` method:
   - `_decompose(task)` → LLM returns `[{role, task}, …]`
   - `_dispatch(subtasks)` → `ThreadPoolExecutor` (or sequential for ≤1 task),
     each calls `run_worker` with the assigned role
   - `_synthesize(task, deliverables)` → LLM combines all outputs into final answer
4. **tools.py**: `ToolRegistry` with `for_role(role)` returning a filtered
   copy; `@tool` decorator registers into module-global registry.
5. **Config**: env vars `AGENT_MODEL`, `AGENT_BASE_URL`, `AGENT_API_KEY`,
   `AGENT_MAX_WORKERS`, `AGENT_MAX_STEPS`. CLI: `<command> "task"`, `--chat`,
   `--model`, `--max-workers`.

## Dependencies

- `openai>=1.0.0`
- `python-dotenv>=1.0.0`
- stdlib only otherwise (`concurrent.futures`, `json`, `dataclasses`, etc.)

## Example tools (adapt to spec)

- `calculate(expression)` — safe `ast`-based arithmetic
- `current_time()` — ISO-8601 timestamp
- `search_web(query)` — stub or wire a real API
- `read_file(path)` / `list_files(dir)` — filesystem helpers
- `summarize(text)` — truncate or LLM-based
- `format_table(rows)` — list-of-dicts → markdown table

## Conventions

- temp=0 for deterministic routing.
- `json.dumps(default=str)` for tool results.
- Workers return `Deliverable(role, content, result)` — never crash on tool errors.
- Never print API keys; ship `.env.example` only.
- Orchestrator falls back to sequential dispatch for single subtask.

## Verify before finishing

1. `pip install -e .`
2. `python -m <package>.cli --help` — no ImportError.
3. `python -m <package>.cli "do X and Y"` — confirm decomposition + worker dispatch + synthesis.
4. `python -m <package>.cli --chat` smoke test.
