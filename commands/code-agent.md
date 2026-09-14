# Build a Code Agent Harness

Given the user's spec, build a sandboxed Code Agent that edits files and runs
commands inside a workspace directory. Ask for a spec if none given.

## What to build

Scaffold a `src/`-layout Python project with `pyproject.toml`. Use only `openai`
+ stdlib (plus `python-dotenv` optionally). The agent has six tools bound to a
workspace dir, a safety blocklist, and a coding loop.

```
code-agent/
├── pyproject.toml
├── .env.example
└── src/code_agent/
    ├── __init__.py
    ├── workspace.py   # path isolation: resolve() canonicalizes + blocks traversal
    ├── tools.py       # @tool registry: read_file, write_file, edit_file, list_dir, run_command, grep
    ├── safety.py      # blocked command patterns (sudo, rm -rf /, mkfs, curl|bash, etc.)
    ├── agent.py       # coding loop: LLM plans → tool call → observe → repeat
    ├── config.py      # env/.env: AGENT_API_KEY, AGENT_WORKSPACE, AGENT_COMMAND_TIMEOUT
    └── cli.py         # argparse: task arg, --workspace, --model, --max-steps, --timeout
```

## Key requirements

1. **`workspace.py`**: `Workspace(base_dir)` class. Every file op goes through
   `resolve(path)` which does `Path.resolve()` and checks `relative_to(base)`.
   Methods: `read_file`, `write_file`, `list_dir`, `grep`, `resolve`.

2. **`tools.py`**: `ToolRegistry` + `@tool` decorator with type-hint schema
   inference. `register_workspace_tools(registry, ws, timeout)` binds the six
   tools to a workspace instance. `run_command` uses `subprocess.run(shell=True,
   cwd=ws.base, capture_output=True, timeout=N)`.

3. **`safety.py`**: `check_command(cmd) -> str|None`. Blocklist of regex patterns:
   `sudo`, `rm -rf /`, `mkfs`, `chmod 777 /`, `curl|bash`, `shutdown`,
   `ffserver`, `systemctl`, `kill -9 1`, fork bombs, etc. `run_command` calls
   this before subprocess.

4. **`agent.py`**: `CodeAgentLoop` with system prompt that tells the model to
   reply `{thought, action:{name,args}}` or `{thought, answer}`. After each tool
   call, observation is fed back. System prompt nudges the agent to run tests
   after edits. `temp=0`, fenced JSON tolerance, 3 retries on bad JSON.

5. **`config.py`**: `AGENT_API_KEY`, `AGENT_MODEL`, `AGENT_BASE_URL`,
   `AGENT_WORKSPACE`, `AGENT_COMMAND_TIMEOUT`, `AGENT_MAX_STEPS` from env/.env.

6. **`cli.py`**: `code-agent "task description" --workspace ./my-project --model
   gpt-4o-mini --max-steps 20`. Exit 2 if no API key.

## Conventions

- Path traversal protection is the top priority — no file op without `resolve()`.
- Tool errors return strings, never raise. Loop must not crash on tool failure.
- `json.dumps(..., default=str)` for observations.
- `temp=0` for deterministic tool calls.
- Ship `.env.example`, never `.env`. Never print API keys.

## Verify

1. `pip install -e .` — no errors.
2. `code-agent --help` — renders, no ImportError.
3. `code-agent -w /tmp/test "create hello.py"` — file appears in workspace.
4. `code-agent -w /tmp/test "sudo rm -rf /"` — blocked.
