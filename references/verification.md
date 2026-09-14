# Verification contract

"It imports and `--help` renders" proves the packaging works. It says nothing
about whether the agent loop terminates, recovers from a tool error, or stops
repeating itself. A generated harness is not done until it passes all three
tiers below.

Tiers 0 and 1 need **no API key and no network**. That is the point: they run
in CI, on a plane, and in the seconds after the agent finishes writing code.

## Tier 0 — it is real code

```bash
uv pip install -e ".[dev]"     # or: pip install -e ".[dev]"
python -c "import <package>"   # imports clean
<command> --help               # CLI renders
```

Catches: bad packaging, circular imports, a missing `__init__.py`, an entry
point pointing at a function that does not exist.

## Tier 1 — the loop behaves (offline, with `FakeChat`)

This is the tier that matters and the one that is usually missing. Every test
below is deterministic and runs in milliseconds. See `llm-seam.md` for the
fake, and `failure-modes.md` for why each one exists.

| Test | Asserts |
| --- | --- |
| `test_plain_answer_returns_immediately` | One model call, no tool calls, answer returned |
| `test_tool_result_is_fed_back` | Tool output appears as a `tool` message before the next call |
| `test_every_tool_call_gets_a_result` | N calls in one turn produce N result messages |
| `test_tool_error_is_reported_not_raised` | A raising tool yields `Error: ...` text; the run continues |
| `test_unknown_tool_is_reported_not_raised` | Names the available tools back to the model |
| `test_arguments_are_coerced` | `"3"` → `3`; `{"type": "integer"}` → default; unknown key dropped |
| `test_repeated_call_is_not_re_run` | Second identical call returns cache + a nudge |
| `test_budget_forces_a_toolless_answer` | On exhaustion the final request has `tools=None` |
| `test_empty_response_is_not_a_successful_answer` | `""` never returned as success |
| `test_conversation_persists_across_turns` | Roles in order; `reset()` clears |

Run them:

```bash
uv run pytest -q
```

A harness that passes Tier 1 has demonstrated every guard in the minimum set.
A harness that skips Tier 1 has, in practice, not implemented them — the
guards are invisible in normal operation and only appear under failure.

## Tier 2 — one real round trip

Exactly one end-to-end run against a live endpoint, to prove the adapter
speaks the provider's dialect correctly:

```bash
<command> "<a question your example tools can actually answer>" -v
```

Check, by eye:

- A tool was called, and its arguments were sensible.
- The result came back and was used in the answer.
- The run ended on a plain-text answer, not the step budget.

If the endpoint is local, also confirm the model is loaded and fits:

```bash
ollama ps          # is it resident, and at what context size?
free -g            # is there headroom, or is it about to be OOM-killed?
```

## What to tell the user when you finish

Report what you actually ran, not what you intended to run. If Tier 2 was
skipped because there was no key, say so explicitly — do not imply an
end-to-end run happened. State the numbers: how many tests, which tiers, what
the smoke question was and what came back.
