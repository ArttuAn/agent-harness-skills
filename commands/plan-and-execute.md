# Build a Plan-and-Execute Agent Harness

Given the user's spec, build a complete, runnable Python project implementing
the Plan-and-Execute pattern: plan → execute → verify → (replan) until done.
Ask for a spec if none given (domain + example tools, target folder; propose
3 tools if vague). Full inline reference implementations live in
`skills/plan-and-execute/SKILL.md` — use them, adapted to the spec.

## The pattern

1. **Planner** LLM turns the task into an ordered step list.
2. **Executor** runs each step through a tool registry and captures the result.
3. **Verifier/checkpoint** checks each step's output against its criteria.
4. **Replan** regenerates the remaining steps if one fails or the plan goes stale.

## Layout

```
<project>/
├── requirements.txt   # openai, python-dotenv (only)
├── .env.example       # OPENAI_API_KEY, PLANNER_MODEL, VERIFIER_MODEL, OBJECTIVE, MAX_ATTEMPTS
├── README.md
├── plan.py            # Step + Plan data model (pending/running/succeeded/failed)
├── config.py          # env/.env loading, OpenAI client, JSON extraction
├── planner.py         # LLM → ordered steps (goal + verification criteria); local fallback
├── executor.py        # ToolRegistry + Executor: run one step, capture result
├── verify.py          # heuristic + LLM verifier: output vs criteria
├── tools.py           # example domain tools + make_registry()
├── agent.py           # Agent loop: plan → execute → verify → replan
└── main.py            # argparse CLI entry point
```

## Core points

- Steps carry `goal`, `verification_criteria`, optional `tool`; status is a
  pending → running → succeeded | failed state machine.
- Tools are plain `(objective, goal) -> str` callables; executors catch tool
  errors so a crash becomes a failed step, never a dead loop.
- The verifier gates progress: output that fails the criteria = failed step.
- `replan` keeps succeeded steps and regenerates the rest; `--max-attempts`
  bounds retries (default 3).
- No `OPENAI_API_KEY` → deterministic local planner + heuristic verifier so the
  demo runs entirely offline.

## Verify before finishing

1. `python -m py_compile plan.py config.py planner.py executor.py verify.py tools.py agent.py main.py`.
2. `python main.py "prepare a weekly status summary"` — runs offline (local
   fallback), final plan JSON prints with all steps `succeeded`.
3. With a key set, `PLANNER_MODEL`/`VERIFIER_MODEL` drive LLM planning + judging.
4. Fail path: a broken tool → step `failed` → replan → stop cleanly at
   `--max-attempts` (remaining steps marked `failed`).

Report the output tree and the exact run command.