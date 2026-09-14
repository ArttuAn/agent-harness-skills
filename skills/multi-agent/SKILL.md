---
name: harness-multi-agent
description: "Build a multi-agent orchestrator harness from a spec — role workers with isolated context, structured handoffs, bounded fan-out and honest cost accounting"
---

# Multi-Agent Orchestrator Harness

An orchestrator decomposes a task, dispatches subtasks to role-specific
workers, and synthesizes their results.

A frontier model will write the orchestrator, the worker class and the role
prompts from a spec. What it will not do is **tell you that you probably should
not build this**, bound the fan-out, keep worker contexts genuinely separate,
or stop the synthesis step from discarding the detail the workers were hired to
find. This skill leads with the first one because it is the most valuable.

## Use this when

There is exactly one good reason, and it is **context isolation**: one worker's
50k tokens of logs must not pollute another worker's window. That is real, and
no amount of prompt engineering substitutes for it.

Secondary, weaker reasons:

- Genuinely parallel subtasks where latency matters more than tokens.
- Tools that must be scoped per role — the summarizer should not have the
  delete tool.

**Don't use this when:**

- **Fewer than about five distinct subtasks.** A single agent with good tools
  beats three agents with a coordination problem. This is the default answer.
- Workers need each other's intermediate state. That is one task with a
  confusing topology; the handoffs will cost more than the split saves.
- You want "specialists" for quality. Role prompts on the same model give you
  the same model with a different preamble — real gains come from different
  tools or different context, not different adjectives.
- The result must be reproducible. Fan-out multiplies nondeterminism and makes
  regressions hard to attribute.

**Say the cost out loud before building:** N workers is N× the tokens, plus
orchestration, plus synthesis. A 5-worker run costs roughly 6–8× a single
agent on the same task. If the user has not accepted that, they have not chosen
this pattern — they have been handed it.

## Workflow

1. **Challenge the pattern.** Count the subtasks. If under five, or if workers
   need shared state, recommend `harness-react` and explain why. Build this only
   if the user confirms after hearing the cost.
2. **Get the spec** — subtasks, which tools each role needs, sequential or
   parallel, what the final output looks like.
3. **Scaffold** the layout below.
4. **Write the orchestrator with fan-out and depth bounded.**
5. **Write the Tier 1 tests** — worker failure isolation above all.
6. **Verify**, reporting real token cost.

## The decisions that matter

### 1. Workers get their own context, or there is no point

Each worker is a fresh agent with its own message list, its own system prompt
and its own tool subset. It receives a **task string**, not the orchestrator's
conversation. If you find yourself passing the orchestrator's history into
workers, you have rebuilt a single agent with extra steps.

```python
@dataclass
class Role:
    name: str
    system_prompt: str
    tools: list[str]          # names, filtered from the shared registry
    max_steps: int = 6
```

Scoping tools per role is the cheapest real safety win in this pattern: the
`researcher` gets `search` and `fetch`; the `writer` gets neither.

### 2. Handoffs are structured, not prose

A worker returning a paragraph forces the orchestrator to re-parse it, and
detail is lost at every hop. Return an object:

```python
@dataclass
class WorkerResult:
    role: str
    task: str
    output: str
    ok: bool
    error: str | None = None
    steps_used: int = 0
    evidence: list[str] = field(default_factory=list)   # ids, URLs, file paths
```

`evidence` is the field that saves the pattern. Synthesis compresses `output`;
evidence survives, so the final answer can still point at where a claim came
from.

### 3. Bound fan-out *and* depth

Two runaway dimensions. Bound both, explicitly:

| Bound | Why | Default |
| --- | --- | --- |
| `max_workers` per dispatch | An orchestrator asked to "research thoroughly" will spawn 30 | 5 |
| `max_depth` | Workers spawning workers is exponential | **1** — workers do not spawn |
| `max_steps` per worker | One stuck worker must not hang the run | 6 |

Depth 1 unless the spec truly demands otherwise. Recursive delegation is where
this pattern stops being debuggable.

### 4. A failed worker is data, not an exception

One worker failing must not kill the run. Catch per worker, mark `ok=False`,
and let the orchestrator decide: proceed with partial results, retry once, or
stop. Then make sure the synthesis prompt **knows** a worker failed — otherwise
it writes a confident summary over a hole.

```python
def dispatch(self, role: Role, task: str) -> WorkerResult:
    try:
        output = Worker(role, self.client, self.registry.for_role(role)).run(task)
        return WorkerResult(role.name, task, output, ok=True)
    except Exception as exc:  # noqa: BLE001 - one worker must not end the run
        self._emit("on_worker_error", role=role.name, error=exc)
        return WorkerResult(role.name, task, "", ok=False, error=str(exc))
```

### 5. Parallel only if the tools are safe for it

`ThreadPoolExecutor` is enough — these are I/O-bound HTTP calls, so the GIL is
not the constraint. But check first: shared SQLite connections, a rate-limited
API, or a workspace two workers both write to will break under concurrency.
Sequential is the safe default; parallelize when you have checked.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
    results = list(pool.map(lambda a: self.dispatch(*a), assignments))
```

`pool.map` preserves input order, which keeps runs comparable.

### 6. Synthesis is where quality is lost

The orchestrator sees summaries of summaries. Guard it:

- Give the synthesizer every worker's `output` **and** `evidence`, plus the
  explicit list of which workers failed.
- Tell it to preserve specifics — numbers, names, quotes — and to attribute
  each claim to the worker it came from.
- If two workers disagree, it must surface the disagreement, not average it.
  Silent averaging of contradictory findings is this pattern's worst output.

## Build it

```text
<project>/
├── src/<package>/
│   ├── config.py        # max_workers, max_depth, per-worker max_steps
│   ├── llm.py           # ChatClient + FakeChat + adapter (references/llm-seam.md)
│   ├── roles.py         # Role definitions and the role registry
│   ├── worker.py        # one role + one task -> WorkerResult (a ReAct loop)
│   ├── orchestrator.py  # decompose, dispatch, synthesize
│   ├── tools.py         # shared registry with for_role(role) filtering
│   └── cli.py           # run, --parallel, --max-workers, -v
└── tests/
```

Each worker is a `harness-react` loop — all of `references/failure-modes.md`
applies inside every one of them. Write the loop once and reuse it; N copies of
a loop is N places to forget a guard.

The orchestrator's shape:

```python
def run(self, goal: str) -> str:
    assignments = self.decompose(goal)[: self.config.max_workers]
    self._emit("on_plan", assignments=assignments)

    results = self._dispatch_all(assignments)
    for result in results:
        self._emit("on_worker_done", result=result)

    if not any(r.ok for r in results):
        return "Every worker failed. " + "; ".join(
            f"{r.role}: {r.error}" for r in results
        )
    return self.synthesize(goal, results)
```

Note the all-failed branch: returning a clear account of what went wrong beats
synthesizing an answer out of nothing.

## Failure modes

Beyond `references/failure-modes.md` (which applies inside every worker):

- **Cost explosion.** Unbounded fan-out or depth. Guard: decision 3, and report
  actual token usage at the end so the user sees the bill.
- **Synthesis flattens the finding.** The one number that mattered does not
  survive two rounds of summarizing. Guard: `evidence` fields, and instruct the
  synthesizer to preserve specifics.
- **A failed worker becomes an invisible hole.** Guard: pass failures into the
  synthesis prompt explicitly.
- **Workers duplicating work.** Three researchers, one search, three times.
  Guard: decompose into genuinely disjoint tasks and say so in each task
  string; a shared result cache keyed by tool+args also helps.
- **Deadlock on a shared resource.** Two workers writing one SQLite file.
  Guard: decision 5, or give each worker its own connection.
- **Attribution loss on regression.** Something got worse and you cannot tell
  which worker. Guard: log every worker's task, output and steps under a run id.
- **Orchestrator reasoning about things it cannot see.** It only has summaries;
  asked to verify a detail, it will confabulate. Guard: send it back to a worker
  rather than letting it answer from the summary.

## Required tests

All offline with `FakeChat` and stub roles:

```python
def test_one_worker_failing_does_not_kill_the_run(config):
    orchestrator = Orchestrator(roles=[good_role, exploding_role], config=config)
    output = orchestrator.run("goal")

    assert output                                     # still produced something
    assert any(not r.ok for r in orchestrator.results)


def test_failed_workers_are_named_in_the_synthesis_prompt(config):
    orchestrator = Orchestrator(roles=[good_role, exploding_role], config=config)
    orchestrator.run("goal")

    assert "exploding" in orchestrator.synthesis_prompt


def test_fan_out_is_bounded(config):
    config.max_workers = 3
    orchestrator = Orchestrator(roles=[a_role] * 10, config=config)
    orchestrator.run("goal")

    assert len(orchestrator.results) == 3


def test_workers_do_not_share_context(config):
    orchestrator = Orchestrator(roles=[role_a, role_b], config=config)
    orchestrator.run("goal")

    a, b = orchestrator.workers
    assert not set(m["content"] for m in a.messages) & set(m["content"] for m in b.messages)


def test_role_tool_scoping_is_enforced(config):
    worker = Worker(role=Role("writer", "...", tools=["draft"]), registry=full_registry)
    assert "delete" not in worker.executor.names()


def test_all_workers_failing_reports_instead_of_synthesizing(config):
    orchestrator = Orchestrator(roles=[exploding_role, exploding_role], config=config)
    output = orchestrator.run("goal")

    assert "failed" in output.lower()
    assert orchestrator.synthesis_prompt is None       # never called the model
```

Plus the ReAct loop tests from `harness-react` for the worker loop.

## Verify

1. **Tier 0** — install, import, `--help`.
2. **Tier 1** — `pytest -q`, offline. The failure-isolation tests must pass.
3. **Sequential first.** Run with `--parallel` off and confirm correctness
   before adding concurrency. A race condition looks like a model quality
   problem and will waste a day.
4. **Tier 2** — one real run with `-v`. Then do the honest comparison: run the
   same goal through a single ReAct agent. **Report both results and both
   costs.** If the single agent did as well, say so — that is the most useful
   finding this verification can produce.

Report worker count, total tokens, wall time, and any worker that failed.
