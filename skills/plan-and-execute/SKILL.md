---
name: harness-plan-and-execute
description: "Build a plan-and-execute agent harness from a spec — planner, executor and verifier with bounded replanning and a plan you can inspect"
---

# Plan-and-Execute Agent Harness

The agent writes a plan first, then executes it step by step, verifying each
step and replanning when one fails. It buys coherence on long tasks: the plan
survives even when the model's attention does not.

It also buys a second control loop, and that is where it goes wrong. A frontier
model will write the planner/executor/verifier trio from a spec without help.
What it will not do is **bound the replanning, keep the plan as inspectable
data, and stop the verifier from rubber-stamping**. Those decide whether this
is better than ReAct or just slower.

## Use this when

- The task has many steps and a model running ReAct loses the thread halfway.
- The steps are mostly knowable up front — the environment does not rewrite the
  problem after step 2.
- A human may want to see, edit or approve the plan before it runs.

**Don't use this when:**

- The task is short, or each observation changes what to do next → `harness-react`.
  A plan made before any observation is a guess, and replanning every step means
  you have paid for planning and gained nothing.
- You cannot tell whether a step succeeded → the verifier is decorative; use
  ReAct and check the final answer instead.
- The steps are genuinely independent and parallelizable → `harness-multi-agent`.

## Workflow

1. **Get the spec** — the task domain, the tools, and crucially **how success
   is checked per step**. If there is no per-step check, say so and recommend
   ReAct instead of building a verifier that always returns "looks fine".
2. **Scaffold** the layout below.
3. **Model the plan as data**, not prose (decision 1).
4. **Write the three stages plus the loop**, with both budgets bounded.
5. **Write the Tier 1 tests** — especially the replan-budget test.
6. **Verify** and report honestly.

## The decisions that matter

### 1. The plan is data, not prose

A plan the model wrote as a paragraph cannot be resumed, edited, diffed, or
partially retried. Make it a typed object from the start:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    id: int
    description: str          # what to do, in one sentence
    success_check: str        # how the verifier will know it worked
    status: Status = Status.PENDING
    result: str | None = None
    attempts: int = 0


@dataclass
class Plan:
    goal: str
    steps: list[Step] = field(default_factory=list)
    revision: int = 0         # bumped on every replan

    def next_pending(self) -> Step | None:
        return next((s for s in self.steps if s.status is Status.PENDING), None)

    def is_complete(self) -> bool:
        return all(s.status in (Status.DONE, Status.SKIPPED) for s in self.steps)
```

`success_check` is the field people leave out, and it is the one that makes the
verifier possible. Require the planner to fill it for every step.

Persist the plan as JSON after every state change. A crashed run you can resume
is worth more than a slightly cleaner data model.

### 2. Two budgets, not one

This pattern has a nested loop and therefore two ways to run forever:

| Budget | Bounds | Typical |
| --- | --- | --- |
| `max_steps_per_step` | Model calls while executing **one** step | 4–6 |
| `max_replans` | Whole-plan revisions | **2–3** |

`max_replans` is the one that matters and the one that gets forgotten. Without
it, a step that can never succeed produces plan revision 47. Set it low: if two
replans have not fixed it, a third will not either. When it runs out, stop and
report the partial plan with the failing step — a partial result plus a clear
blocker beats an infinite loop, and it is what a human can act on.

### 3. Replan on failure, not on surprise

Replanning on every imperfect step collapses this pattern into an expensive
ReAct. Replan only when a step **failed and retrying it is pointless** — a
missing file, a wrong assumption baked into later steps. For a transient
failure, retry the step (`attempts`, cap at 2).

Distinguish the two explicitly in the verifier's output. "Failed" and "failed
permanently" are different signals.

### 4. The verifier must be able to say no

A verifier that is the same model, given the same context, asked "did that
work?" says yes. Make it cheap for it to say no:

- Give it the step's `success_check` **and nothing about the agent's reasoning** —
  judge the result, not the story.
- Prefer a mechanical check where one exists: exit code, file exists, tests
  pass, JSON parses. A real check beats an LLM opinion every time.
- Require a structured verdict, not prose: `{"passed": bool, "reason": str,
  "permanent": bool}`.

If you find yourself writing an LLM verifier for something a `Path.exists()`
would settle, write the `Path.exists()`.

### 5. The planner is blind

The planner writes the plan before any step has run — it sees the goal and the
tool list, nothing else. So:

- Give it the tool descriptions in the prompt; it cannot plan around tools it
  does not know about.
- Ask for 3–7 steps. Fewer means the executor is doing unplanned work; more
  means the plan is a to-do list and will be stale by step 4.
- On replan, give it the **failed step and its reason**, plus completed steps'
  results. Without that it rewrites the same broken plan.

## Build it

```text
<project>/
├── pyproject.toml
├── .env.example
├── src/<package>/
│   ├── config.py        # both budgets live here
│   ├── llm.py           # ChatClient ABC + FakeChat + adapter (references/llm-seam.md)
│   ├── plan.py          # Step, Plan, Status, JSON persistence
│   ├── planner.py       # goal + tools -> Plan
│   ├── executor.py      # one Step -> result (a small ReAct loop)
│   ├── verifier.py      # Step + result -> Verdict
│   ├── agent.py         # the outer loop and both budgets
│   └── cli.py           # run, --plan-only, --resume, -v
└── tests/
```

The executor is a ReAct loop scoped to one step — reuse `harness-react`'s loop
and guards rather than writing a second one. All of
`references/failure-modes.md` applies inside it.

The outer loop is the part worth writing carefully:

```python
def run(self, goal: str) -> Plan:
    plan = self.planner.make_plan(goal)
    self._emit("on_plan", plan=plan)

    for replan in range(self.config.max_replans + 1):
        while (step := plan.next_pending()) is not None:
            step.status = Status.RUNNING
            step.attempts += 1
            self._emit("on_step_start", step=step)

            result = self.executor.run(plan, step)
            verdict = self.verifier.check(step, result)
            self._emit("on_step_end", step=step, verdict=verdict)

            if verdict.passed:
                step.status, step.result = Status.DONE, result
                self.store.save(plan)
                continue

            # Transient failure: retry the same step before touching the plan.
            if not verdict.permanent and step.attempts < self.config.max_attempts:
                step.status = Status.PENDING
                self.store.save(plan)
                continue

            step.status, step.result = Status.FAILED, verdict.reason
            self.store.save(plan)
            break

        if plan.is_complete():
            return plan
        if replan == self.config.max_replans:
            break

        plan = self.planner.replan(plan, failed=self._failed_step(plan))
        plan.revision += 1
        self._emit("on_replan", plan=plan)
        self.store.save(plan)

    # Out of replans: return the partial plan. The caller reports what got
    # done and which step blocked — never raise into the user's face.
    return plan
```

Note what it returns: the `Plan`, not a string. The plan carries what
succeeded, what failed and why. The CLI decides how to print it.

## Failure modes

`references/failure-modes.md` applies in full to the executor. These are
specific to planning:

- **Runaway replanning.** The headline failure. Bounded by `max_replans`;
  test it (below).
- **Plan drift.** The executor quietly solves a different problem than the step
  describes, the verifier checks *that*, and the goal is never met. Guard: the
  verifier sees the step's `success_check`, not the executor's transcript.
- **Steps that cannot fail.** "Analyze the data", "consider the options" — no
  observable outcome, so the verifier always passes. Guard: reject steps whose
  `success_check` is not observable, at plan time.
- **Later steps depending on a skipped one.** Skipping step 3 and running step
  4 produces confident garbage. Guard: when a step fails permanently, mark
  dependent steps `SKIPPED` rather than executing them.
- **The plan outgrows its context.** Long plans plus every step's result will
  overflow. Guard: pass the executor only the goal, the current step, and a
  short summary of completed steps — not every result verbatim.
- **Resume without state.** A crash at step 5 of 7 restarts from zero. Guard:
  persist after every status change; `--resume` loads the saved plan.

## Required tests

Offline with `FakeChat`, plus a fake verifier. The replan-budget test is the
one that matters most — it cannot be written against a live model.

```python
def test_plan_persists_and_resumes(tmp_path, config):
    plan = Plan(goal="g", steps=[Step(1, "do a", "a exists"), Step(2, "do b", "b exists")])
    plan.steps[0].status = Status.DONE
    store = PlanStore(tmp_path / "plan.json")
    store.save(plan)

    loaded = store.load()
    assert loaded.next_pending().id == 2


def test_replans_are_bounded(config):
    config.max_replans = 2
    agent = Agent(planner=AlwaysBadPlanner(), executor=NoopExecutor(),
                  verifier=AlwaysFails(), config=config)

    plan = agent.run("impossible goal")
    assert agent.planner.replan_calls == 2       # not 3, not forever
    assert any(s.status is Status.FAILED for s in plan.steps)


def test_transient_failure_retries_the_step_not_the_plan(config):
    verifier = FailsOnce(permanent=False)
    agent = Agent(planner=OneStepPlanner(), executor=NoopExecutor(),
                  verifier=verifier, config=config)

    plan = agent.run("g")
    assert plan.steps[0].attempts == 2
    assert agent.planner.replan_calls == 0


def test_permanent_failure_skips_dependent_steps(config):
    agent = Agent(planner=TwoStepPlanner(), executor=NoopExecutor(),
                  verifier=AlwaysFails(permanent=True), config=config)

    plan = agent.run("g")
    assert plan.steps[1].status is Status.SKIPPED


def test_partial_plan_is_returned_not_raised(config):
    agent = Agent(planner=OneStepPlanner(), executor=NoopExecutor(),
                  verifier=AlwaysFails(permanent=True), config=config)

    plan = agent.run("g")            # does not raise
    assert not plan.is_complete()
```

Plus the ReAct executor's own tests from `harness-react` — the inner loop needs
every guard the outer loop does.

## Verify

1. **Tier 0** — install, import, `--help`.
2. **Tier 1** — `pytest -q`, offline, all of the above.
3. **Plan-only run** — `<command> "<goal>" --plan-only`. Read the plan. Every
   step should have an observable `success_check`. If any is "analyze" or
   "consider", the planner prompt needs work; fix it before Tier 2.
4. **Tier 2** — one real run with `-v`. Confirm a step actually failed and
   replanned at least once, or force it by pointing a tool at something
   missing. A run where nothing fails proves only the happy path.

Report what ran. If replanning was never exercised, say so — it is the half of
this pattern that ReAct does not have.
