---
name: harness-self-evolving
description: "Build a self-evolving agent harness from a spec — run, evaluate, reflect, apply, gated by a test suite the agent cannot edit and a rollback it cannot skip"
---

# Self-Evolving Agent Harness

The harness runs a task set, scores itself, proposes changes to its own prompts
and tools, applies them, and repeats. Each pass is a *generation*.

A frontier model will write the evolve loop from a spec. What it will not do is
**make the gate something the agent cannot reach**, and that single property is
the difference between a system that improves and one that learns to cheat. An
optimizer with write access to its own scoring function optimizes the scoring
function. Build the containment first; the loop is the easy part.

## Use this when

- You have a **fixed, external evaluation set** with known-good answers, or a
  test suite the agent does not author.
- The harness runs the same class of task repeatedly, so an improvement
  compounds.
- Someone will review the diffs. This pattern produces changes to your system;
  unreviewed, it produces drift.

**Don't use this when:**

- **You cannot write the evaluation set first.** This is disqualifying. No
  fixed metric means no gradient, and the loop will wander while producing
  confident generation logs.
- The task varies too much for one metric to represent quality.
- Nobody will read the diffs. → `harness-reflexion` gives most of the benefit
  with per-task memory and no permanent mutation.

## The containment comes first

Write these down before any code, and state them to the user:

1. **The agent may not edit the evaluation set, the scoring code, or the test
   suite.** Enforce with a path check in the applier, not a prompt instruction.
2. **The agent may not edit the applier, the gate, or this containment.**
3. **Every generation is a git commit**, so every change is reviewable and
   revertible by construction.
4. **A generation that does not improve the score is reverted**, automatically.
5. **Generations are bounded.** A default of 5, not "until it stops improving".

If any of these cannot hold in the target environment, say so plainly and
recommend `harness-reflexion` instead. A self-evolving harness without an
immutable gate is not a weaker version of this pattern — it is a different,
worse thing that looks identical in the logs.

## The decisions that matter

### 1. What is allowed to change

Narrow surface, widening only with evidence:

| Surface | Evolve it? | Why |
| --- | --- | --- |
| System prompts, tool descriptions | **Yes** | Highest value per risk; text only, trivially revertible |
| Few-shot examples | **Yes** | Easy to diff and review |
| Tool *parameters* (limits, thresholds) | Cautiously | Bounded ranges only, validated on apply |
| New tools | Rarely | Arbitrary new code; needs review before it runs |
| The loop itself | **No** | The thing being measured must stay fixed |
| Evaluator, tests, gate | **Never** | See containment |

Most of the available gain is in the first two rows. Start there and probably
stay there.

### 2. The gate is mechanical and runs after every apply

```python
def gate(workspace, baseline_score: float, new_score: float) -> tuple[bool, str]:
    """Accept a generation only if tests pass AND the score improved."""
    tests = run_command(workspace, ["pytest", "-q"], timeout=300)
    if "exit=0" not in tests:
        return False, f"tests failed:\n{tests}"
    if new_score <= baseline_score:
        return False, f"score did not improve ({new_score:.3f} <= {baseline_score:.3f})"
    return True, f"accepted: {baseline_score:.3f} -> {new_score:.3f}"
```

Both conditions. Tests alone let quality slide while staying green; score alone
lets the agent trade correctness for metric.

### 3. Rollback is `git`, and it is not optional

```python
def apply_generation(repo, generation: int, changes) -> bool:
    before = repo.rev_parse("HEAD")
    write(changes)
    repo.commit(f"gen {generation}: {summarize(changes)}")

    accepted, reason = gate(...)
    if not accepted:
        repo.reset_hard(before)          # the whole point
        log(generation, "reverted", reason)
        return False
    log(generation, "accepted", reason)
    return True
```

Refuse to start on a dirty tree. Refuse to start outside a git repo — offer to
`git init` instead. Without rollback this pattern cannot be run unattended, and
running it attended defeats the purpose.

### 4. Hold out an evaluation set the loop never sees

Scoring and optimizing on the same set is overfitting with extra steps: the
score climbs while real performance does not.

- **Dev set** — the loop scores against this and optimizes it.
- **Holdout set** — scored only at the *end*, or every N generations, and never
  used to choose a change.

If the dev score rises and the holdout score does not, the harness is fitting
the dev set. Report both numbers every run; a single rising number is the
easiest thing in this pattern to mistake for progress.

### 5. Assume the metric will be gamed

Not because the model is adversarial, but because optimizers find the cheapest
path. Seen in practice: prompts that make answers longer when the rubric
rewards detail; tools that return cached constants when the metric is latency;
"I cannot answer" when wrong answers are penalized more than refusals.

Guard: score more than one dimension (correctness *and* cost *and* refusal
rate), and read the actual diffs. A generation whose only change is "be more
thorough" is a smell.

### 6. Append-only, and cap it

Prompts grow every generation until they are 4k tokens of accreted advice.
Cap the evolved prompt's length and require the reflector to **replace**
guidance rather than stack it. Log the prompt size per generation — steady
growth with a flat score means it is padding.

## Build it

```text
<project>/
├── src/<package>/
│   ├── config.py        # max_generations, score threshold, protected paths
│   ├── llm.py           # ChatClient + FakeChat + adapter (references/llm-seam.md)
│   ├── agent.py         # the harness being evolved (a harness-react loop)
│   ├── evalset.py       # dev + holdout task sets; loaded read-only
│   ├── scorer.py        # task -> score; mechanical wherever possible
│   ├── reflector.py     # run logs -> structured change proposals
│   ├── applier.py       # validates and writes changes; enforces protected paths
│   ├── gate.py          # tests + score comparison
│   ├── evolve.py        # the generation loop
│   └── cli.py           # run, evolve --generations N, --dry-run
├── prompts/             # the evolvable surface, one file per prompt
├── generations/         # per-generation logs, diffs, scores
└── tests/               # PROTECTED — the applier refuses to touch this
```

Protected paths, enforced in code rather than requested in a prompt:

```python
PROTECTED = ("tests/", "src/<package>/gate.py", "src/<package>/applier.py",
             "src/<package>/scorer.py", "src/<package>/evalset.py", "evalset/")


def validate_change(path: str) -> None:
    normalized = str(Path(path).as_posix())
    if any(normalized.startswith(p) for p in PROTECTED):
        raise PermissionError(f"refusing to modify protected path: {path}")
```

Use the `Workspace.resolve` confinement from `harness-code-agent` as well —
a proposal writing to `../../.claude/` is exactly the case that check exists
for.

The generation loop:

```python
def evolve(self, generations: int) -> list[Generation]:
    baseline = self.score(self.evalset.dev)
    holdout = self.score(self.evalset.holdout)
    self._log(0, baseline, holdout, "baseline")
    history = []

    for number in range(1, generations + 1):
        runs = self.run_all(self.evalset.dev)          # transcripts + scores
        proposals = self.reflector.propose(runs)       # structured changes
        if not proposals:
            break                                       # nothing left to try

        applied = self.applier.apply(proposals)         # raises on protected paths
        new_score = self.score(self.evalset.dev)
        accepted, reason = self.gate(baseline, new_score)

        if accepted:
            baseline = new_score
        else:
            self.repo.reset_hard(self.repo.head_before(number))

        holdout = self.score(self.evalset.holdout)
        history.append(Generation(number, new_score, holdout, accepted, reason, applied))
        self._log(number, new_score, holdout, reason)

    return history
```

Report dev and holdout side by side. Diverging lines are the finding.

## Failure modes

Beyond `references/failure-modes.md`:

- **The agent edits its own gate.** The failure that invalidates everything.
  Guard: `PROTECTED` enforced in the applier, and a test that asserts a proposal
  touching `tests/` raises.
- **Metric gaming.** Guard: multi-dimensional scoring, read the diffs.
- **Dev/holdout divergence.** Guard: report both, always.
- **Prompt bloat.** Guard: cap length, log size per generation.
- **A regression ships because tests were green.** Guard: the gate requires
  improvement *and* green, never either alone.
- **Unbounded generations.** Guard: `max_generations`, default 5.
- **Non-reproducible scoring.** Temperature > 0 in the scorer means the score
  moves on its own and the gate accepts noise. Guard: `temperature=0` for all
  evaluation runs; score each task more than once if it is still noisy.
- **Evolving on a dirty tree.** The user's uncommitted work gets committed as
  "gen 1" and reverted away. Guard: refuse to start.

## Required tests

These are the tests the agent must not be able to edit — which makes them the
most important in the repo:

```python
def test_proposal_touching_tests_is_refused():
    with pytest.raises(PermissionError):
        validate_change("tests/test_scorer.py")


def test_proposal_touching_the_gate_is_refused():
    with pytest.raises(PermissionError):
        validate_change("src/pkg/gate.py")


def test_proposal_escaping_the_repo_is_refused(workspace):
    with pytest.raises(PermissionError):
        workspace.resolve("../../.claude/settings.json")


def test_failing_tests_reject_the_generation():
    accepted, reason = gate(FailingTests(), baseline_score=0.5, new_score=0.9)
    assert not accepted and "tests failed" in reason


def test_flat_score_rejects_the_generation():
    accepted, reason = gate(PassingTests(), baseline_score=0.8, new_score=0.8)
    assert not accepted and "did not improve" in reason


def test_rejected_generation_is_rolled_back(repo, config):
    before = repo.rev_parse("HEAD")
    Evolver(repo=repo, gate=AlwaysRejects(), config=config).evolve(generations=1)
    assert repo.rev_parse("HEAD") == before


def test_generations_are_bounded(config):
    config.max_generations = 3
    history = Evolver(gate=AlwaysAccepts(), config=config).evolve(generations=3)
    assert len(history) == 3


def test_holdout_is_never_used_to_accept(config):
    evolver = Evolver(gate=RecordingGate(), config=config)
    evolver.evolve(generations=1)
    assert evolver.gate.saw_only_dev_scores
```

## Verify

1. **Tier 0** — install, import, `--help`.
2. **Tier 1** — `pytest -q`. The containment tests must pass. If any is
   skipped, stop: the harness is not safe to run unattended.
3. **Dry run** — `evolve --generations 1 --dry-run`. Read the proposed diff.
   Nothing should touch protected paths. This is a human checkpoint; do not
   automate past it on the first run.
4. **One real generation** — confirm a commit was made, the gate ran, and a
   rejected generation actually reverted (force one by inverting the gate).
5. **Tier 2** — 3 generations; report dev and holdout side by side.

Report the score trajectory for both sets, which generations were accepted and
why the rest were rejected. A run where every generation was accepted deserves
suspicion, not celebration — check whether the metric is being gamed.
