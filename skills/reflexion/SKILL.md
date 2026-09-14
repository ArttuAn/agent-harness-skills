---
name: harness-reflexion
description: "Build a Reflexion agent harness from a spec — act, evaluate, reflect, retry, with episodic memory of past mistakes and an evaluator that can actually fail"
---

# Reflexion Agent Harness

The agent attempts a task, an evaluator scores the attempt, and on failure the
agent writes a short lesson about *why* it failed and retries with that lesson
in context (Shinn et al., 2023).

A frontier model will write the act → evaluate → reflect → retry loop from a
spec. What it will not do is **stop the evaluator from rubber-stamping, keep
the reflection a transferable lesson rather than a retelling, or prevent a
wrong lesson from poisoning every future run**. Without those three, this is
ReAct that costs three times as much.

## Use this when

- Attempts can be **scored by something other than the model that wrote them**:
  a test suite, an exact answer, a schema check, a compile step, a human.
- Failures are the kind an agent could diagnose from its own transcript — a
  wrong tool, a skipped constraint, a misread requirement.
- The task is worth 2–3× the tokens.

**Don't use this when:**

- **There is no real evaluator.** This is the disqualifying condition. If the
  only judge is the same model given the same context, it will approve its own
  work, and you have bought retries that change nothing. Use `harness-react`.
- Failures are environmental — a flaky API, a rate limit. Reflection cannot fix
  what reflection cannot see; retry with backoff instead.
- The task is cheap to redo from scratch. Reflection overhead beats it.

## Workflow

1. **Establish the evaluator first.** Before any other design question: what,
   other than the model, decides whether an attempt succeeded? If the answer is
   "the model", stop and recommend ReAct.
2. **Get the spec** — task domain, tools, the success signal, retry budget.
3. **Scaffold** the layout below.
4. **Write the loop**, with memory scoped deliberately (decision 3).
5. **Write the Tier 1 tests** — especially that a failing evaluator actually
   drives a retry and that the lesson reaches the next attempt.
6. **Verify**, including one run that genuinely fails first.

## The decisions that matter

### 1. The evaluator ranked by trustworthiness

Use the highest one available. Do not skip to the bottom because it is easiest:

| Evaluator | Trust | Use when |
| --- | --- | --- |
| Test suite / exit code | Highest | Code tasks — always prefer this |
| Exact or fuzzy match against ground truth | High | Benchmarks, extraction, QA with answers |
| Schema / constraint check | High | Structured output |
| Programmatic rubric (length, required fields, citations present) | Medium | Generation with hard requirements |
| A *different* model, given only the output | Low | Last resort |
| The same model that produced the answer | **None** | Never |

Two rules for the LLM-judge case, if you are forced into it:

- Give the judge the **output and the criteria only** — never the agent's
  reasoning. Seeing the story makes it sympathetic to the story.
- Force a structured verdict and make failure cheap to express:
  `{"passed": bool, "score": float, "reason": str, "fix_hint": str}`.

### 2. A reflection is a lesson, not a retelling

The natural output of "why did that fail?" is a summary of what happened. That
is worthless in the next attempt's context. Demand a **transferable rule**:

```text
Bad  (retelling):  "I called search_users with the wrong ID and got an error."
Good (lesson):     "search_users takes an email, not a numeric ID. Look the
                    email up with find_account first."
```

Constrain it hard in the prompt: one to three sentences, imperative, naming the
specific tool or constraint. Reject anything that merely narrates.

```python
REFLECT_PROMPT = """The attempt below failed.

Task: {task}
Attempt: {attempt}
Why it failed: {reason}

Write 1-3 sentences of instruction for your next attempt. State what to do
differently, naming the specific tool, argument or constraint involved.
Do not narrate what happened. Do not apologize."""
```

### 3. Episodic or persistent — pick deliberately

| | Episodic | Persistent |
| --- | --- | --- |
| Scope | Lessons live for one task, then vanish | Lessons carry across tasks and runs |
| Wins | Cannot poison future work | Learns your codebase, your API's quirks |
| Risk | Relearns the same thing every time | **One wrong lesson contaminates forever** |

Default to **episodic**. Persistent memory is the more exciting option and the
one that fails quietly: a lesson learned from a single misleading failure —
"the API always returns 500, use the cache" — gets retrieved forever after,
long after the API was fixed.

If you build persistent memory, it needs all four: attach the task and a
timestamp, retrieve only lessons relevant to the current task (not all of
them), cap the store, and give the user a way to see and delete lessons.
`harness-rag-memory` handles the retrieval half.

### 4. Retry budget: 2, maybe 3

Reflexion's gains are front-loaded. Attempt 2 fixes most of what attempt 1 got
wrong; attempt 4 is usually the model rephrasing attempt 3. Default to
`max_attempts=3` and stop when the score stops improving:

```python
if score <= best_score and attempt > 1:
    break   # not learning; stop paying for it
```

Return the **best** attempt, not the last — a later attempt can be worse.

### 5. Do not stack the whole history

Every reflection in context makes each attempt slower and more confused. Pass
the **most recent 2–3 lessons**, deduplicated, not the full transcript of every
failure.

## Build it

```text
<project>/
├── src/<package>/
│   ├── config.py        # max_attempts, memory mode, score threshold
│   ├── llm.py           # ChatClient + FakeChat + adapter (references/llm-seam.md)
│   ├── memory.py        # lesson store: add, relevant(task, k), dedupe, cap
│   ├── evaluator.py     # Verdict; mechanical first, LLM judge as fallback
│   ├── reflector.py     # transcript + reason -> lesson
│   ├── agent.py         # the attempt loop (a harness-react loop inside)
│   └── cli.py           # run, --attempts, --memory episodic|persistent, -v
└── tests/
```

The outer loop:

```python
def run(self, task: str) -> Attempt:
    lessons = self.memory.relevant(task, k=3)
    best: Attempt | None = None

    for attempt_number in range(1, self.config.max_attempts + 1):
        prompt = self._build_prompt(task, lessons)
        output = self.inner_agent.run(prompt)          # a full ReAct loop
        verdict = self.evaluator.check(task, output)
        attempt = Attempt(attempt_number, output, verdict)
        self._emit("on_attempt", attempt=attempt)

        if best is None or verdict.score > best.verdict.score:
            best = attempt
        if verdict.passed:
            return attempt
        if attempt_number == self.config.max_attempts:
            break
        # Score stopped improving: reflection is not helping, stop paying.
        if attempt_number > 1 and verdict.score <= best.verdict.score:
            break

        lesson = self.reflector.reflect(task, attempt, verdict)
        self._emit("on_lesson", lesson=lesson)
        self.memory.add(task, lesson)
        lessons = self.memory.relevant(task, k=3)

    return best          # the best attempt, not the last
```

Note it returns the best `Attempt`, carrying its verdict. The caller can tell
the user "3 attempts, best scored 0.7, here is why it did not pass" — which is
far more useful than a bare string.

## Failure modes

Beyond `references/failure-modes.md` (the inner loop needs all of it):

- **The evaluator always passes.** The pattern silently becomes ReAct. Guard:
  a test asserting a known-bad output fails evaluation. Run it in CI.
- **Reflection restates the failure.** Guard: constrain the prompt (decision 2)
  and eyeball the first few lessons. If they narrate, the prompt needs work.
- **Lesson poisoning.** One wrong persistent lesson degrades every later run,
  and nobody notices because the harness looks like it is learning. Guard:
  episodic by default; timestamps, relevance-scoped retrieval, and a delete
  command when persistent.
- **Memory growth.** Unbounded lessons fill the context. Guard: dedupe on
  normalized text, cap the store, retrieve top-k.
- **Retrying an environmental failure.** Three reflections about a 429. Guard:
  classify transport/rate-limit errors as non-reflectable — retry with backoff,
  do not spend an attempt.
- **Non-determinism read as learning.** Attempt 2 passing after an identical
  approach means the task is flaky, not that reflection worked. Guard: log the
  lesson alongside the score so a human can see whether the fix was real.
- **The last attempt is worse than an earlier one.** Guard: return best-scoring.

## Required tests

The evaluator tests are the ones that keep this pattern honest:

```python
def test_evaluator_fails_a_known_bad_output():
    verdict = Evaluator().check(task="sum 2 and 2", output="banana")
    assert not verdict.passed          # if this ever passes, the pattern is dead


def test_failure_triggers_exactly_one_retry(config):
    config.max_attempts = 2
    agent = Agent(inner_agent=Scripted(["wrong", "right"]),
                  evaluator=PassesOn("right"), reflector=FixedLesson("use X"),
                  memory=Memory(), config=config)

    result = agent.run("task")
    assert result.number == 2 and result.verdict.passed


def test_the_lesson_reaches_the_next_attempt(config):
    inner = Scripted(["wrong", "right"])
    agent = Agent(inner_agent=inner, evaluator=PassesOn("right"),
                  reflector=FixedLesson("call find_account first"),
                  memory=Memory(), config=config)
    agent.run("task")

    assert "call find_account first" in inner.prompts[1]


def test_best_attempt_is_returned_not_the_last(config):
    config.max_attempts = 3
    agent = Agent(inner_agent=Scripted(["good", "bad", "bad"]),
                  evaluator=ScoreBy({"good": 0.8, "bad": 0.1}),
                  reflector=FixedLesson("x"), memory=Memory(), config=config)

    assert agent.run("task").output == "good"


def test_attempts_are_bounded(config):
    config.max_attempts = 3
    agent = Agent(inner_agent=Scripted(["no"]), evaluator=AlwaysFails(),
                  reflector=FixedLesson("x"), memory=Memory(), config=config)

    result = agent.run("task")
    assert result.number <= 3 and not result.verdict.passed


def test_lessons_are_deduplicated():
    memory = Memory()
    memory.add("t", "use the email, not the id")
    memory.add("t", "Use the email, not the id.")
    assert len(memory.relevant("t", k=5)) == 1
```

Plus the ReAct loop tests from `harness-react` for the inner agent.

## Verify

1. **Tier 0** — install, import, `--help`.
2. **Tier 1** — `pytest -q`, offline.
3. **Force a real failure.** The happy path proves nothing here. Give it a task
   you know attempt 1 fails, run with `-v`, and read the lesson. If it narrates
   instead of instructing, fix the reflector prompt now.
4. **Tier 2** — one real run end to end; confirm the score improved between
   attempts and that the improvement traces to the lesson.

Report the attempt count, the scores, and the lessons produced. If the
evaluator never failed anything during verification, say so — it means the
pattern was never actually exercised.
