---
name: harness-<name>
description: "<One sentence, under 220 characters. What the IDE will build, and the thing that makes it hard. Start with a verb: Build a ... harness from a spec — ...>"
---

# <Pattern> Agent Harness

<Two or three sentences: what the loop does, and a citation if the pattern has
a paper behind it.>

A frontier model will write <the obvious part> from a spec. What it will not do
is **<the first thing it gets wrong>, <the second>, or <the third>**. Without
those three, this is <the cheaper pattern it degrades into>.

## Use this when

- <A condition that must hold for this pattern to beat a simpler one. Be
  specific enough that a reader can check it against their own situation.>
- <Another.>

Do not use it when <the case where a simpler pattern wins>. Say which pattern
instead — see `references/choosing-a-pattern.md`.

## The decisions that matter

<The heart of the skill. Each decision: what to choose, what the wrong choice
looks like, and why the wrong one is tempting. Concrete defaults with numbers,
not ranges of advice.>

- **<Decision>.** <Choose this. The failure if you choose otherwise.>

## Failure modes

<Things that go wrong in production, not things that go wrong in theory. Real
error strings and named endpoints where you have them. Cross-reference
`references/failure-modes.md` rather than repeating what is already there.>

- **<Symptom.>** <Cause, then the guard.>

## Required tests

<Tests that must exist before this is done. They run offline against a
FakeChat — see `references/llm-seam.md`. Name them as functions.>

```python
def test_<the_guard_that_matters>():
    ...
```

## Verify

<Tiered, and honest about what was not run. See `references/verification.md`.>

0. Install, import, `--help`.
1. `pytest -q` — offline, no API key.
2. One real call against the endpoint. If there was no key, say tier 2 was
   skipped; do not imply an end-to-end run happened.
