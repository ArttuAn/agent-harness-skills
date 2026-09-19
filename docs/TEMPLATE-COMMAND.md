---
description: <One line, imperative. What this command builds.>
argument-hint: your spec — domain, tools, endpoint, constraints
---

Build a <pattern> agent harness for this spec:

$ARGUMENTS

If the spec does not say what the agent is for, which tools it needs, or which
endpoint and model it targets, ask — briefly — then build. Do not ask for
permission to begin.

**Read `~/.claude/skills/<name>/SKILL.md` first** (the `harness-<name>` skill
from agent-harness-skills). It carries the design decisions, the failure modes
and the required tests. If it is not installed, the essentials are below.

## Non-negotiables for this pattern

- <The three or four things that make this pattern work, copied down from the
  skill's "decisions that matter". Terse and imperative — this is what
  survives when the skill file is not installed.>

## Non-negotiables for every pattern

<Copy this block verbatim from an existing command, e.g. commands/react.md.
It is shared across every command on purpose: the LLM seam, tool-error
handling, argument coercion, budget exhaustion, temperature, offline tests.
Do not paraphrase it — drift between commands is a bug.>
