<div align="center">

<img src="docs/logo.svg" width="96" height="96" alt="agent-harness-skills logo"/>

# 🤖 agent-harness-skills

**Skills that make an agentic IDE build an agent harness the way someone who
has run one in production would.**

Seven patterns — ReAct, plan-and-execute, reflexion, multi-agent, code agent,
RAG/memory, self-evolving. One slash command each, for
[opencode](https://opencode.ai) and Claude Code.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Patterns](https://img.shields.io/badge/patterns-7-3b82f6.svg)](#the-skills)
[![Validated](https://img.shields.io/badge/skills-CI%20validated-0d9488.svg)](.github/workflows/check.yml)
[![opencode](https://img.shields.io/badge/opencode-ready-22d3ee.svg)](#install)
[![Claude](https://img.shields.io/badge/Claude%20Code-ready-f97316.svg)](#install)

</div>

## The point

Ask a frontier model for "a ReAct agent harness in Python" and you get a good
one: `src/` layout, `pyproject.toml`, env config, a `@tool` decorator with
type-hint schema inference, a CLI with chat mode. It does not need help with
any of that, and a skill that supplies it is spending your context on the
model's own priors.

What you get instead is a harness that **crashes when a tool raises**, **never
stops when the model keeps calling tools**, **breaks on the next request when
the model emits two tool calls and the loop appends one result**, and **cannot
be tested without an API key** — so the guards never get written, because
nobody can see they are missing.

These skills carry the other 15%: the decisions, the failure modes, and the
tests that prove the guards exist.

```text
/react build an agent that answers questions about my wiki, search tool over a
       sqlite index, max 12 steps, chat mode on
```

## The skills

| Pattern | Skill | Command | The thing it stops you getting wrong |
| --- | --- | --- | --- |
| ReAct | `harness-react` | `/react` | Loops that don't terminate, tool errors that kill the run, an untestable client |
| Plan-and-execute | `harness-plan-and-execute` | `/plan-and-execute` | Unbounded replanning; a verifier that always says yes |
| Reflexion | `harness-reflexion` | `/reflexion` | Self-grading; reflections that narrate instead of instruct; lesson poisoning |
| Multi-agent | `harness-multi-agent` | `/multi-agent` | Building it at all when one agent would win; synthesis that loses the finding |
| Code agent | `harness-code-agent` | `/code-agent` | `startswith` path checks, `shell=True`, no undo, safety claims that aren't true |
| RAG / memory | `harness-rag-memory` | `/rag-memory` | Answering from weights instead of the corpus; exact strings unfindable |
| Self-evolving | `harness-self-evolving` | `/self-evolving` | An optimizer that can edit its own scoring function |

Each skill also says **when not to use the pattern**, and which one to use
instead. See [`references/choosing-a-pattern.md`](references/choosing-a-pattern.md) —
the honest answer is usually ReAct.

## Shared references

Installed alongside every skill, because they apply to all of them:

| File | What it carries |
| --- | --- |
| [`failure-modes.md`](references/failure-modes.md) | 15 failure modes observed against real endpoints — symptom, cause, guard. Schema echoed as an argument value, repeated calls, silent context truncation, OOM that looks like a transport error, provider shape divergence |
| [`llm-seam.md`](references/llm-seam.md) | The one architectural decision that determines testability: an injectable client, a `FakeChat`, and adapters that normalize OpenAI and Ollama |
| [`verification.md`](references/verification.md) | The three-tier contract. Tiers 0 and 1 need no API key |
| [`choosing-a-pattern.md`](references/choosing-a-pattern.md) | When each pattern is right, when it's wrong, which combinations work |

## How a skill is shaped

Every `SKILL.md` has the same five load-bearing sections, and CI enforces it:

```text
## Use this when            — and "don't", pointing at the pattern that wins instead
## The decisions that matter — the forks a model gets wrong unprompted, with a recommendation
## Build it                  — layout, and only the code that carries a decision
## Failure modes             — symptom, cause, guard
## Required tests            — offline, with the fake client; the guards get proven
## Verify                    — three tiers, and report honestly what actually ran
```

Boilerplate the model already writes well — packaging, argparse, config
dataclasses, schema inference — is specified in a sentence, not transcribed in
80 lines. That leaves the context for what it can't.

## Install

```bash
git clone https://github.com/ArttuAn/agent-harness-skills.git
cd agent-harness-skills
./install.sh                 # ~/.config/opencode/skills, ~/.claude/skills, ~/.claude/commands
./install.sh --project       # or into .opencode/ and .claude/ in the current project
./install.sh --check         # validate only, install nothing
```

The installer validates the repo before copying anything — a broken skill is
worse than no skill, because it gets trusted on sight. Skills go to both IDE
locations from one source, with `references/` bundled into each.

## Usage

**Claude Code** — slash command plus spec:

```text
/rag-memory an agent over my markdown notes, cited answers, ollama at
            localhost:11434 with nomic-embed-text
```

**opencode** — name the skill:

```text
harness-react: agent that monitors a log directory and files an issue on an
               error spike, max 8 steps
```

Either way the agent asks for what the spec is missing (domain, tools,
endpoint), scaffolds the project, writes the guards, writes the offline tests,
runs them, and tells you which verification tiers actually passed.

## Validation

```bash
python3 tools/check_skills.py
# OK — 7 skills, 43 python blocks compiled, 5 required sections present in each
```

Checks every Python block parses, frontmatter loads and matches its directory,
each skill has all five required sections, and every skill has a matching
slash command. Runs in CI on every push.

## Adding a pattern

1. `skills/<name>/SKILL.md` with `name: harness-<name>` and a one-line
   `description`.
2. All five required sections. If you cannot fill **"The decisions that
   matter"** and **"Failure modes"** with things you have actually seen go
   wrong, the pattern does not need a skill — the model already handles it.
3. Cite `references/` instead of restating it.
4. `commands/<name>.md` — a thin wrapper with the pattern's non-negotiables.
5. `python3 tools/check_skills.py` must pass.

## Contributing

The most valuable contribution is a failure mode with a reproduction: what you
saw, which endpoint, and the guard that fixed it. Those go in
`references/failure-modes.md` and improve all seven skills at once.

## License

MIT — see [LICENSE](LICENSE).
