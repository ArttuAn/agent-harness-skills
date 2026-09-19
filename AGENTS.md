# AGENTS.md

Instructions for an agent contributing to this repository. Read this before
editing anything. It is the whole contract — there is no implicit convention
you are expected to infer.

## What this repo is

Seven skills that teach an agentic IDE to build an **agent harness** (ReAct,
plan-and-execute, reflexion, multi-agent, code agent, RAG/memory,
self-evolving). Each skill is one `skills/<name>/SKILL.md` plus one matching
`commands/<name>.md`. Shared prose lives in `references/`.

## The bar, before you write anything

> **If a frontier model already gets this right from a bare spec, the pattern
> does not need a skill.**

This repo exists for the things a good model gets *wrong* unprompted: the
evaluator that rubber-stamps its own output, the lesson that poisons every
later run, the tool result that breaks the *next* request rather than the one
that caused it. Content that restates what the model already does is rejected,
however well written.

A contribution is worth making if you can name **a specific failure you have
seen**, and the guard that prevents it.

## Validate

One command. No network, no API key, under a second:

```sh
python3 tools/check_skills.py
```

It must print `OK` before you open a pull request. It enforces:

| Rule | Detail |
|---|---|
| Location | `skills/<name>/SKILL.md` |
| Frontmatter | `name: harness-<name>` (must match the directory), plus `description` |
| Description length | 220 characters maximum |
| Required sections | `## Use this when`, `## The decisions that matter`, `## Failure modes`, `## Required tests`, `## Verify` |
| Python blocks | every ```` ```python ```` block must parse; `<placeholders>` are fine |
| Matching command | `commands/<name>.md` must exist |

## Add a skill

1. Copy `docs/TEMPLATE-SKILL.md` to `skills/<name>/SKILL.md`.
2. Copy `docs/TEMPLATE-COMMAND.md` to `commands/<name>.md`.
3. Fill them in. Cite `references/` rather than restating it — the
   LLM-seam and verification rules are shared and must not be duplicated.
4. Run `python3 tools/check_skills.py`.
5. Open a PR. Say in one line which failure the skill prevents.

## Add a failure mode

The most valuable contribution, and the smallest. Append to
`references/failure-modes.md`: what you saw, which endpoint or model, and the
guard that fixed it. It improves all seven skills at once.

## Rules

- **One skill per pull request.** Additive changes only; do not restructure
  existing skills in the same PR.
- **Do not edit `tools/check_skills.py` to make your contribution pass.** If
  the validator is genuinely wrong, say so in an issue first.
- **Do not add dependencies.** The validator runs on a bare Python 3 with no
  packages installed, and it stays that way.
- **Write from experience, not from the model's priors.** Concrete numbers,
  real error strings, named endpoints. If the text would be equally true of
  any framework, it is too vague to ship.
- Python blocks are illustrative, not runnable programs; they must parse.

## Do not

- Touch `LICENSE`, or the CI workflow, without saying why in the PR.
- Add a skill for a pattern already covered — extend the existing one.
- Open a PR that only fixes typography, reflows prose, or "improves clarity"
  without changing meaning.
