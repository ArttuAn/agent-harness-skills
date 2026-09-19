# Contributing

Contributions are welcome from people and from agents. The contract is the
same for both, and it lives in [AGENTS.md](AGENTS.md) — one page, no implicit
conventions. If you are pointing an agent at this repo, point it there.

## The bar

> If a frontier model already gets this right from a bare spec, the pattern
> does not need a skill.

This repo is for what a good model gets *wrong* unprompted. A contribution
earns its place when you can name a specific failure you have seen and the
guard that prevents it. Content that restates the model's own priors is
rejected however well written — that is not a judgement on the writing, it is
what keeps the library worth reading.

## The three contributions, smallest first

**A failure mode.** The most valuable and the cheapest. Append to
`references/failure-modes.md`: what you saw, which endpoint or model, the
guard that fixed it. It improves all seven skills at once. A good one is three
sentences and a real error string.

**A fix to an existing skill.** A decision that is wrong, a default that has
aged, a test that misses the case it claims to cover. Say what broke.

**A new pattern.** Copy `docs/TEMPLATE-SKILL.md` and
`docs/TEMPLATE-COMMAND.md`, fill them in, run the validator. Check first that
it is not an existing pattern wearing a new name — see
`references/choosing-a-pattern.md`.

## Before you open a PR

```sh
python3 tools/check_skills.py
```

No network, no API key, under a second. It must print `OK`. Everything it
enforces is listed in [AGENTS.md](AGENTS.md#validate). CI runs the same
command plus a scratch install, so a green local run means a green PR.

One skill per pull request. Additive changes only — do not restructure
existing skills in the same PR, and do not edit the validator to make your
contribution pass.

## What gets rejected

- Patterns a frontier model already handles from a spec
- Prose that would be equally true of any framework
- Reflows, typography, and "clarity improvements" that do not change meaning
- New dependencies — the validator runs on bare Python 3
- A new skill for a pattern already covered; extend the existing one

## Issues

Issues labelled [`good first issue`](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
are self-contained: one file, a stated bar, and the validator as the judge.
They are written to be completable without reading the rest of the repo, which
makes them a reasonable target for an agent.
