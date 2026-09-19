## What failure does this prevent?

<!-- One or two sentences. Something you have actually seen, not something
that could theoretically happen. Real error strings and named endpoints help.
If the answer is "a frontier model already handles this", it does not need a
skill — see AGENTS.md. -->

## Type

- [ ] New failure mode in `references/failure-modes.md`
- [ ] Fix to an existing skill
- [ ] New pattern (`skills/<name>/SKILL.md` + `commands/<name>.md`)

## Checks

- [ ] `python3 tools/check_skills.py` prints `OK`
- [ ] One skill / one concern in this PR
- [ ] No new dependencies
- [ ] `tools/check_skills.py` is unchanged (or the PR explains why it had to change)
