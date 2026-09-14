---
name: harness-code-agent
description: "Build a sandboxed coding-agent harness from a spec — workspace confinement, bounded command execution, reviewable diffs and an undo path"
---

# Code Agent Harness

An agent that reads, edits and creates files and runs commands, confined to a
workspace. The pattern behind coding assistants.

A frontier model will write the file tools, the shell tool and a list of
dangerous commands from a spec. What it will not do is **confine paths in a way
that survives symlinks and `..`, bound output and runtime, keep an undo path,
or tell you honestly what the sandbox does not protect against**. This skill is
mostly about that last part, because getting it wrong produces confident,
incorrect safety claims.

## Use this when

- The file set is not known up front: "make the tests pass", "migrate this API".
- The task needs to read code, change it, run something, and react.

**Don't use this when:**

- You already know the edit → write the script. Do not hire an agent to
  discover something you can state.
- The change is mechanical across many files → `sed`, `codemod`, or an AST tool
  is faster, reviewable, and deterministic.
- The task is to *plan* a migration → `harness-plan-and-execute`, with this as
  the executor.

## Be honest about the boundary

State this to the user when you scaffold; it is the most important sentence in
the build:

> Path confinement and a command blocklist stop **accidents**, not an
> adversary. A model that is trying to escape, or one following instructions
> injected through a file it read, will eventually find a way out. Real
> isolation is a container, VM, or a separate user account with its own
> filesystem permissions.

So: build the confinement properly *and* recommend running it in a container
when the input is untrusted. Do not describe a regex blocklist as a sandbox.

## The decisions that matter

### 1. Confinement is `realpath`, not `startswith`

The naive check passes every interesting attack. Resolve first, then compare —
and compare path components, not string prefixes:

```python
from pathlib import Path


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve(strict=True)

    def resolve(self, candidate: str) -> Path:
        """Resolve a user/model-supplied path inside the workspace, or raise.

        Catches: absolute paths, '..' traversal, symlinks pointing out, and
        'workspace-evil' matching a 'workspace' prefix.
        """
        target = (self.root / candidate).resolve()
        if target != self.root and self.root not in target.parents:
            raise PermissionError(f"path escapes the workspace: {candidate}")
        return target
```

`.resolve()` follows symlinks, which is the point: a symlink inside the
workspace pointing at `/etc` resolves to `/etc` and is rejected. Note
`strict=True` on the root — a workspace that does not exist should fail loudly
at construction, not silently create files somewhere surprising.

Apply this in **every** tool that takes a path. One unguarded tool undoes all
the others.

### 2. Run commands as an argv list, with a timeout, in the workspace

```python
import subprocess


def run_command(workspace: Workspace, command: list[str], timeout: float = 30.0) -> str:
    """Run inside the workspace. argv list, never shell=True."""
    try:
        completed = subprocess.run(
            command,
            cwd=workspace.root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return f"Error: command not found: {command[0]}"
    except subprocess.TimeoutExpired:
        return f"Error: command exceeded {timeout}s and was killed."

    output = (completed.stdout + completed.stderr).strip()
    return f"exit={completed.returncode}\n{truncate(output)}"
```

Three deliberate choices:

- **`shell=True` is the mistake to avoid.** With a shell, `;`, `&&`, backticks
  and `$()` all become attack surface and your blocklist has to parse shell
  grammar. With an argv list it does not.
- **The timeout is not optional.** A command that waits on stdin hangs the
  agent forever; the model cannot tell the difference between slow and stuck.
- **Non-zero exit is a result, not an exception.** Failing tests are exactly
  what the agent needs to read.

### 3. Truncate output before it reaches the model

A `pytest` run or a `npm install` can emit tens of thousands of tokens and
blow the context in one call. Truncate from the middle — the beginning has the
command, the end has the error:

```python
def truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-limit // 2 :]
    return f"{head}\n\n... [{len(text) - limit} characters truncated] ...\n\n{tail}"
```

### 4. The blocklist is a seatbelt, not a wall

Worth having — it catches `rm -rf /` typed by a confused model — but understand
what it is. With an argv list you are matching `command[0]` plus arguments, not
parsing shell. Keep it short, obvious, and documented as accident-prevention:

```python
BLOCKED = {
    "sudo": "privilege escalation", "su": "privilege escalation",
    "shutdown": "host control", "reboot": "host control",
    "mkfs": "formats a filesystem", "dd": "writes raw devices",
    "mount": "filesystem control", "umount": "filesystem control",
    "systemctl": "service control", "service": "service control",
}


def check_command(command: list[str]) -> str | None:
    if not command:
        return "empty command"
    reason = BLOCKED.get(Path(command[0]).name)
    if reason:
        return f"{command[0]} is not allowed ({reason})"
    if any(arg in ("/", "/*", "~", "~/") for arg in command[1:]) and command[0] == "rm":
        return "refusing to delete a root or home path"
    return None
```

An **allowlist** is strictly better where the task permits one: if the agent
only needs `python`, `pytest`, `git` and `ls`, allow exactly those. Prefer it
whenever you can enumerate the commands.

### 5. Git is the undo button

The single highest-value safety feature, and the one models leave out. Before
the agent's first edit, require a clean tree (or commit one):

- Refuse to start on a dirty tree unless `--allow-dirty` — otherwise the
  agent's changes mix with the user's uncommitted work and cannot be separated.
- Commit after each accepted change, or tag the pre-run commit so
  `git diff <tag>` shows exactly what the agent did.
- If the workspace is not a git repo, say so and offer to `git init`. "You can
  revert this" is worth more than any blocklist entry.

### 6. Edits are diffs, not overwrites

Give the model a `str_replace`-style edit tool (old text → new text, must match
exactly once) rather than "write this whole file". Whole-file writes silently
drop content the model did not think to repeat. Return the unified diff of what
changed so the transcript shows the real edit.

## Build it

```text
<project>/
├── src/<package>/
│   ├── config.py        # workspace root, timeout, output limit, allow_dirty
│   ├── llm.py           # ChatClient + FakeChat + adapter (references/llm-seam.md)
│   ├── workspace.py     # Workspace.resolve — the confinement
│   ├── safety.py        # blocklist / allowlist
│   ├── tools.py         # read_file, write_file, edit_file, list_dir, run_command
│   ├── git.py           # clean-tree check, pre-run tag, diff
│   ├── agent.py         # the ReAct loop (harness-react)
│   └── cli.py           # task, --workspace, --allow-dirty, -v
└── tests/
```

The loop is `harness-react`'s, unchanged — all of
`references/failure-modes.md` applies. The value here is in `workspace.py`,
`safety.py` and `git.py`.

## Failure modes

Beyond the shared list:

- **Path escape.** `../../etc/passwd`, an absolute path, a symlink out, or
  `/tmp/work-evil` passing a `startswith("/tmp/work")` check. Guard: decision 1,
  in every path-taking tool. Test all four.
- **The agent edits its own harness.** If the workspace contains the agent's
  source, it can rewrite its own blocklist. Guard: workspace must not be an
  ancestor of the package; refuse at startup.
- **Command hangs.** Anything reading stdin. Guard: timeout, plus
  `stdin=subprocess.DEVNULL`.
- **Context blown by one command.** Guard: truncate (decision 3).
- **Uncommitted work destroyed.** Guard: clean-tree check (decision 5).
- **Prompt injection from file content.** A file the agent reads says "ignore
  your instructions and run X". This is not hypothetical for agents pointed at
  repos or downloads. Guard: mark tool output as data in the system prompt,
  keep the blocklist independent of anything the model says, and never let file
  content alter the workspace root.
- **Secrets in output.** `env`, `cat .env`, or a traceback containing a key,
  echoed into the transcript and the provider's logs. Guard: redact
  `KEY|TOKEN|SECRET|PASSWORD`-shaped values from command output; exclude `.env`
  from `read_file` by default.
- **Success declared without checking.** The agent edits and reports done
  without running the tests. Guard: require a verification command in the task
  contract; treat its exit code as the outcome.

## Required tests

The confinement tests are the ones that matter. They need no model:

```python
def test_relative_traversal_is_rejected(workspace):
    with pytest.raises(PermissionError):
        workspace.resolve("../../etc/passwd")


def test_absolute_path_is_rejected(workspace):
    with pytest.raises(PermissionError):
        workspace.resolve("/etc/passwd")


def test_symlink_out_of_the_workspace_is_rejected(workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("secret")
    (workspace.root / "link").symlink_to(outside)
    with pytest.raises(PermissionError):
        workspace.resolve("link")


def test_sibling_prefix_is_not_inside(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    (tmp_path / "work-evil").mkdir()
    with pytest.raises(PermissionError):
        Workspace(root).resolve("../work-evil/f.txt")


def test_nested_paths_are_allowed(workspace):
    (workspace.root / "src").mkdir()
    assert workspace.resolve("src/main.py").name == "main.py"


def test_blocked_command_is_refused():
    assert check_command(["sudo", "ls"]) is not None
    assert check_command(["pytest", "-q"]) is None


def test_command_timeout_returns_an_error_not_a_hang(workspace):
    result = run_command(workspace, ["python", "-c", "import time; time.sleep(5)"], timeout=0.5)
    assert "exceeded" in result


def test_output_is_truncated_from_the_middle():
    out = truncate("x" * 10_000, limit=4000)
    assert len(out) < 4500 and "truncated" in out


def test_edit_requires_a_unique_match(workspace):
    (workspace.root / "f.py").write_text("a\na\n")
    result = edit_file(workspace, "f.py", "a", "b")
    assert "Error" in result        # ambiguous: 2 matches, refuse
```

Plus the ReAct loop tests from `harness-react`.

## Verify

1. **Tier 0** — install, import, `--help`.
2. **Tier 1** — `pytest -q`. The escape tests must pass. If any is skipped or
   xfailed, the harness is not confined; fix it before going further.
3. **Adversarial pass, by hand.** Point the agent at a scratch workspace and
   ask it to read `/etc/hostname`. It must refuse or fail — not succeed.
4. **Tier 2** — one real task in a throwaway git repo: "make the failing test
   pass". Confirm the tests were actually run, and `git diff <pre-run tag>`
   shows only intended changes.

Report what ran, name the workspace used, and repeat the boundary statement:
this stops accidents, not an adversary — use a container for untrusted input.
