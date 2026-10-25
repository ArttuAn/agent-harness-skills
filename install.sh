#!/usr/bin/env bash
# Install agent-harness-skills into opencode and/or Claude Code.
#
# Usage:
#   ./install.sh                  # global: ~/.config/opencode/skills, ~/.claude/skills, ~/.claude/commands
#   ./install.sh --project        # local:  .opencode/skills, .claude/skills, .claude/commands
#   ./install.sh --opencode-dir DIR
#   ./install.sh --claude-skills-dir DIR
#   ./install.sh --claude-dir DIR          # slash commands
#   ./install.sh --check                   # validate the repo, install nothing
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_SRC="$HERE/skills"
COMMANDS_SRC="$HERE/commands"
REFERENCES_SRC="$HERE/references"

OPENCODE_DEST="${OPENCODE_DEST:-}"
CLAUDE_SKILLS_DEST="${CLAUDE_SKILLS_DEST:-}"
CLAUDE_DEST="${CLAUDE_DEST:-}"
MODE="global"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) MODE="project"; shift ;;
    --opencode-dir) [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "--opencode-dir requires a directory" >&2; exit 2; }; OPENCODE_DEST="$2"; shift 2 ;;
    --claude-skills-dir) [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "--claude-skills-dir requires a directory" >&2; exit 2; }; CLAUDE_SKILLS_DEST="$2"; shift 2 ;;
    --claude-dir) [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "--claude-dir requires a directory" >&2; exit 2; }; CLAUDE_DEST="$2"; shift 2 ;;
    --check) python3 "$HERE/tools/check_skills.py"; exit $? ;;
    --help|-h) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "$MODE" == "project" ]]; then
  OPENCODE_DEST="${OPENCODE_DEST:-.opencode/skills}"
  CLAUDE_SKILLS_DEST="${CLAUDE_SKILLS_DEST:-.claude/skills}"
  CLAUDE_DEST="${CLAUDE_DEST:-.claude/commands}"
else
  OPENCODE_DEST="${OPENCODE_DEST:-$HOME/.config/opencode/skills}"
  CLAUDE_SKILLS_DEST="${CLAUDE_SKILLS_DEST:-$HOME/.claude/skills}"
  CLAUDE_DEST="${CLAUDE_DEST:-$HOME/.claude/commands}"
fi

# Never install a repo that does not validate — a broken skill is worse than
# no skill, because it is trusted on sight.
echo "validating..."
python3 "$HERE/tools/check_skills.py" --quiet

exec python3 "$HERE/tools/install_safe.py" \
  --root "$HERE" \
  --opencode-dir "$OPENCODE_DEST" \
  --claude-dir "$CLAUDE_DEST" \
  --claude-skills-dir "$CLAUDE_SKILLS_DEST"
