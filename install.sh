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
    --opencode-dir) OPENCODE_DEST="$2"; shift 2 ;;
    --claude-skills-dir) CLAUDE_SKILLS_DEST="$2"; shift 2 ;;
    --claude-dir) CLAUDE_DEST="$2"; shift 2 ;;
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

install_skills_into() {
  local dest="$1" label="$2"
  echo "$label -> $dest"
  mkdir -p "$dest"
  for skill_dir in "$SKILLS_SRC"/*; do
    name="$(basename "$skill_dir")"
    rm -rf "${dest:?}/$name"
    cp -R "$skill_dir" "$dest/$name"
    # The shared references travel with every skill so a SKILL.md can cite
    # them wherever it was installed.
    cp -R "$REFERENCES_SRC" "$dest/$name/references"
    echo "  installed skill: $name"
  done
}

install_skills_into "$OPENCODE_DEST" "opencode skills"
install_skills_into "$CLAUDE_SKILLS_DEST" "claude skills "

echo "claude commands  -> $CLAUDE_DEST"
mkdir -p "$CLAUDE_DEST"
for command in "$COMMANDS_SRC"/*.md; do
  cp "$command" "$CLAUDE_DEST/$(basename "$command")"
  echo "  installed command: $(basename "$command")"
done

echo
echo "done — 7 skills, 7 slash commands, references bundled with each skill."
