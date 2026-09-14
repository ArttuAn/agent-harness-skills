#!/usr/bin/env bash
# Install agent-harness-skills into opencode and/or Claude Code.
#
# Usage:
#   ./install.sh                 # global: ~/.config/opencode/skills + ~/.claude/commands
#   ./install.sh --project       # local:   .opencode/skills + .claude/commands in this cwd
#   ./install.sh --opencode-dir "$PWD/.opencode/skills"
#   ./install.sh --claude-dir    "$PWD/.claude/commands"
set -euo pipefail

SKILLS_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/skills"
COMMANDS_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/commands"

OPENCODE_DEST="${OPENCODE_DEST:-}"
CLAUDE_DEST="${CLAUDE_DEST:-}"
MODE="global"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) MODE="project"; shift ;;
    --opencode-dir) OPENCODE_DEST="$2"; shift 2 ;;
    --claude-dir) CLAUDE_DEST="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,9p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "$MODE" == "project" ]]; then
  OPENCODE_DEST="${OPENCODE_DEST:-.opencode/skills}"
  CLAUDE_DEST="${CLAUDE_DEST:-.claude/commands}"
else
  OPENCODE_DEST="${OPENCODE_DEST:-$HOME/.config/opencode/skills}"
  CLAUDE_DEST="${CLAUDE_DEST:-$HOME/.claude/commands}"
fi

install_skills() {
  echo "opencode skills -> $OPENCODE_DEST"
  mkdir -p "$OPENCODE_DEST"
  for skill_dir in "$SKILLS_SRC"/*; do
    name="$(basename "$skill_dir")"
    cp -R "$skill_dir" "$OPENCODE_DEST/$name"
    echo "  installed skill: $name"
  done
}

install_commands() {
  echo "claude commands  -> $CLAUDE_DEST"
  mkdir -p "$CLAUDE_DEST"
  for command in "$COMMANDS_SRC"/*.md; do
    cp "$command" "$CLAUDE_DEST/$(basename "$command")"
    echo "  installed command: $(basename "$command")"
  done
}

install_skills
install_commands
echo "done. Skills are now available to your agentic IDEs."