#!/usr/bin/env bash
# Sync the repo-vendored Modal skill into this machine's Claude skill directory.
# Run once per machine after `git pull`, and re-run after updating the canonical
# copy under tools/claude_skills/modal/.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/modal"
DEST="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}/modal"

if [ ! -d "$SRC" ]; then
  echo "missing source skill directory: $SRC" >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -R "$SRC" "$DEST"

echo "synced Modal skill: $DEST"
echo "next: configure Modal credentials with 'modal setup' or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET"
