#!/usr/bin/env bash
# Sync the canonical RAR skill docs (this dir) into this machine's agent skill dirs.
# Run once per machine after `git pull`. Idempotent (overwrites existing copies).
#
#   bash tools/rar_skills/install.sh
#
# Canonical source of truth is the repo (tools/rar_skills/ + tools/rar.py). The agent
# skill dirs (~/.claude/skills, ~/.codex/skills) are per-machine copies and do NOT sync
# across machines on their own — re-run this after editing the canonical files or on a
# new machine (e.g. the cluster).
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../tools/rar_skills
SKILLS=(RAR-make RAR-find)

sync_to() {
  local base="$1"                                     # e.g. $HOME/.claude
  if [ ! -d "$base" ]; then
    echo "skip $base (agent not present on this machine)"
    return
  fi
  mkdir -p "$base/skills"
  for s in "${SKILLS[@]}"; do
    rm -rf "$base/skills/$s"
    cp -R "$SRC/$s" "$base/skills/$s"
    echo "  synced $base/skills/$s"
  done
}

echo "RAR skills canonical: $SRC"
sync_to "$HOME/.claude"
sync_to "$HOME/.codex"
echo "done. CLI: $(cd "$SRC/.." && pwd)/rar.py"
