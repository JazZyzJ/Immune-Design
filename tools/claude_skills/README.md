# Claude Skill Exports

This directory vendors agent skills that should travel with the repository and be
installed on each machine where Claude Code is used.

## Modal

The `modal/` directory is an exported copy of the Modal skill. Install or refresh
it on a machine with:

```bash
bash tools/claude_skills/install_modal.sh
```

The installer copies `tools/claude_skills/modal/` to `~/.claude/skills/modal`.
It is safe to re-run after `git pull`; the existing installed copy is replaced.

Modal authentication is intentionally not stored in this repository. On each
machine, configure credentials with either:

```bash
modal setup
```

or environment variables:

```bash
export MODAL_TOKEN_ID=<token-id>
export MODAL_TOKEN_SECRET=<token-secret>
```
