---
name: RAR-find
description: Use when about to start a data analysis or answer a data/metric question and you want to check whether a reproducible analysis record already exists for it, or when asked to retrieve, reuse, cite, or look up a prior analysis result instead of recomputing it from scratch.
---

# RAR: find — discover & reuse a Reproducible Analysis Record

Canonical source: this file lives in the repo at `tools/rar_skills/RAR-find/` and is copied
into `~/.claude/skills/` (Claude Code) and `~/.codex/skills/` (Codex) on each machine via
`tools/rar_skills/install.sh`. The CLI and the repo-local registry are SHARED across agents,
so records made by any agent are discoverable here.

## Overview

Query the repo-local RAR registry for an existing reproducible record before computing
anything new. Records carry persisted artifacts, so a hit means you can answer from data
WITHOUT re-running the analysis.

## When to use

- Before running an analysis: check whether it was already done (by any agent).
- When asked "what did we find for X", "reuse the prior result", "is there a record for…".

## Workflow

Run the CLI as `python3 tools/rar.py` (vendored in the repo, present on every machine via
git, including the cluster).

```
CLI=tools/rar.py

python3 $CLI find "latency p99 checkout"            # ranked CURRENT records
python3 $CLI show 0001-latency-cfg-v17-vs-v18       # full entry + data files
python3 $CLI list                                   # all current records
```

`find` ranks by overlap with each record's title / objective / keywords / inputs, and
returns only **current** records (superseded ones hidden; add `--all` to include them).

## Consuming a hit

1. Read the printed `record:` path (`record.md`) — it gives Objective / Inputs / Method / Results.
2. Load the files under the printed `data:` path (`data/`) directly for the actual numbers.
   Do NOT re-run the method unless you need a value that was not persisted there.
3. Cite the record `id` + `git_sha` so downstream work stays traceable.

If no record matches, say so plainly, compute what you need, then consider `RAR-make` to
register the new result.
