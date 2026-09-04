---
name: RAR-find
description: Retrieve, reuse, or cite historical RAR archives when asked for previously archived experiment results, recorded baselines or comparisons, or a specific RAR record. Typically used for substantial completed experiments and analyses. Do not use for live job status, current logs or outputs, routine checks, new analysis alone, or RAR tool/skill maintenance without a historical lookup request.
---

# RAR: find — discover & reuse a Reproducible Analysis Record

Canonical source: this file lives in the repo at `tools/rar_skills/RAR-find/` and is copied
into `~/.claude/skills/` (Claude Code) and `~/.codex/skills/` (Codex) on each machine via
`tools/rar_skills/install.sh`. The CLI and the repo-local registry are SHARED across agents,
so records made by any agent are discoverable here.

## Overview

Retrieve deliberately archived research evidence from the repo-local RAR registry.
Records carry persisted artifacts, so a hit can answer a historical question without
re-running the analysis. This is NOT a mandatory preflight for every experiment,
analysis, metric question, or progress check.

## Invocation boundary

Use when the request calls for historical archived evidence:

- Find, reuse, or cite an archived experiment/analysis or a specific RAR record ID.
- Recover a previously recorded result, baseline, or comparison from past work.
- Check whether an earlier substantial experiment has an archived result, when that
  historical lookup is itself requested or needed for an explicitly requested comparison.

Do NOT query the registry or open the index merely because a request mentions an
experiment, a result, a metric, or the RAR tooling. In particular:

- "Has this experiment finished?", "What is the current loss?", and "Why did this job
  fail?" require current scheduler state, logs, or run outputs, not historical RAR records.
- Inspecting a supplied file/run, routine sanity checks, debugging, and new analyses
  do not require an archive search before starting.
- Editing RAR skills, fixing the index, or troubleshooting sync is tool maintenance,
  not a request to retrieve historical analysis results.

Decide by the evidence the user needs, not experiment size alone. A completed job is
not automatically an archived record. If the request is about a current run or supplies
the relevant artifacts, start there. Do not broaden an ordinary status question into a
historical search; ask a focused clarification only if the intended evidence is ambiguous.

Examples:

| Request | Action |
| --- | --- |
| "Has the experiment I submitted yesterday finished?" | Check scheduler/logs; skip RAR. |
| "Read this output file and report the final AUC." | Read the supplied file; skip RAR. |
| "Run a new analysis on these results." | Analyze the specified artifacts; no RAR preflight. |
| "Find the archived baseline from our previous large benchmark." | Use RAR-find. |
| "Compare this run against the baseline in RAR-0054." | Use RAR-find for the archived baseline only. |
| "Make RAR-find less eager to trigger." | Edit the skill; skip archive retrieval. |

## Workflow

Run the CLI as `python3 tools/rar.py` (vendored in the repo, present on every machine via
git, including the cluster).

```bash
python3 tools/rar.py find "archived benchmark baseline"
python3 tools/rar.py show RECORD_ID  # Replace RECORD_ID with the matching full ID.
```

When the full registry record ID is supplied, prefer `show`; otherwise use a targeted
`find` query to locate the relevant record.
Use `python3 tools/rar.py list` only when browsing the archive catalogue is requested.

`find` ranks by overlap with each record's title / objective / keywords / inputs, and
returns only **current** records (superseded ones hidden; add `--all` to include them).
Here, **current** means "not superseded", NOT "currently running" or "latest live status".

## Consuming a hit

1. Read the printed `record:` path (`record.md`) — it gives Objective / Inputs / Method / Results.
2. Load the files under the printed `data:` path (`data/`) directly for the actual numbers.
   Do NOT re-run the method just to confirm an archived answer. A missing value is not
   authorization to start a new computation.
3. Cite the record `id` + `git_sha` so downstream work stays traceable.

If no record matches or its data are unavailable, state that limitation. Continue with
other evidence or new computation only when it is already within the user's request;
otherwise ask before expanding the task. Do not automatically recompute or create a
new RAR record because a lookup missed.
