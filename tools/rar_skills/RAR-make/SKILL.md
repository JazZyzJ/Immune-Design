---
name: RAR-make
description: Create or update a Reproducible Analysis Record only when the user explicitly invokes RAR-make or requests RAR archiving.
---

# RAR: make — write a Reproducible Analysis Record

Canonical source: this file lives in the repo at `tools/rar_skills/RAR-make/` and is copied
into `~/.claude/skills/` (Claude Code) and `~/.codex/skills/` (Codex) on each machine via
`tools/rar_skills/install.sh`. Edit here, then re-run it. The CLI and the repo-local registry are SHARED across agents, so a
record made by one agent is discoverable by the others via the same `index.jsonl`.

## Overview

A Reproducible Analysis Record (RAR) is an objective, reproducible, machine-reusable record
of ONE analysis, written so a DIFFERENT agent can reuse the result WITHOUT re-running it.
It is NOT a human report.

Within `record.md`, separate measured results and prespecified criteria from interpretation.
This format does not constrain the human-facing answer, which may explain conclusions,
limitations, and next steps as requested.

## Invocation boundary

- Use only when the user invokes `$RAR-make`, asks to create/update a RAR record,
  or explicitly includes RAR archiving in the task. An earlier explicit request
  within the same task remains authorization; do not ask again at the end.
- Completing an analysis, producing reusable numbers, writing a summary or handoff,
  and using words such as "interpretation" or "next steps" are not triggers.
- Without an explicit RAR request, deliver the requested analysis and artifacts
  without creating a record, querying the registry, or routinely asking about archiving.
- A generic request to save results or write a report does not imply RAR archiving.
- RAR skill maintenance is not a request to archive the maintenance work.

## The contract (fixed logic chain: Objective → Inputs → Method → Results)

`record.md` MUST contain these headings (the CLI rejects a commit without them):

1. `# <Title>`
2. `## Provenance` — table covering: dataset, exact source paths, **code version**
   (`git rev-parse --short HEAD`), run id / checkpoint digests if any, library + language
   versions. Enough for another agent to locate the exact inputs.
3. One self-contained block **per measurement** (one or many), each with headings:
   - `### … Objective` — the question. If a decision rule was fixed *before* seeing the
     results (e.g. "ρ > 0.7 → prior is usable"), state it HERE as an objective criterion.
   - `### … Inputs` — exact source file(s) + the column/field names used.
   - `### … Method` — numbered exact steps; formulas; filters with counts dropped;
     conventions that affect the numbers (interpolation mode, no outlier capping, …).
   - `### … Results` — the numbers, in tables. Every result must also exist as a file in
     `data/`. Compare value-vs-criterion only; no prose verdict.
4. `## Artifacts` — list each `data/` file + its columns, so an agent loads it directly.
5. `## Reproduction` — one line: deterministic regeneration from the inputs above + deps.

## Two hard rules

**Rule 1 — machine-reusable artifacts are mandatory.** Every reported result is persisted as
a file in the record's `data/` dir (CSV / JSON / parquet). NEVER write "regenerate the table
from the method" — the actual numbers must be on disk. The CLI rejects an empty `data/`.

**Rule 2 — no subjective judgment.** Facts and pre-registered criteria only.

| Forbidden (post-hoc judgement) | Allowed (objective) |
|---|---|
| "Bottom line: latency regressed" | "p99 312 → 488 ms" |
| an `## Interpretation` / "what this means" section | a decision rule stated *before* results in `Objective` |
| `## Caveats` prose ("could be traffic not config") | a `Method` fact ("single day; versions not time-separated") |
| `## Suggested next steps` | (omit entirely) |
| "clearly real", "materially worse", "concerning" | the number + the criterion it is compared against |

Put interpretive conclusions in the accompanying answer or report. Continue the
authorized task; a record-format distinction is not a reason to stop for approval.

## Workflow

Run the CLI as `python3 tools/rar.py` (vendored in the repo, present on every machine via
git, including the cluster). RAR operates on a repo-local registry, so run it from inside the repo.

```
CLI=tools/rar.py

python3 $CLI new --title "Latency cfg_v17 vs v18"
# -> mints id (e.g. 0001-latency-cfg-v17-vs-v18) and creates <root>/<id>/data/

# write <root>/<id>/record.md following the contract above
# save EVERY reported result as a file in <root>/<id>/data/

python3 $CLI commit --id 0001-latency-cfg-v17-vs-v18 \
    --objective "did p99 latency regress cfg_v17->cfg_v18 on checkout-api" \
    --keywords latency,p99,checkout-api --inputs "s3://acme-logs/checkout-api/2026-06-01/"
```

`commit` validates the required sections + non-empty `data/`, stamps git sha + timestamp,
and appends one line to `<root>/index.jsonl`. Registry root defaults to `./rar`; override
once per repo with `python3 $CLI init --root <path>`.

**Re-doing a superseded analysis:** `commit … --supersedes <old-id>` so `RAR-find` returns
only the live record.

## Record-only checks

Keep `record.md` factual, retain the Objective → Inputs → Method → Results chain,
and persist its reported results in `data/`. State methodological limitations as
facts. Check the actual artifacts before committing; a description of how to
regenerate a missing table does not replace that table.

These checks apply only after an explicit RAR request. They neither trigger
archiving nor prohibit interpretation in the human-facing deliverable.
