# CASE schema — one screening decision

A **case** is the structured decision log for **one screening decision**: the event that turns
run artifacts into a chosen set (a shortlist, a seed table, a wet-lab order). It is the unit
because neither a run nor a date identifies a decision — one run can feed several decisions
(List 1 vs List 2) and one decision can consume several runs (backup_20260713: 4 runs → 6 tables).

Relation to the other records:

| file | records | mutability |
|---|---|---|
| `PROTOCOL/*.md` | **why** — decision logic + its scientific rationale | stable |
| `PROTOCOL/*.runner.md` | **how** — the fixed procedure, with `{{slots}}` left empty | stable |
| `cases/*.yaml` | **this one decision** — which inputs, which criteria, what came out | append-only (one per decision) |
| `LOG.md` | implementation changes | append-only |
| `Results/Analysis/` (RAR) | measurements, conclusions forbidden | append-only |

A case **points at** configs; it never copies their contents. Machine parameters stay in the
config (single source of truth); the case records the pointer plus the CLI increments and the
criteria that were actually applied.

## Identity

`tag: <target>_<intent>_<YYYYMMDD>` — e.g. `q00511_drb0401_tetramer_deimm_20260713`.
Filename = `cases/<tag>.yaml`. The tag carries goal + date, which is what makes a case findable
without reading it; `cases/index.jsonl` is the one-line-per-case index.

## Fields

Required:

- `tag`, `date` (YYYY-MM-DD), `goal` (one line, what decision this makes and for what)
- `protocol` — path to the `PROTOCOL/*.md` whose logic governs
- `runner` — path to the `PROTOCOL/*.runner.md` executed
- `status` — `planned` | `run` | `selected` | `ordered` | `abandoned`
- `inputs` — the run dirs / tables consumed (paths, plus `n_designs` where meaningful)
- `criteria` — the thresholds/rank actually applied **for this decision** (the numbers that
  would otherwise only exist in someone's memory)
- `outputs` — produced files + counts

Optional but recommended:

- `config` — path to the config file the run/refine used (pointer only, never inlined)
- `cli_overrides` — the CLI increments layered on top of that config
- `models` — head checkpoint / refolder / immune oracle identifiers used
- `notes` — caveats that affect interpretation (e.g. "immune regressed vs WT")
- `supersedes` — tag of an earlier case this replaces

Filling a case is the agent's whole job for a screening decision; everything else is the
runner's fixed procedure. Validate with `python scripts/validate_case.py cases/<tag>.yaml`
before executing (runner Step 0).
