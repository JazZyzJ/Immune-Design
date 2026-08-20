# RUNNER: constrained-redesign shortlist & refinement (agent execution manual)

Fixed procedure for the decision logic in `shortlist_and_refine_seed_selection.md`. Execute the
steps **in order**, verbatim. Every variable is a `{{slot}}` filled from the case file
(`PROTOCOL/cases/<tag>.yaml`) — never improvise a value, never skip a verification.

> If a step's verification fails, STOP and report. Do not proceed, do not substitute a value,
> do not "fix" it by loosening a threshold. A failed gate is information, not an obstacle.

## Slots (all from the case)

| slot | source | example |
|---|---|---|
| `{{run_dir}}` | `inputs.run_dir` | `Results/RF/HLA-DRB1_04_01/q00511_safetyv1_..._20260713T061730Z` |
| `{{constraint_manifest}}` | `inputs.constraint_manifest` | `{{run_dir}}/meta/constraint_manifest.yaml` |
| `{{allele}}` | `inputs.allele` (NMP form) | `HLA-DRB1*04:01` |
| `{{head_ckpt}}` / `{{head_config_dir}}` | `models.head_*` | — |
| `{{cat_max}}` `{{min_plddt}}` `{{sctm_min}}` | `criteria.list1_floor` | `2.0` / `85` / `0.94` |
| `{{refine_out}}` | `outputs.refine_dir` | — |
| `{{pdb_root}}` `{{test_set_parquet}}` `{{esmfold_cache}}` `{{netmhciipan_bin}}` | `models.*` / cluster paths | CLI only, never hardcoded |

## Step 0 — validate the case (MANDATORY, before any compute)

```bash
python scripts/validate_case.py PROTOCOL/cases/{{tag}}.yaml
```

Proceed only on `ALL-PASS`. This checks: required fields present, referenced paths exist,
protocol/runner paths resolve, criteria are numeric, and (when applicable) the head tag's allele
matches `inputs.allele`. Cheap; catches the class of mistakes that otherwise dies inside a
cluster job an hour later.

## Step 1 — confirm the run is complete and constrained as claimed

```bash
ls {{run_dir}}/eval_structure/structural.parquet \
   {{run_dir}}/eval_structure/structural_residues.parquet \
   {{run_dir}}/eval_immune/imm_nmp.parquet {{constraint_manifest}}
```

Verify: `n_designs` matches `inputs.n_designs`; every design preserved every anchor
(`constraints_applied_summary.json` or `matched_anchor_count`; glycine anchors legitimately
report `no_sidechain` — that is not a failure). Mismatch → STOP.

## Step 2 — selection (List 1 + List 2)

```bash
python scripts/select_redesign_shortlists.py \
    --run-dir {{run_dir}} \
    --cat-max {{cat_max}} --min-plddt {{min_plddt}} --sctm-min {{sctm_min}}
```

Verify: printed survivor counts equal `outputs.list1.n` / `outputs.list2.n` in the case. If they
differ, the case's `criteria` and the actual thresholds have drifted — STOP and reconcile
(update the case, do not silently accept new numbers).

Produces `{{run_dir}}/selection/`: `list1_tetramer_shortlist.{parquet,csv}`,
`list2_refine_seeds.{parquet,csv}`, `selection_manifest.json`.

## Step 3 — List 1 → structure prediction (immune-agnostic)

Hand `list1_tetramer_shortlist.parquet` to the predictor named in `criteria.predictor`
(AF3 / Protenix), in rank order, best-first. Uncapped unless the case caps it.
Results land in `{{run_dir}}/tetramer_*/`. List 1 requires no refinement — it is the structural
ceiling probe, not the de-immunized product.

## Step 4 — List 2 → refinement (anchors frozen)

```bash
SEED_TABLE={{run_dir}}/selection/list2_refine_seeds.parquet \
CONSTRAINT_MANIFEST={{constraint_manifest}} \
ALLELE='{{allele}}' \
HEAD_CHECKPOINT={{head_ckpt}} \
OUT_DIR={{refine_out}} \
sbatch scripts/submit_refine.slurm
```

Mode `refine`. The immune Head **must** be the one trained for `{{allele}}` — a mismatched head
silently optimizes the wrong allele. Step 0 checks this; re-confirm it here before submitting.

## Step 5 — merge shards + post-refine selection

```bash
python scripts/merge_refine_shards.py --run-dir {{refine_out}} --top-k {{top_k}}
```

Keep only (good structure ∧ `core_count_after == 0`). Structure is re-evaluated on refined
outputs; do **not** carry the seed's structure forward as if it still held.

## Step 6 — refined winners → structure prediction

Same predictor as Step 3, on the post-refine survivors. This is the de-immunized product line.

## Step 7 — close the case

1. Fill `outputs` with real paths + counts; set `status` (`selected` → `ordered` once purchased).
2. If sequences were ordered, write them under `Results/Wetlab/<order>/` with a README stating
   the source run, the selection rule, and the tag encoding.
3. Append one line to `PROTOCOL/cases/index.jsonl`.
4. `LOG.md` only if code/data/config behavior changed — a screening decision alone does not
   qualify (see CLAUDE.md §2).

## Slot-filling rules

- Cluster paths (`{{pdb_root}}`, `{{esmfold_cache}}`, `{{netmhciipan_bin}}`, …) are passed on the
  CLI or via the SLURM env, **never** written into a Python module.
- Thresholds come from `criteria` in the case. To change one, edit the case (that is the record
  of the decision) — never edit the protocol's defaults for a single run.
- A value absent from both the case and the config is a **blocking question for the user**, not
  a guess.
