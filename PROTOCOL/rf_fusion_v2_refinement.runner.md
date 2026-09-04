# RUNNER: RF Fusion V2 post-generation Head refinement

Executes `PROTOCOL/rf_fusion_v2_refinement.md`. Fixed order; every `{{slot}}` comes from the CASE.
Nothing here decides science — if a step needs a judgement the PROTOCOL does not already make,
stop and amend the PROTOCOL first.

## Required CASE slots

| slot | CASE field |
|---|---|
| `{{tag}}` | `tag` |
| `{{gen_run_root}}`, `{{gen_campaign_id}}` | `inputs.generation.*` |
| `{{allele}}` | `inputs.allele` |
| `{{test_set_parquet}}`, `{{test_set_sha256}}` | `inputs.*` |
| `{{pdb_root}}` | `inputs.pdb_root` |
| `{{head_checkpoint}}`, `{{head_checkpoint_sha256}}` | `models.*` |
| `{{run_root}}`, `{{log_root}}` | `outputs.*` |
| `{{n_array}}`, `{{walltime}}`, `{{partition}}` | `cli_overrides.resources` |
| search knobs | `criteria.*` |

## Step 0 — validate the CASE

```bash
python scripts/validate_case.py PROTOCOL/cases/{{tag}}.yaml
```

Mandatory. Do not proceed on a validation error.

## Step 1 — assert the generation grid is closed

The refinement consumes immutable products. Confirm from the generation CASE, not from memory:

- every `(protein, root)` cell has a manifest, archive, endpoint, structure and terminal table;
- `realized_caps` are clean;
- the generation CASE's `refinement.status` still reads `reserved`.

Refusing an open grid is the point of this step: a cell that is still being written can change
the eligible pool underneath the selection.

## Step 2 — build the seed table

Read the raw four-root tables and apply the PROTOCOL §2 law. The three groups are built together
so their counts are consistent by construction:

```bash
python {{run_root}}/build_seed_table.py \
  --gen-run-root {{gen_run_root}} \
  --test-set {{test_set_parquet}} \
  --pdb-root {{pdb_root}} \
  --out {{run_root}}/seeds/refine_seeds.parquet
```

The builder must:

1. join `complete_endpoints` + `structure_evaluations` + `terminal_validation` + `archive` per cell;
2. compute `positive_mass_density` from the endpoint's own `head_score_json.residue_hotspot`;
3. apply the four eligibility predicates, dedup on `(protein_id, sequence_md5)`;
4. emit `v2_head_pareto_top8` by the pareto-layer / mean-rank / md5 law, up to 8 per protein;
5. emit `structure_failed_best_sctm` **and** `wild_type_seed` for every protein with zero
   eligible endpoints;
6. drop, and count, any protein whose reference does not resolve under `PDB_ROOT`;
7. write `seed_group`, `selection_rule_id`, `source_campaign`, `source_depth`, `source_cell` on
   every row, re-key `design_idx` per protein, and emit a sidecar manifest with the sha256.

## Step 3 — write `SELECTION.md` beside the seed table

A reader who opens `{{run_root}}` a month later must be able to reconstruct the selection without
this runner. Record: source campaign and its head digest, the four eligibility predicates verbatim,
the ordering law, the three groups with their product/diagnostic status, realized per-group counts,
the structure gate, and every exclusion with its reason.

## Step 4 — preflight the identity bindings

Refuse to submit unless all four hold:

```bash
# 1. the refinement Head is byte-identical to the one the GENERATION used
python - <<'PY'
import json,hashlib,pathlib
m=json.load(open("{{gen_run_root}}/<any_cell>/run_manifest.json"))
gen=m["content_identities"]["head_checkpoint"]
real=hashlib.sha256(pathlib.Path("{{head_checkpoint}}").read_bytes()).hexdigest()
assert gen==real, (gen, real)
print("head identity OK", real)
PY

# 2. every seed protein is in the test set and resolves to a reference
# 3. seed length == that protein's reference length
# 4. no duplicate (protein_id, sequence_md5) and no duplicate (protein_id, design_idx)
```

The driver additionally hard-asserts the head and test-set sha256 and an allele/head name match at
submit time, so a mis-wired run fails before allocation rather than after.

## Step 5 — submit

```bash
cd {{run_root}}
DRY_RUN=1 bash submit_{{tag}}_head_refine.sh     # read the echoed config line by line
DRY_RUN=0 N_ARRAY={{n_array}} WALLTIME={{walltime}} bash submit_{{tag}}_head_refine.sh
```

Walltime: size on the slowest shard, never the mean. Keep it **above one hour** — a Della ailab
request of `--time <= 01:00:00` is demoted to `gpu-test` (`MaxJobsPU=3`, `MaxSubmitPU=25`), which
serializes the array and rejects sibling submissions.

`OUT_DIR` is overridable. A single-protein resubmit MUST set it to a separate directory: with
`N_ARRAY=1` the launcher creates no `shardXXofNN` subdir, so an un-separated run drops
`refined/*.parquet` into the cohort root where the merger does not look.

## Step 6 — monitor from artifacts, not from queue state

Count `seed N done` lines in the logs and shards that printed `final-metrics done`. Identify this
run's logs by its unique run-dir tag: `--output=refine_%j.out` expands `%j` to each task's own
JobID, so globbing on the array id matches nothing, and globbing on `shardNNof16` also catches
older runs. A PENDING task is not progress, and `refined_designs.parquet` appears *before* the
final-metrics pass, so its presence is not completion.

Watch for `Traceback`, `CUDA out of memory`, `DUE TO TIME LIMIT`, `slurmstepd: error`.

## Step 7 — merge

```bash
PYTHONPATH={{project_root}} python scripts/merge_refine_shards.py \
  --run-dir {{run_root}}/cohort \
  --dplm-native-structural {{dplm_native_structural}}
```

`PYTHONPATH` is required — the merger imports `inverse_folding`. Head mode emits
`master.parquet` + `head_pareto_per_seed.{parquet,fasta}`; NMP tables are absent by construction.
Merge any separately-submitted protein directory in the same step and assert the union covers the
seed table exactly once.

For the paper-facing retained run, submit the workers with `OFFICIAL=1` and optionally
`FINAL_CANDIDATES_PER_PROTEIN=N`. The setting is persisted in every shard's `refine_config.json`;
the same merge command reads it and additionally emits `final_candidates.{parquet,fasta}` plus a
content-bound manifest. On a historical evidence-rich run, the equivalent additive operation is:

```bash
PYTHONPATH={{project_root}} python scripts/merge_refine_shards.py \
  --run-dir {{run_root}}/cohort \
  --official \
  --final-candidates-per-protein 8 \
  --seed-table {{run_root}}/seeds/refine_seeds.parquet
```

Neither path removes source shards, the complete front, or prior merged artifacts.

## Step 8 — report, stratified

```bash
python {{run_root}}/report_refine.py \
  --run-dir {{run_root}}/cohort \
  --seed-table {{run_root}}/seeds/refine_seeds.parquet \
  --dplm-native-structural {{dplm_native_structural}}
```

Report per `seed_group` and never pool them. Include coverage, `diverged` count, front width,
best-per-seed delta on both axes, rounds used vs `MAX_ROUNDS`, and `delta_scTM` against the
protein's own same-protocol baseline. Absolute scTM alone is not a conclusion.

## Step 9 — product panel, only if one representative per protein is required

```bash
python {{run_root}}/select_final.py \
  --merged-dir {{run_root}}/cohort/merged \
  --seed-table {{run_root}}/seeds/refine_seeds.parquet \
  --out {{run_root}}/final/final_selection_one_per_seed.parquet
```

One row per seed, minimum `head_global_risk_after`, md5 tie-break. The script asserts that each
protein's globally best front row survives. Rows keep `seed_group`; diagnostic groups stay
labelled and out of any product count.

## Step 10 — close the CASE

Fill `outputs` with realized counts, job IDs, elapsed, and the sha256 of the seed table and the
product panel. Flip `status` to `selected`. Set the generation CASE's `refinement` block to point
at this case. Record every exclusion and every resubmission.
