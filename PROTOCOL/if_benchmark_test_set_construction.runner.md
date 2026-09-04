# IF Benchmark Test-Set v3 Runner

Protocol authority: `PROTOCOL/if_benchmark_test_set_construction.md`, Protocol ID
`if-benchmark-test-set/3`. This runbook does not redefine scientific thresholds.

## Immutable release scope

- Allele: `HLA-DRB1*15:01`; tag: `HLA-DRB1_15_01`.
- Primary targets: Tier 1 = 15, Tier 2 = 3000.
- Every attempt writes to `if_test_set/builds/<build_id>/HLA-DRB1_15_01/`.
- Failed build IDs are never resumed as another scientific attempt.
- No command in `scripts/build_if_benchmark_v3.py` moves `if_ready/main`.
- `finalize-candidate` only performs the immutable candidate-to-release rename and requires
  `--confirm-candidate-only`; publication of the main alias remains a separate human decision.

## Reuse and resume identity

RCSB coordinate cache entries are revision- and content-addressed as
`<PDB>/<revision>/<sha256>.cif`. Reuse requires the cache manifest, expected GraphQL revision,
entry identity, file size, and SHA-256 to replay. A newer revision creates a different directory;
it never replaces old bytes. Failed requests retain a self-hashed request ledger and response body.

NetMHCIIpan evidence is query-sharded. Resume is legal only when the shard manifest's query-table,
tool, data Merkle, parameter, and file identities all match. Native FASTA, explicit native-ID to
original-ID maps, exact argv, stdout, stderr, timeout/nonzero records, parsed window tables, and
summaries are retained per shard. Production validation reparses native stdout and requires exact
frame equality with persisted 0-based, native-percentage windows.

## Short probes

Use the parameterized launcher; its logs always land in
`/scratch/gpfs/KAIYIJIANG/zijie/logs/if_benchmark_v3/`.
The launcher intentionally does not hard-code a partition: pass `--partition=rtx6000` for the
idle RTX route, or use `--qos=gpu-test --gres=gpu:1` without a partition for sub-hour fallback
jobs, as required by Della's test-QoS routing policy.

```bash
MODE=nmp_probe \
OUTPUT_DIR=<build>/HLA-DRB1_15_01/audit/smoke/nmp \
NMP_BINARY=<install>/netMHCIIpan \
NMP_INSTALL_ROOT=<install> \
sbatch scripts/submit_if_benchmark_v3.slurm
```

The probe Merkle-hashes the official install and scores one 100-aa sequence at length 15 and all
lengths 12--25. It validates native 1-based to canonical 0-based positions, complete window counts,
`%Rank_EL` percentage units, filtering disabled by omitting the 4.3i
presence-toggle `-filter` flag, and enabling context with the valueless presence-toggle
`-context` flag. The real-binary stdout must state `Prediction Mode: EL with Context`, and the smoke
must prove all 86 length-15 and 1,155 length-12--25 windows for the frozen
100-residue probe before production scoring is allowed.

```bash
MODE=mmseqs_probe \
OUTPUT_DIR=<build>/HLA-DRB1_15_01/audit/smoke/mmseqs \
MMSEQS_TOOL=<env>/bin/mmseqs \
sbatch scripts/submit_if_benchmark_v3.slurm
```

The MMseqs probe creates its query/reference FASTAs inside the versioned audit root, runs and
native-replays both CATH `cov-mode=0` and Head-homology `cov-mode=2`, then reconstructs each
sidecar from the persisted TSV. Tool/version/argv/input/output identities are mandatory.

```bash
MODE=head_probe \
OUTPUT_DIR=<build>/HLA-DRB1_15_01/audit/smoke/head \
HEAD_CHECKPOINT=<full-data-1501>/epoch_24.pt \
HEAD_RESOLVED_CONFIG=<full-data-1501>/resolved_config.yaml \
HEAD_RUN_SUMMARY=<full-data-1501>/run_summary.json \
sbatch scripts/submit_if_benchmark_v3.slurm
```

The Head probe scores two frozen 100-aa sequences and binds the cohort table, nested checkpoint
metadata (`epoch=24`, `global_step=10055`, `config_hash=ad6ac404027b`, manifest `v1.1`), resolved
config, run summary, model configs, output rows, and scores. `verify-head-annotation --replay`
reloads the registered predictor and requires exact score equality.

```bash
python scripts/build_if_benchmark_v3.py freeze-structure-source \
  --output-dir <build>/HLA-DRB1_15_01/audit/smoke/structure_source \
  --rcsb-entity-id <PDB_entity>

MODE=structure_smoke \
OUTPUT_DIR=<build>/HLA-DRB1_15_01/audit/smoke/structure \
STRUCTURE_SOURCE_MANIFEST=<build>/HLA-DRB1_15_01/audit/smoke/structure_source/source_manifest.json \
sbatch scripts/submit_if_benchmark_v3.slurm
```

The first command runs only on a networked host and freezes the entity-instance GraphQL response
plus revision-bound RCSB mmCIF with request and byte identities. The SLURM smoke is strictly
offline: it rehashes those bytes, enumerates authoritative label/auth instance edges, materializes
each chain, and requires exact DPLM `load_coords()` sequence and length. Set
`REQUIRE_REPEATED_AUTH_ID=1` only for a preverified repeated-auth fixture.

## Production stage order

1. Run the full C1/C2 RCSB entity snapshot with `snapshot-entities`; retain Search and instance
   GraphQL request/response bytes.
2. Reuse the approved strict full-data Tier 1 preflight only by its manifest SHA. Fetch a fresh PDBe
   enriched-mmCIF residue snapshot for every source-valid mapping, with 0.1-second request spacing,
   and cross-check each detailed SIFTS map against an independent RCSB coordinate mmCIF.
3. Materialize the frozen CATH4.3 train reference. Run audited MMseqs source CATH at
   `--min-seq-id 0.3 --cov-mode 0 -c 0.8`; CATH exclusion remains strict identity `>0.30`.
4. Score the entire source-clean C2 pool with sharded length-15 NetMHCIIpan, then replay the frozen
   C5 uniform 20-bin/seed-42 law at rung 5000. A later rung is legal only from a copied, recursively
   replayed insufficient prior-rung bundle.
5. Enumerate every ordered entity/instance edge for each C5 selection and every detailed valid
   Tier 1 SIFTS candidate. Every success binds its source cache CIF/manifest; every download failure
   binds a request-failure ledger. Apply exact C6 mapping, coverage, and DPLM load gates.
6. Exact-deduplicate final Tier 2 sequences; resolve Tier 1 internal and cross-tier collisions; rerun
   final CATH; score the complete remaining C7 pool with sharded lengths 12--25 NetMHCIIpan.
7. Replay Tier 1 projection/ranking/backfill and the frozen C7 Gaussian 20-bin/seed-42 law. A
   14/15 Tier 1 or short Tier 2 cohort cannot release.
8. Assemble only the frozen primary and diagnostic membership. Run primary-only MMseqs family
   clustering and retain the exact command and cluster partition.
9. After membership freezes, annotate primary plus diagnostic rows with DRB1501 full-data
   `epoch_24.pt`, `resolved_config.yaml`, and `run_summary.json`; `best.pt` is forbidden. Run Head
   seen-pool homology at `--cov-mode 2` as annotation only.
10. Build the release evidence bundle, execute all 13 gates, write the dataset manifest, and only
    then run candidate-only finalization. Report publishability; do not move the main alias.

## Production DAG and resume

The production runner freezes one `if-benchmark-v3-production-contract/1` before any full score.
The contract fixes the build/candidate identity, 15/3000 targets, C5 ladder
`5000 -> 7500 -> 10000`, seed 42, all immutable source/tool/model/probe file identities, array
shard counts, and resource classes. `init-production-dag` accepts only the production profile;
the explicit fixture profile exists only behind the Python test API and is unavailable from the
CLI.

The DAG is fixed in this order (independent source stages can be concurrently `ready`):

1. `c1_rcsb_source`, `c8_sifts_residue_source`, and `references`;
2. `c4_source_cath` -> `c5_nmp[]` -> `c5_select`;
3. `c6_coordinate_source` -> `c6_tier2[]` plus `c6_tier1[]`;
4. `c4_final_cath` -> `c8_tier1_finalize` -> `collision_dedup`;
5. `c7_nmp[]` -> `c7_select` -> `assemble_family`;
6. `c9_head` -> `c9_homology` -> `release_validate`.

RCSB, PDBe enriched-SIFTS, and coordinate-byte acquisition stages are marked
`networked_login`; their consumers run offline. C5 NMP, C6 Tier 1/Tier 2 materialization, and C7
NMP are true SLURM arrays with bounded `max_parallel`. Every other compute stage is a one-task
array for the same journal/seal semantics.

For each stage, `write-production-task-table` freezes the complete array cardinality, argv,
producer code, input paths, output roots, and required outputs. A task may run only after every
dependency seal exists and rehashes. `run-production-task` records the exact code, DAG, task table,
input, and predecessor-seal hashes; captures stdout/stderr/argv; rejects unregistered output files;
and resumes only when every identity and output byte is unchanged. `seal-production-stage`
replays every task journal before atomically writing its seal. A failed task or legacy directory
without this identity journal is not resumable under the same build ID.

```bash
python scripts/build_if_benchmark_v3.py freeze-production-contract \
  --build-id <build_id> \
  --candidate-root <candidate> \
  --input-registry <candidate>/audit/orchestrator/input_registry.json \
  --execution-config <candidate>/audit/orchestrator/execution_config.json \
  --output <candidate>/audit/orchestrator/build_contract.json

python scripts/build_if_benchmark_v3.py init-production-dag \
  --contract <candidate>/audit/orchestrator/build_contract.json \
  --output <candidate>/audit/orchestrator/production_dag.json

python scripts/build_if_benchmark_v3.py write-production-task-table \
  --dag <candidate>/audit/orchestrator/production_dag.json \
  --stage c1_rcsb_source \
  --output <candidate>/audit/orchestrator/tasks/c1_rcsb_source.json

python scripts/build_if_benchmark_v3.py render-production-stage \
  --dag <candidate>/audit/orchestrator/production_dag.json \
  --task-table <candidate>/audit/orchestrator/tasks/c1_rcsb_source.json \
  --stage c1_rcsb_source
```

`render-production-stage` never submits. For networked stages it emits exact login task and seal
argv. For compute stages it emits the bounded `sbatch --array` argv and a second seal submission
with `afterok:${ARRAY_JOB_ID}`. `production-status` derives only `ready`, `blocked`, or `complete`
from verified seals; caller booleans cannot advance the DAG.

## Full-build resource envelope

This is an execution budget, not a scientific threshold. It is bound to the current measured
inputs and must be recalculated after C1 if the realized source size differs materially.

| stage | measured scale / throughput basis | frozen execution envelope | retained storage |
|---|---|---|---|
| C1 RCSB | about 155k entity IDs; about 620 GraphQL batches at 250 plus Search pages; 0.1 s normal request spacing | networked login, 1 CPU, 8 GB, 2 h | 1--2 GB |
| Tier 1 detailed SIFTS | approved full preflight: 74 canonical units, 2,864 valid interval mappings, 1,209 successful first attempts | networked login, 1 CPU, 8 GB, 3 h | 2--6 GB enriched/raw bytes |
| CATH reference/source C4 | actual CATH train = 16,699; `chain_set.jsonl` = 537 MB; historical source proxy = 57,127 exact groups | CPU, 8 CPUs, 32 GB, 2 h; MMseqs temp cap 10 GB | <=3 GB retained, temp removed after manifest |
| C5 NMP | historical exact groups contain 15,109,449 aa and 14,309,671 15-mers; real one-sequence probe = 86 windows in 0.894 s (startup-dominated conservative 96 windows/s) | 64 CPU shards, max 32 concurrent, 2 CPUs/8 GB/2 h each; conservative about 42 CPU-h / 1.4 h array wall | 3--5 GB |
| C6 source/IF materialization | historical chain/group multiplicity = 226,048 / 57,127 = 3.96; rung 5,000 therefore projects to about 19.8k chain attempts; existing coordinate cache mean = 759 kB/file | coordinate freeze on networked login, then Tier 2 64 shards max 32 and Tier 1 16 shards max 8; 4 CPUs/8 GB/2 h | 8--20 GB coordinates plus 0.5--1 GB clean mmCIF |
| C7 NMP | at rung 5,000 and historical mean length 264, upper proxy is about 17.3M windows; real probe = 1,155 windows in 6.47 s (179 windows/s conservative) | 64 CPU shards, max 32, 2 CPUs/8 GB/2 h; about 27 CPU-h / <1 h array wall under the conservative probe rate | 4--6 GB at rung 5,000; 8--12 GB at rung 10,000 |
| family/MMseqs | final primary = 3,015; real two-query cov0+cov2 plus native replay completed in 14 s, peak RSS 1.02 GB | CPU, 4 CPUs/8 GB/30 min | <1 GB |
| fixed Head | real two-row score = 21 s, peak RSS 1.08 GB; independent exact GPU replay = 21 s | 1 GPU, 4 CPUs/16 GB/4 h; a 100-row throughput canary is required before choosing the final walltime because the two-row result is model-startup dominated | <1 GB |

The initial production contract uses array counts `C5=64`, `C6-Tier2=64`, `C6-Tier1=16`, and
`C7=64`. CPU arrays use the `cpu` partition and `short` QoS when their calibrated shard bound is
within that QoS. Head uses one GPU under `gpu-short`; a sub-hour canary may use `gpu-test` without
an explicit partition. All SLURM logs remain under `logs/if_benchmark_v3/`.

Reserve 60 GB for a rung-5,000 candidate and 100 GB for the preregistered worst-case rung-10,000
history. These figures include immutable failed-rung evidence but exclude the external 537-MB CATH
source and installed NMP/Head models, which are hashed rather than copied. C1 realized counts,
unique coordinate PDB count, first C5 shard throughput, and the 100-row Head canary are explicit
cost gates; a material overrun pauses scheduling without changing C5/C7 membership laws.

## Audited stage commands

`scripts/build_if_benchmark_v3.py` exposes:

- `tiny-e2e`: deterministic local positive fixture executing the real gate stack;
- `snapshot-entities`: full content-bound RCSB entity/instance source;
- `probe-nmp`: official-install and native-output mini probe;
- `probe-mmseqs`: self-contained real MMseqs cov-mode 0/2 mini probe;
- `probe-head`: fixed real epoch-24 Head mini probe;
- `freeze-structure-source`: networked content-bound GraphQL/mmCIF freeze;
- `smoke-structure`: offline multi-instance mmCIF/materialization/load probe;
- `verify-nmp-probe` / `verify-structure-probe`: post-job byte rehash and semantic replay;
- `verify-mmseqs-probe` / `verify-head-annotation`: native MMseqs and fixed-Head replay;
- `freeze-production-contract`: resolve the complete registered source/tool/model/code set to
  immutable bytes before any score;
- `init-production-dag`, `write-production-task-table`, `run-production-task`,
  `seal-production-stage`, `production-status`, and `render-production-stage`: content-bound
  production DAG/array/resume machinery;
- `run-mmseqs`: exact `easy-search` producer with command/output hashes;
- `annotate-head`: fixed-epoch post-membership annotation;
- `verify-manifest`: byte-level candidate manifest replay;
- `finalize-candidate`: candidate-only immutable rename, never main publication.

Any missing real input, unresolved request, insufficient capacity, identity mismatch, or incomplete
stage remains a failed candidate. The runner never relaxes the protocol to make a release fit.
