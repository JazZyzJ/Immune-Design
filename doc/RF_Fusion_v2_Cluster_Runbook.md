# RF-Refine Fusion V2 — Cluster Runbook (state-transition Canary)

Authority: `PLAN_RF_REFINE_FUSION_V2.md`. This runbook covers everything PLAN §5.5 deferred
("Launcher and Canary commands are outside this PLAN revision and will be added to the runbook
after the local code gate"). The local code gate is complete.

**What this runbook does NOT authorize.** It takes you to a two-protein state-transition Canary and
no further. Production `D>1` stays launch-disabled (PLAN §8.4), the production
`FeedbackSupportPolicySpec` is not frozen, and no result from a Canary may be read as evidence that
feedback transmits — the Canary establishes that the mechanism EXECUTES, not that it works.

---

## 0. What must be true before step 1

| | Why |
|---|---|
| `immune-design` conda env on Della | PyTorch 2.5.1 / Python 3.12 |
| The V2 substrate is frozen and controller-free | `config.substrate` is checked field by field against the runtime sampler config **before the ladder pays its root prefix** (`rf_fusion_v2_cohort.assert_runtime_substrate_matches`) |
| Canary proteins frozen as ordinary `5ZHV_B` and anchored `Q00511` | PLAN V2F5 acceptance names both; `Q00511` uses the committed 24-anchor safety manifest and proves hard anchors never enter a support set |

Cluster paths are **always** CLI arguments. Nothing in this runbook may be hardcoded into a module.

---

## 1. Produce `B(r)` — the schedule-band calibration

This must come first: the projection policy's **reopen cardinality is not a free parameter**. It is
pinned by `B(r_d)` through `admissible_reopen_cardinality`, so the policy is not instantiable until
a real band exists.

### 1.1 Frozen Canary geometry and sampling law

These values are scientific inputs, not operator choices:

| Quantity | Frozen Canary value | Meaning |
|---|---:|---|
| source checkpoint | `c_source=50` | the one source state shared by both re-entry diagnostics |
| re-entry candidates | `r={30,40}` | `r=40` is the late intervention; `r=30` is the earlier intervention |
| propagated checkpoint | `c_next=60` | both interventions are compared after reaching the same later checkpoint |
| attempts per `(protein,r)` | `64` | enough to estimate a diagnostic central band without using mechanism outcomes |
| reported quantiles | `0.05, 0.50, 0.95` | median plus the central 90% empirical schedule band |
| capture floor | `60/64` | more than four failed captures invalidates the cell and writes no artifact |

The coordinates are frozen **before** the scan. Do not substitute another step because one of the
two declared cells is inconvenient. A missing or infeasible band is a geometry/blocking result, not
authorization to tune the Canary after observing it.

Use the actual no-remask substrate. `c1_null.yaml` is remask-on and is invalid for this scan;
`c1_constant_clean_no_remask.yaml` retains the sampler lifecycle while setting the realized
background-remask count to zero.

The ordinary and anchored proteins are two separately signed strata. Run both commands:

```bash
export PYTHONPATH=${PROJECT_ROOT}
GIT_SHA=$(git rev-parse HEAD)

python scripts/rho_maturity_scan.py --mode step \
  --checkpoint       <dplm.ckpt> \
  --rf-config        inverse_folding/reference_flow/configs/c1_constant_clean_no_remask.yaml \
  --test-set         <if_ready.parquet> \
  --pdb-root         <pdbs> \
  --proteins         5ZHV_B \
  --out-parquet      <WORK>/v2_canary/bands/5ZHV_B/rho_step_scan.parquet \
  --steps            30 40 \
  --stratum-key      5zhv_b_unconstrained \
  --quantile-levels  0.05 0.50 0.95 \
  --attempts-per-cell 64 \
  --min-captures-per-cell 60 \
  --calibration-id   band:v2:canary:5zhv_b_unconstrained:v1 \
  --calibration-json <WORK>/v2_canary/bands/5ZHV_B/B_r.json \
  --code-revision    ${GIT_SHA}

python scripts/rho_maturity_scan.py --mode step \
  --checkpoint       <dplm.ckpt> \
  --rf-config        inverse_folding/reference_flow/configs/c1_constant_clean_no_remask.yaml \
  --test-set         <if_ready.parquet> \
  --pdb-root         <pdbs> \
  --proteins         Q00511 \
  --constraint-manifest inverse_folding/reference_flow/configs/uricase_q00511_active_site_safety_v1.yaml \
  --out-parquet      <WORK>/v2_canary/bands/Q00511/rho_step_scan.parquet \
  --steps            30 40 \
  --stratum-key      q00511_anchor24 \
  --quantile-levels  0.05 0.50 0.95 \
  --attempts-per-cell 64 \
  --min-captures-per-cell 60 \
  --calibration-id   band:v2:canary:q00511_anchor24:v1 \
  --calibration-json <WORK>/v2_canary/bands/Q00511/B_r.json \
  --code-revision    ${GIT_SHA}
```

The two artifacts remain separate. The Canary is one campaign with four one-protein cells
(`5ZHV_B/Q00511` x `r=30/40`), not one config pretending that different lengths and anchor domains
share an absolute unresolved-mass distribution. Apart from protein-specific reference, constraint,
band and hotspot artifacts, the four resolved configs must carry identical Head, sampler, policy,
seed schema, `c_source=50`, `c_next=60`, and code revision.

`--code-revision` is required by the scan as well; it enters the band's provenance and the config's
`schedule_band_calibration` identity, so an unrecorded revision makes the artifact unusable
downstream.

**The runtime `stratum_key` must be the exact protein-specific key above.** `lookup_band` is exact
and refuses to interpolate. The cohort/oracle layer must not replace it with the schedule point's
generic cell label: schedule-cell identity and protein/constraint stratum are different axes.

The runtime reads each protein's stratum from `--shard-input protein_stratum_manifest=PATH`, a
`{protein_id: stratum_key}` map. It is **not** derived from the schedule's `band_key`: that names a
schedule CELL, while `stratum_key` names the cohort the band's quantiles were measured over. The
factory additionally derives each protein's realized constraint class and refuses a band calibrated
on a different one.

**The anchored protein needs its OWN stratum.** Hard anchors remove positions from the editable
domain, so at equal length an anchored protein has different unresolved mass at the same step than
an unconstrained one. A `B(r)` measured on an unconstrained cohort does not describe it, and the
reopen cardinality the policy reads off that band would be pinned from the wrong distribution. Run
the scan with a stratification that separates them and give each protein the matching runtime
`stratum_key`.

**All four predeclared cells must have non-empty support.** A cell where one protein has no legal
reopen envelope is a geometry/blocking result. Do not drop that cell, choose a nearest band, widen
the quantiles, or lower the capture floor after seeing it.

Step mode refuses a `remask.fraction_scale != 0.0` substrate: the V1 crossings were compressed into
the last ~16 steps by repeated global remasking, and PLAN §2.1 forbids reusing them to select V2
coordinates.

**Check before moving on:** `B(30)` and `B(40)` must each admit a non-empty reopen envelope for the
protein-specific stratum they sign. Otherwise the Canary is blocked at schedule calibration and
there is nothing to diagnose downstream.

---

## 2. Produce the V2 hotspot calibration artifact

`config.safety.delta_new_cumulative` requires a `HotspotCalibrationArtifact`
(`allele`, `score_scale`, `window_k_min/max`, `calibration_data_digest`, `scope`,
`reference_kind`, `reference_label`, `window_domain`).

**v0's off-halo threshold cannot be inherited and the loader enforces it.** `source_kind` must be
`measured_calibration` or `runbook_frozen`; `inherited_v0` / `default` / `placeholder` / `guess` are
refused, and `source_id` naming `objective.max_offtarget_window_increase` is refused outright —
relabelling v0's number is laundering, not calibration. `window_domain` must be `whole_landscape`;
an off-halo artifact cannot authorize the V2 gate.

Produce it with the frozen per-protein producer specified in §2.1; do **not** hand-write the
block. The Canary threshold statistic is already frozen as
`per_protein_feedback_disabled_head_valid_q90_higher`: the empirical per-protein `Q0.90` over every
**head-valid** feedback-disabled endpoint, evaluated with the `higher` order statistic. It has no
user-selectable alternative in this runbook.

Paste the emitted `delta_new` block into `config.safety.delta_new_cumulative` **verbatim**:
`source_ref` must equal `calibration_source_ref(value, unit, source_kind, source_id, artifact)`, and
the loader recomputes it — so an edited block is refused, which is the point.

### 2.1 Frozen Canary-only calibration law

This artifact authorizes **wiring of the two-protein Canary only**. It is not a production immune
safety threshold and cannot be carried into a mechanism cohort, policy qualification, capability
ladder, or holdout.

For each Canary protein independently:

1. Generate `64` complete trajectories with feedback disabled under the exact frozen V2 substrate:
   controller-free, h-map-free, `constant_one`, `n_steps=100`, temperature `1.0`, and background
   remask fraction `0.0`. Use a disjoint seed namespace
   `v2_hotspot_calibration_1`; include the Q00511 24-anchor manifest in the anchored calibration.
2. Score every candidate and its protein-specific native reference with the exact production Head.
   Compute

   $$
   N_H^{\mathrm{whole}}(y;\bar y_p)
   =\max_w\left[z_w(y)-z_w(\bar y_p)\right]_+.
   $$

   Missing, duplicate or misaligned window grids, non-finite logits, reference mismatch and anchor
   mismatch are hard failures.
3. Run the exact definitive structure gate used by the Canary on **every** endpoint and record its
   verdict and metrics in full. This is a **structure-operability DIAGNOSTIC**; it does not select
   the calibration population. A cache-only label with no resolved verdict is not a definitive
   pass, but it is also not a reason to drop the Head measurement.
4. Require at least `60/64` **head-valid** endpoints for that protein — endpoints that produced a
   valid `N_H` measurement. Below this floor, write no calibration artifact and do not relax the
   floor.
5. Sort the head-valid `N_H` values and set `delta_new_cumulative` to the empirical `Q0.90` using
   the **higher** order statistic: the value at one-indexed rank `ceil(0.90*n)`. Preserve the full
   floating-point value; do not round it before computing `source_ref`.

**Why the population is head-valid, not structure-feasible.** The Head gate and the structure gate
answer independent safety questions — whether a sequence creates a new immune hotspot, and whether
it preserves the target backbone and active-site geometry. Conditioning the Head null on the
structure verdict couples them, and does so at exactly the wrong moment: a low structure-pass rate
shrinks the estimation sample precisely when structure is hardest, and a `Q0.90` over ~19 points is
close to its own second-largest value. The executed Canary made it concrete — `Q00511` preserved
hard anchors `64/64` and passed scTM `64/64`, rejected only because a predicted side-chain RMSD
exceeded an absolute band whose own native baseline under the identical protocol is `1.791 Å`.
Nothing in that verdict makes the sequence's Head hotspot unmeasurable. The structure rate is
reported beside the threshold as `structure_operability` and never inside it.

**Hard-anchor identity preservation is untouched and remains an absolute gate**: an anchor mismatch
is a hard failure and never enters the population. Only the *side-chain RMSD* band is a
recalibration question, and only for a later stage (§7).

The two proteins receive separate thresholds and separate artifacts. Do not pool endpoints across
proteins and do not take a cohort-level maximum: sequence length, window multiplicity and the anchor
domain change the distribution of a whole-landscape maximum. The Canary does not compare their
pass rates.

The producer must be `scripts/calibrate_rf_fusion_v2_hotspot.py`, reuse the shared model factory,
production Head batch scorer, V2 `measure_cumulative` primitive and v0 definitive structure seam,
and be registered in `doc/SCRIPTS.md`. It writes a canonical raw table and one typed artifact per
protein. Its frozen interface is:

**This producer is a required implementation unit before cluster execution.** Until the script,
tests and `doc/SCRIPTS.md` registration exist, stop here; a hand-written scalar or ad-hoc notebook
does not satisfy the calibration contract.

```bash
python scripts/calibrate_rf_fusion_v2_hotspot.py \
  --protein-id <5ZHV_B|Q00511> \
  --n-completions 64 \
  --master-seed 20260806 \
  --seed-namespace v2_hotspot_calibration_1 \
  --threshold-statistic per_protein_feedback_disabled_head_valid_q90_higher \
  --min-head-valid 60 \
  --checkpoint <dplm.ckpt> \
  --rf-config inverse_folding/reference_flow/configs/c1_constant_clean_no_remask.yaml \
  --test-set <if_ready.parquet> \
  --pdb-root <pdbs> \
  --complete-reference-manifest <WORK>/v2_canary/references.json \
  --head-config-dir <head-config-dir> \
  --head-checkpoint <head.ckpt> \
  --structure-config <v0-structure-config> \
  [--constraint-manifest inverse_folding/reference_flow/configs/uricase_q00511_active_site_safety_v1.yaml] \
  --out-rows <WORK>/v2_canary/hotspot/<PID>/calibration_rows.parquet \
  --out-json <WORK>/v2_canary/hotspot/<PID>/hotspot_calibration.json \
  --code-revision $(git rev-parse HEAD)
```

Omit `--constraint-manifest` for `5ZHV_B`; supply the exact path shown for `Q00511`.

`--threshold-statistic` is a closed enum for this Canary, not a free-form label. The producer must
interpret `per_protein_feedback_disabled_head_valid_q90_higher` exactly as step 5 above and reject any other
value. Do not also expose independent `--quantile` or `--quantile-method` overrides, because they
would create two conflicting sources of truth.

The canonical row projection signed by `calibration_data_digest` contains at least protein ID,
replicate index, realized seed, sequence MD5, native-reference digest, Head evaluator/window-grid
digests, `N_H`, definitive structure verdict/metrics and anchor verdict. The artifact additionally
reports `n_attempted=64`, `n_head_valid`, `q50/q90/q95/max`, the order-statistic rank, all failure
counts and the `structure_operability` diagnostic block. A producer that outputs only the chosen scalar is incomplete.

Use `source_kind=measured_calibration` and a protein-specific `source_id` of
`v2-canary-hotspot-head-valid-q90-higher-v1:<protein_id>`. The config value, typed artifact and canonical
`source_ref` are copied verbatim from this JSON; they are never typed manually.

### 2.2 The threshold's unit is `raw_logit`, and the grid is 12–25

**`score_scale: raw_logit`, not `nats`.** The frozen Head emits UNCALIBRATED classifier logits.
Absent an explicit log-probability/calibration transform, calling them nats is scientifically
wrong — and `OnlineHeadScorer` refuses to be built under any other scale, so a `nats` declaration
could never bind. The hotspot threshold is empirically calibrated on this same frozen Head, so a
raw-logit unit is legitimate: nothing about the number, the ordering or the Canary's mechanism
changes, only the declared unit becomes the one the evaluator actually produces. Three fields carry
it, and **only the first is typed by a human**:

```yaml
head:
  score_scale: raw_logit          # <- declared
safety:
  delta_new_cumulative:
    unit: raw_logit               # <- GENERATED by the producer from the realized Head identity
    artifact:
      score_scale: raw_logit      # <- GENERATED likewise
```

Copy the whole `delta_new_cumulative` block verbatim from `hotspot_calibration.json` and let the
loader recompute `source_ref`. **Never relabel an old `nats` threshold and carry it forward**:
`calibration_source_ref` digests `unit` *and* the artifact payload, and `config.py` separately
requires `delta_new_cumulative.unit == head.score_scale` — so a block edited on one side is refused
outright and a block edited on both no longer matches its own digest. The threshold has to be
re-measured on the Head that will enforce it.

**One window grid everywhere: `window_k_min=12`, `window_k_max=25`** — the V2 calibration, the
reference scoring and the Canary alike.

12 is not a preference, it is what the frozen Head emits: `epitope_head/configs/inference.yaml`
declares `min_k: 12, max_k: 25`, and the window grid comes from the Head's own predictor.
`window_k_min/max` are **metadata** — `OnlineHeadScorer` stores them and echoes them into its cache
identity, but never enumerates or filters on them. So a narrower declaration does not narrow the
grid; it makes `whole_landscape_new_hotspot` reject every window outside it, i.e. **every
endpoint**. A declared `13` is a guaranteed 0-of-N on every protein, not a flaky failure — it cost
one smoke allocation to find, and `calibrate_rf_fusion_v2_hotspot` now checks the declared domain
against the reference's realized windows before drawing a single completion.

Sharing V1's grid inherits no threshold: the V2 value is re-measured from scratch under §2.1's
frozen law. What may never be inherited is a **number**, not a grid.

---

### 2.3 EXECUTED (2026-08-06, `2c7af9a`) — `READY_FOR_V2_CANARY`

Both proteins passed at the frozen law: 64 completions, `min_head_valid=60`,
`per_protein_feedback_disabled_head_valid_q90_higher`, grid 12–25, `raw_logit`.

| | `5ZHV_B` (unconstrained) | `Q00511` (anchor24) |
|---|---:|---:|
| head-valid | **64/64** | **64/64** |
| `delta_new_cumulative` | `15.737810611724854` | `20.082843780517578` |
| order-statistic rank | 58 of 64 | 58 of 64 |
| `N_H` q50 / q90 / max | 5.966 / 15.738 / 18.486 | 16.587 / 20.083 / 20.627 |
| hard anchors preserved | n/a | 64/64 |
| **structure operability** (diagnostic) | 32/64 = 50.0% | 19/64 = 29.7% |

`source_ref` recomputes for both; a resolved config carrying either block is accepted by
`load_v2_config`. Steps 3–5 are authorized.

#### The structure-operability finding — carried forward, not resolved

The rates above are a real result and they are **not** a hotspot-calibration failure. They bind on
different gates and neither is a wiring fault:

- **`5ZHV_B`** — the design scTM distribution is centred essentially ON the `scTM_min=0.85` gate
  (median `0.8528`), so it is cut in half by construction. The native reaches `0.9109` under the
  identical protocol, so the gate is not unreachable; de novo completions simply sit ~0.06 below it.
- **`Q00511`** — backbone quality is not the constraint: **every** design passes scTM (median
  `0.9568`, native `0.9764`) and 45 are rejected by the anchor side-chain band alone. The native's
  own worst anchor side chain reconstructs at **`1.791 Å`** against the `2.0 Å` band — `0.209 Å` of
  headroom — while design medians sit at `2.386 Å`.

`scTM_min=0.85` and `max_anchor_sidechain_RMSD_max=2.0` come from v0's
`rf_refine_fusion_final_repair_beam.yaml`, calibrated for **WT-seeded local refine**, where side
chains barely move. §2.1 step 1 specifies **backbone-only de novo completion**. The conjunction has
never been calibrated against that regime.

Distance to each gate, as measurement only — **nothing here authorizes changing a frozen value**:

| `5ZHV_B` scTM_min | 0.80 | 0.82 | 0.84 | **0.85** | 0.86 |
|---|---:|---:|---:|---:|---:|
| pass /64 | 58 | 50 | 39 | **32** | 26 |

| `Q00511` anchor max (Å) | **2.0** | 2.2 | 2.5 | 2.8 | 3.0 |
|---|---:|---:|---:|---:|---:|
| pass /64 | **19** | 28 | 35 | 45 | 52 |

**How this must be read downstream.** For the first state-transition Canary the existing structure
gate stands unchanged as an independent admission gate — both proteins have real passing endpoints,
so it does not block the mechanism test. A structure rejection is **not** evidence of feedback
failure and may not be reported as one. Before any capability or production stage, the V2 de novo
structure policy needs its own calibration: v0's WT-seeded `2.0 Å` may not become V2 production
authority unverified. **Hard-anchor identity preservation stays an absolute gate and is not in
scope for that recalibration; only the side-chain RMSD band is.**

Artifacts: `<WORK>/v2_canary/{bands/<PID>/{rho_step_scan.parquet,B_r.json},
hotspot/<PID>/{calibration_rows.parquet,hotspot_calibration.json}, references.json}`. The
structure-conditioned rows from the superseded law are kept beside them as
`calibration_rows.PREV_structure_conditioned.parquet`; `n_h_whole` is bit-identical across the two,
so the population change altered no measurement.

---

## 3. Fill the Canary config

The Canary campaign contains TWO proteins, so it needs TWO references even though each execution
cell is a one-protein job. `complete_reference_manifest` is a
`{protein_id: path}` JSON map resolved relative to itself:

```json
{
  "5ZHV_B": {"path": "5ZHV_B.seq", "sha256": "<sha256 of 5ZHV_B.seq>"},
  "Q00511": {"path": "Q00511.seq", "sha256": "<sha256 of Q00511.seq>"}
}
```

The digest is **per protein and lives in the manifest**, because
`content[complete_reference_sequence].expected_sha256` is a single value and a two-protein cohort
has two references — one config field cannot sign both. That role's digest therefore signs the
**manifest**, and the manifest signs each sequence, so every byte is still bound and nothing is
signed twice under one number. A file edited after the manifest was written is refused.

A bare path (or a bare sequence file) is refused by type. One file for the cohort binds every protein's
whole-landscape hotspot comparator to the same native: at different lengths the Head binding fails
outright, and at equal lengths — the dangerous case — it succeeds while measuring each design
against another molecule.

Start from `inverse_folding/reference_flow/configs/v2_canary_state_transition.yaml`. Every
`REPLACE_*` is a value the file cannot carry; the loader has no default and refuses the run.

**Do not fill it by hand.** `scripts/materialize_v2_canary_config.py` resolves one cell, refuses
the three ways a config can run and still be wrong (a threshold measured under a different Head
domain, a band with no cell at this `(r_step, stratum_key)`, a surviving placeholder), loads its own
output back through `load_v2_config_file`, and writes `<CELL>.args.sh` — the driver's exact
`--input-file`/`--shard-input` vectors. Those vectors must agree with the frozen/runtime split the
config declares; emitting both from one place is what keeps them agreeing. The notes below are why
each field is what it is, not a filling procedure.

Eight that bite:

1. **`identity.code_revision`** — 7–64 lowercase hex. `unknown` is refused. The CLI's
   `--code-revision` defaults to this value and must equal it, so no flag can sign a run under an
   undeclared revision.
2. **`content[schedule_band_calibration].expected_sha256` is the TABLE's
   `calibration_content_digest`, NOT the file's sha256.** `make_band_table` rebinds it to a
   canonical digest of the table's content, and that is what `q_phi` compares against
   `conditioning.schedule_band_calibration`. Read it off the artifact:
   `python -c "import json;print(json.load(open('B_r.json'))['provenance']['calibration_content_digest'])"`.
   Do **not** pass this role to `--input-file`: that computes the file's sha256 and refuses. Pass
   the path with `--shard-input schedule_band_calibration=...` instead.
3. **`content[complete_reference_sequence].expected_sha256` is THIS CELL'S OWN `.seq`, not the
   manifest.** A Canary cell is one protein, and `bind_cumulative_reference` compares the digest it
   recomputes from the reference BYTES against `policy.complete_reference_content_digest` — which
   IS this field. A manifest digest there can never equal a sequence digest, so the shard dies with
   `ReferenceRebindAttempt` after the checkpoints are already resident (measured, 2026-08-06). The
   cohort manifest binds separately, as the plural `reference_sequences` role plus the
   `complete_reference_manifest` shard input; between them every byte is still signed and nothing is
   signed twice under one number. Each sequence file must be **the canonical sequence itself: no
   FASTA header, no wrapper, no trailing newline** — any container puts bytes in the file that are
   not in the sequence and the two numbers can never agree. Command:
   `--input-file complete_reference_sequence=<WORK>/v2_canary/<PROTEIN_ID>.seq`.
4. **`arm.a2_matching_resource`** — one of five, and the choice IS the experiment. At the measured
   unit costs one refold ≈ 81 DFE and structure is ~94% of a cycle's marginal cost, so matching on
   `logical_dfe` and matching on `definitive_refolds` are different experiments. The other four
   components are reported unmatched, never netted away.
5. **`content[projection_policy_spec].expected_sha256`** — the sha256 of the committed spec FILE
   `inverse_folding/reference_flow/configs/v2_state_derived_probe_policy_v1.json`, which describes
   the already-implemented `state_derived_probe` rule and introduces no parameter. It deliberately
   carries no `stratum_key`, no `B(r)` value or digest, no protein id and no `r_step`: those are
   per-cell RUN configuration, so folding them in would make the spec cell-specific and **no single
   frozen file could sign the four-cell campaign**. The policy's own identity keeps the two apart —
   `policy_spec_digest` is this file (invariant across cells, and what the kernel matches against
   `conditioning.projection_policy_spec`), while `policy_config_digest` carries the per-cell
   `stratum_key` + band binding. Both were previously one canonical dict digest, which could never
   equal a file sha256 and so could never pass.

   ```bash
   POLICY_SPEC=${PROJECT_ROOT}/inverse_folding/reference_flow/configs/v2_state_derived_probe_policy_v1.json
   POLICY_SPEC_SHA=$(sha256sum "${POLICY_SPEC}" | cut -d' ' -f1)
   # config:  - role: projection_policy_spec
   #            binding: frozen
   #            expected_sha256: ${POLICY_SPEC_SHA}
   # command: --input-file projection_policy_spec="${POLICY_SPEC}"
   ```
6. **`head_config` and `head_checkpoint` must be FROZEN, and their digests are the ones the scorer
   itself reports.** They are declared runtime in the template, but `build_v2_oracles` reads both
   through `_content_digest`, which refuses a runtime role — and the numbers are not decoration:
   `bind_cumulative_reference` stamps the reference's `HeadScoreBinding.evaluator` from the
   CONFIG's identity while every endpoint's binding carries the SCORER's, and `bind_head_scores`
   compares the two for equality. A declared digest that merely looks plausible therefore fails
   every endpoint, not just the provenance row. `head_config` is a DIRECTORY: its identity is the
   three-YAML rolling hash `run_if_phase_c1._compute_head_config_hash` computes over
   `model.yaml` + `model_ablation.yaml` + `inference.yaml`, which is why it can never be passed to
   `--input-file`.
7. **An unconstrained cell still has to DECLARE its lack of constraints.** `constraint_manifest`
   and `fixed_token_policy` are two of the eighteen roles `_conditioning` requires, and PLAN §5.2
   grants them no exemption — a run whose conditioning is simply silent about constraints cannot be
   told apart from one whose manifest went missing. `rho_maturity_scan` already answered this with a
   typed ABSENCE digest, and `5ZHV_B`'s own `B(r)` artifact carries it, so the `5ZHV_B` cells freeze
   both roles to that same number
   (`canonical_digest({"schema": "scan-constraint-binding/1", "constraint_manifest": null})`) and
   pass **no** `constraint_manifest` shard input, which is what makes v0 build unconstrained
   oracles. Being frozen, it also refuses any `--input-file constraint_manifest=...` on those cells.
   The `Q00511` cells bind both roles runtime, to the safety24 manifest file.
8. **`structure_backend` is the weight file, not the backend's name.** `esmfold2_live` is code and
   is already covered by `code_revision`; what decides every fold is the ESMFold2 snapshot's
   `model.safetensors`, so that file's sha256 is the role's identity. `code_revision` itself is
   frozen to the git revision — `require_digest` accepts it precisely because a revision is a legal
   content identity that is not a SHA-256.

`config.safety.incremental_gate_enabled` may now be either value. At depth 0 the gate has no
referent and the admission records a typed `IncrementalInapplicable` minted only from a parentless
ledger; it becomes binding from depth 1. If you enable it, `delta_new_incremental` is required and
must carry its own `immediate_parent`-scoped artifact.

Materialize four resolved configs under `<WORK>/v2_canary/resolved_configs/`; do not edit the
tracked template in place:

| Cell | Protein | `r_step` | Runtime stratum | Band artifact | Hotspot artifact | Constraint |
|---|---|---:|---|---|---|---|
| `5zhv_r30` | `5ZHV_B` | 30 | `5zhv_b_unconstrained` | `bands/5ZHV_B/B_r.json` | `hotspot/5ZHV_B/hotspot_calibration.json` | none |
| `5zhv_r40` | `5ZHV_B` | 40 | `5zhv_b_unconstrained` | same | same | none |
| `q00511_r30` | `Q00511` | 30 | `q00511_anchor24` | `bands/Q00511/B_r.json` | `hotspot/Q00511/hotspot_calibration.json` | Q00511 safety24 |
| `q00511_r40` | `Q00511` | 40 | `q00511_anchor24` | same | same | Q00511 safety24 |

Every config keeps `c_source_step=50`, `c_next_step=60`, `n_lookaheads=4`, `depth_cap=1`,
`incremental_gate_enabled=false`, and `a2_matching_resource=definitive_refolds`. The
protein-specific runtime stratum must come from an explicit cohort/shard binding; it must not be
derived from `DepthSchedulePoint.band_key`. The latter is a schedule-cell identity, not a
length/constraint stratum. This separation is a pre-launch code acceptance condition.

---

## 4. Preflight — costs nothing, refuses early

Run the following template once for each row of the four-cell table:

```bash
python scripts/run_rf_fusion_v2.py \
  --v2-config    <WORK>/v2_canary/resolved_configs/<CELL>.yaml \
  --out-dir      <RUN>/v2_canary/<CELL> \
  --cohort       <PROTEIN_ID> \
  --input-file   dplm_checkpoint=<dplm.ckpt> \
                 complete_reference_sequence=<WORK>/v2_canary/references.json \
                 <...one ROLE=PATH per PLAN §5.2 role...> \
  --shard-input  schedule_band_calibration=<WORK>/v2_canary/bands/<PROTEIN_ID>/B_r.json \
                 complete_reference_manifest=<WORK>/v2_canary/references.json \
                 protein_stratum_manifest=<WORK>/v2_canary/strata.json \
  --dry-run
```

There is no `stratum_key` shard input: `build_v2_oracles` resolves the stratum through
`resolve_stratum(protein_stratum_manifest, protein_id)`, so passing one as a path would be a name
nothing reads while looking like the thing that binds the band. `strata.json` is
`{"5ZHV_B": "5zhv_b_unconstrained", "Q00511": "q00511_anchor24"}`.

In practice, source the cell's generated `<CELL>.args.sh` and expand `INPUT_FILES`/`SHARD_INPUTS`
rather than retyping either vector.

Loads **no model**. It resolves the config, derives the one shared `config_digest`, signs every
declared input by CONTENT, and projects the run's logical cost against the declared caps —
`root_capture + screening + segment + descendant pools`, reconciled in the test suite against a
realized ladder's counted forward passes.

A run that declares **no** inputs is refused (`exit 4`), on this path and on the execution path
alike: PLAN §5.2's "missing content identity fails closed".

**`--dry-run` is not sufficient, and knowing why matters.** It loads no model, so it returns before
`build_v2_oracles` — and that is where the launch-blocking faults live. Four consecutive cluster
attempts passed `--dry-run` and then died inside the assembly, each after the allocation and the
3 GB checkpoint load. Run the second gate on every cell too; it needs no GPU and takes seconds:

```bash
python scripts/preflight_v2_canary_assembly.py \
  --cell <WORK>/v2_canary/resolved_configs/<CELL>.yaml=<PROTEIN_ID> ...
```

It exercises the real band load and stratum match, `lookup_band`'s envelope at this `r_step`, the
depth-0 reference BINDING, the admission policy and safety gate, the support policy's identity
against the declaration, and all eighteen roles resolving through the launcher's own `SHARD_INPUTS`.
Only the two torch builds are stubbed — and the stub is checked by AST against `PreparedModel`'s
real surface, so a renamed attribute fails here instead of on the GPU.

Exit codes are the whole interface a cohort runner sees: `0` complete and at least one success;
`2` nothing usable, an empty cohort, every protein failed, **or a realized cohort cap breached**;
`3` partial; `4` not launchable.

---

## 5. Run the Canary

For the same cell, drop `--dry-run` and add the remaining runtime paths the oracle stack needs:

```bash
  --shard-input  base_if_checkpoint=<dplm.ckpt> \
                 rf_sampler_config=inverse_folding/reference_flow/configs/c1_constant_clean_no_remask.yaml \
                 test_set_parquet=<if_ready.parquet> \
                 pdb_root=<pdbs> \
                 schedule_band_calibration=<WORK>/v2_canary/bands/<PROTEIN_ID>/B_r.json \
                 complete_reference_manifest=<WORK>/v2_canary/references.json \
                 protein_stratum_manifest=<WORK>/v2_canary/strata.json \
                 [constraint_manifest=inverse_folding/reference_flow/configs/uricase_q00511_active_site_safety_v1.yaml] \
                 journal_dir=<RUN>/v2_canary/<CELL>/journals \
                 device=cuda \
                 <...one <role>=PATH for every RUNTIME-bound content role...>
```

`scripts/submit_rf_fusion_v2_canary.slurm` is this command: `CELL=<cell> sbatch ...` sources that
cell's `args.sh` and adds only `journal_dir` and `device`, which belong to the job rather than the
cell. It must run under `immune-design-blackwell` on `rtx6000` — the env the band scan and the
hotspot calibration ran under. It also **exports** `PROJECT_ROOT`: the DPLM checkpoint's stored
hydra config resolves paths through `oc.env:PROJECT_ROOT`, so a merely-set shell variable is
invisible to omegaconf and the checkpoint load dies after the GPU has been allocated.

`--shard-input` is `NAME=PATH`. A bare path or a repeated name is refused — guessing which
parameter a path meant is how a checkpoint ends up passed as a PDB root. Every **runtime-bound**
content role must be supplied here: its digest is computed from the bytes you actually give, and a
role that is neither frozen in the config nor supplied fails closed.

Omit `constraint_manifest` in both `5ZHV_B` cells. A four-cell launch is complete only when all
four resolved configs pass dry-run under the same code revision and are submitted before any
feedback outcome is opened.

SLURM: follow `scripts/submit_benchmark.slurm`. `--output/--error` go to `logs/`, never `run/`.

---

## 5.1 EXECUTED (2026-08-06, `4dd0922`) — `WIRING_PASS_WITH_POPULATION_MISMATCH_DIAGNOSTIC`

**The Canary is CLOSED and will not be rerun.** Its one question was whether the mechanism
executes, and both proteins produced at least one legal transition with every state, anchor,
lineage and assimilation invariant closed. The population mismatch below belongs to the DEFINITION
of the next experiment's population, not to whether the Canary passed.

`5zhv_r30` is retained as the real diagnostic it is — of the structure gate and of sibling
correlation. Re-seeding it to obtain a fourth green cell would be post-outcome rescue: it adds no
wiring evidence and only makes the table look better.


Jobs `12094327–30`, `rtx6000` / `immune-design-blackwell`, ~2 min and 8.6–35.9 GPU-s per cell.
Read with `scripts/analysis/read_v2_canary.py`; report at
`<WORK>/v2_canary/canary_read_v1.json`.

| cell | outcome | write/inject/reopen/carry | Σ | definitive | §6 |
|---|---|---|---:|---:|---|
| `5zhv_r30` | `null_no_admissible_endpoint` | — | — | 0/4 | non-null FAILS; the rest not reached |
| `5zhv_r40` | **`committed`** | 1 / 9 / 1 / 91 | **102** | 1/8 | all pass |
| `q00511_r30` | **`committed`** | 1 / 10 / 50 / 217 | **278** | 4/8 | all pass |
| `q00511_r40` | **`committed`** | 1 / 10 / 21 / 246 | **278** | 2/8 | all pass |

`Σ` is the editable domain, exactly: 102 for `5ZHV_B`, and **278 = 302 − 24** for `Q00511` — hard
anchors are in no support class, by construction and now by measurement.

**The band pinned the reopen cardinality, and the coupled identity closes to the digit.**
`u_proj = u_src − a + b_new` with `a = 1` in every cell:

| cell | `u_src(c50)` | reopened | `u_proj(r)` | `B(r)` unresolved | `ρ_proj` | `B(r)` ρ |
|---|---:|---:|---:|---|---:|---|
| `5zhv_r40` | 55 | 1 | **55** | [53, 70] ✓ | 0.461 | [0.315, 0.480] ✓ |
| `q00511_r30` | 138 | 50 | **187** | [186, 208] ✓ | 0.327 | [0.253, 0.330] ✓ |
| `q00511_r40` | 138 | 21 | **158** | [157, 179] ✓ | 0.432 | [0.357, 0.435] ✓ |

Every §6 invariant holds wherever a transition occurred: anchors 24 × 8 endpoints against WT with
**0 violations** in both `Q00511` cells; lineage source → endpoint → projected → propagated closed;
injected positions `pending_assimilation` with a **null** score at projection and `assimilated` with
a finite one after the first propagation; every temporary-protection grant expiring at `c_{d+1}`
exactly; the endpoint's own `evidence_logprob` never equal to an active sampler score; ledger
`physical_cost_complete` with all six phases and no breached cap. `a2_matched_extra_lookaheads = 0`
in all four cells, so **no Head contrast from this Canary is licensed** and the reader refuses to
compute one.

Per §6's own wording the non-null condition is per PROTEIN: both proteins transitioned.
`5zhv_r30` is a cell that admitted nothing, which is the next section.

#### The structure gate is measured on a different population than it enforces on

`5zhv_r30`'s four lookaheads scored scTM `0.784 / 0.788 / 0.810 / 0.812` against a `0.85` floor;
`5zhv_r40`'s single admitted endpoint scored `0.8516` — **1.6 × 10⁻³ above the gate**. Recovered
from the refold cache (`{protein_id}_{sha256(seq)[:12]}`), because a failed verdict is recorded on
the archive row and the endpoint keeps `structure_evaluated=false` with empty metrics by design.

| population | n | scTM median | pLDDT median | ≥ 0.85 |
|---|---:|---:|---:|---:|
| calibration, full de novo trajectory | 64 | **0.8528** | 71.7 | 32 (50%) |
| Canary depth-0 lookahead (resumed at `c=50`) | 8 | **≈0.798** | ≈65 | 1 (12.5%) |
| Canary descendant (after projection → `c=60`) | 4 | **≈0.712** | ≈64 | 0 |

The calibrator draws one complete trajectory per replicate (`sampler.sample`, step 0 → 100); the
Canary's endpoints are completions RESUMED from a captured source state. Those are different
populations, and they measure differently. Two consequences, stated at the strength the evidence
supports:

1. The 50% structure-operability figure recorded in §2.3 does not describe the Canary's endpoints.
   A cell's four lookaheads also share a 50-step prefix, so they are correlated and
   "no admissible endpoint" is a per-cell coin-flip rather than a rare tail.
2. **Now measured — see §5.2.** The hotspot threshold was calibrated on the full-trajectory
   population and enforces on the resumed one, and the two differ on the Head axis as well.

Neither licenses reading a structure rejection as feedback-mechanism failure (§2.3 stands), and
neither is authority to change `n_lookaheads`, the band, or the gate.

#### What follows from it (owner's ruling, 2026-08-06)

The structure axis is now a MEASURED mismatch, so the mechanism stage gets its own null measured on
the population it will actually enforce on. The Head axis is only a suspicion; it is answered
cheaply first, from the bundles already on disk.

1. **Recompute `N_H^whole` on the existing bundles** — every depth-0 lookahead (admitted AND
   rejected) and every propagated descendant, through the same Head, the same 12–25 grid and the
   same native reference. A diagnostic, not a recalibration.
2. **Failed structure evidence must land on disk.** A separate `structure_evaluations` artifact
   records every endpoint regardless of verdict — id, metrics, gate verdict and reason, cache and
   model provenance. Endpoint state semantics are unchanged; what changes is that a cell which
   admitted nothing can say WHY from its own bundle instead of through the refold cache.
3. **A resumed-null calibration, with the SOURCE PREFIX as the sampling unit**: per protein,
   **32 independent `c=50` prefixes × 2 feedback-off completions = 64 endpoints**. Sixty-four
   siblings of one prefix are not sixty-four samples. It measures `N_H^whole`, scTM/pLDDT, the
   `Q00511` anchor side-chain metric, and within-source vs between-source variance. `r` does not
   affect the depth-0 endpoint population, so it is not repeated per `r`.
4. **Mechanism-stage gates, frozen from that null.** The `Q00511` side-chain gate stops being an
   absolute number: the native scores **1.791 Å** under the same backend, so of a 2.0 Å band only
   0.209 Å is anything but backend and rotamer error. Admission moves to the native-relative excess

   $$
   \Delta_{\mathrm{anchor}}(y) = \mathrm{RMSD}_{\mathrm{anchor}}(y) - \mathrm{RMSD}_{\mathrm{anchor}}(\mathrm{native})
   $$

   thresholded at `Q0.90(higher)` of the resumed feedback-off null; the ordinary protein's
   `scTM_min` becomes `Q0.10(lower)` of the same null. **Hard-anchor residue identity remains an
   absolute hard gate** and raw absolute RMSD is still reported in full. These are
   **mechanism-operability gates for the transmission experiment only** — they may not be carried
   into capability or holdout, where the production structure gate still has to be frozen
   independently.
5. **The descendant drop is not yet causal.** `0.798 → 0.712` compares different stages of
   different populations. The mechanism cohort must compare a feedback-on descendant against a
   MATCHED feedback-off descendant at identical source, support, fork seed and propagation horizon.
   Both drop → the resumed / no-remask generation geometry. Only feedback-on drops → projection or
   policy is damaging structure, and `D>1` stays shut. No stable difference → these four
   descendants were small-sample.
6. `D>1` remains launch-disabled throughout.

---

## 5.2 `N_H^whole` on the resumed population (2026-08-06, `9c8b042`) — the Head gate does not bite here

`scripts/analysis/recompute_v2_nh_whole.py`, all 28 Canary endpoints, same Head, k = 12–25, same
native reference, through `whole_landscape_new_hotspot` — the function the admission gate itself
calls. Artifacts: `<WORK>/v2_canary/nh_recompute/{canary_endpoints_nh.parquet,summary.json}`.

| | full-trajectory null (n=64) | resumed, all | depth-0 lookahead | descendant |
|---|---:|---:|---:|---:|
| **`5ZHV_B`** q50 / q90 / max | 5.966 / **15.738** / 18.486 | 2.037 / 10.550 / 11.699 | 1.319 / 11.699 / 11.699 | 2.037 / 7.893 / 7.893 |
| | | n=12 | n=8 | n=4 |
| **`Q00511`** q50 / q90 / max | 16.587 / **20.083** / 20.627 | 14.767 / 19.894 / 19.899 | 13.551 / 18.371 / 18.371 | 14.881 / 19.899 / 19.899 |
| | | n=16 | n=8 | n=8 |

**Resumed endpoints are uniformly LESS hot than the population the threshold came from, and
`0/12` and `0/16` reach it.** The mismatch is real on both axes, but it points opposite ways:
structure is calibrated LOOSE and rejects most of what it sees, while the Head is calibrated HIGH
and rejects nothing. On `5ZHV_B` the entire resumed range (max 11.699) sits below the threshold
(15.738) — the gate cannot fire at this operating point. `Q00511` is closer (max 19.899 vs 20.083)
but still under.

Two things follow, and a third does not:

* The Head gate is **currently non-binding in the mechanism regime**. That is safe in the sense
  that it will not wrongly reject; it is not safe in the sense of providing protection, because a
  gate that never fires is not evidence that nothing was caught.
* Structure quality and hotspot creation are **not co-monotone** here: resumed endpoints fold worse
  AND create fewer new hotspots than full trajectories. Whatever the prefix constrains, it
  constrains both — and not in the same direction.
* It does **not** follow that feedback lowers `N_H`. `n = 12` and `16`, from two source prefixes
  each, with no matched control. Descendants are marginally higher than depth-0 on both proteins;
  that is an observation, not a direction.

The live Head re-scored every design and agreed with the stored artifact to `0.00e+00` (`5ZHV_B`,
102 aa) and `7.07e-03` (`Q00511`, 302 aa) raw logits — reduction-order noise in a bf16 ESMC-6B
stack without fused kernels, length-dependent as expected, and four orders below the thresholds.
Bit-equality is the wrong bar here and the tolerance says so explicitly.

---

## 6. What to read, and what NOT to conclude

Read **only** these, per PLAN §8.3:

| Check | Where | Pass condition |
|---|---|---|
| **Non-null transition** | `feedback_events.outcome` | at least one `committed` per protein. A cohort of `null_invalid_policy_result` means the policy could not answer against the realized state — that is a fact about `B(r)` or the coordinate, not about feedback |
| **Anchors** | `partial_states.hard_anchors_json` vs `complete_endpoints.sequence` | every anchor preserved in the anchored protein's endpoints |
| **Replay** | `partial_states.replay_state_hash`, `complete_endpoints.replay_*` | present and distinct per fork |
| **Lineage** | `partial_states.parent_state_id` / `parent_transition_id` / `origin_endpoint_id` | the chain source → endpoint → projected → propagated closes |
| **Assimilation (4 checks, not 1)** | `partial_states.active_score_status_json` / `active_sampler_score_json` / `temporary_protection_json` / `provenance_json` | see the four rows below |
| — at projection | projected state's injected positions | status `pending_assimilation`, and their `active_sampler_score` is **null**, not a number |
| — after the first propagation forward | propagated state at `c_{d+1}` | those positions have become `assimilated` and now carry a **finite** active sampler score |
| — protection expiry | `temporary_protection_json` on the projected state vs `expired_protection_json` on the propagated one | every grant expires at `c_{d+1}` exactly — not before, not carried past |
| — the two evidence namespaces stay separate | `provenance_json[*].last_origin.evidence_logprob` vs `active_sampler_score_json` | the endpoint's completion log-prob is immutable provenance and must **never** appear as an active sampler score. If it does, the reopen/ranking signal is being driven by the endpoint's own evidence and the mechanism is circular |
| **Ledger** | `cost_ledger.jsonl`, `run_manifest.realized_caps` | phases present; `physical_cost_complete` true; no `unknown_after_start` you cannot explain |

**Do not compute a Head-score contrast from a Canary.** It has no power and no matched control.
`a2_views.matched_extra_lookaheads` is the number that would make an A2 comparison legitimate, and
until it is non-zero the two arms are not compute-matched.

---

## 7. After the Canary

1. A small **one-cycle mechanism cohort**, keeping a trimmed `endpoint_change` + `source_shuffle` +
   `feedback_off`. `source_change` (fixed-support ablation) is admissible only where arm A carried a
   resolved position; where it is not, the row says so in its own words and `source_shuffle` carries
   PLAN §2.6's intervention 2.
2. Only if **source transmission** holds: freeze the production `FeedbackSupportPolicySpec`
   (PLAN §2.5's eight required elements) and only then open `D>1`.

`allow_production_depth_gt_1` is never granted by the oracle factory. Opening it is a runbook
decision that also requires the FeedbackSupportPolicy directionality gate.

---

## 8. Known-open, recorded rather than worked around

- `active_population_width > 1` is refused at config parse — Interface Map OQ7 leaves open whether a
  forked family inherits its parent's `reference_binding_id` or opens its own.
- `reward_ordered` / `policy_source_off` mechanism views raise `NotYetFrozenError` until the
  production policy spec is frozen (PLAN §8.4); a placeholder would yield a number that looks like
  evidence.
- Nothing in V2 has run on a cluster. Every local suite uses fake oracles and validates **wiring
  only**.
