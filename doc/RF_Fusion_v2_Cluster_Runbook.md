# RF-Refine Fusion V2 — Cluster Runbook (state-transition Canary)

Authority: `PLAN_RF_REFINE_FUSION_V2.md`. This runbook covers everything PLAN §5.5 deferred
("Launcher and Canary commands are outside this PLAN revision and will be added to the runbook
after the local code gate"). The local code gate is complete.

**What this runbook does NOT authorize.** It now reaches the one-cycle V2F5A Head-directed policy
qualification in §9, after the state-transition Canary and powered S7 source-transmission result.
It still does not authorize production `D>1`: only a positive, adequately covered §9 verdict on
both predeclared proteins can support a later authority revision that opens a small recursive
capability run. No wiring Canary is evidence of transmission or reward directionality.

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

> **SUPERSEDED by §5.3 (2026-08-06).** Everything in this subsection and in §5.2 was read off
> 8–16 endpoints drawn from **two** source prefixes. The properly-powered measurement — 32
> INDEPENDENT prefixes per protein — does **not** confirm a population-level mismatch on either
> axis. Read §5.3 first; what remains true here is kept below because the Canary cell's behaviour
> is still a fact that needed explaining.

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

## 5.2 `N_H^whole` on the Canary's own endpoints — **SUPERSEDED by §5.3**

> The numbers below are correct as a description of the 28 Canary endpoints. The CONCLUSION drawn
> from them — that the resumed population is systematically colder — did not survive a properly
> powered measurement. Twenty-eight endpoints from four source prefixes, at a measured intraclass
> correlation of 0.37–0.51, are worth roughly a dozen independent samples. §5.3 has the answer.

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
  **CORRECTED by §7.11** — on 4456 endpoints the gate fires on 7–12%. This bullet was a
  28-endpoint artefact.
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

## 5.3 The resumed null, 32 INDEPENDENT prefixes per protein (2026-08-06, `c24ddc9`)

`scripts/calibrate_rf_fusion_v2_hotspot.py --population resumed --c-source 50 --n-sources 32
--completions-per-source 2`, jobs `12096127/8`, 64/64 head-valid per protein, zero failures.
Artifacts: `<WORK>/v2_canary/resumed_null/<PID>/{calibration_rows.parquet,hotspot_calibration.json}`.

### The population mismatch is NOT supported

| | full-trajectory null (n=64) | **resumed null (n=64, 32 prefixes)** | Canary (n=8, 2 prefixes) |
|---|---:|---:|---:|
| `5ZHV_B` `N_H` q50 / **q90** | 5.966 / **15.738** | 5.115 / **15.333** | 1.319 / 11.699 |
| `5ZHV_B` scTM q50 / ≥0.85 | 0.8528 / 50.0% | **0.8571 / 56.3%** | ~0.798 / 12.5% |
| `Q00511` `N_H` q50 / **q90** | 16.587 / **20.083** | 16.921 / **20.561** | 13.551 / 18.371 |
| `Q00511` definitive-feasible | 29.7% | **26.6%** | 37.5% |

The two nulls agree to **2.6%** and **2.4%** on the Head threshold and to a few points on structure
operability. **§5.1's and §5.2's mismatch reading was small-sample.** It came from 8–16 endpoints
drawn from two prefixes, and the measured intraclass correlation says how badly that under-counts:

| | between-source var | within-source var | **ICC** |
|---|---:|---:|---:|
| `5ZHV_B` | 21.74 | 21.02 | **0.508** |
| `Q00511` | 14.74 | 25.16 | **0.369** |

A third to a half of the variance lives BETWEEN prefixes, so eight endpoints from two prefixes are
worth about three independent draws. That is exactly the error "the source prefix is the sampling
unit" exists to prevent, and it is why this measurement was worth running even though it overturned
the reading that motivated it.

`5zhv_r30`'s 0/4 stays a real observation and is now correctly explained: a ~44–56% per-endpoint
rate, four siblings of one prefix, ICC ≈ 0.5. Source maturity at `c=50` explains little on its own
(Spearman −0.16 against scTM; the resumed sources span 41–62 unresolved editable while both Canary
sources sat at 55).

### What IS mis-scaled: `Q00511`'s absolute anchor band

Not the population — the number.

| `Q00511`, n=64 | value |
|---|---:|
| native `max_anchor_sidechain_RMSD`, same backend | **1.790 Å** |
| design absolute, min / q50 / max | 1.330 / **2.409** / 7.345 Å |
| design ≤ the v0 band of 2.0 Å | **17/64** |
| native-relative excess `Δ_anchor`, q50 / **q90** / max | 0.615 / **1.479** / 5.556 Å |
| scTM, min / q50 | 0.921 / 0.957 — **64/64 above 0.85** |

The 2.0 Å band sits between the native (1.790) and the design median (2.409), so it rejects
three quarters of the null for about **0.6 Å of excess over a native that already scores 1.79**.
Every one of those rejections is the anchor band alone: scTM never binds on this protein.
`5ZHV_B` is the mirror case — no anchors, and scTM sitting on the 0.85 gate with a null median of
0.857.

### FROZEN mechanism-stage gates

Measured, not chosen. `Q0.10(lower)` for a floor, `Q0.90(higher)` for a ceiling — a floor and a
ceiling must round in opposite directions or one of them admits a sample it meant to exclude. By
construction each admits ~90% of the feedback-off null, which is what lets the transmission
experiment run.

| | `5ZHV_B` | `Q00511` |
|---|---:|---:|
| `delta_new_cumulative` (`raw_logit`) | **15.333256721496582** | **20.560853958129883** |
| `scTM_min` | **0.79562** | **0.93541** |
| `Δ_anchor_max` | n/a (unconstrained) | **1.47909 Å** |
| ⇒ absolute band `= native + Δ_max` | n/a | **3.26876 Å** |
| hard-anchor residue identity | n/a | **absolute hard gate, unchanged** |

The absolute band is DERIVED from the native and the excess, not chosen: it is written that way
only because `fusion.oracles.structure_feasible` compares an absolute number, and frozen v0 is a
system boundary this project does not mutate. The derivation lives in the config file beside the
value. Configs: `inverse_folding/reference_flow/configs/rf_refine_fusion_v2_mechanism_{5zhv_b,q00511}.yaml`,
both byte-identical to the v0 file apart from those thresholds.

Realized against the null they came from — which is the check that says the control arm can run:

| gate | admits of its own 64-endpoint null |
|---|---|
| `5ZHV_B` v0 (`scTM_min` 0.85) | 36/64 = 56% |
| `5ZHV_B` mechanism-stage | **59/64 = 92%** |
| `Q00511` v0 (2.0 Å absolute) | 17/64 = 27% |
| `Q00511` mechanism-stage | **52/64 = 81%** |

`5ZHV_B`'s file keeps `max_anchor_sidechain_RMSD_max: 2.0` untouched: the protein declares no
anchors, so v0 resolves `has_active_site=False` off the manifest and the branch never fires.

**Authority.** These are **mechanism-operability gates for the transmission experiment only**. A
threshold set at the 10th percentile of a null is chosen so the experiment can run, not so the
product is safe; capability and holdout still need their own frozen structure gate. The Head
thresholds are population-matched replacements for the full-trajectory ones and, per §5.2's
surviving observation, still sit above most of what the mechanism regime produces — a gate that
rarely fires is not evidence that nothing was caught.

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

## 7. The mechanism cohort

The Canary answered "does it execute". This answers **"does it transmit"**, and it is the only stage
whose result can open `D>1`.

### 7.1 What the kernel makes the question about

`FeedbackSupportPolicy` returns `write_from_endpoint=(write,)` — a single position, always
(`policy.py`). `inject_from_source_feedback` writes the **source's own** token, not the endpoint's
(`projection.py`), and `carry_from_source` copies source rows verbatim. So at `D=1` the entire
endpoint channel is **one amino acid**, written at the lowest-indexed masked position and protected
until `c_next`.

The experiment therefore asks a sharp question: **does one conditioning token measurably steer the
completion?** A null answer indicts the write cardinality, not the architecture — and §7.6 reads it
that way.

### 7.2 Two views: one test, one positive control

Both hold the source prefix, the support partition, the fork seeds and the horizon fixed; they are
run in ONE call to `run_mechanism_views`, which pays for the prefix and its lookahead pool once and
forks every arm off that single realized pool.

| view | what differs | role |
|---|---|---|
| `endpoint_change` | a DIFFERENT projected endpoint (one token) | **the mechanism test** |
| `source_shuffle` | the source's resolved bytes permuted, same endpoint | **assay positive control** |

`feedback_off` produces no descendants — feedback-disabled cycles stop after the A2 view — so it is
not a descendant contrast. `source_change` masks a carried position, which moves that position into
the measurement domain on one arm only; it still runs, it is not scored.

### 7.3 The statistic, and why the domain is the whole design

Normalized Hamming between a matched pair of complete descendants, on the **free domain**:

> editable positions still unresolved in the projected state at `r_d`.

One predicate, stated as a property of the projection rather than as a list of exclusions, so it
cannot fall out of date with the kernel. It automatically excludes hard anchors (not editable), the
written position (resolved by the write), and both source-carrying classes.

That last exclusion is not fastidiousness — it is what stops the control from passing on nothing.
Under `source_shuffle` the permuted bytes flow verbatim through `inject` and `carry`, so on the
three executed Canary coordinates the arms differ **by construction** at:

| cell | inject + carry-resolved | whole-editable Hamming |
|---|---:|---:|
| `5zhv_r40` | 9 + 37 = 46 / 101 | **0.455** |
| `q00511_r30` | 10 + 80 = 90 / 277 | **0.325** |
| `q00511_r40` | 10 + 109 = 119 / 277 | **0.430** |

A 0.02 gate on that domain is not a gate. Both denominators are written to
`mechanism_contrasts.parquet`; the free one is primary and the other is reported beside it.

**Descendants are not filtered on their own structure verdict.** That is a post-treatment variable —
conditioning on it is collider stratification and can bias the contrast. Structure is read per arm
as the §7.6 secondary, from `complete_endpoints` and `structure_evaluations`.

### 7.4 The unit of analysis is the SOURCE PREFIX

Measured ICC on the resumed null: `5ZHV_B` **0.508**, `Q00511` **0.369**. A third to a half of the
variance lives between prefixes, so descendants of one prefix are worth well under one draw each.
Matched forks are averaged **within** a prefix first; prefixes are then the sample.

Pairs whose two arms received byte-identical inputs are **structural zeros**, not evidence of no
transmission. Excluded by pre-registered rule (`contrastable=false`), and counted, so the exclusion
is visible rather than merely applied.

### 7.5 Power, from measured spread — as an INTERNAL pilot

The contrast is already a difference, so this is a one-sample problem: $\sigma_d$ is the standard
deviation of the prefix-level means, and the two-sample factor of 2 does not belong in the formula.

$$
n_{\text{pairs}} \;=\; \mathrm{clip}\!\left(\left\lceil \frac{(z_{0.975}+z_{0.80})^{2}\,\sigma_d^{2}}{\delta^{2}} \right\rceil,\; 32,\; 128\right)
$$

Run in two batches, **analysed as one**. Batch 1 is 16 prefixes/protein and estimates $\sigma_d$;
batch 2 completes to $n_{\text{pairs}}$; the verdict is read on all of it. This is a standard
internal pilot — discarding batch 1 would throw away a third of the sample for a purity that
inflates type-I error by less than 0.001 at these sizes. Batch 1 emits **no verdict**
(`read_v2_mechanism.py` without `--confirmatory`), and its only output that feeds a decision is
$\hat\sigma_d$.

Sized on `endpoint_change` — the binding contrast — and not on the maximum over both views:
`source_shuffle` perturbs tens of conditioning tokens where the primary perturbs one, so its spread
belongs to a much larger effect and would push the primary past the cap to over-power a control that
passes at a fraction of the sample. Reference scale at $\delta = 0.02$:

| $\sigma_d$ | 0.03 | 0.04 | 0.05 | 0.06 | 0.08 |
|---|---:|---:|---:|---:|---:|
| prefixes | 18 → floor 32 | 32 | 50 | 71 | 126 |

Past the cap the run is declared `underpowered_unresolved`. **$\delta$ is never raised to fit the
sample.**

**Realized, batch 1 (2026-08-07, jobs `12098724`/`12098725`, 16 prefixes per protein).**
$\hat\sigma_d$ = 0.02432 (`5ZHV_B`) and 0.04285 (`Q00511`); the max gives an unclipped 12, so the
**floor governs and $n_{\text{pairs}} = 32$ scored prefixes per protein**. Batch 1 scored 11 each at
a contrastable rate of 0.688, so batch 2 is fixed at **40 generated prefixes per protein (indices
16-55)**: the naive 21/0.688 ≈ 31 reaches the target only 63% of the time, 40 reaches it 99% of the
time. The count is chosen from the measured rate BEFORE batch 2 runs and does not move afterwards;
every scored prefix enters the analysis, and $n_{\text{pairs}}$ is a power floor rather than a cap.

### 7.6 The gate, pre-registered

**$\delta = 0.036$ on the free domain** — about 2 residues on `5ZHV_B` (free ≈ 55 at `r=40`) and
about 6 on `Q00511` (free ≈ 158). This is the pre-registered "at least ~2 / ~6 extra residues of
downstream propagation", expressed in the only denominator whose positions can move.

Gate: the **two-sided 95% CI lower bound exceeds $\delta$** — confidence that the effect clears the
margin, not merely that a sample mean did. A point-estimate criterion on top would be inert (it is
implied by the CI criterion at this $n$) and would silently halve power at the design effect.

Conjunctive across proteins and views, so it is an **intersection-union test**: each component is
read at its own level and **no multiplicity correction applies**. Adding one would buy no type-I
protection and would under-power a sample sized without it.

| observation | reading |
|---|---|
| primary clears $\delta$, control live | **transmission demonstrated** |
| control live, primary does not clear | the assay works; **the one-token channel does not transmit** → indicts write cardinality, `D>1` stays shut |
| control silent | **assay failure** — the readout cannot see propagation, and the primary's zero says nothing |

Secondary, read even if the primary fails, because "transmits nothing" and "damages structure" call
for different next experiments: if both arms' descendants drop against their own depth-0 pool it is
the resumed/no-remask generation geometry; if only the feedback arm drops, projection is damaging
structure and `D>1` stays shut regardless of transmission.

Admission runs on the **mechanism-operability gates of §5.3**, never the v0 contract. Hard-anchor
residue identity remains an absolute hard gate.

### 7.7 Coordinate and cost

`r = 40` for both strata — the only `r` with a committed Canary transition on both (`5zhv_r30` was
null). `r = 30` may run as a secondary; it does not enter the verdict.

Traceable to the executed Canary (jobs `12094327–30`): one cell ≈ 2 min wall, 8.6–35.9 GPU-s,
MaxRSS 27.5–30.4 GB; refold of a 302-aa design dominates a cycle's marginal cost (~94%). A prefix is
about three cells, so 16 prefixes × 2 proteins ≈ 1.5–2 GPU-hours for batch 1.

### 7.8 Realized result (2026-08-07) — `transmission_demonstrated` AT `r = 40`

Jobs `12098724`/`12098725` (batch 1, prefixes 0-15) and `12099604`/`12099605` (batch 2, prefixes
16-55), analysed as ONE sample per the internal-pilot design.

| | scored | contrastable | mean | CI95 lower | residues | $\hat\sigma_d$ |
|---|---|---:|---:|---:|---:|---:|
| `5ZHV_B` `endpoint_change` | 38/56 | 0.679 | 0.06233 | **0.04903** | 3.4 / 54.3 | 0.04046 |
| `Q00511` `endpoint_change` | 39/56 | 0.696 | 0.09355 | **0.07783** | 14.8 / 158.2 | 0.04850 |
| `5ZHV_B` `source_shuffle` | 55/56 | 0.982 | 0.31985 | 0.30566 | — | 0.05247 |
| `Q00511` `source_shuffle` | 56/56 | 1.000 | 0.22182 | 0.21349 | — | 0.03107 |

**The gate passes on all four components.** Both primaries clear the $\delta = 0.036$ margin by
their two-sided 95% CI lower bound; both positive controls are wide awake. Realized
$\hat\sigma_d^{\max} = 0.0485$ gives an unclipped 15, so 38 and 39 scored prefixes exceed the
32-prefix floor the design was powered on — the study is not underpowered by its own rule.

**What was demonstrated, and at what.** ONE amino acid, written at the lowest-indexed masked
position and protected until `c_next`, measurably steers the completion: 3.4 downstream residues on
`5ZHV_B` and 14.8 on `Q00511`, on domains of 54 and 158. In batch 1 not one of the 22 scored
prefixes across both proteins produced an exact zero.

**The scope is ONE COORDINATE, and the numbers are a point on an unmeasured curve.** Everything
above was measured at `r = 40`, `c_source = 50`, `c_next = 60`, `D = 1`, under one policy with a
write cardinality of 1 and 4 lookaheads. The only factor with two levels is the protein
(unconstrained / anchored). `r` is very unlikely to be inert: a deeper re-entry leaves more forward
steps after the injection, so the effect plausibly grows as `r` falls -- and the band pins more
reopens there too (the Canary measured 50 reopens at `r = 30` against 21 at `r = 40` on `Q00511`),
so the STRUCTURE cost plausibly grows with it as well. **A result at `r = 40` licenses nothing at
`r = 30`, and the effect size is not a property of V2 but of this coordinate.** The `r`-curve is
the obvious next axis and has not been run.

**The domain choice was load-bearing, and it is now measured rather than argued.** For
`endpoint_change`, `n_diff_editable` equals `n_diff_free` fork by fork — every difference the
endpoint channel produced lies inside the free domain. For `source_shuffle`, the whole-editable
reading is 0.594 / 0.529 against 0.320 / 0.222 on the free domain, the gap being exactly the
permuted source bytes that reach both projections by construction. A $\delta = 0.036$ gate on the
editable denominator would have passed the positive control on construction alone.

**Two batches, two code revisions, one sample.** Batch 1 ran under `af83803f0a38` and batch 2 under
`2844d0edad3d`; the only run-path difference is the loop bound that selects WHICH prefix indices a
batch runs. Established by evidence rather than by reading the diff: prefix 0 replayed under the
later revision (job `12099520`) is **byte-identical** to batch 1's prefix 0 in every contrast column
including the descendant sequence digests.

### 7.9 The secondary, CORRECTED (2026-08-07)

**The first reading of this section was wrong, and the error changed its conclusion.** It reported
"descendants fold ~26 points worse than their depth-0 pool" from a descendant population that
POOLED every arm -- including `source_shuffle` arm B, whose source's resolved bytes are permuted by
construction. That arm folds at **0.9% / 0.0%**, which is what a scrambled source is supposed to do:
it is the assay's positive control, not a V2 descendant. Pooling it manufactured most of the drop.

Split by arm, under the same frozen mechanism gate:

| | depth-0 pool | `endpoint_change` arm A (**the V2 descendant**) | arm B | `source_shuffle` arm A | arm B |
|---|---:|---:|---:|---:|---:|
| `5ZHV_B` `r=40` | 94.2% | **90.0%** | 88.2% | 90.0% | 0.9% |
| `5ZHV_B` `r=30` | 94.2% | **93.3%** | 94.2% | 93.3% | 3.6% |
| `Q00511` `r=40` | 90.2% | **85.9%** | 86.4% | 86.2% | 0.0% |
| `Q00511` `r=30` | 90.2% | **85.5%** | 85.9% | 85.7% | 0.0% |

**One V2 cycle costs about 4 percentage points of structural feasibility, not 26.** Two internal
checks pass: `endpoint_change` arm A and `source_shuffle` arm A are the same configuration
(as-selected endpoint, intact source) and agree to within 0.3 points; and arm B of `source_shuffle`
collapsing to ~0% shows the gate discriminates rather than admitting everything.

Three candidate explanations for even that 4 points were tested and TWO are eliminated:

| candidate | test | result |
|---|---|---|
| re-run steps / reopen count (**geometry**) | paired `r=30` vs `r=40`, a 2.5-4.7x dose contrast | **ruled out** -- feasibility flat (`5ZHV_B` +0.032 p=0.41, `Q00511` -0.005 p=0.79) |
| how much the feedback propagated | Spearman(hamming, feasibility) per prefix | **null** (-0.17 p=0.32, +0.04 p=0.80) |
| completing from step 60 rather than step 50 (**stage**) | natural completions captured at `c=60`, same gate, n=128 | **this is most of the 4 points** -- see below |

**Against a STAGE-MATCHED comparator the cost is near zero.** The 4 points above compare a
descendant finished from step 60 against a pool finished from step 50. Completions captured
naturally at `c=60` -- no projection anywhere -- score:

| | natural `c=50` | **natural `c=60`** | V2 descendant `r=40` | V2 descendant `r=30` |
|---|---:|---:|---:|---:|
| `5ZHV_B` | 92.2% | **93.8%** | 90.0% | 93.3% |
| `Q00511` | 81.2% | **85.9%** | **85.9%** | 85.5% |

`Q00511`'s descendant matches its stage-matched control exactly; `5ZHV_B`'s is 3.8 points below at
`r=40` and 0.5 below at `r=30`. So one V2 cycle's structural cost is somewhere in **0-4 points and
centred near zero**, not the 26 first reported and not clearly separable from zero.

**Noise warning, stated because it bounds every absolute number here.** The two independent
estimates of "natural completions from `c=50`" disagree -- 90.2% (the mechanism cohort's own depth-0
pool) against 81.2% (the resumed null, a different seed namespace) on `Q00511`. Cross-run absolute
comparisons therefore carry several points of sampling noise. The reliable comparison is the
WITHIN-run paired one across `r`, and it says flat.

**What is NOT measured is whether the per-cycle cost COMPOUNDS** -- which is precisely what `D>1`
does, and no run has produced a second cycle.

The measured `r`-dependence of the primary is the other half of this section: transmission GROWS with
depth -- 3.42 -> 5.95 residues on `5ZHV_B` and 14.79 -> 19.93 on `Q00511`, paired Wilcoxon
p = 2.3e-4 and 2.0e-4. `B(r)` is calibrated only at `r in {30, 40}` and `lookup_band` never
interpolates, so this is **two points, not a curve**; a third depth needs a new §1 band scan.

### 7.10 Only then

Transmission holds at `r=40` and grows toward `r=30` (§7.8, §7.9).  The structure cost of one cycle
is now MEASURED rather than unattributed: **0-4 points against a stage-matched control, centred near
zero**, insensitive to rollback depth and to how much was transmitted.  That is small.  It is still
not a licence, for one reason: **nothing has measured whether it compounds.**  A near-zero cost per
cycle is tolerable; a cost that multiplies is not, and `D>1` is exactly the experiment that would
find out -- so it cannot be its own precondition.

This section previously named a `D=2` compounding cohort as the next experiment. **§7.11 supersedes
that.** With the reward channel measured inert at one cycle, a second cycle would compound a
transport that carries no preference — it would price a cost with no benefit on the other side of
the ledger. The `D=2` cohort is withdrawn; `D>1` stays launch-disabled, now for a stronger reason
than the unmeasured cost.

(The no-projection arm this section called for before that was proposed to explain a 26-point drop
which turned out to be a pooling error. It is not the question either.)

`allow_production_depth_gt_1` is never granted by the oracle factory. Opening it is a runbook
decision that also requires the FeedbackSupportPolicy directionality gate. Nothing in §7 authorizes
it, and a green secondary readout authorizes nothing at all.

### 7.11 Reward directionality at one cycle — `not_demonstrated` (2026-08-07)

**Purpose.** §7.8 showed the channel TRANSMITS: what the endpoint carries reaches the descendant.
It never asked whether what arrives is *better*. Transmission is a claim about identity; a
capability claim needs one about VALUE. This section asks the second question, at one cycle, on
both scoring axes.

**Method.** No new generation run — the `endpoint_change` view already IS the reward-ordered
contrast. `select_family_representatives` sorts admissible endpoints by
`(head_global_risk, endpoint_id)` ascending, so arm A (`endpoint_rank=0`) is the Head-BEST endpoint
and arm B (`endpoint_rank=1`) the next-best. Both arms run off one source object, one scored
endpoint pool, identical realized propagation and lookahead seeds, one horizon, and support
cardinality forced equal by `required_support_budget`. Measured `n_input_diff in {0, 1}`: the two
arms' projected states differ by **exactly one token**; the zeros are structural (the two endpoints
agree at the written position) and are dropped by the pre-registered `contrastable` rule.

Two readouts, both paired on the source prefix:

* the **primary** — arm A minus arm B on the descendant;
* the **transmission slope** — regress `(desc_A - desc_B)` on `(parent_A - parent_B)` across
  prefixes. The slope is the FRACTION of an injected score gap that reaches the descendants. Being
  dimensionless, it does not loosen if the dose is raised.

`N_H^whole` was recomputed for all 4456 mechanism endpoints with
`scripts/analysis/recompute_v2_nh_whole.py` (job 12113292; live-Head drift 0.011 raw logits under
`--verify-designs`). `source_shuffle` arm B is EXCLUDED throughout — its source is scrambled and its
structure collapses to 0.9% / 0.0%, and pooling it is what produced the retracted 26-point figure of
§7.9.

**Evidence.** Injected dose, `rank0 - rank1`: `head_global_risk` -0.944 (`5ZHV_B`) / -2.596
(`Q00511`) against a within-pool SD of 4.30; `N_H^whole` -1.94 / -3.61. Lower is better on both.

| cell | primary `head_global_risk` | slope | primary `N_H^whole` | slope |
|---|---:|---:|---:|---:|
| `5ZHV_B` r40 | -0.083 [-0.507, +0.341] | +0.003 [-0.170, +0.177] | +0.047 [-0.442, +0.536] | -0.046 [-0.189, +0.097] |
| `Q00511` r40 | -0.228 [-0.717, +0.261] | +0.027 [-0.127, +0.181] | -0.368 [-0.920, +0.184] | -0.060 [-0.191, +0.071] |
| `5ZHV_B` r30 | -0.078 [-0.414, +0.258] | +0.034 [-0.100, +0.168] | -0.308 [-0.799, +0.183] | +0.064 [-0.080, +0.209] |
| `Q00511` r30 | +0.356 [-0.480, +1.191] | +0.029 [-0.234, +0.293] | +0.518 [-0.470, +1.506] | +0.017 [-0.219, +0.252] |

95% CIs. `n = 38 / 39 / 39 / 39` prefixes for the primary (contrastable and analyzable),
`55 / 55 / 56 / 55` for the slope. **Every primary CI straddles zero** (min `p = 0.19`) and the sign
is a coin flip — three of four lean the right way on `head_global_risk`, two of four on `N_H^whole`.
**Every slope point estimate is `|.| <= 0.064` with `R^2 <= 0.016`.**

One trap worth recording: the RAW parent-to-descendant correlation is `r = 0.54 / 0.48`,
`p < 1e-7` at `r=40`. That is prefix-level confounding — a hot source yields hot endpoints AND hot
descendants — and it vanishes within prefix. Read as transmission it is simply wrong.

Single-cycle expectation, arm A best-of-4 against its OWN depth-0 pool best-of-4 (**not
compute-matched**; feedback spends an extra generation round): protein-split and null on both axes.
`5ZHV_B` +0.69 / +1.00 (worse), `Q00511` -0.66 / -0.68 (better) at `r=40`. Nothing clears
`p < 0.05` except one Wilcoxon (`5ZHV_B`, `head_global_risk`, `p = 0.010`) the t-test does not
corroborate (`p = 0.087`).

Artifacts: `<WORK>/v2_canary/nh_mechanism/{mechanism_endpoints_nh.parquet,summary.json}`.

**Conclusion — the channel transmits identity and not value.** One token in produces 2.4–14.1
differing residues out (§7.8), while 0–6% of the injected score gap arrives and the CI excludes
anything above ~25% on both axes. This is not an underpowered null: the bound is FRACTIONAL, so
raising `K` or widening the endpoint spread does not relax it.

The mechanism is not surprising once stated. The only endpoint-dependent quantity in the entire
projection is the amino acid at ONE position, and even that position comes from the source's mask
geometry — the verified policy rule is `write=min_source_masked; carry=inherited_masks+commit_lt_r;
inject=commit_ge_r; reopen=band_pinned_latest_committed`, with no Head or reward term anywhere. The
Head enters only at `select_family_representatives`: it ranks ENDPOINTS, never positions. PLAN
§2.5's `residual Head window-to-residue attribution` — the component that would make position choice
reward-directed — is not implemented.

**What this does NOT say.** It does not show feedback is harmful. It shows the reward channel is
inert at `D=1`, on two proteins, two re-entry points, one cycle, one Head. It says nothing about
compounding, which no run has produced.

**Side finding — the Head gate is binding after all, correcting §5.2.** On the clean populations,
`7-12%` of endpoints exceed the full-trajectory threshold (15.738 / 20.083):

| cell | depth-0 pool | arm A descendant | arm B descendant | Δ(A - own pool), Wilcoxon |
|---|---:|---:|---:|---:|
| `5ZHV_B` r40 | 19/224  8.5% | 18/152  11.8% | 17/152  11.2% | +3.9 pt, p=0.41 |
| `Q00511` r40 | 16/224  7.1% | 13/156   8.3% | 11/156   7.1% | +0.6 pt, p=0.68 |
| `5ZHV_B` r30 | 19/224  8.5% | 17/156  10.9% | 13/156   8.3% | +2.6 pt, p=0.54 |
| `Q00511` r30 | 16/224  7.1% | 12/156   7.7% | 13/156   8.3% | +0.0 pt, p=0.98 |

§5.2's "a gate that never fires" came from 28 endpoints. Descendants breach slightly more often than
their own pool on every cell, but no cell is significant: **no evidence that feedback worsens
safety, which is not the same as evidence that it does not.**

---

## 8. Known-open, recorded rather than worked around

- `active_population_width > 1` is refused at config parse — Interface Map OQ7 leaves open whether a
  forked family inherits its parent's `reference_binding_id` or opens its own.
- `reward_ordered` / `policy_source_off` mechanism views raise `NotYetFrozenError` until the
  production policy spec is frozen (PLAN §8.4); a placeholder would yield a number that looks like
  evidence. The reward-ordering QUESTION is nonetheless answered — §7.11 reaches it through
  `endpoint_change`, which needs no freeze.
- `source_change` (PLAN §2.6 intervention 2 — endpoint AND support pinned, source ablated) is
  implemented in `paired.py` but has never been executed. `source_shuffle` does not substitute for
  it: its arm B collapses structure to 0.9% / 0.0%, so any outcome read on that arm is confounded.
- The Head-directed V2F5A support law has not yet run on the cluster. Sections 5 and 7 record the
  completed real-model Canary and source-transmission campaign; local tests for §9 validate only
  wiring, accounting and analysis semantics until the calibration/replay/qualification artifacts
  below are produced.

---

## 9. V2F5A — one-cycle Head-directed policy qualification

### 9.0 Frozen question and launch boundary

S7 established that a source-coupled transition carries identity, but its support positions were
Head-blind and descendant reward directionality was not demonstrated. V2F5A changes exactly one
thing: the treatment chooses capped write/reopen support from frozen-Head evidence; the control
uses the source-geometry law at the treatment's realized write/reopen cardinalities. Source, donor,
substrate, `r=40`, `c_source=50`, `c_next=60`, maturity band, descendant seeds and horizon remain
matched.

This section freezes one direct confirmatory cohort, not a dose ladder:

| Quantity | Frozen value | Reason |
|---|---:|---|
| proteins | `5ZHV_B`, `Q00511` | ordinary plus anchored predeclared substrates |
| source prefixes | `56` per protein | same generated-prefix depth as S7; at least `32` scorable prefixes are required |
| coordinate | `r=40`, `c_source=50`, `c_next=60`, `D=1` | S7-qualified transmission coordinate; no coordinate search |
| write cap | `ceil(0.05 * N_editable)` | committed V2F5A policy law, a cap rather than a quota |
| counterfactual ceiling | `278` per cycle | maximum editable domain in the predeclared two-protein qualification cohort; a hard mathematical ceiling, not outcome tuning |
| cohort Head cap | `40000` per one-protein job | exceeds conservative preflight `2 * 56 * 278` plus endpoint Head scoring, without changing policy behavior |
| primary | prefix-mean `HeadRisk(treatment) - HeadRisk(control)` | lower is better; matched forks are averaged within source prefix |
| inferential gate | one-sided 95% upper bound `< -epsilon_R` on **both** proteins | `epsilon_R` is frozen below from same-sequence Head repeatability; conjunctive intersection-union gate |

Steps 3–5 below are cluster measurements. Local tests cannot substitute for them. A negative or
underpowered result keeps `D>1` closed; it is not rescued by raising the write cap, changing `r`, or
increasing depth after seeing the outcome.

Set the common cluster inputs once. Every Head path must be the exact instrument used by S7:

```bash
set -euo pipefail
export PROJECT_ROOT=/home/zc1519/src/Immune-Design
export SCRATCH_BASE=/scratch/gpfs/KAIYIJIANG/zijie
export PYTHONPATH=${PROJECT_ROOT}
cd "${PROJECT_ROOT}"

WORK=${SCRATCH_BASE}/work/immune-design/v2_canary
RUN=${SCRATCH_BASE}/run/inverse_folding/v2_canary
QUAL_WORK=${WORK}/v2f5a_head_directed
QUAL_RUN=${RUN}/v2f5a_head_directed
mkdir -p "${QUAL_WORK}"/{calibration,replay,resolved_configs,analysis} "${QUAL_RUN}"

GIT_SHA=$(git rev-parse HEAD)
POLICY_SPEC=${PROJECT_ROOT}/inverse_folding/reference_flow/configs/v2_head_directed_capped_policy_v1.json

# The exact frozen S7 instrument, read off the S7 resolved configs and the hotspot artifact rather
# than retyped: `--head-config` is the config DIRECTORY (it maps to v0's `head_config_dir`), and
# `HEAD_ALLELE_IDX` / `HEAD_WINDOW_BATCH_SIZE` are the values S7's own runtime defaulted to.
HEAD_CONFIG="${PROJECT_ROOT}/epitope_head/configs"
HEAD_CHECKPOINT="${SCRATCH_BASE}/run/epitope_head/cnn_himp_a1_res03_drb0701_seed42_cv5_fold0/runs/LC1/seed_42/best.pt"
HEAD_VARIANT_ID="LC1"
HEAD_ALLELE_IDX=0
HEAD_WINDOW_BATCH_SIZE=64
ALLELE=DRB1_0701
SCORE_SCALE=raw_logit
# 12, NOT 13. The frozen `DRB1_0701` Head emits k = 12..25; the hotspot calibration artifact, the S7
# resolved configs and the stored endpoint window vectors all carry `window_k_min = 12`. Declaring 13
# would not narrow the grid -- `whole_landscape_new_hotspot` refuses any score containing an
# out-of-domain window, so it is a guaranteed 0-of-N rather than a filter (§2.2).
WINDOW_K_MIN=12
WINDOW_K_MAX=25

# The completed r=40 S7 mechanism bundles. Prefix indices are disjoint by construction: batch 1 ran
# 0-15 and batch 2 ran 16-55. r=30 (`v2_mechanism_r30`), the Canary, the resumed null, the smoke and
# replay cells, and the `nh_mechanism` sidecar are all deliberately absent.
S7_5ZHV_B_BUNDLES=("${SCRATCH_BASE}/run/inverse_folding/v2_mechanism/5zhv_mech" \
                   "${SCRATCH_BASE}/run/inverse_folding/v2_mechanism_b2/5zhv_mech")
S7_Q00511_BUNDLES=("${SCRATCH_BASE}/run/inverse_folding/v2_mechanism/q00511_mech" \
                   "${SCRATCH_BASE}/run/inverse_folding/v2_mechanism_b2/q00511_mech")
REF_5ZHV_B="${WORK}/5ZHV_B.seq"
REF_Q00511="${WORK}/Q00511.seq"
```

### 9.1 Step 3 — freeze the Head repeatability floor

Re-score the exact stored S7 complete sequences under the same frozen Head. The artifact records
the maximum same-sequence absolute drift `e` and freezes both difference-scale floors at `2e`.
This is an instrument-noise bound: it is not selected from the treatment/control response.

```bash
CALIB_BUNDLES=()
for bundle in "${S7_5ZHV_B_BUNDLES[@]}" "${S7_Q00511_BUNDLES[@]}"; do
  CALIB_BUNDLES+=(--bundle "${bundle}")
done

python scripts/calibrate_v2_head_policy.py \
  "${CALIB_BUNDLES[@]}" \
  --policy-spec "${POLICY_SPEC}" \
  --out-rows "${QUAL_WORK}/calibration/head_repeatability.parquet" \
  --out-json "${QUAL_WORK}/calibration/head_policy_calibration.json" \
  --min-observations-per-protein 64 \
  --max-sequences-per-protein 0 \
  --max-counterfactual-head-calls-per-cycle 278 \
  --head-config "${HEAD_CONFIG}" \
  --head-checkpoint "${HEAD_CHECKPOINT}" \
  --head-variant-id "${HEAD_VARIANT_ID}" \
  --head-allele-idx "${HEAD_ALLELE_IDX}" \
  --head-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}" \
  --allele "${ALLELE}" --score-scale "${SCORE_SCALE}" \
  --window-k-min "${WINDOW_K_MIN}" --window-k-max "${WINDOW_K_MAX}" \
  --device cuda
```

Stop if the producer reports fewer than 64 unique stored observations for either protein, any Head
config/checkpoint/domain mismatch, a non-finite score, or a stored sequence/digest mismatch. Record
`max_abs_repeat_drift` and `difference_noise_bound`; do not manually replace either scalar.

### 9.2 Materialize the two signed qualification cells

Reuse the exact S7 `r=40` materialization inputs and mechanism-operability structure gates. Do not
edit a resolved YAML. Run `scripts/materialize_v2_canary_config.py` twice with the same arguments as
the S7 `5zhv_r40` and `q00511_r40` cells, changing only the following:

```text
--campaign-id v2f5a_head_directed_r40_v1
--projection-policy-spec ${POLICY_SPEC}
--policy-calibration-json ${QUAL_WORK}/calibration/head_policy_calibration.json
--qualification-max-head-calls 40000
--out ${QUAL_WORK}/resolved_configs/5zhv_v2f5a_r40.yaml     # 5ZHV_B call
--out ${QUAL_WORK}/resolved_configs/q00511_v2f5a_r40.yaml  # Q00511 call
```

The full calls must retain `r_step=40`, the protein-specific `B(r)` artifact/stratum/reference,
the resumed-null hotspot artifact, `c1_constant_clean_no_remask.yaml`, and these structure gates:

```text
5ZHV_B: inverse_folding/reference_flow/configs/rf_refine_fusion_v2_mechanism_5zhv_b.yaml
Q00511: inverse_folding/reference_flow/configs/rf_refine_fusion_v2_mechanism_q00511.yaml
```

Only Q00511 receives
`--constraint-manifest inverse_folding/reference_flow/configs/uricase_q00511_active_site_safety_v1.yaml`.
The materializer verifies that the calibration Head digests and policy-spec sha256 equal the exact
files supplied to each cell. Keep the generated `.args.sh` beside each YAML.

### 9.3 Step 4 — offline executability replay before descendants

Replay the real policy over the existing S7 source states. This loads the frozen Head for
leave-one-out evidence but no denoiser or structure backend and generates no descendant.

```bash
replay_one () {
  local protein_id=$1 config=$2 band=$3 stratum=$4 reference=$5 out=$6
  shift 6
  local bundle_args=()
  for bundle in "$@"; do bundle_args+=(--bundle "${bundle}"); done
  python scripts/analysis/replay_v2_head_directed_policy.py \
    "${bundle_args[@]}" --config "${config}" --band-table "${band}" \
    --stratum-key "${stratum}" --reference-sequence "${reference}" \
    --protein-id "${protein_id}" --out "${out}" \
    --head-config "${HEAD_CONFIG}" --head-checkpoint "${HEAD_CHECKPOINT}" \
    --head-variant-id "${HEAD_VARIANT_ID}" --head-allele-idx "${HEAD_ALLELE_IDX}" \
    --head-window-batch-size "${HEAD_WINDOW_BATCH_SIZE}" \
    --allele "${ALLELE}" --score-scale "${SCORE_SCALE}" \
    --window-k-min "${WINDOW_K_MIN}" --window-k-max "${WINDOW_K_MAX}" --device cuda
}

replay_one 5ZHV_B \
  "${QUAL_WORK}/resolved_configs/5zhv_v2f5a_r40.yaml" \
  "${WORK}/bands/5ZHV_B/B_r.json" 5zhv_b_unconstrained \
  "${REF_5ZHV_B}" "${QUAL_WORK}/replay/5zhv_v2f5a_r40.json" \
  "${S7_5ZHV_B_BUNDLES[@]}"

replay_one Q00511 \
  "${QUAL_WORK}/resolved_configs/q00511_v2f5a_r40.yaml" \
  "${WORK}/bands/Q00511/B_r.json" q00511_anchor24 \
  "${REF_Q00511}" "${QUAL_WORK}/replay/q00511_v2f5a_r40.json" \
  "${S7_Q00511_BUNDLES[@]}"
```

Both replay JSONs must satisfy all of the following before any descendant job is submitted:

```bash
for report in "${QUAL_WORK}"/replay/*_v2f5a_r40.json; do
  jq -e '
    .coverage.head_contribution_measured == true and
    .coverage.n_sources >= 56 and
    .coverage.n_committed_decisions >= 32 and
    ((.coverage.stalls.stall_counterfactual_budget_exceeded // 0) == 0)
  ' "${report}"
done
```

Also inspect `realized_writes`, `required_reopen`, `contribution`, and
`realized_write_bound_by`. A failure is a policy-coverage result: stop and return to the policy;
never substitute arbitrary support or reinterpret the 5% cap as a target count.

### 9.4 Model-free and assembly preflight

For each generated cell, source its `.args.sh` and run the exact launch gate. Qualification uses
`2 * 56 = 112` conservative execution replicates in the budget projection, so a dry-run that still
reports one cycle is invalid.

```bash
preflight_qualification () {
  local cell=$1
  # shellcheck source=/dev/null
  source "${QUAL_WORK}/resolved_configs/${cell}.args.sh"
  python scripts/run_rf_fusion_v2.py \
    --v2-config "${V2_CONFIG}" --out-dir "${QUAL_RUN}/${cell}" --cohort "${V2_COHORT}" \
    --mechanism-prefixes 56 --qualification \
    --input-file "${INPUT_FILES[@]}" \
    --shard-input "${SHARD_INPUTS[@]}" \
                  "journal_dir=${QUAL_RUN}/${cell}/journals" "device=cuda" \
    --dry-run | tee "${QUAL_WORK}/analysis/${cell}.dry_run.json"
  jq -e '.budget_projection.execution_replicates == 112 and
         .budget_projection.feasible == true and
         .head_directed.max_counterfactual_head_calls_per_cycle == 278' \
         "${QUAL_WORK}/analysis/${cell}.dry_run.json"
}

preflight_qualification 5zhv_v2f5a_r40
preflight_qualification q00511_v2f5a_r40

python scripts/preflight_v2_canary_assembly.py \
  --cell "${QUAL_WORK}/resolved_configs/5zhv_v2f5a_r40.yaml=5ZHV_B" \
         "${QUAL_WORK}/resolved_configs/q00511_v2f5a_r40.yaml=Q00511"
```

### 9.5 Step 5 — run the matched one-cycle qualification

Submit both cells before reading either response:

```bash
QUALIFICATION=1 MECHANISM_PREFIXES=56 CELL=5zhv_v2f5a_r40 \
  WORK=${QUAL_WORK} RUNDIR=${QUAL_RUN} \
  sbatch --time=12:00:00 --mem=64G --job-name=v2f5a_5zhv \
  scripts/submit_rf_fusion_v2_canary.slurm

QUALIFICATION=1 MECHANISM_PREFIXES=56 CELL=q00511_v2f5a_r40 \
  WORK=${QUAL_WORK} RUNDIR=${QUAL_RUN} \
  sbatch --time=12:00:00 --mem=64G --job-name=v2f5a_q00511 \
  scripts/submit_rf_fusion_v2_canary.slurm
```

The launcher expects the cell `.args.sh` under `${WORK}/resolved_configs`, so these submissions set
`WORK=${QUAL_WORK}`. Do not copy a YAML without its generated `.args.sh`.

The two jobs are independent one-protein shards. `EXIT=0` means the requested shard completed and
produced usable evidence, not that the scientific gate passed. Any cap breach, partial cohort,
missing endpoint join, anchor violation, Head-identity mismatch, or all-stalled cohort is a run
failure and must be resolved before analysis.

### 9.6 Artifact integrity and the confirmatory read

First run the existing state/anchor/replay/lineage/assimilation/ledger reader. This is an
operational validity check, not the reward verdict:

```bash
python scripts/analysis/read_v2_canary.py \
  --bundle "${QUAL_RUN}/5zhv_v2f5a_r40=${REF_5ZHV_B}" \
           "${QUAL_RUN}/q00511_v2f5a_r40=${REF_Q00511}" \
  --json-out "${QUAL_WORK}/analysis/v2f5a_integrity.json"
```

Then read the paired Head primary. A non-zero exit is a scientific negative/unresolved verdict,
so preserve the JSON even when the shell return code is `1`:

```bash
set +e
python scripts/analysis/read_v2_mechanism.py --qualification \
  --bundle "${QUAL_RUN}/5zhv_v2f5a_r40" "${QUAL_RUN}/q00511_v2f5a_r40" \
  --config "${QUAL_WORK}/resolved_configs/5zhv_v2f5a_r40.yaml" \
           "${QUAL_WORK}/resolved_configs/q00511_v2f5a_r40.yaml" \
  --min-prefixes 32 \
  --out "${QUAL_WORK}/analysis/v2f5a_directionality_verdict.json"
ANALYSIS_RC=$?
set -e
echo "V2F5A_ANALYSIS_EXIT=${ANALYSIS_RC}"
```

Before accepting the verdict, inspect `feedback_events.parquet` and require treatment rows to have
`head_evidence_consulted=true`, positive selected contribution evidence, and the declared
Head-directed policy identity; matched control rows must have `head_evidence_consulted=false`.
`mechanism_contrasts.parquet` must show `view=support_law`, no parity violation, matched fork seeds,
and equal realized write/reopen cardinalities. Structure/Hamming are reported secondaries and must
not be used to filter the Head primary after treatment.

### 9.7 Decision

| Result | Decision |
|---|---|
| both proteins pass, integrity clean | `immune_directed_transition_supported`; V2F5A closes, and a **separate** small `D=2` authority/config may be designed |
| enough prefixes but either protein fails | `source_coupled_policy_unqualified`; keep the source-transmission result, reject this immune policy, keep `D>1` closed |
| fewer than 32 scored prefixes on either protein | `underpowered_unresolved`; do not call a directionality verdict and do not tune the margin/cap from outcomes |
| ordinary passes, anchored fails (or reverse) | substrate-specific response only; no general recursive authorization |
| operational integrity failure | invalid run; repair/replay under the same frozen question before any scientific reading |

Even a positive result is a one-cycle policy-directionality result, not a recursive capability or
biological immune-validity claim. Independent terminal immune validation remains downstream.
