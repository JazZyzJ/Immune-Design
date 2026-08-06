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
| Two Canary proteins chosen: one **ordinary**, one **anchored** | PLAN V2F5 acceptance names both; the anchored one is what proves hard anchors never enter a support set |

Cluster paths are **always** CLI arguments. Nothing in this runbook may be hardcoded into a module.

---

## 1. Produce `B(r)` — the schedule-band calibration

This must come first: the projection policy's **reopen cardinality is not a free parameter**. It is
pinned by `B(r_d)` through `admissible_reopen_cardinality`, so the policy is not instantiable until
a real band exists.

```bash
python scripts/rho_maturity_scan.py --mode step \
  --checkpoint       <dplm.ckpt> \
  --rf-config        inverse_folding/reference_flow/configs/c1_null.yaml \
  --test-set         <if_ready.parquet> \
  --pdb-root         <pdbs> \
  --out-parquet      <WORK>/v2_canary/rho_step_scan.parquet \
  --steps            <r candidates> \
  --stratum-key      <stratification law> \
  --quantile-levels  <levels> \
  --min-captures-per-cell <floor> \
  --calibration-id   band:v2:canary \
  --calibration-json <WORK>/v2_canary/B_r.json
```

Every step-mode flag is required and has no default — the stratification law, the quantile grid and
the capture floor are deferred scientific calibrations, and a cell below the capture floor exits
non-zero and writes **no** artifact.

**Choose the schedule cell so that both strata have non-empty support.** A cell where the ordinary
protein's stratum is calibrated and the anchored one's is not will make the anchored arm decline on
`no calibrated band at r_d`, which is a fact about the scan, not about the mechanism.

Step mode refuses a `remask.fraction_scale != 0.0` substrate: the V1 crossings were compressed into
the last ~16 steps by repeated global remasking, and PLAN §2.1 forbids reusing them to select V2
coordinates.

**Check before moving on:** `B(r)` must admit a non-empty reopen envelope at your chosen `r_d` for
**both** strata. If it does not, the Canary cannot produce a non-null transition and there is
nothing to fix downstream — go back and re-choose the cell.

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

`source_ref` must equal `calibration_source_ref(value, unit, source_kind, source_id, artifact)`, so
the provenance is content-binding rather than a label.

---

## 3. Fill the Canary config

Start from `inverse_folding/reference_flow/configs/v2_canary_state_transition.yaml`. Every
`REPLACE_*` is a value the file cannot carry; the loader has no default and refuses the run.

Three that bite:

1. **`identity.code_revision`** — 7–64 lowercase hex. `unknown` is refused. The CLI's
   `--code-revision` defaults to this value and must equal it, so no flag can sign a run under an
   undeclared revision.
2. **`content[complete_reference_sequence].expected_sha256`** — SHA-256 over the sequence's **ASCII
   bytes**. The file this role names must be **the canonical sequence itself: no FASTA header, no
   wrapper, no trailing newline.** `bind_cumulative_reference` recomputes the digest from those
   bytes and compares; any container format puts bytes in the file that are not in the sequence and
   the two numbers can never agree.
3. **`arm.a2_matching_resource`** — one of five, and the choice IS the experiment. At the measured
   unit costs one refold ≈ 81 DFE and structure is ~94% of a cycle's marginal cost, so matching on
   `logical_dfe` and matching on `definitive_refolds` are different experiments. The other four
   components are reported unmatched, never netted away.

`config.safety.incremental_gate_enabled` may now be either value. At depth 0 the gate has no
referent and the admission records a typed `IncrementalInapplicable` minted only from a parentless
ledger; it becomes binding from depth 1. If you enable it, `delta_new_incremental` is required and
must carry its own `immediate_parent`-scoped artifact.

---

## 4. Preflight — costs nothing, refuses early

```bash
python scripts/run_rf_fusion_v2.py \
  --v2-config    inverse_folding/reference_flow/configs/v2_canary_state_transition.yaml \
  --out-dir      <RUN>/v2_canary \
  --cohort       <ORDINARY_PID> <ANCHORED_PID> \
  --input-file   dplm_checkpoint=<dplm.ckpt> \
                 complete_reference_sequence=<reference.seq> \
                 schedule_band_calibration=<WORK>/v2_canary/B_r.json \
                 <...one ROLE=PATH per PLAN §5.2 role...> \
  --dry-run
```

Loads **no model**. It resolves the config, derives the one shared `config_digest`, signs every
declared input by CONTENT, and projects the run's logical cost against the declared caps —
`root_capture + screening + segment + descendant pools`, reconciled in the test suite against a
realized ladder's counted forward passes.

A run that declares **no** inputs is refused (`exit 4`), on this path and on the execution path
alike: PLAN §5.2's "missing content identity fails closed".

Exit codes are the whole interface a cohort runner sees: `0` complete and at least one success;
`2` nothing usable, an empty cohort, every protein failed, **or a realized cohort cap breached**;
`3` partial; `4` not launchable.

---

## 5. Run the Canary

Drop `--dry-run` and add the runtime paths the oracle stack needs:

```bash
  --shard-input  base_if_checkpoint=<dplm.ckpt> \
                 rf_sampler_config=inverse_folding/reference_flow/configs/c1_null.yaml \
                 test_set_parquet=<if_ready.parquet> \
                 pdb_root=<pdbs> \
                 schedule_band_calibration=<WORK>/v2_canary/B_r.json \
                 complete_reference_sequence=<reference.seq> \
                 constraint_manifest=<anchors.json> \
                 journal_dir=<RUN>/v2_canary/journals \
                 device=cuda \
                 <...one <role>=PATH for every RUNTIME-bound content role...>
```

`--shard-input` is `NAME=PATH`. A bare path or a repeated name is refused — guessing which
parameter a path meant is how a checkpoint ends up passed as a PDB root. Every **runtime-bound**
content role must be supplied here: its digest is computed from the bytes you actually give, and a
role that is neither frozen in the config nor supplied fails closed.

SLURM: follow `scripts/submit_benchmark.slurm`. `--output/--error` go to `logs/`, never `run/`.

---

## 6. What to read, and what NOT to conclude

Read **only** these, per PLAN §8.3:

| Check | Where | Pass condition |
|---|---|---|
| **Non-null transition** | `feedback_events.outcome` | at least one `committed` per protein. A cohort of `null_invalid_policy_result` means the policy could not answer against the realized state — that is a fact about `B(r)` or the coordinate, not about feedback |
| **Anchors** | `partial_states.hard_anchors_json` vs `complete_endpoints.sequence` | every anchor preserved in the anchored protein's endpoints |
| **Replay** | `partial_states.replay_state_hash`, `complete_endpoints.replay_*` | present and distinct per fork |
| **Lineage** | `partial_states.parent_state_id` / `parent_transition_id` / `origin_endpoint_id` | the chain source → endpoint → projected → propagated closes |
| **Assimilation** | `partial_states.active_score_status_json` | injected positions begin `pending_assimilation` with no fabricated score |
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
