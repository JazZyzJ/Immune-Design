# PLAN — RF-Refine Fusion v1-A: Pre-Terminal Continuation-Value Entry

**Status (2026-07-29):** local implementation and deployment Canaries A/B/C are complete.
The frozen 24-protein scientific T0 executed and returned GO; the earliest passing maturity
freezes `rho_target=0.50`. P1 method values and development configs are frozen in the runbook
§7.0. P1 cohort manifests and the P1 analysis gate remain blocking; no P1 outcome exists.

**Implementation target:** a descendant of the `fusion_rf_refine` branch. The audited
baseline is commit `a2427bfd365cee40d0a4b2e279b8778c449118b5` in the Fusion worktree on
2026-07-22. The current `dev_head` checkout does not contain the v0 `fusion/` package and is
not an implementation base.

**Scientific authority:**

1. `doc/FUSION_V1.md` — scientific question, mechanism, and claim boundary;
2. `doc/RF-Refine-Fusion.md` — frozen v0 method and evidence boundary;
3. `doc/RF_Fusion_v1_Cluster_Runbook.md` — T0/P1 experiment and operational contract;
4. this PLAN — code, interface, artifact, and validation contract.

Before implementation, copy or merge the three authority documents and this PLAN into the
implementation branch. Reconcile that branch's older `AGENTS.md` with the current project
governance before coding; do not execute this PLAN under two conflicting process contracts.

**Evolution boundary:** V1-A adds a partial-entry layer in front of v0 Fusion. It does not
relax the v0 complete-state invariant. A selected partial root is completed into a canonical
AA20 parent facade, and only that facade enters the existing terminal repair+beam engine.

**Dual allele:** paper-critical and high priority, but deliberately outside this PLAN. Its
objective, implementation, controls, and experiments remain reserved for a separate
user-directed contract.

---

## 0. Goal, Delivery Boundary, and Task State

### 0.1 Goal

Implement the minimum production machinery needed to measure the following question without
silently changing it:

> Can complete-rollout estimates identify promising incomplete `c1_null` roots, such that allocating
> terminal-generation budget to the selected **partial roots themselves** improves the
> structure-feasible parent frontier and survives the unchanged v0 Fusion package?

The implementation must support:

- Terminal: independent `c1_null` complete trajectories with complete-sequence Head
  allocation;
- Pre-terminal: the same `c1_null`, `controller=None`, no-h-map trajectories with
  continuation-value allocation at a frozen `rho_edit < 1`, followed by fresh completion
  of ranked partial roots;
- the three T0 policies `selected_partial`, `random_partial`, and `independent_full`,
  plus one shared held-out per-root evaluation table.

This PLAN establishes measurement correctness and launch readiness. It does not establish
that a usable continuation-value signal exists; only T0 can do that. It also does not execute
T0, select a winning maturity, or run P1.

### 0.2 Why this requires a persistent PLAN

The change crosses the sampler state machine, exact RNG replay, remask-aware maturity,
batched continuation, Head input safety, root-value selection, facade ordering, v0 admission
provenance, resume identity, and matched-compute accounting. A superficially successful
implementation can otherwise measure terminal best-of-K while reporting partial-root
allocation. The load-bearing behavior is therefore specified and tested here before cluster
execution.

### 0.3 Task state

| Task | Deliverable | Status |
|---|---|---|
| V1F0 | branch/authority reconciliation and coder gap audit | done (local) |
| V1F1 | exact pre-denoiser checkpoint, maturity crossing, and early stop | done (local) |
| V1F2 | exact replay, disjoint forks, and resumed batch parity | done (local) |
| V1F3 | typed partial-entry state, config, hashing, and seed contracts | done (local) |
| V1F4 | complete continuations, root value, controls, and fresh facade construction | done (local) |
| V1F5 | v0 admission mapping, artifacts, resume, and cost ledger | done (local); anchor-checksum gap below |
| V1F6 | production driver, launcher reuse, docs, and regression closure | done (local) |
| V1F7 | real deployment canaries | done on Della (A/B + terminal-arm smoke + T0 Canary C) |

No task is complete from code alone. Its named artifact, acceptance/falsification criteria,
and validation evidence must all exist.

**"done (local)" means exactly this**: the implementation, its artifacts and its unit/contract
tests exist and pass in one interpreter on a laptop. V1F7's ordinary, anchor-heavy, terminal-arm
and T0 deployment canaries executed on Della on 2026-07-29; they establish deployment contracts
only, not a scientific effect. See the runbook status block.

**Scientific decision frozen on 2026-07-29 — policy-faithful B\***: the `independent_full`
ELIGIBLE pool for the `Q_T0` structure subsample is its exact-Head-ranked Terminal frontier,
`full_pool[:F_cap]`. Selected and random partial pools remain the common `K_EVAL` endpoints of the
roots each policy holds; they are not terminal-Head-truncated a second time. All three pools then
use the same frozen sample size and Head-independent subset rule. `Q_T0=N=4` for the active T0
design; partial subsets are root-balanced (one held-out endpoint per held root, with overlapping
selected/random roots reusing the same endpoint). The implementation and Canary C now satisfy
this contract. Runbook §6.0 is the binding rationale.

**V1F5 closure state**, named here so no downstream task over-claims it:

1. **T0 definitive structure — closed.** The integrated T0 evaluator uses the same target-backbone
   structure path/cache/gate as v0 and Canary C produced all 72 definitive verdicts.
2. **Completed-sequence anchor checksum.** `maturity_telemetry.anchor_preservation` is null.
   Root-level preservation is enforced structurally, so a root-level value would be 1.0 by
   construction; the continuation/facade/elite checksum needs the DPLM alphabet and is not
   computed. Canary B (runbook §11.6) asserts it manually and must report it as manual.

The earlier v0-admission gap is closed: every examined facade row now has an admission verdict,
definitive structure reason/metrics and round-0 fold accounting, and the 2026-07-29 canaries closed
the `design_idx -> admission verdict -> initial particle` crosswalk without a sequence join.

### 0.4 Explicit non-goals

- no FK tuning, new stochastic selector, beta sweep, or proposal-ratio correction;
- no D1/D2/D3, SC-GR, dynamic schedule, controller memory, or fuller RERD re-noising;
- no non-null static C1, position-dependent h-map, h-map-conditioned schedule, or actuator
  retraining;
- no partial-sequence Head score or learned partial-feasibility surrogate;
- no masked `ParticleState`, partial handoff mode inside `FusionConfig`, or partial state in the
  v0 ancestry loop;
- no NetMHCIIpan import or in-loop use;
- no new repair study or terminal repair-config change;
- no parent-seeded editable identities and no structure-derived draft-sequence override;
- no invented `rho`, `B`, `K_EST`, `K_EVAL`, cohort size, margin, or statistical threshold;
- no T0/P1 scientific verdict in an implementation completion claim.

---

## 1. Required Preread and Confirmed Code Gaps

### 1.1 Required preread for the coder

Read before editing:

1. all four authority files named at the top of this PLAN;
2. `PROGRESS.md`, especially the current null-generator and runtime Head state;
3. `inverse_folding/reference_flow/sampler.py`;
4. `inverse_folding/reference_flow/runtime.py`;
5. `inverse_folding/reference_flow/head_scoring.py`;
6. `inverse_folding/reference_flow/fusion/{state,config,runner,selection,objective}.py`;
7. `scripts/run_if_phase_c1.py`, `scripts/run_if_signal_diag.py`, and
   `scripts/run_rf_refine_fusion.py`;
8. `doc/SCRIPTS.md` before driver or SLURM work;
9. the current relevant sampler, Phase-C1, Fusion, and driver tests.

`run_if_signal_diag.py` is diagnostic evidence only. Do not promote its `.pt` snapshots,
Python-hash seed path, or controller-specific decision model into the production V1 contract.

### 1.2 Confirmed current-code facts

These are implementation inputs, not hypotheses:

- `SamplerSnapshot` stores visible state and raw logits, but not RNG state, conditioning
  identity, fixed-token digest, or content integrity metadata.
- The current snapshot is physically written after the step's denoiser forward and before
  sampling. The saved `x_t` equals the pre-denoiser/post-previous-remask state, but the forward
  has already been charged.
- `ResumeState` restores `x_t`, scores, unmask history, and `start_step`; `sample()` then creates
  a new RNG from `config.sampler.seed`. It is not exact uninterrupted replay.
- Snapshot capture is side-effect-free but does not stop the sampler; the sampler still pays the
  remaining trajectory and forcibly completes residual masks.
- Snapshot requests are integer-step based. There is no realized-maturity crossing API.
- `sample_batch()` starts every lane from full mask and exposes neither capture nor resume.
- `run_if_phase_c1.py --resume-from` resumes completed designs, not a root/continuation graph.
- `OnlineHeadScorer.score_batch_same_protein()` does not itself reject mask/unknown tokens.
- `build_initial_population()` sorts source rows by `(design_idx, seed)`, folds in that order,
  and takes the first `N` feasible rows. It does not Head-rerank the source pool.
- `ParticleState` correctly requires complete canonical AA20 plus definitive Head and structure
  results. That invariant must remain unchanged.
- The existing v0 Fusion driver has per-protein atomic resume and terminal artifacts, but it
  does not preserve an explicit root-to-initial-slot admission mapping.
- The current v0 repair+beam preset has `N=4` and eight rounds, and its repair kernel is supplied
  separately through `c1_null.yaml` (source: the frozen v0 final config). These are inherited
  values, not new V1 defaults.
- The current v0 repair driver constructs flat `h=1`, uses `controller=None`, and does not load
  a position-dependent h-map. V1-A extends this null runtime rather than reviving the earlier
  h-map actuator.

### 1.3 Blocking semantic mismatch in the current snapshot

The runbook freezes the root boundary as
`pre_denoiser_after_previous_remask` and counts a snapshot at step `s` as `s` prefix DFE plus
`S-s` tail DFE. Reusing the current diagnostic snapshot would pay the step-`s` denoiser before
capture, then recompute it on resume. V1 must add a true **pre-denoiser** checkpoint boundary.
Renaming the existing post-forward snapshot is not an acceptable fix.

---

## 2. Frozen Runtime and Scientific Contract

### 2.1 Complete-state boundary

The process is:

```text
all-editable-mask `c1_null` trajectory
  -> replayable partial root
  -> complete est/eval/final continuations
  -> deterministic root allocation
  -> fresh complete AA20 parent facade
  -> unchanged v0 initial structure admission
  -> unchanged v0 repair+beam rounds
```

The partial-entry layer ends at the complete facade. It may add an admission-provenance seam to
the v0 driver, but may not change v0 proposal, objective, structure, selection, elite, repair,
or `ParticleState` behavior.

### 2.2 Non-parent generative entry

For both P1 entry arms and all T0 policies:

- every editable position starts as the mask token;
- hard anchors/fixed motifs may be seeded only through the explicit `fixed_tokens` contract;
- the DPLM encoder context must set `use_draft_seq_override=False` so no complete draft sequence
  supplies editable identities;
- `ENTRY_RF_CONFIG` resolves exactly to `c1_null.yaml` with
  `amplification.form=constant_one`;
- `controller=None` throughout entry generation and continuation;
- `H_MAPS_PARQUET` is absent/empty and no h-map object is loaded or passed;
- the target backbone, coordinate mask, DPLM checkpoint/tokenizer, and fixed-position policy are
  shared across arms;
- the frozen Head is called only on complete continuations and never alters denoiser inputs.

A test must show that the V1 entry path actually passes the backbone-only override and rejects
any non-null entry config, controller state, or h-map input. Merely starting the decoder tensor
from masks is not sufficient evidence if a draft-sequence encoder path or position-dependent
input remains enabled.

### 2.3 Editable maturity

The entry layer receives an explicit editable-position mask derived from the actual sampler
residue domain minus all permanent fixed tokens. Do not infer editability from
`unmask_step_by_pos` or from the current mask pattern.

For protein `p`:

$$
E_p = \{i: i \text{ belongs to the sampler residue domain and is not permanently fixed}\}.
$$

At a root boundary:

$$
\rho_{\mathrm{edit}}
=
\frac{\#\{i\in E_p: x_i\in\mathrm{AA20}\}}{|E_p|}.
$$

Also report:

$$
\rho_{\mathrm{known\ sequence\ identity}}
=
\frac{n_{\mathrm{fixed}}+n_{\mathrm{resolved,editable}}}{L_{\mathrm{total}}}.
$$

Rules:

- compute maturity from the current `x_t`, never first-unmask history;
- evaluate it at the pre-denoiser boundary after the previous step's sampling and remask;
- capture the first upward crossing of the requested target;
- transient pre-remask maturity does not count;
- `rho_actual` may exceed `rho_target`, but a root with no unresolved editable residue is invalid
  for the pre-terminal arm;
- `n_editable == 0`, conflicting fixed residues, out-of-range fixed indices, length mismatch, or
  changed anchors are hard failures;
- the production V1 driver captures one `rho_target` per root attempt. A T0 grid runs separate
  matched root-attempt groups, so prefix compute is not amortized across maturities. If a generic
  sampler observer exposes several labels for one physical state, it persists one payload and
  never counts those labels as independent roots or independent evidence.

The snapshot phase string is exactly `pre_denoiser_after_previous_remask`. The checkpoint must
be created before the current step's denoiser call, so a root at step `s` has paid exactly `s`
logical lane-DFE.

`B` means a frozen number of **root-prefix attempts**, not a target number of successful or unique
roots. Every attempt has a preassigned seed and is charged. A no-crossing root, invalid payload,
or equivalence duplicate remains in attempt/coverage telemetry. The driver never backfills after
seeing these outcomes. Let `U` be the canonically ordered set of valid unique root-equivalence
groups produced by those `B` attempts; estimator/evaluation/final work is allocated only over
`U`. If `|U|` is below the phase's frozen capacity requirement, emit
`entry_insufficient_unique_roots` and stop that protein.

### 2.4 Replayable sampler state

Add a new typed continuation state. Do not silently change the diagnostic meaning of the legacy
`ResumeState` or `SamplerSnapshot`.

The new state must contain at least:

| Field | Contract |
|---|---|
| `schema_version` | explicit, validated, and included in every hash |
| `x_t` | exact integer token vector; masks allowed only here |
| `scores` | exact committed-token score vector used by remask ranking |
| `unmask_step_by_pos` | exact history vector for replay and telemetry |
| `start_step`, `n_steps`, `t` | unambiguous next-step semantics |
| `rng_bit_generator`, `identity_rng_state` | deep-copied state for uninterrupted identity replay only |
| `fixed_tokens` | 0-based index/token mapping plus expected-AA/checksum metadata |
| `editable_mask` | exact maturity denominator |
| `snapshot_phase` | frozen phase string from §2.3 |
| conditioning digests | DPLM, tokenizer/alphabet, backbone/test row, coordinate mask, fixed-token policy, null entry config |
| runtime semantics | device/dtype/determinism fields needed to scope exact replay |

Exact replay is promised only under the same frozen checkpoint, tokenizer, config, backbone,
device/dtype semantics, and deterministic runtime. A mismatch fails before a denoiser call.

Use three separate identities:

- `root_id`: lineage identity, unique even when two trajectories converge;
- `root_equivalence_hash`: canonical hash of the state that determines fresh-seed continuation
  behavior (`x_t`, scores, step, conditioning, fixed tokens, config), excluding lineage-only
  identity, `unmask_step_by_pos` telemetry, and the uninterrupted-replay RNG stream;
- `snapshot_payload_hash`: integrity hash of the complete serialized payload, including identity
  RNG state.

Distinct-root beam operates on `root_equivalence_hash`; converged root lineages remain visible in
telemetry but cannot masquerade as independent basin slots. Collapse happens immediately after
the frozen prefix attempts and before any `K_EST` work. The lexicographically smallest `root_id`
is the canonical payload representative; all other member lineages remain convergence telemetry
and receive no tail budget. Est/eval/final seeds bind the equivalence hash, so representative
choice or input order cannot change continuation draws. If too few valid unique roots remain,
fail under the frozen coverage rule; never backfill adaptively.

### 2.5 Capture and early-stop behavior

The sampler needs an additive checkpoint/early-stop surface with these properties:

- capture occurs before the target step's denoiser call;
- capture can return a masked state without terminal residual completion;
- enabling observation without early stop leaves final tokens, RNG consumption, trajectory
  telemetry, and DFE unchanged;
- early stop reports the exact paid prefix DFE;
- failure to cross a requested pre-terminal maturity is explicit, not converted to a terminal
  root;
- legacy calls without the new surface are byte-equivalent to the audited baseline.

The existing diagnostic snapshot may remain post-forward because its raw logits are part of its
contract. V1 uses the new pre-denoiser state and does not require a `[L,V]` logits tensor in every
root payload.

### 2.6 Identity replay and independent forks

Identity replay and stochastic fork are different APIs or mutually exclusive modes:

- **identity replay** restores `identity_rng_state` and must reproduce the uninterrupted suffix;
- **fork continuation** starts from the same root bytes but initializes a new explicit RNG stream
  from a frozen continuation seed;
- a fork seed never overwrites or impersonates identity replay;
- root, estimator, evaluation, final-materialization, Terminal-complete,
  independent-full, and random-membership-selection seed namespaces are disjoint;
- seed derivation is stable and process-independent; never use Python `hash()`;
- the resolved derivation algorithm and every realized seed are persisted;
- any derived-seed collision is a hard error.

The canonical derivation tuple must include at least:

```text
root prefix:
  (seed_schema, campaign_id, phase, split_role, master_seed,
   entry_arm="preterminal", protein_id, rho_id,
   set_tag="root", root_index)

unique-root est/eval continuation:
  (seed_schema, campaign_id, phase, split_role, master_seed,
   entry_arm="preterminal", protein_id, rho_id,
   root_equivalence_hash, set_tag, replicate_index)

equivalence-group final continuation:
  (seed_schema, campaign_id, phase, split_role, master_seed,
   entry_arm="preterminal", protein_id, rho_id,
   root_equivalence_hash, set_tag="final", replicate_index)

Terminal complete trajectories:
  (seed_schema, campaign_id, phase, split_role, master_seed,
   entry_arm="terminal", protein_id, rho_id="none",
   set_tag="terminal_complete", replicate_index)

Independent-full control trajectories:
  (seed_schema, campaign_id, phase, split_role, master_seed,
   control_id="independent_full", protein_id, rho_id,
   set_tag="full", replicate_index)
```

Reuse the existing SHA-256 seed convention, but first encode typed fields with an unambiguous
canonical/length-delimited representation rather than raw separator-joined strings. Canonicalize
`rho_id` before hashing; do not rely on platform float rendering. Random-root subset selection
has its own `policy_id="random_partial"` membership seed in the same
campaign/phase/split namespace but creates no new root, `est`, or `eval` streams. Both
`selected_partial` and `random_partial` therefore use `entry_arm="preterminal"` for the
shared generative evidence. `policy_id` never enters a continuation seed.

Different seeds may legitimately produce identical terminal sequences. Keep their logical rows
and multiplicity because convergence is data. Physical Head scoring may deduplicate identical
sequences for efficiency, but root-value aggregation must restore all logical samples.

### 2.7 Batched continuation

Correct scalar replay is the first gate, but production readiness also requires resumed batch
support because tail continuations dominate T0/P1 DFE.

- Each batch lane carries its own replay/fork state, RNG, fixed tokens, root identity, and cost.
- The first implementation may require a batch to share `start_step`; the driver then groups
  lanes by start step.
- The sampler must give scalar/batch parity when supplied identical per-lane logits. The real
  driver uses one canonical within-protein lane order and a frozen batch size so shard placement
  does not change a protein's grouping.
- Seed identities are independent of input order, shard count, and grouping. Batch size/grouping
  is nevertheless part of the resolved runtime identity; changing it invalidates resume rather
  than claiming cross-shape floating-point bit-equivalence from the real DPLM.
- Report both logical lane-DFE and physical batched denoiser-forward calls.

Do not block initial correctness on a heterogeneous-start-step batch scheduler. Do block T0
launch if the measured scalar path cannot fit the frozen walltime bound.

### 2.8 Complete-rollout Head firewall

Head is called only on decoded, complete sequences. Before every Head batch:

- require exact expected length;
- require uppercase canonical AA20 only;
- reject mask, `X`, gap, unknown, lowercase, empty, or malformed input before the scorer runs;
- bind returned records to requested continuation IDs and sequence digests;
- require one finite `global_risk` per logical continuation;
- fail on missing, extra, reordered-without-identity, NaN, or infinite results.

No partial root, partial logits, imputed residue, or masked string may be scored. The guard must
be outside `OnlineHeadScorer`, at the partial-entry boundary, so the rule cannot be bypassed by
the current predictor's unknown-token handling.

NetMHCIIpan remains absent from imports, config, oracles, selection, and stopping. It is a
separate post-hoc evaluator.

### 2.9 Root value and estimator/evaluation separation

For the frozen V1 estimator:

$$
\widehat V(r)
=
\frac{1}{K_{\mathrm{EST}}}
\sum_{k=1}^{K_{\mathrm{EST}}} R_H(x^{(k)}_T\mid r),
$$

where lower is better and `R_H` is complete-sequence Head `global_risk`.

Rules:

- use the arithmetic mean, not min, median, best-of-K, or an adaptive summary;
- require exactly `K_EST` valid logical samples; do not average over a success-only denominator;
- rank unique roots deterministically by `(value, root_equivalence_hash)`;
- estimator rollouts may score and rank a root but may not become a P1 parent;
- `K_EVAL` continuations are disjoint evidence and may not affect selection or inheritance;
- in T0, each valid unique root-equivalence group receives exactly one common held-out
  `K_EVAL` table; selected-root and
  random-root policies are membership views over that same table, never separate policy-specific
  evaluation draws;
- final materialization uses only fresh `final` seeds;
- structure is not silently inserted into the value estimator; T0 evaluates the predeclared
  complete subsets through the definitive structure path.

No T0/P1 policy retains a scored `est` endpoint. Branch-and-materialize is outside this
implementation contract.

### 2.10 Deterministic root allocation and controls

Implement small, model-independent policies rather than reusing v0 `select_beam()`, whose
inputs and constraints are complete-state specific.

Required policies:

- continuation-value beam over distinct `root_equivalence_hash` values;
- deterministic random distinct-root control from its own seed namespace;
- independent full-trajectory control under the reserved DFE cap;
- terminal exact-Head ranking for Terminal;
- fresh one-root/one-completion materialization for Pre-terminal.

For T0, also build one deterministic, equal-size structure-subset manifest over exactly three
policy pools:

- `selected_partial`: held-out `K_EVAL` endpoints of the value-selected roots;
- `random_partial`: held-out endpoints of the random roots from the same common `K_EVAL` table;
- `independent_full`: the exact-Head-ranked Terminal frontier `full_pool[:F_cap]`.

Eligibility is policy-faithful: the partial policies must not receive a second terminal Head
filter, and `independent_full` must not discard the Terminal policy's endpoint Head allocation.
After eligibility is fixed, all policies use the frozen `T0_STRUCTURE_SUBSET_SIZE=Q_T0` and a
Head-independent deterministic subset rule. For the active T0 design, freeze `Q_T0=N=4`.
Partial-policy sampling is root-balanced: choose exactly one held-out endpoint per held root by a
stable content hash, and reuse the same endpoint when selected/random share a root. The
independent-full policy chooses `Q_T0` rows by the same Head-independent content ordering within
its top-`F_cap` frontier.

Fail before generation unless `Q_T0 == N`, `Q_T0 <= F_cap`, and
`Q_T0 <= N*K_EVAL`. At runtime, require at least `F_cap` valid independent-full survivors before
forming its frontier. If any pool has fewer than `Q_T0` eligible endpoints, T0 is incomplete
rather than silently shrinking a denominator or backfilling.

The T0 independent-full control uses the same frozen `c1_null`, `controller=None`, absent h-map,
backbone, anchors, DPLM checkpoint/tokenizer, sampler semantics, and Head as the partial roots.
Its only method difference is uninterrupted completion without pre-terminal allocation. Include
this identity in the runtime-isolation spy test.

All terminal sorts are total and input-order independent: Terminal and independent-full use
`(head_global_risk, sequence_md5, source_id)`; unique-root allocation uses
`(group_value, root_equivalence_hash)`.

A synthetic test must force the globally best single estimator endpoint to belong to root R1
while root R2 has the better arithmetic mean. The selected inherited object must be R2's partial
root, and its facade row must be produced with a fresh `final` seed.

### 2.11 Facade and v0 admission

The partial-entry layer emits complete canonical rows ordered by the intended entry rank.

- Terminal: exact terminal Head rank from the complete generation pool.
- Pre-terminal: root value rank, even if fresh materialized terminal Head values imply a different
  order.
- Encode this rank as `design_idx`; include a deterministic secondary `seed` field.
- Apply one common, predeclared initial-refold-attempt cap before v0 admission.
- For Pre-terminal, P1 uses one fresh completion per ranked distinct root (source: the runbook's
  one-root/one-parent materialization contract); on structure failure, advance to the next
  ranked root.
- Never duplicate a feasible parent to hide missing coverage.
- Never rerank Pre-terminal by final materialized Head or select an estimator/evaluation rollout.

The v0 admission path may receive an additive audit seam, but its decisions remain: ordered row
scan, definitive structure gate, first `N` feasible rows, then fresh Head/weights.

Persist this exact mapping:

```text
root_id (or complete-control source ID)
  -> continuation_id
  -> facade design_idx
  -> admission attempt and verdict
  -> initial slot / particle_id
  -> terminal lineage / elite
```

Do not reconstruct the mapping from sequence alone. Distinct roots may converge to the same
complete sequence.

The V1 config must validate:

- the facade/refold attempt cap is at least `N`;
- Pre-terminal has at least as many ranked distinct roots as the attempt cap;
- the Terminal complete pool contains at least the attempt cap unless the protein is explicitly
  reported as entry-insufficient;
- P1 `completions_per_ranked_root` is exactly `1`;
- `N` equals the frozen v0 Fusion population size.

### 2.12 Null entry and terminal runtime boundary

Keep two mandatory phase-role identities:

- `ENTRY_RF_CONFIG`: `c1_null` for Terminal and Pre-terminal;
- `TERMINAL_REPAIR_RF_CONFIG`: `c1_null` for both arms.

They require different CLI/config fields because entry and terminal repair are different runtime
roles, but their content digests may be identical. The V1-A validator rejects a non-null entry
config, any controller state, and any h-map path/object. A spy integration test records every
sampler call's phase, config digest, controller state, and h-map presence and verifies the frozen
null boundary.

---

## 3. Matched-Compute and Cost Contract

### 3.1 Primary budget unit

Logical denoiser-forward equivalents (DFE) are the primary entry-generation budget. Let `S` be
the resolved entry sampler step count, `A` the frozen set of `B` prefix attempts,
`d_a^prefix <= S` the DFE actually charged to attempt `a`, `U` the valid unique-root set after
collapse, `s_u` the captured next-step index for unique root `u`, and `F` the set of fresh final
materializations. Actual Pre-terminal entry DFE are:

$$
C_{\mathrm{DFE,actual}}^{\mathrm{Pre}}
=
\sum_{a\in A}d_a^{\mathrm{prefix}}
+ K_{\mathrm{EST}}\sum_{u\in U}(S-s_u)
+ \sum_{j\in F}(S-s_j).
$$

For P1, `F` is the frozen facade-attempt count and there is no `K_EVAL`. For T0, `F=0` and the
analogous `K_EVAL` term is added once for the common per-root evaluation table.

### 3.2 Reserved versus consumed budget

Controls must not receive an outcome-dependent budget. For each protein and maturity:

1. execute exactly the frozen `B` prefix attempts, record every `d_a^prefix`, and collapse valid
   states to `U` without backfill;
2. before opening continuation Head values or structure outcomes, compute a reserved
   Pre-terminal cap
   using the full configured estimator/evaluation work and the **worst-case frozen final-attempt
   cap**, not the number of parents that later happen to succeed;
3. allocate Terminal and the independent-full control the largest integer number of `S`-step
   trajectories that does not exceed this reserved cap;
4. persist the integer remainder;
5. report actual consumed work, early-stop savings, failures, and unused reservation separately.

Let `F_cap` be the common initial-refold-attempt cap and
`r_max = max_u(S-s_u)`. After the unique-capacity gate passes, the reserved P1 Pre-terminal
entry budget is exactly:

$$
C_{\mathrm{DFE,reserved}}^{\mathrm{Pre}}
=
\sum_{a\in A}d_a^{\mathrm{prefix}}
+ K_{\mathrm{EST}}\sum_{u\in U}(S-s_u)
+ F_{\mathrm{cap}}r_{\max}.
$$

The reserved T0 root-allocation budget is instead:

$$
C_{\mathrm{DFE,reserved}}^{\mathrm{T0}}
=
\sum_{a\in A}d_a^{\mathrm{prefix}}
+ (K_{\mathrm{EST}}+K_{\mathrm{EVAL}})
  \sum_{u\in U}(S-s_u),
$$

with no `final` continuation. In P1, Pre-terminal generates one fresh final continuation for
every ranked root admitted to the `F_cap` facade before
structure outcomes are opened; v0 then refolds that fixed facade in order until `N` feasible
parents are found or the cap is exhausted. The actual final-tail DFE use the selected roots'
realized `s_u` and are therefore no greater than the reservation. Do not recompute Terminal
from actual final success, actual `F`, or terminal outcomes.

This is the binding implementation interpretation of runbook §7.2: P1 omits `K_EVAL`; T0 adds
it. It is a launch-blocking rule, not a formula for the coder to choose or tune.

#### 3.2.1 Binding order of operations and invariants

The two arms are made comparable by executing exactly this order, per protein:

1. freeze the common facade cap with `F_cap >= N`;
2. require Pre-terminal to hold `|U| >= unique_root_capacity` distinct ranked roots. This is the
   frozen capacity requirement of §2.3 and is deliberately STRICTER than the facade cap, since
   the capacity chain already guarantees `unique_root_capacity >= F_cap`: a protein must supply
   the full configured basin diversity, not merely enough roots to fill the facade;
3. compute the reserved Pre-terminal DFE `C_reserved` above;
4. give Terminal `M_T = floor(C_reserved / S)` complete trajectories;
5. rank Terminal by exact complete-sequence Head and submit only its top `F_cap` rows to initial
   refold admission.

Both arms therefore attempt at most `F_cap` initial refolds. Two failure modes are distinct and
must never be conflated:

- `|U| < unique_root_capacity`: the protein is reported `entry_insufficient_unique_roots`. Never
  shrink the capacity requirement or the cap for that protein, and never hand the shortfall to
  Terminal as extra budget.
- the capacity gate passed but `M_T < F_cap`: this is not a scientific outcome. It is a DFE
  accounting/off-by-one defect and must fail hard rather than silently produce a Terminal arm
  that cannot fill its facade.

Assert both explicitly: the first as a per-protein coverage status, the second as an
implementation invariant checked at preflight and again at run time.

### 3.3 Event-level ledger

`cost_ledger.jsonl` is append-safe and idempotently aggregatable. Each logical method event has a
stable `event_id`; every physical execution/retry has a distinct `attempt_id` pointing to that
logical event. Each row has at least:

- protein, arm, phase, root/continuation ID, set tag, attempt, and status;
- reserved, consumed, and unused logical lane-DFE;
- physical batched denoiser-forward calls;
- logical Head samples, unique sequences scored, physical Head calls, residues/windows scored;
- structure requests, cache hits, cache misses/real model executions, and failures;
- DPLM, Head, and refold GPU-seconds where measurable, plus walltime;
- retry/resume provenance.

Phases distinguish at least `root_prefix`, `est`, `eval`, `final`, `full_control`,
`initial_refold`, and `terminal_fusion`. Failed or retried work is never silently dropped. Resume
must not double-count a committed logical event. Retry/crash overhead increases physical
operations/GPU time but not the method's matched logical DFE. Per-protein atomic fragments are
validated before aggregate JSONL emission; a truncated/uncommitted fragment is ignored and the
attempt is rerun with a new `attempt_id`.

The ledger reports logical equivalence and physical efficiency separately. Batching may reduce
physical forwards but cannot reduce the matched logical DFE assigned to an arm.

### 3.4 Feasibility preflight

Before any real launch, resolve a cost projection from the configured `B`, `K`, `S`, maturity,
facade cap, terminal Fusion config, and requested protein count. Fail if worst-case DFE, refolds,
GPU time, or walltime exceed the submitted resources.

For one P1 protein, the reserved definitive in-loop refold-request upper bound is:

$$
C_{\mathrm{refold,reserved}}
=
F_{\mathrm{cap}}
+ N\,R_{\mathrm{parent}}\,n_{\mathrm{rounds}},
$$

where `R_parent = structure.max_refolds_per_parent` and `n_rounds` come from the frozen v0
Fusion config. T0 adds exactly `3 * Q_T0` logical definitive structure requests from its frozen
three-policy subset manifest. Add independent-repeat evaluation requests to the relevant phase's
preflight. Cache hits reduce actual model
executions but never reduce the reserved request cap. The preflight prints every term rather
than only the total.

The runbook's approximately 2.9 seconds per v0 P1 refold and RAR 0031's 1.30-fold refold / 1.66-fold
logged-round-time repair+beam multipliers are feasibility anchors only. T0-measured unit costs
replace them before P1. No production limit is inferred from these historical values.

---

## 4. Configuration and Provenance Contract

### 4.1 Separate V1 entry config

Create a typed, fail-fast V1 entry config separate from `FusionConfig`. Do not add
`handoff.mode=partial` to v0.

Required conceptual fields:

| Field group | Required content |
|---|---|
| identity | schema version, campaign ID, phase (`t0|p1`), split role, `entry_arm=terminal|preterminal`, master seed; T0 policy identity stored separately |
| entry | `c1_null` config identity, `controller_enabled=false`, `h_maps_present=false`, backbone-only mode |
| maturity | one frozen `rho_target` for P1 or a predeclared grid for T0 |
| allocation | `B` prefix attempts, `K_EST`, T0-only `K_EVAL`, unique-root capacity, `N`, estimator=`mean_global_risk` |
| facade | initial-refold attempt cap, P1 one-completion-per-ranked-root law, ordering law |
| budget | reserved-cap rule, hard DFE/refold/walltime caps |
| controls | exact T0 enum `selected_partial|random_partial|independent_full`, random-membership seed, `Q_T0`, and structure-subsampling seed |
| batching | continuation batch size and same-start-step grouping policy |
| artifacts | schema versions and output policy |

Production values for `rho`, `B`, `K_EST`, `K_EVAL`, caps, margins, and cohort size have no
silent defaults. Unit-test fixtures may use tiny explicit values. The resolved production YAML
is created only after the user freezes the runbook inputs.

Validation rejects fields that do not belong to an arm/phase, mismatched `N`, a terminal
`rho_target`, overlapping seed sets, insufficient root/pool capacity, and any config that would
allow estimator/evaluation endpoints into the P1 facade. It also rejects a non-null entry RF
config, enabled controller, non-empty h-map path/object, or any arm label outside
`terminal|preterminal`; T0 policy labels outside
`selected_partial|random_partial|independent_full` are rejected.

`split_role` is one of `t0_dev|p1_dev|p1_holdout|canary`. Preflight binds the requested cohort,
dev/holdout manifests, and the runbook's prior-cohort exclusion manifest; it rejects overlap,
holdout use during calibration, missing requested proteins, or coverage inferred only from
successful outputs.

### 4.2 Content provenance

Persist and bind resume to content digests, not path/size/mtime fingerprints alone:

- arm and V1 entry config;
- entry RF config and terminal repair RF config separately;
- DPLM checkpoint, tokenizer/alphabet identity, and DPLM runtime config;
- Head checkpoint, variant, allele index, inference/window config, and score scale;
- explicit `controller_enabled=false` and `h_maps_present=false` runtime sentinels;
- test-set row, backbone/coordinate source, and coordinate mask;
- hard-anchor/fixed-token manifest, expected residues, and 0-based index convention;
- requested cohort, dev/holdout split, and prior Fusion-cohort exclusion manifests;
- seed manifest;
- v0 Fusion config;
- code commit, dependency/runtime lock, device/dtype/determinism fields;
- snapshot, table, and cost-ledger schema versions;
- exact CLI and resolved output paths.

A regression test changes file content while preserving size and mtime and verifies that resume
is invalidated.

### 4.3 Path policy

All cluster data, run, log, checkpoint, cache, and model paths enter through CLI arguments or
launcher environment variables. No Python module hardcodes `/scratch`, a username, or a local
worktree path.

---

## 5. Artifact Contract

All tables have stable schemas even when empty, explicit schema versions, unique join keys, and
canonical ordering on write. Large state payloads are content-addressed sidecars; Parquet rows
carry their integrity hash and relative artifact path.

### 5.1 Required entry artifacts

| Artifact | Grain | Load-bearing content |
|---|---|---|
| `root_attempts.parquet` | frozen prefix attempt | attempt seed, paid prefix DFE, crossing/payload/equivalence status, explicit failure/duplicate reason |
| `partial_roots.parquet` | root lineage | root/equivalence/payload IDs, seed, phase, step, target/actual maturity, payload path, all conditioning/config hashes |
| snapshot payloads | root state | exact replayable fields in §2.4, integrity checked before use |
| `maturity_telemetry.parquet` | root | length/fixed/editable/resolved counts and masks, `rho_edit`, known-identity fraction, coordinate-valid fraction, editable-segment summary, anchor preservation |
| `continuations.parquet` | logical continuation | continuation/root IDs, `set_tag=est|eval|final`, seed, sequence/digest, Head result, DFE, status, convergence multiplicity metadata |
| `root_values.parquet` | root | expected/observed K, arithmetic mean, validity, equivalence group, deterministic rank |
| `root_selection.parquet` | selection decision | policy/control, rank, selected flag, tie-break, source value, fresh-materialization authorization |
| `complete_entry_pool.parquet` | Terminal/independent-full candidate | source ID, seed, sequence/digest, exact terminal Head, rank, DFE |
| `t0_control_membership.parquet` | T0 endpoint x policy | exact `selected_partial|random_partial|independent_full` membership, common-eval provenance, deterministic subset key |
| `t0_structure_subset.parquet` | T0 logical structure request | exactly `Q_T0` rows per policy, source endpoint, subset rank, Head-independent sampling |
| `t0_structure_results.parquet` | T0 definitive structure result | target-backbone metrics/gate, request/cache/model status, failure reason |
| `terminal_parent_facade.parquet` | ordered complete attempt | standard generated columns plus source/root/continuation IDs, entry rank, config hashes |
| `terminal_parent_admission.parquet` | facade attempt | rank, structure attempt/verdict, cache status, admitted slot/particle ID or failure reason |
| `cost_ledger.jsonl` | event | §3.3 fields |
| `manifest.json` | run | resolved config, content provenance, seed inventory, requested/succeeded/failed coverage, artifacts, exact command |
| `cohort_coverage.parquet` | requested protein x arm/phase | frozen split/exclusion identity, requested/preflight/root/entry/terminal status and failure stage |

`continuations.parquet` keeps duplicate terminal sequences from distinct logical seeds. If Head
deduplicates physical evaluation, the mapping back to every logical continuation must be exact.

### 5.2 Standard v0 artifacts

After facade admission, emit the unchanged standard v0 Fusion artifact family:

- `fusion_candidates.parquet`;
- `fusion_particles.parquet`;
- `fusion_lineage.parquet`;
- `fusion_elite.parquet`;
- `fusion_rounds.jsonl`;
- `fusion_failures.parquet`;
- `generated.parquet`;
- v0 manifest fields, augmented only with V1 entry/admission provenance.

### 5.3 Replay requirement

The artifacts must reconstruct without inference or sequence-based guessing:

```text
root payload
  -> est/eval evidence
  -> selected partial root
  -> fresh final completion
  -> ordered facade
  -> definitive v0 admission slot
  -> terminal population lineage and elite
```

Any missing edge, duplicate ID, hash mismatch, seed collision, or stale config invalidates the
protein/run.

---

## 6. Code Ownership and Reuse-First Boundary

The exact internal decomposition may change after the coder's V1F0 audit, but responsibilities
must remain separated.

### 6.1 Expected library surfaces

- `inverse_folding/reference_flow/sampler.py` or one adjacent sampler-state module:
  additive pre-denoiser checkpoint, early stop, exact identity replay, explicit fork seed,
  resumed batch lanes, and DFE instrumentation.
- a new torch-free partial-entry module under
  `inverse_folding/reference_flow/fusion/`:
  root/continuation identities, maturity validation, canonical hashes, Head firewall, value,
  deterministic root policies, facade ordering, and pure cost arithmetic.
- a separate typed V1 entry-config module under the Fusion package.
- an additive v0 admission-audit seam if needed. Existing `ParticleState`, proposal/selection,
  objective, structure gate, repair, and elite behavior remain unchanged.

Do not make the pure Fusion package import DPLM, torch, pandas, the Head implementation, or NMP.
Model/runtime/IO wiring stays in scripts or existing runtime adapters.

### 6.2 Driver boundary

The expected production entry is `scripts/run_rf_fusion_v1_entry.py`, because no current
production driver owns the root/continuation/control/facade artifact graph. It must reuse or
extract, not copy:

- DPLM task/backbone/denoiser preparation from the existing runtime and C1 driver;
- constraint validation, explicit no-h-map preflight, and content provenance;
- the existing Head scorer construction;
- the v0 Fusion driver through its complete `--generated-parquet` boundary.

Do not turn `run_if_signal_diag.py` into the production driver. Do not duplicate model loading or
oracle construction across scripts; extract shared helpers to library modules when importing a
large script would create circular or side-effectful coupling.

### 6.3 Launcher boundary

Apply the project reuse gate before creating a SLURM file. The preferred first implementation is:

- extend `scripts/submit_if_phase_c.slurm` with a disjoint V1-entry mode for DPLM + Head;
- reuse `scripts/submit_refine.slurm MODE=fusion` for the terminal complete-state stage;
- use submit-time resource overrides where the existing static header is unsuitable.

The current Phase-C launcher uses an allele h-map default when `H_MAPS_PARQUET` is merely
unset. Any reused V1 mode must explicitly set it empty, omit `--h-maps-parquet`, and fail
preflight unless the resolved value is empty.

A new orchestration launcher is allowed only if the coder records why neither existing launcher
covers at least 60% of the required execution and why a thin wrapper cannot safely compose them
(source: `AGENTS.md` §5 Reuse-First Gate). Any new script/launcher must be registered in
`doc/SCRIPTS.md` before the task is complete.

---

## 7. Build Tasks

### Task V1F0 — Reconcile authority and re-audit the target branch

**Objective:** ensure the coder edits the actual v0 Fusion baseline under the current project
contract and finds no drift that changes the scientific or implementation path.

**Assumptions/inputs:** the audited baseline and four authority documents at the top of this
PLAN.

**Artifacts/interfaces:** fill the V1F0 audit record in Appendix A; update this PLAN only if file
ownership or an interface must change. Do not write a `LOG.md` entry for the audit and do not
rewrite frozen scientific decisions.

**Acceptance:**

- target branch contains v0 Fusion plus the current V1 docs/PLAN;
- project governance is unambiguous;
- coder confirms the facts in §1.2 against current code;
- any mismatch affecting snapshot phase, replay, Head semantics, facade admission, or cost is
  reported before implementation.

**Falsifier/stop:** the target branch lacks the validated v0 engine, authority files conflict,
or a current interface makes a frozen contract impossible without a scientific decision.

**Validation:** record branch, commit, clean/dirty state, relevant file inventory, and the
resolved ownership/reuse decision in Appendix A. A load-bearing mismatch stops for user review;
an empty mismatch list authorizes V1F1.

### Task V1F1 — Exact checkpoint, maturity crossing, and early stop

**Objective:** create a true pre-denoiser partial root whose logical cost and dynamic state are
exact.

**Assumptions/inputs:** controller-free `c1_null` sampler with no h-map, explicit
editable/fixed masks, deterministic fake denoiser.

**Artifacts/interfaces:** new typed sampler continuation state; capture/early-stop result; DFE
counter; legacy snapshot preserved.

**TDD RED cases:**

- step `s` checkpoint currently charges `s+1` forwards;
- remask trace crosses maturity before remask but falls below afterwards;
- `unmask_step_by_pos` incorrectly appears mature after remask;
- early stop accidentally runs terminal residual completion;
- capture-on versus capture-off changes the uninterrupted trajectory.

**Acceptance:**

- checkpoint at step `s` has exactly `s` prefix lane-DFE;
- tail contract is exactly `S-s` lane-DFE;
- first post-remask upward crossing is correct with and without anchors;
- fixed residues are excluded from `rho_edit` but included in known-identity telemetry;
- root retains at least one unresolved editable residue;
- editable/fixed/resolved/unresolved masks and counts are emitted as raw recomputable
  telemetry; the implementation does not invent a T0 "meaningful action" threshold;
- instrumentation is trajectory/RNG neutral;
- legacy sampler calls and diagnostic snapshots remain compatible.

**Falsifier/stop:** phase or DFE cannot be made unambiguous, or the sampler cannot return a
masked root without executing the terminal suffix.

**Validation:** targeted pure sampler/maturity tests plus existing sampler and constraint suite.

### Task V1F2 — Exact replay, fork namespaces, and resumed batch parity

**Objective:** make identity replay auditable while enabling independent, efficient tail
continuations.

**Assumptions/inputs:** V1F1 state; stable seed helper; controller remains `None`.

**Artifacts/interfaces:** exact-replay mode, fork mode, resumed batch lane fields, final replay
digest/telemetry.

**TDD RED cases:**

- current `ResumeState` resets RNG from config;
- est/eval/final share a seed;
- selected/random membership labels change a shared root/`est`/`eval` seed;
- T0 and P1 derive the same seed for the same protein/root tag;
- two shards derive different seeds for the same logical continuation;
- batch regrouping changes a fake-denoiser result despite identical per-lane logits;
- equal sequences from two fork seeds are deduplicated as one logical sample.

**Acceptance:**

- uninterrupted and checkpoint/resume runs match terminal tokens, scores, unmask history, and
  final RNG state under the frozen runtime;
- tampered state/RNG/hash fails before sampling;
- all seed namespaces are deterministic, disjoint, and collision-checked;
- `selected_partial` and `random_partial` reuse the same preterminal root/`est`/`eval`
  seeds; only random membership selection has its own seed;
- same root + same fork seed is reproducible;
- scalar and grouped batch continuation are identical under the deterministic fake-denoiser
  parity contract;
- logical duplicates remain distinct lineage rows;
- logical DFE and physical batched forwards reconcile.

**Falsifier/stop:** exact same-runtime replay fails on the real DPLM canary after deterministic
runtime requirements are met. Do not downgrade the claim to approximate replay silently.

**Validation:** pure fake-denoiser tests, scalar/batch parity, different `PYTHONHASHSEED`, input
order, fixed-group shard partition, and runtime-identity invalidation on batch-policy changes.

### Task V1F3 — Partial-entry state, config, hashes, and seeds

**Objective:** make the scientific objects and provenance independent of model/IO code.

**Assumptions/inputs:** §§2-4 and V1F1/V1F2 state.

**Artifacts/interfaces:** torch-free root/continuation records, V1 config loader/validator,
canonical hashes, seed manifest, payload serialization/integrity helpers.

**TDD RED cases:**

- malformed masks or lengths are accepted;
- identical visible roots with different IDs fill multiple beam slots;
- equivalent roots receive tail work before canonical collapse;
- failed/duplicate roots are adaptively backfilled until `B` unique roots appear;
- changed checkpoint/config/anchor content reuses stale state;
- a non-null entry config, controller, or h-map input is accepted;
- production config silently supplies missing scientific values;
- same-size/same-mtime content edits evade resume invalidation.

**Acceptance:**

- all dataclasses/configs fail fast on invariant violations;
- hashes distinguish lineage, continuation equivalence, and payload integrity correctly;
- root equivalence collapse is deterministic and still reports converged lineages;
- the only accepted V1-A entry runtime is `c1_null`, `controller=None`, and
  `h_maps_present=false`;
- no production scientific knob has an invented default;
- content changes invalidate state/resume;
- all fields round-trip without lossy stringification.

**Falsifier/stop:** serialization cannot round-trip exact sampler/RNG state or a required
conditioning input cannot be content-bound.

**Validation:** pure unit tests, corruption tests, schema round-trip tests, and config matrix.

### Task V1F4 — Complete continuations, root value, controls, and facade

**Objective:** implement the exact objects consumed by T0 and the Terminal/Pre-terminal entry
arms without endpoint leakage.

**Assumptions/inputs:** frozen Head, V1F2 forks, V1F3 policies/config, complete-sequence decoder.

**Artifacts/interfaces:** Head-guarded continuation evaluator; value table; selected-root table;
random/independent-full controls; common held-out evaluation table; three-policy T0 structure-subset
manifest; ordered complete facade.

**TDD RED cases:**

- masked/unknown input produces a plausible finite Head score;
- continuation output order is mismatched to Head records;
- best estimator endpoint is inherited instead of the best-mean root;
- failed continuations shrink the mean denominator;
- materialized Head reranks Pre-terminal;
- duplicate terminal sequences erase logical multiplicity;
- selected and random policies draw separate `K_EVAL` continuation tables;
- T0 structure policies use different sample sizes or Head-dependent subset rules;
- the independent-full or partial path accepts a non-null config, controller, or h-map.

**Acceptance:**

- Head sees only validated complete AA20 and exact continuation identities;
- value is the exact `K_EST` arithmetic mean;
- est/eval/final sets are disjoint and provenance-complete;
- selected/random comparisons reuse the same per-root held-out evaluation rows;
- selected partial roots, never their scored endpoints, authorize P1 final materialization;
- no estimator endpoint can enter any T0/P1 facade;
- Pre-terminal facade order follows root rank despite reversed terminal Head values;
- Terminal ranks by exact complete Head;
- one-root/one-fresh-completion and common attempt-cap rules hold;
- the T0 structure manifest contains exactly `Q_T0` Head-independently sampled endpoints per
  `selected_partial|random_partial|independent_full` policy or fails closed;
- all three policies resolve to the same `c1_null`, controller-free, no-h-map runtime;
- all controls and logical costs are reconstructable.

**Falsifier/stop:** any code path can pass a partial sequence to Head or put an est/eval endpoint
into the Pre-terminal P1 facade.

**Validation:** synthetic T0 oracle tests and a fake Head spy that observes zero invalid calls.

### Task V1F5 — v0 admission mapping, artifacts, resume, and ledger

**Objective:** connect the complete facade to the unmodified scientific behavior of v0 while
preserving exact ancestry, coverage, and cost.

**Assumptions/inputs:** V1F4 facade, v0 structure cache/oracles/driver.

**Artifacts/interfaces:** all §5 artifacts; additive admission audit; definitive evaluation of
the frozen T0 structure-subset manifest through the same v0 target-backbone structure path;
per-protein atomic checkpoint/resume; cost aggregation.

**TDD RED cases:**

- root rank and terminal Head rank disagree;
- rank-0 materialization fails structure and a later root passes;
- distinct roots converge to one sequence;
- cross-arm/policy checkpoint is reused;
- interrupted retry double-counts cost;
- protein order or shard count changes semantic outputs;
- missing/duplicate requested proteins disappear from coverage.

**Acceptance:**

- facade rank controls admission order exactly;
- every attempt maps through initial slot to terminal lineage without sequence guessing;
- v0 rechecks definitive structure and complete Head as before;
- common attempt cap is enforced and failures remain visible;
- all three T0 structure policies use the same backend, target backbone, absolute gate, subset
  size, and frozen outcome-independent membership;
- resume identity binds every §4.2 input and arm;
- one-process, sharded, reordered, and interrupted/resumed results are equal after canonical sort
  under the same frozen within-protein batch policy;
- ledger totals reproduce formula-level DFE, Head, refold/cache, and walltime accounting;
- existing v0 artifacts and behavior remain compatible when no V1 entry metadata is supplied.

**Falsifier/stop:** admission provenance cannot distinguish convergent roots, or V1 requires
masked/unevaluated state inside v0.

**Validation:** two-protein fake-oracle driver smoke, resume/sharding equivalence, and complete v0
Fusion regression suite.

### Task V1F6 — Production driver, launcher reuse, docs, and regression closure

**Objective:** expose the validated library path as a reproducible cluster-ready workflow.

**Assumptions/inputs:** V1F1-V1F5 complete; `doc/SCRIPTS.md` reread.

**Artifacts/interfaces:** production entry CLI, print-config/preflight mode, launcher mode or
justified new launcher, registered script docs, updated runbook placeholders, one substantive
`LOG.md` entry when implementation lands.

**Acceptance:**

- CLI separates entry config from terminal repair config;
- `--print-config` resolves all method values, digests, budgets, and seed inventory without
  running models;
- a dry-run validates all requested proteins, backbones, anchors, output paths, resource caps,
  null entry/repair configs, disabled controller, and absent h-map;
- no NMP import or flag exists in the entry runtime;
- paths are supplied through CLI/environment;
- launcher stdout/stderr resolve under the logs layer;
- every new script is registered;
- runbook §11 contains the real driver/launcher names and exact canary/T0/P1 command templates;
- existing RF/C1/signal-diagnostic/Fusion behavior is unchanged when V1 is unused; those
  regression surfaces are compatibility checks, not V1 method dependencies.

**Falsifier/stop:** the launch path requires copying existing script bodies, accepts a non-null
entry config/controller/h-map, or cannot print a fully resolved budget before model execution.

**Validation:** CLI help, print-config, dry-run, Python compile, shell syntax, targeted suites,
and project regression surfaces below.

### Task V1F7 — Deployment canaries

**Objective:** verify that the model-backed runtime satisfies the exact contracts before T0.

**Assumptions/inputs:** frozen implementation commit and real DPLM/Head/backbone artifacts; these
are deployment checks, not scientific evidence.

**Artifacts/interfaces:** two canary run directories and manifests; uninterrupted-versus-replay
digest comparison; Head-call audit; root/continuation/facade/admission chain; cost ledger; Q00511
anchor audit; resolved configs, exact commands, and logs.

**Canary A — ordinary unconstrained protein:**

- resolved entry/repair roles both use `c1_null`, with `controller=None` and
  `h_maps_present=false`;
- full editable mask and backbone-only DPLM context;
- pre-denoiser root with unresolved residues;
- uninterrupted versus identity-resume equality;
- multiple disjoint tail forks;
- complete-AA20 Head calls only;
- ordered facade accepted by v0;
- measured DFE/Head/refold timing.

**Canary B — anchor-heavy Q00511:**

- resolved entry/repair roles both use `c1_null`, with `controller=None` and
  `h_maps_present=false`;
- use the real frozen constraint manifest;
- verify 0-based expected residues and all anchor checksums;
- verify anchors are excluded from `rho_edit`, included in known-identity telemetry, and preserved
  through root, every continuation, facade, v0 initial population, and terminal elite;
- use only as a contract/application smoke, not evidence for the generic trajectory claim.

**Acceptance:** both canaries emit the full replay chain and pass every implementation invariant.

**Falsifier/stop:** identity replay, anchor preservation, Head firewall, config isolation, or cost
reconstruction fails. Keep the runbook `PRE-IMPLEMENTATION — DO NOT LAUNCH` and return to the
failed task.

**Validation:** execute the committed runbook canary commands and assert every named artifact,
hash, coverage row, and invariant above. Record the exact implementation commit and run paths in
the runbook; do not infer a scientific effect from either canary.

---

## 8. Test Matrix

### 8.1 New targeted tests

Expected test surfaces (names may change during V1F0, responsibilities may not):

1. `tests/inverse_folding/test_reference_flow_sampler_continuation.py`
   - pre-denoiser phase, early stop, DFE, exact replay, fork, corruption;
2. `tests/inverse_folding/test_reference_flow_fusion_v1_*.py` and
   `tests/scripts/test_rf_fusion_v1_*.py`
   - maturity, hashes, seed namespaces, Head firewall, value, policies, facade;
3. `tests/scripts/test_run_rf_fusion_v1_entry.py`
   - artifacts, coverage, resume/sharding, config separation, fake-oracle end to end;
4. additive cases in the existing Fusion runner/driver tests for admission provenance and legacy
   compatibility.

### 8.2 Required adversarial cases

- `s` prefix calls and `S-s` suffix calls exactly;
- transient maturity `0.4 -> 0.8 -> post-remask 0.5 -> 0.75`: capture only `0.75`;
- no-anchor, duplicate-anchor, conflicting-anchor, and anchor-heavy paths;
- snapshot capture on/off neutrality;
- uninterrupted/resumed equality and final RNG equality;
- all seed namespaces disjoint under different process hash seeds, orders, batches, and shards;
- campaign/phase/split seed separation and no adaptive root-attempt backfill;
- mean-root winner conflicts with best endpoint;
- selected/random policy views reuse one common held-out `K_EVAL` table;
- invalid Head input rejected with Head call count zero;
- duplicate sequence multiplicity preserved;
- incomplete `K_EST` invalidates the root;
- root rank conflicts with materialized Head rank;
- first-ranked structure failure advances to next root without rerank/duplication;
- equal-size, Head-independent
  `selected_partial|random_partial|independent_full` T0 structure subsets;
- Terminal, Pre-terminal, and independent-full all resolve to `c1_null`,
  `controller=None`, and no h-map;
- a non-empty h-map argument/environment default or non-null entry config fails before sampling;
- same-size/same-mtime provenance mutation invalidates resume;
- retry/resume does not duplicate ledger events;
- V1-disabled legacy behavior is byte-compatible where promised.

### 8.3 Required local verification commands

The coder may split commands for runtime, but all surfaces must pass:

```bash
pytest \
  tests/inverse_folding/test_reference_flow_sampler.py \
  tests/inverse_folding/test_reference_flow_sampler_constraints.py \
  tests/inverse_folding/test_reference_flow_sampler_continuation.py -q

pytest \
  tests/inverse_folding/test_reference_flow_fusion_*.py \
  tests/inverse_folding/test_reference_flow_v1_null_isolation.py \
  tests/scripts/test_rf_fusion_v1_*.py \
  tests/scripts/test_run_rf_refine_fusion.py \
  tests/scripts/test_run_rf_fusion_v1_entry.py -q

pytest \
  tests/scripts/test_run_if_phase_c1_script.py \
  tests/scripts/test_run_if_phase_c1_d2_d3.py \
  tests/inverse_folding/test_reference_flow_runtime.py \
  tests/inverse_folding/test_reference_flow_signal_diag.py \
  tests/inverse_folding/test_reference_flow_sampler_controller.py \
  tests/inverse_folding/test_reference_flow_d1_controller.py \
  tests/inverse_folding/test_reference_flow_d2_d3_controller.py -q

python -m py_compile \
  inverse_folding/reference_flow/sampler.py \
  scripts/run_rf_fusion_v1_entry.py \
  scripts/run_rf_refine_fusion.py

bash -n scripts/submit_if_phase_c.slurm
bash -n scripts/submit_refine.slurm
```

The legacy C1/controller suites above are regression surfaces for the shared sampler API; they
are not dependencies or actuators of V1-A.

If V1F0 selects different new filenames, update this section and the runbook before marking the
task complete. Do not leave commands pointing to non-existent placeholders.

---

## 9. Launch and Scientific Decision Boundary

### 9.1 Code-complete is not experiment-ready

After local completion:

1. run both V1F7 canaries;
2. replace runbook placeholders with exact committed commands;
3. freeze both runtime roles to `c1_null` and verify `controller=None`,
   `h_maps_present=false`, and no resolved h-map argument;
4. freeze cohort manifests, seed manifest, T0 values, cost caps, tests, and margins;
5. run `--print-config` and preflight from the exact cluster commit;
6. only then change the runbook status from `PRE-IMPLEMENTATION`.

### 9.2 T0 remains the scientific gate

Passing this PLAN means the system correctly measures partial-root continuation allocation. It
does not support HT1 or HT2. T0 must still show independent continuation-value fidelity,
selected-over-random benefit, compute-matched full-trajectory non-domination, structure-feasible
frontier retention, and non-degenerate remaining action under the frozen runbook rule.

If T0 fails, stop V1-A. Do not tune maturity, K, estimator, selector, or cohort until a positive
appears.

### 9.3 P1 interpretation remains bounded

- Pre-terminal versus Terminal is the sole primary P1 contrast under the same `c1_null`
  generator, matched compute, and terminal v0 package.
- It tests reward-facing allocation before completion as a system contrast, not a
  position-dependent schedule contribution or a clock-time-only intervention.
- report both pre-Fusion parent and post-Fusion elite frontiers;
- post-hoc NMP evaluates transfer only;
- constrained Q00511 transfer cannot rescue a negative generic cohort.

---

## 10. Coder Completion Checklist

- [x] Implemented on a descendant of the audited Fusion branch under reconciled governance.
- [x] Reported any PLAN/code mismatch before changing a frozen scientific contract.
- [x] Added a true pre-denoiser, post-previous-remask checkpoint boundary.
- [x] Checkpoint at step `s` charges exactly `s` prefix DFE and returns without completion.
- [x] Legacy snapshot/resume behavior remains explicitly separate and compatible.
- [x] Identity replay restores RNG and matches uninterrupted sampling under the frozen runtime.
- [x] Fork continuations use deterministic, disjoint, persisted seed namespaces.
- [x] Resumed batch lanes match scalar semantics and report logical versus physical cost.
- [x] `rho_edit` excludes every fixed token and uses current `x_t`, not unmask history.
- [x] Entry generation forces all editable mask, `c1_null`, `controller=None`,
      `h_maps_present=false`, and backbone-only DPLM context.
- [x] Head receives complete canonical AA20 only; NMP is absent.
- [x] Root value is the exact `K_EST` arithmetic mean with a fixed denominator.
- [x] Root equivalence prevents duplicate-state beam slots while retaining convergence telemetry.
- [x] Est/eval rollouts never enter the Pre-terminal P1 facade.
- [x] Final materialization uses fresh seeds and preserves root rank despite terminal Head values.
- [x] Terminal complete-Head ranking and Pre-terminal root ranking are implemented distinctly.
- [x] Entry and terminal repair roles both resolve to `c1_null`; controller and h-map absence
      are hashed/recorded and spy-tested.
- [x] Facade rank maps explicitly to v0 admission slot and terminal lineage.
- [x] Common initial-refold attempt cap and insufficient-coverage behavior are visible.
- [x] Reserved control budget is frozen before continuation/structure outcomes; actual usage is
      reported separately.
- [x] Artifacts reconstruct root -> evidence -> selection -> fresh parent -> v0 lineage.
- [x] Resume uses content hashes and is invariant to order/batch/shard partition.
- [x] New driver/launcher surfaces satisfy reuse-first and are registered in `doc/SCRIPTS.md`.
- [x] One `LOG.md` entry is added only when the behavior-changing implementation lands.
- [x] `PROGRESS.md` is overwritten with the current V1-A implementation/deployment state.
- [x] Targeted, regression, fake-oracle, ordinary-protein, and Q00511 canary gates pass.
- [x] Real canaries and all frozen T0 inputs exist; the 24-protein scientific T0 executed and
      froze `rho_target=0.50`. P1 remains a separate post-T0 experiment gate.

---

## Appendix A — V1F0 Audit Record

The coder fills this block before V1F1. It is a planning/implementation handoff record, not a
`LOG.md` entry.

```text
audit_date:
implementation_branch:
implementation_commit:
worktree_status_summary:
authority_docs_present:
governance_reconciled:
confirmed_code_facts:
load_bearing_mismatches: []
resolved_library_ownership:
resolved_driver_path:
resolved_launcher_reuse_decision:
decision: STOP_FOR_REVIEW | PROCEED_V1F1
```
