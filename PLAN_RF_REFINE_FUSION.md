# RF-Refine Fusion Implementation Plan

> **Status:** READY FOR CODER after the scientific contract in
> `doc/RF-Refine-Fusion.md` passed the final code audit.
>
> **Execution contract:** implement task-by-task with RED -> GREEN tests. This PLAN
> specifies behavior, ownership boundaries, interfaces, artifacts, and gates. It does
> not prescribe line-by-line implementation. Checkboxes are for the implementing coder.

**Goal:** turn the current RF generator plus standalone refinement evidence into one
Head-guided, structure-constrained, hard-state edit-and-repair process. Complete RF
sequences become a population; broad edits and RF/DPLM local repair generate complete
children; the frozen Head evaluates whole-sequence immune burden; only definitively
structure-feasible children may enter ancestry; greedy/beam/FK selectors carry exact
accepted sequences into later rounds.

**Authoritative scientific source:** `doc/RF-Refine-Fusion.md`.

**Evolution boundary:** this file supersedes only the deferred "Unified Architecture"
in `PLAN_RF_REFINE.md` §11. The existing NMP-driven standalone refiner and its PLAN remain
an experimental mechanism record. They are not the runtime objective or implementation
template for this Fusion path.

**Core claim implemented here:** Head-guided FK **optimization**, not exact twisted-SMC
posterior sampling. No proposal-ratio correction is required in v0.

---

## 0. Read First

### 0.1 Required preread

Before editing code, read:

1. `doc/RF-Refine-Fusion.md`, especially §§4-10 and §12.
2. `inverse_folding/reference_flow/sampler.py` (`fixed_tokens`, remask lifecycle,
   `ResumeState`, `sample_batch`).
3. `inverse_folding/reference_flow/runtime.py` (backbone preparation and frozen DPLM
   encoder/decoder contexts).
4. `inverse_folding/reference_flow/head_scoring.py` (`HeadScore`, window coordinates,
   `global_risk`, batch scorer).
5. `inverse_folding/reference_flow/refine.py` and `scripts/refine_rf_designs.py` only as
   evidence/reuse references. Do not carry their NMP objective into Fusion.
6. `doc/SCRIPTS.md` and `scripts/CLAUDE.md` before driver/SLURM work.

External `ProDifEvo-Refinement/` and `Fk-Diffusion-Steering/` are read-only scientific
references. Production code must not import them.

### 0.2 Current-code facts that the implementation must respect

- `controller.t_start` is not a phase handoff. It gates controller refresh/logit work;
  it does not materialize particles or terminate sampler/controller state.
- `sample_batch()` contains independent lanes, not same-protein particle interaction.
- `ResumeState` restarts from visible sampler state but does not clone RNG or controller
  state. It is not the Fusion particle representation.
- `fixed_tokens` is sufficient for a terminal-handoff v0 repair call: freeze the sequence
  outside a halo plus protected edited positions, leaving only repair positions masked.
- sampler remask is config-gated and off by default. The post-handoff core does not use
  sampler-level global remask, Stage-A rank, or D3 rank.
- `HeadScore.global_risk` and aligned window records already exist. They have never been
  used as a complete-state ancestry selector.
- the existing refiner's structure gate (`reference_flow.refine.structure_gate`) is
  **seed-relative** (`cand.scTM < seed.scTM - eps`) and cannot be reused for Fusion's absolute
  gate; Fusion needs an absolute target-backbone `scTM >= scTM_min` on every ancestry-eligible
  state (only the refold+TMalign `struct_fn` wiring and the `StructureMetrics` container are
  reusable). The NMP-orientation lives in the refiner's objective (core-count), not the gate.

### 0.3 Frozen v0 architecture decisions

These are not coder choices:

1. **Runtime immune signal is Head-only.** NMP has no driver flag, injected oracle,
   candidate field, selector input, or stopping role. It may be run after Fusion as an
   external evaluator only.
2. **The persistent state is complete.** At every correction-round boundary, every
   particle is an exact complete sequence. Temporary masks exist only inside an RF move.
3. **v0 handoff is terminal-population handoff.** Initial particles are complete designs
   produced by the existing RF generator. Earlier partial-step materialization is deferred
   until the hard-state loop passes its mechanism gates. This is allowed by the scientific
   contract: chronological completion of the first RF sequence does not make repeated
   backbone-conditioned repair a detached filter.
4. **No legacy controller state crosses the handoff.** Do not carry sampler `scores`,
   unmask history, sticky/pending D2 evidence, D3 `m_i`, pressure state, or remask rank.
5. **Current complete Head responsibility chooses where.** v0 targets the highest-risk
   Head window of each parent. SC-GR and typed actionability are optional future proposal
   allocators, not dependencies.
6. **Broad explicit edits are first-class.** RF structural logits may be recorded or used
   for optional ordering, but they never define a hard AA whitelist.
7. **RF repair is controller-free.** Local repair calls use the frozen backbone denoiser
   with `controller=None`. D2 may later be tested as a proposal-efficiency ablation; it is
   not part of the v0 identity.
8. **Structure is a hard complete-state gate.** Any child entering beam/population
   ancestry must have a definitive structure result. A raw proposal may be screened before
   refolding but cannot inherit on a surrogate alone.
9. **The parent is always the null move.** All-failed rounds keep feasible parents; they
   never fall back to infeasible children.
10. **The returned design is the elite archive.** Population dynamics may move uphill;
    the best-so-far feasible Head state is never lost.

### 0.4 Explicitly deferred

- partial-step `t_star < T` handoff and exact sampler RNG continuation;
- exact twisted-SMC/proposal-ratio correction;
- D2 marginal logits, sticky evidence, D3 survival rank, or global remask after handoff;
- SC-GR allocation of beam width/round count;
- learned structure surrogate;
- NMP-in-loop guidance of any kind;
- multi-allele joint objective (v0 is one configured Head/allele per run).

---

## 1. Core Runtime Contract

### 1.1 State objects

Create a dedicated package, `inverse_folding/reference_flow/fusion/`. The state layer uses
immutable dataclasses and plain serializable fields.

| Object | Required fields | Contract |
|---|---|---|
| `ParticleState` | `particle_id`, `protein_id`, `round_idx`, `slot_idx`, `sequence`, `sequence_md5`, `weight`, `parent_particle_id`, `source_proposal_id`, `head_global_risk`, `structure`, `feasible`, `lineage_seed` | `sequence` is complete canonical AA20; no mask token or missing evaluation for an inherited state |
| `Proposal` | `proposal_id`, `parent_particle_id`, `move_family`, `sequence`, `edited_positions`, `target_start_0b`, `target_end_0b`, `halo_start_0b`, `halo_end_0b`, `proposal_seed`, optional generation diagnostics | Proposal identity survives evaluation; do not collapse it into per-position marginals |
| `CandidateEvaluation` | exact `sequence_md5`, parent/target identity, whole-sequence Head score, aligned windows, local target delta, off-target new-hotspot burden, structure result, feasibility/reason | Only this evaluated exact sequence may be selected |
| `PopulationState` | `round_idx`, exactly `N` particle slots, normalized weights, current elite | Population size is fixed after initialization |
| `EliteState` | best feasible `ParticleState`, first/last round seen | Stored outside stochastic resampling; updated from **all definitively-feasible evaluations** each round (not only survivors) on strict Head improvement, so an ε_H-rejected or FK-unsampled feasible best is never lost |

Validation is fail-fast:

- sequence length must match the target backbone and parent;
- sequence characters must be canonical AA20;
- inherited states must have `sum(weights) == 1` within tolerance;
- every inherited state must have a definitive structure verdict;
- IDs are stable and process-independent (no Python `hash()`).

### 1.2 Head objective and responsibility

For v0:

$$
R_H(x) = \texttt{HeadScore.global_risk}(x).
$$

`global_risk` is an **attribute** (`float | None`), not a method — access it as
`head_score.global_risk`. It is emitted by the dynamic predictor
(`epitope_head/inference/predictor.py:382,705`) and populated by `score_batch_same_protein()`.
A `None` `global_risk` from a **fresh dynamic** scoring call is a hard error (do not silently
substitute LME, max-window score, or zero). The reachable-`None` case that is **not** an error:
HeadScores reconstructed from the static parquet cache (`head_scoring._populate_from_rows`) are
windows-only and carry `global_risk=None` by design — so selection must source `global_risk`
from a fresh `score_batch_same_protein()` call, never from the static cache. Score all unique
sequences for one protein in one `score_batch_same_protein()` call per evaluation stage.

The target register is the highest raw-risk Head window of the current parent:

1. sort by `z` descending;
2. deterministic ties: `start_0b`, then `end_0b`, then `k` ascending;
3. use the exact half-open window span as the target;
4. exclude immutable anchor positions from the editable set, not from Head responsibility;
5. if the selected window has no editable positions, try the next window; if none are
   editable, emit `stalled_no_editable_target` and retain the parent.

Local target improvement is telemetry, not acceptance:

$$
\Delta R_{\mathrm{local}}(y;x,B)
=
\operatorname{LME}_{w\cap B\neq\varnothing} z_w(y)
-
\operatorname{LME}_{w\cap B\neq\varnothing} z_w(x).
$$

Whole-sequence `global_risk` controls selection.

### 1.3 Parent-relative no-new-hotspot gate

Do not reuse static-prior subtraction. Parent and child windows must align exactly on
`(start_0b, end_0b, k)` or evaluation fails.

For windows outside the repair halo:

$$
N_H(y;x,H_B)
=
\max_{w\cap H_B=\varnothing}
\max\left(0, z_w(y)-z_w(x)\right).
$$

If no off-halo windows exist, `N_H=0`. A child is hotspot-feasible iff
`N_H <= objective.max_offtarget_window_increase`. Also record off-target positive mass and
count for telemetry, but v0 hard-gates on the maximum increase.

### 1.4 Structure gate and evaluation budget

The definitive gate is absolute to the target backbone:

- `scTM >= structure.scTM_min` (required config; no placeholder/default in the loader) —
  `scTM = run_tmalign(pred, ref)["tm_score"]`;
- `max_anchor_sidechain_RMSD <= structure.max_anchor_sidechain_RMSD_max` when
  `structure.active_site_metric: sidechain_max_anchor` — **hard-gated whenever the protein
  carries hard anchors**. Predicted coordinates receive one global full-trace C-alpha Kabsch
  transform, then every anchor is compared through its exact all-heavy-side-chain atom set.
  ASP/GLU/PHE/TYR naming symmetry is resolved by minimum squared error. The gate uses the worst
  per-anchor RMSD, while pooled active-site RMSD and worst atom displacement remain telemetry.
  Identity mismatch, missing atoms, an unsupported/no-side-chain anchor, non-finite values, or
  incomplete anchor coverage fails closed; there is no common-atom fallback or local active-site
  realignment. `legacy_ca_shell` remains an explicit compatibility mode for historical configs;
- optional `scRMSD <= structure.scRMSD_max` when configured — `scRMSD := run_tmalign["rmsd"]`
  (TMalign aligned-subset RMSD); telemetry / optional ceiling, distinct from the full-trace
  Kabsch `sc_rmsd`;
- a metric that is **configured and applicable but cannot be computed** (e.g. length mismatch for
  the active-site gate on an anchored protein, or a non-finite `scTM`/RMSD) fails closed, never
  silent pass. For a protein with **no** active-site constraint the active-site gate is N/A and
  skipped cleanly. `has_active_site` is derived from `anchors`; the runner/handoff fail-fast if
  an anchored protein lacks the ceiling matching its selected metric. All configured structure
  comparisons treat `None`/`NaN`/`inf` as failures.

Evaluation is two-tier:

1. Optionally evaluate a cheap structure surrogate on the raw pool. The v0
   `structure.surrogate_mode` is `none`, which contributes a neutral factor and performs no
   rejection; do not invent a structural-logit threshold before calibration.
2. Head-score and new-hotspot-check the deduplicated raw pool.
3. Per parent, shortlist at most `structure.max_refolds_per_parent` by ascending global
   Head risk, with deterministic ties by proposal ID. Always retain the null parent.
4. Run the definitive structure oracle on every shortlisted non-parent child.
5. Only definitively feasible children enter a selector.

The structure cache key includes sequence MD5, target-backbone digest, refold backend, and
structure-config hash. A cache hit is valid only if all provenance fields match.

### 1.5 Move families

The move layer returns exact `Proposal` objects. v0 supports three families:

1. `explicit_edit`: AA20 substitutions inside the target register, excluding anchors.
   **Always emit every single** — full breadth over every editable position is the load-bearing
   lever, so `max_raw_candidates_per_parent` caps only the deterministic-seeded distinct-position
   **pairs** added on top; it must never truncate singles (which biases coverage toward low-index
   positions). Raw candidates are only Head-scored (cheap, batched) so the singles count may
   exceed the cap; the binding cost is refolds (`max_refolds_per_parent`), not raw candidates.
   Never generate the unchanged parent as an edit.
2. `rf_reopen`: freeze the complement of the target/halo and regenerate the reopened span
   from the backbone-conditioned DPLM sampler.
3. `edit_repair`: apply an explicit edit, protect edited positions, freeze the complement
   of the repair halo, and regenerate only unprotected halo context.

The direct edit and repaired descendants may both enter evaluation. Repair must not silently
revert the protected immune edit.

For every RF call:

- `controller=None`;
- outside-halo tokens are supplied through `fixed_tokens`;
- protected edited tokens and hard anchors are also in `fixed_tokens`;
- any remask is local to unfixed halo positions and explicitly configured;
- output must be complete;
- outside-halo, edited-core, and anchor identity are asserted after sampling;
- `fixed_tokens` is a flat `{pos: token_id}` map with no halo/target abstraction, so `moves.py`
  owns the halo↔complement enumeration and must place edited-core and anchor tokens **into**
  `fixed_tokens` (positions merely left out of `fixed_tokens` are regenerated, so exclusion is
  not protection);
- batched repair children need a **distinct per-child sampler seed** (base seed + child index):
  `SamplerBatchLane` has no per-lane sampler-seed field and all lanes otherwise seed from
  `config.sampler.seed`, which would make identical-`fixed_tokens` children produce identical
  repairs.

The frozen encoder context must not carry a **structure-derived draft prediction**. The
native/parent sequence is already removed before `forward_encoder` (both builders call
`inject_noise(..., noise='full_mask')`, `runtime.py:449/484`); the only object to suppress is
the draft `init_pred`/`logits` from the `use_draft_seq=True` branch
(`dplm_invfold.forward_encoder:160-170`). v0 uses a `backbone_only` repair-context policy: add a
**tri-state** optional `use_draft_seq_override: bool | None = None` to the two encoder context
builders (`build_dplm_denoiser_context` `runtime.py:454`, `build_batched_dplm_denoiser_context`
`:489`); `None` falls back to `bool(task.hparams.generator.use_draft_seq)` so existing callers
stay byte-identical, while Fusion passes `False` to force backbone-only regardless of the
checkpoint config. Do not touch the `make_*` denoiser wrappers or `generate_native_sequence(s)`
(they read the same flag but are out of scope). A parent-conditioned draft encoder is deferred.

### 1.6 Selection laws

Implement three selectors over the same evaluated candidate pool.

**Greedy control**

- Each parent independently takes its lowest-Head feasible child only if
  `R_H(child) <= R_H(parent) - min_head_improvement`; otherwise it keeps itself.
- No ancestry crosses parent lanes.
- The resulting deterministic control population resets weights to `1/N`.

**Hard-beam control**

- A child is beam-eligible only if it improves on its own parent by the Head margin,
  `R_H(child) <= R_H(parent) - min_head_improvement` (the same `ε_H` as greedy). Per
  `doc/RF-Refine-Fusion.md` §4 and §5.5, greedy and hard-beam are both **conservative,
  require-improvement** controls; they differ only in greedy=per-parent-lane vs beam=global
  flatten. Feasible null parents are always eligible.
- Flatten the beam-eligible children together with the feasible parent/null states.
- Keep the `N` lowest-Head exact states, with deterministic ties by sequence MD5 then
  proposal ID.
- Preserve at least one copy of the elite.
- The resulting deterministic control population resets weights to `1/N`.

**FK optimization**

For parent `a` with `M_a` ancestry-eligible children plus its null parent, assign branch
proposal mass `q_a = 1/(M_a+1)`. The unnormalized state weight is

$$
\widetilde w_r^{(n)}
=
w_{r-1}^{(a)} q_a
\mathbf 1[\mathrm{feasible}]
\exp\left[-\beta_r R_H(y_r^{(n)})
+\beta_{r-1}R_H(x_{r-1}^{(a)})\right].
$$

Here `M_a` is the number of ancestry-eligible children and the `+1` is the retained null
parent, so each of the `M_a+1` branches gets `q_a = 1/(M_a+1)`; `x_{r-1}^{(a)}` is the exact
round-`(r-1)` parent that produced child `n` (the object the doc §5.6 labels `x_r^{(a)}`), and
the potential telescopes to `exp[-β_r R_H(x_r)]`. This is the v0 offspring-count normalization. Normalize globally and multinomial-resample
exactly `N` slots with replacement using a stable round seed. After resampling, weights reset
to `1/N`. Compute ESS before resampling as telemetry. v0 FK resamples every round; adaptive
ESS skipping is deferred so population cardinality cannot become ambiguous.

The elite archive is not itself forced into stochastic weights. It is copied separately to
the final output and may optionally occupy one protected beam slot in the beam control.

### 1.7 Handoff and initial population

v0 reads complete RF designs from the existing generated-parquet surface. It does not use
Head to preselect initial particles.

For each protein:

1. order source rows by `design_idx`, then `seed`;
2. validate sequence length/canonical alphabet and hard anchors;
3. definitively structure-score rows until `N` feasible initial slots are found;
4. fail that protein with `insufficient_feasible_initial_population` if fewer than `N`
   feasible rows exist; do not duplicate a state to hide missing input coverage;
5. Head-score the selected `N`, initialize weights from the configured `beta_0`, and create
   the elite from the lowest-Head feasible state.

Duplicate sequences are retained if they are distinct generator slots; report initial unique
sequence fraction so population collapse is visible.

### 1.8 Cost and feasibility model

The binding constraint is structure evaluation, not Head or (absent) NMP. Sourced unit costs:

- one ancestry-eligible child costs one definitive structure eval = one ESMFold forward + one
  TMalign vs the target backbone + one global C-alpha alignment and anchor side-chain comparison
  that **reuse the same predicted PDB** (no extra ESMFold forward), `(protein_id, sequence)`-cached.
  The ESMFold forward dominates; parsing/alignment is cheap CPU, and the full-atom reference
  context is parsed once per protein.
- Head is cheap and batched: one `score_batch_same_protein` call per protein per evaluation
  stage covers the whole dedup pool [audit of `head_scoring.py`].
- empirical anchor: the standalone refiner measured ~2003 refolds → ~80 min GPU on H2ETE7,
  i.e. order **~2.4 s / refold** amortized, which is why it caps refolds [refiner §7 `refold_cap`].
  Treat this as order-of-magnitude, not a promise.

Per-protein definitive-refold count is bounded by

$$
\text{refolds} \le N \times \texttt{max\_refolds\_per\_parent} \times \texttt{n\_rounds},
$$

with `(protein_id, sequence)` caching removing repeats. The feasibility gate for any configured
run is `refolds × (~2.4 s) ≤ SLURM walltime budget`. This inequality — not a chosen default —
bounds the joint calibration of `N`, `max_refolds_per_parent`, and `n_rounds` (§2.1). A
configuration whose worst-case refold count exceeds the walltime budget must fail fast at config
load, per `AGENTS.md` §2.

---

## 2. Config Contract

Create a separate loader at `inverse_folding/reference_flow/fusion/config.py`. Do not add
Fusion fields to `ControllerConfig`: the post-handoff loop is not a D1/D2/D3 controller mode.

The YAML has one top-level `fusion:` block.

### 2.1 Parameter provenance (sourced vs calibration gate)

No numeric default in this PLAN is invented (`AGENTS.md` §2). Each v0 parameter is one of two kinds.

**Sourced — carry the value with its citation:**

- `moves.explicit.max_edit_order = 2` (singles + distinct-position pairs; triples deferred) —
  refiner `PLAN_RF_REFINE.md` §7 (`max_points=2`, "singles+pairs primary").
- structure backend = ESMFold + TMalign vs the target backbone — refiner §2; audit.
- a per-parent refold cap must exist; the refiner's ~16/round is an **upper-bound reference**,
  not a v0 default — refiner §7 `refold_cap`.

**Calibration gate — NO default value; a named quantity fixed from smoke telemetry (§6):**
`population_size N`, `n_rounds`, `beta_0` / `beta` schedule, `min_head_improvement`,
`max_offtarget_window_increase`, absolute `structure.scTM_min` (the refiner's *relative*
`seed − 0.05` gate does NOT transfer to an absolute floor, so it carries no value here),
`structure.max_anchor_sidechain_RMSD_max` (the strict active-site side-chain ceiling; no default
when `active_site_metric: sidechain_max_anchor`, fixed from smoke telemetry on the anchored cohort),
`targeting.halo_radius`, `max_raw_candidates_per_parent`, `pair_seed_budget`,
`moves.repair.repair_shortlist_per_parent`, and the repair sampler step count.

The loader rejects a missing calibration-gate value (no silent default); each is fixed by the §6
procedure (S-smoke telemetry under the §1.8 walltime bound) and recorded with its provenance in
the run manifest. `max_refolds_per_parent` is jointly calibrated with `N` / `n_rounds` against the
§1.8 inequality.

| Section | Required fields / v0 defaults | Validation |
|---|---|---|
| `fusion` | `enabled`, `seed`, `population_size`, `n_rounds` | positive `N`, non-negative rounds |
| `handoff` | `mode: terminal_population` | reject unsupported partial mode in v0 |
| `targeting` | `mode: head_max_window`, `registers_per_parent: 1`, `halo_radius: 4` | only frozen v0 mode accepted |
| `objective` | `head_field: global_risk`, `min_head_improvement`, `max_offtarget_window_increase` | finite values; global risk only |
| `moves.explicit` | enabled, `max_edit_order: 2`, `max_raw_candidates_per_parent`, `pair_seed_budget` | edit order 1 or 2; positive budget |
| `moves.rf_reopen` | enabled, children per parent | at least one move family enabled |
| `moves.repair` | enabled, `repair_shortlist_per_parent`, children per edit, sampler steps, local remask flag | shortlist/budgets non-negative |
| `structure` | backend (`esmfold2_live` current; `esmfold` compatibility), `surrogate_mode: none`, required `scTM_min`, `active_site_metric: sidechain_max_anchor`, required `max_anchor_sidechain_RMSD_max`, optional legacy CA-shell fields, optional `scRMSD_max`, cache dir, `max_refolds_per_parent` | reject unknown modes or missing selected threshold; active-site gate N/A for no-anchor proteins; exact atom correspondence and non-finite metrics fail closed |
| `selection` | `mode: greedy|beam|fk`, beta schedule, `resample_method: multinomial` | beta length `n_rounds+1` or valid linear endpoints |
| `telemetry` | candidate detail, window detail, trajectory detail flags | defaults keep candidate + lineage evidence |

All config values are printed at startup, one key per line, and serialized into the run
manifest with a canonical hash. Paths arrive through CLI, never module constants.

Add one conservative smoke preset:

`inverse_folding/reference_flow/configs/rf_refine_fusion_smoke.yaml`

It is an engineering preset, not a scientific default. It uses a small population/round and
proposal budget; the structure threshold and Head margins must remain explicit in the file.

---

## 3. File Ownership

| File | Responsibility | Action |
|---|---|---|
| `inverse_folding/reference_flow/fusion/__init__.py` | public Fusion surface | create |
| `inverse_folding/reference_flow/fusion/config.py` | typed config + validation/hash-ready dict | create |
| `inverse_folding/reference_flow/fusion/state.py` | immutable particles, proposals, evaluations, population, elite | create |
| `inverse_folding/reference_flow/fusion/objective.py` | Head-window alignment, target/local metrics, new-hotspot gate | create |
| `inverse_folding/reference_flow/fusion/moves.py` | explicit edits + RF reopen/edit-repair adapters | create |
| `inverse_folding/reference_flow/fusion/selection.py` | greedy, beam, FK weights/resampling, ESS | create |
| `inverse_folding/reference_flow/fusion/runner.py` | per-protein round orchestration, exact feedback, caches, telemetry | create |
| `inverse_folding/reference_flow/fusion/oracles.py` | injected Head/structure/RF protocols and lightweight runtime adapters | create |
| `inverse_folding/reference_flow/runtime.py` | tri-state `use_draft_seq_override` on the two encoder context builders (`:454`/`:489`) only; not `make_*`/`generate_*` | minimal backward-compatible change |
| `scripts/run_rf_refine_fusion.py` | I/O and real-oracle wiring | create; justified because existing refiner is NMP-specific and lifecycle differs by >40% |
| `scripts/submit_refine.slurm` | branch a `fusion` mode to `run_rf_refine_fusion.py` (no NMP flags); reuse GPU/refold launcher | modify, do not add a second SLURM file |
| `doc/SCRIPTS.md` | register new driver and Fusion mode | modify |
| `tests/inverse_folding/test_reference_flow_fusion_*.py` | pure/unit/integration tests | create |
| `tests/scripts/test_run_rf_refine_fusion.py` | fake-oracle driver smoke | create |

Do not modify `controller.py`, `counterfactual.py`, `commit.py`, `allocation.py`, or the
legacy NMP search law in `refine.py` for the core implementation.

---

## 4. Build Tasks

### Task F1: State and config contracts

**Files:** `fusion/state.py`, `fusion/config.py`, unit tests.

- [ ] Write failing tests for complete/canonical sequence validation, normalized population
  weights, missing structure verdict, stable IDs, invalid beta schedule, missing `scTM_min`,
  and rejection of unsupported handoff modes.
- [ ] Implement the immutable state objects and config loader.
- [ ] Add canonical config serialization and hashing.
- [ ] Keep `fusion.config`, `fusion.state`, and package import lightweight: `fusion.__init__`
  must not eagerly load torch, ESMFold, Head checkpoints, or external repos.

**Gate F1:** pure tests pass; invalid scientific states fail before any model call.

### Task F2: Complete-state Head objective and targeting

**Files:** `fusion/objective.py`, tests with synthetic `HeadScore` records.

- [ ] Test deterministic top-window responsibility and anchor-skipping fallback.
- [ ] Test exact window-coordinate alignment and mismatch failure.
- [ ] Test local LME delta separately from global selection risk.
- [ ] Test parent-relative off-halo hotspot maximum, mass, and count.
- [ ] Test `global_risk=None` failure.
- [ ] Implement only the frozen v0 target and objective modes.

**Gate F2:** a local improvement that creates a larger off-target hotspot is rejected; target
selection remains deterministic.

### Task F3: Explicit breadth proposer

**Files:** `fusion/moves.py`, pure proposal tests.

- [ ] Test complete AA20 singles, no unchanged token, anchor exclusion, and length preservation.
- [ ] Test distinct-position pair generation, deterministic cap/seed, sequence deduplication,
  and stable proposal IDs.
- [ ] Test no-editable-target behavior returns no proposals rather than mutating anchors.
- [ ] Implement proposal generation without Head/NMP/structure dependencies.

**Gate F3:** proposal breadth is not restricted to D2 structural top-k support.

### Task F4: Definitive evaluator and structure firewall

**Files:** `fusion/oracles.py`, evaluator portion of `fusion/runner.py`, tests with fake oracles.

- [ ] Define injected batch Head and per-sequence structure protocols.
- [ ] Test deduplicated one-batch Head scoring per protein/stage.
- [ ] Test deterministic per-parent Head shortlist and bounded refold count.
- [ ] Test that every ancestry-eligible child has `structure_evaluated=True` and passes the
  absolute gate.
- [ ] Test unavailable configured RMSD metrics fail closed.
- [ ] Test the side-chain active-site gate: on an anchored protein a child whose
  `max_anchor_sidechain_RMSD` exceeds its ceiling is infeasible even when scTM passes; incomplete
  anchor atom correspondence fails closed; on a no-anchor protein the gate is N/A.
- [ ] Test cache provenance mismatch forces recomputation/failure, never silent reuse.
- [ ] Ensure null parents retain their existing definitive evaluation.

**Gate F4:** no unrefolded or structure-failed child can reach any selector.

### Task F5: Minimum exact-state loop (greedy, explicit edits)

**Files:** `fusion/runner.py`, integration tests with fake Head/structure.

- [ ] Initialize an `N`-slot complete population without Head preselection.
- [ ] Run target -> explicit proposals -> complete evaluation -> greedy transition.
- [ ] Carry the selected exact sequence bytes into the next round.
- [ ] Keep the null parent when no child passes all gates.
- [ ] Maintain a monotone elite independent of working-state movement.
- [ ] Emit in-memory candidate/particle/round records.

Required integration cases:

1. a Head-improving feasible child persists unchanged into the next round;
2. a better local but worse global child is not selected;
3. a lower-Head structure-failed child cannot enter ancestry;
4. all children fail -> parent survives;
5. no NMP callable/import is needed.

**Gate F5 (first usable mechanism):** exact complete-state feedback works end-to-end before
RF repair or FK complexity is introduced.

### Task F6: RF reopen and edit-repair kernel

**Files:** `fusion/moves.py`, minimal change to `runtime.py`, sampler-backed tests using a fake
denoiser plus one runtime integration smoke.

- [ ] Add optional `use_draft_seq_override` to single/batched DPLM context builders; omitted
  value preserves every existing caller.
- [ ] Build a `backbone_only` context for Fusion.
- [ ] Implement direct target/halo reopen with `fixed_tokens` outside the halo.
- [ ] Implement protected-core edit-repair with outside halo + edited core + hard anchors fixed.
- [ ] Use one fixed round order: generate direct edits/direct reopens; Head-score them;
  choose the `repair_shortlist_per_parent` lowest-global-risk new-hotspot-feasible explicit
  edits; generate repaired descendants; Head-score repaired descendants; combine direct and
  repaired exact states; apply the definitive structure shortlist/gate; then select.
- [ ] Assert outside-halo and protected positions are byte-identical after sampling.
- [ ] Assert no mask tokens survive.
- [ ] Ensure `controller=None`; no post-handoff D2, D3, Stage-A rank, or global remask state.
- [ ] Batch repair children through the existing batched denoiser where protein/backbone and
  sampler settings match; give each child a **distinct per-child sampler seed** (base seed +
  child index) so batched repairs are diverse, not identical — `SamplerBatchLane` has no
  per-lane sampler-seed field and all lanes otherwise seed from `config.sampler.seed`.

**Gate F6:** the denoiser can change repair context while never reverting the challenged edit
or altering the sequence outside its allowed halo.

### Task F7: Beam and FK population selection

**Files:** `fusion/selection.py`, selector integration in `runner.py`.

- [ ] Test greedy parent-local behavior and hard-beam global behavior.
- [ ] Test offspring-count normalization with unequal branch sizes.
- [ ] Test infeasible weight zero, null-parent positive support, stable log-space weight
  normalization, ESS, deterministic multinomial resampling, and post-resample `1/N` weights.
- [ ] Test duplicated descendants preserve ancestry IDs and exact sequence identity.
- [ ] Test elite survival when stochastic population loses the elite lineage.
- [ ] Add selector-mode dispatch without changing proposal/evaluator behavior.

Use log weights and log-sum-exp; never exponentiate unbounded raw Head values directly.

**Gate F7:** FK changes descendant allocation rather than token logits, and the same evaluated
candidate pool can be replayed through greedy/beam/FK controls.

### Task F8: Real driver, artifacts, and resume

**Files:** `scripts/run_rf_refine_fusion.py`, driver tests.

Reuse (import paths verified against current code):

- generated-input loading from `load_generated_designs` in the **script module**
  `scripts/evaluate_phase_c.py` (namespace import `scripts.evaluate_phase_c`; requires repo root
  on `sys.path` — there is no `scripts/__init__.py`), which requires only columns
  `{protein_id, design_idx, sequence}`; or replicate the flexible sharded loader **pattern**
  `_load_sharded` in `scripts/refine_rf_designs.py` (module-private — copy the pattern or promote
  it to a shared module, do not cross-import);
- `scripts.run_if_phase_c1.build_head_scorer` for the frozen Head (keyword-only
  `window_k_min`/`window_k_max` are required; needs `setup.allele`,
  `setup.config.head.score_scale='raw_logit'`, and a `Path`-typed `head_checkpoint`);
- `inverse_folding.reference_flow.runtime` for IF task/backbone/denoiser contexts;
- structure lives in **three different modules** — canonical import block is
  `scripts/refine_rf_designs.py:717-720`:
  `from inverse_folding.evaluation.refold import load_refold_model, refold`;
  `from inverse_folding.evaluation.tmalign import run_tmalign`;
  `from inverse_folding.reference_flow.runtime import resolve_structure_path`
  (`run_tmalign` returns `{"tm_score","rmsd",...}`; `pred_pdb`/`ref_pdb` argument order is
  load-bearing — Chain_1 = the first arg normalizes the TM-score);
- `inverse_folding.reference_flow.constraints.load_constraint_manifest` for hard anchors
  (`HardAnchor.index_0b`/`expected_aa`, `hard_anchor_indices`; validate with
  `ActiveSiteConstraint.validate_against_sequence`, 0-based, no-op for `unconstrained_control`);
- reuse the refiner's `struct_fn` refold+TMalign wiring and the `StructureMetrics` container, but
  **define a Fusion oracle bundle without an `nmp_fn` field** — the refiner's `Oracles` namedtuple
  has a mandatory `nmp_fn`, and copying it verbatim would force a `StandaloneRunner` import
  (violating §0.3(1)).

Do not import or instantiate NetMHCIIpan / `StandaloneRunner`.

The real `struct_fn` must, in addition to refold+TMalign scTM, use the lightweight
`inverse_folding.evaluation.structural_metrics_v2` API in side-chain mode. Cache one
`ReferenceContext` per protein and request only `global_ca_rmsd`, `plddt`, and `sidechain`; never
request residue rows in the search loop. Populate distinct `StructureMetrics` fields
(`global_ca_RMSD`, `active_site_sidechain_RMSD`, `max_anchor_sidechain_RMSD`,
`max_anchor_atom_distance`, `active_site_complete`) rather than reusing legacy
`active_site_RMSD`. The standalone benchmark evaluator remains responsible for the complete
summary and residue-indexable artifacts.

For `structure.backend: esmfold2_live`, start one persistent
`inverse_folding.evaluation.esmfold2_live.ESMFold2LiveClient`. The worker uses the active
`immune-design` Python executable and imports its torch before overlaying Biohub's `esm` and
`transformers` packages from an explicit `--esmfold2-site-packages` path. The overlay must stay
inside the worker because Biohub and fair-esm share the top-level `esm` namespace. Reuse the
worker across all candidates and normalize each output to the shared PDB/pLDDT cache.

Required CLI surfaces:

- generated input/run directory, test-set parquet, PDB root;
- base IF checkpoint and RF sampler config used by repair;
- Fusion config;
- Head checkpoint/config/variant/allele/device/batch size;
- refold cache, ESMFold2 package/model/sampling settings, and TMalign settings;
- optional constraint manifest;
- output directory, protein filter, device, resume, and `--print-config`.

Implement per-protein atomic checkpointing and deterministic resume. The resume identity binds
the full input set (Fusion config hash + Head/manifest/base-IF/sampler/window-range signature +
per-protein seed digest), so any changed input re-runs rather than reusing a stale checkpoint.

> **v0 scope (deferred, LOG L0134):** resume granularity is **per-protein**, not round-boundary.
> A resumed protein re-runs from round 1 (it does not restart from a persisted mid-run
> round/population/elite/RNG state). This is acceptable for v0 smokes (n_rounds ≤ 3). Operational
> caveat for the H3/repair arm: DPLM `sampler.sample()` repair outputs are **not** disk-cached
> (only the ESMFold refold of the resulting sequence is cached by sequence MD5), so an interrupted
> repair-enabled protein re-charges its serial repair GPU calls on restart under the §1.8 walltime
> bound. Round-boundary checkpointing (persisted round/population/elite/RNG lineage state + a
> repair-output cache) is a follow-up, not part of the v0 delivery.

**Gate F8:** fake-oracle two-protein CLI smoke writes every required artifact and resumes
without duplicate lineage rows; changing an input (seed/Head/manifest) re-runs rather than
reusing a stale checkpoint.

### Task F9: Artifact contract

The driver writes:

| Artifact | Grain | Required content |
|---|---|---|
| `fusion_candidates.parquet` | proposal/candidate evaluation | parent/proposal/target/move IDs, exact sequence+MD5, edit/halo positions, Head parent/child/global/local deltas, new-hotspot metrics, structure metrics/evaluated/feasible/reason, ancestry eligibility, selection count |
| `fusion_particles.parquet` | round x population slot | exact sequence, weight, lineage IDs, source proposal, Head/structure values, elite flag |
| `fusion_lineage.parquet` | selected edge | parent particle, child particle, round, proposal, selector mode, multiplicity |
| `fusion_rounds.jsonl` | protein x round | raw/dedup/Head-shortlist/refold/feasible counts, ESS, unique fraction, resampling flag, elite risk, cache hits, wall time |
| `fusion_elite.parquet` | protein | initial/final sequence and Head/structure, improvement, rounds, lineage leaf |
| `generated.parquet` | evaluator-compatible elite output | columns `load_generated_designs` requires: `protein_id, design_idx, sequence` (emit `seed, wall_seconds` only as optional extras — do NOT fabricate them); one elite per protein with `design_idx=0`; source particle/design provenance stays in `fusion_elite.parquet` |
| `fusion_failures.parquet` | failed protein/stage | reason, exception class/message, last completed round |
| `manifest.json` | run | config/hash, checkpoint/head/backbone provenance, input hashes, git SHA, artifact paths, NMP-absent flag |

Optional detailed window rows may be written to a sidecar when enabled; do not embed large
window arrays in JSONL. Use Parquet structured/list columns rather than stringified Python
objects where supported.

Add artifact schema tests and join-key uniqueness checks. `particle_id`, `proposal_id`, and
lineage edges must be globally unique within a run.

### Task F10: Launcher, docs, compatibility, and LOG

- [ ] Extend `scripts/submit_refine.slurm` with a `fusion` mode (parameterize, do not add a new
  SLURM file). Current structure: `MODE` (`refine|ceiling`, line 44) is a **pass-through** to
  `refine_rf_designs.py --mode` with **no switch/case**, and the single `srun` block (lines
  173-213) unconditionally passes `--netmhciipan-bin` (line 188) and the `--nmp-*` flags. So
  `MODE=fusion` requires **branching** the `srun` block to invoke `scripts/run_rf_refine_fusion.py`
  with its own disjoint flag set (`--fusion-config`, `--run-dir`; NO `--netmhciipan-bin`,
  `--nmp-*`, or `--seed-table`) so NMP is never loaded, plus its own `JOB_TAG`/`OUT_DIR` wiring
  (the current `refine_${MODE}_0701` tag is refine-oriented). `#SBATCH --cpus-per-task=8` is a
  static, NMP-sized directive tied to `NMP_WORKERS` and cannot branch on `MODE` inline
  (`scripts/CLAUDE.md` §3); a fusion run either accepts the 8-core over-request or is submitted
  with a submit-time `sbatch --cpus-per-task=N` override.
- [ ] Register the driver and launcher mode in `doc/SCRIPTS.md`.
- [ ] Add the smoke YAML and print every resolved field at startup.
- [ ] Run the complete existing reference-flow and legacy-refiner test surfaces after the
  `runtime.py` override change.
- [ ] Verify the old sampler/context behavior is byte-equivalent when the override is omitted.
- [ ] Add one `LOG.md` entry only when the behavior-changing pipeline implementation lands;
  do not log experiment submission or result return.

**Gate F10:** legacy RF, D2/D3, SC-GR, and standalone NMP refiner behavior remains unchanged
unless the new Fusion driver/config is explicitly selected.

---

## 5. Test Matrix

Minimum local test surfaces before cluster smoke:

1. `tests/inverse_folding/test_reference_flow_fusion_state.py`
2. `tests/inverse_folding/test_reference_flow_fusion_objective.py`
3. `tests/inverse_folding/test_reference_flow_fusion_moves.py`
4. `tests/inverse_folding/test_reference_flow_fusion_selection.py`
5. `tests/inverse_folding/test_reference_flow_fusion_runner.py`
6. `tests/scripts/test_run_rf_refine_fusion.py`
7. existing sampler/runtime/controller/refiner regression tests touched by shared imports.

The following named behaviors must be covered:

- complete-state round-boundary invariant;
- exact selected-child carryover;
- Head-global acceptance vs local-only improvement;
- parent-relative off-target hotspot gate;
- definitive structure gate for every inherited child;
- anchor, edited-core, and outside-halo preservation;
- no legacy controller/D3/remask state after handoff;
- null move and all-failed-round behavior;
- elite monotonicity;
- offspring-count debiasing;
- deterministic FK resampling and resume;
- no NMP dependency;
- Fusion-disabled/unused legacy equivalence.

Required verification commands after F10:

- `pytest tests/inverse_folding/test_reference_flow_fusion_*.py tests/scripts/test_run_rf_refine_fusion.py -q`
- `pytest tests/inverse_folding/test_reference_flow_sampler.py tests/inverse_folding/test_reference_flow_sampler_constraints.py tests/inverse_folding/test_reference_flow_runtime.py -q`
- `pytest tests/inverse_folding/test_reference_flow_d1_controller.py tests/inverse_folding/test_reference_flow_d2_d3_controller.py tests/inverse_folding/test_refine.py tests/scripts/test_refine_rf_designs.py -q`
- `python -m py_compile scripts/run_rf_refine_fusion.py`
- `bash -n scripts/submit_refine.slurm`

---

## 6. Real Smoke And Deployment Gates

These are implementation acceptance checks, not the final scientific experiment sweep.

### Smoke S0: one protein, greedy + explicit edit

Use a small source population and fake/real cached structure where possible.

Pass only if:

- all inherited sequences are complete and structure-feasible;
- candidate sequence selected at round `r` is byte-identical to parent at `r+1`;
- Head global risk and window landscape are populated;
- null parent/elite behavior is visible in artifacts;
- NMP is not loaded.

Sourced command (inputs from refiner `PLAN_RF_REFINE.md` §5, a known-good DRB1*07:01 run;
`<…>` = values read from that run's `manifest.json`, a lookup, not a fabricated default):

```
python scripts/run_rf_refine_fusion.py \
  --fusion-config inverse_folding/reference_flow/configs/rf_refine_fusion_smoke.yaml \
  --run-dir Results/RF/Uricase/uricase_characterized23_a1res03_b1open_n16__20260630T041450Z/HLA-DRB1_07_01 \
  --proteins Q00511 --allele HLA-DRB1_07_01 \
  --head-checkpoint <manifest> --head-config-dir <manifest> --head-allele-idx <manifest> --head-device cuda \
  --pdb-root <uricase reference backbones, refiner §5> --refold-cache-dir <cache> --test-set-parquet <…> \
  --esmfold2-site-packages <Biohub env site-packages> --esmfold2-model biohub/ESMFold2 \
  --out-dir Results/RF/Uricase/fusion_smoke_s0_0701 --print-config
```

S1 (`moves.repair.enabled=true`) and S2 (selector replay) reuse these inputs with only the
config/flag change.

### Smoke S1: one protein, RF edit-repair

Pass only if:

- repair changes at least one allowed halo position in a non-degenerate case;
- protected edit, hard anchors, and outside-halo sequence remain identical;
- output is complete and Head/structure re-evaluated after repair;
- no sampler-level accepted-state remask occurs after the child enters ancestry.

### Smoke S2: small same-protein population, selector replay

Persist one evaluated candidate pool and replay it through greedy, beam, and FK.

Pass only if:

- candidate evaluations are identical across selector modes;
- FK descendant multiplicities differ from independent/beam controls when potentials differ;
- population size remains exactly `N`;
- elite output is identical or better than the initial elite in every mode;
- fixed seed reproduces lineage exactly.

### Pre-registered falsifier pilots (run only after S0-S2 pass)

These are pre-registered here, not deferred to a later ad-hoc design. Sizes (`N`, `n_rounds`,
per-arm refold budget) are fixed from S-smoke telemetry under the §1.8 walltime bound and stated
in each RAR **before** the run; no expected effect size is fabricated (`AGENTS.md` §2). All arms
are Head-only; NMP, if computed, is external post-hoc reporting only. Per-arm metrics: terminal
Head `global_risk` distribution, scTM pass-rate and tail, unique-sequence fraction (diversity),
refold count (cost). Cohort = the §6 S0 smoke inputs extended to the low-`global_risk` proteins of
that run.

**P1 — H1: Head-guided hard-state feedback lowers Head burden under the structure constraint**
[doc §11 H1; §13 "Unproven and load-bearing"].
- Inputs: one arm, greedy + explicit edit + edit-repair, Head-only.
- Acceptance (direction pre-registered; threshold fixed from S0 telemetry): median terminal-elite
  vs initial-population `global_risk` reduction > 0 at scTM ≥ the absolute gate, with improved-protein
  fraction ≥ the S0-fixed floor.
- Falsifier: the feasible pool contains lower-Head states but the loop fails to enrich/keep them,
  or local Head gains raise the whole-sequence landscape.
- Validation: `fusion_elite.parquet` initial-vs-final `global_risk`; `fusion_rounds.jsonl` elite risk per round.

**P2 — H3: RF repair improves the immune-structure frontier over explicit edit alone**
[doc §11 H3; §13 "Unproven and load-bearing"].
- Inputs: two arms differing ONLY in `moves.repair.enabled`, matched refold budget.
- Acceptance: `edit+repair` feasible-fraction (or scTM at matched `global_risk` drop) exceeds
  `explicit-edit-only` by a margin fixed from S1 telemetry noise.
- Falsifier: repair merely reverts the immune edit, adds no structure benefit, or is dominated by
  explicit edit + the state-level filter.
- Validation: compare `fusion_candidates.parquet` feasible/scTM at matched Head deltas across arms.

**P3 — H6: FK interaction adds value beyond independent/beam** [doc §11 H6; §13 "attributable by ablation"].
- Inputs: greedy/beam/FK arms over the same evaluated pool (S2 harness), then a full multi-round
  run per selector at matched refold budget.
- Acceptance: FK terminal Head-structure frontier ≥ beam at matched budget AND FK unique-fraction
  not below beam by more than the S2-fixed margin.
- Falsifier: compute-matched beam/independent matches the FK frontier, or FK collapses diversity
  without more feasible Head improvement.
- Validation: `fusion_elite` / `fusion_rounds` across selectors.

Do not bundle D2 or SC-GR into P1-P3: per doc §8.2 / §8.4 they are proposal-efficiency ablations
only after the hard-state core (P1) shows authority. H4 (exact feedback vs marginal actuation) and
H5 (targeted reopen vs global remask) are deferred to a later increment, since the v0 post-handoff
loop runs no global remask at all (§0.4).

---

## 7. Deferred Partial-Handoff Stage

This section is deliberately not part of the first coder delivery.

After the terminal-handoff Fusion passes the deployment gate, add a separate PLAN increment
for `t_star < T`:

- a real early-stop/checkpoint surface for the sampler;
- explicit visible-state materialization into independent completion children;
- no inheritance of old controller/RNG state after materialization;
- matched terminal-handoff and partial-handoff controls;
- evidence that earlier handoff improves the Head-structure frontier rather than merely
  increasing proposal compute.

Do not use the current `ResumeState` as an exact-clone claim. It may be reused as a visible
token-state continuation primitive only after RNG and lifecycle semantics are specified.

---

## 8. Coder Completion Checklist

- [ ] Scientific source and this PLAN agree on Head-only guidance and NMP exclusion.
- [ ] New hard-state loop lives outside `ReferenceFlowController`.
- [ ] Round-boundary state is always complete.
- [ ] Proposal identity survives evaluation and selection.
- [ ] Whole-sequence Head value, not local delta, drives selection.
- [ ] Static prior is absent from the new-hotspot gate.
- [ ] Every inherited child has a definitive target-backbone structure verdict.
- [ ] Parent null move and elite archive prevent destructive failure.
- [ ] RF repair preserves edit core, anchors, and outside halo.
- [ ] Greedy/beam/FK replay the same candidate pool.
- [ ] FK branch-size normalization and resampling semantics match §1.6.
- [ ] Driver artifacts permit exact ancestry and cost reconstruction.
- [ ] No external RERD/FK code or NMP runtime is imported.
- [ ] New script is registered and existing SLURM is extended, not duplicated.
- [ ] Existing RF/controller/refiner tests remain green.
- [ ] Real S0-S2 smokes pass before scientific sweeps begin.
