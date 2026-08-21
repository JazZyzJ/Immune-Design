# PLAN — Optional Dual-Allele Objective Layer for RF Fusion V2

> **Status:** READY FOR CODER after the scientific objective in
> `doc/Dual_Allele_Steering.md` passed the final method review. NOT READY FOR CLUSTER until one
> explicit, outcome-independent $\tau$ is frozen in the tracked objective-law spec; the user
> intentionally deferred numerical selection, so this PLAN does not invent it.
>
> **Execution contract:** implement task-by-task with RED -> GREEN evidence. This PLAN freezes
> behavior, ownership, interfaces, artifacts, compatibility, and qualification gates. It does not
> prescribe line-by-line code.

**Goal:** add simultaneous two-allele steering as an optional objective/evidence layer on the
qualified Fusion V2 recursive substrate. The same exact endpoint is scored by two frozen
allele-specific Heads; one frozen-normalized smooth worst-residual objective controls endpoint
ordering, donor adoption, incumbent advancement, and leave-one-out write evidence. The existing
source-compatible lineage, projection, schedule, structure/anchor feasibility, recursion, ledger,
and resume machinery remain unchanged.

**Implementation target:** `/Users/jerry/Project/MHC-IF-fusion`, branch `fusion_rf_refine`.
The code audit for this PLAN used commit
`cdcde74d0b893354426eed035da4b7593b291a3a`. Before coding, record the actual descendant commit and
stop if the audited seams below have changed materially.

**Scientific authority:** `doc/Dual_Allele_Steering.md`, especially §§2–4.

**Runtime authority:** the latest Fusion-worktree `doc/FUSION_V2.md`,
`doc/RF_Fusion_V2_Cluster_Runbook.md`, and the executed V2 code. The older
`PLAN_RF_REFINE_FUSION_V2.md` remains an interface/test history rather than live progress state.

**Empirical operating point:** RAR 0049, D4/K12/r40, four independent width-1 roots. Dual v0 does
not reopen D/K scaling or introduce an interacting population.

---

## 0. Delivery Boundary

### 0.1 What this implementation adds

1. Two frozen Heads score the **same exact endpoint** and bind their results to one endpoint ID,
   sequence MD5, protein, and compatible window grid.
2. A content-bound calibration maps each raw Head risk to a fixed coordinate

$$
u_a(y)=\frac{R_a(y)-b_a}{s_a},\qquad a\in\{A,B\}.
$$

3. The primary objective is the numerically stable smooth maximum

$$
J_\tau(y)
=
\tau\log\left[
\frac{
\exp\left(u_A(y)/\tau\right)
+
\exp\left(u_B(y)/\tau\right)
}{2}
\right].
$$

   The strict normalized maximum is implemented through the same interface as a frozen control.
4. One objective authority supplies the deterministic key for family representatives, archive
   elite, lineage incumbent, donor gate, and exact leave-one-out write contribution.
5. Both allele evidence views may nominate legal write/reopen positions, but both share the
   existing total schedule budget and reopen cardinality.
6. The existing per-allele catastrophic whole-landscape safety law is evaluated for both Heads;
   definitive structure is evaluated only once.
7. Dual provenance, cost, artifacts, analysis, preflight, and resume identities extend the current
   V2 surfaces without creating a second driver or launcher.

### 0.2 What this implementation does not add

- no second sampler, source state, replay graph, projection kernel, segment law, or recursive
  ladder;
- no online Pareto archive, interacting population, preference lane, FK/SMC, MCTS, or GFlowNet;
- no protected sequential optimization pass;
- no dynamic normalization from the current candidate cloud;
- no separate allele budgets, separate complete passes, or Head-A prefilter;
- no group/Shapley attribution, learned proposal, or multi-residue coalition rescue;
- no D2/D3/SC-GR/static-prior integration;
- no new structure method, refold threshold, anchor law, repair operator, or structure experiment;
- no NMP runtime, acceptance, tuning, or capability verdict;
- no K24/K32/D8 or temperature/weight sweep in the first capability campaign; and
- no new generation driver or SLURM file.

### 0.3 Frozen compatibility rule

The Dual layer is opt-in. With no Dual overlay:

- the second Head is not built or called;
- legacy `v2cfg-1` configs remain valid and retain their canonical digest;
- endpoint IDs, sequences, seeds, donor/write/reopen choices, archive elite, refold count, tables,
  and resume behavior remain byte- or value-equivalent to current single-Head V2; and
- existing single-Head fields keep their current meaning. In particular, Dual must never overwrite
  `head_global_risk` with a joint scalar or fabricate a mixed `HeadScore`.

---

## 1. Audited V2 Reuse Boundary

### 1.1 Reuse unchanged

| Existing surface | Dual treatment |
|---|---|
| `fusion_v2.state` live partial/source/replay/endpoint identities | Reuse; no new generative state machine |
| sampler capture, replay, propagation, and re-entry | Reuse unchanged |
| schedule bands and shared $\mathcal B(r)$ | Reuse unchanged |
| exact counterfactual sequence builder | Reuse for the same reverted sequences |
| source legality, history, anchors, write cap, reopen count | Reuse unchanged |
| projection, segment, ladder, seeds, cost journal, resume fragments | Reuse unchanged |
| definitive structure/refold/cache stack | Build and execute once per exact sequence |
| current complete-endpoint and archive evidence | Preserve; add Dual-bound sidecars/views |
| `run_rf_fusion_v2.py` and `submit_rf_fusion_v2_canary.slurm` | Extend optional mode; do not duplicate |

### 1.2 Single-Head seams that must become objective-aware

The coder must re-audit these exact seams before editing:

- `fusion_v2.state.CompleteEndpoint` carries one legacy Head score;
- `fusion_v2.reward.LineageIncumbent` and `donor_gate()` bind one raw risk;
- `fusion_v2_runtime.lookahead.bind_head_scores()` binds one Head result;
- `fusion_v2_runtime.archive.select_family_representatives()` and `ExactArchive` sort directly on
  `head_global_risk`;
- `fusion_v2.policy.HeadDirectedCappedPolicy` screens windows, scores LOO counterfactuals, and
  orders reopen candidates from one Head;
- `fusion_v2.safety` and `fusion_v2_runtime.admission` bind one cumulative Head ledger;
- `fusion_v2.config.V2HeadConfig`, production oracles, materializer, preflight, artifact writers,
  mechanism reader, and run signature bind one Head; and
- the existing top-k facade ranks `head_global_risk` and must not be reused as a fair Dual
  comparator without an explicit common-objective mode.

The correct change is an injected objective/evidence authority across these seams, not a parallel
Dual policy that acts after a single-Head donor has already been chosen.

### 1.3 State economy

Keep `CompleteEndpoint` and its legacy content identity stable. Add an endpoint-ID-bound immutable
Dual evidence object or equivalent sidecar with:

| Field family | Required content |
|---|---|
| endpoint binding | `endpoint_id`, `protein_id`, `sequence_md5`, length, window-grid binding |
| Head A/B | allele role, raw risk, evaluator/config/checkpoint digest, full score binding |
| objective | $b_A,s_A,b_B,s_B$, $u_A,u_B$, mode, $\tau$ when applicable, $J$, objective digest |
| safety | immutable reference identity and per-allele cumulative hotspot verdict/digest |
| reproducibility | calibration digest, objective version, finite-value validation |

No Dual selector may consume an endpoint until this binding is complete. Pareto membership,
hypervolume, worst-axis labels, and terminal protection are derived views rather than new online
state.

---

## 2. Frozen Scientific and Runtime Contracts

### 2.1 Objective contract

- Compute smooth max with `logaddexp` or subtract-max stabilization; direct unbounded exponentials
  are forbidden.
- $b_a$ and $s_a>0$ are fixed by a content-bound calibration population outside the live endpoint
  cloud. Runtime batches never update them.
- The first calibration uses a frozen guidance-off NoD population. Deduplicate by
  `(protein_id, sequence_md5)`, give each protein total weight $1/P$ and each of its unique
  sequences equal within-protein weight, set $b_a$ to the weighted median, and set

$$
s_a=\frac{Q_{0.75,a}-Q_{0.25,a}}{1.349}.
$$

  Fail rather than invent a floor when the weighted IQR is non-finite, non-positive, or no larger
  than the allele's frozen matched-difference noise bound. Because the initial NoD cohort was
  selected for allele A, this is **target-cohort normalization**, not a claim of allele-neutral
  population calibration.
- The primary deterministic rank key is

$$
\left(J_\tau(y),\operatorname{endpoint\_id}(y)\right).
$$

  Thus the scalar provides a total preorder and endpoint identity closes exact ties.
- The same objective version/digest is used by family selection, archive elite, incumbent update,
  donor gate, and LOO attribution. Mixing objective versions inside one lineage is invalid.
- Hard max uses the same evidence and ordering surfaces with

$$
J_{\max}(y)=\max(u_A(y),u_B(y)).
$$

  It is a control implementation, not a second runtime architecture.
- $\tau$ is a required literal in the tracked objective-law spec. The calibration producer signs
  but does not estimate or tune it. Missing $\tau$ blocks launch; this PLAN deliberately does not
  fabricate a numerical value.

### 2.2 Donor and incumbent contract

For a source-compatible lineage, a donor passes only when

$$
J(y_{\mathrm{inc}})-J(y_{\mathrm{donor}})>\epsilon_{\mathrm{donor}}.
$$

- The donor and incumbent must carry the same two Head identities, normalization, objective, and
  window-grid contracts.
- Incumbent advancement remains immutable and source-compatible.
- A donor selected under raw A, raw B, or a different objective cannot be advanced as a joint
  donor.
- Every family representative and archive elite used by a Dual arm must be ordered by the same
  objective resolver; no hidden `head_global_risk` sort is allowed.

### 2.3 Write contract

For each legal donor position, construct the existing exact sequence with that donor token reverted
to the current lineage incumbent token. Score the same reverted sequence with both Heads and define

$$
a_i
=
J\left(y^{(-i)}\right)-J(y).
$$

The identity is write-eligible only when $a_i>\epsilon_{\mathrm{write}}$ and all existing source,
history, anchor, masked-position, schedule, and cap laws pass.

The initial position screen is the union of allele views: a position covered by a qualifying
improved window for A **or** B may reach the exact joint LOO batch. No A-only prefilter is allowed.
The counterfactual sequence set is deduplicated once and sent to both Heads; each logical position
records both raw contrasts and the joint contribution.

### 2.4 Reopen contract

Reopen timing, resolved-domain legality, schedule band, required cardinality, and shared total
budget remain current V2 behavior. Dual changes only the Head evidence attached to each legal
position.

The minimal symmetric reducer preserves the current lexicographic categories:

```text
new hotspot > worsening > residual burden > active uncertainty > temporal instability > index
```

For each Head-derived category, a position is present when either allele supplies that evidence;
within the category, compare the maximum per-allele scale-normalized magnitude, with deterministic
position ties. Keep both allele values and the winning allele label in evidence. This is one union
ordering and one reopen count—not two ranked lists with two budgets.

If this reducer cannot be expressed using the current Head/window evidence without inventing a
masked-state score, stop DUALF4 and repair the scientific contract rather than falling back to
Head A.

### 2.5 Feasibility and safety contract

Structure, anchors, function, and source legality are inherited unchanged and evaluated once. The
Dual immune safety verdict is conjunctive:

$$
I_{\mathrm{adm}}^{AB}(y\mid\bar y)
=
\mathbf 1\left[
F_{\mathrm{V2}}(y)=1,
N_A^{\mathrm{whole}}(y;\bar y)\leq\delta_A,
N_B^{\mathrm{whole}}(y;\bar y)\leq\delta_B
\right].
$$

- Each allele owns its frozen cumulative reference score, calibration, and verdict.
- Both references describe the same immutable reference sequence and aligned window domain.
- Lineage advance updates both immediate-parent comparison bindings together.
- All-failed safety leaves the lineage at its valid incumbent; it never falls back to an unsafe
  endpoint.
- Strict improvement on both raw risks is not required; catastrophic failure on either axis is
  forbidden.

### 2.6 Output contract

The existing exact archive remains the persistent lifecycle. Add derived outputs:

- **balanced capability elite:** admissible endpoint minimizing frozen $J$;
- **dual non-dominated view:** reconstructed offline from all definitive endpoint evidence, not
  assumed equal to archive membership; and
- **deployable protected elite:** lowest-$J$ endpoint satisfying the predeclared per-allele terminal
  protection bounds, or an explicit no-design outcome, **when** a separate deployment analysis
  contract is supplied.

The balanced elite is the primary method output. The protected view is optional deployment/reporting
analysis; it does not affect generation, ancestry, or capability GO and cannot silently replace or
relabel the balanced result.

---

## 3. Config, Identity, Cost, and Artifact Contracts

### 3.1 Optional overlay

Do not change the canonical digest of legacy configs merely by adding `dual: null`. Introduce a
strict optional Dual overlay, supplied explicitly to the existing materializer/driver. When absent,
the legacy path does not import or instantiate Dual runtime objects.

The overlay has no scientific defaults and must bind:

| Section | Required fields |
|---|---|
| identity | schema/version, enabled flag, allele roles A/B |
| objective | `smooth_max` or `hard_max`; explicit $\tau$ for smooth max; objective version |
| calibration | content-bound artifact with $b_a,s_a$, repeatability, donor/write margins, and digests |
| Head B | allele, checkpoint/config/variant/window-grid/score-scale identity; runtime path supplied by CLI |
| safety | two cumulative calibration/reference bindings and limits |
| terminal (optional) | common deployment reference and explicit per-allele protection margins; analysis-only |
| qualification | frozen matched objective-arm set when qualification mode is selected |

Head A continues to use the existing production Head binding. Head B paths are passed through CLI
or launcher variables; cluster paths are never module constants. The loaded artifacts must match
the overlay's content identities.

### 3.2 Runtime identity and resume

The run signature must change when any of the following changes:

- either Head checkpoint/config/allele/window-grid/score-scale digest;
- either normalization location/scale or calibration population digest;
- objective mode, version, or $\tau$;
- donor/write margins;
- either cumulative safety calibration/reference/limit;
- objective-arm bundle in a matched comparison.

An optional terminal-protection analysis has its own content signature. Changing it invalidates
that analysis output but not a completed generation run whose ancestry never consulted it.

A resume mismatch fails before oracle work. Existing fragments created by a different objective
cannot supply donors, writes, archives, or elites to the current run.

### 3.3 Oracle ownership and cost

- Build Head A plus the definitive structure stack once through the existing production oracle
  builder.
- Build Head B as a separate frozen Head scorer without constructing a second structure/refold
  stack.
- Score unique same-protein exact sequences in batches for both Heads.
- The ledger charges Head work per `(allele, sequence)` and LOO work per
  `(allele, counterfactual sequence)`; shared sequence construction and structure work are not
  double-counted.
- Preflight reports shared DFE/refold cost and per-arm/per-Head logical calls. A breached cap fails
  before launch.

The second Head adds little compared with definitive refolding, but the code must measure rather
than assume that cost. Dual does not authorize extra refolds.

### 3.4 Artifacts

Preserve all current V2 tables and meanings. Add at least:

| Artifact | Grain | Required content |
|---|---|---|
| `dual_endpoint_evidence.parquet` | exact endpoint × objective arm | endpoint/source IDs, sequence MD5, A/B raw risks and score digests, $u_A,u_B,J$, objective/calibration digest, both safety verdicts |
| `dual_feedback_evidence.parquet` | feedback event × tested position | arm, incumbent/donor IDs and $J$, donor margin, reverted sequence MD5, A/B counterfactual risks/deltas, $\Delta J$, eligibility, rank, selected write, allele-tagged reopen evidence |
| `dual_terminal_summary.parquet` | protein/root × arm | shared D0 identity, balanced elite, common-$J$ frontier, raw A/B change, typed stop, cost; optional protected-elite fields only under a signed deployment analysis |

The manifest records `dual_enabled`, both Head identities, calibration/objective digest, matched-arm
bundle, shared-root identity, artifact paths, and `nmp_runtime_absent=true`.

Existing `complete_endpoints.parquet` may receive nullable Dual join keys only if its legacy schema
and consumers remain compatible. Do not store full per-position evidence as stringified Python
objects when a structured sidecar is available.

---

## 4. Code Ownership and Reuse Matrix

| Surface | Planned action |
|---|---|
| `fusion_v2/joint_objective.py` or equivalent pure module | Create typed Dual calibration/evidence/objective resolver; justified as an isolated, torch-free scientific law |
| `fusion_v2/config.py` | Add strict Dual overlay loader/validation without changing legacy config semantics |
| `inverse_folding/reference_flow/configs/v2_dual_smoothmax_policy_v1.json` | Frozen objective-law identity only; create after $\tau$ is scientifically frozen, never with a placeholder; runtime paths and measured calibration stay external |
| `fusion_v2/reward.py` | Parameterize incumbent/donor comparison by objective authority |
| `fusion_v2/policy.py` / `evidence.py` | Reuse counterfactual construction; add joint LOO and symmetric union evidence |
| `fusion_v2/safety.py` | Compose two Head safety ledgers around one inherited structure contract |
| `fusion_v2_runtime/lookahead.py` | Bind paired Head evidence before Dual selection |
| `fusion_v2_runtime/archive.py` | Inject deterministic objective rank key; preserve exact archive lifecycle |
| `fusion_v2_runtime/admission.py`, `cycle.py`, `ladder.py` | Carry Dual evidence/authority through existing cycle; no duplicate runner |
| `scripts/rf_fusion_v2_oracles.py` | Build second Head only; reuse first Head + one structure stack |
| `scripts/calibrate_v2_head_policy.py` | Extend to produce Dual normalization/repeatability/safety artifact |
| `scripts/materialize_v2_canary_config.py` | Resolve/sign optional Dual overlay and matched arm bundle |
| `scripts/preflight_v2_canary_assembly.py` | Fully instantiate Dual policy/oracles model-free where possible; no one-prefix workaround |
| `scripts/run_rf_fusion_v2.py` | Add optional Dual and matched-arm qualification/capability dispatch |
| `scripts/rf_fusion_v2_{artifacts,resume,cohort}.py` | Extend schemas, resume identity, shared-root arm branching, and aggregation |
| `scripts/analysis/read_v2_mechanism.py` | Add protein/prefix-equal Dual qualification and recursive common-$J$ views |
| `scripts/materialize_v2_archive_facade.py` | Add explicit common-objective mode; retain legacy raw-Head mode |
| `scripts/submit_rf_fusion_v2_canary.slurm` | Parameterize existing launcher; no new SLURM |
| `doc/SCRIPTS.md` | Register changed flags/modes and remove stale V2 descriptions when implementation lands |

---

## 5. Build Tasks

### DUALF0 — Authority reconciliation and seam audit

**Objective:** prove the coder is modifying the executed Fusion V2 substrate rather than an older
main-tree copy.

**Inputs/assumptions:** scientific doc, this PLAN, latest worktree V2 science/runbook, RAR 0049,
current worktree commit.

**Deliverables:**

- an implementation-base note in the PR/commit description with the audited commit and changed
  seams;
- a path-level mapping from every direct `head_global_risk` decision to legacy-only or
  objective-aware behavior; and
- confirmation that D4/K12/r40 and active population width 1 remain frozen.

**Stop:** authority files disagree on objective, safety, source compatibility, or operating point.

**Acceptance:** no scientific choice is resolved by an undocumented code default.

### DUALF1 — Pure objective, calibration, and evidence identity

**Objective:** create the torch-free Dual authority used everywhere else.

**RED tests:**

- missing/swapped allele role, invalid $s_a$, non-finite input, same Head bound twice;
- mismatched endpoint/sequence/window-grid identities;
- current-cloud normalization attempt;
- direct exponential overflow;
- Pareto-dominated endpoint outranking its dominator;
- extreme favorable non-worst allele causing unbounded compensation;
- row-order-dependent exact ties; and
- changed Head/calibration/$\tau$/epsilon failing resume identity.

**Interfaces/artifacts:** strict Dual overlay, calibration record, `DualEndpointEvidence`, stable
objective resolver and `(value, endpoint_id)` rank key.

**Acceptance:** smooth max satisfies its bounded-max and monotonicity properties; hard max shares
the interface; canonical serialization/hashing is deterministic; package import remains model-free.

### DUALF2 — Paired Head scoring and endpoint binding

**Objective:** attach two frozen Head results to every exact endpoint before any Dual decision.

**RED tests:**

- only one Head result returned;
- A/B sequence MD5, length, protein, or grid mismatch;
- Head B incorrectly rebuilds or calls structure;
- duplicate sequences charged twice within one scoring stage;
- one allele silently replaced by joint scalar in legacy `HeadScore`; and
- partial paired batch entering selection.

**Interfaces/artifacts:** composite paired scorer, one structure owner, per-allele ledger events,
endpoint evidence sidecar.

**Acceptance:** both Heads score the exact same deduplicated endpoint set; structure/refold count is
identical to single-Head execution; legacy single mode does not build Head B.

### DUALF3 — One objective authority for endpoint selection and lineage

**Objective:** make family selection, archive elite, incumbent update, and donor gate agree on the
same frozen objective.

**RED tests:**

- family representative chosen by A while donor gate uses $J$;
- archive elite still follows `head_global_risk`;
- incumbent advances under a different objective digest;
- exact ties change with input row order;
- a lower-$J$ infeasible endpoint enters ancestry; and
- Dual disabled changes any existing single-Head fixture.

**Interfaces/artifacts:** injected objective resolver across archive/cycle/reward, objective-bound
incumbent and donor evidence.

**Acceptance:** replaying one endpoint pool through A-only, B-only, smooth-max, and hard-max changes
only the declared ordering law; every Dual surface returns the same best endpoint under the same
law.

### DUALF4 — Joint LOO writes, union reopen, and dual safety

**Objective:** turn the joint donor decision into directionally consistent feedback without
changing the V2 intervention budget.

**RED tests:**

- B-only useful position removed by an A prefilter;
- donor selected by $J$ but write contribution computed from raw A;
- counterfactual reverted to source/WT rather than current incumbent;
- duplicate A/B counterfactual sequence construction;
- independent per-allele write or reopen budgets;
- union position counted twice;
- allele-swap changes a symmetric decision except declared tie labels;
- one allele catastrophic-safety failure admitted;
- structure evaluated twice; and
- no positive joint write falling back to a single-Head identity.

**Interfaces/artifacts:** paired LOO scorer, per-position Dual evidence, symmetric union reducer,
two safety ledgers plus one structure verdict, existing `PolicyDecision` projection contract.

**Acceptance:** selected writes have $\Delta J>\epsilon_{\mathrm{write}}`; selected reopen count and
$\mathcal B(r)$ match the frozen single-Head geometry; every admitted donor is safe for both Heads;
typed stalls replace all unauthorized fallbacks.

### DUALF5 — Driver, materializer, artifacts, resume, and preflight

**Objective:** make Dual an auditable optional mode of the existing production pipeline.

**RED tests:**

- Dual config present but Head B absent;
- path/label mismatch between DRB1*07:01 and DRB1*04:01;
- config or calibration hand-edited after materialization;
- resume after Head/objective/safety drift;
- shared D0 branches with non-identical endpoint digests;
- cost ledger charging one Head for two evaluations or duplicating shared refolds;
- facade/reader silently ranking the Dual run by legacy A risk; and
- assembly preflight passing while the real Dual policy cannot be constructed.

**Interfaces/artifacts:** optional driver/materializer flags, existing launcher mode, Dual tables,
manifest/run signature, common-$J$ analysis reader, full config printout. Each generated
`.args.sh` must export the resolved Dual mode and protein ID in addition to the existing config,
cohort, input, and shard vectors so launcher/preflight never infer them from a filename.

**Acceptance:** one model-free resolved cell proves both Heads, objective, safety, budget, arm bundle,
artifacts, and resume identity are complete. A real one-prefix smoke is not a substitute for a
missing assembly contract.

### DUALF6 — Compatibility and qualification handoff

**Objective:** close local implementation and hand only the three scientific experiments in the
Dual runbook to the cluster agent.

**Required local evidence:**

1. complete new pure/unit/integration tests;
2. existing Fusion V2, sampler, policy, safety, archive, artifact, resume, preflight, driver, and
   analysis regressions;
3. Dual-off golden equivalence on seeds, sequence bytes, decisions, artifacts, and costs;
4. fake-oracle end-to-end Dual run with A-only/B-only/joint views sharing one exact D0;
5. observer-only second Head produces no generative change; and
6. no NMP import or runtime field.

**Acceptance:** code can answer whether Head B changes donor/write identity and fresh descendants;
it does not claim Dual capability before the cluster runbook passes.

---

## 6. Adversarial Test Matrix

Minimum named behaviors:

- stable smooth max and hard-max limit;
- fixed normalization and allele-swap symmetry;
- exact endpoint/sequence/grid binding for both Heads;
- objective-consistent family/archive/incumbent/donor ordering;
- same-incumbent exact LOO and B-only useful write admission;
- one union action budget and deterministic reopen order;
- conjunctive per-allele catastrophic safety with one structure evaluation;
- null/incumbent preservation on all-failed events;
- Dual-off single-Head equivalence;
- objective-bound resume refusal;
- shared-D0 exact identity across matched arms;
- deterministic fork/descendant seeds across arm labels;
- per-Head logical cost and shared physical refold accounting;
- common-$J$ reconstruction from all definitive endpoints; and
- protein/prefix equal weighting in readers.

Hard-max replay uses objective-specific donor, write, and analysis margins. A smooth-max margin is
not reused merely because both laws consume the same raw Head observations.

The coder should add focused files under the existing Fusion V2 test families rather than create a
parallel test hierarchy. Run the repository's current V2 regression commands from the latest
worktree runbook/CI; do not copy stale command lists from the historical PLAN.

---

## 7. Real-Experiment Boundary and Cost Model

Code completion authorizes only the companion runbook:

1. one offline calibration plus exact dry-run/assembly preflight;
2. one matched one-cycle Dual directionality experiment; and
3. one matched 100-protein D4/K12 recursive capability comparison.

The recursive comparison must branch `joint`, `A_only_with_joint_safety`, and
`B_only_with_joint_safety` from one byte-identical D0/root checkpoint inside each protein cell. The
control labels are explicit because the non-optimized Head still enforces the common catastrophic
safety contract; it is not merely a passive observer. Three
independent campaigns that merely reuse the same numeric seed are invalid because campaign and
split identity participate in V2 seed derivation. After the shared checkpoint, matched stochastic
draws derive from one comparison identity and fork index rather than the objective-arm label; the
arm remains present in lineage/artifact identity but cannot silently open an independent RNG stream.

At one root per 100 proteins, three recursive branches are approximately 300 single-arm
cell-equivalents, below the executed 400-cell RAR 0049 D4/K12 campaign. This is a feasibility bound,
not a walltime prediction. The materialized preflight must compute exact DFE, Head, refold, and
walltime caps from the current code and fail closed before submission.

No further smoke, K/D sweep, strict-max generation arm, protected sequential run, NMP evaluation,
or structure experiment is part of the initial handoff. Hard max is available for offline replay
and a conditional follow-up only if the primary joint law is ambiguous.

---

## 8. Coder Completion Checklist

- [ ] Implementation is based on the audited Fusion worktree descendant.
- [ ] Dual is an optional objective/evidence layer, not a second V2 state machine.
- [ ] Two Heads bind to the same exact endpoint and compatible window grid.
- [ ] Fixed normalization and stable smooth max are content-bound.
- [ ] Family, archive, incumbent, donor, and LOO share one objective digest.
- [ ] No Head-A prefilter removes Head-B evidence.
- [ ] Write/reopen share the existing total budget and source legality.
- [ ] Both allele hotspot gates pass; structure is evaluated once.
- [ ] Existing endpoint and legacy Head meanings are preserved.
- [ ] Artifacts and resume bind both Heads, calibration, objective, safety, and arm bundle.
- [ ] Matched-arm comparison shares one byte-identical D0/root state.
- [ ] Dual-off behavior is regression-equivalent.
- [ ] Existing driver and SLURM are extended; no duplicate launcher exists.
- [ ] `doc/SCRIPTS.md` reflects the implemented modes and removes stale descriptions.
- [ ] Local gates pass before any cluster submission.
- [ ] `LOG.md` is updated only when the behavior-changing implementation lands.
