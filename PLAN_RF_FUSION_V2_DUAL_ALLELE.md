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
- The calibration population is the **fixed allele-neutral natural panel** of
  `doc/Dual_Allele_Steering.md` §2.2.1, one natural or reference sequence per protein. A generated
  population — NoD included — may not be used: it makes $b_a,s_a$ a function of the sampler
  configuration and normalizes by the distribution the method later claims to have moved away from.
  NoD remains the matched guidance-off experimental comparator only.
  *Panel identity is an open decision; see §5 DUALF1a. The implementation must load it as a
  content-bound artifact and must not embed any candidate path as a module constant.*
- Because the panel holds exactly one sequence per protein, protein-equal weighting and
  `(protein_id, sequence_md5)` deduplication are **not** part of the law. Both frozen Heads score
  the identical panel in one pass and

$$
b_a=\operatorname{median}\left(R_a\right),
\qquad
s_a=\frac{Q_{0.75,a}-Q_{0.25,a}}{1.349}.
$$

  Fail rather than invent a floor when the IQR is non-finite, non-positive, or no larger than that
  allele's frozen matched-difference noise bound.
- The same pass produces a second calibration pair $\left(b_a^{\rho},s_a^{\rho}\right)$ for the
  length-normalized positive hotspot mass $\rho$, used **only** by the return-boundary front
  (`doc/Dual_Allele_Steering.md` §3.2.1). $\rho$ never enters $J_{\mathrm{core}}$, and $\rho_a$ is
  computed from the per-residue hotspot vector already carried by the Head score, so it costs no
  additional Head call.
- The calibration **reports** the realized $\operatorname{SE}\left(b_A/s_A-b_B/s_B\right)$ by bootstrap
  over panel proteins, next to $\tau$, so a reader can see how much of the credit band
  $c_u=\tau\log 2$ the calibration's own imprecision occupies. It is a reported diagnostic, not a
  gate: gating would add a tuning knob to a calibration whose point is that **exactly one number,
  $\tau$, is chosen by a human** and everything else is measured in the panel pass or derived from
  the objective's geometry.
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
- $\tau$ is a required literal in the tracked objective-law spec, declared in **normalized units**
  through the credit $c_u=\tau\log 2$. A raw-scale declaration is rejected by the loader: $s_A\neq
  s_B$ makes one normalized credit correspond to two different raw credits, so a raw-unit $\tau$
  would reintroduce the scale asymmetry the normalization exists to remove. The calibration producer
  signs but does not estimate or tune $\tau$. Missing $\tau$ blocks launch; this PLAN deliberately
  does not fabricate a numerical value.
- The joint margins are **derived, not declared**. $\nabla_u J_\tau$ is the softmax weight vector,
  so $J_\tau$ is $1$-Lipschitz in the supremum norm on $u$ and each joint margin is

$$
\epsilon^{\mathrm{joint}}=\max_a\frac{\epsilon_a^{\mathrm{raw}}}{s_a},
$$

  propagated from that allele's own measured raw-scale floor. This covers
  $\epsilon_{\mathrm{donor}}$ and $\epsilon_{\mathrm{write}}$;
  each records the per-allele raw floor it came from. The hard-max control derives its own margins
  the same way and may not inherit the smooth-max values.

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

Structure, anchors, function, and source legality are inherited unchanged and evaluated once.

**The admission law is inherited unchanged; role B is measured, not gated.** The conjunctive
verdict $I_{\mathrm{adm}}^{AB}$ is not implemented as a second gate in this program. The single-Head
cumulative gate remains the only admission law, and role B's whole-landscape drift
$N_B^{\mathrm{whole}}(y;\bar y)$ is computed with the same pure primitive and recorded as telemetry
on the endpoint's Dual evidence.

Three reasons, in order of weight:

1. **A second gate would have nothing to bind against on the substrate that matters.** The frozen
   high-risk cohort carries `delta_new_cumulative = 1000000.0` with the incremental gate off, so the
   allele-A ceiling is already a no-op there and $I_{\mathrm{adm}}^{AB}$ reduces to the structure
   verdict regardless of what role B does.
2. **No $\delta_B$ exists and none can be produced under the calibration stage's own constraints.**
   The per-protein ceiling comes from `scripts/calibrate_rf_fusion_v2_hotspot.py`, which *generates*
   a feedback-disabled completion population and runs the structure gate. That is a GPU stage the
   calibration stage explicitly excludes and does not budget.
3. **The cost is not code volume, it is identity.** `SafetyReferenceBinding` is a field of
   `LivePartialState` and `ProjectedPartialState`, inside both canonical payloads. A second binding
   moves every state digest, hence every `state_id`, hence every `endpoint_id` -- breaking both the
   §0.3 Dual-off equivalence guarantee and the byte-identical shared-root requirement the recursive
   comparison rests on.

What remains true and is implemented:

- structure, anchors and source legality are evaluated exactly once, unchanged;
- the inherited single-Head cumulative gate still decides admission, and an all-failed event still
  leaves the lineage at its valid incumbent rather than falling back to an unsafe endpoint;
- strict improvement on both raw risks is never required; and
- role B's `max_increase`, `positive_mass` and `positive_count` against the frozen depth-0 reference
  are reported per endpoint.

**Claim boundary.** A result from this program may not state that a conjunctive per-allele safety
gate was enforced. It may state that role-B hotspot drift was measured and report the distribution.
Promoting the telemetry to a gate is a conditional enhancement, activated by observed drift in that
distribution -- not by the symmetry of the formula.

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

### DUALF1a — Calibration panel selection and the frozen coordinate artifact

**Objective:** produce the one content-bound artifact that freezes $b_a,s_a$ for both $R$ and
$\rho$, and prove the panel is admissible.

**Inputs/assumptions:** the two frozen Head checkpoints; a candidate natural panel; the frozen
per-allele raw repeatability floors; the declared $c_u=\tau\log 2$.

**Panel admissibility — every item is a hard gate, not a preference:**

- one natural or reference sequence per protein, canonical AA20, length $\geq$ the maximum Head
  window $k$;
- selected by structure or by nothing at all: never by risk under either allele, never by an
  external immune predictor, never by Fusion or NoD generation;
- **training overlap measured per allele and its effect bounded, rather than required to be zero.**
  Existence of overlap is not the hazard; ASYMMETRIC overlap is, because only the NORMALIZED
  location difference is load-bearing and a shift common in $u$ largely cancels. Both $b_a$ and
  $s_a$ are robust statistics, so a contaminated fraction $f$ moves the median by at most the
  $50\rightarrow(50\pm100f)$ percentile displacement. The producer therefore reports $f_A$, $f_B$,
  and an **overlap-inclusion sensitivity** — a second calibration whose panel build is identical
  except that step 3 keeps what it measured — reporting

$$
\Delta\theta
=
\left(\frac{b_A}{s_A}-\frac{b_B}{s_B}\right)_{\mathrm{overlap\ included}}
-
\left(\frac{b_A}{s_A}-\frac{b_B}{s_B}\right)_{\mathrm{primary}}
\qquad\text{and}\qquad
\Delta\log\frac{s_A}{s_B}
$$

  and stating whether $\lvert\Delta\theta\rvert$ exceeds $c_u$ as a **panel-sensitivity label**
  (AUDIT §J.7 D2). It is deliberately NOT a gate that selects between the two panels: the primary
  panel was chosen by an outcome-independent leakage rule and remains the coordinate authority
  whatever the sensitivity measures, so a large $\lvert\Delta\theta\rvert$ limits the claim to
  this frozen reference panel rather than promoting the contaminated one. An integrity or
  provenance failure of the sensitivity artifact does block launch. The quantity is $b_A/s_A-b_B/s_B$ and NOT $b_A-b_B$: the
  active-worst boundary is $R_A/s_A-R_B/s_B=b_A/s_A-b_B/s_B$, so adding one raw constant to both
  locations leaves $b_A-b_B$ exactly unchanged while moving the boundary by
  $\delta\left(1/s_A-1/s_B\right)$ (see `doc/Dual_Allele_Steering.md` §2.2.3). It is already
  dimensionless, so it is compared against $c_u$ directly rather than divided by an $s$ that is not
  defined for a pair. The scale-ratio term is the boundary's SLOPE: a recomputation may agree on the
  intercept and still disagree about which allele is worst over a whole region. Overlap detection is by homology, not exact identifier match; the
  existing `head_train_overlap_flag` machinery is single-allele and must be extended;
- length coverage spanning the deployment domain;
- no single structural family dominating the panel; and
- sized so that $\operatorname{SE}\left(b_A/s_A-b_B/s_B\right)$ is measured by bootstrap over panel
  proteins and REPORTED, in the same coordinate as the sensitivity above and for the same reason.

  **RESOLVED (AUDIT §J.7 D3, 2026-08-25): the qualitative launch requirement
  $\operatorname{SE}\ll c_u$ is retired.** It was never a fail-closed gate — Appendix F had already
  withdrawn that form, because a threshold on the calibration's own precision is a second
  human-chosen number in a calibration whose whole point is that exactly one ($\tau$) is chosen —
  and as a qualitative requirement it had no decidable content either. **Measured 2026-08-24:
  0.0185 against $c_u=0.10$, i.e. 18.5 % of the credit band**, ACCEPTED for v1. The panel is a
  deliberately frozen reference coordinate system, so its median/IQR values are exact descriptive
  constants OF THAT PANEL and this SE measures sensitivity to which homologous families instantiate
  the broader natural-sequence domain — not runtime measurement noise and not donor/write
  uncertainty. The SE and its bootstrap method stay in the calibration report and the method claim
  is scoped to the content-bound panel. It also cannot be reduced by enlarging the panel (13,836
  homology-independent units is the ceiling the deduplicated Tier-2 pool supports). Reintroducing
  evaluation proteins, loosening clustering to manufacture nominal sample size, or changing $c_u$
  after observing this number are all forbidden.

**RED tests:** panel entry shorter than the maximum window $k$; a non-canonical residue; two
sequences for one protein; an overlap-inclusion sensitivity built on a panel that differs from the
primary in more than step 3, or measured under a different Head identity or objective law, or
reported without its $\lvert\Delta\theta\rvert$ label; a sensitivity report used to SELECT the
contaminated panel as the coordinate authority; a calibration whose stability is asserted on the
RAW difference $b_A-b_B$; a panel whose bootstrap
$\operatorname{SE}\left(b_A/s_A-b_B/s_B\right)$ going unreported; $s_a$ not exceeding that
allele's raw noise floor; an attempt to compute $b_a,s_a$ from anything other than the signed panel;
an attempt to recompute them from a runtime batch.

**Interfaces/artifacts:** one signed calibration JSON binding panel identity and digest, both Head
identities, $\left(b_a^{R},s_a^{R}\right)$ and $\left(b_a^{\rho},s_a^{\rho}\right)$, the per-allele
raw floors, the derived joint margins, the realized bootstrap standard error, and $\tau$/$c_u$ as
validated inputs rather than fitted outputs. Panel paths arrive by CLI; no module constant.

**Stop:** the panel fails any admissibility gate, or per-allele overlap cannot be measured at all.
An unmeasurable panel is not usable; a panel with measured, symmetric, effect-bounded overlap is.

**Acceptance:** re-running the producer on the same panel and checkpoints reproduces the artifact
digest exactly; the artifact is the only source of $b_a,s_a$ anywhere in the codebase.

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
- runs of two arms whose depth-0 pools differ, which means a non-arm argument moved;
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
- depth-0 exact identity across separately launched arms;
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

1. one offline calibration plus exact dry-run/assembly preflight; and
2. one 100-protein D4/K12 recursive capability comparison, run as **one launch per arm**.

**The arms are compared across runs, not inside one.** The Dual arm enters no seed derivation and
the depth-zero root capture never sees the Dual runtime, so two launches that differ only in
`--dual-arm` -- same config, campaign, split, master seed, cohort and inputs -- produce the same
root, the same lookahead pool and the same role A raw risks, and diverge only at selection. The
comparison is therefore paired by construction; this is asserted in
`tests/inverse_folding/test_fusion_v2_dual_arms.py` and verified on the artifacts by comparing each
protein's depth-0 `endpoint_id` / `sequence_md5` / `source_state_id` across the runs.

This PLAN deliberately adds **no in-run matched-arm engine** for the alleles. One would pay for the
shared source prefix once instead of twice; it would change no scientific property of the
comparison, and the existing within-run matched-pair engine (`fusion_v2_runtime/paired.py`) answers
a different question -- two INTERVENTIONS under one policy -- so the driver refuses `--dual-overlay`
together with the mechanism/qualification path rather than producing a same-objective contrast that
reads like an allele one.

The realized $\left(m_{\mathrm{write}},m_{\mathrm{reopen}}\right)$ are reported per arm as
OUTCOMES and are never a validity precondition: both are functions of the objective, so requiring
parity would discard exactly the cells in which the second Head acted
(`doc/Dual_Allele_Steering.md` §5.3.1). The descendant read is stratified by realized dose.

**Arm labels name the objective and nothing else.** The frozen arm vocabulary is exactly
`joint` / `a_only` / `b_only`, lowercase, identical in both experiments, and it is a load-bearing
literal: it appears in the matched-arm bundle, the run signature, the resume fragment key, and every
Dual artifact. The safety contract in force is a separately declared property of the run, recorded
next to the arm rather than inside its name. The earlier `*_with_joint_safety` labels are withdrawn:
on the frozen high-risk substrate the cumulative hotspot ceiling is set to an effectively disabling
value with the incremental gate off, so those labels described a gate that does not exist there
(`doc/Dual_Allele_Steering.md` §4.2.1). A capability comparison on that substrate reports that its
arms differ in objective only.

Three independent campaigns that merely reuse the same numeric seed are invalid because campaign and
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

- [x] Implementation is based on the audited Fusion worktree descendant.
- [x] Dual is an optional objective/evidence layer, not a second V2 state machine.
- [x] Two Heads bind to the same exact endpoint and compatible window grid.
- [x] Fixed normalization and stable smooth max are content-bound.
- [x] Family, archive, incumbent, donor, and LOO share one objective digest.
- [x] No Head-A prefilter removes Head-B evidence.
- [x] Write/reopen share the existing total budget and source legality.
- [~] Structure is evaluated once (asserted by a stack counter). Role B's hotspot drift is **measured and reported, not gated** — see §2.5 and `doc/DUAL_ALLELE_DUALF0_AUDIT.md` Appendix E for why a second gate would bind against nothing on this substrate and would move every endpoint identity.
- [x] Existing endpoint and legacy Head meanings are preserved.
- [x] Artifacts and resume bind both Heads, calibration, objective, safety, and arm bundle.
- [~] Matched arms share one depth-zero **molecule and identity** (`sequence_md5`, `endpoint_id`, `source_state_id`). They cannot share one endpoint OBJECT: an endpoint carries one Head's binding, so `b_only`'s content digest differs by construction. The runbook GO check was corrected accordingly (Appendix F).
- [x] Dual-off behavior is regression-equivalent.
- [x] Existing driver and SLURM are extended; no duplicate launcher exists.
- [x] `doc/SCRIPTS.md` reflects the implemented modes and removes stale descriptions.
- [x] Local gates pass before any cluster submission.
- [ ] `LOG.md` is updated only when the behavior-changing implementation lands. *(pending: the Dual layer is opt-in and no behaviour changes until an overlay is supplied, so this lands with the first Dual run.)*
