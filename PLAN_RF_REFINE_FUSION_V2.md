# PLAN - RF-Refine Fusion v2: Trajectory-Coupled Exact-Endpoint Feedback

**Status:** implementation contract; revised after coder gap review of schedule geometry,
assimilation, paired seeds, safety references, calibration producers, depth-0 ownership, and the
background-remask boundary. No V2
behavior is implemented by this document, and no scientific launch is authorized by PLAN
completion alone.

**Implementation target:** a descendant of `fusion_rf_refine` in
`/Users/jerry/Project/MHC-IF-fusion`. The audited implementation baseline is commit
`e091b36876ddbbeac1c548c56d9fe7f7d113ba26` on 2026-08-04. Before coding, the V2 authority,
audit, and this PLAN must be copied or merged into that worktree under one reconciled governance
contract.

**Scientific authority and evidence order:**

1. `doc/FUSION_V2.md` - V2 scientific state, operator, safety, and falsification contract;
2. `doc/RF-Refine-Fusion.md` - frozen complete-state v0 boundary and its objective/feasibility law;
3. `doc/FUSION_V2_Audit.md` - measured V1 evidence and code-state audit; evidence record only,
   not design authority;
4. `PLAN_RF_REFINE_FUSION_V1.md` - historical V1 implementation contract and reusable
   continuation infrastructure;
5. this PLAN - V2 implementation, artifact, and local-validation contract.

If this PLAN conflicts with `doc/FUSION_V2.md`, stop and repair the PLAN. The coder may not resolve
a scientific ambiguity by inventing a default.

---

## 0. Goal, Delivery Boundary, and Task State

### 0.1 Goal

Implement the minimum trustworthy machinery for this recurrent state transition:

```text
live partial state P_d
  -> shared exact complete lookaheads Y_d
  -> exact Head + definitive feasibility evidence
  -> monotone complete archive A_{d+1}
  -> A2 feedback-off view
  -> select compatible (P_d*, y_d*) lineage
  -> source-coupled projection q_phi(P_d*, y_d*) at re-entry r_d
  -> propagated live descendant P_{d+1} at checkpoint c_{d+1}
  -> next exact lookaheads
```

The first independent research-evidence unit is **one feedback cycle**. It asks:

> With the selected endpoint, reopened support, and descendant seeds controlled, does the source
> partial state materially affect the projected state and the distribution of exact descendants?

Only after that transition is replayable, anchor-safe, and causally non-null may real recursive
depth or a capability ladder be enabled.

### 0.2 Delivery boundary

Local code completion must provide:

- typed V2 state, endpoint, archive, projection, transition, and config contracts;
- a pure source-coupled projection kernel with no hidden model dependency;
- sampler support for projected-state propagation and later partial-state capture;
- shared exact lookahead and archive logic whose pre-feedback view is A2;
- a one-cycle runner and a recursively capable runner tested with fake oracles;
- stable artifacts, content-bound resume, full cost/event accounting, and a model-free preflight;
- one production driver surface or a justified reuse of an existing driver; and
- regression evidence that V1 and v0 contracts remain intact.

This PLAN does **not** freeze production values for feedback depth, checkpoint coordinates,
reopened fraction, active width, breadth, structure cadence, or stopping thresholds. It also does
not write Canary commands; those belong in the V2 runbook after code review.

### 0.3 Task state

| Task | Independent deliverable | Status |
| --- | --- | --- |
| V2F0 | authority merge and implementation-base gap audit | pending |
| V2F1 | typed V2 state, identity, and config contracts | pending |
| V2F2 | pure source-coupled projection and schedule-consistency gate | pending |
| V2F3 | projected-state propagation, temporary protection, replay, and capture | pending |
| V2F4 | shared lookahead, exact archive, and prospective A2 view | pending |
| V2F5 | one-cycle runner and paired feedback-transmission evidence | pending |
| V2F6 | progressive recurrence, stationary diagnostic, population, stopping, and breadth-versus-depth allocation | pending |
| V2F7 | artifacts, ledger, resume, driver, and preflight | pending |
| V2F8 | local regression closure and runbook-ready handoff | pending |

No task is complete from code alone. Its named artifacts and adversarial validation must exist.

### 0.4 Explicit non-goals

- no claim that A2, V2, or source coupling is already beneficial;
- no deletion or weakening of V1's `final`-materialization firewall;
- no reinterpretation of `ContinuationResume` as a V2 projected state;
- no masked sequence in v0 `ParticleState` or complete-state Head input;
- no direct Head-gradient, Head-embedding, or token-logit conditioning claim;
- no NMP or other independent immune evaluator inside selection, projection, or stopping;
- no new trained model, denoiser fine-tuning, controller, h-map, SC-GR, D2, or D3 dependency;
- no background global remask inside the first V2/A2 substrate and no remask-on qualification arm;
- no mandatory long v0 suffix on the V2 primary endpoint;
- no unbounded experiment, invented capability plateau, or unspecified automatic rescue tuning;
- no production policy default for unresolved Section 10 choices in `doc/FUSION_V2.md`; and
- no LOG entry for this PLAN or authority-only edits. LOG is updated only when behavior-changing
  implementation lands.

---

## 1. Required Preread and Audited Implementation Base

### 1.1 Required preread for the coder

Read before editing:

1. all five authority/evidence files named at the top of this PLAN;
2. `PROGRESS.md`, `AGENTS.md`, `doc/SCRIPTS.md`, and `scripts/CLAUDE.md`;
3. `inverse_folding/reference_flow/sampler.py`;
4. `inverse_folding/reference_flow/fusion/v1_{records,config,seeds,alloc,ledger,admission}.py`;
5. `inverse_folding/reference_flow/fusion/{state,objective,moves,runner,selection,oracles}.py`;
6. `scripts/rf_fusion_v1_{entry_core,oracles,artifacts,cohort,preflight}.py`;
7. `scripts/run_rf_fusion_v1_entry.py` and `scripts/run_rf_refine_fusion.py`; and
8. all corresponding sampler, V1 entry, admission, driver, and v0 Fusion tests.

### 1.2 Reusable verified substrate

The V1 branch already supplies reusable mechanisms:

- a true pre-denoiser `ContinuationCheckpoint` with exact prefix DFE;
- self-contained identity and fork resume with replay-state hashing;
- canonical-AA20 maturity and hard-anchor validation;
- scalar and batched continuation surfaces;
- lossless `PartialRootPayload`, conditioning digests, seed namespaces, and stable artifacts;
- identity-bound complete Head records;
- V1 physical/logical cost-event infrastructure; and
- v0 complete-state Head/structure evaluation, local reopen/repair, archive, selection, and lineage.

Reuse these mechanisms through shared library interfaces. Do not copy V1 script logic into a new
parallel stack.

### 1.3 Confirmed gaps that V2 must add

The audited baseline does not contain:

- a projected partial-state schema distinct from a fresh V1 root;
- a temporary-protection class that remains editable across future feedback depths;
- a legal `resume -> propagate -> capture later partial state` segment API;
- a library-owned depth-0 `at_step` capture provider shared with the V1 model/oracle preparation
  path;
- a stable step-indexed producer for empirical schedule bands $\mathcal{B}(r)$;
- a feedback-transition identity, lineage edge, seed namespace, or ledger phase;
- endpoint per-position evidence sufficient to reconstruct a projected sampler state;
- a monotone archive shared prospectively by A2 and V2;
- a source-coupled projection policy or source-ablation diagnostic;
- a V2 runner with active partial population and recurrent depth; or
- V2 manifests, transition artifacts, content-bound resume, and driver dispatch.

These are expected implementation gaps, not evidence against V2.

### 1.4 V1 guards that must remain

`authorize_final_materialization()` and the V1 `continuation_resume` mutual-exclusion rules encode
the historical V1-A1 evidence contract. V2 must add new types and call paths rather than make an
`est` endpoint masquerade as a V1 `final` continuation or weaken exact replay.

### 1.5 Measured substrate constraint and frozen V2 substrate

The V1 remask-on `c1_null` maturity clock is late: median `rho=0.30/0.50/0.70` crossings occurred
at steps `90/95/98` of 100. This does not choose V2 coordinates. Generic reparameterized remask is
an inference-only global reopen rule that still runs with `controller=None`; it is not D3, but it
would compete with V2's explicit feedback-support policy for control of reopened identities.

The first V2/A2 substrate is therefore frozen as controller-free, h-map-free, `constant_one`, and
background-remask-free (`remask.enabled=true`, `fraction_scale=0.0`). This reuses the validated
zero-remask code path while making $q_\phi$ the only operator allowed to introduce new masks after
feedback begins. Existing matched evidence found the large degradation in sequence recovery, an
irrelevant objective here, while ESMFold self-consistency changed only slightly
(`mean/median delta scTM = -0.0095/-0.0048`; `<0.5` changed `17.5% -> 18.0%`). That evidence does
not replace a small V2 terminal/anchor/refold sanity, but it removes the need for a substrate
tournament.

---

## 2. Frozen V2 State and Transition Contract

### 2.1 Depth and sampler-coordinate contract

For feedback depth $d$, every transition requires:

$$
r_d < c_d,
\qquad
c_{d+1}\geq c_d.
$$

- $c_d$: source live-state checkpoint and lookahead fork point;
- $r_d$: earlier coordinate assigned to the immediate projected state;
- $c_{d+1}$: next checkpoint reached by committed propagation; and
- $\rho_{\mathrm{edit}}$: realized editable maturity measured from bytes, never inferred from a
  coordinate label.

The implementation exposes both coordinate laws through one segment API:

- `stationary_checkpoint`: $c_{d+1}=c_d$; or
- `progressive_checkpoint`: $c_{d+1}>c_d$.

The primary V2 mode is `progressive_checkpoint`. `stationary_checkpoint` is retained only as one
small matched comparator, not as a second production architecture. The implementation must reject
coordinates inconsistent with the selected mode and a projected state outside the frozen
empirical maturity band $\mathcal{B}(r_d)$. Depth and step jointly key history and lineage;
repeated step values in stationary mode are not the same event. The band is input calibration with
provenance, not a hardcoded tolerance.

The remask-on `c1_null` crossings may not be reused to select coordinates. V2F5/F6 first extend the
existing scanner to measure explicit steps on the frozen no-remask substrate, then freeze a
progressive schedule and one matched stationary diagnostic schedule. This is coordinate
calibration, not an outcome-dependent substrate qualification.

### 2.2 Separate state layers

V2 must use distinct typed objects for:

1. **LivePartialState** - replayable partial bytes; active origin kind, feedback-origin reference,
   active commit depth/step, active sampler score/status, immutable provenance evidence, anchors,
   transient protection, sampler coordinate, realized maturity, conditioning identity, seed state,
   complete-reference identity, and lineage;
2. **CompleteEndpoint** - canonical sequence, exact sequence digest, source partial identity,
   fork seed, immutable endpoint per-position provenance evidence, complete Head evidence,
   structure-feasibility state, cost, and lineage;
3. **ArchiveEntry** - exact endpoint identity, depth first/last seen, feasibility level, elite or
   diversity-frontier membership, and source lineage;
4. **ProjectedPartialState** - the immediate output of $q_\phi$ at $r_d$, including the full
   endpoint-injected/source-injected/reopened/natural-source partition and score-assimilation
   state; and
5. **FeedbackTransition** - source state, selected endpoint, projection-policy identity, immediate
   projected state, propagated descendant state, seeds, costs, and explicit parent edges.

V1 `PartialRootPayload` may be adapted into the initial V2 live state, but it must not be reused as
the recursive state type because its fresh-root invariant `paid_prefix_dfe == step` does not
represent accumulated ancestry cost.

### 2.3 Exact archive and A2 boundary

Every complete lookahead is retained as an exact endpoint record. Archive update is monotone and
must preserve the previous best definitively feasible endpoint even if every new descendant
regresses.

Distinct logical endpoints remain visible even when sequences converge. Duplicate sequence or
sibling multiplicity may not buy extra ancestry mass; selection must operate on declared lineage
families or sequence-equivalence classes while preserving all raw endpoint rows for audit.

The archive state immediately before feedback is the prospective A2 view. A2 and V2 may not
regenerate separate initial roots, lookaheads, Head results, or structure results.

### 2.4 Projection kernel

The defining transition is:

$$
\widetilde P_{d+1}
\sim
q_\phi\left(\cdot\mid y_d^*,P_d^*,S,C,\pi_d,r_d\right).
$$

The initial implementation must expose a pure `source_writeback`-style kernel whose support sets
are supplied by a typed policy result. It operates on a copy and partitions editable positions
into:

- `write_from_endpoint`: write the selected endpoint token and temporarily protect it through the
  propagation segment;
- `inject_from_source_feedback`: intentionally carry a source identity that arose at or after the
  earlier re-entry boundary, reset its active history, retain its later-source evidence only as
  provenance, and temporarily protect it through the propagation segment;
- `reopen`: set the position to mask, reset its sampling score/history consistently, and leave it
  editable; and
- `carry_from_source`: retain only a temporally valid source token, active score, and history.

Hard anchors form a separate permanent class and can never enter `reopen`. Temporary protection is
a separate typed relation over the editable domain, not a synonym for one action set. The initial
`source_writeback`/`explicit_probe` contract assigns exclusive
`protection_expiry_step=c_{d+1}` to endpoint and source-feedback injections. A position is protected
from remask at step $s$ exactly while $s<\text{expiry}$; expiry at the top of $c_{d+1}$ is recorded
before the captured state is exposed. Natural source carries receive no implicit protection, and
temporary protection must not silently become a permanent fixed token.

Because the frozen V2 substrate has zero background remask, this protection law is not a tunable
survival mechanism or a separate arm. It is one segment-local state invariant, implemented once
and expired deterministically at $c_{d+1}$.

Because checkpoints are pre-denoiser top-of-step states, `carry_from_source` is legal only when the
resolved token satisfies $0\leq s_i^{\mathrm{active}}<r_d$. A token committed at step $\geq r_d$
is future information and must be assigned to `inject_from_source_feedback` or
`reopen`; asking the kernel to carry it naturally fails closed. A masked source position remains
unresolved unless another explicit support action writes it. Hard anchors are identified from the
constraint class, never inferred from an unmask step.

The endpoint record retains immutable `endpoint_provenance_evidence_by_pos`; source-feedback
injection likewise retains immutable source provenance. Neither is an active sampler score. The
projected/live state has separate `active_sampler_score_by_pos` and
`active_score_status_by_pos`, where status is one of `masked`, `historical_natural`,
`pending_assimilation`, `assimilated`, or `not_ranked_anchor`. Reopened positions receive canonical
masked state. Natural source carries preserve temporally valid active history. Endpoint/source
injections begin pending, with no fabricated active score.

The active-origin contract contains at least `active_origin_kind`, `feedback_origin_ref`,
`active_commit_depth_step_by_pos`, and `origin_transition_id`, while immutable provenance retains
older lineage. Every endpoint/source-feedback injection sets its active commit event to
`(depth+1, r_d)`; the later source/endpoint commit evidence remains provenance only. This prevents a
recursively carried endpoint-feedback token from losing its original ancestry merely because it
becomes part of the next source state, and prevents stationary-checkpoint cycles from aliasing two
events at the same sampler step.

For a non-null feedback event:

- source and endpoint lineage must be compatible;
- `write_from_endpoint` and `reopen` must both be non-empty;
- `write_from_endpoint` must change at least one source byte or establish distinct bound endpoint
  provenance;
- `reopen` must newly mask at least one previously resolved editable position;
- the four editable sets must be disjoint and exhaustive;
- all hard anchors must match before and after projection;
- the projected state must contain at least one unresolved editable position;
- the complete endpoint remains byte-identical in the archive; and
- an invalid policy result creates an explicit null/stalled event rather than an invented state.

### 2.5 Policy boundary

The projection **kernel** is frozen by this PLAN; the scientific **support-scoring policy** is not.
Implement a policy interface that returns typed support sets and reason evidence. An
`explicit_probe` policy may accept predeclared sets for deterministic tests and the first
state-transition diagnostic, but it must be labeled diagnostic-only and may not become a silent
production default.

No uncertainty threshold, Head-window projection rule, reopen fraction, or protected fraction is
authorized until it is explicitly frozen in config/runbook provenance.

Kernel completion does not authorize a production feedback method. After real one-cycle
transmission and before production `D>1` capability work, freeze one typed
`FeedbackSupportPolicySpec` containing at least:

- residual complete-Head window-to-residue attribution;
- the frozen reference and rule for protecting improvement-associated endpoint identities;
- use of source instability and active-history evidence;
- endpoint-sibling consensus/disagreement inputs;
- a total conflict priority over endpoint adoption, source injection/carry, reopening, and
  protection;
- target/realized reopen counts, maturity-band compatibility, and null/fallback law;
- per-position reason evidence and policy version/config/content digest; and
- a policy-level source-off comparator matched on endpoint, support cardinality, re-entry horizon,
  fork seeds, and resources.

Do not call an identity `causal-improvement` without a positional intervention; use
`improvement-associated` for evidence derived from endpoint/reference comparison alone.

### 2.6 Bidirectional causal contract

The mechanism evidence must support two paired interventions:

1. hold source $P$ and support fixed, change compatible endpoint $y$;
2. hold endpoint $y$ and support fixed, change or ablate compatible source state $P$.

Both comparisons use matched descendant fork seeds. Add a V2-specific `FeedbackPairSeedContext`
over the shared low-level deterministic hash primitive. Its seed identity contains schema,
master seed, campaign, split, protein, a pair ID frozen before either arm exists, depth, shared
$r_d/c_{d+1}$, fork index, and the `matched_descendant` namespace. It explicitly excludes arm or
intervention labels, endpoint/source/state digests, projected bytes, arm-specific transition IDs,
and arm-specific policy digests. Those are treatment identities; including them would silently
unpair the downstream randomness. The artifact records pair ID, fork index, realized seed,
intervention kind, both treatment identities, and shared horizon/support-match identity, then
validates equal seeds within each pair and no collisions across forks.

Runtime cannot prove universal dependence by type checking, so the artifact graph must expose the
paired identities. Immediate projected-byte
changes are construction invariants and belong in deterministic wiring tests. They are not a
scientific mechanism endpoint. The real mechanism analysis must test whether the matched
descendant sequence distributions differ under a power contract frozen before the mechanism
cohort is inspected.

If endpoint changes do not alter immediate projected bytes, endpoint-sensitive wiring is absent.
If source changes do not alter immediate projected bytes, source-sensitive wiring is absent. If
both wiring checks pass but the adequately powered descendant-distribution test is null, feedback
transmission is not established.

Endpoint sensitivity and source transmission do not establish reward directionality. After the
production policy is frozen, a separate paired test holds source, support, propagation horizon,
and fork seeds fixed while using compatible endpoints with a predeclared exact-Head ordering. It
asks whether descendant Head distributions shift toward the better endpoint. A source-null but
reward-directional result is source-forgetting endpoint feedback and belongs to the v0-style
boundary; a source-positive but non-directional result is trajectory-coupled exploration with an
unqualified immune policy.

Fixed-support source ablation and policy-level source-off matching are separate controls. The
former isolates whether source bytes/history propagate. The latter removes source-derived policy
evidence while matching support size and all downstream randomness, and isolates whether the
source-aware support law is useful.

### 2.7 Complete-state safety boundary

- Head receives only complete canonical AA20 sequences.
- Any endpoint promoted into feedback ancestry must carry definitive structure feasibility under
  the frozen v0 contract, not only a proposal prior.
- The first depth binds a predeclared complete reference sequence for whole-landscape new-hotspot
  checks. Its identity is content-bound before any depth-0 endpoint is scored, and the selected
  endpoint may not serve as its own reference. Its identity-bound `HeadScore` remains the immutable
  cumulative safety reference for that lineage at later depths; the immediate selected parent is
  retained separately for step-local telemetry or an independently frozen incremental gate.
V2 implements a new pure whole-landscape comparator over exactly aligned finite Head-window
records:

$$
N_H^{\mathrm{whole}}(y;\bar y)
=
\max_w\left[z_w(y)-z_w(\bar y)\right]_+.
$$

It retains positive mass/count telemetry. Do not call the v0 immediate-parent off-halo API with an
empty halo to imitate this contract.

- The cumulative threshold and any incremental threshold are explicit calibrated config inputs
  with content provenance. The v0 `0.10` value is not inherited silently.
- NMP and other external immune evaluators remain terminal-only reporting.

---

## 3. Sampler Segment and Replay Contract

### 3.1 New projected-segment API

Add a V2-specific sampler input or segment API. It must:

- start from a `ProjectedPartialState` at $r_d$;
- preserve hard anchors and temporary protection;
- require zero background-remask events; only an explicit projection may add masks;
- run the unchanged controller-free, h-map-free base denoiser;
- on the first ordinary forward at $r_d$, use raw base logits and the frozen sampler temperature
  to assimilate every pending injected token into an active sampler score without resampling it;
- protect pending injected tokens through that forward and fail before any active-score consumer
  if a non-anchor committed token remains pending;
- stop at $c_{d+1}$ on the true pre-denoiser boundary;
- emit a new replayable `LivePartialState` without terminal residual completion;
- optionally continue/fork complete lookaheads only through the existing exact completion path;
- record logical lane-DFE and physical batched forwards; and
- preserve scalar/batch semantic parity.

Before implementing that API, extract one sampler-neutral arbitrary-token log-probability
primitive shared by categorical sampling and assimilation:

$$
\ell_j
=
\log\operatorname{softmax}\left(z_j/T\right)_{a_j}.
$$

It accepts raw base logits, canonical AA20 token IDs, and a finite positive sampler temperature;
it draws no RNG and adds no DFE. Do not import the D3 chosen-token helper: that helper is
controller-family evidence and does not expose the required sampler-temperature contract.

Do not implement V2 by passing `continuation_resume` and `continuation` simultaneously. Do not
change V1 mutual exclusion. A projected state has different identity and accumulated-cost semantics
and deserves a distinct input contract.

### 3.2 Replay and integrity

Integrity coverage includes projected bytes; active origin/reference/commit fields; immutable
provenance evidence; active sampler scores/status; unmask history; hard anchors; transient
protection; editable set; $r_d/c_{d+1}$; conditioning, source, endpoint, and policy identities; RNG
state/fork seed; and accumulated lineage cost.

Tampering or a mismatch in sampler steps, tokenizer, backbone, coordinate mask, constraints,
reference sequence, runtime null sentinels, or policy digest must fail before a denoiser call.

### 3.3 Temporary-protection lifecycle

The sampler must distinguish:

- permanent hard anchors;
- positions protected only during the current feedback propagation segment; and
- ordinary editable positions.

Temporary positions remain part of the maturity denominator. Background remask is absent, so the
first implementation records one deterministic segment-local protection interval and expires it
at the captured descendant checkpoint. The next explicit policy may independently protect or
reopen the identity; no implicit cross-depth protection is allowed.

Endpoint/source-feedback injections must be assimilated before temporary protection expires.
Natural source tokens retain their historical sampling score; an assimilated score is explicitly
the current token's score under the first active re-entry forward, not a retroactive claim about
how that token was originally sampled. Scalar and batch paths must produce identical assimilation
scores/status and protection expiry.

### 3.4 Logical cost

For one source checkpoint $c$ with $K$ lookaheads under an $S$-step sampler:

$$
C_{\mathrm{screen}}=c+K(S-c).
$$

A projected segment from $r_d$ to $c_{d+1}$ costs exactly $c_{d+1}-r_d$ logical lane-DFE before
the next checkpoint. Its later lookaheads pay their own tails. The ledger must not charge the
source prefix again or treat archive reads as model calls.

### 3.5 Step-indexed schedule-band producer

Extend the existing registered `scripts/rho_maturity_scan.py`; do not create a second scan script.
Add mutually exclusive rho-crossing and explicit-step modes. Step mode captures with
`ContinuationRequest(at_step=r)` and writes raw per-attempt states plus summary quantiles for
$\mathcal{B}(r)$, including normalized maturity and absolute unresolved editable mass.

The calibration artifact binds sampler config/content, `n_steps`, tokenizer, cohort, backbone and
coordinate-mask policy, constraint/anchor stratum, seed schema, attempted seeds, and all failed
captures. Replace Python's process-randomized `hash()` seed construction with the stable project
seed primitive, and do not swallow failed forks. The scan is Head-free and outcome-independent.
Its quantile/tolerance choice remains a later scientific calibration, but projected production
states may consume only a frozen artifact with matching identity. Register the extended script in
`doc/SCRIPTS.md`.

---

## 4. Shared Lookahead, Archive, and Population Contract

### 4.1 Exact lookahead pool

All lookaheads at a depth fork from the same committed live-state bytes under persisted, disjoint
seed namespaces. Complete endpoint scoring uses one identity-bearing batch Head interface and
rejects missing, duplicate, extra, or reordered results.

The runner stores the exact sequence and per-position completion evidence before any ranking.
Endpoint values may support deterministic selection, but no scalar aggregate replaces the endpoint
or becomes an inherited parent.

### 4.2 Feasibility and promotion

Endpoints may enter the archive as `unvalidated`, but an endpoint must become `definitive` before
it can:

- become feedback ancestry;
- support the reported feasible frontier; or
- serve as a final returned design.

Structure cache hit/miss, execution, metrics, failure reason, and cost must remain explicit.

### 4.3 A2 feedback-off view

At every depth, persist an A2 view of the exact archive immediately before feedback. For matched
mechanism comparisons, work assigned to V2 after feedback is reassigned to additional exact futures
from unchanged source states in A2, using declared disjoint seeds.

A2 is not a separate root-generation run and may not be reconstructed post hoc from only selected
rows.

### 4.4 Active partial population

Population selection acts on lineage families, not duplicate endpoint rows. The active partial
population carries explicit weights or uniform slots, parent transition IDs, and archive references.
No family receives extra ancestry mass solely because it emitted more identical siblings.

The one-cycle implementation may use one selected lineage. Recursive population width remains a
config field without a default until V2F6.

### 4.5 Stopping and null behavior

Stopping reasons are typed: depth cap, no admissible endpoint, invalid projection, no novel
descendant, frontier plateau, diversity collapse, or operational ceiling. Any stop returns the
best existing definitive archive state; it never destroys or replaces it with a worse descendant.

---

## 5. Configuration, Provenance, Artifacts, and Ledger

### 5.1 Separate V2 config

Create a strict V2 config separate from V1 entry and v0 Fusion configs. It must contain explicit:

- schema/campaign/split/master-seed identity;
- base sampler and tokenizer identity;
- Head and structure-backend identity;
- backbone, coordinate-mask, constraint, and complete-reference identity;
- feedback enabled/disabled mode;
- depth cap and active population width;
- frozen controller-free/h-map-free/no-remask substrate identity, coordinate law, and ordered
  `(r_d, c_d, c_{d+1})` schedule;
- lookahead breadth per depth;
- projection kernel, temporal-history rule, active-score assimilation rule, and support-policy
  identity/spec digest;
- step-indexed schedule-band calibration identity and the coupled admissible mask-load contract;
- cumulative whole-landscape new-hotspot reference/threshold calibration, optional incremental
  gate, and structure cadence contract; and
- DFE, Head, refold, GPU-time, walltime, and retry hard caps.

Scientific numeric fields have no library defaults. Diagnostic-only policy values must be visibly
named and rejected in a capability or holdout phase.

The config validator requires `remask.fraction_scale=0.0` for every V2/A2 arm in this PLAN.
`progressive_checkpoint` is the production-capable law; `stationary_checkpoint` is accepted only
for the explicitly labeled comparator. Neither law changes the segment implementation.

The V2F5/F6 runtime uses one content-bound controller-free, h-map-free, `constant_one`,
background-remask-free substrate for every V2/A2 causal arm. It must use a V2-named config rather
than inheriting the old remask-probe experiment identity. Frozen v0 keeps its existing package as a
system boundary and is not mutated to manufacture a one-knob causal comparison.

### 5.2 Content provenance

The run signature binds file **contents**, not paths alone, for at least:

- test/cohort table;
- source/reference sequences;
- backbone and coordinate mask;
- constraint manifest;
- sampler and tokenizer config;
- Head config/checkpoint;
- structure backend/config;
- V2 config and projection policy;
- schedule-band calibration; and
- code revision.

All cluster paths remain CLI arguments. Missing content identity fails closed.

### 5.3 Required evidence objects

The writer must produce stable typed tables or sidecars for:

| Evidence object | Required load-bearing fields |
| --- | --- |
| run manifest | all content identities, config, split, caps, seed namespaces, code revision |
| partial states | state/transition ID, depth, coordinate, maturity, masks, active origin/reference/commit, active sampler score/status, immutable provenance refs, anchors, temporary protection, replay digest, lineage |
| complete endpoints | endpoint ID, source state, seed, sequence/digest, Head/window evidence, feasibility, immutable endpoint per-position provenance, cost |
| archive | endpoint membership, feasibility level, elite/frontier status, first/last depth, family identity |
| feedback events | source, selected endpoint, policy, four support sets/reasons, origin assignments, assimilation status, projected state, propagated state, paired-control identity |
| A2 views | exact pre-feedback archive membership and any matched extra-lookahead allocation |
| compute ledger | request/attempt/event identity, logical DFE, physical forwards, Head calls, structure attempts, cache, retry, GPU/walltime |
| terminal validation | definitive structure, independent immune results, diversity, optional v0 before/after mapping |

Raw logical duplicates remain in endpoint tables. Summary/frontier tables may collapse by declared
equivalence only while preserving the full mapping.

### 5.4 Resume and failure accounting

Checkpoint at least at each committed state transition and archive update. Resume requires exact
run-signature equality and validates every fragment before aggregation.

Oracle requests are journaled before execution and outcomes afterward. If a backend cannot report
the physical cost burned before failure, persist `unknown_after_start` rather than zero or a guessed
value. Logical assigned work, observed physical work, and unknown physical work are separate fields.

### 5.5 Reuse-first driver boundary

Before adding a driver, compare the V2 requirements against `run_rf_fusion_v1_entry.py` and
`run_rf_refine_fusion.py` under the `scripts/CLAUDE.md` 60% reuse rule.

- Shared model/oracle preparation belongs in library helpers, not copied scripts.
- Do not add V2 as an ambiguous mode inside the V1 scientific config.
- If one existing driver can be safely parameterized, extend it.
- If the CLI/state graph is materially different, one new V2 driver is permitted with the reuse
  decision recorded and the script registered in `doc/SCRIPTS.md`.

Launcher and Canary commands are outside this PLAN revision and will be added to the runbook after
the local code gate.

---

## 6. Build Tasks

### Task V2F0 - Reconcile authority and audit the target branch

**Objective:** ensure the coder operates against one current scientific and governance contract.

**Work:**

- copy/merge `FUSION_V2.md`, `FUSION_V2_Audit.md`, and this PLAN into the Fusion worktree;
- verify the audited commit or record all drift;
- map each V1/v0 reusable surface and each confirmed V2 gap;
- inspect dirty/untracked files before editing; and
- append the audit record to this PLAN without marking implementation tasks complete.

**Acceptance:** no unresolved authority conflict, no silent scientific default, and a reviewed
reuse map exists before code changes.

### Task V2F1 - Typed state, identity, and config contracts

**Objective:** make illegal cross-layer states unrepresentable before adding sampler behavior.

**Deliverables:** V2 state/config types, active-origin/provenance types, active/provenance score
namespaces and score status, canonical serialization, full content hashes, ordinary and matched-pair
seed namespaces, compatibility validation, cycle-coordinate mode, schedule-band calibration type,
typed cumulative complete-reference evidence, schedule ordering, and explicit null/stalled
outcomes.

**TDD acceptance:** corruption, row reordering, duplicate IDs, incompatible endpoint/source pairs,
missing provenance, inconsistent origin/history/score status, illegal coordinates, non-finite
numerics, and diagnostic policy in a production phase all fail before model calls.

### Task V2F2 - Pure source-coupled projection

**Objective:** implement $q_\phi$ as an auditable state transition rather than hidden sampler code.

**Deliverables:** policy-result type, `source_writeback` projection kernel, four-way editable
partition, orthogonal hard/transient/open classes with exclusive expiry, depth-step temporal-history
gate, dual evidence namespaces, schedule-band/mask-load gate, null transition, reason telemetry,
and exact archive non-mutation.

**TDD acceptance:**

- hard anchors never change or reopen;
- a source token committed before $r_d$ carries byte/history/active score exactly;
- a source token committed at or after $r_d$ fails ordinary carry and requires explicit
  source-feedback injection or reopening;
- endpoint/source-feedback identities retain exact immutable provenance but begin without a
  fabricated active score;
- terminal endpoint scores never enter active sampler scores;
- reopened state is canonical;
- same inputs/seed produce identical state regardless of row order;
- paired fixtures demonstrate endpoint dependence and source dependence with fixed support; and
- a source-ignoring or endpoint-ignoring implementation is caught by positive-control tests.

### Task V2F3 - Projected propagation and later capture

**Objective:** run a projected state from $r_d$ to $c_{d+1}$ and capture a new live state without
weakening V1 replay.

**Deliverables:** sampler-neutral arbitrary-token log-probability primitive, V2 segment API,
first-forward active-score assimilation, temporary-protection enforcement/expiry, exact replay,
forked continuation, scalar/batch parity, logical/physical cost, capture telemetry, and a
library-owned depth-0 capture provider. The provider reuses
`ContinuationRequest(at_step=c_0)`, hard-anchor, conditioning, and replay primitives and adapts the
fresh checkpoint into `LivePartialState`; it does not copy the nested V1 script closure.

**TDD acceptance:** uninterrupted-versus-resumed bytes match under identity mode; fork seeds are
disjoint; pending injection becomes assimilated from the first active raw-logit forward without
resampling; no pending non-anchor token reaches an active-score consumer or protection expiry;
scalar/batch
assimilation matches both the categorical sampler's chosen-token score and the configured
temperature; protection and maturity denominators are correct; depth-0 capture has exact
coordinate/DFE/anchor/provenance identity; tampering with origin,
provenance, active score/status, or history fails before denoiser; V1 continuation tests remain
byte-identical.

### Task V2F4 - Shared lookahead, exact archive, and A2 view

**Objective:** retain exact endpoint breadth prospectively and make A2 a view of the same run.

**Deliverables:** endpoint generation/scoring from the typed depth-0/recurrent provider,
per-position evidence, identity-bound Head batch, immutable cumulative-reference Head evidence,
the new whole-landscape hotspot comparator, feasibility promotion, monotone archive, family-aware
selection, A2 snapshot, and matched extra-lookahead allocation.

**TDD acceptance:** no scored endpoint is discarded from raw evidence; reordered Head results fail;
Head/reference identity or window-grid mismatch fails; a rolling-reference ratchet is caught by the
cumulative gate; archive elite cannot regress; duplicates cannot buy ancestry; A2 and V2
pre-feedback records have identical endpoint IDs and costs.

### Task V2F5 - One-cycle runner and feedback-transmission evidence

**Objective:** complete one source-coupled feedback cycle and expose the first causal falsifier.

**Deliverables:** source checkpoint -> lookaheads -> definitive selection -> projection -> propagated
checkpoint -> descendant lookaheads, plus feedback-off, endpoint-change, source-change/ablation, and
source-shuffle paired views using matched supports and seeds. The runner must also support a
reward-ordered endpoint pair and a policy-level source-off view with matched support cardinality,
horizon, seeds, and resources. Emit nested protein/source/fork identities and the sequence-space
fields required by the later mechanism and policy-qualification analyses. All paired descendants
use `FeedbackPairSeedContext`; treatment content may not enter the shared seed.

**Acceptance:** one ordinary and one anchored fake-oracle protein reconstruct every state edge;
archive survives forced descendant degradation; all paired identities and realized seeds are
explicit; paired arms use identical seeds while forks are collision-free; a deliberately
source-blind kernel fails the deterministic source-dependence check.

**Scientific boundary:** local tests validate wiring only. Real feedback transmission remains a
cluster mechanism gate and must not be inferred from fake-oracle success.

### Task V2F6 - Progressive recurrence, stationary diagnostic, population, and stopping

**Objective:** generalize the verified transition to a progressive depth $D\geq1$ without changing
its semantics, and implement one compact stationary comparator through the same segment API.

**Deliverables:** a step-index-calibrated progressive schedule, active partial population,
parent/family accounting, archive carryover, breadth-versus-depth allocation, typed stopping, and
deterministic replay. The same executor also accepts one predeclared stationary $D=2$ schedule.
Every causal arm uses the frozen no-remask substrate. A small terminal-AA20, anchor, and refold
sanity is a launch check, not a third scientific arm.

**Acceptance:** depth-0 equals A2; depth-1 equals the V2F5 runner; progressive checkpoints strictly
advance and reject exhausted horizons; stationary depth-2 may reuse a sampler step without reusing
a state/event identity; both laws call the same segment executor; every propagation segment has
zero background-remask events and unresolved mass cannot increase except at explicit projection;
depth-2 lineage is acyclic and replayable; no multiplicity purchase; null/stalled branches preserve
the archive. Production `D>1` remains launch-disabled until both a powered real one-cycle
source-transmission gate and the frozen FeedbackSupportPolicy one-cycle reward-directionality gate
pass.

### Task V2F7 - Artifacts, ledger, resume, driver, and preflight

**Objective:** make the whole local path executable and scientifically auditable.

**Deliverables:** stable schemas, atomic fragments, content-bound resume, append-only attempt
journal, aggregate validation, strict config/preflight, the step-indexed $\mathcal{B}(r)$ extension
and calibration artifact, lazy real-oracle wiring, driver reuse decision, and `doc/SCRIPTS.md`
registration. Extract common model preparation and fresh-root capture from the V1 script into a
shared library factory; the V1 path delegates to it byte-identically and V2 consumes the same
factory rather than copying a closure.

**Acceptance:** fake full-driver run, crash after paid work, retry, shard reorder, empty/partial
cohort, stale input/config, duplicate fragments, and zero-success cases all produce correct
non-zero exits or exact resume behavior. `--print-config` and `--dry-run` load no model and derive
one shared config digest.

### Task V2F8 - Regression closure and runbook-ready handoff

**Objective:** demonstrate that the local implementation is complete enough for separate Canary
planning without claiming real-model success.

**Deliverables:** targeted and broad regression results, import/compile checks, one ordinary and one
anchored fake end-to-end artifact bundle, completed PLAN checklist, current `PROGRESS.md`, and one
LOG entry for the behavior-changing implementation.

**Acceptance:** all local gates in Section 7 pass; V1-A1 and v0 regression suites remain green; the
handoff lists the still-unmeasured real-model transmission, structure, timing, and calibration
quantities. Do not write `CANARY-READY` before independent review.

---

## 7. Test and Verification Matrix

### 7.1 Required targeted suites

Add focused suites for:

- V2 state/config/hash/seed contracts;
- pure projection and schedule-band validation;
- projected sampler segment, temporary protection, replay, and batch parity;
- endpoint/archive/A2 identity;
- one-cycle paired mechanism views;
- recurrent lineage and stopping;
- artifacts/ledger/resume/preflight; and
- full fake-driver ordinary and anchored proteins.

### 7.2 Required adversarial cases

- source/endpoint incompatibility and sequence-digest mismatch;
- projection opens an anchor or omits an editable position from the partition;
- a token committed at or after $r_d$ is accepted as natural source carry;
- endpoint/source provenance is copied into the active sampler score;
- an injected token reaches an active-score consumer or protection expiry while still pending
  assimilation;
- endpoint-written token provenance does not match the endpoint;
- origin, provenance, active-score/status, or active-history tampering passes integrity checks;
- source or endpoint input is ignored by projection;
- declared coordinate and realized maturity are incompatible;
- stationary recurrence aliases two depths at the same sampler step;
- any background-remask event occurs in a V2/A2 propagation segment;
- a progressive schedule exhausts its incomplete lookahead horizon;
- schedule-band calibration uses process-randomized seeds, hides failed attempts, or mismatches the
  runtime substrate;
- temporary protection leaks across depth or silently becomes permanent;
- endpoint and source-feedback injections receive accidental unequal expiry;
- assimilation score differs from categorical chosen-token log-probability at temperature not
  equal to one;
- masked/non-AA endpoint reaches Head or archive promotion;
- Head/reference grids or Head identities differ in the cumulative new-hotspot gate;
- rolling-reference safety accepts cumulative hotspot drift;
- provisional structure result purchases feedback ancestry;
- duplicate sibling/sequence purchases multiple active slots;
- archive elite is overwritten by a worse descendant;
- V2 and A2 regenerate different pre-feedback pools;
- a treatment endpoint/source/state digest changes a matched descendant seed;
- RNG/config/backbone/policy tampering before resume;
- scalar and batch paths disagree on bytes, scores, or DFE;
- failure/retry loses paid work or records unknown cost as zero;
- stale checkpoint is accepted under changed scientific content;
- all proteins fail but the driver exits zero; and
- a V2 change breaks V1 exact replay or v0 complete-state selection.

### 7.3 Local verification

The coder must report exact commands and counts rather than copying historical numbers. At minimum:

```bash
python -m py_compile <all changed Python files>
pytest <new V2 targeted suites> -q
pytest tests/inverse_folding/test_reference_flow_sampler_continuation.py -q
pytest <V1 entry and v0 Fusion regression suites> -q
git diff --check
```

If local optional dependencies prevent broader collection, name every excluded test and reason.
A green unit suite is not evidence of real Head, structure, or feedback-transmission behavior.

---

## 8. Scientific Launch Boundary

### 8.1 Code-complete is not mechanism-positive

After V2F8, the implementation may be ready for a runbook-defined state-transition Canary. It is
not yet evidence that five late denoiser updates, or any chosen earlier re-entry, transmit selected
endpoint information usefully.

### 8.2 State-transition wiring canary

The first real run should test one ordinary and one anchored protein with:

- the same source state and support but different compatible endpoints;
- the same endpoint and support but real versus ablated/shuffled source state;
- matched descendant seeds;
- one late and one earlier re-entry coordinate chosen from measured schedule bands; and
- complete Head plus the frozen definitive structure gate.

This run validates real-model wiring, anchor preservation, replay, and whether both interventions
reach different immediate projected states. It is not powered to distinguish absent descendant
transmission from high within-root variance and may not produce a negative scientific verdict.

### 8.3 Mechanism-cohort power freeze

Before opening a mechanism cohort, run a development variance calibration and freeze all of:

- the number of proteins, compatible source lineages per protein, and matched descendant forks per
  condition;
- protein as the top-level independent unit, with lineages and forks treated as nested repeated
  measurements;
- normalized descendant Hamming distance as an interpretable effect size;
- one primary sequence-space distribution statistic and paired clustered analysis;
- the minimum scientifically relevant detectable effect and the uncertainty/resampling procedure;
  and
- secondary complete-Head, structure, and archive-direction readouts.

The exact statistic and numerical counts must come from the variance calibration and be frozen in
the runbook; the coder may not invent them. If the feasible design cannot resolve the frozen
minimum effect, report `underpowered_unresolved`, not a mechanism-negative result.

Only the powered mechanism cohort supports the following interpretations:

- transmission at the late point: the frozen substrate supports the minimal V2 transition;
- transmission only at the earlier point: keep late lookahead but use deeper re-entry;
- no change under a strong positive-control projection: reject or redesign the current projection
  kernel before recursion; and
- source ablation has no effect: classify the mechanism as source-forgetting v0-style reopen.

### 8.4 FeedbackSupportPolicy freeze and one-cycle qualification

After powered source transmission is positive and before recursive capability, freeze the
production `FeedbackSupportPolicySpec` from Section 2.5. Then run one matched policy-qualification
comparison with:

- compatible endpoints whose exact complete-Head order is frozen before descendants are opened;
- identical source state, propagation horizon, fork seeds, support cardinality, and resource caps;
- the source-aware production policy versus its endpoint-only source-off matched policy; and
- descendant Head direction as the runtime immune-direction readout, with anchors, whole-landscape
  new-hotspot, and definitive structure as safety constraints.

Record source dependence and reward directionality as orthogonal outcomes:

| Source dependence | Reward directionality | Verdict |
| --- | --- | --- |
| positive | positive | `immune_directed_transition_supported`; capability remains untested |
| positive | null/adverse | `source_coupled_policy_unqualified` |
| null | positive | `source_forgetting_v0_like` |
| null | null | `transition_policy_rejected` |
| underpowered | any | `mechanism_unresolved` |

Only the first row permits production `D>1`. Complete Head qualifies runtime directionality; it
does not replace terminal independent immune validation.

### 8.5 A2 is not blocked by the V2 mechanism gate

Prospective A2 validation may proceed after the shared lookahead/archive path passes its identity,
endpoint-firewall, archive, and cost guards. It does not require a non-null source-coupled
transition because it contains no projection operator. Its comparison, validation endpoint, and
claim boundary must nevertheless be frozen independently; A2 evidence cannot be reported as V2
mechanism evidence.

### 8.6 Recursive V2 capability work remains conditional

Do not run a recursive V2 capability ladder, definitive V2 cohort, or V2 holdout until the powered
mechanism cohort establishes source transmission and the frozen production policy passes the
reward-directionality qualification. After both gates, freeze the first small breadth-depth ladder
and the A2/v0 comparison in the runbook; do not retrofit thresholds after seeing the capability
result.

---

## 9. Coder Completion Checklist

- [ ] Authority and governance are present and reconciled in the Fusion worktree.
- [ ] V1 and v0 guards remain intact.
- [ ] V2 recursive state is distinct from a fresh V1 root.
- [ ] Coordinates, maturity, support, and horizon are explicit and validated.
- [ ] Progressive is the primary coordinate law; stationary is a same-kernel diagnostic;
      `(depth, step)` identities remain distinct.
- [ ] V2 and A2 use the same content-bound no-remask substrate; only explicit projection can add
      masks, and every segment records zero background-remask events.
- [ ] Step-indexed $\mathcal{B}(r)$ is produced with stable seeds, raw attempts, content provenance,
      and both normalized and absolute maturity evidence.
- [ ] Projection materially consumes compatible source and endpoint state.
- [ ] Immediate projected-byte changes are treated as wiring evidence, not mechanism evidence.
- [ ] Source tokens committed at or after re-entry are explicitly injected or reopened, never
      mislabeled as natural history.
- [ ] Endpoint/source provenance evidence is separate from active sampler score/history.
- [ ] Every injected non-anchor token is assimilated before any active-score consumer or
      protection expiry, with scalar/batch parity.
- [ ] Assimilation and categorical sampling share the same sampler-neutral temperature-scaled
      chosen-token log-probability primitive.
- [ ] Hard anchors and temporary protection have separate semantics.
- [ ] Projected propagation captures a later partial state with exact replay and cost.
- [ ] Complete Head receives canonical AA20 only.
- [ ] Feedback ancestry requires definitive structure feasibility.
- [ ] Whole-landscape hotspot safety uses an immutable cumulative reference and calibrated V2
      threshold; rolling parent telemetry cannot replace it.
- [ ] Exact endpoints are retained; archive is monotone.
- [ ] Duplicate endpoints or siblings cannot buy ancestry mass.
- [ ] A2 is a prospective view of the same pre-feedback endpoint pool.
- [ ] Prospective A2 validation is independently gated and is not blocked by a V2 mechanism null.
- [ ] One-cycle endpoint/source paired controls are emitted with matched seeds/support.
- [ ] Matched descendant seeds exclude every treatment-dependent identity and pass parity/collision
      checks.
- [ ] The mechanism statistic, nested unit, fork/cohort counts, and detectable-effect floor are
      frozen from variance calibration before a scientific mechanism verdict.
- [ ] Source dependence and reward directionality receive separate typed verdicts.
- [ ] The production FeedbackSupportPolicy and matched source-off policy are frozen and qualified
      before production `D>1` capability work.
- [ ] The depth-0 complete reference is frozen before endpoint scoring and cannot self-reference
      the selected endpoint.
- [ ] Depth-0/depth-1/depth-2 semantics are consistent under fake oracles.
- [ ] One library-owned depth-0 capture provider is shared by V1-compatible and V2 drivers; no
      script closure is copied.
- [ ] All state and event edges are replayable without sequence guessing.
- [ ] Resume binds complete scientific provenance and retains failed work.
- [ ] Cost ledger separates logical, observed physical, and unknown physical work.
- [ ] Model-free print-config/dry-run fail closed.
- [ ] Driver decision satisfies reuse-first and every new script is registered.
- [ ] Targeted and V1/v0 regressions pass with exact reported evidence.
- [ ] `PROGRESS.md` is current after implementation.
- [ ] One LOG entry exists only for the behavior-changing implementation.
- [ ] Runbook/Canary remains a separate post-review action.

---

## Appendix A - V2F0 Initial Audit Record

**Recorded:** 2026-08-04.

- Main authority worktree: `/Users/jerry/Project/MHC-IF`, branch `dev_head`, commit
  `f80b134e49ba4e332e11f1c2f9b68a076cd569c0`.
- Implementation worktree: `/Users/jerry/Project/MHC-IF-fusion`, branch `fusion_rf_refine`, commit
  `e091b36876ddbbeac1c548c56d9fe7f7d113ba26`, clean at audit time.
- `doc/FUSION_V2.md`, `doc/FUSION_V2_Audit.md`, and this PLAN are not yet present on the
  implementation branch.
- Existing V1 continuation and v0 complete-state infrastructure is reusable, but no projected
  partial state, feedback transition, or recursive V2 runner exists.
- V1 materialization and resume guards are historical experiment protections and must not be
  weakened to create V2.
- The V2 authority now separates executed V1-A1 negatives from post-hoc endpoint diagnostics,
  distinguishes lookahead from re-entry coordinates, and defines v0 as the source-forgetting
  complete-state local-feedback boundary.

## Appendix B - Coder Gap-Review Resolution

**Recorded:** 2026-08-04.

- The measured late `c1_null` clock is real but belongs to the remask-on V1 substrate. The first V2
  program freezes a controller-free, h-map-free no-remask substrate, uses progressive recurrence as
  the main law, and keeps one same-kernel stationary diagnostic. There is no substrate tournament.
- Re-entry coordinate, maturity band, and total mask/reopen cardinality are coupled. Residue
  identity and support geometry remain policy choices within that feasible cardinality.
- Assimilation uses a new sampler-neutral, temperature-matched arbitrary-token log-probability
  primitive. Controller/D3 state remains absent.
- Paired descendant seeds now have an explicit content-exclusion law so the intervention cannot
  change its own matched randomness.
- V2 whole-landscape hotspot safety is a new cumulative-reference contract with a calibrated
  threshold; v0 immediate-parent/off-halo semantics are not reused by name.
- Endpoint and source-feedback injections share a first-version segment-long protection law;
  first-forward protection is only the mandatory assimilation guard.
- The existing maturity scan is extended in place to produce step-indexed $\mathcal{B}(r)$ with
  stable seeds and raw attempts.
- Depth-0 capture and shared model/oracle preparation receive an explicit library owner; V2 may not
  copy the V1 script closure.

## Appendix C - V2F0 Execution Record (2026-08-04)

**Recorded:** 2026-08-04.

- The three authority documents were copied from `/Users/jerry/Project/MHC-IF` into the
  implementation worktree `/Users/jerry/Project/MHC-IF-fusion` and each copy was verified
  byte-for-byte against its source by SHA-256: `doc/FUSION_V2.md` =
  `ecf7b0d8a158acccbfbd936f07d0926b1b880026bbd368fef178f627b4580625`; `doc/FUSION_V2_Audit.md` =
  `0b830045d3149f21fdb5d9ad8f5cdc3bc1b9538d6e48a321fb9b23e7a2d2bca7`;
  `PLAN_RF_REFINE_FUSION_V2.md` =
  `fad466cac75fd650cd8eae87090a5308ce516de5daeba272d34a4debd31016dd`. This appendix was appended
  after that verification, so the copied PLAN now differs from its source by this section alone.
- The audited implementation baseline named in the PLAN header and in Appendix A is verified with no
  drift. `/Users/jerry/Project/MHC-IF-fusion` is on branch `fusion_rf_refine` at HEAD
  `e091b36876ddbbeac1c548c56d9fe7f7d113ba26`, and no tracked file is modified, staged, or deleted.
  The only untracked path present before this task was `doc/FUSION_V2_Plan_Review.md`, the expected
  new review file; the three copies above are the only paths this task added. Appendix A's statement
  that the V2 authority, audit, and this PLAN are not yet present on the implementation branch is
  superseded by that copy.
- `PROGRESS.md` and `doc/SCRIPTS.md` were brought current by a parallel task.
- Two items from the coder gap review are carried forward UNRESOLVED and are not closed by this
  record:
  - No refold-dominated cost or feasibility model exists anywhere in this PLAN. §5.1 requires hard
    caps on DFE, Head, refold, GPU-time, walltime, and retries, and §5.3 requires a compute ledger
    covering logical DFE, physical forwards, Head calls, structure attempts, cache, retry, and
    GPU/walltime, but no section converts those quantities into a projected cost or a feasibility
    bound, even though the measured unit costs of 2.42 s per definitive refold and no more than
    0.029 s per DFE are recorded at `doc/RF_Fusion_v1_Cluster_Runbook.md:607-608`.
  - §4.3 requires that work assigned to V2 after feedback be reassigned to additional exact futures
    from unchanged source states in A2, but does not name the resource on which that matched-compute
    reallocation is matched. That resource is therefore implemented as a required config field with
    no default, so a run cannot proceed until it is declared.

### Baseline drift (recorded 2026-08-04, after Appendix C was first written)

The implementation branch advanced past the audited baseline while V2F1 was in progress. Recorded
here rather than silently accepted, per V2F0's "verify the audited commit or record all drift".

- Audited baseline in the PLAN header and Appendix A: `e091b36876ddbbeac1c548c56d9fe7f7d113ba26`.
- Actual `fusion_rf_refine` HEAD after a fast-forward pull: `aa3ec41` ("update from cluster 0804"),
  which brought in `35179c8` (ADA conservation-lock manifests) and `3566443` (two-chain
  hetero-complex gate; PD1_b2 binder de-immunization).
- Footprint relevant to V2: **none**. Within `inverse_folding/reference_flow/` the three commits add
  only new config YAMLs; `sampler.py`, `fusion/*.py` and every symbol in the interface map's reuse
  ledger are byte-identical to the audited baseline. `doc/SCRIPTS.md` gained ten registrations and
  dropped the `eval_tetramer_gate.py` entry, whose script no longer exists.
- Consequence: the V2F0 reuse map and the frozen interface map remain valid without re-derivation.
  The baseline commit named in Appendix A is retained as the audit reference; `aa3ec41` is the
  branch point actually built on.
