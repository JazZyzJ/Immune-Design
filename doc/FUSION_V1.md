# RF-Refine Fusion v1: Pre-Terminal Continuation-Value Allocation

**Status**: scientific design document for reward allocation before sequence completion.
Dual-allele capability remains a separate, paper-critical high-priority track, but its
scientific law, implementation, and experiment design are intentionally left open here. This
document is not an implementation plan, experiment matrix, or configuration contract.
Companion to `doc/RF-Refine-Fusion.md` (the v0 method and its evidence ledger).

**Provenance of the reasoning**: this document consolidates the v0 adjudication
(RAR 0030 P3, RAR 0031 final integration, RAR 0033 FK weight-lifecycle audit) and three
reviews commissioned while writing it — a code-feasibility audit of the fusion sampler, a
literature survey on intermediate-state reward guidance, and an adversarial review of the
first draft. Several central claims were corrected across those passes; the corrections are
marked in place, because the path is part of what this document records. The most important
correction is that the first draft over-argued a "the FK negative was a theorem" narrative
and confused full-mask generation with the later complete-state Fusion handoff; both are
fixed below (§2, §8).

## 0. Two methods this document keeps distinct

The first draft conflated two different v1 methods under one label. They are separated here
because they have different contracts, costs, and scientific meaning, and the decision gate
(§6) exists largely to decide whether the second is worth promoting beyond calibration.

| method | what it inherits | scientific content |
|---|---|---|
| **Branch-and-materialize** | fork $K$ *complete* rollouts from a partial root, evaluate the complete sequences, select and inherit a complete sequence | keeps the v0 exact-complete-state contract unchanged. It is still terminal selection; the $K$ siblings of one root are more correlated than independent full trajectories. It is a deferred diagnostic, **not** a V1-A arm or implementation requirement. |
| **Partial-root continuation-value allocation** | use complete rollouts to estimate a partial root's continuation value, then use deterministic beam to select and **propagate the partial root itself** | genuine pre-terminal reward allocation. The evaluated object (a completed rollout) differs from the inherited object (a partial state), which requires a new value, RNG, lineage, replay, and feasibility contract that v0 does not have. |

Everywhere below, "the trajectory method" means partial-root continuation-value allocation.
Branch-and-materialize is defined only to prevent endpoint inheritance from being mistaken for
partial-root propagation; it is outside the active T0/P1 comparison. Stochastic SMC/FK is not
part of the v1 program.

## 1. Where v0 leaves us

v0 is converged. Its load-bearing conclusions:

- **The frozen Head is an actionable guidance objective** (H1). Head-guided hard-state
  feedback lowers Head-defined immune burden under a structure constraint.
- **The repair-plus-beam package beats the cheap explicit-plus-greedy baseline** on
  `fast_v2` (its overlap with the P3 `pilot_v3` cohort is 3 proteins). The advantage
  transfers to NetMHCIIpan, an **independent evaluator** (not a
  ground-truth oracle) that never entered the loop — the strongest single result, because it
  is the most direct evidence against Head-gaming. The package buys this at ~1.30× the
  refold cost and a modest structural cost that is **not yet shown to be removable**:
  independently evaluated terminal scTM is lower for the package, with more designs close
  to the 0.85 gate, even though every reported terminal elite passes it.
- **Deterministic global selection (beam) beats independent greedy lanes.** Population-level
  selection has value.
- **The stochastic FK selector was not supported.** At face value a negative result about
  population sampling. §2 states precisely what it does and does not establish.

## 2. What the P3 FK negative actually establishes

The first draft argued that the FK negative was fixed in advance by a theorem — that at the
terminal state the reward is fully observed, so deterministic top-$k$ must beat a stochastic
draw, making P3 a comparison of two baselines. **That framing is wrong, and correcting it is
the reason v1 is worth pursuing rather than foreclosed.** (This reverses my own first draft.)

The theorem-style argument governs only a single terminal selection over independent
endpoints. Fusion is not that. It is **six complete-state macro-rounds**, and each round's
selection is written back as the next round's parent population (`runner.py:431`), so
survivors of rounds 1–5 go on to seed further proposals. There is real future propagation.
The reference-implementation routines `bon` (terminal deterministic sort) and `IS` (terminal
multinomial) are a tight analogy only for the **final** round; across the earlier rounds v0's
beam is repeated tree pruning and v0's FK is per-round resampling on a complete-state edit
Markov chain. The "argmax dominates a draw for a finite $\beta$" fact holds for immediate
max over one fixed pool; it does not imply that beam globally dominates FK across a
multi-round search.

So P3 has genuine empirical content. The precise statement is:

> P3 establishes a clean negative for the frozen complete-state macro-FK configuration:
> under $N=4$, six explicit-only rounds, and $\beta:1\rightarrow4$, multinomial FK is
> Pareto-dominated by deterministic beam on terminal Head and refold cost. It does **not**
> determine whether trajectory-level reward allocation is useful before completion.

P3 therefore measured a real cross-round ancestry question, and the frozen FK configuration
failed it. RAR 0033 also closes the implementation escape hatch: the weight lifecycle and
post-resampling reset conform to the intended potential. For v1, this is enough. FK remains a
reported negative ablation; there is no $\beta$, $N$, or cadence rescue program, and the new
trajectory question is tested with deterministic beam.

## 3. The trajectory-track question: does pre-terminal allocation create predictable new basins?

The sharpened question is not "is the partial state where FK finally becomes valid." A
complete-state mutation chain already carries continuation uncertainty — a lineage's future
depends on edits and selections not yet made — so FK is *definable* at $t=T$, which is what
v0 tested. What changes before completion is subtler and more precise:

> At a partial state the reward must be **estimated** from a continuation-value model, which
> is the object the FK literature actually steers on. At a complete state the potential uses
> the exactly-known current reward. The open empirical question is whether reallocating
> search toward promising **partial** roots creates future basins that are both *different*
> from, and more *predictable* than, those reachable by complete-state allocation.

Two forces plausibly bound where in the trajectory this could pay off. To avoid the sign
error in the first draft, they are stated in the **resolved fraction of editable positions**
$\rho_{\mathrm{edit}}$ (project convention: $t=0$ all-mask, $t=1$ clean):

- the **fidelity of a partial-state value estimate rises with $\rho_{\mathrm{edit}}$** — fewer masked
  positions remain to contribute approximation error;
- the **remaining uncertainty that allocation can influence generally falls with
  $\rho_{\mathrm{edit}}$** — fewer editable identities remain unresolved or reopenable.

Hard anchors, fixed motifs, and the conditioning backbone are excluded from this maturity
coordinate. They are not parent seeding: **non-parent** means that no complete sequence
supplies the editable residue identities inherited by Fusion. They do, however, provide known
context from the first denoising step and reduce conditional entropy. A heavily constrained
task is therefore better described as scaffold inpainting than as generic whole-protein
generation, even when every editable position starts masked. For example, the Q00511
safety-max policy fixes 24 of 302 residues while leaving 278 editable; its maturity must be
measured over those 278 positions, and it is a constrained-application transfer rather than
the sole evidence for a generic trajectory claim. The underlying `c1_null` generator already
starts with all editable positions masked; what v0 lacks is reward-facing selection before
its generated sequence is complete.

These are not a proof of an interior optimum. The product of a rising and a falling function
need not be single-peaked; whether a usable pre-terminal window exists is a **hypothesis**, not a
corollary. Its value is that the two forces are **each a measurable curve**, and they are
exactly the two quantities the compact gate measures (§6): fidelity is "can selection see the
reward yet," influence is "is there anything left to steer." The gate is the test of this
hypothesis, not an illustration of it.

## 4. The connection to prior RF work — conceptual, not actuator inheritance

The project's RF stack separates two questions: how generation is parameterized over time and
positions, and which candidate lineages receive future compute. V1-A deliberately tests only
the second question before completion. It does **not** combine a position-dependent residue
field with ancestry selection, and therefore does not exercise the proposal-allocation clause
of the granularity axiom as a joint mechanism.

The earlier h-map-driven, non-null position-dependent actuator did not establish value as an
inference-time steering component. Reintroducing it here would confound the pre-terminal
allocation question and may require a separate training-level treatment rather than another
inference-only schedule. V1-A therefore inherits the frozen complete-sequence Head objective
and the base null generator, but no position-dependent RF actuator. Non-null static C1/h-map,
D2, D3, SC-GR, dynamic schedules, and controller memory are independent deferred tracks.
This matches the frozen v0 repair runtime, which uses `c1_null`, flat `h=1`, and
`controller=None` rather than a precomputed h-map.

## 5. Existing editability and the deferred RERD extension

Vanilla absorbing (masked) diffusion has total commitment: an unmasked token is frozen for
the rest of the trajectory, which is why late intervention is weak and why remasking and
planning methods exist. The base sampler used by V1-A is not strictly absorbing: the
compatibility preset named `c1_null.yaml` retains the existing reparameterized-remask sampler
while setting reward amplification to `constant_one`. It consumes no position-dependent h-map
and is not evidence for the non-null C1 actuator. The complete-state Fusion loop remains the
project-specific RERD-inspired adaptation described in `doc/RF-Refine-Fusion.md`: repeatedly
reopen or edit, denoise or repair, evaluate complete states, and carry selected states
forward. These are distinct layers. RERD is not an old RF component, and the current code is
not a direct or complete implementation of the external RERD algorithm.

If no usable pre-terminal value exists, a better selector cannot manufacture that
information. A fuller RERD-style clean-state re-noising process or explicit planning would
then be a natural future extension. Neither is part of the present v1 program and neither is
mixed into the first trajectory experiment.

## 6. A compact maturity calibration

V1 does not need a separate diagnostic program. It needs one small preflight that chooses a
late $\rho_{\mathrm{edit}}<1$ and checks that continuation-value beam is acting on signal
rather than correlated completion noise. Four conditions are irreducible:

1. different partial roots have repeatably distinguishable continuation value, beyond the
   rollout variance within a root;
2. roots selected from one rollout set outperform random allocation on independent
   continuations;
3. selected-root allocation is not materially worse than a compute-matched independent
   full-trajectory control on the **structure-feasible terminal frontier**;
4. a frozen minimum unresolved editable mass, descendant branching, and multiple terminal
   basins show that genuine action remains at the handoff.

Every T0 policy uses the same all-editable-mask `c1_null` trajectory,
`controller=None`, and no `H_MAPS_PARQUET`. The frozen Head evaluates only complete
continuations. Its scores estimate continuation value; they are never converted into a
position-dependent field or written back into denoising. Selected and random allocation are
membership views over the same partial-root pool and held-out evaluation table, not separate
generation draws.

The detailed maturity grid, rollout split, reliability statistics, seed contract, and refold
allocation belong in the runbook. The science-level decision is simple: calibrate one late
$\rho_{\mathrm{edit}}$ on a development cohort, freeze it, and validate it once. If no tested
window passes, the all-mask terminal generator remains valid but v1 makes no trajectory-level
claim; it does not keep tuning maturity, estimator, or selector until a positive appears.

## 7. Feasibility and hazards, at design altitude

For **branch-and-materialize** the primitives exist and have run in production, but that
terminal diagnostic is outside the active V1-A implementation. For **partial-root
continuation-value allocation** the machinery is not ready: `ResumeState` does not restore
the RNG stream, the batched sampler cannot resume, and the inherited-state contract
hard-requires a complete AA20 sequence with a definitive structure verdict (`state.py:77`).
A partial population is a new type and a new contract, not a relaxation of the old one.
Because `controller=None` is frozen, controller memory is neither inherited nor an
implementation prerequisite.

Three hazards are prerequisites rather than features, each able to silently corrupt a
partial-state result:

- **The frozen Head is fail-open on non-canonical input.** A mask or invalid residue is
  silently mapped to an unknown token and returns a plausible finite number. Head therefore
  scores only complete rollout sequences, guarded by a strict AA20 boundary; it never scores
  a masked root directly. Complete-sequence scoring supplies continuation value only; it does
  not create an h-map or otherwise actuate the generation path.
- **The "cheap surrogate" is not the partial-feasibility model.** The v0 two-tier structure
  contract's cheap tier is an unimplemented complete-sequence screen. A $\rho<1$ feasibility
  screen scores a *partial continuation's* fold prospects — a different model. Neither exists;
  conflating them (as the first draft did in calling the surrogate a $t<T$ prerequisite) hides
  a second build.
- **Seed degeneracy imitates population collapse.** Forked continuations that accidentally
  reuse the same root/seed pair return identically. Seed identities must be explicit and
  collisions reported; different seeds are not required to produce different sequences,
  because genuine low-entropy convergence is itself measurable.

The deepest hazard is scientific: selecting on a partial-state estimate steers the population
toward *sequences whose greedy completion looks safe*, which is not the target. That bias is
what §6 measures, not an implementation detail.

## 8. The paper-critical v1 program

This document specifies the trajectory track: whether Fusion can allocate search before
completion. The paper-critical dual-allele track is reserved in §8.2 and has no contract here.

### 8.1 V1-A — generative trajectory Fusion

The v0 NoD parents were already generated from full mask. The missing capability is therefore
not non-parent initialization; it is **reward-facing entry before a complete sequence becomes
the Fusion parent**. P1 has only two entry arms:

1. **Terminal** — independent `c1_null` trajectories are completed, scored by the frozen Head,
   and selected as complete parents under the matched entry budget;
2. **Pre-terminal** — the same `c1_null`, `controller=None`, no-h-map trajectory is intercepted
   at a calibrated $\rho_{\mathrm{edit}}<1$; continuation-value beam selects and propagates
   partial roots, which are then freshly completed into parents.

Both complete parent facades enter the same frozen v0 repair+beam package. The contrast tests
pre-terminal reward-facing population allocation on one common null generator. It is still a
matched-compute system contrast rather than a clock-time-only intervention, because
partial-root allocation changes candidate correlation, value precision, and the inherited
object. It does not estimate a schedule contribution or a schedule-by-handoff interaction.

The pre-terminal method need not strictly beat the terminal reference on every endpoint. It
must either improve the matched immune--structure--cost frontier, or satisfy a pre-declared
practical non-inferiority margin while demonstrating genuine pre-terminal action: unresolved
editable mass remains, descendants still branch, and selection changes
which complete basins receive downstream budget. Small differences are reported against the
margin, not ignored after seeing the result. A clear material regression falsifies the main
trajectory claim rather than triggering indefinite tuning toward $\rho_{\mathrm{edit}}=1$.

### 8.2 V1-B — dual-allele objective Fusion

**High-priority reserved track.** Dual-allele capability is paper-critical. Scientific and
experimental content is intentionally blank pending a separate user-directed design.

### 8.3 Supporting v0 closure

Reconcile the independent-refold structural tail and fix the canonical-repeat / fallback
path. A structure-budget replay on the existing candidate pool is fair game, but no further
large repair-mechanism experiment is part of v1.

## 9. Supporting and deferred directions

- **Structure gate as absolute floor + cumulative-loss budget.** The observation that Head
  change and scTM change were uncorrelated makes a relative gate *plausible*, not proven (the
  9-vs-1 independent-refold tail argues against "free"). A pure relative gate also lets drift
  walk down over rounds; the safer form is an absolute floor plus a fixed-start cumulative
  structural-loss budget, validated first by a candidate-level Pareto replay.
- **Cheap surrogate as a cost optimization, not a maturity-calibration prerequisite.** It attacks the dominant
  refold cost of the complete-state loop, but per §7 it is a different model from the
  partial-feasibility screen and is not on the current critical path.
- **Diversity-floor beam, downstream of a proven pruning-regret.** Motivated only if beam is
  shown to prune eventual winners; it must not be upgraded from a single outlier (`5D7H_B`)
  and must be pre-registered on a fresh cohort.
- **Older RF actuators.** D2, D3, SC-GR, and dynamic schedules are deferred. D1 may be
  retained as telemetry but has no proposal or selection authority.
- **Position-dependent generation and additional editability methods.** Non-null static
  C1/position-dependent h-map steering is excluded from V1-A because its value was not
  established and a credible revisit may require retraining. A fuller RERD-style clean-state
  re-noising process and explicit planning are also deferred extensions, not old RF
  components.
- **FK.** The frozen v0 configuration remains a reported negative ablation. FK is not tuned,
  repaired, or re-tested in v1.
- **Function and activity beyond global scTM**, on a cohort kept separate from the method
  cohort.
- **A standing guardrail, not an extension**: broader search intensifies Head exploitation,
  so Head→NMP transfer, OOD/naturalness, and selection-induced error must be reported
  continuously, not validated once.

## 10. Hypotheses for v1

- **HT1 (a usable late pre-terminal window exists).** At some calibrated
  $\rho_{\mathrm{edit}}<1$, partial roots have reproducibly distinguishable continuation value
  and retain non-trivial editable basin choice. *Falsifier*: selected roots do not outperform
  random allocation on independent continuations, or the apparent dispersion is within-root
  noise.
- **HT2 (pre-terminal allocation has practical value).** On the shared `c1_null` substrate,
  pre-terminal beam improves the matched structure-feasible terminal frontier over terminal
  entry after both pass through the same v0 repair+beam package, or is practically
  non-inferior while delivering a verified pre-terminal generative path.
  *Falsifier*: material immune, structure, or cost regression under the frozen margin, with no
  compensating reachability or efficiency gain.

The frozen v0 H6 configuration is negative. H4/H5-style marginal actuation and editability
questions remain future work rather than active v1 hypotheses.

## 11. What is frozen and what is open

Frozen:

- The trajectory track uses a frozen Head evaluated only on complete sequences; NMP stays
  outside the loop as an independent post-hoc evaluator.
- V1-A estimates each root by the arithmetic mean of a frozen `K_EST` set of complete-sequence
  Head scores and evaluates selected/random roots on a disjoint common `K_EVAL` table.
- The v1 trajectory starts from the existing full-mask conditional generator and uses
  `c1_null`, `controller=None`, and no `H_MAPS_PARQUET` for both entry arms. The frozen Head
  scores complete sequences only and never actuates denoising. Maturity is measured over
  editable positions; hard anchors and backbone conditioning are not parent seeding.
- Deterministic beam is the selector. FK, D2, D3, and SC-GR are outside the active v1 program.
- Non-null static C1/position-dependent h-map steering is a separate future extension and is
  outside this coder and experiment scope.
- The v0 inheritance contract (complete AA20 state, definitive structure verdict) remains the
  complete-state contract. Partial-root propagation requires the new explicitly typed contract
  described in §0/§7 and proceeds only after the compact calibration in §6.

Open:

- the late maturity $\rho_{\mathrm{edit}}$;
- the practical non-inferiority margin and matched compute accounting, to be frozen in the
  runbook;
- the partial-continuation feasibility treatment and definitive refold cadence.

## 12. References and project evidence

External work surfaced by the v1 literature review (the four marked † carry full entries in
`doc/RF-Refine-Fusion.md` §14):

1. Cappé, Godsill & Moulines. An overview of existing methods and recent advances in
   sequential Monte Carlo. *Proc. IEEE*, 2007. — resampling conditions only what is
   propagated afterwards; in a multi-round setting there is an afterwards.
2. Wu et al. Practical and asymptotically exact conditional sampling in diffusion models
   (Twisted Diffusion Sampler), NeurIPS 2023. — exact tilt weights at the terminal; the ideal
   intermediate potential is the continuation-reward distribution, approximated by many-sample
   or learned estimators.
3. Singhal et al. A general framework for inference-time scaling and steering of diffusion
   models, 2025†. — `bon`/`IS` are terminal baselines; difference/max/sum potentials.
4. Hartman et al. Controllable protein design through Feynman-Kac steering, 2025†. —
   late-onset guidance and particle count are reported as task-specific findings, not laws.
5. Uehara et al. Reward-guided iterative refinement at test time (RERD), 2025†. — soft value
   $v_t(x_t)=\alpha\log\mathbb{E}[\exp(r(x_0)/\alpha)\mid x_t]$; clean-state re-noising over
   many rounds; global resampling only at the final step.
6. Editability in masked diffusion: ReMDM (Wang et al.); path-planning on DPLM-650M
   (Peng et al.); DDPD re-opening committed tokens (Liu et al.).

Project evidence:

7. RAR 0019 — candidate-level cheap/conditioned signal vs K=8 oracle (correlation present,
   capture low), `../Results/Analysis/0019-candidate-level-signal-direction-v2-pair/record.md`.
8. RAR 0025 — SC-GR where/value ablations (allocation and terminal rollout, limited signal),
   `../Results/Analysis/0025-sc-gr-v1-1-where-value-ablations-triage-/record.md`.
9. RAR 0030 — P3 selector comparison (beam > greedy; frozen-config FK not supported),
   `../Results/Analysis/0030-rf-fusion-p3-h6-selector-comparison/record.md`.
10. RAR 0031 — final integration on `fast_v2` (package vs baseline; NMP transfer; independent
    -refold structural tail), `../Results/Analysis/0031-rf-fusion-final-integration-fastv2-immune-structure/record.md`.
11. RAR 0033 — FK beta sensitivity and weight-lifecycle conformance audit (supersedes 0032;
    the FK negative is configuration, not implementation),
    `../Results/Analysis/0033-rf-fusion-p3-fk-beta-sensitivity-and-fk-/record.md`.
12. `doc/RF-Refine-Fusion.md` — the v0 method, frozen decisions, and evidence ledger.
