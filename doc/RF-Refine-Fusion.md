# RF-Refine Fusion: Reward-Selected Edit-and-Repair Reference Flow

**Status**: scientific design document. This document defines the target method, its evidence base, and its scientific boundaries. It is not an implementation plan, experiment matrix, or configuration contract.

**Related documents**: `doc/Reference_Flow_Derivation.md`, `doc/RF_Controller_Architecture.md`, `doc/Self-Cond_GR.md`, and `PLAN_RF_REFINE.md`.

## 1. Executive statement

The current Reference Flow (RF) controller and the standalone refinement pipeline expose two complementary control laws.

The RF stack acts inside token generation. It observes a hard completion, evaluates local Head risk, converts block-level candidate preferences into residue-level logit shifts, samples tokens, and then uses global remasking and D3 rank scores to decide which residues survive. This stack can move the terminal immune distribution, but the exact hard candidate evaluated by the Head is not preserved. Its effect must survive marginal projection, independent token sampling, and later remasking.

The standalone refiner acts on complete sequences. It searches a broad mutation neighborhood, evaluates hard candidates, keeps selected complete states, and repeats from those states. The NMP-driven run is a mechanism control: it shows that, when the iterative search receives a usable reward, broad hard-state search and feedback can drive that reward strongly while preserving high global scTM in the selected outputs. NMP is not the intended runtime objective and Head-NMP agreement is not a requirement of the proposed method.

The proposed fusion is **Reward-Selected Edit-and-Repair Reference Flow**. The frozen Head is the runtime guidance objective. RERD supplies the repeated move-evaluate-feedback law, while Feynman-Kac (FK) selection supplies a population-level mechanism for reallocating descendants toward lower-Head basins without requiring reward gradients. Its core transition is

```text
complete RF parent population
    -> select a Head-responsible register
    -> generate broad hard edits and/or targeted RF reopen proposals
    -> conditionally repair local context with RF/DPLM
    -> evaluate the complete repaired children
    -> apply a state-level structure constraint
    -> Head-potential weighting and whole-state selection/resampling
    -> inherit exact selected children as the next parents
    -> repeat
```

The defining feedback is not a logit correction and not a remask rank. It is **exact evaluated-state carryover**: a sequence that passed Head and structure evaluation becomes a parent of the next structure-conditioned proposal cycle. The population may be selected greedily, by a beam, or through FK-style weighting and ancestry resampling; these are alternative selection laws over the same hard-state move process.

This changes the role of the structure model. Structural logits remain a useful proposal prior, but they are not treated as the fold-feasible set. Broad immune-directed edits may leave the model's high-probability local support; RF/DPLM then acts as a context repair operator, and a complete-state structure gate decides whether the resulting sequence remains acceptable. The method therefore uses structure to **propose, repair, and veto**, rather than to preemptively remove most of the immune search space.

NetMHCIIpan (NMP) is excluded from runtime targeting, proposal, selection, repair, acceptance, and stopping. It may be reported after generation as an external comparison or terminal filter, but success of the guidance method is defined in Head and structure space rather than by reproducing NMP.

## 2. What the current evidence establishes

### 2.1 The B1 lifecycle changes the terminal distribution

The corrected RAR 0013 comparison uses a diverse matched NoD baseline on a selected 100-protein high-risk stress cohort. The complete B1 lifecycle lowers Head `global_risk` by a mean of 1.217 nats under mean-of-eight aggregation, with 75 of 100 proteins improved.

This establishes that the combined typed-actionability, D2, D3, sampling, and remask lifecycle is not inert. It does **not** identify D2 as the sole cause, and it does not show that every useful D2 candidate survives to the terminal sequence.

Current D2 has the following scientific semantics:

1. Construct hard block tuples in a narrow structural-logit support.
2. Score those tuples on a hard completion, primarily with a local window objective.
3. Form a candidate distribution and project it to per-position token marginals.
4. Write marginal logit shifts into the sampler.
5. Rely on categorical sampling, sticky evidence, D3 ranking, and remasking for delivery and persistence.

The evaluated tuple is not the state that is carried forward. This is the principal control mismatch that the fusion must remove.

### 2.2 NMP-driven refinement validates the search law, not the target objective

In the char24/DRB1*07:01 refinement run, 99 of 123 seeds reach zero distinct NMP strong-binding cores. The mean best-entry scTM is 0.929, 88.6% of best entries retain scTM at or above 0.9, and successful seeds require a median of five edits.

This is strong evidence that many RF seeds have a broad nearby neighborhood in which reward improvement and high global scTM can coexist after selection. It is not evidence that arbitrary mutations are safe. The search refolds only a small selected subset of the much larger mutation pool, freezes specified anchors, and currently validates global scTM rather than active-site geometry or function.

The correct interpretation is therefore:

> The refinement run demonstrates that the **iterative hard-state search law has authority** when supplied with a strong reward, and that the selected neighborhood can remain structure-compatible.

The experiment was not intended to establish that Head reproduces NMP. The paper objective is to use the Head as the design-time immune model. The scientific transfer is therefore from one reward-guided search law to another: replace the NMP reward with the frozen Head while preserving broad proposal, complete-state evaluation, repeated feedback, and state-level structural validation.

The run also reveals a useful separation of roles. The successful refiner does not require every proposed mutation to look structure-native before evaluation. It creates broad candidates, uses an immune objective to reduce the pool, and applies an explicit structure check to selected complete states. This separation is the key architectural lesson for RF-Refine Fusion.

### 2.3 Breadth and feedback are the supported levers

The Path-C diagnostics distinguish broad search from coordinated denoising. Increasing candidate breadth and applying best-of-$K$ selection produced substantial headroom, while multi-step reconditioned joint sampling did not show a clear advantage over independent register proposals. The evidence therefore supports:

- broad candidate support;
- complete hard-sequence evaluation;
- repeated selection from the selected state;
- exact carryover of the evaluated sequence.

It does not currently show that correlated multi-step coordination is the source of the gain. This does not remove RERD or FK from the target architecture: RERD organizes how search is repeatedly reopened, and FK organizes how Head value reallocates a population. The unsupported claim is narrower: neither correlated denoising nor a particular exact FK estimator has yet been shown to be the empirical lever.

### 2.4 Global remask is both useful and misaligned

The present reparameterized remask is a deterministic global bottom-$k$ operation over committed residues, and it is active only in remask-enabled configurations (off by default), so the durability observations in this section are scoped to those runs. A controller rank can change its score, but neither the rank nor the remask creates and evaluates a local hard child. Remask can provide repeated proposals and structural stabilization, so removing all re-noising is not the scientific conclusion.

At the same time, current remask is misaligned with reward-selected edits. It can reopen a useful D2 token without asking whether the resulting complete sequence is better, and it can dismantle a multi-residue candidate after that candidate was evaluated. The no-remask diagnostics show that remask strongly changes the terminal substrate, but also that merely disabling it does not make the RF controller the dominant source of improvement.

The lesson is precise:

> Re-noising is valuable; **uncontrolled global re-noising is not a valid reward acceptance rule**.

The fused method retains structural noising/denoising as a proposal and repair mechanism, while replacing implicit residue survival with explicit complete-state evaluation and acceptance.

## 3. The central design correction

### 3.1 Structural confidence is not structural feasibility

The current D2 support uses per-position structural logits, `top_k_tokens`, and `delta_struct` before candidate tuples are formed. These quantities express local model preference under the current context. They are useful proposal statistics, but they do not establish that a complete mutated sequence will or will not preserve the target fold.

This distinction matters because inverse-folding posteriors may be sharply peaked even when multiple lower-probability amino-acid combinations remain fold-compatible. A low local token probability can mean "not the model's preferred residue," not "structurally impossible."

RF-Refine Fusion therefore separates two objects:

$$
q_{\theta}(y\mid x,S)
\quad \text{as a structure-conditioned proposal prior,}
$$

and

$$
F_S(y)
\quad \text{as a complete-state structural feasibility functional.}
$$

Local structural confidence may rank proposals or allocate compute. It must not be treated as the final feasibility boundary. The final boundary acts on the complete candidate after mutation and repair.

### 3.2 Search first, repair second, veto last

The proposed structure-immune interaction is not "ignore structure until the end." It is a three-stage contract:

1. **Challenge** the narrow model support with broad immune-directed hard edits.
2. **Repair** the surrounding sequence under the target backbone using RF/DPLM conditional generation.
3. **Veto** complete candidates that fail the state-level structure constraint.

This ordering gives the immune objective access to space that a strict token-level support would remove, while still requiring the selected terminal state to remain structure-compatible.

### 3.3 Exact feedback, not marginal reconstruction

The method does not require coordinated mutations to be the main source of headroom. Exact carryover is required for a simpler reason: the next proposal must be conditioned on the same complete state that was evaluated and selected.

If a candidate is scored as sequence $y$, then the next parent is $y$. The method must not replace $y$ with independent marginals and hope that later sampling reconstructs it. Any later modification of an accepted region is a new proposal that must be evaluated again.

### 3.4 Residue fields allocate; complete states select

The project's negative and positive results share one root. The SC-GR residue field succeeded as sensing and failed as a selection actuator; D2 marginal projection moved the terminal distribution but leaked its evaluated candidate through independent sampling and remask; D3 compressed a complete-state Head reading into a per-residue survival rank. In each case a residue-level field was asked to perform a job that belongs to a complete state.

The correction is a single granularity rule:

> Residue-level fields allocate proposals; complete-state Head values select ancestry.

A residue field may nominate where to search and how to bias a proposal. It must not decide whether a complete candidate survives. Selection, acceptance, and inheritance act on the complete evaluated state. This rule subsumes 3.3 and reinterprets the failed direct-actuation results (RAR 0025): they are not tuning failures but granularity errors, a residue-level field applied at selection granularity.

## 4. Scientific objective

Let $S$ be the target backbone and $x$ a complete amino-acid sequence. Let $R_H(x)$ be a frozen-Head whole-sequence immune burden, with lower values preferred. Let $F_S(x)$ be a state-level structure functional relative to $S$.

$R_H$ is the design objective, not a proxy whose success is conditional on reproducing NMP. Its scientific validity comes from the Head training and evaluation program; the role of the fusion method is to make that learned objective actionable during generation.

The same Head landscape is consumed at three distinct levels:

- **responsibility**: nominate registers where search should be concentrated;
- **state value**: rank complete hard sequences by global immune burden and new-hotspot behavior;
- **incremental potential**: determine which parent-child transitions receive more FK ancestry and future proposal budget.

This is the guidance mechanism. Logit steering may improve proposal efficiency, but it is not the only route by which Head information influences generation.

The primary objective is constrained optimization:

$$
x^* \in \arg\min_x R_H(x)
\quad \text{subject to} \quad
F_S(x) \geq \tau_S.
$$

The Head objective must be global even when a local register generated the proposal. Local responsibility identifies **where to search**; the full Head landscape decides **whether the resulting sequence is better**. This prevents a target-window improvement from being accepted when it creates a larger hotspot elsewhere.

For a parent $x_r$ and candidate $y$, define a new-hotspot burden $N_H(y;x_r)$. A conservative acceptance set is

$$
\mathcal A_r =
\left\{
y:
F_S(y) \geq \tau_S,
\ R_H(y) \leq R_H(x_r)-\epsilon_H,
\ N_H(y;x_r) \leq \delta_{\mathrm{new}}
\right\}.
$$

The parent is always retained as a null move. Feasibility ($F_S(y)\geq\tau_S$ and $N_H(y;x_r)\leq\delta_{\mathrm{new}}$) is the hard admissibility every selector enforces; the Head margin $\epsilon_H$ defines the conservative greedy/beam control, whereas the population method (§5.6) treats Head as a soft objective permitting bounded uphill exploration, with an elite archive of the best-so-far feasible state guaranteeing the returned design does not degrade. Uncertainty handling for $R_H$, $\epsilon_H$, and the structure gate is later calibration; the fixed semantics are that feasibility is the hard constraint and Head improvement is measured on a complete sequence.

## 5. Core method: hard-state edit-and-repair cycles

### 5.1 State and correction round

The scientific state at correction round $r$ is a population of complete hard sequences

$$
\mathcal P_r
=
\left\{
(x_r^{(n)},w_r^{(n)},a_r^{(n)})
\right\}_{n=1}^{N},
$$

where $w_r^{(n)}$ is an optional FK weight and $a_r^{(n)}$ records ancestry. A single parent or hard beam is a valid limiting case, but the population form is retained because it permits Head-guided branching, diversity, and resampling across correction rounds. The initial parents are produced by the structure-conditioned RF generator. The core method does not require a cloneable partial RF trajectory or a hidden diffusion state.

This choice is not merely an implementation convenience. The Head is most interpretable on complete sequences, the DPLM runtime is conditioned on the target backbone and current token context, and the selected complete state is exactly the information needed for the next local conditional proposal.

### 5.2 Target: select a responsible register

The Head produces a whole-sequence window landscape. A responsibility operator selects one or more registers $B_r$ that contribute to current burden. Region selection is a proposal-allocation decision, not an acceptance decision.

Existing D1 actionability, Head window prominence, and trajectory memory may help nominate regions. However, the failed direct SC-GR allocation results mean that prospective residue fields should not be assumed to improve outcomes merely by multiplying or replacing `v_target`. The minimal target rule should remain tied to current complete-sequence Head responsibility until a prospective targeting rule proves incremental value.

### 5.3 Challenge kernel: generate broad hard edits

For parent $x_r$ and register $B_r$, a broad edit kernel produces complete edited states

$$
z \sim K_{\mathrm{edit}}(\cdot\mid x_r,B_r).
$$

The support may contain AA20 single substitutions, selected pairs, capped multi-position edits, or stochastic mutations. Structural logits may prioritize this pool, but they do not define a hard token whitelist. An unguided or weakly guided proposal channel must remain available so that the search can challenge the denoiser's local mode.

Explicit mutation is a first-class proposal family, not a fallback used only when denoising fails. The refinement and breadth diagnostics make it the strongest currently supported source of search coverage.

### 5.4 RERD-style move and RF repair kernel

Each correction round applies a RERD-style move: reopen local sequence space, produce complete hard descendants, evaluate them, and feed selected descendants into the next round. In this project, reopening is targeted to Head-responsible registers rather than uniformly random, and the denoiser is conditioned on the fixed backbone.

An edited state $z$ may be structurally acceptable already. It may also require nearby compensatory changes that are difficult to enumerate manually. RF/DPLM is used to repair this context under the fixed backbone.

Let $A_B$ be the immune-directed edited positions and let $H_B$ be a local repair halo containing $A_B$. A protected-core repair keeps the proposed edits and the sequence outside the halo fixed, reopens the remaining halo positions, and conditionally regenerates them:

$$
y \sim
K_{\mathrm{repair}}
\left(
\cdot
\mid z,S,A_B,H_B
\right).
$$

This is the main fusion point. The immune edit challenges the current local mode; the backbone-conditioned denoiser is then allowed to reorganize nearby context so that the edited sequence can return toward a structure-consistent basin.

The candidate set may include both the direct edit $z$ and one or more repaired descendants $y$. If repair simply reverts or neutralizes the immune gain, the complete Head evaluation rejects that descendant. If the edit cannot be repaired without violating structure, the structure gate rejects it. No local logit score is asked to predict either outcome in advance.

A direct RF-reopen kernel, which remasks the register without an explicit mutation challenge, is the second main move family. The complete move kernel may combine direct targeted denoising with edit-then-repair:

$$
K_{\mathrm{move}}
=
\omega K_{\mathrm{RF\text{-}reopen}}
+
(1-\omega)
K_{\mathrm{repair}}\circ K_{\mathrm{edit}}.
$$

Current evidence does not show that multi-step reconditioned sampling has a unique advantage, so direct RF reopen should not replace explicit breadth. Conversely, explicit mutation should not remove the RF repair branch, because repair is how the fixed backbone can reshape context around a reward-directed edit.

### 5.5 Complete-state evaluation and constrained selection

Every ancestry-eligible candidate is evaluated as the exact complete sequence that may be inherited. The evaluator returns:

- whole-sequence Head burden;
- local target improvement;
- new-hotspot burden outside the target;
- state-level structural feasibility;
- optional diversity and movement diagnostics.

The selector retains the parent as a null move. The greedy and hard-beam controls require Head improvement; the FK population method requires only structure and no-new-hotspot feasibility, treating Head as a soft potential that permits bounded uphill exploration, with an elite archive of the best-so-far feasible state so the returned design never degrades. The FK potential preserves multiple promising basins and reallocates future proposal budget.

### 5.6 Head-guided Feynman-Kac selection

For a parent $x_r^{(a)}$ and feasible child $y_r^{(n)}$, the population potential targets the absolute Head level rather than the per-step increment. Its annealed form is

$$
G_r^{(n)}
=
\mathbf 1\!\left[F_S\!\left(y_r^{(n)}\right)\geq\tau_S\right]
\mathbf 1\!\left[N_H\!\left(y_r^{(n)};x_r^{(a)}\right)\leq\delta_{\mathrm{new}}\right]
\exp\!\left[-\beta_r R_H\!\left(y_r^{(n)}\right)+\beta_{r-1}R_H\!\left(x_r^{(a)}\right)\right],
$$

with initial weights $w_0^{(n)}\propto\exp[-\beta_0 R_H(x_0^{(n)})]$. Telescoping along ancestry makes the cumulative weight track $\exp[-\beta_r R_H(x_r)]$, so the population concentrates on absolute low-Head feasible states; at fixed $\beta$ it degenerates to a difference potential $\exp[-\beta(R_H(y_r^{(n)})-R_H(x_r^{(a)}))]$. The schedule $\beta_r$ is deferred calibration (§12).

The population weights update as

$$
\widetilde w_r^{(n)}
\propto
w_{r-1}^{(a)}G_r^{(n)}.
$$

Resampling duplicates promising complete children and prunes poor ancestry while retaining a diversity mechanism or elite parent. FK therefore answers **which Head-favorable hard states receive the next round of RF proposals**. It does not generate candidates; $K_{\mathrm{move}}$ does.

This macro-state FK law is part of the target guidance architecture. The method's positive claim is **Head-guided FK optimization**: population reweighting and ancestry resampling that concentrate mass on structure-feasible, low-Head states, not exact posterior sampling from the original RF path measure. An exact-posterior claim would additionally require normalized proposal kernels, proposal-ratio accounting, and a fully specified resampling law; under the optimization claim the D2 proposal is a valid heuristic that owes no proposal-ratio correction, and such a correction becomes obligatory only if the exact-posterior claim is later made. Hard beam and no-resampling populations remain essential controls for isolating the value of FK interaction.

The population weighting is closed by a fixed contract: each parent draws the same child budget, and if budgets differ the child weights are divided by the parent's offspring count before combination, so multiplicity does not buy ancestry mass; an infeasible child (failing the structure or no-new-hotspot gate) receives zero weight; on a non-resampling round the normalized child weights carry to the next round, and on a resampling round the resampled population resets to uniform $1/N$; because the feasible parent is always retained as a null move, an all-infeasible round keeps the parents rather than falling back to a uniform distribution over infeasible children.

### 5.7 Exact selected-state feedback

Selected children become the next parents:

$$
x_{r+1}^{(n)}
\in
\operatorname{Select}
\left(
\mathcal P_r
\cup
K_{\mathrm{move}}(\mathcal P_r,B_r)
\right).
$$

The next register selection and the next RF move are conditioned on the exact selected states $x_{r+1}^{(n)}$. This repeated conditional feedback is what makes the procedure a fused RF generator rather than a terminal filter.

## 6. Remask and iterative refinement

### 6.1 What should be retained from remask

The initial RF generator may retain global reparameterized remasking because it supplies repeated structure-conditioned proposals and helps resolve uncertain residues while the sequence is forming. Remask is one form of noising followed by denoising; iterative refinement does not replace it merely by using different terminology.

### 6.2 What must change in the reward-facing phase

Once a complete child has been accepted by Head and structure evaluation, generic global remask must not silently dismantle it. In the hard-state correction phase, reopening is explicit and targeted:

```text
choose block
    -> reopen block or repair halo
    -> generate complete children
    -> evaluate children
    -> accept exact child or keep parent
```

The difference from current remask is therefore not simply local versus global. It is **evaluated transition versus unevaluated residue survival**.

This is the project-specific RERD adaptation. The useful idea is not the external implementation's exact random-remask schedule; it is the repeated cycle in which noising creates a new proposal neighborhood and reward evaluation determines which denoised complete states survive. The Head supplies that reward here, and FK or beam selection carries the surviving states forward.

Interleaving hard-state correction with an unfinished RF trajectory is a deferred variant; the frozen formulation is the permanent post-$t_\star$ handoff described next.

The reward-facing phase is not a single terminal polish. It is a **maturity-triggered handoff**. Before a maturity point $t_\star$, the base RF generator runs with structural-confidence remask and no Head-controlled selection, because the Head value of an early hard completion is not yet reliable. At $t_\star$ the state is materialized as a complete hard-particle population, and every subsequent reopen belongs to the edit-and-repair loop under Head potential, structure gate, and FK resampling. The correction loop is therefore the **reward-eligible phase substrate** over $[t_\star, T]$: a repeated, multi-round process, not a one-time step, so in-generation structural co-adaptation accrues across the whole phase rather than at a terminal instant. The current code does not implement this handoff: a `t_start` gate stops immune refresh and logit correction, but there is no particle materialization, no old-state termination, and no permanent phase switch, and in remask-enabled modes the sampler still remasks after `t_start`; the handoff is net-new. It must not inherit low-level controller state — sampled scores, unmask history, sticky or pending D2 evidence, D3 memory, or pressure state — into particle ancestry; such state may feed register nomination or telemetry only. The materialization itself is one base RF completion of $x_{t_\star}$ into $N$ complete parents (§5.1), not a per-round partial-state value rollout; the latter remains deferred (§12).

**On the value of $t_\star$ (clarification).** Moving $t_\star$ earlier only changes outcomes if the post-$t_\star$ generation is itself reward-guided: with pure base completion, materializing at $t_\star < T$ and completing is distributionally the same as $t_\star = T$. That reward-guided completion of the trajectory is exactly the per-round partial-state rollout deferred in §12. Hence *that* a handoff exists is frozen, but $t_\star$'s value stays an open empirical question (H5), and v0 fixes $t_\star = T$ (so it reuses the complete base generation directly).

### 6.3 D3 is not the move kernel

D3 currently supplies residue-level revisit pressure and rank scores. It neither generates children nor evaluates a parent-child transition. After $t_\star$ it is absent from the main actuator, remaining only as telemetry or an ablation baseline. It is not the reward corrector and should not define the inheritance of an accepted sequence.

The hard-state edit-and-repair operator is a new scientific control law. It bypasses the sampler-level remask and the survival ranks (decomposed in §8.3) as the late reward-facing transition, while allowing early structural remask to retain its generative role.

## 7. The structure contract

### 7.1 Proposal prior

RF/DPLM supplies backbone-conditioned sequence preference. It can prioritize likely amino acids, sample repair contexts, and provide cheap model-consistency diagnostics. This improves proposal efficiency.

### 7.2 State-level gate

The actual structure contract is two-tier. A cheap, configurable surrogate (for example a fold-confidence statistic) screens every raw proposal and supplies the feasibility indicator inside the FK potential. The definitive gate is calibrated against refold-based structure metrics such as scTM to the target backbone, and every candidate eligible to enter ancestry must pass it; a proposal that passes only the cheap screen is not inheritable until the definitive gate confirms it. Which cheap surrogate is used is an empirical choice left open. Recovery is a diversity diagnostic, not a structure objective.

The current refinement evidence validates global scTM only. It does not establish active-site geometry or biochemical function because active-site RMSD was not populated in the reported run. The method must not equate preserved global fold with preserved activity until that surface is measured.

### 7.3 Why the gate follows repair

Applying a hard structure gate before the repair step would recreate the narrow-support failure: potentially repairable immune edits would be removed before the structure model can adapt their context. The scientifically relevant candidate is the complete repaired child, not the isolated low-probability token.

## 8. Relationship to the existing RF components

### 8.1 D1 and Head landscapes

D1 provides useful complete-sequence Head observations and window responsibility. Its role in the fused method is to nominate registers and report how the landscape changes across correction rounds. Static sequence priors are not required for runtime target identity, although a calibrated background reference may still be needed to define prominence.

### 8.2 D2

D2 becomes an optional proposal-efficiency mechanism. It may pre-rank mutations, bias a direct RF-reopen proposal, or enrich the candidate pool with locally favorable tokens. It is judged by a concrete question:

> Under the same hard-candidate, Head, and structure-evaluation budget, does D2 increase the probability of proposing at least one feasible improving child?

If not, D2 is not required by the fused method. Marginal projection, sticky delivery, and exact proposal-ratio correction are not part of the core identity.

### 8.3 D3 and global remask

D3 may contribute revisit memory or target nomination. Global remask remains part of the initial structural generator. Neither is allowed to overwrite an accepted hard child without a new complete-state evaluation.

The residue-survival behavior the fusion displaces is not one object called D3. It is three distinct objects: the sampler's global bottom-$k$ remask transition; the four-term Stage-A survival rank used in D2 modes, which overrides the true D3 score when active; and the two-term D3 revisit rank that adds an immune-memory term only on certain refresh steps. The post-$t_\star$ macro-loop bypasses all three — targeted reopen replaces the global remask transition, and FK whole-state resampling replaces the combined residue-survival surrogate they jointly form. That surrogate can be read intuitively as an FK-like credit that compresses complete-state Head evidence into a per-residue rank instead of carrying the complete state forward, but this is an analogy, not a formal identity, since none of the three maintains a particle population, potential weighting, or ancestry resampling. D3 itself is retained only as telemetry or an ablation baseline, never in inheritance. Early structural remask, before $t_\star$ where the Head penalty is inactive, retains its generative role.

### 8.4 SC-GR

SC-GR remains a sensing result. Coarse burden estimates may allocate compute, beam width, or the number of correction rounds. Previous direct global-pressure and residue-allocation actuators did not establish outcome gains, so SC-GR is not a required control dependency. In particular, the fusion should not reintroduce a failed `r_i`-to-selection multiplication under a new name.

### 8.5 NMP

The current NMP-driven refiner is a mechanism demonstration for iterative reward search and an accessible-neighborhood reference. The proposed fused method uses the Head instead. NMP is neither the optimization target nor an agreement gate for Head guidance. If retained, it is applied only after the final Head-driven correction state is produced as an external comparison or terminal filter.

## 9. Why this is fusion rather than detached post-processing

The scientific boundary is not whether correction happens before the software variable `t` reaches one. The DPLM runtime does not carry a hidden continuous-time state that must be preserved for the method to remain RF-based.

The relevant distinction is feedback:

- A detached filter scores terminal sequences and stops.
- A conventional post-hoc mutator proposes edits without returning to the structure-conditioned generator.
- RF-Refine Fusion accepts a complete child and uses that exact child as the context for another backbone-conditioned RF repair/proposal cycle.

Chronologically, the first correction round may begin after an initial RF sequence is complete. Scientifically, the generator continues because RF conditional proposals and reward selection alternate over multiple macro-transitions. This is a composite generative process, not merely selection of the initial RF output.

## 10. RERD and Feynman-Kac as organizing principles

### 10.1 RERD defines the move-evaluate-feedback loop

RERD contributes the central iterative law:

```text
selected states
    -> re-noise or mutate
    -> denoise/repair into complete children
    -> evaluate reward
    -> select states
    -> repeat
```

RF-Refine Fusion does not import the external implementation unchanged. It replaces the unconditional generator with fixed-backbone inverse folding, random global remask with targeted register reopening, and the external reward with the frozen Head. Explicit mutation is added as a breadth branch when denoising support is too conservative.

RERD is therefore not discarded. The external RERD already contains both an inner reward lookahead and an outer population resampling step; what it lacks is exact carryover of the evaluated complete state, a hard feasibility constraint, and explicit lineage. The fusion reconstructs RERD's heuristic resampling into a constrained Head-FK macro-state selection, rather than supplying a population selection that RERD does not have. The edit-and-repair cycle is the project-specific realization of its refinement principle.

### 10.2 FK defines population guidance

FK contributes a different and complementary operation: it moves **population mass and future compute**, not individual logits. Head-derived potentials weight complete children, and ancestry resampling gives lower-burden feasible states more descendants in later RERD rounds.

This is more than independent best-of-$N$. Its value is the interaction between rounds: a promising state changes the distribution of future proposals because it acquires more descendants. A hard beam is a deterministic approximation and an important ablation, but it is not the only intended selector.

The current sampler's independent batch lanes do not already implement this behavior. That is an implementation gap, not a reason to remove FK from the scientific target.

### 10.3 Their roles are complementary

The two ideas answer different questions:

- RERD: **how is a selected state reopened to generate a better local neighborhood?**
- FK: **which Head-favorable states receive more descendants and search budget?**
- RF/DPLM: **how are reopened regions proposed or repaired under the target backbone?**
- Explicit mutation: **how is proposal breadth preserved outside the denoiser's local mode?**
- Head: **which complete sequences and ancestry are desirable?**

The fusion is the composition of these roles, not the forced transplantation of either external algorithm.

### 10.4 Logit guidance remains proposal-side

D-CBG-, GILC-, or other discrete-logit guidance may improve the proposal distribution. Their value is measured by hard-child recall and proposal efficiency. They do not replace complete-state evaluation, structure gating, or exact feedback.

## 11. Scientific hypotheses and falsifiers

### H1. The Head is an actionable guidance objective

**Hypothesis**: whole-sequence Head potentials with new-hotspot control can guide repeated hard-state correction toward lower Head-defined terminal immune burden while retaining structure-feasible states.

**Falsifier**: the candidate population contains lower-Head feasible states but the guidance law fails to enrich or preserve them, or local Head improvements repeatedly increase the final whole-sequence Head landscape.

### H2. The structural feasible set is wider than the D2 token support

**Hypothesis**: broad edits excluded by local structural-logit thresholds can often be repaired into high-scTM complete sequences.

**Falsifier**: state-level structure evaluation rejects nearly all candidates outside the current D2 support, or accepted out-of-support candidates do not improve immune burden.

### H3. RF repair improves the immune-structure frontier

**Hypothesis**: local backbone-conditioned repair increases the feasible fraction or structural quality of immune-improving hard edits relative to direct mutation alone.

**Falsifier**: repair merely reverts immune edits, adds no structure benefit, or is dominated by direct mutation plus state-level filtering.

### H4. Exact feedback outperforms marginal actuation

**Hypothesis**: carrying the evaluated child into the next round recovers more terminal Head gain than projecting the same candidate evidence into marginal logits and remask ranks.

**Falsifier**: compute-matched marginal steering reaches the same terminal Head and structure frontier as exact hard-state carryover.

### H5. Late targeted reopen is better aligned than continued global remask

**Hypothesis**: early global remask followed by late reward-evaluated local reopen preserves structural generation benefits while reducing the loss of useful immune edits.

**Falsifier**: continued global remask yields equal or better Head, scTM, and diversity under matched proposal compute.

### H6. FK interaction adds value beyond independent search

**Hypothesis**: Head-potential weighting and ancestry resampling improve the final Head-structure frontier or reduce the proposal budget relative to independent best-of-$N$ and hard-beam controls.

**Falsifier**: compute-matched independent or beam search matches the population frontier, or FK resampling collapses diversity without increasing feasible Head improvement.

## 12. Frozen scientific decisions

The following decisions define the proposed method:

- Runtime immune control uses the frozen Head on complete hard sequences.
- NMP has no role in guidance; any retained use is terminal external reporting or filtering.
- The persistent state is a population of exact complete sequences; single-parent and hard-beam variants are controls.
- Broad explicit mutation is a first-class proposal family.
- RF/DPLM is used as a backbone-conditioned proposal and local repair operator.
- RERD-style targeted reopen, complete-child evaluation, and repeated feedback define the correction loop.
- Head-guided FK weighting and ancestry resampling are a first-class population selection law, with beam and independent search as controls.
- Local structural logits are a proposal prior, not the fold-feasibility boundary.
- Structural acceptance is a complete-state constraint.
- The parent remains available as a null move.
- Accepted children are inherited exactly.
- Generic global remask does not silently alter accepted late-stage states.
- Residue-level fields allocate proposals; complete-state Head values select ancestry.
- Generation is two-phase: base RF with structural remask before a maturity point $t_\star$, then a reward-eligible complete hard-state population after it. Persistent particle states are complete at correction-round boundaries; transient masked states are permitted only inside $K_{\mathrm{move}}$ and never become inherited ancestry.
- The guidance claim is Head-guided FK optimization, not exact twisted-SMC posterior sampling.
- Feasibility (structure and no-new-hotspot) is the hard admissibility; Head is a soft objective; an elite archive of the best-so-far feasible state makes the returned design monotone.
- The $t_\star$ handoff does not inherit low-level controller state (scores, unmask history, sticky or pending D2 evidence, D3 memory, pressure state) into particle ancestry; such state may feed nomination or telemetry only.
- Structural feasibility is two-tier: a configurable cheap surrogate screens raw proposals and enters the FK potential; the definitive refold gate (scTM to the target backbone) is required for any candidate to enter ancestry.
- The FK population weighting is closed: per-parent child budget or offspring-count normalization; infeasible children get zero weight; non-resampling rounds carry normalized weights, resampling rounds reset to $1/N$; an all-infeasible round keeps the null parent and never falls back to uniform.
- The FK potential targets the absolute Head level (annealed form); the fixed-$\beta$ difference potential is its degenerate case.
- Per-round partial-state value rollouts and exact proposal-ratio correction are deferred extensions.

The following remain open because the current evidence does not determine them:

- the best Head global-risk functional and uncertainty margin;
- the target-register responsibility rule;
- mutation breadth and edit radius;
- repair-halo size and whether edited core positions are always protected;
- the specific cheap structure surrogate (configurable) and the refold budget/cadence for ancestry candidates;
- greedy versus beam selection;
- FK population size, potential tempering, ESS policy, and diversity preservation;
- the exact maturity point $t_\star$ and its trigger criterion (that a $t_\star$ handoff exists is frozen; its value is not);
- whether D2 or SC-GR improves proposal efficiency after the core loop is established.

## 13. Evidence ledger and claim boundaries

| Claim | Current status |
|---|---|
| The B1 controller-remask lifecycle changes terminal immune outcomes | Supported on the high-risk stress cohort |
| D2 alone explains the B1 gain | Not identified |
| Broad iterative reward search can find strongly reward-improved, high-scTM neighbors | Supported by the NMP-driven mechanism control |
| Head guidance must reproduce NMP rankings or outcomes | Not required by the method objective |
| Head-guided hard-state feedback lowers Head burden under the structure constraint | Unproven and load-bearing |
| Candidate breadth is a stronger observed lever than coordinated denoising | Supported by Path-C diagnostics |
| Local structural-logit support equals the fold-feasible set | Rejected as a scientific interpretation |
| Specific successful refinement edits were excluded by D2 support | Not yet measured |
| Global remask changes terminal outcomes and can erase D2 edits | Supported |
| Removing all remask makes the controller sufficient | Not supported |
| RF local repair improves over direct broad mutation | Unproven and load-bearing |
| Global scTM preservation implies activity preservation | Not supported |
| RERD-style targeted reopen improves over continued generic remask | Unproven and load-bearing |
| FK ancestry resampling improves over matched beam or independent search | Unproven and attributable by ablation |

## 14. References and project evidence

1. Singhal R, Horvitz Z, Teehan R, et al. [A General Framework for Inference-time Scaling and Steering of Diffusion Models](https://arxiv.org/abs/2501.06848). 2025.
2. Uehara M, Su X, Zhao Y, et al. [Reward-Guided Iterative Refinement in Diffusion Models at Test-Time with Applications to Protein and DNA Design](https://proceedings.mlr.press/v267/uehara25a.html). *Proceedings of Machine Learning Research*. 2025;267:60515-60529.
3. Hasan M, Ohanesian V, Gazizov A, et al. [Discrete Feynman-Kac Correctors](https://openreview.net/forum?id=M2xR2Osn9E). 2025-2026.
4. Hartman E, Wallin J, Malmstrom J, Olsson J. [Controllable Protein Design through Feynman-Kac Steering](https://arxiv.org/abs/2511.09216). 2025.
5. Project evidence: [RAR 0013 B1 versus diverse NoD](../Results/Analysis/0013-b1-high-risk-controller-vs-dplm-opening-/record.md), git `c247897`.
6. Project evidence: [RAR 0014 D2 disagreement fate](../Results/Analysis/0014-stage-b1-cap20-d2-disagreement-fate-diag/record.md).
7. Project evidence: [RAR 0017 clean remask ablation](../Results/Analysis/0017-remask-clean-ablation-probe-drb1-0701-10/record.md).
8. Project evidence: [RAR 0022 coordination headroom](../Results/Analysis/0022-c0b-path-c-coordination-headroom-existen/record.md).
9. Project evidence: [RAR 0023 breadth and selection decomposition](../Results/Analysis/0023-c0b-headroom-decomposition-downhill-vs-s/record.md).
10. Project evidence: [RAR 0025 SC-GR where/value ablations](../Results/Analysis/0025-sc-gr-v1-1-where-value-ablations-triage-/record.md).
11. Project evidence: [RAR 0026 refinement reach-zero outcomes](../Results/Analysis/0026-char24-0701-refinement-outcomes-reach-0-/record.md), git `824ca53`.
