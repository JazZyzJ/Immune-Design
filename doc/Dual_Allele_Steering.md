# Dual-Allele Steering for RF Fusion V2

**Status:** Layered scientific architecture, revised through RAR 0049.

**Document type:** Method definition, evidence boundary, and research horizon. This is not an
implementation plan or experiment runbook.

**Primary architecture source:** [FUSION_V2.md](./FUSION_V2.md).

**Executed V2 runtime source:** Fusion-worktree
RF_Fusion_v2_Cluster_Runbook.md, especially its Head-only objective, exact-endpoint recursion, and
source-compatible feedback contracts.

**Current empirical authority:**
[RAR 0049](../Results/Analysis/0049-rf-fusion-v2-high-risk-100-protein-multi/record.md), git
21cd73d.

**Historical dual evidence:**
[RAR 0027](../Results/Analysis/0027-dual-allele-steering-evidence-cross-alle/record.md),
[RAR 0028](../Results/Analysis/0028-cross-allele-leakage-of-single-allele-co/record.md), and
[RAR 0029](../Results/Analysis/0029-rar-0028-derived-audit-clustered-leakage/record.md).

**Literature research records:**
[Research 0001](../Results/Research/0001-dual-allele-steering/record.md),
[Research 0002](../Results/Research/0002-dual-allele-steering-lit-round2/record.md), and
[Research 0003](../Results/Research/0003-dual-allele-surrogate-uncertainty/record.md).

---

## 0. Scope and priority

The scientific content in this document has four different statuses. They must not be read as one
flat implementation package.

| Layer | Scientific role | Status in Dual Fusion V2 |
|---|---|---|
| **Foundation** | Existing V2 exact endpoint, source-compatible lineage, projection, recursion, immutable archive, and feasibility contracts | Inherited unchanged |
| **Minimal Dual core** | Two frozen Heads on the same endpoint, fixed normalization, one bounded smooth worst-residual objective, and the same objective used for donor and directional write support | Primary method to validate |
| **Conditional enhancements** | Online Pareto control, multiple preferences, richer action classes, group attribution, and adaptive search allocation | Activated only after a measured core failure |
| **Research horizon** | Pareto MCTS, FK/SMC, learned preference-conditioned proposals, GFlowNets, and direct multi-objective transition guidance | Preserved as future directions; not part of the first Dual identity |

The minimal method below is not a second architecture invented beside Fusion V2. It is the smallest
instantiation of the multi-objective formulation already motivated in earlier versions of this
document:

> Replace the single-Head scalar used by the existing V2 donor and feedback laws with one frozen,
> normalized, simultaneous two-Head objective. Keep the V2 state machine unchanged.

### 0.1 Canonical statement

> **Minimal Dual Fusion V2 evaluates every exact endpoint with two frozen allele-specific Heads,
> ranks donors with a frozen-normalized smooth worst-residual objective, and uses the same objective
> to decide which endpoint identities are written into that endpoint's own source-compatible
> lineage. Existing V2 re-entry coordinates, schedule bands, and total reopen cardinality remain
> unchanged; the eligible reopen action space is the union of the two allele-specific evidence
> views rather than the action set of one privileged Head.**

The second Head is therefore not merely a terminal observer. It changes the selected exact donor
and the feedback that shapes future descendants.

This core does **not** require:

- optimizing one allele to saturation before introducing the other;
- intersecting two complete single-allele policies;
- a new interacting particle population;
- a new online Pareto state machine;
- separate write/reopen budgets for the two alleles; or
- FK, MCTS, or a learned multi-objective proposal.

Those ideas remain scientifically useful, but they are not equally necessary.

### 0.2 Why the objective should be simultaneous from depth zero

RAR 0049 contains 263 no-better-donor stalls, including 202 whose single-allele incumbent risk was
already below −9. This creates a favorable opportunity for Dual steering: many lineages had little
residual pressure under the first objective, while their second-allele opportunity was never used.
It does not guarantee that useful second-allele donors or projectable support exist; those are the
central opportunity and directionality hypotheses below.

The primary hypothesis is therefore to avoid freezing the first allele's single-objective optimum
before optimizing the second. That optimum need not lie near the best joint trade-off, and early
commitment can remove jointly useful paths. In the first simultaneous method, both Heads
participate in endpoint evaluation, donor selection, and feedback from the beginning of Dual
recursion. As one normalized allele becomes satisfied, a smooth worst-residual law naturally
transfers pressure to the other without an explicit phase switch.

Protected sequential search remains an optional attribution control. It is not the primary method
and is not required before the simultaneous law can be studied.

---

## 1. What V2 contributes before Dual

The earlier Dual document was organized around standalone refinement, partial-root continuation
values, candidate-level preference lanes, and possible FK selection. Fusion V2 changed the operative
unit.

At depth $d$, the persistent scientific object is a source-compatible pair

$$
\left(P_d,y_d\right),
$$

where $P_d$ is a live partial inverse-folding state and $y_d$ is an exact complete endpoint
generated from that source. The endpoint is scored and archived. If selected, endpoint evidence is
projected only into its own compatible source trajectory through the existing V2 write/reopen law.

The load-bearing V2 invariants remain:

1. **Complete-state evaluation.** Head evidence is computed on an exact complete sequence, not on a
   fictitious mixture of partial completions.
2. **Source compatibility.** An endpoint may feed back only through its stored source/replay
   lineage; it cannot be assigned to an arbitrary root.
3. **Exact archive preservation.** Projection acts on a copy and cannot destroy a previously found
   complete endpoint.
4. **Shared intervention budget.** Write, protect, and reopen actions remain bounded by the existing
   V2 schedule budget $\mathcal{B}(r)$.
5. **Feasibility inheritance.** Structure, anchors, and functional geometry remain the unchanged
   V2 feasibility contract.

RAR 0049 established that the current single-allele system is recursively executable and that its
successful-cohort archive moves substantially beyond its own D0 frontier. It did not isolate the
causal increment of source coupling over A2, v0, or source-off controls. Dual should extend this
state machine, not replace it with a new population algorithm before the basic two-objective law is
known to work.

---

## 2. Dual objective

### 2.1 Same-sequence risk vector

For two selected HLA-DRB1 alleles $A$ and $B$, every exact endpoint $y$ receives

$$
\mathbf{R}(y)
=
\left(
R_A(y),
R_B(y)
\right).
$$

Both entries must refer to the same sequence digest, protein, endpoint identity, and scoring
window grid. The best endpoint under Head $A$ and a different best endpoint under Head $B$ do not
form a jointly good design.

The two frozen Epitope Heads are the complete runtime and evaluation immune objective. Another
immune predictor does not choose donors, writes, reopens, stopping, configuration, or the reported
Dual capability verdict. Cross-predictor agreement belongs to Head development rather than to the
acceptance of the Dual steering method.

### 2.2 Fixed per-allele coordinates

Raw scores from independently trained Heads are not directly comparable. Each allele is mapped to a
frozen coordinate

$$
u_a(y)
=
\frac{R_a(y)-b_a}{s_a},
\qquad
a\in\{A,B\},
$$

where lower is better, $b_a$ is a declared reference location, and $s_a>0$ is a declared scale.
These values are calibrated on a V2-compatible population such as matched guidance-off NoD or a
disjoint frozen D0 population. They are fixed for the run and are never recomputed from the current
endpoint cloud.

This shared coordinate is required by the runtime decision law, not merely by final reporting.
Scoring the final sequences separately against allele-specific NoD references can describe the
outcome on each axis, but it cannot recover donors or feedback identities that were discarded
during recursion because one raw Head scale dominated the other. Equal coefficients on two
independently trained raw-logit Heads therefore do not by themselves imply equal scientific
pressure. The affine transformation makes the exchange geometry explicit while preserving the
within-allele ordering of exact endpoints.

Calibration and stochastic tolerance are different:

- $b_a,s_a$ define the geometry shared by the two Heads;
- $\epsilon_a$ describes same-sequence repeatability for Head $a$;
- $\epsilon_{\mathrm{donor}}$ is the matched-difference noise floor for comparing two exact
  endpoint values after the complete joint scalarization; and
- $\epsilon_{\mathrm{write}}$ is the matched-difference noise floor for the exact donor-versus-
  reverted-sequence contrast.

All are bound to Head checkpoint, config, score scale, window grid, reference population, and
content digest. The two joint margins may share a numerical value only when repeated-score evidence
supports that equivalence; one single-allele margin is not silently reused for both.

### 2.3 Frozen-normalized smooth maximum

The primary Dual objective is

$$
J_{\mathrm{core}}(y)
=
J_\tau(y)
=
\tau
\log
\left[
\frac{
\exp\left(u_A(y)/\tau\right)
+
\exp\left(u_B(y)/\tau\right)
}{2}
\right],
$$

where lower is better and $\tau>0$ is a frozen softness parameter. Division by two only fixes the
additive convention so that $J_\tau(u,u)=u$; it does not alter endpoint ordering or
leave-one-out differences.

Let

$$
M(y)
=
\max\left(
u_A(y),
u_B(y)
\right).
$$

The smooth maximum obeys

$$
M(y)-\tau\log 2
\leq
J_\tau(y)
\leq
M(y).
$$

Consequently, an arbitrarily favorable non-worst allele cannot supply an arbitrarily large credit
against the worse allele. Its total influence relative to the hard maximum is bounded by
$\tau\log 2$. The objective remains strictly increasing in each normalized allele coordinate, so a
Pareto-dominated endpoint cannot outrank the endpoint that dominates it. It is also continuous
through active-worst switches, allowing donor and leave-one-out evidence to use one stable scalar
rather than a separate tie rule.

As $\tau\rightarrow0$, $J_\tau$ converges to the strict normalized maximum. Thus $\tau$ has a
specific scientific meaning: it controls the width of the region in which the near-worst allele
may influence the decision. It is not an unrestricted compensation weight between the two Heads.

This scalar is intentionally the first method, not a claim that one balanced preference recovers
the complete Pareto frontier. It supplies the total ordering required by the current recursive V2
policy while remaining a bounded relaxation of strict worst-residual steering.

The matched strict control is

$$
J_{\max}(y)
=
\max\left(
u_A(y),
u_B(y)
\right).
$$

It uses the same donor, incumbent, and leave-one-out interfaces but gives no credit to a
non-worst allele until that allele reaches the active maximum. The comparison between $J_\tau$ and
$J_{\max}$ therefore isolates whether bounded near-worst evidence improves steering authority.
Appendix A positions the remaining scalarization families.

### 2.4 Donor law

Within one source-compatible lineage, an exact endpoint becomes the working donor only when

$$
J_{\mathrm{core}}\left(y_{\mathrm{inc}}\right)
-
J_{\mathrm{core}}\left(y_{\mathrm{donor}}\right)
>
\epsilon_{\mathrm{donor}}.
$$

The incumbent and donor remain exact endpoints from the same compatible recursive cell. No
cross-root endpoint reassignment is introduced.

Single-allele saturation is handled automatically. If $u_A$ is already low while $u_B$ remains
high, an endpoint that is neutral for $A$ but improves $B$ can become a valid donor. It would have
been invisible to the original $A$-only law without requiring the method to declare that allele
$A$ has permanently finished.

### 2.5 One objective from selection to actuation

For donor position $i$, let $y^{(-i)}$ be the exact donor endpoint with token $i$ reverted to the
**current lineage incumbent token** $I_{d,i}$. Define

$$
a_i^{\mathrm{core}}
=
J_{\mathrm{core}}\left(y^{(-i)}\right)
-
J_{\mathrm{core}}(y).
$$

An endpoint identity is directional write evidence only when

$$
a_i^{\mathrm{core}}
>
\epsilon_{\mathrm{write}}.
$$

This is the central simplification. Donor selection and feedback attribution are two uses of the
same joint scientific objective. The method does not select a donor with one law and then project it
through an unrelated four-class heuristic.

For interpretation, the per-allele contrasts

$$
a_{i,a}
=
R_a\left(y^{(-i)}\right)-R_a(y)
$$

may also be recorded. They explain whether a selected identity is shared-beneficial,
worst-allele-specific, conflicting, or unresolved, but those labels are derived telemetry in the
minimal method rather than new controller states.

The existing V2 legality, history, source, re-entry coordinate, schedule band, and total
$\mathcal{B}(r)$ constraints remain unchanged. Here, preserving reopen *geometry* means preserving
its timing and cardinality; it does not authorize a hidden single-allele priority over reopen
*identity*.

The minimal action view is a union. A position may enter the legal reopen candidate set when either
allele supplies the declared residual, worsening, or new-hotspot evidence, subject to the unchanged
source/history/anchor laws. It need not be supported by both alleles. The frozen joint policy then
orders this union under one shared cardinality; the two alleles never receive independent action
budgets. The final authority for an endpoint-written identity remains positive
$a_i^{\mathrm{core}}$ beyond its frozen margin. Thus the first method widens the evidence view, not
the intervention count.

This minimal union must not be implemented as “Head A prefilter, then joint scoring.” Such a path
would silently discard positions useful only to Head B and would reduce the method to a Dual donor
selector with a single-allele actuator. Richer conflict classes, group attribution, and separate
allele-conditioned priorities remain conditional enhancements in Section 6.3.

### 2.6 Minimal lifecycle

~~~mermaid
flowchart LR
    P["Live source-compatible state P_d"] --> Y["K exact complete endpoints"]
    Y --> HA["Frozen Head A"]
    Y --> HB["Frozen Head B"]
    HA --> J["Fixed normalization and smooth-max J_core"]
    HB --> J
    J --> D["Strictly better exact donor"]
    D --> C["Leave-one-out delta J_core"]
    C --> Q["Write by delta J_core; existing reopen geometry"]
    Q --> R["Project into donor-compatible source"]
    R --> P
    Y --> X[("Existing exact archive with dual evidence")]
~~~

This loop is already enough to define genuine Dual steering:

- Head $B$ can change which endpoint is selected;
- Head $B$ can change which donor identities are written; and
- those changes can alter fresh complete descendants.

The more ambitious designs in Sections 6 and 7 are not needed to establish this first causal claim.

---

## 3. State economy, archive, and output

### 3.1 No second state machine

Minimal Dual V2 does not introduce new persistent generative state types.

| Object | Role |
|---|---|
| Existing source/replay lineage | Persistent causal state; unchanged |
| Existing lineage incumbent | Persistent donor reference; now bound to dual evidence |
| Persisted complete-endpoint evidence | Evaluated exact sequences and definitive feasibility evidence; extended with the second Head |
| Existing monotone archive | Lifecycle-qualified exact states and return safety; unchanged lifecycle |
| $R_A,R_B$ and Head provenance | New endpoint evidence |
| $u_A,u_B,J_{\mathrm{core}}$ | Derived objective views; may be stored for audit |
| $a_i^{\mathrm{core}}$ and $a_{i,a}$ | Derived only for shortlisted donor/support events |
| Pareto membership, worst risk, hypervolume | Offline archive views in the minimal method |

The endpoint evidence and existing archive therefore gain **dual objective bindings**, not a new
online Pareto ancestry system.

### 3.2 Minimal elite and reporting frontier

The working lineage incumbent and balanced capability elite are ranked by $J_{\mathrm{core}}$ among
admissible exact endpoints. The persisted complete-endpoint evidence surface retains both raw
allele risks. The two-dimensional non-dominated frontier is reconstructed from the declared
complete-endpoint surface joined to definitive feasibility; archive membership is not assumed to
be identical to the set of all evaluated endpoints.

An online Pareto archive becomes necessary only if evidence shows that the scalar working law
irreversibly prunes reproducible, useful trade-off basins. Until then:

- scalar $J_{\mathrm{core}}$ controls recursion;
- persisted complete-endpoint evidence preserves the evaluated scientific surface;
- the existing monotone archive keeps its current V2 lifecycle; and
- Pareto analysis remains a derived scientific view.

This distinction removes the need for a second archive state machine while preserving all
information needed to study trade-offs.

### 3.3 Terminal semantics

Two output views are distinct:

- **Balanced capability elite:** the admissible archived endpoint with lowest frozen
  $J_{\mathrm{core}}$.
- **Deployable protected elite:** the lowest-$J_{\mathrm{core}}$ endpoint inside a predeclared
  named-allele terminal protection subset.

For the current named-allele deployment semantics, bind one common content-defined reference
$y_{\mathrm{ref}}^{\mathrm{deploy}}$ before Dual search and require

$$
R_a(y)
\leq
R_a\left(y_{\mathrm{ref}}^{\mathrm{deploy}}\right)
+
\epsilon_a^{\mathrm{deploy}},
\qquad
a\in\{A,B\}.
$$

If no endpoint satisfies this terminal contract, the method reports no deployable protected design;
it does not silently return the balanced elite as deployment success. This terminal slice does not
optimize one allele first and does not constrain intermediate recursion to improve both raw risks at
every depth.

---

## 4. Safety and scope boundaries

### 4.1 Structure is inherited, not a Dual research target

Dual steering does not solve the structure bottleneck. It inherits the same definitive structure,
anchor, and functional-geometry gates as Fusion V2.

Structure remains outside $J_{\mathrm{core}}$: an immune improvement cannot compensate a structure
failure. Conversely, this document does not propose new structure scoring, repair, root allocation,
or thresholds. Any structural attrition belongs to the shared V2 substrate and must be held matched
when Dual and single-allele objectives are compared.

### 4.2 Do not intersect two full single-allele policies

The primary Dual law is not

$$
\mathrm{pass}_A
\cap
\mathrm{pass}_B
$$

when each pass contains the complete single-allele improvement policy. Such an intersection would
prematurely remove endpoints that make a useful joint trade-off and would recreate the compression
problem of sequential optimization.

Only true invariant safety conditions remain conjunctive:

- unchanged V2 structure and functional feasibility;
- per-allele catastrophic cumulative new-hotspot limits; and
- exact source/history/action legality.

For exact endpoint $y$ and its immutable cumulative safety reference $\bar y$, the minimal
admissibility law is

$$
I_{\mathrm{adm}}^{AB}(y\mid\bar y)
=
\mathbf{1}\left[
F_{\mathrm{V2}}(y)=1,
\;
N_A^{\mathrm{whole}}(y;\bar y)\leq\delta_A,
\;
N_B^{\mathrm{whole}}(y;\bar y)\leq\delta_B
\right],
$$

where $F_{\mathrm{V2}}$ contains the inherited structure, anchor, function, and source-legality
contract. The two $\delta_a$ values are independently calibrated catastrophic-safety bounds. This
is an intersection of invariant safety constraints, not an intersection of two full single-allele
optimization policies.

Strict per-transition improvement in both raw allele scores is **not** required. Intermediate
bounded trade-offs are allowed when they improve the frozen joint objective and pass the hard
safety rules.

The relevant yield quantity is therefore not a product of two single-allele pass rates. It is the
frequency of exact endpoints and support actions that are:

1. feasible under the inherited substrate;
2. legal under the shared action budget;
3. non-catastrophic for either allele; and
4. better under $J_{\mathrm{core}}$ beyond its repeatability margin.

### 4.3 Runtime objective boundary

The two frozen Heads are the only immune steering signals and the only immune quantities used to
accept, reject, tune, or rank the Dual method. WT-relative changes and predictions from another
immune model may be retained as contextual analyses, but they are not a second objective or a
required validation gate.

Historical strong-core analyses still distinguish count from newly strong locus:

$$
C_a(y_{\mathrm{out}})
\leq
C_a(y_{\mathrm{ref}})
$$

does not imply

$$
N_a^{\mathrm{new}}(y_{\mathrm{out}})
=
0.
$$

Historical NMP-guided refinement is evidence that a complete-state search law can exert authority
and that allele landscapes differ. It is motivation only; it is not an objective or acceptance
criterion implemented by Dual V2.

---

## 5. Evidence basis and necessary attribution

### 5.1 Historical reason to make the objective dual

RAR 0027–0029 show that a single-allele improvement cannot certify second-allele safety or utility.

| Observation | Measured result | Dual interpretation |
|---|---:|---|
| WT exact-core Jaccard, 0401–0701 / 0401–1501 / 0701–1501 | 0.133 / 0.067 / 0.000 | Responsibility is strongly allele-tagged; intersection-only targeting is inadequate. |
| Three-allele core union / largest single-allele set | median 1.8 | Multi-allele target breadth grows, but edit cost need not scale linearly. |
| Paired off-target core-count delta median | 0 or −1 | Passive transfer exists but is not systematic. |
| Off-target net core-count regression | 6.5–15.4% per direction | Single-allele steering has a meaningful regression tail. |
| Newly strong off-target locus | 14.3–34.1% per direction | Aggregate improvement does not ensure locus stability. |
| New locus in either off-target allele | 31.1–46.3% | Both allele landscapes must be observed during steering. |
| Incidental pairwise dual-zero endpoints | 10/359; triple-zero 0 | Joint reachability exists but was not a systematic capability. |

These cohorts are historical, NMP-informed, and selected. They motivate the dual objective but do
not calibrate V2 Head thresholds or prove that protected sequential search is optimal.

### 5.2 What RAR 0049 establishes

RAR 0049 used 100 high-risk proteins, four independent roots per protein, D=4, K=12, r=40, and a
single-allele strict-improvement Head policy.

| Measurement | Result | Relevance to Dual |
|---|---:|---|
| Successful roots | 328/400 | Root identity remains stochastic; results require the full denominator. |
| Proteins with at least one successful root | 84/100 | Dual opportunity must not condition away 16 unsuccessful proteins. |
| Exact complete endpoints | 14,544 | A large exact cloud is available for low-cost second-Head scoring. |
| Definitively structure-feasible endpoints | 13,148 | Dual can reuse the V2 substrate; it need not redesign structure handling. |
| D0-to-final improvement beyond report epsilon | 81/84 successful proteins | The archive moved beyond its own D0 frontier; causal superiority over A2/v0/source-off remains unresolved. |
| Final feasible frontier median | −9.4236 | Many lineages reached a single-Head floor. |
| New protein frontiers at D1 / D2 / D3 / D4 | 79 / 51 / 17 / 4 | Under this profile, observed marginal gain concentrated at D1/D2. |
| No-better-donor stalls | 263 | Single-objective saturation is a major stopping mode. |
| Stalled incumbents below −9 | 202/263 | A second residual objective may reactivate useful lineages. |
| Median legal candidates among committed transitions | 47 → 37.5 → 28 → 18 | Later committed events used narrower action sets; stalls are not in this denominator. |
| Median positive contributions among committed transitions | 13 → 9 → 6 → 3 | Single-site joint support may become sparse, but this remains to be measured prospectively. |

RAR 0049 supports recursive executability and rich exact-endpoint evidence. It does not prove that
Dual recursion works, that source coupling is superior, or that the second allele necessarily
increases useful depth.

### 5.3 What must be causally distinguished

The second Head may enter the system at three strengths:

1. **Observer:** scores endpoints but changes nothing.
2. **Selector:** changes the donor but not the projected support.
3. **Actuator:** changes donor and/or directed writes and shifts fresh complete descendants;
   conditional extensions may also change reopen priority.

The core scientific claim requires the third level. A matched causal comparison therefore asks:

> Holding source root, endpoint opportunity, support cardinality, shared budget, replay seeds, and
> horizon fixed, does the joint objective change selected endpoint evidence and shift fresh
> descendant $J_{\mathrm{core}}$ beyond repeatability?

The most important controls are:

- $A$-only V2;
- $B$-only V2;
- balanced simultaneous Dual V2; and
- guidance-off or objective-observer V2 under matched generation authority.

Protected sequential $A\rightarrow B$ and $B\rightarrow A$ can remain attribution controls because
they test whether a simpler ordered policy reproduces the same joint frontier. They are not
prerequisites and should not define the primary search space. If the paper claims a specific
advantage from source-compatible recursion, A2, complete-state v0, or source-off controls are
additionally required; they are not part of the definition of Dual steering itself.

### 5.4 Minimal claim levels

These are scientific claim levels, not an implementation schedule:

| Claim | Evidence required |
|---|---|
| **Joint opportunity** | Existing exact endpoint clouds contain beyond-noise $J_{\mathrm{core}}$ improvements and positive support evidence after independent calibration. |
| **Joint directionality** | Joint donor/support changes fresh descendant distributions in the declared direction under matched sources and budgets. |
| **Recursive Dual authority** | Repeated Dual feedback expands the matched feasible joint frontier beyond observer and single-allele controls. |
| **Generality** | The frozen law holds across proteins, roots, and at least one additional frozen allele pair or untouched protein cohort without hiding the full failure denominator. |

The RAR 0049 cross-score is an efficient opportunity probe, not a generative verdict. Normalization
must come from an independent NoD or disjoint calibration population, and repeatability must come
from repeated scoring of the same exact sequences.

### 5.5 Cost interpretation

Scoring one exact endpoint with a second Head is cheap relative to the shared structure/refold
work, and the structure verdict itself is allele-independent. Dual's main cost is therefore not a
second structure pipeline. It is:

- the extra exact counterfactual Head evaluations needed for $\Delta J_{\mathrm{core}}$;
- any reduction in the number of beyond-noise joint donors or positive write identities; and
- extra search breadth only if the balanced core empirically needs it.

The appropriate efficiency unit is useful joint donor/write yield per fixed V2 generation and
feasibility budget. A preference population or broader search should not be added merely because
the second Head forward is inexpensive.

---

## 6. Conditional enhancements

The following ideas remain part of the scientific program. None should enter the core merely
because it is methodologically attractive.

### 6.1 Pareto-aware lineage selection

The persisted exact-endpoint evidence already permits offline Pareto analysis. Promoting Pareto
membership into online lineage/environmental selection is justified only if the balanced scalar law
repeatedly discards reproducible endpoints that later become useful for the other allele.

If activated, epsilon-dominance should account for frozen per-allele repeatability. Endpoint
$y$ dominates $y'$ only when it is no worse beyond tolerance on both axes and strictly better
beyond tolerance on at least one. The archive must preserve source lineage; it cannot become a
source-free evolutionary population.

### 6.2 Multiple preference laws

A general frozen preference $w$ may use

$$
J_{w,\tau}(y)
=
\tau
\log
\left[
\frac{1}{2}
\sum_{a\in\{A,B\}}
\exp
\left(
\frac{
w_a\left(u_a(y)-z_a^\star\right)
}{\tau}
\right)
\right],
$$

where $z^\star$ is a frozen reference point and $w$ is a frozen positive preference vector
normalized so that $\max_a w_a=1$. The balanced case $w_A=w_B=1$ and $z^\star=0$ recovers the
primary law. This is the preference-weighted smooth-Tchebycheff extension of that law. It can expose
A-protect, balanced, and B-protect regions when the endpoint cloud already reaches them, but it
does not create absent trade-offs.

Multiple preferences are warranted only if one balanced law shows measurable pruning regret or
misses stable trade-off basins. Two possible forms remain:

- **preference-root ensemble:** independent roots use frozen preferences for search diversity;
- **source-compatible preference branching:** the same captured source root forks matched
  preference descendants, each retaining its own source, replay, incumbent, and objective.

Independent roots confound preference with root identity and cannot by themselves establish causal
preference superiority. No fixed one-root-per-preference split should be treated as canonical.

### 6.3 Richer action attribution

The minimal core already admits the union of the two allele evidence views into one shared reopen
candidate space. Per-allele counterfactual vectors can additionally classify positions as
shared-beneficial, preference-beneficial, conflicting, or unresolved. MOG-DFM-style cones or
lexicographic safety-first rules may then refine the priority *within* that union.

This becomes useful only if the scalar $a_i^{\mathrm{core}}$ law produces a reproducible failure,
such as excessive one-allele regressions or failure to reopen the current worst-allele region.
The conditional extension is therefore not “allow Head B positions”; the minimal union already
does that. It is a richer decision among legal union positions using shared/conflicting action
classes while preserving the existing source/history legality and shared $\mathcal{B}(r)$
cardinality. These richer classes and priorities are initially diagnostic evidence, not mutable
controller modes.

Single-position leave-one-out attribution may also fail when a useful endpoint depends on a
multi-residue coalition:

$$
a_i^{\mathrm{core}}\leq\epsilon_{\mathrm{write}}
\quad
\text{for all }i,
$$

despite donor-level improvement. The minimal response is a typed no-positive-core-write stall.
Pair, block, Shapley-like, or sibling-consensus attribution is justified only if that stall becomes
a measured bottleneck.

### 6.4 Adaptive search allocation

Dynamic breadth, depth, root allocation, or preference branching is justified only if fixed
balanced lineages show a reproducible allocation problem:

- useful basins are found but pruned;
- one root repeatedly receives too little opportunity;
- later depth remains valuable for the unsatisfied allele; or
- endpoint breadth rather than support directionality becomes limiting.

Until then, the current independent-root recursion is the cleaner scientific substrate.

---

## 7. Research horizon and literature map

### 7.1 Multi-objective protein design and Pareto selection

Multi-objective protein-design studies motivate explicit vector evaluation and diversity-preserving
environmental selection rather than a hidden pair-specific predictor. They support the scientific
importance of Pareto views, but do not require online NSGA-II-style selection in the minimal V2
method.

MOEA/D and Tchebycheff methods motivate a total preference law that can rank non-convex trade-offs
when the proposal cloud already contains them. Smooth-Tchebycheff theory concerns differentiable
global optimization; here the objective is a black-box scalarization over a finite endpoint cloud.
No completeness result transfers.

### 7.2 Multi-objective discrete flow guidance

MOG-DFM motivates directional filtering of discrete actions and explicit handling of conflicting
objectives. It does not establish that whole-endpoint contrasts yield causal residue writes.
Mapping exact endpoint evidence back to V2 write/reopen support remains a project-specific
hypothesis that requires fresh-descendant directionality evidence.

Direct per-step multi-objective guidance is a future option only if it can preserve the V2 rule that
the Head is evaluated on exact complete sequences rather than noisy partial states.

### 7.3 Pareto tree-guided diffusion

MP2D-style Pareto MCTS is a close conceptual neighbor for dynamic breadth/depth allocation and
iterative remasking. It also retains complete candidates for later refinement. V2's distinctive
contract is that each exact endpoint remains bound to a compatible source-partial replay lineage,
with immutable endpoint identity and causal rollback.

Pareto tree allocation should be considered only if fixed independent roots show measurable
pruning or allocation regret. It is not needed to define the first Dual objective.

### 7.4 FK and SMC

Multiplying two allele potentials gives

$$
\exp\left(-\beta_A R_A\right)
\exp\left(-\beta_B R_B\right)
=
\exp\left[
-\left(
\beta_A R_A+\beta_B R_B
\right)
\right],
$$

which is still a weighted-sum objective. It neither prevents compensation nor automatically
preserves unsupported Pareto points.

The project's tested complete-state scalar FK was also inferior to deterministic beam under its
frozen configuration. FK is therefore not the default Dual extension. Preference-specific SMC may
be reconsidered only if many active compatible lineages exist and stochastic lineage allocation,
rather than donor quality or support attribution, is shown to be limiting.

### 7.5 Learned preference-conditioned proposals

Multi-objective GFlowNets, preference-conditioned generators, and learned support policies may
amortize search after the project accumulates a substantial dual endpoint/lineage corpus. They are
not prerequisites for black-box Dual V2 and cannot replace source compatibility or complete-state
Head evaluation.

### 7.6 Inference-time protein search

RosettaSearch supports the broader principle of generative proposal, complete-state feedback, and
iterative inference-time optimization. It uses LLM/VLM optimizers and structure-predictor feedback
rather than source-compatible diffusion rollback. It motivates the search framing without
supplying V2's exact replay or endpoint-source contract.

### 7.7 Longer-term extensions

Other horizons include:

- more than two alleles and an explicitly population-weighted deployment objective;
- shared or allele-conditioned Heads with calibrated cross-allele uncertainty;
- learned group-level feedback for epistatic write support;
- preference-conditioned inverse-folding proposals; and
- biological presentation, T-cell, activity, and function validation after computational
  capability is frozen.

These directions should remain visible, but none carries the first Dual-V2 claim.

---

## 8. Hypotheses and falsifiers

### 8.1 Core hypotheses

| ID | Hypothesis | Falsifier |
|---|---|---|
| **C1 — joint opportunity** | V2 exact endpoint clouds contain beyond-noise improvements under frozen $J_{\mathrm{core}}$. | The second allele is uniformly adverse, or no joint improvement exceeds repeatability under independent calibration. |
| **C2 — donor authority** | The normalized smooth-maximum law selects different useful exact donors than either single-allele law when residual risks differ. | Rankings remain effectively identical to one Head or are dominated by calibration scale. |
| **C3 — support directionality** | Leave-one-out $\Delta J_{\mathrm{core}}$ writes shift fresh complete descendants toward lower $J_{\mathrm{core}}$ under the unchanged V2 reopen substrate. | Projected states change but fresh descendants are directionally null or adverse. |
| **C4 — recursive value** | Repeated joint feedback expands the matched joint frontier beyond observer and single-allele controls. | Extra recursion only repeats initial breadth, increases stalls, or transfers no second-allele benefit. |

One-allele-only gain is not automatically a C3 failure when the other allele was already within its
repeatability/floor band. Interpretation must condition on the pre-intervention residual vector.

### 8.2 Conditional hypotheses

| ID | Hypothesis | Activation condition | Falsifier |
|---|---|---|---|
| **E1 — Pareto preservation** | Online vector selection preserves useful basins lost by the scalar core. | Reproducible scalar-pruning regret | Offline and online frontiers are equivalent at matched authority. |
| **E2 — preference diversity** | Multiple preferences add stable A-protect/B-protect trade-off regions. | Balanced law misses reproducible basins | Preferences duplicate the balanced archive or only label root noise. |
| **E3 — group attribution** | Multi-residue support resolves donor improvements with no positive single-site $\Delta J$. | No-positive-core-write is a dominant stall | Group attribution adds cost without improving descendant directionality. |
| **E4 — adaptive allocation** | Tree/population allocation improves frontier coverage at fixed generation budget. | Fixed allocation is a measured bottleneck | Extra machinery changes allocation but not the joint frontier. |

This separation prevents an unvalidated extension from becoming part of the method merely because
it is theoretically elegant.

---

## 9. Reporting and claim boundary

The minimal method must preserve enough evidence to reconstruct its causal claim:

- exact endpoint and source/replay identity;
- raw $R_A,R_B$ and Head provenance on the same sequence digest;
- frozen normalization, $\epsilon_{\mathrm{donor}}$, $\epsilon_{\mathrm{write}}$, and objective digest;
- incumbent, donor, and $J_{\mathrm{core}}$ difference;
- counterfactual $a_i^{\mathrm{core}}$ for tested support positions;
- selected writes, inherited reopen events, and shared $\mathcal{B}(r)$ usage;
- fresh descendant scores and typed stalls; and
- unchanged V2 feasibility verdicts and full protein/root denominator.

Per-allele counterfactual vectors, action classes, Pareto membership, hypervolume, and preference
coverage are derived analyses unless a conditional enhancement explicitly promotes them to runtime
control.

The primary generated-distribution comparison uses matched guidance-off NoD/DPLM and both frozen
Heads. WT-relative change is contextual; it is not a standalone verdict. Endpoint rows must never
be pooled as independent proteins.

The strongest claim supported by the minimal method would be:

> A source-compatible recursive inverse-folding process used one frozen-normalized,
> smooth-maximum two-Head objective at both exact-endpoint selection and feedback attribution,
> thereby moving fresh complete descendants toward a better dual-allele risk trade-off without
> changing the underlying V2 feasibility contract.

A claim about broader Pareto coverage, preference diversity, tree-search efficiency, or exact
multi-objective sampling requires the corresponding conditional evidence and must not be borrowed
from the minimal method.

---

## Appendix A. Joint-objective alternatives

### A.1 Scalarization comparison

The primary choice is driven by the needs of the V2 recursive interface: one frozen total order must
serve donor selection, incumbent update, and exact leave-one-out write attribution. It must also be
monotone in each allele coordinate and prevent an arbitrarily favorable value on one axis from
masking a worse value on the other.

| Method | Geometry | Relevance to Dual V2 |
|---|---|---|
| Raw 50/50 weighted sum | Fully compensatory and sensitive to independently trained raw-logit scales | Scale-sensitivity baseline only |
| Frozen-normalized weighted sum | Removes scale dominance but remains fully compensatory | Total-burden comparator, not named balanced steering |
| Frozen-normalized hard maximum | Strict worst-residual order; non-worst evidence is often locally invisible | Primary strict control |
| Signed maximum plus unbounded mean augmentation | A sufficiently favorable negative coordinate can offset a worse active maximum without bound | Retired as the primary law |
| Reference-point achievement scalarization | Measures weighted distance or shortfall from declared aspiration/ideal coordinates | Appropriate when a scientifically meaningful allele-wise target exists |
| Frozen-normalized smooth maximum | Bounded relaxation of the hard maximum with one continuous, strictly monotone scalar | Primary Dual-V2 law |
| Epsilon constraint | Optimizes one objective while hard-bounding another | Asymmetric named-protection or terminal control |
| Lexicographic or tie-band ordering | Enforces priority but introduces hard switching; naive pairwise tie bands may not define a transitive global order | Conservative sensitivity control, not the primary write law |
| Frozen percentile coordinate | Affine-scale invariant but compresses tails and creates finite-reference ties | Offline sensitivity analysis |
| Current-cloud rank coordinate | Depends on the candidate set and changes across depth or root | Incompatible with frozen donor/write reproducibility |

The retired signed augmentation fails for structural rather than numerical reasons. For any fixed
$\rho>0$, an endpoint with coordinates $(\delta,-M)$ can outrank $(0,0)$ under
$\max(u_A,u_B)+\rho(u_A+u_B)/2$ when $M$ is sufficiently large, even though its worst coordinate
has increased from $0$ to $\delta>0$. Standard augmented achievement functions avoid this exact
failure by measuring deviations from a declared ideal or reference geometry rather than attaching
an unbounded signed mean to centered coordinates.

### A.2 Literature transfer boundaries

Smooth Tchebycheff directly motivates the log-sum-exp scalarization and its bounded approximation
to the maximum [3]. Its gradient-convergence and Pareto-completeness results do not transfer to a
finite black-box endpoint cloud; Dual V2 uses only the scalar geometry. MOEA/D demonstrates how
ideal/reference coordinates, objective scaling, and multiple preference subproblems interact [4],
but its dynamic population decomposition is not part of the minimal method.

Hong and Kortemme use NSGA-II/III non-dominated sorting and diversity-preserving environmental
selection rather than a donor scalar [2]. Their work supports retaining exact vector evidence and
Pareto analysis, but it does not provide the total order required by one V2 lineage. MosPro composes
predictor gradients to obtain locally balanced mutation directions [1], which does not transfer to
black-box exact-endpoint ordering. MOG-DFM uses candidate-local rank normalization, directional
alignment, and hypercone filtering [6]; those ideas motivate conditional action filtering, not a
frozen complete-endpoint objective.

---

## Appendix B. Selected references

1. Luo J, Ding K, Luo Y. Pareto-optimal sampling for multi-objective protein sequence design.
   *iScience*. 2025. https://doi.org/10.1016/j.isci.2025.112119
2. Hong L, Kortemme T. An integrative approach to protein sequence design through multiobjective
   optimization. *PLoS Computational Biology*. 2024.
   https://doi.org/10.1371/journal.pcbi.1011953
3. Lin X, et al. Smooth Tchebycheff scalarization for multi-objective optimization. *ICML*. 2024.
   https://proceedings.mlr.press/v235/lin24y.html
4. Zhang Q, Li H. MOEA/D: A multiobjective evolutionary algorithm based on decomposition.
   *IEEE Transactions on Evolutionary Computation*. 2007.
   https://doi.org/10.1109/TEVC.2007.892759
5. Kong Z, et al. MP2D: Constrained Monte Carlo tree-guided diffusion for multi-objective protein
   sequence design. Accepted at *IJCAI-ECAI*. 2026. https://arxiv.org/abs/2605.05829
6. Chen T, et al. Multi-objective-guided discrete flow matching for controllable biological
   sequence design. arXiv:2505.07086. 2025. https://arxiv.org/abs/2505.07086
7. Jain M, et al. Multi-objective GFlowNets. *ICML*. 2023.
   https://proceedings.mlr.press/v202/jain23a.html
8. Schubert B, et al. Population-specific design of de-immunized protein biotherapeutics.
   *PLoS Computational Biology*. 2018. https://doi.org/10.1371/journal.pcbi.1005983
9. Salvat R, et al. Mapping the Pareto optimal design space for a functionally deimmunized
   biotherapeutic candidate. *PLoS Computational Biology*. 2015.
   https://doi.org/10.1371/journal.pcbi.1003988
10. Southwood S, et al. Several common HLA-DR types share largely overlapping peptide-binding
    repertoires. *Journal of Immunology*. 1998. https://pubmed.ncbi.nlm.nih.gov/9531296/
11. Greenbaum J, et al. Functional classification of class II HLA reveals seven supertypes and
    repertoire sharing. *Immunogenetics*. 2011.
    https://doi.org/10.1007/s00251-011-0513-0
12. Singhal R, et al. A general framework for inference-time scaling and steering of diffusion
    models. *ICML*. 2025. https://proceedings.mlr.press/v267/singhal25b.html
13. Kshirsagar M, et al. RosettaSearch: Multi-objective inference-time search for protein sequence
    design. arXiv:2604.17175. 2026. https://arxiv.org/abs/2604.17175

New measurements should first enter a machine-readable RAR. Commands, schemas, implementation
interfaces, launch matrices, and numerical calibration values belong in a future PLAN or runbook,
not in this scientific document.
