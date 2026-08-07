# Temporary Working Note: Fusion V2 Algorithmic Gaps and Directional Feedback

> **Status:** discussion draft, not an authority document and not an execution plan.
>
> **Purpose:** record the algorithmic gap exposed after the S7 transmission result and define the
> smallest scientifically coherent direction for a production feedback operator. This note does
> not replace `doc/FUSION_V2.md`, `PLAN_RF_REFINE_FUSION_V2.md`, or the cluster runbook.

## 1. Current state

S7 established a real but deliberately narrow result: a one-residue endpoint intervention can
survive projection and alter the descendant sequence distribution. This demonstrates **feedback
channel capacity**. It does not establish immune directionality or recursive optimization.

The current cycle has the following shape:

1. capture one live partial source state;
2. generate $K$ complete lookahead endpoints;
3. score every complete endpoint with the exact Head and definitive structure gate;
4. rank feasible endpoints by global Head risk;
5. pass one selected endpoint to a diagnostic support policy;
6. project the resulting support assignment into a re-entry state and propagate again.

The relevant implementation properties are:

- the Head already provides residue-level risk, in addition to global and window-level evidence;
- endpoint selection is Head-directed at the **complete-sequence** level;
- `StateDerivedProbePolicy` is diagnostic-only and does not use residue-level Head risk;
- its endpoint write is selected by source-state legality and index order, while reopen identities
  are selected from temporal history and the calibrated maturity envelope;
- the archive is monotone in exact feasible Head risk, but the active lineage is not required to
  improve before another feedback transition is opened; and
- non-anchor protection is segment-local. The kernel already permits a later policy to reopen a
  previously carried or endpoint-written position after protection expires.

Therefore the current implementation can answer **"does feedback transmit?"**, but it is not yet
a scientifically defined immune-directed optimizer.

## 2. Core gaps

### 2.1 Directionality gap

Selecting a low-risk complete endpoint is insufficient if the projection policy does not identify
which endpoint identities contributed useful evidence and which positions remain problematic. The
current diagnostic policy may transmit a token that is unrelated to the endpoint's Head advantage.

Residue-level Head risk solves the localization interface, but it is not itself a token proposal or
a causal gradient. A production policy must combine:

- exact endpoint identities from actually generated complete sequences; and
- residue-level evidence that prioritizes where to adopt, protect, reopen, or reject.

### 2.2 Accumulation gap

Increasing feedback depth does not by itself create cumulative optimization. A sequence of
transmissible but non-directional transitions is only repeated stochastic search.

Three claims must remain separate:

1. **proposal direction:** the selected complete endpoint is better than the lineage reference;
2. **transition direction:** descendants have a favorable expected Head shift under the feedback
   policy; and
3. **system non-regression:** the best exact feasible endpoint returned by the run never worsens.

The third claim is already supported by the monotone archive. The first should be enforced by the
lineage acceptance rule. The second is empirical and cannot be guaranteed for every stochastic
descendant.

### 2.3 Re-editability gap

The kernel supports re-editing of every non-anchor position, but the diagnostic policy does not
use new Head evidence to decide when a previously protected identity should be reopened.

Production semantics should distinguish:

- **hard anchors:** permanent constraints, never reopened;
- **soft protected identities:** protected for one propagation segment because the current
  accepted endpoint provides favorable evidence;
- **carried identities:** retained by default but not permanently locked; and
- **reopen candidates:** positions whose current residue risk, worsening, or uncertainty justifies
  renewed generative search.

Soft protection must be a reversible commitment. Only hard anchors are permanent.

## 3. Proposed minimal production operator

Working name: `ResidueRiskFeedbackPolicy`.

This is intentionally a minimal exploitation-oriented operator. A separate exploratory population
or valley-crossing lane can be considered later; it is not required to establish the first
immune-directed transition.

### 3.1 Bind a complete lineage reference

Each live partial lineage must bind one exact, complete, feasible reference endpoint $E_d$. This is
the last accepted complete endpoint for that lineage, not an inferred score on a masked state.

At depth zero, the complete reference must be frozen explicitly before endpoint outcomes are
inspected. The exact depth-zero binding remains an open design decision in the authority document.

### 3.2 Generate and evaluate complete candidates

From the live source $P_d$, generate $K$ complete endpoints and evaluate all of them with the same
identity-bound Head and safety contract. Let $Y_d^*$ be the best feasible endpoint under the frozen
global Head ordering.

An endpoint may become a feedback donor only if it improves on $E_d$ under a predeclared strict or
tolerance-aware acceptance rule. If no endpoint improves, the lineage records a typed stall and
retains $E_d$; a worse endpoint must not overwrite the lineage reference.

All raw endpoints remain in the archive regardless of the acceptance decision.

### 3.3 Construct residue-level evidence

For each editable residue $i$, compare the complete reference and selected endpoint using the same
Head binding:

$$
g_i = \left[h_i(E_d) - h_i(Y_d^*)\right]_+
$$

$$
w_i = \left[h_i(Y_d^*) - h_i(E_d)\right]_+
$$

where $g_i$ is local improvement evidence and $w_i$ is local worsening evidence. The policy should
also retain:

- residual risk $h_i(Y_d^*)$;
- disagreement across the best feasible sibling endpoints; and
- source instability, commit history, and active sampler confidence.

These fields are prioritization evidence, not claims that a residue is causally responsible for a
global Head change.

### 3.4 Assign support with deterministic priority

Within the mask cardinality permitted by the calibrated $\mathcal{B}(r_d)$ envelope:

1. **Hard anchors:** preserve unconditionally.
2. **Safety repair:** reject an endpoint with an unrepairable whole-landscape hotspot; otherwise
   prioritize reopening editable positions responsible for new or worsened burden.
3. **Endpoint adoption:** write and temporarily protect endpoint identities with strong positive
   $g_i$, preferably supported by sibling consensus and an actual token change.
4. **Residual search:** reopen positions with high residual risk, positive $w_i$, endpoint
   disagreement, or source instability.
5. **Carry:** retain the remaining stable source identities.

The schedule determines **how many** positions may be reopened. The policy determines **which**
positions. This avoids restoring global D3 remask as an uncontrolled competing mechanism.

The endpoint-write cardinality must not remain fixed at one. A minimal adaptive law is:

$$
m_d = \min\!\left(
    n_{\mathrm{legal\ beneficial}},
    m_{\mathrm{declared\ cap}},
    m_{\mathrm{band\ compatible}}
\right)
$$

where a legal beneficial site is a temporally admissible endpoint-write position with favorable
residue-level evidence under the frozen policy. Increasing endpoint writes can be paired with the
corresponding reopen count required to keep realized unresolved mass inside $\mathcal{B}(r_d)$;
carrying more endpoint information and preserving generative plasticity are therefore not
intrinsically opposed. If $m_d=0$, the transition stalls instead of writing an arbitrary token.

**Provisional implementation decision.** Freeze the first write cap at
$\lceil 0.05N_{\mathrm{editable}}\rceil$: approximately `5` positions for `5ZHV_B` and `14` for
`Q00511`. This is a cap, not a quota; write only legal positions with positive residue-risk benefit,
and stall if none exist. Choose the realized unresolved target at the center of
$\mathcal{B}(r_d)$, then set

$$
m_{\mathrm{reopen}} = u_{\mathrm{target}} - u_{\mathrm{src}} + m_d
$$

and assign those reopen slots to the highest residual-risk, worsened, uncertain, or unstable
non-anchor positions. The one-token probe is retired from scientific use. The only required policy
control is the same donor, cardinality, horizon, and seeds with positions selected by the former
source-geometry rule instead of residue-risk evidence; no dose ladder is required initially.

### 3.5 Make protection reversible

Endpoint-written positions receive protection only through the next propagation segment. At the
next complete evaluation, every non-anchor position is eligible for reassessment.

If a new epitope appears in a previously protected region:

- a non-anchor position may be reopened in the next cycle;
- a hard-anchor position remains immutable, but editable neighboring context may be reopened; and
- a lineage that cannot repair the burden without violating anchors is rejected rather than
  silently preserving the hotspot.

### 3.6 Preserve two levels of memory

The algorithm needs both:

- **lineage memory:** the accepted complete endpoint $E_d$ that defines the next residue-risk
  comparison; and
- **global archive memory:** every exact endpoint plus a monotone feasible elite.

The lineage reference changes only on an accepted improvement. The global archive ensures that a
bad stochastic descendant cannot erase the best design already discovered.

## 4. What can and cannot be guaranteed

The production operator can enforce:

- exact Head and safety evaluation before ancestry;
- no worsening update of the lineage reference;
- reversible soft protection of non-anchor positions;
- whole-landscape re-evaluation after every complete endpoint;
- typed stall when no useful donor exists; and
- monotone non-regression of the returned archive elite.

It cannot guarantee that every descendant improves. Diffusion remains stochastic and residue-level
Head risk is not a causal gradient. Reward directionality means a favorable **expected descendant
shift under matched conditions**, not deterministic improvement of every fork.

## 5. Why a diagnostic $D=2$ run is not the next algorithmic proof

A two-cycle run under `StateDerivedProbePolicy` would mostly show that a transmissible diagnostic
intervention can be repeated. It would not establish that deeper V2 accumulates immune benefit.

Conversely, a short recursive run cannot prove the final long-trajectory capability of V2. Depth
should be studied only after the local transition operator is scientifically defined.

The appropriate order is therefore:

1. freeze and implement the minimal residue-risk production policy;
2. qualify one-cycle reward directionality under matched source, support, horizon, and fork seeds;
3. treat that qualification as a local operator sanity check, not a system-level capability claim;
4. only then study progressive depth over a sufficiently long trajectory and report the complete
   archive path rather than interpreting $D=2$ as the final result.

## 6. Minimal qualification questions

Before recursive capability work, the implementation should answer:

1. Does a better exact endpoint produce a different residue-priority assignment than a worse but
   compatible endpoint?
2. Does the source-aware policy outperform a support-cardinality-matched endpoint-only policy in
   descendant Head direction?
3. Can a previously endpoint-written, non-anchor position be reopened when its later residue risk
   becomes high?
4. Does the lineage stall rather than adopt a globally worse donor?
5. Does the archive elite remain unchanged when all new descendants are worse?
6. Are new whole-landscape hotspots either repaired through editable support or rejected?

Passing these questions would establish an immune-directed local operator. It would not yet prove
that recursive V2 outperforms A2 or v0 at scale.

## 7. Evidence update: one-cycle reward-ordering reanalysis

Source: cluster runbook Section 7.11, commit `a7a0a8c` (2026-08-07). This is a reanalysis of the
existing mechanism cohort, not a new production-policy experiment.

### 7.1 What was compared

For each compatible source prefix, the analysis compared descendants produced from:

- arm A: the lowest-global-risk admissible endpoint (`endpoint_rank=0`); and
- arm B: the second-lowest-global-risk admissible endpoint (`endpoint_rank=1`).

Source state, endpoint pool, descendant seeds, propagation horizon, and support cardinality were
matched. The two projected states differed at zero or one position. Structural zeros were excluded;
contrastable pairs therefore differed by exactly one endpoint-written token.

Across the four protein/re-entry cells, every paired descendant-risk confidence interval crossed
zero. The reported regressions of descendant score difference on complete-parent score difference
had slope magnitudes at most `0.064` and $R^2$ at most `0.016`. The descriptive comparison of a
single-cycle best-of-four against its own depth-zero best-of-four was protein-split and was not
compute matched.

### 7.2 Supported interpretation

This result is direct evidence for the gap described in Sections 2 and 3:

- exact global endpoint ranking alone did not make the current one-token diagnostic projection
  reward-directional;
- the current `StateDerivedProbePolicy` transmitted identity but did not demonstrate a favorable
  descendant Head shift;
- running $D=2$ with the same diagnostic policy would repeat an unqualified local operator and is
  not the next algorithmic proof; and
- the null is mechanistically expected because the written position is selected from source mask
  geometry, while neither write nor reopen identity uses residue-level Head evidence.

The runbook verdict `not_demonstrated` is therefore appropriate **for the current diagnostic
policy**. The result does not show that the feedback mechanism is harmful.

### 7.3 What the single-token probe established

In `endpoint_change`, the support partition contains many carried, injected, and reopened
positions, but those operations are matched across the two arms. The only endpoint-dependent input
is the amino acid at `write_from_endpoint`, and the matched projected states therefore differ at
zero or one token.

That single token is not inert: S7 showed that it can amplify into approximately 2.4--14.1 residue
differences in the complete descendants. The experiment therefore established a real conditioning
channel. However, because the written position was selected by source mask geometry rather than
Head evidence, its immune direction was effectively uncontrolled.

The single-token probe has consequently completed its intended scientific role:

- it demonstrated channel capacity and downstream amplification;
- it did not define a production reward operator;
- a null reward-ordering result under this probe does not reject residue-directed feedback; and
- fixed one-token endpoint-write cardinality should be retired from production policy rather than
  carried into a deeper recursion experiment.

Production feedback should transmit an adaptive multi-position set of legally writable,
residue-supported endpoint identities while separately reopening residual or worsened positions.
The cardinality remains constrained by the re-entry maturity band and an explicit cap; it is not
maximized without evidence.

### 7.4 Interpretation that is not supported

The complete-endpoint global Head gap is not the treatment dose applied at projection. The actual
intervention is one token at one source-selected position, while the global gap between rank-0 and
rank-1 endpoints is produced by their entire sequence and context. Consequently:

- a rank-0 endpoint can be globally better while its token at the written position is locally
  neutral or worse than the rank-1 token;
- the sign of the global endpoint ordering does not guarantee the sign of the applied positional
  intervention;
- the regression slope cannot be interpreted literally as the fraction of the full endpoint reward
  gap transmitted through that one token; and
- this analysis cannot exclude directionality from a policy that deliberately chooses positions
  using residue-level benefit, residual burden, worsening, and endpoint consensus.

Thus the strongest justified statement is:

> The current source-geometry-selected, one-token diagnostic channel did not transmit the global
> reward ordering of its donor endpoints.

It is too strong to conclude that a residue-directed V2 reward channel is intrinsically inert or
that increasing lookahead breadth could not help a different support policy.

### 7.5 No-generation local-dose diagnostic from the existing artifacts

The required evidence was recoverable without new generation by joining the original mechanism
bundles:

- `complete_endpoints.parquet`: endpoint identity, sequence, global Head risk, and the complete
  residue-hotspot vector in `head_score_json`;
- `feedback_events.parquet`: selected endpoint, arm, pair/source identity, and the actual
  `write_from_endpoint` position; and
- `mechanism_contrasts.parquet`: matched descendant endpoint identities.

For the actually written position $i$, the diagnostic computed:

$$
\Delta h_i^{\mathrm{write}} = h_i(Y_{\mathrm{rank0}}) - h_i(Y_{\mathrm{rank1}})
$$

where lower residue hotspot is better, so a negative value favors rank 0 locally. Results on the
contrastable one-token source pairs were:

| cell | $n$ sources | rank 0 locally better | median $\Delta h_i^{\mathrm{write}}$ | median local percentile among endpoint token differences | written position in best local quartile | $\rho$(parent global gap, written local gap) | $\rho$(written local gap, descendant global gap) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5ZHV_B`, $r=30$ | 39 | 43.6% | +0.0418 | 0.600 | 15.4% | -0.073 | -0.098 |
| `5ZHV_B`, $r=40$ | 38 | 44.7% | +0.0371 | 0.588 | 15.8% | -0.048 | +0.354 |
| `Q00511`, $r=30$ | 39 | 41.0% | +0.0577 | 0.534 | 15.4% | +0.099 | -0.186 |
| `Q00511`, $r=40$ | 39 | 41.0% | +0.0577 | 0.534 | 15.4% | +0.099 | -0.127 |

The $r=30$ and $r=40$ cells may reuse the same complete source/lookahead evidence, so they are not
pooled as independent observations. The local percentile ranks the written position among all
positions where the two donor endpoints carry different tokens; it is an informative diagnostic,
not the exact set of positions legally writable from the realized partial source.

The result is nevertheless decisive about the current treatment definition:

- globally rank-0 was locally better at the actual written position in fewer than half of the
  contrastable sources in every cell;
- the median written-position delta was positive in every cell, so the globally better endpoint
  was typically slightly worse at the only position the projection copied;
- complete-endpoint global ordering was essentially unrelated to written-position local ordering;
  and
- the relationship between the written local gap and descendant global gap was inconsistent across
  cells rather than a stable reward-transmission law.

Thus the current experiment did not apply a reliably reward-ordered positional treatment. Its null
descendant result cannot distinguish "a locally beneficial token was forgotten" from the more basic
fact that the copied token was not selected to be locally beneficial. This directly quantifies why
the diagnostic support law is reward-blind. It does not validate the future production policy.

### 7.6 Consequence for the future qualification contract

A future reward-directionality test must order both levels of the intervention:

1. the donor endpoint must be better than the lineage reference under exact complete Head; and
2. the written or reopened support must have a declared residue-level reward rationale.

Merely comparing descendants of globally rank-ordered endpoints is insufficient when the projected
support is selected independently of the Head. The treatment record must persist the actual local
reward evidence at every written/reopened position so that the applied dose, rather than the full
donor sequence gap, is auditable.

## 8. Open decisions to move into the authority after discussion

- the frozen complete reference used at depth zero;
- the donor-improvement tolerance or margin and exact tie law;
- whether residue evidence comes only from $Y_d^*$ or includes consensus across the top feasible
  sibling set;
- the deterministic combination of residual risk, worsening, sibling disagreement, and source
  instability for reopen priority;
- the behavior when local residue improvements coexist with a worse global Head score;
- whether the first production version is strictly exploitative or includes a separately audited
  exploration lane; and
- whether the declared system output is always the archive elite or may include a final active
  lineage state as a secondary artifact.

Until these decisions are frozen, S7 should be interpreted as successful transmission of a
diagnostic channel, not as completion of the V2 optimization algorithm.

## Appendix A. Short-path Head-directed feedback candidate

This appendix freezes a deliberately small candidate for the next implementation and qualification
step. It is a working specification, not an authority update and not a claim that the production
feedback policy has been validated. Its purpose is to change exactly one scientific variable after
S7: support selection becomes Head-directed while the projection kernel and propagation horizon
remain unchanged. Write cardinality follows the capped law frozen in Section 3.4. A dose ladder,
sibling consensus, exploration, block attribution, and recursive depth are explicitly out of scope
for this short path.

### A.1 Interpretation of S7

S7 established channel capacity: an endpoint-derived identity written into the projected state can
change the distribution of later complete descendants. The diagnostic write location was
deterministic rather than random, but it was selected without Head evidence. Head chose the donor
endpoint; it did not choose the written position or the reopened position. The S7 null result for
reward ordering therefore does not show that V2 cannot transmit value. It shows that the current
one-token, Head-blind support law did not transmit the donor endpoints' global ordering.

The next policy must distinguish three guarantees. Donor direction requires feedback to originate
only from an exact, feasible endpoint that improves the current lineage incumbent. Applied-dose
direction requires the actual written identity and reopened support to carry declared local Head
evidence. Archive non-regression requires the best feasible exact endpoint observed so far to remain
available as the system output. None of these guarantees implies that every stochastic descendant
must improve; the response claim remains a shift in the matched descendant distribution.

### A.2 Frozen identities and donor gate

The short-path policy uses three non-interchangeable complete-sequence references. The immutable
cumulative safety reference $\bar y$ defines the whole-landscape new-hotspot ratchet and never
updates. The lineage reward incumbent $I_d$ is the best accepted exact feasible endpoint for the
current family. The current donor $Y_d^*$ is the endpoint proposed to supply feedback at depth $d$.

The donor may open directional feedback only after definitive structure, anchor preservation, and
cumulative-hotspot admissibility have already passed, and only when its frozen complete-sequence
Head risk improves the incumbent by more than the frozen tolerance $\epsilon_R$:

$$
R_H(Y_d^*) < R_H(I_d) - \epsilon_R.
$$

The tolerance must be calibrated from the frozen Head's repeatability or numerical noise floor; it
must not be invented from the desired experimental effect. If no donor passes this gate, the policy
emits a typed `stall_no_better_donor` outcome. It must not fall back to an arbitrary endpoint token,
and an unsafe endpoint cannot become ancestry on the promise that a later feedback cycle may repair
it.

### A.3 Head-directed capped write

A legal candidate position must be editable, non-anchor, unresolved in the live source, carry a
donor identity different from the incumbent identity, and lie in an aligned Head window that
improves in the donor relative to the incumbent. Local window evidence must be reconstructed on one
common score scale from the stored raw aligned-window Head outputs. Cross-sequence subtraction of a
separately centered or clipped residue-hotspot vector is not sufficient evidence.

For each shortlisted position $i$, the frozen Head should score a donor-background leave-one-out
counterfactual in which only the donor identity at $i$ is reverted to the incumbent identity:

$$
a_i = R_H\!\left(Y_d^*{}_{i\leftarrow I_{d,i}}\right) - R_H(Y_d^*).
$$

A positive $a_i$ means that removing the donor identity worsens the donor under the frozen Head.
This is exact Head-model contribution evidence in the donor context; it is not a claim of biological
causality. The policy writes the top $m_d$ legal positions by $a_i$ above the frozen local
tolerance, using residue index only as the final deterministic tie-break, where:

$$
m_d
=
\min\!\left(
    n_{\mathrm{legal\ positive}},
    \left\lceil0.05N_{\mathrm{editable}}\right\rceil,
    m_{\mathrm{band\ compatible}}
\right).
$$

The `0.05` term is a cap rather than a quota. If $m_d=0$, the short-path policy stalls rather than
writing an arbitrary token. It may not silently widen to a block rule. A later block-level
extension may address epistatic improvements only after this capped policy has been qualified.

### A.4 Head-directed band-matched reopen

The reopen channel answers a different question from the write channel. It should target editable,
source-resolved, non-anchor positions that lie in at least one of three Head-defined regions: a
residual high-burden window in the donor, a window that worsened from $I_d$ to $Y_d^*$, or a new
hotspot relative to the immutable cumulative safety reference. The priority order for the short
path is new-hotspot membership, donor-versus-incumbent worsening, donor residual burden, active
uncertainty, source temporal instability, and finally residue index. This uses Head to decide where
new generative freedom is needed without asking Head to invent the replacement token.

Previously adopted endpoint identities are not permanent anchors. Their temporary protection still
expires at the next checkpoint. In a later cycle, a previously written non-anchor position becomes
reopen-eligible when it lies in a residual or worsened Head window and no longer has positive frozen
contribution evidence. This supplies context-dependent re-editability without restoring global D3
remask. Hard anchors remain permanently protected.

The existing exact $\mathcal B(r)$ solver remains authoritative. Its calibration artifact provides
an explicit integer center target $u_{\mathrm{target}}$ and deterministic rounding/tie law. For
source unresolved count $u_{\mathrm{src}}$, the required reopen count is:

$$
m_{\mathrm{reopen}}
=
u_{\mathrm{target}}-u_{\mathrm{src}}+m_d.
$$

The transition fails closed if this count is negative, exceeds legal reopen support, or does not
reproduce the target exactly. All remaining positions continue to obey the frozen rollback-time
carry/inject law, assimilation scoring, matched-seed law, and temporary-protection semantics.

### A.5 Required evidence record

Every directional transition must preserve enough evidence to reconstruct why each support position
was chosen. The record must bind the source, incumbent, immutable safety reference, donor, Head
identity, and policy digest. For the write it must store the aligned-window rationale, donor and
incumbent tokens, $a_i$, selection rank, temporal legality, and band contribution. For the reopen it
must store residual burden, worsening evidence, new-hotspot membership, source-history tie-breaks,
selection rank, temporal legality, and band contribution. A typed stall must preserve the rejected
candidate counts and rejection reasons rather than emitting a reward-blind fallback event.

### A.6 Minimal qualification sequence

The first step is an offline policy replay over the existing S7 complete endpoints and source
states. It should measure the fraction of sources with at least one legal positive write, the
counterfactual contribution distribution, realized $m_d$, overlap with the old diagnostic write
positions, required/legal reopen counts, and exact band-target coverage. This is a coverage and
executability check, not evidence of descendant improvement.

The next cluster experiment remains one cycle and compares the short-path Head-directed policy with
the former source-geometry position law under the same source, donor, realized write/reopen
cardinalities, re-entry/checkpoint horizon, band target, and fork seeds. The control selects the
lowest-index legal source-mask writes and latest-commit legal source-resolved reopens, truncated to
the treatment cardinalities. The primary response endpoint is the paired shift in
complete-descendant Head risk. Descendant sequence distance remains a mechanism secondary endpoint,
while structure, anchors, cumulative-hotspot safety, and archive non-regression remain hard
invariants. The scientific treatment is the identity of the written and reopened support, not the
global rank gap between two complete donors.

Only a positive response gate authorizes a dose ladder, block-level attribution, sibling consensus,
an exploration lane, or recursive $D=2$ execution. A null result under verified positive local
doses would be materially stronger than the S7 null: it would show that locally Head-favorable
capped interventions can propagate sequence identity without detectably shifting the downstream
Head distribution under the tested substrate and horizon.
