# Enzyme De-immunization via Reference Flow — Design Notes (Uricase)

This is the scientific design doc for extending the MHC-IF Reference Flow (RF)
toolchain to **enzymes**: de-immunize a wild-type enzyme while preserving its
function, using active-site constraints on top of the existing position-dependent
RF. Uricase is the working instance. The doc integrates a multi-session Thinker
discussion and is the source for a later implementation PLAN; it is not itself an
implementation plan.

**Status:** design frozen AND implemented. The RF inpainting hook landed
(`inverse_folding/reference_flow/constraints.py` + `run_if_phase_c1.py
--constraint-manifest`, per `PLAN_URICASE_ENZYME_MODE.md`); Appendix A documents the
*pre-implementation* code surface. The full-pool per-protein constraint manifest is
materialized — `configs/uricase_q00511_active_site_perprotein_v0.yaml` (6387 deduped
proteins, 49378 anchors projected from Q00511 by `scripts/build_uricase_active_site_manifest.py`,
all `validate_against_sequence`-clean). Codebase claims and uricase biology were
cross-checked at file:line and against M-CSA entry 118 / primary literature
(2026-06-21); a review pass tightened the locked-floor, proxy-typing, calibration,
routing, and residual-diagnosis claims (2026-06-22); a workflow-verified pass over
Kaiyi's `data/Uricase/` evolutionary analysis grounded the §7 anchor set, conservation
tiering, and immune landscape and corrected the caseset provenance (2026-06-23); the
per-protein manifest + deduped caseset were built and verified (2026-06-24).

## TL;DR — current definition

> **Enzyme mode v0 = active-site-constrained *global* sequence redesign by
> default.** Generation uses only the WT backbone, the active-site motif, and the
> structural posterior. It makes **zero changes to RF controller semantics and
> actuation logic (D1–D3, SC-GR, control law, DPLM decoder)** and adds only
> **sampler-owned hard-constraint state** (anchor init + permanent remask
> protection); it does **not** claim to sense enzyme function online. The full WT
> sequence never enters generation. WT lives only in the post-hoc decision layer —
> function comparator (run through the *same* proxy pipeline), no-op candidate,
> last-resort rescue. The function proxy is a **typed vector** in which each
> coordinate carries an epistemic status (hard invariant / calibrated gate /
> ranking-only / unavailable), not a scalar; **v0 evaluates only `f_anchor` + monomer
> `f_fold`** (assembly / active-site geometry need a multimer predictor and are
> deferred). Local polish and WT-rescue are data-driven branches, not default stages.
> Shell control, reachability-aware gain,
> and routing thresholds are **deferred** until the first run shows the empirical
> distribution.

The two contracts that gate the first run (§9): the uricase `ActiveSiteSpec` and
the `F_post` v0 measurement contract.

---

## 1. Objective and the constrained-optimization spine

Long-term goal: take a wild-type enzyme that is immunogenic (a real liability for
biologic enzymes — uricase is clinically PEGylated *because* of anti-drug
antibodies) and redesign its sequence on the fixed WT backbone to lower
immunogenicity **without losing catalytic function**.

De-immunization is useful for any administered biologic, but the *position-
dependent* RF contribution bites specifically for **enzymes and binders**: these
are the proteins with a localized functional sub-region (active site / paratope)
that must be preserved while the rest is edited. Enzymes differ from de novo
binders decisively — a binder's structure *is* the deliverable (global hard target,
sequence is a means to fold), whereas an enzyme already exists: WT sequence +
structure, and structure is *instrumental* to holding the active site. So the
enzyme constraint is **graded around the active site plus a global fold-stability
floor**, not a global structure guarantee.

The right frame is **constrained optimization**:

$$
\min_{x_{\text{edit}}}\ \rho_{\text{epi}}(x)
\quad\text{s.t.}\quad
x_{\text{AS}} = \text{WT}\ (\text{hard}),\quad
\text{Fold}(x) \gtrsim \text{Fold}(\text{WT})\ (\text{soft})
$$

- **Objective** = total immune load $\rho_{\text{epi}}$, a *global, position-flat*
  quantity (§2 — an epitope anywhere is a liability; location does not reweight
  immune badness).
- **Variable** = the editable residues $x_{\text{edit}}$.
- **Constraints** = function: active-site anchors fixed (hard) + global fold floor
  (soft).
- **WT** is the reference for *both* axes: immune baseline to beat, function floor
  not to fall below.

**The active site restricts the legal sequence space; it does not partition residue
burden into removable vs immutable.** This is a hard line. MHC-II risk is a property
of a peptide *window*, not a residue: a window covering a hard anchor is *not*
automatically unfixable, because changing other editable residues in the same window
can still break the binding register. So a "fixed-position burden" is at most spatial
**attribution telemetry**, never the immutable immune floor. The true locked floor is
a bound over the legal space:

$$
R_{\text{floor}}(S, A) = \inf_{x \in \mathcal{X}_{\text{legal}}(S, A)} R(x),
\qquad
\mathcal{X}_{\text{legal}}(S, A) = \{\, x : x_A = x_A^{\text{WT}},\ x \text{ structurally legal} \,\}
$$

A locked floor emerges only when **no legal combination of edits** can reduce a risky
window. v0 neither knows nor estimates this bound (it is essentially the
de-immunization problem itself); it only needs to respect the constraint and report
attribution. A metric must nonetheless be able to *decline*: when residual risk
cannot be reduced under the constraints, the honest output is "this enzyme cannot be
de-immunized by sequence redesign" — the epitope∩active-site hard case.

## 2. Problem structure: the immune / function asymmetry

The objective and the constraint live in fundamentally different spaces. This
asymmetry is the spine of the architecture.

| | Immune (objective) | Function (constraint) |
|---|---|---|
| Determinant | linear peptide → MHC-II groove | catalytic 3D geometry, electrostatics, dynamics, substrate channel |
| Range | sequence-local, bounded (~9–15mer window; mutation at $i$ touches only windows covering $i$) | structure-global, unbounded; epistatic (a distal mutation can move the backbone and kill activity) |
| Predictor | head in-loop, NetMHCIIpan external — both sequence-only, no structure | no cheap accurate sequence→function predictor; only structural proxy / physics / assay |
| Availability | computable **before** generation; per-position marginal is local | only **after** generation, via an imperfect proxy |

Consequences:

1. The objective can be navigated **analytically** (this is *why* RF steers it
   per-position); the constraint can only be **sampled and checked**. The natural
   algorithm is **generate-then-test**: optimize immune in the loop, filter on
   function after.
2. Gate redesign on immune **a priori** (headroom known before generating); gate on
   function only **a posteriori**. Any full→targeted fallback is
   **function-triggered**, not immune-triggered.
3. Fixing the anchors' *identity* does **not** fix their *geometry* — geometry is
   produced by folding the whole chain including the mutable context. This requires
   (a) a protection field extending beyond catalytic residues (a shell, decaying
   with structural distance) and (b) a genuine function proxy, and it is why a global
   fold floor is non-negotiable.

## 3. WT's role: three uses and the firewall

Split "use WT" into three distinct objects:

1. **WT backbone** $S_{\text{WT}}$ — the inverse-folding structural condition. Always
   used.
2. **Active-site motif** $M_{\text{active}}$ — catalytic residue identity / position
   / geometry. A *task definition*, not a WT sequence prior.
3. **Full WT sequence** $x_{\text{WT}}$ — the object the main generation path must
   **not** read.

The main path uses $S_{\text{WT}} + M_{\text{active}}$ but not
$x_{\text{WT}} \setminus M_{\text{active}}$ — **motif-constrained global redesign**,
not WT mutation.

**Why WT stays out of generation.** Empirically, current RF already beats DPLM-native
outright and slightly beats WT on high-burden proteins *without seeing WT* and without
inpainting. So on the immune axis WT is not the reference — the reference is the
**structural posterior**; WT is one (immunogenic) sample in a large feasible set.
Anchoring generation to WT caps the search inside WT's basin and **weakens the
contribution** ("constrained global redesign that beats WT while preserving function"
is strong; "small edits on WT" is weak and matched by simple methods). Consistent with
the repo's firewall — WT/NoD-relative quantities are **attribution references, not
runtime objects** (`RF_Controller_Architecture.md:350,525`; `Self-Cond_GR.md:137`:
"WT stays attribution only… never a runtime classifier").

WT enters **the decision layer only**: function comparator (same proxy pipeline),
no-op candidate, last-resort rescue substrate (§6). It is **not** a main-path initial
sequence, SC-GR reference, D1 hotspot prior, or D2 candidate prior. Post-hoc
comparison does not shrink the search space, so it is not cheating; WT-guided
*generation* would be.

## 4. Enzyme mode v0

Default and only v0 mode: **all-mask-except-hard-anchors global redesign**, on the
existing RF, with the constraint owned by the **sampler**.

$$
x_T^i =
\begin{cases}
x_{\text{active}}^i & i \in A_{\text{hard}} \\
[\text{MASK}] & i \notin A_{\text{hard}}
\end{cases}
$$

v0 specifics (deliberately minimal):

1. hard anchors are **permanently committed** tokens from step 0;
2. they never enter first-unmask, D2 selection, D3/reparam remask, or schedule
   allocation;
3. all other positions stay in global redesign;
4. SC-GR, D1, D2, D3 keep their current behavior;
5. **no** function-aware online gain;
6. **no** shell-specific $\delta_{\text{struct}}$ / $\beta$ / $\alpha$;
7. **no** automatic polish;
8. WT does not enter the main generation.

**Scope of change (precise).** v0 makes **zero changes to RF controller semantics and
actuation** — D1, D2, D3, SC-GR, the control law, and the DPLM decoder are untouched.
It adds **only sampler-owned hard-constraint state transitions**: anchor
initialization and permanent remask protection. (The sampler is part of the reverse
loop, so "zero loop change" would be imprecise; the precise claim is zero
*control-law / risk-estimation / D1–D3* change.)

**The constraint belongs in the sampler, not in DPLM.** Verified (Appendix A): the RF
denoiser writes `prev_tokens[residue_positions] = x_t` (`runtime.py:526`), so a
native-valued anchor in `x_t` already conditions DPLM — no `partial_masks`. The four
invariants collapse to **two sampler changes**: initialize anchors committed
(invariants 1+2 automatic), and union a *permanent* anchor set into the existing
`_apply_reparam_remask(protected_positions=…)` seam (`sampler.py:572,606-611`) so it
survives every step including `controller=None` runs.

v0 answers exactly one question: *is all-mask-except-active-site global RF already good
enough?* Everything richer is deferred (§8).

### 4.1 Controller risk: visible but noneditable burden

Fixed anchors do not by themselves break the RF sampling state machine: a committed
anchor is not a masked residue, so first-unmask and D2 editable-position selection
should never act on it, and permanent sampler protection keeps D3/reparam remask from
reopening it. The real v0 risk is semantic, not mechanical:

> The controller may see immune risk that is real and visible to the head, but partly
> noneditable under the active-site constraint, and misread it as risk that should keep
> increasing editable pressure.

Therefore v0 deliberately keeps head refreshes full-sequence: anchor-overlap windows
must remain visible because they are part of the final antigenic sequence. But runtime
actions remain legally constrained:

```text
head / SC-GR may score all windows
active-window selection may include an anchor-overlap window
D2 / D3 / remask may only act on editable non-anchor positions
hard anchors must stay unchanged at every step
```

This is not solved by hiding active-site windows from the head. Hiding them would
underestimate residual immune risk. The correct v0 response is to preserve the hard
legality constraint and record enough telemetry to distinguish:

- an anchor-overlap hotspot that is repairable through flanking/global edits;
- an anchor-overlap hotspot that is seen by the controller but never lands a useful
  safe proposal;
- a final NetMHCIIpan hotspot that the in-loop head never saw (`external_only`);
- true residual burden that may require reachability-aware gain or a no-design verdict.

If the first run shows that anchor-overlap residual risk drives excessive global pressure
without realized benefit, that is the empirical trigger for E2 (`B_structurally_reachable`
/ legally-addressable pressure), not a reason to modify v0 in advance.

## 5. Function proxy: typed and calibrated

Do not keep a single $F(x)$. Separate four objects with different epistemics, and
within the online stage separate **legality** from **evidence**:

- **$C_{\text{online}}$ — hard online legality** (may this edit be *considered*?):
  $C_{\text{anchor}} \cap C_{\text{struct}}$ — anchor untouched, and inside the DPLM
  structural trust region ($\delta_{\text{struct}}$, a *structural* safety gate —
  `RF_Controller_Architecture.md:81`, not a function gate).
- **$E_{\text{online}}$ — proposal evidence** (is there *evidence* the edit is worth
  taking?): context reliability $r_{\text{context}}$, $\Delta R$, ensemble sign
  consistency, ESS. This is desirability/credibility, **not** legality.
- **$\mathbf{F}_{\text{post}}$ — post-hoc structure-derived function-plausibility
  vector** (for triage, never steering):
  $(f_{\text{anchor}}, f_{\text{fold}}, f_{\text{assembly}}, f_{\text{site\_scaffold}})$.
  A vector, not a scalar (collapsing re-hides distinct failure modes). It is **not a
  function predictor** — see §9 for what each coordinate may and may not claim.
- **$F_{\text{exp}}$ — real function**: normalized activity, $k_{\text{cat}}$, $K_M$,
  thermostability. Only this layer truly answers function preservation.

$$
C_{\text{online}} \neq E_{\text{online}} \neq \mathbf{F}_{\text{post}} \neq F_{\text{exp}}
$$

Read as: *may consider?* / *worth taking?* / *did the completed design stay
plausible?* / *was the enzyme actually functional?* Keeping these apart prevents
$\mathbf{F}_{\text{post}}$ (or $A_{\text{online}}$) from being misread as a weak
function discriminator.

**Calibration is per-coordinate, by failure class.** The wrong requirement is
"$\mathbf{F}_{\text{post}}$ must separate active from dead." A catalytic knockout
(e.g. Lys10→Ala) can have normal fold, assembly, and active-site backbone geometry yet
be dead — and it is rejected by $f_{\text{anchor}}$ directly, so it never tests
$f_{\text{fold}}$/$f_{\text{site\_scaffold}}$ at all. The right principle:

> Each coordinate is responsible only for the failure class it claims to detect. If a
> coordinate cannot discriminate that class with *axis-matched* controls (especially
> **anchor-preserving** negatives for site/assembly), it cannot be promoted to a hard
> gate; it stays `ranking_only`.

**v0 scope:** only $f_{\text{anchor}}$ and monomer $f_{\text{fold}}$ are evaluated;
$f_{\text{assembly}}$ and $f_{\text{site\_scaffold}}$ are deferred — they need a multimer
predictor (the uricase active site is cross-protomer and the caseset backbones are monomer
AFDB/ESMFold, §7.4–7.5). The calibration manifest (§9.1) is split by axis. **Reality check
(§7.5):** no quantitative kcat/Km, but the 25 `characterized` members are
experimentally-verified actives — a real **positive-control** set. So v0 $f_{\text{fold}}$ can
be calibrated by separating characterized actives from the bulk plus family conservation, but
it lacks **anchor-preserving negatives** (active-site-intact yet dead), so it stays
`ranking_only` (a one-sided gate) until those negatives are sourced. Before round-1 data, decide which coordinate is
*eligible* to be a gate, not the thresholds.

## 6. Decision layer

Route post-hoc on three dimensions — a single scalar cannot tell apart distinct failure
modes:

- $R(x)$ = immune burden. **External NetMHCIIpan EL, not the in-loop head** — the head
  is the signal RF steered against, so grading "design beats WT" on it is circular. Score
  against the **project's own per-allele eval** (§7.3, 15–23mer NMP): on DRB1_0701 the `wt_v2`
  baseline puts WT Q00511 at ≈59th pct — real headroom. (The `data/Uricase` 0401 family scan is
  single-15-mer/rank<1%, **non-comparable** — not an allele claim; a same-protocol 0401 baseline
  is still needed.) The bar is the natural achievable distribution under one fixed protocol.
- $\mathbf{F}_{\text{post}}(x)$ = the typed proxy vector (§5).
- $C(x)$ = residual hotspot **concentration** (concentrated in a few windows vs diffuse).

Admissibility on $\mathbf{F}_{\text{post}}$ is a **typed rule**, not a weighted total —
$\text{ProxyAdmissible}(\mathbf{F}_{\text{post}}; \mathcal{M}_{\text{calibration}})$:

```text
hard-invariant coordinate failed  → reject
calibrated-gate coordinate failed → reject / flag
ranking-only coordinate poor      → down-rank
unavailable coordinate            → NA (not pass, not fail)
```

| Outcome | Route |
|---|---|
| admissible, $R$ low enough | accept global RF design |
| admissible, $R$ from a few windows **and** dilution-limited | local polish of the **best RF design** (not WT) |
| admissible, $R$ broadly distributed | strengthen / rerun global RF (not patchable) |
| not admissible, or none beats WT, or global feasibility failure | WT-rescue: small edit budget, strong local de-immunization on WT |
| WT already good, no real gain | return WT / no redesign |

Two non-obvious points:

- **Polish substrate is the best RF design, not WT** — the RF design already escaped WT's
  basin. WT-rescue is only for *broad* feasibility failure.
- **Residual diagnosis is four-class, and telemetry cannot reach function-limited.** The
  existing telemetry sees only what the *internal head* turned into D2 blocks; final $R$
  is *external* NetMHCIIpan, so a hotspot the head never saw has no trajectory. Classify:

  ```text
  internal_seen_feasible_unlanded   # dilution / actuation (best_delta_R_B<0 but low selected/persistence)
  internal_seen_no_safe_proposal    # structure/proposal-limited (candidate_feasibility=False, safe_support≈1)
  external_only                     # NetMHCIIpan flags it, head never saw it → no telemetry
  unresolved                        # block↔event join or evidence insufficient
  ```

  Rename the old "constraint-limited" to **structure/proposal-limited** (telemetry judges
  structural proposal feasibility, not function feasibility). The `external_only` class is
  also a **head↔validator agreement diagnostic**: if it is large, the in-loop head is
  systematically missing what NetMHCIIpan catches — i.e. steering is optimizing the wrong
  target — which is a finding, not just an unclassifiable bucket. Polish therefore stays a
  data-driven branch (it only helps the `internal_seen_feasible_unlanded` class), never a
  default stage.

## 7. Uricase instantiation (grounded on `data/Uricase`)

Grounded on Kaiyi's uricase evolution + immunogenicity pipeline at `data/Uricase/`
(3034-taxon family MSA, per-position conservation, family-wide NetMHCIIpan) —
workflow-verified by independent re-computation 2026-06-23 — plus M-CSA entry 118.

### 7.1 Active-site facts + canonical anchor set (M-CSA #118, PDB 1wrr)

*Aspergillus flavus* urate oxidase (UOX, UniProt Q00511, EC 1.7.3.3), ~135 kDa,
**homotetramer**, ~301 residues/subunit. **Cofactor-free** — no metal, no flavin
(originally thought to need Cu; disproven, PMID 12680763) — so catalysis comes from
holding substrate + O₂ in precise geometry. Tunnel/T-fold family; **four active sites,
each at a subunit (dimer) interface.**

| Biological role | Residues (PDB 1wrr) | Note |
|---|---|---|
| Catalytic triad (proton relay; general acid/base) | **Thr57, Lys10, His256** | UniProt numbering +1 (Thr58/Lys11/His257) |
| Substrate stabilizers ("molecular tweezers") | **Arg176, Gln228** | hold urate dianion |
| Peroxo-hole / W1 water coordination | **Thr57\*, Asn254** | **cross-protomer** (from the adjacent subunit) |

Provenance to record (not a single undisputed set): early work named a Thr–Lys **diad**
(Imhoff 2003); current M-CSA / atomic-resolution + neutron work (Oksanen 2014; Bui 2014;
joint neutron/X-ray 2021) extend it to a Thr–Lys–**His** triad plus a proton-shuttling
water network.

**Canonical anchor set (UniProt Q00511 + M-CSA #118, conservation-filtered).** Provenance:
the set comes from **UniProt Q00511 functional-site annotation** (3 active sites 11/58/257 =
the Lys/Thr/His charge-relay triad; 7 binding sites at 58, 59, 160, 177, 228, 229, 255 —
ligands urate / O₂ / 5-HIU) **cross-referenced with M-CSA #118** and **filtered by family
conservation** (it is *not* a pure conservation filter). Identity sequence-verified;
**0-based index = PDB 1wrr = UniProt − 1** (both Met-excluded; UniProt is Met-inclusive, +1).

| 0-based | UniProt | AA | family cons | role |
|---|---|---|---|---|
| 10  | 11  | K | 0.972 | active (triad) |
| 57  | 58  | T | 0.975 | active + binding |
| 58  | 59  | D | 0.974 | binding |
| 159 | 160 | F | 0.918 | binding (purine stacking) |
| 176 | 177 | R | 0.985 | binding (stabilizer) |
| 228 | 229 | Q | 0.987 | binding (stabilizer) |
| 254 | 255 | N | 0.970 | binding (cross-protomer W1) |
| 256 | 257 | H | 0.944 | active (triad) |

The 4 UniProt binding sites originally missing from the M-CSA-only set are reconciled by
conservation: **Asp59, Phe160, Asn255 are highly conserved (0.92–0.97) → added**; **Val228
(0-based 227) is only 0.649 conserved (family-variable) → NOT hard-fixed** (left editable /
monitored). Phe160 (0.918) is the softest hard-fix — droppable if a larger editable region
is wanted.

> This table is Q00511 / 1wrr-specific **biological annotation**. It does not by itself
> define hard anchors for every uricase case, nor does its +1 UniProt offset generalize —
> both are per-case decisions resolved against each case's resolved sequence.

### 7.2 `ActiveSiteSpec` — annotation ≠ enforcement ≠ evaluation

A residue index list is insufficient and dangerous: M-CSA's biological role must **not**
auto-equal sampler enforcement. Each spec entry carries three independent typed fields:

```text
biological_annotation   # what the literature says (role, source/PMID, organism, numbering)
runtime_enforcement     # v0 design choice: hard_fix | monitored_shell | free
evaluation_geometry     # which atoms / distances / angles F_post uses for this residue
```

Worked example — **Asn254**: `biological_annotation` = W1/peroxo-hole coordination,
cross-protomer (literature fact); `evaluation_geometry` = include its side-chain heavy
atoms + cross-protomer contact (design choice to *measure* it); `runtime_enforcement` =
**open** — whether to `hard_fix` it or leave it a `monitored_shell` residue is a design
decision, **not** a literature fact. The catalytic triad (Thr57/Lys10/His256) is the only
set that is uncontroversially `hard_fix` in v0; Arg176/Gln228/Asn254 are candidates whose
enforcement tier is decided at freeze time.

**Concrete v0 instantiation for Q00511.** `hard_fix` = 0-based **{10, 57, 58, 159, 176, 228,
254, 256}** (8 residues = catalytic triad + Asp59 + Phe160 + stabilizers Arg/Gln +
cross-protomer Asn255), all UniProt/M-CSA-annotated and family-conserved 0.92–0.99 (§7.1).
UniProt binding **Val228** (0-based 227, cons 0.649, family-variable) is **not** hard-fixed.
**Do NOT import `08_conservation_analysis.py`'s 9-residue list** — workflow-verified,
3 of its 9 entries are **mislabeled** against the actual Q00511 sequence (UniProt pos36
labeled Phe is C, pos64 labeled Phe is T, pos144 labeled Arg is K; all variable in the
family, conservation 0.28/0.81/0.22 — not catalytic). The 6 clean entries match. The single
mitigation is the fail-fast identity assert (§7.4), which would have caught all 3.

By homo-symmetry, fixing the chosen positions on the design chain protects all four sites
**provided the spec is complete** (includes interface-contributing residues). Container
schema:

```text
hard_anchors:    [{chain, residue_number, expected_AA, biological_annotation, source}]
shell_residues:  [{chain, residue_number, biological_annotation, enforcement_tier}]
assembly_context:{biological_assembly, interacting_chain_pairs}
geometry_observables:[{atom_selections, ref_distances, ref_angles, cross_chain_flag, structural_state}]
```

### 7.3 Evolutionary grounding (the function-axis prior)

The function axis is not blind: Kaiyi's family analysis supplies an evolution-based prior —
the project's "ground truth over single-structure" stance.

**Conservation tiering (editable region).** Per-position Shannon conservation over the
3034-taxon family MSA, on the 302 Q00511 ref positions (= 0-based index + 1):

- **safely variable** (conservation < 0.5 AND gap < 0.5): **136/302 = 45%** — the
  conservative global-redesign search space.
- **highly conserved** (≥ 0.95): 11/302 = 3.6% — strongest preserve evidence.
- **intermediate**: 155/302 = 51%.

Two cautions: (a) this is **sequence entropy only — no structure/surface/dynamics**, so
"safely variable" is an editability *prior*, not a fold/function guarantee; a structural
shell still needs a structural source. (b) catalytic His is 0.94 (< 0.95), so anchors must
be **whitelisted** (§7.1), never threshold-derived.

**Immune landscape — the WT 0701 baseline.** The project's own immune eval (**15–23mer**
NetMHCIIpan + the in-loop head — the standard pipeline) over the full 6740 caseset is the run
`Results/RF/HLA-DRB1_07_01/wt_uricase_v2_…__20260607T185243Z`, which scores the WT sequences
themselves (`wt_generated == WT`, a baseline facade). On **DRB1_0701** (NMP rank<2%):
strong-binder median 83; **WT Q00511 = 97 strong ≈ 59th percentile (NMP) / 88th percentile
(head `global_risk`) — mid-to-high, with real redesign headroom.** Use external NMP for the
decision-layer R(x), not the head (anti-circularity).

⚠️ **Not an allele effect (methodology confound).** The `data/Uricase` 0401 family histogram
(where Q00511 looked ~21st pct, median 27) used a **single 15-mer window at rank<1%** — *not*
the project's 15–23mer / rank<2% protocol — so it is **not comparable** to the 0701 eval and
must not be read as "WT is low on 0401, high on 0701." A real per-allele comparison needs the
*same* scanning protocol on both alleles; **no same-protocol 0401 WT baseline exists locally
yet**. Robust takeaway: WT has genuine 0701 headroom, and any low-immune claim must be made per
allele under one fixed protocol.

**Functional positive controls (not templates).** The materialized ground-truth-active set is
the **25 `characterized` uricases** — experimentally-verified actives (`characterized_uricases.fasta`,
LOG 2825; spanning fungi/bacteria/animals/plant incl. pig/mouse/rat/baboon). These are F_post
**positive controls**, not generation templates. (ASR — inferring extinct ancestral sequences
at tree nodes — ran but is unrealized and is **out of scope**: this task redesigns each WT
backbone in place, it does not seed from templates.)

### 7.4 Two consequences of the cross-protomer site

1. **Numbering is three-way with no translator.** The spec exists in UniProt (Thr58…) and
   PDB-author (Thr57…) numbering — off by one — while the RF pipeline indexes 0-based into
   the resolved parquet `sequence` (= h-value space = `x_t` space; telemetry uses
   `residue_index_0b`), with **no PDB-numbering reconciliation in the runtime** (Appendix A).
   Instantiation is a **fail-fast three-way alignment**: assert the resolved-sequence AA at
   each mapped index equals `expected_AA`.
2. **A monomer ESMFold cannot evaluate $f_{\text{site\_scaffold}}$ / $f_{\text{assembly}}$.**
   The catalytic geometry literally does not exist in a single chain (Thr57\*, Asn254, the
   peroxo-hole water are cross-protomer). So those coordinates require a **multimer
   predictor** (AlphaFold-Multimer / AF3-class); monomer self-consistency can only bound
   $f_{\text{fold}}$. Claiming "we evaluated active-site geometry" on a monomer is false.

### 7.5 Data surface (corrected)

The IF design cases are in `uricase_caseset_if_ready_unified.parquet` (workflow-verified):
**6740 rows derived from Kaiyi's curated `filtered_uricases.fasta` (6736/6740 = 99.94%),
NOT the clustered nr90 set** (44.8% overlap) — the design operates on the full, redundant
family pool (sequence length 200/302/502), and that same universe carries the §7.3
conservation + immune annotations. **Backbones are predominantly AlphaFold-DB (5222 = 77%)
+ ESMFold2 (1518 = 23%)** — *not* uniformly ESMFold as earlier stated. All are computational
predictions (no experimental backbone; 38 ESMFold2 rows fail the GT gate), so
$f_{\text{fold}}$ is self-consistency against a *predicted monomer* target. There is **no
quantitative kcat/Km**, but the **25 `characterized` members are experimentally-verified
actives** — a real qualitative **positive-control** set for the function proxy (matched
anchor-preserving *negatives* still need sourcing). Q00511 is fully in-set (seed == case,
AFDB, characterized) — the natural WT instance for the first run. Redundancy in the pool may
need dedup for fair case selection.

## 8. Deferred (and why)

Held back until the first run produces the empirical distribution; the routing branches may
be mostly empty, and designing them now is premature.

- **E1 — Functional-Pareto shell.** The shell (decaying with structural distance) should not
  inherit WT identity; it should inherit a stricter Pareto policy — smaller
  $\delta_{\text{struct}}$, smaller immune tilt $\beta$, higher structural commit weight
  $\alpha$. Distal regions keep the normal trade-off.
- **E2 — Reachability-aware gain.** Drive the SC-GR scalar by **reachable** rather than total
  burden, so blocked active-site risk does not crank pressure everywhere. *Honesty
  constraint:* in the loop you only have **structural** reachability; the in-loop quantity is
  `B_structurally_reachable` / "legally addressable," not function-reachable.
- **In-loop burden telemetry.** Even `B_legally_addressable` need not go in the loop for v0 —
  compute it post-hoc first; add in-loop only if blocked burden actually misfires
  $g_{\text{GR}}$.
- **Formal routing thresholds and an automatic polish stage** — until $C(x)$ and the
  residual taxonomy are observed on real runs.
- **$f_{\text{assembly}}$ / $f_{\text{site\_scaffold}}$ (multimer geometry)** — deferred per
  the v0 anchor+fold scope (§5); need a multimer predictor for the cross-protomer site.
- **Conservation-weighted editable prior.** The 45% / 3.6% / 51% tiers (§7.3) are a graded
  prior; v0 uses the catalytic-anchor whitelist only, but a later version can weight the edit
  penalty by (1 − conservation) instead of a binary mask.
- **Anchor-preserving function negatives.** The function proxy has positive controls (25
  characterized actives) but no active-site-intact-yet-dead negatives; sourcing them (literature
  mutants) would promote `f_fold`/`f_site_scaffold` from `ranking_only` to calibrated gates.

(ASR / ancestral templates are **out of scope**, not merely deferred — this task redesigns each
WT backbone in place and does not seed from templates; §7.3.)

## 9. Contracts to freeze (these gate the first run)

Freeze order (each step prevents the next contract from conflating concerns):

1. **`ActiveSiteSpec` annotation / enforcement / evaluation split** (§7.2) — not the full
   policy, just the three-way typing.
2. **`F_post` v0 measurement contract** — measurement semantics, reference, missingness,
   uncertainty, calibration status (below).
3. **Per-case numbering map + v0 hard-anchor policy** — the per-case three-way alignment and
   the `runtime_enforcement` tier for each annotated residue.
4. **Only then** Coder writes the sampler hook + evaluation pipeline.

Each frozen contract records: `contract_version`, `freeze_date`, `schema_version` /
`spec_hash`, `review_status` — in this doc, not LOG.md (a design freeze is not an
implementation change).

### 9.0 v0 freeze (2026-06-23, user-approved)

- **`ActiveSiteSpec` (Q00511):** `hard_fix` = 0-based **{10, 57, 58, 159, 176, 228, 254, 256}**
  (8: catalytic triad Lys11/Thr58/His257 + binding Asp59/Phe160/Asn255 + stabilizers
  Arg177/Gln229), sourced from **UniProt active+binding sites ∩ M-CSA #118**, all family-conserved
  0.92–0.99. UniProt binding **Val228** (cons 0.649, family-variable) is **not** hard-fixed; any
  further shell is monitored post-hoc (manual OK). Required: fail-fast
  `assert resolved_seq[idx] == expected_AA`.
- **`F_post` v0:** `f_anchor` (hard_invariant) + monomer `f_fold` (`ranking_only`; positive
  controls = 25 characterized actives + conservation; negatives not yet sourced).
  `f_assembly` / `f_site_scaffold` deferred (multimer).
- `contract_version`: v0 · `freeze_date`: 2026-06-23 · `review_status`: user-approved
  (Thinker4Uricases). Only mechanical work remains before Coder: the per-case fail-fast
  numbering map.

### 9.1 `F_post` v0 contract (draft)

A **structure-derived function-plausibility vector**, not a function predictor. Each
coordinate is a typed record (not a single value) returning a common envelope:

```yaml
coordinate_name:
  availability: available | unavailable | predictor_failed
  metrics: { ... }                       # recovery vs reference + predictor confidence, kept separate
  absolute_design: { ... }
  absolute_wt: { ... }                   # WT run under identical predictor/seeds/settings
  delta_vs_wt: { ... }
  uncertainty: { seeds, median, spread } # replicate spread, not a point estimate
  reference: { type, accession, state }  # experimental_same_seq | homolog_template | predicted_target ; apo | ligand_bound
  calibration_status: hard_invariant | calibrated_gate | ranking_only | unavailable
  provenance: { predictor, version, parameters, weights }
```

| Coordinate | Measures | Must NOT claim | Default status |
|---|---|---|---|
| `f_anchor` | exact anchor identities preserved, all mapped positions exist, expected_AA passes, symmetry-expanded chains correct | anything about activity | **hard_invariant** (predictor-independent; mismatch = impl/manifest bug) |
| `f_fold` | recovery vs the **conditioning (WT) backbone** (scTM, core RMSD) **and** predictor confidence (pLDDT/PAE/pTM), reported separately; WT-matched | experimental accuracy (target is itself ESMFold-derived, §7.4) | calibrated_gate *iff* §5 controls pass, else ranking_only |
| `f_assembly` | symmetry-aware assembly recovery vs reference assembly + interface contact recovery; **and** ipTM/inter-chain PAE confidence, separately | function; ipTM is model confidence, not accuracy | **deferred (post-v0)**; `unavailable` on monomer; needs multimer + anchor-preserving negatives |
| `f_site_scaffold` | site-neighborhood backbone RMSD, catalytic/binding side-chain heavy-atom RMSD, cross-protomer contact recovery, selected distance/angle deviations | protonation, water network, O₂ positioning, TS stabilization, dynamics, $k_{\text{cat}}$/$K_M$ | **deferred (post-v0)**; `unavailable` on monomer; needs multimer |

Notes that must be in the contract:

- **Homotetramer matching:** chain-permutation / symmetry-aware matching; do not assume the
  predictor's chains A/B/C/D map 1:1 to PDB labels.
- **Reference ensemble for geometry:** 1wrr is a ligand-analogue complex, so a single-structure
  RMSD bakes in a ligand-bound microstate. Support
  `reference_structure_ensemble: {apo, substrate/analogue-bound, alternate}` and report the
  min / distribution distance to the acceptable ensemble.
- **Calibration manifest, by axis** (positive / **anchor-preserving** negative controls):
  `f_anchor` ← catalytic-substitution negatives; `f_fold` ← grossly destabilizing/core
  negatives; `f_assembly` ← interface/oligomerization-defective negatives; `f_site_scaffold`
  ← anchor-preserving shell/interface mutants with activity loss. Missing matched negatives ⇒
  the coordinate is `ranking_only`.

### 9.2 First-run evaluation protocol (predeclare to avoid selection bias)

The first run yields the complete $(\mathbf{F}_{\text{post}}, R, C)$ joint distribution
**only if every candidate gets multimer evaluation.** Otherwise it is two tiers, fixed
**before** the run:

```text
Tier 1 — all candidates:           f_anchor + f_fold + R + C
Tier 2 — predeclared subset:       f_assembly + f_site_scaffold  (multimer)
         subset = WT + calibration controls + a STRATIFIED/RANDOM sample
         (NOT Tier-1 top-k — that biases the multimer distribution)
```

Missing multimer coordinates are `NA`, never interpreted as pass. Only after §9.1–9.2 are
frozen does the first all-mask-except-anchors run become interpretable; its
$(\mathbf{F}_{\text{post}}, R, C)$ distribution decides which of §8's layers are needed.

---

## Appendix A — Code surface (verified)

Read-only facts about the current code, cross-checked at file:line. The RF wrappers **do not
expose inpainting today**; the C0 path is a bring-up smoke test, **not** the scientific main
line.

### A.1 DPLM has two position-fixing primitives with OPPOSITE polarity

(A) **Model-level `partial_masks` — `True` = KEEP/FREEZE (the one to use).** `dplm.py`
`generate` (`:503`) / `initialize_output_tokens` (`:208`) / `get_non_special_symbol_mask`
(`:377`); gate `non_special_sym_mask &= ~partial_masks` (`:383-384`); frozen positions get
score `1000.0` (`:426-428`) so never decoded or remasked. Same adapter path in
`dplm_invfold.py` (`:188/:276/:433`). Canonical demo: `run/scaffold_generate_dplm.py:35-47`.

(B) **Task-level `inject_noise(noise="selected_mask", sel_mask=…)` — `True` = MASK/REDESIGN
(opposite polarity), NOT wired:** `predict_step` calls `inject_noise` without `sel_mask`
(`tasks/lm/dplm_invfold.py:448-452`). `generate_dplm.py` declares `--cond_seq/--cond_position`
(`:139-140`) but never passes them into `initialize_generation()` (`:84-86`) — dead CLI.

### A.2 The RF path bypasses `partial_masks` entirely

`runtime.py` hardcodes `noise="full_mask"` in **four** places — native helpers (`:415`,
`:363-365`) and RF context builders (`:447`, `:482`). The RF denoiser (`make_dplm_denoiser`,
`:514`) calls the **raw** net `task.model.decoder(...)` (`:527`) and returns logits — never
`forward_decoder`/`generate`, so `partial_masks` is unreachable. **But it writes
`prev_tokens[residue_positions] = x_t` (`:526`)** — a native-valued anchor in `x_t` conditions
DPLM automatically. `sample()` / `_init_batch_lane_state` init the whole sequence to mask
(`sampler.py:88,356`).

### A.3 Invariant → code map for fixed anchors

- Init-as-committed in `sampler.py:88` / `:356` → invariants 1+2 automatic.
- `_apply_reparam_remask(protected_positions=…)` (`sampler.py:572,606-611`) → invariant 3;
  must be a **permanent** union applied in the sampler (runs even with `controller=None`).
- `select_editable_positions` returns only masked residues (`counterfactual.py:37,62-67`) →
  invariant 4 automatic; keeping *pressure* off the anchor region (window scope) would need
  masking in active-window selection (`controller.py:647`) — an E1 concern, not v0.

### A.4 Residual-diagnosis telemetry (for §6)

All real: `candidate_feasibility` / `best_delta_R_B` / `safe_support_sizes` / `skipped_reason`
(block-level), and `selected_after_d2_rate` / `realized_benefit_rate` / `final_persistence_rate`
/ `ensemble_sign_consistency` / `churn_rate` (position/run-level). The block↔event join
(`block_id` + `refresh_step`) needed for a per-window verdict exists in the raw parquet but is
not pre-computed by the run aggregator — and cannot reach the `external_only` class (§6), which
has no internal trajectory at all.

### A.5 Index convention and run-metadata stamping

Positions are 0-based into the resolved parquet `sequence`; no PDB-numbering layer exists.
`protein_id` is the unique per-protein key. Run metadata is written by `write_phase_c_outputs`
(`run_if_phase_c0.py:352-368`); the `d1_manifest_provenance` pattern
(`run_if_phase_c1.py:1049-1129`) is the model for stamping a future constraint/spec hash +
fixed-count.
