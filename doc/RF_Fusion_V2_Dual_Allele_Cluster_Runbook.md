# RF Fusion V2 Dual-Allele — Concise Cluster Runbook

**Status:** the original three-arm high-risk diagnostic is executed and closed (§6). The
paper-facing full-data Dual capability campaign authorized in §7 is **executed and read** (§7.9):
verdict `dual_fulldata_capability_supported` on 89 of 100 proteins, with the claim boundary in
§7.9.7.

**Scientific authority:** `doc/Dual_Allele_Steering.md`.

**Implementation and decision authority:** `PLAN_RF_FUSION_V2_DUAL_ALLELE.md` and
`doc/DUAL_ALLELE_DUALF0_AUDIT.md`, especially Appendices E, H, I, J, and K.

This runbook asks one question:

> Under the same Fusion V2 generation authority, does simultaneous Dual steering reach a lower
> frozen joint objective than either single-allele steering law?

Sections 0–6 preserve the executed diagnostic and its claim boundary. Section 7 asks a different,
paper-facing question on a common Head-blind cohort and is the only active execution authority.

---

## 0. Already frozen and completed

This section describes the executed eval-split-Head diagnostic only. Its calibration, overlay,
cohort, root count, and three-arm comparison must not be reused as the §7 full-data campaign
identity.

Do not rerun or tune the following:

| Item | Frozen value or artifact |
|---|---|
| Role A | `HLA-DRB1*07:01`, checkpoint identity bound by the signed overlay |
| Role B | `HLA-DRB1*04:01`, distinct checkpoint identity bound by the signed overlay |
| Objective | frozen-normalized smooth maximum |
| Credit | $c_u=0.10$ normalized units |
| Temperature | $\tau=0.10/\log 2=0.14426950408889636$ |
| Coordinates | overlap-excluded Tier-2 natural panel, 13,836 homology-cluster representatives |
| Panel sensitivity | measured `within_credit`; $|\Delta\theta|=0.0005688505037048097$ |
| Campaign overlay | `dual_overlay_c1_v1.json`, $C=454$, digest prefix `fb428357287e` |
| Donor/write margin | derived from signed same-sequence repeatability; no hand-set margin |
| Population effect threshold | none; $\epsilon_J$ is deleted |
| Recursive substrate | progressive D4/K12/r40, checkpoints 50→60→70→80→90 |
| Runtime environment | `immune-design` for all three arms |
| Role-B safety | whole-landscape hotspot telemetry, not an admission gate in v1 |
| Structure/anchors | inherited Fusion V2 behavior, unchanged |
| Runtime immune signals | the two frozen Heads only; NMP is absent |

The overlap-inclusion measurement, corrected calibration uncertainty, environment transfer audit,
candidate-domain accounting, and signed C1 overlay already exist. Verify their identities; do not
rebuild or re-sign them as part of this campaign.

The three objective arms are:

- `joint`: smooth-max Dual steering;
- `a_only`: role-A coordinate controls donor/write decisions while role B is still scored; and
- `b_only`: role-B coordinate controls donor/write decisions while role A is still scored.

All arms score both Heads and use the same candidate domain, generation/refold budget, structure
gate, anchors, environment, and seed law. Only the objective law changes.

---

## 1. D0 and root are internal algorithm states

### 1.1 D0 is not a separate experiment

For one root, Fusion V2 captures a source partial state and generates its K=12 complete endpoint
cloud at depth zero. Each objective arm may choose a different donor from the same cloud. D0 is
therefore the first matched molecular decision surface inside the formal run, not a NoD baseline
selected by the operator.

Across `joint`, `a_only`, and `b_only`, D0 pairing requires equality of:

- `source_state_id`;
- `endpoint_id`; and
- `sequence_md5` for every endpoint.

Do not require endpoint `content_digest` equality. Binding the second Head adds different evidence
to the same molecule and legitimately changes that digest.

### 1.2 The Dual campaign generates its own root

The campaign does not reuse a generated root from the previous Fusion run. The previous run
supplies the frozen 100-protein cohort and substrate definition; the Dual campaign derives its own
root from its own `campaign_id`, `master_seed`, and protein identity.

Use **one root per protein** in the first Dual campaign. The scientific comparison is between
objective laws on the same cloud, not a search-coverage study. Four roots in the prior V2 campaign
measured search yield; copying that multiplicity here would double or quadruple the cost without
strengthening the paired objective-law contrast.

The historical 80-protein floor came from a four-root union and is not a scientific failure
criterion for this single-root campaign. Based on the measured per-root success rate, a healthy
single-root run has about a 5% chance of falling below 80. Therefore:

- analyze all valid matched triplets;
- require at least 78 valid proteins for the primary capability statement;
- if fewer than 78 remain, report `undercovered_unresolved`; and
- do not call Dual ineffective or silently add favorable roots.

This 78-protein validity floor has a measured healthy-run false-stop probability of approximately
0.4%. Additional roots are a later deployment-yield decision, not part of this first comparison.

---

## 2. One prelaunch assembly gate

Use the implementation's generated `.args.sh` files and launch bundle as the command authority.
Do not copy volatile checkpoint, structure-cache, or shard paths from an older runbook.

### 2.1 Materialize the formal campaign

Materialize exactly:

```text
100 proteins × 1 campaign root = 100 resolved cells
```

Use the frozen C1 overlay and one common root seed. The arm is a launch-time `--dual-arm` value,
not a separate materialized config. Cell names must describe this campaign's roots (`r0_...`), not
pretend to inherit the source campaign's `root0_...` states.

### 2.2 Model-free preflight

Run the existing materializer and assembly preflight before allocating a GPU. Resolve every cell
under each of the three arm values and verify automatically that:

- the three arms resolve to three different objective digests;
- both distinct Head checkpoint identities are bound;
- the signed C1 overlay declares $C=454$ candidate sequences and at most $2C=908$ logical Head
  evaluations per cycle;
- D4/K12/r40 and one campaign root are exact;
- Role B is telemetry-only for safety;
- sampler, structure, anchor, environment, and seed settings are arm-invariant;
- `dual_arm` does not enter root, lookahead, or repair seed derivation;
- the generated `.args.sh` binds the same overlay that the launcher will pass;
- all input roles and content digests are present; and
- NMP is absent.

Any identity, assembly, budget, or seed-law mismatch blocks launch. Fix the implementation or
materialization under the same campaign identity. Do not add a synthetic GPU smoke.

---

## 3. One formal campaign

### 3.1 Matched execution

For every cohort protein, run:

```text
joint
a_only
b_only
```

This is 300 single-root arm cells. The three runs for a protein share the same `campaign_id`, root
seed, and non-objective inputs. The arm label changes only the objective law. Submit and track the
three arms together by protein so an operational failure cannot masquerade as an allele effect.

Do not run `joint` alone. Joint-vs-D0 would show descent under its own objective, but it could not
distinguish simultaneous coordination from ordinary A steering or a complete switch to B. The two
single-allele arms are the minimum controls needed for that claim.

### 3.2 First formal protein as an operational gate

Run the three arms for the lexicographically first cohort protein, then apply the automatic bundle
reader before releasing the remaining 99. This protein remains in the formal denominator; it is not
a disposable smoke or tuning population.

Continue only when:

- all three manifests and Dual evidence tables exist;
- D0 `source_state_id`, `endpoint_id`, and `sequence_md5` sets match exactly;
- objective digests differ as declared;
- both Heads were scored and correctly metered;
- candidate, Head, DFE, refold, and memory caps were respected; and
- endpoint, archive, feedback, Role-B telemetry, lineage, and cost joins are complete.

Do not inspect performance, require decision divergence, tune a parameter, or replace the first
protein at this gate. The three laws may legitimately make the same decision for one protein.

### 3.3 Release the remaining 99 proteins

After the first protein passes operational integrity, submit all remaining work before opening an
outcome summary. Preserve typed failures and carried incumbents. Do not rerun only an unfavorable
arm or replace a failed protein.

The materialized resource projection is authoritative. If its Head/refold/memory/walltime cap
fails, stop and repair the campaign contract rather than weakening one arm.

---

## 4. One combined mechanism and capability read

Endpoint rows are nested observations, not independent proteins. All inferential summaries use
protein-level matched triplets.

### 4.1 Integrity and denominator

Before reading effects:

- retain all 100 requested proteins in the coverage table;
- verify arm-invariant root/D0 molecule identities and seed law;
- verify $C=454$ and equal nominal generation/refold budgets;
- verify both Head identities and objective/calibration digests;
- verify Role-B hotspot telemetry was never used as an admission gate;
- retain typed stalls through their carried incumbent; and
- classify incomplete or mismatched triplets as invalid, not as zero-effect method failures.

Report the valid-triplet count explicitly. Fewer than 78 valid proteins yields
`undercovered_unresolved`; it does not establish a negative Dual result.

### 4.2 D0→D1 mechanism read

Read the former M1 question from the first transition of the same formal campaign:

- fraction of proteins where joint changes the D0 donor relative to A-only or B-only;
- changed write and reopen identities;
- realized write/reopen cardinalities as outcomes, not parity constraints;
- typed no-donor/no-positive-write stalls;
- paired D1 descendant $J$, raw $R_A$, and raw $R_B$ changes; and
- Head, DFE, refold, and walltime cost.

This determines whether the second Head enters the actuator and whether fresh descendants move in
the declared direction. No separate 5ZHV/Q00511 prefix experiment is required.

### 4.3 D4 capability read

Recompute one common smooth-max frontier from every arm's exact endpoints that pass the inherited
Fusion V2 structure/admission law. Role-B hotspot telemetry is reported separately and does not
define feasibility in v1.

For protein $p$, arm $a$, and depth $d$:

$$
F_{p,a,d}
=
\min_{\substack{
y\in a,\;\operatorname{depth}(y)\le d\\
F_{\mathrm{V2}}(y)=1
}}
J_\tau(y).
$$

The three primary lower-is-better contrasts are:

1. joint final vs shared D0;
2. joint final vs A-only final; and
3. joint final vs B-only final.

For each contrast, resample proteins with replacement 10,000 times using analysis seed `20260821`
and report the one-sided 95% upper percentile-bootstrap bound on the protein-mean difference. The
directionality criterion is:

$$
\operatorname{UCB}_{95\%}\left[\operatorname{mean}(\Delta J)\right] < 0.
$$

There is no $\epsilon_J$. Donor/write margins protect pointwise runtime decisions and are not
subtracted from a population mean.

Also report raw and normalized A/B changes, per-depth frontiers, last improving depth, offline
non-dominated endpoints, Role-B hotspot telemetry, structure feasibility, typed stalls, diversity,
and total cost.

### 4.4 Verdict

| Result | Verdict |
|---|---|
| joint beats shared D0, A-only, and B-only with clean integrity | `dual_recursive_authority_supported` |
| joint beats only one single-allele arm | `partial_dual_result`; simultaneous advantage is not established |
| joint fails to beat A-only | no demonstrated gain over the current Fusion objective |
| joint fails to beat B-only | the method may have switched to B rather than coordinating both |
| root/D0/seed/budget/objective identity mismatch | invalid run; repair under the frozen campaign |
| fewer than 78 valid triplets | `undercovered_unresolved` |

Success does not require both raw alleles to improve for every protein or equal A/B write counts.
It requires a better common joint frontier without hidden scale, seed, feasibility, or compute
asymmetry.

---

## 5. Stop and return

This stop closes the original high-risk three-arm diagnostic. Section 7 is a separately authorized
paper-facing campaign with new full-data Head and cohort identities; it is not a continuation or
parameter sweep of the diagnostic.

After the combined read, stop. Do not automatically run:

- the old P0 opportunity probe;
- the old M1/56-prefix campaign;
- a separate D0 experiment;
- extra roots;
- K24/K32/D8 or a $\tau$ sweep;
- hard-max generation;
- protected sequential search;
- online Pareto/FK/MCTS;
- NMP; or
- a new structure study.

Return through the standard RF/RAR path:

- the frozen primary calibration, overlap-inclusion sensitivity, and signed C1 overlay;
- resolved configs, generated launch bundle, and preflight report;
- all 300 requested cell bundles or typed failure records;
- one protein-level mechanism/capability summary;
- requested-cohort coverage, cost, and provenance;
- Slurm IDs, `sacct`, stdout/stderr, and code/config/content digests; and
- one RAR record containing the frozen question, integrity verdict, primary contrasts, raw A/B
  readout, coverage, and claim boundary.

Experiment submission and result return do not modify `LOG.md`.

---

## 6. EXECUTED (2026-08-25) — `no_demonstrated_gain_over_either_single_allele_arm`

Campaign `dual_c1_smoothmax_d4k12_r40_0701x0401`, `master_seed=20260825`,
code revision `58b30072f1fca96a30d77e9c49decd80bac9cd0d`, signed C1 overlay `fb428357287e`.
300 cells on ailab H200 under `immune-design`: gate `12958829`–`31`, release `12970199`–`12970213`
(15 shards, 1 h 15 m – 1 h 34 m each, every shard `EXIT=3`). Bundle
`work/immune-design/v2_dual/dual_c1_smoothmax_d4k12_r40_0701x0401__58b3007/`, read
`analysis/dual_c1_combined_read.json`.

### 6.1 Integrity (§4.1)

100/100 requested proteins retained. **80 valid triplets** against the 78 floor, so the primary
capability statement is authorized. 17 proteins were structure-rejected in all three arms
(intrinsically infeasible), 3 were scattered. Zero cells with uncertified caps; one overlay digest;
three distinct run signatures; role-B hotspot drift was telemetry throughout and never a gate.

**One §3.2 gate condition failed and was accepted as a frozen deviation.** The three arms do not
share a depth-0 cloud: the depth-0 root partial states are bit-identical in tokens, but the
recorded per-position log-probabilities are irreproducible at `~1e-5`, they sit inside the state's
content digest, and `lookahead_seed` hashes `source_state_id`. The wobble is arm-independent — a
plain repeat of one arm diverges by the same magnitude — so the design is randomised rather than
confounded, but molecule-level pairing is lost. Full diagnosis, the four repair routes, and the
measured sufficiency of route A are in `doc/DUAL_ALLELE_DUALF0_AUDIT.md` Appendix L.

### 6.2 Mechanism, D0→D1 (§4.2) — POSITIVE

Because the arms do not share a cloud, the donor question is answered exactly and offline instead:
every endpoint carries `u_A` and `u_B`, so all three selection laws are replayed on **one** cloud,
matched by construction. Over 240 clouds the joint law selects a **different donor from A-only in
30.8 %** of them and from B-only in 80 %, and its own pick is B-limited in **50/240 = 21 %**.

The second Head demonstrably enters the actuator; the joint objective is not degenerate to A-only.
The dominant terminal stop is `stall_no_better_donor` (212/240) — the greedy plateau, as in the
source campaign.

### 6.3 Capability (§4.3) — NULL

Protein-mean $\Delta$ over 80 valid triplets, 10,000-draw protein bootstrap, seed `20260821`:

| contrast | mean $\Delta J$ | UCB95 | passes |
|---|---:|---:|---|
| joint final vs joint's own D0 | −0.574 | −0.474 | **yes** |
| joint final vs `a_only` final | +0.046 | +0.117 | no |
| joint final vs `b_only` final | −0.043 | +0.030 | no |

Contrast 1 is *not* the runbook's "shared D0" form; under the deviation it reads as descent under
joint's own objective from its own D0, which alone cannot separate coordination from A steering.

**Raw Head risk at each arm's best common-$J$ endpoint** (`raw_logit`, protein means; every
D0→final movement has UCB95 < 0, so all six are real descents):

| arm | $R_A$ D0 | $R_A$ final | $\Delta R_A$ | $R_B$ D0 | $R_B$ final | $\Delta R_B$ |
|---|---:|---:|---:|---:|---:|---:|
| `joint` | −4.665 | −7.963 | **−3.298** | −3.589 | −5.764 | **−2.174** |
| `a_only` | −4.417 | −8.382 | **−3.965** | −3.891 | −6.058 | **−2.167** |
| `b_only` | −5.287 | −7.692 | **−2.405** | −3.457 | −5.744 | **−2.286** |

This is the informative table, and it says more than the scalar does. Each single-allele arm drives
its own allele hardest, and **joint sits between them on both axes** — the signature of a
compromise objective behaving as designed. But two readings matter:

1. **`a_only` improves role B as much as `joint` does** (−2.167 vs −2.174). Steering on A alone
   already delivers joint's entire B benefit on this cohort, which leaves the joint objective
   nothing to add. Note this is compatible with the panel's near-zero cross-allele level
   correlation ($r = 0.0962$): that measures risk *levels* across natural sequences, not whether
   *improvements* co-move under the same sequence edits.
2. **Joint is worse than `a_only` on both raw alleles**, not only on the reduced scalar:

| cross-arm raw contrast (lower is better) | mean | UCB95 | passes |
|---|---:|---:|---|
| joint − `a_only`, $R_A$ | +0.419 | +0.892 | no |
| joint − `a_only`, $R_B$ | +0.294 | +0.698 | no |
| joint − `b_only`, $R_A$ | −0.271 | +0.184 | no |
| joint − `b_only`, $R_B$ | −0.020 | +0.327 | no |

Per-depth common-$J$ frontier (protein mean) — most of the gain is D0→D1 and all arms are flat by
D3:

| arm | d0 | d1 | d2 | d3 | d4 |
|---|---:|---:|---:|---:|---:|
| `joint` | 0.3119 | −0.1164 | −0.2397 | −0.2618 | −0.2623 |
| `a_only` | 0.3567 | −0.1715 | −0.2719 | −0.3079 | −0.3081 |
| `b_only` | 0.2549 | −0.1173 | −0.1948 | −0.2111 | −0.2191 |

Admissible endpoints per protein: `a_only` 41.1, `joint` 39.4, `b_only` 37.2. Role-B hotspot
telemetry is indistinguishable across arms (mean max-increase 15.0 / 15.0 / 14.8) and was never an
admission gate.

### 6.4 Verdict and claim boundary

**`no_demonstrated_gain_over_either_single_allele_arm`.** Simultaneous Dual steering does not reach
a lower frozen joint objective than either single-allele law on this cohort. Neither single-allele
contrast excludes zero in either direction, so this is *no demonstrated gain*, not a demonstrated
loss.

Three things bound the claim:

- the arms are unpaired at depth zero, so the contrasts carry less power than designed and a small
  true effect could be hidden;
- the frontier is a minimum over pools whose size is itself objective-determined, so the arms are
  not matched on endpoint count; and
- the cohort is high-risk under allele **A**, and A is the binding coordinate in ~79 % of joint's
  own selections, so $J\approx u_A$ most of the time. A cohort where the two alleles genuinely
  conflict is the condition under which simultaneous steering could pay, and this campaign did not
  test one. That is a hypothesis this result raises, not a result it establishes, and §5 stops here.

An analysis defect found after the first read is recorded as amendment A1 in the bundle's
`preflight/analysis_preregistration.json`: `joint_risk` in the artifacts is the **arm's own**
objective value, equal to `u_a` exactly in `a_only` bundles and `u_b` exactly in `b_only` bundles,
so the common frontier must be recomputed as $J=\mathrm{smoothmax}(u_A,u_B)$ — which §4.3 already
says — rather than read from that column. Reading it made joint lose by construction. No seed,
criterion, contrast definition, floor, or admission law changed.

---

## 7. AUTHORIZED: full-data Dual capability campaign for the paper

### 7.1 Scientific question and scope

The §6 campaign answered an internal ablation question on a 0701-selected high-risk cohort. It did
not answer the paper-facing question below:

> On one common cohort selected without either Head, can one structure-feasible Dual-Fusion design reduce
> both DRB1*07:01 and DRB1*04:01 production-Head risk relative to WT and ProteinMPNN?

This campaign tests capability, not whether the joint law beats two variants of itself. Run only
the `joint` arm. Do not run `a_only`, `b_only`, NoD, M1, a temperature sweep, a root sweep, NMP, or
a separate structure experiment.

The §6 result remains valid as a mechanism diagnostic: Head B entered the actuator, but the
0701-high-risk cohort supplied little incremental trade-off beyond A-only steering. It is not the
paper comparator.

### 7.2 Frozen production Heads

Every runtime decision, calibration row, WT score, ProteinMPNN score, Fusion endpoint, and final
analysis row must use the following fixed-epoch full-data checkpoints:

| Role | Checkpoint | SHA-256 |
|---|---|---|
| A, DRB1*07:01 | `/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head_fulldata/drb0701/epoch_29.pt` | `144ae7c808e9c69aa9efad523957a16579045017d1f8192bd97720947cbdc163` |
| B, DRB1*04:01 | `/scratch/gpfs/KAIYIJIANG/zijie/run/epitope_head_fulldata/drb0401/epoch_34.pt` | `12e8018fed5a9255e5e467811cd8c8b4520fd1ab7cdad281d93c75472b6ffb23` |

Use `epitope_head/configs/` with the production `cnn_himp_a1_res03`/LC1 inference contract and the
same score scale and window grid for all sequence sources. Never use either full-data run's
`best.pt`: it is selected on a training-contaminated validation curve. The fixed epochs above are
the only production deliverables.

These Heads have no held-out performance estimate by construction. Their scientific validity is
supported by the separate five-fold CV results; this campaign uses them as deployment-consistent
optimization oracles and does not claim to re-evaluate Head generalization.

The old `dual_overlay_c1_v1.json` is invalid for this campaign because it binds different
checkpoint digests. Reusing it or mixing old endpoint scores with the full-data Heads is a hard
failure.

### 7.3 Task DP0 — freeze the common Head-blind cohort and exact comparator sequences

Build `dual_common_fast_v1.parquet` from the exact intersection of:

- `if_ready/main/test_proteins_if_ready_HLA-DRB1_07_01.parquet`; and
- `if_ready/main/test_proteins_if_ready_HLA-DRB1_04_01.parquet`.

Join by `protein_id` and require exact agreement on reference sequence, sequence length, and the
resolved target-backbone digest. Exclude the 100 proteins used by the §6 high-risk diagnostic.
Before sampling, also require that the canonical ProteinMPNN source contains exactly eight complete
design sequences for the protein; this is an availability requirement and must not inspect scores.

Select exactly 100 proteins with a deterministic, protein-equal uniform allocation over sequence
length and IF sequence-coverage strata. No Head, NMP, WT risk, ProteinMPNN risk, or previous Fusion
outcome may enter cohort selection. Record the eligible intersection, bin counts, seed, selected
IDs, source hashes, and backbone hashes in a cohort manifest before any Head scoring.

The resulting cohort is symmetric and Head-blind, but its two source parquets inherit the project's
historical allele-specific NMP prescreen. Record that provenance and do not call the cohort fully
allele-neutral; no additional NMP value may influence the 100-protein sample.

Prepare three exact sequence surfaces for those same 100 proteins:

1. **WT:** one target sequence per protein from the frozen cohort.
2. **ProteinMPNN:** the existing canonical eight-design generation. Choose one exact sequence
   source and retain all eight original designs; do not combine A scores from one generated panel
   with B scores from another unless `(protein_id, sequence_md5)` matches exactly.
3. **Fusion:** generated later by DP3.

After the cohort and its hard-anchor manifests are frozen, measure

$$
C_{\mathrm{common}}
=
\max_p
\left|
\mathcal A_{\mathrm{editable}}(p)
\right|.
$$

**DP0 acceptance:** the frozen manifest contains 100 proteins with one common reference/backbone,
one WT sequence, eight exact ProteinMPNN sequences, complete source hashes, and the measured
$C_{\mathrm{common}}$. No immune score has influenced membership.

### 7.4 Task DP1 — build the full-data-clean panels, sign the overlay, and score comparators

Use the existing panel builder and retain the frozen scalarization law:

- `scripts/build_dual_calibration_panel.py`;
- `inverse_folding/reference_flow/configs/v2_dual_smoothmax_policy_v1.json`.

Keep $c_u=0.10$ and

$$
\tau = \frac{0.10}{\log 2}.
$$

Do not tune either quantity from the new cohort or campaign outcomes.

Build a new primary natural panel for the full-data Heads. The panel recipe remains canonical
AA20, length 100–500, exact deduplication, homology clustering, one representative per cluster,
and deployment-cohort exclusion. Because the production Heads were trained on train+val+test,
exclude the union of homology to **all three splits** for both alleles; the previous train/val-only
panel cannot be relabelled as full-data-clean.

Produce the required overlap-included companion measurement with the same full-data Heads so the
production loader's panel-sensitivity contract remains satisfied. This is provenance calibration,
not a parameter sweep: the overlap-excluded panel remains primary regardless of the measured
shift.

Run `scripts/calibrate_v2_dual_objective.py` on the primary and sensitivity panels with the two
§7.2 full-data Heads. Produce signed calibration rows and
`dual_overlay_fulldata_common_v1.json`, binding:

- both full-data checkpoint paths and exact SHA-256 values;
- newly measured $b_A,s_A,b_B,s_B$ and repeatability-derived donor/write margins;
- the frozen smooth-max policy and $\tau$;
- both panel identities, the measured sensitivity, and all content digests;
- the DP0 $C_{\mathrm{common}}$ in candidate-sequence units; and
- a projected two-Head ceiling of $2C_{\mathrm{common}}$ logical Head evaluations per cycle.

Do not halve the editable domain because there are two Heads, and do not reuse `454` unless DP0
independently measures that exact value.

Score every DP0 WT and ProteinMPNN sequence with both full-data Heads. Existing evaluation rows may
be reused only after exact checkpoint, config, score-scale, window-grid, protein, and sequence-MD5
validation. Missing cross-allele scores are computed; sequences are never regenerated to fill a
scoring gap.

**DP1 acceptance:** both panels reproduce from their manifests; all WT and ProteinMPNN rows carry
finite $R_A,R_B,u_A,u_B,J_\tau$; and the overlay loads through the production path, resolves
`joint`, binds both checkpoint digests and $C_{\mathrm{common}}$, carries panel sensitivity, and
reproduces from its signed row table.

### 7.5 Task DP2 — materialize and preflight the formal joint campaign

Use the existing V2 surfaces registered in `doc/SCRIPTS.md`:

- `scripts/materialize_v2_canary_config.py`;
- `scripts/preflight_v2_canary_assembly.py`;
- `scripts/run_rf_fusion_v2.py`; and
- `scripts/submit_rf_fusion_v2_canary.slurm`.

Freeze the production search contract:

| Item | Value |
|---|---|
| Cohort | `dual_common_fast_v1.parquet`, 100 proteins |
| Objective arm | `joint` only |
| Heads | the two full-data checkpoints in §7.2 |
| Overlay | `dual_overlay_fulldata_common_v1.json` from DP1 |
| Search | D4/K12/r40, checkpoints 50→60→70→80→90 |
| Roots | four independent roots per protein |
| Cells | 400 |
| Candidate domain | measured $C_{\mathrm{common}}$ per cycle; two-Head accounting $2C_{\mathrm{common}}$ |
| Structure/anchors | the frozen V2 production contract, unchanged |
| Runtime | `immune-design` for calibration, generation, and all Head rescoring |

The four-root choice restores the search scale used by the formal V2 K12 results. It is not a
root ablation, and no one-root arm is run.

Run one model-free materialization/assembly preflight over all 400 cells. It must verify cohort and
backbone identities, root seeds, D4/K12/r40, full-data Head digests, overlay digest,
$C_{\mathrm{common}}/2C_{\mathrm{common}}$ budget,
structure config, generated `.args.sh`, and absence of NMP. There is no separate GPU smoke or
first-protein decision gate: the Dual runtime has already executed 300 cells in §6.

### 7.6 Task DP3 — execute the 400-cell joint campaign

Submit all four roots for all 100 proteins under one campaign identity. Preserve every typed cell
failure, carried incumbent, endpoint/archive table, structure verdict, Dual evidence table, cost
ledger, and resolved launch bundle. Do not replace failed proteins, add roots selectively, or
inspect outcomes before all requested cells finish.

Per protein, merge the four roots, exact-sequence deduplicate, and retain only inherited-V2
structure-feasible endpoints. Materialize:

- one **primary design**: minimum frozen $J_\tau$, deterministic tie by endpoint ID; and
- one **reported panel**: the first up-to-eight unique feasible sequences under the same frozen
  `(J_\tau, endpoint_id)` order.

Both raw allele risks reported for a design must come from that same exact sequence. Do not select
one sequence for A and another for B.

### 7.7 Task DP4 — common full-data evaluation against WT and ProteinMPNN

Apply the same DP1 calibration and full-data Head scorers to all three methods.

For ProteinMPNN, define its primary design as the minimum-$J_\tau$ sequence among its original
eight designs, with the same deterministic tie law. WT has one sequence. The primary comparison is
therefore one exact selected design per method and protein. Also report the full Fusion up-to-eight
panel and all eight ProteinMPNN designs as an output-distribution sensitivity.

Primary paired quantities, lower is better:

1. Fusion minus WT for raw $R_A$;
2. Fusion minus WT for raw $R_B$;
3. Fusion minus ProteinMPNN for raw $R_A$;
4. Fusion minus ProteinMPNN for raw $R_B$; and
5. the corresponding frozen $J_\tau$ contrasts.

Use protein-equal summaries and 10,000 protein bootstrap draws with one predeclared analysis seed.
The paper-facing Dual capability claim requires the one-sided 95% upper bound to be below zero for
all four raw-risk contrasts:

$$
\operatorname{UCB}_{95\%}
\left[
\operatorname{mean}
\left(
R_a^{\mathrm{Fusion}}-R_a^{\mathrm{comparator}}
\right)
\right] < 0,
\qquad
a\in\{A,B\},
\quad
\mathrm{comparator}\in\{\mathrm{WT},\mathrm{ProteinMPNN}\}.
$$

Also report, without turning them into new gates:

- fraction of proteins on which the Fusion primary design improves both raw allele risks relative
  to each comparator;
- two-dimensional Pareto dominance/tie/incomparability counts;
- WT-risk-stratified readout, with strata defined only after the cohort is frozen;
- per-depth $J_\tau$, $R_A$, and $R_B$ frontiers;
- output uniqueness/diversity and realized candidate/refold cost; and
- scTM, recovery, and structure-feasible coverage as do-no-harm evidence.

Do not independently optimize or select on the two raw allele columns during analysis. The frozen
joint objective selects membership; raw $R_A$ and $R_B$ test whether that one selection actually
improves both biological directions.

### 7.8 Verdict, evidence return, and stop

| Result | Verdict |
|---|---|
| all four raw-risk UCBs are below zero; integrity and structure gates pass | `dual_fulldata_capability_supported` |
| both alleles beat WT but one does not beat ProteinMPNN | `dual_biological_descent_supported_comparator_gain_partial` |
| only one raw allele improves | `single_axis_only`; no Dual capability claim |
| fewer than 80 proteins have complete WT/ProteinMPNN/Fusion evidence | `undercovered_unresolved` |
| Head/checkpoint/cohort/backbone/objective identity mismatch | invalid campaign |

Return the cohort manifest, full-data calibration rows/overlays, all requested cell bundles,
materialized Fusion/WT/ProteinMPNN panels, paired protein table, structure/cost tables, and resolved
configs through the standard RF/RAR path. Register one RAR containing the deterministic analysis
program, exact inputs and hashes, primary results, raw per-allele evidence, and claim boundary.

After DP4, stop. Do not automatically add single-allele arms, extra roots, a tau sweep, NMP,
conflict-selected cohorts, or compute-matched ProteinMPNN generation. The primary comparison is
one selected exact design per method and protein. The up-to-eight sensitivity reports its actual
row coverage and is called eight-vs-eight only on complete cases. Neither read claims equal
upstream search compute.
---

## 7.9 EXECUTED (2026-08-26)

### 7.9.1 Identity

| Item | Value |
|---|---|
| Campaign | `dual_fulldata_joint_d4k12_r40_4root_0701x0401__4315531` |
| Cells | 400 = 100 Head-blind proteins × 4 roots, `joint` only |
| Root seeds | 20260826 / 27 / 28 / 29 (a root IS a `master_seed` offset) |
| Cohort | `dual_common_fast_v1`, `cohort_sha256 = b0921a6dd159e86c…` |
| Overlay | `dual_overlay_fulldata_common_v1.json`, `content_digest = 852a98360e66…` |
| Objective digest (`joint`) | `0d8414bdeeee` |
| $C_{\mathrm{common}}$ | **483** (`4QQQ_A`), $2C = 966$ |
| Head A / B | `epoch_29.pt` `144ae7c808e9…` / `epoch_34.pt` `12e8018fed5a…` |
| Environment | `immune-design`, ailab H200, one environment throughout |
| Preflight | 26/26 checks pass; 35 executed modules pinned by digest |

### 7.9.2 The cohort (DP0)

Funnel, every step measured:

```
2879 / 2829   the two alleles' IF-ready tables
  →    221    protein_id intersection
  →    206    identical on sequence, length, if_sequence_sha1, if_source_structure,
              if_chain_id and pdb_path — a protein the two tables describe differently
              is not one common target but two
  →    198    minus the 100 §6 high-risk proteins (only 8 were in the intersection:
              §6's cohort was 0701-specific)
  →    198    ProteinMPNN panel complete: exactly 8 designs, each the target length,
              canonical AA20 — availability only, no score read
  →    177    one representative per exact reference sequence
  →    100    protein-equal uniform allocation over 5 length quantiles × 2 coverage
              halves; all ten strata filled exactly, minimum eligible 12
```

Two facts the §7.2 claim boundary needs:

* **Zero exact-sequence overlap** between this cohort and either production Head's complete
  fitting pool (train ∪ val ∪ test, 1308 and 1865 proteins). The full-data Heads have no
  held-out estimate by construction, but they have not seen these proteins.
* Backbones are byte-identical under both alleles' PDB roots, so "one common target" is proved
  rather than assumed, and re-proved at the preflight against the digests DP0 froze.

**Deviation, recorded:** exact-sequence duplicates were collapsed to one representative. §6 carried
eight id groups sharing one reference sequence and could only report it, because that cohort was
inherited. This one is built here and the primary read is a protein-equal bootstrap, in which two
ids carrying one molecule are one draw wearing two names on the outcome axis — their WT risk is
identical by construction. The collapse is Head-blind and outcome-blind; the full 198-protein pool
and all 21 dropped ids are in the manifest.

### 7.9.3 The calibration (DP1), and what got worse

| Quantity | eval-split (§6) | **full-data (§7)** |
|---|---|---|
| $b_A$ / $s_A$ | −6.768 / 6.291 | **−8.834 / 6.074** |
| $b_B$ / $s_B$ | −2.670 / 4.112 | **−9.164 / 5.018** |
| $\theta = b_A/s_A - b_B/s_B$ | −0.4265 | **+0.3719** |
| $\operatorname{SE}(\theta)/c_u$ | 0.240 | **0.524** |
| $\lvert\Delta\theta\rvert/c_u$ (overlap inclusion) | 0.0057 | **0.503** (`within_credit`) |
| cross-allele $r$ | 0.0962 | **0.0925** |
| decision margin | 4.639e-07 (role B) | **1.256e-06** (role A) |
| panel families | 13,836 (train∪val) | **13,757** (train∪val∪test) |

$\theta$ changes sign, and almost all of the movement is role B: $b_B$ goes from −2.670 to −9.164
while $b_A$ moves −6.768 → −8.834. The full-data 0401 Head calls natural background much less
risky than the eval-split fold-0 Head did. This is the re-measurement §7.4 requires, not an error —
but it means **§6's $J$ and §7's $J$ live in different coordinate systems and their values are not
comparable.**

Two calibration statistics are materially worse than §6's and both must travel with the result:

* $\operatorname{SE}(\theta)/c_u = 0.524$. §6's AUDIT J.4 already recorded 0.240 as decision D3
  ("PLAN §5 requires $\operatorname{SE}\ll c_u$; one fifth is not obviously much less than") and
  resolved it by accepting. This is one half. It is not a panel-size effect — 13,757 vs 13,836 is
  a 0.6 % change — and J.3 established that it cannot be improved by enlarging the panel: 44,021
  eligible sequences cluster into ~13.8 k homology-independent units, and reaching $c_u/10$ would
  need roughly four times as many independent families as the deduplicated Tier-2 pool contains.
* $\lvert\Delta\theta\rvert/c_u = 0.503$, against §6's 0.0057. Putting the two Heads' training
  homologs back into the panel moves the equal-risk line by half the credit band. The label is
  still `within_credit`, but the margin is now thin where it used to be three orders clear.

**Why this does not enter the primary claim.** $R_A$ and $R_B$ are raw Head outputs. The affine
coordinates decide only WHICH design the frozen rule selects; they do not enter the paired contrast
that design is then tested by. What $\theta$'s uncertainty widens is the band in which "which
allele is limiting" is a calibration statement rather than a measurement — a mechanism reading, not
a capability gate. This is recorded in the read's own `claim_boundary` (amendment A1, made before
any comparator score and before any campaign cell existed).

### 7.9.4 Two mechanical prerequisites §7 did not name

Both are consequences of §7.2's checkpoint change, not scientific choices, and both are hard
blockers measured against the code rather than argued:

1. **The single-Head V2 policy calibration had to be rebuilt.**
   `materialize_v2_canary_config.py:488` refuses a `--policy-calibration-json` whose
   `head.head_checkpoint_digest` differs from the cell's Head, and `--exploratory-profile` — frozen
   by §7.5 — requires that artifact. The inherited `head_policy_v2_eps005_c454.json` binds the CV
   fold-0 `best.pt`. `calibrate_v2_head_policy.py:154` then refuses to rebuild it against existing
   bundles for the same reason, so a **live-vs-live** mode was added: the bundle supplies only the
   bytes, both passes are the live Head, shuffled and at a different window batch size. AUDIT J.2
   records that varying only the ORDER measures a drift of exactly zero, so an identically batched
   repeat is refused with that reason. Measured on the SAME 1644-observation byte population the
   inherited artifact used, so old and new differ only by the instrument.

   Its two scalars are **inert in a Dual campaign** — verified at both decision sites:
   `reward.py:519` sets `compared_epsilon = joint.epsilon` under an overlay, and `policy.py:1574`
   sets the write tolerance to `dual.objective.decision_margin`. The artifact must still bind them.

2. **§7.5's preflight lists no producer for the per-protein substrate** the materializer needs:
   `references/<pid>.seq`, `references.json`, `strata.json`. All are derivable from DP0's frozen
   cohort with no GPU, and are now emitted by the cohort freezer.

### 7.9.5 What was inherited, and what it cost

* `global_relaxed_hotspot.json` declares `cohort_independent: true` with value 1e6 — the
  whole-landscape hotspot gate is effectively disabled, so it carries across cohorts by
  construction.
* `global_B_r40.json`, the step-indexed maturity band, was measured on the 0701 high-risk cohort
  and is reused deliberately: `materialize_v2_canary_config.py` checks only that a cell exists at
  `(step=40, stratum_key)`, and re-measuring means a fresh D8/K32 ceiling campaign, far outside
  §7.5's "frozen V2 production contract, unchanged". §7.5 also forbids a separate GPU smoke gate,
  so the transfer was measured after the fact instead of assumed. **Realized cost: ZERO** --
  `null_band_incompatible` on 0 of 400 cells. The r=40 maturity clock transfers to this cohort.

### 7.9.6 Integrity

| | |
|---|---|
| Cells requested / executed | 400 / 400 |
| Cells producing a usable lineage | 339 `ok`, 61 `null` |
| Cells whose realized caps were not certified within | **0** |
| Distinct overlay digests across all 400 manifests | **1** (`852a98360e66`) |
| Distinct master seeds | 4, one per root |
| `null_band_incompatible` | **0** |
| Proteins with a structure-feasible Fusion design | **89 / 100** (floor for a resolved verdict: 80) |
| Proteins with complete WT + ProteinMPNN + Fusion evidence | **89** |

Typed outcomes over 400 cells: `committed` 785, `invalid_projection` / `null_invalid_policy_result`
291, `no_admissible_endpoint` / `null_no_admissible_endpoint` 69, `depth_cap` 40. Per cell these
are 1.96 / 0.73 / 0.17 / 0.10 against §6's 1.95 / 0.68 / 0.20 / 0.12 -- the runtime behaved as it
did in the executed diagnostic.

Structure, as do-no-harm evidence: 14,220 endpoints evaluated, **87.96 % structure-feasible**,
scTM mean **0.873** (§6: 0.87). Sequence recovery is NOT reported: the v0 structure gate's
`metrics_json` carries `pLDDT`, `scRMSD` and `scTM` only, so §7.7's recovery line has no source in
this bundle. It is a gap in the requested evidence, not a value withheld.

The eleven proteins with no feasible design are the campaign's own attrition: every one of their
four roots produced endpoints that failed the inherited scTM >= 0.70 gate. §6 lost 17 of 100 the
same way with a single root; four roots recover part of that.

### 7.9.7 Result

**Verdict: `dual_fulldata_capability_supported`.** All four pre-registered raw-risk upper bounds
are below zero.

| contrast (lower is better) | mean | median | UCB95 | improved | gate |
|---|---:|---:|---:|---:|:--:|
| Fusion − WT, $R_A$ | −4.763 | −1.089 | **−3.803** | 86/89 | pass |
| Fusion − WT, $R_B$ | −5.454 | −2.015 | **−4.402** | 84/89 | pass |
| Fusion − ProteinMPNN, $R_A$ | −1.111 | −0.237 | **−0.499** | 76/89 | pass |
| Fusion − ProteinMPNN, $R_B$ | −1.126 | −0.403 | **−0.520** | 80/89 | pass |
| Fusion − WT, $J_\tau$ | −1.375 | −1.579 | −1.177 | 96 % | — |
| Fusion − ProteinMPNN, $J_\tau$ | −0.265 | −0.060 | −0.137 | 89 % | — |

Raw levels, protein-equal means over the 89:

| | $R_A$ | $R_B$ | $J_\tau$ |
|---|---:|---:|---:|
| WT | −4.353 | −3.988 | +1.351 |
| ProteinMPNN (min-$J$ of 8) | −8.005 | −8.316 | +0.240 |
| ProteinMPNN redraw (min-$J$ of 8) | −8.043 | −8.411 | +0.228 |
| **Fusion primary** | **−9.116** | **−9.442** | **−0.024** |

Two-dimensional Pareto counts: against WT **82 both-improved**, 1 dominated, 6 incomparable;
against ProteinMPNN **75 both-improved**, 8 dominated, 6 incomparable.

**The gate is on the mean, and the conclusion does not depend on that choice.** The mean is more
negative than the median in every contrast -- a heavy left tail of large wins -- but the sign is
carried by the majority of proteins, not by the tail: a sign test on the same pairs gives
$p = 1.9\times10^{-22}$ / $7.1\times10^{-20}$ against WT and $2.7\times10^{-12}$ /
$1.2\times10^{-15}$ against ProteinMPNN. Both statistics agree.

**The pre-registered comparator redraw holds.** Scored against a ProteinMPNN panel that shares
**zero** sequences with the primary one on this cohort, the contrast is −1.073 ($R_A$, UCB −0.439)
and −1.031 ($R_B$, UCB −0.502), against −1.111 / −1.126. Per-protein the comparator moves a lot
(sd ≈ 2.2--2.5 raw logit, a quarter of proteins by more than 0.5), and this is exactly why it was
registered before any Head scored anything: the mean contrast is stable under a re-draw, so the
comparator's own sampling noise did not manufacture the result.

#### What this does NOT establish

* **Not equal search.** Fusion's primary design is the minimum-$J$ of a median of **153**
  structure-feasible unique candidates merged over four roots (q1 122, q3 168, max 216);
  ProteinMPNN's is the minimum-$J$ of **8**. That is ~19× the selection pool. §7.8 declares this
  and it is not repaired by the data: this is a comparison of two deployed pipelines at their own
  operating points, not a compute-matched contest.
* **The margin over ProteinMPNN is modest at the typical protein.** Median −0.24 ($R_A$) and −0.40
  ($R_B$) raw logit, and Fusion is *worse* on 15 % / 10 % of proteins. Against WT the same
  medians are −1.09 and −2.01.
* **It is not uniform across risk.** On the WT-risk-stratified read (strata defined after the
  cohort was frozen, from the WT axis only), the high-risk third improves by −8.10 / −11.92 and the
  middle third by −5.73 / −3.61, but the **low-risk third's $R_A$ bound does not clear zero**
  (mean −0.49, UCB +0.34). On proteins that already carry little role-A risk, this campaign does
  not demonstrate role-A improvement.
* **The Heads have no held-out estimate**, by construction (§7.2). What DP0 established is weaker
  and worth stating exactly: zero exact-sequence overlap between this cohort and either Head's
  complete fitting pool.
* **The equal-risk line is loosely measured** on the full-data panel ($\operatorname{SE}/c_u$
  0.524, overlap-inclusion $\lvert\Delta\theta\rvert/c_u$ 0.503). This does not enter the four
  raw-risk gates, which never touch the coordinates; it widens the band around any statement about
  *which* allele is limiting.

#### One mechanism observation, reported not claimed

`active_worst` -- which allele the smooth-max is currently binding on -- is **50/50 on WT** and
47/53 on the ProteinMPNN primary, but **81/89 role A** on the Fusion primary. In normalized units
the campaign moved role B by 1.087 and role A by 0.784. The joint law drives down whichever
coordinate is worse, so an end state this asymmetric says role A stopped yielding first: on this
cohort the two alleles are **not equally steerable**, and the objective spent its remaining effort
where it could still move. That is a hypothesis this run generates, not one it tests -- it is also
the cleanest available contrast with §6, where the cohort was A-selected and A was limiting from
the start.

### 7.9.8 Execution defects found and fixed

Three, all of one kind: **a producer emitting something the runtime refuses**, and in two cases a
test that was easier to satisfy than the consumer.

| # | Defect | How it surfaced | Cost |
|---|---|---|---|
| 1 | DP0 wrote the reference file as `sequence + "\n"`. `load_complete_reference` reads it "as ASCII with NO normalization" and digests the exact bytes, so the newline IS a residue. | `V2LookaheadError: sequence carries non-canonical character(s) ['\n']`, once per cell, after the GPU was allocated. 160 cells finished with `n_ok=0` before the campaign was cancelled. | ~10 GPU-hours |
| 2 | `calibrate_v2_head_policy` hardcoded the v1 `lineage_incumbent_depth0_rule`. `HeadDirectedCappedPolicy` binds version and rule in both directions: v2 is RESERVED for `best_admissible_depth0`. | `V2PolicyError: policy_version='v2' is reserved for best_admissible_depth0` | ~10 min (a single probe shard caught it) |
| 3 | The same producer's policy-spec enum admitted only `v1`, while every executed V2 campaign supplies the v2 spec. | Refused the real spec at materialization time. | none (caught locally) |

Defect 1's test asserted `path.read_text().strip() == sequence`; the `.strip()` let a one-character
defect pass twenty assertions and then fail every cell. It now compares bytes, and a second test
asserts the downstream contract directly — no terminator, nothing outside canonical AA20. The
earlier comparator-scorer defect had the same shape: a test fake carried `window_grid_digest` on
the result object, while the real `V2HeadResult` carries it on `.binding`.

Defect 2 was found in one round rather than several by diffing the generated artifact field by
field against the executed K12 one instead of chasing runtime errors: the `head_directed` key sets
matched exactly and that rule was the ONLY structural difference. The same technique applied to the
resolved configs showed zero structural difference from §6 — same key set, and only five expected
value differences (head-call cap 6000→6400 scaled by $C$, $C$ itself, the two inert margins, and
the epsilon's provenance label).

### 7.9.9 Runtime behaviour matched §6

Before the full launch, one probe shard was run and its aggregate compared with the executed §6
joint arm. This is an operational check, not the first-protein decision gate §7.5 forbids: no risk
value or contrast was read.

| | §6 joint (100 cells) | §7 probe (34 cells) |
|---|---|---|
| `n_ok > 0` | 82 % | 82 % |
| scTM mean / median | 0.87 / 0.94 | 0.880 / 0.948 |
| scTM ≥ 0.70 | 89 % | 85 % |
| `committed` per cell | 1.95 | 1.85 |
| `null_invalid_policy_result` per cell | 0.68 | 0.65 |
| `invalid_projection` per cell | 0.68 | 0.65 |
| `null_no_admissible_endpoint` per cell | 0.20 | 0.24 |
| `depth_cap` per cell | 0.12 | 0.12 |

One caution this run earned: the probe's FIRST cell (`1QFH_A`, 212 aa, coverage 1.00) had all
twelve depth-0 endpoints rejected at scTM 0.50–0.56 against a 0.70 gate, with pLDDT ≈ 83. Length
was checked and ruled out — §6 passes 83–96 % in every length bin including <120 aa — as were the
backbone path convention, backbone coverage, and the resolved config. It was simply a hard protein:
§6 itself had 17 of 100 proteins infeasible in every arm. The aggregate is the readable quantity;
`n = 1` is not.
