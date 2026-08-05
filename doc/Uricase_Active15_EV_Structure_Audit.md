# Active-15 Uricase Evolution and Tetramer-Structure Audit

## Scope and claim boundary

This analysis covers the 15 experimentally active, single-domain characterized
uricases in the RAR 0034 redesign panel. It builds per-parent evolutionary,
tetramer-contact, interface-energy, and WT epitope-core evidence. It does not
select a parent/allele, generate an RF constraint manifest, or treat any
computational metric as proof of activity.

The arithmetic recovery values below are sequence-identity lower bounds under
the counterfactual that every position left open by a policy changes. They are
not experimental activity recovery and do not imply that an opened 9-mer core
will be eliminated immunologically.

## Provenance

- Source git HEAD: `e091b36876ddbbeac1c548c56d9fe7f7d113ba26`.
- The analysis implementation was uncommitted at run time; exact implementation
  and input/output SHA-256 values are recorded in
  `04_join_v1/evidence_join/join_metadata.json` and the per-stage metadata.
- Frozen parent FASTA SHA-256:
  `6106d5b05879ae2e1f7edafec631f625dc127e5982453e422ff967c3bdcc483a`.
- Frozen evidence-manifest SHA-256:
  `77671193d658cf03ce99ff1c27ccc122cb3c2b3559ef1a9674dd04a59addf901`.
- Primary cluster root:
  `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/therapeutic_enzymes/uricase_active15_ev_structure_v1`.
- PLMC model root:
  `/scratch/gpfs/KAIYIJIANG/zijie/run/therapeutic_enzymes/uricase_active15_ev_structure_v1/plmc`.
- Final joined evidence:
  `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/therapeutic_enzymes/uricase_active15_ev_structure_v1/04_join_v1/evidence_join`.

All production stages were run as one whole-cohort job, not arrays or shards:

| Stage | Job | Allocation | Wall time | Status |
|---|---:|---|---:|---|
| 15-query ColabFold MSA | 11901471 | ailab, 8 CPU + GPU | 3m54s | completed |
| MSA normalization/QC | 11902558 | RTX6000 CPU-only, 32 CPU | 5m10s | completed |
| 15 PLMC fits, sequential | 11903824 | RTX6000 CPU-only, 32 CPU | 39m06s | completed |
| 15 evolution analyses, sequential | 11905624 | RTX6000 CPU-only, 4 CPU | 5m57s | completed |
| 15 contact/energy analyses, sequential | 11905871 | RTX6000 CPU-only, 2 CPU | 35m39s | completed |
| Final 15-parent join | 11907087 | RTX6000 CPU-only, 4 CPU | 57s | completed |

## Cohort and join integrity

- 15 unique parents and 4,548 canonical positions.
- 273 strong WT cores: 110 for DRB1*04:01, 107 for DRB1*07:01, and
  56 for DRB1*15:01.
- 2,457 core-position rows, exactly 273 cores times 9 registers.
- 18,564 core-policy-mask rows, exactly 273 cores times 4 policies times
  17 evidence masks.
- 3,060 parent-allele-policy-mask rows, exactly 45 parent-allele cells times
  4 policies times 17 masks.
- 15 EC/contact-validation rows.
- The no-mask policy rows exactly reproduce all 180 RAR 0034 arithmetic
  recovery baselines.
- An independent recomputation from the complete EC tables and five-sample
  residue-pair contacts reproduced all 15 T6 rows exactly.

## MSA and PLMC evidence quality

The uniform 15-query search produced 259,777 raw rows and 259,069 normalized
rows. Normalization removed 693 invalid rows and collapsed 15 duplicate source
records. It retained 1,053 rows whose first header token collided but whose full
headers represented distinct hits. Fourteen parents had an exact natural WT hit;
`A0A9P8P4R1` did not.

At the frozen coverage-0.6, identity-0.8 definition:

- 7 parents have `N_eff/L >= 10` and are covariance-qualified:
  `A0A100I4D7`, `H2ETE7`, `O74409`, `Q00511`, `Q0CXR4`, `Q7SBV5`, and
  `Q9RV70`.
- 8 parents have `5 <= N_eff/L < 10`; covariance is exploratory and cannot
  create a hard lock under the preregistered rule.
- No parent is covariance-insufficient.

All 15 PLMC models completed 500 iterations, matched the exact WT target,
covered indices `1..L`, and had every site valid. All reported
`ROUNDING_ERROR` as the optimizer termination status; this is retained as a
diagnostic rather than treated as model failure because the exact target,
index, all-site, regularization, and downstream structural gates passed.

## Conservation and covariance

Conservation locks require both gap-excluded entropy conservation and weighted
WT identity frequency across the 0.6/0.7/0.8 coverage filters. Sigma masks are
computed independently of conservation: they apply only a maximum gap fraction
of 0.5. There is no conservation or `C_i` filter in sigma.

Across all 15 parents:

- `C80_stable`: 410 positions.
- `sigma80_robust`: 558 positions before the Neff hard-lock eligibility rule.
- `C80_stable` and `sigma80_robust` have zero overlap.
- Among the 7 qualified parents, 262 positions are `sigma80_robust`; only 35
  overlap the homolog-specific full 5-of-5 contact mask. Thus 227/262 are not
  covered by full structure contacts.
- Inside strong WT cores for qualified parents, 97 unique parent-allele
  positions are `sigma80_robust`; 82/97 remain outside both `C80_stable` and
  the full 5-of-5 contact mask.

Sigma depends measurably on the EC cutoff. Cohort stability across `0.5L`, `L`,
and `2L` was:

| Metric | Cutoffs | Min | Median | Max |
|---|---|---:|---:|---:|
| Spearman | 0.5L vs L | 0.8409 | 0.8635 | 0.8732 |
| Spearman | 0.5L vs 2L | 0.6611 | 0.7423 | 0.7609 |
| Spearman | L vs 2L | 0.8591 | 0.8937 | 0.9039 |
| Top-10% Jaccard | 0.5L vs L | 0.4667 | 0.6216 | 0.7222 |
| Top-10% Jaccard | 0.5L vs 2L | 0.2692 | 0.4419 | 0.5897 |
| Top-10% Jaccard | L vs 2L | 0.4348 | 0.5897 | 0.7222 |

This is why `sigma_robust_pct`, the minimum percentile across all three
cutoffs, is the appropriate analytical sigma knob. A single top-L set is too
cutoff-sensitive to be treated as a stable coupling mask.

## Tetramer-aware covariance validation

The table reports 5-of-5 contact precision. `mono` uses within-chain contacts;
`tetra` adds only catalytic and assembly contacts observed in both symmetry
copies within a sample. Diagonal contacts are excluded.

| Parent | Covariance tier | N_eff/L | L/2 mono -> tetra | L mono -> tetra | 2L mono -> tetra | Sigma status |
|---|---|---:|---:|---:|---:|---|
| A0A100I4D7 | qualified | 10.676 | 0.709 -> 0.828 | 0.570 -> 0.679 | 0.409 -> 0.493 | eligible |
| A0A9P8P4R1 | exploratory | 8.612 | 0.646 -> 0.756 | 0.488 -> 0.582 | 0.346 -> 0.418 | descriptive |
| D0VWQ1 | exploratory | 9.571 | 0.762 -> 0.874 | 0.583 -> 0.695 | 0.422 -> 0.503 | descriptive |
| H2ETE7 | qualified | 10.268 | 0.765 -> 0.872 | 0.589 -> 0.687 | 0.412 -> 0.490 | eligible |
| O74409 | qualified | 11.074 | 0.709 -> 0.824 | 0.595 -> 0.693 | 0.429 -> 0.507 | eligible |
| P04670 | exploratory | 9.979 | 0.652 -> 0.748 | 0.521 -> 0.621 | 0.409 -> 0.476 | descriptive |
| P09118 | exploratory | 9.897 | 0.770 -> 0.888 | 0.591 -> 0.696 | 0.413 -> 0.498 | descriptive |
| P16164 | exploratory | 9.953 | 0.757 -> 0.862 | 0.589 -> 0.691 | 0.403 -> 0.482 | descriptive |
| P25688 | exploratory | 9.863 | 0.750 -> 0.849 | 0.601 -> 0.700 | 0.421 -> 0.500 | descriptive |
| P25689 | exploratory | 9.911 | 0.737 -> 0.849 | 0.595 -> 0.691 | 0.401 -> 0.479 | descriptive |
| Q00511 | qualified | 10.703 | 0.669 -> 0.788 | 0.566 -> 0.675 | 0.412 -> 0.495 | eligible |
| Q0CXR4 | qualified | 10.767 | 0.709 -> 0.821 | 0.575 -> 0.681 | 0.415 -> 0.500 | eligible |
| Q6P700 | exploratory | 9.908 | 0.730 -> 0.849 | 0.589 -> 0.694 | 0.400 -> 0.488 | descriptive |
| Q7SBV5 | qualified | 10.996 | 0.696 -> 0.804 | 0.573 -> 0.671 | 0.420 -> 0.497 | eligible |
| Q9RV70 | qualified | 10.601 | 0.664 -> 0.785 | 0.547 -> 0.648 | 0.408 -> 0.480 | eligible |

The tetramer-minus-monomer gain is positive for every parent at all three
top-set sizes. At top-L it ranges from +0.0945 to +0.1126. Fourteen of fifteen
parents pass the inherited top-L tetramer precision threshold of 0.60; the only
failure is `A0A9P8P4R1` at 0.5823. All 7 Neff-qualified parents pass. Seven of
the 8 exploratory parents also pass structurally, but remain descriptive because
of the independent MSA-depth gate.

## Homolog-specific tetramer structure

All 15 five-sample WT ensembles have complete four-chain mapping and a unique
two-copy catalytic relay in every sample. Fourteen parents use catalytic
matching `A:B|C:D`, assembly matching `A:C|B:D`, and diagonal matching
`A:D|B:C`. `D0VWQ1` instead uses catalytic `A:D|B:C`, assembly `A:C|B:D`, and
diagonal `A:B|C:D` in all five samples. This is direct evidence that chain-label
or Q00511-index transfer is unsafe even when the D2 topology is preserved.

The 5-of-5 masks are stable but not static. Depending on the parent, the
1-of-5-minus-5-of-5 boundary contains 1-20 catalytic-contact positions and
1-19 full-contact positions. The 5-of-5 mask is therefore retained as primary;
4-of-5 and 1-of-5 remain sensitivity layers.

## Interface energy

Rosetta alanine scanning used one representative ensemble sample per parent and
both biological interfaces. Across 4,548 positions:

- 1,862 are measured and interpretable.
- 300 native Ala/Gly/Pro positions are scanned but non-interpretable for an
  alanine side-chain claim and remain unknown, not low-energy.
- 2,386 are not interface-scanned and are false for the descriptive energy
  masks.
- 620 positions have interpretable maximum mean `ddG_bind >= 1` REU and 337
  have `ddG_bind >= 2` REU.
- 616/620 positions at `>=1` REU already lie in the full 5-of-5 contact mask.

Energy therefore refines/ranks the contact surface; it is not an independent
position-space axis comparable to sigma. It remains descriptive because the
thresholds are not experimentally calibrated and only one ensemble sample was
scanned. Q00511 Protenix sample 0 is the production energy source; the 1R51
crystal remains a separate experimental control and is not inserted into the
five-sample SHA contract.

## WT epitope-core trade-off

No-mask arithmetic recovery:

| Opening policy | Median open positions [range] | Median recovery [range] | Cells in 90-97% band |
|---|---:|---:|---:|
| All 9 core residues | 54 [9, 91] | 82.18% [70.39, 97.02] | 8/45 |
| P1/P4/P6/P9 | 24 [4, 42] | 92.05% [86.18, 98.68] | 26/45 |
| P1+P4 | 12 [2, 22] | 96.03% [93.09, 99.34] | 33/45 |
| P1 only | 6 [1, 11] | 97.98% [96.38, 99.67] | 4/45 |

For the P1+P4 policy, aggregate mask collisions are:

| Evidence mask | Blocked/open positions | Cores with both P1 and P4 blocked | Role |
|---|---:|---:|---|
| C80 stable | 64/534 (12.0%) | 0/273 | single-site WT conservation |
| Qualified sigma80 robust | 35/534 (6.6%) | 7/273 | eligible covariance only; exploratory sigma excluded |
| Mid contact 5-of-5 | 161/534 (30.1%) | 58/273 | catalytic interface |
| Full contact 5-of-5 | 306/534 (57.3%) | 113/273 | catalytic plus assembly interfaces |
| Energy >=1 REU | 154-182/534 (28.8-34.1%) | unknown-bounded | descriptive, AGP-aware |

The union of C80, qualified sigma80 robust, and mid-contact 5-of-5 blocks
214/534 P1+P4 positions (40.1%). Replacing mid with full blocks 333/534 (62.4%).
This quantifies the intended role of full structure: it is a strong rescue-axis
endpoint but removes too much immune-search space to be a final policy.

Register-level evidence is not interchangeable:

| Register | C80 | Qualified sigma80 | Mid contact 5-of-5 | Full contact 5-of-5 | Energy >=1 REU | Functional analog |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 5.9% | 9.2% | 30.8% | 59.3% | 35.5% | 0.0% |
| P4 | 17.6% | 4.8% | 28.2% | 54.2% | 21.6-31.9% | 9.9% |
| P6 | 10.3% | 1.8% | 29.7% | 53.8% | 11.4-15.8% | 2.6% |
| P9 | 22.3% | 2.9% | 27.8% | 39.2% | 20.1-29.3% | 2.6% |

P1-only has no overlap with the homolog-specific functional annotations, but it
is highly enriched for stable interface contact and positive alanine-scan energy.
Therefore zero active-site overlap does not make P1 intrinsically activity-safe.

## Functional analog annotations

The homolog-specific annotation is complete for 15 parents times 9 roles. It is
kept separate from the older identity-required map so nonidentity analogs are
not silently dropped.

Within P1+P4 opening positions there are 27 functional-analog overlaps across
26 parent-allele cells:

- DRB1*04:01: all 15 parents, 16 unique parent-allele positions.
- DRB1*07:01: 11 parents, 11 positions.
- DRB1*15:01: no overlap.
- All 27 are captured by C90, C80, catalytic/full 5-of-5 contacts, and energy
  `>0` REU.
- Only 3/27 reach energy `>=1` REU, so a 1- or 2-REU rule cannot replace the
  functional annotation or contact/conservation evidence.
- Sigma80 robust captures 0/27.

Thus sigma is complementary rather than substitutive: it supplies positions
missed by single-site conservation and structure, but it does not protect the
known catalytic relay by itself.

## Potts calibration

Raw Hamiltonians are never compared across parents. WT weighted percentiles in
each parent's own natural reference range from 28.71% (`O74409`) to 99.97%
(`P25689`) even though all 15 WTs are experimentally active. `H2ETE7` is 55.31%,
and Q00511 is 93.48%. Therefore WT percentile is not a universal activity gate.

`A0A9P8P4R1` has only 8 unique high-coverage natural sequences after WT
imputation; its 94.74% weighted percentile is not a stable empirical
calibration. No numeric minimum-unique threshold was preregistered, so this is
reported as insufficiently calibrated rather than replaced by an invented
cutoff. Future designs must be evaluated only by within-parent `Delta H` versus
their own WT and their own natural reference.

## Scientific assessment

1. Tetramer-aware validation is necessary. The approximately 9.5-11.3
   percentage-point top-L gain is universal across the active panel and explains
   29-34 otherwise missed top-L EC pairs per parent.
2. Full 5-of-5 contact is a legitimate most-conservative rescue endpoint, but
   it cannot be the final design rule: it blocks 57% of P1+P4 opening positions
   and completely closes both anchors in 113/273 cores.
3. Robust sigma is the cleanest new axis. It has zero C80 overlap and little
   full-contact overlap, while the T6 structure check passes for every
   covariance-qualified parent. It should remain depth-gated: the 8 exploratory
   parents can use sigma for ranking, not hard vetoes.
4. Conservation/contact and sigma protect different mechanisms. The former
   recover all 27 known functional-analog collisions in P1+P4; sigma recovers
   none. A sigma-only rule is therefore incomplete even though sigma is the
   least redundant axis.
5. Energy is useful inside the interface for hotspot ranking and as a
   sensitivity layer, not as an automatic lock or an independent search-space
   axis.
6. The next design decision should compare a ladder of constraints under the
   same RF dynamics, with full contact retained as the rescue-positive endpoint
   and less compressive mid/conservation/qualified-sigma combinations defining
   the release direction. This audit deliberately does not choose that ladder's
   final parent, allele, or manifest.

## Unresolved experimental and computational questions

- No oligomerization measurement exists for the prior 0-activity designs, so
  monomer failure, tetramerization failure, catalytic failure, and assay
  sensitivity remain experimentally confounded.
- Protenix five-sample agreement is a predicted-ensemble criterion, not an
  experimental tetramerization readout.
- Rosetta energy uses one representative sample and has 300 AGP-unknown
  positions; those positions are not safe negatives.
- The `N_eff/L = 10` covariance gate is an evidence-quality threshold, not a
  biological discontinuity. Several exploratory parents are immediately below
  it and pass T6.
- Opening positions does not prove that RF will mutate them, that a WT core will
  cease binding, or that immune burden will decrease.
- Potts scoring of actual RF outputs, tetramer refolding of those outputs, and
  wet-lab activity feedback are still required before any candidate can be
  recommended.

## Canonical output files

- `parent_position_evidence.{parquet,tsv}`: complete 4,548-position evidence
  universe.
- `epitope_core_position_evidence.{parquet,tsv}`: 2,457 core-register rows.
- `core_policy_mask_tradeoff.{parquet,tsv}`: per-core policy/mask collisions,
  exact positions, unknowns, and bounds.
- `parent_allele_policy_mask_tradeoff.{parquet,tsv}`: per-parent/allele recovery
  trade-off surface.
- `parent_ec_contact_validation.{parquet,tsv}`: T6 L/2, L, and 2L monomer versus
  tetramer precision for 5-of-5/4-of-5/1-of-5 contact definitions.
- `join_metadata.json`: definitions, claim boundary, commands, and input/output
  hashes.

