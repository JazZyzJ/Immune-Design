# RF Fusion V2 Dual-Allele — Concise Cluster Runbook

**Status:** ready for cluster-agent execution after one code-generated assembly preflight.

**Scientific authority:** `doc/Dual_Allele_Steering.md`.

**Implementation and decision authority:** `PLAN_RF_FUSION_V2_DUAL_ALLELE.md` and
`doc/DUAL_ALLELE_DUALF0_AUDIT.md`, especially Appendices E, H, I, J, and K.

This runbook asks one question:

> Under the same Fusion V2 generation authority, does simultaneous Dual steering reach a lower
> frozen joint objective than either single-allele steering law?

The execution has one assembly gate, one formal three-arm campaign, and one combined analysis.
There is no separate calibration campaign, opportunity probe, two-protein M1 experiment, D0
experiment, root sweep, temperature sweep, NMP run, or new structure study.

---

## 0. Already frozen and completed

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
