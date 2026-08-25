# DUALF0 — Authority Reconciliation and Seam Audit

**Task:** `PLAN_RF_FUSION_V2_DUAL_ALLELE.md` §5 DUALF0.
**Status:** COMPLETE. DUALF0's own stop condition does **not** fire. The live scientific decisions
are frozen in Appendix I and Appendix J.7; one mandatory panel-sensitivity measurement and the
runbook-versus-code consistency pass remain before cluster launch.

---

## 1. Implementation base

| Item | Value |
|---|---|
| Repository | `/home/zc1519/src/Immune-Design` (Della), branch `fusion_rf_refine` |
| Implementation-base commit | `1bf0fe7b092cc4d990a2267eaa09ea76a2f99c40` |
| PLAN audited commit | `cdcde74d0b893354426eed035da4b7593b291a3a` — verified ancestor of HEAD |
| Intervening commits | `6ecfa66` (epi-head-wave3 merge), `28f647e`, `1bf0fe7` (docs) |

`git diff --stat cdcde74 HEAD -- inverse_folding/` and `-- scripts/*fusion_v2*` are both **empty**.
Every PLAN §1.2 seam is byte-identical to the audited commit.

The PLAN's audit set omits `epitope_head/`, which the merge changed by 905 insertions and which
*defines the frozen objective*. That gap was closed here; see §2.

PLAN line 27 cites `doc/RF_Fusion_V2_Cluster_Runbook.md`. The file on disk is
`doc/RF_Fusion_v2_Cluster_Runbook.md` (lowercase `v2`).

---

## 2. Frozen-objective reproducibility across the epi-head-wave3 merge

**Verdict: PROVABLY UNCHANGED.** Head A loads and scores identically before and after `6ecfa66`.

Head-load chain: `scripts/rf_fusion_v2_oracles.py:791` → `:680` →
`scripts/run_rf_refine_fusion.py:623` → `scripts/run_if_phase_c1.py:1167` → `scripts/infer_v1.py`.
Only the last two files' dependencies changed.

| Check | Result |
|---|---|
| Architecture read from repo YAML at runtime, not from checkpoint | TRUE — `run_if_phase_c1.py:1190` loads `model.yaml`, `model_ablation.yaml`, `inference.yaml` |
| Those three YAMLs changed by the merge | NO — byte-identical; `head_config_hash` unchanged |
| `training/model.py` new submodules | `use_core_scorer=False` (:176) and `enable_boundary_head=False` (:514) defaults; both constructed only inside their guards |
| `inference/predictor.py` new branch | `want_exact = return_exact and getattr(model,'enable_boundary_head',False)` (:237) — always False on the production path |
| `configs/cnn_himp_a1_res03.yaml` | ADDED by the merge, training-only, never read at inference |
| `configs/train.yaml` changed | Yes, but not on the inference import chain and not in `_compute_head_config_hash` |
| Numerical check at HEAD | `build_predictor(...)` loads the real Head-A checkpoint with `strict=True`; state_dict keyset bit-identical |

**Executed-campaign binding matches the runbook exactly.** `BASE_K12_RUN/root0_10PA_A/run_manifest.json`
binds `head_checkpoint` sha256 `3ab78004…` at the same path the Dual runbook names as
`HEAD_A_CHECKPOINT`; `head_config` digest `acb178be…`. The four `--bundle` mechanism dirs bind the
same two digests.

**Head A vs Head B are discriminable only by checkpoint sha256.** Both share one config dir, and
both checkpoints carry the identical metadata `config_hash` `418915354319` with no allele field.
`head_config_hash` is a *shared substrate digest*, not a per-Head discriminator. Any Dual identity
check must pin `head_checkpoint_digest` (A `3ab78004…`, B `7425a927…`) and refuse equality.

`HEAD_A_ALLELE_IDX=0` / `HEAD_B_ALLELE_IDX=0` are correct and are the only legal values:
`model.yaml` declares no `n_alleles`, so `allele_embedding = nn.Embedding(1, 16)` for both heads.

**Head B must remain a plain `cnn_himp_a1_res03` head.** The merge added `cnn_himp_a1res03_core`
and `cnn_himp_a1res03_exact` variants whose checkpoints `infer_v1.build_predictor` physically
cannot load (it never constructs `core_scorer` / `boundary_head`, and `strict=True` would fail
after the GPU is allocated).

---

## 3. PLAN claims verified against code

| PLAN claim | Verdict | Evidence |
|---|---|---|
| §1.2 symbol names (`CompleteEndpoint`, `LineageIncumbent`, `donor_gate`, `bind_head_scores`, `select_family_representatives`, `ExactArchive`, `HeadDirectedCappedPolicy`, `V2HeadConfig`) | all exist, correctly named | `state.py:1611`, `reward.py:122`/`:406`, `lookahead.py:285`, `archive.py:253`/`:74`, `policy.py:1109`, `config.py:671` |
| §1.3 endpoint keys (`endpoint_id`, `protein_id`, `sequence_md5`, length, window-grid binding) all exist on `CompleteEndpoint` | TRUE | `state.py:1618-1636`; `sequence_md5` really is MD5 (`fusion/state.py:38`); grid via `head_binding.window_grid_digest` (`identity.py:437`) |
| The endpoint window-grid digest is allele-neutral, so A/B grids are comparable | TRUE | `identity.py:411-434` hashes only sorted `(start,end,k)` + schema version |
| §0.3 "never overwrite `head_global_risk` with a joint scalar" is structurally enforced | TRUE | `state.py:1697-1699` raises unless the payload's `global_risk` equals the field |
| §2.1 rank key $(J,\text{endpoint\_id})$ is a drop-in for the existing law | TRUE | `archive.py:234`, `:281-289` already sort `(head_global_risk, endpoint_id)` |
| §2.3 the LOO counterfactual reverts to the **current lineage incumbent** token | TRUE | `evidence.py:361`; incumbent carries the bytes and re-digests them (`reward.py:157-162`) |
| §2.3 sign convention $a_i = J(y^{(-i)}) - J(y)$, eligible when $>\epsilon$ | TRUE | `evidence.py:330-332`, `policy.py:1460-1461` |
| §2.4 reopen categories and their lexicographic order | EXACT MATCH | `policy.py:1689-1698` `_reopen_sort_key` is a 6-tuple: new_hotspot / worsening / residual_burden / active_sampler_score / −n_origin_events / position |
| §2.4 **stop condition** ("if the union reducer needs a masked-state score, stop DUALF4") | **NOT TRIGGERED** | all three Head-derived categories are pure functions of complete-endpoint `HeadScore`s |
| §3.3 "the second Head adds little compared with definitive refolding" | CONFIRMED BY MEASUREMENT | 400-cell cost ledger: structure 26 833 gpu-s (66.7 %), head 851 gpu-s (2.1 %). Second Head ≈ **+2 s/cell** against ≈67 s/cell of structure |
| Runbook §3.1 "56 independent source prefixes" is reachable | TRUE | depth-0 `partial_states` per protein: 32 + 80 = **112**, zero digest overlap |
| Runbook §1 `M1_COUNTERFACTUAL_CEILING=278` is traceable | TRUE | signed into both `v2f5a_head_directed` resolved configs |
| Runbook §4.2 job sizing (25 proteins × 3 arms, 8 h) | ADEQUATE, 2.1–2.4× headroom | executed K12 basis is 4:45–4:50 per **100** single-arm cells (jobs 12275741/2/12275813/4), not the 7:55 K24 figure the runbook's neighbourhood implies |

---

## 4. Blocking items

### 4.1 Scientific decisions — user only

> **STATUS as of 2026-08-25 — this table is the ORIGINAL 2026-08-20 statement, kept for the
> record.** Live status: **S1 closed** (Appendix I, $c_u=0.10$, $\tau=0.14426950408889636$, frozen
> in a tracked spec). **S2 closed by deletion** (Appendix I: $\epsilon_J$ conflated a pointwise
> floor with a bootstrap mean; the GO test is $\operatorname{UCB}_{95\%}[\operatorname{mean}(\Delta J)]<0$).
> **S3, S4, S5 withdrawn with the in-process contrast** (Appendix H: arms are compared ACROSS runs,
> so there are no `*_with_joint_safety` arm labels, no free-vs-dose-matched question and no arm
> execution form to choose). **S6** (coverage floor) and **S7** — S7 closed in Appendix I by
> splitting the ceiling. The decisions that are actually open now are **D1–D3 in Appendix J**.

| # | Item | Why it blocks |
|---|---|---|
| S1 | $\tau$ is not frozen | PLAN's own launch blocker. `inverse_folding/reference_flow/configs/v2_dual_smoothmax_policy_v1.json` does not exist, so runbook §1's `test -r "${DUAL_POLICY_SPEC}"` aborts under `set -e` before anything else runs |
| S2 | $\epsilon_J$ is consumed by every GO gate (runbook §3.3, §4.3) and produced by nothing | No population, no estimator, no producer task, no binding site. It is also a different statistical object from $\epsilon_{\mathrm{donor}}/\epsilon_{\mathrm{write}}$: those are per-endpoint matched-difference floors, this one is compared against a bootstrap bound on protein/prefix-paired means |
| S3 | $\delta_A$ on the C1 substrate is `1000000.0` (`highrisk-global-relaxed-hotspot-ceiling-v1`, `incremental_gate_enabled: false`) | The three C1 arm names `*_with_joint_safety` describe a gate that does not exist there. PLAN §7's stated justification for the control labels is false on this substrate. The M1 substrate differs: its $\delta_A$ is genuinely measured (`5ZHV_B` = 15.7378, q90 over n=64), so one answer will not cover both experiments |
| S4 | M1 contrast form: free-cardinality vs dose-matched | $m_d = \min(\lvert\text{positive}\rvert, \text{write\_cap}, \text{band\_max})$ (`policy.py:1507`) and $m_{\text{reopen}}$ (`:1530`) are **functions of the objective**. Routing M1 through the existing `policy_qualification` machinery drops exactly the cells where Head B changed the positive set — i.e. every cell carrying the scientific signal — as UNSCORABLE, biasing M1 toward the null |
| S5 | C1 three-arm shared-D0 execution form | Largest new machinery in the PLAN. `run_depth_ladder` captures its own root unconditionally (`ladder.py:305-332`), always allocates a fresh `ExactArchive()` (`:280`), and passes `source_override`/`source_endpoints`/`archive` explicitly (`:356-368`) so no caller can inject them. The only shared-source arm machinery (`paired.py`) is depth-0 one-cycle and hard-coded to exactly two slots (`PAIR_ARM_SLOTS = {arm_a, arm_b}`) |
| S6 | C1 coverage floor 80/100 with one root per protein | Inherited from a **4-root** 84/100 surface. Per-root success there was 328/(84·4) = 0.976, so single-root expectation is ≈82.0 ± 1.4 — the validity floor sits 1.4 σ below the design's own expectation |
| S7 | Counterfactual-ceiling unit | Pre-existing split, doubled by Dual: the signed artifact declares `unit = editable_positions_per_cycle` for value 454, the config key is `max_counterfactual_head_calls_per_cycle`, and the gate is `len(candidates) > budget` (positions, `policy.py:1421`). Under Dual one position costs two Head calls |

Good news on S1/S2 sequencing: $b_a, s_a$ are **measurements**, not outcomes, so P0 may produce them
before $\tau$ is frozen without contaminating the objective — provided the rule that maps them to
$\tau$ is declared first. P0's first real measurement is a genuine pass/fail risk: PLAN §2.1
requires $s_a$ to exceed the frozen matched-difference noise bound `0.017012596130371094`, and
whether the NoD population's weighted IQR/1.349 clears it **for both alleles** cannot be known
without scoring the 800 sequences.

### 4.2 Contract conflicts the PLAN must resolve before DUALF1

| # | Item | Evidence |
|---|---|---|
| C1 | PLAN §3.1 (overlay outside the config) and §3.2 (run signature must move on Head/objective/margin/safety drift) are **mutually unsatisfiable as written**. `RunSignature` hashes an explicitly enumerated 9-key payload whose `config_digest` cannot see an external overlay, and `mismatch_status()` iterates a hardcoded field list | `identity.py`; `rf_fusion_v2_resume.py:160-172` |
| C2 | `V2Config.canonical_payload()` emits the head block as `vars(self.head)`. **Any** field added to `V2HeadConfig` — optional, `None`-defaulted, anything — changes the digest of every legacy `v2cfg-1` config, hence every RunSignature and every resume fragment | `config.py:742` (verified numerically) |
| C3 | Head B's content identity has no legal container. `CONTENT_ROLE_TO_FIELD` is a **closed 18-role vocabulary and every role is mandatory**; each role maps onto a `V2ConditioningIdentity` field that feeds `make_live_state_id` → `make_endpoint_id`. Adding roles breaks every legacy config load *and* moves every endpoint id — directly against PLAN §0.3 | `identity.py:234-253` |
| C4 | The second cumulative safety reference has no owner. `SafetyReferenceBinding` is single-allele by construction and exactly one instance is a field of `LivePartialState`/`ProjectedPartialState`, inside both canonical payloads and equality-checked on propagation. `safety.py` alone cannot own it — the binding lives in `state.py` | `state.py`, `safety.py:4xx` |
| C5 | The `projection.head_directed` block is **fully closed**: `_reject_unknown` over an 11-key whitelist (`config.py:1157-1162`), `reopen_priority_law` and `control_policy_id` each whitelisted to a single literal (`:1179-1182`), `write_cap_editable_fraction` pinned to 0.05 (`:1193-1198`). No Dual arm label, union reopen law, or joint margin has any legal home there. PLAN §4's "without changing legacy config semantics" is impossible for these | verified |
| C6 | The ledger has no allele grain and **collides on `event_id`**: `_LOGICAL_IDENTITY = ("event_id","protein_id","arm","phase")` with no allele or sequence dimension, and a Dual cycle's two Head batches both land on `f"{event_prefix}:head"`. First-wins merge plus `n_retries = len(attempt_ids) - 1` against `max_retries: 2` fails every cell, while under-reporting `max_head_calls` by exactly one allele | `ledger.py:66-70`, `cycle.py:185`, `policy.py:1710` |
| C7 | `LineageIncumbent` is structurally single-Head (one `head_identity_digest`, one required `head_global_risk` validated against `head_score.global_risk`), and `donor_gate` **refuses** any donor whose evaluator digest differs from the incumbent's. Dual needs a *pair* identity here | `reward.py:141-175`, `:451-459` |
| C8 | `bind_head_scores` refuses two results carrying the same `sequence_md5` and takes one `evaluator`; `score_leave_one_out` builds sequences internally, calls one `scorer`, and matches by a `by_md5` dict that raises on a second result. Neither can fan one sequence set to two Heads | `lookahead.py:285`, `evidence.py` |
| C9 | `build_production_oracles()` has exactly one path to a Head and it is hard-coupled to the refold model (`structure_config`, `test_set_parquet`, `pdb_root`, `refold_cache_dir` all required). Building Head B "without a second structure stack" needs a new seam | `rf_fusion_v2_oracles.py:680` |
| C10 | Resume fragments are keyed by `protein_id` **alone** (`fragment_dir/<protein_id>.json`), so three arms in one protein cell collide as `duplicate_conflict`. Separately, a mismatched fragment is silently **overwritten**, never reported | `rf_fusion_v2_resume.py`, `run_rf_fusion_v2.py` |
| C11 | `preflight_v2_canary_assembly.py` **cannot pass today** on any head-directed cell: `assemble_check` calls `resolve_support_policy(...)` with no `**context`, so the factory raises listing `['head_oracle','incumbent','safety_reference_score','evaluator','window_grid_digest']`. Pre-existing, but runbook §0 forbids the one-prefix smoke that currently substitutes for it | verified |
| C12 | `run_rf_refine_fusion.py:630` hardcodes `score_scale="raw_logit"` inside `_build_head_scorer_from_args`, silently overriding `--score-scale`. Under Dual a Head B declaring another scale would be coerced while every identity check still agrees | verified |
| C13 | `$\epsilon_{\mathrm{donor}}$` maps to **two** code names: config `donor_improvement_epsilon` (`config.py:630`), runtime/verdict/artifact `epsilon_r` (`policy.py:868`, `reward.py:411`). A RED test written against one name does not bind the quantity the gate uses | verified |

### 4.3 Runbook factual corrections (documentation-level)

1. **P0's `--bundle` paths are wrong for the donor/support role.** The four named
   `v2_mechanism*/{5zhv,q00511}_mech` bundles ran under the DIAGNOSTIC `state_derived_probe` kernel
   and carry 39-column `feedback_events` with **no** donor evidence. They are the correct
   *repeatability* population (dedup = 1644, matching the signed artifact's `n_observations`).
   The donor/support substrate that P0's opportunity gate actually needs **exists**:
   `run/inverse_folding/v2_canary/v2f5a_head_directed/{5zhv,q00511}_v2f5a_r40` — 64-column
   `feedback_events`, 112 rows, `donor_gate_passed` and `write_candidate_evidence_json` present,
   56 source prefixes each, plus 221/155 `support_law` contrast rows. P0 needs a **second**
   argument for this role, not a redefinition of the gate.
2. Three `feedback_events` schema generations coexist on disk: **39** (diagnostic), **64**
   (`v2f5a_head_directed`, `v2f5a_smoke`), **66** (`BASE_K12_RUN`, `v2_depth_capacity`).
3. `M1_SAFETY_REFERENCE_MANIFEST=work/immune-design/v2_canary/references.json` covers exactly
   `{5ZHV_B, Q00511}` (255 bytes). C1 needs the manifest the executed cells actually bind:
   `work/immune-design/v2_highrisk/highrisk_ceiling_global_d8k32_v1_0701/references.json`
   (23 703 bytes, 100 entries). A near-twin `highrisk_ceiling_v1_0701/references.json`
   (13 703 bytes) also exists and is **not** the bound one.
4. `NOD_NORMALIZATION_ENDPOINTS` is a v0/v1 parents table — 800 rows, 100 proteins, exactly 8
   unique sequences each, columns `[protein_id, design_idx, sequence, seed, wall_seconds]`. It has
   **no** stored Head score, so both Heads must score all 800 fresh. PLAN §2.1's protein-equal
   weighted median/IQR is computable and here degenerates to the unweighted one.
5. The C1 counterfactual ceiling the runbook says to "derive" is already signed: **454**
   (`artifacts/head_policy_v2_eps005_c454.json`, basis `6T88_A`), carried by all 100 executed
   resolved configs. Cite it rather than re-deriving.
6. `DUAL_QUALIFICATION` / `DUAL_COMPARISON` are read **nowhere**. The launcher reads `QUALIFICATION`
   (not `DUAL_QUALIFICATION`) and has no comparison mode. Because every knob is `${VAR:-default}`,
   the M1 command as written runs the **old** endpoint_change/source_shuffle mechanism cohort and
   exits 0 — a wrong experiment reported as success.
7. `--dual-qualification` would duplicate the existing `--qualification`, which means something
   different and non-substitutable (the V2F5A Head-directed vs Head-blind source-geometry control).
8. Launcher header pins `#SBATCH --qos=gpu-short --partition=rtx6000` and
   `CONDA_ENV=immune-design-blackwell`. The V2F5A substrate M1 must be compared against was produced
   on rtx6000 + blackwell (jobs 12126550/12126551), while the runbook sends M1 to ailab/h200.
   Reproducing the substrate stack and reaching the requested partition are mutually exclusive as
   written.
9. P0 is given no `--time`, `--mem`, `--account`, `--partition`, `--constraint`, or `CONDA_ENV`.
   Its Head workload is small — at the measured 55.3 calls/s, ≈526 s for the 14 544-endpoint
   cross-score, ≈29 s for NoD, ≈59 s for repeatability — but it is *not* Head-only if it must also
   produce the safety contracts (see S3).
10. `--max-counterfactual-sequences-per-cycle` does not exist; the real flag is
    `--max-counterfactual-head-calls-per-cycle`, and `--policy-spec` (required) is omitted from the
    runbook's command while `build_calibration_bundle` rejects any spec whose `policy_id` is not
    `head_directed_capped`.
11. `read_v2_mechanism.py` takes `--bundle` (required, `nargs='+'`) and `--out` (a JSON **file**).
    None of `--run-dir`, `--run-root`, `--requested-cohort`, `--out-dir`,
    `--dual-calibration-json` exist, and there is no bootstrap machinery — the executed V2 gate uses
    a Student-t one-sided UCB with the margin read from the resolved configs, not from a CLI JSON.
12. `--dual-capability` is not an extension of that reader: its only ingest is
    `mechanism_contrasts.parquet`, which recursive D4/K12 runs write with **zero** rows.
13. `V2_PROTEIN_ID` does not exist; the `.args.sh` variable is `V2_COHORT` and it already holds the
    protein id.
14. Arm-label spelling is not frozen: PLAN §7 uses `A_only_with_joint_safety`, runbook §2.2 uses
    `a_only_with_joint_safety`. These strings are load-bearing.
15. `caps`: `max_logical_dfe` (2200) and `max_definitive_refolds` (64) come from the hardcoded
    `EXPLORATORY_PROFILES['highrisk_d4_k12_r40']` with **no** CLI override; `max_head_calls` **is**
    CLI-overridable (`materialize_v2_canary_config.py:505`, not clobbered by the `:509` merge).
16. All four executed RAR 0049 jobs report `State FAILED ExitCode 3:0` despite writing all 400
    bundles. The Dual runbook's "verify `sacct`" step needs a rule that distinguishes this from a
    genuine invalid run.

---

## 5. `head_global_risk` decision-site map (DUALF0 deliverable)

Legacy-only (no change): `inverse_folding/reference_flow/fusion/{runner,selection,state}.py`,
`scripts/{run_rf_refine_fusion,assemble_if_test_set,prescreen_*}.py`,
`inverse_folding/evaluation/{schema,assembly}.py`.

Must become objective-aware:

| Site | Role |
|---|---|
| `fusion_v2/state.py:1632,1649,1698,1699,1749` | endpoint scalar + payload; **keep single-Head**, Dual value goes to the sidecar |
| `fusion_v2/reward.py:143,169,171,173,215,288,343,348,454,455,465` | incumbent binding, donor gate comparison, verdict payload — the primary injection point |
| `fusion_v2_runtime/archive.py:234,281-289` | elite comparison + family-representative sort key |
| `fusion_v2_runtime/lookahead.py`, `cycle.py` | one binding site each |
| `fusion_v2/policy.py`, `evidence.py` | one site each, inside LOO contribution |
| `scripts/materialize_v2_archive_facade.py` (13 sites) | top-k facade — needs an explicit common-objective mode |
| `scripts/rf_fusion_v2_artifacts.py` (4), `rf_fusion_v2_cohort.py` (1), `calibrate_v2_head_policy.py` (7), `scripts/analysis/replay_v2_head_directed_policy.py` (4) | artifact/analysis surfaces |

`config.py` and `identity.py` contain **zero** reads — confirming that objective-awareness is a
runtime concern, not a config-schema one.

## 6. Operating point

D4/K12/r40, four independent roots, `active_population_width=1`, `depth_cap=4`,
`schedule_id=highrisk-d4-k12-r40-global-v2`, `master_seed=20260811`, code revision `21cd73d2`:
**confirmed unchanged and frozen** (400 cells = 100 proteins × 4 roots, 14 544 complete endpoints).

---

## Appendix A. Full finding index

135 findings from 8 read-only auditors, cross-checked by 2 adversarial verifiers
(134 + 88 verdicts). Full evidence with file:line for every row is in the workflow journal:
`~/.claude/projects/-home-zc1519-src-Immune-Design/<session>/subagents/workflows/wf_4f886b1d-643/journal.jsonl`.

Verify column: `C` confirmed, `P` partial (claim holds, detail corrected), `R` refuted, `-` not re-checked.

### BLOCKER (29)

| id | verify | user | summary |
|---|---|---|---|
| `head-config-vars-digest-trap` | C |  | V2Config.canonical_payload() serializes the head block as `vars(self.head)` — a raw __dict__ dump. ANY field added to V2HeadConfig, even optional and … |
| `run-signature-blind-to-dual` | C | yes | RunSignature has exactly 9 fields and hashes an explicitly enumerated 9-key payload; `config_digest` is `V2Config.config_digest()`, which cannot see a… |
| `head-b-content-role-has-no-legal-slot` | C |  | Content provenance runs through a CLOSED, REQUIRED-COMPLETE role vocabulary. `_content_identities` restricts every role to CONTENT_ROLE_TO_FIELD and t… |
| `second-safety-reference-has-no-owner-in-state` | C | yes | SafetyReferenceBinding is single-allele by construction (one head_binding -> one evaluator -> one allele; one head_score_digest), and exactly ONE of t… |
| `dual-loo-budget-unit` | C | yes | No such flag exists anywhere in the repo (`grep -rn 'max-counterfactual-sequences-per-cycle' .` returns ONLY the runbook line itself). The real flag i… |
| `dual-arm-cardinality-parity` | CP | yes | m_d and m_reopen are FUNCTIONS OF THE OBJECTIVE, not of the source geometry: `m_d = min(len(positive), write_cap, band_max)` where `positive` is the s… |
| `control-policy-id-and-reopen-law-whitelists` | C | yes | The head_directed config block hard-whitelists BOTH the control policy id and the reopen law to single literal values: `control_policy_id=_text(node, … |
| `shared-d0-three-arm-ladder-unreachable` | CP | yes | `run_depth_ladder` always captures its OWN root and has no parameter to receive a pre-captured one; it also passes `source_override`, `source_endpoint… |
| `matched-pair-law-is-strictly-two-armed` | C | yes | The matched-arm record/verification law is hard-coded to exactly two arms. `PAIR_ARM_SLOTS = frozenset({'arm_a','arm_b'})`; `assert_matched_pair_seeds… |
| `ledger-has-no-allele-grain-and-collides-on-event-id` | CP | yes | The ledger event schema has NO allele dimension and NO per-sequence grain. Logical identity is `('event_id','protein_id','arm','phase')`; `head_calls`… |
| `delta-ab-vacuous-on-c1-cohort` | C | yes | On the C1 substrate the cumulative hotspot gate is deliberately non-binding: every executed high-risk cell carries delta_new_cumulative.value = 100000… |
| `epsilon-J-consumed-never-produced` | C | yes | epsilon_J occurs in no other document. PLAN §3.1 requires the calibration artifact to bind only "$b_a,s_a$, repeatability, donor/write margins, and di… |
| `c1-coverage-floor-inherited-from-4-root-surface` | - | yes | The 84/100 surface was produced with FOUR independent roots per protein: 328 successful cells out of 400, 16 proteins with no successful root. Among t… |
| `c1-three-arm-cell-breaches-inherited-caps` | CP | yes | The C1 source cell's caps come from the hard-coded EXPLORATORY_PROFILES['highrisk_d4_k12_r40']['caps'] = {max_logical_dfe: 2200, max_definitive_refold… |
| `content-role-vocabulary-is-closed-18-roles` | C | yes | V2 content identity flows exclusively through CONTENT_ROLE_TO_FIELD, a CLOSED 18-role vocabulary. config.content rows are validated with allowed=froze… |
| `assembly-preflight-cannot-build-head-directed-policy` | C | yes | assemble_check calls resolve_support_policy(config, band_table=..., stratum_key=...) with NO **context, so for support_policy_id='head_directed_capped… |
| `p0-opportunity-gate-unsatisfiable-on-named-surfaces` | PR | yes | Donor/support replay needs the V2F5A policy-evidence columns. Every 5ZHV_B/Q00511 V2 bundle on disk predates that schema: 12/12 have 39-column feedbac… |
| `c1-safety-gate-is-disabled-on-the-frozen-substrate` | C | yes | The frozen C1 substrate (BASE_K12_RUN, whose resolved configs the Dual configs are to be derived from) sets delta_new_cumulative.value = 1000000.0 wit… |
| `calib-cannot-produce-per-allele-safety-contracts` | C | yes | The cumulative hotspot limit is a per-PROTEIN CalibratedScalar produced by a different script, scripts/calibrate_rf_fusion_v2_hotspot.py, which GENERA… |
| `epsilon-J-has-no-estimator` | C | yes | Nothing computes any joint margin. The only margin law implemented is single-Head: difference_bound = 2.0 * max(|live - stored| global risk), and it i… |
| `head-b-has-no-stored-vs-live-population` | C | yes | The existing noise bound is measured by comparing a run's STORED head_global_risk against a fresh live score of the same bytes, and score_repeatabilit… |
| `read-mechanism-cannot-ingest-a-recursive-run` | C |  | The reader's ONLY ingest is load_contrasts(), which requires mechanism_contrasts.parquet per bundle and exits if absent; every downstream statistic gr… |
| `recursive-path-has-no-shared-d0-three-arm-branching` | C | yes | One run == one arm on the recursive path. run_v2_shard takes no arm argument and run_depth_ladder's only arm identity is the boolean feedback_enabled … |
| `resume-one-fragment-per-protein-collides-across-arms` | C | yes | Fragments are keyed by protein_id ALONE: the file is fragment_dir/<protein_id>.json, expected_by_protein is a protein->signature map, and scan_fragmen… |
| `slurm-dual-env-vars-do-not-exist-and-fail-silently` | C |  | Neither variable is read anywhere in the script. The launcher reads QUALIFICATION (not DUAL_QUALIFICATION) and has no comparison mode at all. Because … |
| `m1-bundles-carry-no-donor-evidence` | CP |  | Those four bundles were run under the DIAGNOSTIC kernel probe, not the head-directed production policy, and store no donor/write evidence at all. Thei… |
| `delta-b-not-derivable-without-regeneration` | C | yes | The per-allele cumulative hotspot threshold is produced by a DIFFERENT, generative script that the runbook never mentions: scripts/calibrate_rf_fusion… |
| `c1-joint-safety-is-a-noop-on-the-frozen-substrate` | C | yes | On the C1 substrate the Head-A catastrophic gate is deliberately disabled. Every resolved config of the executed 100-protein campaign carries safety.d… |
| `c1-coverage-floor-inherited-from-a-four-root-union` | C | yes | 84/100 is the union over FOUR independent roots. Per-root success in RAR 0049 was 81, 82, 82, 83 out of 100 — only 1-3 proteins above the runbook's ow… |

### MAJOR (60)

| id | verify | user | summary |
|---|---|---|---|
| `dual-margins-are-allele-A-scale-locked` | C | yes | The two existing margins are `donor_improvement_epsilon` and `local_contribution_tolerance`, and the loader HARD-REQUIRES each to carry `unit == confi… |
| `incumbent-is-structurally-single-head` | C |  | LineageIncumbent has ONE `head_identity_digest` (a single HeadEvaluatorIdentity.digest()) and a REQUIRED float `head_global_risk` that is validated to… |
| `incumbent-id-cannot-name-an-objective-arm` | C |  | incumbent_id is minted from a payload containing only {kind, lineage_id, endpoint_id | reference_binding_id, sequence_md5, head_identity_digest, accep… |
| `head-config-hash-cannot-distinguish-a-from-b` | C |  | Head A and Head B share ONE config directory, and the only config digest the code computes hashes just 3 files from it, so the config digest is a CONS… |
| `two-divergent-predictor-construction-paths` | C | yes | The merge added two a1res03 variants whose checkpoints the Fusion V2 stack physically cannot load, and it created a second, divergent construction pat… |
| `counterfactual-ceiling-278-sequences-vs-head-calls` | C | yes | The existing flag is `--max-counterfactual-head-calls-per-cycle`, the config field is `max_counterfactual_head_calls_per_cycle`, and the gate is a raw… |
| `dual-second-head-scores-for-incumbent-and-reference` | C |  | Two of the three Head-derived reopen categories and the ENTIRE write screen compare the donor against objects that are NOT endpoints: `incumbent_windo… |
| `dual-safety-second-binder-and-atomic-advance` | C |  | There is no seam to compose on. `bind_admission_policy(config, head_identity)` is CONFIG-SHAPED and raises HeadIdentityMismatch whenever the runtime H… |
| `scale-normalized-window-magnitude-undefined` | CP | yes | The three Head-derived reopen conjuncts are per-WINDOW quantities on the raw z scale, and they are not all differences: `worsening_at` and the new-hot… |
| `dualf4-acceptance-reopen-count-claim` | C |  | B(r) and u_target genuinely are objective-independent (`band_center_target` reads only step/stratum/n_editable, policy.py:1469-1472), but the reopen C… |
| `reopen-union-is-ordering-not-membership` | CP |  | Head evidence never gates reopen CANDIDATE MEMBERSHIP. `_reopen_priority` iterates over EVERY source-resolved editable position and skips only activel… |
| `structure-verdict-is-per-endpoint-not-per-exact-sequence` | C | yes | The structure verdict is computed once per ENDPOINT ROW, not per unique sequence: the Head batch is deduped by `sequence_md5` but the structure loop i… |
| `archive-rank-key-is-hardcoded-not-injectable` | C |  | There is no injection point today. `ExactArchive.__init__` takes no arguments, the elite comparison reads `row.endpoint.head_global_risk` by direct at… |
| `no-single-objective-authority-seam` | C |  | There is no single narrowest place. The authority must reach four independently constructed owners: (1) `run_one_cycle`, whose signature is fully expl… |
| `bind_head_scores-cannot-carry-a-second-head` | C |  | `bind_head_scores` is structurally single-Head in three independent ways: it takes ONE `results` sequence and refuses two results carrying the same `s… |
| `score_leave_one_out-cannot-fan-one-set-to-two-heads` | C |  | `score_leave_one_out` builds the sequences internally and immediately calls a single `scorer(protein_id, sequences)` in the same function body, then m… |
| `a2-head_calls-collapses-the-two-alleles` | C |  | `head_calls` is a single scalar in a CLOSED resource vocabulary that is asserted equal between the runtime and the config at import time, and it is a … |
| `safety-gate-and-admission-are-single-verdict-typed` | C | yes | Three hard single-verdict types stand in the way. `admit_endpoint` type-checks `isinstance(gate, SafetyGate)` and `SafetyGate` holds exactly one polic… |
| `counterfactual-ceiling-flag-renamed-and-unit-changed` | C |  | The existing required flag is `--max-counterfactual-head-calls-per-cycle` (int, required) and the frozen config field is projection.head_directed.max_… |
| `c1-ceiling-has-no-derivation-procedure` | CP |  | The executed runbook already contains the exact reusable recipe for this, which the Dual runbook neither cites nor reuses: POLICY_C=$(jq '[.[]] | max'… |
| `p0-calibration-cli-does-not-exist` | C |  | The script today accepts: --bundle(append), --policy-spec, --out-rows, --out-json, --min-observations-per-protein, --max-sequences-per-protein, --max-… |
| `reader-cli-conflicts-with-established-convention` | C |  | The reader takes --bundle (nargs=+), --config, --delta, --floor, --cap, --confirmatory, --qualification, --min-prefixes, --out. NOT FOUND: --run-dir, … |
| `bootstrap-replaces-executed-t-ucb-unassigned` | C | yes | The executed V2 gate the M1 design otherwise copies uses a Student-t one-sided UCB and takes its margin from the config, not from a CLI JSON: ucb = me… |
| `p0-opportunity-probe-has-no-owner` | C |  | The PLAN contains zero occurrences of 'opportunity'; §3.4 lists only dual_endpoint_evidence / dual_feedback_evidence / dual_terminal_summary. The exis… |
| `safety-reference-manifest-covers-2-of-100-proteins` | CP |  | That manifest contains exactly two entries, 5ZHV_B and Q00511. The 100-protein cohort used by C1 has its own 100-entry manifest at work/immune-design/… |
| `arm-label-spelling-not-frozen` | C |  | These strings are load-bearing: they are the literal value of the matched arm bundle passed to the materializer, they participate in the run signature… |
| `m1-runner-selection-undefined` | C | yes | The launcher only emits --qualification when QUALIFICATION=1, and the driver selects the policy-qualification runner from that flag. The paired-view m… |
| `p0-payload-not-head-only-and-unsized` | C |  | No --time, --mem, --account, --partition, --constraint, or CONDA_ENV is given anywhere for P0. The Head part is genuinely small (~2.4k sequences x 2 H… |
| `launcher-header-vs-ailab-h200-submission` | C |  | The launcher header pins `#SBATCH --qos=gpu-short` and `#SBATCH --partition=rtx6000`, and its CONDA_ENV default is `immune-design-blackwell` — an env … |
| `eps-J-binding-site-drifts-from-config` | C |  | The established V2 law is the opposite: the reader derives the margin from the resolved configs of the bundles it is reading and refuses to proceed un… |
| `v2-protein-id-does-not-exist` | C |  | The variable is named V2_COHORT and it ALREADY holds exactly the protein id. V2_PROTEIN_ID appears nowhere in the repo. |
| `dual-env-vars-silently-ignored-by-launcher` | C |  | submit_rf_fusion_v2_canary.slurm reads only CELL, CELL_LIST, PROJECT_ROOT, SCRATCH_BASE, WORK, RUNDIR, CONDA_ENV, DEVICE, MECHANISM_PREFIXES, MECHANIS… |
| `dual-qualification-has-no-execution-replicate-law` | C | yes | execution_replicates() returns 2*n_prefixes ONLY when args.qualification is true; otherwise 1. The runbook's mode case never passes --qualification, a… |
| `head-b-cannot-be-built-without-a-second-structure-stack` | C |  | There is exactly ONE path to a Head in the V2 oracle stack, and it is hard-coupled to the refold model. build_production_oracles() requires structure_… |
| `head-b-identity-is-operator-asserted-not-content-derived` | C | yes | (1) The allele string is never read from the checkpoint — OnlineHeadScorer takes allele=setup.allele from the CLI. (2) ProductionHeadOracle's declared… |
| `conda-env-default-is-wrong-for-ailab-h200` | C |  | CONDA_ENV defaults to immune-design-blackwell, an env the launcher's own header says exists for rtx6000 (sm_120) and warns must not be swapped because… |
| `preflight-print-and-verify-list-has-no-owner` | C | yes | The two gates print disjoint things and NEITHER prints most of these. preflight_v2_canary_assembly.py's report dict contains only cell/protein_id/conf… |
| `c1-verdict-under-an-exploratory-identity` | C | yes | --exploratory-depth-override is the ONLY way to open a D>1 ladder in this driver; it requires identity.phase='capability_ladder' AND identity.split_ro… |
| `shard-inputs-parser-does-not-unquote` | C |  | The materializer writes each SHARD_INPUTS entry with shlex.quote, but preflight_v2_canary_assembly._shard_inputs_from re-parses the file TEXTUALLY (li… |
| `dual-qualification-duplicates-qualification` | C | yes | --qualification already exists and means something DIFFERENT and non-substitutable: the V2F5A support-policy contrast (Head-directed treatment vs its … |
| `materializer-head-b-flag-set-is-incomplete` | C |  | Head A needs SIX runtime values plus FOUR scientific ones. head_allele_idx and head_window_batch_size are read from ShardInputs with silent defaults 0… |
| `calib-policy-spec-required-and-dual-spec-rejected` | C |  | --policy-spec is required=True, and build_calibration_bundle rejects any spec whose policy_id != 'head_directed_capped' or policy_version != 'v1'. |
| `counterfactual-cap-unit-change` | C | yes | The existing flag is --max-counterfactual-head-calls-per-cycle and its value is written verbatim into head_directed.max_counterfactual_head_calls_per_… |
| `normalization-parquet-is-not-a-v2-bundle` | C |  | The named file is a v0/v1 fusion parents table with 5 columns only -- protein_id, design_idx, sequence, seed, wall_seconds. It has no sequence_md5, no… |
| `safety-reference-manifest-covers-2-of-100-proteins` | CP |  | That references.json is 255 bytes and names exactly {5ZHV_B, Q00511}, while BASE_K12_RUN spans 100 different proteins. The 100-protein campaign has it… |
| `read-mechanism-cli-shape-is-incompatible` | C |  | None of those exist. --bundle is required=True with nargs='+' (not action='append', so a repeated flag OVERWRITES rather than accumulates), and --out … |
| `read-mechanism-margin-provenance-changes` | C |  | Today the margin is read from the RESOLVED CONFIGS (--config), and the reader additionally cross-checks that every bundle's manifest config_digest is … |
| `support-law-view-never-produced` | R |  | The qualification reader filters on view == 'support_law' and exits if there are none. No bundle on disk has ever contained a support_law row, so this… |
| `m1-control-arm-is-not-the-source-geometry-control` | C |  | The only implemented qualification control is build_source_geometry_control(treatment_policy=...), i.e. the SAME donor with write/reopen positions res… |
| `dual-tables-are-unconditional-once-registered` | C | yes | V2_TABLE_SCHEMAS is a module-level dict with no mode concept, and write_v2_bundle errors on a missing table and writes EVERY declared table even when … |
| `dual-endpoint-evidence-grain-has-no-producer` | C |  | No V2 table carries an arm at endpoint grain. Arm identity exists in run_manifest.arm_role (a single scalar per run) and in feedback_events.(arm_slot,… |
| `resume-mismatch-does-not-fail-before-oracle-work` | C |  | The driver validates the fragment inside the per-protein loop and only ACTS on verdict.records_success (skip). Any other verdict -- foreign_run, stale… |
| `resume-signature-extension-traps` | CP |  | Two concrete traps. (1) validate_fragment reconstructs RunSignature(**signature) from the stored dict; a legacy fragment lacking a new REQUIRED field … |
| `opportunity-surface-conflated-with-repeatability-bundles` | CP |  | The two inputs have disjoint protein sets and different feedback_events schema generations (39 vs 66 columns), yet the P0 stop conditions phrase the o… |
| `c1-safety-reference-manifest-never-named` | C |  | v2_canary/references.json covers ONLY the two mechanism proteins. The 100-protein C1 cohort has its own, differently-formatted manifest that the runbo… |
| `conda-env-default-mismatch-on-ailab-h200` | C | yes | The launcher defaults to the Blackwell env, which is NOT the env the frozen substrate was produced under. scripts/submit_rf_fusion_v2_canary.slurm:59 … |
| `read-v2-mechanism-flags-do-not-exist` | C |  | NOT FOUND. The reader's real input surface is --bundle (required, nargs='+') and its real output is --out, a JSON FILE, not a directory. No --run-dir,… |
| `calibration-flag-renames-and-dropped-required-arg` | C |  | The existing script names these differently and requires one argument the runbook drops entirely. Real flags: --head-config (not --head-a-config-dir),… |
| `c1-counterfactual-ceiling-value-not-named` | C |  | The measured value already exists, is signed, and is 454. Leaving it unnamed invites the coder to re-derive it and get a different number. |
| `assembly-preflight-cannot-pass-today` | C |  | The existing V2 runbook documents in its own words that this preflight CANNOT pass on a head-directed cell, and prescribes exactly the one-prefix real… |

### MINOR (46)

| id | verify | user | summary |
|---|---|---|---|
| `donor-gate-inequality-form-differs-from-plan` | C |  | The code computes `donor_risk < incumbent_risk - epsilon` (epsilon subtracted from the incumbent first), while the RECORDED margin uses the PLAN's for… |
| `null-key-convention-contradicts-plan-3-1` | C |  | The instruction is correct about the risk, but there is NO precedent in canonical_payload() for conditionally omitting an absent optional key — every … |
| `plan-1-3-field-naming-drift` | P |  | Names match exactly for the first three. "length" is `sequence_length`. "window-grid binding" is not a top-level endpoint field — it is reachable only… |
| `arm-role-vocabulary-cannot-hold-objective-arms` | C |  | ARM_ROLES is a closed three-member vocabulary {v2, a2, v0_boundary} and `feedback_enabled` is forced to `arm_role == "v2"`. RunSignature.arm_role is t… |
| `endpoint-cost-events-not-arm-neutral` | P |  | `cost_event_ids` is a CompleteEndpoint field AND is inside canonical_payload, so it participates in `content_digest` (though not in `endpoint_id`, whi… |
| `max-head-calls-is-one-scalar` | C |  | V2CapsConfig carries a single `max_head_calls: int` with no per-allele split, and `max_counterfactual_head_calls_per_cycle` is likewise one scalar in … |
| `dual-calibration-flag-names-do-not-exist` | C |  | The script's real head flags are `--head-checkpoint` / `--head-config` / `--head-variant-id` / `--head-allele-idx` / `--head-window-batch-size` / `--a… |
| `plan-cites-a-runbook-filename-that-does-not-exist` | C |  | The file on disk is `doc/RF_Fusion_v2_Cluster_Runbook.md` (lowercase v2). Linux is case-sensitive; `grep doc/RF_Fusion_V2_Cluster_Runbook.md` -> 'No s… |
| `dualf0-audit-set-omits-epitope-head` | C |  | Between cdcde74 and HEAD, `inverse_folding/` is byte-identical (empty `git diff --stat`), so no PLAN 1.2 seam moved — but epitope_head/ changed by 905… |
| `head-b-checkpoint-selection-provenance-unstated` | C |  | The merge wave also produced newer full-data DRB1*04:01 production heads, and PROGRESS explicitly warns their best.pt is leaky; separately, best.pt is… |
| `reopen-priority-label-vocabulary-two-dead-branches` | C |  | `_reopen_reason` (policy.py:1673-1681) can only return four of the six labels — AST-verified return literals are exactly {'index','new_hotspot','worse… |
| `temporal-instability-conjunct-is-n-origin-events` | C |  | The fifth sort key is `-int(row.n_origin_events)` (policy.py:1696) — a PROVENANCE CHURN count ('makes an elided middle visible rather than implying fi… |
| `epsilon-symbol-names-not-found` | C |  | NOT FOUND: `grep -rn 'epsilon_write|epsilon_donor' inverse_folding/reference_flow/fusion_v2/` returns nothing. The real names are `epsilon_r` (policy.… |
| `loo-scorer-single-evaluator-binding` | C |  | `score_leave_one_out` hard-refuses any result whose `binding.evaluator != evaluator` (evidence.py:440-444) and whose `binding.window_grid_digest != wi… |
| `ledger_stage-refuses-the-policy-head-events` | C |  | `ledger_stage` REFUSES (raises) any `head`/`structure` row whose event id contains neither `:source:` nor `:descendant:`. The policy's leave-one-out H… |
| `arm_role-vocabulary-cannot-name-the-three-dual-arms` | C | yes | `ARM_ROLES` is a closed set `{v2, a2, v0_boundary}` validated by `_text(..., allowed=ARM_ROLES)`, and `arm_role == 'v2'` is forced to imply `feedback_… |
| `reopen-reason-taxonomy-cannot-express-two-of-its-own-categories` | C |  | The SORT honours all six conjuncts, but the recorded `priority_reason` can only ever be one of four: `_reopen_reason` returns new_hotspot / worsened_w… |
| `plan-audited-commit-hash-vs-head` | C |  | HEAD is 1bf0fe7 on branch fusion_rf_refine; cdcde74 is an ancestor and every file under inverse_folding/ and scripts/*fusion_v2* is byte-identical to … |
| `c1-walltime-margin-unstated` | - |  | The executed 100-cell jobs finished in 7:22:34-7:55:15 against the same eight-hour limit, i.e. under 8% margin at 100 cells. Scaling to 75 arm-equival… |
| `m1-walltime-not-sourced` | - |  | The matched executed job (56 prefixes, two arms, same coordinates) was sized at 3h from measured walltimes of 28:41 and 60:32 at four cycles per prefi… |
| `dual-mode-has-two-sources-of-truth` | C |  | The launcher's own design note says the whole point of the .args.sh is that "a hand-maintained command in a second file is how the two drift", and it … |
| `runbook-path-case-and-implementation-target` | C |  | The file on this case-sensitive filesystem is doc/RF_Fusion_v2_Cluster_Runbook.md (lowercase v2); the science doc cites it correctly at Dual_Allele_St… |
| `C1-label-collision` | C |  | Runbook C1 actually tests science-doc C4, and runbook P0 actually tests science-doc C1. The same token means opposite things in the two authority docu… |
| `c1-difference-orientation-unstated` | - |  | The three C1 quantities mix two orientations (after-minus-before and treatment-minus-control); both happen to be negative-is-good, but the reader must… |
| `eps-donor-eps-write-config-field-names` | C |  | The live fields are projection.head_directed.donor_improvement_epsilon and local_contribution_tolerance, both PolicyCalibratedScalar with a source_ref… |
| `nod-population-carries-no-head-scores` | C |  | The named NoD parquet has columns [protein_id, design_idx, sequence, seed, wall_seconds] only — 800 rows over 100 proteins, 8 designs each. No sequenc… |
| `plan-latex-and-bound-completeness` | C |  | Not code — but the PLAN is the contract the coder reads, and DUALF1 acceptance says "smooth max satisfies its bounded-max and monotonicity properties"… |
| `tau-dependents` | C | yes | The file does not exist yet (known, deliberate). Consequences worth recording: runbook §1's guard block aborts at the first `test -r` today, so no par… |
| `counterfactual-ceiling-is-head-calls-not-sequences` | C |  | The existing flag is --max-counterfactual-head-calls-per-cycle and the config field is max_counterfactual_head_calls_per_cycle; project_v2_budget mult… |
| `section-2-2-snippet-has-unbound-cell` | C |  | n/a — shell. §1 line 63 sets `set -euo pipefail`, so ${CELL} (and ${DUAL_MODE}, ${V2_PROTEIN_ID}) are unbound-variable aborts. |
| `analysis-reader-commands-are-not-buildable` | C |  | The reader's full surface is --bundle (nargs='+', REQUIRED), --delta, --floor, --cap, --confirmatory, --qualification, --config, --min-prefixes, --out… |
| `calibrator-flag-names-diverge` | C |  | Existing surface is --bundle (append, required), --policy-spec (required), --out-rows, --out-json, --min-observations-per-protein, --max-sequences-per… |
| `plan-runbook-cross-reference-case` | C |  | On this case-sensitive filesystem the file is doc/RF_Fusion_v2_Cluster_Runbook.md (lowercase v2). The capitalized path does not exist. |
| `facade-common-objective-mode-needs-a-third-join` | C |  | The facade's data model is a strict two-table join (archive OUTER-join complete_endpoints on endpoint_id, validate='one_to_one', suffixes=('_archive',… |
| `calib-flag-renames-break-the-shared-head-loader` | C |  | The loader reads BARE attribute names off the argparse namespace: args.head_checkpoint, args.head_config, args.head_variant_id, args.device, args.head… |
| `head-b-identity-is-under-discriminated` | C |  | head_config is a DIRECTORY rolling-hash, and the runbook points both --head-a-config-dir and --head-b-config-dir at the same ${PROJECT_ROOT}/epitope_h… |
| `score-scale-and-window-grid-are-shared-not-per-allele` | C |  | The runbook passes a single --score-scale and a single --window-k-min/--window-k-max for both heads, matching the existing single-Head parser which ha… |
| `args-sh-lacks-dual-mode-and-protein-id` | C |  | materialize_v2_canary_config.py writes exactly V2_CONFIG, V2_COHORT, INPUT_FILES=(...), SHARD_INPUTS=(...). No DUAL_MODE, no V2_PROTEIN_ID. V2_COHORT … |
| `exploratory-guard-and-cpu-count` | C |  | EXPLORATORY_DEPTH_OVERRIDE=1 currently refuses to combine with MECHANISM_PREFIXES!=0 or QUALIFICATION!=0 and exits 4. Separately, the script hardcodes… |
| `conda-env-and-instrument-drift-across-gpu-arch` | C | yes | The launcher defaults CONDA_ENV=immune-design-blackwell and its own header says that env exists because rtx6000 is sm_120 and 'a different env would f… |
| `head-b-allele-not-verifiable-from-artifact` | C |  | Nothing inside the Head B artifact attests its allele. The checkpoint metadata carries no allele, and the run dir's own resolved config claims the WRO… |
| `head-b-fold0-vs-fulldata-choice-undocumented` | C |  | The pick is defensible and symmetric, but the runbook does not say WHY, and a newer, stronger, explicitly-designated production head for DRB1*04:01 no… |
| `m1-walltime-overprovisioned` | - |  | The matching single-arm 56-prefix V2F5A runs took 18 and 28 minutes. Two arms roughly doubles that. |
| `c1-mem-headroom-thin` | - |  | The executed single-arm equivalent already peaked at 43-45.6 GB against the same 64 GB. |
| `plan-authority-path-case-typo` | C |  | NOT FOUND at that path. The file is doc/RF_Fusion_v2_Cluster_Runbook.md (lowercase v2). Only the Dual runbook uses the uppercase form: doc/RF_Fusion_V… |
| `args-sh-does-not-yet-export-dual-mode-or-protein-id` | C |  | Today's generated .args.sh exports V2_CONFIG, V2_COHORT, INPUT_FILES, SHARD_INPUTS only. Neither DUAL_MODE nor V2_PROTEIN_ID exists, so the case state… |


---

## Appendix B. Decisions taken (2026-08-21)

| id | Decision | Consequence |
|---|---|---|
| S4 | **M1 is a free-cardinality contrast.** Do not route through `run_policy_qualification_view`; drop write/reopen parity from the GO preconditions; report realized $(m_{\text{write}}, m_{\text{reopen}})$ per arm as an **outcome** | Cells where Head B changes the positive-contribution set stay in the denominator. The read must stratify descendant differences by realized dose, since arm dose is no longer held equal |
| S3 | **C1 inherits the permissive ceiling; the arms are renamed `a_only` / `b_only`.** The conjunctive-safety claim is scoped to M1, whose $\delta_A$ is genuinely measured | Delete the "the non-optimized Head still enforces the common catastrophic safety contract" justification from PLAN §7 and the runbook. $I_{\mathrm{adm}}^{AB}$ on the C1 substrate reduces to the structure verdict; say so explicitly in the result |
| C1–C3 | **Identity is handled minimally.** `V2HeadConfig`, `CONTENT_ROLE_TO_FIELD` and `V2ConditioningIdentity` are frozen byte-for-byte. All Dual identity lives in one externally signed overlay whose digest becomes a single optional `RunSignature` component, omitted from the canonical payload when empty | Legacy `v2cfg-1` digests, `endpoint_id`s and resume fragments are untouched by construction. One regression test pins the tracked canary `config_digest()` to a literal hex string. No schema bump, no v2cfg-2, no per-field signature enumeration |

**Open:** S1 ($\tau$), S2 ($\epsilon_J$), S5 (C1 arm execution form), S6 (coverage floor), S7 (ceiling unit),
and the normalization law itself — under discussion.


---

## Appendix C. Calibration panel decision (2026-08-21)

**Question:** can the complete CATH 4.3 training split serve as the fixed allele-neutral
calibration panel?

**Answer: no.** Three read-only investigations; the decisive facts are measurements, not arguments.

### What CATH is

`work/immune-design/cath_4.3/chain_set.jsonl` (513 MB), split by `chain_set_splits.json["train"]`:
16,699 entries, **chain**-level (not domain — so no artificial excision termini), all sequenced,
minimum length 40 aa so nothing is too short for `k in [12,25]`, 16,631 pure AA20. Split is the
ESM/FAIR `cath4.3_topologysplit_202206` topology holdout (0 topology codes shared between splits;
no sequence-identity threshold). Two-Head scoring cost ~10 min GPU.

Incidental defect found: the two pre-computed CATH h_map parquets are empty (0x0, 636 B) because
`scripts/submit_precompute_h_cath.slurm:103` passes `--id-field CATH`, but `CATH` is a *list of
topology codes*; the loader stringifies it to `"['3.30.930']"`, matches no split id, and silently
drops all 22,508 rows. The correct flag is `--id-field name`. `PLAN_IF.md:645` states the same wrong
field. Worth fixing regardless of this decision.

### Why not CATH

1. **CATH is, by enforced policy, permanently disjoint from the deployment domain.** CATH 4.3 is the
   training corpus of the exact DPLM checkpoint Fusion V2 runs, and the IF test set is built to
   exclude anything >30 % identical to it (`inverse_folding/evaluation/assembly.py:95-103`, a
   build-breaking gate; all 100 high-risk rows carry `cath_overlap_flag=False` as a real filter
   result).
2. **The two Heads are nearly uncorrelated**, so panel composition — not sampling noise — sets
   $b_A-b_B$. Pearson $r=0.1908$ / Spearman $0.2040$ on 6,388 sequences scored by both Heads,
   replicated at $0.1905$ / $0.2171$ on an independent cohort. $\operatorname{sd}(u_A-u_B)=1.126$.
   An $A$-leaning versus $B$-leaning split of one panel swings $b_A-b_B$ by 9.25 units against a
   sampling floor of ~0.05.
3. Together: the equal-risk line would be fitted on the excluded region and applied to the admitted
   region, **and the same policy makes that transfer permanently untestable**. The falsifier is
   clean — at $r\approx0.9$ the shift would cancel and CATH would be fine. It is 0.19.
4. Supporting: CATH re-introduces a weaker form of the circularity §2.2.1 removed (the constants
   become a function of the DPLM *corpus*); it is per-chain not per-protein (9.36 % of entries come
   from multi-chain PDB entries; `1vq8` alone contributes 20); 14.2 % falls below the 102 aa
   deployment floor; and composition is topology-concentrated (`3.40.50` = 16.2 %, top-10 = 37.9 %,
   effective diversity $e^{H}=131$ of a nominal 882).

### A methodological correction worth keeping

Two investigators measured Head-training leakage into CATH and disagreed. The repo's frozen search
parameters (`--cov-mode 0 -c 0.8`) demand 80 % coverage of **both** sequences; CATH entries are
chains (median 212 aa) while Head training proteins are full-length UniProt (median ~522 aa), so
symmetric coverage systematically under-detects domain-level homology. Measured both ways:

| coverage mode | DRB1\*07:01 | DRB1\*04:01 | ratio |
|---|---:|---:|---:|
| `cov-mode 0` (under-detects) | 3.30 % | 3.38 % | 1.02 |
| `cov-mode 2` (correct here) | 8.69 % | 10.49 % | 1.21 |
| `cov-mode 2`, fident > 0.90 | 1.93 % | 2.60 % | 1.35 |

So the leakage asymmetry **is** real, and it favours Head B. It is also a **pool-size effect** —
1,308 versus 1,865 training proteins, ratio 1.43, against a leakage ratio of 1.45, with per-pool
leakage identical at 0.288 vs 0.293 — which means it is a property of the IEDB DRB1 data and **every**
structural panel will show it. It is removed by symmetrisation, not by panel choice: drop the
**union** of both alleles' homologous entries, never one allele's.

### How Tier-2 and CATH actually relate (measured 2026-08-21)

Worth stating explicitly, because the intuitive reading is backwards. Tier-2 is **not** a selection
from CATH; `doc/Data_Selection_v1.md` §4 Step 2 is "download PDB candidate pool (single-chain,
100-500 AA, < 2.5 A)" followed by "run MMseqs2 overlap filter against CATH training set" - CATH is
the **subtrahend**, not the source. Measured on the id sets:

| Quantity | Value |
|---|---:|
| Tier-2 distinct chains | 225,821 |
| CATH chains (all splits / train) | 22,508 / 16,699 |
| CATH chains present in Tier-2 | 13,225 = **58.8 %** of CATH |
| Tier-2 chains absent from CATH | 212,596 = **94.1 %** of Tier-2 |

So CATH is largely a subset of Tier-2 rather than the reverse, which is what one expects: CATH is a
redundancy-reduced, domain-classified selection out of the PDB, and a broad single-chain PDB pull
naturally covers most of it. Choosing Tier-2 therefore does not discard CATH's coverage - it adds
212 k chains of further natural sequence space, including the deployment region that CATH is
policy-excluded from.

One residual, recorded rather than acted on: 13,225 panel entries are CATH chains and CATH is the
DPLM training corpus, so the weaker circularity argument of §2.2.1 applies to 5.9 % of the panel.
That exposure is common to both alleles and therefore cancels in $b_A-b_B$, which is the only part
that is load-bearing.

### Selected panel

The deduplicated Tier-2 PDB candidate pool,
`work/immune-design/if_test_set/tier2_candidates_merged.fasta`: 226,048 raw chains → 57,127 distinct
sequences, 100 % already inside 100–500 aa, allele-neutral by construction (pool membership is
"single-chain PDB, 100–500 aa, < 2.5 Å", decided upstream of every immune screen), not spent as an
evaluation set, and — decisively — it **contains** the deployment domain: all 100 high-risk protein
ids are present. Build recipe:

1. deduplicate by exact sequence — 225,821 raw records to **57,127**;
2. keep 100–500 aa — **all 57,127 already are**, so the pool is inside the deployment domain by
   construction;
3. drop the **union** of both alleles' training homology, `cov-mode 2` at 30 % identity / 80 %
   coverage. MEASURED 2026-08-24: **14.06 % for Head A (DRB1\*07:01, 1,182 seen proteins) and
   15.69 % for Head B (DRB1\*04:01, 1,676 seen proteins)**; union 11,216; **asymmetry 1.63
   percentage points**. "Seen" is train ∪ val — train fits the weights and validation selects
   which checkpoint exists — and the held-out test split is not counted;
4. drop the 100 deployment proteins, so no protein's own natural sequence enters its own $u_a$;
5. drop the two per-allele evaluation sets (2,879 + 2,829 ids), so calibrating does not spend the
   test set; and
6. cluster by homology and keep one representative per cluster, so no family dominates — raw PDB
   determination bias is severe (top duplicate multiplicities 1731, 1404, 1091).

Executed by `scripts/build_dual_calibration_panel.py`, which reports every step. The final realized
counts, after the canonical-AA20 gate that this earlier recipe omitted, are recorded in Appendix
J.1 and supersede the intermediate counts previously printed here: 57,127 → 56,475 (AA20) →
45,326 (head homology) → 45,302 (deployment) → 44,021 (evaluation) → **13,836 clusters = the
panel**.

> **Correction (2026-08-24).** An earlier revision of this recipe cited "4.16 % / 4.91 % published
> pool figures" for step 3. Those numbers appear in no code, artifact or other document in this
> repository; they came from the DUALF0 subagent pass and were propagated without their provenance
> being checked, against this project's own traceability rule. The measured values are roughly
> **3.4x larger**. The direction of the old claim was right — `cov-mode 0` under-detects — but the
> magnitudes were not, and they should never have carried the word "published". Separately, that
> revision had no step 5 at all: it would have calibrated on the evaluation set.
Report on the built panel: $f_A$, $f_B$, the overlap-inclusion sensitivity of $b_A-b_B$, the realized
$\operatorname{SE}(b_A-b_B)/s$, and the **realized cross-allele correlation** — the last because the
existing $r\approx0.19$ estimates are within-family (uricase) and the whole domain-transfer argument
is calibrated against that number.

### Unrelated provenance defect

Neither frozen Head checkpoint carries a machine-readable record of which allele it is: `best.pt`
metadata has no allele field, both checkpoints share `config_hash` `418915354319`, and Head B's
`full_resolved_config.json` records `data.target_allele = "HLA-DRB1*07:01"` — a stale default from
`epitope_head/configs/data.yaml:10`. It is cosmetically wrong only (training resolves data from
`--data-dir`, `n_alleles=1`, and the manifest correctly carries `HLA-DRB1*04:01` on all 1,865 rows),
but allele identity currently rests on the run-directory name. For a program whose headline claim is
an A-versus-B asymmetry, the Dual overlay must pin identity to `head_checkpoint_digest` and refuse
equality between the two roles.

---

## Appendix D. Build progress and baseline test state (2026-08-21)

Implementation base `1bf0fe7`. DUALF1, DUALF1a's consumer contract, DUALF2 and DUALF3 are
implemented; DUALF4-F6 are not started.

| Task | Delivered | New tests |
|---|---|---|
| DUALF1 | `fusion_v2/joint_objective.py` | 51 |
| DUALF2 | `fusion_v2/dual_evidence.py`, `fusion_v2_runtime/dual_lookahead.py` | 29 |
| DUALF3 | `fusion_v2_runtime/dual_selection.py`; `archive.py` and `reward.py` parameterized | 11 |
| Dual-off compatibility | `test_fusion_v2_dual_off_compatibility.py`, digests pinned to literals | 5 |

### Baseline test state

Full suite on a compute node (job `12736402`, `--partition=cpu`, 4 cpu / 48 G, MaxRSS 0.9 G):
**2847 passed, 6 failed, 6 skipped** in 110 s.

All six failures are **pre-existing and unrelated** to the Dual work. Both failing files are
unmodified since HEAD, and neither reaches any changed module:

| Failure | Cause |
|---|---|
| `dplm_refiner/test_train_step_contract.py` (5) | `RuntimeError: The expanded size of the tensor (512) must match…` in the DPLM refiner training path. The file imports only `sys`, `pathlib`, `pytest`, `torch`; it contains zero references to `fusion_v2`, `archive` or `reward` |
| `test_reference_flow_fusion_v1_seeds.py::test_derive_seed_is_process_independent_under_pythonhashseed` | Line 74 hardcodes `cwd="/Users/jerry/Project/MHC-IF-fusion"`, a Mac path that cannot exist on Della. It can only pass on one machine, and it violates the project's own no-hardcoded-paths rule |

### Operational note

Do not run the full suite on a Della login node: it is killed by the memory cap (SIGKILL, exit 137)
partway through, and a truncated run reports no summary. A `cmd | tail` pipeline additionally
returns `tail`'s exit status, so such a run can look like a clean exit 0 while having been killed.
Run it under `sbatch` (`--partition=cpu --qos=short --time=02:00:00 --mem=48G` was ample), and read
the `PYTEST_EXIT` value rather than the pipeline's.

---

## Appendix E. Safety scope decision (2026-08-22)

| id | Decision | Consequence |
|---|---|---|
| S3b | **Role B safety is measured, not gated.** The inherited single-Head cumulative admission law is unchanged; `N_B^whole` against the frozen depth-0 reference is computed with the existing public pure primitive `safety.whole_landscape_new_hotspot` and recorded as telemetry on the Dual endpoint evidence | Zero change to `SafetyGate`, `admit_endpoint`, `EndpointAdmission`, `SafetyReferenceBinding`, `LivePartialState` or `config.safety`. No state digest, `state_id` or `endpoint_id` moves. Audit blockers B04 (second safety reference has no owner in state) and M18 (single-verdict typed admission) are **dissolved rather than solved** |

Rationale, in order of weight: on the frozen high-risk substrate the allele-A ceiling is already
`1000000.0` with the incremental gate off, so a second gate binds against nothing there; no
`delta_B` exists and its only producer is generative, which the calibration stage excludes; and a
second `SafetyReferenceBinding` sits inside the live state's canonical payload, so it would move
every endpoint identity and forfeit the byte-identical shared root the recursive comparison rests
on.

Claim boundary, recorded in `PLAN` §2.5 and `doc/Dual_Allele_Steering.md` §4.2.1: no result from
this program may state that a conjunctive per-allele safety gate was enforced. Promotion of the
telemetry to a gate is a conditional enhancement activated by observed drift.

---

## Appendix F. DUALF6 handoff and two corrections found while closing it

### What "one byte-identical D0" can mean

Discovered while building the three-arm evidence test, and it invalidates a GO check as written.

A `CompleteEndpoint` carries **one** Head's binding and score, and `donor_gate` refuses a donor
whose evaluator identity differs from the incumbent's -- correctly, since two evaluators are two
instruments and their difference is not a margin. So the three objective arms **cannot share one
endpoint object**: `b_only`'s endpoints must carry role B's evidence for the same molecule.

What they do share is the molecule and its identity. `make_endpoint_id` derives from
`(source_state_id, fork_index, sequence_md5)` and does **not** include the Head binding, so:

| quantity | joint | a_only | b_only |
|---|---|---|---|
| `sequence_md5` | same | same | same |
| `endpoint_id` | same | same | same |
| `source_state_id` | same | same | same |
| `content_digest` | same | same | **differs** |

`doc/RF_Fusion_V2_Dual_Allele_Cluster_Runbook.md` §4.3 required "shared-D0 endpoint digests are
byte-identical across the three arms". That is unsatisfiable for `b_only` and would fail every valid
run; it has been corrected to compare sequence and endpoint identity instead.

### Withdrawn: the panel-precision gate

`calibration_precision_fraction` and the fail-closed check on
$\operatorname{SE}(b_A-b_B)/s$ were **removed**. The underlying coupling is real and measured, but
turning it into a gate added a tuning knob to a calibration whose whole point is that **exactly one
number, $\tau$, is chosen by a human** and everything else is measured in the panel pass or derived
from the objective's geometry. It is now a reported diagnostic, consistent with the same
measure-don't-gate decision taken for role-B safety (Appendix E). The five panel diagnostics are
optional: a run that did not measure one omits it rather than recording zero.

### DUALF6 local evidence

| PLAN requirement | Evidence |
|---|---|
| complete new pure/unit/integration tests | 175 Dual tests across 10 files |
| existing V2 / sampler / policy / safety / archive / artifact / resume / preflight / driver regressions | full suite on a compute node: 3005 passed, 6 pre-existing failures, 6 skipped |
| Dual-off golden equivalence on seeds, bytes, decisions, artifacts, costs | seed module proven free of any Dual symbol; repeated legacy decisions identical; legacy bundle file set unchanged; legacy head cost event id unchanged and the Dual ids namespaced beside it; config digest, endpoint id and legacy run signature pinned to literal hex |
| fake-oracle end-to-end run with joint / a_only / b_only from one shared D0 | `test_all_three_arms_run_from_one_shared_depth_zero` |
| observer-only second Head produces no generative change | `test_binding_head_B_as_an_observer_changes_no_decision` -- Head B is scored and bound, the policy is untouched, the decision is identical |
| no NMP import or runtime field | structural check over all six Dual modules |

**Acceptance.** The code can answer whether Head B changes donor and write identity: the joint and
single-allele arms are comparable objects produced from one endpoint under one decision sequence,
and on the fixture they produce different transitions. It claims nothing about whether Dual
*works* -- that is the cluster runbook's to measure.

### Not implemented, deliberately

The reader's `--dual-capability` mode and the facade's common-objective ranking are in the PLAN §4
ownership matrix but are **read-only analysis over artifacts no run has yet produced**. Writing join
logic against a schema that has never been emitted is how an analysis path acquires an error nobody
can see. They are deferred to the point where P0/M1 artifacts exist; nothing in the run path depends
on them.

---

## Appendix G. Codex review 2026-08-22: eight findings, all confirmed, all repaired

An independent review of the frozen Dual work reported eight findings. Every one was reproduced
against the code before any change was made. The headline is worth stating plainly because it
invalidates the previous session's completion claim: **the Dual layer was a library that the
production path never called.**

### What was actually wrong

| # | Finding | How it was reproduced |
|---|---|---|
| 1 | **[P0] Dual reached the run signature and stopped there.** `runner_kwargs` carried no overlay, arm, Head B or objective authority; the cycle asked one Head; the archive ranked by `head_global_risk`. Outside tests there were ZERO call sites of `DualObjective(`, `DualSupportAuthority(`, `build_role_b_head_oracle(`, `dual_rank_key(` or `bind_paired_head_scores(`. `joint` / `a_only` / `b_only` would all have executed identical A-only V2 under three different signatures | grep over non-test code; read of `run_rf_fusion_v2.py:659` |
| 2 | **[P0] The Dual tables had no producer and could not hold their own numbers.** 57 of 58 declared columns were absent from `V2_COLUMN_TYPES`. Measured: an empty table typed `raw_risk_a` as `string`, and the first real row raised `ArrowTypeError: Expected bytes, got a 'float' object`. Separately, the joint LOO's A/B contrasts were collapsed to one scalar and the union reducer's two sides and winner were discarded before anything could record them | direct `write_stable_parquet` round trip with the pre-fix type map |
| 3 | **[P1] DUALF1a is not implemented, and the runbook contradicted the frozen contract.** `calibrate_v2_head_policy.py` is still the single-Head repeatability CLI; the runbook still defined $b_a,s_a$ from the NoD endpoints and invoked `--mode dual` / `--head-a-*` / `--head-b-*`, none of which exist | read of both files |
| 4 | **[P1] The overlap-stability criterion was written on the wrong coordinate.** The doc claimed only $b_A-b_B$ is load-bearing; the active-worst boundary is $R_A/s_A-R_B/s_B=b_A/s_A-b_B/s_B$. Adding one raw constant to both locations leaves $b_A-b_B$ exactly unchanged while moving the boundary by $\delta(1/s_A-1/s_B)$. The PLAN's $\lvert\Delta(b_A-b_B)\rvert/s$ also divided by an $s$ that is not defined for a pair | algebra, checked against the implementation |
| 5 | **[P1] The recursion desynchronized.** `donor_score_b` looked a score up by `endpoint_id` and validated nothing; `incumbent_joint_value` was any finite float; and `advance_lineage_incumbent` carried the Dual authority through unchanged, so role A moved to the adopted donor while role B stayed at $I_0$ | a test measured the stale threshold directly: $J(I_0)=19.096$ against the adopted donor's own $14.896$ — a **4.2-unit** phantom margin at every depth $\geq 2$ |
| 6 | **[P1] The gates compared normalized quantities against raw thresholds.** `challenger < held - epsilon_r` with a joint challenger and role A's raw floor; the write filter compared $\Delta J$ against the raw `local_contribution_tolerance`. `QuantityCoordinates.joint_margin` already derived the correct normalized floor and was dead code | read of `reward.py` / `policy.py`; a fixture with $s_A=2,s_B=8$ makes the two differ by 200x |
| 7 | **[P1] The Dual leave-one-out was half-billed and forged a retry.** Both alleles' counterfactual batches used `f"{prefix}:counterfactual"`, and the ledger's logical identity `(event_id, protein_id, arm, phase)` has no allele dimension. Also, the per-cycle cap was checked against the POSITION count while one position costs two Head calls | this is the un-repaired half of finding **C6** in §4.2 of this document: the lookahead events were namespaced, the counterfactual events were not |
| 8 | **[P1] The launch gate was blind to Dual.** `--dry-run` returned `EXIT_OK` before `resolve_dual` ran at all, and `assert_observed_head_b` was called with the overlay's OWN declared digests on both sides — a tautology that passes for any checkpoint on disk | read of the driver's control flow |

### What changed

* **Execution wiring.** New `fusion_v2_runtime/dual_runtime.py` carries the whole Dual half as one
  optional value threaded shard → ladder → cycle. Both Heads score both screens (the source pool
  and the descendant pool — depth $d$'s descendants are depth $d+1$'s source pool, so scoring only
  the first left the next rung un-orderable). Family selection and the archive elite take the joint
  rank key. `build_dual_stack` opens role B from shard inputs, proves the RESOLVED checkpoint
  against the overlay, and scores the safety reference so $J(I_0)$ is a number both instruments
  took. The cycle rebinds the policy with each pool's role-B scores, and the driver/factory must
  agree that this is a Dual run or the shard refuses.
* **Ledger convention, now uniform.** Role A keeps the legacy event id at every stage; only role B
  is namespaced. A Dual ledger is therefore the legacy ledger plus role B's rows, so "what did the
  second Head cost" is one subtraction and an `a_only` arm's cost rows join straight against a
  legacy run's. The per-cycle counterfactual cap is charged in Head calls on the Dual path and left
  untouched on the legacy path, whose signed artifact declares it in editable positions.
* **Recursion.** Role-B scores are proved to describe the same molecule, allele and window grid as
  role A's view before use; $J(I_d)$ is checked against the objective's own value on the incumbent
  the policy holds; and the authority advances with the lineage, clearing the donor map so a donor
  role B has not scored raises instead of resolving to a stale entry.
* **Units.** `JointComparison` carries the derived $\epsilon^{\mathrm{joint}}=\max_a\epsilon_a^{\mathrm{raw}}/s_a$
  and the two raw floors it came from; the donor gate and the write filter both use it on the Dual
  path and report the number they actually compared.
* **Artifacts.** `DUAL_COLUMN_TYPES` declares a type for every Dual column and asserts that set
  complete at import. Three row builders produce the tables from the objects the run decided with.
  The feedback table is keyed over the UNION of write-candidate and reopen-candidate positions,
  which are disjoint by construction — keying it off the contributions alone left the reopen columns
  structurally always empty, which is how the first version of the builder was caught.
* **Gates.** The overlay resolves before `--dry-run` returns; the observed-Head check moved to where
  a checkpoint is actually opened; and the assembly preflight now builds the joint policy model-free
  and refuses a policy that came back without the authority.
* **Documents.** The coordinate-law criterion is corrected in both the science doc and the PLAN, and
  now also gates the scale ratio $s_A/s_B$ — a recomputation can agree on the intercept and still
  disagree about which allele is worst over a whole region. The runbook's calibration section states
  the natural-panel contract and is marked NOT IMPLEMENTED with its blockers named.

### Still not done, and named

* **DUALF1a, the calibration producer.** Blocked on the Tier-2 panel build (five-step recipe in
  Appendix C, with homology recomputed at `cov-mode 2`) and on $\tau$. Nothing fabricates a
  calibration in the meantime.
* **$\tau$ (S1) and $\epsilon_J$ (S2)**, plus S5–S7. Unchanged, and still the launch blockers.
* The reader's `--dual-capability` mode and the facade's common-$J$ ranking, for the reason in
  "Not implemented, deliberately" above.

### A repo defect found while running the regression, unrelated to Dual

`tests/scripts/test_build_tetramer_input.py` and `tests/scripts/test_typed_targeting_confound.py`
abort collection for the WHOLE suite: both import scripts that are absent from this working tree and
untracked by git — `scripts/build_tetramer_input.py` was never added, and
`scripts/analysis/typed_targeting_confound.py` is covered by the `.gitignore` rule
`scripts/analysis/*` despite being documented in `doc/SCRIPTS.md`. Two committed test modules
therefore depend on files git cannot restore. Not caused by this work and not repaired here.

---

## Appendix H. Adversarial workflow review 2026-08-24: the repair itself was broken

An 8-agent adversarial review (5 independent lenses -> 2 skeptics -> 1 completeness critic) was run
against the Appendix G repair. It returned 42 candidate findings; the two skeptics killed 1 and
sustained 38, with 3 contested and 7 further gaps. The five blockers were then **reproduced by hand**
before any change was made, because a 38/42 sustain rate against skeptics instructed to default to
"not a defect" is itself a reason to distrust the panel.

**The headline is that the Appendix G repair replaced one failure with the same failure.** Its
finding #1 was "all three arms execute A-only V2 under three different signatures". After the repair
all three arms executed the *joint* law. Every arm still agreed, and the agreement still meant
nothing.

### The five blockers, each reproduced

| id | Defect | Reproduction |
|---|---|---|
| B1 | **The arm label was inert.** `build_dual_stack` built `DualObjective(overlay.calibration)` unconditionally; `arm` reached `DualRuntime`, the artifact rows and the run signature, and no decision. `grep` for an arm branch in any production decision path returned nothing | PLAN §7 states "Arm labels name the objective and nothing else"; DUALF3's acceptance requires replaying one pool through A-only/B-only/smooth-max/hard-max to change the ordering law. Neither was implemented |
| B2 | **Every Dual ladder run died at its first adopted donor.** The cycle rebinds role B's donor scores onto a LOCAL `support_policy`; `CycleOutcome` carried no policy back; the ladder advanced `cycle_kwargs["support_policy"]`, whose map is still `{}` | ran the real policy with the factory-shaped authority: `V2DualPolicyError: role B has no score for donor …`. The crash lands after the full generation, both Head batches and every definitive refold have been paid for |
| B3 | **The mechanism/qualification shard could not accept Dual.** `run_v2_mechanism_shard` had no `dual` parameter while the driver passed one — an uncaught `TypeError`, with `--dry-run` certifying the submission | read of the signature |
| B4 | **The three Dual tables were built and dropped.** `run_v2_shard` set `payload["dual_tables"]`; `aggregate_fragments` collects payload keys BY TABLE NAME and was given only the legacy names, and `write_v2_bundle` was never passed `dual_tables` | grep for consumers of `payload["dual_tables"]` returned the writer side only. No production run would have written a single Dual parquet |
| B5 | **Four feedback columns were declared bool and written as floats.** | `ArrowInvalid: Could not convert 0.0 with type float: tried to convert to boolean`. The bundle round-trip test missed it because it passed an EMPTY feedback table — a test asserting less than it claimed, which is the failure this very review was convened to look for |

### What changed

* **Arms are real.** `SingleAlleleObjective` (a `DualObjective` subclass, so every consumer's type
  check keeps working) values one allele's coordinate; `build_arm_objective` resolves the label.
  Both Heads are still scored, bound and published in every arm — an `a_only` run MEASURES role B
  and declines to steer on it, which is what makes the contrast interpretable. Each arm carries its
  own objective digest so two arms of one calibration can be split after the fact, and its own
  derived margin `eps_role/s_role` rather than the worse of the two, so the arms differ in the
  ordering law and NOT also in the gate width. `DualRuntime` refuses an arm whose objective is a
  different law.
* **The recursion closes.** `CycleOutcome` carries the policy the cycle actually decided with, and
  the ladder advances that. A legacy cycle hands back the object it was given, so this is a no-op
  there.
* **The tables are published**, as top-level payload keys the aggregator can see, and the manifest
  carries the arm, the overlay and calibration digests, the objective mode and tau, and role B's
  allele and checkpoint digest — so a joint bundle and an `a_only` bundle are distinguishable on
  disk.
* **Artifact fidelity.** `write_eligible` is read against the margin the run applied (carried on the
  decision evidence) instead of being recomputed on 0.0; `selected`/`selection_rank` come from the
  decision's own per-position rows instead of an event key the shard never writes; the reopen
  attribution names the conjunct that ORDERED the row rather than the first one merely present; and
  a typed stall keeps the per-allele evidence two Head batches produced.
* **Role B is watched.** `role_b_hotspot_telemetry` is wired into the runtime against the same frozen
  reference role A's ratchet uses, so the one safety statement Dual is allowed to make now has a
  producer. Absent a declared reference the columns are null, never zero.
* **Cost and ordering.** `ExactArchive.fork_view` carries the ordering law, so a matched arm cannot
  fork an elite ordered by `head_global_risk` while gating on J. The launch-gate projection prices
  both Heads.
* **`joint_comparison_for`** was left behind by the epsilon repair and built a zero-margin gate; it
  now derives the same margin the authority does.

### Test quality, which is where this all came from

The Appendix G repair shipped with a green suite (6 pre-existing failures, 4388 passing) that
coexisted with all five blockers. The reason is worth recording:

* no test drove the ladder with both Dual seams present, so B2 was invisible;
* the bundle round-trip passed an empty feedback table, so B5 was invisible;
* nothing asserted that the arms DIFFER, so B1 was invisible;
* and the DUALF4 union-screen test asserted `joint_set >= single_set`, which the union reducer
  guarantees whatever role B says. Measured, the two sets were identical — because that fixture's
  donor improves under role A at every masked position, leaving role B nothing to add. It has been
  replaced with a fixture whose donor carries `R` at position 9, measured to be rejected by role A
  with `no_improved_window`, and a role B landscape that improves the window covering it. The union
  must now make position 9 legal or the test fails.

New regression cover: `test_fusion_v2_dual_arms.py` (the arms order a real pool differently) and
`test_fusion_v2_dual_repair.py` (one test per defect above).

### Still open after this round

* ~~The M1 engine gap~~ — **withdrawn 2026-08-24, and the requirement with it.** The completeness
  critic reported that `paired.py` cannot express joint-versus-`a_only` on one shared prefix. That is
  true, and it does not matter. The Dual arm enters no seed derivation and the depth-zero root
  capture never sees the Dual runtime, so two launches differing only in `--dual-arm` already
  produce an identical root, lookahead pool and set of role A raw risks, and diverge only at
  selection. Measured on three separately executed runs: pool endpoint ids, sequence md5s and role A
  risks all identical; selected endpoint and J values different. **The comparison is already paired,
  across runs, for free.** An in-run engine would pay for the shared source prefix once instead of
  twice — a compute saving that changes no scientific property of the comparison. So it is not
  built; the in-process contrast procedure (93 lines of `DUAL_QUALIFICATION=1` /
  `--dual-qualification` / in-cell three-branch materialization) has been deleted from the runbook
  and the PLAN; and the driver refuses `--dual-overlay` together with the mechanism/qualification
  path, whose two arms differ by INTERVENTION and would therefore both run the SAME allele law —
  answering no allele question while looking like it does. The pairing property is now asserted in
  `test_fusion_v2_dual_arms.py` rather than assumed.
* ~~DUALF1a, tau, eps_J, S5-S7~~ — **all resolved since.** DUALF1a is built and has been RUN
  (Appendix J); tau and eps_J are frozen/deleted (Appendix I); S5 went with the in-process contrast;
  S7 was the ceiling overload, split in Appendix I.
* The completeness gaps, updated:
  * ~~the materializer's `head_b_*` runtime knobs are inert~~ — **fixed.** `--head-b-variant-id`,
    `--head-b-allele-idx` and `--head-b-window-batch-size` were shard inputs with ZERO readers (the
    signed overlay is the authority and is read from). They are removed rather than wired: a knob
    that reads as a control and changes nothing is worse than a missing one.
  * ~~the signed counterfactual ceiling is halved in the Dual domain~~ — **fixed** (Appendix I S7).
  * ~~the launcher's `DUAL_MODE` cross-check ignores the arm~~ — **fixed, and it was worse than
    reported.** `DUAL_MODE` was baked in at MATERIALIZATION from `--dual-arm`, which would have
    forced one materialized cell per arm — reinstating exactly the per-arm branching that was
    deleted when the arms became a launch-time flag. It is now a CAPABILITY (`dual`/`none`) and
    `--dual-arm` is gone from the materializer.
  * **still open**: nothing stats or digests role B's Head before the GPU allocation is paid for
    (`build_dual_stack` proves the checkpoint, but only once the Head is already being opened); and
    the overlay is still not a signed input role, so the calibration steering a run is bound to no
    input provenance. Both are cheap; neither blocks a launch.

---

## Appendix I. The v1 freeze (2026-08-24) — S1, S2, S7 closed

The user froze the remaining scientific decisions as one minimal rule set. Recorded here because
each one closes an item this document has been carrying as a launch blocker.

### S1 — $\tau$, from the normalized credit

Declared as the credit, derived as $\tau$:

$$
c_u = 0.10 \quad\text{(normalized risk units)}, \qquad \tau = \frac{0.10}{\log 2} = 0.14426950408889636.
$$

Verified bit-identical to `0.10 / math.log(2.0)`, and $\tau\log 2$ recovers `0.10` exactly. The
reading is direct: **the non-worst allele may buy at most 0.10 normalized risk units** against the
worse one. That keeps the objective worst-residual dominated while leaving a smooth two-sided signal
where the two alleles are close — which is what leave-one-out attribution needs in order to see
either allele at all.

Frozen in `inverse_folding/reference_flow/configs/v2_dual_smoothmax_policy_v1.json`, carrying BOTH
`max_nonworst_credit_normalized` and `tau`. `DualObjectiveLaw` cross-checks them (`|c_u - \tau\log2|
\le 10^{-12}`) so one human decision cannot be stored two inconsistent ways. The calibration
producer validates and signs the value and may never select or sweep it from P0/M1/C1 outcomes: a
credit chosen after seeing the result it credits is not a frozen operating point.

### S2 — $\epsilon_J$ is deleted, not estimated

It conflated two objects on two different sampling distributions: $\epsilon_{\mathrm{donor}}$ and
$\epsilon_{\mathrm{write}}$ protect ONE endpoint or leave-one-out decision from measurement drift,
while the experimental read estimates a MEAN effect by bootstrap. Subtracting a pointwise noise
floor from a bootstrap mean is not a valid test. The GO condition is now

$$
\operatorname{UCB}_{95\%}\left[\operatorname{mean}\left(\Delta J\right)\right] < 0,
$$

with the donor/write identity-change requirement, the full raw A/B report, and the shared-D0 / seed
/ budget / safety parity checks all retained. No `delta_practical` in v1; if a minimum practical
effect is ever wanted it gets its own name, because reusing the noise-floor symbol is what caused
this.

### The decision margins are a PAIRED difference

With $d_a = e_a/s_a$ and $e_a$ the Head's maximum same-sequence repeat drift:

$$
\epsilon_{\mathrm{donor}} = \epsilon_{\mathrm{write}} = 2\max\left(d_A, d_B\right).
$$

The factor of two is the conservative worst-case propagation for a paired difference: the donor
gate compares $J(I_d)$ against $J(Y^*_d)$ and the write filter compares $J(y^{(-i)})$ against
$J(y)$, so the two readings may drift in opposite directions by at most $d$ each. It is not an
estimate of the variance of the paired difference. The previous single-measurement bound
$\max_a d_a$ did not cover that worst case. `QuantityCoordinates.joint_margin` keeps its meaning
(the 1-Lipschitz propagation of ONE reading) and the objective gained `decision_margin`, which is
what both gates now compare against. A single-allele arm doubles its OWN floor, not the pair's.

The frozen raw figure `0.017012596130371094` is a RAW-logit quantity and may not be copied into the
normalized joint objective; it has to be re-measured per allele on the panel and divided by $s_a$.

### S7 — the ceiling was a semantic overload, now split

`config.py` describes the per-cycle ceiling as Head calls; `policy.py` has always used it to bound
candidate POSITIONS. Under one Head those are the same number, under two they are not — and an
earlier repair here charged it in Head calls under Dual, which **halved the editable candidate
domain purely because a second instrument existed**. That is a scientific change nobody asked for:
the domain a cycle may consider is a property of the design, not of how many Heads watch it.

The legacy field is left alone. The Dual overlay declares its own, explicitly named:

```ini
max_counterfactual_sequences_per_cycle = C      # candidate/sequence domain, unchanged
logical_head_calls_per_cycle           = 2 * C  # projected by the preflight, never enforced in the policy
```

All three arms bill $2C$: every arm scores both Heads, preserves the inherited role-A admission
gate, and emits role-B hotspot telemetry, so cost and candidate domain are identical across the
comparison. Role-B safety is not a gate in v1 (Appendix E). Overlay schema bumped to `dualcfg-2`.

### The resulting chain

```
fixed Tier-2 natural panel
  -> global protein-equal median/IQR coordinates (b_a, s_a)
  -> c_u = 0.10  ->  tau = 0.1442695
  -> repeatability-derived donor/write margins = 2 max(e_a / s_a)
  -> no epsilon_J
  -> unchanged candidate domain, doubled logical Head-call budget
```

DUALF1a is now closed: the producer, panel, measured calibration and re-signing seam are recorded in
Appendix J.

---

## Appendix J. The panel and the calibration, MEASURED (2026-08-24/25)

Everything below is a measurement on this cluster, not a projection. Artifacts live under
`/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/calibration/`.

### J.1 The panel exists now, and building it found three holes in this document

`scripts/build_dual_calibration_panel.py`, run as SLURM 12891590:

```
225,821 raw records
 →  57,127  exact-sequence dedup
 →  57,127  100–500 aa            (all of them already were)
 →  56,475  canonical AA20 only   (dropped 652: B, O, U, X, Z)
 →  45,326  minus the UNION of both Heads' training homology (11,149)
 →  45,302  minus the 100 deployment proteins
 →  44,021  minus both alleles' evaluation sets (2,879 + 2,829)
 →  13,836  one representative per homology cluster   ← the panel
```

**Measured training homology, for the first time ever in this repository:**

| | seen proteins (train ∪ val) | pool homologs | fraction |
|---|---|---|---|
| Head A — DRB1\*07:01 | 1,182 | 7,975 | **14.12 %** |
| Head B — DRB1\*04:01 | 1,676 | 8,914 | **15.78 %** |

Asymmetry **1.66 percentage points**; union 11,149. `cov-mode 2`, 30 % identity / 80 % coverage.

Three defects in earlier revisions of this document, all in the same passage, all found only by
executing it:

1. **the panel had no producer.** The recipe was prose; nothing built a panel, and
   `calibrate_v2_dual_objective.py` consumed a FASTA that did not exist. A runbook step that
   points at an artifact nothing produces cannot be handed to an agent.
2. **there was no step excluding the evaluation set.** Following the recipe as written would have
   calibrated on the test proteins.
3. **the 4.16 % / 4.91 % figures were unsourced and wrong** (see Appendix C's correction). The real
   values are ~3.4x larger. `head_train_overlap_flag` is a hardcoded `False` at every production
   write site, and `run_mmseqs_overlap` compares against the CATH / inverse-folding training set —
   a different question entirely. This overlap had genuinely never been computed.

**Not a defect but worth recording**: only 24 of the 100 deployment proteins and 1,302 of the 5,487
evaluation proteins survived far enough to be removed by their own filters. The rest were already
gone. That means the homology filter is doing real work — and that WITHOUT it, evaluation proteins
would have leaked into the panel in bulk.

### J.2 The calibration, SLURM 12891919 (1 h 26 m, 42 GB peak)

| quantity | $b_A$ | $s_A$ | $b_B$ | $s_B$ | derived margin |
|---|---|---|---|---|---|
| `global_risk` | −6.768 | 6.291 | −2.670 | 4.112 | 2.32e-07 |
| `positive_mass_density` | 0.570 | 0.738 | 0.935 | 0.669 | 7.07e-07 |
| `window_z` | −9.423 | 0.368 | −7.444 | 0.476 | 1.81e-05 |

$s_A/s_B = 1.53$ on the steering quantity. The scales genuinely differ, which is the empirical
content of the rule that $\tau$ must be declared in normalized units.

**Repeat drift is float-precision.** With the second pass issued through the same checkpoint at a
different window batch size (64 → 32): `global_risk` 9.54e-07 for both Heads, `window_z` 6.68e-06
for both. Relative to values of order 5 that is float32 rounding from a changed reduction order, not
model non-determinism. So

$$
\epsilon_{\mathrm{donor}} = \epsilon_{\mathrm{write}} = 2\max_a \frac{e_a}{s_a} = 4.64\times10^{-7}.
$$

**The consequence must be stated rather than buried**: against a credit of $c_u = 0.10$ that is one
part in 2.6 million. The donor gate and the write filter are, in practice, *strict improvement past
float noise*. This is the freeze rule applied faithfully and it is defensible — the Head is
reproducible to float32, so any difference above that floor is a real difference of instrument
opinion. It is also five orders of magnitude below V2's legacy raw figure `0.017012596130371094`.
Both originate from same-sequence repeatability, but under different populations and scoring-path
comparisons: the legacy number is twice the maximum stored-versus-live raw-logit drift on the V2F5A
mechanism population, while this number is the normalized two-Head bound measured by changing the
production window-batch reduction shape on the natural panel. Neither is a design effect size.

**How the first measurement of this was wrong.** The producer originally varied only the panel
ORDER between passes. `ProductionHeadOracle.score` groups requests by `protein_id` and every natural
panel entry is its own chain, so each group holds one sequence and the order never reaches a
batching decision. It measured `e_a = 0` for both Heads and this document nearly recorded that as
evidence of determinism. It was evidence of nothing. The window batch size is the only shape that
actually varies, and equal batch sizes are now refused with that reason.

### J.3 Two numbers that make the domain-transfer argument weaker, not stronger

**Cross-allele Pearson $r = 0.096$** on the full panel (probe agreed at 0.094). The existing
$r \approx 0.19$ is a WITHIN-family (uricase) estimate. On natural sequences the two Heads are very
nearly uncorrelated. `doc/Dual_Allele_Steering.md` §2.2 argues that panel-composition effects
largely cancel between the alleles, and the strength of that cancellation scales with how correlated
the two Heads are. At half the assumed correlation, it cancels about half as much. This does not
invalidate the calibration; it means the argument it rests on is weaker than when it was written,
and the number it rests on is now measured rather than borrowed.

**Bootstrap SE of the normalized intercept $b_A/s_A - b_B/s_B$ is 0.0185**, against $c_u = 0.10$:

$$
\frac{\operatorname{SE}}{c_u} = 0.185.
$$

PLAN §5 DUALF1a requires $\operatorname{SE} \ll c_u$. One fifth is not obviously "much less than".
And it **cannot be improved by enlarging the panel**: 44,021 eligible sequences cluster into only
13,836 homology-independent units, and the SE is over those units. Reaching $c_u/10$ would need
roughly four times as many independent families than the deduplicated Tier-2 pool contains.

### J.4 Decisions requested during execution — resolved in J.7

| # | Decision | Why it cannot be defaulted | Cost of getting it wrong |
|---|---|---|---|
| **D1** | **$C$**, the per-cycle candidate domain | It is signed into the overlay and therefore into the run signature. The current artifact carries the placeholder **278** (M1's historical value). PLAN §7 says it comes from the frozen cohort's maximum legal editable domain, which is measurable — but signing it is a decision | All three arms are invalidated and must be re-run. Re-signing itself is cheap: `--resign-from` is a 1.7 s CPU operation |
| **D2** | **Overlap-inclusion sensitivity: run it or declare it unmeasured** | The calibration contract requires a measured sensitivity to including Head-training homologs and a bound on the drift in $b_A/s_A - b_B/s_B$. Only the overlap-excluded panel exists. At 14–16 % overlap this is not a formality | The intercept is −0.4265, i.e. $4.3\,c_u$. A drift of even a few percent of it consumes the whole credit band. Recommend running it: ~90 min GPU, one more panel build |
| **D3** | **Accept $\operatorname{SE}/c_u = 0.185$, or change the panel** | The PLAN's own gate is qualitative ("$\ll$") and this is the first time it has had a number to face | The only levers all cost something real: keep the evaluation proteins (contaminates the eval set), loosen clustering (admits correlated units as independent), or accept a wider uncertainty on the equal-risk line |

None of these blocked code completion. Their frozen resolutions and the one remaining measurement
are in J.7.

### J.5 Two defects found while verifying the artifact

**The credit cross-check did not survive a round trip.** `DualObjectiveLaw` gained
`declared_credit_normalized` and a check that $\tau\log 2$ equals it, but the field was in neither
the overlay serializer nor its loader — so a reloaded overlay carried `None` and the check was
skipped everywhere except inside the producer that never needed it. Fixed; the field is always
written and optional on read so a pre-existing overlay still loads.

**Measuring and signing were one act.** Changing $C$ meant re-running 86 minutes of GPU. They are
now separate: `--out-calibration` writes the measured calibration, `--resign-from` re-emits an
overlay from it in 1.7 s with no model loaded, and still through the overlay's own strict loader so
a hand-edited calibration is refused at signing rather than at the launch gate.

### J.6 What is now true of the launch path

* the frozen objective spec exists and its $\tau$ cross-checks against its declared credit;
* the panel exists, is reproducible from one command, and reports every filter;
* the calibration exists and loads through the production seam
  (`load_dual_overlay_file` → `build_arm_objective`), and the three arms resolve to three different
  ordering laws with three different objective digests and their own decision margins;
* D1 and D3 are frozen in J.7; D2 has a frozen measurement protocol and remains to be executed;
* after that artifact exists, the remaining launch gate is the runbook-versus-code consistency
  pass, which has now been justified twice by finding exactly this class of hole.

### J.7 Decisions frozen (2026-08-25)

#### D1 — the candidate ceiling is substrate-specific, not one global scalar

Freeze the scientific unit as **unique counterfactual sequences / legal candidate positions per
cycle**. The number of Heads observing each sequence is resource accounting and may not shrink this
domain.

| Campaign | Frozen $C$ | Maximum logical Head calls per Dual cycle | Source |
|---|---:|---:|---|
| M1 one-cycle mechanism | **278** | **556** | historical signed V2F5A mechanism ceiling |
| C1 high-risk D4/K12 | **454** | **908** | measured maximum legal editable domain, signed artifact basis `6T88_A` |

The measured objective calibration is shared, but the two campaigns receive separately signed
overlays whose run signatures bind their own $C$. Every objective arm within one campaign uses the
same $C$ and scores both Heads, so neither candidate coverage nor Head cost is confounded with the
objective law. Re-signing from the measured calibration is the authorized path; no GPU calibration
rerun and no hand edit are allowed.

#### D2 — run the homology-inclusion sensitivity, but never select the primary panel from it

Run one additional calibration in which the panel build is identical to J.1 except that the union
of Head-A/Head-B training homologs is **not** removed. Canonical-AA20, deployment exclusion,
evaluation-set exclusion, homology clustering, representative selection, Head identities,
objective law and scoring protocol remain identical. This is more precisely an
**overlap-inclusion sensitivity** than a new calibration candidate.

The overlap-excluded 13,836-cluster panel remains the primary coordinate authority regardless of
the sensitivity outcome; it was selected by an outcome-independent leakage rule. The sensitivity
artifact must report at least

$$
\Delta\theta
=
\left(\frac{b_A}{s_A}-\frac{b_B}{s_B}\right)_{\mathrm{overlap\ included}}
-
\left(\frac{b_A}{s_A}-\frac{b_B}{s_B}\right)_{\mathrm{primary}},
$$

the four coordinate constants, $s_A/s_B$, cross-allele correlation, panel/cluster counts and
content digests. Report whether $|\Delta\theta|$ exceeds $c_u=0.10$ as a **panel-sensitivity
label**, not as an outcome-dependent switch to the contaminated panel. An integrity/provenance
failure blocks launch; the measured magnitude limits the claim to this frozen reference panel but
does not retroactively choose another coordinate system. This measurement must finish before M1/C1
submission.

#### D3 — accept the measured precision and retire the qualitative gate

Accept $\operatorname{SE}/c_u=0.185$ for v1. The 13,836-cluster panel is a deliberately frozen
reference coordinate system, so its median/IQR values are exact descriptive constants of that
panel; the bootstrap SE measures sensitivity to which homologous families instantiate the broader
natural-sequence domain, not runtime measurement noise and not donor/write uncertainty.

Delete the qualitative launch requirement $\operatorname{SE}\ll c_u$. Preserve the SE and its
bootstrap method in the calibration report, and scope the method claim to the content-bound panel.
Do **not** reintroduce evaluation proteins, loosen clustering to manufacture nominal sample size,
or change $c_u$ after observing this number. Generalization of the coordinate law to a different
natural panel is a later robustness study, not part of the minimal Dual capability gate.

#### Launch state after these decisions

The science is now frozen. Code execution may proceed immediately with the M1/C1 overlay re-signing
and the overlap-inclusion sensitivity job. Cluster generation remains closed until that sensitivity
artifact and the final runbook-versus-code consistency report both exist.
