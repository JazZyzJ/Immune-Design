# EL Evaluation Protocol — Revision Proposal

> **Scope**: Revision of `doc/Benchmark_proposal.md` §1.4 (Evaluation Protocol) for the EL Ability benchmark (F2 subfigure).
> **Status**: Proposal / design — pending full 126-protein measurement to finalize metric defaults.
> **Relation to existing doc**: Supersedes §1.4 of `Benchmark_proposal.md`; §1.1–1.3 (scientific claim, methods, test set) remain unchanged.

---

## 1. Motivation

The current protocol uses **exact 15-mer span match** for positive labels: a candidate window `(start, end)` is labeled positive iff it exactly matches an IEDB EL-reported epitope span. All other windows — including those that overlap a positive by 14 out of 15 residues — are negatives.

This convention creates a **category error** against the design role of the epitope head:

- The head is architected as a **continuous residue-level immunogenicity landscape** — serving Level 1 (reference-flow guidance gradient) and Level 2 (candidate resampling weight) in the inverse-folding pipeline.
- The head is **not** intended as a Level 3 oracle for exact-span classification — that role is held by NetMHCIIpan (NMP).
- A smooth landscape necessarily assigns comparable risk to residues adjacent to an epitope boundary. Under exact-span labels, these adjacent high scores are counted as false positives, penalizing the designed smoothness.

NMP, by contrast, is trained directly on 15-mer peptide labels and naturally produces concentrated high scores at exact-span windows.

## 2. Empirical Evidence

Diagnostic analysis on 13 proteins where NMP clearly outperforms head on the current protocol (190 GT positive windows total):

| Signal | Observation |
|---|---|
| AP under label relaxation | exact `0.1917` → overlap (any) `0.2785` (+46% relative) → IoU≥0.5 `0.2453` |
| Rank of GT exact window | median `128` |
| Rank of best *overlapping* non-exact window | median `11` |
| GT spans with a better-ranked overlapping non-exact window | **169 / 190 (89%)** |
| Modal boundary shift | (±1, ±2) residues — same region, slight boundary drift |
| Top-10 FP overlapping any GT | 71 / 130 (54.6%); IoU≥0.5: 50.0% |
| Top-10 FP "truly far" (gap > 10 aa from any GT) | 48 / 130 (36.9%) |

**Interpretation**: Head successfully localizes epitope **regions** (median rank 11 of the best covering window) but disagrees with the IEDB 15-mer convention on exact boundaries. This is consistent with the designed Level 1 behavior and with the underlying MHC-II biology — the 9-mer binding core determines presentation; the 15-mer is a reporting convention and off-by-1/2 boundary shifts typically preserve the same core.

The 37% "truly far" top-10 FPs are a separate, smaller category of real model error, tracked independently (see §7).

## 3. Goal-Anchored Metric Requirements

Any candidate metric must satisfy all of:

1. **Mathematically well-defined** — no divergences on edge cases (all-negative windows, tied scores, zero-density residues).
2. **Score-scale invariance** — head outputs unbounded real logits; NMP outputs %Rank ∈ [0,1]. The metric must not depend on absolute score magnitude (rank-based, or explicit normalization step).
3. **Identifiability** — hyperparameter choices must not silently flip rankings between head and NMP. All degrees of freedom are documented and fixed before measurement.
4. **Null baseline computable** — analytic or Monte Carlo baseline for the random scorer.
5. **Reviewer-defensible precedent** — prior use in MHC/epitope prediction (per-residue AUC) or adjacent fields (object detection mAP@IoU, image distribution metrics).

These requirements are hard gates; alignment with the "landscape" framing is considered only after.

## 4. Candidate Metrics — Legitimacy Assessment

| ID | Metric | Well-defined | Scale-invariant | Identifiability | Null | Precedent | Verdict |
|---|---|---|---|---|---|---|---|
| M1 | Overlap-/IoU-AP (exact-k window) | ✅ | ✅ (rank) | ⚠ needs 1-to-1 assignment | ✅ | **Strong** (COCO mAP) | **Accept** w/ COCO-style assignment |
| M2a | Residue AUC/AP, binary coverage label | ✅ | ✅ (rank) | ⚠ aggregation choice (`max`/top-K mean/LSE) | ✅ | **Strong** (segmentation AUC, NMP eval) | **Accept** as primary |
| M3 | 1D EMD (Wasserstein-1) on residue density | ✅ | ⚠ temperature τ in score→density | ⚠ τ-sensitive | requires MC | Medium (WMD, FID analog) | Accept as diagnostic only |
| M4 | JS divergence on residue density | ✅ | ⚠ same τ issue | ⚠ spatially blind (JS is pointwise) | requires MC | Medium | **Reject** — strictly weaker than M3 |
| M5 | Gaussian-smoothed label AUC | ✅ | ✅ | ⚠ σ is hyperparameter | ✅ | Weak — non-standard continuous-label AUC | **Reject** — M2a subsumes |
| M6 | Overlap/IoU-AP with tolerance ladder (COCO-style) | ✅ | ✅ | ✅ (ladder reported, not collapsed) | ✅ | **Strong** (COCO) | **Accept** as secondary |

## 5. Selected Metric Suite

**Span set (shared across all metrics)**: $k \in [12, 25]$ enumerated for every protein, both head and NMP scored on the identical set. This is the existing `benchmark_iedb_test.py` default (`--min-k 12 --max-k 25`); no length restriction is imposed at metric time.

### 5.1 Primary — M2a: Residue-level AUC / AP / Pearson

**Ground truth** per residue `r`:
- `y_cover(r) = 1` iff `r ∈ ⋃ IEDB_EL_epitope_spans`, else `0` (length-agnostic — GT spans of any length contribute)
- `y_density(r) = |{epitope e : r ∈ e}|` (integer coverage count, ≥ 0)

**Model score per residue**:
- `score(r) = max over all spans (s,e) ∈ candidate_set with s ≤ r < e of f_model(s,e)`
- Identical aggregation for head and NMP. Both share the same span set, so residue support is identical at every position.

**Metrics**:
- `residue_AUC` (vs `y_cover`) — ROC-AUC, binary positives, macro-averaged across proteins.
- `residue_AP` (vs `y_cover`) — PR-AUC, macro-averaged.
- `residue_Pearson` (vs `y_density`) — rewards matching the coverage *hotspot* structure (residues inside multiple overlapping epitopes carry higher weight).
- `residue_Spearman` (vs `y_density`) — rank-based variant, robust to score distribution.

**Aggregation fixed as `max`** as default. Alternative aggregations (top-K mean, LogSumExp) are tracked as ablations only.

### 5.2 Secondary — M6: Overlap/IoU-AP Ladder

**Label relaxation** ladder, applied to the same multi-`k` span set as the primary:
- `exact` — IoU = 1.0 (equivalent to current protocol)
- `IoU ≥ 0.7`
- `IoU ≥ 0.5`
- `overlap > 0` — any residue overlaps (most lenient)

IoU is length-agnostic — pred and GT spans of differing length are handled by the formula `|w∩g| / |w∪g|` directly; no length restriction needed.

**Assignment (critical for validity)**:
- Greedy 1-to-1 from COCO mAP convention.
- For each GT span, the highest-scoring predicted window with IoU above threshold is assigned as TP; remaining overlapping predictions are FP.
- Without this rule, multiple overlapping predictions all counting as TP inflates AP — **not comparable** to the exact baseline.

**Metrics**: AP, AUC, Recall@K for K ∈ {50, 100} at each IoU tier.

**Interpretation**: the AP(IoU) curve shows tolerance sensitivity. Exact-IoU AP is the current baseline; the upward trend toward IoU>0 quantifies how much of the head's apparent weakness is exact-boundary penalty.

### 5.3 Diagnostic — M3: Normalized 1D EMD

**Score → density** conversion (uses the same residue-level `score(r)` as §5.1):
- `p_gt(r) = y_density(r) / Σ_r y_density(r)` (skip proteins with all-zero density)
- `p_pred(r) = softplus(score(r)) / Σ_r softplus(score(r))`; τ=1 (fixed, not swept)

**Metric**: `EMD_norm = (1/L) · Σ_r |CDF_pred(r) - CDF_gt(r)|`, reported as `Similarity = 1 - EMD_norm ∈ [0, 1]`.

Role: **supplementary evidence only**. If primary metrics (M2a, M6) converge on the category-error interpretation, M3 serves as a second independent confirmation. Not claimed as the main metric in the paper.

### 5.4 Statistical Protocol

- **Macro-averaging**: compute per-protein metric, then mean across proteins. Matches current protocol and is robust to across-protein variance in positive rate.
- **Bootstrap 95% CI**: 1000 resamples over proteins.
- **Paired significance test**: Wilcoxon signed-rank on per-protein metric (head vs NMP, same protein set). Report p-value and effect size.

## 6. Implementation Plan

| Component | LOC estimate | Dependencies |
|---|---|---|
| M2a residue aggregation + metrics | ~250 | existing `scripts/benchmark_iedb_test.py`, `epitope_head/training/eval_metrics.py` |
| M6 overlap/IoU labels + greedy assignment + ladder | ~200 | `build_window_labels` extension |
| M3 residue density + 1D EMD | ~80 | numpy only |
| Bootstrap CI + Wilcoxon | ~40 | scipy.stats |

Recommended execution order: (1) M2a → validates the framing, (2) M6 → validates NMP-compatibility story, (3) M3 → bonus evidence.

Integration point: extend `scripts/benchmark_iedb_test.py` with `--metric-suite {exact,residue,iou_ladder,emd,all}` flag rather than creating new top-level scripts. Register any new scripts in `doc/SCRIPTS.md` per the Reuse-First Gate.

## 7. Separately-Tracked Item: "Truly Far" Top-K FPs

The 37% of top-10 FPs with gap > 10 aa from any GT are independent of metric reformulation — they are real model errors. Post-measurement audit proposed:

- Examine 40-50 far FPs across proteins
- Classify by: (a) repeat motifs (high-scoring P1/P4/P6/P9-like residues in non-epitope regions), (b) low-conservation flexible regions with generic "immunogenic-looking" features, (c) potential label noise (IEDB under-reporting in that region)
- Decisions on model revision (data augmentation, length-aware weighting, auxiliary loss) deferred until audit completes.

## 8. Open Decisions (to be resolved after full-126 run)

1. **M6 IoU ladder granularity**: {exact, 0.7, 0.5, >0} is proposed default; can expand to {exact, 0.8, 0.6, 0.4, 0.2, >0} if visual curve warrants.
2. **Whether to include M3 in paper main text or supplement**. Default: supplement, unless M2a and M6 alone are considered insufficient by reviewers.
3. **Alternative aggregation ablations** (top-K mean, LogSumExp) — run as supplementary table, not default.

## 9. Relation to Existing Protocol

- `doc/Benchmark_proposal.md` §1.1 (Scientific Claim), §1.2 (Comparison Methods), §1.3 (Why Competitive), §1.5+ (sections after Evaluation Protocol) — **unchanged**.
- §1.4 (Evaluation Protocol) — **superseded** by this document. The exact-span AP remains as the M6 `exact` tier of the ladder (so no data is discarded; the old number is a special case of the new ladder).
- `LOG.md` and `PROGRESS.md` should reference this file once the revised protocol is committed.

## 10. Revision Log

- 2026-04-23 · initial draft · based on 13-protein diagnostic analysis.
