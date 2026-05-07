# Epitope Head Improvement Proposal

> Status: design backbone. This document defines the scientific direction for improving the epitope head. It is not an implementation plan.

## 1. Target Role

The epitope head should not be optimized as a NetMHCIIpan-like exact peptide oracle.

The target role is a **contiguous-window-supported residue/region immunogenicity landscape** for inverse-folding guidance. Exact EL spans remain important observed anchors, but the downstream use case is hotspot localization and redesign pressure, not standalone exact-span classification.

This reframes the original v1 design:

- v1 already defined window logits, residue hotspot maps, and global risk.
- The mismatch is mainly in the training semantics: current losses and hard negatives still treat exact boundaries as if they were the main biological target.
- The revised target is to preserve correct localization while avoiding artificial pressure to sharpen every off-by-one or off-by-two boundary.

## 2. Current Diagnosis

The main issue is not simply model capacity or runtime batch size.

The stronger hypothesis is:

- **Supervision mismatch**: observed EL spans are sparse positive anchors, but nearby unobserved windows are currently treated too much like reliable negatives.
- **Negative semantics mismatch**: overlap or boundary-shift windows often represent the same epitope region, so using them as strong negatives conflicts with the intended landscape behavior.
- **Metric mismatch**: exact AP is useful as a guardrail, but it over-penalizes region-level near misses.
- **Calibration gap**: EL data provides positives but no reliable experimental negatives, so raw cross-protein risk calibration is weak.

The desired model should improve region correctness and maintain contiguous biological structure. Smoothness is useful only when it supports correct hotspot localization; it must not become the primary objective by itself.

## 3. Negative Semantics

Training negatives should be separated by biological reliability:

- **Exact positives**: observed EL spans.
- **Near / ambiguous windows**: overlapping, boundary-shifted, or adjacent windows near an observed positive. These are not reliable negatives.
- **Far decoys**: windows sufficiently far from all observed positives. These are the most reliable contrastive negatives under EL-only supervision.

The first improvement should remove near-positive windows from the strong-negative role.

## 4. Overlap-Aware Weighting

The preferred framing is a **monotone negative-penalty schedule**, not a hard reassignment of near windows into positives.

The schedule controls how strongly a sampled non-positive window is penalized as a negative:

- high overlap / very small gap: zero or near-zero negative penalty
- intermediate overlap / small gap: reduced negative penalty
- far from all positives: full negative penalty

Candidate schedule families:

- linear clamp
- sigmoid transition
- exponential transition
- thresholded smooth schedule, with an ignore zone near positives and a smooth ramp outside it

Linear clamp is only the simplest debugging baseline. A sigmoid or exponential schedule may be more natural if we want a smooth transition between ambiguous and reliable negatives.

Important distinction:

- **Downweighting a negative does not create a new positive.**
- It only states that this window is not reliable evidence against the epitope region.
- Turning near windows into soft positives is a separate stronger assumption and remains to be decided.

To be determined:

- whether near-overlap windows should be ignored or softly downweighted
- whether the schedule should depend on IoU, overlap ratio, residue gap, or a combination
- whether any soft-positive label should be introduced later

## 5. Two-Level Supervision

The current model already produces a residue hotspot map at inference by aggregating window logits. A natural improvement is to supervise this downstream quantity during training.

The proposed structure is:

```text
window logits z(s,k)
  -> contiguous-window aggregation
  -> residue hotspot h(i)
  -> residue / region-level supervision
```

This keeps the residue signal biologically constrained: residue risk is derived from contiguous peptide windows, not from a free independent per-residue classifier.

The two levels should have different meanings:

- span-level supervision anchors observed EL windows against reliable far decoys
- residue-level supervision encourages correct epitope-region localization

The two losses should not require absolute numerical calibration between `z(s,k)` and `h(i)`. They should primarily use ranking or relative objectives.

To be determined:

- aggregation function: max, log-sum-exp, top-k mean, or another window-to-residue operator
- residue target: binary coverage, density, or weighted coverage
- loss form: pairwise ranking, weighted BCE, AP surrogate, or correlation-style auxiliary
- relative weight between span-level and residue-level terms
- whether to add a weak window-residue consistency constraint

## 6. Block-Wise Biology

Epitope biology is contiguous. A residue-level metric is acceptable only if the model remains constrained by contiguous windows.

Therefore:

- do not introduce an unconstrained residue classifier as the primary output
- do not reward smoothness independently of correctness
- preserve window-derived hotspot construction
- track far false positives separately, since they represent real localization errors

Continuity is a biological prior. Correct localization is the objective.

## 7. Global Risk

Global risk is needed because redesign pressure should depend on how immunogenic the current protein appears to be. A protein that truly needs redesign should have a sufficiently prominent hotspot, even if the hotspot is small.

Simple percentile-only summaries are insufficient, because every protein has a top percentile by definition. Global risk should capture hotspot significance, not only rank position.

Raw cross-protein risk calibration is unreliable under EL-only supervision because unobserved windows are not true negatives. Global risk should therefore be designed as protein-wise and WT-relative whenever possible.

Candidate global-risk concepts, formulae to be defined later:

- WT-relative risk
- hotspot prominence
- hotspot burden
- candidate-pool risk reduction
- foldability-risk tradeoff temperature

To be determined:

- whether global risk should be computed from window logits, residue hotspots, or both
- how to distinguish a truly prominent hotspot from background score drift
- how global risk should set redesign temperature or guidance strength
- how much cross-protein calibration is required for the intended redesign workflow

## 8. Working Direction

The next model-improvement direction should prioritize:

- correcting negative semantics before changing model architecture
- avoiding strong negative pressure on near-positive windows
- adding window-derived residue/region supervision only after the negative semantics are clean
- using exact AP as a guardrail, not the main optimization target
- optimizing for residue-level and IoU-tolerant correctness without allowing arbitrary smooth false hotspots

This direction keeps the head aligned with inverse-folding guidance rather than pushing it toward an exact peptide oracle.
