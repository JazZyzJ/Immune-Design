# PLAN — Epitope Head Wave-3: IoU-aligned span objective + 5-fold CV

> Worktree-isolated (branch `epi-head-wave3`). Builds on the converged HIMP sweep
> (`report/epi_himp_convergence.md`). North star unchanged: residue/region landscape for
> IF guidance. Goal metric: **IoU≥0.5/0.7 region AP** (priority 1) then **residue density
> Pearson/Spearman** (priority 2); exact-AP/far-FP guardrails; NMP=reference. Report =
> **5-fold CV test number (mean±std) + one all-data final model**.

## Motivation (established)
- Density (residue Pearson 0.34–0.43) already ≫ NMP (0.14–0.17) — paradigm win, robust.
- Remaining gap = window **IoU region AP** (~0.04–0.07 below NMP on 0701). MECHANISM:
  IoU AP is window-level → driven by the **span scorer logits = span loss**; residue
  hyperparams (λ, β, margin) are decoupled and don't move it. So the lever is the **span
  objective**.
- OVERFITTING: `best.pt` is selected by val **exact-AP** (`monitor_metric=pp_ap`) — wrong
  criterion; it over-trains (npoff/0701 test IoU peaks at epoch 24 = 0.584, decays to
  best.pt 0.555; val-test gap grows 0→0.09). Fix = goal-metric selection + CV.

## Design decision
**Primary lever — window IoU-ranking auxiliary loss (refinement of the workflow's Obj 3).**
Wave-2 `lcsoft` showed that merely *down-weighting* near negatives (≈ grading the margin) is
~neutral on IoU. The M6 metric is an AP over windows ranked by score, so the direct lever is to
**train the score to rank by IoU**: a pairwise margin loss over candidate windows where a window
with higher max-IoU-to-GT must out-score a lower-IoU window by a margin. This actively pulls
overlapping windows *up* (not just stops pushing them down), reuses the proven
`residue_pairwise_margin_loss` pattern (low risk), and is **additive** (λ_iou_rank=0 ⇒
bit-for-bit legacy). Density is protected: the residue loss is untouched (held at λ=0.1).
- Alternatives kept documented: graded-margin in `margin_hard_loss` (workflow Obj 3) and
  IoU-soft-target regression (Obj 1) as fallbacks if ranking underperforms. Reject soft-Dice
  (Obj 2, residue-side, off-target per its own verdict).

**Companion — post-hoc goal-metric checkpoint selection (Harness B).** Score retained
`epoch_*.pt` on held-out by M6 IoU50/70 + M2a Pearson (lexicographic), emit `best_goal.pt`.
Fixes the exact-AP selection overfit. Reuses the benchmark metrics (refactor `_span_iou`/M6/M2a
into an importable module so trainer/benchmark/selector share ONE implementation).

**Substrate — cluster-level 5-fold CV (Harness A).** Partition mmseqs90 clusters (NOT proteins)
into 5 folds so homologues never span train/test. Per fold: internal train/val from the 4
training folds, held-out fold = test. Report mean±std; then an all-data final model for deploy.

## Tasks (TDD; superpowers:executing-plans)
- **T2 (core, self-contained) — DONE-FIRST**: `epitope_head/training/span_geom.py`
  (`window_iou_matrix`, `max_iou_per_window`, vectorized half-open IoU matching
  `benchmark_iedb_test._span_iou`) + `losses.py::window_iou_rank_loss` + wire `lambda_iou_rank`
  into `compute_loss` (additive, default 0). Unit tests RED→GREEN.
- **T3 — trainer wiring**: in `_forward_union_and_compute_losses` compute union-window IoUs vs
  chunk-local positives, pass `window_logits/window_ious/lambda_iou_rank` to `compute_loss`; log
  `loss_iou_rank`. Config keys in `train.yaml`/`configs/__init__.py` (validate). New config
  `cnn_himp_iourank.yaml`.
- **T1 — 5-fold CV**: `epitope_head/data/cv_fold_builder.py` + Stage-D6 in `build_dataset.py`
  → `splits/<profile>/fold_{train,val,test}_{k}.txt`. `submit_epitope_train.slurm` +
  `train_v2_ablation.py` gain `FOLD_ID`. All-data final-model mode.
- **T4 — goal selection + aggregation**: refactor benchmark M6/M2a into importable
  `epitope_head/evaluation/el_metrics.py`; `select_goal_checkpoint.py` → `best_goal.pt`;
  `aggregate_cv_results.py` → mean±std table.
- **T5 — docs**: register new scripts in `doc/SCRIPTS.md`; LOG entry; update this plan.

## Experiment ladder
1. **Fail-fast (single split, 0701, seed42)**: Arm A = mixed_margin (current) vs Arm B =
   +λ_iou_rank, same data, post-hoc goal-select epochs. Decide on held-out **test** IoU at
   matched density.
2. If pass → **5-fold CV** 0701, both arms, goal-selected; report mean±std + all-data final.
3. If 0701 wins → confirm on 0401.

## Success / KILL
- Success: held-out IoU50 AP(B) − (A) ≥ **+0.03** AND Pearson(B) ≥ Pearson(A) − 0.01 (matched
  density) AND exact-AP(B) ≥ (A) − 0.02 AND far-FP not materially up.
- KILL: if span-objective change still moves IoU50 < **+0.01** at matched density → the
  span-loss-bottleneck claim is falsified; stop span-loss variants; the win is the density
  paradigm. (Also: free check — does adding `best_goal` selection alone recover the +0.029 the
  epoch-sweep measured? validates Harness B before judging the loss.)
