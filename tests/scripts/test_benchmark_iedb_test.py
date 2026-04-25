import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_iedb_test import (
    _aggregate_residue_scores,
    _analyze_near_miss,
    _bootstrap_ci,
    _build_iou50_labels,
    _build_overlap_labels,
    _build_residue_labels,
    _compute_emd,
    _compute_iou_ladder,
    _compute_residue_metrics,
    _detection_ap,
    _detection_recall_at_k,
    _invert_nmp_wins,
    _iou_greedy_labels,
    _iou_threshold_key,
    _load_windows_cache,
    _paired_wilcoxon,
    _save_windows_cache,
    _select_near_miss_group,
    _span_gap,
    _span_iou,
    _span_overlap_len,
    _summarize_near_miss_group,
)


def test_span_helpers_cover_overlap_gap_and_iou():
    assert _span_overlap_len((0, 4), (2, 6)) == 2
    assert _span_overlap_len((0, 4), (4, 8)) == 0

    assert _span_gap((0, 4), (4, 8)) == 0
    assert _span_gap((0, 4), (6, 8)) == 2
    assert _span_gap((6, 8), (0, 4)) == 2

    assert _span_iou((0, 4), (2, 6)) == 2 / 6
    assert _span_iou((0, 4), (4, 8)) == 0.0


def test_overlap_and_iou50_labels_use_region_level_match():
    spans = torch.tensor([
        [0, 4],
        [1, 5],
        [2, 6],
        [6, 10],
    ], dtype=torch.long)
    positives = [{"start_0b": 0, "end_0b": 4}]

    overlap = _build_overlap_labels(spans, positives)
    iou50 = _build_iou50_labels(spans, positives)

    assert overlap.tolist() == [1, 1, 1, 0]
    assert iou50.tolist() == [1, 1, 0, 0]


def test_analyze_near_miss_detects_overlap_outranking_exact():
    spans = torch.tensor([
        [0, 4],
        [0, 5],
        [1, 4],
        [8, 12],
    ], dtype=torch.long)
    positives = [{"start_0b": 0, "end_0b": 4}]
    scores = torch.tensor([0.5, 0.9, 0.8, 0.1], dtype=torch.float32)

    result = _analyze_near_miss(spans, positives, scores, fp_cutoffs=(2,))

    assert result["best_exact_rank"] == 3
    assert result["median_exact_rank"] == 3
    assert result["median_best_overlap_nonexact_rank"] == 1
    assert result["n_gt_with_overlap_outranking_exact"] == 1
    assert result["gt_examples"][0]["best_overlap_nonexact_span"] == [0, 5]
    assert result["top_fp_breakdown"]["2"]["overlap_iou_gte_0_8"] == 1
    assert result["top_fp_breakdown"]["2"]["overlap_iou_gte_0_5"] == 1


def test_select_near_miss_group_uses_exact_ap_winner():
    head_row = {
        "metrics": {
            "head_primary": {"ap": 0.4},
            "nmp_primary": {"ap": 0.2},
        },
    }
    nmp_row = {
        "metrics": {
            "head_primary": {"ap": 0.1},
            "nmp_primary": {"ap": 0.3},
        },
    }
    tie_row = {
        "metrics": {
            "head_primary": {"ap": 0.2},
            "nmp_primary": {"ap": 0.2},
        },
    }

    assert _select_near_miss_group(head_row) == "head_better"
    assert _select_near_miss_group(nmp_row) == "nmp_better"
    assert _select_near_miss_group(tie_row) is None


def test_summarize_near_miss_group_aggregates_counts_and_medians():
    rows = [
        {
            "near_miss": {
                "exact_ap": 0.1,
                "overlap_ap": 0.3,
                "iou50_ap": 0.2,
                "n_gt_total": 2,
                "n_gt_with_overlap_outranking_exact": 1,
                "gt_examples": [
                    {"exact_rank": 5, "best_overlap_nonexact_rank": 1},
                    {"exact_rank": 15, "best_overlap_nonexact_rank": 3},
                ],
                "top_fp_breakdown": {
                    "10": {
                        "overlap_iou_gte_0_8": 2,
                        "overlap_iou_gte_0_5": 1,
                        "overlap_iou_lt_0_5": 0,
                        "near_gap_le_2": 0,
                        "near_gap_le_5": 1,
                        "near_gap_le_10": 0,
                        "far_gap_gt_10": 6,
                    },
                },
            },
        },
        {
            "near_miss": {
                "exact_ap": 0.3,
                "overlap_ap": 0.5,
                "iou50_ap": 0.4,
                "n_gt_total": 1,
                "n_gt_with_overlap_outranking_exact": 1,
                "gt_examples": [
                    {"exact_rank": 9, "best_overlap_nonexact_rank": 2},
                ],
                "top_fp_breakdown": {
                    "10": {
                        "overlap_iou_gte_0_8": 1,
                        "overlap_iou_gte_0_5": 0,
                        "overlap_iou_lt_0_5": 1,
                        "near_gap_le_2": 2,
                        "near_gap_le_5": 0,
                        "near_gap_le_10": 1,
                        "far_gap_gt_10": 5,
                    },
                },
            },
        },
    ]

    summary = _summarize_near_miss_group(rows)

    assert summary["n_proteins"] == 2
    assert summary["mean_exact_ap"] == 0.2
    assert summary["mean_overlap_ap"] == 0.4
    assert summary["mean_iou50_ap"] == 0.30000000000000004
    assert summary["median_exact_rank"] == 9
    assert summary["median_best_overlap_nonexact_rank"] == 2
    assert summary["n_gt_total"] == 3
    assert summary["n_gt_with_overlap_outranking_exact"] == 2
    assert summary["gt_with_overlap_outranking_exact_fraction"] == 2 / 3
    assert summary["top_fp_breakdown"]["10"]["overlap_iou_gte_0_8"] == 3
    assert summary["top_fp_breakdown"]["10"]["far_gap_gt_10"] == 11


# ── EL landscape metric suite tests ─────────────────────────────────

def test_invert_nmp_wins_flips_sign():
    wins = {(0, 15): 0.1, (1, 15): 0.8}
    inv = _invert_nmp_wins(wins)
    assert inv == {(0, 15): -0.1, (1, 15): -0.8}


def test_aggregate_residue_scores_respects_k_filter_and_takes_max():
    wins = {
        (0, 15): 2.0,   # covers residues 0..14
        (2, 15): 3.0,   # covers residues 2..16
        (10, 12): 7.0,  # ignored when k_filter=15
    }
    scores, mask = _aggregate_residue_scores(wins, L=20, k_filter=15)
    # Coverage: residues 0..16 covered by some k=15 span; 17..19 not.
    assert mask[:17].all()
    assert not mask[17:].any()
    # r=0: only span (0,15); r=5: both spans → max = 3.0; r=16: only (2,15).
    assert scores[0] == 2.0
    assert scores[5] == 3.0
    assert scores[16] == 3.0
    # With k_filter=None, k=12 span at (10,22) contributes; r=11 → max = 7.0.
    scores_all, mask_all = _aggregate_residue_scores(wins, L=20, k_filter=None)
    assert scores_all[11] == 7.0
    assert mask_all.all()


def test_build_residue_labels_counts_overlapping_epitopes():
    y_cover, y_density = _build_residue_labels(
        [{"start_0b": 2, "end_0b": 8}, {"start_0b": 5, "end_0b": 10}],
        L=12,
    )
    # y_density: positions 2,3,4 → 1; 5,6,7 → 2; 8,9 → 1; else 0.
    assert y_density.tolist() == [0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 0, 0]
    assert y_cover.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]


def test_compute_residue_metrics_returns_none_on_degenerate_labels():
    # Everything covered → y_cover all-1 → not evaluable (no negatives).
    score_res = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    y_cover = np.ones(3, dtype=np.int64)
    y_density = np.ones(3, dtype=np.int64)
    assert _compute_residue_metrics(score_res, mask, y_cover, y_density) is None


def test_compute_residue_metrics_reports_expected_auc_on_separable_example():
    score_res = np.array([0.1, 0.2, 0.9, 0.8, 0.05, 0.02], dtype=np.float32)
    mask = np.ones(6, dtype=bool)
    y_cover = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)
    y_density = y_cover.copy()
    out = _compute_residue_metrics(score_res, mask, y_cover, y_density)
    # Perfectly separable.
    assert out["auc"] == 1.0
    assert out["ap"] == 1.0
    assert out["pearson"] is not None and out["pearson"] > 0.5
    assert out["n_residues_scored"] == 6


def test_iou_greedy_labels_matches_highest_iou_unassigned_gt():
    # Two GTs: [0,15) and [50,65). Three predictions — order by score:
    #   [1,16) score 0.9  → best IoU with GT0 = 14/16 = 0.875 → TP
    #   [50,65) score 0.8 → IoU=1 with GT1 → TP
    #   [0,15)  score 0.5 → GT0 already taken, GT1 already taken → FP
    spans = np.array([[1, 16], [50, 65], [0, 15]], dtype=np.int64)
    scores = np.array([0.9, 0.8, 0.5], dtype=np.float64)
    positives = [
        {"start_0b": 0, "end_0b": 15},
        {"start_0b": 50, "end_0b": 65},
    ]
    tp, n_gt = _iou_greedy_labels(spans, scores, positives, iou_threshold=0.7)
    assert n_gt == 2
    assert tp.tolist() == [1, 1, 0]


def test_iou_greedy_labels_zero_threshold_uses_any_overlap():
    spans = np.array([[14, 29], [0, 15]], dtype=np.int64)
    scores = np.array([0.9, 0.5], dtype=np.float64)
    positives = [{"start_0b": 0, "end_0b": 15}]
    # [14,29) overlaps GT by 1 residue → IoU=1/29, passes threshold=0.0.
    tp, n_gt = _iou_greedy_labels(spans, scores, positives, iou_threshold=0.0)
    assert n_gt == 1
    assert tp.tolist() == [1, 0]


def test_detection_ap_and_recall_match_hand_calculation():
    # 3 GT; predictions in score-descending order: TP, FP, TP, FP.
    sorted_tp = np.array([1, 0, 1, 0])
    # precisions at TP events: 1/1=1.0, 2/3≈0.667; recall denom = 3.
    # AP = (1.0 + 0.667) / 3 = 0.5556
    assert abs(_detection_ap(sorted_tp, n_gt=3) - (1.0 + 2 / 3) / 3) < 1e-9
    # Recall@2 = 1 TP / 3 GT = 0.333
    assert abs(_detection_recall_at_k(sorted_tp, 3, 2) - 1 / 3) < 1e-9
    # Recall@4 = 2 / 3
    assert abs(_detection_recall_at_k(sorted_tp, 3, 4) - 2 / 3) < 1e-9


def test_iou_threshold_key_canonicalizes_values():
    assert _iou_threshold_key(1.0) == "exact"
    assert _iou_threshold_key(0.0) == "overlap_any"
    assert _iou_threshold_key(0.5) == "iou_0p50"


def test_compute_iou_ladder_uses_full_multi_k_span_set():
    # Mix k=12, k=15, k=18 windows. All should participate (length-agnostic
    # IoU). One GT is k=12 (length 12), the other is k=18 — only multi-k
    # support can hit the exact tier on both.
    wins = {
        (0, 15): 0.6,    # IoU=12/15=0.8 with GT0 (12-mer)
        (0, 12): 0.9,    # exact match with GT0
        (50, 18): 0.95,  # exact match with GT1 (18-mer)
        (40, 12): 0.5,   # FP, no overlap
    }
    positives = [
        {"start_0b": 0, "end_0b": 12},   # k=12 GT
        {"start_0b": 50, "end_0b": 68},  # k=18 GT
    ]
    out = _compute_iou_ladder(
        wins, L=100, positives=positives,
        iou_thresholds=[1.0, 0.5, 0.0], recall_ks=(50,),
    )
    assert out["n_pred_windows"] == 4
    assert out["n_gt"] == 2
    # Exact tier: both GTs perfectly matched (different k).
    exact = out["tiers"]["exact"]
    assert exact["ap"] is not None
    # Recall@50 at exact: 2 of 2 GTs found.
    assert exact["recall_50"] == 1.0
    # IoU≥0.5 tier: should also score the (0,15)/(0,12) pair → 0.8 IoU.
    iou50 = out["tiers"]["iou_0p50"]
    assert iou50["ap"] is not None and iou50["ap"] > 0
    assert iou50["recall_50"] == 1.0


def test_compute_emd_returns_none_when_density_is_zero():
    # No positives → y_density all zeros → EMD undefined.
    wins = {(0, 15): 1.0, (1, 15): 2.0}
    y_density = np.zeros(20, dtype=np.int64)
    assert _compute_emd(wins, L=20, y_density=y_density) is None


def test_compute_emd_uses_full_multi_k_aggregation_by_default():
    # Mixed k windows; default (k_filter=None) must aggregate all and
    # require full residue coverage (every residue in some span).
    L = 20
    y_cover, y_density = _build_residue_labels(
        [{"start_0b": 5, "end_0b": 10}], L=L,
    )
    # Cover all 20 residues with a sliding window of k=12.
    wins = {(s, 12): 0.0 for s in range(L - 12 + 1)}
    # Add a k=15 span concentrating mass over residues 0..14 (weak).
    wins[(0, 15)] = 0.0
    # And a k=12 span concentrating mass over residues 5..16 (which covers
    # the GT span 5..9). Multi-k aggregation should reflect this peak.
    wins[(5, 12)] = 50.0
    out = _compute_emd(wins, L=L, y_density=y_density)
    assert out is not None
    assert 0 <= out["emd_norm"] <= 1
    assert abs(out["emd_norm"] + out["similarity"] - 1.0) < 1e-9
    # Sanity: with the peak aligned to GT, similarity should be > 0.5.
    assert out["similarity"] > 0.5


def test_bootstrap_ci_is_seeded_and_brackets_mean():
    vals = [0.4, 0.5, 0.6, 0.7, 0.8]
    ci1 = _bootstrap_ci(vals, n_resamples=200, seed=0)
    ci2 = _bootstrap_ci(vals, n_resamples=200, seed=0)
    assert ci1 == ci2  # deterministic under seed
    assert ci1["low"] <= ci1["mean"] <= ci1["high"]
    assert ci1["n"] == 5


def test_paired_wilcoxon_handles_none_entries():
    head = [0.6, None, 0.7, 0.8]
    nmp = [0.5, 0.4, None, 0.6]
    out = _paired_wilcoxon(head, nmp)
    # Only (0.6, 0.5) and (0.8, 0.6) are valid pairs → n_pairs=2.
    assert out is not None
    assert out["n_pairs"] == 2
    assert out["mean_diff"] > 0


def test_paired_wilcoxon_returns_none_when_fewer_than_two_pairs():
    assert _paired_wilcoxon([0.5], [0.4]) is None
    assert _paired_wilcoxon([None, None], [0.5, 0.6]) is None


def test_windows_cache_round_trip_preserves_dicts(tmp_path):
    # Mixed coverage: P1 has both head + NMP, P2 head-only (NMP failed),
    # P3 NMP-only (head somehow missing — defensive case).
    head_wins = {
        "P1": {(0, 12): 0.5, (0, 15): 0.7, (5, 13): -1.2},
        "P2": {(0, 12): 0.1, (3, 18): 2.5},
    }
    nmp_wins = {
        "P1": {(0, 15): 0.2, (5, 13): 0.05},
        "P3": {(10, 15): 0.4},
    }
    metadata = {
        "allele": "HLA-DRB1*07:01",
        "min_k": 12,
        "max_k": 25,
        "variant_id": "LC1",
        "n_proteins": 3,
    }
    cache_path = str(tmp_path / "cache.parquet")
    _save_windows_cache(cache_path, head_wins, nmp_wins, metadata)

    head_loaded, nmp_loaded, meta_loaded = _load_windows_cache(cache_path)
    assert head_loaded == head_wins
    assert nmp_loaded == nmp_wins
    assert meta_loaded == metadata


def test_windows_cache_handles_predictor_only_proteins(tmp_path):
    # A protein with NMP scores but no head should round-trip with empty
    # head-side dict (no spurious zero entries).
    head_wins = {"P1": {(0, 15): 0.9}}
    nmp_wins = {"P1": {(0, 15): 0.1}, "P2": {(0, 15): 0.2}}
    cache_path = str(tmp_path / "cache.parquet")
    _save_windows_cache(cache_path, head_wins, nmp_wins, metadata={})
    h, n, _ = _load_windows_cache(cache_path)
    assert "P2" not in h
    assert h["P1"] == {(0, 15): 0.9}
    assert n == nmp_wins
