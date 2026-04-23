import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_iedb_test import (
    _analyze_near_miss,
    _build_iou50_labels,
    _build_overlap_labels,
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
