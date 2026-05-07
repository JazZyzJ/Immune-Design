"""HIMP1 tests: span relation classification + schedule families + weighted sample_negatives.

Covers PLAN_EPI_IMP.md §5 HIMP1 TDD Gate:
  1. RED: adjacent_or_near span receives w=1.0 when schedule=ignore.
  2. RED: exact_positive span can be sampled as a negative.
  3. RED: counterfactual_disrupted span overlapping WT positive is dropped by ignore logic.
  4. RED: unsupported schedule name silently falls back to ignore.
  5. GREEN: weights and relation labels match expected behavior across fixtures.

Conventions: 0-based, half-open spans [start, end). Pep length = end - start.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from epitope_head.training.near_positive import (
    SpanRelation,
    classify_span_relation,
    compute_negative_weight,
    get_schedule,
    SCHEDULE_REGISTRY,
    RESERVED_SCHEDULES,
)
from epitope_head.training.negatives import sample_negatives


# ── Helpers ─────────────────────────────────────────────────────────────────

def _pos(start: int, length: int = 15) -> dict:
    return {"start_0b": start, "end_0b": start + length, "pep_len": length}


# ── Span relation classifier ────────────────────────────────────────────────

class TestSpanRelationClassifier:

    def test_exact_match_classified_as_exact_positive(self):
        positives = [(10, 25)]
        rel, gap = classify_span_relation((10, 25), positives, near_gap_max=10)
        assert rel == SpanRelation.EXACT_POSITIVE
        assert gap == 0

    def test_overlap_nonexact_classified_correctly(self):
        # Candidate shifts by 5 from positive; overlap > 0 but not exact.
        positives = [(10, 25)]
        rel, gap = classify_span_relation((15, 30), positives, near_gap_max=10)
        assert rel == SpanRelation.OVERLAP_NONEXACT

    def test_adjacent_within_near_gap_max(self):
        # Non-overlapping candidate; end-point gap = max(|30-10|, |45-25|) = 20.
        positives = [(10, 25)]
        # Below threshold is FAR_DECOY ...
        rel, gap = classify_span_relation((30, 45), positives, near_gap_max=10)
        assert rel == SpanRelation.FAR_DECOY
        assert gap == 20
        # ... at-or-above threshold flips to ADJACENT_OR_NEAR.
        rel2, gap2 = classify_span_relation((30, 45), positives, near_gap_max=20)
        assert rel2 == SpanRelation.ADJACENT_OR_NEAR
        assert gap2 == 20

    def test_just_inside_near_gap_max(self):
        # End-point gap of 5 should be adjacent.
        positives = [(10, 25)]
        rel, gap = classify_span_relation((15, 30), positives, near_gap_max=10)
        # (15, 30) overlaps (10, 25): overlap_nonexact takes precedence
        assert rel == SpanRelation.OVERLAP_NONEXACT
        # Now a non-overlapping case at gap=10
        rel2, gap2 = classify_span_relation((20, 35), positives, near_gap_max=10)
        # overlap_start=20, overlap_end=25 → overlap=5>0
        assert rel2 == SpanRelation.OVERLAP_NONEXACT

    def test_just_outside_near_gap_max_is_far_decoy(self):
        # End-point gap = 11, near_gap_max = 10  → far_decoy
        positives = [(10, 25)]
        rel, gap = classify_span_relation((36, 51), positives, near_gap_max=10)
        # max(|36-10|, |51-25|) = max(26, 26) = 26 → far
        assert rel == SpanRelation.FAR_DECOY
        assert gap == 26

    def test_classify_returns_min_gap_over_multi_positives(self):
        positives = [(10, 25), (100, 115)]
        # Candidate (30, 45) gap to (10,25)=20 and to (100,115)=70 → min=20
        rel, gap = classify_span_relation((30, 45), positives, near_gap_max=20)
        assert rel == SpanRelation.ADJACENT_OR_NEAR
        assert gap == 20

    def test_overlap_only_with_one_positive(self):
        positives = [(10, 25), (100, 115)]
        rel, _ = classify_span_relation((22, 37), positives, near_gap_max=5)
        # (22,37) overlaps (10,25) at [22,25)
        assert rel == SpanRelation.OVERLAP_NONEXACT


# ── Schedule registry & weight computation ──────────────────────────────────

class TestScheduleRegistry:

    def test_v0_schedules_registered(self):
        assert {"ignore", "linear_clamp", "sigmoid"} <= set(SCHEDULE_REGISTRY.keys())

    def test_reserved_schedules_listed(self):
        assert {"exponential", "thresholded_smooth"} <= RESERVED_SCHEDULES

    def test_get_schedule_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            get_schedule("totally_made_up")

    def test_reserved_schedule_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="exponential"):
            get_schedule("exponential")
        with pytest.raises(NotImplementedError, match="thresholded_smooth"):
            get_schedule("thresholded_smooth")


class TestComputeNegativeWeight:

    def test_far_decoy_always_one_regardless_of_schedule(self):
        for schedule in ["ignore", "linear_clamp", "sigmoid"]:
            w = compute_negative_weight(
                SpanRelation.FAR_DECOY, end_point_gap=50,
                near_gap_max=10, schedule_name=schedule,
            )
            assert w == 1.0, f"schedule={schedule}"

    def test_disrupted_always_one_regardless_of_schedule(self):
        for schedule in ["ignore", "linear_clamp", "sigmoid"]:
            w = compute_negative_weight(
                SpanRelation.COUNTERFACTUAL_DISRUPTED, end_point_gap=0,
                near_gap_max=10, schedule_name=schedule,
            )
            assert w == 1.0

    def test_exact_positive_returns_zero_defensively(self):
        # exact_positive should never be sampled as negative; defensive 0.
        w = compute_negative_weight(
            SpanRelation.EXACT_POSITIVE, end_point_gap=0,
            near_gap_max=10, schedule_name="sigmoid",
        )
        assert w == 0.0

    # TDD Gate #1: ignore schedule must zero out adjacent_or_near.
    def test_ignore_zeroes_adjacent_or_near(self):
        for gap in [0, 1, 5, 10]:
            w = compute_negative_weight(
                SpanRelation.ADJACENT_OR_NEAR, end_point_gap=gap,
                near_gap_max=10, schedule_name="ignore",
            )
            assert w == 0.0, f"gap={gap}"

    def test_linear_clamp_endpoints(self):
        # default w_low=0, w_high=1
        w0 = compute_negative_weight(
            SpanRelation.ADJACENT_OR_NEAR, end_point_gap=0,
            near_gap_max=10, schedule_name="linear_clamp",
        )
        w_full = compute_negative_weight(
            SpanRelation.ADJACENT_OR_NEAR, end_point_gap=10,
            near_gap_max=10, schedule_name="linear_clamp",
        )
        assert w0 == pytest.approx(0.0)
        assert w_full == pytest.approx(1.0)

    def test_linear_clamp_midpoint(self):
        w_mid = compute_negative_weight(
            SpanRelation.ADJACENT_OR_NEAR, end_point_gap=5,
            near_gap_max=10, schedule_name="linear_clamp",
        )
        assert w_mid == pytest.approx(0.5)

    def test_linear_clamp_custom_endpoints(self):
        w_low = compute_negative_weight(
            SpanRelation.ADJACENT_OR_NEAR, end_point_gap=0,
            near_gap_max=10, schedule_name="linear_clamp",
            schedule_params={"w_low": 0.2, "w_high": 0.9},
        )
        assert w_low == pytest.approx(0.2)

    def test_sigmoid_monotone(self):
        ws = [
            compute_negative_weight(
                SpanRelation.ADJACENT_OR_NEAR, end_point_gap=g,
                near_gap_max=10, schedule_name="sigmoid",
            )
            for g in range(0, 11)
        ]
        for i in range(len(ws) - 1):
            assert ws[i] <= ws[i + 1] + 1e-9, f"non-monotone at i={i}: {ws[i]} > {ws[i+1]}"

    def test_overlap_nonexact_uses_schedule_at_gap_zero(self):
        # PLAN: overlap_nonexact mirrors adjacent_or_near at gap=0
        w_lc = compute_negative_weight(
            SpanRelation.OVERLAP_NONEXACT, end_point_gap=0,
            near_gap_max=10, schedule_name="linear_clamp",
        )
        # with default w_low=0, w_high=1, gap=0 → 0.0
        assert w_lc == pytest.approx(0.0)

        w_ig = compute_negative_weight(
            SpanRelation.OVERLAP_NONEXACT, end_point_gap=0,
            near_gap_max=10, schedule_name="ignore",
        )
        assert w_ig == 0.0

    # TDD Gate #4: unsupported names must raise (no silent fallback).
    def test_unsupported_schedule_raises_not_silent_fallback(self):
        with pytest.raises(NotImplementedError):
            compute_negative_weight(
                SpanRelation.ADJACENT_OR_NEAR, end_point_gap=5,
                near_gap_max=10, schedule_name="exponential",
            )


# ── sample_negatives: backward-compat + relation/weight attachment ──────────

class TestSampleNegativesBackwardCompat:

    def test_default_call_unchanged(self):
        # Without near_positive_cfg, behavior is identical to legacy.
        positives = [_pos(50)]
        rng = np.random.RandomState(0)
        negs = sample_negatives(
            protein_length=200,
            positives=positives,
            neg_ratio=4,
            hard_negative_fraction=0.5,
            hard_neg_offset_range=20,
            rng=rng,
            strict=False,
        )
        # Legacy result is a list-like with size = neg_ratio * len(positives).
        assert len(negs) == 4
        # No positive should appear as negative.
        pos_set = {(p["start_0b"], p["end_0b"]) for p in positives}
        for n in negs:
            assert (n["start_0b"], n["end_0b"]) not in pos_set

    # TDD Gate #2: exact_positive can never be sampled as negative.
    def test_exact_positive_excluded_from_negatives(self):
        positives = [_pos(50), _pos(100), _pos(150)]
        rng = np.random.RandomState(0)
        negs = sample_negatives(
            protein_length=300,
            positives=positives,
            neg_ratio=10,
            hard_negative_fraction=0.5,
            hard_neg_offset_range=20,
            rng=rng,
            strict=False,
        )
        pos_set = {(p["start_0b"], p["end_0b"]) for p in positives}
        for n in negs:
            assert (n["start_0b"], n["end_0b"]) not in pos_set


class TestSampleNegativesWithSchedule:

    def test_relations_and_weights_attached(self):
        positives = [_pos(50)]
        rng = np.random.RandomState(0)
        np_cfg = {
            "near_gap_max": 10,
            "schedule": "linear_clamp",
            "schedule_params": {},
            "metric": "endpoint_gap",
        }
        negs = sample_negatives(
            protein_length=200,
            positives=positives,
            neg_ratio=4,
            hard_negative_fraction=0.5,
            hard_neg_offset_range=20,
            rng=rng,
            strict=False,
            near_positive_cfg=np_cfg,
        )
        # Lists exposed via attributes; one entry per negative.
        assert len(negs._relations) == len(negs)
        assert len(negs._weights) == len(negs)
        # Each relation is a SpanRelation member.
        for rel in negs._relations:
            assert isinstance(rel, SpanRelation)
        # Each weight in [0, 1].
        for w in negs._weights:
            assert 0.0 <= w <= 1.0

    # TDD Gate #3: counterfactual_disrupted weight=1.0 even with overlap.
    def test_disrupted_weight_one_even_if_overlap_logic_would_filter(self):
        # Disrupted spans (mutation-derived) must keep w=1.0 and the
        # counterfactual_disrupted relation regardless of near-positive logic.
        positives = [_pos(50)]
        # Disrupted span at the same position as a positive (rare safety case)
        # would be filtered by the existing positive_set check; design verifies
        # disrupted spans that DO survive are tagged correctly.
        disrupted = [{"start_0b": 80, "end_0b": 95, "pep_len": 15}]  # gap=30 → far in legacy
        rng = np.random.RandomState(0)
        np_cfg = {
            "near_gap_max": 10,
            "schedule": "ignore",
            "schedule_params": {},
            "metric": "endpoint_gap",
        }
        negs = sample_negatives(
            protein_length=200,
            positives=positives,
            neg_ratio=4,
            hard_negative_fraction=0.5,
            hard_neg_offset_range=20,
            rng=rng,
            strict=False,
            disrupted_spans=disrupted,
            near_positive_cfg=np_cfg,
        )
        # Find the disrupted span in the result.
        found = False
        for n, rel, w in zip(negs, negs._relations, negs._weights):
            if (n["start_0b"], n["end_0b"]) == (80, 95):
                assert rel == SpanRelation.COUNTERFACTUAL_DISRUPTED
                assert w == 1.0
                found = True
                break
        assert found, "Disrupted span should survive sampling and be tagged"

    def test_disrupted_with_exact_overlap_to_positive_is_kept(self):
        """Review fix 5: PLAN HIMP1 #3 mandates counterfactual_disrupted
        spans survive even when their coordinates exactly match a remaining
        WT positive. The legacy `if span not in positive_set` filter dropped
        these — this test pins down the new behavior.
        """
        positives = [_pos(50)]  # WT positive at [50, 65)
        # Disrupted span at the SAME coords — would have been silently
        # filtered by the old positive_set safety check.
        disrupted = [{"start_0b": 50, "end_0b": 65, "pep_len": 15}]
        rng = np.random.RandomState(0)
        np_cfg = {
            "near_gap_max": 10,
            "schedule": "ignore",
            "schedule_params": {},
            "metric": "endpoint_gap",
        }
        negs = sample_negatives(
            protein_length=200,
            positives=positives,
            neg_ratio=4,
            hard_negative_fraction=0.5,
            hard_neg_offset_range=20,
            rng=rng,
            strict=False,
            disrupted_spans=disrupted,
            near_positive_cfg=np_cfg,
        )
        # The exact-overlap disrupted span MUST appear in the result with
        # COUNTERFACTUAL_DISRUPTED relation and full weight.
        match_found = False
        for n, rel, w in zip(negs, negs._relations, negs._weights):
            if (n["start_0b"], n["end_0b"]) == (50, 65):
                assert rel == SpanRelation.COUNTERFACTUAL_DISRUPTED
                assert w == 1.0
                match_found = True
                break
        assert match_found, (
            "Exact-overlap disrupted span must be preserved as a "
            "counterfactual negative, not filtered by positive_set."
        )

    def test_ignore_schedule_zeros_adjacent_negatives(self):
        positives = [_pos(50)]
        rng = np.random.RandomState(0)
        np_cfg = {
            "near_gap_max": 30,  # large so most offset-based hard negs are adjacent
            "schedule": "ignore",
            "schedule_params": {},
            "metric": "endpoint_gap",
        }
        negs = sample_negatives(
            protein_length=200,
            positives=positives,
            neg_ratio=4,
            hard_negative_fraction=1.0,  # all hard
            hard_neg_offset_range=10,    # offsets small → spans adjacent or overlap
            rng=rng,
            strict=False,
            near_positive_cfg=np_cfg,
        )
        # With ignore schedule, adjacent_or_near and overlap_nonexact both → 0.
        for rel, w in zip(negs._relations, negs._weights):
            if rel in (SpanRelation.ADJACENT_OR_NEAR, SpanRelation.OVERLAP_NONEXACT):
                assert w == 0.0, f"ignore schedule should zero rel={rel}, got w={w}"
            elif rel == SpanRelation.FAR_DECOY:
                assert w == 1.0
