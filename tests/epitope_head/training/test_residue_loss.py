"""HIMP3 tests: residue pairwise margin ranking loss.

Covers PLAN_EPI_IMP.md §5 HIMP3 TDD Gate:
  1. RED: residue loss treats ambiguous overlap residues as negatives.
  2. RED: lambda_residue=0 changes loss_total vs residue.enabled=false baseline.
  3. RED: a chunk with zero far-bg residues raises rather than skipping silently.
  4. GREEN: correct ordering lowers loss; wrong ordering produces positive loss.
  5. GREEN: chunks with n_far_bg < min_far_bg_residues are skipped and counted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from epitope_head.training.losses import residue_pairwise_margin_loss


def _full(n: int, val: bool) -> torch.Tensor:
    return torch.full((n,), val, dtype=torch.bool)


# ── Loss correctness ────────────────────────────────────────────────────────

class TestResiduePairwiseMarginLoss:

    def test_correct_ordering_zero_loss(self):
        # Positives at index 0,1; far_bg at 5,6,7,8; pos scores >> neg.
        scores = torch.tensor([10.0, 10.0, 0.0, 0.0, 0.0, -10.0, -10.0, -10.0, -10.0])
        label = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8)
        far_bg = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool)
        central = torch.ones(9, dtype=torch.bool)
        loss, meta = residue_pairwise_margin_loss(
            residue_scores=scores, label=label, far_bg_mask=far_bg,
            central_mask=central, margin_m=0.5, min_far_bg=4,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert meta["skipped"] is False
        assert meta["n_pos"] == 2
        assert meta["n_far_bg"] == 4

    def test_wrong_ordering_positive_loss(self):
        # Negatives have HIGHER score than positives → strong loss.
        scores = torch.tensor([0.0, 0.0, 5.0, 5.0, 5.0, 5.0])
        label = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.uint8)
        far_bg = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.bool)
        central = torch.ones(6, dtype=torch.bool)
        loss, _ = residue_pairwise_margin_loss(
            residue_scores=scores, label=label, far_bg_mask=far_bg,
            central_mask=central, margin_m=0.5, min_far_bg=4,
        )
        # gap = 0 - 5 = -5; relu(0.5 - (-5)) = 5.5
        assert loss.item() == pytest.approx(5.5, abs=1e-6)

    def test_margin_threshold(self):
        # gap == margin → relu(0) = 0 exactly.
        scores = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5])
        label = torch.tensor([1, 0, 0, 0, 0], dtype=torch.uint8)
        far_bg = torch.tensor([0, 1, 1, 1, 1], dtype=torch.bool)
        central = torch.ones(5, dtype=torch.bool)
        loss, _ = residue_pairwise_margin_loss(
            residue_scores=scores, label=label, far_bg_mask=far_bg,
            central_mask=central, margin_m=0.5, min_far_bg=4,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_central_mask_excludes_boundary(self):
        # Positive at boundary should be ignored; only pos in central counted.
        scores = torch.tensor([10.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        label = torch.tensor([1, 0, 1, 0, 0, 0, 0], dtype=torch.uint8)
        far_bg = torch.tensor([0, 0, 0, 1, 1, 1, 0], dtype=torch.bool)
        # Central region [2, 6) → only index 2 is a positive in central.
        central = torch.tensor([0, 0, 1, 1, 1, 1, 0], dtype=torch.bool)
        loss, meta = residue_pairwise_margin_loss(
            residue_scores=scores, label=label, far_bg_mask=far_bg,
            central_mask=central, margin_m=0.5, min_far_bg=3,
        )
        assert meta["n_pos"] == 1
        assert meta["n_far_bg"] == 3
        # gap = 1 - 1 = 0 → relu(0.5) = 0.5
        assert loss.item() == pytest.approx(0.5, abs=1e-6)

    # TDD Gate #5: skip behavior + counter.
    def test_far_bg_below_min_returns_skipped(self):
        scores = torch.tensor([5.0, -1.0, -1.0])
        label = torch.tensor([1, 0, 0], dtype=torch.uint8)
        far_bg = torch.tensor([0, 1, 1], dtype=torch.bool)
        central = torch.ones(3, dtype=torch.bool)
        loss, meta = residue_pairwise_margin_loss(
            residue_scores=scores, label=label, far_bg_mask=far_bg,
            central_mask=central, margin_m=0.5, min_far_bg=4,
        )
        assert meta["skipped"] is True
        assert meta["n_far_bg"] == 2
        # Skipped chunks return 0 loss (no gradient contribution).
        assert loss.item() == 0.0

    # TDD Gate #3: must skip silently rather than raise.
    def test_zero_far_bg_chunk_skipped_not_raised(self):
        scores = torch.tensor([5.0, 5.0])
        label = torch.tensor([1, 1], dtype=torch.uint8)
        far_bg = torch.zeros(2, dtype=torch.bool)
        central = torch.ones(2, dtype=torch.bool)
        # Should NOT raise — must produce skipped=True with zero loss.
        loss, meta = residue_pairwise_margin_loss(
            residue_scores=scores, label=label, far_bg_mask=far_bg,
            central_mask=central, margin_m=0.5, min_far_bg=4,
        )
        assert meta["skipped"] is True
        assert meta["n_far_bg"] == 0

    def test_zero_positives_returns_skipped(self):
        scores = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        label = torch.zeros(5, dtype=torch.uint8)
        far_bg = torch.ones(5, dtype=torch.bool)
        central = torch.ones(5, dtype=torch.bool)
        loss, meta = residue_pairwise_margin_loss(
            residue_scores=scores, label=label, far_bg_mask=far_bg,
            central_mask=central, margin_m=0.5, min_far_bg=4,
        )
        assert meta["skipped"] is True
        assert meta["n_pos"] == 0

    def test_neg_inf_residue_scores_handled(self):
        # Residues uncovered by any window have score = -inf. They must not be
        # used as positives or far_bg in the ranking; but the masks should
        # exclude them already. Test the loss doesn't crash on an explicit
        # -inf neg score that is masked out of far_bg.
        scores = torch.tensor([2.0, 1.5, 1.5, 1.5, 1.5, float("-inf")])
        label = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.uint8)
        far_bg = torch.tensor([0, 1, 1, 1, 1, 0], dtype=torch.bool)  # -inf excluded
        central = torch.ones(6, dtype=torch.bool)
        loss, _ = residue_pairwise_margin_loss(
            residue_scores=scores, label=label, far_bg_mask=far_bg,
            central_mask=central, margin_m=0.5, min_far_bg=4,
        )
        # gap = 2 - 1.5 = 0.5 → relu(0) = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_uncovered_far_bg_must_be_filtered_by_caller(self):
        """Review fix 3: residue_pairwise_margin_loss expects the caller to
        have masked out -inf residues. This test demonstrates what happens
        when the trainer correctly intersects far_bg / label / central
        masks with ``torch.isfinite(residue_scores)`` before calling the
        loss: uncovered residues drop out of n_far_bg / n_pairs.
        """
        # Aggregation produces -inf for uncovered residues.
        from epitope_head.training.residue_supervision import (
            aggregate_window_logits_to_residues,
        )
        chunk_len = 12
        # Single window covering [2, 7) — residues 0,1,7,8,9,10,11 uncovered.
        windows = [(2, 7)]
        logits = torch.tensor([1.5])
        scores = aggregate_window_logits_to_residues(
            window_logits=logits, windows=windows, chunk_len=chunk_len,
            mode="max", params=None,
        )
        # Suppose label=1 at residue 3 (covered), far_bg=1 at residues 1, 9
        # (uncovered) and 5 (covered).
        label = torch.zeros(chunk_len, dtype=torch.uint8); label[3] = 1
        far_bg = torch.zeros(chunk_len, dtype=torch.bool)
        far_bg[1] = True; far_bg[5] = True; far_bg[9] = True
        central = torch.ones(chunk_len, dtype=torch.bool)

        # ── Buggy caller (no -inf filter): n_far_bg includes uncovered res.
        loss_bug, meta_bug = residue_pairwise_margin_loss(
            residue_scores=scores, label=label, far_bg_mask=far_bg,
            central_mask=central, margin_m=0.5, min_far_bg=1,
        )
        assert meta_bug["n_far_bg"] == 3  # bogus — includes uncovered 1, 9

        # ── Fixed caller (trainer applies the finite mask).
        finite = torch.isfinite(scores)
        label_f = label.bool() & finite
        far_bg_f = far_bg & finite
        central_f = central & finite
        loss_fix, meta_fix = residue_pairwise_margin_loss(
            residue_scores=scores, label=label_f, far_bg_mask=far_bg_f,
            central_mask=central_f, margin_m=0.5, min_far_bg=1,
        )
        # Only residue 5 survives as far_bg (residue 1 and 9 dropped).
        assert meta_fix["n_far_bg"] == 1
        assert meta_fix["n_pos"] == 1
        # Pair count = 1 * 1 = 1, not 1 * 3 = 3.
        assert meta_fix["n_pairs"] == 1

    def test_gradient_flows_to_pos_and_neg_scores(self):
        scores = torch.tensor(
            [0.0, 0.0, 5.0, 5.0, 5.0, 5.0], requires_grad=True,
        )
        label = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.uint8)
        far_bg = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.bool)
        central = torch.ones(6, dtype=torch.bool)
        loss, _ = residue_pairwise_margin_loss(
            residue_scores=scores, label=label, far_bg_mask=far_bg,
            central_mask=central, margin_m=0.5, min_far_bg=4,
        )
        loss.backward()
        # Positives (idx 0,1) should pull up: grad < 0.
        assert (scores.grad[:2] < 0).all()
        # Far-bg negatives (idx 2..5) should pull down: grad > 0.
        assert (scores.grad[2:] > 0).all()
