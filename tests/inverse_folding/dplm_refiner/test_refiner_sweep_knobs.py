"""Tests for apply_steps gating + residual fusion + config validation.

Covers the inference-time knobs introduced for the refiner sweep:
- ``DPLMRefinerConfig`` accepts the three new fields and rejects invalid
  values.
- ``should_apply_refiner`` returns the expected step matrix.
- ``residual_fuse_logits`` matches its mathematical definition and
  leaves invalid positions unchanged.
- ``DPLMRefinerLogitProcessor`` skips refiner forward on gated steps
  (returns input logits) and runs it on non-gated steps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from inverse_folding.dplm_refiner.config import (
    APPLY_STEPS_MODES,
    DPLMRefinerConfig,
    FUSION_MODES,
    should_apply_refiner,
)
from inverse_folding.dplm_refiner.fusion import (
    fuse_logits_by_entropy,
    residual_fuse_logits,
)
from inverse_folding.dplm_refiner.logit_processor import (
    DPLMRefinerLogitProcessor,
)


class _FakeAlphabet:
    padding_idx = 1
    cls_idx = 0
    eos_idx = 2
    mask_idx = 32
    unk_idx = 3
    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self) -> None:
        self._tok_to_idx = {aa: i + 4 for i, aa in enumerate(self.AA_ORDER)}

    def get_idx(self, tok: str) -> int:
        return self._tok_to_idx[tok]

    def get_tok(self, idx: int) -> str:
        for tok, i in self._tok_to_idx.items():
            if i == idx:
                return tok
        return "<unk>"


class _UniformRefiner(nn.Module):
    """Returns uniform-zero logits over 20 AAs; used so we can detect
    whether the refiner forward was invoked (call counter)."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(
        self,
        *,
        x_aa: torch.Tensor,
        x_pos: torch.Tensor,
        x_aa_mask: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.call_count += 1
        B, L, _ = x_aa.shape
        return torch.zeros(B, L, 20, device=x_aa.device)


# ── DPLMRefinerConfig validation ─────────────────────────────────────────


def test_config_accepts_new_fields_with_defaults():
    cfg = DPLMRefinerConfig()
    cfg.validate()
    assert cfg.apply_steps == "all"
    assert cfg.fusion_mode == "entropy"
    assert cfg.fusion_alpha == pytest.approx(0.25)


def test_config_rejects_bad_apply_steps():
    cfg = DPLMRefinerConfig(apply_steps="every_other")
    with pytest.raises(ValueError, match="apply_steps"):
        cfg.validate()


def test_config_rejects_bad_fusion_mode():
    cfg = DPLMRefinerConfig(fusion_mode="bayesian")
    with pytest.raises(ValueError, match="fusion_mode"):
        cfg.validate()


def test_config_rejects_alpha_out_of_range():
    cfg = DPLMRefinerConfig(fusion_alpha=-0.1)
    with pytest.raises(ValueError, match="fusion_alpha"):
        cfg.validate()
    cfg = DPLMRefinerConfig(fusion_alpha=1.2)
    with pytest.raises(ValueError, match="fusion_alpha"):
        cfg.validate()


def test_apply_steps_modes_constant_is_consistent():
    assert "all" in APPLY_STEPS_MODES
    assert "last_2" in APPLY_STEPS_MODES
    assert "final_only" in APPLY_STEPS_MODES


def test_fusion_modes_constant_is_consistent():
    assert set(FUSION_MODES) == {"entropy", "residual"}


# ── should_apply_refiner step matrix ─────────────────────────────────────


def test_should_apply_all_returns_true_for_every_step():
    for step in range(1, 11):
        assert should_apply_refiner(step, 10, "all") is True


def test_should_apply_final_only_fires_only_at_max_step():
    for step in range(1, 10):
        assert should_apply_refiner(step, 10, "final_only") is False
    assert should_apply_refiner(10, 10, "final_only") is True


def test_should_apply_last_2_fires_only_for_two_last_steps():
    for step in range(1, 9):
        assert should_apply_refiner(step, 10, "last_2") is False
    assert should_apply_refiner(9, 10, "last_2") is True
    assert should_apply_refiner(10, 10, "last_2") is True


def test_should_apply_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown apply_steps mode"):
        should_apply_refiner(1, 10, "weekly")


# ── residual_fuse_logits ─────────────────────────────────────────────────


def test_residual_fuse_logits_zero_alpha_is_identity():
    base = torch.randn(2, 5, 7)
    refiner = torch.randn(2, 5, 7)
    valid = torch.ones(2, 5, dtype=torch.bool)
    out = residual_fuse_logits(base, refiner, valid_mask=valid, alpha=0.0)
    assert torch.allclose(out, base)


def test_residual_fuse_logits_one_alpha_is_refiner():
    base = torch.randn(2, 5, 7)
    refiner = torch.randn(2, 5, 7)
    valid = torch.ones(2, 5, dtype=torch.bool)
    out = residual_fuse_logits(base, refiner, valid_mask=valid, alpha=1.0)
    assert torch.allclose(out, refiner)


def test_residual_fuse_logits_invalid_positions_kept_as_base():
    base = torch.randn(1, 4, 5)
    refiner = torch.randn(1, 4, 5)
    valid = torch.tensor([[True, True, False, False]])
    out = residual_fuse_logits(base, refiner, valid_mask=valid, alpha=0.5)
    assert torch.allclose(out[0, 2:], base[0, 2:])
    expected_01 = 0.5 * base[0, :2] + 0.5 * refiner[0, :2]
    assert torch.allclose(out[0, :2], expected_01)


def test_residual_fuse_logits_rejects_alpha_out_of_range():
    base = torch.randn(1, 3, 5)
    refiner = torch.randn(1, 3, 5)
    valid = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="alpha"):
        residual_fuse_logits(base, refiner, valid_mask=valid, alpha=1.5)


def test_residual_fuse_logits_rejects_shape_mismatch():
    base = torch.randn(1, 3, 5)
    refiner = torch.randn(1, 3, 6)
    valid = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError):
        residual_fuse_logits(base, refiner, valid_mask=valid, alpha=0.5)


def test_residual_fusion_is_strictly_more_conservative_for_overconfident_refiner():
    """When the refiner is overconfident, entropy fusion gives it more
    weight (its entropy is lower so the softmax-over-neg-entropies puts
    more mass on it). Residual fusion only blends by alpha, so the
    base-refiner distance shrinks with alpha."""
    torch.manual_seed(0)
    L = 5
    base = torch.zeros(1, L, 20)
    # Overconfident refiner: very peaky.
    refiner = torch.zeros(1, L, 20)
    refiner[0, :, 0] = 50.0
    valid = torch.ones(1, L, dtype=torch.bool)

    entropy_out = fuse_logits_by_entropy(
        base, refiner, valid_mask=valid, temperature=1.0
    )
    residual_out = residual_fuse_logits(
        base, refiner, valid_mask=valid, alpha=0.25
    )

    # Distance of entropy fusion from base
    d_entropy = (entropy_out - base).abs().sum().item()
    d_residual = (residual_out - base).abs().sum().item()
    assert d_residual < d_entropy


# ── DPLMRefinerLogitProcessor gating ─────────────────────────────────────


def _build_batch_and_logits(*, B=1, L=6, V=33):
    alphabet = _FakeAlphabet()
    tokens = torch.full((B, L), alphabet.padding_idx, dtype=torch.long)
    native = ["ACDEFG"]
    for b, seq in enumerate(native[:B]):
        for i, aa in enumerate(seq):
            tokens[b, i] = alphabet.get_idx(aa)
    coords = torch.randn(B, L, 4, 3)
    coord_mask = torch.zeros(B, L, dtype=torch.bool)
    coord_mask[:, :L] = True
    batch = {"tokens": tokens, "coords": coords, "coord_mask": coord_mask}
    logits = torch.full((B, L, V), -0.1)
    logits[..., alphabet.get_idx("A")] = 3.0
    return alphabet, batch, logits, tokens


def test_processor_passthrough_when_gated_off():
    alphabet, batch, logits, tokens = _build_batch_and_logits()
    refiner = _UniformRefiner()
    sink_records: list[dict] = []
    config = DPLMRefinerConfig(
        enabled=True,
        mask_ratio_center=0.5,
        mask_ratio_deviation=0.0,
        apply_steps="final_only",
    )
    proc = DPLMRefinerLogitProcessor(
        refiner=refiner,
        alphabet=alphabet,
        config=config,
        diagnostics_sink=sink_records.append,
    )
    # step=3 of max_step=10 -> final_only is gated off
    out = proc(
        logits=logits.clone(),
        output_tokens=tokens.clone(),
        batch=batch,
        step=3,
        max_step=10,
    )
    assert torch.equal(out, logits), "gated step must return logits unchanged"
    assert refiner.call_count == 0, "refiner forward must not run on gated step"
    assert len(sink_records) == 1
    assert sink_records[0]["per_row"][0]["gated"] is True


def test_processor_runs_refiner_when_gate_open():
    alphabet, batch, logits, tokens = _build_batch_and_logits()
    refiner = _UniformRefiner()
    sink_records: list[dict] = []
    config = DPLMRefinerConfig(
        enabled=True,
        mask_ratio_center=0.5,
        mask_ratio_deviation=0.0,
        apply_steps="final_only",
    )
    proc = DPLMRefinerLogitProcessor(
        refiner=refiner,
        alphabet=alphabet,
        config=config,
        diagnostics_sink=sink_records.append,
    )
    out = proc(
        logits=logits.clone(),
        output_tokens=tokens.clone(),
        batch=batch,
        step=10,
        max_step=10,
    )
    assert refiner.call_count == 1
    assert not sink_records[0]["per_row"][0].get("gated", False)


def test_processor_dispatches_residual_fusion_when_configured():
    """Residual fusion with very small alpha should produce logits whose
    AA logits at residue positions are nearly equal to the base AA logits
    (the refiner was uniform-zero, so residual mixes toward zero)."""
    alphabet, batch, logits, tokens = _build_batch_and_logits()
    refiner = _UniformRefiner()
    config = DPLMRefinerConfig(
        enabled=True,
        mask_ratio_center=0.5,
        mask_ratio_deviation=0.0,
        apply_steps="all",
        fusion_mode="residual",
        fusion_alpha=0.1,
    )
    proc = DPLMRefinerLogitProcessor(
        refiner=refiner, alphabet=alphabet, config=config
    )
    out = proc(
        logits=logits.clone(),
        output_tokens=tokens.clone(),
        batch=batch,
        step=1,
        max_step=10,
    )
    assert refiner.call_count == 1
    # At selected residue positions, A-logit shifts toward 0 (refiner's
    # uniform-zero), but only by alpha. Original A-logit was 3.0; after
    # residual fusion at selected positions it should be (1-0.1)*3.0 = 2.7.
    a_idx = alphabet.get_idx("A")
    out_a = out[..., a_idx]
    # Some positions are selected (A logits attenuated toward 2.7), some
    # are not selected (still 3.0). Both should be in [2.7, 3.0] band.
    residues = out_a[..., :6]  # all coord-valid
    assert (residues >= 2.7 - 1e-4).all()
    assert (residues <= 3.0 + 1e-4).all()
