"""Unit tests for DPLMArmDiagnosticProcessor.

Uses a tiny synthetic alphabet + a 1-step deterministic fake refiner so
the tests don't depend on DPLM or MapDiff being importable. Verifies:

- per-step records are emitted on every call (baseline + refiner arms)
- per-position records only fire when refiner is set and sink is wired
- prev_preservation_rate reflects whether DPLM kept or re-masked the
  selected positions between consecutive calls
- init_state_top1_recovery is populated only at step=1
- reset_state() drops cross-call memory
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


from inverse_folding.dplm_refiner.config import DPLMRefinerConfig
from inverse_folding.dplm_refiner.diagnostic_probe import (
    DPLMArmDiagnosticProcessor,
)


class _FakeAlphabet:
    """Tiny alphabet: 20 canonical AAs at ids 4..23, plus pad/cls/eos/mask/unk."""

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


class _FakeRefiner(nn.Module):
    """Deterministic refiner: returns one-hot over (position % 20) for every
    residue. Useful for testing that refiner_top1 is well-defined and
    matches an exact expected value per position."""

    def forward(
        self,
        *,
        x_aa: torch.Tensor,
        x_pos: torch.Tensor,
        x_aa_mask: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = x_aa.shape
        positions = torch.arange(L, device=x_aa.device) % 20
        one_hot = torch.zeros(B, L, 20, device=x_aa.device)
        one_hot[:, :, :] = -1.0
        one_hot.scatter_(-1, positions.view(1, L, 1).expand(B, L, 1), 5.0)
        return one_hot


def _vocab_size() -> int:
    return 33  # max(mask_idx=32) + 1


def _build_batch(
    *,
    B: int = 2,
    L: int = 8,
    native_seq: list[str] | None = None,
    coord_mask_first_n: int = 6,
) -> dict:
    """Build a tiny synthetic DPLM-shaped batch.

    Sets ``batch["tokens"]`` from ``native_seq`` (one per row, padded
    with pad_idx beyond ``coord_mask_first_n``); coord_mask is True
    only for the first ``coord_mask_first_n`` positions.
    """
    alphabet = _FakeAlphabet()
    if native_seq is None:
        native_seq = ["ACDEFGHI"[:L], "VWYACDEF"[:L]]
    tokens = torch.full((B, L), alphabet.padding_idx, dtype=torch.long)
    for b, seq in enumerate(native_seq):
        for i, aa in enumerate(seq):
            tokens[b, i] = alphabet.get_idx(aa)
    coords = torch.randn(B, L, 4, 3)
    coord_mask = torch.zeros(B, L, dtype=torch.bool)
    coord_mask[:, :coord_mask_first_n] = True
    return {"tokens": tokens, "coords": coords, "coord_mask": coord_mask}


def _build_logits_favoring_token(
    *,
    B: int,
    L: int,
    V: int,
    favored_id: int,
    favor_strength: float = 5.0,
) -> torch.Tensor:
    logits = torch.full((B, L, V), -0.1)
    logits[..., favored_id] = favor_strength
    return logits


def test_baseline_probe_emits_per_step_records_no_refiner_path():
    alphabet = _FakeAlphabet()
    step_records: list[dict] = []
    probe = DPLMArmDiagnosticProcessor(
        alphabet=alphabet,
        refiner=None,
        config=None,
        per_step_sink=step_records.append,
    )
    batch = _build_batch()
    B, L = batch["tokens"].shape
    V = _vocab_size()
    logits = _build_logits_favoring_token(
        B=B, L=L, V=V, favored_id=alphabet.get_idx("A")
    )
    output_tokens = batch["tokens"].clone()
    out = probe(
        logits=logits.clone(),
        output_tokens=output_tokens,
        batch=batch,
        step=1,
        max_step=10,
    )
    assert torch.equal(out, logits), "baseline probe must NOT mutate logits"
    assert len(step_records) == B
    for rec in step_records:
        assert rec["n_selected"] is None
        assert rec["base_top1_recovery_all"] is not None
        # Init state recovery is set at step==1
        assert rec["init_state_top1_recovery"] is not None


def test_baseline_probe_init_recovery_only_at_step_1():
    alphabet = _FakeAlphabet()
    step_records: list[dict] = []
    probe = DPLMArmDiagnosticProcessor(
        alphabet=alphabet, per_step_sink=step_records.append
    )
    batch = _build_batch()
    V = _vocab_size()
    B, L = batch["tokens"].shape
    logits = _build_logits_favoring_token(
        B=B, L=L, V=V, favored_id=alphabet.get_idx("A")
    )
    probe(
        logits=logits,
        output_tokens=batch["tokens"].clone(),
        batch=batch,
        step=2,
        max_step=10,
    )
    for rec in step_records:
        assert rec["init_state_top1_recovery"] is None


def test_refiner_probe_emits_selection_and_recovery_fields():
    alphabet = _FakeAlphabet()
    step_records: list[dict] = []
    position_records: list[dict] = []
    refiner = _FakeRefiner()
    config = DPLMRefinerConfig(
        enabled=True,
        mask_ratio_center=0.5,
        mask_ratio_deviation=0.0,
        fusion_temperature=1.0,
        mc_dropout_passes=1,
    )
    probe = DPLMArmDiagnosticProcessor(
        alphabet=alphabet,
        refiner=refiner,
        config=config,
        per_step_sink=step_records.append,
        per_position_sink=position_records.append,
    )

    batch = _build_batch(coord_mask_first_n=6)
    B, L = batch["tokens"].shape
    V = _vocab_size()
    logits = _build_logits_favoring_token(
        B=B, L=L, V=V, favored_id=alphabet.get_idx("A")
    )
    out = probe(
        logits=logits.clone(),
        output_tokens=batch["tokens"].clone(),
        batch=batch,
        step=3,
        max_step=10,
    )

    assert out.shape == (B, L, V)
    assert len(step_records) == B
    for rec in step_records:
        assert rec["n_selected"] is not None
        assert rec["mask_ratio"] is not None
        if rec["n_selected"] > 0:
            assert rec["base_top1_recovery_at_selected"] is not None
            assert rec["refiner_top1_recovery_at_selected"] is not None
            assert rec["fused_top1_recovery_at_selected"] is not None
            assert rec["base_vs_refiner_top1_agree_at_selected"] is not None
            assert rec["fusion_flip_rate_at_selected"] is not None

    # per-position records present (one per selected position)
    total_selected = sum(int(rec["n_selected"] or 0) for rec in step_records)
    assert len(position_records) == total_selected


def test_refiner_probe_preservation_rate_counts_correctly():
    alphabet = _FakeAlphabet()
    step_records: list[dict] = []
    refiner = _FakeRefiner()
    config = DPLMRefinerConfig(
        enabled=True,
        mask_ratio_center=0.5,
        mask_ratio_deviation=0.0,
        fusion_temperature=1.0,
        mc_dropout_passes=1,
    )
    probe = DPLMArmDiagnosticProcessor(
        alphabet=alphabet,
        refiner=refiner,
        config=config,
        per_step_sink=step_records.append,
    )
    batch = _build_batch(coord_mask_first_n=6)
    V = _vocab_size()
    B, L = batch["tokens"].shape
    logits = _build_logits_favoring_token(
        B=B, L=L, V=V, favored_id=alphabet.get_idx("A")
    )

    # Step 1: refiner selects positions, records fused-token AA indices.
    probe(
        logits=logits.clone(),
        output_tokens=batch["tokens"].clone(),
        batch=batch,
        step=1,
        max_step=10,
    )
    step1_records = list(step_records)
    # No preservation rate on the very first step (no prior selection).
    for rec in step1_records:
        assert rec["prev_preservation_rate"] is None
        assert rec["prev_n_selected"] is None

    # Step 2: simulate DPLM keeping ALL previously-selected positions
    # exactly as the refiner had set them (best case → preservation=1.0).
    # We reconstruct what tokens the probe stored by reading back the
    # _prev_selected_state.
    output_tokens_step2 = batch["tokens"].clone()
    for b, prev in probe._prev_selected_state.items():
        positions = prev["positions"]
        fused_aa = prev["fused_tokens"]
        # Map AA index back to DPLM-vocab id and write.
        dplm_tokens = probe.bridge.from_aa_tokens(fused_aa)
        output_tokens_step2[b, positions] = dplm_tokens.long()

    step_records.clear()
    probe(
        logits=logits.clone(),
        output_tokens=output_tokens_step2,
        batch=batch,
        step=2,
        max_step=10,
    )
    for rec in step_records:
        if rec["prev_n_selected"] and rec["prev_n_selected"] > 0:
            assert rec["prev_preservation_rate"] == pytest.approx(1.0)


def test_refiner_probe_preservation_rate_zero_when_all_remasked():
    alphabet = _FakeAlphabet()
    step_records: list[dict] = []
    refiner = _FakeRefiner()
    config = DPLMRefinerConfig(
        enabled=True,
        mask_ratio_center=0.5,
        mask_ratio_deviation=0.0,
        fusion_temperature=1.0,
        mc_dropout_passes=1,
    )
    probe = DPLMArmDiagnosticProcessor(
        alphabet=alphabet,
        refiner=refiner,
        config=config,
        per_step_sink=step_records.append,
    )
    batch = _build_batch(coord_mask_first_n=6)
    V = _vocab_size()
    B, L = batch["tokens"].shape
    logits = _build_logits_favoring_token(
        B=B, L=L, V=V, favored_id=alphabet.get_idx("A")
    )

    probe(
        logits=logits.clone(),
        output_tokens=batch["tokens"].clone(),
        batch=batch,
        step=1,
        max_step=10,
    )

    # Simulate DPLM re-masking ALL previously-selected positions.
    output_tokens_step2 = batch["tokens"].clone()
    for b, prev in probe._prev_selected_state.items():
        positions = prev["positions"]
        output_tokens_step2[b, positions] = alphabet.mask_idx

    step_records.clear()
    probe(
        logits=logits.clone(),
        output_tokens=output_tokens_step2,
        batch=batch,
        step=2,
        max_step=10,
    )
    for rec in step_records:
        if rec["prev_n_selected"] and rec["prev_n_selected"] > 0:
            assert rec["prev_preservation_rate"] == pytest.approx(0.0)


def test_reset_state_clears_prev_selected():
    alphabet = _FakeAlphabet()
    refiner = _FakeRefiner()
    config = DPLMRefinerConfig(
        enabled=True,
        mask_ratio_center=0.5,
        mask_ratio_deviation=0.0,
        fusion_temperature=1.0,
        mc_dropout_passes=1,
    )
    step_records: list[dict] = []
    probe = DPLMArmDiagnosticProcessor(
        alphabet=alphabet,
        refiner=refiner,
        config=config,
        per_step_sink=step_records.append,
    )
    batch = _build_batch(coord_mask_first_n=6)
    V = _vocab_size()
    B, L = batch["tokens"].shape
    logits = _build_logits_favoring_token(
        B=B, L=L, V=V, favored_id=alphabet.get_idx("A")
    )
    probe(
        logits=logits.clone(),
        output_tokens=batch["tokens"].clone(),
        batch=batch,
        step=1,
        max_step=10,
    )
    assert probe._prev_selected_state, "state should be populated after a refiner step"
    probe.reset_state()
    assert probe._prev_selected_state == {}


def test_per_position_sink_requires_refiner():
    alphabet = _FakeAlphabet()
    with pytest.raises(ValueError, match="per_position_sink"):
        DPLMArmDiagnosticProcessor(
            alphabet=alphabet,
            refiner=None,
            per_position_sink=lambda r: None,
        )


def test_refiner_without_config_rejected():
    alphabet = _FakeAlphabet()
    refiner = _FakeRefiner()
    with pytest.raises(ValueError, match="config"):
        DPLMArmDiagnosticProcessor(
            alphabet=alphabet, refiner=refiner, config=None
        )


def test_missing_tokens_in_batch_raises():
    alphabet = _FakeAlphabet()
    probe = DPLMArmDiagnosticProcessor(alphabet=alphabet, per_step_sink=lambda r: None)
    batch = _build_batch()
    del batch["tokens"]
    V = _vocab_size()
    B, L = 2, 8
    logits = _build_logits_favoring_token(B=B, L=L, V=V, favored_id=4)
    with pytest.raises(ValueError, match="tokens"):
        probe(
            logits=logits,
            output_tokens=torch.zeros((B, L), dtype=torch.long),
            batch=batch,
            step=1,
            max_step=10,
        )


def test_init_state_recovery_matches_manual_count():
    alphabet = _FakeAlphabet()
    step_records: list[dict] = []
    probe = DPLMArmDiagnosticProcessor(
        alphabet=alphabet, per_step_sink=step_records.append
    )
    batch = _build_batch(
        B=1, L=6, native_seq=["ACDEFG"], coord_mask_first_n=6
    )
    # output_tokens differ from native at position 0 only -> 5/6 recovery.
    output_tokens = batch["tokens"].clone()
    output_tokens[0, 0] = alphabet.get_idx("V")
    V = _vocab_size()
    logits = _build_logits_favoring_token(
        B=1, L=6, V=V, favored_id=alphabet.get_idx("A")
    )
    probe(
        logits=logits,
        output_tokens=output_tokens,
        batch=batch,
        step=1,
        max_step=10,
    )
    assert step_records[0]["init_state_top1_recovery"] == pytest.approx(5 / 6)
