import torch

from inverse_folding.dplm_refiner.config import DPLMRefinerConfig
from inverse_folding.dplm_refiner.logit_processor import DPLMRefinerLogitProcessor
from inverse_folding.dplm_refiner.tokens import CANONICAL_AA_ORDER


class DummyAlphabet:
    padding_idx = 1
    cls_idx = 0
    eos_idx = 2
    mask_idx = 32
    unk_idx = 3

    def __init__(self):
        self._tok_to_idx = {aa: i + 4 for i, aa in enumerate(CANONICAL_AA_ORDER)}
        self._idx_to_tok = {idx: tok for tok, idx in self._tok_to_idx.items()}

    def get_idx(self, token):
        return self._tok_to_idx[token]

    def get_tok(self, idx):
        return self._idx_to_tok[idx]


class ConstantRefiner(torch.nn.Module):
    def forward(self, *, x_aa, x_pos, x_aa_mask, seq_mask):
        logits = torch.zeros((*x_aa.shape[:2], 20), device=x_aa.device)
        logits[..., 0] = 10.0
        logits[~seq_mask] = 0.0
        return logits


def test_logit_processor_emits_diagnostics_when_sink_provided():
    alphabet = DummyAlphabet()
    captured: list[dict] = []
    processor = DPLMRefinerLogitProcessor(
        refiner=ConstantRefiner(),
        alphabet=alphabet,
        config=DPLMRefinerConfig(mask_ratio_center=0.5, mask_ratio_deviation=0.0, fusion_temperature=1.0),
        diagnostics_sink=lambda rec: captured.append(rec),
    )
    logits = torch.zeros((1, 5, 40))
    for tid in (alphabet.padding_idx, alphabet.cls_idx, alphabet.eos_idx, alphabet.mask_idx, alphabet.unk_idx):
        logits[..., tid] = float("-inf")
    tokens = torch.full((1, 5), alphabet.mask_idx)
    tokens[0, 0] = alphabet.cls_idx
    tokens[0, 4] = alphabet.eos_idx
    coords = torch.randn((1, 5, 4, 3))
    coord_mask = torch.tensor([[False, True, True, True, False]])
    batch = {"coords": coords, "coord_mask": coord_mask, "tokens": tokens}

    processor(logits=logits, output_tokens=tokens, batch=batch, step=3, max_step=10)

    assert len(captured) == 1
    rec = captured[0]
    assert rec["step"] == 3
    assert rec["max_step"] == 10
    assert "per_row" in rec
    assert len(rec["per_row"]) == 1
    row = rec["per_row"][0]
    for key in (
        "row_idx",
        "n_residues",
        "n_selected",
        "base_entropy_mean",
        "fused_entropy_mean",
        "refiner_entropy_mean",
        "base_entropy_q90",
        "fused_entropy_q90",
        "mask_ratio",
    ):
        assert key in row, f"missing per-row diagnostic key: {key}"
    assert row["row_idx"] == 0
    assert row["n_residues"] >= 1
    assert row["probe_only"] is False


def test_logit_processor_only_changes_valid_high_entropy_residue_positions():
    alphabet = DummyAlphabet()
    processor = DPLMRefinerLogitProcessor(
        refiner=ConstantRefiner(),
        alphabet=alphabet,
        config=DPLMRefinerConfig(mask_ratio_center=0.5, mask_ratio_deviation=0.0, fusion_temperature=1.0),
    )
    logits = torch.zeros((1, 5, 40))
    # DPLM's forward_decoder bans special tokens to -inf BEFORE calling the
    # processor (dplm_invfold.py:213-217). Reproduce that here so the
    # invalid-position passthrough contract is faithful to production.
    for tid in (alphabet.padding_idx, alphabet.cls_idx, alphabet.eos_idx, alphabet.mask_idx, alphabet.unk_idx):
        logits[..., tid] = float("-inf")
    tokens = torch.full((1, 5), alphabet.mask_idx)
    tokens[0, 0] = alphabet.cls_idx
    tokens[0, 4] = alphabet.eos_idx
    coords = torch.randn((1, 5, 4, 3))
    coord_mask = torch.tensor([[False, True, True, True, False]])
    batch = {"coords": coords, "coord_mask": coord_mask, "tokens": tokens}

    out = processor(logits=logits, output_tokens=tokens, batch=batch, step=1, max_step=10)

    assert out.shape == logits.shape
    assert torch.allclose(out[0, 0], logits[0, 0])
    assert torch.allclose(out[0, 4], logits[0, 4])
    assert out[0, 1, alphabet.get_idx("A")] > out[0, 1, alphabet.get_idx("C")]
    assert torch.isneginf(out[..., alphabet.mask_idx]).all()
