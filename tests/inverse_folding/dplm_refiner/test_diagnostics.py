import torch

from inverse_folding.dplm_refiner.diagnostics import DPLMBaseEntropyProbe
from inverse_folding.dplm_refiner.tokens import CANONICAL_AA_ORDER


class DummyAlphabet:
    padding_idx = 1
    cls_idx = 0
    eos_idx = 2
    mask_idx = 32
    unk_idx = 3

    def __init__(self):
        self._tok_to_idx = {aa: i + 4 for i, aa in enumerate(CANONICAL_AA_ORDER)}

    def get_idx(self, token):
        return self._tok_to_idx[token]

    def get_tok(self, idx):
        return next(t for t, i in self._tok_to_idx.items() if i == idx)


def test_base_entropy_probe_is_passthrough_and_records_real_entropy():
    captured = []
    probe = DPLMBaseEntropyProbe(
        alphabet=DummyAlphabet(),
        diagnostics_sink=lambda rec: captured.append(rec),
    )
    logits = torch.zeros((1, 4, 40))
    # Make AA col 4 (= "A") sharply preferred at position 0; uniform elsewhere
    logits[0, 0, 4] = 10.0
    tokens = torch.tensor([[10, 11, 12, 13]])  # plain AA tokens, all valid
    coords = torch.randn((1, 4, 4, 3))
    coord_mask = torch.tensor([[True, True, True, True]])
    batch = {"coords": coords, "coord_mask": coord_mask, "tokens": tokens}

    out = probe(logits=logits, output_tokens=tokens, batch=batch, step=2, max_step=10)

    # Passthrough: bit-equivalent to input
    assert torch.equal(out, logits)
    assert len(captured) == 1
    rec = captured[0]
    assert rec["probe_only"] is True
    assert rec["step"] == 2
    assert rec["max_step"] == 10
    assert rec["n_selected"] is None
    assert rec["fused_entropy_mean"] is None
    assert rec["refiner_entropy_mean"] is None
    assert rec["mask_ratio"] is None
    # Position 0 has sharp logits (low entropy); positions 1-3 uniform (high entropy)
    # Mean should be > 0 and finite
    assert rec["base_entropy_mean"] > 0.0
    assert rec["base_entropy_q90"] >= rec["base_entropy_mean"]


def test_base_entropy_probe_requires_sink():
    import pytest

    with pytest.raises(ValueError):
        DPLMBaseEntropyProbe(alphabet=DummyAlphabet(), diagnostics_sink=None)
