import torch

from inverse_folding.dplm_refiner.tokens import (
    CANONICAL_AA_ORDER,
    DPLMTokenBridge,
)


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


def test_token_bridge_extracts_canonical_logits_in_stable_order():
    alphabet = DummyAlphabet()
    bridge = DPLMTokenBridge.from_alphabet(alphabet)
    logits = torch.full((1, 2, 40), -100.0)
    for aa_i, token_i in enumerate(bridge.aa_token_ids.tolist()):
        logits[..., token_i] = aa_i

    aa_logits = bridge.to_aa_logits(logits)

    assert aa_logits.shape == (1, 2, 20)
    assert aa_logits[0, 0].tolist() == [float(i) for i in range(20)]


def test_token_bridge_scatter_aa_logits_back_to_dplm_vocab_preserves_special_bans():
    alphabet = DummyAlphabet()
    bridge = DPLMTokenBridge.from_alphabet(alphabet)
    base_logits = torch.zeros((1, 2, 40))
    aa_logits = torch.arange(40, dtype=torch.float32).view(1, 2, 20)

    full = bridge.scatter_aa_logits(aa_logits, base_logits)

    assert full.shape == base_logits.shape
    assert torch.isneginf(full[..., alphabet.padding_idx]).all()
    assert torch.isneginf(full[..., alphabet.cls_idx]).all()
    assert torch.isneginf(full[..., alphabet.eos_idx]).all()
    assert torch.isneginf(full[..., alphabet.mask_idx]).all()
    assert torch.isneginf(full[..., alphabet.unk_idx]).all()
    assert full[0, 1, alphabet.get_idx("Y")].item() == 39.0
