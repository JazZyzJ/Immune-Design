import torch

from inverse_folding.dplm_refiner.ipa.refiner import DPLMIPARefiner


def test_ipa_refiner_forward_shape_and_invalid_mask_zero_grad():
    model = DPLMIPARefiner(hidden_dim=32, ipa_pairwise_dim=32, ipa_heads=4, ipa_depth=2, dropout=0.0)
    model.eval()
    x_aa = torch.zeros((2, 5, 20))
    x_aa[..., 0] = 1.0
    x_pos = torch.randn((2, 5, 5, 3))
    x_aa_mask = torch.zeros((2, 5), dtype=torch.long)
    x_aa_mask[:, 1] = 1
    seq_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )

    logits = model(x_aa=x_aa, x_pos=x_pos, x_aa_mask=x_aa_mask, seq_mask=seq_mask)

    assert logits.shape == (2, 5, 20)
    assert torch.isfinite(logits[seq_mask]).all()
    assert torch.allclose(logits[~seq_mask], torch.zeros_like(logits[~seq_mask]), atol=1e-6)


def test_ipa_refiner_rejects_wrong_atom_count():
    model = DPLMIPARefiner(hidden_dim=32, ipa_pairwise_dim=32, ipa_heads=4, ipa_depth=2)
    x_aa = torch.zeros((1, 4, 20))
    x_pos = torch.randn((1, 4, 4, 3))
    x_aa_mask = torch.zeros((1, 4), dtype=torch.long)
    seq_mask = torch.ones((1, 4), dtype=torch.bool)

    try:
        model(x_aa=x_aa, x_pos=x_pos, x_aa_mask=x_aa_mask, seq_mask=seq_mask)
    except ValueError as exc:
        assert "x_pos" in str(exc)
    else:
        raise AssertionError("expected ValueError")
