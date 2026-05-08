import torch

from inverse_folding.dplm_refiner.fusion import fuse_logits_by_entropy


def test_fusion_prefers_lower_entropy_logits():
    base_logits = torch.tensor([[[4.0, 0.0], [0.0, 0.0]]])
    refiner_logits = torch.tensor([[[0.0, 0.0], [5.0, -5.0]]])
    valid = torch.tensor([[True, True]])

    fused = fuse_logits_by_entropy(base_logits, refiner_logits, valid_mask=valid, temperature=1.0)

    assert fused.shape == base_logits.shape
    assert fused[0, 0, 0] > fused[0, 0, 1]
    assert fused[0, 1, 0] > fused[0, 1, 1]


def test_fusion_leaves_invalid_positions_at_base_logits():
    base_logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    refiner_logits = torch.tensor([[[9.0, 9.0], [8.0, 8.0]]])
    valid = torch.tensor([[True, False]])

    fused = fuse_logits_by_entropy(base_logits, refiner_logits, valid_mask=valid)

    assert torch.allclose(fused[0, 1], base_logits[0, 1])
