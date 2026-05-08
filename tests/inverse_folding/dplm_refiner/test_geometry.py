import torch

from inverse_folding.dplm_refiner.geometry import (
    dplm_coords_to_ipa_positions,
    place_virtual_cb,
)


def test_place_virtual_cb_returns_finite_cb_for_valid_backbone():
    n = torch.tensor([[0.0, 1.0, 0.0]])
    ca = torch.tensor([[0.0, 0.0, 0.0]])
    c = torch.tensor([[1.5, 0.0, 0.0]])

    cb = place_virtual_cb(n=n, ca=ca, c=c)

    assert cb.shape == (1, 3)
    assert torch.isfinite(cb).all()
    assert torch.linalg.norm(cb - ca, dim=-1).item() > 0.5


def test_dplm_coords_to_ipa_positions_reorders_atoms_and_masks_invalid_rows():
    coords = torch.zeros((1, 3, 4, 3))
    coords[0, :, 0] = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [float("nan"), 0.0, 0.0]])
    coords[0, :, 1] = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    coords[0, :, 2] = torch.tensor([[1.5, 0.0, 0.0], [2.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    coords[0, :, 3] = torch.tensor([[2.0, 0.5, 0.0], [3.0, 0.5, 0.0], [2.0, 0.0, 0.0]])
    coord_mask = torch.tensor([[True, True, False]])
    special_mask = torch.tensor([[False, True, False]])

    out = dplm_coords_to_ipa_positions(coords, coord_mask, special_mask)

    assert out.atom_pos.shape == (1, 3, 5, 3)
    assert out.seq_mask.tolist() == [[True, False, False]]
    assert torch.allclose(out.atom_pos[0, 0, 0], coords[0, 0, 0])
    assert torch.allclose(out.atom_pos[0, 0, 2], coords[0, 0, 2])
    assert torch.allclose(out.atom_pos[0, 0, 4], coords[0, 0, 3])
    assert torch.isfinite(out.atom_pos).all()
    assert out.atom_pos[0, 1].abs().sum().item() == 0.0
