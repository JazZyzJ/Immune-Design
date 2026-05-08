"""DPLM coords → IPA atom-position bridge.

DPLM emits ``coords`` with shape ``[B, L, 4, 3]`` in atom order
``(N, CA, C, O)``; the IPA refiner expects ``[B, L, 5, 3]`` in order
``(N, CA, C, CB, O)`` with a virtual CB. CB construction follows
MapDiff's ``place_missing_cb`` (``MapDiff/utils.py:188-196``), which
calls ``place_fourth_atom(C, N, CA, 1.522, 1.927, -2.143)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Named module-level constants per PLAN_IF_IMP.md Task 2 Step 3 instruction:
# do not inline these literals at the call site.
CB_BOND_LENGTH = 1.522
CB_PLANAR_ANGLE = 1.927
CB_DIHEDRAL = -2.143


@dataclass(frozen=True)
class IPAAtomPositions:
    """Container for IPA-ready atom positions and residue mask."""

    atom_pos: torch.Tensor  # [B, L, 5, 3] in [N, CA, C, CB, O]
    seq_mask: torch.Tensor  # [B, L] boolean, true only for real residues


def _place_fourth_atom(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length: float,
    planar: float,
    dihedral: float,
) -> torch.Tensor:
    """Port of ``MapDiff/utils.py::place_fourth_atom`` for scalar params.

    Computes a fourth point given three reference points plus length /
    planar / dihedral angles. Operates on arbitrary leading-batch shapes.
    """
    bc_vec = b - c
    bc_norm = bc_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    bc_vec = bc_vec / bc_norm

    ba_vec = (b - a).expand_as(bc_vec)
    n_vec = torch.cross(ba_vec, bc_vec, dim=-1)
    n_norm = n_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    n_vec = n_vec / n_norm

    m1 = bc_vec
    m2 = torch.cross(n_vec, bc_vec, dim=-1)
    m3 = n_vec

    d1 = length * torch.cos(torch.tensor(planar, dtype=a.dtype, device=a.device))
    d2 = (
        length
        * torch.sin(torch.tensor(planar, dtype=a.dtype, device=a.device))
        * torch.cos(torch.tensor(dihedral, dtype=a.dtype, device=a.device))
    )
    d3 = (
        -length
        * torch.sin(torch.tensor(planar, dtype=a.dtype, device=a.device))
        * torch.sin(torch.tensor(dihedral, dtype=a.dtype, device=a.device))
    )

    return c + m1 * d1 + m2 * d2 + m3 * d3


def place_virtual_cb(
    n: torch.Tensor,
    ca: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Construct a virtual CB given backbone N, CA, C atoms.

    Mirrors ``MapDiff/utils.py::place_missing_cb`` with arguments named
    in DPLM order. Inputs share shape ``[..., 3]``; output has the same
    leading shape.
    """
    if n.shape != ca.shape or ca.shape != c.shape:
        raise ValueError(
            f"place_virtual_cb shape mismatch: n={tuple(n.shape)} "
            f"ca={tuple(ca.shape)} c={tuple(c.shape)}"
        )
    cb = _place_fourth_atom(
        a=c, b=n, c=ca,
        length=CB_BOND_LENGTH,
        planar=CB_PLANAR_ANGLE,
        dihedral=CB_DIHEDRAL,
    )
    cb = torch.where(torch.isnan(cb), torch.zeros_like(cb), cb)
    return cb


def dplm_coords_to_ipa_positions(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    special_sym_mask: torch.Tensor,
) -> IPAAtomPositions:
    """Reorder DPLM ``[B,L,4,3]`` (N,CA,C,O) into IPA ``[B,L,5,3]``
    (N,CA,C,CB,O) with virtual CB and a residue-validity mask.

    ``seq_mask`` is true only where the position has valid coords, is not
    a special token, and all four backbone atoms are finite. Invalid
    positions are zeroed in the output.
    """
    if coords.dim() != 4 or coords.shape[-2:] != (4, 3):
        raise ValueError(
            f"coords must be [B,L,4,3]; got {tuple(coords.shape)}"
        )
    if coord_mask.shape != coords.shape[:2]:
        raise ValueError(
            f"coord_mask shape {tuple(coord_mask.shape)} != coords [:2] "
            f"{tuple(coords.shape[:2])}"
        )
    if special_sym_mask.shape != coords.shape[:2]:
        raise ValueError(
            f"special_sym_mask shape {tuple(special_sym_mask.shape)} != "
            f"coords [:2] {tuple(coords.shape[:2])}"
        )

    finite_mask = torch.isfinite(coords).all(dim=-1).all(dim=-1)
    seq_mask = coord_mask.bool() & (~special_sym_mask.bool()) & finite_mask

    n_atom = coords[..., 0, :]
    ca_atom = coords[..., 1, :]
    c_atom = coords[..., 2, :]
    o_atom = coords[..., 3, :]
    cb_atom = place_virtual_cb(n=n_atom, ca=ca_atom, c=c_atom)

    atom_pos = torch.stack([n_atom, ca_atom, c_atom, cb_atom, o_atom], dim=-2)
    # zero out invalid rows AFTER stacking so seq_mask is computed from
    # the original input.
    atom_pos = torch.where(
        seq_mask.unsqueeze(-1).unsqueeze(-1),
        atom_pos,
        torch.zeros_like(atom_pos),
    )
    # Replace any residual non-finite values (e.g. NaN from the original
    # coords that survived in valid rows due to upstream bugs) with zero.
    atom_pos = torch.where(
        torch.isfinite(atom_pos), atom_pos, torch.zeros_like(atom_pos)
    )
    return IPAAtomPositions(atom_pos=atom_pos, seq_mask=seq_mask)
