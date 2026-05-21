"""Dense IPA refinement wrapper for the GeoEGNN-IPA encoder.

Bridges the sparse GeoEGNN backbone (flat ``[N_res, H]`` hidden states
scattered to dense ``[B, L_token, H]``) into a stack of Invariant Point
Attention layers anchored on the *original* N/CA/C rigid frames. When
``use_updated_coord_bias=True``, the EGNN-updated CA coordinates are
projected into an additive pair bias but never replace the rigid
frames; the IPA anchor remains the input backbone geometry per
``PLAN_IF_ENCODER.md`` non-negotiable #7.

Reuses :class:`_IPATrunk` and :class:`_EdgePairEncoder` from
``dplm_refiner.ipa.refiner`` and :class:`Rigid` from
``dplm_refiner.ipa.rigid_utils``; we do not duplicate those modules.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from inverse_folding.dplm_refiner.ipa.refiner import (
    _EdgePairEncoder,
    _IPATrunk,
)
from inverse_folding.dplm_refiner.ipa.rigid_utils import Rigid


class IPADenseRefinement(nn.Module):
    """Dense-token IPA refinement of per-residue hidden states.

    Args:
        in_dim: Input hidden dim (output of EGNN stack).
        ipa_dim: IPA node hidden dim. Input is projected from ``in_dim``
            to ``ipa_dim`` before IPA.
        ipa_pairwise_dim: IPA pair-feature dim.
        ipa_heads / ipa_depth / ipa_qk_points / ipa_v_points: IPA arch.
        dropout: dropout used in the IPA trunk.
        use_updated_coord_bias: when True, ``forward(updated_pos=...)``
            adds an additive pair bias derived from the EGNN-updated CA
            distances. The rigid frames stay tied to the original
            ``(N, CA, C)``.
    """

    def __init__(
        self,
        in_dim: int,
        ipa_dim: int = 128,
        ipa_pairwise_dim: int = 128,
        ipa_heads: int = 4,
        ipa_depth: int = 6,
        ipa_qk_points: int = 4,
        ipa_v_points: int = 8,
        dropout: float = 0.2,
        use_updated_coord_bias: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.ipa_dim = ipa_dim
        self.ipa_pairwise_dim = ipa_pairwise_dim
        self.in_proj = nn.Linear(in_dim, ipa_dim)
        # ``_EdgePairEncoder`` emits pair features of dim ``emb_dim``;
        # the IPA trunk's ``ipa_pairwise_dim`` consumes them, so emb_dim
        # MUST equal ipa_pairwise_dim (not ipa_dim — that was a bug
        # found in the E3 code review).
        self.edge_pair_encoder = _EdgePairEncoder(ipa_pairwise_dim)
        self.ipa = _IPATrunk(
            ipa_dim=ipa_dim,
            ipa_pairwise_dim=ipa_pairwise_dim,
            ipa_heads=ipa_heads,
            ipa_depth=ipa_depth,
            ipa_qk_points=ipa_qk_points,
            ipa_v_points=ipa_v_points,
            dropout_rate=dropout,
        )
        self.use_updated_coord_bias = use_updated_coord_bias
        if use_updated_coord_bias:
            self.coord_bias_proj = nn.Linear(1, ipa_pairwise_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        atom_pos: torch.Tensor,
        seq_mask: torch.Tensor,
        updated_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refine ``hidden`` with IPA on original backbone frames.

        Args:
            hidden: ``[B, L, in_dim]`` per-residue hidden states. Invalid
                positions should already be zeroed by the caller.
            atom_pos: ``[B, L, 5, 3]`` atom positions in (N, CA, C, CB, O).
            seq_mask: ``[B, L]`` boolean. True at real residues only.
            updated_pos: optional ``[B, L, 3]`` updated CA positions
                from the EGNN; only used when
                ``use_updated_coord_bias=True``.

        Returns:
            ``[B, L, ipa_dim]`` refined hidden; rows where ``seq_mask=False``
            are zeroed out.
        """
        if hidden.dim() != 3 or hidden.shape[-1] != self.in_dim:
            raise ValueError(
                f"hidden must be [B, L, {self.in_dim}]; got "
                f"{tuple(hidden.shape)}"
            )
        if (
            atom_pos.dim() != 4
            or atom_pos.shape[-2] != 5
            or atom_pos.shape[-1] != 3
        ):
            raise ValueError(
                f"atom_pos must be [B, L, 5, 3]; got {tuple(atom_pos.shape)}"
            )
        if seq_mask.shape != hidden.shape[:2]:
            raise ValueError(
                f"seq_mask shape {tuple(seq_mask.shape)} != hidden[:2] "
                f"{tuple(hidden.shape[:2])}"
            )

        s = self.in_proj(hidden)
        z = self.edge_pair_encoder(atom_pos)
        if self.use_updated_coord_bias:
            if updated_pos is None:
                raise ValueError(
                    "use_updated_coord_bias=True but updated_pos is None"
                )
            d = torch.cdist(updated_pos, updated_pos).unsqueeze(-1)
            z = z + self.coord_bias_proj(d)

        r = Rigid.from_3_points(
            atom_pos[:, :, 0], atom_pos[:, :, 1], atom_pos[:, :, 2]
        )
        seq_mask_long = seq_mask.long()
        s = self.ipa(s, z, r, seq_mask_long, attn_drop_rate=0.0)
        s = torch.where(seq_mask.unsqueeze(-1), s, torch.zeros_like(s))
        return s
