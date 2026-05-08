"""IPA-derived geometry sidecar.

The sidecar consumes the same DPLM ``coords`` + ``coord_mask`` + tokens
the GVP encoder sees, runs a small IPA trunk identical to the refiner,
and returns ``[B, L, output_dim]`` features projected to the DPLM
adapter dimension. It does NOT predict AA logits; it only injects
geometric context into the encoder feature stream.

Integration is opt-in via ``SidecarAttachedEncoder`` (encoder_wrapper.py)
or by a Module K wrapper. Default DPLM behavior is unchanged when the
sidecar is absent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from inverse_folding.dplm_refiner.geometry import dplm_coords_to_ipa_positions
from inverse_folding.dplm_refiner.ipa.ipa_attn import (
    EdgeTransition,
    InvariantPointAttention,
    StructureModuleTransition,
)
from inverse_folding.dplm_refiner.ipa.ipa_utils import LayerNorm
from inverse_folding.dplm_refiner.ipa.refiner import (
    _EdgePairEncoder,
    _IPATrunk,
    _NodeMaskEncoder,
)
from inverse_folding.dplm_refiner.ipa.rigid_utils import Rigid


class DPLMGeometrySidecar(nn.Module):
    """IPA trunk that emits projected hidden states for the GVP encoder.

    Inputs
    ------
    coords          : ``[B, L, 4, 3]`` DPLM-order ``[N, CA, C, O]``
    coord_mask      : ``[B, L]`` boolean
    special_sym_mask: ``[B, L]`` boolean (pad/BOS/EOS)
    tokens          : ``[B, L]`` long (currently unused; reserved for
                      future AA-aware variants)

    Returns
    -------
    ``[B, L, output_dim]`` features. Invalid (non-residue) positions
    are zeroed.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        ipa_pairwise_dim: int = 128,
        ipa_heads: int = 4,
        ipa_depth: int = 6,
        ipa_qk_points: int = 4,
        ipa_v_points: int = 8,
        output_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.node_encoder = _NodeMaskEncoder(hidden_dim)
        self.edge_pair_encoder = _EdgePairEncoder(hidden_dim)
        self.s_dropout = nn.Dropout(dropout)
        self.z_dropout = nn.Dropout(dropout)
        self.ipa = _IPATrunk(
            ipa_dim=hidden_dim,
            ipa_pairwise_dim=ipa_pairwise_dim,
            ipa_heads=ipa_heads,
            ipa_depth=ipa_depth,
            ipa_qk_points=ipa_qk_points,
            ipa_v_points=ipa_v_points,
            dropout_rate=dropout,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        *,
        coords: torch.Tensor,
        coord_mask: torch.Tensor,
        special_sym_mask: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        del tokens  # Reserved; the first sidecar uses neutral AA features.
        if coords.dim() != 4 or coords.shape[-2:] != (4, 3):
            raise ValueError(
                f"coords must be [B,L,4,3]; got {tuple(coords.shape)}"
            )
        ipa_pos = dplm_coords_to_ipa_positions(
            coords=coords,
            coord_mask=coord_mask,
            special_sym_mask=special_sym_mask,
        )
        seq_mask = ipa_pos.seq_mask
        atom_pos = ipa_pos.atom_pos

        B, L = coord_mask.shape
        # Neutral all-zero AA one-hot — sidecar must not depend on AA
        # identity for the first version (keeps it backbone-only).
        x_aa = torch.zeros(
            (B, L, 20), device=coords.device, dtype=coords.dtype
        )
        # Treat all valid residues as "masked" inputs to the IPA so the
        # trunk leans on geometric signal exclusively.
        x_aa_mask = seq_mask.long()

        r = Rigid.from_3_points(
            atom_pos[:, :, 0], atom_pos[:, :, 1], atom_pos[:, :, 2]
        )
        s = self.node_encoder(x_aa, atom_pos, x_aa_mask, seq_mask)
        z = self.edge_pair_encoder(atom_pos)
        s = self.s_dropout(s)
        z = self.z_dropout(z)
        s = self.ipa(s, z, r, seq_mask.long(), attn_drop_rate=0.0)
        feats = self.proj(s)
        feats = torch.where(
            seq_mask.unsqueeze(-1), feats, torch.zeros_like(feats)
        )
        return feats
