"""DPLM-compatible MapDiff IPA refiner.

Adapted from ``MapDiff/model/ipa/ipa_net.py``. Public API matches
``IPANetPredictor.forward(x_aa, x_pos, x_aa_mask, seq_mask)`` but the
constructor and validation are tightened for the DPLM batch contract.

Source: https://github.com/peizhenbai/MapDiff
Local reference checkout: /Users/jerry/Project/MHC-IF/MapDiff
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from inverse_folding.dplm_refiner.ipa.ipa_attn import (
    EdgeTransition,
    InvariantPointAttention,
    StructureModuleTransition,
)
from inverse_folding.dplm_refiner.ipa.ipa_utils import (
    LayerNorm,
    cal_dihedrals,
    cal_pair_rbf,
    relative_pairwise_position_idx,
)
from inverse_folding.dplm_refiner.ipa.rigid_utils import Rigid


class _NodeMaskEncoder(nn.Module):
    def __init__(self, emb_dim: int, num_d_feat: int = 6, num_aa_types: int = 20):
        super().__init__()
        self.emb_dim = emb_dim
        self.aa_in_proj = nn.Linear(num_aa_types, emb_dim)
        self.d_in_proj = nn.Linear(num_d_feat, emb_dim)
        self.mask_emb = nn.Embedding(1, emb_dim)
        self.pad_emb = nn.Embedding(1, emb_dim, padding_idx=0)
        self.pe_encoding = nn.Parameter(
            self._positional_encoding(max_len=1200), requires_grad=False
        )
        nn.init.xavier_uniform_(self.aa_in_proj.weight.data)
        nn.init.xavier_uniform_(self.d_in_proj.weight.data)

    def _positional_encoding(self, max_len: int = 1200) -> torch.Tensor:
        pe = torch.zeros(max_len, self.emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.emb_dim, 2).float()
            * (-math.log(10000.0) / self.emb_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x_aa, x_pos, node_mask, seq_mask):
        x = self.aa_in_proj(x_aa)
        x = torch.where(
            node_mask.unsqueeze(-1).bool(),
            self.mask_emb.weight.expand_as(x),
            x,
        )
        x = torch.where(
            seq_mask.unsqueeze(-1).bool(),
            x,
            self.pad_emb.weight.expand_as(x),
        )
        d_feat = cal_dihedrals(x_pos).float()
        x_d = self.d_in_proj(d_feat)
        x = x + x_d + self.pe_encoding[:, : x.size(1), :].to(x.device)
        return x


class _EdgePairEncoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        dist_bins: int = 24,
        dist_bin_width: float = 0.5,
        rel_pos_k: int = 32,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.dist_bins = dist_bins
        self.dist_bin_width = dist_bin_width
        self.rel_pos_k = rel_pos_k
        self.rbf_in_proj = nn.Linear(self.dist_bins * 16, emb_dim)
        self.relpos_in_proj = nn.Linear(2 * self.rel_pos_k + 1, emb_dim)
        nn.init.xavier_uniform_(self.rbf_in_proj.weight.data)
        nn.init.xavier_uniform_(self.relpos_in_proj.weight.data)

    def forward(self, x_pos):
        batch_size, seq_len = x_pos.shape[:2]
        x_bb = x_pos[:, :, :4]
        x_bb = rearrange(x_bb, "b n c d -> b (n c) d", c=4)
        pairwise_distance = torch.cdist(x_bb, x_bb)
        pairwise_rbf = cal_pair_rbf(
            pairwise_distance, self.dist_bins, self.dist_bin_width
        )
        pairwise_rbf = rearrange(
            pairwise_rbf,
            "b (n1 c1) (n2 c2) d -> b n1 n2 (c1 c2 d)",
            c1=4,
            c2=4,
        )
        z = self.rbf_in_proj(pairwise_rbf)
        rel_pos = relative_pairwise_position_idx(seq_len, self.rel_pos_k)
        rel_pos = F.one_hot(rel_pos, 2 * self.rel_pos_k + 1).float()
        rel_pos = rel_pos.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(x_pos.device)
        z = z + self.relpos_in_proj(rel_pos)
        return z


class _IPATrunk(nn.Module):
    def __init__(
        self,
        ipa_dim: int = 128,
        ipa_pairwise_dim: int = 128,
        ipa_heads: int = 4,
        ipa_depth: int = 8,
        ipa_qk_points: int = 4,
        ipa_v_points: int = 8,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.ipa_layers = nn.ModuleList()
        for i in range(ipa_depth):
            ipa = InvariantPointAttention(
                ipa_dim,
                ipa_pairwise_dim,
                ipa_dim // ipa_heads,
                ipa_heads,
                ipa_qk_points,
                ipa_v_points,
            )
            ipa_dropout = nn.Dropout(dropout_rate)
            layer_norm_ipa = LayerNorm(ipa_dim)
            pre_transit = nn.Linear(ipa_dim, ipa_dim * 4)
            post_transit = nn.Linear(ipa_dim * 4, ipa_dim)
            transition = StructureModuleTransition(ipa_dim * 4, 1, 0.1)
            edge_transition = (
                None
                if i == ipa_depth - 1
                else EdgeTransition(
                    ipa_dim, ipa_pairwise_dim, ipa_pairwise_dim, num_layers=2
                )
            )
            self.ipa_layers.append(
                nn.ModuleList(
                    [
                        ipa,
                        ipa_dropout,
                        layer_norm_ipa,
                        pre_transit,
                        transition,
                        post_transit,
                        edge_transition,
                    ]
                )
            )

    def forward(self, s, z, r, seq_mask, attn_drop_rate: float = 0.0):
        for (
            ipa,
            ipa_dropout,
            layer_norm_ipa,
            pre_transit,
            transition,
            post_transit,
            edge_transition,
        ) in self.ipa_layers:
            s = s + ipa(s, z, r, seq_mask, attn_drop_rate=attn_drop_rate)
            s = ipa_dropout(s)
            s = layer_norm_ipa(s)
            s = pre_transit(s)
            s = transition(s)
            s = post_transit(s)
            if edge_transition is not None:
                z = edge_transition(s, z)
        return s


class DPLMIPARefiner(nn.Module):
    """MapDiff-style IPA refiner with DPLM-compatible batch contract.

    Inputs
    ------
    x_aa      : ``[B, L, 20]`` AA one-hot/probabilities
    x_pos     : ``[B, L, 5, 3]`` atom positions in ``[N, CA, C, CB, O]`` order
    x_aa_mask : ``[B, L]`` long; positions with value ``1`` get a learned
                mask embedding regardless of ``x_aa``
    seq_mask  : ``[B, L]`` boolean; valid residue mask

    Returns
    -------
    ``[B, L, 20]`` logits, zero on invalid positions.
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
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_encoder = _NodeMaskEncoder(hidden_dim)
        self.edge_pair_encoder = _EdgePairEncoder(hidden_dim)
        self.s_dropout = nn.Dropout(dropout)
        self.z_dropout = nn.Dropout(dropout)
        self.node_predictor = nn.Linear(hidden_dim, 20)
        self.ipa = _IPATrunk(
            ipa_dim=hidden_dim,
            ipa_pairwise_dim=ipa_pairwise_dim,
            ipa_heads=ipa_heads,
            ipa_depth=ipa_depth,
            ipa_qk_points=ipa_qk_points,
            ipa_v_points=ipa_v_points,
            dropout_rate=dropout,
        )

    def forward(
        self,
        *,
        x_aa: torch.Tensor,
        x_pos: torch.Tensor,
        x_aa_mask: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x_aa.dim() != 3 or x_aa.shape[-1] != 20:
            raise ValueError(
                f"x_aa must be [B,L,20]; got {tuple(x_aa.shape)}"
            )
        if (
            x_pos.dim() != 4
            or x_pos.shape[-2] != 5
            or x_pos.shape[-1] != 3
        ):
            raise ValueError(
                f"x_pos must be [B,L,5,3] (atoms in [N,CA,C,CB,O]); got "
                f"{tuple(x_pos.shape)}"
            )
        if x_aa_mask.shape != x_aa.shape[:2]:
            raise ValueError(
                f"x_aa_mask shape {tuple(x_aa_mask.shape)} != x_aa[:2] "
                f"{tuple(x_aa.shape[:2])}"
            )
        if seq_mask.shape != x_aa.shape[:2]:
            raise ValueError(
                f"seq_mask shape {tuple(seq_mask.shape)} != x_aa[:2] "
                f"{tuple(x_aa.shape[:2])}"
            )

        r = Rigid.from_3_points(
            x_pos[:, :, 0], x_pos[:, :, 1], x_pos[:, :, 2]
        )
        s = self.node_encoder(x_aa, x_pos, x_aa_mask, seq_mask)
        z = self.edge_pair_encoder(x_pos)
        s = self.s_dropout(s)
        z = self.z_dropout(z)
        seq_mask_long = seq_mask.long()
        s = self.ipa(s, z, r, seq_mask_long, attn_drop_rate=0.0)
        logits = self.node_predictor(s)
        logits = torch.where(
            seq_mask.unsqueeze(-1), logits, torch.zeros_like(logits)
        )
        return logits
