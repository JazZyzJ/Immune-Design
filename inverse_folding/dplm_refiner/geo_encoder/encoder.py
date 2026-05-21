"""GeoEGNN-IPA encoder for DPLM-IF.

Drop-in replacement for ``GVPTransformerEncoderWrapper`` that satisfies
the same ``encoder_out`` contract DPLM's adapter consumes:

- ``forward(batch, output_logits=False, **kwargs) -> encoder_out`` dict
  with keys ``{"feats", "coord_mask", "encoder_attention_mask"}``.
- ``forward(batch, output_logits=True) -> (logits, encoder_out)``.

Internal pipeline::

    coords [B, L_token, 4, 3]
        -> dplm_coords_to_ipa_positions -> atom_pos [B, L_token, 5, 3]
        -> build_geo_graph -> PyG Batch (sparse, per-protein subgraphs)
        -> node_in_proj  -> EGNNStack -> flat [N_res, egnn_hidden_dim]
        -> _scatter_to_token_space -> [B, L_token, egnn_hidden_dim]
        -> IPADenseRefinement (original N/CA/C frames) -> [B, L_token, ipa_dim]
        -> out_proj -> [B, L_token, d_model]

The encoder zeros out invalid/special/pad token rows. See
``PLAN_IF_ENCODER.md`` Task E3 for the exact contract.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from inverse_folding.dplm_refiner.geometry import dplm_coords_to_ipa_positions
from inverse_folding.dplm_refiner.geo_encoder.config import GraphConfig
from inverse_folding.dplm_refiner.geo_encoder.egnn import EGNNStack
from inverse_folding.dplm_refiner.geo_encoder.graph import build_geo_graph
from inverse_folding.dplm_refiner.geo_encoder.ipa import IPADenseRefinement


# Structural special-token IDs (ESM-2 33-vocab). NB: ``<mask>`` (id=32)
# is INTENTIONALLY OMITTED — at DPLM denoising step 0 ``prev_tokens`` is
# all-``<mask>``, but those positions are real residues with valid coords
# that just haven't been decoded yet. Including ``<mask>`` in the
# structural mask would drop every residue out of the graph. ``<unk>``
# (id=3) is also a real residue with unknown identity (geometry is
# valid), so we keep it. Caller can override at construction time.
_DEFAULT_SPECIAL_TOKEN_IDS = (0, 1, 2)  # cls, pad, eos


class GeoEGNNIPAEncoder(nn.Module):
    """GeoEGNN backbone + dense IPA refinement, DPLM-compatible.

    The public API mirrors ``GVPTransformerEncoderWrapper``:
    ``forward(batch, output_logits, **kwargs)`` returns either an
    ``encoder_out`` dict or ``(logits, encoder_out)``.

    Notes on the contract:

    - The returned ``encoder_out["feats"]`` is ``[B, L_token, d_model]``
      with rows zeroed where ``seq_mask`` is False (invalid coords,
      special tokens, or padding).
    - ``encoder_out["coord_mask"]`` is the input ``batch["coord_mask"]``;
      ``DPLMInvFold.forward_encoder`` then leaves this in place because
      of the "ensure coord_mask in both branches" guarantee added in
      Task E0.
    - When ``output_logits=True``, the draft head produces ``[B, L_token,
      vocab_size]`` logits computed from the projected feats.
    """

    def __init__(
        self,
        d_model: int = 512,
        output_logits: bool = False,
        vocab_size: int = 33,
        # graph
        graph_config: Optional[GraphConfig] = None,
        # egnn
        egnn_depth: int = 3,
        egnn_hidden_dim: int = 128,
        egnn_message_dim: int = 16,
        update_coors: bool = False,
        update_global: bool = True,
        norm_coors: bool = True,
        egnn_dropout: float = 0.1,
        # ipa
        ipa_depth: int = 6,
        ipa_hidden_dim: int = 128,
        ipa_pairwise_dim: int = 128,
        ipa_heads: int = 4,
        ipa_qk_points: int = 4,
        ipa_v_points: int = 8,
        ipa_dropout: float = 0.2,
        use_updated_coord_bias: bool = False,
        # misc
        special_token_ids: Tuple[int, ...] = _DEFAULT_SPECIAL_TOKEN_IDS,
    ):
        super().__init__()
        graph_config = self._coerce_graph_config(graph_config)
        graph_config.validate()
        self.graph_config = graph_config
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.update_coors = update_coors
        self.special_token_ids = tuple(int(t) for t in special_token_ids)
        # Store all constructor kwargs verbatim so the checkpoint helper
        # can reconstruct an encoder of the same shape — including
        # non-default values for fields that aren't otherwise reachable
        # via attribute introspection (e.g. egnn_message_dim,
        # ipa_pairwise_dim, dropout rates).
        self._constructor_kwargs = {
            "d_model": int(d_model),
            "output_logits": bool(output_logits),
            "vocab_size": int(vocab_size),
            "egnn_depth": int(egnn_depth),
            "egnn_hidden_dim": int(egnn_hidden_dim),
            "egnn_message_dim": int(egnn_message_dim),
            "update_coors": bool(update_coors),
            "update_global": bool(update_global),
            "norm_coors": bool(norm_coors),
            "egnn_dropout": float(egnn_dropout),
            "ipa_depth": int(ipa_depth),
            "ipa_hidden_dim": int(ipa_hidden_dim),
            "ipa_pairwise_dim": int(ipa_pairwise_dim),
            "ipa_heads": int(ipa_heads),
            "ipa_qk_points": int(ipa_qk_points),
            "ipa_v_points": int(ipa_v_points),
            "ipa_dropout": float(ipa_dropout),
            "use_updated_coord_bias": bool(use_updated_coord_bias),
            "special_token_ids": list(self.special_token_ids),
        }

        self.node_in_proj = nn.Linear(
            graph_config.node_feat_dim, egnn_hidden_dim
        )
        self.egnn_stack = EGNNStack(
            n_layers=egnn_depth,
            feats_dim=egnn_hidden_dim,
            edge_attr_dim=graph_config.edge_feat_dim,
            m_dim=egnn_message_dim,
            update_coors=update_coors,
            update_global=update_global,
            norm_coors=norm_coors,
            dropout=egnn_dropout,
        )
        self.ipa = IPADenseRefinement(
            in_dim=egnn_hidden_dim,
            ipa_dim=ipa_hidden_dim,
            ipa_pairwise_dim=ipa_pairwise_dim,
            ipa_heads=ipa_heads,
            ipa_depth=ipa_depth,
            ipa_qk_points=ipa_qk_points,
            ipa_v_points=ipa_v_points,
            dropout=ipa_dropout,
            use_updated_coord_bias=use_updated_coord_bias,
        )
        self.out_proj = nn.Linear(ipa_hidden_dim, d_model)
        if output_logits:
            self.draft_head: Optional[nn.Linear] = nn.Linear(
                d_model, vocab_size
            )
        else:
            self.draft_head = None

    # ── public forward ──────────────────────────────────────────────

    def forward(self, batch, output_logits: bool = False, **kwargs):
        coords = batch["coords"]
        coord_mask = batch["coord_mask"].bool()
        # Hard requirement: derive the structural special-token mask
        # from ``tokens`` (the native / target sequence). ``prev_tokens``
        # is the denoising state and may be full-``<mask>`` early in
        # generation; using it here is a known footgun (see the earlier
        # P0 bug where every residue got dropped from the graph).
        # Callers — including ``DPLMInvFold.forward_encoder()`` — always
        # have ``tokens`` available, so fail-fast on missing.
        tokens = batch.get("tokens")
        if tokens is None:
            raise ValueError(
                "GeoEGNNIPAEncoder requires batch['tokens'] (native or "
                "target sequence) to derive the structural special-token "
                f"mask; got only {list(batch.keys())}. ``prev_tokens`` "
                "is not an acceptable substitute because it may be "
                "full-``<mask>`` during denoising."
            )
        special_sym_mask = self._compute_special_sym_mask(tokens)

        # 1) Build per-protein backbone graph.
        geo_batch = build_geo_graph(
            coords, coord_mask, special_sym_mask, self.graph_config
        )
        graph = geo_batch.graph

        # 2) Project initial node features and run EGNN.
        node_in = self.node_in_proj(graph.x)
        hidden_flat, updated_pos_flat = self.egnn_stack(
            pos=graph.pos,
            feats=node_in,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            batch=graph.batch,
        )

        # 3) Densify to [B, L_token, egnn_hidden_dim].
        hidden_token = self._scatter_flat_to_token(
            hidden_flat,
            geo_batch.residue_mask,
            geo_batch.n_nodes_per_protein,
            feat_dim=hidden_flat.shape[-1],
        )
        updated_pos_token: Optional[torch.Tensor] = None
        if self.update_coors and self.ipa.use_updated_coord_bias:
            updated_pos_token = self._scatter_flat_to_token(
                updated_pos_flat,
                geo_batch.residue_mask,
                geo_batch.n_nodes_per_protein,
                feat_dim=updated_pos_flat.shape[-1],
            )

        # 4) IPA refinement over original N/CA/C frames.
        ipa_positions = dplm_coords_to_ipa_positions(
            coords, coord_mask, special_sym_mask
        )
        refined = self.ipa(
            hidden=hidden_token,
            atom_pos=ipa_positions.atom_pos,
            seq_mask=ipa_positions.seq_mask,
            updated_pos=updated_pos_token,
        )

        # 5) Project to d_model and zero invalid rows.
        feats = self.out_proj(refined)
        feats = torch.where(
            ipa_positions.seq_mask.unsqueeze(-1),
            feats,
            torch.zeros_like(feats),
        )

        encoder_out = {
            "feats": feats,
            "coord_mask": coord_mask,
            "encoder_attention_mask": ipa_positions.seq_mask,
        }

        if output_logits:
            if self.draft_head is None:
                raise RuntimeError(
                    "GeoEGNNIPAEncoder was constructed with "
                    "output_logits=False; cannot honor output_logits=True "
                    "at forward time. Reinstantiate with output_logits=True."
                )
            logits = self.draft_head(feats)
            return logits, encoder_out
        return encoder_out

    # ── helpers ─────────────────────────────────────────────────────

    def _compute_special_sym_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for tok_id in self.special_token_ids:
            mask = mask | (tokens == tok_id)
        return mask

    @staticmethod
    def _coerce_graph_config(value):
        """Accept ``GraphConfig``, ``dict``, or OmegaConf ``DictConfig``.

        Hydra instantiation passes a ``DictConfig`` for the
        ``graph_config`` field; we coerce it to a real ``GraphConfig``
        so ``.validate()`` and dataclass-style attribute access work.
        """
        if value is None:
            return GraphConfig()
        if isinstance(value, GraphConfig):
            return value
        # OmegaConf DictConfig: convert to a plain dict.
        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(value, DictConfig):
                value = OmegaConf.to_container(value, resolve=True)
        except ImportError:
            pass
        if isinstance(value, dict):
            value = dict(value)
            if "mu_r_sigmas" in value and value["mu_r_sigmas"] is not None:
                value["mu_r_sigmas"] = tuple(value["mu_r_sigmas"])
            return GraphConfig(**value)
        raise TypeError(
            "graph_config must be a GraphConfig, dict, or DictConfig; "
            f"got {type(value).__name__}"
        )

    @staticmethod
    def _scatter_flat_to_token(
        flat: torch.Tensor,
        residue_mask: torch.Tensor,
        n_nodes_per_protein: torch.Tensor,
        feat_dim: int,
    ) -> torch.Tensor:
        """Scatter flat ``[N_total, feat_dim]`` back to ``[B, L_token, feat_dim]``
        using the per-protein residue mask.

        The flat layout is protein-major: protein 0's nodes come first,
        then protein 1's, etc. Within each protein, nodes are ordered
        by ascending token position (the True indices of ``residue_mask[b]``).
        """
        B, L_token = residue_mask.shape
        out = torch.zeros(
            B, L_token, feat_dim, device=flat.device, dtype=flat.dtype
        )
        cumsum = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=flat.device),
                n_nodes_per_protein.to(flat.device, dtype=torch.long).cumsum(0),
            ]
        )
        for b in range(B):
            n = int(n_nodes_per_protein[b].item())
            if n == 0:
                continue
            start = int(cumsum[b].item())
            positions = residue_mask[b].nonzero(as_tuple=False).squeeze(-1)
            out[b, positions] = flat[start : start + n]
        return out
