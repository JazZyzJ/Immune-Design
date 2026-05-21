"""Per-protein backbone-graph builder for the GeoEGNN-IPA encoder.

Takes DPLM batch coordinates ``[B, L_token, 4, 3]`` in atom order
``(N, CA, C, O)`` plus residue / special-symbol masks and produces a
PyG ``Batch`` suitable for sparse EGNN message passing, together with
the metadata needed to scatter flat node hidden states ``[N_res, H]``
back to DPLM token space ``[B, L_token, H]``.

This module never imports ``MapDiff.*``. The graph-construction logic
is adapted from ``MapDiff/dataloader/cath_dataset.py`` (lines 438-605),
with:

- SciPy ``cdist`` replaced by ``torch.cdist``;
- SASA / B-factor / DSSP node features explicitly skipped per
  ``PLAN_IF_ENCODER.md`` non-negotiable constraint #4;
- AA-identity one-hot skipped (same reason);
- per-protein iteration so edges never cross proteins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from inverse_folding.dplm_refiner.geometry import (
    dplm_coords_to_ipa_positions,
)
from inverse_folding.dplm_refiner.ipa.ipa_utils import cal_dihedrals
from inverse_folding.dplm_refiner.geo_encoder.config import GraphConfig


@dataclass
class GeoGraphBatch:
    """Output of :func:`build_geo_graph`.

    Attributes:
        graph: A PyG ``Batch`` with fields ``x`` ``[N_total, D_node]``,
            ``edge_index`` ``[2, E_total]``, ``edge_attr``
            ``[E_total, D_edge]``, ``pos`` ``[N_total, 3]`` (CA coords),
            and ``batch`` ``[N_total]`` (per-node protein index).
        residue_mask: ``[B, L_token]`` bool. True at the L_token positions
            that became nodes; used by downstream code to scatter dense
            ``[B, L_token, H]`` back from flat ``[N_total, H]``.
        n_nodes_per_protein: ``[B]`` int64; sums to ``N_total``.
        batch_size: B (number of proteins in the batch).
        L_token: token length of the source DPLM batch.
    """

    graph: object  # torch_geometric.data.Batch — str-typed to avoid hard dep
    residue_mask: torch.Tensor
    n_nodes_per_protein: torch.Tensor
    batch_size: int
    L_token: int

    @property
    def n_nodes(self) -> int:
        return int(self.n_nodes_per_protein.sum().item())


# ── helpers ──────────────────────────────────────────────────────────────


def _build_local_basis(atom_pos: torch.Tensor) -> torch.Tensor:
    """Per-residue local frame basis ``[n_i, u_i, v_i]``.

    Args:
        atom_pos: ``[N, 5, 3]`` in order (N, CA, C, CB, O); rows must be
            valid (caller filters by mask).

    Returns:
        ``[N, 3, 3]`` tensor where row 0 = ``n_i``, row 1 = ``u_i``,
        row 2 = ``v_i``. This is exactly the basis MapDiff uses at
        ``cath_dataset.py:451-466``.
    """
    n_atom = atom_pos[:, 0, :]
    ca_atom = atom_pos[:, 1, :]
    c_atom = atom_pos[:, 2, :]
    u_i = F.normalize(n_atom - ca_atom, dim=-1, eps=1e-8)
    t_i = F.normalize(c_atom - ca_atom, dim=-1, eps=1e-8)
    n_i_unnorm = torch.cross(u_i, t_i, dim=-1)
    n_i = F.normalize(n_i_unnorm, dim=-1, eps=1e-8)
    v_i = torch.cross(n_i, u_i, dim=-1)
    basis = torch.stack([n_i, u_i, v_i], dim=-2)
    return basis


def _build_knn_edges(
    ca_coords: torch.Tensor,
    k_neighbors: int,
    cutoff: float,
    seq_fallback: bool = True,
    closest_neighbor_fallback: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-protein kNN edges from one protein's valid CA coords.

    Args:
        ca_coords: ``[N, 3]`` valid CA atom positions.
        k_neighbors: max neighbors per node (after self-exclusion).
        cutoff: distance cutoff (Å); candidates beyond this are excluded.
        seq_fallback: when True, union with sequence neighbors
            ``(i, i±1)`` so adjacent residues are always connected.
        closest_neighbor_fallback: when True and a node has zero kNN
            candidates within ``cutoff``, connect it to its closest
            non-self residue regardless of cutoff. With both fallbacks
            disabled, isolated residues stay isolated.

    Returns:
        ``(src, dst, dist)`` where each is shape ``[E]``.
    """
    n = ca_coords.shape[0]
    if n == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=ca_coords.device)
        empty_float = torch.empty(
            0, dtype=ca_coords.dtype, device=ca_coords.device
        )
        return empty_long, empty_long, empty_float

    dist_full = torch.cdist(ca_coords, ca_coords)  # [N, N]
    eye = torch.eye(n, dtype=torch.bool, device=ca_coords.device)
    within = (dist_full <= cutoff) & (~eye)

    edge_set: set = set()
    for i in range(n):
        candidates = within[i].nonzero(as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            if closest_neighbor_fallback and n > 1:
                tmp = dist_full[i].clone()
                tmp[i] = float("inf")
                j = int(tmp.argmin().item())
                edge_set.add((i, j))
            continue
        cand_dists = dist_full[i, candidates]
        if candidates.numel() > k_neighbors:
            topk = torch.topk(cand_dists, k_neighbors, largest=False).indices
            chosen = candidates[topk]
        else:
            chosen = candidates
        for j in chosen.tolist():
            edge_set.add((i, j))

    if seq_fallback:
        for i in range(n - 1):
            edge_set.add((i, i + 1))
            edge_set.add((i + 1, i))

    if not edge_set:
        empty_long = torch.empty(0, dtype=torch.long, device=ca_coords.device)
        empty_float = torch.empty(
            0, dtype=ca_coords.dtype, device=ca_coords.device
        )
        return empty_long, empty_long, empty_float

    pairs = sorted(edge_set)
    src_list = [p[0] for p in pairs]
    dst_list = [p[1] for p in pairs]
    src = torch.tensor(src_list, dtype=torch.long, device=ca_coords.device)
    dst = torch.tensor(dst_list, dtype=torch.long, device=ca_coords.device)
    dist = dist_full[src, dst]
    return src, dst, dist


def _sinusoidal_positional_encoding(
    positions: torch.Tensor, dim: int
) -> torch.Tensor:
    """Standard sinusoidal positional encoding.

    Args:
        positions: ``[N]`` long, original DPLM token indices (NOT compact
            indices). This is the contract change in P2.9: positional
            encoding must reflect the residue's true sequence position
            so internal special tokens / dropped residues don't shift
            the encoding.
        dim: output embedding dim. Must be even.

    Returns:
        ``[N, dim]`` float.
    """
    n = positions.shape[0]
    if dim == 0 or n == 0:
        return torch.zeros(n, dim, device=positions.device)
    half = dim // 2
    div_term = torch.exp(
        torch.arange(half, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / half)
    )
    pos = positions.float().unsqueeze(-1)
    pe = torch.zeros(n, dim, device=positions.device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe


def _rbf_distance(
    dist: torch.Tensor, bins: int, width: float
) -> torch.Tensor:
    """RBF distance featurization, ``[E] -> [E, bins]``.

    Centers are evenly spaced on ``[0, bins * width]``; sigma = width.
    """
    centers = torch.linspace(
        0.0, bins * width, bins, device=dist.device, dtype=dist.dtype
    )
    sigma = max(width, 1e-3)
    diff = dist.unsqueeze(-1) - centers.unsqueeze(0)
    return torch.exp(-((diff / sigma) ** 2))


def _compute_mu_r_norm(
    ca: torch.Tensor, sigmas: Tuple[float, ...]
) -> torch.Tensor:
    """Per-node mean-neighbor distance ratio across sigma scales.

    For each node ``i`` and sigma ``s``::

        w_ij^{(s)} = softmax_j( -d_ij^2 / sigma_s )      (excluding j = i)
        mean_vec_i^{(s)} = sum_j w_ij^{(s)} * (ca_i - ca_j)
        weighted_dist_i^{(s)} = sum_j w_ij^{(s)} * |ca_i - ca_j|
        mu_r_norm_i^{(s)} = |mean_vec_i^{(s)}| / weighted_dist_i^{(s)}

    Output shape: ``[N, len(sigmas)]``.
    """
    n = ca.shape[0]
    s_dim = len(sigmas)
    if n <= 1:
        return torch.zeros(n, s_dim, device=ca.device, dtype=ca.dtype)

    dist = torch.cdist(ca, ca)  # [N, N]
    sigma_t = torch.tensor(
        list(sigmas), device=ca.device, dtype=ca.dtype
    ).view(s_dim, 1, 1)
    neg_d2 = -dist.unsqueeze(0).pow(2) / sigma_t  # [S, N, N]
    self_mask = (
        torch.eye(n, dtype=torch.bool, device=ca.device).unsqueeze(0)
    )
    neg_d2 = neg_d2.masked_fill(self_mask, float("-inf"))
    weights = torch.softmax(neg_d2, dim=-1)  # [S, N, N]

    diff_vec = ca.unsqueeze(1) - ca.unsqueeze(0)  # diff[i,j] = ca[i] - ca[j]
    # einsum: w[s, n, m] * diff[n, m, c] -> mean_vec[s, n, c]
    mean_vec = torch.einsum("snm,nmc->snc", weights, diff_vec)
    weighted_dist = (weights * dist.unsqueeze(0)).sum(dim=-1)  # [S, N]
    mean_norm = mean_vec.norm(dim=-1)  # [S, N]
    ratio = mean_norm / weighted_dist.clamp(min=1e-8)
    return ratio.transpose(0, 1).contiguous()  # [N, S]


def _local_frame_edge_features(
    atom_pos: torch.Tensor,
    basis: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
) -> torch.Tensor:
    """Per-edge 12-dim local-frame orientation features.

    Following MapDiff (``cath_dataset.py:547-568``)::

        p_ij = basis_dst @ (loc_src - loc_dst)
        q_ij = basis_dst @ n_src
        k_ij = basis_dst @ u_src
        t_ij = basis_dst @ v_src
        s_ij = cat(p_ij, q_ij, k_ij, t_ij)   shape (12,)
    """
    ca = atom_pos[:, 1, :]
    loc_src = ca[edge_src]
    loc_dst = ca[edge_dst]
    basis_dst = basis[edge_dst]  # [E, 3, 3]

    delta = loc_src - loc_dst
    p_ij = torch.bmm(basis_dst, delta.unsqueeze(-1)).squeeze(-1)
    n_src = basis[edge_src, 0, :]
    u_src = basis[edge_src, 1, :]
    v_src = basis[edge_src, 2, :]
    q_ij = torch.bmm(basis_dst, n_src.unsqueeze(-1)).squeeze(-1)
    k_ij = torch.bmm(basis_dst, u_src.unsqueeze(-1)).squeeze(-1)
    t_ij = torch.bmm(basis_dst, v_src.unsqueeze(-1)).squeeze(-1)
    return torch.cat([p_ij, q_ij, k_ij, t_ij], dim=-1)


def _empty_protein_data(node_dim: int, edge_dim: int, device, dtype):
    from torch_geometric.data import Data  # local import to keep dep optional

    return Data(
        x=torch.empty(0, node_dim, device=device, dtype=dtype),
        edge_index=torch.empty(2, 0, dtype=torch.long, device=device),
        edge_attr=torch.empty(0, edge_dim, device=device, dtype=dtype),
        pos=torch.empty(0, 3, device=device, dtype=dtype),
    )


def _build_protein_data(
    atom_pos_b: torch.Tensor,  # [N, 5, 3] valid only
    token_positions: torch.Tensor,  # [N] long, original DPLM token indices
    cfg: GraphConfig,
):
    from torch_geometric.data import Data  # local import

    n_res = atom_pos_b.shape[0]
    device = atom_pos_b.device
    dtype = atom_pos_b.dtype
    ca = atom_pos_b[:, 1, :]

    src, dst, dist = _build_knn_edges(
        ca,
        k_neighbors=cfg.k_neighbors,
        cutoff=cfg.cutoff,
        seq_fallback=cfg.seq_fallback,
        closest_neighbor_fallback=cfg.closest_neighbor_fallback,
    )
    edge_index = torch.stack([src, dst], dim=0)

    # Node features
    dihedrals = cal_dihedrals(atom_pos_b.unsqueeze(0)).squeeze(0)  # [N, 6]
    coord_valid = torch.ones(n_res, 1, device=device, dtype=dtype)
    node_parts: List[torch.Tensor] = [dihedrals, coord_valid]
    if cfg.include_mu_r_norm:
        node_parts.append(_compute_mu_r_norm(ca, cfg.mu_r_sigmas))
    if cfg.pos_enc_dim > 0:
        pe = _sinusoidal_positional_encoding(
            token_positions, dim=cfg.pos_enc_dim
        ).to(dtype)
        node_parts.append(pe)
    x = torch.cat(node_parts, dim=-1)

    # Edge features.
    # IMPORTANT (P2.9 fix): use the ORIGINAL DPLM token positions for
    # the sequence-distance feature; using compact indices would
    # collapse any internal gap (special token, NaN-coord residue) and
    # make non-adjacent residues look adjacent.
    if edge_index.shape[1] == 0:
        edge_attr = torch.empty(0, cfg.edge_feat_dim, device=device, dtype=dtype)
    else:
        true_seq_diff = (
            (token_positions[src] - token_positions[dst]).abs().clamp(max=cfg.seq_dist_cut)
        )
        seq_edge_oh = F.one_hot(
            true_seq_diff, num_classes=cfg.seq_dist_cut + 1
        ).to(dtype)
        dist_rbf = _rbf_distance(
            dist, bins=cfg.dist_bins, width=cfg.dist_bin_width
        )
        contact_sig = (dist <= 8.0).to(dtype).unsqueeze(-1)
        edge_parts: List[torch.Tensor] = [seq_edge_oh, dist_rbf, contact_sig]
        if cfg.include_local_frame:
            basis = _build_local_basis(atom_pos_b)
            edge_parts.append(
                _local_frame_edge_features(atom_pos_b, basis, src, dst)
            )
        edge_attr = torch.cat(edge_parts, dim=-1)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=ca.contiguous(),
    )


# ── public entry ─────────────────────────────────────────────────────────


def build_geo_graph(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    special_sym_mask: torch.Tensor,
    cfg: GraphConfig,
) -> GeoGraphBatch:
    """Build a per-protein backbone graph batch from DPLM coordinates.

    Args:
        coords: ``[B, L_token, 4, 3]`` in order (N, CA, C, O).
        coord_mask: ``[B, L_token]`` bool — positions with valid coords.
        special_sym_mask: ``[B, L_token]`` bool — positions that are
            special tokens (mask/pad/bos/eos/x).
        cfg: graph construction config.

    Returns:
        :class:`GeoGraphBatch` whose ``graph`` is a PyG ``Batch`` ready
        for sparse EGNN message passing.
    """
    # Validate args BEFORE importing PyG so callers get a useful error
    # message regardless of whether torch_geometric is installed.
    cfg.validate()
    if coords.dim() != 4 or coords.shape[-2:] != (4, 3):
        raise ValueError(
            f"coords must be [B, L_token, 4, 3]; got {tuple(coords.shape)}"
        )
    if coord_mask.shape != coords.shape[:2]:
        raise ValueError(
            f"coord_mask shape {tuple(coord_mask.shape)} != "
            f"coords[:2] {tuple(coords.shape[:2])}"
        )
    if special_sym_mask.shape != coords.shape[:2]:
        raise ValueError(
            f"special_sym_mask shape {tuple(special_sym_mask.shape)} != "
            f"coords[:2] {tuple(coords.shape[:2])}"
        )

    try:
        from torch_geometric.data import Batch
    except ImportError as e:
        raise ImportError(
            "torch_geometric is required for the GeoEGNN-IPA encoder. "
            "Install it in the active environment "
            "(e.g. `pip install torch_geometric torch_scatter`) or "
            "ensure the cluster's `immune-design` conda env is active."
        ) from e

    B, L_token = coords.shape[:2]
    ipa_positions = dplm_coords_to_ipa_positions(
        coords, coord_mask, special_sym_mask
    )
    atom_pos = ipa_positions.atom_pos  # [B, L_token, 5, 3]
    seq_mask = ipa_positions.seq_mask  # [B, L_token]

    data_list = []
    n_nodes_per: List[int] = []
    for b in range(B):
        valid_idx = seq_mask[b].nonzero(as_tuple=False).squeeze(-1)
        n_res = int(valid_idx.numel())
        n_nodes_per.append(n_res)
        if n_res == 0:
            data_list.append(
                _empty_protein_data(
                    cfg.node_feat_dim,
                    cfg.edge_feat_dim,
                    device=coords.device,
                    dtype=coords.dtype,
                )
            )
            continue
        atom_pos_b = atom_pos[b, valid_idx]
        # ``valid_idx`` IS the per-protein vector of original DPLM token
        # positions for each node; pass it through so seq-distance and
        # positional-encoding features reflect the true position.
        data_list.append(_build_protein_data(atom_pos_b, valid_idx, cfg))

    # Fail-fast: node count must equal residue_mask count.
    n_nodes_per_t = torch.tensor(
        n_nodes_per, dtype=torch.long, device=coords.device
    )
    expected_total = int(seq_mask.sum().item())
    actual_total = int(n_nodes_per_t.sum().item())
    if actual_total != expected_total:
        raise RuntimeError(
            "node_count_per_graph mismatch: built "
            f"{actual_total} nodes but residue_mask has "
            f"{expected_total} true entries"
        )

    batch = Batch.from_data_list(data_list)
    return GeoGraphBatch(
        graph=batch,
        residue_mask=seq_mask,
        n_nodes_per_protein=n_nodes_per_t,
        batch_size=B,
        L_token=L_token,
    )
