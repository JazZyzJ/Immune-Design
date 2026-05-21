"""Tests for the GeoEGNN-IPA graph builder.

PyG / torch_geometric is only present in the cluster ``immune-design``
env; runtime tests skip when it is missing. The structural/shape tests
on pure-torch helpers (_build_knn_edges, _rbf_distance, etc.) always
run because they have no PyG dependency.
"""

from __future__ import annotations

import pytest
import torch

from inverse_folding.dplm_refiner.geo_encoder.config import GraphConfig
from inverse_folding.dplm_refiner.geo_encoder.graph import (
    _build_knn_edges,
    _build_local_basis,
    _compute_mu_r_norm,
    _local_frame_edge_features,
    _rbf_distance,
    _sinusoidal_positional_encoding,
)


# ── GraphConfig validation ───────────────────────────────────────────────


def test_graph_config_defaults_validate():
    cfg = GraphConfig()
    cfg.validate()
    assert cfg.k_neighbors == 10
    assert cfg.cutoff == 30.0
    assert cfg.seq_dist_cut == 64
    assert cfg.dist_bins == 16
    assert cfg.dist_bin_width == 0.5
    assert cfg.include_local_frame is True
    assert cfg.include_mu_r_norm is True
    assert cfg.seq_fallback is True
    assert cfg.closest_neighbor_fallback is True
    assert cfg.pos_enc_dim == 16


def test_graph_config_rejects_invalid_fields():
    with pytest.raises(ValueError, match="k_neighbors"):
        GraphConfig(k_neighbors=0).validate()
    with pytest.raises(ValueError, match="cutoff"):
        GraphConfig(cutoff=0.0).validate()
    with pytest.raises(ValueError, match="seq_dist_cut"):
        GraphConfig(seq_dist_cut=0).validate()
    with pytest.raises(ValueError, match="dist_bins"):
        GraphConfig(dist_bins=0).validate()
    with pytest.raises(ValueError, match="dist_bin_width"):
        GraphConfig(dist_bin_width=0.0).validate()
    with pytest.raises(ValueError, match="pos_enc_dim"):
        GraphConfig(pos_enc_dim=-1).validate()
    with pytest.raises(ValueError, match="pos_enc_dim must be even"):
        GraphConfig(pos_enc_dim=15).validate()


def test_graph_config_node_and_edge_dims_match_components():
    cfg = GraphConfig()
    # dihedral(6) + coord-valid(1) + mu_r_norm(5 sigmas) + pos_enc(16) = 28
    expected = 6 + 1 + len(cfg.mu_r_sigmas) + cfg.pos_enc_dim
    assert cfg.node_feat_dim == expected
    # seq_one_hot(65) + rbf(16) + contact(1) + local_frame(12) = 94
    assert cfg.edge_feat_dim == (cfg.seq_dist_cut + 1) + cfg.dist_bins + 1 + 12

    cfg_no_extras = GraphConfig(
        include_local_frame=False,
        include_mu_r_norm=False,
        pos_enc_dim=0,
    )
    assert cfg_no_extras.node_feat_dim == 6 + 1
    assert (
        cfg_no_extras.edge_feat_dim
        == cfg_no_extras.seq_dist_cut + 1 + cfg_no_extras.dist_bins + 1
    )


# ── _build_knn_edges ─────────────────────────────────────────────────────


def test_build_knn_edges_returns_per_node_topk_within_cutoff():
    # 6 residues on a line, 1 Å apart along x.
    ca = torch.stack(
        [torch.tensor([float(i), 0.0, 0.0]) for i in range(6)], dim=0
    )
    src, dst, dist = _build_knn_edges(
        ca, k_neighbors=2, cutoff=10.0, seq_fallback=False
    )
    # Each node has at most k=2 neighbors. For interior nodes the two
    # closest are i-1 and i+1.
    pairs = set(zip(src.tolist(), dst.tolist()))
    for i in range(1, 5):
        assert (i, i - 1) in pairs
        assert (i, i + 1) in pairs


def test_build_knn_edges_closest_neighbor_fallback_kicks_in_when_cutoff_too_tight():
    """When BOTH ``seq_fallback`` and ``closest_neighbor_fallback`` are
    on, a residue with no kNN candidates within cutoff still picks up
    its closest non-self neighbor. (Previous test conflated this with
    'strict cutoff'.)"""
    ca = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    src, dst, _ = _build_knn_edges(
        ca,
        k_neighbors=2,
        cutoff=5.0,
        seq_fallback=False,
        closest_neighbor_fallback=True,
    )
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert pairs == {(0, 1), (1, 0)}


def test_build_knn_edges_strict_cutoff_with_both_fallbacks_off():
    """With ``closest_neighbor_fallback=False`` AND ``seq_fallback=False``
    a residue beyond cutoff stays isolated — no edges are added."""
    ca = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    src, dst, _ = _build_knn_edges(
        ca,
        k_neighbors=2,
        cutoff=5.0,
        seq_fallback=False,
        closest_neighbor_fallback=False,
    )
    assert src.shape == (0,)
    assert dst.shape == (0,)


def test_build_knn_edges_seq_fallback_guarantees_sequence_edges():
    # 4 nodes far apart; seq fallback ensures (i, i+1) edges exist
    # even when cutoff is too tight.
    ca = 1000.0 * torch.arange(4).float().view(-1, 1).expand(-1, 3)
    src, dst, _ = _build_knn_edges(
        ca, k_neighbors=1, cutoff=0.1, seq_fallback=True
    )
    pairs = set(zip(src.tolist(), dst.tolist()))
    for i in range(3):
        assert (i, i + 1) in pairs
        assert (i + 1, i) in pairs


def test_build_knn_edges_empty_returns_empty():
    ca = torch.empty(0, 3)
    src, dst, dist = _build_knn_edges(ca, k_neighbors=2, cutoff=5.0)
    assert src.shape == (0,)
    assert dst.shape == (0,)
    assert dist.shape == (0,)


# ── _rbf_distance ────────────────────────────────────────────────────────


def test_rbf_distance_shape_and_finite():
    # bins=8, width=0.5 -> centers up to 4 Å; keep distances in range so
    # values stay above float64 underflow.
    dist = torch.tensor([0.0, 0.5, 1.5, 3.5])
    out = _rbf_distance(dist, bins=8, width=0.5)
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()
    # All values lie in (0, 1] when distances are inside the bin span.
    assert (out > 0).all() and (out <= 1.0 + 1e-6).all()


def test_rbf_distance_underflows_to_zero_outside_bin_span():
    """When a distance lies well outside ``[0, bins*width]``, the RBF
    can underflow to zero — that is a valid 'no nearby bin' signal."""
    out = _rbf_distance(torch.tensor([100.0]), bins=4, width=0.5)
    assert out.shape == (1, 4)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1.0 + 1e-6)


def test_rbf_distance_peaks_near_bin_center():
    dist = torch.tensor([1.0])  # bin centers will include 1.0 in [0, 4]
    out = _rbf_distance(dist, bins=9, width=0.5)
    # bin 2 corresponds to center ~1.0; should be max RBF
    assert int(out[0].argmax().item()) == 2


# ── _compute_mu_r_norm ───────────────────────────────────────────────────


def test_mu_r_norm_single_residue_returns_zeros():
    ca = torch.tensor([[0.0, 0.0, 0.0]])
    out = _compute_mu_r_norm(ca, sigmas=(1.0, 2.0))
    assert out.shape == (1, 2)
    assert torch.all(out == 0)


def test_mu_r_norm_shape_matches_sigmas():
    ca = torch.randn(7, 3)
    out = _compute_mu_r_norm(ca, sigmas=(1.0, 2.0, 5.0))
    assert out.shape == (7, 3)
    assert torch.isfinite(out).all()


# ── _build_local_basis ───────────────────────────────────────────────────


def test_local_basis_is_orthonormal():
    # Build a canonical triplet: N at (0,1,0), CA at origin, C at (1,0,0).
    atom_pos = torch.zeros(3, 5, 3)
    atom_pos[:, 0] = torch.tensor([0.0, 1.0, 0.0])
    atom_pos[:, 1] = torch.tensor([0.0, 0.0, 0.0])
    atom_pos[:, 2] = torch.tensor([1.0, 0.0, 0.0])
    basis = _build_local_basis(atom_pos)
    assert basis.shape == (3, 3, 3)
    # Each row should be unit-length.
    norms = basis.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # n_i, u_i pair should be orthogonal.
    n_i, u_i, v_i = basis[:, 0], basis[:, 1], basis[:, 2]
    assert torch.allclose((n_i * u_i).sum(-1), torch.zeros(3), atol=1e-5)
    assert torch.allclose((n_i * v_i).sum(-1), torch.zeros(3), atol=1e-5)


# ── _local_frame_edge_features ───────────────────────────────────────────


def test_sinusoidal_positional_encoding_shape_and_finite():
    positions = torch.tensor([0, 5, 100, 500], dtype=torch.long)
    pe = _sinusoidal_positional_encoding(positions, dim=16)
    assert pe.shape == (4, 16)
    assert torch.isfinite(pe).all()
    # Two distinct positions must map to distinct encodings.
    assert not torch.allclose(pe[0], pe[1])
    assert not torch.allclose(pe[2], pe[3])


def test_sinusoidal_positional_encoding_handles_zero_dim_and_empty():
    pe_zero = _sinusoidal_positional_encoding(
        torch.tensor([0, 1, 2], dtype=torch.long), dim=0
    )
    assert pe_zero.shape == (3, 0)
    pe_empty = _sinusoidal_positional_encoding(
        torch.empty(0, dtype=torch.long), dim=8
    )
    assert pe_empty.shape == (0, 8)


def test_local_frame_edge_feature_shape_is_12():
    n = 5
    atom_pos = torch.randn(n, 5, 3)
    basis = _build_local_basis(atom_pos)
    edge_src = torch.tensor([0, 1, 2])
    edge_dst = torch.tensor([1, 2, 3])
    feats = _local_frame_edge_features(atom_pos, basis, edge_src, edge_dst)
    assert feats.shape == (3, 12)
    assert torch.isfinite(feats).all()


# ── Module-level non-import-of-MapDiff guard ─────────────────────────────


def test_graph_module_does_not_import_mapdiff():
    """Plan non-negotiable: production code must not import MapDiff.*."""
    import inverse_folding.dplm_refiner.geo_encoder.graph as graph_mod

    src = open(graph_mod.__file__).read()
    # The string "MapDiff/" may appear in docstrings as a path reference,
    # but an import statement must not.
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            assert "MapDiff" not in stripped, (
                f"production import of MapDiff: {line}"
            )


# ── Runtime tests (skip when torch_geometric missing) ────────────────────


def _import_pyg():
    pytest.importorskip("torch_geometric")


def _make_batch(B: int = 2, L: int = 8, n_valid: int = 5):
    """Synthetic DPLM-style batch: B proteins, L_token positions, first
    n_valid positions are valid residues with non-degenerate coords."""
    coords = torch.zeros(B, L, 4, 3)
    coord_mask = torch.zeros(B, L, dtype=torch.bool)
    special = torch.zeros(B, L, dtype=torch.bool)
    for b in range(B):
        for i in range(n_valid):
            base = float(i) + 0.1 * b
            coords[b, i, 0] = torch.tensor([base, 1.0, 0.0])  # N
            coords[b, i, 1] = torch.tensor([base, 0.0, 0.0])  # CA
            coords[b, i, 2] = torch.tensor([base + 1.5, 0.0, 0.0])  # C
            coords[b, i, 3] = torch.tensor([base + 2.0, 0.5, 0.0])  # O
            coord_mask[b, i] = True
        # Last L - n_valid positions: pad / mask.
        special[b, n_valid:] = True
    return coords, coord_mask, special


def test_build_geo_graph_returns_pyg_batch_with_correct_node_count():
    _import_pyg()
    from inverse_folding.dplm_refiner.geo_encoder.graph import build_geo_graph

    coords, coord_mask, special = _make_batch(B=3, L=8, n_valid=5)
    cfg = GraphConfig()
    out = build_geo_graph(coords, coord_mask, special, cfg)

    assert out.batch_size == 3
    assert out.L_token == 8
    assert out.n_nodes_per_protein.tolist() == [5, 5, 5]
    assert out.n_nodes == 15
    assert out.residue_mask.shape == (3, 8)
    assert int(out.residue_mask.sum().item()) == 15
    assert out.graph.x.shape == (15, cfg.node_feat_dim)
    assert out.graph.pos.shape == (15, 3)
    assert out.graph.edge_attr.shape[1] == cfg.edge_feat_dim


def test_build_geo_graph_edges_never_cross_proteins():
    _import_pyg()
    from inverse_folding.dplm_refiner.geo_encoder.graph import build_geo_graph

    coords, coord_mask, special = _make_batch(B=2, L=6, n_valid=4)
    cfg = GraphConfig()
    out = build_geo_graph(coords, coord_mask, special, cfg)
    edge_index = out.graph.edge_index
    node_batch = out.graph.batch  # [N_total] per-node protein index
    # For every edge, src and dst must share the same protein index.
    src_b = node_batch[edge_index[0]]
    dst_b = node_batch[edge_index[1]]
    assert torch.all(src_b == dst_b)


def test_build_geo_graph_handles_empty_protein():
    _import_pyg()
    from inverse_folding.dplm_refiner.geo_encoder.graph import build_geo_graph

    coords = torch.zeros(2, 5, 4, 3)
    coord_mask = torch.zeros(2, 5, dtype=torch.bool)
    coord_mask[0, :4] = True
    # Build a non-degenerate frame for protein 0.
    for i in range(4):
        coords[0, i, 0] = torch.tensor([float(i), 1.0, 0.0])
        coords[0, i, 1] = torch.tensor([float(i), 0.0, 0.0])
        coords[0, i, 2] = torch.tensor([float(i) + 1.5, 0.0, 0.0])
        coords[0, i, 3] = torch.tensor([float(i) + 2.0, 0.5, 0.0])
    # Protein 1 has zero valid residues.
    special = torch.zeros(2, 5, dtype=torch.bool)
    cfg = GraphConfig()
    out = build_geo_graph(coords, coord_mask, special, cfg)
    assert out.n_nodes_per_protein.tolist() == [4, 0]
    assert out.n_nodes == 4


def test_build_geo_graph_rejects_wrong_coords_shape():
    cfg = GraphConfig()
    with pytest.raises(ValueError, match=r"\[B, L_token, 4, 3\]"):
        from inverse_folding.dplm_refiner.geo_encoder.graph import (
            build_geo_graph,
        )

        build_geo_graph(
            torch.zeros(2, 5, 3),
            torch.zeros(2, 5, dtype=torch.bool),
            torch.zeros(2, 5, dtype=torch.bool),
            cfg,
        )


def test_build_geo_graph_rejects_mask_shape_mismatch():
    cfg = GraphConfig()
    with pytest.raises(ValueError, match="coord_mask"):
        from inverse_folding.dplm_refiner.geo_encoder.graph import (
            build_geo_graph,
        )

        build_geo_graph(
            torch.zeros(2, 5, 4, 3),
            torch.zeros(2, 4, dtype=torch.bool),  # wrong L_token
            torch.zeros(2, 5, dtype=torch.bool),
            cfg,
        )


def test_build_geo_graph_seq_distance_uses_original_token_positions():
    """P2.9 regression: if an internal token position is dropped from
    the residue mask (e.g., a special token in the middle), the
    sequence-distance edge feature must still reflect the gap in the
    DPLM token sequence — not the compact post-filter index."""
    _import_pyg()
    from inverse_folding.dplm_refiner.geo_encoder.graph import build_geo_graph

    B, L = 1, 6
    coords = torch.zeros(B, L, 4, 3)
    coord_mask = torch.zeros(B, L, dtype=torch.bool)
    special = torch.zeros(B, L, dtype=torch.bool)
    # Valid residues at token positions {0, 1, 3, 5} — 2 and 4 are
    # marked special and dropped from the graph. Compact indices would
    # treat (0, 3) as seq-distance 2, but the true distance is 3.
    valid_positions = [0, 1, 3, 5]
    for i in valid_positions:
        coords[0, i, 0] = torch.tensor([float(i), 1.0, 0.0])
        coords[0, i, 1] = torch.tensor([float(i), 0.0, 0.0])
        coords[0, i, 2] = torch.tensor([float(i) + 1.5, 0.0, 0.0])
        coords[0, i, 3] = torch.tensor([float(i) + 2.0, 0.5, 0.0])
        coord_mask[0, i] = True
    special[0, 2] = True
    special[0, 4] = True

    cfg = GraphConfig(seq_dist_cut=10, include_local_frame=False, pos_enc_dim=0)
    out = build_geo_graph(coords, coord_mask, special, cfg)
    # Node 0 corresponds to token pos 0; node 2 corresponds to token pos 3
    # (token 2 was dropped). Find the edge between them and assert
    # seq-distance one-hot index = 3, not 2.
    edge_index = out.graph.edge_index
    edge_attr = out.graph.edge_attr
    # seq one-hot occupies first (seq_dist_cut + 1) = 11 columns
    seq_oh = edge_attr[:, : cfg.seq_dist_cut + 1]
    # Look at edges that connect node 0 and node 2 (compact indices for
    # token positions 0 and 3 respectively).
    found_match = False
    for e in range(edge_index.shape[1]):
        a, b = int(edge_index[0, e]), int(edge_index[1, e])
        if {a, b} == {0, 2}:
            assert int(seq_oh[e].argmax().item()) == 3
            found_match = True
    assert found_match, (
        "no edge between compact indices 0 and 2 (token positions 0 and 3)"
    )
