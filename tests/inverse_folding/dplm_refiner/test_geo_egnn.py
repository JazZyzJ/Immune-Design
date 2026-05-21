"""Tests for the sparse EGNN backbone in the GeoEGNN-IPA encoder.

PyG / torch_scatter are only present in the cluster ``immune-design``
env; runtime tests skip when either is missing. The structural tests
on the module (helpers + non-MapDiff import guard + fallback ImportError)
always run.
"""

from __future__ import annotations

import pytest
import torch

from inverse_folding.dplm_refiner.geo_encoder import egnn as egnn_mod
from inverse_folding.dplm_refiner.geo_encoder.egnn import (
    EGNNSparseLayer,
    EGNNStack,
    _CoorsNorm,
    _fourier_encode_dist,
)


# ── Always-runnable structural checks ────────────────────────────────────


def test_module_does_not_import_mapdiff():
    src = open(egnn_mod.__file__).read()
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            assert "MapDiff" not in s, f"production import of MapDiff: {s}"


def test_coors_norm_outputs_finite_with_learned_scale():
    norm = _CoorsNorm(scale_init=0.1)
    coors = torch.randn(7, 3) * 5.0
    out = norm(coors)
    assert out.shape == (7, 3)
    assert torch.isfinite(out).all()


def test_coors_norm_zero_input_returns_zero():
    norm = _CoorsNorm()
    out = norm(torch.zeros(4, 3))
    assert torch.all(out == 0)


def test_fourier_encode_dist_shape_and_include_self():
    # input shape [E] -> output [E, 2*num_encodings (+1 with self)]
    x = torch.tensor([0.5, 1.0, 2.0])
    out = _fourier_encode_dist(x, num_encodings=4, include_self=True)
    assert out.shape == (3, 9)
    out_no_self = _fourier_encode_dist(x, num_encodings=4, include_self=False)
    assert out_no_self.shape == (3, 8)
    assert torch.isfinite(out).all()


# ── PyG-gated runtime tests ──────────────────────────────────────────────


def _require_pyg_and_scatter():
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")


def _build_two_graph_batch(n_per_graph: int = 4, feats_dim: int = 8):
    """Build a 2-graph batch where the graphs are connected only
    internally — no edges crossing protein boundaries."""
    n_total = n_per_graph * 2
    pos = torch.randn(n_total, 3)
    feats = torch.randn(n_total, feats_dim)
    edges = []
    for b in range(2):
        offset = b * n_per_graph
        for i in range(n_per_graph - 1):
            edges.append([offset + i, offset + i + 1])
            edges.append([offset + i + 1, offset + i])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_attr = torch.randn(edge_index.shape[1], 3)
    batch = torch.tensor(
        [0] * n_per_graph + [1] * n_per_graph, dtype=torch.long
    )
    return pos, feats, edge_index, edge_attr, batch


def test_egnn_layer_forward_finite_and_shape():
    _require_pyg_and_scatter()
    feats_dim = 8
    pos, feats, edge_index, edge_attr, batch = _build_two_graph_batch(
        4, feats_dim
    )
    layer = EGNNSparseLayer(
        feats_dim=feats_dim,
        edge_attr_dim=edge_attr.shape[-1],
        update_coors=True,
        update_global=True,
    )
    layer.eval()
    x_in = torch.cat([pos, feats], dim=-1)
    x_out, new_edge_attr = layer(x_in, edge_index, edge_attr, batch=batch)
    assert x_out.shape == x_in.shape
    assert torch.isfinite(x_out).all()
    # update_edge=False -> edge attr returned unchanged
    assert new_edge_attr is edge_attr or torch.equal(new_edge_attr, edge_attr)


def test_egnn_stack_returns_separate_hidden_and_pos():
    _require_pyg_and_scatter()
    feats_dim = 8
    pos, feats, edge_index, edge_attr, batch = _build_two_graph_batch(
        4, feats_dim
    )
    stack = EGNNStack(
        n_layers=2,
        feats_dim=feats_dim,
        edge_attr_dim=edge_attr.shape[-1],
        update_coors=True,
        update_global=True,
    )
    stack.eval()
    hidden, pos_out = stack(pos, feats, edge_index, edge_attr, batch=batch)
    assert hidden.shape == (8, feats_dim)
    assert pos_out.shape == (8, 3)
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(pos_out).all()


def test_egnn_stack_update_coors_false_preserves_pos_exactly():
    _require_pyg_and_scatter()
    feats_dim = 8
    pos, feats, edge_index, edge_attr, batch = _build_two_graph_batch(
        4, feats_dim
    )
    stack = EGNNStack(
        n_layers=2,
        feats_dim=feats_dim,
        edge_attr_dim=edge_attr.shape[-1],
        update_coors=False,
        update_global=True,
    )
    stack.eval()
    _, pos_out = stack(pos, feats, edge_index, edge_attr, batch=batch)
    # update_coors=False is contractually a no-op on the public pos.
    assert torch.equal(pos_out, pos)


def test_egnn_stack_update_coors_true_changes_pos():
    _require_pyg_and_scatter()
    torch.manual_seed(0)
    feats_dim = 8
    pos, feats, edge_index, edge_attr, batch = _build_two_graph_batch(
        4, feats_dim
    )
    stack = EGNNStack(
        n_layers=2,
        feats_dim=feats_dim,
        edge_attr_dim=edge_attr.shape[-1],
        update_coors=True,
        update_global=True,
    )
    stack.eval()
    _, pos_out = stack(pos, feats, edge_index, edge_attr, batch=batch)
    # At nonzero initialization, the EGNN coord MLP perturbs positions.
    assert not torch.equal(pos_out, pos)


def test_egnn_no_cross_graph_message_passing():
    """When two proteins share a batch but have no inter-protein edges,
    perturbing protein 1's positions must not change protein 0's hidden
    output. (No-edge ⇒ no-aggregation ⇒ no-leak.)"""
    _require_pyg_and_scatter()
    torch.manual_seed(0)
    feats_dim = 8
    pos, feats, edge_index, edge_attr, batch = _build_two_graph_batch(
        4, feats_dim
    )
    # With update_global=True, the global MLP gates protein-0 hidden by
    # protein-0's mean only (scatter_mean by batch index), so cross-graph
    # bleed is impossible there too.
    stack = EGNNStack(
        n_layers=2,
        feats_dim=feats_dim,
        edge_attr_dim=edge_attr.shape[-1],
        update_coors=True,
        update_global=True,
    )
    stack.eval()
    hidden_a, _ = stack(pos, feats, edge_index, edge_attr, batch=batch)
    # Perturb protein 1's pos + feats.
    pos2 = pos.clone()
    feats2 = feats.clone()
    pos2[4:] += 5.0
    feats2[4:] += torch.randn(4, feats_dim)
    hidden_b, _ = stack(pos2, feats2, edge_index, edge_attr, batch=batch)
    # Protein 0 hidden must be unchanged.
    assert torch.allclose(hidden_a[:4], hidden_b[:4], atol=1e-5)


def test_egnn_layer_update_edge_returns_same_shape():
    _require_pyg_and_scatter()
    feats_dim = 8
    pos, feats, edge_index, edge_attr, batch = _build_two_graph_batch(
        4, feats_dim
    )
    layer = EGNNSparseLayer(
        feats_dim=feats_dim,
        edge_attr_dim=edge_attr.shape[-1],
        update_edge=True,
        update_global=True,
    )
    layer.eval()
    x_in = torch.cat([pos, feats], dim=-1)
    _, new_edge_attr = layer(x_in, edge_index, edge_attr, batch=batch)
    assert new_edge_attr.shape == edge_attr.shape
    assert torch.isfinite(new_edge_attr).all()


def test_pyg_and_scatter_import_smoke():
    _require_pyg_and_scatter()
    import torch_geometric  # noqa: F401
    from torch_scatter import scatter_add  # noqa: F401


def test_egnn_layer_rejects_invalid_aggr():
    with pytest.raises(ValueError, match="aggr"):
        EGNNSparseLayer(feats_dim=4, edge_attr_dim=0, aggr="bogus")


def test_egnn_stack_rejects_invalid_n_layers():
    with pytest.raises(ValueError, match="n_layers"):
        EGNNStack(n_layers=0, feats_dim=4)
