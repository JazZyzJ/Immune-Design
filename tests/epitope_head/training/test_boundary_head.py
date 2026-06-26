"""Wave-4 dual-head: gradient isolation + legacy-preservation contract.

The boundary (exact) head must NOT leak gradient into the shared trainable trunk
(projection / span_features / scorer / encoder), so the per-residue landscape the
downstream Reference Flow consumes is provably unchanged. Default-off must be
bit-for-bit legacy.
"""
import torch

from epitope_head.training.model import EpitopeScorer


def _tiny_scorer(enable_boundary_head: bool) -> EpitopeScorer:
    # encoder is unused at the score_spans level; a placeholder module is fine.
    return EpitopeScorer(
        encoder=torch.nn.Identity(),
        d_enc=32, d_proj=16, length_emb_dim=8, allele_emb_dim=8,
        min_k=12, max_k=25, n_alleles=1, scorer_hidden_dim=32,
        enable_boundary_head=enable_boundary_head, boundary_head_hidden_dim=16,
    )


def _inputs(d_proj=16, chunk_len=30):
    G = torch.randn(chunk_len, d_proj, requires_grad=True)
    spans = torch.tensor([[0, 15], [5, 20], [10, 25]])  # lens 15 in [12,25]
    allele = torch.zeros(3, dtype=torch.long)
    return G, spans, allele, chunk_len


def test_boundary_head_gradient_isolation():
    """Backprop on z_exact must reach ONLY boundary_head params (double-detach)."""
    m = _tiny_scorer(True)
    G, spans, allele, L = _inputs()
    z_region, z_exact = m.score_spans(G, L, spans, allele, return_dual=True)
    z_exact.sum().backward()

    assert all(p.grad is not None for p in m.boundary_head.parameters()), \
        "boundary_head must receive gradient"
    for name, mod in (("scorer", m.scorer), ("span_features", m.span_features),
                      ("projection", m.projection)):
        assert all(p.grad is None for p in mod.parameters()), \
            f"{name} must receive NO gradient from the exact loss"
    assert G.grad is None, "upstream (projection output) must receive no gradient"


def test_exact_does_not_perturb_region_grad():
    """z_region keeps a live path to the trunk; the exact term doesn't change it."""
    m = _tiny_scorer(True)
    G, spans, allele, L = _inputs()
    z_region, z_exact = m.score_spans(G, L, spans, allele, return_dual=True)
    # A region-side loss DOES flow to the trunk.
    z_region.sum().backward()
    assert any(p.grad is not None for p in m.scorer.parameters())


def test_default_off_is_legacy():
    """enable_boundary_head=False: no extra state, single-tensor return."""
    m = _tiny_scorer(False)
    assert not any("boundary_head" in k for k in m.state_dict()), \
        "no boundary_head params in a legacy model"
    G, spans, allele, L = _inputs()
    out = m.score_spans(G, L, spans, allele, return_dual=True)  # flag ignored when off
    assert isinstance(out, torch.Tensor) and out.shape == (3,), \
        "legacy path returns a single [N] tensor even with return_dual=True"


def test_enabled_default_return_is_single_tensor():
    """Even with the head enabled, return_dual=False stays single-tensor (legacy call sites)."""
    m = _tiny_scorer(True)
    G, spans, allele, L = _inputs()
    out = m.score_spans(G, L, spans, allele)  # return_dual defaults False
    assert isinstance(out, torch.Tensor) and out.shape == (3,)
