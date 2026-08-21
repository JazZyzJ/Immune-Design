"""Wave-4 core-aware scorer: 9-mer-core feature appended to phi (default off)."""
import torch

from epitope_head.training.model import EpitopeScorer, SpanFeatureBuilder


def _builder(use_core_scorer):
    return SpanFeatureBuilder(d_proj=128, length_emb_dim=16, allele_emb_dim=16,
                              min_k=12, max_k=25, use_core_scorer=use_core_scorer)


def test_flag_construction():
    off = _builder(False)
    assert not hasattr(off, "core_scorer")
    assert off.d_phi == 5 * 128 + 16 + 16  # 672
    assert not any("core_scorer" in k for k in off.state_dict())

    on = _builder(True)
    assert isinstance(on.core_scorer, torch.nn.Linear)
    assert on.core_scorer.weight.shape == (1, 128)
    assert on.d_phi == 673
    assert "core_scorer.weight" in on.state_dict()


def test_pool_9mer_cores():
    b = _builder(True)
    c = torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.])  # len 14
    prefix = torch.zeros(15); prefix[1:] = torch.cumsum(c, 0)
    # span [0,9): exactly one 9-mer window -> its sum
    r1 = b._pool_9mer_cores(prefix, torch.tensor([0]), torch.tensor([9]), 14)
    assert torch.allclose(r1, (prefix[9] - prefix[0]).unsqueeze(0))
    # span [0,14): 6 windows (offsets 0..5) -> logsumexp of their sums, >= max
    r2 = b._pool_9mer_cores(prefix, torch.tensor([0]), torch.tensor([14]), 14)
    sums = torch.stack([prefix[o + 9] - prefix[o] for o in range(6)])
    assert torch.allclose(r2, torch.logsumexp(sums, 0).unsqueeze(0), atol=1e-5)
    assert r2.item() >= sums.max().item()
    # k<9 fallback -> full-span sum, finite
    r3 = b._pool_9mer_cores(prefix, torch.tensor([0]), torch.tensor([5]), 14)
    assert torch.isfinite(r3).all()
    assert torch.allclose(r3, (prefix[5] - prefix[0]).unsqueeze(0))
    # empty span set (chunk with no pos/neg/extra) -> empty, no crash
    empty = torch.zeros(0, dtype=torch.long)
    r4 = b._pool_9mer_cores(prefix, empty, empty, 14)
    assert r4.shape == (0,)


def test_forward_empty_spans():
    b = _builder(True)
    G = torch.randn(20, 128)
    spans = torch.zeros(0, 2, dtype=torch.long)
    allele = torch.zeros(0, dtype=torch.long)
    phi = b(G, spans, 20, allele)
    assert phi.shape == (0, 673)


def test_forward_disabled_shape_and_enabled_grad():
    L, d = 30, 128
    G = torch.randn(L, d)
    spans = torch.tensor([[0, 15], [5, 20], [10, 25]])
    allele = torch.zeros(3, dtype=torch.long)

    off = _builder(False)
    phi_off = off(G, spans, L, allele)
    assert phi_off.shape == (3, 672)

    on = _builder(True)
    phi_on = on(G, spans, L, allele)
    assert phi_on.shape == (3, 673) and torch.isfinite(phi_on).all()
    phi_on.sum().backward()
    assert on.core_scorer.weight.grad is not None
    assert torch.any(on.core_scorer.weight.grad != 0)


def test_epitope_scorer_default_off_legacy():
    m = EpitopeScorer(encoder=torch.nn.Identity(), d_enc=32, d_proj=128,
                      length_emb_dim=16, allele_emb_dim=16, scorer_hidden_dim=64)
    assert not any("core_scorer" in k for k in m.state_dict())
    assert m.d_phi == 672
