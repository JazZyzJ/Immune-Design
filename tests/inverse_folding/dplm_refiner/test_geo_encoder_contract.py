"""Tests for ``GeoEGNNIPAEncoder``'s public forward contract.

The full encoder pipeline pulls torch_geometric + torch_scatter, which
live only in the cluster ``immune-design`` env. Runtime tests are gated
by ``importorskip``; structural tests on the byprot registration and
the module's non-MapDiff imports always run.
"""

from __future__ import annotations

import pytest
import torch

import inverse_folding.dplm_refiner.geo_encoder.encoder as encoder_mod
import inverse_folding.dplm_refiner.geo_encoder.ipa as ipa_mod
from inverse_folding.dplm_refiner.geo_encoder import (
    GeoEGNNIPAEncoder,
    GraphConfig,
)


# ── Always-runnable structural checks ────────────────────────────────────


def test_encoder_module_does_not_import_mapdiff():
    for mod in (encoder_mod, ipa_mod):
        src = open(mod.__file__).read()
        for line in src.splitlines():
            s = line.strip()
            if s.startswith("import ") or s.startswith("from "):
                assert "MapDiff" not in s, (
                    f"{mod.__name__}: production import of MapDiff: {s}"
                )


def test_encoder_output_logits_false_creates_no_draft_head():
    """Constructing without output_logits must not allocate the head."""
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")
    enc = GeoEGNNIPAEncoder(output_logits=False)
    assert enc.draft_head is None
    # Trainable parameter names should not include 'draft_head'
    for name, _ in enc.named_parameters():
        assert "draft_head" not in name, name


def test_encoder_output_logits_true_registers_draft_head():
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")
    enc = GeoEGNNIPAEncoder(output_logits=True, vocab_size=33)
    assert enc.draft_head is not None
    assert enc.draft_head.out_features == 33


def test_encoder_compute_special_sym_mask_picks_default_ids():
    """Default special set is (cls=0, pad=1, eos=2). NB: ``<mask>``
    (id=32) is intentionally NOT in the default — at DPLM denoising
    step 0 ``prev_tokens`` is full-mask but those positions are real
    residues. ``<unk>`` (id=3) is also kept in the graph."""
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")
    enc = GeoEGNNIPAEncoder()
    tokens = torch.tensor([[0, 5, 1, 32, 4, 2, 3]])
    mask = enc._compute_special_sym_mask(tokens)
    assert mask.dtype == torch.bool
    # cls(True), aa5(False), pad(True), mask(False), aa4(False), eos(True),
    # unk(False)
    assert mask.tolist() == [
        [True, False, True, False, False, True, False]
    ]


def test_encoder_requires_tokens_and_rejects_prev_tokens_only():
    """At generation, ``prev_tokens`` is full-mask early on. The encoder
    must hard-require ``batch['tokens']`` (native or target sequence)
    and refuse to fall back to ``prev_tokens`` so the special-token
    mask cannot conflate ``<mask>`` denoising state with structural
    boundaries (see the L0091 P0.1 bug)."""
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")
    import inspect

    src = inspect.getsource(GeoEGNNIPAEncoder.forward)
    # Source-level guard: encoder reads ``batch.get("tokens")`` directly,
    # not the fallback chain.
    assert 'batch.get("tokens")' in src
    assert 'batch.get("prev_tokens")' not in src
    # Behavioral guard: forward without ``tokens`` raises immediately.
    enc = GeoEGNNIPAEncoder(
        d_model=16,
        egnn_depth=1,
        egnn_hidden_dim=8,
        ipa_depth=1,
        ipa_hidden_dim=8,
        ipa_pairwise_dim=8,
        ipa_heads=2,
    )
    bad_batch = {
        "coords": torch.zeros(1, 4, 4, 3),
        "coord_mask": torch.zeros(1, 4, dtype=torch.bool),
        "prev_tokens": torch.zeros(1, 4, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="requires batch\\['tokens'\\]"):
        enc(bad_batch)


def test_byprot_wrapper_registers_geoegnn_name():
    """The byprot wrapper file must call ``register_model``. We check
    via source inspection because importing byprot pulls in the full
    DPLM runtime."""
    import pathlib

    wrapper_src = pathlib.Path(
        "inverse_folding/dplm/src/byprot/models/dplm/modules/"
        "geoegnn_ipa_encoder.py"
    )
    text = wrapper_src.read_text()
    assert 'register_model("geoegnn_ipa_encoder")' in text


# ── Runtime tests (skip when PyG / torch_scatter missing) ────────────────


def _require_pyg_stack():
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")


def _make_dplm_batch(B=2, L=8, n_valid=5):
    """Synthetic DPLM-style batch with B proteins, L_token positions."""
    coords = torch.zeros(B, L, 4, 3)
    coord_mask = torch.zeros(B, L, dtype=torch.bool)
    tokens = torch.full((B, L), 1, dtype=torch.long)  # pad=1 by default
    for b in range(B):
        for i in range(n_valid):
            base = float(i) + 0.1 * b
            coords[b, i, 0] = torch.tensor([base, 1.0, 0.0])
            coords[b, i, 1] = torch.tensor([base, 0.0, 0.0])
            coords[b, i, 2] = torch.tensor([base + 1.5, 0.0, 0.0])
            coords[b, i, 3] = torch.tensor([base + 2.0, 0.5, 0.0])
            coord_mask[b, i] = True
            # token in regular AA range (5..30 avoids special IDs)
            tokens[b, i] = 5 + (i % 20)
    return {"coords": coords, "coord_mask": coord_mask, "tokens": tokens}


def test_encoder_forward_no_logits_returns_dict_with_required_keys():
    _require_pyg_stack()
    enc = GeoEGNNIPAEncoder(
        d_model=64,
        egnn_depth=1,
        egnn_hidden_dim=32,
        ipa_depth=1,
        ipa_hidden_dim=32,
        ipa_pairwise_dim=32,
        ipa_heads=2,
    )
    enc.eval()
    batch = _make_dplm_batch(B=2, L=8, n_valid=5)
    out = enc(batch, output_logits=False)
    assert isinstance(out, dict)
    assert "feats" in out and out["feats"].shape == (2, 8, 64)
    assert "coord_mask" in out
    assert torch.equal(out["coord_mask"], batch["coord_mask"])
    assert "encoder_attention_mask" in out
    # Rows where seq_mask is False must be zeroed.
    invalid = ~out["encoder_attention_mask"]
    if invalid.any():
        assert torch.all(out["feats"][invalid] == 0)


def test_encoder_forward_with_logits_returns_tuple():
    _require_pyg_stack()
    enc = GeoEGNNIPAEncoder(
        d_model=64,
        output_logits=True,
        vocab_size=33,
        egnn_depth=1,
        egnn_hidden_dim=32,
        ipa_depth=1,
        ipa_hidden_dim=32,
        ipa_pairwise_dim=32,
        ipa_heads=2,
    )
    enc.eval()
    batch = _make_dplm_batch(B=2, L=8, n_valid=5)
    out = enc(batch, output_logits=True)
    assert isinstance(out, tuple) and len(out) == 2
    logits, encoder_out = out
    assert logits.shape == (2, 8, 33)
    assert isinstance(encoder_out, dict)
    assert "feats" in encoder_out
    assert "coord_mask" in encoder_out


def test_encoder_output_logits_false_raises_if_requested_at_forward():
    _require_pyg_stack()
    enc = GeoEGNNIPAEncoder(
        d_model=64,
        output_logits=False,
        egnn_depth=1,
        egnn_hidden_dim=32,
        ipa_depth=1,
        ipa_hidden_dim=32,
        ipa_pairwise_dim=32,
        ipa_heads=2,
    )
    enc.eval()
    batch = _make_dplm_batch(B=1, L=6, n_valid=4)
    with pytest.raises(RuntimeError, match="output_logits=False"):
        enc(batch, output_logits=True)


def test_encoder_zeros_special_token_rows():
    _require_pyg_stack()
    enc = GeoEGNNIPAEncoder(
        d_model=32,
        egnn_depth=1,
        egnn_hidden_dim=16,
        ipa_depth=1,
        ipa_hidden_dim=16,
        ipa_pairwise_dim=16,
        ipa_heads=2,
    )
    enc.eval()
    batch = _make_dplm_batch(B=1, L=6, n_valid=4)
    # Replace position 0's token with cls (id=0); coord_mask there is
    # True but special_sym_mask will flag it.
    batch["tokens"][0, 0] = 0  # cls id
    out = enc(batch, output_logits=False)
    # Row 0 should be zeroed because special_sym_mask=True there.
    assert torch.all(out["feats"][0, 0] == 0)


def test_encoder_returns_coord_mask_in_both_branches():
    """coord_mask must be present whether output_logits is True or False."""
    _require_pyg_stack()
    common_kwargs = dict(
        d_model=32,
        egnn_depth=1,
        egnn_hidden_dim=16,
        ipa_depth=1,
        ipa_hidden_dim=16,
        ipa_pairwise_dim=16,
        ipa_heads=2,
    )
    batch = _make_dplm_batch(B=1, L=6, n_valid=4)

    enc_no_logits = GeoEGNNIPAEncoder(**common_kwargs, output_logits=False)
    enc_no_logits.eval()
    out1 = enc_no_logits(batch, output_logits=False)
    assert "coord_mask" in out1

    enc_with_logits = GeoEGNNIPAEncoder(**common_kwargs, output_logits=True)
    enc_with_logits.eval()
    _, out2 = enc_with_logits(batch, output_logits=True)
    assert "coord_mask" in out2


def test_encoder_handles_update_coors_true_path():
    """When update_coors=True, the EGNN pos channel is allowed to drift
    but the IPA frames remain on original coords. We just verify the
    forward runs end-to-end without shape errors."""
    _require_pyg_stack()
    enc = GeoEGNNIPAEncoder(
        d_model=32,
        egnn_depth=1,
        egnn_hidden_dim=16,
        update_coors=True,
        ipa_depth=1,
        ipa_hidden_dim=16,
        ipa_pairwise_dim=16,
        ipa_heads=2,
        use_updated_coord_bias=True,
    )
    enc.eval()
    batch = _make_dplm_batch(B=1, L=6, n_valid=4)
    out = enc(batch, output_logits=False)
    assert out["feats"].shape == (1, 6, 32)
    assert torch.isfinite(out["feats"]).all()


def test_encoder_can_be_used_via_forward_encoder_contract():
    """Smoke test that the encoder's output dict can be consumed by the
    downstream ``DPLMInvFold.forward_encoder`` code path; specifically
    the ``init_pred`` derivation when ``output_logits=True``."""
    _require_pyg_stack()
    enc = GeoEGNNIPAEncoder(
        d_model=32,
        output_logits=True,
        vocab_size=33,
        egnn_depth=1,
        egnn_hidden_dim=16,
        ipa_depth=1,
        ipa_hidden_dim=16,
        ipa_pairwise_dim=16,
        ipa_heads=2,
    )
    enc.eval()
    batch = _make_dplm_batch(B=1, L=6, n_valid=4)
    # forward_encoder with use_draft_seq=True would set batch['prev_tokens']
    batch["prev_tokens"] = batch["tokens"]
    logits, encoder_out = enc(batch, output_logits=True)
    init_pred = logits.argmax(-1)
    # init_pred should be a long tensor in [0, vocab_size)
    assert init_pred.dtype == torch.long
    assert int(init_pred.max().item()) < 33
    assert int(init_pred.min().item()) >= 0
