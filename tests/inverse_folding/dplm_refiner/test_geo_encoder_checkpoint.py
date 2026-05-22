"""Tests for GeoEGNN-IPA encoder checkpoint save/load + Hydra
registry resolution.

Full encoder instantiation needs PyG + torch_scatter; runtime tests
``importorskip`` when those are missing. Source-level checks for the
experiment YAML and registry wrapper always run.
"""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ── Source-level checks ──────────────────────────────────────────────────


def test_geoegnn_experiment_yaml_exists_and_targets_geoegnn():
    yaml_path = (
        PROJECT_ROOT
        / "inverse_folding/dplm/configs/experiment/dplm/cond_dplm_650m_geoegnn.yaml"
    )
    assert yaml_path.is_file(), f"missing experiment YAML: {yaml_path}"
    text = yaml_path.read_text()
    assert "_target_: geoegnn_ipa_encoder" in text
    assert "detach_encoder_feats: false" in text
    assert "update_coors" in text


def test_requirements_txt_uncomments_torch_scatter():
    req = (PROJECT_ROOT / "inverse_folding/dplm/requirements.txt").read_text()
    # Find the torch_scatter line and verify it's NOT a comment.
    for line in req.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("torch_scatter"):
            return  # found uncommented line
    raise AssertionError(
        "torch_scatter line missing or still commented in "
        "inverse_folding/dplm/requirements.txt"
    )


def test_byprot_wrapper_imports_encoder_from_dplm_refiner():
    wrapper_path = (
        PROJECT_ROOT
        / "inverse_folding/dplm/src/byprot/models/dplm/modules/geoegnn_ipa_encoder.py"
    )
    text = wrapper_path.read_text()
    assert (
        "from inverse_folding.dplm_refiner.geo_encoder.encoder import"
        in text
    )
    assert 'register_model("geoegnn_ipa_encoder")' in text


# ── Runtime tests (PyG-gated) ────────────────────────────────────────────


def _require_pyg_stack():
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")


def _build_tiny_encoder():
    from inverse_folding.dplm_refiner.geo_encoder import GeoEGNNIPAEncoder

    return GeoEGNNIPAEncoder(
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


def test_checkpoint_save_load_roundtrip_restores_state_exactly(tmp_path):
    _require_pyg_stack()
    import torch

    from inverse_folding.dplm_refiner.geo_encoder.checkpoint import (
        load_geo_encoder_checkpoint,
        save_geo_encoder_checkpoint,
    )

    src = _build_tiny_encoder()
    src.eval()

    ckpt_path = tmp_path / "geo_encoder.pt"
    save_geo_encoder_checkpoint(
        ckpt_path,
        src,
        adapter_config={"adapter_num_layers": 1, "adapter_gated": False},
        extra={"epoch": 7, "val_acc": 0.42},
    )
    assert ckpt_path.is_file()

    loaded, report = load_geo_encoder_checkpoint(ckpt_path)
    from inverse_folding.dplm_refiner.geo_encoder.checkpoint import (
        FORMAT_VERSION,
    )

    assert report.format_version == FORMAT_VERSION
    assert report.extra == {"epoch": 7, "val_acc": 0.42}
    assert report.missing_keys == []
    assert report.unexpected_keys == []
    # State match
    for (n_s, p_s), (n_l, p_l) in zip(
        src.named_parameters(), loaded.named_parameters()
    ):
        assert n_s == n_l
        assert torch.allclose(p_s, p_l)


def test_checkpoint_load_into_existing_encoder_reports_missing_keys(tmp_path):
    """Genuine key mismatch (different parameter NAME set, same shapes).

    NB: ``load_state_dict(strict=False)`` only suppresses the missing /
    unexpected KEY error — same-name tensors with different SHAPES
    still raise a ``RuntimeError``. So we exercise the key-set diff by
    saving with ``output_logits=True`` (has ``draft_head.*``) and
    loading into an encoder built with ``output_logits=False`` (no
    draft head), which makes ``draft_head.*`` strictly unexpected.
    """
    _require_pyg_stack()

    from inverse_folding.dplm_refiner.geo_encoder import GeoEGNNIPAEncoder
    from inverse_folding.dplm_refiner.geo_encoder.checkpoint import (
        load_geo_encoder_checkpoint,
        save_geo_encoder_checkpoint,
    )

    src = GeoEGNNIPAEncoder(
        d_model=16,
        output_logits=True,
        vocab_size=33,
        egnn_depth=1,
        egnn_hidden_dim=8,
        ipa_depth=1,
        ipa_hidden_dim=8,
        ipa_pairwise_dim=8,
        ipa_heads=2,
    )
    ckpt = tmp_path / "src.pt"
    save_geo_encoder_checkpoint(ckpt, src)
    target = GeoEGNNIPAEncoder(
        d_model=16,
        output_logits=False,   # no draft head
        egnn_depth=1,
        egnn_hidden_dim=8,
        ipa_depth=1,
        ipa_hidden_dim=8,
        ipa_pairwise_dim=8,
        ipa_heads=2,
    )
    _, report = load_geo_encoder_checkpoint(
        ckpt, encoder=target, strict_state=False
    )
    assert any("draft_head" in k for k in report.unexpected_keys), (
        f"expected draft_head.* in unexpected_keys; got "
        f"{report.unexpected_keys}"
    )


def test_checkpoint_shape_mismatch_raises_with_strict_false():
    """Document the PyTorch contract: ``load_state_dict(strict=False)``
    suppresses MISSING-key errors but still raises ``RuntimeError`` on
    same-name SHAPE mismatch. The previous test wrongly expected a
    silent report here.
    """
    _require_pyg_stack()
    import pytest as _pt

    from inverse_folding.dplm_refiner.geo_encoder import GeoEGNNIPAEncoder
    from inverse_folding.dplm_refiner.geo_encoder.checkpoint import (
        load_geo_encoder_checkpoint,
        save_geo_encoder_checkpoint,
    )

    src = _build_tiny_encoder()
    ckpt_path = (
        Path(__import__("tempfile").mkdtemp(prefix="ckpt_shape_"))
        / "ckpt.pt"
    )
    save_geo_encoder_checkpoint(ckpt_path, src)
    target = GeoEGNNIPAEncoder(
        d_model=8,                # << differs from src (32)
        output_logits=True,
        egnn_depth=1,
        egnn_hidden_dim=16,
        ipa_depth=1,
        ipa_hidden_dim=16,
        ipa_pairwise_dim=16,
        ipa_heads=2,
    )
    with _pt.raises(RuntimeError, match="size mismatch"):
        load_geo_encoder_checkpoint(
            ckpt_path, encoder=target, strict_state=False
        )


def test_checkpoint_format_version_is_present():
    from inverse_folding.dplm_refiner.geo_encoder.checkpoint import (
        FORMAT_VERSION,
    )

    assert FORMAT_VERSION >= 2  # bumped when adapter_state_dict was added


def test_encoder_constructor_kwargs_captures_non_default_fields(tmp_path):
    """P1.5 regression: every non-default constructor knob must survive
    checkpoint round-trip (egnn_message_dim, update_global, norm_coors,
    ipa_pairwise_dim, ipa_heads, dropout, ipa_qk_points, ipa_v_points).
    """
    _require_pyg_stack()

    from inverse_folding.dplm_refiner.geo_encoder import GeoEGNNIPAEncoder
    from inverse_folding.dplm_refiner.geo_encoder.checkpoint import (
        encoder_constructor_kwargs,
    )

    enc = GeoEGNNIPAEncoder(
        d_model=64,
        output_logits=True,
        vocab_size=33,
        egnn_depth=2,
        egnn_hidden_dim=24,
        egnn_message_dim=10,
        update_coors=True,
        update_global=False,
        norm_coors=False,
        egnn_dropout=0.3,
        ipa_depth=1,
        ipa_hidden_dim=32,
        ipa_pairwise_dim=48,
        ipa_heads=2,
        ipa_qk_points=3,
        ipa_v_points=5,
        ipa_dropout=0.4,
        use_updated_coord_bias=True,
    )
    cfg = encoder_constructor_kwargs(enc)
    for key, val in [
        ("egnn_message_dim", 10),
        ("update_global", False),
        ("norm_coors", False),
        ("egnn_dropout", 0.3),
        ("ipa_pairwise_dim", 48),
        ("ipa_heads", 2),
        ("ipa_qk_points", 3),
        ("ipa_v_points", 5),
        ("ipa_dropout", 0.4),
        ("use_updated_coord_bias", True),
    ]:
        assert cfg[key] == val, (
            f"encoder_constructor_kwargs lost {key}; got {cfg[key]!r}"
        )


def test_checkpoint_round_trip_preserves_non_default_dims(tmp_path):
    """Build an encoder with non-default ``ipa_pairwise_dim != ipa_hidden_dim``
    and confirm the round-tripped state dict matches exactly — guards
    P1.4 (edge pair encoder uses ipa_pairwise_dim) and P1.5 together.
    """
    _require_pyg_stack()
    import torch

    from inverse_folding.dplm_refiner.geo_encoder import GeoEGNNIPAEncoder
    from inverse_folding.dplm_refiner.geo_encoder.checkpoint import (
        load_geo_encoder_checkpoint,
        save_geo_encoder_checkpoint,
    )

    src = GeoEGNNIPAEncoder(
        d_model=32,
        output_logits=False,
        egnn_depth=1,
        egnn_hidden_dim=16,
        ipa_depth=1,
        ipa_hidden_dim=24,
        ipa_pairwise_dim=40,  # != ipa_hidden_dim
        ipa_heads=2,
    )
    ckpt = tmp_path / "rt.pt"
    save_geo_encoder_checkpoint(ckpt, src)
    loaded, report = load_geo_encoder_checkpoint(ckpt)
    assert report.missing_keys == []
    assert report.unexpected_keys == []
    # Spot-check a few non-default tensors.
    for n in ("ipa.edge_pair_encoder.rbf_in_proj.weight", "out_proj.weight"):
        s = dict(src.named_parameters())[n]
        l = dict(loaded.named_parameters())[n]
        assert torch.allclose(s, l)


def test_graph_config_accepts_dict_and_dictconfig_at_encoder_construction():
    """P2.10 regression: Hydra passes a DictConfig for ``graph_config``;
    plain dict should also work; both must coerce to a real GraphConfig
    so ``.validate()`` succeeds and dataclass attribute access works."""
    _require_pyg_stack()
    pytest.importorskip("omegaconf")

    from omegaconf import OmegaConf

    from inverse_folding.dplm_refiner.geo_encoder import (
        GeoEGNNIPAEncoder,
        GraphConfig,
    )

    enc_dict = GeoEGNNIPAEncoder(
        graph_config={"k_neighbors": 4, "cutoff": 12.0, "pos_enc_dim": 8},
        d_model=32,
        egnn_depth=1,
        egnn_hidden_dim=16,
        ipa_depth=1,
        ipa_hidden_dim=16,
        ipa_pairwise_dim=16,
        ipa_heads=2,
    )
    assert isinstance(enc_dict.graph_config, GraphConfig)
    assert enc_dict.graph_config.k_neighbors == 4
    assert enc_dict.graph_config.pos_enc_dim == 8

    dc = OmegaConf.create({"k_neighbors": 6, "cutoff": 25.0})
    enc_dc = GeoEGNNIPAEncoder(
        graph_config=dc,
        d_model=32,
        egnn_depth=1,
        egnn_hidden_dim=16,
        ipa_depth=1,
        ipa_hidden_dim=16,
        ipa_pairwise_dim=16,
        ipa_heads=2,
    )
    assert isinstance(enc_dc.graph_config, GraphConfig)
    assert enc_dc.graph_config.k_neighbors == 6


def test_checkpoint_adapter_shape_mismatch_fails_fast_by_default(
    tmp_path, monkeypatch
):
    """If the checkpoint was trained with e.g. last-4 gated adapters
    but the target decoder has last-1 ungated, silent partial loading
    would drop the trained weights. ``load_geo_encoder_checkpoint``
    must fail-fast unless the caller passes
    ``allow_adapter_shape_mismatch=True``.
    """
    _require_pyg_stack()
    import torch
    import torch.nn as nn

    from inverse_folding.dplm_refiner.geo_encoder import GeoEGNNIPAEncoder
    from inverse_folding.dplm_refiner.geo_encoder import checkpoint as ckpt_mod
    from inverse_folding.dplm_refiner.geo_encoder.checkpoint import (
        load_geo_encoder_checkpoint,
        save_geo_encoder_checkpoint,
    )

    enc = GeoEGNNIPAEncoder(
        d_model=16,
        egnn_depth=1,
        egnn_hidden_dim=8,
        ipa_depth=1,
        ipa_hidden_dim=8,
        ipa_pairwise_dim=8,
        ipa_heads=2,
    )

    # Fake decoder with a realistic ESM-like structure: net.esm.encoder
    # .layer is a ModuleList of layers, where some are "AdapterLayer"
    # (class name matching is how the guard introspects).
    #
    # NB: parameters must be ``adapter_*``-prefixed because
    # ``_filter_adapter_state`` only saves keys containing ``adapter`` —
    # an earlier draft of this test used ``self.dummy`` and the saved
    # adapter state came out empty, masking the very mismatch we want
    # to assert.
    class _AdapterLayer(nn.Module):
        adapter_gated = False

        def __init__(self):
            super().__init__()
            self.adapter_proj = nn.Linear(2, 2)

    class _GatedAdapterLayer(nn.Module):
        adapter_gated = True

        def __init__(self):
            super().__init__()
            self.adapter_proj = nn.Linear(2, 2)
            self.adapter_gate = nn.Parameter(torch.zeros(1))

    # IMPORTANT: __name__ must literally be "AdapterLayer" to match the
    # checkpoint guard's class-name introspection.
    _AdapterLayer.__name__ = "AdapterLayer"
    _GatedAdapterLayer.__name__ = "AdapterLayer"

    class _PlainLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(2, 2)

    class _Encoder(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layer = nn.ModuleList(layers)

    class _Esm(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.encoder = _Encoder(layers)

    class _Net(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.esm = _Esm(layers)

    class _Decoder(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.net = _Net(layers)
            self.adapter_proj = nn.Linear(4, 4)
            self.cfg = type(
                "_Cfg",
                (),
                {
                    "adapter_num_layers": 1,
                    "adapter_gated": False,
                    "adapter_gate_init": 0.0,
                },
            )()

    # Trained shape: 4 layers, gated.
    train_layers = [_PlainLayer(), _PlainLayer()] + [
        _GatedAdapterLayer() for _ in range(4)
    ]
    train_decoder = _Decoder(train_layers)

    ckpt = tmp_path / "shape_mismatch.pt"
    save_geo_encoder_checkpoint(
        ckpt,
        enc,
        decoder=train_decoder,
        adapter_config={
            "adapter_num_layers": 4,
            "adapter_gated": True,
            "adapter_gate_init": 0.0,
        },
    )

    # Inference target: only 1 ungated adapter.
    target_layers = [_PlainLayer() for _ in range(5)] + [_AdapterLayer()]
    target_decoder = _Decoder(target_layers)

    with pytest.raises(ValueError, match="adapter shape mismatch"):
        load_geo_encoder_checkpoint(ckpt, decoder=target_decoder)

    # Opt-out flag bypasses the guard (silently partial-loads).
    _, report = load_geo_encoder_checkpoint(
        ckpt, decoder=target_decoder, allow_adapter_shape_mismatch=True
    )
    # Some adapter keys are unexpected (the 3 dropped gated layers).
    assert report.adapter_unexpected_keys

    def _fake_reinstall(decoder, saved_cfg):
        decoder.net.esm.encoder.layer = nn.ModuleList(
            [_PlainLayer(), _PlainLayer()]
            + [_GatedAdapterLayer() for _ in range(4)]
        )
        decoder.cfg.adapter_num_layers = int(saved_cfg["adapter_num_layers"])
        decoder.cfg.adapter_gated = bool(saved_cfg["adapter_gated"])
        decoder.cfg.adapter_gate_init = float(saved_cfg["adapter_gate_init"])
        return True

    monkeypatch.setattr(
        ckpt_mod, "_reinstall_decoder_adapter_shape", _fake_reinstall
    )
    target_auto = _Decoder(
        [_PlainLayer() for _ in range(5)] + [_AdapterLayer()]
    )
    _, report = load_geo_encoder_checkpoint(
        ckpt, decoder=target_auto, auto_install_adapter_shape=True
    )
    assert report.adapter_unexpected_keys == []
    assert target_auto.cfg.adapter_num_layers == 4
    assert target_auto.cfg.adapter_gated is True


def test_train_if_imp_encoder_does_not_inject_mask_or_unk_into_specials():
    """P2 regression: the training script must not pass mask_idx /
    unk_idx into the encoder's ``special_token_ids``; doing so would
    persist into the saved checkpoint and risk dropping all-``<mask>``
    positions out of the graph at any downstream caller that derived
    the special mask from ``prev_tokens``.
    """
    project_root = Path(__file__).resolve().parents[3]
    text = (project_root / "scripts/train_if_imp_encoder.py").read_text()
    # The function building the special-token set must NOT enumerate
    # mask_idx / unk_idx.
    assert "alphabet.cls_idx" in text
    assert "alphabet.padding_idx" in text
    assert "alphabet.eos_idx" in text
    # These two must be absent from the special_token_ids builder.
    # (They may legitimately appear elsewhere in the file as part of
    # comments or other logic; constrain the search to a window
    # around the special_token_ids assignment.)
    builder_idx = text.find("special_token_ids = sorted(")
    assert builder_idx >= 0, "special_token_ids builder not found"
    builder_end = text.find(")", builder_idx + 100)
    builder_block = text[builder_idx : builder_end + 1]
    assert "mask_idx" not in builder_block, builder_block
    assert "unk_idx" not in builder_block, builder_block


def test_checkpoint_adapter_state_round_trip_preserves_adapter_deltas(tmp_path):
    """P1.3 regression: training fine-tunes adapter parameters; the
    checkpoint must capture them and ``load_geo_encoder_checkpoint(...,
    decoder=...)`` must restore them so generation paths don't silently
    drop the adapter fine-tune.
    """
    _require_pyg_stack()
    import torch
    import torch.nn as nn

    from inverse_folding.dplm_refiner.geo_encoder import GeoEGNNIPAEncoder
    from inverse_folding.dplm_refiner.geo_encoder.checkpoint import (
        load_geo_encoder_checkpoint,
        save_geo_encoder_checkpoint,
    )

    enc = GeoEGNNIPAEncoder(
        d_model=16,
        egnn_depth=1,
        egnn_hidden_dim=8,
        ipa_depth=1,
        ipa_hidden_dim=8,
        ipa_pairwise_dim=8,
        ipa_heads=2,
    )

    # Build a minimal fake "decoder" with adapter-named params + frozen
    # non-adapter params; the checkpoint helper must save only the
    # adapter-named subset and the loader must reapply it.
    class _FakeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter_proj = nn.Linear(4, 4)
            self.adapter_gate = nn.Parameter(torch.tensor(0.5))
            self.frozen_backbone = nn.Linear(4, 4)

    dec_src = _FakeDecoder()
    with torch.no_grad():
        dec_src.adapter_proj.weight.fill_(1.234)
        dec_src.adapter_gate.fill_(0.777)
        dec_src.frozen_backbone.weight.fill_(9.999)

    ckpt = tmp_path / "with_adapter.pt"
    save_geo_encoder_checkpoint(ckpt, enc, decoder=dec_src)

    dec_dst = _FakeDecoder()
    with torch.no_grad():
        dec_dst.adapter_proj.weight.fill_(0.0)
        dec_dst.adapter_gate.fill_(0.0)
        dec_dst.frozen_backbone.weight.fill_(0.0)

    _, report = load_geo_encoder_checkpoint(ckpt, decoder=dec_dst)
    assert torch.allclose(dec_dst.adapter_proj.weight.mean(), torch.tensor(1.234))
    assert float(dec_dst.adapter_gate.item()) == pytest.approx(0.777)
    # Non-adapter param must NOT be overwritten (we filter adapter-only).
    assert float(dec_dst.frozen_backbone.weight.mean().item()) == 0.0
    # report should not flag adapter as unexpected.
    assert report.adapter_unexpected_keys == []


def test_byprot_registry_resolves_geoegnn_ipa_encoder():
    """End-to-end: byprot's ``utils.instantiate_from_config`` should
    pick up the ``geoegnn_ipa_encoder`` registry name and return a
    callable encoder instance. This exercises the auto-import path that
    the Hydra ``_target_`` field relies on.
    """
    pytest.importorskip("omegaconf")
    pytest.importorskip("hydra")
    _require_pyg_stack()
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "inverse_folding" / "dplm" / "src"))
    from byprot import utils
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "geoegnn_ipa_encoder",
            "d_model": 32,
            "egnn_depth": 1,
            "egnn_hidden_dim": 16,
            "ipa_depth": 1,
            "ipa_hidden_dim": 16,
            "ipa_pairwise_dim": 16,
            "ipa_heads": 2,
        }
    )
    model = utils.instantiate_from_config(cfg=cfg, group="model")
    assert model.__class__.__name__ == "GeoEGNNIPAEncoder"
    assert getattr(model, "d_model", None) == 32
