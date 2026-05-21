"""Tests for gated / multi-layer adapter installation in
``byprot.models.dplm.modules.dplm_adapter``.

The module's top-level imports pull in the full DPLM/byprot/omegaconf
runtime (only present in the ``immune-design`` cluster env). For local
checks we use AST/source inspection (as in ``test_dplm_hook.py``); the
runtime tests are guarded by ``pytest.importorskip`` so they skip when
omegaconf is missing and run on the cluster.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_SRC = (
    PROJECT_ROOT
    / "inverse_folding/dplm/src/byprot/models/dplm/modules/dplm_adapter.py"
)
INVFOLD_SRC = (
    PROJECT_ROOT / "inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py"
)


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text())


# ── AST / source-level checks (run locally without DPLM deps) ────────────


def test_adapter_config_exposes_new_fields():
    tree = _parse(ADAPTER_SRC)
    cfg_cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DPLMWithAdapterConfig"
    )
    field_names = [
        stmt.target.id
        for stmt in cfg_cls.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]
    for required in (
        "adapter_num_layers",
        "adapter_gated",
        "adapter_gate_init",
    ):
        assert required in field_names, (
            f"{required} missing from DPLMWithAdapterConfig; got {field_names}"
        )


def test_install_adapters_helper_exists_at_module_level():
    tree = _parse(ADAPTER_SRC)
    func_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert "install_adapters" in func_names, (
        f"install_adapters helper missing; module-level funcs={func_names}"
    )


def test_from_pretrained_uses_install_adapters_loop():
    """from_pretrained must delegate to install_adapters; the old
    single-line ``net.esm.encoder.layer[-1] = adapter`` pattern must be
    gone from the classmethod body."""
    tree = _parse(ADAPTER_SRC)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "DPLMWithConditionalAdatper"
    )
    fp = next(
        f for f in cls.body
        if isinstance(f, ast.FunctionDef) and f.name == "from_pretrained"
    )
    src = ast.unparse(fp)
    assert "install_adapters" in src, src
    assert "esm.encoder.layer[-1] = adapter" not in src, src


def test_adapter_layer_reads_gated_flag_and_branches_in_forward():
    text = ADAPTER_SRC.read_text()
    assert "adapter_gated" in text
    assert "self.adapter_gate" in text
    assert "if self.adapter_gated" in text


def test_invfold_config_has_detach_encoder_feats():
    tree = _parse(INVFOLD_SRC)
    cfg_cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DPLMInvFoldConfig"
    )
    field_names = [
        stmt.target.id
        for stmt in cfg_cls.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]
    assert "detach_encoder_feats" in field_names, field_names


def test_invfold_forward_uses_conditional_detach():
    """The forward() path must no longer call ``.repeat(...).detach()``
    unconditionally; instead it must read ``detach_encoder_feats`` and
    gate the detach."""
    text = INVFOLD_SRC.read_text()
    assert ".repeat(2, 1, 1).detach()" not in text, (
        "found old unconditional detach; conditional gate not applied"
    )
    assert "detach_encoder_feats" in text
    assert ".detach()" in text


def test_invfold_forward_encoder_sets_coord_mask_in_both_branches():
    text = INVFOLD_SRC.read_text()
    assert '"coord_mask" not in encoder_out' in text, (
        "forward_encoder must ensure coord_mask in both draft and "
        "non-draft branches"
    )


# ── Runtime checks (skip when omegaconf / transformers missing) ──────────


def _import_dplm_adapter():
    pytest.importorskip("omegaconf")
    pytest.importorskip("transformers")
    pytest.importorskip("hydra")
    sys.path.insert(0, str(PROJECT_ROOT / "inverse_folding" / "dplm" / "src"))
    from byprot.models.dplm.modules import dplm_adapter

    return dplm_adapter


def _small_esm_config():
    from transformers import EsmConfig

    return EsmConfig(
        vocab_size=33,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        layer_norm_eps=1e-5,
    )


class _FakeEncoderLayers:
    def __init__(self, layer):
        self.layer = layer


class _FakeEsm:
    def __init__(self, layer):
        self.encoder = _FakeEncoderLayers(layer)


class _FakeNet:
    def __init__(self, layers, config):
        self.esm = _FakeEsm(layers)
        self.config = config


def _build_fake_net(n_layers: int):
    import torch.nn as nn
    from transformers.models.esm.modeling_esm import EsmLayer

    config = _small_esm_config()
    layers = nn.ModuleList([EsmLayer(config) for _ in range(n_layers)])
    return _FakeNet(layers, config)


def _build_cfg(**overrides):
    mod = _import_dplm_adapter()
    from omegaconf import OmegaConf

    base = OmegaConf.structured(mod.DPLMWithAdapterConfig)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.create(overrides))
    return base


def test_install_adapters_default_replaces_only_last_layer():
    mod = _import_dplm_adapter()
    net = _build_fake_net(n_layers=4)
    cfg = _build_cfg()
    mod.install_adapters(net, cfg, adapter_num_layers=1)
    layers = net.esm.encoder.layer
    assert isinstance(layers[-1], mod.AdapterLayer)
    for i in range(len(layers) - 1):
        assert not isinstance(layers[i], mod.AdapterLayer)


def test_install_adapters_replaces_last_n_layers():
    mod = _import_dplm_adapter()
    net = _build_fake_net(n_layers=6)
    cfg = _build_cfg(adapter_num_layers=3)
    mod.install_adapters(net, cfg, adapter_num_layers=3)
    layers = net.esm.encoder.layer
    assert all(
        isinstance(layers[i], mod.AdapterLayer) for i in range(3, 6)
    )
    assert all(
        not isinstance(layers[i], mod.AdapterLayer) for i in range(0, 3)
    )


def test_install_adapters_rejects_zero_layers():
    mod = _import_dplm_adapter()
    net = _build_fake_net(n_layers=2)
    cfg = _build_cfg()
    with pytest.raises(ValueError, match=">= 1"):
        mod.install_adapters(net, cfg, adapter_num_layers=0)


def test_install_adapters_rejects_too_many_layers():
    mod = _import_dplm_adapter()
    net = _build_fake_net(n_layers=2)
    cfg = _build_cfg()
    with pytest.raises(ValueError, match="exceeds"):
        mod.install_adapters(net, cfg, adapter_num_layers=5)


def test_ungated_adapter_layer_has_no_gate_parameter():
    mod = _import_dplm_adapter()
    config = _small_esm_config()
    cfg = _build_cfg()
    layer = mod.AdapterLayer(cfg, config)
    assert layer.adapter_gated is False
    assert not hasattr(layer, "adapter_gate")


def test_gated_adapter_layer_registers_gate_with_init_value():
    import torch.nn as nn

    mod = _import_dplm_adapter()
    config = _small_esm_config()
    cfg = _build_cfg(adapter_gated=True, adapter_gate_init=0.25)
    layer = mod.AdapterLayer(cfg, config)
    assert layer.adapter_gated is True
    assert isinstance(layer.adapter_gate, nn.Parameter)
    assert float(layer.adapter_gate.item()) == pytest.approx(0.25)
    assert layer.adapter_gate.requires_grad


def test_gated_layer_parameter_name_passes_adapter_filter():
    """The trainable-parameter rule ``"adapter" in pname`` must include
    the new ``adapter_gate`` so the loop in from_pretrained does not
    freeze it."""
    mod = _import_dplm_adapter()
    config = _small_esm_config()
    cfg = _build_cfg(adapter_gated=True)
    layer = mod.AdapterLayer(cfg, config)
    names_with_adapter = [
        n for n, _ in layer.named_parameters() if "adapter" in n
    ]
    assert any("adapter_gate" in n for n in names_with_adapter), (
        names_with_adapter
    )
