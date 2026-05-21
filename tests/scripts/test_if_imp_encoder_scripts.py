"""Tests for the GeoEGNN-IPA encoder training script + SLURM hooks.

The full ``main`` path requires GPU + DPLM 650M + PyG; here we cover:

- ``argparse`` --help wiring on ``scripts/train_if_imp_encoder.py``.
- ``--encoder-checkpoint`` / ``--encoder-kind`` flags wired into
  ``run_if_imp_refiner.py`` and ``diag_if_imp_arms.py``.
- ``scripts/submit_if_imp.slurm`` dispatches the three new
  ``train_encoder|generate_encoder|diag_encoder`` modes.
- ``doc/SCRIPTS.md`` registers the new script.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _run_help(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_train_if_imp_encoder_help_lists_key_flags():
    result = _run_help("scripts/train_if_imp_encoder.py")
    assert result.returncode == 0, result.stderr
    for flag in (
        "--cath-root",
        "--checkpoint",
        "--output-dir",
        "--epochs",
        "--lr",
        "--lambda-aux",
        "--adapter-num-layers",
        "--adapter-gated",
        "--adapter-gate-init",
        "--egnn-depth",
        "--egnn-hidden-dim",
        "--ipa-depth",
        "--ipa-hidden-dim",
        "--update-coors",
        "--use-updated-coord-bias",
    ):
        assert flag in result.stdout, f"missing flag {flag}"


def test_run_if_imp_refiner_exposes_encoder_flags():
    result = _run_help("scripts/run_if_imp_refiner.py")
    assert result.returncode == 0, result.stderr
    assert "--encoder-checkpoint" in result.stdout
    assert "--encoder-kind" in result.stdout
    assert "geoegnn_ipa" in result.stdout


def test_diag_if_imp_arms_exposes_encoder_flags():
    result = _run_help("scripts/diag_if_imp_arms.py")
    assert result.returncode == 0, result.stderr
    assert "--encoder-checkpoint" in result.stdout
    assert "--encoder-kind" in result.stdout


def test_submit_if_imp_slurm_has_new_modes():
    text = (ROOT / "scripts/submit_if_imp.slurm").read_text()
    for mode in ("train_encoder", "generate_encoder", "diag_encoder"):
        assert mode in text, f"submit_if_imp.slurm missing MODE={mode}"
    # ENCODER_* env defaults must be present.
    for env in (
        "ENCODER_CHECKPOINT",
        "ENCODER_KIND",
        "ENCODER_BATCH_SIZE",
        "ENCODER_EPOCHS",
        "ENCODER_LR",
        "ENCODER_ADAPTER_NUM_LAYERS",
        "ENCODER_ADAPTER_GATED",
        "ENCODER_ADAPTER_GATE_INIT",
        "ENCODER_EGNN_DEPTH",
        "ENCODER_IPA_DEPTH",
        "ENCODER_LAMBDA_AUX",
    ):
        assert env in text, f"submit_if_imp.slurm missing env {env}"
    # The script must invoke train_if_imp_encoder.py at least once.
    assert "train_if_imp_encoder.py" in text


def test_doc_scripts_registers_train_if_imp_encoder():
    text = (ROOT / "doc/SCRIPTS.md").read_text()
    assert "scripts/train_if_imp_encoder.py" in text
    assert "PLAN_IF_ENCODER.md" in text


def test_doc_scripts_notes_new_slurm_modes():
    text = (ROOT / "doc/SCRIPTS.md").read_text()
    for mode in ("train_encoder", "generate_encoder", "diag_encoder"):
        assert mode in text, f"doc/SCRIPTS.md does not mention MODE={mode}"


def test_train_if_imp_encoder_uses_save_geo_encoder_checkpoint():
    """The training script must persist via the canonical helper so
    downstream loaders can use ``load_geo_encoder_checkpoint``."""
    text = (ROOT / "scripts/train_if_imp_encoder.py").read_text()
    assert "save_geo_encoder_checkpoint" in text
    assert "encoder_last.pt" in text
    assert "run_config.yaml" in text
    assert "metrics.jsonl" in text
    assert "manifest.json" in text


def test_train_if_imp_encoder_reinstalls_adapter_shape_before_freeze():
    text = (ROOT / "scripts/train_if_imp_encoder.py").read_text()
    reinstall = "install_adapters(\n            task.model.decoder.net"
    freeze = "# Freeze the entire DPLM backbone; unfreeze adapter + new encoder."
    assert reinstall in text
    assert freeze in text
    assert text.index(reinstall) < text.index(freeze)
    assert "--adapter-num-layers > 1 requires --adapter-gated" in text


def test_install_geo_encoder_reinstalls_requested_adapter_shape_runtime():
    """PyG-gated runtime check for the training entrypoint.

    Module-K checkpoints may load as last-1 ungated adapters. The
    encoder training script must be able to rebuild that decoder as
    last-N gated before freezing, while preserving the old last-layer
    adapter weights.
    """
    pytest.importorskip("omegaconf")
    pytest.importorskip("transformers")
    pytest.importorskip("hydra")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")

    import argparse
    import importlib

    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf
    from transformers import EsmConfig
    from transformers.models.esm.modeling_esm import EsmLayer

    byprot_src = ROOT / "inverse_folding" / "dplm" / "src"
    if str(byprot_src) not in sys.path:
        sys.path.insert(0, str(byprot_src))
    from byprot.models.dplm.modules import dplm_adapter

    train_mod = importlib.import_module("scripts.train_if_imp_encoder")

    class _EncoderLayers(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layer = layers

    class _Esm(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.encoder = _EncoderLayers(layers)

    class _Net(nn.Module):
        def __init__(self, layers, config):
            super().__init__()
            self.esm = _Esm(layers)
            self.config = config

    class _Decoder(nn.Module):
        def __init__(self, net, cfg):
            super().__init__()
            self.net = net
            self.cfg = cfg

    class _Model(nn.Module):
        def __init__(self, decoder, cfg):
            super().__init__()
            self.decoder = decoder
            self.cfg = cfg
            self.encoder = nn.Linear(1, 1)

    class _Alphabet:
        cls_idx = 0
        padding_idx = 1
        eos_idx = 2

        def __len__(self):
            return 33

    class _Task:
        def __init__(self, model):
            self.model = model
            self.alphabet = _Alphabet()

    esm_cfg = EsmConfig(
        vocab_size=33,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        layer_norm_eps=1e-5,
    )
    layers = nn.ModuleList([EsmLayer(esm_cfg) for _ in range(4)])
    net = _Net(layers, esm_cfg)
    decoder_cfg = OmegaConf.structured(dplm_adapter.DPLMWithAdapterConfig)
    OmegaConf.set_struct(decoder_cfg, False)
    decoder_cfg.encoder_d_model = 32
    dplm_adapter.install_adapters(net, decoder_cfg, adapter_num_layers=1)
    with torch.no_grad():
        net.esm.encoder.layer[-1].adapter_LayerNorm.weight.fill_(3.0)

    decoder = _Decoder(net, decoder_cfg)
    model_cfg = OmegaConf.create(
        {
            "detach_encoder_feats": True,
            "decoder": {
                "adapter_num_layers": 1,
                "adapter_gated": False,
                "adapter_gate_init": 0.0,
            },
        }
    )
    task = _Task(_Model(decoder, model_cfg))
    args = argparse.Namespace(
        d_model=32,
        egnn_depth=1,
        egnn_hidden_dim=16,
        update_coors=False,
        ipa_depth=1,
        ipa_hidden_dim=16,
        ipa_pairwise_dim=16,
        ipa_heads=2,
        use_updated_coord_bias=False,
        adapter_num_layers=3,
        adapter_gated=True,
        adapter_gate_init=0.0,
    )

    train_mod.install_geo_encoder(task=task, args=args, device="cpu")

    rebuilt_layers = task.model.decoder.net.esm.encoder.layer
    assert all(
        isinstance(rebuilt_layers[i], dplm_adapter.AdapterLayer)
        for i in range(1, 4)
    )
    assert not isinstance(rebuilt_layers[0], dplm_adapter.AdapterLayer)
    assert all(bool(rebuilt_layers[i].adapter_gated) for i in range(1, 4))
    assert float(rebuilt_layers[1].adapter_gate.item()) == pytest.approx(0.0)
    assert torch.allclose(
        rebuilt_layers[-1].adapter_LayerNorm.weight,
        torch.full_like(rebuilt_layers[-1].adapter_LayerNorm.weight, 3.0),
    )
    assert int(task.model.decoder.cfg.adapter_num_layers) == 3
    assert bool(task.model.decoder.cfg.adapter_gated) is True
    assert int(task.model.cfg.decoder.adapter_num_layers) == 3
    assert task.model.cfg.detach_encoder_feats is False

    trainable = [
        name for name, param in task.model.named_parameters() if param.requires_grad
    ]
    assert any(name.startswith("encoder.") for name in trainable)
    assert any("adapter_gate" in name for name in trainable)
    assert all(
        name.startswith("encoder.") or "adapter" in name for name in trainable
    )


def test_run_if_imp_refiner_calls_maybe_replace_encoder():
    text = (ROOT / "scripts/run_if_imp_refiner.py").read_text()
    assert "_maybe_replace_encoder" in text
    assert "load_geo_encoder_checkpoint" in text
