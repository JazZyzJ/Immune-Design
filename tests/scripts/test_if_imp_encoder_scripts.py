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
        "--val-batch-size",
        "--val-eval-every-epochs",
        "--pretrain-aux-only-epochs",
        "--pretrain-target",
        "--resume-from-ckpt",
        "--val-max-iters",
        "--train-recovery-on-val",
    ):
        assert flag in result.stdout, f"missing flag {flag}"


def test_run_if_imp_refiner_exposes_encoder_flags():
    result = _run_help("scripts/run_if_imp_refiner.py")
    assert result.returncode == 0, result.stderr
    assert "--encoder-checkpoint" in result.stdout
    assert "--encoder-kind" in result.stdout
    assert "geoegnn_ipa" in result.stdout


def test_encoder_load_paths_auto_install_checkpoint_adapter_shape():
    """Generation and diagnostics must restore last-N/gated adapter shapes.

    The checkpoint loader still fails fast by default; these production
    entrypoints opt into rebuilding the decoder adapter shape from the
    checkpoint before loading adapter_state_dict.
    """
    run_text = (ROOT / "scripts/run_if_imp_refiner.py").read_text()
    diag_text = (ROOT / "scripts/diag_if_imp_arms.py").read_text()
    assert "auto_install_adapter_shape=True" in run_text
    assert "auto_install_adapter_shape=True" in diag_text


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


def test_submit_if_imp_train_encoder_forwards_wandb_args():
    """Encoder training must honor the launcher's WANDB_* env contract."""
    text = (ROOT / "scripts/submit_if_imp.slurm").read_text()
    start = text.index('elif [ "${MODE}" = "train_encoder" ]')
    end = text.index('elif [ "${MODE}" = "generate_encoder" ]', start)
    body = text[start:end]
    assert "train_if_imp_encoder.py" in body
    assert '"${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}"' in body


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


def test_train_step_bypasses_task_model_forward_and_uses_native_target():
    """PLAN_IF_ENCODER.md Task E7 source-level guards.

    The training step must:
    1. NOT route through ``task.model(...)`` with
       ``output_encoder_logits=True`` — that path rewrites the
       diffusion input from native to encoder.argmax.
    2. Call ``decoder.compute_loss(..., tokens=None, ...)`` so the
       returned target AND the internal ``x_t`` / ``loss_mask`` derive
       from native tokens.
    3. Compute aux CE on ``seq_mask`` / ``encoder_attention_mask`` —
       NOT on ``loss_mask`` (the diffusion-masked subset).
    """
    text = (ROOT / "scripts/train_if_imp_encoder.py").read_text()

    # Locate train_step body specifically (excludes train_step_aux_only).
    train_step_marker = "def train_step(\n"
    aux_only_marker = "def train_step_aux_only("
    assert train_step_marker in text
    start = text.index(train_step_marker)
    end = text.index(aux_only_marker)
    body = text[start:end]

    # Drop the docstring before checking — it intentionally explains
    # the old broken pattern and mentions ``output_encoder_logits=True``.
    code_only = body.split('"""', 2)[-1]

    # The forbidden CALL pattern (not docstring mention) is going
    # through ``task.model(...)`` rather than ``task.model.encoder(...)``.
    assert "task.model(" not in code_only.replace(
        "task.model.encoder(", ""
    ).replace("task.model.decoder", ""), (
        "train_step still routes through task.model(...) — PLAN E7 "
        "requires bypassing forward"
    )
    assert "output_encoder_logits=True" not in code_only, (
        "train_step's code body still references output_encoder_logits "
        "(only the docstring may mention it explaining the old bug)"
    )
    assert "task.model.encoder(" in code_only, (
        "train_step must call task.model.encoder(...) directly"
    )
    assert "tokens=None" in code_only, (
        "decoder.compute_loss must be called with tokens=None to keep "
        "the diffusion target/input on native batch[\"tokens\"]"
    )
    assert "compute_loss(" in code_only
    # Aux CE must be derived from the encoder's seq_mask, not loss_mask.
    assert "_aux_loss_and_recovery" in code_only
    assert "seq_mask = encoder_out[\"encoder_attention_mask\"]" in code_only


def test_aux_loss_helper_uses_seq_mask_not_loss_mask():
    """The aux-loss helper that both train_step and train_step_aux_only
    delegate to must compute CE over seq_mask, not loss_mask."""
    text = (ROOT / "scripts/train_if_imp_encoder.py").read_text()
    helper = "_aux_loss_and_recovery"
    assert helper in text
    start = text.index(f"def {helper}(")
    # End at the next top-level def.
    end = text.index("\n\ndef ", start + len(helper))
    helper_body = text[start:end]
    assert "encoder_logits[seq_mask]" in helper_body
    assert "native_tokens[seq_mask]" in helper_body
    assert "loss_mask" not in helper_body, (
        "_aux_loss_and_recovery must NOT reference loss_mask — that "
        "would re-introduce the sparse-supervision bug PLAN E7 fixes"
    )


def test_train_step_aux_only_does_not_call_decoder():
    """The pre-stage aux-only step must not invoke the DPLM decoder
    (decoder stays frozen during pretrain)."""
    text = (ROOT / "scripts/train_if_imp_encoder.py").read_text()
    aux_only_marker = "def train_step_aux_only("
    val_marker = "def val_step("
    assert aux_only_marker in text
    start = text.index(aux_only_marker)
    end = text.index(val_marker)
    body = text[start:end]
    assert "compute_loss(" not in body, (
        "train_step_aux_only must not invoke decoder.compute_loss"
    )
    assert "task.model.decoder" not in body, (
        "train_step_aux_only must not touch task.model.decoder"
    )


def test_main_loop_writes_encoder_best_and_runs_val_step():
    """Main loop must save encoder_best.pt on val improvement and
    call val_step at the configured cadence."""
    text = (ROOT / "scripts/train_if_imp_encoder.py").read_text()
    assert "encoder_best.pt" in text
    assert "val_step(" in text
    # The pretrain early-exit on --pretrain-target must be present.
    assert "pretrain_target" in text
    assert "pretrain early-exit" in text.lower() or "pretrain-target" in text


def test_metric_keys_use_train_and_val_prefixes():
    """All trainable metrics emitted by ``train_step`` /
    ``train_step_aux_only`` / ``val_step`` must use ``train/`` and
    ``val/`` prefixes — no mixed namespace in wandb."""
    text = (ROOT / "scripts/train_if_imp_encoder.py").read_text()
    # Train-side metric keys.
    for key in (
        '"train/loss"',
        '"train/main_loss"',
        '"train/aux_loss"',
        '"train/draft_recovery"',
        '"train/feats_std"',
        '"train/n_aux_positions"',
        '"train/n_main_positions"',
    ):
        assert key in text, f"missing train-prefixed metric key {key}"
    # Val-side metric keys (some are emitted via _recovery_eval's
    # ``prefix`` arg).
    assert 'prefix="val/"' in text
    assert "full_recovery_iter" in text


def test_resume_from_ckpt_wires_optimizer_pt_and_load_checkpoint():
    text = (ROOT / "scripts/train_if_imp_encoder.py").read_text()
    # The flag itself is parsed.
    assert "--resume-from-ckpt" in text
    # The resume branch loads via load_geo_encoder_checkpoint with
    # adapter shape rebuild + matching optimizer.pt state.
    assert "load_geo_encoder_checkpoint" in text
    assert "auto_install_adapter_shape=True" in text
    assert "optimizer.pt" in text
    # And the train loop starts from ``start_epoch``.
    assert "for epoch in range(start_epoch, int(args.epochs))" in text
    # Resume should also keep W&B / metrics step axes continuous.
    assert "start_step = int(report.extra.get(\"step\", 0) or 0)" in text
    assert "step_idx = start_step" in text


def test_val_step_runs_at_each_max_iter():
    """val_step is wired to ``_recovery_eval`` which generates at every
    configured ``max_iter`` and emits ``val/full_recovery_iter{N}``."""
    text = (ROOT / "scripts/train_if_imp_encoder.py").read_text()
    assert "max_iters=val_max_iters" in text
    assert "full_recovery_iter" in text
    assert "task.model.generate(" in text


def test_submit_if_imp_slurm_forwards_resume_and_val_iter_flags():
    text = (ROOT / "scripts/submit_if_imp.slurm").read_text()
    for env in (
        "ENCODER_RESUME_FROM_CKPT",
        "ENCODER_VAL_MAX_ITERS",
        "ENCODER_TRAIN_RECOVERY_ON_VAL",
    ):
        assert env in text, f"submit_if_imp.slurm missing env {env}"
    for cli in ("--val-max-iters", "--resume-from-ckpt", "--train-recovery-on-val"):
        assert cli in text, f"submit_if_imp.slurm missing CLI {cli}"


def _slurm_branch(text: str, marker: str) -> str:
    """Return the slice of submit_if_imp.slurm for one MODE branch."""
    start = text.index(marker)
    # Branch ends at the next ``elif [ "${MODE}"`` or the final ``else``.
    rest = text[start + len(marker):]
    candidates = [
        rest.find('elif [ "${MODE}"'),
        rest.find("\nelse\n"),
    ]
    ends = [c for c in candidates if c >= 0]
    end = min(ends) if ends else len(rest)
    return rest[:end]


def test_generate_encoder_branch_forwards_full_generation_params():
    """P1 regression: generate_encoder must not silently drop generation
    parameters. Mirror the generate_refiner passthrough."""
    text = (ROOT / "scripts/submit_if_imp.slurm").read_text()
    branch = _slurm_branch(text, 'elif [ "${MODE}" = "generate_encoder" ]; then')
    for flag in (
        "--n-designs-per-protein",
        "--temperature",
        "--batch-size",
        "--mc-dropout-passes",
        "--fusion-mode",
        "--fusion-alpha",
        "--apply-steps",
        "--sidecar-scale",
        "--tag",
        "--cath-max-length",
        "--cath-min-resolved-ratio",
        "--encoder-checkpoint",
        "--encoder-kind",
    ):
        assert flag in branch, f"generate_encoder branch missing {flag}"


def test_diag_encoder_branch_forwards_cath_filters_and_diag_params():
    """P1 regression: diag_encoder must forward CATH filters (so the
    diagnostic protein set matches) and the diagnostic knobs."""
    text = (ROOT / "scripts/submit_if_imp.slurm").read_text()
    branch = _slurm_branch(text, 'elif [ "${MODE}" = "diag_encoder" ]; then')
    for flag in (
        "--cath-max-length",
        "--cath-min-resolved-ratio",
        "--temperature",
        "--mc-dropout-passes",
        "--mask-ratio-center",
        "--mask-ratio-deviation",
        "--fusion-temperature",
        "--fusion-mode",
        "--fusion-alpha",
        "--apply-steps",
        "--progress-every",
        "--encoder-checkpoint",
        "--encoder-kind",
    ):
        assert flag in branch, f"diag_encoder branch missing {flag}"


def test_resume_loads_checkpoint_before_optimizer_construction():
    """P1 regression: ``load_geo_encoder_checkpoint`` (which may reinstall
    adapter modules via auto_install_adapter_shape) must run BEFORE the
    optimizer is constructed, otherwise the optimizer holds stale param
    references and never updates a reshaped adapter."""
    text = (ROOT / "scripts/train_if_imp_encoder.py").read_text()
    # Restrict to main() so we don't match the import inside the loader.
    main_start = text.index("def main(")
    main_body = text[main_start:]
    load_idx = main_body.index("load_geo_encoder_checkpoint(\n")
    optim_idx = main_body.index("torch.optim.AdamW(")
    assert load_idx < optim_idx, (
        "load_geo_encoder_checkpoint must precede optimizer construction "
        "so adapter-shape reinstall doesn't strand stale param refs"
    )
    # The optimizer-state restore (optimizer.pt) must come AFTER the
    # optimizer exists.
    opt_state_idx = main_body.index("optimizer.load_state_dict(")
    assert optim_idx < opt_state_idx
