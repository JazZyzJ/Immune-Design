#!/usr/bin/env python
"""Train the GeoEGNN-IPA encoder + adapter parameters on CATH.

Replaces DPLM-IF's frozen GVP encoder with :class:`GeoEGNNIPAEncoder`
(see ``PLAN_IF_ENCODER.md`` Task E5). DPLM backbone weights stay
frozen; trainable parameters are the new encoder, its draft head, and
the configured decoder adapters. Old Module-K checkpoints load with
the default last-1 ungated adapter; this script can explicitly reinstall
last-N gated adapters before freezing non-adapter decoder parameters.

Loss is the standard DPLM ``diff_loss + lambda_aux * encoder_loss``
contract from ``byprot.tasks.lm.dplm_invfold``, computed via
``task.model.forward(...)`` so the gradient flows back into the new
encoder through the existing decoder path (this requires
``cfg.detach_encoder_feats=False`` — the script sets it explicitly).
The decoder adapter shape is also an explicit training hyperparameter:
old Module-K checkpoints default to last-1 ungated adapters, while
GeoEGNN-IPA runs may request last-N gated adapters before freezing the
rest of the DPLM backbone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the GeoEGNN-IPA encoder on CATH; DPLM weights stay "
            "frozen and the adapter is fine-tuned alongside the encoder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cath-root", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Module K DPLM checkpoint (.ckpt) used as the frozen prior.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--lambda-aux",
        type=float,
        default=1.0,
        help="Weight on the encoder draft-head CE auxiliary loss.",
    )
    parser.add_argument(
        "--adapter-num-layers",
        type=int,
        default=1,
        help=(
            "Number of final ESM decoder layers to wrap with the "
            "structure-conditioned adapter."
        ),
    )
    parser.add_argument(
        "--adapter-gated",
        action="store_true",
        help=(
            "Use a learned scalar gate for adapter deltas. Required for "
            "--adapter-num-layers > 1 so freshly inserted earlier-layer "
            "adapters start as a zero-contribution branch."
        ),
    )
    parser.add_argument(
        "--adapter-gate-init",
        type=float,
        default=0.0,
        help="Initial scalar gate value used when --adapter-gated is set.",
    )
    # Encoder architecture knobs (mirror geo_encoder.encoder defaults).
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--egnn-depth", type=int, default=3)
    parser.add_argument("--egnn-hidden-dim", type=int, default=128)
    parser.add_argument("--ipa-depth", type=int, default=6)
    parser.add_argument("--ipa-hidden-dim", type=int, default=128)
    parser.add_argument("--ipa-pairwise-dim", type=int, default=128)
    parser.add_argument("--ipa-heads", type=int, default=4)
    parser.add_argument(
        "--update-coors",
        action="store_true",
        help="Let the EGNN update internal latent CA coords (ablation).",
    )
    parser.add_argument(
        "--use-updated-coord-bias",
        action="store_true",
        help=(
            "When --update-coors is set, also feed the updated-CA pair "
            "bias into the IPA pair features (ablation knob)."
        ),
    )
    # Validation (held-out CATH split) — PLAN_IF_ENCODER.md Task E7.
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=8,
        help=(
            "Number of proteins in the fixed CATH val batch used for "
            "per-epoch draft-recovery monitoring."
        ),
    )
    parser.add_argument(
        "--val-eval-every-epochs",
        type=int,
        default=1,
        help="Run val_step every N epochs (1 = every epoch).",
    )
    # Fallback aux-only pre-stage — PLAN_IF_ENCODER.md Task E7 fallback path.
    parser.add_argument(
        "--pretrain-aux-only-epochs",
        type=int,
        default=0,
        help=(
            "If > 0, run the first N epochs with aux-only training "
            "(encoder + draft head; decoder frozen and not invoked). "
            "Switches to joint training after N epochs OR when "
            "val/draft_recovery >= --pretrain-target."
        ),
    )
    parser.add_argument(
        "--pretrain-target",
        type=float,
        default=0.30,
        help=(
            "Pre-stage early-exit threshold on val/draft_recovery. "
            "Only consulted when --pretrain-aux-only-epochs > 0."
        ),
    )

    from inverse_folding.observability import add_wandb_cli_args

    add_wandb_cli_args(parser, default_project="mhc-if-imp-encoder")

    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.limit_batches is not None and args.limit_batches <= 0:
        parser.error("--limit-batches must be positive when provided")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.lambda_aux < 0:
        parser.error("--lambda-aux must be >= 0")
    if args.adapter_num_layers <= 0:
        parser.error("--adapter-num-layers must be positive")
    if args.adapter_num_layers > 1 and not args.adapter_gated:
        parser.error("--adapter-num-layers > 1 requires --adapter-gated")
    if args.egnn_depth <= 0 or args.ipa_depth <= 0:
        parser.error("--egnn-depth and --ipa-depth must be positive")
    if args.use_updated_coord_bias and not args.update_coors:
        parser.error(
            "--use-updated-coord-bias requires --update-coors to be set"
        )
    if args.val_batch_size <= 0:
        parser.error("--val-batch-size must be positive")
    if args.val_eval_every_epochs <= 0:
        parser.error("--val-eval-every-epochs must be positive")
    if args.pretrain_aux_only_epochs < 0:
        parser.error("--pretrain-aux-only-epochs must be >= 0")
    if not (0.0 <= args.pretrain_target <= 1.0):
        parser.error("--pretrain-target must be in [0, 1]")
    return args


def print_resolved_hyperparams(args: argparse.Namespace, *, run_dir: Path) -> None:
    resolved = vars(args).copy()
    resolved["cath_root"] = str(Path(args.cath_root).resolve())
    resolved["checkpoint"] = str(Path(args.checkpoint).resolve())
    resolved["output_dir"] = str(Path(args.output_dir).resolve())
    resolved["run_dir"] = str(run_dir)
    print("============================================================")
    print("train_if_imp_encoder resolved parameters")
    for key, value in resolved.items():
        print(f"  {key}: {value}")
    print("============================================================")


def install_geo_encoder(*, task: Any, args: argparse.Namespace, device: str) -> Any:
    """Swap the GVP encoder for a fresh ``GeoEGNNIPAEncoder`` and set
    ``cfg.detach_encoder_feats=False`` so main CE backprops into it.
    """
    import torch
    from omegaconf import OmegaConf

    from inverse_folding.dplm_refiner.geo_encoder import (
        GeoEGNNIPAEncoder,
        GraphConfig,
    )
    from byprot.models.dplm.modules.dplm_adapter import install_adapters

    alphabet = task.alphabet
    # IMPORTANT: do NOT include ``mask_idx`` or ``unk_idx`` in the
    # structural-special set. Those are real residues for graph purposes
    # (``<mask>`` is a denoising state, ``<unk>`` is a residue with
    # unknown identity but valid geometry). Including them would drop
    # masked positions from the graph at any caller that derived the
    # special mask from ``prev_tokens`` instead of ``tokens`` — and that
    # construction value persists into the saved checkpoint. Keep only
    # the real boundary tokens.
    special_token_ids = sorted(
        {
            int(alphabet.cls_idx),
            int(alphabet.padding_idx),
            int(alphabet.eos_idx),
        }
    )
    encoder = GeoEGNNIPAEncoder(
        d_model=int(args.d_model),
        output_logits=True,
        vocab_size=len(alphabet),
        graph_config=GraphConfig(),
        egnn_depth=int(args.egnn_depth),
        egnn_hidden_dim=int(args.egnn_hidden_dim),
        update_coors=bool(args.update_coors),
        ipa_depth=int(args.ipa_depth),
        ipa_hidden_dim=int(args.ipa_hidden_dim),
        ipa_pairwise_dim=int(args.ipa_pairwise_dim),
        ipa_heads=int(args.ipa_heads),
        use_updated_coord_bias=bool(args.use_updated_coord_bias),
        special_token_ids=tuple(special_token_ids),
    )
    encoder.to(device)
    # In-place attribute swap.
    task.model.encoder = encoder

    # Disable detach so main CE flows back through the new encoder.
    OmegaConf.set_struct(task.model.cfg, False)
    task.model.cfg.detach_encoder_feats = False

    current_n = int(getattr(task.model.decoder.cfg, "adapter_num_layers", 1))
    current_gated = bool(
        getattr(task.model.decoder.cfg, "adapter_gated", False)
    )
    requested_n = int(args.adapter_num_layers)
    requested_gated = bool(args.adapter_gated)
    requested_gate_init = float(args.adapter_gate_init)

    OmegaConf.set_struct(task.model.decoder.cfg, False)
    task.model.decoder.cfg.adapter_num_layers = requested_n
    task.model.decoder.cfg.adapter_gated = requested_gated
    task.model.decoder.cfg.adapter_gate_init = requested_gate_init
    if hasattr(task.model.cfg, "decoder"):
        OmegaConf.set_struct(task.model.cfg.decoder, False)
        task.model.cfg.decoder.adapter_num_layers = requested_n
        task.model.cfg.decoder.adapter_gated = requested_gated
        task.model.cfg.decoder.adapter_gate_init = requested_gate_init

    if requested_n != current_n or requested_gated != current_gated:
        print(
            "Reinstalling decoder adapters: "
            f"current_n={current_n} current_gated={current_gated} -> "
            f"requested_n={requested_n} requested_gated={requested_gated} "
            f"gate_init={requested_gate_init}"
        )
        install_adapters(
            task.model.decoder.net,
            task.model.decoder.cfg,
            adapter_num_layers=requested_n,
        )
        task.model.decoder.to(device)

    # Freeze the entire DPLM backbone; unfreeze adapter + new encoder.
    for p in task.model.parameters():
        p.requires_grad_(False)
    for pname, p in task.model.named_parameters():
        if "adapter" in pname or pname.startswith("encoder."):
            p.requires_grad_(True)
    return encoder


def _prepare_fwd_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    """Move the three encoder inputs to ``device`` and return a fresh dict.

    We deliberately don't move other keys (e.g. ``name``, ``length``) — the
    encoder + decoder paths only need ``tokens`` / ``coords`` / ``coord_mask``.
    """
    return {
        "tokens": batch["tokens"].to(device),
        "coords": batch["coords"].to(device),
        "coord_mask": batch["coord_mask"].to(device).bool(),
    }


def _aux_loss_and_recovery(
    encoder_logits, native_tokens, seq_mask
):
    """Compute aux CE + draft_recovery over the full seq_mask (NOT the
    diffusion-masked subset). Returns ``(aux_loss, draft_recovery,
    n_valid_positions)``. If ``seq_mask`` is empty, returns finite
    zero-gradient values.

    Belongs to PLAN_IF_ENCODER.md Task E7: aux must supervise every
    valid residue with native labels, otherwise the encoder draft head
    only sees k-of-L positions per protein and never learns the full
    structure-to-sequence mapping.
    """
    import torch
    import torch.nn.functional as F

    valid_logits = encoder_logits[seq_mask]
    valid_native = native_tokens[seq_mask]
    n_valid = int(seq_mask.sum().item())
    if n_valid == 0:
        zero = encoder_logits.sum() * 0.0
        return zero, torch.tensor(float("nan")), 0
    aux_loss = F.cross_entropy(valid_logits, valid_native)
    draft_recovery = (valid_logits.argmax(-1) == valid_native).float().mean()
    return aux_loss, draft_recovery, n_valid


def train_step(
    *,
    task: Any,
    optimizer: Any,
    batch: dict[str, Any],
    device: str,
    lambda_aux: float,
) -> dict[str, Any]:
    """Joint train step with native target for BOTH main and aux losses.

    PLAN_IF_ENCODER.md Task E7 fix. The previous implementation went
    through ``task.model.forward(..., output_encoder_logits=True)``,
    which made ``decoder.compute_loss`` build ``x_t`` and ``loss_mask``
    from ``encoder_logits.argmax()`` (init_pred) — diffusion-input
    pollution + risk of ``loss_mask`` corruption when init_pred lands
    on special-token ids. Bypassing forward and calling
    ``decoder.compute_loss(..., tokens=None, ...)`` makes both ``x_t``
    and the returned ``target`` come from ``batch["tokens"]`` (native).

    Aux CE is computed on the FULL ``seq_mask`` (all valid residues),
    not on the sparse ``loss_mask``, so the encoder draft head learns
    the full structure-to-sequence mapping rather than the diffusion
    subset.
    """
    import torch
    import torch.nn.functional as F

    fwd_batch = _prepare_fwd_batch(batch, device)
    native = fwd_batch["tokens"]

    # 1) Encoder forward with draft logits AND feats.
    encoder_logits, encoder_out = task.model.encoder(
        fwd_batch, output_logits=True
    )
    seq_mask = encoder_out["encoder_attention_mask"].bool()

    # 2) Aux CE on every valid residue against NATIVE tokens.
    aux_loss, draft_recovery, n_valid = _aux_loss_and_recovery(
        encoder_logits, native, seq_mask
    )

    # 3) Main DPLM diffusion CE with NATIVE target.
    #    ``tokens=None`` here is the load-bearing fix: it makes
    #    ``compute_loss`` derive both target AND x_t from
    #    ``batch["tokens"]`` (native), not from init_pred.
    encoder_out_for_dec = dict(encoder_out)
    encoder_out_for_dec["feats"] = encoder_out["feats"].repeat(2, 1, 1)
    encoder_out_for_dec["encoder_attention_mask"] = seq_mask.repeat(2, 1)
    logits, target, loss_mask, weight = task.model.decoder.compute_loss(
        batch=fwd_batch,
        weighting="linear",
        encoder_out=encoder_out_for_dec,
        tokens=None,
    )
    flat_logits = logits[loss_mask]
    flat_target = target[loss_mask]
    flat_weight = weight[loss_mask]
    if flat_logits.shape[0] == 0:
        main_loss = logits.sum() * 0.0
    else:
        per_pos = F.cross_entropy(flat_logits, flat_target, reduction="none")
        main_loss = (per_pos * flat_weight).sum() / flat_weight.sum().clamp(min=1)

    loss = main_loss + lambda_aux * aux_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    feats_std = float(encoder_out["feats"][seq_mask].std().item()) if n_valid > 0 else float("nan")
    return {
        "loss": float(loss.item()),
        "main_loss": float(main_loss.item()),
        "aux_loss": float(aux_loss.item()),
        "draft_recovery": float(draft_recovery.item()) if n_valid > 0 else float("nan"),
        "feats_std": feats_std,
        "n_aux_positions": n_valid,
        "n_main_positions": int(loss_mask.sum().item()),
    }


def train_step_aux_only(
    *,
    task: Any,
    optimizer: Any,
    batch: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Aux-only train step for the optional pre-stage.

    Only the encoder + draft head receive gradient. The decoder is
    not invoked; the adapter never sees any signal. This is the
    fallback path in PLAN_IF_ENCODER.md Task E7 — used only when
    ``--pretrain-aux-only-epochs > 0``.
    """
    import torch

    fwd_batch = _prepare_fwd_batch(batch, device)
    native = fwd_batch["tokens"]

    encoder_logits, encoder_out = task.model.encoder(
        fwd_batch, output_logits=True
    )
    seq_mask = encoder_out["encoder_attention_mask"].bool()
    aux_loss, draft_recovery, n_valid = _aux_loss_and_recovery(
        encoder_logits, native, seq_mask
    )

    optimizer.zero_grad()
    aux_loss.backward()
    optimizer.step()

    feats_std = float(encoder_out["feats"][seq_mask].std().item()) if n_valid > 0 else float("nan")
    return {
        "loss": float(aux_loss.item()),
        "main_loss": 0.0,
        "aux_loss": float(aux_loss.item()),
        "draft_recovery": float(draft_recovery.item()) if n_valid > 0 else float("nan"),
        "feats_std": feats_std,
        "n_aux_positions": n_valid,
        "n_main_positions": 0,
    }


def val_step(
    *,
    task: Any,
    val_batch: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """No-grad val pass: compute draft_recovery and aux CE on a fixed
    held-out CATH batch. Used to monitor encoder progress + drive the
    best-on-val checkpoint save.
    """
    import torch

    encoder = task.model.encoder
    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            fwd_batch = _prepare_fwd_batch(val_batch, device)
            native = fwd_batch["tokens"]
            encoder_logits, encoder_out = encoder(
                fwd_batch, output_logits=True
            )
            seq_mask = encoder_out["encoder_attention_mask"].bool()
            aux_loss, draft_recovery, n_valid = _aux_loss_and_recovery(
                encoder_logits, native, seq_mask
            )
    finally:
        if was_training:
            encoder.train()

    return {
        "val_aux_loss": float(aux_loss.item()),
        "val_draft_recovery": float(draft_recovery.item()) if n_valid > 0 else float("nan"),
        "val_n_positions": n_valid,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cath_root = Path(args.cath_root).expanduser().resolve()
    if not cath_root.is_dir():
        raise FileNotFoundError(f"--cath-root not found: {cath_root}")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"--checkpoint not found: {checkpoint}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"train_if_imp_encoder_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print_resolved_hyperparams(args, run_dir=run_dir)

    import random

    import numpy as np
    import torch
    import yaml

    from inverse_folding.dplm_refiner.geo_encoder.checkpoint import (
        save_geo_encoder_checkpoint,
    )
    from inverse_folding.observability import (
        finish_wandb,
        init_wandb_from_args,
        log_metrics,
        set_summary,
    )
    from inverse_folding.reference_flow.runtime import load_if_task

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    task = load_if_task(str(checkpoint), device=args.device)
    encoder = install_geo_encoder(task=task, args=args, device=args.device)
    encoder.train()

    trainable_names = [
        n for n, p in task.model.named_parameters() if p.requires_grad
    ]
    print(f"trainable param count: {sum(1 for _ in trainable_names)}")
    print("trainable param prefixes (first 20):")
    for n in trainable_names[:20]:
        print(f"  {n}")

    trainable_params = [p for p in task.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr))

    # Reuse the existing CATH loader builder from train_if_imp_refiner.
    from importlib import import_module

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    refiner_mod = import_module("train_if_imp_refiner")
    loader = refiner_mod.build_cath_loader(
        task=task,
        cath_root=cath_root,
        split=args.split,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=(args.split == "train"),
    )

    # Held-out val batch — first ``--val-batch-size`` proteins of the
    # CATH ``validation`` split, deterministic order, no shuffle. Build
    # once and reuse so per-epoch val numbers are directly comparable.
    val_loader = refiner_mod.build_cath_loader(
        task=task,
        cath_root=cath_root,
        split="validation",
        max_length=int(args.max_length),
        batch_size=int(args.val_batch_size),
        num_workers=0,
        shuffle=False,
    )
    try:
        val_batch = next(iter(val_loader))
        print(
            f"val batch built: {int(args.val_batch_size)} proteins, "
            f"first batch coord_mask sum = "
            f"{int(val_batch['coord_mask'].sum().item())}"
        )
    except StopIteration:
        raise RuntimeError(
            "CATH validation split returned no batches; cannot build a "
            "fixed val batch for draft-recovery monitoring."
        )

    run_config = {
        "stage": "if_imp_encoder_train",
        "split": args.split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "lambda_aux": args.lambda_aux,
        "adapter_num_layers": args.adapter_num_layers,
        "adapter_gated": args.adapter_gated,
        "adapter_gate_init": args.adapter_gate_init,
        "d_model": args.d_model,
        "egnn_depth": args.egnn_depth,
        "egnn_hidden_dim": args.egnn_hidden_dim,
        "ipa_depth": args.ipa_depth,
        "ipa_hidden_dim": args.ipa_hidden_dim,
        "update_coors": args.update_coors,
        "use_updated_coord_bias": args.use_updated_coord_bias,
        "val_batch_size": args.val_batch_size,
        "val_eval_every_epochs": args.val_eval_every_epochs,
        "pretrain_aux_only_epochs": args.pretrain_aux_only_epochs,
        "pretrain_target": args.pretrain_target,
        "device": args.device,
        "seed": args.seed,
    }
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(run_config))

    wandb_run = init_wandb_from_args(
        args, run_name=run_id, config=run_config
    )

    metrics_path = run_dir / "metrics.jsonl"
    manifest = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "trainable_param_count": len(trainable_params),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    adapter_cfg = {
        "adapter_num_layers": int(
            getattr(task.model.decoder.cfg, "adapter_num_layers", 1)
        ),
        "adapter_gated": bool(
            getattr(task.model.decoder.cfg, "adapter_gated", False)
        ),
        "adapter_gate_init": float(
            getattr(task.model.decoder.cfg, "adapter_gate_init", 0.0)
        ),
    }

    pretrain_budget = int(args.pretrain_aux_only_epochs)
    pretrain_target = float(args.pretrain_target)
    pretrain_exited = pretrain_budget == 0
    best_val_recovery: float = float("-inf")

    step_idx = 0
    for epoch in range(int(args.epochs)):
        # PLAN_IF_ENCODER.md Task E7 stage switching:
        #   pretrain phase: aux-only (encoder + draft head); decoder
        #     not invoked. Switch to joint when budget exhausted OR
        #     val/draft_recovery >= --pretrain-target.
        #   joint phase: full main + lambda_aux * aux on native target.
        in_pretrain = (not pretrain_exited) and (epoch < pretrain_budget)
        phase = "pretrain" if in_pretrain else "joint"
        print(
            f"==== epoch {epoch} ({phase}; "
            f"pretrain_budget={pretrain_budget}, "
            f"best_val_recovery={best_val_recovery:.4f}) ===="
        )

        for bidx, batch in enumerate(loader):
            if args.limit_batches is not None and bidx >= args.limit_batches:
                break
            t0 = time.time()
            if in_pretrain:
                metrics = train_step_aux_only(
                    task=task,
                    optimizer=optimizer,
                    batch=batch,
                    device=args.device,
                )
            else:
                metrics = train_step(
                    task=task,
                    optimizer=optimizer,
                    batch=batch,
                    device=args.device,
                    lambda_aux=float(args.lambda_aux),
                )
            metrics["epoch"] = epoch
            metrics["batch_idx"] = bidx
            metrics["step"] = step_idx
            metrics["phase"] = phase
            metrics["wall_seconds"] = time.time() - t0
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(metrics) + "\n")
            log_metrics(wandb_run, metrics, step=step_idx)
            if step_idx % int(args.log_every) == 0:
                print(
                    f"epoch={epoch} step={step_idx} phase={phase} "
                    f"loss={metrics['loss']:.4f} "
                    f"main={metrics['main_loss']:.4f} "
                    f"aux={metrics['aux_loss']:.4f} "
                    f"draft_rec={metrics['draft_recovery']:.4f} "
                    f"feats_std={metrics['feats_std']:.4f}"
                )
            step_idx += 1

        # Per-epoch val + best-on-improve save.
        if (epoch + 1) % int(args.val_eval_every_epochs) == 0:
            val_metrics = val_step(
                task=task, val_batch=val_batch, device=args.device
            )
            val_metrics["epoch"] = epoch
            val_metrics["step"] = step_idx
            val_metrics["phase"] = phase
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(val_metrics) + "\n")
            log_metrics(wandb_run, val_metrics, step=step_idx)
            print(
                f"==== val @ epoch {epoch}: "
                f"draft_recovery={val_metrics['val_draft_recovery']:.4f} "
                f"aux_loss={val_metrics['val_aux_loss']:.4f} "
                f"n_pos={val_metrics['val_n_positions']} ===="
            )
            current = float(val_metrics["val_draft_recovery"])
            # NaN comparison is always False, so this also guards the
            # "no valid positions" edge case.
            if current > best_val_recovery:
                best_val_recovery = current
                save_geo_encoder_checkpoint(
                    run_dir / "encoder_best.pt",
                    encoder,
                    decoder=task.model.decoder,
                    adapter_config=adapter_cfg,
                    extra={
                        "epoch": epoch,
                        "step": step_idx,
                        "val_draft_recovery": current,
                        "phase": phase,
                    },
                )
                print(
                    f"==== saved encoder_best.pt @ epoch {epoch} "
                    f"(val/draft_recovery={current:.4f}) ===="
                )

            # Pre-stage early-exit on target.
            if in_pretrain and current >= pretrain_target:
                print(
                    f"==== pretrain early-exit: "
                    f"val/draft_recovery={current:.4f} "
                    f">= --pretrain-target={pretrain_target:.4f} ===="
                )
                pretrain_exited = True

        # Save last-epoch checkpoint (kept for resume / final consumption).
        save_geo_encoder_checkpoint(
            run_dir / "encoder_last.pt",
            encoder,
            decoder=task.model.decoder,
            adapter_config=adapter_cfg,
            extra={"epoch": epoch, "step": step_idx, "phase": phase},
        )

    set_summary(
        wandb_run,
        {
            "final_step": step_idx,
            "trainable_params": len(trainable_params),
            "best_val_draft_recovery": best_val_recovery,
        },
    )
    finish_wandb(wandb_run)
    print(
        f"Done. encoder_last.pt = {run_dir / 'encoder_last.pt'}; "
        f"encoder_best.pt = {run_dir / 'encoder_best.pt'} "
        f"(best val/draft_recovery={best_val_recovery:.4f})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
