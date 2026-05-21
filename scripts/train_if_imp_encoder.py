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


def train_step(
    *,
    task: Any,
    optimizer: Any,
    batch: dict[str, Any],
    device: str,
    lambda_aux: float,
) -> dict[str, Any]:
    """One optimizer step against ``task.model.forward``."""
    import torch
    import torch.nn.functional as F

    tokens = batch["tokens"].to(device)
    coords = batch["coords"].to(device)
    coord_mask = batch["coord_mask"].to(device).bool()

    fwd_batch = {
        "tokens": tokens,
        "coords": coords,
        "coord_mask": coord_mask,
    }

    logits, target, loss_mask, weight, encoder_logits = task.model(
        fwd_batch,
        weighting="linear",
        return_outputs=True,
        output_encoder_logits=True,
    )

    # Main diffusion CE on masked positions only.
    flat_logits = logits[loss_mask]
    flat_target = target[loss_mask]
    flat_weight = weight[loss_mask]
    if flat_logits.shape[0] == 0:
        main_loss = logits.sum() * 0.0
    else:
        per_pos = F.cross_entropy(flat_logits, flat_target, reduction="none")
        main_loss = (per_pos * flat_weight).sum() / flat_weight.sum().clamp(min=1)

    # Auxiliary encoder CE on the same target positions.
    aux_loss = encoder_logits.new_zeros(())
    if encoder_logits is not None and lambda_aux > 0:
        # encoder_logits is repeat(2, 1, 1) to match target/loss_mask shape.
        enc_flat = encoder_logits[loss_mask]
        if enc_flat.shape[0] > 0:
            aux_loss = F.cross_entropy(enc_flat, flat_target)

    loss = main_loss + lambda_aux * aux_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "main_loss": float(main_loss.item()),
        "aux_loss": float(aux_loss.item()) if isinstance(aux_loss, torch.Tensor) else 0.0,
        "n_positions": int(loss_mask.sum().item()),
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

    step_idx = 0
    for epoch in range(int(args.epochs)):
        for bidx, batch in enumerate(loader):
            if args.limit_batches is not None and bidx >= args.limit_batches:
                break
            t0 = time.time()
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
            metrics["wall_seconds"] = time.time() - t0
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(metrics) + "\n")
            log_metrics(wandb_run, metrics, step=step_idx)
            if step_idx % int(args.log_every) == 0:
                print(
                    f"epoch={epoch} step={step_idx} "
                    f"loss={metrics['loss']:.4f} "
                    f"main={metrics['main_loss']:.4f} "
                    f"aux={metrics['aux_loss']:.4f}"
                )
            step_idx += 1
        # Save epoch checkpoint. Pass ``decoder=`` so the adapter's
        # trained weights (not just the encoder) round-trip through the
        # checkpoint; without this, generation paths would silently lose
        # every adapter update (PLAN_IF_ENCODER.md Task E5).
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
        save_geo_encoder_checkpoint(
            run_dir / "encoder_last.pt",
            encoder,
            decoder=task.model.decoder,
            adapter_config=adapter_cfg,
            extra={"epoch": epoch, "step": step_idx},
        )

    set_summary(
        wandb_run,
        {"final_step": step_idx, "trainable_params": len(trainable_params)},
    )
    finish_wandb(wandb_run)
    print(f"Done. Encoder checkpoint at {run_dir / 'encoder_last.pt'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
