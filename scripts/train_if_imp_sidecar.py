#!/usr/bin/env python
"""Train the DPLM-compatible IPA geometry sidecar on CATH.

The sidecar adds residual features ``[B, L, 512]`` to the GVP encoder
output. Training objective: minimize DPLM decoder cross-entropy on
``inject_noise(noise='full_mask')`` tokens, with DPLM weights frozen
and the sidecar as the only trainable module. This shapes the sidecar
to inject geometric context that improves DPLM's masked-AA recovery.

PLAN_IF_IMP.md Task 11 (training is out-of-scope for the original
plan; this script is added so all 4 ablation arms are executable).
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
        description="Train the DPLM-compatible IPA geometry sidecar on CATH backbones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cath-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="Module K DPLM checkpoint (.ckpt).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ipa-depth", type=int, default=6)
    parser.add_argument("--output-dim", type=int, default=512)

    from inverse_folding.observability import add_wandb_cli_args

    add_wandb_cli_args(parser, default_project="mhc-if-imp-refiner")

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
    if args.hidden_dim <= 0:
        parser.error("--hidden-dim must be positive")
    if args.ipa_depth <= 0:
        parser.error("--ipa-depth must be positive")
    if args.output_dim <= 0:
        parser.error("--output-dim must be positive")
    return args


def print_resolved_hyperparams(args: argparse.Namespace, *, run_dir: Path) -> None:
    resolved = vars(args).copy()
    resolved["cath_root"] = str(Path(args.cath_root).resolve())
    resolved["checkpoint"] = str(Path(args.checkpoint).resolve())
    resolved["output_dir"] = str(Path(args.output_dir).resolve())
    resolved["run_dir"] = str(run_dir)
    print("============================================================")
    print("train_if_imp_sidecar resolved parameters")
    for key, value in resolved.items():
        print(f"  {key}: {value}")
    print("============================================================")


def train_one_batch(
    *,
    sidecar: Any,
    optimizer: Any,
    task: Any,
    bridge: Any,
    batch: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Single optimizer step for the sidecar.

    Wires the sidecar into the encoder via SidecarAttachedEncoder for
    the forward pass, computes DPLM decoder logits on the fully-masked
    input, and minimizes CE on the true tokens at masked positions.
    Only sidecar params receive gradient; DPLM weights are frozen.
    """
    import torch

    from inverse_folding.dplm_refiner.encoder_wrapper import (
        SidecarAttachedEncoder,
    )

    tokens = batch["tokens"].to(device)
    coords = batch["coords"].to(device)
    coord_mask = batch["coord_mask"].to(device).bool()
    alphabet = task.alphabet

    structural_mask = (
        tokens.eq(alphabet.padding_idx)
        | tokens.eq(alphabet.cls_idx)
        | tokens.eq(alphabet.eos_idx)
    )

    prev_tokens, prev_token_mask = task.inject_noise(
        tokens, coord_mask, noise="full_mask"
    )

    enc_batch = {
        "tokens": tokens,
        "coords": coords,
        "coord_mask": coord_mask,
        "prev_tokens": prev_tokens,
        "prev_token_mask": prev_token_mask,
    }

    base_encoder = task.model.encoder
    if not isinstance(base_encoder, SidecarAttachedEncoder):
        # Wrap once for the duration of this batch step.
        wrapped = SidecarAttachedEncoder(
            base_encoder,
            sidecar,
            get_special_sym_mask=lambda b: (
                b["prev_tokens"].eq(alphabet.padding_idx)
                | b["prev_tokens"].eq(alphabet.cls_idx)
                | b["prev_tokens"].eq(alphabet.eos_idx)
            ),
        )
        task.model.encoder = wrapped
        attached_for_this_step = True
    else:
        attached_for_this_step = False

    try:
        encoder_out = task.model.forward_encoder(
            enc_batch,
            use_draft_seq=bool(task.hparams.generator.use_draft_seq),
        )
        decoder_out = task.model.decoder(
            batch={"prev_tokens": prev_tokens},
            encoder_out=encoder_out,
            need_head_weights=False,
        )
        full_logits = decoder_out["logits"]
        aa_logits = bridge.to_aa_logits(full_logits)

        target_aa_tokens = bridge.to_aa_tokens(tokens)
        valid = (
            (~structural_mask)
            & coord_mask
            & (target_aa_tokens >= 0)
            & prev_token_mask
        )
        if int(valid.sum().item()) == 0:
            loss = aa_logits.sum() * 0.0
        else:
            flat_logits = aa_logits[valid]
            flat_target = target_aa_tokens[valid].clamp(min=0).long()
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    finally:
        if attached_for_this_step:
            task.model.encoder = base_encoder

    return {
        "loss": float(loss.item()),
        "n_target_positions": int(valid.sum().item()),
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

    run_id = f"train_if_imp_sidecar_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print_resolved_hyperparams(args, run_dir=run_dir)

    import random

    import numpy as np
    import torch
    import yaml

    from inverse_folding.dplm_refiner.checkpoint import save_sidecar_checkpoint
    from inverse_folding.dplm_refiner.sidecar import DPLMGeometrySidecar
    from inverse_folding.dplm_refiner.tokens import DPLMTokenBridge
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
    bridge = DPLMTokenBridge.from_alphabet(task.alphabet)

    # Freeze EVERYTHING in DPLM; only sidecar params train.
    for p in task.model.parameters():
        p.requires_grad_(False)

    sidecar = DPLMGeometrySidecar(
        hidden_dim=int(args.hidden_dim),
        ipa_depth=int(args.ipa_depth),
        output_dim=int(args.output_dim),
        dropout=0.2,
    )
    sidecar.to(args.device)
    sidecar.train()
    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=float(args.lr))

    # Reuse the refiner's CATH loader builder
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

    wandb_run = init_wandb_from_args(
        args,
        run_name=run_id,
        config={
            "stage": "if_imp_sidecar_train",
            "split": args.split,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
            "hidden_dim": args.hidden_dim,
            "ipa_depth": args.ipa_depth,
            "output_dim": args.output_dim,
            "device": args.device,
            "seed": args.seed,
            "run_dir": str(run_dir),
        },
        extra_tags=["if_imp", "train_sidecar"],
    )

    metrics_path = run_dir / "metrics.jsonl"
    metrics_fp = open(metrics_path, "w")
    config_path = run_dir / "run_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "mode": "train_if_imp_sidecar",
                "cath_root": str(cath_root),
                "checkpoint": str(checkpoint),
                "split": args.split,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "max_length": int(args.max_length),
                "hidden_dim": int(args.hidden_dim),
                "ipa_depth": int(args.ipa_depth),
                "output_dim": int(args.output_dim),
                "device": args.device,
                "seed": int(args.seed),
            },
            f,
            sort_keys=False,
        )

    global_step = 0
    n_batches_total = 0
    train_start = time.time()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader):
            if args.limit_batches is not None and batch_idx >= args.limit_batches:
                break
            metrics = train_one_batch(
                sidecar=sidecar,
                optimizer=optimizer,
                task=task,
                bridge=bridge,
                batch=batch,
                device=args.device,
            )
            global_step += 1
            n_batches_total += 1

            if global_step % args.log_every == 0:
                row = {
                    "epoch": int(epoch),
                    "batch_idx": int(batch_idx),
                    "global_step": int(global_step),
                    **metrics,
                }
                metrics_fp.write(json.dumps(row) + "\n")
                metrics_fp.flush()
                log_metrics(
                    wandb_run,
                    {f"train/{k}": v for k, v in row.items() if k != "epoch"},
                    step=global_step,
                )
                print(
                    f"[train-sidecar] epoch={epoch} step={global_step} "
                    f"batch={batch_idx} loss={metrics['loss']:.4f} "
                    f"n_targets={metrics['n_target_positions']}"
                )

    metrics_fp.close()
    train_seconds = time.time() - train_start
    save_sidecar_checkpoint(
        run_dir / "sidecar_last.pt",
        model=sidecar.cpu(),
        extra={
            "epochs": int(args.epochs),
            "n_batches_total": int(n_batches_total),
            "lr": float(args.lr),
            "split": args.split,
            "hidden_dim": int(args.hidden_dim),
            "ipa_depth": int(args.ipa_depth),
            "output_dim": int(args.output_dim),
        },
    )
    set_summary(
        wandb_run,
        {
            "summary/train_seconds": float(train_seconds),
            "summary/n_batches_total": int(n_batches_total),
        },
    )
    finish_wandb(wandb_run)
    print(
        f"[done] saved sidecar_last.pt to {run_dir} "
        f"after {n_batches_total} batches in {train_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
