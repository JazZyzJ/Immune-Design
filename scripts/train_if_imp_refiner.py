#!/usr/bin/env python
"""Train the DPLM-compatible IPA refiner on CATH.

Trains only ``DPLMIPARefiner`` (DPLM backbone is loaded for alphabet
and featurizer access; its weights are NOT updated). Each batch step:

1. Sample a random ``beta_t_bar`` per row.
2. Derive ``mask_ratio`` via ``sine_mask_ratio``.
3. Compute an entropy proxy. Default proxy is uniform random noise; with
   ``--use-dplm-entropy`` the proxy is the DPLM decoder's entropy on the
   masked input (set via ``inject_noise(noise="full_mask")``).
4. Select top-entropy valid positions.
5. Train the refiner with masked cross-entropy on those positions.

PLAN_IF_IMP.md Task 9.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the MapDiff-style IPA refiner on CATH backbones with "
            "DPLM-compatible batches."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cath-root", required=True, help="Directory with chain_set.jsonl + chain_set_splits.json.")
    parser.add_argument("--checkpoint", required=True, help="Module K DPLM checkpoint (.ckpt). Used for alphabet/featurizer.")
    parser.add_argument("--output-dir", required=True, help="Directory to write refiner_last.pt + metrics.jsonl + run_config.yaml.")
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask-ratio-center", type=float, default=0.4)
    parser.add_argument("--mask-ratio-deviation", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-batches", type=int, default=None, help="Stop after N batches per epoch (for smoke runs).")
    parser.add_argument("--use-dplm-entropy", action="store_true", help="Use DPLM decoder entropy for selection instead of random.")
    parser.add_argument("--log-every", type=int, default=10)

    from inverse_folding.observability import add_wandb_cli_args

    add_wandb_cli_args(parser, default_project="mhc-if-imp-refiner")

    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if not (0.0 <= args.mask_ratio_center <= 1.0):
        parser.error("--mask-ratio-center must be in [0, 1]")
    if not (0.0 <= args.mask_ratio_deviation <= 1.0):
        parser.error("--mask-ratio-deviation must be in [0, 1]")
    if args.mask_ratio_center + args.mask_ratio_deviation > 1.0:
        parser.error("--mask-ratio-center + --mask-ratio-deviation must be <= 1")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.limit_batches is not None and args.limit_batches <= 0:
        parser.error("--limit-batches must be positive when provided")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    return args


def print_resolved_hyperparams(args: argparse.Namespace, *, run_dir: Path) -> None:
    resolved = vars(args).copy()
    resolved["cath_root"] = str(Path(args.cath_root).resolve())
    resolved["checkpoint"] = str(Path(args.checkpoint).resolve())
    resolved["output_dir"] = str(Path(args.output_dir).resolve())
    resolved["run_dir"] = str(run_dir)
    print("============================================================")
    print("train_if_imp_refiner resolved parameters")
    for key, value in resolved.items():
        print(f"  {key}: {value}")
    print("============================================================")


def build_cath_loader(
    *,
    task: Any,
    cath_root: Path,
    split: str,
    max_length: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> Any:
    """Build a CATH DataLoader using the canonical DPLM datamodule path.

    ``CATH(split=(s,), ...)`` returns ``(dataset, alphabet_set)`` with a
    single Subset (auto-unwrapped when ``len(split) == 1``). Collate is
    the DPLM ``Alphabet.featurizer`` so this matches DPLM's batch
    contract bit-for-bit (PLAN Source Anchors §cath.py).
    """
    import torch.utils.data

    from byprot.datamodules.dataset.cath import CATH

    dataset, _alphabet_set = CATH(
        root=str(cath_root),
        split=(split,),
        max_length=int(max_length),
    )
    if isinstance(dataset, list):
        # Defensive: even though split=(s,) auto-unwraps, keep working
        # if a future DPLM version changes that behavior.
        dataset = dataset[0]
    collate_fn = task.alphabet.featurizer
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_fn,
    )


def compute_entropy_proxy(
    *,
    task: Any,
    bridge: Any,
    tokens: Any,
    coords: Any,
    coord_mask: Any,
    use_dplm_entropy: bool,
) -> Any:
    """Return ``[B, L]`` entropy proxy used for top-k mask selection.

    If ``use_dplm_entropy`` is True, run DPLM encoder + decoder once on
    the fully-masked input (via ``task.inject_noise(noise='full_mask')``)
    and compute MapDiff entropy on the AA-axis log-probs. Otherwise
    return a uniform-random proxy (same shape, on the same device).
    """
    import torch

    from inverse_folding.dplm_refiner.entropy import (
        mapdiff_entropy_from_log_probs,
    )

    if not use_dplm_entropy:
        return torch.rand_like(tokens, dtype=torch.float32)

    prev_tokens, prev_token_mask = task.inject_noise(
        tokens, coord_mask, noise="full_mask"
    )
    batch_for_encoder = {
        "tokens": tokens,
        "coords": coords,
        "coord_mask": coord_mask,
        "prev_tokens": prev_tokens,
        "prev_token_mask": prev_token_mask,
    }
    with torch.no_grad():
        encoder_out = task.model.forward_encoder(
            batch_for_encoder,
            use_draft_seq=bool(task.hparams.generator.use_draft_seq),
        )
        decoder_out = task.model.decoder(
            batch={"prev_tokens": prev_tokens},
            encoder_out=encoder_out,
            need_head_weights=False,
        )
        base_logits = decoder_out["logits"]
        base_aa_logits = bridge.to_aa_logits(base_logits)
        base_log_probs = torch.log_softmax(base_aa_logits, dim=-1)
        return mapdiff_entropy_from_log_probs(base_log_probs)


def train_one_batch(
    *,
    model: Any,
    optimizer: Any,
    task: Any,
    bridge: Any,
    batch: dict[str, Any],
    device: str,
    mask_ratio_center: float,
    mask_ratio_deviation: float,
    use_dplm_entropy: bool,
) -> dict[str, Any]:
    """Single optimizer step on one batch.

    Returns a metrics dict: ``loss``, ``n_selected``, ``n_valid``.
    """
    import torch

    from inverse_folding.dplm_refiner.entropy import (
        select_entropy_mask,
        sine_mask_ratio,
    )
    from inverse_folding.dplm_refiner.geometry import (
        dplm_coords_to_ipa_positions,
    )
    from inverse_folding.dplm_refiner.training import (
        masked_refiner_cross_entropy,
    )

    tokens = batch["tokens"].to(device)
    coords = batch["coords"].to(device)
    coord_mask = batch["coord_mask"].to(device).bool()

    structural_mask = (
        tokens.eq(task.alphabet.padding_idx)
        | tokens.eq(task.alphabet.cls_idx)
        | tokens.eq(task.alphabet.eos_idx)
    )

    ipa_pos = dplm_coords_to_ipa_positions(
        coords=coords,
        coord_mask=coord_mask,
        special_sym_mask=structural_mask,
    )
    seq_mask = ipa_pos.seq_mask
    atom_pos = ipa_pos.atom_pos

    B = tokens.shape[0]
    beta_t_bar = torch.rand((B,), device=device)
    ratios = sine_mask_ratio(
        beta_t_bar,
        center=float(mask_ratio_center),
        max_deviation=float(mask_ratio_deviation),
    )

    entropy = compute_entropy_proxy(
        task=task,
        bridge=bridge,
        tokens=tokens,
        coords=coords,
        coord_mask=coord_mask,
        use_dplm_entropy=use_dplm_entropy,
    )

    selected = select_entropy_mask(
        entropy=entropy, valid_mask=seq_mask, mask_ratios=ratios
    )

    x_aa = bridge.one_hot_from_dplm_tokens(tokens)
    x_aa = torch.where(selected.unsqueeze(-1), torch.zeros_like(x_aa), x_aa)
    x_aa_mask = selected.long()

    target_aa_tokens = bridge.to_aa_tokens(tokens)
    target_for_loss = torch.where(
        target_aa_tokens < 0,
        torch.zeros_like(target_aa_tokens),
        target_aa_tokens,
    )

    logits = model(
        x_aa=x_aa,
        x_pos=atom_pos,
        x_aa_mask=x_aa_mask,
        seq_mask=seq_mask,
    )

    valid_for_loss = seq_mask & (target_aa_tokens >= 0)
    loss = masked_refiner_cross_entropy(
        logits=logits,
        target_aa_tokens=target_for_loss,
        selected_mask=selected,
        valid_mask=valid_for_loss,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "n_selected": int(selected.sum().item()),
        "n_valid": int(seq_mask.sum().item()),
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

    run_id = f"train_if_imp_refiner_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print_resolved_hyperparams(args, run_dir=run_dir)

    import random

    import numpy as np
    import torch
    import yaml

    from inverse_folding.dplm_refiner.checkpoint import save_refiner_checkpoint
    from inverse_folding.dplm_refiner.config import DPLMRefinerConfig
    from inverse_folding.dplm_refiner.ipa.refiner import DPLMIPARefiner
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

    loader = build_cath_loader(
        task=task,
        cath_root=cath_root,
        split=args.split,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=(args.split == "train"),
    )

    model = DPLMIPARefiner(dropout=0.2)
    model.to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    wandb_run = init_wandb_from_args(
        args,
        run_name=run_id,
        config={
            "stage": "if_imp_refiner_train",
            "split": args.split,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
            "mask_ratio_center": args.mask_ratio_center,
            "mask_ratio_deviation": args.mask_ratio_deviation,
            "use_dplm_entropy": args.use_dplm_entropy,
            "device": args.device,
            "seed": args.seed,
            "run_dir": str(run_dir),
        },
        extra_tags=["if_imp", "train_refiner"],
    )

    metrics_path = run_dir / "metrics.jsonl"
    metrics_fp = open(metrics_path, "w")
    config_path = run_dir / "run_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "mode": "train_if_imp_refiner",
                "cath_root": str(cath_root),
                "checkpoint": str(checkpoint),
                "split": args.split,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_length": args.max_length,
                "mask_ratio_center": args.mask_ratio_center,
                "mask_ratio_deviation": args.mask_ratio_deviation,
                "use_dplm_entropy": args.use_dplm_entropy,
                "device": args.device,
                "seed": args.seed,
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
                model=model,
                optimizer=optimizer,
                task=task,
                bridge=bridge,
                batch=batch,
                device=args.device,
                mask_ratio_center=float(args.mask_ratio_center),
                mask_ratio_deviation=float(args.mask_ratio_deviation),
                use_dplm_entropy=bool(args.use_dplm_entropy),
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
                    f"[train] epoch={epoch} step={global_step} "
                    f"batch={batch_idx} loss={metrics['loss']:.4f} "
                    f"n_selected={metrics['n_selected']}"
                )

    metrics_fp.close()
    train_seconds = time.time() - train_start

    refiner_config = DPLMRefinerConfig(
        mask_ratio_center=float(args.mask_ratio_center),
        mask_ratio_deviation=float(args.mask_ratio_deviation),
    )
    save_refiner_checkpoint(
        run_dir / "refiner_last.pt",
        model=model.cpu(),
        config=refiner_config,
        extra={
            "epochs": int(args.epochs),
            "n_batches_total": int(n_batches_total),
            "lr": float(args.lr),
            "split": args.split,
            "use_dplm_entropy": bool(args.use_dplm_entropy),
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
        f"[done] saved refiner_last.pt to {run_dir} "
        f"after {n_batches_total} batches in {train_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
