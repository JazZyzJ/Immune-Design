"""Train Epitope Head v2 encoder ablation.

Runs a single (encoder_id, seed) combination from the H-module ablation matrix.
Reuses A-G training pipeline with encoder swapped via factory.

Usage:
    # E1 (Dilated CNN), seed 42, CPU smoke test:
    python scripts/train_v2_ablation.py --encoder-id E1 --seed 42 --device cpu --smoke

    # E0 (Frozen ESM-2), seed 42, GPU:
    python scripts/train_v2_ablation.py --encoder-id E0 --seed 42 --device cuda

    # E2 (Shallow Transformer), seed 43, GPU:
    python scripts/train_v2_ablation.py --encoder-id E2 --seed 43 --device cuda

    # Full matrix (bash loop):
    for enc in E0 E1 E2; do
      for seed in 42 43 44; do
        python scripts/train_v2_ablation.py --encoder-id $enc --seed $seed --device cuda
      done
    done
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from epitope_head.configs import (
    load_ablation_config,
    load_data_config,
    load_model_config,
    load_train_config,
)
from epitope_head.training.datamodule import (
    build_chunk_samples,
    build_dataloader,
    load_split_proteins,
)
from epitope_head.training.encoders import AATokenizer, build_encoder
from epitope_head.training.model import EpitopeScorer, FrozenESMEncoder, ESMTokenizer
from epitope_head.training.trainer import Trainer

logger = logging.getLogger(__name__)

VALID_ENCODER_IDS = {"E0", "E1", "E2"}
VALID_SEEDS = {42, 43, 44}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Epitope Head v2 encoder ablation (single run)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--encoder-id", type=str, required=True, choices=sorted(VALID_ENCODER_IDS),
                    help="Encoder ablation profile (E0, E1, E2)")
    p.add_argument("--seed", type=int, required=True,
                    help="Random seed (must be one of 42, 43, 44)")
    p.add_argument("--profile", type=str, default="strict", choices=["strict"],
                    help="Data profile (frozen to strict for H)")
    p.add_argument("--device", type=str, default="cpu",
                    help="Torch device (cpu, cuda, cuda:0, mps)")

    p.add_argument("--config-dir", type=str,
                    default=str(PROJECT_ROOT / "epitope_head" / "configs"),
                    help="Config directory")
    p.add_argument("--data-dir", type=str, default=None,
                    help="Manifest directory (default: outputs/manifests)")
    p.add_argument("--output-root", type=str, default=None,
                    help="Root for ablation outputs (default: outputs/ablation/encoder_v2)")

    smoke = p.add_argument_group("smoke testing")
    smoke.add_argument("--smoke", action="store_true",
                       help="Smoke mode: small data + few epochs")
    smoke.add_argument("--smoke-n-train", type=int, default=20)
    smoke.add_argument("--smoke-n-val", type=int, default=5)
    smoke.add_argument("--smoke-epochs", type=int, default=2)

    wb = p.add_argument_group("wandb")
    wb.add_argument("--wandb", action="store_true")
    wb.add_argument("--wandb-entity", type=str, default=None)
    wb.add_argument("--wandb-project", type=str, default="Immune-Design")
    wb.add_argument("--wandb-name", type=str, default=None)

    return p.parse_args()


def main() -> dict:
    args = parse_args()

    # Validate seed
    if args.seed not in VALID_SEEDS and not args.smoke:
        logger.warning("Seed %d not in canonical set %s — results may not be comparable", args.seed, VALID_SEEDS)

    log_level = logging.DEBUG if args.smoke else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # ── Load configs ──────────────────────────────────────────────────
    config_dir = Path(args.config_dir)
    data_cfg = load_data_config(config_dir / "data.yaml")
    model_cfg = load_model_config(config_dir / "model.yaml")
    train_cfg = load_train_config(config_dir / "train.yaml")
    ablation_cfg = load_ablation_config(config_dir / "model_ablation.yaml")

    profile_cfg = ablation_cfg["profiles"][args.encoder_id]
    encoder_type = profile_cfg["encoder_type"]
    d_enc = profile_cfg["d_enc"]
    encoder_cfg = profile_cfg["encoder_cfg"]

    logger.info("Ablation run: encoder=%s, seed=%d, profile=%s", args.encoder_id, args.seed, args.profile)

    # ── Override seed ─────────────────────────────────────────────────
    train_cfg["seed"] = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if train_cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True)

    # ── Resolve data paths ────────────────────────────────────────────
    manifest_dir = Path(args.data_dir) if args.data_dir else (PROJECT_ROOT / "outputs" / "manifests")
    profile = args.profile
    samples_path = manifest_dir / f"protein_samples_{profile}.parquet"
    train_ids_path = manifest_dir / "splits" / profile / "train_ids.txt"
    val_ids_path = manifest_dir / "splits" / profile / "val_ids.txt"

    for p in [samples_path, train_ids_path, val_ids_path]:
        if not p.exists():
            logger.error("Required artifact missing: %s", p)
            sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────
    train_entries = load_split_proteins(samples_path, train_ids_path)
    val_entries = load_split_proteins(samples_path, val_ids_path)
    logger.info("Loaded %d train, %d val proteins", len(train_entries), len(val_entries))

    # ── Smoke overrides ───────────────────────────────────────────────
    if args.smoke:
        train_entries = train_entries[:args.smoke_n_train]
        val_entries = val_entries[:args.smoke_n_val]
        train_cfg["max_epochs"] = args.smoke_epochs
        train_cfg["checkpoint_every_n_epochs"] = 1
        train_cfg["early_stopping_patience"] = args.smoke_epochs + 1
        train_cfg["num_workers"] = 0
        logger.info("SMOKE MODE: %d train, %d val, %d epochs",
                     len(train_entries), len(val_entries), train_cfg["max_epochs"])

    if sys.platform == "darwin" and train_cfg["num_workers"] > 0:
        logger.warning("macOS: forcing num_workers=0")
        train_cfg["num_workers"] = 0

    # ── Build chunks ──────────────────────────────────────────────────
    chunking = train_cfg["chunking"]
    ck_params = dict(context_len=chunking["context_len"], stride=chunking["stride"], margin=chunking["margin"])
    train_chunks = build_chunk_samples(train_entries, **ck_params)
    val_chunks = build_chunk_samples(val_entries, **ck_params)
    logger.info("Chunks: %d train, %d val", len(train_chunks), len(val_chunks))

    # ── Build encoder + tokenizer ─────────────────────────────────────
    device = torch.device(args.device)

    if encoder_type == "esm2_frozen":
        logger.info("Loading ESM-2: %s ...", encoder_cfg["encoder_name"])
        try:
            import esm
        except ImportError:
            logger.error("Package 'fair-esm' not installed. Use --encoder-id E1 or E2 for non-ESM ablation.")
            sys.exit(1)
        esm_model, alphabet = getattr(esm.pretrained, encoder_cfg["encoder_name"])()
        encoder = FrozenESMEncoder(esm_model, d_enc=d_enc)
        tokenizer = ESMTokenizer(alphabet)
    else:
        encoder, tokenizer = build_encoder(encoder_type, d_enc, encoder_cfg)

    trainable_enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info("Encoder: %s, trainable params: %d", encoder_type, trainable_enc_params)

    # ── Build dataloaders ─────────────────────────────────────────────
    train_loader = build_dataloader(
        train_chunks, max_tokens=train_cfg["max_tokens"],
        shuffle=True, seed=args.seed, num_workers=train_cfg["num_workers"],
        tokenize_fn=tokenizer,
    )
    val_loader = build_dataloader(
        val_chunks, max_tokens=train_cfg["max_tokens"],
        shuffle=False, seed=args.seed, num_workers=0,
        tokenize_fn=tokenizer,
    )

    # ── Build model ───────────────────────────────────────────────────
    # Use frozen constants from ablation config for head params
    frozen = ablation_cfg["frozen_constants"]
    model = EpitopeScorer(
        encoder=encoder,
        d_enc=d_enc,
        d_proj=frozen["d_proj"],
        length_emb_dim=model_cfg["length_embedding_dim"],
        allele_emb_dim=model_cfg["allele_embedding_dim"],
        min_k=frozen["min_k"],
        max_k=frozen["max_k"],
        n_alleles=1,
        scorer_hidden_dim=frozen["scorer_hidden_dim"],
        scorer_activation=frozen["scorer_activation"],
        scorer_dropout=frozen.get("scorer_dropout", 0.3),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d trainable / %d total params", trainable, total)

    # ── Run directory ─────────────────────────────────────────────────
    output_root = Path(args.output_root) if args.output_root else (
        PROJECT_ROOT / "outputs" / "ablation" / "encoder_v2"
    )
    suffix = "smoke" if args.smoke else f"seed_{args.seed}"
    run_dir = output_root / "runs" / args.encoder_id / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # ── Registry ──────────────────────────────────────────────────────
    registry_path = output_root / "metrics" / "run_registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resolved config snapshot ──────────────────────────────────────
    train_cfg_for_trainer = dict(train_cfg)
    train_cfg_for_trainer["manifest_version"] = "v1.1"
    train_cfg_for_trainer["diff_ids_applied"] = ["d001", "d002"]
    train_cfg_for_trainer["ablation_encoder_id"] = args.encoder_id

    resolved = {
        "data": data_cfg,
        "model": model_cfg,
        "train": train_cfg_for_trainer,
        "ablation": {
            "encoder_id": args.encoder_id,
            "encoder_type": encoder_type,
            "d_enc": d_enc,
            "encoder_cfg": encoder_cfg,
            "frozen_constants": frozen,
        },
        "runtime": {
            "profile": profile,
            "device": str(device),
            "smoke": args.smoke,
            "seed": args.seed,
            "train_proteins": len(train_entries),
            "val_proteins": len(val_entries),
            "train_chunks": len(train_chunks),
            "val_chunks": len(val_chunks),
            "trainable_params": trainable,
            "total_params": total,
        },
    }
    with open(run_dir / "full_resolved_config.json", "w") as f:
        json.dump(resolved, f, indent=2, default=str)

    # ── W&B ───────────────────────────────────────────────────────────
    wandb_cfg = None
    if args.wandb:
        wandb_cfg = {
            "enabled": True,
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": args.wandb_name or f"{args.encoder_id}_seed{args.seed}",
        }

    # ── Train ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg_for_trainer,
        run_dir=run_dir,
        device=device,
        registry_path=registry_path,
        wandb_cfg=wandb_cfg,
        val_entries=val_entries,
        tokenizer=tokenizer,
        min_k=data_cfg["min_k"],
        max_k=data_cfg["max_k"],
        context_len=chunking["context_len"],
    )

    logger.info("=" * 60)
    logger.info("Starting ablation: encoder=%s, seed=%d, epochs=%d, device=%s",
                args.encoder_id, args.seed, train_cfg_for_trainer["max_epochs"], device)
    logger.info("=" * 60)

    summary = trainer.fit()

    logger.info("=" * 60)
    logger.info("Training complete. Summary:")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    main()
