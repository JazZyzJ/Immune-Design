"""Train Epitope Head v1 scorer.

Wires A-G modules into an executable training pipeline.
Supports local smoke testing (--smoke --mock-encoder) and full cluster runs.

Usage:
    # Local smoke (no ESM download, CPU, 2 epochs, 20 proteins):
    python scripts/train_v1.py --smoke --mock-encoder --device cpu

    # Local full strict run on CPU (slow but verifies pipeline):
    python scripts/train_v1.py --device cpu --profile strict

    # Cluster GPU run:
    python scripts/train_v1.py --device cuda --profile strict
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from epitope_head.configs import (
    load_data_config,
    load_model_config,
    load_train_config,
)
from epitope_head.training.datamodule import (
    build_chunk_samples,
    build_dataloader,
    load_split_proteins,
)
from epitope_head.training.model import EpitopeScorer, FrozenESMEncoder, ESMTokenizer
from epitope_head.training.trainer import Trainer

logger = logging.getLogger(__name__)


# ── Mock encoder / tokenizer for pipeline testing without ESM ──────────────

class _MockFrozenEncoder(nn.Module):
    """Deterministic mock encoder matching FrozenESMEncoder interface.

    Uses a learnable embedding (frozen) so outputs are deterministic
    given the same token_ids, but does not require the `esm` package.
    """

    def __init__(self, d_enc: int = 1280, vocab_size: int = 33):
        super().__init__()
        self.d_enc = d_enc
        self.embed = nn.Embedding(vocab_size, d_enc)
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        self.training = mode
        return self

    @torch.no_grad()
    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = token_ids.shape
        hidden = self.embed(token_ids.clamp(0, self.embed.num_embeddings - 1))
        lengths = attention_mask.sum(dim=1) - 2
        L_max = int(lengths.max().item())
        embeddings = torch.zeros(B, L_max, self.d_enc, device=hidden.device, dtype=hidden.dtype)
        for i in range(B):
            L_i = int(lengths[i].item())
            embeddings[i, :L_i] = hidden[i, 1 : 1 + L_i]
        return embeddings, lengths


class _MockTokenizer:
    """Tokenizer matching ESMTokenizer interface without ESM alphabet."""

    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self) -> None:
        self.cls_idx = 0
        self.eos_idx = 2
        self.pad_idx = 1
        self._aa_map: dict[str, int] = {aa: i + 4 for i, aa in enumerate(self.AA_ORDER)}
        self._unk_idx = 3

    def __call__(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        max_len = max(len(s) for s in sequences) + 2
        B = len(sequences)
        token_ids = torch.full((B, max_len), self.pad_idx, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            tokens = [self.cls_idx]
            for ch in seq:
                tokens.append(self._aa_map.get(ch, self._unk_idx))
            tokens.append(self.eos_idx)
            t = len(tokens)
            token_ids[i, :t] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[i, :t] = True
        return {"token_ids": token_ids, "attention_mask": attention_mask}


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Epitope Head v1 scorer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config-dir", type=str,
                    default=str(PROJECT_ROOT / "epitope_head" / "configs"),
                    help="Directory containing data.yaml, model.yaml, train.yaml")
    p.add_argument("--data-dir", type=str, default=None,
                    help="Directory containing manifests (parquet + splits). "
                         "Defaults to PROJECT_ROOT/outputs/manifests")
    p.add_argument("--profile", type=str, default="strict",
                    choices=["strict", "balanced"],
                    help="Data profile to train on")
    p.add_argument("--device", type=str, default="cpu",
                    help="Torch device (cpu, cuda, cuda:0, mps)")
    p.add_argument("--run-dir", type=str, default=None,
                    help="Override run output directory")
    p.add_argument("--registry-path", type=str, default=None,
                    help="Path to run_registry.jsonl (default: outputs/metrics/run_registry.jsonl)")

    smoke = p.add_argument_group("smoke testing")
    smoke.add_argument("--smoke", action="store_true",
                       help="Smoke mode: small data + few epochs for pipeline verification")
    smoke.add_argument("--smoke-n-train", type=int, default=20,
                       help="Number of training proteins in smoke mode")
    smoke.add_argument("--smoke-n-val", type=int, default=5,
                       help="Number of validation proteins in smoke mode")
    smoke.add_argument("--smoke-epochs", type=int, default=2,
                       help="Max epochs in smoke mode")

    encoder = p.add_argument_group("encoder options")
    encoder.add_argument("--mock-encoder", action="store_true",
                         help="Use mock encoder (no ESM download; for pipeline testing)")

    wb = p.add_argument_group("wandb")
    wb.add_argument("--wandb", action="store_true",
                    help="Enable Weights & Biases logging")
    wb.add_argument("--wandb-entity", type=str, default=None,
                    help="W&B entity (team/user)")
    wb.add_argument("--wandb-project", type=str, default="Immune-Design",
                    help="W&B project name")
    wb.add_argument("--wandb-name", type=str, default=None,
                    help="W&B run name (default: auto-generated run_id)")

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.smoke else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    config_dir = Path(args.config_dir)
    data_cfg = load_data_config(config_dir / "data.yaml")
    model_cfg = load_model_config(config_dir / "model.yaml")
    train_cfg = load_train_config(config_dir / "train.yaml")

    # ── Reproducibility ───────────────────────────────────────────────
    seed = train_cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if train_cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True)
    logger.info("Seed=%d, deterministic=%s", seed, train_cfg.get("deterministic", False))

    # ── Resolve data paths ────────────────────────────────────────────
    manifest_dir = Path(args.data_dir) if args.data_dir else (
        PROJECT_ROOT / "outputs" / "manifests"
    )
    splits_dir = manifest_dir / "splits"
    profile = args.profile

    samples_path = manifest_dir / f"protein_samples_{profile}.parquet"
    train_ids_path = splits_dir / profile / "train_ids.txt"
    val_ids_path = splits_dir / profile / "val_ids.txt"

    for p in [samples_path, train_ids_path, val_ids_path]:
        if not p.exists():
            logger.error("Required artifact missing: %s", p)
            logger.error("Run data pipeline first: python -m epitope_head.data.build_dataset --stage abcd")
            sys.exit(1)

    # ── Load protein entries ──────────────────────────────────────────
    logger.info("Loading %s profile data...", profile)
    train_entries = load_split_proteins(samples_path, train_ids_path)
    val_entries = load_split_proteins(samples_path, val_ids_path)
    logger.info("Loaded %d train proteins, %d val proteins", len(train_entries), len(val_entries))

    # ── Smoke mode overrides ──────────────────────────────────────────
    if args.smoke:
        train_entries = train_entries[: args.smoke_n_train]
        val_entries = val_entries[: args.smoke_n_val]
        train_cfg["max_epochs"] = args.smoke_epochs
        train_cfg["checkpoint_every_n_epochs"] = 1
        train_cfg["early_stopping_patience"] = args.smoke_epochs + 1
        train_cfg["num_workers"] = 0
        logger.info(
            "SMOKE MODE: %d train proteins, %d val proteins, %d epochs",
            len(train_entries), len(val_entries), train_cfg["max_epochs"],
        )

    # macOS fork safety: force num_workers=0 on Darwin
    if sys.platform == "darwin" and train_cfg["num_workers"] > 0:
        logger.warning("macOS detected: forcing num_workers=0 (fork safety)")
        train_cfg["num_workers"] = 0

    # ── Build chunk samples ───────────────────────────────────────────
    chunking = train_cfg["chunking"]
    ck_params = dict(
        context_len=chunking["context_len"],
        stride=chunking["stride"],
        margin=chunking["margin"],
    )
    train_chunks = build_chunk_samples(train_entries, **ck_params)
    val_chunks = build_chunk_samples(val_entries, **ck_params)
    logger.info("Chunks: %d train, %d val", len(train_chunks), len(val_chunks))

    # ── Build encoder + tokenizer ─────────────────────────────────────
    device = torch.device(args.device)
    d_enc = model_cfg["d_enc"]

    if args.mock_encoder:
        logger.info("Using MOCK encoder (d_enc=%d) — pipeline test only", d_enc)
        encoder = _MockFrozenEncoder(d_enc=d_enc)
        tokenizer = _MockTokenizer()
    else:
        logger.info("Loading ESM-2 model: %s ...", model_cfg["encoder_name"])
        try:
            import esm
        except ImportError:
            logger.error(
                "Package 'fair-esm' not installed. "
                "Install with: pip install fair-esm\n"
                "Or use --mock-encoder for pipeline testing without ESM."
            )
            sys.exit(1)

        esm_model, alphabet = getattr(esm.pretrained, model_cfg["encoder_name"])()
        encoder = FrozenESMEncoder(esm_model, d_enc=d_enc)
        tokenizer = ESMTokenizer(alphabet)
        logger.info("ESM-2 loaded. Encoder params: %d (all frozen)",
                     sum(p.numel() for p in encoder.parameters()))

    # ── Build dataloaders ─────────────────────────────────────────────
    train_loader = build_dataloader(
        train_chunks,
        max_tokens=train_cfg["max_tokens"],
        shuffle=True,
        seed=train_cfg["seed"],
        num_workers=train_cfg["num_workers"],
        tokenize_fn=tokenizer,
    )
    val_loader = build_dataloader(
        val_chunks,
        max_tokens=train_cfg["max_tokens"],
        shuffle=False,
        seed=train_cfg["seed"],
        num_workers=0,
        tokenize_fn=tokenizer,
    )
    logger.info("DataLoaders built: ~%d train batches, ~%d val batches",
                len(train_loader), len(val_loader))

    # ── Build model ───────────────────────────────────────────────────
    model = EpitopeScorer(
        encoder=encoder,
        d_enc=d_enc,
        d_proj=model_cfg["d_proj"],
        length_emb_dim=model_cfg["length_embedding_dim"],
        allele_emb_dim=model_cfg["allele_embedding_dim"],
        min_k=data_cfg["min_k"],
        max_k=data_cfg["max_k"],
        n_alleles=1,
        scorer_hidden_dim=model_cfg["scorer_hidden_dim"],
        scorer_activation=model_cfg["scorer_activation"],
        scorer_dropout=model_cfg.get("scorer_dropout", 0.1),
        logit_scale_init=model_cfg.get("logit_scale_init", 10.0),
        logit_scale_max=model_cfg.get("logit_scale_max", 20.0),
        projection_layer_norm=model_cfg.get("projection_layer_norm", True),
        pad_left_init=model_cfg.get("pad_left_init", "zeros"),
        pad_right_init=model_cfg.get("pad_right_init", "zeros"),
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model built: %d trainable / %d total params", trainable_params, total_params)

    # ── Run directory ─────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "smoke" if args.smoke else profile
    run_dir = Path(args.run_dir) if args.run_dir else (
        PROJECT_ROOT / "outputs" / "runs" / f"run_{timestamp}_{suffix}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # ── Resolved config snapshot ──────────────────────────────────────
    train_cfg_for_trainer = dict(train_cfg)
    train_cfg_for_trainer["manifest_version"] = "v1.1"
    train_cfg_for_trainer["diff_ids_applied"] = ["d001", "d002"]

    resolved = {
        "data": data_cfg,
        "model": model_cfg,
        "train": train_cfg_for_trainer,
        "runtime": {
            "profile": profile,
            "device": str(device),
            "mock_encoder": args.mock_encoder,
            "smoke": args.smoke,
            "train_proteins": len(train_entries),
            "val_proteins": len(val_entries),
            "train_chunks": len(train_chunks),
            "val_chunks": len(val_chunks),
        },
    }
    with open(run_dir / "full_resolved_config.json", "w") as f:
        json.dump(resolved, f, indent=2, default=str)

    # ── Registry path ─────────────────────────────────────────────────
    registry_path = (
        Path(args.registry_path) if args.registry_path
        else PROJECT_ROOT / "outputs" / "metrics" / "run_registry.jsonl"
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # ── W&B config ─────────────────────────────────────────────────────
    wandb_cfg = None
    if args.wandb:
        wandb_cfg = {
            "enabled": True,
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": args.wandb_name,
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
    )

    logger.info("=" * 60)
    logger.info("Starting training: %d epochs, profile=%s, device=%s",
                train_cfg_for_trainer["max_epochs"], profile, device)
    logger.info("=" * 60)

    summary = trainer.fit()

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("Run summary:")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 60)

    summary_path = run_dir / "run_summary.json"
    logger.info("Summary written to %s", summary_path)
    logger.info("Best checkpoint: %s", run_dir / "best.pt")

    return summary


if __name__ == "__main__":
    main()
