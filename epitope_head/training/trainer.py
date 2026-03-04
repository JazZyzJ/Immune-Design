"""Trainer loop with sanity metrics, checkpointing, and run metadata.

Implements PLAN.md Tasks E5 (trainer loop), E6 (checkpointing/logging), E7 (smoke-ready).
  - Train/val step plumbing with optimizer, scheduler, grad clipping
  - Frozen encoder guard: encoder params excluded from optimizer
  - NaN/Inf guard with fail-fast
  - Sanity metrics: mean_pos_logit, mean_neg_logit, logit_gap, per_protein_auc
  - Epoch checkpoints + best checkpoint by monitor_metric
  - JSONL train/val logs with stable key schema
  - Resolved config snapshot + run summary
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from epitope_head.training.eval_metrics import full_val_eval
from epitope_head.training.losses import compute_loss
from epitope_head.training.negatives import sample_negatives
from epitope_head.training.registry import (
    append_registry_row,
    build_registry_row,
    compute_protocol_signature,
    generate_run_id,
)

logger = logging.getLogger(__name__)


# ── Sanity Metrics ──────────────────────────────────────────────────────────

@dataclass
class StepMetrics:
    """Metrics from a single train/val step (one chunk)."""
    loss_total: float = 0.0
    loss_intra: float = 0.0
    loss_mp: float = 0.0
    loss_smooth: float = 0.0
    mean_pos_logit: float = 0.0
    mean_neg_logit: float = 0.0
    logit_gap: float = 0.0
    per_protein_auc: float | None = None
    n_pos: int = 0
    n_neg: int = 0

    def to_dict(self) -> dict:
        return {
            "loss_total": self.loss_total,
            "loss_intra": self.loss_intra,
            "loss_mp": self.loss_mp,
            "loss_smooth": self.loss_smooth,
            "mean_pos_logit": self.mean_pos_logit,
            "mean_neg_logit": self.mean_neg_logit,
            "logit_gap": self.logit_gap,
            "per_protein_auc": self.per_protein_auc,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
        }


def _compute_binary_auc(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
) -> float | None:
    """Compute AUC from positive/negative logits; returns None if undefined."""
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return None
    # Pairwise AUC: P(score_pos > score_neg) + 0.5 * P(tie)
    cmp = (pos_logits.unsqueeze(1) > neg_logits.unsqueeze(0)).float()
    tie = (pos_logits.unsqueeze(1) == neg_logits.unsqueeze(0)).float()
    return float((cmp + 0.5 * tie).mean().item())


def compute_sanity_metrics(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    loss_dict: dict[str, torch.Tensor],
) -> StepMetrics:
    """Compute sanity metrics from logits and loss terms."""
    mean_pos = pos_logits.mean().item() if pos_logits.numel() > 0 else 0.0
    mean_neg = neg_logits.mean().item() if neg_logits.numel() > 0 else 0.0
    auc = _compute_binary_auc(pos_logits, neg_logits)
    return StepMetrics(
        loss_total=loss_dict["loss_total"].item(),
        loss_intra=loss_dict["loss_intra"].item(),
        loss_mp=loss_dict["loss_mp"].item(),
        loss_smooth=loss_dict["loss_smooth"].item(),
        mean_pos_logit=mean_pos,
        mean_neg_logit=mean_neg,
        logit_gap=mean_pos - mean_neg,
        per_protein_auc=auc,
        n_pos=pos_logits.numel(),
        n_neg=neg_logits.numel(),
    )


def aggregate_epoch_metrics(step_metrics_list: list[StepMetrics]) -> dict:
    """Aggregate step metrics into epoch-level summary.

    Excludes empty-positive chunks (n_pos == 0) to avoid diluting metrics.
    """
    # Filter out steps with no positives (empty chunks contribute no gradient)
    step_metrics_list = [m for m in step_metrics_list if m.n_pos > 0]
    if not step_metrics_list:
        return {}
    n = len(step_metrics_list)
    auc_values = [m.per_protein_auc for m in step_metrics_list if m.per_protein_auc is not None]
    return {
        "loss_total": sum(m.loss_total for m in step_metrics_list) / n,
        "loss_intra": sum(m.loss_intra for m in step_metrics_list) / n,
        "loss_mp": sum(m.loss_mp for m in step_metrics_list) / n,
        "loss_smooth": sum(m.loss_smooth for m in step_metrics_list) / n,
        "mean_pos_logit": sum(m.mean_pos_logit for m in step_metrics_list) / n,
        "mean_neg_logit": sum(m.mean_neg_logit for m in step_metrics_list) / n,
        "logit_gap": sum(m.logit_gap for m in step_metrics_list) / n,
        "per_protein_auc": (sum(auc_values) / len(auc_values)) if auc_values else None,
        "total_pos": sum(m.n_pos for m in step_metrics_list),
        "total_neg": sum(m.n_neg for m in step_metrics_list),
        "n_steps": n,
    }


# ── NaN Guard ───────────────────────────────────────────────────────────────

class NaNDetected(RuntimeError):
    """Raised when NaN or Inf is detected in loss or logits."""
    pass


def nan_guard(tensor: torch.Tensor, name: str = "tensor"):
    """Check for NaN/Inf and raise immediately."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise NaNDetected(f"NaN/Inf detected in {name}: {tensor}")


# ── Optimizer Builder ───────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build optimizer with only learnable (non-frozen) parameters.

    Splits params into decay (Linear.weight only) and no-decay groups
    (bias, LayerNorm, Embedding, learnable Parameters) to avoid
    regularizing normalization/embedding/bias terms.
    """
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Only apply weight decay to Linear weight matrices
        if p.dim() >= 2 and "embedding" not in name.lower():
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr)
    elif optimizer_name == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    warmup_steps: int = 500,
    max_epochs: int = 50,
    steps_per_epoch: int = 100,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build LR scheduler with optional warmup."""
    total_steps = max_epochs * steps_per_epoch
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps),
        )
    elif scheduler_name == "constant":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def normalize_loss_cfg(loss_cfg: dict) -> dict:
    """Normalize train config loss keys to compute_loss(...) signature keys."""
    normalized = dict(loss_cfg)
    if "tau_mp" in normalized:
        if "T_mp" in normalized and normalized["T_mp"] != normalized["tau_mp"]:
            raise ValueError("Both 'tau_mp' and 'T_mp' provided with different values")
        normalized["T_mp"] = normalized.pop("tau_mp")

    required = {"tau", "T_mp", "lambda_mp", "lambda_smooth"}
    missing = required - set(normalized.keys())
    if missing:
        raise ValueError(f"Loss config missing required keys after normalization: {sorted(missing)}")

    unknown = set(normalized.keys()) - required
    if unknown:
        raise ValueError(f"Loss config has unknown keys: {sorted(unknown)}")

    return normalized


# ── Train / Val Step ────────────────────────────────────────────────────────

def prepare_chunk_spans(
    batch: dict,
    batch_idx: int,
    neg_ratio: int = 7,
    hard_negative_fraction: float = 0.3,
    hard_neg_max_overlap_ratio: float = 0.8,
    hard_neg_offset_range: int = 5,
    neg_length_sampling: str = "match_positive",
    min_k: int = 12,
    max_k: int = 25,
    rng: np.random.RandomState | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Prepare positive/negative spans for each chunk in batch.

    Converts protein-global positives to chunk-local coordinates,
    filters to spans within chunk bounds, samples negatives.

    Returns:
        pos_spans_list: list of [P_i, 2] tensors (chunk-local)
        neg_spans_list: list of [N_i, 2] tensors (chunk-local)
        allele_pos_list: list of [P_i] allele index tensors
        allele_neg_list: list of [N_i] allele index tensors
    """
    B = len(batch["protein_ids"])
    pos_spans_list = []
    neg_spans_list = []
    allele_pos_list = []
    allele_neg_list = []

    for i in range(B):
        chunk_start = batch["chunk_starts"][i].item()
        chunk_end = batch["chunk_ends"][i].item()
        chunk_len = chunk_end - chunk_start
        positives = batch["positives"][i]

        # Filter positives to those within chunk bounds (chunk-local coords)
        local_positives = []
        for p in positives:
            s, e = p["start_0b"], p["end_0b"]
            if s >= chunk_start and e <= chunk_end:
                local_positives.append({
                    "start_0b": s - chunk_start,
                    "end_0b": e - chunk_start,
                    "pep_len": p["pep_len"],
                })

        if local_positives:
            pos_spans = torch.tensor(
                [[p["start_0b"], p["end_0b"]] for p in local_positives],
                dtype=torch.long,
            )
            # Sample negatives in chunk-local space
            negs = sample_negatives(
                protein_length=chunk_len,
                positives=local_positives,
                neg_ratio=neg_ratio,
                hard_negative_fraction=hard_negative_fraction,
                hard_neg_max_overlap_ratio=hard_neg_max_overlap_ratio,
                hard_neg_offset_range=hard_neg_offset_range,
                neg_length_sampling=neg_length_sampling,
                min_k=min_k,
                max_k=max_k,
                rng=rng,
                strict=False,
            )
            neg_spans = torch.tensor(
                [[n["start_0b"], n["end_0b"]] for n in negs],
                dtype=torch.long,
            ) if negs else torch.zeros(0, 2, dtype=torch.long)
        else:
            pos_spans = torch.zeros(0, 2, dtype=torch.long)
            neg_spans = torch.zeros(0, 2, dtype=torch.long)

        pos_spans_list.append(pos_spans)
        neg_spans_list.append(neg_spans)
        allele_pos_list.append(torch.zeros(pos_spans.shape[0], dtype=torch.long))
        allele_neg_list.append(torch.zeros(neg_spans.shape[0], dtype=torch.long))

    return pos_spans_list, neg_spans_list, allele_pos_list, allele_neg_list


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    loss_cfg: dict,
    neg_cfg: dict,
    grad_clip: float = 1.0,
    scheduler=None,
    warmup_steps: int = 0,
    global_step: int = 0,
    rng: np.random.RandomState | None = None,
) -> StepMetrics:
    """Execute one training step on a batch of chunks.

    Returns aggregated StepMetrics across all chunks in the batch.
    """
    model.train()
    loss_cfg = normalize_loss_cfg(loss_cfg)

    pos_spans_list, neg_spans_list, allele_pos_list, allele_neg_list = \
        prepare_chunk_spans(batch, batch_idx=0, rng=rng, **neg_cfg)

    # Skip batch if no positives at all
    total_pos = sum(s.shape[0] for s in pos_spans_list)
    if total_pos == 0:
        return StepMetrics()

    # Forward: encode all chunks
    token_ids = batch["token_ids"]
    attention_mask = batch["attention_mask"]
    chunk_lengths = batch["chunk_ends"] - batch["chunk_starts"]

    # Combine pos + neg spans for model forward (move to same device as token_ids)
    device = token_ids.device
    all_spans_list = []
    all_allele_list = []
    pos_counts = []
    for ps, ns, ap, an in zip(pos_spans_list, neg_spans_list, allele_pos_list, allele_neg_list):
        combined_spans = torch.cat([ps, ns], dim=0) if ps.shape[0] > 0 else ns
        combined_allele = torch.cat([ap, an], dim=0) if ap.shape[0] > 0 else an
        all_spans_list.append(combined_spans.to(device))
        all_allele_list.append(combined_allele.to(device))
        pos_counts.append(ps.shape[0])

    logits_list = model(token_ids, attention_mask, all_spans_list, all_allele_list, chunk_lengths)

    # Compute per-chunk losses and aggregate
    all_pos_logits = []
    all_neg_logits = []
    total_loss = torch.tensor(0.0, device=token_ids.device)
    n_chunks_with_pos = 0

    for i, logits in enumerate(logits_list):
        pc = pos_counts[i]
        if pc == 0:
            continue
        pos_logits = logits[:pc]
        neg_logits = logits[pc:]

        nan_guard(pos_logits, f"pos_logits[chunk={i}]")
        nan_guard(neg_logits, f"neg_logits[chunk={i}]")

        loss_dict = compute_loss(pos_logits, neg_logits, **loss_cfg)
        nan_guard(loss_dict["loss_total"], f"loss_total[chunk={i}]")

        total_loss = total_loss + loss_dict["loss_total"]
        n_chunks_with_pos += 1
        all_pos_logits.append(pos_logits.detach())
        all_neg_logits.append(neg_logits.detach())

    if n_chunks_with_pos == 0:
        return StepMetrics()

    avg_loss = total_loss / n_chunks_with_pos

    # Backward + optimize
    optimizer.zero_grad()
    avg_loss.backward()

    if grad_clip > 0:
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            grad_clip,
        )

    optimizer.step()

    # Warmup: linear ramp from 0 to base_lr, then cosine decay
    if warmup_steps > 0 and global_step < warmup_steps:
        warmup_factor = (global_step + 1) / warmup_steps
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * warmup_factor
    elif scheduler is not None:
        scheduler.step()

    # Compute metrics
    cat_pos = torch.cat(all_pos_logits) if all_pos_logits else torch.tensor([])
    cat_neg = torch.cat(all_neg_logits) if all_neg_logits else torch.tensor([])
    loss_for_metrics = compute_loss(cat_pos, cat_neg, **loss_cfg)

    return compute_sanity_metrics(cat_pos, cat_neg, loss_for_metrics)


@torch.no_grad()
def val_step(
    model: nn.Module,
    batch: dict,
    loss_cfg: dict,
    neg_cfg: dict,
    rng: np.random.RandomState | None = None,
) -> StepMetrics:
    """Execute one validation step (no grad, no optimizer)."""
    model.eval()
    loss_cfg = normalize_loss_cfg(loss_cfg)

    pos_spans_list, neg_spans_list, allele_pos_list, allele_neg_list = \
        prepare_chunk_spans(batch, batch_idx=0, rng=rng, **neg_cfg)

    total_pos = sum(s.shape[0] for s in pos_spans_list)
    if total_pos == 0:
        return StepMetrics()

    token_ids = batch["token_ids"]
    attention_mask = batch["attention_mask"]
    chunk_lengths = batch["chunk_ends"] - batch["chunk_starts"]

    device = token_ids.device
    all_spans_list = []
    all_allele_list = []
    pos_counts = []
    for ps, ns, ap, an in zip(pos_spans_list, neg_spans_list, allele_pos_list, allele_neg_list):
        combined_spans = torch.cat([ps, ns], dim=0) if ps.shape[0] > 0 else ns
        combined_allele = torch.cat([ap, an], dim=0) if ap.shape[0] > 0 else an
        all_spans_list.append(combined_spans.to(device))
        all_allele_list.append(combined_allele.to(device))
        pos_counts.append(ps.shape[0])

    logits_list = model(token_ids, attention_mask, all_spans_list, all_allele_list, chunk_lengths)

    all_pos_logits = []
    all_neg_logits = []
    for i, logits in enumerate(logits_list):
        pc = pos_counts[i]
        if pc == 0:
            continue
        all_pos_logits.append(logits[:pc])
        all_neg_logits.append(logits[pc:])

    cat_pos = torch.cat(all_pos_logits) if all_pos_logits else torch.tensor([])
    cat_neg = torch.cat(all_neg_logits) if all_neg_logits else torch.tensor([])
    loss_dict = compute_loss(cat_pos, cat_neg, **loss_cfg)

    return compute_sanity_metrics(cat_pos, cat_neg, loss_dict)


# ── E6: Checkpointing & Logging ────────────────────────────────────────────

CHECKPOINT_METADATA_KEYS = frozenset({
    "epoch", "global_step", "monitor_metric", "monitor_value",
    "config_hash", "manifest_version", "diff_ids_applied",
})

LOG_ENTRY_KEYS = frozenset({
    "epoch", "phase", "loss_total", "loss_intra", "loss_mp", "loss_smooth",
    "mean_pos_logit", "mean_neg_logit", "logit_gap", "per_protein_auc",
    "total_pos", "total_neg", "n_steps", "timestamp",
})


def config_hash(cfg: dict) -> str:
    """Deterministic hash of resolved config."""
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    monitor_metric: str,
    monitor_value: float,
    cfg_hash: str,
    path: Path,
    manifest_version: str = "v1.1",
    diff_ids_applied: list[str] | None = None,
):
    """Save model checkpoint with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": {
            "epoch": epoch,
            "global_step": global_step,
            "monitor_metric": monitor_metric,
            "monitor_value": monitor_value,
            "config_hash": cfg_hash,
            "manifest_version": manifest_version,
            "diff_ids_applied": list(diff_ids_applied or []),
        },
    }, path)


def load_checkpoint(path: Path) -> dict:
    """Load checkpoint and validate metadata keys."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    missing = CHECKPOINT_METADATA_KEYS - set(meta.keys())
    if missing:
        raise ValueError(f"Checkpoint missing metadata keys: {missing}")
    return ckpt


def write_log_entry(log_path: Path, entry: dict):
    """Append a JSONL log entry with schema validation."""
    missing = LOG_ENTRY_KEYS - set(entry.keys())
    if missing:
        raise ValueError(f"Log entry missing keys: {missing}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def save_resolved_config(cfg: dict, run_dir: Path):
    """Save resolved config snapshot."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ── E5+E6+E7: Trainer ──────────────────────────────────────────────────────

class Trainer:
    """Training loop with checkpointing, logging, and early stopping.

    Wires together E1-E4 components into an executable training pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        train_cfg: dict,
        run_dir: Path,
        device: torch.device | str = "cpu",
        registry_path: Path | str | None = None,
        wandb_cfg: dict | None = None,
        # Per-protein full-scan evaluation (optional; enabled when all are provided)
        val_entries: list | None = None,
        tokenizer=None,
        min_k: int = 12,
        max_k: int = 25,
        context_len: int = 1022,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Per-protein eval resources
        self.val_entries = val_entries
        self.tokenizer = tokenizer
        self.min_k = min_k
        self.max_k = max_k
        self.context_len = context_len
        self._pp_eval_enabled = (val_entries is not None and tokenizer is not None)
        self.cfg = train_cfg
        self.run_dir = Path(run_dir)
        self.registry_path = Path(registry_path) if registry_path is not None else None

        # Build optimizer (only learnable params)
        self.optimizer = build_optimizer(
            model,
            optimizer_name=train_cfg["optimizer"],
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        # Store initial_lr for warmup linear ramp
        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        # Build scheduler
        steps_per_epoch = len(train_loader) if hasattr(train_loader, '__len__') else 100
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_name=train_cfg["scheduler"],
            warmup_steps=train_cfg["warmup_steps"],
            max_epochs=train_cfg["max_epochs"],
            steps_per_epoch=steps_per_epoch,
        )

        # Loss config (normalized to compute_loss signature)
        self.loss_cfg = normalize_loss_cfg(train_cfg["loss"])

        # Negative sampling config
        self.neg_cfg = {
            "neg_ratio": train_cfg["neg_ratio"],
            "hard_negative_fraction": train_cfg["hard_negative_fraction"],
            "hard_neg_max_overlap_ratio": train_cfg["hard_neg_max_overlap_ratio"],
            "hard_neg_offset_range": train_cfg["hard_neg_offset_range"],
            "neg_length_sampling": train_cfg["neg_length_sampling"],
        }

        # Checkpointing
        self.cfg_hash = config_hash(train_cfg)
        self.monitor_metric = train_cfg["monitor_metric"]
        # Determine early stopping direction: loss metrics are minimized, others maximized
        self.monitor_mode = "min" if "loss" in self.monitor_metric else "max"
        self.best_monitor_value = float("inf") if self.monitor_mode == "min" else float("-inf")
        self.patience_counter = 0
        self.early_stopping_patience = train_cfg["early_stopping_patience"]
        self.checkpoint_every_n = train_cfg["checkpoint_every_n_epochs"]
        self.diff_ids_applied = list(train_cfg.get("diff_ids_applied", []))

        # Logging
        self.train_log_path = self.run_dir / "train_log.jsonl"
        self.val_log_path = self.run_dir / "val_log.jsonl"
        self.global_step = 0

        # Separate RNGs for train/val negative sampling (avoid cross-contamination)
        self.train_rng = np.random.RandomState(train_cfg["seed"])
        self.val_rng = np.random.RandomState(train_cfg["seed"] + 1)

        # Run identity (G1/G3)
        self.run_id = generate_run_id(seed_str=self.cfg_hash)
        self.manifest_version = str(train_cfg.get("manifest_version", "v1.1"))
        self.protocol_signature = compute_protocol_signature(train_cfg)

        # Save resolved config
        save_resolved_config(train_cfg, self.run_dir)

        # W&B integration
        self._wandb = None
        if wandb_cfg and wandb_cfg.get("enabled", False):
            try:
                import wandb
                self._wandb = wandb
                wandb.init(
                    entity=wandb_cfg.get("entity"),
                    project=wandb_cfg.get("project", "Immune-Design"),
                    name=wandb_cfg.get("name", self.run_id),
                    config=train_cfg,
                    dir=str(self.run_dir),
                    resume="allow",
                )
                wandb.watch(self.model, log="gradients", log_freq=50)
                logger.info("W&B initialized: %s", wandb.run.url or wandb.run.id)
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")
            except Exception as e:
                logger.warning("wandb init failed: %s", e)

    def _frozen_encoder_guard(self):
        """Verify encoder params still have requires_grad=False after step.

        Only applies when the encoder is expected to be frozen (e.g. ESM-2).
        Trainable encoders (E1/E2) skip this check.
        """
        encoder = self.model.encoder
        # Skip guard if encoder has any trainable params (it's intentionally trainable)
        has_trainable = any(p.requires_grad for p in encoder.parameters())
        if has_trainable:
            # For trainable encoders, this guard is not applicable
            # Check if the encoder was supposed to be frozen (FrozenESMEncoder pattern)
            if not hasattr(encoder, 'esm'):
                return  # Non-ESM encoder, trainable by design
        for name, p in encoder.named_parameters():
            if p.requires_grad:
                raise RuntimeError(
                    f"Frozen encoder param '{name}' has requires_grad=True after step"
                )

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch. Returns epoch metrics dict."""
        step_metrics_list = []

        if hasattr(self.train_loader, 'batch_sampler') and \
           hasattr(self.train_loader.batch_sampler, 'set_epoch'):
            self.train_loader.batch_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move tensors to device
            batch = self._to_device(batch)

            metrics = train_step(
                model=self.model,
                batch=batch,
                optimizer=self.optimizer,
                loss_cfg=self.loss_cfg,
                neg_cfg=self.neg_cfg,
                grad_clip=self.cfg["grad_clip"],
                scheduler=self.scheduler,
                warmup_steps=self.cfg["warmup_steps"],
                global_step=self.global_step,
                rng=self.train_rng,
            )
            step_metrics_list.append(metrics)
            self.global_step += 1

        # Frozen encoder guard (after full epoch)
        self._frozen_encoder_guard()

        epoch_metrics = aggregate_epoch_metrics(step_metrics_list)
        epoch_metrics["epoch"] = epoch
        epoch_metrics["phase"] = "train"
        epoch_metrics["timestamp"] = time.time()

        write_log_entry(self.train_log_path, epoch_metrics)
        return epoch_metrics

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        """Run one validation epoch. Returns epoch metrics dict."""
        # Reset val RNG each epoch for deterministic, comparable val metrics
        self.val_rng = np.random.RandomState(self.cfg["seed"] + 1)
        step_metrics_list = []

        for batch_idx, batch in enumerate(self.val_loader):
            batch = self._to_device(batch)
            metrics = val_step(
                model=self.model,
                batch=batch,
                loss_cfg=self.loss_cfg,
                neg_cfg=self.neg_cfg,
                rng=self.val_rng,
            )
            step_metrics_list.append(metrics)

        epoch_metrics = aggregate_epoch_metrics(step_metrics_list)
        epoch_metrics["epoch"] = epoch
        epoch_metrics["phase"] = "val"
        epoch_metrics["timestamp"] = time.time()

        # Per-protein full-window-scan evaluation
        if self._pp_eval_enabled:
            pp_metrics = full_val_eval(
                model=self.model,
                tokenizer=self.tokenizer,
                val_entries=self.val_entries,
                min_k=self.min_k,
                max_k=self.max_k,
                device=self.device,
                recall_ks=(50, 100),
                context_len=self.context_len,
            )
            epoch_metrics.update(pp_metrics)

        write_log_entry(self.val_log_path, epoch_metrics)
        return epoch_metrics

    def fit(self, max_epochs: int | None = None) -> dict:
        """Full training loop with checkpointing and early stopping.

        Returns run summary dict.
        """
        max_epochs = max_epochs or self.cfg["max_epochs"]
        logger.info("Starting training for %d epochs, run_dir=%s", max_epochs, self.run_dir)

        for epoch in range(max_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)

            monitor_val = val_metrics.get(self.monitor_metric)
            if monitor_val is None:
                monitor_val = float("-inf") if self.monitor_mode == "max" else float("inf")

            # Checkpoint every N epochs
            if (epoch + 1) % self.checkpoint_every_n == 0:
                ckpt_path = self.run_dir / f"epoch_{epoch}.pt"
                save_checkpoint(
                    self.model, self.optimizer, epoch, self.global_step,
                    self.monitor_metric, monitor_val, self.cfg_hash, ckpt_path,
                    diff_ids_applied=self.diff_ids_applied,
                )

            # Best checkpoint (direction-aware: min for loss, max for gap/auc)
            improved = (monitor_val < self.best_monitor_value) if self.monitor_mode == "min" \
                else (monitor_val > self.best_monitor_value)
            if improved:
                self.best_monitor_value = monitor_val
                self.patience_counter = 0
                best_path = self.run_dir / "best.pt"
                save_checkpoint(
                    self.model, self.optimizer, epoch, self.global_step,
                    self.monitor_metric, monitor_val, self.cfg_hash, best_path,
                    diff_ids_applied=self.diff_ids_applied,
                )
            else:
                self.patience_counter += 1

            # W&B epoch logging
            if self._wandb is not None:
                log_dict = {"epoch": epoch}
                for k, v in train_metrics.items():
                    if isinstance(v, (int, float)):
                        log_dict[f"train/{k}"] = v
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        log_dict[f"val/{k}"] = v
                log_dict["lr"] = self.optimizer.param_groups[0]["lr"]
                log_dict["patience"] = self.patience_counter
                if hasattr(self.model, "scorer") and hasattr(self.model.scorer, "log_logit_scale"):
                    log_dict["logit_scale"] = self.model.scorer.log_logit_scale.exp().item()
                self._wandb.log(log_dict, step=epoch)

            pp_auc_str = "%.4f" % val_metrics["pp_auc"] if val_metrics.get("pp_auc") is not None else "N/A"
            pp_ap_str = "%.4f" % val_metrics["pp_ap"] if val_metrics.get("pp_ap") is not None else "N/A"
            logger.info(
                "Epoch %d — train_loss=%.4f, val_loss=%.4f, "
                "pp_auc=%s, pp_ap=%s, patience=%d/%d",
                epoch, train_metrics.get("loss_total", 0),
                val_metrics.get("loss_total", 0),
                pp_auc_str, pp_ap_str,
                self.patience_counter, self.early_stopping_patience,
            )

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Run summary (includes run_id for G3 registry integration)
        summary = {
            "run_id": self.run_id,
            "final_epoch": epoch,
            "best_monitor_value": self.best_monitor_value,
            "monitor_metric": self.monitor_metric,
            "global_steps": self.global_step,
            "config_hash": self.cfg_hash,
            "run_dir": str(self.run_dir),
            "manifest_version": self.manifest_version,
            "protocol_signature": self.protocol_signature,
        }

        with open(self.run_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # G3: auto-append registry row if registry_path configured
        if self.registry_path is not None:
            best_ckpt = self.run_dir / "best.pt"
            if best_ckpt.exists():
                primary_metrics = {
                    self.monitor_metric: self.best_monitor_value,
                }
                if "logit_gap" not in primary_metrics:
                    primary_metrics["logit_gap"] = self.best_monitor_value
                row = build_registry_row(
                    run_id=self.run_id,
                    run_dir=self.run_dir,
                    best_checkpoint_path=best_ckpt,
                    config_hash=self.cfg_hash,
                    manifest_version=self.manifest_version,
                    diff_ids_applied=self.diff_ids_applied,
                    protocol_signature=self.protocol_signature,
                    primary_metrics=primary_metrics,
                )
                append_registry_row(row, self.registry_path)
                logger.info("Registry row appended for run_id=%s", self.run_id)
            else:
                logger.warning(
                    "Skipping registry write: best.pt not found at %s", best_ckpt,
                )

        if self._wandb is not None:
            self._wandb.log({"best_monitor_value": self.best_monitor_value})
            self._wandb.finish()

        return summary

    def _to_device(self, batch: dict) -> dict:
        """Move tensor values in batch to device."""
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved


# ── E7: Smoke Diagnostics ──────────────────────────────────────────────────

def verify_run_artifacts(run_dir: Path, n_epochs: int, checkpoint_every_n: int) -> dict:
    """Verify all expected training artifacts exist and are loadable.

    Returns dict with 'ok' bool and 'missing'/'errors' lists.
    """
    run_dir = Path(run_dir)
    missing = []
    errors = []

    # Required files
    required = ["train_log.jsonl", "val_log.jsonl", "resolved_config.yaml",
                 "run_summary.json", "best.pt"]
    for f in required:
        if not (run_dir / f).exists():
            missing.append(f)

    # Epoch checkpoints
    for e in range(n_epochs):
        if (e + 1) % checkpoint_every_n == 0:
            name = f"epoch_{e}.pt"
            if not (run_dir / name).exists():
                missing.append(name)

    # Loadability checks
    for pt_file in run_dir.glob("*.pt"):
        try:
            ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
            meta = ckpt.get("metadata", {})
            meta_missing = CHECKPOINT_METADATA_KEYS - set(meta.keys())
            if meta_missing:
                errors.append(f"{pt_file.name}: missing metadata keys {meta_missing}")
        except Exception as exc:
            errors.append(f"{pt_file.name}: load failed: {exc}")

    # Log schema checks
    for log_name in ["train_log.jsonl", "val_log.jsonl"]:
        log_path = run_dir / log_name
        if log_path.exists():
            with open(log_path) as f:
                for i, line in enumerate(f):
                    entry = json.loads(line)
                    entry_missing = LOG_ENTRY_KEYS - set(entry.keys())
                    if entry_missing:
                        errors.append(f"{log_name}:{i}: missing keys {entry_missing}")

    # Run summary loadability
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            for key in ["final_epoch", "best_monitor_value", "config_hash"]:
                if key not in summary:
                    errors.append(f"run_summary.json missing '{key}'")
        except Exception as exc:
            errors.append(f"run_summary.json: load failed: {exc}")

    return {"ok": len(missing) == 0 and len(errors) == 0,
            "missing": missing, "errors": errors}


def compute_run_digest(run_dir: Path) -> str:
    """Compute a deterministic digest of training logs for reproducibility comparison.

    Hashes the content of train_log.jsonl + val_log.jsonl + run_summary.json.
    """
    run_dir = Path(run_dir)
    h = hashlib.sha256()
    for fname in ["train_log.jsonl", "val_log.jsonl", "run_summary.json"]:
        fpath = run_dir / fname
        if fpath.exists():
            h.update(fpath.read_bytes())
    return h.hexdigest()[:16]


def boundary_distance_bucket_stats(
    span_starts: list[int],
    chunk_starts: list[int],
    chunk_ends: list[int],
    scores: list[float],
    n_buckets: int = 4,
) -> list[dict]:
    """Compute per-bucket score statistics by distance from chunk boundary.

    For each span, d_boundary = min(span_start - chunk_start, chunk_end - span_start).
    Spans are grouped into equal-frequency buckets.

    Returns list of dicts with {bucket, d_min, d_max, mean_score, std_score, count}.
    """
    if not span_starts:
        return []

    distances = []
    for s, cs, ce in zip(span_starts, chunk_starts, chunk_ends):
        d = min(s - cs, ce - s)
        distances.append(d)

    # Sort by distance and partition into buckets
    indexed = sorted(zip(distances, scores), key=lambda x: x[0])
    bucket_size = max(1, len(indexed) // n_buckets)

    buckets = []
    for b in range(n_buckets):
        start_idx = b * bucket_size
        end_idx = start_idx + bucket_size if b < n_buckets - 1 else len(indexed)
        if start_idx >= len(indexed):
            break
        bucket_items = indexed[start_idx:end_idx]
        ds = [x[0] for x in bucket_items]
        ss = [x[1] for x in bucket_items]
        buckets.append({
            "bucket": b,
            "d_min": min(ds),
            "d_max": max(ds),
            "mean_score": sum(ss) / len(ss),
            "std_score": float(np.std(ss)) if len(ss) > 1 else 0.0,
            "count": len(ss),
        })

    return buckets


def long_vs_short_comparison(
    protein_lengths: list[int],
    logit_gaps: list[float],
    aucs: list[float | None],
    threshold: int = 1022,
) -> dict:
    """Compare metrics between long (chunked) and short (single-chunk) proteins.

    Returns dict with {short: {n, mean_gap, mean_auc}, long: {n, mean_gap, mean_auc}}.
    """
    short_gaps, long_gaps = [], []
    short_aucs, long_aucs = [], []

    for length, gap, auc in zip(protein_lengths, logit_gaps, aucs):
        if length <= threshold:
            short_gaps.append(gap)
            if auc is not None:
                short_aucs.append(auc)
        else:
            long_gaps.append(gap)
            if auc is not None:
                long_aucs.append(auc)

    def _stats(gaps, aucs_list):
        return {
            "n": len(gaps),
            "mean_gap": sum(gaps) / len(gaps) if gaps else None,
            "mean_auc": sum(aucs_list) / len(aucs_list) if aucs_list else None,
        }

    return {
        "short": _stats(short_gaps, short_aucs),
        "long": _stats(long_gaps, long_aucs),
        "threshold": threshold,
    }
