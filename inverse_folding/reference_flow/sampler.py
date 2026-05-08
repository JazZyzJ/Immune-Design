"""Sampler implementation for Phase C1 position-dependent DFM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .amplification import amplification_factor, shuffle_h_values
from .config import ReferenceFlowConfig
from .schedule import positionwise_unmask_probabilities


@dataclass
class SamplerOutput:
    tokens: torch.Tensor
    unmask_step_by_pos: list[int]
    g_values: list[float]
    trajectory_rows: list[dict[str, Any]]


class PositionDependentDFMSampler:
    """Sampling-only position-dependent discrete flow matcher."""

    def __init__(self, *, mask_token_id: int, vocab_size: int) -> None:
        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(vocab_size)

    def sample(
        self,
        *,
        sequence_length: int,
        h_values: np.ndarray | list[float],
        denoiser: Callable[[torch.Tensor, float, Any], torch.Tensor],
        config: ReferenceFlowConfig,
        struct: Any = None,
        save_trajectories: bool = False,
        shuffle_seed: int | None = None,
    ) -> SamplerOutput:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        h = np.asarray(h_values, dtype=np.float32)
        if h.shape != (sequence_length,):
            raise ValueError(
                f"h_values shape {h.shape} does not match sequence_length={sequence_length}"
            )
        if config.h_shuffle.enabled:
            effective_shuffle_seed = (
                int(config.h_shuffle.seed)
                if shuffle_seed is None
                else int(shuffle_seed)
            )
            h = shuffle_h_values(h, seed=effective_shuffle_seed)

        g_values = amplification_factor(h, config.amplification)
        rng = np.random.default_rng(int(config.sampler.seed))
        x_t = torch.full((sequence_length,), self.mask_token_id, dtype=torch.long)
        unmask_step_by_pos = [-1] * sequence_length
        # log-prob of the currently committed token at each position; -inf for
        # positions still masked (or freshly remasked). Used by the optional
        # reparam refinement to pick the bottom-k committed positions.
        scores = np.full(sequence_length, -np.inf, dtype=np.float64)
        trajectory_rows: list[dict[str, Any]] = []
        n_steps = int(config.sampler.n_steps)
        dt = 1.0 / float(n_steps)
        last_logits: torch.Tensor | None = None
        remask_enabled = bool(config.sampler.remask.enabled)

        for step in range(n_steps):
            t = step / float(n_steps)
            logits = denoiser(x_t.clone(), t, struct)
            if logits.shape != (sequence_length, self.vocab_size):
                raise ValueError(
                    "denoiser must return logits with shape "
                    f"({sequence_length}, {self.vocab_size})"
                )
            if torch.isnan(logits).any():
                raise FloatingPointError(f"NaN logits encountered at step={step} t={t:.6f}")

            last_logits = logits.detach().cpu()
            probs = positionwise_unmask_probabilities(
                t=t,
                dt=dt,
                g_values=g_values,
                base_form=config.schedule.base_form,
            )
            masked_positions = (x_t == self.mask_token_id).cpu().numpy()
            if masked_positions.any():
                draws = rng.random(sequence_length) < probs
                selected_positions = np.flatnonzero(masked_positions & draws)
                if selected_positions.size:
                    selected_logits = last_logits[selected_positions] / float(
                        config.sampler.temperature
                    )
                    sampled_tokens, sampled_logp = _sample_categorical(
                        selected_logits, rng
                    )
                    x_t[selected_positions] = sampled_tokens
                    scores[selected_positions] = sampled_logp
                    for pos in selected_positions.tolist():
                        if unmask_step_by_pos[pos] < 0:
                            unmask_step_by_pos[pos] = step

            remask_count = 0
            if remask_enabled and step < n_steps - 1:
                remask_count = _apply_reparam_remask(
                    x_t=x_t,
                    scores=scores,
                    unmask_step_by_pos=unmask_step_by_pos,
                    mask_token_id=self.mask_token_id,
                    step=step,
                    n_steps=n_steps,
                )

            if save_trajectories:
                trajectory_rows.append(
                    {
                        "step": step,
                        "t": t,
                        "unmasked_mask": (x_t != self.mask_token_id).tolist(),
                        "token_argmax": last_logits.argmax(dim=-1).tolist(),
                        "remasked_count": int(remask_count),
                    }
                )

        residual = torch.nonzero(x_t == self.mask_token_id, as_tuple=False).flatten()
        if residual.numel():
            if last_logits is None:
                raise RuntimeError("sampler ended without any denoiser logits")
            sampled_tokens, sampled_logp = _sample_categorical(
                last_logits[residual] / float(config.sampler.temperature),
                rng,
            )
            x_t[residual] = sampled_tokens
            residual_idx = residual.tolist()
            scores[residual_idx] = sampled_logp
            for pos in residual_idx:
                if unmask_step_by_pos[pos] < 0:
                    unmask_step_by_pos[pos] = n_steps

        if (x_t == self.mask_token_id).any():
            raise RuntimeError("sampler finished with mask tokens still present")

        return SamplerOutput(
            tokens=x_t,
            unmask_step_by_pos=unmask_step_by_pos,
            g_values=g_values.astype(np.float32, copy=False).tolist(),
            trajectory_rows=trajectory_rows,
        )


def _sample_categorical(
    logits: torch.Tensor, rng: np.random.Generator
) -> tuple[torch.Tensor, np.ndarray]:
    """Categorical sample plus per-row log-prob of the chosen token."""
    log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
    probs = np.exp(log_probs)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    sampled = np.array(
        [int(rng.choice(probs.shape[1], p=row)) for row in probs],
        dtype=np.int64,
    )
    chosen_logp = log_probs[np.arange(probs.shape[0]), sampled]
    return torch.from_numpy(sampled).to(torch.long), chosen_logp.astype(np.float64, copy=False)


def _apply_reparam_remask(
    *,
    x_t: torch.Tensor,
    scores: np.ndarray,
    unmask_step_by_pos: list[int],
    mask_token_id: int,
    step: int,
    n_steps: int,
) -> int:
    """Re-mask the lowest-confidence committed positions.

    Mirrors DPLM's ``reparam-uncond-deterministic-linear`` rule: at step ``s``
    of ``T`` (1-indexed for the rate), keep ``s/T`` of the committed positions
    and re-mask the bottom ``1 - s/T`` by score.

    Returns the number of positions re-masked this step.
    """
    committed_mask = (x_t != mask_token_id).cpu().numpy()
    n_committed = int(committed_mask.sum())
    if n_committed == 0:
        return 0
    rate = 1.0 - (step + 1) / float(n_steps)
    cutoff_len = int(n_committed * rate)
    if cutoff_len <= 0:
        return 0
    committed_positions = np.flatnonzero(committed_mask)
    committed_scores = scores[committed_positions]
    # Lowest cutoff_len scores → re-mask. ``argpartition`` for O(n).
    if cutoff_len >= committed_positions.size:
        bottom_positions = committed_positions
    else:
        partition_idx = np.argpartition(committed_scores, cutoff_len)[:cutoff_len]
        bottom_positions = committed_positions[partition_idx]
    if bottom_positions.size == 0:
        return 0
    x_t[bottom_positions] = mask_token_id
    scores[bottom_positions] = -np.inf
    for pos in bottom_positions.tolist():
        unmask_step_by_pos[pos] = -1
    return int(bottom_positions.size)
