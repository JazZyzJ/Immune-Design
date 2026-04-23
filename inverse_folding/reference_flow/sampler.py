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
        trajectory_rows: list[dict[str, Any]] = []
        dt = 1.0 / float(config.sampler.n_steps)
        last_logits: torch.Tensor | None = None

        for step in range(config.sampler.n_steps):
            t = step / float(config.sampler.n_steps)
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
                    sampled_tokens = _sample_categorical(selected_logits, rng)
                    x_t[selected_positions] = sampled_tokens
                    for pos in selected_positions.tolist():
                        if unmask_step_by_pos[pos] < 0:
                            unmask_step_by_pos[pos] = step

            if save_trajectories:
                trajectory_rows.append(
                    {
                        "step": step,
                        "t": t,
                        "unmasked_mask": (x_t != self.mask_token_id).tolist(),
                        "token_argmax": last_logits.argmax(dim=-1).tolist(),
                    }
                )

        residual = torch.nonzero(x_t == self.mask_token_id, as_tuple=False).flatten()
        if residual.numel():
            if last_logits is None:
                raise RuntimeError("sampler ended without any denoiser logits")
            sampled_tokens = _sample_categorical(
                last_logits[residual] / float(config.sampler.temperature),
                rng,
            )
            x_t[residual] = sampled_tokens
            for pos in residual.tolist():
                if unmask_step_by_pos[pos] < 0:
                    unmask_step_by_pos[pos] = config.sampler.n_steps

        if (x_t == self.mask_token_id).any():
            raise RuntimeError("sampler finished with mask tokens still present")

        return SamplerOutput(
            tokens=x_t,
            unmask_step_by_pos=unmask_step_by_pos,
            g_values=g_values.astype(np.float32, copy=False).tolist(),
            trajectory_rows=trajectory_rows,
        )


def _sample_categorical(logits: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    sampled = [int(rng.choice(probs.shape[1], p=row)) for row in probs]
    return torch.tensor(sampled, dtype=torch.long)
