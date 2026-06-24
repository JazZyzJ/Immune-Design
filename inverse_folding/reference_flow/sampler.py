"""Sampler implementation for Phase C1 position-dependent DFM."""

from __future__ import annotations

from collections.abc import Mapping
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
    # Positions held fixed by a hard-anchor constraint (uricase enzyme mode v0).
    # Distinct from ordinary residues that merely commit at step 0.
    fixed_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class SamplerBatchLane:
    sequence_length: int
    h_values: np.ndarray | list[float]
    config: ReferenceFlowConfig
    struct: Any = None
    shuffle_seed: int | None = None
    controller: Any | None = None
    protein_id: str = ""
    design_idx: int = 0
    fixed_tokens: Mapping[int, int] | None = None


@dataclass
class _SamplerLaneState:
    lane: SamplerBatchLane
    g_values: np.ndarray
    rng: np.random.Generator
    x_t: torch.Tensor
    unmask_step_by_pos: list[int]
    scores: np.ndarray
    trajectory_rows: list[dict[str, Any]]
    last_logits: torch.Tensor | None
    remask_enabled: bool
    remask_fraction_scale: float
    fixed_positions: frozenset[int]


def _resolve_fixed_positions(
    fixed_tokens: Mapping[int, int] | None, sequence_length: int
) -> frozenset[int]:
    """Validate and collect hard-anchor positions (fail-fast on out-of-range index)."""
    if not fixed_tokens:
        return frozenset()
    positions: set[int] = set()
    for idx in fixed_tokens:
        i = int(idx)
        if not (0 <= i < sequence_length):
            raise ValueError(
                f"fixed_tokens position {i} out of range [0, {sequence_length})"
            )
        positions.add(i)
    return frozenset(positions)


def _union_protected(
    post_protected: tuple[int, ...], fixed_positions: frozenset[int]
) -> tuple[int, ...]:
    """Union controller-protected positions with the permanent hard-anchor set.

    With no fixed positions this returns ``post_protected`` unchanged so the
    no-constraint path stays bit-equivalent to legacy behavior.
    """
    if not fixed_positions:
        return post_protected
    return tuple(sorted(set(post_protected) | fixed_positions))


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
        controller: Any | None = None,
        protein_id: str = "",
        design_idx: int = 0,
        fixed_tokens: Mapping[int, int] | None = None,
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
        # Hard-anchor constraint (uricase enzyme mode v0): seed fixed tokens and
        # mark them committed at step 0. They are never re-sampled (non-mask -> not
        # in masked_positions) and are protected from remask via _union_protected.
        fixed_positions = _resolve_fixed_positions(fixed_tokens, sequence_length)
        for fixed_idx, fixed_token_id in (fixed_tokens or {}).items():
            x_t[int(fixed_idx)] = int(fixed_token_id)
            unmask_step_by_pos[int(fixed_idx)] = 0
        # log-prob of the currently committed token at each position; -inf for
        # positions still masked (or freshly remasked). Used by the optional
        # reparam refinement to pick the bottom-k committed positions.
        scores = np.full(sequence_length, -np.inf, dtype=np.float64)
        trajectory_rows: list[dict[str, Any]] = []
        n_steps = int(config.sampler.n_steps)
        dt = 1.0 / float(n_steps)
        last_logits: torch.Tensor | None = None
        remask_enabled = bool(config.sampler.remask.enabled)
        remask_fraction_scale = float(config.sampler.remask.fraction_scale)

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

            structural_logits = logits
            if controller is not None:
                # Local import to avoid pulling controller deps when unused.
                from .controller import PostSamplingContext, SamplerStepContext

                ctx = SamplerStepContext(
                    x_t=x_t.detach().clone(),
                    logits=logits,
                    scores=scores.copy(),
                    step=step,
                    t=t,
                    mask_token_id=self.mask_token_id,
                    protein_id=protein_id,
                    design_idx=design_idx,
                    sequence_length=sequence_length,
                )
                result = controller.step(ctx)
                logits = result.logits
            corrected_logits = logits

            last_logits = logits.detach().cpu()
            probs = positionwise_unmask_probabilities(
                t=t,
                dt=dt,
                g_values=g_values,
                base_form=config.schedule.base_form,
            )
            masked_positions = (x_t == self.mask_token_id).cpu().numpy()
            selected_positions = np.array([], dtype=np.int64)
            sampled_tokens_actual = np.array([], dtype=np.int64)
            sampled_tokens_uncorrected: np.ndarray | None = None
            if masked_positions.any():
                draws = rng.random(sequence_length) < probs
                selected_positions = np.flatnonzero(masked_positions & draws)
                if selected_positions.size:
                    # PLAN §"Sampler integration" 4: snapshot RNG state AFTER
                    # Bernoulli draws are fixed and BEFORE categorical sampling
                    # so the paired branch can replay token sampling without
                    # changing the set of selected positions.
                    saved_state = (
                        rng.bit_generator.state if controller is not None else None
                    )
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
                    sampled_tokens_actual = sampled_tokens.numpy().astype(
                        np.int64, copy=False
                    )
                    # PLAN §"Sampler integration" 6-7: paired uncorrected sample
                    # via an isolated RNG clone so the real RNG is unaffected.
                    # Skipped when corrected == structural (no information gain)
                    # or when D2 paired_uncorrected_sample is disabled.
                    if (
                        controller is not None
                        and saved_state is not None
                        and getattr(controller, "config", None) is not None
                        and getattr(controller.config, "d2", None) is not None
                        and bool(controller.config.d2.enabled)
                        and bool(controller.config.d2.paired_uncorrected_sample)
                        and corrected_logits is not structural_logits
                    ):
                        paired_rng = np.random.default_rng()
                        paired_rng.bit_generator.state = saved_state
                        structural_selected_logits = (
                            structural_logits.detach().cpu()[selected_positions]
                            / float(config.sampler.temperature)
                        )
                        sampled_uncorrected, _ = _sample_categorical(
                            structural_selected_logits, paired_rng
                        )
                        sampled_tokens_uncorrected = (
                            sampled_uncorrected.numpy().astype(np.int64, copy=False)
                        )

            # Post-sampling hook (PLAN §"Sampler integration" 9-10). Skipped
            # when no controller is bound or when remask is disabled, which
            # keeps controller=None bit-equivalent with pre-D2/D3 behavior.
            remask_count = 0
            post_rank_scores: np.ndarray | None = None
            post_protected: tuple[int, ...] = ()
            if controller is not None and remask_enabled and step < n_steps - 1:
                post_ctx = PostSamplingContext(
                    x_t=x_t.detach().clone(),
                    scores=scores.copy(),
                    structural_logits=structural_logits,
                    corrected_logits=corrected_logits,
                    selected_positions=selected_positions,
                    sampled_tokens_actual=sampled_tokens_actual,
                    sampled_tokens_uncorrected=sampled_tokens_uncorrected,
                    step=step,
                    t=t,
                    n_steps=n_steps,
                    mask_token_id=self.mask_token_id,
                    protein_id=protein_id,
                    design_idx=design_idx,
                    sequence_length=sequence_length,
                )
                post_result = controller.post_step(post_ctx)
                post_rank_scores = post_result.rank_scores
                post_protected = post_result.protected_positions
            if remask_enabled and step < n_steps - 1:
                remask_result = _apply_reparam_remask(
                    x_t=x_t,
                    scores=scores,
                    unmask_step_by_pos=unmask_step_by_pos,
                    mask_token_id=self.mask_token_id,
                    step=step,
                    n_steps=n_steps,
                    rank_scores=post_rank_scores,
                    protected_positions=_union_protected(post_protected, fixed_positions),
                    cutoff_scale=remask_fraction_scale,
                )
                remask_count = remask_result.count
                # D3 remask telemetry hook (PLAN §D3-14). Skipped when the
                # controller does not expose post_remask (keeps duck-typed
                # stub controllers in unit tests bit-equivalent).
                if controller is not None and remask_result.remasked_positions:
                    post_remask_fn = getattr(controller, "post_remask", None)
                    if callable(post_remask_fn):
                        post_remask_fn(
                            remasked_positions=remask_result.remasked_positions,
                            step=step,
                            t=t,
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
            fixed_positions=tuple(sorted(fixed_positions)),
        )

    def sample_batch(
        self,
        *,
        lanes: list[SamplerBatchLane],
        batched_denoiser: Callable[
            [list[torch.Tensor], float, list[Any]], list[torch.Tensor]
        ],
        save_trajectories: bool = False,
    ) -> list[SamplerOutput]:
        """Sample independent RF lanes with one batched denoiser call per step.

        Controller, RNG, sampling, remask, and telemetry state remain per-lane.
        The only shared operation is the structural denoiser forward.
        """

        if not lanes:
            return []
        n_steps = int(lanes[0].config.sampler.n_steps)
        if n_steps <= 0:
            raise ValueError("sampler.n_steps must be positive")
        for idx, lane in enumerate(lanes):
            if int(lane.config.sampler.n_steps) != n_steps:
                raise ValueError(
                    "sample_batch requires all lanes to use the same n_steps; "
                    f"lane 0 has {n_steps}, lane {idx} has "
                    f"{int(lane.config.sampler.n_steps)}"
                )

        states = [self._init_batch_lane_state(lane) for lane in lanes]
        dt = 1.0 / float(n_steps)

        for step in range(n_steps):
            t = step / float(n_steps)
            structural_logits_by_lane = batched_denoiser(
                [state.x_t.clone() for state in states],
                t,
                [state.lane.struct for state in states],
            )
            if len(structural_logits_by_lane) != len(states):
                raise ValueError(
                    f"batched_denoiser returned {len(structural_logits_by_lane)} "
                    f"logit tensors for {len(states)} lanes"
                )
            for state, structural_logits in zip(states, structural_logits_by_lane):
                self._sample_batch_lane_step(
                    state=state,
                    structural_logits=structural_logits,
                    step=step,
                    t=t,
                    dt=dt,
                    n_steps=n_steps,
                    save_trajectories=save_trajectories,
                )

        return [self._finalize_batch_lane_state(state, n_steps=n_steps) for state in states]

    def _init_batch_lane_state(self, lane: SamplerBatchLane) -> _SamplerLaneState:
        sequence_length = int(lane.sequence_length)
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        h = np.asarray(lane.h_values, dtype=np.float32)
        if h.shape != (sequence_length,):
            raise ValueError(
                f"h_values shape {h.shape} does not match sequence_length={sequence_length}"
            )
        if lane.config.h_shuffle.enabled:
            effective_shuffle_seed = (
                int(lane.config.h_shuffle.seed)
                if lane.shuffle_seed is None
                else int(lane.shuffle_seed)
            )
            h = shuffle_h_values(h, seed=effective_shuffle_seed)
        g_values = amplification_factor(h, lane.config.amplification)
        x_t = torch.full((sequence_length,), self.mask_token_id, dtype=torch.long)
        unmask_step_by_pos = [-1] * sequence_length
        fixed_positions = _resolve_fixed_positions(lane.fixed_tokens, sequence_length)
        for fixed_idx, fixed_token_id in (lane.fixed_tokens or {}).items():
            x_t[int(fixed_idx)] = int(fixed_token_id)
            unmask_step_by_pos[int(fixed_idx)] = 0
        return _SamplerLaneState(
            lane=lane,
            g_values=g_values,
            rng=np.random.default_rng(int(lane.config.sampler.seed)),
            x_t=x_t,
            unmask_step_by_pos=unmask_step_by_pos,
            scores=np.full(sequence_length, -np.inf, dtype=np.float64),
            trajectory_rows=[],
            last_logits=None,
            remask_enabled=bool(lane.config.sampler.remask.enabled),
            remask_fraction_scale=float(lane.config.sampler.remask.fraction_scale),
            fixed_positions=fixed_positions,
        )

    def _sample_batch_lane_step(
        self,
        *,
        state: _SamplerLaneState,
        structural_logits: torch.Tensor,
        step: int,
        t: float,
        dt: float,
        n_steps: int,
        save_trajectories: bool,
    ) -> None:
        sequence_length = int(state.lane.sequence_length)
        if structural_logits.shape != (sequence_length, self.vocab_size):
            raise ValueError(
                "batched_denoiser must return per-lane logits with shape "
                f"({sequence_length}, {self.vocab_size}); got "
                f"{tuple(structural_logits.shape)}"
            )
        if torch.isnan(structural_logits).any():
            raise FloatingPointError(f"NaN logits encountered at step={step} t={t:.6f}")

        logits = structural_logits
        controller = state.lane.controller
        if controller is not None:
            from .controller import SamplerStepContext

            ctx = SamplerStepContext(
                x_t=state.x_t.detach().clone(),
                logits=logits,
                scores=state.scores.copy(),
                step=step,
                t=t,
                mask_token_id=self.mask_token_id,
                protein_id=state.lane.protein_id,
                design_idx=state.lane.design_idx,
                sequence_length=sequence_length,
            )
            result = controller.step(ctx)
            logits = result.logits
        corrected_logits = logits

        state.last_logits = logits.detach().cpu()
        probs = positionwise_unmask_probabilities(
            t=t,
            dt=dt,
            g_values=state.g_values,
            base_form=state.lane.config.schedule.base_form,
        )
        masked_positions = (state.x_t == self.mask_token_id).cpu().numpy()
        selected_positions = np.array([], dtype=np.int64)
        sampled_tokens_actual = np.array([], dtype=np.int64)
        sampled_tokens_uncorrected: np.ndarray | None = None
        if masked_positions.any():
            draws = state.rng.random(sequence_length) < probs
            selected_positions = np.flatnonzero(masked_positions & draws)
            if selected_positions.size:
                saved_state = (
                    state.rng.bit_generator.state if controller is not None else None
                )
                selected_logits = state.last_logits[selected_positions] / float(
                    state.lane.config.sampler.temperature
                )
                sampled_tokens, sampled_logp = _sample_categorical(
                    selected_logits, state.rng
                )
                state.x_t[selected_positions] = sampled_tokens
                state.scores[selected_positions] = sampled_logp
                for pos in selected_positions.tolist():
                    if state.unmask_step_by_pos[pos] < 0:
                        state.unmask_step_by_pos[pos] = step
                sampled_tokens_actual = sampled_tokens.numpy().astype(np.int64, copy=False)
                if (
                    controller is not None
                    and saved_state is not None
                    and getattr(controller, "config", None) is not None
                    and getattr(controller.config, "d2", None) is not None
                    and bool(controller.config.d2.enabled)
                    and bool(controller.config.d2.paired_uncorrected_sample)
                    and corrected_logits is not structural_logits
                ):
                    paired_rng = np.random.default_rng()
                    paired_rng.bit_generator.state = saved_state
                    structural_selected_logits = (
                        structural_logits.detach().cpu()[selected_positions]
                        / float(state.lane.config.sampler.temperature)
                    )
                    sampled_uncorrected, _ = _sample_categorical(
                        structural_selected_logits, paired_rng
                    )
                    sampled_tokens_uncorrected = sampled_uncorrected.numpy().astype(
                        np.int64, copy=False
                    )

        remask_count = 0
        post_rank_scores: np.ndarray | None = None
        post_protected: tuple[int, ...] = ()
        if controller is not None and state.remask_enabled and step < n_steps - 1:
            from .controller import PostSamplingContext

            post_ctx = PostSamplingContext(
                x_t=state.x_t.detach().clone(),
                scores=state.scores.copy(),
                structural_logits=structural_logits,
                corrected_logits=corrected_logits,
                selected_positions=selected_positions,
                sampled_tokens_actual=sampled_tokens_actual,
                sampled_tokens_uncorrected=sampled_tokens_uncorrected,
                step=step,
                t=t,
                n_steps=n_steps,
                mask_token_id=self.mask_token_id,
                protein_id=state.lane.protein_id,
                design_idx=state.lane.design_idx,
                sequence_length=sequence_length,
            )
            post_result = controller.post_step(post_ctx)
            post_rank_scores = post_result.rank_scores
            post_protected = post_result.protected_positions
        if state.remask_enabled and step < n_steps - 1:
            remask_result = _apply_reparam_remask(
                x_t=state.x_t,
                scores=state.scores,
                unmask_step_by_pos=state.unmask_step_by_pos,
                mask_token_id=self.mask_token_id,
                step=step,
                n_steps=n_steps,
                rank_scores=post_rank_scores,
                protected_positions=_union_protected(
                    post_protected, state.fixed_positions
                ),
                cutoff_scale=state.remask_fraction_scale,
            )
            remask_count = remask_result.count
            if controller is not None and remask_result.remasked_positions:
                post_remask_fn = getattr(controller, "post_remask", None)
                if callable(post_remask_fn):
                    post_remask_fn(
                        remasked_positions=remask_result.remasked_positions,
                        step=step,
                        t=t,
                    )

        if save_trajectories:
            state.trajectory_rows.append(
                {
                    "step": step,
                    "t": t,
                    "unmasked_mask": (state.x_t != self.mask_token_id).tolist(),
                    "token_argmax": state.last_logits.argmax(dim=-1).tolist(),
                    "remasked_count": int(remask_count),
                }
            )

    def _finalize_batch_lane_state(
        self,
        state: _SamplerLaneState,
        *,
        n_steps: int,
    ) -> SamplerOutput:
        residual = torch.nonzero(
            state.x_t == self.mask_token_id, as_tuple=False
        ).flatten()
        if residual.numel():
            if state.last_logits is None:
                raise RuntimeError("sampler ended without any denoiser logits")
            sampled_tokens, sampled_logp = _sample_categorical(
                state.last_logits[residual]
                / float(state.lane.config.sampler.temperature),
                state.rng,
            )
            state.x_t[residual] = sampled_tokens
            residual_idx = residual.tolist()
            state.scores[residual_idx] = sampled_logp
            for pos in residual_idx:
                if state.unmask_step_by_pos[pos] < 0:
                    state.unmask_step_by_pos[pos] = n_steps

        if (state.x_t == self.mask_token_id).any():
            raise RuntimeError("sampler finished with mask tokens still present")

        return SamplerOutput(
            tokens=state.x_t,
            unmask_step_by_pos=state.unmask_step_by_pos,
            g_values=state.g_values.astype(np.float32, copy=False).tolist(),
            trajectory_rows=state.trajectory_rows,
            fixed_positions=tuple(sorted(state.fixed_positions)),
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


@dataclass
class ReparamRemaskResult:
    """Number of positions remasked plus the actual position list for telemetry."""

    count: int
    remasked_positions: tuple[int, ...]


def _apply_reparam_remask(
    *,
    x_t: torch.Tensor,
    scores: np.ndarray,
    unmask_step_by_pos: list[int],
    mask_token_id: int,
    step: int,
    n_steps: int,
    rank_scores: np.ndarray | None = None,
    protected_positions: tuple[int, ...] = (),
    cutoff_scale: float = 1.0,
) -> ReparamRemaskResult:
    """Re-mask the lowest-confidence committed positions.

    Mirrors DPLM's ``reparam-uncond-deterministic-linear`` rule: at step ``s``
    of ``T`` (1-indexed for the rate), keep ``s/T`` of the committed positions
    and re-mask the bottom ``1 - s/T`` by score.

    ``rank_scores`` overrides ``scores[]`` as the per-residue ranking signal
    (D3 commit-score path). When ``None`` the function uses ``scores[]``
    exactly as before. ``protected_positions`` removes positions from the
    remask candidate pool (D3 grace / final freeze). ``cutoff_scale``
    (``sampler.remask.fraction_scale``) multiplies the cutoff length: ``1.0``
    is a no-op, ``0.0`` re-masks nothing. With all defaults the output is
    byte-equivalent to the pre-D2/D3 implementation.

    Returns the number of positions re-masked this step.
    """
    committed_mask = (x_t != mask_token_id).cpu().numpy()
    n_committed = int(committed_mask.sum())
    if n_committed == 0:
        return ReparamRemaskResult(count=0, remasked_positions=())
    rate = 1.0 - (step + 1) / float(n_steps)
    cutoff_len = int(n_committed * rate * cutoff_scale)
    if cutoff_len <= 0:
        return ReparamRemaskResult(count=0, remasked_positions=())
    committed_positions = np.flatnonzero(committed_mask)
    if protected_positions:
        protected_set = {int(p) for p in protected_positions}
        committed_positions = np.array(
            [p for p in committed_positions if int(p) not in protected_set],
            dtype=committed_positions.dtype,
        )
    if committed_positions.size == 0:
        return ReparamRemaskResult(count=0, remasked_positions=())
    ranking = scores if rank_scores is None else rank_scores
    ranking_at_committed = ranking[committed_positions]
    # Lowest cutoff_len scores → re-mask. ``argpartition`` for O(n).
    if cutoff_len >= committed_positions.size:
        bottom_positions = committed_positions
    else:
        partition_idx = np.argpartition(ranking_at_committed, cutoff_len)[:cutoff_len]
        bottom_positions = committed_positions[partition_idx]
    if bottom_positions.size == 0:
        return ReparamRemaskResult(count=0, remasked_positions=())
    x_t[bottom_positions] = mask_token_id
    scores[bottom_positions] = -np.inf
    for pos in bottom_positions.tolist():
        unmask_step_by_pos[pos] = -1
    return ReparamRemaskResult(
        count=int(bottom_positions.size),
        remasked_positions=tuple(int(p) for p in bottom_positions.tolist()),
    )
