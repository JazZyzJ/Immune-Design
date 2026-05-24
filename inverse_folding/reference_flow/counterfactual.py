"""Phase D2 hard-counterfactual primitives (PLAN_RF.md §"D2 step behavior").

Pure functions only. The D2Handler in ``controller.py`` orchestrates these
primitives, owns the head-scorer batch call, and writes telemetry. Keeping
this module head-scorer-free makes every component independently testable.

Conventions
-----------
* Positions are 0-based residue indices into ``x_t`` / ``struct_logits``.
* ``struct_logits`` has shape ``(L, V)`` and is per-protein at one sampler step.
* ``canonical_token_ids`` enumerates the AA tokens that may appear in a
  designed sequence; special tokens (mask/pad/cls/eos) must be excluded by
  the caller so the candidate support is always sampleable.
* Local risk aggregation is LME (log-mean-exp) for the first D2/D3
  implementation (PLAN_RF.md §D2-D3 validation 7).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .controller_config import D2Config


# ---------------------------------------------------------------------------
# Position selection (PLAN §D2 step behavior 5-6)
# ---------------------------------------------------------------------------


def select_editable_positions(
    *,
    start_0b: int,
    end_0b: int,
    x_t: torch.Tensor,
    mask_token_id: int,
    residue_excess: np.ndarray,
    per_pos_entropy: torch.Tensor,
    max_positions: int,
) -> tuple[int, ...]:
    """Pick at most ``max_positions`` masked residues inside ``[start, end)``.

    Ranking: residue excess descending, then per-position structural entropy
    ascending (lower entropy first as tiebreak).
    """
    masked: list[int] = []
    for i in range(int(start_0b), int(end_0b)):
        if int(x_t[i].item()) == int(mask_token_id):
            masked.append(i)
    if not masked:
        return ()
    ranked = sorted(
        masked,
        key=lambda i: (-float(residue_excess[i]), float(per_pos_entropy[i].item())),
    )
    return tuple(ranked[: int(max_positions)])


# ---------------------------------------------------------------------------
# Candidate support (PLAN §D2 step behavior 7)
# ---------------------------------------------------------------------------


def build_candidate_support(
    *,
    struct_logits: torch.Tensor,
    positions: Sequence[int],
    top_k: int,
    canonical_token_ids: Sequence[int],
) -> dict[int, tuple[int, ...]]:
    """Return per-position top-``top_k`` canonical-AA tokens by structural logit.

    Special tokens never enter the support because we restrict to
    ``canonical_token_ids`` before the top-k.
    """
    canonical_arr = np.asarray(list(canonical_token_ids), dtype=np.int64)
    if canonical_arr.size == 0:
        raise ValueError("canonical_token_ids must be non-empty")
    k = min(int(top_k), int(canonical_arr.shape[0]))
    out: dict[int, tuple[int, ...]] = {}
    for i in positions:
        logits_i = struct_logits[int(i), :].detach().cpu()
        logits_canonical = logits_i[torch.from_numpy(canonical_arr)]
        topk_idx = torch.topk(logits_canonical, k).indices.numpy()
        out[int(i)] = tuple(int(canonical_arr[j]) for j in topk_idx)
    return out


# ---------------------------------------------------------------------------
# Candidate enumeration (PLAN §D2 step behavior 8)
# ---------------------------------------------------------------------------


def enumerate_candidates(
    *,
    positions: Sequence[int],
    K_i_per_pos: dict[int, tuple[int, ...]],
    max_candidates: int,
    struct_logits: torch.Tensor,
    seed_tuple: tuple,
) -> tuple[tuple[tuple[int, ...], ...], str]:
    """Build the per-block candidate set.

    * Cartesian product when ``prod(|K_i|) <= max_candidates``.
    * Else i.i.d. with-replacement sampling from per-position structural
      softmax restricted to ``K_i``. Duplicates are retained because they are
      part of the Monte Carlo estimate that feeds ESS.

    Sampling RNG is derived deterministically from ``seed_tuple`` so a refresh
    can be reproduced from telemetry alone.
    """
    pos_list = list(positions)
    K_lists = [K_i_per_pos[int(p)] for p in pos_list]
    n_cart = 1
    for K in K_lists:
        n_cart *= max(1, len(K))
    if n_cart <= int(max_candidates):
        return tuple(tuple(combo) for combo in product(*K_lists)), "cartesian"

    rng = np.random.default_rng(_seed_from_tuple(seed_tuple))
    per_pos_choice_arrays: list[np.ndarray] = []
    per_pos_probs: list[np.ndarray] = []
    for p, K in zip(pos_list, K_lists):
        K_arr = np.asarray(K, dtype=np.int64)
        per_pos_choice_arrays.append(K_arr)
        logits_K = struct_logits[int(p), torch.from_numpy(K_arr)].detach().cpu().numpy().astype(np.float64)
        logits_K = logits_K - float(logits_K.max())
        probs = np.exp(logits_K)
        probs = probs / probs.sum()
        per_pos_probs.append(probs)
    samples: list[tuple[int, ...]] = []
    for _ in range(int(max_candidates)):
        cand: list[int] = []
        for K_arr, probs in zip(per_pos_choice_arrays, per_pos_probs):
            choice = int(rng.choice(K_arr.shape[0], p=probs))
            cand.append(int(K_arr[choice]))
        samples.append(tuple(cand))
    return tuple(samples), "sampled"


def _seed_from_tuple(tup: tuple) -> int:
    """Deterministic 64-bit unsigned seed from a generic tuple key."""
    h = hashlib.sha256(repr(tup).encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


# ---------------------------------------------------------------------------
# Local risk + per-candidate joint structural mass (PLAN §D2 step 10-12)
# ---------------------------------------------------------------------------


def compute_local_risk(
    *,
    window_risks: Sequence[float],
    omega_indices: Sequence[int],
    aggregation: str = "LME",
) -> float:
    """Restrict to ``Omega(B)`` window indices and aggregate via LME."""
    indices = [int(i) for i in omega_indices]
    if not indices:
        return float("-inf")
    vals = np.asarray([float(window_risks[i]) for i in indices], dtype=np.float64)
    if aggregation == "LME":
        m = float(vals.max())
        if not math.isfinite(m):
            return m
        return m + math.log(float(np.exp(vals - m).mean()))
    raise ValueError(f"unsupported aggregation: {aggregation!r}")


def compute_Q_B_per_candidate(
    *,
    candidates: Sequence[tuple[int, ...]],
    positions: Sequence[int],
    K_i_per_pos: dict[int, tuple[int, ...]],
    struct_logits: torch.Tensor,
) -> np.ndarray:
    """Joint structural probability of each candidate, factorized over positions."""
    pos_list = list(positions)
    log_prob_lookup: list[dict[int, float]] = []
    for p in pos_list:
        K = K_i_per_pos[int(p)]
        K_arr = np.asarray(K, dtype=np.int64)
        logits_K = (
            struct_logits[int(p), torch.from_numpy(K_arr)]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        logits_K = logits_K - float(logits_K.max())
        probs = np.exp(logits_K)
        probs = probs / probs.sum()
        log_prob_lookup.append(
            {int(K_arr[j]): float(np.log(probs[j])) for j in range(K_arr.shape[0])}
        )
    Q = np.zeros(len(candidates), dtype=np.float64)
    for c_idx, cand in enumerate(candidates):
        lp = 0.0
        for j, tok in enumerate(cand):
            lp += log_prob_lookup[j][int(tok)]
        Q[c_idx] = float(np.exp(lp))
    return Q


# ---------------------------------------------------------------------------
# Weights + ESS + marginals (PLAN §D2 step 13-16)
# ---------------------------------------------------------------------------


def compute_weights(
    *,
    mode: str,
    Q_B: np.ndarray,
    delta_R_B: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Branch-specific unnormalized immune importance weight.

    * Cartesian enumeration: ``w(a) = Q_B(a) * exp(-beta * ΔR_B(a))``.
    * Sampled from ``Q_B``: ``w(a) = exp(-beta * ΔR_B(a))`` (``Q_B`` is the proposal).
    """
    expo = np.exp(-float(beta) * np.asarray(delta_R_B, dtype=np.float64))
    if mode == "cartesian":
        return np.asarray(Q_B, dtype=np.float64) * expo
    if mode == "sampled":
        return expo
    raise ValueError(f"unknown candidate mode: {mode!r}")


def compute_ess(weights: np.ndarray) -> float:
    """Effective sample size from unnormalized weights (zero-safe)."""
    arr = np.asarray(weights, dtype=np.float64)
    s = float(arr.sum())
    s2 = float((arr * arr).sum())
    if s2 <= 0.0:
        return 0.0
    return (s * s) / s2


def project_marginals(
    *,
    candidates: Sequence[tuple[int, ...]],
    weights_normalized: np.ndarray,
    positions: Sequence[int],
    K_i_per_pos: dict[int, tuple[int, ...]],
) -> dict[int, dict[int, float]]:
    """Marginal probability per (position, token in K_i) under given weights."""
    pos_list = list(positions)
    marginals: dict[int, dict[int, float]] = {
        int(p): {int(t): 0.0 for t in K_i_per_pos[int(p)]} for p in pos_list
    }
    for c_idx, cand in enumerate(candidates):
        w = float(weights_normalized[c_idx])
        for j, p in enumerate(pos_list):
            tok = int(cand[j])
            if tok in marginals[int(p)]:
                marginals[int(p)][tok] += w
    return marginals


# ---------------------------------------------------------------------------
# Logit correction (PLAN §D2 step 17-20)
# ---------------------------------------------------------------------------


def compute_delta_logit(
    *,
    pi_marginals: dict[int, dict[int, float]],
    Q_marginals: dict[int, dict[int, float]],
    eta: float,
    rho_B: float,
    epsilon: float,
    max_abs_shift: float,
) -> dict[tuple[int, int], float]:
    """Per ``(position, token)`` log-ratio correction restricted to candidate support."""
    factor = float(eta) * float(rho_B)
    out: dict[tuple[int, int], float] = {}
    for p, pi_p in pi_marginals.items():
        Q_p = Q_marginals[int(p)]
        for tok in pi_p:
            pi_val = float(pi_p[int(tok)])
            Q_val = float(Q_p[int(tok)])
            shift = factor * (math.log(pi_val + float(epsilon)) - math.log(Q_val + float(epsilon)))
            if shift > float(max_abs_shift):
                shift = float(max_abs_shift)
            elif shift < -float(max_abs_shift):
                shift = -float(max_abs_shift)
            out[(int(p), int(tok))] = float(shift)
    return out


def apply_logit_correction(
    *,
    struct_logits: torch.Tensor,
    delta_logit: dict[tuple[int, int], float],
) -> torch.Tensor:
    """Return a ``.clone()``'d logits tensor with ``delta_logit`` applied.

    The original ``struct_logits`` MUST remain untouched because D3 commit
    scoring and paired-attribution sampling both rely on the uncorrected
    structural distribution.
    """
    corrected = struct_logits.clone()
    for (i, tok), shift in delta_logit.items():
        corrected[int(i), int(tok)] = corrected[int(i), int(tok)] + float(shift)
    return corrected


# ---------------------------------------------------------------------------
# Feasibility (PLAN §D2 step 15)
# ---------------------------------------------------------------------------


def compute_feasibility(
    *,
    delta_R_B: np.ndarray,
    min_delta_R_improvement: float,
) -> tuple[bool, float]:
    """A block is feasible when at least one candidate has ``ΔR_B < -threshold``."""
    arr = np.asarray(delta_R_B, dtype=np.float64)
    if arr.size == 0:
        return False, float("inf")
    best = float(arr.min())
    return (best < -float(min_delta_R_improvement)), best


# ---------------------------------------------------------------------------
# D2Handler — stateless orchestrator composed into the controller (PLAN §D2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D2BlockOutcome:
    """Per-block D2 diagnostics and (when applied) logit correction.

    ``skipped_reason`` is ``None`` when the block contributes a correction.
    ``"no_editable"`` → block had no masked positions to write into.
    ``"low_ess"`` → ESS fraction below ``min_ess_fraction``; ``g_ESS=0``.
    ``"not_feasible"`` → no candidate met the ``min_delta_R_improvement`` gate.

    ``r_current`` is the LME risk over ``omega_indices`` of the hard-completion
    sequence at refresh time. Used by the post-sampling realized ΔR pass to
    compute ``delta_R_corrected`` / ``delta_R_uncorrected`` (PLAN_RF.md D0
    schema) without re-scoring the baseline.
    """

    block_id: int
    omega_indices: tuple[int, ...]
    editable_positions: tuple[int, ...]
    K_i_per_pos: dict[int, tuple[int, ...]]
    candidate_mode: str
    candidate_count: int
    delta_R_B: tuple[float, ...]
    Q_B: tuple[float, ...]
    weights_unnormalized: tuple[float, ...]
    ess: float
    ess_fraction: float
    g_ESS_candidates: float
    feasible: bool
    best_delta_R_B: float
    mean_delta_R_B: float
    rho_B_effective: float
    corrected_positions: tuple[int, ...]
    delta_logit: dict[tuple[int, int], float]
    skipped_reason: str | None
    r_current: float = float("nan")


@dataclass(frozen=True)
class D2RefreshOutcome:
    """Aggregated D2 output for one refresh step."""

    corrected_logits: torch.Tensor
    block_outcomes: tuple[D2BlockOutcome, ...]
    corrected_positions: frozenset


class D2Handler:
    """Composes D2 pure primitives into a single per-refresh correction step.

    The handler owns ``D2Config`` only; everything else (scorer, decode
    callable, structural logits, current completion) is supplied per call so
    the same handler instance can be shared across designs.
    """

    def __init__(self, config: D2Config) -> None:
        self.config = config

    def correct_logits(
        self,
        *,
        structural_logits: torch.Tensor,
        active_blocks: Sequence,
        x_t: torch.Tensor,
        mask_token_id: int,
        completed_tokens: torch.Tensor,
        decode_tokens: Callable[[torch.Tensor], str],
        per_pos_entropy: torch.Tensor,
        residue_excess: np.ndarray,
        current_window_risks: Sequence[float],
        scorer: Any,
        canonical_token_ids: Sequence[int],
        protein_id: str,
        seed: int,
        design_idx: int,
        refresh_step: int,
    ) -> D2RefreshOutcome:
        block_outcomes: list[D2BlockOutcome] = []
        delta_logit_acc: dict[tuple[int, int], float] = {}
        corrected_positions: set[int] = set()

        for blk in active_blocks:
            outcome = self._process_block(
                block=blk,
                structural_logits=structural_logits,
                x_t=x_t,
                mask_token_id=mask_token_id,
                completed_tokens=completed_tokens,
                decode_tokens=decode_tokens,
                per_pos_entropy=per_pos_entropy,
                residue_excess=residue_excess,
                current_window_risks=current_window_risks,
                scorer=scorer,
                canonical_token_ids=canonical_token_ids,
                protein_id=protein_id,
                seed=seed,
                design_idx=design_idx,
                refresh_step=refresh_step,
            )
            block_outcomes.append(outcome)
            if outcome.skipped_reason is None:
                for key, shift in outcome.delta_logit.items():
                    delta_logit_acc[key] = delta_logit_acc.get(key, 0.0) + float(shift)
                for pos in outcome.corrected_positions:
                    corrected_positions.add(int(pos))

        if delta_logit_acc:
            corrected = apply_logit_correction(
                struct_logits=structural_logits, delta_logit=delta_logit_acc
            )
        else:
            corrected = structural_logits.clone()
        return D2RefreshOutcome(
            corrected_logits=corrected,
            block_outcomes=tuple(block_outcomes),
            corrected_positions=frozenset(corrected_positions),
        )

    def _process_block(
        self,
        *,
        block,
        structural_logits: torch.Tensor,
        x_t: torch.Tensor,
        mask_token_id: int,
        completed_tokens: torch.Tensor,
        decode_tokens: Callable[[torch.Tensor], str],
        per_pos_entropy: torch.Tensor,
        residue_excess: np.ndarray,
        current_window_risks: Sequence[float],
        scorer: Any,
        canonical_token_ids: Sequence[int],
        protein_id: str,
        seed: int,
        design_idx: int,
        refresh_step: int,
    ) -> D2BlockOutcome:
        omega = tuple(int(i) for i in block.window_indices)
        editable = select_editable_positions(
            start_0b=int(block.residue_start_0b),
            end_0b=int(block.residue_end_0b),
            x_t=x_t,
            mask_token_id=int(mask_token_id),
            residue_excess=residue_excess,
            per_pos_entropy=per_pos_entropy,
            max_positions=int(self.config.max_positions_per_block),
        )
        if not editable:
            # No editable masked positions inside this block → no candidate
            # enumeration / head re-score happens, so ``r_current`` is not
            # available here. We pass NaN explicitly to document that no
            # realized ΔR can be computed for no_editable blocks (and the
            # realized-ΔR pass at post_step also skips blocks with
            # ``skipped_reason != None``).
            current_window_risks_tuple = tuple(float(z) for z in current_window_risks)
            r_current_unused = compute_local_risk(
                window_risks=current_window_risks_tuple,
                omega_indices=omega,
                aggregation="LME",
            )
            return _empty_block_outcome(
                block_id=int(block.block_id),
                omega=omega,
                reason="no_editable",
                r_current=float(r_current_unused),
            )

        K = build_candidate_support(
            struct_logits=structural_logits,
            positions=editable,
            top_k=int(self.config.top_k_tokens),
            canonical_token_ids=canonical_token_ids,
        )
        candidates, mode = enumerate_candidates(
            positions=editable,
            K_i_per_pos=K,
            max_candidates=int(self.config.max_candidates_per_block),
            struct_logits=structural_logits,
            seed_tuple=(
                int(seed),
                str(protein_id),
                int(design_idx),
                int(refresh_step),
                int(block.block_id),
            ),
        )

        # Build candidate sequences and score in one batch.
        records: list[tuple[str, str]] = []
        for c_idx, cand in enumerate(candidates):
            tokens = completed_tokens.detach().clone()
            for pos, tok in zip(editable, cand):
                tokens[int(pos)] = int(tok)
            records.append(
                (f"d2_b{int(block.block_id)}_c{c_idx}", decode_tokens(tokens))
            )
        batch = scorer.score_batch_same_protein(
            protein_id=str(protein_id), records=records
        )

        r_current = compute_local_risk(
            window_risks=current_window_risks,
            omega_indices=omega,
            aggregation="LME",
        )
        delta_R_B = np.array(
            [
                compute_local_risk(
                    window_risks=tuple(float(w.z) for w in batch.scores[c_idx].windows),
                    omega_indices=omega,
                    aggregation="LME",
                )
                - r_current
                for c_idx in range(len(candidates))
            ],
            dtype=np.float64,
        )

        Q_B = compute_Q_B_per_candidate(
            candidates=candidates,
            positions=editable,
            K_i_per_pos=K,
            struct_logits=structural_logits,
        )
        weights = compute_weights(
            mode=mode, Q_B=Q_B, delta_R_B=delta_R_B, beta=float(self.config.beta)
        )
        ess = compute_ess(weights)
        ess_fraction = ess / float(len(candidates)) if candidates else 0.0
        feasible, best_dR = compute_feasibility(
            delta_R_B=delta_R_B,
            min_delta_R_improvement=float(self.config.min_delta_R_improvement),
        )
        mean_dR = float(delta_R_B.mean()) if delta_R_B.size else float("nan")

        # Low-ESS gate ⇒ g_ESS_candidates = 0 ⇒ rho_B_effective = 0 ⇒ no correction.
        if ess_fraction < float(self.config.min_ess_fraction):
            return D2BlockOutcome(
                block_id=int(block.block_id),
                omega_indices=omega,
                editable_positions=tuple(editable),
                K_i_per_pos=K,
                candidate_mode=mode,
                candidate_count=len(candidates),
                delta_R_B=tuple(float(x) for x in delta_R_B.tolist()),
                Q_B=tuple(float(x) for x in Q_B.tolist()),
                weights_unnormalized=tuple(float(x) for x in weights.tolist()),
                ess=float(ess),
                ess_fraction=float(ess_fraction),
                g_ESS_candidates=0.0,
                feasible=feasible,
                best_delta_R_B=float(best_dR),
                mean_delta_R_B=float(mean_dR),
                rho_B_effective=0.0,
                corrected_positions=(),
                delta_logit={},
                skipped_reason="low_ess",
                r_current=float(r_current),
            )
        g_ESS_candidates = 1.0

        if not feasible:
            return D2BlockOutcome(
                block_id=int(block.block_id),
                omega_indices=omega,
                editable_positions=tuple(editable),
                K_i_per_pos=K,
                candidate_mode=mode,
                candidate_count=len(candidates),
                delta_R_B=tuple(float(x) for x in delta_R_B.tolist()),
                Q_B=tuple(float(x) for x in Q_B.tolist()),
                weights_unnormalized=tuple(float(x) for x in weights.tolist()),
                ess=float(ess),
                ess_fraction=float(ess_fraction),
                g_ESS_candidates=float(g_ESS_candidates),
                feasible=False,
                best_delta_R_B=float(best_dR),
                mean_delta_R_B=float(mean_dR),
                rho_B_effective=0.0,
                corrected_positions=(),
                delta_logit={},
                skipped_reason="not_feasible",
                r_current=float(r_current),
            )

        w_sum = float(weights.sum())
        w_norm = weights / w_sum if w_sum > 0.0 else weights
        pi = project_marginals(
            candidates=candidates,
            weights_normalized=w_norm,
            positions=editable,
            K_i_per_pos=K,
        )
        # Q marginal source depends on enumeration mode:
        # * Cartesian: each candidate appears exactly once with weight Q_B,
        #   so the projection is exactly the analytic per-position softmax
        #   over K_i. Use the analytic form — it is exact and avoids small
        #   numerical drift from re-projection.
        # * Sampled: candidates are i.i.d. draws from Q_B; the corresponding
        #   Q marginal is the empirical projection of the SAME sample with
        #   uniform weights. Using the analytic per-position softmax would
        #   make ``pi`` empirical and ``Q`` analytical, so even at beta=0
        #   the log-ratio would not vanish at finite sample sizes. Project
        #   both ``pi`` and ``Q`` from the same draws so beta=0 yields a
        #   strict null (PLAN_RF.md §"D2 step behavior" 19).
        if mode == "sampled":
            n_cand = len(candidates)
            uniform = (
                np.full(n_cand, 1.0 / float(n_cand), dtype=np.float64)
                if n_cand > 0
                else np.zeros(0, dtype=np.float64)
            )
            Q_marg = project_marginals(
                candidates=candidates,
                weights_normalized=uniform,
                positions=editable,
                K_i_per_pos=K,
            )
        else:
            Q_marg = _per_position_Q_marginal(
                positions=editable, K_i_per_pos=K, struct_logits=structural_logits
            )

        rho_B_effective = (
            float(block.g_time)
            * float(block.g_comp)
            * float(block.g_ent)
            * float(g_ESS_candidates)
        )
        rho_B_effective = max(0.0, min(1.0, rho_B_effective))

        delta = compute_delta_logit(
            pi_marginals=pi,
            Q_marginals=Q_marg,
            eta=float(self.config.eta),
            rho_B=float(rho_B_effective),
            epsilon=float(self.config.epsilon),
            max_abs_shift=float(self.config.max_abs_logit_shift),
        )

        return D2BlockOutcome(
            block_id=int(block.block_id),
            omega_indices=omega,
            editable_positions=tuple(editable),
            K_i_per_pos=K,
            candidate_mode=mode,
            candidate_count=len(candidates),
            delta_R_B=tuple(float(x) for x in delta_R_B.tolist()),
            Q_B=tuple(float(x) for x in Q_B.tolist()),
            weights_unnormalized=tuple(float(x) for x in weights.tolist()),
            ess=float(ess),
            ess_fraction=float(ess_fraction),
            g_ESS_candidates=float(g_ESS_candidates),
            feasible=True,
            best_delta_R_B=float(best_dR),
            mean_delta_R_B=float(mean_dR),
            rho_B_effective=float(rho_B_effective),
            corrected_positions=tuple(editable),
            delta_logit=delta,
            skipped_reason=None,
            r_current=float(r_current),
        )


def _empty_block_outcome(
    *,
    block_id: int,
    omega: tuple[int, ...],
    reason: str,
    r_current: float = float("nan"),
) -> D2BlockOutcome:
    return D2BlockOutcome(
        block_id=block_id,
        omega_indices=omega,
        editable_positions=(),
        K_i_per_pos={},
        candidate_mode="",
        candidate_count=0,
        delta_R_B=(),
        Q_B=(),
        weights_unnormalized=(),
        ess=0.0,
        ess_fraction=0.0,
        g_ESS_candidates=0.0,
        feasible=False,
        best_delta_R_B=float("nan"),
        mean_delta_R_B=float("nan"),
        rho_B_effective=0.0,
        corrected_positions=(),
        delta_logit={},
        skipped_reason=reason,
        r_current=float(r_current),
    )


def _per_position_Q_marginal(
    *,
    positions: Sequence[int],
    K_i_per_pos: dict[int, tuple[int, ...]],
    struct_logits: torch.Tensor,
) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for p in positions:
        K = K_i_per_pos[int(p)]
        K_arr = np.asarray(K, dtype=np.int64)
        logits_K = (
            struct_logits[int(p), torch.from_numpy(K_arr)]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        logits_K = logits_K - float(logits_K.max())
        probs = np.exp(logits_K)
        probs = probs / probs.sum()
        out[int(p)] = {int(K_arr[j]): float(probs[j]) for j in range(K_arr.shape[0])}
    return out
