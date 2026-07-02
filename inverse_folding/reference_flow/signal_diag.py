"""SC-GR signal-direction diagnostic primitives (PLAN_RF_SC_GR_signal_diag.md).

Pure / dependency-injected helpers for the P2 reconstruct-and-score phase. They
REUSE the existing D2 candidate machinery (``counterfactual.py``) and SC-GR probe
machinery (``self_conditioned_gr.py``); nothing here re-derives the candidate gate
or builds a new sampler. The three signals per decision-point candidate tuple t:

    L = R_B(t)   local block risk (LME over Omega(B)) — what D2 ranks by
    P            cheap-probe terminal burden estimate (SC-GR fresh-K, topm_lse)
    Y            oracle terminal effect (real controller-off completions)

Convention: all three are levels, lower = better, the "pick" is ``argmin``.

The scorer is injected (``score_window_risk_batch_same_protein`` /
``score_batch_same_protein``) so these helpers are unit-testable with a fake head.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

from .counterfactual import compute_local_risk_batch
from .self_conditioned_gr import build_probe_samples, compute_risk_aggregates


# ---------------------------------------------------------------------------
# Geometry / selection (pure)
# ---------------------------------------------------------------------------


def windows_covering_positions(
    *,
    window_spans: Sequence[tuple[int, int]],
    positions: Sequence[int],
) -> tuple[int, ...]:
    """Indices of windows whose half-open ``[start, end)`` covers any position.

    Used for ``Omega(B)`` on the terminal design (Y_register): the union of
    binding windows covering ANY residue in the block ``A_B``.
    """
    pos = [int(p) for p in positions]
    out: list[int] = []
    for idx, (start, end) in enumerate(window_spans):
        s, e = int(start), int(end)
        if any(s <= p < e for p in pos):
            out.append(int(idx))
    return tuple(out)


def resolve_omega_indices(
    *,
    window_starts_0b: Sequence[int],
    window_ks: Sequence[int],
    omega_start_k: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Map block ``Omega`` windows captured as ``(start_0b, k)`` to column indices.

    Robust to window ordering: the P1 refresh captures the block's windows by
    their ``(start, k)`` coordinates; here we look each up in the candidate
    scoring window template. Fail-fast if a captured window is absent (template
    drift between the refresh and the candidate re-score).
    """
    lookup: dict[tuple[int, int], int] = {}
    for i, (s, k) in enumerate(zip(window_starts_0b, window_ks)):
        lookup.setdefault((int(s), int(k)), int(i))
    out: list[int] = []
    for s, k in omega_start_k:
        key = (int(s), int(k))
        if key not in lookup:
            raise KeyError(
                f"omega window (start={s}, k={k}) not in candidate window template"
            )
        out.append(lookup[key])
    return tuple(out)


def uniform_subsample_indices(
    *, n_total: int, cap: int, seed: int
) -> np.ndarray:
    """Deterministic uniform-random subsample of ``range(n_total)`` to ``cap``.

    Returns all indices (sorted) when ``n_total <= cap``. Never selects by L or P
    (avoids circularity, PLAN §2). Sorted for stable downstream ordering.
    """
    n = int(n_total)
    c = int(cap)
    if n <= c:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(n, size=c, replace=False)
    return np.sort(chosen.astype(np.int64))


def argmin_pick_flags(
    *, values: Sequence[float], eligible: Sequence[bool] | None = None
) -> np.ndarray:
    """Boolean mask, ``True`` at the (first) argmin over eligible finite values.

    Ineligible / non-finite entries are never picked; all-ineligible -> all
    ``False``.
    """
    vals = np.asarray(values, dtype=np.float64)
    n = vals.shape[0]
    flags = np.zeros(n, dtype=bool)
    if eligible is None:
        elig = np.ones(n, dtype=bool)
    else:
        elig = np.asarray(eligible, dtype=bool)
    mask = elig & np.isfinite(vals)
    if not mask.any():
        return flags
    masked_vals = np.where(mask, vals, np.inf)
    flags[int(np.argmin(masked_vals))] = True
    return flags


def overlay_tuple(
    *, base_tokens: np.ndarray, positions: Sequence[int], candidate_tuple: Sequence[int]
) -> np.ndarray:
    """Clone ``base_tokens`` and write ``candidate_tuple`` at ``positions``."""
    out = np.asarray(base_tokens, dtype=np.int64).copy()
    pos = [int(p) for p in positions]
    if len(pos) != len(candidate_tuple):
        raise ValueError("positions and candidate_tuple length mismatch")
    for p, tok in zip(pos, candidate_tuple):
        out[int(p)] = int(tok)
    return out


# ---------------------------------------------------------------------------
# Signal computation (scorer injected)
# ---------------------------------------------------------------------------


def _decode_records(
    *,
    base_tokens: np.ndarray,
    positions: Sequence[int],
    candidates: Sequence[tuple[int, ...]],
    decode_tokens: Callable[[np.ndarray], str],
    label_prefix: str,
) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for c_idx, cand in enumerate(candidates):
        toks = overlay_tuple(
            base_tokens=base_tokens, positions=positions, candidate_tuple=cand
        )
        records.append((f"{label_prefix}_c{c_idx}", decode_tokens(toks)))
    return records


def score_local_risk_L(
    *,
    completed_tokens: np.ndarray,
    positions: Sequence[int],
    candidates: Sequence[tuple[int, ...]],
    omega_start_k: Sequence[tuple[int, int]],
    decode_tokens: Callable[[np.ndarray], str],
    scorer: Any,
    protein_id: str,
    aggregation: str = "LME",
) -> dict[str, Any]:
    """L = R_B(t) per candidate: LME over ``Omega(B)`` of the hard completion.

    The base sequence is the refresh hard completion (argmax-masked ``completed_
    tokens``); each candidate overlays its tuple at the block positions. Window
    risks are scored as a compact ``[K, W]`` matrix and reduced over the block's
    ``Omega`` windows via :func:`compute_local_risk_batch` — the exact D2 ranking
    signal. Returns L plus the resolved window template (for telemetry).
    """
    records = _decode_records(
        base_tokens=completed_tokens,
        positions=positions,
        candidates=candidates,
        decode_tokens=decode_tokens,
        label_prefix="L",
    )
    compact = scorer.score_window_risk_batch_same_protein(
        protein_id=str(protein_id), records=records
    )
    window_risks = np.asarray(compact.window_risks, dtype=np.float64)
    omega_indices = resolve_omega_indices(
        window_starts_0b=compact.window_starts_0b,
        window_ks=compact.window_ks,
        omega_start_k=omega_start_k,
    )
    L = compute_local_risk_batch(
        window_risks=window_risks,
        omega_indices=omega_indices,
        aggregation=aggregation,
    )
    return {
        "L": L,
        "omega_indices": omega_indices,
        "window_starts_0b": tuple(int(s) for s in compact.window_starts_0b),
        "window_ks": tuple(int(k) for k in compact.window_ks),
    }


def _probe_fork_burden(
    score: Any,
    *,
    length: int,
    positions: Sequence[int],
    aggregation: str,
    tau_ref_B: float,
    top_m: int,
    lse_temperature: float,
    supra_tau_values: Sequence[float],
) -> float:
    """Reduce one probe completion's head windows to a scalar burden.

    ``aggregation="topm_lse"`` (default, deployable SC-GR readout): top-m LSE of
    the whole-sequence rectified excess ``max(0, proj - tau_ref_B)``. Always >=0;
    a DIFFERENT functional + scope than the oracle ``Y_register``.

    ``aggregation="omega_lme"`` (Y-aligned): raw LME over the ``Omega(B)`` windows
    covering ``A_B`` -- the EXACT functional + scope :func:`compute_Y_register`
    uses on the oracle completion, so P and Y_register then differ by exactly one
    axis (single-shot probe completion vs full controller-off trajectory). Use
    this for a clean, apples-to-apples probe-vs-Y comparison.
    """
    spans = [(int(w.start_0b), int(w.end_0b)) for w in score.windows]
    risks = [float(w.z) for w in score.windows]
    if aggregation == "omega_lme":
        return compute_Y_register(
            terminal_window_spans=spans,
            terminal_window_risks=risks,
            positions=positions,
        )
    if aggregation == "topm_lse":
        windows = [
            {"start": s, "end": e, "score": z} for (s, e), z in zip(spans, risks)
        ]
        agg = compute_risk_aggregates(
            windows=windows,
            length=int(length),
            tau_ref_B=float(tau_ref_B),
            top_m=int(top_m),
            lse_temperature=float(lse_temperature),
            supra_tau_values=tuple(supra_tau_values),
        )
        return float(agg.G_topm_lse)
    raise ValueError(f"unknown probe aggregation: {aggregation!r}")


def cheap_probe_P(
    *,
    snapshot_x_t: np.ndarray,
    struct_logits: np.ndarray,
    positions: Sequence[int],
    candidates: Sequence[tuple[int, ...]],
    mask_token_id: int,
    canonical_token_ids: Sequence[int],
    decode_tokens: Callable[[np.ndarray], str],
    scorer: Any,
    protein_id: str,
    tau_ref_B: float,
    k_probe: int = 3,
    struct_temperature: float = 1.0,
    top_m: int = 16,
    lse_temperature: float = 1.0,
    supra_tau_values: Sequence[float] = (11.75,),
    rng_key_base: Sequence[object] = (),
    aggregation: str = "topm_lse",
) -> np.ndarray:
    """Cheap-probe burden P per candidate (deployable signal under test).

    Fix tuple t into ``x_t`` on ``A_B`` (so those positions are committed), do
    NOT re-run the denoiser, and run the SC-GR fresh-arm pseudo-terminal
    completion over the OTHER masked positions using the ORIGINAL pre-D2
    structural logits (``k_probe`` forks). Each fork is head-scored and reduced
    per ``aggregation``; P(t) is the median over forks. This is a cheap ESTIMATE
    of Y, not Y.

    **CRN-paired across candidates**: the fork ``rng_key`` deliberately does NOT
    include the candidate index, so fork ``k`` draws the SAME RNG stream for every
    candidate (the masked-position set is candidate-invariant). The only thing
    that varies across candidates is the committed tuple -- mirroring the oracle
    ``Y``'s paired-seed design so ``P(A)-P(B)`` isolates the tuple instead of
    carrying independent completion noise.
    """
    length = int(np.asarray(snapshot_x_t).shape[0])
    P = np.empty(len(candidates), dtype=np.float64)
    for c_idx, cand in enumerate(candidates):
        x_probe = overlay_tuple(
            base_tokens=snapshot_x_t, positions=positions, candidate_tuple=cand
        )
        samples = build_probe_samples(
            x_t=x_probe,
            structural_logits=np.asarray(struct_logits, dtype=np.float64),
            mask_token_id=int(mask_token_id),
            canonical_token_ids=canonical_token_ids,
            arm="fresh",
            ensemble_size=int(k_probe),
            struct_temperature=float(struct_temperature),
            confidence_threshold=1.0,
            prev_state=None,
            rng_key=tuple(rng_key_base) + (str(protein_id),),
        )
        records = [
            (f"P_c{c_idx}_s{s_idx}", decode_tokens(s.tokens))
            for s_idx, s in enumerate(samples)
        ]
        batch = scorer.score_batch_same_protein(
            protein_id=str(protein_id), records=records
        )
        g_vals = [
            _probe_fork_burden(
                score, length=length, positions=positions, aggregation=aggregation,
                tau_ref_B=tau_ref_B, top_m=top_m, lse_temperature=lse_temperature,
                supra_tau_values=supra_tau_values,
            )
            for score in batch.scores
        ]
        P[c_idx] = float(np.median(g_vals)) if g_vals else float("nan")
    return P


def conditioned_probe_P(
    *,
    snapshot_x_t: np.ndarray,
    positions: Sequence[int],
    candidates: Sequence[tuple[int, ...]],
    denoiser: Callable[[Any, float, Any], Any],
    t: float,
    struct: Any,
    mask_token_id: int,
    canonical_token_ids: Sequence[int],
    decode_tokens: Callable[[np.ndarray], str],
    scorer: Any,
    protein_id: str,
    tau_ref_B: float,
    k_probe: int = 3,
    struct_temperature: float = 1.0,
    top_m: int = 16,
    lse_temperature: float = 1.0,
    supra_tau_values: Sequence[float] = (11.75,),
    rng_key_base: Sequence[object] = (),
    aggregation: str = "topm_lse",
) -> np.ndarray:
    """Conditioned-probe burden P_cond per candidate (Verdict 3 next step).

    Closer to Y than cheap-probe, cheaper than the oracle: commit tuple t into
    ``x_t`` on ``A_B``, **re-run the denoiser ONCE** on the committed sequence at
    the decision-point ``t`` to get logits that CONDITION on the tuple, then run
    the SC-GR fresh-arm single-shot pseudo-terminal completion over the OTHER
    masked positions using those conditioned logits (``k_probe`` forks),
    head-score, reduce per ``aggregation``, median over forks. Differs from
    :func:`cheap_probe_P` only in that the probe's structural logits reflect the
    committed tuple (one extra denoiser forward) instead of the frozen pre-D2
    snapshot logits. One forward + K_P one-shot completions per candidate; no
    iterative sampling to ``t=1`` (that is the oracle Y).

    **CRN-paired across candidates** (fork ``rng_key`` excludes the candidate
    index): mirrors the oracle ``Y`` so the only cross-candidate difference is the
    committed tuple + its conditioned-logit effect, not independent fork noise.
    """
    import torch

    base = np.asarray(snapshot_x_t, dtype=np.int64)
    length = int(base.shape[0])
    P = np.empty(len(candidates), dtype=np.float64)
    for c_idx, cand in enumerate(candidates):
        x_cond_np = overlay_tuple(
            base_tokens=base, positions=positions, candidate_tuple=cand
        )
        x_cond = torch.as_tensor(x_cond_np, dtype=torch.long)
        logits = denoiser(x_cond, float(t), struct)
        logits_np = np.asarray(
            logits.detach().cpu().numpy() if hasattr(logits, "detach") else logits,
            dtype=np.float64,
        )
        samples = build_probe_samples(
            x_t=x_cond_np,
            structural_logits=logits_np,
            mask_token_id=int(mask_token_id),
            canonical_token_ids=canonical_token_ids,
            arm="fresh",
            ensemble_size=int(k_probe),
            struct_temperature=float(struct_temperature),
            confidence_threshold=1.0,
            prev_state=None,
            rng_key=tuple(rng_key_base) + (str(protein_id),),
        )
        records = [
            (f"Pc_c{c_idx}_s{s_idx}", decode_tokens(s.tokens))
            for s_idx, s in enumerate(samples)
        ]
        batch = scorer.score_batch_same_protein(
            protein_id=str(protein_id), records=records
        )
        g_vals = [
            _probe_fork_burden(
                score, length=length, positions=positions, aggregation=aggregation,
                tau_ref_B=tau_ref_B, top_m=top_m, lse_temperature=lse_temperature,
                supra_tau_values=supra_tau_values,
            )
            for score in batch.scores
        ]
        P[c_idx] = float(np.median(g_vals)) if g_vals else float("nan")
    return P


def compute_Y_register(
    *,
    terminal_window_spans: Sequence[tuple[int, int]],
    terminal_window_risks: Sequence[float],
    positions: Sequence[int],
    aggregation: str = "LME",
) -> float:
    """Y_register: LME over Omega(B) of terminal per-window risk on one design.

    ``Omega(B)`` = windows covering ANY position in ``A_B`` on the FINISHED
    design (not just one position). Returns ``-inf`` if no window covers the
    block (degenerate; caller treats as missing).
    """
    omega = windows_covering_positions(
        window_spans=terminal_window_spans, positions=positions
    )
    L = compute_local_risk_batch(
        window_risks=np.asarray([list(terminal_window_risks)], dtype=np.float64),
        omega_indices=omega,
        aggregation=aggregation,
    )
    return float(L[0])


# ---------------------------------------------------------------------------
# Row assembly (pure)
# ---------------------------------------------------------------------------

CANDIDATE_COLUMNS = (
    "protein_id",
    "stratum",
    "decision_id",
    "step",
    "block_id",
    "positions_json",
    "candidate_tokens_json",
    "omega_window_indices_json",
    "L",
    "P",
    "Y_register",
    "Y_whole",
    "is_local_pick",
    "is_oracle_pick",
)


def assemble_candidate_rows(
    *,
    protein_id: str,
    stratum: str,
    decision_id: str,
    step: int,
    block_id: int,
    positions: Sequence[int],
    candidate_tuples: Sequence[tuple[int, ...]],
    omega_indices: Sequence[int],
    L: Sequence[float],
    P: Sequence[float],
    Y_register: Sequence[float],
    Y_whole: Sequence[float],
    y_subset_mask: Sequence[bool],
) -> list[dict[str, Any]]:
    """Build one parquet row per candidate tuple (schema = ``CANDIDATE_COLUMNS``).

    Picks (``is_local_pick`` / ``is_oracle_pick``) are computed over the Y-subset
    only (the tuples that have an oracle Y) so ``oracle_headroom = Y[local] -
    Y[oracle]`` is always computable while keeping the subset unbiased w.r.t. L/P.
    """
    n = len(candidate_tuples)
    y_subset = np.asarray(y_subset_mask, dtype=bool)
    L_arr = np.asarray(L, dtype=np.float64)
    Yr_arr = np.asarray(Y_register, dtype=np.float64)
    local_flags = argmin_pick_flags(values=L_arr, eligible=y_subset)
    oracle_flags = argmin_pick_flags(values=Yr_arr, eligible=y_subset)
    pos_json = json.dumps([int(p) for p in positions])
    omega_json = json.dumps([int(i) for i in omega_indices])
    rows: list[dict[str, Any]] = []
    for i in range(n):
        in_subset = bool(y_subset[i])
        rows.append(
            {
                "protein_id": str(protein_id),
                "stratum": str(stratum),
                "decision_id": str(decision_id),
                "step": int(step),
                "block_id": int(block_id),
                "positions_json": pos_json,
                "candidate_tokens_json": json.dumps(
                    [int(x) for x in candidate_tuples[i]]
                ),
                "omega_window_indices_json": omega_json,
                "L": float(L_arr[i]),
                "P": float(P[i]),
                "Y_register": float(Yr_arr[i]) if in_subset else float("nan"),
                "Y_whole": float(Y_whole[i]) if in_subset else float("nan"),
                "is_local_pick": bool(local_flags[i]),
                "is_oracle_pick": bool(oracle_flags[i]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# C0b — Path-C headroom existence test (PLANNER_README §C0b)
# ---------------------------------------------------------------------------
#
# Offline test: on a high-r_i "register" (a contiguous ~9-residue window with
# top-tercile per-residue immune risk), does a multi-step RECONDITIONED joint
# resample reduce terminal immune (LME local risk R_B over the register's
# covering windows, in head-logit nats) MORE than the best single-position
# structurally-safe edit, at acceptable structure? The four helpers below are the
# pure, scorer/decoder-free core; the DPLM joint decode + head scoring live in
# the ``--mode c0b`` driver. All values are levels, lower R_B = better.


def select_high_ri_registers(
    *,
    residue_risks: Sequence[float],
    register_width: int = 9,
    max_registers: int = 4,
    min_separation: int | None = None,
    quantile: float = 2.0 / 3.0,
) -> tuple[tuple[int, ...], ...]:
    """High-r_i contiguous registers from one design's per-residue risk.

    Score every width-``register_width`` window by its MEAN r_i; keep windows
    whose mean is at or above the design's ``quantile`` of window means (top
    tercile by default), then greedily select in descending mean (ties -> lower
    start), skipping any window whose start is within ``min_separation`` (default
    ``register_width`` => non-overlapping) of an already-selected one, up to
    ``max_registers``. Returns tuples of 0-based residue positions (each of length
    ``register_width``); empty when the design is shorter than the window.
    """
    risks = np.asarray([float(r) for r in residue_risks], dtype=np.float64)
    length = int(risks.shape[0])
    w = int(register_width)
    if length < w or w <= 0:
        return ()
    sep = int(min_separation) if min_separation is not None else w
    starts = list(range(0, length - w + 1))
    means = np.array([float(risks[s : s + w].mean()) for s in starts], dtype=np.float64)
    thr = float(np.quantile(means, float(quantile)))
    eligible = [(int(s), float(means[i])) for i, s in enumerate(starts) if means[i] >= thr]
    eligible.sort(key=lambda sm: (-sm[1], sm[0]))
    chosen: list[int] = []
    for s, _m in eligible:
        if len(chosen) >= int(max_registers):
            break
        if all(abs(s - c) >= sep for c in chosen):
            chosen.append(s)
    chosen.sort()
    return tuple(tuple(range(s, s + w)) for s in chosen)


def single_position_candidate_tuples(
    *,
    design_register_tokens: Sequence[int],
    register_positions: Sequence[int],
    safe_support: dict[int, Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Every single-position edit over the register (exactly one position differs).

    For each register position and each safe-support token that is NOT the
    design's own residue there, emit a register-length tuple equal to
    ``design_register_tokens`` except that one substitution. Deterministic order:
    positions ascending, then ``safe_support`` order. The identity (no-op) is
    excluded so each candidate is a genuine single-token edit.
    """
    design = [int(t) for t in design_register_tokens]
    pos = [int(p) for p in register_positions]
    if len(design) != len(pos):
        raise ValueError("design_register_tokens and register_positions length mismatch")
    out: list[tuple[int, ...]] = []
    for j, p in enumerate(pos):
        for tok in safe_support.get(int(p), ()):  # type: ignore[arg-type]
            if int(tok) == design[j]:
                continue
            cand = list(design)
            cand[j] = int(tok)
            out.append(tuple(cand))
    return tuple(out)


def _best_acceptable(rs, sctms, accept) -> tuple[float, float, int]:
    rs_arr = np.asarray([float(x) for x in rs], dtype=np.float64)
    sct_arr = np.asarray([float(x) for x in sctms], dtype=np.float64)
    idxs = [
        i
        for i in range(rs_arr.shape[0])
        if np.isfinite(rs_arr[i]) and accept(i, float(sct_arr[i]))
    ]
    if not idxs:
        return float("nan"), float("nan"), -1
    best = min(idxs, key=lambda i: float(rs_arr[i]))
    return float(rs_arr[best]), float(sct_arr[best]), int(best)


def compute_c0b_register_verdict(
    *,
    r_single: Sequence[float],
    sctm_single: Sequence[float],
    r_fake: Sequence[float],
    sctm_fake: Sequence[float],
    r_true: Sequence[float],
    sctm_true: Sequence[float],
    r_true_deep: Sequence[float] = (),
    sctm_true_deep: Sequence[float] = (),
    design_sctm: float | None = None,
    coordination_margin_thresh: float = 0.2,
    sctm_tol: float = 0.05,
    sctm_floor: float = 0.5,
) -> dict[str, Any]:
    """Per-register THREE-arm verdict isolating coordination from multi-edit freedom.

    R_B = local immune (head-LME nats, lower = better). Three arms: ``single``
    (one structurally-safe edit), ``fake`` (the whole register resampled in ONE
    forward = per-position marginal product), ``true`` (multi-step RECONDITIONED
    resample). Each arm's best = min R_B among STRUCTURE-ACCEPTABLE candidates
    (``sctm > sctm_floor`` AND, when a design baseline is given, non-inferior to
    it: ``sctm >= design_sctm - sctm_tol``).

    * **Primary — coordination** (the Path-C build/no-build decision):
      ``coordination_margin = R_fake_best - R_true_best`` (true's EXTRA reduction
      over the best fake at acceptable structure). ``coordination_wins`` = true is
      feasible AND (fake has NO structure-acceptable candidate OR
      ``coordination_margin >= coordination_margin_thresh``). The fake-infeasible
      branch is itself a coordination win: reconditioning reached a structure-
      acceptable immune reduction the uncoordinated marginal product could not
      (the user's "9 independent edits wreck the fold" hypothesis).
    * **Secondary — degrees of freedom**: ``dof_margin = R_single_best -
      R_fake_best``. Large ``dof_margin`` with ``coordination_margin ≈ 0`` ⇒ raise
      the D2 per-block editable cap (cheap, factorized), do NOT build a sampler.
    """
    has_baseline = design_sctm is not None and np.isfinite(float(design_sctm))
    bar = (float(design_sctm) - float(sctm_tol)) if has_baseline else None

    def _accept(i: int, s: float) -> bool:
        if not (s > float(sctm_floor)):
            return False
        return True if bar is None else (s >= bar)

    R_single_best, single_sctm, single_idx = _best_acceptable(r_single, sctm_single, _accept)
    R_fake_best, fake_sctm, fake_idx = _best_acceptable(r_fake, sctm_fake, _accept)
    R_true18_best, true18_sctm, true18_idx = _best_acceptable(r_true, sctm_true, _accept)
    R_true36_best, true36_sctm, true36_idx = _best_acceptable(r_true_deep, sctm_true_deep, _accept)
    # Best reconditioned true over the depth sweep: a true~fake at one depth must
    # NOT read as NO-GO if a deeper sweep clears the bar (under-cooked, not useless).
    _true_pool = [x for x in (R_true18_best, R_true36_best) if np.isfinite(x)]
    R_true_best = float(min(_true_pool)) if _true_pool else float("nan")

    single_ok = bool(np.isfinite(R_single_best))
    fake_ok = bool(np.isfinite(R_fake_best))
    true18_ok = bool(np.isfinite(R_true18_best))
    true36_ok = bool(np.isfinite(R_true36_best))
    true_ok = bool(np.isfinite(R_true_best))

    coordination_margin = (
        float(R_fake_best - R_true_best) if (true_ok and fake_ok) else float("nan")
    )
    coordination_margin_18 = (
        float(R_fake_best - R_true18_best) if (true18_ok and fake_ok) else float("nan")
    )
    # > 0 ⇒ the deeper sweep reduces immune MORE than the canonical depth ⇒ 18 was
    # under-cooked (false-NO-GO risk); ≈ 0 ⇒ more reconditioning buys nothing.
    depth_gap = (
        float(R_true18_best - R_true36_best) if (true18_ok and true36_ok) else float("nan")
    )
    dof_margin = (
        float(R_single_best - R_fake_best) if (single_ok and fake_ok) else float("nan")
    )
    true_vs_single_margin = (
        float(R_single_best - R_true_best) if (single_ok and true_ok) else float("nan")
    )

    def _coord_wins(R_true: float) -> bool:
        if not np.isfinite(R_true):
            return False
        if not fake_ok:
            return True  # true feasible where the marginal product is not
        return float(R_fake_best - R_true) >= float(coordination_margin_thresh)

    coordination_wins = _coord_wins(R_true_best)            # best-of-sweep verdict
    coordination_wins_18 = _coord_wins(R_true18_best)       # canonical depth only
    needs_depth = bool(coordination_wins and not coordination_wins_18)
    return {
        "R_single_best": R_single_best,
        "R_fake_best": R_fake_best,
        "R_true_best": R_true_best,
        "R_true18_best": R_true18_best,
        "R_true36_best": R_true36_best,
        "single_best_sctm": single_sctm,
        "fake_best_sctm": fake_sctm,
        "design_sctm": float(design_sctm) if has_baseline else float("nan"),
        "single_acceptable": single_ok,
        "fake_acceptable": fake_ok,
        "true_acceptable": true_ok,
        "true18_acceptable": true18_ok,
        "true36_acceptable": true36_ok,
        "coordination_margin": coordination_margin,
        "coordination_margin_18": coordination_margin_18,
        "depth_gap": depth_gap,
        "dof_margin": dof_margin,
        "true_vs_single_margin": true_vs_single_margin,
        "coordination_wins": coordination_wins,
        "coordination_wins_18": coordination_wins_18,
        "needs_depth": needs_depth,
    }


def c0b_go_fraction(
    *, joint_wins_flags: Sequence[bool], win_fraction: float = 1.0 / 3.0
) -> dict[str, Any]:
    """GO iff at least ``win_fraction`` of registers had ``joint_wins`` (default 1/3)."""
    flags = [bool(f) for f in joint_wins_flags]
    n = len(flags)
    n_wins = int(sum(flags))
    fraction = (n_wins / n) if n else 0.0
    go = bool(n > 0 and fraction >= float(win_fraction))
    return {"n_total": n, "n_wins": n_wins, "fraction": fraction, "go": go}
