"""Selection laws over the evaluated candidate pool (§1.6).

F5 provides the greedy control (per-parent, require Head improvement). F7 adds the hard-beam
control (global, also require the Head margin — per doc §4/§5.5) and Head-guided FK optimization
(offspring-count-normalized log-space weights + multinomial ancestry resampling). All three
replay the SAME evaluated candidate pool; only definitively feasible children are eligible.
Pure Python — no torch.
"""
from __future__ import annotations

import math


def _logsumexp(logs: list[float]) -> float:
    finite = [x for x in logs if x != float("-inf")]
    if not finite:
        return float("-inf")
    m = max(finite)
    return m + math.log(math.fsum(math.exp(x - m) for x in logs if x != float("-inf")))


def select_greedy(parent, child_evals, *, min_head_improvement: float):
    """Greedy control: the parent takes its lowest-Head feasible child only if it improves by
    the margin ``min_head_improvement``; otherwise it keeps itself (null move). Returns the
    selected ``CandidateEvaluation`` or ``None`` (keep parent). No ancestry crosses lanes.
    """
    feasible = [e for e in child_evals if e.feasible]
    if not feasible:
        return None
    best = min(feasible, key=lambda e: (e.head_global_risk, e.proposal_id))
    if best.head_global_risk <= parent.head_global_risk - min_head_improvement:
        return best
    return None


def select_beam(parents, evals_by_parent, *, min_head_improvement: float, N: int):
    """Hard-beam control (§1.6). Flatten the feasible null parents and the beam-eligible
    children — a child is eligible only if it improves its own parent by
    ``min_head_improvement`` (the same ε_H margin as greedy; per doc §4/§5.5 both controls are
    conservative require-improvement controls). Keep the ``N`` lowest-Head exact states,
    deterministic ties by sequence MD5 then proposal ID. Elite preservation is handled by the
    caller. Returns ``N`` picks ``(parent_particle_id, candidate_eval_or_None)`` where ``None``
    is the null (parent) move.
    """
    states = []  # (risk, md5, proposal_tiebreak, parent_id, eval_or_None)
    for p in parents:
        states.append((p.head_global_risk, p.sequence_md5, "", p.particle_id, None))
        for e in evals_by_parent.get(p.particle_id, []):
            if e.feasible and e.head_global_risk <= p.head_global_risk - min_head_improvement:
                states.append((e.head_global_risk, e.sequence_md5, e.proposal_id,
                               p.particle_id, e))
    states.sort(key=lambda s: (s[0], s[1], s[2]))
    kept = states[:N]
    return [(s[3], s[4]) for s in kept]


def fk_weights(parents, evals_by_parent, *, beta_r: float, beta_prev: float):
    """Normalized FK branch weights (§1.6), before resampling. For parent ``a`` with ``M_a``
    feasible children plus its null parent, each of the ``M_a+1`` branches gets proposal mass
    ``q_a = 1/(M_a+1)`` so multiplicity does not buy ancestry mass. The unnormalized log-weight
    of a child ``y`` is

        log w_prev[a] + log q_a + (-beta_r * R_H(y) + beta_prev * R_H(parent_a)),

    with the null branch using ``R_H(parent_a)`` for both terms. Infeasible children are
    omitted (zero weight). Normalization is done in log space (never exponentiate raw Head).
    Returns ``(parent_ids, evals, norm_weights, ess)`` with ``evals[i] is None`` for a null move.
    """
    parent_ids: list = []
    evals: list = []
    log_weights: list = []
    for p in parents:
        feasible = [e for e in evals_by_parent.get(p.particle_id, []) if e.feasible]
        log_q = -math.log(len(feasible) + 1)
        log_w_prev = math.log(p.weight) if p.weight > 0.0 else float("-inf")
        pot_null = (-beta_r + beta_prev) * p.head_global_risk  # null: child == parent
        parent_ids.append(p.particle_id)
        evals.append(None)
        log_weights.append(log_w_prev + log_q + pot_null)
        for e in feasible:
            pot = -beta_r * e.head_global_risk + beta_prev * p.head_global_risk
            parent_ids.append(p.particle_id)
            evals.append(e)
            log_weights.append(log_w_prev + log_q + pot)

    # Canonicalize entry order (parent_id, then proposal_id; null first) so the resample is
    # reproducible regardless of the candidate pool's row order after persistence/reload.
    order = sorted(range(len(parent_ids)),
                   key=lambda i: (parent_ids[i], "" if evals[i] is None else evals[i].proposal_id))
    parent_ids = [parent_ids[i] for i in order]
    evals = [evals[i] for i in order]
    log_weights = [log_weights[i] for i in order]

    log_z = _logsumexp(log_weights)
    norm = [math.exp(lw - log_z) if lw != float("-inf") else 0.0 for lw in log_weights]
    total = math.fsum(norm)
    if total > 0:
        norm = [w / total for w in norm]  # guard tiny drift
    ess = 1.0 / math.fsum(w * w for w in norm) if any(norm) else 0.0
    return parent_ids, evals, norm, ess


def select_fk(parents, evals_by_parent, *, beta_r: float, beta_prev: float, N: int, rng):
    """Head-guided FK optimization: multinomial-resample ``N`` slots with replacement from the
    normalized ``fk_weights`` using the provided numpy Generator. Returns ``(picks, ess)`` where
    each pick is ``(parent_particle_id, candidate_eval_or_None)`` and ``ess`` is pre-resample.
    """
    parent_ids, evals, norm, ess = fk_weights(
        parents, evals_by_parent, beta_r=beta_r, beta_prev=beta_prev)
    idx = rng.choice(len(norm), size=N, replace=True, p=norm)
    return [(parent_ids[i], evals[i]) for i in idx], ess
