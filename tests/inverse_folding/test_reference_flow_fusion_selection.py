"""F7 unit tests — fusion/selection.py greedy / hard-beam / FK selectors (§1.6).

Greedy and beam both require the ε_H Head-improvement margin (doc §4/§5.5). FK uses
offspring-count-normalized log-space weights so multiplicity does not buy ancestry mass;
infeasible children get zero weight; the null parent always has positive support; resampling
is deterministic under a fixed seed; large Head values do not overflow.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from inverse_folding.reference_flow.fusion import selection as sel
from inverse_folding.reference_flow.fusion import state as st

SEQ = "ACDEFGHIKL"


def _parent(idx, gr, weight=0.5):
    return st.ParticleState(
        particle_id=f"P:r0:s{idx}:aaaaaaaaaaaa", protein_id="P", round_idx=0, slot_idx=idx,
        sequence=SEQ, sequence_md5=st.sequence_md5(SEQ), weight=weight, parent_particle_id=None,
        source_proposal_id=None, head_global_risk=gr,
        structure=SimpleNamespace(scTM=0.9, active_site_RMSD=1.0, passed=True), feasible=True,
        lineage_seed=idx)


def _eval(parent_id, pid, gr, *, feasible=True, ld=0.0):
    return st.CandidateEvaluation(
        sequence_md5=st.sequence_md5(pid), parent_particle_id=parent_id, proposal_id=pid,
        target_start_0b=0, target_end_0b=5, head_global_risk=gr, aligned_windows=(),
        local_target_delta=ld, new_hotspot_max=0.0, new_hotspot_mass=0.0, new_hotspot_count=0,
        structure=SimpleNamespace(scTM=0.9, active_site_RMSD=1.0, passed=True) if feasible else None,
        structure_evaluated=feasible, feasible=feasible, reason="ok" if feasible else "x")


# --------------------------------------------------------------------------- #
# greedy — global drives selection, not local
# --------------------------------------------------------------------------- #
def test_greedy_selects_lowest_global_not_best_local():
    p = _parent(0, 5.0)
    low_global = _eval(p.particle_id, "a", 1.0, ld=+0.5)   # worse local, best global
    better_local = _eval(p.particle_id, "b", 3.0, ld=-0.9)  # better local, worse global
    chosen = sel.select_greedy(p, [low_global, better_local], min_head_improvement=0.05)
    assert chosen is low_global


def test_greedy_keeps_parent_when_no_margin():
    p = _parent(0, 5.0)
    tiny = _eval(p.particle_id, "a", 4.99)  # improves by 0.01 < 0.05
    assert sel.select_greedy(p, [tiny], min_head_improvement=0.05) is None


# --------------------------------------------------------------------------- #
# hard-beam — global top-N, requires the ε_H margin
# --------------------------------------------------------------------------- #
def test_beam_keeps_n_lowest_head_eligible_children():
    p0, p1 = _parent(0, 10.0), _parent(1, 10.0)
    evals = {p0.particle_id: [_eval(p0.particle_id, "c00", 2.0)],
             p1.particle_id: [_eval(p1.particle_id, "c10", 3.0)]}
    picks = sel.select_beam([p0, p1], evals, min_head_improvement=0.05, N=2)
    chosen_pids = {e.proposal_id for _pid, e in picks if e is not None}
    assert chosen_pids == {"c00", "c10"}  # both children beat their parents by >= margin


def test_beam_excludes_sub_margin_child():
    p0 = _parent(0, 10.0)
    evals = {p0.particle_id: [_eval(p0.particle_id, "c", 9.99)]}  # improves by 0.01 < margin
    picks = sel.select_beam([p0], evals, min_head_improvement=0.05, N=1)
    assert picks == [(p0.particle_id, None)]  # only the null parent is eligible


# --------------------------------------------------------------------------- #
# FK — offspring-count normalization: multiplicity does not buy ancestry mass
# --------------------------------------------------------------------------- #
def test_fk_offspring_count_normalization_equalizes_parent_mass():
    a, b = _parent(0, 5.0, weight=0.5), _parent(1, 5.0, weight=0.5)
    evals = {a.particle_id: [_eval(a.particle_id, "a0", 5.0)],  # 1 child
             b.particle_id: [_eval(b.particle_id, f"b{i}", 5.0) for i in range(5)]}  # 5 children
    pids, _evals, norm, ess = sel.fk_weights([a, b], evals, beta_r=1.0, beta_prev=1.0)
    mass_a = sum(w for pid, w in zip(pids, norm) if pid == a.particle_id)
    mass_b = sum(w for pid, w in zip(pids, norm) if pid == b.particle_id)
    assert mass_a == pytest.approx(0.5)
    assert mass_b == pytest.approx(0.5)  # despite B having 5x the children


def test_fk_infeasible_children_get_zero_weight():
    a = _parent(0, 5.0, weight=1.0)
    evals = {a.particle_id: [_eval(a.particle_id, "good", 1.0),
                             _eval(a.particle_id, "bad", 0.0, feasible=False)]}
    pids, evs, norm, ess = sel.fk_weights([a], evals, beta_r=1.0, beta_prev=1.0)
    bad_idx = [i for i, e in enumerate(evs) if e is not None and e.proposal_id == "bad"]
    assert bad_idx == []  # infeasible child is omitted entirely (zero weight)


def test_fk_null_parent_has_positive_support():
    a = _parent(0, 5.0, weight=1.0)
    evals = {a.particle_id: [_eval(a.particle_id, "c", 1.0)]}
    pids, evs, norm, ess = sel.fk_weights([a], evals, beta_r=1.0, beta_prev=1.0)
    null_w = sum(w for e, w in zip(evs, norm) if e is None)
    assert null_w > 0.0


def test_fk_lower_head_child_gets_more_mass_at_positive_beta():
    a = _parent(0, 5.0, weight=1.0)
    evals = {a.particle_id: [_eval(a.particle_id, "low", 1.0), _eval(a.particle_id, "high", 4.0)]}
    pids, evs, norm, ess = sel.fk_weights([a], evals, beta_r=1.0, beta_prev=1.0)
    w = {e.proposal_id: nw for e, nw in zip(evs, norm) if e is not None}
    assert w["low"] > w["high"]  # exp(-beta R_H) favors the lower-Head child


def test_fk_resample_is_deterministic_under_fixed_seed():
    a, b = _parent(0, 5.0, weight=0.5), _parent(1, 3.0, weight=0.5)
    evals = {a.particle_id: [_eval(a.particle_id, "a0", 4.0)],
             b.particle_id: [_eval(b.particle_id, "b0", 2.0)]}
    p1, _ = sel.select_fk([a, b], evals, beta_r=1.0, beta_prev=1.0, N=4, rng=np.random.default_rng(42))
    p2, _ = sel.select_fk([a, b], evals, beta_r=1.0, beta_prev=1.0, N=4, rng=np.random.default_rng(42))
    assert p1 == p2 and len(p1) == 4


def test_fk_log_space_is_numerically_stable_for_large_head():
    a = _parent(0, 800.0, weight=1.0)
    evals = {a.particle_id: [_eval(a.particle_id, "c", 700.0)]}
    pids, evs, norm, ess = sel.fk_weights([a], evals, beta_r=2.0, beta_prev=1.0)
    assert all(np.isfinite(w) for w in norm)
    assert sum(norm) == pytest.approx(1.0)


def test_fk_resample_independent_of_candidate_row_order():
    # persisting/reloading the pool in a different row order must NOT change FK ancestry
    a = _parent(0, 5.0, weight=1.0)
    fwd = [_eval(a.particle_id, "c1", 2.0), _eval(a.particle_id, "c2", 3.0), _eval(a.particle_id, "c3", 4.0)]
    rev = list(reversed(fwd))
    p1, _ = sel.select_fk([a], {a.particle_id: fwd}, beta_r=1.0, beta_prev=1.0, N=6, rng=np.random.default_rng(7))
    p2, _ = sel.select_fk([a], {a.particle_id: rev}, beta_r=1.0, beta_prev=1.0, N=6, rng=np.random.default_rng(7))

    def _key(pick):
        pid, ev = pick
        return (pid, None if ev is None else ev.proposal_id)

    assert [_key(x) for x in p1] == [_key(x) for x in p2]
