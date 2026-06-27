"""SC-GR pure probe-helper tests (PLAN_RF_SC_GR.md Task SC0.2)."""

from __future__ import annotations

import numpy as np
import pytest

from inverse_folding.reference_flow.self_conditioned_gr import (
    SCGRProbeSample,
    SCGRRiskAggregates,
    SCGRState,
    build_probe_samples,
    compute_risk_aggregates,
    summarize_probe_refresh,
    supra_tau_label,
    update_state_from_structural_argmax,
)


CANONICAL = list(range(20))  # token ids 0..19 are the canonical AAs
MASK = 99


def _logits(L: int, V: int = 25, *, peak_token: int = 3, peak: float = 5.0) -> np.ndarray:
    """Deterministic logits where canonical token ``peak_token`` is the argmax."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((L, V)) * 0.1
    z[:, peak_token] += peak
    return z


# ---------------------------------------------------------------------------
# build_probe_samples
# ---------------------------------------------------------------------------


def test_committed_positions_stay_fixed():
    L = 8
    x_t = np.array([5, MASK, 7, MASK, MASK, 2, MASK, 1], dtype=np.int64)
    logits = _logits(L)
    samples = build_probe_samples(
        x_t=x_t,
        structural_logits=logits,
        mask_token_id=MASK,
        canonical_token_ids=CANONICAL,
        arm="fresh",
        ensemble_size=3,
        struct_temperature=1.0,
        confidence_threshold=0.7,
        prev_state=None,
        rng_key=("P1", 0, 42, 5),
    )
    assert len(samples) == 3
    committed = x_t != MASK
    for s in samples:
        assert np.array_equal(s.tokens[committed], x_t[committed])
        # masked positions are filled (no MASK left)
        assert not np.any(s.tokens == MASK)


def test_fresh_arm_samples_every_masked_position_and_canonical_only():
    L = 6
    x_t = np.array([MASK, MASK, 4, MASK, MASK, MASK], dtype=np.int64)
    samples = build_probe_samples(
        x_t=x_t,
        structural_logits=_logits(L),
        mask_token_id=MASK,
        canonical_token_ids=CANONICAL,
        arm="fresh",
        ensemble_size=2,
        struct_temperature=1.0,
        confidence_threshold=0.7,
        prev_state=None,
        rng_key=("P1", 0, 42, 5),
    )
    masked = x_t == MASK
    for s in samples:
        assert s.num_masked == int(masked.sum())
        assert s.num_reused_from_prev == 0
        assert s.reuse_fraction == 0.0
        assert s.state_bootstrap_flag is False
        # every filled masked position is a canonical token
        assert set(int(t) for t in s.tokens[masked]).issubset(set(CANONICAL))


def test_self_conditioned_reuses_only_high_confidence_prev_tokens():
    L = 5
    x_t = np.array([MASK, MASK, MASK, MASK, 9], dtype=np.int64)
    prev_tokens = np.array([11, 12, 13, 14, 15], dtype=np.int64)
    # positions 0 and 2 high confidence; 1 and 3 below threshold
    prev_conf = np.array([0.95, 0.10, 0.80, 0.20, 0.99], dtype=float)
    state = SCGRState(prev_x1_hat=prev_tokens, prev_confidence=prev_conf)
    samples = build_probe_samples(
        x_t=x_t,
        structural_logits=_logits(L),
        mask_token_id=MASK,
        canonical_token_ids=CANONICAL,
        arm="self_conditioned",
        ensemble_size=3,
        struct_temperature=1.0,
        confidence_threshold=0.7,
        prev_state=state,
        rng_key=("P1", 0, 42, 10),
    )
    for s in samples:
        # high-confidence masked positions reuse the prev token verbatim
        assert int(s.tokens[0]) == 11
        assert int(s.tokens[2]) == 13
        # low-confidence masked positions are sampled (canonical), not reused
        assert int(s.tokens[1]) != 12 or int(s.tokens[1]) in CANONICAL
        assert int(s.tokens[3]) in CANONICAL
        assert s.num_reused_from_prev == 2
        assert s.reuse_fraction == pytest.approx(2 / 4)
        assert s.mean_prev_confidence_reused == pytest.approx((0.95 + 0.80) / 2)
        assert s.state_bootstrap_flag is False


def test_self_conditioned_without_prev_state_falls_back_to_fresh():
    L = 4
    x_t = np.array([MASK, MASK, 3, MASK], dtype=np.int64)
    samples = build_probe_samples(
        x_t=x_t,
        structural_logits=_logits(L),
        mask_token_id=MASK,
        canonical_token_ids=CANONICAL,
        arm="self_conditioned",
        ensemble_size=2,
        struct_temperature=1.0,
        confidence_threshold=0.7,
        prev_state=None,
        rng_key=("P1", 0, 42, 0),
    )
    for s in samples:
        assert s.state_bootstrap_flag is True
        assert s.num_reused_from_prev == 0
        assert s.reuse_fraction == 0.0


def test_build_probe_samples_is_deterministic():
    L = 6
    x_t = np.array([MASK, MASK, MASK, MASK, MASK, MASK], dtype=np.int64)
    kwargs = dict(
        x_t=x_t,
        structural_logits=_logits(L),
        mask_token_id=MASK,
        canonical_token_ids=CANONICAL,
        arm="fresh",
        ensemble_size=4,
        struct_temperature=1.0,
        confidence_threshold=0.7,
        prev_state=None,
        rng_key=("P1", 0, 42, 5),
    )
    a = build_probe_samples(**kwargs)
    b = build_probe_samples(**kwargs)
    for sa, sb in zip(a, b):
        assert np.array_equal(sa.tokens, sb.tokens)
    # different sample_idx within a refresh should not be identical across all
    assert not all(np.array_equal(a[0].tokens, a[i].tokens) for i in range(1, 4))


# ---------------------------------------------------------------------------
# update_state_from_structural_argmax
# ---------------------------------------------------------------------------


def test_state_update_independent_of_head_and_canonical_confidence():
    L = 4
    logits = _logits(L, peak_token=3, peak=6.0)
    argmax_tokens = np.array([3, 3, 3, 3], dtype=np.int64)
    state = update_state_from_structural_argmax(
        structural_argmax_tokens=argmax_tokens,
        structural_logits=logits,
        canonical_token_ids=CANONICAL,
    )
    assert np.array_equal(state.prev_x1_hat, argmax_tokens)
    assert state.prev_confidence.shape == (L,)
    # confidence is a probability in [0, 1]
    assert np.all(state.prev_confidence >= 0.0) and np.all(state.prev_confidence <= 1.0)
    # the sharply-peaked canonical token gives high confidence
    assert np.all(state.prev_confidence > 0.5)


# ---------------------------------------------------------------------------
# compute_risk_aggregates
# ---------------------------------------------------------------------------


def _win(start, end, score):
    return {"start": start, "end": end, "score": score}


def test_topm_lse_more_tail_sensitive_than_mean_on_focal_spike():
    L = 20
    # one focal hot window, rest flat low
    windows = [_win(0, 20, 1.0), _win(5, 7, 15.0)]
    agg = compute_risk_aggregates(
        windows=windows,
        length=L,
        tau_ref_B=0.0,
        top_m=4,
        lse_temperature=1.0,
        supra_tau_values=[11.75],
    )
    assert agg.G_topm_lse > agg.G_mean_excess
    # mean is diluted by the long flat tail; top-m LSE concentrates on the spike
    assert agg.G_mean_excess < 5.0
    assert agg.G_topm_lse > 10.0


def test_supra_mass_and_length_normalization():
    L = 10
    windows = [_win(0, 10, 5.0), _win(0, 2, 20.0)]  # 2 residues at 20, 8 at 5
    agg = compute_risk_aggregates(
        windows=windows,
        length=L,
        tau_ref_B=0.0,
        top_m=4,
        lse_temperature=1.0,
        supra_tau_values=[11.75],
    )
    # residue_excess: [20,20,5,5,5,5,5,5,5,5]; mean = (40 + 40)/10 = 8.0
    assert agg.G_mean_excess == pytest.approx(8.0)
    # supra mass tau=11.75: only the two 20s contribute (20-11.75)=8.25 each /10
    assert agg.supra_masses[11.75] == pytest.approx(2 * 8.25 / 10)
    assert agg.head_risk_max == pytest.approx(20.0)


def test_compute_risk_aggregates_empty_windows_returns_zeros():
    agg = compute_risk_aggregates(
        windows=[],
        length=8,
        tau_ref_B=0.0,
        top_m=4,
        lse_temperature=1.0,
        supra_tau_values=[11.75],
    )
    assert agg.G_mean_excess == 0.0
    assert agg.G_topm_lse == 0.0
    assert agg.supra_masses[11.75] == 0.0
    assert agg.head_risk_LME == 0.0
    assert agg.head_risk_max == 0.0


def test_compute_risk_aggregates_exposes_residue_excess_map():
    # The per-residue excess map underlying the scalar aggregators is surfaced
    # for per-residue (r_i) targeting telemetry (SC-GR signal-direction
    # follow-up). It must equal max-covering projection minus tau, clipped at 0.
    L = 10
    windows = [_win(0, 10, 5.0), _win(0, 2, 20.0)]  # 2 residues at 20, 8 at 5
    agg = compute_risk_aggregates(
        windows=windows,
        length=L,
        tau_ref_B=0.0,
        top_m=4,
        lse_temperature=1.0,
        supra_tau_values=[11.75],
    )
    assert agg.residue_excess.shape == (L,)
    assert agg.residue_excess.tolist() == [20.0, 20.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    # the scalar mean_excess is exactly the mean of the exposed per-residue map
    assert agg.G_mean_excess == pytest.approx(float(agg.residue_excess.mean()))


def test_residue_excess_empty_windows_is_length_l_zeros():
    agg = compute_risk_aggregates(
        windows=[],
        length=8,
        tau_ref_B=0.0,
        top_m=4,
        lse_temperature=1.0,
        supra_tau_values=[11.75],
    )
    assert agg.residue_excess.shape == (8,)
    assert not agg.residue_excess.any()


def test_tau_ref_subtraction_clips_at_zero():
    L = 4
    windows = [_win(0, 4, 2.0)]
    agg = compute_risk_aggregates(
        windows=windows,
        length=L,
        tau_ref_B=5.0,  # above all scores -> excess clipped to 0
        top_m=4,
        lse_temperature=1.0,
        supra_tau_values=[1.0],
    )
    assert agg.G_mean_excess == 0.0


# ---------------------------------------------------------------------------
# supra_tau_label + summarize_probe_refresh
# ---------------------------------------------------------------------------


def test_supra_tau_label_formats_decimal_with_p():
    assert supra_tau_label(11.75) == "11p75"
    assert supra_tau_label(11.0) == "11"
    assert supra_tau_label(0.5) == "0p5"


def _sample(arm, idx, *, num_masked=4, reuse_fraction=0.0, bootstrap=False):
    return SCGRProbeSample(
        arm=arm,
        sample_idx=idx,
        tokens=np.zeros(4, dtype=np.int64),
        num_masked=num_masked,
        num_reused_from_prev=int(reuse_fraction * num_masked),
        reuse_fraction=reuse_fraction,
        mean_prev_confidence_reused=0.0,
        mean_sample_entropy=0.0,
        state_bootstrap_flag=bootstrap,
    )


def _agg(mean_excess, topm, supra):
    return SCGRRiskAggregates(
        G_mean_excess=mean_excess,
        G_topm_lse=topm,
        supra_masses={11.75: supra},
        head_risk_LME=0.0,
        head_risk_max=0.0,
    )


def test_summarize_probe_refresh_per_arm_reductions_and_old_argmax():
    per_sample = [
        (_sample("fresh", 0), _agg(1.0, 3.0, 0.1)),
        (_sample("fresh", 1), _agg(3.0, 9.0, 0.3)),
        (_sample("self_conditioned", 0, reuse_fraction=0.5), _agg(2.0, 6.0, 0.2)),
    ]
    old_argmax = _agg(0.5, 1.5, 0.05)
    rows = summarize_probe_refresh(
        per_sample=per_sample,
        old_argmax_aggregates=old_argmax,
        supra_tau_values=[11.75],
    )
    by_arm = {r["arm"]: r for r in rows}
    assert set(by_arm) == {"fresh", "self_conditioned"}
    fresh = by_arm["fresh"]
    assert fresh["ensemble_size_effective"] == 2
    assert fresh["B_sc_mean_excess_median"] == pytest.approx(2.0)  # median(1,3)
    assert fresh["G_mean_excess_max"] == pytest.approx(3.0)
    assert fresh["G_topm_lse_max"] == pytest.approx(9.0)
    assert fresh["B_sc_supra_mass_tau_11p75_median"] == pytest.approx(0.2)
    assert fresh["G_mean_excess_std"] == pytest.approx(np.std([1.0, 3.0]))
    # old-argmax columns carried through
    assert fresh["old_argmax_G_mean_excess"] == pytest.approx(0.5)
    assert fresh["old_argmax_G_topm_lse"] == pytest.approx(1.5)
    assert fresh["old_argmax_G_supra_mass_tau_11p75"] == pytest.approx(0.05)

    sc = by_arm["self_conditioned"]
    assert sc["ensemble_size_effective"] == 1
    assert sc["B_sc_mean_excess_median"] == pytest.approx(2.0)
    assert sc["reuse_fraction_mean"] == pytest.approx(0.5)
    assert sc["G_mean_excess_std"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# reduce_residue_excess_over_k (per-residue r_i, PLAN_PLANNER_SC_GR.md Task 1)
# ---------------------------------------------------------------------------


def test_reduce_residue_excess_over_k_medians_fresh_arm():
    from inverse_folding.reference_flow.self_conditioned_gr import (
        reduce_residue_excess_over_k,
        SCGRProbeSample,
        SCGRRiskAggregates,
    )
    import numpy as np

    def mk(arm, idx, excess):
        s = SCGRProbeSample(
            arm=arm,
            sample_idx=idx,
            tokens=np.zeros(1),
            num_masked=0,
            num_reused_from_prev=0,
            reuse_fraction=0.0,
            mean_prev_confidence_reused=0.0,
            mean_sample_entropy=0.0,
            state_bootstrap_flag=False,
        )
        a = SCGRRiskAggregates(
            G_mean_excess=0.0,
            G_topm_lse=0.0,
            supra_masses={},
            head_risk_LME=0.0,
            head_risk_max=0.0,
            residue_excess=np.asarray(excess, dtype=float),
        )
        return (s, a)

    per_sample = [
        mk("fresh", 0, [0.0, 2.0, 4.0]),
        mk("fresh", 1, [0.0, 6.0, 0.0]),
        mk("fresh", 2, [0.0, 4.0, 2.0]),
        mk("self_conditioned", 0, [9.0, 9.0, 9.0]),
    ]
    r_i = reduce_residue_excess_over_k(per_sample, arm="fresh")
    np.testing.assert_allclose(r_i, [0.0, 4.0, 2.0])


def test_reduce_residue_excess_over_k_empty_arm_returns_empty():
    from inverse_folding.reference_flow.self_conditioned_gr import (
        reduce_residue_excess_over_k,
    )

    assert reduce_residue_excess_over_k([], arm="fresh").size == 0
