"""V1F4 allocation-layer tests (PLAN_RF_REFINE_FUSION_V1 §2.8-2.11, §3).

Head firewall + identity-bound records (reorder/missing/extra fail-fast), exact-K_EST
arithmetic-mean value over one est set with distinct replicates, value-beam vs branch-and-
materialize, typed final facade (est/eval can never enter, actual final seed preserved),
common K_EVAL membership views, four-policy Q_T0 manifest, authorization, and cost arithmetic.
"""

from __future__ import annotations

import math

import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_alloc import (
    Continuation,
    FinalMaterialization,
    HeadInputError,
    HeadRecord,
    RootValue,
    ScoredContinuation,
    TerminalCandidate,
    preterminal_entry_dfe,
    authorize_final_materialization,
    build_preterminal_facade,
    build_common_eval_table,
    build_t0_structure_subset,
    build_terminal_facade,
    t0_reserved_structure_requests,
    eval_membership_view,
    evaluate_continuations,
    matched_full_trajectory_allocation,
    reserved_p1_preterminal_dfe,
    reserved_t0_dfe,
    root_value,
    select_random_membership,
    select_value_beam,
    validate_complete_aa20,
)


def _cont(root, cid, seq="ACDE", set_tag="est", replicate=0, seed=1):
    return Continuation(
        continuation_id=cid, root_equivalence_hash=root, set_tag=set_tag, seed=seed,
        sequence=seq, replicate_index=replicate,
    )


def _scored(root, cid, risk, seq="ACDE", set_tag="est", replicate=0):
    return ScoredContinuation(
        continuation=_cont(root, cid, seq, set_tag, replicate), global_risk=risk,
        sequence_md5=sequence_md5(seq),
    )


class SpyHead:
    """Fake Head returning identity-bound records; asserts it never sees a non-canonical residue."""

    def __init__(self, risk_map):
        self.risk_map = risk_map
        self.seen: list[str] = []

    def __call__(self, sequences):
        self.seen.extend(sequences)
        for seq in sequences:
            assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY"), f"Head saw invalid input {seq!r}"
        return [HeadRecord(sequence_md5(seq), self.risk_map[seq]) for seq in sequences]


# ------------------------- Head firewall + identity binding (§2.8) ------------------------- #
@pytest.mark.parametrize("bad", ["ACD#", "ACDX", "AC-E", "acde", "", "ACDEF"])
def test_validate_complete_aa20_rejects_non_canonical_or_wrong_length(bad):
    with pytest.raises(HeadInputError):
        validate_complete_aa20(bad, expected_length=4)


def test_evaluate_never_scores_invalid_input():
    spy = SpyHead({})
    with pytest.raises(HeadInputError):
        evaluate_continuations(spy, [_cont("r1", "c1", seq="AC#E")], expected_length=4)
    assert spy.seen == []


def test_evaluate_binds_by_md5_and_is_reorder_safe():
    risks = {"ACDE": 0.3, "FGHI": 0.7}

    class ReversingHead:
        def __call__(self, sequences):
            recs = [HeadRecord(sequence_md5(s), risks[s]) for s in sequences]
            return list(reversed(recs))  # returns results in the WRONG order

    conts = [_cont("r1", "c1", "ACDE"), _cont("r1", "c2", "ACDE"), _cont("r2", "c3", "FGHI")]
    scored = evaluate_continuations(ReversingHead(), conts, expected_length=4)
    # bound by sequence_md5, so reorder does not misassign risk; multiplicity preserved
    assert [(s.continuation.continuation_id, s.global_risk) for s in scored] == [
        ("c1", 0.3), ("c2", 0.3), ("c3", 0.7)
    ]


def test_evaluate_rejects_wrong_missing_and_nonfinite_records():
    class WrongMd5Head:
        def __call__(self, sequences):
            return [HeadRecord("deadbeef", 0.1) for _ in sequences]

    with pytest.raises(HeadInputError):
        evaluate_continuations(WrongMd5Head(), [_cont("r1", "c1", "ACDE")], expected_length=4)
    with pytest.raises(HeadInputError):
        evaluate_continuations(SpyHead({"ACDE": math.nan}), [_cont("r1", "c1")], expected_length=4)


# ------------------------- root value + est isolation (§2.9) ------------------------- #
def test_root_value_is_exact_k_est_mean_and_requires_full_count():
    est = [_scored("r1", "e1", 0.2, replicate=0), _scored("r1", "e2", 0.6, replicate=1)]
    assert root_value(est, k_est=2) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        root_value(est, k_est=3)


def test_root_value_rejects_eval_mixed_root_and_duplicate_replicate():
    with pytest.raises(ValueError):  # eval, not est
        root_value([_scored("r1", "e1", 0.2, set_tag="eval", replicate=0)], k_est=1)
    with pytest.raises(ValueError):  # mixed root
        root_value(
            [_scored("r1", "e1", 0.2, replicate=0), _scored("r2", "e2", 0.3, replicate=1)],
            k_est=2,
        )
    with pytest.raises(ValueError):  # duplicate replicate
        root_value(
            [_scored("r1", "e1", 0.2, replicate=0), _scored("r1", "e2", 0.3, replicate=0)],
            k_est=2,
        )


def test_value_beam_picks_best_mean_not_best_endpoint():
    r1 = [_scored("r1", "e1", 0.1, replicate=0), _scored("r1", "e2", 0.9, replicate=1)]  # mean .5
    r2 = [_scored("r2", "e3", 0.4, replicate=0), _scored("r2", "e4", 0.4, replicate=1)]  # mean .4
    rv = [RootValue("r1", root_value(r1, 2), 2), RootValue("r2", root_value(r2, 2), 2)]
    # the beam follows the ROOT's mean value; r1 owns the single best endpoint (0.1) and still loses.
    assert select_value_beam(rv, n=1) == ("r2",)


def test_branch_and_materialize_is_out_of_scope_in_v1a():
    # Branch-and-materialize (reusing the best estimator ENDPOINT as a parent) was dropped from
    # V1-A: it is neither a Terminal arm nor a T0 policy, and leaving it importable invites a
    # fourth, unbudgeted policy back into the comparison.
    import inverse_folding.reference_flow.fusion.v1_alloc as alloc

    assert not hasattr(alloc, "branch_and_materialize")


def test_value_beam_is_deterministic_and_distinct():
    rv = [RootValue("hb", 0.4, 2), RootValue("ha", 0.4, 2), RootValue("hc", 0.9, 2)]
    assert select_value_beam(rv, n=2) == ("ha", "hb")


def test_value_beam_rejects_duplicate_root_hash():
    # A duplicated root_equivalence_hash would yield two beam slots for one basin (masquerading
    # as independent roots); the beam must reject it rather than return ("r", "r").
    with pytest.raises(ValueError):
        select_value_beam([RootValue("r", 0.1, 2), RootValue("r", 0.2, 2)], n=2)


# ------------------------- facade (§2.11) ------------------------- #
def test_preterminal_facade_follows_root_rank_uses_final_seed_rejects_est():
    ranked = ("r_lo", "r_hi")
    mats = {
        "r_lo": FinalMaterialization("r_lo", _cont("r_lo", "f_lo", "ACDE", "final", seed=555), "src_lo"),
        "r_hi": FinalMaterialization("r_hi", _cont("r_hi", "f_hi", "FGHI", "final", seed=777), "src_hi"),
    }
    rows = build_preterminal_facade(ranked, mats)
    assert [r.root_equivalence_hash for r in rows] == ["r_lo", "r_hi"]  # root rank, not Head
    assert [r.design_idx for r in rows] == [0, 1]
    assert [r.seed for r in rows] == [555, 777]  # actual final seed, not re-derived
    # an est endpoint can never enter the P1 facade
    bad = {"r_lo": FinalMaterialization("r_lo", _cont("r_lo", "e1", "ACDE", "est"), "src")}
    with pytest.raises(ValueError):
        build_preterminal_facade(("r_lo",), bad)


def test_terminal_facade_ranks_by_head_then_md5_then_source():
    pool = [
        TerminalCandidate("ACDE", "m2", 0.5, "s2"),
        TerminalCandidate("FGHI", "m1", 0.2, "s1"),
        TerminalCandidate("KLMN", "m3", 0.5, "s0"),
    ]
    rows = build_terminal_facade(pool, seed_fn=lambda s: 3, facade_cap=3)
    assert [r.source_id for r in rows] == ["s1", "s2", "s0"]


def test_terminal_facade_submits_only_the_top_f_cap(_=None):
    # §3.2.1: Terminal ranks its M_T complete trajectories by exact terminal Head and submits ONLY
    # the top F_cap to initial refold admission, so both arms attempt at most F_cap initial refolds.
    pool = [TerminalCandidate(f"AC{c}E", f"m{i}", 0.1 * i, f"s{i}")
            for i, c in enumerate("DEFGH")]
    rows = build_terminal_facade(pool, seed_fn=lambda s: 3, facade_cap=2)
    assert [r.source_id for r in rows] == ["s0", "s1"]
    assert [r.design_idx for r in rows] == [0, 1]


def test_terminal_facade_hard_fails_when_the_pool_is_smaller_than_the_cap():
    # M_T < F_cap is a DFE accounting / off-by-one implementation error (§3.2.1), never a reason
    # to quietly shrink the facade: the two arms would then attempt different refold counts.
    pool = [TerminalCandidate("ACDE", "m0", 0.1, "s0")]
    with pytest.raises(ValueError, match="facade_cap"):
        build_terminal_facade(pool, seed_fn=lambda s: 3, facade_cap=2)


def test_terminal_facade_rejects_a_duplicated_source_id():
    # Two rows claiming one source would give the same trajectory two admission slots.
    pool = [TerminalCandidate("ACDE", "m0", 0.1, "s0"), TerminalCandidate("FGHI", "m1", 0.2, "s0")]
    with pytest.raises(ValueError, match="source_id"):
        build_terminal_facade(pool, seed_fn=lambda s: 3, facade_cap=2)


def test_terminal_facade_rejects_a_non_finite_head_score():
    pool = [TerminalCandidate("ACDE", "m0", float("nan"), "s0"),
            TerminalCandidate("FGHI", "m1", 0.2, "s1")]
    with pytest.raises(ValueError, match="finite"):
        build_terminal_facade(pool, seed_fn=lambda s: 3, facade_cap=2)


def test_random_membership_is_deterministic_and_head_independent():
    eligible = ["ha", "hb", "hc", "hd"]
    a = select_random_membership(eligible, n=2, membership_seed=99)
    assert a == select_random_membership(eligible, n=2, membership_seed=99) and len(set(a)) == 2
    assert set(a) <= set(eligible)


def test_random_membership_uses_the_shared_seed_context_stream():
    """The random view differs from the selected view ONLY in which roots it holds. Its membership
    seed must come from ``SeedContext.random_membership_seed()`` -- an ad-hoc per-index stream would
    let a policy label open its own draw and break the shared-pool law (§2.9-2.10)."""
    from inverse_folding.reference_flow.fusion.v1_seeds import SeedContext

    ctx = SeedContext(seed_schema="v1seed-1", campaign_id="c", phase="t0", split_role="t0_dev",
                      master_seed=7, entry_arm="preterminal", protein_id="P1", rho_id="rho0.850")
    eligible = ["ha", "hb", "hc", "hd"]
    chosen = select_random_membership(eligible, n=2, membership_seed=ctx.random_membership_seed())
    assert chosen == select_random_membership(
        eligible, n=2, membership_seed=ctx.random_membership_seed()
    )
    # a DIFFERENT protein's context must not reproduce the same membership
    other = SeedContext(seed_schema="v1seed-1", campaign_id="c", phase="t0", split_role="t0_dev",
                        master_seed=7, entry_arm="preterminal", protein_id="P2", rho_id="rho0.850")
    assert ctx.random_membership_seed() != other.random_membership_seed()


# ------------------------- common eval / Q_T0 / authorization (§2.9-2.10) ------------------------- #
def test_common_eval_table_and_membership_views_share_rows():
    e1 = [_scored("r1", "v1", 0.2, set_tag="eval", replicate=0),
          _scored("r1", "v2", 0.3, set_tag="eval", replicate=1)]
    e2 = [_scored("r2", "v3", 0.5, set_tag="eval", replicate=0),
          _scored("r2", "v4", 0.6, set_tag="eval", replicate=1)]
    table = build_common_eval_table({"r1": e1, "r2": e2}, k_eval=2)
    selected = eval_membership_view(table, ["r1"])
    random = eval_membership_view(table, ["r1"])
    assert selected["r1"] is table["r1"] and random["r1"] is table["r1"]  # same rows, not a new draw
    with pytest.raises(ValueError):
        build_common_eval_table({"r1": e1[:1]}, k_eval=2)  # wrong count
    with pytest.raises(ValueError):
        eval_membership_view(table, ["r_absent"])


def test_t0_structure_subset_three_policies_exact_q_or_fail_closed():
    eps = lambda n: [f"e{i}" for i in range(n)]  # noqa: E731
    by_policy = {"selected_partial": eps(5), "random_partial": eps(5), "independent_full": eps(5)}
    subset = build_t0_structure_subset(by_policy, q_t0=3, subsample_seed=7, content_id=lambda e: e)
    assert set(subset) == {"selected_partial", "random_partial", "independent_full"}
    assert all(len(v) == 3 for v in subset.values())
    with pytest.raises(ValueError):  # a policy short of Q_T0 fails closed
        build_t0_structure_subset({**by_policy, "selected_partial": eps(2)}, q_t0=3, subsample_seed=7, content_id=lambda e: e)
    with pytest.raises(ValueError):  # not exactly the three V1-A policies
        build_t0_structure_subset({**by_policy, "branch": eps(5)}, q_t0=3, subsample_seed=7, content_id=lambda e: e)


def test_t0_structure_budget_is_three_q_not_four():
    # V1-A dropped branch-and-materialize, so T0 reserves structure for THREE policy views. A 4*Q
    # reservation would silently price in a policy that is never run.
    assert t0_reserved_structure_requests(q_t0=12) == 36
    with pytest.raises(ValueError):
        t0_reserved_structure_requests(q_t0=0)


def test_authorize_only_selected_root_via_fresh_final():
    selected = ["r1", "r2"]
    assert authorize_final_materialization(selected, _cont("r1", "f1", set_tag="final"))
    with pytest.raises(ValueError):
        authorize_final_materialization(selected, _cont("r1", "e1", set_tag="est"))  # not final
    with pytest.raises(ValueError):
        authorize_final_materialization(selected, _cont("r3", "f3", set_tag="final"))  # not selected


# ------------------------- cost arithmetic (§3.1-3.2) ------------------------- #
def test_entry_and_reserved_dfe_arithmetic():
    prefix = [3, 3, 5]
    tail = [7, 5]
    assert preterminal_entry_dfe(prefix, k_est=2, tail_dfe_by_root=tail, final_tail_dfe=[7]) == (
        11 + 2 * 12 + 7
    )
    # r_max is derived internally as max(tail) = 7 (no redundant caller parameter).
    assert reserved_p1_preterminal_dfe(prefix, k_est=2, tail_dfe_by_root=tail, f_cap=6) == 11 + 2 * 12 + 6 * 7
    assert reserved_t0_dfe(prefix, k_est=2, k_eval=3, tail_dfe_by_root=tail) == 11 + (2 + 3) * 12


def test_matched_full_trajectory_allocation():
    assert matched_full_trajectory_allocation(reserved_dfe=100, s_steps=30) == (3, 10)
