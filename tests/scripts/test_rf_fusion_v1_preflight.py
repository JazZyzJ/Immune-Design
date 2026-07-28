"""V1F6 model-free preflight / print-config (PLAN_RF_REFINE_FUSION_V1 §3.4, §4.1-4.2, V1F6).

--print-config and --dry-run must resolve every method value, digest, budget term, and seed stream
WITHOUT running any model. The worst-case budget projection (§3.4) uses the frozen entry knobs and
the terminal v0 Fusion params (r_parent, n_rounds, S); nothing is invented here.
"""

from __future__ import annotations

import pytest

from scripts.rf_fusion_v1_preflight import (
    TerminalFusionParams,
    assert_launch_feasible,
    entry_config_digest,
    project_budget,
    render_print_config,
    resolve_entry_config,
    seed_stream_summary,
)
from tests._v1_fixtures import p1_preterminal_config, p1_terminal_config, t0_config


def _terminal():
    return TerminalFusionParams(s_steps=10, r_parent=5, n_rounds=6)


def test_resolve_entry_config_roundtrips_valid_p1():
    cfg = resolve_entry_config(p1_preterminal_config())
    assert cfg.phase == "p1" and cfg.entry_arm == "preterminal" and cfg.rho_target == 0.85


def test_resolve_entry_config_roundtrips_valid_p1_terminal():
    # the terminal arm carries no root-prefix / maturity knobs at all (V1-A arm-conditional law).
    cfg = resolve_entry_config(p1_terminal_config())
    assert cfg.phase == "p1" and cfg.entry_arm == "terminal"
    assert cfg.rho_target is None and cfg.prefix_attempts is None and cfg.k_est is None


def test_project_budget_worst_case_terms():
    cfg = resolve_entry_config(p1_preterminal_config())
    b = project_budget(cfg, n_proteins=3, terminal=_terminal())
    assert b.dfe_prefix == 4 * 10          # B * S
    # Estimator work is allocated over ALL of U, and worst-case |U| == B, so the RESERVATION is
    # sized by B -- NOT by unique_root_capacity, which is only the minimum coverage gate (§2.3).
    assert b.dfe_est == 2 * 4 * 10         # K_EST * B * S
    assert b.dfe_final == 2 * 10           # F_cap * S
    assert b.reserved_dfe_per_protein == 40 + 80 + 20
    assert b.reserved_dfe_total == 140 * 3
    assert b.reserved_refold_per_protein == 2 + 2 * 5 * 6  # F_cap + N * R_parent * n_rounds (§3.4)
    assert b.reserved_refold_total == (2 + 60) * 3
    assert b.dfe_within_cap and b.refold_within_cap


def test_project_budget_flags_cap_exceeded():
    cfg = resolve_entry_config(p1_preterminal_config(max_dfe=10, max_refolds=5))  # tiny caps
    b = project_budget(cfg, n_proteins=1, terminal=_terminal())
    assert not b.dfe_within_cap
    assert not b.refold_within_cap


def test_seed_stream_summary_counts_namespaces():
    cfg = resolve_entry_config(p1_preterminal_config())
    s = seed_stream_summary(cfg)
    assert s["root"] == 4 and s["est"] == 2 * 3 and s["final"] == 2
    assert "eval" not in s              # p1 has no K_EVAL
    # V1-A deleted the random/branch/full control switches; a P1 inventory carries exactly the
    # entry namespaces and nothing else (no control namespace may reappear silently).
    assert set(s) == {"master_seed", "root", "est", "final"}


def test_render_print_config_resolves_without_models():
    cfg = resolve_entry_config(p1_preterminal_config())
    out = render_print_config(cfg, n_proteins=2, terminal=_terminal(), command="run ...")
    assert out["nmp_absent"] is True
    assert out["config_digest"] == entry_config_digest(cfg)  # stable digest
    assert out["entry_arm"] == "preterminal"
    assert out["method_values"]["estimator"] == "mean_global_risk"
    assert out["method_values"]["rho_target"] == 0.85
    # the frozen null entry runtime: backbone-only entry, and no h-map identity to print at all
    # (the free-form h_map field left V1-A scope, so it must not resurface in the payload).
    assert out["method_values"]["backbone_only"] is True
    assert "h_map" not in out["method_values"]
    assert out["budget"]["reserved_dfe_total"] == 140 * 2
    assert out["seed_inventory"]["root"] == cfg.prefix_attempts
    assert out["terminal_fusion"]["r_parent"] == 5


def test_t0_budget_is_projected_not_refused():
    # It used to fail fast here because a T0 branch that silently under-projected would be worse
    # than no branch. The branch now exists, so refusing would be the wrong failure.
    cfg = resolve_entry_config(t0_config())
    b = project_budget(cfg, n_proteins=1, terminal=_terminal())
    assert b.dfe_eval > 0 and b.t0_structure_requests is not None


def test_t0_seed_inventory_excludes_final_includes_eval():
    cfg = resolve_entry_config(t0_config(k_eval=7))
    s = seed_stream_summary(cfg)
    assert "final" not in s  # T0 has no fresh final materialization
    assert s["eval"] == 7 * cfg.unique_root_capacity


def test_terminal_params_reject_non_positive():
    with pytest.raises(ValueError):
        TerminalFusionParams(s_steps=0, r_parent=5, n_rounds=6)
    with pytest.raises(ValueError):
        TerminalFusionParams(s_steps=10, r_parent=-1, n_rounds=6)
    with pytest.raises(ValueError):
        TerminalFusionParams(s_steps=10, r_parent=5, n_rounds=6, seconds_per_refold=0)


def test_project_budget_rejects_non_positive_n_proteins():
    cfg = resolve_entry_config(p1_preterminal_config())
    with pytest.raises(ValueError):
        project_budget(cfg, n_proteins=0, terminal=_terminal())


def test_assert_launch_feasible_is_fail_closed_over_cap():
    cfg = resolve_entry_config(p1_preterminal_config(max_dfe=10, max_refolds=5))
    t = TerminalFusionParams(s_steps=10, r_parent=5, n_rounds=6, seconds_per_refold=2.9)
    with pytest.raises(ValueError):
        assert_launch_feasible(project_budget(cfg, n_proteins=1, terminal=t))


def test_walltime_is_predicted_and_gated():
    cfg = resolve_entry_config(p1_preterminal_config(max_walltime_s=1.0))
    t = TerminalFusionParams(s_steps=10, r_parent=5, n_rounds=6, seconds_per_refold=2.9,
                             seconds_per_dfe=0.01)
    b = project_budget(cfg, n_proteins=1, terminal=t)
    assert b.predicted_walltime_s is not None and b.predicted_walltime_s > 1.0
    assert b.walltime_within_cap is False
    with pytest.raises(ValueError):
        assert_launch_feasible(b)


def test_launch_gate_fails_closed_when_walltime_unverifiable():
    cfg = resolve_entry_config(p1_preterminal_config())
    b = project_budget(cfg, n_proteins=1, terminal=_terminal())  # no unit costs -> unpredictable
    assert b.walltime_within_cap is None
    with pytest.raises(ValueError):
        assert_launch_feasible(b)  # never passes an unverifiable budget


def test_print_config_digest_matches_cohort_digest():
    # the entry config digest MUST be the single source shared with the cohort run_sig, else a
    # print-config digest would not match the digest a shard binds its resume identity to.
    from scripts.rf_fusion_v1_cohort import _config_digest

    cfg = resolve_entry_config(p1_preterminal_config())
    assert entry_config_digest(cfg) == _config_digest(cfg)


def test_missing_dfe_unit_cost_makes_walltime_unverifiable():
    # A reserved DFE term with no measured per-DFE cost must NOT be silently priced at zero:
    # that would let an unverifiable budget pass the launch gate.
    cfg = resolve_entry_config(p1_preterminal_config())
    t = TerminalFusionParams(s_steps=10, r_parent=5, n_rounds=6, seconds_per_refold=2.9)
    b = project_budget(cfg, n_proteins=1, terminal=t)
    assert b.reserved_dfe_total > 0
    assert b.walltime_within_cap is None      # unverifiable, not "cheap"
    assert b.predicted_walltime_s is None
    with pytest.raises(ValueError):
        assert_launch_feasible(b)


# --------------------------------------------------------------------------- #
# Terminal arm budget (§3.2.1): M_T is DERIVED from the persisted pre-terminal
# reservation, so the two arms are compute-matched by construction
# --------------------------------------------------------------------------- #
def test_terminal_trajectory_count_is_derived_from_the_persisted_reservation():
    """The Terminal arm may not choose its own budget: `M_T = floor(C_reserved / S)` where
    C_reserved is the pre-terminal arm's reserved per-protein DFE. Anything else makes "matched
    compute" -- the entire claim of the comparison -- unverifiable."""
    from scripts.rf_fusion_v1_preflight import terminal_trajectory_count

    # C_reserved = 1000, S = 10 -> 100 complete trajectories
    assert terminal_trajectory_count(reserved_dfe_per_protein=1000, s_steps=10, facade_cap=4) == 100
    # exact division is not required; the remainder is simply not spent
    assert terminal_trajectory_count(reserved_dfe_per_protein=1005, s_steps=10, facade_cap=4) == 100


def test_terminal_trajectory_count_hard_fails_below_the_facade_cap():
    """M_T < F_cap is a DFE accounting / off-by-one implementation error (your §3.2.1 ruling), not
    a reason to shrink the Terminal facade: shrinking would give the two arms different initial
    refold budgets, which is exactly the confound the design removes."""
    from scripts.rf_fusion_v1_preflight import terminal_trajectory_count

    with pytest.raises(ValueError, match="M_T"):
        terminal_trajectory_count(reserved_dfe_per_protein=30, s_steps=10, facade_cap=4)  # M_T=3


def test_terminal_budget_projection_uses_the_reservation_not_the_config():
    cfg = resolve_entry_config(p1_terminal_config(n_population=2, initial_refold_attempt_cap=4))
    t = TerminalFusionParams(s_steps=10, r_parent=5, n_rounds=6,
                             seconds_per_refold=0.1, seconds_per_dfe=0.001)
    budget = project_budget(cfg, n_proteins=2, terminal=t, reserved_dfe_per_protein=1000)
    assert budget.terminal_trajectories == 100
    # the Terminal arm spends exactly the reserved DFE it was handed -- never more.
    assert budget.reserved_dfe_per_protein <= 1000
    assert budget.reserved_dfe_total == budget.reserved_dfe_per_protein * 2
    # its refold reservation is the SAME F_cap + N*R_parent*n_rounds shape as the other arm.
    assert budget.reserved_refold_per_protein == 4 + 2 * 5 * 6


def test_terminal_budget_refuses_to_run_without_the_reservation():
    # Without the persisted pre-terminal reservation there is no matched budget to derive from.
    cfg = resolve_entry_config(p1_terminal_config())
    with pytest.raises(ValueError, match="reserved"):
        project_budget(cfg, n_proteins=1, terminal=_terminal())


def test_preterminal_budget_rejects_a_reservation_it_should_be_producing():
    # The pre-terminal arm COMPUTES the reservation; being handed one means the caller has the
    # dependency backwards and the two arms could silently diverge.
    cfg = resolve_entry_config(p1_preterminal_config())
    with pytest.raises(ValueError):
        project_budget(cfg, n_proteins=1, terminal=_terminal(), reserved_dfe_per_protein=1000)


# --------------------------------------------------------------------------- #
# T0 budget (§3.2): three policy views, no final materialization
# --------------------------------------------------------------------------- #
def test_t0_budget_has_no_final_term_and_reserves_three_q_of_structure():
    """T0 runs no `final` materialization at all -- its parents are never built. Its structure
    spend is the Q_T0 subsample of each of the THREE policy views; a 4*Q_T0 reservation would
    price in the dropped branch policy and over-reserve every T0 launch."""
    cfg = resolve_entry_config(t0_config(prefix_attempts=4, k_est=2, k_eval=3,
                                         unique_root_capacity=3, q_t0=11, rho_grid=(0.85,),
                                         n_population=2, initial_refold_attempt_cap=2))
    t = TerminalFusionParams(s_steps=10, r_parent=5, n_rounds=6,
                             seconds_per_refold=0.1, seconds_per_dfe=0.001)
    b = project_budget(cfg, n_proteins=1, terminal=t)
    assert b.dfe_final == 0                       # T0 materializes no parent
    # prefix + (K_EST + K_EVAL) * worst-case unique-root tails, sized by B (every attempt may
    # yield a distinct root) exactly like the pre-terminal reservation
    assert b.dfe_prefix == 4 * 10
    assert b.dfe_est == 2 * 4 * 10
    assert b.dfe_eval == 3 * 4 * 10
    # plus the compute-matched independent-full control, which is real spend the gate must see
    assert b.dfe_full_control == b.dfe_prefix + b.dfe_est + b.dfe_eval
    assert b.reserved_dfe_per_protein == (
        b.dfe_prefix + b.dfe_est + b.dfe_eval + b.dfe_full_control
    )
    assert b.reserved_refold_per_protein == 33     # 3 * Q_T0, NOT 4 * Q_T0
    assert b.t0_structure_requests == 33


def test_t0_budget_terms_are_reported_individually_not_only_as_a_total():
    # §3.4: a single total hides which term blew the cap, and the two policy families spend on
    # different things.
    cfg = resolve_entry_config(t0_config(q_t0=4, n_population=2, initial_refold_attempt_cap=2))
    t = TerminalFusionParams(s_steps=10, r_parent=5, n_rounds=6,
                             seconds_per_refold=0.1, seconds_per_dfe=0.001)
    payload = render_print_config(cfg, n_proteins=1, terminal=t)
    for term in ("dfe_prefix", "dfe_est", "dfe_eval", "dfe_final", "t0_structure_requests"):
        assert term in payload["budget"], term
