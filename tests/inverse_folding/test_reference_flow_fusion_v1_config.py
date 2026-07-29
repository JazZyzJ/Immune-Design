"""V1-A entry-config contract tests (PLAN_RF_REFINE_FUSION_V1 §4.1, runbook §1-§2).

The V1-A scope is narrow and the config is where it is enforced: exactly two P1 arms
(``terminal`` / ``preterminal``), the frozen null entry runtime (c1_null, no controller, no
h-map) declared as explicit sentinels, exactly three T0 policies, and arm-conditional fields that
fail closed when a knob does not belong to the arm.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion.v1_config import V1EntryConfig
from tests._v1_fixtures import p1_preterminal_config, p1_terminal_config, t0_config


# ------------------------- happy paths ------------------------- #
def test_valid_configs_construct_for_both_arms_and_t0():
    pre = V1EntryConfig.from_mapping(p1_preterminal_config())
    assert pre.phase == "p1" and pre.entry_arm == "preterminal" and pre.rho_target == 0.85
    term = V1EntryConfig.from_mapping(p1_terminal_config())
    assert term.entry_arm == "terminal" and term.rho_target is None
    t0 = V1EntryConfig.from_mapping(t0_config())
    assert t0.phase == "t0" and t0.k_eval == 2 and t0.q_t0 == 2  # Q_T0 == N (runbook §6.0)


# ------------------------- arm identity (V1-A has two arms) ------------------------- #
@pytest.mark.parametrize("bad_arm", ["A", "B", "C", "Terminal", "pre_terminal", ""])
def test_rejects_arm_label_outside_terminal_preterminal(bad_arm):
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(entry_arm=bad_arm))


def test_rejects_legacy_arm_id_key():
    data = p1_preterminal_config()
    data["arm_id"] = data.pop("entry_arm")  # the OLD A/B/C field name
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(data)


# ------------------------- arm-conditional field law ------------------------- #
@pytest.mark.parametrize("field,value", [
    ("prefix_attempts", 4), ("k_est", 2), ("unique_root_capacity", 3),
    ("completions_per_ranked_root", 1), ("estimator", "mean_global_risk"),
    ("rho_target", 0.85), ("rho_grid", (0.8,)),
])
def test_terminal_arm_rejects_preterminal_fields(field, value):
    # A terminal arm ranks a complete pool by exact Head; root prefixes, K_EST, unique-root
    # capacity, one-completion-per-root and maturity are all pre-terminal-only concepts.
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_terminal_config(**{field: value}))


@pytest.mark.parametrize("field", [
    "prefix_attempts", "k_est", "unique_root_capacity", "completions_per_ranked_root", "estimator",
])
def test_preterminal_arm_requires_allocation_fields(field):
    data = p1_preterminal_config()
    del data[field]
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(data)


def test_minimal_terminal_config_needs_no_allocation_knobs():
    cfg = V1EntryConfig.from_mapping(p1_terminal_config())
    assert cfg.prefix_attempts is None and cfg.k_est is None and cfg.estimator is None


# ------------------------- frozen null entry runtime ------------------------- #
def test_rejects_enabled_controller_and_present_h_map():
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(controller_enabled=True))
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(h_maps_present=True))


@pytest.mark.parametrize("field", ["controller_enabled", "h_maps_present", "backbone_only"])
def test_null_runtime_sentinels_are_required_and_strictly_boolean(field):
    data = p1_preterminal_config()
    del data[field]
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(data)  # missing sentinel is not "assume the safe default"
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(**{field: 0}))  # no truthy coercion


def test_rejects_legacy_h_map_identity_key():
    # h-map steering is deferred out of V1-A; a config carrying an h-map identity is a scope error.
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(h_map="hmap#a"))


# ------------------------- T0 policies (exactly three) ------------------------- #
def test_t0_requires_exactly_the_three_frozen_policies():
    frozen = ("selected_partial", "random_partial", "independent_full")
    assert V1EntryConfig.from_mapping(t0_config()).t0_policies == frozen
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(t0_config(t0_policies=("selected_partial", "random_partial")))
    with pytest.raises(ValueError):  # branch-and-materialize left V1-A scope
        V1EntryConfig.from_mapping(t0_config(t0_policies=frozen + ("branch_and_materialize",)))


@pytest.mark.parametrize("key", ["enable_branch_control", "enable_random_control",
                                 "enable_full_control"])
def test_rejects_legacy_control_toggles(key):
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(t0_config(**{key: True}))


def test_p1_forbids_t0_only_fields_and_vice_versa():
    for field, value in (("k_eval", 2), ("q_t0", 4), ("structure_subsample_seed", 13),
                         ("random_membership_seed", 17),
                         ("t0_policies", ("selected_partial",))):
        with pytest.raises(ValueError):
            V1EntryConfig.from_mapping(p1_preterminal_config(**{field: value}))
    for field in ("k_eval", "q_t0", "structure_subsample_seed", "random_membership_seed",
                  "t0_policies"):
        data = t0_config()
        del data[field]
        with pytest.raises(ValueError):
            V1EntryConfig.from_mapping(data)


def test_t0_requires_the_preterminal_arm():
    # The three T0 policies are all views over one pre-terminal root pool.
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(t0_config(entry_arm="terminal"))


# ------------------------- maturity + capacity + estimator ------------------------- #
def test_rejects_terminal_maturity_value():
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(rho_target=1.0))
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(t0_config(rho_grid=(0.8, 1.0)))


def test_capacity_chain_is_arm_conditional():
    # pre-terminal: B >= unique_root_capacity >= F_cap >= N
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(prefix_attempts=1))
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(unique_root_capacity=1))
    # terminal: only F_cap >= N applies (there is no root pool)
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_terminal_config(initial_refold_attempt_cap=1,
                                                      n_population=2))
    assert V1EntryConfig.from_mapping(
        p1_terminal_config(initial_refold_attempt_cap=4, n_population=2)
    ).initial_refold_attempt_cap == 4


def test_rejects_non_frozen_estimator_and_multi_completion():
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(estimator="min_global_risk"))
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(completions_per_ranked_root=2))


def test_rejects_unknown_key_and_bad_split_role():
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(bogus=1))
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(split_role="bogus"))


def test_rejects_non_integer_scientific_knobs():
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(k_est=1.9))
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(backbone_only="false"))
    with pytest.raises(ValueError):
        V1EntryConfig.from_mapping(p1_preterminal_config(max_walltime_s=float("nan")))
