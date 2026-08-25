"""The assembly preflight must prove a head-directed policy is CONSTRUCTIBLE without a model.

This file exists because the gap it guards survived precisely by being untested: the check could
never build a `head_directed_capped` policy, so the operator's only way to discover an assembly
failure was a real one-prefix GPU run -- which the Dual contract forbids.
"""

from __future__ import annotations

import inspect

import pytest

import scripts.preflight_v2_canary_assembly as preflight
import scripts.rf_fusion_v2_oracles as oracles


def test_the_stub_head_can_be_held_but_not_asked():
    stub = preflight._StubHeadOracle()
    assert hasattr(stub, "score") and callable(stub.score)
    with pytest.raises(preflight.PreflightAssemblyError):
        stub.score([])


def test_the_assembly_check_supplies_every_binding_the_head_directed_policy_requires():
    src = inspect.getsource(preflight.assemble_check)
    for name in ("head_oracle", "incumbent", "safety_reference_score", "evaluator",
                 "window_grid_digest"):
        assert f'"{name}"' in src, name
    # ...through the channel the factory actually reads
    assert "**context" in inspect.getsource(oracles.resolve_support_policy)


def test_the_assembly_check_loads_no_model():
    src = inspect.getsource(preflight.assemble_check)
    for banned in ("build_production_oracles", "load_refold_model", "build_head_scorer",
                   "torch.load"):
        assert banned not in src, banned


def test_the_placeholder_derivation_is_declared_rather_than_implied():
    """The reported policy digest comes from a shaped placeholder, not the real Head.

    Reporting it without saying so would let an operator compare it against a run's real digest and
    conclude the config had changed.
    """
    assert "policy_config_digest_is_placeholder_bound" in inspect.getsource(
        preflight.assemble_check)


# ------------------------------------------------------------------------------------------
# the Dual assembly contract
# ------------------------------------------------------------------------------------------

def _overlay(**over):
    """The signed overlay fixture, WITH window coordinates.

    The base calibration fixture leaves ``window`` as None; the joint policy requires it, because
    the union reopen law compares per-window residuals and the aggregate risk scale does not
    transfer to them.
    """
    import dataclasses

    from tests.inverse_folding.test_fusion_v2_dual_config import overlay
    from tests.inverse_folding.test_fusion_v2_dual_policy import window_pair

    base = overlay(**over)
    if base.calibration.window is not None:
        return base
    return dataclasses.replace(
        base, calibration=dataclasses.replace(base.calibration, window=window_pair()))


def _incumbent_stub(risk=-1.0):
    return type("_Inc", (), {"head_global_risk": risk, "protein_id": "5ZHV_B",
                             "sequence_md5": "a" * 32, "sequence_length": 12})()


def _reference_score():
    over = _overlay()
    evaluator = over.calibration.risk.a.evaluator
    score = preflight._ReferenceHeadScore(
        "5ZHV_B", "K" * 12, allele=evaluator.allele, score_scale=evaluator.score_scale,
        k_min=evaluator.window_k_min, k_max=evaluator.window_k_max)
    score.sequence_md5 = "a" * 32
    return score


def test_the_joint_policy_can_be_assembled_without_a_model():
    """The gap this closes: a three-arm Dual cell passed every preflight and could only discover an
    assembly failure on the GPU -- which is what the assembly check exists to prevent."""
    authority = preflight._dual_authority(
        _overlay(), arm="joint", incumbent=_incumbent_stub(),
        reference_head_score=_reference_score())
    assert authority.evaluator_b == _overlay().head_b_runtime.evaluator


def test_the_assembled_incumbent_joint_value_is_derived_not_declared():
    overlay = _overlay()
    authority = preflight._dual_authority(
        overlay, arm="joint", incumbent=_incumbent_stub(risk=-1.0),
        reference_head_score=_reference_score())
    from inverse_folding.reference_flow.fusion_v2.joint_objective import DualObjective

    expected = DualObjective(overlay.calibration).evaluate(raw_a=-1.0, raw_b=0.0).value
    assert authority.incumbent_joint_value == pytest.approx(expected)


def test_the_stub_role_b_scorer_can_be_held_but_not_asked():
    scorer = preflight._StubCounterfactualScorer()
    assert callable(scorer)
    with pytest.raises(preflight.PreflightAssemblyError):
        scorer("5ZHV_B", ["KKKK"])


def test_a_calibration_with_no_window_coordinates_is_refused():
    import dataclasses

    overlay = _overlay()
    stripped = dataclasses.replace(
        overlay, calibration=dataclasses.replace(overlay.calibration, window=None))
    with pytest.raises(preflight.PreflightAssemblyError, match="window"):
        preflight._dual_authority(stripped, arm="joint", incumbent=_incumbent_stub(),
                                  reference_head_score=_reference_score())


def test_the_dual_context_reaches_the_policy_factory_and_is_verified():
    src = inspect.getsource(preflight.assemble_check)
    assert '"dual"' in src
    assert "holds no Dual authority" in src, (
        "assembling a policy without checking it actually took the authority would pass a cell "
        "that runs A-only under a Dual signature")


def test_the_dual_assembly_check_still_loads_no_model():
    src = inspect.getsource(preflight.assemble_check) + inspect.getsource(preflight._dual_authority)
    for banned in ("build_role_b_head_oracle", "build_head_scorer", "torch.load"):
        assert banned not in src, banned


def test_half_a_dual_specification_is_refused_by_the_cli():
    with pytest.raises(SystemExit, match="together"):
        preflight.main(["--cell", "x.yaml=P", "--dual-arm", "joint"])
    with pytest.raises(SystemExit, match="together"):
        preflight.main(["--cell", "x.yaml=P", "--dual-overlay", "o.json"])


# -- what --dry-run REPORTS must be what the gate ENFORCES --------------------------------------

def test_the_printed_budget_counts_both_heads_for_a_dual_run():
    """The launch gate already multiplied by n_heads; the payload an operator records did not.

    ``run_rf_fusion_v2`` calls ``project_v2_budget(..., n_heads=2, counterfactual_sequences_per_cycle
    =C)`` for ``assert_launch_feasible`` but called ``print_config_payload`` with neither. So the
    gate was right and the evidence printed beside it understated Head spend by exactly one allele
    -- and §4.3's GO criterion compares realized Head calls against that printed projection.
    """
    import inspect

    from scripts import rf_fusion_v2_preflight as pf

    sig = inspect.signature(pf.print_config_payload)
    assert "n_heads" in sig.parameters, "the payload must be able to price a two-Head run"
    assert "counterfactual_sequences_per_cycle" in sig.parameters

    src = inspect.getsource(pf.print_config_payload)
    assert "n_heads=n_heads" in src.replace(" ", "").replace("\n", "") or \
           "n_heads=int(n_heads)" in src.replace(" ", "").replace("\n", ""), \
        "print_config_payload must forward n_heads into project_v2_budget"

    from scripts import run_rf_fusion_v2 as drv
    call = inspect.getsource(drv.main)
    idx = call.index("print_config_payload(")
    assert "n_heads=n_heads" in call[idx:idx + 400], \
        "the driver must pass the resolved n_heads into the payload it prints"
