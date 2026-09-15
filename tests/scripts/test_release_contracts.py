"""Release defaults and provenance survive removal of the research entry points."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts.refine_rf_designs import build_arg_parser
from scripts.rf_fusion_v2_cohort import ShardInputs, _terminal_stop_label
from scripts.run_rf_fusion_v2 import resolve_dual
from inverse_folding.reference_flow.fusion_v2.errors import V2Error
from inverse_folding.reference_flow.fusion_v2.dual_config import dual_overlay_payload
from inverse_folding.reference_flow.fusion_v2.joint_objective import arm_objective_digest
from tests.inverse_folding.test_fusion_v2_dual_config import overlay
from tests.inverse_folding.test_fusion_v2_dual_arms import ARMS, _objective


def test_public_refinement_defaults_to_head_and_exact_front():
    args = build_arg_parser().parse_args(['--allele', 'A', '--out-dir', 'out'])
    assert args.target_source == 'head'
    assert args.official is True


def test_merge_cli_runs_without_an_inherited_pythonpath():
    import sys

    root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    result = subprocess.run(
        [sys.executable, str(root / 'scripts/merge_refine_shards.py'), '--help'],
        cwd=root, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()


def test_dual_overlay_must_match_declared_content_not_path(tmp_path):
    expected = overlay()
    first = tmp_path / 'declared.json'
    alias = tmp_path / 'alias.json'
    wrong = tmp_path / 'wrong.json'
    first.write_text(json.dumps(dual_overlay_payload(expected)))
    alias.write_text(json.dumps(dual_overlay_payload(expected), indent=2))
    wrong.write_text(json.dumps(dual_overlay_payload(overlay(objective_spec_digest='a' * 64))))
    args = SimpleNamespace(dual_overlay=str(alias), dual_arm='joint')
    inputs = ShardInputs(dual_overlay=str(first))
    assert resolve_dual(args, shard_inputs=inputs)[0].content_digest == expected.content_digest
    args.dual_overlay = str(wrong)
    with pytest.raises(V2Error, match='differs'):
        resolve_dual(args, shard_inputs=inputs)
    with pytest.raises(V2Error, match='dual-overlay'):
        resolve_dual(args, shard_inputs=ShardInputs())


def test_all_arm_values_use_the_same_canonical_identity_constructor():
    identities = set()
    for arm in ARMS:
        objective = _objective(arm)
        value = objective.evaluate(raw_a=-1.0, raw_b=2.0)
        assert value.objective_digest == arm_objective_digest(objective.calibration, arm)
        assert value.objective_digest == objective.objective_digest
        identities.add(value.objective_digest)
    assert len(identities) == 3


def test_terminal_stall_is_not_reported_as_invalid_geometry():
    outcome = SimpleNamespace(stopping_reason=SimpleNamespace(value='invalid_projection'),
        cycles=[SimpleNamespace(cycle=SimpleNamespace(
            policy_evidence=SimpleNamespace(stall_reason='stall_no_better_donor'),
            outcome=SimpleNamespace(value='null_invalid_policy_result')))])
    assert _terminal_stop_label(outcome) == 'stall_no_better_donor'
    outcome.cycles[0].cycle.policy_evidence = None
    assert _terminal_stop_label(outcome) == 'null_invalid_policy_result'
    outcome.stopping_reason = SimpleNamespace(value='depth_cap')
    assert _terminal_stop_label(outcome) == 'depth_cap'
