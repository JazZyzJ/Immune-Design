"""V2F5A frozen-Head repeatability calibration is an auditable instrument bound."""

from __future__ import annotations

import json
import types

import pytest

from inverse_folding.reference_flow.fusion_v2.identity import HeadEvaluatorIdentity
from inverse_folding.reference_flow.fusion.state import sequence_md5
from scripts.calibrate_v2_head_policy import (
    build_calibration_bundle,
    score_repeatability,
)


def _identity():
    return HeadEvaluatorIdentity(
        allele="DRB1_0701", score_scale="raw_logit", window_k_min=13, window_k_max=25,
        head_config_hash="c" * 64, head_checkpoint_digest="d" * 64,
    )


def _policy_spec(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({
        "policy_id": "head_directed_capped", "policy_version": "v1",
    }))
    return path


def test_difference_floor_is_twice_the_largest_same_sequence_repeat_drift(tmp_path):
    rows = [
        {"protein_id": "P1", "sequence_md5": "a" * 32, "abs_repeat_drift": 0.01},
        {"protein_id": "P2", "sequence_md5": "b" * 32, "abs_repeat_drift": 0.03},
    ]
    payload = build_calibration_bundle(
        rows, evaluator=_identity(), policy_spec=_policy_spec(tmp_path),
        max_counterfactual_head_calls_per_cycle=278,
    )
    assert payload["max_abs_repeat_drift"] == pytest.approx(0.03)
    assert payload["difference_noise_bound"] == pytest.approx(0.06)
    block = payload["head_directed"]
    assert block["donor_improvement_epsilon"]["value"] == pytest.approx(0.06)
    assert block["local_contribution_tolerance"]["value"] == pytest.approx(0.06)
    assert block["write_cap_editable_fraction"]["value"] == pytest.approx(0.05)
    assert block["max_counterfactual_head_calls_per_cycle"] == 278
    assert len(payload["policy_spec_sha256"]) == 64


def test_repeatability_requires_the_stored_and_live_head_identity_to_match():
    identity = _identity()

    class FakeHead:
        def evaluator_identity(self):
            return identity

        def score(self, requests):
            return [types.SimpleNamespace(sequence_md5=request.sequence_md5, global_risk=-1.0)
                    for request in requests]

    records = [{
        "protein_id": "P1", "sequence": "AAAA", "sequence_md5": sequence_md5("AAAA"),
        "stored_head_global_risk": -1.1,
        "head_score_json": json.dumps({"allele": identity.allele,
                                       "score_scale": identity.score_scale}),
        "bundle": "/bundle",
        "manifest_head_config_hash": identity.head_config_hash,
        "manifest_head_checkpoint_digest": identity.head_checkpoint_digest,
    }]
    rows, observed = score_repeatability(
        records, head_oracle=FakeHead(), min_observations_per_protein=1)
    assert observed == identity
    assert rows[0]["abs_repeat_drift"] == pytest.approx(0.1)
