"""The V2F5A materializer binds calibration to the exact Head and policy spec."""

from __future__ import annotations

import hashlib
import json
import types

import pytest

from inverse_folding.reference_flow.fusion_v2.identity import HeadEvaluatorIdentity
from inverse_folding.reference_flow.fusion_v2.config import load_v2_config
from scripts.calibrate_v2_head_policy import build_calibration_bundle
from scripts.materialize_v2_canary_config import MaterializeError, fill_config
from tests.inverse_folding.test_fusion_v2_config import _mapping


D = "a" * 64


def _case(tmp_path):
    template = _mapping(**{
        "schedule.depth_cap": 1,
        "schedule.points": [{
            "depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
            "n_lookaheads": 4, "band_key": "step40",
        }],
    })
    hotspot = tmp_path / "hotspot.json"
    hotspot.write_text(json.dumps({"delta_new": template["safety"]["delta_new_cumulative"]}))
    spec = tmp_path / "policy.json"
    spec.write_text(json.dumps({"policy_id": "head_directed_capped", "policy_version": "v1"}))
    spec_sha = hashlib.sha256(spec.read_bytes()).hexdigest()
    evaluator = HeadEvaluatorIdentity(
        allele=template["head"]["allele"], score_scale=template["head"]["score_scale"],
        window_k_min=template["head"]["window_k_min"],
        window_k_max=template["head"]["window_k_max"],
        head_config_hash=D, head_checkpoint_digest=D,
    )
    calibration = build_calibration_bundle(
        [{"protein_id": "P1", "sequence_md5": "b" * 32, "abs_repeat_drift": 0.01}],
        evaluator=evaluator, policy_spec=spec,
        max_counterfactual_head_calls_per_cycle=278,
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    args = types.SimpleNamespace(
        code_revision="deadbeef", campaign_id="v2f5a-test", r_step=40,
        hotspot_json=hotspot, policy_calibration_json=calibration_path,
        qualification_max_head_calls=40000,
    )
    frozen = {row["role"]: D for row in template["content"]}
    frozen.update({"head_config": D, "head_checkpoint": D,
                   "projection_policy_spec": spec_sha})
    return template, args, frozen


def test_policy_qualification_materialization_copies_only_a_matching_calibration(tmp_path):
    template, args, frozen = _case(tmp_path)
    config = fill_config(template, args=args, frozen=frozen, runtime={})
    assert config["identity"]["phase"] == "policy_qualification"
    assert config["projection"]["support_policy_id"] == "head_directed_capped"
    assert config["projection"]["head_directed"][
        "max_counterfactual_head_calls_per_cycle"] == 278
    assert config["caps"]["max_head_calls"] == 40000
    assert load_v2_config(config).projection.head_directed is not None


def test_a_calibration_from_another_head_is_refused(tmp_path):
    template, args, frozen = _case(tmp_path)
    frozen["head_checkpoint"] = "f" * 64
    with pytest.raises(MaterializeError, match="different Head instrument"):
        fill_config(template, args=args, frozen=frozen, runtime={})
