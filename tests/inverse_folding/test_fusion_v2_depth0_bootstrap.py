"""V2 depth-zero reward bootstrap: no WT reward comparison, then strict I1+ gating.

These tests deliberately reuse the qualified Head-directed policy harness.  They verify wiring and
identity only; fake-oracle success is not evidence that feedback improves real designs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from inverse_folding.reference_flow.fusion_v2 import policy as pol
from inverse_folding.reference_flow.fusion_v2 import reward as rw
from inverse_folding.reference_flow.fusion_v2_runtime.cycle import run_one_cycle
from scripts.rf_fusion_v2_artifacts import feedback_event_rows
from tests.inverse_folding.test_fusion_v2_head_directed_policy import (
    _config_payload,
    _coords,
    _cycle_kwargs,
    _donor,
    _policy,
    _source,
    SOURCE_SEQ,
)


def _bootstrap_policy(*, head=None):
    legacy = _policy(head=head)
    return dataclasses.replace(
        legacy,
        incumbent=None,
        depth0_attribution_reference=legacy.incumbent,
        depth0_incumbent_rule=rw.DEPTH0_BOOTSTRAP_RULE,
        policy_version="v2",
    )


def _bootstrap_cycle(*, policy=None, **over):
    policy = policy or _bootstrap_policy()
    kwargs = _cycle_kwargs(
        support_policy=policy,
        declared_policy=pol.DeclaredPolicy(
            policy_id=pol.HEAD_DIRECTED_CAPPED_POLICY_ID,
            policy_version="v2",
            is_diagnostic=False,
            phase="policy_qualification",
        ),
        **over,
    )
    return run_one_cycle(**kwargs)


class _GloballyShiftedHead:
    """Keep every local difference fixed while making all generated designs worse than WT."""

    def __init__(self, offset: float):
        from tests.inverse_folding.test_fusion_v2_head_directed_policy import _ScriptedHead

        self._delegate = _ScriptedHead()
        self.offset = float(offset)

    def evaluator_identity(self):
        return self._delegate.evaluator_identity()

    def score(self, requests):
        shifted = []
        for row in self._delegate.score(requests):
            risk = float(row.global_risk) + self.offset
            shifted.append(dataclasses.replace(
                row,
                score=dataclasses.replace(row.score, global_risk=risk),
                global_risk=risk,
            ))
        return shifted


def test_v2_depth_zero_has_no_reward_incumbent_and_fails_closed_without_cycle_proof():
    policy = _bootstrap_policy()
    assert policy.incumbent is None

    result = policy.decide(
        source=_source(SOURCE_SEQ), endpoint=_donor(), coordinates=_coords(), runtime=None,
    )
    assert isinstance(result, pol.PolicyRejection)
    assert pol.StallReason.MISSING_DEPTH0_SELECTION_PROOF.value in result.reason
    assert result.decision_evidence.reward_gate_kind == rw.RewardGateKind.DEPTH0_BOOTSTRAP.value
    assert result.decision_evidence.donor_gate is None
    assert result.decision_evidence.depth0_selection_proof is None


def test_cycle_issues_rank0_proof_from_the_real_ordered_admissible_pool():
    outcome = _bootstrap_cycle()
    assert outcome.committed, outcome.detail

    proof = outcome.depth0_selection_proof
    assert isinstance(proof, rw.Depth0SelectionProof)
    assert proof.depth == 0
    assert proof.selected_rank == 0
    assert proof.selected_endpoint_id == outcome.selected_endpoint.endpoint_id
    assert proof.selected_endpoint_content_digest == outcome.selected_endpoint.content_digest
    assert [(row.head_global_risk, row.endpoint_id) for row in proof.ordered_candidates] == sorted(
        (row.head_global_risk, row.endpoint_id) for row in proof.ordered_candidates
    )

    evidence = outcome.policy_evidence
    assert evidence.reward_gate_kind == rw.RewardGateKind.DEPTH0_BOOTSTRAP.value
    assert evidence.donor_gate is None, "D0 must not compare the donor to WT with epsilon"
    assert evidence.incumbent_id is None
    assert evidence.depth0_selection_proof == proof
    assert evidence.attribution_reference_sequence_md5 == (
        _bootstrap_policy().depth0_attribution_reference.sequence_md5
    )


def test_depth_zero_rank0_is_not_required_to_beat_wt():
    """D0 optimizes the admissible generated pool; WT remains attribution/safety only."""
    head = _GloballyShiftedHead(offset=100.0)
    policy = _bootstrap_policy(head=head)
    outcome = _bootstrap_cycle(policy=policy, head_oracle=head)

    assert outcome.committed, outcome.detail
    assert outcome.selected_endpoint.head_global_risk > (
        policy.depth0_attribution_reference.head_global_risk
    )
    assert outcome.policy_evidence.donor_gate is None
    assert outcome.policy_evidence.reward_gate_kind == (
        rw.RewardGateKind.DEPTH0_BOOTSTRAP.value
    )


def test_only_rank0_can_bootstrap_at_depth_zero():
    outcome = _bootstrap_cycle(endpoint_rank=1)
    assert not outcome.committed
    assert outcome.selected_endpoint is not None
    assert outcome.depth0_selection_proof is None
    assert outcome.policy_evidence.stall_reason == (
        pol.StallReason.MISSING_DEPTH0_SELECTION_PROOF.value
    )


def test_committed_depth_zero_donor_becomes_i1_and_future_updates_stay_strict():
    policy = _bootstrap_policy()
    outcome = _bootstrap_cycle()
    assert outcome.committed, outcome.detail

    advanced = policy.advance_lineage_incumbent(
        donor=outcome.selected_endpoint,
        verdict=None,
        depth0_selection_proof=outcome.depth0_selection_proof,
        accepted_at_depth=1,
    )
    assert advanced.incumbent is not None
    assert advanced.incumbent.source_endpoint_id == outcome.selected_endpoint.endpoint_id
    assert advanced.incumbent.accepted_at_depth == 1

    source = _source(SOURCE_SEQ)
    worse = _donor("Y" * len(SOURCE_SEQ), source=source, fork_index=9, fork_seed=99)
    refused = advanced.decide(
        source=source, endpoint=worse, coordinates=_coords(depth=1),
        runtime=pol.PolicyRuntime(selection_proof=outcome.depth0_selection_proof),
    )
    assert isinstance(refused, pol.PolicyRejection)
    assert refused.decision_evidence.reward_gate_kind == (
        rw.RewardGateKind.STRICT_IMPROVEMENT.value
    )
    assert refused.decision_evidence.depth0_selection_proof is None
    assert refused.decision_evidence.donor_gate is not None
    assert refused.decision_evidence.donor_gate.epsilon_r == advanced.calibration.epsilon_r


def test_a_bootstrap_proof_cannot_advance_another_donor():
    policy = _bootstrap_policy()
    outcome = _bootstrap_cycle()
    source = outcome.source
    other = _donor("ACDEFGHIKLMN", source=source, fork_index=99, fork_seed=999)

    with pytest.raises(rw.V2RewardError, match="proof selected"):
        policy.advance_lineage_incumbent(
            donor=other,
            verdict=None,
            depth0_selection_proof=outcome.depth0_selection_proof,
            accepted_at_depth=1,
        )


def test_feedback_artifact_distinguishes_bootstrap_from_a_strict_donor_gate():
    outcome = _bootstrap_cycle()
    row = feedback_event_rows([dict(
        source=outcome.source,
        endpoint=outcome.selected_endpoint,
        projected=outcome.projected,
        propagated=outcome.propagated,
        policy=outcome.policy_identity,
        policy_evidence=outcome.policy_evidence,
        outcome=outcome.outcome.value,
        detail=outcome.detail,
        arm_slot="arm_a",
        treatment_identity="v2",
    )])[0]
    assert row["reward_gate_kind"] == rw.RewardGateKind.DEPTH0_BOOTSTRAP.value
    assert row["incumbent_id"] is None
    assert row["donor_gate_epsilon"] is None
    proof = json.loads(row["depth0_selection_proof_json"])
    assert proof["selected_endpoint_id"] == outcome.selected_endpoint.endpoint_id
    assert row["attribution_reference_sequence_md5"] is not None


def test_bootstrap_depth0_rule_requires_the_v2_policy_spec_version():
    from inverse_folding.reference_flow.fusion_v2 import config as cfg

    payload = _config_payload()
    payload["projection"]["support_policy_version"] = "v2"
    payload["projection"]["head_directed"]["lineage_incumbent_depth0_rule"] = (
        rw.DEPTH0_BOOTSTRAP_RULE
    )
    loaded = cfg.load_v2_config(payload)
    assert loaded.projection.support_policy_version == "v2"

    payload["projection"]["support_policy_version"] = "v1"
    with pytest.raises(cfg.V2ConfigError, match="requires support_policy_version='v2'"):
        cfg.load_v2_config(payload)


def test_bootstrap_has_an_independent_frozen_policy_spec_identity():
    config_dir = (
        Path(__file__).parents[2] / "inverse_folding" / "reference_flow" / "configs"
    )
    v1_path = config_dir / "v2_head_directed_capped_policy_v1.json"
    v2_path = config_dir / "v2_head_directed_capped_policy_v2.json"
    v1, v2 = json.loads(v1_path.read_text()), json.loads(v2_path.read_text())

    assert v1["policy_id"] == v2["policy_id"] == pol.HEAD_DIRECTED_CAPPED_POLICY_ID
    assert (v1["policy_version"], v2["policy_version"]) == ("v1", "v2")
    assert v2["depth0_reward_bootstrap"]["reward_incumbent"] is None
    assert v2["depth0_reward_bootstrap"]["epsilon_R"] is None
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() != hashlib.sha256(
        v2_path.read_bytes()
    ).hexdigest()


def test_new_bootstrap_fields_do_not_rewrite_the_published_v1_policy_identity():
    from inverse_folding.reference_flow.fusion_v2.identity import canonical_digest

    legacy = _policy()
    expected = canonical_digest({
        "policy_id": pol.HEAD_DIRECTED_CAPPED_POLICY_ID,
        "policy_version": legacy.policy_version,
        "write_window_rule": pol.WRITE_WINDOW_RULE,
        "reopen_count_law": pol.REOPEN_COUNT_LAW,
        "reopen_priority_law": pol.REOPEN_PRIORITY_LAW,
        "policy_spec_digest": legacy.policy_spec_digest,
        "stratum_key": legacy.stratum_key,
        "band_calibration_id": legacy.band_table.provenance.calibration_id,
        "band_calibration_digest": legacy.band_table.provenance.calibration_content_digest,
        "calibration": legacy.calibration.canonical_payload(),
        "incumbent": legacy.incumbent.canonical_payload(),
        "incumbent_update_law": legacy.incumbent_update_law,
        "head_identity_digest": legacy.evaluator.digest(),
        "window_grid_digest": legacy.window_grid_digest,
    })
    assert legacy.identity().policy_config_digest == expected
    assert _bootstrap_policy().identity().policy_config_digest != expected


def test_policy_v2_cannot_be_constructed_with_the_legacy_wt_reward_incumbent():
    with pytest.raises(pol.V2PolicyError, match="reserved for best_admissible_depth0"):
        dataclasses.replace(_policy(), policy_version="v2")
