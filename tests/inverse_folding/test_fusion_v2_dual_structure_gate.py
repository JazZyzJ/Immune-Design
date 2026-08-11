"""Typed semantics for the exploratory dual-scTM ancestry/strict gate."""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from inverse_folding.reference_flow.fusion_v2 import state as st
from inverse_folding.reference_flow.fusion_v2 import safety as sf
from inverse_folding.reference_flow.fusion_v2 import reward as rw
from inverse_folding.reference_flow.fusion_v2.errors import V2Error
from inverse_folding.reference_flow.fusion_v2.structure_gate import (
    DualScTMGatePolicy,
    authorize_ancestry,
)
from inverse_folding.reference_flow.fusion_v2_runtime.archive import (
    ExactArchive,
    select_family_representatives,
)
from inverse_folding.reference_flow.fusion_v2_runtime.admission import admit_endpoint
from tests.inverse_folding import _v2_fixtures as F


PROFILE = "exploratory_dual_sctm_ancestry070_strict085_v1"


def _policy():
    return DualScTMGatePolicy(
        profile_id=PROFILE, ancestry_sctm_min=0.70, strict_sctm_min=0.85,
    )


def _search_admission(endpoint, verdict, *, hard_anchors=()):
    """Mint the exact typed search evidence used by the runtime authorization path."""
    return admit_endpoint(
        F.safety_gate(), endpoint=endpoint,
        structure_definitive=verdict.strict_structure_passed,
        structure_ancestry=verdict.ancestry_structure_passed,
        structure_evidence_digest=verdict.verdict_digest,
        endpoint_tokens=tuple(
            evidence.token for evidence in endpoint.endpoint_provenance_evidence_by_pos
        ),
        hard_anchors=hard_anchors,
    )


@pytest.mark.parametrize(
    ("sctm", "ancestry", "strict"),
    [
        (0.699999, False, False),
        (0.70, True, False),
        (0.849999, True, False),
        (0.85, True, True),
    ],
)
def test_dual_sctm_boundaries_are_inclusive_and_nested(sctm, ancestry, strict):
    verdict = _policy().evaluate(
        StructureOutcome(feasible=True, metrics={"scTM": sctm}),
    )
    assert verdict.ancestry_structure_passed is ancestry
    assert verdict.strict_structure_passed is strict


@pytest.mark.parametrize(
    "outcome",
    [
        StructureOutcome(feasible=False, metrics={"scTM": 0.95}),
        StructureOutcome(feasible=True, metrics={}),
        StructureOutcome(feasible=True, metrics={"scTM": float("nan")}),
        StructureOutcome.deferred("not run"),
    ],
)
def test_dual_gate_fails_closed_when_common_structure_evidence_is_not_usable(outcome):
    verdict = _policy().evaluate(outcome)
    assert not verdict.ancestry_structure_passed
    assert not verdict.strict_structure_passed


def test_ancestry_authorization_is_bound_to_one_exact_endpoint():
    endpoint = F.endpoint(
        feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None,
    )
    outcome = StructureOutcome(feasible=True, metrics={"scTM": 0.75})
    verdict = _policy().evaluate(outcome)
    admission = _search_admission(endpoint, verdict)
    authorization = authorize_ancestry(
        endpoint=endpoint, structure_verdict=verdict,
        search_admission_evidence=admission.search_evidence,
    )
    provisional = dataclasses.replace(
        endpoint, feasibility_level=st.FeasibilityLevel.PROVISIONAL,
        structure_outcome=outcome, dual_structure_verdict=verdict,
        ancestry_authorization=authorization, endpoint_id=None,
    )
    assert st.endpoint_may_become_ancestry(provisional)

    foreign = dataclasses.replace(
        endpoint, fork_index=1, fork_seed=4243,
        replay=st.ReplayIdentity(
            mode="fork", rng_state=None, fork_seed=4243, replay_state_hash=F.E,
        ),
        endpoint_id=None,
    )
    with pytest.raises(st.V2StateError, match="authorization|endpoint"):
        dataclasses.replace(
            foreign, feasibility_level=st.FeasibilityLevel.PROVISIONAL,
            structure_outcome=outcome, dual_structure_verdict=verdict,
            ancestry_authorization=authorization, endpoint_id=None,
        )


def _dual_endpoint(sctm: float, **endpoint_overrides):
    endpoint = F.endpoint(
        feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None,
        **endpoint_overrides,
    )
    outcome = StructureOutcome(feasible=True, metrics={"scTM": sctm})
    verdict = _policy().evaluate(outcome)
    admission = _search_admission(endpoint, verdict)
    authorization = authorize_ancestry(
        endpoint=endpoint, structure_verdict=verdict,
        search_admission_evidence=admission.search_evidence,
    )
    level = (
        st.FeasibilityLevel.DEFINITIVE
        if verdict.strict_structure_passed else st.FeasibilityLevel.PROVISIONAL
    )
    return dataclasses.replace(
        endpoint, feasibility_level=level, structure_outcome=outcome,
        dual_structure_verdict=verdict, ancestry_authorization=authorization,
        endpoint_id=None,
    )


def test_authorized_provisional_endpoint_can_bind_the_recursive_reward_incumbent():
    endpoint = _dual_endpoint(0.75)
    incumbent = rw.bind_incumbent_from_endpoint(
        endpoint=endpoint, lineage_id="5ZHV_B:fam0",
        evaluator=endpoint.head_binding.evaluator,
        safety_reference_sequence_md5=F.safety_reference().sequence_md5,
        accepted_at_depth=1,
    )
    assert incumbent.source_endpoint_id == endpoint.endpoint_id
    assert incumbent.sequence_md5 == endpoint.sequence_md5


def test_authorized_provisional_donor_reaches_the_head_margin_gate():
    strict_parent = _dual_endpoint(0.90, tokens=(10, 19, 12, 13, 14, 15))
    incumbent = rw.bind_incumbent_from_endpoint(
        endpoint=strict_parent, lineage_id="5ZHV_B:fam0",
        evaluator=strict_parent.head_binding.evaluator,
        safety_reference_sequence_md5=F.safety_reference().sequence_md5,
        accepted_at_depth=1,
    )
    donor = _dual_endpoint(0.75)
    verdict = rw.donor_gate(
        donor=donor, incumbent=incumbent, epsilon_r=0.0,
        epsilon_source_ref=F.digest("dual-epsilon"),
    )
    assert verdict.reason is rw.DonorGateReason.STALL_NO_BETTER_DONOR

    unvalidated = F.endpoint(
        tokens=(10, 18, 12, 13, 14, 15),
        feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None,
    )
    refused = rw.donor_gate(
        donor=unvalidated, incumbent=incumbent, epsilon_r=0.0,
        epsilon_source_ref=F.digest("dual-epsilon"),
    )
    assert refused.reason is rw.DonorGateReason.DONOR_NOT_DEFINITIVE


def test_provisional_can_be_search_ancestry_but_never_archive_elite():
    endpoint = _dual_endpoint(0.75)
    archive = ExactArchive()
    archive.admit(endpoint, depth=0)

    assert archive.may_become_ancestry(endpoint.endpoint_id)
    assert select_family_representatives([endpoint]) == (endpoint,)
    assert archive.elite() is None


def test_strict_dual_endpoint_is_final_elite():
    endpoint = _dual_endpoint(0.90)
    archive = ExactArchive()
    archive.admit(endpoint, depth=0)

    assert archive.may_become_ancestry(endpoint.endpoint_id)
    assert archive.elite() is endpoint


def test_search_admission_separates_ancestry_from_strict_final():
    endpoint = F.endpoint(
        feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None,
    )
    verdict = _policy().evaluate(
        StructureOutcome(feasible=True, metrics={"scTM": 0.75}),
    )
    admission = _search_admission(endpoint, verdict)
    assert not admission.admitted
    assert admission.search_admitted
    assert admission.admission_evidence_digest
    assert admission.search_evidence is not None
    assert admission.search_evidence.verdict is admission.verdict


def test_authorization_refuses_a_bare_digest_even_when_it_is_real():
    endpoint = F.endpoint(
        feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None,
    )
    verdict = _policy().evaluate(
        StructureOutcome(feasible=True, metrics={"scTM": 0.75}),
    )
    admission = _search_admission(endpoint, verdict)
    with pytest.raises(TypeError, match="admission_evidence_digest"):
        authorize_ancestry(
            endpoint=endpoint, structure_verdict=verdict,
            admission_evidence_digest=admission.admission_evidence_digest,
        )


def _forge_search_evidence(evidence, **changes):
    forged = object.__new__(type(evidence))
    for field in dataclasses.fields(evidence):
        object.__setattr__(
            forged, field.name, changes.get(field.name, getattr(evidence, field.name)),
        )
    return forged


def test_authorization_recomputes_typed_admission_digest_live():
    endpoint = F.endpoint(
        feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None,
    )
    verdict = _policy().evaluate(
        StructureOutcome(feasible=True, metrics={"scTM": 0.75}),
    )
    admission = _search_admission(endpoint, verdict)
    forged = _forge_search_evidence(
        admission.search_evidence,
        admission_evidence_digest=F.digest("forged-admission"),
    )
    with pytest.raises(V2Error, match="digest"):
        authorize_ancestry(
            endpoint=endpoint, structure_verdict=verdict,
            search_admission_evidence=forged,
        )


def test_authorization_refuses_foreign_endpoint_and_admission_verdict():
    endpoint = F.endpoint(
        feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None,
    )
    verdict = _policy().evaluate(
        StructureOutcome(feasible=True, metrics={"scTM": 0.75}),
    )
    admission = _search_admission(endpoint, verdict)
    foreign = F.endpoint(
        feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None,
        fork_index=1, fork_seed=4243,
        replay=st.ReplayIdentity(
            mode="fork", rng_state=None, fork_seed=4243, replay_state_hash=F.E,
        ),
    )
    with pytest.raises(V2Error, match="endpoint"):
        authorize_ancestry(
            endpoint=foreign, structure_verdict=verdict,
            search_admission_evidence=admission.search_evidence,
        )

    foreign_admission = _search_admission(foreign, verdict)
    forged = _forge_search_evidence(
        admission.search_evidence, verdict=foreign_admission.verdict,
    )
    with pytest.raises(V2Error, match="verdict|endpoint|digest"):
        authorize_ancestry(
            endpoint=endpoint, structure_verdict=verdict,
            search_admission_evidence=forged,
        )


def test_authorization_refuses_failed_anchor_gate_and_foreign_structure_verdict():
    endpoint = F.endpoint(
        feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None,
    )
    verdict = _policy().evaluate(
        StructureOutcome(feasible=True, metrics={"scTM": 0.75}),
    )
    token0 = endpoint.endpoint_provenance_evidence_by_pos[0].token
    failed = _search_admission(endpoint, verdict, hard_anchors=((0, token0 + 1),))
    assert not failed.search_admitted
    assert failed.search_evidence is None
    with pytest.raises(V2Error, match="SearchAdmissionEvidence"):
        authorize_ancestry(
            endpoint=endpoint, structure_verdict=verdict,
            search_admission_evidence=failed.search_evidence,
        )

    admission = _search_admission(endpoint, verdict)
    foreign_structure = DualScTMGatePolicy(
        profile_id="foreign-profile", ancestry_sctm_min=0.71, strict_sctm_min=0.86,
    ).evaluate(StructureOutcome(feasible=True, metrics={"scTM": 0.75}))
    with pytest.raises(V2Error, match="structure verdict"):
        authorize_ancestry(
            endpoint=endpoint, structure_verdict=foreign_structure,
            search_admission_evidence=admission.search_evidence,
        )


def test_authorized_provisional_parent_advances_only_the_immune_ratchet():
    endpoint = F.endpoint(
        feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None,
    )
    gate = F.safety_gate()
    outcome = StructureOutcome(feasible=True, metrics={"scTM": 0.75})
    verdict = _policy().evaluate(outcome)
    admission = admit_endpoint(
        gate, endpoint=endpoint, structure_definitive=False, structure_ancestry=True,
        structure_evidence_digest=verdict.verdict_digest,
        endpoint_tokens=tuple(
            evidence.token for evidence in endpoint.endpoint_provenance_evidence_by_pos
        ),
        hard_anchors=(),
    )
    authorization = authorize_ancestry(
        endpoint=endpoint, structure_verdict=verdict,
        search_admission_evidence=admission.search_evidence,
    )
    selected = sf.bind_immediate_parent(
        endpoint_id=endpoint.endpoint_id, head_score=endpoint.head_score,
        depth=1, policy=gate.policy,
    )
    advanced = sf.advance_exploratory_lineage(
        gate.ledger, depth=1, selected=selected,
        admissibility=admission.verdict, authorization=authorization,
    )
    assert advanced.depth == 1
    assert advanced.cumulative_reference is gate.ledger.cumulative_reference
    assert advanced.immediate_parent is selected

    tampered = object.__new__(type(authorization))
    for field in dataclasses.fields(authorization):
        object.__setattr__(
            tampered, field.name,
            F.digest("foreign") if field.name == "admission_evidence_digest"
            else getattr(authorization, field.name),
        )
    with pytest.raises(sf.V2SafetyError, match="admission evidence"):
        sf.advance_exploratory_lineage(
            gate.ledger, depth=1, selected=selected,
            admissibility=admission.verdict,
            authorization=tampered,
        )


def test_archive_promotion_persists_dual_evidence_without_leaking_into_elite():
    endpoint = F.endpoint(
        feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None,
    )
    gate = F.safety_gate()
    outcome = StructureOutcome(feasible=True, metrics={"scTM": 0.75})
    verdict = _policy().evaluate(outcome)
    admission = admit_endpoint(
        gate, endpoint=endpoint, structure_definitive=False, structure_ancestry=True,
        structure_evidence_digest=verdict.verdict_digest,
        endpoint_tokens=tuple(
            evidence.token for evidence in endpoint.endpoint_provenance_evidence_by_pos
        ), hard_anchors=(),
    )
    authorization = authorize_ancestry(
        endpoint=endpoint, structure_verdict=verdict,
        search_admission_evidence=admission.search_evidence,
    )
    archive = ExactArchive()
    archive.admit(endpoint, depth=0)
    old_digest = archive.raw_rows()[0].endpoint_content_digest
    archive.promote(
        endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.PROVISIONAL,
        structure_outcome=outcome, dual_structure_verdict=verdict,
        ancestry_authorization=authorization, depth=0,
    )
    stored = archive.endpoints()[0]
    assert stored.feasibility_level is st.FeasibilityLevel.PROVISIONAL
    assert stored.ancestry_authorization == authorization
    assert archive.raw_rows()[0].endpoint_content_digest != old_digest
    assert archive.may_become_ancestry(endpoint.endpoint_id)
    assert archive.elite() is None


def test_repeated_archive_promotion_cannot_erase_existing_dual_evidence():
    endpoint = _dual_endpoint(0.90)
    archive = ExactArchive()
    archive.admit(endpoint, depth=0)

    # A legacy-shaped repeat promotion supplies no dual kwargs.  It must be idempotent, not turn a
    # strict dual endpoint into a legacy endpoint whose content no longer proves the two thresholds.
    archive.promote(
        endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.DEFINITIVE,
        structure_outcome=endpoint.structure_outcome, depth=1,
    )

    stored = archive.endpoints()[0]
    assert stored.dual_structure_verdict == endpoint.dual_structure_verdict
    assert stored.ancestry_authorization == endpoint.ancestry_authorization
    assert archive.dual_structure_verdict(endpoint.endpoint_id) == endpoint.dual_structure_verdict
    assert archive.ancestry_authorization(endpoint.endpoint_id) == endpoint.ancestry_authorization


def test_failed_repromotion_is_transactional_and_leaves_dual_archive_unchanged():
    endpoint = _dual_endpoint(0.90)
    archive = ExactArchive()
    archive.admit(endpoint, depth=0)
    before_entry = archive.raw_rows()[0]

    incompatible_outcome = StructureOutcome(feasible=True, metrics={"scTM": 0.95})
    with pytest.raises(st.V2StateError, match="authorization|structure evidence"):
        archive.promote(
            endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.DEFINITIVE,
            structure_outcome=incompatible_outcome, depth=1,
        )

    assert archive.endpoints()[0] == endpoint
    assert archive.raw_rows()[0] == before_entry
    assert archive.structure_outcome(endpoint.endpoint_id) == endpoint.structure_outcome
    assert archive.dual_structure_verdict(endpoint.endpoint_id) == endpoint.dual_structure_verdict
    assert archive.ancestry_authorization(endpoint.endpoint_id) == endpoint.ancestry_authorization


def test_cycle_can_continue_through_provisional_without_creating_a_final_elite():
    from tests.inverse_folding.test_fusion_v2_cycle import _run

    outcome = _run(
        structure_oracle=lambda request: StructureOutcome(
            feasible=True, metrics={"scTM": 0.75},
        ),
        dual_structure_policy=_policy(),
    )
    assert outcome.committed, outcome.detail
    assert outcome.selected_endpoint.feasibility_level is st.FeasibilityLevel.PROVISIONAL
    assert st.endpoint_may_become_ancestry(outcome.selected_endpoint)
    assert outcome.archive.elite() is None


def test_cycle_strict_dual_pass_remains_definitive_and_final_eligible():
    from tests.inverse_folding.test_fusion_v2_cycle import _run

    outcome = _run(
        structure_oracle=lambda request: StructureOutcome(
            feasible=True, metrics={"scTM": 0.90},
        ),
        dual_structure_policy=_policy(),
    )
    assert outcome.committed, outcome.detail
    assert outcome.selected_endpoint.feasibility_level is st.FeasibilityLevel.DEFINITIVE
    assert outcome.archive.elite() is not None


def test_full_ladder_can_continue_through_authorized_provisional_parents():
    from tests.inverse_folding.test_fusion_v2_ladder import _run

    outcome = _run(
        structure_oracle=lambda request: StructureOutcome(
            feasible=True, metrics={"scTM": 0.75},
        ),
        dual_structure_policy=_policy(),
    )
    assert outcome.depth_reached == 2
    assert outcome.final_safety_ledger.depth == 2
    assert outcome.best_definitive is None
    assert all(
        record.cycle.selected_endpoint.feasibility_level is st.FeasibilityLevel.PROVISIONAL
        for record in outcome.cycles
    )
