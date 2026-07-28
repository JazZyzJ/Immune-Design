"""V1F5 admission-mapping + coverage tests (PLAN_RF_REFINE_FUSION_V1 §2.11, §5).

The complete facade is admitted in design_idx order through the definitive structure gate,
first-N-feasible up to a common attempt cap; the audit maps design_idx -> slot BY ID (never by
sequence, so convergent roots stay distinguishable); coverage surfaces missing / duplicate
requested proteins.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_admission import (
    FacadeCandidate,
    admit_facade,
    build_cohort_coverage,
)


def _c(design_idx, source_id, seq="ACDE", root_hash=None, cid=None, seed=0):
    return FacadeCandidate(
        design_idx=design_idx, seed=seed, sequence=seq, sequence_md5=sequence_md5(seq),
        source_id=source_id, continuation_id=cid, root_equivalence_hash=root_hash,
    )


def test_admit_scans_by_design_idx_and_skips_infeasible():
    cands = [_c(0, "A"), _c(1, "F"), _c(2, "K")]
    feasible = {"A": False, "F": True, "K": True}  # rank-0 fails structure, later roots pass
    res = admit_facade(cands, lambda c: feasible[c.source_id], n_population=2, attempt_cap=10)
    assert res.n_admitted == 2
    assert [a.source_id for a in res.admitted] == ["F", "K"]
    assert [a.slot_idx for a in res.admitted] == [0, 1]
    assert res.attempts[0].source_id == "A" and res.attempts[0].verdict == "structure_infeasible"
    assert not res.insufficient


def test_admit_order_is_facade_rank_never_reordered():
    cands = [_c(0, "lo"), _c(1, "hi")]  # facade order = root-value rank (already applied upstream)
    res = admit_facade(cands, lambda c: True, n_population=2, attempt_cap=10)
    assert [a.source_id for a in res.admitted] == ["lo", "hi"]


def test_admit_maps_convergent_roots_by_id_not_sequence():
    c0 = _c(0, "r0", seq="ACDE", root_hash="h0", cid="k0")
    c1 = _c(1, "r1", seq="ACDE", root_hash="h1", cid="k1")  # SAME sequence, distinct root/id
    res = admit_facade([c0, c1], lambda c: True, n_population=2, attempt_cap=10)
    assert [a.root_equivalence_hash for a in res.admitted] == ["h0", "h1"]
    assert [a.continuation_id for a in res.admitted] == ["k0", "k1"]
    assert [a.slot_idx for a in res.admitted] == [0, 1]


def test_admit_insufficient_is_visible_within_cap():
    cands = [_c(i, f"c{i}") for i in range(3)]
    res = admit_facade(cands, lambda c: False, n_population=2, attempt_cap=3)
    assert res.n_admitted == 0 and res.insufficient
    assert res.reason == "insufficient_feasible_initial_population"
    assert len(res.attempts) == 3  # every attempt is recorded, not hidden


def test_admit_stops_at_common_attempt_cap():
    cands = [_c(i, f"c{i}") for i in range(10)]
    res = admit_facade(cands, lambda c: False, n_population=2, attempt_cap=4)
    assert len(res.attempts) == 4  # capped
    assert res.insufficient


def test_admit_never_duplicates_a_slot_or_a_parent():
    cands = [_c(0, "a"), _c(1, "b"), _c(2, "c")]
    res = admit_facade(cands, lambda c: True, n_population=2, attempt_cap=10)
    assert len(res.admitted) == 2
    assert {a.slot_idx for a in res.admitted} == {0, 1}
    assert len({a.source_id for a in res.admitted}) == 2  # distinct feasible parents


def test_admit_is_input_order_independent():
    cands = [_c(2, "c"), _c(0, "a"), _c(1, "b")]  # shuffled input
    res = admit_facade(cands, lambda c: True, n_population=3, attempt_cap=10)
    assert [a.design_idx for a in res.admitted] == [0, 1, 2]


def test_cohort_coverage_surfaces_missing_and_rejects_duplicates():
    cov = build_cohort_coverage(
        ["P1", "P2", "P3"], {"P1": "terminal_success", "P2": "entry_insufficient"}
    )
    assert {r.protein_id: r.status for r in cov} == {
        "P1": "terminal_success", "P2": "entry_insufficient", "P3": "missing"
    }
    with pytest.raises(ValueError):
        build_cohort_coverage(["P1", "P1"], {})  # duplicate requested protein
    with pytest.raises(ValueError):
        build_cohort_coverage(["P1"], {"P9": "x"})  # status for a non-requested protein


# --------------------------------------------------------------------------- #
# honest structure deferral (§2.11): the entry stage does NOT run structure
# --------------------------------------------------------------------------- #
def test_a_deferred_outcome_may_not_claim_feasibility():
    """The entry stage defers definitive structure to the unchanged v0 admission. An outcome that
    says "I did not evaluate" while also reporting ``feasible=True`` is the exact shape of the
    fabrication this guards: every downstream table would read it as a passed structure gate."""
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome

    with pytest.raises(ValueError):
        StructureOutcome(feasible=True, evaluated=False)
    with pytest.raises(ValueError):
        StructureOutcome(feasible=None, evaluated=True)  # evaluated must state a verdict
    with pytest.raises(ValueError):  # nothing was executed, so it cannot claim a model run
        StructureOutcome(feasible=None, evaluated=False, model_executed=True)
    deferred = StructureOutcome.deferred("structure rechecked by v0 admission")
    assert deferred.evaluated is False and deferred.feasible is None


def test_deferred_structure_is_recorded_as_deferred_not_admitted():
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome

    cands = [_c(i, f"s{i}") for i in range(4)]
    res = admit_facade(
        cands, lambda c: StructureOutcome.deferred("v0 rechecks"), n_population=2, attempt_cap=3
    )
    # nothing was admitted, because nothing was evaluated
    assert res.n_admitted == 0 and res.admitted == ()
    assert res.n_deferred == 3  # the whole attempt_cap is offered to v0
    assert all(a.verdict == "deferred_to_v0" for a in res.attempts)
    assert all(a.slot_idx is None and a.structure_feasible is None for a in res.attempts)
    # "insufficient" would claim the population FAILED; the decision merely moved downstream.
    assert not res.insufficient
    assert res.reason == "structure_deferred_to_v0"


def test_admission_refuses_to_mix_evaluated_and_deferred_rows():
    """A partially-evaluated facade produces an ``n_admitted`` that reads like a full count but
    covers only some rows; the policy must be one or the other for the whole protein."""
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome

    cands = [_c(0, "A"), _c(1, "B")]
    outcomes = {"A": StructureOutcome(feasible=True), "B": StructureOutcome.deferred("v0")}
    with pytest.raises(ValueError, match="deferred"):
        admit_facade(cands, lambda c: outcomes[c.source_id], n_population=2, attempt_cap=2)
    # ...and in the other order, where the deferred row comes first
    flipped = {"A": StructureOutcome.deferred("v0"), "B": StructureOutcome(feasible=True)}
    with pytest.raises(ValueError, match="deferred"):
        admit_facade(cands, lambda c: flipped[c.source_id], n_population=2, attempt_cap=2)
