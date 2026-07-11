"""F1 unit tests — fusion/state.py immutable state objects and fail-fast validation.

Scientific/interface contract: PLAN_RF_REFINE_FUSION.md §1.1. Every inherited state is a
complete canonical AA20 sequence with a definitive structure verdict and a Head value;
IDs are content-derived (process-independent, never Python ``hash()``); a population's
weights are normalized; the elite is a feasible best-so-far.

torch is available locally but these objects are pure Python — importing the fusion state
layer must not require a model call (PLAN F1 gate).
"""
from types import SimpleNamespace

import pytest

from inverse_folding.reference_flow.fusion import state as st


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_UNSET = object()


def _struct(scTM=0.92, active_site_RMSD=None, passed=True):
    return SimpleNamespace(scTM=scTM, pLDDT=80.0, scRMSD=None,
                           active_site_RMSD=active_site_RMSD, passed=passed, reason="")


def _particle(seq="ACDEFGHIKL", *, weight=1.0, round_idx=0, slot_idx=0,
              head_global_risk=1.5, structure=_UNSET, feasible=True,
              parent=None, source_proposal=None, lineage_seed=7):
    structure = _struct() if structure is _UNSET else structure
    return st.ParticleState(
        particle_id=st.make_particle_id("Q00511", round_idx, slot_idx, st.sequence_md5(seq)),
        protein_id="Q00511",
        round_idx=round_idx,
        slot_idx=slot_idx,
        sequence=seq,
        sequence_md5=st.sequence_md5(seq),
        weight=weight,
        parent_particle_id=parent,
        source_proposal_id=source_proposal,
        head_global_risk=head_global_risk,
        structure=structure,
        feasible=feasible,
        lineage_seed=lineage_seed,
    )


# --------------------------------------------------------------------------- #
# sequence md5 + id helpers
# --------------------------------------------------------------------------- #
def test_sequence_md5_is_deterministic_and_content_derived():
    assert st.sequence_md5("ACDE") == st.sequence_md5("ACDE")
    assert st.sequence_md5("ACDE") != st.sequence_md5("ACDF")


def test_ids_are_content_derived_not_python_hash():
    md5 = st.sequence_md5("ACDEFGHIKL")
    pid = st.make_particle_id("Q00511", 3, 5, md5)
    # deterministic + pins a content-derived scheme (no Python hash(), which is salted/unstable)
    assert pid == st.make_particle_id("Q00511", 3, 5, md5)
    assert "Q00511" in pid and "r3" in pid and "s5" in pid and md5[:12] in pid


def test_proposal_id_is_content_derived():
    md5 = st.sequence_md5("ACDEFGHIKL")
    a = st.make_proposal_id("parentX", "explicit_edit", md5)
    assert a == st.make_proposal_id("parentX", "explicit_edit", md5)
    assert a != st.make_proposal_id("parentX", "rf_reopen", md5)


# --------------------------------------------------------------------------- #
# ParticleState
# --------------------------------------------------------------------------- #
def test_particle_state_valid_construction():
    p = _particle()
    assert p.sequence == "ACDEFGHIKL"
    assert p.feasible is True
    assert p.structure.scTM == pytest.approx(0.92)


def test_particle_state_rejects_noncanonical_sequence():
    with pytest.raises(st.FusionStateError):
        _particle(seq="ACDEFGHIKX")  # X is not canonical AA20


def test_particle_state_rejects_mask_or_lowercase():
    with pytest.raises(st.FusionStateError):
        _particle(seq="ACDEFGHIK-")  # mask-like token


def test_particle_state_rejects_md5_mismatch():
    good = _particle()
    with pytest.raises(st.FusionStateError):
        st.ParticleState(**{**good.__dict__, "sequence_md5": "deadbeef" * 4})


def test_particle_state_requires_definitive_structure_verdict():
    # an inherited state must carry a structure result (no missing evaluation)
    with pytest.raises(st.FusionStateError):
        _particle(structure=None)


def test_particle_state_requires_head_global_risk():
    with pytest.raises(st.FusionStateError):
        _particle(head_global_risk=None)


def test_particle_state_rejects_negative_weight():
    with pytest.raises(st.FusionStateError):
        _particle(weight=-0.1)


def test_particle_state_is_frozen():
    p = _particle()
    with pytest.raises(Exception):
        p.sequence = "AAAA"  # frozen dataclass


# --------------------------------------------------------------------------- #
# Proposal
# --------------------------------------------------------------------------- #
def _proposal(seq="ACDEFGHIKL", edited=(2,), t=(1, 4), halo=(0, 6), family="explicit_edit"):
    return st.Proposal(
        proposal_id=st.make_proposal_id("parent0", family, st.sequence_md5(seq)),
        parent_particle_id="parent0",
        move_family=family,
        sequence=seq,
        edited_positions=tuple(edited),
        target_start_0b=t[0],
        target_end_0b=t[1],
        halo_start_0b=halo[0],
        halo_end_0b=halo[1],
        proposal_seed=11,
    )


def test_proposal_valid_construction():
    pr = _proposal()
    assert pr.move_family == "explicit_edit"
    assert pr.edited_positions == (2,)


def test_proposal_rejects_unknown_move_family():
    with pytest.raises(st.FusionStateError):
        _proposal(family="magic_edit")


def test_proposal_halo_must_contain_target():
    with pytest.raises(st.FusionStateError):
        _proposal(t=(1, 4), halo=(2, 3))  # target not inside halo


def test_proposal_edited_positions_must_be_in_range():
    with pytest.raises(st.FusionStateError):
        _proposal(seq="ACDEFGHIKL", edited=(99,))


def test_proposal_rejects_noncanonical_sequence():
    with pytest.raises(st.FusionStateError):
        _proposal(seq="ACDEFGHIKZ")


# --------------------------------------------------------------------------- #
# CandidateEvaluation
# --------------------------------------------------------------------------- #
def _candidate(*, structure_evaluated=True, feasible=True, structure=None):
    if structure is None and structure_evaluated:
        structure = _struct()
    return st.CandidateEvaluation(
        sequence_md5=st.sequence_md5("ACDEFGHIKL"),
        parent_particle_id="parent0",
        proposal_id="prop0",
        target_start_0b=1,
        target_end_0b=4,
        head_global_risk=1.2,
        aligned_windows=(),
        local_target_delta=-0.3,
        new_hotspot_max=0.0,
        new_hotspot_mass=0.0,
        new_hotspot_count=0,
        structure=structure,
        structure_evaluated=structure_evaluated,
        feasible=feasible,
        reason="",
    )


def test_candidate_evaluation_feasible_implies_structure_evaluated():
    with pytest.raises(st.FusionStateError):
        _candidate(structure_evaluated=False, feasible=True, structure=None)


def test_candidate_evaluation_unrefolded_is_ok_when_infeasible():
    c = _candidate(structure_evaluated=False, feasible=False, structure=None)
    assert c.structure_evaluated is False
    assert c.feasible is False


# --------------------------------------------------------------------------- #
# PopulationState + EliteState
# --------------------------------------------------------------------------- #
def _elite(p=None):
    p = _particle() if p is None else p
    return st.EliteState(particle=p, first_round_seen=0, last_round_seen=0)


def test_population_weights_must_sum_to_one():
    p0 = _particle(weight=0.5, slot_idx=0)
    p1 = _particle(weight=0.5, slot_idx=1)
    pop = st.PopulationState(round_idx=0, particles=(p0, p1), elite=_elite(p0))
    assert len(pop.particles) == 2

    bad0 = _particle(weight=0.9, slot_idx=0)
    bad1 = _particle(weight=0.9, slot_idx=1)
    with pytest.raises(st.FusionStateError):
        st.PopulationState(round_idx=0, particles=(bad0, bad1), elite=_elite(bad0))


def test_population_rejects_mixed_round_index():
    p0 = _particle(weight=0.5, slot_idx=0, round_idx=0)
    p1 = _particle(weight=0.5, slot_idx=1, round_idx=1)  # wrong round
    with pytest.raises(st.FusionStateError):
        st.PopulationState(round_idx=0, particles=(p0, p1), elite=_elite(p0))


def test_population_rejects_empty():
    with pytest.raises(st.FusionStateError):
        st.PopulationState(round_idx=0, particles=(), elite=_elite())


def test_uniform_population_helper_resets_weights():
    ps = [_particle(weight=0.3, slot_idx=i) for i in range(4)]
    pop = st.uniform_population(round_idx=0, particles=ps, elite=_elite(ps[0]))
    assert sum(p.weight for p in pop.particles) == pytest.approx(1.0)
    assert all(p.weight == pytest.approx(0.25) for p in pop.particles)


def test_elite_must_be_feasible():
    infeasible = _particle(feasible=False)
    with pytest.raises(st.FusionStateError):
        st.EliteState(particle=infeasible, first_round_seen=0, last_round_seen=0)


def test_elite_rounds_ordered():
    with pytest.raises(st.FusionStateError):
        st.EliteState(particle=_particle(), first_round_seen=3, last_round_seen=1)


def test_population_rejects_duplicate_particle_ids_or_noncontiguous_slots():
    p0 = _particle(weight=0.5, slot_idx=0)
    dup = _particle(weight=0.5, slot_idx=0)  # same slot -> same content-derived id
    with pytest.raises(st.FusionStateError):
        st.PopulationState(round_idx=0, particles=(p0, dup), elite=_elite(p0))
    p_gap = _particle(weight=0.5, slot_idx=2)  # slots {0,2} not 0..1
    with pytest.raises(st.FusionStateError):
        st.PopulationState(round_idx=0, particles=(p0, p_gap), elite=_elite(p0))
