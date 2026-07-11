"""F3 unit tests — fusion/moves.py explicit breadth proposer (pure, no Head/NMP/structure).

Contract: PLAN_RF_REFINE_FUSION.md §1.5 / Task F3. Complete AA20 single substitutions inside
the target register (anchors already excluded upstream, never the unchanged parent, length
preserved); then deterministic-seeded distinct-position pairs with dedup and stable
content-derived proposal IDs; respect the per-parent raw budget; a target with no editable
position yields no proposals rather than mutating anchors.
"""
from types import SimpleNamespace

import pytest

from inverse_folding.reference_flow.fusion import moves as mv
from inverse_folding.reference_flow.fusion import state as st

PARENT_SEQ = "ACDEFGHIKLMNPQRST"  # length 17, canonical


def _parent(seq=PARENT_SEQ):
    return SimpleNamespace(sequence=seq, particle_id="Q:r0:s0:abc123abc123")


def _target(start, end, editable):
    return SimpleNamespace(start_0b=start, end_0b=end, editable_positions=tuple(editable))


def _props(target, **kw):
    kw.setdefault("max_edit_order", 2)
    kw.setdefault("max_raw_candidates", 10_000)
    kw.setdefault("pair_seed_budget", 64)
    kw.setdefault("base_seed", 7)
    return mv.explicit_edit_proposals(_parent(), target, **kw)


# --------------------------------------------------------------------------- #
# singles
# --------------------------------------------------------------------------- #
def test_singles_complete_aa20_no_unchanged_length_preserved():
    tgt = _target(4, 8, (4, 5, 6, 7))
    props = _props(tgt, max_edit_order=1)
    assert len(props) == 19 * 4  # 19 alternatives per editable position
    for p in props:
        assert len(p.edited_positions) == 1
        pos = p.edited_positions[0]
        assert pos in (4, 5, 6, 7)
        assert p.sequence != PARENT_SEQ
        assert p.sequence[pos] != PARENT_SEQ[pos]
        # differs from parent ONLY at pos
        assert [i for i in range(len(PARENT_SEQ)) if p.sequence[i] != PARENT_SEQ[i]] == [pos]
        assert len(p.sequence) == len(PARENT_SEQ)
        assert p.move_family == "explicit_edit"
    # all 19 non-parent AAs appear at position 4
    aa_at_4 = {p.sequence[4] for p in props if p.edited_positions == (4,)}
    assert aa_at_4 == set(st.CANONICAL_AA20) - {PARENT_SEQ[4]}


def test_singles_only_touch_editable_positions():
    # positions 5 and 7 are anchors -> excluded from editable upstream
    tgt = _target(4, 8, (4, 6))
    props = _props(tgt, max_edit_order=1)
    edited = {p.edited_positions[0] for p in props}
    assert edited == {4, 6}


def test_max_edit_order_one_has_no_pairs():
    tgt = _target(4, 8, (4, 5, 6, 7))
    props = _props(tgt, max_edit_order=1, pair_seed_budget=64)
    assert all(len(p.edited_positions) == 1 for p in props)


# --------------------------------------------------------------------------- #
# pairs
# --------------------------------------------------------------------------- #
def test_pairs_distinct_positions_within_editable():
    tgt = _target(4, 8, (4, 5, 6, 7))
    props = _props(tgt, max_edit_order=2, pair_seed_budget=32)
    pairs = [p for p in props if len(p.edited_positions) == 2]
    assert pairs, "expected some 2-position pair proposals"
    for p in pairs:
        a, b = p.edited_positions
        assert a != b
        assert a in (4, 5, 6, 7) and b in (4, 5, 6, 7)
        assert p.sequence[a] != PARENT_SEQ[a] and p.sequence[b] != PARENT_SEQ[b]


def test_deterministic_same_seed_same_proposals():
    tgt = _target(4, 8, (4, 5, 6, 7))
    a = _props(tgt, base_seed=123)
    b = _props(tgt, base_seed=123)
    assert [p.sequence for p in a] == [p.sequence for p in b]
    assert [p.edited_positions for p in a] == [p.edited_positions for p in b]


def test_sequences_are_deduplicated():
    tgt = _target(4, 8, (4, 5, 6, 7))
    props = _props(tgt)
    seqs = [p.sequence for p in props]
    assert len(seqs) == len(set(seqs))


def test_proposal_ids_are_content_derived():
    tgt = _target(4, 8, (4, 5, 6, 7))
    props = _props(tgt)
    for p in props:
        assert p.proposal_id == st.make_proposal_id(
            p.parent_particle_id, "explicit_edit", st.sequence_md5(p.sequence))


# --------------------------------------------------------------------------- #
# budget + degenerate target
# --------------------------------------------------------------------------- #
def test_all_singles_always_emitted_cap_governs_pairs():
    tgt = _target(4, 8, (4, 5, 6, 7))  # 4 editable positions -> 76 singles
    # cap BELOW the single count -> all 76 singles still emitted (breadth over every position), 0 pairs
    props = _props(tgt, max_raw_candidates=5, pair_seed_budget=64)
    assert len(props) == 76
    assert all(len(p.edited_positions) == 1 for p in props)
    edited_positions = {p.edited_positions[0] for p in props}
    assert edited_positions == {4, 5, 6, 7}          # NO position dropped (was the P1-6 bug)
    # cap ABOVE singles -> all singles + pairs up to the remaining headroom
    props2 = _props(tgt, max_raw_candidates=80, pair_seed_budget=64)
    assert sum(1 for p in props2 if len(p.edited_positions) == 1) == 76
    assert sum(1 for p in props2 if len(p.edited_positions) == 2) == 4  # 80 - 76 singles


def test_empty_editable_returns_no_proposals():
    tgt = _target(4, 8, ())
    assert _props(tgt) == []


# =========================================================================== #
# F6: RF reopen / edit-repair adapters (fake repair_fn — no torch)
# =========================================================================== #
import numpy as np  # noqa: E402

RPARENT = "ACDEFGHIKLMNPQRSTVWY" * 2  # length 40


def _rparent():
    return SimpleNamespace(sequence=RPARENT, particle_id="Q:r0:s0:abc123abc123")


def _rtarget(start=20, end=25):
    return SimpleNamespace(start_0b=start, end_0b=end, editable_positions=tuple(range(start, end)))


def _fake_repair(base, frozen, regen, seed):
    """Regenerate only `regen` positions (seeded), preserving every frozen position."""
    rng = np.random.default_rng(seed)
    chars = list(base)
    for p in sorted(regen):
        chars[p] = "ACDEFGHIKLMNPQRSTVWY"[int(rng.integers(20))]
    return "".join(chars)


def test_rf_reopen_freezes_complement_and_regenerates_halo():
    parent = _rparent()
    props = mv.rf_reopen_proposals(parent, _rtarget(20, 25), halo_radius=4, anchors=set(),
                                   repair_fn=_fake_repair, base_seed=1, children=3)
    assert props
    for p in props:
        assert p.move_family == "rf_reopen" and p.edited_positions == ()
        assert (p.halo_start_0b, p.halo_end_0b) == (16, 29)
        assert len(p.sequence) == len(RPARENT)
        assert set(p.sequence) <= set("ACDEFGHIKLMNPQRSTVWY")  # complete, no mask
        # everything OUTSIDE the halo is byte-identical to the parent
        assert p.sequence[:16] == RPARENT[:16] and p.sequence[29:] == RPARENT[29:]


def test_rf_reopen_freezes_anchors_inside_halo():
    parent = _rparent()
    props = mv.rf_reopen_proposals(parent, _rtarget(20, 25), halo_radius=4, anchors={20, 22},
                                   repair_fn=_fake_repair, base_seed=2, children=3)
    for p in props:
        assert p.sequence[20] == RPARENT[20] and p.sequence[22] == RPARENT[22]  # anchors frozen


def test_edit_repair_protects_edit_anchors_and_outside_halo():
    parent = _rparent()
    edit = {22: "W"}  # immune edit inside the target [20,25)
    props = mv.edit_repair_proposals(parent, _rtarget(20, 25), edit, halo_radius=4, anchors={18},
                                     repair_fn=_fake_repair, base_seed=3, children=4)
    assert props
    for p in props:
        assert p.move_family == "edit_repair" and p.edited_positions == (22,)
        assert p.sequence[22] == "W"                 # protected immune edit never reverted
        assert p.sequence[18] == RPARENT[18]          # anchor preserved (outside halo here)
        assert p.sequence[:16] == RPARENT[:16] and p.sequence[29:] == RPARENT[29:]  # outside halo


def test_edit_repair_asserts_when_repair_reverts_a_frozen_position():
    parent = _rparent()

    def _bad_repair(base, frozen, regen, seed):  # maliciously flips the protected edit
        chars = list(_fake_repair(base, frozen, regen, seed))
        chars[22] = "A"  # 22 is the frozen edit position
        return "".join(chars)

    with pytest.raises(mv.MovesError):
        mv.edit_repair_proposals(parent, _rtarget(20, 25), {22: "W"}, halo_radius=4, anchors=set(),
                                 repair_fn=_bad_repair, base_seed=4, children=1)


def test_repair_rejects_mask_or_noncanonical_survivor():
    parent = _rparent()

    def _mask_repair(base, frozen, regen, seed):
        chars = list(base)
        for p in regen:
            chars[p] = "-"  # a surviving mask token
        return "".join(chars)

    with pytest.raises(mv.MovesError):
        mv.rf_reopen_proposals(parent, _rtarget(20, 25), halo_radius=4, anchors=set(),
                               repair_fn=_mask_repair, base_seed=5, children=1)


def test_per_child_distinct_seeds_yield_distinct_children():
    parent = _rparent()
    props = mv.rf_reopen_proposals(parent, _rtarget(20, 25), halo_radius=4, anchors=set(),
                                   repair_fn=_fake_repair, base_seed=6, children=5)
    assert len({p.sequence for p in props}) > 1  # per-child seeds -> diverse repairs


def test_rf_reopen_skips_reproduction_of_parent():
    parent = _rparent()

    def _identity_repair(base, frozen, regen, seed):
        return base  # regenerates back to the (edited) base == parent for rf_reopen

    props = mv.rf_reopen_proposals(parent, _rtarget(20, 25), halo_radius=4, anchors=set(),
                                   repair_fn=_identity_repair, base_seed=7, children=3)
    assert props == []  # a reopen that reproduces the parent is not a proposal


def test_edit_repair_rejects_editing_an_anchor_or_outside_target():
    parent = _rparent()
    with pytest.raises(mv.MovesError):
        mv.edit_repair_proposals(parent, _rtarget(20, 25), {18: "W"}, halo_radius=4, anchors={18},
                                 repair_fn=_fake_repair, base_seed=8, children=1)
    with pytest.raises(mv.MovesError):
        mv.edit_repair_proposals(parent, _rtarget(20, 25), {5: "W"}, halo_radius=4, anchors=set(),
                                 repair_fn=_fake_repair, base_seed=8, children=1)
