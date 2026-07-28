"""V1F3 partial-root record / hash / provenance tests (PLAN_RF_REFINE_FUSION_V1 §2.4, §4.2).

Three distinct identities (lineage root_id, root_equivalence_hash, snapshot_payload_hash),
deterministic equivalence collapse that retains converged lineages, fail-fast validation,
lossless round-trip, and content-digest resume invalidation.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion.v1_records import (
    ConditioningDigest,
    PartialRootPayload,
    collapse_roots,
    content_digest,
    make_root_id,
)


def _cond(**over):
    base = dict(
        dplm_checkpoint="dplm#a", tokenizer="tok#a", backbone_row="bb#a",
        coordinate_mask="cm#a", entry_config="ecfg#a",
        fixed_token_policy="unconstrained", controller_enabled=False, h_maps_present=False,
    )
    base.update(over)
    return ConditioningDigest(**base)


def _payload(root_id="P:C:rho0.850:r0", *, x_t=(0, 1, 4, 4), scores=(0.0, -1.0, -9e9, -9e9),
             step=3, cond=None, fixed=(), editable=(0, 1, 2, 3), n_unresolved=2, mask=4):
    return PartialRootPayload(
        schema_version="v1root-1", root_id=root_id, protein_id="P", arm_id="C",
        rho_id="rho0.850", x_t=tuple(x_t), scores=tuple(scores),
        unmask_step_by_pos=(0, 1, -1, -1), step=step, n_steps=10, t=step / 10,
        snapshot_phase="pre_denoiser_after_previous_remask", fixed_tokens=tuple(fixed),
        editable_positions=tuple(editable), n_unresolved_editable=n_unresolved, mask_token_id=mask,
        paid_prefix_dfe=step, rng_state={"bit_generator": "PCG64", "state": {"state": 1, "inc": 3}, "has_uint32": 0, "uinteger": 0},
        conditioning=cond or _cond(),
    )


def test_n_unresolved_editable_is_derived_from_x_t_not_trusted():
    # A fully-resolved payload (no editable position holds the mask token) cannot claim rho<1.
    with pytest.raises(ValueError):
        _payload(x_t=(0, 1, 2, 3), n_unresolved=1)  # nothing equals mask token 4
    # A declared count that disagrees with the masked positions is rejected.
    with pytest.raises(ValueError):
        _payload(x_t=(0, 1, 4, 4), n_unresolved=1)  # two masked (token 4), not one


def test_actual_rho_edit_reflects_masked_fraction_not_target():
    p = _payload(x_t=(0, 1, 4, 4), editable=(0, 1, 2, 3), n_unresolved=2)
    assert p.actual_rho_edit == 0.5  # 2 of 4 editable resolved, independent of rho_id


def test_payload_rejects_mask_token_or_inconsistent_fixed_anchor():
    # a fixed anchor is a resolved residue: it cannot hold the mask token ...
    with pytest.raises(ValueError):
        _payload(x_t=(4, 1, 4, 4), fixed=((0, 4),), editable=(1, 2, 3), n_unresolved=2)
    # ... and x_t at the anchor must equal the declared fixed token.
    with pytest.raises(ValueError):
        _payload(x_t=(0, 1, 4, 4), fixed=((0, 9),), editable=(1, 2, 3), n_unresolved=2)


def test_numeric_int_float_twins_collapse_to_one():
    a = _payload(root_id="a", scores=(0, -1, -9, -9))          # int scores
    b = _payload(root_id="b", scores=(0.0, -1.0, -9.0, -9.0))  # float, numerically identical
    assert a.root_equivalence_hash == b.root_equivalence_hash
    assert collapse_roots([a, b]).n_unique == 1  # one basin, one slot


def test_equivalence_and_payload_hash_stable_across_roundtrip():
    a = _payload(scores=(0, -1, -9, -9))  # ints in, floats after coercion
    b = PartialRootPayload.from_dict(a.to_dict())
    assert a.root_equivalence_hash == b.root_equivalence_hash
    assert a.snapshot_payload_hash == b.snapshot_payload_hash


def test_fixed_tokens_and_editable_order_do_not_affect_hash():
    a = _payload(root_id="a", x_t=(9, 9, 4, 4), fixed=((0, 9), (1, 9)), editable=(2, 3))
    b = _payload(root_id="b", x_t=(9, 9, 4, 4), fixed=((1, 9), (0, 9)), editable=(3, 2))
    assert a.root_equivalence_hash == b.root_equivalence_hash
    assert collapse_roots([a, b]).n_unique == 1


def test_scores_reject_nan_and_normalize_negative_zero():
    with pytest.raises(ValueError):
        _payload(scores=(float("nan"), -1, -9, -9))
    pos = _payload(scores=(0.0, -1, -9, -9))
    neg = _payload(scores=(-0.0, -1, -9, -9))
    assert pos.root_equivalence_hash == neg.root_equivalence_hash


def test_payload_requires_at_least_one_unresolved_editable():
    with pytest.raises(ValueError):
        _payload(n_unresolved=0)  # rho_edit == 1.0 is not a valid pre-terminal root


# ------------------------- fail-fast validation ------------------------- #
def test_payload_rejects_length_mismatch():
    with pytest.raises(ValueError):
        _payload(x_t=(0, 1, 4))  # scores/unmask are length 4


def test_payload_rejects_editable_fixed_overlap_and_out_of_range():
    with pytest.raises(ValueError):
        _payload(fixed=((0, 1),), editable=(0, 1, 2, 3))  # 0 in both
    with pytest.raises(ValueError):
        _payload(editable=(0, 1, 2, 9))  # 9 out of range for len 4


def test_payload_rejects_paid_prefix_dfe_not_equal_step():
    with pytest.raises(ValueError):
        PartialRootPayload(
            schema_version="v1root-1", root_id="r", protein_id="P", arm_id="C", rho_id="rho0.850",
            x_t=(0, 4), scores=(0.0, -9e9), unmask_step_by_pos=(0, -1), step=1, n_steps=4, t=0.25,
            snapshot_phase="pre_denoiser_after_previous_remask", fixed_tokens=(),
            editable_positions=(0, 1), n_unresolved_editable=1, mask_token_id=4, paid_prefix_dfe=99,
            rng_state={}, conditioning=_cond(),
        )


# ------------------------- three distinct identities ------------------------- #
def test_equivalence_hash_ignores_lineage_unmask_and_rng():
    a = _payload(root_id="P:C:rho0.850:r0")
    b = _payload(root_id="P:C:rho0.850:r7")  # different lineage id, same visible state
    # different rng stream + different unmask telemetry must NOT change the equivalence hash
    b2 = PartialRootPayload(
        schema_version=b.schema_version, root_id=b.root_id, protein_id=b.protein_id,
        arm_id=b.arm_id, rho_id=b.rho_id, x_t=b.x_t, scores=b.scores,
        unmask_step_by_pos=(3, 2, -1, -1), step=b.step, n_steps=b.n_steps, t=b.t,
        snapshot_phase=b.snapshot_phase, fixed_tokens=b.fixed_tokens,
        editable_positions=b.editable_positions, n_unresolved_editable=b.n_unresolved_editable,
        mask_token_id=b.mask_token_id, paid_prefix_dfe=b.paid_prefix_dfe,
        rng_state={"bit_generator": "PCG64", "state": {"state": 999, "inc": 5}, "has_uint32": 1, "uinteger": 7},
        conditioning=b.conditioning,
    )
    assert a.root_equivalence_hash == b.root_equivalence_hash == b2.root_equivalence_hash
    # but the full payload hash DOES depend on lineage + rng + unmask
    assert a.snapshot_payload_hash != b.snapshot_payload_hash
    assert b.snapshot_payload_hash != b2.snapshot_payload_hash


def test_equivalence_hash_depends_on_visible_state_and_conditioning():
    a = _payload()
    assert a.root_equivalence_hash != _payload(x_t=(0, 1, 2, 4), n_unresolved=1).root_equivalence_hash
    assert a.root_equivalence_hash != _payload(cond=_cond(backbone_row="bb#B")).root_equivalence_hash
    assert a.root_equivalence_hash != _payload(cond=_cond(entry_config="ecfg#B")).root_equivalence_hash


# ------------------------- equivalence collapse ------------------------- #
def test_collapse_dedups_by_equivalence_keeps_lex_smallest_representative():
    # two identical visible roots, different lineage ids -> ONE unique slot
    p_hi = _payload(root_id="P:C:rho0.850:r9")
    p_lo = _payload(root_id="P:C:rho0.850:r1")
    distinct = _payload(root_id="P:C:rho0.850:r2", x_t=(0, 1, 2, 4), n_unresolved=1)
    collapse = collapse_roots([p_hi, p_lo, distinct])
    assert collapse.n_unique == 2
    reps = {p.root_id for p in collapse.unique}
    assert "P:C:rho0.850:r1" in reps  # lex-smallest representative of the converged group
    assert "P:C:rho0.850:r9" not in reps  # the converged sibling is not an independent slot
    # converged lineages are retained as telemetry, not discarded
    conv = collapse.converged[p_lo.root_equivalence_hash]
    assert set(conv) == {"P:C:rho0.850:r1", "P:C:rho0.850:r9"}


def test_collapse_is_input_order_independent():
    p1 = _payload(root_id="a", x_t=(0, 1, 4, 4))
    p2 = _payload(root_id="b", x_t=(0, 2, 4, 4))
    p3 = _payload(root_id="c", x_t=(0, 1, 4, 4))  # equiv to p1
    forward = collapse_roots([p1, p2, p3])
    backward = collapse_roots([p3, p2, p1])
    assert [p.root_id for p in forward.unique] == [p.root_id for p in backward.unique]


def test_make_root_id_is_unique_per_lineage_index():
    assert make_root_id("P", "C", "rho0.850", 0) != make_root_id("P", "C", "rho0.850", 1)


# ------------------------- round-trip + provenance ------------------------- #
def test_payload_round_trips_without_loss():
    a = _payload()
    b = PartialRootPayload.from_dict(a.to_dict())
    assert b == a
    assert b.root_equivalence_hash == a.root_equivalence_hash
    assert b.snapshot_payload_hash == a.snapshot_payload_hash


def test_content_digest_detects_same_size_edit(tmp_path):
    p = tmp_path / "h_map.parquet"
    p.write_bytes(b"AAAA")
    d1 = content_digest(p)
    p.write_bytes(b"AABA")  # same size, different content
    d2 = content_digest(p)
    assert d1 != d2  # resume bound to content digest is invalidated


# --------------------------------------------------------------------------- #
# V1-A conditioning identity (PLAN §4.2, runbook §2): no h-map slot, no placeholders,
# and the frozen null runtime is PART of the identity.
# --------------------------------------------------------------------------- #
def test_conditioning_has_no_h_map_field():
    import dataclasses as _dc

    names = {f.name for f in _dc.fields(ConditioningDigest)}
    assert "h_map" not in names  # V1-A consumes no h-map: there is nothing to hash and no
    assert {"controller_enabled", "h_maps_present"} <= names  # ... placeholder to fake


def test_conditioning_rejects_placeholder_digests():
    with pytest.raises(ValueError):
        _cond(dplm_checkpoint="unset:dplm")
    with pytest.raises(ValueError):
        _cond(backbone_row="unset:backbone")
    with pytest.raises(ValueError):
        _cond(dplm_checkpoint="")


def test_conditioning_refuses_non_null_runtime():
    with pytest.raises(ValueError):
        _cond(controller_enabled=True)
    with pytest.raises(ValueError):
        _cond(h_maps_present=True)


def test_equivalence_hash_binds_the_null_runtime_and_anchor_policy():
    base = _payload()
    # a different anchor policy is a different conditioning identity, even at identical x_t
    other = _payload(cond=_cond(fixed_token_policy="manifest:abc123"))
    assert base.root_equivalence_hash != other.root_equivalence_hash
    # and the sentinels are inside the hashed envelope
    assert "controller_enabled" in base.conditioning.canonical()
    assert "h_maps_present" in base.conditioning.canonical()
