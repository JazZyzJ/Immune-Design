"""V1F5 cost-ledger tests (PLAN_RF_REFINE_FUSION_V1 §3.3).

A logical method event (event_id) is counted ONCE regardless of physical retries; physical work
sums per distinct attempt_id; a resume that re-emits committed events must not double-count; a
conflicting second commit of the same logical event is a hard error; totals split by phase.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion.v1_ledger import LedgerEvent, aggregate_ledger


def _e(event_id, attempt_id, status, **kw):
    base = dict(protein_id="P", arm="C", phase="est", logical_dfe=0,
                physical_forward_calls=0, head_samples=0, structure_requests=0,
                structure_cache_hits=0, walltime_s=0.0)
    base.update(kw)
    return LedgerEvent(event_id=event_id, attempt_id=attempt_id, status=status, **base)


def test_logical_counted_once_physical_per_attempt():
    events = [
        _e("e1", "a1", "failed", logical_dfe=0, physical_forward_calls=5),
        _e("e1", "a2", "ok", logical_dfe=7, physical_forward_calls=7, head_samples=1),
    ]
    agg = aggregate_ledger(events)
    assert agg["logical_dfe"] == 7  # logical counted once (from the committed attempt)
    assert agg["physical_forward_calls"] == 12  # retry adds physical work
    assert agg["head_samples"] == 1
    assert agg["n_logical_events"] == 1 and agg["n_physical_attempts"] == 2


def test_resume_re_emission_does_not_double_count():
    e = _e("e1", "a1", "ok", logical_dfe=7, physical_forward_calls=7, head_samples=1)
    agg = aggregate_ledger([e, e, e])  # resume re-emits the same committed event
    assert agg["logical_dfe"] == 7 and agg["physical_forward_calls"] == 7


def test_conflicting_commit_is_a_hard_error():
    events = [
        _e("e1", "a1", "ok", logical_dfe=7),
        _e("e1", "a2", "ok", logical_dfe=9),  # same logical event, different committed value
    ]
    with pytest.raises(ValueError):
        aggregate_ledger(events)


def test_totals_split_by_phase():
    events = [
        _e("e1", "a1", "ok", phase="root_prefix", logical_dfe=3),
        _e("e2", "a2", "ok", phase="est", logical_dfe=7, head_samples=8),
        _e("e3", "a3", "ok", phase="est", logical_dfe=5, head_samples=4),
    ]
    agg = aggregate_ledger(events)
    assert agg["by_phase"]["root_prefix"]["logical_dfe"] == 3
    assert agg["by_phase"]["est"]["logical_dfe"] == 12
    assert agg["by_phase"]["est"]["head_samples"] == 12


def test_ledger_event_rejects_bad_phase_and_status():
    with pytest.raises(ValueError):
        _e("e1", "a1", "ok", phase="bogus")
    with pytest.raises(ValueError):
        _e("e1", "a1", "maybe")


# --------------------------------------------------------------------------- #
# Step 1: cost-integrity primitives (PLAN §3.3)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,bad", [
    ("logical_dfe", -1), ("head_samples", -1), ("structure_requests", -1),
    ("physical_forward_calls", -1), ("structure_cache_hits", -1),
    ("completion_cache_hits", -1), ("walltime_s", -0.1), ("walltime_s", float("nan")),
])
def test_ledger_event_rejects_negative_or_nonfinite_counters(field, bad):
    with pytest.raises(ValueError):
        _e("e1", "a1", "ok", **{field: bad})


@pytest.mark.parametrize("field", ["event_id", "attempt_id"])
def test_ledger_event_rejects_empty_identity(field):
    kw = {"event_id": "e1", "attempt_id": "a1"}
    kw[field] = ""
    with pytest.raises(ValueError):
        LedgerEvent(protein_id="P", arm="C", phase="est", status="ok", **kw)


def test_conflicting_physical_values_for_one_attempt_is_a_hard_error():
    # One physical execution must have ONE physical cost. A second row for the same attempt_id
    # carrying a different physical count is a corrupted ledger, not a silent last-write-wins.
    events = [
        _e("e1", "a1", "ok", physical_forward_calls=1),
        _e("e1", "a1", "ok", physical_forward_calls=99),
    ]
    with pytest.raises(ValueError):
        aggregate_ledger(events)


def test_byte_identical_reemission_still_dedups():
    e = _e("e1", "a1", "ok", logical_dfe=7, physical_forward_calls=7)
    agg = aggregate_ledger([e, e, e])
    assert agg["logical_dfe"] == 7 and agg["physical_forward_calls"] == 7


def test_deferred_status_contributes_physical_but_no_logical_cost():
    # A deferred attempt (e.g. structure deferred to v0) stays visible and may carry real
    # walltime, but must never contribute logical method cost.
    events = [_e("e1", "a1", "deferred", logical_dfe=5, physical_forward_calls=3, walltime_s=1.5)]
    agg = aggregate_ledger(events)
    assert agg["logical_dfe"] == 0          # not committed
    assert agg["physical_forward_calls"] == 3  # but really spent
    assert agg["walltime_s"] == 1.5


def test_one_attempt_cannot_serve_two_logical_events():
    # A physical execution belongs to exactly ONE logical event. Letting one attempt_id hang off
    # two event_ids would report 2 logical events for 1 physical attempt -- free method work.
    events = [
        _e("e1", "a1", "ok", logical_dfe=5),
        _e("e2", "a1", "ok", logical_dfe=5),  # same attempt, different logical event
    ]
    with pytest.raises(ValueError):
        aggregate_ledger(events)


def test_one_event_id_cannot_span_two_proteins():
    # Identical logical values must not let an event_id silently merge across proteins: that
    # would erase one protein's method cost entirely.
    events = [
        _e("e1", "a1", "ok", protein_id="P1", logical_dfe=5),
        _e("e1", "a2", "ok", protein_id="P2", logical_dfe=5),
    ]
    with pytest.raises(ValueError):
        aggregate_ledger(events)
