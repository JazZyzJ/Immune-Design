"""The Canary reader against a PAIRED bundle.

Every check here was written against the unpaired ladder, where a `committed` event always projected
and every fork seed was globally unique. A V2F5A qualification bundle breaks both premises:

* when the treatment arm stalls, the control slot holds the feedback-off A2 VIEW -- `committed`,
  because the view is a completed observation, but with no projection and therefore no transition
  coordinates at all;
* a matched treatment/control pair is REQUIRED to share realized fork seeds, and each arm's
  descendants hang off its own propagated state, so every matched fork appears as one seed under two
  `source_state_id`s.

Read literally, the reader called both a defect -- and `_check_protection_expiry` crashed on
`int(NaN)`, which cost the first V2F5A qualification its entire integrity verdict.
"""

from __future__ import annotations

import pandas as pd

from scripts.analysis.read_v2_canary import (
    _check_lineage,
    _check_protection_expiry,
    _check_replay,
)


def _transition_event(**over) -> dict:
    """One real committed transition: projected, propagated, coordinates present."""
    return {
        "transition_id": "t1", "outcome": "committed", "c_next_step": 60.0,
        "source_state_id": "src", "projected_state_id": "proj",
        "propagated_state_id": "prop", "selected_endpoint_id": "ep",
        "write_from_endpoint_json": "[0]", "inject_from_source_feedback_json": "[]",
        **over,
    }


def _view_event(**over) -> dict:
    """The feedback-off A2 view: committed, but it projected nothing."""
    return _transition_event(
        transition_id="null:src", c_next_step=float("nan"), projected_state_id=None,
        propagated_state_id=None, selected_endpoint_id=None, **over)


def _states() -> pd.DataFrame:
    return pd.DataFrame([
        {"state_id": "src", "layer": "live", "replay_state_hash": "h0",
         "parent_state_id": None, "origin_endpoint_id": None, "parent_transition_id": None,
         "temporary_protection_json": "[]", "expired_protection_json": "[]"},
        {"state_id": "proj", "layer": "projected", "replay_state_hash": None,
         "parent_state_id": "src", "origin_endpoint_id": "ep", "parent_transition_id": "t0",
         "temporary_protection_json": '[{"position": 3, "expiry_step": 60}]',
         "expired_protection_json": "[]"},
        {"state_id": "prop", "layer": "live", "replay_state_hash": "h1",
         "parent_state_id": "proj", "origin_endpoint_id": None, "parent_transition_id": "t0",
         "temporary_protection_json": "[]",
         "expired_protection_json": '[{"position": 3, "expiry_step": 60}]'},
    ])


def test_protection_expiry_excludes_the_view_row_instead_of_crashing():
    events = pd.DataFrame([_transition_event(), _view_event()])
    verdict = _check_protection_expiry(_states(), events)
    assert verdict["pass"], verdict
    # Visibly excluded, not quietly dropped -- otherwise a lost transition and a view row read the
    # same way in the artifact.
    assert "1 feedback-off view row(s) excluded" in verdict["detail"]


def test_lineage_excludes_the_view_row_rather_than_calling_it_a_break():
    events = pd.DataFrame([_transition_event(), _view_event()])
    verdict = _check_lineage(_states(), events)
    assert verdict["pass"], verdict
    assert "1 committed transition(s), 0 break(s)" in verdict["detail"]
    assert "1 feedback-off view row(s) excluded" in verdict["detail"]


def test_lineage_still_breaks_on_a_transition_whose_state_is_genuinely_absent():
    """The exclusion is keyed on "did not project", not on "has a null field"."""
    events = pd.DataFrame([_transition_event(propagated_state_id="gone")])
    verdict = _check_lineage(_states(), events)
    assert not verdict["pass"]
    assert "propagated_state_id" in verdict["detail"]


def _endpoints(rows) -> pd.DataFrame:
    return pd.DataFrame([
        {"replay_state_hash": "h", "source_state_id": s, "replay_fork_seed": seed}
        for s, seed in rows
    ])


def test_replay_accepts_a_matched_pair_sharing_one_fork_seed():
    # Two arms, two forks each: seeds repeat ACROSS the arms' propagated states and are distinct
    # within each. This is the §9.0 matching requirement, not a collision.
    endpoints = _endpoints([("prop_a", 11), ("prop_a", 22), ("prop_b", 11), ("prop_b", 22)])
    verdict = _check_replay(_states(), endpoints)
    assert verdict["pass"], verdict
    assert "0 within-state collision(s)" in verdict["detail"]
    assert "2 shared by a matched pair" in verdict["detail"]


def test_replay_fails_when_two_forks_of_one_state_collide():
    """The collision the check exists for: one state's own forks cannot share a seed."""
    endpoints = _endpoints([("prop_a", 11), ("prop_a", 11)])
    verdict = _check_replay(_states(), endpoints)
    assert not verdict["pass"], verdict
    assert "1 within-state collision(s)" in verdict["detail"]


def test_replay_fails_when_a_seed_reaches_three_states():
    """A matched pair is TWO. Three is a shape no legal run produces, so the width is bounded."""
    endpoints = _endpoints([("prop_a", 11), ("prop_b", 11), ("prop_c", 11)])
    verdict = _check_replay(_states(), endpoints)
    assert not verdict["pass"], verdict
    assert "1 shared by 3+ states" in verdict["detail"]
