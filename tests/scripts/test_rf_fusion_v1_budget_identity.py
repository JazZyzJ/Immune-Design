"""The §3.4 launch gate must project what the run will actually spend, and the Terminal arm's
budget must be bound to the experiment that produced it.

Two distinct silent failures live here:

* **an under-projection is a gate that does not fire.** `max_dfe` / `max_refolds` /
  `max_walltime_s` are checked ONCE, before launch. If the projection is smaller than the run, the
  declared caps are simply not enforced and the run blows through them with exit 0.
* **an unbound reservation is a Terminal arm on someone else's budget.** `C_reserved` is the ONLY
  input that makes the two arms compute-matched. A manifest from a different campaign, split, or
  config resolves and runs, and every artifact reports a clean run at the wrong scale.
"""

from __future__ import annotations

from scripts.rf_fusion_v1_cohort import _checkpoint_name

import json

import pytest

from scripts.rf_fusion_v1_preflight import (
    TerminalFusionParams,
    project_budget,
    render_print_config,
    resolve_entry_config,
)
from tests._v1_fixtures import p1_terminal_config as terminal_config, t0_config

_TP = TerminalFusionParams(s_steps=10, r_parent=8, n_rounds=8,
                           seconds_per_refold=1.0, seconds_per_dfe=0.1)


def _t0(**over):
    base = dict(prefix_attempts=4, k_est=2, k_eval=2, unique_root_capacity=3, q_t0=2,
                n_population=2, initial_refold_attempt_cap=2)
    base.update(over)
    return resolve_entry_config(t0_config(**base))


# --------------------------------------------------------------------------- #
# T0: the projection covers the WHOLE grid, and the control it pays for
# --------------------------------------------------------------------------- #


def test_the_t0_projection_scales_with_the_number_of_maturities():
    """The cohort runs the whole `rho_grid` per protein in one shard. A one-point projection
    under-fires the §3.4 gate by exactly `len(rho_grid)`."""
    one = project_budget(_t0(rho_grid=(0.85,)), n_proteins=1, terminal=_TP)
    three = project_budget(_t0(rho_grid=(0.80, 0.85, 0.90)), n_proteins=1, terminal=_TP)

    assert three.reserved_dfe_total == 3 * one.reserved_dfe_total
    assert three.reserved_refold_total == 3 * one.reserved_refold_total


def test_the_t0_projection_includes_the_compute_matched_full_control():
    """`run_t0_protein` spends `matched_full * S` on the independent-full control ON TOP of the
    prefix/est/eval work -- that control is the point of T0's condition 3. Omitting it from the
    projection understates the reservation by up to a factor of two."""
    cfg = _t0(rho_grid=(0.85,))
    projected = project_budget(cfg, n_proteins=1, terminal=_TP)
    partial = (cfg.prefix_attempts * _TP.s_steps
               + cfg.k_est * cfg.prefix_attempts * _TP.s_steps
               + cfg.k_eval * cfg.prefix_attempts * _TP.s_steps)
    assert projected.reserved_dfe_total >= 2 * partial


def test_the_t0_projection_is_not_smaller_than_what_the_run_actually_spends(tmp_path):
    """End-to-end: project, then RUN, then compare the ledger. A projection below the real spend is
    a declared cap that was never enforced."""
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
    from inverse_folding.reference_flow.fusion.v1_ledger import LedgerEvent, aggregate_ledger
    from scripts.rf_fusion_v1_cohort import EntryOracles, run_entry_shard

    from tests.scripts.test_rf_fusion_v1_t0 import _S, _World

    cfg = _t0(rho_grid=(0.80, 0.85, 0.90))
    tp = TerminalFusionParams(s_steps=_S, r_parent=8, n_rounds=8,
                              seconds_per_refold=1.0, seconds_per_dfe=0.1)
    projected = project_budget(cfg, n_proteins=1, terminal=tp)

    world = _World(cfg)
    run_entry_shard(
        shard_proteins=["P"], expected_length_by_protein={"P": 4},
        input_signature_by_protein={"P": "sig"}, config=cfg,
        oracles=EntryOracles(world.gen, world.complete, world.head,
                             lambda c: StructureOutcome(feasible=True),
                             trajectory_generator=world.full),
        out_dir=tmp_path, s_steps=_S,
    )
    ckpt = json.loads((tmp_path / "checkpoints" / _checkpoint_name("P", "preterminal", "t0")).read_text())
    totals = aggregate_ledger([LedgerEvent(**row) for row in ckpt["ledger"]])

    spent = int(totals["logical_dfe"])
    assert spent <= projected.reserved_dfe_total, (
        f"the run spent {spent} DFE against a projected reservation of "
        f"{projected.reserved_dfe_total}: the §3.4 cap was never enforced"
    )
    assert int(totals["structure_requests"]) <= projected.reserved_refold_total


# --------------------------------------------------------------------------- #
# Terminal: print-config, and a cohort total that is not a single protein's
# --------------------------------------------------------------------------- #


def test_print_config_resolves_for_the_terminal_arm_when_the_reservation_is_known():
    """Runbook §11.4 requires `--print-config` to resolve for BOTH arms; `V1_SUBMODE=print_config`
    is how a cluster operator audits a launch before spending anything."""
    cfg = resolve_entry_config(terminal_config(initial_refold_attempt_cap=2, n_population=2))
    payload = render_print_config(
        cfg, n_proteins=1, terminal=_TP, reserved_dfe_per_protein=40,
    )
    assert payload["entry_arm"] == "terminal"
    assert payload["budget"]["terminal_trajectories"] == 4


def test_the_terminal_cohort_projection_sums_the_per_protein_reservations():
    """Each protein runs `M_T = floor(C_reserved_i / S)` trajectories from ITS OWN reservation, so a
    cohort total built from one scalar under-reports whenever the proteins differ."""
    from scripts.run_rf_fusion_v1_entry import project_terminal_cohort_budget

    cfg = resolve_entry_config(terminal_config(initial_refold_attempt_cap=2, n_population=2))
    budget = project_terminal_cohort_budget(
        cfg, reserved_by_protein={"A": 40, "B": 400}, terminal=_TP,
    )
    assert budget.reserved_dfe_total == 440


# --------------------------------------------------------------------------- #
# Terminal: the reservation is bound to the experiment that produced it
# --------------------------------------------------------------------------- #


def _manifest(tmp_path, **over):
    payload = {
        "arm": "preterminal", "campaign_id": "camp", "phase": "p1", "split_role": "p1_dev",
        "reserved_dfe_by_protein": {"P": 40},
        "reservation_detail": {"P": {"reserved_dfe": 40, "f_cap": 2, "k_est": 2}},
    }
    payload.update(over)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize("field,value", [
    ("campaign_id", "a_different_campaign"),
    ("phase", "t0"),
    ("split_role", "p1_holdout"),
])
def test_a_reservation_from_a_different_experiment_is_refused(tmp_path, field, value):
    from scripts.run_rf_fusion_v1_entry import load_preterminal_reservation

    cfg = resolve_entry_config(terminal_config(initial_refold_attempt_cap=2, n_population=2))
    path = _manifest(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=field):
        load_preterminal_reservation(path, ["P"], config=cfg)


def test_a_reservation_frozen_under_a_different_f_cap_is_refused(tmp_path):
    """`M_T >= F_cap` is the invariant that keeps the two arms' refold budgets equal. A reservation
    frozen under a different `F_cap` breaks it silently."""
    from scripts.run_rf_fusion_v1_entry import load_preterminal_reservation

    cfg = resolve_entry_config(terminal_config(initial_refold_attempt_cap=2, n_population=2))
    path = _manifest(tmp_path, reservation_detail={"P": {"reserved_dfe": 40, "f_cap": 5}})
    with pytest.raises(ValueError, match="f_cap"):
        load_preterminal_reservation(path, ["P"], config=cfg)


def test_a_matching_reservation_is_accepted(tmp_path):
    from scripts.run_rf_fusion_v1_entry import load_preterminal_reservation

    cfg = resolve_entry_config(terminal_config(initial_refold_attempt_cap=2, n_population=2))
    assert load_preterminal_reservation(_manifest(tmp_path), ["P"], config=cfg) == {"P": 40}


def test_editing_the_reservation_in_place_invalidates_resume(tmp_path):
    """PLAN §4.2:713 -- "bind resume to content digests, not path/size/mtime fingerprints alone".
    A reservation is an INPUT: changing C_reserved changes M_T, so a checkpoint frozen under the
    old value must not be reused."""
    from scripts.run_rf_fusion_v1_entry import reservation_content_digest

    a = _manifest(tmp_path, reserved_dfe_by_protein={"P": 40})
    first = reservation_content_digest(a)
    a.write_text(json.dumps({
        "arm": "preterminal", "campaign_id": "camp", "phase": "p1", "split_role": "p1_dev",
        "reserved_dfe_by_protein": {"P": 400},
        "reservation_detail": {"P": {"reserved_dfe": 400, "f_cap": 2}},
    }))
    assert reservation_content_digest(a) != first


# --------------------------------------------------------------------------- #
# The reservation must bind the SCIENTIFIC SUBSTRATE, not just the labels
# --------------------------------------------------------------------------- #


def _substrate(**over):
    base = {
        "s_steps": 10,
        "input_signature": "sig-abc",
        "file_digests": {"head_checkpoint": "H0", "dplm_checkpoint": "D0",
                         "fusion_config": "F0", "cohort_structures": "B0"},
        "entry_rf_config": "c1_null.yaml",
        "controller_enabled": False,
        "h_maps_present": False,
    }
    base.update(over)
    return base


def _manifest_with_substrate(tmp_path, **over):
    payload = {
        "arm": "preterminal", "campaign_id": "camp", "phase": "p1", "split_role": "p1_dev",
        "reserved_dfe_by_protein": {"P": 40},
        "reservation_detail": {"P": {"reserved_dfe": 40, "f_cap": 2}},
        "substrate": _substrate(**over),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_the_manifest_records_the_substrate_the_reservation_was_frozen_under(tmp_path):
    """Runbook §2:118 and §9:471-477 say the manifest MUST record the resolved kernel identity and
    the input hashes. Without them the two arms cannot be shown to share a Head, a generator or a
    backbone -- and that shared substrate is what makes the allocation point the ONLY variable."""
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
    from scripts.rf_fusion_v1_cohort import (
        EntryOracles,
        aggregate_entry_artifacts,
        run_entry_shard,
    )

    from tests.scripts.test_rf_fusion_v1_entry_core import _World as _P1World
    from tests.scripts.test_rf_fusion_v1_entry_core import _cfg as _p1_cfg, _payload

    cfg = _p1_cfg()
    payloads = [_payload((5 + i, 6, 7, 8), f"r{i}") for i in range(4)]
    world = _P1World(payloads, cfg, est_risk={i: 0.1 * (i + 1) for i in range(4)},
                     final_risk={i: 0.5 for i in range(4)})
    run_entry_shard(
        shard_proteins=["P"], expected_length_by_protein={"P": 4},
        input_signature_by_protein={"P": "sig-abc"}, config=cfg,
        oracles=EntryOracles(world.gen, world.complete, world.head,
                             lambda c: StructureOutcome.deferred("v0")),
        out_dir=tmp_path,
    )
    manifest = aggregate_entry_artifacts(
        out_dir=tmp_path, requested_cohort=["P"], config=cfg,
        input_signature_by_protein={"P": "sig-abc"},
        substrate=_substrate(),
    )
    sub = manifest["substrate"]
    assert sub["s_steps"] == 10
    assert sub["file_digests"]["head_checkpoint"] == "H0"
    assert sub["entry_rf_config"] == "c1_null.yaml"
    assert sub["controller_enabled"] is False
    assert sub["h_maps_present"] is False


@pytest.mark.parametrize("field,value,match", [
    ("s_steps", 5, "s_steps"),
    ("entry_rf_config", "c1_linear.yaml", "entry_rf_config"),
    ("controller_enabled", True, "controller_enabled"),
])
def test_a_reservation_frozen_under_a_different_substrate_is_refused(tmp_path, field, value, match):
    """A Terminal shard consuming C_reserved frozen under a different S runs a DIFFERENT number of
    trajectories: S=5 against a reservation frozen at S=10 doubles M_T. Nothing downstream can see
    it -- every artifact is internally consistent at the wrong scale."""
    from scripts.run_rf_fusion_v1_entry import load_preterminal_reservation

    cfg = resolve_entry_config(terminal_config(initial_refold_attempt_cap=2, n_population=2))
    path = _manifest_with_substrate(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=match):
        load_preterminal_reservation(path, ["P"], config=cfg, substrate=_substrate())


@pytest.mark.parametrize("key", ["head_checkpoint", "dplm_checkpoint", "cohort_structures"])
def test_a_reservation_frozen_under_a_different_model_or_backbone_is_refused(tmp_path, key):
    from scripts.run_rf_fusion_v1_entry import load_preterminal_reservation

    cfg = resolve_entry_config(terminal_config(initial_refold_attempt_cap=2, n_population=2))
    digests = dict(_substrate()["file_digests"])
    digests[key] = "DIFFERENT"
    path = _manifest_with_substrate(tmp_path, file_digests=digests)
    with pytest.raises(ValueError, match=key):
        load_preterminal_reservation(path, ["P"], config=cfg, substrate=_substrate())


def test_a_matching_substrate_is_accepted(tmp_path):
    from scripts.run_rf_fusion_v1_entry import load_preterminal_reservation

    cfg = resolve_entry_config(terminal_config(initial_refold_attempt_cap=2, n_population=2))
    path = _manifest_with_substrate(tmp_path)
    assert load_preterminal_reservation(
        path, ["P"], config=cfg, substrate=_substrate()) == {"P": 40}


def test_a_reservation_with_no_recorded_substrate_is_refused(tmp_path):
    """A pre-V1 manifest cannot prove it shares this run's substrate. Accepting it because the key
    is absent would make the whole check opt-out."""
    from scripts.run_rf_fusion_v1_entry import load_preterminal_reservation

    cfg = resolve_entry_config(terminal_config(initial_refold_attempt_cap=2, n_population=2))
    assert load_preterminal_reservation(_manifest(tmp_path), ["P"], config=cfg) == {"P": 40}
    with pytest.raises(ValueError, match="substrate"):
        load_preterminal_reservation(
            _manifest(tmp_path), ["P"], config=cfg, substrate=_substrate())


@pytest.mark.parametrize("field", [
    "head_variant_id", "head_allele_idx", "head_window_k_min", "head_window_k_max", "allele",
])
def test_a_reservation_frozen_under_a_different_head_inference_config_is_refused(tmp_path, field):
    """Runbook §1 requires both arms to share the "frozen runtime Head checkpoint AND INFERENCE
    CONFIGURATION"; PLAN §4.2 spells it out as "Head checkpoint, variant, allele index,
    inference/window config". The checkpoint digest alone does not cover any of them: windowing
    decides WHICH sub-sequences are scored, and the allele index decides WHICH head output is read,
    so two arms can run the same checkpoint and still rank on different signals."""
    from scripts.run_rf_fusion_v1_entry import load_preterminal_reservation

    cfg = resolve_entry_config(terminal_config(initial_refold_attempt_cap=2, n_population=2))
    theirs = _substrate()
    theirs[field] = "CHANGED" if isinstance(theirs.get(field), (str, type(None))) else 999
    path = _manifest_with_substrate(tmp_path, **{field: theirs[field]})
    with pytest.raises(ValueError, match=field):
        load_preterminal_reservation(path, ["P"], config=cfg, substrate=_substrate())
