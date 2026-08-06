"""The per-protein V2 hotspot calibration producer, against the FROZEN law (runbook §2.1).

The previous version of this suite pinned an earlier contract -- a `--pairs` JSONL of pre-scored
Head payloads, a free choice among five statistics, `numpy.quantile` interpolation, no structure
gate and no floor. All ten of its tests passed, and every one of them passed against the wrong
program: the runbook had since frozen a different law. This suite pins the law the runbook states.
"""

from __future__ import annotations

import json
import types

import pytest

from scripts.calibrate_rf_fusion_v2_hotspot import (
    SEED_NAMESPACE,
    THRESHOLD_STATISTIC,
    CalibrationSeams,
    V2HotspotCalibrationError,
    build_parser,
    calibrate_protein,
    main,
    q90_higher_order_statistic,
    source_id_for,
)


# --------------------------------------------------------------------------------------------
# step 5: the order statistic is an ORDER statistic
# --------------------------------------------------------------------------------------------


def test_q90_is_the_higher_order_statistic_at_rank_ceil_of_point_nine_n():
    """Runbook §2.1 step 5: "the value at one-indexed rank `ceil(0.90*n)`"."""
    values = [float(i) for i in range(1, 11)]          # n=10 -> rank 9 -> the 9th smallest
    assert q90_higher_order_statistic(values) == (9.0, 9)
    assert q90_higher_order_statistic([float(i) for i in range(1, 65)]) == (58.0, 58)


def test_the_threshold_is_a_value_some_endpoint_actually_produced():
    """`numpy.quantile` interpolates linearly between order statistics and returns a number no
    endpoint produced.  A threshold is a bound on measured designs, so it has to BE one of them."""
    import numpy as np

    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    chosen, _rank = q90_higher_order_statistic(values)
    assert chosen in values
    assert chosen != float(np.quantile(np.asarray(values), 0.90))


def test_the_full_floating_point_value_is_preserved():
    """§2.1: "do not round it before computing `source_ref`" -- a rounded value would make the
    digest describe a different number than the one enforced."""
    values = [0.1234567890123456, 0.9876543210987654]
    chosen, _ = q90_higher_order_statistic(values)
    assert repr(chosen) == repr(0.9876543210987654)


def test_an_empty_retained_set_is_refused():
    with pytest.raises(V2HotspotCalibrationError, match="no retained endpoints"):
        q90_higher_order_statistic([])


# --------------------------------------------------------------------------------------------
# the frozen interface
# --------------------------------------------------------------------------------------------


def test_the_statistic_is_a_closed_enum_with_exactly_one_admissible_value():
    """§2.1: "a closed enum for this Canary, not a free-form label"."""
    action = next(a for a in build_parser()._actions if a.dest == "threshold_statistic")
    assert tuple(action.choices) == (THRESHOLD_STATISTIC,)
    assert action.required is True


def test_no_independent_quantile_override_exists():
    """§2.1: they "would create two conflicting sources of truth"."""
    dests = {a.dest for a in build_parser()._actions}
    assert "quantile" not in dests and "quantile_method" not in dests


def test_the_frozen_interface_requires_every_flag_the_runbook_names():
    required = {a.dest for a in build_parser()._actions if getattr(a, "required", False)}
    assert {"protein_id", "n_completions", "master_seed", "seed_namespace",
            "threshold_statistic", "min_definitive_feasible", "checkpoint", "rf_config",
            "test_set", "pdb_root", "complete_reference_manifest", "head_config_dir",
            "head_checkpoint", "structure_config", "out_rows", "out_json",
            "code_revision"} <= required


def test_the_source_id_is_protein_specific():
    """The two proteins receive separate thresholds, so they receive separate identities."""
    assert source_id_for("Q00511") == "v2-canary-hotspot-null-q90-higher-v1:Q00511"
    assert source_id_for("5ZHV_B") != source_id_for("Q00511")


# --------------------------------------------------------------------------------------------
# steps 1-4, driven end to end over injected seams
# --------------------------------------------------------------------------------------------

AA = "ACDEFGHIKLMNPQRSTVWY"


def _windows(zs):
    return tuple(types.SimpleNamespace(start_0b=i, end_0b=i + 3, k=4, z=float(z))
                 for i, z in enumerate(zs))


def _identity():
    from inverse_folding.reference_flow.fusion_v2.identity import HeadEvaluatorIdentity

    return HeadEvaluatorIdentity(allele="DRB1_0701", score_scale="nats", window_k_min=4,
                                 window_k_max=4, head_config_hash="a" * 64,
                                 head_checkpoint_digest="b" * 64)


class _Scorer:
    """A Head whose z on the middle window is the sequence's own index -- so N_H is controllable."""

    def evaluator_identity(self):
        return _identity()

    def score_reference(self, *, protein_id, sequence):
        from inverse_folding.reference_flow.fusion.state import sequence_md5

        z = float(AA.index(sequence[-1])) if sequence[-1] in AA else 0.0
        return types.SimpleNamespace(
            protein_id=protein_id, sequence_md5=sequence_md5(sequence),
            sequence_length=len(sequence), allele="DRB1_0701", score_scale="nats",
            windows=_windows([0.0, z, 0.0]), residue_hotspot=(0.0,) * len(sequence),
            global_risk=z)


class _Model:
    alphabet = {i: c for i, c in enumerate(AA)}
    aa_token_ids = frozenset(range(len(AA)))

    def fixed_tokens(self, protein_id):
        return None


def _seams(*, n_ok=64, structure_ok=None):
    """Completion ``i`` ends in ``AA[i % 20]``, so its N_H is ``i % 20``."""
    def generate(*, model, protein_id, seed):
        return "ACDEF" + AA[(seed % 1000) % 20]

    def structure(*, protein_id, sequence):
        ok = structure_ok(sequence) if structure_ok else True
        return types.SimpleNamespace(evaluated=True, feasible=ok, metrics={"scTM": 0.9})

    return {
        "head_scorer": _Scorer(), "structure_gate": structure,
        "generate_completion": generate,
        "derive_seed": lambda *fields: sum(ord(c) for c in "".join(map(str, fields))) % 1000,
        "build_model_factory": None,
    }


def _run_law(**over):
    kw = dict(protein_id="5ZHV_B", n_completions=64, min_definitive_feasible=48,
              master_seed=20260806, reference_sequence="ACDEFA", reference_digest="c" * 64,
              head_identity=_identity(), model=_Model(), seams=_seams())
    kw.update(over)
    return calibrate_protein(**kw)


def test_the_law_retains_only_definitively_feasible_endpoints():
    """§2.1 step 3: "cache-only labels without a resolved verdict do not count"."""
    def unresolved(*, protein_id, sequence):
        return types.SimpleNamespace(evaluated=False, feasible=True, metrics={})

    seams = _seams()
    seams["structure_gate"] = unresolved
    with pytest.raises(V2HotspotCalibrationError, match="below the frozen floor"):
        _run_law(seams=seams)


def test_the_floor_is_not_relaxed_and_no_artifact_is_produced(tmp_path):
    """§2.1 step 4: "Below this floor, write no calibration artifact and do not relax the floor"."""
    seams = _seams(structure_ok=lambda sequence: sequence.endswith(("A", "C", "D")))
    with pytest.raises(V2HotspotCalibrationError) as excinfo:
        _run_law(seams=seams)
    assert "48" in str(excinfo.value)


def test_every_attempt_is_a_row_including_the_rejected_ones():
    """A table holding only the retained endpoints makes the definitive-feasible RATE
    unrecoverable -- and that rate is exactly what the floor is checked against."""
    seams = _seams(structure_ok=lambda sequence: sequence[-1] != "B")
    rows, summary = _run_law(seams=seams)
    assert len(rows) == 64
    assert summary["n_attempted"] == 64
    assert summary["n_definitive_feasible"] == sum(1 for r in rows if r["status"] == "retained")


def test_the_summary_reports_the_distribution_not_only_the_scalar():
    """§2.1: "A producer that outputs only the chosen scalar is incomplete"."""
    _rows, summary = _run_law()
    assert {"q50", "q90_higher", "q95", "max", "order_statistic_rank", "n_attempted",
            "n_definitive_feasible", "failure_counts"} <= set(summary)


def test_the_canonical_row_carries_every_field_the_runbook_signs():
    rows, _summary = _run_law()
    retained = next(row for row in rows if row["status"] == "retained")
    assert {"protein_id", "replicate_index", "seed", "sequence_md5", "reference_digest",
            "head_evaluator_digest", "window_grid_digest", "n_h_whole", "structure_evaluated",
            "structure_feasible", "structure_metrics_json", "anchor_verdict"} <= set(retained)


def test_an_anchor_mismatch_is_a_hard_failure_not_a_retained_endpoint():
    """§2.1 step 2 lists anchor mismatch among the hard failures."""
    class _Anchored(_Model):
        def fixed_tokens(self, protein_id):
            return {0: AA.index("W")}          # every completion starts with 'A', so all mismatch

    with pytest.raises(V2HotspotCalibrationError, match="below the frozen floor"):
        _run_law(model=_Anchored())


def test_the_seed_namespace_is_the_disjoint_calibration_one():
    """Calibration draws must never collide with the Canary's own sampling streams."""
    seen = []
    seams = _seams()
    seams["derive_seed"] = lambda *fields: (seen.append(fields), 7)[1]
    _run_law(seams=seams)
    assert all(fields[0] == SEED_NAMESPACE for fields in seen)


def test_the_measurement_is_the_admission_gates_own_comparator():
    """A second implementation of `N_H^whole` would calibrate the threshold against a quantity the
    gate does not compute."""
    from inverse_folding.reference_flow.fusion_v2.safety import (
        ReferenceKind,
        whole_landscape_new_hotspot,
    )

    seams = _seams()
    rows, _summary = _run_law(seams=seams)
    retained = next(row for row in rows if row["status"] == "retained")
    # Re-derived from the row's OWN realized seed rather than a guessed one: a test that guessed
    # would drift the moment the seed law changed and would then be asserting nothing.
    sequence = seams["generate_completion"](
        model=_Model(), protein_id="5ZHV_B", seed=retained["seed"])
    scorer = _Scorer()
    direct = whole_landscape_new_hotspot(
        scorer.score_reference(protein_id="5ZHV_B", sequence=sequence),
        scorer.score_reference(protein_id="5ZHV_B", sequence="ACDEFA"),
        endpoint_id="endpoint:x", head_identity=_identity(),
        reference_kind=ReferenceKind.CUMULATIVE_DEPTH0, reference_binding_id="ref:x")
    assert retained["n_h_whole"] == pytest.approx(direct.max_increase)
