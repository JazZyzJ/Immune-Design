"""The V2 whole-landscape hotspot calibration producer (PLAN §2.7, §2.5).

The schema existed and the producer did not, so a cluster agent could only invent
`delta_new_cumulative` — exactly what PLAN §2.5 forbids. This suite pins the two properties that
make the artifact trustworthy: the threshold is measured through the GATE's own comparator, and the
statistic that reduces the distribution to a scalar is DECLARED rather than chosen by the script.
"""

from __future__ import annotations

import json

import pytest

from scripts.calibrate_rf_fusion_v2_hotspot import (
    THRESHOLD_STATISTICS,
    V2HotspotCalibrationError,
    build_parser,
    main,
    measure_pairs,
    reduce_to_threshold,
)


def _windows(zs):
    return [{"start_0b": i, "end_0b": i + 3, "k": 4, "z": float(z)} for i, z in enumerate(zs)]


def _score(sequence, zs, protein_id="5ZHV_B"):
    from inverse_folding.reference_flow.fusion.state import sequence_md5

    return {"protein_id": protein_id, "sequence_md5": sequence_md5(sequence),
            "sequence_length": len(sequence), "allele": "DRB1_0701", "score_scale": "nats",
            "windows": _windows(zs), "residue_hotspot": [0.0] * len(sequence),
            "global_risk": float(sum(zs))}


def _identity():
    from inverse_folding.reference_flow.fusion_v2.identity import HeadEvaluatorIdentity

    return HeadEvaluatorIdentity(allele="DRB1_0701", score_scale="nats", window_k_min=4,
                                 window_k_max=4, head_config_hash="a" * 64,
                                 head_checkpoint_digest="b" * 64)


def _pairs(deltas):
    """One calibration pair per delta: the design's worst window is `delta` above the reference."""
    rows = []
    for index, delta in enumerate(deltas):
        rows.append({
            "pair_id": f"p{index}",
            "reference": _score("ACDEFG", [0.0, 0.0, 0.0]),
            "design": _score("ACDEFG" if delta == 0 else f"ACDEF{'GHIKLMNPQ'[index % 9]}",
                             [0.0, float(delta), 0.0]),
        })
    return rows


# --------------------------------------------------------------------------------------------
# the statistic is DECLARED, never chosen here
# --------------------------------------------------------------------------------------------


def test_the_threshold_statistic_is_required_with_no_default():
    """`doc/FUSION_V2.md` leaves the choice open, so the script records it rather than making it.

    A default would be the script silently freezing a scientific decision PLAN §2.5 reserves for
    the runbook.
    """
    parser = build_parser()
    action = next(a for a in parser._actions if a.dest == "threshold_statistic")
    assert action.required is True
    assert action.default is None
    assert tuple(action.choices) == THRESHOLD_STATISTICS


def test_an_unimplemented_statistic_is_refused_rather_than_approximated():
    with pytest.raises(V2HotspotCalibrationError, match="q50"):
        reduce_to_threshold([1.0, 2.0], statistic="q50")


def test_each_declared_statistic_reduces_the_distribution_it_names():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert reduce_to_threshold(values, statistic="max") == 4.0
    assert reduce_to_threshold(values, statistic="q90") == pytest.approx(3.6)
    assert reduce_to_threshold(values, statistic="q95") == pytest.approx(3.8)
    assert reduce_to_threshold(values, statistic="max") >= \
        reduce_to_threshold(values, statistic="q99")


def test_an_empty_calibration_set_is_refused():
    """A threshold reduced from no measurements is an invented number wearing a measurement's
    provenance."""
    with pytest.raises(V2HotspotCalibrationError, match="no calibration measurements"):
        reduce_to_threshold([], statistic="max")


# --------------------------------------------------------------------------------------------
# the measurement is the GATE's own
# --------------------------------------------------------------------------------------------


def test_the_measurement_is_the_admission_gates_own_comparator():
    """A threshold calibrated with a second implementation of `N_H^whole` would be calibrated
    against a quantity the gate does not compute.

    Checked by re-deriving one pair through `whole_landscape_new_hotspot` directly and requiring
    byte-equal evidence.
    """
    from inverse_folding.reference_flow.fusion_v2.safety import (
        ReferenceKind,
        whole_landscape_new_hotspot,
    )
    from scripts.calibrate_rf_fusion_v2_hotspot import _head_score

    rows = _pairs([2.5])
    measured = measure_pairs(rows, head_identity=_identity(),
                             reference_binding_id="ref:calibration")
    direct = whole_landscape_new_hotspot(
        _head_score(rows[0]["design"]), _head_score(rows[0]["reference"]),
        endpoint_id="endpoint:p0", head_identity=_identity(),
        reference_kind=ReferenceKind.CUMULATIVE_DEPTH0, reference_binding_id="ref:calibration")
    assert measured[0]["max_increase"] == direct.max_increase == pytest.approx(2.5)


def test_a_pair_the_gate_cannot_measure_stops_the_calibration():
    """A window grid the two scores do not share is not a measurement of anything; silently
    dropping the pair would quietly change the distribution the threshold came from."""
    rows = _pairs([1.0])
    rows[0]["design"]["windows"] = _windows([0.0, 1.0])       # different grid
    with pytest.raises(V2HotspotCalibrationError, match="p0"):
        measure_pairs(rows, head_identity=_identity(), reference_binding_id="ref:calibration")


# --------------------------------------------------------------------------------------------
# the emitted artifact is one the config loader accepts
# --------------------------------------------------------------------------------------------


def _run(tmp_path, *, statistic="q95", deltas=(0.5, 1.0, 2.0, 4.0)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("\n".join(json.dumps(row) for row in _pairs(list(deltas))))
    out = tmp_path / "calib.json"
    code = main([
        "--pairs", str(pairs), "--allele", "DRB1_0701", "--score-scale", "nats",
        "--window-k-min", "4", "--window-k-max", "4",
        "--head-config-digest", "a" * 64, "--head-checkpoint-digest", "b" * 64,
        "--threshold-statistic", statistic, "--scope", "cumulative_depth0",
        "--reference-kind", "wt_native", "--reference-label", "wt_native",
        "--source-id", "v2.whole_landscape_hotspot.canary",
        "--code-revision", "deadbeef", "--out-json", str(out),
    ])
    return code, json.loads(out.read_text())


def test_the_emitted_block_is_accepted_by_the_config_loader(tmp_path):
    """The artifact exists to be pasted into `config.safety.delta_new_cumulative`.  The loader
    recomputes `source_ref` over the value and the typed artifact, so a block it rejects is a block
    that would have to be hand-edited -- and a hand-edited threshold is an invented one."""
    from inverse_folding.reference_flow.fusion_v2.config import load_v2_config
    from tests.inverse_folding.test_fusion_v2_config import _mapping

    code, payload = _run(tmp_path)
    assert code == 0
    config = load_v2_config(_mapping(**{
        "head.allele": "DRB1_0701", "head.score_scale": "nats",
        "head.window_k_min": 4, "head.window_k_max": 4,
        "safety.cumulative_reference_kind": "wt_native",
        "safety.cumulative_reference_label": "wt_native",
        "safety.delta_new_cumulative": payload["delta_new"],
    }))
    assert config.safety.delta_new_cumulative.value == payload["delta_new"]["value"]


def test_the_artifact_records_the_distribution_the_number_came_from(tmp_path):
    """A reviewer must be able to see the measurements, not only the scalar."""
    _code, payload = _run(tmp_path, deltas=(0.5, 1.0, 2.0, 4.0))
    assert payload["n_calibration_pairs"] == 4
    assert [row["pair_id"] for row in payload["measurements"]] == ["p0", "p1", "p2", "p3"]
    assert payload["threshold_statistic"] == "q95"


def test_the_data_digest_tracks_the_measurements_not_the_input_file(tmp_path):
    """Two runs over the same pairs must produce the same artifact; a different STATISTIC over the
    same pairs must not."""
    _c1, first = _run(tmp_path / "a", statistic="q95")
    _c2, same = _run(tmp_path / "b", statistic="q95")
    _c3, other = _run(tmp_path / "c", statistic="max")
    a = first["delta_new"]["artifact"]["calibration_data_digest"]
    assert a == same["delta_new"]["artifact"]["calibration_data_digest"]
    assert a != other["delta_new"]["artifact"]["calibration_data_digest"]


def test_the_producer_loads_no_model():
    """The calibration cohort's scoring is its own job; folding it in would make an expensive
    artifact impossible to re-derive cheaply."""
    import pathlib
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import scripts.calibrate_rf_fusion_v2_hotspot; print('torch' in sys.modules)"],
        capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
