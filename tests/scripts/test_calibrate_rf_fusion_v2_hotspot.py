"""The per-protein V2 hotspot calibration producer, against the FROZEN law (runbook §2.1).

The previous version of this suite pinned an earlier contract -- a `--pairs` JSONL of pre-scored
Head payloads, a free choice among five statistics, `numpy.quantile` interpolation, no structure
gate and no floor. All ten of its tests passed, and every one of them passed against the wrong
program: the runbook had since frozen a different law. This suite pins the law the runbook states.
"""

from __future__ import annotations

import dataclasses
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


def test_the_statistic_is_a_closed_enum_with_one_admissible_value_PER_POPULATION():
    """§2.1: "a closed enum for this Canary, not a free-form label".

    Widened to two entries because there are two POPULATIONS, and the first Canary measured that
    they differ. Still closed, still required -- and the statistic must now agree with the declared
    population, because two independently settable names for one choice is exactly how a resumed
    threshold ends up wearing the full-trajectory label.
    """
    from scripts.calibrate_rf_fusion_v2_hotspot import THRESHOLD_STATISTIC_BY_POPULATION

    action = next(a for a in build_parser()._actions if a.dest == "threshold_statistic")
    assert set(action.choices) == set(THRESHOLD_STATISTIC_BY_POPULATION.values())
    assert action.required is True
    population = next(a for a in build_parser()._actions if a.dest == "population")
    assert set(population.choices) == set(THRESHOLD_STATISTIC_BY_POPULATION)
    assert population.required is True


def test_a_statistic_that_does_not_name_its_population_is_refused(tmp_path, capsys):
    """The cross-check, not just the enum: both flags are settable, so both must agree."""
    from scripts.calibrate_rf_fusion_v2_hotspot import THRESHOLD_STATISTIC_BY_POPULATION

    code = main(_argv(tmp_path, **{
        "--population": "resumed",
        "--threshold-statistic": THRESHOLD_STATISTIC_BY_POPULATION["full_trajectory"]}))
    assert code == 2
    assert "does not name the" in capsys.readouterr().err


def test_no_independent_quantile_override_exists():
    """§2.1: they "would create two conflicting sources of truth"."""
    dests = {a.dest for a in build_parser()._actions}
    assert "quantile" not in dests and "quantile_method" not in dests


def test_the_frozen_interface_requires_every_flag_the_runbook_names():
    required = {a.dest for a in build_parser()._actions if getattr(a, "required", False)}
    assert {"protein_id", "n_completions", "master_seed", "seed_namespace",
            "threshold_statistic", "min_head_valid", "checkpoint", "rf_config",
            "test_set", "pdb_root", "complete_reference_manifest", "head_config_dir",
            "head_checkpoint", "structure_config", "out_rows", "out_json",
            "code_revision"} <= required


def test_the_head_domain_the_artifact_records_is_required_not_defaulted():
    """The artifact RECORDS `(allele, score_scale, window_k_min, window_k_max)` and
    `bind_admission_policy` refuses a run whose `config.head` differs, so a default here would let
    a threshold be measured over one window grid and enforced over another."""
    required = {a.dest for a in build_parser()._actions if getattr(a, "required", False)}
    assert {"allele", "score_scale", "window_k_min", "window_k_max",
            "head_variant_id", "refold_cache_dir"} <= required


def test_the_source_id_is_protein_specific():
    """The two proteins receive separate thresholds, so they receive separate identities."""
    assert source_id_for("Q00511") == "v2-canary-hotspot-head-valid-q90-higher-v1:Q00511"
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

    return HeadEvaluatorIdentity(allele="DRB1_0701", score_scale="raw_logit", window_k_min=4,
                                 window_k_max=4, head_config_hash="a" * 64,
                                 head_checkpoint_digest="b" * 64)


class _Scorer:
    """A Head whose z on the middle window is the sequence's own index -- so N_H is controllable.

    Exposes the SAME contract the V2 runtime scores through (``score(list[OracleRequest])``), not a
    calibration-only convenience method: the producer and the Canary must reach the Head the same
    way, or the threshold is calibrated against a path the gate never takes.
    """

    def evaluator_identity(self):
        return _identity()

    def score(self, requests):
        return [self._one(request.protein_id, request.sequence) for request in requests]

    def _one(self, protein_id, sequence):
        from inverse_folding.reference_flow.fusion.state import sequence_md5

        z = float(AA.index(sequence[-1])) if sequence[-1] in AA else 0.0
        return types.SimpleNamespace(
            protein_id=protein_id, sequence_md5=sequence_md5(sequence),
            sequence_length=len(sequence), allele="DRB1_0701", score_scale="raw_logit",
            windows=_windows([0.0, z, 0.0]), residue_hotspot=(0.0,) * len(sequence),
            global_risk=z)


#: Every attribute the producer reads off the model factory's ``PreparedModel``.  Named once so
#: :func:`test_the_fake_model_matches_the_real_prepared_model_surface` can check the fake against
#: the real class -- the fake previously called the residue map ``alphabet`` while ``PreparedModel``
#: calls it ``id_to_aa``, so the suite and production were wrong in exactly the same way and the
#: tests could not see it.  It cost a cluster allocation to find.
MODEL_SURFACE = ("id_to_aa", "aa_token_ids", "rf_config", "sampler", "fixed_tokens",
                 "backbone_and_denoiser", "sequence_length", "null_h_values")


class _Model:
    id_to_aa = {i: c for i, c in enumerate(AA)}
    aa_token_ids = frozenset(range(len(AA)))

    def fixed_tokens(self, protein_id):
        return None


def test_the_fake_model_matches_the_real_prepared_model_surface():
    """A fake that renames a real attribute makes the suite agree with the bug."""
    from scripts.rf_fusion_model_factory import PreparedModel

    missing = [name for name in MODEL_SURFACE if not hasattr(PreparedModel, name)
               and name not in getattr(PreparedModel, "__annotations__", {})]
    assert not missing, f"the producer reads {missing}, which PreparedModel does not carry"


def test_the_generator_decodes_through_the_map_prepared_model_actually_carries():
    """REGRESSION. `_generate_completion` ended in `decode_tokens_to_aa(tokens, model.alphabet)`.
    The sampler ran all 100 steps fine and the attempt died on the LAST line, so every replicate
    became `generation_failed` -- 0/4 on both Canary proteins."""
    import inspect

    from scripts.calibrate_rf_fusion_v2_hotspot import _generate_completion

    src = inspect.getsource(_generate_completion)
    assert "model.id_to_aa" in src
    assert "model.alphabet" not in src


def _seams(*, n_ok=64, structure_ok=None):
    """Completion ``i`` ends in ``AA[i % 20]``, so its N_H is ``i % 20``."""
    def generate(*, model, protein_id, seed):
        return "ACDEF" + AA[(seed % 1000) % 20]

    def structure(request):
        ok = structure_ok(request.sequence) if structure_ok else True
        return types.SimpleNamespace(evaluated=True, feasible=ok, metrics={"scTM": 0.9})

    return {
        "head_scorer": _Scorer(), "structure_gate": structure,
        "generate_completion": generate,
        "derive_seed": lambda *fields: sum(ord(c) for c in "".join(map(str, fields))) % 1000,
        "build_model_factory": None,
    }


def _run_law(**over):
    kw = dict(protein_id="5ZHV_B", n_completions=64, min_head_valid=60,
              master_seed=20260806, reference_sequence="ACDEFA", reference_digest="c" * 64,
              head_identity=_identity(), model=_Model(), seams=_seams())
    kw.update(over)
    return calibrate_protein(**kw)


def test_a_structure_rejection_does_not_remove_an_endpoint_from_the_population():
    """THE LAW CHANGED, and this asserts the new direction.

    The Head gate and the structure gate answer independent safety questions.  Conditioning the
    Head null on the structure verdict shrinks the estimation sample exactly when structure is
    hardest, and `Q00511` showed why it is also wrong on the merits: hard anchors preserved 64/64,
    scTM 64/64, rejected only by an absolute side-chain band whose native baseline is 1.791 A.
    That verdict does not make the sequence's hotspot unmeasurable."""
    seams = _seams(structure_ok=lambda sequence: False)          # every endpoint fails structure
    rows, summary = _run_law(seams=seams)

    assert summary["n_head_valid"] == 64, "structure must not filter the population"
    assert all(row["status"] == "head_valid" for row in rows)
    assert summary["structure_operability"]["n_definitive_feasible"] == 0
    assert summary["structure_operability"]["rate"] == 0.0


def test_an_unresolved_structure_verdict_is_still_only_a_diagnostic():
    """A cache-only label with no resolved verdict is not a definitive pass -- but it is also not a
    reason to drop the Head measurement."""
    seams = _seams()
    seams["structure_gate"] = lambda request: types.SimpleNamespace(
        evaluated=False, feasible=True, metrics={})
    rows, summary = _run_law(seams=seams)
    assert summary["n_head_valid"] == 64
    assert summary["structure_operability"]["n_definitive_feasible"] == 0


def test_capability_ceiling_calibration_can_explicitly_skip_the_independent_structure_diagnostic():
    """Skipping the diagnostic must not change the Head population or pretend structure passed.

    The high-risk campaign folds every realized search endpoint under its dual structure policy;
    refolding all 64 null draws per protein adds no information to the Head Q0.90 that the
    calibration artifact authorizes.  The omission is explicit and signed, never a silent missing
    oracle or a fabricated zero operability rate.
    """
    seams = _seams()
    seams["structure_gate"] = lambda request: (_ for _ in ()).throw(
        AssertionError("structure diagnostic must not execute"))
    rows, summary = _run_law(
        seams=seams,
        structure_diagnostic_mode="skip_independent_capability_ceiling",
    )
    assert summary["n_head_valid"] == 64
    assert summary["structure_operability"]["status"] == "not_measured"
    assert summary["structure_operability"]["rate"] is None
    assert all(row["structure_evaluated"] is False for row in rows)
    assert all(row["structure_definitive_feasible"] is False for row in rows)




class _FlakyHead(_Scorer):
    """Fails on chosen completion suffixes.  Never on 'A': that is the reference's suffix, and a
    reference the Head cannot score raises before the floor is ever evaluated."""

    FAIL_SUFFIXES = "CDEFG"

    def score(self, requests):
        out = []
        for request in requests:
            if request.sequence != "ACDEFA" and request.sequence[-1] in self.FAIL_SUFFIXES:
                raise RuntimeError("head refused")
            out.append(self._one(request.protein_id, request.sequence))
        return out


def _head_starved_seams():
    seams = _seams()
    seams["head_scorer"] = _FlakyHead()
    return seams


def test_the_head_valid_floor_is_not_relaxed_and_no_artifact_is_produced():
    """§2.1 step 4 on the new population: the floor is on HEAD-valid endpoints."""
    with pytest.raises(V2HotspotCalibrationError) as excinfo:
        _run_law(seams=_head_starved_seams())
    assert "60" in str(excinfo.value) and "valid Head measurement" in str(excinfo.value)


def test_every_attempt_is_a_row_including_the_rejected_ones():
    """A table holding only the passing endpoints makes both RATES unrecoverable -- the head-valid
    one the floor is checked against, and the structure one reported beside the threshold."""
    seams = _seams(structure_ok=lambda sequence: sequence[-1] != "B")
    rows, summary = _run_law(seams=seams)
    assert len(rows) == 64
    assert summary["n_attempted"] == 64
    assert summary["n_head_valid"] == sum(1 for r in rows if r["status"] == "head_valid")
    assert summary["structure_operability"]["n_definitive_feasible"] == sum(
        1 for r in rows if r["structure_definitive_feasible"])


def test_the_summary_reports_the_distribution_not_only_the_scalar():
    """§2.1: "A producer that outputs only the chosen scalar is incomplete"."""
    _rows, summary = _run_law()
    assert {"q50", "q90_higher", "q95", "max", "order_statistic_rank", "n_attempted",
            "n_head_valid", "failure_counts", "structure_operability"} <= set(summary)


def test_the_structure_rate_is_reported_beside_the_threshold_never_inside_it():
    """Keeping it visible is what stops "measured over every endpoint" from being misread as
    "structure was not checked"."""
    seams = _seams(structure_ok=lambda sequence: sequence[-1] in "ACDEFGHIJ")
    rows, summary = _run_law(seams=seams)
    structure = summary["structure_operability"]
    expected = sum(1 for r in rows if r["structure_definitive_feasible"])
    assert structure["n_definitive_feasible"] == expected
    assert 0 < expected < 64, "the fixture must exercise a partial rate"
    assert structure["rate"] == pytest.approx(expected / summary["n_head_valid"])
    assert "DIAGNOSTIC" in structure["note"]


def test_the_canonical_row_carries_every_field_the_runbook_signs():
    rows, _summary = _run_law()
    row = next(row for row in rows if row["status"] == "head_valid")
    assert {"protein_id", "replicate_index", "seed", "sequence_md5", "reference_digest",
            "head_evaluator_digest", "window_grid_digest", "n_h_whole", "structure_evaluated",
            "structure_feasible", "structure_definitive_feasible", "structure_metrics_json",
            "anchor_verdict"} <= set(row)


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
    retained = next(row for row in rows if row["status"] == "head_valid")
    # Re-derived from the row's OWN realized seed rather than a guessed one: a test that guessed
    # would drift the moment the seed law changed and would then be asserting nothing.
    sequence = seams["generate_completion"](
        model=_Model(), protein_id="5ZHV_B", seed=retained["seed"])
    scorer = _Scorer()
    direct = whole_landscape_new_hotspot(
        scorer._one("5ZHV_B", sequence),
        scorer._one("5ZHV_B", "ACDEFA"),
        endpoint_id="endpoint:x", head_identity=_identity(),
        reference_kind=ReferenceKind.CUMULATIVE_DEPTH0, reference_binding_id="ref:x")
    assert retained["n_h_whole"] == pytest.approx(direct.max_increase)


def test_both_oracles_are_reached_through_the_runtimes_own_request_type():
    """The producer and the Canary must call the Head and the structure gate the SAME way.

    ``OracleRequest`` re-derives the digest from the bytes and refuses a non-canonical residue, so
    scoring through it is what makes "the sequence the Head saw" and "the sequence the row claims"
    one object.  A calibration-only `(protein_id=, sequence=)` convenience signature would let the
    two paths diverge silently.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    seen = []
    seams = _seams()
    scorer = seams["head_scorer"]
    seams["head_scorer"] = types.SimpleNamespace(
        evaluator_identity=scorer.evaluator_identity,
        score=lambda requests: (seen.extend(requests), scorer.score(requests))[1])
    structure = seams["structure_gate"]
    seams["structure_gate"] = lambda request: (seen.append(request), structure(request))[1]
    _run_law(seams=seams)

    assert seen, "neither oracle was reached"
    assert all(isinstance(request, OracleRequest) for request in seen)


# --------------------------------------------------------------------------------------------
# the production path: the declared paths must actually BUILD the oracles
# --------------------------------------------------------------------------------------------


def _reference_manifest(tmp_path, protein_id="5ZHV_B", sequence="ACDEFA"):
    """A `{protein_id: {path, sha256}}` manifest over canonical sequence BYTES (runbook §3)."""
    import hashlib

    seq_file = tmp_path / f"{protein_id}.seq"
    seq_file.write_text(sequence, encoding="utf-8")
    manifest = tmp_path / "references.json"
    manifest.write_text(json.dumps({protein_id: {
        "path": seq_file.name,
        "sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
    }}), encoding="utf-8")
    return manifest


def _argv(tmp_path, **over):
    args = {
        "--protein-id": "5ZHV_B", "--population": "full_trajectory",
        "--n-completions": "64", "--master-seed": "20260806",
        "--seed-namespace": SEED_NAMESPACE, "--threshold-statistic": THRESHOLD_STATISTIC,
        "--min-head-valid": "60", "--checkpoint": "/nx/dplm.ckpt",
        "--rf-config": "/nx/rf.yaml", "--test-set": "/nx/test.parquet", "--pdb-root": "/nx/pdbs",
        "--complete-reference-manifest": str(_reference_manifest(tmp_path)),
        "--head-config-dir": "/nx/head/configs", "--head-checkpoint": "/nx/head/best.pt",
        "--structure-config": "/nx/v0_fusion.yaml", "--refold-cache-dir": "/nx/refold",
        "--allele": "DRB1_0701", "--score-scale": "nats",
        "--window-k-min": "4", "--window-k-max": "4", "--head-variant-id": "LC1",
        "--out-rows": str(tmp_path / "rows.parquet"),
        "--out-json": str(tmp_path / "calib.json"), "--code-revision": "a" * 40,
    }
    args.update(over)
    return [token for pair in args.items() for token in pair]


def _production_seams(recorder):
    fake = _seams()

    def build_production_oracles(**kwargs):
        recorder.update(kwargs)
        return fake["head_scorer"], fake["structure_gate"]

    return CalibrationSeams(
        build_model_factory=lambda **_: _Model(),
        generate_completion=fake["generate_completion"], derive_seed=fake["derive_seed"],
        build_production_oracles=build_production_oracles,
    )


def test_main_builds_the_real_oracles_from_the_declared_paths(tmp_path):
    """REGRESSION.  `--head-config-dir`, `--head-checkpoint` and `--structure-config` were declared
    `required=True`, parsed, and then never read: `head_scorer`/`structure_gate` stayed `None` and
    `main` died on `NoneType.evaluator_identity()` -- after the DPLM checkpoint had already been
    loaded onto the GPU.  Every declared path must reach the builder."""
    recorder: dict = {}
    assert main(_argv(tmp_path), seams=_production_seams(recorder)) == 0

    assert recorder["head_config_dir"] == "/nx/head/configs"
    assert recorder["head_checkpoint"] == "/nx/head/best.pt"
    assert recorder["structure_config"] == "/nx/v0_fusion.yaml"
    assert recorder["refold_cache_dir"] == "/nx/refold"
    # The Head DOMAIN is passed through verbatim: it is what the artifact records and what
    # bind_admission_policy checks the realized scorer against.
    assert (recorder["allele"], recorder["score_scale"]) == ("DRB1_0701", "nats")
    assert (recorder["window_k_min"], recorder["window_k_max"]) == (4, 4)


def test_main_writes_both_artifacts_and_the_paste_ready_block(tmp_path):
    """The producer's two outputs: the canonical raw table and the typed artifact."""
    assert main(_argv(tmp_path), seams=_production_seams({})) == 0

    payload = json.loads((tmp_path / "calib.json").read_text())
    assert payload["threshold_statistic"] == THRESHOLD_STATISTIC
    assert payload["delta_new"]["source_id"] == source_id_for("5ZHV_B")
    assert payload["delta_new"]["artifact"]["window_domain"] == "whole_landscape"
    assert payload["n_attempted"] == 64 and payload["n_head_valid"] >= 60
    assert (tmp_path / "rows.parquet").exists()


def test_a_run_below_the_floor_writes_no_artifact(tmp_path):
    """§2.1 step 4 on the PRODUCTION path, not only inside the law helper."""
    fake = _head_starved_seams()
    seams = CalibrationSeams(
        build_model_factory=lambda **_: _Model(),
        generate_completion=fake["generate_completion"], derive_seed=fake["derive_seed"],
        build_production_oracles=lambda **_: (fake["head_scorer"], fake["structure_gate"]),
    )
    assert main(_argv(tmp_path), seams=seams) == 2
    assert not (tmp_path / "calib.json").exists()
    # ...but the EVIDENCE survives.  "Write no calibration artifact" withholds the threshold, not
    # the raw table: the definitive-feasible rate the floor is checked against lives only there, so
    # discarding it at the moment the floor fails discards the one record that says why.
    import pandas as pd

    rows = pd.read_parquet(tmp_path / "rows.parquet")
    assert len(rows) == 64, "every attempt is a row, including the rejected ones"
    assert set(rows["status"]) - {"retained"}, "the failing statuses must be recoverable"


def test_the_refusal_message_names_the_failure_kinds(capsys, tmp_path):
    """A below-floor run whose only record is `only N of M` is undiagnosable without a re-run."""
    fake = _head_starved_seams()
    seams = CalibrationSeams(
        build_model_factory=lambda **_: _Model(),
        generate_completion=fake["generate_completion"], derive_seed=fake["derive_seed"],
        build_production_oracles=lambda **_: (fake["head_scorer"], fake["structure_gate"]),
    )
    assert main(_argv(tmp_path), seams=seams) == 2
    err = capsys.readouterr().err
    assert "status counts" in err and "head_failed" in err
    assert "failure counts" in err and "head" in err


# --------------------------------------------------------------------------------------------
# the declared unit must be the one the evaluator actually emits
# --------------------------------------------------------------------------------------------


def test_the_unit_and_the_artifact_scale_come_from_the_realized_head(tmp_path):
    """`unit` and `artifact.score_scale` are GENERATED from the Head that scored, never typed.

    The frozen Head emits uncalibrated classifier logits; `OnlineHeadScorer` refuses any other
    `score_scale`. Declaring `nats` would be a unit the evaluator does not produce -- and a
    threshold cannot be re-derived from measurements taken in a different unit.
    """
    assert main(_argv(tmp_path), seams=_production_seams({})) == 0
    payload = json.loads((tmp_path / "calib.json").read_text())
    scale = _identity().score_scale
    assert payload["delta_new"]["unit"] == scale
    assert payload["delta_new"]["artifact"]["score_scale"] == scale


def test_the_artifact_records_the_window_grid_it_was_measured_over(tmp_path):
    """No threshold measured on another grid may be inherited: `N_H^whole` is a maximum over the
    windows these bounds define."""
    assert main(_argv(tmp_path), seams=_production_seams({})) == 0
    artifact = json.loads((tmp_path / "calib.json").read_text())["delta_new"]["artifact"]
    assert artifact["window_k_min"] == _identity().window_k_min
    assert artifact["window_k_max"] == _identity().window_k_max
    assert artifact["allele"] == _identity().allele


def test_the_source_ref_is_recomputed_over_the_value_and_the_artifact(tmp_path):
    """The loader recomputes it and refuses a mismatch, so relabelling a threshold's unit without
    re-measuring cannot survive: the digest covers the unit."""
    from inverse_folding.reference_flow.fusion_v2.config import (
        HotspotCalibrationArtifact,
        calibration_source_ref,
    )

    assert main(_argv(tmp_path), seams=_production_seams({})) == 0
    block = json.loads((tmp_path / "calib.json").read_text())["delta_new"]
    artifact = HotspotCalibrationArtifact(**block["artifact"])
    assert block["source_ref"] == calibration_source_ref(
        value=block["value"], unit=block["unit"], source_kind=block["source_kind"],
        source_id=block["source_id"], artifact=artifact)
    assert block["unit"] == "raw_logit"
    # Relabelling the unit produces a DIFFERENT source_ref, which is what makes the laundering the
    # runbook forbids detectable rather than cosmetic.  `config.py` additionally requires
    # `scalar.unit == head.score_scale`, so a block relabelled on ONE side is refused outright and
    # a block relabelled on BOTH sides no longer matches its own recomputed digest.
    assert block["source_ref"] != calibration_source_ref(
        value=block["value"], unit="nats", source_kind=block["source_kind"],
        source_id=block["source_id"], artifact=artifact)
    relabelled = dataclasses.replace(artifact, score_scale="nats")
    assert block["source_ref"] != calibration_source_ref(
        value=block["value"], unit="nats", source_kind=block["source_kind"],
        source_id=block["source_id"], artifact=relabelled)


def test_the_shipped_canary_template_declares_the_unit_the_head_emits():
    """REGRESSION. The template declared `nats` in three places while the only Head that exists
    hard-refuses anything but `raw_logit`, so no resolved config could ever bind."""
    import pathlib

    import yaml

    template = yaml.safe_load((pathlib.Path(__file__).resolve().parents[2]
                               / "inverse_folding/reference_flow/configs"
                               / "v2_canary_state_transition.yaml").read_text())
    delta = template["safety"]["delta_new_cumulative"]
    assert template["head"]["score_scale"] == "raw_logit"
    assert delta["unit"] == "raw_logit"
    assert delta["artifact"]["score_scale"] == "raw_logit"
    # The reference is named ONE way across the config and the emitted artifact.
    from scripts.calibrate_rf_fusion_v2_hotspot import (
        CUMULATIVE_REFERENCE_KIND,
        CUMULATIVE_REFERENCE_LABEL,
    )

    assert template["safety"]["cumulative_reference_kind"] == CUMULATIVE_REFERENCE_KIND
    assert template["safety"]["cumulative_reference_label"] == CUMULATIVE_REFERENCE_LABEL
    assert delta["artifact"]["reference_kind"] == CUMULATIVE_REFERENCE_KIND
    assert delta["artifact"]["reference_label"] == CUMULATIVE_REFERENCE_LABEL


def test_the_template_grid_is_the_one_the_v2_calibration_must_use():
    """12-25 -- the domain the frozen Head's inference config actually emits.

    Declaring a narrower one does not narrow the grid: `window_k_min/max` are metadata and the
    windows come from the Head, so a narrower declaration rejects every endpoint instead. Using
    V1's grid inherits no THRESHOLD: the V2 value is re-measured under §2.1's frozen law."""
    import pathlib

    import yaml

    template = yaml.safe_load((pathlib.Path(__file__).resolve().parents[2]
                               / "inverse_folding/reference_flow/configs"
                               / "v2_canary_state_transition.yaml").read_text())
    assert (template["head"]["window_k_min"], template["head"]["window_k_max"]) == (12, 25)
    artifact = template["safety"]["delta_new_cumulative"]["artifact"]
    assert (artifact["window_k_min"], artifact["window_k_max"]) == (12, 25)


def test_a_declared_window_domain_the_head_does_not_have_fails_before_any_completion():
    """REGRESSION.  The run declared [13,25]; the frozen Head's inference config emits k=12..25, and
    `window_k_min/max` are METADATA -- the scorer stores them and never filters on them.  So the
    narrower declaration rejected every endpoint, one at a time, after the allocation was paid.
    64 identical `head_failed` rows is not a measurement; it is a declaration error."""
    class _K12(_Scorer):
        def _one(self, protein_id, sequence):
            score = super()._one(protein_id, sequence)
            score.windows = score.windows + (
                types.SimpleNamespace(start_0b=0, end_0b=12, k=12, z=0.0),)
            return score

    seams = _seams()
    seams["head_scorer"] = _K12()
    with pytest.raises(V2HotspotCalibrationError, match=r"emits k in \[4, 12\]"):
        _run_law(seams=seams)


def test_an_empty_window_grid_is_refused_rather_than_scored():
    class _NoWindows(_Scorer):
        def _one(self, protein_id, sequence):
            score = super()._one(protein_id, sequence)
            score.windows = ()
            return score

    seams = _seams()
    seams["head_scorer"] = _NoWindows()
    with pytest.raises(V2HotspotCalibrationError, match="no windows"):
        _run_law(seams=seams)


# --------------------------------------------------------------------------------------------
# the RESUMED population (owner's ruling, 2026-08-06): the source prefix is the sampling unit
# --------------------------------------------------------------------------------------------


def test_the_statistic_and_the_source_id_carry_the_population():
    """Two populations of the same quantity. A config that named only the protein could carry one
    population's threshold under the other's name and nothing would contradict it."""
    from scripts.calibrate_rf_fusion_v2_hotspot import (
        THRESHOLD_STATISTIC_BY_POPULATION, source_id_for)

    assert (THRESHOLD_STATISTIC_BY_POPULATION["full_trajectory"]
            != THRESHOLD_STATISTIC_BY_POPULATION["resumed"])
    assert source_id_for("Q00511") != source_id_for("Q00511", population="resumed")
    # The original law's id is unchanged, so the executed artifacts stay valid.
    assert source_id_for("Q00511") == "v2-canary-hotspot-head-valid-q90-higher-v1:Q00511"


def test_source_variance_reports_how_many_samples_64_endpoints_actually_are():
    """Sixty-four siblings of one prefix are one sample of the prefix distribution. If between-
    source variance dominates, the effective n is the number of PREFIXES."""
    from scripts.calibrate_rf_fusion_v2_hotspot import source_variance

    # Two prefixes, tight within and far apart between: ICC must be near 1.
    rows = [{"status": "head_valid", "source_index": 0, "n_h_whole": v} for v in (1.0, 1.1)]
    rows += [{"status": "head_valid", "source_index": 1, "n_h_whole": v} for v in (20.0, 20.1)]
    clustered = source_variance(rows)
    assert clustered["n_sources"] == 2 and clustered["n_endpoints"] == 4
    assert clustered["icc_between_over_total"] > 0.99, clustered

    # Same spread, but it lives WITHIN each prefix: ICC must be near 0.
    rows = [{"status": "head_valid", "source_index": 0, "n_h_whole": v} for v in (1.0, 20.0)]
    rows += [{"status": "head_valid", "source_index": 1, "n_h_whole": v} for v in (1.1, 20.1)]
    spread = source_variance(rows)
    assert spread["icc_between_over_total"] < 0.01, spread

    assert "not estimable" in source_variance(
        [{"status": "head_valid", "source_index": 0, "n_h_whole": 1.0}])["note"]


def test_a_floor_and_a_ceiling_round_in_opposite_directions():
    """`scTM_min` is a FLOOR taken from the lower tail and the anchor band is a CEILING from the
    upper tail. Rounding both the same way silently admits a sample one of them meant to exclude."""
    from scripts.calibrate_rf_fusion_v2_hotspot import (
        _quantile_lower, q90_higher_order_statistic)

    values = [float(i) for i in range(1, 11)]          # 1..10
    assert q90_higher_order_statistic(values)[0] == 9.0     # ceil(0.90*10) = 9th
    assert _quantile_lower(values, 0.10) == 1.0             # floor(0.10*10) = 1st


def test_the_anchor_gate_is_native_relative_and_says_so_when_it_cannot_be():
    """`Q00511`'s native scores ~1.79 A under this backend, so of a 2.0 A band only ~0.2 A is
    anything but predictor error. Admission moves to the excess over the native -- and when the
    native was never evaluated, the artifact must SAY the baseline is missing rather than fall back
    to an absolute number."""
    import types as _types

    from scripts.calibrate_rf_fusion_v2_hotspot import mechanism_stage_structure_gates

    rows = [{"status": "head_valid",
             "structure_metrics_json": json.dumps(
                 {"scTM": 0.80 + 0.01 * i, "max_anchor_sidechain_RMSD": 1.9 + 0.05 * i})}
            for i in range(10)]
    native = _types.SimpleNamespace(metrics={"max_anchor_sidechain_RMSD": 1.791, "scTM": 0.9764})

    gates = mechanism_stage_structure_gates(rows, native_structure=native)
    assert gates["authority"].startswith("mechanism_operability_only")
    assert "absolute hard gate" in gates["hard_anchor_identity"]
    delta = gates["delta_anchor"]
    assert delta["native_absolute"] == 1.791
    # The excess, not the raw value: every design here is ABOVE 1.9 A absolute, and the proposed
    # ceiling is a fraction of an angstrom because that is what the designs add to the native.
    assert delta["proposed_delta_anchor_max"] < 1.0, delta
    assert delta["absolute_rmsd"]["max"] > 2.0, "raw absolute RMSD must still be reported in full"
    # A floor from the LOWER tail, so the null's own worst folds are not excluded by their own gate.
    assert gates["scTM"]["proposed_scTM_min"] <= gates["scTM"]["q50"]

    blind = mechanism_stage_structure_gates(rows, native_structure=None)
    assert "unmeasurable" in blind["delta_anchor"]
    assert "proposed_delta_anchor_max" not in blind["delta_anchor"]


def test_a_lost_source_is_recorded_as_draws_rather_than_shrinking_the_sample():
    """A capture that fails costs its whole fork group. Dropping those endpoints would make the
    head-valid floor pass on a sample smaller than the one that was requested."""
    from scripts.calibrate_rf_fusion_v2_hotspot import resumed_draws

    def _boom(**_):
        raise RuntimeError("no checkpoint was captured at step 50")

    import scripts.calibrate_rf_fusion_v2_hotspot as mod
    import inverse_folding.reference_flow.fusion_v2_runtime.capture as capture_mod

    original = capture_mod.capture_depth_zero
    capture_mod.capture_depth_zero = _boom
    try:
        draws = resumed_draws(
            protein_id="Q00511", n_sources=3, completions_per_source=2, c_source=50,
            master_seed=1, cycle_kwargs={"config": _types_config()},
            seams={"derive_seed": lambda *parts: abs(hash(parts)) % (2 ** 31)})
    finally:
        capture_mod.capture_depth_zero = original

    assert len(draws) == 6, "three lost sources must still account for six endpoints"
    assert all(d.sequence is None and "capture failed" in d.error for d in draws)
    assert [d.labels["replicate_index"] for d in draws] == list(range(6))


@dataclasses.dataclass(frozen=True)
class _FakeSampler:
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class _FakeRfConfig:
    sampler: _FakeSampler = dataclasses.field(default_factory=_FakeSampler)


def _types_config():
    """A real dataclass: `resumed_draws` reseeds the sampler with `dataclasses.replace`, and a
    SimpleNamespace would make the test pass on a shape production never sees."""
    return _FakeRfConfig()


def test_resumed_draws_read_maturity_off_the_state_the_capture_returns():
    """REGRESSION. `resumed_draws` read `source.maturity`, which a `LivePartialState` does not have:
    it DERIVES `realized_maturity` from its own tokens and stores nothing. The continuation
    checkpoint has `.maturity`; the adapted state does not. Cost: one GPU allocation."""
    import ast
    import inspect

    from inverse_folding.reference_flow.fusion_v2.state import LivePartialState
    from scripts.calibrate_rf_fusion_v2_hotspot import resumed_draws

    tree = ast.parse(inspect.getsource(resumed_draws))
    read = sorted({node.attr for node in ast.walk(tree)
                   if isinstance(node, ast.Attribute)
                   and isinstance(node.value, ast.Name) and node.value.id == "source"})
    declared = set(dir(LivePartialState)) | set(
        getattr(LivePartialState, "__annotations__", {}))
    missing = [name for name in read if name not in declared]
    assert not missing, f"resumed_draws reads source.{missing}; LivePartialState carries {read}"
