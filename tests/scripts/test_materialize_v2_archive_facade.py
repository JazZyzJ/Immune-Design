"""Strict V2 archive -> Phase-C/v0 facade materialization."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from scripts.materialize_v2_archive_facade import (
    V2FacadeError,
    main,
    materialize_archive_facade,
)


def _write_bundle(
    root,
    *,
    protein_id: str,
    endpoints: list[dict],
    archive_overrides: dict[str, dict] | None = None,
    manifest_overrides: dict | None = None,
):
    root.mkdir(parents=True)
    archive_overrides = archive_overrides or {}
    complete_rows = []
    archive_rows = []
    for index, spec in enumerate(endpoints):
        endpoint_id = spec.get("endpoint_id", f"endpoint:{protein_id}:{index}")
        sequence = spec.get("sequence", "ACDEFGHIK")
        sequence_md5 = spec.get(
            "sequence_md5", hashlib.md5(sequence.encode("ascii"), usedforsecurity=False).hexdigest(),
        )
        digest = spec.get("endpoint_content_digest", f"digest:{protein_id}:{index}")
        complete_rows.append({
            "endpoint_id": endpoint_id,
            "endpoint_content_digest": digest,
            "protein_id": protein_id,
            "depth": spec.get("depth", index),
            "sequence": sequence,
            "sequence_md5": sequence_md5,
            "sequence_equivalence_key": spec.get("sequence_equivalence_key", sequence_md5),
            "sequence_length": len(sequence),
            "head_evaluator_digest": spec.get("head_evaluator_digest", "head-digest"),
            "head_window_grid_digest": spec.get("head_window_grid_digest", "grid-digest"),
            "head_global_risk": spec.get("head_global_risk", float(index)),
            "feasibility_level": spec.get("feasibility_level", "definitive"),
            "structure_evaluated": spec.get("structure_evaluated", True),
            "structure_feasible": spec.get("structure_feasible", True),
        })
        archive = {
            "endpoint_id": endpoint_id,
            "endpoint_content_digest": digest,
            "protein_id": protein_id,
            "sequence_equivalence_key": spec.get("sequence_equivalence_key", sequence_md5),
            "feasibility_level": spec.get("archive_feasibility_level", "definitive"),
            "is_elite": spec.get("is_elite", index == 0),
            "elite_rank": 0 if spec.get("is_elite", index == 0) else None,
        }
        archive.update(archive_overrides.get(endpoint_id, {}))
        archive_rows.append(archive)
    pd.DataFrame(complete_rows).to_parquet(root / "complete_endpoints.parquet", index=False)
    pd.DataFrame(archive_rows).to_parquet(root / "archive.parquet", index=False)
    manifest = {
        "campaign_id": "fusion-v2-exploratory-uricase",
        "phase": "capability_ladder",
        "split_role": "exploratory_uricase",
        "code_revision": "deadbeef",
        "schedule_id": "exploratory-uricase-d4-k12-r40-v1",
        "coordinate_law": "progressive_checkpoint",
        "depth_cap": 4,
        "exploratory_depth_override": True,
        "production_depth_authorized": False,
        # These three define the common terminal structure instrument.  The backbone is
        # deliberately protein-specific and must not make two cells look like different studies.
        "content_identities": {
            "structure_backend": "a" * 64,
            "v0_structure_gate_config": "b" * 64,
            "structure_config": "c" * 64,
            "backbone": hashlib.sha256(protein_id.encode("ascii")).hexdigest(),
        },
        "config_digest": hashlib.sha256(f"config:{protein_id}".encode("ascii")).hexdigest(),
    }
    for key, value in (manifest_overrides or {}).items():
        if key == "content_identities":
            manifest[key].update(value)
        else:
            manifest[key] = value
    (root / "run_manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return root


def test_elite_mode_joins_by_endpoint_and_preserves_phase_c_and_lineage_columns(tmp_path):
    first = _write_bundle(
        tmp_path / "p1",
        protein_id="P1",
        endpoints=[
            {"endpoint_id": "endpoint:P1:best", "head_global_risk": -3.0, "depth": 2},
            {"endpoint_id": "endpoint:P1:other", "head_global_risk": -1.0,
             "is_elite": False},
        ],
    )
    second = _write_bundle(
        tmp_path / "p2", protein_id="P2",
        endpoints=[{"endpoint_id": "endpoint:P2:best", "head_global_risk": -2.0,
                    "depth": 1}],
    )

    out = materialize_archive_facade([first, second], mode="elite")

    assert out[["protein_id", "design_idx", "entry_source_id"]].to_dict("records") == [
        {"protein_id": "P1", "design_idx": 0, "entry_source_id": "endpoint:P1:best"},
        {"protein_id": "P2", "design_idx": 0, "entry_source_id": "endpoint:P2:best"},
    ]
    assert list(out["endpoint_id"]) == list(out["entry_source_id"])
    assert list(out["depth"]) == [2, 1]
    assert list(out["head_global_risk"]) == [-3.0, -2.0]


def test_top_k_is_risk_sorted_stably_and_does_not_give_duplicate_sequences_mass(tmp_path):
    bundle = _write_bundle(
        tmp_path / "bundle",
        protein_id="P1",
        endpoints=[
            {"endpoint_id": "endpoint:z", "sequence": "ACDEFGHIK",
             "head_global_risk": -5.0, "is_elite": True},
            {"endpoint_id": "endpoint:a", "sequence": "ACDEFGHIK",
             "head_global_risk": -5.0, "is_elite": False},
            {"endpoint_id": "endpoint:b", "sequence": "LMNPQRSTV",
             "head_global_risk": -5.0, "is_elite": False},
            {"endpoint_id": "endpoint:c", "sequence": "WYACDEFGH",
             "head_global_risk": -4.0, "is_elite": False},
        ],
    )

    out = materialize_archive_facade([bundle], mode="top-k", k=3, require_k=True)

    # endpoint:a is the stable representative of the two convergent logical endpoints.
    assert list(out["endpoint_id"]) == ["endpoint:a", "endpoint:b", "endpoint:c"]
    assert list(out["design_idx"]) == [0, 1, 2]


def test_top_k_require_k_fails_when_distinct_feasible_pool_is_too_small(tmp_path):
    bundle = _write_bundle(
        tmp_path / "bundle", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:only"}],
    )
    with pytest.raises(V2FacadeError, match="requires 2.*has 1"):
        materialize_archive_facade([bundle], mode="top-k", k=2, require_k=True)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda root: pd.concat([
            pd.read_parquet(root / "archive.parquet"),
            pd.read_parquet(root / "archive.parquet"),
        ]).to_parquet(root / "archive.parquet", index=False), "duplicate endpoint_id"),
        (lambda root: pd.DataFrame(columns=pd.read_parquet(
            root / "complete_endpoints.parquet").columns).to_parquet(
                root / "complete_endpoints.parquet", index=False), "no complete endpoint"),
        (lambda root: _replace_column(root / "complete_endpoints.parquet",
                                      "structure_feasible", False), "not definitive-feasible"),
        (lambda root: _replace_column(root / "complete_endpoints.parquet",
                                      "sequence", "ACD#"), "AA20"),
    ],
)
def test_malformed_or_ineligible_evidence_fails_closed(tmp_path, mutate, message):
    bundle = _write_bundle(
        tmp_path / "bundle", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:only"}],
    )
    mutate(bundle)
    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade([bundle], mode="elite")


def _replace_column(path, column, value):
    frame = pd.read_parquet(path)
    frame[column] = value
    frame.to_parquet(path, index=False)


def test_a_protein_may_not_be_silently_merged_across_two_bundles(tmp_path):
    first = _write_bundle(tmp_path / "first", protein_id="P1",
                          endpoints=[{"endpoint_id": "endpoint:first"}])
    second = _write_bundle(tmp_path / "second", protein_id="P1",
                           endpoints=[{"endpoint_id": "endpoint:second"}])
    with pytest.raises(V2FacadeError, match="appears in more than one bundle"):
        materialize_archive_facade([first, second], mode="elite")


@pytest.mark.parametrize(
    ("manifest_overrides", "message"),
    [
        ({"campaign_id": "other-campaign"}, "campaign_id"),
        ({"code_revision": "cafebabe"}, "code_revision"),
        ({"schedule_id": "another-schedule"}, "schedule_id"),
        ({"coordinate_law": "stationary_checkpoint"}, "coordinate_law"),
        ({"depth_cap": 3}, "depth_cap"),
        ({"content_identities": {"structure_backend": "d" * 64}},
         "structure_backend"),
        ({"content_identities": {"v0_structure_gate_config": "e" * 64}},
         "v0_structure_gate_config"),
        ({"content_identities": {"structure_config": "f" * 64}},
         "structure_config"),
    ],
)
def test_bundles_from_different_experiments_cannot_be_merged(
    tmp_path, manifest_overrides, message,
):
    first = _write_bundle(
        tmp_path / "first", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:first"}],
    )
    second = _write_bundle(
        tmp_path / "second", protein_id="P2",
        endpoints=[{"endpoint_id": "endpoint:second"}],
        manifest_overrides=manifest_overrides,
    )
    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade([first, second], mode="elite")


@pytest.mark.parametrize(
    ("manifest_overrides", "message"),
    [
        ({"phase": "policy_qualification"}, "phase.*capability_ladder"),
        ({"split_role": "exploratory_other"}, "split_role.*exploratory_uricase"),
        ({"exploratory_depth_override": False}, "exploratory_depth_override"),
        ({"production_depth_authorized": True}, "production_depth_authorized"),
    ],
)
def test_every_bundle_must_be_an_exploratory_uricase_capability_run(
    tmp_path, manifest_overrides, message,
):
    bundle = _write_bundle(
        tmp_path / "bundle", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:only"}],
        manifest_overrides=manifest_overrides,
    )
    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade([bundle], mode="elite")


def test_missing_manifest_is_refused_before_parquet_merge(tmp_path):
    bundle = _write_bundle(
        tmp_path / "bundle", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:only"}],
    )
    (bundle / "run_manifest.json").unlink()
    (bundle / "archive.parquet").unlink()  # manifest identity must fail before table discovery
    with pytest.raises(V2FacadeError, match="run_manifest"):
        materialize_archive_facade([bundle], mode="elite")


def test_protein_specific_content_identities_may_differ(tmp_path):
    first = _write_bundle(
        tmp_path / "first", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:first"}],
        manifest_overrides={"content_identities": {
            "constraint_manifest": "1" * 64,
            "schedule_band_calibration": "2" * 64,
            "complete_reference_sequence": "3" * 64,
        }},
    )
    second = _write_bundle(
        tmp_path / "second", protein_id="P2",
        endpoints=[{"endpoint_id": "endpoint:second"}],
        manifest_overrides={"content_identities": {
            "constraint_manifest": "4" * 64,
            "schedule_band_calibration": "5" * 64,
            "complete_reference_sequence": "6" * 64,
        }},
    )
    assert list(materialize_archive_facade(
        [first, second], mode="elite",
    )["protein_id"]) == ["P1", "P2"]


def test_cli_output_is_accepted_by_the_existing_phase_c_loader(tmp_path):
    from scripts.evaluate_phase_c import load_generated_designs
    from scripts.run_rf_refine_fusion import _seed_rows_by_protein

    bundle = _write_bundle(tmp_path / "bundle", protein_id="P1",
                           endpoints=[{"endpoint_id": "endpoint:only"}])
    output = tmp_path / "generation" / "generated.parquet"
    assert main([
        "--bundle", str(bundle), "--mode", "elite", "--output", str(output),
    ]) == 0

    loaded = load_generated_designs(output)
    assert loaded.loc[0, "entry_source_id"] == "endpoint:only"
    assert loaded.loc[0, "design_id"] == "design_0000"
    v0_rows = _seed_rows_by_protein(loaded, ["P1"])
    assert v0_rows["P1"][0]["entry_source_id"] == "endpoint:only"
