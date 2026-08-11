"""Strict V2 archive -> Phase-C/v0 facade materialization."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from inverse_folding.reference_flow.fusion_v2.identity import (
    CONTENT_ROLE_TO_FIELD,
    canonical_digest,
)
from inverse_folding.reference_flow.fusion_v2.structure_gate import (
    DUAL_SCTM_POLICY_KIND,
    HIGH_RISK_ANCESTRY_SCTM_MIN,
    HIGH_RISK_DUAL_SCTM_PROFILE_ID,
    HIGH_RISK_STRICT_SCTM_MIN,
)
from scripts.materialize_v2_archive_facade import (
    HighRiskFacadeIdentity,
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


def _replace_highrisk_evidence(bundle, column, value):
    for table in (
        "complete_endpoints.parquet", "archive.parquet", "structure_evaluations.parquet",
    ):
        _replace_column(bundle / table, column, value)


def _highrisk_expected(
    *, schedule_id="highrisk-d2-k32-r40-v1", depth_cap=2, master_seed=41,
):
    return HighRiskFacadeIdentity(
        campaign_id="fusion-v2-highrisk-ceiling-v1",
        phase="capability_ladder",
        split_role="exploratory_highrisk_ceiling_v1",
        schedule_id=schedule_id,
        depth_cap=depth_cap,
        master_seed=master_seed,
        dual_structure_profile_id=HIGH_RISK_DUAL_SCTM_PROFILE_ID,
    )


_HIGH_RISK_FROZEN_ROLES = frozenset({
    "code_revision",
    "complete_reference_sequence",
    "projection_policy_spec",
    "schedule_band_calibration",
    "head_config",
    "head_checkpoint",
    "structure_backend",
})
_HIGH_RISK_OBSERVED_ROLES = frozenset({
    "backbone",
    "cohort_table",
    "complete_reference_sequence",
    "constraint_manifest",
    "dplm_checkpoint",
    "head_checkpoint",
    "projection_policy_spec",
    "reference_sequences",
    "rf_sampler_config",
    "structure_backend",
    "structure_config",
    "v0_structure_gate_config",
})
_HIGH_RISK_GLOBAL_CONTENT = {
    "cohort_table": canonical_digest({"fixture": "cohort_table"}),
    "reference_sequences": canonical_digest({"fixture": "reference_sequences"}),
    "rf_sampler_config": canonical_digest({"fixture": "rf_sampler_config"}),
    "dplm_checkpoint": canonical_digest({"fixture": "dplm_checkpoint"}),
    "head_config": canonical_digest({"fixture": "head_config"}),
    "head_checkpoint": canonical_digest({"fixture": "head_checkpoint"}),
    "projection_policy_spec": canonical_digest({"fixture": "projection_policy_spec"}),
    "structure_backend": "a" * 64,
    "structure_config": "c" * 64,
    "v0_structure_gate_config": "b" * 64,
}


def _highrisk_role_digest(role: str, protein_id: str) -> str | None:
    if role == "code_revision":
        return "deadbeef"
    if role in _HIGH_RISK_GLOBAL_CONTENT:
        return _HIGH_RISK_GLOBAL_CONTENT[role]
    if role in {"coordinate_mask", "tokenizer", "fixed_token_policy"}:
        # These runtime conditioning identities are realized inside the shard and intentionally
        # are not standalone --input-file digests in the real driver manifest.
        return None
    return canonical_digest({"fixture": role, "protein_id": protein_id})


def _highrisk_content_rows(protein_id: str) -> list[dict]:
    rows = []
    for role in sorted(CONTENT_ROLE_TO_FIELD):
        frozen = role in _HIGH_RISK_FROZEN_ROLES
        rows.append({
            "role": role,
            "label": f"{role}.bin",
            "binding": "frozen" if frozen else "runtime",
            "expected_sha256": _highrisk_role_digest(role, protein_id) if frozen else None,
        })
    return rows


def _highrisk_content_evidence(config: dict, protein_id: str):
    provenance = []
    observed_rows = []
    identities = {}
    for declared in sorted(config["content"], key=lambda row: row["role"]):
        role = declared["role"]
        observed_digest = (
            _highrisk_role_digest(role, protein_id)
            if role in _HIGH_RISK_OBSERVED_ROLES else None
        )
        provenance.append({
            "role": role,
            "declared_label": declared["label"],
            "binding": declared["binding"],
            "declared_sha256": declared["expected_sha256"],
            "observed_path": None if observed_digest is None else f"/fixture/{role}",
            "observed_sha256": observed_digest,
        })
        established = observed_digest or declared["expected_sha256"]
        if established is not None:
            identities[role] = established
        if observed_digest is not None:
            observed_rows.append({"role": role, "sha256": observed_digest})
    encoded = json.dumps(
        sorted(observed_rows, key=lambda row: row["role"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return identities, provenance, hashlib.sha256(encoded).hexdigest()


def _replace_manifest_content_identity(manifest_path, role: str, digest: str) -> None:
    """Produce a self-consistent manifest for a genuinely different content realization."""
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(manifest["config_canonical_json"])
    config_row = next(row for row in config["content"] if row["role"] == role)
    provenance_row = next(
        row for row in manifest["content_provenance"] if row["role"] == role
    )
    if config_row["binding"] == "frozen":
        config_row["expected_sha256"] = digest
        provenance_row["declared_sha256"] = digest
    if provenance_row["observed_sha256"] is not None:
        provenance_row["observed_sha256"] = digest
    manifest["content_identities"][role] = digest
    observed_rows = [
        {"role": row["role"], "sha256": row["observed_sha256"]}
        for row in manifest["content_provenance"]
        if row["observed_sha256"] is not None
    ]
    encoded = json.dumps(
        sorted(observed_rows, key=lambda row: row["role"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest["input_signature"] = hashlib.sha256(encoded).hexdigest()
    manifest["config_canonical_json"] = json.dumps(
        config, sort_keys=True, separators=(",", ":"),
    )
    manifest["config_digest"] = canonical_digest(config)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))


def _rewrite_manifest_config(manifest_path, mutate) -> None:
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(manifest["config_canonical_json"])
    mutate(config)
    manifest["config_canonical_json"] = json.dumps(
        config, sort_keys=True, separators=(",", ":"),
    )
    manifest["config_digest"] = canonical_digest(config)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))


def _recompute_hotspot_source_ref(config: dict) -> None:
    threshold = config["safety"]["delta_new_cumulative"]
    threshold["source_ref"] = canonical_digest({
        key: threshold[key]
        for key in ("value", "unit", "source_kind", "source_id", "artifact")
    })


def _highrisk_config(
    protein_id: str,
    *,
    expected: HighRiskFacadeIdentity,
    method_overrides: dict | None = None,
):
    policy = {
        "policy_kind": DUAL_SCTM_POLICY_KIND,
        "profile_id": expected.dual_structure_profile_id,
        "ancestry_sctm_min": HIGH_RISK_ANCESTRY_SCTM_MIN,
        "strict_sctm_min": HIGH_RISK_STRICT_SCTM_MIN,
    }
    hotspot_artifact = {
        "schema_version": "v2-hotspot-calibration-1",
        "gate_kind": "whole_landscape_new_hotspot",
        "scope": "cumulative_depth0",
        "reference_kind": "native_wt",
        "reference_label": "wt_native",
        "window_domain": "whole_landscape",
        "allele": "DRB1_0701",
        "score_scale": "raw_logit",
        "window_k_min": 12,
        "window_k_max": 25,
        "calibration_data_digest": canonical_digest({"hotspot_rows": protein_id}),
    }
    hotspot_value = 10.0 + (sum(protein_id.encode("ascii")) % 7)
    hotspot = {
        "value": hotspot_value,
        "unit": "raw_logit",
        "source_kind": "measured_calibration",
        "source_id": (
            "v2-canary-hotspot-head-valid-q90-higher-v1:"
            f"{protein_id}"
        ),
        "artifact": hotspot_artifact,
    }
    hotspot["source_ref"] = canonical_digest({
        "value": hotspot_value,
        "unit": hotspot["unit"],
        "source_kind": hotspot["source_kind"],
        "source_id": hotspot["source_id"],
        "artifact": hotspot_artifact,
    })
    payload = {
        "schema": "v2cfg-1",
        "identity": {
            "campaign_id": expected.campaign_id,
            "split_role": expected.split_role,
            "phase": expected.phase,
            "master_seed": expected.master_seed,
            "seed_schema": "v2seed-1",
            "code_revision": "deadbeef",
        },
        "substrate": {
            "n_steps": 100,
            "temperature": 1.0,
            "amplification_form": "constant_one",
            "controller_enabled": False,
            "remask_enabled": True,
            "remask_fraction_scale": 0.0,
            "rf_config_label": "v2_null_no_remask",
        },
        "arm": {
            "feedback_enabled": True,
            "arm_role": "v2",
            "a2_matching_resource": "definitive_refolds",
            "a2_unmatched_reported": [
                "gpu_seconds", "head_calls", "logical_dfe", "walltime_s",
            ],
        },
        "schedule": {
            "schedule_id": expected.schedule_id,
            "coordinate_law": "progressive_checkpoint",
            "depth_cap": expected.depth_cap,
            "active_population_width": 1,
            "min_lookahead_tail_steps": 10,
            "stratum_key": f"highrisk_nod_v1_{protein_id}_unconstrained",
            "points": [
                {
                    "depth": depth,
                    "r_step": 40,
                    "c_source_step": (
                        50 + depth * 20 if expected.depth_cap == 2 else 50 + depth * 5
                    ),
                    "c_next_step": (
                        70 + depth * 20 if expected.depth_cap == 2 else 55 + depth * 5
                    ),
                    "n_lookaheads": 32,
                    "band_key": "step40",
                }
                for depth in range(expected.depth_cap)
            ],
        },
        "projection": {
            "support_policy_id": "source_writeback_v1",
            "support_policy_version": "v1",
            "support_policy_is_diagnostic": False,
            "temporal_history_rule": "commit_before_reentry",
            "assimilation_rule": "first_forward_raw_logits",
            "admissible_mask_load_unit": "absolute_positions",
            "head_directed": None,
        },
        "head": {
            "allele": "DRB1_0701",
            "score_scale": "raw_logit",
            "window_k_min": 12,
            "window_k_max": 25,
            "head_variant_id": "LC1",
            "head_allele_idx": 0,
            "head_window_batch_size": 64,
        },
        "safety": {
            "cumulative_reference_kind": "native_wt",
            "cumulative_reference_label": "wt_native",
            "delta_new_cumulative": hotspot,
            "incremental_gate_enabled": False,
            "delta_new_incremental": None,
            "structure_cadence": "every_endpoint",
            "search_structure": policy,
        },
        "caps": {
            "max_logical_dfe": 3072 if expected.depth_cap == 2 else 9216,
            "max_head_calls": 1024 if expected.depth_cap == 2 else 4096,
            "max_definitive_refolds": 128 if expected.depth_cap == 2 else 320,
            "max_gpu_seconds": 14400 if expected.depth_cap == 2 else 43200,
            "max_walltime_s": 14400 if expected.depth_cap == 2 else 43200,
            "max_retries": 2,
            "retry_scope": "per_request",
        },
        # Mirrors the producer's full 18-role config surface.  The runtime manifest below remains
        # intentionally sparse where no standalone --input-file digest exists.
        "content": _highrisk_content_rows(protein_id),
    }
    for dotted, value in (method_overrides or {}).items():
        node = payload
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return payload


def _dual_columns(*, endpoint_id: str, sctm: float, strict_passed: bool | None = None):
    strict = sctm >= HIGH_RISK_STRICT_SCTM_MIN if strict_passed is None else strict_passed
    ancestry = sctm >= HIGH_RISK_ANCESTRY_SCTM_MIN
    raw_digest = canonical_digest({"endpoint_id": endpoint_id, "raw_sctm": sctm})
    verdict = {
        "profile_id": HIGH_RISK_DUAL_SCTM_PROFILE_ID,
        "ancestry_sctm_min": HIGH_RISK_ANCESTRY_SCTM_MIN,
        "strict_sctm_min": HIGH_RISK_STRICT_SCTM_MIN,
        "evaluated": True,
        "common_structure_feasible": True,
        "sctm": sctm,
        "ancestry_structure_passed": ancestry,
        "strict_structure_passed": strict,
        "ancestry_reason": "passed" if ancestry else "sctm_below_0.7",
        "strict_reason": "passed" if strict else "sctm_below_0.85",
        "raw_outcome_digest": raw_digest,
    }
    policy = {
        "policy_kind": DUAL_SCTM_POLICY_KIND,
        "profile_id": HIGH_RISK_DUAL_SCTM_PROFILE_ID,
        "ancestry_sctm_min": HIGH_RISK_ANCESTRY_SCTM_MIN,
        "strict_sctm_min": HIGH_RISK_STRICT_SCTM_MIN,
    }
    return {
        "structure_metrics_json": json.dumps({"scTM": sctm}, sort_keys=True),
        "dual_structure_profile_id": HIGH_RISK_DUAL_SCTM_PROFILE_ID,
        "dual_structure_policy_digest": canonical_digest(policy),
        "dual_structure_raw_outcome_digest": raw_digest,
        "dual_structure_ancestry_sctm_min": HIGH_RISK_ANCESTRY_SCTM_MIN,
        "dual_structure_strict_sctm_min": HIGH_RISK_STRICT_SCTM_MIN,
        "dual_structure_sctm": sctm,
        "dual_structure_common_feasible": True,
        "dual_structure_ancestry_passed": ancestry,
        "dual_structure_strict_passed": strict,
        "dual_structure_verdict_digest": canonical_digest(verdict),
        "dual_structure_verdict_json": json.dumps(verdict, sort_keys=True),
        "ancestry_authorization_digest": canonical_digest({"authorization": endpoint_id}),
        "admission_evidence_digest": canonical_digest({"admission": endpoint_id}),
    }


def _write_highrisk_root_bundle(
    root,
    *,
    protein_id: str,
    root_index: int,
    endpoint_specs: list[dict] | None = None,
    expected: HighRiskFacadeIdentity | None = None,
    method_overrides: dict | None = None,
    manifest_overrides: dict | None = None,
):
    expected = expected or _highrisk_expected()
    endpoint_specs = endpoint_specs or [{}]
    prepared = []
    for index, overrides in enumerate(endpoint_specs):
        endpoint_id = overrides.get(
            "endpoint_id", f"endpoint:{protein_id}:root{root_index}:{index}",
        )
        sctm = overrides.get("dual_structure_sctm", 0.90)
        dual = _dual_columns(
            endpoint_id=endpoint_id,
            sctm=sctm,
            strict_passed=overrides.get("dual_structure_strict_passed"),
        )
        spec = {
            "endpoint_id": endpoint_id,
            "head_global_risk": float(root_index * 10 + index),
            "head_window_grid_digest": f"grid:{protein_id}",
            "is_elite": index == 0,
            **dual,
            **overrides,
        }
        strict = bool(spec["dual_structure_strict_passed"])
        spec.setdefault("feasibility_level", "definitive" if strict else "provisional")
        spec.setdefault("archive_feasibility_level", "definitive" if strict else "provisional")
        prepared.append(spec)

    bundle = _write_bundle(root, protein_id=protein_id, endpoints=prepared)
    complete_path = bundle / "complete_endpoints.parquet"
    complete = pd.read_parquet(complete_path)
    for index, spec in enumerate(prepared):
        complete.loc[index, "root_id"] = spec.get(
            "root_id", f"{protein_id}:v2:d0:r{root_index}",
        )
        complete.loc[index, "family_id"] = spec.get(
            "family_id", f"fam{root_index}",
        )
        for field, value in _dual_columns(
            endpoint_id=spec["endpoint_id"],
            sctm=spec["dual_structure_sctm"],
            strict_passed=spec["dual_structure_strict_passed"],
        ).items():
            complete.loc[index, field] = spec.get(field, value)
    complete.to_parquet(complete_path, index=False)

    archive_path = bundle / "archive.parquet"
    archive = pd.read_parquet(archive_path)
    archive["root_id"] = complete["root_id"]
    archive["family_id"] = complete["family_id"]
    dual_columns = [
        column for column in complete.columns
        if column.startswith("dual_structure_")
        or column in {"ancestry_authorization_digest", "admission_evidence_digest"}
    ]
    for column in dual_columns:
        archive[column] = complete[column]
    archive.to_parquet(archive_path, index=False)

    structure = pd.DataFrame({
        "endpoint_id": complete["endpoint_id"],
        "protein_id": complete["protein_id"],
        "depth": complete["depth"],
        "sequence_md5": complete["sequence_md5"],
        "evaluated": complete["structure_evaluated"],
        "feasible": complete["structure_feasible"],
        "metrics_json": complete["structure_metrics_json"],
        "feasibility_level": complete["feasibility_level"],
        "structure_backend_digest": "a" * 64,
        "v0_structure_gate_config_digest": "b" * 64,
    })
    for column in dual_columns:
        structure[column] = complete[column]
    structure.to_parquet(bundle / "structure_evaluations.parquet", index=False)

    config = _highrisk_config(
        protein_id, expected=expected, method_overrides=method_overrides,
    )
    manifest = json.loads((bundle / "run_manifest.json").read_text())
    content_identities, content_provenance, input_signature = _highrisk_content_evidence(
        config, protein_id,
    )
    manifest.update({
        "campaign_id": expected.campaign_id,
        "phase": expected.phase,
        "split_role": expected.split_role,
        "schedule_id": expected.schedule_id,
        "depth_cap": expected.depth_cap,
        "master_seed": expected.master_seed,
        "active_population_width": config["schedule"]["active_population_width"],
        "root_index": root_index,
        "root_seeds_by_protein": {protein_id: 1000 + 10 * root_index + len(protein_id)},
        "n_ok": int(any(
            spec["feasibility_level"] == "definitive" for spec in prepared
        )),
        "n_complete_negative": int(not any(
            spec["feasibility_level"] == "definitive" for spec in prepared
        )),
        "fragment_result_status_by_protein": {
            protein_id: (
                "ok" if any(spec["feasibility_level"] == "definitive" for spec in prepared)
                else "complete_negative"
            ),
        },
        "content_identities": content_identities,
        "content_provenance": content_provenance,
        "input_signature": input_signature,
        "config_canonical_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "config_digest": canonical_digest(config),
    })
    for key, value in (manifest_overrides or {}).items():
        manifest[key] = value
    (bundle / "run_manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return bundle


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


def _r4_bundles(tmp_path, *, proteins=("P1",), expected=None, specs_by_cell=None):
    expected = expected or _highrisk_expected()
    specs_by_cell = specs_by_cell or {}
    return [
        _write_highrisk_root_bundle(
            tmp_path / f"{protein_id}.root{root_index}",
            protein_id=protein_id,
            root_index=root_index,
            expected=expected,
            endpoint_specs=specs_by_cell.get((protein_id, root_index)),
        )
        for protein_id in proteins
        for root_index in range(4)
    ]


def test_highrisk_elite_pools_all_four_roots_and_allows_per_protein_head_grids(tmp_path):
    specs = {
        ("P1", 0): [{"head_global_risk": -1.0}],
        ("P1", 1): [{"head_global_risk": -4.0}],
        ("P1", 2): [{"head_global_risk": -2.0}],
        ("P1", 3): [{"head_global_risk": -3.0}],
        ("P2", 0): [{"head_global_risk": -8.0}],
        ("P2", 1): [{"head_global_risk": -5.0}],
        ("P2", 2): [{"head_global_risk": -7.0}],
        ("P2", 3): [{"head_global_risk": -6.0}],
    }
    bundles = _r4_bundles(
        tmp_path, proteins=("P1", "P2"), specs_by_cell=specs,
    )

    out = materialize_archive_facade(
        bundles, mode="elite", contract="high-risk-r4",
        expected_identity=_highrisk_expected(),
    )

    assert out[["protein_id", "root_index", "head_global_risk"]].to_dict("records") == [
        {"protein_id": "P1", "root_index": 1, "head_global_risk": -4.0},
        {"protein_id": "P2", "root_index": 0, "head_global_risk": -8.0},
    ]
    assert out["head_evaluator_digest"].nunique() == 1
    assert out.groupby("protein_id")["head_window_grid_digest"].nunique().eq(1).all()
    assert out["head_window_grid_digest"].nunique() == 2
    assert set(out["dual_structure_profile_id"]) == {HIGH_RISK_DUAL_SCTM_PROFILE_ID}
    assert set(out["root_seed"]) == {1012, 1002}
    assert out["config_digest"].nunique() == 2  # legitimate per-protein calibration/config
    assert out["method_profile_digest"].nunique() == 1


def test_highrisk_top_k_deduplicates_sequences_after_pooling_roots(tmp_path):
    bundles = _r4_bundles(tmp_path, specs_by_cell={
        ("P1", 0): [{"sequence": "ACDEFGHIK", "head_global_risk": -2.0}],
        ("P1", 1): [{"sequence": "ACDEFGHIK", "head_global_risk": -5.0}],
        ("P1", 2): [{"sequence": "LMNPQRSTV", "head_global_risk": -4.0}],
        ("P1", 3): [{"sequence": "WYACDEFGH", "head_global_risk": -3.0}],
    })

    out = materialize_archive_facade(
        bundles, mode="top-k", k=3, require_k=True, contract="high-risk-r4",
        expected_identity=_highrisk_expected(),
    )

    assert list(out["root_index"]) == [1, 2, 3]
    assert list(out["sequence"]) == ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH"]


def test_highrisk_contract_materializes_d8_as_a_separate_explicit_profile(tmp_path):
    expected = _highrisk_expected(schedule_id="highrisk-d8-k32-r40-v1", depth_cap=8)
    bundles = _r4_bundles(tmp_path, expected=expected)
    out = materialize_archive_facade(
        bundles, mode="elite", contract="high-risk-r4", expected_identity=expected,
    )
    assert out["depth_cap"].tolist() == [8]
    assert out["schedule_id"].tolist() == ["highrisk-d8-k32-r40-v1"]


@pytest.mark.parametrize("missing_root", [0, 1, 2, 3])
def test_highrisk_requires_exactly_one_complete_cell_for_each_r4_root(tmp_path, missing_root):
    bundles = _r4_bundles(tmp_path)
    bundles = [path for path in bundles if not path.name.endswith(f"root{missing_root}")]
    with pytest.raises(V2FacadeError, match=r"roots.*0, 1, 2, 3|root coverage"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_refuses_duplicate_root_cell_and_duplicate_root_seed(tmp_path):
    bundles = _r4_bundles(tmp_path)
    duplicate = _write_highrisk_root_bundle(
        tmp_path / "P1.duplicate.root0", protein_id="P1", root_index=0,
    )
    with pytest.raises(V2FacadeError, match="root 0.*more than one bundle|duplicate root"):
        materialize_archive_facade(
            [*bundles, duplicate], mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )

    manifest_path = bundles[1] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["root_seeds_by_protein"]["P1"] = 1002
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(V2FacadeError, match="root seed.*unique"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_requires_endpoint_lineage_to_bind_the_manifest_root_index(tmp_path):
    bundles = _r4_bundles(tmp_path)
    endpoint_path = bundles[0] / "complete_endpoints.parquet"
    archive_path = bundles[0] / "archive.parquet"
    _replace_column(endpoint_path, "root_id", "P1:v2:d0:r3")
    _replace_column(archive_path, "root_id", "P1:v2:d0:r3")
    with pytest.raises(V2FacadeError, match="lineage.*root_index=0"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_excludes_zero_strict_protein_but_records_it(tmp_path):
    specs = {
        ("P2", root_index): [{
            "dual_structure_sctm": 0.80,
            "dual_structure_strict_passed": False,
        }]
        for root_index in range(4)
    }
    bundles = _r4_bundles(
        tmp_path, proteins=("P1", "P2"), specs_by_cell=specs,
    )

    out = materialize_archive_facade(
        bundles, mode="elite", contract="high-risk-r4",
        expected_identity=_highrisk_expected(),
    )

    assert set(out["protein_id"]) == {"P1"}
    provenance = out.attrs["selection_provenance"]
    assert provenance["excluded_proteins"] == [
        {"protein_id": "P2", "reason": "zero_strict_endpoints"},
    ]
    assert provenance["root_indices"] == [0, 1, 2, 3]


def test_highrisk_requires_each_declared_root_cell_to_have_a_closed_result_status(tmp_path):
    bundles = _r4_bundles(tmp_path)
    manifest_path = bundles[0] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fragment_result_status_by_protein"]["P1"] = "failed"
    manifest["n_ok"] = 0
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(V2FacadeError, match="closed|complete_negative|failed"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_manifest_status_must_agree_with_the_realized_strict_pool(tmp_path):
    bundles = _r4_bundles(tmp_path, specs_by_cell={
        ("P1", 0): [{
            "dual_structure_sctm": 0.80,
            "dual_structure_strict_passed": False,
        }],
    })
    manifest_path = bundles[0] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fragment_result_status_by_protein"]["P1"] = "ok"
    manifest["n_ok"] = 1
    manifest["n_complete_negative"] = 0
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(V2FacadeError, match="status.*strict|strict.*status"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_strict_structure_with_failed_immune_admission_is_excluded_not_malformed(
    tmp_path,
):
    specs = {
        ("P1", root_index): [{
            "dual_structure_sctm": 0.90,
            "dual_structure_strict_passed": True,
            "feasibility_level": "unvalidated",
            "archive_feasibility_level": "unvalidated",
            "ancestry_authorization_digest": None,
        }]
        for root_index in range(4)
    }
    bundles = _r4_bundles(tmp_path, specs_by_cell=specs)
    for bundle in bundles:
        endpoint_path = bundle / "complete_endpoints.parquet"
        endpoints = pd.read_parquet(endpoint_path)
        endpoints["structure_evaluated"] = False
        endpoints["structure_feasible"] = False
        endpoints["structure_metrics_json"] = "{}"
        endpoints.to_parquet(endpoint_path, index=False)

    out = materialize_archive_facade(
        bundles, mode="elite", contract="high-risk-r4",
        expected_identity=_highrisk_expected(),
    )
    assert out.empty
    assert out.attrs["selection_provenance"]["excluded_proteins"] == [
        {"protein_id": "P1", "reason": "zero_strict_endpoints"},
    ]


def test_highrisk_keeps_rejected_raw_structure_evidence_out_of_the_strict_facade(tmp_path):
    bundles = _r4_bundles(tmp_path, specs_by_cell={
        ("P1", root_index): [{
            "dual_structure_sctm": 0.60,
            "dual_structure_strict_passed": False,
            "feasibility_level": "unvalidated",
            "archive_feasibility_level": "unvalidated",
            "ancestry_authorization_digest": None,
        }]
        for root_index in range(4)
    })
    # This is the producer's intentional split: raw rejected evidence lives in the structure table
    # and the dual columns, while the unvalidated complete endpoint carries no StructureOutcome.
    for bundle in bundles:
        endpoint_path = bundle / "complete_endpoints.parquet"
        endpoints = pd.read_parquet(endpoint_path)
        endpoints["structure_evaluated"] = False
        endpoints["structure_feasible"] = False
        endpoints["structure_metrics_json"] = "{}"
        endpoints.to_parquet(endpoint_path, index=False)

    out = materialize_archive_facade(
        bundles, mode="elite", contract="high-risk-r4",
        expected_identity=_highrisk_expected(),
    )
    assert out.empty
    assert out.attrs["selection_provenance"]["excluded_proteins"] == [
        {"protein_id": "P1", "reason": "zero_strict_endpoints"},
    ]


def test_highrisk_requires_one_cross_checked_structure_evaluation_per_endpoint(tmp_path):
    bundles = _r4_bundles(tmp_path)
    (bundles[0] / "structure_evaluations.parquet").unlink()
    with pytest.raises(V2FacadeError, match="structure_evaluations"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_refuses_dual_evidence_forged_in_only_one_table(tmp_path):
    bundles = _r4_bundles(tmp_path)
    _replace_column(
        bundles[0] / "archive.parquet", "dual_structure_verdict_digest", "f" * 64,
    )
    with pytest.raises(V2FacadeError, match="cross-table dual evidence mismatch"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("dual_structure_sctm", 0.84, "strict.*scTM|scTM.*strict"),
        ("dual_structure_common_feasible", False, "common.*feasible"),
        ("dual_structure_ancestry_passed", False, "strict.*ancestry"),
        ("dual_structure_policy_digest", "f" * 64, "policy digest"),
        ("dual_structure_verdict_digest", "e" * 64, "verdict.*digest"),
        ("ancestry_authorization_digest", None, "ancestry authorization"),
        ("admission_evidence_digest", None, "admission evidence"),
    ],
)
def test_highrisk_strict_candidates_require_bound_dual_evidence(
    tmp_path, column, value, message,
):
    bundles = _r4_bundles(tmp_path)
    _replace_highrisk_evidence(bundles[0], column, value)
    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_refuses_a_nonfinite_strict_sctm(tmp_path):
    bundles = _r4_bundles(tmp_path)
    _replace_highrisk_evidence(bundles[0], "dual_structure_sctm", float("nan"))
    with pytest.raises(V2FacadeError, match="finite.*scTM|scTM.*finite"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_requires_same_head_evaluator_globally_but_grid_only_within_protein(tmp_path):
    bundles = _r4_bundles(tmp_path, proteins=("P1", "P2"))
    _replace_column(
        bundles[-1] / "complete_endpoints.parquet", "head_evaluator_digest", "other-head",
    )
    with pytest.raises(V2FacadeError, match="Head evaluator"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )

    bundles = _r4_bundles(tmp_path / "grid-case")
    _replace_column(
        bundles[-1] / "complete_endpoints.parquet", "head_window_grid_digest", "other-grid",
    )
    with pytest.raises(V2FacadeError, match="window-grid.*protein|protein.*window-grid"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_requires_complete_config_per_protein_and_common_method_across_proteins(tmp_path):
    bundles = _r4_bundles(tmp_path)
    manifest_path = bundles[-1] / "run_manifest.json"
    _replace_manifest_content_identity(
        manifest_path, "schedule_band_calibration", "9" * 64,
    )
    with pytest.raises(V2FacadeError, match="same protein.*config|config identity"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_hotspot_threshold_identity_is_protein_bound_but_its_law_is_method_bound(
    tmp_path,
):
    # The ordinary two-protein fixture already has different threshold values, the exact
    # protein suffix of source_id, source_refs and calibration-data digests; those are legal.
    # The estimator prefix, WT reference label and schedule lookup band remain frozen.
    bundles = _r4_bundles(tmp_path / "legal", proteins=("P1", "P2"))
    out = materialize_archive_facade(
        bundles, mode="elite", contract="high-risk-r4",
        expected_identity=_highrisk_expected(),
    )
    assert out["config_digest"].nunique() == 2
    assert out["method_profile_digest"].nunique() == 1

    # Changing the estimator's declared source kind is a method change, even after producing a
    # self-consistent new source_ref.  It may not hide inside "per-protein calibration".
    bundles = _r4_bundles(tmp_path / "law-change", proteins=("P1", "P2"))
    for bundle in bundles[4:]:
        path = bundle / "run_manifest.json"
        manifest = json.loads(path.read_text())
        config = json.loads(manifest["config_canonical_json"])
        threshold = config["safety"]["delta_new_cumulative"]
        threshold["source_kind"] = "runbook_frozen"
        threshold["source_ref"] = canonical_digest({
            key: threshold[key]
            for key in ("value", "unit", "source_kind", "source_id", "artifact")
        })
        manifest["config_canonical_json"] = json.dumps(
            config, sort_keys=True, separators=(",", ":"),
        )
        manifest["config_digest"] = canonical_digest(config)
        path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(V2FacadeError, match="method profile"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_accepts_the_real_sparse_driver_content_identity_surface(tmp_path):
    bundles = _r4_bundles(tmp_path)
    manifest = json.loads((bundles[0] / "run_manifest.json").read_text())

    assert len(manifest["content_provenance"]) == len(CONTENT_ROLE_TO_FIELD) == 18
    assert {
        "coordinate_mask", "tokenizer", "fixed_token_policy",
    }.isdisjoint(manifest["content_identities"])
    assert set(manifest["content_identities"]) == {
        row["role"] for row in manifest["content_provenance"]
        if row["observed_sha256"] is not None or row["declared_sha256"] is not None
    }

    out = materialize_archive_facade(
        bundles, mode="elite", contract="high-risk-r4",
        expected_identity=_highrisk_expected(),
    )
    assert out["protein_id"].tolist() == ["P1"]


def test_highrisk_fixture_is_a_real_typed_driver_canonical_config():
    from inverse_folding.reference_flow.fusion_v2.config import load_v2_config

    canonical = _highrisk_config("P1", expected=_highrisk_expected())
    launch_payload = json.loads(json.dumps(canonical))
    launch_payload["schema_version"] = launch_payload.pop("schema")
    launch_payload["projection"].pop("head_directed")
    launch_payload["safety"].pop("delta_new_incremental")

    resolved = load_v2_config(launch_payload)
    assert resolved.canonical_payload() == canonical
    assert resolved.config_digest() == canonical_digest(canonical)


def test_highrisk_schedule_band_key_is_method_bound_not_a_protein_label(tmp_path):
    bundles = _r4_bundles(tmp_path, proteins=("P1", "P2"))
    for bundle in bundles[4:]:
        _rewrite_manifest_config(
            bundle / "run_manifest.json",
            lambda config: config["schedule"]["points"][0].__setitem__(
                "band_key", "step30",
            ),
        )

    with pytest.raises(V2FacadeError, match="method profile"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_wt_reference_label_is_frozen(tmp_path):
    bundles = _r4_bundles(tmp_path, proteins=("P1", "P2"))

    def change_reference(config):
        config["safety"]["cumulative_reference_label"] = "native:P2"
        config["safety"]["delta_new_cumulative"]["artifact"][
            "reference_label"
        ] = "native:P2"
        _recompute_hotspot_source_ref(config)

    for bundle in bundles[4:]:
        _rewrite_manifest_config(bundle / "run_manifest.json", change_reference)

    with pytest.raises(V2FacadeError, match="native_wt/wt_native|reference"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_hotspot_source_population_prefix_is_frozen(tmp_path):
    bundles = _r4_bundles(tmp_path)

    def change_population(config):
        config["safety"]["delta_new_cumulative"]["source_id"] = (
            "v2-canary-hotspot-head-valid-q95-higher-v1:P1"
        )
        _recompute_hotspot_source_ref(config)

    _rewrite_manifest_config(bundles[0] / "run_manifest.json", change_population)
    with pytest.raises(V2FacadeError, match="full-trajectory Q90 population|source_id"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_hotspot_source_ref_must_digest_the_full_measurement(tmp_path):
    bundles = _r4_bundles(tmp_path)
    _rewrite_manifest_config(
        bundles[0] / "run_manifest.json",
        lambda config: config["safety"]["delta_new_cumulative"].__setitem__(
            "source_ref", "0" * 64,
        ),
    )
    with pytest.raises(V2FacadeError, match="source_ref.*digest"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


@pytest.mark.parametrize("role", ["cohort_table", "reference_sequences"])
def test_highrisk_campaign_wide_inputs_must_match_across_proteins(tmp_path, role):
    bundles = _r4_bundles(tmp_path, proteins=("P1", "P2"))
    for bundle in bundles[4:]:
        _replace_manifest_content_identity(bundle / "run_manifest.json", role, "8" * 64)

    with pytest.raises(V2FacadeError, match="campaign-wide|cohort/reference"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_same_protein_roots_must_share_the_full_observed_input_signature(tmp_path):
    bundles = _r4_bundles(tmp_path)
    _replace_manifest_content_identity(
        bundles[-1] / "run_manifest.json", "backbone", "7" * 64,
    )

    with pytest.raises(V2FacadeError, match=r"observed input identity.*roots 0\.\.3"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_realized_model_content_is_part_of_the_method_profile(tmp_path):
    bundles = _r4_bundles(tmp_path, proteins=("P1", "P2"))
    for bundle in bundles[4:]:
        path = bundle / "run_manifest.json"
        _replace_manifest_content_identity(path, "dplm_checkpoint", "9" * 64)
    with pytest.raises(V2FacadeError, match="method profile"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_schedule_coordinates_and_lookahead_k_are_method_bound(tmp_path):
    bundles = _r4_bundles(tmp_path / "method-case", proteins=("P1", "P2"))
    for path in bundles[4:]:
        manifest_path = path / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        config = json.loads(manifest["config_canonical_json"])
        config["schedule"]["points"][0]["n_lookaheads"] = 8
        manifest["config_canonical_json"] = json.dumps(
            config, sort_keys=True, separators=(",", ":"),
        )
        manifest["config_digest"] = canonical_digest(config)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(V2FacadeError, match="method profile"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_refuses_missing_or_forged_canonical_config_identity(tmp_path):
    bundles = _r4_bundles(tmp_path)
    manifest_path = bundles[0] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("config_canonical_json")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(V2FacadeError, match="config_canonical_json"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )

    bundles = _r4_bundles(tmp_path / "forged")
    manifest_path = bundles[0] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(V2FacadeError, match="config_digest"):
        materialize_archive_facade(
            bundles, mode="elite", contract="high-risk-r4",
            expected_identity=_highrisk_expected(),
        )


def test_highrisk_cli_requires_explicit_identity_and_writes_selection_provenance(tmp_path):
    bundles = _r4_bundles(tmp_path)
    output = tmp_path / "generation" / "generated.parquet"
    with pytest.raises(SystemExit, match="expected identity"):
        main([
            *(arg for bundle in bundles for arg in ("--bundle", str(bundle))),
            "--contract", "high-risk-r4", "--mode", "elite", "--output", str(output),
        ])

    expected = _highrisk_expected()
    assert main([
        *(arg for bundle in bundles for arg in ("--bundle", str(bundle))),
        "--contract", "high-risk-r4",
        "--expected-campaign-id", expected.campaign_id,
        "--expected-phase", expected.phase,
        "--expected-split-role", expected.split_role,
        "--expected-schedule-id", expected.schedule_id,
        "--expected-depth-cap", str(expected.depth_cap),
        "--expected-master-seed", str(expected.master_seed),
        "--expected-dual-profile-id", expected.dual_structure_profile_id,
        "--mode", "elite", "--output", str(output),
    ]) == 0
    sidecar = output.with_suffix(output.suffix + ".selection.json")
    assert sidecar.is_file()
    provenance = json.loads(sidecar.read_text())
    assert provenance["contract"] == "high-risk-r4"
    assert provenance["root_indices"] == [0, 1, 2, 3]
