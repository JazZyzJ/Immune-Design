#!/usr/bin/env python3
"""Materialize an exact V2 archive selection as a Phase-C/v0 ``generated.parquet``.

The archive table is a membership view, not sequence evidence.  This exporter therefore joins it
to ``complete_endpoints.parquet`` on the persisted ``endpoint_id`` and validates the redundant
content/protein/feasibility identities before selecting anything.  It never reconstructs lineage
from sequence bytes.  The additive ``high-risk-r4`` contract also requires the raw
``structure_evaluations.parquet`` evidence, exact roots 0..3, a caller-specified experiment
identity and the signed dual-scTM profile before globally pooling strict endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion.state import sequence_md5  # noqa: E402
from inverse_folding.reference_flow.fusion.v1_alloc import (  # noqa: E402
    HeadInputError,
    validate_complete_aa20,
)
from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402
from inverse_folding.reference_flow.fusion_v2.identity import canonical_digest  # noqa: E402
from inverse_folding.reference_flow.fusion_v2.identity import (  # noqa: E402
    CONTENT_ROLE_TO_FIELD, FROZEN_DIGEST_ROLES,
)
from inverse_folding.reference_flow.fusion_v2.structure_gate import (  # noqa: E402
    DUAL_SCTM_POLICY_KIND,
    HIGH_RISK_ANCESTRY_SCTM_MIN,
    HIGH_RISK_STRICT_SCTM_MIN,
)

__all__ = [
    "HighRiskFacadeIdentity", "V2FacadeError", "materialize_archive_facade", "main",
]


class V2FacadeError(V2Error):
    """The named evidence bundles cannot support an exact facade selection."""


@dataclass(frozen=True)
class HighRiskFacadeIdentity:
    """Caller-supplied identity of one D2 *or* D8 high-risk R4 experiment.

    D2 and D8 are deliberately separate materializations.  Requiring every label here prevents
    the exporter from deciding that a directory merely *looks like* the intended campaign.
    """

    campaign_id: str
    phase: str
    split_role: str
    schedule_id: str
    depth_cap: int
    master_seed: int
    dual_structure_profile_id: str

    def __post_init__(self) -> None:
        for field in (
            "campaign_id", "phase", "split_role", "schedule_id",
            "dual_structure_profile_id",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise V2FacadeError(f"expected identity {field} must be a non-empty string")
        if isinstance(self.depth_cap, bool) or not isinstance(self.depth_cap, int) \
                or self.depth_cap < 2:
            raise V2FacadeError("expected identity depth_cap must be an integer >= 2")
        if isinstance(self.master_seed, bool) or not isinstance(self.master_seed, int) \
                or self.master_seed < 0:
            raise V2FacadeError("expected identity master_seed must be a non-negative integer")


_ARCHIVE_REQUIRED = {
    "endpoint_id",
    "endpoint_content_digest",
    "protein_id",
    "sequence_equivalence_key",
    "feasibility_level",
    "is_elite",
    "elite_rank",
}
_ENDPOINT_REQUIRED = {
    "endpoint_id",
    "endpoint_content_digest",
    "protein_id",
    "depth",
    "sequence",
    "sequence_md5",
    "sequence_equivalence_key",
    "sequence_length",
    "head_evaluator_digest",
    "head_window_grid_digest",
    "head_global_risk",
    "feasibility_level",
    "structure_evaluated",
    "structure_feasible",
}
_DUAL_EVIDENCE_COLUMNS = {
    "root_id",
    "family_id",
    "dual_structure_profile_id",
    "dual_structure_policy_digest",
    "dual_structure_raw_outcome_digest",
    "dual_structure_ancestry_sctm_min",
    "dual_structure_strict_sctm_min",
    "dual_structure_sctm",
    "dual_structure_common_feasible",
    "dual_structure_ancestry_passed",
    "dual_structure_strict_passed",
    "dual_structure_verdict_digest",
    "ancestry_authorization_digest",
    "admission_evidence_digest",
}
_HIGH_RISK_ARCHIVE_REQUIRED = _ARCHIVE_REQUIRED | _DUAL_EVIDENCE_COLUMNS
_HIGH_RISK_ENDPOINT_REQUIRED = _ENDPOINT_REQUIRED | _DUAL_EVIDENCE_COLUMNS | {
    "structure_metrics_json",
}
_HIGH_RISK_STRUCTURE_REQUIRED = {
    "endpoint_id",
    "protein_id",
    "depth",
    "sequence_md5",
    "evaluated",
    "feasible",
    "metrics_json",
    "feasibility_level",
    "structure_backend_digest",
    "v0_structure_gate_config_digest",
    *(_DUAL_EVIDENCE_COLUMNS - {"root_id", "family_id"}),
}
_OUTPUT_COLUMNS = (
    "protein_id",
    "design_idx",
    "sequence",
    "entry_source_id",
    "endpoint_id",
    "endpoint_content_digest",
    "depth",
    "head_global_risk",
    "head_evaluator_digest",
    "head_window_grid_digest",
    "sequence_md5",
)
_HIGH_RISK_OUTPUT_COLUMNS = (
    *_OUTPUT_COLUMNS,
    "root_index",
    "root_seed",
    "root_id",
    "family_id",
    "source_bundle",
    "config_digest",
    "method_profile_digest",
    "schedule_id",
    "depth_cap",
    "dual_structure_profile_id",
    "dual_structure_policy_digest",
    "dual_structure_ancestry_sctm_min",
    "dual_structure_strict_sctm_min",
    "dual_structure_verdict_digest",
    "dual_structure_raw_outcome_digest",
    "dual_structure_sctm",
    "ancestry_authorization_digest",
    "admission_evidence_digest",
    "archive_is_elite",
)
_HIGH_RISK_ROOT_INDICES = frozenset(range(4))
_HIGH_RISK_COMPLETED_STATUSES = frozenset({"ok", "complete_negative"})
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_METHOD_CONTENT_ROLES = frozenset({
    "dplm_checkpoint",
    "rf_sampler_config",
    "head_config",
    "head_checkpoint",
    "projection_policy_spec",
})
_CAMPAIGN_CONTENT_ROLES = frozenset({"cohort_table", "reference_sequences"})
_HIGH_RISK_HOTSPOT_SOURCE_PREFIX = "v2-canary-hotspot-head-valid-q90-higher-v1"

# These fields define the one exploratory experiment whose per-protein bundles may be merged into
# a cohort facade.  Protein-bound content (backbone, constraints, band cell, safety reference and
# the resolved config digest) is deliberately absent: requiring those to match would make every
# legitimate multi-protein cohort unmergeable.
_COMMON_MANIFEST_FIELDS = (
    "campaign_id",
    "phase",
    "split_role",
    "code_revision",
    "schedule_id",
    "coordinate_law",
    "depth_cap",
    "exploratory_depth_override",
    "production_depth_authorized",
    "structure_backend",
    "v0_structure_gate_config",
    "structure_config",
)
_REQUIRED_STRUCTURE_CONTENT = ("structure_backend", "v0_structure_gate_config")


def _bundle_root(raw: str | Path) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.is_file() and path.name in {"archive.parquet", "complete_endpoints.parquet"}:
        path = path.parent
    if not path.is_dir():
        raise V2FacadeError(f"V2 bundle is not a directory: {path}")
    return path


def _required_manifest_text(manifest: Mapping[str, Any], field: str, *, path: Path) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise V2FacadeError(f"{path} requires non-empty string {field}")
    return value


def _read_manifest(root: Path) -> tuple[Path, Mapping[str, Any]]:
    path = root / "run_manifest.json"
    if not path.is_file():
        raise V2FacadeError(f"V2 bundle is missing run_manifest.json: {root}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V2FacadeError(f"cannot read valid JSON run_manifest {path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise V2FacadeError(f"{path} must contain a JSON object")
    return path, manifest


def _load_manifest_identity(root: Path) -> dict[str, Any]:
    """Read one bundle's run identity before any endpoint tables are merged.

    The exporter is intentionally specific to the unblinded uricase capability ladder.  A bundle
    from qualification, production authorization, or another exploratory split is valid evidence
    for its own question, but it is not a cell in this cohort and must not be made one by a parquet
    concatenation.
    """
    path, manifest = _read_manifest(root)

    phase = _required_manifest_text(manifest, "phase", path=path)
    if phase != "capability_ladder":
        raise V2FacadeError(
            f"{path} phase must be 'capability_ladder', got {phase!r}"
        )
    split_role = _required_manifest_text(manifest, "split_role", path=path)
    if split_role != "exploratory_uricase":
        raise V2FacadeError(
            f"{path} split_role must be 'exploratory_uricase', got {split_role!r}"
        )
    if manifest.get("exploratory_depth_override") is not True:
        raise V2FacadeError(
            f"{path} requires exploratory_depth_override=true; an ordinary or production D>1 "
            "run cannot enter this exploratory facade"
        )
    if manifest.get("production_depth_authorized") is not False:
        raise V2FacadeError(
            f"{path} requires production_depth_authorized=false; exploratory and production "
            "evidence are non-interchangeable"
        )

    depth_cap = manifest.get("depth_cap")
    if isinstance(depth_cap, bool) or not isinstance(depth_cap, int) or depth_cap < 2:
        raise V2FacadeError(f"{path} requires integer depth_cap >= 2, got {depth_cap!r}")

    content = manifest.get("content_identities")
    if not isinstance(content, Mapping):
        raise V2FacadeError(f"{path} requires object content_identities")
    structure: dict[str, str | None] = {}
    for role in _REQUIRED_STRUCTURE_CONTENT:
        value = content.get(role)
        if not isinstance(value, str) or not value.strip():
            raise V2FacadeError(
                f"{path} requires content identity {role!r}; the common structure instrument "
                "cannot be inferred from endpoint rows"
            )
        structure[role] = value
    optional_structure = content.get("structure_config")
    if optional_structure is not None and (
        not isinstance(optional_structure, str) or not optional_structure.strip()
    ):
        raise V2FacadeError(
            f"{path} content identity 'structure_config' must be a non-empty string when present"
        )
    structure["structure_config"] = optional_structure

    return {
        "campaign_id": _required_manifest_text(manifest, "campaign_id", path=path),
        "phase": phase,
        "split_role": split_role,
        "code_revision": _required_manifest_text(manifest, "code_revision", path=path),
        "schedule_id": _required_manifest_text(manifest, "schedule_id", path=path),
        "coordinate_law": _required_manifest_text(manifest, "coordinate_law", path=path),
        "depth_cap": depth_cap,
        "exploratory_depth_override": True,
        "production_depth_authorized": False,
        **structure,
    }


def _required_digest(value: object, *, field: str, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise V2FacadeError(f"{context} requires {field} as a lowercase SHA-256 digest")
    return value


def _required_mapping(value: object, *, field: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V2FacadeError(f"{path} requires object {field}")
    return value


def _same_scalar(
    observed: object, expected: object, *, field: str, path: Path,
) -> None:
    if type(observed) is not type(expected) or observed != expected:  # noqa: E721 - identity type
        raise V2FacadeError(
            f"{path} {field} does not match explicit expected identity: "
            f"expected {expected!r}, observed {observed!r}"
        )


def _validate_highrisk_content_provenance(
    *, manifest: Mapping[str, Any], config: Mapping[str, Any], path: Path,
) -> tuple[Mapping[str, str], Mapping[str, Mapping[str, Any]], str]:
    """Validate the driver's real nullable provenance surface and its input signature.

    ``content_identities`` is intentionally sparse: tokenizer, coordinate-mask and constrained
    fixed-token identities are realized inside model conditioning but are not standalone
    ``--input-file`` digests.  Requiring synthetic keys for them would reject every real bundle.
    The complete config/provenance rows still prove that every role exists and explain exactly why
    a digest is or is not available.
    """
    config_rows = config.get("content")
    if not isinstance(config_rows, list) or not all(
        isinstance(row, Mapping) for row in config_rows
    ):
        raise V2FacadeError(f"{path} config.content must be a list of objects")
    config_by_role: dict[str, Mapping[str, Any]] = {}
    for row in config_rows:
        if set(row) != {"role", "label", "binding", "expected_sha256"}:
            raise V2FacadeError(
                f"{path} config.content rows must use exactly role/label/binding/expected_sha256"
            )
        role = row.get("role")
        if not isinstance(role, str) or not role or role in config_by_role:
            raise V2FacadeError(f"{path} config.content has an empty or duplicate role")
        label = row.get("label")
        binding = row.get("binding")
        expected_digest = row.get("expected_sha256")
        if not isinstance(label, str) or not label:
            raise V2FacadeError(f"{path} config.content role {role!r} has no label")
        if binding not in {"frozen", "runtime"}:
            raise V2FacadeError(
                f"{path} config.content role {role!r} has invalid binding {binding!r}"
            )
        if binding == "runtime" and expected_digest is not None:
            raise V2FacadeError(
                f"{path} runtime content role {role!r} may not declare expected_sha256"
            )
        if binding == "frozen":
            if role == "code_revision":
                if expected_digest != manifest.get("code_revision"):
                    raise V2FacadeError(
                        f"{path} frozen code_revision content disagrees with manifest"
                    )
            else:
                _required_digest(
                    expected_digest, field=f"config.content[{role}].expected_sha256",
                    context=str(path),
                )
        config_by_role[role] = row
    expected_roles = set(CONTENT_ROLE_TO_FIELD)
    if set(config_by_role) != expected_roles:
        raise V2FacadeError(
            f"{path} config.content role vocabulary differs from the V2 conditioning contract: "
            f"missing={sorted(expected_roles - set(config_by_role))} "
            f"extra={sorted(set(config_by_role) - expected_roles)}"
        )
    nonfrozen_required = sorted(
        role for role in FROZEN_DIGEST_ROLES
        if config_by_role[role].get("binding") != "frozen"
    )
    if nonfrozen_required:
        raise V2FacadeError(
            f"{path} pre-runtime content role(s) must be frozen: {nonfrozen_required}"
        )

    provenance = manifest.get("content_provenance")
    if not isinstance(provenance, list) or not all(
        isinstance(row, Mapping) for row in provenance
    ):
        raise V2FacadeError(f"{path} requires list content_provenance")
    provenance_by_role: dict[str, Mapping[str, Any]] = {}
    observed_for_signature: list[dict[str, str]] = []
    established: dict[str, str] = {}
    for row in provenance:
        if set(row) != {
            "role", "declared_label", "binding", "declared_sha256",
            "observed_path", "observed_sha256",
        }:
            raise V2FacadeError(
                f"{path} content_provenance rows must use the driver's exact six-field schema"
            )
        role = row.get("role")
        if not isinstance(role, str) or not role or role in provenance_by_role:
            raise V2FacadeError(f"{path} content_provenance has an empty or duplicate role")
        provenance_by_role[role] = row
    if set(provenance_by_role) != expected_roles:
        raise V2FacadeError(
            f"{path} content_provenance must cover exactly config.content roles"
        )

    for role in sorted(expected_roles):
        declared = config_by_role[role]
        observed = provenance_by_role[role]
        for provenance_field, config_field in (
            ("declared_label", "label"),
            ("binding", "binding"),
            ("declared_sha256", "expected_sha256"),
        ):
            if observed.get(provenance_field) != declared.get(config_field):
                raise V2FacadeError(
                    f"{path} content_provenance role {role!r} disagrees with config.content on "
                    f"{provenance_field}"
                )
        declared_digest = observed.get("declared_sha256")
        observed_digest = observed.get("observed_sha256")
        if declared_digest is not None and role != "code_revision":
            _required_digest(
                declared_digest, field=f"content_provenance[{role}].declared_sha256",
                context=str(path),
            )
        if role == "code_revision" and declared_digest != manifest.get("code_revision"):
            raise V2FacadeError(
                f"{path} code_revision content row disagrees with manifest code_revision"
            )
        if observed_digest is not None:
            digest = _required_digest(
                observed_digest, field=f"content_provenance[{role}].observed_sha256",
                context=str(path),
            )
            observed_path = observed.get("observed_path")
            if not isinstance(observed_path, str) or not observed_path:
                raise V2FacadeError(
                    f"{path} observed content role {role!r} has no observed_path"
                )
            if declared_digest is not None and role != "code_revision" \
                    and digest != declared_digest:
                raise V2FacadeError(
                    f"{path} observed content role {role!r} contradicts its frozen digest"
                )
            observed_for_signature.append({"role": role, "sha256": digest})
        established_digest = observed_digest or declared_digest
        if established_digest is not None:
            established[role] = str(established_digest)

    identities = _required_mapping(
        manifest.get("content_identities"), field="content_identities", path=path,
    )
    if dict(identities) != established:
        raise V2FacadeError(
            f"{path} content_identities does not equal the identities established by "
            "content_provenance"
        )
    encoded = json.dumps(
        sorted(observed_for_signature, key=lambda row: row["role"]),
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    input_signature = _required_digest(
        manifest.get("input_signature"), field="input_signature", context=str(path),
    )
    if hashlib.sha256(encoded).hexdigest() != input_signature:
        raise V2FacadeError(
            f"{path} input_signature does not digest the observed role/content pairs"
        )
    return established, config_by_role, input_signature


def _hotspot_method_law(
    value: object, *, field: str, path: Path, protein_id: str,
    reference_kind: str, reference_label: str,
) -> Mapping[str, Any] | None:
    """Project a per-protein hotspot calibration onto the law shared by the campaign.

    The measured threshold and the records that bind that measurement are intentionally
    protein-specific.  The scientific meaning of the measurement is not: unit, estimator source
    kind, gate/statistic, reference kind, Head domain and window domain must remain exact.
    """
    if value is None:
        return None
    block = _required_mapping(value, field=field, path=path)
    artifact = _required_mapping(block.get("artifact"), field=f"{field}.artifact", path=path)
    law_fields = ("value", "unit", "source_kind", "source_id", "source_ref")
    artifact_law_fields = (
        "schema_version",
        "gate_kind",
        "scope",
        "reference_kind",
        "reference_label",
        "window_domain",
        "allele",
        "score_scale",
        "window_k_min",
        "window_k_max",
    )
    missing = [name for name in law_fields if name not in block]
    missing.extend(
        f"artifact.{name}" for name in artifact_law_fields if name not in artifact
    )
    if missing:
        raise V2FacadeError(
            f"{path} {field} lacks method-law field(s) {missing}; a protein threshold cannot "
            "be normalized unless the statistic it estimates remains explicit"
        )
    measured = block["value"]
    if isinstance(measured, bool) or not isinstance(measured, (int, float)) \
            or not math.isfinite(float(measured)) or float(measured) < 0.0:
        raise V2FacadeError(f"{path} {field}.value must be finite and non-negative")
    expected_source_id = f"{_HIGH_RISK_HOTSPOT_SOURCE_PREFIX}:{protein_id}"
    if block["source_id"] != expected_source_id:
        raise V2FacadeError(
            f"{path} {field}.source_id must name the frozen full-trajectory Q90 population "
            f"{expected_source_id!r}"
        )
    if artifact.get("reference_kind") != reference_kind \
            or artifact.get("reference_label") != reference_label:
        raise V2FacadeError(
            f"{path} {field} reference identity disagrees with config.safety"
        )
    _required_digest(
        artifact.get("calibration_data_digest"),
        field=f"{field}.artifact.calibration_data_digest", context=str(path),
    )
    source_ref = _required_digest(
        block["source_ref"], field=f"{field}.source_ref", context=str(path),
    )
    source_payload = {
        "value": float(measured),
        "unit": block["unit"],
        "source_kind": block["source_kind"],
        "source_id": block["source_id"],
        "artifact": dict(artifact),
    }
    if canonical_digest(source_payload) != source_ref:
        raise V2FacadeError(
            f"{path} {field}.source_ref does not digest its full measured calibration block"
        )
    return {
        "unit": block["unit"],
        "source_kind": block["source_kind"],
        "source_id_prefix": _HIGH_RISK_HOTSPOT_SOURCE_PREFIX,
        "artifact_law": {name: artifact[name] for name in artifact_law_fields},
        # Protein measurements stay out of the cross-protein method digest: value, full source_id,
        # source_ref and calibration_data_digest.  Their shape/content binding was validated above.
    }


def _canonical_method_profile(
    config: Mapping[str, Any], *, path: Path, protein_id: str,
    content_by_role: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return the cross-protein algorithm identity, excluding only typed cell calibration.

    This is an explicit projection rather than a recursive "drop every digest" heuristic.  New
    config fields remain binding by default; only the named protein/cell fields below are removed.
    """
    schedule = _required_mapping(config.get("schedule"), field="config.schedule", path=path)
    points = schedule.get("points")
    if not isinstance(points, list) or not points or not all(isinstance(row, Mapping) for row in points):
        raise V2FacadeError(f"{path} config.schedule.points must be a non-empty list of objects")
    normalized_points = []
    for index, point in enumerate(points):
        if "band_key" not in point:
            raise V2FacadeError(
                f"{path} config.schedule.points[{index}] lacks band_key; the exporter cannot "
                "distinguish a protein calibration label from an omitted coordinate field"
            )
        # ``band_key`` is the lookup/method cell (frozen to step40 here); the protein-specific
        # stratum and table bytes live in schedule_band_calibration content identity instead.
        normalized_points.append(dict(point))
    stratum_key = schedule.get("stratum_key")
    if not isinstance(stratum_key, str) or not stratum_key:
        raise V2FacadeError(
            f"{path} config.schedule.stratum_key must bind this protein's calibrated band cell"
        )
    # ``stratum_key`` names the protein/constraint calibration population.  It is bound by the
    # per-protein config plus schedule_band_calibration digest and therefore is the one schedule
    # field legitimately normalized out of the cross-protein method identity.
    schedule_profile = {
        **{key: value for key, value in schedule.items() if key != "stratum_key"},
        "points": normalized_points,
        "stratum_key_binding": "protein_bound_schedule_band_calibration",
    }

    safety = _required_mapping(config.get("safety"), field="config.safety", path=path)
    if "delta_new_cumulative" not in safety:
        raise V2FacadeError(f"{path} config.safety lacks delta_new_cumulative")
    if "incremental_gate_enabled" not in safety or "delta_new_incremental" not in safety:
        raise V2FacadeError(
            f"{path} config.safety must explicitly declare incremental_gate_enabled and "
            "delta_new_incremental"
        )
    reference_kind = safety.get("cumulative_reference_kind")
    reference_label = safety.get("cumulative_reference_label")
    if reference_kind != "native_wt" or reference_label != "wt_native":
        raise V2FacadeError(
            f"{path} high-risk safety reference must be native_wt/wt_native, got "
            f"{reference_kind!r}/{reference_label!r}"
        )
    safety_profile = {
        key: value for key, value in safety.items()
        if key not in {"delta_new_cumulative", "delta_new_incremental"}
    }
    safety_profile["delta_new_cumulative_law"] = _hotspot_method_law(
        safety["delta_new_cumulative"], field="config.safety.delta_new_cumulative", path=path,
        protein_id=protein_id, reference_kind=reference_kind,
        reference_label=reference_label,
    )
    safety_profile["delta_new_incremental_law"] = _hotspot_method_law(
        safety["delta_new_incremental"], field="config.safety.delta_new_incremental", path=path,
        protein_id=protein_id, reference_kind=reference_kind,
        reference_label=reference_label,
    )

    globally_typed_roles = (
        _METHOD_CONTENT_ROLES | _CAMPAIGN_CONTENT_ROLES
        | set(_REQUIRED_STRUCTURE_CONTENT)
        | {"structure_config", "tokenizer", "code_revision"}
    )
    content_contract = {
        "role_vocabulary": sorted(content_by_role),
        "global_role_declarations": {
            role: {
                "label": content_by_role[role].get("label"),
                "binding": content_by_role[role].get("binding"),
            }
            for role in sorted(globally_typed_roles)
        },
        "protein_bound_roles": sorted(set(content_by_role) - globally_typed_roles),
    }

    return {
        "method_profile_schema": "rf-fusion-v2-cross-protein-method-2",
        **{
            key: value for key, value in config.items()
            if key not in {"content", "schedule", "safety"}
        },
        "schedule": schedule_profile,
        "safety": safety_profile,
        "content_contract": content_contract,
    }


def _load_highrisk_manifest_identity(
    root: Path, expected: HighRiskFacadeIdentity,
) -> dict[str, Any]:
    """Load and cryptographically bind one high-risk root cell to its canonical config."""
    path, manifest = _read_manifest(root)
    for field in ("campaign_id", "phase", "split_role", "schedule_id", "depth_cap", "master_seed"):
        _same_scalar(manifest.get(field), getattr(expected, field), field=field, path=path)
    if manifest.get("exploratory_depth_override") is not True:
        raise V2FacadeError(f"{path} requires exploratory_depth_override=true")
    if manifest.get("production_depth_authorized") is not False:
        raise V2FacadeError(f"{path} requires production_depth_authorized=false")

    root_index = manifest.get("root_index")
    if isinstance(root_index, bool) or not isinstance(root_index, int) \
            or root_index not in _HIGH_RISK_ROOT_INDICES:
        raise V2FacadeError(f"{path} root_index must be exactly one of 0, 1, 2, 3")
    raw_seeds = _required_mapping(
        manifest.get("root_seeds_by_protein"), field="root_seeds_by_protein", path=path,
    )
    if not raw_seeds:
        raise V2FacadeError(f"{path} root_seeds_by_protein may not be empty")
    seeds: dict[str, int] = {}
    for protein_id, seed in raw_seeds.items():
        if not isinstance(protein_id, str) or not protein_id.strip() \
                or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise V2FacadeError(
                f"{path} root_seeds_by_protein must map non-empty protein IDs to "
                "non-negative integers"
            )
        seeds[protein_id] = seed

    raw_statuses = _required_mapping(
        manifest.get("fragment_result_status_by_protein"),
        field="fragment_result_status_by_protein", path=path,
    )
    if set(raw_statuses) != set(seeds):
        raise V2FacadeError(
            f"{path} fragment_result_status_by_protein must cover exactly the proteins in "
            "root_seeds_by_protein"
        )
    statuses: dict[str, str] = {}
    for protein_id, status in raw_statuses.items():
        if status not in _HIGH_RISK_COMPLETED_STATUSES:
            raise V2FacadeError(
                f"{path} protein {protein_id!r} has result status {status!r}; a high-risk root "
                "cell must be a closed 'ok' or 'complete_negative' result, never a retryable "
                "operational failure"
            )
        statuses[str(protein_id)] = str(status)
    n_ok = manifest.get("n_ok")
    n_complete_negative = manifest.get("n_complete_negative")
    if isinstance(n_ok, bool) or not isinstance(n_ok, int) \
            or n_ok != sum(status == "ok" for status in statuses.values()):
        raise V2FacadeError(f"{path} n_ok does not match fragment result statuses")
    if isinstance(n_complete_negative, bool) or not isinstance(n_complete_negative, int) \
            or n_complete_negative != sum(
                status == "complete_negative" for status in statuses.values()
            ):
        raise V2FacadeError(
            f"{path} n_complete_negative does not match fragment result statuses"
        )

    raw_config = manifest.get("config_canonical_json")
    if not isinstance(raw_config, str) or not raw_config.strip():
        raise V2FacadeError(
            f"{path} requires non-empty config_canonical_json; config_digest alone cannot prove "
            "schedule points/K, substrate, projection policy, or the dual structure profile"
        )
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise V2FacadeError(f"{path} config_canonical_json is invalid JSON: {exc}") from exc
    if not isinstance(config, Mapping):
        raise V2FacadeError(f"{path} config_canonical_json must encode an object")
    config_digest = _required_digest(
        manifest.get("config_digest"), field="config_digest", context=str(path),
    )
    if canonical_digest(config) != config_digest:
        raise V2FacadeError(
            f"{path} config_digest does not digest config_canonical_json"
        )

    identity = _required_mapping(config.get("identity"), field="config.identity", path=path)
    for field in ("campaign_id", "phase", "split_role", "master_seed"):
        _same_scalar(identity.get(field), getattr(expected, field), field=f"config.identity.{field}",
                     path=path)
    code_revision = _required_manifest_text(manifest, "code_revision", path=path)
    _same_scalar(
        identity.get("code_revision"), code_revision,
        field="config.identity.code_revision", path=path,
    )

    schedule = _required_mapping(config.get("schedule"), field="config.schedule", path=path)
    for field in ("schedule_id", "depth_cap"):
        _same_scalar(
            schedule.get(field), getattr(expected, field), field=f"config.schedule.{field}",
            path=path,
        )
    coordinate_law = _required_manifest_text(manifest, "coordinate_law", path=path)
    _same_scalar(
        schedule.get("coordinate_law"), coordinate_law,
        field="config.schedule.coordinate_law", path=path,
    )
    active_width = manifest.get("active_population_width")
    if isinstance(active_width, bool) or not isinstance(active_width, int) or active_width < 1:
        raise V2FacadeError(f"{path} requires positive integer active_population_width")
    _same_scalar(
        schedule.get("active_population_width"), active_width,
        field="config.schedule.active_population_width", path=path,
    )
    points = schedule.get("points")
    if not isinstance(points, list) or not points or not all(isinstance(row, Mapping) for row in points):
        raise V2FacadeError(f"{path} config.schedule.points must be a non-empty list of objects")
    for field in ("substrate", "projection", "head", "safety", "caps"):
        _required_mapping(config.get(field), field=f"config.{field}", path=path)

    safety = _required_mapping(config["safety"], field="config.safety", path=path)
    policy = _required_mapping(
        safety.get("search_structure"), field="config.safety.search_structure", path=path,
    )
    expected_policy = {
        "policy_kind": DUAL_SCTM_POLICY_KIND,
        "profile_id": expected.dual_structure_profile_id,
        "ancestry_sctm_min": HIGH_RISK_ANCESTRY_SCTM_MIN,
        "strict_sctm_min": HIGH_RISK_STRICT_SCTM_MIN,
    }
    if dict(policy) != expected_policy:
        raise V2FacadeError(
            f"{path} dual structure policy does not match the explicit high-risk profile: "
            f"expected {expected_policy!r}, observed {dict(policy)!r}"
        )
    policy_digest = canonical_digest(expected_policy)

    content, content_by_role, input_signature = _validate_highrisk_content_provenance(
        manifest=manifest, config=config, path=path,
    )
    if len(seeds) != 1:
        raise V2FacadeError(
            f"{path} high-risk config is protein-bound and must declare exactly one protein; "
            f"observed {sorted(seeds)}"
        )
    protein_id = next(iter(seeds))
    structure: dict[str, str | None] = {}
    for role in _REQUIRED_STRUCTURE_CONTENT:
        structure[role] = _required_digest(
            content.get(role), field=f"content_identities.{role}", context=str(path),
        )
    optional_structure = content.get("structure_config")
    if optional_structure is not None:
        optional_structure = _required_digest(
            optional_structure, field="content_identities.structure_config", context=str(path),
        )
    structure["structure_config"] = optional_structure
    method_content = {
        role: _required_digest(
            content.get(role), field=f"content_identities.{role}", context=str(path),
        )
        for role in sorted(_METHOD_CONTENT_ROLES)
    }
    campaign_content = {
        role: _required_digest(
            content.get(role), field=f"content_identities.{role}", context=str(path),
        )
        for role in sorted(_CAMPAIGN_CONTENT_ROLES)
    }
    method_payload = {
        "config_method": _canonical_method_profile(
            config, path=path, protein_id=protein_id, content_by_role=content_by_role,
        ),
        "realized_method_content": method_content,
        "structure_content": structure,
    }
    campaign_payload = {
        "campaign_profile_schema": "rf-fusion-v2-highrisk-campaign-1",
        "campaign_id": expected.campaign_id,
        "phase": expected.phase,
        "split_role": expected.split_role,
        "schedule_id": expected.schedule_id,
        "depth_cap": expected.depth_cap,
        "master_seed": expected.master_seed,
        "code_revision": code_revision,
        "campaign_content": campaign_content,
    }
    return {
        "root": root,
        "path": path,
        "protein_id": protein_id,
        "root_index": root_index,
        "root_seeds_by_protein": seeds,
        "fragment_result_status_by_protein": statuses,
        "n_complete_negative": n_complete_negative,
        "config_digest": config_digest,
        "input_signature": input_signature,
        "method_profile_digest": canonical_digest(method_payload),
        "campaign_profile_digest": canonical_digest(campaign_payload),
        "dual_structure_policy_digest": policy_digest,
        "dual_structure_profile_id": expected.dual_structure_profile_id,
        "schedule_id": expected.schedule_id,
        "depth_cap": expected.depth_cap,
        "structure_identity": structure,
    }


def _validate_common_manifest_identity(roots: Sequence[Path]) -> None:
    identities = [(root, _load_manifest_identity(root)) for root in roots]
    baseline_root, baseline = identities[0]
    for root, identity in identities[1:]:
        for field in _COMMON_MANIFEST_FIELDS:
            if identity[field] != baseline[field]:
                raise V2FacadeError(
                    f"V2 bundle manifest identity mismatch for {field}: "
                    f"{baseline_root} has {baseline[field]!r}, {root} has {identity[field]!r}"
                )


def _require_columns(frame: pd.DataFrame, required: set[str], *, table: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise V2FacadeError(f"{table} is missing required columns {missing}")


def _require_unique_endpoint_ids(frame: pd.DataFrame, *, table: Path) -> None:
    if frame["endpoint_id"].isna().any():
        raise V2FacadeError(f"{table} contains null endpoint_id")
    if frame["endpoint_id"].astype(str).str.len().eq(0).any():
        raise V2FacadeError(f"{table} contains empty endpoint_id")
    duplicate = frame.loc[frame["endpoint_id"].duplicated(keep=False), "endpoint_id"]
    if not duplicate.empty:
        raise V2FacadeError(
            f"{table} contains duplicate endpoint_id values {sorted(set(map(str, duplicate)))[:3]}"
        )


def _mismatched_evidence(left: pd.Series, right: pd.Series) -> pd.Series:
    """Null-safe exact equality for redundant parquet evidence columns."""
    left_text = left.astype("string")
    right_text = right.astype("string")
    return left_text.isna().ne(right_text.isna()) | (
        left_text.notna() & right_text.notna() & left_text.ne(right_text)
    )


def _strict_bool(value: object, *, field: str, endpoint_id: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise V2FacadeError(
            f"endpoint {endpoint_id} has non-boolean {field}={value!r}; no truthy coercion allowed"
        )
    return bool(value)


def _load_joined_bundle(root: Path, *, highrisk: bool = False) -> pd.DataFrame:
    archive_path = root / "archive.parquet"
    endpoint_path = root / "complete_endpoints.parquet"
    structure_path = root / "structure_evaluations.parquet"
    required_paths = (
        (archive_path, endpoint_path, structure_path)
        if highrisk else (archive_path, endpoint_path)
    )
    for path in required_paths:
        if not path.is_file():
            raise V2FacadeError(f"V2 bundle is missing {path.name}: {root}")
    archive = pd.read_parquet(archive_path)
    endpoints = pd.read_parquet(endpoint_path)
    _require_columns(
        archive, _HIGH_RISK_ARCHIVE_REQUIRED if highrisk else _ARCHIVE_REQUIRED,
        table=archive_path,
    )
    _require_columns(
        endpoints, _HIGH_RISK_ENDPOINT_REQUIRED if highrisk else _ENDPOINT_REQUIRED,
        table=endpoint_path,
    )
    _require_unique_endpoint_ids(archive, table=archive_path)
    _require_unique_endpoint_ids(endpoints, table=endpoint_path)
    structure = None
    if highrisk:
        structure = pd.read_parquet(structure_path)
        _require_columns(structure, _HIGH_RISK_STRUCTURE_REQUIRED, table=structure_path)
        _require_unique_endpoint_ids(structure, table=structure_path)

    joined = archive.merge(
        endpoints,
        on="endpoint_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_archive", "_endpoint"),
        indicator=True,
    )
    unmatched = joined[joined["_merge"] != "both"]
    if not unmatched.empty:
        examples = sorted(map(str, unmatched["endpoint_id"]))[:3]
        side = sorted(set(map(str, unmatched["_merge"])))
        raise V2FacadeError(
            f"bundle {root} has endpoint ids with no complete endpoint/archive peer: "
            f"merge_status={side} examples={examples}"
        )
    joined = joined.drop(columns=["_merge"])

    for field in (
        "endpoint_content_digest",
        "protein_id",
        "sequence_equivalence_key",
        "feasibility_level",
        *(("root_id", "family_id") if highrisk else ()),
    ):
        left = joined[f"{field}_archive"]
        right = joined[f"{field}_endpoint"]
        mismatch = left.isna() | right.isna() | _mismatched_evidence(left, right)
        if mismatch.any():
            ids = sorted(map(str, joined.loc[mismatch, "endpoint_id"]))[:3]
            raise V2FacadeError(
                f"bundle {root} has {field} identity mismatch for endpoint(s) {ids}"
            )

    if highrisk:
        assert structure is not None
        structure = structure.rename(columns={
            column: f"{column}_evaluation"
            for column in structure.columns if column != "endpoint_id"
        })
        joined = joined.merge(
            structure, on="endpoint_id", how="outer", validate="one_to_one", indicator=True,
        )
        unmatched = joined[joined["_merge"] != "both"]
        if not unmatched.empty:
            examples = sorted(map(str, unmatched["endpoint_id"]))[:3]
            raise V2FacadeError(
                f"bundle {root} has endpoint ids without one structure_evaluations peer: "
                f"{examples}"
            )
        joined = joined.drop(columns=["_merge"])
        pairings = {
            "protein_id": ("protein_id_endpoint", "protein_id_evaluation"),
            "depth": ("depth", "depth_evaluation"),
            "sequence_md5": ("sequence_md5", "sequence_md5_evaluation"),
            "feasibility_level": (
                "feasibility_level_endpoint", "feasibility_level_evaluation",
            ),
        }
        for field, (left_name, right_name) in pairings.items():
            mismatch = _mismatched_evidence(joined[left_name], joined[right_name])
            if mismatch.any():
                ids = sorted(map(str, joined.loc[mismatch, "endpoint_id"]))[:3]
                raise V2FacadeError(
                    f"bundle {root} has endpoint/structure-evaluation {field} mismatch for "
                    f"endpoint(s) {ids}"
                )
        for field in sorted(_DUAL_EVIDENCE_COLUMNS - {"root_id", "family_id"}):
            archive_name = f"{field}_archive"
            endpoint_name = f"{field}_endpoint"
            evaluation_name = f"{field}_evaluation"
            mismatch = (
                _mismatched_evidence(joined[archive_name], joined[endpoint_name])
                | _mismatched_evidence(joined[endpoint_name], joined[evaluation_name])
            )
            if mismatch.any():
                ids = sorted(map(str, joined.loc[mismatch, "endpoint_id"]))[:3]
                raise V2FacadeError(
                    f"bundle {root} has cross-table dual evidence mismatch for {field} on "
                    f"endpoint(s) {ids}"
                )

    joined["source_bundle"] = str(root)
    return joined


def _validate_endpoint_content(row: pd.Series) -> None:
    endpoint_id = str(row["endpoint_id"])
    sequence = row["sequence"]
    try:
        expected_length = int(row["sequence_length"])
        validate_complete_aa20(sequence, expected_length)
    except (HeadInputError, TypeError, ValueError) as exc:
        raise V2FacadeError(
            f"endpoint {endpoint_id} is not complete uppercase AA20: {exc}"
        ) from exc
    observed_md5 = sequence_md5(sequence)
    if str(row["sequence_md5"]) != observed_md5:
        raise V2FacadeError(
            f"endpoint {endpoint_id} sequence_md5 does not digest its AA20 sequence"
        )
    if str(row["sequence_equivalence_key_endpoint"]) != observed_md5:
        raise V2FacadeError(
            f"endpoint {endpoint_id} sequence_equivalence_key does not equal sequence_md5"
        )
    try:
        risk = float(row["head_global_risk"])
    except (TypeError, ValueError) as exc:
        raise V2FacadeError(f"endpoint {endpoint_id} has invalid Head risk") from exc
    if not math.isfinite(risk):
        raise V2FacadeError(f"endpoint {endpoint_id} has non-finite Head risk {risk!r}")
    try:
        depth_value = float(row["depth"])
    except (TypeError, ValueError) as exc:
        raise V2FacadeError(f"endpoint {endpoint_id} has invalid depth") from exc
    if not math.isfinite(depth_value) or not depth_value.is_integer() or depth_value < 0:
        raise V2FacadeError(f"endpoint {endpoint_id} has invalid depth {row['depth']!r}")


def _validate_endpoint_row(row: pd.Series) -> None:
    endpoint_id = str(row["endpoint_id"])
    definitive = (
        str(row["feasibility_level_archive"]) == "definitive"
        and str(row["feasibility_level_endpoint"]) == "definitive"
    )
    evaluated = _strict_bool(
        row["structure_evaluated"], field="structure_evaluated", endpoint_id=endpoint_id,
    )
    feasible = _strict_bool(
        row["structure_feasible"], field="structure_feasible", endpoint_id=endpoint_id,
    )
    if not (definitive and evaluated and feasible):
        raise V2FacadeError(
            f"endpoint {endpoint_id} is not definitive-feasible: "
            f"archive_level={row['feasibility_level_archive']!r} "
            f"endpoint_level={row['feasibility_level_endpoint']!r} "
            f"evaluated={evaluated} feasible={feasible}"
        )
    _validate_endpoint_content(row)


def _candidate_rows(joined: pd.DataFrame) -> pd.DataFrame:
    eligible_mask = (
        joined["feasibility_level_archive"].eq("definitive")
        & joined["feasibility_level_endpoint"].eq("definitive")
        & joined["structure_evaluated"].eq(True)  # noqa: E712 - exact evidence value
        & joined["structure_feasible"].eq(True)  # noqa: E712 - exact evidence value
    )
    candidates = joined.loc[eligible_mask].copy()
    for _, row in candidates.iterrows():
        _validate_endpoint_row(row)
    return candidates


def _json_object(value: object, *, field: str, endpoint_id: str) -> Mapping[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise V2FacadeError(f"endpoint {endpoint_id} requires non-empty {field}")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise V2FacadeError(f"endpoint {endpoint_id} has invalid {field}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise V2FacadeError(f"endpoint {endpoint_id} {field} must encode an object")
    return payload


def _same_finite_measurement(
    left: object, right: object, *, endpoint_id: str, fields: str,
) -> None:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError) as exc:
        raise V2FacadeError(f"endpoint {endpoint_id} has invalid {fields}") from exc
    if not math.isfinite(left_value) or not math.isfinite(right_value) \
            or not math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-12):
        raise V2FacadeError(f"endpoint {endpoint_id} has inconsistent finite {fields}")


def _validate_highrisk_endpoint_evidence(
    row: pd.Series, *, manifest_identity: Mapping[str, Any],
) -> bool:
    """Validate one dual-gate endpoint and return whether it is strict-facade eligible."""
    endpoint_id = str(row["endpoint_id"])
    _validate_endpoint_content(row)
    for field in ("root_id_endpoint", "family_id_endpoint"):
        value = row[field]
        if not isinstance(value, str) or not value:
            raise V2FacadeError(f"endpoint {endpoint_id} has empty {field.removesuffix('_endpoint')}")
    protein_id = str(row["protein_id_endpoint"])
    root_index = int(manifest_identity["root_index"])
    expected_root_id = f"{protein_id}:v2:d0:r{root_index}"
    expected_family_id = f"fam{root_index}"
    if row["root_id_endpoint"] != expected_root_id \
            or row["family_id_endpoint"] != expected_family_id:
        raise V2FacadeError(
            f"endpoint {endpoint_id} lineage does not bind manifest root_index={root_index}: "
            f"expected root_id/family_id {expected_root_id!r}/{expected_family_id!r}, observed "
            f"{row['root_id_endpoint']!r}/{row['family_id_endpoint']!r}"
        )

    profile_id = row["dual_structure_profile_id_endpoint"]
    if profile_id != manifest_identity["dual_structure_profile_id"]:
        raise V2FacadeError(
            f"endpoint {endpoint_id} dual structure profile does not match its signed config"
        )
    policy_digest = _required_digest(
        row["dual_structure_policy_digest_endpoint"], field="dual_structure_policy_digest",
        context=f"endpoint {endpoint_id}",
    )
    if policy_digest != manifest_identity["dual_structure_policy_digest"]:
        raise V2FacadeError(
            f"endpoint {endpoint_id} dual structure policy digest does not match its signed config"
        )
    raw_digest = _required_digest(
        row["dual_structure_raw_outcome_digest_endpoint"],
        field="dual_structure_raw_outcome_digest", context=f"endpoint {endpoint_id}",
    )
    verdict_digest = _required_digest(
        row["dual_structure_verdict_digest_endpoint"], field="dual_structure_verdict_digest",
        context=f"endpoint {endpoint_id}",
    )

    common = _strict_bool(
        row["dual_structure_common_feasible_endpoint"], field="dual_structure_common_feasible",
        endpoint_id=endpoint_id,
    )
    ancestry = _strict_bool(
        row["dual_structure_ancestry_passed_endpoint"], field="dual_structure_ancestry_passed",
        endpoint_id=endpoint_id,
    )
    strict = _strict_bool(
        row["dual_structure_strict_passed_endpoint"], field="dual_structure_strict_passed",
        endpoint_id=endpoint_id,
    )
    evaluated = _strict_bool(
        row["evaluated_evaluation"], field="structure_evaluations.evaluated",
        endpoint_id=endpoint_id,
    )
    common_outcome_feasible = _strict_bool(
        row["feasible_evaluation"], field="structure_evaluations.feasible",
        endpoint_id=endpoint_id,
    )
    try:
        ancestry_min = float(row["dual_structure_ancestry_sctm_min_endpoint"])
        strict_min = float(row["dual_structure_strict_sctm_min_endpoint"])
        sctm = float(row["dual_structure_sctm_endpoint"])
    except (TypeError, ValueError) as exc:
        raise V2FacadeError(
            f"endpoint {endpoint_id} has invalid dual structure threshold/scTM evidence"
        ) from exc
    if ancestry_min != HIGH_RISK_ANCESTRY_SCTM_MIN \
            or strict_min != HIGH_RISK_STRICT_SCTM_MIN:
        raise V2FacadeError(
            f"endpoint {endpoint_id} dual structure thresholds do not match its signed profile"
        )
    finite_sctm = math.isfinite(sctm)

    if strict and not finite_sctm:
        raise V2FacadeError(f"endpoint {endpoint_id} strict pass requires finite scTM")
    if strict and sctm < HIGH_RISK_STRICT_SCTM_MIN:
        raise V2FacadeError(
            f"endpoint {endpoint_id} strict pass carries scTM {sctm:g} below "
            f"{HIGH_RISK_STRICT_SCTM_MIN:g}"
        )
    if strict and not common:
        raise V2FacadeError(
            f"endpoint {endpoint_id} strict pass requires common structure feasible evidence"
        )
    if strict and not ancestry:
        raise V2FacadeError(
            f"endpoint {endpoint_id} strict pass must also carry ancestry pass"
        )
    expected_ancestry = bool(
        evaluated and common and finite_sctm and sctm >= HIGH_RISK_ANCESTRY_SCTM_MIN
    )
    expected_strict = bool(
        evaluated and common and finite_sctm and sctm >= HIGH_RISK_STRICT_SCTM_MIN
    )
    if ancestry != expected_ancestry or strict != expected_strict:
        raise V2FacadeError(
            f"endpoint {endpoint_id} dual structure verdict disagrees with finite scTM thresholds"
        )
    if common_outcome_feasible != common:
        raise V2FacadeError(
            f"endpoint {endpoint_id} common feasible evidence disagrees with raw structure "
            "evaluation"
        )
    if row["structure_backend_digest_evaluation"] != \
            manifest_identity["structure_identity"]["structure_backend"]:
        raise V2FacadeError(
            f"endpoint {endpoint_id} structure evaluation backend disagrees with run manifest"
        )
    if row["v0_structure_gate_config_digest_evaluation"] != \
            manifest_identity["structure_identity"]["v0_structure_gate_config"]:
        raise V2FacadeError(
            f"endpoint {endpoint_id} structure gate config disagrees with run manifest"
        )

    metrics = _json_object(
        row["metrics_json_evaluation"], field="structure_evaluations.metrics_json",
        endpoint_id=endpoint_id,
    )
    if strict:
        if "scTM" not in metrics:
            raise V2FacadeError(f"endpoint {endpoint_id} strict pass has no measured scTM")
        _same_finite_measurement(
            metrics["scTM"], sctm, endpoint_id=endpoint_id,
            fields="structure_evaluations.metrics_json.scTM and dual_structure_sctm",
        )

    if "dual_structure_verdict_json_endpoint" in row.index:
        raw_json = row["dual_structure_verdict_json_endpoint"]
        missing = raw_json is None or (
            isinstance(raw_json, (float, np.floating)) and math.isnan(float(raw_json))
        )
        if not missing:
            verdict = _json_object(
                raw_json, field="dual_structure_verdict_json", endpoint_id=endpoint_id,
            )
            comparisons = {
                "profile_id": profile_id,
                "ancestry_sctm_min": ancestry_min,
                "strict_sctm_min": strict_min,
                "evaluated": evaluated,
                "raw_outcome_digest": raw_digest,
                "common_structure_feasible": common,
                "ancestry_structure_passed": ancestry,
                "strict_structure_passed": strict,
            }
            for field, expected_value in comparisons.items():
                if verdict.get(field) != expected_value:
                    raise V2FacadeError(
                        f"endpoint {endpoint_id} verdict JSON disagrees on {field}"
                    )
            if strict:
                _same_finite_measurement(
                    verdict.get("sctm"), sctm, endpoint_id=endpoint_id,
                    fields="verdict JSON scTM and explicit scTM",
                )
            if canonical_digest(verdict) != verdict_digest:
                raise V2FacadeError(
                    f"endpoint {endpoint_id} dual structure verdict digest does not digest "
                    "dual_structure_verdict_json"
                )

    # Every measured endpoint, including a scientific negative, must retain the immune/anchor
    # admission evidence that explains why it did or did not reach final feasibility.
    _required_digest(
        row["admission_evidence_digest_endpoint"], field="admission evidence digest",
        context=f"endpoint {endpoint_id}",
    )
    definitive = (
        str(row["feasibility_level_archive"]) == "definitive"
        and str(row["feasibility_level_endpoint"]) == "definitive"
    )
    endpoint_evaluated = _strict_bool(
        row["structure_evaluated"], field="complete_endpoints.structure_evaluated",
        endpoint_id=endpoint_id,
    )
    endpoint_feasible = _strict_bool(
        row["structure_feasible"], field="complete_endpoints.structure_feasible",
        endpoint_id=endpoint_id,
    )
    if not definitive:
        # A raw strict structure pass can still fail the independent immune/anchor admission.  The
        # endpoint then correctly remains UNVALIDATED with no bound StructureOutcome, while the
        # raw structure table retains its finite strict-pass evidence.  That is a final negative,
        # not malformed evidence and not a strict facade candidate.
        if strict and not (
            str(row["feasibility_level_archive"]) == "unvalidated"
            and str(row["feasibility_level_endpoint"]) == "unvalidated"
            and not endpoint_evaluated and not endpoint_feasible
        ):
            raise V2FacadeError(
                f"endpoint {endpoint_id} strict structure pass has inconsistent non-final "
                "endpoint state"
            )
        return False
    if not strict or not evaluated or not common_outcome_feasible \
            or not endpoint_evaluated or not endpoint_feasible:
        raise V2FacadeError(
            f"endpoint {endpoint_id} strict pass is not definitive-feasible"
        )
    _required_digest(
        row["ancestry_authorization_digest_endpoint"], field="ancestry authorization digest",
        context=f"endpoint {endpoint_id}",
    )
    return True


def _validate_highrisk_head_domains(joined: pd.DataFrame) -> None:
    evaluator = joined["head_evaluator_digest"]
    grid = joined["head_window_grid_digest"]
    if evaluator.isna().any() or evaluator.astype(str).str.len().eq(0).any():
        raise V2FacadeError("high-risk endpoints carry null or empty Head evaluator identity")
    if grid.isna().any() or grid.astype(str).str.len().eq(0).any():
        raise V2FacadeError("high-risk endpoints carry null or empty Head window-grid identity")
    if evaluator.astype(str).nunique() != 1:
        raise V2FacadeError("high-risk cohort does not share one frozen Head evaluator")
    domains = pd.DataFrame({
        "protein_id": joined["protein_id_endpoint"].astype(str),
        "grid": grid.astype(str),
    }).groupby("protein_id", sort=True)["grid"].nunique()
    bad = domains[domains != 1]
    if not bad.empty:
        raise V2FacadeError(
            "Head window-grid identity must be constant within each protein; mismatched "
            f"protein(s)={list(bad.index)}"
        )


def _materialize_legacy_archive_facade(
    bundles: Sequence[str | Path],
    *,
    mode: str,
    k: int | None = None,
    require_k: bool = False,
) -> pd.DataFrame:
    """Return a deterministic Phase-C facade selected from one or more disjoint V2 bundles.

    ``top-k`` first collapses sequence-equivalent logical endpoints to the stable representative
    under ``(head_global_risk, endpoint_id)``.  Raw multiplicity remains intact in the source V2
    bundle, but cannot buy duplicate round-0 ancestry mass in the optional v0 suffix.
    """
    if mode not in {"elite", "top-k"}:
        raise V2FacadeError("mode must be 'elite' or 'top-k'")
    if mode == "top-k":
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise V2FacadeError("top-k mode requires integer k >= 1")
    elif k is not None or require_k:
        raise V2FacadeError("k/require_k are valid only in top-k mode")
    if not bundles:
        raise V2FacadeError("at least one V2 bundle is required")

    roots = [_bundle_root(path) for path in bundles]
    if len(set(roots)) != len(roots):
        raise V2FacadeError("the same V2 bundle was supplied more than once")
    # Establish that these are cells of ONE frozen experiment before opening their endpoint
    # tables.  Once frames are concatenated, source-bundle provenance is too late to prevent a
    # mixed campaign from looking like one cohort.
    _validate_common_manifest_identity(roots)
    joined_by_root = [(root, _load_joined_bundle(root)) for root in roots]

    protein_owner: dict[str, Path] = {}
    for root, frame in joined_by_root:
        proteins = set(map(str, frame["protein_id_archive"].dropna().unique()))
        for protein_id in proteins:
            previous = protein_owner.get(protein_id)
            if previous is not None:
                raise V2FacadeError(
                    f"protein {protein_id!r} appears in more than one bundle: {previous}, {root}"
                )
            protein_owner[protein_id] = root
    joined = pd.concat([frame for _, frame in joined_by_root], ignore_index=True)
    if joined.empty or not protein_owner:
        raise V2FacadeError("the named V2 bundles contain no archived endpoints")
    duplicated_ids = joined.loc[
        joined["endpoint_id"].duplicated(keep=False), "endpoint_id"
    ]
    if not duplicated_ids.empty:
        raise V2FacadeError(
            "endpoint_id is duplicated across V2 bundles: "
            f"{sorted(set(map(str, duplicated_ids)))[:3]}"
        )

    candidates = _candidate_rows(joined)

    selected: list[pd.DataFrame] = []
    for protein_id in sorted(protein_owner):
        all_rows = joined[joined["protein_id_archive"].astype(str) == protein_id]
        pool = candidates[candidates["protein_id_archive"].astype(str) == protein_id].copy()
        if mode == "elite":
            elite_mask = [
                _strict_bool(
                    row["is_elite"], field="is_elite", endpoint_id=str(row["endpoint_id"]),
                )
                for _, row in all_rows.iterrows()
            ]
            elite_rows = all_rows.loc[elite_mask]
            if len(elite_rows) != 1:
                raise V2FacadeError(
                    f"protein {protein_id!r} requires exactly one current archive elite; "
                    f"found {len(elite_rows)}"
                )
            elite_row = elite_rows.iloc[0]
            try:
                elite_rank = float(elite_row["elite_rank"])
            except (TypeError, ValueError) as exc:
                raise V2FacadeError(
                    f"protein {protein_id!r} current archive elite has invalid elite_rank"
                ) from exc
            if not math.isfinite(elite_rank) or not elite_rank.is_integer() or elite_rank != 0:
                raise V2FacadeError(
                    f"protein {protein_id!r} current archive elite must carry elite_rank=0"
                )
            _validate_endpoint_row(elite_row)
            pick = elite_rows.copy()
        else:
            if pool.empty:
                raise V2FacadeError(
                    f"protein {protein_id!r} has no definitive-feasible archive endpoint"
                )
            pool = pool.sort_values(
                ["head_global_risk", "endpoint_id"], kind="mergesort",
            )
            pool = pool.drop_duplicates("sequence_equivalence_key_endpoint", keep="first")
            if require_k and len(pool) < int(k):
                raise V2FacadeError(
                    f"top-k requires {k} distinct definitive-feasible endpoints for protein "
                    f"{protein_id!r}, but it has {len(pool)}"
                )
            pick = pool.head(int(k))
        pick = pick.sort_values(["head_global_risk", "endpoint_id"], kind="mergesort").copy()
        pick["design_idx"] = range(len(pick))
        selected.append(pick)

    chosen = pd.concat(selected, ignore_index=True)
    if chosen.empty:
        raise V2FacadeError("selection produced zero facade rows")
    head_domains = chosen[["head_evaluator_digest", "head_window_grid_digest"]].drop_duplicates()
    if chosen[["head_evaluator_digest", "head_window_grid_digest"]].isna().any().any():
        raise V2FacadeError("selected V2 endpoints carry null Head identity")
    if any(
        not str(value)
        for value in chosen["head_evaluator_digest"].tolist()
        + chosen["head_window_grid_digest"].tolist()
    ):
        raise V2FacadeError("selected V2 endpoints carry empty Head identity")
    if len(head_domains) != 1:
        raise V2FacadeError(
            "selected V2 endpoints do not share one frozen Head evaluator/window-grid identity"
        )
    out = pd.DataFrame({
        "protein_id": chosen["protein_id_endpoint"].astype(str),
        "design_idx": chosen["design_idx"].astype(int),
        "sequence": chosen["sequence"].astype(str),
        "entry_source_id": chosen["endpoint_id"].astype(str),
        "endpoint_id": chosen["endpoint_id"].astype(str),
        "endpoint_content_digest": chosen["endpoint_content_digest_endpoint"].astype(str),
        "depth": chosen["depth"].astype(int),
        "head_global_risk": chosen["head_global_risk"].astype(float),
        "head_evaluator_digest": chosen["head_evaluator_digest"].astype(str),
        "head_window_grid_digest": chosen["head_window_grid_digest"].astype(str),
        "sequence_md5": chosen["sequence_md5"].astype(str),
    })
    out = out.sort_values(["protein_id", "design_idx"], kind="mergesort").reset_index(drop=True)
    if out[["protein_id", "design_idx"]].duplicated().any():
        raise V2FacadeError("selection produced duplicate (protein_id, design_idx) facade keys")
    if out["entry_source_id"].duplicated().any():
        raise V2FacadeError("selection produced duplicate endpoint lineage keys")
    return out.loc[:, _OUTPUT_COLUMNS]


def _materialize_highrisk_archive_facade(
    bundles: Sequence[str | Path],
    *,
    mode: str,
    k: int | None,
    require_k: bool,
    expected_identity: HighRiskFacadeIdentity,
) -> pd.DataFrame:
    if mode not in {"elite", "top-k"}:
        raise V2FacadeError("mode must be 'elite' or 'top-k'")
    if mode == "top-k":
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise V2FacadeError("top-k mode requires integer k >= 1")
    elif k is not None or require_k:
        raise V2FacadeError("k/require_k are valid only in top-k mode")
    if not bundles:
        raise V2FacadeError("at least one V2 bundle is required")

    roots = [_bundle_root(path) for path in bundles]
    if len(set(roots)) != len(roots):
        raise V2FacadeError("the same V2 bundle was supplied more than once")
    identities = [
        _load_highrisk_manifest_identity(root, expected_identity) for root in roots
    ]
    method_domains = {identity["method_profile_digest"] for identity in identities}
    if len(method_domains) != 1:
        raise V2FacadeError(
            "high-risk bundles do not share one cross-protein canonical method profile"
        )
    method_profile_digest = next(iter(method_domains))
    campaign_domains = {identity["campaign_profile_digest"] for identity in identities}
    if len(campaign_domains) != 1:
        raise V2FacadeError(
            "high-risk bundles do not share one campaign-wide cohort/reference identity"
        )
    campaign_profile_digest = next(iter(campaign_domains))
    baseline = identities[0]
    for identity in identities[1:]:
        for field in (*_REQUIRED_STRUCTURE_CONTENT, "structure_config"):
            left = baseline["structure_identity"][field]
            right = identity["structure_identity"][field]
            if left != right:
                raise V2FacadeError(
                    f"high-risk bundle structure identity mismatch for {field}: "
                    f"{baseline['root']} has {left!r}, {identity['root']} has {right!r}"
                )

    joined_by_root: list[tuple[dict[str, Any], pd.DataFrame]] = []
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    requested_proteins: set[str] = set()
    for identity in identities:
        frame = _load_joined_bundle(identity["root"], highrisk=True)
        frame_proteins = set(map(str, frame["protein_id_archive"].dropna().unique()))
        declared_proteins = set(identity["root_seeds_by_protein"])
        undeclared = sorted(frame_proteins - declared_proteins)
        if undeclared:
            raise V2FacadeError(
                f"bundle {identity['root']} archives protein(s) absent from "
                f"root_seeds_by_protein: {undeclared}"
            )
        missing_evidence = sorted(declared_proteins - frame_proteins)
        if missing_evidence:
            raise V2FacadeError(
                f"bundle {identity['root']} declares root {identity['root_index']} but has no "
                f"archived endpoint evidence for protein(s) {missing_evidence}"
            )
        requested_proteins.update(declared_proteins)
        for protein_id in sorted(declared_proteins):
            key = (protein_id, identity["root_index"])
            previous = cells.get(key)
            if previous is not None:
                raise V2FacadeError(
                    f"protein {protein_id!r} root {identity['root_index']} appears in more than "
                    f"one bundle: {previous['root']}, {identity['root']} (duplicate root cell)"
                )
            cells[key] = identity
        frame["root_index"] = int(identity["root_index"])
        frame["root_seed"] = frame["protein_id_endpoint"].astype(str).map(
            identity["root_seeds_by_protein"]
        )
        frame["config_digest"] = identity["config_digest"]
        frame["method_profile_digest"] = identity["method_profile_digest"]
        frame["schedule_id"] = identity["schedule_id"]
        frame["manifest_depth_cap"] = identity["depth_cap"]
        strict_flags: list[bool] = []
        for _, row in frame.iterrows():
            strict_flags.append(_validate_highrisk_endpoint_evidence(
                row, manifest_identity=identity,
            ))
        frame["_strict_candidate"] = strict_flags
        for protein_id in sorted(declared_proteins):
            realized_strict = bool(frame.loc[
                frame["protein_id_endpoint"].astype(str) == protein_id,
                "_strict_candidate",
            ].any())
            declared_status = identity["fragment_result_status_by_protein"][protein_id]
            expected_status = "ok" if realized_strict else "complete_negative"
            if declared_status != expected_status:
                raise V2FacadeError(
                    f"bundle {identity['root']} protein {protein_id!r} result status "
                    f"{declared_status!r} disagrees with its realized strict pool; expected "
                    f"{expected_status!r}"
                )
        joined_by_root.append((identity, frame))

    if not requested_proteins:
        raise V2FacadeError("the named high-risk bundles declare no proteins")
    for protein_id in sorted(requested_proteins):
        observed_roots = {root_index for (cell_protein, root_index) in cells
                          if cell_protein == protein_id}
        if observed_roots != _HIGH_RISK_ROOT_INDICES:
            raise V2FacadeError(
                f"protein {protein_id!r} root coverage must be exactly [0, 1, 2, 3]; "
                f"observed {sorted(observed_roots)}"
            )
        protein_cells = [cells[(protein_id, root_index)] for root_index in range(4)]
        config_domains = {identity["config_digest"] for identity in protein_cells}
        if len(config_domains) != 1:
            raise V2FacadeError(
                f"same protein {protein_id!r} does not carry one complete config identity "
                "across roots 0..3"
            )
        input_domains = {identity["input_signature"] for identity in protein_cells}
        if len(input_domains) != 1:
            raise V2FacadeError(
                f"same protein {protein_id!r} does not carry one complete observed input "
                "identity across roots 0..3"
            )
        root_seeds = [
            identity["root_seeds_by_protein"][protein_id] for identity in protein_cells
        ]
        if len(set(root_seeds)) != 4:
            raise V2FacadeError(
                f"protein {protein_id!r} root seed must be unique across roots 0..3; "
                f"observed {root_seeds}"
            )

    joined = pd.concat([frame for _, frame in joined_by_root], ignore_index=True)
    if joined.empty:
        raise V2FacadeError("the named high-risk bundles contain no archived endpoints")
    duplicated_ids = joined.loc[
        joined["endpoint_id"].duplicated(keep=False), "endpoint_id"
    ]
    if not duplicated_ids.empty:
        raise V2FacadeError(
            "endpoint_id is duplicated across high-risk root bundles: "
            f"{sorted(set(map(str, duplicated_ids)))[:3]}"
        )
    _validate_highrisk_head_domains(joined)
    candidates = joined.loc[joined["_strict_candidate"]].copy()

    selected: list[pd.DataFrame] = []
    exclusions: list[dict[str, str]] = []
    candidate_counts: dict[str, int] = {}
    for protein_id in sorted(requested_proteins):
        pool = candidates[
            candidates["protein_id_archive"].astype(str) == protein_id
        ].copy()
        pool = pool.sort_values(
            ["head_global_risk", "endpoint_id"], kind="mergesort",
        ).drop_duplicates("sequence_md5", keep="first")
        candidate_counts[protein_id] = len(pool)
        if pool.empty:
            exclusions.append({
                "protein_id": protein_id,
                "reason": "zero_strict_endpoints",
            })
            continue
        if mode == "top-k":
            if require_k and len(pool) < int(k):
                raise V2FacadeError(
                    f"top-k requires {k} distinct strict endpoints for protein "
                    f"{protein_id!r}, but it has {len(pool)}"
                )
            pick = pool.head(int(k)).copy()
        else:
            # Archive elite flags are root-local.  The experiment's elite is the global strict
            # minimum after all four roots are pooled and sequence-equivalent siblings collapse.
            pick = pool.head(1).copy()
        pick["design_idx"] = range(len(pick))
        selected.append(pick)

    if selected:
        chosen = pd.concat(selected, ignore_index=True)
        archive_elite = [
            _strict_bool(row["is_elite"], field="is_elite", endpoint_id=str(row["endpoint_id"]))
            for _, row in chosen.iterrows()
        ]
        out = pd.DataFrame({
            "protein_id": chosen["protein_id_endpoint"].astype(str),
            "design_idx": chosen["design_idx"].astype(int),
            "sequence": chosen["sequence"].astype(str),
            "entry_source_id": chosen["endpoint_id"].astype(str),
            "endpoint_id": chosen["endpoint_id"].astype(str),
            "endpoint_content_digest": chosen["endpoint_content_digest_endpoint"].astype(str),
            "depth": chosen["depth"].astype(int),
            "head_global_risk": chosen["head_global_risk"].astype(float),
            "head_evaluator_digest": chosen["head_evaluator_digest"].astype(str),
            "head_window_grid_digest": chosen["head_window_grid_digest"].astype(str),
            "sequence_md5": chosen["sequence_md5"].astype(str),
            "root_index": chosen["root_index"].astype(int),
            "root_seed": chosen["root_seed"].astype(int),
            "root_id": chosen["root_id_endpoint"].astype(str),
            "family_id": chosen["family_id_endpoint"].astype(str),
            "source_bundle": chosen["source_bundle"].astype(str),
            "config_digest": chosen["config_digest"].astype(str),
            "method_profile_digest": chosen["method_profile_digest"].astype(str),
            "schedule_id": chosen["schedule_id"].astype(str),
            "depth_cap": chosen["manifest_depth_cap"].astype(int),
            "dual_structure_profile_id": chosen[
                "dual_structure_profile_id_endpoint"
            ].astype(str),
            "dual_structure_policy_digest": chosen[
                "dual_structure_policy_digest_endpoint"
            ].astype(str),
            "dual_structure_ancestry_sctm_min": chosen[
                "dual_structure_ancestry_sctm_min_endpoint"
            ].astype(float),
            "dual_structure_strict_sctm_min": chosen[
                "dual_structure_strict_sctm_min_endpoint"
            ].astype(float),
            "dual_structure_verdict_digest": chosen[
                "dual_structure_verdict_digest_endpoint"
            ].astype(str),
            "dual_structure_raw_outcome_digest": chosen[
                "dual_structure_raw_outcome_digest_endpoint"
            ].astype(str),
            "dual_structure_sctm": chosen[
                "dual_structure_sctm_endpoint"
            ].astype(float),
            "ancestry_authorization_digest": chosen[
                "ancestry_authorization_digest_endpoint"
            ].astype(str),
            "admission_evidence_digest": chosen[
                "admission_evidence_digest_endpoint"
            ].astype(str),
            "archive_is_elite": archive_elite,
        }).loc[:, _HIGH_RISK_OUTPUT_COLUMNS]
        out = out.sort_values(
            ["protein_id", "design_idx"], kind="mergesort",
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=_HIGH_RISK_OUTPUT_COLUMNS)

    if out[["protein_id", "design_idx"]].duplicated().any():
        raise V2FacadeError("selection produced duplicate (protein_id, design_idx) facade keys")
    if out["entry_source_id"].duplicated().any():
        raise V2FacadeError("selection produced duplicate endpoint lineage keys")
    out.attrs["selection_provenance"] = {
        "schema_version": "rf-fusion-v2-highrisk-facade-selection-1",
        "contract": "high-risk-r4",
        "expected_identity": asdict(expected_identity),
        "strict_sctm_min": HIGH_RISK_STRICT_SCTM_MIN,
        "root_indices": sorted(_HIGH_RISK_ROOT_INDICES),
        "input_bundles": [str(root) for root in roots],
        "method_profile_digest": method_profile_digest,
        "campaign_profile_digest": campaign_profile_digest,
        "candidate_counts_by_protein": candidate_counts,
        "included_proteins": sorted(set(map(str, out["protein_id"]))),
        "excluded_proteins": exclusions,
    }
    return out


def materialize_archive_facade(
    bundles: Sequence[str | Path],
    *,
    mode: str,
    k: int | None = None,
    require_k: bool = False,
    contract: str = "legacy",
    expected_identity: HighRiskFacadeIdentity | None = None,
) -> pd.DataFrame:
    """Materialize either the legacy single-root facade or one explicit high-risk R4 profile.

    The high-risk contract pools roots 0..3 within each protein, derives elite/top-k globally,
    and records zero-strict proteins as exclusions instead of laundering ancestry-only endpoints
    into a terminal facade.
    """
    if contract == "legacy":
        if expected_identity is not None:
            raise V2FacadeError("expected_identity is valid only for contract='high-risk-r4'")
        return _materialize_legacy_archive_facade(
            bundles, mode=mode, k=k, require_k=require_k,
        )
    if contract != "high-risk-r4":
        raise V2FacadeError("contract must be 'legacy' or 'high-risk-r4'")
    if not isinstance(expected_identity, HighRiskFacadeIdentity):
        raise V2FacadeError(
            "high-risk-r4 requires an explicit HighRiskFacadeIdentity"
        )
    return _materialize_highrisk_archive_facade(
        bundles, mode=mode, k=k, require_k=require_k,
        expected_identity=expected_identity,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", action="append", required=True,
        help="V2 bundle directory (repeat; high-risk-r4 expects every protein's roots 0..3)",
    )
    parser.add_argument(
        "--contract", choices=("legacy", "high-risk-r4"), default="legacy",
        help="selection/evidence contract (default: legacy exploratory-uricase exporter)",
    )
    parser.add_argument("--expected-campaign-id")
    parser.add_argument("--expected-phase")
    parser.add_argument("--expected-split-role")
    parser.add_argument("--expected-schedule-id")
    parser.add_argument("--expected-depth-cap", type=int)
    parser.add_argument("--expected-master-seed", type=int)
    parser.add_argument("--expected-dual-profile-id")
    parser.add_argument("--mode", required=True, choices=("elite", "top-k"))
    parser.add_argument("--k", type=int, help="maximum distinct endpoints per protein in top-k mode")
    parser.add_argument(
        "--require-k", action="store_true",
        help=("fail if an eligible protein has fewer than K distinct strict endpoints; "
              "high-risk proteins with zero strict endpoints remain explicit exclusions"),
    )
    parser.add_argument("--output", required=True, help="output generated.parquet path")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    expected_fields = {
        "campaign_id": args.expected_campaign_id,
        "phase": args.expected_phase,
        "split_role": args.expected_split_role,
        "schedule_id": args.expected_schedule_id,
        "depth_cap": args.expected_depth_cap,
        "master_seed": args.expected_master_seed,
        "dual_structure_profile_id": args.expected_dual_profile_id,
    }
    if args.contract == "high-risk-r4":
        missing = sorted(field for field, value in expected_fields.items() if value is None)
        if missing:
            raise SystemExit(
                "FATAL: high-risk-r4 requires explicit expected identity fields: "
                + ", ".join(missing)
            )
        try:
            expected_identity = HighRiskFacadeIdentity(**expected_fields)
        except V2FacadeError as exc:
            raise SystemExit(f"FATAL: {exc}") from exc
    else:
        supplied = sorted(field for field, value in expected_fields.items() if value is not None)
        if supplied:
            raise SystemExit(
                "FATAL: --expected-* identity flags require --contract high-risk-r4: "
                + ", ".join(supplied)
            )
        expected_identity = None
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite to replace it: {output}")
    selection_output = output.with_suffix(output.suffix + ".selection.json")
    if args.contract == "high-risk-r4" and selection_output.exists() and not args.overwrite:
        raise SystemExit(
            f"selection provenance exists; pass --overwrite to replace it: {selection_output}"
        )
    try:
        facade = materialize_archive_facade(
            args.bundle, mode=args.mode, k=args.k, require_k=bool(args.require_k),
            contract=args.contract, expected_identity=expected_identity,
        )
    except V2FacadeError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    facade.to_parquet(output, index=False)
    if args.contract == "high-risk-r4":
        provenance = facade.attrs.get("selection_provenance")
        if not isinstance(provenance, Mapping):
            raise SystemExit("FATAL: high-risk facade lost its selection provenance")
        temporary = selection_output.with_suffix(selection_output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(selection_output)
    print(
        f"wrote {len(facade)} rows across {facade['protein_id'].nunique()} proteins to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
