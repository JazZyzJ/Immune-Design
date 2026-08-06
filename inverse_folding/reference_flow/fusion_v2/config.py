"""The strict, defaults-free V2 run configuration.

This module exists to make an unstated scientific choice impossible to run with. Two rules do most
of the work:

* **No library defaults for scientific fields.** Every value whose choice is an open decision --
  the A2 matching resource, the cumulative hotspot threshold, the population width, the minimum
  lookahead tail, every schedule coordinate -- is required. Omitting one is a load failure naming
  the offending path, never a fallback (PLAN §5.1).
* **Closed vocabularies stay closed, and unknown keys are rejected.** A deferred or legacy knob
  must fail rather than be silently dropped, or a config that looks like it enabled something will
  have run without it.

It performs no coordinate arithmetic of its own: every ordering law is delegated to
``fusion_v2.schedule``, so the config and the schedule can never disagree. It also touches no file
and imports no model, so ``--print-config`` and ``--dry-run`` derive the same digest with nothing
loaded.

h-maps are retired: there is no h-map field on this surface. The substrate guarantee is enforced by
the ``ConditioningDigest`` constructor in the identity layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import V2Error
from .identity import CONTENT_ROLE_TO_FIELD, FROZEN_DIGEST_ROLES, canonical_digest, require_digest
from .policy import (
    DIAGNOSTIC_ALLOWED_PHASES,
    DIAGNOSTIC_POLICY_ID_PREFIX,
    DIAGNOSTIC_ALLOWED_PHASES as _POLICY_DIAGNOSTIC_ALLOWED_PHASES,
    DIAGNOSTIC_POLICY_ID_PREFIX,
    DIAGNOSTIC_POLICY_IDS,
    DeclaredPolicy,
    policy_id_is_diagnostic,
    V2PolicyError,
    policy_id_is_diagnostic,
)
from .schedule import CoordinateLaw, DepthSchedule, make_cycle, make_depth_schedule

__all__ = [
    "V2ConfigError", "V2_CONFIG_SCHEMA_VERSION", "OMIT", "RUN_PHASES", "ARM_ROLES",
    "A2_RESOURCE_COMPONENTS", "MASK_LOAD_UNITS", "RETRY_SCOPES", "CALIBRATION_SOURCE_KINDS",
    "FORBIDDEN_SOURCE_KINDS", "FORBIDDEN_CALIBRATION_SOURCE_IDS",
    "V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION", "HOTSPOT_GATE_KIND", "HOTSPOT_WINDOW_DOMAIN",
    "CALIBRATION_SCOPES", "INCREMENTAL_REFERENCE_KIND", "INCREMENTAL_REFERENCE_LABEL",
    "DIAGNOSTIC_VALUE_PREFIX", "DIAGNOSTIC_ALLOWED_PHASES", "CONTENT_BINDINGS",
    "HotspotCalibrationArtifact", "CalibratedScalar", "calibration_source_ref",
    "ContentIdentity", "V2IdentityConfig",
    "V2SubstrateConfig", "V2ArmConfig", "DepthSchedulePoint", "V2ScheduleConfig",
    "V2ProjectionConfig", "V2HeadConfig", "V2SafetyConfig", "V2CapsConfig", "V2Config",
    "load_v2_config", "default_content_bindings",
]

V2_CONFIG_SCHEMA_VERSION = "v2cfg-1"


class _Omitted:
    """Sentinel used by tests and callers to delete a key from a payload."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "OMIT"


OMIT = _Omitted()


class V2ConfigError(V2Error):
    """The run configuration violated a PLAN §5.1 or §5.2 requirement.

    Deliberately not ``V1EntryConfigError`` nor ``FusionConfigError``: a V2 payload must never be
    accepted by a V1 or v0 loader, nor the reverse.
    """


RUN_PHASES = frozenset({
    "state_transition_canary", "mechanism_cohort", "policy_qualification", "capability_ladder",
    "holdout_application",
})
ARM_ROLES = frozenset({"v2", "a2", "v0_boundary"})

#: The five resource components of the FUSION_V2 §6.1 envelope. It is closed because the doc
#: forbids collapsing the vector "into one invented universal cost" -- a sixth name would be one.
A2_RESOURCE_COMPONENTS = frozenset({
    "logical_dfe", "head_calls", "definitive_refolds", "gpu_seconds", "walltime_s",
})
MASK_LOAD_UNITS = frozenset({"absolute_positions", "normalized_editable_fraction"})
RETRY_SCOPES = frozenset({"per_request", "per_run"})
CALIBRATION_SOURCE_KINDS = frozenset({"measured_calibration", "runbook_frozen"})
FORBIDDEN_SOURCE_KINDS = frozenset({"inherited_v0", "default", "placeholder", "guess"})
FORBIDDEN_CALIBRATION_SOURCE_IDS = frozenset({"objective.max_offtarget_window_increase"})
CONTENT_BINDINGS = frozenset({"frozen", "runtime"})

V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION = "v2-hotspot-calibration-1"
HOTSPOT_GATE_KIND = "whole_landscape_new_hotspot"
HOTSPOT_WINDOW_DOMAIN = "whole_landscape"
CALIBRATION_SCOPES = frozenset({"cumulative_depth0", "immediate_parent"})
INCREMENTAL_REFERENCE_KIND = "selected_endpoint"
INCREMENTAL_REFERENCE_LABEL = "immediate_parent"

#: Re-exported from ``fusion_v2.policy``, which OWNS the diagnostic vocabulary because it owns the
#: policies being classified. These used to be independent definitions here, and they disagreed:
#: the config called a policy diagnostic when its id started with ``diag_``, while the only
#: diagnostic policy in the repo names itself ``explicit_probe``. Declaring that probe honestly
#: therefore satisfied the config's consistency check and walked past the phase gate.
DIAGNOSTIC_VALUE_PREFIX = DIAGNOSTIC_POLICY_ID_PREFIX

#: Likewise owned by ``fusion_v2.policy``: PLAN §2.5 admits a diagnostic policy for "deterministic
#: tests and the first state-transition diagnostic" only, and widening it is an authority edit in
#: code, never a config key that self-authorizes.
DIAGNOSTIC_ALLOWED_PHASES = _POLICY_DIAGNOSTIC_ALLOWED_PHASES


def default_content_bindings() -> dict[str, str]:
    """Every PLAN §5.2 role and whether its digest must be frozen in the config."""
    return {
        role: ("frozen" if role in FROZEN_DIGEST_ROLES else "runtime")
        for role in sorted(CONTENT_ROLE_TO_FIELD)
    }


# --------------------------------------------------------------------------------------------
# Typed readers. Each one names the path it failed on, so a missing field is actionable.
# --------------------------------------------------------------------------------------------

def _section(payload: Mapping[str, Any], name: str, path: str) -> Mapping[str, Any]:
    if name not in payload:
        raise V2ConfigError(f"{path}: required section {name!r} is missing")
    value = payload[name]
    if not isinstance(value, Mapping):
        raise V2ConfigError(f"{path}.{name} must be a mapping")
    return value


def _require(node: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in node:
        raise V2ConfigError(
            f"{path}.{key} is required and has no default; a scientific field must be declared "
            "(PLAN §5.1)"
        )
    return node[key]


def _text(node: Mapping[str, Any], key: str, path: str, *, allowed: frozenset[str] | None = None) -> str:
    value = _require(node, key, path)
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise V2ConfigError(f"{path}.{key} must be a non-empty str, got {value!r}")
    if allowed is not None and value not in allowed:
        raise V2ConfigError(f"{path}.{key} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _int(node: Mapping[str, Any], key: str, path: str, *, minimum: int = 0) -> int:
    value = _require(node, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise V2ConfigError(f"{path}.{key} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise V2ConfigError(f"{path}.{key} must be >= {minimum}, got {value}")
    return value


def _float(node: Mapping[str, Any], key: str, path: str) -> float:
    value = _require(node, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2ConfigError(f"{path}.{key} must be a real number, got {type(value).__name__}")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise V2ConfigError(f"{path}.{key} must be finite, got {out!r}")
    return out


def _bool(node: Mapping[str, Any], key: str, path: str) -> bool:
    value = _require(node, key, path)
    if not isinstance(value, bool):
        raise V2ConfigError(
            f"{path}.{key} must be an explicit bool, got {value!r}; a coercible value is not a "
            "declaration"
        )
    return value


def _code_revision(node: Mapping[str, Any], key: str, path: str) -> str:
    """One rule for the run's code revision, owned by the schedule layer.

    ``fusion_v2.schedule`` already refuses placeholders and requires an explicit hexadecimal
    revision for the identical quantity on a band-calibration row.  A second, looser rule here
    would mean the same field was signed under two different standards.
    """
    from .schedule import V2ScheduleError, _require_code_revision

    raw = _text(node, key, path)
    try:
        return _require_code_revision(raw)
    except V2ScheduleError as exc:
        raise V2ConfigError(f"{path}.{key}: {exc}") from exc


def _reject_unknown(node: Mapping[str, Any], known: Sequence[str], path: str) -> None:
    extra = sorted(set(node) - set(known))
    if extra:
        raise V2ConfigError(
            f"{path}: unknown key(s) {extra}; a deferred or legacy knob is rejected, never ignored"
        )


@dataclass(frozen=True)
class HotspotCalibrationArtifact:
    """Typed identity of the data domain that calibrated one V2 hotspot threshold."""

    schema_version: str
    gate_kind: str
    scope: str
    reference_kind: str
    reference_label: str
    window_domain: str
    allele: str
    score_scale: str
    window_k_min: int
    window_k_max: int
    calibration_data_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION:
            raise V2ConfigError(
                f"calibration artifact schema must be {V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )
        if self.gate_kind != HOTSPOT_GATE_KIND:
            raise V2ConfigError(
                f"calibration gate_kind must be {HOTSPOT_GATE_KIND!r}, got {self.gate_kind!r}; "
                "v0 off-halo calibration is a different measurement"
            )
        if self.scope not in CALIBRATION_SCOPES:
            raise V2ConfigError(
                f"calibration scope must be one of {sorted(CALIBRATION_SCOPES)}, got {self.scope!r}"
            )
        if self.window_domain != HOTSPOT_WINDOW_DOMAIN:
            raise V2ConfigError(
                f"calibration window_domain must be {HOTSPOT_WINDOW_DOMAIN!r}, got "
                f"{self.window_domain!r}; an off-halo artifact cannot authorize the V2 gate"
            )
        for name in ("reference_kind", "reference_label", "allele", "score_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
                raise V2ConfigError(f"calibration artifact {name} must be a non-empty str")
        for name in ("window_k_min", "window_k_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise V2ConfigError(f"calibration artifact {name} must be a positive int")
        if self.window_k_max < self.window_k_min:
            raise V2ConfigError("calibration artifact window_k_max is below window_k_min")
        require_digest(self.calibration_data_digest, "calibration_data_digest")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate_kind": self.gate_kind,
            "scope": self.scope,
            "reference_kind": self.reference_kind,
            "reference_label": self.reference_label,
            "window_domain": self.window_domain,
            "allele": self.allele,
            "score_scale": self.score_scale,
            "window_k_min": self.window_k_min,
            "window_k_max": self.window_k_max,
            "calibration_data_digest": self.calibration_data_digest,
        }


def calibration_source_ref(
    *, value: float, unit: str, source_kind: str, source_id: str,
    artifact: HotspotCalibrationArtifact,
) -> str:
    """Bind a threshold source reference to all decision-relevant calibration content."""
    return canonical_digest({
        "value": float(value),
        "unit": unit,
        "source_kind": source_kind,
        "source_id": source_id,
        "artifact": artifact.canonical_payload(),
    })


@dataclass(frozen=True)
class CalibratedScalar:
    """A scalar bound to a typed V2 calibration artifact, not a provenance label alone."""

    value: float
    unit: str
    source_kind: str
    source_id: str
    source_ref: str
    artifact: HotspotCalibrationArtifact

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise V2ConfigError("calibrated value must be a real number")
        value = float(self.value)
        if value != value or value in (float("inf"), float("-inf")) or value < 0.0:
            raise V2ConfigError(f"calibrated value must be finite and >= 0, got {self.value!r}")
        object.__setattr__(self, "value", value)
        if isinstance(self.unit, bool) or not isinstance(self.unit, str) or not self.unit.strip():
            raise V2ConfigError("calibration unit must be a non-empty str")
        if self.source_kind in FORBIDDEN_SOURCE_KINDS:
            raise V2ConfigError(
                f"source_kind {self.source_kind!r} is not a calibration; v0's threshold was "
                "measured against a different window domain and is never inherited (PLAN §2.7)"
            )
        if self.source_kind not in CALIBRATION_SOURCE_KINDS:
            raise V2ConfigError(
                f"source_kind must be one of {sorted(CALIBRATION_SOURCE_KINDS)}"
            )
        normalized_source_id = (
            self.source_id.strip().lower() if isinstance(self.source_id, str) else ""
        )
        if normalized_source_id in FORBIDDEN_CALIBRATION_SOURCE_IDS:
            raise V2ConfigError(
                f"source_id {self.source_id!r} names the v0 off-halo objective; changing only its "
                "source_kind is label laundering, not V2 whole-landscape calibration"
            )
        if isinstance(self.source_id, bool) or not isinstance(self.source_id, str) \
                or not self.source_id.strip():
            raise V2ConfigError("source_id must be a non-empty str")
        if not isinstance(self.artifact, HotspotCalibrationArtifact):
            raise V2ConfigError("artifact must be a HotspotCalibrationArtifact")
        require_digest(self.source_ref, "source_ref")
        expected = calibration_source_ref(
            value=value, unit=self.unit, source_kind=self.source_kind,
            source_id=self.source_id, artifact=self.artifact,
        )
        if self.source_ref != expected:
            raise V2ConfigError(
                "source_ref does not equal the canonical digest of the threshold and typed "
                "calibration artifact; provenance labels are not content binding"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "value": self.value, "unit": self.unit, "source_kind": self.source_kind,
            "source_id": self.source_id, "source_ref": self.source_ref,
            "artifact": self.artifact.canonical_payload(),
        }


def _calibrated(node: Mapping[str, Any], key: str, path: str) -> CalibratedScalar:
    raw = _require(node, key, path)
    if not isinstance(raw, Mapping):
        raise V2ConfigError(f"{path}.{key} must be a mapping carrying value and provenance")
    _reject_unknown(raw, ("value", "unit", "source_kind", "source_id", "source_ref", "artifact"),
                    f"{path}.{key}")
    artifact_raw = _require(raw, "artifact", f"{path}.{key}")
    if not isinstance(artifact_raw, Mapping):
        raise V2ConfigError(f"{path}.{key}.artifact must be a mapping")
    artifact_path = f"{path}.{key}.artifact"
    _reject_unknown(
        artifact_raw,
        ("schema_version", "gate_kind", "scope", "reference_kind", "reference_label",
         "window_domain", "allele", "score_scale", "window_k_min", "window_k_max",
         "calibration_data_digest"),
        artifact_path,
    )
    artifact = HotspotCalibrationArtifact(
        schema_version=_text(artifact_raw, "schema_version", artifact_path),
        gate_kind=_text(artifact_raw, "gate_kind", artifact_path),
        scope=_text(artifact_raw, "scope", artifact_path),
        reference_kind=_text(artifact_raw, "reference_kind", artifact_path),
        reference_label=_text(artifact_raw, "reference_label", artifact_path),
        window_domain=_text(artifact_raw, "window_domain", artifact_path),
        allele=_text(artifact_raw, "allele", artifact_path),
        score_scale=_text(artifact_raw, "score_scale", artifact_path),
        window_k_min=_int(artifact_raw, "window_k_min", artifact_path, minimum=1),
        window_k_max=_int(artifact_raw, "window_k_max", artifact_path, minimum=1),
        calibration_data_digest=_text(
            artifact_raw, "calibration_data_digest", artifact_path),
    )
    return CalibratedScalar(
        value=_float(raw, "value", f"{path}.{key}"),
        unit=_text(raw, "unit", f"{path}.{key}"),
        source_kind=_text(raw, "source_kind", f"{path}.{key}"),
        source_id=_text(raw, "source_id", f"{path}.{key}"),
        source_ref=_text(raw, "source_ref", f"{path}.{key}"),
        artifact=artifact,
    )


@dataclass(frozen=True)
class ContentIdentity:
    """One PLAN §5.2 provenance row."""

    role: str
    label: str
    binding: str
    expected_sha256: str | None

    def canonical_payload(self) -> dict[str, Any]:
        return {"role": self.role, "label": self.label, "binding": self.binding,
                "expected_sha256": self.expected_sha256}


@dataclass(frozen=True)
class V2IdentityConfig:
    campaign_id: str
    split_role: str
    phase: str
    master_seed: int
    seed_schema: str
    code_revision: str


@dataclass(frozen=True)
class V2SubstrateConfig:
    n_steps: int
    temperature: float
    amplification_form: str
    controller_enabled: bool
    remask_enabled: bool
    remask_fraction_scale: float
    rf_config_label: str


@dataclass(frozen=True)
class V2ArmConfig:
    feedback_enabled: bool
    arm_role: str
    a2_matching_resource: str
    a2_unmatched_reported: tuple[str, ...]


@dataclass(frozen=True)
class DepthSchedulePoint:
    depth: int
    r_step: int
    c_source_step: int
    c_next_step: int
    n_lookaheads: int
    band_key: str


@dataclass(frozen=True)
class V2ScheduleConfig:
    schedule_id: str
    coordinate_law: CoordinateLaw
    depth_cap: int
    active_population_width: int
    min_lookahead_tail_steps: int
    points: tuple[DepthSchedulePoint, ...]

    def __post_init__(self) -> None:
        # ``active_population_width`` is a required field precisely so a run must state how many
        # lineages it advances -- but the engine advances exactly one per cycle (PLAN §4.4), and the
        # granularity of the cumulative safety binding under a wider population is an OPEN question
        # (Interface Map OQ7: nothing says whether a forked family inherits its parent's
        # ``reference_binding_id`` or opens its own).  Accepting a width the engine cannot honour
        # would let a config claim a breadth that silently never happened, and a run would report a
        # population search it never performed.
        if int(self.active_population_width) != 1:
            raise V2ConfigError(
                f"config.schedule.active_population_width={self.active_population_width} is not "
                "supported: the depth ladder advances exactly one selected lineage per cycle, and "
                "the lineage granularity of the cumulative safety reference under a wider active "
                "population is an unresolved question (Interface Map OQ7).  Declare 1 until that "
                "is decided; the field stays required so the choice is always explicit."
            )

    def to_depth_schedule(self, n_steps: int) -> DepthSchedule:
        """Delegate every ordering check to the schedule layer, which owns the law."""
        cycles = [
            make_cycle(depth=point.depth, r_step=point.r_step,
                       c_source_step=point.c_source_step, c_next_step=point.c_next_step,
                       n_steps=n_steps, law=self.coordinate_law)
            for point in self.points
        ]
        return make_depth_schedule(
            schedule_id=self.schedule_id, law=self.coordinate_law, n_steps=n_steps,
            depth_cap=self.depth_cap, cycles=cycles,
        )


@dataclass(frozen=True)
class V2ProjectionConfig:
    support_policy_id: str
    support_policy_version: str
    support_policy_is_diagnostic: bool
    temporal_history_rule: str
    assimilation_rule: str
    admissible_mask_load_unit: str


@dataclass(frozen=True)
class V2HeadConfig:
    allele: str
    score_scale: str
    window_k_min: int
    window_k_max: int


@dataclass(frozen=True)
class V2SafetyConfig:
    cumulative_reference_kind: str
    cumulative_reference_label: str
    delta_new_cumulative: CalibratedScalar
    incremental_gate_enabled: bool
    delta_new_incremental: CalibratedScalar | None
    structure_cadence: str


@dataclass(frozen=True)
class V2CapsConfig:
    max_logical_dfe: int
    max_head_calls: int
    max_definitive_refolds: int
    max_gpu_seconds: int
    max_walltime_s: int
    max_retries: int
    retry_scope: str


@dataclass(frozen=True)
class V2Config:
    schema_version: str
    identity: V2IdentityConfig
    substrate: V2SubstrateConfig
    arm: V2ArmConfig
    schedule: V2ScheduleConfig
    projection: V2ProjectionConfig
    head: V2HeadConfig
    safety: V2SafetyConfig
    caps: V2CapsConfig
    content: tuple[ContentIdentity, ...]

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "identity": vars(self.identity),
            "substrate": vars(self.substrate),
            "arm": {**vars(self.arm),
                    "a2_unmatched_reported": list(self.arm.a2_unmatched_reported)},
            "schedule": {
                "schedule_id": self.schedule.schedule_id,
                "coordinate_law": self.schedule.coordinate_law.value,
                "depth_cap": self.schedule.depth_cap,
                "active_population_width": self.schedule.active_population_width,
                "min_lookahead_tail_steps": self.schedule.min_lookahead_tail_steps,
                "points": [vars(point) for point in self.schedule.points],
            },
            "projection": vars(self.projection),
            "head": vars(self.head),
            "safety": {
                "cumulative_reference_kind": self.safety.cumulative_reference_kind,
                "cumulative_reference_label": self.safety.cumulative_reference_label,
                "delta_new_cumulative": self.safety.delta_new_cumulative.canonical_payload(),
                "incremental_gate_enabled": self.safety.incremental_gate_enabled,
                "delta_new_incremental": (
                    None if self.safety.delta_new_incremental is None
                    else self.safety.delta_new_incremental.canonical_payload()
                ),
                "structure_cadence": self.safety.structure_cadence,
            },
            "caps": vars(self.caps),
            "content": [row.canonical_payload() for row in self.content],
        }

    def declared_policy(self) -> DeclaredPolicy:
        """The run's policy declaration, as the projection kernel consumes it (PLAN §2.5).

        Emitted as a TYPE rather than left as three loose config fields, so the kernel matches the
        answering identity against one object instead of re-deriving a phase rule it has no
        business knowing. Constructing it is itself the parse-time gate: a diagnostic declaration
        outside ``DIAGNOSTIC_ALLOWED_PHASES`` cannot be built, which is PLAN V2F1's "fails before
        model calls".
        """
        return DeclaredPolicy(
            policy_id=self.projection.support_policy_id,
            policy_version=self.projection.support_policy_version,
            is_diagnostic=self.projection.support_policy_is_diagnostic,
            phase=self.identity.phase,
        )

    def config_digest(self) -> str:
        """One shared digest, derivable with no model and no file access."""
        return canonical_digest(self.canonical_payload())


def _assert_hotspot_calibration_matches(
    scalar: CalibratedScalar,
    *,
    scope: str,
    reference_kind: str,
    reference_label: str,
    head: V2HeadConfig,
    path: str,
) -> None:
    artifact = scalar.artifact
    if scalar.unit != head.score_scale:
        raise V2ConfigError(
            f"{path}.unit={scalar.unit!r} does not match Head score_scale={head.score_scale!r}"
        )
    expected = {
        "scope": scope,
        "reference_kind": reference_kind,
        "reference_label": reference_label,
        "allele": head.allele,
        "score_scale": head.score_scale,
        "window_k_min": head.window_k_min,
        "window_k_max": head.window_k_max,
    }
    for field, wanted in expected.items():
        observed = getattr(artifact, field)
        if observed != wanted:
            raise V2ConfigError(
                f"{path}.artifact.{field}={observed!r} does not match the configured "
                f"V2 gate domain {wanted!r}"
            )


def load_v2_config(payload: Mapping[str, Any]) -> V2Config:
    """Parse and validate a V2 run configuration. Reads nothing; loads nothing."""
    if not isinstance(payload, Mapping):
        raise V2ConfigError("a V2 config must be a mapping")
    _reject_unknown(payload, ("schema_version", "identity", "substrate", "arm", "schedule",
                              "projection", "head", "safety", "caps", "content"), "config")
    schema = _text(payload, "schema_version", "config")
    if schema != V2_CONFIG_SCHEMA_VERSION:
        raise V2ConfigError(
            f"schema_version {schema!r} != {V2_CONFIG_SCHEMA_VERSION!r}"
        )

    ident_node = _section(payload, "identity", "config")
    _reject_unknown(ident_node, ("campaign_id", "split_role", "phase", "master_seed",
                                 "seed_schema", "code_revision"), "config.identity")
    identity = V2IdentityConfig(
        campaign_id=_text(ident_node, "campaign_id", "config.identity"),
        split_role=_text(ident_node, "split_role", "config.identity"),
        phase=_text(ident_node, "phase", "config.identity", allowed=RUN_PHASES),
        master_seed=_int(ident_node, "master_seed", "config.identity"),
        seed_schema=_text(ident_node, "seed_schema", "config.identity"),
        # Delegated to the schedule layer's rule for the SAME quantity, rather than accepting any
        # non-empty string.  ``code_revision`` signs the run (PLAN §5.2) and is inside
        # ``config_digest``; a config declaring ``unknown`` would be signed under a revision that
        # names nothing, and a resume could not tell two builds apart -- which is the one question
        # the field exists to answer.  Reused, not re-implemented, so the two cannot drift.
        code_revision=_code_revision(ident_node, "code_revision", "config.identity"),
    )

    sub_node = _section(payload, "substrate", "config")
    _reject_unknown(sub_node, ("n_steps", "temperature", "amplification_form",
                               "controller_enabled", "remask_enabled", "remask_fraction_scale",
                               "rf_config_label"), "config.substrate")
    substrate = V2SubstrateConfig(
        n_steps=_int(sub_node, "n_steps", "config.substrate", minimum=1),
        temperature=_float(sub_node, "temperature", "config.substrate"),
        amplification_form=_text(sub_node, "amplification_form", "config.substrate"),
        controller_enabled=_bool(sub_node, "controller_enabled", "config.substrate"),
        remask_enabled=_bool(sub_node, "remask_enabled", "config.substrate"),
        remask_fraction_scale=_float(sub_node, "remask_fraction_scale", "config.substrate"),
        rf_config_label=_text(sub_node, "rf_config_label", "config.substrate"),
    )
    _assert_frozen_substrate(substrate)

    arm_node = _section(payload, "arm", "config")
    _reject_unknown(arm_node, ("feedback_enabled", "arm_role", "a2_matching_resource",
                               "a2_unmatched_reported"), "config.arm")
    matched = _text(arm_node, "a2_matching_resource", "config.arm",
                    allowed=A2_RESOURCE_COMPONENTS)
    unmatched = tuple(_require(arm_node, "a2_unmatched_reported", "config.arm"))
    if set(unmatched) != A2_RESOURCE_COMPONENTS - {matched}:
        raise V2ConfigError(
            f"config.arm.a2_unmatched_reported must be exactly the complement of "
            f"{matched!r}, i.e. {sorted(A2_RESOURCE_COMPONENTS - {matched})}; the envelope is a "
            "vector and the unmatched components must be reported, not dropped (FUSION_V2 §6.1)"
        )
    feedback_enabled = _bool(arm_node, "feedback_enabled", "config.arm")
    arm_role = _text(arm_node, "arm_role", "config.arm", allowed=ARM_ROLES)
    expected_feedback = arm_role == "v2"
    if feedback_enabled is not expected_feedback:
        raise V2ConfigError(
            f"config.arm arm_role={arm_role!r} requires feedback_enabled={expected_feedback}; "
            "only V2 enables source-coupled projection feedback; A2 is its feedback-disabled "
            "view and v0 remains the source-forgetting boundary comparator"
        )
    arm = V2ArmConfig(
        feedback_enabled=feedback_enabled,
        arm_role=arm_role,
        a2_matching_resource=matched, a2_unmatched_reported=tuple(sorted(unmatched)),
    )

    sched_node = _section(payload, "schedule", "config")
    _reject_unknown(sched_node, ("schedule_id", "coordinate_law", "depth_cap",
                                 "active_population_width", "min_lookahead_tail_steps", "points"),
                    "config.schedule")
    law_text = _text(sched_node, "coordinate_law", "config.schedule",
                     allowed=frozenset(member.value for member in CoordinateLaw))
    raw_points = _require(sched_node, "points", "config.schedule")
    if not isinstance(raw_points, Sequence) or not raw_points:
        raise V2ConfigError("config.schedule.points must be a non-empty sequence")
    points = []
    for index, raw in enumerate(raw_points):
        where = f"config.schedule.points[{index}]"
        if not isinstance(raw, Mapping):
            raise V2ConfigError(f"{where} must be a mapping")
        _reject_unknown(raw, ("depth", "r_step", "c_source_step", "c_next_step", "n_lookaheads",
                              "band_key"), where)
        points.append(DepthSchedulePoint(
            depth=_int(raw, "depth", where), r_step=_int(raw, "r_step", where),
            c_source_step=_int(raw, "c_source_step", where),
            c_next_step=_int(raw, "c_next_step", where),
            n_lookaheads=_int(raw, "n_lookaheads", where, minimum=1),
            band_key=_text(raw, "band_key", where),
        ))
    schedule = V2ScheduleConfig(
        schedule_id=_text(sched_node, "schedule_id", "config.schedule"),
        coordinate_law=CoordinateLaw(law_text),
        depth_cap=_int(sched_node, "depth_cap", "config.schedule", minimum=1),
        active_population_width=_int(sched_node, "active_population_width", "config.schedule",
                                     minimum=1),
        min_lookahead_tail_steps=_int(sched_node, "min_lookahead_tail_steps", "config.schedule",
                                      minimum=1),
        points=tuple(points),
    )
    # Delegate: the schedule layer owns every ordering law, so the two can never disagree.
    schedule.to_depth_schedule(substrate.n_steps)
    for point in schedule.points:
        tail = substrate.n_steps - point.c_source_step
        if tail < schedule.min_lookahead_tail_steps:
            raise V2ConfigError(
                f"depth {point.depth} leaves a lookahead tail of {tail} steps, below the declared "
                f"min_lookahead_tail_steps={schedule.min_lookahead_tail_steps}; as c_d approaches "
                "S the K lookaheads differ by too few draws for endpoint selection to have input"
            )

    proj_node = _section(payload, "projection", "config")
    _reject_unknown(proj_node, ("support_policy_id", "support_policy_version",
                                "support_policy_is_diagnostic", "temporal_history_rule",
                                "assimilation_rule", "admissible_mask_load_unit"),
                    "config.projection")
    policy_id = _text(proj_node, "support_policy_id", "config.projection")
    is_diagnostic = _bool(proj_node, "support_policy_is_diagnostic", "config.projection")
    if policy_id_is_diagnostic(policy_id) != is_diagnostic:
        raise V2ConfigError(
            f"config.projection: policy {policy_id!r} and support_policy_is_diagnostic="
            f"{is_diagnostic} disagree with the POLICY vocabulary, which classifies it as "
            f"{policy_id_is_diagnostic(policy_id)}. Diagnostic ids are the registry "
            f"{sorted(DIAGNOSTIC_POLICY_IDS)} plus anything prefixed {DIAGNOSTIC_POLICY_ID_PREFIX!r} "
            "(PLAN §2.5)"
        )
    if is_diagnostic and identity.phase not in DIAGNOSTIC_ALLOWED_PHASES:
        raise V2ConfigError(
            f"diagnostic policy {policy_id!r} is not admissible in phase {identity.phase!r}; "
            f"allowed phases are {sorted(DIAGNOSTIC_ALLOWED_PHASES)} and widening that requires an "
            "authority edit, never a config key (PLAN §2.5, §5.1)"
        )
    projection = V2ProjectionConfig(
        support_policy_id=policy_id,
        support_policy_version=_text(proj_node, "support_policy_version", "config.projection"),
        support_policy_is_diagnostic=is_diagnostic,
        temporal_history_rule=_text(proj_node, "temporal_history_rule", "config.projection"),
        assimilation_rule=_text(proj_node, "assimilation_rule", "config.projection"),
        admissible_mask_load_unit=_text(proj_node, "admissible_mask_load_unit",
                                        "config.projection", allowed=MASK_LOAD_UNITS),
    )

    head_node = _section(payload, "head", "config")
    _reject_unknown(head_node, ("allele", "score_scale", "window_k_min", "window_k_max"),
                    "config.head")
    head = V2HeadConfig(
        allele=_text(head_node, "allele", "config.head"),
        score_scale=_text(head_node, "score_scale", "config.head"),
        window_k_min=_int(head_node, "window_k_min", "config.head", minimum=1),
        window_k_max=_int(head_node, "window_k_max", "config.head", minimum=1),
    )
    if head.window_k_max < head.window_k_min:
        raise V2ConfigError("config.head.window_k_max is below window_k_min")

    safety_node = _section(payload, "safety", "config")
    _reject_unknown(safety_node, ("cumulative_reference_kind", "cumulative_reference_label",
                                  "delta_new_cumulative", "incremental_gate_enabled",
                                  "delta_new_incremental", "structure_cadence"), "config.safety")
    cumulative_reference_kind = _text(
        safety_node, "cumulative_reference_kind", "config.safety")
    cumulative_reference_label = _text(
        safety_node, "cumulative_reference_label", "config.safety")
    cumulative_threshold = _calibrated(
        safety_node, "delta_new_cumulative", "config.safety")
    _assert_hotspot_calibration_matches(
        cumulative_threshold,
        scope="cumulative_depth0",
        reference_kind=cumulative_reference_kind,
        reference_label=cumulative_reference_label,
        head=head,
        path="config.safety.delta_new_cumulative",
    )
    incremental_enabled = _bool(safety_node, "incremental_gate_enabled", "config.safety")
    has_incremental = "delta_new_incremental" in safety_node
    if incremental_enabled and not has_incremental:
        raise V2ConfigError(
            "config.safety.delta_new_incremental is required when incremental_gate_enabled is true"
        )
    if not incremental_enabled and has_incremental:
        raise V2ConfigError(
            "config.safety.delta_new_incremental is forbidden when the incremental gate is off; "
            "whether an incremental gate exists at all is an open decision, so a dormant threshold "
            "would be an invented default (PLAN §0.4)"
        )
    incremental_threshold = (
        _calibrated(safety_node, "delta_new_incremental", "config.safety")
        if incremental_enabled else None
    )
    if incremental_threshold is not None:
        _assert_hotspot_calibration_matches(
            incremental_threshold,
            scope="immediate_parent",
            reference_kind=INCREMENTAL_REFERENCE_KIND,
            reference_label=INCREMENTAL_REFERENCE_LABEL,
            head=head,
            path="config.safety.delta_new_incremental",
        )
    safety = V2SafetyConfig(
        cumulative_reference_kind=cumulative_reference_kind,
        cumulative_reference_label=cumulative_reference_label,
        delta_new_cumulative=cumulative_threshold,
        incremental_gate_enabled=incremental_enabled,
        delta_new_incremental=incremental_threshold,
        structure_cadence=_text(safety_node, "structure_cadence", "config.safety"),
    )

    caps_node = _section(payload, "caps", "config")
    _reject_unknown(caps_node, ("max_logical_dfe", "max_head_calls", "max_definitive_refolds",
                                "max_gpu_seconds", "max_walltime_s", "max_retries",
                                "retry_scope"), "config.caps")
    caps = V2CapsConfig(
        max_logical_dfe=_int(caps_node, "max_logical_dfe", "config.caps", minimum=1),
        max_head_calls=_int(caps_node, "max_head_calls", "config.caps", minimum=1),
        max_definitive_refolds=_int(caps_node, "max_definitive_refolds", "config.caps", minimum=1),
        max_gpu_seconds=_int(caps_node, "max_gpu_seconds", "config.caps", minimum=1),
        max_walltime_s=_int(caps_node, "max_walltime_s", "config.caps", minimum=1),
        max_retries=_int(caps_node, "max_retries", "config.caps"),
        retry_scope=_text(caps_node, "retry_scope", "config.caps", allowed=RETRY_SCOPES),
    )

    content = _content_identities(payload)
    return V2Config(
        schema_version=schema, identity=identity, substrate=substrate, arm=arm,
        schedule=schedule, projection=projection, head=head, safety=safety, caps=caps,
        content=content,
    )


def _assert_frozen_substrate(substrate: V2SubstrateConfig) -> None:
    """The first V2/A2 substrate is frozen (PLAN §5.1) and every arm must declare it identically."""
    if substrate.controller_enabled:
        raise V2ConfigError("config.substrate: every V2/A2 arm is controller-free")
    if substrate.amplification_form != "constant_one":
        raise V2ConfigError(
            f"config.substrate.amplification_form must be 'constant_one', got "
            f"{substrate.amplification_form!r}; a position-dependent kernel is a different method"
        )
    if not substrate.remask_enabled:
        raise V2ConfigError(
            "config.substrate.remask_enabled must be true with remask_fraction_scale=0.0; "
            "disabling remask entirely also skips the post-step lifecycle, which conflates 'no "
            "remask' with 'no post-step'"
        )
    if substrate.remask_fraction_scale != 0.0:
        raise V2ConfigError(
            f"config.substrate.remask_fraction_scale must be 0.0, got "
            f"{substrate.remask_fraction_scale}; background remask is a global reopen rule that "
            "would compete with q_phi for control of reopened identities (PLAN §1.5, §5.1)"
        )
    if substrate.temperature <= 0.0:
        raise V2ConfigError("config.substrate.temperature must be positive")


def _content_identities(payload: Mapping[str, Any]) -> tuple[ContentIdentity, ...]:
    raw_rows = payload.get("content")
    if not isinstance(raw_rows, Sequence) or not raw_rows:
        raise V2ConfigError("config.content must be a non-empty sequence of provenance rows")
    rows: dict[str, ContentIdentity] = {}
    for index, raw in enumerate(raw_rows):
        where = f"config.content[{index}]"
        if not isinstance(raw, Mapping):
            raise V2ConfigError(f"{where} must be a mapping")
        _reject_unknown(raw, ("role", "label", "binding", "expected_sha256"), where)
        role = _text(raw, "role", where, allowed=frozenset(CONTENT_ROLE_TO_FIELD))
        binding = _text(raw, "binding", where, allowed=CONTENT_BINDINGS)
        expected = raw.get("expected_sha256")
        if binding == "frozen":
            require_digest(expected, f"{where}.expected_sha256")
        elif expected is not None:
            raise V2ConfigError(
                f"{where}: a runtime-bound role carries no expected digest; it is observed, not "
                "declared"
            )
        if role in rows:
            raise V2ConfigError(f"{where}: duplicate content role {role!r}")
        rows[role] = ContentIdentity(
            role=role, label=_text(raw, "label", where), binding=binding, expected_sha256=expected,
        )
    missing = sorted(set(CONTENT_ROLE_TO_FIELD) - set(rows))
    if missing:
        raise V2ConfigError(
            f"config.content is missing role(s) {missing}; missing content identity fails closed "
            "(PLAN §5.2)"
        )
    for role in sorted(FROZEN_DIGEST_ROLES):
        if rows[role].binding != "frozen":
            raise V2ConfigError(
                f"config.content: role {role!r} must be frozen, because its content has to be "
                "bound before runtime (PLAN §2.1, §2.5, §2.7)"
            )
    return tuple(rows[role] for role in sorted(rows))
