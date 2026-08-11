"""Pure typed semantics for an explicitly requested exploratory dual-scTM gate.

The existing V2 structure path has one boolean verdict whose meaning is the strict/final gate.
This module does not weaken that legacy contract.  It provides a separate policy which derives two
nested decisions from one evaluated raw :class:`StructureOutcome` and an authorization that is
bound to the exact endpoint allowed to participate in exploratory ancestry search.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Mapping

from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome

from .errors import V2Error
from .identity import canonical_digest

if TYPE_CHECKING:
    from .safety import SearchAdmissionEvidence

__all__ = [
    "V2StructureGateError",
    "DUAL_SCTM_POLICY_KIND",
    "HIGH_RISK_DUAL_SCTM_PROFILE_ID",
    "HIGH_RISK_ANCESTRY_SCTM_MIN",
    "HIGH_RISK_STRICT_SCTM_MIN",
    "DualScTMGatePolicy",
    "DualStructureVerdict",
    "AncestryAuthorization",
    "authorize_ancestry",
]


DUAL_SCTM_POLICY_KIND = "exploratory_dual_sctm"
HIGH_RISK_DUAL_SCTM_PROFILE_ID = "exploratory_dual_sctm_ancestry070_strict085_v1"
HIGH_RISK_ANCESTRY_SCTM_MIN = 0.70
HIGH_RISK_STRICT_SCTM_MIN = 0.85


class V2StructureGateError(V2Error):
    """The dual structure policy or its evidence binding is malformed."""


def _finite_real(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _outcome_payload(outcome: StructureOutcome) -> dict[str, Any]:
    """Canonical evidence payload which remains JSON-safe for a malformed metric."""
    metrics: dict[str, float | str] = {}
    for name, value in sorted(dict(outcome.metrics or {}).items()):
        finite = _finite_real(value)
        metrics[str(name)] = finite if finite is not None else repr(value)
    return {
        "evaluated": bool(outcome.evaluated),
        "feasible": outcome.feasible,
        "cache_status": str(outcome.cache_status),
        "model_executed": bool(outcome.model_executed),
        "failure_reason": outcome.failure_reason,
        "walltime_s": _finite_real(outcome.walltime_s),
        "metrics": metrics,
    }


@dataclass(frozen=True, init=False)
class DualStructureVerdict:
    """Two nested scTM decisions derived from one raw structure evaluation."""

    profile_id: str
    ancestry_sctm_min: float
    strict_sctm_min: float
    evaluated: bool
    common_structure_feasible: bool
    sctm: float | None
    ancestry_structure_passed: bool
    strict_structure_passed: bool
    ancestry_reason: str
    strict_reason: str
    raw_outcome_digest: str

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "ancestry_sctm_min": self.ancestry_sctm_min,
            "strict_sctm_min": self.strict_sctm_min,
            "evaluated": self.evaluated,
            "common_structure_feasible": self.common_structure_feasible,
            "sctm": self.sctm,
            "ancestry_structure_passed": self.ancestry_structure_passed,
            "strict_structure_passed": self.strict_structure_passed,
            "ancestry_reason": self.ancestry_reason,
            "strict_reason": self.strict_reason,
            "raw_outcome_digest": self.raw_outcome_digest,
        }

    @property
    def verdict_digest(self) -> str:
        return canonical_digest(self.canonical_payload())


@dataclass(frozen=True)
class DualScTMGatePolicy:
    """Explicit dual gate; constructing and passing this object is the opt-in.

    ``StructureOutcome.feasible`` represents the structure predicates shared by both tiers.  The
    future oracle adapter must therefore evaluate those common predicates once and leave the two
    scTM thresholds to this policy.  Missing, deferred, non-finite, or common-infeasible evidence
    fails both tiers closed.
    """

    profile_id: str
    ancestry_sctm_min: float
    strict_sctm_min: float

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise V2StructureGateError("profile_id must be a non-empty string")
        ancestry = _finite_real(self.ancestry_sctm_min)
        strict = _finite_real(self.strict_sctm_min)
        if ancestry is None or strict is None or not (0.0 < ancestry < strict <= 1.0):
            raise V2StructureGateError(
                "dual scTM thresholds must satisfy 0 < ancestry_sctm_min < "
                "strict_sctm_min <= 1"
            )
        object.__setattr__(self, "ancestry_sctm_min", ancestry)
        object.__setattr__(self, "strict_sctm_min", strict)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "policy_kind": DUAL_SCTM_POLICY_KIND,
            "profile_id": self.profile_id,
            "ancestry_sctm_min": self.ancestry_sctm_min,
            "strict_sctm_min": self.strict_sctm_min,
        }

    @property
    def policy_digest(self) -> str:
        return canonical_digest(self.canonical_payload())

    def evaluate(self, outcome: StructureOutcome) -> DualStructureVerdict:
        if not isinstance(outcome, StructureOutcome):
            raise V2StructureGateError(
                "dual structure evaluation requires a raw StructureOutcome"
            )
        sctm = _finite_real((outcome.metrics or {}).get("scTM"))
        evaluated = bool(outcome.evaluated)
        common = evaluated and outcome.feasible is True
        ancestry_passed = bool(
            common and sctm is not None and sctm >= self.ancestry_sctm_min
        )
        strict_passed = bool(
            common and sctm is not None and sctm >= self.strict_sctm_min
        )
        if strict_passed and not ancestry_passed:  # defensive: constructor nesting should imply it
            raise V2StructureGateError("strict structure pass must imply ancestry structure pass")

        def reason(*, passed: bool, threshold: float) -> str:
            if passed:
                return "passed"
            if not evaluated:
                return "structure_not_evaluated"
            if outcome.feasible is not True:
                return "common_structure_gate_failed"
            if sctm is None:
                return "sctm_missing_or_nonfinite"
            return f"sctm_below_{threshold:g}"

        verdict = object.__new__(DualStructureVerdict)
        values = {
            "profile_id": self.profile_id,
            "ancestry_sctm_min": self.ancestry_sctm_min,
            "strict_sctm_min": self.strict_sctm_min,
            "evaluated": evaluated,
            "common_structure_feasible": bool(common),
            "sctm": sctm,
            "ancestry_structure_passed": ancestry_passed,
            "strict_structure_passed": strict_passed,
            "ancestry_reason": reason(
                passed=ancestry_passed, threshold=self.ancestry_sctm_min,
            ),
            "strict_reason": reason(
                passed=strict_passed, threshold=self.strict_sctm_min,
            ),
            "raw_outcome_digest": canonical_digest(_outcome_payload(outcome)),
        }
        for name, value in values.items():
            object.__setattr__(verdict, name, value)
        return verdict


def _endpoint_binding_payload(endpoint: Any) -> dict[str, Any]:
    binding = getattr(endpoint, "head_binding", None)
    evaluator = getattr(binding, "evaluator", None)
    return {
        "endpoint_id": str(getattr(endpoint, "endpoint_id", "")),
        "sequence_md5": str(getattr(endpoint, "sequence_md5", "")),
        "source_state_id": str(getattr(endpoint, "source_state_id", "")),
        "source_state_content_digest": str(
            getattr(endpoint, "source_state_content_digest", "")
        ),
        "head_evaluator_digest": (
            evaluator.digest() if evaluator is not None and hasattr(evaluator, "digest") else ""
        ),
        "head_window_grid_digest": str(getattr(binding, "window_grid_digest", "")),
    }


@dataclass(frozen=True, init=False)
class AncestryAuthorization:
    """Capability token for one exact ancestry-search endpoint, never a final-feasibility label."""

    profile_id: str
    endpoint_id: str
    sequence_md5: str
    endpoint_binding_digest: str
    structure_verdict_digest: str
    raw_outcome_digest: str
    admission_evidence_digest: str
    ancestry_sctm_min: float
    strict_sctm_min: float
    sctm: float
    strict_structure_passed: bool

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "authorization_kind": "exploratory_search_ancestry",
            "profile_id": self.profile_id,
            "endpoint_id": self.endpoint_id,
            "sequence_md5": self.sequence_md5,
            "endpoint_binding_digest": self.endpoint_binding_digest,
            "structure_verdict_digest": self.structure_verdict_digest,
            "raw_outcome_digest": self.raw_outcome_digest,
            "admission_evidence_digest": self.admission_evidence_digest,
            "ancestry_sctm_min": self.ancestry_sctm_min,
            "strict_sctm_min": self.strict_sctm_min,
            "sctm": self.sctm,
            "strict_structure_passed": self.strict_structure_passed,
        }

    @property
    def authorization_digest(self) -> str:
        return canonical_digest(self.canonical_payload())

    def binds_endpoint(self, endpoint: Any) -> bool:
        return (
            self.endpoint_id == getattr(endpoint, "endpoint_id", None)
            and self.sequence_md5 == getattr(endpoint, "sequence_md5", None)
            and self.endpoint_binding_digest
            == canonical_digest(_endpoint_binding_payload(endpoint))
        )

    def binds_structure(
        self, outcome: StructureOutcome, verdict: DualStructureVerdict,
    ) -> bool:
        """Return whether this capability was minted from these exact structure records."""
        return (
            isinstance(outcome, StructureOutcome)
            and isinstance(verdict, DualStructureVerdict)
            and self.profile_id == verdict.profile_id
            and self.structure_verdict_digest == verdict.verdict_digest
            and self.raw_outcome_digest == canonical_digest(_outcome_payload(outcome))
            and verdict.raw_outcome_digest == self.raw_outcome_digest
            and self.ancestry_sctm_min == verdict.ancestry_sctm_min
            and self.strict_sctm_min == verdict.strict_sctm_min
            and self.sctm == verdict.sctm
            and self.strict_structure_passed is verdict.strict_structure_passed
        )


def authorize_ancestry(
    *, endpoint: Any, structure_verdict: DualStructureVerdict,
    search_admission_evidence: SearchAdmissionEvidence,
) -> AncestryAuthorization:
    """Mint an endpoint-bound authorization from live typed admission and structure evidence."""
    if not isinstance(structure_verdict, DualStructureVerdict):
        raise V2StructureGateError("structure_verdict must be a DualStructureVerdict")
    if not structure_verdict.ancestry_structure_passed:
        raise V2StructureGateError(
            "an ancestry authorization requires a passing ancestry structure verdict"
        )
    if structure_verdict.sctm is None:
        raise V2StructureGateError("a passing ancestry verdict must carry finite scTM")
    # Constructor checks are not an authorization boundary: fixtures and deserializers can bypass
    # them.  Re-run the anchor/immune conjunction and recompute its digest from the carried verdict
    # at the exact point the capability is minted.
    # Local import keeps ``structure_gate`` a dependency leaf for ``state``.  ``safety`` imports
    # config -> policy -> state, so importing it at module load time would close that cycle.
    from .safety import validate_search_admission_evidence

    try:
        search_evidence = validate_search_admission_evidence(
            search_admission_evidence,
        )
    except V2Error as exc:
        raise V2StructureGateError(
            f"invalid SearchAdmissionEvidence: {exc}"
        ) from exc
    payload = _endpoint_binding_payload(endpoint)
    if not payload["endpoint_id"] or not payload["sequence_md5"]:
        raise V2StructureGateError("ancestry authorization requires an identified endpoint")
    if search_evidence.endpoint_id != payload["endpoint_id"]:
        raise V2StructureGateError(
            "search admission evidence describes a different endpoint"
        )
    if search_evidence.verdict.endpoint_sequence_md5 != payload["sequence_md5"]:
        raise V2StructureGateError(
            "search admission verdict describes a different endpoint sequence"
        )
    if search_evidence.structure_evidence_digest != structure_verdict.verdict_digest:
        raise V2StructureGateError(
            "search admission evidence was minted from a different structure verdict"
        )
    authorization = object.__new__(AncestryAuthorization)
    values = {
        "profile_id": structure_verdict.profile_id,
        "endpoint_id": payload["endpoint_id"],
        "sequence_md5": payload["sequence_md5"],
        "endpoint_binding_digest": canonical_digest(payload),
        "structure_verdict_digest": structure_verdict.verdict_digest,
        "raw_outcome_digest": structure_verdict.raw_outcome_digest,
        "admission_evidence_digest": search_evidence.admission_evidence_digest,
        "ancestry_sctm_min": structure_verdict.ancestry_sctm_min,
        "strict_sctm_min": structure_verdict.strict_sctm_min,
        "sctm": structure_verdict.sctm,
        "strict_structure_passed": structure_verdict.strict_structure_passed,
    }
    for name, value in values.items():
        object.__setattr__(authorization, name, value)
    return authorization
