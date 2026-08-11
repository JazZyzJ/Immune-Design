"""The whole-landscape safety admission gate, wired into the ancestry decision (PLAN §2.7, §4.2).

:mod:`fusion_v2.safety` implements the science: a cumulative reference frozen at depth 0, an exact
window-aligned ``N_H^whole`` comparator, a calibrated threshold, and an admission conjunction that
fails closed.  This module is the only thing that connects it to a running cycle.

**Why that connection is load-bearing.**  Structural feasibility and immune safety answer different
questions.  A design can fold perfectly and still open a new epitope hotspot the reference never
had -- and because feedback ancestry is *recursive*, admitting one puts that hotspot into the
parent of everything downstream, where the cumulative ratchet can no longer see it as new.  So
ancestry eligibility is a CONJUNCTION, and a cycle that promoted on the structure verdict alone
would be running V2's mechanism with its safety half disconnected while still reporting the
"definitive" label that is supposed to mean both.

The gate is a required argument of the cycle, not an optional one.  An optional safety gate defaults
to "off", and a run with it off is indistinguishable in its outputs from a run with it on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..fusion_v2.errors import V2Error
from ..fusion_v2.identity import SafetyReferenceBinding
from ..fusion_v2.safety import (
    AdmissibilityVerdict,
    IncrementalInapplicable,
    LineageSafetyLedger,
    SearchAdmissionEvidence,
    SafetyAdmissionPolicy,
    SelfReferentialReference,
    WholeHotspotVerdict,
    apply_threshold,
    admissibility_evidence_digest,
    make_search_admission_evidence,
    make_admissibility_verdict,
    measure_cumulative,
    incremental_inapplicable,
    measure_incremental,
    validate_search_admission_evidence,
)

__all__ = ["V2AdmissionError", "SafetyGate", "EndpointAdmission", "admit_endpoint"]


class V2AdmissionError(V2Error):
    """The admission gate was wired wrongly -- a caller mistake, not a design verdict."""


@dataclass(frozen=True)
class SafetyGate:
    """One lineage's admission policy and its ratchet state, bound together.

    The two must come from the same place: :func:`~fusion_v2.safety.open_lineage_ledger` builds the
    ledger from a reference that already carries its policy, so a gate assembled from a policy and
    an unrelated ledger would apply one run's threshold to another run's reference.
    """

    policy: SafetyAdmissionPolicy
    ledger: LineageSafetyLedger

    def __post_init__(self) -> None:
        if not isinstance(self.policy, SafetyAdmissionPolicy):
            raise V2AdmissionError("policy must be a SafetyAdmissionPolicy")
        if not isinstance(self.ledger, LineageSafetyLedger):
            raise V2AdmissionError("ledger must be a LineageSafetyLedger")
        if self.ledger.cumulative_reference.policy is not self.policy:
            raise V2AdmissionError(
                "the ledger's cumulative reference was bound under a DIFFERENT admission policy; "
                "carrying the reference by object identity is what makes the ratchet inescapable, "
                "and pairing it with another policy would re-base the threshold it was calibrated "
                "against"
            )
        # NOTE: an enabled incremental gate at depth 0 is NOT a misconfiguration.  This used to
        # raise, which made every legal config carrying ``incremental_gate_enabled: true``
        # unlaunchable -- the ledger a depth-0 gate is built from always has
        # ``immediate_parent=None``, and there is no way to advance it before depth 0 exists.  The
        # gate simply has no referent yet; ``admit_endpoint`` records that as a typed
        # ``IncrementalInapplicable`` fact and the gate becomes binding from depth 1.

    @property
    def reference_binding(self) -> SafetyReferenceBinding:
        """The depth-0 ``SafetyReferenceBinding`` a state stamps and an artifact records.

        Read off the reference that DECIDES ancestry rather than built beside it (Interface Map §3
        C5).  ``LivePartialState.safety_reference`` and ``CumulativeSafetyReference`` are one
        concept split by layer -- identity on the state, identity plus the live Head score here --
        so a second construction of the identity is a second reference that can disagree with the
        one every design was actually measured against, while both stay individually well-formed.
        """
        return self.ledger.cumulative_reference.binding


@dataclass(frozen=True)
class EndpointAdmission:
    """One endpoint's admission decision plus the evidence behind it."""

    endpoint_id: str
    admitted: bool
    verdict: AdmissibilityVerdict | None
    reason: str
    search_admitted: bool = False
    search_reason: str = "not_search_admitted"
    admission_evidence_digest: str | None = None
    structure_ancestry_passed: bool = False
    search_evidence: SearchAdmissionEvidence | None = None

    def __post_init__(self) -> None:
        """Keep the decision, typed verdict, and positive capability as one atomic fact."""
        if not isinstance(self.admitted, bool) or not isinstance(self.search_admitted, bool):
            raise V2AdmissionError("admitted and search_admitted must be explicit bools")
        if not isinstance(self.structure_ancestry_passed, bool):
            raise V2AdmissionError("structure_ancestry_passed must be an explicit bool")
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id.strip():
            raise V2AdmissionError("endpoint_id must be a non-empty string")
        if self.verdict is None:
            if (
                self.admitted or self.search_admitted
                or self.admission_evidence_digest is not None
                or self.search_evidence is not None
            ):
                raise V2AdmissionError(
                    "an admission without a verdict cannot admit, authorize search, or carry a "
                    "verdict digest"
                )
            return
        if not isinstance(self.verdict, AdmissibilityVerdict):
            raise V2AdmissionError("verdict must be an AdmissibilityVerdict or None")
        if self.endpoint_id != self.verdict.endpoint_id:
            raise V2AdmissionError("EndpointAdmission endpoint does not match its verdict")
        if self.admitted is not self.verdict.admitted:
            raise V2AdmissionError("EndpointAdmission.admitted disagrees with its verdict")

        incremental = self.verdict.incremental
        incremental_passed = (
            incremental is None
            or isinstance(incremental, IncrementalInapplicable)
            or (isinstance(incremental, WholeHotspotVerdict) and incremental.passed)
        )
        immune_constraints_passed = bool(
            self.verdict.constraints_preserved
            and self.verdict.cumulative.passed
            and incremental_passed
        )
        expected_search = bool(
            self.structure_ancestry_passed and immune_constraints_passed
        )
        if self.search_admitted is not expected_search:
            raise V2AdmissionError(
                "EndpointAdmission.search_admitted disagrees with structure, immune, or anchor "
                "evidence"
            )
        expected_digest = admissibility_evidence_digest(
            self.verdict, structure_ancestry=self.structure_ancestry_passed,
        )
        if self.admission_evidence_digest != expected_digest:
            raise V2AdmissionError(
                "EndpointAdmission evidence digest does not match its live verdict"
            )
        if not expected_search:
            if self.search_evidence is not None:
                raise V2AdmissionError(
                    "a failed search admission cannot carry positive search evidence"
                )
            return
        if self.search_evidence is None:
            raise V2AdmissionError(
                "search_admitted=true requires typed SearchAdmissionEvidence"
            )
        try:
            validate_search_admission_evidence(self.search_evidence)
        except V2Error as exc:
            raise V2AdmissionError("search evidence is invalid") from exc
        if self.search_evidence.verdict is not self.verdict:
            raise V2AdmissionError(
                "search evidence must carry the exact EndpointAdmission verdict object"
            )
        if (
            self.search_evidence.endpoint_id != self.endpoint_id
            or self.search_evidence.admission_evidence_digest != expected_digest
            or self.search_evidence.structure_ancestry_passed
               is not self.structure_ancestry_passed
        ):
            raise V2AdmissionError(
                "search evidence is not bound to this exact EndpointAdmission"
            )

    @property
    def structure_only(self) -> bool:
        """True when nothing but the structure verdict was available to decide on."""
        return self.verdict is None


def admit_endpoint(
    gate: SafetyGate,
    *,
    endpoint: Any,
    structure_definitive: bool,
    structure_ancestry: bool | None = None,
    structure_evidence_digest: str | None = None,
    endpoint_tokens: Sequence[int],
    hard_anchors: Sequence[tuple[int, int]],
) -> EndpointAdmission:
    """Decide whether one scored endpoint may become feedback ancestry.

    ``constraints_preserved`` is MEASURED here from the endpoint's own tokens against the source's
    hard anchors rather than asserted by the caller: it is one of the three conjuncts, and a conjunct
    supplied as a constant is not a gate.
    """
    if not isinstance(gate, SafetyGate):
        raise V2AdmissionError("gate must be a SafetyGate")
    if not isinstance(structure_definitive, bool):
        raise V2AdmissionError("structure_definitive must be an explicit bool")
    if structure_ancestry is not None and not isinstance(structure_ancestry, bool):
        raise V2AdmissionError("structure_ancestry must be an explicit bool or None")
    ancestry_passed = (
        structure_definitive if structure_ancestry is None else structure_ancestry
    )
    if structure_definitive and not ancestry_passed:
        raise V2AdmissionError("definitive structure pass must imply ancestry structure pass")
    tokens = tuple(int(token) for token in endpoint_tokens)
    constraints_preserved = all(
        tokens[int(position)] == int(token) for position, token in hard_anchors
    )

    try:
        cumulative_evidence = measure_cumulative(
            gate.ledger, endpoint.head_score, endpoint_id=endpoint.endpoint_id,
        )
    except SelfReferentialReference as exc:
        # The design IS the frozen reference.  ``N_H(y; y) == 0`` would make the gate vacuous, so
        # the comparator refuses to compute it -- and refusing ANCESTRY is the fail-closed reading:
        # feeding the reference back as its own parent teaches the lineage nothing, and the one
        # thing that must never happen is an unmeasured design acquiring the definitive label.
        return EndpointAdmission(
            endpoint_id=endpoint.endpoint_id, admitted=False, verdict=None,
            reason=f"design is the cumulative reference itself, so the gate cannot measure it: "
                   f"{exc}",
            search_admitted=False,
            search_reason="cumulative reference is self-referential",
            structure_ancestry_passed=ancestry_passed,
        )
    cumulative = apply_threshold(cumulative_evidence, gate.policy.cumulative_threshold)

    incremental = None
    if gate.policy.incremental_gate_enabled:
        if gate.ledger.immediate_parent is None:
            # Depth 0 has nothing to compare against.  Recorded as a typed FACT minted from this
            # ledger rather than left as ``None``, which would be indistinguishable from a gate
            # that was simply skipped -- and the marker cannot be forged for a lineage that does
            # have a parent, so it can never launder a skipped measurement.
            incremental = incremental_inapplicable(gate.ledger)
        else:
            incremental = apply_threshold(
                measure_incremental(gate.ledger, endpoint.head_score,
                                    endpoint_id=endpoint.endpoint_id),
                gate.policy.incremental_threshold,
            )

    verdict = make_admissibility_verdict(
        policy=gate.policy, cumulative=cumulative, incremental=incremental,
        structure_definitive=bool(structure_definitive),
        structure_evidence_level="definitive" if structure_definitive else "unvalidated",
        constraints_preserved=bool(constraints_preserved),
    )
    incremental_passed = (
        incremental is None
        or isinstance(incremental, IncrementalInapplicable)
        or incremental.passed
    )
    immune_constraints_passed = bool(
        constraints_preserved and cumulative.passed and incremental_passed
    )
    search_admitted = bool(ancestry_passed and immune_constraints_passed)
    evidence_digest = admissibility_evidence_digest(
        verdict, structure_ancestry=ancestry_passed,
    )
    search_evidence = (
        make_search_admission_evidence(
            endpoint_id=endpoint.endpoint_id,
            verdict=verdict,
            structure_ancestry_passed=ancestry_passed,
            structure_evidence_digest=structure_evidence_digest,
        )
        if search_admitted else None
    )

    if verdict.admitted:
        reason = "admitted"
    else:
        failed = []
        if not structure_definitive:
            failed.append("structure not definitively feasible")
        if not constraints_preserved:
            failed.append("hard anchors not preserved")
        if not cumulative.passed:
            failed.append(
                f"whole-landscape new hotspot {cumulative.evidence.max_increase:.4f} exceeds the "
                f"cumulative threshold {cumulative.threshold.value:.4f}")
        if isinstance(incremental, WholeHotspotVerdict) and not incremental.passed:
            failed.append(
                f"incremental new hotspot {incremental.evidence.max_increase:.4f} exceeds "
                f"{incremental.threshold.value:.4f}")
        reason = "; ".join(failed)
    if search_admitted:
        search_reason = "search_admitted"
    else:
        search_failed = []
        if not ancestry_passed:
            search_failed.append("structure not ancestry-feasible")
        if not constraints_preserved:
            search_failed.append("hard anchors not preserved")
        if not cumulative.passed:
            search_failed.append("cumulative immune gate failed")
        if not incremental_passed:
            search_failed.append("incremental immune gate failed")
        search_reason = "; ".join(search_failed)
    return EndpointAdmission(
        endpoint_id=endpoint.endpoint_id, admitted=verdict.admitted, verdict=verdict,
        reason=reason, search_admitted=search_admitted, search_reason=search_reason,
        admission_evidence_digest=evidence_digest,
        structure_ancestry_passed=ancestry_passed,
        search_evidence=search_evidence,
    )
