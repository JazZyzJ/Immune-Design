"""V2F2: the support-scoring policy boundary (PLAN §2.5).

The projection **kernel** is frozen by the PLAN; the scientific **support-scoring policy** is not.
This module owns the contract between them, and nothing else.

Three design decisions are load-bearing and deliberate:

1. **A policy result carries RAW TUPLES, never a constructed** :class:`~fusion_v2.state.SupportPartition`.
   ``SupportPartition.__post_init__`` refuses to construct when ``write_from_endpoint`` or
   ``reopen`` is empty, so a partition-typed result would turn an invalid policy answer into an
   *exception*.  PLAN §2.4 requires the opposite: "an invalid policy result creates an explicit
   null/stalled event rather than an invented state".  Validation is therefore the kernel's job,
   and this module stays a dumb, honest carrier.

2. **This module never imports** :mod:`fusion_v2.config`.  PLAN §2.5: "No uncertainty threshold,
   Head-window projection rule, reopen fraction, or protected fraction is authorized until it is
   explicitly frozen in config/runbook provenance."  A policy that cannot reach the run config
   cannot quietly read an unfrozen threshold out of it.  The restriction is structural, not
   conventional, and :mod:`tests.inverse_folding.test_fusion_v2_projection` asserts it by AST.

3. **The reason vocabulary is closed here, not in** :mod:`fusion_v2.state`.  ``SupportPartition``
   accepts free-form ``str`` reasons and V2F1 is frozen, so this module supplies the enum and the
   kernel serialises it to the string the state layer stores.  Telemetry stays comparable across
   runs without reopening a frozen type.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .errors import V2Error
from .evidence import (
    AlignedWindowEvidence,
    LeaveOneOutContribution,
    V2EvidenceError,
    build_window_evidence,
    score_leave_one_out,
)
from .identity import HeadEvaluatorIdentity, ProjectionPolicyIdentity, require_digest
from .reward import (
    DonorGateVerdict,
    LineageIncumbent,
    donor_gate,
)
from .schedule import (
    BandCenterRule,
    CycleCoordinates,
    EmptyBandIntersectionError,
    MissingScheduleBandError,
    ScheduleBandTable,
    admissible_reopen_cardinality,
    band_center_target,
    lookup_band,
    required_reopen_count,
)
from .state import CompleteEndpoint, LivePartialState

__all__ = [
    "V2PolicyError",
    "SupportReason",
    "PolicyDecision",
    "PolicyRejection",
    "PolicyResult",
    "PolicyRuntime",
    "FeedbackSupportPolicy",
    "ExplicitProbePolicy",
    "EXPLICIT_PROBE_POLICY_ID",
    "STATE_DERIVED_PROBE_POLICY_ID",
    "HEAD_DIRECTED_CAPPED_POLICY_ID",
    "SOURCE_GEOMETRY_CONTROL_POLICY_ID",
    "StateDerivedProbePolicy",
    "HeadDirectedCappedPolicy",
    "SourceGeometryControlPolicy",
    "HeadDirectedCalibration",
    "SourceView",
    "HeadDirectedDecisionEvidence",
    "WriteCandidateEvidence",
    "ReopenCandidateEvidence",
    "StallReason",
    "REOPEN_PRIORITY_LAW",
    "REOPEN_COUNT_LAW",
    "WRITE_WINDOW_RULE",
    "DIAGNOSTIC_POLICY_ID_PREFIX",
    "DIAGNOSTIC_POLICY_IDS",
    "DIAGNOSTIC_ALLOWED_PHASES",
    "policy_id_is_diagnostic",
    "assert_identity_vocabulary",
    "DeclaredPolicy",
]


class V2PolicyError(V2Error):
    """A policy contract violation.

    Note this is *not* how a policy declines: declining is a :class:`PolicyRejection` value.  This
    exception is reserved for a malformed policy object, e.g. non-integer positions.
    """


# --------------------------------------------------------------------------------------------
# The diagnostic vocabulary.  ONE definition, in the module that owns policy identity.
# --------------------------------------------------------------------------------------------
#
# This used to be two: ``config.DIAGNOSTIC_VALUE_PREFIX = "diag_"`` described what a diagnostic
# policy id looks like, while this module named its own probe ``explicit_probe`` -- no prefix.  The
# two were never compared, so the config's notion of "diagnostic" could not see the only diagnostic
# policy that exists.  Declaring the probe HONESTLY (``support_policy_id: "explicit_probe"``,
# ``support_policy_is_diagnostic: false``) satisfied the prefix/flag consistency check and walked
# past the phase gate, and the probe then drove real transitions in a production phase.
#
# The vocabulary therefore lives here, next to the policies it classifies, and ``config`` imports
# it rather than restating it.  Two rules, deliberately overlapping in the FAIL-CLOSED direction:
#
# * registration -- adding a diagnostic policy is an authority edit to ``DIAGNOSTIC_POLICY_IDS``,
#   never a config key that self-authorizes; and
# * the prefix -- an id that advertises itself as diagnostic is treated as diagnostic even if
#   nobody registered it.
#
# Neither rule can ever make a diagnostic policy look like a production one, which is the only
# direction that matters.

#: Ids that advertise diagnostic status in their own name.  Sufficient, never necessary.
DIAGNOSTIC_POLICY_ID_PREFIX = "diag_"

#: The predeclared-sets probe PLAN §2.5 admits for "deterministic tests and the first
#: state-transition diagnostic".  It is what :class:`ExplicitProbePolicy` reports as its own
#: ``policy_id``, so a config that names it is naming the object that actually answers.
EXPLICIT_PROBE_POLICY_ID = "explicit_probe"

#: The state-derived probe the V2 runbook freezes for the first state-transition diagnostic.
#: Predeclared sets cannot survive a realized state (see :class:`StateDerivedProbePolicy`), so this
#: is what the canary actually runs.
STATE_DERIVED_PROBE_POLICY_ID = "state_derived_probe"

#: V2F5A's minimal Head-directed production policy.  NOT diagnostic: it is the first support law
#: whose write and reopen identities carry declared local Head evidence, and it is meant to drive
#: real transitions in the ``policy_qualification`` phase.
HEAD_DIRECTED_CAPPED_POLICY_ID = "head_directed_capped"

#: Its matched control (PLAN §2.5, §8.4): the same donor, the same realized cardinalities, the same
#: band and seeds -- positions chosen by the FORMER source-geometry law instead.  Also not
#: diagnostic: a control arm of a scientific comparison is as production as its treatment.
SOURCE_GEOMETRY_CONTROL_POLICY_ID = "source_geometry_control"

#: The registry.  Widening it is an authority edit here, in code, under review.
DIAGNOSTIC_POLICY_IDS = frozenset({EXPLICIT_PROBE_POLICY_ID, STATE_DERIVED_PROBE_POLICY_ID})

#: PLAN §2.5 admits a diagnostic policy for "deterministic tests and the first state-transition
#: diagnostic" only.  Widening this is likewise an authority edit, never a config key.
DIAGNOSTIC_ALLOWED_PHASES = frozenset({"state_transition_canary"})


def policy_id_is_diagnostic(policy_id: str) -> bool:
    """Whether ``policy_id`` names a diagnostic-only policy, per the vocabulary above.

    The single authority both the config loader and the projection kernel consult, so "diagnostic"
    cannot mean one thing at parse time and another when a policy answers.
    """
    if isinstance(policy_id, bool) or not isinstance(policy_id, str) or not policy_id.strip():
        raise V2PolicyError(f"policy_id must be a non-empty str, got {policy_id!r}")
    return policy_id in DIAGNOSTIC_POLICY_IDS or policy_id.startswith(DIAGNOSTIC_POLICY_ID_PREFIX)


def assert_identity_vocabulary(identity: ProjectionPolicyIdentity) -> None:
    """Require a policy's self-reported ``is_diagnostic_only`` to agree with its own id.

    ``ProjectionPolicyIdentity`` carries the flag and the id independently, and nothing compared
    them.  A probe could therefore keep its registered diagnostic id while reporting itself
    production, and every gate keyed on the flag would wave it through; a production policy could
    equally claim the flag and lose its transitions to the diagnostic gate for no reason.

    Equality, not implication: a policy that calls itself diagnostic must be REGISTERED (or carry
    the prefix), so "there is a diagnostic policy in this run" is always visible to the config
    loader and can never be a private fact of one object.
    """
    if not isinstance(identity, ProjectionPolicyIdentity):
        raise V2PolicyError(
            f"policy identity must be a ProjectionPolicyIdentity, got {type(identity).__name__}")
    expected = policy_id_is_diagnostic(identity.policy_id)
    if identity.is_diagnostic_only is not expected:
        raise V2PolicyError(
            f"policy {identity.policy_id!r} reports is_diagnostic_only="
            f"{identity.is_diagnostic_only} but the policy vocabulary says {expected}; a policy "
            "that calls itself diagnostic must be registered in DIAGNOSTIC_POLICY_IDS (or carry "
            f"the {DIAGNOSTIC_POLICY_ID_PREFIX!r} prefix) so the config gate can see it, and a "
            "registered diagnostic policy may not relabel itself production (PLAN §2.5)"
        )


@dataclass(frozen=True)
class DeclaredPolicy:
    """What the RUN declared its support policy to be -- the config's half of the gate.

    **Why this type exists at all.**  PLAN task V2F1 requires "diagnostic policy in a production
    phase" to fail *before model calls*, but the identity of the policy that actually ANSWERS does
    not exist until it answers -- after the source prefix and the K lookaheads have already run.
    The two halves are therefore split, and neither is sufficient alone:

    * **Parse time** (this type, built by ``config.V2Config.declared_policy``): the DECLARED id and
      flag must agree with the policy vocabulary, and a diagnostic declaration is unconstructible
      outside :data:`DIAGNOSTIC_ALLOWED_PHASES`.  That is the check PLAN V2F1 asks for, and it runs
      with nothing loaded.
    * **Runtime** (``projection.source_writeback``): the ANSWERING identity must equal this
      declaration.

    Composed, they give the property the PLAN is really after: an answer that matches its run's
    declaration cannot be diagnostic in a production phase, because such a declaration cannot be
    constructed.  That is also why the kernel needs no phase logic of its own -- it never has to
    reason about phases, only about whether the answer is the declared object.

    ``phase`` is validated against ``config.RUN_PHASES`` by the config loader, not here: the run
    phase vocabulary is a config concept, and this module must never import the run config
    (PLAN §2.5 -- a policy that cannot reach the config cannot read an unfrozen threshold from it).
    """

    policy_id: str
    policy_version: str
    is_diagnostic: bool
    phase: str

    def __post_init__(self) -> None:
        for name in ("policy_id", "policy_version", "phase"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
                raise V2PolicyError(f"declared {name} must be a non-empty str, got {value!r}")
        if not isinstance(self.is_diagnostic, bool):
            raise V2PolicyError(
                "declared is_diagnostic must be an explicit bool; a coercible value is not a "
                "declaration (PLAN §2.5)"
            )
        expected = policy_id_is_diagnostic(self.policy_id)
        if self.is_diagnostic is not expected:
            raise V2PolicyError(
                f"declared policy {self.policy_id!r} says is_diagnostic={self.is_diagnostic} but "
                f"the policy vocabulary says {expected}; the declaration is checked against the "
                "vocabulary the runtime policies use, not against what the declaration says about "
                "itself (PLAN §2.5)"
            )
        if self.is_diagnostic and self.phase not in DIAGNOSTIC_ALLOWED_PHASES:
            raise V2PolicyError(
                f"diagnostic policy {self.policy_id!r} is not admissible in phase {self.phase!r}; "
                f"allowed phases are {sorted(DIAGNOSTIC_ALLOWED_PHASES)} and widening that requires "
                "an authority edit, never a config key (PLAN §2.5, §5.1)"
            )

    def assert_answered_by(self, identity: ProjectionPolicyIdentity) -> None:
        """Require ``identity`` -- the policy that actually answered -- to BE this declaration.

        Checked on a refusal as well as on a decision: a foreign policy's refusal is not this run's
        refusal either, and recording it would put "the policy declined" in the feedback-event
        table under a policy that was never asked.
        """
        assert_identity_vocabulary(identity)
        for name, declared, answered in (
            ("policy_id", self.policy_id, identity.policy_id),
            ("policy_version", self.policy_version, identity.policy_version),
            ("is_diagnostic", self.is_diagnostic, identity.is_diagnostic_only),
        ):
            if declared != answered:
                raise V2PolicyError(
                    f"the answering policy's {name}={answered!r} is not the run's declared "
                    f"{declared!r}; the transition would be produced by a policy this run never "
                    "declared, and every artifact would name the declared one (PLAN §2.5, §5.3)"
                )


class SupportReason(str, enum.Enum):
    """Per-position reason evidence (PLAN §2.5, "per-position reason evidence").

    Deliberately descriptive rather than causal.  PLAN §2.5: "Do not call an identity
    ``causal-improvement`` without a positional intervention; use ``improvement-associated`` for
    evidence derived from endpoint/reference comparison alone."
    """

    #: The endpoint identity is associated with an improvement against the frozen reference.
    IMPROVEMENT_ASSOCIATED = "improvement_associated"
    #: The source identity was committed at or after the re-entry boundary: future information.
    FUTURE_SOURCE_IDENTITY = "future_source_identity"
    #: The position carries enough active-history instability to be worth reopening.
    UNCERTAIN = "uncertain"
    #: The source token is temporally valid and was left alone.
    TEMPORALLY_VALID = "temporally_valid"
    #: The position was already unresolved in the source and stays that way.
    INHERITED_MASK = "inherited_mask"
    #: Sibling endpoints disagree at this position.
    SIBLING_DISAGREEMENT = "sibling_disagreement"
    #: The policy had no evidence and fell back to its null law.
    NO_EVIDENCE = "no_evidence"
    #: V2F5A write: reverting this donor identity to the incumbent's WORSENS the donor under the
    #: frozen Head (``a_i > 0``).  Deliberately not ``causal_improvement`` -- PLAN §2.5 reserves
    #: that name for a positional intervention and this is a model contribution in one context.
    FROZEN_HEAD_CONTRIBUTION = "frozen_head_contribution"
    #: V2F5A reopen: the position lies in a window that is a NEW hotspot against the immutable
    #: cumulative safety reference.
    NEW_HOTSPOT = "new_hotspot"
    #: V2F5A reopen: the position lies in a window that worsened from the incumbent to the donor.
    WORSENED_WINDOW = "worsened_window"
    #: V2F5A reopen: the donor still carries high absolute burden in a window covering it.
    RESIDUAL_BURDEN = "residual_burden"
    #: V2F5A reopen: the source's own commit history at this position is unsettled.
    TEMPORAL_INSTABILITY = "temporal_instability"
    #: The matched CONTROL's reason, and its whole point: the position was chosen by source mask
    #: geometry with no Head evidence consulted at all.
    SOURCE_GEOMETRY = "source_geometry"


def _positions(raw, name: str) -> tuple[int, ...]:
    """Normalise a support set: integers only, de-duplicated, sorted.

    Sorting here is what makes the kernel's output invariant to the row order a policy happens to
    emit (PLAN task V2F2: "same inputs/seed produce identical state regardless of row order").
    """
    out: set[int] = set()
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise V2PolicyError(f"{name} must contain integer positions, got {value!r}")
        if value < 0:
            raise V2PolicyError(f"{name} must contain non-negative positions, got {value}")
        out.add(int(value))
    return tuple(sorted(out))


def _reasons(raw: Mapping[int, SupportReason] | None, name: str) -> Mapping[int, SupportReason]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise V2PolicyError(f"{name} must be a mapping of position -> SupportReason")
    out: dict[int, SupportReason] = {}
    for position, reason in raw.items():
        if isinstance(position, bool) or not isinstance(position, int):
            raise V2PolicyError(f"{name} keys must be integer positions, got {position!r}")
        if not isinstance(reason, SupportReason):
            raise V2PolicyError(
                f"{name}[{position}] must be a SupportReason, got {reason!r}; the vocabulary is "
                "closed so telemetry stays comparable across runs"
            )
        out[int(position)] = reason
    # Sorted so the mapping's own iteration order cannot reach a digest.
    return {key: out[key] for key in sorted(out)}


@dataclass(frozen=True)
class PolicyDecision:
    """A policy's proposed support sets.

    **Unvalidated by construction.**  Disjointness, exhaustiveness, anchor exclusion, the temporal
    gate, and the mask-load band are all checked by the kernel, which converts a violation into a
    typed null event.  Constructing this record must never raise for a merely *wrong* proposal --
    only for a structurally malformed one.
    """

    write_from_endpoint: tuple[int, ...]
    inject_from_source_feedback: tuple[int, ...]
    reopen: tuple[int, ...]
    carry_from_source: tuple[int, ...]
    reason_by_pos: Mapping[int, SupportReason] = field(default_factory=dict)
    policy: ProjectionPolicyIdentity | None = None
    #: The policy's own per-position decision record (PLAN §5.3, V2F5A: "per-position decision
    #: evidence for every accepted and rejected write/reopen candidate").  Carried on the DECISION
    #: rather than pushed through the kernel because the kernel's ``SupportPartition`` is a frozen
    #: V2F1 type whose reason vocabulary is one string per position: a policy that had to compress
    #: ``a_i``, its rank and its rejection reason into that string would be writing an artifact
    #: nobody can join back to the Head that produced it.  The kernel neither reads nor validates
    #: this; the runner carries it to the feedback-event row.
    decision_evidence: Any = None

    def __post_init__(self) -> None:
        for name in ("write_from_endpoint", "inject_from_source_feedback", "reopen",
                     "carry_from_source"):
            object.__setattr__(self, name, _positions(getattr(self, name), name))
        object.__setattr__(self, "reason_by_pos", _reasons(self.reason_by_pos, "reason_by_pos"))
        if self.policy is not None and not isinstance(self.policy, ProjectionPolicyIdentity):
            raise V2PolicyError("policy must be a ProjectionPolicyIdentity")

    @property
    def claimed_positions(self) -> tuple[int, ...]:
        """Every position the policy named, with duplicates preserved across sets.

        The kernel uses the multiset length against the set length to detect overlap.
        """
        return (self.write_from_endpoint + self.inject_from_source_feedback
                + self.reopen + self.carry_from_source)


@dataclass(frozen=True)
class PolicyRejection:
    """A policy declining to act, as a VALUE rather than an exception.

    PLAN §4.5 requires typed null behaviour; a policy that cannot find an admissible support set
    must be able to say so without aborting the run.
    """

    reason: str
    reason_by_pos: Mapping[int, SupportReason] = field(default_factory=dict)
    policy: ProjectionPolicyIdentity | None = None
    #: PLAN §2.5/V2F5A: "A typed stall must preserve the rejected candidate counts and rejection
    #: reasons rather than emitting a reward-blind fallback event."  A stall with no evidence is
    #: indistinguishable from a stall nobody looked into.
    decision_evidence: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise V2PolicyError("a rejection must carry a non-empty human-readable reason")
        object.__setattr__(self, "reason_by_pos", _reasons(self.reason_by_pos, "reason_by_pos"))
        if self.policy is not None and not isinstance(self.policy, ProjectionPolicyIdentity):
            raise V2PolicyError("policy must be a ProjectionPolicyIdentity")


#: What a policy hands the kernel.
PolicyResult = PolicyDecision | PolicyRejection


@dataclass(frozen=True)
class PolicyRuntime:
    """The run-owned facilities a policy may need WHILE deciding, handed in at call time.

    Only one policy needs any of this: :class:`HeadDirectedCappedPolicy` scores leave-one-out
    counterfactuals through the frozen Head, and PLAN §5.4 requires every oracle request to be
    journaled before execution.  The meter is created per shard, long after the policy object is
    built by the oracle factory, so it cannot be a constructor field -- and a policy that quietly
    called the Head off-ledger would spend real GPU work that the run's own cap accounting cannot
    see.

    Passed only to policies that advertise ``consumes_runtime``; every existing policy keeps the
    three-argument call shape the kernel has always used.
    """

    cost_meter: Any = None
    event_prefix: str = ""


@runtime_checkable
class FeedbackSupportPolicy(Protocol):
    """The interface PLAN §2.5 requires: typed support sets plus reason evidence.

    A production policy will additionally have to satisfy the frozen ``FeedbackSupportPolicySpec``
    of PLAN §2.5 (residual Head window-to-residue attribution, conflict priority, maturity-band
    compatibility, and a source-off comparator).  That spec is not authorised yet, so this
    Protocol stays minimal on purpose: widening it now would bless an unfrozen policy shape.
    """

    def identity(self) -> ProjectionPolicyIdentity:
        """Content identity of this policy, for provenance and the diagnostic-phase gate."""

    def decide(self, *, source: LivePartialState, endpoint: CompleteEndpoint,
               coordinates: CycleCoordinates) -> PolicyResult:
        """Propose support sets for one projection, or decline."""

    def __call__(self, source: LivePartialState, endpoint: CompleteEndpoint,
                 coordinates: CycleCoordinates) -> PolicyResult:
        """The shape ``run_one_cycle`` actually invokes: positional, and callable.

        The kernel types its ``support_policy`` parameter as ``Callable[..., Any]`` and calls it
        directly.  This Protocol declared only ``decide``, so an implementation could satisfy the
        declared interface and still be unusable by the only thing that consumes it -- which is
        what happened: every cycle-level test passes a bare function or a fake with ``__call__``,
        so the divergence was invisible until the real policy reached a real projection on the
        cluster and raised ``'StateDerivedProbePolicy' object is not callable``.  Declaring both
        here makes "implements the Protocol" and "the kernel can use it" the same statement.
        """


@dataclass(frozen=True)
class ExplicitProbePolicy:
    """A predeclared-sets policy for deterministic tests and the first state-transition diagnostic.

    PLAN §2.5 permits exactly this and constrains it: "it must be labeled diagnostic-only and may
    not become a silent production default".

    ``is_diagnostic_only`` is hard-wired ``True`` here, but a self-label is only half of it, and
    the half that cannot be trusted on its own -- this class is not the only object that can build
    a :class:`~fusion_v2.identity.ProjectionPolicyIdentity`.  The label is load-bearing because
    :func:`policy_id_is_diagnostic` classifies :data:`EXPLICIT_PROBE_POLICY_ID` independently, so:

    * ``config`` refuses to DECLARE this id outside :data:`DIAGNOSTIC_ALLOWED_PHASES`, before any
      model or file access; and
    * ``projection.source_writeback`` refuses to let it ANSWER unless the run declared it, which
      by construction it can only have done in an allowed phase.

    It ignores ``source`` and ``endpoint`` by design: it is a probe, not a scientific policy.  That
    is precisely why the kernel -- not the policy -- owns every invariant.
    """

    write_from_endpoint: tuple[int, ...]
    inject_from_source_feedback: tuple[int, ...]
    reopen: tuple[int, ...]
    carry_from_source: tuple[int, ...]
    reason_by_pos: Mapping[int, SupportReason] = field(default_factory=dict)
    policy_version: str = "v0"
    policy_config_digest: str | None = None
    policy_spec_digest: str | None = None

    def __post_init__(self) -> None:
        for name in ("write_from_endpoint", "inject_from_source_feedback", "reopen",
                     "carry_from_source"):
            object.__setattr__(self, name, _positions(getattr(self, name), name))
        object.__setattr__(self, "reason_by_pos", _reasons(self.reason_by_pos, "reason_by_pos"))

    def identity(self) -> ProjectionPolicyIdentity:
        from .identity import canonical_digest

        digest = canonical_digest({
            "policy_id": EXPLICIT_PROBE_POLICY_ID,
            "policy_version": self.policy_version,
            "write_from_endpoint": list(self.write_from_endpoint),
            "inject_from_source_feedback": list(self.inject_from_source_feedback),
            "reopen": list(self.reopen),
            "carry_from_source": list(self.carry_from_source),
            "reason_by_pos": {str(k): v.value for k, v in sorted(self.reason_by_pos.items())},
        })
        return ProjectionPolicyIdentity(
            policy_id=EXPLICIT_PROBE_POLICY_ID,
            policy_version=self.policy_version,
            policy_config_digest=self.policy_config_digest or digest,
            policy_spec_digest=self.policy_spec_digest or digest,
            is_diagnostic_only=True,
        )

    def decide(self, *, source: LivePartialState, endpoint: CompleteEndpoint,
               coordinates: CycleCoordinates) -> PolicyResult:
        del source, endpoint, coordinates          # a probe replays what it was handed
        return PolicyDecision(
            write_from_endpoint=self.write_from_endpoint,
            inject_from_source_feedback=self.inject_from_source_feedback,
            reopen=self.reopen,
            carry_from_source=self.carry_from_source,
            reason_by_pos=self.reason_by_pos,
            policy=self.identity(),
        )

    def __call__(self, source: LivePartialState, endpoint: CompleteEndpoint,
                 coordinates: CycleCoordinates) -> PolicyResult:
        """``run_one_cycle`` calls the policy positionally; ``decide`` is the named contract."""
        return self.decide(source=source, endpoint=endpoint, coordinates=coordinates)


@dataclass(frozen=True)
class StateDerivedProbePolicy:
    """The diagnostic probe the V2 runbook freezes for the first state-transition canary.

    **Why a predeclared probe cannot do this job.**  :class:`ExplicitProbePolicy` replays fixed
    support sets, but the kernel requires ``reopen`` to name only source-RESOLVED positions and
    ``inject_from_source_feedback`` to name only resolved ones -- and which positions are resolved
    at ``c_d`` is stochastic.  A predeclared partition therefore returns
    ``null_invalid_policy_result`` on a realized state, so a canary configured with one pays a
    prefix, K lookaheads, a Head batch and K refolds per cycle and measures nothing.

    **The rule, frozen in the runbook and applied verbatim here.**

    * the lowest-indexed source-masked editable position becomes ``write_from_endpoint``;
    * every other inherited mask is carried;
    * resolved tokens committed strictly BEFORE ``r_d`` are carried;
    * resolved tokens committed AT OR AFTER ``r_d`` are injected as source feedback;
    * hard anchors never enter any set;
    * ``reopen`` cardinality is NOT a free parameter.  ``schedule`` already states it -- "reopen
      size is not a free parameter" -- because the coupled identity ``u_proj = u_src - a + b_new``
      pins ``b_new`` once ``r_d`` is chosen.  It is read from ``B(r_d)``, which is why this policy
      is instantiable only after a real ``rho_maturity_scan --mode step`` has produced one.

    ``reopen`` is drawn from the LATEST-committed resolved positions first.  Two reasons: the most
    recently committed identity is the one carrying the most future information relative to
    ``r_d``, and taking from that end preserves the older ``carry_from_source`` class -- which is
    the only class a fixed-support ablation (PLAN §2.6 intervention 2) can target, because masking
    a position in any other class makes the pinned partition illegal.

    Every way the rule fails to apply is a typed :class:`PolicyRejection`.  A quiet fallback --
    writing over a resolved position when no mask is available, or reopening fewer than the band
    demands -- would produce a transition under a partition nobody declared, and the artifact would
    name the declared one.

    Diagnostic-only, and registered as such: PLAN §2.5 forbids a diagnostic policy becoming "a
    silent production default", and the gate is keyed on the registry rather than on a config
    string.
    """

    band_table: ScheduleBandTable
    stratum_key: str
    #: SHA-256 of the FROZEN spec file (``configs/v2_state_derived_probe_policy_v1.json``), the
    #: run's ``projection_policy_spec`` content role.  Required and never derived: the kernel
    #: compares it against ``conditioning.projection_policy_spec``, so a locally computed stand-in
    #: would be a policy identity the run's own provenance does not describe.
    policy_spec_digest: str = ""
    policy_version: str = "v1"

    def __post_init__(self) -> None:
        if not isinstance(self.band_table, ScheduleBandTable):
            raise V2PolicyError(
                "band_table must be a content-verified ScheduleBandTable; the reopen cardinality "
                "is read from it, so an unverified table would set the one quantity PLAN §2.5 "
                "forbids leaving free"
            )
        if not isinstance(self.stratum_key, str) or not self.stratum_key.strip():
            raise V2PolicyError("stratum_key must be a non-empty str")
        require_digest(self.policy_spec_digest, "policy_spec_digest")

    def identity(self) -> ProjectionPolicyIdentity:
        from .identity import canonical_digest

        # SPEC and CONFIG are two different questions and were previously answered with one digest:
        #
        #   * the SPEC is the rule -- invariant across protein, stratum, r_d and calibration.  It is
        #     the sha256 of the frozen spec FILE, because that is what the config's
        #     ``projection_policy_spec`` content role signs and what the kernel matches against.  A
        #     canonical dict digest could never equal a file sha256, so the check could not pass;
        #     and folding the band/stratum in made the spec cell-specific, so no single frozen file
        #     could sign a multi-cell campaign like the four-cell Canary.
        #   * the CONFIG is the per-cell realized binding.  The band belongs HERE: two runs under
        #     different calibrations pinned different reopen cardinalities, so they ran the same
        #     rule under different configuration.
        return ProjectionPolicyIdentity(
            policy_id=STATE_DERIVED_PROBE_POLICY_ID,
            policy_version=self.policy_version,
            policy_config_digest=canonical_digest({
                "policy_id": STATE_DERIVED_PROBE_POLICY_ID,
                "policy_version": self.policy_version,
                "rule": "write=min_source_masked;carry=inherited_masks+commit_lt_r;"
                        "inject=commit_ge_r;reopen=band_pinned_latest_committed",
                "policy_spec_digest": self.policy_spec_digest,
                "stratum_key": self.stratum_key,
                "band_calibration_id": self.band_table.provenance.calibration_id,
                "band_calibration_digest":
                    self.band_table.provenance.calibration_content_digest,
            }),
            policy_spec_digest=self.policy_spec_digest,
            is_diagnostic_only=True,
        )

    def _decline(self, reason: str) -> PolicyRejection:
        return PolicyRejection(reason=reason, policy=self.identity())

    def decide(self, *, source: LivePartialState, endpoint: CompleteEndpoint,
               coordinates: CycleCoordinates) -> PolicyResult:
        del endpoint                      # the probe partitions the SOURCE; y enters via the write
        r_step = int(coordinates.r_step)
        editable = [int(p) for p in source.editable_positions]
        masked = [p for p in editable if source.tokens[p] == source.mask_token_id]
        resolved = [p for p in editable if source.tokens[p] != source.mask_token_id]

        if not masked:
            return self._decline(
                f"no source-masked write site at r_d={r_step}: every editable position is already "
                "resolved, and writing the endpoint over a resolved position would change the "
                "coupled mask-load 'a' term and move the admissible reopen envelope -- a different "
                "experiment wearing this one's name"
            )
        write = min(masked)

        try:
            band = lookup_band(self.band_table, step=r_step, stratum_key=self.stratum_key)
            load = admissible_reopen_cardinality(
                band=band, n_editable=len(editable), n_unresolved_source=len(masked),
                n_endpoint_writes_over_masked=1,
            )
        except (MissingScheduleBandError, EmptyBandIntersectionError) as exc:
            # An absent or empty envelope is a fact about the SCHEDULE, not a crash: the policy
            # declines so the cycle records a typed null the cohort can aggregate.
            return self._decline(f"B(r_d={r_step}) is unusable here: {exc}")
        if not load.feasible:
            return self._decline(
                f"B(r_d={r_step}) admits no reopen cardinality here: "
                f"{load.infeasible_reason or 'empty envelope'}"
            )
        # NOTE: "the band pins more reopens than there are resolved positions" needs no check
        # here.  ``admissible_reopen_cardinality`` already caps ``max_newly_masked`` at the number
        # of resolved editable positions and reports the empty interval itself, with a better
        # message than this layer could write.  A second check would be dead code that reads like
        # a live guard -- and mutation testing showed exactly that: neutering it changed nothing.

        def _commit_step(position: int) -> int:
            commit = source.active_commit_depth_step_by_pos[position]
            return -1 if commit is None else int(commit.step)

        # Latest-committed first: the most future-bearing identities relative to r_d, and it
        # preserves the older carry class the fixed-support ablation needs.
        ordered = sorted(resolved, key=lambda p: (-_commit_step(p), p))
        reopen = sorted(ordered[:load.min_newly_masked])
        reopened = set(reopen)

        inject = sorted(p for p in resolved
                        if p not in reopened and _commit_step(p) >= r_step)
        carry = sorted(
            [p for p in masked if p != write]
            + [p for p in resolved if p not in reopened and _commit_step(p) < r_step]
        )

        reasons: dict[int, SupportReason] = {write: SupportReason.IMPROVEMENT_ASSOCIATED}
        reasons.update({p: SupportReason.UNCERTAIN for p in reopen})
        reasons.update({p: SupportReason.FUTURE_SOURCE_IDENTITY for p in inject})
        reasons.update({
            p: (SupportReason.INHERITED_MASK if source.tokens[p] == source.mask_token_id
                else SupportReason.TEMPORALLY_VALID)
            for p in carry
        })
        return PolicyDecision(
            write_from_endpoint=(write,), inject_from_source_feedback=tuple(inject),
            reopen=tuple(reopen), carry_from_source=tuple(carry),
            reason_by_pos=reasons, policy=self.identity(),
        )

    def __call__(self, source: LivePartialState, endpoint: CompleteEndpoint,
                 coordinates: CycleCoordinates) -> PolicyResult:
        """``run_one_cycle`` calls the policy positionally; ``decide`` is the named contract."""
        return self.decide(source=source, endpoint=endpoint, coordinates=coordinates)


# --------------------------------------------------------------------------------------------
# V2F5A: the minimal Head-directed capped policy and its matched source-geometry control
# --------------------------------------------------------------------------------------------
#
# S7 qualified the TRANSPORT channel and falsified the SUPPORT law: the donor was chosen by exact
# complete Head rank, but the one written position came from source mask geometry, so the applied
# dose was uncorrelated with anything the Head said about that position (runbook §7.11, working note
# §7.5).  The two policies below change exactly one scientific variable -- support IDENTITY -- while
# the projection kernel, the substrate, the propagation horizon, the band and the descendant seeds
# stay byte-identical.  Everything richer (dose ladder, sibling consensus, block attribution,
# exploration, recursive depth) is out of scope by PLAN §0.4 and stays unimplemented rather than
# stubbed, so a placeholder can never be mistaken for evidence.


#: The frozen laws, named so an artifact records WHICH rule ran rather than that some rule did.
WRITE_WINDOW_RULE = "raw_aligned_window_improved_in_donor"
REOPEN_COUNT_LAW = "u_target_minus_u_src_plus_m_write"
REOPEN_PRIORITY_LAW = (
    "new_hotspot>worsened>residual_burden>active_uncertainty>temporal_instability>index"
)


class StallReason(str, enum.Enum):
    """Every way a V2F5A transition declines to act, as a closed vocabulary.

    PLAN §2.5 forbids a fallback: "no arbitrary-token fallback is allowed" and "If ``m_d=0``, it
    emits ``stall_no_positive_local_write``; it may not fall back to the diagnostic position or
    silently widen to a block rule."  A stall is therefore a first-class outcome with its own name,
    and the names are distinct because they call for different operator actions -- "no donor beat
    the incumbent" is a cohort fact, "no legal candidate exists at this coordinate" is a schedule
    fact, and "candidates existed but none contributed" is a policy fact.
    """

    #: PLAN A.2: no compatible endpoint improves the lineage incumbent by more than ``epsilon_R``.
    NO_BETTER_DONOR = "stall_no_better_donor"
    #: No editable, non-anchor, source-unresolved position differs from the incumbent inside an
    #: improved aligned window: there was nothing to ask the Head about.
    NO_LEGAL_WRITE_CANDIDATE = "stall_no_legal_write_candidate"
    #: PLAN §2.5's named stall: candidates existed, none had positive frozen-Head contribution.
    NO_POSITIVE_LOCAL_WRITE = "stall_no_positive_local_write"
    #: The exact band solver admits no write count that reproduces ``u_target`` from this source.
    BAND_INFEASIBLE = "stall_band_infeasible"
    #: The required reopen count exceeds the legal source-resolved support.
    REOPEN_INFEASIBLE = "stall_reopen_infeasible"
    #: The control could not match the treatment's realized cardinalities.
    CONTROL_CARDINALITY_UNMATCHABLE = "stall_control_cardinality_unmatchable"
    #: More legal candidates than the run declared it would pay Head calls for.  Failing closed
    #: keeps the projected budget a real bound; the offline replay measures how often it fires
    #: before anything launches.
    COUNTERFACTUAL_BUDGET_EXCEEDED = "stall_counterfactual_budget_exceeded"


@dataclass(frozen=True)
class SourceView:
    """Every fact about a live source state the V2F5A support law reads -- and nothing else.

    Two callers must apply ONE law: the live cycle, which holds a fully validated
    :class:`~fusion_v2.state.LivePartialState`, and the offline replay, which holds parquet rows
    from a finished run and can reconstruct the state's per-position vectors but not its
    conditioning identity, replay stream or provenance objects.  Making the law take this view
    rather than the state means the replay executes the same code instead of a second
    implementation whose coverage numbers would describe a policy that never ran.

    Deliberately NOT a general state abstraction: it carries the six vectors the law consults, so
    anything the law starts reading has to be added here in the open.
    """

    editable_positions: tuple[int, ...]
    masked_positions: tuple[int, ...]
    resolved_positions: tuple[int, ...]
    commit_step_by_pos: Mapping[int, int | None]
    active_sampler_score_by_pos: Mapping[int, float | None]
    n_origin_events_by_pos: Mapping[int, int]
    #: Positions whose temporary protection is still ACTIVE.  Empty on every live state the runner
    #: produces -- protection expires at the captured checkpoint before the state is exposed -- and
    #: kept because "reopen excludes active temporary protection" is a PLAN requirement that must
    #: not quietly stop holding if a future state layer carries live protection forward.
    protected_positions: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        for name in ("editable_positions", "masked_positions", "resolved_positions"):
            object.__setattr__(self, name, _positions(getattr(self, name), name))
        editable = set(self.editable_positions)
        masked, resolved = set(self.masked_positions), set(self.resolved_positions)
        if masked & resolved:
            raise V2PolicyError(
                f"positions {sorted(masked & resolved)} are both masked and resolved in the source "
                "view; the law partitions the editable domain on exactly this distinction"
            )
        if masked | resolved != editable:
            raise V2PolicyError(
                "masked and resolved positions must partition the editable domain; got "
                f"{sorted((masked | resolved) ^ editable)} outside it"
            )
        object.__setattr__(self, "protected_positions", frozenset(
            int(position) for position in self.protected_positions))

    @property
    def masked(self) -> frozenset[int]:
        return frozenset(self.masked_positions)

    @property
    def n_editable(self) -> int:
        return len(self.editable_positions)

    def commit_step(self, position: int) -> int:
        """The step this identity was committed at; ``-1`` when it has no commit event.

        ``-1`` is not a sentinel for "missing evidence" in a ranking -- it is used only by the
        temporal gate, where "no commit" and "committed before ``r_d``" get the same answer because
        an unresolved position introduces no future information either way.
        """
        commit = self.commit_step_by_pos.get(int(position))
        return -1 if commit is None else int(commit)

    @classmethod
    def of(cls, source: LivePartialState) -> "SourceView":
        editable = tuple(int(p) for p in source.editable_positions)
        masked = tuple(p for p in editable if source.tokens[p] == source.mask_token_id)
        resolved = tuple(p for p in editable if source.tokens[p] != source.mask_token_id)
        commits = source.active_commit_depth_step_by_pos
        return cls(
            editable_positions=editable, masked_positions=masked, resolved_positions=resolved,
            commit_step_by_pos={
                p: (None if commits[p] is None else int(commits[p].step)) for p in editable},
            active_sampler_score_by_pos={
                p: source.active_sampler_score_by_pos[p] for p in editable},
            n_origin_events_by_pos={
                p: int(source.provenance_by_pos[p].n_origin_events) for p in editable},
            protected_positions=frozenset(
                int(getattr(protection, "position", -1))
                for protection in getattr(source, "active_temporary_protection", ()) or ()
            ),
        )


@dataclass(frozen=True)
class HeadDirectedCalibration:
    """Every scientific number the Head-directed policy reads, with its provenance digest.

    Plain floats, not ``config.CalibratedScalar``: PLAN §2.5 forbids this module from importing the
    run config, so the config layer resolves each typed calibration artifact and hands the value
    down together with the ``source_ref`` that binds it.  Nothing here has a default -- omitting one
    is a ``TypeError`` at construction, which is the only way "no library defaults for scientific
    fields" can be structural rather than aspirational.
    """

    #: PLAN §2.5's frozen cap: ``m_cap = ceil(fraction * N_editable)``.  A CAP, never a quota.
    write_cap_editable_fraction: float
    write_cap_source_ref: str
    #: ``epsilon_R``: the donor-improvement margin, from the frozen Head's repeatability floor.
    epsilon_r: float
    epsilon_source_ref: str
    #: The local tolerance ``a_i`` must EXCEED to count as a positive contribution.
    local_contribution_tolerance: float
    local_contribution_source_ref: str
    #: The declared rounding/tie law for the integer band centre.
    band_center_rule: BandCenterRule
    #: The run's declared ceiling on the leave-one-out batch, in Head calls per cycle.  ENFORCED
    #: here, not merely projected: the preflight charges this number against ``max_head_calls``, and
    #: a policy that silently exceeded it would make the dry-run's verdict meaningless.
    max_counterfactual_head_calls_per_cycle: int

    def __post_init__(self) -> None:
        fraction = _finite_number(self.write_cap_editable_fraction, "write_cap_editable_fraction")
        if not 0.0 < fraction <= 1.0:
            raise V2PolicyError(
                f"write_cap_editable_fraction must lie in (0, 1], got {fraction}; a zero cap can "
                "never write and a cap above 1 is not a fraction of the editable domain"
            )
        epsilon = _finite_number(self.epsilon_r, "epsilon_r")
        if epsilon < 0.0:
            raise V2PolicyError(
                f"epsilon_r must be non-negative, got {epsilon}; a negative donor margin would "
                "admit a donor WORSE than the incumbent as an improvement"
            )
        tolerance = _finite_number(
            self.local_contribution_tolerance, "local_contribution_tolerance")
        if tolerance < 0.0:
            raise V2PolicyError(
                f"local_contribution_tolerance must be non-negative, got {tolerance}; a negative "
                "local tolerance would count a NEGATIVE contribution as positive evidence"
            )
        for name in ("write_cap_source_ref", "epsilon_source_ref",
                     "local_contribution_source_ref"):
            require_digest(getattr(self, name), name)
        if not isinstance(self.band_center_rule, BandCenterRule):
            raise V2PolicyError("band_center_rule must be a BandCenterRule")
        budget = self.max_counterfactual_head_calls_per_cycle
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise V2PolicyError(
                "max_counterfactual_head_calls_per_cycle must be a positive int; the leave-one-out "
                "batch is real Head work and an unbounded one makes --dry-run's cap verdict "
                "meaningless"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "write_cap_editable_fraction": float(self.write_cap_editable_fraction),
            "write_cap_source_ref": self.write_cap_source_ref,
            "epsilon_r": float(self.epsilon_r),
            "epsilon_source_ref": self.epsilon_source_ref,
            "local_contribution_tolerance": float(self.local_contribution_tolerance),
            "local_contribution_source_ref": self.local_contribution_source_ref,
            "band_center_rule": self.band_center_rule.value,
            "max_counterfactual_head_calls_per_cycle":
                int(self.max_counterfactual_head_calls_per_cycle),
        }


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2PolicyError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise V2PolicyError(f"{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class WriteCandidateEvidence:
    """One position the write selector considered, accepted or rejected.

    Rejected rows are kept.  PLAN V2F5A requires "per-position decision evidence for every accepted
    and rejected write/reopen candidate", and without the rejections an artifact cannot distinguish
    "the Head found nothing here" from "the policy never looked".
    """

    position: int
    donor_residue: str
    incumbent_residue: str
    legal: bool
    #: ``None`` while the position was still legal; a closed-vocabulary string once it was dropped.
    rejection_reason: str | None
    #: The most-improved aligned window containing the position (``None`` if uncovered).
    min_window_delta: float | None
    #: ``a_i``, present only for positions that reached the counterfactual batch.
    contribution: float | None
    #: Rank among positive contributions, 0-based; ``None`` if it never ranked.
    selection_rank: int | None
    selected: bool

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "position": self.position, "donor_residue": self.donor_residue,
            "incumbent_residue": self.incumbent_residue, "legal": self.legal,
            "rejection_reason": self.rejection_reason,
            "min_window_delta": self.min_window_delta, "contribution": self.contribution,
            "selection_rank": self.selection_rank, "selected": self.selected,
        }


@dataclass(frozen=True)
class ReopenCandidateEvidence:
    """One source-resolved position the reopen selector considered, with its priority evidence.

    ``None`` means the evidence is ABSENT, and absent evidence sorts after present evidence rather
    than being replaced by a numeric sentinel (PLAN §2.5): a position no Head window covers is not
    a position with zero burden.
    """

    position: int
    new_hotspot: float | None
    worsening: float | None
    residual_burden: float | None
    active_sampler_score: float | None
    n_origin_events: int
    commit_step: int | None
    selection_rank: int | None
    selected: bool
    #: Which conjunct put it in front, for the artifact's "declared reward rationale".
    priority_reason: str

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "position": self.position, "new_hotspot": self.new_hotspot,
            "worsening": self.worsening, "residual_burden": self.residual_burden,
            "active_sampler_score": self.active_sampler_score,
            "n_origin_events": self.n_origin_events, "commit_step": self.commit_step,
            "selection_rank": self.selection_rank, "selected": self.selected,
            "priority_reason": self.priority_reason,
        }


@dataclass(frozen=True)
class HeadDirectedDecisionEvidence:
    """Everything needed to reconstruct WHY each support position was chosen (PLAN A.5, §5.3).

    Emitted on a stall as well as on a decision: a stall that preserved no candidate counts cannot
    be told apart from a policy that was never asked.
    """

    policy_id: str
    stall_reason: str | None
    donor_gate: DonorGateVerdict | None
    incumbent_id: str | None
    incumbent_sequence_md5: str | None
    safety_reference_sequence_md5: str | None
    donor_endpoint_id: str
    donor_sequence_md5: str
    #: Content identity of the raw aligned-window comparisons this decision was built on.
    incumbent_window_evidence_digest: str | None
    safety_window_evidence_digest: str | None
    calibration: Mapping[str, Any]
    n_editable: int
    n_unresolved_source: int
    u_target: int | None
    band_center_rule: str | None
    write_cap: int
    n_legal_write_candidates: int
    n_positive_contributions: int
    m_band_min: int | None
    m_band_max: int | None
    realized_writes: int
    required_reopen: int | None
    realized_reopen: int
    n_legal_reopen_candidates: int
    write_candidates: tuple[WriteCandidateEvidence, ...]
    reopen_candidates: tuple[ReopenCandidateEvidence, ...]
    head_calls: int
    #: Explicitly recorded so the CONTROL arm can prove it consulted no Head evidence at all.
    head_evidence_consulted: bool

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "stall_reason": self.stall_reason,
            "donor_gate": None if self.donor_gate is None else self.donor_gate.canonical_payload(),
            "incumbent_id": self.incumbent_id,
            "incumbent_sequence_md5": self.incumbent_sequence_md5,
            "safety_reference_sequence_md5": self.safety_reference_sequence_md5,
            "donor_endpoint_id": self.donor_endpoint_id,
            "donor_sequence_md5": self.donor_sequence_md5,
            "incumbent_window_evidence_digest": self.incumbent_window_evidence_digest,
            "safety_window_evidence_digest": self.safety_window_evidence_digest,
            "calibration": dict(self.calibration),
            "n_editable": self.n_editable,
            "n_unresolved_source": self.n_unresolved_source,
            "u_target": self.u_target,
            "band_center_rule": self.band_center_rule,
            "write_cap": self.write_cap,
            "n_legal_write_candidates": self.n_legal_write_candidates,
            "n_positive_contributions": self.n_positive_contributions,
            "m_band_min": self.m_band_min,
            "m_band_max": self.m_band_max,
            "realized_writes": self.realized_writes,
            "required_reopen": self.required_reopen,
            "realized_reopen": self.realized_reopen,
            "n_legal_reopen_candidates": self.n_legal_reopen_candidates,
            "write_candidates": [row.canonical_payload() for row in self.write_candidates],
            "reopen_candidates": [row.canonical_payload() for row in self.reopen_candidates],
            "head_calls": self.head_calls,
            "head_evidence_consulted": self.head_evidence_consulted,
        }


def _band_compatible_write_interval(
    *, u_target: int, n_unresolved_source: int, n_editable: int, n_legal_reopen: int,
) -> tuple[int, int]:
    r"""The closed interval of write counts for which a legal partition EXISTS.

    Every V2F5A write lands on a source-MASKED position, so ``a == m`` in the kernel's byte-level
    identity ``u_proj = u_src - a + b_new`` and the reopen count is pinned to
    ``m_reopen = u_target - u_src + m``.  Substituting that into the exact solver's own bounds turns
    four inequalities into an interval in ``m``:

    * ``m_reopen >= 1``            (the kernel refuses an empty reopen)  ->  ``m >= 1 - u_target + u_src``
    * ``m_reopen <= n_editable - u_src`` (only resolved positions can be newly masked)
      ->  ``m <= n_editable - u_target``
    * ``m_reopen <= n_legal_reopen``  ->  ``m <= n_legal_reopen + u_src - u_target``
    * ``m <= u_src``               (a write needs a masked position to land on)

    The band's own ``[pinned.min, pinned.max]`` bounds are automatically satisfied because
    ``u_target`` lies inside them by construction, which is exactly why the target is read from the
    band rather than chosen.  Returned as an interval rather than a single number so the caller can
    distinguish "the cap bound the write count" from "the schedule did".
    """
    lower = max(0, 1 - int(u_target) + int(n_unresolved_source))
    upper = min(
        int(n_unresolved_source),
        int(n_editable) - int(u_target),
        int(n_legal_reopen) + int(n_unresolved_source) - int(u_target),
    )
    return lower, upper


@dataclass(frozen=True)
class HeadDirectedCappedPolicy:
    """PLAN task V2F5A: the smallest auditable immune-directed support law.

    One cycle, three guarantees, and nothing else:

    * **donor direction** -- feedback may originate only from an exact, definitively feasible
      endpoint that beats the lineage incumbent by more than the calibrated ``epsilon_R``;
    * **applied-dose direction** -- every written identity carries positive identity-bound
      frozen-Head leave-one-out contribution evidence, and every reopened position carries declared
      residual/worsened/new-hotspot/uncertainty evidence; and
    * **fail closed** -- when either is unavailable the transition STALLS with a typed reason.  It
      never falls back to an arbitrary token, and it never widens to a block rule.

    The three references it binds are non-interchangeable by type: ``incumbent`` is a
    :class:`~fusion_v2.reward.LineageIncumbent`, ``safety_reference_score`` is the immutable
    depth-0 reference's own Head score, and the donor arrives per call.

    **What this class does NOT decide.**  The reopen CARDINALITY is not a policy choice -- the
    coupled identity pins it once ``r_d`` and the band centre are fixed -- and neither is the write
    cap, which is frozen calibration.  The policy chooses only WHICH positions occupy those counts,
    which is precisely the variable S7 left uncontrolled.
    """

    band_table: ScheduleBandTable
    stratum_key: str
    incumbent: LineageIncumbent
    #: The immutable cumulative safety reference's exact Head score (``ybar``), for the new-hotspot
    #: conjunct of the reopen priority.  Held separately from ``incumbent`` even when the depth-0
    #: rule makes the two the same sequence: the artifact must always say which reference each
    #: conjunct was measured against.
    safety_reference_score: Any
    evaluator: HeadEvaluatorIdentity
    window_grid_digest: str
    calibration: HeadDirectedCalibration
    #: ``scorer(protein_id, sequences) -> results``.  Injected so this module stays free of the
    #: runtime's request type, of torch, and of the cost journal.
    counterfactual_scorer: Any
    policy_spec_digest: str = ""
    policy_version: str = "v1"

    #: The kernel hands a ``PolicyRuntime`` only to policies that ask for one.
    consumes_runtime: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.band_table, ScheduleBandTable):
            raise V2PolicyError(
                "band_table must be a content-verified ScheduleBandTable; the integer band centre "
                "is read from it and an unverified table would pin the one quantity PLAN §2.5 "
                "forbids leaving free"
            )
        if not isinstance(self.stratum_key, str) or not self.stratum_key.strip():
            raise V2PolicyError("stratum_key must be a non-empty str")
        if not isinstance(self.incumbent, LineageIncumbent):
            raise V2PolicyError(
                "incumbent must be a bound LineageIncumbent; a bare sequence or score could not "
                "say which reference the donor gate compared against"
            )
        if not isinstance(self.evaluator, HeadEvaluatorIdentity):
            raise V2PolicyError("evaluator must be a HeadEvaluatorIdentity")
        if self.evaluator.digest() != self.incumbent.head_identity_digest:
            raise V2PolicyError(
                "the incumbent was scored by a different Head evaluator than this policy runs; "
                "every margin and every a_i would subtract two instruments"
            )
        reference_md5 = getattr(self.safety_reference_score, "sequence_md5", None)
        if reference_md5 != self.incumbent.safety_reference_sequence_md5:
            raise V2PolicyError(
                f"the supplied safety-reference Head score describes {reference_md5!r} but the "
                f"incumbent was bound against safety reference "
                f"{self.incumbent.safety_reference_sequence_md5!r}; the new-hotspot conjunct would "
                "be measured against a reference this lineage never froze"
            )
        if not isinstance(self.calibration, HeadDirectedCalibration):
            raise V2PolicyError("calibration must be a HeadDirectedCalibration")
        if self.counterfactual_scorer is None or not callable(self.counterfactual_scorer):
            raise V2PolicyError(
                "counterfactual_scorer must be callable: without the frozen-Head leave-one-out "
                "batch there is no applied-dose evidence, and a policy that wrote anyway would be "
                "the Head-blind law V2F5A exists to replace"
            )
        require_digest(self.policy_spec_digest, "policy_spec_digest")

    # -- identity ---------------------------------------------------------------------------

    def identity(self) -> ProjectionPolicyIdentity:
        from .identity import canonical_digest

        # SPEC = the frozen rule file (invariant across protein, band and calibration); CONFIG =
        # this cell's realized binding.  Same split as the state-derived probe, and for the same
        # reason: one frozen spec must be able to sign a multi-cell campaign whose cells pin
        # different bands, incumbents and thresholds.
        return ProjectionPolicyIdentity(
            policy_id=HEAD_DIRECTED_CAPPED_POLICY_ID,
            policy_version=self.policy_version,
            policy_config_digest=canonical_digest({
                "policy_id": HEAD_DIRECTED_CAPPED_POLICY_ID,
                "policy_version": self.policy_version,
                "write_window_rule": WRITE_WINDOW_RULE,
                "reopen_count_law": REOPEN_COUNT_LAW,
                "reopen_priority_law": REOPEN_PRIORITY_LAW,
                "policy_spec_digest": self.policy_spec_digest,
                "stratum_key": self.stratum_key,
                "band_calibration_id": self.band_table.provenance.calibration_id,
                "band_calibration_digest": self.band_table.provenance.calibration_content_digest,
                "calibration": self.calibration.canonical_payload(),
                "incumbent": self.incumbent.canonical_payload(),
                "head_identity_digest": self.evaluator.digest(),
                "window_grid_digest": self.window_grid_digest,
            }),
            policy_spec_digest=self.policy_spec_digest,
            is_diagnostic_only=False,
        )

    # -- the decision -----------------------------------------------------------------------

    def decide(self, *, source: LivePartialState, endpoint: CompleteEndpoint,
               coordinates: CycleCoordinates,
               runtime: PolicyRuntime | None = None) -> PolicyResult:
        """The kernel's entry point: read the law's inputs off a live state and apply it."""
        return self.select(source_view=SourceView.of(source), donor=endpoint,
                           r_step=int(coordinates.r_step), runtime=runtime)

    def select(self, *, source_view: "SourceView", donor: Any, r_step: int,
               runtime: PolicyRuntime | None = None) -> PolicyResult:
        """The law itself, over the FACTS it reads rather than over a live state object.

        Split out so the offline replay (``scripts/analysis/replay_v2_head_directed_policy.py``)
        executes this same code path.  A replay that re-implemented the selection would measure a
        second policy's coverage and report it as this one's -- exactly the class of error V2F5A
        exists to remove from the support law.
        """
        endpoint = donor
        r_step = int(r_step)
        view = source_view
        editable = list(view.editable_positions)
        n_editable = len(editable)
        masked = list(view.masked_positions)
        resolved = list(view.resolved_positions)
        donor_sequence = endpoint.sequence
        incumbent_sequence = self.incumbent.sequence

        def evidence(**over: Any) -> HeadDirectedDecisionEvidence:
            base = dict(
                policy_id=HEAD_DIRECTED_CAPPED_POLICY_ID, stall_reason=None, donor_gate=None,
                incumbent_id=self.incumbent.incumbent_id,
                incumbent_sequence_md5=self.incumbent.sequence_md5,
                safety_reference_sequence_md5=self.incumbent.safety_reference_sequence_md5,
                donor_endpoint_id=str(endpoint.endpoint_id),
                donor_sequence_md5=str(endpoint.sequence_md5),
                incumbent_window_evidence_digest=None, safety_window_evidence_digest=None,
                calibration=self.calibration.canonical_payload(),
                n_editable=n_editable, n_unresolved_source=len(masked), u_target=None,
                band_center_rule=self.calibration.band_center_rule.value,
                write_cap=self._write_cap(n_editable), n_legal_write_candidates=0,
                n_positive_contributions=0, m_band_min=None, m_band_max=None,
                realized_writes=0, required_reopen=None, realized_reopen=0,
                n_legal_reopen_candidates=len(resolved), write_candidates=(),
                reopen_candidates=(), head_calls=0, head_evidence_consulted=True,
            )
            base.update(over)
            return HeadDirectedDecisionEvidence(**base)

        def stall(reason: StallReason, detail: str, **over: Any) -> PolicyRejection:
            return PolicyRejection(
                reason=f"{reason.value}: {detail}", policy=self.identity(),
                decision_evidence=evidence(stall_reason=reason.value, **over),
            )

        # ---- 1. the donor gate ---------------------------------------------------------------
        gate = donor_gate(
            donor=endpoint, incumbent=self.incumbent,
            epsilon_r=self.calibration.epsilon_r,
            epsilon_source_ref=self.calibration.epsilon_source_ref,
        )
        if not gate.passed:
            return stall(
                StallReason.NO_BETTER_DONOR,
                f"donor {endpoint.endpoint_id} scores {gate.donor_global_risk:.6f} against "
                f"incumbent {gate.incumbent_global_risk:.6f} (margin {gate.margin:.6f}, required "
                f"> {gate.epsilon_r:.6f}); reason={gate.reason.value}. The lineage keeps its "
                "incumbent rather than adopting a donor whose advantage is inside the frozen "
                "Head's own noise floor",
                donor_gate=gate,
            )

        # ---- 2. raw aligned-window evidence, on ONE common scale ------------------------------
        try:
            incumbent_windows = build_window_evidence(
                donor_score=endpoint.head_score, reference_score=self.incumbent.head_score,
                evaluator=self.evaluator, reference_label="lineage_incumbent",
            )
            safety_windows = build_window_evidence(
                donor_score=endpoint.head_score, reference_score=self.safety_reference_score,
                evaluator=self.evaluator, reference_label="cumulative_safety_reference",
            )
        except V2EvidenceError as exc:
            # An unusable comparison is a fact about the evidence, not a crash: the cycle records a
            # typed null the cohort can aggregate.
            return stall(
                StallReason.NO_LEGAL_WRITE_CANDIDATE,
                f"the donor and its references do not align on one window grid: {exc}",
                donor_gate=gate,
            )

        # ---- 3. legal write candidates -------------------------------------------------------
        candidates: list[int] = []
        rows: dict[int, dict[str, Any]] = {}
        for position in editable:
            donor_residue = donor_sequence[position]
            incumbent_residue = incumbent_sequence[position]
            row: dict[str, Any] = {
                "position": position, "donor_residue": donor_residue,
                "incumbent_residue": incumbent_residue, "legal": False,
                "rejection_reason": None,
                "min_window_delta": incumbent_windows.min_delta_at(position),
                "contribution": None, "selection_rank": None, "selected": False,
            }
            if position not in view.masked:
                # PLAN A.3: a legal candidate is UNRESOLVED in the live source.  Writing over a
                # resolved position would move the ``a`` term of the coupled mask-load identity and
                # silently shift the admissible reopen envelope.
                row["rejection_reason"] = "source_resolved"
            elif donor_residue == incumbent_residue:
                row["rejection_reason"] = "identical_to_incumbent"
            elif not incumbent_windows.covered(position):
                row["rejection_reason"] = "no_window_coverage"
            elif not incumbent_windows.improves_at(position):
                row["rejection_reason"] = "no_improved_window"
            else:
                row["legal"] = True
                candidates.append(position)
            rows[position] = row

        if not candidates:
            return stall(
                StallReason.NO_LEGAL_WRITE_CANDIDATE,
                f"none of the {len(masked)} source-unresolved editable position(s) both differ "
                "from the incumbent and lie in an aligned Head window improved in the donor",
                donor_gate=gate,
                incumbent_window_evidence_digest=incumbent_windows.evidence_digest,
                safety_window_evidence_digest=safety_windows.evidence_digest,
                write_candidates=tuple(WriteCandidateEvidence(**row) for row in rows.values()),
            )

        # ---- 4. the frozen-Head leave-one-out counterfactual ---------------------------------
        budget = int(self.calibration.max_counterfactual_head_calls_per_cycle)
        if len(candidates) > budget:
            # Checked BEFORE the batch, so the refusal costs nothing.  Truncating instead would
            # change the science: ``a_i`` is compared across ALL legal candidates to take the exact
            # top ``m_d``, so a partial batch would select the best of an arbitrary subset while the
            # artifact still claimed the exact rule.
            return stall(
                StallReason.COUNTERFACTUAL_BUDGET_EXCEEDED,
                f"{len(candidates)} legal write candidate(s) would need {len(candidates)} Head "
                f"call(s), above the run's declared {budget} per cycle; the exact top-m_d rule "
                "needs every candidate scored, so the transition fails closed rather than ranking "
                "an arbitrary subset",
                donor_gate=gate,
                incumbent_window_evidence_digest=incumbent_windows.evidence_digest,
                safety_window_evidence_digest=safety_windows.evidence_digest,
                n_legal_write_candidates=len(candidates),
                write_candidates=tuple(WriteCandidateEvidence(**row) for row in rows.values()),
            )
        try:
            contributions = self._score_contributions(
                protein_id=endpoint.protein_id, donor_sequence=donor_sequence,
                donor_global_risk=float(endpoint.head_global_risk),
                incumbent_sequence=incumbent_sequence, positions=candidates, runtime=runtime,
            )
        except V2EvidenceError as exc:
            return stall(
                StallReason.NO_POSITIVE_LOCAL_WRITE,
                f"the frozen-Head counterfactual batch could not be bound to its candidates: "
                f"{exc}",
                donor_gate=gate,
                incumbent_window_evidence_digest=incumbent_windows.evidence_digest,
                safety_window_evidence_digest=safety_windows.evidence_digest,
                n_legal_write_candidates=len(candidates),
                write_candidates=tuple(WriteCandidateEvidence(**row) for row in rows.values()),
            )
        for position, contribution in contributions.items():
            rows[position]["contribution"] = contribution.contribution

        tolerance = float(self.calibration.local_contribution_tolerance)
        positive = [p for p in candidates
                    if contributions[p].contribution > tolerance]
        for position in candidates:
            if position not in positive:
                rows[position]["rejection_reason"] = "non_positive_contribution"

        # ---- 5. cap, band compatibility, and the realized write count ------------------------
        write_cap = self._write_cap(n_editable)
        try:
            centre = band_center_target(
                table=self.band_table, step=r_step, stratum_key=self.stratum_key,
                n_editable=n_editable, rule=self.calibration.band_center_rule,
            )
        except (MissingScheduleBandError, EmptyBandIntersectionError) as exc:
            return stall(
                StallReason.BAND_INFEASIBLE, f"B(r_d={r_step}) is unusable here: {exc}",
                donor_gate=gate,
                incumbent_window_evidence_digest=incumbent_windows.evidence_digest,
                safety_window_evidence_digest=safety_windows.evidence_digest,
                n_legal_write_candidates=len(candidates), n_positive_contributions=len(positive),
                head_calls=len(contributions),
                write_candidates=tuple(WriteCandidateEvidence(**row) for row in rows.values()),
            )

        common = dict(
            donor_gate=gate,
            incumbent_window_evidence_digest=incumbent_windows.evidence_digest,
            safety_window_evidence_digest=safety_windows.evidence_digest,
            n_legal_write_candidates=len(candidates), n_positive_contributions=len(positive),
            u_target=centre.u_target, head_calls=len(contributions),
        )
        band_min, band_max = _band_compatible_write_interval(
            u_target=centre.u_target, n_unresolved_source=len(masked), n_editable=n_editable,
            n_legal_reopen=len(resolved),
        )
        common.update(m_band_min=band_min, m_band_max=band_max)

        if not positive:
            return stall(
                StallReason.NO_POSITIVE_LOCAL_WRITE,
                f"all {len(candidates)} legal candidate(s) have frozen-Head contribution a_i <= "
                f"{tolerance:.6f}; the policy stalls rather than writing a token no local evidence "
                "supports",
                write_candidates=tuple(WriteCandidateEvidence(**row) for row in rows.values()),
                **common,
            )

        m_d = min(len(positive), write_cap, band_max)
        if m_d < max(1, band_min):
            return stall(
                StallReason.BAND_INFEASIBLE,
                f"the exact band solver needs between {band_min} and {band_max} endpoint write(s) "
                f"to reproduce u_target={centre.u_target} from u_src={len(masked)} with "
                f"{len(resolved)} legal reopen site(s), but only {len(positive)} positive "
                f"candidate(s) exist under a cap of {write_cap}",
                write_candidates=tuple(WriteCandidateEvidence(**row) for row in rows.values()),
                **common,
            )

        ranked = sorted(positive, key=lambda p: (-contributions[p].contribution, p))
        for rank, position in enumerate(ranked):
            rows[position]["selection_rank"] = rank
        write = sorted(ranked[:m_d])
        for position in write:
            rows[position]["selected"] = True
            rows[position]["rejection_reason"] = None
        for position in ranked[m_d:]:
            rows[position]["rejection_reason"] = "below_realized_write_count"

        # ---- 6. the exact reopen count, and Head-directed reopen identities -------------------
        m_reopen = required_reopen_count(
            u_target=centre.u_target, n_unresolved_source=len(masked), n_writes=m_d)
        common.update(realized_writes=m_d, required_reopen=m_reopen)
        if m_reopen < 1 or m_reopen > len(resolved):
            return stall(
                StallReason.REOPEN_INFEASIBLE,
                f"the reopen equation demands {m_reopen} position(s) (u_target={centre.u_target}, "
                f"u_src={len(masked)}, m_d={m_d}) but {len(resolved)} source-resolved editable "
                "position(s) are available",
                write_candidates=tuple(WriteCandidateEvidence(**row) for row in rows.values()),
                **common,
            )

        reopen_rows = self._reopen_priority(
            view=view, resolved=resolved, incumbent_windows=incumbent_windows,
            safety_windows=safety_windows,
        )
        reopen = sorted(row.position for row in reopen_rows[:m_reopen])
        reopen_rows = tuple(
            ReopenCandidateEvidence(
                **{**row.canonical_payload(), "selection_rank": rank,
                   "selected": row.position in set(reopen)})
            for rank, row in enumerate(reopen_rows)
        )

        # ---- 7. the rest of the partition, under the frozen temporal law ---------------------
        reopened = set(reopen)
        written = set(write)

        inject = sorted(p for p in resolved
                        if p not in reopened and view.commit_step(p) >= r_step)
        carry = sorted(
            [p for p in masked if p not in written]
            + [p for p in resolved if p not in reopened and view.commit_step(p) < r_step]
        )
        reasons: dict[int, SupportReason] = {
            p: SupportReason.FROZEN_HEAD_CONTRIBUTION for p in write}
        reasons.update({row.position: _REOPEN_REASONS[row.priority_reason]
                        for row in reopen_rows if row.selected})
        reasons.update({p: SupportReason.FUTURE_SOURCE_IDENTITY for p in inject})
        reasons.update({
            p: (SupportReason.INHERITED_MASK if p in view.masked
                else SupportReason.TEMPORALLY_VALID)
            for p in carry
        })
        return PolicyDecision(
            write_from_endpoint=tuple(write), inject_from_source_feedback=tuple(inject),
            reopen=tuple(reopen), carry_from_source=tuple(carry),
            reason_by_pos=reasons, policy=self.identity(),
            decision_evidence=evidence(
                write_candidates=tuple(WriteCandidateEvidence(**row) for row in rows.values()),
                reopen_candidates=reopen_rows, realized_reopen=len(reopen), **common,
            ),
        )

    def __call__(self, source: LivePartialState, endpoint: CompleteEndpoint,
                 coordinates: CycleCoordinates,
                 runtime: PolicyRuntime | None = None) -> PolicyResult:
        """``run_one_cycle`` calls the policy positionally; ``decide`` is the named contract."""
        return self.decide(source=source, endpoint=endpoint, coordinates=coordinates,
                           runtime=runtime)

    # -- internals ---------------------------------------------------------------------------

    def _write_cap(self, n_editable: int) -> int:
        r"""``m_cap = ceil(fraction * N_editable)`` over the EDITABLE domain (PLAN §2.5).

        Over the editable domain, not the sequence length: hard anchors are not writable, so a cap
        computed on length would authorize more writes than the domain contains and the "cap" would
        stop binding on a heavily constrained protein.
        """
        return int(math.ceil(float(self.calibration.write_cap_editable_fraction)
                             * int(n_editable)))

    def _score_contributions(
        self, *, protein_id: str, donor_sequence: str, donor_global_risk: float,
        incumbent_sequence: str, positions: Sequence[int], runtime: PolicyRuntime | None,
    ) -> dict[int, LeaveOneOutContribution]:
        scorer = self.counterfactual_scorer
        if runtime is not None and getattr(runtime, "cost_meter", None) is not None:
            # PLAN §5.4: journal before execution.  The policy's counterfactual batch is real Head
            # work on real GPUs; off-ledger it would spend against a cap the run cannot see.
            scorer = _metered_scorer(self.counterfactual_scorer, runtime)
        return score_leave_one_out(
            scorer=scorer, protein_id=protein_id, donor_sequence=donor_sequence,
            donor_global_risk=donor_global_risk, incumbent_sequence=incumbent_sequence,
            positions=positions, evaluator=self.evaluator,
            window_grid_digest=self.window_grid_digest,
        )

    def _reopen_priority(
        self, *, view: "SourceView", resolved: Sequence[int],
        incumbent_windows: AlignedWindowEvidence, safety_windows: AlignedWindowEvidence,
    ) -> list[ReopenCandidateEvidence]:
        """PLAN §2.5's frozen reopen order, applied to every legal candidate.

        ``new-hotspot > worsening > residual burden > active uncertainty > temporal instability >
        index``.  Head decides WHERE new generative freedom is needed; it is never asked to invent
        the replacement token -- the segment resamples it.

        Absent evidence sorts AFTER present evidence in each conjunct and is recorded as ``None``.
        A numeric sentinel would make "no window covers this position" indistinguishable from "this
        position carries zero burden", and the two are opposite facts about the Head's reach.
        """
        rows: list[ReopenCandidateEvidence] = []
        for position in resolved:
            if position in view.protected_positions:
                # Defensive: a LIVE state carries only EXPIRED protection (the segment expires it
                # at c_{d+1} before the capture is exposed), so this is empty on every state the
                # runner produces.  Kept because "reopen excludes active temporary protection" is a
                # PLAN requirement, and a future state layer that carried live protection forward
                # must not silently start reopening protected identities.
                continue
            new_hotspot = safety_windows.worsening_at(position) \
                if safety_windows.covered(position) else None
            worsening = incumbent_windows.worsening_at(position) \
                if incumbent_windows.covered(position) else None
            residual = incumbent_windows.residual_burden_at(position)
            commit = view.commit_step_by_pos.get(position)
            rows.append(ReopenCandidateEvidence(
                position=position, new_hotspot=new_hotspot, worsening=worsening,
                residual_burden=residual,
                active_sampler_score=view.active_sampler_score_by_pos.get(position),
                n_origin_events=int(view.n_origin_events_by_pos.get(position, 0)),
                commit_step=None if commit is None else int(commit),
                selection_rank=None, selected=False,
                priority_reason=_reopen_reason(new_hotspot, worsening, residual),
            ))
        return sorted(rows, key=_reopen_sort_key)


#: Which SupportReason each reopen priority conjunct maps to, so the kernel's per-position reason
#: string names the evidence that actually put the position in the set.
_REOPEN_REASONS = {
    "new_hotspot": SupportReason.NEW_HOTSPOT,
    "worsened_window": SupportReason.WORSENED_WINDOW,
    "residual_burden": SupportReason.RESIDUAL_BURDEN,
    "active_uncertainty": SupportReason.UNCERTAIN,
    "temporal_instability": SupportReason.TEMPORAL_INSTABILITY,
    "index": SupportReason.NO_EVIDENCE,
}


def _reopen_reason(new_hotspot: float | None, worsening: float | None,
                   residual: float | None) -> str:
    if new_hotspot:
        return "new_hotspot"
    if worsening:
        return "worsened_window"
    if residual is not None:
        return "residual_burden"
    return "index"


def _present(value: float | None) -> tuple[int, float]:
    """Sort helper: present evidence first, then by magnitude descending."""
    return (1, 0.0) if value is None else (0, -float(value))


def _reopen_sort_key(row: ReopenCandidateEvidence):
    return (
        _present(row.new_hotspot),
        _present(row.worsening),
        _present(row.residual_burden),
        # Lower token log-probability is MORE uncertain, so ascending; absent last.
        (1, 0.0) if row.active_sampler_score is None else (0, float(row.active_sampler_score)),
        -int(row.n_origin_events),
        int(row.position),
    )


def _metered_scorer(scorer: Any, runtime: PolicyRuntime) -> Any:
    """Wrap the injected scorer so its Head batch is journaled before it runs (PLAN §5.4)."""
    meter = runtime.cost_meter
    prefix = runtime.event_prefix or "policy"

    def metered(protein_id: str, sequences: Sequence[str]):
        from .identity import canonical_digest

        with meter.attempt(
            event_id=f"{prefix}:counterfactual",
            phase="head", request_kind="head_batch",
            request_digest=canonical_digest({"protein_id": str(protein_id),
                                             "sequences": list(sequences)}),
            head_calls=len(sequences),
        ) as receipt:
            results = scorer(protein_id, sequences)
            receipt.observe(physical_forwards=0, head_calls=len(sequences))
        return results

    return metered


@dataclass(frozen=True)
class SourceGeometryControlPolicy:
    """The matched control of PLAN §2.5/§8.4: the FORMER position law at the treatment's counts.

    The scientific treatment in V2F5A is support IDENTITY, so the control must differ in identity
    and in nothing else.  It receives the same donor, runs the same donor gate against the same
    incumbent, and is handed the treatment's REALIZED ``(m_write, m_reopen)`` -- then chooses:

    * writes: the lowest-indexed legal source-MASKED editable positions; and
    * reopens: the latest-committed legal source-RESOLVED editable positions,

    which is exactly the geometry :class:`StateDerivedProbePolicy` used, truncated to the treatment
    cardinalities.  No Head evidence is consulted anywhere, and the decision evidence records
    ``head_evidence_consulted=False`` so the artifact can PROVE the control was Head-blind rather
    than merely assert it.

    It refuses rather than improvises when the cardinalities cannot be met: a control that quietly
    reopened one position fewer would differ from its treatment in dose as well as in identity, and
    the contrast could no longer be attributed.
    """

    band_table: ScheduleBandTable
    stratum_key: str
    incumbent: LineageIncumbent
    evaluator: HeadEvaluatorIdentity
    calibration: HeadDirectedCalibration
    #: The treatment's realized cardinalities.  Required: a control that chose its own would be a
    #: second treatment.
    required_writes: int = 0
    required_reopens: int = 0
    policy_spec_digest: str = ""
    policy_version: str = "v1"

    def __post_init__(self) -> None:
        if not isinstance(self.band_table, ScheduleBandTable):
            raise V2PolicyError("band_table must be a content-verified ScheduleBandTable")
        if not isinstance(self.stratum_key, str) or not self.stratum_key.strip():
            raise V2PolicyError("stratum_key must be a non-empty str")
        if not isinstance(self.incumbent, LineageIncumbent):
            raise V2PolicyError("incumbent must be a bound LineageIncumbent")
        if not isinstance(self.evaluator, HeadEvaluatorIdentity):
            raise V2PolicyError("evaluator must be a HeadEvaluatorIdentity")
        if not isinstance(self.calibration, HeadDirectedCalibration):
            raise V2PolicyError("calibration must be a HeadDirectedCalibration")
        for name in ("required_writes", "required_reopens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise V2PolicyError(
                    f"{name} must be a positive int taken from the treatment arm's REALIZED "
                    f"support, got {value!r}; the control's whole contract is that it matches"
                )
        require_digest(self.policy_spec_digest, "policy_spec_digest")

    def identity(self) -> ProjectionPolicyIdentity:
        from .identity import canonical_digest

        return ProjectionPolicyIdentity(
            policy_id=SOURCE_GEOMETRY_CONTROL_POLICY_ID,
            policy_version=self.policy_version,
            policy_config_digest=canonical_digest({
                "policy_id": SOURCE_GEOMETRY_CONTROL_POLICY_ID,
                "policy_version": self.policy_version,
                "rule": "write=lowest_index_source_masked;reopen=latest_commit_source_resolved",
                "policy_spec_digest": self.policy_spec_digest,
                "stratum_key": self.stratum_key,
                "band_calibration_id": self.band_table.provenance.calibration_id,
                "band_calibration_digest": self.band_table.provenance.calibration_content_digest,
                "required_writes": int(self.required_writes),
                "required_reopens": int(self.required_reopens),
                "incumbent": self.incumbent.canonical_payload(),
                "calibration": self.calibration.canonical_payload(),
            }),
            policy_spec_digest=self.policy_spec_digest,
            is_diagnostic_only=False,
        )

    def decide(self, *, source: LivePartialState, endpoint: CompleteEndpoint,
               coordinates: CycleCoordinates) -> PolicyResult:
        r_step = int(coordinates.r_step)
        editable = [int(p) for p in source.editable_positions]
        masked = [p for p in editable if source.tokens[p] == source.mask_token_id]
        resolved = [p for p in editable if source.tokens[p] != source.mask_token_id]

        def evidence(stall_reason: str | None, **over: Any) -> HeadDirectedDecisionEvidence:
            base = dict(
                policy_id=SOURCE_GEOMETRY_CONTROL_POLICY_ID, stall_reason=stall_reason,
                donor_gate=None, incumbent_id=self.incumbent.incumbent_id,
                incumbent_sequence_md5=self.incumbent.sequence_md5,
                safety_reference_sequence_md5=self.incumbent.safety_reference_sequence_md5,
                donor_endpoint_id=str(endpoint.endpoint_id),
                donor_sequence_md5=str(endpoint.sequence_md5),
                incumbent_window_evidence_digest=None, safety_window_evidence_digest=None,
                calibration=self.calibration.canonical_payload(),
                n_editable=len(editable), n_unresolved_source=len(masked), u_target=None,
                band_center_rule=self.calibration.band_center_rule.value,
                write_cap=int(self.required_writes), n_legal_write_candidates=len(masked),
                n_positive_contributions=0, m_band_min=None, m_band_max=None,
                realized_writes=0, required_reopen=int(self.required_reopens), realized_reopen=0,
                n_legal_reopen_candidates=len(resolved), write_candidates=(),
                reopen_candidates=(), head_calls=0,
                # The claim this arm exists to make.
                head_evidence_consulted=False,
            )
            base.update(over)
            return HeadDirectedDecisionEvidence(**base)

        gate = donor_gate(
            donor=endpoint, incumbent=self.incumbent,
            epsilon_r=self.calibration.epsilon_r,
            epsilon_source_ref=self.calibration.epsilon_source_ref,
        )
        if not gate.passed:
            # The control gates the donor identically, so both arms stall on the same donors and a
            # difference in outcome can never come from one arm running on endpoints the other
            # refused.
            return PolicyRejection(
                reason=f"{StallReason.NO_BETTER_DONOR.value}: the control gates the donor on the "
                       f"same incumbent and margin as the treatment; reason={gate.reason.value}",
                policy=self.identity(),
                decision_evidence=evidence(StallReason.NO_BETTER_DONOR.value, donor_gate=gate),
            )

        if len(masked) < int(self.required_writes) or len(resolved) < int(self.required_reopens):
            return PolicyRejection(
                reason=(
                    f"{StallReason.CONTROL_CARDINALITY_UNMATCHABLE.value}: the treatment realized "
                    f"{self.required_writes} write(s) over source masks and "
                    f"{self.required_reopens} reopen(s), but this source carries {len(masked)} "
                    f"masked and {len(resolved)} resolved editable position(s)"
                ),
                policy=self.identity(),
                decision_evidence=evidence(
                    StallReason.CONTROL_CARDINALITY_UNMATCHABLE.value, donor_gate=gate),
            )

        def commit_step(position: int) -> int:
            commit = source.active_commit_depth_step_by_pos[position]
            return -1 if commit is None else int(commit.step)

        write = sorted(masked)[:int(self.required_writes)]
        reopen = sorted(
            sorted(resolved, key=lambda p: (-commit_step(p), p))[:int(self.required_reopens)])
        written, reopened = set(write), set(reopen)
        inject = sorted(p for p in resolved
                        if p not in reopened and commit_step(p) >= r_step)
        carry = sorted(
            [p for p in masked if p not in written]
            + [p for p in resolved if p not in reopened and commit_step(p) < r_step]
        )
        reasons: dict[int, SupportReason] = {p: SupportReason.SOURCE_GEOMETRY for p in write}
        reasons.update({p: SupportReason.SOURCE_GEOMETRY for p in reopen})
        reasons.update({p: SupportReason.FUTURE_SOURCE_IDENTITY for p in inject})
        reasons.update({
            p: (SupportReason.INHERITED_MASK if source.tokens[p] == source.mask_token_id
                else SupportReason.TEMPORALLY_VALID)
            for p in carry
        })
        return PolicyDecision(
            write_from_endpoint=tuple(write), inject_from_source_feedback=tuple(inject),
            reopen=tuple(reopen), carry_from_source=tuple(carry),
            reason_by_pos=reasons, policy=self.identity(),
            decision_evidence=evidence(
                None, donor_gate=gate, realized_writes=len(write), realized_reopen=len(reopen),
                write_candidates=tuple(
                    WriteCandidateEvidence(
                        position=p, donor_residue=endpoint.sequence[p],
                        incumbent_residue=self.incumbent.sequence[p], legal=True,
                        rejection_reason=None if p in written else "below_required_write_count",
                        min_window_delta=None, contribution=None,
                        selection_rank=index if p in written else None,
                        selected=p in written,
                    )
                    for index, p in enumerate(sorted(masked))
                ),
            ),
        )

    def __call__(self, source: LivePartialState, endpoint: CompleteEndpoint,
                 coordinates: CycleCoordinates) -> PolicyResult:
        return self.decide(source=source, endpoint=endpoint, coordinates=coordinates)
