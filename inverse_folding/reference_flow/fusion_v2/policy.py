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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .errors import V2Error
from .identity import ProjectionPolicyIdentity, require_digest
from .schedule import (
    CycleCoordinates,
    EmptyBandIntersectionError,
    MissingScheduleBandError,
    ScheduleBandTable,
    admissible_reopen_cardinality,
    lookup_band,
)
from .state import CompleteEndpoint, LivePartialState

__all__ = [
    "V2PolicyError",
    "SupportReason",
    "PolicyDecision",
    "PolicyRejection",
    "PolicyResult",
    "FeedbackSupportPolicy",
    "ExplicitProbePolicy",
    "EXPLICIT_PROBE_POLICY_ID",
    "STATE_DERIVED_PROBE_POLICY_ID",
    "StateDerivedProbePolicy",
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

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise V2PolicyError("a rejection must carry a non-empty human-readable reason")
        object.__setattr__(self, "reason_by_pos", _reasons(self.reason_by_pos, "reason_by_pos"))
        if self.policy is not None and not isinstance(self.policy, ProjectionPolicyIdentity):
            raise V2PolicyError("policy must be a ProjectionPolicyIdentity")


#: What a policy hands the kernel.
PolicyResult = PolicyDecision | PolicyRejection


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
