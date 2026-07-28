"""V1F5 torch-free cost ledger (PLAN_RF_REFINE_FUSION_V1 §3.3).

Append-safe and idempotently aggregatable. Each logical method event has a stable ``event_id``;
each physical execution/retry has a distinct ``attempt_id``. Logical accounting (lane-DFE, Head
samples, structure requests) is counted ONCE per committed logical event; physical accounting
(batched forwards, cache hits, walltime) sums per distinct attempt. Resume that re-emits a
committed event never double-counts; a conflicting second commit is a hard error.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: ``terminal_complete`` is the P1 TERMINAL arm's own generation phase. It is deliberately distinct
#: from ``full_control`` (the T0 ``independent_full`` policy): both draw complete trajectories, but
#: conflating them in the ledger would make it impossible to tell a P1 arm's spend from a T0
#: control's when reconciling matched compute.
PHASES = frozenset(
    {"root_prefix", "est", "eval", "final", "full_control", "terminal_complete",
     "initial_refold", "terminal_fusion"}
)
#: ``deferred`` marks work that was deliberately NOT performed at this stage (e.g. definitive
#: structure deferred to the unchanged v0 admission). It stays visible and may carry real physical
#: cost, but never contributes committed logical method cost.
_STATUSES = frozenset({"ok", "failed", "retried", "deferred"})
_IDENTITY_FIELDS = ("event_id", "protein_id", "arm", "attempt_id")


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    protein_id: str
    arm: str
    phase: str
    attempt_id: str
    status: str
    # logical accounting (counted once per committed event_id)
    logical_dfe: int = 0
    head_samples: int = 0
    structure_requests: int = 0
    # physical accounting (summed per distinct attempt_id)
    physical_forward_calls: int = 0
    structure_cache_hits: int = 0
    completion_cache_hits: int = 0
    walltime_s: float = 0.0
    #: What actually HAPPENED, independent of whether the cost was committed. ``status`` governs
    #: accounting ("was this cost committed?"); a failed trajectory still commits its DFE because
    #: the compute came out of the matched budget. Without this field cost_ledger.jsonl would read
    #: as if every charged unit succeeded. Descriptive only -- never used in any total.
    outcome: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {sorted(PHASES)}, got {self.phase!r}")
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {sorted(_STATUSES)}, got {self.status!r}")
        for name in _IDENTITY_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str, got {value!r}")
        # Cost counters are physical/logical quantities: a negative or non-finite value is a
        # corrupted measurement, never a legitimate accounting entry (PLAN §3.3).
        for name in _LOGICAL + ("physical_forward_calls", "structure_cache_hits",
                                "completion_cache_hits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if isinstance(self.walltime_s, bool) or not isinstance(self.walltime_s, (int, float)):
            raise ValueError(f"walltime_s must be a number, got {self.walltime_s!r}")
        if not math.isfinite(float(self.walltime_s)) or float(self.walltime_s) < 0.0:
            raise ValueError(f"walltime_s must be finite and >= 0, got {self.walltime_s!r}")


_LOGICAL = ("logical_dfe", "head_samples", "structure_requests")
_PHYSICAL = (
    "physical_forward_calls", "structure_cache_hits", "completion_cache_hits", "walltime_s"
)


#: The logical identity an attempt belongs to. A physical execution serves exactly ONE logical
#: event, and a logical event belongs to exactly one protein/arm/phase.
_LOGICAL_IDENTITY = ("event_id", "protein_id", "arm", "phase")


def _logical_identity(event: LedgerEvent) -> tuple:
    return tuple(getattr(event, field) for field in _LOGICAL_IDENTITY)


def _same_physical(a: LedgerEvent, b: LedgerEvent) -> bool:
    """Two rows describe the SAME physical execution only if they belong to the same logical
    event AND every physical counter agrees (walltime compared with a tolerance)."""
    if _logical_identity(a) != _logical_identity(b) or a.status != b.status:
        return False
    for field in ("physical_forward_calls", "structure_cache_hits", "completion_cache_hits"):
        if getattr(a, field) != getattr(b, field):
            return False
    return math.isclose(a.walltime_s, b.walltime_s, rel_tol=1e-9, abs_tol=1e-9)


def aggregate_ledger(events: Sequence[LedgerEvent]) -> dict:
    """Idempotently aggregate an append-only ledger. Logical fields are counted once per
    committed ``event_id`` (a conflicting second commit raises); physical fields sum per distinct
    ``attempt_id`` (a re-emitted attempt is deduplicated)."""
    committed: dict[str, LedgerEvent] = {}
    physical_by_attempt: dict[str, LedgerEvent] = {}
    attempt_owner: dict[str, tuple] = {}
    event_owner: dict[str, tuple] = {}
    for event in events:
        # An attempt is bound to ONE logical event for the whole run: seeing the same attempt_id
        # under a different logical identity means the ledger cannot be reconciled at all.
        bound = attempt_owner.get(event.attempt_id)
        if bound is not None and bound != _logical_identity(event):
            raise ValueError(
                f"attempt {event.attempt_id!r} is bound to logical event {bound!r} but was also "
                f"emitted for {_logical_identity(event)!r}"
            )
        attempt_owner[event.attempt_id] = _logical_identity(event)
        # An event_id likewise identifies ONE logical event; the same id under a different
        # protein/arm/phase would silently merge two proteins' method cost.
        owner = event_owner.get(event.event_id)
        if owner is not None and owner != _logical_identity(event):
            raise ValueError(
                f"event_id {event.event_id!r} used for two different logical events: "
                f"{owner!r} vs {_logical_identity(event)!r}"
            )
        event_owner[event.event_id] = _logical_identity(event)
        prior_attempt = physical_by_attempt.get(event.attempt_id)
        if prior_attempt is not None and not _same_physical(prior_attempt, event):
            # One physical execution has ONE physical cost. A second, different row for the same
            # attempt_id is a corrupted ledger; silently keeping the last write would let a bogus
            # count replace a measured one.
            raise ValueError(
                f"conflicting physical values for attempt {event.attempt_id!r}"
            )
        physical_by_attempt[event.attempt_id] = event  # idempotent dedup by attempt_id
        if event.status == "ok":
            prior = committed.get(event.event_id)
            if prior is not None and any(
                getattr(prior, field) != getattr(event, field) for field in _LOGICAL
            ):
                raise ValueError(
                    f"conflicting committed logical values for event {event.event_id!r}"
                )
            committed[event.event_id] = event

    totals: dict = {}
    for field in _LOGICAL:
        totals[field] = sum(getattr(ev, field) for ev in committed.values())
    for field in _PHYSICAL:
        totals[field] = sum(getattr(a, field) for a in physical_by_attempt.values())
    totals["n_logical_events"] = len(committed)
    totals["n_physical_attempts"] = len(physical_by_attempt)

    by_phase: dict[str, dict] = {}
    for event in committed.values():
        bucket = by_phase.setdefault(event.phase, {field: 0 for field in _LOGICAL})
        for field in _LOGICAL:
            bucket[field] += getattr(event, field)
    totals["by_phase"] = by_phase
    return totals
