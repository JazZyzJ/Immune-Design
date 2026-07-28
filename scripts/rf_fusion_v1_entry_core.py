"""V1F5 per-protein entry orchestration (PLAN_RF_REFINE_FUSION_V1 §2.3, §2.9-§2.11, §3.3, §5.3).

Torch/pandas-free coordination of the PRE-TERMINAL entry: root prefix -> estimator value -> deterministic
continuation-value beam -> fresh ``final`` materialization -> ordered facade -> definitive v0
admission, with an event-level cost ledger. The model surfaces are injected as rich OUTCOMES
(carrying every attempt's paid DFE, physical forwards, and any failure reason) so accounting is
never lost and the whole replay graph is testable with fakes; the real DPLM/Head/structure wiring
is supplied by the V1F6 driver (:mod:`scripts.run_rf_fusion_v1_entry`).

Contract highlights:
- EVERY frozen prefix attempt (no-crossing / invalid / duplicate) is charged and recorded (§2.3);
- if the number of valid unique roots is below the frozen unique-root capacity, the protein stops
  with ``entry_insufficient_unique_roots`` before any estimator work (§2.3);
- the facade offers the top ``initial_refold_attempt_cap`` roots by value so a rank-0 structure
  failure falls back to a lower-ranked root (§2.11);
- facade order follows ROOT VALUE, never the materialized terminal Head; final continuations still
  pass the complete-AA20 firewall and are Head-scored as evidence (§2.8/§2.11).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from inverse_folding.reference_flow.fusion.v1_admission import (
    AdmissionResult,
    FacadeCandidate,
    _coerce_structure_outcome,
    admit_facade,
)
from inverse_folding.reference_flow.fusion.v1_alloc import (
    Continuation,
    preterminal_entry_dfe,
    reserved_p1_preterminal_dfe,
    FacadeRow,
    FinalMaterialization,
    RootValue,
    ScoredContinuation,
    TerminalCandidate,
    build_preterminal_facade,
    build_terminal_facade,
    authorize_final_materialization,
    build_common_eval_table,
    build_t0_structure_subset,
    collapse_facade_by_sequence,
    eval_membership_view,
    matched_full_trajectory_allocation,
    reserved_t0_dfe,
    select_random_membership,
    t0_reserved_structure_requests,
    evaluate_continuations,
    head_risk_by_md5,
    root_value,
    select_value_beam,
)
from inverse_folding.reference_flow.fusion.v1_config import V1EntryConfig
from inverse_folding.reference_flow.fusion.v1_ledger import LedgerEvent
from inverse_folding.reference_flow.fusion.v1_records import (
    PartialRootPayload,
    RootCollapse,
    collapse_roots,
)
from inverse_folding.reference_flow.fusion.v1_seeds import SeedContext


def rho_id_for(rho_target: float) -> str:
    """Canonical, stable three-decimal rho identity used across payloads, seeds, and artifacts."""
    return f"rho{float(rho_target):.3f}"


# --------------------------------------------------------------------------- #
# injected seams (real wiring lives in the driver) and their rich outcomes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RootAttemptRequest:
    protein_id: str
    arm_id: str
    rho_id: str
    attempt_index: int
    seed: int


@dataclass(frozen=True)
class RootAttemptOutcome:
    """One frozen prefix attempt. ``payload is None`` means no upward crossing or an invalid
    payload; ``paid_prefix_dfe`` is charged regardless of outcome (§2.3)."""

    attempt_index: int
    seed: int
    payload: PartialRootPayload | None
    paid_prefix_dfe: int
    physical_forward_calls: int = 0
    status: str = "crossed"  # crossed | no_crossing | invalid
    failure_reason: str | None = None
    walltime_s: float = 0.0


@dataclass(frozen=True)
class CompletionRequest:
    payload: PartialRootPayload
    set_tag: str  # est | eval | final
    replicate_index: int
    seed: int


@dataclass(frozen=True)
class CompletionOutcome:
    sequence: str
    logical_dfe: int
    physical_forward_calls: int = 0
    cache_hit: bool = False
    model_executed: bool = True
    walltime_s: float = 0.0


@dataclass(frozen=True)
class TerminalTrajectoryRequest:
    """One independent complete ``c1_null`` trajectory of the TERMINAL arm. It carries no root and
    no maturity: the arm's whole point is that selection happens at the endpoint."""

    protein_id: str
    arm: str
    replicate_index: int
    seed: int


@dataclass(frozen=True)
class TerminalTrajectoryOutcome:
    """A complete trajectory plus the cost it actually consumed. ``sequence is None`` marks a
    failed trajectory, which still cost its full DFE and is still recorded (§3.3)."""

    replicate_index: int
    seed: int
    sequence: str | None
    logical_dfe: int
    physical_forward_calls: int = 0
    walltime_s: float = 0.0
    status: str = "complete"  # complete | failed
    failure_reason: str | None = None


RootGenerator = Callable[[RootAttemptRequest], RootAttemptOutcome]
Completer = Callable[[CompletionRequest], CompletionOutcome]
TerminalTrajectoryGenerator = Callable[[TerminalTrajectoryRequest], TerminalTrajectoryOutcome]
HeadFn = Callable[[list], list]
StructureGate = Callable[[FacadeCandidate], bool]


@dataclass(frozen=True)
class EntryOracles:
    """The four injected model surfaces for one arm/campaign. The driver builds these from real
    DPLM/Head/structure runtimes; tests inject deterministic fakes."""

    root_generator: RootGenerator
    completer: Completer
    head_fn: HeadFn
    structure_gate: StructureGate
    #: TERMINAL arm only. The two arms share head_fn and structure_gate (the same firewall and the
    #: same deferral); they differ only in how candidates are produced.
    trajectory_generator: TerminalTrajectoryGenerator | None = None


# --------------------------------------------------------------------------- #
# per-protein records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RootSelectionRecord:
    root_equivalence_hash: str
    policy: str
    rank: int
    selected: bool
    source_value: float
    fresh_materialization_authorized: bool


@dataclass(frozen=True)
class MaturityRecord:
    root_equivalence_hash: str
    length: int
    n_fixed: int
    n_editable: int
    n_resolved_editable: int
    rho_edit: float
    known_identity_fraction: float


@dataclass
class EntryJournal:
    """A live sink for already-paid work. ``run_entry_protein`` accumulates ledger events and
    root-attempt outcomes here AS it progresses, so a mid-run failure preserves the paid prefix /
    estimator work instead of discarding it (PLAN §3.3: failed/retried work is never silently
    dropped)."""

    ledger: list = field(default_factory=list)
    root_attempts: list = field(default_factory=list)
    #: Which EXECUTION of this protein this is. A crashed attempt already burned GPU time; the
    #: retry must not overwrite that record, so every attempt_id is suffixed with its epoch and the
    #: physical costs sum instead of deduplicating. The LOGICAL event ids stay stable, because one
    #: root attempt is one unit of method cost however many times it had to be executed (§3.3).
    attempt_epoch: int = 0

    def attempt_id(self, base: str) -> str:
        return f"{base}#e{int(self.attempt_epoch)}"


@dataclass(frozen=True)
class PreterminalReservation:
    """The per-protein matched budget ``C_reserved``, and everything it was derived from.

    This is the TERMINAL arm's only input: it generates ``M_T = floor(C_reserved / S)`` complete
    trajectories. It is therefore computed and frozen the moment the unique roots exist and BEFORE
    any estimator Head score is seen -- a reservation that could react to Head evidence would make
    the Terminal arm's budget outcome-dependent, which is the backfill §3.2 forbids.
    ``head_samples_seen_at_reservation`` records that ordering as data, not as a comment.
    """

    protein_id: str
    arm_id: str
    reserved_dfe: int
    prefix_dfe_total: int
    k_est: int
    tail_dfe_by_root: tuple[int, ...]
    r_max: int
    f_cap: int
    n_unique_roots: int
    head_samples_seen_at_reservation: int


@dataclass(frozen=True)
class ProteinEntryResult:
    protein_id: str
    arm_id: str
    root_attempts: tuple[RootAttemptOutcome, ...]
    unique_roots: tuple[PartialRootPayload, ...]
    root_collapse: RootCollapse
    est_scored_by_root: dict[str, tuple[ScoredContinuation, ...]]
    root_values: tuple[RootValue, ...]  # value-rank order
    ranked_root_hashes: tuple[str, ...]
    root_selection: tuple[RootSelectionRecord, ...]
    materializations: dict[str, FinalMaterialization]
    final_continuations: tuple[Continuation, ...]
    final_scored: tuple[ScoredContinuation, ...]
    facade: tuple[FacadeRow, ...]
    admission: AdmissionResult
    maturity: tuple[MaturityRecord, ...]
    ledger: tuple[LedgerEvent, ...]
    status: str
    #: TERMINAL arm only: every generated complete trajectory, including the failed ones and the
    #: ones that ranked below F_cap. They were all paid for, so they all stay visible.
    terminal_trajectories: tuple["TerminalTrajectoryOutcome", ...] = ()
    #: TERMINAL arm only: every SCORED trajectory with its measured terminal Head risk, including
    #: the ones that ranked below F_cap (they were scored, so recording null would lose real data).
    terminal_pool: tuple[TerminalCandidate, ...] = ()
    #: PRE-TERMINAL arm only: the matched budget the Terminal arm will be handed. ``None`` when the
    #: protein never formed a root pool, so an absent budget can never read as ``M_T = 0``.
    reservation: "PreterminalReservation | None" = None
    #: Facade rows dropped because their complete sequence duplicated a better-ranked row's. They
    #: are recorded, not silently discarded: which distinct roots converged is real information.
    facade_collapsed: tuple = ()
    #: What the protein ACTUALLY spent, beside the pre-outcome ``reservation.reserved_dfe`` it was
    #: budgeted against. Without both numbers there is no way to see a run come in under its own
    #: reservation -- and the Terminal arm is matched to the reservation, not to the spend.
    actual_entry_dfe: int = 0

    @property
    def n_facade_collapsed(self) -> int:
        return len(self.facade_collapsed)


def _validate_root_outcome(outcome, index, seed, protein_id, arm, rho_id) -> None:
    """A seam may not silently rewrite an attempt's identity: the outcome must echo the requested
    (index, seed), and its status must agree with payload presence and the requested lineage."""
    if outcome.attempt_index != index or outcome.seed != seed:
        raise ValueError(
            f"root outcome identity mismatch: got (index={outcome.attempt_index}, "
            f"seed={outcome.seed}), expected (index={index}, seed={seed})"
        )
    if outcome.payload is None:
        if outcome.status not in ("no_crossing", "invalid"):
            raise ValueError(f"no-payload attempt {index} has status {outcome.status!r}")
        return
    if outcome.status != "crossed":
        raise ValueError(f"attempt {index} has a payload but status {outcome.status!r}")
    p = outcome.payload
    if (p.protein_id, p.arm_id, p.rho_id) != (protein_id, arm, rho_id):
        raise ValueError(
            f"attempt {index} payload lineage {(p.protein_id, p.arm_id, p.rho_id)} != "
            f"{(protein_id, arm, rho_id)}"
        )


def _validate_completion(completion, payload, set_tag) -> None:
    """A completion must charge the exact matched tail (S - s); a seam reporting logical_dfe=0
    would silently zero real work in the ledger (matched-compute integrity, §3.1)."""
    expected = payload.n_steps - payload.step
    if completion.logical_dfe != expected:
        raise ValueError(
            f"{set_tag} completion logical_dfe {completion.logical_dfe} != matched tail "
            f"{expected} (n_steps {payload.n_steps} - step {payload.step})"
        )


def _maturity_record(payload: PartialRootPayload) -> MaturityRecord:
    length = len(payload.x_t)
    n_fixed = len(payload.fixed_tokens)
    n_editable = len(payload.editable_positions)
    n_resolved = n_editable - payload.n_unresolved_editable
    return MaturityRecord(
        root_equivalence_hash=payload.root_equivalence_hash, length=length, n_fixed=n_fixed,
        n_editable=n_editable, n_resolved_editable=n_resolved, rho_edit=payload.actual_rho_edit,
        known_identity_fraction=(n_fixed + n_resolved) / length,
    )


def _insufficient_result(protein_id, arm, attempts, ledger) -> ProteinEntryResult:
    from inverse_folding.reference_flow.fusion.v1_admission import AdmissionResult as _AR

    return ProteinEntryResult(
        protein_id=protein_id, arm_id=arm, root_attempts=tuple(attempts), unique_roots=(),
        root_collapse=RootCollapse(unique=(), converged={}), est_scored_by_root={},
        root_values=(), ranked_root_hashes=(), root_selection=(), materializations={},
        final_continuations=(), final_scored=(), facade=(),
        admission=_AR(attempts=(), n_admitted=0, insufficient=True,
                      reason="entry_insufficient_unique_roots"),
        maturity=(), ledger=tuple(ledger), status="entry_insufficient_unique_roots",
    )


def run_entry_protein(
    *,
    protein_id: str,
    config: V1EntryConfig,
    expected_length: int,
    root_generator: RootGenerator,
    completer: Completer,
    head_fn: HeadFn,
    structure_gate: StructureGate,
    seed_ctx: SeedContext,
    journal: "EntryJournal | None" = None,
) -> ProteinEntryResult:
    """Run the pre-terminal entry for one protein and return its complete replay-graph records. Emits no
    files (the cohort runner owns IO); the ledger is returned for aggregation. Pass a ``journal``
    to keep the already-paid ledger/attempts recoverable if a later stage raises."""
    if config.entry_arm != "preterminal":
        raise NotImplementedError(
            f"run_entry_protein implements the pre-terminal arm only; got entry_arm="
            f"{config.entry_arm!r}. Terminal-arm orchestration (independent complete trajectories "
            "ranked by exact terminal Head) is a separate unit."
        )
    if config.rho_target is None:
        raise NotImplementedError(
            "run_entry_protein implements the single-target P1 path; a T0 rho_grid config runs "
            "the three-policy calibration, which is a separate unit."
        )
    arm = config.entry_arm
    rho_id = rho_id_for(config.rho_target)
    if journal is None:
        journal = EntryJournal()
    ledger = journal.ledger

    # 1. root prefix — run every frozen attempt, charge each, keep the crossed payloads.
    attempts = journal.root_attempts
    for i in range(config.prefix_attempts):
        seed = seed_ctx.root_seed(i)
        outcome = root_generator(RootAttemptRequest(protein_id, arm, rho_id, i, seed))
        _validate_root_outcome(outcome, i, seed, protein_id, arm, rho_id)
        attempts.append(outcome)
        ledger.append(
            LedgerEvent(
                event_id=f"{protein_id}:{arm}:root:attempt{i}",
                protein_id=protein_id, arm=arm, phase="root_prefix",
                attempt_id=journal.attempt_id(f"{protein_id}:{arm}:root:attempt{i}"), status="ok",
                logical_dfe=outcome.paid_prefix_dfe,
                # physical is the reported count; 0 is a real value (e.g. a cached rollout), never
                # a "not set" sentinel to be back-filled from the logical DFE.
                physical_forward_calls=outcome.physical_forward_calls,
                walltime_s=outcome.walltime_s,
            )
        )
    crossed = [o.payload for o in attempts if o.payload is not None]
    collapse = collapse_roots(crossed)

    # Frozen capacity GATE (§2.3): too few valid unique roots -> stop before any estimator work.
    # The capacity is a MINIMUM coverage requirement, never a truncation: estimator/final work is
    # allocated over ALL of U, because slicing by root_id order would silently discard whichever
    # root happens to be the best one.
    if collapse.n_unique < config.unique_root_capacity:
        return _insufficient_result(protein_id, arm, attempts, ledger)
    unique_roots = tuple(collapse.unique)
    payload_by_hash = {p.root_equivalence_hash: p for p in unique_roots}
    maturity = tuple(_maturity_record(p) for p in unique_roots)

    # Freeze the matched budget HERE: the roots exist, and no Head score has been computed yet.
    tail_dfe_by_root = tuple(p.n_steps - p.step for p in unique_roots)
    prefix_dfe_by_attempt = [a.paid_prefix_dfe for a in attempts]
    reservation = PreterminalReservation(
        protein_id=protein_id, arm_id=arm,
        reserved_dfe=reserved_p1_preterminal_dfe(
            prefix_dfe_by_attempt, config.k_est, tail_dfe_by_root,
            config.initial_refold_attempt_cap,
        ),
        prefix_dfe_total=sum(int(d) for d in prefix_dfe_by_attempt),
        k_est=config.k_est, tail_dfe_by_root=tail_dfe_by_root,
        r_max=max(tail_dfe_by_root) if tail_dfe_by_root else 0,
        f_cap=config.initial_refold_attempt_cap, n_unique_roots=collapse.n_unique,
        head_samples_seen_at_reservation=sum(e.head_samples for e in ledger),
    )

    # 2. estimator value — exact K_EST arithmetic mean of complete-sequence Head risk per root.
    est_scored_by_root: dict[str, tuple[ScoredContinuation, ...]] = {}
    root_values: list[RootValue] = []
    for p in unique_roots:
        h = p.root_equivalence_hash
        conts = []
        for r in range(config.k_est):
            seed = seed_ctx.continuation_seed(h, "est", r)
            completion = completer(CompletionRequest(p, "est", r, seed))
            _validate_completion(completion, p, "est")
            conts.append(
                Continuation(
                    continuation_id=f"{h}:est:{r}", root_equivalence_hash=h, set_tag="est",
                    seed=seed, sequence=completion.sequence, replicate_index=r,
                )
            )
            ledger.append(
                LedgerEvent(
                    event_id=f"{protein_id}:{arm}:est:{h}:{r}",
                    protein_id=protein_id, arm=arm, phase="est",
                    attempt_id=journal.attempt_id(f"{protein_id}:{arm}:est:{h}:{r}"), status="ok",
                    logical_dfe=completion.logical_dfe, head_samples=1,
                    physical_forward_calls=completion.physical_forward_calls,
                    completion_cache_hits=int(completion.cache_hit),
                    walltime_s=completion.walltime_s,
                )
            )
        scored = evaluate_continuations(head_fn, conts, expected_length)
        est_scored_by_root[h] = scored
        root_values.append(RootValue(h, root_value(scored, config.k_est), config.k_est))

    # 3. deterministic value beam over the facade capacity; result.root_values is value-ranked.
    root_values.sort(key=lambda rv: (rv.value, rv.root_equivalence_hash))
    facade_capacity = min(len(root_values), config.initial_refold_attempt_cap)
    ranked_root_hashes = select_value_beam(root_values, facade_capacity)
    ranked_set = set(ranked_root_hashes)
    root_selection = tuple(
        RootSelectionRecord(
            root_equivalence_hash=rv.root_equivalence_hash, policy="continuation_value_beam",
            rank=rank, selected=rv.root_equivalence_hash in ranked_set, source_value=rv.value,
            fresh_materialization_authorized=rv.root_equivalence_hash in ranked_set,
        )
        for rank, rv in enumerate(root_values)
    )

    # 4. one fresh final continuation per ranked root; Head-scored through the AA20 firewall.
    materializations: dict[str, FinalMaterialization] = {}
    final_continuations: list[Continuation] = []
    for h in ranked_root_hashes:
        p = payload_by_hash[h]
        final_seed = seed_ctx.continuation_seed(h, "final", 0)
        completion = completer(CompletionRequest(p, "final", 0, final_seed))
        _validate_completion(completion, p, "final")
        cont = Continuation(
            continuation_id=f"{h}:final:0", root_equivalence_hash=h, set_tag="final",
            seed=final_seed, sequence=completion.sequence, replicate_index=0,
        )
        # A P1 parent is authorized ONLY by a selected partial root via a fresh `final`
        # continuation. The loop already only builds those, but a by-construction guarantee is one
        # refactor away from lapsing silently, so the guard is IN the path rather than merely
        # available next to it (§2.9/§2.11).
        authorize_final_materialization(ranked_root_hashes, cont)
        materializations[h] = FinalMaterialization(
            root_equivalence_hash=h, continuation=cont, source_id=p.root_id
        )
        final_continuations.append(cont)
        ledger.append(
            LedgerEvent(
                event_id=f"{protein_id}:{arm}:final:{h}",
                protein_id=protein_id, arm=arm, phase="final",
                attempt_id=journal.attempt_id(f"{protein_id}:{arm}:final:{h}"), status="ok",
                logical_dfe=completion.logical_dfe, head_samples=1,
                physical_forward_calls=completion.physical_forward_calls,
                completion_cache_hits=int(completion.cache_hit),
                walltime_s=completion.walltime_s,
            )
        )
    # complete-AA20 firewall + Head evidence for the final continuations (NOT a facade re-rank).
    final_scored = evaluate_continuations(head_fn, tuple(final_continuations), expected_length)

    # 5. ordered facade (root-value rank) -> 6. definitive v0 admission (first N feasible).
    facade_all = build_preterminal_facade(ranked_root_hashes, materializations)
    # Convergent roots can complete to the SAME sequence. v0 must see each sequence once, or it
    # fills two population slots with one basin and its particle_id stops identifying the source.
    collapse_facade = collapse_facade_by_sequence(facade_all)
    facade = collapse_facade.rows
    candidates = [
        FacadeCandidate(
            design_idx=f.design_idx, seed=f.seed, sequence=f.sequence,
            sequence_md5=f.sequence_md5, source_id=f.source_id,
            continuation_id=materializations[f.root_equivalence_hash].continuation.continuation_id,
            root_equivalence_hash=f.root_equivalence_hash,
        )
        for f in facade
    ]
    admission = admit_facade(
        candidates, structure_gate, config.n_population, config.initial_refold_attempt_cap
    )
    for attempt in admission.attempts:
        # A DEFERRED row stays visible (it WAS offered to v0) but books no logical structure
        # request: the refold has not happened yet, and v0 charges it when it does. Booking it here
        # would count the entire initial-refold budget twice, which breaks the matched-compute
        # claim the whole comparison rests on (§3.3).
        deferred = attempt.verdict == "deferred_to_v0"
        ledger.append(
            LedgerEvent(
                event_id=f"{protein_id}:{arm}:refold:{attempt.attempt_order}",
                protein_id=protein_id, arm=arm, phase="initial_refold",
                attempt_id=journal.attempt_id(
                    f"{protein_id}:{arm}:refold:{attempt.attempt_order}"),
                status="deferred" if deferred else "ok",
                structure_requests=0 if deferred else 1,
                structure_cache_hits=1 if attempt.cache_status == "hit" else 0,
                walltime_s=attempt.walltime_s,
            )
        )

    if admission.structure_deferred:
        # The entry stage completed; the definitive structure verdict belongs to v0 (§2.11). This
        # is NOT a terminal success -- no parent has been structurally admitted yet.
        status = "entry_complete_structure_deferred"
    else:
        status = "terminal_success" if not admission.insufficient else "entry_insufficient"
    status = _facade_status(status, facade, collapse_facade.collapsed, config, protein_id)
    return ProteinEntryResult(
        protein_id=protein_id, arm_id=arm, root_attempts=tuple(attempts),
        unique_roots=unique_roots, root_collapse=collapse, est_scored_by_root=est_scored_by_root,
        root_values=tuple(root_values), ranked_root_hashes=ranked_root_hashes,
        root_selection=root_selection, materializations=materializations,
        final_continuations=tuple(final_continuations), final_scored=final_scored, facade=facade,
        admission=admission, maturity=maturity, ledger=tuple(ledger), status=status,
        reservation=reservation, facade_collapsed=collapse_facade.collapsed,
        actual_entry_dfe=preterminal_entry_dfe(
            [a.paid_prefix_dfe for a in attempts], config.k_est, tail_dfe_by_root,
            [payload_by_hash[h].n_steps - payload_by_hash[h].step for h in ranked_root_hashes],
        ),
    )


# --------------------------------------------------------------------------- #
# TERMINAL arm (§3.2.1)
# --------------------------------------------------------------------------- #
def run_terminal_protein(
    *,
    protein_id: str,
    config: V1EntryConfig,
    expected_length: int,
    n_trajectories: int,
    s_steps: int,
    trajectory_generator: TerminalTrajectoryGenerator,
    head_fn: HeadFn,
    structure_gate: StructureGate,
    seed_ctx: SeedContext,
    journal: "EntryJournal | None" = None,
) -> ProteinEntryResult:
    """Run the TERMINAL arm for one protein: ``M_T`` independent complete ``c1_null`` trajectories,
    ranked by exact terminal Head, top ``F_cap`` handed to the same v0 admission.

    The generative half (complete c1_null trajectories) and the admission half (v0's
    ``build_initial_population``) already exist. What this adds is the piece between them: v0
    deliberately does NOT preselect on Head, so its ``design_idx`` is whatever order generation
    emitted. The pre-terminal arm's facade is ordered by ROOT VALUE -- a Head-derived quantity --
    so an arbitrarily-ordered Terminal facade would confound "does Head selection help at all"
    with "does selecting at a partial state beat selecting at the endpoint". V1-A isolates the
    latter, so this arm selects on Head too, just at the endpoint.

    ``n_trajectories`` is NOT a free knob: the driver derives it as ``M_T = floor(C_reserved / S)``
    from the persisted pre-terminal reservation (``preflight.terminal_trajectory_count``).
    """
    if config.entry_arm != "terminal":
        raise ValueError(
            f"run_terminal_protein requires entry_arm='terminal', got {config.entry_arm!r}"
        )
    if seed_ctx.entry_arm != "terminal":
        raise ValueError(
            f"terminal orchestration needs a terminal SeedContext, got "
            f"entry_arm={seed_ctx.entry_arm!r}: the arm is part of every seed prefix, so a "
            "mix-up would silently reuse the other arm's draws"
        )
    if not isinstance(n_trajectories, int) or isinstance(n_trajectories, bool) or n_trajectories <= 0:
        raise ValueError(f"n_trajectories must be a positive int, got {n_trajectories!r}")
    if not isinstance(s_steps, int) or isinstance(s_steps, bool) or s_steps <= 0:
        raise ValueError(f"s_steps must be a positive int, got {s_steps!r}")
    arm = config.entry_arm
    if journal is None:
        journal = EntryJournal()
    ledger = journal.ledger

    # 1. generate M_T independent complete trajectories; charge EVERY one, including failures and
    #    the ones that will rank below F_cap -- they were all paid out of the matched budget.
    outcomes: list[TerminalTrajectoryOutcome] = []
    for i in range(n_trajectories):
        seed = seed_ctx.terminal_complete_seed(i)
        outcome = trajectory_generator(TerminalTrajectoryRequest(protein_id, arm, i, seed))
        _validate_trajectory_outcome(outcome, i, seed, s_steps)
        outcomes.append(outcome)
        ledger.append(
            LedgerEvent(
                event_id=f"{protein_id}:{arm}:traj:{i}",
                protein_id=protein_id, arm=arm, phase="terminal_complete",
                attempt_id=journal.attempt_id(f"{protein_id}:{arm}:traj:{i}"),
                # A failed trajectory is a COMMITTED unit of method cost, exactly like a
                # no-crossing prefix attempt on the other arm: the DFE came out of the matched
                # budget and the other arm does not get it back. Marking it "failed" here would
                # drop its logical DFE from the totals and make this arm look cheaper than the
                # budget it was actually handed. The failure stays visible in
                # ``terminal_trajectories``, not by suppressing the cost (§3.3).
                status="ok",
                outcome=outcome.status,
                logical_dfe=int(outcome.logical_dfe),
                head_samples=1 if outcome.sequence is not None else 0,
                physical_forward_calls=int(outcome.physical_forward_calls),
                walltime_s=float(outcome.walltime_s),
            )
        )

    # 2. exact complete-sequence Head over the surviving trajectories (same firewall as the other
    #    arm: complete AA20 only, results bound by sequence_md5, never a positional zip).
    survivors = [o for o in outcomes if o.sequence is not None]
    md5_by_sequence, risk_by_md5 = head_risk_by_md5(
        head_fn, [o.sequence for o in survivors], expected_length
    )
    source_id_of = {o.replicate_index: f"{protein_id}:{arm}:t{o.replicate_index}" for o in survivors}
    seed_by_source = {source_id_of[o.replicate_index]: int(o.seed) for o in survivors}
    pool = [
        TerminalCandidate(
            sequence=o.sequence,
            sequence_md5=md5_by_sequence[o.sequence],
            global_risk=risk_by_md5[md5_by_sequence[o.sequence]],
            source_id=source_id_of[o.replicate_index],
        )
        for o in survivors
    ]

    # 3. rank by exact terminal Head, submit only the top F_cap (a short pool is a hard failure).
    facade_all = build_terminal_facade(
        pool, seed_fn=lambda sid: seed_by_source[sid],
        facade_cap=config.initial_refold_attempt_cap,
    )
    # Independent trajectories can also converge; the handoff must be sequence-unique either way.
    collapse_facade = collapse_facade_by_sequence(facade_all)
    facade = collapse_facade.rows
    candidates = [
        FacadeCandidate(
            design_idx=f.design_idx, seed=f.seed, sequence=f.sequence,
            sequence_md5=f.sequence_md5, source_id=f.source_id,
            continuation_id=None, root_equivalence_hash=None,
        )
        for f in facade
    ]
    admission = admit_facade(
        candidates, structure_gate, config.n_population, config.initial_refold_attempt_cap
    )
    for attempt in admission.attempts:
        deferred = attempt.verdict == "deferred_to_v0"
        ledger.append(
            LedgerEvent(
                event_id=f"{protein_id}:{arm}:refold:{attempt.attempt_order}",
                protein_id=protein_id, arm=arm, phase="initial_refold",
                attempt_id=journal.attempt_id(
                    f"{protein_id}:{arm}:refold:{attempt.attempt_order}"),
                status="deferred" if deferred else "ok",
                structure_requests=0 if deferred else 1,
                structure_cache_hits=1 if attempt.cache_status == "hit" else 0,
                walltime_s=attempt.walltime_s,
            )
        )

    if admission.structure_deferred:
        status = "entry_complete_structure_deferred"
    else:
        status = "terminal_success" if not admission.insufficient else "entry_insufficient"
    status = _facade_status(status, facade, collapse_facade.collapsed, config, protein_id)
    return ProteinEntryResult(
        protein_id=protein_id, arm_id=arm, root_attempts=(), unique_roots=(),
        root_collapse=collapse_roots([]), est_scored_by_root={}, root_values=(),
        ranked_root_hashes=(), root_selection=(), materializations={},
        final_continuations=(), final_scored=(), facade=facade, admission=admission,
        maturity=(), ledger=tuple(ledger), status=status,
        terminal_trajectories=tuple(outcomes), terminal_pool=tuple(pool),
        facade_collapsed=collapse_facade.collapsed,
    )


def _facade_status(status, facade, collapsed, config, protein_id: str) -> str:
    """Surface a facade that convergence shortened below the common ``F_cap``.

    Both arms submit at most ``F_cap`` rows to initial-refold admission -- that equality is half of
    "matched compute". Collapse is the right response to two roots reaching the same sequence (v0
    must not fill two of its N slots with one basin), but a collapsed facade DELIVERS FEWER rows
    than the budget reserved, and reporting that with the same status as a full facade makes the
    under-spend invisible.

    Below ``N`` it is worse than an under-spend: v0 raises ``insufficient_feasible_initial_population``
    and the failure is attributed to the terminal stage, long after this driver returned 0.
    """
    if not collapsed:
        return status
    if len(facade) < config.n_population:
        raise ValueError(
            f"{protein_id}: convergence collapsed the facade to {len(facade)} rows, below the "
            f"frozen population size N={config.n_population}; v0 cannot form a population from it "
            "and would report the shortfall as a terminal-stage failure (§2.11)"
        )
    return "entry_facade_collapsed"


def _validate_trajectory_outcome(
    outcome: TerminalTrajectoryOutcome, index: int, seed: int, s_steps: int
) -> None:
    """A seam may not silently rewrite a trajectory's identity or hide a failure as a success."""
    if outcome.replicate_index != index or outcome.seed != seed:
        raise ValueError(
            f"terminal trajectory identity mismatch: got (index={outcome.replicate_index}, "
            f"seed={outcome.seed}), expected (index={index}, seed={seed})"
        )
    if outcome.status not in ("complete", "failed"):
        raise ValueError(f"terminal trajectory {index} has status {outcome.status!r}")
    if (outcome.sequence is None) != (outcome.status == "failed"):
        raise ValueError(
            f"terminal trajectory {index}: status={outcome.status!r} disagrees with "
            f"sequence presence"
        )
    if int(outcome.logical_dfe) != int(s_steps):
        # ">0" is not a check: an oracle under-reporting a full S-step trajectory as 1 DFE makes
        # this arm look ~S times cheaper than the budget it was handed, and that budget IS the
        # comparison. A complete trajectory costs exactly S, failures included (§3.3).
        raise ValueError(
            f"terminal trajectory {index} charged {outcome.logical_dfe} DFE but a complete "
            f"trajectory costs exactly S={s_steps}, whether it succeeds or fails"
        )


# --------------------------------------------------------------------------- #
# T0 calibration: three policy VIEWS over one shared pool (§2.9-§2.10, §3.2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class T0PolicyView:
    """One policy's VIEW. ``eval_rows`` holds the very tuples from the shared table (identity, not
    a copy), so a test can assert reuse rather than equality -- equality would still pass if a
    policy had redrawn continuations that happened to score the same."""

    policy: str
    root_hashes: tuple[str, ...] = ()
    eval_rows: dict = field(default_factory=dict)
    n_trajectories: int = 0


@dataclass(frozen=True)
class T0StructureResult:
    """One of the ``3 * Q_T0`` reserved structure requests. ``evaluated=False`` records a deferral
    honestly rather than as a pass, exactly like the P1 admission audit."""

    policy: str
    request_id: str
    source_endpoint_id: str
    subset_rank: int
    evaluated: bool
    feasible: bool | None
    cache_status: str
    model_executed: bool
    failure_reason: str | None


@dataclass(frozen=True)
class T0Result:
    protein_id: str
    unique_root_hashes: tuple[str, ...]
    root_values: tuple[RootValue, ...]
    common_eval_table: dict
    policy_views: dict
    full_trajectories: tuple["TerminalTrajectoryOutcome", ...]
    matched_full_trajectories: int
    structure_subset: dict
    structure_results: tuple = ()
    structure_requests: int = 0
    reserved_dfe: int = 0
    ledger: tuple[LedgerEvent, ...] = ()
    status: str = "t0_complete"
    rho_id: str = ""
    #: T0 builds no parent at all, so there is no facade. Kept explicit so a reader does not have
    #: to infer the absence.
    facade: tuple = ()
    #: The raw evidence this grid point produced. All four runbook §6.1 GO/KILL conditions are
    #: computed from these tables, so they must leave the function: a T0 point that returns only
    #: policy membership cannot be interpreted, however cleanly it exits.
    root_attempts: tuple = ()
    unique_roots: tuple = ()
    maturity: tuple = ()
    est_scored_by_root: dict = field(default_factory=dict)
    eval_scored_by_root: dict = field(default_factory=dict)
    #: The independent-full control ranked by EXACT complete-sequence Head under the Terminal law
    #: (runbook §6:268-270). Without it the control carries no quantity comparable to the partial
    #: views and GO/KILL condition 3 is unanswerable.
    full_pool: tuple = ()


def run_t0_protein(
    *,
    protein_id: str,
    config: V1EntryConfig,
    expected_length: int,
    rho_target: float,
    root_generator: RootGenerator,
    completer: Completer,
    head_fn: HeadFn,
    full_trajectory_generator: TerminalTrajectoryGenerator,
    structure_gate: StructureGate,
    s_steps: int,
    seed_ctx: SeedContext,
    journal: "EntryJournal | None" = None,
) -> T0Result:
    """Run one T0 maturity point for one protein.

    The three policy views differ in exactly ONE thing -- which roots they hold. ``selected_partial``
    takes the continuation-value beam, ``random_partial`` takes a Head-INDEPENDENT membership draw
    from the SAME pool, and both read the SAME held-out ``K_EVAL`` table; ``independent_full`` is the
    compute-matched full-trajectory control. If a policy label could open its own continuation
    stream, T0 would be comparing different draws instead of different selections, and the whole
    calibration would measure sampling noise.

    T0 materializes NO parent: no ``final`` continuation, no admission. Building one would spend the
    P1 budget and let a calibration outcome leak into the P1 population.
    """
    if config.phase != "t0":
        raise ValueError(f"run_t0_protein requires phase='t0', got {config.phase!r}")
    arm = config.entry_arm
    rho_id = rho_id_for(rho_target)
    if journal is None:
        journal = EntryJournal()
    # The cohort runs the WHOLE rho grid against ONE journal so a crash mid-grid still preserves
    # every paid attempt. But a grid point is its own matched root-attempt group: PLAN §2.3:238-241
    # -- "A T0 grid runs separate matched root-attempt groups, so prefix compute is not amortized
    # across maturities." So this point APPENDS to the journal (crash recovery) and READS only the
    # tail it wrote (isolation). Aliasing the journal's lists instead would let a later maturity
    # pass its own coverage gate on an earlier maturity's roots and inflate its reserved budget --
    # producing real-looking, wrong numbers with no error anywhere.
    ledger_start = len(journal.ledger)
    attempts_start = len(journal.root_attempts)
    ledger = journal.ledger

    def _own_ledger() -> tuple:
        return tuple(journal.ledger[ledger_start:])

    # 1. root prefix — identical to the pre-terminal arm; every attempt is charged.
    for i in range(config.prefix_attempts):
        seed = seed_ctx.root_seed(i)
        outcome = root_generator(RootAttemptRequest(protein_id, arm, rho_id, i, seed))
        _validate_root_outcome(outcome, i, seed, protein_id, arm, rho_id)
        journal.root_attempts.append(outcome)
        ledger.append(LedgerEvent(
            event_id=f"{protein_id}:{arm}:t0root:{rho_id}:{i}", protein_id=protein_id, arm=arm,
            phase="root_prefix", attempt_id=journal.attempt_id(
                f"{protein_id}:{arm}:t0root:{rho_id}:{i}"), status="ok",
            logical_dfe=outcome.paid_prefix_dfe,
            physical_forward_calls=outcome.physical_forward_calls, walltime_s=outcome.walltime_s,
        ))
    attempts = tuple(journal.root_attempts[attempts_start:])
    collapse = collapse_roots([o.payload for o in attempts if o.payload is not None])
    if collapse.n_unique < config.unique_root_capacity:
        return T0Result(
            protein_id=protein_id, unique_root_hashes=(), root_values=(), common_eval_table={},
            policy_views={}, full_trajectories=(), matched_full_trajectories=0,
            structure_subset={}, structure_requests=0, reserved_dfe=0, ledger=_own_ledger(),
            status="entry_insufficient_unique_roots", rho_id=rho_id, root_attempts=attempts,
        )
    unique_roots = tuple(collapse.unique)
    unique_hashes = tuple(p.root_equivalence_hash for p in unique_roots)

    # Freeze the matched budget before ANY Head evidence, exactly as the pre-terminal arm does.
    tail_dfe_by_root = tuple(p.n_steps - p.step for p in unique_roots)
    reserved_dfe = reserved_t0_dfe(
        [a.paid_prefix_dfe for a in attempts], config.k_est, config.k_eval, tail_dfe_by_root
    )

    # 2. estimator set -> continuation value (the SELECTION signal)
    est_by_root, root_values = {}, []
    for payload in unique_roots:
        est_by_root[payload.root_equivalence_hash] = _scored_set(
            payload, "est", config.k_est, completer, head_fn, expected_length,
            protein_id, arm, seed_ctx, journal,
        )
        root_values.append(RootValue(
            payload.root_equivalence_hash,
            root_value(est_by_root[payload.root_equivalence_hash], config.k_est), config.k_est,
        ))

    # 3. ONE common held-out K_EVAL table per root — the EVALUATION signal, drawn once for all
    #    policies so a membership change can never become a resampling.
    eval_by_root = {
        payload.root_equivalence_hash: _scored_set(
            payload, "eval", config.k_eval, completer, head_fn, expected_length,
            protein_id, arm, seed_ctx, journal,
        )
        for payload in unique_roots
    }
    common_eval_table = build_common_eval_table(eval_by_root, config.k_eval)

    # 4. the three views
    n_view = min(config.n_population, len(unique_hashes))
    selected = select_value_beam(tuple(root_values), n_view)
    random_view = select_random_membership(
        unique_hashes, n_view, seed_ctx.random_membership_seed()
    )
    matched_full, _remainder = matched_full_trajectory_allocation(reserved_dfe, s_steps)
    full_outcomes: list[TerminalTrajectoryOutcome] = []
    for i in range(matched_full):
        seed = seed_ctx.independent_full_seed(i)
        outcome = full_trajectory_generator(
            TerminalTrajectoryRequest(protein_id, arm, i, seed)
        )
        _validate_trajectory_outcome(outcome, i, seed, s_steps)
        full_outcomes.append(outcome)
        # The maturity belongs in the id: every grid point runs its own control, and two points
        # sharing a logical event id would silently deduplicate one point's control cost out of the
        # matched-compute total.
        base = f"{protein_id}:{arm}:{rho_id}:full:{i}"
        ledger.append(LedgerEvent(
            event_id=base, protein_id=protein_id, arm=arm,
            phase="full_control", attempt_id=journal.attempt_id(base),
            status="ok", outcome=outcome.status, logical_dfe=int(outcome.logical_dfe),
            head_samples=1 if outcome.sequence is not None else 0,
            physical_forward_calls=int(outcome.physical_forward_calls),
            walltime_s=float(outcome.walltime_s),
        ))

    # The control is only a control if it carries a quantity comparable to the partial views.
    # Runbook §6:268-270 requires it "ranked by exact complete-sequence Head under the Terminal
    # law" -- the SAME firewall and the SAME total order the Terminal arm uses, so condition 3
    # compares like with like. Booking head_samples above without ever calling Head would also make
    # the ledger assert work that never ran.
    full_survivors = [o for o in full_outcomes if o.sequence is not None]
    full_pool: tuple = ()
    if full_survivors:
        md5_by_sequence, risk_by_md5 = head_risk_by_md5(
            head_fn, [o.sequence for o in full_survivors], expected_length
        )
        full_pool = tuple(sorted(
            (
                TerminalCandidate(
                    sequence=o.sequence, sequence_md5=md5_by_sequence[o.sequence],
                    global_risk=risk_by_md5[md5_by_sequence[o.sequence]],
                    source_id=f"{protein_id}:{arm}:{rho_id}:full:{o.replicate_index}",
                )
                for o in full_survivors
            ),
            key=lambda c: (c.global_risk, c.sequence_md5, c.source_id),
        ))
    policy_views = {
        "selected_partial": T0PolicyView(
            "selected_partial", selected, eval_membership_view(common_eval_table, selected)
        ),
        "random_partial": T0PolicyView(
            "random_partial", random_view, eval_membership_view(common_eval_table, random_view)
        ),
        "independent_full": T0PolicyView(
            "independent_full", (), {}, n_trajectories=len(full_outcomes)
        ),
    }

    # 5. equal-sized, Head-INDEPENDENT structure subsample per view (3 * Q_T0 total)
    #
    # All three pools are the FULL endpoint set of whatever each policy holds -- every eval endpoint
    # of the selected roots, every eval endpoint of the random roots, every surviving complete
    # trajectory -- and the Q_T0 draw within each is by Head-independent content hash. PLAN §2.10
    # (:469-470) requires exactly that: "order eligible IDs by a Head-independent content hash; no
    # policy may use another rule or sample size."
    #
    # This is deliberately NOT the Terminal law's top-`F_cap`. Truncating only the control to its
    # Head-best rows would give one policy an eligibility filter the other two do not get: sized
    # with the shipped canary the control would be drawn from its best ~25% while the partial views
    # are drawn from 100% of theirs, which inflates the control and biases GO/KILL condition 3
    # toward a false KILL. `full_pool` stays Head-ranked because `complete_entry_pool` (runbook
    # §9:451) and the Terminal-law comparison need that order -- not because it gates eligibility.
    #
    # OPEN SCIENTIFIC DECISION, NOT SETTLED HERE: no authority defines the independent-full
    # ELIGIBLE pool. The reading above ("all survivors", one rule for all three) is what PLAN:466
    # most plainly says, but a "frontier" reading (FUSION_V1:172, runbook:296) would truncate ALL
    # THREE pools by the same law instead. A scientific T0 must not launch until that is frozen;
    # the runbook T0 section records the question.
    endpoints = {
        "selected_partial": [sc for h in selected for sc in common_eval_table[h]],
        "random_partial": [sc for h in random_view for sc in common_eval_table[h]],
        "independent_full": list(full_survivors),
    }
    try:
        structure_subset = build_t0_structure_subset(
            endpoints, config.q_t0, config.structure_subsample_seed, _t0_endpoint_id
        )
    except ValueError as exc:
        raise ValueError(f"{protein_id}: {exc} (Q_T0 subsets must be equal-sized)") from exc

    # 6. the reserved 3*Q_T0 structure spend, charged per request. This is what T0 buys: a
    #    per-policy structure pass rate over EQUAL-SIZED subsets.
    structure_results = []
    for policy, endpoints in structure_subset.items():
        for rank, endpoint in enumerate(endpoints):
            endpoint_id = _t0_endpoint_id(endpoint)
            request_id = f"{protein_id}:{rho_id}:{policy}:{rank}"
            outcome = _coerce_structure_outcome(structure_gate(endpoint))
            structure_results.append(
                T0StructureResult(
                    policy=policy, request_id=request_id, source_endpoint_id=endpoint_id,
                    subset_rank=rank, evaluated=outcome.evaluated, feasible=outcome.feasible,
                    cache_status=outcome.cache_status, model_executed=outcome.model_executed,
                    failure_reason=outcome.failure_reason,
                )
            )
            deferred = not outcome.evaluated
            ledger.append(LedgerEvent(
                event_id=request_id, protein_id=protein_id, arm=arm, phase="initial_refold",
                attempt_id=journal.attempt_id(request_id),
                status="deferred" if deferred else "ok",
                structure_requests=0 if deferred else 1,
                structure_cache_hits=1 if outcome.cache_status == "hit" else 0,
                walltime_s=float(outcome.walltime_s),
            ))

    # T0's ONLY structure spend is this subsample, and GO/KILL condition 3 is a per-policy pass
    # rate over it. If nothing was evaluated the condition is unanswerable, so the point must not
    # report the status the cohort treats as success -- the P1 arms make the same distinction with
    # `entry_complete_structure_deferred`.
    evaluated_any = any(r.evaluated for r in structure_results)
    status = "t0_complete" if evaluated_any else "t0_structure_deferred"

    return T0Result(
        protein_id=protein_id, unique_root_hashes=unique_hashes,
        root_values=tuple(root_values), common_eval_table=common_eval_table,
        policy_views=policy_views, full_trajectories=tuple(full_outcomes),
        matched_full_trajectories=matched_full, structure_subset=structure_subset,
        structure_results=tuple(structure_results),
        structure_requests=t0_reserved_structure_requests(config.q_t0),
        reserved_dfe=reserved_dfe, ledger=_own_ledger(), status=status, rho_id=rho_id,
        root_attempts=attempts, unique_roots=unique_roots,
        maturity=tuple(_maturity_record(p) for p in unique_roots),
        est_scored_by_root=est_by_root, eval_scored_by_root=eval_by_root, full_pool=full_pool,
    )


def _t0_endpoint_id(endpoint) -> str:
    """Head-INDEPENDENT content id for the Q_T0 subsample: never the risk value."""
    continuation = getattr(endpoint, "continuation", None)
    if continuation is not None:
        return str(continuation.continuation_id)
    return f"full:{int(endpoint.replicate_index)}:{int(endpoint.seed)}"


def _scored_set(payload, set_tag, k, completer, head_fn, expected_length,
                protein_id, arm, seed_ctx, journal):
    """Draw and Head-score one root's ``k`` continuations of ``set_tag``, charging each."""
    continuations = []
    for r in range(k):
        seed = seed_ctx.continuation_seed(payload.root_equivalence_hash, set_tag, r)
        outcome = completer(CompletionRequest(payload, set_tag, r, seed))
        # The independent-full control this branch is matched against IS tail-validated
        # (`_validate_trajectory_outcome`). Leaving the partial branch unchecked would let the
        # ledger book a tail cost the run never paid and read as a compute difference between the
        # two things T0 compares.
        _validate_completion(outcome, payload, set_tag)
        continuations.append(Continuation(
            continuation_id=f"{payload.root_equivalence_hash}:{set_tag}:{r}",
            root_equivalence_hash=payload.root_equivalence_hash, set_tag=set_tag, seed=seed,
            sequence=outcome.sequence, replicate_index=r,
        ))
        base = f"{protein_id}:{arm}:{set_tag}:{payload.root_equivalence_hash}:{r}"
        journal.ledger.append(LedgerEvent(
            event_id=base, protein_id=protein_id, arm=arm, phase=set_tag,
            attempt_id=journal.attempt_id(base), status="ok",
            logical_dfe=outcome.logical_dfe, head_samples=1,
            physical_forward_calls=outcome.physical_forward_calls,
            completion_cache_hits=int(outcome.cache_hit), walltime_s=outcome.walltime_s,
        ))
    return evaluate_continuations(head_fn, tuple(continuations), expected_length)
