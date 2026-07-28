"""V1F6 model-free preflight and print-config (PLAN_RF_REFINE_FUSION_V1 §3.4, §4.1-4.2).

Everything here resolves WITHOUT loading a model: config resolution, the worst-case cost
projection (§3.4), the seed-stream inventory, and the assembled ``--print-config`` payload. Imports
are restricted to the pure ``fusion`` package so the entry driver can print/validate on a login
node. Production knob values are never invented here -- they come from the resolved entry config and
the terminal v0 Fusion params supplied by the caller.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from inverse_folding.reference_flow.fusion.v1_config import V1EntryConfig


def resolve_entry_config(data: Mapping) -> V1EntryConfig:
    """Resolve a fail-fast :class:`V1EntryConfig` from a raw mapping (YAML/JSON)."""
    return V1EntryConfig.from_mapping(data)


# --------------------------------------------------------------------------- #
# V1-A null-runtime firewall (§2.12)
# --------------------------------------------------------------------------- #
#: The one amplification form whose ``g(h)`` is independent of h. Every other form makes the
#: unmasking schedule a function of the position-dependent signal, which is the pathway V1-A freezes.
_NULL_AMPLIFICATION_FORM = "constant_one"


@dataclass(frozen=True)
class NullKernelReport:
    """What the firewall actually verified, carried into ``--print-config`` / ``--dry-run`` output
    and the run provenance so an audit reads the VERIFIED runtime, not the intended one."""

    rf_sampler_config: str
    rf_sampler_digest: str
    entry_rf_config_label: str
    amplification_form: str
    h_shuffle_enabled: bool
    controller_enabled: bool
    h_maps_present: bool
    #: The v0 REPAIR kernel is a distinct §2.12 identity that must ALSO be the null kernel: it runs
    #: inside every one of the n_rounds repair rounds, so a position-dependent or controller-driven
    #: repair YAML makes the whole downstream a different method.
    repair_rf_config: str | None = None
    repair_rf_digest: str | None = None


@dataclass(frozen=True)
class ResolvedRuntimeParams:
    """The numbers that define "matched compute", read from the FROZEN YAMLs.

    ``s_steps`` comes from the entry RF sampler config; ``n_population`` / ``r_parent`` /
    ``n_rounds`` from the frozen v0 Fusion config. They are never taken from a CLI flag: S alone
    scales every budget term (``dfe_prefix = B*S``, ``M_T = floor(C_reserved / S)``), so a flag that
    disagrees with the YAML mis-sizes the whole matched budget while still passing the gate.
    """

    s_steps: int
    n_population: int
    r_parent: int
    n_rounds: int
    rf_sampler_config: str
    fusion_config: str


def resolve_runtime_params(*, rf_sampler_config, fusion_config) -> ResolvedRuntimeParams:
    """Resolve S / N / R_parent / n_rounds from the two frozen configs (model-free)."""
    from inverse_folding.reference_flow.fusion.config import load_fusion_config

    sampler = _load_sampler_payload(rf_sampler_config)
    sampler_section = sampler.get("sampler")
    if not isinstance(sampler_section, dict) or "n_steps" not in sampler_section:
        raise ValueError(
            f"entry sampler config {rf_sampler_config} has no sampler.n_steps; S is the unit of "
            "every budget term and must be declared, never defaulted"
        )
    s_steps = int(sampler_section["n_steps"])
    if s_steps <= 0:
        raise ValueError(f"sampler.n_steps must be positive, got {s_steps}")
    fusion_path = Path(fusion_config)
    if not fusion_path.is_file():
        raise ValueError(f"frozen Fusion config not found: {fusion_path}")
    fusion = load_fusion_config(fusion_path)
    return ResolvedRuntimeParams(
        s_steps=s_steps,
        n_population=int(fusion.core.population_size),
        r_parent=int(fusion.structure.max_refolds_per_parent),
        n_rounds=int(fusion.core.n_rounds),
        rf_sampler_config=str(rf_sampler_config), fusion_config=str(fusion_path),
    )


def assert_cli_matches_resolved(
    resolved: ResolvedRuntimeParams, *, s_steps=None, r_parent=None, n_rounds=None,
    entry_n_population=None,
) -> None:
    """Cross-check every CLI-declared budget parameter against the frozen YAMLs (YAML is
    authoritative, the CLI is a cross-check). A disagreement is a hard failure naming both values:
    it means the operator believes a different experiment is running than the configs define."""
    mismatches = []
    for name, cli, frozen in (
        ("--s-steps", s_steps, resolved.s_steps),
        ("--r-parent", r_parent, resolved.r_parent),
        ("--n-rounds", n_rounds, resolved.n_rounds),
    ):
        if cli is not None and int(cli) != int(frozen):
            mismatches.append(f"{name}={cli} but the frozen config says {frozen}")
    if entry_n_population is not None and int(entry_n_population) != resolved.n_population:
        # The entry facade is sized for N and v0 admits N; two different Ns means the facade is
        # built for a population the terminal stage will never form.
        mismatches.append(
            f"entry config n_population={entry_n_population} but the frozen Fusion config's "
            f"population_size is {resolved.n_population}"
        )
    if mismatches:
        raise ValueError(
            "launch parameters disagree with the frozen configs (§4.1): " + "; ".join(mismatches)
        )


def _load_sampler_payload(path) -> dict:
    """Read the entry sampler YAML as a raw mapping. Deliberately does NOT go through
    ``load_reference_flow_config``: the firewall must refuse a kernel whose declared sections it
    does not recognise (e.g. a ``controller`` block), and a typed loader would drop them."""
    import yaml

    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(
            f"entry sampler config not found: {resolved} (the V1-A entry kernel identity is the "
            "CONTENT of this file; it cannot be assumed)"
        )
    payload = yaml.safe_load(resolved.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"entry sampler config {resolved} must be a YAML mapping")
    return payload


def assert_null_entry_kernel(args, config) -> NullKernelReport:
    """Refuse any entry runtime that is not the frozen V1-A null kernel (PLAN §2.12).

    V1-A attributes its effect to PRE-TERMINAL REWARD ALLOCATION. Two independent pathways would
    make the entry schedule position-dependent and confound that attribution, and both are refused
    here rather than merely left unconfigured:

    (A) static amplification -- ``g = amplification_factor(h, ...)`` is computed once and scales the
        per-position unmask probabilities. Only ``constant_one`` gives ``g == 1`` for any h.
    (B) online control -- a ``controller.step(ctx)`` returning corrected logits.

    Any h-map is refused outright: V1-A never loads one, so a supplied path can only mean the
    caller believes a different method is running.

    Model-free by construction (raw YAML + hashing only), so it runs BEFORE torch is imported on
    both the ``--dry-run`` and the real launch path.
    """
    rf_path = getattr(args, "rf_sampler_config", None)
    if not rf_path:
        raise ValueError(
            "--rf-sampler-config is required: the V1-A entry kernel must be verified as the null "
            "kernel before any generation, and an unverified kernel is not a runnable identity"
        )
    for arg_name, label in (("h_map", "h-map"), ("h_maps_parquet", "h-map"),
                            ("controller_config", "controller config"),
                            ("h_corpus_stats", "h-map corpus stats")):
        supplied = getattr(args, arg_name, None)
        if supplied:
            raise ValueError(
                f"V1-A refuses a {label} ({arg_name}={supplied!r}): the entry runtime is frozen to "
                "g(h)==1 with controller=None, so position-dependent inputs are out of scope and "
                "must not be silently inert"
            )
    # Defense in depth: the config layer already pins these sentinels, but the firewall is the gate
    # that must hold even if a later config revision stops enforcing them.
    if getattr(config, "controller_enabled", False):
        raise ValueError("entry config declares controller_enabled=True; V1-A runs controller=None")
    if getattr(config, "h_maps_present", False):
        raise ValueError("entry config declares h_maps_present=True; V1-A consumes no h-map")

    form, digest = _assert_null_rf_kernel(rf_path, role="entry")
    # §2.12: entry and repair are DISTINCT identities that must BOTH be the null kernel. The repair
    # kernel runs inside v0 for every one of n_rounds rounds, so a position-dependent or
    # controller-driven repair YAML makes the whole downstream a different method -- content
    # digesting it (which the provenance does) records WHICH file ran, not whether it was valid.
    repair_path = getattr(args, "terminal_repair_config", None)
    repair_digest = None
    if repair_path:
        _, repair_digest = _assert_null_rf_kernel(repair_path, role="terminal repair")
    return NullKernelReport(
        rf_sampler_config=str(rf_path), rf_sampler_digest=digest,
        entry_rf_config_label=str(getattr(config, "entry_rf_config", "")),
        amplification_form=form, h_shuffle_enabled=False,
        controller_enabled=False, h_maps_present=False,
        repair_rf_config=str(repair_path) if repair_path else None,
        repair_rf_digest=repair_digest,
    )


def _assert_null_rf_kernel(path, *, role: str) -> tuple[str, str]:
    """Verify ONE reference-flow YAML is the frozen null kernel. Returns ``(form, digest)``."""
    payload = _load_sampler_payload(path)
    if "controller" in payload:
        raise ValueError(
            f"{role} RF config {path} declares a 'controller' section; V1-A passes controller=None "
            "and must not run a kernel that expects online correction"
        )
    amplification = payload.get("amplification")
    if not isinstance(amplification, dict) or "form" not in amplification:
        raise ValueError(
            f"{role} RF config {path} has no amplification.form; the null kernel must be declared "
            f"explicitly as {_NULL_AMPLIFICATION_FORM!r}, never defaulted"
        )
    form = str(amplification["form"])
    if form != _NULL_AMPLIFICATION_FORM:
        raise ValueError(
            f"{role} amplification.form={form!r} is a function of h; V1-A requires "
            f"{_NULL_AMPLIFICATION_FORM!r} so g == 1 at every position. A position-dependent "
            "schedule would confound the pre-terminal allocation effect this study measures"
        )
    h_shuffle = payload.get("h_shuffle") or {}
    if bool(h_shuffle.get("enabled", False)):
        raise ValueError(
            f"{role} RF config {path} enables h_shuffle; it is inert under "
            f"{_NULL_AMPLIFICATION_FORM!r} but declares a position-dependent study, and a "
            "contradictory declaration is refused rather than silently ignored"
        )
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:32]
    return form, f"rfcfg-{digest}"


def entry_config_digest(config: V1EntryConfig) -> str:
    """Content digest of the resolved entry config. Single source of truth shared with the cohort
    resume identity (``run_sig``), so a printed digest matches the digest a shard binds to."""
    return hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TerminalFusionParams:
    """Terminal v0 Fusion params needed for the §3.4 projection, sourced from the frozen terminal
    repair config (never invented here): ``s_steps`` = entry sampler step count S,
    ``r_parent`` = ``structure.max_refolds_per_parent``, ``n_rounds`` = v0 Fusion rounds. Optional
    ``seconds_per_refold`` / ``seconds_per_dfe`` are measured unit costs (T0-measured, or explicit
    runbook anchors) used to predict GPU/walltime; when absent the launch gate cannot verify
    walltime and fails closed."""

    s_steps: int
    r_parent: int
    n_rounds: int
    seconds_per_refold: float | None = None
    seconds_per_dfe: float | None = None

    def __post_init__(self) -> None:
        for name in ("s_steps", "r_parent", "n_rounds"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        for name in ("seconds_per_refold", "seconds_per_dfe"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or value <= 0):
                raise ValueError(f"{name} must be a positive number or None, got {value!r}")


@dataclass(frozen=True)
class BudgetProjection:
    n_proteins: int
    dfe_prefix: int
    dfe_est: int
    dfe_final: int
    reserved_dfe_per_protein: int
    reserved_dfe_total: int
    reserved_refold_per_protein: int
    reserved_refold_total: int
    predicted_refold_gpu_s: float | None
    predicted_walltime_s: float | None
    dfe_cap: int
    refold_cap: int
    walltime_cap_s: float
    dfe_within_cap: bool
    refold_within_cap: bool
    walltime_within_cap: bool | None  # None => not predictable without unit costs
    #: Terminal arm only: M_T complete trajectories derived from the persisted reservation.
    terminal_trajectories: int | None = None
    #: T0 only: the held-out evaluation DFE, the compute-matched independent-full control, and the
    #: 3*Q_T0 structure reservation. Reported as their own terms because a single total hides WHICH
    #: term blew the cap (§3.4). ``dfe_full_control`` is deliberately NOT folded into ``dfe_final``:
    #: T0 materializes no parent at all, and a non-zero ``dfe_final`` would say it does.
    dfe_eval: int = 0
    dfe_full_control: int = 0
    t0_structure_requests: int | None = None


def terminal_trajectory_count(
    *, reserved_dfe_per_protein: int, s_steps: int, facade_cap: int
) -> int:
    """``M_T = floor(C_reserved / S)`` -- the Terminal arm's complete-trajectory count (§3.2.1).

    The Terminal arm does not choose its own budget: it spends the SAME per-protein DFE the
    pre-terminal arm reserved, which is what makes "matched compute" checkable rather than
    asserted. ``M_T < facade_cap`` is a hard failure: ``F_cap`` is frozen and common to both arms,
    so a short Terminal pool can only mean a DFE accounting / off-by-one error. Shrinking the
    facade instead would hand the two arms different initial-refold budgets -- the exact confound
    this design exists to remove.
    """
    for name, value in (("reserved_dfe_per_protein", reserved_dfe_per_protein),
                        ("s_steps", s_steps), ("facade_cap", facade_cap)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    m_t = int(reserved_dfe_per_protein) // int(s_steps)
    if m_t < int(facade_cap):
        raise ValueError(
            f"M_T={m_t} < F_cap={facade_cap} (C_reserved={reserved_dfe_per_protein}, S={s_steps}): "
            "this is a DFE accounting/off-by-one error (§3.2.1), not a reason to shrink the "
            "Terminal facade -- both arms must attempt at most F_cap initial refolds"
        )
    return m_t


def project_budget(
    config: V1EntryConfig, *, n_proteins: int, terminal: TerminalFusionParams,
    reserved_dfe_per_protein: int | None = None, reserved_dfe_by_protein=None,
) -> BudgetProjection:
    """Worst-case cost projection (§3.4). Every prefix attempt is charged the full S; every unique
    root gets K_EST completions and the facade gets one fresh final per attempt-cap slot, each with
    the worst-case tail S. The reserved refold cap follows the §3.4 formula
    ``F_cap + N * R_parent * n_rounds``. Cache hits reduce real executions but never this cap.

    Both P1 arms are projected here. The TERMINAL arm carries no prefix/K_EST fields at all: its
    trajectory count is derived from the persisted pre-terminal reservation
    (``M_T = floor(C_reserved / S)``, §3.2.1), which is what makes the two arms compute-matched by
    construction rather than by assertion. T0 dispatches to :func:`_project_t0_budget`, which adds
    the ``K_EVAL`` completions, the ``3 * Q_T0`` structure requests, the compute-matched full
    control, and the ``len(rho_grid)`` grid factor.
    """
    if not isinstance(n_proteins, int) or isinstance(n_proteins, bool) or n_proteins <= 0:
        raise ValueError(f"n_proteins must be a positive int, got {n_proteins!r}")
    s = terminal.s_steps
    if config.phase == "t0":
        if reserved_dfe_per_protein is not None:
            raise ValueError("T0 computes its own reservation; reserved_dfe_per_protein is a "
                             "Terminal-arm input")
        return _project_t0_budget(config, n_proteins=n_proteins, terminal=terminal)
    if config.entry_arm == "terminal":
        if reserved_dfe_by_protein and reserved_dfe_per_protein is None:
            # The invariant is per protein, so the THINNEST reservation is what must satisfy
            # M_T >= F_cap; the cohort TOTAL is then summed over all of them.
            reserved_dfe_per_protein = min(int(v) for v in reserved_dfe_by_protein.values())
        return _project_terminal_budget(
            config, n_proteins=n_proteins, terminal=terminal,
            reserved_dfe_by_protein=reserved_dfe_by_protein,
            reserved_dfe_per_protein=reserved_dfe_per_protein,
        )
    if reserved_dfe_per_protein is not None:
        # The pre-terminal arm PRODUCES the reservation the Terminal arm consumes; being handed
        # one means the dependency is backwards and the two arms could silently diverge.
        raise ValueError(
            "the pre-terminal arm computes its own reservation; reserved_dfe_per_protein is a "
            "Terminal-arm input (§3.2.1)"
        )
    dfe_prefix = config.prefix_attempts * s
    # Worst case |U| == B (every prefix attempt yields a distinct root), and estimator work is
    # allocated over ALL of U -- so the reservation must be sized by B, not by the capacity gate.
    dfe_est = config.k_est * config.prefix_attempts * s
    dfe_final = config.initial_refold_attempt_cap * s
    reserved_dfe_pp = dfe_prefix + dfe_est + dfe_final
    reserved_refold_pp = (
        config.initial_refold_attempt_cap
        + config.n_population * terminal.r_parent * terminal.n_rounds
    )
    return _assemble_projection(
        config, terminal=terminal, n_proteins=n_proteins,
        dfe_prefix=dfe_prefix, dfe_est=dfe_est, dfe_final=dfe_final,
        reserved_dfe_pp=reserved_dfe_pp, reserved_refold_pp=reserved_refold_pp,
    )


def _project_terminal_budget(
    config: V1EntryConfig, *, n_proteins: int, terminal: TerminalFusionParams,
    reserved_dfe_per_protein: int | None, reserved_dfe_by_protein=None,
) -> BudgetProjection:
    """Terminal arm (§3.2.1). It generates ``M_T = floor(C_reserved / S)`` independent complete
    trajectories, ranks them by exact terminal Head, and submits only the top ``F_cap`` to the same
    v0 initial-refold admission. Its refold reservation therefore has the SAME shape as the
    pre-terminal arm's -- the arms differ in WHERE selection happens, not in what they may spend.

    ``reserved_dfe_by_protein`` supplies the REAL cohort total: every protein runs ``M_T`` out of
    its OWN ``C_reserved``, so ``per_protein * n_proteins`` is only correct for a homogeneous
    cohort and under-reports otherwise. The scalar still drives the per-protein invariant
    (``M_T >= F_cap`` must hold for the thinnest protein), and the cap verdicts below are computed
    from whichever total is authoritative -- never from one and reported as the other.
    """
    if reserved_dfe_per_protein is None:
        raise ValueError(
            "the Terminal arm needs the persisted pre-terminal reserved DFE per protein: its "
            "trajectory count is M_T = floor(C_reserved / S) (§3.2.1), and without it 'matched "
            "compute' -- the claim the whole comparison rests on -- cannot be verified"
        )
    f_cap = config.initial_refold_attempt_cap
    m_t = terminal_trajectory_count(
        reserved_dfe_per_protein=int(reserved_dfe_per_protein),
        s_steps=terminal.s_steps, facade_cap=f_cap,
    )
    # Spend exactly what the trajectories cost; the floor remainder is recorded, never handed over.
    reserved_dfe_pp = m_t * terminal.s_steps
    reserved_refold_pp = f_cap + config.n_population * terminal.r_parent * terminal.n_rounds
    total_override = None
    if reserved_dfe_by_protein:
        total_override = sum(
            terminal_trajectory_count(
                reserved_dfe_per_protein=int(value), s_steps=terminal.s_steps, facade_cap=f_cap,
            ) * terminal.s_steps
            for value in reserved_dfe_by_protein.values()
        )
    return _assemble_projection(
        config, terminal=terminal, n_proteins=n_proteins,
        dfe_prefix=0, dfe_est=0, dfe_final=reserved_dfe_pp,
        reserved_dfe_pp=reserved_dfe_pp, reserved_refold_pp=reserved_refold_pp,
        terminal_trajectories=m_t, reserved_dfe_total_override=total_override,
    )


def _project_t0_budget(
    config: V1EntryConfig, *, n_proteins: int, terminal: TerminalFusionParams
) -> BudgetProjection:
    """T0 (§3.2): prefix + (K_EST + K_EVAL) unique-root tails, NO final materialization at all --
    T0 never builds a parent. Its structure spend is ``t0_reserved_structure_requests(Q_T0)`` =
    ``3 * Q_T0``, one Q_T0 subsample per policy VIEW; the dropped fourth policy would have made it
    4*Q_T0 and over-reserved every T0 launch.

    Two multipliers are load-bearing, and each one, if dropped, turns the §3.4 gate into a no-op --
    a run blows its declared ``max_dfe``/``max_refolds`` and still exits 0, because the projection
    is the ONLY place those caps are checked:

    * **the whole grid runs per protein.** One T0 shard executes every ``rho_grid`` point into one
      checkpoint (separate matched root-attempt groups, §2.3:238-241), so every per-protein term
      scales with ``len(rho_grid)``.
    * **the compute-matched control is real spend.** Each grid point additionally generates
      ``floor(C_reserved / S)`` independent complete trajectories -- that control IS what condition
      3 compares against. Its cost is ``matched_full * S``, i.e. the reservation again minus the
      floor remainder, so the worst case is a second ``reserved_dfe`` per point.
    """
    from inverse_folding.reference_flow.fusion.v1_alloc import t0_reserved_structure_requests

    s = terminal.s_steps
    n_points = max(1, len(config.rho_grid or ()))
    dfe_prefix = config.prefix_attempts * s
    # sized by B, not by the capacity gate: worst case every prefix attempt yields a distinct root
    # and estimator/eval work is allocated over ALL of U.
    dfe_est = config.k_est * config.prefix_attempts * s
    dfe_eval = config.k_eval * config.prefix_attempts * s
    partial_pp = dfe_prefix + dfe_est + dfe_eval
    dfe_full_control = partial_pp  # matched_full * S <= reserved_dfe, by construction
    structure_requests = t0_reserved_structure_requests(config.q_t0) * n_points
    return _assemble_projection(
        config, terminal=terminal, n_proteins=n_proteins,
        dfe_prefix=dfe_prefix * n_points, dfe_est=dfe_est * n_points,
        dfe_final=0,  # T0 materializes no parent -- a non-zero term here would say it does
        reserved_dfe_pp=(partial_pp + dfe_full_control) * n_points,
        reserved_refold_pp=structure_requests,
        dfe_eval=dfe_eval * n_points, dfe_full_control=dfe_full_control * n_points,
        t0_structure_requests=structure_requests,
    )


def _assemble_projection(
    config: V1EntryConfig, *, terminal: TerminalFusionParams, n_proteins: int,
    dfe_prefix: int, dfe_est: int, dfe_final: int,
    reserved_dfe_pp: int, reserved_refold_pp: int, terminal_trajectories: int | None = None,
    dfe_eval: int = 0, dfe_full_control: int = 0, t0_structure_requests: int | None = None,
    reserved_dfe_total_override: int | None = None,
) -> BudgetProjection:
    """Shared cap/walltime assembly, so both arms are judged by exactly the same gate.

    ``reserved_dfe_total_override`` is for a heterogeneous cohort whose true total is not
    ``per_protein * n``. It flows through the SAME cap and walltime computation below, so the
    reported total and the pass/fail verdict can never disagree.
    """
    reserved_dfe_total = (reserved_dfe_pp * n_proteins if reserved_dfe_total_override is None
                          else int(reserved_dfe_total_override))
    reserved_refold_total = reserved_refold_pp * n_proteins

    # A walltime prediction needs EVERY reserved cost term. Treating a missing per-DFE unit cost as
    # zero would under-predict (a 120-DFE reservation would "cost" only its refold seconds) and let
    # an unverifiable budget pass the gate, so both unit costs are required whenever the
    # corresponding reserved quantity is non-zero.
    predicted_refold_gpu_s = predicted_walltime_s = walltime_within_cap = None
    have_refold_cost = terminal.seconds_per_refold is not None or reserved_refold_total == 0
    have_dfe_cost = terminal.seconds_per_dfe is not None or reserved_dfe_total == 0
    if have_refold_cost and have_dfe_cost:
        predicted_refold_gpu_s = reserved_refold_total * (terminal.seconds_per_refold or 0.0)
        dfe_seconds = reserved_dfe_total * (terminal.seconds_per_dfe or 0.0)
        predicted_walltime_s = predicted_refold_gpu_s + dfe_seconds
        walltime_within_cap = predicted_walltime_s <= config.max_walltime_s

    return BudgetProjection(
        n_proteins=n_proteins, dfe_prefix=dfe_prefix, dfe_est=dfe_est, dfe_final=dfe_final,
        reserved_dfe_per_protein=reserved_dfe_pp, reserved_dfe_total=reserved_dfe_total,
        reserved_refold_per_protein=reserved_refold_pp,
        reserved_refold_total=reserved_refold_total,
        predicted_refold_gpu_s=predicted_refold_gpu_s, predicted_walltime_s=predicted_walltime_s,
        dfe_cap=config.max_dfe, refold_cap=config.max_refolds,
        walltime_cap_s=config.max_walltime_s,
        dfe_within_cap=reserved_dfe_total <= config.max_dfe,
        refold_within_cap=reserved_refold_total <= config.max_refolds,
        walltime_within_cap=walltime_within_cap,
        terminal_trajectories=terminal_trajectories,
        dfe_eval=dfe_eval, dfe_full_control=dfe_full_control,
        t0_structure_requests=t0_structure_requests,
    )


def assert_launch_feasible(projection: BudgetProjection) -> None:
    """Fail-closed launch gate (§3.4): raise if the worst-case DFE, refolds, or walltime exceed the
    submitted caps. Walltime that cannot be predicted (no measured unit cost supplied) is itself a
    hard failure -- the gate never passes an unverifiable budget."""
    violations = []
    if not projection.dfe_within_cap:
        violations.append(
            f"reserved DFE {projection.reserved_dfe_total} > cap {projection.dfe_cap}"
        )
    if not projection.refold_within_cap:
        violations.append(
            f"reserved refolds {projection.reserved_refold_total} > cap {projection.refold_cap}"
        )
    if projection.walltime_within_cap is None:
        violations.append(
            "walltime unverifiable: a reserved cost term has no measured unit cost "
            "(seconds_per_refold / seconds_per_dfe). §3.4 requires a predicted walltime, not a "
            "bare cap echo, and a missing unit cost must never be treated as zero"
        )
    elif not projection.walltime_within_cap:
        violations.append(
            f"predicted walltime {projection.predicted_walltime_s:.1f}s > cap "
            f"{projection.walltime_cap_s}s"
        )
    if violations:
        raise ValueError("launch infeasible (§3.4): " + "; ".join(violations))


def seed_stream_summary(config: V1EntryConfig) -> dict:
    """Pre-run seed-stream inventory: namespace -> realized count. Realized seed VALUES need the
    per-run unique-root hashes, but the counts and master seed are resolvable without a run.

    V1-A has no control-enable switches: the T0 policy views are declared by ``t0_policies`` and
    share ONE root pool and ONE held-out evaluation table, so a policy label never opens its own
    continuation stream. Only membership is separately seeded."""
    summary: dict = {"master_seed": config.master_seed}
    if config.entry_arm == "terminal":
        # A terminal arm draws independent complete trajectories; it has no root prefix, no
        # estimator set, and no fresh final materialization. The trajectory count is derived from
        # the reserved pre-terminal cap (§3.2.1), not configured here.
        summary["terminal_complete"] = None  # resolved from the persisted reserved cap
        return summary
    summary["root"] = config.prefix_attempts
    summary["est"] = config.k_est * config.unique_root_capacity
    if config.phase == "p1":
        # one fresh final materialization per ranked root, only in the P1 pre-terminal facade.
        summary["final"] = config.initial_refold_attempt_cap
    else:
        summary["eval"] = config.k_eval * config.unique_root_capacity
        # a single membership draw picks WHICH roots the random view holds; it never redraws
        # continuations, so selected/random remain views over the same eval table.
        summary["random_membership"] = 1
    return summary


def render_print_config(
    config: V1EntryConfig, *, n_proteins: int, terminal: TerminalFusionParams,
    requested_proteins=None, command: str | None = None,
    reserved_dfe_per_protein: int | None = None, budget: "BudgetProjection | None" = None,
) -> dict:
    """Assemble the fully-resolved ``--print-config`` payload: identity, config digest, method
    values, terminal params, worst-case budget, and seed inventory. No model is loaded.

    The Terminal arm's budget cannot be computed without ``C_reserved``, so either the caller passes
    the already-projected ``budget`` or the reservation itself. Recomputing here without it made
    ``--print-config`` -- the one pre-launch audit a cluster operator has -- unusable for the arm
    whose budget most needs auditing.
    """
    if budget is None:
        budget = project_budget(
            config, n_proteins=n_proteins, terminal=terminal,
            reserved_dfe_per_protein=reserved_dfe_per_protein,
        )
    return {
        "schema_version": config.schema_version, "campaign_id": config.campaign_id,
        "phase": config.phase, "entry_arm": config.entry_arm, "split_role": config.split_role,
        "config_digest": entry_config_digest(config), "nmp_absent": True,
        "method_values": {
            "rho_target": config.rho_target, "rho_grid": config.rho_grid,
            "prefix_attempts": config.prefix_attempts, "k_est": config.k_est,
            "k_eval": config.k_eval, "unique_root_capacity": config.unique_root_capacity,
            "n_population": config.n_population,
            "initial_refold_attempt_cap": config.initial_refold_attempt_cap,
            "estimator": config.estimator, "backbone_only": config.backbone_only,
            "entry_rf_config": config.entry_rf_config,
            "t0_policies": config.t0_policies,
            # the frozen V1-A null entry runtime, printed so a launch can be audited before it runs
            "controller_enabled": config.controller_enabled,
            "h_maps_present": config.h_maps_present,
        },
        "terminal_fusion": asdict(terminal),
        "budget": asdict(budget),
        "seed_inventory": seed_stream_summary(config),
        "requested_proteins": list(requested_proteins) if requested_proteins else None,
        "command": command,
    }
