"""V1F5 cohort runner and artifact aggregation (PLAN_RF_REFINE_FUSION_V1 §4.2, §5, §3.3).

A shard runs a subset of the requested cohort, writing one atomic per-``(protein, arm)``
checkpoint whose ``run_sig`` binds the resolved config and the protein's §4.2 input signature; a
matching checkpoint is reused, a changed one re-runs. Aggregation reads the checkpoints for the
FROZEN requested cohort under the current arm and emits the §5 artifact family with canonical,
shard/order-independent ordering, a coverage row for every requested protein (missing ones
surfaced), a single-counted cost ledger, and a manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from inverse_folding.reference_flow.fusion.v1_admission import build_cohort_coverage
from inverse_folding.reference_flow.fusion.v1_ledger import LedgerEvent, aggregate_ledger
from inverse_folding.reference_flow.fusion.v1_records import collapse_roots
from scripts.rf_fusion_v1_artifacts import (
    V1_TABLE_SCHEMAS,
    admission_rows,
    continuation_rows,
    coverage_rows,
    facade_rows,
    maturity_rows,
    partial_root_row,
    root_attempt_rows,
    root_selection_rows,
    root_value_rows,
    write_cost_ledger_jsonl,
    write_manifest,
    write_payload_sidecar,
    write_stable_parquet,
)
from scripts.rf_fusion_v1_entry_core import (
    EntryJournal,
    EntryOracles,  # re-exported for callers
    RootAttemptRequest,  # noqa: F401  (documents the seam)
    rho_id_for,
    run_entry_protein,
    run_t0_protein,
    run_terminal_protein,
)
from inverse_folding.reference_flow.fusion.v1_seeds import SeedContext, realized_seed_manifest
from scripts.rf_fusion_v1_preflight import entry_config_digest as _config_digest

__all__ = [
    "EntryOracles",
    "ENTRY_OK_STATUSES",
    "run_entry_shard",
    "aggregate_entry_artifacts",
]

#: Statuses that mean the protein actually produced a usable entry population.
#:
#: ``entry_insufficient_unique_roots`` and ``entry_insufficient`` RETURN a result rather than
#: raising, so "did not crash" is not a success criterion: a cohort where every protein failed the
#: capacity gate would otherwise report ``n_proteins_ok == n_proteins`` -- and the driver's exit
#: code (0 = the requested cohort is complete) is derived from exactly this number.
ENTRY_OK_STATUSES = frozenset(
    {"terminal_success", "entry_complete_structure_deferred", "t0_complete"}
)

# Row-set tables materialized per protein into the checkpoint (the rest of the §5 tables are
# driver-runtime evidence filled in the V1F6/T0 path; they are still emitted empty-with-schema).
_CHECKPOINT_TABLES = (
    "root_attempts", "partial_roots", "maturity_telemetry", "continuations", "root_values",
    "root_selection", "terminal_parent_facade", "terminal_parent_admission",
    "complete_entry_pool", "facade_collapse", "t0_control_membership", "t0_structure_subset",
    "t0_structure_results",
)


def _run_sig(config_digest: str, arm: str, protein_id: str, input_signature: str) -> str:
    payload = json.dumps(
        {"cfg": config_digest, "arm": arm, "protein": protein_id, "input": input_signature},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: The TERMINAL arm has no maturity, so its seed prefix carries this literal in the rho slot. It is
#: a distinct value (never a rho id), so a terminal stream can never collide with a maturity stream.
TERMINAL_RHO_ID = "terminal"


def _rho_id(config) -> str:
    return TERMINAL_RHO_ID if config.rho_target is None else rho_id_for(config.rho_target)


def _seed_ctx(config, protein_id: str, rho_id: str | None = None) -> SeedContext:
    """``rho_id`` is overridden per T0 grid point: each maturity is its own seed namespace, so two
    grid points can never draw the same continuations."""
    return SeedContext(
        seed_schema="v1seed-1", campaign_id=config.campaign_id, phase=config.phase,
        split_role=config.split_role, master_seed=config.master_seed, entry_arm=config.entry_arm,
        protein_id=protein_id, rho_id=rho_id or _rho_id(config),
        # T0 declares this; it enters ONLY the membership draw, so re-drawing the random control
        # leaves every continuation seed identical. Threading it here is what stops the declared
        # config field from being an inert knob.
        declared_membership_seed=getattr(config, "random_membership_seed", None),
    )


def _checkpoint_name(protein_id: str, arm: str, phase: str = "p1") -> str:
    """T0 declares ``entry_arm: preterminal`` -- its roots ARE pre-terminal roots -- so naming a
    checkpoint by arm alone makes a T0 run collide with the P1 pre-terminal arm's. The run_sig
    differs, so the collision is not a false REUSE: it is a silent OVERWRITE that destroys the
    per-protein ``C_reserved`` the Terminal arm reads. The phase is therefore part of the name."""
    return f"{protein_id}__{phase}__{arm}.json"


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True))
    os.replace(tmp, path)


def run_entry_shard(
    *,
    shard_proteins,
    expected_length_by_protein,
    input_signature_by_protein,
    config,
    oracles: EntryOracles,
    out_dir,
    resume: bool = True,
    n_trajectories_by_protein=None,
    s_steps: int | None = None,
) -> dict[str, str]:
    """Run one shard of the configured ARM and write one atomic checkpoint each. Returns
    ``{protein_id: status}``. A failed protein is recorded (never aborts the shard) so its
    coverage stays visible.

    The TERMINAL arm additionally needs ``n_trajectories_by_protein`` (``M_T`` derived from the
    persisted pre-terminal reservation) and ``s_steps``; they are required rather than defaulted,
    because a Terminal shard that invents its own budget is not compute-matched (§3.2.1)."""
    if len(set(shard_proteins)) != len(shard_proteins):
        raise ValueError("duplicate protein in shard")
    t0 = config.phase == "t0"
    if t0 and s_steps is None:
        raise ValueError("T0 needs s_steps: its independent_full control is compute-matched")
    terminal = config.entry_arm == "terminal"
    if terminal:
        if oracles.trajectory_generator is None:
            raise ValueError("the terminal arm needs EntryOracles.trajectory_generator")
        if not n_trajectories_by_protein or s_steps is None:
            raise ValueError(
                "the terminal arm needs n_trajectories_by_protein (M_T from the persisted "
                "pre-terminal reservation) and s_steps; it may not invent its own budget"
            )
        missing = [p for p in shard_proteins if p not in n_trajectories_by_protein]
        if missing:
            raise ValueError(f"no persisted M_T for {sorted(missing)}")
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    arm = config.entry_arm
    config_digest = _config_digest(config)

    statuses: dict[str, str] = {}
    for pid in shard_proteins:
        ckpt = ckpt_dir / _checkpoint_name(pid, arm, config.phase)
        run_sig = _run_sig(config_digest, arm, pid, input_signature_by_protein[pid])
        prior_ledger: list = []
        epoch = 0
        if resume and ckpt.exists():
            prev = json.loads(ckpt.read_text())
            if prev.get("run_sig") == run_sig and prev.get("ok"):
                statuses[pid] = prev.get("status", "terminal_success")
                continue
            if prev.get("run_sig") == run_sig:
                # A FAILED attempt under the same identity already burned real compute. Re-running
                # overwrites this checkpoint, so its ledger is carried forward and the retry runs in
                # the next epoch; otherwise that GPU time silently disappears from the totals and
                # "matched compute" stops being checkable across a crash (§3.3).
                prior_ledger = list(prev.get("ledger", []))
                epoch = int(prev.get("attempt_epoch", 0)) + 1
        journal = EntryJournal(attempt_epoch=epoch)
        try:
            if t0:
                # One checkpoint per protein covering the whole rho grid: the grid points share a
                # cohort and a budget, so a partially-written grid is not a usable calibration.
                points = [
                    run_t0_protein(
                        protein_id=pid, config=config,
                        expected_length=expected_length_by_protein[pid], rho_target=float(rho),
                        root_generator=oracles.root_generator, completer=oracles.completer,
                        head_fn=oracles.head_fn,
                        full_trajectory_generator=oracles.trajectory_generator,
                        structure_gate=oracles.structure_gate, s_steps=int(s_steps),
                        seed_ctx=_seed_ctx(config, pid, rho_id=rho_id_for(float(rho))),
                        journal=journal,
                    )
                    for rho in config.rho_grid
                ]
                payload = _t0_checkpoint(points, pid, config, out_dir, config_digest)
                payload["ledger"] = prior_ledger + payload["ledger"]
                # The grid points share a cohort and a budget, so a grid missing a maturity is not a
                # usable calibration. Deriving the status from `points[0]` reported the FIRST
                # point's outcome as the whole grid's: a grid whose later maturities all failed read
                # as `t0_complete`, ok=true, and the driver exited 0. The status is now the first
                # NON-complete point's, and every point's own outcome is persisted beside it.
                failed = [p for p in points if p.status != "t0_complete"]
                payload.update({
                    "run_sig": run_sig, "arm": arm, "config_digest": config_digest,
                    "attempt_epoch": epoch, "ok": not failed,
                    "status": "t0_complete" if not failed else failed[0].status,
                    "per_rho_status": {p.rho_id: p.status for p in points},
                })
                _write_json_atomic(ckpt, payload)
                statuses[pid] = payload["status"]
                continue
            if terminal:
                result = run_terminal_protein(
                    protein_id=pid, config=config,
                    expected_length=expected_length_by_protein[pid],
                    n_trajectories=int(n_trajectories_by_protein[pid]), s_steps=int(s_steps),
                    trajectory_generator=oracles.trajectory_generator,
                    head_fn=oracles.head_fn, structure_gate=oracles.structure_gate,
                    seed_ctx=_seed_ctx(config, pid), journal=journal,
                )
            else:
                result = run_entry_protein(
                    protein_id=pid, config=config,
                    expected_length=expected_length_by_protein[pid],
                    root_generator=oracles.root_generator, completer=oracles.completer,
                    head_fn=oracles.head_fn, structure_gate=oracles.structure_gate,
                    seed_ctx=_seed_ctx(config, pid), journal=journal,
                )
            payload = _result_to_checkpoint(result, out_dir, config, config_digest)
            payload["ledger"] = prior_ledger + payload["ledger"]
            # PLAN §2.6: persist EVERY realized seed, collision-checked. Enumerating them only in
            # a helper nothing calls means a collision surfaces as an unexplained duplicate
            # trajectory instead of a refused launch.
            payload["realized_seeds"] = _realized_seeds(result, config, _seed_ctx(config, pid))
            payload.update({"run_sig": run_sig, "arm": arm, "config_digest": config_digest,
                            "attempt_epoch": epoch, "ok": True, "status": result.status,
                            # C_reserved is the Terminal shard's only input (§3.2.1); it is
                            # persisted per protein WITH its derivation inputs so an independent
                            # shard can verify the budget it was handed rather than trust it.
                            "reservation": (asdict(result.reservation)
                                            if result.reservation is not None else None)})
        except Exception as exc:  # noqa: BLE001 — a failed protein must not abort the shard
            # preserve already-paid work from the journal (§3.3: never silently drop paid work).
            payload = _failed_checkpoint(journal, pid, config, arm, terminal=terminal)
            payload["ledger"] = prior_ledger + payload["ledger"]
            payload.update({"run_sig": run_sig, "arm": arm, "config_digest": config_digest,
                            "attempt_epoch": epoch, "ok": False, "status": "failed",
                            "failure": repr(exc)})
        _write_json_atomic(ckpt, payload)
        statuses[pid] = payload["status"]
    return statuses


def _t0_checkpoint(points, protein_id: str, config, out_dir, config_digest) -> dict:
    """Emit the T0 evidence chain.

    The three ``t0_*`` policy tables answer "who held what". They are NOT sufficient on their own:
    every one of the four runbook §6.1 GO/KILL conditions is computed from the STANDARD tables --
    ``root_values`` + ``continuations`` for conditions 1-2, ``complete_entry_pool`` +
    ``t0_structure_results`` for 3, ``maturity_telemetry`` + ``continuations`` for 4. A T0 campaign
    that emitted only membership would spend its whole budget and exit 0 with evidence nobody can
    interpret, so runbook §9:442-451 requires these of "every T0/P1 run" and they are written here.

    Each grid point is a separate matched root-attempt group, so rows are namespaced by ``rho_id``
    and the per-point tables are concatenated rather than merged.

    ``t0_control_membership`` records FULL membership on every row (not just the row's own policy)
    and carries ``common_eval_provenance`` -- the continuation id the endpoint resolves to in the
    SHARED eval table -- so a reviewer reading the parquet can SEE that selected and random point
    at the same evidence rather than having to trust it.
    """
    membership, subset_rows, result_rows = [], [], []
    attempt_rows, partial_rows, maturity, continuations = [], [], [], []
    value_rows, selection_rows, full_pool_rows = [], [], []
    reserved_by_rho: dict = {}
    seeds: dict = {}
    for point in points:
        views = point.policy_views
        if not views:
            continue
        in_subset = {
            policy: {_t0_endpoint_key(e) for e in endpoints}
            for policy, endpoints in point.structure_subset.items()
        }
        member_of = {policy: set() for policy in views}
        provenance = {}
        for policy in ("selected_partial", "random_partial"):
            for root_hash, rows in views[policy].eval_rows.items():
                for scored in rows:
                    key = scored.continuation.continuation_id
                    member_of[policy].add(key)
                    provenance[key] = key  # the SHARED table's own id, identical for both views
        for i in range(views["independent_full"].n_trajectories):
            member_of["independent_full"].add(f"full:{i}")
        for policy, keys in member_of.items():
            for key in sorted(keys):
                membership.append({
                    "protein_id": protein_id, "endpoint_id": f"{point.rho_id}:{key}",
                    "policy": policy,
                    "selected_partial_member": key in member_of["selected_partial"],
                    "random_partial_member": key in member_of["random_partial"],
                    "independent_full_member": key in member_of["independent_full"],
                    "common_eval_provenance": provenance.get(key),
                    "subset_key": key in in_subset.get(policy, set()),
                })
        for policy, endpoints in point.structure_subset.items():
            for rank, endpoint in enumerate(endpoints):
                subset_rows.append({
                    "protein_id": protein_id, "policy": policy, "subset_rank": rank,
                    "source_endpoint_id": f"{point.rho_id}:{_t0_endpoint_key(endpoint)}",
                    "request_id": f"{protein_id}:{point.rho_id}:{policy}:{rank}",
                    # the Q_T0 draw is ordered by a Head-INDEPENDENT content hash (§2.10)
                    "head_independent_sampling": True,
                })
        for res in point.structure_results:
            result_rows.append({
                "protein_id": protein_id, "policy": res.policy, "request_id": res.request_id,
                "source_endpoint_id": f"{point.rho_id}:{res.source_endpoint_id}",
                "target_backbone_metric": None,
                "gate_pass": res.feasible,
                "request_status": "deferred_to_v0" if not res.evaluated else "evaluated",
                "cache_status": res.cache_status,
                "model_status": "executed" if res.model_executed else "not_executed",
                "failure_reason": res.failure_reason,
            })
        _t0_standard_tables(
            point, protein_id=protein_id, config=config, out_dir=out_dir,
            attempt_rows=attempt_rows, partial_rows=partial_rows, maturity=maturity,
            continuations=continuations, value_rows=value_rows, selection_rows=selection_rows,
            full_pool_rows=full_pool_rows,
        )
        reserved_by_rho[point.rho_id] = int(point.reserved_dfe)
        seeds.update(_t0_realized_seeds(point, protein_id=protein_id, config=config))

    ledger = [asdict(e) for point in points for e in point.ledger]
    return {
        **{name: [] for name in _CHECKPOINT_TABLES},
        "root_attempts": attempt_rows,
        "partial_roots": partial_rows,
        "maturity_telemetry": maturity,
        "continuations": continuations,
        "root_values": value_rows,
        "root_selection": selection_rows,
        "complete_entry_pool": full_pool_rows,
        "t0_control_membership": membership,
        "t0_structure_subset": subset_rows,
        "t0_structure_results": result_rows,
        "ledger": ledger,
        # T0 builds no parent, so there is no PreterminalReservation; what it DOES reserve is the
        # per-maturity matched budget that sizes the independent-full control. Dropping it would
        # make the matched-compute claim unauditable from artifacts alone.
        "reservation": None,
        "reserved_dfe_by_rho": reserved_by_rho,
        "realized_seeds": seeds,
    }


def _t0_realized_seeds(point, *, protein_id: str, config) -> dict:
    """PLAN §2.6:324 -- "every realized seed is persisted", and §2.6:325 -- a derived collision is a
    hard error. Keys are rho-prefixed because the grid points share a protein and would otherwise
    clobber each other (``root:0`` repeats at every maturity)."""
    ctx = _seed_ctx(config, protein_id, rho_id=point.rho_id)
    manifest = realized_seed_manifest(
        ctx, root_count=config.prefix_attempts,
        unique_root_hashes=list(point.unique_root_hashes), k_est=config.k_est,
        k_eval=config.k_eval, final_count=0,
        independent_full_count=point.matched_full_trajectories,
        include_random_membership=True,
    )
    return {f"{point.rho_id}:{k}": str(v) for k, v in manifest.items()}


def _t0_standard_tables(point, *, protein_id, config, out_dir, attempt_rows, partial_rows,
                        maturity, continuations, value_rows, selection_rows, full_pool_rows):
    """Append one grid point's rows to the standard §5.1 tables."""
    arm = config.entry_arm
    representative_by_hash = {p.root_equivalence_hash: p.root_id for p in point.unique_roots}
    attempt_rows.extend(root_attempt_rows(
        point.root_attempts, representative_root_id_by_hash=representative_by_hash,
        protein_id=protein_id, arm_id=arm, rho_id=point.rho_id,
    ))
    target_maturity = _rho_target_of(point.rho_id)
    for payload in point.unique_roots:
        rel, _ = write_payload_sidecar(out_dir, payload)
        partial_rows.append(partial_root_row(
            payload, payload_path=rel, target_maturity=target_maturity,
            actual_maturity=payload.actual_rho_edit,
        ))
    maturity.extend(maturity_rows(point.maturity, protein_id=protein_id))

    dfe_by_id: dict[str, int] = {}
    scored: list = []
    for payload in point.unique_roots:
        tail = payload.n_steps - payload.step
        h = payload.root_equivalence_hash
        for set_tag, table in (("est", point.est_scored_by_root), ("eval", point.eval_scored_by_root)):
            rows = table.get(h, ())
            scored.extend(rows)
            for row in rows:
                dfe_by_id[row.continuation.continuation_id] = tail
    continuations.extend(continuation_rows(scored, protein_id=protein_id, dfe_by_id=dfe_by_id))

    selected = point.policy_views["selected_partial"].root_hashes if point.policy_views else ()
    value_rows.extend(root_value_rows(
        point.root_values, expected_k=config.k_est, protein_id=protein_id, selected=selected,
    ))
    # T0 records BOTH partial policies here: `selected_partial` is the value beam and
    # `random_partial` is the Head-independent membership draw. Neither authorizes a fresh final
    # materialization -- T0 builds no parent (§2.9).
    value_by_hash = {rv.root_equivalence_hash: float(rv.value) for rv in point.root_values}
    for policy in ("selected_partial", "random_partial"):
        held = point.policy_views[policy].root_hashes if point.policy_views else ()
        for rank, h in enumerate(held):
            selection_rows.append({
                "protein_id": protein_id, "root_equivalence_hash": h, "policy": policy,
                "control": point.rho_id, "rank": rank, "selected": True, "tie_break": None,
                "source_value": value_by_hash.get(h),
                "fresh_materialization_authorized": False,
            })
    # The compute-matched control, ranked by exact terminal Head. EVERY generated trajectory is
    # recorded (including a failed one, which carries no risk) because all of them were paid for
    # out of the reserved budget and dropping them would understate this maturity's spend.
    rank_by_source = {c.source_id: rank for rank, c in enumerate(point.full_pool)}
    risk_by_source = {c.source_id: float(c.global_risk) for c in point.full_pool}
    md5_by_source = {c.source_id: c.sequence_md5 for c in point.full_pool}
    for traj in point.full_trajectories:
        source_id = f"{protein_id}:{arm}:{point.rho_id}:full:{traj.replicate_index}"
        full_pool_rows.append({
            "protein_id": protein_id, "arm_id": arm, "source_id": source_id,
            "seed": int(traj.seed), "sequence": traj.sequence,
            "sequence_md5": md5_by_source.get(source_id),
            "terminal_global_risk": risk_by_source.get(source_id),
            "rank": rank_by_source.get(source_id),  # null => generated but not ranked (no sequence)
            "dfe": int(traj.logical_dfe),
        })


def _rho_target_of(rho_id: str) -> float:
    return float(str(rho_id).removeprefix("rho"))


def _t0_endpoint_key(endpoint) -> str:
    continuation = getattr(endpoint, "continuation", None)
    if continuation is not None:
        return str(continuation.continuation_id)
    return f"full:{int(endpoint.replicate_index)}"


def _realized_seeds(result, config, ctx) -> dict:
    """Enumerate this protein's realized seeds through the single derivation helper, so the
    persisted values ARE the ones the run drew rather than a re-derivation that could drift."""
    if config.entry_arm == "terminal":
        return {k: str(v) for k, v in realized_seed_manifest(
            ctx, root_count=0, unique_root_hashes=(), k_est=0, k_eval=0, final_count=0,
            terminal_count=len(result.terminal_trajectories),
        ).items()}
    return {k: str(v) for k, v in realized_seed_manifest(
        ctx, root_count=len(result.root_attempts),
        unique_root_hashes=[p.root_equivalence_hash for p in result.unique_roots],
        k_est=config.k_est, k_eval=0, final_count=1,
        # only the ranked beam ran a final completion; the unique-root pool is a minimum coverage
        # requirement and is normally larger (§2.3)
        final_root_hashes=list(result.ranked_root_hashes),
    ).items()}


def _failed_checkpoint(
    journal: EntryJournal, protein_id: str, config, arm: str, *, terminal: bool = False
) -> dict:
    """Build a failed protein's checkpoint from the journal so its charged work (ledger + recorded
    attempts) is preserved rather than written as empty."""
    if terminal:
        # A terminal failure has no root graph; only the paid ledger survives.
        return {**{name: [] for name in _CHECKPOINT_TABLES},
                "ledger": [asdict(event) for event in journal.ledger]}
    rho_id = _rho_id(config)
    crossed = [o.payload for o in journal.root_attempts if o.payload is not None]
    representative_by_hash = (
        {p.root_equivalence_hash: p.root_id for p in collapse_roots(crossed).unique}
        if crossed else {}
    )
    tables = {name: [] for name in _CHECKPOINT_TABLES}
    tables["root_attempts"] = root_attempt_rows(
        journal.root_attempts, representative_root_id_by_hash=representative_by_hash,
        protein_id=protein_id, arm_id=arm, rho_id=rho_id,
    )
    return {**tables, "ledger": [asdict(event) for event in journal.ledger]}


def _result_to_checkpoint(result, out_dir, config, config_digest) -> dict:
    arm = result.arm_id
    if config.entry_arm == "terminal":
        return _terminal_checkpoint(result, config_digest)
    rho_id = _rho_id(config)
    payload_by_hash = {p.root_equivalence_hash: p for p in result.unique_roots}
    representative_by_hash = {p.root_equivalence_hash: p.root_id for p in result.unique_roots}

    # partial roots + content-addressed sidecars; actual maturity is DERIVED from x_t, not rho.
    partial_rows = []
    for payload in result.unique_roots:
        rel, _ = write_payload_sidecar(out_dir, payload)
        partial_rows.append(
            partial_root_row(
                payload, payload_path=rel, target_maturity=float(config.rho_target),
                actual_maturity=payload.actual_rho_edit,
            )
        )

    # continuations (est + final) with each continuation's tail DFE.
    dfe_by_id: dict[str, int] = {}
    for payload in result.unique_roots:
        tail = payload.n_steps - payload.step
        for r in range(config.k_est):
            dfe_by_id[f"{payload.root_equivalence_hash}:est:{r}"] = tail
    for h in result.ranked_root_hashes:
        dfe_by_id[f"{h}:final:0"] = payload_by_hash[h].n_steps - payload_by_hash[h].step
    all_scored = [sc for scored in result.est_scored_by_root.values() for sc in scored]
    all_scored += list(result.final_scored)
    continuation_id_by_hash = {
        h: mat.continuation.continuation_id for h, mat in result.materializations.items()
    }

    return {
        "root_attempts": root_attempt_rows(
            result.root_attempts, representative_root_id_by_hash=representative_by_hash,
            protein_id=result.protein_id, arm_id=arm, rho_id=rho_id,
        ),
        "partial_roots": partial_rows,
        "maturity_telemetry": maturity_rows(result.maturity, protein_id=result.protein_id),
        "continuations": continuation_rows(
            all_scored, protein_id=result.protein_id, dfe_by_id=dfe_by_id
        ),
        "root_values": root_value_rows(
            result.root_values, expected_k=config.k_est, protein_id=result.protein_id,
            selected=result.ranked_root_hashes,
        ),
        "root_selection": root_selection_rows(
            result.root_selection, protein_id=result.protein_id
        ),
        "terminal_parent_facade": facade_rows(
            result.facade, protein_id=result.protein_id, arm_id=arm, config_hash=config_digest,
            continuation_id_by_hash=continuation_id_by_hash,
        ),
        "terminal_parent_admission": admission_rows(
            result.admission, protein_id=result.protein_id, arm_id=arm
        ),
        "facade_collapse": _facade_collapse_rows(result),
        "ledger": [asdict(event) for event in result.ledger],
    }


def _facade_collapse_rows(result) -> list[dict]:
    """One row per facade lineage dropped by sequence convergence. Keeping this only in memory is
    the same as not recording it: the chain PLAN §2.11:489-501 mandates is read from parquet."""
    return [
        {
            "protein_id": result.protein_id, "arm_id": result.arm_id,
            "design_idx": int(row.design_idx), "source_id": row.source_id,
            "root_equivalence_hash": row.root_equivalence_hash,
            "sequence_md5": row.sequence_md5,
            "representative_source_id": row.representative_source_id,
            "representative_design_idx": int(row.representative_design_idx),
        }
        for row in result.facade_collapsed
    ]


def _terminal_checkpoint(result, config_digest) -> dict:
    """The TERMINAL arm has no partial roots, estimator set or maturity, so those tables stay
    empty. ``complete_entry_pool`` records EVERY generated trajectory -- including the ones that
    ranked below F_cap and the failed ones -- because all of them were paid for out of the matched
    budget and dropping them would understate this arm's spend."""
    rank_by_source = {row.source_id: int(row.design_idx) for row in result.facade}
    # A trajectory that ranked INSIDE F_cap and lost only to sequence convergence is not the same
    # thing as one that ranked below the cap. Both have rank=None; only the first has a
    # representative, and recording it is what keeps them distinguishable.
    collapsed_into = {row.source_id: row.representative_source_id
                      for row in result.facade_collapsed}
    scored_by_source = {c.source_id: c for c in result.terminal_pool}
    pool = []
    for traj in result.terminal_trajectories:
        source_id = f"{result.protein_id}:{result.arm_id}:t{traj.replicate_index}"
        scored = scored_by_source.get(source_id)
        pool.append({
            "protein_id": result.protein_id, "arm_id": result.arm_id, "source_id": source_id,
            "seed": int(traj.seed), "sequence": traj.sequence,
            "sequence_md5": scored.sequence_md5 if scored is not None else None,
            # measured for every surviving trajectory, not just the submitted ones; null only for
            # a trajectory that failed before producing a sequence.
            "terminal_global_risk": float(scored.global_risk) if scored is not None else None,
            "rank": rank_by_source.get(source_id),   # null => holds no facade slot
            "dfe": int(traj.logical_dfe),
            "collapsed_into_source_id": collapsed_into.get(source_id),
        })
    return {
        **{name: [] for name in _CHECKPOINT_TABLES},
        "complete_entry_pool": pool,
        "facade_collapse": _facade_collapse_rows(result),
        "terminal_parent_facade": facade_rows(
            result.facade, protein_id=result.protein_id, arm_id=result.arm_id,
            config_hash=config_digest,
        ),
        "terminal_parent_admission": admission_rows(
            result.admission, protein_id=result.protein_id, arm_id=result.arm_id
        ),
        "ledger": [asdict(event) for event in result.ledger],
    }


def _assert_unique_join_keys(rows_by_table: Mapping[str, list]) -> None:
    """A duplicate replay-graph join key (root_id / continuation_id) invalidates the run (§5.3);
    fail-fast rather than emit an ambiguous graph."""
    for table, key in (("partial_roots", "root_id"), ("continuations", "continuation_id")):
        ids = [r[key] for r in rows_by_table.get(table, []) if r.get(key) is not None]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate {key} in {table}: broken replay join key")


def aggregate_entry_artifacts(
    *, out_dir, requested_cohort, config, input_signature_by_protein=None, campaign_command=None,
    substrate=None,
) -> dict:
    """Aggregate the frozen requested cohort's checkpoints (under the current arm) into the §5
    artifact family. Byte-stable under shard/order permutations (every table is canonically sorted
    on write), surfaces missing proteins, single-counts cost, re-verifies each checkpoint's resume
    identity (config digest + optional §4.2 input signature), and rejects duplicate join keys."""
    requested = list(requested_cohort)
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate protein id in the requested cohort")
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    arm = config.entry_arm
    config_digest = _config_digest(config)

    rows_by_table: dict[str, list] = {name: [] for name in V1_TABLE_SCHEMAS}
    ledger_events: list[LedgerEvent] = []
    status_by_protein: dict[str, str] = {}
    reservation_by_protein: dict[str, dict] = {}
    reserved_by_rho_by_protein: dict[str, dict] = {}
    realized_seeds_by_protein: dict[str, dict] = {}
    n_ok = 0
    for pid in requested:
        ckpt = ckpt_dir / _checkpoint_name(pid, arm, config.phase)
        if not ckpt.exists():
            continue  # never processed under this arm -> build_cohort_coverage marks it missing
        data = json.loads(ckpt.read_text())
        if data.get("config_digest") != config_digest:
            continue  # stale-config checkpoint is ignored, not silently trusted
        # re-verify the resume identity: a checkpoint whose §4.2 input signature no longer matches
        # (input content changed but the shard was not re-run) is stale and must not be accepted.
        if input_signature_by_protein is not None:
            expected = _run_sig(config_digest, arm, pid, input_signature_by_protein[pid])
            if data.get("run_sig") != expected:
                continue
        status_by_protein[pid] = data.get("status", "failed")
        if data.get("ok") and status_by_protein[pid] in ENTRY_OK_STATUSES:
            n_ok += 1
        if data.get("reservation"):
            reservation_by_protein[pid] = data["reservation"]
        if data.get("reserved_dfe_by_rho"):
            # T0 has no PreterminalReservation (it builds no parent), but it DOES reserve a matched
            # budget per maturity -- the one that sizes the independent-full control. Recording it
            # is what makes the compute-matching auditable from the artifacts alone.
            reserved_by_rho_by_protein[pid] = {
                k: int(v) for k, v in sorted(data["reserved_dfe_by_rho"].items())
            }
        if data.get("realized_seeds"):
            realized_seeds_by_protein[pid] = data["realized_seeds"]
        for name in _CHECKPOINT_TABLES:
            rows_by_table[name].extend(data.get(name, []))
        for record in data.get("ledger", []):
            ledger_events.append(LedgerEvent(**record))

    _assert_unique_join_keys(rows_by_table)

    coverage = build_cohort_coverage(requested, status_by_protein)
    rows_by_table["cohort_coverage"] = coverage_rows(
        coverage, arm_id=arm, phase=config.phase, frozen_split_id=config.split_role
    )

    for name, schema in V1_TABLE_SCHEMAS.items():
        write_stable_parquet(
            out_dir / f"{name}.parquet", rows_by_table[name],
            columns=schema.columns, sort_by=schema.sort_by,
        )
    write_cost_ledger_jsonl(
        out_dir / "cost_ledger.jsonl", [LedgerEvent(**asdict(e)) for e in ledger_events]
    )

    ledger_totals = aggregate_ledger(ledger_events)
    manifest = {
        "config_digest": config_digest, "arm": arm, "phase": config.phase,
        "split_role": config.split_role, "campaign_id": config.campaign_id,
        "nmp_absent": True, "n_proteins": len(requested), "n_proteins_ok": n_ok,
        "n_proteins_missing": sum(1 for r in coverage if r.status == "missing"),
        "requested_cohort": requested, "ledger_totals": ledger_totals,
        # The matched budget the TERMINAL arm must be handed, per protein (§3.2.1). For T0 there is
        # no single per-protein reservation -- each maturity reserves its own -- so the total the
        # protein reserved across the grid is recorded and the per-maturity breakdown sits beside
        # it. Either way the reserved side of the matched-compute audit reaches disk.
        "reserved_dfe_by_protein": {
            **{pid: sum(r.values()) for pid, r in sorted(reserved_by_rho_by_protein.items())},
            **{pid: int(r["reserved_dfe"]) for pid, r in sorted(reservation_by_protein.items())},
        },
        "reserved_dfe_by_rho": dict(sorted(reserved_by_rho_by_protein.items())),
        "reservation_detail": dict(sorted(reservation_by_protein.items())),
        "realized_seeds_by_protein": dict(sorted(realized_seeds_by_protein.items())),
        "artifacts": [f"{name}.parquet" for name in V1_TABLE_SCHEMAS]
        + ["cost_ledger.jsonl"],
        "command": campaign_command,
        # The SCIENTIFIC SUBSTRATE this run's reservation was frozen under: S, the §4.2 input
        # content digests, and the resolved null-kernel identity. Runbook §2:118 and §9:471-477
        # require the manifest to record these; more importantly, the Terminal arm consumes
        # `C_reserved` from here as an INDEPENDENT shard, and without them nothing can show the two
        # arms shared a Head, a generator, a backbone or an S. S alone is decisive: a Terminal run
        # reading S from its own sampler YAML derives M_T = floor(C_reserved / S), so half the S
        # doubles its trajectory count against a budget frozen at the other S -- and every artifact
        # stays internally consistent at the wrong scale.
        "substrate": dict(substrate) if substrate else None,
    }
    write_manifest(out_dir / "manifest.json", manifest)
    return manifest
