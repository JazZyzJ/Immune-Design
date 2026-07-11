"""Per-protein round orchestration: the definitive evaluator/firewall (F4) and the exact
complete-state edit-and-repair loop (F5+).

Pure control logic — Head/structure are injected oracles (fake in tests, real in the driver),
so this module is torch-free. The evaluator scores the whole deduplicated pool in one Head
batch, applies the parent-relative off-halo hotspot gate, shortlists a bounded per-parent set
by ascending Head risk, refolds only that shortlist through the provenance-keyed cache, and
lets only definitively feasible children reach a selector (§1.4).
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import moves as mv
from . import objective as ob
from . import selection as sel
from .oracles import structure_feasible
from .state import (
    CANONICAL_AA20,
    CandidateEvaluation,
    EliteState,
    FusionStateError,
    ParticleState,
    PopulationState,
    make_elite_id,
    make_particle_id,
    sequence_md5,
)

_AA20 = frozenset(CANONICAL_AA20)


class FusionRunnerError(RuntimeError):
    """Raised on an invalid initial population, a mis-scoped config, or an unmet contract."""


def _derive_seed(*parts) -> int:
    """Deterministic, process-independent seed from parts (never Python hash())."""
    digest = hashlib.md5(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _guard_active_site_config(config, anchors) -> None:
    """An anchored protein MUST carry a configured shell ceiling, else the gate is fail-open."""
    if anchors and config.structure.active_site_RMSD_max is None:
        raise FusionRunnerError(
            "anchored protein requires structure.active_site_RMSD_max (else active-site gate is skipped)")


def _validate_anchor_indices(anchors, seq_len: int) -> None:
    """Fail-closed on anchors outside [0, seq_len): a negative index would silently wrap to a
    terminal residue (freezing / shell membership), a too-large index would drop off the trace."""
    for a in anchors:
        if a < 0 or a >= seq_len:
            raise FusionRunnerError(f"anchor index {a} out of range [0, {seq_len}) — malformed manifest")


def evaluate_children(*, protein_id, parents, proposals_by_parent, oracles,
                      structure_cache, config, has_active_site: bool) -> list[CandidateEvaluation]:
    """Evaluate every child proposal and return a CandidateEvaluation per child (§1.4).

    One Head batch over ``unique(parent seqs ∪ all child seqs)``; parent-relative off-halo
    hotspot gate; per-parent ascending-Head-risk shortlist capped at
    ``structure.max_refolds_per_parent``; definitive structure gate only on the shortlist.
    Non-shortlisted / hotspot-infeasible children are returned unrefolded and infeasible.
    """
    # 1) deduplicated sequence pool (parents first, then children), order-preserving
    order: list[str] = []
    seen: set[str] = set()
    for p in parents:
        if p.sequence not in seen:
            seen.add(p.sequence)
            order.append(p.sequence)
    for p in parents:
        for prop in proposals_by_parent.get(p.particle_id, []):
            if prop.sequence not in seen:
                seen.add(prop.sequence)
                order.append(prop.sequence)

    # 2) one Head batch for the whole stage
    head_scores = oracles.head_fn(protein_id, order)
    score_map = {s: hs for s, hs in zip(order, head_scores)}
    parent_hs = {p.particle_id: score_map[p.sequence] for p in parents}

    offtarget_max = config.objective.max_offtarget_window_increase
    max_refolds = config.structure.max_refolds_per_parent
    window_detail = config.telemetry.window_detail  # S0 window-landscape telemetry (opt-in)

    def _aligned_windows(child_hs):
        if not window_detail:
            return ()
        return tuple((int(w.start_0b), int(w.end_0b), int(w.k), float(w.z))
                     for w in sorted(child_hs.windows,
                                     key=lambda w: (w.start_0b, w.end_0b, w.k)))

    results: list[CandidateEvaluation] = []
    for p in parents:
        p_hs = parent_hs[p.particle_id]
        pending = []  # (prop, g, nh, ld, hotspot_ok)
        for prop in proposals_by_parent.get(p.particle_id, []):
            child_hs = score_map[prop.sequence]
            g = ob.global_risk_of(child_hs)
            nh = ob.new_hotspot(child_hs, p_hs, halo_start=prop.halo_start_0b, halo_end=prop.halo_end_0b)
            ld = ob.local_target_delta(child_hs, p_hs,
                                       target_start=prop.target_start_0b, target_end=prop.target_end_0b)
            hotspot_ok = nh.max_increase <= offtarget_max
            pending.append((prop, g, nh, ld, hotspot_ok))

        # shortlist hotspot-feasible children by ascending Head risk (deterministic tie: proposal_id)
        feasible = sorted((x for x in pending if x[4]), key=lambda x: (x[1], x[0].proposal_id))
        shortlist_ids = {x[0].proposal_id for x in feasible[:max_refolds]}

        for prop, g, nh, ld, hotspot_ok in pending:
            if prop.proposal_id in shortlist_ids:
                metrics, _hit = structure_cache.evaluate(protein_id, prop.sequence, oracles.struct_fn)
                passed, reason = structure_feasible(metrics, config, has_active_site=has_active_site)
                structure, structure_evaluated, is_feasible = metrics, True, passed
            else:
                structure, structure_evaluated, is_feasible = None, False, False
                reason = "offtarget_hotspot" if not hotspot_ok else "not_shortlisted"
            results.append(CandidateEvaluation(
                sequence_md5=sequence_md5(prop.sequence),
                parent_particle_id=p.particle_id,
                proposal_id=prop.proposal_id,
                target_start_0b=prop.target_start_0b,
                target_end_0b=prop.target_end_0b,
                head_global_risk=g,
                aligned_windows=_aligned_windows(score_map[prop.sequence]),
                local_target_delta=ld,
                new_hotspot_max=nh.max_increase,
                new_hotspot_mass=nh.positive_mass,
                new_hotspot_count=nh.positive_count,
                structure=structure,
                structure_evaluated=structure_evaluated,
                feasible=is_feasible,
                reason="ok" if is_feasible else reason,
            ))
    return results


@dataclass
class RoundRecord:
    round_idx: int
    n_parents: int
    n_proposals: int
    n_refolds: int
    n_feasible_children: int
    n_selected_children: int
    elite_risk: float
    ess: float | None = None
    resampled: bool = False
    cache_hits: int = 0
    wall_time_s: float = 0.0


@dataclass
class ProteinResult:
    protein_id: str
    initial_population: PopulationState
    final_population: PopulationState
    elite: ParticleState
    rounds: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    lineage: list = field(default_factory=list)
    populations: list = field(default_factory=list)  # round 0 (init) + each round's population
    proposals: dict = field(default_factory=dict)    # proposal_id -> Proposal (all rounds)


def _keep_parent(parent, *, protein_id, round_idx, slot, weight) -> ParticleState:
    """Null move: carry the exact parent bytes forward with a fresh per-round identity."""
    return ParticleState(
        particle_id=make_particle_id(protein_id, round_idx, slot, parent.sequence_md5),
        protein_id=protein_id, round_idx=round_idx, slot_idx=slot,
        sequence=parent.sequence, sequence_md5=parent.sequence_md5, weight=weight,
        parent_particle_id=parent.particle_id, source_proposal_id=None,
        head_global_risk=parent.head_global_risk, structure=parent.structure,
        feasible=True, lineage_seed=parent.lineage_seed)


def _is_canonical(seq: str) -> bool:
    return bool(seq) and set(seq) <= _AA20


def build_initial_population(source_rows, *, protein_id, oracles, structure_cache, config,
                             anchors, anchor_expected=None, expected_length=None) -> PopulationState:
    """Terminal handoff (§1.7): build the N-slot initial population from source RF designs.

    Order rows by ``(design_idx, seed)``; validate canonical alphabet, length (against the target
    backbone ``expected_length`` when supplied — else the first valid row), and hard anchors; run
    the definitive structure gate until ``N`` feasible slots are found (fail with
    ``insufficient_feasible_initial_population`` if fewer exist — never duplicate to hide missing
    coverage); then fresh-Head-score the N, initialise weights from ``beta_0``, and seed the elite
    from the lowest-Head feasible state. Head is NOT used to preselect.
    """
    _guard_active_site_config(config, anchors)
    if anchors and not anchor_expected:
        raise FusionRunnerError(
            "anchored protein requires anchor_expected (index->residue) for §1.7 anchor validation")
    has_active_site = bool(anchors)
    n = config.core.population_size
    rows = sorted(source_rows, key=lambda r: (int(r["design_idx"]), int(r.get("seed", 0))))

    chosen: list = []  # (sequence, metrics)
    ref_len = int(expected_length) if expected_length is not None else None
    for row in rows:
        seq = str(row["sequence"])
        if not _is_canonical(seq):
            continue
        if ref_len is None:
            ref_len = len(seq)
        elif len(seq) != ref_len:  # length must match the target backbone / other slots
            continue
        if anchor_expected and any(i >= len(seq) or seq[i] != aa
                                   for i, aa in anchor_expected.items()):
            continue
        metrics, _hit = structure_cache.evaluate(protein_id, seq, oracles.struct_fn)
        passed, _reason = structure_feasible(metrics, config, has_active_site=has_active_site)
        if not passed:
            continue
        chosen.append((seq, metrics))
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        raise FusionRunnerError(
            f"insufficient_feasible_initial_population for {protein_id}: {len(chosen)} < N={n}")

    seqs = [c[0] for c in chosen]
    head_scores = oracles.head_fn(protein_id, seqs)
    risks = [ob.global_risk_of(hs) for hs in head_scores]

    beta_0 = config.selection.beta[0]
    log_w = [-beta_0 * risk for risk in risks]
    m = max(log_w)
    raw = [math.exp(lw - m) for lw in log_w]
    total = math.fsum(raw)
    weights = [w / total for w in raw]

    particles = tuple(
        ParticleState(
            particle_id=make_particle_id(protein_id, 0, i, sequence_md5(seq)),
            protein_id=protein_id, round_idx=0, slot_idx=i, sequence=seq,
            sequence_md5=sequence_md5(seq), weight=wt, parent_particle_id=None,
            source_proposal_id=None, head_global_risk=risk, structure=metrics,
            feasible=True, lineage_seed=_derive_seed(config.core.seed, protein_id, "init", i))
        for i, ((seq, metrics), risk, wt) in enumerate(zip(chosen, risks, weights)))
    best_i = min(range(n), key=lambda i: (risks[i], seqs[i]))
    elite_particle = replace(particles[best_i],
                             particle_id=make_elite_id(protein_id, 0, particles[best_i].sequence_md5))
    elite = EliteState(particle=elite_particle, first_round_seen=0, last_round_seen=0)
    return PopulationState(round_idx=0, particles=particles, elite=elite)


def _round_proposals(protein_id, r, parents, parent_windows, *, oracles, config, anchors, repair_fn):
    """Generate the round's proposal pool per parent (§1.5, F6 round order): direct explicit
    edits (always) and direct rf_reopen (when enabled); then, when edit-repair is enabled,
    Head-score the direct explicit edits, shortlist the lowest-Head new-hotspot-feasible ones,
    and append their repaired descendants. Returns ``(proposals_by_parent, prop_lookup)``."""
    ex = config.moves.explicit
    seed = config.core.seed
    proposals_by_parent: dict = {}
    prop_lookup: dict = {}
    targets: dict = {}
    for p in parents:
        target = ob.select_target_register(parent_windows[p.particle_id], anchors=anchors)
        targets[p.particle_id] = target
        if target is None:  # stalled_no_editable_target
            proposals_by_parent[p.particle_id] = []
            continue
        props = list(mv.explicit_edit_proposals(
            p, target, max_edit_order=ex.max_edit_order,
            max_raw_candidates=ex.max_raw_candidates_per_parent, pair_seed_budget=ex.pair_seed_budget,
            base_seed=_derive_seed(seed, protein_id, r, p.slot_idx, "explicit"))) if ex.enabled else []
        if config.moves.rf_reopen.enabled:
            props += mv.rf_reopen_proposals(
                p, target, halo_radius=config.targeting.halo_radius, anchors=anchors,
                repair_fn=repair_fn, base_seed=_derive_seed(seed, protein_id, r, p.slot_idx, "reopen"),
                children=config.moves.rf_reopen.children_per_parent)
        proposals_by_parent[p.particle_id] = props
        for pr in props:
            prop_lookup[pr.proposal_id] = pr

    if config.moves.repair.enabled:
        for p in parents:
            target = targets[p.particle_id]
            if target is None:
                continue
            explicit_props = [pr for pr in proposals_by_parent[p.particle_id]
                              if pr.move_family == "explicit_edit"]
            if not explicit_props:
                continue
            p_hs = parent_windows[p.particle_id]
            # measure off-target hotspots against the REPAIR halo (what edit_repair will actually
            # regenerate = target +/- halo_radius), NOT the explicit proposal's narrow target span,
            # else a hotspot in the halo margin (which repair fixes) drops a fixable candidate.
            rh_start, rh_end = mv.repair_halo_span(target, config.targeting.halo_radius, len(p.sequence))
            hs_list = oracles.head_fn(protein_id, [pr.sequence for pr in explicit_props])
            scored = []
            for pr, hs in zip(explicit_props, hs_list):
                nh = ob.new_hotspot(hs, p_hs, halo_start=rh_start, halo_end=rh_end)
                if nh.max_increase <= config.objective.max_offtarget_window_increase:
                    scored.append((ob.global_risk_of(hs), pr))
            scored.sort(key=lambda x: (x[0], x[1].proposal_id))
            for _g, pr in scored[:config.moves.repair.repair_shortlist_per_parent]:
                edit = {pos: pr.sequence[pos] for pos in pr.edited_positions}
                repaired = mv.edit_repair_proposals(
                    p, target, edit, halo_radius=config.targeting.halo_radius, anchors=anchors,
                    repair_fn=repair_fn,
                    base_seed=_derive_seed(seed, protein_id, r, p.slot_idx, "repair", pr.proposal_id),
                    children=config.moves.repair.children_per_edit)
                proposals_by_parent[p.particle_id].extend(repaired)
                for rp in repaired:
                    prop_lookup[rp.proposal_id] = rp

    # Global per-parent sequence dedup across ALL families: an identity edit-repair must not
    # duplicate a direct explicit state, and FK must not count identical sequences as multiple
    # offspring (which would distort ancestry mass). Keep the first (direct edits precede repairs).
    for pid, props in proposals_by_parent.items():
        seen_seq: set = set()
        unique = []
        for pr in props:
            if pr.sequence in seen_seq:
                continue
            seen_seq.add(pr.sequence)
            unique.append(pr)
        proposals_by_parent[pid] = unique
    prop_lookup = {pr.proposal_id: pr for props in proposals_by_parent.values() for pr in props}
    return proposals_by_parent, prop_lookup


def run_protein(*, protein_id, initial_population, oracles, structure_cache, config,
                anchors, repair_fn=None) -> ProteinResult:
    """Exact complete-state edit-and-repair loop for one protein (§5).

    Each round: fresh-Head-score parents (drives BOTH register selection and the improvement
    comparison, so a stale stored risk can't distort selection), propose broad edits, evaluate
    the pool, update the independent best-feasible elite archive from ALL feasible evaluations,
    then dispatch greedy/beam/FK over the same pool and carry the exact selected bytes forward.
    The parent is always the null move; the elite is monotone. ``has_active_site`` is derived
    from ``anchors`` (single source of truth). v0 requires the explicit move family (F6 adds
    rf_reopen / edit-repair).
    """
    has_active_site = bool(anchors)
    _guard_active_site_config(config, anchors)
    if not config.moves.explicit.enabled:  # explicit breadth is the load-bearing lever
        raise NotImplementedError("v0 runner requires moves.explicit.enabled (the breadth lever)")
    if (config.moves.rf_reopen.enabled or config.moves.repair.enabled) and repair_fn is None:
        raise FusionRunnerError("rf_reopen/edit-repair enabled but no repair_fn was provided")
    n = config.core.population_size
    if len(initial_population.particles) != n:
        raise FusionRunnerError(
            f"initial population size {len(initial_population.particles)} != config N={n}")
    if anchors:  # fail-closed on a malformed manifest (negative / out-of-range anchor index)
        _validate_anchor_indices(anchors, len(initial_population.particles[0].sequence))
    if not all(p.feasible for p in initial_population.particles):
        raise FusionRunnerError("initial population contains an infeasible particle")

    population = initial_population
    elite = population.elite
    inv_n = 1.0 / n
    rounds: list = []
    all_candidates: list = []
    lineage: list = []
    populations: list = [initial_population]
    all_proposals: dict = {}

    for r in range(1, config.core.n_rounds + 1):
        round_t0 = time.time()
        base_parents = list(population.particles)

        # Stage A — fresh-Head-score parents; refresh their global risk (fail-fast on None/NaN).
        parent_scores = oracles.head_fn(protein_id, [p.sequence for p in base_parents])
        parents: list = []
        parent_windows: dict = {}
        for p, p_hs in zip(base_parents, parent_scores):
            fresh = ob.global_risk_of(p_hs)
            parents.append(p if p.head_global_risk == fresh else replace(p, head_global_risk=fresh))
            parent_windows[p.particle_id] = p_hs

        proposals_by_parent, prop_lookup = _round_proposals(
            protein_id, r, parents, parent_windows,
            oracles=oracles, config=config, anchors=anchors, repair_fn=repair_fn)
        all_proposals.update(prop_lookup)

        # Stage B — one evaluation pass; refolds/hits counted from actual cache deltas.
        misses_before = structure_cache.misses
        hits_before = structure_cache.hits
        evals = evaluate_children(
            protein_id=protein_id, parents=parents, proposals_by_parent=proposals_by_parent,
            oracles=oracles, structure_cache=structure_cache, config=config,
            has_active_site=has_active_site)
        n_refolds = structure_cache.misses - misses_before
        all_candidates.extend(evals)
        evals_by_parent: dict = {}
        for e in evals:
            evals_by_parent.setdefault(e.parent_particle_id, []).append(e)

        # Independent elite archive: update from ALL feasible evaluations (before any selector
        # rejects them), so an epsilon_H-rejected or FK-unsampled feasible best is never lost.
        feasible_evals = [e for e in evals if e.feasible]
        if feasible_evals:
            best_e = min(feasible_evals, key=lambda e: (e.head_global_risk, e.proposal_id))
            if best_e.head_global_risk < elite.particle.head_global_risk:
                bp = prop_lookup[best_e.proposal_id]
                cand = ParticleState(
                    particle_id=make_elite_id(protein_id, r, sequence_md5(bp.sequence)),
                    protein_id=protein_id, round_idx=r, slot_idx=0, sequence=bp.sequence,
                    sequence_md5=sequence_md5(bp.sequence), weight=inv_n,
                    parent_particle_id=best_e.parent_particle_id, source_proposal_id=best_e.proposal_id,
                    head_global_risk=best_e.head_global_risk, structure=best_e.structure,
                    feasible=True, lineage_seed=_derive_seed(config.core.seed, protein_id, r, "elite"))
                elite = EliteState(particle=cand, first_round_seen=r, last_round_seen=r)

        # Stage C — selector dispatch; greedy/beam/FK all replay the SAME evaluated pool.
        mode = config.selection.mode
        ess = None
        if mode == "greedy":
            picks = [(p.particle_id,
                      sel.select_greedy(p, evals_by_parent.get(p.particle_id, []),
                                        min_head_improvement=config.objective.min_head_improvement))
                     for p in parents]
        elif mode == "beam":
            picks = sel.select_beam(parents, evals_by_parent,
                                    min_head_improvement=config.objective.min_head_improvement, N=n)
        elif mode == "fk":
            beta = config.selection.beta
            rng = np.random.default_rng(_derive_seed(config.core.seed, protein_id, r, "fk"))
            picks, ess = sel.select_fk(parents, evals_by_parent,
                                       beta_r=beta[r], beta_prev=beta[r - 1], N=n, rng=rng)
        else:  # pragma: no cover - config validated upstream
            raise ValueError(f"unknown selection mode {mode!r}")

        # Materialize picks (exact carryover; FK duplicates -> distinct slots, shared ancestry).
        parent_by_id = {p.particle_id: p for p in parents}
        new_particles = []
        for slot, (parent_id, ev) in enumerate(picks):
            parent = parent_by_id[parent_id]
            if ev is None:
                new_particles.append(_keep_parent(
                    parent, protein_id=protein_id, round_idx=r, slot=slot, weight=inv_n))
                continue
            bp = prop_lookup[ev.proposal_id]
            new_particles.append(ParticleState(
                particle_id=make_particle_id(protein_id, r, slot, sequence_md5(bp.sequence)),
                protein_id=protein_id, round_idx=r, slot_idx=slot, sequence=bp.sequence,
                sequence_md5=sequence_md5(bp.sequence), weight=inv_n,
                parent_particle_id=parent_id, source_proposal_id=ev.proposal_id,
                head_global_risk=ev.head_global_risk, structure=ev.structure,
                feasible=True, lineage_seed=_derive_seed(config.core.seed, protein_id, r, slot)))

        # Beam elite protection — applied BEFORE lineage/telemetry so no dangling child is logged.
        # The carry re-instates the elite by re-applying its OWN originating proposal to its OWN
        # (persisted) parent, so S2 exact-ancestry replay resolves: pointing at the archive id
        # (absent from the particle table) with a null proposal would strand the carry. A seed
        # elite (never produced by a proposal) legitimately carries a null proposal -> no edge; it
        # is replayable by matching its sequence to the initial population (round 0).
        protected_slot = None
        if mode == "beam" and elite.particle.sequence_md5 not in {p.sequence_md5 for p in new_particles}:
            protected_slot = max(range(len(new_particles)),
                                 key=lambda i: new_particles[i].head_global_risk)
            e = elite.particle
            if e.source_proposal_id is not None and e.source_proposal_id not in all_proposals:
                raise FusionRunnerError(
                    f"protected elite proposal {e.source_proposal_id} is not resolvable")
            new_particles[protected_slot] = ParticleState(
                particle_id=make_particle_id(protein_id, r, protected_slot, e.sequence_md5),
                protein_id=protein_id, round_idx=r, slot_idx=protected_slot, sequence=e.sequence,
                sequence_md5=e.sequence_md5, weight=inv_n, parent_particle_id=e.parent_particle_id,
                source_proposal_id=e.source_proposal_id,  # the elite's own originating proposal
                head_global_risk=e.head_global_risk, structure=e.structure, feasible=True,
                lineage_seed=e.lineage_seed)
            if e.source_proposal_id is not None:
                lineage.append({"parent_particle_id": e.parent_particle_id,
                                "child_particle_id": new_particles[protected_slot].particle_id,
                                "round_idx": r, "proposal_id": e.source_proposal_id,
                                "selector": "elite_protect", "multiplicity": 1})

        # Lineage + counts derived from the FINAL population (genuine selected children only; the
        # protected-elite slot already emitted its own elite_protect edge above).
        for i, p in enumerate(new_particles):
            if i == protected_slot:
                continue
            if p.source_proposal_id is not None:
                lineage.append({"parent_particle_id": p.parent_particle_id,
                                "child_particle_id": p.particle_id, "round_idx": r,
                                "proposal_id": p.source_proposal_id, "selector": mode,
                                "multiplicity": 1})
        n_selected = sum(1 for i, p in enumerate(new_particles)
                         if i != protected_slot and p.source_proposal_id is not None)

        # Elite may also improve via a selected/duplicated survivor.
        best_new = min(new_particles, key=lambda x: x.head_global_risk)
        if best_new.head_global_risk < elite.particle.head_global_risk:
            elite_particle = replace(best_new,
                                     particle_id=make_elite_id(protein_id, r, best_new.sequence_md5))
            elite = EliteState(particle=elite_particle, first_round_seen=r, last_round_seen=r)

        population = PopulationState(round_idx=r, particles=tuple(new_particles), elite=elite)
        populations.append(population)
        rounds.append(RoundRecord(
            round_idx=r, n_parents=len(parents),
            n_proposals=sum(len(v) for v in proposals_by_parent.values()),
            n_refolds=n_refolds,
            n_feasible_children=sum(1 for e in evals if e.feasible),
            n_selected_children=n_selected, elite_risk=elite.particle.head_global_risk,
            ess=ess, resampled=(mode == "fk"),
            cache_hits=structure_cache.hits - hits_before, wall_time_s=time.time() - round_t0))

    return ProteinResult(
        protein_id=protein_id, initial_population=initial_population,
        final_population=population, elite=elite.particle,
        rounds=rounds, candidates=all_candidates, lineage=lineage, populations=populations,
        proposals=all_proposals)
