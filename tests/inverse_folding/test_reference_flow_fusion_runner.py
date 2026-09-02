"""F4/F5 tests — fusion/runner.py evaluator + structure firewall, and the greedy exact-state loop.

Contract: PLAN_RF_REFINE_FUSION.md §1.4 / Task F4 (this file's first half) and Task F5 (added
below). Fake Head/structure oracles — NO torch, NO real models. Head is scored once per
protein per stage on the deduplicated pool; refolds are per-parent Head-shortlisted and
bounded; only definitively feasible children can be inherited; the active-site shell gate is
enforced on anchored proteins and skipped (N/A) otherwise; a cache-provenance mismatch
recomputes rather than reusing a stale fold.
"""
from types import SimpleNamespace

import pytest

from inverse_folding.reference_flow.fusion import config as fc
from inverse_folding.reference_flow.fusion import oracles as orc
from inverse_folding.reference_flow.fusion import runner as rn
from inverse_folding.reference_flow.fusion import state as st

PARENT_SEQ = "A" * 50
GRID = [(0, 9, 9), (20, 29, 9), (40, 49, 9)]


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def zmap(z0, z20, z40):
    return {(0, 9, 9): z0, (20, 29, 9): z20, (40, 49, 9): z40}


class FakeHead:
    def __init__(self, score_map):
        self.score_map = score_map  # seq -> (global_risk, {(s,e,k): z})
        self.calls = []

    def __call__(self, protein_id, sequences):
        self.calls.append(list(sequences))
        out = []
        for s in sequences:
            gr, zm = self.score_map[s]
            windows = [SimpleNamespace(start_0b=a, end_0b=b, k=k, z=z) for (a, b, k), z in zm.items()]
            out.append(SimpleNamespace(windows=windows, global_risk=gr))
        return out


class FakeStruct:
    def __init__(self, metric_map):
        self.metric_map = metric_map  # seq -> dict(scTM=, asr=, scrmsd=)
        self.calls = []

    def __call__(self, protein_id, sequence):
        self.calls.append(sequence)
        m = self.metric_map[sequence]
        return SimpleNamespace(scTM=m["scTM"], active_site_RMSD=m.get("asr"),
                               scRMSD=m.get("scrmsd"), pLDDT=80.0, passed=True, reason="")


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def child_seq(pos, aa):
    s = list(PARENT_SEQ)
    s[pos] = aa
    return "".join(s)


def _parent(seq=PARENT_SEQ, gr=3.0, slot=0):
    return st.ParticleState(
        particle_id=st.make_particle_id("P", 0, slot, st.sequence_md5(seq)),
        protein_id="P", round_idx=0, slot_idx=slot, sequence=seq,
        sequence_md5=st.sequence_md5(seq), weight=1.0, parent_particle_id=None,
        source_proposal_id=None, head_global_risk=gr,
        structure=SimpleNamespace(scTM=0.95, active_site_RMSD=1.0, scRMSD=1.0, passed=True, reason=""),
        feasible=True, lineage_seed=0)


def _prop(parent, seq, edited=(20,)):
    return st.Proposal(
        proposal_id=st.make_proposal_id(parent.particle_id, "explicit_edit", st.sequence_md5(seq)),
        parent_particle_id=parent.particle_id, move_family="explicit_edit", sequence=seq,
        edited_positions=edited, target_start_0b=20, target_end_0b=25,
        halo_start_0b=20, halo_end_0b=29, proposal_seed=1)


def _cfg(*, max_refolds=8, scTM_min=0.85, active_site_max=2.0,
         active_site_metric="legacy_ca_shell", max_anchor_sidechain_max=None, offtarget_max=0.5,
         min_head_improvement=0.05, mode="greedy", n_rounds=1, population_size=1,
         explicit_enabled=True, rf_reopen_enabled=False,
         repair_enabled=False, repair_shortlist=2, children_per_edit=1, window_detail=False):
    return fc.load_fusion_config({"fusion": {
        "enabled": True, "seed": 1, "population_size": population_size, "n_rounds": n_rounds,
        "handoff": {"mode": "terminal_population"},
        "targeting": {"mode": "head_max_window", "registers_per_parent": 1, "halo_radius": 4},
        "objective": {"head_field": "global_risk", "min_head_improvement": min_head_improvement,
                      "max_offtarget_window_increase": offtarget_max},
        "moves": {"explicit": {"enabled": explicit_enabled, "max_edit_order": 2,
                               "max_raw_candidates_per_parent": 64, "pair_seed_budget": 16},
                  "rf_reopen": {"enabled": rf_reopen_enabled, "children_per_parent": 1},
                  "repair": {"enabled": repair_enabled, "repair_shortlist_per_parent": repair_shortlist,
                             "children_per_edit": children_per_edit, "sampler_steps": 8,
                             "local_remask": False}},
        "structure": {"backend": "esmfold", "surrogate_mode": "none", "scTM_min": scTM_min,
                      "active_site_metric": active_site_metric,
                      "active_site_RMSD_max": active_site_max, "active_site_shell_radius": 6.0,
                      "max_anchor_sidechain_RMSD_max": max_anchor_sidechain_max,
                      "scRMSD_max": None, "cache_dir": "/tmp/c", "max_refolds_per_parent": max_refolds},
        "selection": {"mode": mode, "beta_start": 1.0, "beta_end": 2.0, "resample_method": "multinomial"},
        "telemetry": {"candidate_detail": True, "window_detail": window_detail,
                      "trajectory_detail": True},
    }})


def _cache(cfg, digest="bbA"):
    return orc.StructureCache(backbone_digest=digest, backend="esmfold", config_hash=cfg.config_hash())


def _evaluate(parent, children_specs, *, cfg=None, has_active_site=False, cache=None):
    """children_specs: list of (seq, global_risk, zmap, metrics_dict)."""
    cfg = cfg or _cfg()
    head_map = {PARENT_SEQ: (parent.head_global_risk, zmap(1.0, 3.0, 1.0))}
    struct_map = {}
    props = []
    for seq, gr, zm, m in children_specs:
        head_map[seq] = (gr, zm)
        struct_map[seq] = m
        props.append(_prop(parent, seq))
    head = FakeHead(head_map)
    struct = FakeStruct(struct_map)
    oracles = orc.FusionOracles(head_fn=head, struct_fn=struct)
    cache = cache or _cache(cfg)
    evals = rn.evaluate_children(
        protein_id="P", parents=[parent], proposals_by_parent={parent.particle_id: props},
        oracles=oracles, structure_cache=cache, config=cfg, has_active_site=has_active_site)
    return evals, head, struct


def test_aligned_windows_populated_only_when_window_detail_enabled():
    parent = _parent()
    child = child_seq(20, "G")
    spec = [(child, 2.0, zmap(1.0, 2.0, 1.0), {"scTM": 0.95, "asr": 1.0, "scrmsd": 1.0})]
    on, _h, _s = _evaluate(parent, spec, cfg=_cfg(window_detail=True))
    assert on[0].aligned_windows                                        # S0 window landscape recorded
    coords = {(s, e, k) for (s, e, k, z) in on[0].aligned_windows}
    assert (20, 29, 9) in coords and all(isinstance(z, float) for *_r, z in on[0].aligned_windows)
    off, _h2, _s2 = _evaluate(parent, spec, cfg=_cfg(window_detail=False))
    assert off[0].aligned_windows == ()                                # detail off -> no window rows


# --------------------------------------------------------------------------- #
# F4: deduplicated one-batch Head scoring
# --------------------------------------------------------------------------- #
def test_head_scored_once_on_deduplicated_pool():
    parent = _parent()
    specs = [(child_seq(20, "C"), 2.0, zmap(1, 1, 1), {"scTM": 0.9, "asr": 1.0}),
             (child_seq(20, "D"), 2.5, zmap(1, 1, 1), {"scTM": 0.9, "asr": 1.0})]
    evals, head, struct = _evaluate(parent, specs)
    assert len(head.calls) == 1  # one batch for the whole stage
    batch = head.calls[0]
    assert len(batch) == len(set(batch))  # deduplicated
    assert set(batch) == {PARENT_SEQ, child_seq(20, "C"), child_seq(20, "D")}


# --------------------------------------------------------------------------- #
# F4: bounded, deterministic per-parent refold shortlist
# --------------------------------------------------------------------------- #
def test_refold_shortlist_bounded_and_lowest_risk_first():
    parent = _parent(gr=5.0)
    specs = [(child_seq(20, "C"), 3.0, zmap(1, 1, 1), {"scTM": 0.9, "asr": 1.0}),
             (child_seq(20, "D"), 1.0, zmap(1, 1, 1), {"scTM": 0.9, "asr": 1.0}),
             (child_seq(20, "E"), 2.0, zmap(1, 1, 1), {"scTM": 0.9, "asr": 1.0}),
             (child_seq(20, "F"), 4.0, zmap(1, 1, 1), {"scTM": 0.9, "asr": 1.0})]
    evals, head, struct = _evaluate(parent, specs, cfg=_cfg(max_refolds=2))
    # only the 2 lowest-global-risk hotspot-feasible children are refolded
    assert set(struct.calls) == {child_seq(20, "D"), child_seq(20, "E")}
    refolded = {e.sequence_md5 for e in evals if e.structure_evaluated}
    assert refolded == {st.sequence_md5(child_seq(20, "D")), st.sequence_md5(child_seq(20, "E"))}


def test_every_shortlisted_child_is_gated_and_feasible_only_when_passing():
    parent = _parent()
    specs = [(child_seq(20, "C"), 1.0, zmap(1, 1, 1), {"scTM": 0.90, "asr": 1.0}),   # pass
             (child_seq(20, "D"), 1.5, zmap(1, 1, 1), {"scTM": 0.80, "asr": 1.0})]   # scTM < min
    evals, head, struct = _evaluate(parent, specs)
    by_seq = {e.sequence_md5: e for e in evals}
    good = by_seq[st.sequence_md5(child_seq(20, "C"))]
    bad = by_seq[st.sequence_md5(child_seq(20, "D"))]
    assert good.structure_evaluated and good.feasible
    assert bad.structure_evaluated and not bad.feasible  # gated out by scTM floor


# --------------------------------------------------------------------------- #
# F4: hotspot gate keeps infeasible children out of refold and ancestry
# --------------------------------------------------------------------------- #
def test_offtarget_hotspot_child_not_refolded_and_infeasible():
    parent = _parent()
    # child raises the off-halo window (40,49) far above parent -> new hotspot
    specs = [(child_seq(20, "C"), 0.5, zmap(1, 1, 9.0), {"scTM": 0.99, "asr": 1.0})]
    evals, head, struct = _evaluate(parent, specs, cfg=_cfg(offtarget_max=0.5))
    assert struct.calls == []  # never refolded
    e = evals[0]
    assert not e.structure_evaluated and not e.feasible
    assert e.new_hotspot_max > 0.5


# --------------------------------------------------------------------------- #
# F4: active-site shell gate
# --------------------------------------------------------------------------- #
def test_active_site_gate_infeasible_when_anchored_and_shell_high():
    parent = _parent()
    specs = [(child_seq(20, "C"), 1.0, zmap(1, 1, 1), {"scTM": 0.95, "asr": 3.0})]  # asr > 2.0
    evals, head, struct = _evaluate(parent, specs, has_active_site=True)
    assert struct.calls  # it was shortlisted + refolded
    assert not evals[0].feasible  # active-site shell RMSD too high despite good scTM


def test_active_site_gate_skipped_when_no_anchors():
    parent = _parent()
    specs = [(child_seq(20, "C"), 1.0, zmap(1, 1, 1), {"scTM": 0.95, "asr": 9.9})]
    evals, head, struct = _evaluate(parent, specs, has_active_site=False)
    assert evals[0].feasible  # shell gate N/A for a no-anchor protein


def test_active_site_missing_metric_fails_closed_when_anchored():
    parent = _parent()
    specs = [(child_seq(20, "C"), 1.0, zmap(1, 1, 1), {"scTM": 0.95})]  # asr missing -> None
    evals, head, struct = _evaluate(parent, specs, has_active_site=True)
    assert not evals[0].feasible


# --------------------------------------------------------------------------- #
# F4: structure-cache provenance
# --------------------------------------------------------------------------- #
def test_cache_hit_avoids_recompute_same_provenance():
    cfg = _cfg()
    cache = _cache(cfg)
    struct = FakeStruct({PARENT_SEQ: {"scTM": 0.9, "asr": 1.0}})
    m1, hit1 = cache.evaluate("P", PARENT_SEQ, struct)
    m2, hit2 = cache.evaluate("P", PARENT_SEQ, struct)
    assert (hit1, hit2) == (False, True)
    assert len(struct.calls) == 1


def test_cache_provenance_mismatch_recomputes():
    cfg = _cfg()
    struct = FakeStruct({PARENT_SEQ: {"scTM": 0.9, "asr": 1.0}})
    ca = orc.StructureCache(backbone_digest="A", backend="esmfold", config_hash=cfg.config_hash())
    cb = orc.StructureCache(backbone_digest="B", backend="esmfold", config_hash=cfg.config_hash())
    ca.evaluate("P", PARENT_SEQ, struct)
    cb.evaluate("P", PARENT_SEQ, struct)
    assert len(struct.calls) == 2  # different backbone digest -> recompute, no stale reuse


# =========================================================================== #
# F5: minimum exact-state greedy loop (integration, functional fakes)
# =========================================================================== #
def _fwindows(seq):
    # static grid; the (20,29) register is highest-z so it is always the target, and
    # positions 20/21 (where risk drops) are the FIRST editable slots the proposer reaches.
    zm = {(20, 29, 9): 5.0, (0, 9, 9): 1.0, (35, 44, 9): 1.0}
    return [SimpleNamespace(start_0b=a, end_0b=b, k=k, z=z) for (a, b, k), z in zm.items()]


def _frisk(seq):
    # risk drops only when position 20 -> G (by 10) and/or 21 -> G (by 5)
    return 100.0 - 10.0 * (seq[20] == "G") - 5.0 * (seq[21] == "G")


class FuncHead:
    def __init__(self):
        self.calls = []

    def __call__(self, protein_id, sequences):
        self.calls.append(list(sequences))
        return [SimpleNamespace(windows=_fwindows(s), global_risk=_frisk(s)) for s in sequences]


class FuncStruct:
    def __init__(self, fail=lambda s: False):
        self.fail = fail
        self.calls = []

    def __call__(self, protein_id, sequence):
        self.calls.append(sequence)
        return SimpleNamespace(scTM=0.5 if self.fail(sequence) else 0.95,
                               active_site_RMSD=1.0, scRMSD=1.0, pLDDT=80.0, passed=True, reason="")


def _init_pop(seq=PARENT_SEQ, n=1, gr=100.0):
    parts = []
    for i in range(n):
        parts.append(st.ParticleState(
            particle_id=st.make_particle_id("P", 0, i, st.sequence_md5(seq)),
            protein_id="P", round_idx=0, slot_idx=i, sequence=seq,
            sequence_md5=st.sequence_md5(seq), weight=1.0 / n, parent_particle_id=None,
            source_proposal_id=None, head_global_risk=gr,
            structure=SimpleNamespace(scTM=0.95, active_site_RMSD=1.0, scRMSD=1.0, passed=True, reason=""),
            feasible=True, lineage_seed=i))
    elite = st.EliteState(particle=parts[0], first_round_seen=0, last_round_seen=0)
    return st.PopulationState(round_idx=0, particles=tuple(parts), elite=elite)


def _run(*, n_rounds=1, min_head_improvement=0.05, fail=lambda s: False, n=1, mode="greedy"):
    cfg = _cfg(mode=mode, n_rounds=n_rounds, population_size=n,
               min_head_improvement=min_head_improvement)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct(fail=fail))
    cache = _cache(cfg)
    return rn.run_protein(protein_id="P", initial_population=_init_pop(n=n), oracles=oracles,
                          structure_cache=cache, config=cfg, anchors=set())


def test_population_size_preserved():
    res = _run(n=3, n_rounds=2)
    assert len(res.final_population.particles) == 3


def test_improving_child_selected_and_persists_unchanged_next_round():
    # round 1 improves by 10 (>=7); round 2's only further gain is 5 (<7) -> parent kept
    res = _run(n_rounds=2, min_head_improvement=7.0)
    final = res.final_population.particles[0]
    assert final.sequence[20] == "G"           # the improving edit was carried
    assert final.head_global_risk == 90.0
    assert res.rounds[0].n_selected_children == 1
    assert res.rounds[1].n_selected_children == 0   # population unchanged (gain 5 < margin 7)
    # the independent elite archive still captures the strictly-better feasible child (85),
    # even though the ε_H margin kept it out of the population (P1-3 fix)
    assert res.elite.head_global_risk == 85.0


def test_structure_failed_lowest_head_child_cannot_inherit():
    # the lowest-Head edits (G at 20) all fail structure; a higher-Head feasible child wins
    res = _run(n_rounds=1, fail=lambda s: s[20] == "G")
    final = res.final_population.particles[0]
    assert final.sequence[20] != "G"
    assert final.sequence[21] == "G"
    assert final.head_global_risk == 95.0


def test_all_children_fail_parent_survives():
    res = _run(n_rounds=1, fail=lambda s: True)
    final = res.final_population.particles[0]
    assert final.sequence == PARENT_SEQ            # null move
    assert res.rounds[0].n_selected_children == 0
    assert res.elite.head_global_risk == 100.0


def test_elite_is_monotone():
    res = _run(n_rounds=2, min_head_improvement=0.05)
    risks = [rec.elite_risk for rec in res.rounds]
    assert risks == sorted(risks, reverse=True)    # never increases
    assert res.elite.head_global_risk <= 100.0


def test_no_nmp_dependency():
    assert "nmp_fn" not in orc.FusionOracles._fields
    # a run completes with only head_fn + struct_fn present
    res = _run(n_rounds=1)
    assert res.protein_id == "P"


# =========================================================================== #
# F7: beam / FK dispatch end-to-end + selector replay over the same pool
# =========================================================================== #
import pytest as _pytest  # noqa: E402


@_pytest.mark.parametrize("mode", ["greedy", "beam", "fk"])
def test_selector_produces_valid_population_and_monotone_elite(mode):
    res = _run(mode=mode, n=3, n_rounds=2, min_head_improvement=0.05)
    assert len(res.final_population.particles) == 3           # population size fixed
    assert res.elite.head_global_risk <= 100.0               # never worse than initial
    risks = [rec.elite_risk for rec in res.rounds]
    assert risks == sorted(risks, reverse=True)              # elite monotone in every mode


def test_fk_records_ess_and_resample_flag():
    res = _run(mode="fk", n=4, n_rounds=1)
    rec = res.rounds[0]
    assert rec.resampled is True
    assert rec.ess is not None and rec.ess > 0.0


def test_beam_and_fk_reach_the_improving_state():
    # both population selectors should surface the low-Head feasible child in the elite
    beam = _run(mode="beam", n=3, n_rounds=1)
    fk = _run(mode="fk", n=4, n_rounds=1)
    assert beam.elite.head_global_risk <= 90.0
    assert fk.elite.head_global_risk <= 90.0


def test_fk_duplicated_descendants_have_unique_ids_and_lineage():
    res = _run(mode="fk", n=4, n_rounds=1)
    parts = res.final_population.particles
    assert len({p.particle_id for p in parts}) == len(parts)          # globally unique ids
    seqs = [p.sequence_md5 for p in parts]
    assert len(set(seqs)) < len(seqs)                                 # a favored descendant duplicated
    child_ids = {p.particle_id for p in parts if p.source_proposal_id is not None}
    lineage_children = {e["child_particle_id"] for e in res.lineage if e["round_idx"] == 1}
    assert child_ids <= lineage_children                             # every child has a lineage edge


# =========================================================================== #
# adversarial: structure firewall must fail closed (P1-1)
# =========================================================================== #
def test_nan_scTM_fails_closed():
    cfg = _cfg()
    m = SimpleNamespace(scTM=float("nan"), active_site_RMSD=1.0, scRMSD=None, passed=True)
    ok, _reason = orc.structure_feasible(m, cfg, has_active_site=False)
    assert ok is False


def test_anchored_without_ceiling_fails_closed():
    # anchored protein but no active_site_RMSD_max configured -> must NOT silently pass
    cfg = _cfg(active_site_max=None)
    m = SimpleNamespace(scTM=0.99, active_site_RMSD=1.0, scRMSD=None, passed=True)
    ok, _reason = orc.structure_feasible(m, cfg, has_active_site=True)
    assert ok is False


def test_nan_active_site_rmsd_fails_closed_when_anchored():
    cfg = _cfg(active_site_max=2.0)
    m = SimpleNamespace(scTM=0.99, active_site_RMSD=float("nan"), scRMSD=None, passed=True)
    ok, _reason = orc.structure_feasible(m, cfg, has_active_site=True)
    assert ok is False


def test_none_active_site_metric_keeps_only_the_sctm_gate_for_anchored_proteins():
    cfg = _cfg(scTM_min=0.70, active_site_metric="none", active_site_max=None)
    metrics = SimpleNamespace(scTM=0.70, scRMSD=None)
    assert orc.structure_feasible(metrics, cfg, has_active_site=True) == (True, "ok")


def test_sidechain_active_site_gate_uses_complete_max_anchor_metric():
    cfg = _cfg(
        active_site_metric="sidechain_max_anchor",
        active_site_max=None,
        max_anchor_sidechain_max=1.5,
    )
    good = SimpleNamespace(
        scTM=0.99,
        max_anchor_sidechain_RMSD=1.2,
        active_site_complete=True,
        scRMSD=None,
    )
    assert orc.structure_feasible(good, cfg, has_active_site=True) == (True, "ok")

    incomplete = SimpleNamespace(
        scTM=0.99,
        max_anchor_sidechain_RMSD=1.2,
        active_site_complete=False,
        scRMSD=None,
    )
    ok, reason = orc.structure_feasible(incomplete, cfg, has_active_site=True)
    assert ok is False and "incomplete" in reason

    high = SimpleNamespace(
        scTM=0.99,
        max_anchor_sidechain_RMSD=1.8,
        active_site_complete=True,
        scRMSD=None,
    )
    ok, reason = orc.structure_feasible(high, cfg, has_active_site=True)
    assert ok is False and "max_anchor_sidechain_RMSD" in reason


# =========================================================================== #
# adversarial: elite archive, initial population, fresh parent, config/exec (P1-3/4/5/7)
# =========================================================================== #
def test_elite_captures_feasible_child_rejected_by_margin():
    # margin 15 rejects the 20->G child (gain 10), but it is FEASIBLE at risk 90
    res = _run(n_rounds=1, min_head_improvement=15.0)
    assert res.final_population.particles[0].sequence == PARENT_SEQ   # greedy kept the parent
    assert res.final_population.particles[0].head_global_risk == 100.0
    assert res.elite.head_global_risk == 90.0                        # archive kept the feasible best
    assert res.elite.sequence[20] == "G"


def test_selection_uses_fresh_parent_head_not_stale():
    # stale stored parent risk 50 is LOWER than the true 100; only fresh scoring selects the child
    cfg = _cfg(mode="greedy", n_rounds=1, min_head_improvement=0.05, population_size=1)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    stale = _init_pop(n=1, gr=50.0)
    res = rn.run_protein(protein_id="P", initial_population=stale, oracles=oracles,
                         structure_cache=_cache(cfg), config=cfg, anchors=set())
    assert res.final_population.particles[0].sequence[20] == "G"  # fresh 100 -> improvement selected


def test_run_protein_rejects_population_size_mismatch():
    cfg = _cfg(population_size=4, n_rounds=1)      # config says N=4
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    with pytest.raises(rn.FusionRunnerError):
        rn.run_protein(protein_id="P", initial_population=_init_pop(n=1), oracles=oracles,
                       structure_cache=_cache(cfg), config=cfg, anchors=set())


def test_run_protein_anchored_without_ceiling_raises():
    cfg = _cfg(population_size=1, active_site_max=None, n_rounds=1)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    with pytest.raises(rn.FusionRunnerError):
        rn.run_protein(protein_id="P", initial_population=_init_pop(n=1), oracles=oracles,
                       structure_cache=_cache(cfg), config=cfg, anchors={20})


def test_run_protein_rejects_out_of_range_anchor():
    cfg = _cfg(population_size=1, n_rounds=1)  # PARENT_SEQ len 50
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    with pytest.raises(rn.FusionRunnerError):
        rn.run_protein(protein_id="P", initial_population=_init_pop(n=1), oracles=oracles,
                       structure_cache=_cache(cfg), config=cfg, anchors={999})


def test_run_protein_rejects_negative_anchor():
    cfg = _cfg(population_size=1, n_rounds=1)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    with pytest.raises(rn.FusionRunnerError):
        rn.run_protein(protein_id="P", initial_population=_init_pop(n=1), oracles=oracles,
                       structure_cache=_cache(cfg), config=cfg, anchors={-1})


def test_run_protein_requires_explicit_move_family():
    cfg = _cfg(population_size=1, n_rounds=1, explicit_enabled=False, rf_reopen_enabled=True)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    with pytest.raises(NotImplementedError):
        rn.run_protein(protein_id="P", initial_population=_init_pop(n=1), oracles=oracles,
                       structure_cache=_cache(cfg), config=cfg, anchors=set())


def test_refold_count_excludes_cache_hits():
    # two identical parents -> the shared child sequences fold once; n_refolds counts misses only
    res = _run(mode="greedy", n=2, n_rounds=1)
    # every refold recorded is a genuine ESMFold call (cache miss), not a hit
    assert res.rounds[0].n_refolds <= sum(1 for e in res.candidates if e.structure_evaluated)


# --------------------------------------------------------------------------- #
# build_initial_population (§1.7 terminal handoff)
# --------------------------------------------------------------------------- #
def test_build_initial_selects_n_feasible_in_order_with_beta0_weights():
    rows = [{"design_idx": 0, "seed": 0, "sequence": PARENT_SEQ},
            {"design_idx": 1, "seed": 0, "sequence": child_seq(20, "G")},
            {"design_idx": 2, "seed": 0, "sequence": child_seq(21, "G")}]
    cfg = _cfg(population_size=2, n_rounds=1)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                     structure_cache=_cache(cfg), config=cfg, anchors=set())
    assert [p.sequence for p in pop.particles] == [PARENT_SEQ, child_seq(20, "G")]  # design_idx order
    assert pop.elite.particle.sequence == child_seq(20, "G")            # lowest Head (90)
    assert sum(p.weight for p in pop.particles) == pytest.approx(1.0)
    w = {p.sequence: p.weight for p in pop.particles}
    assert w[child_seq(20, "G")] > w[PARENT_SEQ]                        # beta_0 favors lower risk


def test_build_initial_insufficient_feasible_raises():
    rows = [{"design_idx": 0, "seed": 0, "sequence": PARENT_SEQ}]
    cfg = _cfg(population_size=3)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    with pytest.raises(rn.FusionRunnerError):
        rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                    structure_cache=_cache(cfg), config=cfg, anchors=set())


def test_build_initial_skips_structure_infeasible_rows():
    rows = [{"design_idx": 0, "seed": 0, "sequence": PARENT_SEQ},
            {"design_idx": 1, "seed": 0, "sequence": child_seq(20, "G")},   # will fail structure
            {"design_idx": 2, "seed": 0, "sequence": child_seq(21, "G")}]
    cfg = _cfg(population_size=2)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct(fail=lambda s: s[20] == "G"))
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                     structure_cache=_cache(cfg), config=cfg, anchors=set())
    assert [p.sequence for p in pop.particles] == [PARENT_SEQ, child_seq(21, "G")]


def test_build_initial_skips_anchor_mismatch():
    rows = [{"design_idx": 0, "seed": 0, "sequence": child_seq(20, "G")},  # violates anchor 20==A
            {"design_idx": 1, "seed": 0, "sequence": PARENT_SEQ}]
    cfg = _cfg(population_size=1, active_site_max=2.0)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                     structure_cache=_cache(cfg), config=cfg,
                                     anchors={20}, anchor_expected={20: "A"})
    assert pop.particles[0].sequence == PARENT_SEQ


def test_build_initial_requires_anchor_expected_when_anchored():
    # anchored protein (positions given) MUST also supply expected residues or §1.7 fails open
    rows = [{"design_idx": 0, "seed": 0, "sequence": PARENT_SEQ}]
    cfg = _cfg(population_size=1, active_site_max=2.0)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    with pytest.raises(rn.FusionRunnerError):
        rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                    structure_cache=_cache(cfg), config=cfg, anchors={20})


def test_run_protein_rejects_disabled_explicit_even_with_zero_rounds():
    cfg = _cfg(population_size=1, n_rounds=0, explicit_enabled=False, rf_reopen_enabled=True)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    with pytest.raises(NotImplementedError):
        rn.run_protein(protein_id="P", initial_population=_init_pop(n=1), oracles=oracles,
                       structure_cache=_cache(cfg), config=cfg, anchors=set())


# =========================================================================== #
# F6 runner integration: edit-repair merged into the round pool
# =========================================================================== #
def _fake_repair_regen(base, frozen, regen, seed):
    chars = list(base)
    for p in sorted(regen):
        chars[p] = "ACDEFGHIKLMNPQRSTVWY"[(seed + p) % 20]
    return "".join(chars)


def test_repair_enabled_requires_repair_fn():
    cfg = _cfg(population_size=1, repair_enabled=True)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    with pytest.raises(rn.FusionRunnerError):
        rn.run_protein(protein_id="P", initial_population=_init_pop(n=1), oracles=oracles,
                       structure_cache=_cache(cfg), config=cfg, anchors=set())  # no repair_fn


def test_edit_repair_integration_adds_repaired_descendants_to_pool():
    cfg = _cfg(population_size=1, n_rounds=1, repair_enabled=True,
               repair_shortlist=2, children_per_edit=1)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    res = rn.run_protein(protein_id="P", initial_population=_init_pop(n=1), oracles=oracles,
                         structure_cache=_cache(cfg), config=cfg, anchors=set(),
                         repair_fn=_fake_repair_regen)
    families = {p.move_family for p in res.proposals.values()}
    assert "explicit_edit" in families
    assert "edit_repair" in families            # repaired descendants integrated into the pool
    # every repaired descendant preserves its protected immune edit residue
    for prop in res.proposals.values():
        if prop.move_family == "edit_repair":
            for pos in prop.edited_positions:
                assert prop.sequence[pos] != PARENT_SEQ[pos]


class _MarginHead:
    """Target register (20,29); a margin window (10,17) sits OUTSIDE the target span but INSIDE
    the repair halo (target +/- 4 = [16,33)). Editing position 20 spikes the margin window above
    the off-target ceiling. Under the correct repair-halo gate that spike is in-halo (repair will
    regenerate it) so the improving pos-20->G edit stays in the repair shortlist; a target-span
    gate would treat it as off-target and drop the only fixable candidate."""

    def __init__(self):
        self.calls = []

    def __call__(self, protein_id, sequences):
        self.calls.append(list(sequences))
        out = []
        for s in sequences:
            margin_z = 2.0 if s[20] != "A" else 0.0  # any pos-20 edit spikes the margin window
            zm = {(20, 29, 9): 5.0, (10, 17, 9): margin_z, (40, 49, 9): 0.0}
            windows = [SimpleNamespace(start_0b=a, end_0b=b, k=k, z=z) for (a, b, k), z in zm.items()]
            out.append(SimpleNamespace(windows=windows, global_risk=100.0 - 10.0 * (s[20] == "G")))
        return out


def test_repair_shortlist_gate_measures_offtarget_against_repair_halo_not_target_span():
    # halo_radius=4, target (20,29) -> repair halo [16,33) overlaps the margin window (10,17);
    # the pos-20->G edit (best global risk) must survive the off-target gate and seed edit-repair.
    cfg = _cfg(population_size=1, n_rounds=1, repair_enabled=True,
               repair_shortlist=2, children_per_edit=1)
    oracles = orc.FusionOracles(head_fn=_MarginHead(), struct_fn=FuncStruct())
    res = rn.run_protein(protein_id="P", initial_population=_init_pop(n=1), oracles=oracles,
                         structure_cache=_cache(cfg), config=cfg, anchors=set(),
                         repair_fn=_fake_repair_regen)
    seeded_from_pos20_G = [p for p in res.proposals.values()
                           if p.move_family == "edit_repair"
                           and p.edited_positions == (20,) and p.sequence[20] == "G"]
    assert seeded_from_pos20_G, "fixable pos-20 edit was dropped by a target-span off-target gate"


class _SubEpsilonHead:
    """Editing position 20 -> C yields a feasible improvement SMALLER than the eps_H beam margin:
    the child is beam-INELIGIBLE yet is the global-best feasible state, so the independent archive
    captures it as the elite and beam must protect it into the population (the P1-5 trigger)."""

    def __init__(self):
        self.calls = []

    def __call__(self, protein_id, sequences):
        self.calls.append(list(sequences))
        out = []
        for s in sequences:
            zm = {(20, 29, 9): 5.0, (0, 9, 9): 1.0, (40, 49, 9): 1.0}
            w = [SimpleNamespace(start_0b=a, end_0b=b, k=k, z=z) for (a, b, k), z in zm.items()]
            out.append(SimpleNamespace(windows=w, global_risk=100.0 - 0.02 * (s[20] == "C")))
        return out


def test_beam_protected_elite_has_replayable_lineage_and_persisted_parent():
    # eps_H = 0.05; the pos-20->C child improves by only 0.02 -> beam-ineligible but archived as
    # elite. The protected carry must inherit the elite's REAL provenance (persisted parent +
    # resolvable originating proposal), not point at the archive id with a null proposal.
    cfg = _cfg(population_size=1, n_rounds=1, mode="beam", min_head_improvement=0.05)
    oracles = orc.FusionOracles(head_fn=_SubEpsilonHead(), struct_fn=FuncStruct())
    res = rn.run_protein(protein_id="P", initial_population=_init_pop(n=1), oracles=oracles,
                         structure_cache=_cache(cfg), config=cfg, anchors=set())
    final = res.populations[-1]
    protected = [p for p in final.particles if p.sequence[20] == "C"]
    assert protected, "sub-eps_H feasible best was not protected into the beam population"
    pc = protected[0]
    particle_ids = {p.particle_id for pop in res.populations for p in pop.particles}
    # parent must be a persisted particle in the particle table, not a dangling elite-archive id
    assert pc.parent_particle_id in particle_ids
    edges = [e for e in res.lineage if e["child_particle_id"] == pc.particle_id]
    assert edges and edges[0]["selector"] == "elite_protect"
    assert edges[0]["proposal_id"] in res.proposals  # exact-ancestry replay resolves


def test_beam_elite_archive_is_distinct_and_no_particle_self_loops():
    res = _run(mode="beam", n=3, n_rounds=2)
    assert ":elite:" in res.elite.particle_id                     # distinct elite-archive id scheme
    assert res.elite.parent_particle_id != res.elite.particle_id  # elite never its own parent
    for pop in res.populations:                                   # no population particle self-loops
        for p in pop.particles:
            assert p.parent_particle_id != p.particle_id


def test_round_proposal_pool_has_no_duplicate_sequences():
    cfg = _cfg(population_size=1, n_rounds=1, repair_enabled=True, repair_shortlist=2, children_per_edit=2)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    res = rn.run_protein(protein_id="P", initial_population=_init_pop(n=1), oracles=oracles,
                         structure_cache=_cache(cfg), config=cfg, anchors=set(),
                         repair_fn=_fake_repair_regen)
    seqs = [p.sequence for p in res.proposals.values()]
    assert len(seqs) == len(set(seqs))  # global dedup across explicit/reopen/repair


def test_build_initial_rejects_wrong_backbone_length():
    rows = [{"design_idx": 0, "seed": 0, "sequence": PARENT_SEQ}]  # len 50
    cfg = _cfg(population_size=1)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    with pytest.raises(rn.FusionRunnerError):  # expected_length 60 != 50 -> skipped -> insufficient
        rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                    structure_cache=_cache(cfg), config=cfg, anchors=set(),
                                    expected_length=60)
