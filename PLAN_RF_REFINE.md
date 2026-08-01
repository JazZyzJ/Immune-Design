# RF Refinement Mode (targeted epitope elimination) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:executing-plans` to run
> this task-by-task; `superpowers:test-driven-development` for each RED→GREEN step.
> Build tasks (§4) are for the **coder**; experiments (§5) are for an **agent to run
> directly after the coder finishes**, using the exact commands + criteria given.
> Status: **R1, R2 DONE** (implemented + tests green). Start at **R3**.

**Goal:** A post-hoc **refinement** stage that takes a best-of-N RF de-immunized design
and, by mutating a handful of residual high-risk positions, drives the count of distinct
MHC-II epitope cores toward **0** while preserving fold (⇒ activity) and freezing
active-site anchors.

**Design source:** this file + RAR `0024-step-0-refine-feasibility-residual-core-`
(Step-0 feasibility). Enzyme anchors reuse `inverse_folding/reference_flow/constraints.py`
/ `PLAN_URICASE_ENZYME_MODE.md`.

**Architecture:** A standalone stage **between** generation (`scripts/run_if_phase_c1.py`)
and evaluation (`scripts/evaluate_phase_c.py`). Pure search/selection logic lives in
`inverse_folding/reference_flow/refine.py`; the three expensive oracles (immune head,
NetMHCIIpan, ESMFold refold) are **injected as callables**, so the logic is unit-testable
without them. A driver `scripts/refine_rf_designs.py` wires the real oracles and does I/O.
Mechanism = local combinatorial search on the sequence (NOT re-running the RF sampler; the
sampler's `fixed_tokens` inpainting path is the deferred alternate in §9).

**Tech stack:** Python 3.12 (`immune-design` env), NumPy, pandas/pyarrow, existing
`inverse_folding/reference_flow/head_scoring.py` (`OnlineHeadScorer`),
`epitope_head/data/netmhciipan_runner.py` (`score_batch`),
`inverse_folding/evaluation/refold.py` (`refold`) + `run_tmalign`,
`inverse_folding/reference_flow/constraints.py`.

---

## 1. Scientific Framing And Frozen Decisions (authoritative for v0)

**Objective (primary):** minimise the number of **distinct NMP strong epitope cores**
(`rank_EL < 2%`), target **0**. This is the `n_distinct_cores` metric of RAR 0024 (median 5,
range 1–12 over char23/0701; Q00511 = 3). Head risk is an inner-loop proxy, not the objective.

**Roles — Mode 1 (NMP-dominant, the v0 default):**
- **NMP** defines the target cores, is the periodic in-loop check, and the **final gate**.
  Legal: NMP is the decision-layer validator, not a generation guidance signal — the search
  optimises head/NMP-surrogate, never trains against NMP (anti-circularity preserved).
- **Head** (`OnlineHeadScorer`) is the cheap inner-loop ranker (scores many candidates per
  NMP call).
- **Refold** (a `StructureMetrics` bundle: ESMFold pLDDT + TMalign scTM vs WT backbone, and
  later scRMSD / active-site RMSD) is the structure/activity gate.

**Experimental framing makes NMP-as-gate honest:** phase-1 wet-lab readout is **activity
recovery**, not an immunogenicity assay. NMP is an in-silico filter producing
activity-testable candidates, not an immune-efficacy claim (that needs later blood work).

**Soft search progress, hard output accept.** The distinct-core-count drop is the FINAL
accept for the output shortlist; as a per-step rule it stalls multi-point synergy (a move
can push a core 0.2%→1.8% without crossing 2%). The search carries a **beam** ordered by
`(distinct_core_count asc, rank_margin_mass asc)`, where
`rank_margin_mass = Σ over cores with rank_EL < margin_band of max(strong_rank − rank_EL, 0)`
— a continuous NMP surrogate rewarding progress toward the 2% threshold, read for free off
the candidates already sent to NMP.

**Structure is gated on outputs, not on every search step.** Mutations are not monotonically
fold-degrading (n64 data: 66/1532 designs are global-ok-but-active-site-shifted; scRMSD has a
heavy tail but many edits are neutral/positive), so structure must NOT prune the immune search
mid-flight (it would kill candidates a later edit heals, and waste refolds). v0 policy:
(i) the beam advances on immune `(count, margin)` only — **margin-progress states enter the
beam with NO refold**; (ii) a candidate that **drops the distinct-core count** is a potential
OUTPUT → refold + `structure_gate` it now: pass → shortlist + beam, **fail → excluded from both**
(never stack on a broken backbone); (iii) branch-waste is bounded by a cheap, refold-free
`max_path_mutations` cap (a count-0 solution needing many edits is structurally doomed anyway).

**Effect ceiling before cost — but pure NMP-all is infeasible on multi-core seeds.** The pure
ceiling (`topB=None`, NMP every candidate) exploded in practice (H2ETE7 round 1: pool 2208 →
60 min NMP + 80 min refold → walltime kill): the pool is `beam_width × block enumeration`, and
single mutations kill cores readily so most candidates drop the count. v0 therefore runs
**head-pruned from the start** (`topB≈64`) with a per-round **`refold_cap≈16`**; the NMP-all
ceiling probe (E0) is kept only on a few EASY (few-core) seeds where it is affordable, to
calibrate what the prune/cap cost. head proxy = target-window risk, so top-B surfaces
count-droppers by construction (the load-bearing E0 measurement).

**Block-structured enumeration is the primary proposer.** Search unit = connected-component
hotspot **block** (overlapping strong windows), not a single core. Per block: exhaustive
singles (editable positions × AA20), rank by **head only** (no cheap in-loop structural prior
exists; refold stays at the gate), pairs among the top head-ranked single (pos, AA) choices,
triples only for whack-a-mole blocks. Uniform MC is supplementary exploration only.

**Editable set & move.** Per target core, editable = its P1/P4/P6/P9 pockets ∪ high
per-residue head positions in the core span, **minus** frozen active-site anchor positions
(freeze only the exact anchor AA; the shell stays mutable — 2026-06-25/30 decision). A move
is a multi-point combination; overlapping cores are co-targeted as one block (whack-a-mole).
**Acceptance is on the global distinct-core count (NMP), never per-core** — a move killing
core A but creating core B is rejected.

**Structure reference.** scTM/RMSD are measured against the **WT/target backbone** in
`pdb_root` (the inverse-folding design target; decision-layer eval, not a firewall violation).
Floor anchored at the **seed design's** metrics (accept iff `scTM ≥ scTM₀ − ε`, plus optional
scRMSD / active-site-RMSD ceilings when wired). pLDDT is self-referential.

**Two modes (keep the interface).** `--target-source nmp` (Mode 1, default) vs
`--target-source head` (Mode 2, deferred use — to demonstrate head-beats-NMP once blood data
exists). v0 implements Mode 1 fully; Mode 2 is a wired switch.

**R0/E0 candidate-level agreement is the load-bearing risk** (distinct from the sequence-level
0.635 in RAR 0024): does head rank the NMP-eliminating candidates into `topB`? E0 measures it
before the full E1 run.

---

## 2. Architecture, File Structure, Oracle Interfaces

| File | Responsibility | Status |
|---|---|---|
| `inverse_folding/reference_flow/refine.py` | Pure logic: `EpitopeCore`/`Candidate`/`StructureMetrics`, `extract_target_cores`, `editable_positions`, `hotspot_blocks`, `enumerate_singles`/`enumerate_pairs`, `rank_margin_mass`, `structure_gate`, `accept_refinement`, `refine_sequence`. No torch, no I/O. | R1+R2 DONE; R3 pending |
| `scripts/refine_rf_designs.py` | Driver: load a gen+eval run, build the real oracle callables, run `--mode ceiling` (score all → dump candidate table) or `--mode refine` (beam search → shortlist), write outputs incl. an evaluator-ready parquet. | R4 |
| `tests/inverse_folding/test_refine.py` | TDD for every pure function, via fake oracles. | R1+R2 done; R3 tests pending |
| `tests/scripts/test_refine_rf_designs.py` | Driver smoke: fake oracles + 2-row generated.parquet → assert both output schemas + a killable-core row reaches count 0. | R4 |
| `doc/SCRIPTS.md` | Register `refine_rf_designs.py`. | R4 |

**Reuse-First (checked `doc/SCRIPTS.md`):** generation is sampling-only, eval is
scoring-only; refinement is a new stage between them → one new driver, registered in §8.

**Injected oracle callables** (the driver builds these from the real APIs in §4-R4):

```python
HeadFn   = Callable[[str, list[str]], "np.ndarray"]           # (protein_id, seqs) -> window_risks [K, W]
NmpFn    = Callable[[str, list[str]], list[list[dict]]]       # (protein_id, seqs) -> per-seq rows {pos,pep_length,peptide,core,rank_EL}
StructFn = Callable[[str, str], "StructureMetrics"]           # (protein_id, seq)  -> StructureMetrics
```

**NMP is BATCHED (confirmed 2026-07-02):** `NmpFn` takes a LIST of sequences and
`refine_sequence` scores each round's whole `chosen` set in ONE `nmp_fn` call —
NetMHCIIpan model-load is amortized across the batch (~10-13 s/seq batched vs ~15-28
single-entry; E0 ceiling scores 1-3k candidates in one round, so per-candidate calls
are infeasible). `head_fn` is likewise batched; `struct_fn` stays per-sequence (only
improving candidates refold).

**Confirmed R4 oracle-wiring corrections (vs the R4 sketch below):**
1. `rank_EL` unit = **fraction** (runner `el_rank` is 0-1, e.g. 0.02=2%); `refine.py`
   defaults `strong_rank=0.02`/`margin_band=0.10` are fractions. `nmp_fn` passes the
   raw `el_rank` **without** the ×100 that `immunogenicity.peptide_scores_to_dataframe`
   applies. Do NOT ingest `imm_nmp_peptides.parquet.rank_EL` (percent) into
   `extract_target_cores` without `/100`; targets come from the in-loop `nmp_fn`.
2. `nmp_fn` calls `StandaloneRunner.score_batch(entries=[(pid,s) for s in seqs], allele, NMP_PEP_LENGTHS)`
   → returns nested `dict[pid][pep_length] -> list[PeptideScore]`; flatten per input seq
   to rows using PeptideScore fields `pos/peptide/core/el_rank`. Entry ids must be unique.
3. `struct_fn`: `run_tmalign(pred_pdb=…, ref_pdb=…, tmalign_bin=…, cache_dir=None)` returns
   a **dict** `{"tm_score", "rmsd"}`, NOT a tuple; `ref_pdb = resolve_structure_path(test_row, pdb_root)`
   where `resolve_structure_path(entry, pdb_root)` takes a test-set ROW → the driver needs
   `--test-set-parquet` + a `{protein_id: row}` lookup. `refold(...)['pdb_path']` is `None`
   unless `--esmfold-cache-dir` is set → that flag is **required** for refine/ceiling.
4. Inputs are 8-shard, unmerged → glob `generation/*shard*of08*/generated.parquet` and
   `eval_immune/*shard*of08*/imm_{head,nmp_peptides}.parquet` and concat.
5. Head predictor+scorer: reuse the existing `build_head_scorer` helper
   (`run_if_phase_c1.py:1160`), not a new inline build.

`StructureMetrics` is a dataclass (`scTM, pLDDT, scRMSD, active_site_RMSD, passed, reason`).
The verdict logic lives in `structure_gate(seed, cand)` and the driver's `struct_fn`, so
adding scRMSD / active-site-RMSD ceilings never changes `accept_refinement`'s signature. In
v0 only `scTM`/`pLDDT` are populated (the rest `None`). Point mutations preserve length, so
the head window template is stable and `[K, W]` rows are comparable across candidates.

---

## 3. v0 Search Law (Mode 1)

```text
seed s0 (a best-of-N design), anchors A, allele a
T0 = extract_target_cores(NMP(s0), A); seed_count=|T0|; seed_margin=rank_margin_mass(T0)
seed_metrics = StructFn(s0)                              # seed is a real design (structure known)
beam = [state(s0, T0, seed_metrics)]; best = s0; shortlist = []
repeat up to max_rounds, stop if best.count == 0:
    pool = { propose(st) : st in beam }                 # block singles + head-ranked pairs (per state)
    W    = HeadFn(pid, pool.seqs); proxy = max over each state's target windows
    chosen = pool            if topB is None (ceiling)  # NMP-all establishes the ceiling
             top-topB(proxy) otherwise (cost mode)
    beam_admits = []
    for c in chosen:
        st_c = NMP-evaluate(c)                           # cores, count, rank_margin_mass
        if not improves-parent(count↓ or count= & margin↓) or hamming(c,seed) > max_path_mutations:
            continue
        if st_c.count < seed_count:                       # potential OUTPUT -> refold + gate NOW
            m_c = StructFn(c); passed = structure_gate(seed_metrics, m_c)
            if not (passed or allow_structure_unknown):   # structure-failed output -> drop (no beam)
                continue
            shortlist += {c, st_c.count, m_c}; best = min(best, by count); beam_admits += st_c
        else:                                             # margin-progress only -> beam, NO refold
            beam_admits += st_c
    beam = top-beam_width( beam ∪ beam_admits , key=(count asc, rank_margin_mass asc) )
    stop if beam-best (count, margin) has not improved for `patience` rounds
return RefineResult(best, shortlist ranked by count asc, trace, diverged = best.count==seed_count)
```

`--mode ceiling` runs one round with `topB=None` and no beam recursion (score every proposed
candidate, refold every improving one) — E0's data + the effect ceiling. `--mode refine` runs
the full loop.

---

## 4. Build Tasks (coder)

### Task R1 — core extraction, blocks, editable  ✅ DONE

`refine.py`: `MHC2_POCKETS`, `EpitopeCore`, `extract_target_cores`, `editable_positions`,
`hotspot_blocks`. Tests green (`test_extract_*`, `test_pocket_within_15mer_*`, `test_editable_*`,
`test_hotspot_blocks_*`). Run: `pytest tests/inverse_folding/test_refine.py -q`.

### Task R2 — block-structured enumeration  ✅ DONE

`refine.py`: `AA20`, `Candidate`, `enumerate_singles`, `enumerate_pairs`. Tests green
(`test_singles_*`, `test_pairs_*`).

### Task R3 — margin, structure gate, accept rule, beam search loop  (TDD)

**Files:** modify `inverse_folding/reference_flow/refine.py`; add tests to
`tests/inverse_folding/test_refine.py`.

- [ ] **Step 1 — write failing tests** (fake oracles; load via the importlib harness already
  at the top of the test file):

```python
import numpy as np
rank_margin_mass = refine.rank_margin_mass
structure_gate = refine.structure_gate
accept_refinement = refine.accept_refinement
refine_sequence = refine.refine_sequence
StructureMetrics = refine.StructureMetrics

_OK = StructureMetrics(scTM=0.90, pLDDT=90.0)          # passes any reasonable floor


def _propose_singles(seq, cores):
    ed = sorted({p for c in cores for p in c.editable_positions})
    return refine.enumerate_singles(seq, ed)


def test_rank_margin_mass_rewards_pushing_toward_threshold():
    assert rank_margin_mass([0.002], strong_rank=0.02, margin_band=0.10) == 0.018   # deep = most mass
    assert round(rank_margin_mass([0.018], strong_rank=0.02, margin_band=0.10), 3) == 0.002
    assert rank_margin_mass([0.05], strong_rank=0.02, margin_band=0.10) == 0.0      # outside band


def test_structure_gate_enforces_scTM_floor():
    seed = StructureMetrics(scTM=0.80, pLDDT=90.0)
    ok, _ = structure_gate(seed, StructureMetrics(scTM=0.78, pLDDT=88.0), scTM_eps=0.05)
    bad, reason = structure_gate(seed, StructureMetrics(scTM=0.70, pLDDT=88.0), scTM_eps=0.05)
    assert ok is True and bad is False and "scTM" in reason


def test_accept_requires_count_drop_and_structure_pass():
    assert accept_refinement(seed_core_count=3, cand_core_count=2, structure_passed=True) is True
    assert accept_refinement(seed_core_count=3, cand_core_count=3, structure_passed=True) is False
    assert accept_refinement(seed_core_count=3, cand_core_count=2, structure_passed=False) is False


def test_refine_drives_count_to_zero_single_step():
    seq = "A" * 30
    def nmp(pid, s):
        return [] if s[10] != "A" else [
            {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": 0.001}]
    def head(pid, seqs):
        return np.array([[0.0 if s[10] != "A" else 9.0] for s in seqs])
    def struct(pid, s):
        return _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=5, scTM_eps=0.05, target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 0 and res.best_seq[10] != "A" and res.n_accepts >= 1


def test_soft_beam_enables_two_step_elimination():
    # the core needs TWO edits (positions 10 and 13) to cross 2%; each single edit only lowers
    # the rank (count stays 1). Only a soft beam that carries the margin-improved 1-edit state
    # (which must PASS the structure gate to be admitted) can reach 0 on the next round.
    seq = "A" * 30
    def nmut(s):
        return sum(1 for p in (10, 13) if s[p] != "A")
    def nmp(pid, s):
        rank = {0: 0.001, 1: 0.015}.get(nmut(s), 0.05)
        return [] if rank >= 0.02 else [
            {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": rank}]
    def head(pid, seqs):
        return np.array([[-nmut(s)] for s in seqs])
    def struct(pid, s):
        return _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=6, patience=3, scTM_eps=0.05,
                          target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 0 and res.n_accepts >= 1


def test_structure_floor_blocks_fold_breaking_elimination_and_beam():
    seq = "A" * 30
    def nmp(pid, s):
        return [] if s[10] != "A" else [
            {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": 0.001}]
    def head(pid, seqs):
        return np.array([[0.0 if s[10] != "A" else 9.0] for s in seqs])
    def struct(pid, s):    # the only count-dropping edit also breaks the fold
        return StructureMetrics(scTM=0.50, pLDDT=40.0) if s[10] != "A" else _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=5, scTM_eps=0.05, target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 1 and res.n_accepts == 0    # rejected; never admitted to the beam
```

- [ ] **Step 2:** run `pytest tests/inverse_folding/test_refine.py -q` → FAIL (`rank_margin_mass`
  / `structure_gate` / `refine_sequence` missing).
- [ ] **Step 3 — implement** in `refine.py` (add `from dataclasses import dataclass, field, replace`
  to the imports):

```python
@dataclass(frozen=True)
class StructureMetrics:
    scTM: float
    pLDDT: float
    scRMSD: float | None = None            # populated when wired; None in v0
    active_site_RMSD: float | None = None  # populated when the active-site gate is wired
    passed: bool = True                    # set by structure_gate, relative to the seed
    reason: str = ""


def rank_margin_mass(core_ranks, *, strong_rank: float = 0.02, margin_band: float = 0.10) -> float:
    """Continuous NMP surrogate to MINIMISE: Σ over cores with rank_EL < margin_band of
    max(strong_rank − rank_EL, 0). Deep strong cores contribute most; a core pushed to
    ≥ strong_rank contributes 0 (eliminated from the strong count)."""
    return float(sum(max(strong_rank - r, 0.0) for r in core_ranks if r < margin_band))


def structure_gate(seed: "StructureMetrics", cand: "StructureMetrics", *, scTM_eps: float,
                   scRMSD_max: float | None = None,
                   active_site_RMSD_max: float | None = None) -> tuple[bool, str]:
    """Verdict for a candidate fold vs the seed design. v0 enforces the scTM floor
    (scTM ≥ seed.scTM − scTM_eps); the scRMSD / active-site-RMSD ceilings activate only when
    their thresholds are passed (None in v0). Returns (passed, reason)."""
    if cand.scTM < seed.scTM - scTM_eps:
        return False, f"scTM {cand.scTM:.3f} < {seed.scTM - scTM_eps:.3f}"
    if scRMSD_max is not None and cand.scRMSD is not None and cand.scRMSD > scRMSD_max:
        return False, f"scRMSD {cand.scRMSD:.2f} > {scRMSD_max}"
    if (active_site_RMSD_max is not None and cand.active_site_RMSD is not None
            and cand.active_site_RMSD > active_site_RMSD_max):
        return False, f"active_site_RMSD {cand.active_site_RMSD:.2f} > {active_site_RMSD_max}"
    return True, "ok"


def accept_refinement(*, seed_core_count: int, cand_core_count: int,
                      structure_passed: bool) -> bool:
    """Hard output-accept: strictly fewer distinct cores than the seed AND the candidate
    passed the structure gate. Structure thresholds live in ``structure_gate`` so adding
    scRMSD / active-site metrics never touches this predicate."""
    return cand_core_count < seed_core_count and structure_passed


@dataclass
class RefineResult:
    protein_id: str
    seed_seq: str
    seed_core_count: int
    best_seq: str
    best_core_count: int
    best_structure: "StructureMetrics"
    n_accepts: int
    diverged: bool
    shortlist: list = field(default_factory=list)    # dicts: seq, core_count, scTM, pLDDT, scRMSD, active_site_RMSD, muts
    trace: list = field(default_factory=list)


def refine_sequence(protein_id, seed_seq, *, propose_fn, head_fn, nmp_fn, struct_fn, anchors,
                    target_window_idx_fn=None, strong_rank=0.02, margin_band=0.10,
                    scTM_eps=0.05, scRMSD_max=None, active_site_RMSD_max=None,
                    topB=None, beam_width=8, max_rounds=20, patience=3,
                    max_path_mutations=8, allow_structure_unknown=False):
    import numpy as np

    def nmp_states(seqs):   # BATCHED: one nmp_fn call for the whole list (§2 confirmed)
        results = nmp_fn(protein_id, list(seqs))
        states = []
        for seq, rows in zip(seqs, results):
            cores = extract_target_cores(rows, anchors=anchors, strong_rank=strong_rank)
            margin = rank_margin_mass([c.best_rank for c in cores], strong_rank=strong_rank,
                                      margin_band=margin_band)
            states.append({"seq": seq, "cores": cores, "count": len(cores), "margin": margin})
        return states

    seed = nmp_states([seed_seq])[0]
    seed_metrics = struct_fn(protein_id, seed_seq)
    seed["structure"] = seed_metrics
    seed_count, seed_margin = seed["count"], seed["margin"]
    beam = [seed]
    best = {"seq": seed_seq, "count": seed_count, "structure": seed_metrics}
    shortlist, trace, best_key, stale = [], [], (seed_count, seed_margin), 0

    for rnd in range(max_rounds):
        if best["count"] == 0:
            break
        pool = [(st, c) for st in beam for c in propose_fn(st["seq"], st["cores"])]
        if not pool:
            break
        W = head_fn(protein_id, [c.seq for _, c in pool])
        proxy = np.empty(len(pool))
        for i, (st, _) in enumerate(pool):
            tw = target_window_idx_fn(st["cores"]) if target_window_idx_fn else None
            row = np.asarray(W[i])
            proxy[i] = row[tw].max() if tw else row.max()
        chosen = np.argsort(proxy) if topB is None else np.argsort(proxy)[:topB]

        chosen_pairs = [pool[int(k)] for k in chosen]
        chosen_states = nmp_states([c.seq for _, c in chosen_pairs])   # one batched NMP call/round
        beam_admits, n_nmp, n_refold = [], len(chosen_pairs), 0
        for (st, c), state in zip(chosen_pairs, chosen_states):
            n_mut = sum(a != b for a, b in zip(c.seq, seed_seq))
            improving = (state["count"] < st["count"]
                         or (state["count"] == st["count"] and state["margin"] < st["margin"]))
            if not improving or n_mut > max_path_mutations:
                continue                                        # no help / too many edits -> drop
            if state["count"] < seed_count:                     # potential OUTPUT -> refold + gate NOW
                metrics = struct_fn(protein_id, c.seq); n_refold += 1
                passed, reason = structure_gate(seed_metrics, metrics, scTM_eps=scTM_eps,
                                                scRMSD_max=scRMSD_max,
                                                active_site_RMSD_max=active_site_RMSD_max)
                if not (passed or allow_structure_unknown):
                    continue                                    # structure-failed output -> excluded from beam
                metrics = replace(metrics, passed=passed, reason=reason)
                state["structure"] = metrics
                beam_admits.append(state)
                shortlist.append({"seq": c.seq, "core_count": state["count"], "scTM": metrics.scTM,
                                  "pLDDT": metrics.pLDDT, "scRMSD": metrics.scRMSD,
                                  "active_site_RMSD": metrics.active_site_RMSD, "muts": c.desc})
                if state["count"] < best["count"]:
                    best = {"seq": c.seq, "count": state["count"], "structure": metrics}
            else:                                               # margin-progress only -> beam, NO refold
                beam_admits.append(state)
        beam = sorted(beam + beam_admits, key=lambda s: (s["count"], s["margin"]))[:beam_width]
        key = (beam[0]["count"], beam[0]["margin"])
        trace.append({"round": rnd, "n_candidates": len(pool), "n_nmp": n_nmp, "n_refold": n_refold,
                      "best_count": best["count"], "beam_best_count": beam[0]["count"],
                      "beam_best_margin": beam[0]["margin"]})
        if key < best_key:
            best_key, stale = key, 0
        else:
            stale += 1
            if stale >= patience:
                break

    return RefineResult(protein_id, seed_seq, seed_count, best["seq"], best["count"],
                        best["structure"], len(shortlist), best["count"] == seed_count,
                        shortlist, trace)
```

- [ ] **Step 4:** run the tests → PASS. **Step 5:** commit `feat(refine): margin + structure
  gate + accept rule + beam search loop (R3)`.

### Task R4 — driver `scripts/refine_rf_designs.py` (real oracles, ceiling + refine)  (TDD smoke)

**Files:** create `scripts/refine_rf_designs.py`; test `tests/scripts/test_refine_rf_designs.py`.

- [ ] **Step 1 — write failing smoke test:** build a 2-row `generated.parquet`, a fake
  `imm_nmp_peptides.parquet` (one killable core on row A), inject fake `head_fn/nmp_fn/struct_fn`
  via the module's `build_oracles` seam, call `run_refinement(args, oracles)`, and assert:
  (i) `refined/refined_designs.parquet` has `{protein_id, design_idx, sequence_original,
  sequence_refined, n_mutations, core_count_before, core_count_after, scTM_after, pLDDT_after,
  diverged}`; (ii) `refined/evaluator_ready.parquet` has EXACTLY the generated.parquet schema
  (§ below) with `sequence` = the refined sequence; (iii) row A reaches
  `core_count_after < core_count_before`.
- [ ] **Step 2:** run → FAIL. **Step 3 — implement** the driver:

  **(a) argparse** (echo all via `--print-config`, hyperparam-print rule): `--run-dir`
  (`generated.parquet`), `--eval-immune-dir` (`imm_head.parquet` for best-of-N selection by
  `global_risk`; `imm_nmp_peptides.parquet` to seed targets) OR `--seed-table` (a self-contained
  `protein_id/design_id/sequence` parquet — e.g. the `refine_seed_selection_0701/refine_seeds_0701.parquet`
  produced for E1 — which bypasses `--run-dir` selection and refines the listed seeds directly),
  `--constraint-manifest`,
  `--allele`, `--proteins` (comma list or `all`), `--seeds-per-protein`, `--mode
  {ceiling,refine}`, `--target-source {nmp,head}`; head args mirroring `run_if_phase_c1.py`
  (`--head-checkpoint --head-config-dir --head-variant-id --head-device --head-allele-idx
  --head-window-batch-size`); NMP args mirroring `evaluate_phase_c.py` (`--nmp-batch-size
  --nmp-max-lengths-per-call --nmp-workers --nmp-timeout`); structure (`--pdb-root
  --esmfold-cache-dir`); search knobs (§7): `--strong-rank --margin-band --scTM-eps
  --scRMSD-max --active-site-RMSD-max --topB --beam-width --max-rounds --patience --max-pairs
  --head-high-topk --allow-structure-unknown`; `--out-dir`.

  **(b) `build_oracles(args) -> (head_fn, nmp_fn, struct_fn, target_window_idx_fn)`** — the
  only place touching heavy deps (the smoke test bypasses it):
  - `head_fn`: build the predictor + `OnlineHeadScorer` **exactly as `run_if_phase_c1.py`
    does** (predictor ~lines 240–244; head CLI ~1287–1296; window k-range from `inference.yaml`
    ~1849–1862 — factor a shared helper). Return `scorer.score_window_risk_batch_same_protein(
    protein_id=pid, records=[(str(i), s) for i, s in enumerate(seqs)]).window_risks`; capture
    `window_starts_0b/window_ends_0b` to build `target_window_idx_fn(cores)` = indices of
    windows whose `[start, end)` cover any core span.
  - `nmp_fn`: `from epitope_head.data.netmhciipan_runner import <StandaloneRunner>` +
    `from inverse_folding.evaluation.immunogenicity import NMP_PEP_LENGTHS`; call
    `runner.score_batch([(pid, seq)], allele, NMP_PEP_LENGTHS)` and flatten to rows
    `{pos, pep_length, peptide, core, rank_EL:=el_rank}`.
  - `struct_fn`: `model = load_refold_model("esmfold", device=args.head_device)` once; per call
    `pred = refold(seq, pid, design_id, backend="esmfold", cache_dir=args.esmfold_cache_dir,
    model=model)`, `scTM, rmsd = run_tmalign(pred["pdb_path"], resolve_structure_path(<test_row
    for pid>, args.pdb_root))`; return `StructureMetrics(scTM=scTM, pLDDT=pred["pLDDT"],
    scRMSD=rmsd)`. (Confirm `resolve_structure_path`'s exact signature in
    `scripts/evaluate_phase_c.py` — it takes the test-set row, not a bare id.)

  **(c) `run_refinement(args, oracles)`**: load `generated.parquet` + `imm_head.parquet`; per
  protein pick top `--seeds-per-protein` designs by `global_risk`; anchors via
  `build_run_constraints(load_constraint_manifest(args.constraint_manifest), {pid: seq},
  aa_to_token=...)`; `head_high_positions` = top-`head-high-topk` per-residue-head positions in
  each core span; `propose_fn` = block enumeration (`hotspot_blocks` → per block
  `enumerate_singles`, head-rank the singles, `enumerate_pairs` on top (pos,AA), cap `--max-pairs`).
  - `--mode ceiling`: propose once per seed, score ALL with head + NMP, refold every
    count/margin-improving candidate; write `ceiling/candidates.parquet` (`protein_id,
    design_idx, muts, positions, head_proxy, core_count, rank_margin_mass, scTM, pLDDT,
    passed, eliminates`).
  - `--mode refine`: run `refine_sequence(...)` per seed; write **both** outputs:
    - `refined/refined_designs.parquet` (rich: `protein_id, design_idx, sequence_original,
      sequence_refined, n_mutations, muts, core_count_before, core_count_after, scTM_after,
      pLDDT_after, scRMSD_after, active_site_RMSD_after, diverged`) — one row per shortlist
      entry, `design_idx` a fresh contiguous int per protein.
    - `refined/evaluator_ready.parquet` — **exactly the `generated.parquet` schema** produced
      by `write_phase_c_outputs` (verify: `protein_id, design_idx:int, sequence, seed,
      wall_seconds`), with `sequence = sequence_refined` and the SAME `(protein_id, design_idx)`
      as the rich file (join key). This is the drop-in `--generated` input for E1 re-eval.
    - `refined/refine_trace.parquet` + `refined/refine_config.json`.

- [ ] **Step 4:** run smoke → PASS. **Step 5:** commit `feat(refine): driver with ceiling +
  refine modes, real oracles, evaluator-ready output (R4)`.

---

## 5. Experiments (agent runs directly after the coder finishes R1–R4)

> The commands below are the **payload**. Per CLAUDE.md §2, an experiment agent runs them as a
> Della SLURM job via the existing submit pattern (change params, not the raw `.py`); see §6.

### E0 — candidate-level ceiling + head↔NMP retrieval (de-risk gate; run BEFORE E1)

**Objective (decision rules fixed up front):** on 5 seeds, measure (i) the **effect ceiling**
— can block-enumerated single/pair moves reduce the distinct-core count, and by how much per
seed; (ii) **head retrieval** — of the candidates that reduce the count (hard) or improve
`rank_margin_mass` (soft), what fraction does the head rank into `topB ∈ {8,16,32}`.
Pre-registered read: **Recall@topB=16 for count-dropping candidates < 0.5** ⇒ keep NMP-all for
E1 and raise `topB`; **≥ 0.8** ⇒ the cheap head-pruned mode is usable. (Sizes cost; not GO/NO-GO.)

**Inputs (exact):** run dir
`Results/RF/Uricase/uricase_characterized23_a1res03_b1open_n16__20260630T041450Z/HLA-DRB1_07_01`
+ its `eval_immune/` + `manifest.json` (read head checkpoint/config/allele-idx from here);
`inverse_folding/reference_flow/configs/uricase_q00511_active_site_perprotein_v0.yaml`;
`--pdb-root` = the uricase reference-backbone dir used by the char23 `eval_structure` (from
that run's struct config); seeds = `Q00511` + the 4 next-lowest-`global_risk` proteins.

```
python scripts/refine_rf_designs.py --mode ceiling \
  --run-dir <run>/HLA-DRB1_07_01 --eval-immune-dir <run>/HLA-DRB1_07_01/eval_immune \
  --constraint-manifest inverse_folding/reference_flow/configs/uricase_q00511_active_site_perprotein_v0.yaml \
  --allele HLA-DRB1_07_01 --proteins Q00511,<p2>,<p3>,<p4>,<p5> --seeds-per-protein 1 \
  --head-checkpoint <from manifest> --head-config-dir <…> --head-variant-id <…> \
  --head-allele-idx <…> --head-device cuda --pdb-root <uricase backbones> --esmfold-cache-dir <cache> \
  --strong-rank 0.02 --margin-band 0.10 --max-pairs 200 \
  --out-dir Results/RF/Uricase/refine_ceiling_probe_0701 --print-config
```

- [ ] Run. - [ ] RAR (`tools/rar.py new/commit`): retrieval metrics (Recall@topB for hard+soft
  "good", TopB hit rate, NMP-calls-per-eliminated-core, head false-negative rate) + per-seed
  ceiling (min reachable count via single+pair moves). **Difficulty-axis test:** per residual
  core, record single-mutation-killability (does any enumerated single move push it ≥ 2%) vs its
  depth (`rank_EL`), register degeneracy (# overlapping strong frames in its block), and
  attackability (leverage pocket mutable) — to test whether depth predicts difficulty (prior:
  it does not / anti-correlates) and to pick the real admission difficulty axis. Objective record,
  no verdicts; tables to `data/`.

### E1 — full refinement, char23/0701, re-evaluation

**Objective:** per protein, a ranked shortlist of refined sequences with **lower distinct-core
count** than the seed at `scTM ≥ scTM₀ − ε`; report how many proteins reach **count 0**.
Success (figure-worthy): ≥ 1 protein to 0 and a positive median count reduction vs the seed;
Q00511 (3 cores) is the primary smoke target.

**Validation seeds & method bake-off:** use `--seed-table
Results/RF/Uricase/refine_seed_selection_0701/refine_seeds_0701.parquet` — 123 difficulty-
stratified, structurally-valid (`global_ok_active_site_ok`, anchors preserved) seeds over 22
char24 proteins incl. **Pegloticase** (the FDA uricase drug); slice `methodcmp_core` (58 seeds)
for the method comparison, `difficulty_tier` (T0–T4) for the inflection curve. Compare (goal 2,
effect-first): block singles+pairs vs +triples (`--max-points`) vs +MC supplement; Mode-1
(`--target-source nmp`) vs Mode-2 (`head`); run the ceiling (`topB=None`) on `methodcmp_core`
first, then extend the winner to the full belly.

- [ ] **Smoke — Q00511:** `--mode refine --proteins Q00511 --seeds-per-protein 1
  --topB <E0-informed or omit for NMP-all> --beam-width 8 --max-rounds 20 --patience 3
  --scTM-eps 0.05 --out-dir Results/RF/Uricase/refine_q00511_smoke_0701` (+ the head/pdb/nmp
  args from E0). Confirm `refined/refined_designs.parquet` + `refined/evaluator_ready.parquet`
  exist; `core_count_after ≤ before`; scTM floor respected; config echoed; inspect `refine_trace.parquet`.
- [ ] **Full — char23/0701:** as above with `--proteins all --seeds-per-protein 3
  --out-dir Results/RF/Uricase/refine_char23_0701`.
- [ ] **Re-evaluate** (drop-in, no bridging):
```
python scripts/evaluate_phase_c.py --generated <out>/refined/evaluator_ready.parquet \
  --modes imm,struct --allele HLA-DRB1_07_01 --pdb-root <…> [nmp/esmfold args] --out-dir <out>/reeval
```
- [ ] **RAR:** per protein, distinct-core count and `n_strong_binders` seed→refined; scTM
  seed→refined; count reaching 0; `diverged` flags. Objective record; tables to `data/`.

---

## 6. Run Requirements (no SLURM authored here)

- **Environment:** `immune-design` conda env (Python 3.12, PyTorch 2.5.1) on Della; source at
  `/home/zc1519/src/Immune-Design`. Pure-logic tests (R1–R3) also run in a plain numpy env
  (they load `refine.py` by file path, bypassing the torch-backed package `__init__`).
- **Hardware:** one GPU (immune head forward + ESMFold refold). NetMHCIIpan on CPU via the
  existing `netmhciipan_runner` (the char23 `nmp_local` run confirms a working NMP install).
- **Inputs (all via CLI, never hardcoded):** the char23/0701 gen+eval run dir, its
  `manifest.json` (head checkpoint/config/allele-idx), the active-site manifest yaml, and
  `--pdb-root` (uricase reference backbones used by the char23 `eval_structure`).
- **Cost:** NMP dominates. E0 (5 seeds × block singles ~≤5 cores × ~4 editable × 19 ≈ few
  hundred candidates each + pairs) ⇒ NMP on ~1–3k sequences, batched 8×4/subprocess
  (~600 s timeout each) ⇒ order **hours**. Effect-first v0 refolds every improving candidate,
  so ESMFold load is non-trivial too. E1 full is larger (23 × 3 seeds × rounds) ⇒ **multi-hour**.
- **Execution:** run E0/E1 as Della SLURM jobs via the existing submit pattern (per CLAUDE.md
  §2: use `doc/SCRIPTS.md` scripts + SLURM, change params, not raw `.py`); if the refinement
  driver needs its own launcher, add one following `scripts/submit_cnn_enhance.slurm` and
  register it. `--output/--error` → `logs/`. Results return via `mhc-if-local` to
  `Results/RF/Uricase/refine_*`; RARs to `Results/Analysis/` (no LOG.md entry for runs/returns).

---

## 7. Knobs And Defaults

| Knob | v0 default | Note |
|---|---|---|
| `strong_rank` | 0.02 | strong = `rank_EL < 2%`; "eliminated" = crosses ≥ 2%. |
| distinct-core identity | `core_start` | per RAR 0024; adjacent-register merge deferred. |
| `margin_band` | 0.10 | `rank_margin_mass` counts cores with `rank_EL < 10%`. |
| `scTM_eps` | 0.05 | floor vs seed scTM₀; recalibrate on the Q00511 smoke. |
| `active_site_RMSD_max` | `None` (advisory v0) | `active_site_RMSD` := `active_site_spatial_shell6A_ca_rmsd` — function-critical (global scTM diverges from it: 66/1532 designs are global-ok/active-site-shifted). Monomer-partial for uricase (tetramer-interface active site); scTM stays the reliable global floor; anchors already frozen (mismatch 0/1532). |
| `scRMSD_max` | `None` | global backbone-RMSD ceiling (more local-sensitive than scTM); optional. |
| `max_path_mutations` | 8 | cheap refold-free cap bounding branch-waste on stacked edits. |
| `refold_cap` | 16 / round | max count-dropping candidates refolded per round (top by count↓, then edits↓). **Essential:** single mutations kill cores readily (H2ETE7: 2003/2208 candidates dropped count) so unbounded refold explodes (80 min GPU); the cap bounds ESMFold AND is exactly the lean ranked shortlist wanted. |
| `allow_structure_unknown` | `False` | v0 refolds+gates ONLY count-dropping (output) candidates; margin-progress states enter the beam with no refold. |
| `topB` | **64** for refine (`None` only for a small ceiling probe) | head-prunes the pool before NMP. NMP-all (`None`) proved infeasible on multi-core seeds (H2ETE7 r1: pool 2208 → 60 min NMP → walltime kill). head proxy = target-window risk, so top-B surfaces count-droppers by construction; 32–64 is safe when droppers are abundant. |
| `beam_width` | 4 | working states carried between rounds; larger multiplies the pool (`beam × per-state enumeration`) — the second explosion factor. |
| `max_rounds` / `patience` | 20 / 3 | stop on count 0, budget, or stalled (count,margin). |
| `max_pairs` / `max_points` | 200 / 2 | singles+pairs primary; triples only for stuck blocks. |
| `head_high_topk` | 0 (v0) | extra in-span high-head positions for a core's editable set; **v0 deferral** — non-zero fail-fasts, editable = pockets − anchors (needs per-residue head, not wired). |
| `seeds_per_protein` | 3 | refine top-k best-of-N (different basins). |
| `target_source` | `nmp` | Mode 1; `head` = Mode 2 switch (deferred). |
| allele | DRB1*07:01 | single-allele objective; multi-allele (union of cores) deferred. |

---

## 8. Registration And Gates

- [ ] Register `scripts/refine_rf_designs.py` in `doc/SCRIPTS.md` under a new "RF Refinement
  (PLAN_RF_REFINE.md)" section: inputs (a gen+eval run dir), the injected oracles, the two
  modes, outputs (`ceiling/candidates.parquet`, `refined/refined_designs.parquet`,
  `refined/evaluator_ready.parquet`, `refine_trace.parquet`), and the `--target-source` switch.
- [ ] `--print-config` echoes every knob to stdout (hyperparam-print rule).
- [ ] LOG.md: one entry when R4 lands (new pipeline stage). E0/E1 runs + RAR returns get none.

---

## 9. Deferred Mechanism B (RF-sampler inpainting refinement)

The RF sampler already seeds an existing sequence and regenerates an arbitrary subset via
`fixed_tokens` (`sampler.sample(fixed_tokens={pos: token})`, `sampler.py:126-134`; no code
change). Mechanism B freezes `S = all positions except the target cores` to the seed residues
and lets the RF denoiser re-propose the cores. Deferred: the count→0 objective favours the
off-distribution substitutions Mechanism A's explicit search reaches but the on-manifold
denoiser is unlikely to sample. Revisit only if A stalls on whack-a-mole blocks needing
jointly-coherent multi-position edits.

---

## 10. Self-Review (spec coverage)

- Objective (count→0): R1 extraction, R3 accept-on-count + soft margin, E1 report. ✓
- Mode 1 (NMP target/gate, head guide): §1, §3, R3 (`nmp_fn`/`head_fn`), R4 wiring. ✓
- Soft beam vs hard accept: `rank_margin_mass` + beam order + `accept_refinement` (R3),
  `test_soft_beam_enables_two_step_elimination`. ✓
- Structure oracle is a `StructureMetrics` bundle; gate logic in `structure_gate`; accept takes
  only `structure_passed` (extensible to scRMSD/active-site without touching accept). ✓
- Beam admits only structure-valid states (refold every improving candidate first): R3 loop,
  `test_structure_floor_blocks_fold_breaking_elimination_and_beam`. ✓
- Effect ceiling before cost: `--mode ceiling` (R4), E0. ✓
- Block enumeration + whack-a-mole: R2 + `hotspot_blocks` (R1), driver `propose_fn` (R4). ✓
- Editable = pockets ∪ high-head − anchors: `editable_positions` (R1), `--head-high-topk` (R4). ✓
- Structure gate vs WT backbone, floor scTM₀: `structure_gate` (R3), `struct_fn` via
  `run_tmalign` vs `--pdb-root` (R4). ✓
- Evaluator-ready output pinned to the generated.parquet schema (drop-in re-eval): R4, E1. ✓
- Candidate-level head↔NMP retrieval: E0 with pre-registered decision rule. ✓
- Run requirements + SLURM-via-existing-pattern stated, none authored: §6. ✓
- Type consistency: `EpitopeCore`/`Candidate`/`StructureMetrics`/`RefineResult`/`HeadFn`/`NmpFn`/
  `StructFn` used identically across R1–R4. ✓

---

## 11. Unified Architecture (target; DEFERRED — gen and refine are separate today)

A starting point for the eventual single pipeline. NOT implemented in v0 (this PLAN owns only
the standalone refiner); recorded so the admission/routing is designed before the pieces merge.

```
generate best-of-N (RF, WT-firewalled)
  │
  ├─ post-gen admission  (ONE NMP call on the selected design):
  │     raw NMP → any strong binder?  NO  → count 0 → DONE (no refinement)   [count-0 needs no de-inflate]
  │                                   YES → de-inflate to distinct cores, then compute:
  │            admission metric = ( attackable de-inflated block count , register degeneracy )   [NOT depth-weighted]
  │            seed structure    = { scTM , active_site_spatial_shell6A RMSD }
  │
  ├─ seed re-selection (admission drives selection, not the reverse):
  │     among the N, prefer the design MINIMISING the admission metric.
  │     RAR 0024: head-selected ≠ NMP-best — re-selecting by NMP pulls designs into the band.
  │
  ├─ route:
  │     count 0                                        → DONE
  │     in tractable band & scTM headroom & attackable → REFINE  (this PLAN)
  │     out of band & design worse than WT             → WT-FALLBACK: seed = WT, refine WT
  │                                                       (firewall-sanctioned "rescue"; label provenance
  │                                                        distinctly; EMPIRICALLY DORMANT on uricase
  │                                                        char24/0701 = 0/24 designs worse than WT)
  │     out of band & design better than WT            → accept design as-is (partial) / more sampling
  │     seed structure too weak (scTM below floor+δ)   → reject / re-sample
  │
  └─ refine → NMP-gated shortlist → (later phase) blood-immunogenicity validation of Head-beats-NMP
```

Notes: (0) **Difficulty is register-degeneracy + attackability, NOT depth.** MHC-II binding is
anchor-dominated and nonlinear, so a deep (very-strong) binder often falls to a single anchor
mutation; rank-depth weakly ANTI-predicts removal difficulty. `rank_margin_mass` is only a
search-progress surrogate (beam gradient in multi-step cases), never the admission difficulty
metric. E0 measures single-mutation-killability vs (depth, degeneracy, attackability) to fix the
real difficulty axis. (1) admission uses **NMP** (one call/design, accurate) — Head is the
inner-loop ranker, not the admission oracle. (2) The bars are **per-allele** and read off the
first round's
(difficulty → outcome/cost) curve (E0/E1), never hardcoded — the immune bar sits where P(reach 0
within budget) collapses or NMP-calls-per-eliminated-core explodes. (3) ONE structure threshold
is reused at admission (seed headroom `scTM₀ − floor > δ`, δ ≈ expected edits × per-edit scTM
cost) and as the per-candidate output gate. (4) The structure headroom δ and the immune band are
themselves calibrated from round-1 telemetry.
