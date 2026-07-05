"""Pure logic for the RF refinement stage (PLAN_RF_REFINE.md).

Target extraction, hotspot-block grouping, and editable-position computation for
post-hoc epitope-elimination refinement. No torch / no I/O: the expensive oracles
(immune head, NetMHCIIpan, ESMFold refold) are injected into the search loop as
callables, so this module is unit-testable without them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

MHC2_POCKETS = (0, 3, 5, 8)  # P1, P4, P6, P9 offsets within the 9-mer MHC-II core
AA20 = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class EpitopeCore:
    core_start: int                       # 0-based core start in the sequence
    core_seq: str                         # the 9-mer binding core
    best_rank: float                      # min NMP rank_EL over covering strong windows
    pocket_positions: tuple[int, ...]     # absolute P1/P4/P6/P9 positions
    editable_positions: tuple[int, ...]   # positions this core may mutate (pockets minus anchors)


@dataclass(frozen=True)
class Candidate:
    desc: str                     # e.g. "D2R" or "D2R,G5W"
    seq: str                      # full mutated sequence (same length; no indels)
    positions: tuple[int, ...]    # edited positions, sorted ascending


def extract_target_cores(peptides: Iterable[dict], *, anchors,
                         strong_rank: float = 0.02) -> list[EpitopeCore]:
    """Aggregate NMP strong-binder windows into distinct 9-mer core targets.

    ``peptides``: rows with keys ``pos, peptide, core, rank_EL``. A window is a target
    iff ``rank_EL < strong_rank``. Cores are keyed by 0-based ``core_start`` (= ``pos``
    plus the offset of ``core`` within ``peptide``); the min-rank window wins per
    ``core_start``. Returned sorted by ``best_rank`` ascending (strongest first).
    """
    anchors = set(anchors)
    by_start: dict[int, dict] = {}
    for r in peptides:
        if float(r["rank_EL"]) >= strong_rank:
            continue
        offset = r["peptide"].find(r["core"])
        if offset < 0:
            continue
        cs = int(r["pos"]) + offset
        cur = by_start.get(cs)
        if cur is None or float(r["rank_EL"]) < cur["rank_EL"]:
            by_start[cs] = {"core": r["core"], "rank_EL": float(r["rank_EL"])}
    cores: list[EpitopeCore] = []
    for cs in sorted(by_start):
        pockets = tuple(cs + d for d in MHC2_POCKETS)
        cores.append(EpitopeCore(
            core_start=cs,
            core_seq=by_start[cs]["core"],
            best_rank=by_start[cs]["rank_EL"],
            pocket_positions=pockets,
            editable_positions=tuple(p for p in pockets if p not in anchors),
        ))
    return sorted(cores, key=lambda c: c.best_rank)


def editable_positions(core: EpitopeCore, *, anchors,
                       head_high_positions=frozenset()) -> tuple[int, ...]:
    """Positions of a core that may be mutated: its P1/P4/P6/P9 pockets plus any
    high-head residue inside the 9-mer span, minus frozen active-site anchors."""
    span = range(core.core_start, core.core_start + 9)
    positions = set(core.pocket_positions) | {p for p in head_high_positions if p in span}
    return tuple(sorted(positions - set(anchors)))


def hotspot_blocks(cores: Iterable[EpitopeCore], *, gap: int = 0) -> list[list[EpitopeCore]]:
    """Group cores into connected-component blocks: cores whose 9-mer spans overlap
    (or lie within ``gap`` positions) share a block. A block is the co-targeting search
    unit for overlapping epitopes (register-shift / whack-a-mole)."""
    ordered = sorted(cores, key=lambda c: c.core_start)
    blocks: list[list[EpitopeCore]] = []
    current: list[EpitopeCore] = []
    current_end: int | None = None
    for c in ordered:
        start, end = c.core_start, c.core_start + 9
        if current and current_end is not None and start <= current_end + gap:
            current.append(c)
            current_end = max(current_end, end)
        else:
            if current:
                blocks.append(current)
            current, current_end = [c], end
    if current:
        blocks.append(current)
    return blocks


def _apply(seq: str, muts) -> str:
    """Return ``seq`` with each ``(position, amino_acid)`` in ``muts`` substituted."""
    chars = list(seq)
    for pos, aa in muts:
        chars[pos] = aa
    return "".join(chars)


def enumerate_singles(seq: str, editable, *, alphabet: str = AA20) -> list[Candidate]:
    """Every single substitution over ``editable`` positions, skipping the wild-type AA
    at each position. This is the primary (exhaustive) proposer for a small block."""
    out: list[Candidate] = []
    for p in sorted({int(x) for x in editable}):
        wt = seq[p]
        for a in alphabet:
            if a != wt:
                out.append(Candidate(f"{wt}{p}{a}", _apply(seq, [(p, a)]), (p,)))
    return out


def enumerate_pairs(seq: str, single_choices, *, max_pairs: int | None = None) -> list[Candidate]:
    """Double substitutions drawn from ranked single ``(position, amino_acid)`` choices
    (best first). Same-position pairs are skipped and duplicate unordered pairs deduped;
    ``max_pairs`` caps the count. Feed the top head-ranked singles here to build the
    pair layer of block-structured enumeration."""
    out: list[Candidate] = []
    seen: set = set()
    n = len(single_choices)
    for i in range(n):
        pi, ai = single_choices[i]
        for j in range(i + 1, n):
            pj, aj = single_choices[j]
            if pi == pj:
                continue
            key = tuple(sorted([(pi, ai), (pj, aj)]))
            if key in seen:
                continue
            seen.add(key)
            muts = sorted([(pi, ai), (pj, aj)])
            out.append(Candidate(",".join(f"{seq[p]}{p}{a}" for p, a in muts),
                                 _apply(seq, muts), tuple(p for p, _ in muts)))
            if max_pairs is not None and len(out) >= max_pairs:
                return out
    return out


# --------------------------------------------------------------------------- #
# Task R3 — margin surrogate, structure gate, accept rule, beam search loop
# --------------------------------------------------------------------------- #


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
                    max_path_mutations=8, refold_cap=16, allow_structure_unknown=False):
    """Beam search that eliminates NMP epitope cores. ``nmp_fn`` is BATCHED:
    ``(protein_id, list[seq]) -> list[list[dict]]`` (one NMP row-list per input
    sequence). Each round scores its whole ``chosen`` candidate set in ONE ``nmp_fn``
    call so NetMHCIIpan model-load is amortized across the batch (PLAN_RF_REFINE NMP
    acceleration decision). ``head_fn`` is likewise batched ``(pid, seqs) -> [K, W]``.

    Structure is gated on OUTPUTS, not every step (PLAN §1): a candidate that drops the
    distinct-core count below the seed is a potential output → refold + ``structure_gate``
    now (pass → shortlist + beam, fail → excluded from both); a margin-progress state
    (count not below the seed) enters the beam with NO refold. ``max_path_mutations``
    is a cheap refold-free cap bounding branch-waste on stacked edits."""
    import numpy as np

    def nmp_states(seqs):
        seqs = list(seqs)
        results = nmp_fn(protein_id, seqs)   # BATCHED: one score_batch call for the list
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
        droppers = []   # count-dropping candidates (potential outputs) -> refold-capped below
        for (st, c), state in zip(chosen_pairs, chosen_states):
            improving = (state["count"] < st["count"]
                         or (state["count"] == st["count"] and state["margin"] < st["margin"]))
            n_mut = sum(a != b for a, b in zip(c.seq, seed_seq))
            if not improving or n_mut > max_path_mutations:
                continue                                        # no help / too many edits -> drop
            if state["count"] < seed_count:
                droppers.append((state, c, n_mut))              # potential OUTPUT -> refold below
            else:
                beam_admits.append(state)                       # margin-progress -> beam, NO refold

        # Refold only the top-`refold_cap` count-droppers (biggest drop, then fewest edits).
        # Single mutations kill cores readily, so most candidates drop the count -> unbounded
        # refold explodes (2000+/round on multi-core seeds); the cap bounds ESMFold cost AND
        # yields the lean ranked shortlist the experiment wants (PLAN_RF_REFINE cost fix).
        droppers.sort(key=lambda x: (x[0]["count"], x[2]))
        if refold_cap is not None:
            droppers = droppers[:refold_cap]
        for state, c, _ in droppers:
            metrics = struct_fn(protein_id, c.seq); n_refold += 1
            passed, reason = structure_gate(seed_metrics, metrics, scTM_eps=scTM_eps,
                                            scRMSD_max=scRMSD_max,
                                            active_site_RMSD_max=active_site_RMSD_max)
            metrics = replace(metrics, passed=passed, reason=reason)
            if not (passed or allow_structure_unknown):
                continue                                        # structure-failed output -> no beam
            state["structure"] = metrics
            beam_admits.append(state)
            shortlist.append({"seq": c.seq, "core_count": state["count"], "scTM": metrics.scTM,
                              "pLDDT": metrics.pLDDT, "scRMSD": metrics.scRMSD,
                              "active_site_RMSD": metrics.active_site_RMSD, "muts": c.desc})
            if state["count"] < best["count"]:
                best = {"seq": c.seq, "count": state["count"], "structure": metrics}
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
