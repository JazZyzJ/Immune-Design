"""Pure logic for the RF refinement stage (PLAN_RF_REFINE.md).

Target extraction, hotspot-block grouping, and editable-position computation for
post-hoc epitope-elimination refinement. No torch / no I/O: the expensive oracles
(immune head, NetMHCIIpan, ESMFold refold) are injected into the search loop as
callables, so this module is unit-testable without them.
"""
from __future__ import annotations

import math
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
    active_site_RMSD: float | None = None  # legacy C-alpha active-site compatibility field
    global_ca_RMSD: float | None = None
    active_site_sidechain_RMSD: float | None = None
    max_anchor_sidechain_RMSD: float | None = None
    max_anchor_atom_distance: float | None = None
    active_site_complete: bool | None = None
    active_site_min_pLDDT: float | None = None
    cat_max_scRMSD: float | None = None
    predicted_active_site_min_pLDDT: float | None = None
    cache_hit: bool | None = None
    model_executed: bool | None = None
    passed: bool = True                    # set by structure_gate
    reason: str = ""


def rank_margin_mass(core_ranks, *, strong_rank: float = 0.02, margin_band: float = 0.10) -> float:
    """Continuous NMP surrogate to MINIMISE: Σ over cores with rank_EL < margin_band of
    max(strong_rank − rank_EL, 0). Deep strong cores contribute most; a core pushed to
    ≥ strong_rank contributes 0 (eliminated from the strong count)."""
    return float(sum(max(strong_rank - r, 0.0) for r in core_ranks if r < margin_band))


def structure_gate(seed: "StructureMetrics", cand: "StructureMetrics", *,
                   scTM_eps: float | None,
                   scTM_min: float | None = None,
                   cat_max_scRMSD_max: float | None = None,
                   predicted_active_site_min_pLDDT_min: float | None = None,
                   scRMSD_max: float | None = None,
                   active_site_RMSD_max: float | None = None,
                   max_anchor_sidechain_RMSD_max: float | None = None) -> tuple[bool, str]:
    """Verdict for a candidate fold against configured structure criteria.

    The protocol gate uses an absolute scTM floor, direct-functional catalytic side-chain
    maximum, and worst active-site confidence. Legacy seed-relative/global ceilings remain
    available for explicit old callers. Every configured metric is fail-closed.
    """
    if scTM_min is not None:
        if not math.isfinite(float(cand.scTM)):
            return False, f"scTM unavailable/not finite: {cand.scTM}"
        if cand.scTM < scTM_min:
            return False, f"scTM {cand.scTM:.3f} < {scTM_min:.3f}"
    if scTM_eps is not None:
        if not math.isfinite(float(cand.scTM)) or not math.isfinite(float(seed.scTM)):
            return False, f"scTM unavailable/not finite: seed={seed.scTM}, cand={cand.scTM}"
        if cand.scTM < seed.scTM - scTM_eps:
            return False, f"scTM {cand.scTM:.3f} < {seed.scTM - scTM_eps:.3f}"
    if cat_max_scRMSD_max is not None:
        value = cand.cat_max_scRMSD
        if value is None or not math.isfinite(float(value)):
            return False, f"cat_max_scRMSD unavailable/not finite: {value}"
        if value > cat_max_scRMSD_max:
            return False, f"cat_max_scRMSD {value:.2f} > {cat_max_scRMSD_max}"
    if predicted_active_site_min_pLDDT_min is not None:
        value = cand.predicted_active_site_min_pLDDT
        if value is None or not math.isfinite(float(value)):
            return False, (
                "predicted_active_site_min_pLDDT unavailable/not finite: "
                f"{value}"
            )
        if value < predicted_active_site_min_pLDDT_min:
            return False, (
                f"predicted_active_site_min_pLDDT {value:.2f} < "
                f"{predicted_active_site_min_pLDDT_min}"
            )
    if scRMSD_max is not None:
        if cand.scRMSD is None or not math.isfinite(float(cand.scRMSD)):
            return False, f"scRMSD unavailable/not finite: {cand.scRMSD}"
        if cand.scRMSD > scRMSD_max:
            return False, f"scRMSD {cand.scRMSD:.2f} > {scRMSD_max}"
    if active_site_RMSD_max is not None:
        if cand.active_site_RMSD is None or not math.isfinite(float(cand.active_site_RMSD)):
            return False, f"active_site_RMSD unavailable/not finite: {cand.active_site_RMSD}"
        if cand.active_site_RMSD > active_site_RMSD_max:
            return False, f"active_site_RMSD {cand.active_site_RMSD:.2f} > {active_site_RMSD_max}"
    if max_anchor_sidechain_RMSD_max is not None:
        if cand.active_site_complete is not True:
            return False, (
                "max_anchor_sidechain_RMSD unavailable: side-chain active-site metrics "
                "incomplete"
            )
        value = cand.max_anchor_sidechain_RMSD
        if value is None or not math.isfinite(float(value)):
            return False, f"max_anchor_sidechain_RMSD unavailable/not finite: {value}"
        if value > max_anchor_sidechain_RMSD_max:
            return False, (
                f"max_anchor_sidechain_RMSD {value:.2f} > "
                f"{max_anchor_sidechain_RMSD_max}"
            )
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
    shortlist: list = field(default_factory=list)    # per-output immune + selected structure metrics
    trace: list = field(default_factory=list)


def _hamming_positions(a: str, b: str) -> list[int]:
    """0-based positions where two equal-length strings differ (the point-edit set)."""
    return [i for i in range(min(len(a), len(b))) if a[i] != b[i]]


def splice_nmp_rows(seed_rows, sub_rows, edited_positions, offset, *, context=3):
    """Merge a candidate's sub-sequence NMP rows over the cached seed rows.

    A window (peptide) at protein position ``pos`` of length ``pep_length`` COVERS an edit
    iff ``[pos-context, pos+pep_length+context)`` intersects ``edited_positions`` — the
    ``context`` flanks matter because NetMHCIIpan runs with ``-context`` (±3 residues), so an
    edit just OUTSIDE a peptide still changes its %Rank_EL. Covered windows are taken from
    ``sub_rows`` (``pos`` remapped by ``+offset``); every other window is copied verbatim from
    ``seed_rows`` — byte-identical to a full re-score because neither the peptide nor its
    context touches the mutation. EXACT iff each covered window keeps its full peptide+context
    inside the sub-sequence; the caller guarantees this with margin ≥ max_len + context.
    """
    edited = set(int(p) for p in edited_positions)

    def covers(pos, plen):
        return any(pos - context <= p < pos + plen + context for p in edited)

    sub_covered = {}
    for r in sub_rows:
        pos, plen = int(r["pos"]) + offset, int(r["pep_length"])
        if covers(pos, plen):
            rr = dict(r)
            rr["pos"] = pos
            sub_covered[(pos, plen)] = rr
    merged = []
    for r in seed_rows:
        pos, plen = int(r["pos"]), int(r["pep_length"])
        if covers(pos, plen):
            repl = sub_covered.get((pos, plen))
            if repl is not None:
                merged.append(repl)
            # else: a covered window absent from the sub-seq means the margin was too small;
            # the caller's span-budget fallback prevents this, so drop defensively.
        else:
            merged.append(dict(r))
    return merged


def _incremental_nmp_rows(protein_id, seed_seq, seed_rows, cand_seqs, nmp_fn, *, margin=30,
                          context=3):
    """Score candidate sequences against a cached seed via the sub-sequence splice.

    Per candidate: diff vs ``seed_seq`` → edited positions; unchanged → reuse ``seed_rows``;
    edit span ≤ ``2*margin+1`` → score only ``seq[minP-margin : maxP+margin+1]`` and splice;
    else fall back to a full-length re-score. ``margin`` must be ≥ ``max_pep_len + context``
    (30 ≥ 25 + 3) so every window whose peptide OR ``-context`` flank touches an edit keeps its
    full footprint inside the sub-sequence. All sub-sequences (+ fallbacks) go through ONE
    batched ``nmp_fn`` call so the NetMHCIIpan model-load is amortized. Returns one row-list
    per candidate, in order.
    """
    n = len(seed_seq)
    budget = 2 * margin + 1
    plans, batch = [], []
    for s in cand_seqs:
        pos = _hamming_positions(s, seed_seq)
        if not pos:
            plans.append(("hit", None, None))
        elif pos[-1] - pos[0] + 1 > budget:
            plans.append(("full", len(batch), None))
            batch.append(s)
        else:
            offset = max(0, pos[0] - margin)
            plans.append(("splice", len(batch), (offset, pos)))
            batch.append(s[offset:min(n, pos[-1] + margin + 1)])
    scored = nmp_fn(protein_id, batch) if batch else []
    out = []
    for kind, idx, extra in plans:
        if kind == "hit":
            out.append(seed_rows)
        elif kind == "full":
            out.append(scored[idx])
        else:
            offset, pos = extra
            out.append(splice_nmp_rows(seed_rows, scored[idx], pos, offset, context=context))
    return out


def refine_sequence(protein_id, seed_seq, *, propose_fn, head_fn, nmp_fn, struct_fn, anchors,
                    target_window_idx_fn=None, strong_rank=0.02, margin_band=0.10,
                    scTM_eps=0.05, scRMSD_max=None, active_site_RMSD_max=None,
                    scTM_min=None, cat_max_scRMSD_max=None,
                    predicted_active_site_min_pLDDT_min=None,
                    max_anchor_sidechain_RMSD_max=None,
                    topB=None, beam_width=8, max_rounds=20, patience=3,
                    max_path_mutations=8, refold_cap=16, allow_structure_unknown=False,
                    incremental_nmp=False, nmp_context_margin=30, log_fn=None):
    """Beam search that eliminates NMP epitope cores. ``nmp_fn`` is BATCHED:
    ``(protein_id, list[seq]) -> list[list[dict]]`` (one NMP row-list per input sequence);
    each round scores its whole candidate set in ONE ``nmp_fn`` call and duplicate sequences
    are scored once. ``head_fn`` (``(pid, seqs) -> [K, W]``) is only invoked when ``topB`` is
    finite — NMP-all (``topB=None``) discards the ranking, so the head pass is skipped.

    ``incremental_nmp`` (exact for per-window NMP): score only each candidate's mutated
    sub-sequence and splice it over the cached seed rows (a ~10x NMP-volume cut with identical
    cores/count/margin — see ``splice_nmp_rows``). Off by default; the driver enables it since
    real NetMHCIIpan scores each window against a fixed allele background.

    Structure is gated on OUTPUTS, not every step (PLAN §1): a candidate that drops the
    distinct-core count below the seed is a potential output → refold + ``structure_gate`` now
    (pass → shortlist + beam, fail → excluded from both); a margin-progress state (count not
    below the seed) enters the beam with NO refold. ``max_path_mutations`` caps stacked edits.
    ``log_fn`` (optional) receives one flushed line per round for progress visibility."""
    import time

    import numpy as np

    def _rows_to_state(seq, rows):
        cores = extract_target_cores(rows, anchors=anchors, strong_rank=strong_rank)
        margin = rank_margin_mass([c.best_rank for c in cores], strong_rank=strong_rank,
                                  margin_band=margin_band)
        return {"seq": seq, "cores": cores, "count": len(cores), "margin": margin}

    def score_states(cand_seqs):
        """NMP → per-seq state, DEDUPED (identical seqs scored once); uses the incremental
        splice vs the cached seed when enabled."""
        uniq = list(dict.fromkeys(cand_seqs))
        if not uniq:
            return {}
        rows_list = (_incremental_nmp_rows(protein_id, seed_seq, seed_rows, uniq, nmp_fn,
                                           margin=nmp_context_margin)
                     if incremental_nmp else nmp_fn(protein_id, uniq))
        return {s: _rows_to_state(s, r) for s, r in zip(uniq, rows_list)}

    seed_rows = nmp_fn(protein_id, [seed_seq])[0]          # full seed baseline for the splice
    seed = _rows_to_state(seed_seq, seed_rows)
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

        t0 = time.time()
        if topB is None:
            chosen_pairs = pool                            # NMP-all: the head ranking is discarded
        else:
            uniq_seqs = list(dict.fromkeys(c.seq for _, c in pool))
            head_rows = dict(zip(uniq_seqs, head_fn(protein_id, uniq_seqs)))
            tw_by_state = {}                               # hoist target windows: invariant per state
            proxy = np.empty(len(pool))
            for i, (st, c) in enumerate(pool):
                if id(st) not in tw_by_state:
                    tw_by_state[id(st)] = (target_window_idx_fn(st["cores"])
                                           if target_window_idx_fn else None)
                tw = tw_by_state[id(st)]
                row = np.asarray(head_rows[c.seq])
                proxy[i] = row[tw].max() if tw else row.max()
            chosen_pairs = [pool[int(k)] for k in np.argsort(proxy)[:topB]]
        t_head = time.time() - t0

        t0 = time.time()
        states_by_seq = score_states([c.seq for _, c in chosen_pairs])
        t_nmp = time.time() - t0

        # Split chosen into count-droppers (potential outputs -> refold, capped) and
        # margin-progress states (enter the beam with no refold).
        beam_admits, droppers = [], []
        for st, c in chosen_pairs:
            state = dict(states_by_seq[c.seq])             # copy: per-parent 'structure' must not alias
            improving = (state["count"] < st["count"]
                         or (state["count"] == st["count"] and state["margin"] < st["margin"]))
            n_mut = sum(a != b for a, b in zip(c.seq, seed_seq))
            if not improving or n_mut > max_path_mutations:
                continue                                    # no help / too many edits -> drop
            if state["count"] < seed_count:
                droppers.append((state, c, n_mut))          # potential OUTPUT -> refold below (capped)
            else:
                beam_admits.append(state)                   # margin-progress only -> beam, NO refold

        # Refold only the top-`refold_cap` count-droppers (biggest drop, then fewest edits).
        # Single mutations kill cores readily, so most candidates drop the count -> unbounded
        # refold explodes (2000+/round on multi-core seeds); the cap bounds ESMFold cost AND
        # yields the lean ranked shortlist the experiment wants (PLAN_RF_REFINE §7 cost fix).
        droppers.sort(key=lambda x: (x[0]["count"], x[2]))
        if refold_cap is not None:
            droppers = droppers[:refold_cap]
        n_refold, t_refold = 0, 0.0
        for state, c, _ in droppers:
            tr = time.time()
            metrics = struct_fn(protein_id, c.seq); n_refold += 1
            passed, reason = structure_gate(seed_metrics, metrics, scTM_eps=scTM_eps,
                                            scTM_min=scTM_min,
                                            cat_max_scRMSD_max=cat_max_scRMSD_max,
                                            predicted_active_site_min_pLDDT_min=(
                                                predicted_active_site_min_pLDDT_min
                                            ),
                                            scRMSD_max=scRMSD_max,
                                            active_site_RMSD_max=active_site_RMSD_max,
                                            max_anchor_sidechain_RMSD_max=(
                                                max_anchor_sidechain_RMSD_max
                                            ))
            t_refold += time.time() - tr
            metrics = replace(metrics, passed=passed, reason=reason)
            if not (passed or allow_structure_unknown):
                continue                                    # structure-failed output -> no beam
            state["structure"] = metrics
            beam_admits.append(state)
            shortlist.append({"seq": c.seq, "core_count": state["count"], "scTM": metrics.scTM,
                              "pLDDT": metrics.pLDDT, "scRMSD": metrics.scRMSD,
                              "active_site_RMSD": metrics.active_site_RMSD,
                              "global_ca_RMSD": metrics.global_ca_RMSD,
                              "active_site_sidechain_RMSD": metrics.active_site_sidechain_RMSD,
                              "max_anchor_sidechain_RMSD": metrics.max_anchor_sidechain_RMSD,
                              "max_anchor_atom_distance": metrics.max_anchor_atom_distance,
                              "active_site_complete": metrics.active_site_complete,
                              "active_site_min_pLDDT": metrics.active_site_min_pLDDT,
                              "cat_max_scRMSD": metrics.cat_max_scRMSD,
                              "predicted_active_site_min_pLDDT": (
                                  metrics.predicted_active_site_min_pLDDT
                              ),
                              "muts": c.desc})
            if state["count"] < best["count"]:
                best = {"seq": c.seq, "count": state["count"], "structure": metrics}
        beam = sorted(beam + beam_admits, key=lambda s: (s["count"], s["margin"]))[:beam_width]
        key = (beam[0]["count"], beam[0]["margin"])
        trace.append({"round": rnd, "n_candidates": len(pool), "n_nmp": len(states_by_seq),
                      "n_refold": n_refold, "best_count": best["count"],
                      "beam_best_count": beam[0]["count"], "beam_best_margin": beam[0]["margin"],
                      "t_head": round(t_head, 2), "t_nmp": round(t_nmp, 2),
                      "t_refold": round(t_refold, 2)})
        if log_fn is not None:
            log_fn(f"[refine] {protein_id} r{rnd}: pool={len(pool)} nmp={len(states_by_seq)} "
                   f"refold={n_refold} best={best['count']} beam={beam[0]['count']}/"
                   f"{beam[0]['margin']:.3f} t_nmp={t_nmp:.1f}s t_head={t_head:.1f}s "
                   f"t_refold={t_refold:.1f}s")
        if key < best_key:
            best_key, stale = key, 0
        else:
            stale += 1
            if stale >= patience:
                break

    return RefineResult(protein_id, seed_seq, seed_count, best["seq"], best["count"],
                        best["structure"], len(shortlist), best["count"] == seed_count,
                        shortlist, trace)
