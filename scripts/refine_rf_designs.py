#!/usr/bin/env python
"""RF refinement driver — targeted NMP epitope-core elimination (PLAN_RF_REFINE.md R4).

A post-hoc stage between generation (`run_if_phase_c1.py`) and evaluation
(`evaluate_phase_c.py`). Loads a best-of-N RF design run, refines seeds by mutating
residual high-risk positions to drive the distinct NMP strong-core count toward 0
while preserving fold and freezing active-site anchors.

Pure search logic lives in `inverse_folding/reference_flow/refine.py`; the three
expensive oracles (immune head, NetMHCIIpan, ESMFold refold) are built in
`build_oracles` and injected into `run_refinement`, so the smoke test can bypass the
heavy deps by passing fake oracles.

Confirmed wiring (PLAN_RF_REFINE §"NMP is BATCHED" + R4 corrections):
- NMP is batched: nmp_fn(pid, list[seq]) -> list[list[dict]] with rank_EL in FRACTION
  units (raw runner el_rank, NOT the ×100 percent of the eval parquet).
- run_tmalign returns {"tm_score","rmsd"}; resolve_structure_path takes a test-set row
  (=> --test-set-parquet); refold pdb_path needs --esmfold-cache-dir.
- Inputs are 8-shard, unmerged => glob generation/*shard*/ + eval_immune/*shard*/.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.constraints import load_constraint_manifest
from inverse_folding.reference_flow.refine import (
    editable_positions,
    enumerate_pairs,
    enumerate_singles,
    extract_target_cores,
    hotspot_blocks,
    rank_margin_mass,
    refine_sequence,
    structure_gate,
)

# Injected oracle bundle; the smoke test builds fakes with this exact shape.
# window_coords_fn(seq) -> (window_starts_0b, window_ends_0b) for THIS sequence's length
# (the head window template is length-dependent — a fixed template mis-maps long proteins).
Oracles = namedtuple("Oracles", ["head_fn", "nmp_fn", "struct_fn", "window_coords_fn"])

_REFINEMENT_STRUCTURE_METRICS = frozenset(
    {"global_ca_rmsd", "plddt", "sidechain"}
)


def _parse_refinement_structure_metrics(value: str | None) -> frozenset[str]:
    requested = frozenset(
        item.strip().lower() for item in str(value or "").split(",") if item.strip()
    )
    unknown = requested - _REFINEMENT_STRUCTURE_METRICS
    if unknown:
        raise ValueError(
            "unknown --refinement-structure-metrics values: "
            f"{sorted(unknown)}; expected {sorted(_REFINEMENT_STRUCTURE_METRICS)}"
        )
    return requested


def _make_target_window_idx_fn(win_starts, win_ends):
    """Build the cores->covering-window-indices map for a specific sequence length.

    Rebuilt per seed from that seed's actual window template (all point-mutation
    candidates share the seed's length, so the template is stable within a seed but
    NOT across proteins of different lengths)."""
    ws = np.asarray(win_starts)
    we = np.asarray(win_ends)

    def target_window_idx_fn(cores):
        return [
            w for w in range(len(ws))
            if any(c.core_start < we[w] and c.core_start + 9 > ws[w] for c in cores)
        ]

    return target_window_idx_fn

# generated.parquet schema (write_phase_c_outputs) — evaluator_ready must match exactly.
GENERATED_COLUMNS = ["protein_id", "design_idx", "sequence", "seed", "wall_seconds"]


# --------------------------------------------------------------------------- #
# Sharded input loading (glob + concat; falls back to a flat file)
# --------------------------------------------------------------------------- #
def _load_sharded(root: Path, subdir: str, filename: str) -> pd.DataFrame:
    """Load a clean refinement input flexibly (the caller hands whatever is tidiest).

    Accepts, in priority order:
      1. ``root`` is a parquet FILE            -> read it directly (a merged/clean parquet);
      2. ``root/<subdir>/*shard*/filename``    -> return-package layout;
      3. ``root/*shard*/filename``             -> shard dirs directly under root (no wrapper);
      4. flat ``root/<subdir>/filename`` or ``root/filename``.
    Shards are concatenated. The ``generation``/``eval_immune`` wrapper is therefore
    optional, so a pre-organized input works without staging a return-package tree.
    """
    root = Path(root)
    if root.is_file():
        return pd.read_parquet(root)
    paths: list[str] = []
    for pattern in (root / subdir / "*shard*" / filename, root / "*shard*" / filename):
        paths = sorted(glob.glob(str(pattern)))
        if paths:
            break
    if not paths:
        for flat in (root / subdir / filename, root / filename):
            if flat.exists():
                paths = [str(flat)]
                break
    if not paths:
        raise FileNotFoundError(
            f"no {filename} found under {root} (tried '{subdir}/*shard*/', '*shard*/', "
            "a flat file, or a direct parquet path)"
        )
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _canonical_allele(allele: str) -> str:
    """Normalize an allele to the canonical NetMHCIIpan form ``HLA-DRB1*07:01``.

    ``StandaloneRunner.score_batch`` normalizes ``*``->``_`` and drops ``:``; a run-dir
    filesystem tag like ``HLA-DRB1_07_01`` would mis-normalize to an INVALID
    ``DRB1_07_01``. The ``*``/``:`` form maps to a valid ``DRB1_0701``. Already-canonical
    input (contains ``*`` or ``:``) passes through unchanged.
    """
    a = allele.strip()
    if "*" in a or ":" in a:
        return a
    parts = a.split("_")  # 'HLA-DRB1_07_01' -> ['HLA-DRB1', '07', '01'] (gene, field1, field2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return f"{parts[0]}*{parts[1]}:{parts[2]}"
    return a


# --------------------------------------------------------------------------- #
# Proposer: block-structured enumeration (singles exhaustive, head-ranked pairs)
# --------------------------------------------------------------------------- #
def make_propose_fn(oracles: Oracles, protein_id: str, anchors: set, *, max_pairs: int,
                    target_window_idx_fn):
    """Block-structured proposer (PLAN §1). Per hotspot block: exhaustive singles over
    the block's editable pocket positions (minus anchors), then head-ranked pairs among
    the singles (best head-proxy first), capped by ``max_pairs``. head_high editable
    augmentation is a v0 deferral (pockets only). ``target_window_idx_fn`` is the
    per-seed (length-correct) core->window map."""

    def propose(seq: str, cores):
        if not cores:
            return []
        candidates = []
        for block in hotspot_blocks(cores):
            editable = sorted(
                {p for c in block for p in editable_positions(c, anchors=anchors)}
            )
            if not editable:
                continue
            singles = enumerate_singles(seq, editable)
            candidates.extend(singles)
            if max_pairs > 0 and len(singles) > 1:
                risks = oracles.head_fn(protein_id, [s.seq for s in singles])
                tw = target_window_idx_fn(block) if target_window_idx_fn is not None else None
                proxy = np.array(
                    [
                        (np.asarray(row)[tw].max() if tw else np.asarray(row).max())
                        for row in risks
                    ]
                )
                ranked = [
                    (singles[i].positions[0], singles[i].seq[singles[i].positions[0]])
                    for i in np.argsort(proxy)
                ]
                candidates.extend(enumerate_pairs(seq, ranked, max_pairs=max_pairs))
        return candidates

    return propose


# --------------------------------------------------------------------------- #
# Seed selection + anchor resolution
# --------------------------------------------------------------------------- #
def _select_seeds(generated: pd.DataFrame, imm_head: pd.DataFrame, protein_id: str,
                  n_seeds: int) -> list[dict]:
    """Top-``n_seeds`` designs for a protein by ascending global_risk (best de-immunized).

    When ``imm_head is None`` (a clean seq-only input with no global_risk), fall back to
    the provided designs in stable ``design_idx`` order — refine what the caller handed."""
    g = generated[generated["protein_id"].astype(str) == protein_id]
    if g.empty:
        return []
    if imm_head is None:
        return g.sort_values("design_idx").head(n_seeds).to_dict("records")
    risk = imm_head[imm_head["protein_id"].astype(str) == protein_id][
        ["design_idx", "global_risk"]
    ]
    merged = g.merge(risk, on="design_idx", how="left").sort_values(
        "global_risk", na_position="last"
    )
    return merged.head(n_seeds).to_dict("records")


def _anchors_for(manifest, protein_id: str, sequence: str) -> set:
    """Hard-anchor positions for a protein (fail-fast identity validation), or empty."""
    if manifest is None or not manifest.has_protein(protein_id):
        return set()
    constraint = manifest.constraint_for_protein(protein_id)
    constraint.validate_against_sequence(sequence)  # fail-fast AA-identity assert
    return set(constraint.hard_anchor_indices)


# --------------------------------------------------------------------------- #
# Ceiling mode: score every proposed candidate for one round (E0 dataset)
# --------------------------------------------------------------------------- #
def _run_ceiling_seed(oracles, protein_id, seed_seq, design_idx, anchors, args) -> list[dict]:
    tw_fn = _make_target_window_idx_fn(*oracles.window_coords_fn(seed_seq))
    seed_rows = oracles.nmp_fn(protein_id, [seed_seq])[0]
    seed_cores = extract_target_cores(seed_rows, anchors=anchors, strong_rank=args.strong_rank)
    seed_count = len(seed_cores)
    seed_margin = rank_margin_mass([x.best_rank for x in seed_cores],
                                   strong_rank=args.strong_rank, margin_band=args.margin_band)
    seed_metrics = oracles.struct_fn(protein_id, seed_seq)
    propose_fn = make_propose_fn(oracles, protein_id, anchors, max_pairs=args.max_pairs,
                                 target_window_idx_fn=tw_fn)
    pool = propose_fn(seed_seq, seed_cores)
    if not pool:
        return []
    risks = oracles.head_fn(protein_id, [c.seq for c in pool])
    tw = tw_fn(seed_cores)
    head_proxy = [
        float(np.asarray(row)[tw].max() if tw else np.asarray(row).max()) for row in risks
    ]
    cand_rows = oracles.nmp_fn(protein_id, [c.seq for c in pool])
    out = []
    for c, hp, rows in zip(pool, head_proxy, cand_rows):
        cores = extract_target_cores(rows, anchors=anchors, strong_rank=args.strong_rank)
        count = len(cores)
        margin = rank_margin_mass([x.best_rank for x in cores], strong_rank=args.strong_rank,
                                  margin_band=args.margin_band)
        # Soft improvement (PLAN §1): count drop, OR count tie with lower margin mass.
        improving = count < seed_count or (count == seed_count and margin < seed_margin)
        sc_tm = plddt = passed = None
        global_ca_rmsd = active_site_sidechain_rmsd = None
        max_anchor_sidechain_rmsd = max_anchor_atom_distance = None
        active_site_complete = active_site_min_plddt = None
        if improving:
            m = oracles.struct_fn(protein_id, c.seq)
            sc_tm, plddt = m.scTM, m.pLDDT
            global_ca_rmsd = getattr(m, "global_ca_RMSD", None)
            active_site_sidechain_rmsd = getattr(m, "active_site_sidechain_RMSD", None)
            max_anchor_sidechain_rmsd = getattr(m, "max_anchor_sidechain_RMSD", None)
            max_anchor_atom_distance = getattr(m, "max_anchor_atom_distance", None)
            active_site_complete = getattr(m, "active_site_complete", None)
            active_site_min_plddt = getattr(m, "active_site_min_pLDDT", None)
            passed, _ = structure_gate(seed_metrics, m, scTM_eps=args.scTM_eps,
                                       scRMSD_max=args.scRMSD_max,
                                       active_site_RMSD_max=args.active_site_RMSD_max,
                                       max_anchor_sidechain_RMSD_max=(
                                           args.max_anchor_sidechain_RMSD_max
                                       ))
        out.append({
            "protein_id": protein_id, "design_idx": int(design_idx), "muts": c.desc,
            "positions": list(c.positions), "head_proxy": hp, "core_count": count,
            "rank_margin_mass": margin, "scTM": sc_tm, "pLDDT": plddt, "passed": passed,
            "global_ca_RMSD": global_ca_rmsd,
            "active_site_sidechain_RMSD": active_site_sidechain_rmsd,
            "max_anchor_sidechain_RMSD": max_anchor_sidechain_rmsd,
            "max_anchor_atom_distance": max_anchor_atom_distance,
            "active_site_complete": active_site_complete,
            "active_site_min_pLDDT": active_site_min_plddt,
            "eliminates": bool(count < seed_count),
        })
    return out


# --------------------------------------------------------------------------- #
# Refine mode: full beam search per seed -> shortlist
# --------------------------------------------------------------------------- #
def _run_refine_seed(oracles, protein_id, seed_seq, orig_design_idx, seed_val, anchors, args):
    tw_fn = _make_target_window_idx_fn(*oracles.window_coords_fn(seed_seq))
    propose_fn = make_propose_fn(oracles, protein_id, anchors, max_pairs=args.max_pairs,
                                 target_window_idx_fn=tw_fn)
    res = refine_sequence(
        protein_id, seed_seq,
        propose_fn=propose_fn, head_fn=oracles.head_fn, nmp_fn=oracles.nmp_fn,
        struct_fn=oracles.struct_fn, anchors=anchors,
        target_window_idx_fn=tw_fn,
        strong_rank=args.strong_rank, margin_band=args.margin_band, scTM_eps=args.scTM_eps,
        scRMSD_max=args.scRMSD_max, active_site_RMSD_max=args.active_site_RMSD_max,
        max_anchor_sidechain_RMSD_max=args.max_anchor_sidechain_RMSD_max,
        topB=args.topB, beam_width=args.beam_width, max_rounds=args.max_rounds,
        patience=args.patience, max_path_mutations=args.max_path_mutations,
        refold_cap=args.refold_cap, allow_structure_unknown=args.allow_structure_unknown,
        incremental_nmp=args.incremental_nmp, nmp_context_margin=args.nmp_context_margin,
        log_fn=(lambda m: print(m, flush=True)),
    )
    rich = []
    if res.shortlist:
        for entry in res.shortlist:
            rich.append({
                "protein_id": protein_id, "orig_design_idx": int(orig_design_idx),
                "sequence_original": seed_seq, "sequence_refined": entry["seq"],
                "n_mutations": len(entry["muts"].split(",")) if entry["muts"] else 0,
                "muts": entry["muts"], "core_count_before": res.seed_core_count,
                "core_count_after": entry["core_count"], "scTM_after": entry["scTM"],
                "pLDDT_after": entry["pLDDT"], "scRMSD_after": entry["scRMSD"],
                "active_site_RMSD_after": entry["active_site_RMSD"], "diverged": False,
                "global_ca_RMSD_after": entry["global_ca_RMSD"],
                "active_site_sidechain_RMSD_after": entry["active_site_sidechain_RMSD"],
                "max_anchor_sidechain_RMSD_after": entry["max_anchor_sidechain_RMSD"],
                "max_anchor_atom_distance_after": entry["max_anchor_atom_distance"],
                "active_site_complete_after": entry["active_site_complete"],
                "active_site_min_pLDDT_after": entry["active_site_min_pLDDT"],
            })
    else:
        # No accepted refinement: emit a no-op row so every seed appears for re-eval.
        m = res.best_structure
        rich.append({
            "protein_id": protein_id, "orig_design_idx": int(orig_design_idx),
            "sequence_original": seed_seq, "sequence_refined": res.best_seq,
            "n_mutations": 0, "muts": "", "core_count_before": res.seed_core_count,
            "core_count_after": res.best_core_count, "scTM_after": getattr(m, "scTM", None),
            "pLDDT_after": getattr(m, "pLDDT", None), "scRMSD_after": getattr(m, "scRMSD", None),
            "active_site_RMSD_after": getattr(m, "active_site_RMSD", None),
            "global_ca_RMSD_after": getattr(m, "global_ca_RMSD", None),
            "active_site_sidechain_RMSD_after": getattr(
                m, "active_site_sidechain_RMSD", None
            ),
            "max_anchor_sidechain_RMSD_after": getattr(
                m, "max_anchor_sidechain_RMSD", None
            ),
            "max_anchor_atom_distance_after": getattr(m, "max_anchor_atom_distance", None),
            "active_site_complete_after": getattr(m, "active_site_complete", None),
            "active_site_min_pLDDT_after": getattr(m, "active_site_min_pLDDT", None),
            "diverged": bool(res.diverged),
        })
    trace = [{"protein_id": protein_id, "orig_design_idx": int(orig_design_idx), **t}
             for t in res.trace]
    return rich, trace, seed_val


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _resolve_proteins(args, generated: pd.DataFrame) -> list[str]:
    if args.proteins and args.proteins != "all":
        return [p.strip() for p in args.proteins.split(",") if p.strip()]
    return sorted(generated["protein_id"].astype(str).unique())


def _seed_design_idx(row, running: int) -> int:
    """Provenance design_idx for a seed-table row: explicit design_idx, else canonical
    design_NNNN, else a running per-protein index."""
    if "design_idx" in row and pd.notna(row["design_idx"]):
        return int(row["design_idx"])
    match = re.match(r"^design_(\d+)$", str(row.get("design_id", "")))
    return int(match.group(1)) if match else running


def _seeds_from_table(seed_table: str, proteins_arg: str,
                      cost_col: str | None = None) -> dict[str, list[dict]]:
    """Seeds from a self-contained protein_id/design_id/sequence parquet (bypasses
    --run-dir best-of-N selection; refines every listed row). When ``cost_col`` is a
    present column, each seed carries ``_cost`` for difficulty-balanced array sharding."""
    df = pd.read_parquet(seed_table)
    for col in ("protein_id", "sequence"):
        if col not in df.columns:
            raise ValueError(f"--seed-table missing required column {col!r}")
    if proteins_arg and proteins_arg != "all":
        want = {p.strip() for p in proteins_arg.split(",")}
        df = df[df["protein_id"].astype(str).isin(want)]
    has_cost = bool(cost_col) and cost_col in df.columns
    seeds_by_protein: dict[str, list[dict]] = {}
    has_seed = "seed" in df.columns
    for _, row in df.iterrows():
        pid = str(row["protein_id"])
        lst = seeds_by_protein.setdefault(pid, [])
        lst.append({
            "sequence": str(row["sequence"]),
            "design_idx": _seed_design_idx(row, len(lst)),
            "seed": int(row["seed"]) if has_seed and pd.notna(row.get("seed")) else -1,
            "_cost": float(row[cost_col]) if has_cost and pd.notna(row.get(cost_col)) else None,
        })
    return seeds_by_protein


def _shard_seeds(seeds_by_protein: dict[str, list[dict]], n_shards: int, shard_idx: int,
                 shard_by: str) -> dict[str, list[dict]]:
    """Partition the flattened seed list across array tasks (deterministic — every task
    recomputes the identical partition). ``balanced`` = LPT bin-pack by each seed's
    ``_cost`` (falls back to stride if any cost is missing); ``stride`` = round-robin."""
    flat = [(pid, s) for pid in sorted(seeds_by_protein) for s in seeds_by_protein[pid]]
    if shard_by == "balanced" and flat and all(s.get("_cost") is not None for _, s in flat):
        order = sorted(range(len(flat)), key=lambda i: -float(flat[i][1]["_cost"]))
        load = [0.0] * n_shards
        assign: dict[int, int] = {}
        for i in order:
            j = min(range(n_shards), key=lambda s: load[s])
            load[j] += max(float(flat[i][1]["_cost"]), 1.0)
            assign[i] = j
        keep = [flat[i] for i in range(len(flat)) if assign[i] == shard_idx]
    else:
        keep = [flat[i] for i in range(len(flat)) if i % n_shards == shard_idx]
    out: dict[str, list[dict]] = {}
    for pid, s in keep:
        out.setdefault(pid, []).append(s)
    return out


def _all_seeds(generated: pd.DataFrame, protein_id: str) -> list[dict]:
    """Every design for a protein — used when --eval-immune-dir is omitted (no best-of-N)."""
    g = generated[generated["protein_id"].astype(str) == protein_id]
    has_seed = "seed" in g.columns
    return [
        {"sequence": str(r["sequence"]), "design_idx": int(r["design_idx"]),
         "seed": int(r["seed"]) if has_seed and pd.notna(r.get("seed")) else -1}
        for _, r in g.iterrows()
    ]


def _gather_seeds(args) -> tuple[dict[str, list[dict]], list[str]]:
    """Resolve per-protein seed lists from --seed-table, or --run-dir (with optional
    --eval-immune-dir best-of-N; omit it to refine every design)."""
    if args.seed_table:
        seeds_by_protein = _seeds_from_table(args.seed_table, args.proteins, args.shard_cost_col)
    else:
        if not args.run_dir:
            raise ValueError("provide either --seed-table or --run-dir")
        generated = _load_sharded(Path(args.run_dir), "generation", "generated.parquet")
        imm_head = (_load_sharded(Path(args.eval_immune_dir), ".", "imm_head.parquet")
                    if args.eval_immune_dir else None)
        seeds_by_protein = {
            pid: (_select_seeds(generated, imm_head, pid, args.seeds_per_protein)
                  if imm_head is not None else _all_seeds(generated, pid))
            for pid in _resolve_proteins(args, generated)
        }
    if args.n_shards > 1:
        seeds_by_protein = _shard_seeds(seeds_by_protein, args.n_shards, args.shard_idx,
                                        args.shard_by)
    return seeds_by_protein, sorted(seeds_by_protein)


def run_refinement(args, oracles: Oracles) -> int:
    # v0 deferrals: fail-fast rather than silently ignoring a non-default switch.
    if args.target_source != "nmp":
        raise NotImplementedError(
            "--target-source head (Mode 2) is deferred in v0; only 'nmp' is wired")
    if args.head_high_topk:
        raise NotImplementedError(
            "--head-high-topk (head-high editable augmentation) is a v0 deferral; pass 0 "
            "(editable = pockets minus anchors)")
    manifest = load_constraint_manifest(args.constraint_manifest) if args.constraint_manifest else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds_by_protein, proteins = _gather_seeds(args)
    n_seeds = sum(len(v) for v in seeds_by_protein.values())
    print(
        f"[refine] mode={args.mode} proteins={len(proteins)} seeds={n_seeds} "
        f"source={'seed-table' if args.seed_table else 'run-dir'} topB={args.topB} "
        f"beam={args.beam_width} refold_cap={args.refold_cap} strong_rank={args.strong_rank} margin_band={args.margin_band} "
        f"scTM_eps={args.scTM_eps} max_pairs={args.max_pairs} "
        f"max_path_mutations={args.max_path_mutations}",
        flush=True,
    )

    if args.mode == "ceiling":
        rows = []
        for pid in proteins:
            for seed in seeds_by_protein[pid]:
                anchors = _anchors_for(manifest, pid, str(seed["sequence"]))
                rows.extend(_run_ceiling_seed(
                    oracles, pid, str(seed["sequence"]), seed["design_idx"], anchors, args))
        (out_dir / "ceiling").mkdir(exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_dir / "ceiling" / "candidates.parquet", index=False)
        print(f"[refine] ceiling: {len(rows)} candidate rows -> ceiling/candidates.parquet", flush=True)
        _write_config(out_dir / "ceiling" / "refine_config.json", args)
        return 0

    # refine mode
    refined_dir = out_dir / "refined"
    refined_dir.mkdir(exist_ok=True)
    _write_config(refined_dir / "refine_config.json", args)
    rich_rows, trace_rows, eval_rows = [], [], []
    per_protein_idx: dict[str, int] = {}
    n_done = 0
    for pid in proteins:
        for seed in seeds_by_protein[pid]:
            anchors = _anchors_for(manifest, pid, str(seed["sequence"]))
            rich, trace, seed_val = _run_refine_seed(
                oracles, pid, str(seed["sequence"]), seed["design_idx"],
                int(seed.get("seed", -1)), anchors, args)
            for r in rich:
                didx = per_protein_idx.get(pid, 0)
                per_protein_idx[pid] = didx + 1
                r["design_idx"] = didx
                rich_rows.append(r)
                eval_rows.append({
                    "protein_id": pid, "design_idx": didx, "sequence": r["sequence_refined"],
                    "seed": int(seed_val), "wall_seconds": 0.0,
                })
            trace_rows.extend(trace)
            n_done += 1
            # Incremental, ATOMIC flush after every seed: a walltime/GPU-idle kill keeps all
            # completed seeds' designs (the driver otherwise only wrote at the very end).
            _flush_refine_outputs(refined_dir, rich_rows, eval_rows, trace_rows)
            print(f"[refine] seed {n_done} done ({pid}) -> {len(rich_rows)} designs so far",
                  flush=True)

    rich_df = _flush_refine_outputs(refined_dir, rich_rows, eval_rows, trace_rows)
    n0 = int((rich_df["core_count_after"] == 0).sum()) if len(rich_df) else 0
    print(
        f"[refine] refine: {len(rich_rows)} refined rows ({n0} reached count 0) "
        f"-> refined/refined_designs.parquet + evaluator_ready.parquet",
        flush=True,
    )
    return 0


def _write_config(path: Path, args) -> None:
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)


def _atomic_write(refined_dir: Path, df: pd.DataFrame, name: str) -> None:
    """Write ``df`` to ``refined_dir/name`` via a temp file + atomic rename, so a walltime /
    GPU-idle kill mid-write never leaves a half-written parquet."""
    tmp = refined_dir / (name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(refined_dir / name)


def _flush_refine_outputs(refined_dir: Path, rich_rows, eval_rows, trace_rows) -> pd.DataFrame:
    """Atomically (re)write the cumulative refine outputs so a mid-run kill/timeout keeps
    every completed seed's designs (write to a temp then rename). Returns the rich DataFrame."""
    rich_df = pd.DataFrame(rich_rows).drop(columns=["orig_design_idx"], errors="ignore")
    _atomic_write(refined_dir, rich_df, "refined_designs.parquet")
    _atomic_write(refined_dir, pd.DataFrame(eval_rows, columns=GENERATED_COLUMNS), "evaluator_ready.parquet")
    _atomic_write(refined_dir, pd.DataFrame(trace_rows), "refine_trace.parquet")
    return rich_df


def _free_gpu() -> None:
    """Drop pending refs + free the CUDA caching allocator (called between the driver's
    oracle models and the eval-side ESMFold so only one ESMFold is resident at a time)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _write_final_metrics_status(refined_dir: Path, *, ok: bool, error: str | None = None,
                                n_designs: int | None = None, n_failures: int | None = None,
                                structural_metrics_version: str | None = None) -> None:
    """Completeness marker written as the LAST step of the metrics pass (ok=True) or by main()'s
    guard on failure (ok=False). Consumers gate a full metric set on ok=true."""
    status = {"ok": ok}
    if error is not None:
        status["error"] = error
    if n_designs is not None:
        status["n_designs"] = n_designs
    if n_failures is not None:
        status["n_failures"] = n_failures
    if structural_metrics_version is not None:
        status["structural_metrics_version"] = str(structural_metrics_version)
    with open(refined_dir / "final_metrics_status.json", "w") as f:
        json.dump(status, f, indent=2, default=str)


def _run_final_metrics(args) -> None:
    """Emit the full evaluate_phase_c metric tables for EVERY final refined design by
    REUSING evaluate_phase_c's own row builders (zero schema drift), reading the just-written
    ``refined/evaluator_ready.parquet``. Runs once after the refine loop.

    The search keeps only scalar metrics (core_count, scTM) and discards the raw per-window
    NMP rows, per-residue head hotspots, and side-chain geometry; this pass
    re-derives and persists them. Every final sequence was ESMFold-refolded during search,
    so ``refold`` is a cache hit here (keyed on (protein_id, sequence)) and structure is cheap.

    Writes (evaluate basenames, atomic): imm_head.parquet, imm_nmp.parquet, structural.parquet,
    structural_residues.parquet (+ imm_head_residues / imm_nmp_peptides under --imm-full).
    NOTE: imm_nmp.n_strong_binders (rank_EL% < strong_binder_threshold, per-window) is a
    DIFFERENT quantity from the search objective core_count_after (distinct 9-mer cores).
    """
    refined_dir = Path(args.out_dir) / "refined"
    gen_parquet = refined_dir / "evaluator_ready.parquet"
    if not gen_parquet.exists():
        print(f"[refine] final-metrics: {gen_parquet} missing -- skipping", flush=True)
        return

    from scripts.evaluate_phase_c import (
        load_generated_designs, load_test_lookup,
        build_head_predictor, build_nmp_runner,
        evaluate_immunogenicity_rows, evaluate_structural_rows,
        _load_anchor_indices_by_protein,
    )

    gdf = load_generated_designs(gen_parquet)
    if len(gdf) == 0:
        print("[refine] final-metrics: 0 designs -- skipping", flush=True)
        return
    _test_df, test_lookup = load_test_lookup(args.test_set_parquet)
    print(
        f"[refine] final-metrics: {len(gdf)} designs -> imm_head/imm_nmp/structural/"
        f"structural_residues (imm_full={args.imm_full})",
        flush=True,
    )

    # --- immunogenicity: head (global_risk/hotspots) + NMP (n_strong_binders, ...) ---
    predictor = build_head_predictor(
        checkpoint_path=args.head_checkpoint, config_dir=args.head_config_dir,
        variant_id=args.head_variant_id, device=args.head_device)
    nmp_runner = build_nmp_runner(
        binary_path=args.netmhciipan_bin, batch_size=args.nmp_batch_size,
        n_workers=args.nmp_workers, timeout=args.nmp_timeout,
        max_lengths_per_call=args.nmp_max_lengths_per_call)
    head_df, nmp_df, imm_fail, imm_res_df, imm_pep_df = evaluate_immunogenicity_rows(
        gdf, predictor=predictor, nmp_runner=nmp_runner, allele=args.allele,
        strong_binder_threshold=args.strong_binder_threshold,
        nmp_batch_size=args.nmp_batch_size, hotspot_threshold=args.hotspot_threshold,
        full=args.imm_full, run_nmp=True, return_full_tables=True)
    del predictor, nmp_runner
    _free_gpu()  # release the head predictor before evaluate_structural_rows loads ESMFold

    # Persist the immunogenicity tables NOW — the NMP re-score is the CPU-costly part of this
    # pass, and a later structural failure must not discard already-completed work.
    _atomic_write(refined_dir, head_df, "imm_head.parquet")
    _atomic_write(refined_dir, nmp_df, "imm_nmp.parquet")
    if args.imm_full:
        _atomic_write(refined_dir, imm_res_df, "imm_head_residues.parquet")
        _atomic_write(refined_dir, imm_pep_df, "imm_nmp_peptides.parquet")

    # --- structure: canonical v2 summary + index-addressable side-chain rows ---
    anchor_indices_by_protein = _load_anchor_indices_by_protein(
        args.constraint_manifest,
        test_lookup,
    )
    evaluated = evaluate_structural_rows(
        gdf, test_lookup, pdb_root=args.pdb_root, refold_backend="esmfold",
        device=args.head_device, tmalign_bin="TMalign",
        esmfold_cache_dir=args.esmfold_cache_dir, return_residue_metrics=True,
        return_v2_metrics=True,
        legacy_metrics=False,
        anchor_indices_by_protein=anchor_indices_by_protein,
    )
    structural_df, struct_fail, struct_res_df = evaluated
    _atomic_write(refined_dir, structural_df, "structural.parquet")
    _atomic_write(refined_dir, struct_res_df, "structural_residues.parquet")

    # surface per-design failures (do not silently drop them)
    failures = list(imm_fail) + list(struct_fail)
    if failures:
        with open(refined_dir / "final_metrics_failures.json", "w") as f:
            json.dump(failures, f, indent=2, default=str)
    # completeness marker (LAST write): consumers gate a full set on ok=true
    _write_final_metrics_status(
        refined_dir,
        ok=True,
        n_designs=len(gdf),
        n_failures=len(failures),
        structural_metrics_version="v2",
    )
    print(
        f"[refine] final-metrics done: imm_head={len(head_df)} imm_nmp={len(nmp_df)} "
        f"structural={len(structural_df)} structural_residues={len(struct_res_df)} "
        f"failures={len(failures)} -> {refined_dir}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Real oracle construction (heavy deps; bypassed by the smoke test)
# --------------------------------------------------------------------------- #
def build_oracles(args) -> Oracles:
    """Build the real (head, nmp, struct, target_window_idx) oracle callables.

    Only place that touches torch / NetMHCIIpan / ESMFold. See PLAN_RF_REFINE R4(b).
    """
    required = {
        "--esmfold-cache-dir": args.esmfold_cache_dir,   # refold pdb_path is None without it
        "--test-set-parquet": args.test_set_parquet,      # resolve_structure_path needs the row
        "--pdb-root": args.pdb_root,                       # reference backbone for TMalign
        "--head-checkpoint": args.head_checkpoint,         # head predictor weights
        "--head-config-dir": args.head_config_dir,         # head model/ablation/inference yamls
        "--netmhciipan-bin": args.netmhciipan_bin,         # NetMHCIIpan binary (StandaloneRunner)
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"build_oracles requires: {', '.join(missing)}")
    requested_structure_metrics = _parse_refinement_structure_metrics(
        args.refinement_structure_metrics
    )
    if args.active_site_RMSD_max is not None:
        raise ValueError(
            "--active-site-RMSD-max is the retired C-alpha active-site gate and has no "
            "real standalone oracle; use --max-anchor-sidechain-RMSD-max"
        )
    if (
        args.max_anchor_sidechain_RMSD_max is not None
        and (
            not np.isfinite(args.max_anchor_sidechain_RMSD_max)
            or args.max_anchor_sidechain_RMSD_max <= 0.0
        )
    ):
        raise ValueError("--max-anchor-sidechain-RMSD-max must be finite and positive")
    if (
        args.max_anchor_sidechain_RMSD_max is not None
        and "sidechain" not in requested_structure_metrics
    ):
        raise ValueError(
            "--max-anchor-sidechain-RMSD-max requires "
            "--refinement-structure-metrics sidechain"
        )
    if args.max_anchor_sidechain_RMSD_max is not None and not args.constraint_manifest:
        raise ValueError(
            "--max-anchor-sidechain-RMSD-max requires --constraint-manifest"
        )

    # --- Head (reuse the existing build_head_scorer helper) ---
    from scripts.run_if_phase_c1 import build_head_scorer  # ControllerSetup-style seam
    from types import SimpleNamespace
    import yaml

    head_config_dir = Path(args.head_config_dir)
    inf = yaml.safe_load(open(head_config_dir / "inference.yaml")) or {}
    inf_sec = inf.get("inference", {}) if isinstance(inf, dict) else {}
    # build_head_scorer reads setup.{head_config_dir, head_checkpoint(Path), head_variant_id,
    # head_device, allele, head_allele_idx, config.head.score_scale, head_window_batch_size}.
    setup = SimpleNamespace(
        head_config_dir=head_config_dir, head_checkpoint=Path(args.head_checkpoint),
        head_variant_id=args.head_variant_id, head_device=args.head_device,
        head_allele_idx=args.head_allele_idx,
        head_window_batch_size=args.head_window_batch_size,
        allele=args.allele,   # OnlineHeadScorer allele
        config=SimpleNamespace(head=SimpleNamespace(score_scale="raw_logit")),  # only value supported
    )
    scorer = build_head_scorer(
        setup, window_k_min=int(inf_sec.get("min_k", 12)),
        window_k_max=int(inf_sec.get("max_k", 25)),
    )

    head_chunk = max(1, int(args.head_chunk))

    def head_fn(pid, seqs):
        # Chunk the encoder forward: a beam-multiplied pool (beam_width x per-state candidates,
        # ~8k on hard seeds) in one forward_batched allocates >60 GiB and OOMs the H200. The
        # window template is length-invariant across a protein's candidates, so per-chunk
        # window_risks [chunk, W] concatenate along axis 0 exactly like a single call.
        seqs = list(seqs)
        rows = []
        for i in range(0, len(seqs), head_chunk):
            chunk = seqs[i:i + head_chunk]
            compact = scorer.score_window_risk_batch_same_protein(
                protein_id=pid, records=[(str(j), s) for j, s in enumerate(chunk)])
            rows.append(np.asarray(compact.window_risks))
        return rows[0] if len(rows) == 1 else np.concatenate(rows, axis=0)

    def window_coords_fn(seq):
        # Per-sequence window template (length-dependent) — a fixed template would
        # mis-map cores past its length in ~300aa uricases.
        compact = scorer.score_window_risk_batch_same_protein(
            protein_id="__windows__", records=[("0", seq)])
        return np.asarray(compact.window_starts_0b), np.asarray(compact.window_ends_0b)

    # --- NMP (batched; FRACTION units) ---
    from epitope_head.data.netmhciipan_runner import StandaloneRunner
    from inverse_folding.evaluation.immunogenicity import NMP_PEP_LENGTHS

    runner = StandaloneRunner(
        binary_path=args.netmhciipan_bin,
        batch_size=args.nmp_batch_size, subprocess_timeout=args.nmp_timeout,
        max_lengths_per_call=args.nmp_max_lengths_per_call, n_workers=args.nmp_workers)

    def nmp_fn(pid, seqs):
        entries = [(f"{pid}__c{i}", s) for i, s in enumerate(seqs)]
        scored = runner.score_batch(entries, args.allele, NMP_PEP_LENGTHS)
        out = []
        for i, seq in enumerate(seqs):
            key = f"{pid}__c{i}"
            # NMP is the final gate: a missing/incomplete result must fail-fast, NOT be
            # read as an empty (= no-epitope) row set (that would silently pass the count).
            if key not in scored:
                raise RuntimeError(
                    f"NetMHCIIpan returned no result for {key}; refusing to treat a failed "
                    "NMP call as 'no epitope' (NMP is the final gate).")
            by_len = scored[key]
            # Every peptide length that FITS the sequence yields windows for a real
            # (200+aa) design, so a missing fitting length signals a partial failure.
            missing = [k for k in NMP_PEP_LENGTHS if k <= len(seq) and k not in by_len]
            if missing:
                raise RuntimeError(
                    f"NetMHCIIpan result for {key} is missing peptide lengths {missing}; "
                    "incomplete scoring would under-count cores — aborting.")
            rows = []
            for _pep_len, plist in by_len.items():
                for ps in plist:
                    rows.append({"pos": ps.pos, "pep_length": ps.pep_length,
                                 "peptide": ps.peptide, "core": ps.core,
                                 "rank_EL": ps.el_rank})  # FRACTION, no ×100
            out.append(rows)
        return out

    # --- Structure (ESMFold refold + TMalign vs the WT/target backbone) ---
    from inverse_folding.evaluation.refold import load_refold_model, refold
    from inverse_folding.evaluation.tmalign import run_tmalign
    from inverse_folding.reference_flow.refine import StructureMetrics
    from inverse_folding.reference_flow.runtime import resolve_structure_path

    evaluate_prediction_v2 = None
    prepare_reference_context_v2 = None
    if requested_structure_metrics:
        from inverse_folding.evaluation.structural_metrics_v2 import (
            evaluate_prediction as evaluate_prediction_v2,
            prepare_reference_context as prepare_reference_context_v2,
        )

    model = load_refold_model("esmfold", device=args.head_device)
    test_df = pd.read_parquet(args.test_set_parquet)
    test_lookup = {str(r["protein_id"]): r for _, r in test_df.iterrows()}
    manifest = load_constraint_manifest(args.constraint_manifest) if args.constraint_manifest else None
    ref_path_cache: dict[str, Path] = {}
    reference_context_cache: dict[str, object] = {}

    def _reference_path(pid: str) -> Path:
        if pid not in test_lookup:
            raise KeyError(f"protein {pid!r} is absent from --test-set-parquet")
        if pid not in ref_path_cache:
            ref_path_cache[pid] = Path(resolve_structure_path(test_lookup[pid], args.pdb_root))
        return ref_path_cache[pid]

    def _reference_context(pid: str):
        if pid not in reference_context_cache:
            ref_sequence = str(test_lookup[pid]["sequence"])
            anchors = _anchors_for(manifest, pid, ref_sequence)
            reference_context_cache[pid] = prepare_reference_context_v2(
                _reference_path(pid),
                ref_sequence=ref_sequence,
                anchor_indices=anchors,
            )
        return reference_context_cache[pid]

    def struct_fn(pid, seq):
        pred = refold(seq, pid, "refine", backend="esmfold",
                      cache_dir=args.esmfold_cache_dir, model=model)
        ref = _reference_path(pid)
        tm = run_tmalign(pred_pdb=str(pred["pdb_path"]), ref_pdb=str(ref), cache_dir=None)
        v2 = None
        if requested_structure_metrics:
            v2 = evaluate_prediction_v2(
                _reference_context(pid),
                str(pred["pdb_path"]),
                design_sequence=seq,
                metrics=requested_structure_metrics,
            )
        return StructureMetrics(scTM=float(tm["tm_score"]), pLDDT=float(pred["pLDDT"]),
                                scRMSD=float(tm["rmsd"]),
                                global_ca_RMSD=(None if v2 is None else v2.global_ca_rmsd),
                                active_site_sidechain_RMSD=(
                                    None if v2 is None else v2.active_site_sidechain_rmsd
                                ),
                                max_anchor_sidechain_RMSD=(
                                    None if v2 is None else v2.max_anchor_sidechain_rmsd
                                ),
                                max_anchor_atom_distance=(
                                    None if v2 is None else v2.max_anchor_atom_distance
                                ),
                                active_site_complete=(
                                    None if v2 is None else v2.active_site_complete
                                ),
                                active_site_min_pLDDT=(
                                    None if v2 is None
                                    else v2.predicted_active_site_min_plddt
                                ))

    return Oracles(head_fn=head_fn, nmp_fn=nmp_fn, struct_fn=struct_fn,
                   window_coords_fn=window_coords_fn)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RF refinement (targeted epitope elimination)")
    p.add_argument("--run-dir", default=None,
                   help="a clean generated.parquet (file), a dir holding it, or "
                        "generation/*shard*/generated.parquet (alternative to --seed-table)")
    p.add_argument("--eval-immune-dir", default=None,
                   help="optional dir holding */imm_head.parquet (global_risk best-of-N seed "
                        "selection); omit to refine EVERY design in --run-dir")
    p.add_argument("--seed-table", default=None,
                   help="self-contained protein_id/design_id/sequence parquet; bypasses "
                        "--run-dir selection and refines every listed seed")
    p.add_argument("--test-set-parquet", default=None, help="test-set parquet (rows for resolve_structure_path)")
    p.add_argument("--constraint-manifest", default=None, help="active-site hard-anchor manifest (optional)")
    p.add_argument("--allele", required=True)
    p.add_argument("--proteins", default="all", help="comma list or 'all'")
    p.add_argument("--seeds-per-protein", type=int, default=3)
    p.add_argument("--mode", choices=("ceiling", "refine"), default="refine")
    p.add_argument("--target-source", choices=("nmp", "head"), default="nmp",
                   help="Mode 1 (nmp, default); 'head' (Mode 2) is deferred in v0")
    # head (mirror run_if_phase_c1.py)
    p.add_argument("--head-checkpoint", default=None)
    p.add_argument("--head-config-dir", default=None)
    p.add_argument("--head-variant-id", default=None)
    p.add_argument("--head-device", default="cuda")
    p.add_argument("--head-allele-idx", type=int, default=0)
    p.add_argument("--head-window-batch-size", type=int, default=None)
    p.add_argument("--head-chunk", type=int, default=512,
                   help="max candidate seqs per head encoder forward (bounds GPU mem; the "
                        "beam-multiplied pool OOMs the untiled forward on hard seeds)")
    # nmp (mirror evaluate_phase_c.py accelerated knobs)
    p.add_argument("--netmhciipan-bin", default=None, help="path to the NetMHCIIpan binary")
    p.add_argument("--nmp-batch-size", type=int, default=8)
    p.add_argument("--nmp-max-lengths-per-call", type=int, default=4)
    p.add_argument("--nmp-workers", type=int, default=8)
    p.add_argument("--nmp-timeout", type=int, default=1800)
    # structure
    p.add_argument("--pdb-root", default=None)
    p.add_argument("--esmfold-cache-dir", default=None)
    # incremental NMP splice (EXACT for per-window NMP; the primary NMP-volume cut)
    p.add_argument("--incremental-nmp", dest="incremental_nmp", action="store_true", default=True,
                   help="score only each candidate's mutated sub-sequence + splice vs the seed (exact, default on)")
    p.add_argument("--no-incremental-nmp", dest="incremental_nmp", action="store_false",
                   help="disable the splice; score every candidate over its full length")
    p.add_argument("--nmp-context-margin", type=int, default=30,
                   help="sub-sequence half-window for the splice (>= NMP context radius; 30 covers 25-mer+context)")
    # seed job-array sharding (SLURM --array over seeds; each task refines its shard)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0, help="this task's shard (SLURM_ARRAY_TASK_ID)")
    p.add_argument("--shard-by", choices=("balanced", "stride"), default="balanced",
                   help="balanced = LPT bin-pack by --shard-cost-col; stride = round-robin")
    p.add_argument("--shard-cost-col", default="nmp_n_strong_binders",
                   help="seed-table column used as the difficulty weight for balanced sharding")
    # search knobs (§7)
    p.add_argument("--strong-rank", type=float, default=0.02)
    p.add_argument("--margin-band", type=float, default=0.10)
    p.add_argument("--scTM-eps", dest="scTM_eps", type=float, default=0.05)
    p.add_argument("--scRMSD-max", dest="scRMSD_max", type=float, default=None)
    p.add_argument("--active-site-RMSD-max", dest="active_site_RMSD_max", type=float, default=None)
    p.add_argument(
        "--refinement-structure-metrics",
        default="",
        help=(
            "comma-separated lightweight v2 metrics evaluated inside the refold loop: "
            "global_ca_rmsd,plddt,sidechain (default: none; reference contexts are cached)"
        ),
    )
    p.add_argument(
        "--max-anchor-sidechain-RMSD-max",
        dest="max_anchor_sidechain_RMSD_max",
        type=float,
        default=None,
        help=(
            "fail-closed ceiling on the worst per-anchor all-heavy-side-chain RMSD; "
            "requires sidechain in --refinement-structure-metrics and a constraint manifest"
        ),
    )
    p.add_argument("--topB", type=int, default=None)
    p.add_argument("--beam-width", type=int, default=8,
                   help="working states between rounds; head-ranks the pool (cheap), topB caps NMP")
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--max-path-mutations", type=int, default=8,
                   help="cheap refold-free cap on total edits per search path (branch-waste bound)")
    p.add_argument("--refold-cap", dest="refold_cap", type=int, default=16,
                   help="max count-dropping candidates refolded per round; bounds ESMFold cost "
                        "and yields the lean ranked shortlist (pass a large value to disable)")
    p.add_argument("--head-high-topk", type=int, default=0,
                   help="head-high editable augmentation (v0 deferral: must be 0)")
    p.add_argument("--allow-structure-unknown", action="store_true")
    # full-metrics emission: after refinement, reuse evaluate_phase_c's row builders on the
    # final designs to persist the metrics the search discards (per-window NMP aggregates,
    # head hotspots, per-residue Kabsch CA RMSD). Refolds are ESMFold cache hits.
    p.add_argument("--emit-eval-metrics", dest="emit_eval_metrics", action="store_true", default=True,
                   help="emit evaluate_phase_c-schema metric tables (imm_head/imm_nmp/structural/"
                        "structural_residues) for every final design (default on)")
    p.add_argument("--no-eval-metrics", dest="emit_eval_metrics", action="store_false",
                   help="skip the post-refine full-metrics emission")
    p.add_argument("--strong-binder-threshold", dest="strong_binder_threshold", type=float, default=2.0,
                   help="imm_nmp strong-binder rank_EL%% cutoff (evaluate_phase_c default 2.0; distinct "
                        "from --strong-rank, the search's FRACTION distinct-core threshold)")
    p.add_argument("--hotspot-threshold", dest="hotspot_threshold", type=float, default=0.5,
                   help="imm_head n_hotspot_positions cutoff (evaluate_phase_c default 0.5)")
    p.add_argument("--imm-full", dest="imm_full", action="store_true", default=False,
                   help="also emit imm_head_residues + imm_nmp_peptides (evaluate_phase_c --imm-full)")
    p.add_argument(
        "--structural-metrics-v2",
        action="store_true",
        default=True,
        help=(
            "deprecated no-op: v2 is the canonical structural output"
        ),
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--print-config", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.allele = _canonical_allele(args.allele)  # tolerate the 'HLA-DRB1_07_01' run-dir tag
    if args.print_config:
        print("[refine] config: " + json.dumps(vars(args), sort_keys=True, default=str), flush=True)
    oracles = build_oracles(args)
    rc = run_refinement(args, oracles)
    if rc == 0 and args.mode == "refine" and args.emit_eval_metrics:
        del oracles          # free the driver's ESMFold/head before the eval pass reloads ESMFold
        _free_gpu()
        # Best-effort: the refinement outputs are already flushed and are the expensive,
        # protected artifact. An eval-pass failure (poison-pill length mismatch, CUDA OOM on
        # model reload, NMP hiccup) must NOT fail the job or lose refinement — the metrics are
        # cheap and independently re-runnable via evaluate_phase_c on refined/evaluator_ready.parquet.
        try:
            _run_final_metrics(args)
        except Exception:
            import traceback
            print("[refine] final-metrics FAILED (refinement outputs intact; re-run eval on "
                  "refined/evaluator_ready.parquet):", flush=True)
            traceback.print_exc()
            _write_final_metrics_status(Path(args.out_dir) / "refined", ok=False,
                                        error=traceback.format_exc(),
                                        structural_metrics_version="v2")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
