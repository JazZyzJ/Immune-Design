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
        if improving:
            m = oracles.struct_fn(protein_id, c.seq)
            sc_tm, plddt = m.scTM, m.pLDDT
            passed, _ = structure_gate(seed_metrics, m, scTM_eps=args.scTM_eps,
                                       scRMSD_max=args.scRMSD_max,
                                       active_site_RMSD_max=args.active_site_RMSD_max)
        out.append({
            "protein_id": protein_id, "design_idx": int(design_idx), "muts": c.desc,
            "positions": list(c.positions), "head_proxy": hp, "core_count": count,
            "rank_margin_mass": margin, "scTM": sc_tm, "pLDDT": plddt, "passed": passed,
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
        topB=args.topB, beam_width=args.beam_width, max_rounds=args.max_rounds,
        patience=args.patience, max_path_mutations=args.max_path_mutations,
        allow_structure_unknown=args.allow_structure_unknown,
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


def _seeds_from_table(seed_table: str, proteins_arg: str) -> dict[str, list[dict]]:
    """Seeds from a self-contained protein_id/design_id/sequence parquet (bypasses
    --run-dir best-of-N selection; refines every listed row)."""
    df = pd.read_parquet(seed_table)
    for col in ("protein_id", "sequence"):
        if col not in df.columns:
            raise ValueError(f"--seed-table missing required column {col!r}")
    if proteins_arg and proteins_arg != "all":
        want = {p.strip() for p in proteins_arg.split(",")}
        df = df[df["protein_id"].astype(str).isin(want)]
    seeds_by_protein: dict[str, list[dict]] = {}
    has_seed = "seed" in df.columns
    for _, row in df.iterrows():
        pid = str(row["protein_id"])
        lst = seeds_by_protein.setdefault(pid, [])
        lst.append({
            "sequence": str(row["sequence"]),
            "design_idx": _seed_design_idx(row, len(lst)),
            "seed": int(row["seed"]) if has_seed and pd.notna(row.get("seed")) else -1,
        })
    return seeds_by_protein


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
        seeds_by_protein = _seeds_from_table(args.seed_table, args.proteins)
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
        f"beam={args.beam_width} strong_rank={args.strong_rank} margin_band={args.margin_band} "
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
    rich_rows, trace_rows, eval_rows = [], [], []
    per_protein_idx: dict[str, int] = {}
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

    (out_dir / "refined").mkdir(exist_ok=True)
    rich_df = pd.DataFrame(rich_rows).drop(columns=["orig_design_idx"], errors="ignore")
    rich_df.to_parquet(out_dir / "refined" / "refined_designs.parquet", index=False)
    pd.DataFrame(eval_rows, columns=GENERATED_COLUMNS).to_parquet(
        out_dir / "refined" / "evaluator_ready.parquet", index=False)
    pd.DataFrame(trace_rows).to_parquet(out_dir / "refined" / "refine_trace.parquet", index=False)
    _write_config(out_dir / "refined" / "refine_config.json", args)
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

    def head_fn(pid, seqs):
        compact = scorer.score_window_risk_batch_same_protein(
            protein_id=pid, records=[(str(i), s) for i, s in enumerate(seqs)])
        return np.asarray(compact.window_risks)

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

    model = load_refold_model("esmfold", device=args.head_device)
    test_df = pd.read_parquet(args.test_set_parquet)
    test_lookup = {str(r["protein_id"]): r for _, r in test_df.iterrows()}

    def struct_fn(pid, seq):
        pred = refold(seq, pid, "refine", backend="esmfold",
                      cache_dir=args.esmfold_cache_dir, model=model)
        ref = resolve_structure_path(test_lookup[pid], args.pdb_root)
        tm = run_tmalign(pred_pdb=str(pred["pdb_path"]), ref_pdb=str(ref), cache_dir=None)
        return StructureMetrics(scTM=float(tm["tm_score"]), pLDDT=float(pred["pLDDT"]),
                                scRMSD=float(tm["rmsd"]))

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
    # nmp (mirror evaluate_phase_c.py accelerated knobs)
    p.add_argument("--netmhciipan-bin", default=None, help="path to the NetMHCIIpan binary")
    p.add_argument("--nmp-batch-size", type=int, default=8)
    p.add_argument("--nmp-max-lengths-per-call", type=int, default=4)
    p.add_argument("--nmp-workers", type=int, default=8)
    p.add_argument("--nmp-timeout", type=int, default=1800)
    # structure
    p.add_argument("--pdb-root", default=None)
    p.add_argument("--esmfold-cache-dir", default=None)
    # search knobs (§7)
    p.add_argument("--strong-rank", type=float, default=0.02)
    p.add_argument("--margin-band", type=float, default=0.10)
    p.add_argument("--scTM-eps", dest="scTM_eps", type=float, default=0.05)
    p.add_argument("--scRMSD-max", dest="scRMSD_max", type=float, default=None)
    p.add_argument("--active-site-RMSD-max", dest="active_site_RMSD_max", type=float, default=None)
    p.add_argument("--topB", type=int, default=None)
    p.add_argument("--beam-width", type=int, default=8)
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--max-path-mutations", type=int, default=8,
                   help="cheap refold-free cap on total edits per search path (branch-waste bound)")
    p.add_argument("--head-high-topk", type=int, default=0,
                   help="head-high editable augmentation (v0 deferral: must be 0)")
    p.add_argument("--allow-structure-unknown", action="store_true")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--print-config", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.allele = _canonical_allele(args.allele)  # tolerate the 'HLA-DRB1_07_01' run-dir tag
    if args.print_config:
        print("[refine] config: " + json.dumps(vars(args), sort_keys=True, default=str), flush=True)
    oracles = build_oracles(args)
    return run_refinement(args, oracles)


if __name__ == "__main__":
    raise SystemExit(main())
