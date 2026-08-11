#!/usr/bin/env python3
"""RF-Refine Fusion driver (Task F8/F9): Head-guided, structure-constrained edit-and-repair.

Reads complete RF designs, builds an N-slot initial population per protein (terminal handoff,
§1.7), runs the exact complete-state loop (greedy/beam/FK), and writes the artifact set (§F9)
with per-protein atomic checkpointing and deterministic resume. Runtime immune signal is
Head-only — NetMHCIIpan / StandaloneRunner are never imported (§0.3(1)).

The orchestration (`run_fusion`) + artifact writers are oracle-agnostic and unit-tested with
fake oracles. `build_oracles` wires the real Head / configured refold backend + TMalign +
selected geometry / DPLM-repair oracles (torch; interfaces verified against source — the end-to-end run is pending
cluster validation, S0-S2, not yet executed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # namespace import of scripts.evaluate_phase_c etc.
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion import runner as rn  # noqa: E402
from inverse_folding.reference_flow.fusion.config import load_fusion_config  # noqa: E402

# Stable artifact schemas so empty tables never write a zero-column parquet.
_CANDIDATE_COLS = ["protein_id", "parent_particle_id", "proposal_id", "move_family", "sequence",
                   "sequence_md5", "edited_positions", "target_start_0b", "target_end_0b",
                   "halo_start_0b", "halo_end_0b", "head_global_risk", "local_target_delta",
                   "new_hotspot_max", "new_hotspot_mass", "new_hotspot_count", "scTM", "pLDDT",
                   "global_ca_RMSD", "active_site_RMSD", "active_site_sidechain_RMSD",
                   "max_anchor_sidechain_RMSD", "max_anchor_atom_distance",
                   "active_site_complete", "active_site_min_pLDDT", "scRMSD",
                   "structure_evaluated", "feasible", "reason",
                   "ancestry_eligible", "selection_count", "aligned_windows"]
_PARTICLE_COLS = ["protein_id", "round_idx", "slot_idx", "particle_id", "parent_particle_id",
                  "source_proposal_id", "sequence", "sequence_md5", "weight", "head_global_risk",
                  "scTM", "pLDDT", "global_ca_RMSD", "active_site_RMSD",
                  "active_site_sidechain_RMSD", "max_anchor_sidechain_RMSD",
                  "max_anchor_atom_distance", "active_site_complete", "active_site_min_pLDDT",
                  "feasible", "is_elite", "entry_source_id"]
_LINEAGE_COLS = ["protein_id", "parent_particle_id", "child_particle_id", "round_idx",
                 "proposal_id", "selector", "multiplicity"]
_ELITE_COLS = ["protein_id", "initial_sequence", "initial_head_global_risk", "initial_scTM",
               "initial_pLDDT", "initial_global_ca_RMSD", "initial_active_site_RMSD",
               "initial_active_site_sidechain_RMSD", "initial_max_anchor_sidechain_RMSD",
               "initial_max_anchor_atom_distance", "initial_active_site_complete",
               "initial_active_site_min_pLDDT",
               "final_sequence", "final_head_global_risk", "final_scTM", "final_pLDDT",
               "final_global_ca_RMSD", "final_active_site_RMSD",
               "final_active_site_sidechain_RMSD", "final_max_anchor_sidechain_RMSD",
               "final_max_anchor_atom_distance", "final_active_site_complete",
               "final_active_site_min_pLDDT",
               "improvement", "n_rounds", "elite_particle_id",
               "source_parent_particle_id", "source_proposal_id",
               # V1 entry lineage on the row the experiment REPORTS. Without it, closing the
               # facade -> elite chain needs a join back through fusion_particles, and for a
               # round-0 seed elite that join lands on the SEQUENCE -- which PLAN §2.11:531
               # forbids. Null for a standalone v0 run.
               "entry_source_id", "initial_entry_source_id"]
_FAILURE_COLS = ["protein_id", "reason", "message", "last_completed_round"]
#: Round-0 admission verdicts (PLAN_RF_REFINE_FUSION_V1 §2.11's additive audit seam). Closes the
#: `facade design_idx -> admission attempt and verdict -> initial slot / particle_id` edge, which
#: the ENTRY stage cannot close: its own gate defers, so the definitive verdict exists only here.
_ADMISSION_COLS = ["protein_id", "design_idx", "entry_source_id", "sequence_md5", "verdict",
                   "reason", "slot_idx", "particle_id", "structure_evaluated", "cache_hit",
                   "scTM", "pLDDT"]


# --------------------------------------------------------------------------- #
# artifact row extraction (pure — JSON-serializable)
# --------------------------------------------------------------------------- #
def _sm(structure, field):
    return None if structure is None else getattr(structure, field, None)


def candidate_rows(result, selection_counts=None):
    props = result.proposals
    selection_counts = selection_counts or {}
    rows = []
    for e in result.candidates:
        p = props.get(e.proposal_id)
        rows.append({
            "protein_id": result.protein_id, "parent_particle_id": e.parent_particle_id,
            "proposal_id": e.proposal_id, "move_family": getattr(p, "move_family", None),
            "sequence": getattr(p, "sequence", None), "sequence_md5": e.sequence_md5,
            "edited_positions": list(getattr(p, "edited_positions", ()) or ()),
            "target_start_0b": e.target_start_0b, "target_end_0b": e.target_end_0b,
            "halo_start_0b": getattr(p, "halo_start_0b", None),
            "halo_end_0b": getattr(p, "halo_end_0b", None),
            "head_global_risk": e.head_global_risk, "local_target_delta": e.local_target_delta,
            "new_hotspot_max": e.new_hotspot_max, "new_hotspot_mass": e.new_hotspot_mass,
            "new_hotspot_count": e.new_hotspot_count,
            "scTM": _sm(e.structure, "scTM"), "pLDDT": _sm(e.structure, "pLDDT"),
            "global_ca_RMSD": _sm(e.structure, "global_ca_RMSD"),
            "active_site_RMSD": _sm(e.structure, "active_site_RMSD"),
            "active_site_sidechain_RMSD": _sm(e.structure, "active_site_sidechain_RMSD"),
            "max_anchor_sidechain_RMSD": _sm(e.structure, "max_anchor_sidechain_RMSD"),
            "max_anchor_atom_distance": _sm(e.structure, "max_anchor_atom_distance"),
            "active_site_complete": _sm(e.structure, "active_site_complete"),
            "active_site_min_pLDDT": _sm(e.structure, "active_site_min_pLDDT"),
            "scRMSD": _sm(e.structure, "scRMSD"),
            "structure_evaluated": e.structure_evaluated, "feasible": e.feasible, "reason": e.reason,
            "ancestry_eligible": e.feasible,
            "selection_count": selection_counts.get(e.proposal_id, 0),
            "aligned_windows": [list(w) for w in (e.aligned_windows or ())],
        })
    return rows


def admission_verdict_rows(result):
    """One row per facade row the round-0 scan examined -- admitted or rejected, with the reason.

    A bare ``continue`` used to drop rejected rows, so the rank-vs-feasibility evidence for §2.11's
    "on structure failure, advance to the next ranked root" was unreconstructible, and an anchor
    violation on a completed sequence (which ONLY v0 detects) left no trace at all.
    """
    return [
        {"protein_id": result.protein_id, **attempt}
        for attempt in result.initial_population.admission_attempts
    ]


def particle_rows(result):
    rows = []
    for pop in result.populations:
        elite_md5 = pop.elite.particle.sequence_md5
        for p in pop.particles:
            rows.append({
                "protein_id": result.protein_id, "round_idx": pop.round_idx, "slot_idx": p.slot_idx,
                "particle_id": p.particle_id, "parent_particle_id": p.parent_particle_id,
                "source_proposal_id": p.source_proposal_id, "sequence": p.sequence,
                "sequence_md5": p.sequence_md5, "weight": p.weight,
                "head_global_risk": p.head_global_risk,
                "scTM": _sm(p.structure, "scTM"), "pLDDT": _sm(p.structure, "pLDDT"),
                "global_ca_RMSD": _sm(p.structure, "global_ca_RMSD"),
                "active_site_RMSD": _sm(p.structure, "active_site_RMSD"),
                "active_site_sidechain_RMSD": _sm(p.structure, "active_site_sidechain_RMSD"),
                "max_anchor_sidechain_RMSD": _sm(p.structure, "max_anchor_sidechain_RMSD"),
                "max_anchor_atom_distance": _sm(p.structure, "max_anchor_atom_distance"),
                "active_site_complete": _sm(p.structure, "active_site_complete"),
                "active_site_min_pLDDT": _sm(p.structure, "active_site_min_pLDDT"),
                "feasible": p.feasible, "is_elite": p.sequence_md5 == elite_md5,
                # V1 entry lineage: which facade row this particle's round-0 ancestor came from.
                # Null for a standalone v0 run. Persisting it is the point -- an in-memory-only key
                # is the same as no key when the reviewer reads parquet (§2.11:531).
                "entry_source_id": p.entry_source_id,
            })
    return rows


def round_rows(result):
    out = []
    for rec in result.rounds:
        out.append({"protein_id": result.protein_id, "round_idx": rec.round_idx,
                    "n_parents": rec.n_parents, "n_proposals": rec.n_proposals,
                    "n_refolds": rec.n_refolds, "n_feasible_children": rec.n_feasible_children,
                    "n_selected_children": rec.n_selected_children, "elite_risk": rec.elite_risk,
                    "ess": rec.ess, "resampled": rec.resampled,
                    "cache_hits": rec.cache_hits, "wall_time_s": rec.wall_time_s})
    return out


def elite_row(result):
    e = result.elite
    init = result.initial_population.elite.particle
    return {"protein_id": result.protein_id,
            "entry_source_id": e.entry_source_id,
            "initial_entry_source_id": init.entry_source_id,
            "initial_sequence": init.sequence, "initial_head_global_risk": init.head_global_risk,
            "initial_scTM": _sm(init.structure, "scTM"),
            "initial_pLDDT": _sm(init.structure, "pLDDT"),
            "initial_global_ca_RMSD": _sm(init.structure, "global_ca_RMSD"),
            "initial_active_site_RMSD": _sm(init.structure, "active_site_RMSD"),
            "initial_active_site_sidechain_RMSD": _sm(init.structure, "active_site_sidechain_RMSD"),
            "initial_max_anchor_sidechain_RMSD": _sm(init.structure, "max_anchor_sidechain_RMSD"),
            "initial_max_anchor_atom_distance": _sm(init.structure, "max_anchor_atom_distance"),
            "initial_active_site_complete": _sm(init.structure, "active_site_complete"),
            "initial_active_site_min_pLDDT": _sm(init.structure, "active_site_min_pLDDT"),
            "final_sequence": e.sequence, "final_head_global_risk": e.head_global_risk,
            "final_scTM": _sm(e.structure, "scTM"),
            "final_pLDDT": _sm(e.structure, "pLDDT"),
            "final_global_ca_RMSD": _sm(e.structure, "global_ca_RMSD"),
            "final_active_site_RMSD": _sm(e.structure, "active_site_RMSD"),
            "final_active_site_sidechain_RMSD": _sm(e.structure, "active_site_sidechain_RMSD"),
            "final_max_anchor_sidechain_RMSD": _sm(e.structure, "max_anchor_sidechain_RMSD"),
            "final_max_anchor_atom_distance": _sm(e.structure, "max_anchor_atom_distance"),
            "final_active_site_complete": _sm(e.structure, "active_site_complete"),
            "final_active_site_min_pLDDT": _sm(e.structure, "active_site_min_pLDDT"),
            "improvement": init.head_global_risk - e.head_global_risk, "n_rounds": len(result.rounds),
            # provenance: the elite id is the distinct ARCHIVE scheme; trace via source parent/proposal
            "elite_particle_id": e.particle_id, "source_parent_particle_id": e.parent_particle_id,
            "source_proposal_id": e.source_proposal_id}


def generated_row(result):  # evaluator-compatible elite output (protein_id, design_idx, sequence)
    return {"protein_id": result.protein_id, "design_idx": 0, "sequence": result.elite.sequence}


def protein_payload(result):
    """A JSON-serializable per-protein checkpoint payload with every artifact's rows."""
    sel_counts: dict = {}
    for edge in result.lineage:
        sel_counts[edge["proposal_id"]] = sel_counts.get(edge["proposal_id"], 0) + 1
    return {
        "protein_id": result.protein_id, "status": "ok",
        "candidates": candidate_rows(result, sel_counts), "particles": particle_rows(result),
        "lineage": [{"protein_id": result.protein_id, **edge} for edge in result.lineage],
        "rounds": round_rows(result), "elite": elite_row(result), "generated": generated_row(result),
        # §2.11 audit seam: what round-0 admission examined, admitted or rejected, and what it spent
        "initial_admission": admission_verdict_rows(result),
        "initial_refolds": int(result.initial_population.initial_refolds),
    }


def failure_payload(protein_id, exc, last_round=None):
    return {"protein_id": protein_id, "status": "failed",
            "failure": {"protein_id": protein_id, "reason": type(exc).__name__,
                        "message": str(exc), "last_completed_round": last_round}}


# --------------------------------------------------------------------------- #
# atomic IO
# --------------------------------------------------------------------------- #
def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=list))
    os.replace(tmp, path)  # atomic on POSIX


# --------------------------------------------------------------------------- #
# orchestration (oracle-agnostic — unit-tested with fakes)
# --------------------------------------------------------------------------- #
def _path_fingerprint(path) -> str:
    """Cheap change-detecting fingerprint for a possibly-large input file: path + size + mtime
    (hashing multi-GB checkpoints every run is impractical; size+mtime catches real edits)."""
    if not path:
        return "-"
    p = Path(path)
    if not p.exists():
        return f"{path}:missing"
    s = p.stat()
    return f"{path}:{s.st_size}:{s.st_mtime_ns}"


def _input_signature(args) -> str:
    """Signature of all external inputs that must invalidate resume when changed (P1-4): Head
    checkpoint/config/variant, constraint manifest, base-IF + sampler config, allele, and the
    target-backbone source. Combined with the per-protein seed digest in ``_protein_run_sig``."""
    parts = [
        _path_fingerprint(getattr(args, "head_checkpoint", None)),
        _path_fingerprint(getattr(args, "head_config_dir", None)),
        str(getattr(args, "head_variant_id", "")),
        str(getattr(args, "head_allele_idx", "")),
        # window range changes Head scores -> must invalidate resume (same bug class as P1-4)
        str(getattr(args, "window_k_min", "")),
        str(getattr(args, "window_k_max", "")),
        str(getattr(args, "allele", "")),
        _path_fingerprint(getattr(args, "constraint_manifest", None)),
        _path_fingerprint(getattr(args, "base_if_checkpoint", None)),
        _path_fingerprint(getattr(args, "rf_sampler_config", None)),
        _path_fingerprint(getattr(args, "test_set_parquet", None)),
        str(getattr(args, "pdb_root", "")),
        str(getattr(args, "_refold_backend", "")),
        _path_fingerprint(getattr(args, "esmfold2_site_packages", None)),
        str(getattr(args, "esmfold2_model", "")),
        str(getattr(args, "esmfold2_num_loops", "")),
        str(getattr(args, "esmfold2_num_sampling_steps", "")),
        str(getattr(args, "esmfold2_num_diffusion_samples", "")),
        str(getattr(args, "esmfold2_seed", "")),
    ]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _provenance(args) -> dict:
    """Full input provenance for the manifest so an S0/S2 run can be reconstructed / audited:
    a fingerprint per external input plus the combined input signature (P1-7)."""
    return {
        "input_signature": _input_signature(args),
        "head_checkpoint": _path_fingerprint(getattr(args, "head_checkpoint", None)),
        "head_config_dir": _path_fingerprint(getattr(args, "head_config_dir", None)),
        "head_variant_id": str(getattr(args, "head_variant_id", "")),
        "head_allele_idx": getattr(args, "head_allele_idx", None),
        "window_k_min": getattr(args, "window_k_min", None),
        "window_k_max": getattr(args, "window_k_max", None),
        "constraint_manifest": _path_fingerprint(getattr(args, "constraint_manifest", None)),
        "base_if_checkpoint": _path_fingerprint(getattr(args, "base_if_checkpoint", None)),
        "rf_sampler_config": _path_fingerprint(getattr(args, "rf_sampler_config", None)),
        "test_set_parquet": _path_fingerprint(getattr(args, "test_set_parquet", None)),
        "pdb_root": str(getattr(args, "pdb_root", "")),
        "refold_backend": str(getattr(args, "_refold_backend", "esmfold")),
        "refold_cache_dir": str(getattr(args, "refold_cache_dir", "")),
        "esmfold2_site_packages": _path_fingerprint(
            getattr(args, "esmfold2_site_packages", None)
        ),
        "esmfold2_model": str(getattr(args, "esmfold2_model", "")),
        "esmfold2_num_loops": getattr(args, "esmfold2_num_loops", None),
        "esmfold2_num_sampling_steps": getattr(args, "esmfold2_num_sampling_steps", None),
        "esmfold2_num_diffusion_samples": getattr(
            args, "esmfold2_num_diffusion_samples", None
        ),
        "esmfold2_seed": getattr(args, "esmfold2_seed", None),
        "refold_runtime": getattr(args, "_refold_runtime", None),
    }


def _seed_digest(rows) -> str:
    """Content digest of a protein's seed designs (order-independent), so a changed input
    sequence set invalidates its resume checkpoint."""
    # `entry_source_id` is part of the key: PLAN_RF_REFINE_FUSION_V1 §2.11 warns that distinct roots
    # may converge to the same complete sequence, so two entry runs with identical ranked sequences
    # but different roots would otherwise share this digest. v0 would reuse the earlier checkpoint
    # and every lineage column would then name the wrong root -- the sequence-derived attribution
    # §2.11 forbids, moved from the join into the cache key. Empty for a standalone v0 run, so the
    # legacy key is unchanged.
    items = sorted((str(r.get("design_idx")), str(r.get("seed", "")), str(r["sequence"]),
                    str(r.get("entry_source_id", ""))) for r in rows)
    return hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()


def _protein_run_sig(config, input_signature: str, seed_rows) -> str:
    """Per-protein resume identity: Fusion config + all external inputs (Head/manifest/base-IF/
    sampler via ``input_signature``) + this protein's seed sequences. Reusing a checkpoint whose
    inputs changed would silently return stale results (P1-4)."""
    h = hashlib.sha256()
    h.update(config.config_hash().encode("utf-8"))
    h.update(b"\x00"); h.update(str(input_signature).encode("utf-8"))
    h.update(b"\x00"); h.update(_seed_digest(seed_rows).encode("utf-8"))
    return h.hexdigest()


def run_fusion(*, proteins, seed_rows_by_protein, oracles, config, out_dir,
               structure_cache_factory, anchors_by_protein=None, anchor_expected_by_protein=None,
               expected_length_by_protein=None, repair_fn=None, repair_fn_factory=None,
               resume=True, manifest_extra=None, input_signature=""):
    """Run Fusion for each protein with per-protein atomic checkpoint + resume, then aggregate
    the checkpoints into the final artifact set + manifest. Returns the manifest dict.

    Resume identity is per-protein: ``run_sig`` binds the Fusion config, the external-input
    signature (Head/manifest/base-IF/sampler), and the protein's seed sequences, so any changed
    input re-runs rather than reusing a stale checkpoint. ``config_hash`` still scopes the
    aggregate + manifest to the current config."""
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    anchors_by_protein = anchors_by_protein or {}
    anchor_expected_by_protein = anchor_expected_by_protein or {}
    cfg_hash = config.config_hash()  # aggregate/manifest are scoped to THIS config

    for pid in proteins:
        ckpt = ckpt_dir / f"{pid}.json"
        run_sig = _protein_run_sig(config, input_signature, seed_rows_by_protein[pid])
        if resume and ckpt.exists():
            prev = json.loads(ckpt.read_text())
            # reuse ONLY a successful checkpoint whose FULL input identity is unchanged
            if prev.get("status") == "ok" and prev.get("run_sig") == run_sig:
                continue
        try:
            cache = structure_cache_factory(pid)
            pop = rn.build_initial_population(
                seed_rows_by_protein[pid], protein_id=pid, oracles=oracles, structure_cache=cache,
                config=config, anchors=anchors_by_protein.get(pid, set()),
                anchor_expected=anchor_expected_by_protein.get(pid),
                expected_length=(expected_length_by_protein or {}).get(pid))
            rf = repair_fn_factory(pid) if repair_fn_factory is not None else repair_fn
            result = rn.run_protein(
                protein_id=pid, initial_population=pop, oracles=oracles, structure_cache=cache,
                config=config, anchors=anchors_by_protein.get(pid, set()), repair_fn=rf)
            payload = protein_payload(result)
        except Exception as exc:  # noqa: BLE001 - a failed protein must not abort the run
            payload = failure_payload(pid, exc)
        payload["config_hash"] = cfg_hash    # scopes aggregate/manifest to this config
        payload["run_sig"] = run_sig          # full per-protein input identity (resume gate)
        _write_json_atomic(ckpt, payload)

    return _aggregate_artifacts(out_dir, ckpt_dir, config, manifest_extra, cfg_hash, proteins)


def _aggregate_artifacts(out_dir: Path, ckpt_dir: Path, config, manifest_extra, cfg_hash, proteins):
    want = {f"{p}.json" for p in proteins}
    payloads = [json.loads((ckpt_dir / f).read_text())
                for f in sorted(os.listdir(ckpt_dir))
                if f.endswith(".json") and f in want]
    payloads = [p for p in payloads if p.get("config_hash") == cfg_hash]  # ignore stale-config JSON
    ok = [p for p in payloads if p.get("status") == "ok"]
    failed = [p["failure"] for p in payloads if p.get("status") == "failed"]

    def _df(rows, columns):  # stable schema even when a table is empty (no zero-column parquet)
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)

    def _rows(key):
        return [row for p in ok for row in p.get(key, [])]

    _df(_rows("candidates"), _CANDIDATE_COLS).to_parquet(out_dir / "fusion_candidates.parquet", index=False)
    _df(_rows("particles"), _PARTICLE_COLS).to_parquet(out_dir / "fusion_particles.parquet", index=False)
    _df(_rows("lineage"), _LINEAGE_COLS).to_parquet(out_dir / "fusion_lineage.parquet", index=False)
    _df(_rows("initial_admission"), _ADMISSION_COLS).to_parquet(
        out_dir / "fusion_initial_admission_verdicts.parquet", index=False)
    _df([p["elite"] for p in ok], _ELITE_COLS).to_parquet(out_dir / "fusion_elite.parquet", index=False)
    _df([p["generated"] for p in ok], ["protein_id", "design_idx", "sequence"]).to_parquet(
        out_dir / "generated.parquet", index=False)
    _df(failed, _FAILURE_COLS).to_parquet(out_dir / "fusion_failures.parquet", index=False)
    with open(out_dir / "fusion_rounds.jsonl", "w") as fh:
        for p in ok:
            for rec in p.get("rounds", []):
                fh.write(json.dumps(rec) + "\n")

    manifest = {
        "config_hash": cfg_hash, "config": config.to_canonical_dict(),
        "nmp_absent": True, "git_sha": _git_sha(),
        "n_proteins": len(proteins), "n_proteins_ok": len(ok), "n_proteins_failed": len(failed),
        "proteins": list(proteins),
        "artifacts": ["fusion_candidates.parquet", "fusion_particles.parquet",
                      "fusion_lineage.parquet", "fusion_elite.parquet", "generated.parquet",
                      "fusion_failures.parquet", "fusion_rounds.jsonl",
                      "fusion_initial_admission_verdicts.parquet"],
        # Definitive round-0 folds. The ENTRY ledger books 0 structure requests for a deferred
        # attempt on the grounds that v0 charges it; v0 counted refolds only from round 1, so this
        # spend was charged in no artifact at all and a matched-compute check on refolds compared
        # two zeros.
        "initial_refolds_by_protein": {
            p["protein_id"]: int(p.get("initial_refolds", 0)) for p in ok
        },
        **(manifest_extra or {}),
    }
    _write_json_atomic(out_dir / "manifest.json", manifest)
    return manifest


def _git_sha():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=str(PROJECT_ROOT), timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# real oracle wiring (torch; interfaces verified against source — live run pending S0-S2)
# --------------------------------------------------------------------------- #
def _require(path, what):
    if not path:
        raise SystemExit(f"missing required input: {what}")
    return path


def build_oracles(args, config):
    """Wire the real Head / structure / DPLM-repair oracles. NetMHCIIpan is never imported."""
    from inverse_folding.evaluation.refold import load_refold_model, refold
    from inverse_folding.evaluation.tmalign import run_tmalign
    from inverse_folding.reference_flow.constraints import load_constraint_manifest
    from inverse_folding.reference_flow.runtime import resolve_structure_path

    _require(args.head_checkpoint, "--head-checkpoint")
    _require(args.refold_cache_dir, "--refold-cache-dir")
    _require(args.test_set_parquet, "--test-set-parquet")
    _require(args.pdb_root, "--pdb-root")

    # Head: OnlineHeadScorer -> head_fn(protein_id, sequences) -> [HeadScore]
    from scripts.run_if_phase_c1 import build_head_scorer  # namespace import
    scorer = _build_head_scorer_from_args(args, build_head_scorer)

    def head_fn(protein_id, sequences):
        records = [(f"cand{i}", s) for i, s in enumerate(sequences)]
        batch = scorer.score_batch_same_protein(protein_id=protein_id, records=records)
        return list(batch.scores)

    # Structure: TMalign provides scTM; active-site geometry is selected explicitly by config.
    backend = config.structure.backend
    backend_options = {}
    if backend == "esmfold2_live":
        backend_options = {
            "site_packages": _require(
                args.esmfold2_site_packages, "--esmfold2-site-packages"
            ),
            "model_name": args.esmfold2_model,
            "esmc_model": args.esmfold2_esmc_model,
            "ccd_path": args.esmfold2_ccd_path,
            "num_loops": args.esmfold2_num_loops,
            "num_sampling_steps": args.esmfold2_num_sampling_steps,
            "num_diffusion_samples": args.esmfold2_num_diffusion_samples,
            "seed": args.esmfold2_seed,
        }
    model = load_refold_model(backend, device=args.head_device, **backend_options)
    args._refold_backend = backend
    args._refold_runtime = getattr(model, "metadata", None)
    test_df = pd.read_parquet(args.test_set_parquet)
    test_lookup = {str(r["protein_id"]): r for _, r in test_df.iterrows()}
    manifest = load_constraint_manifest(args.constraint_manifest) if args.constraint_manifest else None
    ref_path_cache: dict[str, Path] = {}

    def _reference_path(pid: str) -> Path:
        if pid not in test_lookup:
            raise KeyError(f"protein {pid!r} is absent from --test-set-parquet")
        if pid not in ref_path_cache:
            ref_path_cache[pid] = Path(resolve_structure_path(test_lookup[pid], args.pdb_root))
        return ref_path_cache[pid]

    shell_indices_fn = None
    compute_ca_self_consistency_fn = None
    reference_context_fn = None
    evaluate_prediction_fn = None
    v2_metric_names = None
    if config.structure.active_site_metric == "legacy_ca_shell":
        import numpy as np

        from inverse_folding.evaluation.sc_rmsd import (
            compute_ca_self_consistency as compute_ca_self_consistency_fn,
            parse_ca_trace,
        )
        from inverse_folding.reference_flow.fusion import structure_metrics as sm

        shell_cache: dict[str, frozenset[int]] = {}

        def shell_indices_fn(pid, ref_pdb, anchor_idx):
            if pid not in shell_cache:
                ca = parse_ca_trace(ref_pdb)
                coords = np.vstack([res.coord for res in ca]).astype(float)
                shell_cache[pid] = sm.shell_indices_from_ca(
                    coords,
                    anchor_idx,
                    config.structure.active_site_shell_radius,
                )
            return shell_cache[pid]
    else:
        from inverse_folding.evaluation.structural_metrics_v2 import (
            METRIC_GLOBAL_CA_RMSD,
            METRIC_PLDDT,
            METRIC_SIDECHAIN,
            evaluate_prediction as evaluate_prediction_fn,
            prepare_reference_context,
        )

        reference_context_cache: dict[str, object] = {}
        v2_metric_names = {
            METRIC_GLOBAL_CA_RMSD,
            METRIC_PLDDT,
            METRIC_SIDECHAIN,
        }

        def reference_context_fn(pid: str):
            if pid not in reference_context_cache:
                ref_sequence = str(test_lookup[pid]["sequence"])
                if manifest is not None and manifest.has_protein(pid):
                    manifest.constraint_for_protein(pid).validate_against_sequence(ref_sequence)
                reference_context_cache[pid] = prepare_reference_context(
                    _reference_path(pid),
                    ref_sequence=ref_sequence,
                    anchor_indices=_anchor_indices(manifest, pid),
                )
            return reference_context_cache[pid]

    from inverse_folding.reference_flow.refine import StructureMetrics

    def struct_fn(protein_id, sequence):
        pred = refold(sequence, protein_id, "fusion", backend=backend,
                      cache_dir=args.refold_cache_dir, model=model)
        cache_hit = bool(pred.get("cache_hit", False))
        ref = _reference_path(protein_id)
        tm = run_tmalign(pred_pdb=str(pred["pdb_path"]), ref_pdb=str(ref), cache_dir=None)
        asr = None
        v2 = None
        anchor_idx = _anchor_indices(manifest, protein_id)
        if config.structure.active_site_metric == "legacy_ca_shell":
            if anchor_idx:
                shell = shell_indices_fn(protein_id, str(ref), anchor_idx)
                _scr, per_res = compute_ca_self_consistency_fn(
                    pred_pdb=str(pred["pdb_path"]), ref_pdb=str(ref), protein_id=protein_id,
                    design_id="fusion", design_idx=0, design_sequence=sequence,
                    ref_sequence=str(test_lookup[protein_id]["sequence"]),
                    refold_backend=backend)
                asr = sm.shell_rmsd_from_rows(per_res, shell)
        else:
            v2 = evaluate_prediction_fn(
                reference_context_fn(protein_id),
                str(pred["pdb_path"]),
                design_sequence=sequence,
                metrics=v2_metric_names,
            )
        plddt = (
            v2.predicted_global_plddt
            if v2 is not None and v2.predicted_global_plddt is not None
            else float(pred["pLDDT"])
        )
        return StructureMetrics(
            scTM=float(tm["tm_score"]),
            pLDDT=float(plddt),
            scRMSD=float(tm["rmsd"]),
            active_site_RMSD=asr,
            global_ca_RMSD=None if v2 is None else v2.global_ca_rmsd,
            active_site_sidechain_RMSD=(
                None if v2 is None else v2.active_site_sidechain_rmsd
            ),
            max_anchor_sidechain_RMSD=(
                None if v2 is None else v2.max_anchor_sidechain_rmsd
            ),
            max_anchor_atom_distance=(
                None if v2 is None else v2.max_anchor_atom_distance
            ),
            active_site_complete=None if v2 is None else v2.active_site_complete,
            active_site_min_pLDDT=(
                None if v2 is None else v2.predicted_active_site_min_plddt
            ),
            cache_hit=cache_hit,
            model_executed=not cache_hit,
        )

    from inverse_folding.reference_flow.fusion.oracles import FusionOracles
    return FusionOracles(head_fn=head_fn, struct_fn=struct_fn), manifest


def _anchor_indices(manifest, protein_id):
    # ConstraintManifest has NO .get(): use has_protein()/constraint_for_protein() (verified).
    if manifest is None or not manifest.has_protein(protein_id):
        return set()
    return set(manifest.constraint_for_protein(protein_id).hard_anchor_indices)


def _build_head_scorer_from_args(args, build_head_scorer):
    from pathlib import Path as _P
    from types import SimpleNamespace
    setup = SimpleNamespace(
        head_config_dir=_P(args.head_config_dir), head_checkpoint=_P(args.head_checkpoint),
        head_variant_id=args.head_variant_id, head_device=args.head_device,
        head_allele_idx=args.head_allele_idx, head_window_batch_size=args.head_window_batch_size,
        allele=args.allele, config=SimpleNamespace(head=SimpleNamespace(score_scale="raw_logit")))
    return build_head_scorer(setup, window_k_min=args.window_k_min, window_k_max=args.window_k_max)


# --------------------------------------------------------------------------- #
# repair oracle (DPLM sampler-backed) — the H3 arm (rf_reopen / edit-repair)
# --------------------------------------------------------------------------- #
def make_sampler_repair_fn(*, aa_to_token, decode_tokens, denoiser, sampler, rf_config,
                           sequence_length, sampler_steps, local_remask):
    """Wrap the DPLM sampler as a ``moves.repair_fn(base_seq, frozen, regen, seed) -> new_seq``.

    Frozen positions become ``fixed_tokens`` (seeded, never re-sampled, remask-protected); every
    other position (== the regen halo) is regenerated. ``controller=None`` + a backbone-only
    context; per-child diversity comes from cloning the config with a distinct ``sampler.seed``.

    The Fusion config's declared repair kernel OVERRIDES the base rf-sampler-config: ``sampler_steps``
    -> ``sampler.n_steps`` and ``local_remask`` -> ``sampler.remask.enabled``. This makes the H3
    experiment label match the executed sampler (single source of truth = the Fusion config).
    Injected callables keep this unit-testable without a real checkpoint.
    """
    import numpy as np
    from dataclasses import replace as _replace
    h = np.ones(int(sequence_length), dtype=np.float32)  # flat h -> controller-free unmask ordering
    remask = _replace(rf_config.sampler.remask, enabled=bool(local_remask))

    def repair_fn(base_seq, frozen, regen, seed):
        del base_seq, regen  # regen is implicit: every non-fixed position is regenerated
        fixed_tokens = {int(pos): int(aa_to_token(aa)) for pos, aa in frozen.items()}
        cfg = _replace(rf_config, sampler=_replace(
            rf_config.sampler, seed=int(seed), n_steps=int(sampler_steps), remask=remask))
        out = sampler.sample(sequence_length=int(sequence_length), h_values=h, denoiser=denoiser,
                             config=cfg, struct=None, controller=None, fixed_tokens=fixed_tokens)
        return decode_tokens(out.tokens)

    return repair_fn


def build_repair_fn_factory(args, config, test_lookup, device):
    """Per-protein sampler-backed repair_fn factory (torch). The interfaces are verified against
    source and the wrapper logic is unit-tested with a fake sampler, but the real DPLM sampler run
    is NOT yet executed — it is pending cluster validation (S0-S2).

    Requires ``--base-if-checkpoint`` + ``--rf-sampler-config``. Builds one backbone-only denoiser
    per protein and wraps it via ``make_sampler_repair_fn``.
    """
    _require(args.base_if_checkpoint, "--base-if-checkpoint (needed when repair/reopen enabled)")
    _require(args.rf_sampler_config, "--rf-sampler-config (needed when repair/reopen enabled)")
    from inverse_folding.reference_flow.config import load_reference_flow_config
    from inverse_folding.reference_flow.runtime import (
        build_dplm_denoiser_context, decode_residue_tokens, load_if_task, make_dplm_denoiser,
        prepare_backbone)
    from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler

    task = load_if_task(args.base_if_checkpoint, device=device)
    rf_config = load_reference_flow_config(args.rf_sampler_config)
    sampler = PositionDependentDFMSampler(mask_token_id=task.alphabet.mask_idx,
                                          vocab_size=len(task.alphabet))

    def factory(protein_id):
        prepared = prepare_backbone(task=task, entry=test_lookup[protein_id],
                                    pdb_root=args.pdb_root, device=device)
        ctx = build_dplm_denoiser_context(task=task, prepared=prepared, use_draft_seq_override=False)
        denoiser = make_dplm_denoiser(ctx)
        return make_sampler_repair_fn(
            aa_to_token=task.alphabet.get_idx,
            decode_tokens=lambda toks: decode_residue_tokens(task, toks),
            denoiser=denoiser, sampler=sampler, rf_config=rf_config,
            sequence_length=prepared.sequence_length,
            sampler_steps=config.moves.repair.sampler_steps,
            local_remask=config.moves.repair.local_remask)

    return factory


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(description="RF-Refine Fusion driver (Head-only, structure-gated).")
    p.add_argument("--fusion-config", required=True)
    p.add_argument("--run-dir", required=True, help="directory with the generated designs parquet")
    p.add_argument("--generated-parquet", default=None, help="explicit generated.parquet path")
    p.add_argument("--proteins", nargs="*", default=None, help="protein filter (default: all)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--allele", required=True)
    p.add_argument("--test-set-parquet", default=None)
    p.add_argument("--pdb-root", default=None)
    p.add_argument(
        "--refold-cache-dir",
        "--esmfold-cache-dir",
        dest="refold_cache_dir",
        default=None,
        help="shared structure cache; --esmfold-cache-dir is a deprecated alias",
    )
    p.add_argument(
        "--esmfold2-site-packages",
        default=None,
        help="Biohub esm/transformers site-packages overlay for backend=esmfold2_live",
    )
    p.add_argument("--esmfold2-model", default="biohub/ESMFold2")
    p.add_argument("--esmfold2-esmc-model", default=None)
    p.add_argument("--esmfold2-ccd-path", default=None)
    p.add_argument("--esmfold2-num-loops", type=int, default=3)
    p.add_argument("--esmfold2-num-sampling-steps", type=int, default=50)
    p.add_argument("--esmfold2-num-diffusion-samples", type=int, default=1)
    p.add_argument("--esmfold2-seed", type=int, default=0)
    p.add_argument("--constraint-manifest", default=None)
    # required only when rf_reopen / edit-repair is enabled (the H3 arm)
    p.add_argument("--base-if-checkpoint", default=None, help="base IF checkpoint for the DPLM repair sampler")
    p.add_argument("--rf-sampler-config", default=None, help="ReferenceFlow YAML for the repair sampler")
    p.add_argument("--head-checkpoint", default=None)
    p.add_argument("--head-config-dir", default=None)
    p.add_argument("--head-variant-id", default=None)
    p.add_argument("--head-allele-idx", type=int, default=0)
    p.add_argument("--head-window-batch-size", type=int, default=64)
    p.add_argument("--head-device", default="cuda")
    p.add_argument("--window-k-min", type=int, default=12)
    p.add_argument("--window-k-max", type=int, default=25)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--print-config", action="store_true")
    return p


def _load_generated(args):
    """Load generated designs: an explicit parquet, a clean run_dir/generated.parquet, OR the
    sharded return-package layout run_dir/generation/*shard*/generated.parquet (concatenated)."""
    import glob
    from scripts.evaluate_phase_c import load_generated_designs
    if args.generated_parquet:
        return load_generated_designs(args.generated_parquet)
    run_dir = Path(args.run_dir)
    clean = run_dir / "generated.parquet"
    if clean.is_file():
        return load_generated_designs(str(clean))
    for pattern in (str(run_dir / "generation" / "*shard*" / "generated.parquet"),
                    str(run_dir / "*shard*" / "generated.parquet")):
        hits = sorted(glob.glob(pattern))
        if hits:
            gen = pd.concat([pd.read_parquet(h) for h in hits], ignore_index=True)
            missing = {"protein_id", "design_idx", "sequence"} - set(gen.columns)
            if missing:
                raise SystemExit(f"generated parquet missing columns: {sorted(missing)}")
            gen["protein_id"] = gen["protein_id"].astype(str)
            gen["design_idx"] = gen["design_idx"].astype(int)
            gen["sequence"] = gen["sequence"].astype(str).str.upper()
            return gen
    raise SystemExit(f"no generated.parquet under {run_dir} (clean or */shard*/ layout)")


def _seed_rows_by_protein(generated_df, proteins):
    rows_by = {}
    has_entry_lineage = "entry_source_id" in generated_df.columns
    for pid in proteins:
        sub = generated_df[generated_df["protein_id"] == pid]
        rows = []
        for _, r in sub.iterrows():
            row = {"design_idx": int(r["design_idx"]), "seed": int(r.get("seed", 0)),
                   "sequence": str(r["sequence"])}
            # V1 handoff only: the entry facade row id, carried so the entry->v0 edge is a PERSISTED
            # key rather than one re-derived from the sequence (PLAN_RF_REFINE_FUSION_V1 §2.11:531).
            # A plain v0 generated parquet has no such column and the field stays absent.
            if has_entry_lineage and pd.notna(r.get("entry_source_id")):
                row["entry_source_id"] = str(r["entry_source_id"])
            rows.append(row)
        rows_by[pid] = rows
    return rows_by


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    config = load_fusion_config(args.fusion_config)
    if args.print_config:
        print("[fusion] resolved config:")
        for k, v in sorted(_flatten(config.to_canonical_dict()).items()):
            print(f"  {k} = {v}")
    if not config.core.enabled:
        raise SystemExit("fusion.enabled is false; nothing to run")
    generated = _load_generated(args)
    present = set(generated["protein_id"].astype(str))
    proteins = args.proteins
    if not proteins or "all" in proteins:  # launcher passes the literal 'all' default -> means all
        proteins = sorted(present)
    proteins = [p for p in proteins if p in present]
    if not proteins:
        raise SystemExit("no matching proteins in the generated designs")
    seed_rows = _seed_rows_by_protein(generated, proteins)
    oracles, manifest = build_oracles(args, config)
    anchors_by = {pid: _anchor_indices(manifest, pid) for pid in proteins}
    anchor_expected_by = {pid: _anchor_expected(manifest, pid) for pid in proteins}

    # expected backbone length per protein (target-backbone handoff validation, §1.7)
    expected_len_by = {}
    test_lookup = {}
    if args.test_set_parquet:
        test_df = pd.read_parquet(args.test_set_parquet)
        test_lookup = {str(r["protein_id"]): r for _, r in test_df.iterrows()}
        expected_len_by = {pid: len(str(row["sequence"])) for pid, row in test_lookup.items()}

    # anchor firewall: fail-fast if the manifest's hard anchors do not match the reference/WT
    # sequence (bounds + expected_aa) BEFORE any generation work (P1-6).
    _validate_manifest_against_references(manifest, proteins, test_lookup)

    repair_factory = None
    if config.moves.repair.enabled or config.moves.rf_reopen.enabled:
        repair_factory = build_repair_fn_factory(args, config, test_lookup, args.head_device)

    from inverse_folding.reference_flow.fusion.oracles import StructureCache

    def cache_factory(pid):
        return StructureCache(backbone_digest=pid, backend=config.structure.backend,
                              config_hash=config.config_hash())

    manifest_out = run_fusion(
        proteins=proteins, seed_rows_by_protein=seed_rows, oracles=oracles, config=config,
        out_dir=args.out_dir, structure_cache_factory=cache_factory, anchors_by_protein=anchors_by,
        anchor_expected_by_protein=anchor_expected_by, expected_length_by_protein=expected_len_by,
        repair_fn_factory=repair_factory, resume=not args.no_resume,
        input_signature=_input_signature(args),
        manifest_extra={"allele": args.allele, "run_dir": args.run_dir,
                        "provenance": _provenance(args)})
    print(f"[fusion] done: {manifest_out['n_proteins_ok']} ok, "
          f"{manifest_out['n_proteins_failed']} failed -> {args.out_dir}")
    if manifest_out["n_proteins_ok"] == 0:
        raise SystemExit(f"[fusion] 0 proteins succeeded ({manifest_out['n_proteins_failed']} failed)")


def _validate_manifest_against_references(manifest, proteins, test_lookup) -> None:
    """Fail-fast if any manifest hard anchor disagrees with the reference/WT sequence (bounds +
    expected_aa) before any generation work — the driver-side half of the P1-6 anchor firewall."""
    if manifest is None:
        return
    for pid in proteins:
        if manifest.has_protein(pid) and pid in test_lookup:
            manifest.constraint_for_protein(pid).validate_against_sequence(
                str(test_lookup[pid]["sequence"]))


def _anchor_expected(manifest, protein_id):
    if manifest is None or not manifest.has_protein(protein_id):
        return None
    c = manifest.constraint_for_protein(protein_id)
    if not c.hard_anchors:
        return None
    return {int(a.index_0b): str(a.expected_aa) for a in c.hard_anchors}


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


if __name__ == "__main__":
    main()
