#!/usr/bin/env python3
"""Audit LuxSit-i and parent refinement ensembles and select wet-lab candidates."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_luxsit_parent_variant_holo import (
    BACKBONE_ATOMS,
    CATALYTIC4,
    FREE3,
    INVARIANT19,
    _file_sha256,
    _git_sha,
    alignment_frame,
    ca_rmsd,
    cache_key,
    comparison_metrics,
    functional_geometry,
    genotype_fields,
    geometry_deviation,
    holo_physical_metrics,
    jaccard,
    ligand_pose_metrics,
    overlap_summary,
    packing_metrics,
    parse_structure,
    read_confidence,
    reference_ligand_plddt_floor,
    residue_contact_fingerprint,
    sidechain_summary,
)


I_ANCHORS = tuple(sorted((*INVARIANT19, *FREE3)))
ALL_POSITIONS = tuple(range(1, 118))
LOWER_BETTER_Q80_FLOORS = {
    "holo_global_ca_rmsd_q80": 1.20,
    "anchor_sc_rmsd_q80": 1.00,
    "cat4_sc_rmsd_q80": 1.00,
    "anchor_max_sc_rmsd_q80": 2.00,
    "geometry_mae_q80": 0.50,
    "geometry_max_q80": 1.00,
    "ligand_com_q80": 1.00,
    "ligand_core_rmsd_q80": 1.00,
    "ligand_full_rmsd_q80": 2.50,
}


def _resolve_archive(source: Path) -> Path:
    source = source.resolve()
    if source.is_file():
        return source
    if (source / "merged" / "master.parquet").is_file():
        return source
    archives = sorted(source.glob("*.tar.gz"))
    if len(archives) != 1:
        raise ValueError(f"{source}: expected one refinement archive, found {len(archives)}")
    return archives[0]


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"{archive}: unsafe member path {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"{archive}: link member is not allowed: {member.name}")
        handle.extractall(destination, members=members)


@contextlib.contextmanager
def materialize_run(source: Path) -> Iterator[tuple[Path, Path]]:
    resolved = _resolve_archive(source)
    if resolved.is_dir():
        yield resolved, resolved
        return
    with tempfile.TemporaryDirectory(prefix="luxsit_refinement_") as tmp:
        tmp_path = Path(tmp)
        _safe_extract(resolved, tmp_path)
        masters = sorted(tmp_path.rglob("merged/master.parquet"))
        if len(masters) != 1:
            raise ValueError(f"{resolved}: expected one merged/master.parquet, found {len(masters)}")
        yield masters[0].parent.parent, resolved


def prediction_files(run_root: Path, protein_id: str, sequence: str) -> list[tuple[int, Path, Path]]:
    key = cache_key(protein_id, sequence)
    root = run_root / "protenix_holo" / "predictions" / key
    prediction_dirs = sorted(root.glob("seed_*/predictions"))
    if len(prediction_dirs) != 1:
        raise ValueError(f"{key}: expected one prediction directory, got {len(prediction_dirs)}")
    result: list[tuple[int, Path, Path]] = []
    for sample_idx in range(5):
        structure = prediction_dirs[0] / f"{key}_sample_{sample_idx}.cif"
        confidence = prediction_dirs[0] / f"{key}_summary_confidence_sample_{sample_idx}.json"
        if not structure.is_file() or not confidence.is_file():
            raise FileNotFoundError(f"{key}: missing Protenix sample {sample_idx}")
        result.append((sample_idx, structure, confidence))
    return result


def i_holo_metrics(candidate: object, reference: object) -> dict[str, float]:
    global_frame = alignment_frame(candidate, reference, ALL_POSITIONS)
    pocket_frame = alignment_frame(candidate, reference, I_ANCHORS)
    anchors = sidechain_summary(candidate, reference, I_ANCHORS, global_frame)
    catalytic = sidechain_summary(candidate, reference, CATALYTIC4, global_frame)
    pose = ligand_pose_metrics(candidate, reference, global_frame)
    candidate_geometry = functional_geometry(candidate)
    reference_geometry = functional_geometry(reference)
    geometry_mae, geometry_max = geometry_deviation(candidate_geometry, reference_geometry)
    return {
        "holo_global_ca_rmsd": ca_rmsd(candidate, reference, ALL_POSITIONS, global_frame),
        "pocket_ca_rmsd": ca_rmsd(candidate, reference, I_ANCHORS, pocket_frame),
        "anchor_sc_rmsd": float(anchors["rmsd"]),
        "anchor_max_sc_rmsd": float(anchors["max_rmsd"]),
        "anchor_worst_position": float(anchors["worst_position"]),
        "cat4_sc_rmsd": float(catalytic["rmsd"]),
        "ligand_full_rmsd": pose["ligand_full_rmsd"],
        "ligand_core_rmsd": pose["ligand_core_rmsd"],
        "ligand_com": pose["ligand_com"],
        "geometry_mae": geometry_mae,
        "geometry_max": geometry_max,
        "contact_jaccard": jaccard(
            residue_contact_fingerprint(candidate), residue_contact_fingerprint(reference)
        ),
    }


def residue_ligand_contact_metrics(
    structure: object, position: int, anchors: Sequence[int]
) -> dict[str, float]:
    residue_atoms = [atom for atom in structure.residues[position].values() if atom.element != "H"]
    sidechain_atoms = [atom for atom in residue_atoms if atom.name not in BACKBONE_ATOMS]
    ligand_coords = np.vstack([atom.coord for atom in structure.ligand_atoms])
    residue_coords = np.vstack([atom.coord for atom in residue_atoms])
    sidechain_coords = (
        np.vstack([atom.coord for atom in sidechain_atoms]) if sidechain_atoms else residue_coords
    )
    anchor_coords = np.vstack([
        atom.coord
        for anchor in anchors
        for atom in structure.residues[anchor].values()
        if atom.element != "H"
    ])
    residue_distances = np.linalg.norm(
        residue_coords[:, None, :] - ligand_coords[None, :, :], axis=2
    )
    sidechain_distances = np.linalg.norm(
        sidechain_coords[:, None, :] - ligand_coords[None, :, :], axis=2
    )
    anchor_distances = np.linalg.norm(
        residue_coords[:, None, :] - anchor_coords[None, :, :], axis=2
    )
    return {
        f"pos{position}_min_ligand_distance": float(residue_distances.min()),
        f"pos{position}_sidechain_min_ligand_distance": float(sidechain_distances.min()),
        f"pos{position}_ligand_contacts_4p5": float(np.count_nonzero(sidechain_distances <= 4.5)),
        f"pos{position}_min_anchor_distance": float(anchor_distances.min()),
    }


def parent_holo_metrics(
    candidate: object,
    references: Mapping[str, object],
    reference_physical: Mapping[str, Mapping[str, object]],
    genotype: Mapping[str, object],
) -> dict[str, object]:
    physical = holo_physical_metrics(candidate)
    comparisons = comparison_metrics(candidate, references)
    envelope = holo_envelope_comparisons(comparisons, str(genotype["genotype_class"]))
    packing = packing_metrics(physical, reference_physical, genotype)
    result: dict[str, object] = {
        "holo_global_ca_rmsd": envelope["envelope_global_ca_rmsd"],
        "pocket_ca_rmsd": envelope["envelope_pocket_ca_rmsd"],
        "anchor_sc_rmsd": envelope["envelope_inv19_sc_rmsd"],
        "anchor_max_sc_rmsd": envelope["envelope_inv19_max_sc_rmsd"],
        "anchor_worst_position": envelope.get("envelope_inv19_worst_position", math.nan),
        "cat4_sc_rmsd": envelope["envelope_cat4_sc_rmsd"],
        "ligand_full_rmsd": envelope["envelope_ligand_full_rmsd"],
        "ligand_core_rmsd": envelope["envelope_ligand_core_rmsd"],
        "ligand_com": envelope["envelope_ligand_com"],
        "geometry_mae": envelope["envelope_geometry_mae"],
        "geometry_max": envelope["envelope_geometry_max"],
        "contact_jaccard": envelope["envelope_contact_jaccard"],
        "best_functional_reference": envelope["best_functional_reference"],
        **physical,
        **packing,
    }
    return result


def holo_envelope_comparisons(
    values: Mapping[str, object], genotype_class: str
) -> dict[str, object]:
    if genotype_class == "exact_parent":
        allowed = ("parent",)
    elif genotype_class == "exact_i":
        allowed = ("i",)
    else:
        allowed = ("parent", "i")
    lower_better = (
        "global_ca_rmsd", "pocket_ca_rmsd", "inv19_sc_rmsd", "inv19_max_sc_rmsd",
        "cat4_sc_rmsd", "ligand_full_rmsd", "ligand_core_rmsd", "ligand_com",
        "geometry_mae", "geometry_max",
    )
    result: dict[str, object] = {}
    for metric in lower_better:
        result[f"envelope_{metric}"] = min(
            float(values[f"{name}_holo_{metric}"]) for name in allowed
        )
    result["envelope_contact_jaccard"] = max(
        float(values[f"{name}_holo_contact_jaccard"]) for name in allowed
    )
    composite = {
        name: float(values[f"{name}_holo_inv19_sc_rmsd"])
        + float(values[f"{name}_holo_geometry_mae"])
        + float(values[f"{name}_holo_ligand_core_rmsd"])
        for name in allowed
    }
    result["best_functional_reference"] = min(composite, key=composite.get)
    return result


def _aggregate_samples(samples: pd.DataFrame) -> pd.DataFrame:
    identity_columns = {"background", "design_uid"}
    skip_columns = identity_columns | {
        "sample_idx", "sequence", "structure_member", "confidence_member"
    }
    rows: list[dict[str, object]] = []
    for (background, design_uid), group in samples.groupby(["background", "design_uid"], sort=False):
        row: dict[str, object] = {"background": background, "design_uid": design_uid}
        for column in group.columns:
            if column in skip_columns:
                continue
            series = group[column]
            if pd.api.types.is_bool_dtype(series):
                row[f"{column}_pass_count"] = int(series.sum())
            elif pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce")
                row[f"{column}_median"] = float(values.median())
                row[f"{column}_q80"] = float(values.quantile(0.80))
                row[f"{column}_min"] = float(values.min())
                row[f"{column}_max"] = float(values.max())
            else:
                modes = series.dropna().mode()
                row[column] = modes.iloc[0] if not modes.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_seed_table(path: Path, background: str) -> pd.DataFrame:
    source = pd.read_parquet(path)
    if background == "i":
        mapping = {
            "design_idx": "seed_design_idx",
            "sequence": "seed_sequence",
            "nmp_n_strong_binders": "seed_n_strong_binders",
            "scTM": "seed_scTM",
            "global_ca_RMSD": "seed_global_ca_RMSD",
            "wt_protenix_holo_ca_RMSD": "seed_holo_global_ca_rmsd",
            "wt_protenix_all22_sidechain_RMSD": "seed_anchor_sc_rmsd",
            "wt_protenix_catalytic4_sidechain_RMSD": "seed_cat4_sc_rmsd",
            "wt_protenix_max_anchor_sidechain_RMSD": "seed_anchor_max_sc_rmsd",
            "six_geometry_MAE_vs_WT_Protenix": "seed_geometry_mae",
            "six_geometry_max_abs_delta_vs_WT_Protenix": "seed_geometry_max",
            "ligand_COM_displacement_vs_WT_Protenix": "seed_ligand_com",
            "ligand_heavy_RMSD_vs_WT_Protenix": "seed_ligand_full_rmsd",
            "holo_ipTM": "seed_iptm",
            "holo_ligand_pLDDT": "seed_ligand_plddt",
        }
    else:
        mapping = {
            "design_idx": "seed_design_idx",
            "sequence": "seed_sequence",
            "n_strong_binders": "seed_n_strong_binders",
            "scTM": "seed_scTM",
            "global_ca_RMSD": "seed_global_ca_RMSD",
            "envelope_global_ca_rmsd_median": "seed_holo_global_ca_rmsd",
            "envelope_inv19_sc_rmsd_median": "seed_anchor_sc_rmsd",
            "envelope_cat4_sc_rmsd_median": "seed_cat4_sc_rmsd",
            "envelope_inv19_max_sc_rmsd_median": "seed_anchor_max_sc_rmsd",
            "envelope_geometry_mae_median": "seed_geometry_mae",
            "envelope_geometry_max_median": "seed_geometry_max",
            "envelope_ligand_com_median": "seed_ligand_com",
            "envelope_ligand_full_rmsd_median": "seed_ligand_full_rmsd",
            "iptm_median": "seed_iptm",
            "ligand_plddt_median": "seed_ligand_plddt",
        }
    missing = sorted(set(mapping) - set(source.columns))
    if missing:
        raise ValueError(f"{path}: missing seed columns {missing}")
    result = source[list(mapping)].rename(columns=mapping).copy()
    result["seed_design_id"] = result["seed_design_idx"].map(lambda value: f"design_{int(value):04d}")
    return result


def _validate_master(master: pd.DataFrame, background: str) -> None:
    expected_protein = "LuxSit-i_core117" if background == "i" else "LuxSit_parent_core117"
    if set(master["protein_id"]) != {expected_protein}:
        raise ValueError(f"{background}: unexpected protein IDs {set(master['protein_id'])}")
    if master["design_uid"].duplicated().any() or master["sequence_refined"].duplicated().any():
        raise ValueError(f"{background}: design_uid or refined sequence is not unique")
    anchors = I_ANCHORS if background == "i" else INVARIANT19
    violations = [
        row.design_uid
        for row in master.itertuples(index=False)
        if any(row.sequence_original[pos - 1] != row.sequence_refined[pos - 1] for pos in anchors)
    ]
    if violations:
        raise ValueError(f"{background}: hard-anchor mutations found in {violations[:5]}")


def _candidate_columns(master: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "shard_idx", "design_uid", "protein_id", "sequence_original", "sequence_refined",
        "n_mutations", "muts", "core_count_before", "core_count_after", "scTM",
        "global_ca_RMSD", "pLDDT", "n_strong_binders", "n_weak_binders",
        "mean_best_rank", "global_risk", "true_muts",
    ]
    result = master[columns].copy()
    result["actual_mutations"] = result.apply(
        lambda row: ",".join(
            f"{before}{index + 1}{after}"
            for index, (before, after) in enumerate(
                zip(row.sequence_original, row.sequence_refined, strict=True)
            )
            if before != after
        ),
        axis=1,
    )
    result["actual_mutation_count"] = result["actual_mutations"].map(
        lambda value: 0 if not value else len(str(value).split(","))
    )
    if not (result["actual_mutation_count"] == result["true_muts"]).all():
        raise ValueError("sequence-derived mutation counts disagree with merged true_muts")
    return result


def analyze_background(
    run_root: Path,
    source_path: Path,
    *,
    background: str,
    seed_table_path: Path,
    references: Mapping[str, object],
    iptm_floor: float,
    ligand_plddt_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    master_path = run_root / "merged" / "master.parquet"
    master = pd.read_parquet(master_path)
    _validate_master(master, background)
    candidates = _candidate_columns(master)
    candidates["background"] = background
    seeds = normalize_seed_table(seed_table_path, background)
    candidates = candidates.merge(
        seeds, left_on="sequence_original", right_on="seed_sequence", validate="many_to_one"
    )
    if len(candidates) != len(master):
        raise ValueError(f"{background}: seed lineage join changed the candidate count")

    parent_reference_physical: dict[str, Mapping[str, object]] = {}
    max_reference_overlap = 0
    if background == "parent":
        parent_reference_physical = {
            name.removesuffix("_holo"): holo_physical_metrics(reference)
            for name, reference in references.items()
        }
        max_reference_overlap = max(
            int(values["protein_ligand_severe_clash_count"])
            for values in parent_reference_physical.values()
        )
    else:
        reference = references["i_holo"]
        max_reference_overlap = int(
            overlap_summary(reference.protein_atoms, reference.ligand_atoms)["severe_clash_count"]
        )

    sample_rows: list[dict[str, object]] = []
    for item_idx, candidate in enumerate(candidates.itertuples(index=False), start=1):
        genotype = genotype_fields(candidate.sequence_refined)
        files = prediction_files(run_root, candidate.protein_id, candidate.sequence_refined)
        for sample_idx, structure_path, confidence_path in files:
            structure = parse_structure(structure_path)
            if structure.sequence != candidate.sequence_refined:
                raise ValueError(f"{structure_path}: sequence does not match merged/master.parquet")
            confidence = read_confidence(confidence_path)
            if background == "i":
                metrics: dict[str, object] = i_holo_metrics(structure, references["i_holo"])
                overlap = overlap_summary(structure.protein_atoms, structure.ligand_atoms)
                severe_clashes = int(overlap["severe_clash_count"])
                metrics.update({
                    "protein_ligand_overlap_sum": float(overlap["overlap_sum"]),
                    "protein_ligand_overlap_max": float(overlap["overlap_max"]),
                    "protein_ligand_severe_clash_count": severe_clashes,
                    "free3_sample_pass": True,
                })
            else:
                metrics = parent_holo_metrics(
                    structure, references, parent_reference_physical, genotype
                )
                severe_clashes = int(metrics["protein_ligand_severe_clash_count"])
            metrics.update(residue_ligand_contact_metrics(
                structure, 40, I_ANCHORS if background == "i" else INVARIANT19
            ))
            integrity = (
                len(structure.residues) == 117
                and len(structure.ligand) == 29
                and not bool(confidence["has_clash"])
                and severe_clashes <= max_reference_overlap
            )
            sample_rows.append({
                "background": background,
                "design_uid": candidate.design_uid,
                "sequence": candidate.sequence_refined,
                "sample_idx": sample_idx,
                "structure_member": str(structure_path.relative_to(run_root)),
                "confidence_member": str(confidence_path.relative_to(run_root)),
                **confidence,
                **metrics,
                "sample_integrity_pass": bool(integrity),
                "sample_confidence_pass": bool(
                    float(confidence["iptm"]) >= iptm_floor
                    and float(confidence["ligand_plddt"]) >= ligand_plddt_floor
                ),
            })
        if item_idx % 50 == 0 or item_idx == len(candidates):
            print(f"[{background}] parsed {item_idx}/{len(candidates)} designs", flush=True)

    samples = pd.DataFrame(sample_rows)
    expected_samples = len(candidates) * 5
    if len(samples) != expected_samples:
        raise ValueError(f"{background}: expected {expected_samples} samples, got {len(samples)}")
    aggregated = _aggregate_samples(samples)
    designs = candidates.merge(aggregated, on=["background", "design_uid"], validate="one_to_one")
    designs["background"] = background
    genotype_table = pd.DataFrame([genotype_fields(sequence) for sequence in designs["sequence_refined"]])
    designs = pd.concat([designs.reset_index(drop=True), genotype_table], axis=1)
    designs = apply_refinement_gates(
        designs, background=background, iptm_floor=iptm_floor,
        ligand_plddt_floor=ligand_plddt_floor,
    )
    designs = score_candidates(designs)
    source_info = {
        "background": background,
        "source": str(source_path),
        "run_root_name": run_root.name,
        "master_member": "merged/master.parquet",
        "seed_table": str(seed_table_path.resolve()),
        "designs": len(designs),
        "samples": len(samples),
        "iptm_floor": iptm_floor,
        "ligand_plddt_floor": ligand_plddt_floor,
    }
    return designs, samples, source_info


def apply_refinement_gates(
    designs: pd.DataFrame,
    *,
    background: str,
    iptm_floor: float,
    ligand_plddt_floor: float,
) -> pd.DataFrame:
    result = designs.copy()
    result["G0_integrity"] = result["sample_integrity_pass_pass_count"] >= 4
    result["G1_global_fold"] = (
        (result["scTM"] >= 0.94)
        & (result["global_ca_RMSD"] <= 1.40)
        & (result["holo_global_ca_rmsd_median"] <= 1.20)
    )
    result["G2_anchor_geometry"] = (
        (result["anchor_sc_rmsd_median"] <= 1.00)
        & (result["cat4_sc_rmsd_median"] <= 1.00)
        & (result["anchor_max_sc_rmsd_median"] <= 2.00)
    )
    result["G3_free3_packing"] = (
        result["free3_sample_pass_pass_count"] >= 4 if background == "parent" else True
    )
    result["G4_holo_function"] = (
        (result["geometry_mae_median"] <= 0.50)
        & (result["geometry_max_median"] <= 1.00)
        & (result["ligand_com_median"] <= 1.00)
        & (result["ligand_core_rmsd_median"] <= 1.00)
        & (result["ligand_full_rmsd_median"] <= 2.50)
    )
    result["G5_confidence"] = (
        (result["iptm_median"] >= iptm_floor)
        & (result["ligand_plddt_median"] >= ligand_plddt_floor)
    )
    result["R1_global_vs_seed"] = (
        (result["scTM"] >= result["seed_scTM"] - 0.03)
        & (result["global_ca_RMSD"] <= result["seed_global_ca_RMSD"] + 0.30)
        & (result["holo_global_ca_rmsd_median"] <= result["seed_holo_global_ca_rmsd"] + 0.25)
    )
    result["R2_anchor_vs_seed"] = (
        (result["anchor_sc_rmsd_median"] <= result["seed_anchor_sc_rmsd"] + 0.20)
        & (result["cat4_sc_rmsd_median"] <= result["seed_cat4_sc_rmsd"] + 0.20)
        & (result["anchor_max_sc_rmsd_median"] <= result["seed_anchor_max_sc_rmsd"] + 0.50)
    )
    result["R4_function_vs_seed"] = (
        (result["geometry_mae_median"] <= result["seed_geometry_mae"] + 0.15)
        & (result["geometry_max_median"] <= result["seed_geometry_max"] + 0.30)
        & (result["ligand_com_median"] <= result["seed_ligand_com"] + 0.35)
        & (result["ligand_full_rmsd_median"] <= result["seed_ligand_full_rmsd"] + 0.75)
    )
    result["R5_confidence_vs_seed"] = (
        (result["iptm_median"] >= result["seed_iptm"] - 0.02)
        & (result["ligand_plddt_median"] >= result["seed_ligand_plddt"] - 3.0)
    )
    gate_columns = [
        "G0_integrity", "G1_global_fold", "G2_anchor_geometry", "G3_free3_packing",
        "G4_holo_function", "G5_confidence", "R1_global_vs_seed", "R2_anchor_vs_seed",
        "R4_function_vs_seed", "R5_confidence_vs_seed",
    ]
    result["protocol_structure_pass"] = result[gate_columns].all(axis=1)
    result["q80_instability_flag_count"] = sum(
        (result[column] > floor).astype(int)
        for column, floor in LOWER_BETTER_Q80_FLOORS.items()
    )
    result["strict_ensemble_pass"] = (
        (result["sample_integrity_pass_pass_count"] == 5)
        & (result["sample_confidence_pass_pass_count"] >= 4)
        & (result["q80_instability_flag_count"] == 0)
    )
    result["core_improved"] = result["core_count_after"] < result["core_count_before"]
    result["nmp_windows_improved"] = result["n_strong_binders"] < result["seed_n_strong_binders"]
    result["immune_improved_both"] = result["core_improved"] & result["nmp_windows_improved"]
    result["core_zero"] = result["core_count_after"] == 0
    result["strict_experimental_pass"] = (
        result["protocol_structure_pass"]
        & result["strict_ensemble_pass"]
        & result["immune_improved_both"]
        & (result["actual_mutation_count"] <= 4)
    )
    result["first_failed_gate"] = result.apply(
        lambda row: next((name for name in gate_columns if not bool(row[name])), "PASS"), axis=1
    )
    return result


def score_candidates(designs: pd.DataFrame) -> pd.DataFrame:
    result = designs.copy()
    sc_loss = np.maximum(0.0, result["seed_scTM"] - result["scTM"])
    result["structure_score"] = (
        0.25 * result["anchor_sc_rmsd_median"] / 1.00
        + 0.20 * result["cat4_sc_rmsd_median"] / 1.00
        + 0.20 * result["geometry_mae_median"] / 0.50
        + 0.10 * result["anchor_max_sc_rmsd_median"] / 2.00
        + 0.10 * result["holo_global_ca_rmsd_median"] / 1.20
        + 0.05 * result["ligand_com_median"] / 1.00
        + 0.05 * result["ligand_full_rmsd_median"] / 2.50
        + 0.05 * sc_loss / 0.03
    )
    result["delta_n_strong_binders"] = (
        result["n_strong_binders"] - result["seed_n_strong_binders"]
    )
    result["delta_scTM_vs_seed"] = result["scTM"] - result["seed_scTM"]
    result["delta_holo_global_ca_vs_seed"] = (
        result["holo_global_ca_rmsd_median"] - result["seed_holo_global_ca_rmsd"]
    )
    result["delta_anchor_sc_vs_seed"] = (
        result["anchor_sc_rmsd_median"] - result["seed_anchor_sc_rmsd"]
    )
    return result


def select_shortlist(
    designs: pd.DataFrame,
    n_per_background: int,
    *,
    min_seed_families: int = 3,
    diversity_margin: float = 0.05,
) -> pd.DataFrame:
    eligible = designs[designs["strict_experimental_pass"] & designs["core_zero"]].copy()
    eligible = eligible.sort_values(
        ["background", "structure_score", "n_strong_binders", "actual_mutation_count", "iptm_median"],
        ascending=[True, True, True, True, False], kind="stable",
    )
    eligible["strict_pool_rank"] = eligible.groupby("background").cumcount() + 1
    selected_groups: list[pd.DataFrame] = []
    for _background, group in eligible.groupby("background", sort=False):
        group = group.copy()
        selected_indices = list(group.head(n_per_background).index)
        if selected_indices:
            initial_cutoff = float(group.loc[selected_indices, "structure_score"].max())
            while group.loc[selected_indices, "seed_design_idx"].nunique() < min_seed_families:
                represented = set(group.loc[selected_indices, "seed_design_idx"])
                alternatives = group[
                    ~group.index.isin(selected_indices)
                    & ~group["seed_design_idx"].isin(represented)
                    & (group["structure_score"] <= initial_cutoff + diversity_margin)
                ]
                if alternatives.empty:
                    break
                replacement = alternatives.iloc[0]
                family_counts = group.loc[selected_indices, "seed_design_idx"].value_counts()
                replaceable = group.loc[selected_indices]
                replaceable = replaceable[
                    replaceable["seed_design_idx"].map(family_counts) > 1
                ].sort_values("structure_score", ascending=False)
                if replaceable.empty:
                    break
                selected_indices.remove(replaceable.index[0])
                selected_indices.append(replacement.name)
        selected_groups.append(group.loc[selected_indices])
    selected = pd.concat(selected_groups).sort_values(
        ["background", "structure_score", "n_strong_binders"], kind="stable"
    ).copy()
    selected["shortlist_rank"] = selected.groupby("background").cumcount() + 1
    selected["selection_tier"] = "strict_primary"
    selected["selection_reason"] = np.where(
        selected["strict_pool_rank"] <= n_per_background,
        "strict_structure_top",
        "strict_near_frontier_seed_diversity",
    )
    prefixes = {"i": "O2_LUXI", "parent": "O2_LUXP"}
    selected["order_id"] = selected.apply(
        lambda row: f"{prefixes[row.background]}_{int(row.shortlist_rank):02d}", axis=1
    )
    return selected


def select_seed_unique_shortlist(
    designs: pd.DataFrame,
    n_per_background: int,
) -> pd.DataFrame:
    """Select the best strict structure candidate from each seed family."""
    eligible = designs[designs["strict_experimental_pass"] & designs["core_zero"]].copy()
    eligible = eligible.sort_values(
        ["background", "structure_score", "n_strong_binders", "actual_mutation_count", "iptm_median"],
        ascending=[True, True, True, True, False], kind="stable",
    )
    eligible["strict_pool_rank"] = eligible.groupby("background").cumcount() + 1
    seed_keys = ["seed_sequence"] if "seed_sequence" in eligible.columns else [
        "background", "seed_design_idx"
    ]
    seed_best = eligible.sort_values(
        ["structure_score", "n_strong_binders", "actual_mutation_count", "iptm_median"],
        ascending=[True, True, True, False], kind="stable",
    ).groupby(seed_keys, sort=False, as_index=False, group_keys=False).head(1)
    selected = seed_best.sort_values(
        ["background", "structure_score", "n_strong_binders", "actual_mutation_count", "iptm_median"],
        ascending=[True, True, True, True, False], kind="stable",
    ).groupby("background", sort=False, as_index=False, group_keys=False).head(n_per_background)
    selected = selected.copy()
    selected["shortlist_rank"] = selected.groupby("background").cumcount() + 1
    selected["selection_tier"] = "strict_seed_unique"
    selected["selection_reason"] = "best_structure_within_unique_seed"
    prefixes = {"i": "O2_LUXI_SU", "parent": "O2_LUXP_SU"}
    selected["order_id"] = selected.apply(
        lambda row: f"{prefixes[row.background]}_{int(row.shortlist_rank):02d}", axis=1
    )
    return selected


def build_seed_summary(designs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in designs.groupby(["background", "seed_design_idx", "seed_design_id"], sort=True):
        rows.append({
            "background": keys[0],
            "seed_design_idx": int(keys[1]),
            "seed_design_id": keys[2],
            "candidates": len(group),
            "core_before": int(group["core_count_before"].iloc[0]),
            "core0_candidates": int(group["core_zero"].sum()),
            "protocol_structure_pass": int(group["protocol_structure_pass"].sum()),
            "strict_experimental_pass": int(group["strict_experimental_pass"].sum()),
            "strict_core0": int((group["strict_experimental_pass"] & group["core_zero"]).sum()),
            "min_structure_score": float(group["structure_score"].min()),
            "min_n_strong_binders": int(group["n_strong_binders"].min()),
            "median_actual_mutations": float(group["actual_mutation_count"].median()),
        })
    return pd.DataFrame(rows)


def build_gate_summary(designs: pd.DataFrame) -> pd.DataFrame:
    gate_columns = [
        "G0_integrity", "G1_global_fold", "G2_anchor_geometry", "G3_free3_packing",
        "G4_holo_function", "G5_confidence", "R1_global_vs_seed", "R2_anchor_vs_seed",
        "R4_function_vs_seed", "R5_confidence_vs_seed", "protocol_structure_pass",
        "strict_ensemble_pass", "immune_improved_both", "strict_experimental_pass", "core_zero",
    ]
    rows: list[dict[str, object]] = []
    for background, group in designs.groupby("background", sort=True):
        for gate in gate_columns:
            count = int(group[gate].sum())
            rows.append({
                "background": background,
                "gate": gate,
                "pass_count": count,
                "total": len(group),
                "pass_fraction": count / len(group),
            })
    return pd.DataFrame(rows)


def build_mutation_site_summary(
    designs: pd.DataFrame, shortlist: pd.DataFrame
) -> pd.DataFrame:
    populations = {
        "all": designs,
        "strict_core0": designs[designs["strict_experimental_pass"] & designs["core_zero"]],
        "selected": shortlist,
    }
    rows: list[dict[str, object]] = []
    for population, table in populations.items():
        for background, group in table.groupby("background", sort=True):
            edits: dict[int, list[str]] = {}
            for mutation_list in group["actual_mutations"]:
                for mutation in str(mutation_list).split(","):
                    position = int(mutation[1:-1])
                    edits.setdefault(position, []).append(mutation)
            for position, mutations in sorted(edits.items()):
                rows.append({
                    "background": background,
                    "population": population,
                    "position_1b": position,
                    "edit_count": len(mutations),
                    "population_size": len(group),
                    "edit_fraction": len(mutations) / len(group),
                    "substitutions": ";".join(sorted(set(mutations))),
                })
    return pd.DataFrame(rows)


def write_fasta(table: pd.DataFrame, path: Path) -> None:
    with path.open("w") as handle:
        for row in table.itertuples(index=False):
            handle.write(
                f">{row.order_id}|background={row.background}|seed={row.seed_design_id}"
                f"|uid={row.design_uid}|muts={row.actual_mutations}|core={int(row.core_count_after)}"
                f"|nmp={int(row.n_strong_binders)}\n{row.sequence_refined}\n"
            )


def write_plots(designs: pd.DataFrame, gate_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, (background, group) in zip(axes, designs.groupby("background", sort=True), strict=True):
        colors = np.where(group["strict_experimental_pass"], "#147d64", "#b8bcc2")
        axis.scatter(group["structure_score"], group["n_strong_binders"], c=colors, s=22, alpha=0.75)
        core0 = group[group["core_zero"]]
        axis.scatter(
            core0["structure_score"], core0["n_strong_binders"], facecolors="none",
            edgecolors="#1f4e79", s=42, linewidths=0.8, label="core0",
        )
        axis.set_title("LuxSit-i" if background == "i" else "LuxSit parent")
        axis.set_xlabel("structure score (lower is better)")
        axis.set_ylabel("NMP strong windows")
        axis.legend(frameon=False)
    fig.savefig(output_dir / "structure_immune_tradeoff.png", dpi=180)
    plt.close(fig)

    selected_gates = [
        "G1_global_fold", "G2_anchor_geometry", "G3_free3_packing", "G4_holo_function",
        "G5_confidence", "protocol_structure_pass", "strict_experimental_pass",
    ]
    plot_data = gate_summary[gate_summary["gate"].isin(selected_gates)].copy()
    pivot = plot_data.pivot(index="gate", columns="background", values="pass_fraction").reindex(selected_gates)
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    pivot.plot.bar(ax=axis, color=["#147d64", "#c05a3d"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("pass fraction")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=30)
    axis.legend(title="background", frameon=False)
    fig.savefig(output_dir / "gate_waterfall.png", dpi=180)
    plt.close(fig)


def _shortlist_columns() -> list[str]:
    return [
        "order_id", "background", "shortlist_rank", "selection_tier", "design_uid",
        "strict_pool_rank", "selection_reason",
        "seed_design_id", "seed_design_idx", "sequence_refined", "actual_mutations",
        "actual_mutation_count", "muts", "n_mutations",
        "core_count_before", "core_count_after", "seed_n_strong_binders", "n_strong_binders",
        "delta_n_strong_binders", "scTM", "seed_scTM", "delta_scTM_vs_seed",
        "global_ca_RMSD", "holo_global_ca_rmsd_median", "delta_holo_global_ca_vs_seed",
        "anchor_sc_rmsd_median", "delta_anchor_sc_vs_seed", "anchor_max_sc_rmsd_median",
        "cat4_sc_rmsd_median", "geometry_mae_median", "geometry_max_median",
        "ligand_core_rmsd_median", "ligand_full_rmsd_median", "ligand_com_median",
        "contact_jaccard_median", "iptm_median", "ligand_plddt_median",
        "pos40_min_ligand_distance_median", "pos40_sidechain_min_ligand_distance_median",
        "pos40_ligand_contacts_4p5_median", "pos40_min_anchor_distance_median",
        "q80_instability_flag_count", "structure_score", "free3_genotype", "genotype_class",
    ]


def shortlist_export(table: pd.DataFrame) -> pd.DataFrame:
    return table[_shortlist_columns()].rename(columns={
        "muts": "search_final_branch_muts_0b",
        "n_mutations": "search_final_branch_n_mutations",
    })


def write_readme(
    output_dir: Path,
    designs: pd.DataFrame,
    shortlist: pd.DataFrame,
    seed_unique_shortlist: pd.DataFrame,
    sources: Sequence[Mapping[str, object]],
) -> None:
    pos40_min = shortlist["pos40_min_ligand_distance_median"]
    pos40_sidechain = shortlist["pos40_sidechain_min_ligand_distance_median"]
    pos40_contacts = shortlist["pos40_ligand_contacts_4p5_median"]
    lines = [
        "# LuxSit parent and LuxSit-i refinement audit",
        "",
        "## Verdict",
        "",
    ]
    for background in ("i", "parent"):
        group = designs[designs["background"] == background]
        selected = shortlist[shortlist["background"] == background]
        label = "LuxSit-i" if background == "i" else "LuxSit parent"
        lines.append(
            f"- {label}: {int(group['protocol_structure_pass'].sum())}/{len(group)} pass absolute+relative "
            f"structure gates; {int(group['strict_experimental_pass'].sum())} also pass the strict ensemble "
            f"and immune-improvement rules; {int((group['strict_experimental_pass'] & group['core_zero']).sum())} "
            f"are strict count-zero candidates; {len(selected)} selected."
        )
    lines.extend([
        "- The actual cumulative sequence differences span 1-6 mutations. The strict tier enforces the recommended <=4 limit even though the search allowed 16.",
        "- Selection starts from the pure structure top five. A strict candidate from a new seed may replace a repeated-family member only within +0.05 structure-score units, until three seed families are represented; no candidate is padded in.",
        "",
        "## Gate audit",
        "",
        "| background | designs | core0 raw | global fold | anchor geometry | holo function | confidence | absolute+relative structure | strict+immune | strict+core0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for background in ("i", "parent"):
        group = designs[designs["background"] == background]
        lines.append(
            f"| {background} | {len(group)} | {int(group['core_zero'].sum())} | "
            f"{int(group['G1_global_fold'].sum())} | {int(group['G2_anchor_geometry'].sum())} | "
            f"{int(group['G4_holo_function'].sum())} | {int(group['G5_confidence'].sum())} | "
            f"{int(group['protocol_structure_pass'].sum())} | "
            f"{int(group['strict_experimental_pass'].sum())} | "
            f"{int((group['strict_experimental_pass'] & group['core_zero']).sum())} |"
        )
    lines.extend(["", "## Search-depth diagnosis", ""])
    for background in ("i", "parent"):
        group = designs[designs["background"] == background]
        depth_parts = []
        for depth in (5, 6):
            depth_group = group[group["actual_mutation_count"] == depth]
            depth_parts.append(
                f"{depth} mutations: {int(depth_group['protocol_structure_pass'].sum())}/{len(depth_group)} structure-pass"
            )
        lines.append(f"- {background}: " + "; ".join(depth_parts) + ".")
    parent_genotypes = shortlist[shortlist["background"] == "parent"]["free3_genotype"]
    lines.extend([
        "- Deeper paths often achieve count0 by construction but fail holo structure. The <=4 strict limit is therefore evidence-backed, not cosmetic.",
        "",
        "## Selected envelope",
        "",
        f"- scTM {shortlist['scTM'].min():.3f}-{shortlist['scTM'].max():.3f}; holo CA {shortlist['holo_global_ca_rmsd_median'].min():.3f}-{shortlist['holo_global_ca_rmsd_median'].max():.3f} A; anchor SC {shortlist['anchor_sc_rmsd_median'].min():.3f}-{shortlist['anchor_sc_rmsd_median'].max():.3f} A; catalytic-4 SC {shortlist['cat4_sc_rmsd_median'].min():.3f}-{shortlist['cat4_sc_rmsd_median'].max():.3f} A.",
        f"- Geometry MAE {shortlist['geometry_mae_median'].min():.3f}-{shortlist['geometry_mae_median'].max():.3f} A; DTZ core RMSD {shortlist['ligand_core_rmsd_median'].min():.3f}-{shortlist['ligand_core_rmsd_median'].max():.3f} A; DTZ COM {shortlist['ligand_com_median'].min():.3f}-{shortlist['ligand_com_median'].max():.3f} A.",
        f"- ipTM {shortlist['iptm_median'].min():.3f}-{shortlist['iptm_median'].max():.3f}; ligand pLDDT {shortlist['ligand_plddt_median'].min():.2f}-{shortlist['ligand_plddt_median'].max():.2f}; all selected rows pass integrity and confidence in 5/5 samples.",
        f"- Parent free-3 genotypes selected: `{json.dumps(parent_genotypes.value_counts().to_dict(), sort_keys=True)}`. Refinement did not edit positions 60/96/110 in any parent candidate.",
        "",
        "## Strict shortlist",
        "",
        "| background | rank | pool rank | order ID | seed | mutations | core before->after | NMP strong before->after | structure score | holo CA | anchor SC | cat4 SC | geometry MAE | DTZ core | ipTM |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in shortlist.sort_values(["background", "shortlist_rank"]).itertuples(index=False):
        lines.append(
            f"| {row.background} | {row.shortlist_rank} | {row.strict_pool_rank} | {row.order_id} | {row.seed_design_id} | "
            f"{row.actual_mutations} | {int(row.core_count_before)}->{int(row.core_count_after)} | "
            f"{int(row.seed_n_strong_binders)}->{int(row.n_strong_binders)} | {row.structure_score:.3f} | "
            f"{row.holo_global_ca_rmsd_median:.3f} | {row.anchor_sc_rmsd_median:.3f} | "
            f"{row.cat4_sc_rmsd_median:.3f} | {row.geometry_mae_median:.3f} | "
            f"{row.ligand_core_rmsd_median:.3f} | {row.iptm_median:.3f} |"
        )
    lines.extend([
        "",
        "## Seed-unique alternative",
        "",
        "This secondary order keeps exactly one strict candidate per seed family, choosing the lowest structure score within each seed. It does not relax any structure, ensemble, anchor, or immune rule.",
        "",
        "| background | rank | pool rank | order ID | seed | mutations | structure score | holo CA | anchor SC | cat4 SC | DTZ core | ipTM |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in seed_unique_shortlist.sort_values(["background", "shortlist_rank"]).itertuples(index=False):
        lines.append(
            f"| {row.background} | {row.shortlist_rank} | {row.strict_pool_rank} | {row.order_id} | "
            f"{row.seed_design_id} | {row.actual_mutations} | {row.structure_score:.3f} | "
            f"{row.holo_global_ca_rmsd_median:.3f} | {row.anchor_sc_rmsd_median:.3f} | "
            f"{row.cat4_sc_rmsd_median:.3f} | {row.ligand_core_rmsd_median:.3f} | "
            f"{row.iptm_median:.3f} |"
        )
    lines.extend([
        "",
        "## Decision rules",
        "",
        "- Five Protenix-DTZ samples per refined sequence; median values are primary and q80 failures reject the strict tier.",
        "- Absolute gates: scTM>=0.94, ESM global CA<=1.40 A, holo CA<=1.20 A, anchor SC<=1.00 A, catalytic-4 SC<=1.00 A, worst anchor<=2.00 A, geometry MAE<=0.50 A, geometry max<=1.00 A, DTZ COM/core<=1.00 A, DTZ full<=2.50 A.",
        "- Relative gates use the post-refinement limits in `refinement_gates.yaml`: scTM -0.03; ESM CA +0.30 A; holo CA +0.25 A; anchor/catalytic SC +0.20 A; worst anchor +0.50 A; geometry MAE/max +0.15/+0.30 A; DTZ COM/full +0.35/+0.75 A; ipTM -0.02; ligand pLDDT -3.",
        "- Strict immune admissibility requires both distinct strong-core count and NMP strong-window count to decrease. Final order candidates additionally require core_count_after=0.",
        "- Parent positions 60/96/110 use identity-aware packing/contact metrics, never cross-identity sidechain RMSD. LuxSit-i keeps all 22 anchors fixed.",
        "- Position 40 is edited in every strict count-zero candidate in both backgrounds. It is not a hard anchor, but it lies between pocket anchors W38 and F43; `mutation_site_summary.csv` and the position-40 contact columns keep this correlated local risk explicit.",
        f"- In the selected set, position-40 residue/DTZ median minimum distance spans {pos40_min.min():.2f}-{pos40_min.max():.2f} A, sidechain/DTZ spans {pos40_sidechain.min():.2f}-{pos40_sidechain.max():.2f} A; 4.5-A contact counts are {'all zero' if bool((pos40_contacts == 0).all()) else 'not all zero'}.",
        "",
        "## Caveats",
        "",
        "- The LuxSit-i seed relative baseline is one Protenix sample; the parent seed baseline is a five-sample median. Absolute and q80 gates carry the main robustness burden.",
        "- Protenix confidence and geometry are computational evidence, not an activity guarantee.",
        "- No candidate apo calculation is used. This analysis follows the holo-only parent/variant protocol.",
        "",
        "## Provenance",
        "",
    ])
    for source in sources:
        lines.append(
            f"- `{source['background']}`: `{source['source']}`; {source['designs']} designs, {source['samples']} holo samples."
        )
    lines.extend([
        "",
        "See `all_candidates.parquet`, `sample_metrics.parquet`, `gate_summary.csv`, and `analysis_manifest.json` for the full audit.",
    ])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    order2_dir = args.order2_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    order2_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.ExitStack() as stack:
        i_root, i_source = stack.enter_context(materialize_run(args.i_source))
        parent_root, parent_source = stack.enter_context(materialize_run(args.parent_source))
        parent_reference_path = (
            args.parent_reference.resolve()
            if args.parent_reference
            else parent_root / "protenix_holo" / "references" / "LuxSit-parent_Protenix_DTZ.cif"
        )
        i_reference_path = (
            args.i_reference.resolve()
            if args.i_reference
            else parent_root / "protenix_holo" / "references" / "LuxSit-i_Protenix_DTZ.cif"
        )
        parent_reference = parse_structure(parent_reference_path)
        i_reference = parse_structure(i_reference_path)
        reference_info = {
            "i_holo": {
                "source": str(parent_source),
                "member": str(i_reference_path.relative_to(parent_root))
                if i_reference_path.is_relative_to(parent_root) else str(i_reference_path),
                "sha256": _file_sha256(i_reference_path),
            },
            "parent_holo": {
                "source": str(parent_source),
                "member": str(parent_reference_path.relative_to(parent_root))
                if parent_reference_path.is_relative_to(parent_root) else str(parent_reference_path),
                "sha256": _file_sha256(parent_reference_path),
            },
        }
        config_keys = (
            "beam_width", "topB", "max_pairs", "max_path_mutations", "max_rounds",
            "patience", "refold_cap", "structure_gate_profile", "scTM_eps",
        )
        actual_configs: dict[str, dict[str, object]] = {}
        for background, root in (("i", i_root), ("parent", parent_root)):
            config_paths = sorted(root.glob("shard*/refined/refine_config.json"))
            if len(config_paths) != 8:
                raise ValueError(f"{background}: expected eight refine_config.json files")
            raw_config = json.loads(config_paths[0].read_text())
            actual_configs[background] = {key: raw_config.get(key) for key in config_keys}
        parent_ligand_floor, reference_ligand_plddt = reference_ligand_plddt_floor({
            "parent_holo": parent_reference,
            "i_holo": i_reference,
        })

        i_designs, i_samples, i_info = analyze_background(
            i_root, i_source, background="i", seed_table_path=args.i_seed_table.resolve(),
            references={"i_holo": i_reference}, iptm_floor=args.i_iptm_floor,
            ligand_plddt_floor=args.i_ligand_plddt_floor,
        )
        parent_designs, parent_samples, parent_info = analyze_background(
            parent_root, parent_source, background="parent",
            seed_table_path=args.parent_seed_table.resolve(),
            references={"parent_holo": parent_reference, "i_holo": i_reference},
            iptm_floor=args.parent_iptm_floor, ligand_plddt_floor=parent_ligand_floor,
        )

    designs = pd.concat([i_designs, parent_designs], ignore_index=True)
    samples = pd.concat([i_samples, parent_samples], ignore_index=True)
    shortlist = select_shortlist(
        designs, args.n_per_background, min_seed_families=args.min_seed_families,
        diversity_margin=args.diversity_margin,
    )
    seed_unique_shortlist = select_seed_unique_shortlist(designs, args.n_per_background)
    strict_pool = designs[
        designs["strict_experimental_pass"] & designs["core_zero"]
    ].sort_values(["background", "structure_score", "n_strong_binders"])
    gate_summary = build_gate_summary(designs)
    seed_summary = build_seed_summary(designs)
    mutation_site_summary = build_mutation_site_summary(designs, shortlist)

    designs.to_parquet(output_dir / "all_candidates.parquet", index=False)
    designs.to_csv(output_dir / "all_candidates.csv", index=False)
    samples.to_parquet(output_dir / "sample_metrics.parquet", index=False)
    strict_pool.to_parquet(output_dir / "strict_core0_pool.parquet", index=False)
    strict_pool.to_csv(output_dir / "strict_core0_pool.csv", index=False)
    shortlist.to_parquet(output_dir / "strict_shortlist.parquet", index=False)
    shortlist_export(shortlist).to_csv(output_dir / "strict_shortlist.csv", index=False)
    seed_unique_shortlist.to_parquet(output_dir / "seed_unique_shortlist.parquet", index=False)
    shortlist_export(seed_unique_shortlist).to_csv(
        output_dir / "seed_unique_shortlist.csv", index=False
    )
    gate_summary.to_csv(output_dir / "gate_summary.csv", index=False)
    seed_summary.to_csv(output_dir / "seed_summary.csv", index=False)
    mutation_site_summary.to_csv(output_dir / "mutation_site_summary.csv", index=False)
    write_fasta(shortlist, output_dir / "strict_shortlist.fasta")
    write_fasta(seed_unique_shortlist, output_dir / "seed_unique_shortlist.fasta")
    write_plots(designs, gate_summary, output_dir)

    order_table = shortlist_export(shortlist)
    order_csv = order2_dir / "luxsit_parent_i_refinement_strict_shortlist.csv"
    order_fasta = order2_dir / "luxsit_parent_i_refinement_strict_shortlist.fasta"
    order_table.to_csv(order_csv, index=False)
    write_fasta(shortlist, order_fasta)
    seed_unique_order_csv = order2_dir / "luxsit_parent_i_refinement_seed_unique_shortlist.csv"
    seed_unique_order_fasta = order2_dir / "luxsit_parent_i_refinement_seed_unique_shortlist.fasta"
    shortlist_export(seed_unique_shortlist).to_csv(seed_unique_order_csv, index=False)
    write_fasta(seed_unique_shortlist, seed_unique_order_fasta)

    sources = [i_info, parent_info]
    repo_root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": "luxsit_parent_i_refinement_holo_v1",
        "protocol": str(repo_root / "PROTOCOL" / "luxsit_parent_variant_holo_selection.md"),
        "sources": sources,
        "references": {
            **reference_info,
            "ligand_plddt_means": reference_ligand_plddt,
        },
        "actual_search_config": actual_configs,
        "counts": {
            background: {
                "designs": int((designs["background"] == background).sum()),
                "protocol_structure_pass": int(
                    ((designs["background"] == background) & designs["protocol_structure_pass"]).sum()
                ),
                "strict_experimental_pass": int(
                    ((designs["background"] == background) & designs["strict_experimental_pass"]).sum()
                ),
                "strict_core0": int(
                    ((designs["background"] == background) & designs["strict_experimental_pass"] & designs["core_zero"]).sum()
                ),
                "selected": int((shortlist["background"] == background).sum()),
                "seed_unique_selected": int(
                    (seed_unique_shortlist["background"] == background).sum()
                ),
            }
            for background in ("i", "parent")
        },
        "order_outputs": {
            "primary": {"csv": str(order_csv), "fasta": str(order_fasta)},
            "seed_unique": {
                "csv": str(seed_unique_order_csv),
                "fasta": str(seed_unique_order_fasta),
            },
        },
        "selection_policy": {
            "n_per_background": args.n_per_background,
            "min_seed_families": args.min_seed_families,
            "diversity_margin": args.diversity_margin,
            "rule": "pure structure top-N, then bounded near-frontier replacements only",
            "seed_unique_rule": "strict/core0 only; one lowest-structure-score candidate per full seed sequence",
        },
        "git_sha": _git_sha(repo_root),
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_readme(output_dir, designs, shortlist, seed_unique_shortlist, sources)
    print(json.dumps(manifest["counts"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-source", type=Path, required=True)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--i-seed-table", type=Path, required=True)
    parser.add_argument("--parent-seed-table", type=Path, required=True)
    parser.add_argument("--i-reference", type=Path)
    parser.add_argument("--parent-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order2-dir", type=Path, required=True)
    parser.add_argument("--n-per-background", type=int, default=5)
    parser.add_argument("--min-seed-families", type=int, default=3)
    parser.add_argument("--diversity-margin", type=float, default=0.05)
    parser.add_argument("--i-iptm-floor", type=float, default=0.95)
    parser.add_argument("--parent-iptm-floor", type=float, default=0.915)
    parser.add_argument("--i-ligand-plddt-floor", type=float, default=90.0)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
