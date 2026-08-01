#!/usr/bin/env python3
"""Analyze LuxSit parent-background free-3 Protenix-DTZ holo ensembles.

Implements PROTOCOL/luxsit_parent_variant_holo_selection.md. The analysis is
read-only with respect to the source run and writes a self-contained result
bundle under ``analysis/parent_variant_holo`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.evaluation.esmfold_runner import cache_key


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})
INVARIANT19 = (13, 14, 17, 18, 35, 37, 38, 43, 49, 52, 53, 56, 65, 81, 83, 94, 98, 100, 112)
CATALYTIC4 = (14, 18, 65, 98)
FREE3 = (60, 96, 110)
PARENT_AA = {60: "R", 96: "A", 110: "M"}
I_AA = {60: "S", 96: "L", 110: "V"}
DTZ_CORE_ATOMS = ("O1", "C1", "C2", "N1", "C10", "C11", "N2", "C18", "C25", "N3")
VDW_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}
SYMMETRY_SWAPS = {
    "ASP": (("OD1", "OD2"),),
    "GLU": (("OE1", "OE2"),),
    "ARG": (("NH1", "NH2"),),
    "VAL": (("CG1", "CG2"),),
    "LEU": (("CD1", "CD2"),),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
}
SAMPLE_RE = re.compile(r"_sample_(\d+)\.cif$")


@dataclass(frozen=True, eq=False)
class Atom:
    group: str
    element: str
    name: str
    resname: str
    chain: str
    seq: int
    coord: np.ndarray
    bfactor: float = math.nan


@dataclass
class Structure:
    residues: dict[int, dict[str, Atom]]
    residue_names: dict[int, str]
    ligand: dict[str, Atom]

    @property
    def protein_atoms(self) -> list[Atom]:
        return [atom for atoms in self.residues.values() for atom in atoms.values()]

    @property
    def ligand_atoms(self) -> list[Atom]:
        return list(self.ligand.values())

    @property
    def sequence(self) -> str:
        return "".join(THREE_TO_ONE.get(self.residue_names[pos], "X") for pos in sorted(self.residue_names))


@dataclass(frozen=True)
class RigidFrame:
    rotation: np.ndarray
    mobile_center: np.ndarray
    target_center: np.ndarray

    def apply(self, coords: np.ndarray) -> np.ndarray:
        arr = np.asarray(coords, dtype=float)
        return (arr - self.mobile_center) @ self.rotation + self.target_center


def _clean(value: object) -> str:
    text = str(value).strip()
    if text in {".", "?", "None"}:
        return ""
    return text.strip("'\"")


def _iter_cif_tokens(path: Path) -> Iterable[str]:
    in_text_block = False
    with path.open() as handle:
        for raw_line in handle:
            if raw_line.startswith(";"):
                in_text_block = not in_text_block
                continue
            if in_text_block:
                continue
            line = raw_line.strip()
            if line and line != "#":
                yield from shlex.split(line, comments=True, posix=True)


def _atom_site_rows(path: Path) -> list[dict[str, str]]:
    tokens = list(_iter_cif_tokens(path))
    idx = 0
    while idx < len(tokens):
        if tokens[idx] != "loop_":
            idx += 1
            continue
        idx += 1
        headers: list[str] = []
        while idx < len(tokens) and tokens[idx].startswith("_"):
            headers.append(tokens[idx])
            idx += 1
        if not headers or not any(h.startswith("_atom_site.") for h in headers):
            continue
        n_headers = len(headers)
        rows: list[dict[str, str]] = []
        while idx + n_headers <= len(tokens):
            if tokens[idx] == "loop_" or tokens[idx].startswith("_"):
                break
            rows.append(dict(zip(headers, tokens[idx: idx + n_headers], strict=True)))
            idx += n_headers
        return rows
    raise ValueError(f"{path}: no _atom_site loop")


def parse_mmcif(path: str | Path) -> Structure:
    path = Path(path)
    protein: list[Atom] = []
    ligand: list[Atom] = []
    for row in _atom_site_rows(path):
        group = _clean(row.get("_atom_site.group_PDB", ""))
        altloc = _clean(row.get("_atom_site.label_alt_id", ""))
        if altloc not in {"", "A"}:
            continue
        atom_name = _clean(row.get("_atom_site.label_atom_id") or row.get("_atom_site.auth_atom_id"))
        resname = _clean(row.get("_atom_site.label_comp_id") or row.get("_atom_site.auth_comp_id")).upper()
        chain = _clean(row.get("_atom_site.auth_asym_id") or row.get("_atom_site.label_asym_id"))
        seq_text = _clean(row.get("_atom_site.auth_seq_id") or row.get("_atom_site.label_seq_id"))
        element = _clean(row.get("_atom_site.type_symbol", atom_name[:1])).upper()
        try:
            seq = int(float(seq_text))
            coord = np.array([
                float(_clean(row["_atom_site.Cartn_x"])),
                float(_clean(row["_atom_site.Cartn_y"])),
                float(_clean(row["_atom_site.Cartn_z"])),
            ])
        except (KeyError, ValueError):
            continue
        try:
            bfactor = float(_clean(row.get("_atom_site.B_iso_or_equiv", "nan")))
        except ValueError:
            bfactor = math.nan
        atom = Atom(group, element, atom_name, resname, chain, seq, coord, bfactor)
        if group == "ATOM" and 1 <= seq <= 117:
            protein.append(atom)
        elif group == "HETATM" and element != "H":
            ligand.append(atom)
    if not protein:
        raise ValueError(f"{path}: no protein atoms")
    chain_counts: dict[str, int] = {}
    for atom in protein:
        chain_counts[atom.chain] = chain_counts.get(atom.chain, 0) + 1
    protein_chain = max(chain_counts, key=chain_counts.get)
    protein = [atom for atom in protein if atom.chain == protein_chain and atom.element != "H"]
    residues: dict[int, dict[str, Atom]] = {}
    residue_names: dict[int, str] = {}
    for atom in protein:
        residues.setdefault(atom.seq, {})[atom.name] = atom
        residue_names[atom.seq] = atom.resname
    ligand_by_name = {atom.name: atom for atom in ligand}
    return Structure(residues, residue_names, ligand_by_name)


def parse_pdb(path: str | Path) -> Structure:
    path = Path(path)
    atoms: list[Atom] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM") or len(line) < 54:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "A"
            try:
                seq = int(line[22:26])
                coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            element = line[76:78].strip().upper() if len(line) >= 78 else atom_name[:1].upper()
            try:
                bfactor = float(line[60:66])
            except ValueError:
                bfactor = math.nan
            if element != "H" and 1 <= seq <= 117:
                atoms.append(Atom("ATOM", element, atom_name, resname, chain, seq, coord, bfactor))
    if not atoms:
        raise ValueError(f"{path}: no protein atoms")
    chain_counts: dict[str, int] = {}
    for atom in atoms:
        chain_counts[atom.chain] = chain_counts.get(atom.chain, 0) + 1
    protein_chain = max(chain_counts, key=chain_counts.get)
    residues: dict[int, dict[str, Atom]] = {}
    residue_names: dict[int, str] = {}
    for atom in atoms:
        if atom.chain != protein_chain:
            continue
        residues.setdefault(atom.seq, {})[atom.name] = atom
        residue_names[atom.seq] = atom.resname
    return Structure(residues, residue_names, {})


def parse_structure(path: str | Path) -> Structure:
    path = Path(path)
    return parse_mmcif(path) if path.suffix.lower() in {".cif", ".mmcif"} else parse_pdb(path)


def fit_frame(mobile: np.ndarray, target: np.ndarray) -> RigidFrame:
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"invalid alignment shapes: mobile={mobile.shape}, target={target.shape}")
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    u_mat, _singular, vt_mat = np.linalg.svd((mobile - mobile_center).T @ (target - target_center))
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u_mat @ vt_mat)) or 1.0
    return RigidFrame(u_mat @ correction @ vt_mat, mobile_center, target_center)


def alignment_frame(mobile: Structure, target: Structure, positions: Sequence[int]) -> RigidFrame:
    mobile_ca = np.vstack([mobile.residues[pos]["CA"].coord for pos in positions])
    target_ca = np.vstack([target.residues[pos]["CA"].coord for pos in positions])
    return fit_frame(mobile_ca, target_ca)


def ca_rmsd(mobile: Structure, target: Structure, positions: Sequence[int], frame: RigidFrame) -> float:
    mobile_ca = np.vstack([mobile.residues[pos]["CA"].coord for pos in positions])
    target_ca = np.vstack([target.residues[pos]["CA"].coord for pos in positions])
    return float(np.sqrt(np.mean(np.sum((frame.apply(mobile_ca) - target_ca) ** 2, axis=1))))


def _sidechain_names(structure: Structure, position: int) -> set[str]:
    return {
        name for name, atom in structure.residues[position].items()
        if name not in BACKBONE_ATOMS and atom.element != "H"
    }


def residue_sidechain_errors(
    mobile: Structure,
    target: Structure,
    position: int,
    frame: RigidFrame,
) -> tuple[np.ndarray, float] | None:
    if mobile.residue_names[position] != target.residue_names[position]:
        return None
    names = sorted(_sidechain_names(mobile, position) & _sidechain_names(target, position))
    if not names:
        return None
    mappings = [{name: name for name in names}]
    swaps = SYMMETRY_SWAPS.get(target.residue_names[position], ())
    if swaps:
        swapped = {name: name for name in names}
        for left, right in swaps:
            if left in swapped and right in swapped:
                swapped[left], swapped[right] = right, left
        mappings.append(swapped)
    best_errors: np.ndarray | None = None
    best_rmsd = math.inf
    for mapping in mappings:
        mobile_coords = np.vstack([mobile.residues[position][name].coord for name in names])
        target_coords = np.vstack([target.residues[position][mapping[name]].coord for name in names])
        errors = np.sum((frame.apply(mobile_coords) - target_coords) ** 2, axis=1)
        rmsd = float(np.sqrt(np.mean(errors)))
        if rmsd < best_rmsd:
            best_errors, best_rmsd = errors, rmsd
    assert best_errors is not None
    return best_errors, best_rmsd


def sidechain_summary(
    mobile: Structure,
    target: Structure,
    positions: Sequence[int],
    frame: RigidFrame,
) -> dict[str, object]:
    all_errors: list[float] = []
    per_residue: dict[int, float] = {}
    for position in positions:
        result = residue_sidechain_errors(mobile, target, position, frame)
        if result is None:
            continue
        errors, rmsd = result
        all_errors.extend(float(value) for value in errors)
        per_residue[position] = rmsd
    if not all_errors:
        return {"rmsd": math.nan, "max_rmsd": math.nan, "worst_position": None, "per_residue": {}}
    worst_position = max(per_residue, key=per_residue.get)
    return {
        "rmsd": float(np.sqrt(np.mean(all_errors))),
        "max_rmsd": float(per_residue[worst_position]),
        "worst_position": int(worst_position),
        "per_residue": per_residue,
    }


def genotype_fields(sequence: str) -> dict[str, object]:
    if len(sequence) != 117:
        raise ValueError(f"expected 117-aa LuxSit core, got {len(sequence)}")
    aas = {position: sequence[position - 1] for position in FREE3}
    n_i = sum(aas[position] == I_AA[position] for position in FREE3)
    n_parent = sum(aas[position] == PARENT_AA[position] for position in FREE3)
    n_alternative = 3 - n_i - n_parent
    if n_alternative:
        genotype_class = "alternative"
    elif n_i == 0:
        genotype_class = "exact_parent"
    elif n_i == 3:
        genotype_class = "exact_i"
    else:
        genotype_class = f"clean_partial_{n_i}"
    return {
        "aa60": aas[60],
        "aa96": aas[96],
        "aa110": aas[110],
        "free3_genotype": "".join(aas[position] for position in FREE3),
        "genotype_class": genotype_class,
        "n_i_recovered": int(n_i),
        "n_parent_retained": int(n_parent),
        "n_alternative": int(n_alternative),
    }


def atom_distance(left: Atom, right: Atom) -> float:
    return float(np.linalg.norm(left.coord - right.coord))


def functional_geometry(structure: Structure) -> dict[str, float]:
    ligand = structure.ligand
    residues = structure.residues
    required = ((14, "OH"), (98, "ND1"), (98, "NE2"), (65, "CZ"), (65, "NE"), (65, "N"))
    if any(name not in residues[pos] for pos, name in required) or not {"O1", "N1"} <= set(ligand):
        raise ValueError("structure lacks required LuxSit catalytic atoms")
    d18_oxygens = [residues[18][name] for name in ("OD1", "OD2") if name in residues[18]]
    r65_terminals = [residues[65][name] for name in ("NH1", "NH2") if name in residues[65]]
    return {
        "Y14OH_H98ND1": atom_distance(residues[14]["OH"], residues[98]["ND1"]),
        "H98NE2_DTZ_O1": atom_distance(residues[98]["NE2"], ligand["O1"]),
        "R65CZ_DTZ_N1": atom_distance(residues[65]["CZ"], ligand["N1"]),
        "R65NH_DTZ_N1": min(atom_distance(atom, ligand["N1"]) for atom in r65_terminals),
        "D18O_R65NE_min": min(atom_distance(atom, residues[65]["NE"]) for atom in d18_oxygens),
        "D18O_R65N_min": min(atom_distance(atom, residues[65]["N"]) for atom in d18_oxygens),
    }


def geometry_deviation(values: Mapping[str, float], reference: Mapping[str, float]) -> tuple[float, float]:
    deviations = np.array([abs(values[key] - reference[key]) for key in reference], dtype=float)
    return float(deviations.mean()), float(deviations.max())


def ligand_pose_metrics(mobile: Structure, target: Structure, frame: RigidFrame) -> dict[str, float]:
    names = sorted(set(mobile.ligand) & set(target.ligand))
    if len(names) != 29:
        raise ValueError(f"expected 29 matched DTZ atoms, got {len(names)}")
    mobile_coords = frame.apply(np.vstack([mobile.ligand[name].coord for name in names]))
    target_coords = np.vstack([target.ligand[name].coord for name in names])
    full_rmsd = float(np.sqrt(np.mean(np.sum((mobile_coords - target_coords) ** 2, axis=1))))
    index = {name: idx for idx, name in enumerate(names)}
    core_mobile = np.vstack([mobile_coords[index[name]] for name in DTZ_CORE_ATOMS])
    core_target = np.vstack([target_coords[index[name]] for name in DTZ_CORE_ATOMS])
    core_rmsd = float(np.sqrt(np.mean(np.sum((core_mobile - core_target) ** 2, axis=1))))
    com_displacement = float(np.linalg.norm(mobile_coords.mean(axis=0) - target_coords.mean(axis=0)))
    return {"ligand_full_rmsd": full_rmsd, "ligand_core_rmsd": core_rmsd, "ligand_com": com_displacement}


def residue_contact_fingerprint(structure: Structure, cutoff: float = 4.5) -> set[int]:
    ligand_coords = np.vstack([atom.coord for atom in structure.ligand_atoms])
    contacts: set[int] = set()
    for position, atoms in structure.residues.items():
        coords = np.vstack([atom.coord for atom in atoms.values() if atom.element != "H"])
        if np.min(np.linalg.norm(coords[:, None, :] - ligand_coords[None, :, :], axis=2)) <= cutoff:
            contacts.add(position)
    return contacts


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _radius(atom: Atom) -> float:
    return VDW_RADII.get(atom.element.upper(), 1.70)


def fibonacci_sphere(n_points: int = 64) -> np.ndarray:
    indices = np.arange(n_points, dtype=float) + 0.5
    z_coord = 1.0 - 2.0 * indices / n_points
    radial = np.sqrt(np.maximum(0.0, 1.0 - z_coord * z_coord))
    theta = math.pi * (1.0 + math.sqrt(5.0)) * indices
    return np.column_stack((radial * np.cos(theta), radial * np.sin(theta), z_coord))


SPHERE64 = fibonacci_sphere(64)
SPHERE32 = fibonacci_sphere(32)


def _sidechain_atoms(structure: Structure, position: int) -> list[Atom]:
    return [
        atom for name, atom in structure.residues[position].items()
        if name not in BACKBONE_ATOMS and atom.element != "H"
    ]


def atom_set_sasa(
    targets: Sequence[Atom],
    occluders: Sequence[Atom],
    *,
    probe: float = 1.4,
    sphere: np.ndarray = SPHERE64,
) -> float:
    """Approximate target-atom SASA with a Shrake-Rupley point surface."""
    if not targets:
        return math.nan
    occluder_coords = np.vstack([atom.coord for atom in occluders])
    occluder_radii = np.array([_radius(atom) + probe for atom in occluders])
    tree = cKDTree(occluder_coords)
    total = 0.0
    for atom in targets:
        radius = _radius(atom) + probe
        points = atom.coord + radius * sphere
        neighbors = tree.query_ball_point(atom.coord, radius + float(occluder_radii.max()))
        visible = np.ones(len(points), dtype=bool)
        for index in neighbors:
            other = occluders[index]
            if other is atom:
                continue
            visible &= np.linalg.norm(points - other.coord, axis=1) >= occluder_radii[index]
            if not visible.any():
                break
        total += float(visible.mean() * 4.0 * math.pi * radius * radius)
    return total


def overlap_summary(left: Sequence[Atom], right: Sequence[Atom]) -> dict[str, float | int]:
    if not left or not right:
        return {"overlap_sum": 0.0, "overlap_max": 0.0, "severe_clash_count": 0}
    left_coords = np.vstack([atom.coord for atom in left])
    right_coords = np.vstack([atom.coord for atom in right])
    distances = np.linalg.norm(left_coords[:, None, :] - right_coords[None, :, :], axis=2)
    radii = (
        np.array([_radius(atom) for atom in left])[:, None]
        + np.array([_radius(atom) for atom in right])[None, :]
    )
    overlap = np.maximum(0.0, radii - distances)
    return {
        "overlap_sum": float(overlap.sum()),
        "overlap_max": float(overlap.max(initial=0.0)),
        "severe_clash_count": int(np.count_nonzero(overlap > 0.80)),
    }


def _nearest_surface_gap(points: np.ndarray, atoms: Sequence[Atom], k: int = 8) -> np.ndarray:
    coords = np.vstack([atom.coord for atom in atoms])
    radii = np.array([_radius(atom) for atom in atoms])
    tree = cKDTree(coords)
    n_neighbors = min(k, len(atoms))
    distances, indices = tree.query(points, k=n_neighbors)
    if n_neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return np.min(distances - radii[indices], axis=1)


def ligand_surface_points(structure: Structure) -> np.ndarray:
    ligand_atoms = structure.ligand_atoms
    points: list[np.ndarray] = []
    for atom in ligand_atoms:
        surface = atom.coord + _radius(atom) * SPHERE32
        keep = np.ones(len(surface), dtype=bool)
        for other in ligand_atoms:
            if other is atom:
                continue
            keep &= np.linalg.norm(surface - other.coord, axis=1) >= _radius(other) - 0.05
        points.extend(surface[keep])
    return np.asarray(points, dtype=float)


def ligand_shape_proxies(structure: Structure, grid_spacing: float = 1.5) -> dict[str, float]:
    ligand_points = ligand_surface_points(structure)
    protein_gap = _nearest_surface_gap(ligand_points, structure.protein_atoms)
    contact_coverage = float(np.mean((protein_gap >= -0.40) & (protein_gap <= 1.50)))
    overlap_fraction = float(np.mean(protein_gap < -0.40))

    ligand_atoms = structure.ligand_atoms
    ligand_coords = np.vstack([atom.coord for atom in ligand_atoms])
    lower = ligand_coords.min(axis=0) - 4.0
    upper = ligand_coords.max(axis=0) + 4.0
    axes = [np.arange(lower[i], upper[i] + grid_spacing, grid_spacing) for i in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack([axis.ravel() for axis in mesh])
    ligand_gap = _nearest_surface_gap(grid, ligand_atoms)
    protein_gap_grid = _nearest_surface_gap(grid, structure.protein_atoms)
    free_shell = (ligand_gap >= 0.0) & (ligand_gap <= 4.0) & (protein_gap_grid >= 0.0)
    return {
        "ligand_contact_coverage_proxy": contact_coverage,
        "ligand_surface_overlap_fraction_proxy": overlap_fraction,
        "shape_complementarity_proxy": contact_coverage * (1.0 - overlap_fraction),
        "pocket_shell_free_volume_proxy": float(free_shell.sum() * grid_spacing ** 3),
    }


def site_anchor_network(structure: Structure, position: int, cutoff: float = 4.5) -> set[int]:
    site_atoms = _sidechain_atoms(structure, position) or list(structure.residues[position].values())
    site_coords = np.vstack([atom.coord for atom in site_atoms])
    network: set[int] = set()
    for anchor in INVARIANT19:
        anchor_coords = np.vstack([atom.coord for atom in structure.residues[anchor].values()])
        if np.min(np.linalg.norm(site_coords[:, None, :] - anchor_coords[None, :, :], axis=2)) <= cutoff:
            network.add(anchor)
    return network


def holo_physical_metrics(structure: Structure) -> dict[str, object]:
    ligand_atoms = structure.ligand_atoms
    if len(ligand_atoms) != 29:
        raise ValueError(f"expected 29 DTZ heavy atoms, got {len(ligand_atoms)}")
    protein_atoms = structure.protein_atoms
    all_atoms = protein_atoms + ligand_atoms
    result: dict[str, object] = {}
    global_overlap = overlap_summary(protein_atoms, ligand_atoms)
    result.update({f"protein_ligand_{key}": value for key, value in global_overlap.items()})
    result.update(ligand_shape_proxies(structure))
    result["ligand_contact_positions_json"] = json.dumps(sorted(residue_contact_fingerprint(structure)))
    ligand_coords = np.vstack([atom.coord for atom in ligand_atoms])
    ligand_com = ligand_coords.mean(axis=0)

    for position in FREE3:
        sidechain = _sidechain_atoms(structure, position)
        metric_atoms = sidechain or [structure.residues[position]["CA"]]
        coords = np.vstack([atom.coord for atom in metric_atoms])
        pair_distances = np.linalg.norm(coords[:, None, :] - ligand_coords[None, :, :], axis=2)
        overlap = overlap_summary(metric_atoms, ligand_atoms)
        if sidechain:
            sasa_without_ligand = atom_set_sasa(sidechain, protein_atoms)
            sasa_with_ligand = atom_set_sasa(sidechain, all_atoms)
        else:
            # Gly has no sidechain heavy atom; zero is the physical SASA/burial value.
            sasa_without_ligand = 0.0
            sasa_with_ligand = 0.0
        prefix = f"pos{position}_"
        result.update({
            prefix + "sidechain_heavy_atom_count": len(sidechain),
            prefix + "min_ligand_distance": float(pair_distances.min()),
            prefix + "contacts_4A": int(np.count_nonzero(pair_distances <= 4.0)),
            prefix + "contacts_4p5A": int(np.count_nonzero(pair_distances <= 4.5)),
            prefix + "sidechain_centroid_ligand_com": float(np.linalg.norm(coords.mean(axis=0) - ligand_com)),
            prefix + "sasa_without_ligand": sasa_without_ligand,
            prefix + "sasa_with_ligand": sasa_with_ligand,
            prefix + "ligand_induced_burial": sasa_without_ligand - sasa_with_ligand,
            prefix + "overlap_sum": overlap["overlap_sum"],
            prefix + "overlap_max": overlap["overlap_max"],
            prefix + "severe_clash_count": overlap["severe_clash_count"],
            prefix + "anchor_network_json": json.dumps(sorted(site_anchor_network(structure, position))),
        })
    return result


def _comparison_core(
    mobile: Structure,
    target: Structure,
    *,
    include_ligand: bool,
) -> dict[str, object]:
    positions = tuple(range(1, 118))
    if any(pos not in mobile.residues or pos not in target.residues for pos in positions):
        raise ValueError("comparison requires all 117 LuxSit residues")
    global_frame = alignment_frame(mobile, target, positions)
    pocket_frame = alignment_frame(mobile, target, INVARIANT19)
    invariant = sidechain_summary(mobile, target, INVARIANT19, global_frame)
    catalytic = sidechain_summary(mobile, target, CATALYTIC4, global_frame)
    result: dict[str, object] = {
        "global_ca_rmsd": ca_rmsd(mobile, target, positions, global_frame),
        "pocket_ca_rmsd": ca_rmsd(mobile, target, INVARIANT19, pocket_frame),
        "inv19_sc_rmsd": invariant["rmsd"],
        "inv19_max_sc_rmsd": invariant["max_rmsd"],
        "inv19_worst_position": invariant["worst_position"],
        "cat4_sc_rmsd": catalytic["rmsd"],
    }
    for position in FREE3:
        sidechain = residue_sidechain_errors(mobile, target, position, pocket_frame)
        result[f"pos{position}_identity_matched_sc_rmsd"] = math.nan if sidechain is None else sidechain[1]
    if include_ligand:
        result.update(ligand_pose_metrics(mobile, target, pocket_frame))
        mobile_geometry = functional_geometry(mobile)
        target_geometry = functional_geometry(target)
        geometry_mae, geometry_max = geometry_deviation(mobile_geometry, target_geometry)
        result["geometry_mae"] = geometry_mae
        result["geometry_max"] = geometry_max
        for key, value in mobile_geometry.items():
            result[f"geometry_{key}"] = value
            result[f"geometry_delta_{key}"] = value - target_geometry[key]
        result["contact_jaccard"] = jaccard(
            residue_contact_fingerprint(mobile), residue_contact_fingerprint(target)
        )
    return result


def comparison_metrics(
    mobile: Structure,
    references: Mapping[str, Structure],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, reference in references.items():
        values = _comparison_core(mobile, reference, include_ligand=name.endswith("holo"))
        result.update({f"{name}_{key}": value for key, value in values.items()})
    return result


def _allowed_reference_names(genotype_class: str) -> tuple[str, ...]:
    if genotype_class == "exact_parent":
        return ("parent",)
    if genotype_class == "exact_i":
        return ("i",)
    return ("parent", "i")


def envelope_comparisons(values: Mapping[str, object], genotype_class: str) -> dict[str, object]:
    allowed = _allowed_reference_names(genotype_class)
    lower_better = (
        "global_ca_rmsd", "pocket_ca_rmsd", "inv19_sc_rmsd", "inv19_max_sc_rmsd",
        "cat4_sc_rmsd", "ligand_full_rmsd", "ligand_core_rmsd", "ligand_com",
        "geometry_mae", "geometry_max",
    )
    result: dict[str, object] = {}
    for metric in lower_better:
        candidates = [float(values[f"{name}_holo_{metric}"]) for name in allowed]
        result[f"envelope_{metric}"] = min(candidates)
    result["envelope_contact_jaccard"] = max(
        float(values[f"{name}_holo_contact_jaccard"]) for name in allowed
    )
    for metric in (
        "global_ca_rmsd", "pocket_ca_rmsd", "inv19_sc_rmsd", "inv19_max_sc_rmsd", "cat4_sc_rmsd"
    ):
        result[f"envelope_apo_{metric}"] = min(
            float(values[f"{name}_apo_{metric}"]) for name in allowed
        )
        result[f"apo_holo_reference_shift_{metric}"] = (
            result[f"envelope_apo_{metric}"] - result[f"envelope_{metric}"]
        )
    composite = {
        name: float(values[f"{name}_holo_inv19_sc_rmsd"])
        + float(values[f"{name}_holo_geometry_mae"])
        + float(values[f"{name}_holo_ligand_core_rmsd"])
        for name in allowed
    }
    result["best_functional_reference"] = min(composite, key=composite.get)
    return result


def _network_from_metric(metrics: Mapping[str, object], position: int) -> set[int]:
    raw = metrics[f"pos{position}_anchor_network_json"]
    return {int(value) for value in json.loads(str(raw))}


def packing_metrics(
    candidate: Mapping[str, object],
    references: Mapping[str, Mapping[str, object]],
    genotype: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    passes: list[bool] = []
    disruptions: list[float] = []
    for position in FREE3:
        aa = str(genotype[f"aa{position}"])
        if aa == PARENT_AA[position]:
            names = ("parent",)
        elif aa == I_AA[position]:
            names = ("i",)
        else:
            names = ("parent", "i")
        comparisons: list[tuple[float, str, bool]] = []
        for name in names:
            reference = references[name]
            prefix = f"pos{position}_"
            candidate_network = _network_from_metric(candidate, position)
            reference_network = _network_from_metric(reference, position)
            disruption = float(np.mean([
                abs(float(candidate[prefix + "min_ligand_distance"]) - float(reference[prefix + "min_ligand_distance"])) / 1.0,
                abs(float(candidate[prefix + "sidechain_centroid_ligand_com"]) - float(reference[prefix + "sidechain_centroid_ligand_com"])) / 1.5,
                abs(float(candidate[prefix + "ligand_induced_burial"]) - float(reference[prefix + "ligand_induced_burial"])) / 25.0,
                abs(float(candidate[prefix + "contacts_4p5A"]) - float(reference[prefix + "contacts_4p5A"]))
                / max(2.0, float(reference[prefix + "contacts_4p5A"])),
                1.0 - jaccard(candidate_network, reference_network),
            ]))
            clash_pass = int(candidate[prefix + "severe_clash_count"]) <= int(reference[prefix + "severe_clash_count"])
            if position == 60:
                site_pass = clash_pass
            else:
                ref_distance = float(reference[prefix + "min_ligand_distance"])
                candidate_distance = float(candidate[prefix + "min_ligand_distance"])
                contact_pass = (
                    int(reference[prefix + "contacts_4p5A"]) == 0
                    or int(candidate[prefix + "contacts_4p5A"]) > 0
                )
                site_pass = clash_pass and contact_pass and candidate_distance <= max(5.0, ref_distance + 1.0)
            comparisons.append((disruption, name, site_pass))
        best = min(comparisons, key=lambda item: item[0])
        result[f"pos{position}_packing_disruption"] = best[0]
        result[f"pos{position}_packing_reference"] = best[1]
        result[f"pos{position}_packing_pass"] = bool(best[2])
        disruptions.append(best[0])
        passes.append(bool(best[2]))
    result["free3_packing_disruption"] = float(np.mean(disruptions))
    result["free3_sample_pass"] = all(passes)
    return result


def read_confidence(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    chain_plddt = data.get("chain_plddt", [])
    pair_gpde = data.get("chain_pair_gpde", [])
    return {
        "plddt": float(data["plddt"]),
        "ptm": float(data["ptm"]),
        "iptm": float(data["iptm"]),
        "ligand_plddt": float(chain_plddt[1]) * 100.0 if len(chain_plddt) > 1 else math.nan,
        "protein_plddt": float(chain_plddt[0]) * 100.0 if chain_plddt else math.nan,
        "pair_gpde": float(pair_gpde[0][1]) if len(pair_gpde) > 1 else math.nan,
        "has_clash": bool(data.get("has_clash", False)),
        "ranking_score": float(data.get("ranking_score", math.nan)),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_constraint_manifest(path: Path, designs: pd.DataFrame) -> dict[int, str]:
    manifest = yaml.safe_load(path.read_text())
    entries = manifest.get("entries", [])
    matching = [entry for entry in entries if entry.get("protein_id") == "LuxSit_parent_core117"]
    if len(matching) != 1:
        raise ValueError(f"{path}: expected one LuxSit_parent_core117 entry")
    anchors = {
        int(item["index_0b"]) + 1: str(item["expected_aa"])
        for item in matching[0].get("hard_anchors", [])
    }
    if tuple(sorted(anchors)) != INVARIANT19:
        raise ValueError(f"{path}: hard anchors do not match invariant-19 protocol")
    for row in designs.itertuples(index=False):
        failures = [position for position, aa in anchors.items() if row.sequence[position - 1] != aa]
        if failures:
            raise ValueError(f"design {row.design_idx}: hard anchors changed at {failures}")
    return anchors


def reference_ligand_plddt_floor(references: Mapping[str, Structure]) -> tuple[float, dict[str, float]]:
    means = {
        name: float(np.nanmean([atom.bfactor for atom in reference.ligand_atoms]))
        for name, reference in references.items()
        if name.endswith("holo")
    }
    return max(85.0, min(means.values()) - 5.0), means


def prediction_files(run_dir: Path, protein_id: str, sequence: str) -> list[tuple[int, Path, Path]]:
    key = cache_key(protein_id, sequence)
    root = run_dir / "eval_structure" / "raw_holo" / "predictions" / key
    prediction_dirs = sorted(root.glob("seed_*/predictions"))
    if len(prediction_dirs) != 1:
        raise ValueError(f"{key}: expected one seed prediction directory, got {len(prediction_dirs)}")
    prediction_dir = prediction_dirs[0]
    result: list[tuple[int, Path, Path]] = []
    for sample_idx in range(5):
        structure = prediction_dir / f"{key}_sample_{sample_idx}.cif"
        confidence = prediction_dir / f"{key}_summary_confidence_sample_{sample_idx}.json"
        if not structure.is_file() or not confidence.is_file():
            raise FileNotFoundError(f"{key}: missing sample {sample_idx} structure/confidence")
        result.append((sample_idx, structure, confidence))
    return result


def aggregate_design_metrics(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    identity_columns = [
        "protein_id", "design_id", "design_idx", "sequence", "seed", "aa60", "aa96", "aa110",
        "free3_genotype", "genotype_class", "n_i_recovered", "n_parent_retained", "n_alternative",
    ]
    excluded_numeric = {"design_idx", "seed", "sample_idx", "n_i_recovered", "n_parent_retained", "n_alternative"}
    rows: list[dict[str, object]] = []
    for _design_idx, group in sample_metrics.groupby("design_idx", sort=True):
        row = {column: group.iloc[0][column] for column in identity_columns}
        for column in group.columns:
            if column in identity_columns or column in {"sample_idx", "structure_path", "confidence_path"}:
                continue
            series = group[column]
            if pd.api.types.is_bool_dtype(series):
                row[f"{column}_pass_count"] = int(series.sum())
            elif pd.api.types.is_numeric_dtype(series) and column not in excluded_numeric:
                values = pd.to_numeric(series, errors="coerce")
                row[f"{column}_median"] = float(values.median())
                row[f"{column}_q80"] = float(values.quantile(0.80))
                row[f"{column}_min"] = float(values.min())
                row[f"{column}_max"] = float(values.max())
            elif column not in row:
                modes = series.dropna().mode()
                row[column] = modes.iloc[0] if not modes.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def apply_gates(
    designs: pd.DataFrame,
    *,
    iptm_floor: float,
    ligand_plddt_floor: float,
) -> pd.DataFrame:
    result = designs.copy()
    result["G0_integrity_confidence"] = result["sample_integrity_pass_pass_count"] >= 4
    result["G1_global_fold"] = result["envelope_global_ca_rmsd_median"] <= 1.20
    result["G2_invariant19"] = (
        (result["envelope_inv19_sc_rmsd_median"] <= 1.00)
        & (result["envelope_cat4_sc_rmsd_median"] <= 1.00)
        & (result["envelope_inv19_max_sc_rmsd_median"] <= 2.00)
    )
    result["G3_free3_packing"] = result["free3_sample_pass_pass_count"] >= 4
    result["G4_holo_function"] = (
        (result["envelope_geometry_mae_median"] <= 0.50)
        & (result["envelope_geometry_max_median"] <= 1.00)
        & (result["envelope_ligand_com_median"] <= 1.00)
        & (result["envelope_ligand_core_rmsd_median"] <= 1.00)
        & (result["envelope_ligand_full_rmsd_median"] <= 2.50)
    )
    result["G5_ensemble_confidence"] = (
        (result["iptm_median"] >= iptm_floor)
        & (result["ligand_plddt_median"] >= ligand_plddt_floor)
    )
    q80_floors = {
        "envelope_global_ca_rmsd_q80": 1.20,
        "envelope_inv19_sc_rmsd_q80": 1.00,
        "envelope_cat4_sc_rmsd_q80": 1.00,
        "envelope_inv19_max_sc_rmsd_q80": 2.00,
        "envelope_geometry_mae_q80": 0.50,
        "envelope_geometry_max_q80": 1.00,
        "envelope_ligand_com_q80": 1.00,
        "envelope_ligand_core_rmsd_q80": 1.00,
        "envelope_ligand_full_rmsd_q80": 2.50,
    }
    result["q80_instability_flag_count"] = sum(
        (result[column] > floor).astype(int) for column, floor in q80_floors.items()
    )
    result["q80_instability_excess"] = sum(
        np.maximum(0.0, result[column] - floor) / floor for column, floor in q80_floors.items()
    )
    result["q80_manual_review"] = result["q80_instability_flag_count"] > 0
    gate_columns = [f"G{index}_{name}" for index, name in enumerate((
        "integrity_confidence", "global_fold", "invariant19", "free3_packing",
        "holo_function", "ensemble_confidence",
    ))]
    result["all_structure_gates_pass"] = result[gate_columns].all(axis=1)
    result["first_failed_gate"] = result.apply(
        lambda row: next((column.split("_", 1)[0] for column in gate_columns if not bool(row[column])), "PASS"),
        axis=1,
    )
    return result


def robust_structure_score(designs: pd.DataFrame) -> pd.DataFrame:
    result = designs.copy()
    result["confidence_instability"] = (
        (1.0 - result["iptm_median"])
        + np.maximum(0.0, result["iptm_max"] - result["iptm_min"])
        + np.maximum(0.0, result["envelope_ligand_core_rmsd_q80"] - result["envelope_ligand_core_rmsd_median"])
        + 0.25 * result["q80_instability_excess"]
    )
    components = {
        "envelope_inv19_sc_rmsd_median": 0.20,
        "envelope_cat4_sc_rmsd_median": 0.15,
        "envelope_geometry_mae_median": 0.20,
        "envelope_ligand_core_rmsd_median": 0.10,
        "envelope_ligand_com_median": 0.10,
        "envelope_pocket_ca_rmsd_median": 0.10,
        "free3_packing_disruption_median": 0.10,
        "confidence_instability": 0.05,
    }
    calibration = result[result["all_structure_gates_pass"]]
    if len(calibration) < 4:
        calibration = result[result[["G0_integrity_confidence", "G1_global_fold", "G2_invariant19", "G3_free3_packing", "G4_holo_function"]].all(axis=1)]
    if len(calibration) < 4:
        calibration = result
    raw_score = np.zeros(len(result), dtype=float)
    for column, weight in components.items():
        median = float(calibration[column].median())
        iqr = float(calibration[column].quantile(0.75) - calibration[column].quantile(0.25))
        scale = iqr if iqr > 1e-8 else max(float(calibration[column].std()), 1.0)
        normalized_column = f"score_z_{column}"
        result[normalized_column] = (result[column] - median) / scale
        raw_score += weight * result[normalized_column].to_numpy(dtype=float)
    result["structure_score_raw"] = raw_score
    pass_scores = result.loc[result["all_structure_gates_pass"], "structure_score_raw"]
    offset = float(pass_scores.min()) if not pass_scores.empty else float(np.nanmin(raw_score))
    result["structure_score"] = result["structure_score_raw"] - offset
    result["structure_rank"] = pd.NA
    passing_order = result[result["all_structure_gates_pass"]].sort_values(
        ["structure_score", "n_strong_binders", "design_idx"], kind="stable"
    ).index
    result.loc[passing_order, "structure_rank"] = np.arange(1, len(passing_order) + 1)
    return result


def select_shortlist(designs: pd.DataFrame, n_primary: int = 8, n_reserve: int = 4) -> pd.DataFrame:
    passing = designs[designs["all_structure_gates_pass"]].sort_values(
        ["structure_score", "n_strong_binders", "design_idx"], kind="stable"
    ).copy()
    primary = passing.head(n_primary).copy()
    primary["selection_tier"] = "primary"
    primary["selection_reason"] = "strict-gate pass; structure rank"

    remaining = passing.iloc[len(primary):].copy()
    reserve = remaining.head(n_reserve).copy()
    if len(primary) == n_primary and not reserve.empty:
        cutoff = float(primary["structure_score"].max()) + 0.5
        interpretable = remaining[
            remaining["genotype_class"].isin(["exact_i", "clean_partial_1", "clean_partial_2"])
            & (remaining["structure_score"] <= cutoff)
        ]
        represented = set(primary["genotype_class"]) | set(reserve["genotype_class"])
        for genotype_class in ("exact_i", "clean_partial_2", "clean_partial_1"):
            if genotype_class in represented:
                continue
            candidate = interpretable[interpretable["genotype_class"] == genotype_class].head(1)
            if candidate.empty:
                continue
            if len(reserve) < n_reserve:
                reserve = pd.concat([reserve, candidate]).drop_duplicates("design_idx").head(n_reserve)
            else:
                reserve = pd.concat([reserve.iloc[:-1], candidate]).drop_duplicates("design_idx").head(n_reserve)
            represented.add(genotype_class)
    reserve["selection_tier"] = "reserve"
    reserve["selection_reason"] = "strict-gate pass; structure-competitive reserve"
    shortlist = pd.concat([primary, reserve], ignore_index=True)
    shortlist.loc[shortlist["q80_manual_review"], "selection_reason"] += "; q80 manual review"
    shortlist["shortlist_rank"] = np.arange(1, len(shortlist) + 1)
    return shortlist


def write_fasta(table: pd.DataFrame, path: Path) -> None:
    with path.open("w") as handle:
        for row in table.itertuples(index=False):
            rank = getattr(row, "shortlist_rank", getattr(row, "structure_rank", "NA"))
            handle.write(
                f">{row.design_id}|rank={rank}|tier={getattr(row, 'selection_tier', 'ranked')}"
                f"|genotype={row.free3_genotype}|score={row.structure_score:.4f}\n{row.sequence}\n"
            )


def genotype_outputs(designs: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for (genotype_class, genotype), group in designs.groupby(["genotype_class", "free3_genotype"], sort=False):
        rows.append({
            "genotype_class": genotype_class,
            "free3_genotype": genotype,
            "count": len(group),
            "fraction": len(group) / len(designs),
            "strict_gate_pass_count": int(group["all_structure_gates_pass"].sum()),
            "strict_gate_pass_fraction": float(group["all_structure_gates_pass"].mean()),
            "median_structure_score": float(group["structure_score"].median()),
            "median_global_ca_rmsd": float(group["envelope_global_ca_rmsd_median"].median()),
            "median_invariant19_sc_rmsd": float(group["envelope_inv19_sc_rmsd_median"].median()),
            "median_ligand_core_rmsd": float(group["envelope_ligand_core_rmsd_median"].median()),
            "median_geometry_mae": float(group["envelope_geometry_mae_median"].median()),
        })
    pd.DataFrame(rows).sort_values(["count", "genotype_class"], ascending=[False, True]).to_csv(
        output_dir / "genotype_summary.csv", index=False
    )
    recovery: dict[str, object] = {
        "n_designs": len(designs),
        "sites": {},
        "n_i_recovered_distribution": {
            str(key): int(value) for key, value in designs["n_i_recovered"].value_counts().sort_index().items()
        },
        "n_parent_retained_distribution": {
            str(key): int(value) for key, value in designs["n_parent_retained"].value_counts().sort_index().items()
        },
    }
    marginal_i: list[float] = []
    for position in FREE3:
        frequencies = designs[f"aa{position}"].value_counts(normalize=True).sort_values(ascending=False)
        i_frequency = float(frequencies.get(I_AA[position], 0.0))
        marginal_i.append(i_frequency)
        recovery["sites"][str(position)] = {
            "parent_aa": PARENT_AA[position],
            "luxsit_i_aa": I_AA[position],
            "luxsit_i_count": int((designs[f"aa{position}"] == I_AA[position]).sum()),
            "luxsit_i_frequency": i_frequency,
            "parent_count": int((designs[f"aa{position}"] == PARENT_AA[position]).sum()),
            "aa_counts": {str(key): int(value) for key, value in designs[f"aa{position}"].value_counts().items()},
        }
    expected_exact_i = len(designs) * float(np.prod(marginal_i))
    observed_exact_i = int((designs["genotype_class"] == "exact_i").sum())
    pairwise: dict[str, object] = {}
    for left, right in ((60, 96), (60, 110), (96, 110)):
        left_i = designs[f"aa{left}"] == I_AA[left]
        right_i = designs[f"aa{right}"] == I_AA[right]
        expected = len(designs) * float(left_i.mean()) * float(right_i.mean())
        observed = int((left_i & right_i).sum())
        pairwise[f"{left}_{right}"] = {
            "observed_count": observed,
            "independence_expected_count": expected,
            "observed_over_expected": observed / expected if expected > 0 else None,
        }
    recovery.update({
        "observed_exact_i_count": observed_exact_i,
        "independence_expected_exact_i_count": expected_exact_i,
        "exact_i_enrichment_observed_over_expected": (
            observed_exact_i / expected_exact_i if expected_exact_i > 0 else None
        ),
        "pairwise_i_co_recovery": pairwise,
        "interpretation": "pipeline-level spontaneous recovery; token-level causal attribution unavailable",
    })
    (output_dir / "genotype_recovery.json").write_text(json.dumps(recovery, indent=2) + "\n")
    return recovery


def apo_holo_sensitivity(designs: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    metrics = {
        "global_ca_rmsd": 1.20,
        "inv19_sc_rmsd": 1.00,
        "cat4_sc_rmsd": 1.00,
        "inv19_max_sc_rmsd": 2.00,
    }
    table = designs[["protein_id", "design_id", "design_idx", "genotype_class", "free3_genotype"]].copy()
    summary: dict[str, object] = {"metrics": {}}
    concordances: list[float] = []
    correlations: list[float] = []
    absolute_shifts: list[float] = []
    for metric, floor in metrics.items():
        holo = designs[f"envelope_{metric}_median"]
        apo = designs[f"envelope_apo_{metric}_median"]
        table[f"holo_reference_{metric}"] = holo
        table[f"apo_reference_{metric}"] = apo
        table[f"apo_minus_holo_{metric}"] = apo - holo
        table[f"gate_concordant_{metric}"] = (holo <= floor) == (apo <= floor)
        correlation = float(spearmanr(holo, apo, nan_policy="omit").statistic)
        concordance = float(table[f"gate_concordant_{metric}"].mean())
        median_abs_shift = float((apo - holo).abs().median())
        correlations.append(correlation)
        concordances.append(concordance)
        absolute_shifts.append(median_abs_shift)
        summary["metrics"][metric] = {
            "spearman": correlation,
            "gate_concordance": concordance,
            "median_apo_minus_holo": float((apo - holo).median()),
            "median_absolute_shift": median_abs_shift,
        }
    summary["overall"] = {
        "minimum_spearman": min(correlations),
        "minimum_gate_concordance": min(concordances),
        "maximum_median_absolute_shift": max(absolute_shifts),
        "reference_choice_negligible": (
            min(correlations) >= 0.90
            and min(concordances) >= 0.90
            and max(absolute_shifts) <= 0.15
        ),
        "interpretation": "same candidate holo coordinates against apo vs holo WT references; not induced fit",
    }
    table.to_csv(output_dir / "apo_holo_reference_sensitivity.csv", index=False)
    (output_dir / "apo_holo_reference_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def _git_sha(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() or None


def write_readme(
    output_dir: Path,
    designs: pd.DataFrame,
    shortlist: pd.DataFrame,
    recovery: Mapping[str, object],
    sensitivity: Mapping[str, object],
    *,
    iptm_floor: float,
    ligand_plddt_floor: float,
) -> None:
    failed = designs["first_failed_gate"].value_counts().to_dict()
    lines = [
        "# LuxSit parent/variant holo refinement selection",
        "",
        "## Verdict",
        "",
        f"- {int(designs['all_structure_gates_pass'].sum())}/{len(designs)} designs pass all G0-G5 structure gates.",
        f"- First-failure counts: `{json.dumps(failed, sort_keys=True)}`.",
        f"- Exact SLV recovery: {recovery['observed_exact_i_count']} observed vs "
        f"{recovery['independence_expected_exact_i_count']:.2f} expected from marginal frequencies.",
        f"- q80 manual-review flags among the shortlist: {int(shortlist['q80_manual_review'].sum())}.",
        f"- Apo-reference choice negligible: {sensitivity['overall']['reference_choice_negligible']}. "
        "This is reference sensitivity, not candidate apo-to-holo motion.",
        "- Refinement selection is structure-first; immune burden is annotation/tie-break only.",
        "",
        "## Floors",
        "",
        "`global_CA<=1.20 A; invariant19_SC<=1.00 A; catalytic4_SC<=1.00 A; "
        "worst_anchor<=2.00 A; geometry_MAE<=0.50 A; geometry_max<=1.00 A; "
        "DTZ_COM<=1.00 A; DTZ_core<=1.00 A; DTZ_full<=2.50 A; "
        f"median_ipTM>={iptm_floor:.3f}; median_ligand_pLDDT>={ligand_plddt_floor:.2f}`.",
        "",
        "## Shortlist",
        "",
        "| shortlist rank | tier | design | genotype | structure score | global CA | invariant19 SC | DTZ core | geometry MAE | ipTM | q80 flags | NMP strong |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in shortlist.itertuples(index=False):
        lines.append(
            f"| {row.shortlist_rank} | {row.selection_tier} | {row.design_id} | {row.free3_genotype} | "
            f"{row.structure_score:.3f} | {row.envelope_global_ca_rmsd_median:.3f} | "
            f"{row.envelope_inv19_sc_rmsd_median:.3f} | {row.envelope_ligand_core_rmsd_median:.3f} | "
            f"{row.envelope_geometry_mae_median:.3f} | {row.iptm_median:.3f} | "
            f"{int(row.q80_instability_flag_count)} | {int(row.n_strong_binders)} |"
        )
    lines.extend([
        "",
        "## Caveats",
        "",
        "- Candidate apo structures were not generated. `apo_holo_reference_sensitivity.csv` changes only the WT reference.",
        "- SASA, pocket-shell volume, overlap, and shape complementarity are deterministic geometric proxies, not physics energies.",
        "- Positions 60/96/110 never use cross-identity sidechain RMSD.",
        "- Controller token telemetry was not returned, so SLV recovery cannot be assigned to DPLM, D2, or D3 separately.",
        "",
        "See `analysis_manifest.json` and `sample_metrics.parquet` for full provenance.",
    ])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "analysis" / "parent_variant_holo").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_path = run_dir / "generation" / "generated.parquet"
    canonical_path = run_dir / "eval_structure" / "canonical" / "structural.parquet"
    immune_path = run_dir / "eval_immune" / "imm_nmp.parquet"
    designs = pd.read_parquet(generated_path).sort_values("design_idx").reset_index(drop=True)
    if len(designs) != 200 or designs["sequence"].nunique() != 200:
        raise ValueError("expected 200 unique generated sequences")
    if set(designs["protein_id"]) != {"LuxSit_parent_core117"}:
        raise ValueError("run is not the LuxSit parent core117 target")
    designs["design_id"] = designs["design_idx"].map(lambda value: f"design_{int(value):04d}")
    genotype_table = pd.DataFrame([genotype_fields(sequence) for sequence in designs["sequence"]])
    designs = pd.concat([designs, genotype_table], axis=1)

    manifests = sorted((run_dir / "meta").glob("*free_60_96_110*active_site*.yaml"))
    if len(manifests) != 1:
        raise ValueError(f"expected one free-3 constraint manifest, got {manifests}")
    anchors = validate_constraint_manifest(manifests[0], designs)

    reference_dir = run_dir / "eval_structure" / "references"
    reference_paths = {
        "parent_holo": reference_dir / "LuxSit-parent_Protenix_DTZ_holo.cif",
        "i_holo": reference_dir / "LuxSit-i_Protenix_DTZ_holo.cif",
        "parent_apo": reference_dir / "LuxSit-parent_Protenix_apo.pdb",
        "i_apo": reference_dir / "LuxSit-i_Protenix_apo.pdb",
    }
    references = {name: parse_structure(path) for name, path in reference_paths.items()}
    reference_physical = {
        name.removesuffix("_holo"): holo_physical_metrics(reference)
        for name, reference in references.items() if name.endswith("holo")
    }
    ligand_plddt_floor, reference_ligand_plddt = reference_ligand_plddt_floor(references)
    if args.ligand_plddt_floor is not None:
        ligand_plddt_floor = args.ligand_plddt_floor

    sample_rows: list[dict[str, object]] = []
    for design in designs.itertuples(index=False):
        genotype = genotype_fields(design.sequence)
        for sample_idx, structure_path, confidence_path in prediction_files(run_dir, design.protein_id, design.sequence):
            structure = parse_structure(structure_path)
            if structure.sequence != design.sequence:
                raise ValueError(f"{structure_path}: structure sequence differs from generated sequence")
            physical = holo_physical_metrics(structure)
            comparisons = comparison_metrics(structure, references)
            envelope = envelope_comparisons(comparisons, str(genotype["genotype_class"]))
            packing = packing_metrics(physical, reference_physical, genotype)
            confidence = read_confidence(confidence_path)
            sample_integrity = (
                len(structure.residues) == 117
                and len(structure.ligand) == 29
                and not confidence["has_clash"]
                and int(physical["protein_ligand_severe_clash_count"])
                <= max(int(reference_physical[name]["protein_ligand_severe_clash_count"]) for name in reference_physical)
            )
            sample_rows.append({
                "protein_id": design.protein_id,
                "design_id": design.design_id,
                "design_idx": int(design.design_idx),
                "sequence": design.sequence,
                "seed": int(design.seed),
                **genotype,
                "sample_idx": sample_idx,
                "structure_path": str(structure_path),
                "confidence_path": str(confidence_path),
                **confidence,
                **physical,
                **comparisons,
                **envelope,
                **packing,
                "sample_integrity_pass": bool(sample_integrity),
            })
    sample_metrics = pd.DataFrame(sample_rows).sort_values(["design_idx", "sample_idx"])
    if len(sample_metrics) != 1000:
        raise ValueError(f"expected 1000 sample rows, got {len(sample_metrics)}")
    sample_metrics.to_parquet(output_dir / "sample_metrics.parquet", index=False)

    design_metrics = aggregate_design_metrics(sample_metrics)
    canonical = pd.read_parquet(canonical_path)
    canonical_columns = [
        "protein_id", "design_idx", "scTM", "pLDDT", "global_ca_RMSD",
        "active_site_sidechain_RMSD", "max_anchor_sidechain_RMSD",
        "predicted_active_site_min_pLDDT", "recovery",
    ]
    immune = pd.read_parquet(immune_path)[
        ["protein_id", "design_idx", "n_strong_binders", "n_weak_binders", "mean_best_rank"]
    ]
    design_metrics = design_metrics.merge(canonical[canonical_columns], on=["protein_id", "design_idx"], validate="one_to_one")
    design_metrics = design_metrics.merge(immune, on=["protein_id", "design_idx"], validate="one_to_one")
    design_metrics = apply_gates(
        design_metrics, iptm_floor=args.iptm_floor, ligand_plddt_floor=ligand_plddt_floor
    )
    design_metrics = robust_structure_score(design_metrics)
    design_metrics = design_metrics.sort_values(
        ["all_structure_gates_pass", "structure_score", "design_idx"], ascending=[False, True, True]
    ).reset_index(drop=True)

    shortlist = select_shortlist(design_metrics, args.n_primary, args.n_reserve)
    design_metrics.to_parquet(output_dir / "design_metrics.parquet", index=False)
    design_metrics.to_csv(output_dir / "design_metrics.csv", index=False)
    recovery = genotype_outputs(design_metrics, output_dir)
    sensitivity = apo_holo_sensitivity(design_metrics, output_dir)

    shortlist.to_parquet(output_dir / "refinement_shortlist_top12.parquet", index=False)
    shortlist.to_csv(output_dir / "refinement_shortlist_top12.csv", index=False)
    write_fasta(shortlist, output_dir / "refinement_shortlist_top12.fasta")
    primary = shortlist[shortlist["selection_tier"] == "primary"].copy()
    reserve = shortlist[shortlist["selection_tier"] == "reserve"].copy()
    for name, table in (("refinement_primary8", primary), ("refinement_reserve4", reserve)):
        table.to_parquet(output_dir / f"{name}.parquet", index=False)
        table.to_csv(output_dir / f"{name}.csv", index=False)
        write_fasta(table, output_dir / f"{name}.fasta")

    repo_root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": "luxsit_parent_variant_holo_analysis_v1",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "protocol": str(repo_root / "PROTOCOL" / "luxsit_parent_variant_holo_selection.md"),
        "source_files": {
            "generated": str(generated_path),
            "canonical_structure": str(canonical_path),
            "immune_nmp": str(immune_path),
            "constraint_manifest": str(manifests[0]),
        },
        "reference_files": {
            name: {"path": str(path), "sha256": _file_sha256(path)}
            for name, path in reference_paths.items()
        },
        "reference_ligand_plddt": reference_ligand_plddt,
        "floors": {
            "global_ca_rmsd": 1.20,
            "invariant19_sidechain_rmsd": 1.00,
            "catalytic4_sidechain_rmsd": 1.00,
            "worst_invariant_anchor_rmsd": 2.00,
            "geometry_mae": 0.50,
            "geometry_max": 1.00,
            "ligand_com": 1.00,
            "ligand_core_rmsd": 1.00,
            "ligand_full_rmsd": 2.50,
            "iptm": args.iptm_floor,
            "ligand_plddt": ligand_plddt_floor,
        },
        "counts": {
            "designs": len(design_metrics),
            "holo_samples": len(sample_metrics),
            "all_structure_gates_pass": int(design_metrics["all_structure_gates_pass"].sum()),
            "shortlist": len(shortlist),
            "primary": len(primary),
            "reserve": len(reserve),
        },
        "hard_anchors": anchors,
        "git_sha": _git_sha(repo_root),
        "caveats": [
            "Candidate apo was not generated; apo-holo analysis is WT-reference sensitivity only.",
            "SASA, shape complementarity, pocket volume, and overlap are geometric proxies.",
            "No cross-identity sidechain RMSD is computed at positions 60, 96, or 110.",
            "Token-level controller telemetry is absent; genotype recovery is pipeline-level.",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_readme(
        output_dir, design_metrics, shortlist, recovery, sensitivity,
        iptm_floor=args.iptm_floor, ligand_plddt_floor=ligand_plddt_floor,
    )
    print(json.dumps(manifest["counts"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--iptm-floor", type=float, default=0.915,
        help="WT-relative floor: min(parent 0.968, LuxSit-i 0.965) - 0.05",
    )
    parser.add_argument(
        "--ligand-plddt-floor", type=float,
        help="Override the default min(reference DTZ mean B-factor)-5 floor",
    )
    parser.add_argument("--n-primary", type=int, default=8)
    parser.add_argument("--n-reserve", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
