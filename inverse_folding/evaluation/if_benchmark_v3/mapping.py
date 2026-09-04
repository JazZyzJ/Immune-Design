"""C6 residue mapping and IF-ready sequence construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .entities import validate_observed_chain_entity


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}
REQUIRED_BACKBONE = ("N", "CA", "C", "O")


@dataclass(frozen=True)
class MaterializedMapping:
    sequence: str
    final_to_source: tuple[int, ...]
    if_sequence_coverage: float
    mapping_status: str
    mapping_ledger: tuple[dict[str, Any], ...]
    selected_atoms: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MaterializedStructure:
    mapping: MaterializedMapping
    output_path: Path
    structure_sha256: str
    output_chain_id: str
    load_coords_evidence: dict[str, Any]


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mmcif_atom_records(path: Path) -> list[dict[str, Any]]:
    """Read identity-bearing atom_site fields; PDB fallback is intentionally forbidden."""

    path = Path(path)
    if path.suffix.lower() not in {".cif", ".mmcif"}:
        raise ValueError("v3 entity/chain materialization requires mmCIF atom_site identity")
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    payload = MMCIF2Dict(str(path))
    required = {
        "group": "_atom_site.group_PDB", "atom_name": "_atom_site.label_atom_id",
        "alt_id": "_atom_site.label_alt_id", "resname": "_atom_site.label_comp_id",
        "label_asym_id": "_atom_site.label_asym_id",
        "auth_asym_id": "_atom_site.auth_asym_id", "entity_id": "_atom_site.label_entity_id",
        "label_seq_id": "_atom_site.label_seq_id", "occupancy": "_atom_site.occupancy",
        "auth_seq_id": "_atom_site.auth_seq_id",
        "insertion_code": "_atom_site.pdbx_PDB_ins_code",
        "x": "_atom_site.Cartn_x", "y": "_atom_site.Cartn_y", "z": "_atom_site.Cartn_z",
        "model": "_atom_site.pdbx_PDB_model_num",
    }
    missing = [source for source in required.values() if source not in payload]
    if missing:
        raise ValueError(f"mmCIF atom_site missing identity/coordinate fields: {missing}")

    def values(key: str) -> list[str]:
        raw = payload[required[key]]
        return [str(raw)] if isinstance(raw, str) else [str(value) for value in raw]

    columns = {key: values(key) for key in required}
    lengths = {len(value) for value in columns.values()}
    if len(lengths) != 1:
        raise ValueError("mmCIF atom_site columns have inconsistent lengths")
    rows: list[dict[str, Any]] = []
    for index in range(lengths.pop()):
        if columns["model"][index] not in {"1", ".", "?"}:
            continue
        atom_name = columns["atom_name"][index].strip()
        if atom_name not in REQUIRED_BACKBONE:
            continue
        raw_position = columns["label_seq_id"][index]
        if raw_position in {".", "?", ""}:
            continue
        try:
            position = int(raw_position)
            auth_position = int(columns["auth_seq_id"][index])
            occupancy = float(columns["occupancy"][index])
            xyz = [float(columns[axis][index]) for axis in ("x", "y", "z")]
        except ValueError as exc:
            raise ValueError("mmCIF atom_site has invalid position/occupancy/coordinate") from exc
        if not math.isfinite(occupancy) or not all(math.isfinite(value) for value in xyz):
            raise ValueError("mmCIF atom_site has non-finite occupancy/coordinate")
        rows.append({
            "group_PDB": columns["group"][index], "atom_name": atom_name,
            "alt_id": columns["alt_id"][index], "resname": columns["resname"][index],
            "label_asym_id": columns["label_asym_id"][index],
            "auth_asym_id": columns["auth_asym_id"][index],
            "entity_id": columns["entity_id"][index], "label_seq_id": position,
            "auth_seq_id": auth_position,
            "insertion_code": (
                "" if columns["insertion_code"][index] in {".", "?"}
                else columns["insertion_code"][index]
            ),
            "occupancy": occupancy, "x": xyz[0], "y": xyz[1], "z": xyz[2],
        })
    if not rows:
        raise ValueError("mmCIF has no model-1 polymer backbone atom records")
    return rows


def write_materialized_mmcif(
    mapping: MaterializedMapping,
    *, observed_to_source: Mapping[int, int], output_path: Path, output_chain_id: str,
) -> Path:
    """Write only selected complete N/CA/C/O residues in final-sequence order."""

    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".cif", ".mmcif"}:
        raise ValueError("IF-ready structure output must be mmCIF")
    if not output_chain_id or any(character.isspace() for character in output_chain_id):
        raise ValueError("invalid output mmCIF chain ID")
    source_to_final = {source: index + 1 for index, source in enumerate(mapping.final_to_source)}
    observed_map = {int(key): int(value) for key, value in observed_to_source.items()}
    atoms: list[tuple[int, dict[str, Any]]] = []
    for row in mapping.selected_atoms:
        observed = int(row["label_seq_id"])
        source = observed_map[observed]
        if source not in source_to_final:
            raise ValueError("selected atom is absent from final-to-source mapping")
        atoms.append((source_to_final[source], dict(row)))
    atoms.sort(key=lambda item: (item[0], REQUIRED_BACKBONE.index(str(item[1]["atom_name"]))))
    if len(atoms) != 4 * len(mapping.sequence):
        raise ValueError("selected atom count does not match complete final backbone")
    fields = [
        "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.label_alt_id", "_atom_site.label_comp_id",
        "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
        "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge", "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id", "_atom_site.auth_asym_id", "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    lines = ["data_if_benchmark_v3", "#", "loop_", *fields]
    for serial, (residue_index, row) in enumerate(atoms, start=1):
        atom_name = str(row["atom_name"])
        resname = "MET" if str(row["resname"]).upper() == "MSE" else str(row["resname"]).upper()
        element = "C" if atom_name in {"CA", "C"} else atom_name[0]
        lines.append(
            f"ATOM {serial} {element} {atom_name} . {resname} {output_chain_id} 1 "
            f"{residue_index} ? {float(row['x']):.3f} {float(row['y']):.3f} "
            f"{float(row['z']):.3f} 1.00 0.00 ? {residue_index} {resname} "
            f"{output_chain_id} {atom_name} 1"
        )
    lines.append("#")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def _alt_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("alt_id") or "")
    return "" if value in {".", "?"} else value


def _select_complete_atoms(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]] | None:
    explicit = sorted({_alt_id(row) for row in rows if _alt_id(row)})
    candidates = explicit or [""]
    best: tuple[float, str, list[dict[str, Any]]] | None = None
    for alt in candidates:
        allowed = [row for row in rows if _alt_id(row) in {"", alt}]
        chosen: list[dict[str, Any]] = []
        for atom_name in REQUIRED_BACKBONE:
            atom_rows = [row for row in allowed if str(row.get("atom_name")) == atom_name]
            if not atom_rows:
                break
            atom_rows.sort(
                key=lambda row: (
                    -float(row.get("occupancy") if row.get("occupancy") is not None else 0.0),
                    _alt_id(row),
                )
            )
            chosen.append(dict(atom_rows[0]))
        if len(chosen) != len(REQUIRED_BACKBONE):
            continue
        score = sum(float(row.get("occupancy") or 0.0) for row in chosen)
        candidate = (score, alt, chosen)
        if best is None or (-candidate[0], candidate[1]) < (-best[0], best[1]):
            best = candidate
    return None if best is None else best[2]


def materialize_atom_records(
    atom_records: Iterable[Mapping[str, Any]],
    *,
    expected_edge: Mapping[str, Any],
    source_sequence: str,
    observed_to_source: Mapping[int, int],
    denominator_positions: Iterable[int],
    min_coverage: float = 0.8,
) -> MaterializedMapping:
    """Create the exact final sequence from mapped complete-backbone residues.

    ``observed_to_source`` is identity (`label_seq_id - 1`) for Tier 2 and a frozen detailed
    SIFTS map for Tier 1. Every observed residue in the selected entity/chain must appear in this
    map; callers may not hide expression tags or insertions behind a sequence alignment.
    """

    source_sequence = "".join(str(source_sequence).split()).upper()
    denominator = tuple(sorted(set(int(value) for value in denominator_positions)))
    if not source_sequence or not denominator:
        raise ValueError("source sequence and denominator positions are required")
    if any(value < 0 or value >= len(source_sequence) for value in denominator):
        raise ValueError("denominator position outside source sequence")
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be in [0,1]")

    position_map = {int(key): int(value) for key, value in observed_to_source.items()}
    ordered_map = [position_map[key] for key in sorted(position_map)]
    if any(a >= b for a, b in zip(ordered_map, ordered_map[1:])):
        raise ValueError("observed-to-source mapping is not strictly order-preserving")

    expected_label = str(expected_edge["label_asym_id"])
    expected_auth = str(expected_edge["auth_asym_id"])
    selected_chain_rows = [
        dict(row)
        for row in atom_records
        if str(row.get("label_asym_id")) == expected_label
        and str(row.get("auth_asym_id")) == expected_auth
    ]
    if not selected_chain_rows:
        raise ValueError("expected entity chain has no atom records")
    for row in selected_chain_rows:
        validate_observed_chain_entity(
            expected_edge,
            observed_entity_id=str(row.get("entity_id") or ""),
            observed_label_asym_id=str(row.get("label_asym_id") or ""),
            observed_auth_asym_id=str(row.get("auth_asym_id") or ""),
        )

    by_observed: dict[int, list[dict[str, Any]]] = {}
    for row in selected_chain_rows:
        raw_position = row.get("label_seq_id")
        try:
            observed_position = int(raw_position)
        except (TypeError, ValueError) as exc:
            raise ValueError("unmapped observed residue without label_seq_id") from exc
        if observed_position not in position_map:
            raise ValueError(f"unmapped observed residue at label_seq_id={observed_position}")
        source_position = position_map[observed_position]
        if source_position not in denominator:
            raise ValueError(
                f"mapped observed residue lies outside frozen denominator: {source_position}"
            )
        by_observed.setdefault(observed_position, []).append(row)

    complete_by_source: dict[int, tuple[str, list[dict[str, Any]], int]] = {}
    ledger: list[dict[str, Any]] = []
    for observed_position, rows in sorted(by_observed.items()):
        source_position = position_map[observed_position]
        resnames = {str(row.get("resname") or "").upper() for row in rows}
        if len(resnames) != 1:
            raise ValueError(f"conflicting residue names at label_seq_id={observed_position}")
        resname = resnames.pop()
        if resname not in AA3_TO_1:
            raise ValueError(f"unsupported residue {resname} at label_seq_id={observed_position}")
        aa = AA3_TO_1[resname]
        expected_aa = source_sequence[source_position]
        if aa != expected_aa:
            raise ValueError(
                f"amino-acid substitution at source position {source_position}: {aa}!={expected_aa}"
            )
        atoms = _select_complete_atoms(rows)
        if atoms is None:
            ledger.append(
                {
                    "source_position": source_position,
                    "observed_position": observed_position,
                    "amino_acid": aa,
                    "status": "incomplete_backbone",
                }
            )
            continue
        complete_by_source[source_position] = (aa, atoms, observed_position)

    final_positions = sorted(complete_by_source)
    sequence = "".join(complete_by_source[position][0] for position in final_positions)
    for source_position in denominator:
        if source_position in complete_by_source:
            aa, _atoms, observed_position = complete_by_source[source_position]
            ledger.append(
                {
                    "source_position": source_position,
                    "observed_position": observed_position,
                    "amino_acid": aa,
                    "status": "resolved_complete_backbone",
                }
            )
        elif not any(item["source_position"] == source_position for item in ledger):
            ledger.append(
                {
                    "source_position": source_position,
                    "observed_position": None,
                    "amino_acid": source_sequence[source_position],
                    "status": "unresolved_deletion",
                }
            )
    ledger.sort(key=lambda row: row["source_position"])
    coverage = len(final_positions) / len(denominator)
    if coverage < min_coverage:
        raise ValueError(f"IF-ready coverage {coverage:.6f} below {min_coverage:.6f}")
    selected_atoms: list[dict[str, Any]] = []
    for position in final_positions:
        selected_atoms.extend(complete_by_source[position][1])
    return MaterializedMapping(
        sequence=sequence,
        final_to_source=tuple(final_positions),
        if_sequence_coverage=coverage,
        mapping_status="complete",
        mapping_ledger=tuple(ledger),
        selected_atoms=tuple(selected_atoms),
    )


def validate_load_coords_result(
    *, expected_sequence: str, loaded_sequence: str, loaded_length: int
) -> None:
    if int(loaded_length) != len(expected_sequence):
        raise ValueError(
            f"load_coords length {loaded_length} != expected {len(expected_sequence)}"
        )
    if str(loaded_sequence) != str(expected_sequence):
        raise ValueError("load_coords sequence does not exactly match IF-ready bytes")


def materialize_mmcif_chain(
    *, source_mmcif: Path, expected_edge: Mapping[str, Any], source_sequence: str,
    observed_to_source: Mapping[int, int], denominator_positions: Iterable[int],
    output_path: Path, output_chain_id: str,
    load_coords_fn: Any, min_coverage: float = 0.8,
) -> MaterializedStructure:
    """Production C6 adapter: parse identity, materialize, write, and rerun load_coords."""

    atom_records = read_mmcif_atom_records(source_mmcif)
    mapping = materialize_atom_records(
        atom_records, expected_edge=expected_edge, source_sequence=source_sequence,
        observed_to_source=observed_to_source, denominator_positions=denominator_positions,
        min_coverage=min_coverage,
    )
    output_path = write_materialized_mmcif(
        mapping, observed_to_source=observed_to_source,
        output_path=output_path, output_chain_id=output_chain_id,
    ).resolve()
    loaded = load_coords_fn(str(output_path), chain=output_chain_id)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("DPLM load_coords adapter returned an invalid result")
    coords, loaded_sequence = loaded
    loaded_length = len(coords)
    validate_load_coords_result(
        expected_sequence=mapping.sequence, loaded_sequence=str(loaded_sequence),
        loaded_length=loaded_length,
    )
    structure_sha = _sha_file(output_path)
    evidence = {
        "sequence_sha256": hashlib.sha256(mapping.sequence.encode()).hexdigest(),
        "structure_sha256": structure_sha,
        "loaded_sequence_sha256": hashlib.sha256(str(loaded_sequence).encode()).hexdigest(),
        "loaded_length": loaded_length, "status": "pass",
    }
    return MaterializedStructure(
        mapping=mapping, output_path=output_path, structure_sha256=structure_sha,
        output_chain_id=output_chain_id, load_coords_evidence=evidence,
    )
