"""Content-bound residue-level Tier-1 maps from PDBe enriched mmCIF files.

PDBe's enriched mmCIF is the SIFTS authority here.  The producer never expands a
``best_structures`` interval linearly: every UniProt/PDB residue pair must occur in
``_pdbx_sifts_xref_db`` and is cross-checked against the independently downloaded RCSB
coordinate mmCIF before it can enter a C6 attempt.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import http.client
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .preflight import SiftsHttpResponse
from .mapping import read_mmcif_atom_records
from .structures import (
    RcsbMmcifAcquisitionError, _validated_revision_date, validate_cached_rcsb_mmcif,
)


PDBE_ENRICHED_CIF_URL = "https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_id}_updated.cif"
AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUEST_COLUMNS = [
    "pdb_id", "url", "queried_at_utc", "http_status", "attempts_json",
    "response_path", "response_size_bytes", "response_sha256", "response_headers_json",
    "request_error",
    "observed_revision_date", "coordinate_cif_path", "coordinate_manifest_path",
    "coordinate_cif_sha256", "coordinate_manifest_sha256",
]
CANDIDATE_COLUMNS = [
    "candidate_id", "source_id", "sifts_item_index", "pdb_id", "entity_id",
    "rcsb_entity_id", "label_asym_id", "auth_chain_id", "unp_start", "unp_end",
    "source_sequence_sha256", "residue_count", "observed_residue_count",
    "residue_map_json", "raw_sifts_path", "raw_sifts_sha256", "source_revision_date",
    "coordinate_cif_path", "coordinate_cif_sha256", "coordinate_manifest_path",
    "coordinate_manifest_sha256",
]
FAILURE_COLUMNS = [
    "source_id", "sifts_item_index", "pdb_id", "auth_chain_id", "unp_start", "unp_end",
    "failure_code", "failure_detail",
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _as_values(payload: Mapping[str, Any], key: str) -> list[str]:
    if key not in payload:
        raise ValueError(f"PDBe enriched mmCIF lacks required SIFTS field {key}")
    raw = payload[key]
    values = raw if isinstance(raw, list) else [raw]
    return [str(value) for value in values]


def _table(payload: Mapping[str, Any], fields: Mapping[str, str], label: str) -> list[dict[str, str]]:
    columns = {name: _as_values(payload, source) for name, source in fields.items()}
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError(f"{label} columns have inconsistent lengths")
    count = lengths.pop()
    return [{name: values[index] for name, values in columns.items()} for index in range(count)]


def _normal_ins_code(value: Any) -> str:
    raw = str(value or "")
    return "" if raw in {".", "?"} else raw


def _parse_bool(value: Any) -> bool:
    raw = str(value).strip().lower()
    if raw in {"y", "yes", "true", "1"}:
        return True
    if raw in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"invalid SIFTS observed flag: {value!r}")


class Tier1CapacityError(RuntimeError):
    """Raised when a sealed C8 snapshot would carry too few viable Tier 1 source units."""


def _entry_identity(payload: Mapping[str, Any], *, pdb_id: str) -> None:
    """PDBe supplies residue mapping and entry identity -- NOT the coordinate revision.

    Production PDBe enriched mmCIF carries no `_pdbx_audit_revision_history` loop at all, so
    requiring one here rejected every entry. The revision authority is the frozen RCSB source
    snapshot (`frozen_revision_map`), which is content-bound at build time and can therefore be
    cross-checked against what RCSB actually serves.
    """

    entries = _as_values(payload, "_entry.id")
    if len(entries) != 1 or entries[0].upper() != pdb_id.upper():
        raise ValueError("PDBe enriched mmCIF PDB entry identity mismatch")


def _validated_revision(value: Any, *, label: str) -> str:
    """Normalize to the `YYYY-MM-DD` form the RCSB cache keys on.

    The RCSB source snapshot stores an ISO timestamp (`2026-08-12T00:00:00Z`), so this reuses the
    coordinate layer's own normalizer instead of re-deciding the accepted shape here.
    """

    try:
        return _validated_revision_date(str(value or "").strip())
    except ValueError as exc:
        raise ValueError(f"{label} revision date is invalid: {value!r}") from exc


def frozen_revision_map(
    frame: pd.DataFrame, *, pdb_column: str = "pdb_id",
    revision_column: str = "structure_revision_date",
) -> dict[str, str]:
    """One normalized revision date per PDB ID from a frozen source snapshot.

    Fails closed on a missing/invalid date and on two different revisions claimed for the same
    entry: a build cannot be content-bound to a revision it cannot name uniquely.
    """

    if pdb_column not in frame.columns or revision_column not in frame.columns:
        raise ValueError("frozen revision source lacks the required columns")
    revisions: dict[str, str] = {}
    for pdb_id, revision in zip(frame[pdb_column], frame[revision_column]):
        key = str(pdb_id).strip().upper()
        value = _validated_revision(revision, label=f"frozen source {key}")
        if revisions.setdefault(key, value) != value:
            raise ValueError(
                f"frozen source declares conflicting revision dates for {key}"
            )
    return revisions


def parse_pdbe_enriched_residue_maps(
    path: Path, *, pdb_id: str, source_uniprot_id: str, auth_chain_id: str,
    canonical_sequence: str, frozen_uniprot_range: tuple[int, int],
    source_revision_date: str,
) -> list[dict[str, Any]]:
    """Parse exact per-residue SIFTS maps for every matching label-chain instance."""

    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    pdb_id = str(pdb_id).upper()
    source_uniprot_id = str(source_uniprot_id)
    auth_chain_id = str(auth_chain_id)
    sequence = "".join(str(canonical_sequence).split()).upper()
    start_1b, end_1b = map(int, frozen_uniprot_range)
    if (
        len(pdb_id) != 4 or not source_uniprot_id or not auth_chain_id
        or not sequence or not set(sequence) <= AA20
        or not 1 <= start_1b <= end_1b <= len(sequence)
    ):
        raise ValueError("invalid detailed SIFTS mapping request")
    revision = _validated_revision(source_revision_date, label="frozen source")
    payload = MMCIF2Dict(str(path))
    _entry_identity(payload, pdb_id=pdb_id)
    asym_rows = _table(payload, {
        "asym_id": "_struct_asym.id", "entity_id": "_struct_asym.entity_id",
    }, "struct_asym")
    entity_by_asym = {row["asym_id"]: row["entity_id"] for row in asym_rows}
    if len(entity_by_asym) != len(asym_rows):
        raise ValueError("PDBe enriched mmCIF has duplicate label asym IDs")
    scheme_rows = _table(payload, {
        "asym_id": "_pdbx_poly_seq_scheme.asym_id",
        "entity_id": "_pdbx_poly_seq_scheme.entity_id",
        "seq_id": "_pdbx_poly_seq_scheme.seq_id",
        "mon_id": "_pdbx_poly_seq_scheme.mon_id",
        "auth_seq_num": "_pdbx_poly_seq_scheme.auth_seq_num",
        "pdb_ins_code": "_pdbx_poly_seq_scheme.pdb_ins_code",
        "pdb_strand_id": "_pdbx_poly_seq_scheme.pdb_strand_id",
    }, "pdbx_poly_seq_scheme")
    xref_rows = _table(payload, {
        "entity_id": "_pdbx_sifts_xref_db.entity_id",
        "asym_id": "_pdbx_sifts_xref_db.asym_id",
        "seq_id": "_pdbx_sifts_xref_db.seq_id",
        "mon_aa": "_pdbx_sifts_xref_db.mon_id_one_letter_code",
        "unp_res": "_pdbx_sifts_xref_db.unp_res",
        "unp_num": "_pdbx_sifts_xref_db.unp_num",
        "unp_acc": "_pdbx_sifts_xref_db.unp_acc",
        "observed": "_pdbx_sifts_xref_db.observed",
    }, "pdbx_sifts_xref_db")

    scheme_by_asym: dict[str, list[dict[str, str]]] = {}
    for row in scheme_rows:
        if row["asym_id"] not in entity_by_asym or row["entity_id"] != entity_by_asym[row["asym_id"]]:
            raise ValueError("PDBe enriched mmCIF scheme/entity identity mismatch")
        scheme_by_asym.setdefault(row["asym_id"], []).append(row)
    matching_asym = sorted(
        asym for asym, rows in scheme_by_asym.items()
        if {row["pdb_strand_id"] for row in rows} == {auth_chain_id}
    )
    if not matching_asym:
        raise ValueError("PDBe enriched mmCIF has no exact auth chain mapping")

    candidates: list[dict[str, Any]] = []
    for asym_id in matching_asym:
        scheme = sorted(scheme_by_asym[asym_id], key=lambda row: int(row["seq_id"]))
        seq_ids = [int(row["seq_id"]) for row in scheme]
        if len(seq_ids) != len(set(seq_ids)):
            raise ValueError("PDBe enriched mmCIF has duplicate chain label_seq_id")
        local_index = {seq_id: index for index, seq_id in enumerate(seq_ids)}
        scheme_by_seq = {int(row["seq_id"]): row for row in scheme}
        mapped: dict[int, dict[str, Any]] = {}
        label_to_uniprot: dict[int, int] = {}
        for row in xref_rows:
            if row["asym_id"] != asym_id or row["unp_acc"] != source_uniprot_id:
                continue
            try:
                unp_num = int(row["unp_num"])
                label_seq_id = int(row["seq_id"])
            except ValueError as exc:
                raise ValueError("PDBe SIFTS residue numbering is not integral") from exc
            if not start_1b <= unp_num <= end_1b:
                continue
            if label_seq_id not in scheme_by_seq:
                raise ValueError("PDBe SIFTS residue lacks poly-sequence scheme identity")
            scheme_row = scheme_by_seq[label_seq_id]
            entity_id = entity_by_asym[asym_id]
            if row["entity_id"] != entity_id or scheme_row["entity_id"] != entity_id:
                raise ValueError("PDBe SIFTS residue entity identity mismatch")
            position = unp_num - 1
            aa = str(row["unp_res"]).upper()
            if aa not in AA20 or aa != sequence[position]:
                raise ValueError("PDBe SIFTS residue amino-acid differs from canonical UniProt")
            mon_aa = str(row["mon_aa"]).upper()
            if mon_aa not in {aa, "M" if aa == "M" else aa}:
                raise ValueError("PDBe SIFTS PDB/UniProt amino-acid identity mismatch")
            observed_flag = _parse_bool(row["observed"])
            raw_auth = str(scheme_row["auth_seq_num"])
            if raw_auth in {"", ".", "?"}:
                auth_seq_id = None
            else:
                try:
                    auth_seq_id = int(raw_auth)
                except ValueError as exc:
                    raise ValueError("PDBe SIFTS author residue number is not integral") from exc
            if observed_flag and auth_seq_id is None:
                raise ValueError("observed PDBe SIFTS residue lacks author residue number")
            residue = {
                "label_seq_id": label_seq_id,
                "auth_seq_id": auth_seq_id,
                "insertion_code": _normal_ins_code(scheme_row["pdb_ins_code"]),
                "chain_local_index_0b": local_index[label_seq_id],
                "amino_acid": aa,
                "observed": observed_flag,
            }
            prior = mapped.get(position)
            if prior is not None and prior != residue:
                raise ValueError("PDBe SIFTS has conflicting mappings for one UniProt residue")
            prior_unp = label_to_uniprot.get(label_seq_id)
            if prior_unp is not None and prior_unp != position:
                raise ValueError("PDBe SIFTS maps one chain residue to multiple UniProt residues")
            mapped[position] = residue
            label_to_uniprot[label_seq_id] = position
        if not mapped:
            continue
        ordered = sorted(mapped)
        label_order = [int(mapped[position]["label_seq_id"]) for position in ordered]
        if any(left >= right for left, right in zip(label_order, label_order[1:])):
            raise ValueError("PDBe SIFTS residue map is not order preserving")
        entity_id = entity_by_asym[asym_id]
        candidates.append({
            "pdb_id": pdb_id, "entity_id": entity_id,
            "rcsb_entity_id": f"{pdb_id}_{entity_id}", "label_asym_id": asym_id,
            "auth_chain_id": auth_chain_id, "source_uniprot_id": source_uniprot_id,
            "unp_start": start_1b, "unp_end": end_1b, "source_revision_date": revision,
            "residue_map": mapped, "residue_count": len(mapped),
            "observed_residue_count": sum(bool(row["observed"]) for row in mapped.values()),
        })
    if not candidates:
        raise ValueError("PDBe enriched mmCIF has no exact UniProt/range residue mapping")
    return candidates


def _coordinate_scheme(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], str]:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    payload = MMCIF2Dict(str(path))
    entry = _as_values(payload, "_entry.id")[0].upper()
    asym_rows = _table(payload, {
        "asym_id": "_struct_asym.id", "entity_id": "_struct_asym.entity_id",
    }, "RCSB struct_asym")
    entity_by_asym = {row["asym_id"]: row["entity_id"] for row in asym_rows}
    scheme_rows = _table(payload, {
        "asym_id": "_pdbx_poly_seq_scheme.asym_id",
        "entity_id": "_pdbx_poly_seq_scheme.entity_id",
        "seq_id": "_pdbx_poly_seq_scheme.seq_id",
        "auth_seq_num": "_pdbx_poly_seq_scheme.auth_seq_num",
        "pdb_ins_code": "_pdbx_poly_seq_scheme.pdb_ins_code",
        "pdb_strand_id": "_pdbx_poly_seq_scheme.pdb_strand_id",
    }, "RCSB pdbx_poly_seq_scheme")
    out: dict[tuple[str, int], dict[str, Any]] = {}
    by_asym: dict[str, list[int]] = {}
    for row in scheme_rows:
        asym = row["asym_id"]
        if entity_by_asym.get(asym) != row["entity_id"]:
            raise ValueError("RCSB coordinate scheme/entity identity mismatch")
        seq_id = int(row["seq_id"])
        by_asym.setdefault(asym, []).append(seq_id)
        key = (asym, seq_id)
        if key in out:
            raise ValueError("RCSB coordinate scheme has duplicate residue key")
        raw_auth = str(row["auth_seq_num"])
        if raw_auth in {"", ".", "?"}:
            auth_seq_id = None
        else:
            try:
                auth_seq_id = int(raw_auth)
            except ValueError as exc:
                raise ValueError("RCSB coordinate author residue number is not integral") from exc
        out[key] = {
            "entity_id": row["entity_id"], "auth_chain_id": row["pdb_strand_id"],
            "auth_seq_id": auth_seq_id,
            "insertion_code": _normal_ins_code(row["pdb_ins_code"]),
        }
    for asym, values in by_asym.items():
        for local_index, seq_id in enumerate(sorted(values)):
            out[(asym, seq_id)]["chain_local_index_0b"] = local_index
    return out, entry


def validate_residue_map_coordinate_identity(
    candidate: Mapping[str, Any], *, coordinate_cif_path: Path,
    coordinate_manifest_path: Path,
) -> None:
    manifest = json.loads(Path(coordinate_manifest_path).read_text())
    validate_cached_rcsb_mmcif(
        cif_path=coordinate_cif_path, manifest_path=coordinate_manifest_path,
        pdb_id=str(candidate["pdb_id"]),
        expected_revision_date=str(manifest["expected_revision_date"]),
    )
    if str(manifest["observed_revision_date"]) != str(candidate["source_revision_date"]):
        raise ValueError("PDBe SIFTS/RCSB coordinate revision mismatch")
    scheme, entry = _coordinate_scheme(coordinate_cif_path)
    if entry != str(candidate["pdb_id"]):
        raise ValueError("RCSB coordinate entry identity mismatch")
    asym = str(candidate["label_asym_id"])
    entity = str(candidate["entity_id"])
    auth_chain = str(candidate["auth_chain_id"])
    atoms_by_label: dict[int, list[dict[str, Any]]] = {}
    for atom in read_mmcif_atom_records(coordinate_cif_path):
        if str(atom["label_asym_id"]) == asym:
            atoms_by_label.setdefault(int(atom["label_seq_id"]), []).append(atom)
    for residue in candidate["residue_map"].values():
        key = (asym, int(residue["label_seq_id"]))
        observed = scheme.get(key)
        if (
            observed is None or observed["entity_id"] != entity
            or observed["auth_chain_id"] != auth_chain
            or observed["auth_seq_id"] != residue["auth_seq_id"]
            or observed["insertion_code"] != str(residue["insertion_code"])
            or observed["chain_local_index_0b"] != int(residue["chain_local_index_0b"])
        ):
            raise ValueError("PDBe SIFTS/RCSB coordinate chain identity mismatch")
        if bool(residue.get("observed")):
            atom_rows = atoms_by_label.get(int(residue["label_seq_id"]), [])
            if not atom_rows or any(
                str(atom["entity_id"]) != entity
                or str(atom["auth_asym_id"]) != auth_chain
                or int(atom["auth_seq_id"]) != int(residue["auth_seq_id"])
                or str(atom["insertion_code"]) != str(residue["insertion_code"])
                for atom in atom_rows
            ):
                raise ValueError("PDBe SIFTS/RCSB coordinate atom identity mismatch")


def _request(
    url: str, *, request_fn: Callable[[str, float], SiftsHttpResponse],
    sleep_fn: Callable[[float], None], timeout_s: float, max_attempts: int,
) -> tuple[SiftsHttpResponse | None, list[dict[str, Any]], str, str | None]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        queried = _utc()
        try:
            response = request_fn(url, timeout_s)
            status = int(response.status)
            attempts.append({
                "attempt": attempt, "queried_at_utc": queried,
                "http_status": status, "error": None,
            })
            if status == 429 or 500 <= status <= 599:
                if attempt < max_attempts:
                    sleep_fn(float(2 ** (attempt - 1)))
                    continue
                return response, attempts, queried, f"transient_http_exhausted:{status}"
            if status != 200:
                return response, attempts, queried, f"terminal_http:{status}"
            return response, attempts, queried, None
        except (
            # IncompleteRead/BadStatusLine are HTTPExceptions, not OSErrors: a truncated
            # chunked response must be retried, not abort a multi-hour network stage.
            urllib.error.URLError, TimeoutError, ConnectionError, OSError,
            http.client.HTTPException,
        ) as exc:
            attempts.append({
                "attempt": attempt, "queried_at_utc": queried, "http_status": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
            if attempt < max_attempts:
                sleep_fn(float(2 ** (attempt - 1)))
                continue
            return None, attempts, queried, f"transport_exhausted:{type(exc).__name__}"
    return None, attempts, _utc(), "no_response"


def _file_table(root: Path, *, relative_to: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(path.relative_to(relative_to)), "size_bytes": path.stat().st_size,
         "sha256": _sha(path)}
        for path in sorted(root.rglob("*"), key=lambda value: str(value.relative_to(root)))
        if path.is_file()
    ]


def _relative_candidate(path: Path, *, candidate_root: Path) -> str:
    resolved = Path(path).resolve()
    root = Path(candidate_root).resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("Tier1 residue snapshot coordinate evidence is outside candidate root")
    return str(resolved.relative_to(root))


def _input_identity(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve().relative_to(Path(root).resolve())),
        "size_bytes": Path(path).stat().st_size, "sha256": _sha(path),
    }


def snapshot_tier1_residue_maps(
    *, source_attempts: pd.DataFrame, canonical_sources: pd.DataFrame, output_dir: Path,
    candidate_root: Path, coordinate_provider: Callable[[str, str], tuple[Path, Path]],
    request_fn: Callable[[str, float], SiftsHttpResponse],
    frozen_revisions: Mapping[str, str],
    sleep_fn: Callable[[float], None] = time.sleep, timeout_s: float = 60.0,
    max_attempts: int = 4, api_delay_s: float = 0.1,
    min_tier1_source_units: int | None = None, command_argv: Sequence[str],
) -> Path:
    """Fetch every unique enriched entry and persist a replayable residue-map snapshot."""

    required_attempt = {
        "source_id", "pdb_id", "auth_chain_id", "unp_start", "unp_end",
        "sifts_item_index", "source_valid", "source_sequence_sha256",
    }
    required_source = {"source_id", "canonical_sequence", "canonical_sequence_sha256"}
    if (
        not required_attempt <= set(source_attempts)
        or not required_source <= set(canonical_sources)
        or not math.isfinite(float(api_delay_s)) or float(api_delay_s) < 0.0
    ):
        raise ValueError("Tier1 residue snapshot input schema mismatch")
    frozen = {
        str(pdb).strip().upper(): _validated_revision(revision, label=f"frozen source {pdb}")
        for pdb, revision in dict(frozen_revisions).items()
    }
    attempts = source_attempts[source_attempts["source_valid"] == True].copy()  # noqa: E712
    attempts = attempts.sort_values(["source_id", "sifts_item_index"], kind="stable")
    if attempts.duplicated(["source_id", "sifts_item_index"]).any():
        raise ValueError("Tier1 source-valid attempt keys are not unique")
    sources = canonical_sources.copy().sort_values("source_id", kind="stable")
    if sources["source_id"].duplicated().any():
        raise ValueError("Tier1 canonical source keys are not unique")
    source_by_id = sources.set_index("source_id")
    output_dir = Path(output_dir).resolve()
    candidate_root = Path(candidate_root).resolve()
    if candidate_root not in output_dir.parents:
        raise ValueError("Tier1 residue snapshot output is outside candidate root")
    candidate_relative_root = output_dir.relative_to(candidate_root)
    if (
        not candidate_relative_root.parts
        or candidate_relative_root.parts[0] != "audit"
        or candidate_relative_root.name != "tier1_residue_maps"
    ):
        raise ValueError("Tier1 residue snapshot is not under candidate audit root")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    input_dir = output_dir / "inputs"
    raw_dir = output_dir / "raw"
    input_dir.mkdir()
    raw_dir.mkdir()
    attempts_path = input_dir / "source_attempts.parquet"
    sources_path = input_dir / "canonical_sources.parquet"
    attempts.to_parquet(attempts_path, index=False)
    sources.to_parquet(sources_path, index=False)

    request_rows: list[dict[str, Any]] = []
    raw_by_pdb: dict[str, Path] = {}
    coordinate_by_pdb: dict[str, tuple[Path, Path]] = {}
    pdb_ids = sorted(set(attempts["pdb_id"].astype(str).str.upper()))
    unavailable_by_pdb: dict[str, tuple[str, str]] = {}
    for pdb_index, pdb_id in enumerate(pdb_ids):
        url = PDBE_ENRICHED_CIF_URL.format(pdb_id=pdb_id.lower())
        response, request_attempts, queried, request_error = _request(
            url, request_fn=request_fn, sleep_fn=sleep_fn,
            timeout_s=timeout_s, max_attempts=max_attempts,
        )
        status = int(response.status) if response is not None else None
        body = bytes(response.body) if response is not None else b""
        body_sha = _sha_bytes(body) if response is not None else None
        entry_dir = raw_dir / pdb_id
        entry_dir.mkdir()
        raw_path = None
        revision = None
        coordinate_cif = None
        coordinate_manifest = None
        parse_error = request_error
        if response is not None:
            suffix = ".cif" if status == 200 else ".bin"
            raw_path = entry_dir / f"{body_sha}{suffix}"
            raw_path.write_bytes(body)
        if status == 200 and raw_path is not None:
            try:
                from Bio.PDB.MMCIF2Dict import MMCIF2Dict
                _entry_identity(MMCIF2Dict(str(raw_path)), pdb_id=pdb_id)
                if pdb_id not in frozen:
                    raise ValueError(
                        "no frozen source revision for this PDB entry; the frozen RCSB source "
                        "snapshot is the revision authority"
                    )
                revision = frozen[pdb_id]
                coordinate_cif, coordinate_manifest = coordinate_provider(pdb_id, revision)
                coordinate_cif = Path(coordinate_cif).resolve()
                coordinate_manifest = Path(coordinate_manifest).resolve()
                coordinate_payload = json.loads(coordinate_manifest.read_text())
                validate_cached_rcsb_mmcif(
                    cif_path=coordinate_cif, manifest_path=coordinate_manifest, pdb_id=pdb_id,
                    expected_revision_date=str(coordinate_payload["expected_revision_date"]),
                )
                if str(coordinate_payload["observed_revision_date"]) != revision:
                    raise ValueError("frozen source/RCSB coordinate revision mismatch")
                raw_by_pdb[pdb_id] = raw_path
                coordinate_by_pdb[pdb_id] = (coordinate_cif, coordinate_manifest)
            except (
                # RcsbMmcifAcquisitionError is a RuntimeError, not a ValueError: a single stale
                # or unavailable entry must be recorded as a typed per-PDB failure, not kill a
                # multi-hour login-node stage that has already paid for hundreds of downloads.
                ValueError, OSError, json.JSONDecodeError, RcsbMmcifAcquisitionError,
            ) as exc:
                parse_error = f"malformed_or_coordinate_failure:{type(exc).__name__}:{exc}"
        if parse_error is not None:
            unavailable_by_pdb[pdb_id] = (
                "source_request_failure" if status != 200 else "malformed_or_coordinate_failure",
                parse_error,
            )
        request_rows.append({
            "pdb_id": pdb_id, "url": url, "queried_at_utc": queried,
            "http_status": status,
            "attempts_json": json.dumps(request_attempts, sort_keys=True, separators=(",", ":")),
            "response_path": (
                str(raw_path.relative_to(output_dir)) if raw_path is not None else None
            ),
            "response_size_bytes": raw_path.stat().st_size if raw_path is not None else None,
            "response_sha256": body_sha,
            "response_headers_json": json.dumps(
                dict(response.headers) if response is not None else {},
                sort_keys=True, separators=(",", ":")
            ),
            "request_error": parse_error,
            "observed_revision_date": revision,
            "coordinate_cif_path": (
                _relative_candidate(coordinate_cif, candidate_root=candidate_root)
                if coordinate_cif is not None else None
            ),
            "coordinate_manifest_path": (
                _relative_candidate(coordinate_manifest, candidate_root=candidate_root)
                if coordinate_manifest is not None else None
            ),
            "coordinate_cif_sha256": _sha(coordinate_cif) if coordinate_cif is not None else None,
            "coordinate_manifest_sha256": (
                _sha(coordinate_manifest) if coordinate_manifest is not None else None
            ),
        })
        if pdb_index + 1 < len(pdb_ids) and api_delay_s > 0.0:
            sleep_fn(float(api_delay_s))

    candidate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for attempt in attempts.to_dict("records"):
        source_id = str(attempt["source_id"])
        pdb_id = str(attempt["pdb_id"]).upper()
        base_failure = {
            "source_id": source_id, "sifts_item_index": int(attempt["sifts_item_index"]),
            "pdb_id": pdb_id, "auth_chain_id": str(attempt["auth_chain_id"]),
            "unp_start": int(attempt["unp_start"]), "unp_end": int(attempt["unp_end"]),
        }
        if pdb_id in unavailable_by_pdb:
            failure_code, failure_detail = unavailable_by_pdb[pdb_id]
            failure_rows.append({
                **base_failure, "failure_code": failure_code,
                "failure_detail": failure_detail,
            })
            continue
        if source_id not in source_by_id.index:
            raise ValueError("Tier1 residue snapshot attempt lacks canonical source")
        source = source_by_id.loc[source_id]
        sequence = str(source["canonical_sequence"])
        sequence_sha = _sha_bytes(sequence.encode())
        if (
            sequence_sha != str(source["canonical_sequence_sha256"])
            or sequence_sha != str(attempt["source_sequence_sha256"])
        ):
            raise ValueError("Tier1 residue snapshot canonical source digest mismatch")
        try:
            parsed = parse_pdbe_enriched_residue_maps(
                raw_by_pdb[pdb_id], pdb_id=pdb_id, source_uniprot_id=source_id,
                auth_chain_id=str(attempt["auth_chain_id"]), canonical_sequence=sequence,
                frozen_uniprot_range=(int(attempt["unp_start"]), int(attempt["unp_end"])),
                source_revision_date=frozen[pdb_id],
            )
        except ValueError as exc:
            failure_rows.append({
                **base_failure, "failure_code": "no_valid_detailed_sifts_map",
                "failure_detail": str(exc),
            })
            continue
        coordinate_cif, coordinate_manifest = coordinate_by_pdb[pdb_id]
        # PDBe and RCSB disagreeing about a chain is a database disagreement, the same class as the
        # amino-acid mismatch above -- typed and per-attempt, never fatal. Raising would discard a
        # multi-hour login stage's completed downloads because one entry of ~1,650 disagrees.
        # Validate EVERY chain copy before admitting any: the failure table is keyed per attempt,
        # so an attempt is admitted all-or-nothing.
        try:
            for candidate in parsed:
                validate_residue_map_coordinate_identity(
                    candidate, coordinate_cif_path=coordinate_cif,
                    coordinate_manifest_path=coordinate_manifest,
                )
        except ValueError as exc:
            failure_rows.append({
                **base_failure, "failure_code": "coordinate_identity_mismatch",
                "failure_detail": str(exc),
            })
            continue
        for candidate in parsed:
            candidate_id = (
                f"{source_id}|{int(attempt['sifts_item_index'])}|{pdb_id}|"
                f"{candidate['label_asym_id']}"
            )
            candidate_rows.append({
                "candidate_id": candidate_id, "source_id": source_id,
                "sifts_item_index": int(attempt["sifts_item_index"]), "pdb_id": pdb_id,
                "entity_id": str(candidate["entity_id"]),
                "rcsb_entity_id": str(candidate["rcsb_entity_id"]),
                "label_asym_id": str(candidate["label_asym_id"]),
                "auth_chain_id": str(candidate["auth_chain_id"]),
                "unp_start": int(candidate["unp_start"]), "unp_end": int(candidate["unp_end"]),
                "source_sequence_sha256": sequence_sha,
                "residue_count": int(candidate["residue_count"]),
                "observed_residue_count": int(candidate["observed_residue_count"]),
                "residue_map_json": json.dumps(
                    candidate["residue_map"], sort_keys=True, separators=(",", ":")
                ),
                "raw_sifts_path": str(raw_by_pdb[pdb_id].relative_to(output_dir)),
                "raw_sifts_sha256": _sha(raw_by_pdb[pdb_id]),
                "source_revision_date": str(candidate["source_revision_date"]),
                "coordinate_cif_path": _relative_candidate(
                    coordinate_cif, candidate_root=candidate_root
                ),
                "coordinate_cif_sha256": _sha(coordinate_cif),
                "coordinate_manifest_path": _relative_candidate(
                    coordinate_manifest, candidate_root=candidate_root
                ),
                "coordinate_manifest_sha256": _sha(coordinate_manifest),
            })
    candidates = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS).sort_values(
        "candidate_id", kind="stable"
    ) if candidate_rows else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    failures = pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS).sort_values(
        ["source_id", "sifts_item_index"], kind="stable"
    ) if failure_rows else pd.DataFrame(columns=FAILURE_COLUMNS)
    requests = pd.DataFrame(request_rows, columns=REQUEST_COLUMNS).sort_values(
        "pdb_id", kind="stable"
    ) if request_rows else pd.DataFrame(columns=REQUEST_COLUMNS)
    requests_path = output_dir / "requests.parquet"
    candidates_path = output_dir / "tier1_residue_map_candidates.parquet"
    failures_path = output_dir / "tier1_residue_map_failures.parquet"
    requests.to_parquet(requests_path, index=False)
    candidates.to_parquet(candidates_path, index=False)
    failures.to_parquet(failures_path, index=False)
    raw_files = _file_table(raw_dir, relative_to=output_dir)
    successful_request_rows = [
        row for row in request_rows if row["coordinate_cif_path"] is not None
    ]
    coordinate_files = sorted([
        {"path": row["coordinate_cif_path"], "size_bytes": int(
            (candidate_root / row["coordinate_cif_path"]).stat().st_size
        ), "sha256": row["coordinate_cif_sha256"]}
        for row in successful_request_rows
    ] + [
        {"path": row["coordinate_manifest_path"], "size_bytes": int(
            (candidate_root / row["coordinate_manifest_path"]).stat().st_size
        ), "sha256": row["coordinate_manifest_sha256"]}
        for row in successful_request_rows
    ], key=lambda row: row["path"])
    # Capacity is measured in Tier 1 SOURCE UNITS (allele+UniProt), not candidate rows: several
    # chains of one entry are one opportunity. The gate is evaluated here but raised only after
    # the manifest is on disk, so a capacity failure still leaves a complete audit trail.
    viable_source_units = int(candidates["source_id"].nunique()) if len(candidates) else 0
    capacity_short = (
        min_tier1_source_units is not None
        and viable_source_units < int(min_tier1_source_units)
    )
    payload = {
        "schema_version": "if-benchmark-v3-tier1-residue-snapshot/1",
        "authority": "PDBe enriched mmCIF _pdbx_sifts_xref_db",
        "endpoint_template": PDBE_ENRICHED_CIF_URL,
        "candidate_relative_root": str(candidate_relative_root),
        "command_argv": list(command_argv), "timeout_s": float(timeout_s),
        "max_attempts": int(max_attempts), "api_delay_s": float(api_delay_s),
        "inputs": {
            "source_attempts": _input_identity(attempts_path, root=output_dir),
            "canonical_sources": _input_identity(sources_path, root=output_dir),
        },
        "outputs": {
            "requests": _input_identity(requests_path, root=output_dir),
            "candidates": _input_identity(candidates_path, root=output_dir),
            "failures": _input_identity(failures_path, root=output_dir),
        },
        "raw_files": raw_files,
        "raw_file_merkle_sha256": _sha_bytes(_canonical_json(raw_files)),
        "coordinate_files": coordinate_files,
        "coordinate_file_merkle_sha256": _sha_bytes(_canonical_json(coordinate_files)),
        "counts": {
            "source_valid_attempts": len(attempts), "unique_pdb_requests": len(requests),
            "residue_map_candidates": len(candidates), "mapping_failures": len(failures),
            "request_success": sum(row["http_status"] == 200 and row["pdb_id"] not in unavailable_by_pdb for row in request_rows),
            "terminal_http_failure": sum(
                row["http_status"] is not None and row["http_status"] != 200
                for row in request_rows
            ),
            "network_failure": sum(row["http_status"] is None for row in request_rows),
            "malformed_or_coordinate_failure": sum(
                row["pdb_id"] in unavailable_by_pdb and row["http_status"] == 200
                for row in request_rows
            ),
            "viable_tier1_source_units": viable_source_units,
        },
        "min_tier1_source_units": (
            None if min_tier1_source_units is None else int(min_tier1_source_units)
        ),
        "release_blocked": bool(unavailable_by_pdb) or capacity_short,
    }
    payload["manifest_sha256"] = _sha_bytes(_canonical_json(payload))
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if capacity_short:
        raise Tier1CapacityError(
            f"Tier 1 capacity {viable_source_units} < required {int(min_tier1_source_units)}; "
            f"C8 must not seal (audit trail written to {manifest_path})"
        )
    return manifest_path


def _check_identity(identity: Mapping[str, Any], *, root: Path) -> Path:
    raw = Path(str(identity.get("path") or ""))
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("Tier1 residue snapshot artifact path escapes root")
    path = (root / raw).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError("Tier1 residue snapshot artifact is missing")
    if path.stat().st_size != int(identity["size_bytes"]) or _sha(path) != identity["sha256"]:
        raise ValueError("Tier1 residue snapshot artifact identity mismatch")
    return path


def validate_tier1_residue_map_snapshot(
    manifest_path: Path, *, require_complete: bool = True,
) -> dict[str, int]:
    """Reparse every raw body and coordinate file, then frame-compare persisted ledgers."""

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text())
    unsigned = dict(payload)
    digest = unsigned.pop("manifest_sha256", None)
    if (
        payload.get("schema_version") != "if-benchmark-v3-tier1-residue-snapshot/1"
        or digest != _sha_bytes(_canonical_json(unsigned))
        or payload.get("authority") != "PDBe enriched mmCIF _pdbx_sifts_xref_db"
        or payload.get("endpoint_template") != PDBE_ENRICHED_CIF_URL
        or float(payload.get("api_delay_s", -1.0)) != 0.1
    ):
        raise ValueError("Tier1 residue snapshot manifest identity mismatch")
    candidate_relative_root = Path(str(payload.get("candidate_relative_root") or ""))
    if (
        candidate_relative_root.is_absolute()
        or not candidate_relative_root.parts
        or ".." in candidate_relative_root.parts
        or candidate_relative_root.parts[0] != "audit"
        or candidate_relative_root.name != "tier1_residue_maps"
    ):
        raise ValueError("Tier1 residue snapshot candidate-relative root is invalid")
    candidate_root = root
    for _part in candidate_relative_root.parts:
        candidate_root = candidate_root.parent
    candidate_root = candidate_root.resolve()
    if (candidate_root / candidate_relative_root).resolve() != root:
        raise ValueError("Tier1 residue snapshot is not under candidate audit root")
    input_paths = {key: _check_identity(value, root=root) for key, value in payload["inputs"].items()}
    output_paths = {
        key: _check_identity(value, root=root) for key, value in payload["outputs"].items()
    }
    raw_files = _file_table(root / "raw", relative_to=root)
    if (
        raw_files != payload.get("raw_files")
        or payload.get("raw_file_merkle_sha256") != _sha_bytes(_canonical_json(raw_files))
    ):
        raise ValueError("Tier1 residue snapshot raw file table/Merkle mismatch")
    coordinate_files = []
    for item in payload.get("coordinate_files") or []:
        raw = Path(str(item.get("path") or ""))
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError("Tier1 coordinate evidence escapes candidate root")
        path = (candidate_root / raw).resolve()
        if (
            candidate_root not in path.parents or not path.is_file()
            or path.stat().st_size != int(item["size_bytes"]) or _sha(path) != item["sha256"]
        ):
            raise ValueError("Tier1 coordinate file table identity mismatch")
        coordinate_files.append({
            "path": str(raw), "size_bytes": path.stat().st_size, "sha256": _sha(path),
        })
    coordinate_files.sort(key=lambda row: row["path"])
    if (
        coordinate_files != payload.get("coordinate_files")
        or payload.get("coordinate_file_merkle_sha256")
        != _sha_bytes(_canonical_json(coordinate_files))
    ):
        raise ValueError("Tier1 coordinate file table/Merkle mismatch")

    attempts = pd.read_parquet(input_paths["source_attempts"])
    sources = pd.read_parquet(input_paths["canonical_sources"]).set_index("source_id")
    requests = pd.read_parquet(output_paths["requests"])
    observed_candidates = pd.read_parquet(output_paths["candidates"])
    observed_failures = pd.read_parquet(output_paths["failures"])
    if list(requests.columns) != REQUEST_COLUMNS:
        raise ValueError("Tier1 residue request ledger schema mismatch")
    request_by_pdb = requests.set_index("pdb_id")
    if set(request_by_pdb.index.astype(str)) != set(attempts["pdb_id"].astype(str).str.upper()):
        raise ValueError("Tier1 residue request set differs from source-valid PDB set")
    expected_candidates: list[dict[str, Any]] = []
    expected_failures: list[dict[str, Any]] = []
    unavailable_pdbs: set[str] = set()
    for attempt in attempts.to_dict("records"):
        source_id = str(attempt["source_id"])
        pdb_id = str(attempt["pdb_id"]).upper()
        request = request_by_pdb.loc[pdb_id]
        raw_path = None
        if not pd.isna(request["response_path"]):
            raw_path = (root / str(request["response_path"])).resolve()
            if (
                root not in raw_path.parents or not raw_path.is_file()
                or _sha(raw_path) != str(request["response_sha256"])
                or raw_path.stat().st_size != int(request["response_size_bytes"])
            ):
                raise ValueError("Tier1 residue request/raw identity mismatch")
        elif not (
            pd.isna(request["response_sha256"]) and pd.isna(request["response_size_bytes"])
            and pd.isna(request["http_status"])
        ):
            raise ValueError("Tier1 residue missing-response identity mismatch")
        coordinate_present = not pd.isna(request["coordinate_cif_path"])
        coordinate_fields = [
            "coordinate_cif_path", "coordinate_manifest_path",
            "coordinate_cif_sha256", "coordinate_manifest_sha256",
        ]
        if coordinate_present != all(not pd.isna(request[field]) for field in coordinate_fields):
            raise ValueError("Tier1 residue coordinate request fields are partial")
        coordinate_cif = None
        coordinate_manifest = None
        if coordinate_present:
            coordinate_cif = (candidate_root / str(request["coordinate_cif_path"])).resolve()
            coordinate_manifest = (
                candidate_root / str(request["coordinate_manifest_path"])
            ).resolve()
            if (
                _sha(coordinate_cif) != str(request["coordinate_cif_sha256"])
                or _sha(coordinate_manifest) != str(request["coordinate_manifest_sha256"])
            ):
                raise ValueError("Tier1 residue request/coordinate identity mismatch")
        sequence = str(sources.loc[source_id, "canonical_sequence"])
        base_failure = {
            "source_id": source_id, "sifts_item_index": int(attempt["sifts_item_index"]),
            "pdb_id": pdb_id, "auth_chain_id": str(attempt["auth_chain_id"]),
            "unp_start": int(attempt["unp_start"]), "unp_end": int(attempt["unp_end"]),
        }
        if not coordinate_present or int(request["http_status"]) != 200:
            unavailable_pdbs.add(pdb_id)
            failure_code = (
                "source_request_failure"
                if pd.isna(request["http_status"]) or int(request["http_status"]) != 200
                else "malformed_or_coordinate_failure"
            )
            expected_failures.append({
                **base_failure, "failure_code": failure_code,
                "failure_detail": str(request["request_error"]),
            })
            continue
        if raw_path is None or coordinate_cif is None or coordinate_manifest is None:
            raise ValueError("Tier1 successful residue request lacks source/coordinate bytes")
        try:
            parsed = parse_pdbe_enriched_residue_maps(
                raw_path, pdb_id=pdb_id, source_uniprot_id=source_id,
                auth_chain_id=str(attempt["auth_chain_id"]), canonical_sequence=sequence,
                frozen_uniprot_range=(int(attempt["unp_start"]), int(attempt["unp_end"])),
                # replay uses the revision the snapshot persisted, which the producer already
                # cross-checked against the RCSB coordinate manifest
                source_revision_date=str(request["observed_revision_date"]),
            )
        except ValueError as exc:
            expected_failures.append({
                **base_failure, "failure_code": "no_valid_detailed_sifts_map",
                "failure_detail": str(exc),
            })
            continue
        try:
            for candidate in parsed:
                validate_residue_map_coordinate_identity(
                    candidate, coordinate_cif_path=coordinate_cif,
                    coordinate_manifest_path=coordinate_manifest,
                )
        except ValueError as exc:
            expected_failures.append({
                **base_failure, "failure_code": "coordinate_identity_mismatch",
                "failure_detail": str(exc),
            })
            continue
        for candidate in parsed:
            expected_candidates.append({
                "candidate_id": (
                    f"{source_id}|{int(attempt['sifts_item_index'])}|{pdb_id}|"
                    f"{candidate['label_asym_id']}"
                ),
                "source_id": source_id, "sifts_item_index": int(attempt["sifts_item_index"]),
                "pdb_id": pdb_id, "entity_id": str(candidate["entity_id"]),
                "rcsb_entity_id": str(candidate["rcsb_entity_id"]),
                "label_asym_id": str(candidate["label_asym_id"]),
                "auth_chain_id": str(candidate["auth_chain_id"]),
                "unp_start": int(candidate["unp_start"]), "unp_end": int(candidate["unp_end"]),
                "source_sequence_sha256": _sha_bytes(sequence.encode()),
                "residue_count": int(candidate["residue_count"]),
                "observed_residue_count": int(candidate["observed_residue_count"]),
                "residue_map_json": json.dumps(
                    candidate["residue_map"], sort_keys=True, separators=(",", ":")
                ),
                "raw_sifts_path": str(request["response_path"]),
                "raw_sifts_sha256": str(request["response_sha256"]),
                "source_revision_date": str(candidate["source_revision_date"]),
                "coordinate_cif_path": str(request["coordinate_cif_path"]),
                "coordinate_cif_sha256": str(request["coordinate_cif_sha256"]),
                "coordinate_manifest_path": str(request["coordinate_manifest_path"]),
                "coordinate_manifest_sha256": str(request["coordinate_manifest_sha256"]),
            })
    expected_candidate_frame = pd.DataFrame(expected_candidates, columns=CANDIDATE_COLUMNS)
    expected_failure_frame = pd.DataFrame(expected_failures, columns=FAILURE_COLUMNS)
    if len(expected_candidate_frame):
        expected_candidate_frame = expected_candidate_frame.sort_values("candidate_id").reset_index(drop=True)
    if len(expected_failure_frame):
        expected_failure_frame = expected_failure_frame.sort_values(
            ["source_id", "sifts_item_index"]
        ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            observed_candidates.reset_index(drop=True), expected_candidate_frame,
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            observed_failures.reset_index(drop=True), expected_failure_frame,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError("Tier1 residue map candidates/failures do not replay raw bytes") from exc
    counts = payload.get("counts") or {}
    request_records = requests.to_dict("records")
    expected_counts = {
        "source_valid_attempts": len(attempts), "unique_pdb_requests": len(requests),
        "residue_map_candidates": len(expected_candidate_frame),
        "mapping_failures": len(expected_failure_frame),
        "request_success": sum(
            int(row["http_status"]) == 200 and not pd.isna(row["coordinate_cif_path"])
            for row in request_records if not pd.isna(row["http_status"])
        ),
        "terminal_http_failure": sum(
            not pd.isna(row["http_status"]) and int(row["http_status"]) != 200
            for row in request_records
        ),
        "network_failure": sum(pd.isna(row["http_status"]) for row in request_records),
        "malformed_or_coordinate_failure": sum(
            not pd.isna(row["http_status"]) and int(row["http_status"]) == 200
            and pd.isna(row["coordinate_cif_path"])
            for row in request_records
        ),
        "viable_tier1_source_units": (
            int(expected_candidate_frame["source_id"].nunique())
            if len(expected_candidate_frame) else 0
        ),
    }
    if counts != expected_counts:
        raise ValueError("Tier1 residue snapshot count funnel mismatch")
    minimum = payload.get("min_tier1_source_units")
    capacity_short = (
        minimum is not None
        and expected_counts["viable_tier1_source_units"] < int(minimum)
    )
    blocked = bool(unavailable_pdbs) or capacity_short
    if bool(payload.get("release_blocked")) != blocked:
        raise ValueError("Tier1 residue snapshot blocked-state mismatch")
    if require_complete and blocked:
        raise ValueError("Tier1 residue snapshot has unresolved source requests")
    return {"candidate_count": len(expected_candidate_frame),
            "failure_count": len(expected_failure_frame)}
