"""Identity-bound runner primitives and a no-network v3 end-to-end fixture."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .evidence import (
    alias_identity,
    canonical_json,
    sha256_file,
    stable_family_cluster_id,
    write_descendant_inventory,
    write_release_evidence_bundle,
)
from .leakage import build_mmseqs_sidecar
from .head_annotation import _path_identity as _head_path_identity
from .preflight import SiftsHttpResponse, run_tier1_source_preflight
from .rcsb_snapshot import GRAPHQL_URL, SEARCH_URL, snapshot_rcsb_entities
from .references import materialize_cath_train_reference, materialize_head_seen_reference
from .nmp import (
    nmp_parameter_payload,
    write_nmp_install_manifest,
    write_nmp_sharded_evidence,
)
from .release import validate_release_tables, write_dataset_manifest
from .selection import binned_sample_to_dict, deterministic_binned_sample
from .tier1 import finalize_tier1_pool, resolve_tier1_tier2_collisions
from .sifts_residue import snapshot_tier1_residue_maps
from .structures import acquire_rcsb_mmcif, write_structure_download_registry


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _tiny_coordinate_cif_bytes(
    *, pdb_id: str, chain: str, amino_acid: str, revision: str = "2020-02-01",
) -> bytes:
    monomer = {"A": "ALA", "C": "CYS", "D": "ASP"}[amino_acid]
    fields = [
        "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.label_alt_id",
        "_atom_site.label_comp_id", "_atom_site.label_asym_id",
        "_atom_site.label_entity_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x",
        "_atom_site.Cartn_y", "_atom_site.Cartn_z", "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv", "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id", "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id", "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    lines = [
        f"data_{pdb_id}", f"_entry.id {pdb_id}", "loop_",
        "_pdbx_audit_revision_history.ordinal",
        "_pdbx_audit_revision_history.revision_date", f"1 {revision}", "#",
        "loop_", "_struct_asym.id", "_struct_asym.entity_id", f"{chain} 1", "#",
        "loop_", "_pdbx_poly_seq_scheme.asym_id",
        "_pdbx_poly_seq_scheme.entity_id", "_pdbx_poly_seq_scheme.seq_id",
        "_pdbx_poly_seq_scheme.mon_id", "_pdbx_poly_seq_scheme.auth_seq_num",
        "_pdbx_poly_seq_scheme.pdb_ins_code",
        "_pdbx_poly_seq_scheme.pdb_strand_id",
        *[
            f"{chain} 1 {position} {monomer} {position} ? {chain}"
            for position in range(1, 101)
        ],
        "#", "loop_", *fields,
    ]
    serial = 1
    for position in range(1, 101):
        for atom_index, atom in enumerate(("N", "CA", "C", "O")):
            element = "C" if atom in {"CA", "C"} else atom
            lines.append(
                f"ATOM {serial} {element} {atom} . {monomer} {chain} 1 {position} ? "
                f"{position:.1f} {atom_index:.1f} 0.0 1.0 10.0 ? {position} "
                f"{monomer} {chain} {atom} 1"
            )
            serial += 1
    lines.append("#")
    return ("\n".join(lines) + "\n").encode()


class StageJournal:
    """Resume a stage only when input identity and every output byte still match."""

    def __init__(self, stage_dir: Path):
        self.stage_dir = Path(stage_dir)
        self.path = self.stage_dir / "stage_identity.json"

    @staticmethod
    def _validate_identity(identity: Mapping[str, Any]) -> None:
        for field in ("code", "config"):
            record = identity.get(field)
            if not isinstance(record, Mapping):
                raise ValueError(f"stage identity requires typed {field} file identity")
            StageJournal._validate_file_identity(record, field)
        inputs = identity.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("stage identity requires non-empty input file identities")
        for index, record in enumerate(inputs):
            if not isinstance(record, Mapping):
                raise ValueError("stage input identity is not typed")
            StageJournal._validate_file_identity(record, f"input[{index}]")

    @staticmethod
    def _validate_file_identity(record: Mapping[str, Any], label: str) -> None:
        path = Path(str(record.get("path") or ""))
        digest = record.get("sha256")
        if not path.is_file() or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"stage {label} file identity is incomplete")
        if _sha_file(path) != digest:
            raise ValueError(f"stage {label} file digest mismatch")

    def _verify_registered_file_set(self, outputs: Sequence[Mapping[str, Any]]) -> None:
        registered = {str(item["path"]) for item in outputs}
        observed = {
            str(path.relative_to(self.stage_dir))
            for path in self.stage_dir.rglob("*")
            if path.is_file() and path != self.path
        }
        extra = sorted(observed - registered)
        missing = sorted(registered - observed)
        if extra:
            raise ValueError(f"stage has unregistered extra files: {extra}")
        if missing:
            raise ValueError(f"stage registered outputs are missing: {missing}")

    def _verify(self, payload: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
        expected_digest = hashlib.sha256(_canonical(identity)).hexdigest()
        if payload.get("identity_sha256") != expected_digest or payload.get("identity") != identity:
            raise ValueError("resume identity mismatch")
        if payload.get("status") != "complete":
            raise ValueError("stage journal is not complete")
        self._verify_registered_file_set(payload.get("outputs") or [])
        for output in payload.get("outputs") or []:
            path = self.stage_dir / output["path"]
            if (
                not path.is_file()
                or path.stat().st_size != output["size_bytes"]
                or _sha_file(path) != output["sha256"]
            ):
                raise ValueError(f"resume output digest mismatch: {output['path']}")
        return dict(payload)

    def run(
        self,
        identity: Mapping[str, Any],
        producer: Callable[[Path], Sequence[Path]],
        *,
        resume: bool,
    ) -> dict[str, Any]:
        identity = dict(identity)
        if self.path.is_file():
            if not resume:
                raise FileExistsError(f"stage already has identity journal: {self.stage_dir}")
            self._validate_identity(identity)
            return self._verify(json.loads(self.path.read_text()), identity)
        if self.stage_dir.exists() and any(self.stage_dir.iterdir()):
            raise ValueError(f"legacy/unbound cache cannot be resumed: {self.stage_dir}")
        self._validate_identity(identity)
        if resume and self.stage_dir.exists():
            # An empty directory is safe to initialize; it carries no reusable state.
            pass
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        try:
            produced = [Path(path).resolve() for path in producer(self.stage_dir)]
            root = self.stage_dir.resolve()
            outputs = []
            for path in produced:
                if root not in path.parents or not path.is_file():
                    raise ValueError(f"stage producer returned invalid output: {path}")
                outputs.append(
                    {
                        "path": str(path.relative_to(root)),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha_file(path),
                    }
                )
            outputs.sort(key=lambda row: row["path"])
            self._verify_registered_file_set(outputs)
            payload = {
                "schema_version": "if-benchmark-v3-stage-journal/1",
                "status": "complete",
                "identity": identity,
                "identity_sha256": hashlib.sha256(_canonical(identity)).hexdigest(),
                "outputs": outputs,
            }
        except Exception as exc:
            failed = {
                "schema_version": "if-benchmark-v3-stage-journal/1",
                "status": "failed",
                "identity": identity,
                "identity_sha256": hashlib.sha256(_canonical(identity)).hexdigest(),
                "error": f"{type(exc).__name__}: {exc}",
            }
            temporary = self.stage_dir / ".stage_identity.failed.tmp"
            temporary.write_text(json.dumps(failed, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, self.path)
            raise
        temporary = self.stage_dir / ".stage_identity.complete.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.path)
        return payload


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _tiny_row(
    *, pid: str, tier: str, amino_acid: str, structure_path: str,
    structure_sha256: str, role: str, cath_overlap: bool,
    selection_unit_id: str, pdb_id: str, entity_id: str,
    auth_chain_id: str, source_uniprot_id: str | None = None,
) -> dict[str, Any]:
    sequence = amino_acid * 100
    sequence_sha = _sha_text(sequence)
    row = {
        "dataset_release_id": "tiny-v3-release", "allele": "HLA-DRB1*15:01",
        "protein_id": pid, "selection_unit_id": selection_unit_id, "pdb_id": pdb_id,
        "entity_id": entity_id, "rcsb_entity_id": f"{pdb_id}_{entity_id}",
        "auth_chain_id": auth_chain_id, "label_asym_id": auth_chain_id,
        "source_pool_memberships": [tier], "selected_tier_memberships": [tier],
        "evaluation_role": role, "sequence": sequence, "sequence_length": len(sequence),
        "sequence_sha256": sequence_sha, "entity_sequence": sequence,
        "entity_sequence_sha256": sequence_sha, "source_sequence": sequence,
        "source_sequence_sha256": sequence_sha, "source_range_start_0b": 0,
        "source_range_end_0b": len(sequence), "source_denominator_length": len(sequence),
        "final_to_source_json": json.dumps(list(range(len(sequence)))),
        "mapping_status": "complete", "if_sequence_coverage": 1.0,
        "selected_chain_provenance_json": json.dumps(
            {"rcsb_entity_id": f"{pdb_id}_{entity_id}", "auth_chain_id": auth_chain_id,
             "label_asym_id": auth_chain_id}, sort_keys=True
        ),
        "rejected_chain_provenance_json": "[]",
        "pdb_path": structure_path, "structure_sha256": structure_sha256,
        "experimental_method": "X-RAY DIFFRACTION", "resolution": 1.5,
        "nmp_status": "complete", "nmp_query_sequence_sha256": sequence_sha,
        "nmp_evidence_sha256": None, "nmp_tool_sha256": None,
        "nmp_parameters_sha256": None, "coverage_fraction": 0.0,
        "selection_bin": 0, "selection_reason": "tiny_fixture",
        "cath_overlap_flag": cath_overlap,
        "cath_best_target": "CATH1" if cath_overlap else None,
        "cath_best_identity": 0.4 if cath_overlap else None,
        "family_cluster_id": None,
        "source_uniprot_id": None, "sifts_range_json": None,
        "sifts_residue_map_json": None,
        "uniprot_spans_json": None, "chain_spans_json": None, "if_spans_json": None,
        "verified_span_count": None, "verified_covered_residue_count": None,
        "final_epitope_coverage": None,
    }
    if tier == "tier1":
        spans = [
            {"start_0b": 0, "end_0b": 10, "peptide": sequence[:10]},
            {"start_0b": 20, "end_0b": 30, "peptide": sequence[20:30]},
        ]
        residue_map = {
            str(position): {"label_seq_id": position + 1, "auth_seq_id": position + 1,
                            "insertion_code": "", "chain_local_index_0b": position,
                            "amino_acid": amino_acid, "observed": True}
            for position in range(100)
        }
        chain_spans = [
            {**span, "uniprot_positions_0b": list(range(span["start_0b"], span["end_0b"])),
             "label_seq_ids": list(range(span["start_0b"] + 1, span["end_0b"] + 1)),
             "auth_residues": [
                 {"auth_seq_id": position + 1, "insertion_code": ""}
                 for position in range(span["start_0b"], span["end_0b"])
             ]}
            for span in spans
        ]
        row.update(
            {
                "source_uniprot_id": source_uniprot_id, "sifts_range_json": "[1,100]",
                "sifts_residue_map_json": json.dumps(residue_map, sort_keys=True),
                "uniprot_spans_json": json.dumps(spans),
                "chain_spans_json": json.dumps(chain_spans), "if_spans_json": json.dumps(spans),
                "verified_span_count": 2, "verified_covered_residue_count": 20,
                "final_epitope_coverage": 0.2,
            }
        )
    return row


def _path_identity(path: Path) -> tuple[Any, ...] | None:
    if not os.path.lexists(path):
        return None
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if path.is_file():
        return ("file", path.stat().st_size, _sha_file(path))
    return ("directory", path.stat().st_dev, path.stat().st_ino)


def _write_tiny_source_snapshots(audit_dir: Path) -> tuple[Path, Path, Path]:
    """Run the real C1 and Tier1 producers against deterministic in-memory HTTP fixtures."""

    source_root = audit_dir / "sources"
    rcsb_payload = {
        "data": {
            "polymer_entities": [
                {
                    "rcsb_id": "1BBB_1",
                    "entity_poly": {
                        "pdbx_seq_one_letter_code_can": "C" * 100,
                        "rcsb_sample_sequence_length": 100,
                        "rcsb_entity_polymer_type": "Protein",
                    },
                    "rcsb_polymer_entity_container_identifiers": {
                        "entry_id": "1BBB", "entity_id": "1",
                    },
                    "polymer_entity_instances": [{
                        "rcsb_id": "1BBB.B",
                        "rcsb_polymer_entity_instance_container_identifiers": {
                            "entry_id": "1BBB", "entity_id": "1",
                            "asym_id": "B", "auth_asym_id": "B",
                        },
                    }],
                    "entry": {
                        "exptl": [{"method": "X-RAY DIFFRACTION"}],
                        "rcsb_entry_info": {"resolution_combined": [1.5]},
                        "rcsb_accession_info": {
                            "initial_release_date": "2020-01-01",
                            "revision_date": "2020-02-01",
                        },
                    },
                }
            ]
        }
    }

    def rcsb_transport(_method, url, _body, _headers, _timeout):
        if url == SEARCH_URL:
            body = json.dumps(
                {"total_count": 1, "result_set": [{"identifier": "1BBB_1"}]}
            ).encode()
        elif url == GRAPHQL_URL:
            body = json.dumps(rcsb_payload).encode()
        else:
            raise AssertionError(url)
        return SiftsHttpResponse(200, body, {"content-type": "application/json"})

    rcsb_dir = source_root / "rcsb"
    snapshot_rcsb_entities(
        output_dir=rcsb_dir, transport=rcsb_transport,
        sleep_fn=lambda _seconds: None, search_page_rows=10, graphql_batch_size=10,
        api_delay_s=0.0,
    )

    inputs = source_root / "tier1_inputs"
    inputs.mkdir(parents=True)
    ids = inputs / "ids.txt"
    ids.write_text("P12345\nQ12345\n")
    proteins = inputs / "protein_samples.parquet"
    spans = inputs / "span_records.parquet"
    pd.DataFrame([
        {"protein_id": "P12345", "allele": "HLA-DRB1*15:01",
         "protein_seq": "A" * 100, "sequence_length": 100},
        {"protein_id": "Q12345", "allele": "HLA-DRB1*15:01",
         "protein_seq": "D" * 100, "sequence_length": 100},
    ]).to_parquet(proteins, index=False)
    span_rows = []
    for source_id, amino_acid in (("P12345", "A"), ("Q12345", "D")):
        for start in (0, 20):
            span_rows.append(
                {"protein_id": source_id, "allele": "HLA-DRB1*15:01",
                 "start_0b": start, "end_0b": start + 10,
                 "peptide_seq": amino_acid * 10, "source": "fixture_iedb"}
            )
    pd.DataFrame(span_rows).to_parquet(spans, index=False)

    def uniprot_transport(_method, url, _body, _headers, _timeout):
        if url.endswith("/configure/idmapping/fields"):
            body = b'{"groups":[],"rules":[]}'
        else:
            body = (
                b"Entry\tSequence\tLength\nP12345\t" + b"A" * 100 + b"\t100\n"
                b"Q12345\t" + b"D" * 100 + b"\t100\n"
            )
        return SiftsHttpResponse(200, body, {"x-uniprot-release": "fixture"})

    def sifts_request(url, _timeout):
        source_id = url.rsplit("/", 1)[1]
        pdb_id, chain_id = (("1AAA", "A") if source_id == "P12345" else ("1AAC", "C"))
        body = json.dumps(
            {source_id: [{
                "pdb_id": pdb_id, "chain_id": chain_id,
                "unp_start": 1, "unp_end": 100, "start": 1, "end": 100,
                "resolution": 1.5, "experimental_method": "X-ray diffraction",
                "coverage": 1.0,
            }]}
        ).encode()
        return SiftsHttpResponse(200, body, {"content-type": "application/json"})

    tier1_dir = source_root / "tier1_preflight"
    run_tier1_source_preflight(
        allele="HLA-DRB1*15:01", test_ids_path=ids,
        protein_samples_path=proteins, span_records_path=spans,
        output_dir=tier1_dir, tier1_target=1,
        request_fn=sifts_request, uniprot_transport=uniprot_transport,
        sleep_fn=lambda _seconds: None, api_delay_s=0.0,
        command_argv=["tiny-fixture-producer"],
    )

    tier1_structures = {
        "1AAA": {"auth_chain": "A", "amino_acid": "A", "source_id": "P12345"},
        "1AAC": {"auth_chain": "C", "amino_acid": "D", "source_id": "Q12345"},
    }

    def identity_prefix(pdb_id: str, auth_chain: str, amino_acid: str) -> list[str]:
        # No `_pdbx_audit_revision_history`: production PDBe enriched mmCIF has none, and the
        # revision authority is the frozen C1 RCSB snapshot.
        monomer = {"A": "ALA", "D": "ASP"}[amino_acid]
        return [
            f"data_{pdb_id}", f"_entry.id {pdb_id}",
            "loop_", "_struct_asym.id", "_struct_asym.entity_id", f"{auth_chain} 1", "#",
            "loop_", "_pdbx_poly_seq_scheme.asym_id",
            "_pdbx_poly_seq_scheme.entity_id", "_pdbx_poly_seq_scheme.seq_id",
            "_pdbx_poly_seq_scheme.mon_id", "_pdbx_poly_seq_scheme.auth_seq_num",
            "_pdbx_poly_seq_scheme.pdb_ins_code",
            "_pdbx_poly_seq_scheme.pdb_strand_id",
            *[
                f"{auth_chain} 1 {position} {monomer} {position} ? {auth_chain}"
                for position in range(1, 101)
            ],
            "#",
        ]

    def enriched_cif(pdb_id: str) -> bytes:
        item = tier1_structures[pdb_id]
        chain = item["auth_chain"]
        amino_acid = item["amino_acid"]
        lines = identity_prefix(pdb_id, chain, amino_acid) + [
            "loop_", "_pdbx_sifts_xref_db.entity_id", "_pdbx_sifts_xref_db.asym_id",
            "_pdbx_sifts_xref_db.seq_id",
            "_pdbx_sifts_xref_db.mon_id_one_letter_code",
            "_pdbx_sifts_xref_db.unp_res", "_pdbx_sifts_xref_db.unp_num",
            "_pdbx_sifts_xref_db.unp_acc", "_pdbx_sifts_xref_db.observed",
            *[
                f"1 {chain} {position} {amino_acid} {amino_acid} {position} "
                f"{item['source_id']} Y"
                for position in range(1, 101)
            ],
            "#",
        ]
        return ("\n".join(lines) + "\n").encode()

    def coordinate_cif(pdb_id: str) -> bytes:
        item = tier1_structures[pdb_id]
        return _tiny_coordinate_cif_bytes(
            pdb_id=pdb_id, chain=str(item["auth_chain"]),
            amino_acid=str(item["amino_acid"]),
        )

    coordinate_cache = audit_dir / "sources" / "tier1_coordinates"

    def coordinate_provider(pdb_id: str, revision: str):
        path, _manifest = acquire_rcsb_mmcif(
            pdb_id=pdb_id, expected_revision_date=revision,
            cache_dir=coordinate_cache,
            request_fn=lambda _url, _timeout: SiftsHttpResponse(
                200, coordinate_cif(pdb_id), {"content-type": "chemical/x-mmcif"}
            ),
            sleep_fn=lambda _seconds: None,
        )
        return path, path.parent / "manifest.json"

    def residue_request(url: str, _timeout: float):
        pdb_id = next(
            pdb for pdb in tier1_structures if f"/{pdb.lower()}_updated.cif" in url
        )
        return SiftsHttpResponse(
            200, enriched_cif(pdb_id), {"content-type": "chemical/x-mmcif"}
        )

    residue_manifest = snapshot_tier1_residue_maps(
        source_attempts=pd.read_parquet(tier1_dir / "tier1_sifts_chain_attempts.parquet"),
        canonical_sources=pd.read_parquet(tier1_dir / "tier1_canonical_source_units.parquet"),
        output_dir=audit_dir / "tier1_residue_maps", candidate_root=audit_dir.parent,
        coordinate_provider=coordinate_provider, request_fn=residue_request,
        sleep_fn=lambda _seconds: None,
        frozen_revisions={pdb_id: "2020-02-01" for pdb_id in tier1_structures},
        command_argv=["tiny-residue-map-producer"],
    )
    return (
        rcsb_dir / "source_manifest.json", tier1_dir / "preflight_manifest.json",
        residue_manifest,
    )


def _write_tiny_nmp(
    audit_dir: Path, rows: list[dict[str, Any]], *, tier2_pid: str,
    nmp_install_paths: Mapping[str, Path], nmp_install_identity: Mapping[str, Any],
) -> dict[str, Path]:
    nmp_dir = audit_dir / "nmp"
    nmp_dir.mkdir()
    tool = nmp_install_paths["tool"]
    parameters = nmp_dir / "parameters.json"
    parameters.write_text(json.dumps(nmp_parameter_payload(
        stage="c7", allele="HLA-DRB1*15:01", peptide_lengths=list(range(12, 26)),
        install_manifest_sha256=nmp_install_identity["manifest_sha256"],
        batch_size=8, subprocess_timeout_s=600, max_lengths_per_call=4, n_workers=1,
    ), indent=2, sort_keys=True) + "\n")
    tier2 = next(row for row in rows if row["protein_id"] == tier2_pid)
    sequence = str(tier2["sequence"])
    queries = pd.DataFrame([{
        "protein_id": tier2_pid, "sequence": sequence,
        "sequence_sha256": tier2["sequence_sha256"],
    }])

    def score_shard(_queries, _shard_dir):
        return pd.DataFrame([
            {"protein_id": tier2_pid, "peptide_length": length,
             "start_0b": start, "end_0b": start + length,
             "peptide": sequence[start:start + length],
             "rank_el": 1.0 if (length, start) == (12, 0) else 3.0}
            for length in range(12, 26)
            for start in range(len(sequence) - length + 1)
        ])

    shard_manifest = write_nmp_sharded_evidence(
        output_dir=nmp_dir / "evidence", queries=queries, key_column="protein_id",
        sequence_column="sequence", sequence_sha_column="sequence_sha256", stage="c7",
        peptide_lengths=list(range(12, 26)), tool_path=tool,
        parameters_path=parameters, score_shard_fn=score_shard, shard_query_count=1,
    )
    shard_payload = json.loads(shard_manifest.read_text())
    raw = shard_manifest.parent / shard_payload["shards"][0]["files"]["raw_output"]["path"]
    coverage = 12 / len(sequence)
    tier2.update(
        {"nmp_status": "complete", "coverage_fraction": coverage,
         "nmp_query_sequence_sha256": tier2["sequence_sha256"],
         "nmp_evidence_sha256": _sha_file(raw), "nmp_tool_sha256": _sha_file(tool),
         "nmp_parameters_sha256": _sha_file(parameters), "selection_bin": 0}
    )
    return {"shard_manifest": shard_manifest, "tool": tool, "parameters": parameters}


def _write_tiny_nmp_install(audit_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    root = audit_dir / "nmp_install"
    root.mkdir()
    tool = root / "netMHCIIpan.fixture.bin"
    tool.write_bytes(b"fixture-netmhciipan-binary")
    data_root = root / "data"
    data_root.mkdir()
    (data_root / "model.dat").write_bytes(b"fixture-netmhciipan-model-data")
    manifest = root / "install_manifest.json"
    write_nmp_install_manifest(
        tool_path=tool, data_root=data_root, tool_version="fixture-NetMHCIIpan-4.3",
        manifest_path=manifest, effective_environment={"LC_ALL": "C"},
        identity_mode="fixture",
    )
    paths = {
        "tool": tool, "data_manifest": manifest,
        "contract_producer": Path(__file__).with_name("nmp.py"),
        "runner_producer": Path(__file__).parents[3] / "epitope_head" / "data" /
        "netmhciipan_runner.py",
    }
    return paths, {"manifest_sha256": _sha_file(manifest)}


def _write_tiny_mmseqs_references(audit_dir: Path) -> tuple[dict[str, Path], dict[str, str]]:
    root = audit_dir / "mmseqs"
    root.mkdir()
    tool = root / "mmseqs.fixture.bin"
    tool.write_bytes(b"fixture-mmseqs-binary")
    version = "fixture-mmseqs-13.45111"
    identity_path = root / "tool_identity.json"
    identity_path.write_text(json.dumps(
        {"schema_version": "if-benchmark-v3-mmseqs-tool/1",
         "tool_sha256": _sha_file(tool), "tool_version": version},
        indent=2, sort_keys=True
    ) + "\n")

    chain_set = root / "chain_set.jsonl"
    chain_set.write_text(json.dumps({"name": "CATH1", "seq": "G" * 100}) + "\n")
    cath_splits = root / "chain_set_splits.json"
    cath_splits.write_text(json.dumps({"train": ["CATH1"], "validation": [], "test": []}))
    cath_fasta = root / "cath_train.fasta"
    cath_manifest = root / "cath_reference_manifest.json"
    materialize_cath_train_reference(
        chain_set_jsonl=chain_set, splits_json=cath_splits,
        output_fasta=cath_fasta, manifest_path=cath_manifest,
    )

    head_train = root / "head_train_ids.txt"
    head_val = root / "head_val_ids.txt"
    head_test = root / "head_test_ids.txt"
    head_train.write_text("H1\n")
    head_val.write_text("H2\n")
    head_test.write_text("H3\n")
    head_proteins = root / "head_protein_samples.parquet"
    pd.DataFrame([
        {"protein_id": "H1", "allele": "HLA-DRB1*15:01", "protein_seq": "H" * 100},
        {"protein_id": "H2", "allele": "HLA-DRB1*15:01", "protein_seq": "I" * 100},
        {"protein_id": "H3", "allele": "HLA-DRB1*15:01", "protein_seq": "K" * 100},
    ]).to_parquet(head_proteins, index=False)
    head_fasta = root / "head_seen.fasta"
    head_manifest = root / "head_reference_manifest.json"
    materialize_head_seen_reference(
        split_id_paths={"train": head_train, "val": head_val, "test": head_test},
        protein_samples_path=head_proteins, allele="HLA-DRB1*15:01",
        output_fasta=head_fasta, manifest_path=head_manifest,
    )
    paths = {
        "tool": tool, "identity": identity_path,
        "reference_producer": Path(__file__).with_name("references.py"),
        "search_producer": Path(__file__).with_name("leakage.py"),
        "cath_reference_manifest": cath_manifest, "cath_reference_fasta": cath_fasta,
        "head_reference_manifest": head_manifest, "head_reference_fasta": head_fasta,
    }
    identities = {
        "tool_sha256": _sha_file(tool), "tool_version": version,
        "cath_reference_sha256": _sha_file(cath_fasta),
        "head_reference_sha256": _sha_file(head_fasta),
        "cath_chain_set_sha256": _sha_file(chain_set),
        "cath_splits_sha256": _sha_file(cath_splits),
    }
    return paths, identities


def _write_tiny_selection(
    audit_dir: Path, *, tier2_row: Mapping[str, Any],
    mmseqs_identity: Mapping[str, str], nmp_install_paths: Mapping[str, Path],
    nmp_install_identity: Mapping[str, Any],
) -> dict[str, Path]:
    selection_dir = audit_dir / "selection"
    selection_dir.mkdir()
    source_cath_queries = selection_dir / "source_cath_queries.parquet"
    source_query_frame = pd.DataFrame([{
        "protein_id": tier2_row["selection_unit_id"],
        "sequence": tier2_row["entity_sequence"],
        "sequence_sha256": tier2_row["entity_sequence_sha256"],
    }])
    source_query_frame.to_parquet(source_cath_queries, index=False)
    source_cath_fasta = selection_dir / "source_cath_queries.fasta"
    source_cath_fasta.write_text(
        f">{tier2_row['selection_unit_id']}\n{tier2_row['entity_sequence']}\n"
    )
    source_cath_raw = selection_dir / "source_cath.tsv"
    source_cath_raw.write_text("")
    source_cath_frame = build_mmseqs_sidecar(
        source_query_frame[["protein_id", "sequence"]],
        raw_tsv_path=source_cath_raw, query_fasta_path=source_cath_fasta,
        search_kind="cath", reference_sha256=mmseqs_identity["cath_reference_sha256"],
        tool_sha256=mmseqs_identity["tool_sha256"],
        tool_version=mmseqs_identity["tool_version"], cov_mode=0, coverage=0.8,
        min_seq_id=0.3, path_root=audit_dir.parent,
    )
    source_cath_sidecar = selection_dir / "source_cath_sidecar.parquet"
    source_cath_frame.to_parquet(source_cath_sidecar, index=False)
    c5_tool = nmp_install_paths["tool"]
    c5_parameters = selection_dir / "c5_nmp_parameters.json"
    c5_parameters.write_text(json.dumps(nmp_parameter_payload(
        stage="c5", allele="HLA-DRB1*15:01", peptide_lengths=[15],
        install_manifest_sha256=nmp_install_identity["manifest_sha256"],
        batch_size=8, subprocess_timeout_s=600, max_lengths_per_call=4, n_workers=1,
    ), indent=2, sort_keys=True) + "\n")
    source_sequence = str(tier2_row["entity_sequence"])
    c5_queries = pd.DataFrame([{
        "selection_unit_id": tier2_row["selection_unit_id"],
        "source_sequence": source_sequence,
        "source_sequence_sha256": tier2_row["entity_sequence_sha256"],
    }])

    def score_c5(_queries, _shard_dir):
        return pd.DataFrame([
            {"protein_id": tier2_row["selection_unit_id"], "peptide_length": 15,
             "start_0b": start, "end_0b": start + 15,
             "peptide": source_sequence[start:start + 15],
             "rank_el": 1.0 if start == 0 else 3.0}
            for start in range(len(source_sequence) - 15 + 1)
        ])

    c5_shard_manifest = write_nmp_sharded_evidence(
        output_dir=selection_dir / "c5_nmp_evidence", queries=c5_queries,
        key_column="selection_unit_id", sequence_column="source_sequence",
        sequence_sha_column="source_sequence_sha256", stage="c5", peptide_lengths=[15],
        tool_path=c5_tool, parameters_path=c5_parameters,
        score_shard_fn=score_c5, shard_query_count=1,
    )
    c5_score_identity = hashlib.sha256(canonical_json(
        {"manifest": _sha_file(c5_shard_manifest), "tool": _sha_file(c5_tool),
         "parameters": _sha_file(c5_parameters)}
    )).hexdigest()
    c5_eligible = pd.DataFrame([{
        "selection_unit_id": tier2_row["selection_unit_id"],
        "source_sequence": source_sequence,
        "source_sequence_sha256": tier2_row["entity_sequence_sha256"],
        "score_identity_sha256": c5_score_identity,
        "selection_source_coverage_fraction_15": 0.15,
    }])
    c5 = deterministic_binned_sample(
        c5_eligible, value_col="selection_source_coverage_fraction_15", target=1,
        n_bins=20, mode="uniform", stage="c5", min_per_populated_bin=0, seed=42,
    )
    c7_eligible = pd.DataFrame([{
        "selection_unit_id": tier2_row["selection_unit_id"],
        "protein_id": tier2_row["protein_id"], "sequence": tier2_row["sequence"],
        "sequence_sha256": tier2_row["sequence_sha256"],
        "cath_query_sha256": tier2_row["sequence_sha256"],
        "cath_reference_sha256": "r" * 64, "cath_tool_sha256": "t" * 64,
        "cath_parameters_sha256": "p" * 64, "cath_overlap_flag": False,
        "nmp_evidence_sha256": tier2_row["nmp_evidence_sha256"],
        "coverage_fraction": tier2_row["coverage_fraction"],
    }])
    c7 = deterministic_binned_sample(
        c7_eligible, value_col="coverage_fraction", target=1, n_bins=20,
        mode="gaussian", stage="c7", min_per_populated_bin=1, seed=42,
    )
    paths: dict[str, Path] = {}
    for stage, frame, result in (("c5", c5_eligible, c5), ("c7", c7_eligible, c7)):
        eligible_path = selection_dir / f"{stage}_eligible.parquet"
        result_path = selection_dir / f"{stage}_result.json"
        frame.to_parquet(eligible_path, index=False)
        result_path.write_text(json.dumps(
            {"schema_version": "if-benchmark-v3-binned-sample/1",
             "protocol_profile": "tiny_fixture",
             "result": binned_sample_to_dict(result)}, indent=2, sort_keys=True
        ) + "\n")
        paths[f"{stage}_eligible"] = eligible_path
        paths[f"{stage}_result"] = result_path
    expansion = selection_dir / "c5_expansion.json"
    expansion.write_text(json.dumps(
        {"schema_version": "if-benchmark-v3-c5-expansion/1",
         "preregistered_ladder": [1],
         "attempts": [{"target": 1, "observed_final_clean_capacity": 1}]},
        indent=2, sort_keys=True
    ) + "\n")
    paths["c5_expansion"] = expansion
    prior_rungs = selection_dir / "c5_prior_rungs.json"
    prior_rungs.write_text(json.dumps(
        {"schema_version": "if-benchmark-v3-c5-prior-rungs/1", "rungs": []},
        indent=2, sort_keys=True
    ) + "\n")
    paths["c5_prior_rungs"] = prior_rungs
    paths.update({
        "c5_nmp_shard_manifest": c5_shard_manifest, "c5_nmp_tool": c5_tool,
        "c5_nmp_parameters": c5_parameters,
    })

    final_cath_queries = selection_dir / "final_cath_queries.parquet"
    pd.DataFrame([{
        "selection_unit_id": tier2_row["selection_unit_id"],
        "protein_id": tier2_row["protein_id"], "sequence": tier2_row["sequence"],
    }]).to_parquet(final_cath_queries, index=False)
    final_cath_fasta = selection_dir / "final_cath_queries.fasta"
    final_cath_fasta.write_text(f">{tier2_row['protein_id']}\n{tier2_row['sequence']}\n")
    final_cath_raw = selection_dir / "final_cath.tsv"
    final_cath_raw.write_text("")
    final_cath = build_mmseqs_sidecar(
        pd.DataFrame([{"protein_id": tier2_row["protein_id"],
                       "sequence": tier2_row["sequence"]}]),
        raw_tsv_path=final_cath_raw, query_fasta_path=final_cath_fasta,
        search_kind="cath", reference_sha256=mmseqs_identity["cath_reference_sha256"],
        tool_sha256=mmseqs_identity["tool_sha256"],
        tool_version=mmseqs_identity["tool_version"],
        cov_mode=0, coverage=0.8, min_seq_id=0.3, path_root=audit_dir.parent,
    )
    final_cath_sidecar = selection_dir / "final_cath_sidecar.parquet"
    final_cath.to_parquet(final_cath_sidecar, index=False)
    measured = final_cath.iloc[0]
    c7_eligible.loc[:, "cath_query_sha256"] = measured["query_sequence_sha256"]
    c7_eligible.loc[:, "cath_reference_sha256"] = measured["reference_sha256"]
    c7_eligible.loc[:, "cath_tool_sha256"] = measured["tool_sha256"]
    c7_eligible.loc[:, "cath_parameters_sha256"] = measured["parameters_sha256"]
    c7_eligible.loc[:, "cath_overlap_flag"] = bool(measured["overlap_flag"])
    c7_replayed = deterministic_binned_sample(
        c7_eligible, value_col="coverage_fraction", target=1, n_bins=20,
        mode="gaussian", stage="c7", min_per_populated_bin=1, seed=42,
    )
    c7_eligible.to_parquet(paths["c7_eligible"], index=False)
    paths["c7_result"].write_text(json.dumps(
        {"schema_version": "if-benchmark-v3-binned-sample/1",
         "protocol_profile": "tiny_fixture",
         "result": binned_sample_to_dict(c7_replayed)}, indent=2, sort_keys=True
    ) + "\n")
    paths["final_cath_queries"] = final_cath_queries
    paths["final_cath_sidecar"] = final_cath_sidecar
    paths["source_cath_queries"] = source_cath_queries
    paths["source_cath_sidecar"] = source_cath_sidecar
    return paths


def _write_tiny_ledgers(audit_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Path]:
    ledger_dir = audit_dir / "ledgers"
    ledger_dir.mkdir()
    tier1_residue_candidates = pd.read_parquet(
        audit_dir / "tier1_residue_maps" / "tier1_residue_map_candidates.parquet"
    )
    tier1_residue_by_key = {
        (str(item.source_id), str(item.label_asym_id)): item
        for item in tier1_residue_candidates.itertuples(index=False)
    }
    tier2_coordinate_path, _tier2_coordinate_manifest = acquire_rcsb_mmcif(
        pdb_id="1BBB", expected_revision_date="2020-02-01",
        cache_dir=audit_dir / "sources" / "tier2_coordinates",
        request_fn=lambda _url, _timeout: SiftsHttpResponse(
            200, _tiny_coordinate_cif_bytes(
                pdb_id="1BBB", chain="B", amino_acid="C"
            ), {"content-type": "chemical/x-mmcif"},
        ),
        sleep_fn=lambda _seconds: None,
    )
    attempts = []
    for row in rows:
        tier1_source = None
        if row["selected_tier_memberships"] == ["tier1"]:
            tier1_source = tier1_residue_by_key[
                (str(row["source_uniprot_id"]), str(row["label_asym_id"]))
            ]
        attempts.append(
            {"selection_unit_id": row["selection_unit_id"],
             "source_tier": row["selected_tier_memberships"][0],
             "protein_id": row["protein_id"], "sequence": row["sequence"],
             "source_uniprot_id": row["source_uniprot_id"],
             "sifts_item_index": 0 if row["selected_tier_memberships"] == ["tier1"] else None,
             "fallback_rank": 1, "attempt_order": 1,
             "attempt_status": "viable",
             "source_revision_date": (
                 "2020-02-01" if row["selected_tier_memberships"] == ["tier2"]
                 else str(tier1_source.source_revision_date)
             ),
             "download_sha256": (
                 _sha_file(tier2_coordinate_path) if tier1_source is None
                 else str(tier1_source.coordinate_cif_sha256)
             ),
             "rcsb_entity_id": row["rcsb_entity_id"],
             "auth_chain_id": row["auth_chain_id"], "label_asym_id": row["label_asym_id"],
             "viable": True, "if_sequence_coverage": row["if_sequence_coverage"],
             "resolution": row["resolution"], "sequence_sha256": row["sequence_sha256"],
             "structure_sha256": row["structure_sha256"], "failure_reason": None}
        )
    attempts_path = ledger_dir / "chain_attempts.parquet"
    attempt_frame = pd.DataFrame(attempts)
    attempt_frame.to_parquet(attempts_path, index=False)
    download_evidence = []
    for attempt in attempts:
        evidence = dict(attempt)
        if attempt["source_tier"] == "tier2":
            evidence.update({
                "coordinate_cif_path": tier2_coordinate_path,
                "coordinate_manifest_path": tier2_coordinate_path.parent / "manifest.json",
            })
        else:
            source = tier1_residue_by_key[
                (str(attempt["source_uniprot_id"]), str(attempt["label_asym_id"]))
            ]
            evidence.update({
                "coordinate_cif_path": audit_dir.parent / str(source.coordinate_cif_path),
                "coordinate_manifest_path": (
                    audit_dir.parent / str(source.coordinate_manifest_path)
                ),
            })
        download_evidence.append(evidence)
    structure_downloads_path = ledger_dir / "structure_downloads.parquet"
    write_structure_download_registry(
        structure_downloads_path, candidate_root=audit_dir.parent,
        chain_attempts=attempt_frame, evidence_rows=download_evidence,
    )
    tier1_candidates = []
    for row in rows:
        if row["selected_tier_memberships"] != ["tier1"]:
            continue
        tier1_candidates.append({
            **row, "projection_valid": True,
            "source_uniprot_id": row["source_uniprot_id"],
            "final_epitope_coverage": row["final_epitope_coverage"],
            "verified_span_count": row["verified_span_count"],
            "verified_covered_residue_count": row["verified_covered_residue_count"],
        })
    tier1_pool_path = ledger_dir / "tier1_candidate_pool.parquet"
    pd.DataFrame(tier1_candidates).to_parquet(tier1_pool_path, index=False)
    _t1_primary, _t1_diagnostic, tier1_internal_ledger = finalize_tier1_pool(
        tier1_candidates, target=1
    )
    tier1_internal_path = ledger_dir / "tier1_internal_collision_ledger.json"
    tier1_internal_path.write_text(json.dumps(
        {"schema_version": "if-benchmark-v3-tier1-internal-collisions/1",
         "rows": tier1_internal_ledger}, indent=2, sort_keys=True
    ) + "\n")
    def collision_row(row: Mapping[str, Any], *, t2: bool = False) -> dict[str, Any]:
        out = {
            "protein_id": row["protein_id"], "selection_unit_id": row["selection_unit_id"],
            "rcsb_entity_id": row["rcsb_entity_id"],
            "sequence": row["sequence"], "sequence_sha256": row["sequence_sha256"],
            "source_pool_memberships": list(row["source_pool_memberships"]),
            "selected_tier_memberships": list(row["selected_tier_memberships"]),
        }
        if t2:
            out["group_entity_ids"] = [row["rcsb_entity_id"]]
        return out
    t1_primary = [collision_row(row) for row in rows if row["protein_id"] == "T1A"]
    t1_diagnostic = [collision_row(row) for row in rows if row["protein_id"] == "D1C"]
    t2_rows = [collision_row(row, t2=True) for row in rows if row["protein_id"] == "T2B"]
    replayed = resolve_tier1_tier2_collisions(
        t1_primary=t1_primary, t1_diagnostic=t1_diagnostic, t2_rows=t2_rows
    )
    collision_path = ledger_dir / "collision_replay.json"
    collision_path.write_text(json.dumps(
        {"schema_version": "if-benchmark-v3-collision-replay/1",
         "inputs": {"tier1_primary": t1_primary, "tier1_diagnostic": t1_diagnostic,
                    "tier2_rows": t2_rows},
         "outputs": {"tier1_primary": replayed[0], "tier1_diagnostic": replayed[1],
                     "tier2_rows": replayed[2]}, "collision_ledger": replayed[3]},
        indent=2, sort_keys=True
    ) + "\n")
    dedup_source_rows = [row for row in rows if row["protein_id"] == "T2B"]
    dedup_inputs = [
        {key: row[key] for key in (
            "protein_id", "selection_unit_id", "sequence_sha256", "if_sequence_coverage", "resolution",
            "rcsb_entity_id", "auth_chain_id", "label_asym_id",
        )}
        for row in dedup_source_rows
    ]
    dedup_ledger = [
        {"protein_id": row["protein_id"], "sequence_sha256": row["sequence_sha256"],
         "representative_protein_id": row["protein_id"], "action": "keep"}
        for row in sorted(dedup_source_rows, key=lambda value: value["protein_id"])
    ]
    dedup_path = ledger_dir / "final_dedup_replay.json"
    dedup_path.write_text(json.dumps(
        {"schema_version": "if-benchmark-v3-final-dedup-replay/1",
         "inputs": dedup_inputs,
         "selected_protein_ids": sorted(row["protein_id"] for row in dedup_source_rows),
         "dedup_ledger": dedup_ledger}, indent=2, sort_keys=True
    ) + "\n")
    return {"chain_attempts": attempts_path, "collision_replay": collision_path,
            "final_dedup_replay": dedup_path,
            "tier1_candidate_pool": tier1_pool_path,
            "tier1_internal_collision_ledger": tier1_internal_path,
            "structure_downloads": structure_downloads_path}


def run_tiny_e2e(*, build_root: Path, build_id: str, main_alias: Path) -> Path:
    """Run the real producer/replay validator stack on deterministic local fixture inputs."""

    candidate = Path(build_root) / build_id / "HLA-DRB1_15_01"
    if candidate.exists():
        raise FileExistsError(candidate)
    alias_before = _path_identity(Path(main_alias))
    pdb_root = candidate / "pdbs_if_ready"
    annotation_dir = candidate / "annotations"
    audit_dir = candidate / "audit"
    for directory in (pdb_root, annotation_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rcsb_manifest, tier1_manifest, tier1_residue_manifest = _write_tiny_source_snapshots(
        audit_dir
    )
    rows: list[dict[str, Any]] = []
    t2_selection_unit = f"t2-seq:{_sha_text('C' * 100)}"
    specs = [
        ("T1A", "tier1", "A", "primary_generalization", False,
         "t1:HLA-DRB1_15_01:P12345", "1AAA", "1", "A", "P12345"),
        ("T2B", "tier2", "C", "primary_generalization", False,
         t2_selection_unit, "1BBB", "1", "B", None),
        ("D1C", "tier1", "D", "tier1_overlap_diagnostic", True,
         "t1:HLA-DRB1_15_01:Q12345", "1AAC", "1", "C", "Q12345"),
    ]
    for pid, tier, aa, role, overlap, unit, pdb_id, entity_id, chain_id, uniprot_id in specs:
        path = pdb_root / f"{pid}.cif"
        path.write_bytes(_tiny_coordinate_cif_bytes(
            pdb_id=pdb_id, chain="A", amino_acid=aa,
        ))
        rows.append(
            _tiny_row(
                pid=pid, tier=tier, amino_acid=aa, structure_path=path.name,
                structure_sha256=_sha_file(path), role=role, cath_overlap=overlap,
                selection_unit_id=unit, pdb_id=pdb_id, entity_id=entity_id,
                auth_chain_id=chain_id, source_uniprot_id=uniprot_id,
            )
        )
    mmseqs_paths, mmseqs_identity = _write_tiny_mmseqs_references(audit_dir)
    nmp_install_paths, nmp_install_seed = _write_tiny_nmp_install(audit_dir)
    nmp_paths = _write_tiny_nmp(
        audit_dir, rows, tier2_pid="T2B",
        nmp_install_paths=nmp_install_paths, nmp_install_identity=nmp_install_seed,
    )
    selection_paths = _write_tiny_selection(
        audit_dir, tier2_row=next(row for row in rows if row["protein_id"] == "T2B"),
        mmseqs_identity=mmseqs_identity, nmp_install_paths=nmp_install_paths,
        nmp_install_identity=nmp_install_seed,
    )
    ledger_paths = _write_tiny_ledgers(audit_dir, rows)

    family_dir = audit_dir / "family"
    family_dir.mkdir()
    family_fasta = family_dir / "primary.fasta"
    primary_rows = [row for row in rows if row["evaluation_role"] == "primary_generalization"]
    family_fasta.write_text("".join(
        f">{row['protein_id']}\n{row['sequence']}\n"
        for row in sorted(primary_rows, key=lambda value: value["protein_id"])
    ))
    family_tsv = family_dir / "clusters.tsv"
    family_tsv.write_text("T1A\tT1A\nT2B\tT2B\n")
    family_tool = mmseqs_paths["tool"]
    family_params = family_dir / "parameters.json"
    family_command = [
        "mmseqs", "easy-cluster", "final.fasta", "clu", "tmp",
        "--min-seq-id", "0.3", "--cov-mode", "0", "-c", "0.8",
        "--cluster-mode", "0", "--threads", "1",
    ]
    family_params.write_text(json.dumps(
        {"schema_version": "if-benchmark-v3-family-parameters/1",
         "min_seq_id": 0.3, "cov_mode": 0, "coverage": 0.8,
         "cluster_mode": 0, "threads": 1,
         "tool_sha256": _sha_file(family_tool),
         "tool_version": mmseqs_identity["tool_version"],
         "command_argv": family_command, "effective_defaults": {}},
        indent=2, sort_keys=True
    ) + "\n")
    for row in primary_rows:
        row["family_cluster_id"] = stable_family_cluster_id([row["sequence_sha256"]])

    import torch

    head_checkpoint = annotation_dir / "epoch_24.fixture.pt"
    head_config = annotation_dir / "resolved_config.fixture.yaml"
    head_summary = annotation_dir / "run_summary.fixture.json"
    torch.save({
        "metadata": {
            "epoch": 24, "global_step": 10055,
            "config_hash": "ad6ac404027b", "manifest_version": "v1.1",
        },
        "model_state_dict": {},
    }, head_checkpoint)
    head_config.write_text("ablation_encoder_id: LC1\nmanifest_version: v1.1\n")
    head_summary.write_text(json.dumps({
        "allele": "HLA-DRB1*15:01", "final_epoch": 24, "global_steps": 10055,
        "config_hash": "ad6ac404027b", "manifest_version": "v1.1",
    }, sort_keys=True) + "\n")
    head_model_configs = annotation_dir / "head_model_configs"
    head_model_configs.mkdir()
    for name in ("model.yaml", "model_ablation.yaml", "inference.yaml"):
        (head_model_configs / name).write_text("fixture: true\n")
    approved_sources = audit_dir / "approved_sources.json"
    approved_payload = {
        "schema_version": "if-benchmark-v3-approved-sources/1",
        "protocol_profile": "tiny_fixture",
        "head": {
            "allele": "HLA-DRB1*15:01", "fixed_epoch": 24,
            "checkpoint_sha256": _sha_file(head_checkpoint),
            "config_sha256": _sha_file(head_config),
            "run_summary_sha256": _sha_file(head_summary),
            "checkpoint_config_hash": None,
            "checkpoint_global_step": None,
            "checkpoint_manifest_version": None,
        },
        "cath": {
            "release": "CATH4.3",
            "chain_set_sha256": mmseqs_identity["cath_chain_set_sha256"],
            "splits_sha256": mmseqs_identity["cath_splits_sha256"],
        },
    }
    approved_payload["manifest_sha256"] = hashlib.sha256(
        canonical_json(approved_payload)
    ).hexdigest()
    approved_sources.write_text(json.dumps(approved_payload, indent=2, sort_keys=True) + "\n")

    primary = pd.DataFrame(rows[:2])
    diagnostic = pd.DataFrame(rows[2:])
    combined = pd.concat([primary, diagnostic], ignore_index=True)
    queries = combined[["protein_id", "sequence"]]
    query_fasta = audit_dir / "final_queries.fasta"
    query_fasta.write_text(
        "".join(
            f">{row.protein_id}\n{row.sequence}\n"
            for row in queries.sort_values("protein_id").itertuples(index=False)
        )
    )
    cath_raw = audit_dir / "cath.tsv"
    cath_raw.write_text("D1C\tCATH1\t0.4\t0.8\t0.8\n")
    cath = build_mmseqs_sidecar(
        queries, raw_tsv_path=cath_raw, query_fasta_path=query_fasta,
        search_kind="cath", reference_sha256=mmseqs_identity["cath_reference_sha256"],
        tool_sha256=mmseqs_identity["tool_sha256"],
        tool_version=mmseqs_identity["tool_version"],
        cov_mode=0, coverage=0.8, min_seq_id=0.3, path_root=candidate,
    )
    head_raw = audit_dir / "head_homology.tsv"
    head_raw.write_text("T1A\tH1\t0.3\t0.8\t0.8\n")
    head_homology = build_mmseqs_sidecar(
        queries, raw_tsv_path=head_raw, query_fasta_path=query_fasta,
        search_kind="head", reference_sha256=mmseqs_identity["head_reference_sha256"],
        tool_sha256=mmseqs_identity["tool_sha256"],
        tool_version=mmseqs_identity["tool_version"],
        cov_mode=2, coverage=0.8, min_seq_id=0.3, path_root=candidate,
    )
    head = pd.DataFrame(
        [
            {
                "dataset_release_id": "tiny-v3-release", "protein_id": row.protein_id,
                "sequence_sha256": row.sequence_sha256,
                "head_checkpoint_sha256": _sha_file(head_checkpoint),
                "head_config_sha256": _sha_file(head_config),
                "global_risk": 0.0, "status": "complete",
            }
            for row in combined.itertuples()
        ]
    )
    sequence_sha_by_id = combined.set_index("protein_id")["sequence_sha256"]
    head_homology["dataset_release_id"] = "tiny-v3-release"
    head_homology["sequence_sha256"] = head_homology["protein_id"].map(
        sequence_sha_by_id
    )
    head_homology["head_checkpoint_sha256"] = _sha_file(head_checkpoint)
    load = pd.DataFrame(
        [
            {
                "protein_id": row.protein_id, "sequence_sha256": row.sequence_sha256,
                "structure_sha256": row.structure_sha256,
                "loaded_sequence_sha256": row.sequence_sha256,
                "loaded_length": row.sequence_length, "status": "pass",
            }
            for row in combined.itertuples()
        ]
    )
    main_alias = Path(main_alias).absolute()
    warehouse_root = Path(build_root).resolve().parent
    expected_alias = warehouse_root / "if_ready" / "main" / "HLA-DRB1_15_01"
    if main_alias != expected_alias:
        raise ValueError("tiny e2e main_alias must be the allele-specific warehouse target")
    if os.path.lexists(main_alias):
        raise ValueError("tiny first-release fixture requires an absent allele main alias")
    descendant_inventory = audit_dir / "descendant_inventory.json"
    write_descendant_inventory(
        descendant_inventory, warehouse_root=warehouse_root,
        allele_tag="HLA-DRB1_15_01", bindings=[],
    )

    publication_context = audit_dir / "publication_context.json"
    releases_root = warehouse_root / "releases"
    publication_context.write_text(json.dumps(
        {"schema_version": "if-benchmark-v3-publication-context/1",
         "release_id": "tiny-v3-release", "allele_tag": "HLA-DRB1_15_01",
         "mode": "first_release", "warehouse_root": str(warehouse_root),
         "candidate_root": str(candidate.resolve()),
         "releases_root": str(releases_root), "allele_main_alias": str(main_alias),
         "main_alias_identity": None,
         "prior_release_manifest": None, "prior_manifest_sha256": None},
        indent=2, sort_keys=True
    ) + "\n")

    release_table_paths = {
        "primary": candidate / "cohort_if_ready.parquet",
        "diagnostic": candidate / "tier1_overlap_diagnostic_if_ready.parquet",
        "cath": annotation_dir / "cath.parquet", "head": annotation_dir / "head.parquet",
        "head_homology": annotation_dir / "head_homology.parquet",
        "load_coords": audit_dir / "load_coords.parquet",
    }
    for label, frame in (
        ("primary", primary), ("diagnostic", diagnostic), ("cath", cath), ("head", head),
        ("head_homology", head_homology), ("load_coords", load),
    ):
        frame.to_parquet(release_table_paths[label], index=False)

    head_manifest = annotation_dir / "head.manifest.json"
    head_manifest_payload = {
        "schema_version": "if-benchmark-v3-head-annotation/1",
        "allele": "HLA-DRB1*15:01", "fixed_epoch": 24,
        "checkpoint_sha256": _sha_file(head_checkpoint),
        "resolved_config_sha256": _sha_file(head_config),
        "run_summary_sha256": _sha_file(head_summary),
        "model_config_files": {
            name: _sha_file(head_model_configs / name)
            for name in ("inference.yaml", "model.yaml", "model_ablation.yaml")
        },
        "cohort_inputs": [
            _head_path_identity(release_table_paths[label], root=annotation_dir)
            for label in ("primary", "diagnostic")
        ],
        "device": "cpu", "window_batch_size": 1,
        "row_count": len(head), "output_path": "head.parquet",
        "output_sha256": _sha_file(annotation_dir / "head.parquet"),
    }
    head_manifest_payload["manifest_sha256"] = hashlib.sha256(
        canonical_json(head_manifest_payload)
    ).hexdigest()
    head_manifest.write_text(
        json.dumps(head_manifest_payload, indent=2, sort_keys=True) + "\n"
    )

    evidence_bundle = audit_dir / "release_evidence.json"
    write_release_evidence_bundle(
        evidence_bundle, candidate_root=candidate, protocol_profile="tiny_fixture",
        sections={
            "source": {
                "rcsb_manifest": rcsb_manifest, "tier1_manifest": tier1_manifest,
                "tier1_residue_manifest": tier1_residue_manifest,
            },
            "ledgers": ledger_paths,
            "selection": selection_paths,
            "nmp": nmp_paths,
            "nmp_install": nmp_install_paths,
            "family": {"query_fasta": family_fasta, "cluster_tsv": family_tsv,
                       "tool": family_tool, "parameters": family_params},
            "mmseqs": mmseqs_paths,
            "head": {
                "checkpoint": head_checkpoint, "config": head_config,
                "run_summary": head_summary, "annotation_manifest": head_manifest,
                "model_yaml": head_model_configs / "model.yaml",
                "model_ablation_yaml": head_model_configs / "model_ablation.yaml",
                "inference_yaml": head_model_configs / "inference.yaml",
                "cohort_primary": release_table_paths["primary"],
                "cohort_diagnostic": release_table_paths["diagnostic"],
            },
            "approved_sources": {"manifest": approved_sources},
            "descendants": {"inventory": descendant_inventory},
            "publication": {"context": publication_context},
            "release_tables": release_table_paths,
        },
    )
    gates = validate_release_tables(
        primary, diagnostic, cath, head, head_homology, load,
        evidence_bundle_path=evidence_bundle, candidate_root=candidate, pdb_root=pdb_root,
        release_id="tiny-v3-release", allele="HLA-DRB1*15:01",
        tier1_target=1, tier2_target=1,
    )
    (audit_dir / "release_gates.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n"
    )
    write_dataset_manifest(
        candidate,
        {
            "dataset_release_id": "tiny-v3-release",
            "protocol_id": "if-benchmark-test-set/3",
            "fixture": True,
            "release_gates": gates,
        },
    )
    if _path_identity(Path(main_alias)) != alias_before:
        raise RuntimeError("tiny e2e changed main alias")
    return candidate
