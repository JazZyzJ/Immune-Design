"""Reduced-schema integration of every registered v3 scientific stage producer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from inverse_folding.evaluation.if_benchmark_v3 import stage_execution as stages
from inverse_folding.evaluation.if_benchmark_v3.nmp import (
    NMP_WINDOW_COLUMNS,
    write_nmp_install_manifest,
)
from inverse_folding.evaluation.if_benchmark_v3.preflight import (
    SiftsHttpResponse,
    run_tier1_source_preflight,
)
from inverse_folding.evaluation.if_benchmark_v3.production import STAGES


ALLELE = "HLA-DRB1*15:01"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()), "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _coordinate_cif(pdb_id: str, sequence: str, *, revision: str = "2020-02-01") -> bytes:
    monomer = {"A": "ALA", "C": "CYS"}[sequence[0]]
    lines = [
        f"data_{pdb_id}", f"_entry.id {pdb_id}",
        "loop_", "_pdbx_audit_revision_history.ordinal",
        "_pdbx_audit_revision_history.revision_date", f"1 {revision}", "#",
        "loop_", "_entity_poly.entity_id", "_entity_poly.pdbx_seq_one_letter_code_can",
        f"1 {sequence}", "#",
        "loop_", "_struct_asym.id", "_struct_asym.entity_id", "A 1", "#",
        "loop_", "_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.entity_id",
        "_pdbx_poly_seq_scheme.seq_id", "_pdbx_poly_seq_scheme.mon_id",
        "_pdbx_poly_seq_scheme.auth_seq_num", "_pdbx_poly_seq_scheme.pdb_ins_code",
        "_pdbx_poly_seq_scheme.pdb_strand_id",
        *[f"A 1 {index} {monomer} {index} ? A" for index in range(1, len(sequence) + 1)],
        "#", "loop_", "_atom_site.group_PDB", "_atom_site.id",
        "_atom_site.type_symbol", "_atom_site.label_atom_id", "_atom_site.label_alt_id",
        "_atom_site.label_comp_id", "_atom_site.label_asym_id",
        "_atom_site.label_entity_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
        "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge", "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id", "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id", "_atom_site.pdbx_PDB_model_num",
    ]
    serial = 1
    for position in range(1, len(sequence) + 1):
        for atom_index, atom in enumerate(("N", "CA", "C", "O")):
            element = "C" if atom in {"CA", "C"} else atom
            lines.append(
                f"ATOM {serial} {element} {atom} . {monomer} A 1 {position} ? "
                f"{position:.1f} {atom_index:.1f} 0.0 1.0 10.0 ? {position} "
                f"{monomer} A {atom} 1"
            )
            serial += 1
    lines.append("#")
    return ("\n".join(lines) + "\n").encode()


def _enriched_sifts(sequence: str) -> bytes:
    lines = [
        "data_1AAA", "_entry.id 1AAA", "loop_",
        "_pdbx_audit_revision_history.ordinal",
        "_pdbx_audit_revision_history.revision_date", "1 2020-02-01", "#",
        "loop_", "_struct_asym.id", "_struct_asym.entity_id", "A 1", "#",
        "loop_", "_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.entity_id",
        "_pdbx_poly_seq_scheme.seq_id", "_pdbx_poly_seq_scheme.mon_id",
        "_pdbx_poly_seq_scheme.auth_seq_num", "_pdbx_poly_seq_scheme.pdb_ins_code",
        "_pdbx_poly_seq_scheme.pdb_strand_id",
        *[f"A 1 {index} ALA {index} ? A" for index in range(1, len(sequence) + 1)],
        "#", "loop_", "_pdbx_sifts_xref_db.entity_id",
        "_pdbx_sifts_xref_db.asym_id", "_pdbx_sifts_xref_db.seq_id",
        "_pdbx_sifts_xref_db.mon_id_one_letter_code", "_pdbx_sifts_xref_db.unp_res",
        "_pdbx_sifts_xref_db.unp_num", "_pdbx_sifts_xref_db.unp_acc",
        "_pdbx_sifts_xref_db.observed",
        *[f"1 A {index} A A {index} P12345 y" for index in range(1, len(sequence) + 1)],
        "#",
    ]
    return ("\n".join(lines) + "\n").encode()


def _preflight(tmp_path: Path) -> Path:
    sequence = "A" * 120
    ids = tmp_path / "tier1_ids.txt"
    proteins = tmp_path / "tier1_proteins.parquet"
    spans = tmp_path / "tier1_spans.parquet"
    ids.write_text("P12345\n")
    pd.DataFrame([{
        "protein_id": "P12345", "allele": ALLELE,
        "protein_seq": sequence, "sequence_length": len(sequence),
    }]).to_parquet(proteins, index=False)
    pd.DataFrame([
        {"protein_id": "P12345", "allele": ALLELE, "start_0b": start,
         "end_0b": start + 15, "peptide_seq": "A" * 15, "source": "iedb"}
        for start in (0, 20)
    ]).to_parquet(spans, index=False)
    sifts = json.dumps({"P12345": [{
        "pdb_id": "1aaa", "chain_id": "A", "unp_start": 1, "unp_end": 120,
        "start": 1, "end": 120, "resolution": 2.0,
        "experimental_method": "X-ray diffraction", "coverage": 1.0,
    }]}).encode()

    def uniprot(_method, url, _body, _headers, _timeout):
        if url.endswith("/configure/idmapping/fields"):
            return SiftsHttpResponse(200, b'{"groups":[],"rules":[]}', {})
        return SiftsHttpResponse(
            200, b"Entry\tSequence\tLength\nP12345\t" + sequence.encode() + b"\t120\n", {}
        )

    output = tmp_path / "preflight"
    run_tier1_source_preflight(
        allele=ALLELE, test_ids_path=ids, protein_samples_path=proteins,
        span_records_path=spans, output_dir=output, tier1_target=1,
        request_fn=lambda _url, _timeout: SiftsHttpResponse(200, sifts, {}),
        uniprot_transport=uniprot, sleep_fn=lambda _seconds: None, api_delay_s=0.0,
        command_argv=["fixture-preflight"],
    )
    return output / "preflight_manifest.json"


def _mmseqs_tool(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "if args[0]=='version': print('13.45111'); raise SystemExit(0)\n"
        "if args[-1]=='-h': print('--min-seq-id --cov-mode --format-output --threads defaults'); raise SystemExit(0)\n"
        "if args[0]=='easy-search': Path(args[3]).write_text(''); print('ok'); raise SystemExit(0)\n"
        "if args[0]=='easy-cluster':\n"
        " ids=[line[1:].split()[0] for line in Path(args[1]).read_text().splitlines() if line.startswith('>')]\n"
        " Path(args[2]+'_cluster.tsv').write_text(''.join(f'{i}\\t{i}\\n' for i in ids))\n"
        " print('ok'); raise SystemExit(0)\n"
        "raise SystemExit(2)\n"
    )
    path.chmod(0o755)
    return path


def _inputs(tmp_path: Path, preflight: Path) -> dict[str, Path]:
    cath = tmp_path / "chain_set.jsonl"
    splits = tmp_path / "chain_set_splits.json"
    cath.write_text(json.dumps({"name": "CATH1", "seq": "G" * 120}) + "\n")
    splits.write_text(json.dumps({"train": ["CATH1"], "validation": [], "test": []}))
    split_paths = {}
    for role, values in (("train", ["SEEN1"]), ("val", []), ("test", [])):
        path = tmp_path / f"{role}_ids.txt"
        path.write_text("".join(f"{value}\n" for value in values))
        split_paths[role] = path
    proteins = tmp_path / "protein_samples.parquet"
    pd.DataFrame([{
        "protein_id": "SEEN1", "allele": ALLELE, "protein_seq": "H" * 120,
    }]).to_parquet(proteins, index=False)
    mmseqs = _mmseqs_tool(tmp_path / "mmseqs")
    nmp = tmp_path / "netMHCIIpan"
    nmp.write_text("#!/bin/sh\nexit 0\n")
    nmp.chmod(0o755)
    data_root = tmp_path / "nmp_data"
    data_root.mkdir()
    (data_root / "model.dat").write_text("fixture\n")
    install = tmp_path / "manifests" / "nmp_install_manifest.json"
    install.parent.mkdir()
    write_nmp_install_manifest(
        manifest_path=install, tool_path=nmp, data_root=data_root,
        tool_version="fixture-4.3", effective_environment={"fixture": "1"},
        identity_mode="fixture",
    )
    checkpoint = tmp_path / "epoch_24.pt"
    torch.save({"metadata": {
        "epoch": 24, "global_step": 10055, "config_hash": "ad6ac404027b",
        "manifest_version": "v1.1",
    }, "model_state_dict": {}}, checkpoint)
    resolved = tmp_path / "resolved_config.yaml"
    resolved.write_text("ablation_encoder_id: LC1\nmanifest_version: v1.1\n")
    summary = tmp_path / "run_summary.json"
    summary.write_text(json.dumps({
        "final_epoch": 24, "global_steps": 10055,
        "config_hash": "ad6ac404027b", "manifest_version": "v1.1",
    }))
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in ("model.yaml", "model_ablation.yaml", "inference.yaml"):
        (configs / name).write_text("fixture: true\n")
    descendants = tmp_path / "descendants.json"
    descendants.write_text(json.dumps({
        "schema_version": "if-benchmark-v3-descendant-bindings/1", "bindings": [],
    }))
    return {
        "tier1_preflight_manifest": preflight,
        "cath_chain_set": cath, "cath_splits": splits,
        "strict_train_ids": split_paths["train"], "strict_val_ids": split_paths["val"],
        "strict_test_ids": split_paths["test"], "protein_samples": proteins,
        "mmseqs_binary": mmseqs, "nmp_binary": nmp, "nmp_install_manifest": install,
        "head_checkpoint": checkpoint, "head_resolved_config": resolved,
        "head_run_summary": summary, "head_model_yaml": configs / "model.yaml",
        "head_ablation_yaml": configs / "model_ablation.yaml",
        "head_inference_yaml": configs / "inference.yaml",
        "descendant_scope": descendants,
    }


def _dag(tmp_path: Path, inputs: dict[str, Path]) -> dict[str, object]:
    candidate = tmp_path / "if_test_set" / "builds" / "fixture-build" / "HLA-DRB1_15_01"
    (candidate / "audit" / "orchestrator" / "stages").mkdir(parents=True)
    return {
        "identity_profile": "fixture", "build_id": "fixture-build",
        "candidate_root": str(candidate), "dag_sha256": "d" * 64,
        "targets": {"tier1": 1, "tier2": 1}, "c5_ladder": [1],
        "current_c5_rung": 1, "prior_rung_snapshots": [], "seed": 42,
        "stage_parameters": {
            "nmp": {"batch_size": 1, "subprocess_timeout_s": 60,
                    "max_lengths_per_call": 4, "n_workers": 1},
            "head": {"window_batch_size": 8, "device": "cpu"},
        },
        "immutable_inputs": {
            role: _identity(path) for role, path in inputs.items()
        },
        "stages": [{
            "name": spec.name, "dependencies": list(spec.dependencies),
            "execution": spec.execution, "task_count": 1,
            "resource_class": spec.resource_class,
            "resources": {"cpus": 1},
        } for spec in STAGES],
    }


def test_all_18_stage_producers_form_one_replayable_reduced_dag(tmp_path):
    preflight = _preflight(tmp_path)
    inputs = _inputs(tmp_path, preflight)
    dag = _dag(tmp_path, inputs)
    t2_sequence = "C" * 120
    # The C1 snapshot is the frozen revision authority for BOTH tiers, so it must also carry the
    # Tier 1 entry. `entities_all` is the pre-eligibility superset (production: 155,541 entities
    # of which 1,833 are rejected), so 1AAA enters at 3.0 A -- present for its revision, excluded
    # from the Tier 2 population.
    search_body = json.dumps({
        "total_count": 2,
        "result_set": [{"identifier": "1AAA_1"}, {"identifier": "2BBB_1"}],
    }).encode()
    graphql_body = json.dumps({"data": {"polymer_entities": [{
        "rcsb_id": "1AAA_1",
        "rcsb_polymer_entity_container_identifiers": {"entry_id": "1AAA", "entity_id": "1"},
        "entity_poly": {"pdbx_seq_one_letter_code_can": "A" * 120,
                        "rcsb_sample_sequence_length": 120,
                        "rcsb_entity_polymer_type": "Protein"},
        "entry": {
            "exptl": [{"method": "X-RAY DIFFRACTION"}],
            "rcsb_entry_info": {"resolution_combined": [3.0]},
            "rcsb_accession_info": {"initial_release_date": "2020-01-01T00:00:00Z",
                                     "revision_date": "2020-02-01T00:00:00Z"},
        },
        "polymer_entity_instances": [{
            "rcsb_id": "1AAA.A",
            "rcsb_polymer_entity_instance_container_identifiers": {
                "entry_id": "1AAA", "entity_id": "1", "asym_id": "A", "auth_asym_id": "A",
            },
        }],
    }, {
        "rcsb_id": "2BBB_1",
        "rcsb_polymer_entity_container_identifiers": {"entry_id": "2BBB", "entity_id": "1"},
        "entity_poly": {"pdbx_seq_one_letter_code_can": t2_sequence,
                        "rcsb_sample_sequence_length": 120,
                        "rcsb_entity_polymer_type": "Protein"},
        "entry": {
            "exptl": [{"method": "X-RAY DIFFRACTION"}],
            "rcsb_entry_info": {"resolution_combined": [2.0]},
            "rcsb_accession_info": {"initial_release_date": "2020-01-01T00:00:00Z",
                                     "revision_date": "2020-02-01T00:00:00Z"},
        },
        "polymer_entity_instances": [{
            "rcsb_id": "2BBB.A",
            "rcsb_polymer_entity_instance_container_identifiers": {
                "entry_id": "2BBB", "entity_id": "1", "asym_id": "A", "auth_asym_id": "A",
            },
        }],
    }]}}).encode()

    def rcsb_transport(_method, url, _body, _headers, _timeout):
        return SiftsHttpResponse(
            200, search_body if "search.rcsb.org" in url else graphql_body, {}
        )

    def cif_request(url, _timeout):
        pdb_id = Path(url).stem.upper()
        sequence = "A" * 120 if pdb_id == "1AAA" else t2_sequence
        return SiftsHttpResponse(200, _coordinate_cif(pdb_id, sequence), {})

    def nmp_factory(**kwargs):
        lengths = [int(value) for value in kwargs["peptide_lengths"]]
        key = kwargs["key_column"]
        sequence_col = kwargs["sequence_column"]

        def score(frame, _shard_dir):
            return pd.DataFrame([{
                "protein_id": str(row[key]), "peptide_length": length,
                "start_0b": start, "end_0b": start + length,
                "peptide": str(row[sequence_col])[start:start + length], "rank_el": 3.0,
            } for row in frame.to_dict("records") for length in lengths
              for start in range(len(str(row[sequence_col])) - length + 1)],
                columns=NMP_WINDOW_COLUMNS)
        return score

    class Predictor:
        def predict_proteins(self, records, **_kwargs):
            return [{
                "protein_id": protein_id,
                "prediction": {"global_risk": float(index)},
            } for index, (protein_id, _sequence) in enumerate(records)]

    adapters = {
        "rcsb_transport": rcsb_transport, "rcsb_cif_request": cif_request,
        "pdbe_residue_request": lambda _url, _timeout: SiftsHttpResponse(
            200, _enriched_sifts("A" * 120), {}
        ),
        "sleep_fn": lambda _seconds: None,
        "nmp_score_shard_factory": nmp_factory,
        "head_predictor_factory": lambda **_kwargs: Predictor(),
    }
    token = stages._ACTIVE_ADAPTERS.set(adapters)
    executed = []
    try:
        for spec in STAGES:
            output = (
                Path(dag["candidate_root"]) / "audit" / "orchestrator" / "stages" /
                spec.name / "00000" / "payload"
            )
            output.mkdir(parents=True)
            summary = stages.PRODUCERS[spec.name](dag, output, 0)
            stages._write_result(
                output, dag=dag, stage=spec.name, task_index=0, summary=summary,
            )
            executed.append(spec.name)
    finally:
        stages._ACTIVE_ADAPTERS.reset(token)
    assert executed == [spec.name for spec in STAGES]
    release = (
        Path(dag["candidate_root"]) / "audit" / "orchestrator" / "stages" /
        "release_validate" / "00000" / "payload"
    )
    assert json.loads((release / "release_gates.json").read_text()) == {
        str(index): "pass" for index in range(1, 14)
    }


def test_array_partition_leaves_no_empty_shard_and_covers_every_row():
    """4,453 C7 queries over 192 shards used to leave shards 186-191 empty, which blocked the
    stage seal even though every query had been scored."""
    import pandas as pd
    from inverse_folding.evaluation.if_benchmark_v3.stage_execution import _contiguous_partition

    for total, count in ((4453, 192), (52, 16), (59, 16), (5000, 64), (7, 7), (1, 1)):
        frame = pd.DataFrame({"i": range(total)})
        parts = [_contiguous_partition(frame, index=i, count=count) for i in range(count)]
        assert all(len(p) > 0 for p in parts), (total, count)
        assert sum(len(p) for p in parts) == total
        assert list(pd.concat(parts)["i"]) == list(range(total))   # contiguous, ordered, no gaps
        assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1
