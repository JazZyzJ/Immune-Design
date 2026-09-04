"""Tests for all-parent uricase structure terminal and energy orchestration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


terminal = _load(
    "build_uricase_structure_terminal_manifest",
    ROOT / "scripts/build_uricase_structure_terminal_manifest.py",
)
energy = _load(
    "run_uricase_energy_parent",
    ROOT / "scripts/run_uricase_energy_parent.py",
)
prepare = _load(
    "prepare_uricase_1501_evidence_inputs",
    ROOT / "scripts/prepare_uricase_1501_evidence_inputs.py",
)
repair = _load(
    "build_uricase_energy_repair_manifest",
    ROOT / "scripts/build_uricase_energy_repair_manifest.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contact_parent(
    root: Path,
    *,
    protein_id: str,
    sequence: str,
    row_index: int,
    selection: Path,
    relay_complete: bool,
    resolved: bool,
) -> None:
    out = root / protein_id
    out.mkdir(parents=True)
    name = f"{protein_id}_wt5"
    for filename in terminal._expected_contact_names(protein_id):
        (out / filename).write_text("x\n")
    pairs = pd.DataFrame(
        [
            {
                "sample_id": "sample_0",
                "chain_1": "A",
                "chain_pair": "A:B",
                "mechanistic_class": "catalytic" if resolved else "unresolved",
                "bsa_is_biological_interface": True,
                "mechanistic_is_biological_interface": resolved,
            },
            {
                "sample_id": "sample_0",
                "chain_1": "A",
                "chain_pair": "A:C",
                "mechanistic_class": "assembly" if resolved else "unresolved",
                "bsa_is_biological_interface": True,
                "mechanistic_is_biological_interface": resolved,
            },
        ]
    )
    pairs.to_parquet(out / f"{name}_contact_chain_pairs.parquet", index=False)
    source = root.parent / f"{protein_id}_sample0.cif"
    source.write_text("data_test\n")
    resolution = (
        "relay_unique_both_copies" if resolved else "unresolved_no_annotation"
    )
    metadata = {
        "schema_version": "tetramer_contact_consensus_v1",
        "protein_id": protein_id,
        "canonical_length": len(sequence),
        "canonical_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "mechanism": {
            "source": "relay_annotation" if relay_complete else "none",
            "status": "resolved" if resolved else "unresolved",
            "resolution_by_sample": {
                f"sample_{sample}": resolution for sample in range(5)
            },
        },
        "sources": {
            f"sample_{sample}": {"path": str(source), "sha256": _sha(source)}
            for sample in range(5)
        },
    }
    (out / f"{name}_contact_metadata.json").write_text(json.dumps(metadata))
    completion = {
        "status": "complete",
        "protein_id": protein_id,
        "row_index": row_index,
        "relay_candidate_complete": relay_complete,
        "pilot_manifest": str(selection.resolve()),
        "pilot_manifest_sha256": _sha(selection),
        "mapping_rows": 20 * len(sequence),
        "chain_pair_rows": 30,
        "consensus_rows": 3 * len(sequence),
        "runtime_sec": 2.0,
    }
    (out / "parent_completion.json").write_text(json.dumps(completion))


def test_terminal_builder_separates_resolved_and_unresolved(tmp_path: Path) -> None:
    selection = tmp_path / "selection.parquet"
    frame = pd.DataFrame(
        [
            {
                "frozen_order_6387": 0,
                "protein_id": "P1",
                "sequence": "ACDE",
                "sequence_length": 4,
                "prediction_name": "tetra_wt_P1",
                "relay_complete": True,
            },
            {
                "frozen_order_6387": 1,
                "protein_id": "P2",
                "sequence": "ACDF",
                "sequence_length": 4,
                "prediction_name": "tetra_wt_P2",
                "relay_complete": False,
            },
        ]
    )
    frame.to_parquet(selection, index=False)
    contacts = tmp_path / "contacts"
    contacts.mkdir()
    _contact_parent(
        contacts,
        protein_id="P1",
        sequence="ACDE",
        row_index=0,
        selection=selection,
        relay_complete=True,
        resolved=True,
    )
    _contact_parent(
        contacts,
        protein_id="P2",
        sequence="ACDF",
        row_index=1,
        selection=selection,
        relay_complete=False,
        resolved=False,
    )
    output = tmp_path / "terminal"
    terminal.run(
        argparse.Namespace(
            selection_manifest=selection,
            contact_root=contacts,
            output_dir=output,
            expected_parents=2,
        ),
        command=["test"],
    )

    ledger = pd.read_parquet(output / "structure_terminal_manifest.parquet")
    eligible = pd.read_parquet(output / "energy_eligible_manifest.parquet")
    assert ledger.structure_status.tolist() == ["qualified", "unresolved"]
    assert eligible.protein_id.tolist() == ["P1"]
    assert eligible.loc[0, "catalytic_pair"] == "A:B"
    assert eligible.loc[0, "assembly_pair"] == "A:C"
    assert json.loads((output / "manifest.json").read_text())["counts"] == {
        "parents": 2,
        "qualified": 1,
        "unresolved": 1,
        "unresolved_no_relay_annotation": 1,
        "unresolved_relay_geometry": 0,
    }


def test_energy_spec_and_output_validation(tmp_path: Path) -> None:
    structure = tmp_path / "sample0.cif"
    structure.write_text("data_test\n")
    contact_pairs = tmp_path / "pairs.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "sample_0",
                "chain_pair": "A:B",
                "mechanistic_class": "catalytic",
                "mechanistic_is_biological_interface": True,
            },
            {
                "sample_id": "sample_0",
                "chain_pair": "A:C",
                "mechanistic_class": "assembly",
                "mechanistic_is_biological_interface": True,
            },
        ]
    ).to_parquet(contact_pairs, index=False)
    manifest = pd.DataFrame(
        [
            {
                "energy_order": 0,
                "protein_id": "P1",
                "sequence_length": 4,
                "structure_status": "qualified",
                "energy_eligible": True,
                "representative_sample_id": "sample_0",
                "representative_structure_path": str(structure),
                "representative_structure_sha256": _sha(structure),
                "catalytic_pair": "A:B",
                "assembly_pair": "A:C",
                "contact_chain_pairs_path": str(contact_pairs),
            }
        ]
    )
    spec = energy.resolve_energy_spec(manifest, row_index=0)
    out = tmp_path / "work" / "P1"
    run = tmp_path / "run" / "P1"
    logs = tmp_path / "logs" / "P1"
    out.mkdir(parents=True)
    run.mkdir(parents=True)
    logs.mkdir(parents=True)
    for filename in energy._expected_output_names(spec):
        (out / filename).write_text("x\n")
    name = "P1_sample0_energy"
    table = pd.DataFrame(
        [
            {
                "interface_pair": pair,
                "res_id": index,
                "wt_residue_name3": "VAL",
                "ddg_bind_mean_reu": float(index),
                "is_interpretable_sidechain_alanine": True,
                "interface_class": f"interface_{index}",
                "rosetta_score_function": "ref2015",
            }
            for index, pair in enumerate(("A:B", "A:C"), start=1)
        ]
    )
    table.to_parquet(out / f"{name}_alanine_scan_by_position.parquet", index=False)
    runtime_artifact = run / "score.sc"
    log_artifact = logs / "score.log"
    runtime_artifact.write_text("x\n")
    log_artifact.write_text("x\n")
    interface_analyzer = tmp_path / "InterfaceAnalyzer"
    rosetta_scripts = tmp_path / "rosetta_scripts"
    interface_analyzer.write_text("x\n")
    rosetta_scripts.write_text("x\n")
    metadata = {
        "schema_version": "tetramer_reference_metrics_v2",
        "name": name,
        "source_structure": str(structure),
        "source_sha256": _sha(structure),
        "persistent_output_dir": str(out),
        "runtime_dir": str(run),
        "log_dir": str(logs),
        "parameters": {
            "alanine_scan_pairs": [["A", "B"], ["A", "C"]],
            "ddg_sign_convention": "mutant_minus_wildtype",
            "rosetta_score_function": "ref2015",
            "interface_residue_cutoff_a": 5.0,
        },
        "software": {
            "rosetta_interface_analyzer": str(interface_analyzer),
            "rosetta_scripts": str(rosetta_scripts),
        },
        "outputs": [str(out / filename) for filename in energy._expected_output_names(spec)],
        "runtime_artifacts": [str(runtime_artifact)],
        "log_artifacts": [str(log_artifact)],
        "row_counts": {"alanine_scan_by_position": 2},
    }
    (out / "metadata.json").write_text(json.dumps(metadata))

    result = energy.validate_energy_output(
        spec=spec,
        out_dir=out,
        run_dir=run,
        log_dir=logs,
        rosetta_interface_analyzer=interface_analyzer,
        rosetta_scripts=rosetta_scripts,
    )
    assert result["energy_rows"] == 2
    worker = ROOT / "scripts/submit_uricase_energy_array.slurm"
    syntax = subprocess.run(["bash", "-n", str(worker)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    assert "ROW_INDEX+=ARRAY_TASKS" in worker.read_text()


def test_prepare_1501_inputs_keeps_qualified_nonclean_parent(tmp_path: Path) -> None:
    selection = tmp_path / "selection.parquet"
    pd.DataFrame(
        [
            {
                "frozen_order_6387": 0,
                "protein_id": "P1",
                "sequence": "ACDEFGHIKLMN",
                "sequence_length": 12,
                "wt_nmp_clean": False,
            },
            {
                "frozen_order_6387": 1,
                "protein_id": "P2",
                "sequence": "MNPQRSTVWYAC",
                "sequence_length": 12,
                "wt_nmp_clean": True,
            },
        ]
    ).to_parquet(selection, index=False)
    terminal_path = tmp_path / "terminal.parquet"
    pd.DataFrame(
        [
            {"protein_id": protein_id, "structure_status": "qualified", "energy_eligible": True}
            for protein_id in ("P1", "P2")
        ]
    ).to_parquet(terminal_path, index=False)
    cores = tmp_path / "cores.parquet"
    pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "allele": "HLA-DRB1*15:01",
                "core_start_0b": 0,
                "core_start_1b": 1,
                "core_seq": "ACDEFGHIK",
                "best_rank_EL": 1.0,
                "n_supporting_windows": 1,
            }
        ]
    ).to_parquet(cores, index=False)
    prior = tmp_path / "prior.parquet"
    pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "kind": "hard",
                "ref_index_0b": 0,
                "label": "Lys11",
                "target_index_0b": 0,
                "target_aa": "A",
                "identity_match": True,
            }
        ]
    ).to_parquet(prior, index=False)
    annotations = tmp_path / "annotations.parquet"
    pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "target_index_0b": 0,
                "target_aa": "A",
                "functional_label": "Lys11",
                "mapping_status": "mapped_identity",
            }
        ]
    ).to_parquet(annotations, index=False)
    output = tmp_path / "prepared"

    prepare.run(
        argparse.Namespace(
            selection_manifest=selection,
            structure_terminal_manifest=terminal_path,
            cores=cores,
            q00511_projection_prior=prior,
            position_annotations=annotations,
            output_dir=output,
            expected_compute_parents=2,
            expected_qualified_parents=2,
            expected_redesign_parents=1,
            source_allele="HLA-DRB1*15:01",
            output_allele="HLA-DRB1_15_01",
        ),
        command=["test"],
    )
    cohort = pd.read_parquet(output / "cohort.parquet")
    baseline = pd.read_parquet(output / "open_policy_baseline_1501.parquet")
    assert cohort.protein_id.tolist() == ["P1"]
    assert set(baseline.allele) == {"HLA-DRB1_15_01"}
    assert len(baseline) == 4
    assert json.loads((output / "manifest.json").read_text())["counts"] == {
        "compute_parents": 2,
        "structure_qualified_parents": 2,
        "redesign_nonclean_parents": 1,
        "cores": 1,
        "homolog_analog_rows": 1,
    }


def test_energy_repair_manifest_reindexes_only_incomplete_rows(tmp_path: Path) -> None:
    source = tmp_path / "energy.parquet"
    pd.DataFrame(
        [
            {"energy_order": 0, "protein_id": "P1", "value": 1},
            {"energy_order": 1, "protein_id": "P2", "value": 2},
        ]
    ).to_parquet(source, index=False)
    energy_root = tmp_path / "energy"
    (energy_root / "P1").mkdir(parents=True)
    (energy_root / "P2").mkdir()
    (energy_root / "P1" / "parent_completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protein_id": "P1",
                "row_index": 0,
                "energy_manifest": str(source.resolve()),
                "energy_manifest_sha256": _sha(source),
            }
        )
    )
    output = tmp_path / "repair"

    repair.run(
        argparse.Namespace(
            energy_manifest=source,
            energy_output_root=energy_root,
            output_dir=output,
            expected_parents=2,
            expected_missing=1,
        ),
        command=["test"],
    )
    frame = pd.read_parquet(output / "energy_repair_manifest.parquet")
    assert frame[["energy_order", "source_energy_order", "protein_id"]].to_dict(
        "records"
    ) == [{"energy_order": 0, "source_energy_order": 1, "protein_id": "P2"}]
