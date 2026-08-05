"""Contract tests for the Active-15 WT-core release bundle builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from inverse_folding.reference_flow.constraints import load_constraint_manifest
from scripts.build_uricase_active15_core_release import ContractError, main


AA3 = {
    "A": "ALA",
    "C": "CYS",
}
REGISTERS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9")


def _write_pdb(path: Path, sequence: str) -> None:
    lines: list[str] = []
    serial = 1
    for residue_index, aa in enumerate(sequence, start=1):
        residue_name = AA3[aa]
        for atom_offset, atom_name in enumerate(("N", "CA", "C", "O")):
            x = float(residue_index * 4 + atom_offset)
            lines.append(
                f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} A"
                f"{residue_index:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
                "  1.00 90.00           "
                f"{atom_name[0]:>2s}\n"
            )
            serial += 1
    lines.extend(("TER\n", "END\n"))
    path.write_text("".join(lines))


def _core_rows(
    *,
    protein_id: str,
    allele: str,
    sequence: str,
    core_start_0b: int,
    blockers: dict[str, str],
    analog_registers: set[str] | None = None,
) -> list[dict[str, object]]:
    analog_registers = analog_registers or set()
    rows: list[dict[str, object]] = []
    for offset, register in enumerate(REGISTERS):
        blocker = blockers.get(register, "")
        energy_status = (
            "scanned_noninterpretable_AGP"
            if blocker == "agp"
            else "measured_interpretable"
        )
        rows.append(
            {
                "protein_id": protein_id,
                "allele": allele,
                "core_start_0b": core_start_0b,
                "core_start_1b": core_start_0b + 1,
                "core_seq": sequence[core_start_0b : core_start_0b + 9],
                "best_rank_EL": 0.25,
                "n_supporting_windows": 2,
                "core_offset_0b": offset,
                "core_register": register,
                "index_0b": core_start_0b + offset,
                "in_policy_anchors_P1P4P6P9": register in {"P1", "P4", "P6", "P9"},
                "sequence_length": len(sequence),
                "position_1b": core_start_0b + offset + 1,
                "wt_aa": sequence[core_start_0b + offset],
                "in_C80_stable": blocker == "c80",
                "in_sigma80_robust": blocker in {"sigma", "sigma_exploratory"},
                "sigma_hard_lock_status": (
                    "exploratory_not_hard_lock"
                    if blocker == "sigma_exploratory"
                    else "eligible"
                ),
                "in_full_contact_5of5": blocker == "full5",
                "energy_position_status": energy_status,
                "in_energy_ddg_gt0_reu": (
                    pd.NA if blocker == "agp" else blocker == "ddg"
                ),
                # ddG tiers are nested: ge1/ge2 are subsets of gt0. The synthetic
                # "ddg" blocker is a genuine hotspot, so it sits in every tier.
                "in_energy_ddg_ge1_reu": (
                    pd.NA if blocker == "agp" else blocker == "ddg"
                ),
                "in_energy_ddg_ge2_reu": (
                    pd.NA if blocker == "agp" else blocker == "ddg"
                ),
                # Annotation by default; a hard veto only under --lock-functional-analogs.
                "homolog_analog_overlap": register in analog_registers,
                "legacy_identity_required_overlap": False,
            }
        )
    return rows


def _make_inputs(tmp_path: Path, *, structure_sequence: str | None = None) -> dict[str, Path]:
    protein_id = "PARENT1"
    sequence = "A" * 60
    allele_open = "HLA-DRB1_04_01"
    allele_blocked = "HLA-DRB1_07_01"

    evidence_dir = tmp_path / "evidence"
    pdb_root = tmp_path / "pdbs"
    evidence_dir.mkdir()
    pdb_root.mkdir()

    fasta = tmp_path / "parents.fasta"
    fasta.write_text(f">{protein_id}\n{sequence}\n")
    _write_pdb(pdb_root / f"{protein_id}.pdb", structure_sequence or sequence)

    positions = pd.DataFrame(
        {
            "protein_id": protein_id,
            "sequence_length": len(sequence),
            "index_0b": range(len(sequence)),
            "position_1b": range(1, len(sequence) + 1),
            "wt_aa": list(sequence),
            "homolog_analog_overlap": False,
            "legacy_identity_required_overlap": False,
            "legacy_identity_required_labels_json": "[]",
            "homolog_analog_annotations_json": "[]",
        }
    )
    parent_evidence = evidence_dir / "parent_position_evidence.parquet"
    positions.to_parquet(parent_evidence, index=False)

    core_rows = _core_rows(
        protein_id=protein_id,
        allele=allele_open,
        sequence=sequence,
        core_start_0b=5,
        blockers={"P4": "c80", "P6": "ddg"},
        analog_registers={"P1"},
    )
    # A second openable core overlaps the first at index 13. Every safe anchor
    # from both cores must enter the union; exploratory sigma remains editable.
    core_rows += _core_rows(
        protein_id=protein_id,
        allele=allele_open,
        sequence=sequence,
        core_start_0b=13,
        blockers={"P4": "sigma_exploratory", "P6": "full5"},
    )
    core_rows += _core_rows(
        protein_id=protein_id,
        allele=allele_blocked,
        sequence=sequence,
        core_start_0b=30,
        blockers={"P1": "c80", "P4": "sigma", "P6": "full5", "P9": "agp"},
    )
    core_evidence = evidence_dir / "epitope_core_position_evidence.parquet"
    pd.DataFrame(core_rows).to_parquet(core_evidence, index=False)
    (evidence_dir / "join_metadata.json").write_text(
        json.dumps(
            {
                "cohort": {
                    "parent_count": 1,
                    "parent_ids": [protein_id],
                    "alleles": [allele_open, allele_blocked],
                    "canonical_position_count": len(sequence),
                    "core_count": 3,
                    "core_position_count": 27,
                },
                "outputs": {
                    parent_evidence.name: {
                        "sha256": hashlib.sha256(parent_evidence.read_bytes()).hexdigest()
                    },
                    core_evidence.name: {
                        "sha256": hashlib.sha256(core_evidence.read_bytes()).hexdigest()
                    },
                },
            }
        )
    )

    if_ready = tmp_path / "if_ready.parquet"
    pd.DataFrame(
        [
            {
                "protein_id": protein_id,
                "sequence": sequence,
                "sequence_length": len(sequence),
                "pdb_path": f"{protein_id}.pdb",
                "if_chain_id": "A",
                "if_ready": True,
            }
        ]
    ).to_parquet(if_ready, index=False)

    sampler_config = tmp_path / "c1_null.yaml"
    sampler_config.write_text(
        "sampler:\n"
        "  n_steps: 100\n"
        "  seed: 42\n"
        "  temperature: 1.0\n"
        "  n_designs_per_protein: 1\n"
        "  remask:\n"
        "    enabled: true\n"
        "    fraction_scale: 1.0\n"
        "schedule:\n"
        "  base_form: linear\n"
        "amplification:\n"
        "  form: constant_one\n"
        "  h_source: h_processed\n"
        "  g_max_cap: 20.0\n"
        "  mu: 0.0\n"
        "  c: null\n"
        "  kappa: null\n"
        "  p: null\n"
        "h_shuffle:\n"
        "  enabled: false\n"
        "  seed: null\n"
    )
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"checkpoint-fixture")
    return {
        "evidence_dir": evidence_dir,
        "fasta": fasta,
        "if_ready": if_ready,
        "pdb_root": pdb_root,
        "sampler_config": sampler_config,
        "checkpoint": checkpoint,
    }


def _rewrite_core_evidence(
    inputs: dict[str, Path], core: pd.DataFrame
) -> None:
    core_path = inputs["evidence_dir"] / "epitope_core_position_evidence.parquet"
    core.to_parquet(core_path, index=False)
    metadata_path = inputs["evidence_dir"] / "join_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    cohort = metadata["cohort"]
    cohort["alleles"] = sorted(set(core["allele"].astype(str)))
    cohort["core_count"] = len(
        core[["protein_id", "allele", "core_start_0b"]].drop_duplicates()
    )
    cohort["core_position_count"] = len(core)
    metadata["outputs"][core_path.name]["sha256"] = hashlib.sha256(
        core_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata))


def _argv(inputs: dict[str, Path], output_dir: Path) -> list[str]:
    return [
        "--evidence-dir",
        str(inputs["evidence_dir"]),
        "--parent-fasta",
        str(inputs["fasta"]),
        "--if-ready-parquet",
        str(inputs["if_ready"]),
        "--pdb-root",
        str(inputs["pdb_root"]),
        "--sampler-config",
        str(inputs["sampler_config"]),
        "--expected-sampler-config-sha256",
        hashlib.sha256(inputs["sampler_config"].read_bytes()).hexdigest(),
        "--checkpoint",
        str(inputs["checkpoint"]),
        "--expected-checkpoint-sha256",
        hashlib.sha256(inputs["checkpoint"].read_bytes()).hexdigest(),
        "--run-output-root",
        str(output_dir.parent / "rf_runs"),
        "--output-dir",
        str(output_dir),
        "--expected-parent-count",
        "1",
        "--expected-alleles",
        "HLA-DRB1_04_01",
        "HLA-DRB1_07_01",
        "--expected-cell-count",
        "2",
        "--expected-core-count",
        "3",
        "--expected-openable-core-count",
        "2",
        "--expected-emitted-cell-count",
        "1",
        "--expected-represented-parent-count",
        "1",
    ]


def test_builds_all_safe_anchor_complement_bundle_and_round_trips(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    output_dir = tmp_path / "bundle"

    assert main(_argv(inputs, output_dir)) == 0

    manifests = list((output_dir / "constraint_manifests").glob("*.yaml"))
    cohorts = list((output_dir / "cohorts" / "by_parent").glob("*.parquet"))
    assert len(manifests) == 1
    assert len(cohorts) == 1
    assert len(pd.read_parquet(cohorts[0])) == 1

    anchor_ledger = pd.read_parquet(
        output_dir / "decision" / "core_anchor_decisions.parquet"
    )
    assert len(anchor_ledger) == 12
    open_core = anchor_ledger[anchor_ledger["allele"].eq("HLA-DRB1_04_01")]
    safe = open_core[open_core["safe_anchor"]]
    assert set(safe["core_register"]) == {"P1", "P4", "P9"}
    assert safe["included_in_free_set"].all()
    assert bool(
        open_core.loc[
            open_core.core_register.eq("P1"), "homolog_analog_overlap"
        ].iloc[0]
    )
    assert bool(open_core.loc[open_core.core_register.eq("P1"), "safe_anchor"].iloc[0])
    exploratory = open_core[
        open_core.core_start_0b.eq(13) & open_core.core_register.eq("P4")
    ].iloc[0]
    assert exploratory.blocked_reasons_json == "[]"
    assert bool(exploratory.safe_anchor)
    assert bool(exploratory.included_in_free_set)

    manifest = load_constraint_manifest(manifests[0])
    constraint = manifest.constraint_for_protein("PARENT1")
    constraint.validate_against_sequence("A" * 60)
    fixed = set(constraint.hard_anchor_indices)
    free = {5, 13, 16, 21}
    assert fixed == set(range(60)) - free
    assert len(fixed) == 56
    raw_manifest = yaml.safe_load(manifests[0].read_text())
    assert raw_manifest["entries"][0]["sequence_md5"] == hashlib.md5(
        ("A" * 60).encode("ascii")
    ).hexdigest()

    cells = pd.read_parquet(
        output_dir / "decision" / "parent_allele_core_release_candidates.parquet"
    )
    assert len(cells) == 2
    assert cells["emitted"].sum() == 1
    emitted = cells[cells["emitted"]].iloc[0]
    assert emitted.n_openable_cores == 2
    assert emitted.n_free_positions == 4
    assert emitted.fixed_fraction == pytest.approx(56 / 60)

    launch = pd.read_csv(
        output_dir / "launch_manifest.tsv", sep="\t", keep_default_na=False
    )
    assert len(launch) == 1
    assert launch.iloc[0].runtime_allele == "HLA-DRB1*04:01"
    assert launch.iloc[0].allele_tag == "HLA-DRB1_04_01"
    assert launch.iloc[0].h_maps_parquet == ""
    assert launch.iloc[0].controller_config == ""
    assert Path(launch.iloc[0].sampler_config).is_absolute()
    assert Path(launch.iloc[0].checkpoint).is_absolute()
    assert len(launch.iloc[0].sampler_config_sha256) == 64
    assert launch.iloc[0].sampler_config_resolved_sha256 == (
        "f2cd1204d6eaafd0eb46a9b13bb4687e392100434de22aa93936f135cb3fa343"
    )
    assert len(launch.iloc[0].checkpoint_sha256) == 64
    assert launch.iloc[0]["mode"] == "reference_flow"
    assert launch.iloc[0].seed == 42
    assert launch.iloc[0].n_steps == 100
    assert launch.iloc[0].n_designs_per_protein == 1
    assert launch.iloc[0].rf_batch_size == 1
    assert Path(launch.iloc[0].run_output_dir).is_absolute()

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["counts"] == {
        "parents": 1,
        "cells": 2,
        "cores": 3,
        "openable_cores": 2,
        "emitted_cells": 1,
        "represented_parents": 1,
        "free_gate_collisions": 0,
    }
    assert metadata["claim_boundary"]["editable_anchor_is_not_resolved"] is True


def test_functional_analog_lock_and_energy_tier_change_the_free_set(
    tmp_path: Path,
) -> None:
    """--lock-functional-analogs promotes the analog annotation to a hard veto.

    The open core's P1 carries the analog annotation and is otherwise safe, so the flag
    must remove it from the free set, record it as a labelled functional lock, and leave
    the core openable via its remaining safe anchors. --energy-ddg-tier is echoed into the
    frozen gate policy so a manifest always states which hotspot tier produced it.
    """
    inputs = _make_inputs(tmp_path)
    output_dir = tmp_path / "bundle"
    argv = _argv(inputs, output_dir) + [
        "--lock-functional-analogs",
        "--energy-ddg-tier",
        "ge1",
        "--n-designs-per-protein",
        "150",
    ]
    assert main(argv) == 0

    ledger = pd.read_parquet(output_dir / "decision" / "core_anchor_decisions.parquet")
    open_core = ledger[ledger["allele"].eq("HLA-DRB1_04_01")]
    p1 = open_core[open_core.core_register.eq("P1") & open_core.core_start_0b.eq(5)].iloc[0]
    assert bool(p1.blocked)
    assert "functional_analog_overlap" in json.loads(p1.blocked_reasons_json)
    assert not bool(p1.included_in_free_set)
    # The core survives on its other safe anchors: locking analogs must not drop cores.
    assert bool(p1.core_openable)

    manifest_path = next((output_dir / "constraint_manifests").glob("*.yaml"))
    raw = yaml.safe_load(manifest_path.read_text())
    provenance = raw["annotation_provenance"]
    assert provenance["functional_analog_hard_lock"] is True
    assert provenance["gate_policy"] == (
        "C80_or_eligible_sigma80_or_full5_or_functional_analog"
        "_or_ddg_ge1_reu_or_scanned_AGP"
    )
    assert "numbering_contract" in provenance
    assert 5 not in raw["entries"][0]["free_positions_0b"]
    # Index 0 is never editable: an unlocked N-terminal Met gets rewritten by RF.
    assert 0 not in raw["entries"][0]["free_positions_0b"]

    launch = pd.read_csv(
        output_dir / "launch_manifest.tsv", sep="\t", keep_default_na=False
    )
    assert launch.iloc[0].n_designs_per_protein == 150
    # The frozen sampler config itself is never mutated by the override.
    assert launch.iloc[0].sampler_config_resolved_sha256 == (
        "f2cd1204d6eaafd0eb46a9b13bb4687e392100434de22aa93936f135cb3fa343"
    )


def test_structure_sequence_mismatch_fails_without_publishing(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path, structure_sequence="C" + "A" * 59)
    output_dir = tmp_path / "bundle"

    with pytest.raises(ContractError, match="structure sequence"):
        main(_argv(inputs, output_dir))

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".bundle.tmp.*"))


def test_existing_output_is_refused_without_mutation(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("do-not-replace")

    with pytest.raises(ContractError, match="already exists"):
        main(_argv(inputs, output_dir))

    assert sentinel.read_text() == "do-not-replace"


def test_fixed_fraction_gate_fails_before_publish(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    output_dir = tmp_path / "bundle"
    argv = _argv(inputs, output_dir) + ["--min-fixed-fraction", "0.95"]

    with pytest.raises(ContractError, match="emitted cell count"):
        main(argv)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".bundle.tmp.*"))


def test_c1_null_semantic_drift_fails_before_publish(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    inputs["sampler_config"].write_text(
        inputs["sampler_config"].read_text().replace("n_steps: 100", "n_steps: 99")
    )
    output_dir = tmp_path / "bundle"

    with pytest.raises(ContractError, match="c1_null"):
        main(_argv(inputs, output_dir))

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".bundle.tmp.*"))


def test_source_table_hash_drift_fails_before_publish(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    parent_evidence = inputs["evidence_dir"] / "parent_position_evidence.parquet"
    parent = pd.read_parquet(parent_evidence)
    parent.loc[0, "wt_aa"] = "C"
    parent.to_parquet(parent_evidence, index=False)
    output_dir = tmp_path / "bundle"

    with pytest.raises(ContractError, match="source table SHA256 drift"):
        main(_argv(inputs, output_dir))

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".bundle.tmp.*"))


def test_physical_position_gate_inconsistency_fails_before_publish(
    tmp_path: Path,
) -> None:
    inputs = _make_inputs(tmp_path)
    core_path = inputs["evidence_dir"] / "epitope_core_position_evidence.parquet"
    core = pd.read_parquet(core_path)
    target = (
        core["allele"].eq("HLA-DRB1_04_01")
        & core["core_start_0b"].eq(13)
        & core["index_0b"].eq(13)
    )
    assert int(target.sum()) == 1
    core.loc[target, "in_C80_stable"] = True
    _rewrite_core_evidence(inputs, core)
    output_dir = tmp_path / "bundle"

    with pytest.raises(
        ContractError,
        match=r"physical-position evidence inconsistency.*in_C80_stable",
    ):
        main(_argv(inputs, output_dir))

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".bundle.tmp.*"))


def test_parent_allele_cell_topology_mismatch_fails_before_publish(
    tmp_path: Path,
) -> None:
    inputs = _make_inputs(tmp_path)
    core_path = inputs["evidence_dir"] / "epitope_core_position_evidence.parquet"
    core = pd.read_parquet(core_path)
    core.loc[
        core["allele"].eq("HLA-DRB1_07_01"), "allele"
    ] = "HLA-DRB1_15_01"
    _rewrite_core_evidence(inputs, core)
    output_dir = tmp_path / "bundle"

    with pytest.raises(ContractError, match="cell topology mismatch"):
        main(_argv(inputs, output_dir))

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".bundle.tmp.*"))
