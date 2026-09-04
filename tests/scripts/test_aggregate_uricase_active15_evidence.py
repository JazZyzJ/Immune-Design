"""Contract tests for the Active-15 epitope/evolution/structure evidence join."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.aggregate_uricase_active15_evidence import JoinContractError, main
from scripts.merge_uricase_evidence_join_shards import merge_shards


ALLELES = ("HLA-DRB1_04_01", "HLA-DRB1_07_01", "HLA-DRB1_15_01")
POLICIES = (
    "all_core_residues",
    "anchors_P1P4P6P9",
    "P1_plus_P4",
    "P1_only",
)
MASKS = (
    "C90_stable",
    "C80_stable",
    "C70_stable",
    "sigma90_L",
    "sigma90_robust",
    "sigma80_L",
    "sigma80_robust",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_evolution(parent_dir: Path, protein_id: str, sequence: str) -> None:
    parent_dir.mkdir(parents=True)
    length = len(sequence)
    frame = pd.DataFrame(
        {
            "protein_id": [protein_id] * length,
            "covariance_status": ["qualified"] * length,
            "index_0b": range(length),
            "position_1b": range(1, length + 1),
            "wt_aa": list(sequence),
            "C_nogap_min": [0.95, 0.85, 0.75] + [0.5] * (length - 3),
            "pWT_nogap_min": [0.95, 0.85, 0.75] + [0.5] * (length - 3),
            "gap_frac_max": [0.0] * length,
            "sigma_L_pct": [0.1, 0.2, 0.3, 0.95, 0.85] + [0.1] * (length - 5),
            "sigma_robust_pct": [0.1] * 5 + [0.95, 0.85] + [0.1] * (length - 7),
        }
    )
    positions = {
        "C90_stable": [0],
        "C80_stable": [0, 1],
        "C70_stable": [0, 1, 2],
        "sigma90_L": [3],
        "sigma90_robust": [5],
        "sigma80_L": [3, 4],
        "sigma80_robust": [5, 6],
    }
    mask_payload = {
        "schema_version": 1,
        "protein_id": protein_id,
        "covariance_status": "qualified",
        "masks": {},
    }
    for mask in MASKS:
        frame[f"in_{mask}"] = frame["index_0b"].isin(positions[mask])
        is_sigma = mask.startswith("sigma")
        mask_payload["masks"][mask] = {
            "kind": "covariance" if is_sigma else "conservation",
            "status": "qualified" if is_sigma else "available",
            "metric": mask,
            "threshold": 0.9 if "90" in mask else 0.8 if "80" in mask else 0.7,
            "gap_cap": 0.5 if is_sigma else 0.2,
            "n_positions": len(positions[mask]),
            "positions_index_0b": positions[mask],
            "positions_position_1b": [index + 1 for index in positions[mask]],
            "position_labels": [f"{sequence[index]}{index + 1}" for index in positions[mask]],
        }
    frame.to_csv(parent_dir / "per_position_evolution.tsv", sep="\t", index=False)
    (parent_dir / "wt_lock_masks.json").write_text(
        json.dumps(mask_payload, indent=2) + "\n"
    )
    metadata = {
        "schema_version": 1,
        "protein_id": protein_id,
        "sequence": {
            "length": length,
            "md5": hashlib.md5(sequence.encode()).hexdigest(),
        },
        "model": {"neff_per_length": 12.0, "covariance_status": "qualified"},
        "contact_gate": {"status": "pass"},
    }
    (parent_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    pd.DataFrame(
        [
            {
                "protein_id": protein_id,
                "covariance_status": "qualified",
                "top_set": top_set,
                "pair_fraction_of_L": fraction,
                "requested_k": int(fraction * length),
                "common_resolved_positions_n": length,
                "monomer_hits": monomer_hits,
                "monomer_precision": monomer_precision,
                "tetramer_hits": tetramer_hits,
                "tetramer_precision": tetramer_precision,
                "tetramer_minus_monomer_precision": tetramer_precision - monomer_precision,
                "tetramer_only_hits": tetramer_hits - monomer_hits,
            }
            for top_set, fraction, monomer_hits, monomer_precision, tetramer_hits, tetramer_precision in (
                ("L/2", 0.5, 2, 2 / 6, 3, 3 / 6),
                ("L", 1.0, 6, 0.5, 9, 0.75),
                ("2L", 2.0, 12, 0.5, 18, 0.75),
            )
        ]
    ).to_csv(
        parent_dir / "ec_contact_monomer_tetramer_comparison.tsv", sep="\t", index=False
    )
    ec_rows = []
    long_range = [
        (left, right)
        for left in range(length)
        for right in range(left + 1, length)
        if right - left >= 6
    ]
    long_rank = {pair: rank for rank, pair in enumerate(long_range)}
    for left in range(length):
        for right in range(left + 1, length):
            pair = (left, right)
            ec_rows.append(
                {
                    "protein_id": protein_id,
                    "i": left + 1,
                    "A_i": sequence[left],
                    "j": right + 1,
                    "A_j": sequence[right],
                    "cn": (
                        100.0 - long_rank[pair]
                        if pair in long_rank
                        else -float(right - left)
                    ),
                }
            )
    pd.DataFrame(ec_rows).to_csv(
        parent_dir / "complete_ec_table.tsv", sep="\t", index=False
    )


def _write_structure(
    root: Path,
    protein_id: str,
    sequence: str,
    *,
    resolved: bool,
) -> tuple[Path, Path, Path, Path, Path]:
    root.mkdir(parents=True)
    rows = []
    if resolved:
        classes = (("catalytic", "catalytic"), ("assembly", "assembly"), ("diagonal", "diagonal"))
    else:
        classes = (("interface_1", "unresolved"), ("interface_2", "unresolved"), ("diagonal", "unresolved"))
    for class_name, mechanism in classes:
        for index, aa in enumerate(sequence):
            n_both = 0
            if resolved and class_name == "catalytic" and index == 0:
                n_both = 5
            elif resolved and class_name == "catalytic" and index == 1:
                n_both = 4
            elif resolved and class_name == "assembly" and index == 2:
                n_both = 5
            rows.append(
                {
                    "protein_id": protein_id,
                    "name": f"{protein_id}_wt",
                    "consensus_class": class_name,
                    "mechanistic_class": mechanism,
                    "bsa_rank_classes_seen": class_name,
                    "index_0b": index,
                    "position_1b": index + 1,
                    "aa": aa,
                    "n_samples": 5,
                    "n_symmetry_copy_opportunities": 10,
                    "n_copy_contacts": 2 * n_both,
                    "copy_contact_frequency": n_both / 5,
                    "n_samples_any_copy": n_both,
                    "sample_any_copy_frequency": n_both / 5,
                    "n_samples_both_copies": n_both,
                    "sample_both_copies_frequency": n_both / 5,
                    "mask_all_N": n_both == 5,
                    "mask_5ofN": n_both >= 5,
                    "mask_4ofN": n_both >= 4,
                    "mask_1ofN": n_both >= 1,
                }
            )
    consensus = pd.DataFrame(rows)
    consensus_path = root / f"{protein_id}_wt_contact_consensus.parquet"
    consensus.to_parquet(consensus_path, index=False)
    classes_json = {}
    for class_name, mechanism in classes:
        group = consensus[consensus["consensus_class"] == class_name]
        classes_json[class_name] = {
            "mechanistic_class": mechanism,
            "bsa_rank_classes_seen": [class_name],
            "positions_index_0b": {
                "all_N": group.loc[group.mask_all_N, "index_0b"].tolist(),
                "5ofN": group.loc[group.mask_5ofN, "index_0b"].tolist(),
                "4ofN": group.loc[group.mask_4ofN, "index_0b"].tolist(),
                "1ofN": group.loc[group.mask_1ofN, "index_0b"].tolist(),
            },
        }
    masks_path = root / f"{protein_id}_wt_contact_masks.json"
    masks_path.write_text(
        json.dumps(
            {
                "schema_version": "tetramer_contact_masks_v1",
                "protein_id": protein_id,
                "n_samples": 5,
                "mechanism_status": "resolved" if resolved else "unresolved",
                "mask_semantics": {
                    "symmetry_copy_requirement": "both copies",
                    "literal_thresholds": True,
                },
                "classes": classes_json,
            },
            indent=2,
        )
        + "\n"
    )
    metadata_path = root / f"{protein_id}_wt_contact_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "tetramer_contact_consensus_v1",
                "protein_id": protein_id,
                "canonical_length": len(sequence),
                "canonical_sequence_md5": hashlib.md5(sequence.encode()).hexdigest(),
                "canonical_sequence_sha256": _sha256_text(sequence),
                "mechanism": {"status": "resolved" if resolved else "unresolved"},
                "sources": {
                    f"s{sample_index}": {
                        "sha256": _sha256_text(
                            f"{protein_id}:s{sample_index}"
                        )
                    }
                    for sample_index in range(5)
                },
                "parameters": {
                    "mask_symmetry_copy_requirement": "both copies",
                    "mask_threshold_counts": [5, 4, 1],
                    "residue_pair_contact_cutoff_a": 5.0,
                },
            },
            indent=2,
        )
        + "\n"
    )
    long_range = [
        (left, right)
        for left in range(len(sequence))
        for right in range(left + 1, len(sequence))
        if right - left >= 6
    ]
    residue_pair_rows = []
    for sample_index in range(5):
        sample_id = f"s{sample_index}"
        source_sha256 = _sha256_text(f"{protein_id}:{sample_id}")
        within_pairs = list(long_range[:7])
        if sample_index < 4:
            within_pairs.append(long_range[10])
        if sample_index == 0:
            within_pairs.append(long_range[11])
        for left, right in within_pairs:
            residue_pair_rows.append(
                {
                    "protein_id": protein_id,
                    "sample_id": sample_id,
                    "source_sha256": source_sha256,
                    "index_1_0b": left,
                    "index_2_0b": right,
                    "canonical_pair_min_0b": left,
                    "canonical_pair_max_0b": right,
                    "is_same_canonical_index": False,
                    "is_inter_chain": False,
                    "mechanistic_class": "within_chain",
                    "interface_copy": 0,
                    "min_heavy_atom_distance_a": 4.0,
                    "contact_kind": "within_chain",
                }
            )
        if resolved:
            inter_pairs = (
                (*long_range[7], "catalytic"),
                (*long_range[8], "assembly"),
                (*long_range[9], "catalytic"),
                (*long_range[12], "diagonal"),
            )
            chain_pairs_by_mechanism = {
                "catalytic": (("A", "B", 1), ("C", "D", 2)),
                "assembly": (("A", "C", 1), ("B", "D", 2)),
                "diagonal": (("A", "D", 1), ("B", "C", 2)),
            }
            for left, right, mechanism in inter_pairs:
                for chain_1, chain_2, interface_copy in chain_pairs_by_mechanism[
                    mechanism
                ]:
                    residue_pair_rows.append(
                        {
                            "protein_id": protein_id,
                            "sample_id": sample_id,
                            "source_sha256": source_sha256,
                            "chain_1": chain_1,
                            "chain_2": chain_2,
                            "index_1_0b": left,
                            "index_2_0b": right,
                            "canonical_pair_min_0b": left,
                            "canonical_pair_max_0b": right,
                            "is_same_canonical_index": False,
                            "is_inter_chain": True,
                            "mechanistic_class": mechanism,
                            "interface_copy": interface_copy,
                            "chain_pair": f"{chain_1}:{chain_2}",
                            "bsa_rank_class": {
                                "catalytic": "interface_1",
                                "assembly": "interface_2",
                                "diagonal": "diagonal",
                            }[mechanism],
                            "min_heavy_atom_distance_a": (
                                5.0 if mechanism == "assembly" else 4.0
                            ),
                            "contact_kind": "tetramer_only",
                        }
                    )
    residue_pairs_path = root / f"{protein_id}_wt_residue_pair_contacts.parquet"
    pd.DataFrame(residue_pair_rows).to_parquet(residue_pairs_path, index=False)
    aa3_by_aa1 = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "Y": "TYR",
    }
    mapping_rows = []
    for sample_index in range(5):
        sample_id = f"s{sample_index}"
        source_sha256 = _sha256_text(f"{protein_id}:{sample_id}")
        for chain in ("A", "B", "C", "D"):
            for index_0b, aa in enumerate(sequence):
                mapping_rows.append(
                    {
                        "protein_id": protein_id,
                        "sample_id": sample_id,
                        "source_sha256": source_sha256,
                        "chain": chain,
                        "chain_position_0b": index_0b,
                        "res_id": index_0b + 101,
                        "ins_code": "",
                        "observed_residue_name3": aa3_by_aa1[aa],
                        "observed_aa": aa,
                        "index_0b": index_0b,
                        "position_1b": index_0b + 1,
                        "expected_aa": aa,
                    }
                )
    residue_mapping_path = root / f"{protein_id}_wt_residue_mapping.parquet"
    pd.DataFrame(mapping_rows).to_parquet(residue_mapping_path, index=False)
    return (
        consensus_path,
        masks_path,
        metadata_path,
        residue_pairs_path,
        residue_mapping_path,
    )


def _write_energy(
    root: Path,
    protein_id: str,
    sequence: str,
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    rows = []
    aa3 = {"A": "ALA", "C": "CYS", "D": "ASP"}
    definitions = (
        ("A:B", "interface_1", 1, 0.0),
        ("A:B", "interface_1", 2, 2.5),
        ("A:C", "interface_2", 2, 0.5),
        ("A:C", "interface_2", 3, -0.5),
    )
    for interface_pair, interface_class, position_1b, ddg in definitions:
        residue_name = aa3[sequence[position_1b - 1]]
        interpretable = residue_name not in {"ALA", "GLY", "PRO"}
        rows.append(
            {
                "interface_pair": interface_pair,
                "res_id": position_1b + 100,
                "ins_code": "",
                "wt_residue_name3": residue_name,
                "source_chains": interface_pair.replace(":", ","),
                "n_chain_sides": 2,
                "ddg_bind_mean_reu": ddg,
                "ddg_bind_min_reu": ddg - 0.1,
                "ddg_bind_max_reu": ddg + 0.1,
                "is_native_alanine": residue_name == "ALA",
                "is_glycine_to_alanine": residue_name == "GLY",
                "is_proline_to_alanine": residue_name == "PRO",
                "is_interpretable_sidechain_alanine": interpretable,
                "ddg_bind_chain_range_reu": 0.2,
                "min_cross_chain_heavy_atom_distance_a": 4.0,
                "interface_class": interface_class,
                "rosetta_score_function": "ref2015",
                "ddg_bind_rank_desc": 1,
                "sidechain_ddg_bind_rank_desc": 1 if interpretable else pd.NA,
            }
        )
    energy_path = root / f"{protein_id}_energy_alanine_scan_by_position.parquet"
    pd.DataFrame(rows).to_parquet(energy_path, index=False)
    metadata = {
        "schema_version": "tetramer_reference_metrics_v2",
        "name": f"{protein_id}_energy",
        "source_sha256": _sha256_text(f"{protein_id}:s0"),
        "row_counts": {"alanine_scan_by_position": len(rows)},
        "parameters": {
            "alanine_scan_pairs": [["A", "B"], ["A", "C"]],
            "ddg_sign_convention": "mutant_minus_wildtype",
            "interface_residue_cutoff_a": 5.0,
            "interpretable_sidechain_scan_excludes": ["ALA", "GLY", "PRO"],
            "rosetta_score_function": "ref2015",
        },
    }
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return energy_path, metadata_path


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    sequences = {"P1": "ACDEFGHIKLMNPQ", "P2": "MNPQRSTVWYACDE"}
    fasta = tmp_path / "parents.fasta"
    fasta.write_text("".join(f">{protein_id}\n{sequence}\n" for protein_id, sequence in sequences.items()))

    cores = []
    starts = dict(zip(ALLELES, (0, 1, 2), strict=True))
    for protein_id, sequence in sequences.items():
        for allele, start in starts.items():
            cores.append(
                {
                    "protein_id": protein_id,
                    "allele": allele,
                    "core_start_0b": start,
                    "core_start_1b": start + 1,
                    "core_seq": sequence[start : start + 9],
                    "best_rank_EL": 0.5,
                    "n_supporting_windows": 2,
                }
            )
    cores.append(
        {
            "protein_id": "P_EXTRA",
            "allele": ALLELES[0],
            "core_start_0b": 0,
            "core_start_1b": 1,
            "core_seq": "ACDEFGHIK",
            "best_rank_EL": 0.1,
            "n_supporting_windows": 1,
        }
    )
    cores_path = tmp_path / "cores.csv"
    pd.DataFrame(cores).to_csv(cores_path, index=False)

    offsets = {
        "all_core_residues": tuple(range(9)),
        "anchors_P1P4P6P9": (0, 3, 5, 8),
        "P1_plus_P4": (0, 3),
        "P1_only": (0,),
    }
    baseline = []
    for protein_id, sequence in sequences.items():
        for allele in ALLELES:
            for policy, policy_offsets in offsets.items():
                count = len(policy_offsets)
                fraction = round(count / len(sequence), 4)
                baseline.append(
                    {
                        "protein_id": protein_id,
                        "allele": allele,
                        "sequence_length": len(sequence),
                        "open_policy": policy,
                        "n_open_positions": count,
                        "open_fraction": fraction,
                        "min_recovery_if_all_open": round(1 - count / len(sequence), 4),
                        "in_90_97_recovery_band": 0.03 <= count / len(sequence) <= 0.10,
                        "n_cores": 1,
                    }
                )
    baseline_path = tmp_path / "baseline.csv"
    pd.DataFrame(baseline).to_csv(baseline_path, index=False)

    evolution_dir = tmp_path / "evolution" / "P1"
    _write_evolution(evolution_dir, "P1", sequences["P1"])
    s1 = _write_structure(
        tmp_path / "structure" / "P1", "P1", sequences["P1"], resolved=True
    )
    s2 = _write_structure(
        tmp_path / "structure" / "P2", "P2", sequences["P2"], resolved=False
    )
    e1 = _write_energy(tmp_path / "energy" / "P1", "P1", sequences["P1"])
    manifest_path = tmp_path / "evidence_manifest.tsv"
    pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "evolution_status": "available",
                "evolution_dir": str(evolution_dir),
                "structure_status": "qualified",
                "structure_consensus_path": str(s1[0]),
                "structure_masks_path": str(s1[1]),
                "structure_metadata_path": str(s1[2]),
                "structure_residue_pairs_path": str(s1[3]),
                "structure_residue_mapping_path": str(s1[4]),
                "energy_status": "available",
                "energy_per_position_path": str(e1[0]),
                "energy_metadata_path": str(e1[1]),
            },
            {
                "protein_id": "P2",
                "evolution_status": "unavailable",
                "evolution_dir": "",
                "structure_status": "qualified",
                "structure_consensus_path": str(s2[0]),
                "structure_masks_path": str(s2[1]),
                "structure_metadata_path": str(s2[2]),
                "structure_residue_pairs_path": str(s2[3]),
                "structure_residue_mapping_path": str(s2[4]),
                "energy_status": "unavailable",
                "energy_per_position_path": "",
                "energy_metadata_path": "",
            },
        ]
    ).to_csv(manifest_path, sep="\t", index=False)

    legacy_path = tmp_path / "legacy.csv"
    pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "hard_anchor_indices_0b": "0",
                "hard_anchor_labels": "legacy_A",
                "hard_anchor_expected_aa": "A",
                "monitored_shell_indices_0b": "1",
                "monitored_shell_labels": "legacy_C",
                "source_config": "legacy.yaml",
            },
            {
                "protein_id": "P2",
                "hard_anchor_indices_0b": "0",
                "hard_anchor_labels": "legacy_M",
                "hard_anchor_expected_aa": "M",
                "monitored_shell_indices_0b": "",
                "monitored_shell_labels": "",
                "source_config": "legacy.yaml",
            },
        ]
    ).to_csv(legacy_path, index=False)

    analog_path = tmp_path / "homolog_analogs.csv"
    pd.DataFrame(
        [
            {
                "protein_id": "P2",
                "target_index_0b": 2,
                "actual_aa": "P",
                "analog_role": "Phe160",
                "mapping_status": "resolved",
                "source": "synthetic_alignment",
            }
        ]
    ).to_csv(analog_path, index=False)
    return {
        "fasta": fasta,
        "cores": cores_path,
        "baseline": baseline_path,
        "manifest": manifest_path,
        "legacy": legacy_path,
        "analogs": analog_path,
    }


def _run(paths: dict[str, Path], out_dir: Path) -> None:
    assert main(
        [
            "--parent-fasta",
            str(paths["fasta"]),
            "--evidence-manifest",
            str(paths["manifest"]),
            "--cores-long",
            str(paths["cores"]),
            "--open-policy-baseline",
            str(paths["baseline"]),
            "--legacy-identity-map",
            str(paths["legacy"]),
            "--homolog-analog-annotations",
            str(paths["analogs"]),
            "--expected-parent-count",
            "2",
            "--out-dir",
            str(out_dir),
        ]
    ) == 0


def test_join_writes_four_tables_with_nullable_evidence_and_exact_baseline(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run(paths, out_dir)

    expected_stems = {
        "parent_position_evidence",
        "epitope_core_position_evidence",
        "core_policy_mask_tradeoff",
        "parent_allele_policy_mask_tradeoff",
    }
    for stem in expected_stems:
        assert (out_dir / f"{stem}.parquet").is_file()
        assert (out_dir / f"{stem}.tsv").is_file()

    positions = pd.read_parquet(out_dir / "parent_position_evidence.parquet")
    assert len(positions) == 28
    assert set(positions.protein_id) == {"P1", "P2"}
    assert positions.loc[positions.protein_id == "P2", "in_C90_stable"].isna().all()
    assert positions.loc[positions.protein_id == "P2", "in_mid_contact_5of5"].isna().all()
    assert bool(positions.loc[(positions.protein_id == "P1") & (positions.index_0b == 0), "legacy_identity_required_overlap"].iloc[0])
    assert bool(positions.loc[(positions.protein_id == "P2") & (positions.index_0b == 2), "homolog_analog_overlap"].iloc[0])
    assert not positions.loc[positions.protein_id == "P1", "homolog_analog_overlap"].any()
    p1_positions = positions[positions.protein_id == "P1"].set_index("index_0b")
    assert bool(p1_positions.loc[0, "in_mid_contact_5of5"])
    assert not bool(p1_positions.loc[2, "in_mid_contact_5of5"])
    assert bool(p1_positions.loc[2, "in_full_contact_5of5"])
    assert p1_positions.loc[1, "energy_catalytic_ddg_bind_mean_reu"] == pytest.approx(
        2.5
    )
    assert p1_positions.loc[1, "energy_assembly_ddg_bind_mean_reu"] == pytest.approx(
        0.5
    )
    assert p1_positions.loc[1, "energy_ddg_bind_max_reu"] == pytest.approx(2.5)
    energy_table_sha256 = str(p1_positions.loc[1, "energy_table_sha256"])
    assert len(energy_table_sha256) == 64
    assert set(energy_table_sha256).issubset(set("0123456789abcdef"))
    assert p1_positions["energy_table_sha256"].nunique() == 1
    assert bool(p1_positions.loc[1, "in_energy_ddg_gt0_reu"])
    assert bool(p1_positions.loc[1, "in_energy_ddg_ge1_reu"])
    assert bool(p1_positions.loc[1, "in_energy_ddg_ge2_reu"])
    assert not bool(p1_positions.loc[2, "in_energy_ddg_gt0_reu"])
    assert pd.isna(p1_positions.loc[0, "in_energy_ddg_gt0_reu"])
    assert p1_positions.loc[0, "energy_position_status"] == (
        "scanned_noninterpretable_AGP"
    )
    assert p1_positions.loc[0, "energy_raw_ddg_bind_max_reu"] == pytest.approx(0.0)
    assert pd.isna(p1_positions.loc[0, "energy_ddg_bind_max_reu"])
    assert pd.isna(p1_positions.loc[0, "energy_catalytic_ddg_bind_mean_reu"])
    assert p1_positions.loc[
        0, "energy_catalytic_raw_ddg_bind_mean_reu"
    ] == pytest.approx(0.0)
    assert not bool(p1_positions.loc[3, "in_energy_ddg_gt0_reu"])
    assert p1_positions.loc[3, "energy_position_status"] == "not_scanned"
    assert pd.isna(p1_positions.loc[3, "energy_raw_ddg_bind_max_reu"])
    assert positions.loc[
        positions.protein_id == "P2", "in_energy_ddg_gt0_reu"
    ].isna().all()

    core_positions = pd.read_parquet(out_dir / "epitope_core_position_evidence.parquet")
    assert len(core_positions) == 2 * 3 * 9
    assert set(core_positions.core_register) == {f"P{index}" for index in range(1, 10)}
    assert "P_EXTRA" not in set(core_positions.protein_id)

    core_tradeoff = pd.read_parquet(out_dir / "core_policy_mask_tradeoff.parquet")
    row = core_tradeoff[
        (core_tradeoff.protein_id == "P1")
        & (core_tradeoff.allele == ALLELES[0])
        & (core_tradeoff.core_start_0b == 0)
        & (core_tradeoff.open_policy == "P1_plus_P4")
        & (core_tradeoff.evidence_mask == "C90_stable")
    ].iloc[0]
    assert row.n_collision_positions == 1
    assert json.loads(row.positions_not_in_mask_index_0b_json) == [3]
    unavailable = core_tradeoff[
        (core_tradeoff.protein_id == "P2")
        & (core_tradeoff.evidence_mask == "C90_stable")
    ]
    assert unavailable.n_collision_positions.isna().all()
    energy_unknown = core_tradeoff[
        (core_tradeoff.protein_id == "P1")
        & (core_tradeoff.allele == ALLELES[0])
        & (core_tradeoff.core_start_0b == 0)
        & (core_tradeoff.open_policy == "P1_only")
        & (core_tradeoff.evidence_mask == "energy_ddg_gt0_reu")
    ].iloc[0]
    assert pd.isna(energy_unknown.n_collision_positions)
    assert energy_unknown.n_unknown_positions == 1
    assert energy_unknown.n_collision_known == 0
    assert energy_unknown.n_collision_min == 0
    assert energy_unknown.n_collision_max == 1
    assert energy_unknown.n_collision_positions_lower_bound == 0
    assert energy_unknown.n_collision_positions_upper_bound == 1
    assert energy_unknown.n_positions_not_in_mask_lower_bound == 0
    assert energy_unknown.n_positions_not_in_mask_upper_bound == 1

    parent_tradeoff = pd.read_parquet(
        out_dir / "parent_allele_policy_mask_tradeoff.parquet"
    )
    none_rows = parent_tradeoff[parent_tradeoff.evidence_mask == "none"].sort_values(
        ["protein_id", "allele", "open_policy"]
    )
    baseline = pd.read_csv(paths["baseline"]).sort_values(
        ["protein_id", "allele", "open_policy"]
    )
    assert len(none_rows) == len(baseline)
    assert none_rows.n_remaining_open_positions.tolist() == baseline.n_open_positions.tolist()
    assert none_rows.remaining_open_fraction.tolist() == baseline.open_fraction.tolist()
    assert none_rows.min_recovery_if_all_remaining_open.tolist() == baseline.min_recovery_if_all_open.tolist()
    parent_energy_unknown = parent_tradeoff[
        (parent_tradeoff.protein_id == "P1")
        & (parent_tradeoff.allele == ALLELES[0])
        & (parent_tradeoff.open_policy == "P1_only")
        & (parent_tradeoff.evidence_mask == "energy_ddg_gt0_reu")
    ].iloc[0]
    assert pd.isna(parent_energy_unknown.n_remaining_open_positions)
    assert parent_energy_unknown.n_unknown_positions == 1
    assert parent_energy_unknown.n_collision_known == 0
    assert parent_energy_unknown.n_collision_min == 0
    assert parent_energy_unknown.n_collision_max == 1
    assert parent_energy_unknown.n_remaining_open_positions_lower_bound == 0
    assert parent_energy_unknown.n_remaining_open_positions_upper_bound == 1

    metadata = json.loads((out_dir / "join_metadata.json").read_text())
    assert metadata["schema_version"] == "uricase_active15_evidence_join_v1"
    assert metadata["cohort"]["parent_count"] == 2
    assert metadata["cohort"]["core_count"] == 6
    expected_outputs = {
        f"{stem}.{suffix}" for stem in expected_stems for suffix in ("parquet", "tsv")
    }
    expected_outputs |= {
        "parent_ec_contact_validation.parquet",
        "parent_ec_contact_validation.tsv",
    }
    assert set(metadata["outputs"]) == expected_outputs
    assert all(len(record["sha256"]) == 64 for record in metadata["inputs"])
    assert all(len(record["sha256"]) == 64 for record in metadata["outputs"].values())

    validation = pd.read_parquet(out_dir / "parent_ec_contact_validation.parquet")
    assert set(validation.protein_id) == {"P1", "P2"}
    p1 = validation.set_index("protein_id").loc["P1"]
    assert p1.top_L_tetramer_precision == pytest.approx(10 / 14)
    assert p1.top_L_tetramer_minus_monomer_precision == pytest.approx(3 / 14)
    assert p1.top_L_tetramer_only_hits == 3
    assert len(json.loads(p1.top_L_tetramer_only_pairs_index_0b_json)) == 3
    assert p1.contact_4of5_top_L_tetramer_precision == pytest.approx(11 / 14)
    assert p1.contact_1of5_top_L_tetramer_precision == pytest.approx(12 / 14)
    assert p1.contact_4of5_top_L_tetramer_only_hits == 3
    assert p1.contact_1of5_top_L_tetramer_only_hits == 3
    assert p1.top_L_gate_status == "pass"
    assert p1.sigma_hard_lock_status == "eligible"
    assert (
        validation.set_index("protein_id").loc["P2", "sigma_hard_lock_status"]
        == "unavailable"
    )
    assert (out_dir / "parent_ec_contact_validation.tsv").is_file()


def test_join_accepts_one_explicit_allele_and_parquet_core_inputs(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    allele = "HLA-DRB1_15_01"
    cores = pd.read_csv(paths["cores"])
    baseline = pd.read_csv(paths["baseline"])
    cores_path = tmp_path / "cores_1501.parquet"
    baseline_path = tmp_path / "baseline_1501.parquet"
    cores[cores.allele.eq(allele)].to_parquet(cores_path, index=False)
    baseline[baseline.allele.eq(allele)].to_parquet(baseline_path, index=False)
    out_dir = tmp_path / "out_1501"

    assert main(
        [
            "--parent-fasta", str(paths["fasta"]),
            "--evidence-manifest", str(paths["manifest"]),
            "--cores-long", str(cores_path),
            "--open-policy-baseline", str(baseline_path),
            "--legacy-identity-map", str(paths["legacy"]),
            "--homolog-analog-annotations", str(paths["analogs"]),
            "--expected-parent-count", "2",
            "--allele", allele,
            "--out-dir", str(out_dir),
        ]
    ) == 0
    expanded = pd.read_parquet(out_dir / "epitope_core_position_evidence.parquet")
    assert len(expanded) == 2 * 9
    assert set(expanded.allele) == {allele}
    metadata = json.loads((out_dir / "join_metadata.json").read_text())
    assert metadata["cohort"]["alleles"] == [allele]


def test_parent_shards_are_an_exact_partition_of_the_full_join(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    full_dir = tmp_path / "full"
    _run(paths, full_dir)

    shard_dirs = []
    for shard_index in range(2):
        shard_dir = tmp_path / f"shard_{shard_index:03d}"
        shard_dirs.append(shard_dir)
        assert main(
            [
                "--parent-fasta", str(paths["fasta"]),
                "--evidence-manifest", str(paths["manifest"]),
                "--cores-long", str(paths["cores"]),
                "--open-policy-baseline", str(paths["baseline"]),
                "--legacy-identity-map", str(paths["legacy"]),
                "--homolog-analog-annotations", str(paths["analogs"]),
                "--expected-parent-count", "2",
                "--n-shards", "2",
                "--shard-index", str(shard_index),
                "--out-dir", str(shard_dir),
            ]
        ) == 0
        metadata = json.loads((shard_dir / "join_metadata.json").read_text())
        assert metadata["shard"] == {
            "count": 2,
            "full_parent_count": 2,
            "index": shard_index,
            "selection": "round_robin",
        }
        assert metadata["cohort"]["parent_ids"] == [f"P{shard_index + 1}"]

    table_keys = {
        "parent_position_evidence": ["protein_id", "index_0b"],
        "epitope_core_position_evidence": [
            "protein_id", "allele", "core_start_0b", "core_offset_0b"
        ],
        "core_policy_mask_tradeoff": [
            "protein_id", "allele", "core_start_0b", "open_policy", "evidence_mask"
        ],
        "parent_allele_policy_mask_tradeoff": [
            "protein_id", "allele", "open_policy", "evidence_mask"
        ],
        "parent_ec_contact_validation": ["protein_id"],
    }
    for table, key in table_keys.items():
        expected = pd.read_parquet(full_dir / f"{table}.parquet")
        observed = pd.concat(
            [pd.read_parquet(path / f"{table}.parquet") for path in shard_dirs],
            ignore_index=True,
        )
        expected = expected.sort_values(key).reset_index(drop=True)
        observed = observed.sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            observed.convert_dtypes(), expected.convert_dtypes(), check_like=True
        )

    merged_dir = tmp_path / "merged"
    merge_shards(
        shard_root=tmp_path,
        shard_name_pattern="shard_{index:03d}",
        parent_fasta=paths["fasta"],
        expected_parent_count=2,
        n_shards=2,
        out_dir=merged_dir,
        command="test merge",
    )
    merged_metadata = json.loads((merged_dir / "join_metadata.json").read_text())
    assert merged_metadata["cohort"]["parent_ids"] == ["P1", "P2"]
    assert merged_metadata["shards"]["count"] == 2
    for table, key in table_keys.items():
        expected = pd.read_parquet(full_dir / f"{table}.parquet").sort_values(key)
        observed = pd.read_parquet(merged_dir / f"{table}.parquet").sort_values(key)
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True).convert_dtypes(),
            expected.reset_index(drop=True).convert_dtypes(),
            check_like=True,
        )


def _set_covariance_gate(
    paths: dict[str, Path], *, covariance_status: str, neff_per_length: float
) -> None:
    parent_dir = paths["manifest"].parent / "evolution" / "P1"
    position_path = parent_dir / "per_position_evolution.tsv"
    positions = pd.read_csv(position_path, sep="\t")
    positions["covariance_status"] = covariance_status
    masks_path = parent_dir / "wt_lock_masks.json"
    masks = json.loads(masks_path.read_text())
    masks["covariance_status"] = covariance_status
    sigma_status = {
        "exploratory": "exploratory_not_hard_lock",
        "insufficient": "suppressed_low_neff",
    }[covariance_status]
    for mask, definition in masks["masks"].items():
        if mask.startswith("sigma"):
            definition["status"] = sigma_status
            if covariance_status == "insufficient":
                positions[f"in_{mask}"] = False
                definition["n_positions"] = 0
                definition["positions_index_0b"] = []
                definition["positions_position_1b"] = []
                definition["position_labels"] = []
    positions.to_csv(position_path, sep="\t", index=False)
    masks_path.write_text(json.dumps(masks, indent=2) + "\n")
    metadata_path = parent_dir / "analysis_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model"]["neff_per_length"] = neff_per_length
    metadata["model"]["covariance_status"] = covariance_status
    metadata["contact_gate"]["status"] = "not_computed"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    (parent_dir / "ec_contact_monomer_tetramer_comparison.tsv").unlink()


def test_exploratory_sigma_is_descriptive_but_membership_remains_available(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    _set_covariance_gate(paths, covariance_status="exploratory", neff_per_length=8.0)
    out_dir = tmp_path / "out"
    _run(paths, out_dir)

    positions = pd.read_parquet(out_dir / "parent_position_evidence.parquet")
    p1 = positions[positions.protein_id == "P1"]
    assert p1.in_sigma90_L.notna().all()
    assert p1.in_sigma90_L.any()
    validation = pd.read_parquet(out_dir / "parent_ec_contact_validation.parquet")
    assert (
        validation.set_index("protein_id").loc["P1", "sigma_hard_lock_status"]
        == "exploratory_not_hard_lock"
    )
    tradeoff = pd.read_parquet(out_dir / "core_policy_mask_tradeoff.parquet")
    sigma = tradeoff[
        (tradeoff.protein_id == "P1") & (tradeoff.evidence_mask == "sigma90_L")
    ]
    assert sigma.n_collision_positions.notna().all()
    assert set(sigma.hard_lock_use_status) == {"exploratory_not_hard_lock"}


def test_insufficient_sigma_is_nullable_not_false_free_space(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    _set_covariance_gate(paths, covariance_status="insufficient", neff_per_length=4.0)
    out_dir = tmp_path / "out"
    _run(paths, out_dir)

    positions = pd.read_parquet(out_dir / "parent_position_evidence.parquet")
    p1 = positions[positions.protein_id == "P1"]
    assert p1.in_sigma80_L.isna().all()
    validation = pd.read_parquet(out_dir / "parent_ec_contact_validation.parquet")
    assert (
        validation.set_index("protein_id").loc["P1", "sigma_hard_lock_status"]
        == "suppressed_low_neff"
    )
    tradeoff = pd.read_parquet(out_dir / "core_policy_mask_tradeoff.parquet")
    sigma = tradeoff[
        (tradeoff.protein_id == "P1") & (tradeoff.evidence_mask == "sigma80_L")
    ]
    assert sigma.n_collision_positions.isna().all()
    assert sigma.n_positions_not_in_mask.isna().all()


def test_degenerate_sigma_column_is_normalized_and_suppressed(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    parent_dir = paths["manifest"].parent / "evolution" / "P1"
    position_path = parent_dir / "per_position_evolution.tsv"
    positions = pd.read_csv(position_path, sep="\t")
    positions["sigma_evidence_status"] = "unavailable_nonpositive_cn"
    for mask in MASKS:
        if mask.startswith("sigma"):
            positions[f"in_{mask}"] = False
    positions.to_csv(position_path, sep="\t", index=False)

    masks_path = parent_dir / "wt_lock_masks.json"
    masks = json.loads(masks_path.read_text())
    for mask, definition in masks["masks"].items():
        if mask.startswith("sigma"):
            definition["status"] = "suppressed_degenerate_couplings"
            definition["n_positions"] = 0
            definition["positions_index_0b"] = []
            definition["positions_position_1b"] = []
            definition["position_labels"] = []
    masks_path.write_text(json.dumps(masks))

    metadata_path = parent_dir / "analysis_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["ec_table"] = {
        "sigma_evidence_status": "unavailable_nonpositive_cn"
    }
    metadata_path.write_text(json.dumps(metadata))

    out_dir = tmp_path / "out"
    _run(paths, out_dir)
    parent = pd.read_parquet(out_dir / "parent_position_evidence.parquet")
    p1 = parent[parent.protein_id.eq("P1")]
    assert set(p1.sigma_evidence_status) == {"unavailable_nonpositive_cn"}
    assert p1.in_sigma80_robust.isna().all()
    validation = pd.read_parquet(out_dir / "parent_ec_contact_validation.parquet")
    assert (
        validation.set_index("protein_id").loc["P1", "sigma_hard_lock_status"]
        == "suppressed_degenerate_couplings"
    )


def test_core_sequence_mismatch_fails_before_output(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    cores = pd.read_csv(paths["cores"])
    cores.loc[(cores.protein_id == "P1") & (cores.allele == ALLELES[0]), "core_seq"] = "AAAAAAAAA"
    cores.to_csv(paths["cores"], index=False)
    out_dir = tmp_path / "out"

    with pytest.raises(JoinContractError, match="core_seq"):
        _run(paths, out_dir)
    assert not out_dir.exists()


def test_baseline_count_drift_fails_before_output(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    baseline = pd.read_csv(paths["baseline"])
    baseline.loc[0, "n_open_positions"] += 1
    baseline.to_csv(paths["baseline"], index=False)
    out_dir = tmp_path / "out"

    with pytest.raises(JoinContractError, match="baseline n_open_positions"):
        _run(paths, out_dir)
    assert not out_dir.exists()

def _p1_residue_pairs_path(paths: dict[str, Path]) -> Path:
    manifest = pd.read_csv(paths["manifest"], sep="	")
    return Path(
        manifest.loc[
            manifest["protein_id"].eq("P1"), "structure_residue_pairs_path"
        ].iloc[0]
    )


def test_direct_t6_requires_both_biological_interface_copies(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    contact_path = _p1_residue_pairs_path(paths)
    contacts = pd.read_parquet(contact_path)
    biological = contacts[
        contacts["is_inter_chain"]
        & contacts["mechanistic_class"].isin({"catalytic", "assembly"})
    ]
    target = biological[
        ["canonical_pair_min_0b", "canonical_pair_max_0b"]
    ].iloc[0]
    remove = (
        contacts["canonical_pair_min_0b"].eq(target["canonical_pair_min_0b"])
        & contacts["canonical_pair_max_0b"].eq(target["canonical_pair_max_0b"])
        & contacts["interface_copy"].eq(2)
    )
    contacts.loc[~remove].to_parquet(contact_path, index=False)

    out_dir = tmp_path / "out"
    _run(paths, out_dir)
    p1 = pd.read_parquet(
        out_dir / "parent_ec_contact_validation.parquet"
    ).set_index("protein_id").loc["P1"]
    assert p1.top_L_tetramer_hits == 9
    assert p1.top_L_tetramer_precision == pytest.approx(9 / 14)
    assert p1.top_L_tetramer_only_hits == 2


def test_direct_t6_rejects_incomplete_ensemble_provenance(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    contact_path = _p1_residue_pairs_path(paths)
    contacts = pd.read_parquet(contact_path)
    contacts.loc[~contacts["sample_id"].eq("s4")].to_parquet(
        contact_path, index=False
    )
    out_dir = tmp_path / "out"

    with pytest.raises(JoinContractError, match="sample provenance mismatch"):
        _run(paths, out_dir)
    assert not out_dir.exists()


def test_direct_t6_rejects_incomplete_ec_table(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    ec_path = paths["manifest"].parent / "evolution" / "P1" / "complete_ec_table.tsv"
    ecs = pd.read_csv(ec_path, sep="	")
    ecs.iloc[:-1].to_csv(ec_path, sep="	", index=False)
    out_dir = tmp_path / "out"

    with pytest.raises(JoinContractError, match="exactly one row for every"):
        _run(paths, out_dir)
    assert not out_dir.exists()


def test_unresolved_structure_keeps_qualified_sigma_pending(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    sequence = "ACDEFGHIKLMNPQ"
    unresolved = _write_structure(
        tmp_path / "structure" / "P1_unresolved",
        "P1",
        sequence,
        resolved=False,
    )
    manifest = pd.read_csv(paths["manifest"], sep="	")
    hit = manifest["protein_id"].eq("P1")
    manifest.loc[hit, "structure_consensus_path"] = str(unresolved[0])
    manifest.loc[hit, "structure_masks_path"] = str(unresolved[1])
    manifest.loc[hit, "structure_metadata_path"] = str(unresolved[2])
    manifest.loc[hit, "structure_residue_pairs_path"] = str(unresolved[3])
    manifest.loc[hit, "structure_residue_mapping_path"] = str(unresolved[4])
    manifest.to_csv(paths["manifest"], sep="	", index=False)

    out_dir = tmp_path / "out"
    _run(paths, out_dir)
    validation = pd.read_parquet(
        out_dir / "parent_ec_contact_validation.parquet"
    ).set_index("protein_id")
    assert (
        validation.loc["P1", "top_L_gate_status"]
        == "pending_ensemble_contact_validation"
    )
    assert (
        validation.loc["P1", "sigma_hard_lock_status"]
        == "pending_ensemble_contact_validation"
    )
    positions = pd.read_parquet(out_dir / "parent_position_evidence.parquet")
    assert set(
        positions.loc[
            positions["protein_id"].eq("P1"), "sigma_hard_lock_status"
        ]
    ) == {"pending_ensemble_contact_validation"}

def _p1_energy_paths(paths: dict[str, Path]) -> tuple[Path, Path]:
    manifest = pd.read_csv(paths["manifest"], sep="	")
    row = manifest.loc[manifest["protein_id"].eq("P1")].iloc[0]
    return Path(row.energy_per_position_path), Path(row.energy_metadata_path)


def test_energy_reversed_chain_pair_is_matched_as_undirected(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    energy_path, _ = _p1_energy_paths(paths)
    energy = pd.read_parquet(energy_path)
    energy.loc[energy["interface_pair"].eq("A:B"), "interface_pair"] = "B:A"
    energy.to_parquet(energy_path, index=False)

    out_dir = tmp_path / "out"
    _run(paths, out_dir)
    positions = pd.read_parquet(out_dir / "parent_position_evidence.parquet")
    p1 = positions[positions["protein_id"].eq("P1")].set_index("index_0b")
    assert p1.loc[1, "energy_catalytic_ddg_bind_mean_reu"] == pytest.approx(2.5)


def test_disulfide_cys_is_retained_as_scanned_noninterpretable(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    energy_path, metadata_path = _p1_energy_paths(paths)
    energy = pd.read_parquet(energy_path)
    # The same canonical Cys is disulfide-bound in only one biological
    # interface. Global position evidence must use any(disulfide)/all(interpretable).
    disulfide = energy["wt_residue_name3"].eq("CYS") & energy[
        "interface_pair"
    ].eq("A:B")
    energy["is_disulfide_cysteine"] = disulfide
    energy.loc[disulfide, "is_interpretable_sidechain_alanine"] = False
    for column in (
        "ddg_bind_mean_reu",
        "ddg_bind_min_reu",
        "ddg_bind_max_reu",
        "ddg_bind_chain_range_reu",
    ):
        energy.loc[disulfide, column] = float("nan")
    energy.to_parquet(energy_path, index=False)
    metadata = json.loads(metadata_path.read_text())
    metadata["parameters"]["interpretable_sidechain_scan_excludes"].append(
        "DISULFIDE_CYS"
    )
    metadata_path.write_text(json.dumps(metadata))

    out_dir = tmp_path / "out"
    _run(paths, out_dir)
    row = pd.read_parquet(
        out_dir / "parent_position_evidence.parquet"
    ).set_index(["protein_id", "index_0b"]).loc[("P1", 1)]
    assert row.energy_position_status == "scanned_noninterpretable_disulfide_CYS"
    assert bool(row.energy_is_disulfide_cysteine)
    assert pd.isna(row.in_energy_ddg_ge1_reu)
    assert pd.isna(row.energy_ddg_bind_max_reu)


def test_energy_rejects_contact_source_sha_drift(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    contact_path = _p1_residue_pairs_path(paths)
    contacts = pd.read_parquet(contact_path)
    contacts.loc[contacts["sample_id"].eq("s0"), "source_sha256"] = "0" * 64
    contacts.to_parquet(contact_path, index=False)
    out_dir = tmp_path / "out"

    with pytest.raises(JoinContractError, match="source SHA256 differs"):
        _run(paths, out_dir)
    assert not out_dir.exists()


def test_energy_rejects_source_chains_outside_interface_pair(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    energy_path, _ = _p1_energy_paths(paths)
    energy = pd.read_parquet(energy_path)
    energy.loc[0, "source_chains"] = "A,C"
    energy.to_parquet(energy_path, index=False)
    out_dir = tmp_path / "out"

    with pytest.raises(JoinContractError, match="source_chains disagree"):
        _run(paths, out_dir)
    assert not out_dir.exists()


def test_energy_rejects_bsa_annotation_drift(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    energy_path, _ = _p1_energy_paths(paths)
    energy = pd.read_parquet(energy_path)
    energy.loc[energy["interface_pair"].eq("A:B"), "interface_class"] = "interface_2"
    energy.to_parquet(energy_path, index=False)
    out_dir = tmp_path / "out"

    with pytest.raises(JoinContractError, match="BSA annotations differ"):
        _run(paths, out_dir)
    assert not out_dir.exists()


def test_energy_rejects_numeric_qc_drift(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    energy_path, _ = _p1_energy_paths(paths)
    energy = pd.read_parquet(energy_path)
    energy.loc[0, "ddg_bind_chain_range_reu"] = 0.9
    energy.to_parquet(energy_path, index=False)
    out_dir = tmp_path / "out"

    with pytest.raises(JoinContractError, match="chain range differs"):
        _run(paths, out_dir)
    assert not out_dir.exists()
