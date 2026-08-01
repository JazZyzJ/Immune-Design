from __future__ import annotations

import pandas as pd

from scripts.analyze_luxsit_refinement_holo import (
    apply_refinement_gates,
    select_seed_unique_shortlist,
    select_shortlist,
)


def _passing_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "background": "i",
        "design_uid": "s00_LuxSit-i_core117_d0001",
        "seed_design_idx": 170,
        "sample_integrity_pass_pass_count": 5,
        "sample_confidence_pass_pass_count": 5,
        "free3_sample_pass_pass_count": 5,
        "scTM": 0.98,
        "global_ca_RMSD": 0.55,
        "holo_global_ca_rmsd_median": 0.55,
        "anchor_sc_rmsd_median": 0.40,
        "cat4_sc_rmsd_median": 0.35,
        "anchor_max_sc_rmsd_median": 1.10,
        "geometry_mae_median": 0.10,
        "geometry_max_median": 0.25,
        "ligand_com_median": 0.20,
        "ligand_core_rmsd_median": 0.35,
        "ligand_full_rmsd_median": 1.10,
        "iptm_median": 0.97,
        "ligand_plddt_median": 95.0,
        "seed_scTM": 0.97,
        "seed_global_ca_RMSD": 0.65,
        "seed_holo_global_ca_rmsd": 0.50,
        "seed_anchor_sc_rmsd": 0.35,
        "seed_cat4_sc_rmsd": 0.30,
        "seed_anchor_max_sc_rmsd": 0.90,
        "seed_geometry_mae": 0.08,
        "seed_geometry_max": 0.20,
        "seed_ligand_com": 0.15,
        "seed_ligand_full_rmsd": 1.00,
        "seed_iptm": 0.97,
        "seed_ligand_plddt": 96.0,
        "core_count_before": 2,
        "core_count_after": 0,
        "seed_n_strong_binders": 8,
        "n_strong_binders": 0,
        "n_mutations": 2,
        "actual_mutation_count": 2,
    }
    for metric, value in {
        "holo_global_ca_rmsd_q80": 0.60,
        "anchor_sc_rmsd_q80": 0.45,
        "cat4_sc_rmsd_q80": 0.40,
        "anchor_max_sc_rmsd_q80": 1.20,
        "geometry_mae_q80": 0.12,
        "geometry_max_q80": 0.30,
        "ligand_com_q80": 0.25,
        "ligand_core_rmsd_q80": 0.40,
        "ligand_full_rmsd_q80": 1.20,
    }.items():
        row[metric] = value
    row.update(overrides)
    return row


def test_strict_gate_requires_relative_structure_and_immune_improvement() -> None:
    passing = apply_refinement_gates(
        pd.DataFrame([_passing_row()]), background="i", iptm_floor=0.95,
        ligand_plddt_floor=90.0,
    ).iloc[0]
    assert passing.strict_experimental_pass
    assert passing.core_zero

    relative_failure = apply_refinement_gates(
        pd.DataFrame([_passing_row(anchor_sc_rmsd_median=0.70)]), background="i",
        iptm_floor=0.95, ligand_plddt_floor=90.0,
    ).iloc[0]
    assert relative_failure.G2_anchor_geometry
    assert not relative_failure.R2_anchor_vs_seed
    assert not relative_failure.strict_experimental_pass

    immune_failure = apply_refinement_gates(
        pd.DataFrame([_passing_row(n_strong_binders=8)]), background="i",
        iptm_floor=0.95, ligand_plddt_floor=90.0,
    ).iloc[0]
    assert not immune_failure.immune_improved_both
    assert not immune_failure.strict_experimental_pass


def test_shortlist_does_not_pad_when_diversity_is_unavailable() -> None:
    rows = []
    for index, score in enumerate((0.10, 0.20, 0.30)):
        row = _passing_row(
            design_uid=f"uid_{index}", seed_design_idx=170,
            strict_experimental_pass=True, core_zero=True,
            structure_score=score, iptm_median=0.97,
        )
        row.update({
            "seed_design_id": "design_0170",
            "sequence_refined": "A" * 117,
            "muts": f"A{index + 1}V",
            "actual_mutations": f"A{index + 1}V",
        })
        rows.append(row)
    selected = select_shortlist(pd.DataFrame(rows), n_per_background=5)
    assert len(selected) == 3
    assert selected.seed_design_idx.nunique() == 1
    assert selected.structure_score.tolist() == [0.10, 0.20, 0.30]


def test_shortlist_adds_only_near_frontier_seed_families() -> None:
    rows = []
    for index, (seed, score) in enumerate(((1, 0.10), (1, 0.11), (1, 0.12), (2, 0.14), (3, 0.30))):
        row = _passing_row(
            design_uid=f"uid_{index}", seed_design_idx=seed,
            strict_experimental_pass=True, core_zero=True,
            structure_score=score, iptm_median=0.97,
        )
        row.update({
            "seed_design_id": f"design_{seed:04d}",
            "sequence_refined": "A" * 116 + str(index),
            "muts": f"A{index + 1}V",
            "actual_mutations": f"A{index + 1}V",
        })
        rows.append(row)
    selected = select_shortlist(
        pd.DataFrame(rows), n_per_background=3, min_seed_families=3,
        diversity_margin=0.05,
    )
    assert selected.seed_design_idx.nunique() == 2
    assert 2 in set(selected.seed_design_idx)
    assert 3 not in set(selected.seed_design_idx)


def test_seed_unique_shortlist_selects_best_structure_per_seed() -> None:
    rows = []
    for index, (background, seed, score) in enumerate((
        ("i", 1, 0.20), ("i", 1, 0.10), ("i", 2, 0.15),
        ("parent", 3, 0.30), ("parent", 4, 0.25),
    )):
        row = _passing_row(
            background=background, design_uid=f"uid_{index}", seed_design_idx=seed,
            strict_experimental_pass=True, core_zero=True, structure_score=score,
            iptm_median=0.97,
        )
        row.update({
            "seed_design_id": f"design_{seed:04d}",
            "seed_sequence": f"SEED_{background}_{seed}",
            "sequence_refined": "A" * 116 + str(index),
            "muts": f"A{index + 1}V",
            "actual_mutations": f"A{index + 1}V",
        })
        rows.append(row)

    selected = select_seed_unique_shortlist(pd.DataFrame(rows), n_per_background=5)

    assert len(selected) == 4
    assert not selected.duplicated(["background", "seed_design_idx"]).any()
    assert set(selected.design_uid) == {"uid_1", "uid_2", "uid_3", "uid_4"}
    assert selected.groupby("background").structure_score.apply(list).to_dict() == {
        "i": [0.10, 0.15], "parent": [0.25, 0.30],
    }


def test_seed_unique_shortlist_deduplicates_actual_seed_sequence() -> None:
    rows = []
    for index, (background, seed, score) in enumerate((
        ("i", 1, 0.20), ("parent", 9, 0.10),
    )):
        row = _passing_row(
            background=background, design_uid=f"uid_{index}", seed_design_idx=seed,
            strict_experimental_pass=True, core_zero=True, structure_score=score,
            iptm_median=0.97,
        )
        row.update({
            "seed_design_id": f"design_{seed:04d}", "seed_sequence": "SAME_SEED",
            "sequence_refined": "A" * 116 + str(index), "muts": f"A{index + 1}V",
            "actual_mutations": f"A{index + 1}V",
        })
        rows.append(row)

    selected = select_seed_unique_shortlist(pd.DataFrame(rows), n_per_background=5)

    assert selected.design_uid.tolist() == ["uid_1"]
