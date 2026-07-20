"""Tests for post-prediction uricase tetramer interface metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inverse_folding.evaluation.tetramer_interfaces import (
    classify_d2_interface_pairs,
    pair_residue_contacts,
    pair_salt_bridges,
    parse_rosetta_scorefile,
    summarize_interface_pairs,
)
from scripts.eval_tetramer_gate import add_interface_wt_normalization


def _pair_table() -> pd.DataFrame:
    values = {
        ("A", "B"): 6000.0,
        ("A", "C"): 700.0,
        ("A", "D"): 5100.0,
        ("B", "C"): 5000.0,
        ("B", "D"): 650.0,
        ("C", "D"): 5900.0,
    }
    return pd.DataFrame([
        {
            "chain_1": a,
            "chain_2": b,
            "bsa_total_a2": bsa,
            "bsa_per_partner_a2": bsa / 2,
            "n_residue_contacts_8a": int(bsa / 30),
            "n_salt_bridges_4a": int(bsa / 2000),
        }
        for (a, b), bsa in values.items()
    ])


def _atom_array(rows):
    import biotite.structure as struc

    arr = struc.AtomArray(len(rows))
    arr.coord = np.asarray([row[0] for row in rows], dtype=float)
    arr.chain_id = np.asarray([row[1] for row in rows])
    arr.res_id = np.asarray([row[2] for row in rows])
    arr.res_name = np.asarray([row[3] for row in rows])
    arr.atom_name = np.asarray([row[4] for row in rows])
    arr.element = np.asarray([row[5] for row in rows])
    arr.hetero = np.zeros(len(rows), dtype=bool)
    return arr


def test_d2_classification_keeps_disjoint_symmetry_copies_together():
    out = classify_d2_interface_pairs(_pair_table())
    classes = {
        row.chain_1 + row.chain_2: row.interface_class
        for row in out.itertuples()
    }
    assert classes == {
        "AB": "interface_1",
        "AC": "diagonal",
        "AD": "interface_2",
        "BC": "interface_2",
        "BD": "diagonal",
        "CD": "interface_1",
    }
    assert out.groupby("interface_class").size().to_dict() == {
        "diagonal": 2,
        "interface_1": 2,
        "interface_2": 2,
    }


def test_interface_summary_uses_worst_copy_in_the_correct_direction():
    pairs = classify_d2_interface_pairs(_pair_table())
    pairs["rosetta_dg_separated_reu"] = [-20, np.nan, -12, -10, np.nan, -25]
    pairs["rosetta_delta_unsat_hbonds"] = [1, np.nan, 3, 5, np.nan, 2]
    summary = summarize_interface_pairs(pairs)

    assert summary["interface_1_bsa_total_min_a2"] == 5900.0
    assert summary["interface_2_bsa_total_min_a2"] == 5000.0
    assert summary["diagonal_bsa_total_max_a2"] == 700.0
    assert summary["interface_1_rosetta_dg_separated_max_reu"] == -20.0
    assert summary["interface_2_rosetta_dg_separated_max_reu"] == -10.0
    assert summary["interface_2_rosetta_delta_unsat_hbonds_max"] == 5.0
    assert summary["interface_topology_bsa_gap_a2"] == 4300.0


def test_contacts_and_salt_bridges_count_residue_pairs_not_atom_pairs():
    chain_a = _atom_array([
        ((0.0, 0.0, 0.0), "A", 1, "ASP", "CB", "C"),
        ((0.0, 0.0, 0.0), "A", 1, "ASP", "OD1", "O"),
        ((0.0, 0.5, 0.0), "A", 1, "ASP", "OD2", "O"),
        ((20.0, 0.0, 0.0), "A", 2, "GLY", "CA", "C"),
    ])
    chain_b = _atom_array([
        ((3.5, 0.0, 0.0), "B", 1, "ARG", "CB", "C"),
        ((3.5, 0.0, 0.0), "B", 1, "ARG", "NH1", "N"),
        ((3.5, 0.5, 0.0), "B", 1, "ARG", "NH2", "N"),
        ((40.0, 0.0, 0.0), "B", 2, "GLY", "CA", "C"),
    ])
    assert pair_residue_contacts(chain_a, chain_b, cutoff=8.0) == 1
    assert pair_salt_bridges(chain_a, chain_b, cutoff=4.0) == 1


def test_parse_rosetta_scorefile_preserves_canonical_interface_fields(tmp_path):
    score = tmp_path / "scores.sc"
    score.write_text(
        "SEQUENCE:\n"
        "SCORE: total_score dSASA_int dG_separated dG_separated/dSASAx100 "
        "packstat hbonds_int hbond_E_fraction delta_unsatHbonds description\n"
        "SCORE: -100.0 2100.0 -20.0 -0.952 0.71 8 0.23 2 ifc_000000_0001\n"
    )
    out = parse_rosetta_scorefile(score)
    assert out.loc[0, "dG_separated/dSASAx100"] == pytest.approx(-0.952)
    assert out.loc[0, "packstat"] == pytest.approx(0.71)
    assert out.loc[0, "description"] == "ifc_000000_0001"


def test_wt_normalization_uses_real_parent_wt_and_can_fail_fast():
    df = pd.DataFrame([
        {
            "parent": "P1", "kind": "WT", "predicted": True,
            "interface_metrics_ok": True, "interface_1_bsa_total_min_a2": 5000.0,
        },
        {
            "parent": "P1", "kind": "design", "predicted": True,
            "interface_metrics_ok": True, "interface_1_bsa_total_min_a2": 4000.0,
        },
    ])
    out = add_interface_wt_normalization(df, require_wt=True)
    assert out.loc[1, "wt_interface_1_bsa_total_min_a2"] == 5000.0
    assert out.loc[1, "interface_1_bsa_total_min_a2_ratio_to_wt"] == pytest.approx(0.8)
    assert bool(out.loc[1, "interface_wt_baseline_available"])

    without_wt = df[df["kind"] != "WT"].copy()
    with pytest.raises(ValueError, match="P1"):
        add_interface_wt_normalization(without_wt, require_wt=True)
