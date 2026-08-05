"""Synthetic contract tests for the generic evolutionary-evidence analysis.

The real PLMC binary model is deliberately not a fixture: these tests exercise
all deterministic downstream logic with a small model adapter and explicit MSA
weights.  The production CLI lazy-loads EVcouplings for the two dependency-bound
operations (sequence weights and binary-model loading).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inverse_folding.analysis.evolution_evidence import (
    FocusAlignment,
    SequenceRecord,
    build_lock_masks,
    calibrate_potts,
    compute_conservation_evidence,
    compute_coupling_evidence,
    compute_contact_precision,
    load_focus_alignment,
    load_structure_contacts,
    read_complete_ec_table,
    validate_focus_alignment_family,
    write_mask_artifacts,
)


AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _uniform_weights(_ids: list[str], sequences: list[str], _theta: float) -> np.ndarray:
    return np.ones(len(sequences), dtype=float)


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> Path:
    path.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in records))
    return path


def _alignment(label: str, threshold: float, wt: str, rows: list[tuple[str, str]]) -> FocusAlignment:
    return FocusAlignment(
        protein_id="PTEST",
        label=label,
        coverage_threshold=threshold,
        records=[SequenceRecord(name, name, sequence) for name, sequence in rows],
        weights=np.ones(len(rows), dtype=float),
        source_path=Path(f"{label}.fasta"),
    )


def _complete_ec_table(wt: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(1, len(wt) + 1):
        for j in range(i + 1, len(wt) + 1):
            rows.append(
                {
                    "i": i,
                    "A_i": wt[i - 1],
                    "j": j,
                    "A_j": wt[j - 1],
                    "fn": 0.0,
                    "cn": float(1000 - 10 * i - j),
                }
            )
    return pd.DataFrame(rows)


def test_focus_alignment_is_strict_and_family_is_nested(tmp_path: Path) -> None:
    wt = "ACDEFGHI"
    cov60 = load_focus_alignment(
        _write_fasta(tmp_path / "cov60.fa", [("PTEST", wt), ("h1", "ACDEFG--")]),
        protein_id="PTEST",
        wt=wt,
        label="cov60",
        coverage_threshold=0.60,
        theta=0.8,
        weight_provider=_uniform_weights,
    )
    cov80 = load_focus_alignment(
        _write_fasta(tmp_path / "cov80.fa", [("PTEST", wt)]),
        protein_id="PTEST",
        wt=wt,
        label="cov80",
        coverage_threshold=0.80,
        theta=0.8,
        weight_provider=_uniform_weights,
    )
    validate_focus_alignment_family([cov60, cov80], wt)

    malformed = _write_fasta(
        tmp_path / "bad.fa", [("PTEST", wt), ("h1", "ACDEFGx-")]
    )
    with pytest.raises(ValueError, match="normalized AA20.*gap"):
        load_focus_alignment(
            malformed,
            protein_id="PTEST",
            wt=wt,
            label="cov60",
            coverage_threshold=0.60,
            theta=0.8,
            weight_provider=_uniform_weights,
        )

    duplicate_ids = _write_fasta(
        tmp_path / "dup.fa", [("PTEST", wt), ("h1", wt), ("h1", wt)]
    )
    with pytest.raises(ValueError, match="duplicate sequence identifiers"):
        load_focus_alignment(
            duplicate_ids,
            protein_id="PTEST",
            wt=wt,
            label="cov60",
            coverage_threshold=0.60,
            theta=0.8,
            weight_provider=_uniform_weights,
        )


def test_conservation_retains_coverage_and_both_numbering_systems() -> None:
    wt = "ACDE"
    rows = [("PTEST", wt), ("h1", "AC-E")]
    alignments = [
        _alignment("cov60", 0.6, wt, rows),
        _alignment("cov70", 0.7, wt, rows),
        _alignment("cov80", 0.8, wt, rows),
    ]
    long, stable, metadata = compute_conservation_evidence(alignments, wt)

    assert set(long.coverage_threshold) == {0.6, 0.7, 0.8}
    assert set(long.protein_id) == {"PTEST"}
    assert stable.index_0b.tolist() == [0, 1, 2, 3]
    assert stable.position_1b.tolist() == [1, 2, 3, 4]
    pos3 = stable.loc[stable.position_1b.eq(3)].iloc[0]
    assert pos3.gap_frac_max == pytest.approx(0.5)
    assert pos3.pWT_nogap_min == pytest.approx(1.0)
    assert pos3.C_nogap_min == pytest.approx(1.0)
    assert [row["coverage_threshold"] for row in metadata] == [0.6, 0.7, 0.8]


def test_complete_ec_validation_and_sigma_cutoffs(tmp_path: Path) -> None:
    # L=20 leaves enough |i-j|>=6 pairs to exercise the 2L cutoff.
    wt = AA20
    ecs = _complete_ec_table(wt)
    path = tmp_path / "ecs.txt"
    ecs.to_csv(path, sep=" ", header=False, index=False)

    parsed = read_complete_ec_table(path, wt)
    per_position, ranked, stability = compute_coupling_evidence(parsed, wt)

    assert len(parsed) == len(wt) * (len(wt) - 1) // 2
    assert {"sigma_0p5L", "sigma_L", "sigma_2L", "sigma_robust_pct"}.issubset(
        per_position.columns
    )
    assert ranked.in_top_0p5L.sum() == 10
    assert ranked.in_top_L.sum() == 20
    assert ranked.in_top_2L.sum() == 40
    assert set(stability.metric) == {"spearman", "jaccard_top10", "jaccard_top20"}

    incomplete = tmp_path / "incomplete.txt"
    ecs.iloc[:-1].to_csv(incomplete, sep=" ", header=False, index=False)
    with pytest.raises(ValueError, match="complete EC table"):
        read_complete_ec_table(incomplete, wt)


def test_lock_masks_emit_exact_lists_and_suppress_unqualified_sigma(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "protein_id": ["PTEST"] * 4,
            "index_0b": [0, 1, 2, 3],
            "position_1b": [1, 2, 3, 4],
            "wt_aa": list("ACDE"),
            "C_nogap_min": [0.95, 0.85, 0.75, 0.65],
            "pWT_nogap_min": [0.95, 0.85, 0.75, 0.95],
            "gap_frac_max": [0.0, 0.1, 0.1, 0.0],
            "sigma_L_pct": [1.0, 0.9, 0.8, 0.7],
            "sigma_robust_pct": [0.95, 0.85, 0.75, 0.65],
        }
    )
    qualified, summary, positions, payload = build_lock_masks(
        frame.copy(), covariance_status="qualified"
    )
    assert payload["masks"]["C90_stable"]["positions_index_0b"] == [0]
    assert payload["masks"]["C80_stable"]["positions_position_1b"] == [1, 2]
    assert set(positions.protein_id) == {"PTEST"}
    assert qualified.in_sigma90_L.tolist() == [True, True, False, False]

    out = tmp_path / "out"
    write_mask_artifacts(out, summary, positions, payload)
    persisted = json.loads((out / "wt_lock_masks.json").read_text())
    assert persisted["protein_id"] == "PTEST"
    assert persisted["numbering"]["index_0b"] == "zero-based protein position"

    suppressed, _, suppressed_positions, suppressed_payload = build_lock_masks(
        frame.copy(), covariance_status="insufficient"
    )
    assert not suppressed.in_sigma80_L.any()
    assert not suppressed_positions["mask"].str.startswith("sigma").any()
    assert suppressed_payload["masks"]["sigma80_L"]["status"] == "suppressed_low_neff"


class _FakeCouplingsModel:
    def __init__(self, wt: str) -> None:
        self.L = len(wt)
        self.index_list = np.arange(1, len(wt) + 1)
        self.target_seq = list(wt)
        self.alphabet = list("-" + AA20)
        self.N_eff = 250.0
        self.N_valid = 3

    def hamiltonians(self, sequences: list[str]) -> np.ndarray:
        rows = []
        wt = "".join(self.target_seq)
        for sequence in sequences:
            fields = float(sum(a == b for a, b in zip(wt, sequence, strict=True)))
            couplings = float(sum(sequence[i] == sequence[i + 1] for i in range(self.L - 1)))
            rows.append([fields + couplings, couplings, fields])
        return np.asarray(rows, dtype=float)


def test_potts_calibration_is_within_model_and_imputes_only_high_coverage_rows() -> None:
    wt = AA20
    alignment = FocusAlignment(
        protein_id="PTEST",
        label="cov60",
        coverage_threshold=0.6,
        records=[
            SequenceRecord("PTEST", "PTEST", wt),
            SequenceRecord("one_gap", "one_gap", wt[:-1] + "-"),
            SequenceRecord("too_gappy", "too_gappy", wt[:-2] + "--"),
        ],
        weights=np.asarray([1.0, 0.5, 0.25]),
        source_path=Path("cov60.fa"),
    )
    per_sequence, summary = calibrate_potts(
        _FakeCouplingsModel(wt),
        alignment,
        wt,
        protein_id="PTEST",
        minimum_coverage=0.95,
        minimum_unique_gate=3,
    )

    assert set(per_sequence.sequence_id) == {"PTEST", "one_gap"}
    imputed = per_sequence.loc[per_sequence.sequence_id.eq("one_gap")].iloc[0]
    assert imputed.n_gaps_before_imputation == 1
    assert imputed.sequence_imputed == wt
    assert summary["comparison_scope"] == "within_model_only:PTEST"
    assert summary["cross_parent_thresholds_emitted"] is False
    assert summary["calibration_status"] == "insufficient"
    assert summary["natural_reference"]["n_unique_imputed_sequences"] == 1


def _pdb_atom_line(serial: int, aa3: str, chain: str, residue: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {aa3:>3s} {chain}{residue:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00           C\n"
    )


def test_optional_structure_contacts_and_paired_precision(tmp_path: Path) -> None:
    # L=14 leaves >=2L long-range pairs while keeping the fixture readable.
    wt = AA20[:14]
    aa3 = [
        "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS",
        "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN",
    ]

    monomer = tmp_path / "monomer.pdb"
    monomer.write_text(
        "".join(
            _pdb_atom_line(i, name, "A", i, 0.0 if i in {1, 7} else 100.0 + i * 10)
            for i, name in enumerate(aa3, start=1)
        )
        + "END\n"
    )
    mono_map = tmp_path / "monomer_map.tsv"
    pd.DataFrame(
        {
            "chain_id": ["A"] * len(wt),
            "residue_number": list(range(1, len(wt) + 1)),
            "position_1b": list(range(1, len(wt) + 1)),
            "expected_aa": list(wt),
        }
    ).to_csv(mono_map, sep="\t", index=False)

    tetramer = tmp_path / "tetramer.pdb"
    lines: list[str] = []
    mapping_rows: list[dict[str, object]] = []
    serial = 1
    for chain_index, chain in enumerate("ABCD"):
        for position, name in enumerate(aa3, start=1):
            # Pair 2--8 is a tetramer-only contact between chains A and B.
            if (chain, position) in {("A", 2), ("B", 8)}:
                x = 20.0
            else:
                x = 1000.0 * chain_index + 100.0 + position * 10
            lines.append(_pdb_atom_line(serial, name, chain, position, x))
            mapping_rows.append(
                {
                    "chain_id": chain,
                    "residue_number": position,
                    "position_1b": position,
                    "expected_aa": wt[position - 1],
                }
            )
            serial += 1
    tetramer.write_text("".join(lines) + "END\n")
    tet_map = tmp_path / "tetramer_map.tsv"
    pd.DataFrame(mapping_rows).to_csv(tet_map, sep="\t", index=False)

    mono = load_structure_contacts(monomer, mono_map, wt, kind="monomer")
    tetra = load_structure_contacts(tetramer, tet_map, wt, kind="tetramer")
    assert (1, 7) in mono.contacts
    assert (2, 8) in tetra.contacts

    ecs = _complete_ec_table(wt)
    # Put the two known long-range pairs first.
    ecs.loc[(ecs.i.eq(1)) & (ecs.j.eq(7)), "cn"] = 10000.0
    ecs.loc[(ecs.i.eq(2)) & (ecs.j.eq(8)), "cn"] = 9999.0
    precision, pairs, comparison = compute_contact_precision(ecs, mono, tetra, wt)
    assert set(precision.structure_kind) == {"monomer", "tetramer"}
    assert comparison.loc[comparison.top_set.eq("L/2"), "tetramer_only_hits"].iloc[0] >= 1
    assert {"position_i_1b", "index_i_0b", "position_j_1b", "index_j_0b"}.issubset(
        pairs.columns
    )
