"""Phase B4 evaluator contract tests."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_phase_c import (
    aggregate_nmp_scores_with_threshold,
    build_manifest,
    compute_recovery,
    evaluate_immunogenicity_rows,
    evaluate_structural_rows,
    resolve_nmp_runtime_params,
)


_AA3 = {
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


def _write_ca_pdb(path: Path, sequence: str, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    lines = []
    dx, dy, dz = offset
    for idx, aa in enumerate(sequence, start=1):
        x = (idx - 1) * 1.6 + dx
        y = ((idx - 1) % 2) * 0.7 + dy
        z = ((idx - 1) % 3) * 0.4 + dz
        lines.append(
            f"ATOM  {idx:5d}  CA  {_AA3[aa]:>3} A{idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines))


class _FakePredictor:
    def predict_protein(self, sequence: str):
        hotspot = [float(idx) for idx, _ in enumerate(sequence, start=1)]
        return {
            "global_risk": float(len(sequence)) / 10.0,
            "residue_hotspot": pd.Series(hotspot, dtype=float).to_numpy(),
        }


class _FakeRunner:
    def score_batch(self, entries, allele, pep_lengths):
        del allele, pep_lengths
        out = {}
        for pid, seq in entries:
            scores = []
            for idx in range(max(1, len(seq) - 2)):
                rank = 0.01 if idx == 0 else 0.03
                scores.append(
                    SimpleNamespace(
                        peptide=seq[idx: idx + 3],
                        el_rank=rank,
                        pos=idx,
                        core="AAA",
                        el_score=1.0,
                    )
                )
            out[pid] = {12: scores}
        return out


def _generated_fixture() -> pd.DataFrame:
    rows = []
    for protein_id, seqs in {
        "p1": ["AAAA", "AAAT"],
        "p2": ["CCCCC", "CCCCA"],
        "p3": ["GGGGGG", "GGGGGA"],
    }.items():
        for design_idx, sequence in enumerate(seqs):
            rows.append(
                {
                    "protein_id": protein_id,
                    "design_idx": design_idx,
                    "design_id": f"design_{design_idx:04d}",
                    "sequence": sequence,
                    "seed": 42,
                    "wall_seconds": 0.1,
                }
            )
    return pd.DataFrame(rows)


def _test_lookup_fixture(tmp_path: Path) -> dict[str, dict]:
    lookup = {}
    for protein_id, sequence in {
        "p1": "AAAA",
        "p2": "CCCCC",
        "p3": "GGGGGG",
    }.items():
        pdb_path = tmp_path / f"{protein_id}.pdb"
        _write_ca_pdb(pdb_path, sequence)
        lookup[protein_id] = {
            "protein_id": protein_id,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "pdb_path": pdb_path.name,
        }
    return lookup


def test_evaluate_immunogenicity_rows_matches_predictor_and_nmp_counts():
    generated = _generated_fixture()
    head_df, nmp_df, failures = evaluate_immunogenicity_rows(
        generated,
        predictor=_FakePredictor(),
        nmp_runner=_FakeRunner(),
        allele="HLA-DRB1*07:01",
        strong_binder_threshold=2.0,
        nmp_batch_size=2,
        progress_every=0,
    )
    assert failures == []
    assert len(head_df) == 6
    assert len(nmp_df) == 6
    assert head_df.iloc[0]["global_risk"] == pytest.approx(0.4)
    assert nmp_df.iloc[0]["n_strong_binders"] == 1


def test_resolve_nmp_runtime_params_original_matches_iedb_baseline():
    args = SimpleNamespace(
        nmp_mode="original",
        nmp_batch_size=8,
        nmp_max_lengths_per_call=4,
        nmp_workers=8,
    )

    assert resolve_nmp_runtime_params(args) == (1, 14, 1)


def test_resolve_nmp_runtime_params_accelerated_uses_explicit_knobs():
    args = SimpleNamespace(
        nmp_mode="accelerated",
        nmp_batch_size=6,
        nmp_max_lengths_per_call=3,
        nmp_workers=5,
    )

    assert resolve_nmp_runtime_params(args) == (6, 3, 5)


def test_evaluate_structural_rows_emits_expected_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    generated = _generated_fixture()
    test_lookup = _test_lookup_fixture(tmp_path)

    def _fake_refold(sequence, protein_id, design_id, backend, cache_dir=None, model=None):
        pdb_path = tmp_path / f"{protein_id}_{design_id}.pdb"
        _write_ca_pdb(pdb_path, sequence, offset=(10.0, -2.0, 5.0))
        return {
            "pdb_path": str(pdb_path),
            "pLDDT": 88.0,
        }

    monkeypatch.setattr(
        "scripts.evaluate_phase_c.load_refold_model",
        lambda backend, device="cuda": object(),
    )
    monkeypatch.setattr(
        "scripts.evaluate_phase_c.refold",
        _fake_refold,
    )
    monkeypatch.setattr(
        "inverse_folding.evaluation.tmalign.run_tmalign",
        lambda pred_pdb, ref_pdb, tmalign_bin="TMalign", cache_dir=None: {
            "tm_score": 0.75,
            "rmsd": 1.25,
        },
    )
    monkeypatch.setattr(
        "inverse_folding.reference_flow.runtime.resolve_structure_path",
        lambda entry, pdb_root: Path(pdb_root) / str(entry["pdb_path"]),
    )

    df, failures, residues_df = evaluate_structural_rows(
        generated,
        test_lookup,
        pdb_root=tmp_path,
        refold_backend="esmfold",
        device="cuda",
        tmalign_bin="TMalign",
        esmfold_cache_dir=str(tmp_path / ".cache"),
        progress_every=0,
        return_residue_metrics=True,
    )
    assert failures == []
    assert len(df) == 6
    assert set(["protein_id", "design_id", "design_idx", "sequence", "scTM", "pLDDT", "bb_RMSD", "scRMSD", "recovery", "foldability", "refold_backend"]).issubset(df.columns)
    assert (df["refold_backend"] == "esmfold").all()
    assert df.loc[(df["protein_id"] == "p1") & (df["design_idx"] == 1), "recovery"].iloc[0] == pytest.approx(0.75)
    assert df["scRMSD"].max() < 1e-5
    assert len(residues_df) == sum(len(seq) for seq in generated["sequence"])
    assert set(
        [
            "protein_id",
            "design_id",
            "design_idx",
            "residue_idx",
            "residue_idx_1based",
            "ref_aa",
            "design_aa",
            "sc_ca_distance",
            "aligned_pred_ca_x",
        ]
    ).issubset(residues_df.columns)
    p1_mut = residues_df[
        (residues_df["protein_id"] == "p1")
        & (residues_df["design_idx"] == 1)
        & (residues_df["residue_idx"] == 3)
    ].iloc[0]
    assert p1_mut["residue_idx_1based"] == 4
    assert p1_mut["ref_aa"] == "A"
    assert p1_mut["design_aa"] == "T"
    assert residues_df["sc_ca_distance"].max() < 1e-5


def test_af3_refold_backend_raises():
    from inverse_folding.evaluation.refold import refold

    with pytest.raises(NotImplementedError, match="af3 backend"):
        refold("AAAA", "p1", "design_0000", backend="af3")


def test_build_manifest_populates_digests(tmp_path: Path):
    generated_path = tmp_path / "generated.parquet"
    generated_df = pd.DataFrame(
        [{"protein_id": "p1", "design_idx": 0, "sequence": "AAAA", "seed": 42, "wall_seconds": 0.1}]
    )
    generated_df.to_parquet(generated_path, index=False)
    test_set_path = tmp_path / "test.parquet"
    pd.DataFrame(
        [{"protein_id": "p1", "sequence": "AAAA", "sequence_length": 4, "pdb_path": "p1.pdb"}]
    ).to_parquet(test_set_path, index=False)
    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"checkpoint")
    nmp_bin = tmp_path / "netMHCIIpan"
    nmp_bin.write_text("#!/bin/sh\necho netMHCIIpan 4.3\n")
    nmp_bin.chmod(0o755)

    args = SimpleNamespace(
        generated_parquet=str(generated_path),
        test_set_parquet=str(test_set_path),
        epitope_ckpt=str(ckpt_path),
        netmhciipan_bin=str(nmp_bin),
        allele="HLA-DRB1*07:01",
        refold_model="esmfold",
    )
    manifest = build_manifest(
        args=args,
        run_id="eval_all_esmfold_DRB1_07_01_20260424T080000Z",
        modes_run=["imm", "struct"],
        rows_per_mode={"imm": 1, "struct": 1},
        wall_seconds_per_mode={"imm": 1.0, "struct": 2.0},
        generated_df=generated_df,
    )
    assert manifest["git_sha"]
    assert manifest["head_ckpt_digest"]
    assert manifest["generated_parquet_sha256"]


def test_aggregate_nmp_scores_with_custom_threshold():
    scores = pd.DataFrame(
        [
            {"rank_EL": 1.5},
            {"rank_EL": 2.5},
            {"rank_EL": 9.0},
        ]
    )
    agg = aggregate_nmp_scores_with_threshold(scores, strong_binder_threshold=2.0)
    assert agg["n_strong_binders"] == 1
    assert agg["n_weak_binders"] == 3
    assert math.isclose(agg["mean_best_rank"], (1.5 + 2.5 + 9.0) / 3.0)


def test_compute_recovery_matches_expected_identity():
    assert compute_recovery("AAAT", "AAAA") == pytest.approx(0.75)
