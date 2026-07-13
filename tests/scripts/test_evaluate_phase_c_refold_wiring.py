"""Wiring tests: --refold-model choices for the new cache-read backends, the
generic --refold-cache-dir alias, and an end-to-end evaluate_structural_rows run
against a PRE-POPULATED cache (the eval-time cache-read path, no live model)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_phase_c import evaluate_structural_rows, parse_args
from inverse_folding.evaluation.esmfold_runner import cache_key

_BASE_ARGV = [
    "--generated-parquet", "g.parquet",
    "--test-set-parquet", "t.parquet",
    "--allele", "HLA-DRB1*07:01",
    "--mode", "struct",
    "--output-root", "out",
]


def test_structural_defaults_promote_esmfold2_v2():
    args = parse_args(_BASE_ARGV)
    assert args.refold_model == "esmfold2"
    assert args.structural_metrics_v2 is True


@pytest.mark.parametrize("backend", ["esmfold", "esmfold2", "protenix"])
def test_parse_args_accepts_refold_backends(backend):
    args = parse_args(_BASE_ARGV + ["--refold-model", backend])
    assert args.refold_model == backend


def test_refold_cache_dir_aliases_esmfold_cache_dir():
    args = parse_args(_BASE_ARGV + ["--refold-cache-dir", "/tmp/rc"])
    assert args.esmfold_cache_dir == "/tmp/rc"


def test_parse_args_accepts_standalone_structural_v2_inputs():
    args = parse_args(
        _BASE_ARGV
        + ["--structural-metrics-v2", "--constraint-manifest", "anchors.yaml"]
    )
    assert args.structural_metrics_v2 is True
    assert args.constraint_manifest == "anchors.yaml"


# ── end-to-end cache-read through evaluate_structural_rows ──────────────────
_AA3 = {"A": "ALA", "C": "CYS", "G": "GLY", "T": "THR"}


def _write_ca_pdb(path: Path, sequence: str, plddt: float = 77.0) -> None:
    lines = []
    for idx, aa in enumerate(sequence, start=1):
        x, y, z = (idx - 1) * 1.6, ((idx - 1) % 2) * 0.7, ((idx - 1) % 3) * 0.4
        lines.append(
            f"ATOM  {idx:5d}  CA  {_AA3[aa]:>3} A{idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt:6.2f}           C\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines))


def test_evaluate_structural_rows_cache_read_backend(tmp_path, monkeypatch):
    backend = "protenix"
    generated = pd.DataFrame(
        [
            {"protein_id": "p1", "design_idx": 0, "design_id": "design_0000",
             "sequence": "AAAA", "seed": 42, "wall_seconds": 0.1},
            {"protein_id": "p1", "design_idx": 1, "design_id": "design_0001",
             "sequence": "AAAT", "seed": 42, "wall_seconds": 0.1},
        ]
    )
    ref_pdb = tmp_path / "p1.pdb"
    _write_ca_pdb(ref_pdb, "AAAA")
    test_lookup = {"p1": {"protein_id": "p1", "sequence": "AAAA", "sequence_length": 4,
                          "pdb_path": ref_pdb.name}}

    # Pre-populate the refold cache (what the SLURM precompute step would produce).
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for seq in ("AAAA", "AAAT"):
        key = cache_key("p1", seq)
        _write_ca_pdb(cache_dir / f"{key}.pdb", seq, plddt=91.0)
        (cache_dir / f"{key}.plddt").write_text("91.0000")

    monkeypatch.setattr(
        "inverse_folding.evaluation.tmalign.run_tmalign",
        lambda pred_pdb, ref_pdb, tmalign_bin="TMalign", cache_dir=None: {"tm_score": 0.72, "rmsd": 1.4},
    )
    monkeypatch.setattr(
        "inverse_folding.reference_flow.runtime.resolve_structure_path",
        lambda entry, pdb_root: Path(pdb_root) / str(entry["pdb_path"]),
    )

    df, failures, _ = evaluate_structural_rows(
        generated, test_lookup, pdb_root=tmp_path,
        refold_backend=backend, device="cpu", tmalign_bin="TMalign",
        esmfold_cache_dir=str(cache_dir), progress_every=0, return_residue_metrics=True,
    )

    assert failures == []
    assert len(df) == 2
    assert (df["refold_backend"] == backend).all()
    assert (df["pLDDT"] == 91.0).all()  # read straight from the cache sidecar (0-100)
    assert (df["scTM"] == 0.72).all()
