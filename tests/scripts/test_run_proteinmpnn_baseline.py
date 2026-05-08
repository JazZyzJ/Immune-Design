"""Unit tests for scripts/run_proteinmpnn_baseline.py.

Covers the pieces the script owns directly:
  * FASTA parser for ProteinMPNN ``seqs/<pid>.fa`` output.
  * NMP per-protein selection rule (mocked NMP scores).
  * CLI validation gates for ``--apply-nmp-filter``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_proteinmpnn_baseline.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_proteinmpnn_baseline", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_proteinmpnn_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script_mod():
    return _load_script_module()


def _write_mpnn_fa(path: Path) -> None:
    """Mimic the layout produced by ProteinMPNN's seqs/<pid>.fa."""
    path.write_text(
        # WT/input record (no `sample=` key)
        ">3HTN, score=1.1705, global_score=1.2045, fixed_chains=['B'], "
        "designed_chains=['A', 'C'], model_name=v_48_020, git_hash=abc, seed=37\n"
        "AAAA/BBBB\n"
        # Design 0 at T=0.1
        ">T=0.1, sample=1, score=0.7291, global_score=0.9330, seq_recovery=0.5736\n"
        "CCCC/DDDD\n"
        # Design 1 at T=0.1
        ">T=0.1, sample=2, score=0.8000, global_score=0.9500, seq_recovery=0.6000\n"
        "EEEE/FFFF\n"
    )


def test_parse_mpnn_seqs_skips_wt_and_extracts_designs(tmp_path, script_mod):
    seqs_dir = tmp_path / "seqs"
    seqs_dir.mkdir()
    _write_mpnn_fa(seqs_dir / "3HTN.fa")

    out = script_mod.parse_mpnn_seqs(seqs_dir)

    assert set(out.keys()) == {"3HTN"}
    designs = out["3HTN"]
    assert len(designs) == 2

    d0, d1 = designs
    assert d0["protein_id"] == "3HTN"
    assert d0["design_idx"] == 0
    assert d0["sequence"] == "CCCC/DDDD"
    assert d0["mpnn_score"] == pytest.approx(0.7291)
    assert d0["seq_recovery"] == pytest.approx(0.5736)
    assert d0["mpnn_temperature"] == pytest.approx(0.1)
    assert d0["mpnn_sample"] == 1

    assert d1["design_idx"] == 1
    assert d1["sequence"] == "EEEE/FFFF"
    assert d1["mpnn_sample"] == 2


def test_designed_chain_seqs_splits_and_drops_empty(script_mod):
    assert script_mod._designed_chain_seqs("AAAA/BBBB") == ["AAAA", "BBBB"]
    assert script_mod._designed_chain_seqs("XYZ") == ["XYZ"]
    assert script_mod._designed_chain_seqs("AAA//BBB/") == ["AAA", "BBB"]


def test_select_per_protein_min_n_sb_uses_mean_rank_tiebreak(script_mod):
    annotated = {
        "P1": [
            {"design_idx": 0, "sequence": "AAA", "nmp_n_sb": 3, "nmp_mean_rank": 0.05, "nmp_n_windows": 30},
            {"design_idx": 1, "sequence": "BBB", "nmp_n_sb": 1, "nmp_mean_rank": 0.10, "nmp_n_windows": 30},
            # Tied n_sb with idx=1, but worse mean_rank → loses.
            {"design_idx": 2, "sequence": "CCC", "nmp_n_sb": 1, "nmp_mean_rank": 0.20, "nmp_n_windows": 30},
        ]
    }
    sel = script_mod.select_per_protein(annotated, rule="min_n_sb")
    assert len(sel) == 1
    assert sel[0]["protein_id"] == "P1"
    assert sel[0]["selected_design_idx"] == 1
    assert sel[0]["nmp_n_sb"] == 1
    assert sel[0]["sequence"] == "BBB"
    assert sel[0]["rule"] == "min_n_sb"


def test_select_per_protein_min_mean_rank(script_mod):
    annotated = {
        "P1": [
            {"design_idx": 0, "sequence": "AAA", "nmp_n_sb": 5, "nmp_mean_rank": 0.04, "nmp_n_windows": 30},
            {"design_idx": 1, "sequence": "BBB", "nmp_n_sb": 1, "nmp_mean_rank": 0.10, "nmp_n_windows": 30},
        ]
    }
    sel = script_mod.select_per_protein(annotated, rule="min_mean_rank")
    assert sel[0]["selected_design_idx"] == 0  # lowest mean_rank wins


def test_apply_nmp_filter_requires_binary_and_allele(script_mod):
    base = [
        "--input-pdb-folder", "/tmp/pdbs",
        "--output-dir", "/tmp/out",
        "--apply-nmp-filter",
    ]
    with pytest.raises(SystemExit):
        script_mod.parse_args(base)
    with pytest.raises(SystemExit):
        script_mod.parse_args(base + ["--allele", "HLA-DRB1*07:01"])


def test_parse_args_default_no_nmp_filter(script_mod, tmp_path):
    pdb_dir = tmp_path / "in"
    pdb_dir.mkdir()
    out_dir = tmp_path / "out"
    args = script_mod.parse_args([
        "--input-pdb-folder", str(pdb_dir),
        "--output-dir", str(out_dir),
        "--num-seq-per-target", "4",
    ])
    assert args.num_seq_per_target == 4
    assert args.apply_nmp_filter is False
    assert args.netmhciipan_bin is None
    assert args.nmp_pep_lengths == "15"
    assert args.nmp_rank_threshold == 2.0


def test_help_runs_cleanly():
    """argparse %% escaping regression — ensure --help formats without errors."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )
    assert result.returncode == 0, result.stderr
    assert "ProteinMPNN inverse-folding baseline" in result.stdout
    assert "--apply-nmp-filter" in result.stdout
