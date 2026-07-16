"""TDD for the AlphaFold3 output -> shared refold-cache normalizer.

AF3 writes, per fold `name` (sanitised_name preserves cache_key — alnum/_/-/. kept,
no lowercasing): <out>/<name>/<name>_model.cif (top-ranked best, mmCIF, B-factor pLDDT
0-100) + <name>_confidences.json (atom_plddts, 0-100). The runner reads the best model
+ mean(atom_plddts) and normalizes into <cache_dir>/<key>.pdb + <key>.plddt (0-100).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb

from inverse_folding.evaluation.af3_runner import (
    read_mean_plddt_af3,
    normalize_af3_to_cache,
)


def _write_cif(path: Path, n_res: int = 5, ca_plddt: float = 88.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Real AF3 model.cif carries per-atom pLDDT (0-100) in the B-factor column; the CA B-factors
    # are what the sidecar / v2 predicted_global_plddt aggregate, so set them explicitly.
    atoms = [
        struc.Atom([float(i) * 3.8, 0.0, 0.0], chain_id="A", res_id=i + 1,
                   res_name="GLY", atom_name="CA", element="C", b_factor=ca_plddt)
        for i in range(n_res)
    ]
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, struc.array(atoms))
    cif.write(str(path))


def _write_af3_fold(out_dir: Path, name: str, atom_plddts, ca_plddt: float = 88.0) -> None:
    d = out_dir / name
    _write_cif(d / f"{name}_model.cif", ca_plddt=ca_plddt)
    (d / f"{name}_confidences.json").write_text(json.dumps({"atom_plddts": list(atom_plddts)}))
    (d / f"{name}_summary_confidences.json").write_text(json.dumps({"ptm": 0.8, "ranking_score": 0.9}))


def test_read_mean_plddt_af3(tmp_path):
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"atom_plddts": [80.0, 90.0, 100.0]}))
    assert read_mean_plddt_af3(str(j)) == pytest.approx(90.0)


def test_normalize_af3_writes_cache_entry_0_100(tmp_path):
    name = "Q00511_abc123def456"  # cache_key keeps case; AF3 preserves it
    # CA (per-residue) pLDDT 92.0 differs from the all-atom atom_plddts mean 88.0. The sidecar must
    # report the CA mean (= evaluate_phase_c's v2 predicted_global_plddt), NOT the all-atom mean,
    # or _validate_v2_prediction_plddt fails closed at struct-eval time.
    _write_af3_fold(tmp_path, name, atom_plddts=[88.0] * 10, ca_plddt=92.0)

    out = normalize_af3_to_cache(str(tmp_path), name, cache_dir=str(tmp_path / "cache"), key=name)

    pdb_path = Path(out["pdb_path"])
    assert pdb_path == tmp_path / "cache" / f"{name}.pdb"
    assert pdb_path.is_file()
    # sidecar = CA-atom B-factor mean (0-100), not the all-atom atom_plddts mean
    assert float(Path(out["plddt_path"]).read_text().strip()) == pytest.approx(92.0)
    arr = pdb.PDBFile.read(str(pdb_path)).get_structure(model=1)
    assert set(str(c) for c in set(arr.chain_id)) == {"A"}


def test_normalize_af3_missing_output_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        normalize_af3_to_cache(str(tmp_path), "nonexistent", cache_dir=str(tmp_path / "c"), key="nonexistent")
