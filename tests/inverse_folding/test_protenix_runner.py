"""TDD for the Protenix output -> shared refold-cache normalizer.

Protenix v2 writes, per fold `name`:
  <out>/<name>/seed_<seed>/predictions/<name>_sample_<rank>.cif   (mmCIF, B-factor pLDDT 0-100)
  <out>/<name>/seed_<seed>/predictions/<name>_summary_confidence_sample_<rank>.json
`_sample_0` is the best (max ranking_score) WITHIN a seed; across seeds the global best
is the seed whose _sample_0 has the max `ranking_score`. Summary `plddt` is mean pLDDT 0-100.
The runner finds the global best and normalizes it into <cache_dir>/<key>.pdb + <key>.plddt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb

from inverse_folding.evaluation.protenix_runner import (
    find_best_sample,
    normalize_protenix_to_cache,
)


def _write_cif(path: Path, n_res: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atoms = [
        struc.Atom([float(i) * 3.8, 0.0, 0.0], chain_id="A", res_id=i + 1,
                   res_name="GLY", atom_name="CA", element="C")
        for i in range(n_res)
    ]
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, struc.array(atoms))
    cif.write(str(path))


def _write_seed(out_dir: Path, name: str, seed: int, plddt: float, ranking: float) -> None:
    pred = out_dir / name / f"seed_{seed}" / "predictions"
    _write_cif(pred / f"{name}_sample_0.cif")
    (pred / f"{name}_summary_confidence_sample_0.json").write_text(
        json.dumps({"plddt": plddt, "ranking_score": ranking, "ptm": 0.7})
    )


def test_find_best_sample_picks_global_max_ranking_across_seeds(tmp_path):
    name = "p1_abc123def456"
    _write_seed(tmp_path, name, seed=101, plddt=80.0, ranking=0.55)
    _write_seed(tmp_path, name, seed=202, plddt=90.0, ranking=0.91)  # global best
    cif_path, summary_path = find_best_sample(str(tmp_path), name)
    assert "seed_202" in cif_path
    assert cif_path.endswith(f"{name}_sample_0.cif")
    assert Path(summary_path).is_file()


def test_normalize_protenix_writes_cache_entry_0_100(tmp_path):
    name = "p1_abc123def456"
    key = name
    _write_seed(tmp_path, name, seed=101, plddt=83.81, ranking=0.9)

    out = normalize_protenix_to_cache(str(tmp_path), name, cache_dir=str(tmp_path / "cache"), key=key)

    pdb_path = Path(out["pdb_path"])
    assert pdb_path == tmp_path / "cache" / f"{key}.pdb"
    assert pdb_path.is_file()
    # Protenix summary 'plddt' is already 0-100 -> unchanged
    assert float(Path(out["plddt_path"]).read_text().strip()) == pytest.approx(83.81)
    arr = pdb.PDBFile.read(str(pdb_path)).get_structure(model=1)
    assert set(str(c) for c in set(arr.chain_id)) == {"A"}


def test_find_best_sample_missing_output_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_best_sample(str(tmp_path), "nonexistent_fold")
