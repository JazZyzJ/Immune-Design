"""TDD for the shared refold-cache normalizer.

Cache-read backends (esmfold2, protenix) emit native mmCIF + a mean confidence on
a backend-specific scale. The normalizer converts that to the ONE cache format all
backends share: a single-chain ``<cache_dir>/<key>.pdb`` + a ``<cache_dir>/<key>.plddt``
sidecar holding mean pLDDT on the 0-100 scale (the esmfold sidecar convention).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb

from inverse_folding.evaluation.refold_normalize import (
    normalize_plddt,
    normalize_to_cache,
)


def _build_cif(
    path: Path,
    chains=("A",),
    n_res: int = 4,
    *,
    b_factor: float | None = None,
) -> None:
    """Write a tiny CA-only mmCIF with the given chains (fixture)."""
    atoms = []
    for ch in chains:
        for i in range(n_res):
            atoms.append(
                struc.Atom(
                    [float(i) * 3.8, float(ord(ch)), 0.0],
                    chain_id=ch,
                    res_id=i + 1,
                    res_name="GLY",
                    atom_name="CA",
                    element="C",
                )
            )
    arr = struc.array(atoms)
    if b_factor is not None:
        arr.set_annotation("b_factor", np.full(arr.array_length(), b_factor, dtype=float))
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, arr)
    cif.write(str(path))


def test_normalize_plddt_scales_unit_to_percent():
    assert normalize_plddt(0.90, "0-1") == pytest.approx(90.0)
    assert normalize_plddt(0.945, "0-1") == pytest.approx(94.5)


def test_normalize_plddt_leaves_percent_scale():
    assert normalize_plddt(87.5, "0-100") == pytest.approx(87.5)


def test_normalize_plddt_rejects_unknown_scale():
    with pytest.raises(ValueError):
        normalize_plddt(0.9, "bogus")


def test_normalize_to_cache_writes_single_chain_pdb_and_plddt(tmp_path):
    cif = tmp_path / "src.cif"
    _build_cif(cif, chains=("A",), n_res=5, b_factor=87.5)
    key = "p1_abc123def456"

    out = normalize_to_cache(
        str(tmp_path), key, cif_path=str(cif), mean_plddt=0.90, native_scale="0-1"
    )

    pdb_path = Path(out["pdb_path"])
    plddt_path = Path(out["plddt_path"])
    assert pdb_path == tmp_path / f"{key}.pdb"
    assert plddt_path == tmp_path / f"{key}.plddt"
    assert pdb_path.is_file() and plddt_path.is_file()

    # plddt sidecar normalized to 0-100
    assert float(plddt_path.read_text().strip()) == pytest.approx(90.0)

    # the emitted PDB is single-chain and parseable
    arr = pdb.PDBFile.read(str(pdb_path)).get_structure(
        model=1, extra_fields=["b_factor"]
    )
    assert set(np.unique(arr.chain_id)) == {"A"}
    assert np.asarray(arr.b_factor) == pytest.approx([87.5] * arr.array_length())


def test_normalize_to_cache_rejects_multichain(tmp_path):
    cif = tmp_path / "multi.cif"
    _build_cif(cif, chains=("A", "B"), n_res=3)
    with pytest.raises(RuntimeError) as exc:
        normalize_to_cache(
            str(tmp_path), "p2_deadbeef0000", cif_path=str(cif), mean_plddt=88.0, native_scale="0-100"
        )
    assert "chain" in str(exc.value).lower()


def test_normalize_to_cache_missing_cif_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        normalize_to_cache(
            str(tmp_path), "p3_0000", cif_path=str(tmp_path / "nope.cif"),
            mean_plddt=0.9, native_scale="0-1",
        )
