"""TDD for the ESMFold2 refold-cache precompute helpers added to
scripts/predict_esmfold2_gt.py (reuse-first: extend the existing folder rather
than adding a new esmfold2 script).

  * build_fold_records_from_parquet: dedup unique (protein_id, sequence) from a
    generated parquet, header id == cache_key(protein_id, sequence) so on-disk
    names already equal the shared refold cache key.
  * cache_layout_from_manifest: after folding, normalize each ESMFold2 mmCIF into
    <cache_dir>/<key>.pdb + <key>.plddt (0-100), resume-safe over the manifest.

Both are pure (no torch/esm) so they are unit-tested in the immune-design env.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

from inverse_folding.evaluation.esmfold_runner import cache_key
from scripts.predict_esmfold2_gt import (
    build_fold_records_from_parquet,
    cache_layout_from_manifest,
)


def _build_cif(path: Path, n_res: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atoms = [
        struc.Atom([float(i) * 3.8, 0.0, 0.0], chain_id="A", res_id=i + 1,
                   res_name="GLY", atom_name="CA", element="C")
        for i in range(n_res)
    ]
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, struc.array(atoms))
    cif.write(str(path))


def test_build_fold_records_dedups_and_uses_cache_key_headers(tmp_path):
    df = pd.DataFrame(
        [
            {"protein_id": "p1", "design_idx": 0, "sequence": "AAAA"},
            {"protein_id": "p1", "design_idx": 1, "sequence": "AAAA"},  # exact dup -> one fold
            {"protein_id": "p1", "design_idx": 2, "sequence": "AAAT"},
            {"protein_id": "p2", "design_idx": 0, "sequence": "CCCC"},
        ]
    )
    pq = tmp_path / "generated.parquet"
    df.to_parquet(pq)

    recs = build_fold_records_from_parquet(str(pq))

    assert len(recs) == 3  # (p1,AAAA), (p1,AAAT), (p2,CCCC)
    rec_set = set(recs)
    assert (cache_key("p1", "AAAA"), "AAAA") in rec_set
    assert (cache_key("p1", "AAAT"), "AAAT") in rec_set
    assert (cache_key("p2", "CCCC"), "CCCC") in rec_set


def test_cache_layout_from_manifest_writes_0_100_entries(tmp_path):
    mmcif_dir = tmp_path / "mmcif"
    _build_cif(mmcif_dir / "safeid.cif")
    key = cache_key("p1", "AAAA")
    manifest = tmp_path / "gt_manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": key, "file": "safeid", "len": 4, "mean_plddt": 0.90, "ptm": 0.85}) + "\n"
    )
    cache_dir = tmp_path / "cache"

    n = cache_layout_from_manifest(str(manifest), str(mmcif_dir), str(cache_dir))

    assert n == 1
    assert (cache_dir / f"{key}.pdb").is_file()
    # ESMFold2 mean_plddt is 0-1 -> normalized to 0-100 in the sidecar
    assert float((cache_dir / f"{key}.plddt").read_text().strip()) == pytest.approx(90.0)
