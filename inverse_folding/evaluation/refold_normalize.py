"""Normalize a cache-read backend's native output into the shared refold cache.

Cache-read backends (esmfold2, protenix) emit native mmCIF + a mean confidence on
a backend-specific scale. This module converts that to the ONE format every refold
backend shares, so downstream scTM/sc_rmsd/foldability are backend-agnostic:

  * a single-chain ``<cache_dir>/<key>.pdb`` (biotite mmCIF -> PDB; multi-chain is
    a fail-fast error — a monomer refold must be one chain), and
  * a ``<cache_dir>/<key>.plddt`` sidecar holding mean pLDDT on the **0-100** scale
    (the esmfold B-factor convention; esmfold2's native 0-1 is scaled here).

The cache key is ``inverse_folding.evaluation.esmfold_runner.cache_key(protein_id,
sequence)`` — the same function the eval-time cache read uses, so keys cannot drift.
"""

from __future__ import annotations

import io
import os

import numpy as np


def normalize_plddt(value: float, native_scale: str) -> float:
    """Return mean pLDDT on the 0-100 scale.

    native_scale: ``"0-1"`` (ESMFold2 / AlphaFold-style unit interval -> x100) or
    ``"0-100"`` (ESMFold / AF3 atom_plddts B-factor scale -> unchanged).
    """
    if native_scale == "0-1":
        return float(value) * 100.0
    if native_scale == "0-100":
        return float(value)
    raise ValueError(f"unknown pLDDT native_scale: {native_scale!r} (expected '0-1' or '0-100')")


def cif_to_single_chain_pdb_text(cif_path: str) -> str:
    """Convert an mmCIF file to single-chain PDB text (biotite); fail-fast on multi-chain."""
    import biotite.structure.io.pdb as pdb
    import biotite.structure.io.pdbx as pdbx

    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"refold normalize: mmCIF not found: {cif_path}")
    cif = pdbx.CIFFile.read(cif_path)
    arr = pdbx.get_structure(cif, model=1)
    chains = sorted(str(c) for c in np.unique(arr.chain_id))
    if len(chains) != 1:
        raise RuntimeError(
            f"refold normalize: expected a single-chain monomer, got chains {chains} "
            f"in {cif_path}"
        )
    pdb_file = pdb.PDBFile()
    pdb.set_structure(pdb_file, arr)
    sink = io.StringIO()
    pdb_file.write(sink)
    return sink.getvalue()


def fold_records_from_parquet(parquet_path: str) -> list[tuple[str, str]]:
    """Unique ``(cache_key, sequence)`` fold records from a generated parquet.

    Single source of truth for refold-cache key enumeration used by BOTH precompute
    drivers (esmfold2, protenix): dedups unique ``(protein_id, sequence)`` and sets the
    fold id to ``cache_key(protein_id, sequence)`` so precompute output names cannot
    drift from the eval-time cache-read key.
    """
    import pandas as pd

    from inverse_folding.evaluation.esmfold_runner import cache_key

    df = pd.read_parquet(parquet_path)
    for col in ("protein_id", "sequence"):
        if col not in df.columns:
            raise ValueError(f"'{col}' not in {parquet_path}")
    seen: set[tuple[str, str]] = set()
    recs: list[tuple[str, str]] = []
    for pid, seq in zip(df["protein_id"].astype(str), df["sequence"].astype(str)):
        pair = (pid, seq)
        if pair in seen:
            continue
        seen.add(pair)
        recs.append((cache_key(pid, seq), seq))
    return recs


def normalize_to_cache(
    cache_dir: str,
    key: str,
    *,
    cif_path: str,
    mean_plddt: float,
    native_scale: str,
) -> dict[str, str]:
    """Write ``<cache_dir>/<key>.pdb`` (single-chain) + ``<key>.plddt`` (0-100).

    Returns ``{"pdb_path": ..., "plddt_path": ...}``. Fail-fast on a missing/multi-chain
    mmCIF or an unknown scale (no placeholder — the eval-time read is load-bearing).
    """
    os.makedirs(cache_dir, exist_ok=True)
    plddt_100 = normalize_plddt(mean_plddt, native_scale)  # validate scale before touching disk
    pdb_text = cif_to_single_chain_pdb_text(cif_path)

    pdb_path = os.path.join(cache_dir, f"{key}.pdb")
    plddt_path = os.path.join(cache_dir, f"{key}.plddt")
    with open(pdb_path, "w") as handle:
        handle.write(pdb_text)
    with open(plddt_path, "w") as handle:
        handle.write(f"{plddt_100:.4f}")
    return {"pdb_path": pdb_path, "plddt_path": plddt_path}
