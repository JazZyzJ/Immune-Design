"""Contract tests for the multi-backend refold dispatcher.

Foundation for wiring cache-read structure-prediction backends (esmfold2,
protenix) alongside the in-process esmfold v1 backend. Cache-read backends are
produced by a separate SLURM pre-compute step in a foreign env and normalized
into the SAME on-disk cache (<cache_dir>/<cache_key>.pdb + <cache_key>.plddt,
pLDDT 0-100) that the esmfold runner already uses; at eval time load_refold_model
returns None and refold() is a pure cache read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inverse_folding.evaluation.esmfold_runner import cache_key
from inverse_folding.evaluation.refold import load_refold_model, refold

_DUMMY_PDB = (
    "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 87.50           C\n"
    "ATOM      2  CA  GLY A   2      13.800  10.000  10.000  1.00 87.50           C\n"
    "END\n"
)


def _prepopulate(cache_dir: Path, protein_id: str, sequence: str, plddt: float) -> str:
    key = cache_key(protein_id, sequence)
    (cache_dir / f"{key}.pdb").write_text(_DUMMY_PDB)
    (cache_dir / f"{key}.plddt").write_text(f"{plddt:.4f}")
    return key


@pytest.mark.parametrize("backend", ["esmfold2", "protenix", "af3"])
def test_load_refold_model_returns_none_for_cache_read_backends(backend):
    # cache-read backends run in a foreign env; the eval process never loads a live model.
    assert load_refold_model(backend, device="cpu") is None


@pytest.mark.parametrize("backend", ["esmfold2", "protenix", "af3"])
def test_cache_read_backend_reads_prepopulated_cache(tmp_path, backend):
    seq = "MKTAYIAKQR"
    pid = "p1"
    key = _prepopulate(tmp_path, pid, seq, 87.5)

    pred = refold(seq, pid, "design_0000", backend=backend, cache_dir=str(tmp_path), model=None)

    assert pred["pdb_path"] == str(tmp_path / f"{key}.pdb")
    assert Path(pred["pdb_path"]).is_file()
    assert pred["pLDDT"] == pytest.approx(87.5)  # normalized 0-100 scale
    assert pred.get("cache_hit") is True


@pytest.mark.parametrize("backend", ["esmfold2", "protenix", "af3"])
def test_cache_read_backend_missing_cache_fails_fast(tmp_path, backend):
    # A missing precomputed structure must fail-fast with a clear, non-OOM error
    # (returning None would stringify to 'None' into TMalign; an 'out of memory'
    # message would wrongly trigger the eval's cuda->cpu OOM retry path).
    with pytest.raises(RuntimeError) as exc:
        refold("MKTAYIAKQR", "p1", "design_0000", backend=backend, cache_dir=str(tmp_path), model=None)

    msg = str(exc.value)
    assert backend in msg
    assert "out of memory" not in msg.lower()


@pytest.mark.parametrize("backend", ["esmfold2", "protenix", "af3"])
def test_cache_read_backend_requires_cache_dir(backend):
    with pytest.raises(RuntimeError) as exc:
        refold("MKTAYIAKQR", "p1", "design_0000", backend=backend, cache_dir=None, model=None)
    assert "cache_dir" in str(exc.value)


def test_af3_is_cache_read_not_stub(tmp_path):
    # af3 (official DeepMind AlphaFold3) is now a cache-read backend like esmfold2/protenix.
    seq, pid = "MKTAYIAKQR", "p1"
    key = _prepopulate(tmp_path, pid, seq, 88.0)
    pred = refold(seq, pid, "design_0000", backend="af3", cache_dir=str(tmp_path), model=None)
    assert pred["pdb_path"] == str(tmp_path / f"{key}.pdb")
    assert pred["pLDDT"] == pytest.approx(88.0)
