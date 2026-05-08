"""Phase C1 script helper tests."""

from __future__ import annotations

import pandas as pd
import pytest

from inverse_folding.evaluation.h_maps import sequence_md5
from scripts.run_if_phase_c1 import (
    assert_h_maps_align_with_test_set,
    derive_h_shuffle_seed,
)


def test_derive_h_shuffle_seed_varies_by_protein_and_design():
    base_seed = 123
    assert derive_h_shuffle_seed(base_seed, "p1", 0) == derive_h_shuffle_seed(base_seed, "p1", 0)
    assert derive_h_shuffle_seed(base_seed, "p1", 0) != derive_h_shuffle_seed(base_seed, "p1", 1)
    assert derive_h_shuffle_seed(base_seed, "p1", 0) != derive_h_shuffle_seed(base_seed, "p2", 0)


def _entries(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protein_id": pid,
                "sequence": seq,
                "sequence_length": len(seq),
                "pdb_path": f"{pid}.pdb",
            }
            for pid, seq in rows
        ]
    )


def _h_maps(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"protein_id": pid, "sequence_md5": md5, "sequence_length": 1, "h_raw": [0.0]}
            for pid, md5 in rows
        ]
    )


def test_alignment_passes_when_md5_matches():
    entries = _entries([("p1", "ACDE"), ("p2", "FGHIK")])
    h_maps = _h_maps(
        [
            ("p1", sequence_md5("ACDE")),
            ("p2", sequence_md5("FGHIK")),
        ]
    )
    assert_h_maps_align_with_test_set(entries=entries, h_maps_df=h_maps)


def test_alignment_fails_on_same_length_different_residues():
    entries = _entries([("p1", "ACDE")])
    h_maps = _h_maps([("p1", sequence_md5("DCBA"))])
    with pytest.raises(ValueError, match="sequence_md5"):
        assert_h_maps_align_with_test_set(entries=entries, h_maps_df=h_maps)


def test_alignment_fails_when_h_maps_missing_protein():
    entries = _entries([("p1", "ACDE"), ("p2", "FGHIK")])
    h_maps = _h_maps([("p1", sequence_md5("ACDE"))])
    with pytest.raises(ValueError, match="missing protein_id"):
        assert_h_maps_align_with_test_set(entries=entries, h_maps_df=h_maps)


def test_alignment_passes_with_extra_h_maps_proteins():
    """Extra proteins in h_maps that aren't in the test set are tolerated.

    A larger h-map corpus is fine; what matters is that every test-set
    protein has a matching, byte-identical row in h_maps.
    """
    entries = _entries([("p1", "ACDE")])
    h_maps = _h_maps(
        [
            ("p1", sequence_md5("ACDE")),
            ("extra", sequence_md5("ZZZZZZ")),
        ]
    )
    assert_h_maps_align_with_test_set(entries=entries, h_maps_df=h_maps)
