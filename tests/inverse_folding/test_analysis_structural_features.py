"""Regression tests for `inverse_folding.analysis.structural_features`.

The span ↔ sequence coordinate contract was mis-implemented initially (end_0b
treated as inclusive, offset missing the chain_range-start 1-based→0-based
adjustment). This suite locks the fix so both invariants are preserved
together:

  1. [start_0b, end_0b) is half-open and end_0b == start_0b + len(peptide).
  2. Absolute → seq_idx offset is (chain_range_start - 1).

`test_build_is_epitope_1LI1_C_fixture` covers a concrete real-data example.
`test_build_is_epitope_tier1_json_sweep` is the broad safety net: it iterates
every span in the shipped tier1_candidates.json and fails on the first silent
mismatch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inverse_folding.analysis.structural_features import (
    AlignmentError,
    AtomResidue,
    SpanSanityError,
    align_atom_to_fasta,
    build_is_epitope,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER1_JSON = REPO_ROOT / "outputs" / "if" / "test_set" / "tier1_candidates.json"


# ────────────────────────────────────────────────────────────────────────
# Minimal synthetic fixture (independent of external data)
# ────────────────────────────────────────────────────────────────────────

def test_build_is_epitope_synthetic_half_open():
    """Synthetic 10-residue chain at chain_range 101-110.

    Peptide "KVL" occupies seq_idx [2, 5). With chain_range_start=101, the
    absolute coordinates are start_0b=102, end_0b=105 → end_0b-start_0b=3=len.
    """
    fasta = "MKKVLQAGKT"
    spans = [{"start_0b": 102, "end_0b": 105, "peptide": "KVL"}]
    labels = build_is_epitope(spans, fasta, chain_range_start=101)

    expected = np.zeros(10, dtype=np.int8)
    expected[2:5] = 1
    np.testing.assert_array_equal(labels, expected)


def test_build_is_epitope_rejects_inclusive_end_span():
    """A span where end_0b was (wrongly) written as inclusive must abort."""
    fasta = "MKKVLQAGKT"
    # If KVL were treated inclusive, end_0b would be 104 and len(peptide)=3
    # does NOT equal end-start=2 → slice mismatch → SpanSanityError.
    spans = [{"start_0b": 102, "end_0b": 104, "peptide": "KVL"}]
    with pytest.raises(SpanSanityError, match="peptide="):
        build_is_epitope(spans, fasta, chain_range_start=101)


def test_build_is_epitope_allows_end_equal_to_seq_len():
    """Half-open end can hit the sentinel `end == len(fasta)` without tripping
    the range check."""
    fasta = "MKKVL"  # seq_idx 0..4; chain_range_start=101 ⇒ abs_offset=100
    # whole chain is the epitope: absolute [100, 105) ⇒ seq [0, 5)
    spans = [{"start_0b": 100, "end_0b": 105, "peptide": "MKKVL"}]
    labels = build_is_epitope(spans, fasta, chain_range_start=101)
    np.testing.assert_array_equal(labels, np.ones(5, dtype=np.int8))


# ────────────────────────────────────────────────────────────────────────
# Real Tier 1 data fixture — 1LI1_C
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def _tier1_entries():
    if not TIER1_JSON.is_file():
        pytest.skip(f"Tier 1 JSON missing at {TIER1_JSON}")
    with TIER1_JSON.open() as f:
        return json.load(f)


def _entry(entries, pid):
    for e in entries:
        if e["protein_id"] == pid:
            return e
    raise KeyError(pid)


def test_build_is_epitope_1LI1_C_fixture(_tier1_entries):
    """Known real entry — 1LI1_C has two EL spans at absolute coordinates
    (1560, 1575) → 'NDKSYWLSTTAPLPM' and (1537, 1547) → 'LARFSTMPFL'.
    Chain starts at 1485 (1-based), so seq_idx = abs - 1484.
    """
    e = _entry(_tier1_entries, "1LI1_C")
    chain_start = int(e["chain_range"].split("-")[0])
    assert chain_start == 1485

    labels = build_is_epitope(e["experimental_epitopes"], e["sequence"], chain_start)

    # Expected slices (sanity re-derived here to anchor the test to data):
    for sp in e["experimental_epitopes"]:
        start = sp["start_0b"] - (chain_start - 1)
        end = sp["end_0b"] - (chain_start - 1)
        assert e["sequence"][start:end] == sp["peptide"], (
            f"fixture invariant broken for {sp}"
        )
        assert labels[start:end].all() == 1
        assert end - start == len(sp["peptide"])  # half-open check

    # Nothing outside the two spans should be flagged.
    expected = np.zeros(len(e["sequence"]), dtype=np.int8)
    for sp in e["experimental_epitopes"]:
        s = sp["start_0b"] - (chain_start - 1)
        ee = sp["end_0b"] - (chain_start - 1)
        expected[s:ee] = 1
    np.testing.assert_array_equal(labels, expected)


def test_build_is_epitope_tier1_json_sweep(_tier1_entries):
    """Every span in the shipped JSON must round-trip without SpanSanityError.

    This is the broad regression net: if any future edit re-breaks the
    offset or the half-open convention, at least one of the 261 spans
    across 15 proteins will fail.
    """
    failing = []
    for e in _tier1_entries:
        chain_start = int(e["chain_range"].split("-")[0])
        try:
            build_is_epitope(e["experimental_epitopes"], e["sequence"], chain_start)
        except SpanSanityError as err:
            failing.append((e["protein_id"], str(err)))

    assert not failing, (
        f"{len(failing)}/{len(_tier1_entries)} Tier 1 entries failed span sanity:\n"
        + "\n".join(f"  {pid}: {msg.splitlines()[0]}" for pid, msg in failing[:5])
    )


def test_build_is_epitope_tier1_half_open_invariant(_tier1_entries):
    """Independent check that every span obeys `end_0b - start_0b == len(peptide)`."""
    bad = []
    for e in _tier1_entries:
        for sp in e["experimental_epitopes"]:
            if sp["end_0b"] - sp["start_0b"] != len(sp["peptide"]):
                bad.append((e["protein_id"], sp))
    assert not bad, (
        f"{len(bad)} spans break the half-open invariant; first few: {bad[:3]}"
    )


def _make_atom_residues(seq: str, start_resnum: int = 1):
    residues = []
    for i, aa in enumerate(seq):
        residues.append(
            AtomResidue(
                author_resnum=start_resnum + i,
                ins_code=" ",
                aa1=aa,
                ca_coord=np.array([float(i), 0.0, 0.0], dtype=np.float64),
                ca_bfactor=10.0 + i,
            )
        )
    return residues


def test_align_atom_to_fasta_accepts_contiguous_offset_fallback():
    fasta = "SVSIGYLLVKHSQ"
    atom_residues = _make_atom_residues("GYLLVKHSQ", start_resnum=5)

    seq_to_author, seq_to_atom = align_atom_to_fasta(
        atom_residues, fasta_seq=fasta, chain_range_start=1485
    )

    assert seq_to_author[:4] == [None, None, None, None]
    assert seq_to_author[4:] == list(range(5, 14))
    assert [r.aa1 if r is not None else None for r in seq_to_atom] == [
        None, None, None, None, "G", "Y", "L", "L", "V", "K", "H", "S", "Q"
    ]


def test_align_atom_to_fasta_accepts_high_identity_alignment_fallback():
    fasta = "RPPLPNQQFGVSLQHLQEKN"
    atom_residues = _make_atom_residues("PLPNQQFGVSLQHLQEKN", start_resnum=39)

    seq_to_author, seq_to_atom = align_atom_to_fasta(
        atom_residues, fasta_seq=fasta, chain_range_start=234
    )

    assert seq_to_author[0] is None
    assert seq_to_author[1] is None
    assert seq_to_author[2:] == list(range(39, 57))
    assert [r.aa1 if r is not None else None for r in seq_to_atom[:4]] == [
        None, None, "P", "L"
    ]


def test_align_atom_to_fasta_rejects_ambiguous_fallback_alignment():
    fasta = "ACDEFGACDEFG"
    atom_residues = _make_atom_residues("ACDEFG", start_resnum=10)

    with pytest.raises(AlignmentError, match="ambiguous"):
        align_atom_to_fasta(atom_residues, fasta_seq=fasta, chain_range_start=100)


def test_align_atom_to_fasta_accepts_fasta_as_unique_substring_of_atom_sequence():
    fasta = "VIRVYIASSSGSTAIKKKQ"
    atom_residues = _make_atom_residues("VIRVYIASSSGSTAIKKKQHHHHHH", start_resnum=1)

    seq_to_author, seq_to_atom = align_atom_to_fasta(
        atom_residues, fasta_seq=fasta, chain_range_start=500
    )

    assert seq_to_author == list(range(1, 20))
    assert "".join(r.aa1 for r in seq_to_atom if r is not None) == fasta
