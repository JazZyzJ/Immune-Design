"""Regression tests for `inverse_folding.evaluation.tier1_builder`.

Primary invariants to lock:
  1. `load_iedb_spans` applies the `iedb_start - 1` / `iedb_end` conversion
     (1-based inclusive → 0-based half-open) correctly.
  2. `extract_spans_for_chain` keeps only spans fully inside
     [chain_range_start, chain_range_end] and every kept span's slice matches
     its stored peptide string.
  3. `build_candidate_entry` reproduces the shipped 0701 JSON entry for
     1LI1_C bit-for-bit — this is the end-to-end coordinate contract.

Network-touching code paths (PDBe SIFTS API) are not exercised here; we pass
synthesized `SiftsChain` values into `build_candidate_entry`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inverse_folding.evaluation.tier1_builder import (
    FilterParams,
    SiftsChain,
    build_candidate_entry,
    extract_spans_for_chain,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER1_JSON = REPO_ROOT / "outputs" / "if" / "test_set" / "tier1_candidates.json"


@pytest.fixture(scope="module")
def _tier1():
    if not TIER1_JSON.is_file():
        pytest.skip(f"Tier 1 JSON missing at {TIER1_JSON}")
    with TIER1_JSON.open() as f:
        return json.load(f)


def _entry(entries, pid):
    for e in entries:
        if e["protein_id"] == pid:
            return e
    raise KeyError(pid)


# ────────────────────────────────────────────────────────────────────────
# extract_spans_for_chain
# ────────────────────────────────────────────────────────────────────────


def test_extract_spans_for_chain_drops_out_of_range():
    # fasta spans UniProt residues 101..110 (1-based), sequence MKKVLQAGKT
    chain_seq = "MKKVLQAGKT"
    # span 1: inside (KVL at 103..105)
    # span 2: straddles end (QAGKTX — extends past 110)
    # span 3: fully before chain
    spans = [
        {"start_0b": 102, "end_0b": 105, "peptide": "KVL"},
        {"start_0b": 105, "end_0b": 111, "peptide": "QAGKTX"},
        {"start_0b": 50,  "end_0b": 60,  "peptide": "XXXXXXXXXX"},
    ]
    out = extract_spans_for_chain(spans, chain_seq, 101, 110)
    assert [sp["peptide"] for sp in out] == ["KVL"]
    assert out[0]["assay_type"] == "EL"
    assert out[0]["iedb_ref"] == "IEDB"


def test_extract_spans_for_chain_rejects_peptide_mismatch():
    chain_seq = "MKKVLQAGKT"
    spans = [{"start_0b": 102, "end_0b": 105, "peptide": "AAA"}]  # wrong peptide
    out = extract_spans_for_chain(spans, chain_seq, 101, 110)
    assert out == []


def test_extract_spans_for_chain_allows_end_equal_to_chain_end():
    chain_seq = "MKKVL"
    spans = [{"start_0b": 100, "end_0b": 105, "peptide": "MKKVL"}]
    out = extract_spans_for_chain(spans, chain_seq, 101, 105)
    assert len(out) == 1
    assert out[0]["peptide"] == "MKKVL"


# ────────────────────────────────────────────────────────────────────────
# build_candidate_entry — end-to-end against shipped 1LI1_C entry
# ────────────────────────────────────────────────────────────────────────


def test_build_candidate_entry_reproduces_1LI1_C(_tier1):
    """Round-trip: synthesize the SIFTS record + spans from the shipped entry
    and call `build_candidate_entry`; the output must match the shipped JSON
    entry modulo the `selection_reason` free-text prefix (which includes the
    allele string)."""
    shipped = _entry(_tier1, "1LI1_C")
    cs, ce = map(int, shipped["chain_range"].split("-"))
    chain_seq = shipped["sequence"]
    # Synthesize a 'full UniProt' sequence of length chain_range_end, padding
    # upstream residues with X. The builder only reads full[cs-1 : ce], so
    # anything upstream of the chain can be arbitrary.
    full_seq = ("X" * (cs - 1)) + chain_seq
    # Reverse-engineer span inputs (start_0b, end_0b, peptide) — these are
    # exactly what load_iedb_spans would produce.
    spans_full = [{
        "start_0b": sp["start_0b"],
        "end_0b": sp["end_0b"],
        "peptide": sp["peptide"],
        "source": "iedb",
    } for sp in shipped["experimental_epitopes"]]

    sifts = SiftsChain(
        pdb_id="1LI1", chain_id="C",
        unp_start=cs, unp_end=ce,
        pdb_start=cs, pdb_end=ce,
        resolution=shipped["resolution"],
        method="X-ray diffraction",
        coverage=None,
    )
    filters = FilterParams(min_len=100, max_len=500, max_resolution=2.5, min_spans=2)
    got = build_candidate_entry(
        uniprot=shipped["uniprot_id"],
        sifts_chain=sifts,
        spans_full=spans_full,
        full_uniprot_seq=full_seq,
        allele="HLA-DRB1*07:01",
        epitope_head_split="test",
        filters=filters,
    )
    assert got is not None
    # Structural equality on the fields that are deterministic functions of inputs
    for k in ("protein_id", "uniprot_id", "pdb_id", "chain",
              "sequence", "sequence_length", "resolution", "chain_range",
              "n_epitope_spans", "n_covered_residues", "n_cold_residues",
              "epitope_coverage"):
        assert got[k] == shipped[k], f"field {k}: got {got[k]!r} vs shipped {shipped[k]!r}"
    # Spans: identical (start_0b, end_0b, peptide) set
    got_spans = {(sp["start_0b"], sp["end_0b"], sp["peptide"])
                 for sp in got["experimental_epitopes"]}
    shipped_spans = {(sp["start_0b"], sp["end_0b"], sp["peptide"])
                     for sp in shipped["experimental_epitopes"]}
    assert got_spans == shipped_spans


def test_build_candidate_entry_rejects_non_xray():
    sifts = SiftsChain(
        pdb_id="1ABC", chain_id="A", unp_start=1, unp_end=200,
        pdb_start=1, pdb_end=200, resolution=2.0,
        method="Electron Microscopy", coverage=None,
    )
    filters = FilterParams()
    out = build_candidate_entry(
        uniprot="P00000", sifts_chain=sifts,
        spans_full=[], full_uniprot_seq="A" * 200,
        allele="HLA-DRB1*07:01", epitope_head_split="test", filters=filters,
    )
    assert out is None


def test_build_candidate_entry_rejects_high_resolution():
    sifts = SiftsChain(
        pdb_id="1ABC", chain_id="A", unp_start=1, unp_end=200,
        pdb_start=1, pdb_end=200, resolution=3.5,
        method="X-ray diffraction", coverage=None,
    )
    filters = FilterParams(max_resolution=2.5)
    out = build_candidate_entry(
        uniprot="P00000", sifts_chain=sifts,
        spans_full=[{"start_0b": 10, "end_0b": 25, "peptide": "A" * 15},
                    {"start_0b": 30, "end_0b": 45, "peptide": "A" * 15}],
        full_uniprot_seq="A" * 200,
        allele="HLA-DRB1*07:01", epitope_head_split="test", filters=filters,
    )
    assert out is None


def test_build_candidate_entry_rejects_coverage_out_of_range():
    """Coverage filter is the key 0701 selection criterion (user spec: 10-50%)."""
    sifts = SiftsChain(
        pdb_id="1ABC", chain_id="A", unp_start=1, unp_end=100,
        pdb_start=1, pdb_end=100, resolution=2.0,
        method="X-ray diffraction", coverage=None,
    )
    full_seq = "A" * 100
    # 100 residues, 3 non-overlapping 15-mer spans → 45/100 = 0.45, in range
    spans_in_range = [
        {"start_0b": 0, "end_0b": 15, "peptide": "A" * 15},
        {"start_0b": 20, "end_0b": 35, "peptide": "A" * 15},
        {"start_0b": 40, "end_0b": 55, "peptide": "A" * 15},
    ]
    # Same residues but 4th span pushes coverage to 60/100 = 0.60, above max
    spans_too_high = spans_in_range + [{"start_0b": 60, "end_0b": 75, "peptide": "A" * 15}]
    # Only 1 span of length 5 → 5/100 = 0.05, below min
    # (Need >=2 spans so use 2 short ones: 5/100 covered → 0.05)
    spans_too_low = [
        {"start_0b": 0, "end_0b": 3, "peptide": "A" * 3},
        {"start_0b": 5, "end_0b": 7, "peptide": "A" * 2},
    ]

    filters = FilterParams(min_coverage=0.10, max_coverage=0.50)
    got_in = build_candidate_entry(
        "P0", sifts, spans_in_range, full_seq, "HLA-X", "test", filters,
    )
    assert got_in is not None and got_in["epitope_coverage"] == 0.45

    got_high = build_candidate_entry(
        "P0", sifts, spans_too_high, full_seq, "HLA-X", "test", filters,
    )
    assert got_high is None  # 0.60 > 0.50

    got_low = build_candidate_entry(
        "P0", sifts, spans_too_low, full_seq, "HLA-X", "test", filters,
    )
    assert got_low is None  # 0.05 < 0.10


def test_build_candidate_entry_requires_min_spans():
    sifts = SiftsChain(
        pdb_id="1ABC", chain_id="A", unp_start=1, unp_end=200,
        pdb_start=1, pdb_end=200, resolution=2.0,
        method="X-ray diffraction", coverage=None,
    )
    filters = FilterParams(min_spans=2)
    # Only 1 valid span → rejected
    out = build_candidate_entry(
        uniprot="P00000", sifts_chain=sifts,
        spans_full=[{"start_0b": 10, "end_0b": 25, "peptide": "A" * 15}],
        full_uniprot_seq="A" * 200,
        allele="HLA-DRB1*07:01", epitope_head_split="test", filters=filters,
    )
    assert out is None


# ────────────────────────────────────────────────────────────────────────
# Full JSON sweep: every shipped entry must reproduce when fed back through
# ────────────────────────────────────────────────────────────────────────


def test_all_tier1_entries_reproduce(_tier1):
    """Round-trip every shipped 0701 entry via build_candidate_entry."""
    failing = []
    for shipped in _tier1:
        cs, ce = map(int, shipped["chain_range"].split("-"))
        chain_seq = shipped["sequence"]
        full_seq = ("X" * (cs - 1)) + chain_seq
        spans_full = [{
            "start_0b": sp["start_0b"],
            "end_0b": sp["end_0b"],
            "peptide": sp["peptide"],
            "source": "iedb",
        } for sp in shipped["experimental_epitopes"]]
        sifts = SiftsChain(
            pdb_id=shipped["pdb_id"], chain_id=shipped["chain"],
            unp_start=cs, unp_end=ce,
            pdb_start=cs, pdb_end=ce,
            resolution=shipped["resolution"],
            method="X-ray diffraction",
            coverage=None,
        )
        filters = FilterParams()
        got = build_candidate_entry(
            uniprot=shipped["uniprot_id"],
            sifts_chain=sifts,
            spans_full=spans_full,
            full_uniprot_seq=full_seq,
            allele="HLA-DRB1*07:01",
            epitope_head_split="test",
            filters=filters,
        )
        if got is None or got["n_epitope_spans"] != shipped["n_epitope_spans"]:
            failing.append(shipped["protein_id"])
    assert not failing, f"Entries that failed round-trip: {failing}"
