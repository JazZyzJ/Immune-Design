"""The panel builder's overlap policy: AUDIT J.7 D2.

The primary panel REMOVES the union of both Heads' training homology; the sensitivity panel
MEASURES the same union and keeps it. Both are asserted here against one fixture, because the
whole point of the sensitivity is that the two panels differ in exactly one filter and in nothing
else -- a builder that also changed, say, the clustering step between the two would report a drift
in the equal-risk line that is not attributable to the overlap.
"""

from __future__ import annotations

import argparse

import pytest

from scripts import build_dual_calibration_panel as bp

#: Distinct 120-aa canonical sequences, so length and AA20 never fire.
def _seq(letter: str) -> str:
    return (letter + "ACDEFGHIKLMNPQRSTVWY") * 6


POOL = {f"P{i}": _seq(c) for i, c in enumerate("ACDEFG")}
#: P0/P1 are Head-A homologs, P2 is a Head-B homolog. Union = 3 of 6.
HITS = {"a": {"P0", "P1"}, "b": {"P2"}}


def _args(tmp_path, policy):
    return argparse.Namespace(
        candidate_fasta=tmp_path / "pool.fasta",
        head_a_manifest_dir=tmp_path / "a", head_b_manifest_dir=tmp_path / "b",
        deployment_protein_ids=None, evaluation_protein_ids=None,
        seen_splits="train,val", work_dir=tmp_path / "work", mmseqs="mmseqs",
        head_overlap=policy)


@pytest.fixture
def patched(monkeypatch, tmp_path):
    bp.write_fasta(tmp_path / "pool.fasta", POOL)
    monkeypatch.setattr(bp, "head_seen_sequences", lambda d, s: {"S": _seq("W")})
    monkeypatch.setattr(bp, "homology_hits",
                        lambda pool, ref, *, mmseqs, tmp_dir, label: set(HITS[label]))
    monkeypatch.setattr("scripts.freeze_t0_dev_cohort.mmseqs_clusters",
                        lambda binary, pool: {pid: pid for pid in pool})
    return tmp_path


def test_the_default_policy_removes_the_measured_union(patched):
    out = bp.build(_args(patched, "exclude"))
    assert out["report"]["head_overlap_policy"] == "exclude"
    assert out["report"]["head_homology_union"] == 3
    assert out["report"]["after_head_homology"] == 3
    assert set(out["panel"]) == {"P3", "P4", "P5"}


def test_the_sensitivity_policy_measures_the_same_union_and_keeps_it(patched):
    out = bp.build(_args(patched, "include"))
    assert out["report"]["head_overlap_policy"] == "include"
    # MEASURED identically -- this is what makes the two panels comparable on the overlap itself.
    assert out["report"]["head_homology_union"] == 3
    assert out["report"]["head_homology_hits"] == {"a": 2, "b": 1}
    assert out["report"]["after_head_homology"] == 6
    assert set(out["panel"]) == set(POOL)


def test_the_two_policies_differ_in_exactly_the_one_filter(patched):
    keep = bp.build(_args(patched, "include"))["report"]
    drop = bp.build(_args(patched, "exclude"))["report"]
    differing = {k for k in keep if keep[k] != drop.get(k)}
    assert differing == {"head_overlap_policy", "after_head_homology", "after_deployment",
                         "after_evaluation", "n_clusters", "panel_size"}
    # every count BEFORE step 3 is identical, so the pool the fractions are measured against is one
    # pool measured twice
    for key in ("raw_entries", "after_dedup", "after_length", "after_canonical",
                "head_homology_fraction", "head_homology_asymmetry"):
        assert keep[key] == drop[key]


def test_the_default_is_the_primary_panel():
    parser = bp.build_parser()
    args = parser.parse_args([
        "--candidate-fasta", "p.fasta", "--head-a-manifest-dir", "a", "--head-b-manifest-dir", "b",
        "--deployment-protein-ids", "d.txt", "--out-fasta", "o.fasta", "--out-report", "r.json"])
    assert args.head_overlap == "exclude"
