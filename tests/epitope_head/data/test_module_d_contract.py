"""Module D v1.1 contract tests — seq_hash isolation, cluster isolation,
weighted split balance, disjointness, reproducibility."""

import pytest
import pandas as pd

from epitope_head.data.create_splits import (
    build_cluster_assignment,
    build_seq_hash_groups,
    compute_seq_hash,
    validate_no_cross_split,
    validate_split_disjointness,
    weighted_cluster_split,
)


def _make_protein_df(records: list[dict]) -> pd.DataFrame:
    """Helper: build minimal protein_samples DataFrame from record dicts."""
    defaults = {"allele": "HLA-DRB1*07:01", "positives_json": "[]",
                "sequence_length": 100, "input_span_count": 1,
                "duplicate_span_count": 0, "metadata_json": "{}"}
    rows = []
    for r in records:
        row = {**defaults, **r}
        if "protein_seq" not in row:
            row["protein_seq"] = "A" * row["sequence_length"]
        if "positive_count" not in row:
            row["positive_count"] = row["input_span_count"]
        rows.append(row)
    return pd.DataFrame(rows)


# ── D1: seq_hash grouping ───────────────────────────────────────────────────

class TestD1SeqHash:

    def test_identical_sequences_get_same_hash(self):
        seq = "ACDEFGHIKLMNPQRSTVWY" * 5
        df = _make_protein_df([
            {"protein_id": "P001", "protein_seq": seq},
            {"protein_id": "P002", "protein_seq": seq},
        ])
        groups = build_seq_hash_groups(df)
        hashes = groups.set_index("protein_id")["seq_hash"]
        assert hashes["P001"] == hashes["P002"]

    def test_different_sequences_get_different_hash(self):
        df = _make_protein_df([
            {"protein_id": "P001", "protein_seq": "ACDEFGHIKLMNPQRSTVWY" * 5},
            {"protein_id": "P002", "protein_seq": "WWWWWWWWWWWWWWWWWWWW" * 5},
        ])
        groups = build_seq_hash_groups(df)
        hashes = groups.set_index("protein_id")["seq_hash"]
        assert hashes["P001"] != hashes["P002"]

    def test_hash_is_deterministic(self):
        seq = "ACDEFGHIKLMNPQRSTVWY"
        assert compute_seq_hash(seq) == compute_seq_hash(seq)

    def test_all_proteins_covered(self):
        df = _make_protein_df([
            {"protein_id": f"P{i:03d}", "protein_seq": f"ACDEF{i}" + "A" * 94}
            for i in range(20)
        ])
        groups = build_seq_hash_groups(df)
        assert set(groups["protein_id"]) == {f"P{i:03d}" for i in range(20)}


# ── D2: Disjointness and coverage ───────────────────────────────────────────

class TestD2Disjointness:

    def test_valid_split_passes(self):
        validate_split_disjointness(["A", "B", "C"], ["D"], ["E"], {"A", "B", "C", "D", "E"})

    def test_overlap_detected(self):
        with pytest.raises(ValueError, match="Train-val overlap"):
            validate_split_disjointness(["A", "B"], ["B", "C"], ["D"], {"A", "B", "C", "D"})

    def test_missing_ids_detected(self):
        with pytest.raises(ValueError, match="Missing"):
            validate_split_disjointness(["A", "B"], ["C"], ["D"], {"A", "B", "C", "D", "E"})


# ── D3: Weighted cluster split ──────────────────────────────────────────────

class TestD3WeightedSplit:

    def _make_cluster_assignment(self, n_clusters, proteins_per_cluster=1):
        """Create synthetic cluster assignment."""
        records = []
        for c in range(n_clusters):
            cluster_rep = f"hash_{c:04d}"
            for p in range(proteins_per_cluster):
                pid = f"P_{c:04d}_{p}"
                records.append({
                    "protein_id": pid,
                    "seq_hash": f"hash_{c:04d}_{p}",
                    "cluster_rep": cluster_rep,
                })
        return pd.DataFrame(records)

    def _make_protein_df_for_clusters(self, cluster_assignment, weight=5):
        """Create protein_df matching the cluster assignment."""
        return _make_protein_df([
            {"protein_id": pid, "positive_count": weight}
            for pid in cluster_assignment["protein_id"]
        ])

    def test_cluster_non_crossing(self):
        """No cluster appears in multiple splits."""
        ca = self._make_cluster_assignment(50, proteins_per_cluster=3)
        pdf = self._make_protein_df_for_clusters(ca)
        train, val, test = weighted_cluster_split(ca, pdf, [0.8, 0.1, 0.1], seed=42)
        # Should not raise
        validate_no_cross_split(train, val, test, "cluster_rep", ca)

    def test_all_proteins_assigned(self):
        """All proteins end up in exactly one split."""
        ca = self._make_cluster_assignment(100, proteins_per_cluster=2)
        pdf = self._make_protein_df_for_clusters(ca)
        train, val, test = weighted_cluster_split(ca, pdf, [0.8, 0.1, 0.1], seed=42)
        all_ids = set(train) | set(val) | set(test)
        assert all_ids == set(ca["protein_id"])

    def test_weight_balance_reasonable(self):
        """Weight fractions are within 5% of targets for 200+ clusters."""
        ca = self._make_cluster_assignment(200, proteins_per_cluster=1)
        pdf = self._make_protein_df_for_clusters(ca, weight=10)
        train, val, test = weighted_cluster_split(ca, pdf, [0.8, 0.1, 0.1], seed=42)
        total = len(train) + len(val) + len(test)
        assert abs(len(train) / total - 0.8) < 0.05
        assert abs(len(val) / total - 0.1) < 0.05
        assert abs(len(test) / total - 0.1) < 0.05

    def test_skewed_cluster_handled(self):
        """One giant cluster doesn't crash; all proteins still assigned."""
        records = []
        # One big cluster with 50 proteins
        for p in range(50):
            records.append({"protein_id": f"BIG_{p}", "seq_hash": f"bighash_{p}", "cluster_rep": "big_cluster"})
        # 20 small clusters
        for c in range(20):
            records.append({"protein_id": f"SMALL_{c}", "seq_hash": f"smallhash_{c}", "cluster_rep": f"small_{c}"})
        ca = pd.DataFrame(records)
        pdf = _make_protein_df([
            {"protein_id": pid, "positive_count": 5} for pid in ca["protein_id"]
        ])
        train, val, test = weighted_cluster_split(ca, pdf, [0.8, 0.1, 0.1], seed=42)
        assert set(train) | set(val) | set(test) == set(ca["protein_id"])

    def test_deterministic(self):
        """Same input + seed produces same split."""
        ca = self._make_cluster_assignment(100)
        pdf = self._make_protein_df_for_clusters(ca)
        t1, v1, te1 = weighted_cluster_split(ca, pdf, [0.8, 0.1, 0.1], seed=42)
        t2, v2, te2 = weighted_cluster_split(ca, pdf, [0.8, 0.1, 0.1], seed=42)
        assert t1 == t2
        assert v1 == v2
        assert te1 == te2


# ── D4: Cross-split isolation validation ────────────────────────────────────

class TestD4CrossSplitValidation:

    def test_seq_hash_cross_split_detected(self):
        """Same seq_hash in different splits raises."""
        assignment = pd.DataFrame([
            {"protein_id": "P1", "seq_hash": "SAME", "cluster_rep": "c1"},
            {"protein_id": "P2", "seq_hash": "SAME", "cluster_rep": "c1"},
            {"protein_id": "P3", "seq_hash": "OTHER", "cluster_rep": "c2"},
        ])
        with pytest.raises(ValueError, match="seq_hash.*spans multiple splits"):
            validate_no_cross_split(["P1"], ["P2"], ["P3"], "seq_hash", assignment)

    def test_cluster_cross_split_detected(self):
        """Same cluster_rep in different splits raises."""
        assignment = pd.DataFrame([
            {"protein_id": "P1", "seq_hash": "h1", "cluster_rep": "SAME_CLUSTER"},
            {"protein_id": "P2", "seq_hash": "h2", "cluster_rep": "SAME_CLUSTER"},
            {"protein_id": "P3", "seq_hash": "h3", "cluster_rep": "other"},
        ])
        with pytest.raises(ValueError, match="cluster_rep.*spans multiple splits"):
            validate_no_cross_split(["P1"], ["P2"], ["P3"], "cluster_rep", assignment)

    def test_valid_isolation_passes(self):
        """Properly isolated splits pass validation."""
        assignment = pd.DataFrame([
            {"protein_id": "P1", "seq_hash": "h1", "cluster_rep": "c1"},
            {"protein_id": "P2", "seq_hash": "h2", "cluster_rep": "c2"},
            {"protein_id": "P3", "seq_hash": "h3", "cluster_rep": "c3"},
        ])
        validate_no_cross_split(["P1"], ["P2"], ["P3"], "seq_hash", assignment)
        validate_no_cross_split(["P1"], ["P2"], ["P3"], "cluster_rep", assignment)
