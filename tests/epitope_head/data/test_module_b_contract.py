"""Module B contract tests — lookup fallback, boundary validation, sequence match."""

import hashlib
from pathlib import Path

import pandas as pd

from epitope_head.data import build_dataset as build_dataset_module
from epitope_head.data.join_fasta import JoinStats, join_and_validate, resolve_protein


# ── B1: Lookup resolver ──────────────────────────────────────────────────────

class TestB1Lookup:

    FASTA_INDEX = {
        "P00001": "ACDEFGHIKLMNPQRSTVWY" * 5,  # len=100
        "Q99999": "MMMMMMMMMM" * 10,            # len=100
    }

    def test_exact_hit(self):
        seq, hit = resolve_protein("P00001", self.FASTA_INDEX)
        assert hit == "exact_hit"
        assert seq is not None

    def test_stripped_hit(self):
        seq, hit = resolve_protein("P00001.3", self.FASTA_INDEX)
        assert hit == "stripped_hit"
        assert seq == self.FASTA_INDEX["P00001"]

    def test_miss(self):
        seq, hit = resolve_protein("XXXXXX", self.FASTA_INDEX)
        assert hit == "miss"
        assert seq is None

    def test_no_false_strip(self):
        """ID without dot that doesn't exist should miss, not try stripping."""
        seq, hit = resolve_protein("NOTHERE", self.FASTA_INDEX)
        assert hit == "miss"


# ── B3: Coordinate boundary validation ───────────────────────────────────────

class TestB3Boundary:

    FASTA_INDEX = {"P001": "ACDEFGHIKLMNPQRSTVWY"}  # len=20

    def _make_span_df(self, start_0b, end_0b, peptide_seq=None):
        if peptide_seq is None:
            peptide_seq = self.FASTA_INDEX["P001"][start_0b:end_0b]
        return pd.DataFrame([{
            "protein_id": "P001", "allele": "HLA-DRB1*07:01",
            "start_1b": start_0b + 1, "end_1b": end_0b,
            "start_0b": start_0b, "end_0b": end_0b,
            "pep_len": end_0b - start_0b,
            "peptide_seq": peptide_seq,
            "source": "test", "dataset_source": "test_v1",
        }])

    def test_valid_span_retained(self):
        df = self._make_span_df(0, 15)
        stats = JoinStats()
        out = join_and_validate(df, self.FASTA_INDEX, stats)
        assert len(out) == 1
        assert stats.counts["retained"] == 1

    def test_end_exceeds_seq_length_dropped(self):
        df = self._make_span_df(10, 25, peptide_seq="X" * 15)  # end=25 > len=20
        stats = JoinStats()
        out = join_and_validate(df, self.FASTA_INDEX, stats)
        assert len(out) == 0
        assert stats.counts.get("drop_out_of_range", 0) == 1

    def test_negative_start_dropped(self):
        df = self._make_span_df(-1, 10, peptide_seq="X" * 11)
        stats = JoinStats()
        out = join_and_validate(df, self.FASTA_INDEX, stats)
        assert len(out) == 0
        assert stats.counts.get("drop_out_of_range", 0) == 1

    def test_start_equals_end_dropped(self):
        df = self._make_span_df(5, 5, peptide_seq="")
        stats = JoinStats()
        out = join_and_validate(df, self.FASTA_INDEX, stats)
        assert len(out) == 0
        assert stats.counts.get("drop_out_of_range", 0) == 1

    def test_pep_len_invariant_violation_dropped(self):
        """pep_len != end_0b - start_0b should be rejected."""
        df = pd.DataFrame([{
            "protein_id": "P001", "allele": "HLA-DRB1*07:01",
            "start_1b": 1, "end_1b": 15,
            "start_0b": 0, "end_0b": 15, "pep_len": 999,
            "peptide_seq": self.FASTA_INDEX["P001"][0:15],
            "source": "test", "dataset_source": "test_v1",
        }])
        stats = JoinStats()
        out = join_and_validate(df, self.FASTA_INDEX, stats)
        assert len(out) == 0
        assert stats.counts.get("drop_len_invariant", 0) == 1

    def test_empty_output_preserves_schema(self):
        """All-rejected input should still produce DataFrame with correct columns."""
        df = pd.DataFrame([{
            "protein_id": "MISSING", "allele": "X",
            "start_1b": 1, "end_1b": 15,
            "start_0b": 0, "end_0b": 15, "pep_len": 15,
            "peptide_seq": "XXX",
            "source": "t", "dataset_source": "t",
        }])
        stats = JoinStats()
        out = join_and_validate(df, self.FASTA_INDEX, stats)
        assert len(out) == 0
        assert "protein_seq" in out.columns


# ── B4: Peptide exact-match ──────────────────────────────────────────────────

class TestB4PeptideMatch:

    FASTA_INDEX = {"P001": "ACDEFGHIKLMNPQRSTVWY"}  # len=20

    def _make_span_df(self, start_0b, end_0b, peptide_seq):
        return pd.DataFrame([{
            "protein_id": "P001", "allele": "HLA-DRB1*07:01",
            "start_1b": start_0b + 1, "end_1b": end_0b,
            "start_0b": start_0b, "end_0b": end_0b,
            "pep_len": end_0b - start_0b,
            "peptide_seq": peptide_seq,
            "source": "test", "dataset_source": "test_v1",
        }])

    def test_exact_match_retained(self):
        seq = "ACDEFGHIKLMNPQR"  # positions 0-15
        df = self._make_span_df(0, 15, seq)
        stats = JoinStats()
        out = join_and_validate(df, self.FASTA_INDEX, stats)
        assert len(out) == 1

    def test_mismatch_dropped(self):
        df = self._make_span_df(0, 15, "XXXXXXXXXXXXXYZ")
        stats = JoinStats()
        out = join_and_validate(df, self.FASTA_INDEX, stats)
        assert len(out) == 0
        assert stats.counts.get("drop_seq_mismatch", 0) == 1

    def test_single_char_mismatch_dropped(self):
        mutated = "XCDEFGHIKLMNPQR"  # first char wrong
        df = self._make_span_df(0, 15, mutated)
        stats = JoinStats()
        out = join_and_validate(df, self.FASTA_INDEX, stats)
        assert len(out) == 0
        assert stats.counts.get("drop_seq_mismatch", 0) == 1


class TestB2B5B6StageB:
    """Stage-B integration checks for profile parity, schema completeness, reproducibility."""

    REQUIRED_WITH_SEQ = {
        "protein_id",
        "allele",
        "start_1b",
        "end_1b",
        "start_0b",
        "end_0b",
        "pep_len",
        "peptide_seq",
        "source",
        "dataset_source",
        "protein_seq",
    }

    @staticmethod
    def _write_fasta(path: Path, records: dict[str, str]) -> None:
        lines = []
        for protein_id, seq in records.items():
            lines.append(f">{protein_id}")
            lines.append(seq)
        path.write_text("\n".join(lines) + "\n")

    def _prepare_stage_b_inputs(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / "outputs" / "manifests"
        data_dir = tmp_path / "data"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        strict_df = pd.DataFrame(
            [
                {
                    "protein_id": "P001",
                    "allele": "HLA-DRB1*07:01",
                    "start_1b": 1,
                    "end_1b": 15,
                    "start_0b": 0,
                    "end_0b": 15,
                    "pep_len": 15,
                    "peptide_seq": "ACDEFGHIKLMNPQR",
                    "source": "test",
                    "dataset_source": "test_v1",
                }
            ]
        )
        balanced_df = pd.DataFrame(
            [
                {
                    "protein_id": "MISSING.1",
                    "allele": "HLA-DRB1*07:01",
                    "start_1b": 1,
                    "end_1b": 15,
                    "start_0b": 0,
                    "end_0b": 15,
                    "pep_len": 15,
                    "peptide_seq": "ACDEFGHIKLMNPQR",
                    "source": "test",
                    "dataset_source": "test_v1",
                }
            ]
        )

        strict_df.to_parquet(manifest_dir / "span_records_strict.parquet", index=False)
        balanced_df.to_parquet(manifest_dir / "span_records_balanced.parquet", index=False)
        self._write_fasta(
            data_dir / "all_sequences.fasta",
            {"P001": "ACDEFGHIKLMNPQRSTVWY"},
        )

    @staticmethod
    def _df_digest(df: pd.DataFrame) -> str:
        row_hashes = pd.util.hash_pandas_object(df, index=False).values.tobytes()
        return hashlib.sha256(row_hashes).hexdigest()

    def test_b2_profile_outputs_have_schema_parity(self, tmp_path, monkeypatch):
        self._prepare_stage_b_inputs(tmp_path)
        monkeypatch.setattr(build_dataset_module, "PROJECT_ROOT", tmp_path)

        build_dataset_module.build_stage_b({"fasta_path": "data/all_sequences.fasta"})

        strict_out = pd.read_parquet(tmp_path / "outputs/manifests/span_records_with_seq_strict.parquet")
        balanced_out = pd.read_parquet(
            tmp_path / "outputs/manifests/span_records_with_seq_balanced.parquet"
        )
        assert list(strict_out.columns) == list(balanced_out.columns)

    def test_b5_outputs_always_include_required_stage_c_fields(self, tmp_path, monkeypatch):
        self._prepare_stage_b_inputs(tmp_path)
        monkeypatch.setattr(build_dataset_module, "PROJECT_ROOT", tmp_path)

        build_dataset_module.build_stage_b({"fasta_path": "data/all_sequences.fasta"})

        strict_out = pd.read_parquet(tmp_path / "outputs/manifests/span_records_with_seq_strict.parquet")
        balanced_out = pd.read_parquet(
            tmp_path / "outputs/manifests/span_records_with_seq_balanced.parquet"
        )
        assert self.REQUIRED_WITH_SEQ.issubset(strict_out.columns)
        assert self.REQUIRED_WITH_SEQ.issubset(balanced_out.columns)

    def test_b6_repeated_stage_b_runs_are_summary_and_artifact_deterministic(
        self, tmp_path, monkeypatch
    ):
        self._prepare_stage_b_inputs(tmp_path)
        monkeypatch.setattr(build_dataset_module, "PROJECT_ROOT", tmp_path)

        summary_1 = build_dataset_module.build_stage_b({"fasta_path": "data/all_sequences.fasta"})
        strict_1 = pd.read_parquet(tmp_path / "outputs/manifests/span_records_with_seq_strict.parquet")
        balanced_1 = pd.read_parquet(
            tmp_path / "outputs/manifests/span_records_with_seq_balanced.parquet"
        )
        digest_1 = {
            "strict": self._df_digest(strict_1),
            "balanced": self._df_digest(balanced_1),
        }

        summary_2 = build_dataset_module.build_stage_b({"fasta_path": "data/all_sequences.fasta"})
        strict_2 = pd.read_parquet(tmp_path / "outputs/manifests/span_records_with_seq_strict.parquet")
        balanced_2 = pd.read_parquet(
            tmp_path / "outputs/manifests/span_records_with_seq_balanced.parquet"
        )
        digest_2 = {
            "strict": self._df_digest(strict_2),
            "balanced": self._df_digest(balanced_2),
        }

        assert summary_1 == summary_2
        assert digest_1 == digest_2
