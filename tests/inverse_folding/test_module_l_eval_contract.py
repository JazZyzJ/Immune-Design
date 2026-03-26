"""Module L contract tests — evaluation pipeline schema and wrappers.

TDD gates from PLAN_IF.md:
  L0: evaluator can NOT emit partial metric rows without failing.
  L2: wrapper/parser pair rejects malformed outputs; fixtures parse to deterministic records.
  L3: NetMHCIIpan aggregation handles boundary cases; wrapper returns deterministic metrics.
  L4: end-to-end evaluator produces complete artifact bundle with aligned row counts.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from inverse_folding.evaluation.schema import (
    COMPARISON_COLUMNS,
    IMMUNOGENICITY_HEAD_COLUMNS,
    IMMUNOGENICITY_NMP_COLUMNS,
    STRUCTURAL_COLUMNS,
    EvalSchemaError,
    validate_dataframe,
)


# ── L0: Metric schema contract ──────────────────────────────────────────────

class TestMetricSchema:

    def _make_structural_row(self, **overrides):
        row = {
            "protein_id": "test_001",
            "design_id": "d_001",
            "sequence": "ACDEF",
            "scTM": 0.85,
            "pLDDT": 78.5,
            "bb_RMSD": 1.2,
            "recovery": 0.65,
            "foldability": True,
        }
        row.update(overrides)
        return row

    def _make_immuno_head_row(self, **overrides):
        row = {
            "protein_id": "test_001",
            "design_id": "d_001",
            "global_risk": 0.42,
            "mean_hotspot": 0.35,
            "max_hotspot": 0.78,
            "n_hotspot_positions": 5,
        }
        row.update(overrides)
        return row

    def _make_immuno_nmp_row(self, **overrides):
        row = {
            "protein_id": "test_001",
            "design_id": "d_001",
            "n_strong_binders": 3,
            "n_weak_binders": 8,
            "mean_best_rank": 2.5,
            "n_windows_scored": 45,
        }
        row.update(overrides)
        return row

    def _make_comparison_row(self, **overrides):
        row = {
            "protein_id": "test_001",
            "design_id": "d_001",
            "delta_global_risk_head": -0.15,
            "delta_mean_best_rank_nmp": -1.2,
            "delta_n_strong_binders": -2,
            "hotspot_reduction": 0.25,
            "mutation_count": 12,
            "risk_per_mutation": -0.0125,
        }
        row.update(overrides)
        return row

    # ── Complete rows accepted ───────────────────────────────────────────

    def test_valid_structural_row_accepted(self):
        df = pd.DataFrame([self._make_structural_row()])
        result = validate_dataframe(df, STRUCTURAL_COLUMNS)
        assert len(result) == 1

    def test_valid_immuno_head_row_accepted(self):
        df = pd.DataFrame([self._make_immuno_head_row()])
        result = validate_dataframe(df, IMMUNOGENICITY_HEAD_COLUMNS)
        assert len(result) == 1

    def test_valid_immuno_nmp_row_accepted(self):
        df = pd.DataFrame([self._make_immuno_nmp_row()])
        result = validate_dataframe(df, IMMUNOGENICITY_NMP_COLUMNS)
        assert len(result) == 1

    def test_valid_comparison_row_accepted(self):
        df = pd.DataFrame([self._make_comparison_row()])
        result = validate_dataframe(df, COMPARISON_COLUMNS)
        assert len(result) == 1

    # ── Missing columns rejected ─────────────────────────────────────────

    def test_structural_missing_scTM_rejected(self):
        row = self._make_structural_row()
        del row["scTM"]
        df = pd.DataFrame([row])
        with pytest.raises(EvalSchemaError, match="scTM"):
            validate_dataframe(df, STRUCTURAL_COLUMNS)

    def test_immuno_head_missing_global_risk_rejected(self):
        row = self._make_immuno_head_row()
        del row["global_risk"]
        df = pd.DataFrame([row])
        with pytest.raises(EvalSchemaError, match="global_risk"):
            validate_dataframe(df, IMMUNOGENICITY_HEAD_COLUMNS)

    def test_immuno_nmp_missing_n_strong_binders_rejected(self):
        row = self._make_immuno_nmp_row()
        del row["n_strong_binders"]
        df = pd.DataFrame([row])
        with pytest.raises(EvalSchemaError, match="n_strong_binders"):
            validate_dataframe(df, IMMUNOGENICITY_NMP_COLUMNS)

    def test_comparison_missing_delta_rejected(self):
        row = self._make_comparison_row()
        del row["delta_global_risk_head"]
        df = pd.DataFrame([row])
        with pytest.raises(EvalSchemaError, match="delta_global_risk_head"):
            validate_dataframe(df, COMPARISON_COLUMNS)

    # ── Empty dataframe rejected ─────────────────────────────────────────

    def test_empty_structural_df_rejected(self):
        df = pd.DataFrame(columns=list(STRUCTURAL_COLUMNS))
        with pytest.raises(EvalSchemaError, match="empty"):
            validate_dataframe(df, STRUCTURAL_COLUMNS)

    # ── Extra columns preserved ──────────────────────────────────────────

    def test_extra_columns_preserved(self):
        row = self._make_structural_row(custom_note="test")
        df = pd.DataFrame([row])
        result = validate_dataframe(df, STRUCTURAL_COLUMNS)
        assert "custom_note" in result.columns


# ── L2: Structural wrappers — TM-align output parsing ───────────────────────

class TestTMAlignParser:

    FIXTURE_TMALIGN_OUTPUT = """
 **************************************************************************
 *                        TM-align (Version 20220412)                     *
 * An algorithm for protein structure alignment and comparison            *
 **************************************************************************

Name of Chain_1: /tmp/pred.pdb (to be superimposed onto Chain_2)
Name of Chain_2: /tmp/ref.pdb
Length of Chain_1:  150 residues
Length of Chain_2:  150 residues

Aligned length=  148, RMSD=   1.23, Seq_ID=n_identical/n_aligned=  0.650
TM-score= 0.8765 (if normalized by length of Chain_1, i.e., LN=150, d0=4.23)
TM-score= 0.8765 (if normalized by length of Chain_2, i.e., LN=150, d0=4.23)
"""

    def test_parse_tm_score(self):
        from inverse_folding.evaluation.tmalign import parse_tmalign_output
        result = parse_tmalign_output(self.FIXTURE_TMALIGN_OUTPUT)
        assert abs(result["tm_score"] - 0.8765) < 1e-4

    def test_parse_rmsd(self):
        from inverse_folding.evaluation.tmalign import parse_tmalign_output
        result = parse_tmalign_output(self.FIXTURE_TMALIGN_OUTPUT)
        assert abs(result["rmsd"] - 1.23) < 1e-2

    def test_parse_aligned_length(self):
        from inverse_folding.evaluation.tmalign import parse_tmalign_output
        result = parse_tmalign_output(self.FIXTURE_TMALIGN_OUTPUT)
        assert result["aligned_length"] == 148

    def test_parse_seq_identity(self):
        from inverse_folding.evaluation.tmalign import parse_tmalign_output
        result = parse_tmalign_output(self.FIXTURE_TMALIGN_OUTPUT)
        assert abs(result["seq_identity"] - 0.65) < 1e-2

    def test_malformed_output_raises(self):
        from inverse_folding.evaluation.tmalign import parse_tmalign_output
        with pytest.raises(ValueError, match="TM-score"):
            parse_tmalign_output("garbage output\nno scores here\n")


# ── L3: NetMHCIIpan aggregation ─────────────────────────────────────────────

class TestNetMHCIIpanAggregation:

    def test_aggregation_counts_strong_binders(self):
        from inverse_folding.evaluation.immunogenicity import aggregate_nmp_scores
        # %Rank_EL: <2% = strong, <10% = weak
        scores = pd.DataFrame({
            "peptide": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "rank_EL": [0.5, 1.5, 5.0, 15.0, 25.0],
        })
        agg = aggregate_nmp_scores(scores)
        assert agg["n_strong_binders"] == 2   # 0.5, 1.5
        assert agg["n_weak_binders"] == 3     # 0.5, 1.5, 5.0

    def test_aggregation_mean_best_rank(self):
        from inverse_folding.evaluation.immunogenicity import aggregate_nmp_scores
        scores = pd.DataFrame({
            "peptide": [f"P{i}" for i in range(10)],
            "rank_EL": [0.5, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        })
        agg = aggregate_nmp_scores(scores)
        # mean of top-5: (0.5+1.0+2.0+3.0+4.0)/5 = 2.1
        assert abs(agg["mean_best_rank"] - 2.1) < 1e-6

    def test_aggregation_empty_scores(self):
        from inverse_folding.evaluation.immunogenicity import aggregate_nmp_scores
        scores = pd.DataFrame(columns=["peptide", "rank_EL"])
        agg = aggregate_nmp_scores(scores)
        assert agg["n_strong_binders"] == 0
        assert agg["n_weak_binders"] == 0
        assert agg["n_windows_scored"] == 0

    def test_aggregation_preserves_window_count(self):
        from inverse_folding.evaluation.immunogenicity import aggregate_nmp_scores
        scores = pd.DataFrame({
            "peptide": [f"P{i}" for i in range(20)],
            "rank_EL": [5.0] * 20,
        })
        agg = aggregate_nmp_scores(scores)
        assert agg["n_windows_scored"] == 20


class TestPeptideScoreConversion:
    """Verify unit conversion from PeptideScore (0-1 fraction) to DataFrame (percentage)."""

    def test_fraction_to_percentage(self):
        from inverse_folding.evaluation.immunogenicity import peptide_scores_to_dataframe
        # Simulate PeptideScore objects with el_rank as 0-1 fraction
        from types import SimpleNamespace
        scores = [
            SimpleNamespace(peptide="AAA", el_rank=0.015, pos=0, core="AAA", el_score=0.9),
            SimpleNamespace(peptide="BBB", el_rank=0.08, pos=1, core="BBB", el_score=0.5),
        ]
        df = peptide_scores_to_dataframe(scores)
        # 0.015 fraction → 1.5 percentage (strong binder)
        assert abs(df.iloc[0]["rank_EL"] - 1.5) < 1e-6
        # 0.08 fraction → 8.0 percentage (weak binder)
        assert abs(df.iloc[1]["rank_EL"] - 8.0) < 1e-6

    def test_converted_scores_aggregate_correctly(self):
        """End-to-end: PeptideScore → DataFrame → aggregate."""
        from inverse_folding.evaluation.immunogenicity import (
            aggregate_nmp_scores,
            peptide_scores_to_dataframe,
        )
        from types import SimpleNamespace
        scores = [
            SimpleNamespace(peptide="A" * 15, el_rank=0.01, pos=0, core="A" * 9, el_score=0.95),   # 1% → strong
            SimpleNamespace(peptide="B" * 15, el_rank=0.05, pos=1, core="B" * 9, el_score=0.7),    # 5% → weak only
            SimpleNamespace(peptide="C" * 15, el_rank=0.20, pos=2, core="C" * 9, el_score=0.3),    # 20% → neither
        ]
        df = peptide_scores_to_dataframe(scores)
        agg = aggregate_nmp_scores(df)
        assert agg["n_strong_binders"] == 1   # only 1%
        assert agg["n_weak_binders"] == 2     # 1% and 5%
        assert agg["n_windows_scored"] == 3


# ── L4: Artifact bundle alignment ───────────────────────────────────────────

class TestArtifactBundleAlignment:

    def test_aligned_row_counts(self):
        """All output tables must have the same number of rows."""
        from inverse_folding.evaluation.aggregate import validate_bundle_alignment
        structural = pd.DataFrame([
            {"protein_id": "p1", "design_id": "d1", "scTM": 0.8},
            {"protein_id": "p1", "design_id": "d2", "scTM": 0.7},
        ])
        immuno_head = pd.DataFrame([
            {"protein_id": "p1", "design_id": "d1", "global_risk": 0.3},
            {"protein_id": "p1", "design_id": "d2", "global_risk": 0.4},
        ])
        immuno_nmp = pd.DataFrame([
            {"protein_id": "p1", "design_id": "d1", "n_strong_binders": 2},
            {"protein_id": "p1", "design_id": "d2", "n_strong_binders": 3},
        ])
        # Should not raise
        validate_bundle_alignment(structural, immuno_head, immuno_nmp)

    def test_mismatched_row_counts_rejected(self):
        from inverse_folding.evaluation.aggregate import validate_bundle_alignment
        structural = pd.DataFrame([
            {"protein_id": "p1", "design_id": "d1", "scTM": 0.8},
        ])
        immuno_head = pd.DataFrame([
            {"protein_id": "p1", "design_id": "d1", "global_risk": 0.3},
            {"protein_id": "p1", "design_id": "d2", "global_risk": 0.4},
        ])
        immuno_nmp = pd.DataFrame([
            {"protein_id": "p1", "design_id": "d1", "n_strong_binders": 2},
        ])
        with pytest.raises(EvalSchemaError, match="row count"):
            validate_bundle_alignment(structural, immuno_head, immuno_nmp)

    def test_mismatched_design_ids_rejected(self):
        from inverse_folding.evaluation.aggregate import validate_bundle_alignment
        structural = pd.DataFrame([
            {"protein_id": "p1", "design_id": "d1", "scTM": 0.8},
        ])
        immuno_head = pd.DataFrame([
            {"protein_id": "p1", "design_id": "d_WRONG", "global_risk": 0.3},
        ])
        immuno_nmp = pd.DataFrame([
            {"protein_id": "p1", "design_id": "d1", "n_strong_binders": 2},
        ])
        with pytest.raises(EvalSchemaError, match="design_id"):
            validate_bundle_alignment(structural, immuno_head, immuno_nmp)

    def test_cross_protein_swap_detected(self):
        """Tables with same design_ids but swapped protein_ids must be caught."""
        from inverse_folding.evaluation.aggregate import validate_bundle_alignment
        structural = pd.DataFrame([
            {"protein_id": "p1", "design_id": "d1", "scTM": 0.8},
            {"protein_id": "p2", "design_id": "d1", "scTM": 0.7},
        ])
        immuno_head = pd.DataFrame([
            {"protein_id": "p2", "design_id": "d1", "global_risk": 0.3},  # swapped
            {"protein_id": "p1", "design_id": "d1", "global_risk": 0.4},
        ])
        immuno_nmp = pd.DataFrame([
            {"protein_id": "p1", "design_id": "d1", "n_strong_binders": 2},
            {"protein_id": "p2", "design_id": "d1", "n_strong_binders": 3},
        ])
        with pytest.raises(EvalSchemaError, match="protein_id"):
            validate_bundle_alignment(structural, immuno_head, immuno_nmp)


class TestValidateIFBaselineWrapper:

    @staticmethod
    def _load_module():
        module_path = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "validate_if_baseline.py"
        )
        spec = importlib.util.spec_from_file_location(
            "validate_if_baseline", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_build_test_command_uses_prediction_mode(self, tmp_path):
        module = self._load_module()

        dplm_root = tmp_path / "dplm"
        dplm_root.mkdir()
        (dplm_root / "test.py").write_text("# stub\n")

        args = module.argparse.Namespace(
            dplm_root=str(dplm_root),
            experiment_path=str(tmp_path / "exp"),
            ckpt_path=str(tmp_path / "exp" / "checkpoints" / "best.ckpt"),
            data_dir=str(tmp_path / "data"),
            output_dir=str(tmp_path / "exp"),
            max_iter=10,
            temperature=1.0,
            experiment="dplm/cond_dplm_650m",
            num_gpus=1,
        )

        cmd = module.build_test_command(args)

        assert "mode=predict" in cmd

    def test_build_test_command_disables_eval_sc_during_generation(self, tmp_path):
        module = self._load_module()

        dplm_root = tmp_path / "dplm"
        dplm_root.mkdir()
        (dplm_root / "test.py").write_text("# stub\n")

        args = module.argparse.Namespace(
            dplm_root=str(dplm_root),
            experiment_path=str(tmp_path / "exp"),
            ckpt_path=str(tmp_path / "exp" / "checkpoints" / "best.ckpt"),
            data_dir=str(tmp_path / "data"),
            output_dir=str(tmp_path / "exp"),
            max_iter=10,
            temperature=1.0,
            experiment="dplm/cond_dplm_650m",
            num_gpus=1,
        )

        cmd = module.build_test_command(args, eval_sc=False)

        assert any(token.endswith("task.generator.eval_sc=false") for token in cmd)
        assert "+task.generator.eval_sc=True" not in cmd

    def test_merge_sc_metrics_overwrites_placeholder_scores(self):
        module = self._load_module()

        results = [{
            "name": "protA",
            "AAR": 0.42,
            "scTM": 0.0,
            "scRMSD": 0.0,
            "plddt": 0.0,
            "sequence": "ACDE",
        }]
        sc_metrics = {
            "protA": {"scTM": 0.81, "scRMSD": 1.23, "plddt": 77.5},
        }

        merged = module.merge_sc_metrics(results, sc_metrics)

        assert merged[0]["AAR"] == 0.42
        assert merged[0]["scTM"] == 0.81
        assert merged[0]["scRMSD"] == 1.23
        assert merged[0]["plddt"] == 77.5


# ── L4: compute_comparison fail-fast on missing WT data ──────────────────────

class TestComparisonFailFast:

    def test_missing_wt_nmp_strong_raises(self):
        import numpy as np
        from inverse_folding.evaluation.aggregate import compute_comparison
        structural = pd.DataFrame([{
            "protein_id": "p1", "design_id": "d1", "sequence": "ACDEF",
        }])
        immuno_head = pd.DataFrame([{
            "protein_id": "p1", "design_id": "d1", "global_risk": 0.3,
            "mean_hotspot": 0.2, "max_hotspot": 0.5,
        }])
        immuno_nmp = pd.DataFrame([{
            "protein_id": "p1", "design_id": "d1",
            "n_strong_binders": 1, "mean_best_rank": 3.0,
        }])
        with pytest.raises(EvalSchemaError, match="wt_nmp_strong"):
            compute_comparison(
                structural, immuno_head, immuno_nmp,
                wt_head={"p1": 0.5}, wt_nmp={"p1": 4.0},
                wt_sequences={"p1": "ACDEF"},
                wt_nmp_strong=None,
                wt_hotspot_scores={"p1": np.array([0.1, 0.2, 0.3, 0.4, 0.5])},
            )

    def test_missing_wt_hotspot_scores_raises(self):
        from inverse_folding.evaluation.aggregate import compute_comparison
        structural = pd.DataFrame([{
            "protein_id": "p1", "design_id": "d1", "sequence": "ACDEF",
        }])
        immuno_head = pd.DataFrame([{
            "protein_id": "p1", "design_id": "d1", "global_risk": 0.3,
            "mean_hotspot": 0.2, "max_hotspot": 0.5,
        }])
        immuno_nmp = pd.DataFrame([{
            "protein_id": "p1", "design_id": "d1",
            "n_strong_binders": 1, "mean_best_rank": 3.0,
        }])
        with pytest.raises(EvalSchemaError, match="wt_hotspot_scores"):
            compute_comparison(
                structural, immuno_head, immuno_nmp,
                wt_head={"p1": 0.5}, wt_nmp={"p1": 4.0},
                wt_sequences={"p1": "ACDEF"},
                wt_nmp_strong={"p1": 2},
                wt_hotspot_scores=None,
            )

    def test_missing_wt_sequence_for_protein_raises(self):
        import numpy as np
        from inverse_folding.evaluation.aggregate import compute_comparison
        structural = pd.DataFrame([{
            "protein_id": "p_unknown", "design_id": "d1", "sequence": "ACDEF",
        }])
        immuno_head = pd.DataFrame([{
            "protein_id": "p_unknown", "design_id": "d1", "global_risk": 0.3,
            "mean_hotspot": 0.2, "max_hotspot": 0.5,
        }])
        immuno_nmp = pd.DataFrame([{
            "protein_id": "p_unknown", "design_id": "d1",
            "n_strong_binders": 1, "mean_best_rank": 3.0,
        }])
        with pytest.raises(EvalSchemaError, match="p_unknown"):
            compute_comparison(
                structural, immuno_head, immuno_nmp,
                wt_head={"p1": 0.5}, wt_nmp={"p1": 4.0},
                wt_sequences={"p1": "ACDEF"},
                wt_nmp_strong={"p1": 2},
                wt_hotspot_scores={"p1": np.array([0.1, 0.2, 0.3, 0.4, 0.5])},
            )

    def test_valid_comparison_computes_real_deltas(self):
        import numpy as np
        from inverse_folding.evaluation.aggregate import compute_comparison
        structural = pd.DataFrame([{
            "protein_id": "p1", "design_id": "d1", "sequence": "XCDEY",
        }])
        immuno_head = pd.DataFrame([{
            "protein_id": "p1", "design_id": "d1", "global_risk": 0.3,
            "mean_hotspot": 0.15, "max_hotspot": 0.4,
        }])
        immuno_nmp = pd.DataFrame([{
            "protein_id": "p1", "design_id": "d1",
            "n_strong_binders": 1, "mean_best_rank": 3.0,
        }])
        result = compute_comparison(
            structural, immuno_head, immuno_nmp,
            wt_head={"p1": 0.5},
            wt_nmp={"p1": 4.0},
            wt_sequences={"p1": "ACDEF"},
            wt_nmp_strong={"p1": 3},
            wt_hotspot_scores={"p1": np.array([0.1, 0.2, 0.3, 0.4, 0.5])},
        )
        row = result.iloc[0]
        # delta_risk_head = 0.3 - 0.5 = -0.2
        assert abs(row["delta_global_risk_head"] - (-0.2)) < 1e-6
        # delta_rank_nmp = 3.0 - 4.0 = -1.0
        assert abs(row["delta_mean_best_rank_nmp"] - (-1.0)) < 1e-6
        # delta_strong = 1 - 3 = -2
        assert row["delta_n_strong_binders"] == -2
        # mutation_count: X!=A, C==C, D==D, E==E, Y!=F → 2
        assert row["mutation_count"] == 2
