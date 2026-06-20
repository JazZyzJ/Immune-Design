"""T6 contract tests: D2/D3 script wiring in ``run_if_phase_c1.py``.

Focuses on the script-level behaviors that PLAN_RF.md §"Test plan" 6
demands: widened manifest, ``arm`` matching ``controller.mode``,
``controller_surface_version`` stamp, refresh_log widening, and the
startup print of the resolved controller config. End-to-end fixture
behavior (running the actual sampler) is exercised by other test files;
this file uses helper functions in isolation so it stays under a second.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from inverse_folding.reference_flow.controller import (
    D1RefreshRecord,
    UnifiedActionabilityState,
)
from inverse_folding.reference_flow.head_scoring import WindowRiskRecord
from scripts.run_if_phase_c1 import (
    CONTROLLER_SURFACE_VERSION,
    SC_GR_PROBE_REFRESH_FILE,
    SC_GR_PROBE_SAMPLES_FILE,
    compute_per_protein_summary,
    d1_manifest_provenance,
    load_controller_setup,
    write_actionability_artifacts,
    write_d1_artifacts,
    write_sc_gr_probe_artifacts,
    _actionability_state_to_records,
    _refresh_record_to_jsonable,
)


_SCGR_MONITOR_PRESET = (
    Path(__file__).resolve().parents[2]
    / "inverse_folding"
    / "reference_flow"
    / "configs"
    / "d2_d3_full_stageB_aopen_scgr_monitor.yaml"
)


_STAGEB_PRESET = (
    Path(__file__).resolve().parents[2]
    / "inverse_folding"
    / "reference_flow"
    / "configs"
    / "d2_d3_full_stageB.yaml"
)


def _make_actionability_state(L: int, refresh_step: int = 0) -> UnifiedActionabilityState:
    arange = np.arange(L, dtype=float)
    return UnifiedActionabilityState(
        tau_ref_B=0.5,
        h_cur=arange,
        b_cur=arange * 0.1,
        b_env=arange * 0.2,
        env_peak=arange * 0.2,
        env_consistency=np.ones(L),
        e_fresh=arange * 0.15,
        b_mem=arange * 0.05,
        r_ctx=np.full(L, 0.3),
        v_target=arange * 0.25,
        u_pressure=arange * 0.2,
        G=0.041,
        g_GR_diagnostic=0.52,
        context_pnll=np.where(arange > 0, arange, np.nan),
        g_time_pre=np.full(L, 0.9),
        g_comp_pre=np.full(L, 0.5),
        g_ent_pre=np.full(L, 0.8),
        g_pnll_pre=np.full(L, 0.4),
        g_stability_pre=np.ones(L),
        cluster_support=np.full(L, 0.7),
        env_coverage_flag=np.ones(L, dtype=bool),
        legacy_residue_excess=np.zeros(L),
        active_target_flag=np.array([i >= L - 2 for i in range(L)]),
        active_block_id=np.array([0 if i >= L - 2 else -1 for i in range(L)]),
        num_seed_windows=8,
        num_env_head_calls=3,
        num_active_windows=2,
        num_active_blocks=1,
        stability_available=False,
        refresh_step=int(refresh_step),
        step=5,
        t=0.6,
        num_actionable_windows_pre_cap=6,
    )


def _write_d2_d3_yaml(tmp_path: Path) -> Path:
    body = dedent(
        """
        controller:
          enabled: true
          mode: d2_d3_full
          t_start: 0.5
          refresh_interval: 5
          completion:
            method: argmax
          head:
            score_scale: raw_logit
            static_cache_policy: lazy_write
            local_risk_aggregation: LME
          active_windows:
            excess_threshold: 0.0
            max_windows: 16
            selection: threshold_then_top_n
            merge_overlapping_scoring_windows: true
          reliability:
            time_k: 20.0
            entropy_h0: 1.5
            min_completion_fraction: 0.0
            min_rho_to_emit_event: 0.0
          d2:
            enabled: true
            beta: 1.0
            eta: 0.7
            struct_temperature: 1.0
            delta_struct: 1.5
            context_pnll_h0: 2.0
            context_jsd_h0: 0.5
            epsilon: 1.0e-8
            candidate_mode: structure_topk
            top_k_tokens: 4
            max_positions_per_block: 2
            max_candidates_per_block: 32
            selection_score: residue_excess_then_low_entropy
            min_delta_R_improvement: 0.0
            min_ess_fraction: 0.25
            max_abs_logit_shift: 5.0
            paired_uncorrected_sample: true
            completion_ensemble_enabled: true
            completion_ensemble_size: 3
            completion_ensemble_scope: local_windows
            completion_ensemble_rescore_top_m: 8
            completion_ensemble_use_variance_gate: false
            sticky_ttl_steps: 5
            sticky_clear_on_selected: true
            sticky_clear_on_remask: true
            sticky_overwrite_on_refresh: true
          d3:
            enabled: true
            window_to_residue_projection: max_covering_window
            gamma_min: 0.4
            gamma_max: 0.9
            lambda_commit: 1.0
            alpha_struct: 0.3
            d2_evidence_nu: 1.0
            d2_evidence_ttl_steps: 5
            d2_evidence_requires_benefit: true
            zscore_epsilon: 1.0e-6
            same_refresh_grace: true
            final_freeze_steps: 1
          attribution:
            write_paired_counterfactual: true
            write_independent_delta_R_i: true
            productive_delta_logp: 0.5
          controls:
            allow_wrong_allele_head: true
            allow_shuffled_head: true
        """
    )
    path = tmp_path / "controller.yaml"
    path.write_text(body)
    return path


def _make_args(tmp_path: Path, ctrl_yaml: Path, *, global_pressure_calibration_json=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        controller_config=str(ctrl_yaml),
        head_checkpoint=str(tmp_path / "head.ckpt"),
        head_config_dir=str(tmp_path / "cfgs"),
        head_variant_id="V1",
        head_device="cpu",
        head_window_batch_size=64,
        head_allele_idx=0,
        allele="DRB1*01:01",
        global_pressure_calibration_json=global_pressure_calibration_json,
    )


def _fake_head_dir(tmp_path: Path) -> None:
    head_ckpt = tmp_path / "head.ckpt"
    head_ckpt.write_bytes(b"head-fake-bytes")
    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    (cfg_dir / "model.yaml").write_text("m: 1\n")
    (cfg_dir / "model_ablation.yaml").write_text("a: 1\n")
    (cfg_dir / "inference.yaml").write_text("i: 1\n")


def test_d2_d3_manifest_contains_surface_version_and_per_mode_configs(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_d2_d3_yaml(tmp_path)
    args = _make_args(tmp_path, ctrl_yaml)
    setup = load_controller_setup(args)
    assert setup is not None
    manifest = d1_manifest_provenance(
        setup,
        static_cache_path=tmp_path / "cache.parquet",
        static_cache_meta_path=tmp_path / "cache.meta.json",
        window_k_min=12,
        window_k_max=25,
    )
    assert manifest["controller_mode"] == "d2_d3_full"
    assert manifest["controller_surface_version"] == CONTROLLER_SURFACE_VERSION == 3
    # D2/D3/attribution/controls config blocks present as top-level views.
    assert manifest["d2_config"]["enabled"] is True
    assert manifest["d2_config"]["beta"] == 1.0
    assert manifest["d3_config"]["enabled"] is True
    assert manifest["d3_config"]["final_freeze_steps"] == 1
    assert manifest["attribution_config"]["productive_delta_logp"] == 0.5
    assert manifest["controls_config"]["allow_shuffled_head"] is True


def test_per_protein_summary_aggregates_d2_d3_event_counts(tmp_path: Path):
    """F5 contract: D2 / D3 event rows must drive the summary aggregates."""
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_d2_d3_yaml(tmp_path)
    args = _make_args(tmp_path, ctrl_yaml)
    setup = load_controller_setup(args)
    assert setup is not None

    d2_rows = [
        {
            "event_type": "D2",
            "block_id": 0,
            "refresh_step": 0,
            "position_i": 4,
            "kl_struct_corrected": 0.3,
        },
        {
            "event_type": "D2",
            "block_id": 0,
            "refresh_step": 0,
            "position_i": 5,
            "kl_struct_corrected": 0.4,
        },
        {
            "event_type": "D2",
            "block_id": 1,
            "refresh_step": 1,
            "position_i": 7,
            "kl_struct_corrected": 0.2,
        },
    ]
    d3_rows = [
        {"event_type": "D3", "position_i": 4, "grace_flag": True},
        {"event_type": "D3", "position_i": 4, "grace_flag": False},
        {"event_type": "D3", "position_i": 8, "grace_flag": False},
    ]
    # Construct two refresh records carrying three active blocks total so
    # candidate_feasibility_rate = 2 distinct (refresh_step, block_id) pairs / 3 blocks.
    from inverse_folding.reference_flow.controller import ActiveBlock

    def _blk(rid: int) -> ActiveBlock:
        return ActiveBlock(
            block_id=int(rid),
            residue_start_0b=0,
            residue_end_0b=5,
            window_indices=(0,),
            g_time=1.0,
            g_comp=1.0,
            g_ent=1.0,
            g_ESS=1.0,
            rho_B=1.0,
            completion_fraction=1.0,
            mean_struct_entropy=0.5,
        )

    refresh_records = [
        D1RefreshRecord(
            protein_id="P1", design_idx=0, seed=42, refresh_step=0, step=0, t=0.5,
            r_windows_dyn=(), r_windows_static=(), window_excess=(),
            active_blocks=(_blk(0), _blk(1)),
            new_hotspot_count=0, completion_fraction_global=0.5,
            mean_struct_entropy_global=1.0, head_risk_LME=1.0, head_risk_max=1.0,
        ),
        D1RefreshRecord(
            protein_id="P1", design_idx=0, seed=42, refresh_step=1, step=5, t=0.6,
            r_windows_dyn=(), r_windows_static=(), window_excess=(),
            active_blocks=(_blk(1),),
            new_hotspot_count=0, completion_fraction_global=0.6,
            mean_struct_entropy_global=1.0, head_risk_LME=1.0, head_risk_max=1.0,
        ),
    ]
    # H4 contract: feasibility is read from refresh_addenda's
    # d2_block_diagnostics (canonical per-block flag). Construct addenda
    # with 2 feasible + 1 infeasible (low_ess) blocks → rate = 2/3.
    refresh_addenda_list = [
        {
            "d2_block_diagnostics": [
                {"block_id": 0, "candidate_feasibility": True},
                {"block_id": 1, "candidate_feasibility": False},
            ]
        },
        {
            "d2_block_diagnostics": [
                {"block_id": 1, "candidate_feasibility": True},
            ]
        },
    ]
    summary = compute_per_protein_summary(
        protein_id="P1",
        design_idx=0,
        seed=42,
        allele="DRB1*01:01",
        arm=setup.config.mode,
        refresh_records=refresh_records,
        event_rows=d2_rows + d3_rows,
        controller_config=setup.config,
        refresh_addenda=refresh_addenda_list,
    )
    assert summary["total_D2_events"] == 3
    assert summary["total_D3_events"] == 3
    assert summary["total_corrected_positions"] == 3
    assert summary["total_recommits"] == 3
    assert summary["total_KL_budget"] == pytest.approx(0.9, abs=1e-9)
    # 2 feasible / 3 total scanned blocks (canonical d2_block_diagnostics path).
    assert summary["candidate_feasibility_rate"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    # Position 4 remasked twice in D3 rows → 1 churned / 2 unique positions = 0.5.
    assert summary["churn_rate"] == pytest.approx(0.5, abs=1e-9)
    # 1 grace_flag=True in D3 rows / 3 D2 corrected positions = 1/3.
    assert summary["same_refresh_conflict_rate"] == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_per_protein_summary_aggregates_stage_a_actuation_rates(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_d2_d3_yaml(tmp_path)
    setup = load_controller_setup(_make_args(tmp_path, ctrl_yaml))
    assert setup is not None
    d2_rows = [
        {
            "event_type": "D2",
            "position_i": 4,
            "sticky_age_steps": 0,
            "sticky_selected_flag": True,
            "paired_disagreement_flag": True,
            "realized_benefit_flag": True,
            "rank_percentile_d2_written": 0.9,
            "kl_struct_corrected": 0.1,
        },
        {
            "event_type": "D2",
            "position_i": 5,
            "sticky_age_steps": 1,
            "sticky_selected_flag": True,
            "paired_disagreement_flag": True,
            "realized_benefit_flag": False,
            "rank_percentile_d2_written": 0.5,
            "kl_struct_corrected": 0.2,
        },
        {
            "event_type": "D2",
            "position_i": 6,
            "sticky_age_steps": 1,
            "sticky_selected_flag": False,
            "paired_disagreement_flag": None,
            "realized_benefit_flag": None,
            "rank_percentile_d2_written": None,
            "kl_struct_corrected": 0.0,
        },
    ]
    d3_rows = [
        {"event_type": "D3", "position_i": 5, "remask_flag": True, "grace_flag": False}
    ]
    addenda = [
        {
            "d2_block_diagnostics": [
                {
                    "g_ESS_candidates": 0.0,
                    "ensemble_delta_R_std": 0.4,
                    "ensemble_sign_consistency": 0.75,
                    "argmax_to_ensemble_rank_flip_rate": 0.25,
                    "candidate_feasibility": True,
                },
                {
                    "g_ESS_candidates": 1.0,
                    "ensemble_delta_R_std": 0.2,
                    "ensemble_sign_consistency": 1.0,
                    "argmax_to_ensemble_rank_flip_rate": 0.0,
                    "candidate_feasibility": False,
                },
            ]
        }
    ]
    summary = compute_per_protein_summary(
        protein_id="P1",
        design_idx=0,
        seed=42,
        allele="DRB1*01:01",
        arm=setup.config.mode,
        refresh_records=[],
        event_rows=d2_rows + d3_rows,
        controller_config=setup.config,
        refresh_addenda=addenda,
    )
    assert summary["selected_after_d2_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["paired_disagreement_rate"] == pytest.approx(1.0)
    assert summary["realized_benefit_rate"] == pytest.approx(0.5)
    assert summary["final_persistence_rate"] == pytest.approx(0.5)
    assert summary["stage_a_rates_by_sticky_age"]["1"]["selected_after_d2_rate"] == pytest.approx(0.5)
    assert summary["rank_percentile_d2_written_median"] == pytest.approx(0.7)
    assert summary["argmax_to_ensemble_rank_flip_rate"] == pytest.approx(0.125)
    assert summary["ensemble_delta_R_std_mean"] == pytest.approx(0.3)
    assert summary["ensemble_sign_consistency_mean"] == pytest.approx(0.875)
    assert summary["g_ESS_suppression_rate"] == pytest.approx(0.5)


def test_stage_a_selected_after_d2_rate_uses_per_correction_denominator(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_d2_d3_yaml(tmp_path)
    setup = load_controller_setup(_make_args(tmp_path, ctrl_yaml))
    assert setup is not None
    # One sticky correction delivered over three steps; selected once at age 2.
    d2_rows = [
        {
            "event_type": "D2",
            "refresh_step": 0,
            "position_i": 4,
            "sticky_created_step": 5,
            "sticky_age_steps": 0,
            "sticky_selected_flag": False,
            "kl_struct_corrected": 0.1,
        },
        {
            "event_type": "D2",
            "refresh_step": 0,
            "position_i": 4,
            "sticky_created_step": 5,
            "sticky_age_steps": 1,
            "sticky_selected_flag": False,
            "kl_struct_corrected": 0.1,
        },
        {
            "event_type": "D2",
            "refresh_step": 0,
            "position_i": 4,
            "sticky_created_step": 5,
            "sticky_age_steps": 2,
            "sticky_selected_flag": True,
            "paired_disagreement_flag": True,
            "realized_benefit_flag": True,
            "kl_struct_corrected": 0.1,
        },
    ]
    summary = compute_per_protein_summary(
        protein_id="P1",
        design_idx=0,
        seed=42,
        allele="DRB1*01:01",
        arm=setup.config.mode,
        refresh_records=[],
        event_rows=d2_rows,
        controller_config=setup.config,
    )
    assert summary["selected_after_d2_rate"] == pytest.approx(1.0)
    assert summary["stage_a_rates_by_sticky_age"]["2"]["selected_after_d2_rate"] == pytest.approx(1.0)


def test_stage_a_final_persistence_uses_remask_ledger_not_only_d3_rows(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_d2_d3_yaml(tmp_path)
    setup = load_controller_setup(_make_args(tmp_path, ctrl_yaml))
    assert setup is not None
    d2_rows = [
        {
            "event_type": "D2",
            "refresh_step": 0,
            "position_i": 4,
            "sticky_created_step": 5,
            "sticky_age_steps": 0,
            "sticky_selected_flag": True,
            "paired_disagreement_flag": True,
            "realized_benefit_flag": True,
            "kl_struct_corrected": 0.1,
        },
        {
            "event_type": "D2",
            "refresh_step": 0,
            "position_i": 5,
            "sticky_created_step": 5,
            "sticky_age_steps": 0,
            "sticky_selected_flag": True,
            "paired_disagreement_flag": True,
            "realized_benefit_flag": True,
            "kl_struct_corrected": 0.1,
        },
    ]
    remask_rows = [
        {
            "event_type": "remask",
            "position_i": 5,
            "remask_flag": True,
            "grace_flag": False,
        }
    ]
    summary = compute_per_protein_summary(
        protein_id="P1",
        design_idx=0,
        seed=42,
        allele="DRB1*01:01",
        arm=setup.config.mode,
        refresh_records=[],
        event_rows=d2_rows + remask_rows,
        controller_config=setup.config,
    )
    assert summary["final_persistence_rate"] == pytest.approx(0.5)
    assert summary["total_recommits"] == 1


def test_per_protein_summary_aggregates_productive_revisit_outcomes(tmp_path: Path):
    """G2 contract: productive_revisit_* fields come from the controller's
    resolved snapshots, NOT a hard-coded null."""
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_d2_d3_yaml(tmp_path)
    args = _make_args(tmp_path, ctrl_yaml)
    setup = load_controller_setup(args)
    assert setup is not None
    outcomes = [
        {"immune_only": True, "structure_only": True, "joint": True},
        {"immune_only": True, "structure_only": False, "joint": False},
        {"immune_only": False, "structure_only": True, "joint": False},
        {"immune_only": True, "structure_only": True, "joint": True},
    ]
    summary = compute_per_protein_summary(
        protein_id="P1",
        design_idx=0,
        seed=42,
        allele="DRB1*01:01",
        arm=setup.config.mode,
        refresh_records=[],
        event_rows=[],
        controller_config=setup.config,
        productive_revisit_outcomes=outcomes,
    )
    assert summary["productive_revisit_immune_only"] == pytest.approx(0.75)
    assert summary["productive_revisit_structure_only"] == pytest.approx(0.75)
    assert summary["productive_revisit_joint"] == pytest.approx(0.5)


def test_per_protein_summary_productive_revisit_null_when_no_outcomes(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_d2_d3_yaml(tmp_path)
    args = _make_args(tmp_path, ctrl_yaml)
    setup = load_controller_setup(args)
    summary = compute_per_protein_summary(
        protein_id="P1",
        design_idx=0,
        seed=42,
        allele="DRB1*01:01",
        arm=setup.config.mode,
        refresh_records=[],
        event_rows=[],
        controller_config=setup.config,
        productive_revisit_outcomes=None,
    )
    assert summary["productive_revisit_immune_only"] is None
    assert summary["productive_revisit_structure_only"] is None
    assert summary["productive_revisit_joint"] is None


def test_per_protein_summary_arm_matches_controller_mode(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_d2_d3_yaml(tmp_path)
    args = _make_args(tmp_path, ctrl_yaml)
    setup = load_controller_setup(args)
    assert setup is not None
    summary = compute_per_protein_summary(
        protein_id="P1",
        design_idx=0,
        seed=42,
        allele="DRB1*01:01",
        arm=setup.config.mode,
        refresh_records=[],
        event_rows=[],
        controller_config=setup.config,
    )
    assert summary["arm"] == "d2_d3_full"


def test_refresh_log_includes_nullable_d3_fields_for_monitor_only(tmp_path: Path):
    """Without an addendum the refresh log carries explicit null e_i/m_i fields."""
    record = D1RefreshRecord(
        protein_id="P1",
        design_idx=0,
        seed=42,
        refresh_step=0,
        step=0,
        t=0.5,
        r_windows_dyn=(WindowRiskRecord(0, 5, 5, 1.0),),
        r_windows_static=(WindowRiskRecord(0, 5, 5, 0.0),),
        window_excess=(1.0,),
        active_blocks=(),
        new_hotspot_count=0,
        completion_fraction_global=0.5,
        mean_struct_entropy_global=1.0,
        head_risk_LME=1.0,
        head_risk_max=1.0,
    )
    write_d1_artifacts(
        run_dir=tmp_path,
        refresh_records_all=[record],
        event_rows_all=[],
        per_protein_summaries=[],
    )
    lines = (tmp_path / "refresh_log.jsonl").read_text().strip().splitlines()
    parsed = json.loads(lines[0])
    for key in ("e_i", "m_i", "rho_i", "grace_positions"):
        assert key in parsed
        assert parsed[key] is None


def test_empty_controller_events_parquet_has_stage_a_columns(tmp_path: Path):
    write_d1_artifacts(
        run_dir=tmp_path,
        refresh_records_all=[],
        event_rows_all=[],
        per_protein_summaries=[],
    )
    df = pd.read_parquet(tmp_path / "controller_events.parquet")
    for col in (
        "sticky_age_steps",
        "sticky_created_step",
        "sticky_selected_flag",
        "realized_benefit_flag",
        "d2_evidence",
        "rank_score",
        "rank_percentile_d2_written",
        "ell_struct",
        "struct_top1_token",
        "struct_top1_logprob",
        "struct_top2_token",
        "struct_top2_logprob",
        "struct_top2_gap",
        "struct_top3_token",
        "struct_top3_logprob",
        "struct_top3_gap",
        "struct_top4_token",
        "struct_top4_logprob",
        "struct_top4_gap",
        "ell_sample",
        "alpha_struct",
        "d2_evidence_nu",
        "delta_struct",
        "struct_temperature",
        "context_pnll",
        "g_pnll",
        "context_jsd",
        "g_stability",
        "ensemble_delta_R_std",
        "ensemble_sign_consistency",
        "ESS_candidates",
    ):
        assert col in df.columns


def test_refresh_log_merges_d3_addendum_when_provided(tmp_path: Path):
    record = D1RefreshRecord(
        protein_id="P1",
        design_idx=0,
        seed=42,
        refresh_step=0,
        step=5,
        t=0.5,
        r_windows_dyn=(WindowRiskRecord(0, 5, 5, 1.0),),
        r_windows_static=(WindowRiskRecord(0, 5, 5, 0.0),),
        window_excess=(1.0,),
        active_blocks=(),
        new_hotspot_count=0,
        completion_fraction_global=0.5,
        mean_struct_entropy_global=1.0,
        head_risk_LME=1.0,
        head_risk_max=1.0,
    )
    addenda = {
        ("P1", 0, 0): {
            "e_i": [0.0, 0.5, 0.5, 0.0],
            "m_i": [0.0, 0.5, 0.5, 0.0],
            "rho_i": [0.0, 0.7, 0.7, 0.0],
            "grace_positions": [1, 2],
        }
    }
    write_d1_artifacts(
        run_dir=tmp_path,
        refresh_records_all=[record],
        event_rows_all=[],
        per_protein_summaries=[],
        refresh_addenda_by_key=addenda,
    )
    lines = (tmp_path / "refresh_log.jsonl").read_text().strip().splitlines()
    parsed = json.loads(lines[0])
    assert parsed["e_i"] == [0.0, 0.5, 0.5, 0.0]
    assert parsed["m_i"] == [0.0, 0.5, 0.5, 0.0]
    assert parsed["grace_positions"] == [1, 2]


def test_refresh_log_merges_d2_block_diagnostics_when_provided(tmp_path: Path):
    """F6 contract: per-block D2 diagnostics + refresh-level delta_logit_max
    end up in the serialized refresh_log entry."""
    record = D1RefreshRecord(
        protein_id="P1",
        design_idx=0,
        seed=42,
        refresh_step=0,
        step=5,
        t=0.5,
        r_windows_dyn=(WindowRiskRecord(0, 5, 5, 1.0),),
        r_windows_static=(WindowRiskRecord(0, 5, 5, 0.0),),
        window_excess=(1.0,),
        active_blocks=(),
        new_hotspot_count=0,
        completion_fraction_global=0.5,
        mean_struct_entropy_global=1.0,
        head_risk_LME=1.0,
        head_risk_max=1.0,
    )
    addenda = {
        ("P1", 0, 0): {
            "d2_block_diagnostics": [
                {
                    "block_id": 0,
                    "candidate_mode": "cartesian",
                    "candidate_count": 16,
                    "candidate_feasibility": True,
                    "best_delta_R_B": -0.5,
                    "mean_delta_R_B": 0.1,
                    "ESS_B_candidates": 8.5,
                    "g_ESS_candidates": 1.0,
                    "rho_B_effective": 0.7,
                    "corrected_positions": [4, 5],
                    "skipped_reason": None,
                    "delta_logit_max_block": 1.2,
                }
            ],
            "delta_logit_max_refresh": 1.2,
        }
    }
    write_d1_artifacts(
        run_dir=tmp_path,
        refresh_records_all=[record],
        event_rows_all=[],
        per_protein_summaries=[],
        refresh_addenda_by_key=addenda,
    )
    parsed = json.loads(
        (tmp_path / "refresh_log.jsonl").read_text().strip().splitlines()[0]
    )
    assert parsed["delta_logit_max_refresh"] == pytest.approx(1.2)
    assert isinstance(parsed["d2_block_diagnostics"], list)
    diag = parsed["d2_block_diagnostics"][0]
    assert diag["candidate_count"] == 16
    assert diag["candidate_feasibility"] is True
    assert diag["best_delta_R_B"] == pytest.approx(-0.5)
    assert diag["ESS_B_candidates"] == pytest.approx(8.5)
    assert diag["corrected_positions"] == [4, 5]


def test_startup_prints_resolved_controller_config(tmp_path: Path, capsys):
    """Smoke test the print contract: every top-level controller key on its own line."""
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_d2_d3_yaml(tmp_path)
    args = _make_args(tmp_path, ctrl_yaml)
    setup = load_controller_setup(args)
    assert setup is not None

    from inverse_folding.reference_flow.controller_config import (
        controller_config_to_dict,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        resolved = controller_config_to_dict(setup.config)
        print("[controller] resolved controller config:", flush=True)
        for key, val in resolved.items():
            print(f"[controller]   {key}: {val}", flush=True)
        print(
            f"[controller] surface_version={CONTROLLER_SURFACE_VERSION} "
            f"mode={setup.config.mode}",
            flush=True,
        )
    out = buf.getvalue()
    assert "[controller] resolved controller config:" in out
    assert "d2_d3_full" in out
    assert "surface_version=3" in out
    # Every top-level key appears on its own line.
    for top in ("d2", "d3", "attribution", "controls", "head", "reliability"):
        assert f"  {top}:" in out


# ---------------------------------------------------------------------------
# Stage B telemetry + manifest (PLAN_RF_UNI_CTRL.md Task B6)
# ---------------------------------------------------------------------------


def test_stageB_manifest_contains_targeting_config_and_hash(tmp_path: Path):
    _fake_head_dir(tmp_path)
    args = _make_args(tmp_path, _STAGEB_PRESET)
    setup = load_controller_setup(args)
    assert setup is not None
    manifest = d1_manifest_provenance(
        setup,
        static_cache_path=tmp_path / "cache.parquet",
        static_cache_meta_path=tmp_path / "cache.meta.json",
        window_k_min=12,
        window_k_max=25,
    )
    assert manifest["targeting_config"]["mode"] == "typed_actionability"
    assert manifest["global_pressure_config"]["enabled"] is False
    assert manifest["targeting_config_hash"]
    assert manifest["d3_config"]["evidence_source"] == "typed_fresh"


# ---------------------------------------------------------------------------
# Stage C.1 global-pressure calibration wiring (PLAN_RF_UNI_CTRL.md C1.3)
# ---------------------------------------------------------------------------


def _write_stage_c_yaml(tmp_path: Path) -> Path:
    """Stage C.1 config (G10): typed targeting + legacy_window_excess D3 (the base
    d2/d3 block defaults to legacy) + global_pressure ENABLED with the thresholded
    driver, but with tau_prom/B_low/B_high omitted — they are stamped from the
    calibration JSON at load time (PLAN C1.3)."""
    base = _write_d2_d3_yaml(tmp_path).read_text()
    # Explicit 2-space indent so targeting/global_pressure nest under controller:
    # (the d2/d3 sections in the base sit at 2 spaces, fields at 4).
    extra = (
        "  targeting:\n"
        "    mode: typed_actionability\n"
        "    within_block_source: v_target\n"
        "  global_pressure:\n"
        "    enabled: true\n"
        "    pressure_source: trajectory_thresholded_G\n"
        "    tau_prom_source: calibration_json\n"
        "    mapping: smoothstep\n"
        "    g_min: 0.0\n"
        "    g_max: 1.0\n"
        "    min_reliable_refreshes: 1\n"
        "    unready_g: 0.0\n"
        "    scale_beta: true\n"
        "    scale_lambda: true\n"
    )
    path = tmp_path / "controller_stage_c.yaml"
    path.write_text(base + extra)
    return path


def _write_calibration_json(
    tmp_path, *, B_low=1.33, B_high=1.97, tau_prom=11.75, name="calib.json",
    drop_high=False, drop_tau=False, pressure_source="trajectory_thresholded_G",
):
    payload = {
        "schema_version": "stageC1_global_pressure_calibration.v2",
        "pressure_source": pressure_source,
        "low_quantile": 1.0 / 3.0,
        "high_quantile": 2.0 / 3.0,
        "B_low": B_low,
        "n_units": 50,
        "source_artifact": "actionability_residue_rows.parquet",
    }
    if not drop_high:
        payload["B_high"] = B_high
    if not drop_tau:
        payload["tau_prom"] = tau_prom
    p = tmp_path / name
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return p


def test_stage_c_calibration_stamps_band_and_provenance(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_stage_c_yaml(tmp_path)
    calib = _write_calibration_json(tmp_path, B_low=1.33, B_high=1.97, tau_prom=11.75)
    setup = load_controller_setup(
        _make_args(tmp_path, ctrl_yaml, global_pressure_calibration_json=str(calib))
    )
    assert setup is not None
    # tau_prom + band stamped into the config from the JSON.
    assert setup.config.global_pressure.enabled is True
    assert setup.config.global_pressure.pressure_source == "trajectory_thresholded_G"
    assert setup.config.global_pressure.tau_prom == 11.75
    assert setup.config.global_pressure.B_low == 1.33
    assert setup.config.global_pressure.B_high == 1.97
    assert setup.global_pressure_calibration_path == calib.resolve()
    assert setup.global_pressure_calibration_hash
    # Manifest carries flat accessors + the calibration provenance.
    manifest = d1_manifest_provenance(
        setup,
        static_cache_path=None,
        static_cache_meta_path=None,
        window_k_min=12,
        window_k_max=25,
    )
    assert manifest["global_pressure_config"]["enabled"] is True
    assert manifest["global_pressure_config"]["tau_prom"] == 11.75
    assert manifest["global_pressure_B_low"] == 1.33
    assert manifest["global_pressure_B_high"] == 1.97
    assert manifest["global_pressure_tau_prom"] == 11.75
    assert manifest["global_pressure_pressure_source"] == "trajectory_thresholded_G"
    assert manifest["global_pressure_calibration_path"] == str(calib.resolve())
    assert manifest["global_pressure_calibration_hash"] == setup.global_pressure_calibration_hash


def test_stage_c_calibration_missing_tau_prom_fails_fast(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_stage_c_yaml(tmp_path)
    bad = _write_calibration_json(tmp_path, drop_tau=True, name="notau.json")
    with pytest.raises(SystemExit):
        load_controller_setup(
            _make_args(tmp_path, ctrl_yaml, global_pressure_calibration_json=str(bad))
        )


def test_stage_c_calibration_wrong_pressure_source_fails_fast(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_stage_c_yaml(tmp_path)
    bad = _write_calibration_json(
        tmp_path, pressure_source="trajectory_median_G", name="legacy.json"
    )
    with pytest.raises(SystemExit):
        load_controller_setup(
            _make_args(tmp_path, ctrl_yaml, global_pressure_calibration_json=str(bad))
        )


def test_stage_c_band_is_reflected_in_config_hash(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_stage_c_yaml(tmp_path)
    setup_a = load_controller_setup(
        _make_args(
            tmp_path, ctrl_yaml,
            global_pressure_calibration_json=str(
                _write_calibration_json(tmp_path, B_low=0.02, B_high=0.10, name="a.json")
            ),
        )
    )
    setup_b = load_controller_setup(
        _make_args(
            tmp_path, ctrl_yaml,
            global_pressure_calibration_json=str(
                _write_calibration_json(tmp_path, B_low=0.05, B_high=0.20, name="b.json")
            ),
        )
    )
    assert setup_a.config_hash != setup_b.config_hash


def test_stage_c_enabled_requires_calibration_flag(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_stage_c_yaml(tmp_path)
    with pytest.raises(SystemExit):
        load_controller_setup(_make_args(tmp_path, ctrl_yaml))  # no calibration json


def test_stage_c_calibration_missing_key_fails_fast(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_stage_c_yaml(tmp_path)
    bad = _write_calibration_json(tmp_path, B_low=0.02, drop_high=True, name="bad.json")
    with pytest.raises(SystemExit):
        load_controller_setup(
            _make_args(tmp_path, ctrl_yaml, global_pressure_calibration_json=str(bad))
        )


def test_stage_c_calibration_inverted_band_fails_fast(tmp_path: Path):
    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_stage_c_yaml(tmp_path)
    bad = _write_calibration_json(tmp_path, B_low=0.20, B_high=0.10, name="inv.json")
    with pytest.raises(SystemExit):
        load_controller_setup(
            _make_args(tmp_path, ctrl_yaml, global_pressure_calibration_json=str(bad))
        )


def test_disabled_global_pressure_ignores_calibration_flag(tmp_path: Path):
    # The Stage B preset keeps global_pressure disabled; passing a calibration
    # JSON is harmless and is NOT stamped; path/hash are manifest provenance only.
    _fake_head_dir(tmp_path)
    calib = _write_calibration_json(tmp_path, B_low=0.02, B_high=0.10)
    setup = load_controller_setup(
        _make_args(tmp_path, _STAGEB_PRESET, global_pressure_calibration_json=str(calib))
    )
    assert setup is not None
    assert setup.config.global_pressure.enabled is False
    assert setup.global_pressure_calibration_path == calib.resolve()
    assert setup.global_pressure_calibration_hash
    assert setup.config.global_pressure.B_low is None
    assert setup.config.global_pressure.B_high is None

    manifest = d1_manifest_provenance(
        setup,
        static_cache_path=None,
        static_cache_meta_path=None,
        window_k_min=12,
        window_k_max=25,
    )
    assert manifest["global_pressure_config"]["enabled"] is False
    assert manifest["global_pressure_config"]["B_low"] is None
    assert manifest["global_pressure_calibration_path"] == str(calib.resolve())
    assert manifest["global_pressure_calibration_hash"] == setup.global_pressure_calibration_hash


def test_actionability_state_records_explode_to_residue_rows_and_summary():
    L = 4
    state = _make_actionability_state(L)
    rows, summary = _actionability_state_to_records(
        state, protein_id="P1", design_idx=0, seed=42,
        targeting_mode="typed_actionability", tau_ref_source="static_median",
        d3_evidence_source="typed_fresh",
    )
    assert len(rows) == L
    # required telemetry columns present
    for col in (
        "protein_id", "design_idx", "seed", "refresh_step", "t", "residue_index_0b",
        "tau_ref_B", "h_cur", "b_cur", "b_env", "env_peak", "env_consistency",
        "env_coverage_flag", "b_mem", "e_fresh", "r_ctx", "context_pnll",
        "g_time_pre", "g_comp_pre", "g_ent_pre", "g_pnll_pre", "g_stability_pre",
        "v_target", "u_pressure", "active_target_flag", "cluster_support",
        "legacy_residue_excess", "active_block_id", "d3_fresh_input_flag",
    ):
        assert col in rows[0]
    assert rows[0]["context_pnll"] is None  # nan → null
    assert rows[0]["d3_fresh_input_flag"] is True
    assert summary["targeting_mode"] == "typed_actionability"
    assert summary["d3_evidence_source"] == "typed_fresh"
    assert summary["num_env_head_calls"] == 3
    assert summary["num_actionable_windows_pre_cap"] == 6
    assert summary["G"] == pytest.approx(0.041)
    # Stage C.1 pressure columns present in the summary (None when pressure off,
    # as in this default state) so the calibration command + burden analysis can
    # consume them (PLAN_RF_UNI_CTRL.md C1.4).
    for col in (
        "B_GR", "g_GR_effective", "pressure_burden_bin", "pressure_reliable",
        "beta_base", "beta_eff", "lambda_base", "lambda_eff",
    ):
        assert col in summary
    assert summary["B_GR"] is None
    assert summary["g_GR_effective"] is None


def test_write_actionability_artifacts_emits_residues_and_summary(tmp_path: Path):
    state = _make_actionability_state(4)
    rows, summary = _actionability_state_to_records(
        state, protein_id="P1", design_idx=0, seed=42,
        targeting_mode="typed_actionability", tau_ref_source="static_median",
        d3_evidence_source="typed_fresh",
    )
    write_actionability_artifacts(run_dir=tmp_path, residue_rows=rows, summaries=[summary])
    df = pd.read_parquet(tmp_path / "actionability_residues.parquet")
    assert len(df) == 4
    assert set(df["residue_index_0b"]) == {0, 1, 2, 3}
    summ_lines = (tmp_path / "actionability_refresh_summary.jsonl").read_text().splitlines()
    assert len(summ_lines) == 1
    assert json.loads(summ_lines[0])["max_v_target"] == pytest.approx(3 * 0.25)


# ---------------------------------------------------------------------------
# SC-GR monitor telemetry + manifest (PLAN_RF_SC_GR.md Task SC0.4)
# ---------------------------------------------------------------------------


def _sc_gr_sample_row(arm: str, idx: int) -> dict:
    return {
        "protein_id": "P1", "design_idx": 0, "seed": 42, "refresh_step": 0,
        "step": 10, "t": 0.5, "arm": arm, "sample_idx": idx,
        "sequence_md5": "abc", "num_masked": 4, "num_reused_from_prev": 0,
        "reuse_fraction": 0.0, "mean_prev_confidence_reused": 0.0,
        "mean_sample_entropy": 1.0, "state_bootstrap_flag": arm == "self_conditioned",
        "G_mean_excess": 1.0, "G_topm_lse": 3.0, "G_supra_mass_tau_11p75": 0.0,
        "head_risk_LME": 2.0, "head_risk_max": 5.0,
    }


def _sc_gr_refresh_row(arm: str) -> dict:
    return {
        "protein_id": "P1", "design_idx": 0, "seed": 42, "refresh_step": 0,
        "step": 10, "t": 0.5, "arm": arm, "ensemble_size_effective": 2,
        "B_sc_mean_excess_median": 1.0, "B_sc_topm_lse_median": 3.0,
        "B_sc_supra_mass_tau_11p75_median": 0.0,
        "G_mean_excess_max": 1.0, "G_mean_excess_std": 0.0,
        "G_topm_lse_max": 3.0, "G_topm_lse_std": 0.0,
        "G_supra_mass_tau_11p75_max": 0.0, "G_supra_mass_tau_11p75_std": 0.0,
        "num_masked": 4, "reuse_fraction_mean": 0.0,
        "state_bootstrap_flag": arm == "self_conditioned",
        "old_argmax_G_mean_excess": 0.5, "old_argmax_G_topm_lse": 1.5,
        "old_argmax_G_supra_mass_tau_11p75": 0.0,
    }


def test_write_sc_gr_probe_artifacts_emits_both_sidecars(tmp_path: Path):
    sample_rows = [
        _sc_gr_sample_row("fresh", 0), _sc_gr_sample_row("fresh", 1),
        _sc_gr_sample_row("self_conditioned", 0), _sc_gr_sample_row("self_conditioned", 1),
    ]
    refresh_rows = [_sc_gr_refresh_row("fresh"), _sc_gr_refresh_row("self_conditioned")]
    write_sc_gr_probe_artifacts(
        run_dir=tmp_path, sample_rows=sample_rows, refresh_rows=refresh_rows
    )
    samples = pd.read_parquet(tmp_path / SC_GR_PROBE_SAMPLES_FILE)
    refresh = pd.read_parquet(tmp_path / SC_GR_PROBE_REFRESH_FILE)
    assert len(samples) == 4
    assert set(samples["arm"]) == {"fresh", "self_conditioned"}
    assert "G_supra_mass_tau_11p75" in samples.columns
    assert len(refresh) == 2
    assert "B_sc_supra_mass_tau_11p75_median" in refresh.columns
    assert "old_argmax_G_topm_lse" in refresh.columns


def test_write_sc_gr_probe_artifacts_empty_rows_schema_compatible(tmp_path: Path):
    write_sc_gr_probe_artifacts(run_dir=tmp_path, sample_rows=[], refresh_rows=[])
    samples = pd.read_parquet(tmp_path / SC_GR_PROBE_SAMPLES_FILE)
    refresh = pd.read_parquet(tmp_path / SC_GR_PROBE_REFRESH_FILE)
    assert len(samples) == 0
    assert len(refresh) == 0
    # base schema columns present so a downstream reader gets a typed frame
    assert "G_mean_excess" in samples.columns
    assert "arm" in samples.columns
    assert "B_sc_mean_excess_median" in refresh.columns
    assert "old_argmax_G_mean_excess" in refresh.columns


def test_manifest_contains_self_conditioned_gr_config(tmp_path: Path):
    _fake_head_dir(tmp_path)
    args = _make_args(tmp_path, _SCGR_MONITOR_PRESET)
    setup = load_controller_setup(args)
    assert setup is not None
    manifest = d1_manifest_provenance(
        setup,
        static_cache_path=tmp_path / "cache.parquet",
        static_cache_meta_path=tmp_path / "cache.meta.json",
        window_k_min=12,
        window_k_max=25,
    )
    sc = manifest["self_conditioned_gr_config"]
    assert sc["enabled"] is True
    assert sc["mode"] == "monitor_only"
    assert sc["arms"] == ("fresh", "self_conditioned")
    # Monitor preset keeps the pressure actuator off.
    assert manifest["global_pressure_config"]["enabled"] is False


def test_startup_print_includes_self_conditioned_gr(tmp_path: Path):
    _fake_head_dir(tmp_path)
    args = _make_args(tmp_path, _SCGR_MONITOR_PRESET)
    setup = load_controller_setup(args)
    assert setup is not None

    from inverse_folding.reference_flow.controller_config import (
        controller_config_to_dict,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        resolved = controller_config_to_dict(setup.config)
        print("[controller] resolved self_conditioned_gr config:", flush=True)
        for key, val in resolved["self_conditioned_gr"].items():
            print(f"[controller]   self_conditioned_gr.{key}: {val}", flush=True)
    out = buf.getvalue()
    assert "resolved self_conditioned_gr config:" in out
    for key in ("enabled", "mode", "arms", "ensemble_size", "supra_tau_values"):
        assert f"self_conditioned_gr.{key}:" in out


def test_refresh_log_includes_actionability_pointers_when_typed(tmp_path: Path):
    record = D1RefreshRecord(
        protein_id="P1", design_idx=0, seed=42, refresh_step=0, step=5, t=0.6,
        r_windows_dyn=(WindowRiskRecord(0, 5, 5, 1.0),),
        r_windows_static=(WindowRiskRecord(0, 5, 5, 0.0),),
        window_excess=(1.0,), active_blocks=(), new_hotspot_count=0,
        completion_fraction_global=0.5, mean_struct_entropy_global=1.0,
        head_risk_LME=1.0, head_risk_max=1.0,
    )
    write_d1_artifacts(
        run_dir=tmp_path,
        refresh_records_all=[record],
        event_rows_all=[],
        per_protein_summaries=[],
        actionability_constant_pointer={
            "actionability_residues_path": "actionability_residues.parquet",
            "actionability_refresh_summary_path": "actionability_refresh_summary.jsonl",
            "targeting_mode": "typed_actionability",
        },
        actionability_g_by_key={("P1", 0, 0): {"G": 0.041, "g_GR_diagnostic": 0.52}},
    )
    payload = json.loads((tmp_path / "refresh_log.jsonl").read_text().splitlines()[0])
    assert payload["actionability_residues_path"] == "actionability_residues.parquet"
    assert payload["actionability_refresh_summary_path"] == "actionability_refresh_summary.jsonl"
    assert payload["targeting_mode"] == "typed_actionability"
    assert payload["G"] == pytest.approx(0.041)


def test_refresh_log_omits_actionability_pointers_when_static(tmp_path: Path):
    # No actionability pointer → static-run refresh_log schema is unchanged.
    record = D1RefreshRecord(
        protein_id="P1", design_idx=0, seed=42, refresh_step=0, step=5, t=0.6,
        r_windows_dyn=(WindowRiskRecord(0, 5, 5, 1.0),),
        r_windows_static=(WindowRiskRecord(0, 5, 5, 0.0),),
        window_excess=(1.0,), active_blocks=(), new_hotspot_count=0,
        completion_fraction_global=0.5, mean_struct_entropy_global=1.0,
        head_risk_LME=1.0, head_risk_max=1.0,
    )
    write_d1_artifacts(
        run_dir=tmp_path, refresh_records_all=[record], event_rows_all=[],
        per_protein_summaries=[],
    )
    payload = json.loads((tmp_path / "refresh_log.jsonl").read_text().splitlines()[0])
    assert "actionability_residues_path" not in payload
    assert "targeting_mode" not in payload


# ---------------------------------------------------------------------------
# Stage B P2 fixes: static cache policy forwarding (PLAN_RF_UNI_CTRL.md §97)
# ---------------------------------------------------------------------------


def _write_read_only_head_yaml(tmp_path: Path) -> Path:
    body = dedent(
        """
        controller:
          enabled: true
          mode: monitor_only
          t_start: 0.5
          refresh_interval: 5
          completion:
            method: argmax
          head:
            score_scale: raw_logit
            static_cache_policy: read_only
            local_risk_aggregation: LME
          active_windows:
            excess_threshold: 0.0
            max_windows: 16
            selection: threshold_then_top_n
            merge_overlapping_scoring_windows: true
          reliability:
            time_k: 20.0
            entropy_h0: 1.5
            min_completion_fraction: 0.0
            min_rho_to_emit_event: 0.0
        """
    )
    path = tmp_path / "controller.yaml"
    path.write_text(body)
    return path


def test_make_head_scorer_forwards_config_static_cache_policy(tmp_path: Path, monkeypatch):
    import scripts.run_if_phase_c1 as driver

    _fake_head_dir(tmp_path)
    ctrl_yaml = _write_read_only_head_yaml(tmp_path)
    setup = load_controller_setup(_make_args(tmp_path, ctrl_yaml))
    assert setup is not None
    assert setup.config.head.static_cache_policy == "read_only"

    captured: dict = {}

    def _fake_build(setup_arg, **kw):
        captured.update(kw)
        return object()

    monkeypatch.setattr(driver, "build_head_scorer", _fake_build)
    driver._make_head_scorer(
        setup,
        window_k_min=12,
        window_k_max=25,
        static_cache_path=None,
        static_cache_meta_path=None,
    )
    # Driver must forward the config policy, not the hardcoded "lazy_write".
    assert captured["static_cache_policy"] == "read_only"
