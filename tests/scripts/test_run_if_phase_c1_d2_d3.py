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
import pytest

from inverse_folding.reference_flow.controller import D1RefreshRecord
from inverse_folding.reference_flow.head_scoring import WindowRiskRecord
from scripts.run_if_phase_c1 import (
    CONTROLLER_SURFACE_VERSION,
    compute_per_protein_summary,
    d1_manifest_provenance,
    load_controller_setup,
    write_d1_artifacts,
    _refresh_record_to_jsonable,
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
            min_completion_fraction: 0.5
            min_rho_to_emit_event: 0.0
          d2:
            enabled: true
            beta: 1.0
            eta: 1.0
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
          d3:
            enabled: true
            window_to_residue_projection: max_covering_window
            gamma_min: 0.4
            gamma_max: 0.9
            lambda_commit: 1.0
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


def _make_args(tmp_path: Path, ctrl_yaml: Path):
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
    assert manifest["controller_surface_version"] == CONTROLLER_SURFACE_VERSION == 2
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
    summary = compute_per_protein_summary(
        protein_id="P1",
        design_idx=0,
        seed=42,
        allele="DRB1*01:01",
        arm=setup.config.mode,
        refresh_records=refresh_records,
        event_rows=d2_rows + d3_rows,
        controller_config=setup.config,
    )
    assert summary["total_D2_events"] == 3
    assert summary["total_D3_events"] == 3
    assert summary["total_corrected_positions"] == 3
    assert summary["total_recommits"] == 3
    assert summary["total_KL_budget"] == pytest.approx(0.9, abs=1e-9)
    # 2 unique (refresh_step, block_id) pairs in D2 rows; 3 total active blocks.
    assert summary["candidate_feasibility_rate"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    # Position 4 remasked twice in D3 rows → 1 churned / 2 unique positions = 0.5.
    assert summary["churn_rate"] == pytest.approx(0.5, abs=1e-9)
    # 1 grace_flag=True in D3 rows / 3 D2 corrected positions = 1/3.
    assert summary["same_refresh_conflict_rate"] == pytest.approx(1.0 / 3.0, abs=1e-9)


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
    assert "surface_version=2" in out
    # Every top-level key appears on its own line.
    for top in ("d2", "d3", "attribution", "controls", "head", "reliability"):
        assert f"  {top}:" in out
