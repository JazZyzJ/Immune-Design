"""HIMP0 config schema tests.

Validates train.near_positive and train.residue config blocks.

Backward-compat invariants:
  - Existing configs that omit near_positive / residue load unchanged with both
    blocks disabled by default.

New schema invariants:
  - Unknown keys inside near_positive / residue raise ValueError.
  - schedule names: v0 accepts {ignore, linear_clamp, sigmoid}; reserved
    names (exponential, thresholded_smooth) and unknown names rejected.
  - aggregation: v0 accepts {max, log_mean_exp}; topk_mean (reserved) rejected.
  - loss_mode: v0 accepts {pairwise_margin}; pairwise_bce (reserved) rejected.
  - window_mode: v0 accepts {sampled, all}.
  - label_mode: v0 accepts {binary_coverage}.
  - metric: v0 accepts {endpoint_gap}.
  - Type checks for ints / floats / bools / dicts.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from epitope_head.configs import load_train_config


# ── Fixture helpers ─────────────────────────────────────────────────────────

BASE_TRAIN_YAML = """\
train:
  neg_ratio: 7
  hard_negative_fraction: 0.3
  hard_neg_max_overlap_ratio: 0.8
  hard_neg_offset_range: 5
  neg_length_sampling: match_positive
  loss:
    tau: 1.0
    T_mp: 0.1
    lambda_mp: 0.0
    lambda_smooth: 0.0
  lr: 5.0e-4
  weight_decay: 1.0e-3
  optimizer: adamw
  scheduler: cosine
  warmup_steps: 500
  max_epochs: 50
  grad_clip: 1.0
  max_tokens: 4096
  num_workers: 4
  seed: 42
  deterministic: false
  checkpoint_every_n_epochs: 5
  early_stopping_patience: 15
  monitor_metric: pp_auc
  chunking:
    enabled: true
    context_len: 1022
    stride: 512
    margin: 32
    stitch_mode: per_residue_stitch
    window_mode: per_residue_stitch
    enable_reliability: true
"""


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "train.yaml"
    p.write_text(body)
    return p


def _augment(extra_train_block: str) -> str:
    """Append extra YAML keys under the existing train: block."""
    return BASE_TRAIN_YAML + textwrap.indent(extra_train_block, "  ")


# ── Backward-compat: omitted blocks default to disabled ────────────────────

class TestBackwardCompat:

    def test_baseline_yaml_loads_unchanged(self, tmp_path):
        path = _write_yaml(tmp_path, BASE_TRAIN_YAML)
        cfg = load_train_config(path)
        # Baseline keys still present
        assert cfg["neg_ratio"] == 7
        assert cfg["loss"]["tau"] == 1.0

    def test_near_positive_defaults_disabled(self, tmp_path):
        path = _write_yaml(tmp_path, BASE_TRAIN_YAML)
        cfg = load_train_config(path)
        np_cfg = cfg["near_positive"]
        assert np_cfg["enabled"] is False
        assert np_cfg["near_gap_max"] == 10
        assert np_cfg["schedule"] == "ignore"
        assert np_cfg["metric"] == "endpoint_gap"
        assert np_cfg["apply_to_span_negatives"] is True
        assert np_cfg["apply_to_residue_labels"] is True

    def test_residue_defaults_disabled(self, tmp_path):
        path = _write_yaml(tmp_path, BASE_TRAIN_YAML)
        cfg = load_train_config(path)
        r = cfg["residue"]
        assert r["enabled"] is False
        assert r["lambda_residue"] == 0.0
        assert r["aggregation"] == "log_mean_exp"
        assert r["loss_mode"] == "pairwise_margin"
        assert r["window_mode"] == "sampled"
        assert r["margin_m_residue"] == 0.5
        assert r["min_far_bg_residues"] == 4
        assert r["max_windows_per_chunk"] == 1024
        assert r["label_mode"] == "binary_coverage"


# ── near_positive validation ────────────────────────────────────────────────

class TestNearPositiveValidation:

    def test_explicit_enable_with_sigmoid(self, tmp_path):
        body = _augment(textwrap.dedent("""\
            near_positive:
              enabled: true
              near_gap_max: 10
              schedule: sigmoid
              schedule_params:
                center: 5.0
                slope: 1.0
              metric: endpoint_gap
              apply_to_span_negatives: true
              apply_to_residue_labels: true
            """))
        cfg = load_train_config(_write_yaml(tmp_path, body))
        assert cfg["near_positive"]["enabled"] is True
        assert cfg["near_positive"]["schedule"] == "sigmoid"
        assert cfg["near_positive"]["schedule_params"]["center"] == 5.0

    def test_unknown_schedule_rejected(self, tmp_path):
        body = _augment("near_positive:\n  enabled: true\n  schedule: bogus_schedule\n")
        with pytest.raises(ValueError, match=r"near_positive.schedule"):
            load_train_config(_write_yaml(tmp_path, body))

    @pytest.mark.parametrize("reserved", ["exponential", "thresholded_smooth"])
    def test_reserved_schedule_rejected_in_v0(self, tmp_path, reserved):
        body = _augment(f"near_positive:\n  enabled: true\n  schedule: {reserved}\n")
        with pytest.raises(ValueError, match=r"near_positive.schedule"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_unknown_key_rejected(self, tmp_path):
        body = _augment("near_positive:\n  enabled: true\n  bogus_key: 1\n")
        with pytest.raises(ValueError, match=r"near_positive.*unknown"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_unknown_metric_rejected(self, tmp_path):
        body = _augment("near_positive:\n  enabled: true\n  metric: iou_overlap\n")
        with pytest.raises(ValueError, match=r"near_positive.metric"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_negative_near_gap_max_rejected(self, tmp_path):
        body = _augment("near_positive:\n  enabled: true\n  near_gap_max: -1\n")
        with pytest.raises(ValueError, match=r"near_gap_max"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_non_dict_schedule_params_rejected(self, tmp_path):
        body = _augment("near_positive:\n  enabled: true\n  schedule_params: not_a_dict\n")
        with pytest.raises(ValueError, match=r"schedule_params"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_non_bool_apply_flags_rejected(self, tmp_path):
        body = _augment(
            "near_positive:\n  enabled: true\n  apply_to_span_negatives: 1\n"
        )
        with pytest.raises(ValueError, match=r"apply_to_span_negatives"):
            load_train_config(_write_yaml(tmp_path, body))


# ── residue validation ─────────────────────────────────────────────────────

class TestResidueValidation:

    def test_explicit_enable_full_block(self, tmp_path):
        body = _augment(textwrap.dedent("""\
            residue:
              enabled: true
              lambda_residue: 0.1
              label_mode: binary_coverage
              aggregation: log_mean_exp
              aggregation_params:
                beta: 1.0
              loss_mode: pairwise_margin
              margin_m_residue: 0.5
              window_mode: sampled
              max_windows_per_chunk: 1024
              min_far_bg_residues: 4
            """))
        cfg = load_train_config(_write_yaml(tmp_path, body))
        r = cfg["residue"]
        assert r["enabled"] is True
        assert r["lambda_residue"] == 0.1
        assert r["aggregation_params"]["beta"] == 1.0

    def test_unknown_aggregation_rejected(self, tmp_path):
        body = _augment("residue:\n  enabled: true\n  aggregation: median\n")
        with pytest.raises(ValueError, match=r"residue.aggregation"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_topk_mean_aggregation_rejected_in_v0(self, tmp_path):
        body = _augment("residue:\n  enabled: true\n  aggregation: topk_mean\n")
        with pytest.raises(ValueError, match=r"residue.aggregation"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_pairwise_bce_loss_mode_rejected_in_v0(self, tmp_path):
        body = _augment("residue:\n  enabled: true\n  loss_mode: pairwise_bce\n")
        with pytest.raises(ValueError, match=r"residue.loss_mode"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_unknown_window_mode_rejected(self, tmp_path):
        body = _augment("residue:\n  enabled: true\n  window_mode: greedy\n")
        with pytest.raises(ValueError, match=r"residue.window_mode"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_unknown_label_mode_rejected(self, tmp_path):
        body = _augment("residue:\n  enabled: true\n  label_mode: density\n")
        with pytest.raises(ValueError, match=r"residue.label_mode"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_unknown_key_rejected(self, tmp_path):
        body = _augment("residue:\n  enabled: true\n  bogus_residue_key: 1\n")
        with pytest.raises(ValueError, match=r"residue.*unknown"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_negative_lambda_residue_rejected(self, tmp_path):
        body = _augment("residue:\n  enabled: true\n  lambda_residue: -0.1\n")
        with pytest.raises(ValueError, match=r"lambda_residue"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_negative_margin_m_rejected(self, tmp_path):
        body = _augment("residue:\n  enabled: true\n  margin_m_residue: -0.5\n")
        with pytest.raises(ValueError, match=r"margin_m_residue"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_zero_max_windows_rejected(self, tmp_path):
        body = _augment("residue:\n  enabled: true\n  max_windows_per_chunk: 0\n")
        with pytest.raises(ValueError, match=r"max_windows_per_chunk"):
            load_train_config(_write_yaml(tmp_path, body))

    def test_negative_min_far_bg_rejected(self, tmp_path):
        body = _augment("residue:\n  enabled: true\n  min_far_bg_residues: -1\n")
        with pytest.raises(ValueError, match=r"min_far_bg_residues"):
            load_train_config(_write_yaml(tmp_path, body))


# ── Production yamls under epitope_head/configs/ still load ────────────────

class TestProductionConfigsStillLoad:

    @pytest.mark.parametrize("config_name", [
        "train.yaml",
    ])
    def test_existing_config_loads(self, config_name):
        cfg_dir = Path(__file__).resolve().parents[3] / "epitope_head" / "configs"
        cfg = load_train_config(cfg_dir / config_name)
        assert "near_positive" in cfg
        assert "residue" in cfg
        assert cfg["near_positive"]["enabled"] is False
        assert cfg["residue"]["enabled"] is False


# ── HIMP override config validates against the schema ──────────────────────

class TestCnnHimpV1OverrideConfig:

    def _deep_update(self, base: dict, patch: dict) -> dict:
        # Mirrors train_v2_ablation.py:185-191 _deep_update implementation.
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v
        return base

    def test_cnn_himp_v1_yaml_merges_and_passes_validation(self, tmp_path):
        from epitope_head.configs import (
            _validate_and_default_near_positive,
            _validate_and_default_residue,
        )
        cfg_dir = Path(__file__).resolve().parents[3] / "epitope_head" / "configs"
        # Start from the resolved train.yaml (with defaulted blocks).
        train_cfg = load_train_config(cfg_dir / "train.yaml")
        # Apply the HIMP override exactly the way train_v2_ablation.py does.
        with open(cfg_dir / "cnn_himp_v1.yaml") as f:
            override_raw = yaml.safe_load(f)
        override = override_raw.get("train", override_raw)
        self._deep_update(train_cfg, override)
        # Re-validate the merged near_positive / residue blocks.
        np_cfg = _validate_and_default_near_positive(train_cfg["near_positive"])
        r_cfg = _validate_and_default_residue(train_cfg["residue"])
        assert np_cfg["enabled"] is True
        assert np_cfg["schedule"] == "sigmoid"
        assert np_cfg["near_gap_max"] == 10
        assert r_cfg["enabled"] is True
        assert r_cfg["lambda_residue"] == 0.1
        assert r_cfg["aggregation"] == "log_mean_exp"
        assert r_cfg["window_mode"] == "sampled"


# ── Override revalidation gate (review fix 1) ──────────────────────────────

class TestValidateHimpTrainBlocks:
    """Public revalidation entry point — used by train_v2_ablation.py after
    its raw _deep_update merges --override-config into the resolved cfg."""

    def test_idempotent_on_already_valid_cfg(self, tmp_path):
        from epitope_head.configs import validate_himp_train_blocks
        path = _write_yaml(tmp_path, BASE_TRAIN_YAML)
        cfg = load_train_config(path)
        snapshot = (cfg["near_positive"].copy(), cfg["residue"].copy())
        validate_himp_train_blocks(cfg)
        assert cfg["near_positive"] == snapshot[0]
        assert cfg["residue"] == snapshot[1]

    def test_rejects_invalid_override_schedule(self, tmp_path):
        from epitope_head.configs import validate_himp_train_blocks
        cfg = load_train_config(_write_yaml(tmp_path, BASE_TRAIN_YAML))
        # Simulate a buggy --override-config merge that injected an
        # unsupported schedule name into the resolved cfg.
        cfg["near_positive"]["schedule"] = "bogus_schedule"
        with pytest.raises(ValueError, match="near_positive.schedule"):
            validate_himp_train_blocks(cfg)

    def test_rejects_invalid_override_aggregation(self, tmp_path):
        from epitope_head.configs import validate_himp_train_blocks
        cfg = load_train_config(_write_yaml(tmp_path, BASE_TRAIN_YAML))
        cfg["residue"]["aggregation"] = "topk_mean"  # reserved, not in v0
        with pytest.raises(ValueError, match="residue.aggregation"):
            validate_himp_train_blocks(cfg)

    def test_fills_defaults_when_block_missing(self):
        from epitope_head.configs import validate_himp_train_blocks
        # Cfg constructed without going through load_train_config (e.g. test
        # fixture or in-process patching) is still backfilled.
        cfg = {"near_positive": None, "residue": None}
        validate_himp_train_blocks(cfg)
        assert cfg["near_positive"]["enabled"] is False
        assert cfg["residue"]["enabled"] is False
