"""Module K contract tests — baseline config validation for IF v1.

TDD gates from PLAN_IF.md Task K0:
  RED:   invalid baseline config passes schema validation.
  GREEN: schema validator rejects incomplete configs and accepts the frozen v1
         baseline profile only when required keys are present.
"""

import copy

import pytest

from inverse_folding.configs.schema import (
    BaselineConfigError,
    load_baseline_config,
    validate_baseline_config,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_config():
    """A complete, valid baseline config dict."""
    return {
        "baseline": {
            "dplm_repo": "github.com/bytedance/dplm",
            "dplm_commit": "abc1234",
            "checkpoint_id": "airkingbd/dplm_650m",
            "trainable_params_pattern": "adapter",
            "gvp_encoder": "esm_if1_gvp4_t16_142M_UR50",
            "gvp_frozen": True,
            "backbone_frozen": True,
            "dataset_root": "/scratch/data/cath_4.3",
            "dataset_name": "cath_4.3",
            "max_length": 500,
            "atoms": ["N", "CA", "C", "O"],
            "seed": 42,
            "diffusion_timesteps": 100,
            "noise_strategy": "random_mask",
            "max_steps": 200000,
            "lr": 1.0e-3,
            "warmup_steps": 4000,
            "max_tokens_per_batch": 6000,
            "artifact_root": "outputs/if/runs",
            "run_dir_template": "outputs/if/runs/dplm_v1_adapter/${run_id}",
            "required_run_artifacts": [
                "resolved_config.yaml",
                "checkpoints/",
                "baseline_validation.json",
                "failure_ledger.json",
            ],
        }
    }


# ── K0.1: Required keys must be present ──────────────────────────────────────

class TestRequiredKeys:

    REQUIRED_KEYS = [
        "checkpoint_id",
        "trainable_params_pattern",
        "dataset_root",
        "seed",
        "artifact_root",
    ]

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_missing_required_key_rejected(self, valid_config, key):
        cfg = copy.deepcopy(valid_config)
        del cfg["baseline"][key]
        with pytest.raises(BaselineConfigError, match=key):
            validate_baseline_config(cfg)

    @pytest.mark.parametrize("key", ["dataset_root", "dplm_commit"])
    def test_null_runtime_key_rejected(self, valid_config, key):
        cfg = copy.deepcopy(valid_config)
        cfg["baseline"][key] = None
        with pytest.raises(BaselineConfigError, match=key):
            validate_baseline_config(cfg)

    def test_missing_baseline_section_rejected(self):
        with pytest.raises(BaselineConfigError, match="baseline"):
            validate_baseline_config({"other": {}})

    def test_empty_config_rejected(self):
        with pytest.raises(BaselineConfigError):
            validate_baseline_config({})


# ── K0.2: Frozen values must match contract ──────────────────────────────────

class TestFrozenValues:

    def test_trainable_params_must_be_adapter(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["baseline"]["trainable_params_pattern"] = "all"
        with pytest.raises(BaselineConfigError, match="trainable_params_pattern"):
            validate_baseline_config(cfg)

    def test_checkpoint_must_be_dplm_650m(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["baseline"]["checkpoint_id"] = "some_other/model"
        with pytest.raises(BaselineConfigError, match="checkpoint_id"):
            validate_baseline_config(cfg)

    def test_gvp_must_be_frozen(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["baseline"]["gvp_frozen"] = False
        with pytest.raises(BaselineConfigError, match="gvp_frozen"):
            validate_baseline_config(cfg)

    def test_backbone_must_be_frozen(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["baseline"]["backbone_frozen"] = False
        with pytest.raises(BaselineConfigError, match="backbone_frozen"):
            validate_baseline_config(cfg)

    def test_seed_must_be_42(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["baseline"]["seed"] = 123
        with pytest.raises(BaselineConfigError, match="seed"):
            validate_baseline_config(cfg)


# ── K0.3: Valid config accepted ──────────────────────────────────────────────

class TestValidConfig:

    def test_complete_config_accepted(self, valid_config):
        result = validate_baseline_config(valid_config)
        assert result["baseline"]["checkpoint_id"] == "airkingbd/dplm_650m"
        assert result["baseline"]["seed"] == 42

    def test_extra_keys_preserved(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["baseline"]["custom_note"] = "experiment_1"
        result = validate_baseline_config(cfg)
        assert result["baseline"]["custom_note"] == "experiment_1"


# ── K0.4: Run directory schema ───────────────────────────────────────────────

class TestRunDirectorySchema:

    def test_artifact_root_not_empty(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["baseline"]["artifact_root"] = ""
        with pytest.raises(BaselineConfigError, match="artifact_root"):
            validate_baseline_config(cfg)

    def test_required_run_artifacts_present(self, valid_config):
        result = validate_baseline_config(valid_config)
        artifacts = result["baseline"]["required_run_artifacts"]
        assert "resolved_config.yaml" in artifacts
        assert "checkpoints/" in artifacts
        assert "baseline_validation.json" in artifacts
        assert "failure_ledger.json" in artifacts


# ── K0.5: YAML loader integration ───────────────────────────────────────────

class TestYAMLLoader:

    def test_load_default_config_has_baseline_section(self):
        cfg = load_baseline_config()
        assert "baseline" in cfg

    def test_load_default_config_has_checkpoint_id(self):
        cfg = load_baseline_config()
        assert cfg["baseline"]["checkpoint_id"] == "airkingbd/dplm_650m"

    def test_load_with_overrides(self, tmp_path):
        override = tmp_path / "override.yaml"
        override.write_text(
            "baseline:\n"
            "  dataset_root: /my/data\n"
            "  dplm_commit: deadbeef\n"
        )
        cfg = load_baseline_config(override_path=str(override))
        assert cfg["baseline"]["dataset_root"] == "/my/data"
        assert cfg["baseline"]["dplm_commit"] == "deadbeef"
        # non-overridden values preserved
        assert cfg["baseline"]["checkpoint_id"] == "airkingbd/dplm_650m"
