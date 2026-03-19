"""Module K2 contract tests — training launcher and checkpoint registry.

TDD gates from PLAN_IF.md Task K2:
  RED:   launcher accepts invalid run arguments or writes incomplete run metadata.
  GREEN: smoke launcher creates the expected run directory tree and metadata files
         on fixture inputs.
"""

import json
import os

import pytest
import yaml

from inverse_folding.training.launcher import (
    LauncherConfig,
    LauncherError,
    build_dplm_command,
    create_run_directory,
    resolve_run_id,
    write_run_metadata,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def base_launcher_cfg(tmp_path):
    """A valid LauncherConfig pointing to temp directories."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "cath_4.3").mkdir()
    (data_dir / "cath_4.3" / "chain_set.jsonl").write_text("{}")
    (data_dir / "cath_4.3" / "chain_set_splits.json").write_text("{}")

    output_root = tmp_path / "runs"
    output_root.mkdir()

    dplm_root = tmp_path / "dplm"
    dplm_root.mkdir()
    (dplm_root / "train.py").write_text("# stub")

    return LauncherConfig(
        data_dir=str(data_dir),
        output_root=str(output_root),
        dplm_root=str(dplm_root),
        run_id="test_run_001",
        num_gpus=1,
        smoke=False,
    )


# ── K2.1: Reject invalid launcher arguments ─────────────────────────────────

class TestLauncherValidation:

    def test_missing_data_dir_rejected(self, base_launcher_cfg):
        cfg = base_launcher_cfg
        cfg.data_dir = "/nonexistent/path"
        with pytest.raises(LauncherError, match="data_dir"):
            create_run_directory(cfg)

    def test_missing_cath_data_rejected(self, base_launcher_cfg, tmp_path):
        empty_dir = tmp_path / "empty_data"
        empty_dir.mkdir()
        cfg = base_launcher_cfg
        cfg.data_dir = str(empty_dir)
        with pytest.raises(LauncherError, match="cath_4.3"):
            create_run_directory(cfg)

    def test_missing_dplm_root_rejected(self, base_launcher_cfg):
        cfg = base_launcher_cfg
        cfg.dplm_root = "/nonexistent/dplm"
        with pytest.raises(LauncherError, match="dplm_root"):
            create_run_directory(cfg)

    def test_missing_train_py_rejected(self, base_launcher_cfg, tmp_path):
        empty_dplm = tmp_path / "empty_dplm"
        empty_dplm.mkdir()
        cfg = base_launcher_cfg
        cfg.dplm_root = str(empty_dplm)
        with pytest.raises(LauncherError, match="train.py"):
            create_run_directory(cfg)


# ── K2.2: Run directory creation ─────────────────────────────────────────────

class TestRunDirectoryCreation:

    def test_creates_run_directory_tree(self, base_launcher_cfg):
        run_dir = create_run_directory(base_launcher_cfg)
        assert os.path.isdir(run_dir)
        assert os.path.isdir(os.path.join(run_dir, "checkpoints"))

    def test_run_id_appears_in_path(self, base_launcher_cfg):
        run_dir = create_run_directory(base_launcher_cfg)
        assert "test_run_001" in run_dir

    def test_artifact_root_structure(self, base_launcher_cfg):
        run_dir = create_run_directory(base_launcher_cfg)
        assert run_dir.endswith(
            os.path.join("dplm_v1_adapter", "test_run_001")
        )

    def test_duplicate_run_id_rejected(self, base_launcher_cfg):
        create_run_directory(base_launcher_cfg)
        with pytest.raises(LauncherError, match="already exists"):
            create_run_directory(base_launcher_cfg)


# ── K2.3: Run metadata ──────────────────────────────────────────────────────

class TestRunMetadata:

    def test_writes_resolved_config(self, base_launcher_cfg):
        run_dir = create_run_directory(base_launcher_cfg)
        write_run_metadata(base_launcher_cfg, run_dir)

        config_path = os.path.join(run_dir, "resolved_config.yaml")
        assert os.path.isfile(config_path)
        with open(config_path) as f:
            saved = yaml.safe_load(f)
        assert saved["run_id"] == "test_run_001"
        assert saved["data_dir"] == base_launcher_cfg.data_dir
        assert saved["num_gpus"] == 1

    def test_writes_environment_metadata(self, base_launcher_cfg):
        run_dir = create_run_directory(base_launcher_cfg)
        write_run_metadata(base_launcher_cfg, run_dir)

        env_path = os.path.join(run_dir, "environment.json")
        assert os.path.isfile(env_path)
        with open(env_path) as f:
            env = json.load(f)
        assert "python_version" in env
        assert "timestamp" in env
        assert "hostname" in env

    def test_metadata_includes_seed(self, base_launcher_cfg):
        run_dir = create_run_directory(base_launcher_cfg)
        write_run_metadata(base_launcher_cfg, run_dir)

        config_path = os.path.join(run_dir, "resolved_config.yaml")
        with open(config_path) as f:
            saved = yaml.safe_load(f)
        assert saved["seed"] == 42


# ── K2.4: DPLM command construction ─────────────────────────────────────────

class TestDPLMCommand:

    def test_command_targets_train_py(self, base_launcher_cfg):
        run_dir = create_run_directory(base_launcher_cfg)
        cmd = build_dplm_command(base_launcher_cfg, run_dir)
        assert cmd[0] == "python"
        assert cmd[1].endswith("train.py")

    def test_command_sets_experiment(self, base_launcher_cfg):
        run_dir = create_run_directory(base_launcher_cfg)
        cmd = build_dplm_command(base_launcher_cfg, run_dir)
        assert "experiment=dplm/cond_dplm_650m" in cmd

    def test_command_overrides_data_dir(self, base_launcher_cfg):
        run_dir = create_run_directory(base_launcher_cfg)
        cmd = build_dplm_command(base_launcher_cfg, run_dir)
        data_override = [c for c in cmd if c.startswith("paths.data_dir=")]
        assert len(data_override) == 1
        assert data_override[0] == f"paths.data_dir={base_launcher_cfg.data_dir}"

    def test_command_overrides_log_dir(self, base_launcher_cfg):
        run_dir = create_run_directory(base_launcher_cfg)
        cmd = build_dplm_command(base_launcher_cfg, run_dir)
        log_override = [c for c in cmd if c.startswith("paths.log_dir=")]
        assert len(log_override) == 1
        assert run_dir in log_override[0]

    def test_smoke_mode_reduces_steps(self, base_launcher_cfg):
        base_launcher_cfg.smoke = True
        run_dir = create_run_directory(base_launcher_cfg)
        cmd = build_dplm_command(base_launcher_cfg, run_dir)
        steps_override = [c for c in cmd if c.startswith("trainer.max_steps=")]
        assert len(steps_override) == 1
        steps = int(steps_override[0].split("=")[1])
        assert steps <= 100

    def test_smoke_mode_limits_data(self, base_launcher_cfg):
        base_launcher_cfg.smoke = True
        run_dir = create_run_directory(base_launcher_cfg)
        cmd = build_dplm_command(base_launcher_cfg, run_dir)
        length_override = [c for c in cmd if c.startswith("datamodule.max_length=")]
        assert len(length_override) == 1


# ── K2.5: Run ID generation ─────────────────────────────────────────────────

class TestRunIDGeneration:

    def test_auto_run_id_contains_seed(self):
        run_id = resolve_run_id(None, seed=42)
        assert "seed42" in run_id

    def test_auto_run_id_contains_date(self):
        run_id = resolve_run_id(None, seed=42)
        # Should contain YYYYMMDD pattern
        import re
        assert re.search(r"\d{8}", run_id)

    def test_explicit_run_id_preserved(self):
        run_id = resolve_run_id("my_custom_run", seed=42)
        assert run_id == "my_custom_run"
