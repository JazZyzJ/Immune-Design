"""Training launcher for DPLM v1 adapter on CATH.

Wraps the DPLM Hydra-based training with:
  - Input validation (data dir, DPLM root, output paths)
  - Run directory creation under the frozen artifact schema
  - Resolved config and environment metadata persistence
  - Smoke mode for local testing before cluster submission
"""

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import yaml


class LauncherError(Exception):
    """Raised when the launcher detects an invalid or unsafe configuration."""


@dataclass
class LauncherConfig:
    """All parameters needed to launch a DPLM adapter training run."""

    data_dir: str                # Parent of cath_4.3/ directory
    output_root: str             # Parent for runs (e.g., .../run/inverse_folding)
    dplm_root: str               # Path to cloned DPLM repo
    run_id: Optional[str] = None
    num_gpus: int = 1
    smoke: bool = False
    seed: int = 42
    extra_overrides: List[str] = field(default_factory=list)


# ── Constants ────────────────────────────────────────────────────────────────

_SMOKE_MAX_STEPS = 50
_SMOKE_MAX_LENGTH = 100
_SMOKE_MAX_EPOCHS = 2
_EXPERIMENT = "dplm/cond_dplm_650m"
_RUN_SUBDIR = "dplm_v1_adapter"


# ── Public API ───────────────────────────────────────────────────────────────

def resolve_run_id(run_id: Optional[str], seed: int = 42) -> str:
    """Generate a deterministic run ID if none is provided."""
    if run_id is not None:
        return run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"seed{seed}_{timestamp}"


def create_run_directory(cfg: LauncherConfig) -> str:
    """Validate inputs and create the run directory tree.

    Returns the absolute path to the created run directory.
    """
    # Validate data_dir exists
    if not os.path.isdir(cfg.data_dir):
        raise LauncherError(
            f"data_dir does not exist: {cfg.data_dir}"
        )

    # Validate CATH data is present
    cath_dir = os.path.join(cfg.data_dir, "cath_4.3")
    if not os.path.isdir(cath_dir):
        raise LauncherError(
            f"cath_4.3 directory not found under data_dir: {cfg.data_dir}"
        )

    # Validate DPLM root
    if not os.path.isdir(cfg.dplm_root):
        raise LauncherError(
            f"dplm_root does not exist: {cfg.dplm_root}"
        )

    # Validate train.py exists
    train_py = os.path.join(cfg.dplm_root, "train.py")
    if not os.path.isfile(train_py):
        raise LauncherError(
            f"train.py not found in dplm_root: {cfg.dplm_root}"
        )

    # Resolve run_id
    if cfg.run_id is None:
        cfg.run_id = resolve_run_id(None, seed=cfg.seed)

    # Create run directory
    run_dir = os.path.join(
        cfg.output_root, _RUN_SUBDIR, cfg.run_id
    )
    if os.path.exists(run_dir):
        raise LauncherError(
            f"Run directory already exists: {run_dir}"
        )

    os.makedirs(run_dir)
    os.makedirs(os.path.join(run_dir, "checkpoints"))

    return run_dir


def write_run_metadata(
    cfg: LauncherConfig,
    run_dir: str,
    frozen_baseline: Optional[dict] = None,
) -> None:
    """Persist resolved config and environment metadata to the run directory."""
    # Resolved config
    resolved = {
        "run_id": cfg.run_id,
        "data_dir": cfg.data_dir,
        "output_root": cfg.output_root,
        "dplm_root": cfg.dplm_root,
        "num_gpus": cfg.num_gpus,
        "smoke": cfg.smoke,
        "seed": cfg.seed,
        "experiment": _EXPERIMENT,
        "extra_overrides": cfg.extra_overrides,
    }
    if frozen_baseline is not None:
        resolved["frozen_baseline"] = frozen_baseline
    config_path = os.path.join(run_dir, "resolved_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(resolved, f, default_flow_style=False, sort_keys=False)

    # Environment metadata
    env_meta = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
    }
    env_path = os.path.join(run_dir, "environment.json")
    with open(env_path, "w") as f:
        json.dump(env_meta, f, indent=2)


def build_dplm_command(cfg: LauncherConfig, run_dir: str) -> List[str]:
    """Construct the DPLM training command with Hydra overrides."""
    train_py = os.path.join(cfg.dplm_root, "train.py")

    cmd = [
        "python",
        train_py,
        f"experiment={_EXPERIMENT}",
        f"name={cfg.run_id}",
        f"paths.data_dir={cfg.data_dir}",
        f"paths.log_dir={run_dir}",
        f"train.seed={cfg.seed}",
        f"seed={cfg.seed}",
        "logger=tensorboard",
    ]

    # GPU configuration
    if cfg.num_gpus > 1:
        cmd.append(f"trainer.devices={cfg.num_gpus}")
    else:
        cmd.extend([
            "trainer.devices=1",
            "trainer.strategy=auto",
        ])

    # Smoke mode overrides
    if cfg.smoke:
        cmd.extend([
            f"trainer.max_steps={_SMOKE_MAX_STEPS}",
            f"trainer.max_epochs={_SMOKE_MAX_EPOCHS}",
            f"datamodule.max_length={_SMOKE_MAX_LENGTH}",
            "trainer.enable_progress_bar=true",
            "trainer.val_check_interval=1.0",
        ])

    # User-provided extra overrides
    cmd.extend(cfg.extra_overrides)

    return cmd
