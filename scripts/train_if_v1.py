"""CLI entry point for DPLM v1 adapter training.

Usage (local smoke test):
    python -m scripts.train_if_v1 \
        --data-dir ./inverse_folding/dplm/data-bin \
        --output-root ./outputs/if/runs \
        --dplm-root ./inverse_folding/dplm \
        --smoke

Usage (cluster, via SLURM script):
    See scripts/submit_if_train.slurm
"""

import argparse
import os
import subprocess
import sys

from inverse_folding.configs.schema import (
    BaselineConfigError,
    load_baseline_config,
    validate_baseline_config,
)
from inverse_folding.training.launcher import (
    LauncherConfig,
    build_dplm_command,
    create_run_directory,
    resolve_run_id,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DPLM v1 IF adapter on CATH")

    p.add_argument("--data-dir", required=True,
                    help="Parent directory containing cath_4.3/")
    p.add_argument("--output-root", required=True,
                    help="Root for run artifacts (e.g., .../run/inverse_folding)")
    p.add_argument("--dplm-root", required=True,
                    help="Path to cloned DPLM repo")
    p.add_argument("--run-id", default=None,
                    help="Explicit run ID (auto-generated if omitted)")
    p.add_argument("--num-gpus", type=int, default=1,
                    help="Number of GPUs for training")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (frozen to 42 in v1)")
    p.add_argument("--smoke", action="store_true",
                    help="Run a quick smoke test with reduced data and steps")
    p.add_argument("--dry-run", action="store_true",
                    help="Print the training command without executing it")
    p.add_argument("extra_overrides", nargs="*",
                    help="Additional Hydra overrides passed to DPLM train.py")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── K0 contract enforcement ──────────────────────────────────────────
    # Load frozen baseline config with runtime overrides
    override_cfg = {
        "baseline": {
            "dataset_root": args.data_dir,
        }
    }
    # If user provided a dplm_commit override, use it; otherwise the YAML default
    baseline_cfg = load_baseline_config()
    baseline_cfg["baseline"]["dataset_root"] = args.data_dir

    # Resolve dplm_commit from the cloned repo if still null in YAML
    if baseline_cfg["baseline"].get("dplm_commit") is None:
        commit_file = os.path.join(args.dplm_root, ".git", "HEAD")
        if os.path.isfile(commit_file):
            import subprocess as _sp
            result = _sp.run(
                ["git", "rev-parse", "HEAD"],
                cwd=args.dplm_root, capture_output=True, text=True,
            )
            if result.returncode == 0:
                baseline_cfg["baseline"]["dplm_commit"] = result.stdout.strip()

    try:
        validate_baseline_config(baseline_cfg)
    except BaselineConfigError as e:
        print(f"ERROR: Baseline config contract violation: {e}")
        return 1

    frozen = baseline_cfg["baseline"]
    print(f"K0 contract validated: checkpoint={frozen['checkpoint_id']}, "
          f"trainable={frozen['trainable_params_pattern']}, "
          f"seed={frozen['seed']}, commit={frozen['dplm_commit'][:12]}...")

    # ── Build launcher config ────────────────────────────────────────────
    run_id = resolve_run_id(args.run_id, seed=args.seed)

    cfg = LauncherConfig(
        data_dir=args.data_dir,
        output_root=args.output_root,
        dplm_root=args.dplm_root,
        run_id=run_id,
        num_gpus=args.num_gpus,
        smoke=args.smoke,
        seed=args.seed,
        extra_overrides=args.extra_overrides,
    )

    # Create run directory and write metadata (includes frozen K0 pins)
    run_dir = create_run_directory(cfg)
    write_run_metadata(cfg, run_dir, frozen_baseline=frozen)

    # Build the DPLM training command
    cmd = build_dplm_command(cfg, run_dir)

    print("=" * 60)
    print(f"IF v1 Training Launcher")
    print(f"  Run ID:      {run_id}")
    print(f"  Run dir:     {run_dir}")
    print(f"  Data dir:    {cfg.data_dir}")
    print(f"  DPLM root:   {cfg.dplm_root}")
    print(f"  GPUs:        {cfg.num_gpus}")
    print(f"  Smoke mode:  {cfg.smoke}")
    print(f"  Seed:        {cfg.seed}")
    print("=" * 60)
    print(f"\nCommand:\n  {' '.join(cmd)}\n")

    if args.dry_run:
        print("[dry-run] Skipping execution.")
        return 0

    # Execute training from the DPLM repo directory
    result = subprocess.run(cmd, cwd=cfg.dplm_root)

    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        return result.returncode

    print(f"\nTraining complete. Artifacts at: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
