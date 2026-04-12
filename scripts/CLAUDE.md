# Scripts — Code Reuse Standards

This file governs how scripts and SLURM submission files are written in this project. The goal is to minimize duplication and maximize parameterization.

## 1. Reuse-First Check

Before writing any new script:

1. Read `doc/SCRIPTS.md` and identify the target module section.
2. If an existing script covers ≥60% of the intended functionality, **extend it with CLI arguments** rather than creating a new file.
3. If a new script is genuinely needed, register it in `doc/SCRIPTS.md` before marking the task as complete.

## 2. Parameterize over Duplicate

Two scripts that differ only in hyperparameters (allele, seed, encoder variant, eta grid, profile name, etc.) must be merged into one script with CLI arguments or environment variables.

**Decision rule**: if a `diff` between two candidate scripts shows >60% shared lines, they must be a single parameterized script.

Examples of parameters that should be CLI args, not separate files:

| Parameter | Flag pattern | Bad (separate file) |
|-----------|-------------|---------------------|
| Allele | `--allele DRB1*07:01` | `submit_train_0701.slurm` vs `submit_train_0401.slurm` |
| Seed | `--seed 42` | `submit_train_seed42.slurm` vs `submit_train_seed7.slurm` |
| Training profile | `--profile strict` | `submit_train_strict.slurm` vs `submit_train_balanced.slurm` |
| Encoder variant | `--variant-id LC1` | `submit_v2_lc1.slurm` vs `submit_v2_e1.slurm` |

## 3. SLURM Consolidation Guide

### Target structure

SLURM scripts should be organized by **module**, not by individual experiment. The ideal set is one parameterized submission script per module:

| Module | Script | Key variables |
|--------|--------|---------------|
| Epitope Head training | `submit_epitope_train.slurm` | `PROFILE`, `VARIANT_ID`, `SEED`, `REGISTRY`, `P_AUG` |
| Inverse Folding | `submit_if.slurm` | `MODE={train,validate,sweep}`, `CKPT_PATH`, `ETA_GRID` |
| Data Augmentation | `submit_mutation.slurm` | `STAGE={augment,merge}`, `N_SHARDS` |
| Data Selection | `submit_data_sel.slurm` | `STEP={prescreen,assemble}`, `ALLELE`, `TIER` |
| Ablation | `submit_ablation.slurm` | `TYPE={encoder,flank}`, `ENCODERS`, `SEEDS` |

### Parameterization pattern for SLURM

Use environment variables with defaults so the script works standalone but can be overridden at submission time:

```bash
# At the top of the script, after set -euo pipefail:
MODE="${MODE:-train}"          # default to train
ALLELE="${ALLELE:-DRB1*07:01}" # default allele
SEED="${SEED:-42}"             # default seed
```

Submit with overrides:

```bash
MODE=sweep ALLELE="DRB1*04:01" sbatch scripts/submit_if.slurm
```

### SBATCH header parameterization

Use conditional SBATCH-compatible patterns for resource differences:

```bash
# ── Resources (mode-dependent) ──
if [ "$MODE" = "train" ]; then
    TIME="48:00:00"; NGPU=2; QOS="gpu-medium"
elif [ "$MODE" = "validate" ]; then
    TIME="12:00:00"; NGPU=1; QOS="gpu-short"
elif [ "$MODE" = "sweep" ]; then
    TIME="24:00:00"; NGPU=1; QOS="gpu-short"
fi
```

Note: since SBATCH directives are static, dynamic resource selection requires `sbatch --time=$TIME --gres=gpu:$NGPU` on the command line rather than inline `#SBATCH` headers.

## 4. Python Script Standards

- Prefer adding a CLI argument to an existing script over creating a new one.
- Use `argparse` with clear `--flag` names; avoid positional arguments for optional behavior.
- Shared logic across scripts should live in library modules (`inverse_folding/`, `epitope_head/`), not be copy-pasted between scripts.

## 5. Existing Script Index

The authoritative index is `doc/SCRIPTS.md`. Consult it before any script work.

Note: The current codebase has legacy duplication in SLURM scripts that predates this protocol. New scripts must follow these rules; legacy scripts will be consolidated in a future cleanup pass.
