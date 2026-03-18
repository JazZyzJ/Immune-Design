#!/usr/bin/env python3
"""Stage J — Task J6: Materialize augmented train-only ProteinSamples.

Reads mutation registry + source parquet, produces augmented train artifact.

Usage:
    python -m scripts.materialize_augmented_train_set
    python -m scripts.materialize_augmented_train_set --config path/to/augmentation.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from epitope_head.data.materialize_augmented_samples import (
    materialize_augmented_samples,
    validate_augmented_samples,
)
from epitope_head.data.netmhciipan_mutation import load_augmentation_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Materialize augmented train samples")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override manifest directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (same as registry output)")
    args = parser.parse_args()

    cfg = load_augmentation_config(args.config)

    # Resolve paths
    if args.data_root:
        data_root = Path(args.data_root).expanduser()
        source_parquet = data_root / "protein_samples_strict.parquet"
        val_ids_path = data_root / "splits" / "strict" / "val_ids.txt"
        test_ids_path = data_root / "splits" / "strict" / "test_ids.txt"
    else:
        source_parquet = Path(cfg.inputs["source_parquet"])
        val_ids_path = Path(cfg.inputs["val_ids"])
        test_ids_path = Path(cfg.inputs["test_ids"])

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser()
    else:
        out_dir = Path(cfg.outputs["registry"]).parent

    registry_path = out_dir / "mutation_registry_strict.parquet"

    # Load inputs
    registry_df = pd.read_parquet(registry_path)
    source_df = pd.read_parquet(source_parquet)
    val_ids = set(val_ids_path.read_text().strip().split("\n"))
    test_ids = set(test_ids_path.read_text().strip().split("\n"))

    logger.info(f"Registry: {len(registry_df)} mutations")
    logger.info(f"Source: {len(source_df)} proteins")

    # Materialize
    aug_df = materialize_augmented_samples(registry_df, source_df)
    logger.info(f"Materialized: {len(aug_df)} augmented samples")

    # Validate
    errors = validate_augmented_samples(aug_df, source_df, val_ids, test_ids)
    if errors:
        for e in errors:
            logger.error(f"VALIDATION FAIL: {e}")
        raise RuntimeError(f"Augmented samples failed validation with {len(errors)} errors")

    logger.info("Validation passed: schema, split safety, provenance all OK")

    # Write artifact
    out_path = out_dir / "protein_samples_strict_aug_train.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    aug_df.to_parquet(out_path, index=False)
    logger.info(f"Written: {out_path}")

    # Write summary
    if len(aug_df) > 0:
        pos_counts = aug_df["positive_count"].sum()
        summary = {
            "n_augmented_samples": len(aug_df),
            "total_remaining_positives": int(pos_counts),
            "avg_positives_per_sample": round(pos_counts / max(len(aug_df), 1), 2),
            "n_zero_positive_samples": int((aug_df["positive_count"] == 0).sum()),
            "unique_source_proteins": int(
                aug_df["protein_id"].str.extract(r"AUG::(.+?)::")[0].nunique()
            ),
            "sequence_length_stats": {
                "min": int(aug_df["sequence_length"].min()),
                "max": int(aug_df["sequence_length"].max()),
                "mean": round(float(aug_df["sequence_length"].mean()), 1),
            },
        }
    else:
        summary = {
            "n_augmented_samples": 0,
            "total_remaining_positives": 0,
            "avg_positives_per_sample": 0.0,
            "n_zero_positive_samples": 0,
            "unique_source_proteins": 0,
            "sequence_length_stats": {"min": 0, "max": 0, "mean": 0.0},
        }

    summary_path = out_dir / "protein_samples_strict_aug_train_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary_path}")

    print(f"\n{'='*60}")
    print(f"MATERIALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Augmented samples:    {summary['n_augmented_samples']}")
    print(f"  Remaining positives:  {summary['total_remaining_positives']}")
    print(f"  Avg pos/sample:       {summary['avg_positives_per_sample']}")
    print(f"  Zero-positive:        {summary['n_zero_positive_samples']}")
    print(f"  Source proteins:      {summary['unique_source_proteins']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
