#!/usr/bin/env python
"""Split a Phase C generated ("facade") parquet into N shard files for the
``submit_benchmark.slurm MODE=phase_c`` ``N_SHARDS`` job array.

Writes ``<out-base>.shard{k:02d}of{N:02d}.parquet`` for ``k`` in ``[0, N)``,
matching the ``SHARD_TAG`` convention (``printf 'shard%02dof%02d'``) that the
launcher and ``merge_eval_immune_shards.py`` expect. Round-robin over rows
balances NMP load (sequence length varies per design) across shards. Every
input row lands in exactly one shard; a final assert guards the partition.

Recommended sharded CPU immune-eval submit config (the one that schedules fast on
Della — ~30 min/shard, high concurrency), reflected in the defaults below:
    N_SHARDS=192  (~120 designs/shard for a full ~23k-design pool)
    sbatch --array=0-191 --partition=cpu --qos=short \\
           --cpus-per-task=4 --mem=16G --time=2:00:00   NMP_WORKERS=4
WARNING: keep --time >= 2h. On Della, --time <= 1h auto-demotes to the `test`
qos (GrpJobs=2 -> only 2 concurrent), which is why sharded arrays "only run 3-4
at once". The 2h is a nominal ceiling; each shard still finishes in ~30 min.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="generated.parquet (facade) to split.")
    ap.add_argument(
        "--n-shards",
        type=int,
        default=192,
        help="Number of shards (default 192: ~120 designs/shard for a full ~23k pool, "
        "~30 min each at 4 cores on qos=short).",
    )
    ap.add_argument(
        "--out-base",
        required=True,
        help="Output base path; writes <out-base>.shard{k:02d}of{N:02d}.parquet.",
    )
    ap.add_argument("--mode", choices=("round_robin", "contiguous"), default="round_robin")
    args = ap.parse_args()
    if args.n_shards <= 0:
        ap.error("--n-shards must be positive")

    df = pd.read_parquet(args.input)
    n = len(df)
    if n == 0:
        ap.error(f"empty input: {args.input}")
    if args.n_shards > n:
        ap.error(f"--n-shards {args.n_shards} exceeds row count {n}")

    out_base = Path(args.out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    for k in range(args.n_shards):
        if args.mode == "round_robin":
            shard = df.iloc[k :: args.n_shards]
        else:
            idx = np.array_split(np.arange(n), args.n_shards)[k]
            shard = df.iloc[idx]
        path = f"{out_base}.shard{k:02d}of{args.n_shards:02d}.parquet"
        shard.to_parquet(path, index=False)
        total += len(shard)
        print(f"shard {k:02d}/{args.n_shards}: {len(shard)} rows -> {path}")
    assert total == n, f"row-count mismatch after split: {total} != {n}"
    print(f"[done] split {n} rows into {args.n_shards} shards (mode={args.mode})")
    nsh = args.n_shards
    print(
        "[submit hint] sharded CPU immune eval (fast-scheduling config):\n"
        f"  sbatch --array=0-{nsh - 1} --partition=cpu --qos=short "
        "--cpus-per-task=4 --mem=16G --time=2:00:00  (NMP_WORKERS=4)\n"
        "  keep --time >= 2h: <=1h auto-demotes to qos=test (GrpJobs=2, only 2 concurrent)."
    )


if __name__ == "__main__":
    main()
