#!/usr/bin/env python
"""Protenix refold-cache precompute driver (two shell-separated phases).

Populates the shared refold cache (``<cache_dir>/<cache_key>.pdb`` + ``.plddt`` on the
0-100 scale) with Protenix v2 structures so ``evaluate_phase_c.py --refold-model
protenix`` becomes a pure cache read. Protenix ships its own interpreter (the AF3 env)
and is CALLED, never imported here; the group install at
``/scratch/gpfs/KAIYIJIANG/tools/protenix`` is never modified.

A SLURM shard runs three steps (see scripts/submit_protenix_refold.slurm):
  1. ``--mode build-json`` (immune-design): build a Protenix-dialect JSON of this
     shard's UNIQUE ``(protein_id, sequence)`` monomers, fold name = cache_key.
  2. protenix env: ``module load proxy/default; source .../protenix/bin/protenix-env.sh;
     protenix-run pred -i <shard.json> -o <protenix_out> --use_msa true`` (full-MSA
     needs compute-node egress via proxy/default; --use_msa false for the offline
     no-MSA fallback).
  3. ``--mode normalize`` (immune-design): pick each fold's global-best sample and
     write it into ``--cache-dir`` as ``<cache_key>.pdb`` + ``.plddt``. Fail-fast if any
     fold is missing (no placeholder — a partial cache would silently break struct eval).

Registered in doc/SCRIPTS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_on_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


_repo_on_path()
from inverse_folding.evaluation.refold_normalize import fold_records_from_parquet  # noqa: E402


def build_protenix_json(records: list[tuple[str, str]]) -> list[dict]:
    """Protenix-dialect job list (one single-count protein monomer per record)."""
    return [
        {"name": key, "sequences": [{"proteinChain": {"sequence": seq, "count": 1}}]}
        for key, seq in records
    ]


def records_for_shard(parquet_path: str, n_shards: int, shard_idx: int) -> list[tuple[str, str]]:
    """This shard's ``(cache_key, sequence)`` records (round-robin over unique folds)."""
    recs = fold_records_from_parquet(parquet_path)
    return recs[shard_idx::n_shards] if n_shards > 1 else recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("build-json", "normalize"), required=True)
    ap.add_argument("--from-parquet", required=True, help="generated parquet (protein_id, sequence)")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--out-json", default=None, help="build-json: Protenix JSON output path")
    ap.add_argument("--protenix-out", default=None, help="normalize: Protenix `pred` output dir")
    ap.add_argument("--cache-dir", default=None, help="normalize: shared refold cache dir")
    args = ap.parse_args()

    recs = records_for_shard(args.from_parquet, args.n_shards, args.shard_idx)

    if args.mode == "build-json":
        if not args.out_json:
            raise SystemExit("--out-json is required for --mode build-json")
        jobs = build_protenix_json(recs)
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(jobs, indent=2))
        print(f"[build-json] shard {args.shard_idx}/{args.n_shards}: {len(jobs)} monomer jobs -> {out}")
        return

    # normalize
    if not (args.protenix_out and args.cache_dir):
        raise SystemExit("--protenix-out and --cache-dir are required for --mode normalize")
    from inverse_folding.evaluation.protenix_runner import normalize_protenix_to_cache

    n_ok, missing = 0, []
    for key, _seq in recs:
        try:
            normalize_protenix_to_cache(args.protenix_out, key, cache_dir=args.cache_dir, key=key)
            n_ok += 1
        except FileNotFoundError:
            missing.append(key)
    print(f"[normalize] shard {args.shard_idx}/{args.n_shards}: {n_ok} cached, "
          f"{len(missing)} missing -> {args.cache_dir}")
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} Protenix fold(s) missing (first: {missing[:3]}); "
            "precompute incomplete — do not run struct eval on a partial cache"
        )
    print("[done]")


if __name__ == "__main__":
    main()
