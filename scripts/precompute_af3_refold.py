#!/usr/bin/env python
"""AlphaFold3 refold-cache precompute driver (build-json / normalize).

Populates the shared refold cache (``<cache_dir>/<cache_key>.pdb`` + ``.plddt`` on the
0-100 scale) with official DeepMind AlphaFold3 (v3.0.3) structures so
``evaluate_phase_c.py --refold-model af3`` becomes a pure cache read. AF3 ships its own
interpreter and is CALLED via the group wrapper (``/scratch/gpfs/KAIYIJIANG/tools/alphafold3/bin/af3-run``),
never imported here; the shared install is never modified.

AF3's MSA is a LOCAL database search (no external server), so full-MSA is a clean two-stage
that stays on compute nodes: ``--mode data`` (CPU, local jackhmmer/nhmmer) then ``--mode
inference`` (GPU). No-MSA is a single GPU ``--mode inference`` on empty-MSA JSONs.

Modes (both immune-design):
  build-json : one AF3-dialect JSON per unique (protein_id, sequence) monomer (name =
               cache_key) into --json-dir. --no-msa emits empty MSA fields (inference fast
               path); otherwise the MSA fields are omitted so --mode data fills them.
  normalize  : pick each fold's best model (<name>/<name>_model.cif) + mean(atom_plddts)
               and write it into --cache-dir; fail-fast if any fold is missing.

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


def build_af3_json(key: str, seq: str, no_msa: bool = True) -> dict:
    """AlphaFold3-dialect single-protein-monomer JSON (fold name = cache_key)."""
    protein: dict = {"id": "A", "sequence": seq}
    if no_msa:
        # empty MSA + no templates -> --mode inference skips the genetic search
        protein.update({"unpairedMsa": "", "pairedMsa": "", "templates": []})
    return {
        "name": key,
        "sequences": [{"protein": protein}],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
    }


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
    ap.add_argument("--json-dir", default=None, help="build-json: dir for per-protein AF3 JSONs")
    ap.add_argument("--no-msa", action="store_true",
                    help="build-json: empty-MSA inference fast path (else omit MSA fields for --mode data)")
    ap.add_argument("--af3-out", default=None, help="normalize: AF3 inference output dir")
    ap.add_argument("--cache-dir", default=None, help="normalize: shared refold cache dir")
    args = ap.parse_args()

    recs = records_for_shard(args.from_parquet, args.n_shards, args.shard_idx)

    if args.mode == "build-json":
        if not args.json_dir:
            raise SystemExit("--json-dir is required for --mode build-json")
        json_dir = Path(args.json_dir)
        json_dir.mkdir(parents=True, exist_ok=True)
        for key, seq in recs:
            (json_dir / f"{key}.json").write_text(
                json.dumps(build_af3_json(key, seq, no_msa=args.no_msa), indent=2)
            )
        print(f"[build-json] shard {args.shard_idx}/{args.n_shards}: {len(recs)} AF3 JSONs "
              f"(no_msa={args.no_msa}) -> {json_dir}")
        return

    # normalize
    if not (args.af3_out and args.cache_dir):
        raise SystemExit("--af3-out and --cache-dir are required for --mode normalize")
    from inverse_folding.evaluation.af3_runner import normalize_af3_to_cache

    n_ok, missing = 0, []
    for key, _seq in recs:
        try:
            normalize_af3_to_cache(args.af3_out, key, cache_dir=args.cache_dir, key=key)
            n_ok += 1
        except FileNotFoundError:
            missing.append(key)
    print(f"[normalize] shard {args.shard_idx}/{args.n_shards}: {n_ok} cached, "
          f"{len(missing)} missing -> {args.cache_dir}")
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} AF3 fold(s) missing (first: {missing[:3]}); "
            "precompute incomplete — do not run struct eval on a partial cache"
        )
    print("[done]")


if __name__ == "__main__":
    main()
