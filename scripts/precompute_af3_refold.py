#!/usr/bin/env python
"""AlphaFold3 refold-cache precompute driver (build-json / normalize).

Populates the shared refold cache (``<cache_dir>/<cache_key>.pdb`` + ``.plddt`` on the
0-100 scale) with official DeepMind AlphaFold3 (v3.0.3) structures so
``evaluate_phase_c.py --refold-model af3`` becomes a pure cache read. AF3 ships its own
interpreter and is CALLED via the group wrapper (``/scratch/gpfs/KAIYIJIANG/tools/alphafold3/bin/af3-run``),
never imported here; the shared install is never modified.

AF3 MSA options: (a) native local jackhmmer/nhmmer via ``--mode data`` (CPU, slow, hours/protein);
(b) **our local ColabFold** injected into ``unpairedMsa`` via ``--msa-a3m-dir`` (fast mmseqs GPU,
validated TM 0.970 vs native on WT mCherry) — skips ``--mode data`` entirely; (c) no-MSA
single-sequence. Then ``--mode inference`` (GPU) folds whichever JSON.

Modes (all immune-design):
  emit-fasta : write this shard's <cache_key>\\n<seq> FASTA (feed submit_tetramer_msa_local.slurm
               to produce <cache_key>.a3m for the ColabFold path).
  build-json : one AF3-dialect JSON per unique (protein_id, sequence) monomer (name = cache_key)
               into --json-dir. --msa-a3m-dir injects our ColabFold a3m into unpairedMsa;
               --no-msa emits empty MSA (inference fast path); else MSA fields are omitted so
               --mode data fills them.
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


def _clean_colabfold_a3m(text: str) -> str:
    """ColabFold a3m -> AF3-ready a3m: drop the leading ``#len\\tcard`` comment line so the
    first record is the query (AF3's A3M parser expects the query as record 0)."""
    lines = text.splitlines()
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    return "\n".join(lines) + "\n"


def build_af3_json(
    key: str, seq: str, no_msa: bool = True, ligand_smiles: str | None = None,
    unpaired_msa: str | None = None,
) -> dict:
    """AlphaFold3-dialect single-protein-monomer JSON (fold name = cache_key).

    ``ligand_smiles`` (optional) cofolds a SMILES ligand as a distinct chain ``L`` for a
    HOLO prediction; empty/None keeps the apo protein-only monomer. AF3 RDKit-parses the
    SMILES and preserves its formal charge (e.g. an anionic ``[n-]``).

    MSA precedence: ``unpaired_msa`` (an inline A3M string, e.g. from our local ColabFold)
    wins — it sets ``unpairedMsa`` + empty ``pairedMsa`` + no templates so ``--mode inference``
    uses it directly and skips AF3's slow jackhmmer/nhmmer ``--mode data``. Else ``no_msa``
    emits empty MSA (single-sequence fast path). Else the MSA fields are omitted so
    ``--mode data`` fills them (AF3 native local search).
    """
    protein: dict = {"id": "A", "sequence": seq}
    if unpaired_msa is not None:
        # our-ColabFold MSA injected inline -> inference reads it, no genetic search
        protein.update({"unpairedMsa": unpaired_msa, "pairedMsa": "", "templates": []})
    elif no_msa:
        # empty MSA + no templates -> --mode inference skips the genetic search
        protein.update({"unpairedMsa": "", "pairedMsa": "", "templates": []})
    sequences: list[dict] = [{"protein": protein}]
    if ligand_smiles:
        sequences.append({"ligand": {"id": "L", "smiles": ligand_smiles}})
    return {
        "name": key,
        "sequences": sequences,
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
    ap.add_argument("--mode", choices=("build-json", "normalize", "emit-fasta"), required=True)
    ap.add_argument("--from-parquet", required=True, help="generated parquet (protein_id, sequence)")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--json-dir", default=None, help="build-json: dir for per-protein AF3 JSONs")
    ap.add_argument("--fasta-out", default=None,
                    help="emit-fasta: write this shard's <cache_key>\\n<seq> FASTA (feed local ColabFold MSA)")
    ap.add_argument("--msa-a3m-dir", default=None,
                    help="build-json: inject <dir>/<cache_key>.a3m (our local ColabFold) into unpairedMsa, "
                         "skipping AF3's --mode data jackhmmer search")
    ap.add_argument("--no-msa", action="store_true",
                    help="build-json: empty-MSA inference fast path (else omit MSA fields for --mode data)")
    ap.add_argument("--ligand-smiles", default=None,
                    help="build-json: cofold this SMILES ligand (chain L) for a HOLO prediction; "
                         "empty/omitted = apo protein-only monomer")
    ap.add_argument("--af3-out", default=None, help="normalize: AF3 inference output dir")
    ap.add_argument("--cache-dir", default=None, help="normalize: shared refold cache dir")
    args = ap.parse_args()

    recs = records_for_shard(args.from_parquet, args.n_shards, args.shard_idx)

    if args.mode == "emit-fasta":
        if not args.fasta_out:
            raise SystemExit("--fasta-out is required for --mode emit-fasta")
        out = Path(args.fasta_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f">{key}\n{seq}\n" for key, seq in recs))
        print(f"[emit-fasta] shard {args.shard_idx}/{args.n_shards}: {len(recs)} seqs -> {out}")
        return

    if args.mode == "build-json":
        if not args.json_dir:
            raise SystemExit("--json-dir is required for --mode build-json")
        json_dir = Path(args.json_dir)
        json_dir.mkdir(parents=True, exist_ok=True)
        msa_dir = Path(args.msa_a3m_dir) if args.msa_a3m_dir else None
        for key, seq in recs:
            unpaired = None
            if msa_dir is not None:
                a3m = msa_dir / f"{key}.a3m"
                if not a3m.is_file():
                    raise SystemExit(f"--msa-a3m-dir set but {a3m} missing (run local ColabFold MSA first)")
                unpaired = _clean_colabfold_a3m(a3m.read_text())
            (json_dir / f"{key}.json").write_text(
                json.dumps(
                    build_af3_json(key, seq, no_msa=args.no_msa, ligand_smiles=args.ligand_smiles,
                                   unpaired_msa=unpaired),
                    indent=2,
                )
            )
        src = "colabfold-a3m" if msa_dir is not None else ("empty" if args.no_msa else "af3-data")
        print(f"[build-json] shard {args.shard_idx}/{args.n_shards}: {len(recs)} AF3 JSONs "
              f"(msa={src}, ligand={'yes' if args.ligand_smiles else 'no'}) -> {json_dir}")
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
