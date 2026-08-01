#!/usr/bin/env python3
"""
Phase 2: fit the Potts model with plmc.

Deliberately goes through evcouplings.couplings.tools.run_plmc rather than
shelling out to plmc directly. Two conventions live in that wrapper that are
easy to get wrong by hand and silently produce a garbage model:

  1. theta. plmc's -t is 1 - theta. Passing -t 0.8 when you mean "cluster at 80%
     identity" actually clusters at 20% identity, collapsing the alignment.
  2. lambda_J. EVcouplings' standard protocol scales the configured lambda_J by
     (num_symbols - 1) * (L - 1) to compensate for there being far more J_ij
     parameters than h_i. For L=302, q=21 that is 0.01 -> 60.2. Using the raw
     0.01 leaves the couplings badly under-regularized.

We keep control of the input alignment ourselves (rather than using the
align:existing protocol) so that no extra column filtering happens between
12_prepare_alignment.py and the model -- alignment column j stays UniProt
position j+1 / mature position j all the way through.

Usage
  python 13_run_plmc.py --alignment 01_msa/nr90_baseline_focus.fasta \
                        --prefix 02_couplings/nr90_baseline --cpu 32
"""

import argparse
import json
import time
from pathlib import Path

from evcouplings.align.alignment import read_fasta
from evcouplings.couplings.tools import run_plmc

PROJECT = Path(__file__).resolve().parent.parent
PLMC_BIN = PROJECT / "env" / "plmc" / "bin" / "plmc"

# EVcouplings "standard" couplings protocol defaults.
ALPHABET_PROTEIN = "-ACDEFGHIKLMNPQRSTVWY"  # gap first, by plmc convention
THETA = 0.8
LAMBDA_H = 0.01
LAMBDA_J = 0.01
ITERATIONS = 100


def scaled_lambda_J(alignment, lambda_J=LAMBDA_J, alphabet=ALPHABET_PROTEIN):
    """Reproduce the standard protocol's lambda_J *= (q-1)*(L-1) scaling."""
    with open(alignment) as fh:
        _, seq = next(read_fasta(fh))
    gap = alphabet[0]
    L = len([c for c in seq if c == c.upper() or c == gap])
    return lambda_J * (len(alphabet) - 1) * (L - 1), L


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alignment", required=True, help="focus alignment from 12_prepare_alignment.py")
    p.add_argument("--prefix", required=True, help="output prefix, e.g. 02_couplings/nr90_baseline")
    p.add_argument("--focus-seq", default="Q00511", help="substring identifying the focus sequence")
    p.add_argument("--theta", type=float, default=THETA)
    p.add_argument("--iterations", type=int, default=ITERATIONS)
    p.add_argument("--cpu", default="max")
    args = p.parse_args()

    aln = Path(args.alignment)
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # plmc's -f wants the exact sequence identifier, not a substring
    with open(aln) as fh:
        first_id, _ = next(read_fasta(fh))
    if args.focus_seq not in first_id:
        raise SystemExit(
            f"first sequence in {aln} is {first_id!r}, which does not contain "
            f"{args.focus_seq!r}. 12_prepare_alignment.py puts the target first; "
            "check the input."
        )

    lam_J, L = scaled_lambda_J(aln)
    print(f"alignment : {aln}")
    print(f"focus seq : {first_id}")
    print(f"L         : {L}")
    print(f"theta     : {args.theta}  (plmc receives -t {1 - args.theta:.2f})")
    print(f"lambda_h  : {LAMBDA_H}")
    print(f"lambda_J  : {lam_J:.4g}  (= {LAMBDA_J} * {len(ALPHABET_PROTEIN) - 1} * {L - 1})")
    print(f"iterations: {args.iterations}")
    print(f"plmc      : {PLMC_BIN}", flush=True)

    t0 = time.time()
    res = run_plmc(
        alignment=str(aln),
        couplings_file=str(prefix) + "_ECs.txt",
        param_file=str(prefix) + ".model",
        focus_seq=first_id,
        alphabet=None,           # None => plmc's built-in protein alphabet (focus-mode optimized)
        theta=args.theta,        # wrapper applies the 1-theta transform
        scale=None,
        ignore_gaps=False,
        iterations=args.iterations,
        lambda_h=LAMBDA_H,
        lambda_J=lam_J,
        lambda_g=None,
        cpu=args.cpu,
        binary=str(PLMC_BIN),
    )
    dt = time.time() - t0

    summary = {
        "alignment": str(aln),
        "focus_seq": first_id,
        "L": L,
        "theta": args.theta,
        "lambda_h": LAMBDA_H,
        "lambda_J_scaled": lam_J,
        "iterations": args.iterations,
        "runtime_sec": round(dt, 1),
        "num_total_seqs": res.num_total_seqs,
        "num_valid_seqs": res.num_valid_seqs,
        "num_total_sites": res.num_total_sites,
        "num_valid_sites": res.num_valid_sites,
        "focus_seq_index": res.focus_seq_index,
        "region_start": res.region_start,
        "effective_samples": res.effective_samples,
        "opt_status": res.optimization_status,
    }
    print("\n=== plmc summary ===")
    for k, v in summary.items():
        print(f"  {k:18s} {v}")

    n_eff = summary["effective_samples"]
    if n_eff:
        print(f"\n  N_eff/L = {n_eff / L:.2f}"
              f"   {'(good)' if n_eff / L >= 10 else '(marginal - couplings will be noisy)' if n_eff / L < 5 else '(usable)'}")

    with open(str(prefix) + "_plmc_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {prefix}.model, {prefix}_ECs.txt, {prefix}_plmc_summary.json")


if __name__ == "__main__":
    main()
