#!/usr/bin/env python3
"""
Phase 1b: turn a raw alignment into a focus alignment on Q00511's columns, and
report what each filtering threshold costs in effective depth.

Works on either input we care about:
  * an aligned FASTA (e.g. 03_alignment/raw/nr90_aln.fasta) - focus columns are
    the columns where the target row is not a gap;
  * an a3m from colabfold_search - focus columns are the uppercase/gap columns,
    lowercase being insertions relative to the query.

Either way the output has exactly one column per target residue: 0-based column j
is UniProt position j+1 and mature position j. See 00_target/reference.yml for why
the mature convention (UniProt minus 1) is what the literature and PDB use.

N_eff is computed with EVcouplings' own Alignment.set_weights(), i.e. the same
reweighting plmc will apply, rather than a lookalike of our own.

Usage
  # report the threshold sweep only
  python 12_prepare_alignment.py --input <aln> --sweep

  # write the chosen focus alignment
  python 12_prepare_alignment.py --input <aln> --min-coverage 0.6 --out-prefix 01_msa/nr90
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from evcouplings.align import Alignment

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = "Q00511"
THETA = 0.8  # EVcouplings/plmc default reweighting identity threshold


def read_fasta(path):
    """Yield (id, seq) preserving file order."""
    name, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:], []
            else:
                buf.append(line)
    if name is not None:
        yield name, "".join(buf)


def to_focus(records, target_id, fmt):
    """
    Reduce an alignment to the target's own columns.

    Returns (ids, matrix) where matrix is an (N, L) char array and column j is
    UniProt position j+1 of Q00511 (so j+1 includes the initiator Met; mature
    numbering is j, i.e. one less -- see 00_target/reference.yml).
    """
    ids = [r[0] for r in records]
    hits = [i for i, n in enumerate(ids) if target_id in n]
    if not hits:
        raise SystemExit(f"target {target_id!r} not found among {len(ids)} sequences")
    if len(hits) > 1:
        print(f"  note: {len(hits)} headers match {target_id!r}; using the first", file=sys.stderr)
    t_idx = hits[0]
    target = records[t_idx][1]

    rows = []
    if fmt == "a3m":
        # a3m rows vary in length only through lowercase inserts relative to the
        # query; strip those and every row is already exactly the query's length.
        L = sum(1 for c in target if not c.islower())
        for _, seq in records:
            s = "".join(c for c in seq if not c.islower()).replace(".", "-")
            if len(s) != L:
                s = s[:L].ljust(L, "-")
            rows.append(np.frombuffer(s.encode(), dtype="S1"))
    else:
        keep = np.array([j for j, c in enumerate(target) if c not in "-."])
        L = len(keep)
        for _, seq in records:
            s = seq.ljust(len(target), "-")
            rows.append(np.frombuffer(s.encode(), dtype="S1")[keep])

    matrix = np.vstack(rows)
    matrix[matrix == b"."] = b"-"

    # move target to row 0 so downstream tools that assume "first sequence is the
    # reference" cannot get it wrong
    order = [t_idx] + [i for i in range(len(ids)) if i != t_idx]
    return [ids[i] for i in order], matrix[order]


def n_eff(ids, matrix, theta=THETA, max_exact=60000, seed=0):
    """
    Effective sequence count via EVcouplings' reweighting.

    evcouplings 0.2.1's num_cluster_members is a single-threaded @jit doing the
    full N(N-1)/2 comparison, so cost is O(N^2 L) on one core.

    WARNING: the subsample path above max_exact is BIASED LOW -- do not trust it.
    The rescaling-free argument (cluster of size m shrinks to f*m, so weight
    1/m -> 1/(f*m) and the sum is preserved) holds only for large clusters. A
    singleton cannot shrink below 1: its weight stays 1 while only f as many
    singletons are retained, so every small cluster is undercounted, and worse at
    smaller f. On the 17305-sequence colabfold a3m, min_cov=0 measured:

        max_exact  4000 -> N_eff/L  6.10   (f=0.23)
        max_exact  9000 -> N_eff/L 11.03   (f=0.52)
        exact           -> N_eff/L 17.16

    Exact on that alignment takes 27 s, so the cap earns nothing at this scale;
    it is kept only as a guard against a pathologically large input. Default is
    now well above any alignment we expect, i.e. exact in practice.

    (plmc itself reweights in C with OpenMP, so this only ever affected the sweep
    report, never the model that gets fit.)
    """
    n = matrix.shape[0]
    sub = None
    if n > max_exact:
        print(
            f"WARNING: subsampling {n} -> {max_exact} for N_eff; the result is "
            "BIASED LOW (see n_eff docstring). Raise --max-exact for a real number.",
            file=sys.stderr,
        )
        rng = np.random.default_rng(seed)
        sub = np.sort(rng.choice(n, max_exact, replace=False))
        if 0 not in sub:  # always retain the target
            sub[0] = 0
        ids_u = [ids[i] for i in sub]
        mat_u = matrix[sub]
    else:
        ids_u, mat_u = ids, matrix

    seqs = ((i, "".join(c.decode() for c in row)) for i, row in zip(ids_u, mat_u))
    ali = Alignment.from_dict(dict(seqs))
    ali.set_weights(identity_threshold=theta)
    return float(ali.weights.sum()), (sub is not None)


def sweep(ids, matrix, coverages, theta=THETA, max_exact=None):
    """Report N_seqs / N_eff / N_eff-per-L across minimum-coverage cutoffs."""
    L = matrix.shape[1]
    cov = 1.0 - (matrix == b"-").mean(axis=1)
    rows = []
    for c in coverages:
        keep = cov >= c
        keep[0] = True  # never drop the target
        k = int(keep.sum())
        if k < 10:
            rows.append((c, k, float("nan"), float("nan"), False))
            continue
        kw = {} if max_exact is None else {"max_exact": max_exact}
        ne, approx = n_eff([ids[i] for i in np.flatnonzero(keep)], matrix[keep], theta, **kw)
        rows.append((c, k, ne, ne / L, approx))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="aligned FASTA or a3m")
    p.add_argument("--format", choices=["auto", "fasta", "a3m"], default="auto")
    p.add_argument("--target-id", default=DEFAULT_TARGET)
    p.add_argument("--theta", type=float, default=THETA)
    p.add_argument("--sweep", action="store_true", help="report thresholds and exit")
    p.add_argument("--min-coverage", type=float, default=0.5)
    p.add_argument("--max-exact", type=int, default=None,
                   help="subsample above this many sequences for N_eff; the estimate "
                        "is biased LOW, so prefer exact (default) unless it is too slow")
    p.add_argument("--out-prefix", help="write <prefix>_focus.fasta and <prefix>_stats.tsv")
    args = p.parse_args()

    fmt = args.format
    if fmt == "auto":
        fmt = "a3m" if args.input.endswith((".a3m", ".a3m.gz")) else "fasta"

    records = list(read_fasta(args.input))
    print(f"read {len(records)} sequences from {args.input} (format={fmt})")

    ids, matrix = to_focus(records, args.target_id, fmt)
    N, L = matrix.shape
    print(f"focus alignment: N={N}  L={L}  (target = {ids[0][:70]})")
    if L != 302:
        print(f"  WARNING: expected L=302 UniProt positions for Q00511, got {L}", file=sys.stderr)

    coverages = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8]
    print(f"\ncoverage sweep (theta={args.theta}):")
    print(f"  {'min_cov':>8} {'N_seqs':>9} {'N_eff':>10} {'N_eff/L':>9}  {'':>6}")
    rows = sweep(ids, matrix, coverages, args.theta, args.max_exact)
    for c, k, ne, nel, approx in rows:
        flag = "approx" if approx else ""
        print(f"  {c:8.2f} {k:9d} {ne:10.1f} {nel:9.2f}  {flag:>6}")

    if args.sweep:
        return

    if not args.out_prefix:
        raise SystemExit("--out-prefix required unless --sweep")

    cov = 1.0 - (matrix == b"-").mean(axis=1)
    keep = cov >= args.min_coverage
    keep[0] = True
    idx = np.flatnonzero(keep)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fa = out_prefix.with_name(out_prefix.name + "_focus.fasta")
    with open(fa, "w") as fh:
        for i in idx:
            seq = "".join(c.decode() for c in matrix[i])
            fh.write(f">{ids[i]}\n")
            for s in range(0, len(seq), 60):
                fh.write(seq[s:s + 60] + "\n")
    print(f"\nwrote {fa}  (N={len(idx)}, L={L}, min_coverage={args.min_coverage})")

    st = out_prefix.with_name(out_prefix.name + "_stats.tsv")
    with open(st, "w") as fh:
        fh.write("min_coverage\tn_seqs\tn_eff\tn_eff_per_L\tapprox\n")
        for c, k, ne, nel, approx in rows:
            fh.write(f"{c}\t{k}\t{ne:.2f}\t{nel:.3f}\t{int(approx)}\n")
    print(f"wrote {st}")


if __name__ == "__main__":
    main()
