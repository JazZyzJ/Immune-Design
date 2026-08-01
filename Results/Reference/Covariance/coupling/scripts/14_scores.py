#!/usr/bin/env python3
"""
Phase 3: the two per-residue scores the masking strategy runs on.

C_i  positional conservation, EVcouplings' Alignment.conservation(normalize=True)
     with sequence reweighting on, i.e. 1 - H_i/log2(q) over reweighted single-site
     frequencies. Range 0 (variable) to 1 (invariant) -- the same scale as the
     paper's C0 = 0.25/0.35/0.65.

sigma_i  cumulative coupling strength, EVcouplings' pairs.enrichment(): of the
     top-L long-range (|i-j| >= 6) APC-corrected couplings, sum each position's
     cn across the pairs it participates in, divided by the mean cn of those
     top-L pairs.

     This is the adaptation the port requires. In the TnpB paper sigma is a
     protein<->nucleobase coupling from a PAIRED protein/nucleic-acid alignment.
     Uricase binds a small molecule, not a sequence, so there is no partner
     alignment to pair with. The intra-protein enrichment score is the closest
     faithful analogue, and it was introduced (Hopf et al., Cell 2012) precisely
     to surface functionally constrained residues -- catalytic sites, ligand
     pockets, oligomer interfaces -- that ordinary conservation misses.

Two traps this handles explicitly:

  * EVcouplings counts the gap as an alphabet symbol, which distorts C_i in BOTH
    directions. A mostly-gapped column concentrates frequency on the gap symbol,
    so it scores as spuriously CONSERVED (measured on the nr90 baseline: the 26
    columns above 50% gap average C_i 0.624 native vs 0.367 gap-excluded --
    mature 198 reads 0.706 native but is really 0.204). A moderately gapped
    column gets the opposite treatment: the gap acts as an extra symbol and
    DILUTES real conservation (His256: 0.826 native vs 0.939 gap-excluded).
    We emit C_i (native, for fidelity to EVcouplings), C_i_nogap (gap dropped and
    renormalized over the 20 amino acids), and gap_frac. C_i_nogap is what the
    masks key on.

  * enrichment() returns rows only for positions that appear in some top-L pair.
    Positions coupled to nothing silently vanish. We reindex onto the full
    length and fill 0.

Numbering: alignment column j (0-based) is UniProt position j+1 and mature
position j. The literature and every PDB entry use mature. See reference.yml.

Usage
  python 14_scores.py --alignment 01_msa/deep_focus.fasta \
                      --ecs 02_couplings/deep_ECs.txt \
                      --out 03_scores/per_residue_scores.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evcouplings.align import Alignment
from evcouplings.couplings.pairs import enrichment, read_raw_ec_file

PROJECT = Path(__file__).resolve().parent.parent
MIN_SEQDIST = 6          # exclude local backbone-neighbour couplings
NUM_PAIRS_MAIN = 1.0     # top-L pairs, as in Hopf et al.
NUM_PAIRS_SENS = (0.5, 1.0, 2.0)


def load_reference():
    """Read 00_target/reference.yml (ruamel ships with evcouplings)."""
    from ruamel.yaml import YAML
    with open(PROJECT / "00_target" / "reference.yml") as fh:
        return YAML(typ="safe").load(fh)


def conservation_scores(alignment_file, theta=0.8):
    """
    Returns (C_native, C_nogap, gap_frac, wt_seq), each length L.

    C_native reproduces EVcouplings exactly (gap is a symbol). C_nogap drops the
    gap column from each position's frequency vector, renormalizes over the
    amino acids present, and rescales entropy by log2(20) instead of log2(q).
    """
    with open(alignment_file) as fh:
        ali = Alignment.from_file(fh, format="fasta")
    ali.set_weights(identity_threshold=theta)

    C_native = ali.conservation(normalize=True)

    freqs = ali.frequencies                      # (L, q), reweighted
    alphabet = list(ali.alphabet)
    gap_idx = alphabet.index("-") if "-" in alphabet else 0
    gap_frac = freqs[:, gap_idx].copy()

    aa = np.delete(freqs, gap_idx, axis=1)       # (L, q-1)
    tot = aa.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        aa = np.where(tot > 0, aa / tot, 0.0)
    n_aa = aa.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(aa > 0, aa * np.log2(aa), 0.0), axis=1)
    C_nogap = 1.0 - H / np.log2(n_aa)

    wt = "".join(c for c in ali.matrix[0])
    return C_native, C_nogap, gap_frac, wt, ali


def coupling_scores(ec_file, L, num_pairs=NUM_PAIRS_MAIN, min_seqdist=MIN_SEQDIST):
    """
    Per-residue cumulative coupling strength, reindexed onto 1..L with 0 fill.

    Also returns the untruncated per-position cn sum as a robustness check --
    enrichment() depends on the top-L cut, this one does not.
    """
    ecs = read_raw_ec_file(ec_file, sort=True, score="cn")

    lo, hi = int(min(ecs.i.min(), ecs.j.min())), int(max(ecs.i.max(), ecs.j.max()))
    if (lo, hi) != (1, L):
        print(f"  WARNING: EC positions span {lo}-{hi}, expected 1-{L}. "
              "Check plmc focus-sequence parsing / region offset.")

    e = enrichment(ecs, num_pairs=num_pairs, score="cn", min_seqdist=min_seqdist)
    sigma = pd.Series(0.0, index=pd.RangeIndex(1, L + 1, name="i"))
    sigma.update(e.set_index("i")["enrichment"])

    lr = ecs.query("abs(i-j) >= @min_seqdist")
    tot = (
        pd.concat([lr[["i", "cn"]], lr[["j", "cn"]].rename(columns={"j": "i"})])
        .groupby("i")["cn"].sum()
    )
    cn_sum = pd.Series(0.0, index=pd.RangeIndex(1, L + 1, name="i"))
    cn_sum.update(tot)

    return sigma.to_numpy(), cn_sum.to_numpy(), ecs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alignment", required=True)
    p.add_argument("--ecs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--theta", type=float, default=0.8)
    args = p.parse_args()

    ref = load_reference()
    site = set(ref["active_site_4p5A"]["same_chain"]) | set(ref["active_site_4p5A"]["cross_chain"])
    cross = set(ref["active_site_4p5A"]["cross_chain"])

    print(f"conservation from {args.alignment} (theta={args.theta}) ...")
    C, C_nogap, gap_frac, wt, ali = conservation_scores(args.alignment, args.theta)
    L = len(C)
    print(f"  L={L}  N={ali.N}  N_eff={ali.weights.sum():.1f}  (N_eff/L={ali.weights.sum()/L:.2f})")

    print(f"couplings from {args.ecs} ...")
    sigma, cn_sum, ecs = coupling_scores(args.ecs, L)
    print(f"  {len(ecs)} pairs; {int((sigma > 0).sum())}/{L} positions appear in the top-L long-range set")

    pos_uni = np.arange(1, L + 1)
    df = pd.DataFrame({
        "pos_uniprot": pos_uni,
        "pos_mature": pos_uni - 1,          # 0 == initiator Met, absent from mature protein
        "wt_aa": list(wt),
        "C_i": C,
        "C_i_nogap": C_nogap,
        "gap_frac": gap_frac,
        "sigma_raw": sigma,
        "cn_sum": cn_sum,
    })
    df["sigma_pct"] = df.sigma_raw.rank(pct=True)
    df["C_pct"] = df.C_i_nogap.rank(pct=True)
    df["C_rank"] = df.C_i_nogap.rank(ascending=False).astype(int)
    df["sigma_rank"] = df.sigma_raw.rank(ascending=False).astype(int)
    df["in_mature"] = df.pos_mature >= 1
    df["active_site"] = df.pos_mature.isin(site)
    df["cross_subunit"] = df.pos_mature.isin(cross)

    # sensitivity: does the top-sigma set survive changing the top-L cut?
    # Reference must be the 1.0L run, not whichever happens to come first.
    print("\nsigma sensitivity to the top-N cut:")
    tops = {}
    for npairs in NUM_PAIRS_SENS:
        s, _, _ = coupling_scores(args.ecs, L, num_pairs=npairs)
        tops[npairs] = set(np.argsort(-s)[:30] + 1)
    base = tops[NUM_PAIRS_MAIN]
    for npairs in NUM_PAIRS_SENS:
        n = len(tops[npairs] & base)
        tag = "  <- reference" if npairs == NUM_PAIRS_MAIN else ""
        print(f"  num_pairs={npairs:>4}L : top-30 overlap with {NUM_PAIRS_MAIN}L = {n}/30{tag}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False, float_format="%.5f")
    print(f"\nwrote {out}  ({len(df)} rows)")

    known = df[df.active_site].sort_values("sigma_rank")
    print(f"\nthe {len(known)} structure-derived active-site residues:")
    print(f"  {'mature':>7} {'aa':>3} {'C_i':>7} {'C_rank':>7} {'sigma':>7} {'s_rank':>7}  {'':<6}")
    for _, r in known.iterrows():
        tag = "cross" if r.cross_subunit else ""
        print(f"  {int(r.pos_mature):7d} {r.wt_aa:>3} {r.C_i_nogap:7.3f} {int(r.C_rank):7d} "
              f"{r.sigma_raw:7.3f} {int(r.sigma_rank):7d}  {tag:<6}")


if __name__ == "__main__":
    main()
