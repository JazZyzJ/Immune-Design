#!/usr/bin/env python3
"""
Phase 4: the go/no-go gate. Do not build masks from an unvalidated Potts model.

Three checks:

  1. EC precision against the crystal structure. Top-L long-range couplings are
     scored against contact maps built two ways -- the MONOMER (chain A only) and
     the TETRAMER (any pair of the four chains). Because uricase is a homotetramer
     and a single-family Potts model cannot separate intra- from inter-chain
     covariation, we expect a set of strong ECs that the monomer cannot explain
     but the tetramer can. That gap is the quantitative signature that the model
     is reading the oligomeric constraint, which is where 7 of the 15 active-site
     residues live.

     GATE: top-L precision against the tetramer map >= 0.60. Below that the
     alignment is too shallow or contaminated, and masks derived from sigma
     would be noise dressed up as signal.

  2. Functional-site recovery. How well C_i and sigma_i rank the 15 residues
     within 4.5 A of the bound ligand -- and specifically whether sigma_i
     recovers the cross-subunit residues that a monomer distance cutoff misses.

  3. Depth: final N_eff/L against the nr90 baseline of 3.59.

Numbering: ECs are in UniProt (1..302); the structure is in mature numbering.
mature = uniprot - 1. See 00_target/reference.yml.

Usage
  python 15_validate.py --ecs 02_couplings/deep_ECs.txt \
                        --scores 03_scores/per_residue_scores.tsv \
                        --out 05_validation/report.txt
"""

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from evcouplings.couplings.pairs import read_raw_ec_file

warnings.simplefilter("ignore")

PROJECT = Path(__file__).resolve().parent.parent
CONTACT_CUTOFF = 5.0     # heavy-atom, Angstrom
MIN_SEQDIST = 6
GATE_PRECISION = 0.60


def load_reference():
    from ruamel.yaml import YAML
    with open(PROJECT / "00_target" / "reference.yml") as fh:
        return YAML(typ="safe").load(fh)


def contact_maps(cif, cutoff=CONTACT_CUTOFF):
    """
    Build monomer and tetramer residue-contact sets in mature numbering.

    Returns (monomer, tetramer, resolved) where the first two are sets of
    (i, j) with i < j, and `resolved` is the set of mature positions that
    actually have coordinates.
    """
    from Bio.PDB import MMCIFParser, NeighborSearch

    struct = MMCIFParser(QUIET=True).get_structure("x", cif)
    model = next(iter(struct))

    atoms, resolved = [], set()
    for chain in model:
        for res in chain:
            if res.id[0] != " ":          # skip waters/ligands/SAC
                continue
            resolved.add(res.id[1])
            for atom in res:
                if atom.element != "H":
                    atoms.append(atom)

    ns = NeighborSearch(atoms)
    monomer, tetramer = set(), set()
    ref_chain = model.child_list[0].id

    for r1, r2 in ns.search_all(cutoff, level="R"):
        i, j = r1.id[1], r2.id[1]
        if i == j:
            continue
        c1, c2 = r1.get_parent().id, r2.get_parent().id
        pair = (min(i, j), max(i, j))
        tetramer.add(pair)
        if c1 == c2 == ref_chain:
            monomer.add(pair)

    return monomer, tetramer, resolved


def precision(ecs, contacts, resolved, n_top, min_seqdist=MIN_SEQDIST):
    """Fraction of the top-n long-range ECs that are contacts, ignoring pairs
    we cannot evaluate because a residue is unresolved."""
    lr = ecs.query("abs(i-j) >= @min_seqdist").sort_values("cn", ascending=False)
    hits = tot = 0
    unscorable = 0
    used = []
    for _, r in lr.iterrows():
        # UniProt -> mature
        i, j = int(r.i) - 1, int(r.j) - 1
        if i not in resolved or j not in resolved:
            unscorable += 1
            continue
        pair = (min(i, j), max(i, j))
        hits += pair in contacts
        tot += 1
        used.append((pair, pair in contacts, r.cn))
        if tot >= n_top:
            break
    return (hits / tot if tot else float("nan")), tot, unscorable, used


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecs", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--structure", default=str(PROJECT / "00_target" / "1r4u_assembly1.cif"))
    p.add_argument("--out", default=str(PROJECT / "05_validation" / "report.txt"))
    p.add_argument("--baseline-neff-per-l", type=float, default=3.59)
    args = p.parse_args()

    ref = load_reference()
    site_same = set(ref["active_site_4p5A"]["same_chain"])
    site_cross = set(ref["active_site_4p5A"]["cross_chain"])
    site = site_same | site_cross

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("PHASE 4 VALIDATION")
    emit("=" * 72)

    scores = pd.read_csv(args.scores, sep="\t")
    L = len(scores)
    ecs = read_raw_ec_file(args.ecs, sort=True, score="cn")

    emit("\n[1] EC precision vs structure")
    mono, tetra, resolved = contact_maps(args.structure)
    emit(f"    structure          : {Path(args.structure).name}")
    emit(f"    residues resolved  : {len(resolved)} (mature {min(resolved)}-{max(resolved)})")
    emit(f"    contacts <= {CONTACT_CUTOFF} A   : monomer {len(mono)}, tetramer {len(tetra)}"
         f"  (+{len(tetra) - len(mono)} from inter-chain)")

    n_pos = len(set(ecs.i) | set(ecs.j))
    emit(f"\n    {'top-N':>8} {'vs monomer':>12} {'vs tetramer':>13} {'scored':>8} {'skipped':>8}")
    gate_prec = None
    for frac, label in ((0.5, "L/2"), (1.0, "L"), (2.0, "2L")):
        n = int(round(frac * n_pos))
        pm, tot, unsc, used = precision(ecs, mono, resolved, n)
        pt, _, _, _ = precision(ecs, tetra, resolved, n)
        emit(f"    {label:>8} {pm:12.3f} {pt:13.3f} {tot:8d} {unsc:8d}")
        if frac == 1.0:
            gate_prec = pt
            top_used = used

    # top_used carries `pair in mono` as its hit flag (it came from the monomer
    # call), so ask the two sets directly rather than reusing that flag.
    inter_only = [(pair, cn) for pair, _, cn in top_used
                  if pair in tetra and pair not in mono]
    emit(f"\n    top-L ECs that ONLY the tetramer explains: {len(inter_only)}"
         f"  ({100 * len(inter_only) / max(len(top_used), 1):.1f}% of scored)")
    if inter_only:
        preview = ", ".join(f"{i}-{j}" for (i, j), _ in inter_only[:12])
        emit(f"      e.g. {preview}")
        emit("      These are the inter-subunit couplings a monomer-based mask cannot see.")

    emit(f"\n    GATE: top-L precision vs tetramer = {gate_prec:.3f} "
         f"(threshold {GATE_PRECISION:.2f}) -> {'PASS' if gate_prec >= GATE_PRECISION else 'FAIL'}")

    emit("\n[2] Functional-site recovery")
    s = scores.set_index("pos_mature")
    emit(f"    {'mature':>7} {'aa':>3} {'C_nogap':>8} {'C_rank':>7} {'sigma':>8} {'s_rank':>7}  origin")
    for pos in sorted(site):
        if pos not in s.index:
            continue
        r = s.loc[pos]
        origin = "CROSS-SUBUNIT" if pos in site_cross else "same-chain"
        emit(f"    {pos:7d} {r.wt_aa:>3} {r.C_i_nogap:8.3f} {int(r.C_rank):7d} "
             f"{r.sigma_raw:8.3f} {int(r.sigma_rank):7d}  {origin}")

    for name, subset in (("all 15", site), ("cross-subunit", site_cross), ("same-chain", site_same)):
        sub = [p for p in subset if p in s.index]
        cr = np.median([s.loc[p, "C_rank"] for p in sub])
        sr = np.median([s.loc[p, "sigma_rank"] for p in sub])
        emit(f"    median rank ({name:>13}): C {cr:5.0f}/{L}   sigma {sr:5.0f}/{L}")

    emit("\n    residues each score would MISS at its top-25% cut:")
    for col, label in (("C_pct", "C_nogap"), ("sigma_pct", "sigma")):
        missed = sorted(p for p in site if p in s.index and s.loc[p, col] < 0.75)
        emit(f"      {label:>8}: {missed if missed else 'none'}")
    both = sorted(p for p in site if p in s.index
                  and s.loc[p, "C_pct"] < 0.75 and s.loc[p, "sigma_pct"] < 0.75)
    emit(f"      {'union':>8}: {both if both else 'none'}   <- what dual masking still misses")

    emit("\n[3] Alignment depth")
    emit(f"    baseline (nr90)  N_eff/L = {args.baseline_neff_per_l:.2f}")
    emit("    (this run's N_eff/L is reported by 13_run_plmc.py / 14_scores.py)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    emit(f"\nwrote {out}")

    raise SystemExit(0 if gate_prec >= GATE_PRECISION else 1)


if __name__ == "__main__":
    main()
