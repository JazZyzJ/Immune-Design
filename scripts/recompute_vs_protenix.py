#!/usr/bin/env python
"""Recompute design-vs-WT structural metrics using a **Protenix-WT** reference (instead of AF3-WT).

Rationale: AF3 and Protenix disagree on FP chromophore sidechains (~4 Å on mCherry Tyr), so comparing
Protenix-refold designs against an AF3 baseline is biased; the self-consistent baseline is the
Protenix-WT prediction. Per design (best sample): TM vs Protenix-WT (+ optional AF3-WT for contrast),
pocket CA-RMSD, max active-site-anchor sidechain RMSD, chromophore-triad sidechain RMSD — all after a
local active-site-pocket CA superposition. Design-intrinsic plddt/ptm/has_clash pass through.
"""
import argparse, os, glob, json, re, subprocess
import numpy as np, pandas as pd
from multiprocessing import Pool
import biotite.structure.io.pdbx as pdbx, biotite.structure.io.pdb as pdb, biotite.structure as struc

# res_id = index_0b + 1
ANCHORS = {
    "egfp":    dict(anchors=[65, 66, 67, 68, 97, 149, 204, 223], chromo=[66, 67, 68], tyr=67),
    "mcherry": dict(anchors=[71, 72, 73, 75, 98, 100, 168, 202, 220], chromo=[71, 72, 73], tyr=72),
}
BB = {"N", "CA", "C", "O", "OXT"}


def load_prot(path):
    if path.endswith(".pdb"):
        a = pdb.get_structure(pdb.PDBFile.read(path), model=1)
    else:
        a = pdbx.get_structure(pdbx.CIFFile.read(path), model=1, extra_fields=["b_factor"])
    return a[struc.filter_amino_acids(a)]


def sc_rmsd(mob_fit, ref, rid, tyr_sym=False):
    p = mob_fit[(mob_fit.res_id == rid) & (~np.isin(mob_fit.atom_name, list(BB))) & (mob_fit.element != "H")]
    w = ref[(ref.res_id == rid) & (~np.isin(ref.atom_name, list(BB))) & (ref.element != "H")]
    if len(p) == 0 or len(w) == 0:
        return None
    def rmsd_with(nmap):
        idx = [(i, nmap[n]) for i, n in enumerate(p.atom_name) if n in nmap]
        if not idx:
            return None
        pc = p.coord[[i for i, _ in idx]]; wc = w.coord[[j for _, j in idx]]
        return float(np.sqrt(((pc - wc) ** 2).sum(1).mean()))
    r0 = rmsd_with({n: i for i, n in enumerate(w.atom_name)})
    if not tyr_sym:
        return r0
    swap = {"CD1": "CD2", "CD2": "CD1", "CE1": "CE2", "CE2": "CE1"}
    r1 = rmsd_with({swap.get(n, n): i for i, n in enumerate(w.atom_name)})
    return min(x for x in (r0, r1) if x is not None)


def tm(usalign, cif, ref):
    out = subprocess.run([usalign, cif, ref], capture_output=True, text=True).stdout
    tms = [float(m) for m in re.findall(r"TM-score=\s*([\d.]+)", out)]
    return max(tms) if tms else np.nan


def best(pred_root, name):
    pdirs = glob.glob(f"{pred_root}/{name}/*/seed_*/predictions")
    if not pdirs:
        return None
    pdir = pdirs[0]; b = None
    for js in glob.glob(f"{pdir}/{name}_summary_confidence_sample_*.json"):
        j = json.load(open(js)); k = int(re.search(r"_sample_(\d+)\.json", js).group(1))
        rs = float(j.get("ranking_score", float("nan"))); cif = f"{pdir}/{name}_sample_{k}.cif"
        if os.path.exists(cif) and (b is None or (rs == rs and rs > b[0])):
            b = (rs, j, cif)
    return b


def _work(a):
    pred_root, pid, spec, wt_cif, af3_ref, usalign, wt_prot = a
    bb = best(pred_root, f"mono_{pid}")
    if not bb:
        return dict(pred_id=pid, predicted=False)
    rs, j, cif = bb
    rec = dict(pred_id=pid, predicted=True, ranking_score=rs, plddt=float(j.get("plddt", np.nan)),
               ptm=float(j.get("ptm", np.nan)), has_clash=bool(j.get("has_clash", False)),
               tm_vs_protenix_wt=tm(usalign, cif, wt_cif))
    if af3_ref:
        rec["tm_vs_af3_wt"] = tm(usalign, cif, af3_ref)
    prot = load_prot(cif)
    dca = {int(r): i for i, r in zip(prot.res_id[prot.atom_name == "CA"], range(int((prot.atom_name == "CA").sum())))}
    dca_a = prot[prot.atom_name == "CA"]
    wca = {int(r): i for i, r in zip(wt_prot.res_id[wt_prot.atom_name == "CA"], range(int((wt_prot.atom_name == "CA").sum())))}
    wca_a = wt_prot[wt_prot.atom_name == "CA"]
    common = [r for r in spec["anchors"] if r in dca and r in wca]
    if len(common) >= 4:
        dfix = dca_a[[dca[r] for r in common]]; wfix = wca_a[[wca[r] for r in common]]
        _, T = struc.superimpose(wfix, dfix)
        rec["pocket_ca_rmsd_vs_protenix_wt"] = float(struc.rmsd(wfix.coord, T.apply(dfix).coord))
        pf = T.apply(prot)
        anc = [sc_rmsd(pf, wt_prot, r, tyr_sym=(r == spec["tyr"])) for r in spec["anchors"]]
        anc = [x for x in anc if x is not None]
        rec["active_site_max_sc_rmsd_vs_protenix_wt"] = float(np.max(anc)) if anc else np.nan
        chromo = [sc_rmsd(pf, wt_prot, r, tyr_sym=(r == spec["tyr"])) for r in spec["chromo"]]
        chromo = [x for x in chromo if x is not None]
        rec["chromophore_max_sc_rmsd_vs_protenix_wt"] = float(np.max(chromo)) if chromo else np.nan
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", required=True, choices=["egfp", "mcherry"])
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--wt-cif", required=True, help="Protenix-WT reference CIF")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--usalign", required=True)
    ap.add_argument("--af3-ref", default=None, help="optional AF3-WT pdb, kept as tm_vs_af3_wt for contrast")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    spec = ANCHORS[args.protein]
    wt_prot = load_prot(args.wt_cif)
    man = pd.read_parquet(args.manifest)
    tasks = [(args.pred_root, r.pred_id, spec, args.wt_cif, args.af3_ref, args.usalign, wt_prot)
             for r in man.itertuples()]
    with Pool(args.procs) as p:
        recs = p.map(_work, tasks)
    drop = [c for c in ["sequence_pred", "sequence_refined"] if c in man.columns]
    res = man.drop(columns=drop).merge(pd.DataFrame(recs), on="pred_id", how="left")
    res.to_parquet(args.out); res.to_csv(args.out.replace(".parquet", ".csv"), index=False)
    sub = res[res.predicted == True]
    print(f"{args.protein}: {len(sub)}/{len(res)} | TM vs Protenix-WT mean {sub.tm_vs_protenix_wt.mean():.3f} "
          f"min {sub.tm_vs_protenix_wt.min():.3f} | active-site max scRMSD mean "
          f"{sub.active_site_max_sc_rmsd_vs_protenix_wt.mean():.2f} | chromophore max scRMSD mean "
          f"{sub.chromophore_max_sc_rmsd_vs_protenix_wt.mean():.2f}")


if __name__ == "__main__":
    main()
