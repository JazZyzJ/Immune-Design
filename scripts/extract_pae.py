#!/usr/bin/env python
"""Extract inter-chain PAE information from Protenix full_data JSONs (which are ~68MB each and
too large to ship). Reduces token_pair_{pae,pde} + contact_probs to per-entity-pair block means
(8x8: 4 protein chains + 4 ligands) plus protein-protein inter/intra summaries. Numbering-robust
(uses token_asym_id groups only). Saves a compact npz per design + a scalar summary table.

Per design we process all 5 diffusion samples and keep them (axis 0 = sample).
"""
import argparse, json, os, glob
import numpy as np
import pandas as pd
from multiprocessing import Pool

FIELDS = ["token_pair_pae", "token_pair_pde", "contact_probs"]


def block_means(mat, groups):
    """mat: (T,T); groups: list of index arrays per asym. Returns (G,G) block-mean."""
    G = len(groups)
    out = np.full((G, G), np.nan, dtype=np.float32)
    for a in range(G):
        ia = groups[a]
        if ia.size == 0:
            continue
        sub = mat[np.ix_(ia, range(mat.shape[1]))]
        for b in range(G):
            ib = groups[b]
            if ib.size == 0:
                continue
            out[a, b] = float(sub[:, ib].mean())
    return out


def find_samples(pred_root, name):
    dirs = glob.glob(os.path.join(pred_root, name, "*", "seed_*", "predictions"))
    if not dirs:
        return None
    pdir = dirs[0]
    samples = []
    for k in range(16):
        f = os.path.join(pdir, f"{name}_full_data_sample_{k}.json")
        s = os.path.join(pdir, f"{name}_summary_confidence_sample_{k}.json")
        if os.path.isfile(f) and os.path.isfile(s):
            samples.append((k, f, s))
    return samples if samples else None


def process_one(task):
    pred_root, out_dir, pred_id, design_uid = task
    name = f"tetra_{pred_id}"
    samples = find_samples(pred_root, name)
    if not samples:
        return {"pred_id": pred_id, "design_uid": design_uid, "predicted": False}
    cp_pae, cp_pde, cp_con, rankscores = [], [], [], []
    asym = None
    prot_asym = None
    summ_best = None
    best_rank = -1e9
    for k, fj, sj in samples:
        d = json.load(open(fj))
        a = np.asarray(d["token_asym_id"], dtype=np.int64)
        if asym is None:
            asym = a
            uniq = sorted(set(a.tolist()))
            groups = [np.where(a == u)[0] for u in uniq]
            sizes = np.array([g.size for g in groups])
            # protein entities = the large groups (residue tokens); ligands are small
            prot_asym = [i for i, s in enumerate(sizes) if s >= max(50, sizes.max() // 2)]
        else:
            uniq = sorted(set(a.tolist()))
            groups = [np.where(a == u)[0] for u in uniq]
        pae = np.asarray(d["token_pair_pae"], dtype=np.float32)
        pde = np.asarray(d["token_pair_pde"], dtype=np.float32)
        con = np.asarray(d["contact_probs"], dtype=np.float32)
        cp_pae.append(block_means(pae, groups))
        cp_pde.append(block_means(pde, groups))
        cp_con.append(block_means(con, groups))
        sj_d = json.load(open(sj))
        rs = float(sj_d.get("ranking_score", float("nan")))
        rankscores.append(rs)
        if rs == rs and rs > best_rank:
            best_rank, summ_best = rs, sj_d
    cp_pae = np.stack(cp_pae); cp_pde = np.stack(cp_pde); cp_con = np.stack(cp_con)  # (S,G,G)
    np.savez_compressed(
        os.path.join(out_dir, f"{pred_id}_pae.npz"),
        chain_pair_pae=cp_pae, chain_pair_pde=cp_pde, chain_pair_contact=cp_con,
        token_asym_id=asym.astype(np.int16), protein_asym=np.array(prot_asym, dtype=np.int16),
        ranking_score=np.array(rankscores, dtype=np.float32),
    )
    # protein-protein inter/intra PAE from the best sample (highest ranking_score)
    best_idx = int(np.nanargmax(rankscores))
    P = prot_asym
    ppae = cp_pae[best_idx][np.ix_(P, P)]
    off = ppae[~np.eye(len(P), dtype=bool)]
    diag = np.diag(ppae)
    rec = {
        "pred_id": pred_id, "design_uid": design_uid, "predicted": True,
        "n_samples": cp_pae.shape[0], "n_protein_chains": len(P),
        "best_sample": best_idx, "ranking_score": best_rank,
        "mean_interchain_pae": float(off.mean()),
        "max_interchain_pae": float(off.max()),
        "min_chainpair_interchain_pae": float(off.min()),
        "mean_intrachain_pae": float(diag.mean()),
    }
    if summ_best is not None:
        cpi = np.asarray(summ_best.get("chain_pair_iptm", []), dtype=float)
        cpg = np.asarray(summ_best.get("chain_pair_gpde", []), dtype=float)
        pp = [(i, j) for i in P for j in P if i < j]
        rec.update(
            iptm=float(summ_best.get("iptm", np.nan)),
            ptm=float(summ_best.get("ptm", np.nan)),
            plddt=float(summ_best.get("plddt", np.nan)),
            has_clash=bool(summ_best.get("has_clash", False)),
            min_pp_chain_pair_iptm=float(min(cpi[i, j] for i, j in pp)) if cpi.size else np.nan,
            mean_pp_chain_pair_gpde=float(np.mean([cpg[i, j] for i, j in pp])) if cpg.size else np.nan,
        )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-table", required=True)
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    man = pd.read_parquet(args.manifest)
    os.makedirs(args.out_dir, exist_ok=True)
    uid_col = "design_uid" if "design_uid" in man.columns else ("design_id" if "design_id" in man.columns else "pred_id")
    tasks = [(args.pred_root, args.out_dir, r.pred_id, getattr(r, uid_col)) for r in man.itertuples()]
    with Pool(args.procs) as pool:
        recs = pool.map(process_one, tasks)
    df = pd.DataFrame(recs)
    df.to_parquet(args.out_table)
    df.to_csv(args.out_table.replace(".parquet", ".csv"), index=False)
    ok = int(df["predicted"].sum())
    print(f"extracted PAE for {ok}/{len(df)} designs -> {args.out_table}")
    if ok:
        print(df[df.predicted].describe(include=[np.number]).T[["mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
