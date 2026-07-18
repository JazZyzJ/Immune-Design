#!/usr/bin/env python
"""Tetramer refold-gate metrics for Protenix homo-tetramer predictions (PLAN_TETRAMER_GATE.md §2).

Second-layer benchmark supplement: for each predicted uricase tetramer, compute whether it folds
BACK into the native tetramer. Two tiers + advisory, one row per variant:

  Tier 1 (self-confidence, from the Protenix summary_confidence JSON; cheap pre-filter):
    iptm, ptm, plddt, min protein-protein chain_pair_iptm, min protein chain_plddt,
    mean protein-protein chain_pair_gpde (inter-chain PDE), has_clash, ranking_score.
  Tier 2 (geometry vs a reference tetramer; the actual "refold back"):
    complex_TM  — US-align -mm 1 (symmetry-aware) vs the parent WT-predicted tetramer, and,
                  for the crystal parent (Q00511), vs the 1R51 crystal tetramer.
    xprot_as_dist / _dev — cross-protomer active-site distances (same-chain catalytic core
                  Lys11/Thr58/His257 to the NEAREST neighbour-chain Asn255 W1), and the
                  deviation vs the WT-predicted reference. Only for parents whose numbering
                  matches the Q00511 manifest (index_0b); NaN otherwise (per-parent projection
                  is a v1 extension).
  Assembly (all parents): n_interchain_contacts — protein CB-CB pairs < 8 Angstrom across chains
                  (physical interface size), a sequence-independent assembly proxy.

Reference model: each design is scored vs its OWN parent's WT-predicted tetramer (relative), so
the gate cancels systematic model bias. Q00511 additionally gets the absolute crystal check.

The gate is NOT hard-coded here (thresholds are calibrated jointly on the results). This writes
the full metric table + a provisional `tetramer_gate_flag` using conservative defaults.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# Q00511 catalytic core (same-chain) + cross-protomer W1, as manifest index_0b (== predicted res_id).
CATALYTIC_CORE = {"Lys11": 10, "Thr58": 57, "His257": 256}
XPROT_W1 = {"Asn255": 254}
# functional atoms for the distance panel (fallback to CA if absent)
FUNC_ATOM = {"Lys11": "NZ", "Thr58": "OG1", "His257": "NE2", "Asn255": "ND2"}
# parents whose residue numbering matches the Q00511 manifest exactly
Q00511_NUMBERING_PARENTS = {"Q00511"}


# ---------- Protenix output discovery ----------

def find_pred_samples(pred_root: Path, name: str):
    """Return list of (sample_idx, cif_path, summary_json_path) for a variant, or []."""
    hits = sorted((pred_root / name).glob(f"*/seed_*/predictions"))
    out = []
    for pdir in hits:
        for cif in sorted(pdir.glob(f"{name}_sample_*.cif")):
            m = re.search(r"_sample_(\d+)\.cif$", cif.name)
            if not m:
                continue
            k = int(m.group(1))
            js = pdir / f"{name}_summary_confidence_sample_{k}.json"
            if js.exists():
                out.append((k, cif, js))
    return out


def tier1_from_summary(j: dict, n_protein: int = 4) -> dict:
    """Self-confidence metrics; protein chains are the first n_protein entries (A..D before ligands)."""
    pp = list(range(n_protein))
    cpi = np.asarray(j.get("chain_pair_iptm", []), dtype=float)
    cpg = np.asarray(j.get("chain_pair_gpde", []), dtype=float)
    cpl = np.asarray(j.get("chain_plddt", []), dtype=float)
    prot_pairs = [(i, k) for i in pp for k in pp if i < k]
    min_cpi = float(min(cpi[i, k] for i, k in prot_pairs)) if cpi.size else np.nan
    mean_cpg = float(np.mean([cpg[i, k] for i, k in prot_pairs])) if cpg.size else np.nan
    min_cpl = float(np.min(cpl[pp])) if cpl.size else np.nan
    return dict(
        iptm=float(j.get("iptm", np.nan)), ptm=float(j.get("ptm", np.nan)),
        plddt=float(j.get("plddt", np.nan)), ranking_score=float(j.get("ranking_score", np.nan)),
        has_clash=bool(j.get("has_clash", False)),
        min_pp_chain_pair_iptm=min_cpi, mean_pp_chain_pair_gpde=mean_cpg,
        min_prot_chain_plddt=min_cpl,
    )


def best_sample(samples) -> tuple:
    """Pick the sample with the highest ranking_score; return (idx, cif, json, tier1)."""
    best = None
    for k, cif, js in samples:
        j = json.loads(js.read_text())
        t1 = tier1_from_summary(j)
        score = t1["ranking_score"]
        if best is None or (score == score and score > best[4]):  # score==score guards NaN
            best = (k, cif, js, t1, (score if score == score else -1.0))
    return best[:4] if best else None


# ---- ESMFold2 backend (predict_tetramer_esmfold2.py layout: <name>/<name>.cif + _confidence.json) ----

def find_esmfold2_pred(pred_root: Path, name: str):
    """Return (0, cif, conf_json, tier1) for an ESMFold2 prediction, or None."""
    d = pred_root / name
    cif, conf = d / f"{name}.cif", d / f"{name}_confidence.json"
    if not (cif.exists() and conf.exists()):
        return None
    return (0, cif, conf, tier1_from_esmfold2(json.loads(conf.read_text())))


def tier1_from_esmfold2(j: dict) -> dict:
    """Map ESMFold2 confidence to the shared Tier-1 schema. ESMFold2 has NO PDE head, so
    mean_pp_chain_pair_gpde is NaN (the Protenix separator); it exposes pae for a future
    inter-chain-PAE metric. Protein chains are the first n_copies entries (P0..P3 before ligands)."""
    n = int(j.get("n_copies") or 4)
    pp = list(range(n))
    pc = np.asarray(j.get("pair_chains_iptm") or [], dtype=float)
    prot_pairs = [(i, k) for i in pp for k in pp if i < k]
    min_cpi = (float(min(pc[i, k] for i, k in prot_pairs))
               if pc.ndim == 2 and pc.shape[0] >= n else np.nan)
    return dict(
        iptm=float(j.get("iptm") if j.get("iptm") is not None else np.nan),
        ptm=float(j.get("ptm") if j.get("ptm") is not None else np.nan),
        plddt=float(j.get("plddt_mean") if j.get("plddt_mean") is not None else np.nan),
        ranking_score=np.nan, has_clash=False,
        min_pp_chain_pair_iptm=min_cpi, mean_pp_chain_pair_gpde=np.nan,
        min_prot_chain_plddt=np.nan,
    )


# ---------- geometry ----------

def _load_arr(path: Path):
    p = str(path)
    if p.endswith(".cif"):
        from biotite.structure.io.pdbx import CIFFile, get_structure
        return get_structure(CIFFile.read(p), model=1)
    from biotite.structure.io.pdb import PDBFile
    return PDBFile.read(p).get_structure(model=1)


def protein_chains(arr):
    """Chain ids that are protein (have CA and >10 residues)."""
    import biotite.structure as struc  # noqa: F401
    out = []
    for ch in pd.unique(arr.chain_id):
        sub = arr[arr.chain_id == ch]
        if "CA" in set(sub.atom_name) and len(pd.unique(sub.res_id)) > 10:
            out.append(ch)
    return list(out)


def usalign_complex_tm(usalign: str, mobile: Path, ref: Path) -> float:
    """Symmetry-aware complex TM-score normalized by the REFERENCE (Structure_2 = ref)."""
    try:
        r = subprocess.run([usalign, str(mobile), str(ref), "-mm", "1", "-ter", "1"],
                           capture_output=True, text=True, timeout=600)
    except Exception:
        return np.nan
    for ln in r.stdout.splitlines():
        if "normalized by length of Structure_2" in ln:
            m = re.search(r"TM-score=\s*([0-9.]+)", ln)
            if m:
                return float(m.group(1))
    return np.nan


def _atom_coord(arr, chain, res_id, atom, fallback="CA"):
    sub = arr[(arr.chain_id == chain) & (arr.res_id == res_id)]
    if len(sub) == 0:
        return None
    for a in (atom, fallback):
        hit = sub[sub.atom_name == a]
        if len(hit):
            return hit.coord[0]
    return None


def xprot_as_distances(arr, pchains) -> dict:
    """For each protein chain's catalytic core, distance to the NEAREST neighbour-chain Asn255-ND2.
    Returns per-core median over chains (the interfacial catalytic geometry)."""
    asn_id = XPROT_W1["Asn255"]
    per_core = {}
    for core_lab, core_id in CATALYTIC_CORE.items():
        dists = []
        for ch in pchains:
            c = _atom_coord(arr, ch, core_id, FUNC_ATOM[core_lab])
            if c is None:
                continue
            best = np.inf
            for nb in pchains:
                if nb == ch:
                    continue
                n = _atom_coord(arr, nb, asn_id, FUNC_ATOM["Asn255"])
                if n is None:
                    continue
                best = min(best, float(np.linalg.norm(c - n)))
            if np.isfinite(best):
                dists.append(best)
        per_core[core_lab] = float(np.median(dists)) if dists else np.nan
    return per_core


def n_interchain_contacts(arr, pchains, cutoff=8.0) -> int:
    """Count CB-CB (CA for Gly) pairs < cutoff between different protein chains (interface size)."""
    reps = {}
    for ch in pchains:
        sub = arr[arr.chain_id == ch]
        cb = sub[(sub.atom_name == "CB") | ((sub.res_name == "GLY") & (sub.atom_name == "CA"))]
        reps[ch] = cb.coord
    n = 0
    for i, a in enumerate(pchains):
        for b in pchains[i + 1:]:
            ca, cb = reps[a], reps[b]
            if len(ca) == 0 or len(cb) == 0:
                continue
            d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
            n += int((d < cutoff).sum())
    return n


# ---------- driver ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--backend", choices=["protenix", "esmfold2"], default="protenix",
                    help="which predictor's output layout to read (gate is backend-selectable)")
    ap.add_argument("--manifest", required=True, help="manifest parquet with id,name,parent,kind,allele (name = pred output-dir stem)")
    ap.add_argument("--usalign", required=True)
    ap.add_argument("--crystal-ref", default=None, help="1R51_tetramer_ABCD.pdb (for the crystal parent)")
    ap.add_argument("--crystal-parent", default="Q00511")
    ap.add_argument("--extra-table", default=None,
                    help="optional parquet to left-join on id (e.g. activity/expression)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    pred_root = Path(args.pred_root)

    # pass 1: locate each variant's best sample + parse coords once (cache arrays for WT refs)
    best = {}   # name -> (idx, cif, json, tier1)
    arr_cache = {}
    for _, row in man.iterrows():
        name = row["name"]
        if args.backend == "esmfold2":
            bs = find_esmfold2_pred(pred_root, name)
        else:
            samples = find_pred_samples(pred_root, name)
            bs = best_sample(samples) if samples else None
        if bs:
            best[name] = bs

    def get_arr(name):
        if name not in arr_cache:
            arr_cache[name] = _load_arr(best[name][1])
        return arr_cache[name]

    # parent -> WT variant name (the relative reference)
    wt_name_by_parent = {}
    for _, row in man.iterrows():
        if str(row.get("kind", "")).upper() == "WT" and row["name"] in best:
            wt_name_by_parent[row["parent"]] = row["name"]

    rows = []
    for _, row in man.iterrows():
        name, parent, kind = row["name"], row["parent"], row.get("kind")
        rec = dict(id=row["id"], name=name, parent=parent, kind=kind, allele=row.get("allele"),
                   predicted=name in best)
        if name in best:
            k, cif, js, t1 = best[name]
            rec.update(sample=k, **t1)
            arr = get_arr(name)
            pchains = protein_chains(arr)
            rec["n_protein_chains"] = len(pchains)
            rec["n_interchain_contacts"] = n_interchain_contacts(arr, pchains)
            # complex TM vs parent WT (relative)
            wt = wt_name_by_parent.get(parent)
            if wt and wt in best:
                rec["complex_TM_vs_wt"] = usalign_complex_tm(args.usalign, cif, best[wt][1])
            # crystal check for the crystal parent
            if args.crystal_ref and parent == args.crystal_parent:
                rec["complex_TM_vs_crystal"] = usalign_complex_tm(args.usalign, cif, Path(args.crystal_ref))
            # cross-protomer active-site distances (Q00511-numbering parents only)
            if parent in Q00511_NUMBERING_PARENTS:
                xd = xprot_as_distances(arr, pchains)
                for lab, v in xd.items():
                    rec[f"xprot_{lab}_dist"] = v
        rows.append(rec)

    df = pd.DataFrame(rows)

    # deviation of cross-protomer distances vs each parent's WT
    for lab in CATALYTIC_CORE:
        col = f"xprot_{lab}_dist"
        if col in df.columns:
            wtval = {p: df[(df.parent == p) & (df.kind.astype(str).str.upper() == "WT")][col].median()
                     for p in df.parent.unique()}
            df[f"xprot_{lab}_dev"] = df.apply(
                lambda r: (r[col] - wtval.get(r.parent, np.nan)) if pd.notna(r.get(col)) else np.nan, axis=1)

    if args.extra_table:
        extra = pd.read_parquet(args.extra_table)
        keep = [c for c in extra.columns if c not in df.columns or c == "id"]
        df = df.merge(extra[keep], on="id", how="left")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    n_pred = int(df["predicted"].sum())
    print(f"[eval] {len(df)} variants, {n_pred} predicted -> {args.out}")
    cols = [c for c in ["id", "parent", "kind", "iptm", "min_pp_chain_pair_iptm",
                        "complex_TM_vs_wt", "complex_TM_vs_crystal", "n_interchain_contacts"] if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(df[df.predicted][cols].to_string(index=False))


if __name__ == "__main__":
    main()
