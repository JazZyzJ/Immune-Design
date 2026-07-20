#!/usr/bin/env python
"""Post-prediction tetramer-gate metrics for uricase structure predictions.

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
  Assembly (all parents): six chain-pair rows are split into the two repeated D2-interface
                  classes and the diagonal non-interface. Coordinate-only metrics include
                  buried surface area (BSA), residue contacts, and salt bridges.
  Rosetta (optional): InterfaceAnalyzer scores each of the four biological-interface dimers,
                  adding dSASA, binding dG and dG/dSASA, packstat, interface H-bonds,
                  H-bond energy fraction, shape complementarity, and buried unsatisfied
                  H-bonds. Rosetta runs after prediction and never inside the GPU predictor.

Reference model: each design is scored vs its OWN parent's WT-predicted tetramer (relative), so
the gate cancels systematic model bias. Q00511 additionally gets the absolute crystal check.

The gate is NOT hard-coded here: thresholds are calibrated on same-backend WT predictions and
experimental outcomes. The main output is one row per variant; a companion long table preserves
every chain pair so downstream selection never depends only on lossy aggregate values.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.evaluation.tetramer_interfaces import (
    interface_pair_metrics,
    run_rosetta_interface_analyzer,
    summarize_interface_pairs,
)

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


WT_NORMALIZATION_FIELDS = {
    "interface_1_bsa_total_min_a2": "ratio",
    "interface_2_bsa_total_min_a2": "ratio",
    "interface_1_n_residue_contacts_min_8a": "ratio",
    "interface_2_n_residue_contacts_min_8a": "ratio",
    "interface_1_n_salt_bridges_min_4a": "delta",
    "interface_2_n_salt_bridges_min_4a": "delta",
    "interface_1_rosetta_dsasa_int_min_a2": "ratio",
    "interface_2_rosetta_dsasa_int_min_a2": "ratio",
    "interface_1_rosetta_dg_separated_max_reu": "delta",
    "interface_2_rosetta_dg_separated_max_reu": "delta",
    "interface_1_rosetta_dg_per_sasa_x100_max": "delta",
    "interface_2_rosetta_dg_per_sasa_x100_max": "delta",
    "interface_1_rosetta_dg_per_sasa_max_reu_per_a2": "delta",
    "interface_2_rosetta_dg_per_sasa_max_reu_per_a2": "delta",
    "interface_1_rosetta_packstat_min": "delta",
    "interface_2_rosetta_packstat_min": "delta",
    "interface_1_rosetta_hbonds_int_min": "delta",
    "interface_2_rosetta_hbonds_int_min": "delta",
    "interface_1_rosetta_hbond_energy_fraction_min": "delta",
    "interface_2_rosetta_hbond_energy_fraction_min": "delta",
    "interface_1_rosetta_delta_unsat_hbonds_max": "delta",
    "interface_2_rosetta_delta_unsat_hbonds_max": "delta",
    "interface_1_rosetta_shape_complementarity_min": "delta",
    "interface_2_rosetta_shape_complementarity_min": "delta",
}


def add_interface_wt_normalization(df: pd.DataFrame, *, require_wt: bool) -> pd.DataFrame:
    """Attach real same-backend WT baselines; never substitute a placeholder baseline."""
    out = df.copy()
    out["interface_wt_baseline_available"] = False
    missing_parents = []
    for parent, group in out.groupby("parent", dropna=False):
        group_index = group.index
        metrics_ok = (group["interface_metrics_ok"] if "interface_metrics_ok" in group
                      else pd.Series(False, index=group.index))
        wt = group[
            group["kind"].astype(str).str.upper().eq("WT")
            & metrics_ok.fillna(False)
        ]
        if wt.empty:
            predicted = (group["predicted"] if "predicted" in group
                         else pd.Series(False, index=group.index))
            if predicted.fillna(False).any():
                missing_parents.append(str(parent))
            continue
        out.loc[group_index, "interface_wt_baseline_available"] = True
        for metric, operation in WT_NORMALIZATION_FIELDS.items():
            if metric not in out.columns:
                continue
            baseline = pd.to_numeric(wt[metric], errors="coerce").median()
            baseline_col = f"wt_{metric}"
            out.loc[group_index, baseline_col] = baseline
            if not np.isfinite(baseline):
                continue
            if operation == "ratio":
                result_col = f"{metric}_ratio_to_wt"
                if baseline != 0:
                    out.loc[group_index, result_col] = out.loc[group_index, metric] / baseline
            else:
                result_col = f"{metric}_delta_vs_wt"
                out.loc[group_index, result_col] = out.loc[group_index, metric] - baseline
    if require_wt and missing_parents:
        raise ValueError(
            "real predicted WT interface baselines are missing for parents: "
            + ", ".join(sorted(set(missing_parents)))
        )
    return out


# ---------- driver ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--backend", choices=["protenix", "esmfold2"], default="protenix",
                    help="which predictor's output layout to read (gate is backend-selectable)")
    ap.add_argument(
        "--manifest", required=True,
        help="manifest parquet with id,name,parent,kind,allele (name = pred output-dir stem)",
    )
    ap.add_argument("--usalign", required=True)
    ap.add_argument(
        "--crystal-ref", default=None,
        help="1R51_tetramer_ABCD.pdb (for the crystal parent)",
    )
    ap.add_argument("--crystal-parent", default="Q00511")
    ap.add_argument("--extra-table", default=None,
                    help="optional parquet to left-join on id (e.g. activity/expression)")
    ap.add_argument(
        "--interface-pairs-out", default=None,
        help="chain-pair long-table parquet (default: <out stem>_interface_pairs.parquet)",
    )
    ap.add_argument("--sasa-probe-radius", type=float, default=1.4)
    ap.add_argument("--sasa-point-number", type=int, default=1000)
    ap.add_argument("--contact-cutoff", type=float, default=8.0)
    ap.add_argument("--salt-bridge-cutoff", type=float, default=4.0)
    ap.add_argument(
        "--rosetta-interface-analyzer",
        default=os.environ.get("ROSETTA_INTERFACE_ANALYZER"),
        help="optional InterfaceAnalyzer executable; enables CPU Rosetta interface metrics",
    )
    ap.add_argument("--rosetta-work-dir", default=None,
                    help="persistent Rosetta dimer/score/log directory")
    ap.add_argument("--rosetta-timeout-seconds", type=int, default=21600)
    ap.add_argument(
        "--require-interface-wt",
        action="store_true",
        help="fail if a predicted parent lacks a real predicted WT row in the manifest",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.sasa_probe_radius <= 0:
        ap.error("--sasa-probe-radius must be positive")
    if args.sasa_point_number <= 0:
        ap.error("--sasa-point-number must be positive")
    if args.contact_cutoff <= 0 or args.salt_bridge_cutoff <= 0:
        ap.error("interface distance cutoffs must be positive")
    if args.rosetta_timeout_seconds <= 0:
        ap.error("--rosetta-timeout-seconds must be positive")

    man = pd.read_parquet(args.manifest)
    required_manifest = {"id", "name", "parent", "kind"}
    missing_manifest = sorted(required_manifest - set(man.columns))
    if missing_manifest:
        raise ValueError(f"manifest is missing required columns: {missing_manifest}")
    pred_root = Path(args.pred_root)
    out_path = Path(args.out)
    pairs_out_path = (Path(args.interface_pairs_out) if args.interface_pairs_out else
                      out_path.with_name(f"{out_path.stem}_interface_pairs.parquet"))
    if pairs_out_path.resolve() == out_path.resolve():
        ap.error("--interface-pairs-out must differ from --out")

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
    pair_frames = []
    for manifest_row_index, (_, row) in enumerate(man.iterrows()):
        name, parent, kind = row["name"], row["parent"], row.get("kind")
        rec = dict(
            manifest_row_index=manifest_row_index,
            id=row["id"],
            name=name,
            parent=parent,
            kind=kind,
            allele=row.get("allele"),
            predicted=name in best,
            interface_metrics_ok=False,
            interface_metrics_error=(None if name in best else "prediction_missing"),
            rosetta_interface_requested=bool(args.rosetta_interface_analyzer),
            rosetta_interface_metrics_ok=False,
        )
        if name in best:
            k, cif, js, t1 = best[name]
            rec.update(sample=k, **t1)
            arr = get_arr(name)
            pchains = protein_chains(arr)
            rec["n_protein_chains"] = len(pchains)
            if len(pchains) == 4:
                pairs = interface_pair_metrics(
                    arr,
                    pchains,
                    sasa_probe_radius=args.sasa_probe_radius,
                    sasa_point_number=args.sasa_point_number,
                    contact_cutoff=args.contact_cutoff,
                    salt_bridge_cutoff=args.salt_bridge_cutoff,
                )
                pairs.insert(0, "prediction_path", str(cif))
                pairs.insert(0, "sample", k)
                pairs.insert(0, "backend", args.backend)
                pairs.insert(0, "allele", row.get("allele"))
                pairs.insert(0, "kind", kind)
                pairs.insert(0, "parent", parent)
                pairs.insert(0, "name", name)
                pairs.insert(0, "id", row["id"])
                pairs.insert(0, "manifest_row_index", manifest_row_index)
                pairs["bsa_method"] = "Shrake-Rupley/ProtOr"
                pairs["sasa_probe_radius_a"] = args.sasa_probe_radius
                pairs["sasa_point_number"] = args.sasa_point_number
                pairs["contact_cutoff_a"] = args.contact_cutoff
                pairs["salt_bridge_cutoff_a"] = args.salt_bridge_cutoff
                pair_frames.append(pairs)
                rec["interface_metrics_ok"] = True
                rec["interface_metrics_error"] = None
                rec["n_interchain_contacts"] = int(pairs["n_residue_contacts_8a"].sum())
            else:
                rec["interface_metrics_error"] = (
                    f"expected_4_protein_chains_got_{len(pchains)}"
                )
            # complex TM vs parent WT (relative)
            wt = wt_name_by_parent.get(parent)
            if wt and wt in best:
                rec["complex_TM_vs_wt"] = usalign_complex_tm(args.usalign, cif, best[wt][1])
            # crystal check for the crystal parent
            if args.crystal_ref and parent == args.crystal_parent:
                rec["complex_TM_vs_crystal"] = usalign_complex_tm(
                    args.usalign, cif, Path(args.crystal_ref)
                )
            # cross-protomer active-site distances (Q00511-numbering parents only)
            if parent in Q00511_NUMBERING_PARENTS:
                xd = xprot_as_distances(arr, pchains)
                for lab, v in xd.items():
                    rec[f"xprot_{lab}_dist"] = v
        rows.append(rec)

    df = pd.DataFrame(rows)
    pair_df = (pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(
        columns=[
            "manifest_row_index", "id", "name", "parent", "kind", "allele", "backend",
            "sample", "prediction_path", "chain_1", "chain_2", "chain_pair",
            "bsa_total_a2", "bsa_per_partner_a2", "n_residue_contacts_8a",
            "n_salt_bridges_4a", "interface_class", "interface_class_rank",
            "interface_copy", "d2_matching", "d2_matching_mean_bsa_total_a2",
            "is_biological_interface",
        ]
    ))

    if args.rosetta_interface_analyzer:
        if pair_df.empty:
            raise ValueError(
                "Rosetta interface scoring requested, but no valid tetramer pairs exist"
            )
        rosetta_work_dir = (
            Path(args.rosetta_work_dir) if args.rosetta_work_dir else
            out_path.with_name(f"{out_path.stem}_rosetta")
        )
        rosetta_work_dir.mkdir(parents=True, exist_ok=True)
        pair_df = run_rosetta_interface_analyzer(
            pair_df,
            get_arr,
            executable=args.rosetta_interface_analyzer,
            work_dir=rosetta_work_dir,
            timeout_seconds=args.rosetta_timeout_seconds,
        )
        scored_manifest_rows = set(
            pair_df.loc[
                pair_df["is_biological_interface"]
                & pair_df["rosetta_dsasa_int_a2"].notna(),
                "manifest_row_index",
            ]
        )
        df["rosetta_interface_metrics_ok"] = df["manifest_row_index"].isin(
            scored_manifest_rows
        )

    if not pair_df.empty:
        summary_rows = []
        for manifest_row_index, group in pair_df.groupby("manifest_row_index", sort=False):
            summary_rows.append({
                "manifest_row_index": manifest_row_index,
                **summarize_interface_pairs(group),
            })
        df = df.merge(pd.DataFrame(summary_rows), on="manifest_row_index", how="left",
                      validate="one_to_one")
    df = add_interface_wt_normalization(df, require_wt=args.require_interface_wt)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_out_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["manifest_row_index"]).to_parquet(out_path, index=False)
    pair_df.to_parquet(pairs_out_path, index=False)
    n_pred = int(df["predicted"].sum())
    print(f"[eval] {len(df)} variants, {n_pred} predicted -> {out_path}")
    print(f"[eval] {len(pair_df)} chain-pair rows -> {pairs_out_path}")
    cols = [c for c in ["id", "parent", "kind", "iptm", "min_pp_chain_pair_iptm",
                        "complex_TM_vs_wt", "complex_TM_vs_crystal", "n_interchain_contacts",
                        "interface_1_bsa_total_min_a2", "interface_2_bsa_total_min_a2",
                        "interface_1_rosetta_packstat_min",
                        "interface_2_rosetta_packstat_min"] if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(df[df.predicted][cols].to_string(index=False))


if __name__ == "__main__":
    main()
