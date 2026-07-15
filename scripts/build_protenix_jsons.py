#!/usr/bin/env python
"""Convert per-design ColabFold a3m -> Protenix homo-tetramer JSON (count=4 + 4 ligands).

For a homo-oligomer, Protenix pairing MSA = query-only (identical chains self-pair);
non_pairing MSA = the full ColabFold a3m. Validated on design_0000 (pairing=1, non_pairing=3076).
"""
import argparse, json, os, sys
import pandas as pd

LIGAND_SMILES = "O=C1NC(=O)C2=C(N1)NC(=O)N2"  # uric acid (CCD URC)


def a3m_query(a3m_path):
    """Return (header_seq) of the first record = the query."""
    hdr = seq = None
    with open(a3m_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if hdr is not None:
                    break
                hdr = line.strip()
                seq = ""
            elif hdr is not None:
                seq += line.strip()
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--msa-cf", required=True, help="dir with <pred_id>.a3m from colabfold_search")
    ap.add_argument("--msa-px", required=True, help="output dir for <pred_id>/{pairing,non_pairing}.a3m")
    ap.add_argument("--inputs", required=True, help="output dir for <pred_id>-update-msa.json")
    ap.add_argument("--ready-list", required=True)
    ap.add_argument("--ligand-smiles", default=LIGAND_SMILES)
    ap.add_argument("--n-copies", type=int, default=4)
    ap.add_argument("--apo", action="store_true", help="omit the ligand entity (monomer/apo prediction)")
    ap.add_argument("--name-prefix", default="tetra_", help="prediction name prefix (e.g. 'mono_')")
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    os.makedirs(args.msa_px, exist_ok=True)
    os.makedirs(args.inputs, exist_ok=True)
    ready, missing = [], []
    for r in man.itertuples():
        pid = r.pred_id
        a3m = os.path.join(args.msa_cf, f"{pid}.a3m")
        if not os.path.isfile(a3m):
            missing.append(pid)
            continue
        qseq = a3m_query(a3m)
        # cross-check the a3m query matches the manifest sequence (upper, gaps removed)
        exp = r.sequence_pred.upper()
        if qseq.replace("-", "").upper() != exp:
            print(f"WARN {pid}: a3m query != manifest seq (len {len(qseq)} vs {len(exp)})", file=sys.stderr)
        d = os.path.join(args.msa_px, pid)
        os.makedirs(d, exist_ok=True)
        # non_pairing = full a3m verbatim
        with open(a3m) as src, open(os.path.join(d, "non_pairing.a3m"), "w") as dst:
            dst.write(src.read())
        # pairing = query only
        with open(os.path.join(d, "pairing.a3m"), "w") as f:
            f.write(f">query\n{exp}\n")
        seqs_list = [
            {"proteinChain": {
                "sequence": exp,
                "count": args.n_copies,
                "pairedMsaPath": os.path.join(d, "pairing.a3m"),
                "unpairedMsaPath": os.path.join(d, "non_pairing.a3m"),
            }},
        ]
        if not args.apo:
            seqs_list.append({"ligand": {"ligand": args.ligand_smiles, "count": args.n_copies}})
        obj = [{"name": f"{args.name_prefix}{pid}", "sequences": seqs_list}]
        jp = os.path.join(args.inputs, f"{args.name_prefix}{pid}-update-msa.json")
        with open(jp, "w") as f:
            json.dump(obj, f, indent=2)
        ready.append(jp)
    with open(args.ready_list, "w") as f:
        f.write("\n".join(ready) + "\n")
    print(f"built {len(ready)} JSONs; missing a3m for {len(missing)}: {missing[:10]}")


if __name__ == "__main__":
    main()
