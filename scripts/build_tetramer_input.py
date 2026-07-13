#!/usr/bin/env python
"""Build Protenix homo-oligomer (tetramer) input JSONs for the tetramer refold gate.

Each input row (a WT or a designed uricase sequence) becomes ONE Protenix job JSON with a
single ``proteinChain`` at ``--count`` copies (default 4 = homotetramer) plus, unless
``--apo``, a ``--count``-copy ligand entity (holo). One JSON per variant so predictions can
run as a SLURM array (one array task per JSON). See ``PLAN_TETRAMER_GATE.md`` §3-§5.

MSA is added out-of-band on the LOGIN node afterwards:
  ``protenix-run msa --input <json> --out_dir <msa_out> --msa_server_mode protenix``
which writes ``<json stem>-update-msa.json`` (the file fed to ``protenix pred --use_msa true``
on a compute node). This builder only emits the pre-MSA JSONs + a manifest.

Ligand default is uric acid (URC) as a raw SMILES; switchable to a CCD code (e.g. ``AZA`` for
the 1R51 crystal-pose overlay) or ``--apo`` (no ligand). Protenix distinguishes CCD from SMILES
by a ``CCD_`` prefix, so a CCD code is emitted as ``CCD_<code>`` and a SMILES verbatim.

Outputs (under ``--out-dir``):
  * ``<name_prefix>_<id>.json`` per variant (Protenix job list with one job)
  * ``manifest.parquet`` — id, parent, kind, allele, name, json path, seq form, length, ligand
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# uric acid (natural substrate), Kaiyi's SMILES; the default active-site marker ligand.
URC_SMILES = "O=C1NC(=O)C2=C(N1)NC(=O)N2"


def clean_seq(seq: str) -> str:
    """Uppercase, strip anything but letters (defends against whitespace/gaps/'*')."""
    return re.sub(r"[^A-Za-z]", "", str(seq)).upper()


def maybe_strip_met(seq: str, do_strip: bool) -> str:
    return seq[1:] if (do_strip and seq.startswith("M")) else seq


def ligand_field(ligand_smiles: str | None, ligand_ccd: str | None) -> str | None:
    """Return the Protenix ``ligand`` string (SMILES verbatim, CCD as ``CCD_<code>``)."""
    if ligand_ccd:
        return f"CCD_{ligand_ccd.upper().removeprefix('CCD_')}"
    if ligand_smiles:
        return ligand_smiles
    return None


def build_job(name: str, sequence: str, count: int, ligand: str | None) -> dict:
    seqs = [{"proteinChain": {"sequence": sequence, "count": count}}]
    if ligand is not None:
        seqs.append({"ligand": {"ligand": ligand, "count": count}})
    return {"name": name, "sequences": seqs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="parquet/csv with the sequences")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--seq-col", default="seq_raw", help="raw sequence column (Met handled here)")
    ap.add_argument("--parent-col", default="parent")
    ap.add_argument("--kind-col", default="kind")
    ap.add_argument("--allele-col", default="allele")
    ap.add_argument("--count", type=int, default=4, help="homo-oligomer copies (4 = tetramer)")
    ap.add_argument("--met-strip", action="store_true",
                    help="drop a leading Met so numbering matches the Met-excluded crystal/manifest")
    lig = ap.add_mutually_exclusive_group()
    lig.add_argument("--ligand-smiles", default=URC_SMILES, help="ligand SMILES (default: uric acid)")
    lig.add_argument("--ligand-ccd", default=None, help="ligand CCD code (e.g. AZA); overrides SMILES")
    ap.add_argument("--apo", action="store_true", help="no ligand (apo prediction)")
    ap.add_argument("--max-len", type=int, default=None,
                    help="skip variants whose (pre-strip) length exceeds this (drop outliers)")
    ap.add_argument("--name-prefix", default="tetra")
    args = ap.parse_args()

    df = pd.read_parquet(args.table) if args.table.endswith(".parquet") else pd.read_csv(args.table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ligand = None if args.apo else ligand_field(args.ligand_smiles, args.ligand_ccd)

    rows, skipped = [], []
    for _, r in df.iterrows():
        vid = str(r[args.id_col])
        raw = clean_seq(r[args.seq_col])
        if not raw:
            skipped.append((vid, "empty_seq")); continue
        if args.max_len is not None and len(raw) > args.max_len:
            skipped.append((vid, f"len_{len(raw)}_gt_{args.max_len}")); continue
        seq = maybe_strip_met(raw, args.met_strip)
        name = f"{args.name_prefix}_{vid}"
        job = build_job(name, seq, args.count, ligand)
        jpath = out_dir / f"{name}.json"
        jpath.write_text(json.dumps([job], indent=2))
        rows.append(dict(
            id=vid, name=name, json_path=str(jpath),
            parent=r.get(args.parent_col), kind=r.get(args.kind_col),
            allele=r.get(args.allele_col),
            seq_len=len(seq), met_stripped=bool(args.met_strip and raw.startswith("M")),
            ligand=(ligand if ligand is not None else "apo"), count=args.count,
        ))

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(out_dir / "manifest.parquet", index=False)
    print(f"[build] wrote {len(rows)} JSONs -> {out_dir}")
    print(f"[build] manifest.parquet ({len(manifest)} rows); ligand={ligand!r} count={args.count} "
          f"met_strip={args.met_strip} max_len={args.max_len}")
    if skipped:
        print(f"[build] skipped {len(skipped)}: " + ", ".join(f"{i}({why})" for i, why in skipped))


if __name__ == "__main__":
    main()
