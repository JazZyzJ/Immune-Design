#!/usr/bin/env python
"""Predict a homo-oligomer (tetramer) + ligand with MSA using Biohub ESMFold2.

Runs in the `esmfold2` conda env (transformers 4.57 + Biohub `esm`); NOT immune-design (which
carries fair-esm). ESMFold2 natively supports all three (source-verified, PLAN_TETRAMER_GATE.md):
  * homo-oligomer: ProteinInput(id=['A','B','C','D'], sequence=seq) -> one entity, 4 chains
  * MSA:          ProteinInput(msa=MSA.from_a3m(a3m)); the ESMFold2Model MSAEncoder consumes it
  * ligand:       LigandInput(id='L', smiles=... | ccd=['URC'])

Writes, under <out-dir>/<name>/: <name>.cif (full complex incl. ligand hetero atoms) and
<name>_confidence.json (plddt mean/per-res, ptm, iptm, pair_chains_iptm). Mirrors enough of the
Protenix output that eval_tetramer_gate.py can read it (an `--esmfold2` reader is added there).

CAVEAT (per feasibility review): ESMFold2 has no published ligand-pose benchmark — treat the
ligand placement as a hypothesis and cross-check top designs against Protenix/AF3.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _read_fasta_seq(path: str) -> str:
    s = "".join(l.strip() for l in Path(path).read_text().splitlines() if not l.startswith(">"))
    return re.sub(r"[^A-Za-z]", "", s).upper()


def _to_py(x):
    """Tensor/array -> json-able (scalars to float, small vectors to list, matrices summarized)."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu()
            return float(x) if x.ndim == 0 else x.numpy().tolist()
    except Exception:
        pass
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence")
    src.add_argument("--fasta")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-copies", type=int, default=4)
    ap.add_argument("--msa-a3m", default=None, help="ColabFold a3m for MSA conditioning (query must match --sequence)")
    lig = ap.add_mutually_exclusive_group()
    lig.add_argument("--ligand-smiles", default=None)
    lig.add_argument("--ligand-ccd", default=None, help="comma-separated CCD codes, e.g. URC")
    ap.add_argument("--ligand-copies", type=int, default=0,
                    help="number of ligand copies; 0 = match --n-copies (one per active site)")
    ap.add_argument("--save-distogram", action="store_true",
                    help="also dump the (large) distogram to the arrays npz")
    ap.add_argument("--num-loops", type=int, default=8)
    ap.add_argument("--num-sampling-steps", type=int, default=100)
    ap.add_argument("--num-diffusion-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--model-name", default="biohub/ESMFold2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seq = _read_fasta_seq(args.fasta) if args.fasta else re.sub(r"[^A-Za-z]", "", args.sequence).upper()
    assert seq, "empty sequence"

    import torch
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder, ProteinInput, StructurePredictionInput, LigandInput, MSA,
    )
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    print(f"[esmfold2] loading {args.model_name} ...", flush=True)
    model = ESMFold2Model.from_pretrained(args.model_name).to(args.device).eval()
    builder = ESMFold2InputBuilder()

    chain_ids = [f"P{i}" for i in range(args.n_copies)]  # id LIST -> homo-oligomer (shared entity, sym_id 0..N)
    msa = MSA.from_a3m(args.msa_a3m) if args.msa_a3m else None
    if args.msa_a3m and msa is None:
        print("[esmfold2] WARN MSA.from_a3m returned None", flush=True)
    entities = [ProteinInput(id=chain_ids if args.n_copies > 1 else chain_ids[0], sequence=seq, msa=msa)]

    if args.ligand_smiles or args.ligand_ccd:
        n_lig = args.ligand_copies or args.n_copies   # default: one ligand per active site
        ccd = args.ligand_ccd.split(",") if args.ligand_ccd else None
        # one SEPARATE LigandInput entity per copy: a single LigandInput places ONE ligand
        # regardless of its id, so a tetramer's 4 interfacial sites need 4 distinct entities.
        for i in range(n_lig):
            entities.append(LigandInput(id=f"L{i}", smiles=args.ligand_smiles, ccd=ccd))
    spi = StructurePredictionInput(sequences=entities)
    print(f"[esmfold2] fold: {args.n_copies}x{len(seq)}aa msa={'yes' if msa else 'no'} "
          f"ligand={args.ligand_smiles or args.ligand_ccd or 'none'} loops={args.num_loops} steps={args.num_sampling_steps}",
          flush=True)

    with torch.inference_mode():
        result = builder.fold(
            model, spi,
            num_loops=args.num_loops, num_sampling_steps=args.num_sampling_steps,
            num_diffusion_samples=args.num_diffusion_samples, seed=args.seed,
        )
    if isinstance(result, list):
        result = result[0]

    out = Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)

    # 1) full complex structure (4 protein chains + ligand hetero atoms)
    (out / f"{args.name}.cif").write_text(result.complex.to_mmcif())

    def _arr(name):
        v = getattr(result, name, None)
        if v is None:
            return None
        try:
            import torch
            return v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        except Exception:
            return None

    # 2) COMPLETE raw arrays for defining new metrics later (PAE, per-residue pLDDT, token maps)
    keys = ["plddt", "pae", "residue_index", "entity_id", "pair_chains_iptm"]
    if args.save_distogram:
        keys.append("distogram")
    arrays = {k: a for k in keys if (a := _arr(k)) is not None}
    np.savez_compressed(out / f"{args.name}_arrays.npz", **arrays)

    # 3) small scalar-confidence json (arrays live in the npz)
    plddt = _arr("plddt")
    conf = {
        "backend": "esmfold2", "model": args.model_name, "name": args.name,
        "n_copies": args.n_copies, "ligand": args.ligand_smiles or args.ligand_ccd,
        "num_loops": args.num_loops, "num_sampling_steps": args.num_sampling_steps, "seed": args.seed,
        "plddt_mean": float(plddt.mean()) if plddt is not None else None,
        "ptm": _to_py(getattr(result, "ptm", None)),
        "iptm": _to_py(getattr(result, "iptm", None)),
        "pair_chains_iptm": _to_py(getattr(result, "pair_chains_iptm", None)),
        "arrays_npz": f"{args.name}_arrays.npz",
        "arrays_saved": {k: list(np.shape(v)) for k, v in arrays.items()},
        "result_attrs": sorted(a for a in dir(result) if not a.startswith("_")),
    }
    (out / f"{args.name}_confidence.json").write_text(json.dumps(conf, indent=2))
    print(f"[esmfold2] wrote {out}/: {args.name}.cif + _confidence.json + _arrays.npz"
          f"({','.join(f'{k}{list(np.shape(v))}' for k, v in arrays.items())}) "
          f"| plddt {conf['plddt_mean']:.4f} ptm {conf['ptm']} iptm {conf['iptm']}", flush=True)


if __name__ == "__main__":
    main()
