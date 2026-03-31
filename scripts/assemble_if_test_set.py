#!/usr/bin/env python
"""CLI: Assemble final test_proteins.parquet from all three tiers.

Extracts WT sequences from PDB files, generates WT artifacts
(epitope head + NetMHCIIpan), then assembles into the frozen parquet.

Usage:
    python scripts/assemble_if_test_set.py \
        --tier1-json outputs/if/test_set/tier1_candidates.json \
        --tier2-parquet outputs/if/test_set/tier2_prescreened.parquet \
        --tier3-json outputs/if/test_set/tier3_candidates.json \
        --pdb-dir outputs/if/test_set/pdbs/ \
        --epitope-ckpt run/epitope_head/best.pt \
        --netmhciipan-bin /path/to/netMHCIIpan \
        --output-dir outputs/if/test_set/ \
        [--allele HLA-DRB1*07:01] \
        [--device cuda] \
        [--variant-id LC1]
"""

import argparse
import json
import os
import sys

import pandas as pd

from inverse_folding.evaluation.assembly import (
    assemble_test_set,
    validate_test_set,
    compute_assembly_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble final IF test set from all three tiers."
    )
    parser.add_argument("--tier1-json", required=True)
    parser.add_argument("--tier2-parquet", required=True)
    parser.add_argument("--tier3-json", required=True)
    parser.add_argument("--pdb-dir", required=True,
                        help="Directory containing PDB files for WT extraction.")
    parser.add_argument("--epitope-ckpt", required=True)
    parser.add_argument("--netmhciipan-bin", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allele", default="HLA-DRB1*07:01")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant-id", default="LC1")
    parser.add_argument("--config-dir", default=None)
    return parser.parse_args()


def _extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extract single-chain amino acid sequence from a PDB file."""
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    # Take first model, first chain
    model = next(structure.get_models())
    chain = next(model.get_chains())
    residues = [r for r in chain.get_residues()
                if r.get_id()[0] == " "]  # standard residues only
    return "".join(seq1(r.get_resname()) for r in residues)


def _load_tier1(path: str, pdb_dir: str) -> list:
    """Load Tier 1 candidates, extracting WT sequences from PDBs."""
    with open(path) as f:
        candidates = json.load(f)
    entries = []
    for c in candidates:
        if not c.get("accepted", True):
            continue
        pid = c["protein_id"]
        pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")
        # Extract WT sequence from PDB if not already present
        seq = c.get("sequence", "")
        if not seq and os.path.isfile(pdb_path):
            seq = _extract_sequence_from_pdb(pdb_path)
        if not seq:
            print(f"  ERROR: No sequence for {pid} (PDB: {pdb_path})")
            sys.exit(1)
        entries.append({
            "protein_id": pid,
            "tier": 1,
            "sequence": seq,
            "sequence_length": len(seq),
            "pdb_path": f"pdbs/{pid}.pdb",
            "resolution": c.get("resolution"),
            "cath_overlap_flag": False,
            "cath_overlap_id": None,
            "head_train_overlap_flag": c.get("head_train_overlap_flag", False),
            # Placeholder — filled by artifact generation below
            "netmhciipan_n_strong": 0,
            "netmhciipan_mean_best_rank": 0.0,
            "head_global_risk": 0.0,
            "head_n_hotspot": 0,
            "cath_topology": None,
            "experimental_epitopes_json": json.dumps(c.get("experimental_epitopes", [])),
            "literature_evidence": None,
            "selection_reason": c.get("selection_reason", "tier1"),
        })
    return entries


def _load_tier2(path: str) -> list:
    """Load Tier 2 pre-screened parquet (already has scores from prescreen)."""
    df = pd.read_parquet(path)
    entries = []
    for _, row in df.iterrows():
        seq = row.get("sequence", "")
        if not seq:
            print(f"  ERROR: Empty sequence for Tier 2 protein {row['protein_id']}")
            sys.exit(1)
        entries.append({
            "protein_id": row["protein_id"],
            "tier": 2,
            "sequence": seq,
            "sequence_length": int(row.get("sequence_length", len(seq))),
            "pdb_path": f"pdbs/{row['protein_id']}.pdb",
            "resolution": row.get("resolution"),
            "cath_overlap_flag": bool(row.get("cath_overlap_flag", False)),
            "cath_overlap_id": row.get("cath_overlap_id"),
            "head_train_overlap_flag": bool(row.get("head_train_overlap_flag", False)),
            "netmhciipan_n_strong": int(row.get("netmhciipan_n_strong", 0)),
            "netmhciipan_mean_best_rank": float(row.get("netmhciipan_mean_best_rank", 0.0)),
            "head_global_risk": float(row.get("head_global_risk", 0.0)),
            "head_n_hotspot": int(row.get("head_n_hotspot", 0)),
            "cath_topology": row.get("cath_topology"),
            "experimental_epitopes_json": None,
            "literature_evidence": None,
            "selection_reason": row.get("selection_reason", "pre-screened"),
        })
    return entries


def _load_tier3(path: str, pdb_dir: str) -> list:
    """Load Tier 3 candidates, extracting WT sequences from PDBs."""
    with open(path) as f:
        candidates = json.load(f)
    entries = []
    for c in candidates:
        if not c.get("accepted", True):
            continue
        pid = c["protein_id"]
        pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")
        seq = c.get("sequence", "")
        if not seq and os.path.isfile(pdb_path):
            seq = _extract_sequence_from_pdb(pdb_path)
        if not seq:
            print(f"  ERROR: No sequence for {pid} (PDB: {pdb_path})")
            sys.exit(1)
        entries.append({
            "protein_id": pid,
            "tier": 3,
            "sequence": seq,
            "sequence_length": len(seq),
            "pdb_path": f"pdbs/{pid}.pdb",
            "resolution": c.get("resolution"),
            "cath_overlap_flag": False,
            "cath_overlap_id": None,
            "head_train_overlap_flag": c.get("head_train_overlap_flag", False),
            "netmhciipan_n_strong": 0,
            "netmhciipan_mean_best_rank": 0.0,
            "head_global_risk": 0.0,
            "head_n_hotspot": 0,
            "cath_topology": None,
            "experimental_epitopes_json": None,
            "literature_evidence": c.get("literature_evidence", ""),
            "selection_reason": c.get("selection_reason", "therapeutic"),
        })
    return entries


def _generate_wt_artifacts(entries: list, args) -> list:
    """Generate WT epitope head + NetMHCIIpan artifacts for entries missing them.

    Mutates entries in place: fills netmhciipan_*, head_global_risk, head_n_hotspot.
    Skips entries that already have non-zero scores (Tier 2 from prescreen).
    """
    # Identify entries needing scoring
    needs_scoring = [
        e for e in entries
        if e["netmhciipan_n_strong"] == 0 and e["head_global_risk"] == 0.0
    ]
    if not needs_scoring:
        print("  All entries already have WT artifacts.")
        return entries

    print(f"  Generating WT artifacts for {len(needs_scoring)} proteins...")

    # Load NMP runner
    from epitope_head.data.netmhciipan_runner import build_runner
    from inverse_folding.evaluation.immunogenicity import (
        aggregate_nmp_scores,
        score_protein_all_lengths,
    )

    nmp_runner = build_runner(
        backend="standalone", binary_path=args.netmhciipan_bin,
    )

    # Load epitope head predictor
    from scripts.run_if_guidance_sweep import load_epitope_predictor

    config_dir = args.config_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "epitope_head", "configs",
    )
    predictor = load_epitope_predictor(
        config_dir=config_dir,
        checkpoint_path=args.epitope_ckpt,
        variant_id=args.variant_id,
        device=args.device,
    )

    import torch

    for e in needs_scoring:
        pid, seq = e["protein_id"], e["sequence"]

        # NetMHCIIpan
        scores_df = score_protein_all_lengths(
            nmp_runner, pid, seq, args.allele,
        )
        agg = aggregate_nmp_scores(scores_df)
        e["netmhciipan_n_strong"] = agg["n_strong_binders"]
        e["netmhciipan_mean_best_rank"] = agg["mean_best_rank"]

        # Epitope head
        pred = predictor.predict_protein(seq, allele_idx=0)
        e["head_global_risk"] = float(pred["global_risk"])
        h = pred["residue_hotspot"]
        if isinstance(h, torch.Tensor) and h.numel() > 0:
            median_h = h.median().item()
            std_h = h.std().item()
            e["head_n_hotspot"] = int((h > median_h + std_h).sum().item())
        else:
            e["head_n_hotspot"] = 0

        print(f"    {pid}: NMP_strong={e['netmhciipan_n_strong']}, "
              f"risk={e['head_global_risk']:.3f}")

    return entries


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load all tiers ───────────────────────────────────────────────────
    print("[1/5] Loading tier entries...")
    tier1 = _load_tier1(args.tier1_json, args.pdb_dir)
    tier2 = _load_tier2(args.tier2_parquet)
    tier3 = _load_tier3(args.tier3_json, args.pdb_dir)
    print(f"  Tier 1: {len(tier1)}, Tier 2: {len(tier2)}, Tier 3: {len(tier3)}")

    all_entries = tier1 + tier2 + tier3

    # ── Generate WT artifacts ────────────────────────────────────────────
    print("[2/5] Generating WT artifacts...")
    all_entries = _generate_wt_artifacts(all_entries, args)

    # ── Write per-protein FASTA ──────────────────────────────────────────
    print("[3/5] Writing per-protein FASTA files...")
    fasta_dir = os.path.join(args.output_dir, "fastas")
    os.makedirs(fasta_dir, exist_ok=True)
    for e in all_entries:
        fasta_path = os.path.join(fasta_dir, f"{e['protein_id']}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{e['protein_id']}\n{e['sequence']}\n")

    # ── Assemble ─────────────────────────────────────────────────────────
    print("[4/5] Assembling test set...")
    df = assemble_test_set(all_entries)

    parquet_path = os.path.join(args.output_dir, "test_proteins.parquet")
    df.to_parquet(parquet_path, index=False)

    summary = compute_assembly_summary(df)
    summary_path = os.path.join(args.output_dir, "test_proteins_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Validate ─────────────────────────────────────────────────────────
    print("[5/5] Validating assembled test set...")
    errors = validate_test_set(parquet_path)
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  {e}")
        return 1

    print(f"Assembly complete: {len(df)} proteins → {parquet_path}")
    print(f"Summary → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
