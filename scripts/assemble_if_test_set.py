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
    parser.add_argument("--tier2-parquet", default=None)
    parser.add_argument("--tier2-candidate-merged-fasta", default=None,
                        help="Merged Tier 2 FASTA used to reconstruct entries from prescreen checkpoints.")
    parser.add_argument("--tier2-head-ckpt-parquet", default=None,
                        help="Prescreen head checkpoint parquet for Tier 2 reconstruction.")
    parser.add_argument("--tier2-nmp-ckpt-parquet", default=None,
                        help="Prescreen NMP checkpoint parquet for Tier 2 reconstruction.")
    parser.add_argument("--tier2-min-strong-windows", type=int, default=5,
                        help="Tier 2 NMP threshold used when reconstructing from checkpoints.")
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


def _read_merged_fasta(path: str) -> dict:
    sequences = {}
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id and current_seq:
        sequences[current_id] = "".join(current_seq)
    return sequences


def _safe_file_id(protein_id: str) -> str:
    """Make a filesystem-safe artifact id while preserving raw protein_id metadata."""
    return "".join(
        c if (c.isalnum() or c in ("-", "_", ".")) else "_"
        for c in protein_id
    )


def _allele_tag(allele: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", ".")) else "_" for c in allele)


# Structure files are not uniformly `<id>.pdb` under `pdb_dir`. The initial bulk download is
# flat, but structures added later land in a per-allele subdirectory -- named either with the
# short digits form (`0401`) or the full allele tag (`HLA-DRB1_04_01`) -- and a minority are
# `.cif` rather than `.pdb`. `pdb_path` is a data contract read downstream, so it has to name a
# file that actually exists instead of a guessed one.
_STRUCTURE_EXTENSIONS = (".pdb", ".cif")


def _pdb_subdir_candidates(allele_tag: str | None) -> list:
    """Per-allele subdirectory names to probe, most specific spelling first."""
    if not allele_tag:
        return []
    digits = "".join(c for c in allele_tag if c.isdigit())
    short = digits[-4:] if len(digits) >= 4 else ""
    return [s for s in (short, allele_tag) if s]


def _resolve_pdb_relpath(
    protein_id: str, pdb_dir: str | None = None, allele_tag: str | None = None
) -> str:
    """Return the `pdbs/`-relative path of the structure that is actually on disk.

    Probes the flat layout first, then each per-allele subdirectory, trying `.pdb` before
    `.cif` at each level. Falls back to the legacy `pdbs/<id>.pdb` guess when nothing resolves
    (or when no ``pdb_dir`` is supplied) so the assembler still runs before structures have been
    downloaded; the caller reports how many entries fell back.
    """
    safe = _safe_file_id(protein_id)
    if pdb_dir:
        for sub in ["", *_pdb_subdir_candidates(allele_tag)]:
            for ext in _STRUCTURE_EXTENSIONS:
                rel = f"{sub}/{safe}{ext}" if sub else f"{safe}{ext}"
                if os.path.isfile(os.path.join(pdb_dir, rel)):
                    return f"pdbs/{rel}"
    return f"pdbs/{safe}.pdb"


def _load_tier1(path: str, pdb_dir: str, allele_tag: str | None = None) -> list:
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
            "pdb_path": _resolve_pdb_relpath(pid, pdb_dir, allele_tag),
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


def _load_tier2(path: str, pdb_dir: str | None = None, allele_tag: str | None = None) -> list:
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
            "pdb_path": _resolve_pdb_relpath(row["protein_id"], pdb_dir, allele_tag),
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


def _load_tier2_from_checkpoints(
    candidate_fasta_path: str,
    head_ckpt_path: str,
    nmp_ckpt_path: str,
    min_strong_windows: int,
    pdb_dir: str | None = None,
    allele_tag: str | None = None,
) -> list:
    """Reconstruct Tier 2 entries from prescreen checkpoints without diversity sampling."""
    sequences = _read_merged_fasta(candidate_fasta_path)
    head_df = pd.read_parquet(head_ckpt_path)
    nmp_df = pd.read_parquet(nmp_ckpt_path)

    head_rows = {
        row["protein_id"]: row.to_dict()
        for _, row in head_df.iterrows()
    }
    risk_values = [float(row["global_risk"]) for row in head_rows.values()]
    risk_median = float(pd.Series(risk_values).median()) if risk_values else 0.0

    entries = []
    for _, row in nmp_df.iterrows():
        payload = row.to_dict()
        protein_id = payload["protein_id"]
        status = payload.get("nmp_status", "complete")
        if status != "complete":
            continue
        if int(payload.get("n_strong_windows", 0)) < min_strong_windows:
            continue

        head = head_rows.get(protein_id)
        if head is None:
            continue
        if float(head.get("global_risk", 0.0)) < risk_median:
            continue

        seq = sequences.get(protein_id, "")
        if not seq:
            print(f"  ERROR: Missing sequence for reconstructed Tier 2 protein {protein_id}")
            sys.exit(1)

        entries.append({
            "protein_id": protein_id,
            "tier": 2,
            "sequence": seq,
            "sequence_length": len(seq),
            "pdb_path": _resolve_pdb_relpath(protein_id, pdb_dir, allele_tag),
            "resolution": None,
            "cath_overlap_flag": False,
            "cath_overlap_id": None,
            "head_train_overlap_flag": False,
            "netmhciipan_n_strong": int(payload.get("n_strong_windows", 0)),
            "netmhciipan_mean_best_rank": float(payload.get("mean_best_rank", 0.0)),
            "head_global_risk": float(head.get("global_risk", 0.0)),
            "head_n_hotspot": int(head.get("n_hotspot_positions", 0)),
            "cath_topology": None,
            "experimental_epitopes_json": None,
            "literature_evidence": None,
            "selection_reason": "pre-screened-no-diversity",
        })

    return entries


def _load_tier3(path: str, pdb_dir: str, allele_tag: str | None = None) -> list:
    """Load Tier 3 candidates from JSON candidate list or uricase prescreen parquet."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        entries = []
        for _, row in df.iterrows():
            seq = row.get("sequence", "")
            if not seq:
                print(f"  ERROR: Empty sequence for Tier 3 protein {row['protein_id']}")
                sys.exit(1)
            entries.append({
                "protein_id": row["protein_id"],
                "tier": 3,
                "sequence": seq,
                "sequence_length": int(row.get("sequence_length", len(seq))),
                "pdb_path": _resolve_pdb_relpath(row["protein_id"], pdb_dir, allele_tag),
                "resolution": row.get("resolution"),
                "cath_overlap_flag": False,
                "cath_overlap_id": None,
                "head_train_overlap_flag": bool(row.get("head_train_overlap_flag", False)),
                "netmhciipan_n_strong": int(row.get("netmhciipan_n_strong", 0)),
                "netmhciipan_mean_best_rank": float(row.get("netmhciipan_mean_best_rank", 0.0)),
                "head_global_risk": float(row.get("head_global_risk", 0.0)),
                "head_n_hotspot": int(row.get("head_n_hotspot", 0)),
                "cath_topology": None,
                "experimental_epitopes_json": None,
                "literature_evidence": row.get("literature_evidence", ""),
                "selection_reason": row.get("selection_reason", "uricase-prescreen"),
            })
        return entries

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
            "pdb_path": _resolve_pdb_relpath(pid, pdb_dir, allele_tag),
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
    allele_tag = _allele_tag(args.allele)

    if args.tier2_parquet is None:
        required = [
            args.tier2_candidate_merged_fasta,
            args.tier2_head_ckpt_parquet,
            args.tier2_nmp_ckpt_parquet,
        ]
        if not all(required):
            print(
                "ERROR: Provide either --tier2-parquet or all of "
                "--tier2-candidate-merged-fasta, --tier2-head-ckpt-parquet, "
                "--tier2-nmp-ckpt-parquet."
            )
            return 1

    # ── Load all tiers ───────────────────────────────────────────────────
    print("[1/5] Loading tier entries...")
    tier1 = _load_tier1(args.tier1_json, args.pdb_dir, allele_tag)
    if args.tier2_parquet is not None:
        tier2 = _load_tier2(args.tier2_parquet, args.pdb_dir, allele_tag)
    else:
        tier2 = _load_tier2_from_checkpoints(
            candidate_fasta_path=args.tier2_candidate_merged_fasta,
            head_ckpt_path=args.tier2_head_ckpt_parquet,
            nmp_ckpt_path=args.tier2_nmp_ckpt_parquet,
            min_strong_windows=args.tier2_min_strong_windows,
            pdb_dir=args.pdb_dir,
            allele_tag=allele_tag,
        )
    tier3 = _load_tier3(args.tier3_json, args.pdb_dir, allele_tag)
    print(f"  Tier 1: {len(tier1)}, Tier 2: {len(tier2)}, Tier 3: {len(tier3)}")

    all_entries = tier1 + tier2 + tier3

    # `pdb_path` is a downstream data contract, so surface any entry whose structure could not
    # be located: those rows carry the legacy `pdbs/<id>.pdb` guess and will not resolve.
    # `pdb_path` is emitted as "pdbs/<rel>" where <rel> is relative to --pdb-dir.
    unresolved = [
        e["protein_id"]
        for e in all_entries
        if not os.path.isfile(
            os.path.join(args.pdb_dir, e["pdb_path"].split("/", 1)[1])
        )
    ]
    if unresolved:
        print(
            f"  WARNING: {len(unresolved)}/{len(all_entries)} entries have no structure on disk; "
            f"their pdb_path is an unverified fallback: {', '.join(unresolved[:10])}"
            + (" ..." if len(unresolved) > 10 else "")
        )

    # ── Generate WT artifacts ────────────────────────────────────────────
    print("[2/5] Generating WT artifacts...")
    all_entries = _generate_wt_artifacts(all_entries, args)

    # ── Write per-protein FASTA ──────────────────────────────────────────
    print("[3/5] Writing per-protein FASTA files...")
    fasta_dir = os.path.join(args.output_dir, "fastas")
    os.makedirs(fasta_dir, exist_ok=True)
    for e in all_entries:
        fasta_path = os.path.join(
            fasta_dir, f"{_safe_file_id(e['protein_id'])}.fasta"
        )
        with open(fasta_path, "w") as f:
            f.write(f">{e['protein_id']}\n{e['sequence']}\n")

    # ── Assemble ─────────────────────────────────────────────────────────
    print("[4/5] Assembling test set...")
    df = assemble_test_set(all_entries)

    parquet_path = os.path.join(
        args.output_dir, f"test_proteins_{allele_tag}.parquet"
    )
    df.to_parquet(parquet_path, index=False)

    summary = compute_assembly_summary(df)
    summary_path = os.path.join(
        args.output_dir, f"test_proteins_summary_{allele_tag}.json"
    )
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
