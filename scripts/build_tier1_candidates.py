#!/usr/bin/env python
"""Build a Tier 1 candidate JSON for one allele.

Reverse-engineered from the shipped 0701 set; span extraction is exact (15/15
proteins, 261/261 spans reproduced bit-for-bit). PDB chain selection uses the
PDBe SIFTS REST API (`/mappings/best_structures/{uniprot}`) — the first chain
per UniProt that passes (X-ray, ≤ max-resolution, chain length ∈ [min, max])
becomes that UniProt's candidate.

Usage:
    python scripts/build_tier1_candidates.py \\
        --allele         "HLA-DRB1*04:01" \\
        --manifest-dir   outputs/manifests/drb0401 \\
        --mhc-if-v2      data/mhc_if_v2.tsv \\
        --uniprot-fasta  data/all_sequences.fasta \\
        --output         outputs/if/test_set/tier1_candidates_DRB1_04_01.json \\
        --n-select       15

Validation mode (compare to existing 0701 JSON):

    python scripts/build_tier1_candidates.py \\
        --allele         "HLA-DRB1*07:01" \\
        --manifest-dir   outputs/manifests \\
        --mhc-if-v2      data/mhc_if_v2.tsv \\
        --uniprot-fasta  data/all_sequences.fasta \\
        --output         /tmp/tier1_0701_dryrun.json \\
        --validate-against outputs/if/test_set/tier1_candidates.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Project root → sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inverse_folding.evaluation.tier1_builder import (  # noqa: E402
    FilterParams,
    build_tier1_candidates,
    load_iedb_spans,
    load_test_uniprots,
    load_uniprot_seqs,
)
from inverse_folding.evaluation.tier_validators import validate_tier1_candidate  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("build_tier1_candidates")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--allele", required=True,
                   help="Exact IEDB allele string, e.g. 'HLA-DRB1*04:01'.")
    p.add_argument("--manifest-dir", required=True, type=Path,
                   help="Per-allele manifest dir, must contain splits/test_ids.txt.")
    p.add_argument("--mhc-if-v2", required=True, type=Path,
                   help="Path to data/mhc_if_v2.tsv (combined IEDB + Atlas spans).")
    p.add_argument("--uniprot-fasta", required=True, type=Path,
                   help="UniProt/NCBI FASTA, e.g. data/all_sequences.fasta.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON path for the selected candidates.")
    p.add_argument("--n-select", type=int, default=15,
                   help="How many candidates to keep (default: 15).")
    p.add_argument("--min-spans", type=int, default=2,
                   help="Minimum EL spans per candidate chain (default: 2).")
    p.add_argument("--min-len", type=int, default=100,
                   help="Minimum chain length AA (default: 100).")
    p.add_argument("--max-len", type=int, default=500,
                   help="Maximum chain length AA (default: 500).")
    p.add_argument("--max-resolution", type=float, default=2.5,
                   help="Maximum X-ray resolution Å (default: 2.5).")
    p.add_argument("--min-coverage", type=float, default=0.10,
                   help="Minimum epitope coverage density (default: 0.10 = 10%%).")
    p.add_argument("--max-coverage", type=float, default=0.50,
                   help="Maximum epitope coverage density (default: 0.50 = 50%%).")
    p.add_argument("--rank-by", default="epitope_coverage",
                   choices=["epitope_coverage", "n_epitope_spans", "n_covered_residues"],
                   help="Ranking metric used to pick top --n-select (default: epitope_coverage).")
    p.add_argument("--rank-ascending", action="store_true",
                   help="Rank ascending instead of descending.")
    p.add_argument("--sifts-cache-dir", type=Path, default=Path("outputs/.cache/sifts"),
                   help="Directory to cache PDBe SIFTS responses (default: outputs/.cache/sifts).")
    p.add_argument("--api-delay-s", type=float, default=0.10,
                   help="Sleep after each SIFTS API call in seconds (default: 0.10).")
    p.add_argument("--epitope-head-split", default="test",
                   help="Value to write into entry['epitope_head_split'] (default: 'test').")
    p.add_argument("--validate-against", type=Path, default=None,
                   help="Existing tier1 JSON to overlap-compare against (validation only).")
    p.add_argument("--no-write", action="store_true",
                   help="Compute everything but skip writing --output (for dry-run).")
    return p.parse_args()


def log_hyperparams(args: argparse.Namespace) -> None:
    logger.info("── Hyperparameters ──")
    for k, v in sorted(vars(args).items()):
        logger.info("  %s = %s", k, v)
    logger.info("──────────────────────")


def overlap_report(selected: list, against_path: Path) -> dict:
    with against_path.open() as f:
        existing = json.load(f)
    sel_pids = {c["protein_id"] for c in selected}
    sel_unis = {c["uniprot_id"] for c in selected}
    ex_pids = {c["protein_id"] for c in existing}
    ex_unis = {c["uniprot_id"] for c in existing}
    return {
        "n_selected": len(sel_pids),
        "n_existing": len(ex_pids),
        "protein_id_overlap": sorted(sel_pids & ex_pids),
        "protein_id_only_new": sorted(sel_pids - ex_pids),
        "protein_id_only_old": sorted(ex_pids - sel_pids),
        "uniprot_overlap": sorted(sel_unis & ex_unis),
        "uniprot_overlap_count": len(sel_unis & ex_unis),
    }


def main() -> int:
    args = parse_args()
    log_hyperparams(args)

    test_unis = load_test_uniprots(args.manifest_dir)
    logger.info("Loaded test split: %d UniProts from %s",
                len(test_unis), args.manifest_dir / "splits" / "test_ids.txt")

    uniprot_seqs = load_uniprot_seqs(args.uniprot_fasta)
    logger.info("Loaded %d UniProt sequences from %s",
                len(uniprot_seqs), args.uniprot_fasta)

    spans_by_uni = load_iedb_spans(args.mhc_if_v2, args.allele)
    n_spans_total = sum(len(v) for v in spans_by_uni.values())
    logger.info("Loaded %s spans across %d UniProts (allele=%s)",
                n_spans_total, len(spans_by_uni), args.allele)

    filters = FilterParams(
        min_len=args.min_len,
        max_len=args.max_len,
        max_resolution=args.max_resolution,
        min_spans=args.min_spans,
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
    )

    selected, stats = build_tier1_candidates(
        test_uniprots=test_unis,
        spans_by_uniprot=spans_by_uni,
        uniprot_seqs=uniprot_seqs,
        allele=args.allele,
        filters=filters,
        n_select=args.n_select,
        rank_by=args.rank_by,
        rank_descending=not args.rank_ascending,
        sifts_cache_dir=args.sifts_cache_dir,
        api_delay_s=args.api_delay_s,
        epitope_head_split=args.epitope_head_split,
    )

    logger.info("── Pipeline stats ──")
    logger.info("  test UniProts:                %d", stats["n_test_uniprots"])
    logger.info("  with %s spans: %d", args.allele, stats["n_with_spans"])
    logger.info("  with UniProt sequence:        %d", stats["n_with_seq"])
    logger.info("  with SIFTS chains:            %d", stats["n_with_sifts_chains"])
    logger.info("  candidates passing filters:   %d", stats["n_candidates_built"])
    logger.info("  selected (top --n-select):    %d", stats["n_selected"])
    drop = stats["uniprots_no_qualifying_chain"]
    if drop:
        logger.info("  drop reasons (first 10):")
        for uni, reason in drop[:10]:
            logger.info("    %s: %s", uni, reason)

    invalid = []
    for c in selected:
        errs = validate_tier1_candidate(c)
        if errs:
            invalid.append((c["protein_id"], errs))
    if invalid:
        logger.warning("Validation found %d issue(s):", len(invalid))
        for pid, errs in invalid:
            logger.warning("  %s: %s", pid, errs)
    else:
        logger.info("All selected candidates pass tier_validators.validate_tier1_candidate.")

    if args.validate_against is not None:
        rep = overlap_report(selected, args.validate_against)
        logger.info("── Overlap vs %s ──", args.validate_against)
        logger.info("  selected/existing protein_id overlap: %d / %d",
                    len(rep["protein_id_overlap"]), rep["n_existing"])
        logger.info("  uniprot overlap: %d", rep["uniprot_overlap_count"])
        if rep["protein_id_only_new"]:
            logger.info("  in NEW only (different chain or different selection): %s",
                        rep["protein_id_only_new"])
        if rep["protein_id_only_old"]:
            logger.info("  in OLD only (dropped): %s", rep["protein_id_only_old"])

    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(selected, f, indent=2)
        logger.info("Wrote %s (%d entries)", args.output, len(selected))
    else:
        logger.info("--no-write set; skipping output write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
