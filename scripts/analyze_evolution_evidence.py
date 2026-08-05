#!/usr/bin/env python3
"""Analyze one query-centered MSA/PLMC model into reusable WT-lock evidence.

Inputs are deliberately explicit and target-local.  The three MSAs must already
be normalized to query columns and filtered at row coverage 0.6/0.7/0.8.  This
script does not search databases or train PLMC.

Residue-map TSV contract for optional structures:

``chain_id  residue_number  position_1b  [insertion_code]  [expected_aa]``

The monomer map must name exactly one chain and the tetramer map exactly four.
Partial mappings are accepted and structural precision skips unresolved ECs,
but every referenced residue must exist and match the WT amino acid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inverse_folding.analysis.evolution_evidence import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--protein-id", required=True)
    parser.add_argument("--wt-fasta", type=Path, required=True)
    parser.add_argument("--msa-cov60", type=Path, required=True)
    parser.add_argument("--msa-cov70", type=Path, required=True)
    parser.add_argument("--msa-cov80", type=Path, required=True)
    parser.add_argument("--plmc-model", type=Path, required=True)
    parser.add_argument("--ec-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--natural-minimum-coverage", type=float, default=0.95)
    parser.add_argument(
        "--minimum-calibration-unique",
        type=int,
        help=(
            "Optional analysis-owned minimum unique WT-imputed natural sequences. "
            "If omitted, calibration remains descriptive and no sufficiency gate is invented."
        ),
    )
    parser.add_argument("--min-sequence-separation", type=int, default=6)
    parser.add_argument("--score-chunk-size", type=int, default=256)
    parser.add_argument("--monomer-structure", type=Path)
    parser.add_argument("--monomer-mapping", type=Path)
    parser.add_argument("--tetramer-structure", type=Path)
    parser.add_argument("--tetramer-mapping", type=Path)
    parser.add_argument("--contact-cutoff-A", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = run_analysis(
        protein_id=args.protein_id,
        wt_fasta=args.wt_fasta,
        msa_specs=(
            ("cov60", 0.6, args.msa_cov60),
            ("cov70", 0.7, args.msa_cov70),
            ("cov80", 0.8, args.msa_cov80),
        ),
        model_path=args.plmc_model,
        ec_table_path=args.ec_table,
        out_dir=args.out_dir,
        theta=args.theta,
        natural_minimum_coverage=args.natural_minimum_coverage,
        minimum_calibration_unique=args.minimum_calibration_unique,
        min_sequence_separation=args.min_sequence_separation,
        monomer_structure=args.monomer_structure,
        monomer_mapping=args.monomer_mapping,
        tetramer_structure=args.tetramer_structure,
        tetramer_mapping=args.tetramer_mapping,
        contact_cutoff_A=args.contact_cutoff_A,
        score_chunk_size=args.score_chunk_size,
    )
    print(
        json.dumps(
            {
                "protein_id": metadata["protein_id"],
                "covariance_status": metadata["model"]["covariance_status"],
                "potts_calibration_status": metadata["potts"]["calibration_status"],
                "contact_gate_status": metadata["contact_gate"]["status"],
                "out_dir": str(args.out_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
