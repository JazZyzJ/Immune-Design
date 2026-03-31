#!/usr/bin/env python
"""Download and filter PDB sequences for Tier 2 candidate pool.

Downloads the full RCSB sequence database (pdb_seqres.txt.gz), then
filters to single-chain X-ray structures with 100-500 AA and resolution
<= 2.5 Å using a pre-computed entity ID list.

Usage:
    python scripts/download_tier2_candidates.py \
        --entity-ids outputs/if/test_set/tier2_all_entity_ids.txt \
        --output-dir outputs/if/test_set/tier2_candidates/ \
        [--seqres-gz /path/to/pdb_seqres.txt.gz]

If --seqres-gz is not provided, the script downloads it from RCSB.
"""

import argparse
import gzip
import os
import sys
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and filter Tier 2 PDB candidate sequences."
    )
    parser.add_argument(
        "--entity-ids", required=True,
        help="Text file with one RCSB polymer entity ID per line (XXXX_N).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for individual FASTA files.",
    )
    parser.add_argument(
        "--seqres-gz", default=None,
        help="Path to pre-downloaded pdb_seqres.txt.gz. "
             "If not provided, downloads from RCSB (~100MB).",
    )
    parser.add_argument(
        "--merged-fasta", default=None,
        help="Also write a single merged FASTA (for MMseqs2 input).",
    )
    return parser.parse_args()


def download_seqres(dest_path: str) -> str:
    """Download pdb_seqres.txt.gz from RCSB FTP."""
    url = "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"  Saved to {dest_path} ({os.path.getsize(dest_path) / 1e6:.1f} MB)")
    return dest_path


def load_entity_ids(path: str) -> set:
    """Load entity IDs and build lookup sets.

    Entity IDs are in format 'XXXX_N'. We need to match against
    FASTA headers which use 'XXXX_C' (PDB ID + chain).
    Maps entity to PDB ID for matching.
    """
    ids = set()
    pdb_ids = set()
    with open(path) as f:
        for line in f:
            eid = line.strip()
            if eid:
                ids.add(eid)
                pdb_ids.add(eid.split("_")[0].upper())
    return ids, pdb_ids


def parse_and_filter_seqres(
    seqres_path: str,
    pdb_ids: set,
    output_dir: str,
    merged_fasta_path: str = None,
) -> dict:
    """Parse pdb_seqres.txt.gz and extract matching protein sequences.

    FASTA headers in pdb_seqres.txt look like:
    >XXXX_C mol:protein length:NNN  description
    where XXXX is PDB ID (lowercase) and C is chain ID.

    Returns stats dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    stats = {"total_entries": 0, "matched": 0, "proteins_only": 0, "written": 0}
    merged_fh = open(merged_fasta_path, "w") if merged_fasta_path else None

    open_fn = gzip.open if seqres_path.endswith(".gz") else open
    current_header = None
    current_seq = []
    is_protein = False
    is_match = False

    def flush():
        nonlocal current_header, current_seq, is_protein, is_match
        if current_header and is_protein and is_match and current_seq:
            seq = "".join(current_seq)
            seq_len = len(seq)
            if 100 <= seq_len <= 500:
                # Extract chain ID: >xxxx_C → XXXX_C
                parts = current_header.split()[0][1:]  # remove >
                pdb_chain = parts.upper()
                fasta_content = f">{pdb_chain}\n{seq}\n"

                # Write individual file
                out_path = os.path.join(output_dir, f"{pdb_chain}.fasta")
                with open(out_path, "w") as f:
                    f.write(fasta_content)

                # Write to merged if requested
                if merged_fh:
                    merged_fh.write(fasta_content)

                stats["written"] += 1
        current_header = None
        current_seq = []
        is_protein = False
        is_match = False

    with open_fn(seqres_path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                stats["total_entries"] += 1
                current_header = line
                # Check if protein
                if "mol:protein" in line:
                    is_protein = True
                    stats["proteins_only"] += 1
                    # Check PDB ID match
                    pdb_id = line[1:5].upper()
                    if pdb_id in pdb_ids:
                        is_match = True
                        stats["matched"] += 1
            else:
                if is_protein and is_match:
                    current_seq.append(line.strip())

            if stats["total_entries"] % 500000 == 0 and stats["total_entries"] > 0:
                print(f"  Parsed {stats['total_entries']} entries, "
                      f"matched {stats['matched']}, written {stats['written']}...")

    flush()  # last entry

    if merged_fh:
        merged_fh.close()

    return stats


def main() -> int:
    args = parse_args()

    # Load entity IDs
    entity_ids, pdb_ids = load_entity_ids(args.entity_ids)
    print(f"Loaded {len(entity_ids)} entity IDs "
          f"({len(pdb_ids)} unique PDB codes)")

    # Download or use existing seqres file
    if args.seqres_gz and os.path.isfile(args.seqres_gz):
        seqres_path = args.seqres_gz
        print(f"Using existing: {seqres_path}")
    else:
        seqres_path = os.path.join(
            os.path.dirname(args.output_dir), "pdb_seqres.txt.gz"
        )
        if os.path.isfile(seqres_path):
            print(f"Using cached: {seqres_path}")
        else:
            download_seqres(seqres_path)

    # Parse and filter
    print("Parsing and filtering sequences...")
    merged_path = args.merged_fasta or os.path.join(
        os.path.dirname(args.output_dir), "tier2_candidates_merged.fasta"
    )
    stats = parse_and_filter_seqres(
        seqres_path, pdb_ids, args.output_dir, merged_path,
    )

    print(f"\nDone:")
    print(f"  Total FASTA entries parsed: {stats['total_entries']}")
    print(f"  Protein entries: {stats['proteins_only']}")
    print(f"  Matched to our PDB list: {stats['matched']}")
    print(f"  Written (100-500 AA): {stats['written']}")
    print(f"  Individual files → {args.output_dir}")
    print(f"  Merged FASTA → {merged_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
