#!/usr/bin/env python
"""B1: Download PDB structure files for assembled IF test set.

Reads the assembled test set parquet, extracts unique PDB codes from
protein_id (format: XXXX_C where XXXX is PDB code, C is chain), and
downloads structure files from RCSB. Each protein_id gets its own copy
(or symlink) of the PDB file at pdbs/{protein_id}.pdb, matching the
pdb_path convention in the assembled parquet.

Usage:
    python scripts/download_test_set_pdbs.py \
        --test-set-parquet /path/to/test_proteins_HLA-DRB1_07_01.parquet \
        --output-dir /path/to/if_test_set/pdbs \
        --workers 8

Network access required — run from login node, not compute node.
"""

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


RCSB_PDB_URL = "https://files.rcsb.org/download/{}.pdb"
RCSB_CIF_URL = "https://files.rcsb.org/download/{}.cif"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B1: Download PDB structures for IF test set",
    )
    p.add_argument(
        "--test-set-parquet", required=True,
        help="Path to assembled test set parquet (from L5 assembly).",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Output directory for PDB files (one per protein_id).",
    )
    p.add_argument(
        "--format", choices=["pdb", "cif"], default="pdb",
        help="Structure file format to download (default: pdb).",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel download threads (default: 4).",
    )
    p.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries per PDB code on network failure (default: 3).",
    )
    p.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip protein_ids whose PDB file already exists (default: True).",
    )
    p.add_argument(
        "--no-skip-existing", action="store_false", dest="skip_existing",
        help="Re-download even if file exists.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print download plan without downloading.",
    )
    return p.parse_args()


def extract_pdb_code(protein_id: str) -> str:
    """Extract 4-char PDB code from protein_id (e.g., '1ABC_A' → '1ABC')."""
    return protein_id.split("_")[0].upper()


def download_one_pdb(
    pdb_code: str,
    cache_dir: str,
    fmt: str,
    max_retries: int,
) -> tuple[str, str | None]:
    """Download a single PDB file to cache_dir.

    Returns (pdb_code, error_message_or_None).
    """
    url_template = RCSB_PDB_URL if fmt == "pdb" else RCSB_CIF_URL
    url = url_template.format(pdb_code.lower())
    dest = os.path.join(cache_dir, f"{pdb_code}.{fmt}")

    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return pdb_code, None  # already cached

    for attempt in range(1, max_retries + 1):
        try:
            urllib.request.urlretrieve(url, dest)
            if os.path.getsize(dest) > 0:
                return pdb_code, None
            else:
                os.remove(dest)
                return pdb_code, "empty file"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return pdb_code, f"404 Not Found"
            if attempt == max_retries:
                return pdb_code, f"HTTP {e.code} after {max_retries} retries"
        except (urllib.error.URLError, OSError) as e:
            if attempt == max_retries:
                return pdb_code, f"{type(e).__name__}: {e}"
        time.sleep(1.0 * attempt)  # backoff

    return pdb_code, "unknown failure"


def main() -> int:
    args = parse_args()

    # ── Read parquet ─────────────────────────────────────────────────────
    df = pd.read_parquet(args.test_set_parquet)
    if "protein_id" not in df.columns:
        print(f"ERROR: parquet missing 'protein_id' column")
        return 1

    protein_ids = df["protein_id"].unique().tolist()
    pdb_codes = sorted(set(extract_pdb_code(pid) for pid in protein_ids))

    # Map PDB code → list of protein_ids that need it
    code_to_pids: dict[str, list[str]] = {}
    for pid in protein_ids:
        code = extract_pdb_code(pid)
        code_to_pids.setdefault(code, []).append(pid)

    # ── Check what already exists ────────────────────────────────────────
    fmt = args.format
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, ".pdb_cache")
    os.makedirs(cache_dir, exist_ok=True)

    if args.skip_existing:
        already_done = set()
        for pid in protein_ids:
            dest = os.path.join(args.output_dir, f"{pid}.{fmt}")
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                already_done.add(pid)
        remaining_pids = [p for p in protein_ids if p not in already_done]
        remaining_codes = sorted(set(
            extract_pdb_code(pid) for pid in remaining_pids
        ))
    else:
        already_done = set()
        remaining_pids = protein_ids
        remaining_codes = pdb_codes

    # ── Tier breakdown for diagnostics ───────────────────────────────────
    tier_counts = {}
    if "tier" in df.columns:
        tier_counts = df.groupby("tier")["protein_id"].nunique().to_dict()

    # ── Print diagnostics ────────────────────────────────────────────────
    print("=" * 60)
    print("B1: Download PDB Structures for IF Test Set")
    print(f"  Parquet        : {args.test_set_parquet}")
    print(f"  Format         : {fmt}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Workers        : {args.workers}")
    print(f"  Max retries    : {args.max_retries}")
    print(f"  Skip existing  : {args.skip_existing}")
    print(f"  Total proteins : {len(protein_ids)}")
    print(f"  Unique PDB codes: {len(pdb_codes)}")
    for tier_val, cnt in sorted(tier_counts.items()):
        print(f"    Tier {tier_val}: {cnt} proteins")
    print(f"  Already done   : {len(already_done)}")
    print(f"  To download    : {len(remaining_codes)} PDB codes "
          f"→ {len(remaining_pids)} protein files")
    print("=" * 60)

    if args.dry_run:
        print("\n[dry-run] Would download:")
        for code in remaining_codes[:20]:
            pids = code_to_pids[code]
            print(f"  {code} → {len(pids)} protein(s): {pids[:3]}...")
        if len(remaining_codes) > 20:
            print(f"  ... and {len(remaining_codes) - 20} more")
        return 0

    if not remaining_codes:
        print("\nAll PDB files already present. Nothing to download.")
        return 0

    # ── Download unique PDB codes ────────────────────────────────────────
    print(f"\n[1/2] Downloading {len(remaining_codes)} unique PDB files...")
    t0 = time.time()
    successes = 0
    failures = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one_pdb, code, cache_dir, fmt, args.max_retries,
            ): code
            for code in remaining_codes
        }
        for i, future in enumerate(as_completed(futures), 1):
            code, error = future.result()
            if error:
                failures[code] = error
            else:
                successes += 1
            if i % 100 == 0 or i == len(futures):
                elapsed = time.time() - t0
                print(f"  [{i}/{len(futures)}] downloaded, "
                      f"{successes} ok, {len(failures)} failed "
                      f"({elapsed:.0f}s)")

    # ── Distribute to per-protein_id files ───────────────────────────────
    print(f"\n[2/2] Linking PDB files to {len(remaining_pids)} protein IDs...")
    linked = 0
    link_failures = []

    for pid in remaining_pids:
        code = extract_pdb_code(pid)
        src = os.path.join(cache_dir, f"{code}.{fmt}")
        dest = os.path.join(args.output_dir, f"{pid}.{fmt}")

        if code in failures:
            link_failures.append((pid, failures[code]))
            continue

        if not os.path.isfile(src):
            link_failures.append((pid, "cache file missing"))
            continue

        # Hard link to save space (same filesystem); fall back to copy
        try:
            if os.path.exists(dest):
                os.remove(dest)
            os.link(src, dest)
            linked += 1
        except OSError:
            shutil.copy2(src, dest)
            linked += 1

    # ── Report ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s:")
    print(f"  PDB codes downloaded: {successes}/{len(remaining_codes)}")
    print(f"  Protein files linked : {linked}/{len(remaining_pids)}")
    print(f"  Failures             : {len(failures)} codes, "
          f"{len(link_failures)} proteins")
    print(f"  Output               : {args.output_dir}")

    if failures:
        fail_path = os.path.join(args.output_dir, "download_failures.txt")
        with open(fail_path, "w") as f:
            for code, err in sorted(failures.items()):
                pids = code_to_pids.get(code, [])
                f.write(f"{code}\t{err}\t{','.join(pids)}\n")
        print(f"  Failure log          : {fail_path}")

    # Exit 0 if most succeeded, exit 1 if majority failed
    fail_rate = len(failures) / max(len(remaining_codes), 1)
    if fail_rate > 0.5:
        print(f"\nERROR: {fail_rate:.0%} failure rate — check network access")
        return 1
    elif failures:
        print(f"\nWARNING: {len(failures)} PDB codes failed — "
              f"see download_failures.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
