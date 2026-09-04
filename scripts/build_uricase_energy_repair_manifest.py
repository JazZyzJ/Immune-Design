#!/usr/bin/env python3
"""Freeze the incomplete subset of an absent-only uricase energy campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


class RepairManifestError(ValueError):
    """Raised when completed/missing energy rows cannot be reconciled."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-manifest", type=Path, required=True)
    parser.add_argument("--energy-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-parents", type=int, required=True)
    parser.add_argument("--expected-missing", type=int, required=True)
    return parser


def run(args: argparse.Namespace, *, command: list[str]) -> Path:
    manifest_path = args.energy_manifest.expanduser().resolve()
    energy_root = args.energy_output_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not manifest_path.is_file() or not energy_root.is_dir():
        raise RepairManifestError("energy manifest or output root is missing")
    if output_dir.exists():
        raise RepairManifestError(f"output directory already exists: {output_dir}")
    frame = pd.read_parquet(manifest_path).reset_index(drop=True)
    if len(frame) != args.expected_parents or frame["protein_id"].astype(str).duplicated().any():
        raise RepairManifestError("energy manifest count/identity mismatch")
    if frame["energy_order"].astype(int).tolist() != list(range(len(frame))):
        raise RepairManifestError("energy_order is not row-aligned")
    by_id = frame.set_index(frame["protein_id"].astype(str), drop=False)
    manifest_sha256 = sha256_file(manifest_path)
    observed_dirs = {path.name for path in energy_root.iterdir() if path.is_dir()}
    unexpected = sorted(observed_dirs - set(by_id.index))
    if unexpected:
        raise RepairManifestError(f"energy output has unexpected parent directories: {unexpected}")
    complete_ids = []
    for protein_id in sorted(observed_dirs):
        marker = energy_root / protein_id / "parent_completion.json"
        if not marker.is_file():
            continue
        try:
            payload = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RepairManifestError(f"invalid completion marker: {marker}") from exc
        expected_order = int(by_id.loc[protein_id, "energy_order"])
        if not (
            payload.get("status") == "complete"
            and payload.get("protein_id") == protein_id
            and int(payload.get("row_index", -1)) == expected_order
            and payload.get("energy_manifest") == str(manifest_path)
            and payload.get("energy_manifest_sha256") == manifest_sha256
        ):
            raise RepairManifestError(f"completion marker identity mismatch: {protein_id}")
        complete_ids.append(protein_id)
    missing = frame[~frame["protein_id"].astype(str).isin(complete_ids)].copy()
    if len(missing) != args.expected_missing:
        raise RepairManifestError(
            f"missing rows {len(missing)} != expected {args.expected_missing}"
        )
    missing = missing.rename(columns={"energy_order": "source_energy_order"})
    missing.insert(0, "energy_order", range(len(missing)))
    missing["prior_partial_output_path"] = missing["protein_id"].map(
        lambda protein_id: str((energy_root / str(protein_id)).resolve())
        if (energy_root / str(protein_id)).exists()
        else None
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        missing.to_parquet(stage / "energy_repair_manifest.parquet", index=False)
        missing.to_csv(stage / "energy_repair_manifest.tsv", sep="\t", index=False)
        (stage / "energy_repair.ids").write_text("\n".join(missing["protein_id"]) + "\n")
        outputs = (
            "energy_repair_manifest.parquet",
            "energy_repair_manifest.tsv",
            "energy_repair.ids",
        )
        metadata = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "source_parents": len(frame),
                "complete_parents": len(complete_ids),
                "repair_parents": len(missing),
                "prior_partial_parent_dirs": int(missing["prior_partial_output_path"].notna().sum()),
            },
            "inputs": {
                "energy_manifest": {
                    "path": str(manifest_path),
                    "sha256": manifest_sha256,
                },
                "energy_output_root": str(energy_root),
            },
            "outputs": {name: sha256_file(stage / name) for name in outputs},
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "command": command,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [str(Path(__file__).resolve()), *(list(argv) if argv is not None else sys.argv[1:])]
    output = run(args, command=command)
    print(json.dumps({"output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepairManifestError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
