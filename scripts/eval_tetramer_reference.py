#!/usr/bin/env python
"""Score one static tetramer reference, with optional Rosetta interface and Ala-scan metrics."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.evaluation.complex_interfaces import (
    aggregate_homomer_alanine_scan,
    canonical_protein_atoms,
    interface_pair_metrics,
    run_rosetta_alanine_scan,
    run_rosetta_interface_analyzer,
    summarize_interface_pairs,
)


def _load_structure(path: Path):
    lower_name = path.name.lower()
    if lower_name.endswith((".cif", ".mmcif")):
        from biotite.structure.io.pdbx import CIFFile, get_structure

        return get_structure(CIFFile.read(str(path)), model=1)
    if lower_name.endswith((".pdb", ".ent", ".pdb1")):
        from biotite.structure.io.pdb import PDBFile

        return PDBFile.read(str(path)).get_structure(model=1)
    raise ValueError(f"unsupported structure format: {path}")


def _parse_chain_pair(value: str) -> tuple[str, str]:
    fields = [field.strip() for field in re.split(r"[:,]", value) if field.strip()]
    if len(fields) != 2 or fields[0] == fields[1]:
        raise argparse.ArgumentTypeError(
            f"chain pair must contain two distinct IDs separated by ':' (got {value!r})"
        )
    return fields[0], fields[1]


def _parse_chains(value: str) -> list[str]:
    chains = [field.strip() for field in value.split(",") if field.strip()]
    if len(chains) != 4 or len(set(chains)) != 4:
        raise argparse.ArgumentTypeError("--chains requires four distinct comma-separated IDs")
    return chains


def _write_protein_pdb(arr, path: Path) -> None:
    from biotite.structure.io.pdb import PDBFile

    path.parent.mkdir(parents=True, exist_ok=True)
    pdb = PDBFile()
    pdb.set_structure(canonical_protein_atoms(arr))
    pdb.write(str(path))


def _write_table(df: pd.DataFrame, path_stem: Path) -> list[Path]:
    parquet_path = path_stem.with_suffix(".parquet")
    csv_path = path_stem.with_suffix(".csv")
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return [parquet_path, csv_path]


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="persistent reference tables and standardized structure (project work/ layer)",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        help="Rosetta inputs, protocols, commands, and scorefiles (project run/ layer)",
    )
    parser.add_argument(
        "--log-dir", type=Path,
        help="captured Rosetta stdout/stderr (project logs/ layer)",
    )
    parser.add_argument(
        "--chains", type=_parse_chains,
        help="four comma-separated protein chain IDs; default: auto-detect",
    )
    parser.add_argument("--sasa-probe-radius", type=float, default=1.4)
    parser.add_argument("--sasa-point-number", type=int, default=1000)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--salt-bridge-cutoff", type=float, default=4.0)
    parser.add_argument(
        "--rosetta-interface-analyzer",
        help="optional InterfaceAnalyzer executable",
    )
    parser.add_argument(
        "--rosetta-scripts",
        help="optional rosetta_scripts executable for per-residue binding ddG scans",
    )
    parser.add_argument(
        "--alanine-scan-pair", action="append", type=_parse_chain_pair, default=[],
        help="repeatable original-chain pair, e.g. --alanine-scan-pair A:B",
    )
    parser.add_argument("--interface-residue-cutoff", type=float, default=5.0)
    parser.add_argument("--rosetta-score-function", default="ref2015")
    parser.add_argument("--rosetta-timeout-seconds", type=int, default=3600)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.structure.is_file():
        parser.error(f"structure does not exist: {args.structure}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        parser.error("--name may contain only letters, numbers, '.', '_' and '-'")
    if args.sasa_point_number <= 0:
        parser.error("--sasa-point-number must be positive")
    if args.alanine_scan_pair and not args.rosetta_scripts:
        parser.error("--alanine-scan-pair requires --rosetta-scripts")
    rosetta_requested = bool(args.rosetta_interface_analyzer or args.alanine_scan_pair)
    if rosetta_requested and args.run_dir is None:
        parser.error("Rosetta scoring requires --run-dir")
    if rosetta_requested and args.log_dir is None:
        parser.error("Rosetta scoring requires --log-dir")

    source = args.structure.resolve()
    out_dir = args.out_dir.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir is not None else None
    log_dir = args.log_dir.resolve() if args.log_dir is not None else None
    roots = [("out", out_dir), ("run", run_dir), ("log", log_dir)]
    for index, (left_name, left_path) in enumerate(roots):
        if left_path is None:
            continue
        for right_name, right_path in roots[index + 1:]:
            if right_path is not None and _paths_overlap(left_path, right_path):
                parser.error(
                    f"--{left_name}-dir and --{right_name}-dir must be separate directory trees"
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    arr = _load_structure(source)
    protein = canonical_protein_atoms(arr)
    detected_chains = sorted(set(protein.chain_id.astype(str)))
    chains = args.chains or detected_chains
    if len(chains) != 4 or any(chain not in detected_chains for chain in chains):
        parser.error(
            f"expected exactly four selected protein chains; selected={chains}, "
            f"detected={detected_chains}"
        )

    output_paths = []
    cleaned_path = out_dir / f"{args.name}_protein_standardized.pdb"
    _write_protein_pdb(arr, cleaned_path)
    output_paths.append(cleaned_path)

    chain_rows = []
    protein_ins_codes = (
        protein.ins_code.astype(str)
        if "ins_code" in protein.get_annotation_categories()
        else np.full(len(protein), "", dtype="U1")
    )
    for chain in chains:
        chain_mask = protein.chain_id == chain
        residue_keys = sorted(set(zip(
            protein.res_id[chain_mask].astype(int), protein_ins_codes[chain_mask]
        )))
        residue_ids = sorted(set(res_id for res_id, _ in residue_keys))
        missing_internal = sorted(set(range(residue_ids[0], residue_ids[-1] + 1)) - set(residue_ids))
        chain_rows.append({
            "name": args.name,
            "chain": chain,
            "n_standardized_residues": len(residue_keys),
            "first_res_id": residue_ids[0],
            "last_res_id": residue_ids[-1],
            "missing_internal_res_ids": ",".join(map(str, missing_internal)),
        })
    chain_summary = pd.DataFrame(chain_rows)
    output_paths.extend(_write_table(
        chain_summary, out_dir / f"{args.name}_chain_summary"
    ))

    pairs = interface_pair_metrics(
        arr,
        chains,
        sasa_probe_radius=args.sasa_probe_radius,
        sasa_point_number=args.sasa_point_number,
        contact_cutoff=args.contact_cutoff,
        salt_bridge_cutoff=args.salt_bridge_cutoff,
    )
    pairs.insert(0, "name", args.name)
    if args.rosetta_interface_analyzer:
        assert run_dir is not None and log_dir is not None
        pairs = run_rosetta_interface_analyzer(
            pairs,
            lambda _: arr,
            executable=args.rosetta_interface_analyzer,
            run_dir=run_dir / "interface_analyzer",
            log_dir=log_dir / "interface_analyzer",
            timeout_seconds=args.rosetta_timeout_seconds,
        )

    summary = {
        "name": args.name,
        "source_structure": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "protein_chains": ",".join(chains),
        "n_protein_chains": len(chains),
        "standardized_residue_count_min": int(chain_summary["n_standardized_residues"].min()),
        "standardized_residue_count_max": int(chain_summary["n_standardized_residues"].max()),
        "sasa_probe_radius_a": float(args.sasa_probe_radius),
        "sasa_point_number": int(args.sasa_point_number),
        "contact_cutoff_a": float(args.contact_cutoff),
        "salt_bridge_cutoff_a": float(args.salt_bridge_cutoff),
        **summarize_interface_pairs(pairs),
    }
    output_paths.extend(_write_table(pairs, out_dir / f"{args.name}_interface_pairs"))

    scan = None
    by_position = None
    if args.alanine_scan_pair:
        assert run_dir is not None and log_dir is not None
        scan = run_rosetta_alanine_scan(
            arr,
            args.alanine_scan_pair,
            executable=args.rosetta_scripts,
            run_dir=run_dir / "alanine_scan",
            log_dir=log_dir / "alanine_scan",
            timeout_seconds=args.rosetta_timeout_seconds,
            interface_cutoff=args.interface_residue_cutoff,
            score_function=args.rosetta_score_function,
        )
        class_by_pair = pairs.set_index("chain_pair")["interface_class"].to_dict()
        scan["interface_class"] = scan["interface_pair"].map(class_by_pair)
        if scan["interface_class"].isna().any():
            unknown = sorted(scan.loc[scan["interface_class"].isna(), "interface_pair"].unique())
            raise ValueError(f"alanine-scan pairs are absent from tetramer pair table: {unknown}")
        by_position = aggregate_homomer_alanine_scan(scan)
        output_paths.extend(_write_table(
            scan, out_dir / f"{args.name}_alanine_scan_chain_residue"
        ))
        output_paths.extend(_write_table(
            by_position, out_dir / f"{args.name}_alanine_scan_by_position"
        ))
        for interface_pair, group in by_position.groupby("interface_pair", sort=True):
            prefix = "alanine_scan_" + interface_pair.replace(":", "")
            interpretable = group[group["is_interpretable_sidechain_alanine"]]
            summary[f"{prefix}_n_positions"] = int(len(group))
            summary[f"{prefix}_n_interpretable_sidechain_positions"] = int(len(interpretable))
            summary[f"{prefix}_all_mutations_ddg_bind_mean_reu"] = float(
                group["ddg_bind_mean_reu"].mean()
            )
            summary[f"{prefix}_all_mutations_ddg_bind_max_reu"] = float(
                group["ddg_bind_mean_reu"].max()
            )
            summary[f"{prefix}_sidechain_ddg_bind_mean_reu"] = float(
                interpretable["ddg_bind_mean_reu"].mean()
            )
            summary[f"{prefix}_sidechain_ddg_bind_max_reu"] = float(
                interpretable["ddg_bind_mean_reu"].max()
            )
            summary[f"{prefix}_sidechain_ddg_bind_min_reu"] = float(
                interpretable["ddg_bind_mean_reu"].min()
            )
            summary[f"{prefix}_max_chain_range_reu"] = float(
                group["ddg_bind_chain_range_reu"].max()
            )

    summary_df = pd.DataFrame([summary])
    output_paths.extend(_write_table(summary_df, out_dir / f"{args.name}_interface_summary"))
    summary_json_path = out_dir / f"{args.name}_interface_summary.json"
    summary_json_path.write_text(summary_df.to_json(orient="records", indent=2) + "\n")
    output_paths.append(summary_json_path)

    metadata = {
        "schema_version": "tetramer_reference_metrics_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "source_structure": str(source),
        "source_sha256": summary["source_sha256"],
        "persistent_output_dir": str(out_dir),
        "runtime_dir": str(run_dir) if run_dir is not None else None,
        "log_dir": str(log_dir) if log_dir is not None else None,
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "parameters": {
            "chains": chains,
            "sasa_probe_radius_a": args.sasa_probe_radius,
            "sasa_point_number": args.sasa_point_number,
            "contact_cutoff_a": args.contact_cutoff,
            "salt_bridge_cutoff_a": args.salt_bridge_cutoff,
            "interface_residue_cutoff_a": args.interface_residue_cutoff,
            "alanine_scan_pairs": [list(pair) for pair in args.alanine_scan_pair],
            "rosetta_score_function": args.rosetta_score_function,
            "alanine_scan_repack_bound": False,
            "alanine_scan_repack_unbound": False,
            "ddg_sign_convention": "mutant_minus_wildtype",
            "interpretable_sidechain_scan_excludes": ["ALA", "GLY", "PRO"],
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "biotite": _package_version("biotite"),
            "rosetta_interface_analyzer": args.rosetta_interface_analyzer,
            "rosetta_scripts": args.rosetta_scripts,
        },
        "outputs": [str(path) for path in output_paths],
        "runtime_artifacts": (
            [str(path) for path in sorted(run_dir.rglob("*")) if path.is_file()]
            if run_dir is not None else []
        ),
        "log_artifacts": (
            [str(path) for path in sorted(log_dir.rglob("*")) if path.is_file()]
            if log_dir is not None else []
        ),
        "row_counts": {
            "interface_pairs": int(len(pairs)),
            "chain_summary": int(len(chain_summary)),
            "alanine_scan_chain_residue": int(len(scan)) if scan is not None else 0,
            "alanine_scan_by_position": int(len(by_position)) if by_position is not None else 0,
        },
    }
    metadata_path = out_dir / "metadata.json"
    metadata["outputs"].append(str(metadata_path))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Wrote 1 summary row and {len(pairs)} interface-pair rows to {out_dir}")
    if scan is not None:
        print(
            f"Wrote {len(scan)} chain-residue and {len(by_position)} position-level "
            "alanine-scan rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
