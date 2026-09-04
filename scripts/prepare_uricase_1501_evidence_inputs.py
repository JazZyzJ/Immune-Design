#!/usr/bin/env python3
"""Prepare the qualified non-clean DRB1*15:01 cohort for the strict evidence join."""

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


POLICY_OFFSETS = {
    "all_core_residues": tuple(range(9)),
    "anchors_P1P4P6P9": (0, 3, 5, 8),
    "P1_plus_P4": (0, 3),
    "P1_only": (0,),
}


class PreparationError(ValueError):
    """Raised when the frozen 15:01 evidence-input contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PreparationError(f"input table is missing: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")


def _semicolon(values: Sequence[object]) -> str:
    return ";".join(map(str, values))


def _build_baseline(
    *, cores: pd.DataFrame, sequence_by_id: dict[str, str], allele: str
) -> pd.DataFrame:
    rows = []
    for protein_id, sequence in sequence_by_id.items():
        parent_cores = cores[cores["protein_id"].astype(str).eq(protein_id)]
        for policy, offsets in POLICY_OFFSETS.items():
            positions = sorted(
                {
                    int(row.core_start_0b) + offset
                    for row in parent_cores.itertuples(index=False)
                    for offset in offsets
                }
            )
            fraction = len(positions) / len(sequence)
            rows.append(
                {
                    "protein_id": protein_id,
                    "allele": allele,
                    "sequence_length": len(sequence),
                    "open_policy": policy,
                    "n_open_positions": len(positions),
                    "open_fraction": round(fraction, 4),
                    "min_recovery_if_all_open": round(1.0 - fraction, 4),
                    "in_90_97_recovery_band": 0.03 <= fraction <= 0.10,
                    "n_cores": len(parent_cores),
                }
            )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--structure-terminal-manifest", type=Path, required=True)
    parser.add_argument("--cores", type=Path, required=True)
    parser.add_argument("--q00511-projection-prior", type=Path, required=True)
    parser.add_argument("--position-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-compute-parents", type=int, required=True)
    parser.add_argument("--expected-qualified-parents", type=int, required=True)
    parser.add_argument("--expected-redesign-parents", type=int, required=True)
    parser.add_argument("--source-allele", default="HLA-DRB1*15:01")
    parser.add_argument("--output-allele", default="HLA-DRB1_15_01")
    return parser


def run(args: argparse.Namespace, *, command: list[str]) -> Path:
    paths = {
        "selection_manifest": args.selection_manifest.expanduser().resolve(),
        "structure_terminal_manifest": args.structure_terminal_manifest.expanduser().resolve(),
        "cores": args.cores.expanduser().resolve(),
        "q00511_projection_prior": args.q00511_projection_prior.expanduser().resolve(),
        "position_annotations": args.position_annotations.expanduser().resolve(),
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise PreparationError(f"output directory already exists: {output_dir}")
    selection = _read_table(paths["selection_manifest"])
    terminal = _read_table(paths["structure_terminal_manifest"])
    cores_all = _read_table(paths["cores"])
    prior = _read_table(paths["q00511_projection_prior"])
    annotations = _read_table(paths["position_annotations"])
    if len(selection) != args.expected_compute_parents or len(terminal) != len(selection):
        raise PreparationError("selection/terminal parent count mismatch")
    if selection["protein_id"].astype(str).duplicated().any() or terminal[
        "protein_id"
    ].astype(str).duplicated().any():
        raise PreparationError("selection/terminal protein IDs are not unique")
    if set(selection["protein_id"].astype(str)) != set(terminal["protein_id"].astype(str)):
        raise PreparationError("selection and terminal parent sets differ")
    selection = selection.copy()
    selection["protein_id"] = selection["protein_id"].astype(str)
    terminal = terminal.copy()
    terminal["protein_id"] = terminal["protein_id"].astype(str)
    joined = selection.merge(
        terminal[["protein_id", "structure_status", "energy_eligible"]],
        on="protein_id",
        validate="one_to_one",
    )
    qualified = joined[
        joined["structure_status"].astype(str).eq("qualified")
        & joined["energy_eligible"].astype(bool)
    ].copy()
    if len(qualified) != args.expected_qualified_parents:
        raise PreparationError(
            f"qualified parents {len(qualified)} != expected {args.expected_qualified_parents}"
        )
    redesign = qualified[~qualified["wt_nmp_clean"].astype(bool)].copy()
    redesign = redesign.sort_values("frozen_order_6387").reset_index(drop=True)
    if len(redesign) != args.expected_redesign_parents:
        raise PreparationError(
            f"redesign parents {len(redesign)} != expected {args.expected_redesign_parents}"
        )
    sequence_by_id = dict(
        zip(redesign["protein_id"].astype(str), redesign["sequence"].astype(str), strict=True)
    )
    if any(len(sequence) != int(length) for sequence, length in zip(
        redesign["sequence"], redesign["sequence_length"], strict=True
    )):
        raise PreparationError("redesign sequence length mismatch")

    cores = cores_all[
        cores_all["protein_id"].astype(str).isin(sequence_by_id)
        & cores_all["allele"].astype(str).eq(args.source_allele)
    ].copy()
    if set(cores["protein_id"].astype(str)) != set(sequence_by_id):
        raise PreparationError("one or more redesign parents lack a 15:01 WT core")
    cores["allele"] = args.output_allele
    for row in cores.itertuples(index=False):
        sequence = sequence_by_id[str(row.protein_id)]
        start = int(row.core_start_0b)
        if int(row.core_start_1b) != start + 1 or sequence[start : start + 9] != str(
            row.core_seq
        ):
            raise PreparationError(f"{row.protein_id}: core numbering/sequence mismatch")
    cores = cores.sort_values(["protein_id", "core_start_0b"]).reset_index(drop=True)
    baseline = _build_baseline(
        cores=cores, sequence_by_id=sequence_by_id, allele=args.output_allele
    )

    selected_prior = prior[prior["protein_id"].astype(str).isin(sequence_by_id)].copy()
    if set(selected_prior["protein_id"].astype(str)) != set(sequence_by_id):
        raise PreparationError("Q00511 prior does not cover the redesign cohort")
    legacy_rows = []
    for protein_id in sequence_by_id:
        parent = selected_prior[selected_prior["protein_id"].astype(str).eq(protein_id)]
        hard = parent[
            parent["kind"].astype(str).eq("hard") & parent["identity_match"].astype(bool)
        ].sort_values("ref_index_0b")
        monitored = parent[
            ~parent["kind"].astype(str).eq("hard") & parent["identity_match"].astype(bool)
        ].sort_values("ref_index_0b")
        legacy_rows.append(
            {
                "protein_id": protein_id,
                "hard_anchor_indices_0b": _semicolon(hard["target_index_0b"].astype(int)),
                "hard_anchor_labels": _semicolon(hard["label"].astype(str)),
                "hard_anchor_expected_aa": _semicolon(hard["target_aa"].astype(str)),
                "monitored_shell_indices_0b": _semicolon(
                    monitored["target_index_0b"].astype(int)
                ),
                "monitored_shell_labels": _semicolon(monitored["label"].astype(str)),
                "source_config": str(paths["q00511_projection_prior"]),
            }
        )
    legacy = pd.DataFrame(legacy_rows)

    analog = annotations[annotations["protein_id"].astype(str).isin(sequence_by_id)].copy()
    analog = analog.rename(
        columns={
            "target_aa": "actual_aa",
            "functional_label": "analog_role",
        }
    )
    analog["source"] = "q00511_projection_plus_resolved_five_sample_relay"
    required_analog = {
        "protein_id",
        "target_index_0b",
        "actual_aa",
        "analog_role",
        "mapping_status",
        "source",
    }
    if not required_analog.issubset(analog.columns):
        raise PreparationError("position annotations cannot form homolog analog evidence")
    analog = analog.sort_values(["protein_id", "target_index_0b", "analog_role"])
    for row in analog.itertuples(index=False):
        sequence = sequence_by_id[str(row.protein_id)]
        if sequence[int(row.target_index_0b)] != str(row.actual_aa):
            raise PreparationError(f"{row.protein_id}: homolog analog AA mismatch")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        (stage / "parents.fasta").write_text(
            "".join(f">{protein_id}\n{sequence}\n" for protein_id, sequence in sequence_by_id.items())
        )
        (stage / "parents.ids").write_text("\n".join(sequence_by_id) + "\n")
        redesign.to_parquet(stage / "cohort.parquet", index=False)
        cores.to_parquet(stage / "cores_1501.parquet", index=False)
        baseline.to_parquet(stage / "open_policy_baseline_1501.parquet", index=False)
        legacy.to_csv(stage / "legacy_identity_map.csv", index=False)
        analog.to_parquet(stage / "homolog_analog_annotations.parquet", index=False)
        output_names = (
            "parents.fasta",
            "parents.ids",
            "cohort.parquet",
            "cores_1501.parquet",
            "open_policy_baseline_1501.parquet",
            "legacy_identity_map.csv",
            "homolog_analog_annotations.parquet",
        )
        metadata = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "compute_parents": len(selection),
                "structure_qualified_parents": len(qualified),
                "redesign_nonclean_parents": len(redesign),
                "cores": len(cores),
                "homolog_analog_rows": len(analog),
            },
            "allele": {"source": args.source_allele, "output": args.output_allele},
            "inputs": {
                label: {"path": str(path), "sha256": sha256_file(path)}
                for label, path in paths.items()
            },
            "outputs": {name: sha256_file(stage / name) for name in output_names},
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
    except PreparationError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
