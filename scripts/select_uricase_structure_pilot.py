#!/usr/bin/env python3
"""Select and materialize the frozen 32-parent uricase structure pilot.

The pilot is a throughput and qualification stress set, not a prevalence
sample.  Selection is independent of predicted tetramer outcomes and covers
evolution depth, sequence length, structure source, legacy-Q00511 eligibility,
relay completeness, WT core burden, and known Active-15 positive controls.
Raw ColabFold A3Ms are content-verified from normalization QC and exposed as
absolute symlinks so Protenix input construction does not duplicate search
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
DEFAULT_SEED = "uricase-all6388-structure-pilot32-v1"
QUOTA_TARGETS: dict[str, int] = {
    "covariance_status=qualified": 8,
    "covariance_status=exploratory": 8,
    "covariance_status=insufficient": 8,
    "structure_source=afdb": 12,
    "structure_source=esmfold2": 6,
    "structure_source=af3": 1,
    "relay_complete=true": 12,
    "relay_complete=false": 8,
    "design_viable=true": 12,
    "design_viable=false": 8,
    "wt_nmp_clean=true": 4,
    "q00511_all_hard_roles_identity=false": 8,
    "length_bin=Q1": 6,
    "length_bin=Q2": 6,
    "length_bin=Q3": 6,
    "length_bin=Q4": 6,
    "depth_bin=Q1": 6,
    "depth_bin=Q2": 6,
    "depth_bin=Q3": 6,
    "depth_bin=Q4": 6,
    "core_burden_bin=low": 4,
    "core_burden_bin=mid": 4,
    "core_burden_bin=high": 4,
}


class PilotContractError(ValueError):
    """Raised when the frozen pilot cannot be reproduced safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_value(raw: str) -> object:
    if raw == "true":
        return True
    if raw == "false":
        return False
    return raw


def _quota_mask(frame: pd.DataFrame, key: str) -> pd.Series:
    if "=" not in key:
        raise PilotContractError(f"invalid quota key: {key!r}")
    column, raw_value = key.split("=", 1)
    if column not in frame:
        raise PilotContractError(f"quota column absent: {column}")
    expected = _expected_value(raw_value)
    return frame[column].eq(expected)


def quota_deficits(
    frame: pd.DataFrame, targets: Mapping[str, int]
) -> dict[str, int]:
    deficits = {}
    for key, target in targets.items():
        observed = int(_quota_mask(frame, key).sum())
        if observed < target:
            deficits[key] = target - observed
    return deficits


def _rank_bins(series: pd.Series, labels: Sequence[str]) -> pd.Series:
    ranks = series.astype(float).rank(method="first")
    return pd.qcut(ranks, q=len(labels), labels=list(labels)).astype(str)


def _prepare_selection_table(
    *,
    cohort: pd.DataFrame,
    evolution: pd.DataFrame,
    legacy: pd.DataFrame,
    relay: pd.DataFrame,
) -> pd.DataFrame:
    required_cohort = {
        "frozen_order_6387",
        "protein_id",
        "sequence",
        "sequence_length",
        "structure_source",
        "wt_nmp_clean",
        "n_distinct_strong_cores",
        "q00511_all_hard_roles_identity",
    }
    required_evolution = {
        "protein_id",
        "covariance_status",
        "potts_rows",
        "potts_unique_imputed_sequences",
        "n_C80_stable",
        "n_sigma80_robust",
    }
    required_legacy = {
        "protein_id",
        "design_viable",
        "viability",
        "n_anchors_matched",
        "active_site_complete",
    }
    required_relay = {
        "protein_id",
        "status",
        "n_hard_roles_mapped",
        "missing_hard_labels",
        "relay_annotation_path",
    }
    for name, table, required in (
        ("cohort", cohort, required_cohort),
        ("evolution", evolution, required_evolution),
        ("legacy", legacy, required_legacy),
        ("relay", relay, required_relay),
    ):
        missing = sorted(required - set(table.columns))
        if missing:
            raise PilotContractError(f"{name} table lacks columns: {missing}")
        if table["protein_id"].isna().any() or table["protein_id"].astype(str).duplicated().any():
            raise PilotContractError(f"{name} protein_id must be non-null and unique")

    cohort = cohort.copy()
    cohort["protein_id"] = cohort["protein_id"].astype(str)
    ids = set(cohort["protein_id"])
    joins = []
    for name, table in (("evolution", evolution), ("legacy", legacy), ("relay", relay)):
        table = table.copy()
        table["protein_id"] = table["protein_id"].astype(str)
        if not ids.issubset(set(table["protein_id"])):
            raise PilotContractError(f"{name} is missing cohort parents")
        joins.append(table.loc[table["protein_id"].isin(ids)].copy())
    evolution, legacy, relay = joins
    if set(evolution["protein_id"]) != ids or set(legacy["protein_id"]) != ids or set(relay["protein_id"]) != ids:
        raise PilotContractError("joined evidence tables do not exactly cover the cohort")

    frame = cohort.merge(evolution, on="protein_id", validate="one_to_one")
    frame = frame.merge(
        legacy[list(required_legacy)], on="protein_id", validate="one_to_one"
    )
    frame = frame.merge(
        relay[list(required_relay)], on="protein_id", validate="one_to_one"
    )
    frame["sequence"] = frame["sequence"].astype(str)
    if not frame.apply(lambda row: len(row.sequence) == int(row.sequence_length), axis=1).all():
        raise PilotContractError("cohort sequence length mismatch")
    if not set(frame["covariance_status"].astype(str)).issubset(
        {"qualified", "exploratory", "insufficient"}
    ):
        raise PilotContractError("invalid covariance status")
    frame["structure_source"] = frame["structure_source"].astype(str).str.lower()
    frame["relay_complete"] = frame["status"].eq(
        "candidate_complete_pending_geometry"
    )
    allowed_relay = {
        "candidate_complete_pending_geometry",
        "incomplete_projection",
    }
    if set(frame["status"].astype(str)) - allowed_relay:
        raise PilotContractError("invalid relay status")
    for column in (
        "design_viable",
        "wt_nmp_clean",
        "q00511_all_hard_roles_identity",
    ):
        frame[column] = frame[column].astype(bool)

    frame["length_bin"] = _rank_bins(frame["sequence_length"], ("Q1", "Q2", "Q3", "Q4"))
    frame["depth_bin"] = _rank_bins(frame["potts_rows"], ("Q1", "Q2", "Q3", "Q4"))
    frame["core_burden_bin"] = "clean"
    nonclean = frame.loc[~frame["wt_nmp_clean"]]
    frame.loc[nonclean.index, "core_burden_bin"] = _rank_bins(
        nonclean["n_distinct_strong_cores"], ("low", "mid", "high")
    )
    frame["_length_pct"] = frame["sequence_length"].rank(method="average", pct=True)
    frame["_depth_pct"] = frame["potts_rows"].rank(method="average", pct=True)
    frame["_core_pct"] = frame["n_distinct_strong_cores"].rank(
        method="average", pct=True
    )
    return frame


def _selection_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {key: int(_quota_mask(frame, key).sum()) for key in QUOTA_TARGETS}


def _hash_tiebreak(seed: str, protein_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}|{protein_id}".encode("utf-8")).digest(), "big"
    )


def select_pilot(
    *,
    cohort: pd.DataFrame,
    evolution: pd.DataFrame,
    legacy: pd.DataFrame,
    relay: pd.DataFrame,
    n_select: int,
    controls: Sequence[str],
    seed: str,
) -> pd.DataFrame:
    if n_select < 1:
        raise PilotContractError("n_select must be positive")
    frame = _prepare_selection_table(
        cohort=cohort, evolution=evolution, legacy=legacy, relay=relay
    )
    if n_select > len(frame):
        raise PilotContractError("n_select exceeds cohort size")
    controls = list(dict.fromkeys(str(value) for value in controls))
    missing_controls = sorted(set(controls) - set(frame["protein_id"]))
    if missing_controls:
        raise PilotContractError(f"controls absent from cohort: {missing_controls}")
    if len(controls) > n_select:
        raise PilotContractError("more controls than pilot slots")
    if n_select == len(frame):
        selected = frame.sort_values("frozen_order_6387").reset_index(drop=True)
        selected.insert(
            0, "selection_rank", np.arange(1, len(selected) + 1, dtype=int)
        )
        selected.insert(2, "is_control", selected["protein_id"].isin(controls))
        selected["selection_roles"] = selected["is_control"].map(
            {True: "positive_control", False: "full_cohort"}
        )
        return selected
    availability_deficits = quota_deficits(frame, QUOTA_TARGETS)
    if availability_deficits:
        raise PilotContractError(
            f"cohort cannot satisfy pilot quotas: {availability_deficits}"
        )

    by_id = frame.set_index("protein_id", drop=False)
    selected_ids = list(controls)
    numeric_columns = ["_length_pct", "_depth_pct", "_core_pct"]
    quota_masks = {key: _quota_mask(frame, key) for key in QUOTA_TARGETS}
    quota_prevalence = {key: int(mask.sum()) for key, mask in quota_masks.items()}
    while len(selected_ids) < n_select:
        selected = by_id.loc[selected_ids] if selected_ids else frame.iloc[0:0]
        deficits = quota_deficits(selected, QUOTA_TARGETS)
        selected_numeric = (
            selected[numeric_columns].to_numpy(dtype=float)
            if len(selected)
            else np.empty((0, len(numeric_columns)))
        )
        best_id: str | None = None
        best_score: tuple[float, float, float, int] | None = None
        for row_index, row_series in frame.loc[
            ~frame["protein_id"].isin(selected_ids)
        ].iterrows():
            matched_deficits = [
                key
                for key in deficits
                if bool(quota_masks[key].loc[row_index])
            ]
            quota_gain = sum(1.0 / QUOTA_TARGETS[key] for key in matched_deficits)
            candidate_numeric = np.asarray(
                [float(row_series[column]) for column in numeric_columns], dtype=float
            )
            diversity = (
                1.0
                if not len(selected_numeric)
                else float(
                    np.sqrt(
                        np.square(selected_numeric - candidate_numeric).sum(axis=1)
                    ).min()
                )
            )
            scarcity = sum(
                1.0 / max(1, quota_prevalence[key])
                for key in matched_deficits
            )
            score = (
                float(bool(matched_deficits)),
                quota_gain,
                diversity + scarcity,
                -_hash_tiebreak(seed, str(row_series["protein_id"])),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_id = str(row_series["protein_id"])
        if best_id is None:
            raise PilotContractError("selector exhausted candidates")
        selected_ids.append(best_id)

    selected = by_id.loc[selected_ids].copy().reset_index(drop=True)
    deficits = quota_deficits(selected, QUOTA_TARGETS)
    if deficits:
        raise PilotContractError(
            f"greedy pilot selection left quota deficits: {deficits}"
        )
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1, dtype=int))
    selected.insert(2, "is_control", selected["protein_id"].isin(controls))
    selected["selection_roles"] = selected.apply(
        lambda row: ";".join(
            (["positive_control"] if bool(row["is_control"]) else [])
            + [key for key in QUOTA_TARGETS if bool(_quota_mask(pd.DataFrame([row]), key).iloc[0])]
        ),
        axis=1,
    )
    return selected


def load_frozen_shard_mapping(
    normalized_shard_root: Path,
    *,
    expected_shards: int,
    expected_parents: int,
) -> tuple[dict[str, str], dict[str, str]]:
    expected_columns = [
        "protein_id",
        "query_id",
        "alignment_path",
        "covariance_gate",
        "neff_exact",
        "neff_per_length",
    ]
    mapping: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    for shard_index in range(expected_shards):
        tag = f"{shard_index:03d}"
        manifest = normalized_shard_root / tag / "plmc_manifest.tsv"
        if not manifest.is_file():
            raise PilotContractError(f"normalized shard manifest missing: {manifest}")
        with manifest.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != expected_columns:
                raise PilotContractError(
                    f"normalized shard manifest header mismatch: {manifest}"
                )
            for row_number, row in enumerate(reader, start=2):
                protein_id = str(row["protein_id"]).strip()
                query_id = str(row["query_id"]).strip()
                if not protein_id or query_id != protein_id:
                    raise PilotContractError(
                        f"invalid identity at {manifest}:{row_number}"
                    )
                if protein_id in mapping:
                    raise PilotContractError(
                        f"normalized shard overlap for {protein_id}: "
                        f"{mapping[protein_id]} and {tag}"
                    )
                mapping[protein_id] = tag
        manifest_hashes[tag] = sha256_file(manifest)
    if len(mapping) != expected_parents:
        raise PilotContractError(
            f"normalized shard parent count {len(mapping)} != {expected_parents}"
        )
    return mapping, manifest_hashes


def _atomic_materialize(
    *,
    selected: pd.DataFrame,
    normalized_shard_root: Path,
    output_dir: Path,
    expected_shards: int,
    expected_parents: int,
    inputs: dict[str, Path],
    controls: Sequence[str],
    seed: str,
    command: list[str],
) -> None:
    normalized_shard_root = normalized_shard_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise PilotContractError(f"output directory already exists: {output_dir}")
    if not normalized_shard_root.is_dir():
        raise PilotContractError(
            f"normalized shard root does not exist: {normalized_shard_root}"
        )
    shard_by_id, shard_manifest_hashes = load_frozen_shard_mapping(
        normalized_shard_root,
        expected_shards=expected_shards,
        expected_parents=expected_parents,
    )
    missing_selected = sorted(set(selected["protein_id"].astype(str)) - set(shard_by_id))
    if missing_selected:
        raise PilotContractError(
            f"selected parents absent from frozen shard mapping: {missing_selected}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        msa_dir = stage / "msa_cf"
        msa_dir.mkdir()
        rows = []
        for row in selected.itertuples(index=False):
            protein_id = str(row.protein_id)
            shard = shard_by_id[protein_id]
            qc_path = normalized_shard_root / shard / protein_id / "msa_qc.json"
            try:
                qc = json.loads(qc_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise PilotContractError(f"invalid MSA QC for {protein_id}: {qc_path}") from exc
            if (
                qc.get("schema_version") != 1
                or qc.get("protein_id") != protein_id
                or qc.get("query_sha256")
                != hashlib.sha256(str(row.sequence).encode("ascii")).hexdigest()
            ):
                raise PilotContractError(f"MSA QC identity mismatch for {protein_id}")
            raw_a3m = Path(str(qc.get("a3m_path", ""))).expanduser().resolve()
            if not raw_a3m.is_file() or sha256_file(raw_a3m) != qc.get("a3m_sha256"):
                raise PilotContractError(f"raw A3M hash/path mismatch for {protein_id}")
            link = msa_dir / f"{protein_id}.a3m"
            os.symlink(raw_a3m, link)
            record = row._asdict()
            record.update(
                {
                    "normalized_shard": shard,
                    "msa_qc_path": str(qc_path.resolve()),
                    "msa_qc_sha256": sha256_file(qc_path),
                    "raw_a3m_path": str(raw_a3m),
                    "raw_a3m_sha256": qc["a3m_sha256"],
                    "pilot_a3m_path": str(
                        (output_dir / "msa_cf" / f"{protein_id}.a3m").resolve()
                    ),
                    "prediction_name": f"tetra_wt_{protein_id}",
                }
            )
            rows.append(record)
        manifest = pd.DataFrame(rows)
        internal_columns = [column for column in manifest if column.startswith("_")]
        manifest = manifest.drop(columns=internal_columns)
        manifest.to_parquet(stage / "pilot_manifest.parquet", index=False)
        manifest.to_csv(stage / "pilot_manifest.tsv", sep="\t", index=False)
        (stage / "pilot.ids").write_text("\n".join(manifest["protein_id"]) + "\n")
        (stage / "pilot.fasta").write_text(
            "".join(
                f">{row.protein_id}\n{row.sequence}\n"
                for row in manifest[["protein_id", "sequence"]].itertuples(index=False)
            )
        )
        compact_msa = manifest[
            [
                "protein_id",
                "normalized_shard",
                "msa_qc_path",
                "msa_qc_sha256",
                "raw_a3m_path",
                "raw_a3m_sha256",
                "pilot_a3m_path",
            ]
        ]
        compact_msa.to_csv(stage / "msa_sources.tsv", sep="\t", index=False)
        output_names = (
            "pilot_manifest.parquet",
            "pilot_manifest.tsv",
            "pilot.ids",
            "pilot.fasta",
            "msa_sources.tsv",
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_contract": {
                "purpose": "throughput_and_structure_qualification_stress_set_not_prevalence_estimate",
                "n_select": len(manifest),
                "seed": seed,
                "controls": list(controls),
                "quota_targets": QUOTA_TARGETS,
                "quota_observed": _selection_counts(selected),
                "outcome_metrics_used": False,
            },
            "inputs": {
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in inputs.items()
            },
            "normalized_shard_root": str(normalized_shard_root),
            "normalized_shard_manifest_sha256": shard_manifest_hashes,
            "outputs": {name: sha256_file(stage / name) for name in output_names},
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "command": command,
                "python": sys.version,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--evolution-manifest", type=Path, required=True)
    parser.add_argument("--legacy-labels", type=Path, required=True)
    parser.add_argument("--relay-manifest", type=Path, required=True)
    parser.add_argument("--normalized-shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-parents", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, default=64)
    parser.add_argument("--n-select", type=int, default=32)
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = {
        "cohort": args.cohort.expanduser().resolve(),
        "evolution_manifest": args.evolution_manifest.expanduser().resolve(),
        "legacy_labels": args.legacy_labels.expanduser().resolve(),
        "relay_manifest": args.relay_manifest.expanduser().resolve(),
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise PilotContractError(f"{name} input does not exist: {path}")
    cohort = pd.read_parquet(inputs["cohort"])
    if len(cohort) != args.expected_parents:
        raise PilotContractError(
            f"cohort parent count {len(cohort)} != {args.expected_parents}"
        )
    selected = select_pilot(
        cohort=cohort,
        evolution=pd.read_parquet(inputs["evolution_manifest"]),
        legacy=pd.read_parquet(inputs["legacy_labels"]),
        relay=pd.read_parquet(inputs["relay_manifest"]),
        n_select=args.n_select,
        controls=args.control,
        seed=args.seed,
    )
    command = [str(Path(__file__).resolve()), *(list(argv) if argv is not None else sys.argv[1:])]
    _atomic_materialize(
        selected=selected,
        normalized_shard_root=args.normalized_shard_root,
        output_dir=args.output_dir,
        expected_shards=args.expected_shards,
        expected_parents=args.expected_parents,
        inputs=inputs,
        controls=args.control,
        seed=args.seed,
        command=command,
    )
    print(
        json.dumps(
            {
                "selected": len(selected),
                "controls": list(args.control),
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotContractError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
