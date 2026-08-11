#!/usr/bin/env python3
"""Evaluate the exact stochastic Gumbel DPLM-native comparator on the V2 live stack.

This is a thin evidence adapter, not a second evaluator implementation.  It builds the frozen Head
and structure oracles through :func:`scripts.rf_fusion_v2_oracles.build_production_oracles`, sends
the eight designs of one protein through one ``ProductionHeadOracle`` batch, and sends every design
through the same persistent ``esmfold2_live`` structure path used by V2.

The source gate is intentionally narrow: the producer must be the complete ``c0_native_sampler``
run with ``sampling_strategy=gumbel_argmax`` and exactly eight unique designs per selected protein.
An argmax facade with eight repeated rows is not this comparator and is refused.
Temperature 1.0, ``max_iter=100`` and seeds 42--49 are part of the closed sampling law. The caller
must also supply pre-frozen SHA-256 values for the source parquet, run config, manifest, DPLM
checkpoint and selected campaign cohort; observed hashes written only after evaluation would not
prevent source substitution.

The normalized refold cache predates evaluator provenance in its filename.  This adapter therefore
claims a dedicated cache with an identity sidecar before reusing any entry.  A cache containing PDB
or pLDDT files without that claim is foreign and is refused; otherwise an old dedicated-env fold
could silently masquerade as an exact V2-live fold.

``model_selector`` is deliberately narrower than Hugging Face's general interface: it must be an
absolute local snapshot directory whose ``model.safetensors`` bytes match the declared SHA-256.
Registry aliases are refused, and the live worker must report that exact resolved directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml


EXPECTED_DESIGNS_PER_PROTEIN = 8
EXPECTED_GUMBEL_TEMPERATURE = 1.0
EXPECTED_GUMBEL_SEED_BASE = 42
EXPECTED_GUMBEL_MAX_ITER = 100
CACHE_IDENTITY_NAME = ".rf_fusion_v2_exact_live_cache_identity.json"
SCHEMA_VERSION = "rf-fusion-v2-exact-live-gumbel-baseline/1"
CACHE_SCHEMA_VERSION = "rf-fusion-v2-exact-live-cache/1"
AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")


class ExactLiveBaselineError(RuntimeError):
    """An input or result cannot support the claimed exact-live comparison."""


def _sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _canonical_digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _require_file(path: Any, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ExactLiveBaselineError(f"{label} is not a readable file: {resolved}")
    return resolved


def _require_dir(path: Any, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ExactLiveBaselineError(f"{label} is not a readable directory: {resolved}")
    return resolved


def _require_expected_sha256(path: Path, expected: Any, label: str) -> str:
    expected_text = str(expected)
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ExactLiveBaselineError(
            f"expected {label} SHA-256 must be 64 lowercase hexadecimal characters, "
            f"got {expected!r}"
        )
    observed = _sha256_file(path)
    if observed != expected_text:
        raise ExactLiveBaselineError(
            f"{label} SHA-256 disagrees with the pre-frozen source identity: "
            f"observed {observed}, expected {expected_text}"
        )
    return observed


def _mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except Exception as exc:  # noqa: BLE001 - normalize parser failures at the input boundary
        raise ExactLiveBaselineError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ExactLiveBaselineError(f"{label} must contain a mapping, got {type(value).__name__}")
    return dict(value)


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ExactLiveBaselineError(f"{label} must be an integer, got {value!r}")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ExactLiveBaselineError(f"{label} must be an integer, got {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ExactLiveBaselineError(f"{label} must be an integer, got {value!r}")
    return integer


def _as_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ExactLiveBaselineError(f"{label} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExactLiveBaselineError(f"{label} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ExactLiveBaselineError(f"{label} must be a finite number, got {value!r}")
    return number


def load_gumbel_baseline(
    *,
    source_parquet: Any,
    cohort_parquet: Any,
    source_run_config: Any,
    source_manifest: Any,
    source_checkpoint: Any,
    expected_source_parquet_sha256: str,
    expected_source_run_config_sha256: str,
    expected_source_manifest_sha256: str,
    expected_source_checkpoint_sha256: str,
    expected_cohort_parquet_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and select the exact eight-design Gumbel comparator for one cohort.

    The source parquet must be the producer's complete output: ``manifest.n_rows_generated`` is
    checked against its row count before cohort selection.  This prevents a hand-built facade from
    borrowing a valid producer manifest while omitting or replacing source rows.
    """
    source_path = _require_file(source_parquet, "source parquet")
    cohort_path = _require_file(cohort_parquet, "cohort parquet")
    config_path = _require_file(source_run_config, "source run config")
    manifest_path = _require_file(source_manifest, "source manifest")
    checkpoint_path = _require_file(source_checkpoint, "source DPLM checkpoint")

    source_parquet_digest = _require_expected_sha256(
        source_path, expected_source_parquet_sha256, "source parquet")
    source_run_config_digest = _require_expected_sha256(
        config_path, expected_source_run_config_sha256, "source run config")
    source_manifest_digest = _require_expected_sha256(
        manifest_path, expected_source_manifest_sha256, "source manifest")
    checkpoint_digest = _require_expected_sha256(
        checkpoint_path, expected_source_checkpoint_sha256, "source checkpoint")
    cohort_digest = _require_expected_sha256(
        cohort_path, expected_cohort_parquet_sha256, "cohort parquet")

    config = _mapping(config_path, label="source run config")
    manifest = _mapping(manifest_path, label="source manifest")
    sampler = config.get("sampler")
    if not isinstance(sampler, Mapping):
        raise ExactLiveBaselineError("source run config has no sampler mapping")

    for where, mode in (("run config", config.get("mode")), ("manifest", manifest.get("mode"))):
        if mode != "c0_native_sampler":
            raise ExactLiveBaselineError(
                f"source {where} mode must be 'c0_native_sampler', got {mode!r}"
            )
    strategy = sampler.get("sampling_strategy")
    if strategy != "gumbel_argmax":
        raise ExactLiveBaselineError(
            "the V2 comparator must be the stochastic gumbel_argmax DPLM-native baseline; "
            f"source declares {strategy!r}"
        )
    n_designs = _as_int(sampler.get("n_designs_per_protein"), "n_designs_per_protein")
    if n_designs != EXPECTED_DESIGNS_PER_PROTEIN:
        raise ExactLiveBaselineError(
            f"source must declare exactly {EXPECTED_DESIGNS_PER_PROTEIN} designs per protein, "
            f"got {n_designs}"
        )
    batch_size = _as_int(sampler.get("batch_size"), "source sampler batch_size")
    if batch_size != EXPECTED_DESIGNS_PER_PROTEIN:
        raise ExactLiveBaselineError(
            f"source Gumbel batch_size must be {EXPECTED_DESIGNS_PER_PROTEIN}, got {batch_size}"
        )
    temperature = _as_finite_float(
        sampler.get("temperature"), "source sampler temperature")
    if temperature != EXPECTED_GUMBEL_TEMPERATURE:
        raise ExactLiveBaselineError(
            "the frozen Gumbel comparator requires temperature="
            f"{EXPECTED_GUMBEL_TEMPERATURE}, got {temperature}"
        )
    seed_base = _as_int(sampler.get("seed"), "source sampler seed")
    if seed_base != EXPECTED_GUMBEL_SEED_BASE:
        raise ExactLiveBaselineError(
            f"the frozen Gumbel comparator requires seed={EXPECTED_GUMBEL_SEED_BASE}, "
            f"got {seed_base}"
        )
    max_iter = _as_int(sampler.get("max_iter"), "source sampler max_iter")
    if max_iter != EXPECTED_GUMBEL_MAX_ITER:
        raise ExactLiveBaselineError(
            f"the frozen Gumbel comparator requires max_iter={EXPECTED_GUMBEL_MAX_ITER}, "
            f"got {max_iter}"
        )
    if _as_int(manifest.get("n_failures", -1), "source manifest n_failures") != 0:
        raise ExactLiveBaselineError("source manifest reports generation failures")

    declared_checkpoint_digest = str(manifest.get("checkpoint_digest", ""))
    if checkpoint_digest != declared_checkpoint_digest:
        raise ExactLiveBaselineError(
            "source checkpoint bytes disagree with manifest.checkpoint_digest: "
            f"observed {checkpoint_digest}, declared {declared_checkpoint_digest!r}"
        )

    source = pd.read_parquet(source_path)
    cohort = pd.read_parquet(cohort_path)
    required_source = {"protein_id", "design_idx", "sequence", "seed"}
    missing_source = sorted(required_source - set(source.columns))
    if missing_source:
        raise ExactLiveBaselineError(f"source parquet is missing columns {missing_source}")
    required_cohort = {"protein_id", "sequence"}
    missing_cohort = sorted(required_cohort - set(cohort.columns))
    if missing_cohort:
        raise ExactLiveBaselineError(f"cohort parquet is missing columns {missing_cohort}")
    declared_rows = _as_int(manifest.get("n_rows_generated"), "manifest n_rows_generated")
    if len(source) != declared_rows:
        raise ExactLiveBaselineError(
            f"source parquet has {len(source)} rows but its manifest declares {declared_rows}; "
            "pass the complete producer parquet, not a reconstructed facade"
        )

    source = source.copy()
    cohort = cohort.copy()
    source["protein_id"] = source["protein_id"].astype(str)
    cohort["protein_id"] = cohort["protein_id"].astype(str)
    if source[["protein_id", "design_idx"]].duplicated().any():
        duplicated = source.loc[
            source[["protein_id", "design_idx"]].duplicated(keep=False),
            ["protein_id", "design_idx"],
        ].head(5).to_dict("records")
        raise ExactLiveBaselineError(f"source has duplicate protein/design keys: {duplicated}")
    if cohort["protein_id"].duplicated().any():
        duplicated = sorted(cohort.loc[cohort["protein_id"].duplicated(), "protein_id"].unique())
        raise ExactLiveBaselineError(f"cohort has duplicate protein_id rows: {duplicated[:5]}")

    cohort_ids = set(cohort["protein_id"])
    source_ids = set(source["protein_id"])
    missing_ids = sorted(cohort_ids - source_ids)
    if missing_ids:
        raise ExactLiveBaselineError(
            f"source is missing {len(missing_ids)} cohort protein(s): {missing_ids[:5]}"
        )
    selected = source[source["protein_id"].isin(cohort_ids)].copy()
    selected["design_idx"] = selected["design_idx"].map(
        lambda value: _as_int(value, "design_idx")
    )
    selected["seed"] = selected["seed"].map(lambda value: _as_int(value, "design seed"))
    cohort_lookup = cohort.set_index("protein_id", drop=False)

    for protein_id, group in selected.groupby("protein_id", sort=True):
        if len(group) != EXPECTED_DESIGNS_PER_PROTEIN:
            raise ExactLiveBaselineError(
                f"{protein_id} has {len(group)} source rows; expected exactly "
                f"{EXPECTED_DESIGNS_PER_PROTEIN}"
            )
        design_indices = sorted(group["design_idx"].tolist())
        if design_indices != list(range(EXPECTED_DESIGNS_PER_PROTEIN)):
            raise ExactLiveBaselineError(
                f"{protein_id} design_idx must be 0..7 exactly, got {design_indices}"
            )
        sequences = [str(sequence) for sequence in group["sequence"]]
        if len(set(sequences)) != EXPECTED_DESIGNS_PER_PROTEIN:
            raise ExactLiveBaselineError(
                f"{protein_id} does not have 8/8 unique Gumbel sequences; an argmax/repeated "
                "facade is not the stochastic comparator"
            )
        expected_seeds = {idx: seed_base + idx for idx in range(EXPECTED_DESIGNS_PER_PROTEIN)}
        observed_seeds = dict(zip(group["design_idx"], group["seed"]))
        if observed_seeds != expected_seeds:
            raise ExactLiveBaselineError(
                f"{protein_id} design seeds do not match seed_base + design_idx: "
                f"{observed_seeds} != {expected_seeds}"
            )
        reference = str(cohort_lookup.loc[protein_id, "sequence"])
        expected_length = (
            _as_int(cohort_lookup.loc[protein_id, "sequence_length"], "cohort sequence_length")
            if "sequence_length" in cohort.columns
            else len(reference)
        )
        if len(reference) != expected_length:
            raise ExactLiveBaselineError(
                f"cohort reference length mismatch for {protein_id}: "
                f"len(sequence)={len(reference)} != sequence_length={expected_length}"
            )
        for sequence in sequences:
            if not sequence or sequence != sequence.upper() or set(sequence) - AA20:
                raise ExactLiveBaselineError(
                    f"{protein_id} carries a non-AA20 source sequence: {sequence!r}"
                )
            if len(sequence) != expected_length:
                raise ExactLiveBaselineError(
                    f"{protein_id} source design length {len(sequence)} != cohort length "
                    f"{expected_length}"
                )

    selected = selected.sort_values(["protein_id", "design_idx"], kind="mergesort").reset_index(
        drop=True
    )
    identity = {
        "schema_version": "gumbel-dplm-native-source/1",
        "mode": "c0_native_sampler",
        "sampling_strategy": "gumbel_argmax",
        "n_designs_per_protein": EXPECTED_DESIGNS_PER_PROTEIN,
        "source_batch_size": batch_size,
        "seed_base": seed_base,
        "temperature": temperature,
        "max_iter": max_iter,
        "source_generation_git_revision": manifest.get("git_sha"),
        "source_checkpoint_digest": checkpoint_digest,
        "source_parquet_sha256": source_parquet_digest,
        "cohort_parquet_sha256": cohort_digest,
        "source_run_config_sha256": source_run_config_digest,
        "source_manifest_sha256": source_manifest_digest,
        "n_source_rows": len(source),
        "n_selected_proteins": len(cohort_ids),
        "n_selected_rows": len(selected),
        "paths": {
            "source_parquet": str(source_path),
            "cohort_parquet": str(cohort_path),
            "source_run_config": str(config_path),
            "source_manifest": str(manifest_path),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_declared_by_run_config": config.get("checkpoint"),
        },
    }
    identity["identity_digest"] = _canonical_digest(identity)
    return selected, identity


def _cache_paths(cache_dir: Path, protein_id: str, sequence: str) -> tuple[Path, Path]:
    from inverse_folding.evaluation.esmfold_runner import cache_key

    key = cache_key(protein_id, sequence)
    return cache_dir / f"{key}.pdb", cache_dir / f"{key}.plddt"


def claim_exact_live_cache(cache_dir: Any, identity: Mapping[str, Any]) -> dict[str, Any]:
    """Claim or verify a refold cache under one exact live-runtime identity.

    The exclusive create closes the race between two workers both seeing an empty directory.  A
    losing worker re-reads and verifies the winner's identity before it may consume cache entries.
    """
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache / CACHE_IDENTITY_NAME
    normalized_identity = json.loads(_canonical_bytes(dict(identity)))
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity": normalized_identity,
        "identity_digest": _canonical_digest(normalized_identity),
    }
    raw = _canonical_bytes(payload)

    def verify_existing() -> dict[str, Any]:
        try:
            existing = json.loads(lock.read_text())
        except Exception as exc:  # noqa: BLE001
            raise ExactLiveBaselineError(f"cache identity sidecar is unreadable: {lock}: {exc}") from exc
        if existing != payload:
            raise ExactLiveBaselineError(
                "refold cache identity differs from this exact-live runtime; use a fresh cache "
                f"instead of mixing evaluators ({lock})"
            )
        return payload

    if lock.exists():
        return verify_existing()
    foreign = sorted([*cache.glob("*.pdb"), *cache.glob("*.plddt")])
    if foreign:
        raise ExactLiveBaselineError(
            "refold cache contains unclaimed structure files; they may have been produced by a "
            f"foreign runtime: {[path.name for path in foreign[:5]]}"
        )
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return verify_existing()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        lock.unlink(missing_ok=True)
        raise
    return payload


def _head_score_payload(score: Any) -> dict[str, Any]:
    windows = sorted(
        (
            {
                "start_0b": int(window.start_0b),
                "end_0b": int(window.end_0b),
                "k": int(window.k),
                "z": float(window.z),
            }
            for window in score.windows
        ),
        key=lambda row: (row["start_0b"], row["end_0b"], row["k"]),
    )
    if not windows:
        raise ExactLiveBaselineError("Head returned an empty window grid")
    global_risk = None if score.global_risk is None else float(score.global_risk)
    if global_risk is not None and not math.isfinite(global_risk):
        raise ExactLiveBaselineError(f"Head returned non-finite global_risk {global_risk}")
    hotspot = getattr(score, "residue_hotspot", None)
    return {
        "protein_id": str(score.protein_id),
        "sequence_md5": str(score.sequence_md5),
        "sequence_length": int(score.sequence_length),
        "allele": str(score.allele),
        "score_scale": str(score.score_scale),
        "windows": windows,
        "residue_hotspot": None if hotspot is None else [float(value) for value in hotspot],
        "global_risk": global_risk,
    }


def evaluate_baseline_rows(
    source: pd.DataFrame,
    *,
    head_oracle: Any,
    structure_oracle: Any,
    refold_cache_dir: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score eight designs in one Head batch per protein and structure-evaluate every design."""
    required = {"protein_id", "design_idx", "sequence", "seed"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ExactLiveBaselineError(f"selected source frame is missing columns {missing}")
    cache = Path(refold_cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    head_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []

    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    ordered_source = source.sort_values(["protein_id", "design_idx"], kind="mergesort")
    for protein_id, group in ordered_source.groupby("protein_id", sort=True):
        group = group.sort_values("design_idx", kind="mergesort")
        if len(group) != EXPECTED_DESIGNS_PER_PROTEIN:
            raise ExactLiveBaselineError(
                f"{protein_id} reached evaluation with {len(group)} designs, expected 8"
            )
        requests = [
            OracleRequest(
                protein_id=str(protein_id),
                sequence=str(row.sequence),
                sequence_md5=sequence_md5(str(row.sequence)),
                sequence_length=len(str(row.sequence)),
            )
            for row in group.itertuples(index=False)
        ]
        # ONE call per protein is the batching contract. ProductionHeadOracle then invokes
        # OnlineHeadScorer.score_batch_same_protein once with all eight records.
        scores = list(head_oracle.score(requests))
        score_by_digest = {str(score.sequence_md5): score for score in scores}
        requested_digests = {request.sequence_md5 for request in requests}
        if len(scores) != EXPECTED_DESIGNS_PER_PROTEIN or set(score_by_digest) != requested_digests:
            raise ExactLiveBaselineError(
                f"Head result set for {protein_id} does not match its eight requested sequences: "
                f"returned={sorted(score_by_digest)}, requested={sorted(requested_digests)}"
            )

        for source_row, request in zip(group.itertuples(index=False), requests):
            score = score_by_digest[request.sequence_md5]
            if str(score.protein_id) != request.protein_id:
                raise ExactLiveBaselineError(
                    f"Head result protein {score.protein_id!r} != request {request.protein_id!r}"
                )
            evaluator = score.binding.evaluator
            evaluator_digest = str(evaluator.digest())
            source_row_payload = {
                "protein_id": request.protein_id,
                "design_idx": int(source_row.design_idx),
                "seed": int(source_row.seed),
                "sequence": request.sequence,
                "sequence_md5": request.sequence_md5,
            }
            source_row_digest = _canonical_digest(source_row_payload)
            head_rows.append(
                {
                    **source_row_payload,
                    "sequence_length": request.sequence_length,
                    "source_row_digest": source_row_digest,
                    "head_evaluator_digest": evaluator_digest,
                    "head_window_grid_digest": str(score.binding.window_grid_digest),
                    "head_global_risk": float(score.global_risk),
                    "head_score_json": _json(_head_score_payload(score)),
                }
            )

            pdb_path, plddt_path = _cache_paths(cache, request.protein_id, request.sequence)
            cache_hit_before_call = pdb_path.is_file() and plddt_path.is_file()
            started = dt.datetime.now(dt.timezone.utc)
            try:
                outcome = structure_oracle(request)
                evaluated = bool(getattr(outcome, "evaluated", False))
                feasible = bool(getattr(outcome, "feasible", False))
                metrics = dict(getattr(outcome, "metrics", None) or {})
                failure_reason = getattr(outcome, "failure_reason", None)
                walltime_s = float(getattr(outcome, "walltime_s", 0.0))
                oracle_cache_status = getattr(outcome, "cache_status", None)
                oracle_model_executed = getattr(outcome, "model_executed", None)
            except Exception as exc:  # noqa: BLE001 - retain the attempted row, then fail the run
                evaluated = False
                feasible = False
                metrics = {}
                failure_reason = f"{type(exc).__name__}: {exc}"
                walltime_s = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
                oracle_cache_status = None
                oracle_model_executed = None
            finite_metrics: dict[str, float] = {}
            for name, value in metrics.items():
                number = float(value)
                if not math.isfinite(number):
                    raise ExactLiveBaselineError(
                        f"structure metric {name!r} is non-finite for "
                        f"{request.protein_id}/{source_row.design_idx}: {value!r}"
                    )
                finite_metrics[str(name)] = number
            structure_rows.append(
                {
                    "protein_id": request.protein_id,
                    "design_idx": int(source_row.design_idx),
                    "seed": int(source_row.seed),
                    "sequence_md5": request.sequence_md5,
                    "source_row_digest": source_row_digest,
                    "structure_evaluated": evaluated,
                    "structure_feasible": feasible,
                    "structure_metrics_json": _json(finite_metrics),
                    "scTM": finite_metrics.get("scTM"),
                    "pLDDT": finite_metrics.get("pLDDT"),
                    "scRMSD": finite_metrics.get("scRMSD"),
                    "active_site_RMSD": finite_metrics.get("active_site_RMSD"),
                    "max_anchor_sidechain_RMSD": finite_metrics.get(
                        "max_anchor_sidechain_RMSD"
                    ),
                    "failure_reason": None if failure_reason is None else str(failure_reason),
                    "walltime_s": walltime_s,
                    # The production adapter currently reports every call as a miss/execution.
                    # These two fields are instead observed immediately before the call from the
                    # normalized cache files the live client itself consumes.
                    "cache_hit_before_call": cache_hit_before_call,
                    "model_executed_this_call": not cache_hit_before_call,
                    "oracle_cache_status": (
                        None if oracle_cache_status is None else str(oracle_cache_status)
                    ),
                    "oracle_model_executed": (
                        None if oracle_model_executed is None else bool(oracle_model_executed)
                    ),
                }
            )
    return head_rows, structure_rows


def _verify_local_model_snapshot(
    selector: Any,
    declared_sha256: Any,
) -> tuple[Path, Path, str]:
    """Resolve one local HF snapshot and bind the exact weight bytes the worker will open."""
    if not isinstance(selector, str) or not selector:
        raise ExactLiveBaselineError("structure runtime has no model_selector")
    selector_path = Path(selector).expanduser()
    if not selector_path.is_absolute():
        raise ExactLiveBaselineError(
            "structure model_selector must be an absolute local HF snapshot directory; "
            f"registry ids and relative selectors are refused, got {selector!r}"
        )
    snapshot_dir = _require_dir(selector_path, "local ESMFold2 snapshot directory")
    weights = _require_file(
        snapshot_dir / "model.safetensors",
        "local ESMFold2 model.safetensors",
    )
    if (
        not isinstance(declared_sha256, str)
        or len(declared_sha256) != 64
        or any(char not in "0123456789abcdef" for char in declared_sha256)
    ):
        raise ExactLiveBaselineError("structure runtime has no valid model snapshot SHA-256")
    observed_sha256 = _sha256_file(weights)
    if observed_sha256 != declared_sha256:
        raise ExactLiveBaselineError(
            "local ESMFold2 model.safetensors SHA-256 disagrees with "
            "local_model_snapshot_sha256: "
            f"observed {observed_sha256}, declared {declared_sha256}"
        )
    return snapshot_dir, weights, observed_sha256


def _load_structure_runtime(path: Any) -> tuple[dict[str, Any], str, Path]:
    runtime_path = _require_file(path, "V2 structure runtime identity")
    payload = _mapping(runtime_path, label="V2 structure runtime identity")
    if payload.get("schema_version") != "v2-esmfold2-runtime-1":
        raise ExactLiveBaselineError(
            "structure runtime identity must use schema v2-esmfold2-runtime-1"
        )
    if payload.get("backend") != "esmfold2_live":
        raise ExactLiveBaselineError(
            f"structure runtime backend must be esmfold2_live, got {payload.get('backend')!r}"
        )
    selector = payload.get("model_selector")
    snapshot = payload.get("local_model_snapshot_sha256")
    protocol = payload.get("protocol")
    snapshot_dir, weights, observed_snapshot_sha256 = _verify_local_model_snapshot(
        selector,
        snapshot,
    )
    if not isinstance(protocol, Mapping):
        raise ExactLiveBaselineError("structure runtime has no protocol mapping")
    normalized_protocol: dict[str, int] = {}
    for name, minimum in (
        ("num_loops", 1),
        ("num_sampling_steps", 1),
        ("num_diffusion_samples", 1),
        ("seed", 0),
    ):
        value = _as_int(protocol.get(name), f"structure protocol {name}")
        if value < minimum:
            raise ExactLiveBaselineError(
                f"structure protocol {name} must be >= {minimum}, got {value}"
            )
        normalized_protocol[name] = value
    # Downstream consumers receive the canonical absolute directory, never a registry alias or a
    # cwd-relative spelling. The resolved weight record makes the declaration -> bytes edge
    # explicit in both the cache claim and the final manifest.
    payload["model_selector"] = str(snapshot_dir)
    payload["resolved_model_safetensors"] = {
        "path": str(weights),
        "sha256": observed_snapshot_sha256,
    }
    from inverse_folding.evaluation.esmfold2_live import (
        ESMC_SNAPSHOT_REQUIRED_FILES,
        ESMFOLD2_SNAPSHOT_REQUIRED_FILES,
        hf_snapshot_identity,
    )

    model_snapshot = hf_snapshot_identity(snapshot_dir, ESMFOLD2_SNAPSHOT_REQUIRED_FILES)
    if payload.get("model_snapshot") != model_snapshot:
        raise ExactLiveBaselineError(
            "structure runtime ESMFold2 snapshot metadata/config identity disagrees with disk"
        )
    esmc_selector = payload.get("esmc_model_selector")
    if not isinstance(esmc_selector, str) or not Path(esmc_selector).is_absolute():
        raise ExactLiveBaselineError(
            "structure runtime requires an absolute local ESMC snapshot selector"
        )
    esmc_dir = _require_dir(esmc_selector, "local ESMC snapshot directory")
    esmc_snapshot = hf_snapshot_identity(esmc_dir, ESMC_SNAPSHOT_REQUIRED_FILES)
    if payload.get("esmc_snapshot") != esmc_snapshot:
        raise ExactLiveBaselineError("structure runtime ESMC snapshot identity disagrees with disk")
    ccd_path = _require_file(payload.get("ccd_path"), "local ESMFold2 CCD file")
    if _sha256_file(ccd_path) != payload.get("ccd_sha256"):
        raise ExactLiveBaselineError("structure runtime CCD SHA-256 disagrees with disk")
    payload["esmc_model_selector"] = str(esmc_dir)
    payload["ccd_path"] = str(ccd_path)
    payload["protocol"] = normalized_protocol
    return payload, _sha256_file(runtime_path), runtime_path


def _build_exact_live_oracles(
    args: argparse.Namespace,
    runtime_declaration: Mapping[str, Any],
    *,
    production_builder: Any | None = None,
    v0_builder: Any | None = None,
):
    """Build the existing production pair and capture the live worker metadata it realized."""
    if production_builder is None:
        from scripts.rf_fusion_v2_oracles import build_production_oracles as production_builder
    if v0_builder is None:
        from scripts.run_rf_refine_fusion import build_oracles as v0_builder

    realized: dict[str, Any] = {}

    def capturing_builder(v0_args: Any, config: Any):
        result = v0_builder(v0_args, config)
        metadata = getattr(v0_args, "_refold_runtime", None)
        if not isinstance(metadata, Mapping) or not metadata:
            raise ExactLiveBaselineError(
                "esmfold2_live worker exposed no runtime metadata; exact-runtime provenance "
                "cannot be established"
            )
        realized.update(dict(metadata))
        return result

    protocol = dict(runtime_declaration["protocol"])
    head, structure = production_builder(
        structure_config=args.structure_config,
        head_config_dir=args.head_config_dir,
        head_checkpoint=args.head_checkpoint,
        test_set_parquet=args.test_set_parquet,
        pdb_root=args.pdb_root,
        refold_cache_dir=args.refold_cache_dir,
        allele=args.allele,
        score_scale=args.score_scale,
        window_k_min=args.window_k_min,
        window_k_max=args.window_k_max,
        head_variant_id=args.head_variant_id,
        head_allele_idx=args.head_allele_idx,
        head_window_batch_size=64,
        constraint_manifest=args.constraint_manifest,
        device=args.device,
        esmfold2={
            "esmfold2_site_packages": args.esmfold2_site_packages,
            "esmfold2_model": runtime_declaration["model_selector"],
            "esmfold2_esmc_model": runtime_declaration["esmc_model_selector"],
            "esmfold2_ccd_path": runtime_declaration["ccd_path"],
            "esmfold2_num_loops": protocol["num_loops"],
            "esmfold2_num_sampling_steps": protocol["num_sampling_steps"],
            "esmfold2_num_diffusion_samples": protocol["num_diffusion_samples"],
            "esmfold2_seed": protocol["seed"],
        },
        build_oracles=capturing_builder,
        bind_extended_head_identity=True,
    )
    from scripts.rf_fusion_v2_oracles import (
        V2OracleError,
        verify_esmfold2_worker_runtime,
    )

    try:
        verify_esmfold2_worker_runtime(runtime_declaration, realized)
    except V2OracleError as exc:
        raise ExactLiveBaselineError(str(exc)) from exc
    return head, structure, dict(realized)


def _git_identity(project_root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "revision": revision,
        "worktree_dirty": bool(dirty),
        "script_sha256": _sha256_file(Path(__file__)),
    }


_GENERATED_COLUMNS = ("protein_id", "design_idx", "sequence", "seed", "sequence_md5")
_HEAD_COLUMNS = (
    "protein_id", "design_idx", "seed", "sequence", "sequence_md5", "sequence_length",
    "source_row_digest", "head_evaluator_digest", "head_window_grid_digest",
    "head_global_risk", "head_score_json",
)
_STRUCTURE_COLUMNS = (
    "protein_id", "design_idx", "seed", "sequence_md5", "source_row_digest",
    "structure_evaluated", "structure_feasible", "structure_metrics_json", "scTM", "pLDDT",
    "scRMSD", "active_site_RMSD", "max_anchor_sidechain_RMSD", "failure_reason",
    "walltime_s", "cache_hit_before_call", "model_executed_this_call", "oracle_cache_status",
    "oracle_model_executed",
)


def _write_parquet_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
    types: Mapping[str, str],
) -> None:
    from scripts.rf_fusion_v1_artifacts import write_stable_parquet

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write_stable_parquet(
            temporary,
            list(rows),
            columns=columns,
            sort_by=("protein_id", "design_idx"),
            types=types,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bundle(
    out_dir: Path,
    *,
    selected: pd.DataFrame,
    head_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from scripts.rf_fusion_v1_artifacts import write_manifest

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "generated": out_dir / "generated.parquet",
        "head": out_dir / "head_evaluations.parquet",
        "structure": out_dir / "structure_evaluations.parquet",
        "manifest": out_dir / "manifest.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise ExactLiveBaselineError(
            f"output artifacts already exist; pass --overwrite only for an intentional exact "
            f"rerun: {existing}"
        )

    generated_rows = [
        {
            "protein_id": str(row.protein_id),
            "design_idx": int(row.design_idx),
            "sequence": str(row.sequence),
            "seed": int(row.seed),
            "sequence_md5": sequence_md5(str(row.sequence)),
        }
        for row in selected.itertuples(index=False)
    ]
    common_types = {"design_idx": "int", "seed": "uint"}
    _write_parquet_atomic(
        paths["generated"],
        generated_rows,
        columns=_GENERATED_COLUMNS,
        types=common_types,
    )
    _write_parquet_atomic(
        paths["head"],
        head_rows,
        columns=_HEAD_COLUMNS,
        types={
            **common_types,
            "sequence_length": "int",
            "head_global_risk": "float",
        },
    )
    _write_parquet_atomic(
        paths["structure"],
        structure_rows,
        columns=_STRUCTURE_COLUMNS,
        types={
            **common_types,
            "structure_evaluated": "bool",
            "structure_feasible": "bool",
            "scTM": "float",
            "pLDDT": "float",
            "scRMSD": "float",
            "active_site_RMSD": "float",
            "max_anchor_sidechain_RMSD": "float",
            "walltime_s": "float",
            "cache_hit_before_call": "bool",
            "model_executed_this_call": "bool",
            "oracle_model_executed": "bool",
        },
    )
    manifest["artifacts"] = {
        name: {"path": path.name, "sha256": _sha256_file(path)}
        for name, path in paths.items()
        if name != "manifest"
    }
    manifest["manifest_payload_digest"] = _canonical_digest(manifest)
    write_manifest(paths["manifest"], manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", required=True)
    parser.add_argument("--source-run-config", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--expected-source-parquet-sha256", required=True)
    parser.add_argument("--expected-source-run-config-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-source-checkpoint-sha256", required=True)
    parser.add_argument("--expected-cohort-parquet-sha256", required=True)
    parser.add_argument("--cohort-parquet", required=True)
    parser.add_argument("--expected-proteins", required=True, type=int)
    parser.add_argument("--head-config-dir", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--head-variant-id", required=True)
    parser.add_argument("--head-allele-idx", type=int, default=0)
    parser.add_argument("--allele", required=True)
    parser.add_argument("--score-scale", required=True)
    parser.add_argument("--window-k-min", required=True, type=int)
    parser.add_argument("--window-k-max", required=True, type=int)
    parser.add_argument("--test-set-parquet", required=True)
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--structure-config", required=True)
    parser.add_argument("--structure-runtime-json", required=True)
    parser.add_argument("--constraint-manifest")
    parser.add_argument("--refold-cache-dir", required=True)
    parser.add_argument("--esmfold2-site-packages", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _preflight(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    selected, source_identity = load_gumbel_baseline(
        source_parquet=args.source_parquet,
        cohort_parquet=args.cohort_parquet,
        source_run_config=args.source_run_config,
        source_manifest=args.source_manifest,
        source_checkpoint=args.source_checkpoint,
        expected_source_parquet_sha256=args.expected_source_parquet_sha256,
        expected_source_run_config_sha256=args.expected_source_run_config_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_source_checkpoint_sha256=args.expected_source_checkpoint_sha256,
        expected_cohort_parquet_sha256=args.expected_cohort_parquet_sha256,
    )
    n_proteins = int(selected["protein_id"].nunique())
    if args.expected_proteins < 1 or n_proteins != args.expected_proteins:
        raise ExactLiveBaselineError(
            f"selected cohort has {n_proteins} proteins but --expected-proteins declares "
            f"{args.expected_proteins}"
        )
    expected_rows = args.expected_proteins * EXPECTED_DESIGNS_PER_PROTEIN
    if len(selected) != expected_rows:
        raise ExactLiveBaselineError(
            f"selected comparator has {len(selected)} rows; expected {expected_rows}"
        )
    _require_dir(args.head_config_dir, "Head config directory")
    _require_file(args.head_checkpoint, "Head checkpoint")
    _require_file(args.test_set_parquet, "structure test-set parquet")
    _require_dir(args.pdb_root, "structure PDB root")
    _require_file(args.structure_config, "v0 structure gate config")
    _require_dir(args.esmfold2_site_packages, "ESMFold2 package overlay")
    if args.constraint_manifest is not None:
        _require_file(args.constraint_manifest, "constraint manifest")
    if args.window_k_min < 1 or args.window_k_max < args.window_k_min:
        raise ExactLiveBaselineError("invalid Head window range")

    from inverse_folding.reference_flow.fusion.config import load_fusion_config

    structure_config = load_fusion_config(str(args.structure_config))
    if structure_config.structure.backend != "esmfold2_live":
        raise ExactLiveBaselineError(
            "the supplied structure config is not an esmfold2_live V2 gate: "
            f"{structure_config.structure.backend!r}"
        )
    runtime_declaration, runtime_digest, _runtime_path = _load_structure_runtime(
        args.structure_runtime_json
    )
    from inverse_folding.evaluation.esmfold2_live import esmfold2_overlay_identity

    observed_overlay = esmfold2_overlay_identity(args.esmfold2_site_packages)
    if runtime_declaration.get("site_packages_overlay") != observed_overlay:
        raise ExactLiveBaselineError(
            "structure runtime Biohub overlay identity disagrees with --esmfold2-site-packages"
        )
    from scripts.run_rf_fusion_v1_entry import (
        cohort_structure_digest,
        load_cohort_rows,
        validate_hard_anchors,
    )

    requested = sorted(selected["protein_id"].unique().tolist())
    try:
        backbone_rows = load_cohort_rows(
            args.test_set_parquet,
            requested,
            pdb_root=args.pdb_root,
        )
        anchor_count = (
            0
            if args.constraint_manifest is None
            else validate_hard_anchors(args.constraint_manifest, backbone_rows)
        )
    except ValueError as exc:
        raise ExactLiveBaselineError(f"structure cohort preflight failed: {exc}") from exc
    backbone_identity = {
        "cohort_structure_digest": cohort_structure_digest(backbone_rows),
        "n_resolved_backbones": len(backbone_rows),
        "n_hard_anchors": anchor_count,
        "rows": [
            {
                "protein_id": row.protein_id,
                "structure_source": row.structure_source,
                "structure_digest": row.structure_digest,
            }
            for row in backbone_rows
        ],
    }
    return (
        selected,
        source_identity,
        runtime_declaration,
        runtime_digest,
        backbone_identity,
    )


def main(argv: list[str] | None = None, *, oracles_factory: Any | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        (
            selected,
            source_identity,
            runtime_declaration,
            runtime_declaration_digest,
            backbone_identity,
        ) = _preflight(args)
        preflight_payload = {
            "schema_version": SCHEMA_VERSION,
            "source_identity": source_identity,
            "head_batches": int(selected["protein_id"].nunique()),
            "designs_per_head_batch": EXPECTED_DESIGNS_PER_PROTEIN,
            "head_window_batch_size": 64,
            "live_structure_requests": len(selected),
            "structure_runtime_declaration": runtime_declaration,
            "structure_runtime_declaration_sha256": runtime_declaration_digest,
            "backbone_identity": backbone_identity,
        }
        if args.dry_run:
            print(_json({"status": "DRY_RUN_OK", **preflight_payload}))
            return 0

        # Refuse a visibly foreign cache before paying for either model. The full identity, which
        # includes worker metadata, is claimed after the worker starts.
        cache = Path(args.refold_cache_dir).expanduser().resolve()
        cache_lock = cache / CACHE_IDENTITY_NAME
        if not cache_lock.exists() and cache.exists():
            foreign = [*cache.glob("*.pdb"), *cache.glob("*.plddt")]
            if foreign:
                raise ExactLiveBaselineError(
                    "refold cache contains unclaimed structure files; use a fresh exact-live cache"
                )

        oracle_factory = oracles_factory or _build_exact_live_oracles
        head_oracle, structure_oracle, worker_runtime = oracle_factory(
            args, runtime_declaration
        )
        head_identity = head_oracle.evaluator_identity()
        if _sha256_file(args.head_checkpoint) != head_identity.head_checkpoint_digest:
            raise ExactLiveBaselineError(
                "Head evaluator identity does not match the supplied checkpoint bytes"
            )
        structure_identity = {
            "schema_version": "rf-fusion-v2-live-structure-instrument/1",
            "runtime_declaration": runtime_declaration,
            "runtime_declaration_sha256": runtime_declaration_digest,
            "worker_runtime": worker_runtime,
            "esmfold2_site_packages": str(Path(args.esmfold2_site_packages).resolve()),
            "device": args.device,
            "structure_gate_config_sha256": _sha256_file(args.structure_config),
        }
        structure_identity["identity_digest"] = _canonical_digest(structure_identity)
        cache_claim = claim_exact_live_cache(args.refold_cache_dir, structure_identity)

        head_rows, structure_rows = evaluate_baseline_rows(
            selected,
            head_oracle=head_oracle,
            structure_oracle=structure_oracle,
            refold_cache_dir=args.refold_cache_dir,
        )
        measured = sum(
            bool(row["structure_evaluated"]) and row["scTM"] is not None
            for row in structure_rows
        )
        status = "complete" if measured == len(selected) else "partial_structure"
        project_root = Path(__file__).resolve().parents[1]
        manifest = {
            **preflight_payload,
            "status": status,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "code": _git_identity(project_root),
            "head": {
                "identity": head_identity.canonical_payload(),
                "identity_digest": head_identity.digest(),
                "head_window_batch_size": 64,
                "head_variant_id": args.head_variant_id,
                "head_allele_idx": args.head_allele_idx,
                "head_checkpoint_path": str(Path(args.head_checkpoint).resolve()),
                "head_config_dir": str(Path(args.head_config_dir).resolve()),
            },
            "structure": structure_identity,
            "cache_claim": cache_claim,
            "inputs": {
                "test_set_parquet": {
                    "path": str(Path(args.test_set_parquet).resolve()),
                    "sha256": _sha256_file(args.test_set_parquet),
                },
                "pdb_root": str(Path(args.pdb_root).resolve()),
                "cohort_structure_digest": backbone_identity["cohort_structure_digest"],
                "structure_runtime_json": {
                    "path": str(Path(args.structure_runtime_json).resolve()),
                    "sha256": runtime_declaration_digest,
                },
                "structure_config": {
                    "path": str(Path(args.structure_config).resolve()),
                    "sha256": _sha256_file(args.structure_config),
                },
                "constraint_manifest": (
                    None
                    if args.constraint_manifest is None
                    else {
                        "path": str(Path(args.constraint_manifest).resolve()),
                        "sha256": _sha256_file(args.constraint_manifest),
                    }
                ),
            },
            "counts": {
                "proteins": int(selected["protein_id"].nunique()),
                "designs": len(selected),
                "head_results": len(head_rows),
                "structure_requests": len(structure_rows),
                "structure_measured": measured,
                "structure_feasible": sum(bool(row["structure_feasible"]) for row in structure_rows),
                "structure_cache_hits": sum(bool(row["cache_hit_before_call"]) for row in structure_rows),
            },
        }
        _write_bundle(
            Path(args.out_dir).expanduser().resolve(),
            selected=selected,
            head_rows=head_rows,
            structure_rows=structure_rows,
            manifest=manifest,
            overwrite=args.overwrite,
        )
        print(_json({"status": status, "counts": manifest["counts"]}))
        return 0 if status == "complete" else 3
    except ExactLiveBaselineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
