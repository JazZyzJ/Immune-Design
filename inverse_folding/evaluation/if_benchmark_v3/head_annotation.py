"""Fixed-epoch full-data Head annotation after benchmark membership freezes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd


AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _validate_head_source_files(
    *, checkpoint_path: Path, resolved_config_path: Path,
    run_summary_path: Path, model_config_dir: Path,
) -> tuple[str, str, str, dict[str, str]]:
    """Validate and return the frozen DRB1501 epoch-24 source identities."""

    import torch
    import yaml

    checkpoint_path = Path(checkpoint_path).resolve()
    resolved_config_path = Path(resolved_config_path).resolve()
    run_summary_path = Path(run_summary_path).resolve()
    model_config_dir = Path(model_config_dir).resolve()
    model_paths = [
        model_config_dir / name
        for name in ("model.yaml", "model_ablation.yaml", "inference.yaml")
    ]
    if not all(path.is_file() for path in (
        checkpoint_path, resolved_config_path, run_summary_path, *model_paths,
    )):
        raise FileNotFoundError("Head checkpoint/config inputs are incomplete")
    if checkpoint_path.name == "best.pt":
        raise ValueError("full-data Head best.pt is forbidden")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
    resolved = yaml.safe_load(resolved_config_path.read_text())
    summary = json.loads(run_summary_path.read_text())
    if (
        not isinstance(metadata, dict) or int(metadata.get("epoch", -1)) != 24
        or int(metadata.get("global_step", -1)) != 10055
        or str(metadata.get("config_hash") or "") != "ad6ac404027b"
        or str(metadata.get("manifest_version") or "") != "v1.1"
        or not isinstance(resolved, dict)
        or resolved.get("ablation_encoder_id") != "LC1"
        or resolved.get("manifest_version") != "v1.1"
        or int(summary.get("final_epoch", -1)) != 24
        or int(summary.get("global_steps", -1)) != 10055
        or summary.get("config_hash") != "ad6ac404027b"
        or summary.get("manifest_version") != "v1.1"
    ):
        raise ValueError("Head inputs do not prove the fixed DRB1501 epoch24 identity")
    return (
        _sha(checkpoint_path), _sha(resolved_config_path), _sha(run_summary_path),
        {path.name: _sha(path) for path in model_paths},
    )


def _path_identity(path: Path, *, root: Path) -> dict[str, Any]:
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        rendered = str(path.relative_to(root))
    except ValueError:
        rendered = str(path)
    return {"path": rendered, "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def annotate_fixed_epoch_head(
    *, cohort: pd.DataFrame, checkpoint_path: Path, resolved_config_path: Path,
    run_summary_path: Path, model_config_dir: Path, output_path: Path,
    manifest_path: Path, allele: str = "HLA-DRB1*15:01", device: str = "cpu",
    window_batch_size: int = 1024,
    predictor_factory: Callable[..., Any] | None = None,
    cohort_input_paths: Sequence[Path] | None = None,
) -> pd.DataFrame:
    """Score immutable release rows without changing membership or row order."""

    required = {"dataset_release_id", "protein_id", "sequence", "sequence_sha256"}
    if (
        not required <= set(cohort) or cohort["protein_id"].isna().any()
        or cohort["protein_id"].duplicated().any() or allele != "HLA-DRB1*15:01"
        or int(window_batch_size) < 1
    ):
        raise ValueError("Head annotation cohort/allele contract is invalid")
    release_ids = set(cohort["dataset_release_id"].astype(str))
    if len(release_ids) != 1 or not next(iter(release_ids)):
        raise ValueError("Head annotation requires one frozen dataset release ID")
    checkpoint_path = Path(checkpoint_path).resolve()
    resolved_config_path = Path(resolved_config_path).resolve()
    run_summary_path = Path(run_summary_path).resolve()
    model_config_dir = Path(model_config_dir).resolve()
    checkpoint_sha, config_sha, summary_sha, model_config_hashes = _validate_head_source_files(
        checkpoint_path=checkpoint_path, resolved_config_path=resolved_config_path,
        run_summary_path=run_summary_path, model_config_dir=model_config_dir,
    )
    input_paths = [Path(path).resolve() for path in (cohort_input_paths or [])]
    if input_paths:
        if not all(path.is_file() for path in input_paths):
            raise FileNotFoundError("Head annotation cohort input table is missing")
        source_cohort = pd.concat(
            [pd.read_parquet(path) for path in input_paths], ignore_index=True,
        )
        try:
            pd.testing.assert_frame_equal(
                cohort.reset_index(drop=True), source_cohort.reset_index(drop=True),
                check_dtype=False,
            )
        except AssertionError as exc:
            raise ValueError("Head annotation cohort differs from persisted input tables") from exc
    records: list[tuple[str, str]] = []
    for row in cohort.to_dict("records"):
        sequence = str(row["sequence"])
        if (
            not sequence or not set(sequence) <= AA20
            or hashlib.sha256(sequence.encode()).hexdigest() != row["sequence_sha256"]
        ):
            raise ValueError("Head annotation row sequence identity mismatch")
        records.append((str(row["protein_id"]), sequence))
    if predictor_factory is None:
        from scripts.evaluate_phase_c import build_head_predictor
        predictor_factory = build_head_predictor
    predictor = predictor_factory(
        checkpoint_path=checkpoint_path, config_dir=model_config_dir,
        variant_id="LC1", device=device,
    )
    if hasattr(predictor, "predict_proteins"):
        predictions = predictor.predict_proteins(
            records, allele_idx=0, window_batch_size=int(window_batch_size)
        )
    else:
        predictions = [
            {"protein_id": protein_id,
             "prediction": predictor.predict_protein(
                 sequence, allele_idx=0, window_batch_size=int(window_batch_size)
             )}
            for protein_id, sequence in records
        ]
    if [str(item.get("protein_id")) for item in predictions] != [item[0] for item in records]:
        raise ValueError("Head predictor output order/key set differs from frozen cohort")
    release_id = next(iter(release_ids))
    sequence_sha_by_id = dict(zip(
        cohort["protein_id"].astype(str), cohort["sequence_sha256"].astype(str), strict=True
    ))
    output_rows = []
    for item in predictions:
        protein_id = str(item["protein_id"])
        prediction = item.get("prediction")
        risk = float(prediction.get("global_risk")) if isinstance(prediction, dict) else math.nan
        if not math.isfinite(risk):
            raise ValueError("Head predictor returned non-finite global_risk")
        output_rows.append({
            "dataset_release_id": release_id, "protein_id": protein_id,
            "sequence_sha256": sequence_sha_by_id[protein_id],
            "head_checkpoint_sha256": checkpoint_sha,
            "head_config_sha256": config_sha, "global_risk": risk,
            "status": "complete",
        })
    output = pd.DataFrame(output_rows)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    if output_path.resolve().parent != manifest_path.resolve().parent:
        raise ValueError("Head annotation output and manifest must share one stage directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    manifest = {
        "schema_version": "if-benchmark-v3-head-annotation/1", "allele": allele,
        "fixed_epoch": 24, "checkpoint_sha256": checkpoint_sha,
        "resolved_config_sha256": config_sha,
        "run_summary_sha256": summary_sha,
        "model_config_files": model_config_hashes,
        "cohort_inputs": [
            _path_identity(path, root=manifest_path.parent) for path in input_paths
        ],
        "device": device, "window_batch_size": int(window_batch_size),
        "row_count": len(output), "output_path": output_path.name,
        "output_sha256": _sha(output_path),
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def validate_head_annotation(
    *, manifest_path: Path, checkpoint_path: Path, resolved_config_path: Path,
    run_summary_path: Path, model_config_dir: Path,
    cohort_input_paths: Sequence[Path], replay: bool = False,
    predictor_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Rehash a Head annotation and optionally rerun the fixed predictor."""

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    unsigned = dict(manifest)
    digest = unsigned.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version") != "if-benchmark-v3-head-annotation/1"
        or digest != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or int(manifest.get("fixed_epoch", -1)) != 24
        or manifest.get("allele") != "HLA-DRB1*15:01"
        or not isinstance(manifest.get("cohort_inputs"), list)
    ):
        raise ValueError("Head annotation manifest identity mismatch")
    checkpoint_sha, config_sha, summary_sha, model_hashes = _validate_head_source_files(
        checkpoint_path=checkpoint_path, resolved_config_path=resolved_config_path,
        run_summary_path=run_summary_path, model_config_dir=model_config_dir,
    )
    if (
        manifest.get("checkpoint_sha256") != checkpoint_sha
        or manifest.get("resolved_config_sha256") != config_sha
        or manifest.get("run_summary_sha256") != summary_sha
        or manifest.get("model_config_files") != model_hashes
    ):
        raise ValueError("Head annotation model source identity mismatch")
    input_paths = [Path(path).resolve() for path in cohort_input_paths]
    expected_identities = [_path_identity(path, root=root) for path in input_paths]
    if manifest["cohort_inputs"] != expected_identities or not input_paths:
        raise ValueError("Head annotation cohort source identity mismatch")
    cohort = pd.concat([pd.read_parquet(path) for path in input_paths], ignore_index=True)
    output_raw = Path(str(manifest.get("output_path") or ""))
    output_path = (root / output_raw).resolve()
    if (
        output_raw.is_absolute() or ".." in output_raw.parts or output_path.parent != root
        or not output_path.is_file() or _sha(output_path) != manifest.get("output_sha256")
    ):
        raise ValueError("Head annotation output file identity mismatch")
    output = pd.read_parquet(output_path)
    if len(output) != int(manifest.get("row_count", -1)) or len(output) != len(cohort):
        raise ValueError("Head annotation output row count mismatch")
    expected_keys = cohort[["dataset_release_id", "protein_id", "sequence_sha256"]].copy()
    observed_keys = output[["dataset_release_id", "protein_id", "sequence_sha256"]].copy()
    try:
        pd.testing.assert_frame_equal(observed_keys, expected_keys, check_dtype=False)
    except AssertionError as exc:
        raise ValueError("Head annotation output keys differ from frozen cohort") from exc
    if (
        not all(math.isfinite(float(value)) for value in output["global_risk"])
        or set(output["head_checkpoint_sha256"].astype(str)) != {checkpoint_sha}
        or set(output["head_config_sha256"].astype(str)) != {config_sha}
        or set(output["status"].astype(str)) != {"complete"}
    ):
        raise ValueError("Head annotation output value/model identity mismatch")
    if replay:
        if predictor_factory is None:
            from scripts.evaluate_phase_c import build_head_predictor
            predictor_factory = build_head_predictor
        predictor = predictor_factory(
            checkpoint_path=Path(checkpoint_path).resolve(),
            config_dir=Path(model_config_dir).resolve(), variant_id="LC1",
            device=str(manifest["device"]),
        )
        records = list(zip(
            cohort["protein_id"].astype(str), cohort["sequence"].astype(str), strict=True
        ))
        predictions = predictor.predict_proteins(
            records, allele_idx=0,
            window_batch_size=int(manifest["window_batch_size"]),
        )
        replay_rows = [
            (str(item["protein_id"]), float(item["prediction"]["global_risk"]))
            for item in predictions
        ]
        persisted = list(zip(
            output["protein_id"].astype(str), output["global_risk"].astype(float), strict=True
        ))
        if replay_rows != persisted:
            raise ValueError("Head predictor replay differs from persisted annotation")
    return manifest
