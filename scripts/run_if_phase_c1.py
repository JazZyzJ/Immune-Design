#!/usr/bin/env python
"""Phase C1 driver: sampling-only position-dependent DFM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import asdict, dataclass

from inverse_folding.evaluation.h_maps import load_h_maps, sequence_md5
from inverse_folding.reference_flow import (
    PositionDependentDFMSampler,
    load_reference_flow_config,
    normalize_h_values,
    reference_flow_config_to_dict,
    with_reference_flow_overrides,
)
from inverse_folding.reference_flow.controller import (
    D1MonitorController,
    D1RefreshRecord,
)
from inverse_folding.reference_flow.controller_config import (
    ControllerConfig,
    controller_config_hash,
    controller_config_to_dict,
    load_controller_config,
)
from inverse_folding.reference_flow.head_scoring import OnlineHeadScorer
from scripts.run_if_phase_c0 import _fmt_hms, write_phase_c_outputs


# ============================================================
# Phase D1 helpers (controller setup, telemetry writers,
# manifest provenance). Kept as module-level helpers so the
# unit tests can exercise them without booting DPLM.
# ============================================================


# PLAN_RF.md §"Telemetry migration" — controller_surface_version. D1
# manifests that lack this field are read as version 1; D2/D3 runs stamp 2 so
# downstream analyzers can dispatch on schema lineage without re-running.
CONTROLLER_SURFACE_VERSION = 2


def canonical_aa_token_ids(task: Any) -> tuple[int, ...]:
    """Resolve the 20 canonical amino-acid DPLM token ids from a loaded task.

    Used by the D-phase controller to restrict candidate support to sampleable
    AA tokens. Computed from ``task.alphabet.get_idx`` so the ordering matches
    the DPLMTokenBridge convention shared with the IPA refiner.
    """
    # Local import to keep the script importable without dplm_refiner deps in
    # the unit-test environment that exercises the lightweight helpers.
    from inverse_folding.dplm_refiner.tokens import CANONICAL_AA_ORDER

    return tuple(int(task.alphabet.get_idx(aa)) for aa in CANONICAL_AA_ORDER)


@dataclass(frozen=True)
class ControllerSetup:
    """Resolved D-phase controller setup loaded from CLI + YAML."""

    config: ControllerConfig
    config_path: Path
    config_hash: str
    head_checkpoint: Path
    head_config_dir: Path
    head_variant_id: str
    head_device: str
    head_window_batch_size: int | None
    head_allele_idx: int
    allele: str


def _compute_head_config_hash(config_dir: Path) -> str:
    """SHA-256 over the three head config YAMLs in order (model/ablation/inference)."""
    h = hashlib.sha256()
    for fname in ("model.yaml", "model_ablation.yaml", "inference.yaml"):
        path = Path(config_dir) / fname
        if path.exists():
            h.update(path.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()


def load_controller_setup(args: argparse.Namespace) -> ControllerSetup | None:
    """Resolve --controller-config + head flags into a ControllerSetup.

    Returns None if the controller is not requested or if the loaded YAML has
    ``enabled=False``. Calls ``sys.exit`` (via parser.error semantics) when the
    controller is enabled but the required head flags are missing.
    """
    config_path = getattr(args, "controller_config", None)
    if not config_path:
        return None
    config = load_controller_config(config_path)
    if not config.enabled:
        return None

    missing = []
    if not getattr(args, "head_checkpoint", None):
        missing.append("--head-checkpoint")
    if not getattr(args, "head_config_dir", None):
        missing.append("--head-config-dir")
    if not getattr(args, "head_variant_id", None):
        missing.append("--head-variant-id")
    if missing:
        print(
            "ERROR: --controller-config enabled but required head flags missing: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        raise SystemExit(2)

    return ControllerSetup(
        config=config,
        config_path=Path(config_path).resolve(),
        config_hash=controller_config_hash(config),
        head_checkpoint=Path(args.head_checkpoint).resolve(),
        head_config_dir=Path(args.head_config_dir).resolve(),
        head_variant_id=str(args.head_variant_id),
        head_device=str(getattr(args, "head_device", "cpu")),
        head_window_batch_size=(
            int(args.head_window_batch_size)
            if getattr(args, "head_window_batch_size", None) is not None
            else None
        ),
        head_allele_idx=int(getattr(args, "head_allele_idx", 0)),
        allele=str(getattr(args, "allele", "")),
    )


def compute_per_protein_summary(
    *,
    protein_id: str,
    design_idx: int,
    seed: int,
    allele: str,
    arm: str,
    refresh_records: list[D1RefreshRecord],
    event_rows: list[dict],
    controller_config: ControllerConfig | None = None,
) -> dict[str, Any]:
    """Aggregate one design's controller telemetry into per_protein_summary fields.

    The ``new_hotspot_static_threshold`` field is read from the controller
    config's ``active_windows.excess_threshold`` (per D1 step #13), so any
    threshold sweep is correctly stamped into the per-protein summary
    instead of relying on a hard-coded value. When ``controller_config`` is
    omitted the threshold defaults to ``0.0`` for backward-compat with
    callers that have not been updated yet.

    D2/D3 fields are present but zero/null because D1 is monitor-only.
    """
    n_active_windows_per_refresh = [
        sum(1 for e in r.window_excess if e > 0.0) for r in refresh_records
    ]
    total_active_windows = sum(n_active_windows_per_refresh)
    total_new_hotspots = sum(int(r.new_hotspot_count) for r in refresh_records)
    new_hotspot_rate = (
        total_new_hotspots / float(total_active_windows)
        if total_active_windows > 0
        else 0.0
    )
    new_hotspot_static_threshold = (
        float(controller_config.active_windows.excess_threshold)
        if controller_config is not None
        else 0.0
    )

    d2_rows = [r for r in event_rows if r.get("event_type") == "D2"]
    d3_rows = [r for r in event_rows if r.get("event_type") == "D3"]

    # D2 candidate feasibility rate: fraction of refresh / active-block pairs
    # where at least one candidate met the feasibility gate. We approximate
    # using the D2 event rows we actually emit (rows are only emitted when
    # ``feasible=true`` because the controller skips block correction
    # otherwise). Total active blocks across refreshes is the denominator.
    d2_corrected_blocks_seen: set[tuple[int, int]] = set()
    for row in d2_rows:
        d2_corrected_blocks_seen.add((int(row["refresh_step"]), int(row["block_id"])))
    total_active_blocks = int(sum(len(r.active_blocks) for r in refresh_records))
    candidate_feasibility_rate = (
        float(len(d2_corrected_blocks_seen)) / float(total_active_blocks)
        if total_active_blocks > 0
        else 0.0
    )

    total_KL_budget = float(
        sum(
            float(r["kl_struct_corrected"])
            for r in d2_rows
            if r.get("kl_struct_corrected") is not None
        )
    )
    total_corrected_positions = len(d2_rows)
    total_recommits = len(d3_rows)

    # Churn rate: residues that were remasked more than once across the run.
    remask_counts: dict[int, int] = {}
    for row in d3_rows:
        pos = row.get("position_i")
        if pos is None:
            continue
        remask_counts[int(pos)] = remask_counts.get(int(pos), 0) + 1
    churned_positions = sum(1 for c in remask_counts.values() if c > 1)
    churn_rate = (
        float(churned_positions) / float(len(remask_counts))
        if remask_counts
        else 0.0
    )

    # Same-refresh conflict rate: D2-corrected positions that D3 immediately
    # remasked in the same refresh window. We use grace_flag=True on D3 rows
    # as the marker since the D3 path sets grace_flag for any position that
    # received a D2 correction in the current refresh.
    grace_remasks = sum(1 for r in d3_rows if r.get("grace_flag") is True)
    same_refresh_conflict_rate = (
        float(grace_remasks) / float(total_corrected_positions)
        if total_corrected_positions > 0
        else 0.0
    )

    return {
        "protein_id": protein_id,
        "design_idx": int(design_idx),
        "seed": int(seed),
        "allele": str(allele),
        "arm": str(arm),
        "n_refreshes": len(refresh_records),
        "n_active_blocks_total": total_active_blocks,
        "new_hotspot_rate": float(new_hotspot_rate),
        "new_hotspot_static_threshold": float(new_hotspot_static_threshold),
        "candidate_feasibility_rate": float(candidate_feasibility_rate),
        "total_D2_events": len(d2_rows),
        "total_D3_events": len(d3_rows),
        "total_corrected_positions": int(total_corrected_positions),
        "total_recommits": int(total_recommits),
        "total_KL_budget": float(total_KL_budget),
        # Productive-revisit rates require post-hoc head re-scoring of the
        # remasked positions; that pass is computed by D0 analysis tooling
        # against the generated.parquet + refresh_log, not here, so we leave
        # the fields nullable.
        "productive_revisit_immune_only": None,
        "productive_revisit_structure_only": None,
        "productive_revisit_joint": None,
        "churn_rate": float(churn_rate),
        "same_refresh_conflict_rate": float(same_refresh_conflict_rate),
        "head_regression_flag": None,
        "nmp_regression_flag": None,
        "structure_regression_flag": None,
        "active_block_mutation_count": None,
    }


def _refresh_record_to_jsonable(
    record: D1RefreshRecord, addendum: dict | None = None
) -> dict[str, Any]:
    """Serialize one refresh record plus any D2 / D3 controller addendum.

    Addendum keys covered:

    * D3 EMA: ``e_i`` / ``m_i`` / ``rho_i`` / ``grace_positions`` (null
      outside D3 modes).
    * D2 block diagnostics: ``d2_block_diagnostics`` with per-block
      candidate_count / candidate_feasibility / best_delta_R_B /
      mean_delta_R_B / ESS_B_candidates / g_ESS_candidates /
      rho_B_effective / corrected_positions / skipped_reason, plus the
      refresh-level ``delta_logit_max_refresh`` summary (PLAN_RF.md
      §"Telemetry migration" line 967).
    """
    payload = asdict(record)
    # asdict converts dataclass tuples to tuples; json dumps them as lists fine.
    payload["e_i"] = None
    payload["m_i"] = None
    payload["rho_i"] = None
    payload["grace_positions"] = None
    payload["d2_block_diagnostics"] = None
    payload["delta_logit_max_refresh"] = None
    if addendum:
        for key in (
            "e_i",
            "m_i",
            "rho_i",
            "grace_positions",
            "d2_block_diagnostics",
            "delta_logit_max_refresh",
        ):
            if key in addendum:
                payload[key] = addendum[key]
    return payload


def write_d1_artifacts(
    *,
    run_dir: Path,
    refresh_records_all: list[D1RefreshRecord],
    event_rows_all: list[dict],
    per_protein_summaries: list[dict],
    refresh_addenda_by_key: dict[tuple[str, int, int], dict] | None = None,
) -> None:
    """Write refresh_log.jsonl, controller_events.parquet, per_protein_summary.json.

    ``refresh_addenda_by_key`` maps ``(protein_id, design_idx, refresh_step)``
    to the D3 post_step addendum (``e_i``, ``m_i``, ``rho_i``,
    ``grace_positions``). When ``None`` every record gets the nullable
    placeholders, which is the D1 monitor-only schema.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    refresh_log = run_dir / "refresh_log.jsonl"
    with open(refresh_log, "w") as f:
        for record in refresh_records_all:
            addendum = None
            if refresh_addenda_by_key is not None:
                addendum = refresh_addenda_by_key.get(
                    (record.protein_id, int(record.design_idx), int(record.refresh_step))
                )
            f.write(
                json.dumps(
                    _refresh_record_to_jsonable(record, addendum=addendum),
                    default=str,
                )
            )
            f.write("\n")

    events_parquet = run_dir / "controller_events.parquet"
    if event_rows_all:
        pd.DataFrame(event_rows_all).to_parquet(events_parquet, index=False)
    else:
        # Touch an empty parquet so downstream tooling can rely on the file
        # existing in every D1 run directory. Schema mirrors the D0 minimum
        # event schema (PLAN_RF.md §D0) plus the D1 reliability factor extras
        # so a zero-event run is column-compatible with a non-empty run.
        pd.DataFrame(
            columns=[
                "protein_id", "design_idx", "seed", "refresh_step", "step", "t",
                "event_type", "block_id", "position_i", "window_start", "window_end",
                "a_before", "a_after", "a_uncorrected",
                "delta_R_corrected", "delta_R_uncorrected", "paired_disagreement_flag",
                "logit_struct", "logit_corrected", "delta_logit_max", "kl_struct_corrected",
                "delta_R_B", "delta_R_i", "ESS", "rho_B",
                "m_i", "commit_score", "remask_flag", "grace_flag", "reason",
                "g_time", "g_comp", "g_ent", "g_ESS",
            ]
        ).to_parquet(events_parquet, index=False)

    summary_json = run_dir / "per_protein_summary.json"
    with open(summary_json, "w") as f:
        json.dump({"summaries": per_protein_summaries}, f, indent=2, sort_keys=True)


def d1_manifest_provenance(
    setup: ControllerSetup,
    *,
    static_cache_path: Path | None,
    static_cache_meta_path: Path | None,
    window_k_min: int,
    window_k_max: int,
) -> dict[str, Any]:
    """Return manifest additions for a D1 run.

    Captures every value needed to detect controller / head config drift on a
    resumed run independently of the C1 reference-flow config.
    """
    head_config_hash = _compute_head_config_hash(setup.head_config_dir)
    head_checkpoint_digest = ""
    if setup.head_checkpoint.exists():
        h = hashlib.sha256()
        with open(setup.head_checkpoint, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        head_checkpoint_digest = h.hexdigest()
    return {
        "controller_mode": setup.config.mode,
        "controller_surface_version": CONTROLLER_SURFACE_VERSION,
        "controller_config": controller_config_to_dict(setup.config),
        "controller_config_hash": setup.config_hash,
        "controller_config_path": str(setup.config_path),
        # Top-level convenience views; the full nested config is already in
        # ``controller_config`` but downstream analyzers prefer flat accessors.
        "d2_config": controller_config_to_dict(setup.config)["d2"],
        "d3_config": controller_config_to_dict(setup.config)["d3"],
        "attribution_config": controller_config_to_dict(setup.config)["attribution"],
        "controls_config": controller_config_to_dict(setup.config)["controls"],
        "head_checkpoint_path": str(setup.head_checkpoint),
        "head_checkpoint_digest": head_checkpoint_digest,
        "head_variant_id": setup.head_variant_id,
        "head_config_dir": str(setup.head_config_dir),
        "head_config_hash": head_config_hash,
        "head_device": setup.head_device,
        "head_allele_idx": setup.head_allele_idx,
        "head_window_batch_size": setup.head_window_batch_size,
        "window_k_min": int(window_k_min),
        "window_k_max": int(window_k_max),
        "static_window_cache_path": (
            str(static_cache_path) if static_cache_path is not None else None
        ),
        "static_window_cache_meta_path": (
            str(static_cache_meta_path) if static_cache_meta_path is not None else None
        ),
    }


def build_head_scorer(
    setup: ControllerSetup,
    *,
    window_k_min: int,
    window_k_max: int,
    static_cache_path: Path | None = None,
    static_cache_meta_path: Path | None = None,
    static_cache_policy: str = "lazy_write",
) -> OnlineHeadScorer:
    """Construct the production InferencePredictor and wrap it in OnlineHeadScorer.

    Called ONCE per process (when the controller is enabled). The head model is
    reused across all proteins and designs.
    """
    # Local import keeps the script importable without the head deps in unit tests.
    from epitope_head.configs import (
        load_ablation_config,
        load_inference_config,
        load_model_config,
    )
    from scripts.infer_v1 import build_predictor

    predictor = build_predictor(
        model_cfg=load_model_config(setup.head_config_dir / "model.yaml"),
        ablation_cfg=load_ablation_config(setup.head_config_dir / "model_ablation.yaml"),
        inference_cfg=load_inference_config(setup.head_config_dir / "inference.yaml"),
        variant_id=setup.head_variant_id,
        checkpoint_path=setup.head_checkpoint,
        device=setup.head_device,
    )
    return OnlineHeadScorer(
        predictor=predictor,
        allele=setup.allele,
        allele_idx=setup.head_allele_idx,
        head_checkpoint_digest=hashlib.sha256(
            setup.head_checkpoint.read_bytes()
        ).hexdigest() if setup.head_checkpoint.exists() else "",
        head_config_hash=_compute_head_config_hash(setup.head_config_dir),
        score_scale=setup.config.head.score_scale,
        window_k_min=int(window_k_min),
        window_k_max=int(window_k_max),
        static_cache_path=static_cache_path,
        static_cache_meta_path=static_cache_meta_path,
        static_cache_policy=static_cache_policy,
        window_batch_size=setup.head_window_batch_size,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase C1: sampling-only position-dependent DFM on the IF test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-set-parquet", required=True)
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--h-maps-parquet", required=True)
    parser.add_argument("--allele", required=True)
    parser.add_argument("--config", required=True, help="Reference-flow YAML config.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--h-corpus-stats", default=None)
    parser.add_argument("--n-designs-per-protein", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--fail-pct-threshold", type=float, default=0.05)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--flag-name", default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Emit a per-protein progress line every N proteins (0 to disable).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty run_dir (default: refuse unless resuming).",
    )

    # Phase D controller wiring (optional). When --controller-config is supplied
    # and the YAML has enabled=true, the D1 monitor-only controller is layered
    # on top of the C1 sampler without changing logits, schedule, or remask.
    parser.add_argument(
        "--controller-config",
        default=None,
        help="Phase D controller YAML (e.g. inverse_folding/reference_flow/configs/d1_monitor.yaml).",
    )
    parser.add_argument("--head-checkpoint", default=None)
    parser.add_argument(
        "--head-config-dir",
        default=None,
        help="Directory containing model.yaml / model_ablation.yaml / inference.yaml.",
    )
    parser.add_argument("--head-variant-id", default=None)
    parser.add_argument("--head-device", default="cpu")
    parser.add_argument("--head-window-batch-size", type=int, default=None)
    parser.add_argument("--head-allele-idx", type=int, default=0)

    from inverse_folding.observability import add_wandb_cli_args

    add_wandb_cli_args(parser, default_project="mhc-if-phase-c-sampling")

    args = parser.parse_args(argv)

    if args.fail_pct_threshold < 0.0 or args.fail_pct_threshold > 1.0:
        parser.error("--fail-pct-threshold must be in [0, 1]")
    if args.n_steps is not None and args.n_steps <= 0:
        parser.error("--n-steps must be positive")
    if args.n_designs_per_protein is not None and args.n_designs_per_protein <= 0:
        parser.error("--n-designs-per-protein must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args


def print_resolved_hyperparams(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    run_dir: Path,
    n_entries: int,
) -> None:
    resolved = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_set_parquet": str(Path(args.test_set_parquet).resolve()),
        "pdb_root": str(Path(args.pdb_root).resolve()),
        "h_maps_parquet": str(Path(args.h_maps_parquet).resolve()),
        "h_corpus_stats": str(Path(args.h_corpus_stats).resolve()) if args.h_corpus_stats else None,
        "allele": args.allele,
        "config": str(Path(args.config).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "run_dir": str(run_dir),
        "device": args.device,
        "save_trajectories": args.save_trajectories,
        "fail_pct_threshold": args.fail_pct_threshold,
        "resume_from": args.resume_from,
        "progress_every": args.progress_every,
        "overwrite": args.overwrite,
        "n_input_proteins": n_entries,
        "resolved_config": config,
    }
    print("============================================================")
    print("Phase C1 resolved parameters")
    for key, value in resolved.items():
        print(f"  {key}: {value}")
    print("============================================================")


def _build_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return str(args.run_id)
    from inverse_folding.reference_flow.runtime import safe_allele_tag

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_stem = Path(args.config).stem
    # strip a redundant leading "c1_" from the config name so the run_id stays
    # single-prefixed (e.g. "c1_null.yaml" → "c1_null_DRB1_07_01_<stamp>",
    # not "c1_c1_null_DRB1_07_01_<stamp>")
    if config_stem.startswith("c1_"):
        config_stem = config_stem[len("c1_"):]
    return f"c1_{config_stem}_{safe_allele_tag(args.allele)}_{args.flag_name}_{stamp}"


def _select_h_values(
    h_row: pd.Series,
    *,
    h_source: str,
    corpus_stats: dict[str, Any] | None,
) -> Any:
    if h_source == "h_raw":
        return h_row["h_raw"]
    if h_source == "h_processed":
        return h_row["h_processed"]
    if corpus_stats is None:
        raise ValueError("h_corpus_stats is required when h_source == h_normalized_corpus")
    return normalize_h_values(h_row["h_raw"], corpus_stats)


def _load_corpus_stats(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path) as f:
        payload = json.load(f)
    required = {"corpus_h_raw_mean", "corpus_h_raw_std", "corpus_h_raw_n_residues"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"h-corpus-stats missing required keys: {sorted(missing)}")
    return payload


def _should_abort_failures(
    failures: list[dict[str, Any]],
    *,
    n_total_designs: int,
    threshold: float,
) -> bool:
    if n_total_designs <= 0:
        return False
    return (len(failures) / float(n_total_designs)) > threshold


def _partial_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "generated": run_dir / "generated.partial.parquet",
        "run_config": run_dir / "run_config.partial.yaml",
        "manifest": run_dir / "manifest.partial.json",
        "failures": run_dir / "failures.partial.json",
    }


def _resolve_resume_dir(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if not args.resume_from:
        return None
    candidate = Path(args.resume_from)
    if candidate.exists():
        return candidate
    return run_dir.parent / str(args.resume_from)


def _load_resume_state(
    *,
    resume_dir: Path,
    current_signature: dict[str, Any],
    entries: pd.DataFrame,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    generated_path = resume_dir / "generated.partial.parquet"
    if not generated_path.exists():
        generated_path = resume_dir / "generated.parquet"
    manifest_path = resume_dir / "manifest.partial.json"
    if not manifest_path.exists():
        manifest_path = resume_dir / "manifest.json"
    failures_path = resume_dir / "failures.partial.json"
    if not failures_path.exists():
        failures_path = resume_dir / "failures.json"

    if not generated_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"resume directory {resume_dir} must contain generated(.partial).parquet "
            "and manifest(.partial).json"
        )

    with open(manifest_path) as f:
        manifest = json.load(f)
    signature = manifest.get("resume_signature")
    if signature != current_signature:
        raise ValueError(
            "resume signature mismatch; checkpoint/config/test-set/h-map context changed"
        )

    df = pd.read_parquet(generated_path)
    valid_proteins = set(entries["protein_id"].astype(str))
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in df.iterrows():
        protein_id = str(row["protein_id"])
        design_idx = int(row["design_idx"])
        if protein_id not in valid_proteins:
            raise ValueError(f"resume parquet contains protein_id outside current input: {protein_id}")
        rows_by_key[(protein_id, design_idx)] = row.to_dict()

    failures: list[dict[str, Any]] = []
    if failures_path.exists():
        with open(failures_path) as f:
            payload = json.load(f)
        failures = list(payload.get("failures", []))
    return rows_by_key, failures


def _write_partial_state(
    *,
    run_dir: Path,
    rows_by_key: dict[tuple[str, int], dict[str, Any]],
    failures: list[dict[str, Any]],
    run_config: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    partial = _partial_paths(run_dir)
    partial["generated"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_by_key.values()).to_parquet(partial["generated"], index=False)
    with open(partial["run_config"], "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)
    with open(partial["manifest"], "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with open(partial["failures"], "w") as f:
        json.dump({"failures": failures}, f, indent=2, sort_keys=True)


def _write_trajectories(
    *,
    run_dir: Path,
    protein_id: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    from inverse_folding.reference_flow.runtime import safe_file_id

    trajectory_dir = run_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        trajectory_dir / f"{safe_file_id(protein_id)}.parquet",
        index=False,
    )


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _emit_progress(
    *,
    entry_idx: int,
    total_entries: int,
    protein_id: str,
    n_rows: int,
    n_failures: int,
    total_designs: int,
    run_start: float,
) -> None:
    elapsed = time.time() - run_start
    done = n_rows + n_failures
    if done > 0 and total_designs > done:
        eta = elapsed * (total_designs - done) / float(done)
    else:
        eta = 0.0
    avg = elapsed / float(done) if done > 0 else 0.0
    print(
        f"[progress] {entry_idx}/{total_entries} proteins "
        f"protein_id={protein_id} rows={n_rows} failures={n_failures} "
        f"elapsed={_fmt_hms(elapsed)} avg_per_design={avg:.2f}s eta={_fmt_hms(eta)}",
        flush=True,
    )


def derive_h_shuffle_seed(base_seed: int, protein_id: str, design_idx: int) -> int:
    """Derive a deterministic permutation seed for one (protein, design) pair."""
    payload = f"{int(base_seed)}::{protein_id}::{int(design_idx)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def assert_h_maps_align_with_test_set(
    *,
    entries: pd.DataFrame,
    h_maps_df: pd.DataFrame,
) -> None:
    """Fail-fast if the test parquet and h-maps disagree on (protein_id, sequence).

    Length checks only catch a subset of misalignment: same protein_id with
    same residue count but a different residue ordering (e.g. a chain re-cut
    or a different chain stretch) would silently produce position-wise wrong
    h-values. We bind each h-map row to the exact sequence it was scored on
    by comparing md5 digests stored in the parquet against the test set.
    """
    h_md5_by_id = {
        str(row["protein_id"]): str(row["sequence_md5"])
        for _, row in h_maps_df.iterrows()
    }
    mismatches: list[tuple[str, str, str]] = []
    missing: list[str] = []
    for _, entry in entries.iterrows():
        protein_id = str(entry["protein_id"])
        expected = sequence_md5(str(entry["sequence"]))
        actual = h_md5_by_id.get(protein_id)
        if actual is None:
            missing.append(protein_id)
            continue
        if actual != expected:
            mismatches.append((protein_id, expected, actual))
    if missing:
        raise ValueError(
            "h-maps parquet missing protein_id values present in test set "
            f"(first 5: {missing[:5]}; total {len(missing)})"
        )
    if mismatches:
        first = mismatches[0]
        raise ValueError(
            "h-maps sequence_md5 does not match test-set sequence for "
            f"{len(mismatches)} protein(s); first mismatch: protein_id={first[0]} "
            f"expected={first[1]} found={first[2]}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from inverse_folding.dplm_refiner.tokens import CANONICAL_AA_ORDER
    from inverse_folding.reference_flow.runtime import (
        build_dplm_denoiser_context,
        checkpoint_digest,
        decode_residue_tokens,
        git_sha,
        load_if_task,
        load_test_entries,
        make_dplm_denoiser,
        prepare_backbone,
        safe_allele_tag,
        utc_timestamp,
        write_json,
    )

    run_id = _build_run_id(args)
    run_dir = Path(args.output_root) / safe_allele_tag(args.allele) / run_id
    if (
        run_dir.exists()
        and any(run_dir.iterdir())
        and not args.resume_from
        and not args.overwrite
    ):
        print(
            f"ERROR: run_dir {run_dir} already exists and is non-empty; "
            "pass --resume-from <run_id> to resume, --overwrite to clobber, "
            "or choose a distinct --run-id / config stem.",
            file=sys.stderr,
        )
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_reference_flow_config(args.config)
    config = with_reference_flow_overrides(
        config,
        n_steps=args.n_steps,
        seed=args.seed,
        n_designs_per_protein=args.n_designs_per_protein,
    )
    config_dict = reference_flow_config_to_dict(config)

    entries = load_test_entries(args.test_set_parquet)
    h_maps_df, h_maps_meta = load_h_maps(args.h_maps_parquet)
    assert_h_maps_align_with_test_set(entries=entries, h_maps_df=h_maps_df)
    h_map_rows = {str(row["protein_id"]): row for _, row in h_maps_df.iterrows()}
    corpus_stats = _load_corpus_stats(args.h_corpus_stats)
    if config.amplification.h_source == "h_normalized_corpus" and corpus_stats is None:
        raise ValueError("--h-corpus-stats is required when config amplification.h_source = h_normalized_corpus")

    print_resolved_hyperparams(
        args=args,
        config=config_dict,
        run_dir=run_dir,
        n_entries=len(entries),
    )

    from inverse_folding.observability import (
        finish_wandb,
        init_wandb_from_args,
        log_metrics,
        set_summary,
    )

    wandb_run = init_wandb_from_args(
        args,
        run_name=run_id,
        config={
            "stage": "phase_c_sampling",
            "arm": "c1_reference_flow",
            "allele": args.allele,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config_path": str(Path(args.config).resolve()),
            "h_maps_parquet": str(Path(args.h_maps_parquet).resolve()),
            "n_input_proteins": int(len(entries)),
            "device": args.device,
            "save_trajectories": args.save_trajectories,
            "fail_pct_threshold": args.fail_pct_threshold,
            "run_dir": str(run_dir),
            "reference_flow_config": config_dict,
        },
        extra_tags=[
            "c1",
            "reference_flow",
            args.allele,
            f"amp_form={config.amplification.form}",
            f"schedule={config.schedule.base_form}",
            f"h_source={config.amplification.h_source}",
        ],
    )

    task = load_if_task(args.checkpoint, device=args.device)
    cpu_task = None
    sampler = PositionDependentDFMSampler(
        mask_token_id=task.alphabet.mask_idx,
        vocab_size=len(task.alphabet),
    )

    # Phase D1 controller wiring. Built ONCE per process when enabled; reused
    # across all proteins and designs. Telemetry buffers accumulate across
    # designs and are flushed at the end of the run.
    controller_setup = load_controller_setup(args)
    head_scorer: OnlineHeadScorer | None = None
    static_cache_path: Path | None = None
    static_cache_meta_path: Path | None = None
    refresh_records_all: list[D1RefreshRecord] = []
    refresh_addenda_by_key: dict[tuple[str, int, int], dict] = {}
    event_rows_all: list[dict] = []
    per_protein_summaries: list[dict] = []
    window_k_min_resolved = 0
    window_k_max_resolved = 0
    if controller_setup is not None:
        # Print the fully resolved controller config at startup (PLAN
        # §"Planned file touchpoints" / CLAUDE.md feedback "print hyperparams").
        resolved_controller_cfg = controller_config_to_dict(controller_setup.config)
        print("[controller] resolved controller config:", flush=True)
        for top_key, top_val in resolved_controller_cfg.items():
            print(f"[controller]   {top_key}: {top_val}", flush=True)
        print(
            f"[controller] surface_version={CONTROLLER_SURFACE_VERSION} "
            f"mode={controller_setup.config.mode}",
            flush=True,
        )
    if controller_setup is not None and args.resume_from:
        # v1 policy: D1 resume is rejected. Partial telemetry alignment is
        # not supported because controller_config / head provenance / window
        # k-range / cache policy can drift across runs and the existing
        # refresh_log / controller_events / per_protein_summary artifacts
        # would silently mix two arms. See PLAN_RF.md D0 attribution layer.
        print(
            "ERROR: --controller-config + --resume-from is not supported in D1 v1; "
            "rerun fresh (or omit --controller-config to resume C1 only).",
            file=sys.stderr,
        )
        return 2
    if controller_setup is not None:
        static_cache_path = run_dir / "static_window_cache.parquet"
        static_cache_meta_path = run_dir / "static_window_cache.meta.json"
        # Resolve window range from the inference config so the cache key
        # matches the predictor's enumeration. Reading raw YAML avoids a hard
        # import of epitope_head.configs in unit tests.
        inf_yaml = controller_setup.head_config_dir / "inference.yaml"
        if inf_yaml.exists():
            with open(inf_yaml) as f:
                inf_payload = yaml.safe_load(f) or {}
            inf_section = inf_payload.get("inference", {}) if isinstance(inf_payload, dict) else {}
            window_k_min_resolved = int(inf_section.get("min_k", 12))
            window_k_max_resolved = int(inf_section.get("max_k", 25))
        head_scorer = build_head_scorer(
            controller_setup,
            window_k_min=window_k_min_resolved,
            window_k_max=window_k_max_resolved,
            static_cache_path=static_cache_path,
            static_cache_meta_path=static_cache_meta_path,
            static_cache_policy="lazy_write",
        )

    resume_signature = {
        "mode": "c1_reference_flow",
        "allele": args.allele,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_digest": checkpoint_digest(args.checkpoint),
        "test_set_parquet": str(Path(args.test_set_parquet).resolve()),
        "h_maps_parquet": str(Path(args.h_maps_parquet).resolve()),
        "config": config_dict,
    }
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    resume_dir = _resolve_resume_dir(args, run_dir)
    if resume_dir is not None:
        rows_by_key, failures = _load_resume_state(
            resume_dir=resume_dir,
            current_signature=resume_signature,
            entries=entries,
        )

    total_designs = len(entries) * config.sampler.n_designs_per_protein
    total_entries = int(len(entries))
    run_start = time.time()
    aborted = False

    for entry_idx, (_, entry) in enumerate(entries.iterrows(), start=1):
        protein_id = str(entry["protein_id"])
        h_row = h_map_rows.get(protein_id)
        if h_row is None:
            failures.append({"protein_id": protein_id, "reason": "missing_h_map"})
            if _should_abort_failures(
                failures,
                n_total_designs=total_designs,
                threshold=args.fail_pct_threshold,
            ):
                break
            continue

        sequence_length = int(entry["sequence_length"])
        h_values = _select_h_values(
            h_row,
            h_source=config.amplification.h_source,
            corpus_stats=corpus_stats,
        )
        if len(h_values) != sequence_length:
            print(
                f"ERROR: h length mismatch for protein_id={protein_id}: "
                f"{len(h_values)} != {sequence_length}",
                file=sys.stderr,
            )
            return 2

        trajectory_rows: list[dict[str, Any]] = []
        for design_idx in range(config.sampler.n_designs_per_protein):
            row_key = (protein_id, design_idx)
            if row_key in rows_by_key:
                continue

            design_seed = int(config.sampler.seed) + int(design_idx)
            used_task = task

            def _build_d1_controller(task_for_decode, seed_for_design):
                if controller_setup is None or head_scorer is None:
                    return None
                # D2/D3 candidate enumeration restricts to canonical AA token
                # IDs (excludes mask/pad/cls/eos/unk). For monitor_only mode
                # without nested D2/D3 sections the controller ignores this
                # tuple; for any d2.enabled config it is required.
                canonical_ids = canonical_aa_token_ids(task_for_decode)
                return D1MonitorController(
                    protein_id=protein_id,
                    design_idx=int(design_idx),
                    seed=int(seed_for_design),
                    static_sequence=str(entry["sequence"]),
                    scorer=head_scorer,
                    config=controller_setup.config,
                    decode_tokens=lambda toks: decode_residue_tokens(task_for_decode, toks),
                    canonical_token_ids=canonical_ids,
                )

            d1_controller = _build_d1_controller(task, design_seed)
            try:
                started = time.time()
                prepared = prepare_backbone(
                    task=task,
                    entry=entry,
                    pdb_root=args.pdb_root,
                    device=args.device,
                )
                context = build_dplm_denoiser_context(task=task, prepared=prepared)
                out = sampler.sample(
                    sequence_length=sequence_length,
                    h_values=h_values,
                    denoiser=make_dplm_denoiser(context),
                    config=replace(
                        config,
                        sampler=replace(config.sampler, seed=design_seed),
                    ),
                    save_trajectories=args.save_trajectories,
                    shuffle_seed=(
                        derive_h_shuffle_seed(
                            int(config.h_shuffle.seed),
                            protein_id,
                            design_idx,
                        )
                        if config.h_shuffle.enabled
                        else None
                    ),
                    controller=d1_controller,
                    protein_id=protein_id,
                    design_idx=int(design_idx),
                )
            except RuntimeError as exc:
                if _is_oom(exc) and str(args.device).startswith("cuda"):
                    if cpu_task is None:
                        cpu_task = load_if_task(args.checkpoint, device="cpu")
                    try:
                        started = time.time()
                        used_task = cpu_task
                        prepared = prepare_backbone(
                            task=cpu_task,
                            entry=entry,
                            pdb_root=args.pdb_root,
                            device="cpu",
                        )
                        context = build_dplm_denoiser_context(task=cpu_task, prepared=prepared)
                        d1_controller = _build_d1_controller(cpu_task, design_seed)
                        out = sampler.sample(
                            sequence_length=sequence_length,
                            h_values=h_values,
                            denoiser=make_dplm_denoiser(context),
                            config=replace(
                                config,
                                sampler=replace(config.sampler, seed=design_seed),
                            ),
                            save_trajectories=args.save_trajectories,
                            shuffle_seed=(
                                derive_h_shuffle_seed(
                                    int(config.h_shuffle.seed),
                                    protein_id,
                                    design_idx,
                                )
                                if config.h_shuffle.enabled
                                else None
                            ),
                            controller=d1_controller,
                            protein_id=protein_id,
                            design_idx=int(design_idx),
                        )
                    except Exception as retry_exc:  # noqa: BLE001
                        failures.append(
                            {
                                "protein_id": protein_id,
                                "design_idx": design_idx,
                                "reason": f"oom_cpu_retry_failed:{type(retry_exc).__name__}:{retry_exc}",
                            }
                        )
                        continue
                else:
                    failures.append(
                        {
                            "protein_id": protein_id,
                            "design_idx": design_idx,
                            "reason": f"runtime_failed:{type(exc).__name__}:{exc}",
                        }
                    )
                    continue
            except FloatingPointError as exc:
                failures.append(
                    {
                        "protein_id": protein_id,
                        "design_idx": design_idx,
                        "reason": f"nan_logits:{exc}",
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "protein_id": protein_id,
                        "design_idx": design_idx,
                        "reason": f"runtime_failed:{type(exc).__name__}:{exc}",
                    }
                )
                continue

            rows_by_key[row_key] = {
                "protein_id": protein_id,
                "design_idx": int(design_idx),
                "sequence": decode_residue_tokens(used_task, out.tokens),
                "seed": design_seed,
                "wall_seconds": time.time() - started,
            }
            if args.save_trajectories:
                for row in out.trajectory_rows:
                    trajectory_rows.append(
                        {
                            "protein_id": protein_id,
                            "design_idx": int(design_idx),
                            "seed": design_seed,
                            **row,
                        }
                    )
            if d1_controller is not None:
                design_refreshes = d1_controller.refresh_records()
                design_events = d1_controller.controller_event_rows()
                refresh_records_all.extend(design_refreshes)
                event_rows_all.extend(design_events)
                design_addenda = d1_controller.refresh_addenda()
                for refresh_step_key, addendum in design_addenda.items():
                    refresh_addenda_by_key[
                        (protein_id, int(design_idx), int(refresh_step_key))
                    ] = addendum
                per_protein_summaries.append(
                    compute_per_protein_summary(
                        protein_id=protein_id,
                        design_idx=int(design_idx),
                        seed=design_seed,
                        allele=args.allele,
                        arm=controller_setup.config.mode,
                        refresh_records=design_refreshes,
                        event_rows=design_events,
                        controller_config=controller_setup.config,
                    )
                )

        if args.save_trajectories:
            _write_trajectories(run_dir=run_dir, protein_id=protein_id, rows=trajectory_rows)

        if entry_idx % 25 == 0:
            partial_manifest = {
                "run_id": run_id,
                "mode": "c1_reference_flow",
                "timestamp": utc_timestamp(),
                "git_sha": git_sha(PROJECT_ROOT),
                "resume_signature": resume_signature,
                "n_rows_generated": len(rows_by_key),
                "n_failures": len(failures),
            }
            partial_config = {
                "mode": "c1_reference_flow",
                "resume_signature": resume_signature,
                "resolved_reference_flow_config": config_dict,
            }
            _write_partial_state(
                run_dir=run_dir,
                rows_by_key=rows_by_key,
                failures=failures,
                run_config=partial_config,
                manifest=partial_manifest,
            )

        if args.progress_every > 0 and (
            entry_idx % args.progress_every == 0 or entry_idx == total_entries
        ):
            _emit_progress(
                entry_idx=entry_idx,
                total_entries=total_entries,
                protein_id=protein_id,
                n_rows=len(rows_by_key),
                n_failures=len(failures),
                total_designs=total_designs,
                run_start=run_start,
            )
            elapsed = time.time() - run_start
            done = len(rows_by_key) + len(failures)
            avg = elapsed / float(done) if done > 0 else 0.0
            log_metrics(
                wandb_run,
                {
                    "progress/proteins_done": entry_idx,
                    "progress/rows_generated": len(rows_by_key),
                    "progress/failures": len(failures),
                    "progress/elapsed_seconds": elapsed,
                    "progress/avg_seconds_per_design": avg,
                },
                step=entry_idx,
            )

        if _should_abort_failures(
            failures,
            n_total_designs=total_designs,
            threshold=args.fail_pct_threshold,
        ):
            aborted = True
            print(
                f"[abort] failure rate exceeded threshold at entry_idx={entry_idx}: "
                f"{len(failures)}/{total_designs} > {args.fail_pct_threshold:.2%}; "
                "stopping main loop.",
                flush=True,
            )
            break

    if _should_abort_failures(
        failures,
        n_total_designs=total_designs,
        threshold=args.fail_pct_threshold,
    ):
        partial_manifest = {
            "run_id": run_id,
            "mode": "c1_reference_flow",
            "timestamp": utc_timestamp(),
            "git_sha": git_sha(PROJECT_ROOT),
            "resume_signature": resume_signature,
            "n_rows_generated": len(rows_by_key),
            "n_failures": len(failures),
        }
        partial_config = {
            "mode": "c1_reference_flow",
            "resume_signature": resume_signature,
            "resolved_reference_flow_config": config_dict,
        }
        _write_partial_state(
            run_dir=run_dir,
            rows_by_key=rows_by_key,
            failures=failures,
            run_config=partial_config,
            manifest=partial_manifest,
        )
        print(
            f"ERROR: failure rate exceeded threshold ({len(failures)}/{total_designs}); "
            f"partial artifacts retained in {run_dir}",
            file=sys.stderr,
        )
        return 2

    entry_order = {str(pid): idx for idx, pid in enumerate(entries["protein_id"].astype(str))}
    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (entry_order[str(row["protein_id"])], int(row["design_idx"])),
    )
    run_config = {
        "mode": "c1_reference_flow",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_set_parquet": str(Path(args.test_set_parquet).resolve()),
        "pdb_root": str(Path(args.pdb_root).resolve()),
        "h_maps_parquet": str(Path(args.h_maps_parquet).resolve()),
        "h_corpus_stats": str(Path(args.h_corpus_stats).resolve()) if args.h_corpus_stats else None,
        "allele": args.allele,
        "device": args.device,
        "resolved_reference_flow_config": config_dict,
    }
    manifest = {
        "run_id": run_id,
        "mode": "c1_reference_flow",
        "git_sha": git_sha(PROJECT_ROOT),
        "checkpoint_digest": checkpoint_digest(args.checkpoint),
        "timestamp": utc_timestamp(),
        "allele": args.allele,
        "h_maps_source_path": str(Path(args.h_maps_parquet).resolve()),
        "h_maps_source_run_id": h_maps_meta["run_id"],
        "h_source": config.amplification.h_source,
        "schedule_form": config.schedule.base_form,
        "amplification": config_dict["amplification"],
        "resume_signature": resume_signature,
        "n_input_proteins": int(len(entries)),
        "n_rows_generated": int(len(rows)),
        "n_failures": int(len(failures)),
        "failures_path": "failures.json" if failures else None,
        "wall_clock_seconds": float(time.time() - run_start),
        "aborted_on_failure_threshold": bool(aborted),
    }
    if controller_setup is not None and head_scorer is not None:
        manifest.update(
            d1_manifest_provenance(
                controller_setup,
                static_cache_path=static_cache_path,
                static_cache_meta_path=static_cache_meta_path,
                window_k_min=window_k_min_resolved,
                window_k_max=window_k_max_resolved,
            )
        )
        run_config["controller_config"] = controller_config_to_dict(controller_setup.config)
        run_config["controller_config_hash"] = controller_setup.config_hash
        run_config["controller_config_path"] = str(controller_setup.config_path)

    # D1 telemetry + static cache flush MUST happen before manifest write so
    # the manifest can only claim cache paths that actually exist on disk.
    # Failure to flush is treated as a hard error: refresh_log embeds
    # r_windows_static, but the cache is still the contract for cross-run
    # reuse; a manifest pointing at a missing cache would silently break D0
    # downstream tooling.
    if controller_setup is not None and head_scorer is not None:
        write_d1_artifacts(
            run_dir=run_dir,
            refresh_records_all=refresh_records_all,
            event_rows_all=event_rows_all,
            per_protein_summaries=per_protein_summaries,
            refresh_addenda_by_key=refresh_addenda_by_key,
        )
        head_scorer.flush_static_cache(
            source_dataset=str(Path(args.test_set_parquet).resolve()),
            source_dataset_rowcount=int(len(entries)),
        )

    write_phase_c_outputs(run_dir, rows, run_config, manifest)
    if failures:
        write_json(run_dir / "failures.json", {"failures": failures})

    total_wall = time.time() - run_start
    n_rows = int(len(rows))
    n_failures = int(len(failures))
    avg_per_design = (
        total_wall / float(n_rows + n_failures) if (n_rows + n_failures) > 0 else 0.0
    )
    failure_rate = (n_failures / float(total_designs)) if total_designs > 0 else 0.0

    print("============================================================")
    print(
        f"[done] run_id={run_id} "
        f"generated_rows={n_rows}/{int(total_designs)} "
        f"failures={n_failures} failure_rate={failure_rate:.2%} "
        f"wall={_fmt_hms(total_wall)} avg_per_design={avg_per_design:.2f}s "
        f"output_dir={run_dir}"
    )
    print("============================================================")

    set_summary(
        wandb_run,
        {
            "summary/n_input_proteins": int(len(entries)),
            "summary/n_total_designs": int(total_designs),
            "summary/n_rows_generated": n_rows,
            "summary/n_failures": n_failures,
            "summary/failure_rate": failure_rate,
            "summary/wall_seconds": float(total_wall),
            "summary/avg_seconds_per_design": float(avg_per_design),
            "summary/aborted_on_failure_threshold": bool(aborted),
        },
    )
    finish_wandb(wandb_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
