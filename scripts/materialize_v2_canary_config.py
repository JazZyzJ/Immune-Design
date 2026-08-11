"""Resolve ONE V2 Canary cell's config from the tracked template and the measured artifacts.

`doc/RF_Fusion_v2_Cluster_Runbook.md` §3 requires four resolved configs, one per
`(protein, r_step)` cell, materialized outside the repo and never by editing the tracked template.
Each carries eighteen PLAN §5.2 content identities, a threshold that must be copied VERBATIM out of
a calibration artifact, and a band digest that is deliberately NOT the band file's sha256.  Filling
that by hand is how a cell ends up signed under another cell's calibration, so it is done here and
checked here.

**The script emits the launch command with the config, and that is the point.**  A resolved config
declares which roles are frozen and which are observed at runtime; the driver's `--input-file` and
`--shard-input` lists must agree with that declaration or the run fails after the allocation is
paid.  Two hands maintaining the same fact is how they drift, so the producer that computes the
digests also writes the argument vectors that supply the files those digests came from
(`<out>.args.sh`, `source`-able from a SLURM script).

Everything is a CLI argument -- no cluster path appears in this module.

Three checks refuse rather than warn, because each is a way to produce a config that runs and is
wrong:

* the hotspot artifact's Head domain must equal the template's `(allele, score_scale, k_min,
  k_max)`.  `bind_admission_policy` compares that 4-tuple, and `N_H^whole` is a maximum over the
  grid it defines: a threshold measured under one domain does not bound designs scored under
  another.
* the band artifact must actually carry a cell at the requested `(r_step, stratum_key)`.
  `lookup_band` is exact and never interpolates, so a missing cell yields a typed null for every
  transition -- a fact about the band, reported after the GPU time, that is knowable here.
* no `REPLACE_*` placeholder may survive.  The loader refuses them, but it refuses at launch; this
  refuses at materialization with the field named.

Registered in `doc/SCRIPTS.md` under the RF-Refine Fusion V2 section.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.structure_gate import (  # noqa: E402
    DUAL_SCTM_POLICY_KIND,
    HIGH_RISK_ANCESTRY_SCTM_MIN,
    HIGH_RISK_DUAL_SCTM_PROFILE_ID,
    HIGH_RISK_STRICT_SCTM_MIN,
)
from inverse_folding.reference_flow.fusion_v2.policy import (  # noqa: E402
    HEAD_DIRECTED_CAPPED_POLICY_ID,
)
from inverse_folding.reference_flow.fusion_v2.reward import (  # noqa: E402
    DEPTH0_BOOTSTRAP_RULE,
)

__all__ = [
    "build_parser", "main",
    "file_digest", "head_config_hash", "no_constraint_manifest_digest",
    "resolve_content_bindings", "fill_config",
]


# These are named, closed exploratory profiles rather than a bag of schedule flags.  Each is an
# unblinded capability run, not another way to edit the V2F5A qualification cell.  Phase/split,
# schedule, caps and any method-specific safety policy are therefore materialized together, while
# the separate runtime launch override remains required by the driver.
EXPLORATORY_PROFILES: dict[str, dict[str, Any]] = {
    "uricase_d4_k12_r40": {
        "phase": "capability_ladder",
        "split_role": "exploratory_uricase",
        "schedule": {
            "schedule_id": "exploratory-uricase-d4-k12-r40-v1",
            "coordinate_law": "progressive_checkpoint",
            "depth_cap": 4,
            "active_population_width": 1,
            "min_lookahead_tail_steps": 10,
            "points": [
                {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
                 "n_lookaheads": 12, "band_key": "step40"},
                {"depth": 1, "r_step": 40, "c_source_step": 60, "c_next_step": 70,
                 "n_lookaheads": 12, "band_key": "step40"},
                {"depth": 2, "r_step": 40, "c_source_step": 70, "c_next_step": 80,
                 "n_lookaheads": 12, "band_key": "step40"},
                {"depth": 3, "r_step": 40, "c_source_step": 80, "c_next_step": 90,
                 "n_lookaheads": 12, "band_key": "step40"},
            ],
        },
        # Exact plan: 1,990 logical DFE and 60 definitive endpoint refolds per protein.  These are
        # close fail-closed ceilings with small operational slack, not the template's broad cohort
        # qualification caps.  Head calls remain a CLI-bound quantity because the counterfactual
        # ceiling is calibrated from the realized cohort's editable domains.
        "caps": {
            "max_logical_dfe": 2200,
            "max_definitive_refolds": 64,
        },
    },
    "highrisk_d2_k32_r40": {
        "phase": "capability_ladder",
        "split_role": "exploratory_highrisk_ceiling_v1",
        "required_policy_version": "v2",
        "required_counterfactual_head_calls_per_cycle": 454,
        "required_run_max_head_calls": 1024,
        "require_absolute_esmfold2_snapshot": True,
        "schedule": {
            "schedule_id": "highrisk-d2-k32-r40-v1",
            "coordinate_law": "progressive_checkpoint",
            "depth_cap": 2,
            "active_population_width": 1,
            "min_lookahead_tail_steps": 10,
            "points": [
                {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 70,
                 "n_lookaheads": 32, "band_key": "step40"},
                {"depth": 1, "r_step": 40, "c_source_step": 70, "c_next_step": 90,
                 "n_lookaheads": 32, "band_key": "step40"},
            ],
        },
        # Exact upper bounds before slack: 3,010 logical DFE, 96 definitive refolds and 1,004
        # Head calls (96 endpoint scores + 2 * 454 counterfactual ceilings), per protein/root.
        "caps": {
            "max_logical_dfe": 3072,
            "max_head_calls": 1024,
            "max_definitive_refolds": 128,
            "max_gpu_seconds": 14400,
            "max_walltime_s": 14400,
            "max_retries": 2,
        },
        "search_structure": {
            "policy_kind": DUAL_SCTM_POLICY_KIND,
            "profile_id": HIGH_RISK_DUAL_SCTM_PROFILE_ID,
            "ancestry_sctm_min": HIGH_RISK_ANCESTRY_SCTM_MIN,
            "strict_sctm_min": HIGH_RISK_STRICT_SCTM_MIN,
        },
    },
    "highrisk_d8_k32_r40": {
        "phase": "capability_ladder",
        "split_role": "exploratory_highrisk_ceiling_v1",
        "required_policy_version": "v2",
        "required_counterfactual_head_calls_per_cycle": 454,
        "required_run_max_head_calls": 4096,
        "require_absolute_esmfold2_snapshot": True,
        "schedule": {
            "schedule_id": "highrisk-d8-k32-r40-v1",
            "coordinate_law": "progressive_checkpoint",
            "depth_cap": 8,
            "active_population_width": 1,
            "min_lookahead_tail_steps": 10,
            "points": [
                {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 55,
                 "n_lookaheads": 32, "band_key": "step40"},
                {"depth": 1, "r_step": 40, "c_source_step": 55, "c_next_step": 60,
                 "n_lookaheads": 32, "band_key": "step40"},
                {"depth": 2, "r_step": 40, "c_source_step": 60, "c_next_step": 65,
                 "n_lookaheads": 32, "band_key": "step40"},
                {"depth": 3, "r_step": 40, "c_source_step": 65, "c_next_step": 70,
                 "n_lookaheads": 32, "band_key": "step40"},
                {"depth": 4, "r_step": 40, "c_source_step": 70, "c_next_step": 75,
                 "n_lookaheads": 32, "band_key": "step40"},
                {"depth": 5, "r_step": 40, "c_source_step": 75, "c_next_step": 80,
                 "n_lookaheads": 32, "band_key": "step40"},
                {"depth": 6, "r_step": 40, "c_source_step": 80, "c_next_step": 85,
                 "n_lookaheads": 32, "band_key": "step40"},
                {"depth": 7, "r_step": 40, "c_source_step": 85, "c_next_step": 90,
                 "n_lookaheads": 32, "band_key": "step40"},
            ],
        },
        # Exact upper bounds before slack: 8,950 logical DFE, 288 definitive refolds and 3,920
        # Head calls (288 endpoint scores + 8 * 454 counterfactual ceilings), per protein/root.
        "caps": {
            "max_logical_dfe": 9216,
            "max_head_calls": 4096,
            "max_definitive_refolds": 320,
            "max_gpu_seconds": 14400,
            "max_walltime_s": 14400,
            "max_retries": 2,
        },
        "search_structure": {
            "policy_kind": DUAL_SCTM_POLICY_KIND,
            "profile_id": HIGH_RISK_DUAL_SCTM_PROFILE_ID,
            "ancestry_sctm_min": HIGH_RISK_ANCESTRY_SCTM_MIN,
            "strict_sctm_min": HIGH_RISK_STRICT_SCTM_MIN,
        },
    },
}


class MaterializeError(RuntimeError):
    """A resolved config this producer refuses to write."""


def file_digest(path: Any) -> str:
    """SHA-256 over a file's bytes -- the same function the driver signs declared inputs with."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _structure_runtime_protocol(
    args, *, profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the explicit ESMFold2 selector/protocol required by an exploratory run."""
    model = getattr(args, "esmfold2_model", None)
    if not isinstance(model, str) or not model.strip():
        raise MaterializeError(
            "--esmfold2-model is required by an exploratory profile; relying on v0's parser "
            "default would leave the executed structure model outside the run identity"
        )
    model_selector = model.strip()
    esmc_selector: str | None = None
    ccd_path: str | None = None
    if profile is not None and profile.get("require_absolute_esmfold2_snapshot"):
        selector = Path(model_selector)
        if not selector.is_absolute():
            raise MaterializeError(
                "this high-risk profile requires --esmfold2-model to be an absolute local "
                f"snapshot directory, got {model_selector!r}"
            )
        try:
            selector = selector.resolve(strict=True)
        except FileNotFoundError as exc:
            raise MaterializeError(
                f"absolute ESMFold2 snapshot does not exist: {selector}"
            ) from exc
        if not selector.is_dir():
            raise MaterializeError(
                f"absolute ESMFold2 snapshot selector must be a directory, got {selector}"
            )
        selector_weights = selector / "model.safetensors"
        if not selector_weights.is_file():
            raise MaterializeError(
                f"absolute ESMFold2 snapshot {selector} has no model.safetensors"
            )
        backend_raw = getattr(args, "structure_backend", None)
        backend = Path(backend_raw) if backend_raw is not None else None
        if backend is None or not backend.is_file():
            raise MaterializeError(
                "--structure-backend must be the model.safetensors file whose SHA is bound to "
                "the absolute ESMFold2 snapshot"
            )
        selector_sha = file_digest(selector_weights)
        backend_sha = file_digest(backend)
        if selector_sha != backend_sha:
            raise MaterializeError(
                f"absolute snapshot model.safetensors SHA {selector_sha} does not match "
                f"--structure-backend SHA {backend_sha}"
            )
        model_selector = str(selector)
        esmc_raw = getattr(args, "esmfold2_esmc_model", None)
        if not isinstance(esmc_raw, str) or not esmc_raw.strip():
            raise MaterializeError(
                "this high-risk profile requires --esmfold2-esmc-model to be an absolute local "
                "ESMC snapshot; ESMFold2 may not resolve its language model through a registry id"
            )
        esmc = Path(esmc_raw).expanduser()
        if not esmc.is_absolute() or not esmc.resolve().is_dir():
            raise MaterializeError(
                f"--esmfold2-esmc-model must be an existing absolute directory, got {esmc_raw!r}"
            )
        esmc_selector = str(esmc.resolve())
        ccd_raw = getattr(args, "esmfold2_ccd_path", None)
        if not isinstance(ccd_raw, str) or not ccd_raw.strip():
            raise MaterializeError(
                "this high-risk profile requires --esmfold2-ccd-path; CCD lookup may not fall "
                "back to a mutable Hugging Face cache"
            )
        ccd = Path(ccd_raw).expanduser()
        if not ccd.is_absolute() or not ccd.resolve().is_file():
            raise MaterializeError(
                f"--esmfold2-ccd-path must be an existing absolute file, got {ccd_raw!r}"
            )
        ccd_path = str(ccd.resolve())

    values: dict[str, int] = {}
    for field, flag, minimum in (
        ("num_loops", "--esmfold2-num-loops", 1),
        ("num_sampling_steps", "--esmfold2-num-sampling-steps", 1),
        ("num_diffusion_samples", "--esmfold2-num-diffusion-samples", 1),
        ("seed", "--esmfold2-seed", 0),
    ):
        value = getattr(args, f"esmfold2_{field}", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise MaterializeError(
                f"{flag} must be an integer >= {minimum} for an exploratory profile, got "
                f"{value!r}"
            )
        values[field] = int(value)
    resolved = {"model_selector": model_selector, "protocol": values}
    if esmc_selector is not None:
        resolved["esmc_model_selector"] = esmc_selector
        resolved["ccd_path"] = ccd_path
    return resolved


def _structure_runtime_bytes(payload: dict[str, Any]) -> bytes:
    """Canonical bytes signed by both the config content row and the run input signature."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _structure_runtime_identity(args) -> tuple[Path, dict[str, Any], str]:
    """Bind the realized ESMFold2 selector/protocol to the supplied local model snapshot.

    Legacy exploratory profiles may use a registry selector.  High-risk profiles instead require
    an absolute local snapshot directory and prove that its ``model.safetensors`` bytes match the
    separately supplied backend file before signing the selector and protocol together.
    """
    profile = EXPLORATORY_PROFILES.get(getattr(args, "exploratory_profile", None))
    declared = _structure_runtime_protocol(args, profile=profile)
    snapshot = Path(args.structure_backend)
    if not snapshot.is_file():
        raise MaterializeError(
            f"--structure-backend must be the local ESMFold2 model snapshot file, got {snapshot}"
        )
    payload = {
        "schema_version": "v2-esmfold2-runtime-1",
        "backend": "esmfold2_live",
        **declared,
        "local_model_snapshot_sha256": file_digest(snapshot),
    }
    if profile is not None and profile.get("require_absolute_esmfold2_snapshot"):
        from inverse_folding.evaluation.esmfold2_live import (
            ESMC_SNAPSHOT_REQUIRED_FILES,
            ESMFOLD2_SNAPSHOT_REQUIRED_FILES,
            esmfold2_overlay_identity,
            hf_snapshot_identity,
        )

        payload["site_packages_overlay"] = esmfold2_overlay_identity(
            args.esmfold2_site_packages)
        payload["model_snapshot"] = hf_snapshot_identity(
            declared["model_selector"], ESMFOLD2_SNAPSHOT_REQUIRED_FILES)
        payload["esmc_snapshot"] = hf_snapshot_identity(
            declared["esmc_model_selector"], ESMC_SNAPSHOT_REQUIRED_FILES)
        payload["ccd_sha256"] = file_digest(declared["ccd_path"])
    raw = _structure_runtime_bytes(payload)
    path = Path(args.out).with_suffix(".structure_runtime.json")
    return path, payload, hashlib.sha256(raw).hexdigest()


def head_config_hash(config_dir: Any) -> str:
    """The Head config directory's realized identity, read from the ONE implementation.

    `OnlineHeadScorer` reports this number as `head_config_hash`, and the V2 runtime compares the
    reference's evaluator identity against every endpoint's.  A second rolling hash here that
    agreed today would be a second implementation to drift tomorrow, so the producer imports v0's.
    """
    from scripts.run_if_phase_c1 import _compute_head_config_hash

    return _compute_head_config_hash(Path(config_dir))


def no_constraint_manifest_digest() -> str:
    """The project's canonical digest for "this cohort declares NO constraint manifest".

    An unconstrained protein has no manifest file, but PLAN §5.2 gives `constraint_manifest` no
    exemption: a run whose conditioning is simply silent about constraints cannot be told apart
    from one whose manifest went missing.  `rho_maturity_scan` already answered this -- it records
    a typed ABSENCE digest, and `5ZHV_B`'s own `B(r)` artifact carries it -- so the resolved config
    declares that same number rather than inventing a second convention or hashing a file written
    only to be hashed.  Being frozen, it also makes `--input-file constraint_manifest=...` refuse
    any manifest on an unconstrained cell.
    """
    from scripts.rho_maturity_scan import _canonical_digest

    return _canonical_digest({"schema": "scan-constraint-binding/1", "constraint_manifest": None})


def _band_content_digest(band_json: Any, *, r_step: int, stratum_key: str) -> str:
    """The band TABLE's `calibration_content_digest`, and proof the requested cell exists.

    Runbook §3 note 2: this role's `expected_sha256` is the table's canonical content digest, NOT
    the file's sha256.  `make_band_table` rebinds it and `q_phi` compares against it; the file
    digest would refuse every real run.
    """
    payload = json.loads(Path(band_json).read_text(encoding="utf-8"))
    provenance = payload.get("provenance") or {}
    digest = provenance.get("calibration_content_digest")
    if not digest:
        raise MaterializeError(f"{band_json} carries no provenance.calibration_content_digest")
    cells = [band for band in payload.get("bands") or ()
             if int(band.get("step", -1)) == int(r_step)
             and str(band.get("stratum_key")) == str(stratum_key)]
    if not cells:
        available = sorted({(int(b.get("step", -1)), str(b.get("stratum_key")))
                            for b in payload.get("bands") or ()})
        raise MaterializeError(
            f"{band_json} has no band at (step={r_step}, stratum_key={stratum_key!r}); it carries "
            f"{available}.  lookup_band is exact and never interpolates, so this cell would yield "
            "a typed null for every transition -- knowable now rather than after the GPU time"
        )
    return str(digest)


def _delta_new_block(
    hotspot_json: Any, *, head: dict, highrisk_protein_id: str | None = None,
    code_revision: str | None = None, master_seed: int | None = None,
    head_config_digest: str | None = None, head_checkpoint_digest: str | None = None,
) -> dict:
    """The `delta_new` block, copied VERBATIM, after checking it was measured on THIS Head domain."""
    payload = json.loads(Path(hotspot_json).read_text(encoding="utf-8"))
    block = payload.get("delta_new")
    if not isinstance(block, dict):
        raise MaterializeError(f"{hotspot_json} carries no delta_new block")
    artifact = block.get("artifact") or {}
    declared = (str(head["allele"]), str(head["score_scale"]),
                int(head["window_k_min"]), int(head["window_k_max"]))
    measured = (str(artifact.get("allele")), str(artifact.get("score_scale")),
                int(artifact.get("window_k_min", -1)), int(artifact.get("window_k_max", -1)))
    if declared != measured:
        names = ("allele", "score_scale", "window_k_min", "window_k_max")
        diff = ", ".join(f"{n}: config {d!r} != artifact {m!r}"
                         for n, d, m in zip(names, declared, measured) if d != m)
        raise MaterializeError(
            f"the hotspot threshold in {hotspot_json} was measured under a different Head domain "
            f"({diff}).  N_H^whole is a maximum over the grid these fields define, so a threshold "
            "measured under one domain does not bound designs scored under another"
        )
    if highrisk_protein_id is not None:
        from scripts.calibrate_rf_fusion_v2_hotspot import (
            SEED_NAMESPACE,
            THRESHOLD_STATISTIC_BY_POPULATION,
            source_id_for,
        )

        expected = {
            "protein_id": str(highrisk_protein_id),
            "code_revision": str(code_revision),
            "population": "full_trajectory",
            "sampling_unit": "trajectory",
            "threshold_statistic": THRESHOLD_STATISTIC_BY_POPULATION["full_trajectory"],
            "seed_namespace": SEED_NAMESPACE,
            "master_seed": int(master_seed),
            "min_head_valid": 64,
            "n_attempted": 64,
            "n_head_valid": 64,
            "n_retained": 64,
            "failure_counts": {},
            "structure_diagnostic_mode": "skip_independent_capability_ceiling",
        }
        mismatches = {
            name: {"observed": payload.get(name), "expected": value}
            for name, value in expected.items() if payload.get(name) != value
        }
        if mismatches:
            raise MaterializeError(
                f"high-risk hotspot artifact {hotspot_json} violates the frozen per-protein "
                f"full-trajectory n=64 law: {mismatches}"
            )
        from inverse_folding.reference_flow.fusion_v2.identity import HeadEvaluatorIdentity

        expected_head = HeadEvaluatorIdentity(
            allele=str(head["allele"]), score_scale=str(head["score_scale"]),
            window_k_min=int(head["window_k_min"]),
            window_k_max=int(head["window_k_max"]),
            head_config_hash=str(head_config_digest),
            head_checkpoint_digest=str(head_checkpoint_digest),
            head_variant_id=str(head["head_variant_id"]),
            head_allele_idx=int(head["head_allele_idx"]),
            head_window_batch_size=int(head["head_window_batch_size"]),
        ).canonical_payload()
        if payload.get("head") != expected_head:
            raise MaterializeError(
                "high-risk hotspot artifact was measured by a different complete Head evaluator "
                f"identity: observed {payload.get('head')!r}, expected {expected_head!r}"
            )
        expected_source_id = source_id_for(
            str(highrisk_protein_id), population="full_trajectory")
        if block.get("source_id") != expected_source_id:
            raise MaterializeError(
                f"high-risk hotspot source_id must be {expected_source_id!r}, got "
                f"{block.get('source_id')!r}"
            )
        if payload.get("q90_higher") != block.get("value"):
            raise MaterializeError(
                "high-risk hotspot top-level q90_higher does not equal the copied delta_new.value"
            )
        operability = payload.get("structure_operability")
        if (not isinstance(operability, dict)
                or operability.get("status") != "not_measured"
                or operability.get("n_definitive_feasible") is not None
                or operability.get("rate") is not None):
            raise MaterializeError(
                "high-risk hotspot structure_operability must explicitly be not_measured with "
                "null count/rate under skip_independent_capability_ceiling"
            )
    return json.loads(json.dumps(block))  # a plain, YAML-safe copy


def _projection_policy_method(policy_spec: Any) -> tuple[str, str]:
    """Return the policy version and its closed D0 rule from the exact supplied spec bytes."""
    path = Path(policy_spec)
    if not path.is_file():
        raise MaterializeError(f"projection policy spec does not exist or is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializeError(f"cannot read projection policy spec {path}: {exc}") from exc
    if payload.get("policy_id") != HEAD_DIRECTED_CAPPED_POLICY_ID:
        raise MaterializeError(
            f"{path} does not declare policy_id={HEAD_DIRECTED_CAPPED_POLICY_ID!r}"
        )
    version = payload.get("policy_version")
    depth0_rules = {
        "v1": "cumulative_safety_reference",
        "v2": DEPTH0_BOOTSTRAP_RULE,
    }
    if version not in depth0_rules:
        raise MaterializeError(
            f"{path} declares unsupported head_directed_capped policy_version {version!r}; "
            "only v1 and v2 can be materialized"
        )
    return str(version), depth0_rules[str(version)]


def resolve_content_bindings(
    args, *, structure_runtime_identity: tuple[Path, dict[str, Any], str] | None = None,
) -> tuple[dict[str, str | None], dict[str, str]]:
    """`role -> frozen digest or None`, plus `role -> the runtime file that supplies it`.

    A role appears in exactly one of the two.  The split is not cosmetic: a frozen role is bound
    before the run and a supplied file that contradicts it is refused, while a runtime role is
    signed by whatever bytes actually arrived.  Three roles here are frozen for a reason no file
    can express -- the Head config is a DIRECTORY rolling-hash, the band is a table digest rather
    than a file digest, and an unconstrained cell's constraint manifest is a typed absence.
    """
    constrained = bool(args.constraint_manifest)
    frozen: dict[str, str] = {
        "code_revision": str(args.code_revision),
        # THIS protein's canonical sequence file, not the manifest that resolves it.  A Canary cell
        # is one protein, and `bind_cumulative_reference` compares the digest it recomputes from
        # the reference BYTES against `policy.complete_reference_content_digest`, which is exactly
        # this field -- so a manifest digest here can never equal it and the shard dies with
        # `ReferenceRebindAttempt` after the checkpoints are resident.  The cohort-level manifest
        # is a resolution table and binds separately, as `reference_sequences`.
        "complete_reference_sequence": file_digest(args.reference_sequence),
        "projection_policy_spec": file_digest(args.projection_policy_spec),
        "schedule_band_calibration": _band_content_digest(
            args.band_json, r_step=int(args.r_step), stratum_key=str(args.stratum_key)),
        "head_config": head_config_hash(args.head_config_dir),
        "head_checkpoint": file_digest(args.head_checkpoint),
    }
    runtime: dict[str, str] = {
        "cohort_table": str(args.cohort_table),
        # Plural: the cohort's reference TABLE, which is what `complete_reference_manifest` reads.
        "reference_sequences": str(args.reference_manifest),
        "backbone": str(args.backbone),
        # The mask is DERIVED from these coordinates.  `_conditioning` replaces this digest with
        # the realized per-protein mask, so the file records where the mask came from.
        "coordinate_mask": str(args.backbone),
        "rf_sampler_config": str(args.rf_sampler_config),
        "dplm_checkpoint": str(args.dplm_checkpoint),
        # Likewise: the alphabet lives in the checkpoint, and `_conditioning` replaces this with
        # the realized tokenizer digest.
        "tokenizer": str(args.dplm_checkpoint),
        "structure_config": str(args.structure_config),
        "v0_structure_gate_config": str(args.v0_structure_gate_config),
    }
    if getattr(args, "exploratory_profile", None):
        identity = structure_runtime_identity or _structure_runtime_identity(args)
        # Unlike an ordinary Canary, the recursive sandbox refuses v0 defaults.  Its config digest
        # freezes a canonical artifact containing BOTH the local snapshot digest and the actual
        # selector/protocol handed to the worker.
        frozen["structure_backend"] = identity[2]
    else:
        # Existing D1 behavior: the supplied local weight file itself is the runtime-bound role.
        runtime["structure_backend"] = str(args.structure_backend)
    if constrained:
        runtime["constraint_manifest"] = str(args.constraint_manifest)
        runtime["fixed_token_policy"] = str(args.constraint_manifest)
    else:
        absence = no_constraint_manifest_digest()
        frozen["constraint_manifest"] = absence
        frozen["fixed_token_policy"] = absence
    return frozen, runtime


def fill_config(template: dict, *, args, frozen: dict, runtime: dict) -> dict:
    """Return the resolved config; refuse if any `REPLACE_*` placeholder would survive."""
    config = json.loads(json.dumps(template))  # deep copy without YAML aliases

    config["identity"]["code_revision"] = str(args.code_revision)
    # The campaign names the EXPERIMENT.  Two runs that share a template but answer different
    # questions -- the Canary's "does it execute", the mechanism cohort's "does it transmit" --
    # must not sign their artifacts under one campaign, or the second reads as more of the first.
    if getattr(args, "campaign_id", None):
        config["identity"]["campaign_id"] = str(args.campaign_id)

    profile_name = getattr(args, "exploratory_profile", None)
    profile = EXPLORATORY_PROFILES.get(profile_name) if profile_name else None
    if profile_name and profile is None:
        raise MaterializeError(
            f"unknown --exploratory-profile {profile_name!r}; allowed values are "
            f"{sorted(EXPLORATORY_PROFILES)}"
        )
    if profile is None and getattr(args, "run_max_head_calls", None) is not None:
        raise MaterializeError(
            "--run-max-head-calls requires an explicit --exploratory-profile; an ordinary "
            "Canary/qualification materialization may not silently consume or ignore an "
            "exploratory run cap"
        )
    if profile is not None:
        _structure_runtime_protocol(args, profile=profile)
        if int(args.r_step) != 40:
            raise MaterializeError(
                f"exploratory profile {profile_name!r} requires --r-step 40, got {args.r_step}; "
                "every rung is bound to the measured B(40) cell"
            )
        substrate = config.get("substrate") or {}
        required_substrate = {
            "n_steps": 100,
            "amplification_form": "constant_one",
            "controller_enabled": False,
            # Keep the post-step lifecycle but set the remask fraction to zero: this is the
            # project's frozen no-remask spelling, not remask_enabled=false.
            "remask_enabled": True,
            "remask_fraction_scale": 0.0,
        }
        mismatch = {
            key: (substrate.get(key), value)
            for key, value in required_substrate.items() if substrate.get(key) != value
        }
        if mismatch:
            raise MaterializeError(
                f"exploratory profile {profile_name!r} requires the frozen 100-step null/no-remask "
                f"substrate, but the template disagrees: {mismatch}"
            )
        if profile.get("search_structure") is not None:
            structure_config = getattr(args, "structure_config", None)
            gate_config = getattr(args, "v0_structure_gate_config", None)
            if not structure_config or not gate_config:
                raise MaterializeError(
                    "a high-risk profile requires both --structure-config and "
                    "--v0-structure-gate-config"
                )
            if file_digest(structure_config) != file_digest(gate_config):
                raise MaterializeError(
                    "high-risk structure-config and v0-structure-gate-config must be "
                    "byte-identical; search and definitive gates may not drift"
                )
            if not isinstance(getattr(args, "stratum_key", None), str) or not args.stratum_key:
                raise MaterializeError(
                    "a high-risk profile requires the exact per-protein --stratum-key in the "
                    "config; a mutable runtime manifest may not choose B(r) outside config_digest"
                )
            variant = getattr(args, "head_variant_id", None)
            allele_idx = getattr(args, "head_allele_idx", None)
            batch_size = getattr(args, "head_window_batch_size", None)
            if not isinstance(variant, str) or not variant.strip():
                raise MaterializeError(
                    "a high-risk profile requires a non-empty --head-variant-id"
                )
            if isinstance(allele_idx, bool) or not isinstance(allele_idx, int) or allele_idx < 0:
                raise MaterializeError(
                    "a high-risk profile requires --head-allele-idx >= 0"
                )
            if (isinstance(batch_size, bool) or not isinstance(batch_size, int)
                    or batch_size < 1):
                raise MaterializeError(
                    "a high-risk profile requires --head-window-batch-size >= 1"
                )
            config["head"].update({
                "head_variant_id": variant.strip(),
                "head_allele_idx": int(allele_idx),
                "head_window_batch_size": int(batch_size),
            })

    policy_calibration = getattr(args, "policy_calibration_json", None)
    if profile is not None and not policy_calibration:
        raise MaterializeError(
            f"exploratory profile {profile_name!r} requires --policy-calibration-json; "
            "head_directed_capped may not run from invented thresholds"
        )
    if policy_calibration:
        calibration_path = Path(policy_calibration)
        if not calibration_path.is_file():
            raise MaterializeError(
                f"policy calibration artifact does not exist or is not a file: "
                f"{calibration_path}"
            )
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "v2-head-policy-calibration-bundle/1":
            raise MaterializeError(
                f"{policy_calibration} is not a v2-head-policy-calibration-bundle/1 artifact"
            )
        block = payload.get("head_directed")
        if not isinstance(block, dict):
            raise MaterializeError(f"{policy_calibration} carries no head_directed config block")
        measured_head = payload.get("head")
        expected_head = {
            "allele": config["head"]["allele"],
            "score_scale": config["head"]["score_scale"],
            "window_k_min": int(config["head"]["window_k_min"]),
            "window_k_max": int(config["head"]["window_k_max"]),
            "head_config_hash": frozen["head_config"],
            "head_checkpoint_digest": frozen["head_checkpoint"],
        }
        if profile is not None and profile.get("search_structure") is not None:
            expected_head.update({
                "head_variant_id": config["head"]["head_variant_id"],
                "head_allele_idx": config["head"]["head_allele_idx"],
                "head_window_batch_size": config["head"]["head_window_batch_size"],
            })
        if not isinstance(measured_head, dict):
            raise MaterializeError(
                f"{policy_calibration} carries no frozen Head evaluator identity"
            )
        mismatch = {
            key: (measured_head.get(key), value)
            for key, value in expected_head.items() if measured_head.get(key) != value
        }
        if mismatch:
            raise MaterializeError(
                f"{policy_calibration} was measured on a different Head instrument/domain: "
                f"{mismatch}"
            )
        if payload.get("policy_spec_sha256") != frozen["projection_policy_spec"]:
            raise MaterializeError(
                f"{policy_calibration} was bound to policy spec "
                f"{payload.get('policy_spec_sha256')!r}, but this cell supplies "
                f"{frozen['projection_policy_spec']!r}"
            )
        policy_version, expected_depth0_rule = _projection_policy_method(
            args.projection_policy_spec)
        observed_depth0_rule = block.get("lineage_incumbent_depth0_rule")
        if observed_depth0_rule != expected_depth0_rule:
            raise MaterializeError(
                f"{policy_calibration} carries lineage_incumbent_depth0_rule="
                f"{observed_depth0_rule!r}, but policy spec {policy_version} requires "
                f"{expected_depth0_rule!r}"
            )
        if profile is not None:
            required_version = profile.get("required_policy_version")
            if required_version is not None and policy_version != required_version:
                raise MaterializeError(
                    f"exploratory profile {profile_name!r} requires policy spec "
                    f"{required_version!r}, got {policy_version!r}"
                )
            required_counterfactual = profile.get(
                "required_counterfactual_head_calls_per_cycle")
            observed_counterfactual = block.get(
                "max_counterfactual_head_calls_per_cycle")
            if (required_counterfactual is not None
                    and observed_counterfactual != required_counterfactual):
                raise MaterializeError(
                    f"exploratory profile {profile_name!r} freezes the counterfactual Head "
                    f"ceiling at {required_counterfactual} calls/cycle, got "
                    f"{observed_counterfactual!r}"
                )
        config["identity"]["phase"] = (
            str(profile["phase"]) if profile is not None else "policy_qualification"
        )
        config["identity"]["split_role"] = (
            str(profile["split_role"]) if profile is not None else "policy_qualification"
        )
        config["projection"].update({
            "support_policy_id": "head_directed_capped",
            "support_policy_version": policy_version,
            "support_policy_is_diagnostic": False,
            "head_directed": json.loads(json.dumps(block)),
        })
        head_cap = (
            getattr(args, "run_max_head_calls", None)
            if profile is not None
            else getattr(args, "qualification_max_head_calls", None)
        )
        if isinstance(head_cap, bool) or not isinstance(head_cap, int) or head_cap < 1:
            flag = (
                "--run-max-head-calls" if profile is not None
                else "--qualification-max-head-calls"
            )
            raise MaterializeError(
                f"{flag} is required with --policy-calibration-json"
                + (" for an explicit exploratory run Head cap" if profile is not None else
                   "; the one-cycle paired cohort must carry an explicit cohort Head cap")
            )
        required_head_cap = (
            profile.get("required_run_max_head_calls") if profile is not None else None
        )
        if required_head_cap is not None and head_cap != required_head_cap:
            raise MaterializeError(
                f"exploratory profile {profile_name!r} freezes its whole-run Head cap at "
                f"{required_head_cap}, got {head_cap}"
            )
        config["caps"]["max_head_calls"] = int(head_cap)

    if profile is not None:
        config["schedule"] = json.loads(json.dumps(profile["schedule"]))
        if profile.get("search_structure") is not None:
            config["schedule"]["stratum_key"] = str(args.stratum_key)
        config["caps"].update(json.loads(json.dumps(profile["caps"])))
        if profile.get("search_structure") is not None:
            config["safety"]["search_structure"] = json.loads(json.dumps(
                profile["search_structure"]))
    else:
        points = config["schedule"]["points"]
        if len(points) != 1:
            raise MaterializeError(
                f"the template declares {len(points)} schedule points; a Canary cell is ONE "
                "(protein, r_step) and a multi-point template cannot be resolved into one"
            )
        points[0]["r_step"] = int(args.r_step)
        points[0]["band_key"] = f"step{int(args.r_step)}"

    highrisk_hotspot = bool(profile is not None and profile.get("search_structure") is not None)
    config["safety"]["delta_new_cumulative"] = _delta_new_block(
        args.hotspot_json,
        head=config["head"],
        highrisk_protein_id=(str(args.protein_id) if highrisk_hotspot else None),
        code_revision=(str(args.code_revision) if highrisk_hotspot else None),
        master_seed=(int(config["identity"]["master_seed"]) if highrisk_hotspot else None),
        head_config_digest=(frozen["head_config"] if highrisk_hotspot else None),
        head_checkpoint_digest=(frozen["head_checkpoint"] if highrisk_hotspot else None),
    )

    for row in config["content"]:
        role = row["role"]
        if role in frozen:
            row["binding"], row["expected_sha256"] = "frozen", frozen[role]
        elif role in runtime:
            row["binding"], row["expected_sha256"] = "runtime", None
        else:
            raise MaterializeError(
                f"content role {role!r} was bound neither frozen nor runtime; PLAN §5.2 fails "
                "closed on missing content identity"
            )

    leftovers = sorted(_placeholders(config))
    if leftovers:
        raise MaterializeError(
            f"unresolved placeholder(s) remain: {leftovers}.  The loader would refuse them at "
            "launch; refusing here names the field instead"
        )
    return config


def _placeholders(node: Any, path: str = "") -> list[str]:
    if isinstance(node, dict):
        return [item for key, value in node.items()
                for item in _placeholders(value, f"{path}.{key}" if path else str(key))]
    if isinstance(node, list):
        return [item for index, value in enumerate(node)
                for item in _placeholders(value, f"{path}[{index}]")]
    return [path] if isinstance(node, str) and node.startswith("REPLACE") else []


def _args_script(
    args, *, frozen: dict, runtime: dict, config_path: Path,
    structure_runtime_identity: tuple[Path, dict[str, Any], str] | None = None,
) -> str:
    """The driver argument vectors this config requires, as a `source`-able shell fragment.

    `--input-file ROLE=PATH` signs the run and is checked against any frozen digest; `--shard-input
    NAME=PATH` is what the oracle stack and the conditioning identity actually read.  A role can
    need both, under two different names (`dplm_checkpoint` / `base_if_checkpoint`), which is
    exactly the kind of thing a hand-written command gets wrong once and then copies four times.

    `schedule_band_calibration`, `head_config`, `fixed_token_policy` and an unconstrained
    `constraint_manifest` are deliberately absent from `--input-file`: their frozen digests are a
    table digest, a directory rolling-hash, a policy label and a typed absence, so the driver's
    file-sha256 check would refuse the very files that produced them.
    """
    input_files = [
        # Frozen roles whose digest IS the file's sha256 -- these are the ones the driver can
        # genuinely verify, so declaring them turns the run signature into a real check.
        f"complete_reference_sequence={args.reference_sequence}",
        f"projection_policy_spec={args.projection_policy_spec}",
        f"head_checkpoint={args.head_checkpoint}",
    ] + [f"{role}={path}" for role, path in sorted(runtime.items())
         # coordinate_mask/tokenizer alias another role's file; recording them twice in the
         # manifest would claim two content identities where one file was read.
         if role not in {"coordinate_mask", "tokenizer", "fixed_token_policy"}]

    exploratory = bool(getattr(args, "exploratory_profile", None))
    if exploratory:
        identity = structure_runtime_identity or _structure_runtime_identity(args)
        input_files.append(f"structure_backend={identity[0]}")

    shard_inputs = dict(runtime)
    shard_inputs.update({
        # Names the oracle stack owns, distinct from the role vocabulary on purpose.
        "base_if_checkpoint": str(args.dplm_checkpoint),
        "test_set_parquet": str(args.cohort_table),
        "pdb_root": str(args.pdb_root),
        "refold_cache_dir": str(args.refold_cache_dir),
        "head_config": str(args.head_config_dir),
        "head_checkpoint": str(args.head_checkpoint),
        "head_variant_id": str(args.head_variant_id),
        "head_allele_idx": str(getattr(args, "head_allele_idx", 0)),
        "head_window_batch_size": str(getattr(args, "head_window_batch_size", 64)),
        "esmfold2_site_packages": str(args.esmfold2_site_packages),
        "schedule_band_calibration": str(args.band_json),
        "complete_reference_manifest": str(args.reference_manifest),
        "protein_stratum_manifest": str(args.stratum_manifest),
    })
    if exploratory:
        identity = structure_runtime_identity or _structure_runtime_identity(args)
        profile = EXPLORATORY_PROFILES.get(getattr(args, "exploratory_profile", None))
        protocol = _structure_runtime_protocol(args, profile=profile)
        shard_inputs.update({
            # The oracle consumes these exact names.  They are repeated in the signed runtime
            # identity artifact above, so changing a worker knob changes both config_digest and
            # run input_signature rather than silently falling back to v0's parser defaults.
            "structure_backend": str(identity[0]),
            "esmfold2_runtime_identity": str(identity[0]),
            "esmfold2_model": str(protocol["model_selector"]),
            "esmfold2_num_loops": str(protocol["protocol"]["num_loops"]),
            "esmfold2_num_sampling_steps": str(
                protocol["protocol"]["num_sampling_steps"]),
            "esmfold2_num_diffusion_samples": str(
                protocol["protocol"]["num_diffusion_samples"]),
            "esmfold2_seed": str(protocol["protocol"]["seed"]),
        })
        if protocol.get("esmc_model_selector") is not None:
            shard_inputs["esmfold2_esmc_model"] = str(protocol["esmc_model_selector"])
            shard_inputs["esmfold2_ccd_path"] = str(protocol["ccd_path"])
    # `fixed_token_policy` stays in SHARD_INPUTS even though no oracle reads that key: it is one of
    # the eighteen roles `_conditioning` must find a digest for, and on an anchored cell it is bound
    # RUNTIME, so dropping it made the shard refuse with "no content identity for role(s)
    # ['fixed_token_policy']".  On an unconstrained cell it is frozen and absent from `runtime`, so
    # it never appears here at all.

    lines = [
        "# GENERATED by scripts/materialize_v2_canary_config.py -- do not edit by hand.",
        f"# cell: {args.protein_id} r={int(args.r_step)}  stratum={args.stratum_key}",
        f"V2_CONFIG={shlex.quote(str(config_path))}",
        f"V2_COHORT={shlex.quote(str(args.protein_id))}",
        "INPUT_FILES=(",
        *[f"  {shlex.quote(item)}" for item in input_files],
        ")",
        "SHARD_INPUTS=(",
        *[f"  {shlex.quote(f'{name}={path}')}" for name, path in sorted(shard_inputs.items())],
        ")",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="materialize_v2_canary_config",
        description="resolve one V2 Canary cell's config + launch arguments from measured artifacts",
    )
    parser.add_argument("--template", required=True, help="tracked V2 Canary config template")
    parser.add_argument("--out", required=True, help="resolved config destination (.yaml)")
    parser.add_argument("--protein-id", required=True)
    parser.add_argument("--r-step", required=True, type=int)
    parser.add_argument("--stratum-key", required=True,
                        help="the COHORT stratum the band was measured over; not the band_key")
    parser.add_argument("--code-revision", required=True, help="7-64 lowercase hex; 'unknown' is refused")
    parser.add_argument("--band-json", required=True, help="this stratum's B(r) artifact")
    parser.add_argument("--hotspot-json", required=True, help="this protein's hotspot calibration")
    parser.add_argument("--reference-manifest", required=True, help="{protein_id: {path, sha256}}")
    parser.add_argument("--stratum-manifest", required=True, help="{protein_id: stratum_key}")
    parser.add_argument("--reference-sequence", required=True, help="this protein's .seq")
    parser.add_argument("--projection-policy-spec", required=True)
    parser.add_argument("--head-config-dir", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--head-variant-id", required=True)
    parser.add_argument("--head-allele-idx", type=int, default=0)
    parser.add_argument("--head-window-batch-size", type=int, default=64)
    parser.add_argument("--structure-backend", required=True,
                        help="the structure model's weight file (its bytes ARE the backend)")
    parser.add_argument("--structure-config", required=True)
    parser.add_argument("--v0-structure-gate-config", required=True)
    parser.add_argument("--rf-sampler-config", required=True)
    parser.add_argument("--dplm-checkpoint", required=True)
    parser.add_argument("--cohort-table", required=True, help="the test-set parquet")
    parser.add_argument("--backbone", required=True, help="this protein's structure file")
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--refold-cache-dir", required=True)
    parser.add_argument("--esmfold2-site-packages", required=True)
    parser.add_argument("--constraint-manifest", default=None,
                        help="omit for an unconstrained cell; its absence is then declared, "
                             "not left silent")
    parser.add_argument("--campaign-id", default=None,
                        help="override identity.campaign_id (default: the template's).  Use a "
                             "distinct campaign for a distinct question, e.g. the mechanism cohort")
    parser.add_argument(
        "--policy-calibration-json", default=None,
        help="optional v2-head-policy-calibration-bundle/1 artifact. When supplied, materialize "
             "a head_directed_capped config instead of the template's diagnostic policy. The "
             "default identity is policy_qualification; an explicit exploratory profile replaces "
             "it with that profile's capability identity. Calibrated values are copied verbatim "
             "and validated by the typed config loader",
    )
    parser.add_argument(
        "--qualification-max-head-calls", type=int, default=None,
        help="cohort Head-call hard cap written into a policy-qualification config; required with "
             "--policy-calibration-json and ignored otherwise",
    )
    parser.add_argument(
        "--exploratory-profile", choices=sorted(EXPLORATORY_PROFILES), default=None,
        help="explicitly materialize one closed, non-confirmatory capability-ladder profile. "
             "The profile freezes its identity, schedule, caps and optional search-safety block; "
             "the driver still requires its separate exploratory D>1 launch override",
    )
    parser.add_argument(
        "--run-max-head-calls", type=int, default=None,
        help="generic whole-run Head hard cap required by an exploratory profile. It is not a "
             "qualification-arm cap and is rejected unless --exploratory-profile is supplied; "
             "the high-risk D2/D8 profiles require their frozen values 1024/4096",
    )
    parser.add_argument("--esmfold2-model", default=None,
                        help="actual ESMFold2 model selector; required by an exploratory profile. "
                             "High-risk profiles require an absolute local snapshot directory "
                             "whose model.safetensors matches --structure-backend")
    parser.add_argument("--esmfold2-esmc-model", default=None,
                        help="absolute local ESMC snapshot used internally by ESMFold2; required "
                             "by high-risk profiles")
    parser.add_argument("--esmfold2-ccd-path", default=None,
                        help="absolute local ccd.pkl consumed by ESMFold2InputBuilder; required "
                             "by high-risk profiles")
    parser.add_argument("--esmfold2-num-loops", type=int, default=None)
    parser.add_argument("--esmfold2-num-sampling-steps", type=int, default=None)
    parser.add_argument("--esmfold2-num-diffusion-samples", type=int, default=None)
    parser.add_argument("--esmfold2-seed", type=int, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)

    template = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))
    structure_identity = (
        _structure_runtime_identity(args) if args.exploratory_profile else None
    )
    frozen, runtime = resolve_content_bindings(
        args, structure_runtime_identity=structure_identity)
    config = fill_config(template, args=args, frozen=frozen, runtime=runtime)

    out.parent.mkdir(parents=True, exist_ok=True)
    if structure_identity is not None:
        identity_path, identity_payload, identity_digest = structure_identity
        identity_path.write_bytes(_structure_runtime_bytes(identity_payload))
        if file_digest(identity_path) != identity_digest:  # pragma: no cover - disk corruption
            raise MaterializeError(
                f"structure runtime identity changed while writing {identity_path}"
            )
    out.write_text(yaml.safe_dump(config, sort_keys=True, default_flow_style=False),
                   encoding="utf-8")

    # Written, then LOADED: a config this producer cannot itself resolve is not a config, and the
    # digest below is the one the run will sign, computed by the loader rather than predicted here.
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    resolved = load_v2_config_file(out)

    args_path = out.with_suffix(".args.sh")
    args_path.write_text(_args_script(
        args, frozen=frozen, runtime=runtime, config_path=out,
        structure_runtime_identity=structure_identity),
                         encoding="utf-8")

    print(json.dumps({
        "config": str(out),
        "args": str(args_path),
        "protein_id": args.protein_id,
        "r_step": int(args.r_step),
        "stratum_key": args.stratum_key,
        "config_digest": resolved.config_digest(),
        "delta_new_cumulative": resolved.safety.delta_new_cumulative.value,
        "delta_new_unit": resolved.safety.delta_new_cumulative.unit,
        "frozen_roles": {role: frozen[role] for role in sorted(frozen)},
        "runtime_roles": sorted(runtime),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
