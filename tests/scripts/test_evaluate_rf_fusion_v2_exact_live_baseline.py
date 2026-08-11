"""Contracts for the exact-live Gumbel DPLM-native V2 comparator adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _write_source(
    tmp_path: Path, *, strategy: str = "gumbel_argmax", temperature: float = 1.0,
    seed: int = 42, max_iter: int = 100,
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "dplm.ckpt"
    checkpoint.write_bytes(b"frozen dplm")
    import hashlib

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame([
        {"protein_id": "P1", "sequence": "A" * 20, "sequence_length": 20},
        {"protein_id": "P2", "sequence": "C" * 20, "sequence_length": 20},
    ]).to_parquet(cohort, index=False)
    generated = tmp_path / "generated.parquet"
    rows = []
    for protein_id in ("P1", "P2"):
        for design_idx in range(8):
            sequence = "ACDEFGHIKLMNPQRSTVWY"[design_idx:] + "ACDEFGHIKLMNPQRSTVWY"[:design_idx]
            rows.append({
                "protein_id": protein_id,
                "design_idx": design_idx,
                "sequence": sequence,
                "seed": seed + design_idx,
            })
    pd.DataFrame(rows).to_parquet(generated, index=False)
    run_config = tmp_path / "run_config.yaml"
    run_config.write_text(
        "mode: c0_native_sampler\n"
        f"checkpoint: {checkpoint}\n"
        "sampler:\n"
        f"  max_iter: {max_iter}\n"
        f"  temperature: {temperature}\n"
        f"  sampling_strategy: {strategy}\n"
        "  n_designs_per_protein: 8\n"
        f"  seed: {seed}\n"
        "  batch_size: 8\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "mode": "c0_native_sampler",
        "checkpoint_digest": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "git_sha": "a" * 40,
        "n_failures": 0,
        "n_rows_generated": 16,
    }))
    return {
        "checkpoint": checkpoint,
        "cohort": cohort,
        "generated": generated,
        "run_config": run_config,
        "manifest": manifest,
    }


def _source_identity_kwargs(paths: dict[str, Path]) -> dict[str, str]:
    frozen = {
        f"expected_source_{role}_sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest()
        for role, name in (
            ("parquet", "generated"),
            ("run_config", "run_config"),
            ("manifest", "manifest"),
            ("checkpoint", "checkpoint"),
        )
    }
    frozen["expected_cohort_parquet_sha256"] = hashlib.sha256(
        paths["cohort"].read_bytes()).hexdigest()
    return frozen


def _write_structure_runtime(tmp_path: Path) -> dict[str, Path | str]:
    from inverse_folding.evaluation.esmfold2_live import (
        ESMC_SNAPSHOT_REQUIRED_FILES,
        ESMFOLD2_SNAPSHOT_REQUIRED_FILES,
        esmfold2_overlay_identity,
        hf_snapshot_identity,
    )

    snapshot_dir = tmp_path / "hf_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    weights = snapshot_dir / "model.safetensors"
    weights.write_bytes(b"frozen esmfold2 weights")
    (snapshot_dir / "config.json").write_text('{"model_type":"esmfold2"}\n')
    ccd = snapshot_dir / "ccd.pkl"
    ccd.write_bytes(b"frozen ccd")
    esmc = tmp_path / "esmc_snapshot"
    esmc.mkdir()
    for name in ESMC_SNAPSHOT_REQUIRED_FILES:
        (esmc / name).write_bytes(f"frozen:{name}".encode())
    site = tmp_path / "site-packages"
    (site / "esm" / "models" / "esmfold2").mkdir(parents=True)
    (site / "transformers" / "models" / "esmfold2").mkdir(parents=True)
    (site / "esm" / "models" / "esmfold2" / "model.py").write_text("ESM = 1\n")
    (site / "transformers" / "models" / "esmfold2" / "model.py").write_text("HF = 1\n")
    snapshot_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    runtime_json = tmp_path / "structure_runtime.json"
    runtime_json.write_text(json.dumps({
        "schema_version": "v2-esmfold2-runtime-1",
        "backend": "esmfold2_live",
        "model_selector": str(snapshot_dir.resolve()),
        "esmc_model_selector": str(esmc.resolve()),
        "ccd_path": str(ccd.resolve()),
        "local_model_snapshot_sha256": snapshot_sha256,
        "model_snapshot": hf_snapshot_identity(
            snapshot_dir, ESMFOLD2_SNAPSHOT_REQUIRED_FILES),
        "esmc_snapshot": hf_snapshot_identity(esmc, ESMC_SNAPSHOT_REQUIRED_FILES),
        "ccd_sha256": hashlib.sha256(ccd.read_bytes()).hexdigest(),
        "site_packages_overlay": esmfold2_overlay_identity(site),
        "protocol": {
            "num_loops": 3, "num_sampling_steps": 50,
            "num_diffusion_samples": 1, "seed": 0,
        },
    }))
    return {
        "runtime_json": runtime_json,
        "snapshot_dir": snapshot_dir.resolve(),
        "weights": weights.resolve(),
        "snapshot_sha256": snapshot_sha256,
        "esmc": esmc.resolve(),
        "ccd": ccd.resolve(),
        "site": site.resolve(),
    }


def test_source_gate_accepts_only_exact_eight_unique_gumbel_designs(tmp_path):
    from scripts.evaluate_rf_fusion_v2_exact_live_baseline import load_gumbel_baseline

    paths = _write_source(tmp_path)
    frame, identity = load_gumbel_baseline(
        source_parquet=paths["generated"],
        cohort_parquet=paths["cohort"],
        source_run_config=paths["run_config"],
        source_manifest=paths["manifest"],
        source_checkpoint=paths["checkpoint"],
        **_source_identity_kwargs(paths),
    )
    assert len(frame) == 16
    assert frame.groupby("protein_id")["sequence"].nunique().to_dict() == {"P1": 8, "P2": 8}
    assert identity["sampling_strategy"] == "gumbel_argmax"
    assert identity["n_designs_per_protein"] == 8


def test_source_gate_refuses_argmax_even_when_the_table_shape_is_eight(tmp_path):
    from scripts.evaluate_rf_fusion_v2_exact_live_baseline import (
        ExactLiveBaselineError,
        load_gumbel_baseline,
    )

    paths = _write_source(tmp_path, strategy="argmax")
    with pytest.raises(ExactLiveBaselineError, match="gumbel_argmax"):
        load_gumbel_baseline(
            source_parquet=paths["generated"],
            cohort_parquet=paths["cohort"],
            source_run_config=paths["run_config"],
            source_manifest=paths["manifest"],
            source_checkpoint=paths["checkpoint"],
            **_source_identity_kwargs(paths),
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("temperature", 0.7, "temperature"),
        ("seed", 43, "seed"),
        ("max_iter", 80, "max_iter"),
    ],
)
def test_source_gate_refuses_a_different_gumbel_sampling_law(
    tmp_path, field, value, expected_error,
):
    from scripts.evaluate_rf_fusion_v2_exact_live_baseline import (
        ExactLiveBaselineError,
        load_gumbel_baseline,
    )

    kwargs = {field: value}
    paths = _write_source(tmp_path, **kwargs)
    with pytest.raises(ExactLiveBaselineError, match=expected_error):
        load_gumbel_baseline(
            source_parquet=paths["generated"], cohort_parquet=paths["cohort"],
            source_run_config=paths["run_config"], source_manifest=paths["manifest"],
            source_checkpoint=paths["checkpoint"],
            **_source_identity_kwargs(paths),
        )


def test_source_gate_refuses_a_shape_valid_parquet_substituted_after_identity_freeze(tmp_path):
    from scripts.evaluate_rf_fusion_v2_exact_live_baseline import (
        ExactLiveBaselineError,
        load_gumbel_baseline,
    )

    paths = _write_source(tmp_path)
    expected = _source_identity_kwargs(paths)
    source = pd.read_parquet(paths["generated"])
    source.loc[0, "sequence"] = "WYACDEFGHIKLMNPQRSTV"
    assert source.groupby("protein_id")["sequence"].nunique().eq(8).all()
    source.to_parquet(paths["generated"], index=False)

    with pytest.raises(ExactLiveBaselineError, match="source parquet SHA-256"):
        load_gumbel_baseline(
            source_parquet=paths["generated"], cohort_parquet=paths["cohort"],
            source_run_config=paths["run_config"], source_manifest=paths["manifest"],
            source_checkpoint=paths["checkpoint"], **expected,
        )


def test_evaluator_batches_eight_head_requests_per_protein_and_folds_every_design(tmp_path):
    from scripts.evaluate_rf_fusion_v2_exact_live_baseline import evaluate_baseline_rows

    paths = _write_source(tmp_path)
    source = pd.read_parquet(paths["generated"])
    head_batch_sizes: list[tuple[str, int]] = []
    structure_keys: list[tuple[str, str]] = []

    class Head:
        def score(self, requests):
            head_batch_sizes.append((requests[0].protein_id, len(requests)))
            return [type("Score", (), {
                "protein_id": request.protein_id,
                "sequence_md5": request.sequence_md5,
                "sequence_length": request.sequence_length,
                "allele": "HLA-DRB1*07:01",
                "score_scale": "raw_logit",
                "windows": (type("Window", (), {
                    "start_0b": 0, "end_0b": 12, "k": 12, "z": 1.0,
                })(),),
                "residue_hotspot": (0.0,) * request.sequence_length,
                "global_risk": -1.0,
                "binding": type("Binding", (), {
                    "window_grid_digest": "b" * 64,
                    "evaluator": type("Evaluator", (), {"digest": lambda self: "c" * 64})(),
                })(),
            })() for request in requests]

    def structure(request):
        structure_keys.append((request.protein_id, request.sequence_md5))
        return type("Outcome", (), {
            "evaluated": True,
            "feasible": True,
            "failure_reason": None,
            "walltime_s": 0.1,
            "metrics": {"scTM": 0.9},
        })()

    head_rows, structure_rows = evaluate_baseline_rows(
        source, head_oracle=Head(), structure_oracle=structure,
        refold_cache_dir=tmp_path / "cache",
    )
    assert head_batch_sizes == [("P1", 8), ("P2", 8)]
    assert len(structure_keys) == len(structure_rows) == len(head_rows) == 16


def test_cache_identity_refuses_foreign_files_and_runtime_drift(tmp_path):
    from scripts.evaluate_rf_fusion_v2_exact_live_baseline import (
        ExactLiveBaselineError,
        claim_exact_live_cache,
    )

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "foreign.pdb").write_text("foreign")
    with pytest.raises(ExactLiveBaselineError, match="unclaimed"):
        claim_exact_live_cache(cache, {"runtime": "A"})
    (cache / "foreign.pdb").unlink()
    claim_exact_live_cache(cache, {"runtime": "A"})
    claim_exact_live_cache(cache, {"runtime": "A"})
    with pytest.raises(ExactLiveBaselineError, match="identity"):
        claim_exact_live_cache(cache, {"runtime": "B"})


def test_source_gate_refuses_repeated_designs_and_checkpoint_rebinding(tmp_path):
    from scripts.evaluate_rf_fusion_v2_exact_live_baseline import (
        ExactLiveBaselineError,
        load_gumbel_baseline,
    )

    paths = _write_source(tmp_path)
    source = pd.read_parquet(paths["generated"])
    source.loc[(source["protein_id"] == "P1") & (source["design_idx"] == 1), "sequence"] = \
        source.loc[(source["protein_id"] == "P1") & (source["design_idx"] == 0), "sequence"].iloc[0]
    source.to_parquet(paths["generated"], index=False)
    with pytest.raises(ExactLiveBaselineError, match="8/8 unique"):
        load_gumbel_baseline(
            source_parquet=paths["generated"], cohort_parquet=paths["cohort"],
            source_run_config=paths["run_config"], source_manifest=paths["manifest"],
            source_checkpoint=paths["checkpoint"],
            **_source_identity_kwargs(paths),
        )

    paths = _write_source(tmp_path / "other")
    paths["checkpoint"].write_bytes(b"substituted checkpoint")
    with pytest.raises(ExactLiveBaselineError, match="checkpoint bytes disagree"):
        load_gumbel_baseline(
            source_parquet=paths["generated"], cohort_parquet=paths["cohort"],
            source_run_config=paths["run_config"], source_manifest=paths["manifest"],
            source_checkpoint=paths["checkpoint"],
            **_source_identity_kwargs(paths),
        )


def test_structure_runtime_requires_verified_local_snapshot_directory(tmp_path):
    from scripts.evaluate_rf_fusion_v2_exact_live_baseline import (
        ExactLiveBaselineError,
        _load_structure_runtime,
    )

    paths = _write_structure_runtime(tmp_path)
    declaration = json.loads(paths["runtime_json"].read_text())
    declaration["model_selector"] = "biohub/ESMFold2"
    paths["runtime_json"].write_text(json.dumps(declaration))
    with pytest.raises(ExactLiveBaselineError, match="absolute local HF snapshot directory"):
        _load_structure_runtime(paths["runtime_json"])

    declaration["model_selector"] = str(paths["snapshot_dir"])
    declaration["local_model_snapshot_sha256"] = "0" * 64
    paths["runtime_json"].write_text(json.dumps(declaration))
    with pytest.raises(ExactLiveBaselineError, match="model.safetensors SHA-256 disagrees"):
        _load_structure_runtime(paths["runtime_json"])

    declaration["local_model_snapshot_sha256"] = paths["snapshot_sha256"]
    paths["runtime_json"].write_text(json.dumps(declaration))
    realized, _digest, _path = _load_structure_runtime(paths["runtime_json"])
    assert realized["model_selector"] == str(paths["snapshot_dir"])
    assert realized["resolved_model_safetensors"] == {
        "path": str(paths["weights"]),
        "sha256": paths["snapshot_sha256"],
    }


def test_production_adapter_pins_v2_protocol_and_head_window_batch_size(tmp_path):
    from scripts.evaluate_rf_fusion_v2_exact_live_baseline import (
        ExactLiveBaselineError,
        _build_exact_live_oracles,
        _load_structure_runtime,
    )

    args = SimpleNamespace(
        structure_config="structure.yaml", head_config_dir="head-config",
        head_checkpoint="head.pt", test_set_parquet="cohort.parquet", pdb_root="pdbs",
        refold_cache_dir="cache", allele="HLA-DRB1*07:01", score_scale="raw_logit",
        window_k_min=12, window_k_max=25, head_variant_id="LC1", head_allele_idx=0,
        constraint_manifest=None, device="cuda", esmfold2_site_packages="overlay",
    )
    runtime_paths = _write_structure_runtime(tmp_path)
    args.esmfold2_site_packages = str(runtime_paths["site"])
    declaration, _digest, _path = _load_structure_runtime(runtime_paths["runtime_json"])
    seen = {}
    reported_model_name = [str(runtime_paths["snapshot_dir"])]

    def fake_v0(v0_args, _config):
        v0_args._refold_runtime = {
            "model_name": reported_model_name[0],
            "esmc_model": str(runtime_paths["esmc"]),
            "ccd_path": str(runtime_paths["ccd"]),
            "esm_module": str(runtime_paths["site"] / "esm" / "__init__.py"),
            "transformers_module": str(
                runtime_paths["site"] / "transformers" / "__init__.py"),
            "python_executable": "/envs/immune-design/bin/python",
            "torch_version": "2.5.1",
        }
        return object()

    head, structure = object(), object()

    def fake_production(**kwargs):
        seen.update(kwargs)
        kwargs["build_oracles"](SimpleNamespace(), object())
        return head, structure

    actual_head, actual_structure, runtime = _build_exact_live_oracles(
        args, declaration, production_builder=fake_production, v0_builder=fake_v0,
    )
    assert (actual_head, actual_structure) == (head, structure)
    assert seen["head_window_batch_size"] == 64
    assert seen["esmfold2"] == {
        "esmfold2_site_packages": str(runtime_paths["site"]),
        "esmfold2_model": str(runtime_paths["snapshot_dir"]),
        "esmfold2_esmc_model": str(runtime_paths["esmc"]),
        "esmfold2_ccd_path": str(runtime_paths["ccd"]),
        "esmfold2_num_loops": 3,
        "esmfold2_num_sampling_steps": 50,
        "esmfold2_num_diffusion_samples": 1,
        "esmfold2_seed": 0,
    }
    assert runtime["python_executable"] == "/envs/immune-design/bin/python"

    reported_model_name[0] = "biohub/ESMFold2"
    with pytest.raises(ExactLiveBaselineError, match="model_name|verified.*snapshot"):
        _build_exact_live_oracles(
            args, declaration, production_builder=fake_production, v0_builder=fake_v0,
        )


def _launch_inputs(tmp_path: Path) -> tuple[list[str], dict[str, Path | str]]:
    paths = _write_source(tmp_path)
    pdb_root = tmp_path / "pdbs"
    pdb_root.mkdir()
    cohort = pd.read_parquet(paths["cohort"])
    for protein_id in cohort["protein_id"]:
        structure = pdb_root / f"{protein_id}.pdb"
        structure.write_text(f"REMARK test backbone for {protein_id}\n")
    cohort.to_parquet(paths["cohort"], index=False)
    head_config = tmp_path / "head_config"
    head_config.mkdir()
    (head_config / "model.yaml").write_text("model: fake\n")
    head_checkpoint = tmp_path / "head.pt"
    head_checkpoint.write_bytes(b"frozen head")
    runtime_paths = _write_structure_runtime(tmp_path)
    overlay = runtime_paths["site"]
    runtime_json = runtime_paths["runtime_json"]
    project_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "out"
    cache = tmp_path / "cache"
    argv = [
        "--source-parquet", str(paths["generated"]),
        "--source-run-config", str(paths["run_config"]),
        "--source-manifest", str(paths["manifest"]),
        "--source-checkpoint", str(paths["checkpoint"]),
        "--expected-source-parquet-sha256",
        hashlib.sha256(paths["generated"].read_bytes()).hexdigest(),
        "--expected-source-run-config-sha256",
        hashlib.sha256(paths["run_config"].read_bytes()).hexdigest(),
        "--expected-source-manifest-sha256",
        hashlib.sha256(paths["manifest"].read_bytes()).hexdigest(),
        "--expected-source-checkpoint-sha256",
        hashlib.sha256(paths["checkpoint"].read_bytes()).hexdigest(),
        "--expected-cohort-parquet-sha256",
        hashlib.sha256(paths["cohort"].read_bytes()).hexdigest(),
        "--cohort-parquet", str(paths["cohort"]),
        "--expected-proteins", "2",
        "--head-config-dir", str(head_config),
        "--head-checkpoint", str(head_checkpoint),
        "--head-variant-id", "LC1",
        "--allele", "HLA-DRB1*07:01",
        "--score-scale", "raw_logit",
        "--window-k-min", "12",
        "--window-k-max", "25",
        "--test-set-parquet", str(paths["cohort"]),
        "--pdb-root", str(pdb_root),
        "--structure-config", str(
            project_root / "inverse_folding/reference_flow/configs/"
            "rf_refine_fusion_v2_mechanism_5zhv_b.yaml"
        ),
        "--structure-runtime-json", str(runtime_json),
        "--refold-cache-dir", str(cache),
        "--esmfold2-site-packages", str(overlay),
        "--device", "cuda",
        "--out-dir", str(out_dir),
    ]
    return argv, {
        **paths, "head_checkpoint": head_checkpoint, "runtime_json": runtime_json,
        "out_dir": out_dir, "cache": cache, **runtime_paths,
    }


def test_dry_run_stops_before_oracle_or_cache_construction(tmp_path, capsys):
    from scripts.evaluate_rf_fusion_v2_exact_live_baseline import main

    argv, paths = _launch_inputs(tmp_path)

    def forbidden_factory(*_args, **_kwargs):
        raise AssertionError("dry-run must not construct the Head or structure model")

    assert main([*argv, "--dry-run"], oracles_factory=forbidden_factory) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN_OK"
    assert payload["head_batches"] == 2
    assert payload["live_structure_requests"] == 16
    assert not paths["out_dir"].exists()
    assert not paths["cache"].exists()


def test_complete_bundle_binds_source_head_snapshot_protocol_runtime_git_and_artifacts(tmp_path):
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from scripts.evaluate_rf_fusion_v2_exact_live_baseline import main

    argv, paths = _launch_inputs(tmp_path)

    class Identity:
        head_checkpoint_digest = hashlib.sha256(paths["head_checkpoint"].read_bytes()).hexdigest()

        def canonical_payload(self):
            return {
                "allele": "HLA-DRB1*07:01", "score_scale": "raw_logit",
                "window_k_min": 12, "window_k_max": 25,
                "head_config_hash": "e" * 64,
                "head_checkpoint_digest": self.head_checkpoint_digest,
            }

        def digest(self):
            return hashlib.sha256(
                json.dumps(self.canonical_payload(), sort_keys=True).encode()
            ).hexdigest()

    identity = Identity()

    class Head:
        def evaluator_identity(self):
            return identity

        def score(self, requests):
            assert len(requests) == 8
            return [SimpleNamespace(
                protein_id=request.protein_id,
                sequence_md5=sequence_md5(request.sequence),
                sequence_length=request.sequence_length,
                allele="HLA-DRB1*07:01",
                score_scale="raw_logit",
                windows=(SimpleNamespace(start_0b=0, end_0b=12, k=12, z=1.0),),
                residue_hotspot=(0.0,) * request.sequence_length,
                global_risk=-1.0,
                binding=SimpleNamespace(
                    evaluator=identity, window_grid_digest="f" * 64,
                ),
            ) for request in requests]

    def structure(request):
        return SimpleNamespace(
            evaluated=True, feasible=True, metrics={"scTM": 0.91, "pLDDT": 90.0},
            failure_reason=None, walltime_s=0.1, cache_status="miss", model_executed=True,
        )

    def factory(_args, declaration):
        assert declaration["local_model_snapshot_sha256"] == paths["snapshot_sha256"]
        return Head(), structure, {
            "python_executable": "/envs/immune-design/bin/python",
            "torch_version": "2.5.1",
            "model_name": str(paths["snapshot_dir"]),
        }

    assert main(argv, oracles_factory=factory) == 0
    manifest = json.loads((paths["out_dir"] / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["counts"]["designs"] == manifest["counts"]["structure_measured"] == 16
    assert manifest["source_identity"]["sampling_strategy"] == "gumbel_argmax"
    assert manifest["head"]["identity"]["head_checkpoint_digest"] == \
        hashlib.sha256(paths["head_checkpoint"].read_bytes()).hexdigest()
    assert manifest["structure"]["runtime_declaration"][
        "local_model_snapshot_sha256"] == paths["snapshot_sha256"]
    assert manifest["structure"]["runtime_declaration"]["model_selector"] == \
        str(paths["snapshot_dir"])
    assert manifest["structure"]["runtime_declaration"]["resolved_model_safetensors"] == {
        "path": str(paths["weights"]),
        "sha256": paths["snapshot_sha256"],
    }
    assert manifest["structure"]["runtime_declaration"]["protocol"][
        "num_sampling_steps"] == 50
    assert manifest["structure"]["worker_runtime"]["torch_version"] == "2.5.1"
    assert manifest["code"]["revision"] and manifest["code"]["script_sha256"]
    for artifact in manifest["artifacts"].values():
        path = paths["out_dir"] / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    assert (paths["cache"] / ".rf_fusion_v2_exact_live_cache_identity.json").is_file()
