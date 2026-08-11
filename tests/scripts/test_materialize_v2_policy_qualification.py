"""The V2F5A materializer binds calibration to the exact Head and policy spec."""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest

from inverse_folding.reference_flow.fusion_v2.identity import HeadEvaluatorIdentity
from inverse_folding.reference_flow.fusion_v2.config import load_v2_config
from inverse_folding.reference_flow.fusion_v2.structure_gate import (
    DUAL_SCTM_POLICY_KIND,
    HIGH_RISK_ANCESTRY_SCTM_MIN,
    HIGH_RISK_DUAL_SCTM_PROFILE_ID,
    HIGH_RISK_STRICT_SCTM_MIN,
)
from scripts.calibrate_v2_head_policy import build_calibration_bundle
from scripts.materialize_v2_canary_config import (
    MaterializeError,
    _args_script,
    _structure_runtime_identity,
    fill_config,
)
from scripts.rf_fusion_v2_preflight import project_v2_budget
from tests.inverse_folding.test_fusion_v2_config import _mapping


D = "a" * 64


def _case(tmp_path, *, policy_version="v1", counterfactual_calls=278):
    template = _mapping(**{
        "schedule.depth_cap": 1,
        "schedule.points": [{
            "depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
            "n_lookaheads": 4, "band_key": "step40",
        }],
    })
    hotspot = tmp_path / "hotspot.json"
    hotspot.write_text(json.dumps({"delta_new": template["safety"]["delta_new_cumulative"]}))
    spec = tmp_path / "policy.json"
    spec_payload = {
        "policy_id": "head_directed_capped", "policy_version": policy_version,
    }
    if policy_version == "v2":
        spec_payload.update({
            "depth0_reward_bootstrap": {"reward_incumbent": None},
            "artifact_contract": {"d0_gate_kind": "depth0_bootstrap"},
        })
    spec.write_text(json.dumps(spec_payload))
    spec_sha = hashlib.sha256(spec.read_bytes()).hexdigest()
    evaluator = HeadEvaluatorIdentity(
        allele=template["head"]["allele"], score_scale=template["head"]["score_scale"],
        window_k_min=template["head"]["window_k_min"],
        window_k_max=template["head"]["window_k_max"],
        head_config_hash=D, head_checkpoint_digest=D,
    )
    calibration = build_calibration_bundle(
        [{"protein_id": "P1", "sequence_md5": "b" * 32, "abs_repeat_drift": 0.01}],
        evaluator=evaluator, policy_spec=spec,
        max_counterfactual_head_calls_per_cycle=counterfactual_calls,
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    args = types.SimpleNamespace(
        code_revision="deadbeef", campaign_id="v2f5a-test", r_step=40,
        hotspot_json=hotspot, policy_calibration_json=calibration_path,
        qualification_max_head_calls=40000,
        exploratory_profile=None, run_max_head_calls=None,
        esmfold2_model=None, esmfold2_num_loops=None,
        esmfold2_num_sampling_steps=None, esmfold2_num_diffusion_samples=None,
        esmfold2_seed=None,
        projection_policy_spec=spec, structure_backend=None,
    )
    frozen = {row["role"]: D for row in template["content"]}
    frozen.update({"head_config": D, "head_checkpoint": D,
                   "projection_policy_spec": spec_sha})
    return template, args, frozen


def _highrisk_case(tmp_path, *, profile, run_max_head_calls):
    template, args, frozen = _case(
        tmp_path, policy_version="v2", counterfactual_calls=454)
    from inverse_folding.reference_flow.fusion_v2.config import (
        HotspotCalibrationArtifact,
        calibration_source_ref,
    )
    from scripts.calibrate_rf_fusion_v2_hotspot import source_id_for

    args.protein_id = "P1"
    args.stratum_key = "highrisk_nod_v1_P1_unconstrained"
    args.head_variant_id = "LC1"
    args.head_allele_idx = 0
    args.head_window_batch_size = 64
    calibration_path = Path(args.policy_calibration_json)
    calibration = json.loads(calibration_path.read_text())
    calibration["head"].update({
        "head_variant_id": args.head_variant_id,
        "head_allele_idx": args.head_allele_idx,
        "head_window_batch_size": args.head_window_batch_size,
    })
    calibration_path.write_text(json.dumps(calibration))
    hotspot_path = Path(args.hotspot_json)
    hotspot = json.loads(hotspot_path.read_text())
    block = hotspot["delta_new"]
    source_id = source_id_for(args.protein_id, population="full_trajectory")
    artifact = HotspotCalibrationArtifact(**block["artifact"])
    block["source_id"] = source_id
    block["source_ref"] = calibration_source_ref(
        value=block["value"], unit=block["unit"], source_kind=block["source_kind"],
        source_id=source_id, artifact=artifact,
    )
    hotspot.update({
        "protein_id": args.protein_id,
        "code_revision": args.code_revision,
        "population": "full_trajectory",
        "sampling_unit": "trajectory",
        "threshold_statistic": "per_protein_feedback_disabled_head_valid_q90_higher",
        "seed_namespace": "v2_hotspot_calibration_1",
        "master_seed": template["identity"]["master_seed"],
        "min_head_valid": 64,
        "n_attempted": 64,
        "n_head_valid": 64,
        "n_retained": 64,
        "q90_higher": block["value"],
        "failure_counts": {},
        "structure_diagnostic_mode": "skip_independent_capability_ceiling",
        "structure_operability": {
            "status": "not_measured", "n_definitive_feasible": None, "rate": None,
        },
        "head": calibration["head"],
    })
    hotspot_path.write_text(json.dumps(hotspot))
    snapshot = tmp_path / "esmfold2-snapshot"
    snapshot.mkdir()
    model = snapshot / "model.safetensors"
    model.write_bytes(b"frozen ESMFold2 model")
    (snapshot / "config.json").write_text('{"model_type":"esmfold2"}\n')
    (snapshot / "ccd.pkl").write_bytes(b"frozen CCD")
    esmc = tmp_path / "esmc-snapshot"
    esmc.mkdir()
    for name in (
        "config.json", "model.safetensors.index.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json",
        *(f"model-{index:05d}-of-00006.safetensors" for index in range(1, 7)),
    ):
        (esmc / name).write_bytes(f"frozen {name}".encode())
    args.exploratory_profile = profile
    args.run_max_head_calls = run_max_head_calls
    args.esmfold2_model = str(snapshot.resolve())
    args.esmfold2_esmc_model = str(esmc.resolve())
    args.esmfold2_ccd_path = str((snapshot / "ccd.pkl").resolve())
    args.esmfold2_num_loops = 3
    args.esmfold2_num_sampling_steps = 50
    args.esmfold2_num_diffusion_samples = 1
    args.esmfold2_seed = 0
    args.structure_backend = model
    structure_config = tmp_path / "structure.yaml"
    structure_config.write_text("structure: frozen\n")
    args.structure_config = str(structure_config)
    args.v0_structure_gate_config = str(structure_config)
    site = tmp_path / "site-packages"
    esmfold2_impl = site / "esm" / "models" / "esmfold2"
    transformers_impl = site / "transformers" / "models" / "esmfold2"
    esmfold2_impl.mkdir(parents=True)
    transformers_impl.mkdir(parents=True)
    (esmfold2_impl / "model.py").write_text("ESMFOLD2_IMPL = 1\n")
    (transformers_impl / "modeling_esmfold2.py").write_text("TRANSFORMERS_IMPL = 1\n")
    args.esmfold2_site_packages = str(site)
    args.out = tmp_path / "resolved.yaml"
    return template, args, frozen


def test_policy_qualification_materialization_copies_only_a_matching_calibration(tmp_path):
    template, args, frozen = _case(tmp_path)
    config = fill_config(template, args=args, frozen=frozen, runtime={})
    assert config["identity"]["phase"] == "policy_qualification"
    assert config["projection"]["support_policy_id"] == "head_directed_capped"
    assert config["projection"]["head_directed"][
        "max_counterfactual_head_calls_per_cycle"] == 278
    assert config["caps"]["max_head_calls"] == 40000
    assert config["schedule"]["depth_cap"] == 1
    assert config["schedule"]["points"] == [{
        "depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
        "n_lookaheads": 4, "band_key": "step40",
    }]
    assert load_v2_config(config).projection.head_directed is not None


def test_a_calibration_from_another_head_is_refused(tmp_path):
    template, args, frozen = _case(tmp_path)
    frozen["head_checkpoint"] = "f" * 64
    with pytest.raises(MaterializeError, match="different Head instrument"):
        fill_config(template, args=args, frozen=frozen, runtime={})


def test_explicit_uricase_recursive_profile_materializes_the_frozen_d4_k12_schedule(tmp_path):
    template, args, frozen = _case(tmp_path)
    args.exploratory_profile = "uricase_d4_k12_r40"
    args.run_max_head_calls = 90000
    args.esmfold2_model = "biohub/ESMFold2"
    args.esmfold2_num_loops = 3
    args.esmfold2_num_sampling_steps = 50
    args.esmfold2_num_diffusion_samples = 1
    args.esmfold2_seed = 0

    config = fill_config(template, args=args, frozen=frozen, runtime={})

    assert config["identity"]["phase"] == "capability_ladder"
    assert config["identity"]["split_role"] == "exploratory_uricase"
    assert config["schedule"]["schedule_id"] == "exploratory-uricase-d4-k12-r40-v1"
    assert config["schedule"]["depth_cap"] == 4
    assert config["schedule"]["active_population_width"] == 1
    assert config["schedule"]["points"] == [
        {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
         "n_lookaheads": 12, "band_key": "step40"},
        {"depth": 1, "r_step": 40, "c_source_step": 60, "c_next_step": 70,
         "n_lookaheads": 12, "band_key": "step40"},
        {"depth": 2, "r_step": 40, "c_source_step": 70, "c_next_step": 80,
         "n_lookaheads": 12, "band_key": "step40"},
        {"depth": 3, "r_step": 40, "c_source_step": 80, "c_next_step": 90,
         "n_lookaheads": 12, "band_key": "step40"},
    ]
    assert config["caps"]["max_logical_dfe"] == 2200
    assert config["caps"]["max_definitive_refolds"] == 64
    assert config["caps"]["max_head_calls"] == 90000
    loaded = load_v2_config(config)
    assert loaded.identity.phase == "capability_ladder"
    assert loaded.schedule.depth_cap == 4


def test_recursive_profile_is_never_inferred_from_an_ordinary_calibrated_call(tmp_path):
    template, args, frozen = _case(tmp_path)

    config = fill_config(template, args=args, frozen=frozen, runtime={})

    assert config["identity"]["phase"] == "policy_qualification"
    assert config["identity"]["split_role"] == "policy_qualification"
    assert config["schedule"]["depth_cap"] == 1
    assert config["caps"]["max_head_calls"] == 40000

    args.run_max_head_calls = 90000
    with pytest.raises(MaterializeError, match="requires an explicit --exploratory-profile"):
        fill_config(template, args=args, frozen=frozen, runtime={})


def test_recursive_profile_requires_its_calibration_r40_cell_and_generic_head_cap(tmp_path):
    template, args, frozen = _case(tmp_path)
    args.exploratory_profile = "uricase_d4_k12_r40"
    args.esmfold2_model = "biohub/ESMFold2"
    args.esmfold2_num_loops = 3
    args.esmfold2_num_sampling_steps = 50
    args.esmfold2_num_diffusion_samples = 1
    args.esmfold2_seed = 0
    args.run_max_head_calls = None
    with pytest.raises(MaterializeError, match="--run-max-head-calls"):
        fill_config(template, args=args, frozen=frozen, runtime={})

    args.run_max_head_calls = 90000
    args.policy_calibration_json = None
    with pytest.raises(MaterializeError, match="--policy-calibration-json"):
        fill_config(template, args=args, frozen=frozen, runtime={})

    args.policy_calibration_json = tmp_path / "missing-policy-calibration.json"
    with pytest.raises(MaterializeError, match="does not exist"):
        fill_config(template, args=args, frozen=frozen, runtime={})

    args.policy_calibration_json = _case(tmp_path)[1].policy_calibration_json
    args.r_step = 30
    with pytest.raises(MaterializeError, match="requires --r-step 40"):
        fill_config(template, args=args, frozen=frozen, runtime={})


def test_recursive_profile_requires_an_explicit_esmfold2_selector_and_protocol(tmp_path):
    template, args, frozen = _case(tmp_path)
    args.exploratory_profile = "uricase_d4_k12_r40"
    args.run_max_head_calls = 90000

    for name, value in (
        ("esmfold2_model", "biohub/ESMFold2"),
        ("esmfold2_num_loops", 3),
        ("esmfold2_num_sampling_steps", 50),
        ("esmfold2_num_diffusion_samples", 1),
        ("esmfold2_seed", 0),
    ):
        with pytest.raises(MaterializeError, match=name.replace("_", "-")):
            fill_config(template, args=args, frozen=frozen, runtime={})
        setattr(args, name, value)

    assert fill_config(template, args=args, frozen=frozen, runtime={})[
        "identity"]["split_role"] == "exploratory_uricase"


def test_structure_runtime_identity_binds_snapshot_selector_protocol_and_shard_inputs(tmp_path):
    template, args, frozen = _case(tmp_path)
    args.exploratory_profile = "uricase_d4_k12_r40"
    args.run_max_head_calls = 90000
    args.esmfold2_model = "biohub/ESMFold2"
    args.esmfold2_num_loops = 3
    args.esmfold2_num_sampling_steps = 50
    args.esmfold2_num_diffusion_samples = 1
    args.esmfold2_seed = 7
    args.out = tmp_path / "resolved.yaml"
    args.structure_backend = tmp_path / "model.safetensors"
    args.structure_backend.write_bytes(b"frozen model snapshot")

    path, payload, digest = _structure_runtime_identity(args)
    assert path == tmp_path / "resolved.structure_runtime.json"
    assert payload["schema_version"] == "v2-esmfold2-runtime-1"
    assert payload["backend"] == "esmfold2_live"
    assert payload["model_selector"] == "biohub/ESMFold2"
    assert payload["protocol"] == {
        "num_loops": 3, "num_sampling_steps": 50,
        "num_diffusion_samples": 1, "seed": 7,
    }
    assert payload["local_model_snapshot_sha256"] == hashlib.sha256(
        b"frozen model snapshot").hexdigest()
    assert len(digest) == 64
    frozen["structure_backend"] = digest
    config = fill_config(template, args=args, frozen=frozen, runtime={})
    structure_row = next(row for row in config["content"]
                         if row["role"] == "structure_backend")
    assert structure_row == {
        "role": "structure_backend", "label": "structure_backend.bin",
        "binding": "frozen", "expected_sha256": digest,
    }

    for name in (
        "reference_sequence", "projection_policy_spec", "head_checkpoint", "dplm_checkpoint",
        "cohort_table", "pdb_root", "refold_cache_dir", "head_config_dir", "head_variant_id",
        "esmfold2_site_packages", "band_json", "reference_manifest", "stratum_manifest",
        "protein_id", "stratum_key",
    ):
        if not hasattr(args, name):
            setattr(args, name, f"/{name}")
    args.reference_sequence = "/reference.seq"
    args.projection_policy_spec = "/policy.json"
    args.head_checkpoint = "/head.pt"
    args.dplm_checkpoint = "/dplm.pt"
    args.cohort_table = "/cohort.parquet"
    args.pdb_root = "/pdbs"
    args.refold_cache_dir = "/refolds"
    args.head_config_dir = "/head-config"
    args.head_variant_id = "LC1"
    args.esmfold2_site_packages = "/site-packages"
    args.band_json = "/band.json"
    args.reference_manifest = "/references.json"
    args.stratum_manifest = "/strata.json"
    args.protein_id = "P1"
    args.r_step = 40
    args.stratum_key = "uricase"
    text = _args_script(args, frozen={}, runtime={}, config_path=tmp_path / "resolved.yaml")
    assert f"structure_backend={path}" in text
    assert "esmfold2_model=biohub/ESMFold2" in text
    assert "esmfold2_num_loops=3" in text
    assert "esmfold2_num_sampling_steps=50" in text
    assert "esmfold2_num_diffusion_samples=1" in text
    assert "esmfold2_seed=7" in text


@pytest.mark.parametrize(
    ("profile", "head_cap", "depth_cap", "schedule_id", "points", "logical_cap",
     "refold_cap", "projected_dfe", "projected_head", "projected_refolds"),
    [
        (
            "highrisk_d2_k32_r40", 1024, 2, "highrisk-d2-k32-r40-v1",
            [
                {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 70,
                 "n_lookaheads": 32, "band_key": "step40"},
                {"depth": 1, "r_step": 40, "c_source_step": 70, "c_next_step": 90,
                 "n_lookaheads": 32, "band_key": "step40"},
            ],
            3072, 128, 3010, 1004, 96,
        ),
        (
            "highrisk_d8_k32_r40", 4096, 8, "highrisk-d8-k32-r40-v1",
            [
                {"depth": depth, "r_step": 40,
                 "c_source_step": 50 + 5 * depth,
                 "c_next_step": 55 + 5 * depth,
                 "n_lookaheads": 32, "band_key": "step40"}
                for depth in range(8)
            ],
            9216, 320, 8950, 3920, 288,
        ),
    ],
)
def test_highrisk_profiles_freeze_v2_schedule_caps_and_dual_structure(
    tmp_path, profile, head_cap, depth_cap, schedule_id, points, logical_cap, refold_cap,
    projected_dfe, projected_head, projected_refolds,
):
    template, args, frozen = _highrisk_case(
        tmp_path, profile=profile, run_max_head_calls=head_cap)

    config = fill_config(template, args=args, frozen=frozen, runtime={})

    assert config["identity"]["phase"] == "capability_ladder"
    assert config["identity"]["split_role"] == "exploratory_highrisk_ceiling_v1"
    assert config["projection"]["support_policy_version"] == "v2"
    assert config["projection"]["head_directed"][
        "lineage_incumbent_depth0_rule"] == "best_admissible_depth0"
    assert config["schedule"] == {
        "schedule_id": schedule_id,
        "coordinate_law": "progressive_checkpoint",
        "depth_cap": depth_cap,
        "active_population_width": 1,
        "min_lookahead_tail_steps": 10,
        "stratum_key": args.stratum_key,
        "points": points,
    }
    assert config["head"]["head_variant_id"] == "LC1"
    assert config["head"]["head_allele_idx"] == 0
    assert config["head"]["head_window_batch_size"] == 64
    assert config["caps"] == {
        "max_logical_dfe": logical_cap,
        "max_head_calls": head_cap,
        "max_definitive_refolds": refold_cap,
        "max_gpu_seconds": 14400,
        "max_walltime_s": 14400,
        "max_retries": 2,
        "retry_scope": "per_request",
    }
    assert config["safety"]["search_structure"] == {
        "policy_kind": DUAL_SCTM_POLICY_KIND,
        "profile_id": HIGH_RISK_DUAL_SCTM_PROFILE_ID,
        "ancestry_sctm_min": HIGH_RISK_ANCESTRY_SCTM_MIN,
        "strict_sctm_min": HIGH_RISK_STRICT_SCTM_MIN,
    }
    loaded = load_v2_config(config)
    assert loaded.schedule.depth_cap == depth_cap
    assert loaded.dual_structure_policy().profile_id == HIGH_RISK_DUAL_SCTM_PROFILE_ID
    projection = project_v2_budget(loaded, n_proteins=1)
    assert projection.feasible
    assert projection.total_logical_dfe == projected_dfe
    assert projection.total_head_calls == projected_head
    assert projection.total_definitive_refolds == projected_refolds


def test_highrisk_profile_refuses_a_changed_head_cap_or_counterfactual_ceiling(tmp_path):
    template, args, frozen = _highrisk_case(
        tmp_path, profile="highrisk_d2_k32_r40", run_max_head_calls=1023)
    with pytest.raises(MaterializeError, match="1024|Head cap"):
        fill_config(template, args=args, frozen=frozen, runtime={})

    args.run_max_head_calls = 1024
    payload = json.loads(Path(args.policy_calibration_json).read_text())
    payload["head_directed"]["max_counterfactual_head_calls_per_cycle"] = 453
    Path(args.policy_calibration_json).write_text(json.dumps(payload))
    with pytest.raises(MaterializeError, match="454|counterfactual"):
        fill_config(template, args=args, frozen=frozen, runtime={})


def test_highrisk_profile_requires_one_byte_identical_structure_gate_config(tmp_path):
    template, args, frozen = _highrisk_case(
        tmp_path, profile="highrisk_d2_k32_r40", run_max_head_calls=1024)
    drifted = tmp_path / "drifted-structure.yaml"
    drifted.write_text("structure: drifted\n")
    args.v0_structure_gate_config = str(drifted)

    with pytest.raises(MaterializeError, match="structure.*config|byte-identical"):
        fill_config(template, args=args, frozen=frozen, runtime={})


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("protein_id",), "P2", "protein"),
        (("population",), "resumed", "population"),
        (("threshold_statistic",),
         "per_protein_resumed_feedback_off_head_valid_q90_higher", "statistic"),
        (("n_attempted",), 63, "n_attempted|64"),
        (("n_head_valid",), 63, "n_head_valid|64"),
        (("min_head_valid",), 63, "min_head_valid|64"),
        (("seed_namespace",), "other", "seed_namespace"),
        (("master_seed",), 7, "master_seed"),
        (("code_revision",), "cafebabe", "code_revision"),
        (("structure_diagnostic_mode",), "evaluate", "structure_diagnostic_mode"),
        (("structure_operability", "status"), "measured", "structure_operability"),
        (("delta_new", "source_id"), "foreign:P1", "source_id"),
    ],
)
def test_highrisk_profile_refuses_a_foreign_or_nonfrozen_hotspot_artifact(
    tmp_path, path, value, match,
):
    template, args, frozen = _highrisk_case(
        tmp_path, profile="highrisk_d2_k32_r40", run_max_head_calls=1024)
    artifact_path = Path(args.hotspot_json)
    payload = json.loads(artifact_path.read_text())
    node = payload
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    artifact_path.write_text(json.dumps(payload))

    with pytest.raises(MaterializeError, match=match):
        fill_config(template, args=args, frozen=frozen, runtime={})


def test_highrisk_profile_requires_absolute_snapshot_and_matching_model_sha(tmp_path):
    template, args, frozen = _highrisk_case(
        tmp_path, profile="highrisk_d2_k32_r40", run_max_head_calls=1024)
    args.esmfold2_model = "biohub/ESMFold2"
    with pytest.raises(MaterializeError, match="absolute.*snapshot"):
        fill_config(template, args=args, frozen=frozen, runtime={})

    snapshot = tmp_path / "other-snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"different ESMFold2 model")
    args.esmfold2_model = str(snapshot.resolve())
    with pytest.raises(MaterializeError, match="model.safetensors.*SHA|SHA.*model.safetensors"):
        fill_config(template, args=args, frozen=frozen, runtime={})


def test_highrisk_structure_identity_records_the_verified_absolute_snapshot(tmp_path):
    _, args, _ = _highrisk_case(
        tmp_path, profile="highrisk_d2_k32_r40", run_max_head_calls=1024)

    _, payload, _ = _structure_runtime_identity(args)

    snapshot = Path(args.esmfold2_model)
    assert payload["model_selector"] == str(snapshot.resolve())
    assert payload["local_model_snapshot_sha256"] == hashlib.sha256(
        (snapshot / "model.safetensors").read_bytes()).hexdigest()
    assert payload["site_packages_overlay"]["root"] == str(
        Path(args.esmfold2_site_packages).resolve())
    assert payload["site_packages_overlay"]["n_files"] == 2
    assert len(payload["site_packages_overlay"]["sha256"]) == 64
    assert payload["esmc_model_selector"] == str(
        Path(args.esmfold2_esmc_model).resolve())
    assert payload["ccd_path"] == str(Path(args.esmfold2_ccd_path).resolve())
    assert payload["model_snapshot"]["root"] == str(snapshot.resolve())
    assert payload["esmc_snapshot"]["root"] == str(
        Path(args.esmfold2_esmc_model).resolve())
    assert payload["ccd_sha256"] == hashlib.sha256(
        Path(args.esmfold2_ccd_path).read_bytes()).hexdigest()
