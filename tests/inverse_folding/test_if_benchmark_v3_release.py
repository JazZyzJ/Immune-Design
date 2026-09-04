"""v3 row/cross-artifact, manifest, and atomic candidate release gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from inverse_folding.evaluation.if_benchmark_v3.release import (
    finalize_candidate_directory,
    validate_release_tables,
    verify_dataset_manifest,
    write_dataset_manifest,
)
from inverse_folding.evaluation.if_benchmark_v3.runner import run_tiny_e2e
from inverse_folding.evaluation.if_benchmark_v3.evidence import (
    canonical_json,
    FAILED_C5_NMP_KEYS,
    FAILED_C5_SELECTION_KEYS,
    load_release_evidence_bundle,
    sha256_bytes,
    validate_c5_nmp_evidence,
    validate_chain_collision_dedup_ledgers,
    validate_descendant_inventory,
    validate_nmp_evidence,
    validate_nmp_install_evidence,
    validate_mmseqs_reference_evidence,
    validate_publication_context,
    validate_selection_replay,
    validate_c5_prior_rung_history,
    validate_approved_source_contract,
    validate_family_evidence,
    validate_tier1_release_sources,
    stable_family_cluster_id,
    write_c5_prior_rung_index,
    write_failed_c5_attempt_bundle,
)


def _fixture(tmp_path):
    warehouse = tmp_path / "if_test_set"
    main = warehouse / "if_ready" / "main" / "HLA-DRB1_15_01"
    main.parent.mkdir(parents=True)
    candidate = run_tiny_e2e(
        build_root=warehouse / "builds", build_id="release-fixture", main_alias=main
    )
    annotation = candidate / "annotations"
    audit = candidate / "audit"
    return {
        "primary": pd.read_parquet(candidate / "cohort_if_ready.parquet"),
        "diagnostic": pd.read_parquet(candidate / "tier1_overlap_diagnostic_if_ready.parquet"),
        "cath": pd.read_parquet(annotation / "cath.parquet"),
        "head": pd.read_parquet(annotation / "head.parquet"),
        "head_homology": pd.read_parquet(annotation / "head_homology.parquet"),
        "load": pd.read_parquet(audit / "load_coords.parquet"),
        "evidence_bundle_path": audit / "release_evidence.json",
        "candidate_root": candidate,
        "pdb_root": candidate / "pdbs_if_ready",
    }


def _validate(
    fixture, *, t1_target=1, t2_target=1, load_coords_fn=None, **overrides
):
    values = dict(fixture)
    values.update(overrides)
    return validate_release_tables(
        values["primary"], values["diagnostic"], values["cath"], values["head"],
        values["head_homology"], values["load"],
        evidence_bundle_path=values["evidence_bundle_path"],
        candidate_root=values["candidate_root"], pdb_root=values["pdb_root"],
        release_id="tiny-v3-release", allele="HLA-DRB1*15:01",
        tier1_target=t1_target, tier2_target=t2_target,
        load_coords_fn=load_coords_fn,
    )


def test_all_13_release_gates_pass_on_complete_fixture(tmp_path):
    report = _validate(_fixture(tmp_path))
    assert report == {str(i): "pass" for i in range(1, 14)}


def test_gate4_rejects_self_consistent_load_evidence_when_registered_replay_disagrees(
    tmp_path,
):
    fixture = _fixture(tmp_path)

    def forged_loader(_path, *, chain):
        assert chain == "A"
        return [None] * 99, "A" * 100

    with pytest.raises(ValueError, match="registered load_coords replay differs"):
        _validate(fixture, load_coords_fn=forged_loader)


@pytest.mark.parametrize("tamper", ["row", "sidecar", "count", "target"])
def test_release_tampering_fails_closed(tmp_path, tamper):
    fixture = _fixture(tmp_path)
    primary, cath = fixture["primary"].copy(), fixture["cath"].copy()
    t1_target, t2_target = 1, 1
    if tamper == "row":
        primary.loc[0, "sequence"] = "C" + primary.loc[0, "sequence"][1:]
    elif tamper == "sidecar":
        cath.loc[cath.protein_id == "T2B", "overlap_flag"] = True
    elif tamper == "count":
        primary = primary.iloc[:1].copy()
    elif tamper == "target":
        t2_target = 2
    with pytest.raises(ValueError):
        _validate(fixture, primary=primary, cath=cath,
                  t1_target=t1_target, t2_target=t2_target)


@pytest.mark.parametrize("tamper", [
    "mapping_index", "mapping_source_mismatch", "fake64_nmp", "family_id", "cath_best_hit",
])
def test_scientific_evidence_tampering_fails_closed(tmp_path, tamper):
    fixture = _fixture(tmp_path)
    primary = fixture["primary"].copy()
    diagnostic = fixture["diagnostic"].copy()
    if tamper == "mapping_index":
        primary.loc[0, "final_to_source_json"] = json.dumps([1000, *range(1, 100)])
    elif tamper == "mapping_source_mismatch":
        primary.loc[0, "source_sequence"] = "C" + primary.loc[0, "source_sequence"][1:]
        import hashlib
        primary.loc[0, "source_sequence_sha256"] = hashlib.sha256(
            primary.loc[0, "source_sequence"].encode()
        ).hexdigest()
    elif tamper == "fake64_nmp":
        primary.loc[primary.protein_id == "T2B", "nmp_evidence_sha256"] = "f" * 64
    elif tamper == "family_id":
        primary.loc[0, "family_cluster_id"] = "forged-family"
    else:
        diagnostic.loc[0, "cath_best_target"] = "FORGED"
    with pytest.raises(ValueError):
        _validate(fixture, primary=primary, diagnostic=diagnostic)


@pytest.mark.parametrize("legacy_key", [
    "source_snapshot_valid", "ledgers_replay_valid", "selection_replay_valid",
    "descendants_bound_or_pinned", "publication_target_immutable",
    "prior_release_recoverable",
])
def test_legacy_boolean_gate_forgery_is_not_an_accepted_api(tmp_path, legacy_key):
    fixture = _fixture(tmp_path)
    with pytest.raises(TypeError, match="stage_evidence"):
        validate_release_tables(
            fixture["primary"], fixture["diagnostic"], fixture["cath"], fixture["head"],
            fixture["head_homology"], fixture["load"],
            evidence_bundle_path=fixture["evidence_bundle_path"],
            candidate_root=fixture["candidate_root"], pdb_root=fixture["pdb_root"],
            release_id="tiny-v3-release", allele="HLA-DRB1*15:01",
            tier1_target=1, tier2_target=1, stage_evidence={legacy_key: True},
        )


def _resolved_evidence(fixture):
    return load_release_evidence_bundle(
        fixture["evidence_bundle_path"], candidate_root=fixture["candidate_root"]
    )


def _validate_selection_direct(fixture, evidence):
    return validate_selection_replay(
        evidence["selection"], ledger_paths=evidence["ledgers"],
        primary=fixture["primary"], tier2_target=1, allele="HLA-DRB1*15:01",
        rcsb_manifest_path=evidence["source"]["rcsb_manifest"],
        mmseqs_identity=validate_mmseqs_reference_evidence(evidence["mmseqs"]),
        candidate_root=fixture["candidate_root"],
        nmp_install_identity=validate_nmp_install_evidence(evidence["nmp_install"]),
        protocol_profile=evidence["protocol_profile"],
    )


def test_c5_membership_must_be_completely_accounted_by_c6_attempts(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    attempts_path = evidence["ledgers"]["chain_attempts"]
    attempts = pd.read_parquet(attempts_path)
    attempts[attempts.source_tier != "tier2"].to_parquet(attempts_path, index=False)
    with pytest.raises(ValueError, match="exactly account"):
        _validate_selection_direct(fixture, evidence)


@pytest.mark.parametrize("missing_scope", ["best_chain", "entire_fallback_entity"])
def test_c6_attempts_cannot_omit_any_frozen_entity_chain_edge(tmp_path, missing_scope):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    fallback_path = evidence["source"]["rcsb_manifest"].parent / "ordered_entity_fallbacks.parquet"
    fallbacks = pd.read_parquet(fallback_path)
    if missing_scope == "best_chain":
        edges = json.loads(fallbacks.loc[0, "entity_chain_edges_json"])
        edges.append({
            **edges[0], "label_asym_id": "A", "auth_asym_id": "A",
        })
        fallbacks.loc[0, "entity_chain_edges_json"] = json.dumps(edges)
    else:
        extra = fallbacks.iloc[[0]].copy()
        extra.loc[:, "rcsb_entity_id"] = "2CCC_1"
        extra.loc[:, "fallback_rank"] = 2
        extra.loc[:, "structure_revision_date"] = "2021-03-04"
        extra.loc[:, "entity_chain_edges_json"] = json.dumps([{
            "rcsb_entity_id": "2CCC_1", "pdb_id": "2CCC", "entity_id": "1",
            "label_asym_id": "C", "auth_asym_id": "C",
        }])
        fallbacks = pd.concat([fallbacks, extra], ignore_index=True)
    fallbacks.to_parquet(fallback_path, index=False)
    with pytest.raises(ValueError, match="complete ordered entity-chain attempt set"):
        _validate_selection_direct(fixture, evidence)


def test_tier1_chain_attempts_must_cover_every_source_valid_sifts_candidate(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    attempt_path = evidence["ledgers"]["chain_attempts"]
    attempts = pd.read_parquet(attempt_path)
    attempts = attempts[~(
        (attempts["source_tier"] == "tier1")
        & (attempts["source_uniprot_id"] == "P12345")
    )]
    attempts.to_parquet(attempt_path, index=False)
    rows = {
        **fixture["primary"].set_index("protein_id").to_dict("index"),
        **fixture["diagnostic"].set_index("protein_id").to_dict("index"),
    }
    with pytest.raises(ValueError, match="complete source-valid SIFTS materialization attempt set"):
        validate_tier1_release_sources(
            evidence["source"]["tier1_manifest"], rows=rows,
            allele="HLA-DRB1*15:01", chain_attempt_path=attempt_path,
            residue_manifest_path=evidence["source"]["tier1_residue_manifest"],
        )


def test_c5_capacity_boolean_cannot_replace_recomputed_capacity(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    path = evidence["selection"]["c5_expansion"]
    payload = json.loads(path.read_text())
    payload["attempts"][0]["capacity_sufficient"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="caller assertions"):
        _validate_selection_direct(fixture, evidence)


def test_c5_pool_cannot_drop_a_c2_source_cath_clean_group(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    path = evidence["selection"]["c5_eligible"]
    pd.read_parquet(path).iloc[:0].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="exact C2 source-CATH-clean"):
        _validate_selection_direct(fixture, evidence)


def test_source_cath_cannot_use_fake_reference_or_tool_digest(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    path = evidence["selection"]["source_cath_sidecar"]
    sidecar = pd.read_parquet(path)
    sidecar.loc[:, "reference_sha256"] = "f" * 64
    sidecar.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="frozen reference/tool"):
        _validate_selection_direct(fixture, evidence)


def test_multiple_c5_rungs_are_rejected_without_replayable_attempt_artifacts(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    path = evidence["selection"]["c5_expansion"]
    payload = json.loads(path.read_text())
    payload["preregistered_ladder"].append(2)
    payload["attempts"].append({"target": 2, "observed_final_clean_capacity": 1})
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exactly one independently replayed"):
        _validate_selection_direct(fixture, evidence)


def test_binned_sample_artifact_cannot_redefine_frozen_c5_or_c7_law(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    tampers = {
        "c5_result": [
            ("stage", "c7"), ("mode", "gaussian"), ("value_col", "coverage_fraction"),
            ("n_bins", 10), ("min_per_populated_bin", 1), ("seed", 7), ("mu", 0.2),
        ],
        "c7_result": [
            ("stage", "c5"), ("mode", "uniform"),
            ("value_col", "selection_source_coverage_fraction_15"),
            ("n_bins", 10), ("min_per_populated_bin", 0), ("seed", 7),
            ("mu", 0.9), ("sigma", 0.5),
        ],
    }
    for artifact, changes in tampers.items():
        path = evidence["selection"][artifact]
        original = json.loads(path.read_text())
        for field, value in changes:
            payload = json.loads(json.dumps(original))
            payload["result"][field] = value
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="frozen protocol"):
                _validate_selection_direct(fixture, evidence)
        path.write_text(json.dumps(original))


def test_cath_reference_fasta_must_replay_from_chain_set_train_split(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    fasta = evidence["mmseqs"]["cath_reference_fasta"]
    fasta.write_text(">CATH1\n" + "A" * 100 + "\n")
    manifest_path = evidence["mmseqs"]["cath_reference_manifest"]
    manifest = json.loads(manifest_path.read_text())
    import hashlib
    manifest["reference_fasta"]["size_bytes"] = fasta.stat().st_size
    manifest["reference_fasta"]["sha256"] = hashlib.sha256(fasta.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not replay"):
        validate_mmseqs_reference_evidence(evidence["mmseqs"])


def test_nmp_data_merkle_and_shared_install_identity_are_not_optional(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    install_manifest = json.loads(evidence["nmp_install"]["data_manifest"].read_text())
    raw_root = install_manifest["data_root"]
    data_root = Path(raw_root)
    if not data_root.is_absolute():
        data_root = evidence["nmp_install"]["data_manifest"].parent / data_root
    (data_root / "model.dat").write_bytes(b"tampered-model")
    with pytest.raises(ValueError, match="data/model directory Merkle"):
        validate_nmp_install_evidence(evidence["nmp_install"])


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("head", "fixed_epoch", 23),
        ("head", "allele", "HLA-DRB1*04:01"),
        ("cath", "chain_set_sha256", "f" * 64),
    ],
)
def test_approved_source_contract_rejects_wrong_epoch_allele_or_cath(
    tmp_path, section, field, value
):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    path = evidence["approved_sources"]["manifest"]
    payload = json.loads(path.read_text())
    payload[section][field] = value
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256")
    payload["manifest_sha256"] = sha256_bytes(canonical_json(unsigned))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="epoch24/CATH4.3"):
        validate_approved_source_contract(
            manifest_path=path, head_paths=evidence["head"],
            mmseqs_identity=validate_mmseqs_reference_evidence(evidence["mmseqs"]),
            protocol_profile="tiny_fixture",
        )


def test_approved_head_rejects_best_pt_even_with_matching_hash(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    best = tmp_path / "best.pt"
    best.write_bytes(evidence["head"]["checkpoint"].read_bytes())
    head_paths = {**evidence["head"], "checkpoint": best}
    with pytest.raises(ValueError, match="epoch24/CATH4.3"):
        validate_approved_source_contract(
            manifest_path=evidence["approved_sources"]["manifest"], head_paths=head_paths,
            mmseqs_identity=validate_mmseqs_reference_evidence(evidence["mmseqs"]),
            protocol_profile="tiny_fixture",
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("epoch", 23),
        ("global_step", 10054),
        ("config_hash", "wrong-config"),
        ("manifest_version", "v1.0"),
    ],
)
def test_production_head_contract_reads_strict_nested_checkpoint_metadata(
    tmp_path, field, bad_value
):
    import torch

    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    checkpoint = fixture["candidate_root"] / "annotations" / "epoch_24.production.pt"
    metadata = {
        "epoch": 24,
        "global_step": 10055,
        "config_hash": "ad6ac404027b",
        "manifest_version": "v1.1",
    }
    metadata[field] = bad_value
    # Misleading top-level values must never satisfy the production contract.
    torch.save(
        {
            "metadata": metadata,
            "epoch": 24,
            "global_step": 10055,
            "config_hash": "ad6ac404027b",
            "manifest_version": "v1.1",
            "model_state_dict": {},
            "optimizer_state_dict": {},
        },
        checkpoint,
    )
    head_paths = {**evidence["head"], "checkpoint": checkpoint}
    contract_path = evidence["approved_sources"]["manifest"]
    contract = json.loads(contract_path.read_text())
    contract["protocol_profile"] = "drb1501_production"
    contract["head"].update({
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_config_hash": "ad6ac404027b",
        "checkpoint_global_step": 10055,
        "checkpoint_manifest_version": "v1.1",
    })
    unsigned = dict(contract)
    unsigned.pop("manifest_sha256")
    contract["manifest_sha256"] = sha256_bytes(canonical_json(unsigned))
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="checkpoint metadata"):
        validate_approved_source_contract(
            manifest_path=contract_path, head_paths=head_paths,
            mmseqs_identity=validate_mmseqs_reference_evidence(evidence["mmseqs"]),
            protocol_profile="drb1501_production",
        )


def test_production_head_nested_checkpoint_metadata_positive(tmp_path):
    import hashlib
    import torch

    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    checkpoint = fixture["candidate_root"] / "annotations" / "epoch_24.production.pt"
    torch.save(
        {
            "metadata": {
                "epoch": 24,
                "global_step": 10055,
                "config_hash": "ad6ac404027b",
                "manifest_version": "v1.1",
            },
            "model_state_dict": {},
            "optimizer_state_dict": {},
        },
        checkpoint,
    )
    head_paths = {**evidence["head"], "checkpoint": checkpoint}
    contract_path = evidence["approved_sources"]["manifest"]
    contract = json.loads(contract_path.read_text())
    contract["protocol_profile"] = "drb1501_production"
    contract["head"].update({
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_config_hash": "ad6ac404027b",
        "checkpoint_global_step": 10055,
        "checkpoint_manifest_version": "v1.1",
    })
    unsigned = dict(contract)
    unsigned.pop("manifest_sha256")
    contract["manifest_sha256"] = sha256_bytes(canonical_json(unsigned))
    contract_path.write_text(json.dumps(contract))
    validated = validate_approved_source_contract(
        manifest_path=contract_path, head_paths=head_paths,
        mmseqs_identity=validate_mmseqs_reference_evidence(evidence["mmseqs"]),
        protocol_profile="drb1501_production",
    )
    assert validated["head"]["checkpoint_global_step"] == 10055


def test_family_partition_forgery_fails_bound_mmseqs_rerun(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    primary = fixture["primary"].copy()
    members = sorted(primary["protein_id"].astype(str))
    forged_id = stable_family_cluster_id(
        primary.set_index("protein_id").loc[members, "sequence_sha256"].astype(str).tolist()
    )
    primary.loc[:, "family_cluster_id"] = forged_id
    evidence["family"]["cluster_tsv"].write_text(
        "".join(f"{members[0]}\t{member}\n" for member in members)
    )
    help_text = "fixture complete easy-cluster help\n"
    help_path = evidence["family"]["parameters"].parent / "easy_cluster_help.txt"
    help_path.write_text(help_text)
    params = json.loads(evidence["family"]["parameters"].read_text())
    params["effective_defaults"] = {
        "contract": "complete_easy_cluster_help_stdout",
        "help_path": help_path.name,
        "help_sha256": hashlib.sha256(help_text.encode()).hexdigest(),
    }
    evidence["family"]["parameters"].write_text(json.dumps(params))

    def fake_run(command, **_kwargs):
        if command[-1] == "-h":
            return SimpleNamespace(returncode=0, stdout=help_text, stderr="")
        prefix = Path(command[3])
        Path(f"{prefix}_cluster.tsv").write_text(
            "".join(f"{member}\t{member}\n" for member in members)
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "inverse_folding.evaluation.if_benchmark_v3.evidence.subprocess.run", fake_run
    )
    with pytest.raises(ValueError, match="differs from bound MMseqs rerun"):
        validate_family_evidence(
            evidence["family"], primary=primary, diagnostic=fixture["diagnostic"],
            mmseqs_identity=validate_mmseqs_reference_evidence(evidence["mmseqs"]),
            protocol_profile="drb1501_production",
        )


def test_copied_failed_rung_recursively_replays_and_authorizes_next_rung(tmp_path):
    fixture = _fixture(tmp_path)
    retry_candidate = (
        tmp_path / "if_test_set" / "builds" / "retry-build" / "HLA-DRB1_15_01"
    )
    shutil.copytree(fixture["candidate_root"], retry_candidate)
    evidence = load_release_evidence_bundle(
        retry_candidate / "audit" / "release_evidence.json", candidate_root=retry_candidate
    )
    failed_root = retry_candidate / "audit" / "c5_attempts" / "1"
    failed_root.mkdir(parents=True)
    failed_bundle = failed_root / "failed_attempt.json"
    write_failed_c5_attempt_bundle(
        failed_bundle, candidate_root=retry_candidate, target=1,
        ledgers=evidence["ledgers"],
        selection={key: evidence["selection"][key] for key in FAILED_C5_SELECTION_KEYS},
        nmp={key: evidence["nmp"][key] for key in FAILED_C5_NMP_KEYS},
    )
    index_path = retry_candidate / "audit" / "c5_prior_rungs_retry.json"
    write_c5_prior_rung_index(
        index_path, candidate_root=retry_candidate, attempt_bundles={1: failed_bundle},
        ladder=[1, 2],
    )
    mmseqs_identity = validate_mmseqs_reference_evidence(evidence["mmseqs"])
    nmp_identity = validate_nmp_install_evidence(evidence["nmp_install"])
    measured = validate_c5_prior_rung_history(
        index_path=index_path, ladder=[1, 2], current_target=2, tier2_target=2,
        candidate_root=retry_candidate, allele="HLA-DRB1*15:01",
        rcsb_manifest_path=evidence["source"]["rcsb_manifest"],
        mmseqs_identity=mmseqs_identity, nmp_install_identity=nmp_identity,
        primary_columns=list(fixture["primary"].columns), protocol_profile="tiny_fixture",
    )
    assert measured == [{"target": 1, "measured_final_clean_capacity": 1}]

    # Missing or tampered copied bytes are rejected before a new rung can be authorized.
    chain_path = evidence["ledgers"]["chain_attempts"]
    original_chain = chain_path.read_bytes()
    chain_path.write_bytes(original_chain + b"tamper")
    with pytest.raises(ValueError, match="(size|SHA-256) mismatch"):
        validate_c5_prior_rung_history(
            index_path=index_path, ladder=[1, 2], current_target=2, tier2_target=2,
            candidate_root=retry_candidate, allele="HLA-DRB1*15:01",
            rcsb_manifest_path=evidence["source"]["rcsb_manifest"],
                mmseqs_identity=mmseqs_identity, nmp_install_identity=nmp_identity,
                primary_columns=list(fixture["primary"].columns), protocol_profile="tiny_fixture",
        )
    chain_path.write_bytes(original_chain)

    payload = json.loads(failed_bundle.read_text())
    payload["artifacts"]["nmp"]["shard_manifest"]["path"] = str(
        evidence["nmp"]["shard_manifest"]
    )
    unsigned = dict(payload)
    unsigned.pop("bundle_sha256")
    payload["bundle_sha256"] = sha256_bytes(canonical_json(unsigned))
    failed_bundle.write_text(json.dumps(payload))
    # Rebind only the outer copied-bundle identity so the inner absolute-path gate is exercised.
    write_c5_prior_rung_index(
        index_path, candidate_root=retry_candidate, attempt_bundles={1: failed_bundle},
        ladder=[1, 2],
    )
    with pytest.raises(ValueError, match="must be copied"):
        validate_c5_prior_rung_history(
            index_path=index_path, ladder=[1, 2], current_target=2, tier2_target=2,
            candidate_root=retry_candidate, allele="HLA-DRB1*15:01",
            rcsb_manifest_path=evidence["source"]["rcsb_manifest"],
            mmseqs_identity=mmseqs_identity, nmp_install_identity=nmp_identity,
            primary_columns=list(fixture["primary"].columns), protocol_profile="tiny_fixture",
        )


def test_full_nmp_must_cover_every_c7_eligible_row(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    path = evidence["selection"]["c7_eligible"]
    eligible = pd.read_parquet(path)
    extra = eligible.iloc[[0]].copy()
    extra.loc[:, "selection_unit_id"] = "t2-seq:extra"
    extra.loc[:, "protein_id"] = "EXTRA"
    extra.loc[:, "sequence"] = "A" * 100
    import hashlib
    extra.loc[:, "sequence_sha256"] = hashlib.sha256(b"A" * 100).hexdigest()
    pd.concat([eligible, extra], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="shard manifest/query"):
        validate_nmp_evidence(
            evidence["nmp"], c7_eligible_path=path,
            rows={**fixture["primary"].set_index("protein_id").to_dict("index"),
                  **fixture["diagnostic"].set_index("protein_id").to_dict("index")},
            allele="HLA-DRB1*15:01",
            install_identity=validate_nmp_install_evidence(evidence["nmp_install"]),
        )


def test_c5_nmp_rank_outside_percentile_range_fails(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    manifest_path = evidence["selection"]["c5_nmp_shard_manifest"]
    manifest = json.loads(manifest_path.read_text())
    shard = manifest["shards"][0]
    raw_path = manifest_path.parent / shard["files"]["raw_output"]["path"]
    windows_path = manifest_path.parent / shard["files"]["windows"]["path"]
    columns = ["protein_id", "peptide_length", "start_0b", "end_0b", "peptide", "rank_el"]
    raw = pd.read_csv(raw_path, sep="\t", names=columns, header=None)
    raw.loc[0, "rank_el"] = 101.0
    raw.to_csv(raw_path, sep="\t", header=False, index=False)
    raw.to_parquet(windows_path, index=False)
    import hashlib
    for label, path in (("raw_output", raw_path), ("windows", windows_path)):
        shard["files"][label]["size_bytes"] = path.stat().st_size
        shard["files"][label]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["shard_merkle_sha256"] = sha256_bytes(canonical_json(manifest["shards"]))
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(unsigned))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"outside \[0,100\]"):
        validate_c5_nmp_evidence(
            evidence["selection"], allele="HLA-DRB1*15:01",
            install_identity=validate_nmp_install_evidence(evidence["nmp_install"]),
        )


def test_chain_ledger_must_cover_every_released_unit_before_group_iteration(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    attempts_path = evidence["ledgers"]["chain_attempts"]
    attempts = pd.read_parquet(attempts_path)
    attempts[attempts.protein_id != "T1A"].to_parquet(attempts_path, index=False)
    rows = {
        **fixture["primary"].set_index("protein_id").to_dict("index"),
        **fixture["diagnostic"].set_index("protein_id").to_dict("index"),
    }
    with pytest.raises(ValueError, match="misses released"):
        validate_chain_collision_dedup_ledgers(evidence["ledgers"], rows=rows)


def test_source_snapshot_bundle_rejects_absolute_mutable_reference(tmp_path):
    fixture = _fixture(tmp_path)
    bundle_path = fixture["evidence_bundle_path"]
    payload = json.loads(bundle_path.read_text())
    relative = payload["source"]["rcsb_manifest"]["path"]
    payload["source"]["rcsb_manifest"]["path"] = str(
        (fixture["candidate_root"] / relative).resolve()
    )
    unsigned = dict(payload)
    unsigned.pop("bundle_sha256")
    payload["bundle_sha256"] = sha256_bytes(canonical_json(unsigned))
    bundle_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="self-contained"):
        load_release_evidence_bundle(bundle_path, candidate_root=fixture["candidate_root"])


def test_stage_output_and_nested_mmseq_paths_must_be_candidate_relative(tmp_path):
    fixture = _fixture(tmp_path)
    bundle_path = fixture["evidence_bundle_path"]
    payload = json.loads(bundle_path.read_text())
    relative = payload["selection"]["c5_result"]["path"]
    payload["selection"]["c5_result"]["path"] = str(
        (fixture["candidate_root"] / relative).resolve()
    )
    unsigned = dict(payload)
    unsigned.pop("bundle_sha256")
    payload["bundle_sha256"] = sha256_bytes(canonical_json(unsigned))
    bundle_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="candidate-relative"):
        load_release_evidence_bundle(bundle_path, candidate_root=fixture["candidate_root"])

    # A sidecar may not hide an external raw/query path behind an otherwise internal parquet.
    payload["selection"]["c5_result"]["path"] = relative
    payload["bundle_sha256"] = sha256_bytes(canonical_json({
        key: value for key, value in payload.items() if key != "bundle_sha256"
    }))
    bundle_path.write_text(json.dumps(payload))
    evidence = _resolved_evidence(fixture)
    sidecar_path = evidence["selection"]["source_cath_sidecar"]
    sidecar = pd.read_parquet(sidecar_path)
    sidecar.loc[:, "raw_output_path"] = str(
        (fixture["candidate_root"] / sidecar.loc[0, "raw_output_path"]).resolve()
    )
    sidecar.to_parquet(sidecar_path, index=False)
    with pytest.raises(ValueError, match="candidate-relative"):
        _validate_selection_direct(fixture, evidence)


def test_final_cath_relative_parent_escape_fails(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    sidecar_path = evidence["selection"]["final_cath_sidecar"]
    sidecar = pd.read_parquet(sidecar_path)
    sidecar.loc[:, "query_fasta_path"] = "../../outside.fasta"
    sidecar.to_parquet(sidecar_path, index=False)
    with pytest.raises(ValueError, match="escapes candidate root"):
        _validate_selection_direct(fixture, evidence)


@pytest.mark.parametrize(
    ("table_key", "fixture_key"), [("cath", "cath"), ("head_homology", "head_homology")]
)
def test_release_cath_and_head_raw_query_paths_cannot_escape_candidate(
    tmp_path, table_key, fixture_key
):
    fixture = _fixture(tmp_path)
    frame = fixture[fixture_key].copy()
    frame.loc[:, "raw_output_path"] = "../../outside.tsv"
    bundle_path = fixture["evidence_bundle_path"]
    bundle = json.loads(bundle_path.read_text())
    relative = bundle["release_tables"][table_key]["path"]
    persisted = fixture["candidate_root"] / relative
    frame.to_parquet(persisted, index=False)
    import hashlib
    bundle["release_tables"][table_key]["size_bytes"] = persisted.stat().st_size
    bundle["release_tables"][table_key]["sha256"] = hashlib.sha256(
        persisted.read_bytes()
    ).hexdigest()
    unsigned = dict(bundle)
    unsigned.pop("bundle_sha256")
    bundle["bundle_sha256"] = sha256_bytes(canonical_json(unsigned))
    bundle_path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="escapes candidate root"):
        _validate(fixture, **{fixture_key: frame})


def test_first_release_rejects_forged_prior_and_inventory_scope(tmp_path):
    fixture = _fixture(tmp_path)
    evidence = _resolved_evidence(fixture)
    context_path = evidence["publication"]["context"]
    context = json.loads(context_path.read_text())
    context["prior_manifest_sha256"] = "f" * 64
    context_path.write_text(json.dumps(context))
    with pytest.raises(ValueError, match="unexpectedly has a prior"):
        validate_publication_context(
            context_path, release_id="tiny-v3-release",
            candidate_root=fixture["candidate_root"],
        )

    context["prior_manifest_sha256"] = None
    context_path.write_text(json.dumps(context))
    inventory_path = evidence["descendants"]["inventory"]
    inventory = json.loads(inventory_path.read_text())
    inventory["scope_identity"]["roots"] = []
    inventory["scope_identity_sha256"] = sha256_bytes(canonical_json(inventory["scope_identity"]))
    inventory_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="frozen DRB1501"):
        validate_descendant_inventory(
            inventory_path, release_id="tiny-v3-release",
            publication_context_path=context_path,
        )


def test_manifest_detects_file_tamper_and_has_no_hash_cycle(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "cohort_if_ready.parquet").write_bytes(b"rows")
    manifest = write_dataset_manifest(candidate, {"dataset_release_id": "rel-v3"})
    assert "manifest_sha256" in manifest
    verify_dataset_manifest(candidate)
    (candidate / "cohort_if_ready.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact"):
        verify_dataset_manifest(candidate)


def test_atomic_finalize_never_moves_main_alias(tmp_path):
    candidate = tmp_path / "builds" / "b1" / "HLA-DRB1_15_01"
    candidate.mkdir(parents=True)
    (candidate / "payload").write_text("x")
    write_dataset_manifest(candidate, {"dataset_release_id": "rel-v3"})
    prior = tmp_path / "releases" / "old"
    prior.mkdir(parents=True)
    main = tmp_path / "if_ready" / "main"
    main.parent.mkdir()
    main.symlink_to(prior)
    before = os.readlink(main)
    final = finalize_candidate_directory(
        candidate, releases_root=tmp_path / "releases", release_id="rel-v3",
        allele_tag="HLA-DRB1_15_01", main_alias=main,
    )
    assert final.is_dir() and not candidate.exists()
    assert os.readlink(main) == before
