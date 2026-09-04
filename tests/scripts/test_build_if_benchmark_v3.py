"""Executable v3 runner CLI smoke tests."""

from __future__ import annotations

from scripts.build_if_benchmark_v3 import main
def test_tiny_e2e_cli_creates_candidate_without_touching_main(tmp_path):
    warehouse = tmp_path / "if_test_set"
    main_alias = warehouse / "if_ready" / "main" / "HLA-DRB1_15_01"
    main_alias.parent.mkdir(parents=True)
    rc = main([
        "tiny-e2e", "--build-root", str(warehouse / "builds"),
        "--build-id", "tiny-v3", "--main-alias", str(main_alias),
    ])
    assert rc == 0
    assert (warehouse / "builds" / "tiny-v3" / "HLA-DRB1_15_01" /
            "dataset_manifest.json").is_file()
    assert not main_alias.exists()


def test_cli_rejects_path_traversal_build_id(tmp_path):
    assert main([
        "tiny-e2e", "--build-root", str(tmp_path / "builds"),
        "--build-id", "../escape", "--main-alias", str(tmp_path / "main"),
    ]) == 2
