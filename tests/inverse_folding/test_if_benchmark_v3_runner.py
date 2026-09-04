"""Runner identity/resume and tiny end-to-end contracts."""

from __future__ import annotations

import json

import pytest

from inverse_folding.evaluation.if_benchmark_v3.runner import (
    StageJournal,
    run_tiny_e2e,
)
def test_resume_requires_exact_identity_and_untampered_outputs(tmp_path):
    journal = StageJournal(tmp_path / "stage")
    calls = []
    code = tmp_path / "code.py"
    config = tmp_path / "config.json"
    source = tmp_path / "input.parquet"
    code.write_text("code")
    config.write_text("config")
    source.write_text("input")
    import hashlib
    identity = {
        "code": {"path": str(code), "sha256": hashlib.sha256(code.read_bytes()).hexdigest()},
        "config": {"path": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
        "inputs": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
    }

    def producer(stage_dir):
        calls.append(1)
        path = stage_dir / "result.txt"
        path.write_text("result")
        return [path]

    first = journal.run(identity, producer, resume=False)
    second = journal.run(identity, producer, resume=True)
    assert calls == [1]
    assert first == second
    config2 = tmp_path / "config2.json"
    config2.write_text("changed")
    changed_identity = {
        **identity,
        "config": {"path": str(config2), "sha256": hashlib.sha256(config2.read_bytes()).hexdigest()},
    }
    with pytest.raises(ValueError, match="identity mismatch"):
        journal.run(changed_identity, producer, resume=True)
    (tmp_path / "stage" / "result.txt").write_text("tampered")
    with pytest.raises(ValueError, match="output digest"):
        journal.run(identity, producer, resume=True)


def test_stage_journal_rejects_unregistered_extra_file(tmp_path):
    import hashlib
    code = tmp_path / "code"
    config = tmp_path / "config"
    source = tmp_path / "input"
    for path in (code, config, source):
        path.write_text(path.name)
    identity = {
        "code": {"path": str(code), "sha256": hashlib.sha256(code.read_bytes()).hexdigest()},
        "config": {"path": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
        "inputs": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
    }
    journal = StageJournal(tmp_path / "stage")
    def producer(stage_dir):
        result = stage_dir / "result"
        result.write_text("x")
        return [result]
    journal.run(identity, producer, resume=False)
    (tmp_path / "stage" / "extra").write_text("unregistered")
    with pytest.raises(ValueError, match="unregistered extra"):
        journal.run(identity, lambda _d: [], resume=True)


def test_old_cache_without_identity_fails_fast(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "legacy.parquet").write_text("old")
    journal = StageJournal(stage)
    with pytest.raises(ValueError, match="legacy/unbound cache"):
        journal.run({"input_sha256": "a" * 64}, lambda _path: [], resume=True)


def test_tiny_e2e_builds_versioned_candidate_and_does_not_touch_main(tmp_path):
    warehouse = tmp_path / "if_test_set"
    main = warehouse / "if_ready" / "main" / "HLA-DRB1_15_01"
    main.parent.mkdir(parents=True)
    candidate = run_tiny_e2e(
        build_root=warehouse / "builds", build_id="tiny-v3", main_alias=main,
    )
    assert candidate == warehouse / "builds" / "tiny-v3" / "HLA-DRB1_15_01"
    assert json.loads((candidate / "audit" / "release_gates.json").read_text()) == {
        str(i): "pass" for i in range(1, 14)
    }
    assert (candidate / "dataset_manifest.json").is_file()
    assert not main.exists() and not main.is_symlink()
    with pytest.raises(FileExistsError):
        run_tiny_e2e(build_root=warehouse / "builds", build_id="tiny-v3", main_alias=main)
