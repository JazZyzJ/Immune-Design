"""Unit contracts for the isolated, persistent ESMFold2 live worker."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from inverse_folding.evaluation.esmfold2_live import (
    READY_PREFIX,
    RESULT_PREFIX,
    ESMFold2LiveClient,
    build_worker_command,
    esmfold2_overlay_identity,
    hf_snapshot_identity,
)


def _fake_site_packages(root: Path) -> Path:
    site = root / "site-packages"
    (site / "esm" / "models" / "esmfold2").mkdir(parents=True)
    (site / "transformers" / "models" / "esmfold2").mkdir(parents=True)
    return site


def test_overlay_identity_is_content_bound_and_ignores_bytecode(tmp_path):
    site = _fake_site_packages(tmp_path)
    implementation = site / "esm" / "models" / "esmfold2" / "model.py"
    implementation.write_text("VERSION = 1\n")

    first = esmfold2_overlay_identity(site)
    assert first["root"] == str(site.resolve())
    assert first["n_files"] == 1
    assert len(first["sha256"]) == 64

    bytecode = implementation.parent / "__pycache__" / "model.cpython-312.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"ephemeral")
    assert esmfold2_overlay_identity(site) == first

    implementation.write_text("VERSION = 2\n")
    assert esmfold2_overlay_identity(site)["sha256"] != first["sha256"]


def test_hf_snapshot_identity_binds_required_load_files(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"x"}\n')
    (snapshot / "model.safetensors").write_bytes(b"weights")
    first = hf_snapshot_identity(snapshot, ("config.json", "model.safetensors"))
    assert first["root"] == str(snapshot.resolve())
    assert [row["path"] for row in first["files"]] == [
        "config.json", "model.safetensors",
    ]
    (snapshot / "config.json").write_text('{"model_type":"y"}\n')
    assert hf_snapshot_identity(
        snapshot, ("config.json", "model.safetensors"),
    )["sha256"] != first["sha256"]


def test_worker_command_uses_current_environment_python_and_isolated_overlay(tmp_path):
    site = _fake_site_packages(tmp_path)
    command = build_worker_command(
        site_packages=str(site),
        python_executable="/envs/immune-design/bin/python",
        device="cuda",
        model_name="biohub/ESMFold2",
        esmc_model="/models/esmc",
        ccd_path="/models/esmfold2/ccd.pkl",
        num_loops=3,
        num_sampling_steps=50,
        num_diffusion_samples=1,
        seed=7,
    )

    assert command[:4] == [
        "/envs/immune-design/bin/python",
        "-u",
        "-m",
        "inverse_folding.evaluation.esmfold2_live",
    ]
    assert command[4] == "worker"
    assert command[command.index("--site-packages") + 1] == str(site)
    assert command[command.index("--esmc-model") + 1] == "/models/esmc"
    assert command[command.index("--ccd-path") + 1] == "/models/esmfold2/ccd.pkl"
    assert "conda" not in command


class _FakeProcess:
    def __init__(self, command, *, stdout_text):
        self.command = command
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(stdout_text)
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminated = True
        self.returncode = -15


def test_client_loads_once_and_reuses_worker_for_predictions(tmp_path):
    site = _fake_site_packages(tmp_path)
    ready = {
        "python_executable": "/envs/immune-design/bin/python",
        "torch_version": "2.5.1+cu121",
        "esm_module": str(site / "esm" / "__init__.py"),
    }
    result = {
        "request_id": 1,
        "pdb_path": str(tmp_path / "cache" / "key.pdb"),
        "pLDDT": 91.25,
        "pTM": 0.88,
        "cache_hit": False,
    }
    stdout_text = (
        "third-party startup chatter\n"
        f"{READY_PREFIX}\t{json.dumps(ready)}\n"
        f"{RESULT_PREFIX}\t{json.dumps(result)}\n"
    )
    created = []

    def popen_factory(command, **kwargs):
        proc = _FakeProcess(command, stdout_text=stdout_text)
        created.append((proc, kwargs))
        return proc

    client = ESMFold2LiveClient(
        site_packages=str(site),
        device="cuda",
        python_executable="/envs/immune-design/bin/python",
        popen_factory=popen_factory,
    )
    pred = client.predict(
        sequence="MKTAYIAKQR",
        protein_id="P1",
        design_id="fusion",
        cache_dir=str(tmp_path / "cache"),
    )

    assert len(created) == 1
    assert created[0][1]["env"]["MKL_THREADING_LAYER"] == "GNU"
    assert client.metadata["torch_version"] == "2.5.1+cu121"
    assert pred["pLDDT"] == pytest.approx(91.25)
    request = json.loads(created[0][0].stdin.getvalue().splitlines()[0])
    assert request == {
        "op": "fold",
        "request_id": 1,
        "sequence": "MKTAYIAKQR",
        "protein_id": "P1",
        "design_id": "fusion",
        "cache_dir": str(tmp_path / "cache"),
        "msa_a3m_path": None,
    }
    client.close()


def test_predict_threads_optional_msa_path_into_request(tmp_path):
    site = _fake_site_packages(tmp_path)
    ready = {"torch_version": "2.5.1+cu121", "esm_module": str(site / "esm" / "__init__.py")}
    result = {"request_id": 1, "pdb_path": str(tmp_path / "k.pdb"), "pLDDT": 90.0, "cache_hit": False}
    stdout_text = f"{READY_PREFIX}\t{json.dumps(ready)}\n{RESULT_PREFIX}\t{json.dumps(result)}\n"
    created = []

    def popen_factory(command, **kwargs):
        proc = _FakeProcess(command, stdout_text=stdout_text)
        created.append(proc)
        return proc

    client = ESMFold2LiveClient(site_packages=str(site), device="cpu", popen_factory=popen_factory)
    client.predict(
        sequence="MKTAYIAKQR", protein_id="P1", design_id="d0",
        cache_dir=str(tmp_path / "cache"), msa_a3m_path="/path/to/msa.a3m",
    )
    request = json.loads(created[0].stdin.getvalue().splitlines()[0])
    assert request["msa_a3m_path"] == "/path/to/msa.a3m"
    client.close()


def test_cache_key_with_msa_differs_from_msa_free(tmp_path):
    from inverse_folding.evaluation.esmfold2_live import _cache_key_with_msa

    base = _cache_key_with_msa("P1", "MKTAYIAKQR", None)
    with_msa = _cache_key_with_msa("P1", "MKTAYIAKQR", "/path/to/msa.a3m")
    assert base != with_msa and with_msa.startswith(base + ".msa")


def test_client_surfaces_worker_error(tmp_path):
    site = _fake_site_packages(tmp_path)
    stdout_text = (
        f"{READY_PREFIX}\t{{}}\n"
        f"{RESULT_PREFIX}\t"
        + json.dumps({"request_id": 1, "error_type": "RuntimeError", "error": "fold failed"})
        + "\n"
    )

    client = ESMFold2LiveClient(
        site_packages=str(site),
        device="cpu",
        popen_factory=lambda command, **kwargs: _FakeProcess(command, stdout_text=stdout_text),
    )
    with pytest.raises(RuntimeError, match="fold failed"):
        client.predict(
            sequence="AAAA",
            protein_id="P1",
            design_id="d0",
            cache_dir=str(tmp_path / "cache"),
        )
    client.close()
