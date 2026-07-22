from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "submit_packed_pred.slurm"
COMPAT_SCRIPT = ROOT / "scripts" / "submit_tetramer_predict.slurm"


def _write_fake_protenix(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
assert args.pop(0) == "pred"

def option(name, default=None):
    if name in args:
        return args[args.index(name) + 1]
    prefix = name + "="
    for value in args:
        if value.startswith(prefix):
            return value[len(prefix):]
    return default

input_dir = Path(option("-i"))
out_dir = Path(option("-o"))
sample_count = int(option("--sample", option("-e", "5")))
seeds = option("--seeds", option("-s", "101")).split(",")
need_atom_confidence = option("--need_atom_confidence", "false").lower() == "true"
json_paths = sorted(input_dir.glob("*.json"))
with open(os.environ["FAKE_PROTENIX_LOG"], "a") as handle:
    handle.write(f"invoke inputs={len(json_paths)}\\n")

skip_name = os.environ.get("FAKE_PROTENIX_SKIP_NAME")
for json_path in json_paths:
    for job in json.loads(json_path.read_text()):
        name = job["name"]
        if name == skip_name:
            continue
        for seed in seeds:
            pred = out_dir / name / f"seed_{seed}" / "predictions"
            pred.mkdir(parents=True, exist_ok=True)
            for sample_idx in range(sample_count):
                (pred / f"{name}_sample_{sample_idx}.cif").write_text("data_fake\\n")
                (pred / f"{name}_summary_confidence_sample_{sample_idx}.json").write_text("{}")
                if need_atom_confidence:
                    (pred / f"{name}_full_data_sample_{sample_idx}.json").write_text("{}")
"""
    )
    path.chmod(0o755)


def _write_inputs(root: Path, count: int) -> tuple[Path, list[str]]:
    input_dir = root / "inputs"
    input_dir.mkdir()
    msa_dir = root / "msa"
    msa_dir.mkdir()
    paths = []
    names = []
    for idx in range(count):
        name = f"tetra_design_{idx}"
        design_msa_dir = msa_dir / name
        design_msa_dir.mkdir()
        paired_msa = design_msa_dir / "pairing.a3m"
        unpaired_msa = design_msa_dir / "non_pairing.a3m"
        paired_msa.write_text(">query\nAAAA\n")
        unpaired_msa.write_text(">query\nAAAA\n")
        json_path = input_dir / f"{name}-update-msa.json"
        json_path.write_text(
            json.dumps(
                [
                    {
                        "name": name,
                        "sequences": [
                            {
                                "proteinChain": {
                                    "sequence": "AAAA",
                                    "count": 4,
                                    "pairedMsaPath": str(paired_msa),
                                    "unpairedMsaPath": str(unpaired_msa),
                                }
                            }
                        ],
                    }
                ]
            )
        )
        paths.append(str(json_path))
        names.append(name)
    ready_list = root / "ready.txt"
    ready_list.write_text("\n".join(paths) + "\n")
    return ready_list, names


def _run(
    tmp_path: Path,
    ready_list: Path,
    pred_root: Path,
    *,
    task_id: int,
    task_count: int,
    skip_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    fake = tmp_path / "protenix-run"
    if not fake.exists():
        _write_fake_protenix(fake)
    env = os.environ.copy()
    env.update(
        {
            "PROTENIX_RUN": str(fake),
            "FAKE_PROTENIX_LOG": str(tmp_path / "invocations.log"),
            "GPU_TELEMETRY": "0",
            "SLURM_ARRAY_JOB_ID": "1234",
            "SLURM_ARRAY_TASK_ID": str(task_id),
            "SLURM_TMPDIR": str(tmp_path / "slurm-tmp"),
        }
    )
    if skip_name is not None:
        env["FAKE_PROTENIX_SKIP_NAME"] = skip_name
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(ready_list),
            str(pred_root),
            str(task_count),
            "--sample",
            "2",
            "--need_atom_confidence",
            "true",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_packed_prediction_uses_one_process_and_preserves_layout(tmp_path: Path) -> None:
    ready_list, names = _write_inputs(tmp_path, 3)
    pred_root = tmp_path / "pred"

    first = _run(tmp_path, ready_list, pred_root, task_id=0, task_count=1)
    assert first.returncode == 0, first.stderr + first.stdout
    assert (tmp_path / "invocations.log").read_text().splitlines() == ["invoke inputs=3"]

    for name in names:
        prediction_dir = pred_root / name / name / "seed_101" / "predictions"
        for sample_idx in range(2):
            assert (prediction_dir / f"{name}_sample_{sample_idx}.cif").is_file()
            assert (prediction_dir / f"{name}_summary_confidence_sample_{sample_idx}.json").is_file()
            assert (prediction_dir / f"{name}_full_data_sample_{sample_idx}.json").is_file()

    second = _run(tmp_path, ready_list, pred_root, task_id=0, task_count=1)
    assert second.returncode == 0, second.stderr + second.stdout
    assert (tmp_path / "invocations.log").read_text().splitlines() == ["invoke inputs=3"]
    assert "done=0 skip=3 fail=0" in second.stdout


def test_packed_prediction_strides_and_fails_on_missing_output(tmp_path: Path) -> None:
    ready_list, names = _write_inputs(tmp_path, 5)
    pred_root = tmp_path / "pred"

    result = _run(
        tmp_path,
        ready_list,
        pred_root,
        task_id=1,
        task_count=2,
        skip_name=names[3],
    )

    assert result.returncode != 0
    assert (tmp_path / "invocations.log").read_text().splitlines() == ["invoke inputs=2"]
    assert (pred_root / names[1] / names[1]).is_dir()
    assert not (pred_root / names[0]).exists()
    assert not (pred_root / names[2]).exists()
    assert not (pred_root / names[4]).exists()
    assert f"FAILED {names[3]}" in result.stdout
    assert "done=1 skip=0 fail=1" in result.stdout


def test_tetramer_array_entry_point_delegates_to_persistent_runner(tmp_path: Path) -> None:
    ready_list, names = _write_inputs(tmp_path, 5)
    pred_root = tmp_path / "pred"
    fake = tmp_path / "protenix-run"
    _write_fake_protenix(fake)
    env = os.environ.copy()
    env.update(
        {
            "PROTENIX_RUN": str(fake),
            "FAKE_PROTENIX_LOG": str(tmp_path / "invocations.log"),
            "GPU_TELEMETRY": "0",
            "SLURM_ARRAY_JOB_ID": "5678",
            "SLURM_ARRAY_TASK_ID": "1",
            "SLURM_ARRAY_TASK_COUNT": "2",
            "SLURM_TMPDIR": str(tmp_path / "slurm-tmp"),
            "PACKED_PRED_SCRIPT": str(SCRIPT),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(COMPAT_SCRIPT),
            str(ready_list),
            str(pred_root),
            "--sample",
            "2",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (tmp_path / "invocations.log").read_text().splitlines() == ["invoke inputs=2"]
    assert (pred_root / names[1] / names[1]).is_dir()
    assert (pred_root / names[3] / names[3]).is_dir()
    assert not (pred_root / names[0]).exists()


def test_missing_local_msa_fails_before_starting_protenix(tmp_path: Path) -> None:
    ready_list, _names = _write_inputs(tmp_path, 1)
    json_path = Path(ready_list.read_text().strip())
    payload = json.loads(json_path.read_text())
    payload[0]["sequences"][0]["proteinChain"]["pairedMsaPath"] = str(
        tmp_path / "missing" / "pairing.a3m"
    )
    json_path.write_text(json.dumps(payload))

    result = _run(tmp_path, ready_list, tmp_path / "pred", task_id=0, task_count=1)

    assert result.returncode != 0
    assert not (tmp_path / "invocations.log").exists()
    assert "local MSA preflight" in result.stdout
    assert "missing local MSA" in result.stderr


def test_completed_prediction_skips_even_if_msa_was_moved(tmp_path: Path) -> None:
    ready_list, _names = _write_inputs(tmp_path, 1)
    pred_root = tmp_path / "pred"
    first = _run(tmp_path, ready_list, pred_root, task_id=0, task_count=1)
    assert first.returncode == 0, first.stderr + first.stdout

    for msa_path in (tmp_path / "msa").rglob("*.a3m"):
        msa_path.unlink()
    second = _run(tmp_path, ready_list, pred_root, task_id=0, task_count=1)

    assert second.returncode == 0, second.stderr + second.stdout
    assert (tmp_path / "invocations.log").read_text().splitlines() == ["invoke inputs=1"]
    assert "done=0 skip=1 fail=0" in second.stdout


def test_resume_merges_direct_and_partial_legacy_layouts(tmp_path: Path) -> None:
    ready_list, names = _write_inputs(tmp_path, 1)
    name = names[0]
    pred_root = tmp_path / "pred"
    direct = pred_root / name / "seed_101" / "predictions"
    legacy = pred_root / name / name / "seed_101" / "predictions"
    direct.mkdir(parents=True)
    legacy.mkdir(parents=True)

    for root, sample_idx in ((legacy, 0), (direct, 1)):
        (root / f"{name}_sample_{sample_idx}.cif").write_text("data_fake\n")
        (root / f"{name}_summary_confidence_sample_{sample_idx}.json").write_text("{}")
        (root / f"{name}_full_data_sample_{sample_idx}.json").write_text("{}")

    result = _run(tmp_path, ready_list, pred_root, task_id=0, task_count=1)

    assert result.returncode == 0, result.stderr + result.stdout
    assert not direct.parent.exists()
    assert not (tmp_path / "invocations.log").exists()
    assert "done=0 skip=1 fail=0" in result.stdout
    for sample_idx in range(2):
        assert (legacy / f"{name}_sample_{sample_idx}.cif").is_file()
