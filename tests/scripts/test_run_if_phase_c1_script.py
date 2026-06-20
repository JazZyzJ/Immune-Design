"""Phase C1 script helper tests."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from inverse_folding.evaluation.h_maps import sequence_md5
from inverse_folding.reference_flow.sampler import SamplerOutput
import scripts.run_if_phase_c1 as c1
from scripts.run_if_phase_c1 import (
    assert_h_maps_align_with_test_set,
    derive_h_shuffle_seed,
    parse_args,
)


def test_derive_h_shuffle_seed_varies_by_protein_and_design():
    base_seed = 123
    assert derive_h_shuffle_seed(base_seed, "p1", 0) == derive_h_shuffle_seed(base_seed, "p1", 0)
    assert derive_h_shuffle_seed(base_seed, "p1", 0) != derive_h_shuffle_seed(base_seed, "p1", 1)
    assert derive_h_shuffle_seed(base_seed, "p1", 0) != derive_h_shuffle_seed(base_seed, "p2", 0)


def _entries(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protein_id": pid,
                "sequence": seq,
                "sequence_length": len(seq),
                "pdb_path": f"{pid}.pdb",
            }
            for pid, seq in rows
        ]
    )


def _h_maps(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"protein_id": pid, "sequence_md5": md5, "sequence_length": 1, "h_raw": [0.0]}
            for pid, md5 in rows
        ]
    )


def test_alignment_passes_when_md5_matches():
    entries = _entries([("p1", "ACDE"), ("p2", "FGHIK")])
    h_maps = _h_maps(
        [
            ("p1", sequence_md5("ACDE")),
            ("p2", sequence_md5("FGHIK")),
        ]
    )
    assert_h_maps_align_with_test_set(entries=entries, h_maps_df=h_maps)


def test_alignment_fails_on_same_length_different_residues():
    entries = _entries([("p1", "ACDE")])
    h_maps = _h_maps([("p1", sequence_md5("DCBA"))])
    with pytest.raises(ValueError, match="sequence_md5"):
        assert_h_maps_align_with_test_set(entries=entries, h_maps_df=h_maps)


def test_alignment_fails_when_h_maps_missing_protein():
    entries = _entries([("p1", "ACDE"), ("p2", "FGHIK")])
    h_maps = _h_maps([("p1", sequence_md5("ACDE"))])
    with pytest.raises(ValueError, match="missing protein_id"):
        assert_h_maps_align_with_test_set(entries=entries, h_maps_df=h_maps)


def test_alignment_passes_with_extra_h_maps_proteins():
    """Extra proteins in h_maps that aren't in the test set are tolerated.

    A larger h-map corpus is fine; what matters is that every test-set
    protein has a matching, byte-identical row in h_maps.
    """
    entries = _entries([("p1", "ACDE")])
    h_maps = _h_maps(
        [
            ("p1", sequence_md5("ACDE")),
            ("extra", sequence_md5("ZZZZZZ")),
        ]
    )
    assert_h_maps_align_with_test_set(entries=entries, h_maps_df=h_maps)


def _required_cli_args() -> list[str]:
    return [
        "--checkpoint", "ckpt.pt",
        "--test-set-parquet", "test.parquet",
        "--pdb-root", "pdbs",
        "--h-maps-parquet", "h.parquet",
        "--allele", "DRB1_0101",
        "--config", "cfg.yaml",
        "--output-root", "out",
    ]


def test_parse_args_accepts_reference_flow_batch_size():
    args = parse_args([*_required_cli_args(), "--batch-size", "4"])
    assert args.batch_size == 4


def test_parse_args_rejects_nonpositive_reference_flow_batch_size():
    with pytest.raises(SystemExit):
        parse_args([*_required_cli_args(), "--batch-size", "0"])


def test_main_batch_size_uses_batched_sampler_path(monkeypatch, tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    config = tmp_path / "rf.yaml"
    config.write_text(
        "sampler:\n"
        "  n_steps: 1\n"
        "  seed: 17\n"
        "  temperature: 1.0\n"
        "  n_designs_per_protein: 1\n"
        "schedule:\n"
        "  base_form: linear\n"
        "amplification:\n"
        "  form: constant_one\n"
        "  h_source: h_processed\n"
    )
    entries = pd.DataFrame(
        [
            {"protein_id": "p1", "sequence": "ACDE", "sequence_length": 4, "pdb_path": "p1.pdb"},
            {"protein_id": "p2", "sequence": "FGHIK", "sequence_length": 5, "pdb_path": "p2.pdb"},
        ]
    )
    h_maps = pd.DataFrame(
        [
            {
                "protein_id": "p1",
                "sequence_md5": sequence_md5("ACDE"),
                "h_processed": [0.0] * 4,
            },
            {
                "protein_id": "p2",
                "sequence_md5": sequence_md5("FGHIK"),
                "h_processed": [0.0] * 5,
            },
        ]
    )

    class _Alphabet:
        mask_idx = 4
        unk_idx = 5
        padding_idx = 6
        cls_idx = 7
        eos_idx = 8

        def __len__(self) -> int:
            return 9

        def get_tok(self, idx: int) -> str:
            del idx
            return "A"

    fake_task = SimpleNamespace(alphabet=_Alphabet())
    import inverse_folding.reference_flow.runtime as runtime
    import inverse_folding.observability as observability

    monkeypatch.setattr(runtime, "load_if_task", lambda checkpoint, device: fake_task)
    monkeypatch.setattr(runtime, "load_test_entries", lambda path: entries.copy())
    monkeypatch.setattr(c1, "load_h_maps", lambda path: (h_maps.copy(), {"run_id": "h"}))
    monkeypatch.setattr(
        runtime,
        "prepare_backbone_batch",
        lambda **kwargs: (
            {"fake": torch.tensor([1])},
            [int(e["sequence_length"]) for e in kwargs["entries"]],
            [],
            [],
            list(range(len(kwargs["entries"]))),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "build_batched_dplm_denoiser_context",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(runtime, "make_batched_dplm_denoiser", lambda context: None)
    monkeypatch.setattr(observability, "init_wandb_from_args", lambda *a, **k: None)
    monkeypatch.setattr(observability, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(observability, "set_summary", lambda *a, **k: None)
    monkeypatch.setattr(observability, "finish_wandb", lambda *a, **k: None)

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        c1,
        "write_phase_c_outputs",
        lambda run_dir, rows, run_config, manifest: captured.update(
            {
                "run_dir": run_dir,
                "rows": rows,
                "run_config": run_config,
                "manifest": manifest,
            }
        ),
    )
    calls: list[list[int]] = []

    def fake_sample_batch(self, *, lanes, batched_denoiser, save_trajectories=False):
        del self, batched_denoiser, save_trajectories
        calls.append([lane.sequence_length for lane in lanes])
        return [
            SamplerOutput(
                tokens=torch.zeros(int(lane.sequence_length), dtype=torch.long),
                unmask_step_by_pos=[0] * int(lane.sequence_length),
                g_values=[1.0] * int(lane.sequence_length),
                trajectory_rows=[],
            )
            for lane in lanes
        ]

    monkeypatch.setattr(c1.PositionDependentDFMSampler, "sample_batch", fake_sample_batch)

    rc = c1.main(
        [
            "--checkpoint", str(checkpoint),
            "--test-set-parquet", str(tmp_path / "test.parquet"),
            "--pdb-root", str(tmp_path),
            "--h-maps-parquet", str(tmp_path / "h.parquet"),
            "--allele", "DRB1_0101",
            "--config", str(config),
            "--output-root", str(tmp_path / "out"),
            "--run-id", "batch-test",
            "--batch-size", "2",
            "--progress-every", "0",
        ]
    )

    assert rc == 0
    assert calls == [[5, 4]]
    rows = captured["rows"]
    assert [row["protein_id"] for row in rows] == ["p1", "p2"]
    assert [row["sequence"] for row in rows] == ["AAAA", "AAAAA"]
    assert captured["run_config"]["batch_size"] == 2
    assert captured["manifest"]["batch_size"] == 2
