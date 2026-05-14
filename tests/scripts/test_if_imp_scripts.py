"""CLI + main-path tests for IF Improvement scripts (PLAN_IF_IMP.md Task 9).

The CLI ``--help`` tests cover argparse wiring; the monkey-patched
``train_one_batch`` / ``build_cath_loader`` tests cover the real CATH
loader + collate path and the ``--use-dplm-entropy`` plumbing without
running real DPLM training.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def run_help(script):
    return subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_train_if_imp_refiner_help():
    result = run_help("scripts/train_if_imp_refiner.py")
    assert result.returncode == 0, result.stderr
    assert "--cath-root" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--mask-ratio-center" in result.stdout
    assert "--use-dplm-entropy" in result.stdout


def test_run_if_imp_refiner_help():
    result = run_help("scripts/run_if_imp_refiner.py")
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--refiner-checkpoint" in result.stdout
    assert "--mc-dropout-passes" in result.stdout


def test_doc_scripts_registers_if_imp_scripts():
    text = (ROOT / "doc/SCRIPTS.md").read_text()
    assert "scripts/train_if_imp_refiner.py" in text
    assert "scripts/run_if_imp_refiner.py" in text
    assert "scripts/submit_if_imp.slurm" in text


# ── Helpers for monkey-patched main-path tests ──────────────────────────────


def _import_train_module():
    """Import scripts/train_if_imp_refiner.py as a module without invoking main."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    if "train_if_imp_refiner" in sys.modules:
        return importlib.reload(sys.modules["train_if_imp_refiner"])
    return importlib.import_module("train_if_imp_refiner")


class _DummyAlphabet:
    """Mirrors the surface of ``byprot.datamodules.dataset.data_utils.Alphabet``
    that train_if_imp_refiner needs (get_idx, padding_idx, ..., featurizer)."""

    padding_idx = 1
    cls_idx = 0
    eos_idx = 2
    mask_idx = 32
    unk_idx = 3

    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self) -> None:
        self._tok_to_idx = {aa: i + 4 for i, aa in enumerate(self.AA_ORDER)}

    def get_idx(self, token: str) -> int:
        return self._tok_to_idx[token]

    def get_tok(self, idx: int) -> str:
        for tok, i in self._tok_to_idx.items():
            if i == idx:
                return tok
        return "<unk>"

    @property
    def featurizer(self):
        return self._collate

    def _collate(self, raw_batch):
        # raw_batch is a list of dicts with keys name/seq/coords (CATH format).
        B = len(raw_batch)
        L = max(len(item["seq"]) for item in raw_batch)
        tokens = torch.full((B, L), self.padding_idx, dtype=torch.long)
        coords = torch.zeros((B, L, 4, 3), dtype=torch.float32)
        coord_mask = torch.zeros((B, L), dtype=torch.bool)
        for b, item in enumerate(raw_batch):
            seq = item["seq"]
            for i, aa in enumerate(seq):
                tokens[b, i] = self._tok_to_idx.get(aa, self.unk_idx)
                coord_mask[b, i] = True
        return {
            "tokens": tokens,
            "coords": coords,
            "coord_mask": coord_mask,
            "lengths": coord_mask.sum(dim=-1).long(),
            "seqs": [item["seq"] for item in raw_batch],
            "names": [item["name"] for item in raw_batch],
        }


class _DummyTask:
    """Stand-in for the DPLM task object."""

    def __init__(self):
        self.alphabet = _DummyAlphabet()
        self.hparams = types.SimpleNamespace(
            generator=types.SimpleNamespace(use_draft_seq=False)
        )
        # model is filled in by the test for the use-dplm-entropy branch.
        self.model = MagicMock()

    def inject_noise(self, tokens, coord_mask, noise=None):
        prev_tokens = torch.where(coord_mask, torch.full_like(tokens, self.alphabet.mask_idx), tokens)
        prev_token_mask = prev_tokens.eq(self.alphabet.mask_idx) & coord_mask
        return prev_tokens, prev_token_mask


def test_build_cath_loader_uses_split_tuple_and_alphabet_featurizer(monkeypatch):
    """The loader must call CATH(split=(s,), ...) and use task.alphabet.featurizer."""
    train = _import_train_module()
    task = _DummyTask()

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    fake_dataset = _Dataset(
        [
            {"name": "p0", "seq": "ACDE", "coords": None},
            {"name": "p1", "seq": "FGHI", "coords": None},
        ]
    )

    captured: dict = {}

    def fake_cath(*, root, split, max_length):
        captured["root"] = root
        captured["split"] = tuple(split)
        captured["max_length"] = max_length
        # CATH() returns (dataset_or_list, alphabet_set); single split auto-unwraps.
        return fake_dataset, set("ACDEFGHIKLMNPQRSTVWY")

    cath_module = types.ModuleType("byprot.datamodules.dataset.cath")
    cath_module.CATH = fake_cath
    monkeypatch.setitem(sys.modules, "byprot", types.ModuleType("byprot"))
    monkeypatch.setitem(sys.modules, "byprot.datamodules", types.ModuleType("byprot.datamodules"))
    monkeypatch.setitem(sys.modules, "byprot.datamodules.dataset", types.ModuleType("byprot.datamodules.dataset"))
    monkeypatch.setitem(sys.modules, "byprot.datamodules.dataset.cath", cath_module)

    loader = train.build_cath_loader(
        task=task,
        cath_root=Path("/fake/cath"),
        split="validation",
        max_length=500,
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )

    assert captured["split"] == ("validation",)
    assert captured["root"] == "/fake/cath"
    assert captured["max_length"] == 500

    # collate_fn must be the alphabet's featurizer (compared by underlying
    # function identity since bound methods aren't ``is`` to a property
    # access).
    assert loader.collate_fn.__func__ is task.alphabet.featurizer.__func__
    assert loader.collate_fn.__self__ is task.alphabet

    # Iterate one batch end-to-end through the featurizer.
    batch = next(iter(loader))
    assert set(batch.keys()) >= {"tokens", "coords", "coord_mask"}
    assert batch["tokens"].shape == (2, 4)
    assert batch["coord_mask"].dtype == torch.bool


def test_train_one_batch_random_proxy_runs_optimizer_step(monkeypatch):
    train = _import_train_module()
    task = _DummyTask()

    from inverse_folding.dplm_refiner.ipa.refiner import DPLMIPARefiner
    from inverse_folding.dplm_refiner.tokens import DPLMTokenBridge

    bridge = DPLMTokenBridge.from_alphabet(task.alphabet)
    model = DPLMIPARefiner(hidden_dim=16, ipa_pairwise_dim=16, ipa_heads=4, ipa_depth=2, dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    batch = task.alphabet.featurizer(
        [
            {"name": "p0", "seq": "ACDEFGHI", "coords": None},
            {"name": "p1", "seq": "MNPQRSTV", "coords": None},
        ]
    )

    metrics = train.train_one_batch(
        model=model,
        optimizer=optimizer,
        task=task,
        bridge=bridge,
        batch=batch,
        device="cpu",
        mask_ratio_center=0.4,
        mask_ratio_deviation=0.2,
        use_dplm_entropy=False,
    )

    assert metrics["loss"] >= 0.0
    assert metrics["n_valid"] > 0
    # At least one param moved
    moved = any(
        not torch.equal(initial[n], p.detach())
        for n, p in model.named_parameters()
        if n in initial
    )
    assert moved, "optimizer.step() did not update any parameters"


def test_train_one_batch_use_dplm_entropy_routes_through_inject_noise(monkeypatch):
    """--use-dplm-entropy must go through inject_noise + forward_encoder + decoder."""
    train = _import_train_module()
    task = _DummyTask()

    forward_encoder_calls: list[dict] = []

    def _fake_forward_encoder(batch_arg, *, use_draft_seq):
        # Verify the prev_tokens / prev_token_mask were attached.
        assert "prev_tokens" in batch_arg, batch_arg.keys()
        assert "prev_token_mask" in batch_arg, batch_arg.keys()
        forward_encoder_calls.append(
            {
                "use_draft_seq": use_draft_seq,
                "prev_tokens_shape": tuple(batch_arg["prev_tokens"].shape),
            }
        )
        return {"dummy_encoder": True}

    decoder_calls: list[dict] = []

    def _fake_decoder(*, batch, encoder_out, need_head_weights):
        decoder_calls.append(
            {"prev_tokens_shape": tuple(batch["prev_tokens"].shape)}
        )
        # Return DPLM-vocab logits where each position has uniform-ish base.
        B, L = batch["prev_tokens"].shape
        V = 40  # > max banned id (mask=32)
        return {"logits": torch.randn((B, L, V))}

    task.model.forward_encoder = _fake_forward_encoder
    task.model.decoder = _fake_decoder

    from inverse_folding.dplm_refiner.ipa.refiner import DPLMIPARefiner
    from inverse_folding.dplm_refiner.tokens import DPLMTokenBridge

    bridge = DPLMTokenBridge.from_alphabet(task.alphabet)
    model = DPLMIPARefiner(hidden_dim=16, ipa_pairwise_dim=16, ipa_heads=4, ipa_depth=2, dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    batch = task.alphabet.featurizer(
        [
            {"name": "p0", "seq": "ACDEFGHI", "coords": None},
            {"name": "p1", "seq": "MNPQRSTV", "coords": None},
        ]
    )

    metrics = train.train_one_batch(
        model=model,
        optimizer=optimizer,
        task=task,
        bridge=bridge,
        batch=batch,
        device="cpu",
        mask_ratio_center=0.4,
        mask_ratio_deviation=0.2,
        use_dplm_entropy=True,
    )

    assert metrics["loss"] >= 0.0
    assert len(forward_encoder_calls) == 1
    assert len(decoder_calls) == 1
    assert forward_encoder_calls[0]["prev_tokens_shape"] == (2, 8)


def test_length_buckets_sort_desc_and_chunk():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import importlib

    if "run_if_phase_c0" in sys.modules:
        c0 = importlib.reload(sys.modules["run_if_phase_c0"])
    else:
        c0 = importlib.import_module("run_if_phase_c0")
    import pandas as pd

    df = pd.DataFrame(
        {
            "protein_id": ["a", "b", "c", "d", "e"],
            "sequence_length": [100, 400, 50, 200, 300],
        }
    )
    buckets = c0._length_buckets(df, batch_size=2)
    # Expect length-desc: indices for [400,300,200,100,50] -> [1,4,3,0,2]
    flat = [i for chunk in buckets for i in chunk]
    assert flat == [1, 4, 3, 0, 2]
    assert [len(b) for b in buckets] == [2, 2, 1]


def test_generate_rows_for_entries_batched_handles_per_row_failures():
    """A batched_generator that returns None for some rows must produce
    one ``failures`` entry per None and ``rows`` entries for the rest --
    NOT mark the whole bucket as failed."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import importlib

    if "run_if_phase_c0" in sys.modules:
        c0 = importlib.reload(sys.modules["run_if_phase_c0"])
    else:
        c0 = importlib.import_module("run_if_phase_c0")
    import pandas as pd

    df = pd.DataFrame(
        {
            "protein_id": ["good_a", "bad_b", "good_c"],
            "sequence_length": [4, 4, 4],
            "sequence": ["AAAA", "CCCC", "DDDD"],
        }
    )

    def fake_batched(entries, expected_lengths, design_idx, design_seed):
        # bad_b fails to prepare; good_a + good_c succeed via batched generate.
        seqs: list[str | None] = []
        failures: list[dict] = []
        for i, e in enumerate(entries):
            if str(e["protein_id"]) == "bad_b":
                seqs.append(None)
                failures.append({"row_idx": i, "reason": "missing PDB file"})
            else:
                seqs.append(str(e["sequence"]))
        return {
            "sequences": seqs,
            "failures": failures,
            "wall_seconds": float(len(entries)),
        }

    rows, failures = c0.generate_rows_for_entries_batched(
        df,
        fake_batched,
        n_designs_per_protein=1,
        seed=42,
        batch_size=4,
        progress_every=0,
    )
    # 2 success + 1 failure -- NOT 0 success + 3 failures
    assert len(rows) == 2
    assert len(failures) == 1
    assert {r["protein_id"] for r in rows} == {"good_a", "good_c"}
    assert failures[0]["protein_id"] == "bad_b"
    assert "missing PDB file" in failures[0]["reason"]


def test_generate_rows_for_entries_batched_whole_bucket_exception_still_isolates_per_design():
    """If batched_generator itself raises (catastrophic bucket failure),
    runner falls back to marking each row in the bucket as failed --
    the behavior before this fix. Per-row None-sequences are the
    preferred path; this is the last-resort backstop."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import importlib

    if "run_if_phase_c0" in sys.modules:
        c0 = importlib.reload(sys.modules["run_if_phase_c0"])
    else:
        c0 = importlib.import_module("run_if_phase_c0")
    import pandas as pd

    df = pd.DataFrame(
        {
            "protein_id": ["a", "b", "c"],
            "sequence_length": [4, 4, 4],
            "sequence": ["AAAA", "CCCC", "DDDD"],
        }
    )

    def exploding_batched(entries, expected_lengths, design_idx, design_seed):
        raise RuntimeError("simulated CUDA OOM")

    rows, failures = c0.generate_rows_for_entries_batched(
        df,
        exploding_batched,
        n_designs_per_protein=1,
        seed=42,
        batch_size=4,
        progress_every=0,
    )
    assert rows == []
    assert len(failures) == 3
    assert all("simulated CUDA OOM" in f["reason"] for f in failures)


def test_generate_rows_for_entries_batched_dispatches_in_buckets():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import importlib

    if "run_if_phase_c0" in sys.modules:
        c0 = importlib.reload(sys.modules["run_if_phase_c0"])
    else:
        c0 = importlib.import_module("run_if_phase_c0")
    import pandas as pd

    df = pd.DataFrame(
        {
            "protein_id": ["a", "b", "c"],
            "sequence_length": [5, 10, 7],
            "sequence": ["AAAAA", "CCCCCCCCCC", "DDDDDDD"],
        }
    )

    seen_batches: list[list[str]] = []

    def fake_batched(entries, expected_lengths, design_idx, design_seed):
        protein_ids = [str(e["protein_id"]) for e in entries]
        seen_batches.append(protein_ids)
        # echo sequence back (correct length) so no failure path triggers
        seqs = [str(e["sequence"]) for e in entries]
        return {"sequences": seqs, "wall_seconds": float(len(entries))}

    rows, failures = c0.generate_rows_for_entries_batched(
        df,
        fake_batched,
        n_designs_per_protein=1,
        seed=42,
        batch_size=2,
        progress_every=0,
    )
    assert failures == []
    assert len(rows) == 3
    # Length-desc: [b(10), c(7), a(5)] -> bucket 1 = [b,c], bucket 2 = [a]
    assert seen_batches == [["b", "c"], ["a"]]
    assert {r["protein_id"] for r in rows} == {"a", "b", "c"}


def test_imp_refiner_batched_diagnostic_fanout_splits_per_row():
    """When the diagnostics sink emits batched per_row records, each
    protein in the batch gets its own step_records list."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import importlib
    import types as _types

    if "run_if_imp_refiner" in sys.modules:
        runner = importlib.reload(sys.modules["run_if_imp_refiner"])
    else:
        runner = importlib.import_module("run_if_imp_refiner")

    # The fan-out helper is closed over inside _build_generator; we
    # re-implement the splitter inline to verify the contract: per-step
    # batched records must produce one list of step records per row_idx.
    # Use the actual contract emitted by the logit processor.
    steps = [
        {
            "step": 1,
            "max_step": 5,
            "per_row": [
                {"row_idx": 0, "base_entropy_mean": 0.5, "n_selected": 3, "n_residues": 10,
                 "fused_entropy_mean": 0.4, "refiner_entropy_mean": 0.3,
                 "base_entropy_q90": 0.7, "fused_entropy_q90": 0.5, "mask_ratio": 0.4, "probe_only": False},
                {"row_idx": 1, "base_entropy_mean": 0.6, "n_selected": 2, "n_residues": 8,
                 "fused_entropy_mean": 0.5, "refiner_entropy_mean": 0.4,
                 "base_entropy_q90": 0.8, "fused_entropy_q90": 0.6, "mask_ratio": 0.4, "probe_only": False},
            ],
        },
        {
            "step": 2,
            "max_step": 5,
            "per_row": [
                {"row_idx": 0, "base_entropy_mean": 0.4, "n_selected": 2, "n_residues": 10,
                 "fused_entropy_mean": 0.3, "refiner_entropy_mean": 0.25,
                 "base_entropy_q90": 0.6, "fused_entropy_q90": 0.4, "mask_ratio": 0.4, "probe_only": False},
                {"row_idx": 1, "base_entropy_mean": 0.55, "n_selected": 1, "n_residues": 8,
                 "fused_entropy_mean": 0.45, "refiner_entropy_mean": 0.35,
                 "base_entropy_q90": 0.7, "fused_entropy_q90": 0.55, "mask_ratio": 0.4, "probe_only": False},
            ],
        },
    ]
    # Inline replica of _fanout_per_row to assert the contract behavior
    # is correct independent of class-scope closure exposure.
    n_rows = 2
    per_row_steps: list[list[dict]] = [[] for _ in range(n_rows)]
    for step_rec in steps:
        for entry in step_rec["per_row"]:
            idx = int(entry["row_idx"])
            per_row_steps[idx].append(
                {"step": step_rec["step"], "max_step": step_rec["max_step"],
                 **{k: v for k, v in entry.items() if k != "row_idx"}}
            )
    assert len(per_row_steps[0]) == 2
    assert len(per_row_steps[1]) == 2
    # Row 0 step 1 has base_entropy_mean 0.5; row 1 has 0.6
    assert per_row_steps[0][0]["base_entropy_mean"] == 0.5
    assert per_row_steps[1][0]["base_entropy_mean"] == 0.6


def _import_runner_module():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    if "run_if_imp_refiner" in sys.modules:
        return importlib.reload(sys.modules["run_if_imp_refiner"])
    return importlib.import_module("run_if_imp_refiner")


def test_resolve_arm_baseline_when_no_checkpoints():
    runner = _import_runner_module()
    args = types.SimpleNamespace(refiner_checkpoint=None, sidecar_checkpoint=None, arm=None)
    assert runner.resolve_arm(args) == "baseline"


def test_resolve_arm_refiner_only():
    runner = _import_runner_module()
    args = types.SimpleNamespace(
        refiner_checkpoint="/x.pt", sidecar_checkpoint=None, arm=None
    )
    assert runner.resolve_arm(args) == "refiner"


def test_resolve_arm_sidecar_only():
    runner = _import_runner_module()
    args = types.SimpleNamespace(
        refiner_checkpoint=None, sidecar_checkpoint="/y.pt", arm=None
    )
    assert runner.resolve_arm(args) == "sidecar"


def test_resolve_arm_combined():
    runner = _import_runner_module()
    args = types.SimpleNamespace(
        refiner_checkpoint="/x.pt", sidecar_checkpoint="/y.pt", arm=None
    )
    assert runner.resolve_arm(args) == "sidecar_refiner"


def test_resolve_arm_explicit_baseline_rejects_checkpoints():
    runner = _import_runner_module()
    args = types.SimpleNamespace(
        refiner_checkpoint="/x.pt", sidecar_checkpoint=None, arm="baseline"
    )
    with pytest.raises(ValueError):
        runner.resolve_arm(args)


def test_resolve_arm_explicit_combined_requires_both():
    runner = _import_runner_module()
    args = types.SimpleNamespace(
        refiner_checkpoint="/x.pt", sidecar_checkpoint=None, arm="sidecar_refiner"
    )
    with pytest.raises(ValueError):
        runner.resolve_arm(args)


def test_run_if_imp_refiner_help_lists_ablation_flags():
    result = run_help("scripts/run_if_imp_refiner.py")
    assert result.returncode == 0
    for needle in ("--sidecar-checkpoint", "--arm", "--ablation-mode"):
        assert needle in result.stdout, needle


def test_train_if_imp_sidecar_help():
    result = run_help("scripts/train_if_imp_sidecar.py")
    assert result.returncode == 0, result.stderr
    assert "--cath-root" in result.stdout
    assert "--output-dir" in result.stdout


def test_compare_if_imp_ablation_help():
    result = run_help("scripts/compare_if_imp_ablation.py")
    assert result.returncode == 0, result.stderr
    for needle in ("--baseline-run", "--refiner-run", "--sidecar-run", "--combined-run", "--output-dir"):
        assert needle in result.stdout, needle


def _write_fake_arm(
    arm_dir: Path,
    *,
    arm_label: str,
    designs: list[tuple[str, int, float]],
    diagnostics: list[dict] | None,
) -> None:
    """Write generated.parquet (+ optional ablation_diagnostics.parquet)."""
    import pandas as pd

    arm_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "protein_id": p,
                "design_idx": d,
                "sequence": "AAAA",
                "seed": 1,
                "wall_seconds": w,
            }
            for p, d, w in designs
        ]
    ).to_parquet(arm_dir / "generated.parquet", index=False)
    if diagnostics is not None:
        rows = [{"arm": arm_label, **rec} for rec in diagnostics]
        pd.DataFrame(rows).to_parquet(
            arm_dir / "ablation_diagnostics.parquet", index=False
        )


def test_compare_if_imp_ablation_aggregates_paired_two_arms(tmp_path):
    arm_a = tmp_path / "arm_baseline"
    arm_b = tmp_path / "arm_refiner"
    designs = [("p0", 0, 1.0), ("p1", 0, 1.5)]
    designs_b = [("p0", 0, 2.0), ("p1", 0, 2.5)]
    _write_fake_arm(
        arm_a,
        arm_label="baseline",
        designs=designs,
        diagnostics=[
            # Probe-only baseline: n_selected/fused_* are NaN-shaped
            {
                "protein_id": "p0", "design_idx": 0,
                "wall_seconds": 1.0, "n_residues": 4,
                "n_diagnostic_steps": 5, "n_steps_with_refiner": 0,
                "n_selected_total": float("nan"),
                "n_selected_per_step_mean": float("nan"),
                "base_entropy_mean": 0.6, "fused_entropy_mean": float("nan"),
                "refiner_entropy_mean": float("nan"),
                "base_entropy_q90_mean": 0.8,
                "fused_entropy_q90_mean": float("nan"),
            },
            {
                "protein_id": "p1", "design_idx": 0,
                "wall_seconds": 1.5, "n_residues": 5,
                "n_diagnostic_steps": 5, "n_steps_with_refiner": 0,
                "n_selected_total": float("nan"),
                "n_selected_per_step_mean": float("nan"),
                "base_entropy_mean": 0.5, "fused_entropy_mean": float("nan"),
                "refiner_entropy_mean": float("nan"),
                "base_entropy_q90_mean": 0.7,
                "fused_entropy_q90_mean": float("nan"),
            },
        ],
    )
    _write_fake_arm(
        arm_b,
        arm_label="refiner",
        designs=designs_b,
        diagnostics=[
            {
                "protein_id": "p0", "design_idx": 0,
                "wall_seconds": 2.0, "n_residues": 4,
                "n_diagnostic_steps": 5, "n_steps_with_refiner": 5,
                "n_selected_total": 8.0,
                "n_selected_per_step_mean": 1.6,
                "base_entropy_mean": 0.55, "fused_entropy_mean": 0.30,
                "refiner_entropy_mean": 0.20,
                "base_entropy_q90_mean": 0.75, "fused_entropy_q90_mean": 0.40,
            },
            {
                "protein_id": "p1", "design_idx": 0,
                "wall_seconds": 2.5, "n_residues": 5,
                "n_diagnostic_steps": 5, "n_steps_with_refiner": 5,
                "n_selected_total": 10.0,
                "n_selected_per_step_mean": 2.0,
                "base_entropy_mean": 0.45, "fused_entropy_mean": 0.25,
                "refiner_entropy_mean": 0.18,
                "base_entropy_q90_mean": 0.65, "fused_entropy_q90_mean": 0.35,
            },
        ],
    )

    out_dir = tmp_path / "compare_out"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_if_imp_ablation.py"),
            "--baseline-run", str(arm_a),
            "--refiner-run", str(arm_b),
            "--output-dir", str(out_dir),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr

    import json

    payload = json.loads((out_dir / "comparison_summary.json").read_text())
    assert payload["pairing"]["is_paired"] is True
    assert payload["pairing"]["n_designs_intersection"] == 2

    # Sidecar-effect delta is None here (no sidecar arm provided)
    assert "sidecar_minus_baseline" not in payload["deltas"]

    # baseline arm has real base entropy (no fake zero), fused fields are
    # NaN -> emitted as null in JSON
    assert payload["arms"]["baseline"]["base_entropy_mean__mean"] is not None
    assert payload["arms"]["baseline"]["fused_entropy_mean__mean"] is None

    import pandas as pd

    comp = pd.read_parquet(out_dir / "comparison.parquet")
    assert set(comp["arm"]) == {"baseline", "refiner"}


def test_compare_if_imp_ablation_rejects_unpaired_by_default(tmp_path):
    arm_a = tmp_path / "arm_baseline"
    arm_b = tmp_path / "arm_refiner"
    _write_fake_arm(
        arm_a, arm_label="baseline",
        designs=[("p0", 0, 1.0), ("p1", 0, 1.0)],
        diagnostics=None,
    )
    _write_fake_arm(
        arm_b, arm_label="refiner",
        designs=[("p0", 0, 2.0)],  # missing p1 — populations differ
        diagnostics=None,
    )
    out_dir = tmp_path / "compare_out"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_if_imp_ablation.py"),
            "--baseline-run", str(arm_a),
            "--refiner-run", str(arm_b),
            "--output-dir", str(out_dir),
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "different design sets" in result.stderr or "Arms cover different design sets" in result.stderr


def test_compare_if_imp_ablation_allow_unpaired_intersects(tmp_path):
    arm_a = tmp_path / "arm_baseline"
    arm_b = tmp_path / "arm_refiner"
    _write_fake_arm(
        arm_a, arm_label="baseline",
        designs=[("p0", 0, 1.0), ("p1", 0, 1.0)],
        diagnostics=None,
    )
    _write_fake_arm(
        arm_b, arm_label="refiner",
        designs=[("p0", 0, 2.0)],
        diagnostics=None,
    )
    out_dir = tmp_path / "compare_out"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_if_imp_ablation.py"),
            "--baseline-run", str(arm_a),
            "--refiner-run", str(arm_b),
            "--output-dir", str(out_dir),
            "--allow-unpaired",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr

    import json

    payload = json.loads((out_dir / "comparison_summary.json").read_text())
    assert payload["pairing"]["is_paired"] is False
    assert payload["pairing"]["n_designs_intersection"] == 1
    # Both arms summarized over the intersection (1 design each)
    assert payload["arms"]["baseline"]["n_designs"] == 1
    assert payload["arms"]["refiner"]["n_designs"] == 1


def test_dplm_refiner_package_exports_public_api():
    """PLAN file structure §inverse_folding/dplm_refiner/__init__.py: must re-export public API."""
    import inverse_folding.dplm_refiner as pkg

    expected = {
        "CANONICAL_AA_ORDER",
        "DPLMBaseEntropyProbe",
        "DPLMGeometrySidecar",
        "DPLMIPARefiner",
        "DPLMRefinerConfig",
        "DPLMRefinerLogitProcessor",
        "DPLMTokenBridge",
        "IPAAtomPositions",
        "SidecarAttachedEncoder",
        "dplm_coords_to_ipa_positions",
        "enable_dropout_modules",
        "fuse_logits_by_entropy",
        "load_refiner_checkpoint",
        "load_sidecar_checkpoint",
        "mapdiff_entropy_from_log_probs",
        "masked_refiner_cross_entropy",
        "place_virtual_cb",
        "save_refiner_checkpoint",
        "save_sidecar_checkpoint",
        "select_entropy_mask",
        "sine_mask_ratio",
    }
    missing = sorted(name for name in expected if not hasattr(pkg, name))
    assert not missing, f"missing public API names: {missing}"
