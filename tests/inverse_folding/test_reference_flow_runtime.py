"""Phase C runtime helper contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from inverse_folding.reference_flow.runtime import (
    BatchedDPLMDenoiserContext,
    DPLMDenoiserContext,
    decode_residue_tokens,
    make_batched_dplm_denoiser,
    make_dplm_denoiser,
    prepare_backbone_batch,
)


class _FakeAlphabet:
    def __init__(self, tokens: dict[int, str]):
        self._tokens = tokens
        self.mask_idx = 5
        self.unk_idx = 6
        self.padding_idx = 7
        self.cls_idx = 8
        self.eos_idx = 9

    def get_tok(self, idx: int) -> str:
        return self._tokens[idx]


def test_decode_residue_tokens_rejects_non_canonical_outputs():
    task = SimpleNamespace(alphabet=_FakeAlphabet({0: "A", 1: "X"}))
    with pytest.raises(RuntimeError, match="non-canonical"):
        decode_residue_tokens(task, torch.tensor([0, 1], dtype=torch.long))


def test_make_dplm_denoiser_masks_x_token_logits():
    x_id = 4

    class _FakeDecoder:
        def __call__(self, batch, encoder_out, need_head_weights=False):
            del batch, encoder_out, need_head_weights
            logits = torch.zeros((1, 3, 10), dtype=torch.float32)
            logits[..., x_id] = 100.0
            return {"logits": logits}

    task = SimpleNamespace(
        alphabet=_FakeAlphabet(
            {
                0: "A",
                1: "C",
                2: "D",
                3: "E",
                4: "X",
                5: "<mask>",
                6: "<unk>",
                7: "<pad>",
                8: "<cls>",
                9: "<eos>",
            }
        ),
        model=SimpleNamespace(decoder=_FakeDecoder(), x_id=x_id),
    )
    context = DPLMDenoiserContext(
        task=task,
        encoder_out={},
        template_prev_tokens=torch.tensor([8, 5, 5], dtype=torch.long),
        residue_mask=torch.tensor([False, True, True]),
        tokens_template=torch.tensor([8, 0, 1], dtype=torch.long),
        sequence_length=2,
    )

    denoiser = make_dplm_denoiser(context)
    logits = denoiser(torch.tensor([0, 1], dtype=torch.long), 0.0, None)
    assert torch.isneginf(logits[:, x_id]).all()


def test_make_batched_dplm_denoiser_splits_rows_and_masks_invalid_logits():
    x_id = 4

    class _RecordingDecoder:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def __call__(self, batch, encoder_out, need_head_weights=False):
            del encoder_out, need_head_weights
            prev_tokens = batch["prev_tokens"]
            self.calls.append(prev_tokens.detach().clone())
            batch_size, padded_length = prev_tokens.shape
            logits = torch.zeros((batch_size, padded_length, 10), dtype=torch.float32)
            for b in range(batch_size):
                logits[b, :, b] = 10.0
            logits[..., x_id] = 100.0
            return {"logits": logits}

    decoder = _RecordingDecoder()
    task = SimpleNamespace(
        alphabet=_FakeAlphabet(
            {
                0: "A",
                1: "C",
                2: "D",
                3: "E",
                4: "X",
                5: "<mask>",
                6: "<unk>",
                7: "<pad>",
                8: "<cls>",
                9: "<eos>",
            }
        ),
        model=SimpleNamespace(decoder=decoder, x_id=x_id),
    )
    context = BatchedDPLMDenoiserContext(
        task=task,
        encoder_out={},
        template_prev_tokens=torch.tensor(
            [
                [8, 5, 5, 9],
                [8, 5, 9, 7],
            ],
            dtype=torch.long,
        ),
        residue_mask=torch.tensor(
            [
                [False, True, True, False],
                [False, True, False, False],
            ]
        ),
        tokens_template=torch.tensor(
            [
                [8, 0, 1, 9],
                [8, 2, 9, 7],
            ],
            dtype=torch.long,
        ),
        sequence_lengths=(2, 1),
    )

    denoiser = make_batched_dplm_denoiser(context)
    logits = denoiser(
        [
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
        ],
        0.0,
        [None, None],
    )

    assert len(decoder.calls) == 1
    assert decoder.calls[0].tolist() == [
        [8, 0, 1, 9],
        [8, 2, 9, 7],
    ]
    assert [tuple(row.shape) for row in logits] == [(2, 10), (1, 10)]
    assert torch.isneginf(logits[0][:, x_id]).all()
    assert torch.isneginf(logits[1][:, x_id]).all()


# ── prepare_backbone_batch partial-safety ─────────────────────────────────


def _build_fake_featurizer():
    """Returns a featurizer that just echoes back the items as a batch dict.
    Used to assert prepare_backbone_batch's per-entry isolation behavior."""
    def _featurizer(items):
        return {
            "tokens": torch.zeros((len(items), 4), dtype=torch.long),
            "coords": torch.zeros((len(items), 4, 4, 3)),
            "coord_mask": torch.ones((len(items), 4), dtype=torch.bool),
            "_names": [it["name"] for it in items],
        }

    return _featurizer


def test_prepare_backbone_batch_skips_bad_entries_and_returns_failures(monkeypatch):
    """A bad PDB path on one entry must NOT kill the whole batch."""
    import inverse_folding.reference_flow.runtime as runtime

    # Stub _load_byprot_imports so load_coords returns a controlled tuple.
    call_log: list[str] = []

    def fake_load_coords(path, chain=None):
        call_log.append(str(path))
        if "bad" in str(path):
            raise FileNotFoundError(f"missing: {path}")
        return torch.zeros((4, 4, 3)), "ACDE"

    monkeypatch.setattr(
        runtime,
        "_load_byprot_imports",
        lambda: {"load_coords": fake_load_coords},
    )
    monkeypatch.setattr(
        runtime,
        "resolve_structure_path",
        lambda row, root: f"/tmp/{row['protein_id']}.pdb",
    )

    task = SimpleNamespace(alphabet=SimpleNamespace(featurizer=_build_fake_featurizer()))
    entries = [
        {"protein_id": "good1", "sequence": "ACDE", "sequence_length": 4, "pdb_path": "good1.pdb"},
        {"protein_id": "bad",   "sequence": "ACDE", "sequence_length": 4, "pdb_path": "bad.pdb"},
        {"protein_id": "good2", "sequence": "ACDE", "sequence_length": 4, "pdb_path": "good2.pdb"},
    ]

    batch, seq_lengths, paths, failures, kept = prepare_backbone_batch(
        task=task, entries=entries, pdb_root="/tmp", device="cpu"
    )

    assert batch is not None
    assert seq_lengths == [4, 4]                # only 2 survivors
    assert kept == [0, 2]                        # original indices preserved
    assert len(failures) == 1
    assert failures[0]["entry"]["protein_id"] == "bad"
    assert "FileNotFoundError" in failures[0]["reason"]
    assert batch["_names"] == ["good1", "good2"]


def test_prepare_backbone_batch_strict_mode_still_raises(monkeypatch):
    """skip_invalid=False preserves the old fail-fast contract."""
    import inverse_folding.reference_flow.runtime as runtime

    def fake_load_coords(path, chain=None):
        raise FileNotFoundError(f"missing: {path}")

    monkeypatch.setattr(
        runtime,
        "_load_byprot_imports",
        lambda: {"load_coords": fake_load_coords},
    )
    monkeypatch.setattr(
        runtime,
        "resolve_structure_path",
        lambda row, root: f"/tmp/{row['protein_id']}.pdb",
    )

    task = SimpleNamespace(alphabet=SimpleNamespace(featurizer=_build_fake_featurizer()))
    entries = [{"protein_id": "any", "sequence": "ACDE", "sequence_length": 4, "pdb_path": "any.pdb"}]

    with pytest.raises(FileNotFoundError):
        prepare_backbone_batch(
            task=task,
            entries=entries,
            pdb_root="/tmp",
            device="cpu",
            skip_invalid=False,
        )


def test_prepare_backbone_batch_all_bad_returns_none_batch(monkeypatch):
    """When EVERY entry fails to prepare, batch is None and all entries
    appear in prep_failures; kept_indices is empty."""
    import inverse_folding.reference_flow.runtime as runtime

    def fake_load_coords(path, chain=None):
        raise FileNotFoundError(f"missing: {path}")

    monkeypatch.setattr(
        runtime,
        "_load_byprot_imports",
        lambda: {"load_coords": fake_load_coords},
    )
    monkeypatch.setattr(
        runtime,
        "resolve_structure_path",
        lambda row, root: f"/tmp/{row['protein_id']}.pdb",
    )

    task = SimpleNamespace(alphabet=SimpleNamespace(featurizer=_build_fake_featurizer()))
    entries = [
        {"protein_id": "a", "sequence": "AAAA", "sequence_length": 4, "pdb_path": "a.pdb"},
        {"protein_id": "b", "sequence": "BBBB", "sequence_length": 4, "pdb_path": "b.pdb"},
    ]

    batch, seq_lengths, paths, failures, kept = prepare_backbone_batch(
        task=task, entries=entries, pdb_root="/tmp", device="cpu"
    )
    assert batch is None
    assert seq_lengths == []
    assert paths == []
    assert kept == []
    assert len(failures) == 2
    assert {f["entry"]["protein_id"] for f in failures} == {"a", "b"}


# --------------------------------------------------------------------------- #
# F6: tri-state use_draft_seq_override on the encoder context builders
# --------------------------------------------------------------------------- #
import pytest as _pytest  # noqa: E402

from inverse_folding.reference_flow import runtime as _rt  # noqa: E402


def _draft_task_and_recorder(config_use_draft_seq):
    recorded = {}

    def forward_encoder(batch, use_draft_seq):
        recorded["use_draft_seq"] = use_draft_seq
        return {"feats": torch.zeros(1)}

    def inject_noise(tokens, coord_mask, noise):
        assert noise == "full_mask"
        return tokens.clone(), coord_mask.clone()

    task = SimpleNamespace(
        hparams=SimpleNamespace(generator=SimpleNamespace(use_draft_seq=config_use_draft_seq)),
        alphabet=SimpleNamespace(padding_idx=0, cls_idx=1, eos_idx=2),
        model=SimpleNamespace(forward_encoder=forward_encoder),
        inject_noise=inject_noise,
    )
    return task, recorded


def _draft_prepared():
    tokens = torch.tensor([[3, 4, 5, 6, 7]])  # residue tokens (no special 0/1/2)
    coord_mask = torch.ones(1, 5, dtype=torch.bool)
    return SimpleNamespace(batch={"tokens": tokens, "coord_mask": coord_mask})


@_pytest.mark.parametrize("cfg,override,expected", [
    (True, None, True),    # legacy: config drives (byte-equivalent to before)
    (False, None, False),  # legacy
    (True, False, False),  # Fusion forces backbone-only regardless of checkpoint config
    (False, True, True),   # override can also force it on
])
def test_use_draft_seq_override_tristate(cfg, override, expected):
    task, rec = _draft_task_and_recorder(cfg)
    _rt.build_dplm_denoiser_context(task=task, prepared=_draft_prepared(),
                                    use_draft_seq_override=override)
    assert rec["use_draft_seq"] == expected


def test_use_draft_seq_override_omitted_is_byte_equivalent_to_legacy():
    for cfg in (True, False):
        task, rec = _draft_task_and_recorder(cfg)
        _rt.build_dplm_denoiser_context(task=task, prepared=_draft_prepared())  # no kwarg
        assert rec["use_draft_seq"] == cfg


def test_batched_builder_forces_backbone_only_when_overridden():
    task, rec = _draft_task_and_recorder(config_use_draft_seq=True)
    batch = {"tokens": torch.tensor([[3, 4, 5, 6, 7]]), "coord_mask": torch.ones(1, 5, dtype=torch.bool)}
    _rt.build_batched_dplm_denoiser_context(task=task, batch=batch, use_draft_seq_override=False)
    assert rec["use_draft_seq"] is False
