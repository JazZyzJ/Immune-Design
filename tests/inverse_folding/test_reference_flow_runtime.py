"""Phase C runtime helper contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from inverse_folding.reference_flow.runtime import (
    DPLMDenoiserContext,
    decode_residue_tokens,
    make_dplm_denoiser,
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
