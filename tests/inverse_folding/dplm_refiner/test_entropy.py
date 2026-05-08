import torch
from torch import nn

from inverse_folding.dplm_refiner.entropy import (
    enable_dropout_modules,
    mapdiff_entropy_from_log_probs,
    select_entropy_mask,
    sine_mask_ratio,
)


def test_mapdiff_entropy_matches_source_mean_definition():
    log_probs = torch.log_softmax(torch.tensor([[2.0, 0.0], [0.0, 0.0]]), dim=-1)
    probs = log_probs.exp()
    expected = -(probs * log_probs).mean(dim=-1)

    actual = mapdiff_entropy_from_log_probs(log_probs)

    assert torch.allclose(actual, expected)


def test_sine_mask_ratio_uses_mapdiff_formula():
    beta_t_bar = torch.tensor([[0.0], [1.0]])

    ratios = sine_mask_ratio(beta_t_bar, center=0.4, max_deviation=0.2)

    assert torch.allclose(ratios, torch.tensor([0.4, 0.6]), atol=1e-6)


def test_select_entropy_mask_respects_valid_mask_per_batch():
    entropy = torch.tensor([[0.1, 0.4, 0.3, 0.2], [0.9, 0.8, 0.1, 0.0]])
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    ratios = torch.tensor([0.5, 0.5])

    mask = select_entropy_mask(entropy, valid, ratios)

    assert mask.tolist() == [[False, True, True, False], [True, False, False, False]]


def test_enable_dropout_modules_only_switches_dropout_to_train():
    model = nn.Sequential(nn.Linear(3, 3), nn.Dropout(0.5))
    model.eval()

    changed = enable_dropout_modules(model)

    assert changed == 1
    assert model[0].training is False
    assert model[1].training is True
