"""Wave-4 dual-head: exact_margin_loss behavior."""
import torch

from epitope_head.training.losses import exact_margin_loss


def test_ranks_and_decreases_as_pos_rises():
    neg = torch.tensor([1.0, 0.9, 0.8])
    lo = exact_margin_loss(torch.tensor([0.0, 0.1]), neg, margin_m=0.3)
    hi = exact_margin_loss(torch.tensor([2.0, 2.1]), neg, margin_m=0.3)
    assert lo.item() > 0.0
    assert hi.item() < lo.item()  # raising positives above negatives lowers the loss


def test_zero_when_separated():
    pos = torch.tensor([3.0, 3.1])
    neg = torch.tensor([0.5, 0.4])  # pos - neg >> margin everywhere
    assert exact_margin_loss(pos, neg, margin_m=0.3).item() == 0.0


def test_empty_returns_zero():
    neg = torch.tensor([0.5])
    assert exact_margin_loss(torch.tensor([]), neg).item() == 0.0
    assert exact_margin_loss(torch.tensor([1.0]), torch.tensor([])).item() == 0.0


def test_differentiable():
    pos = torch.tensor([0.0, 0.1], requires_grad=True)
    neg = torch.tensor([1.0, 0.9])
    exact_margin_loss(pos, neg, margin_m=0.3).backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()
