"""Wave-3 T2: window IoU geometry + IoU-ranking span loss."""
import pytest

torch = pytest.importorskip("torch")

from epitope_head.training.span_geom import window_iou_matrix, max_iou_per_window
from epitope_head.training.losses import window_iou_rank_loss, compute_loss


def _t(rows):
    return torch.tensor(rows, dtype=torch.long)


# ── geometry (half-open, matches benchmark _span_iou) ──────────────────────────
def test_iou_matrix_values():
    windows = _t([[10, 20], [10, 18], [0, 5], [12, 22]])
    positives = _t([[10, 20]])
    iou = window_iou_matrix(windows, positives)  # [4,1]
    assert iou.shape == (4, 1)
    assert iou[0, 0].item() == pytest.approx(1.0)          # exact
    assert iou[1, 0].item() == pytest.approx(8 / 10)        # inter8 / union10
    assert iou[2, 0].item() == pytest.approx(0.0)           # disjoint
    # (12,22) vs (10,20): inter [12,20)=8, union=12 -> 8/12
    assert iou[3, 0].item() == pytest.approx(8 / 12)


def test_max_iou_over_positives():
    windows = _t([[10, 20], [40, 50]])
    positives = _t([[10, 20], [38, 52]])
    m = max_iou_per_window(windows, positives)
    assert m[0].item() == pytest.approx(1.0)                # matches pos0 exactly
    # (40,50) vs (38,52): inter [40,50)=10, union=14 -> 10/14
    assert m[1].item() == pytest.approx(10 / 14)


def test_empty_shapes():
    assert window_iou_matrix(_t([]).reshape(0, 2), _t([[1, 2]])).shape == (0, 1)
    assert max_iou_per_window(_t([[1, 2]]), _t([]).reshape(0, 2)).tolist() == [0.0]


# ── IoU-ranking loss ───────────────────────────────────────────────────────────
def test_equal_ious_no_pairs():
    logits = torch.tensor([0.3, -0.1, 2.0])
    ious = torch.tensor([0.5, 0.5, 0.5])
    loss, meta = window_iou_rank_loss(logits, ious, min_iou_gap=0.1)
    assert meta["n_pairs"] == 0
    assert loss.item() == pytest.approx(0.0)


def test_correct_order_beats_reversed():
    ious = torch.tensor([1.0, 0.5, 0.0])
    good = torch.tensor([3.0, 1.0, -1.0])   # high IoU -> high logit
    bad = torch.tensor([-1.0, 1.0, 3.0])    # reversed
    lg, _ = window_iou_rank_loss(good, ious, margin_m=0.5)
    lb, _ = window_iou_rank_loss(bad, ious, margin_m=0.5)
    assert lb.item() > lg.item()
    # well-separated correct order incurs zero hinge loss
    assert lg.item() == pytest.approx(0.0, abs=1e-6)


def test_min_iou_gap_excludes_near_ties():
    logits = torch.tensor([0.0, 5.0])      # logit order contradicts iou order
    ious = torch.tensor([0.55, 0.50])      # gap 0.05
    loss_strict, meta_strict = window_iou_rank_loss(logits, ious, min_iou_gap=0.1)
    loss_loose, meta_loose = window_iou_rank_loss(logits, ious, min_iou_gap=0.01)
    assert meta_strict["n_pairs"] == 0 and loss_strict.item() == pytest.approx(0.0)
    assert meta_loose["n_pairs"] == 1 and loss_loose.item() > 0.0


def test_differentiable():
    logits = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    ious = torch.tensor([0.0, 0.5, 1.0])
    loss, _ = window_iou_rank_loss(logits, ious, margin_m=1.0)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# ── compute_loss integration / backward-compat ─────────────────────────────────
def test_compute_loss_backward_compat():
    pos = torch.tensor([2.0, 1.5])
    neg = torch.tensor([0.0, -0.5, 0.2])
    base = compute_loss(pos, neg, objective_mode="mixed_margin", lambda_margin=0.1)
    assert "loss_iou_rank" in base
    assert base["loss_iou_rank"].item() == pytest.approx(0.0)
    # adding window args but lambda_iou_rank=0 must NOT change loss_total
    wl = torch.tensor([2.0, 1.5, 0.0, -0.5, 0.2])
    wi = torch.tensor([1.0, 1.0, 0.0, 0.3, 0.1])
    same = compute_loss(pos, neg, objective_mode="mixed_margin", lambda_margin=0.1,
                        window_logits=wl, window_ious=wi, lambda_iou_rank=0.0)
    assert same["loss_total"].item() == pytest.approx(base["loss_total"].item())


def test_compute_loss_adds_iou_rank_term():
    pos = torch.tensor([2.0, 1.5])
    neg = torch.tensor([0.0, -0.5, 0.2])
    # window set where a low-IoU window outscores a high-IoU one -> positive rank loss
    wl = torch.tensor([0.0, 3.0])
    wi = torch.tensor([0.9, 0.1])
    base = compute_loss(pos, neg, objective_mode="mixed_margin", lambda_margin=0.1)
    out = compute_loss(pos, neg, objective_mode="mixed_margin", lambda_margin=0.1,
                       window_logits=wl, window_ious=wi, lambda_iou_rank=0.5,
                       iou_rank_margin=0.5, iou_rank_min_gap=0.1)
    assert out["loss_iou_rank"].item() > 0.0
    assert out["loss_total"].item() == pytest.approx(
        base["loss_total"].item() + 0.5 * out["loss_iou_rank"].item(), rel=1e-5)
