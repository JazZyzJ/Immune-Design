"""F8 unit tests — fusion/structure_metrics.py active-site shell RMSD geometry (§1.4)."""
import numpy as np
import pytest

from inverse_folding.reference_flow.fusion import structure_metrics as sm


def test_shell_indices_includes_anchor_and_within_radius_neighbors():
    # 5 residues on a line 0,1,2,3,4 Å apart; anchor at index 2, radius 1.5 -> {1,2,3}
    ca = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)
    shell = sm.shell_indices_from_ca(ca, {2}, radius=1.5)
    assert shell == frozenset({1, 2, 3})


def test_shell_indices_union_over_multiple_anchors():
    ca = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0], [11, 0, 0]], dtype=float)
    shell = sm.shell_indices_from_ca(ca, {0, 4}, radius=1.0)
    assert shell == frozenset({0, 1, 3, 4})


def test_shell_indices_empty_when_no_anchors():
    ca = np.zeros((5, 3))
    assert sm.shell_indices_from_ca(ca, set(), radius=6.0) == frozenset()


def test_shell_indices_rejects_out_of_range_anchor():
    ca = np.zeros((3, 3))
    with pytest.raises(ValueError):
        sm.shell_indices_from_ca(ca, {5}, radius=6.0)


def test_shell_indices_rejects_negative_anchor():
    # a negative index would silently wrap to a terminal residue under numpy fancy-indexing;
    # the firewall must fail closed instead (§1.4 anchor contract).
    ca = np.zeros((3, 3))
    with pytest.raises(ValueError):
        sm.shell_indices_from_ca(ca, {-1}, radius=6.0)


def test_shell_rmsd_is_rms_over_subset():
    rows = [{"residue_idx": 0, "sc_ca_distance": 0.0},
            {"residue_idx": 1, "sc_ca_distance": 3.0},
            {"residue_idx": 2, "sc_ca_distance": 4.0},
            {"residue_idx": 3, "sc_ca_distance": 100.0}]  # outside shell -> ignored
    val = sm.shell_rmsd_from_rows(rows, {1, 2})
    assert val == pytest.approx(np.sqrt((9 + 16) / 2))  # sqrt(mean(3^2, 4^2))


def test_shell_rmsd_none_when_shell_empty():
    rows = [{"residue_idx": 0, "sc_ca_distance": 1.0}]
    assert sm.shell_rmsd_from_rows(rows, set()) is None
