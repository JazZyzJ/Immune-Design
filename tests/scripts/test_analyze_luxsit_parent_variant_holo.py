from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_luxsit_parent_variant_holo import (
    Atom,
    RigidFrame,
    Structure,
    fit_frame,
    genotype_fields,
    residue_sidechain_errors,
)


def _atom(name: str, coord: tuple[float, float, float]) -> Atom:
    return Atom("ATOM", name[0], name, "ASP", "A", 18, np.asarray(coord, dtype=float))


def test_genotype_fields_distinguishes_clean_partial_from_alternative() -> None:
    parent = list("A" * 117)
    parent[59], parent[95], parent[109] = "R", "A", "M"
    assert genotype_fields("".join(parent))["genotype_class"] == "exact_parent"

    parent[59], parent[95], parent[109] = "S", "L", "V"
    assert genotype_fields("".join(parent))["genotype_class"] == "exact_i"

    parent[59], parent[95], parent[109] = "S", "A", "M"
    assert genotype_fields("".join(parent))["genotype_class"] == "clean_partial_1"

    parent[59], parent[95], parent[109] = "Q", "L", "M"
    fields = genotype_fields("".join(parent))
    assert fields["genotype_class"] == "alternative"
    assert fields["n_i_recovered"] == 1
    assert fields["n_alternative"] == 1


def test_fit_frame_recovers_rigid_transform() -> None:
    mobile = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = mobile @ rotation + np.asarray([3.0, -2.0, 1.0])
    frame = fit_frame(mobile, target)
    np.testing.assert_allclose(frame.apply(mobile), target, atol=1e-8)


def test_asp_terminal_atom_swap_is_symmetry_corrected() -> None:
    target_atoms = {
        "CB": _atom("CB", (0.0, 0.0, 0.0)),
        "CG": _atom("CG", (1.0, 0.0, 0.0)),
        "OD1": _atom("OD1", (2.0, 1.0, 0.0)),
        "OD2": _atom("OD2", (2.0, -1.0, 0.0)),
    }
    mobile_atoms = dict(target_atoms)
    mobile_atoms["OD1"] = _atom("OD1", (2.0, -1.0, 0.0))
    mobile_atoms["OD2"] = _atom("OD2", (2.0, 1.0, 0.0))
    mobile = Structure({18: mobile_atoms}, {18: "ASP"}, {})
    target = Structure({18: target_atoms}, {18: "ASP"}, {})
    identity = RigidFrame(np.eye(3), np.zeros(3), np.zeros(3))
    _errors, rmsd = residue_sidechain_errors(mobile, target, 18, identity)
    assert rmsd == pytest.approx(0.0)
