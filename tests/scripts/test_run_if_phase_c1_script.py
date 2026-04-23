"""Phase C1 script helper tests."""

from scripts.run_if_phase_c1 import derive_h_shuffle_seed


def test_derive_h_shuffle_seed_varies_by_protein_and_design():
    base_seed = 123
    assert derive_h_shuffle_seed(base_seed, "p1", 0) == derive_h_shuffle_seed(base_seed, "p1", 0)
    assert derive_h_shuffle_seed(base_seed, "p1", 0) != derive_h_shuffle_seed(base_seed, "p1", 1)
    assert derive_h_shuffle_seed(base_seed, "p1", 0) != derive_h_shuffle_seed(base_seed, "p2", 0)
