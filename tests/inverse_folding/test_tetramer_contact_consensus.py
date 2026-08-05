"""Contract tests for homolog-specific tetramer contact consensus."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inverse_folding.evaluation.tetramer_contact_consensus import (
    aggregate_contact_consensus,
    assign_explicit_mechanistic_classes,
    canonical_residue_pair_contacts,
    compute_relay_pair_evidence,
    extract_exact_homomer_mapping,
    load_position_annotations,
    read_single_fasta,
    resolve_relay_mechanism,
)


AA1_TO_3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU"}


def _atom_array(rows):
    import biotite.structure as struc

    arr = struc.AtomArray(len(rows))
    arr.coord = np.asarray([row[0] for row in rows], dtype=float)
    arr.chain_id = np.asarray([row[1] for row in rows])
    arr.res_id = np.asarray([row[2] for row in rows])
    arr.res_name = np.asarray([row[3] for row in rows])
    arr.atom_name = np.asarray([row[4] for row in rows])
    arr.element = np.asarray([row[5] for row in rows])
    arr.hetero = np.zeros(len(rows), dtype=bool)
    return arr


def _residue(chain, res_id, aa, origin):
    x, y, z = origin
    name = AA1_TO_3[aa]
    return [
        ((x, y, z), chain, res_id, name, "N", "N"),
        ((x + 1.0, y, z), chain, res_id, name, "CA", "C"),
        ((x + 2.0, y, z), chain, res_id, name, "C", "C"),
        ((x + 2.5, y, z), chain, res_id, name, "O", "O"),
        ((x + 1.0, y + 1.0, z), chain, res_id, name, "CB", "C"),
    ]


def _homotetramer(sequence="ACD", *, mutate_chain=None):
    rows = []
    for chain_i, chain in enumerate("ABCD"):
        observed = sequence if chain != mutate_chain else sequence[:-1] + "E"
        for pos, aa in enumerate(observed):
            rows += _residue(
                chain,
                10 + pos * 10,
                aa,
                (pos * 12.0, chain_i * 30.0, 0.0),
            )
    return _atom_array(rows)


def _d2_pairs(sample_id="s0"):
    values = {
        ("A", "B"): (6000.0, "interface_1", 1, "A:B|C:D"),
        ("C", "D"): (5900.0, "interface_1", 2, "A:B|C:D"),
        ("A", "D"): (5100.0, "interface_2", 1, "A:D|B:C"),
        ("B", "C"): (5000.0, "interface_2", 2, "A:D|B:C"),
        ("A", "C"): (700.0, "diagonal", 1, "A:C|B:D"),
        ("B", "D"): (650.0, "diagonal", 2, "A:C|B:D"),
    }
    return pd.DataFrame([
        {
            "sample_id": sample_id,
            "chain_1": a,
            "chain_2": b,
            "chain_pair": f"{a}:{b}",
            "bsa_total_a2": bsa,
            "bsa_rank_class": rank_class,
            "interface_class": rank_class,
            "interface_copy": copy,
            "d2_matching": matching,
            "d2_matching_mean_bsa_total_a2": np.mean([
                value[0] for value in values.values() if value[3] == matching
            ]),
        }
        for (a, b), (bsa, rank_class, copy, matching) in values.items()
    ])


def test_exact_homomer_mapping_uses_unique_canonical_substring_not_author_numbering():
    mapping = extract_exact_homomer_mapping(
        _homotetramer("ACD"),
        canonical_sequence="MACD",
        chains=list("ABCD"),
    )

    assert mapping.groupby("chain")["index_0b"].apply(list).to_dict() == {
        chain: [1, 2, 3] for chain in "ABCD"
    }
    assert mapping.groupby("chain")["res_id"].apply(list).to_dict() == {
        chain: [10, 20, 30] for chain in "ABCD"
    }
    assert (mapping["observed_aa"] == mapping["expected_aa"]).all()


def test_exact_homomer_mapping_fails_on_chain_mismatch_and_ambiguous_substring():
    with pytest.raises(ValueError, match="chain D.*exact canonical substring"):
        extract_exact_homomer_mapping(
            _homotetramer("ACD", mutate_chain="D"),
            canonical_sequence="MACD",
            chains=list("ABCD"),
        )

    ambiguous = _homotetramer("AAA")
    with pytest.raises(ValueError, match="multiple exact occurrences"):
        extract_exact_homomer_mapping(
            ambiguous,
            canonical_sequence="AAAAA",
            chains=list("ABCD"),
        )
    resolved = extract_exact_homomer_mapping(
        ambiguous,
        canonical_sequence="AAAAA",
        chains=list("ABCD"),
        canonical_start_0b=1,
    )
    assert set(resolved["index_0b"]) == {1, 2, 3}


def test_explicit_catalytic_pair_can_select_bsa_interface_2_without_guessing_interface_1():
    pairs = assign_explicit_mechanistic_classes(
        _d2_pairs(), catalytic_pair=("A", "D")
    )

    by_pair = pairs.set_index("chain_pair")
    assert by_pair.loc["A:D", "mechanistic_class"] == "catalytic"
    assert by_pair.loc["B:C", "mechanistic_class"] == "catalytic"
    assert by_pair.loc["A:B", "mechanistic_class"] == "assembly"
    assert by_pair.loc["A:C", "mechanistic_class"] == "diagonal"
    assert by_pair.loc["A:D", "bsa_rank_class"] == "interface_2"

    unresolved = assign_explicit_mechanistic_classes(_d2_pairs())
    assert set(unresolved["mechanistic_class"]) == {"unresolved"}
    assert set(unresolved["consensus_class"]) == {"interface_1", "interface_2", "diagonal"}


def test_explicit_catalytic_matching_accepts_two_symmetry_copy_pairs():
    pairs = assign_explicit_mechanistic_classes(
        _d2_pairs(), catalytic_matching=(("A", "D"), ("B", "C"))
    )

    by_pair = pairs.set_index("chain_pair")
    assert by_pair.loc["A:D", "mechanistic_class"] == "catalytic"
    assert by_pair.loc["B:C", "mechanistic_class"] == "catalytic"
    assert by_pair.loc["A:B", "mechanistic_class"] == "assembly"


def _relay_tetramer():
    """AB and CD have bidirectional donor(0)-acceptor(1) contacts; other pairs do not."""
    rows = []
    origins = {
        "A": [(0.0, 0.0, 0.0), (12.0, 0.0, 0.0)],
        "B": [(12.0, 3.0, 0.0), (0.0, 3.0, 0.0)],
        "C": [(0.0, 100.0, 0.0), (12.0, 100.0, 0.0)],
        "D": [(12.0, 103.0, 0.0), (0.0, 103.0, 0.0)],
    }
    for chain in "ABCD":
        rows += _residue(chain, 1, "A", origins[chain][0])
        rows += _residue(chain, 2, "C", origins[chain][1])
    return _atom_array(rows)


def test_relay_annotation_requires_unique_matching_and_both_symmetry_copies():
    arr = _relay_tetramer()
    mapping = extract_exact_homomer_mapping(
        arr, canonical_sequence="AC", chains=list("ABCD")
    )
    evidence = compute_relay_pair_evidence(
        arr,
        mapping,
        _d2_pairs(),
        donor_indices_0b=[0],
        acceptor_indices_0b=[1],
        cutoff=5.0,
        min_contacts_per_direction=1,
    )
    out = resolve_relay_mechanism(evidence)

    by_pair = out.set_index("chain_pair")
    assert bool(by_pair.loc["A:B", "relay_supported"])
    assert bool(by_pair.loc["C:D", "relay_supported"])
    assert not bool(by_pair.loc["A:D", "relay_supported"])
    assert by_pair.loc["A:B", "mechanistic_class"] == "catalytic"
    assert by_pair.loc["A:D", "mechanistic_class"] == "assembly"

    one_copy_missing = evidence.copy()
    one_copy_missing.loc[one_copy_missing["chain_pair"] == "C:D", "relay_supported"] = False
    unresolved = resolve_relay_mechanism(one_copy_missing)
    assert set(unresolved["mechanistic_class"]) == {"unresolved"}


def test_consensus_masks_require_contact_in_both_symmetry_copies_per_sample():
    pair_frames = []
    candidate_rows = []
    for sample_i in range(5):
        pairs = assign_explicit_mechanistic_classes(
            _d2_pairs(f"s{sample_i}"), catalytic_pair=("A", "B")
        )
        pair_frames.append(pairs)
        for pair in ("A:B", "C:D"):
            candidate_rows.append({
                "sample_id": f"s{sample_i}", "chain_pair": pair, "index_0b": 0,
            })
            if sample_i < 4:
                candidate_rows.append({
                    "sample_id": f"s{sample_i}", "chain_pair": pair, "index_0b": 1,
                })
            if sample_i == 0:
                candidate_rows.append({
                    "sample_id": f"s{sample_i}", "chain_pair": pair, "index_0b": 2,
                })
        # Position 3 is present in every sample, but only in one of the two copies.
        candidate_rows.append({
            "sample_id": f"s{sample_i}", "chain_pair": "A:B", "index_0b": 3,
        })

    consensus = aggregate_contact_consensus(
        pd.DataFrame(candidate_rows),
        pd.concat(pair_frames, ignore_index=True),
        canonical_sequence="ACDE",
    )
    catalytic = consensus[consensus["consensus_class"] == "catalytic"].set_index("index_0b")

    assert catalytic.loc[0, "n_copy_contacts"] == 10
    assert catalytic.loc[0, "n_samples_both_copies"] == 5
    assert bool(catalytic.loc[0, "mask_5ofN"])
    assert catalytic.loc[1, "n_samples_both_copies"] == 4
    assert not bool(catalytic.loc[1, "mask_5ofN"])
    assert bool(catalytic.loc[1, "mask_4ofN"])
    assert catalytic.loc[2, "n_samples_both_copies"] == 1
    assert bool(catalytic.loc[2, "mask_1ofN"])
    assert catalytic.loc[3, "n_samples_any_copy"] == 5
    assert catalytic.loc[3, "n_samples_both_copies"] == 0
    assert not bool(catalytic.loc[3, "mask_1ofN"])


def test_exact_residue_pair_contacts_mark_tetramer_only_pairs():
    arr = _relay_tetramer()
    mapping = extract_exact_homomer_mapping(
        arr, canonical_sequence="AC", chains=list("ABCD")
    )
    contacts = canonical_residue_pair_contacts(arr, mapping, cutoff=5.0)

    ab = contacts[
        (contacts["chain_1"] == "A")
        & (contacts["chain_2"] == "B")
        & (contacts["index_1_0b"] != contacts["index_2_0b"])
    ]
    assert not ab.empty
    assert ab["is_inter_chain"].all()
    assert ab["is_tetramer_only"].all()
    assert (ab["min_heavy_atom_distance_a"] < 5.0).all()


def test_position_annotations_preserve_target_aa_analogs_without_reference_identity(tmp_path):
    path = tmp_path / "analogs.csv"
    pd.DataFrame([
        {
            "protein_id": "P1",
            "target_index_0b": 1,
            "target_aa": "A",
            "reference_expected_aa": "F",
            "functional_label": "Phe160 analog",
        },
        {
            "protein_id": "P1",
            "target_index_0b": 2,
            "target_aa": "C",
            "reference_expected_aa": "V",
            "functional_label": "Val228 analog",
        },
    ]).to_csv(path, index=False)

    annotations = load_position_annotations(path, "MACD", protein_id="P1")

    assert annotations["index_0b"].tolist() == [1, 2]
    assert annotations["annotation_target_aa"].tolist() == ["A", "C"]
    assert annotations["annotation_reference_expected_aa"].tolist() == ["F", "V"]
    assert annotations["annotation_functional_label"].tolist() == [
        "Phe160 analog", "Val228 analog"
    ]


def test_position_annotations_expand_frozen_semicolon_anchor_map_per_parent(tmp_path):
    path = tmp_path / "active_site_anchor_map.csv"
    pd.DataFrame([
        {
            "protein_id": "P1",
            "hard_anchor_indices_0b": "1;2",
            "hard_anchor_labels": "site_a;site_c",
            "hard_anchor_expected_aa": "A;C",
            "monitored_shell_indices_0b": "3",
            "monitored_shell_labels": "site_d",
            "source_config": "manifest.yaml",
        },
        {
            "protein_id": "P2",
            "hard_anchor_indices_0b": "0",
            "hard_anchor_labels": "other",
            "hard_anchor_expected_aa": "M",
            "monitored_shell_indices_0b": "",
            "monitored_shell_labels": "",
            "source_config": "manifest.yaml",
        },
    ]).to_csv(path, index=False)

    annotations = load_position_annotations(path, "MACD", protein_id="P1")

    assert annotations["index_0b"].tolist() == [1, 2, 3]
    assert annotations["annotation_constraint_tier"].tolist() == [
        "hard_anchor", "hard_anchor", "monitored_shell"
    ]
    assert annotations["annotation_functional_label"].tolist() == [
        "site_a", "site_c", "site_d"
    ]


def test_multi_fasta_selects_exact_protein_id_and_rejects_missing_or_duplicate(tmp_path):
    path = tmp_path / "parents.fasta"
    path.write_text(">P1\nMACD\n>P2\nACDE\n")
    assert read_single_fasta(path, protein_id="P2") == ("P2", "ACDE")

    with pytest.raises(ValueError, match="no FASTA record"):
        read_single_fasta(path, protein_id="P3")

    path.write_text(">P1\nMACD\n>P1\nACDE\n")
    with pytest.raises(ValueError, match="duplicate FASTA records"):
        read_single_fasta(path, protein_id="P1")

