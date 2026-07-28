"""entry -> v0 lineage (PLAN_RF_REFINE_FUSION_V1 §2.11, §5.3).

v0 builds its particle ids as ``{protein}:r0:s{slot}:{sequence_md5[:12]}`` and never records which
facade row a slot came from. So the ONLY way to attribute a v0 outcome back to the root (or
trajectory) that produced it is the sequence -- and that join is unsound the moment two distinct
roots converge to the same complete sequence, which is a real and expected phenomenon.

Convergence is also a defect independent of attribution: v0 would refold the identical sequence
twice and fill two of its N slots with one basin, spending matched budget on a duplicate.

So the handoff collapses convergent facade rows to ONE representative, records which sources
collapsed into it, and the emitted generated.parquet is sequence-unique per protein -- which is
what makes the md5 join a key rather than a guess.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_alloc import FacadeRow, collapse_facade_by_sequence


def _row(design_idx, seq, source_id, root_hash=None):
    return FacadeRow(design_idx=design_idx, seed=100 + design_idx, sequence=seq,
                     sequence_md5=sequence_md5(seq), source_id=source_id,
                     root_equivalence_hash=root_hash)


def test_distinct_sequences_pass_through_untouched():
    rows = (_row(0, "ACDE", "s0", "h0"), _row(1, "FGHI", "s1", "h1"))
    collapse = collapse_facade_by_sequence(rows)
    assert collapse.rows == rows
    assert collapse.collapsed == ()
    assert collapse.n_collapsed == 0


def test_convergent_rows_collapse_to_the_best_ranked_representative():
    # two DISTINCT roots produced the same complete sequence
    rows = (_row(0, "ACDE", "s0", "h0"), _row(1, "ACDE", "s1", "h1"), _row(2, "FGHI", "s2", "h2"))
    collapse = collapse_facade_by_sequence(rows)
    assert [r.source_id for r in collapse.rows] == ["s0", "s2"]
    # design_idx is RE-DENSIFIED so v0's ordered scan sees a contiguous rank
    assert [r.design_idx for r in collapse.rows] == [0, 1]
    # the collapsed row is recorded, never silently dropped
    assert len(collapse.collapsed) == 1
    dropped = collapse.collapsed[0]
    assert dropped.source_id == "s1" and dropped.representative_source_id == "s0"
    assert dropped.root_equivalence_hash == "h1"


def test_the_representative_keeps_its_own_seed_and_lineage():
    rows = (_row(0, "ACDE", "s0", "h0"), _row(1, "ACDE", "s1", "h1"))
    kept = collapse_facade_by_sequence(rows).rows[0]
    assert kept.seed == 100 and kept.source_id == "s0" and kept.root_equivalence_hash == "h0"


def test_collapse_is_order_independent_of_input_permutation():
    rows = (_row(0, "ACDE", "s0"), _row(1, "ACDE", "s1"), _row(2, "FGHI", "s2"))
    a = collapse_facade_by_sequence(rows)
    b = collapse_facade_by_sequence(tuple(reversed(rows)))
    assert [r.source_id for r in a.rows] == [r.source_id for r in b.rows]


# --------------------------------------------------------------------------- #
# the v0 handoff must be sequence-unique, and the join table must be explicit
# --------------------------------------------------------------------------- #
def test_generated_parquet_is_sequence_unique_per_protein(tmp_path):
    import pandas as pd

    from scripts.rf_fusion_v1_artifacts import V1_TABLE_SCHEMAS, write_stable_parquet
    from scripts.run_rf_fusion_v1_entry import write_facade_generated_parquet

    schema = V1_TABLE_SCHEMAS["terminal_parent_facade"]
    rows = [
        {"protein_id": "P1", "arm_id": "preterminal", "design_idx": 0, "entry_rank": 0, "seed": 1,
         "sequence": "ACDE", "sequence_md5": sequence_md5("ACDE"), "source_id": "s0",
         "root_equivalence_hash": "h0", "continuation_id": "c0", "config_hash": "cfg"},
        {"protein_id": "P1", "arm_id": "preterminal", "design_idx": 1, "entry_rank": 1, "seed": 2,
         "sequence": "ACDE", "sequence_md5": sequence_md5("ACDE"), "source_id": "s1",
         "root_equivalence_hash": "h1", "continuation_id": "c1", "config_hash": "cfg"},
    ]
    out = tmp_path / "out"
    out.mkdir()
    write_stable_parquet(out / "terminal_parent_facade.parquet", rows,
                         columns=schema.columns, sort_by=schema.sort_by)
    # a duplicate sequence reaching v0 would fill two of its N slots with one basin AND make the
    # md5 join ambiguous; the handoff must refuse rather than emit it.
    with pytest.raises(ValueError, match="duplicate"):
        write_facade_generated_parquet(out)
    assert not (out / "generated.parquet").exists()


def test_the_handoff_writes_an_explicit_join_table(tmp_path):
    import pandas as pd

    from scripts.rf_fusion_v1_artifacts import V1_TABLE_SCHEMAS, write_stable_parquet
    from scripts.run_rf_fusion_v1_entry import write_facade_generated_parquet

    schema = V1_TABLE_SCHEMAS["terminal_parent_facade"]
    rows = [
        {"protein_id": "P1", "arm_id": "preterminal", "design_idx": i, "entry_rank": i,
         "seed": 10 + i, "sequence": seq, "sequence_md5": sequence_md5(seq),
         "source_id": f"s{i}", "root_equivalence_hash": f"h{i}", "continuation_id": f"c{i}",
         "config_hash": "cfg"}
        for i, seq in enumerate(("ACDE", "FGHI"))
    ]
    out = tmp_path / "out"
    out.mkdir()
    write_stable_parquet(out / "terminal_parent_facade.parquet", rows,
                         columns=schema.columns, sort_by=schema.sort_by)
    write_facade_generated_parquet(out)

    join = pd.read_parquet(out / "fusion_initial_admission.parquet")
    # every handed-over row carries the lineage v0 will NOT record for us
    assert set(join.columns) >= {
        "protein_id", "design_idx", "sequence_md5", "source_id", "root_equivalence_hash",
        "continuation_id", "entry_rank", "arm_id",
    }
    assert len(join) == 2
    assert set(join["root_equivalence_hash"]) == {"h0", "h1"}
    # v0's particle_id embeds sequence_md5[:12]; that prefix is recorded here, so closing the
    # join needs no string surgery on the id and no guessing.
    assert join.loc[join["source_id"] == "s0", "expected_particle_id_md5_prefix"].iloc[0] == (
        sequence_md5("ACDE")[:12]
    )
    gen = pd.read_parquet(out / "generated.parquet")
    assert set(gen["sequence"]) == {"ACDE", "FGHI"}
    assert len(set(gen["sequence"])) == len(gen)
