"""Declared seeds must act, and persisted seeds must be the ones that were drawn.

Two failures that never raise:

* **an inert knob.** `random_membership_seed` is a REQUIRED, validated, digest-bearing T0 config
  field with no reader. Changing it changes the config digest -- invalidating every checkpoint and
  re-running the whole cohort -- and produces a byte-identical random control. A robustness replicate
  of the random baseline is impossible, and the result reads as "the baseline is seed-stable".
* **an over-claimed record.** `realized_seeds` enumerates a `final:` seed for every unique root, but
  finals are drawn only for the ranked beam. The file is named for what was REALIZED; listing seeds
  that were never drawn makes it a derivation table wearing a provenance table's name.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion.v1_seeds import SeedContext
from scripts.rf_fusion_v1_preflight import resolve_entry_config
from tests._v1_fixtures import t0_config

from tests.scripts.test_rf_fusion_v1_t0 import _S, _World


def _cfg(**over):
    base = dict(prefix_attempts=4, k_est=2, k_eval=2, unique_root_capacity=3, q_t0=2,
                n_population=2, initial_refold_attempt_cap=2, rho_grid=(0.85,))
    base.update(over)
    return resolve_entry_config(t0_config(**base))


def _membership(cfg):
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
    from scripts.rf_fusion_v1_cohort import _seed_ctx
    from scripts.rf_fusion_v1_entry_core import run_t0_protein

    world = _World(cfg)
    res = run_t0_protein(
        protein_id="P", config=cfg, expected_length=4, rho_target=0.85,
        root_generator=world.gen, completer=world.complete, head_fn=world.head,
        full_trajectory_generator=world.full,
        structure_gate=lambda e: StructureOutcome(feasible=True), s_steps=_S,
        seed_ctx=_seed_ctx(cfg, "P", rho_id="rho0.850"),
    )
    return tuple(res.policy_views["random_partial"].root_hashes)


def test_the_declared_random_membership_seed_actually_moves_the_random_control():
    """A config field that is mandatory, validated and hashed but never read is worse than absent:
    it advertises a knob the operator cannot turn."""
    a = _membership(_cfg(random_membership_seed=17))
    b = _membership(_cfg(random_membership_seed=987_654))
    assert a and b
    assert a != b, (
        "changing random_membership_seed left the random_partial membership byte-identical: the "
        "declared knob is inert, so the random control cannot be re-drawn for a robustness check"
    )


def test_the_random_membership_seed_is_the_only_thing_that_moved():
    """POSITIVE CONTROL for the test above: the selected view and the shared eval table must NOT
    react to the membership seed, or the two views would stop being views over one draw."""
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
    from scripts.rf_fusion_v1_cohort import _seed_ctx
    from scripts.rf_fusion_v1_entry_core import run_t0_protein

    def run(seed):
        cfg = _cfg(random_membership_seed=seed)
        world = _World(cfg)
        return run_t0_protein(
            protein_id="P", config=cfg, expected_length=4, rho_target=0.85,
            root_generator=world.gen, completer=world.complete, head_fn=world.head,
            full_trajectory_generator=world.full,
            structure_gate=lambda e: StructureOutcome(feasible=True), s_steps=_S,
            seed_ctx=_seed_ctx(cfg, "P", rho_id="rho0.850"),
        )

    a, b = run(17), run(987_654)
    assert a.policy_views["selected_partial"].root_hashes == \
        b.policy_views["selected_partial"].root_hashes
    for root_hash, rows in a.common_eval_table.items():
        assert [sc.continuation.seed for sc in rows] == \
            [sc.continuation.seed for sc in b.common_eval_table[root_hash]]


def test_realized_seeds_lists_only_the_final_seeds_that_were_drawn():
    """`final` continuations are drawn for the RANKED beam only. Listing one per unique root claims
    draws that never happened and contradicts the run's own `root_selection` / `continuations`."""
    from scripts.rf_fusion_v1_cohort import _realized_seeds, _seed_ctx
    from tests.scripts.test_rf_fusion_v1_entry_core import _cfg as _p1_cfg, _payload, _run
    from tests.scripts.test_rf_fusion_v1_entry_core import _World as _P1World

    cfg = _p1_cfg(prefix_attempts=4, unique_root_capacity=4, initial_refold_attempt_cap=2,
                  n_population=2)
    payloads = [_payload((5 + i, 6, 7, 8), f"r{i}") for i in range(4)]
    world = _P1World(payloads, cfg, est_risk={i: 0.1 * (i + 1) for i in range(4)},
                     final_risk={i: 0.5 for i in range(4)})
    res = _run(world, cfg)

    assert len(res.unique_roots) > len(res.ranked_root_hashes), "fixture must over-supply roots"
    seeds = _realized_seeds(res, cfg, _seed_ctx(cfg, res.protein_id))
    finals = {k for k in seeds if k.startswith("final:")}
    assert len(finals) == len(res.ranked_root_hashes)
    for root_hash in res.ranked_root_hashes:
        assert any(root_hash in k for k in finals)
    unranked = set(res.unique_root_hashes if hasattr(res, "unique_root_hashes") else
                   [p.root_equivalence_hash for p in res.unique_roots]) - set(res.ranked_root_hashes)
    assert unranked
    for root_hash in unranked:
        assert not any(root_hash in k for k in finals), (
            f"realized_seeds claims a final seed for {root_hash}, which never ran a final completion"
        )


def test_no_v1_entry_config_field_is_declared_without_a_reader():
    """A declared, validated, digest-bearing knob with zero readers is worse than an absent one: it
    advertises control the operator does not have, and changing it invalidates every checkpoint to
    produce an identical run. This is the check that caught `random_membership_seed`."""
    import re
    from pathlib import Path

    from inverse_folding.reference_flow.fusion.v1_config import V1EntryConfig

    root = Path(__file__).resolve().parents[2]
    sources = [
        (root / "scripts" / name).read_text()
        for name in ("rf_fusion_v1_entry_core.py", "rf_fusion_v1_cohort.py",
                     "rf_fusion_v1_preflight.py", "run_rf_fusion_v1_entry.py",
                     "rf_fusion_v1_oracles.py", "rf_fusion_v1_artifacts.py")
    ] + [
        p.read_text()
        for p in (root / "inverse_folding" / "reference_flow" / "fusion").glob("v1_*.py")
        if p.name != "v1_config.py"
    ]
    blob = "\n".join(sources)

    # PINNED fields are exempt: `v1_config` validates them to exactly one legal value, so they
    # declare an invariant rather than offer a knob and cannot silently change a run.
    pinned = {"completions_per_ranked_root"}

    unread = []
    for field in V1EntryConfig.__dataclass_fields__:
        if field in pinned:
            continue
        # a reader is any `config.<field>` / `cfg.<field>` / `getattr(config, "<field>")` use
        if not re.search(rf"(?:\.{re.escape(field)}\b|\"{re.escape(field)}\")", blob):
            unread.append(field)
    assert not unread, f"config fields declared but never read by any V1 module: {unread}"
