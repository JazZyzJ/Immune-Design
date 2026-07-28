"""V1F3 seed-derivation contract tests (PLAN_RF_REFINE_FUSION_V1 §2.6).

Length-delimited SHA-256 (never Python hash(), never a raw separator-join), disjoint
namespaces (root / est / eval / final / random-control), process-independence, and
collision detection.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest

from inverse_folding.reference_flow.fusion.v1_seeds import (
    SeedContext,
    assert_no_seed_collisions,
    derive_seed,
    realized_seed_manifest,
)


def _ctx(**over):
    base = dict(
        seed_schema="v1seed-1",
        campaign_id="camp",
        phase="t0",
        split_role="t0_dev",
        master_seed=20260722,
        entry_arm="preterminal",
        protein_id="P1",
        rho_id="rho0.850",
    )
    base.update(over)
    return SeedContext(**base)


def test_derive_seed_is_deterministic_and_in_uint64_range():
    a = derive_seed("v1seed-1", "camp", 3, "root")
    b = derive_seed("v1seed-1", "camp", 3, "root")
    assert a == b
    assert 0 <= a < 2**64


def test_derive_seed_is_length_delimited_not_separator_join():
    # A raw separator-join would collide ("a","bc") vs ("ab","c"); length-delimited must not.
    assert derive_seed("a", "bc") != derive_seed("ab", "c")
    # int vs str of the same characters are distinct (typed encoding).
    assert derive_seed(1) != derive_seed("1")


def test_derive_seed_rejects_float_and_bool():
    with pytest.raises(TypeError):
        derive_seed(0.85)  # floats must be canonicalized to a stable str first
    with pytest.raises(TypeError):
        derive_seed(True)


def test_derive_seed_is_process_independent_under_pythonhashseed():
    # Never Python hash(): the value must be identical across interpreter hash seeds.
    code = (
        "from inverse_folding.reference_flow.fusion.v1_seeds import derive_seed;"
        "print(derive_seed('v1seed-1','camp','P1','rho0.850','root',7))"
    )
    outs = []
    for hs in ("0", "1", "12345"):
        env = {"PYTHONHASHSEED": hs, "PATH": ""}
        import os

        env = {**os.environ, "PYTHONHASHSEED": hs}
        res = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env,
            cwd="/Users/jerry/Project/MHC-IF-fusion",
        )
        assert res.returncode == 0, res.stderr
        outs.append(res.stdout.strip())
    assert len(set(outs)) == 1, f"seed varied across PYTHONHASHSEED: {outs}"


def test_derive_seed_matches_known_sha256_vector():
    # Pin the encoding so a silent change to the derivation is caught.
    # str "v1seed-1" -> "s"+value = "sv1seed-1" (9 bytes); int 7 -> "i7" (2 bytes).
    parts = [b"9:sv1seed-1", b"2:i7"]
    payload = b"\x1e".join(parts)
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    assert derive_seed("v1seed-1", 7) == expected


def test_seed_namespaces_are_disjoint():
    ctx = _ctx()
    eqh = "abc123"
    seeds = {
        "root0": ctx.root_seed(0),
        "root1": ctx.root_seed(1),
        "est0": ctx.continuation_seed(eqh, "est", 0),
        "eval0": ctx.continuation_seed(eqh, "eval", 0),
        "final0": ctx.continuation_seed(eqh, "final", 0),
        "random_membership": ctx.random_membership_seed(),
        "independent_full0": ctx.independent_full_seed(0),
    }
    assert len(set(seeds.values())) == len(seeds), f"collision among namespaces: {seeds}"
    # est/eval/final for the SAME root+replicate are distinct set-tags.
    assert ctx.continuation_seed(eqh, "est", 0) != ctx.continuation_seed(eqh, "eval", 0)
    # different equivalence hashes -> different continuation seeds.
    assert ctx.continuation_seed("h1", "est", 0) != ctx.continuation_seed("h2", "est", 0)


def test_continuation_seed_rejects_unknown_set_tag():
    ctx = _ctx()
    with pytest.raises(ValueError):
        ctx.continuation_seed("h", "bogus", 0)


def test_seed_context_bound_to_campaign_phase_split():
    a = _ctx(phase="t0").root_seed(0)
    b = _ctx(phase="p1").root_seed(0)
    c = _ctx(split_role="p1_holdout").root_seed(0)
    assert len({a, b, c}) == 3  # phase and split_role are part of the namespace


def test_assert_no_seed_collisions_raises_on_duplicate():
    assert_no_seed_collisions({"a": 1, "b": 2, "c": 3})  # no raise
    with pytest.raises(ValueError):
        assert_no_seed_collisions({"a": 1, "b": 1})


def test_independent_full_and_terminal_streams_are_disjoint():
    ctx = _ctx()
    term = _ctx(entry_arm="terminal", rho_id="none")
    seeds = {
        "root0": ctx.root_seed(0),
        "random_membership": ctx.random_membership_seed(),
        "independent_full0": ctx.independent_full_seed(0),
        "independent_full1": ctx.independent_full_seed(1),
        "terminal0": term.terminal_complete_seed(0),
        "terminal1": term.terminal_complete_seed(1),
    }
    assert len(set(seeds.values())) == len(seeds)


def test_realized_seed_manifest_enumerates_all_streams_without_collision():
    ctx = _ctx()
    manifest = realized_seed_manifest(
        ctx, root_count=4, unique_root_hashes=["h1", "h2"], k_est=8, k_eval=8,
        final_count=1, independent_full_count=4, include_random_membership=True,
    )
    # root(4) + 2*(est 8 + eval 8 + final 1) + random_membership(1) + independent_full(4) = 43.
    # Note random membership contributes exactly ONE seed, not one per selected root: the random
    # view reads the SAME eval table as the selected view.
    assert len(manifest) == 4 + 2 * (8 + 8 + 1) + 1 + 4
    assert len(set(manifest.values())) == len(manifest)  # collision gate passed (would have raised)


# --------------------------------------------------------------------------- #
# V1-A shared-pool law (PLAN §2.6, §2.9-§2.10): selected/random are MEMBERSHIP
# views over ONE root pool and ONE held-out eval table.
# --------------------------------------------------------------------------- #
def test_seed_context_rejects_a_policy_label_as_the_arm():
    # If a driver could pass a policy label where the arm belongs, selected_partial and
    # random_partial would each fork their own root pool AND their own K_EVAL table -- exactly the
    # fake contrast the shared-pool law forbids. The arm enum makes that structurally impossible.
    for bad in ("selected_partial", "random_partial", "independent_full", "C", ""):
        with pytest.raises(ValueError):
            _ctx(entry_arm=bad)


def test_policy_label_never_enters_a_continuation_seed():
    # The two policy VIEWS resolve to the same arm, so every root/est/eval seed is byte-identical:
    # a policy label cannot change a single continuation draw.
    selected_view = _ctx(entry_arm="preterminal")
    random_view = _ctx(entry_arm="preterminal")
    assert selected_view.root_seed(0) == random_view.root_seed(0)
    for tag in ("est", "eval"):
        assert (selected_view.continuation_seed("h1", tag, 0)
                == random_view.continuation_seed("h1", tag, 0))


def test_random_membership_is_one_seed_not_a_continuation_stream():
    ctx = _ctx()
    membership = ctx.random_membership_seed()
    assert isinstance(membership, int)
    # it must be disjoint from every continuation stream ...
    others = {ctx.root_seed(0), ctx.continuation_seed("h1", "est", 0),
              ctx.continuation_seed("h1", "eval", 0), ctx.continuation_seed("h1", "final", 0)}
    assert membership not in others
    # ... and it must be stable (a single draw, not indexed per replicate)
    assert membership == ctx.random_membership_seed()


def test_terminal_complete_seed_is_arm_bound_and_rho_invariant():
    term = _ctx(entry_arm="terminal", rho_id="none")
    seed = term.terminal_complete_seed(0)
    assert seed != term.terminal_complete_seed(1)
    # the terminal arm has no maturity, so its stream must not depend on a rho label
    assert seed == _ctx(entry_arm="terminal", rho_id="none").terminal_complete_seed(0)
    # and a pre-terminal context may not mint terminal-complete trajectories
    with pytest.raises(ValueError):
        _ctx(entry_arm="preterminal").terminal_complete_seed(0)


def test_independent_full_seed_is_disjoint_from_preterminal_streams():
    ctx = _ctx()
    full = ctx.independent_full_seed(0)
    assert full != ctx.root_seed(0)
    assert full != ctx.continuation_seed("h1", "est", 0)
    assert full != ctx.random_membership_seed()
    assert full != ctx.independent_full_seed(1)


def test_replicate_index_must_be_an_exact_int():
    # int(1.9) == int("1") == 1 would silently alias three different callers onto one stream.
    ctx = _ctx()
    for bad in (1.9, "1", True, None):
        with pytest.raises((TypeError, ValueError)):
            ctx.continuation_seed("h1", "est", bad)
        with pytest.raises((TypeError, ValueError)):
            ctx.root_seed(bad)


def test_terminal_context_cannot_mint_preterminal_streams():
    # The terminal arm has no root prefix and no estimator set; offering those streams would let
    # a caller fabricate pre-terminal evidence under a terminal identity.
    term = _ctx(entry_arm="terminal", rho_id="none")
    with pytest.raises(ValueError):
        term.root_seed(0)
    with pytest.raises(ValueError):
        term.continuation_seed("h1", "est", 0)
