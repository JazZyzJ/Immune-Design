"""The V2 production oracle stack (PLAN §2.5, §2.7, §5.1-5.2).

**Verification boundary, stated first.**  ``build_v2_oracles`` cannot run here: it needs torch, a
DPLM checkpoint, a Head checkpoint, a refold backend and PDB inputs.  What this suite proves is the
BINDING -- that every object handed to the cycle is tied to the run's frozen config rather than to a
local default, and that a mismatch is refused before any model is touched.  Real-oracle behaviour is
a cluster check and is not claimed here.
"""

from __future__ import annotations

import types

import pytest

from scripts.rf_fusion_v2_oracles import (
    SUPPORT_POLICY_REGISTRY,
    OracleSeams,
    V2OracleError,
    resolve_support_policy,
)
from tests.inverse_folding import _v2_fixtures as F


def _config(**over):
    from inverse_folding.reference_flow.fusion_v2.config import load_v2_config
    from tests.inverse_folding.test_fusion_v2_config import _mapping

    payload = {"projection.support_policy_id": "state_derived_probe",
               "projection.support_policy_version": "v1",
               "projection.support_policy_is_diagnostic": True,
               "identity.phase": "state_transition_canary"}
    payload.update(over)
    return load_v2_config(_mapping(**payload))


# --------------------------------------------------------------------------------------------
# the support policy is RESOLVED from the config, never substituted
# --------------------------------------------------------------------------------------------


def test_the_declared_policy_is_the_one_that_gets_built():
    policy = resolve_support_policy(_config(), band_table=F.band_table(), stratum_key=F.STRATUM)
    assert policy.identity().policy_id == _config().projection.support_policy_id


def test_a_policy_id_with_no_authorized_implementation_is_refused_by_name():
    """A factory that quietly substituted a policy would produce transitions the run's own
    declaration does not describe, and the artifact would name the declared one.

    The refusal lists what IS implemented, because the operator's next question is always that.
    """
    config = _config(**{"projection.support_policy_id": "source_writeback_v1",
                        "projection.support_policy_is_diagnostic": False,
                        "identity.phase": "mechanism_cohort"})
    with pytest.raises(V2OracleError, match="source_writeback_v1"):
        resolve_support_policy(config, band_table=F.band_table(), stratum_key=F.STRATUM)


def test_the_predeclared_probe_is_deliberately_absent_from_the_registry():
    """``explicit_probe`` measures only typed nulls against a realized state: its predeclared reopen
    set names positions whose resolvedness is stochastic.  Leaving it registered would let a canary
    be configured with it and burn a prefix, K lookaheads, a Head batch and K refolds per cycle for
    nothing."""
    assert "explicit_probe" not in SUPPORT_POLICY_REGISTRY()


def test_a_version_the_run_did_not_declare_is_caught_before_the_prefix_is_paid():
    """The kernel would refuse every transition on this, but only after the ladder had bought a
    root prefix of real forward passes."""
    config = _config(**{"projection.support_policy_version": "v9"})
    with pytest.raises(V2OracleError, match="version"):
        resolve_support_policy(config, band_table=F.band_table(), stratum_key=F.STRATUM)


# --------------------------------------------------------------------------------------------
# the GPU instrument
# --------------------------------------------------------------------------------------------


def test_a_cuda_process_with_no_gpu_clock_is_refused_rather_than_credited_with_zero():
    """``check_caps`` treats GPU-seconds as a MEASURED quantity, so a fabricated ``0.0`` would let
    ``max_gpu_seconds`` read as satisfied on a number nobody took."""
    from scripts import rf_fusion_v2_oracles as mod

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: True))
    import sys

    original = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        with pytest.raises(V2OracleError, match="gpu_clock"):
            mod._default_gpu_clock()
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


def test_a_process_that_never_touched_cuda_truthfully_reports_zero():
    """The guard must not be satisfiable by refusing every run: a CPU process really does burn no
    GPU seconds, and saying so is a measurement rather than a default."""
    from scripts import rf_fusion_v2_oracles as mod

    import sys

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: False, is_initialized=lambda: False))
    original = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        assert mod._default_gpu_clock() == 0.0
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


# --------------------------------------------------------------------------------------------
# nothing here loads a model
# --------------------------------------------------------------------------------------------


def test_importing_the_oracle_stack_loads_no_torch():
    """``--print-config`` and ``--dry-run`` must not pull in a model (PLAN V2F7 acceptance), and the
    driver imports this module's package path on the execution branch only."""
    import pathlib
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import scripts.rf_fusion_v2_oracles as m; m.OracleSeams(); "
         "print('torch' in sys.modules)"],
        capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


# --------------------------------------------------------------------------------------------
# content identity: declared where frozen, OBSERVED where runtime-bound, never absent
# --------------------------------------------------------------------------------------------


class _Model:
    tokenizer_digest = "a" * 64
    fixed_token_policy = "unconstrained"

    def coordinate_mask_digest(self, protein_id):
        return "b" * 64


def _runtime_inputs(config, tmp_path):
    from scripts.rf_fusion_v2_cohort import ShardInputs

    paths = {}
    for row in config.content:
        if row.expected_sha256 is None:
            path = tmp_path / f"{row.role}.bin"
            path.write_text(row.role)
            paths[row.role] = str(path)
    return ShardInputs(**paths)


def test_a_runtime_bound_role_is_signed_by_its_OBSERVED_bytes(tmp_path):
    """PLAN §5.2 binds file CONTENTS, not paths.  A runtime-bound role carries no declared digest
    by construction, so the only honest identity for it is the one computed from what was actually
    supplied."""
    from scripts import rf_fusion_v2_oracles as mod

    config = _config()
    inputs = _runtime_inputs(config, tmp_path)
    first = mod._conditioning(config, model=_Model(), protein_id="5ZHV_B", inputs=inputs)

    role = next(r.role for r in config.content if r.expected_sha256 is None)
    (tmp_path / f"{role}.bin").write_text("different bytes entirely")
    second = mod._conditioning(config, model=_Model(), protein_id="5ZHV_B", inputs=inputs)
    assert first.digest() != second.digest(), (
        f"changing the {role!r} file did not change the run's conditioning identity")


def test_a_role_that_is_neither_frozen_nor_supplied_fails_closed(tmp_path):
    """"Missing content identity fails closed" (PLAN §5.2).  Signing a run whose inputs nobody can
    name is the one outcome that must be impossible."""
    from scripts import rf_fusion_v2_oracles as mod
    from scripts.rf_fusion_v2_cohort import ShardInputs

    with pytest.raises(V2OracleError, match="content identity"):
        mod._conditioning(_config(), model=_Model(), protein_id="5ZHV_B", inputs=ShardInputs())


def test_the_per_protein_coordinate_mask_overrides_any_file_level_digest(tmp_path):
    """The mask is a property of the PREPARED backbone, not of a file: two proteins under one
    coordinate-mask policy have different masks, and the conditioning must say so."""
    from scripts import rf_fusion_v2_oracles as mod

    config = _config()
    inputs = _runtime_inputs(config, tmp_path)

    class _Other(_Model):
        def coordinate_mask_digest(self, protein_id):
            return "c" * 64

    assert mod._conditioning(config, model=_Model(), protein_id="5ZHV_B",
                             inputs=inputs).digest() != \
        mod._conditioning(config, model=_Other(), protein_id="5ZHV_B", inputs=inputs).digest()
