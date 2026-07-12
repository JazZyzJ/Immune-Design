"""F8/F9 tests — scripts/run_rf_refine_fusion.py orchestration, artifacts, resume (fake oracles).

The driver's orchestration + artifact writers are oracle-agnostic; these tests exercise them
with fake Head/structure oracles (NO torch, NO real models). Gate F8: a fake-oracle multi-protein
run writes every required artifact and resumes without recomputation or duplicate lineage rows.
"""
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load the driver by file path (scripts/ is a namespace dir, not a package).
_spec = importlib.util.spec_from_file_location(
    "run_rf_refine_fusion", _ROOT / "scripts" / "run_rf_refine_fusion.py")
drv = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = drv
_spec.loader.exec_module(drv)

from inverse_folding.reference_flow.fusion import oracles as orc  # noqa: E402
from inverse_folding.reference_flow.fusion.config import load_fusion_config  # noqa: E402

BASE = "A" * 50
_GRID = {(20, 29, 9): 5.0, (0, 9, 9): 1.0}


def _windows(seq):
    return [SimpleNamespace(start_0b=a, end_0b=b, k=k, z=z) for (a, b, k), z in _GRID.items()]


def _risk(seq):
    return 100.0 - 10.0 * (seq[20] == "G")


class FHead:
    def __init__(self):
        self.n = 0

    def __call__(self, protein_id, sequences):
        self.n += 1
        return [SimpleNamespace(windows=_windows(s), global_risk=_risk(s)) for s in sequences]


class FStruct:
    def __init__(self):
        self.calls = 0

    def __call__(self, protein_id, sequence):
        self.calls += 1
        return SimpleNamespace(scTM=0.95, active_site_RMSD=1.0, scRMSD=1.0, pLDDT=80.0,
                               passed=True, reason="")


def _cfg(population_size=2, n_rounds=1, mode="greedy"):
    return load_fusion_config({"fusion": {
        "enabled": True, "seed": 7, "population_size": population_size, "n_rounds": n_rounds,
        "handoff": {"mode": "terminal_population"},
        "targeting": {"mode": "head_max_window", "registers_per_parent": 1, "halo_radius": 4},
        "objective": {"head_field": "global_risk", "min_head_improvement": 0.05,
                      "max_offtarget_window_increase": 10.0},
        "moves": {"explicit": {"enabled": True, "max_edit_order": 2,
                               "max_raw_candidates_per_parent": 64, "pair_seed_budget": 8},
                  "rf_reopen": {"enabled": False, "children_per_parent": 0},
                  "repair": {"enabled": False, "repair_shortlist_per_parent": 0,
                             "children_per_edit": 0, "sampler_steps": 0, "local_remask": False}},
        "structure": {"backend": "esmfold", "surrogate_mode": "none", "scTM_min": 0.85,
                      "active_site_RMSD_max": 2.0, "active_site_shell_radius": 6.0, "scRMSD_max": None,
                      "cache_dir": "/tmp/c", "max_refolds_per_parent": 8},
        "selection": {"mode": mode, "beta_start": 1.0, "beta_end": 2.0, "resample_method": "multinomial"},
        "telemetry": {"candidate_detail": True, "window_detail": False, "trajectory_detail": True},
    }})


def _designs(pid):
    return [{"design_idx": 0, "seed": 0, "sequence": BASE},
            {"design_idx": 1, "seed": 0, "sequence": BASE[:20] + "C" + BASE[21:]},
            {"design_idx": 2, "seed": 0, "sequence": BASE[:20] + "D" + BASE[21:]}]


def _oracles():
    return orc.FusionOracles(head_fn=FHead(), struct_fn=FStruct())


def _cache_factory(cfg):
    return lambda pid: orc.StructureCache(backbone_digest=pid, backend="esmfold",
                                          config_hash=cfg.config_hash())


ARTIFACTS = ["fusion_candidates.parquet", "fusion_particles.parquet", "fusion_lineage.parquet",
             "fusion_elite.parquet", "generated.parquet", "fusion_failures.parquet",
             "fusion_rounds.jsonl", "manifest.json"]


def test_run_fusion_writes_all_artifacts(tmp_path):
    cfg = _cfg()
    proteins = ["P1", "P2"]
    seed_rows = {p: _designs(p) for p in proteins}
    manifest = drv.run_fusion(proteins=proteins, seed_rows_by_protein=seed_rows, oracles=_oracles(),
                              config=cfg, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg))
    for f in ARTIFACTS:
        assert (tmp_path / f).exists(), f"missing artifact {f}"
    assert manifest["nmp_absent"] is True and manifest["n_proteins_ok"] == 2

    gen = pd.read_parquet(tmp_path / "generated.parquet")
    assert set(gen["protein_id"]) == {"P1", "P2"}
    assert list(gen.columns) == ["protein_id", "design_idx", "sequence"]
    assert (gen["design_idx"] == 0).all()  # one elite per protein at design_idx 0


def test_particle_and_lineage_join_keys_are_unique(tmp_path):
    cfg = _cfg()
    proteins = ["P1", "P2"]
    drv.run_fusion(proteins=proteins, seed_rows_by_protein={p: _designs(p) for p in proteins},
                   oracles=_oracles(), config=cfg, out_dir=tmp_path,
                   structure_cache_factory=_cache_factory(cfg))
    particles = pd.read_parquet(tmp_path / "fusion_particles.parquet")
    assert particles["particle_id"].is_unique                       # globally unique particle ids
    lineage = pd.read_parquet(tmp_path / "fusion_lineage.parquet")
    if not lineage.empty:
        assert lineage["child_particle_id"].is_unique               # no duplicate lineage edges


def test_resume_skips_completed_proteins(tmp_path):
    cfg = _cfg()
    proteins = ["P1", "P2"]
    seed_rows = {p: _designs(p) for p in proteins}
    struct = FStruct()
    oracles = orc.FusionOracles(head_fn=FHead(), struct_fn=struct)
    drv.run_fusion(proteins=proteins, seed_rows_by_protein=seed_rows, oracles=oracles, config=cfg,
                   out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg))
    calls_after_first = struct.calls
    assert calls_after_first > 0
    # second run with resume: every protein checkpoint exists -> no protein is re-charged
    drv.run_fusion(proteins=proteins, seed_rows_by_protein=seed_rows, oracles=oracles, config=cfg,
                   out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg), resume=True)
    assert struct.calls == calls_after_first                        # nothing recomputed on resume


def test_failed_protein_is_recorded_and_others_succeed(tmp_path):
    cfg = _cfg(population_size=2)
    proteins = ["P1", "PBAD"]
    seed_rows = {"P1": _designs("P1"), "PBAD": [_designs("PBAD")[0]]}  # only 1 design < N=2
    manifest = drv.run_fusion(proteins=proteins, seed_rows_by_protein=seed_rows, oracles=_oracles(),
                              config=cfg, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg))
    assert manifest["n_proteins_ok"] == 1 and manifest["n_proteins_failed"] == 1
    failures = pd.read_parquet(tmp_path / "fusion_failures.parquet")
    assert set(failures["protein_id"]) == {"PBAD"}
    assert "insufficient_feasible_initial_population" in failures.iloc[0]["message"]
    gen = pd.read_parquet(tmp_path / "generated.parquet")
    assert set(gen["protein_id"]) == {"P1"}  # the good protein still produced its elite


def test_arg_parser_has_fusion_surfaces_and_no_nmp():
    parser = drv.build_arg_parser()
    opts = {a.option_strings[0] for a in parser._actions if a.option_strings}
    assert {"--fusion-config", "--run-dir", "--out-dir", "--allele", "--print-config"} <= opts
    # the H3 repair-arm flags forwarded by submit_refine.slurm MODE=fusion must exist here (P1-2
    # SLURM<->driver contract): a rename that desyncs the launcher would fail this.
    assert {"--base-if-checkpoint", "--rf-sampler-config"} <= opts
    assert not any("nmp" in o.lower() or "netmhc" in o.lower() for o in opts)  # Head-only driver


# --------------------------------------------------------------------------- #
# review-2 fixes: constraint lookup (P1-1), resume signature (P1-4), provenance (P1-8)
# --------------------------------------------------------------------------- #
class _FakeConstraint:
    def __init__(self, anchors):  # anchors: list of (index_0b, aa)
        self.hard_anchors = [SimpleNamespace(index_0b=i, expected_aa=a) for i, a in anchors]

    @property
    def hard_anchor_indices(self):
        return tuple(a.index_0b for a in self.hard_anchors)


class _FakeManifest:
    def __init__(self, entries):  # {pid: [(index_0b, aa)]}
        self._e = {p: _FakeConstraint(a) for p, a in entries.items()}

    def has_protein(self, pid):
        return pid in self._e

    def constraint_for_protein(self, pid):
        return self._e[pid]


def test_anchor_helpers_use_constraint_for_protein_not_get():
    m = _FakeManifest({"P1": [(10, "K"), (57, "T")]})
    assert drv._anchor_indices(m, "P1") == {10, 57}
    assert drv._anchor_expected(m, "P1") == {10: "K", 57: "T"}
    assert drv._anchor_indices(m, "PX") == set()      # absent protein -> empty (no KeyError, no .get)
    assert drv._anchor_expected(m, "PX") is None
    assert drv._anchor_indices(None, "P1") == set()


def test_resume_reruns_when_config_changes(tmp_path):
    import json as _json
    proteins = ["P1"]
    seed_rows = {"P1": _designs("P1")}
    cfg_a = _cfg(n_rounds=1)
    drv.run_fusion(proteins=proteins, seed_rows_by_protein=seed_rows, oracles=_oracles(),
                   config=cfg_a, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg_a))
    cfg_b = _cfg(n_rounds=2)  # different config -> different config_hash
    manifest = drv.run_fusion(proteins=proteins, seed_rows_by_protein=seed_rows, oracles=_oracles(),
                              config=cfg_b, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg_b),
                              resume=True)
    assert manifest["config_hash"] == cfg_b.config_hash()          # aggregate reflects the NEW config
    rounds = [_json.loads(line) for line in open(tmp_path / "fusion_rounds.jsonl")]
    assert max(r["round_idx"] for r in rounds) == 2                # protein was re-run, not reused stale


class _ValidatingConstraint:
    def __init__(self, expected):  # {index_0b: aa}
        self._e = expected

    def validate_against_sequence(self, seq):
        for i, aa in self._e.items():
            if i >= len(seq) or seq[i] != aa:
                raise ValueError(f"anchor {i} expected {aa}, got {seq[i:i+1]!r}")


class _ValidatingManifest:
    def __init__(self, entries):  # {pid: {index_0b: aa}}
        self._e = {p: _ValidatingConstraint(a) for p, a in entries.items()}

    def has_protein(self, pid):
        return pid in self._e

    def constraint_for_protein(self, pid):
        return self._e[pid]


def test_driver_firewall_fails_on_manifest_reference_mismatch():
    # the driver must fail-fast (via validate_against_sequence) when a manifest anchor disagrees
    # with the reference/WT sequence, before any generation work (P1-6 driver layer).
    m = _ValidatingManifest({"P1": {2: "K"}})            # expects K at index 2
    drv._validate_manifest_against_references(m, ["P1"], {"P1": {"sequence": "AAKAA"}})  # ok, no raise
    with pytest.raises(Exception):
        drv._validate_manifest_against_references(m, ["P1"], {"P1": {"sequence": "AAAAA"}})  # A != K


def test_resume_reruns_when_input_signature_changes(tmp_path):
    # same config + seeds but a different input signature (e.g. a new Head checkpoint or manifest)
    # must NOT reuse the stale checkpoint — else changed inputs silently return old results (P1-4).
    cfg = _cfg(n_rounds=1)
    oracles = _oracles()
    drv.run_fusion(proteins=["P1"], seed_rows_by_protein={"P1": _designs("P1")}, oracles=oracles,
                   config=cfg, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg),
                   input_signature="head@v1")
    n1 = oracles.head_fn.n
    drv.run_fusion(proteins=["P1"], seed_rows_by_protein={"P1": _designs("P1")}, oracles=oracles,
                   config=cfg, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg),
                   input_signature="head@v2", resume=True)
    assert oracles.head_fn.n > n1  # re-ran on the changed input signature


def test_resume_reruns_when_seed_sequences_change(tmp_path):
    cfg = _cfg(n_rounds=1)
    oracles = _oracles()
    drv.run_fusion(proteins=["P1"], seed_rows_by_protein={"P1": _designs("P1")}, oracles=oracles,
                   config=cfg, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg))
    n1 = oracles.head_fn.n
    changed = [dict(r) for r in _designs("P1")]
    changed[0]["sequence"] = BASE[:20] + "E" + BASE[21:]  # different seed content, same config
    drv.run_fusion(proteins=["P1"], seed_rows_by_protein={"P1": changed}, oracles=oracles,
                   config=cfg, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg), resume=True)
    assert oracles.head_fn.n > n1  # re-ran on the changed seed sequences


def test_elite_parquet_has_provenance_and_manifest_has_git_sha(tmp_path):
    cfg = _cfg()
    m = drv.run_fusion(proteins=["P1"], seed_rows_by_protein={"P1": _designs("P1")}, oracles=_oracles(),
                       config=cfg, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg))
    elite = pd.read_parquet(tmp_path / "fusion_elite.parquet")
    for col in ["elite_particle_id", "source_parent_particle_id", "source_proposal_id",
                "initial_scTM", "final_scTM"]:
        assert col in elite.columns
    assert m["config_hash"] == cfg.config_hash() and "git_sha" in m and m["nmp_absent"] is True


from dataclasses import dataclass


@dataclass(frozen=True)
class _Remask:
    enabled: bool = False
    fraction_scale: float = 1.0


@dataclass(frozen=True)
class _Samp:
    seed: int = 0
    n_steps: int = 100
    remask: _Remask = _Remask()


@dataclass(frozen=True)
class _RF:
    sampler: _Samp


def _make_repair(seen, *, sampler_steps, local_remask, rf=None):
    class FakeSampler:
        def sample(self, *, sequence_length, h_values, denoiser, config, struct, controller, fixed_tokens):
            seen["seed"] = config.sampler.seed
            seen["n_steps"] = config.sampler.n_steps
            seen["remask_enabled"] = config.sampler.remask.enabled
            seen["controller"] = controller
            seen["struct"] = struct
            toks = [ord("A")] * sequence_length
            for pos, tid in fixed_tokens.items():
                toks[pos] = tid
            return SimpleNamespace(tokens=toks)

    return drv.make_sampler_repair_fn(
        aa_to_token=ord, decode_tokens=lambda ts: "".join(chr(int(t)) for t in ts),
        denoiser=object(), sampler=FakeSampler(),
        rf_config=rf or _RF(sampler=_Samp(seed=0, n_steps=100, remask=_Remask(enabled=True))),
        sequence_length=10, sampler_steps=sampler_steps, local_remask=local_remask)


def test_make_sampler_repair_fn_preserves_fixed_and_decodes():
    seen = {}
    repair = _make_repair(seen, sampler_steps=8, local_remask=False)
    out = repair("AAAAAAAAAA", {2: "W", 5: "K"}, {0, 1, 3, 4, 6, 7, 8, 9}, seed=42)
    assert out[2] == "W" and out[5] == "K"                       # fixed positions preserved
    assert len(out) == 10 and set(out) <= set("ACDEFGHIKLMNPQRSTVWY")  # complete AA20
    assert seen["seed"] == 42                                    # per-child sampler seed
    assert seen["controller"] is None and seen["struct"] is None  # controller-free backbone-only


def test_make_sampler_repair_fn_applies_fusion_config_kernel_to_sampler():
    # the fusion config's declared kernel (sampler_steps / local_remask) must OVERRIDE the base
    # rf-sampler-config, else the H3 experiment label diverges from the executed sampler (P1-1).
    seen = {}
    repair = _make_repair(seen, sampler_steps=8, local_remask=False)  # base rf has n_steps=100, remask on
    repair("AAAAAAAAAA", {2: "W"}, {0, 1, 3, 4, 5, 6, 7, 8, 9}, seed=1)
    assert seen["n_steps"] == 8               # fusion-config sampler_steps wins over base 100
    assert seen["remask_enabled"] is False    # fusion-config local_remask wins over base True


def test_failures_parquet_has_stable_schema_when_empty(tmp_path):
    cfg = _cfg()
    drv.run_fusion(proteins=["P1"], seed_rows_by_protein={"P1": _designs("P1")}, oracles=_oracles(),
                   config=cfg, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg))
    failures = pd.read_parquet(tmp_path / "fusion_failures.parquet")
    assert failures.empty
    assert list(failures.columns) == drv._FAILURE_COLS  # stable schema, not zero-column


def test_candidates_parquet_carries_aligned_windows_column(tmp_path):
    cfg = _cfg()
    drv.run_fusion(proteins=["P1"], seed_rows_by_protein={"P1": _designs("P1")}, oracles=_oracles(),
                   config=cfg, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg))
    cand = pd.read_parquet(tmp_path / "fusion_candidates.parquet")
    assert "aligned_windows" in cand.columns  # S0 window-landscape column present in schema


def test_input_signature_and_provenance_exercise_real_wiring():
    # exercise the ACTUAL _input_signature / _provenance (not an injected dict): the head window
    # range affects Head scores, so it must be in the signature, and provenance must reflect it.
    base = dict(head_checkpoint=None, head_config_dir=None, head_variant_id="LC1",
                head_allele_idx=0, window_k_min=13, window_k_max=17, allele="DRB1*07:01",
                constraint_manifest="none", base_if_checkpoint=None, rf_sampler_config=None,
                test_set_parquet=None, pdb_root="/x")
    a1 = SimpleNamespace(**base)
    a2 = SimpleNamespace(**{**base, "window_k_max": 21})
    assert drv._input_signature(a1) != drv._input_signature(a2)  # window range invalidates resume
    prov = drv._provenance(a1)
    assert prov["input_signature"] == drv._input_signature(a1)   # provenance wiring is real
    assert prov["window_k_max"] == 17 and prov["head_variant_id"] == "LC1"


def test_manifest_records_input_provenance(tmp_path):
    # the manifest must carry a reconstructable provenance block so an S0/S2 run can be audited.
    cfg = _cfg()
    m = drv.run_fusion(proteins=["P1"], seed_rows_by_protein={"P1": _designs("P1")}, oracles=_oracles(),
                       config=cfg, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg),
                       input_signature="sig-xyz",
                       manifest_extra={"provenance": {"input_signature": "sig-xyz", "head_checkpoint": "h"}})
    assert m["provenance"]["input_signature"] == "sig-xyz"


def test_rounds_jsonl_surfaces_cache_hits_and_wall_time(tmp_path):
    import json as _json
    cfg = _cfg(n_rounds=1)
    drv.run_fusion(proteins=["P1"], seed_rows_by_protein={"P1": _designs("P1")}, oracles=_oracles(),
                   config=cfg, out_dir=tmp_path, structure_cache_factory=_cache_factory(cfg))
    rounds = [_json.loads(line) for line in open(tmp_path / "fusion_rounds.jsonl")]
    assert rounds and "cache_hits" in rounds[0] and "wall_time_s" in rounds[0]
