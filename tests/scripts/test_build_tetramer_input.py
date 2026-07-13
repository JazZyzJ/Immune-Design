"""Unit tests for scripts/build_tetramer_input.py pure logic."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_tetramer_input",
    Path(__file__).resolve().parents[2] / "scripts" / "build_tetramer_input.py",
)
bti = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bti)


def test_clean_seq_strips_and_uppercases():
    assert bti.clean_seq(" ma v\nk*r ") == "MAVKR"
    assert bti.clean_seq("mavkr") == "MAVKR"


def test_maybe_strip_met_only_when_enabled_and_leading_m():
    assert bti.maybe_strip_met("MSAVK", True) == "SAVK"
    assert bti.maybe_strip_met("MSAVK", False) == "MSAVK"
    assert bti.maybe_strip_met("SAVK", True) == "SAVK"  # no leading M -> unchanged


def test_ligand_field_ccd_vs_smiles_vs_none():
    assert bti.ligand_field(None, "URC") == "CCD_URC"
    assert bti.ligand_field("something", "aza") == "CCD_AZA"        # CCD overrides SMILES
    assert bti.ligand_field(None, "CCD_AZA") == "CCD_AZA"           # idempotent prefix
    assert bti.ligand_field(bti.URC_SMILES, None) == bti.URC_SMILES  # SMILES verbatim
    assert bti.ligand_field(None, None) is None


def test_build_job_holo_has_ligand_and_count():
    job = bti.build_job("t1", "SAVK", 4, bti.URC_SMILES)
    assert job["name"] == "t1"
    pc = job["sequences"][0]["proteinChain"]
    assert pc["sequence"] == "SAVK" and pc["count"] == 4
    lig = job["sequences"][1]["ligand"]
    assert lig["ligand"] == bti.URC_SMILES and lig["count"] == 4


def test_build_job_apo_has_no_ligand_entity():
    job = bti.build_job("t1", "SAVK", 4, None)
    assert len(job["sequences"]) == 1
    assert "proteinChain" in job["sequences"][0]
