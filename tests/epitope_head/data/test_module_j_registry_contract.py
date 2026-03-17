"""Module J contract tests — config validation and WT classification.

TDD gates for PLAN.md Tasks J0, J2.
"""

import copy
import pytest

from epitope_head.data.netmhciipan_mutation import (
    AugmentationConfig,
    classify_wt_rank,
    is_disrupted,
    load_augmentation_config,
    validate_augmentation_config,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

VALID_RAW_CONFIG = {
    "profile": "strict",
    "tool": "netmhciipan",
    "tool_version": "4.3",
    "tool_mode": "EL",
    "allele": "HLA-DRB1*07:01",
    "wt_rank_threshold": 0.02,
    "wt_uncertain_upper": 0.05,
    "mut_rank_threshold": 0.20,
    "delta_rank_threshold": 0.18,
    "top_n_per_protein": 64,
    "inputs": {
        "source_parquet": "outputs/manifests/protein_samples_strict.parquet",
        "train_ids": "outputs/manifests/splits/strict/train_ids.txt",
        "val_ids": "outputs/manifests/splits/strict/val_ids.txt",
        "test_ids": "outputs/manifests/splits/strict/test_ids.txt",
    },
    "outputs": {
        "pilot": "outputs/augmentation/mutation_pilot_strict.json",
        "registry": "outputs/augmentation/mutation_registry_strict.parquet",
        "registry_summary": "outputs/augmentation/mutation_registry_strict_summary.json",
        "aug_train_samples": "outputs/augmentation/protein_samples_strict_aug_train.parquet",
        "aug_train_summary": "outputs/augmentation/protein_samples_strict_aug_train_summary.json",
    },
}


# ── J0: Config Validation ─────────────────────────────────────────────────

class TestJ0ConfigValidation:
    """TDD gate J0: validator rejects incomplete config, accepts complete."""

    def test_valid_config_returns_dataclass(self):
        cfg = validate_augmentation_config(VALID_RAW_CONFIG)
        assert isinstance(cfg, AugmentationConfig)
        assert cfg.tool == "netmhciipan"
        assert cfg.tool_version == "4.3"
        assert cfg.allele == "HLA-DRB1*07:01"
        assert cfg.wt_rank_threshold == 0.02
        assert cfg.mut_rank_threshold == 0.20
        assert cfg.delta_rank_threshold == 0.18
        assert cfg.top_n_per_protein == 64

    def test_missing_top_key_raises(self):
        for key in ["tool", "tool_version", "allele", "wt_rank_threshold",
                     "mut_rank_threshold", "delta_rank_threshold",
                     "top_n_per_protein", "inputs", "outputs"]:
            broken = copy.deepcopy(VALID_RAW_CONFIG)
            del broken[key]
            with pytest.raises(ValueError, match="missing required keys"):
                validate_augmentation_config(broken)

    def test_missing_input_key_raises(self):
        broken = copy.deepcopy(VALID_RAW_CONFIG)
        del broken["inputs"]["train_ids"]
        with pytest.raises(ValueError, match="inputs missing keys"):
            validate_augmentation_config(broken)

    def test_missing_output_key_raises(self):
        broken = copy.deepcopy(VALID_RAW_CONFIG)
        del broken["outputs"]["registry"]
        with pytest.raises(ValueError, match="outputs missing keys"):
            validate_augmentation_config(broken)

    def test_invalid_tool_raises(self):
        broken = copy.deepcopy(VALID_RAW_CONFIG)
        broken["tool"] = "mhcflurry"
        with pytest.raises(ValueError, match="Unknown tool"):
            validate_augmentation_config(broken)

    def test_invalid_tool_mode_raises(self):
        broken = copy.deepcopy(VALID_RAW_CONFIG)
        broken["tool_mode"] = "RANK"
        with pytest.raises(ValueError, match="Unknown tool_mode"):
            validate_augmentation_config(broken)

    def test_threshold_order_violation_raises(self):
        broken = copy.deepcopy(VALID_RAW_CONFIG)
        # wt_rank_threshold >= wt_uncertain_upper
        broken["wt_rank_threshold"] = 0.05
        broken["wt_uncertain_upper"] = 0.02
        with pytest.raises(ValueError, match="Threshold order violated"):
            validate_augmentation_config(broken)

    def test_invalid_top_n_raises(self):
        broken = copy.deepcopy(VALID_RAW_CONFIG)
        broken["top_n_per_protein"] = 0
        with pytest.raises(ValueError, match="top_n_per_protein"):
            validate_augmentation_config(broken)

    def test_load_from_yaml(self):
        cfg = load_augmentation_config()
        assert isinstance(cfg, AugmentationConfig)
        assert cfg.profile == "strict"

    def test_config_is_frozen(self):
        cfg = validate_augmentation_config(VALID_RAW_CONFIG)
        with pytest.raises(AttributeError):
            cfg.tool = "other"


# ── J2: WT Classification ─────────────────────────────────────────────────

class TestJ2WTClassification:
    """TDD gate J2: boundary-exact threshold classification."""

    @pytest.mark.parametrize("rank,expected", [
        (0.001, "keep_wt_confirmed"),
        (0.019, "keep_wt_confirmed"),
        (0.0199, "keep_wt_confirmed"),
        (0.02, "skip_uncertain"),       # boundary: 2% is uncertain
        (0.03, "skip_uncertain"),
        (0.05, "skip_uncertain"),       # boundary: 5% is still uncertain
        (0.0501, "reject_wt_disagree"),
        (0.10, "reject_wt_disagree"),
    ])
    def test_classify_wt_rank_boundaries(self, rank, expected):
        result = classify_wt_rank(rank, wt_threshold=0.02, uncertain_upper=0.05)
        assert result == expected, f"rank={rank}: expected {expected}, got {result}"


# ── Disruption Check ──────────────────────────────────────────────────────

class TestDisruptionCheck:
    """is_disrupted must require both mut_rank > threshold AND delta > threshold."""

    def test_clear_disruption(self):
        assert is_disrupted(0.35, 0.003, mut_threshold=0.20, delta_threshold=0.18)

    def test_mut_rank_below_threshold(self):
        assert not is_disrupted(0.15, 0.003, mut_threshold=0.20, delta_threshold=0.18)

    def test_delta_below_threshold(self):
        # mut_rank > 0.20, but delta = 0.20 - 0.05 = 0.15 < 0.18
        assert not is_disrupted(0.20, 0.05, mut_threshold=0.20, delta_threshold=0.18)

    def test_boundary_mut_rank(self):
        # mut_rank == 0.20 is NOT > 0.20
        assert not is_disrupted(0.20, 0.01, mut_threshold=0.20, delta_threshold=0.18)

    def test_boundary_delta(self):
        # delta == 0.18 is NOT > 0.18
        assert not is_disrupted(0.21, 0.03, mut_threshold=0.20, delta_threshold=0.18)
