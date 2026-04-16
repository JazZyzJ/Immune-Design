from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "analysis" / "tier1_structural_analysis.py"
SPEC = importlib.util.spec_from_file_location("tier1_structural_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_compute_per_protein_stats_accepts_pd_na_numeric_columns():
    df = pd.DataFrame(
        {
            "protein_id": ["toy"] * 4,
            "is_epitope": [1, 1, 0, 0],
            "rsa": [0.2, np.nan, 0.5, 0.6],
            "bfactor_ca_z": [0.0, 1.0, -0.2, np.nan],
            "contact_number": [pd.NA, 8, 10, 11],
            "ss3": ["C", "H", "C", "E"],
        }
    )

    stats = MODULE.compute_per_protein_stats(df)

    assert "contact_number" in stats
    assert stats["contact_number"]["n_pos"] == 1
    assert stats["contact_number"]["n_neg"] == 2
