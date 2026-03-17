"""NetMHCIIpan runner abstraction — standalone binary and mock backends.

Implements PLAN.md Module J (J1+):
  - Protocol-based runner interface
  - Standalone subprocess runner (cluster)
  - Mock runner (local testing)
"""

from __future__ import annotations

import csv
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PeptideScore:
    """Score for a single peptide window from NetMHCIIpan."""
    pos: int            # 0-based position in protein
    peptide: str        # peptide sequence
    core: str           # binding core (9-mer)
    el_score: float     # raw EL score
    el_rank: float      # %Rank_EL (as fraction, e.g. 0.02 = 2%)


class NetMHCIIpanRunner(ABC):
    """Abstract interface for NetMHCIIpan prediction."""

    @abstractmethod
    def score_protein(
        self,
        protein_id: str,
        protein_seq: str,
        allele: str,
        pep_length: int,
    ) -> list[PeptideScore]:
        """Score all windows of given length in a protein sequence.

        Args:
            protein_id: identifier for the protein
            protein_seq: full protein AA sequence
            allele: MHC allele (e.g. 'HLA-DRB1*07:01')
            pep_length: peptide length to scan

        Returns:
            List of PeptideScore for each valid window position.
        """
        ...


class StandaloneRunner(NetMHCIIpanRunner):
    """Calls NetMHCIIpan standalone binary via subprocess."""

    def __init__(self, binary_path: str | Path):
        self.binary_path = Path(binary_path)
        if not self.binary_path.exists():
            raise FileNotFoundError(f"NetMHCIIpan binary not found: {self.binary_path}")

    def score_protein(
        self,
        protein_id: str,
        protein_seq: str,
        allele: str,
        pep_length: int,
    ) -> list[PeptideScore]:
        # Write temp FASTA
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False
        ) as f:
            f.write(f">{protein_id}\n{protein_seq}\n")
            fasta_path = f.name

        # NetMHCIIpan uses DRB1_0701 format: HLA-DRB1*07:01 → DRB1_0701
        allele_fmt = allele.replace("HLA-", "").replace("*", "_").replace(":", "")

        cmd = [
            str(self.binary_path),
            "-f", fasta_path,
            "-a", allele_fmt,
            "-length", str(pep_length),
            "-context",  # enable context encoding
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"NetMHCIIpan failed (rc={result.returncode}): {result.stderr[:500]}"
                )
            return self._parse_output(result.stdout)
        finally:
            Path(fasta_path).unlink(missing_ok=True)

    def _parse_output(self, stdout: str) -> list[PeptideScore]:
        """Parse NetMHCIIpan 4.3 stdout into PeptideScore list.

        Verified column layout (4.3i):
          [0] Pos  [1] MHC  [2] Peptide  [3] Of  [4] Core  [5] Core_Rel
          [6] Inverted  [7] Identity  [8] Score_EL  [9] %Rank_EL
          [10] Exp_Bind  [11-12] BindLevel (optional: <= SB/WB)
        """
        scores = []
        in_data = False
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("---"):
                in_data = not in_data
                continue
            if not in_data or not line:
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                pos = int(parts[0]) - 1  # 1-based → 0-based
                peptide = parts[2]
                core = parts[4]
                el_score = float(parts[8])
                el_rank = float(parts[9]) / 100.0  # percent → fraction
                scores.append(PeptideScore(
                    pos=pos, peptide=peptide, core=core,
                    el_score=el_score, el_rank=el_rank,
                ))
            except (ValueError, IndexError):
                continue
        return scores


class MockRunner(NetMHCIIpanRunner):
    """Deterministic mock runner for testing without NetMHCIIpan binary.

    Generates biologically plausible synthetic scores:
    - Most windows get moderate-to-high ranks (not presented)
    - ~30% of windows get low ranks (< 5%), simulating real binders
    - Mutations at hydrophobic anchor positions tend to increase rank
    """

    HYDROPHOBIC = set("AILMFWVP")

    def __init__(self, seed: int = 42):
        import numpy as np
        self.rng = np.random.RandomState(seed)

    def score_protein(
        self,
        protein_id: str,
        protein_seq: str,
        allele: str,
        pep_length: int,
    ) -> list[PeptideScore]:
        import numpy as np

        n_windows = len(protein_seq) - pep_length + 1
        if n_windows <= 0:
            return []

        scores = []
        for i in range(n_windows):
            peptide = protein_seq[i:i + pep_length]
            # Deterministic seed per (protein, pos, peptide)
            h = hash((protein_id, i, peptide)) & 0xFFFFFFFF
            local_rng = np.random.RandomState(h % (2**31))

            # Base rank: bimodal — ~30% binders (low rank), ~70% non-binders
            if local_rng.random() < 0.3:
                el_rank = local_rng.exponential(0.008)  # mostly < 2%
            else:
                el_rank = 0.05 + local_rng.exponential(0.15)  # mostly > 5%
            el_rank = min(el_rank, 0.99)

            scores.append(PeptideScore(
                pos=i,
                peptide=peptide,
                core=peptide[3:12] if len(peptide) >= 12 else peptide[:9],
                el_score=max(0.0, 1.0 - el_rank * 2),
                el_rank=el_rank,
            ))
        return scores


def build_runner(
    backend: str = "standalone",
    binary_path: str | Path | None = None,
    seed: int = 42,
) -> NetMHCIIpanRunner:
    """Factory for NetMHCIIpan runners."""
    if backend == "standalone":
        if binary_path is None:
            raise ValueError("binary_path required for standalone backend")
        return StandaloneRunner(binary_path)
    elif backend == "mock":
        return MockRunner(seed=seed)
    else:
        raise ValueError(f"Unknown backend: {backend}")
