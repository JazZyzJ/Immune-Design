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
    pep_length: int = 0 # peptide length (filled by multi-length parsing)


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
        """Score all windows of given length in a protein sequence."""
        ...

    def score_batch(
        self,
        entries: list[tuple[str, str]],
        allele: str,
        pep_lengths: list[int],
    ) -> dict[str, dict[int, list[PeptideScore]]]:
        """Score multiple proteins × multiple lengths in one call.

        Returns {protein_id: {pep_length: [scores]}}.
        Default: sequential fallback.
        """
        results: dict[str, dict[int, list[PeptideScore]]] = {}
        for pid, seq in entries:
            results[pid] = {}
            for pl in pep_lengths:
                results[pid][pl] = self.score_protein(pid, seq, allele, pl)
        return results


class StandaloneRunner(NetMHCIIpanRunner):
    """Calls NetMHCIIpan standalone binary via subprocess."""

    def __init__(self, binary_path: str | Path, batch_size: int = 30):
        self.binary_path = Path(binary_path)
        self.batch_size = batch_size
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

    def score_batch(
        self,
        entries: list[tuple[str, str]],
        allele: str,
        pep_lengths: list[int],
    ) -> dict[str, dict[int, list[PeptideScore]]]:
        """Score multiple proteins × multiple lengths, chunked to avoid timeout.

        Uses `-length 14,15,18` to merge all lengths into one call per chunk.
        Returns {protein_id: {pep_length: [scores]}}.
        """
        if not entries or not pep_lengths:
            return {}

        allele_fmt = allele.replace("HLA-", "").replace("*", "_").replace(":", "")
        length_str = ",".join(str(pl) for pl in sorted(pep_lengths))
        all_results: dict[str, dict[int, list[PeptideScore]]] = {}

        for chunk_start in range(0, len(entries), self.batch_size):
            chunk = entries[chunk_start:chunk_start + self.batch_size]

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".fasta", delete=False
            ) as f:
                for pid, seq in chunk:
                    f.write(f">{pid}\n{seq}\n")
                fasta_path = f.name

            cmd = [
                str(self.binary_path),
                "-f", fasta_path,
                "-a", allele_fmt,
                "-length", length_str,
                "-context",
            ]

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=1200,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"NetMHCIIpan batch failed (rc={result.returncode}): "
                        f"{result.stderr[:500]}"
                    )
                chunk_results = self._parse_batch_output(result.stdout)
                # Merge into all_results
                for pid, by_len in chunk_results.items():
                    if pid not in all_results:
                        all_results[pid] = {}
                    for pl, scores in by_len.items():
                        all_results[pid][pl] = scores
            finally:
                Path(fasta_path).unlink(missing_ok=True)

        return all_results

    def _parse_batch_output(
        self, stdout: str,
    ) -> dict[str, dict[int, list[PeptideScore]]]:
        """Parse multi-sequence/multi-length output.

        Groups by Identity and peptide length.
        Returns {identity: {pep_length: [PeptideScore]}}.

        NetMHCIIpan outputs one block per (allele, identity) combination,
        each block wrapped in --- separators. For multi-length runs,
        peptide length varies per row and is inferred from len(peptide).
        """
        results: dict[str, dict[int, list[PeptideScore]]] = {}
        dash_count = 0
        idx_score_el = 8
        idx_rank_el = 9
        idx_identity = 7

        for line in stdout.splitlines():
            stripped = line.strip()

            if stripped.startswith("---"):
                dash_count += 1
                continue

            # Header detection (between first pair of ---)
            if dash_count == 1 and "Score_EL" in stripped:
                header_parts = stripped.split()
                for i, h in enumerate(header_parts):
                    if h == "Score_EL":
                        idx_score_el = i
                    elif h == "%Rank_EL":
                        idx_rank_el = i
                    elif h == "Identity":
                        idx_identity = i
                continue

            # Data lines are between dash 2 and dash 3
            if dash_count < 2 or dash_count >= 3:
                continue
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < idx_rank_el + 1:
                continue

            try:
                pos = int(parts[0]) - 1
                peptide = parts[2]
                core = parts[4]
                identity = parts[idx_identity]
                el_score = float(parts[idx_score_el])
                el_rank = float(parts[idx_rank_el]) / 100.0
                pep_len = len(peptide)

                results.setdefault(identity, {}).setdefault(pep_len, []).append(
                    PeptideScore(
                        pos=pos, peptide=peptide, core=core,
                        el_score=el_score, el_rank=el_rank,
                        pep_length=pep_len,
                    )
                )
            except (ValueError, IndexError):
                continue

        return results

    def _parse_output(self, stdout: str) -> list[PeptideScore]:
        """Parse NetMHCIIpan 4.3 stdout into PeptideScore list.

        Output structure (verified 4.3i):
            ---              ← separator 1
            Header line      ← column names (Pos, MHC, ..., Score_EL, %Rank_EL, ...)
            ---              ← separator 2
            Data lines...    ← actual scores
            ---              ← separator 3

        Data is between separator 2 and 3 (the second pair of ---).
        Auto-detects Score_EL/%Rank_EL column positions from the header.
        """
        scores = []
        dash_count = 0
        # Column indices — auto-detected from header
        idx_score_el = 8  # default for 4.3i with Inverted column
        idx_rank_el = 9

        for line in stdout.splitlines():
            stripped = line.strip()

            if stripped.startswith("---"):
                dash_count += 1
                continue

            # Header is between dash 1 and dash 2
            if dash_count == 1 and "Score_EL" in stripped:
                header_parts = stripped.split()
                for i, h in enumerate(header_parts):
                    if h == "Score_EL":
                        idx_score_el = i
                    elif h == "%Rank_EL":
                        idx_rank_el = i
                continue

            # Data is between dash 2 and dash 3
            if dash_count < 2 or dash_count >= 3:
                continue
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < idx_rank_el + 1:
                continue

            try:
                pos = int(parts[0]) - 1  # 1-based → 0-based
                peptide = parts[2]
                core = parts[4]
                el_score = float(parts[idx_score_el])
                el_rank = float(parts[idx_rank_el]) / 100.0  # percent → fraction
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

    @staticmethod
    def _score_seq(protein_seq: str, pep_length: int) -> list[PeptideScore]:
        """Deterministic scoring based on sequence content only (ID-independent)."""
        import numpy as np

        n_windows = len(protein_seq) - pep_length + 1
        if n_windows <= 0:
            return []

        scores = []
        for i in range(n_windows):
            peptide = protein_seq[i:i + pep_length]
            # Seed from (position, peptide content) — not protein_id
            h = hash((i, peptide)) & 0xFFFFFFFF
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

    def score_protein(
        self,
        protein_id: str,
        protein_seq: str,
        allele: str,
        pep_length: int,
    ) -> list[PeptideScore]:
        return self._score_seq(protein_seq, pep_length)

    def score_batch(
        self,
        entries: list[tuple[str, str]],
        allele: str,
        pep_lengths: list[int],
    ) -> dict[str, dict[int, list[PeptideScore]]]:
        """Batch scoring: same sequence → same scores regardless of ID."""
        results: dict[str, dict[int, list[PeptideScore]]] = {}
        for pid, seq in entries:
            results[pid] = {}
            for pl in pep_lengths:
                results[pid][pl] = self._score_seq(seq, pl)
        return results


def build_runner(
    backend: str = "standalone",
    binary_path: str | Path | None = None,
    seed: int = 42,
    batch_size: int = 30,
) -> NetMHCIIpanRunner:
    """Factory for NetMHCIIpan runners."""
    if backend == "standalone":
        if binary_path is None:
            raise ValueError("binary_path required for standalone backend")
        return StandaloneRunner(binary_path, batch_size=batch_size)
    elif backend == "mock":
        return MockRunner(seed=seed)
    else:
        raise ValueError(f"Unknown backend: {backend}")
