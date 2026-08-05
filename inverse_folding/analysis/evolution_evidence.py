"""Generic downstream conservation, coupling, Potts, and contact analysis.

This module consumes *normalized* query-column focus MSAs and an already fitted
PLMC model.  It intentionally does not search sequence databases or fit PLMC;
those expensive upstream steps have separate resource and provenance contracts.

Two operations are dependency-bound and therefore imported lazily:

* EVcouplings ``Alignment.set_weights`` for exact theta-based MSA weights;
* EVcouplings ``CouplingsModel`` for reading/scoring a binary PLMC model.

Everything else is deterministic NumPy/pandas/Biopython downstream analysis and
is covered by synthetic tests without requiring a large binary model fixture.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


AA20 = "ACDEFGHIKLMNPQRSTVWY"
ALLOWED_FOCUS_SYMBOLS = set(AA20 + "-")
DEFAULT_THETA = 0.8
DEFAULT_MIN_SEQUENCE_SEPARATION = 6
DEFAULT_PAIR_FRACTIONS = (0.5, 1.0, 2.0)
PAIR_LABELS = {0.5: "0p5L", 1.0: "L", 2.0: "2L"}
TOP_SET_LABELS = {0.5: "L/2", 1.0: "L", 2.0: "2L"}
HAMILTONIAN_COMPONENTS = ("total", "couplings", "fields")


@dataclass(frozen=True)
class SequenceRecord:
    """One FASTA row with both its stable identifier and full header."""

    identifier: str
    header: str
    sequence: str


@dataclass(frozen=True)
class FocusAlignment:
    """Validated, query-first, query-column MSA and exact sequence weights."""

    protein_id: str
    label: str
    coverage_threshold: float
    records: list[SequenceRecord]
    weights: np.ndarray
    source_path: Path

    @property
    def length(self) -> int:
        return len(self.records[0].sequence)

    @property
    def effective_samples(self) -> float:
        return float(np.sum(self.weights))


@dataclass(frozen=True)
class StructureContacts:
    """Canonical-position contact set derived through an explicit residue map."""

    kind: str
    structure_path: Path
    mapping_path: Path
    contact_cutoff_A: float
    contacts: set[tuple[int, int]]
    within_chain_contacts: set[tuple[int, int]]
    cross_chain_contacts: set[tuple[int, int]]
    resolved_positions: set[int]
    chains: tuple[str, ...]
    mapped_positions_by_chain: dict[str, tuple[int, ...]]


WeightProvider = Callable[[list[str], list[str], float], np.ndarray]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_md5(sequence: str) -> str:
    return hashlib.md5(sequence.encode("ascii")).hexdigest()


def read_fasta(path: Path) -> list[SequenceRecord]:
    """Read FASTA strictly, preserving order and rejecting empty records."""

    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[SequenceRecord] = []
    header: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = "".join(chunks)
                    if not sequence:
                        raise ValueError(f"{path}: empty FASTA record {header!r}")
                    records.append(
                        SequenceRecord(header.split()[0], header, sequence)
                    )
                header = line[1:].strip()
                chunks = []
                if not header:
                    raise ValueError(f"{path}:{line_number}: empty FASTA header")
            else:
                if header is None:
                    raise ValueError(
                        f"{path}:{line_number}: sequence encountered before FASTA header"
                    )
                chunks.append(line)
    if header is not None:
        sequence = "".join(chunks)
        if not sequence:
            raise ValueError(f"{path}: empty FASTA record {header!r}")
        records.append(SequenceRecord(header.split()[0], header, sequence))
    if not records:
        raise ValueError(f"{path}: no FASTA records")
    return records


def read_wt_fasta(path: Path, protein_id: str) -> str:
    records = read_fasta(path)
    if len(records) != 1:
        raise ValueError(f"{path}: WT FASTA must contain exactly one record, found {len(records)}")
    record = records[0]
    if record.identifier != protein_id:
        raise ValueError(
            f"{path}: WT FASTA identifier {record.identifier!r} != protein_id {protein_id!r}"
        )
    sequence = record.sequence.upper()
    invalid = sorted(set(sequence) - set(AA20))
    if invalid or record.sequence != sequence:
        raise ValueError(
            f"{path}: WT must be uppercase AA20 only; invalid={''.join(invalid)!r}"
        )
    return sequence


def evcouplings_sequence_weights(
    identifiers: list[str], sequences: list[str], theta: float
) -> np.ndarray:
    """Compute weights through EVcouplings' exact implementation."""

    try:
        from evcouplings.align import Alignment
    except ImportError as error:  # pragma: no cover - environment-specific boundary
        raise RuntimeError(
            "EVcouplings is required for exact Alignment.set_weights semantics. "
            "Run this CLI in the PLMC/EVcouplings environment used to fit the model."
        ) from error

    alignment = Alignment.from_dict(OrderedDict(zip(identifiers, sequences, strict=True)))
    alignment.set_weights(identity_threshold=theta)
    observed_ids = [str(identifier) for identifier in alignment.ids]
    if observed_ids != identifiers:
        raise RuntimeError(
            "EVcouplings changed MSA row order; cannot bind returned weights to input rows"
        )
    return np.asarray(alignment.weights, dtype=float)


def load_focus_alignment(
    path: Path,
    *,
    protein_id: str,
    wt: str,
    label: str,
    coverage_threshold: float,
    theta: float = DEFAULT_THETA,
    weight_provider: WeightProvider = evcouplings_sequence_weights,
) -> FocusAlignment:
    """Load one normalized MSA, rejecting every lossy-normalization symptom."""

    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError(f"coverage threshold must be in [0, 1], got {coverage_threshold}")
    records = read_fasta(path)
    identifiers = [record.identifier for record in records]
    duplicate_ids = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"{path}: duplicate sequence identifiers: {duplicate_ids[:10]}")
    if records[0].identifier != protein_id:
        raise ValueError(
            f"{path}: query must be first and have identifier {protein_id!r}; "
            f"found {records[0].identifier!r}"
        )
    if records[0].sequence != wt:
        raise ValueError(f"{path}: first query sequence differs from canonical WT")

    sequences = [record.sequence for record in records]
    bad_lengths = sorted({len(sequence) for sequence in sequences if len(sequence) != len(wt)})
    if bad_lengths:
        raise ValueError(
            f"{path}: focus-MSA rows must all have WT length {len(wt)}; "
            f"observed invalid lengths {bad_lengths}"
        )
    invalid_rows: list[tuple[str, str]] = []
    for record in records:
        invalid = sorted(set(record.sequence) - ALLOWED_FOCUS_SYMBOLS)
        if invalid:
            invalid_rows.append((record.identifier, "".join(invalid)))
    if invalid_rows:
        raise ValueError(
            f"{path}: expected normalized AA20 + gap ('-') only; invalid rows "
            f"{invalid_rows[:10]}"
        )
    coverages = np.asarray(
        [sum(char != "-" for char in sequence) / len(wt) for sequence in sequences],
        dtype=float,
    )
    below = np.flatnonzero(coverages + 1e-12 < coverage_threshold)
    if below.size:
        examples = [(identifiers[index], float(coverages[index])) for index in below[:10]]
        raise ValueError(
            f"{path}: rows below declared coverage threshold {coverage_threshold}: {examples}"
        )

    weights = np.asarray(weight_provider(identifiers, sequences, theta), dtype=float)
    if weights.shape != (len(records),):
        raise ValueError(
            f"{path}: weight provider returned shape {weights.shape}, expected {(len(records),)}"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0) or weights.sum() <= 0:
        raise ValueError(f"{path}: sequence weights must be finite and strictly positive")
    return FocusAlignment(
        protein_id=protein_id,
        label=label,
        coverage_threshold=float(coverage_threshold),
        records=records,
        weights=weights,
        source_path=path,
    )


def validate_focus_alignment_family(alignments: Sequence[FocusAlignment], wt: str) -> None:
    """Require one coherent nested family of coverage-filtered alignments."""

    if not alignments:
        raise ValueError("at least one focus alignment is required")
    protein_ids = {alignment.protein_id for alignment in alignments}
    if len(protein_ids) != 1:
        raise ValueError(f"focus alignments contain multiple protein IDs: {protein_ids}")
    labels = [alignment.label for alignment in alignments]
    thresholds = [alignment.coverage_threshold for alignment in alignments]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate coverage labels: {labels}")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError(f"duplicate coverage thresholds: {thresholds}")
    for alignment in alignments:
        if alignment.length != len(wt) or alignment.records[0].sequence != wt:
            raise ValueError(f"{alignment.label}: WT sequence/length mismatch")

    ordered = sorted(alignments, key=lambda alignment: alignment.coverage_threshold)
    previous = {record.identifier: record.sequence for record in ordered[0].records}
    for alignment in ordered[1:]:
        current = {record.identifier: record.sequence for record in alignment.records}
        extra = sorted(set(current) - set(previous))
        if extra:
            raise ValueError(
                f"{alignment.label}: higher-coverage alignment is not nested; extra IDs {extra[:10]}"
            )
        changed = sorted(
            identifier for identifier, sequence in current.items() if previous[identifier] != sequence
        )
        if changed:
            raise ValueError(
                f"{alignment.label}: shared IDs changed sequence across coverage filters: "
                f"{changed[:10]}"
            )
        previous = current


def _alignment_position_metrics(alignment: FocusAlignment, wt: str) -> pd.DataFrame:
    matrix = np.asarray([list(record.sequence) for record in alignment.records], dtype="U1")
    probabilities = alignment.weights / alignment.weights.sum()
    gap_fraction = (matrix == "-").T @ probabilities
    aa_frequencies = np.column_stack(
        [(matrix == aa).T @ probabilities for aa in AA20]
    )
    nongap_total = aa_frequencies.sum(axis=1)
    aa_normalized = np.divide(
        aa_frequencies,
        nongap_total[:, None],
        out=np.zeros_like(aa_frequencies),
        where=nongap_total[:, None] > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(
            np.where(aa_normalized > 0, aa_normalized * np.log2(aa_normalized), 0.0),
            axis=1,
        )
    conservation = 1.0 - entropy / np.log2(20.0)
    wt_indices = np.asarray([AA20.index(aa) for aa in wt], dtype=int)
    p_wt = aa_normalized[np.arange(len(wt)), wt_indices]
    return pd.DataFrame(
        {
            "protein_id": alignment.protein_id,
            "coverage_label": alignment.label,
            "coverage_threshold": alignment.coverage_threshold,
            "index_0b": np.arange(len(wt), dtype=int),
            "position_1b": np.arange(1, len(wt) + 1, dtype=int),
            "wt_aa": list(wt),
            "C_nogap": conservation,
            "pWT_nogap": p_wt,
            "gap_frac": gap_fraction,
            "n_sequences": len(alignment.records),
            "effective_samples": alignment.effective_samples,
            "neff_per_length": alignment.effective_samples / len(wt),
        }
    )


def compute_conservation_evidence(
    alignments: Sequence[FocusAlignment], wt: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Return long coverage evidence and one coverage-stability table."""

    validate_focus_alignment_family(alignments, wt)
    ordered = sorted(alignments, key=lambda alignment: alignment.coverage_threshold)
    frames = [_alignment_position_metrics(alignment, wt) for alignment in ordered]
    long = pd.concat(frames, ignore_index=True)

    base = frames[0][["protein_id", "index_0b", "position_1b", "wt_aa"]].copy()
    for frame, alignment in zip(frames, ordered, strict=True):
        suffix = alignment.label
        selected = frame[
            ["index_0b", "C_nogap", "pWT_nogap", "gap_frac"]
        ].rename(
            columns={
                "C_nogap": f"C_nogap_{suffix}",
                "pWT_nogap": f"pWT_nogap_{suffix}",
                "gap_frac": f"gap_frac_{suffix}",
            }
        )
        base = base.merge(selected, on="index_0b", validate="one_to_one")
    c_columns = [f"C_nogap_{alignment.label}" for alignment in ordered]
    p_columns = [f"pWT_nogap_{alignment.label}" for alignment in ordered]
    g_columns = [f"gap_frac_{alignment.label}" for alignment in ordered]
    base["C_nogap_min"] = base[c_columns].min(axis=1)
    base["C_nogap_max"] = base[c_columns].max(axis=1)
    base["C_nogap_range"] = base["C_nogap_max"] - base["C_nogap_min"]
    base["pWT_nogap_min"] = base[p_columns].min(axis=1)
    base["pWT_nogap_max"] = base[p_columns].max(axis=1)
    base["pWT_nogap_range"] = base["pWT_nogap_max"] - base["pWT_nogap_min"]
    base["gap_frac_max"] = base[g_columns].max(axis=1)
    base["gap_frac_min"] = base[g_columns].min(axis=1)

    metadata = [
        {
            "coverage_label": alignment.label,
            "coverage_threshold": alignment.coverage_threshold,
            "path": str(alignment.source_path),
            "sha256": sha256_file(alignment.source_path)
            if alignment.source_path.is_file()
            else None,
            "n_sequences": len(alignment.records),
            "effective_samples": alignment.effective_samples,
            "neff_per_length": alignment.effective_samples / len(wt),
        }
        for alignment in ordered
    ]
    return long, base, metadata


def read_complete_ec_table(path: Path, wt: str) -> pd.DataFrame:
    """Read a PLMC EC table and prove it contains every unordered site pair."""

    if not path.is_file():
        raise FileNotFoundError(path)
    first_tokens = path.read_text().splitlines()[0].split()
    try:
        int(first_tokens[0])
        headerless = True
    except (ValueError, IndexError):
        headerless = False
    if headerless:
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            names=["i", "A_i", "j", "A_j", "fn", "cn"],
            engine="python",
        )
    else:
        frame = pd.read_csv(path, sep=r"\s+", engine="python")
    required = {"i", "A_i", "j", "A_j", "cn"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: EC table missing columns {missing}")
    if frame.empty:
        raise ValueError(f"{path}: empty EC table")

    frame = frame.copy()
    for column in ("i", "j"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["cn"] = pd.to_numeric(frame["cn"], errors="raise").astype(float)
    if "fn" in frame:
        frame["fn"] = pd.to_numeric(frame["fn"], errors="raise").astype(float)
    if not np.isfinite(frame["cn"]).all():
        raise ValueError(f"{path}: EC cn values must be finite")

    swap = frame["i"] > frame["j"]
    if swap.any():
        old_i = frame.loc[swap, "i"].copy()
        old_ai = frame.loc[swap, "A_i"].copy()
        frame.loc[swap, "i"] = frame.loc[swap, "j"].to_numpy()
        frame.loc[swap, "A_i"] = frame.loc[swap, "A_j"].to_numpy()
        frame.loc[swap, "j"] = old_i.to_numpy()
        frame.loc[swap, "A_j"] = old_ai.to_numpy()
    length = len(wt)
    if (frame["i"] < 1).any() or (frame["j"] > length).any() or (frame["i"] >= frame["j"]).any():
        raise ValueError(f"{path}: EC positions must be unique unordered pairs within 1..{length}")
    duplicates = frame.duplicated(["i", "j"], keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, ["i", "j"]].head(10).values.tolist()
        raise ValueError(f"{path}: duplicate EC pairs: {examples}")
    expected_count = length * (length - 1) // 2
    if len(frame) != expected_count:
        raise ValueError(
            f"{path}: complete EC table required; expected {expected_count} pairs for L={length}, "
            f"found {len(frame)}"
        )
    expected_i = np.asarray([wt[position - 1] for position in frame["i"]])
    expected_j = np.asarray([wt[position - 1] for position in frame["j"]])
    mismatch = (frame["A_i"].astype(str).to_numpy() != expected_i) | (
        frame["A_j"].astype(str).to_numpy() != expected_j
    )
    if mismatch.any():
        examples = frame.loc[mismatch, ["i", "A_i", "j", "A_j"]].head(10)
        raise ValueError(f"{path}: EC target residues differ from WT:\n{examples}")
    frame = frame.sort_values(
        ["cn", "i", "j"], ascending=[False, True, True], kind="mergesort"
    ).reset_index(drop=True)
    frame["ec_rank_all"] = np.arange(1, len(frame) + 1, dtype=int)
    frame["sequence_separation"] = (frame["i"] - frame["j"]).abs()
    return frame


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def _sigma_for_top_pairs(
    long_range: pd.DataFrame, length: int, n_pairs: int
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(long_range) < n_pairs:
        raise ValueError(
            f"only {len(long_range)} long-range ECs available; need {n_pairs} for sigma"
        )
    top = long_range.iloc[:n_pairs].copy()
    average_strength = float(top["cn"].mean())
    if not np.isfinite(average_strength) or average_strength <= 0:
        raise ValueError(
            f"mean cn of top {n_pairs} long-range pairs must be positive and finite"
        )
    sums = np.zeros(length, dtype=float)
    np.add.at(sums, top["i"].to_numpy(dtype=int) - 1, top["cn"].to_numpy(dtype=float))
    np.add.at(sums, top["j"].to_numpy(dtype=int) - 1, top["cn"].to_numpy(dtype=float))
    return sums / average_strength, top


def _partners_by_position(top: pd.DataFrame, length: int, zero_based: bool) -> list[str]:
    partners: list[list[int]] = [[] for _ in range(length)]
    for row in top.itertuples(index=False):
        i, j = int(row.i), int(row.j)
        partners[i - 1].append(j - 1 if zero_based else j)
        partners[j - 1].append(i - 1 if zero_based else i)
    return [",".join(str(value) for value in values) for values in partners]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return math.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def compute_coupling_evidence(
    ecs: pd.DataFrame,
    wt: str,
    *,
    min_sequence_separation: int = DEFAULT_MIN_SEQUENCE_SEPARATION,
    pair_fractions: Sequence[float] = DEFAULT_PAIR_FRACTIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute EVcouplings enrichment at multiple cutoffs and stability metrics."""

    length = len(wt)
    required_fractions = set(DEFAULT_PAIR_FRACTIONS)
    if set(float(value) for value in pair_fractions) != required_fractions:
        raise ValueError(f"pair fractions must be exactly {sorted(required_fractions)}")
    ranked = ecs.copy().sort_values(
        ["cn", "i", "j"], ascending=[False, True, True], kind="mergesort"
    ).reset_index(drop=True)
    if "sequence_separation" not in ranked:
        ranked["sequence_separation"] = (ranked["i"] - ranked["j"]).abs()
    ranked["is_long_range"] = ranked["sequence_separation"] >= min_sequence_separation
    long_range = ranked.loc[ranked["is_long_range"]].copy()
    long_range["ec_rank_long_range"] = np.arange(1, len(long_range) + 1, dtype=int)
    ranked = ranked.merge(
        long_range[["i", "j", "ec_rank_long_range"]],
        on=["i", "j"],
        how="left",
        validate="one_to_one",
    )

    per_position = pd.DataFrame(
        {
            "index_0b": np.arange(length, dtype=int),
            "position_1b": np.arange(1, length + 1, dtype=int),
            "wt_aa": list(wt),
        }
    )
    sigma_by_label: dict[str, np.ndarray] = {}
    pct_by_label: dict[str, np.ndarray] = {}
    top_by_label: dict[str, pd.DataFrame] = {}
    for fraction in sorted(pair_fractions):
        label = PAIR_LABELS[float(fraction)]
        n_pairs = int(math.ceil(float(fraction) * length))
        sigma, top = _sigma_for_top_pairs(long_range, length, n_pairs)
        percentile = _percentile_rank(sigma)
        sigma_by_label[label] = sigma
        pct_by_label[label] = percentile
        top_by_label[label] = top
        per_position[f"sigma_{label}"] = sigma
        per_position[f"sigma_{label}_pct"] = percentile
        per_position[f"sigma_{label}_rank_desc"] = pd.Series(sigma).rank(
            method="min", ascending=False
        ).astype(int)
        per_position[f"top_{label}_partners_1b"] = _partners_by_position(
            top, length, zero_based=False
        )
        per_position[f"top_{label}_partners_0b"] = _partners_by_position(
            top, length, zero_based=True
        )
        per_position[f"in_top_{label}"] = per_position[f"top_{label}_partners_1b"].ne("")
        top_pairs = set(zip(top["i"].astype(int), top["j"].astype(int)))
        ranked[f"in_top_{label}"] = [
            (int(i), int(j)) in top_pairs for i, j in zip(ranked["i"], ranked["j"], strict=True)
        ]
    per_position["sigma_robust_pct"] = np.min(
        np.column_stack([pct_by_label[label] for label in ("0p5L", "L", "2L")]),
        axis=1,
    )

    stability_rows: list[dict[str, object]] = []
    labels = ("0p5L", "L", "2L")
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            stability_rows.append(
                {
                    "metric": "spearman",
                    "cutoff_left": left,
                    "cutoff_right": right,
                    "percentile_threshold": np.nan,
                    "value": _spearman(sigma_by_label[left], sigma_by_label[right]),
                }
            )
            for threshold, metric in ((0.9, "jaccard_top10"), (0.8, "jaccard_top20")):
                left_set = set(np.flatnonzero(pct_by_label[left] >= threshold).tolist())
                right_set = set(np.flatnonzero(pct_by_label[right] >= threshold).tolist())
                stability_rows.append(
                    {
                        "metric": metric,
                        "cutoff_left": left,
                        "cutoff_right": right,
                        "percentile_threshold": threshold,
                        "value": _jaccard(left_set, right_set),
                    }
                )
    return per_position, ranked, pd.DataFrame(stability_rows)


def build_lock_masks(
    per_position: pd.DataFrame, *, covariance_status: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build exact analytical WT-lock tiers with explicit mask status."""

    allowed_status = {"qualified", "exploratory", "insufficient"}
    if covariance_status not in allowed_status:
        raise ValueError(
            f"covariance_status must be one of {sorted(allowed_status)}, got {covariance_status!r}"
        )
    required = {
        "protein_id",
        "index_0b",
        "position_1b",
        "wt_aa",
        "C_nogap_min",
        "pWT_nogap_min",
        "gap_frac_max",
        "sigma_L_pct",
        "sigma_robust_pct",
    }
    missing = sorted(required - set(per_position.columns))
    if missing:
        raise ValueError(f"per-position evidence missing columns {missing}")
    protein_ids = per_position["protein_id"].drop_duplicates().tolist()
    if len(protein_ids) != 1:
        raise ValueError(f"expected exactly one protein_id, found {protein_ids}")
    protein_id = str(protein_ids[0])
    frame = per_position.copy()
    expected_indices = list(range(len(frame)))
    if frame["index_0b"].astype(int).tolist() != expected_indices:
        raise ValueError("per-position evidence must be ordered contiguously by index_0b")
    if frame["position_1b"].astype(int).tolist() != [value + 1 for value in expected_indices]:
        raise ValueError("position_1b must equal index_0b + 1")

    definitions: list[dict[str, object]] = []
    for threshold, name in ((0.9, "C90_stable"), (0.8, "C80_stable"), (0.7, "C70_stable")):
        definitions.append(
            {
                "name": name,
                "kind": "conservation",
                "metric": "min(C_nogap,pWT_nogap)_across_coverage_filters",
                "threshold": threshold,
                "gap_cap": 0.2,
                "mask": (frame["C_nogap_min"] >= threshold)
                & (frame["pWT_nogap_min"] >= threshold)
                & (frame["gap_frac_max"] <= 0.2),
                "status": "available",
            }
        )
    sigma_status = {
        "qualified": "qualified",
        "exploratory": "exploratory_not_hard_lock",
        "insufficient": "suppressed_low_neff",
    }[covariance_status]
    for threshold, suffix in ((0.9, "90"), (0.8, "80")):
        for metric, postfix in (("sigma_L_pct", "L"), ("sigma_robust_pct", "robust")):
            mask = (frame[metric] >= threshold) & (frame["gap_frac_max"] <= 0.5)
            if covariance_status == "insufficient":
                mask = pd.Series(False, index=frame.index)
            definitions.append(
                {
                    "name": f"sigma{suffix}_{postfix}",
                    "kind": "covariance",
                    "metric": metric,
                    "threshold": threshold,
                    "gap_cap": 0.5,
                    "mask": mask,
                    "status": sigma_status,
                }
            )

    summaries: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    masks_json: dict[str, dict[str, object]] = {}
    for definition in definitions:
        name = str(definition["name"])
        selected = frame.loc[pd.Series(definition["mask"], index=frame.index)].copy()
        frame[f"in_{name}"] = frame.index.isin(selected.index)
        positions_0b = selected["index_0b"].astype(int).tolist()
        positions_1b = selected["position_1b"].astype(int).tolist()
        labels = [
            f"{aa}{position}"
            for aa, position in zip(selected["wt_aa"], positions_1b, strict=True)
        ]
        record = {
            "protein_id": protein_id,
            "mask": name,
            "kind": definition["kind"],
            "status": definition["status"],
            "metric": definition["metric"],
            "threshold": float(definition["threshold"]),
            "gap_cap": float(definition["gap_cap"]),
            "n_positions": len(selected),
            "positions_index_0b_json": json.dumps(positions_0b),
            "positions_position_1b_json": json.dumps(positions_1b),
            "position_labels_json": json.dumps(labels),
        }
        summaries.append(record)
        for row in selected.itertuples(index=False):
            position_rows.append(
                {
                    "protein_id": protein_id,
                    "mask": name,
                    "kind": definition["kind"],
                    "status": definition["status"],
                    "index_0b": int(row.index_0b),
                    "position_1b": int(row.position_1b),
                    "wt_aa": str(row.wt_aa),
                }
            )
        masks_json[name] = {
            "kind": definition["kind"],
            "status": definition["status"],
            "metric": definition["metric"],
            "threshold": float(definition["threshold"]),
            "gap_cap": float(definition["gap_cap"]),
            "n_positions": len(selected),
            "positions_index_0b": positions_0b,
            "positions_position_1b": positions_1b,
            "position_labels": labels,
        }
    positions_columns = [
        "protein_id",
        "mask",
        "kind",
        "status",
        "index_0b",
        "position_1b",
        "wt_aa",
    ]
    position_table = pd.DataFrame(position_rows, columns=positions_columns)
    payload: dict[str, object] = {
        "schema_version": 1,
        "protein_id": protein_id,
        "numbering": {
            "index_0b": "zero-based protein position",
            "position_1b": "one-based protein position; position_1b = index_0b + 1",
        },
        "covariance_status": covariance_status,
        "masks": masks_json,
    }
    return frame, pd.DataFrame(summaries), position_table, payload


def write_mask_artifacts(
    out_dir: Path,
    summary: pd.DataFrame,
    positions: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "wt_lock_mask_summary.tsv", sep="\t", index=False)
    positions.to_csv(out_dir / "wt_lock_mask_positions.tsv", sep="\t", index=False)
    (out_dir / "wt_lock_masks.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def load_couplings_model(path: Path):
    try:
        from evcouplings.couplings.model import CouplingsModel
    except ImportError as error:  # pragma: no cover - environment-specific boundary
        raise RuntimeError(
            "EVcouplings is required to load and score a PLMC binary model. "
            "Run this CLI in the PLMC/EVcouplings environment used to fit the model."
        ) from error
    return CouplingsModel(str(path))


def validate_couplings_model(model, wt: str) -> dict[str, object]:
    length = int(model.L)
    if length != len(wt):
        raise ValueError(f"PLMC model L={length} differs from WT length {len(wt)}")
    indices = [int(value) for value in model.index_list]
    if indices != list(range(1, len(wt) + 1)):
        raise ValueError("PLMC model indices must be exactly canonical positions 1..L")
    target = "".join(str(value) for value in model.target_seq)
    if target != wt:
        raise ValueError("PLMC model target sequence differs from canonical WT")
    alphabet = [str(value) for value in model.alphabet]
    missing = sorted(set(wt) - set(alphabet))
    if missing:
        raise ValueError(f"PLMC alphabet does not contain WT residues: {missing}")
    n_eff = float(model.N_eff)
    if not np.isfinite(n_eff) or n_eff <= 0:
        raise ValueError(f"PLMC model N_eff must be positive and finite, got {n_eff}")
    return {
        "length": length,
        "target_sequence_md5": sequence_md5(target),
        "index_start": indices[0],
        "index_end": indices[-1],
        "alphabet": "".join(alphabet),
        "effective_samples": n_eff,
        "neff_per_length": n_eff / length,
        "n_valid": int(model.N_valid),
    }


def covariance_qualification(neff_per_length: float) -> str:
    if neff_per_length >= 10.0:
        return "qualified"
    if neff_per_length >= 5.0:
        return "exploratory"
    return "insufficient"


def _score_sequences(model, sequences: Sequence[str], chunk_size: int) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not sequences:
        return np.empty((0, 3), dtype=float)
    blocks: list[np.ndarray] = []
    for start in range(0, len(sequences), chunk_size):
        values = np.asarray(model.hamiltonians(list(sequences[start : start + chunk_size])), dtype=float)
        if values.shape != (min(chunk_size, len(sequences) - start), 3):
            raise ValueError(f"CouplingsModel.hamiltonians returned unexpected shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("CouplingsModel.hamiltonians returned non-finite values")
        blocks.append(values)
    return np.vstack(blocks)


def _empirical_percentile(value: float, values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    left = int(np.searchsorted(np.sort(values), value, side="left"))
    right = int(np.searchsorted(np.sort(values), value, side="right"))
    return float(100.0 * (left + right) / (2.0 * len(values)))


def _weighted_empirical_percentile(
    value: float, values: np.ndarray, weights: np.ndarray
) -> float | None:
    if values.size == 0:
        return None
    total = float(weights.sum())
    below = float(weights[values < value].sum())
    tied = float(weights[values == value].sum())
    return 100.0 * (below + 0.5 * tied) / total


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = q * float(sorted_weights.sum())
    index = min(int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left")), len(values) - 1)
    return float(sorted_values[index])


def _distribution_summary(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, object]:
    if values.size == 0:
        return {"n": 0}
    quantile_grid = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    result: dict[str, object] = {
        "n": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    if weights is None:
        result["mean"] = float(np.mean(values))
        for q in quantile_grid:
            result[f"q{int(round(100 * q)):02d}"] = float(np.quantile(values, q))
    else:
        result["weight_sum"] = float(weights.sum())
        result["mean"] = float(np.average(values, weights=weights))
        for q in quantile_grid:
            result[f"q{int(round(100 * q)):02d}"] = _weighted_quantile(values, weights, q)
    return result


def calibrate_potts(
    model,
    natural_alignment: FocusAlignment,
    wt: str,
    *,
    protein_id: str,
    minimum_coverage: float = 0.95,
    minimum_unique_gate: int | None = None,
    chunk_size: int = 256,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score WT and high-coverage natural rows only within one PLMC model."""

    model_metadata = validate_couplings_model(model, wt)
    model_metadata["covariance_status"] = covariance_qualification(
        float(model_metadata["neff_per_length"])
    )
    if natural_alignment.protein_id != protein_id:
        raise ValueError("natural alignment protein_id differs from requested protein_id")
    if natural_alignment.records[0].sequence != wt:
        raise ValueError("natural alignment query differs from PLMC WT")
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in [0, 1]")
    if minimum_unique_gate is not None and minimum_unique_gate <= 0:
        raise ValueError("minimum_unique_gate must be positive when supplied")

    selected: list[dict[str, object]] = []
    for row_index, (record, weight) in enumerate(
        zip(natural_alignment.records, natural_alignment.weights, strict=True)
    ):
        n_gaps = record.sequence.count("-")
        coverage = 1.0 - n_gaps / len(wt)
        if coverage + 1e-12 < minimum_coverage:
            continue
        imputed = "".join(
            wt[index] if aa == "-" else aa for index, aa in enumerate(record.sequence)
        )
        selected.append(
            {
                "protein_id": protein_id,
                "sequence_id": record.identifier,
                "source_row_index": row_index,
                "is_query": row_index == 0,
                "source_coverage_label": natural_alignment.label,
                "minimum_coverage_threshold": minimum_coverage,
                "coverage_before_imputation": coverage,
                "n_gaps_before_imputation": n_gaps,
                "sequence_original": record.sequence,
                "sequence_original_md5": sequence_md5(record.sequence),
                "sequence_imputed": imputed,
                "sequence_imputed_md5": sequence_md5(imputed),
                "msa_weight": float(weight),
                "n_mismatches_vs_wt": sum(a != b for a, b in zip(wt, imputed, strict=True)),
            }
        )

    wt_scores = _score_sequences(model, [wt], chunk_size)[0]
    natural_scores = _score_sequences(
        model, [str(row["sequence_imputed"]) for row in selected], chunk_size
    )
    for row, scores in zip(selected, natural_scores, strict=True):
        for component, score, wt_score in zip(
            HAMILTONIAN_COMPONENTS, scores, wt_scores, strict=True
        ):
            row[f"H_{component}"] = float(score)
            row[f"delta_H_{component}_vs_WT"] = float(score - wt_score)
    columns = [
        "protein_id",
        "sequence_id",
        "source_row_index",
        "is_query",
        "source_coverage_label",
        "minimum_coverage_threshold",
        "coverage_before_imputation",
        "n_gaps_before_imputation",
        "sequence_original",
        "sequence_original_md5",
        "sequence_imputed",
        "sequence_imputed_md5",
        "msa_weight",
        "n_mismatches_vs_wt",
        "H_total",
        "H_couplings",
        "H_fields",
        "delta_H_total_vs_WT",
        "delta_H_couplings_vs_WT",
        "delta_H_fields_vs_WT",
    ]
    per_sequence = pd.DataFrame(selected, columns=columns)
    total_values = per_sequence["H_total"].to_numpy(dtype=float) if len(per_sequence) else np.asarray([])
    weights = per_sequence["msa_weight"].to_numpy(dtype=float) if len(per_sequence) else np.asarray([])
    n_unique = int(per_sequence["sequence_imputed"].nunique()) if len(per_sequence) else 0
    if n_unique == 0:
        calibration_status = "insufficient"
    elif minimum_unique_gate is None:
        calibration_status = "descriptive_gate_not_supplied"
    elif n_unique < minimum_unique_gate:
        calibration_status = "insufficient"
    else:
        calibration_status = "sufficient"

    component_summaries: dict[str, object] = {}
    for component in HAMILTONIAN_COMPONENTS:
        values = (
            per_sequence[f"H_{component}"].to_numpy(dtype=float)
            if len(per_sequence)
            else np.asarray([])
        )
        component_summaries[component] = {
            "unweighted_rows": _distribution_summary(values),
            "msa_weighted_rows": _distribution_summary(values, weights),
        }
    summary: dict[str, object] = {
        "schema_version": 1,
        "protein_id": protein_id,
        "comparison_scope": f"within_model_only:{protein_id}",
        "cross_parent_thresholds_emitted": False,
        "score_status": "descriptive_available_regardless_of_covariance_depth",
        "score_definition": "H_total = H_couplings + H_fields from CouplingsModel.hamiltonians",
        "score_semantics": "higher H is more model-compatible; delta_H = H(sequence) - H(own WT)",
        "model": model_metadata,
        "WT": {
            f"H_{component}": float(value)
            for component, value in zip(HAMILTONIAN_COMPONENTS, wt_scores, strict=True)
        },
        "natural_reference": {
            "source_coverage_label": natural_alignment.label,
            "minimum_coverage": minimum_coverage,
            "n_rows_input": len(natural_alignment.records),
            "n_rows_excluded_below_coverage": len(natural_alignment.records) - len(per_sequence),
            "gap_handling": "WT-impute each remaining gap after coverage filtering",
            "n_rows": len(per_sequence),
            "n_unique_imputed_sequences": n_unique,
            "minimum_unique_gate": minimum_unique_gate,
            "WT_percentile_total_unweighted_rows": _empirical_percentile(
                float(wt_scores[0]), total_values
            ),
            "WT_percentile_total_msa_weighted_rows": _weighted_empirical_percentile(
                float(wt_scores[0]), total_values, weights
            ),
            "component_distributions": component_summaries,
        },
        "calibration_status": calibration_status,
    }
    return per_sequence, summary


AA3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def _structure_parser(path: Path):
    try:
        from Bio.PDB import MMCIFParser, PDBParser
    except ImportError as error:  # pragma: no cover - project dependency boundary
        raise RuntimeError("Biopython is required for optional EC contact validation") from error
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def load_structure_contacts(
    structure_path: Path,
    mapping_path: Path,
    wt: str,
    *,
    kind: str,
    contact_cutoff_A: float = 5.0,
) -> StructureContacts:
    """Map a monomer/tetramer structure to canonical positions and find contacts."""

    if kind not in {"monomer", "tetramer"}:
        raise ValueError("structure kind must be 'monomer' or 'tetramer'")
    expected_chain_count = 1 if kind == "monomer" else 4
    if contact_cutoff_A <= 0:
        raise ValueError("contact cutoff must be positive")
    if not structure_path.is_file():
        raise FileNotFoundError(structure_path)
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)
    mapping = pd.read_csv(mapping_path, sep="\t", dtype={"chain_id": str})
    required = {"chain_id", "residue_number", "position_1b"}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"{mapping_path}: residue map missing columns {missing}")
    if mapping.empty:
        raise ValueError(f"{mapping_path}: empty residue map")
    mapping = mapping.copy()
    mapping["chain_id"] = mapping["chain_id"].astype(str)
    mapping["residue_number"] = pd.to_numeric(
        mapping["residue_number"], errors="raise"
    ).astype(int)
    mapping["position_1b"] = pd.to_numeric(mapping["position_1b"], errors="raise").astype(int)
    if "insertion_code" not in mapping:
        mapping["insertion_code"] = ""
    mapping["insertion_code"] = mapping["insertion_code"].fillna("").astype(str).str.strip()
    if not mapping["position_1b"].between(1, len(wt)).all():
        raise ValueError(f"{mapping_path}: position_1b must lie within 1..{len(wt)}")
    if mapping.duplicated(["chain_id", "residue_number", "insertion_code"]).any():
        raise ValueError(f"{mapping_path}: duplicate structure residue keys")
    if mapping.duplicated(["chain_id", "position_1b"]).any():
        raise ValueError(f"{mapping_path}: duplicate canonical position within a chain")
    chains = tuple(sorted(mapping["chain_id"].unique().tolist()))
    if len(chains) != expected_chain_count:
        raise ValueError(
            f"{mapping_path}: {kind} mapping must contain exactly {expected_chain_count} chains; "
            f"found {chains}"
        )
    if "expected_aa" in mapping:
        expected = np.asarray([wt[position - 1] for position in mapping["position_1b"]])
        mismatch = mapping["expected_aa"].astype(str).to_numpy() != expected
        if mismatch.any():
            raise ValueError(f"{mapping_path}: expected_aa differs from canonical WT")

    structure = _structure_parser(structure_path).get_structure(structure_path.stem, structure_path)
    models = list(structure)
    if len(models) != 1:
        raise ValueError(f"{structure_path}: expected exactly one structural model, found {len(models)}")
    model = models[0]
    residue_lookup: dict[tuple[str, int, str], object] = {}
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            key = (str(chain.id), int(residue.id[1]), str(residue.id[2]).strip())
            if key in residue_lookup:
                raise ValueError(f"{structure_path}: duplicate parsed residue key {key}")
            residue_lookup[key] = residue

    residue_to_mapping: dict[object, tuple[str, int]] = {}
    mapped_by_chain: dict[str, list[int]] = {chain: [] for chain in chains}
    for row in mapping.itertuples(index=False):
        key = (str(row.chain_id), int(row.residue_number), str(row.insertion_code).strip())
        residue = residue_lookup.get(key)
        if residue is None:
            raise ValueError(f"{mapping_path}: mapped structure residue not found: {key}")
        actual = AA3_TO_1.get(str(residue.resname).upper())
        if actual is None:
            raise ValueError(f"{structure_path}: mapped residue {key} is not standard AA20")
        expected = wt[int(row.position_1b) - 1]
        if actual != expected:
            raise ValueError(
                f"{structure_path}: residue {key} is {actual}, expected WT {expected} at "
                f"position {int(row.position_1b)}"
            )
        residue_to_mapping[residue] = (str(row.chain_id), int(row.position_1b))
        mapped_by_chain[str(row.chain_id)].append(int(row.position_1b))

    try:
        from Bio.PDB import NeighborSearch
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Biopython is required for optional EC contact validation") from error
    atoms = [
        atom
        for residue in residue_to_mapping
        for atom in residue
        if (atom.element or "").upper() != "H"
    ]
    neighbor_search = NeighborSearch(atoms)
    within: set[tuple[int, int]] = set()
    cross: set[tuple[int, int]] = set()
    for residue_a, residue_b in neighbor_search.search_all(contact_cutoff_A, level="R"):
        chain_a, position_a = residue_to_mapping[residue_a]
        chain_b, position_b = residue_to_mapping[residue_b]
        if position_a == position_b:
            continue
        pair = tuple(sorted((position_a, position_b)))
        if chain_a == chain_b:
            within.add(pair)
        else:
            cross.add(pair)
    contacts = within if kind == "monomer" else within | cross
    return StructureContacts(
        kind=kind,
        structure_path=structure_path,
        mapping_path=mapping_path,
        contact_cutoff_A=float(contact_cutoff_A),
        contacts=contacts,
        within_chain_contacts=within,
        cross_chain_contacts=cross,
        resolved_positions=set(mapping["position_1b"].astype(int)),
        chains=chains,
        mapped_positions_by_chain={
            chain: tuple(sorted(positions)) for chain, positions in mapped_by_chain.items()
        },
    )


def _select_scorable_pairs(
    long_range: pd.DataFrame,
    resolved_positions: set[int],
    contacts: set[tuple[int, int]],
    n_top: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected: list[dict[str, object]] = []
    skipped = 0
    for global_rank, row in enumerate(long_range.itertuples(index=False), start=1):
        i, j = int(row.i), int(row.j)
        if i not in resolved_positions or j not in resolved_positions:
            skipped += 1
            continue
        pair = tuple(sorted((i, j)))
        selected.append(
            {
                "position_i_1b": pair[0],
                "index_i_0b": pair[0] - 1,
                "position_j_1b": pair[1],
                "index_j_0b": pair[1] - 1,
                "cn": float(row.cn),
                "global_long_range_rank": global_rank,
                "is_contact": pair in contacts,
            }
        )
        if len(selected) == n_top:
            break
    if len(selected) != n_top:
        raise ValueError(
            f"only {len(selected)} structurally scorable ECs available; requested {n_top}"
        )
    hits = sum(bool(row["is_contact"]) for row in selected)
    return (
        {
            "requested_k": n_top,
            "scored": len(selected),
            "hits": hits,
            "precision": hits / len(selected),
            "skipped_before_k_scorable": skipped,
            "last_global_long_range_rank": selected[-1]["global_long_range_rank"],
        },
        selected,
    )


def compute_contact_precision(
    ecs: pd.DataFrame,
    monomer: StructureContacts | None,
    tetramer: StructureContacts | None,
    wt: str,
    *,
    min_sequence_separation: int = DEFAULT_MIN_SEQUENCE_SEPARATION,
    pair_fractions: Sequence[float] = DEFAULT_PAIR_FRACTIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score identical EC ranks against monomer and/or biological tetramer contacts."""

    if monomer is None and tetramer is None:
        raise ValueError("at least one structure contact map is required")
    ranked = ecs.copy().sort_values(
        ["cn", "i", "j"], ascending=[False, True, True], kind="mergesort"
    )
    long_range = ranked.loc[(ranked["i"] - ranked["j"]).abs() >= min_sequence_separation]
    length = len(wt)
    precision_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for structure in (monomer, tetramer):
        if structure is None:
            continue
        for fraction in sorted(pair_fractions):
            n_top = int(math.ceil(float(fraction) * length))
            measurement, selected = _select_scorable_pairs(
                long_range, structure.resolved_positions, structure.contacts, n_top
            )
            top_set = TOP_SET_LABELS[float(fraction)]
            precision_rows.append(
                {
                    "structure_kind": structure.kind,
                    "top_set": top_set,
                    "pair_fraction_of_L": float(fraction),
                    **measurement,
                    "contact_cutoff_A": structure.contact_cutoff_A,
                    "min_sequence_separation": min_sequence_separation,
                }
            )
            for row in selected:
                pair_rows.append(
                    {
                        "structure_kind": structure.kind,
                        "top_set": top_set,
                        "pair_fraction_of_L": float(fraction),
                        **row,
                    }
                )

    comparison_rows: list[dict[str, object]] = []
    if monomer is not None and tetramer is not None:
        if not math.isclose(monomer.contact_cutoff_A, tetramer.contact_cutoff_A):
            raise ValueError("monomer and tetramer contact cutoffs must match for paired precision")
        common_resolved = monomer.resolved_positions & tetramer.resolved_positions
        for fraction in sorted(pair_fractions):
            n_top = int(math.ceil(float(fraction) * length))
            mono_measurement, selected = _select_scorable_pairs(
                long_range, common_resolved, monomer.contacts, n_top
            )
            tetra_measurement, tetra_selected = _select_scorable_pairs(
                long_range, common_resolved, tetramer.contacts, n_top
            )
            mono_pairs = [
                (row["position_i_1b"], row["position_j_1b"]) for row in selected
            ]
            tetra_pairs = [
                (row["position_i_1b"], row["position_j_1b"]) for row in tetra_selected
            ]
            if mono_pairs != tetra_pairs:
                raise RuntimeError("paired contact comparison did not select identical EC ranks")
            tetramer_only = sum(
                pair in tetramer.contacts and pair not in monomer.contacts for pair in mono_pairs
            )
            comparison_rows.append(
                {
                    "top_set": TOP_SET_LABELS[float(fraction)],
                    "pair_fraction_of_L": float(fraction),
                    "requested_k": n_top,
                    "common_resolved_positions_n": len(common_resolved),
                    "monomer_hits": mono_measurement["hits"],
                    "monomer_precision": mono_measurement["precision"],
                    "tetramer_hits": tetra_measurement["hits"],
                    "tetramer_precision": tetra_measurement["precision"],
                    "tetramer_minus_monomer_precision": float(tetra_measurement["precision"])
                    - float(mono_measurement["precision"]),
                    "tetramer_only_hits": tetramer_only,
                }
            )
    precision = pd.DataFrame(precision_rows)
    pairs = pd.DataFrame(pair_rows)
    comparison = pd.DataFrame(comparison_rows)
    return precision, pairs, comparison


def run_analysis(
    *,
    protein_id: str,
    wt_fasta: Path,
    msa_specs: Sequence[tuple[str, float, Path]],
    model_path: Path,
    ec_table_path: Path,
    out_dir: Path,
    theta: float = DEFAULT_THETA,
    natural_minimum_coverage: float = 0.95,
    minimum_calibration_unique: int | None = None,
    min_sequence_separation: int = DEFAULT_MIN_SEQUENCE_SEPARATION,
    monomer_structure: Path | None = None,
    monomer_mapping: Path | None = None,
    tetramer_structure: Path | None = None,
    tetramer_mapping: Path | None = None,
    contact_cutoff_A: float = 5.0,
    score_chunk_size: int = 256,
    weight_provider: WeightProvider = evcouplings_sequence_weights,
    model_loader: Callable[[Path], object] = load_couplings_model,
) -> dict[str, object]:
    """Execute the complete downstream analysis and persist its full schema."""

    if len(msa_specs) != 3:
        raise ValueError("exactly three normalized focus MSAs are required")
    expected_specs = {("cov60", 0.6), ("cov70", 0.7), ("cov80", 0.8)}
    observed_specs = {(label, float(threshold)) for label, threshold, _ in msa_specs}
    if observed_specs != expected_specs:
        raise ValueError(
            f"MSA specs must be cov60=0.6, cov70=0.7, cov80=0.8; got {observed_specs}"
        )
    structure_pairs = (
        ("monomer", monomer_structure, monomer_mapping),
        ("tetramer", tetramer_structure, tetramer_mapping),
    )
    for kind, structure, mapping in structure_pairs:
        if (structure is None) != (mapping is None):
            raise ValueError(f"{kind} structure and residue mapping must be supplied together")

    wt = read_wt_fasta(wt_fasta, protein_id)
    alignments = [
        load_focus_alignment(
            path,
            protein_id=protein_id,
            wt=wt,
            label=label,
            coverage_threshold=threshold,
            theta=theta,
            weight_provider=weight_provider,
        )
        for label, threshold, path in msa_specs
    ]
    validate_focus_alignment_family(alignments, wt)
    conservation_long, conservation_stable, alignment_metadata = (
        compute_conservation_evidence(alignments, wt)
    )

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = model_loader(model_path)
    model_metadata = validate_couplings_model(model, wt)
    covariance_status = covariance_qualification(float(model_metadata["neff_per_length"]))
    ecs = read_complete_ec_table(ec_table_path, wt)
    coupling, ranked_ecs, sigma_stability = compute_coupling_evidence(
        ecs, wt, min_sequence_separation=min_sequence_separation
    )
    per_position = conservation_stable.merge(
        coupling, on=["index_0b", "position_1b", "wt_aa"], validate="one_to_one"
    )
    per_position.insert(1, "covariance_status", covariance_status)
    per_position, mask_summary, mask_positions, masks_payload = build_lock_masks(
        per_position, covariance_status=covariance_status
    )

    natural_alignment = next(
        alignment for alignment in alignments if alignment.label == "cov60"
    )
    potts_per_sequence, potts_summary = calibrate_potts(
        model,
        natural_alignment,
        wt,
        protein_id=protein_id,
        minimum_coverage=natural_minimum_coverage,
        minimum_unique_gate=minimum_calibration_unique,
        chunk_size=score_chunk_size,
    )

    monomer = (
        load_structure_contacts(
            monomer_structure,
            monomer_mapping,
            wt,
            kind="monomer",
            contact_cutoff_A=contact_cutoff_A,
        )
        if monomer_structure is not None and monomer_mapping is not None
        else None
    )
    tetramer = (
        load_structure_contacts(
            tetramer_structure,
            tetramer_mapping,
            wt,
            kind="tetramer",
            contact_cutoff_A=contact_cutoff_A,
        )
        if tetramer_structure is not None and tetramer_mapping is not None
        else None
    )
    contact_precision: pd.DataFrame | None = None
    contact_pairs: pd.DataFrame | None = None
    contact_comparison: pd.DataFrame | None = None
    if monomer is not None or tetramer is not None:
        contact_precision, contact_pairs, contact_comparison = compute_contact_precision(
            ecs,
            monomer,
            tetramer,
            wt,
            min_sequence_separation=min_sequence_separation,
        )
        for table in (contact_precision, contact_pairs, contact_comparison):
            if table is not None and "protein_id" not in table:
                table.insert(0, "protein_id", protein_id)
                table.insert(1, "covariance_status", covariance_status)

    out_dir.mkdir(parents=True, exist_ok=True)
    conservation_long.to_csv(
        out_dir / "conservation_by_coverage.tsv", sep="\t", index=False, float_format="%.10g"
    )
    per_position.to_csv(
        out_dir / "per_position_evolution.tsv", sep="\t", index=False, float_format="%.10g"
    )
    ranked_output = ranked_ecs.copy()
    ranked_output.insert(0, "protein_id", protein_id)
    ranked_output.insert(1, "covariance_status", covariance_status)
    ranked_output.insert(2, "index_i_0b", ranked_output["i"].astype(int) - 1)
    ranked_output.insert(3, "position_i_1b", ranked_output["i"].astype(int))
    ranked_output.insert(4, "index_j_0b", ranked_output["j"].astype(int) - 1)
    ranked_output.insert(5, "position_j_1b", ranked_output["j"].astype(int))
    ranked_output.to_csv(
        out_dir / "complete_ec_table.tsv", sep="\t", index=False, float_format="%.10g"
    )
    sigma_stability.insert(0, "protein_id", protein_id)
    sigma_stability.to_csv(
        out_dir / "sigma_cutoff_stability.tsv", sep="\t", index=False, float_format="%.10g"
    )
    write_mask_artifacts(out_dir, mask_summary, mask_positions, masks_payload)
    potts_per_sequence.to_csv(
        out_dir / "potts_calibration_per_sequence.tsv",
        sep="\t",
        index=False,
        float_format="%.10g",
    )
    (out_dir / "potts_calibration_summary.json").write_text(
        json.dumps(potts_summary, indent=2, sort_keys=True) + "\n"
    )
    if contact_precision is not None:
        contact_precision.to_csv(
            out_dir / "ec_contact_precision.tsv", sep="\t", index=False, float_format="%.10g"
        )
        contact_pairs.to_csv(
            out_dir / "ec_contact_selected_pairs.tsv",
            sep="\t",
            index=False,
            float_format="%.10g",
        )
        contact_comparison.to_csv(
            out_dir / "ec_contact_monomer_tetramer_comparison.tsv",
            sep="\t",
            index=False,
            float_format="%.10g",
        )

    structure_metadata: dict[str, object] = {}
    for structure in (monomer, tetramer):
        if structure is None:
            continue
        structure_metadata[structure.kind] = {
            "structure_path": str(structure.structure_path),
            "structure_sha256": sha256_file(structure.structure_path),
            "mapping_path": str(structure.mapping_path),
            "mapping_sha256": sha256_file(structure.mapping_path),
            "contact_cutoff_A": structure.contact_cutoff_A,
            "chains": list(structure.chains),
            "mapped_positions_by_chain": {
                chain: list(positions)
                for chain, positions in structure.mapped_positions_by_chain.items()
            },
            "resolved_positions_n": len(structure.resolved_positions),
            "within_chain_contacts_n": len(structure.within_chain_contacts),
            "cross_chain_contacts_n": len(structure.cross_chain_contacts),
        }
    tetramer_top_l_precision: float | None = None
    if contact_precision is not None and tetramer is not None:
        row = contact_precision.loc[
            contact_precision["structure_kind"].eq("tetramer")
            & contact_precision["top_set"].eq("L")
        ]
        if len(row) == 1:
            tetramer_top_l_precision = float(row.iloc[0]["precision"])
    metadata: dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protein_id": protein_id,
        "sequence": {
            "length": len(wt),
            "md5": sequence_md5(wt),
            "wt_fasta": str(wt_fasta),
            "wt_fasta_sha256": sha256_file(wt_fasta),
            "numbering": {
                "index_0b": "zero-based protein position",
                "position_1b": "one-based protein position; position_1b = index_0b + 1",
            },
        },
        "alignment": {
            "theta": theta,
            "coverage_runs": alignment_metadata,
            "normalization_contract": (
                "query first and unique ID; fixed WT length; uppercase AA20+'-' only; "
                "higher coverage rows nested within lower coverage rows"
            ),
        },
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            **model_metadata,
            "covariance_status": covariance_status,
        },
        "ec_table": {
            "path": str(ec_table_path),
            "sha256": sha256_file(ec_table_path),
            "complete_pair_count": len(ecs),
            "min_sequence_separation": min_sequence_separation,
            "sigma_pair_fractions": list(DEFAULT_PAIR_FRACTIONS),
            "sigma_definition": (
                "per-position sum of cn over top fraction*L long-range pairs divided by "
                "mean cn of those pairs (EVcouplings enrichment definition)"
            ),
        },
        "potts": potts_summary,
        "structures": structure_metadata,
        "contact_gate": {
            "inherited_top_L_tetramer_precision_threshold": 0.60,
            "measured_top_L_tetramer_precision": tetramer_top_l_precision,
            "status": (
                "not_computed"
                if tetramer_top_l_precision is None
                else "pass"
                if tetramer_top_l_precision >= 0.60
                else "fail_descriptive_sigma_only"
            ),
        },
        "outputs": {
            "conservation_by_coverage": "conservation_by_coverage.tsv",
            "per_position_evolution": "per_position_evolution.tsv",
            "complete_ec_table": "complete_ec_table.tsv",
            "sigma_cutoff_stability": "sigma_cutoff_stability.tsv",
            "mask_summary": "wt_lock_mask_summary.tsv",
            "mask_positions": "wt_lock_mask_positions.tsv",
            "masks_json": "wt_lock_masks.json",
            "potts_per_sequence": "potts_calibration_per_sequence.tsv",
            "potts_summary": "potts_calibration_summary.json",
            "contact_precision": "ec_contact_precision.tsv" if contact_precision is not None else None,
            "contact_pairs": "ec_contact_selected_pairs.tsv" if contact_pairs is not None else None,
            "contact_comparison": (
                "ec_contact_monomer_tetramer_comparison.tsv"
                if contact_comparison is not None
                else None
            ),
        },
    }
    (out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata
