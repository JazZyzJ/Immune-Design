"""Stage J: NetMHCIIpan mutation augmentation — config and core logic.

Implements PLAN.md Module J (J0-J5):
  - Augmentation config validation
  - WT verification threshold classification
  - Pilot protein selection
  - Mutation candidate enumeration and deduplication
  - Affected-span recalculation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ── Frozen constants (PLAN.md J0) ──────────────────────────────────────────

REQUIRED_TOP_KEYS = {
    "profile", "tool", "tool_version", "tool_mode", "allele",
    "wt_rank_threshold", "wt_uncertain_upper",
    "mut_rank_threshold", "delta_rank_threshold",
    "top_n_per_protein",
    "inputs", "outputs",
}

REQUIRED_INPUT_KEYS = {"source_parquet", "train_ids", "val_ids", "test_ids"}

REQUIRED_OUTPUT_KEYS = {
    "pilot", "registry", "registry_summary",
    "aug_train_samples", "aug_train_summary",
}

VALID_TOOLS = {"netmhciipan"}
VALID_MODES = {"EL", "BA"}


# ── Config dataclass ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class AugmentationConfig:
    """Validated, immutable Stage-J augmentation configuration."""
    profile: str
    tool: str
    tool_version: str
    tool_mode: str
    allele: str
    wt_rank_threshold: float
    wt_uncertain_upper: float
    mut_rank_threshold: float
    delta_rank_threshold: float
    top_n_per_protein: int
    inputs: dict
    outputs: dict


# ── Config loader + validator ──────────────────────────────────────────────

def load_augmentation_config(path: Path | str | None = None) -> AugmentationConfig:
    """Load and validate augmentation config. Raises ValueError on violations."""
    if path is None:
        path = Path(__file__).parents[1] / "configs" / "augmentation.yaml"
    path = Path(path)

    with open(path) as f:
        cfg = yaml.safe_load(f)

    raw = cfg.get("augmentation")
    if raw is None:
        raise ValueError("Config missing top-level 'augmentation' key")

    return validate_augmentation_config(raw)


def validate_augmentation_config(raw: dict) -> AugmentationConfig:
    """Validate raw augmentation config dict. Returns AugmentationConfig or raises."""
    # Top-level keys
    missing = REQUIRED_TOP_KEYS - set(raw.keys())
    if missing:
        raise ValueError(f"Augmentation config missing required keys: {sorted(missing)}")

    # Tool validation
    if raw["tool"] not in VALID_TOOLS:
        raise ValueError(f"Unknown tool '{raw['tool']}', expected one of {sorted(VALID_TOOLS)}")
    if raw["tool_mode"] not in VALID_MODES:
        raise ValueError(f"Unknown tool_mode '{raw['tool_mode']}', expected one of {sorted(VALID_MODES)}")

    # Threshold validation
    wt = raw["wt_rank_threshold"]
    wt_upper = raw["wt_uncertain_upper"]
    mut = raw["mut_rank_threshold"]
    delta = raw["delta_rank_threshold"]

    if not (0 < wt < wt_upper < 1):
        raise ValueError(
            f"Threshold order violated: need 0 < wt_rank_threshold ({wt}) "
            f"< wt_uncertain_upper ({wt_upper}) < 1"
        )
    if not (0 < mut < 1):
        raise ValueError(f"mut_rank_threshold must be in (0, 1), got {mut}")
    if not (0 < delta < 1):
        raise ValueError(f"delta_rank_threshold must be in (0, 1), got {delta}")

    # top_n validation
    top_n = raw["top_n_per_protein"]
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n_per_protein must be a positive integer, got {top_n}")

    # Input paths
    inputs = raw["inputs"]
    if not isinstance(inputs, dict):
        raise ValueError("augmentation.inputs must be a dict")
    missing_in = REQUIRED_INPUT_KEYS - set(inputs.keys())
    if missing_in:
        raise ValueError(f"augmentation.inputs missing keys: {sorted(missing_in)}")

    # Output paths
    outputs = raw["outputs"]
    if not isinstance(outputs, dict):
        raise ValueError("augmentation.outputs must be a dict")
    missing_out = REQUIRED_OUTPUT_KEYS - set(outputs.keys())
    if missing_out:
        raise ValueError(f"augmentation.outputs missing keys: {sorted(missing_out)}")

    return AugmentationConfig(
        profile=raw["profile"],
        tool=raw["tool"],
        tool_version=str(raw["tool_version"]),
        tool_mode=raw["tool_mode"],
        allele=raw["allele"],
        wt_rank_threshold=float(wt),
        wt_uncertain_upper=float(wt_upper),
        mut_rank_threshold=float(mut),
        delta_rank_threshold=float(delta),
        top_n_per_protein=int(top_n),
        inputs=dict(inputs),
        outputs=dict(outputs),
    )


# ── WT classification ──────────────────────────────────────────────────────

def classify_wt_rank(
    wt_el_rank: float,
    wt_threshold: float,
    uncertain_upper: float,
) -> str:
    """Classify a WT EL_Rank into threshold buckets.

    Returns one of: 'keep_wt_confirmed', 'skip_uncertain', 'reject_wt_disagree'.
    """
    if wt_el_rank < wt_threshold:
        return "keep_wt_confirmed"
    elif wt_el_rank <= uncertain_upper:
        return "skip_uncertain"
    else:
        return "reject_wt_disagree"


def is_disrupted(
    mut_el_rank: float,
    wt_el_rank: float,
    mut_threshold: float,
    delta_threshold: float,
) -> bool:
    """Check if a mutation qualifies as a disruption."""
    delta = mut_el_rank - wt_el_rank
    return mut_el_rank > mut_threshold and delta > delta_threshold


# ── Pilot protein selection (J1) ──────────────────────────────────────────

def select_pilot_proteins(
    df: pd.DataFrame,
    train_ids: set[str],
    n_proteins: int = 8,
    seed: int = 42,
) -> list[str]:
    """Select a stratified pilot subset from strict-train proteins.

    Stratifies by positive_count and sequence_length to cover diversity.
    Returns protein_id list.
    """
    train_df = df[df["protein_id"].isin(train_ids)].copy()
    if len(train_df) == 0:
        raise ValueError("No train proteins found in parquet")

    # Stratify: bin by positive_count (low/high) and seq_length (short/long)
    med_pos = train_df["positive_count"].median()
    med_len = train_df["sequence_length"].median()
    train_df["_stratum"] = (
        (train_df["positive_count"] > med_pos).astype(int) * 2
        + (train_df["sequence_length"] > med_len).astype(int)
    )

    rng = np.random.RandomState(seed)
    selected = []
    per_stratum = max(1, n_proteins // 4)
    for stratum in sorted(train_df["_stratum"].unique()):
        group = train_df[train_df["_stratum"] == stratum]
        n_take = min(per_stratum, len(group))
        picked = group.sample(n=n_take, random_state=rng)
        selected.extend(picked["protein_id"].tolist())

    return selected[:n_proteins]


# ── WT verification (J2) ──────────────────────────────────────────────────

@dataclass
class WTVerificationResult:
    """Result of WT verification for one positive span."""
    protein_id: str
    start_0b: int
    end_0b: int
    pep_len: int
    support_n: int
    wt_el_rank: float
    classification: str  # keep_wt_confirmed / skip_uncertain / reject_wt_disagree


def verify_wt_positives(
    protein_id: str,
    protein_seq: str,
    positives: list[dict],
    runner,  # NetMHCIIpanRunner
    allele: str,
    cfg: AugmentationConfig,
) -> list[WTVerificationResult]:
    """Verify WT positives against NetMHCIIpan and classify each."""
    results = []
    # Group positives by pep_len for batched scoring
    by_length: dict[int, list[dict]] = {}
    for pos in positives:
        by_length.setdefault(pos["pep_len"], []).append(pos)

    for pep_len, spans in by_length.items():
        scores = runner.score_protein(protein_id, protein_seq, allele, pep_len)
        # Build lookup: pos -> PeptideScore
        score_by_pos = {s.pos: s for s in scores}

        for span in spans:
            ps = score_by_pos.get(span["start_0b"])
            wt_rank = ps.el_rank if ps is not None else 1.0  # missing = reject
            classification = classify_wt_rank(
                wt_rank, cfg.wt_rank_threshold, cfg.wt_uncertain_upper,
            )
            results.append(WTVerificationResult(
                protein_id=protein_id,
                start_0b=span["start_0b"],
                end_0b=span["end_0b"],
                pep_len=span["pep_len"],
                support_n=span.get("support_n", 0),
                wt_el_rank=wt_rank,
                classification=classification,
            ))

    return results


# ── Mutation enumeration (J3) ─────────────────────────────────────────────

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class MutationCandidate:
    """A single-point mutation candidate before scoring."""
    source_protein_id: str
    mut_pos_0b: int
    wt_aa: str
    mut_aa: str
    seed_spans: list[dict] = field(default_factory=list)


def enumerate_mutations(
    protein_id: str,
    protein_seq: str,
    eligible_spans: list[dict],
) -> list[MutationCandidate]:
    """Enumerate deduplicated single-point mutations within eligible spans.

    Dedup key: (protein_id, mut_pos_0b, mut_aa).
    Seed spans from multiple eligible spans are merged.
    """
    # Collect unique (pos, mut_aa) -> candidate
    candidates: dict[tuple[int, str], MutationCandidate] = {}

    for span in eligible_spans:
        start, end = span["start_0b"], span["end_0b"]
        for pos in range(start, end):
            wt_aa = protein_seq[pos]
            for mut_aa in AMINO_ACIDS:
                if mut_aa == wt_aa:
                    continue
                key = (pos, mut_aa)
                if key not in candidates:
                    candidates[key] = MutationCandidate(
                        source_protein_id=protein_id,
                        mut_pos_0b=pos,
                        wt_aa=wt_aa,
                        mut_aa=mut_aa,
                    )
                candidates[key].seed_spans.append({
                    "start_0b": start,
                    "end_0b": end,
                    "pep_len": span["pep_len"],
                })

    return sorted(candidates.values(), key=lambda c: (c.mut_pos_0b, c.mut_aa))


# ── Mutation scoring + affected-span recalculation (J3+J4) ────────────────

def _span_key(pep_len: int, start_0b: int) -> str:
    """JSON-safe key for per-span rank storage: '15:85'."""
    return f"{pep_len}:{start_0b}"


@dataclass
class ScoredMutation:
    """A scored and filtered mutation with affected spans."""
    source_protein_id: str
    mut_pos_0b: int
    wt_aa: str
    mut_aa: str
    seed_spans: list[dict]
    affected_spans: list[dict]  # spans disrupted by this mutation
    wt_ranks: dict              # {"pep_len:start_0b": wt_el_rank} per affected span
    mut_ranks: dict             # {"pep_len:start_0b": mut_el_rank} per affected span


def score_and_filter_mutations(
    protein_id: str,
    protein_seq: str,
    candidates: list[MutationCandidate],
    all_positives: list[dict],
    runner,  # NetMHCIIpanRunner
    allele: str,
    cfg: AugmentationConfig,
    wt_scores_cache: dict | None = None,
) -> list[ScoredMutation]:
    """Score mutations and determine affected spans (J3+J4 combined).

    Batches all mutant proteins into a single runner.score_batch() call per
    pep_len, instead of one subprocess per candidate.

    Per-span ranks use key 'pep_len:start_0b' to avoid collisions when
    multiple affected spans share the same pep_len.
    """
    if wt_scores_cache is None:
        wt_scores_cache = {}

    # Pre-compute: for each candidate, find overlapping positives and needed lengths
    cand_overlaps: list[tuple[MutationCandidate, list[dict], str, str]] = []
    all_lengths_needed: set[int] = set()

    for cand in candidates:
        mut_seq = protein_seq[:cand.mut_pos_0b] + cand.mut_aa + protein_seq[cand.mut_pos_0b + 1:]
        overlapping = [
            p for p in all_positives
            if p["start_0b"] <= cand.mut_pos_0b < p["end_0b"]
        ]
        if not overlapping:
            continue
        mut_id = f"{protein_id}__mut_{cand.mut_pos_0b}{cand.wt_aa}>{cand.mut_aa}"
        cand_overlaps.append((cand, overlapping, mut_seq, mut_id))
        for p in overlapping:
            all_lengths_needed.add(p["pep_len"])

    if not cand_overlaps:
        return []

    # Cache WT scores (one call per pep_len for the source protein)
    for pep_len in all_lengths_needed:
        cache_key = (protein_id, pep_len)
        if cache_key not in wt_scores_cache:
            wt_scores = runner.score_protein(protein_id, protein_seq, allele, pep_len)
            wt_scores_cache[cache_key] = {s.pos: s.el_rank for s in wt_scores}

    # Batch-score all mutant proteins × all pep_lens in one call
    # Build deduplicated batch entries
    seen_ids = set()
    batch_entries = []
    for cand, overlapping, mut_seq, mut_id in cand_overlaps:
        if mut_id not in seen_ids:
            seen_ids.add(mut_id)
            batch_entries.append((mut_id, mut_seq))

    batch_out = runner.score_batch(batch_entries, allele, sorted(all_lengths_needed))

    # Reshape: mut_batch_results[pep_len][mut_id] = {pos: el_rank}
    mut_batch_results: dict[int, dict[str, dict[int, float]]] = {}
    for mut_id, by_len in batch_out.items():
        for pep_len, scores in by_len.items():
            mut_batch_results.setdefault(pep_len, {})[mut_id] = {
                s.pos: s.el_rank for s in scores
            }

    # Check disruption per candidate using batched results
    scored = []
    for cand, overlapping, mut_seq, mut_id in cand_overlaps:
        affected = []
        wt_ranks = {}
        mut_ranks = {}

        for p in overlapping:
            pep_len = p["pep_len"]
            pos = p["start_0b"]
            sk = _span_key(pep_len, pos)

            wt_r = wt_scores_cache.get((protein_id, pep_len), {}).get(pos, 1.0)
            mut_r = mut_batch_results.get(pep_len, {}).get(mut_id, {}).get(pos, 0.0)

            if is_disrupted(mut_r, wt_r, cfg.mut_rank_threshold, cfg.delta_rank_threshold):
                affected.append(p)
                wt_ranks[sk] = wt_r
                mut_ranks[sk] = mut_r

        if affected:
            scored.append(ScoredMutation(
                source_protein_id=protein_id,
                mut_pos_0b=cand.mut_pos_0b,
                wt_aa=cand.wt_aa,
                mut_aa=cand.mut_aa,
                seed_spans=cand.seed_spans,
                affected_spans=affected,
                wt_ranks=wt_ranks,
                mut_ranks=mut_ranks,
            ))

    return scored


# ── Top-N selection per protein (J5) ──────────────────────────────────────

def select_top_mutations(
    mutations: list[ScoredMutation],
    top_n: int,
) -> list[ScoredMutation]:
    """Select top-N mutations per protein by max delta_rank across affected spans."""
    def max_delta(m: ScoredMutation) -> float:
        deltas = []
        for sk in m.mut_ranks:
            deltas.append(m.mut_ranks[sk] - m.wt_ranks.get(sk, 0))
        return max(deltas) if deltas else 0.0

    ranked = sorted(mutations, key=max_delta, reverse=True)
    return ranked[:top_n]
