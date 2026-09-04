"""Final Tier-1 projection, backfill, and cross-tier collision handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SpanProjectionResult:
    valid: bool
    uniprot_spans: tuple[dict[str, Any], ...]
    chain_spans: tuple[dict[str, Any], ...]
    if_spans: tuple[dict[str, Any], ...]
    verified_span_count: int
    verified_covered_residue_count: int
    if_coverage: float
    drop_counts: dict[str, int]


def project_verified_spans(
    *,
    canonical_sequence: str,
    frozen_uniprot_range: tuple[int, int],
    raw_uniprot_spans: Sequence[Mapping[str, Any]],
    final_sequence: str,
    final_to_uniprot: Sequence[int],
    uniprot_to_chain_residue: Mapping[int, Mapping[str, Any]],
    min_spans: int = 2,
    min_coverage: float = 0.10,
    max_coverage: float = 0.50,
) -> SpanProjectionResult:
    """Project peptide-exact UniProt spans into the final IF-ready frame."""

    canonical_sequence = str(canonical_sequence).upper()
    final_sequence = str(final_sequence).upper()
    unp_start_1b, unp_end_1b = map(int, frozen_uniprot_range)
    if not 1 <= unp_start_1b <= unp_end_1b <= len(canonical_sequence):
        raise ValueError("invalid frozen UniProt range")
    if len(final_to_uniprot) != len(final_sequence):
        raise ValueError("final_to_uniprot length does not match final sequence")
    mapped = [int(value) for value in final_to_uniprot]
    if len(mapped) != len(set(mapped)) or any(a >= b for a, b in zip(mapped, mapped[1:])):
        raise ValueError("final_to_uniprot must be unique and strictly increasing")
    range_start_0b = unp_start_1b - 1
    range_end_0b = unp_end_1b
    if any(position < range_start_0b or position >= range_end_0b for position in mapped):
        raise ValueError("final residue maps outside frozen UniProt range")
    for final_idx, source_position in enumerate(mapped):
        if final_sequence[final_idx] != canonical_sequence[source_position]:
            raise ValueError("final sequence substitution relative to canonical UniProt")
    source_to_final = {source: final_idx for final_idx, source in enumerate(mapped)}
    detailed_map = {int(position): dict(value) for position, value in uniprot_to_chain_residue.items()}
    for position, residue in detailed_map.items():
        required = {"label_seq_id", "auth_seq_id", "insertion_code", "chain_local_index_0b",
                    "amino_acid"}
        if not required <= set(residue):
            raise ValueError("detailed SIFTS residue map lacks chain identity fields")
        if str(residue["amino_acid"]) != canonical_sequence[position]:
            raise ValueError("detailed SIFTS residue amino acid differs from canonical UniProt")

    drop_counts = {
        "outside_frozen_range": 0,
        "source_peptide_mismatch": 0,
        "mapping_gap": 0,
        "if_peptide_mismatch": 0,
        "duplicate_span": 0,
    }
    uniprot_spans: list[dict[str, Any]] = []
    chain_spans: list[dict[str, Any]] = []
    if_spans: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for raw in raw_uniprot_spans:
        start = int(raw["start_0b"])
        end = int(raw["end_0b"])
        peptide = str(raw.get("peptide") or raw.get("peptide_seq") or "")
        if start < range_start_0b or end > range_end_0b or end <= start:
            drop_counts["outside_frozen_range"] += 1
            continue
        if not peptide or canonical_sequence[start:end] != peptide:
            drop_counts["source_peptide_mismatch"] += 1
            continue
        key = (start, end, peptide)
        if key in seen:
            drop_counts["duplicate_span"] += 1
            continue
        source_positions = list(range(start, end))
        if any(position not in detailed_map for position in source_positions):
            drop_counts["mapping_gap"] += 1
            continue
        chain_residues = [detailed_map[position] for position in source_positions]
        if any(
            not bool(residue.get("observed", True)) or residue.get("auth_seq_id") is None
            for residue in chain_residues
        ):
            drop_counts["mapping_gap"] += 1
            continue
        chain_indices = [int(residue["chain_local_index_0b"]) for residue in chain_residues]
        if chain_indices != list(range(chain_indices[0], chain_indices[0] + len(chain_indices))):
            drop_counts["mapping_gap"] += 1
            continue
        if any(position not in source_to_final for position in source_positions):
            drop_counts["mapping_gap"] += 1
            continue
        final_indices = [source_to_final[position] for position in source_positions]
        if final_indices != list(range(final_indices[0], final_indices[0] + len(final_indices))):
            drop_counts["mapping_gap"] += 1
            continue
        if_start = final_indices[0]
        if_end = final_indices[-1] + 1
        if final_sequence[if_start:if_end] != peptide:
            drop_counts["if_peptide_mismatch"] += 1
            continue
        seen.add(key)
        uniprot_spans.append({"start_0b": start, "end_0b": end, "peptide": peptide})
        chain_spans.append(
            {
                "start_0b": chain_indices[0],
                "end_0b": chain_indices[-1] + 1,
                "peptide": peptide,
                "uniprot_positions_0b": source_positions,
                "label_seq_ids": [int(residue["label_seq_id"]) for residue in chain_residues],
                "auth_residues": [
                    {"auth_seq_id": int(residue["auth_seq_id"]),
                     "insertion_code": str(residue["insertion_code"])}
                    for residue in chain_residues
                ],
            }
        )
        if_spans.append({"start_0b": if_start, "end_0b": if_end, "peptide": peptide})

    covered = {
        position
        for span in if_spans
        for position in range(int(span["start_0b"]), int(span["end_0b"]))
    }
    coverage = len(covered) / len(final_sequence) if final_sequence else 0.0
    valid = len(if_spans) >= min_spans and min_coverage <= coverage <= max_coverage
    return SpanProjectionResult(
        valid=valid,
        uniprot_spans=tuple(uniprot_spans),
        chain_spans=tuple(chain_spans),
        if_spans=tuple(if_spans),
        verified_span_count=len(if_spans),
        verified_covered_residue_count=len(covered),
        if_coverage=coverage,
        drop_counts=drop_counts,
    )


def finalize_tier1_pool(
    candidates: Iterable[Mapping[str, Any]], *, target: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter the complete materialized pool, then C8-rank/backfill clean units."""

    rows = [dict(row) for row in candidates]
    source_ids = [str(row.get("source_uniprot_id") or "") for row in rows]
    if not all(source_ids) or len(source_ids) != len(set(source_ids)):
        raise ValueError("Tier1 candidates require unique source_uniprot_id")
    if target < 1:
        raise ValueError("Tier1 target must be positive")
    otherwise_valid = [row for row in rows if bool(row.get("projection_valid"))]
    if any(not isinstance(row.get("cath_overlap_flag"), bool) for row in otherwise_valid):
        raise ValueError("Tier1 CATH flags must be measured booleans")
    clean = [row for row in otherwise_valid if not row["cath_overlap_flag"]]
    diagnostic_pool = [row for row in otherwise_valid if row["cath_overlap_flag"]]
    clean.sort(
        key=lambda row: (
            -float(row["final_epitope_coverage"]),
            -int(row["verified_span_count"]),
            -int(row["verified_covered_residue_count"]),
            str(row["source_uniprot_id"]),
        )
    )
    diagnostic_pool.sort(
        key=lambda row: (
            -float(row["final_epitope_coverage"]), -int(row["verified_span_count"]),
            -int(row["verified_covered_residue_count"]), str(row["source_uniprot_id"]),
        )
    )

    def collapse(
        ordered: list[dict[str, Any]], *, role: str,
        occupied_entities: dict[str, dict[str, Any]], occupied_sequences: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        kept: list[dict[str, Any]] = []
        ledger: list[dict[str, Any]] = []
        for row in ordered:
            entity = str(row.get("rcsb_entity_id") or "")
            sequence = str(row.get("sequence_sha256") or "")
            if not entity or not sequence:
                raise ValueError("Tier1 candidate lacks entity/final-sequence collision identity")
            owner = occupied_entities.get(entity) or occupied_sequences.get(sequence)
            if owner is not None:
                ledger.append({
                    "dropped_source_uniprot_id": str(row["source_uniprot_id"]),
                    "kept_source_uniprot_id": str(owner["source_uniprot_id"]),
                    "role": role,
                    "collision_reason": (
                        "entity" if entity in occupied_entities else "final_sequence"
                    ),
                    "action": "drop_tier1_duplicate",
                })
                continue
            kept.append(row)
            occupied_entities[entity] = row
            occupied_sequences[sequence] = row
        return kept, ledger

    clean_unique, ledger = collapse(
        clean, role="primary_pool", occupied_entities={}, occupied_sequences={}
    )
    if len(clean_unique) < target:
        raise ValueError(f"Tier1 clean capacity {len(clean_unique)} < target {target}")
    primary = clean_unique[:target]
    occupied_entities = {str(row["rcsb_entity_id"]): row for row in primary}
    occupied_sequences = {str(row["sequence_sha256"]): row for row in primary}
    diagnostic, diagnostic_ledger = collapse(
        diagnostic_pool, role="tier1_overlap_diagnostic",
        occupied_entities=occupied_entities, occupied_sequences=occupied_sequences,
    )
    ledger.extend(diagnostic_ledger)
    for row in primary:
        row["evaluation_role"] = "primary_generalization"
        row["source_pool_memberships"] = ["tier1"]
        row["selected_tier_memberships"] = ["tier1"]
    for row in diagnostic:
        row["evaluation_role"] = "tier1_overlap_diagnostic"
        row["source_pool_memberships"] = ["tier1"]
        row["selected_tier_memberships"] = ["tier1"]
    return primary, diagnostic, ledger


def resolve_tier1_tier2_collisions(
    *,
    t1_primary: Iterable[Mapping[str, Any]],
    t1_diagnostic: Iterable[Mapping[str, Any]],
    t2_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply Tier-1 precedence by source entity or final exact sequence."""

    primary = [dict(row) for row in t1_primary]
    diagnostic = [dict(row) for row in t1_diagnostic]
    all_t1 = [("primary", row) for row in primary] + [
        ("diagnostic", row) for row in diagnostic
    ]
    entity_owner: dict[str, tuple[str, dict[str, Any]]] = {}
    sequence_owner: dict[str, tuple[str, dict[str, Any]]] = {}
    for role, row in all_t1:
        for value, owners, label in (
            (str(row.get("rcsb_entity_id") or ""), entity_owner, "entity"),
            (str(row.get("sequence_sha256") or ""), sequence_owner, "sequence"),
        ):
            if not value:
                continue
            if value in owners and owners[value][1] is not row:
                raise ValueError(f"multiple Tier1 units share one {label} collision key")
            owners[value] = (role, row)

    remaining: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for source in t2_rows:
        t2 = dict(source)
        owners = []
        group_entity_ids = t2.get("group_entity_ids")
        if hasattr(group_entity_ids, "tolist"):
            group_entity_ids = group_entity_ids.tolist()
        if not isinstance(group_entity_ids, (list, tuple)) or not group_entity_ids:
            raise ValueError("Tier2 row requires complete group_entity_ids provenance")
        group_entity_ids = [str(value) for value in group_entity_ids]
        if len(group_entity_ids) != len(set(group_entity_ids)):
            raise ValueError("Tier2 group_entity_ids contains duplicates")
        sequence = str(t2.get("sequence_sha256") or "")
        for entity in group_entity_ids:
            if entity in entity_owner:
                owners.append(("group_entity", entity_owner[entity]))
        if sequence and sequence in sequence_owner:
            owners.append(("sequence", sequence_owner[sequence]))
        unique_rows = {id(owner[1][1]): owner for owner in owners}
        if len(unique_rows) > 1:
            raise ValueError("Tier2 row collides with multiple Tier1 source units")
        if not owners:
            remaining.append(t2)
            continue
        reason, (role, owner) = owners[0]
        owner["source_pool_memberships"] = sorted(
            set(owner.get("source_pool_memberships") or ["tier1"]) | {"tier1", "tier2"}
        )
        owner["selected_tier_memberships"] = ["tier1"]
        ledger.append(
            {
                "tier2_protein_id": t2.get("protein_id"),
                "tier1_protein_id": owner.get("protein_id"),
                "tier1_role": role,
                "collision_reason": reason,
                "action": "remove_tier2_keep_tier1",
            }
        )
    return primary, diagnostic, remaining, ledger
