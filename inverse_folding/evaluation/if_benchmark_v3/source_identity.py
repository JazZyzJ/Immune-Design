"""Resolve Head-split accessions to canonical UniProt Tier-1 source units."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence


AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
_VERSION_SUFFIX = re.compile(r"\.\d+$")
_UNIPROT_ACCESSION = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}){1,2}[0-9])$"
)
_REFSEQ_PROTEIN = re.compile(r"^(?:NP|XP|WP|YP)_\d+\.\d+$")
_GENBANK_PROTEIN = re.compile(r"^[A-Z]{3}\d{5}(?:\.\d+)?$")


def _sha256_sequence(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def classify_source_accession(source_id: str) -> tuple[str, str | None]:
    """Classify a raw split identifier and canonicalize only valid UniProt aliases."""

    source_id = str(source_id).strip()
    base = _VERSION_SUFFIX.sub("", source_id)
    if _UNIPROT_ACCESSION.fullmatch(base):
        return "uniprot", base
    if _REFSEQ_PROTEIN.fullmatch(source_id):
        return "refseq", None
    if _GENBANK_PROTEIN.fullmatch(source_id):
        return "genbank_cds", None
    return "unmapped", None


def resolve_canonical_source_units(
    *,
    raw_sequences: Mapping[str, str],
    raw_spans: Mapping[str, Sequence[Mapping[str, Any]]],
    external_mappings: Mapping[str, Sequence[str]],
    canonical_sequences: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Resolve aliases and return valid canonical source units plus a full ledger.

    External accessions are accepted only when exactly one mapped UniProt canonical sequence is
    byte-identical to the manifest sequence. All aliases assigned to one canonical UniProt must
    themselves equal that canonical sequence; a single conflicting version fails the whole source
    unit instead of silently pooling spans across coordinate frames.
    """

    ledger: list[dict[str, Any]] = []
    provisional_by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_id in sorted(raw_sequences):
        raw_sequence = str(raw_sequences[raw_id]).replace(" ", "").upper()
        accession_type, direct_canonical = classify_source_accession(raw_id)
        mapping_candidates = sorted(set(str(x) for x in external_mappings.get(raw_id, [])))
        row: dict[str, Any] = {
            "raw_source_id": raw_id,
            "accession_type": accession_type,
            "raw_sequence_length": len(raw_sequence),
            "raw_sequence_sha256": _sha256_sequence(raw_sequence),
            "mapping_candidates_json": json.dumps(mapping_candidates, separators=(",", ":")),
            "canonical_uniprot_id": None,
            "canonical_sequence_length": None,
            "canonical_sequence_sha256": None,
            "mapping_resolution": None,
            "alias_resolution_status": None,
            "unit_resolution_status": None,
            "unit_invalidation_reason": None,
            "unit_aliases_json": None,
            "resolution_status": None,
        }

        chosen: str | None = None
        if accession_type == "uniprot":
            chosen = direct_canonical
            row["mapping_resolution"] = (
                "uniprot_version_canonicalized"
                if raw_id != direct_canonical
                else "direct_uniprot"
            )
        elif accession_type in {"refseq", "genbank_cds"}:
            if not mapping_candidates:
                row["resolution_status"] = "unmapped_external_accession"
            else:
                exact = [
                    candidate
                    for candidate in mapping_candidates
                    if canonical_sequences.get(candidate) == raw_sequence
                ]
                if len(exact) == 1:
                    chosen = exact[0]
                    row["mapping_resolution"] = "external_exact_sequence"
                elif len(exact) > 1:
                    row["resolution_status"] = "ambiguous_external_mapping"
                elif len(mapping_candidates) == 1:
                    chosen = mapping_candidates[0]
                    row["mapping_resolution"] = "external_unique_mapping"
                else:
                    row["resolution_status"] = "external_mapping_sequence_mismatch"
        else:
            row["resolution_status"] = "unmapped_accession_namespace"

        if chosen is not None:
            row["canonical_uniprot_id"] = chosen
            canonical_sequence = canonical_sequences.get(chosen)
            if canonical_sequence is None:
                row["resolution_status"] = "canonical_sequence_missing"
            else:
                canonical_sequence = str(canonical_sequence).replace(" ", "").upper()
                row["canonical_sequence_length"] = len(canonical_sequence)
                row["canonical_sequence_sha256"] = _sha256_sequence(canonical_sequence)
                if not canonical_sequence or not set(canonical_sequence) <= AA20:
                    row["resolution_status"] = "canonical_sequence_non_aa20"
                elif raw_sequence != canonical_sequence:
                    row["resolution_status"] = "alias_sequence_conflict"
                else:
                    row["resolution_status"] = "resolved"
            provisional_by_canonical[chosen].append(row)
        ledger.append(row)

    for row in ledger:
        row["alias_resolution_status"] = row["resolution_status"]

    # One conflicting alias invalidates the entire canonical coordinate frame while preserving
    # each alias's own resolution outcome in ``alias_resolution_status``.
    for canonical_id, rows in provisional_by_canonical.items():
        aliases = sorted(row["raw_source_id"] for row in rows)
        unit_status = (
            "alias_sequence_conflict"
            if any(row["alias_resolution_status"] == "alias_sequence_conflict" for row in rows)
            else (
                "resolved"
                if all(row["alias_resolution_status"] == "resolved" for row in rows)
                else "identity_resolution_failed"
            )
        )
        for row in rows:
            row["unit_aliases_json"] = json.dumps(aliases, separators=(",", ":"))
            row["unit_resolution_status"] = unit_status
            row["resolution_status"] = unit_status
            if unit_status != "resolved":
                row["unit_invalidation_reason"] = unit_status
    for row in ledger:
        if row["unit_resolution_status"] is None:
            row["unit_resolution_status"] = row["alias_resolution_status"]
            row["resolution_status"] = row["alias_resolution_status"]
            if row["resolution_status"] != "resolved":
                row["unit_invalidation_reason"] = row["resolution_status"]
            row["unit_aliases_json"] = json.dumps(
                [row["raw_source_id"]], separators=(",", ":")
            )

    units: dict[str, dict[str, Any]] = {}
    for canonical_id, rows in sorted(provisional_by_canonical.items()):
        if any(row["unit_resolution_status"] != "resolved" for row in rows):
            continue
        canonical_sequence = str(canonical_sequences[canonical_id]).replace(" ", "").upper()
        aliases = sorted(row["raw_source_id"] for row in rows)
        merged_spans: list[dict[str, Any]] = []
        seen_spans: set[tuple[int, int, str]] = set()
        for raw_id in aliases:
            for span in raw_spans.get(raw_id, []):
                start = int(span["start_0b"])
                end = int(span["end_0b"])
                peptide = str(span.get("peptide") or span.get("peptide_seq") or "")
                key = (start, end, peptide)
                if key in seen_spans:
                    continue
                seen_spans.add(key)
                merged_spans.append(
                    {
                        "start_0b": start,
                        "end_0b": end,
                        "peptide": peptide,
                        "source": str(span.get("source") or "iedb"),
                        "raw_source_id": raw_id,
                    }
                )
        merged_spans.sort(
            key=lambda span: (
                span["start_0b"], span["end_0b"], span["peptide"], span["raw_source_id"]
            )
        )
        units[canonical_id] = {
            "canonical_uniprot_id": canonical_id,
            "canonical_sequence": canonical_sequence,
            "canonical_sequence_length": len(canonical_sequence),
            "canonical_sequence_sha256": _sha256_sequence(canonical_sequence),
            "raw_source_ids": aliases,
            "raw_source_id_count": len(aliases),
            "spans": merged_spans,
            "span_count": len(merged_spans),
        }
    return units, ledger
