"""Versioned row schema for ``if-benchmark-test-set/3`` releases."""

from __future__ import annotations


ROW_SCHEMA_VERSION = "if-benchmark-v3-row/1"
V3_ROW_COLUMNS = frozenset({
    "dataset_release_id", "allele", "protein_id", "selection_unit_id", "pdb_id",
    "entity_id", "rcsb_entity_id", "auth_chain_id", "label_asym_id",
    "source_pool_memberships", "selected_tier_memberships", "evaluation_role",
    "sequence", "sequence_length", "sequence_sha256", "entity_sequence",
    "entity_sequence_sha256", "source_sequence", "source_sequence_sha256",
    "source_range_start_0b", "source_range_end_0b", "source_denominator_length",
    "final_to_source_json", "mapping_status", "if_sequence_coverage",
    "selected_chain_provenance_json", "rejected_chain_provenance_json", "pdb_path",
    "structure_sha256", "experimental_method", "resolution", "nmp_status",
    "coverage_fraction", "nmp_query_sequence_sha256", "nmp_evidence_sha256",
    "nmp_tool_sha256", "nmp_parameters_sha256", "selection_bin", "selection_reason",
    "cath_overlap_flag", "cath_best_target", "cath_best_identity", "family_cluster_id",
    "source_uniprot_id", "sifts_range_json", "uniprot_spans_json", "chain_spans_json",
    "sifts_residue_map_json",
    "if_spans_json", "verified_span_count", "verified_covered_residue_count",
    "final_epitope_coverage",
})
