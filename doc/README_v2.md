# MHC Class II Immunopeptidome Dataset V2 (Unified & Verified)

This directory contains the unified and verified MHC class II ligand dataset (V2), which consolidates records from the Immune Epitope Database (IEDB) and the HLA Ligand Atlas. Every record in this dataset has been cross-verified against UniProt protein sequences.

---

## File Summary

| File | Format | Records | Description |
| :---- | :---- | :---- | :---- |
| `mhc_if_v2.tsv` | TSV | 1,246,872 | **Primary Unified Dataset**: Includes start/end positions and 3-mer flanking context. |

### Source Breakdown

- **IEDB (Low Resolution)**: 740,383 records  
- **IEDB (Standard v1)**: 364,025 records  
- **HLA Ligand Atlas**: 142,464 records

---

## 1\. Record Schema

The V2 dataset includes detailed positional and contextual information for each peptide-protein match.

### Positional Info (peptide\_position\_info)

Stored as a JSON list of objects:

\[

  {

    "protein\_id": "P49006.2",

    "start": 156.0,

    "end": 172.0

  }

\]

### Flanking Context (flanking\_context)

Stored as a JSON list of objects containing 3-mer residues:

\[

  {

    "protein\_id": "P49006",

    "upstream": "EPQ",

    "downstream": "ATE"

  }

\]

---

## 2\. Column Specification (mhc\_if\_v2.tsv)

| Column | Type | Description |
| :---- | :---- | :---- |
| `source` | `str` | Original data source (`iedb`, `atlas`) |
| `dataset_source` | `str` | Internal processing source (`iedb_v1`, `iedb_low_reso`, `atlas`) |
| `peptide_seq` | `str` | Peptide amino acid sequence |
| `hla_class` | `str` | MHC class, fixed as `II` |
| `alleles` | `JSON array` | HLA allele identifiers |
| `tissues` | `JSON array` | Source tissue types |
| `protein_accessions` | `JSON array` | Original protein accession numbers |
| `peptide_position_info` | `JSON array` | Verified start and end coordinates (1-indexed) |
| `flanking_context` | `JSON array` | 3-mer flanking sequences (Upstream/Downstream) |
| `resolution` | `str` | Allele resolution classification: |
|  |  | \- **`generic`**: Non-specific terms (e.g., `HLA-DR`, `HLA-DQ`) |
|  |  | \- **`serotypes`**: Serotype-level names without `*` (e.g., `HLA-DR1`) |
|  |  | \- **`high_res_single`**: Specific single-gene alleles (e.g., `DRB1*01:01`) |
|  |  | \- **`high_res_dimer`**: Specific heterodimers (e.g., `DQA1*01:01-DQB1*05:01`) |
|  |  | \- **`multi`**: Mixed or complex allele identifications (e.g. Atlas data) |

---

## 3\. Processing and Verification

The V2 dataset was generated through a rigorous multi-step pipeline:

1. **Coordinate Extraction**: Positions were extracted from IEDB raw exports.  
2. **Coordinate Discovery**: For Atlas data, positions were found by searching peptide substrings within UniProt protein sequences.  
3. **UniProt Sync**: All protein IDs were mapped to standard UniProt accessions, and over 16,000 full sequences were downloaded.  
4. **Shift Correction**: Minor coordinate shifts (\~7.8k cases) were automatically corrected to match the UniProt sequences.  
5. **Flank Extraction**: Upstream and downstream 3-mer residues were extracted directly from the verified protein sequences.  
6. **Consistency Check**: A final 100% verification was performed for every record (Peptide Sequence \== Protein Sequence\[Start:End\]).

---

## Processing Scripts (V2 Layer)

| Script | Function |
| :---- | :---- |
| `add_positions.py` | Extracts initial coordinates from raw data |
| `fetch_sequences.py` | Batch downloads UniProt FASTA sequences |
| `atlas_find_positions.py` | Substring search for Atlas peptide positions |
| `add_flanks.py` | Extracts residues around the peptide |
| `validate_consistency.py` | Performs global sequence-coordinate verification |
| `finalize_v2.py` | Corrects shifts, filters mismatches, and merges sources |

---

## Notes

1. **Strict Verification**: Rows where the peptide could not be found in the UniProt sequence were removed to ensure 100% data quality.  
2. **Multiple Matches**: If a peptide sequence appears multiple times within a protein, all valid positions and their respective flanking contexts are retained.  
3. **Coordinate Type**: `start` and `end` positions follow 1-based indexing (IEDB standard).

