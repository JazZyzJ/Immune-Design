# PROTOCOL: New redesign target → RF-ready data

## Scope / when to use

A fresh wet-lab redesign target arrives as **WT sequence + a collaborator active-site residue
list** (optionally: an experimental PDB, a ligand). This SOP is the **data-preparation pipeline
only** — it stops the moment the target is **RF-ready**, i.e. produces the three artifacts that a
constrained RF / refinement / inpainting run consumes:

1. an **IF-ready test-set parquet** (`sequence` = resolved-structure residues, `pdb_path`, …),
2. a **`PDB_ROOT`** of cleaned single-chain structures, and
3. an **active-site constraint manifest** (hard anchors).

It does **not** cover the redesign itself (B1Aopen generation, immune/structure evaluation, seed
selection) — those are the RF run + the selection protocols. First instantiated on the backup
collection `if_test_set/backup/` (EGFP, mCherry, PrASNase, NanoLuc); PrASNase is the clean
exemplar, the FPs the "structure must be AF3, not the crystal" exemplar.

The *logic* below is fixed; the target's sequence / active sites are the only inputs.

## Core principles (why this pipeline)

1. **Structure from AF3, not the crystal.** Crystals fuse PTM groups into non-standard HETATM
   residues (fluorescent-protein chromophore = `CRO`/`CH6`, spanning 3 residues); `build_if_ready`
   keeps only the standard 20 AAs, so those residues get **dropped** — the exact positions you
   must protect become unencodable and the backbone gets a hole. AF3 full-MSA monomer predicts
   **standard residues everywhere** (incl. the pre-cyclization chromophore Thr/Met-Tyr-Gly). Keep
   any crystal only as a validation reference (FP AF3-vs-crystal Cα RMSD 0.29 / 0.44 Å).
2. **Keep the IF-ready sequence == WT (offset 0).** AF3 predicts the full input sequence, so the
   resolved IF-ready sequence equals the WT → anchor `index_0b` = WT 0-based **directly**, and the
   numbering-projection step that a gappy crystal (missing termini/loops) would force disappears.
   Always verify `if_sequence_offset == 0`.
3. **Verify collaborator numbering first — the cheapest bug-catch.** Their numbering ≠ your
   sequence index. Map each `(letter, position)` to the WT 0-based index and confirm the letter.
   Watch canonical-vs-construct offsets (mCherry active sites are canonical chromophore numbering,
   **+5** vs the supplied WT seq) and Met inclusion. `validate_against_sequence` (§3) is the
   fail-closed backstop, but catch it here.

## Inputs (collect up front)

| Input | Required | Note |
|---|---|---|
| WT sequence | yes | full-length, standard AAs |
| Active-site residues to protect | yes | with numbering — **verify** (§0) |
| Experimental PDB | no | validation reference only; never the IF-ready structure if it has PTM HETATMs |
| Ligand | no | annotation only (`ligand_annotation` in the manifest) |

## Procedure

### 0. Verify active-site numbering
`seq[pos-1] == letter` for every site? If not, recover the constant offset (or align) and record
the verified **0-based** indices + expected AA. (mCherry example: labels are canonical, +5 vs the
supplied seq.)

### 1. Structure — AF3 (full-MSA monomer)
- Build `af3_input.parquet` (`protein_id, sequence`).
- `submit_af3_data.slurm` (CPU **local** MSA — no egress) → `submit_af3_refold.slurm USE_MSA=true`
  (ailab GPU inference) → normalize → `<cache>/<cache_key>.pdb` (+ `.plddt`).
- pLDDT sanity: monomers here fold to ~90–95. If a crystal exists, superpose (Cα RMSD) as a
  confidence check only — do not substitute it for the AF3 structure.

### 2. IF-ready
`build_if_ready_test_set.py --min-coverage 0.0` on the AF3 PDB (chain `A`; input parquet needs
`protein_id, sequence, sequence_length, pdb_path`, plus a `chain` column or `--default-chain A`).
Verify each row: `if_sequence_offset == 0` and resolved `sequence == WT`. Outputs:
`<target>_caseset_if_ready.parquet` (**artifact 1**) + cleaned single-chain PDBs in
`--output-structure-dir` (**artifact 2 = `PDB_ROOT`**).

### 3. Active-site manifest (**artifact 3**)
Author `inverse_folding/reference_flow/configs/<target>_active_site_v0.yaml`
(schema `uricase_active_site_v0`): per protein
`hard_anchors: [{index_0b, expected_aa, label, biological_role}]`, `index_0b` 0-based in the
IF-ready sequence (= WT 0-based given §2). Include catalytic + binding + structurally-essential
residues. Add `ligand_annotation` if any (informational). **Validate before shipping**:
`load_constraint_manifest` + `validate_against_sequence` on every entry (fail-fast identity — a
mislabeled index aborts here, not mid-run).

## RF-ready outputs → how RF consumes them

| Artifact | Path convention | Consumed by |
|---|---|---|
| IF-ready parquet | `if_test_set/<target>/<target>_caseset_if_ready.parquet` | `--test-set-parquet` |
| Cleaned structures (`PDB_ROOT`) | `if_test_set/<target>/pdbs_if_ready/<protein_id>.pdb` | `--pdb-root` |
| Active-site manifest | `inverse_folding/reference_flow/configs/<target>_active_site_v0.yaml` | `--constraint-manifest` |

These same three feed `run_if_phase_c1.py` (generation/inpainting), `refine_rf_designs.py`
(refinement), and `run_rf_refine_fusion.py` (fusion) — the constrained-redesign entry points. The
target is **RF-ready** once all three exist and the manifest round-trips through
`validate_against_sequence`. (h-maps are **not** required — B1Aopen / amplification=`constant_one`
does not consume them.)

## Gotchas (each cost real time on the backup collection)

- **Numbering** (principle 3): supplied active-site numbering may be canonical, not construct
  (mCherry +5). Verify per target.
- **Crystal chromophore/PTM** (principle 1): if the target has a matured chromophore or other PTM,
  the crystal is unusable as the IF-ready structure — use AF3.
- **IF-ready coverage filter**: `build_if_ready` defaults `--min-coverage 0.8`; pass `0.0` for a
  full-length AF3 monomer so nothing is silently dropped, and assert `if_sequence_offset == 0`.
- **Gly active-site anchor** (e.g. FP chromophore Gly): still put it in the manifest (identity is
  frozen at generation), but note that Gly has no side chain, so downstream *structure eval* will
  report it as `no_sidechain` — that is an eval-metric detail, not a data-prep problem.

## Concrete commands (backup collection instantiation)

```bash
# 1. AF3 structure (local MSA -> GPU inference -> normalized cache)
FROM_PARQUET=<target>/af3_input.parquet AF3_OUT=<target>/af3/af3_out N_SHARDS=4 \
  sbatch --array=0-3 scripts/submit_af3_data.slurm
FROM_PARQUET=<target>/af3_input.parquet AF3_OUT=<target>/af3/af3_out CACHE_DIR=<target>/af3/cache \
  N_SHARDS=4 USE_MSA=true sbatch --array=0-3 --dependency=afterok:<dataJob> scripts/submit_af3_refold.slurm

# 2. IF-ready (input parquet: protein_id, sequence(WT), sequence_length, pdb_path=<cache_key>.pdb, chain=A)
python scripts/build_if_ready_test_set.py \
  --input-parquet <target>/if_ready_input.parquet --pdb-root <target>/af3/cache \
  --output-parquet <target>/<target>_caseset_if_ready.parquet \
  --output-structure-dir <target>/pdbs_if_ready \
  --manifest-json <target>/if_ready.manifest.json --failure-csv <target>/if_ready.failures.csv \
  --default-chain A --min-coverage 0.0 --min-if-length 50
# -> verify: if_sequence_offset == 0 and resolved sequence == WT for every row

# 3. active-site manifest: author <target>_active_site_v0.yaml, then round-trip validate:
python -c "
from inverse_folding.reference_flow.constraints import load_constraint_manifest
import pandas as pd
mf = load_constraint_manifest('inverse_folding/reference_flow/configs/<target>_active_site_v0.yaml')
seqs = dict(zip(*[pd.read_parquet('<target>/<target>_caseset_if_ready.parquet')[c] for c in ('protein_id','sequence')]))
for p in mf.entries: mf.entries[p].validate_against_sequence(seqs[p])
print('RF-ready:', len(mf.entries), 'entries,', mf.num_hard_anchors_total, 'anchors validated')"
```
