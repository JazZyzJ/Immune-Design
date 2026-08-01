# Evolution-guided residue masks for uricase (Q00511)

Ports the masking strategy of Skopintsev et al., *Science* 2026 (`science.aed6123.pdf`)
to *A. flavus* uricase, to decide which residues to hold fixed during inverse-folding
redesign.

The paper builds its fixed-residue mask from evolutionary statistics rather than from a
distance radius around a ligand:

    mask(C0, sigma0) = { i : C_i >= C0 }  UNION  { i : sigma_i >= sigma0 }

- **C_i** — positional conservation from an MSA of natural homologs, 0–1.
- **sigma_i** — coupling strength from a Potts model.

Raising either threshold fixes fewer residues and yields sequences further from WT. The
paper's key result is that *dual* (C0, sigma0) masking beats conservation-only masking:
sigma pins the functionally critical residues that conservation ranks mid-pack, which
lets C0 rise and free up the rest.

## The adaptation uricase forces

In the paper sigma is a **protein↔nucleobase** coupling, fit to *paired* TnpB–RNA/DNA
alignments. Uricase binds a small molecule, not a sequence — there is no partner
alignment to pair with. So here:

**sigma_i = intra-protein cumulative coupling strength**, EVcouplings' `pairs.enrichment`
(Hopf et al., *Cell* 2012), which was introduced precisely to surface functionally
constrained residues — catalytic sites, ligand pockets, oligomer interfaces — that
ordinary conservation misses.

This fits uricase unusually well. It is a **homotetramer whose active sites sit at the
protomer interface**: of the 15 residues within 4.5 Å of the bound ligand, **7 are donated
by the adjacent subunit**, including the catalytic Lys10–Thr57 dyad. A single-family Potts
model cannot separate intra- from inter-chain covariation — normally a caveat, here
exactly the point.

## Two things that will silently corrupt this analysis

**1. Numbering.** The uricase literature and every PDB entry number the **mature** protein
(initiator Met cleaved); UniProt Q00511 includes Met1.

    mature position == UniProt position - 1

Verified twice: 16/17 literature active-site residues match mature numbering and 0/17
match UniProt; PDB 1R4U chain A matches 294/294 vs 9/294. Masking in UniProt numbering
produces a plausible-looking mask that pins the wrong side chains. All tables carry both
columns; masks are emitted in mature numbering plus a zero-indexed variant.

**2. Gaps count as an alphabet symbol** in EVcouplings' conservation, which distorts C_i in
both directions. On the nr90 baseline the 26 columns above 50% gap average C_i 0.624
native vs 0.367 gap-excluded (mature 198 reads 0.706 native but is really 0.204 — it just
happens to be gappy); moderately gapped columns get the opposite treatment, with the gap
diluting real conservation (His256: 0.826 native vs 0.939 excluded). Masks key on
`C_i_nogap`.

Also: plmc's `-t` is **1 − theta**, and EVcouplings scales `lambda_J` by `(q−1)(L−1)`
(0.01 → 60.2 at L=302). We call `evcouplings.couplings.tools.run_plmc` rather than plmc
directly so neither convention can be got wrong by hand.

## Why not reuse `03_alignment/`

That MSA was built for phylogenetics/ASR, a different objective:

| | |
|---|---|
| `raw/nr90_aln.fasta` | 3034 × 3895 cols. Usable — restricted to Q00511's columns the median gap fraction is 2%. But **N_eff/L = 3.59**, marginal for Potts inference (want ≥5, ideally ≥10). |
| `trimmed/nr90_aln_trim.fasta` | 235 cols. **Unusable** — trimAl dropped 67 of the target's 302 positions. |

Hence a fresh deep search against the group's local UniRef30 + ColabFold envdb.

## Pipeline

| Step | Script | Where |
|---|---|---|
| 0 | `env/environment.yml`, `env/plmc/` | login node |
| 1 | `11_msa.slurm` — MMseqs2 vs UniRef30 + envdb, `--filter 0` | Slurm `cpu`, 32c/300G |
| 1b | `12_prepare_alignment.py` — focus alignment + N_eff sweep | login node |
| 2 | `13_couplings.slurm` → `13_run_plmc.py` — fit the Potts model | Slurm `cpu`, 32c/48G |
| 3 | `14_scores.py` — C_i and sigma_i per residue | login node |
| 4 | `15_validate.py` — **go/no-go gate** | login node |
| 5 | `16_masks.py` — (C0, sigma0) grid → masks | login node |

The env lives on `/scratch` (not a named env in `/home`, which is at 39/50 GB):

    conda activate /scratch/gpfs/KAIYIJIANG/kaiyi/Uricase/coupling/env/uricase-ec

## The validation gate

Masks are not emitted from an unvalidated model. `15_validate.py` scores the top-L
long-range ECs against contact maps built from the **monomer** and from the **tetramer**
(1R4U biological assembly — the asymmetric unit holds one chain, so using it instead
makes every cross-subunit contact vanish). 24.1% of long-range tetramer contacts are
inter-chain only, so the two maps genuinely differ.

**Gate: top-L precision against the tetramer ≥ 0.60.** Below that the alignment is too
shallow or contaminated and sigma-derived masks would be noise.

## Q00511 round-1 rescue-v2 contract

The activity-rescue plate does not use the generic `C OR sigma` grid. Its seven constraint
manifests use the following fixed contract:

- sigma: `sigma_pct >= {0.90,0.80} AND gap_frac <= 0.5`, with **no conservation
  exclusion**; for `deep_iter500` this retains the complete raw 31/61-position sets;
- universal add-on: the eight `C_i_nogap >= 0.8` positions outside the original full
  structural mask (`index_0b = 40,50,77,145,157,186,224,301`);
- sampler: the unchanged `inverse_folding/reference_flow/configs/c1_null.yaml`; only the
  constraint manifest changes across cells.

Exact membership, gap sensitivity, matrix counts, generation code, and runtime preflight
are under `06_round1_sigma_v2/`. The earlier `06_round1_clean_sigma/` directory records a
superseded C-filtered/no-remask proposal and is not the round-1 execution contract.

## Layout

```
00_target/   Q00511.fasta, 1r4u_assembly1.cif, reference.yml  <- numbering + ground truth
01_msa/      colabfold a3m, focus alignment, depth sweep
02_couplings/ plmc .model, EC table, run summary
03_scores/   per_residue_scores.tsv
04_masks/    mask_C*_S*.json, mask_grid_summary.tsv, always_fix.json
05_validation/ report.txt
06_round1_sigma_v2/ rescue-v2 masks, counts, audit/generation/preflight scripts
env/         environment.yml, plmc build
scripts/     11-16
logs/        %x_%j.out
```

`always_fix.json` (the 15 structure-derived active-site residues) is kept **separate** from
the grid masks, so you can always tell what the scores recovered on their own from what
was forced.
