# PLAN: Tetramer Refold Gate (Uricase quaternary + ligand-pocket validation)

Status: **DESIGN / v0 = WT pilot + metric calibration.** Gate-on-designs is v1 (deferred).
Owner: coder4tetramer. Reference role model: `PLAN_RF_REFINE.md` (phased, comparable, executable).

---

## 0. Objective and scope

Our current structural gate folds each designed sequence as a **monomer** (ESMFold → scTM / bb_RMSD / pLDDT + monomer active-site shell RMSD). Uricase (urate oxidase) is a **homotetramer whose catalytic pocket is built at the subunit interface** — the `uricase_q00511_active_site_v0.yaml` manifest already records `Asn255 = binding_cross_protomer_W1`, i.e. at least one substrate-binding residue comes from the **neighbouring** protomer. A monomer prediction is therefore **structurally blind** to whether a de-immunized design still assembles the functional quaternary pocket. A design can pass every monomer metric and still fail to fold back into the native tetramer.

**Core objective of this task:** add a **tetramer refold gate** — predict a design as a 4-copy homo-oligomer (holo, with a substrate/analog ligand) and decide whether it **folds back into the native tetramer** with the interfacial active-site pocket intact.

**v0 scope (this PLAN):** stand up the Protenix tetramer+ligand prediction path, run the **WT positive control** (Q00511, known active), calibrate the gate metrics and thresholds against the experimental holo tetramer crystal **1R51**, and establish negative-control separation. **v0 does NOT wire the gate onto the design population** — that is v1 (§7).

**Non-goals (v0):** batch-screening the existing de-immunized designs; per-design MSA generation at scale; ligand-distance as a hard gate.

---

## 1. Locked decisions (from design discussion)

| Decision | Value | Rationale |
|---|---|---|
| Predictor | **Protenix v2.0.0** primary; AF3 for WT cross-check only | Open weights, `count` field, fits GPU, easy to batch. AF3 weights gated + heavy MSA setup → poor fit for per-design scale. |
| MSA | **ON (mandatory), via login-node ColabFold server.** No-MSA rejected (empirically too inaccurate). | MSA-free too inaccurate. Compute nodes have no internet; the **login node reaches `api.colabfold.com` (verified)**. So run `protenix msa --msa_server_mode protenix` on the **login node** → `protenix pred` on the **compute node**. AF3 local-DB HMMER search = slow fallback only. |
| Gate placement | **Second-layer benchmark supplement** — post-hoc on the returned/final design set; **NOT** used inside generation or refinement | Layer-1 monomer gate (monomer active-site RMSD + global structure) already exists; monomer fold quality is *necessary* for tetramer assembly (bad monomer ⇒ bad tetramer), so monomer-first is a valid cheap pre-filter and the tetramer gate only adds information on monomer-passing designs. |
| Tetramer-gate metrics | complex TM + D2 + **cross-protomer active-site distance panel**; **drop monomer active-site RMSD** (already in layer-1) | The unique value-add is the **inter-subunit** catalytic-pocket geometry — which no monomer metric and no confidence metric captures. Monomer active-site RMSD would be redundant. |
| Ligand mode | **holo throughout** (WT and designs both predicted WITH ligand) | 1R51 is holo; relative gate (design-holo vs WT-holo) is apples-to-apples; free pocket readout. |
| Ligand identity | **uric acid (URC)** default, SMILES `O=C1NC(=O)C2=C(N1)NC(=O)N2`; **WT pilot additionally runs 8-azaxanthine (AZA)** for the 1R51 crystal-pose overlay | URC = true substrate (best probe *if* the model generalizes); AZA = the actual ligand in 1R51, needed for a precise pose sanity check. Ligand identity does **not** affect the gate (gate is on protein assembly, relative, both holo). |
| Oligomer | `count: 4` homotetramer, D2/222 target symmetry | Uricase biological assembly. |
| Sequence form | **Met-excluded, 301 aa** (drop the initial Met of the 302-aa FASTA) | Matches 1R51 / 1wrr crystal numbering and the `index_0b` convention of the active-site manifest. |
| Ligand-distance role | **computed + reported, NEVER a gate** | Trustworthy on WT (1R51 in training set) but unreliable on mutants (co-fold pose can be hallucinated). Advisory only. |
| Gate hard criterion | **quaternary geometry vs WT (Tier 2)**; self-confidence (Tier 1) is a cheap pre-filter, not sufficient | Confidence metrics answer "is this a confident assembly", not "is this the *right* assembly". |

---

## 2. Metrics: the two-tier gate

Rationale: every metric Kaiyi tested (ipTM, chain_pair_ipTM, inter-chain PAE/PDE, pTM, has_clash) is a **model self-confidence** signal — necessary but **not sufficient**, because a mutant can be *confidently wrong* (high ipTM on a non-native assembly, or a subtle interface shift that displaces the catalytic pocket while ipTM stays high). The gate therefore layers a **geometry-vs-WT** tier on top.

### Tier 1 — self-confidence pre-filter (native Protenix outputs, ~free; reject cheaply)
| Metric | Source | Gate |
|---|---|---|
| `ipTM` | summary confidence | ≥ τ₁ |
| `min(chain_pair_ipTM)` over protein–protein pairs | chain-pair matrix | ≥ τ₂ **(retained per Kaiyi)** |
| `has_clash` | summary confidence | must be `False` |
| per-chain mean `pLDDT` (each subunit folds) | per-atom/residue pLDDT | ≥ τ₃ |
| **active-site-interface inter-chain PAE** | PAE submatrix between the two protomers' pocket residues | ≤ τ₄ **(retained, but focused on the catalytic interface residue pairs, sharper than global off-diagonal PAE)** |

### Tier 2 — geometry-vs-WT hard gate (the actual "refold **back**")
| Metric | Definition | Gate |
|---|---|---|
| **Complex TM-score** | symmetry-aware multi-chain alignment (US-align `-mm 1` / MM-align) of predicted tetramer vs **WT-predicted** tetramer (and vs 1R51 for WT pilot), TM normalized by reference | ≥ τ₅ |
| **Cross-protomer active-site distance panel** | for each of the 4 interfacial active sites, a small fixed set of **inter-chain** distances between the same-chain catalytic core (Lys11/Thr58/His257) and the neighbour-chain contributed residue(s) (manifest `Asn255 = binding_cross_protomer_W1`, + any additional neighbour-chain contacts found in 1R51), functional-atom or Cα; gate on **deviation vs WT-predicted** (relative) / vs 1R51 (absolute, pilot). Exact residue pairs fixed in Phase 0 from the 1R51 interface contact map, anchored on the manifest anchors. | |Δd| ≤ δ |
| **D2/222 symmetry recovery** | does the assembly reconstitute the dimer-of-dimers 222 point group (vs a collapsed/wrong assembly) | boolean pass |

**Why the distance panel and not monomer active-site RMSD (per review):** the layer-1 structure gate already enforces the *intra-chain* active-site RMSD, so re-checking it here is redundant. What layer-1 (and every metric above) cannot see is the **inter-subunit** juxtaposition that forms the real catalytic pocket: `ipTM`/`chain_pair_ipTM`/inter-chain PAE report only *confidence* that an interface is defined (not its geometry vs native); complex TM + D2 report only the *global* assembly shape (can stay high while a local pocket drifts a few Å); BSA/contact-recovery report interface *size* (not catalytic-residue placement). The cross-protomer distance panel is the one metric that directly asks "did the two chains reassemble the catalytic pocket with the right geometry" — it is the tetramer gate's reason to exist. It is more trustworthy than ligand distance (protein backbone/side-chain geometry, not a hallucination-prone docked pose), and the relative-to-WT form cancels systematic model bias, so it **can** gate (soft), unlike ligand distance.

### Advisory (reported, NOT gated)
- **Ligand–pocket distance**: min heavy-atom distance from ligand to catalytic residues (e.g. His257 / Lys11), and ligand token PAE vs pocket. Compared to WT-predicted and (WT only) to the 1R51 crystal pose.
- **Interface BSA / inter-chain contact count vs WT** — physical interface size (freesasa); complements ipTM (a collapsed interface can still get an anomalously high ipTM).
- **Interface contact recovery** — fraction of native inter-chain contacts (CB–CB < 8 Å) recovered, focused on the active-site interface.

Thresholds τ₁…τ₅, ε are **calibrated in the WT pilot** (§5–§6), not pre-set.

---

## 3. Method / execution environment

Two-environment split, reusing the existing "predict out-of-env, read cache" architecture already encoded in `inverse_folding/evaluation/refold.py` (`CACHE_READ_BACKENDS = {esmfold2, protenix, af3}`):

1. **Predict (Protenix env, out-of-env):** `/scratch/gpfs/KAIYIJIANG/tools/protenix` (v2.0.0, Python 3.12 / torch 2.7.1). Wrapper `bin/protenix-env.sh` + `bin/protenix-run`; submit template `slurm/protenix_pred.slurm` (A100-40G, gpu-short, 128G, 10h). → writes ranked mmCIF (full per-atom coords incl. ligand) + `*_summary_confidences.json` + `*_confidences.json` (PAE/PDE/pLDDT/ipTM/chain_pair_iptm/ranking/has_clash).
2. **Evaluate (immune-design env):** read predicted CIF + confidence JSON + reference structures → compute Tier 1/Tier 2/advisory → `tetramer_gate.parquet`.

### 3.1 MSA wiring (Phase 0 — resolved: server-first)
The shared Protenix wrapper is no-MSA by default (`PROTENIX_USE_MSA=1` flips it), and ships no local databases. MSA-free is rejected (too inaccurate). Strategy, in priority order:

1. **PRIMARY — login-node ColabFold server.** Protenix v2 ships `protenix msa --msa_server_mode protenix`, which queries the remote ColabFold MMseqs2 API (`api.colabfold.com`, in `protenix/web_service/colab_request_utils.py`). Della **compute nodes have no internet, but the login node does** (reachability to `api.colabfold.com` verified). So: run the MSA search on the **login node** (light — a remote API call, not local compute), persist the MSA, then run `protenix pred` on the **compute node** with the MSA attached (`proteinChain.pairedMsaPath` / the `msa` out_dir convention). This is the fast path the user asked to prioritize.
2. **Fallback A — local ColabFold search.** `scripts/colabfold_msa.py` + local ColabFold DB + MMseqs2 (`docs/colabfold_compatible_msa.md`); offline but needs the DB downloaded.
3. **Fallback B — AF3 local-DB HMMER.** AF3 at `/scratch/gpfs/KAIYIJIANG/tools/alphafold3` (patched HMMER `hmmer/3.4-patched/bin`) was used for a full-MSA local-DB Pegloticase prediction, so its genetic DBs exist locally. **Slow** (user's experience) → last resort only.

Homotetramer reuses **one** MSA for the single sequence (built once, shared across the 4 copies). **The WT pilot needs exactly one login-node MSA call.** Deferred to v1: for design screening, de-immunized variants are sparse substitutions (same length, no indels) → **reuse the WT MSA with the query row swapped** ("MSA transplant") to avoid a per-design search; validate transplant-vs-fresh on one design before relying on it.

### 3.2 Input JSON shape (Protenix dialect, confirmed on v2 example)
```json
[{
  "name": "uricase_WT_holo_urc",
  "sequences": [
    { "proteinChain": { "sequence": "<301aa Met-excluded>", "count": 4,
                        "pairedMsaPath": "<wt_msa.a3m/paired>" } },
    { "ligand": { "ligand": "<SMILES O=C1NC(=O)C2=C(N1)NC(=O)N2 | CCD_URC | CCD_AZA>", "count": 4 } }
  ]
}]
```
Phase 0 smoke must confirm the `ligand` entity + `count` + `pairedMsaPath` are accepted on the installed v2.0.0 (docs/ locally only has README; ligand schema is from upstream docs).

---

## 4. Scripts (Reuse-First; register in `doc/SCRIPTS.md`)

| Script | Env | Purpose | Reuse note |
|---|---|---|---|
| `scripts/build_tetramer_input.py` | immune-design | FASTA/seq (+ optional MSA path, ligand code) → Protenix JSON with `count:4` protein + `count:4` ligand; Met-strip option; batch mode for a design table | new |
| `scripts/submit_tetramer_predict.slurm` | protenix | thin wrapper: `protenix-run msa/prep` (AF3 DBs) → `protenix-run pred`; array-capable over a JSON dir | **wraps** `tools/protenix/slurm/protenix_pred.slurm`; do NOT duplicate the base template |
| `scripts/eval_tetramer_gate.py` | immune-design | read predicted CIF + confidence JSON + reference (WT-predicted / 1R51) → Tier1+Tier2+advisory → `tetramer_gate.parquet` + pass/fail | reuse `run_tmalign` / Kabsch from `scripts/sc_rmsd.py`; add US-align/MM-align call + freesasa |

External binary to add (Phase 0): **US-align** (MM-align mode) — single static binary; `TMalign` already used single-chain in the repo but is not symmetry-aware multi-chain.

---

## 5. WT pilot protocol (v0, executable)

### Phase 0 — verify & wire (no science yet)
0.1 **MSA server path (priority):** on the login node, run `protenix msa --msa_server_mode protenix` on a single test sequence → confirm it returns an MSA; confirm the MSA feeds `protenix pred` on a compute node. (Fallbacks: local ColabFold search, then AF3 local-DB HMMER — locate AF3 DBs only if the server path fails.)
0.2 Install US-align binary (MM-align mode) into the immune-design toolchain; confirm `TMalign` location in the env.
0.3 Ligand-schema smoke: minimal 2-copy protein + 1 ligand JSON through `protenix-run pred` (no-MSA, fast) → confirm the ligand entity + `count` parse and a CIF with ligand atoms is produced on v2.0.0.
0.4 Fetch **1R51 biological assembly** (tetramer) + ligand from RCSB: `1r51.pdb1` / `1R51-assembly1.cif`; extract the 4 protein chains + AZA; record the crystal active-site + AZA pose as the ground-truth reference. **Fix the cross-protomer distance-panel residue pairs from the 1R51 interface contact map** (§2 Tier 2), anchored on the manifest anchors.

### Phase 1 — WT holo prediction + method accuracy
WT = the if-ready Q00511 row from `work/.../Uricases_RF/uricase_caseset_if_ready_dedup_labeled.parquet` (pipeline form), cross-checked against the 302-aa `Q00511_Aspergillus_flavus.fasta` seed; **predicted Met-excluded 301-aa** to match 1R51. (P16164/Pegloticase is an optional second positive control but has no A. flavus crystal match — skip in v0.)
1.1 Build WT MSA once (login-node server, 301-aa Met-excluded Q00511).
1.2 Predict **WT holo tetramer, ligand = URC** (`count:4` + URC `count:4`, MSA on). Multiple seeds.
1.3 Predict **WT holo tetramer, ligand = AZA** (for the 1R51 overlay).
1.4 Compute all metrics for WT-predicted vs **1R51**: complex TM-score, active-site interface RMSD, D2 recovery, per-chain pLDDT, ipTM/chain_pair, active-site-interface PAE, ligand(AZA)–pocket RMSD vs crystal.
1.5 **This establishes the achievable ceiling** and the metric values a genuinely-native tetramer produces.

### Phase 2 — negative controls + threshold lock
2.1 Negative controls (no known assembly-failing mutant provided → synthetic; user can substitute real ones):
   - **Monomer degenerate control**: predict WT as `count:1` (no interface can form) → Tier-1/interface metrics must collapse.
   - **Interface-scrambled control**: randomize/ablate a block of interface residues (keep the fold core) → should fail Tier 2 while possibly retaining monomer fold.
2.2 Set τ₁…τ₅, ε in the **gap** between the WT ceiling (Phase 1) and the negative-control floor (Phase 2). Record calibration table.
2.3 (Optional) AF3 cross-check on WT holo to confirm Protenix's WT assembly is not a single-model artifact.

---

## 6. Acceptance criteria (comparable)

**Phase 1 (method reproduces the known tetramer) — pilot succeeds iff:**
- WT-predicted holo tetramer vs 1R51: symmetry-aware **complex TM-score ≥ 0.90**, **D2 recovered**, per-chain **pLDDT ≥ 0.80**, **ipTM ≥ 0.90**, `has_clash = False`.
- Cross-protomer active-site distance panel (WT-pred vs 1R51) matches within a tight tolerance (target **≤ ~2 Å** per pair): the neighbour-chain `Asn255` (W1) and any other cross-chain contacts are correctly juxtaposed with the same-chain catalytic core (Lys11/Thr58/His257) at all 4 interfacial sites.
- AZA placed in the interfacial pocket with heavy-atom RMSD to the 1R51 crystal pose **≤ ~2–3 Å** (advisory sanity; confirms the pocket readout is meaningful on a training-set case).

**Phase 2 (gate is discriminative) — succeeds iff:**
- Both negative controls fall **clearly below** the chosen Tier-1 and/or Tier-2 thresholds (measurable margin, not overlapping the WT distribution).
- A calibration table with each threshold justified by the ceiling/floor separation is written.

If Phase 1 fails (WT does not reproduce 1R51), the gate is not trustworthy — **stop and diagnose** (MSA quality, seq form, ligand handling) before any design use.

---

## 7. Pipeline integration design (v1, deferred — designed here, not built)

**Placement (per review): a second-layer BENCHMARK supplement, not a refinement-time gate.** The tetramer gate runs **post-hoc on the final/returned design set** — the designs that already passed the layer-1 monomer gate (monomer active-site RMSD + global structure) during generation/refinement. It is **not** consulted inside the generation or refinement loop. Rationale: monomer fold quality is *necessary* for tetramer assembly, so the cheap monomer gate is a valid pre-filter and the expensive tetramer prediction only runs on survivors.

- **Mechanism to be aware of (not a reason to move it earlier):** de-immunization targets *surface* MHC-II epitopes, and protein–protein interfaces are also surface-located → de-immunizing mutations may **preferentially** land on/near interface residues. So the tetramer gate is not a rubber-stamp on monomer-passers — it may reject a non-trivial fraction by catching real interface/pocket damage the monomer gate is blind to. That is exactly its purpose.
- Implementation: `eval_tetramer_gate.py` consumes a design table + the Protenix tetramer predictions and **adds columns** to the benchmark output (it does not feed back). Tier 1 pre-filters (skip the geometry compute on obvious fails), Tier 2 sets `tetramer_gate_pass`, ligand distance flows to the report only.
- The predicted tetramer + confidence are stored as a **new multi-chain cache-read product** analogous to `refold.py`'s existing single-chain `protenix` backend (tetramer CIF + confidence JSON keyed on `(protein_id, sequence)`), with a multimer reader.
- Design-scale MSA via the **WT MSA transplant** (§3.1), validated before rollout.
- Column additions to the benchmark/design table: `tetra_complex_TM`, `tetra_xprot_as_dist_dev` (cross-protomer active-site distance deviation vs WT), `tetra_D2_ok`, `tetra_ipTM`, `tetra_min_chain_pair_ipTM`, `tetra_as_interface_PAE`, `tetra_has_clash`, `tetra_interface_BSA`, `tetra_ligand_pocket_dist` (advisory), `tetramer_gate_pass`.

---

## 8. Risks & open items

- **MSA path (medium; primary verified):** login-node `protenix msa --msa_server_mode protenix` → `api.colabfold.com` is reachable from the login node, but depends on that external service (rate limits / downtime) and on the login-node→compute-node MSA hand-off working cleanly. Fallbacks (local ColabFold search, AF3 local-DB HMMER) are slower. Phase 0 must prove the hand-off before Phase 1.
- **Ligand pose hallucination:** the reason ligand distance is advisory-only; do not let a good-looking ligand pose on a mutant imply an intact pocket.
- **Ligand ≠ crystal ligand:** default URC vs 1R51's AZA — only matters for the precise pose overlay (Phase 1 runs AZA for that); irrelevant to the gate.
- **Negative-control realism:** synthetic controls may not mimic the true failure mode (subtle interface drift). Prefer a real assembly-failing / inactive mutant if the user can supply one.
- **Symmetry-aware alignment correctness:** chain mapping under D2 symmetry must use MM-align/US-align proper multi-chain mode, not naive chain-order alignment.
- **GPU fit:** 4×301 + ligand ≈ 1.2k tokens ≈ 20–40 GB; A100-40G template may be tight → be ready to request an 80 GB card (H200/A100-80G) if OOM.
- **Met-numbering:** predict Met-excluded 301-aa so anchor `index_0b` and 1R51 numbering align; a 1-off here silently corrupts every interface-RMSD number.

---

## 9. Deliverables (v0)

1. `PLAN_TETRAMER_GATE.md` (this file).
2. Three registered scripts (§4).
3. `tetramer_gate.parquet` for WT (URC + AZA) + negative controls, with the full metric set.
4. Calibration table (τ₁…τ₅, ε) with ceiling/floor justification.
5. A short result note: does Protenix reproduce the 1R51 tetramer + pocket, and is the gate discriminative — go/no-go for v1 integration.
