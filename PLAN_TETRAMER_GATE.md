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
| MSA | **ON (mandatory), via local GPU ColabFold.** No-MSA rejected (empirically too inaccurate). | The shared GPU-indexed UniRef30 + environmental DB under `/scratch/gpfs/KAIYIJIANG/databases/colabfold` is faster, deeper, offline, and unthrottled. The deprecated login-node `api.colabfold.com` path is no longer part of the workflow. |
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

1. **Predict (Protenix env, out-of-env):** `/scratch/gpfs/KAIYIJIANG/tools/protenix` (v2.0.0, Python 3.12 / torch 2.7.1). Wrapper `bin/protenix-env.sh` + `bin/protenix-run`; the public `slurm/protenix_pred.slurm` now defaults to `MSA_MODE=local_gpu` on ailab H200 (240 GB host memory, 10 h), with `MSA_MODE=input|none|protenix_server` retained explicitly. It writes ranked mmCIF (full per-atom coordinates including ligand) + `*_summary_confidences.json` + `*_confidences.json` (PAE/PDE/pLDDT/ipTM/chain_pair_iptm/ranking/has_clash).
2. **Evaluate (immune-design env):** read predicted CIF + confidence JSON + reference structures → compute Tier 1/Tier 2/advisory → `tetramer_gate.parquet`.

### 3.1 MSA wiring (Phase 0 — resolved: local GPU)
MSA-free prediction is rejected as too inaccurate. The implemented strategy is:

1. **PRIMARY — local GPU ColabFold.** `scripts/submit_tetramer_msa_local.slurm` and the shared public launchers use GPU MMseqs2 over `/scratch/gpfs/KAIYIJIANG/databases/colabfold` (`uniref30_2302_db` + `colabfold_envdb_202108_db`, GPU indexed). The workflow is offline, unthrottled, and runs on a GPU compute node.
2. **Precomputed/custom MSA.** Protenix reads `pairedMsaPath` + `unpairedMsaPath`; use the public `MSA_MODE=input` for a custom heteromer-paired MSA. The local default conservatively uses a query-only paired MSA plus the full-depth unpaired A3M and does not invent heteromer pairing.
3. **Compatibility only.** `MSA_MODE=protenix_server` retains the upstream remote server path when egress is available; `MSA_MODE=none` retains no-MSA. AF3 native HMMER remains the slow `MSA_MODE=af3_native` fallback in the shared AF3 launcher.

A homotetramer reuses **one** MSA for its single unique sequence across all four copies. Large benchmark arrays precompute each design's A3M once and reuse it across prediction modes rather than loading the ColabFold database independently in every prediction task.

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
| `scripts/build_protenix_jsons.py` | immune-design | Per-design ColabFold A3M + manifest → Protenix JSON with `pairedMsaPath`/`unpairedMsaPath`, configurable protein/ligand copy count, and ready-list | implemented |
| `scripts/submit_tetramer_msa_local.slurm` | colabfold module | Batched local GPU ColabFold search against the shared GPU-indexed database | implemented |
| `scripts/submit_tetramer_predict.slurm` | protenix | Array-capable Protenix prediction over MSA-ready JSONs; keeps MSA search separate so one database load serves a batch | implemented; reuses the shared `protenix-run` wrapper |
| `scripts/eval_complex_gate.py` | immune-design | read predicted CIF + confidence JSON + reference (WT-predicted / 1R51) → Tier1+Tier2+advisory → `tetramer_gate.parquet` + pass/fail | reuse `run_tmalign` / Kabsch from `scripts/sc_rmsd.py`; add US-align/MM-align call + freesasa |

External binary to add (Phase 0): **US-align** (MM-align mode) — single static binary; `TMalign` already used single-chain in the repo but is not symmetry-aware multi-chain.

---

## 5. WT pilot protocol (v0, executable)

### Phase 0 — verify & wire (no science yet)
0.1 **Local GPU MSA path:** run `scripts/submit_tetramer_msa_local.slurm` on a single test sequence, confirm a query-matched A3M is emitted, build the Protenix MSA paths with `scripts/build_protenix_jsons.py`, and confirm `protenix pred --use_msa true` consumes them without a server call.
0.2 Install US-align binary (MM-align mode) into the immune-design toolchain; confirm `TMalign` location in the env.
0.3 Ligand-schema smoke: minimal 2-copy protein + 1 ligand JSON through `protenix-run pred` (no-MSA, fast) → confirm the ligand entity + `count` parse and a CIF with ligand atoms is produced on v2.0.0.
0.4 Fetch **1R51 biological assembly** (tetramer) + ligand from RCSB: `1r51.pdb1` / `1R51-assembly1.cif`; extract the 4 protein chains + AZA; record the crystal active-site + AZA pose as the ground-truth reference. **Fix the cross-protomer distance-panel residue pairs from the 1R51 interface contact map** (§2 Tier 2), anchored on the manifest anchors.

### Phase 1 — WT holo prediction + method accuracy
WT = the if-ready Q00511 row from `work/.../Uricases_RF/uricase_caseset_if_ready_dedup_labeled.parquet` (pipeline form), cross-checked against the 302-aa `Q00511_Aspergillus_flavus.fasta` seed; **predicted Met-excluded 301-aa** to match 1R51. (P16164/Pegloticase is an optional second positive control but has no A. flavus crystal match — skip in v0.)
1.1 Build WT MSA once with local GPU ColabFold (301-aa Met-excluded Q00511).
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
- Implementation: `eval_complex_gate.py` consumes a design table + the Protenix tetramer predictions and **adds columns** to the benchmark output (it does not feed back). Tier 1 pre-filters (skip the geometry compute on obvious fails), Tier 2 sets `tetramer_gate_pass`, ligand distance flows to the report only.
- The predicted tetramer + confidence are stored as a **new multi-chain cache-read product** analogous to `refold.py`'s existing single-chain `protenix` backend (tetramer CIF + confidence JSON keyed on `(protein_id, sequence)`), with a multimer reader.
- Design-scale MSA uses fresh per-design local ColabFold A3Ms, batched so the database load is amortized across many sequences.
- Column additions to the benchmark/design table: `tetra_complex_TM`, `tetra_xprot_as_dist_dev` (cross-protomer active-site distance deviation vs WT), `tetra_D2_ok`, `tetra_ipTM`, `tetra_min_chain_pair_ipTM`, `tetra_as_interface_PAE`, `tetra_has_clash`, `tetra_interface_BSA`, `tetra_ligand_pocket_dist` (advisory), `tetramer_gate_pass`.

---

## 8. Risks & open items

- **MSA path (low; verified):** local GPU ColabFold uses the shared GPU-indexed UniRef30 + environmental database and emits query-validated A3Ms entirely on compute nodes. There is no external-service or login-node hand-off dependency. The main operational cost is one large database load per MSA batch.
- **Ligand pose hallucination:** the reason ligand distance is advisory-only; do not let a good-looking ligand pose on a mutant imply an intact pocket.
- **Ligand ≠ crystal ligand:** default URC vs 1R51's AZA — only matters for the precise pose overlay (Phase 1 runs AZA for that); irrelevant to the gate.
- **Negative-control realism:** synthetic controls may not mimic the true failure mode (subtle interface drift). Prefer a real assembly-failing / inactive mutant if the user can supply one.
- **Symmetry-aware alignment correctness:** chain mapping under D2 symmetry must use MM-align/US-align proper multi-chain mode, not naive chain-order alignment.
- **GPU fit:** 4×301 + ligand ≈ 1.2k tokens ≈ 20–40 GB. The public default is H200; A100-40G may be tight and should be treated as an explicit resource override, not the baseline.
- **Met-numbering:** predict Met-excluded 301-aa so anchor `index_0b` and 1R51 numbering align; a 1-off here silently corrupts every interface-RMSD number.

---

## 9. Deliverables (v0)

1. `PLAN_TETRAMER_GATE.md` (this file).
2. Three registered scripts (§4).
3. `tetramer_gate.parquet` for WT (URC + AZA) + negative controls, with the full metric set.
4. Calibration table (τ₁…τ₅, ε) with ceiling/floor justification.
5. A short result note: does Protenix reproduce the 1R51 tetramer + pocket, and is the gate discriminative — go/no-go for v1 integration.
