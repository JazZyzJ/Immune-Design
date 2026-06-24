# PLAN — SC-GR Signal-Direction Diagnostic

**Thread:** SC-GR. Follow-up to RAR 0017 (remask close-out). Linked from `PLAN_RF_SC_GR.md`.

## 0. Question & hypothesis

Decision B (Stage 1) + RAR 0017 closed the actuation **strength/retention** surface and the
**remask** substrate. The one untested steering axis is the **candidate-ranking signal
direction**: D2 ranks **block candidates** by a **local** signal `R_B` (the block's local head
risk), which RAR 0014 suggested is decorrelated from terminal immune. This diagnostic asks, at
the level D2 actually operates:

> Within a real D2 decision point's structure-safe **block-candidate pool**, (a) is there
> immune headroom at all (do candidates differ in terminal immune?), and if so (b) does a cheap
> **probe-based** signal rank candidates by their true terminal effect better than the local `R_B`?

**Unit = block candidate, not single token.** D2 edits up to `max_positions_per_block` positions
jointly (`A_B`) and scores **token tuples** over those positions; `R_B` is the tuple's local
risk, not an independent per-token score. The whole diagnostic is at tuple granularity.

Caveat to keep in front: the probe was validated **protein-level** (ρ0.75, RAR 0010). This tests
a different, harder task — **within-block candidate discrimination** — which is **unvalidated**.

## 1. The three signals (per decision point i = one D2 block, per candidate tuple t)

Convention: **all three are levels, lower = less immunogenic = better. The "pick" is `argmin`.**

| symbol | meaning | how |
|---|---|---|
| **L**_{i,t} = R_B(t) | local block risk D2 ranks by (LME over the block's windows on the partial completion) | reconstructed in **P2** on the snapshot (see §2); cross-checked vs telemetry |
| **P**_{i,t} | cheap probe estimate of terminal burden with tuple t fixed | **cheap-probe** (below), K_P=3 forks, `topm_lse` |
| **Y**_{i,t} | oracle terminal effect of t | commit t on `A_B`, **freeze `A_B`**, **K_Y=3 paired-seed real completions** (controller OFF, full denoiser to t=1), head-score |

**Y is scored two ways:**
- **Y_register** (primary): `LME` over **Ω(B)** = the union of binding windows covering **any**
  position in `A_B` (not just one position), of terminal per-window risk on the finished design.
- **Y_whole** (secondary, real target): whole-design `global_risk`.

**cheap-probe (the deployable signal, pin this down):** fix tuple t into x_t on `A_B`; **do NOT
re-run the denoiser**; run the SC-GR probe's pseudo-terminal completion over the *other* masked
positions using the **original pre-D2 structural logits** (K_P=3, `topm_lse`). This is what a
future affordable ranker could compute per candidate. It is a **cheap estimate of Y, not equal to
Y** — we are explicitly testing whether this cheap signal predicts the true Y better than `R_B`
does. (A pricier **conditioned-probe** — re-run the denoiser after committing t, then probe — is
closer to Y but is effectively a new mechanism; only consider it if cheap-probe shows partial
signal. Not in primary scope.)

**Paired seed (mandatory):** all tuples at one decision point share the K_Y completion RNG, so
`Y_t − Y_t'` isolates the tuple (the rest of the completion cancels) — contrasts/rankings are
clean at small K even though absolute Y is noisy.

## 2. Sampling (scope = 22 proteins) and pool construction

- **Proteins:** 22 from the 100 worst, **stratified 11 + 11** via RAR 0013 per-protein
  B1-vs-DPLM reduction: 11 **responders** (largest reduction) + 11 **responder-tail** (≈0).
- **Designs:** 1 per protein, deterministic seed 42 (reuse B1 config).
- **Decision points:** per protein, the **2–3 highest-excess D2 blocks** (refreshes where D2
  acted). A decision point = (protein, refresh step, block id / `A_B`).
- **Candidate pool — built in P2, NOT P0:** local telemetry lacks `struct_logits` and the full
  tuple pool. P2 replays to the snapshot and **reconstructs** the safe support + block-candidate
  pool + `R_B` via `build_safe_candidate_support` + the D2 scoring path, under the **same D2
  config** as the source run. Use telemetry `support_size` / `selected_positions` only as a
  **consistency check**.
- **No P-based candidate selection (avoid circularity).** Reconstruct with a manageable cap
  (`max_candidates_per_block ≈ 16`); if the pool ≤ ~16 run Y on **all** tuples (cleanest
  interpretation); if larger, **uniform-random** subsample to ~16 (never select by L or P).
  cheap-P (one-shot, negligible cost) is computed for the **whole** pool regardless.

## 3. Metrics & pre-registered decision rules

Per decision point (then aggregate, **stratified by responder/tail**). Picks: `local = argmin_t L`,
`probe = argmin_t P`, `oracle = argmin_t Y`.

- **oracle_headroom** = `Y[local] − Y[oracle]` (≥0; computed for Y_register and Y_whole).
- **Spearman(L, Y)**, **Spearman(P, Y)** across tuples (Y_register primary).
- **probe_capture** = `(Y[local] − Y[probe]) / (Y[local] − Y[oracle])` (1 = matches oracle, 0 = no better than local, <0 = worse).
- **disagreement** = `Spearman(L, P)` (sanity).

**Pre-registered verdicts (fixed before results):**

1. **median oracle_headroom ≈ 0** (both Y_register & Y_whole; ε from design-level `global_risk`
   noise) → structure-safe pool is immune-flat → **no ranking signal helps; bottleneck is the
   candidate set / manifold → close out.**
2. **headroom material AND mean[Spearman(P,Y) − Spearman(L,Y)] ≥ 0.15 AND median probe_capture ≥ 0.5**
   → **cheap-probe is a useful candidate ranker → build probe-as-ranker** (swap `R_B`→P in D2 selection).
3. **headroom material BUT probe_capture ≈ 0 / Spearman(P,Y) ≈ Spearman(L,Y)** → cheap signal
   not candidate-resolved → **try conditioned-probe or a different signal**, not a different probe.
4. **whack-a-mole flag:** `Y_register` has headroom but `Y_whole` does not → fixing the block's
   windows doesn't move the design (distributed immunogenicity) → ranking can't help the hard
   cohort (expect concentrated in the tail stratum).

## 4. Phases (and where each runs)

| phase | what | where |
|---|---|---|
| **P0 — select** | pick 22 stratified proteins (RAR 0013); list decision points as **(protein, refresh step, block id) only** — identifiers, not pools | **local** |
| **P1 — snapshot** | deterministically replay each protein (seed 42, B1 config) to dump x_t at the chosen decision steps | **cluster (GPU)** |
| **P2 — reconstruct + score** | on each snapshot: rebuild safe support + tuple pool + `R_B` (same D2 config; cross-check vs telemetry); compute cheap-P (whole pool); commit+freeze `A_B` and run K_Y oracle (Y) on the (capped) pool; write `signal_diag_candidates.parquet` | **cluster (GPU)** |
| **P3 — analyze** | headroom / Spearman / probe_capture / whack-a-mole, stratified; apply §3 rules; RAR | **local (CPU)** |

`signal_diag_candidates.parquet` schema (one row per tuple):
`protein_id, stratum{responder,tail}, decision_id, step, block_id, positions_json,
candidate_tokens_json, omega_window_indices_json, L, P, Y_register, Y_whole,
is_local_pick(bool), is_oracle_pick(bool)`.

## 5. Reuse (do not write a new sampler)

- P1/P2 **extend the SC-GR probe machinery from RAR 0010** (it already builds pseudo-terminal
  completions from structural logits, head-scores, aggregates `topm_lse`). Add a **block-candidate
  scoring mode**: input snapshot x_t + `A_B` + reconstructed tuple pool; per tuple commit+freeze
  `A_B`, compute cheap-P (one-shot, original logits) and K_Y oracle completions (controller off);
  emit the parquet. Register in `doc/SCRIPTS.md`.
- Tuple pool + `R_B` from `build_safe_candidate_support` (counterfactual.py) + the D2 scoring path
  — do **not** re-derive the candidate gate; do **not** trust local telemetry for the pool.
- All cluster paths via CLI args. Print K_Y, K_P, n_proteins, n_decision_points,
  max_candidates_per_block, subsample rule, controller-off flag, paired-seed base to stdout.

## 6. Cost

Cost is dominated by **Y** (full denoiser continuations); cheap-P is one-shot (negligible).
≈ 22 × 2.5 blocks × ~16 tuples × K_Y=3 ≈ **2.6k real half-completions** (~50 steps, batchable) →
a few GPU-hours. If P3 hits rule 1 (no headroom) the K_Y oracle alone is decisive.

## 7. Out of scope

Not building the ranker (gated on verdict 2). Not touching remask (RAR 0017) or strength/retention
(Decision B). conditioned-probe and `Y_realistic` (controller-on continuation) are secondary.
