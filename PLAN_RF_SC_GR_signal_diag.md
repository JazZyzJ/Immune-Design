# PLAN — SC-GR Signal-Direction Diagnostic

**Thread:** SC-GR (self-conditioned GR probe). Follow-up to RAR 0017 (remask close-out).
Linked from `PLAN_RF_SC_GR.md`.

## 0. Question & hypothesis

Decision B (Stage 1) + RAR 0017 closed the actuation **strength/retention** surface and the
**remask** substrate. The one untested steering axis is the **candidate-ranking signal
direction**: D2 ranks candidates by a **local** head ΔR (`realized_benefit`), which RAR 0014
showed is decorrelated from terminal immune. This diagnostic asks, at the **candidate level**:

> Within a real D2 decision point's structure-safe candidate set, (a) is there immune
> headroom at all (do candidates differ in terminal immune?), and if so (b) does the SC-GR
> probe's terminal estimate rank candidates by their true terminal effect better than the
> local ΔR?

Caveat to keep in front: the probe was validated **protein-level** (ρ0.75, RAR 0010). This
tests a different, harder task — **within-position candidate discrimination** — which is
**unvalidated**. A good protein-level sensor is not automatically a good candidate ranker.

## 1. The three quantities (per decision point i, candidate c)

| symbol | meaning | how |
|---|---|---|
| **L**_{i,c} | local signal D2 uses now = ΔR_B at the refresh on the partial completion | read from D2 attribution telemetry (`write_independent_delta_R_i`); fallback: recompute via `build_safe_candidate_support` + head at the snapshot x_t |
| **P**_{i,c} | probe terminal estimate of c | commit c at p, freeze p, run the SC-GR probe (K_P=3 forks, `topm_lse`) — the RAR-0010 machinery |
| **Y**_{i,c} | oracle terminal effect of c | commit c at p, freeze p, **K_Y=4 paired-seed pure completions** (controller OFF), head-score the finished designs |

**Y** is scored two ways (both reported):
- **Y_register** (primary, sensitive): terminal head excess summed over the binding windows covering position p.
- **Y_whole** (secondary, the real target): whole-design `global_risk`.

**Paired seed is mandatory:** all candidates at one decision point share the same completion
RNG, so `Y_c − Y_c'` isolates the candidate (the rest of the completion cancels). This makes the
**contrasts/rankings clean even at small K**; absolute Y is noisy, contrasts are not.

`Y` uses **controller OFF + frozen p** in the continuation because that is exactly the quantity
the probe estimates (terminal burden of x_t with c committed) — so P-vs-Y is apples-to-apples.
(Optional secondary `Y_realistic` = controller ON continuation, to check whether the downstream
controller/remask overwrites good candidates; not required for the primary verdict.)

## 2. Sampling (scope = 22 proteins)

- **Proteins:** 22 from the 100 worst, **stratified 11 + 11** using RAR 0013 per-protein
  B1-vs-DPLM reduction: 11 **responders** (largest reduction) + 11 **responder-tail** (≈0
  reduction / resistant). Stratify so we can see if headroom is tail-specific (tail with no
  headroom = direct hard-limit evidence).
- **Designs:** 1 per protein (reuse the existing B1 design; deterministic seed 42).
- **Decision points:** per protein, the **2–3 highest-excess** D2 decision points (refreshes
  where D2 acted on a high-risk position).
- **Candidates per point:** the structure-safe set; if ≤10 take all, else
  `top-4-by-L ∪ top-4-by-P ∪ 2 random` (span the agreement/disagreement region).
- Size: ≈ 22 × 2.5 × 8 ≈ **440 (decision, candidate) pairs**.

## 3. Metrics & pre-registered decision rules

Per decision point (then aggregate, **stratified by responder/tail**):

- **oracle_headroom** = `Y[local-chosen candidate] − min_c Y` (≥0; register & whole separately).
- **Spearman(L, Y)**, **Spearman(P, Y)** across candidates.
- **probe_capture** = `(Y[local] − Y[probe-pick]) / (Y[local] − Y[oracle])` (1 = matches oracle, 0 = no better than local, <0 = worse).
- **disagreement** = `Spearman(L, P)` (Stage-0 sanity).

**Pre-registered verdicts (fixed before results):**

1. **median oracle_headroom ≈ 0** (both register & whole; below ε set from design-level
   `global_risk` noise) → the structure-safe candidate set is immune-flat → **no ranking signal
   can help; the bottleneck is the candidate set / manifold → close out.**
2. **headroom material AND mean[Spearman(P,Y) − Spearman(L,Y)] ≥ 0.15 AND median probe_capture ≥ 0.5**
   → **probe is a useful candidate ranker → build probe-as-ranker** (replace L with P in D2 selection).
3. **headroom material BUT probe_capture ≈ 0 / Spearman(P,Y) ≈ Spearman(L,Y)** → **the probe is
   not candidate-resolved → need a different signal**, not a different probe.
4. **whack-a-mole flag:** `Y_register` has headroom but `Y_whole` does not → fixing the register
   doesn't move the design (distributed immunogenicity) → ranking can't help the hard cohort
   regardless of signal (expect concentrated in the tail stratum).

## 4. Phases (and where each runs)

| phase | what | where |
|---|---|---|
| **P0 — select** | pick 22 stratified proteins (RAR 0013); enumerate decision points + candidate sets + L from B1 telemetry | **local** (telemetry already returned) |
| **P1 — snapshot** | deterministically re-run each protein (seed 42, B1 config) to dump x_t at the chosen decision steps | **cluster (GPU)** |
| **P2 — score** | for each (snapshot, candidate): commit+freeze p; Y = K_Y=4 paired pure completions + head; P = K_P=3 probe forks; write `signal_diag_candidates.parquet` | **cluster (GPU)** |
| **P3 — analyze** | headroom / Spearman / probe_capture / whack-a-mole, stratified; apply §3 rules; RAR | **local (CPU)** |

`signal_diag_candidates.parquet` schema (one row per candidate):
`protein_id, stratum{responder,tail}, decision_id, step, position, candidate_token,
L, P, Y_register, Y_whole, is_local_pick(bool), is_oracle_pick(bool)`.

## 5. Reuse (do not write a new sampler)

- **P1/P2 extend the SC-GR probe machinery from RAR 0010** (it already forks completions from an
  x_t, head-scores, and aggregates `topm_lse`). Add a **candidate-scoring mode**: input a snapshot
  x_t + position p + candidate set; for each candidate commit+freeze p, run K_Y oracle completions
  (controller off) and K_P probe forks; emit the parquet. Register the extension in `doc/SCRIPTS.md`.
- Candidate sets + L from `build_safe_candidate_support` (counterfactual.py) and the D2 attribution
  telemetry — do **not** re-derive the candidate gate.
- All cluster paths via CLI args (no hardcoding). Print K_Y, K_P, n_proteins, n_decision_points,
  candidate-selection rule, controller-off flag, paired-seed base to stdout.

## 6. Cost

P1: 22 deterministic re-runs (~half-trajectory each). P2: ≈440 × (K_Y=4 + K_P=3) ≈ **3.1k
half-completions** (~50 steps, batchable). Order a few GPU-hours. If P3 hits rule 1 (no
headroom) the result is decisive on the K_Y oracle alone; the K_P probe forks could even be
deferred until headroom is confirmed.

## 7. Out of scope

Not building the ranker here (that's gated on verdict 2). Not touching remask (closed, RAR 0017)
or the strength/retention knobs (closed, Decision B). `Y_realistic` (controller-on continuation)
is optional and secondary.
