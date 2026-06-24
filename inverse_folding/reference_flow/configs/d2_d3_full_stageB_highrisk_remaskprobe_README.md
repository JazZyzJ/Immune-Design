# Remask probe — clean no-remask architectural test (post-Stage-1, Decision-B follow-up)

Stage 1 (the `alpha/nu/lambda/shift` rank-face sweep,
`d2_d3_full_stageB_highrisk_rankface_*`) hit pre-registered **Decision B**: the
logit-actuation surface cannot move the Pareto axis we care about (terminal immune ↓
and scTM tail ↓ both unresponsive), and the proposal-write clamp is not the limiter
(shift 10→30 changed 1/200 designs). The one architectural variable never isolated is
the **remask** itself. This probe isolates it cleanly, as the agreed close-out test.

## Why a clean probe (not just `remask.enabled=false`)

In `sampler.py` the controller `post_step()` is gated on `remask_enabled`
(`sampler.py:243` single-lane / `:512` batched). So `sampler.remask.enabled=false`
would ALSO disable D2 post-step / D3 telemetry — conflating "no remask" with "no
controller post-step lifecycle". The clean probe keeps `enabled=true` and sets the new
knob `sampler.remask.fraction_scale=0.0` (`config.py` / `sampler.py`), which zeroes the
remask cutoff (`cutoff_len = int(n_committed * rate * fraction_scale)`,
`sampler.py:_apply_reparam_remask`) while `post_step` / telemetry still run.

`fraction_scale=1.0` (default) is byte-identical to the legacy remask — locked by
`tests/inverse_folding/test_reference_flow_sampler.py`
(`test_apply_reparam_remask_cutoff_scale_one_matches_default` and the 11 pre-existing
remask tests). `0.0` re-masks nothing.

## Arms (100 worst / n2, seed 42, constant_one)

This is a **paired remask on/off ablation at an identical controller + sampler**. The
remask-ON side already exists — the Stage-1 **`a0p0_shift30`** run (confirmed `shift30,
remask ON, n2, seed42, constant_one` at
`stageB1_highrisk_rankface_stage1__20260624T003654Z/generation/a0p0_shift30`) — so it is
reused, not re-run. (L2 is **not** the anchor: it is `shift10 + remask-ON`, which would
confound "remask off" with "shift 10→30".)

| arm | remask | `--config` | `--controller-config` | run |
|---|---|---|---|---|
| **A2-ON** (matched control) | on | Stage-1 sampler | `..._rankface_a0p0_shift30.yaml` | **EXISTING** — reuse |
| **A2-OFF** (RF clean-no-remask) | off | `c1_constant_clean_no_remask.yaml` | `..._rankface_a0p0_shift30.yaml` | NEW |
| **A3-OFF** (DPLM substrate) | off | `c1_constant_clean_no_remask.yaml` | (none) | NEW |
| A3-ON (substrate control) | on | constant_one + no controller + remask on | (none) | NEW (small) or read DPLM-native scTM (RAR 0013, ≈0.819) |

- **A2-OFF vs A2-ON** is the clean test: identical controller (`a0p0_shift30`) and identical
  sampler except `fraction_scale` (1.0 → 0.0), so the only change is the remask. Reusing
  the `a0p0_shift30` controller is exact here — under no-remask `alpha_struct` is inert
  (the rank face is only consulted to pick which committed positions to remask; with 0
  remasks it is never used), and Stage 1 showed it invariant under remask-ON, so `a0p0`
  ≡ the L2-base rank face for this probe. `shift30` is the right write strength: under
  no-remask, off-native tokens persist, so we want them written. The test: do D2's
  off-native corrections, allowed to persist, break structure / move immune?
- **A3-OFF vs A3-ON** (no controller either side) discriminates "remask is a general DPLM
  structure stabilizer" from "remask is an RF-specific bottleneck".

## Pre-registered read table

- **RF (arm 2) no-remask: structure drops, immune drops** → remask is the main boundary;
  sampler surgery has research room (Direction 1 lives).
- **RF no-remask: structure drops, immune flat** → the actuator has authority but the
  local signal is wrong → terminal-probe ranker becomes meaningful.
- **RF no-remask: structure still does NOT drop** → the boundary is mainly the
  frozen-backbone posterior, not the remask schedule → the logit RF line closes out.
- **DPLM (arm 3) no-remask itself drops structure** → remask is a general DPLM structure
  stabilizer, not RF-specific; any arm-2 structure drop is then NOT controller-attributable.

## Required metrics (every arm)

scTM p5/p10/frac<0.5; commit proxy / canonical-final persistence; **later-remask fraction
(must be ~0 for the no-remask arms — sanity-check the knob fired)**; terminal global_risk
paired **A2-OFF vs A2-ON (`a0p0_shift30`)** and vs B1 (Wilcoxon); plus the revisit/churn family (revisit_churn_rate,
same-position remask count, D2-fixed-position reopened rate) — already computed in
`scripts/run_if_phase_c1.py`.

## After this probe

This is the close-out architectural test. Whatever the verdict, the `alpha/nu/lambda/shift`
line stays closed. SC-GR / the probe are retained as a **burden sensor / boundary sensor**,
not as a logit actuator.
