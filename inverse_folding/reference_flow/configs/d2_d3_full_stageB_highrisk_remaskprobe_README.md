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

## Arms (100 worst / n2, same cohort & seed as L2)

| # | arm | `--config` (CONFIG_PATH) | `--controller-config` |
|---|---|---|---|
| 1 | L2 anchor | (existing run) | (existing run) — reuse, do not re-run |
| 2 | RF clean-no-remask | `c1_constant_clean_no_remask.yaml` | `d2_d3_full_stageB_highrisk_remaskprobe_shift30.yaml` |
| 3 | DPLM substrate clean-no-remask | `c1_constant_clean_no_remask.yaml` | (none) |

- Arm 2 = cap20 + L2 + shift30 + no-remask. `alpha_struct` is **inert** under no-remask
  (the rank face is only consulted to choose which committed positions to remask; with
  0 remasks it is never used), so it stays at the L2 base 0.3. The test: do D2's
  off-native corrections, allowed to persist (no remask), break structure / move immune?
- Arm 3 = the same sampler substrate with **no controller** — the DPLM no-remask control
  that discriminates "remask is a general DPLM structure stabilizer" from "remask is an
  RF-specific bottleneck".

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
paired vs L2 and B1 (Wilcoxon); plus the revisit/churn family (revisit_churn_rate,
same-position remask count, D2-fixed-position reopened rate) — already computed in
`scripts/run_if_phase_c1.py`.

## After this probe

This is the close-out architectural test. Whatever the verdict, the `alpha/nu/lambda/shift`
line stays closed. SC-GR / the probe are retained as a **burden sensor / boundary sensor**,
not as a logit actuator.
