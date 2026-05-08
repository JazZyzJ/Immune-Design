# DPLM Inverse Folding Improvement Plan

> Scope: standalone improvement of DPLM's inverse folding capability.
> This plan is separate from the immunogenicity/reference-flow contribution in `doc/Reference_Flow_Derivation.md`.

## Goal

Improve the frozen DPLM v1 inverse-folding baseline by importing the strongest MapDiff ideas and then testing whether DPLM benefits more from a better denoising/refinement process or from replacing its current GVP structure encoder.

The working hypothesis is:

> DPLM already has a strong sequence prior. Its inverse-folding weakness is mostly the structure-conditioning path: frozen GVP features are injected through a single late structural adapter, while recent IPF models such as MapDiff spend most of their capacity on geometry-conditioned denoising and uncertainty-aware refinement.

## Current Baseline

Current local DPLM-IF implementation:

- Model: `inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`
- Adapter: `inverse_folding/dplm/src/byprot/models/dplm/modules/dplm_adapter.py`
- Structure encoder wrapper: `inverse_folding/dplm/src/byprot/models/dplm/modules/gvp_transformer_encoder.py`
- Config: `inverse_folding/dplm/configs/experiment/dplm/cond_dplm_650m.yaml`

Key baseline facts:

- The encoder is ESM-IF1 GVP-Transformer via `esm.pretrained.esm_if1_gvp4_t16_142M_UR50()`.
- The encoder is frozen by default.
- DPLM is loaded from `airkingbd/dplm_650m`.
- The original DPLM last layer is replaced by one `AdapterLayer`.
- Only parameters whose name contains `adapter` are trainable.
- The adapter performs cross-attention from DPLM hidden states to structure encoder features.
- Training uses `output_encoder_logits: true`, so the encoder also provides a draft sequence path.
- Inference currently uses `max_iter: 10`, `sampling_strategy: argmax`, and `use_draft_seq: true`.

Baseline result already established in `PROGRESS.md`:

- CATH 4.3 test, 1,864 proteins, `T=1.0`, `max_iter=10`, argmax.
- Recovery mean 0.5146, median 0.55.
- scTM mean 0.8114, median 0.8702.
- scTM > 0.5: 91.95%.
- scTM > 0.8: 68.24%.
- pLDDT mean 76.62.

The user has already handled Phase 0/1 style baseline and sampling checks. This document starts from the next optimization phases.

## MapDiff Source Grounding

Primary paper:

- Bai et al., "Mask-prior-guided denoising diffusion improves inverse protein folding", Nature Machine Intelligence 7, 876-888, published 2025-06-16, DOI `10.1038/s42256-025-01042-6`.
- Zotero item: `Q3VQM9CT`.
- Local PDF read: `/Users/jerry/Zotero/storage/FK537NN7/Bai 等 - 2025 - Mask-prior-guided denoising diffusion improves inverse protein folding.pdf`.

Official code cloned for inspection:

- `MapDiff/`
- GitHub source declared by paper: `https://github.com/peizhenbai/MapDiff`.

MapDiff paper facts used by this plan:

- MapDiff formulates IPF as a discrete denoising diffusion process conditioned on a given 3D backbone.
- The denoising network has three operations per denoising step:
  - an EGNN structure-based sequence predictor,
  - entropy-based low-confidence residue masking with a mask-ratio adaptor,
  - a pretrained IPA masked sequence designer that refines masked residues using sequence context and backbone geometry.
- The diffusion process uses discrete transition matrices. The paper discusses uniform and marginal priors.
- DDIM is used to skip denoising steps during inference.
- Monte-Carlo dropout is used at inference by keeping dropout active and aggregating multiple stochastic predictions.
- MapDiff pretrains the IPA masked designer using a BERT-style masking objective on the same CATH training data, not external data.
- Paper Methods report: CATH 4.2 split 18,024 train / 608 validation / 1,120 test; CATH 4.3 split 16,630 train / 1,516 validation / 1,864 test.
- Paper Methods report: EGNN predictor has 6 global-aware EGCL layers with 128 hidden dimensions; IPA masked designer has 6 IPA layers, 128 hidden dimensions, 4 attention heads; dropout 0.1 in EGCL and 0.2 in IPA (per `MapDiff/conf/model/egnn.yaml` `drop_out: 0.1` / `ipa_drop_out: 0.2`); 500 diffusion timesteps; DDIM skip steps 100; Monte-Carlo forward passes 50; mask-ratio minimum 0.4 and deviation 0.2.
- Paper Table 1 reports MapDiff CATH median recovery:
  - CATH 4.2, uniform prior: full 61.03%; marginal prior: full 60.93%.
  - CATH 4.3, uniform prior: full 60.86%; marginal prior: full 60.68%.

MapDiff code facts used by this plan:

- Diffusion/refinement wrapper: `MapDiff/model/prior_diff.py`.
- EGNN predictor: `MapDiff/model/egnn_pytorch/egnn_net.py`.
- IPA masked designer: `MapDiff/model/ipa/ipa_net.py`.
- Diffusion training entry: `MapDiff/main.py`.
- IPA pretraining entry: `MapDiff/mask_ipa_pretrain.py`.
- Default diffusion config: `MapDiff/conf/diff_config.yaml`.
- Default IPA pretraining config: `MapDiff/conf/mask_pretrain.yaml`.
- EGNN config: `MapDiff/conf/model/egnn.yaml`.
- Diffusion config: `MapDiff/conf/diffusion/discrete_default.yaml`.
- Mask prior config: `MapDiff/conf/mask_prior/default.yaml`.

Important implementation details from code:

- `Prior_Diff.forward()` trains both base EGNN loss and prior-mask loss; total training loss in `trainer/trainer.py` is `base_loss + mask_loss`.
- `Prior_Diff.sample_p_zs_given_zt()` computes EGNN base logits, entropy, an entropy mask, IPA prior logits, then fuses base and prior logits by entropy-weighted logit fusion.
- `sin_mask_ratio_adapter()` computes mask ratio as `center + sin(beta_t_bar * pi / 2) * max_deviation`.
- Code default `min_mask_ratio=0.4`, `dev_mask_ratio=0.2`.
- Code default `noise_type: marginal`, but the paper reports both uniform and marginal prior variants.
- Code default `timesteps: 500`, `ddim_steps: 100`, `ensemble_num: 50`.
- `trainer/trainer.py` enables dropout before validation/test sampling and repeats `mc_ddim_sample()` `ensemble_num` times, then averages logits.
- `IPANetPredictor` builds local rigid frames from backbone atoms via `Rigid.from_3_points(...)`.
- `NodeMaskEncoder` adds amino-acid input, backbone dihedral features, mask/pad embeddings, and sinusoidal position encoding.
- `EdgePairEncoder` uses pairwise distances over 4 backbone atoms, RBF embedding, and relative position embedding.

Do not overclaim:

- MapDiff is not a DPLM method and does not prove that DPLM will improve if the same modules are bolted on.
- MapDiff's reported Table 1 numbers are not directly comparable to our DPLM baseline unless we reproduce identical splits, preprocessing, evaluation, and folding metrics.
- MapDiff's foldability analysis uses AlphaFold2 pTM with ColabFold/MMseqs2 MSA in the paper. Our existing structural validation uses ESMFold/scTM unless explicitly changed.
- The paper supports entropy-guided refinement and MC dropout for IPF generally; it does not specifically support immunogenicity-aware guidance.

## Phase A: MapDiff-Style DPLM Refiner

### Scientific Question

Can DPLM-IF be improved by importing MapDiff's uncertainty-aware refinement without replacing DPLM as the sequence prior?

This phase treats the current DPLM adapter output as the base denoiser, analogous to MapDiff's EGNN base predictor, and adds a MapDiff-style refinement path for high-entropy positions.

### Rationale

MapDiff's strongest directly portable idea is not simply "use EGNN"; it is the denoising control loop:

1. produce a full-sequence base prediction,
2. identify uncertain positions by entropy,
3. mask/refine those positions with a geometry-aware masked designer,
4. fuse base and prior/refiner logits by uncertainty,
5. use iterative discrete denoising with DDIM-style skipping and MC dropout aggregation.

For DPLM, this is attractive because it can preserve the pretrained DPLM sequence prior while adding a targeted geometry-aware correction layer at the positions where DPLM is least confident.

### Proposed Design

Implement a DPLM-compatible refiner with the following components:

- Base logits: current DPLM-IF decoder logits at each denoising step.
- Uncertainty: per-position entropy over amino-acid logits, excluding special tokens.
- Mask-ratio adaptor: MapDiff-style sine adaptor with default center 0.4 and max deviation 0.2.
- Refiner input sequence: base argmax sequence with high-entropy positions marked as mask.
- Refiner geometry input: cleaned backbone coordinates already used by DPLM.
- Refiner model: port/adapt `MapDiff/model/ipa/ipa_net.py` as a standalone IPA masked designer.
- Fusion: entropy-weighted fusion analogous to MapDiff's `fuse_logits_by_log_probs`.
- MC dropout: enable dropout in the refiner path at inference and average logits across `K` stochastic passes.

The first implementation should avoid retraining DPLM itself. Train only the refiner on CATH 4.3 using the same CATH training split as Module K.

### Implementation Notes

Recommended new package boundary:

- `inverse_folding/dplm_refiner/`
  - `entropy.py`: entropy, mask selection, sine mask-ratio adaptor.
  - `ipa_refiner.py`: DPLM-compatible IPA masked designer, derived from MapDiff's `IPANetPredictor` design.
  - `fusion.py`: entropy-weighted base/refiner logit fusion.
  - `training.py`: CATH refiner training loop or Lightning/Hydra wrapper.
  - `sampler.py`: DPLM generation loop with optional refiner pass.

Reuse existing assets:

- DPLM model loading and generation from `inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`.
- CATH data paths and DPLM training launcher conventions from Module K.
- Evaluation schema from `scripts/validate_if_baseline.py` and Phase C evaluator where possible.

MapDiff source files to inspect during implementation:

- `MapDiff/model/prior_diff.py`
- `MapDiff/model/ipa/ipa_net.py`
- `MapDiff/model/ipa/ipa_attn.py`
- `MapDiff/model/ipa/ipa_utils.py`
- `MapDiff/utils.py`
- `MapDiff/mask_ipa_pretrain.py`
- `MapDiff/trainer/mask_ipa_trainer.py`

### Ablations

Minimum ablation set:

- Current DPLM baseline, no refiner.
- DPLM + entropy mask selection only, no IPA refiner, to isolate schedule effects.
- DPLM + IPA refiner, deterministic single pass.
- DPLM + IPA refiner + MC dropout, `K in {5, 10, 50}`.
- Mask-ratio center/deviation sweep around MapDiff defaults:
  - center 0.2 / 0.4 / 0.6.
  - deviation 0.1 / 0.2.
- Fusion sweep:
  - base logits only,
  - refiner logits only at masked positions,
  - entropy-weighted fusion.

Primary metrics:

- Recovery mean/median on CATH 4.3.
- Perplexity if logits are retained.
- scTM mean/median and scTM pass rates.
- pLDDT and scRMSD.
- Recovery stratified by DSSP class if available, especially coil/bend/turn regions.
- Recovery stratified by baseline DPLM entropy quantile.

Expected positive signal:

- Recovery improves most at high-entropy positions.
- scTM does not degrade relative to baseline.
- Improvement is enriched in flexible/less-ordered regions, mirroring the MapDiff motivation.

Failure interpretation:

- If entropy-refinement improves recovery but hurts scTM, the refiner is overfitting native residue identity rather than preserving foldability.
- If entropy-refinement has no effect, the bottleneck is likely upstream structure encoding or DPLM adapter integration, not local refinement.

## Phase B: Replace Or Augment The Structure Encoder

### Scientific Question

Is the frozen ESM-IF1 GVP encoder plus one late adapter the limiting factor for DPLM-IF?

### Rationale

DPLM's current structure path is narrow:

- The encoder is frozen.
- It produces per-residue features.
- Structure enters DPLM through one last-layer cross-attention adapter.
- There is no explicit pair representation in the DPLM adapter path.

MapDiff's result suggests that stronger geometric modeling matters. Its EGNN uses coordinate-aware message passing with global-aware updates, and its IPA refiner uses rigid frames plus pairwise distance representations.

### Proposed Design

Compare three encoder variants while keeping DPLM fixed as much as possible:

1. **GVP baseline**: existing ESM-IF1 GVP encoder.
2. **GVP + MapDiff geometry sidecar**: keep GVP features, add a lightweight EGNN/IPA-derived geometry feature stream, project to adapter dimension, and concatenate or sum before adapter cross-attention.
3. **MapDiff-style encoder replacement**: replace GVP features with a trained EGNN encoder/draft predictor derived from `MapDiff/model/egnn_pytorch/egnn_net.py`.

The sidecar is the safest first implementation because it preserves the current working DPLM path and tests whether extra geometry helps without fully discarding GVP.

### Implementation Notes

Candidate model interfaces:

- New encoder wrapper should return the same keys expected by `DPLMInvFold.forward_encoder()`:
  - `feats`: `[batch, length, d_model]`.
  - optional `logits`: draft sequence logits when `output_logits=True`.
  - `coord_mask` or compatible attention mask.
- Keep `encoder_attention_mask` semantics identical to existing DPLM code.
- If adding sidecar features, introduce a projection layer to keep adapter input dimension stable.

MapDiff source files to inspect during implementation:

- `MapDiff/model/egnn_pytorch/egnn_net.py`
- `MapDiff/model/egnn_pytorch/egnn_pyg.py`
- `MapDiff/model/egnn_pytorch/utils.py`
- `MapDiff/dataloader/collator.py`
- `MapDiff/dataloader/utils.py`

Current DPLM files likely affected:

- `inverse_folding/dplm/src/byprot/models/dplm/modules/gvp_transformer_encoder.py`
- `inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`
- `inverse_folding/dplm/src/byprot/models/dplm/modules/dplm_adapter.py`
- `inverse_folding/dplm/configs/experiment/dplm/cond_dplm_650m.yaml`

### Ablations

Minimum ablation set:

- GVP baseline.
- GVP + sidecar, train sidecar + adapter only.
- EGNN replacement, train encoder + adapter only.
- Optional: unfreeze last N DPLM layers only if the above saturates.

Metrics:

- Same CATH recovery/scTM metrics as baseline.
- Draft sequence recovery from the encoder path.
- Downstream DPLM recovery after iterative denoising.
- Encoder-only vs final DPLM improvement gap.

Expected positive signal:

- Encoder draft recovery improves.
- DPLM final recovery improves beyond draft recovery, showing DPLM is using the better condition rather than merely copying the draft.
- scTM pass rates remain at least baseline-level.

Failure interpretation:

- If draft improves but final DPLM does not, adapter integration is the bottleneck.
- If neither draft nor final improves, CATH preprocessing or graph feature construction is likely mismatched.
- If final recovery improves but scTM drops, the model is improving native identity without preserving robust foldability.

## Phase C: Open Slot

This phase is intentionally left as a decision slot until Phase A and Phase B identify the dominant bottleneck.

No technical direction is assigned yet. Do not start Phase C until Phase A/B results show whether the limiting factor is refinement, encoder geometry, or adapter depth.

## Evaluation Policy

Primary development benchmark:

- CATH 4.3 test set, because it matches the current Module K DPLM baseline and MapDiff's reported split size.

Secondary benchmark:

- Project IF-ready test sets and fast subsets from `PROGRESS.md`, only after CATH behavior is understood.

Metrics:

- Recovery mean and median.
- Perplexity where logits are available.
- scTM mean/median and pass rates at 0.5 and 0.8.
- pLDDT and scRMSD.
- Mutation count and sequence diversity for sampled variants.
- Stratified recovery by entropy quantile and structural region when available.

Fairness constraints:

- Do not compare MapDiff paper numbers directly to local DPLM numbers as a headline unless the preprocessing and evaluation are reproduced.
- Always report whether structural validation used ESMFold or AlphaFold2/ColabFold.
- Keep CATH test structures separate from training.
- For MapDiff-derived modules, use the same CATH training set as the DPLM adapter unless deliberately testing external-data effects.

## Near-Term Execution Order

1. Freeze a reproducible CATH baseline manifest from the existing DPLM checkpoint.
2. Port MapDiff's entropy/mask/fusion utilities into a small tested local module.
3. Port/adapt IPA masked designer as a DPLM-compatible refiner.
4. Train IPA refiner on CATH 4.3 using DPLM base predictions as input context.
5. Evaluate Phase A ablations on CATH.
6. If Phase A positive, integrate the refiner into project IF-ready evaluation.
7. If Phase A weak or limited to recovery-only gains, start Phase B sidecar encoder.
8. Fill Phase C only after Phase A/B readout.

## Completion Criteria

Phase A is considered successful if:

- recovery median improves over the Module K baseline,
- scTM mean and scTM > 0.8 do not materially degrade,
- improvement is concentrated at high-entropy positions or flexible regions.

Phase B is considered successful if:

- GVP + sidecar or EGNN replacement improves final DPLM recovery,
- final DPLM improvement exceeds encoder draft improvement alone,
- foldability remains at least baseline-level.

The overall improvement program is successful if it produces a DPLM-IF variant that improves CATH recovery without losing the structural-quality profile that made the current baseline usable.
