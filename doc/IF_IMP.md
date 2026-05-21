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
- `EGNN_NET.forward()` consumes PyG graph fields `x`, `extra_x`, `pos`, `edge_index`, `edge_attr`, `ss`, and `batch`, applies coordinate-aware graph message passing, and returns per-residue 20-way amino-acid logits.
- `MapDiff/conf/model/egnn.yaml` enables `update_edge: True`, `update_coors: True`, `update_global: True`, `norm_coors: True`, 6 EGNN layers, 128 hidden dimensions, and 128 embedding dimensions.
- MapDiff graph construction uses CA-neighbor topology, sequence-distance features, inter-residue distance/contact features, local-frame orientation features, secondary-structure features, and `mu_r_norm` neighborhood-direction statistics.
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

## Current Readout And Pivot Rule

The first implemented MapDiff-inspired add-ons should be treated as diagnostic rather than successful final mechanisms.

Observed CATH recovery readout from the current local runs:

- Baseline: 0.5290.
- Sidecar: 0.5307.
- Sidecar scale sweep best near `sidecar_scale=0.5`: 0.5308.
- Entropy-aware refiner: approximately baseline-level at 0.5290.
- Refiner without DPLM-entropy alignment: 0.5247.
- Weak residual refiner sweep: effectively no-op at the final sequence level.

Interpretation:

- The sidecar is not a no-op, but its improvement is too small and too shallow to support a strong mechanism claim.
- Entropy alignment prevents the refiner from becoming harmful, but the refiner still does not beat the DPLM baseline.
- Further broad sweeps over sidecar scale, mask ratio, fusion weight, or apply-step windows are lower leverage than changing the structure-conditioning path itself.

Pivot rule:

- Keep sidecar/refiner as context and optional ablation arms.
- Make the next main mechanism a direct replacement of the frozen GVP structure encoder and the shallow final-layer adapter path.

## Phase B: GeoEGNN-IPA Structure Encoder Replacement

### Scientific Question

Is the frozen ESM-IF1 GVP encoder plus one late adapter the limiting factor for DPLM-IF, and can a MapDiff-inspired geometry encoder provide a stronger structure memory for the frozen DPLM denoising prior?

### Rationale

DPLM's current structure path is narrow:

- The encoder is frozen.
- It produces per-residue features.
- Structure enters DPLM through one last-layer cross-attention adapter.
- There is no explicit pair representation in the DPLM adapter path.

MapDiff's result suggests that stronger geometric modeling matters. Its EGNN uses coordinate-aware message passing with global-aware updates, and its IPA refiner uses rigid frames plus pairwise distance representations.

The replacement should not copy MapDiff's full diffusion loop. MapDiff's EGNN is the denoiser body and returns 20-way amino-acid logits. DPLM already has a pretrained sequence prior and a mask-denoising decoder. The portable idea is therefore:

- use EGNN for local topology/contact message passing,
- use IPA for dense rigid-frame geometric refinement,
- expose the result as DPLM-compatible structure memory, not as the final sequence predictor.

### Proposed Design

Replace the GVP feature generator with a GeoEGNN-IPA encoder and deepen the adapter path:

```text
backbone coords
  -> kNN graph and local-frame structure features
  -> EGNN-lite message passing
  -> dense residue hidden [B, L, H]
  -> IPA refinement with backbone frames and pair geometry
  -> projection + LayerNorm to [B, L, 512]
  -> late-layer gated DPLM adapters
```

Role separation:

- EGNN handles graph/topology/contact message passing over sparse CA-neighbor edges.
- IPA maps the EGNN hidden states back into a dense rigid-frame geometry representation.
- DPLM remains the frozen sequence prior and denoising decoder.
- The adapter path injects the structure memory into DPLM hidden states; it is no longer only a one-layer final correction.

This is a replacement, not a residual sidecar. GVP can remain only as the baseline and optional compatibility fallback.

### Encoder Inputs

The first implementation should use backbone-safe features only.

Required node features:

- backbone dihedral sin/cos features derived from `N/CA/C/O`;
- coordinate-validity mask;
- relative residue index or positional encoding;
- optional `mu_r_norm` neighborhood-direction statistics if easy to compute from the same kNN graph.

Required edge features:

- CA kNN or cutoff graph;
- clipped sequence-distance one-hot or embedding;
- CA-distance RBF/contact feature;
- local-frame orientation features derived from backbone frames.

Intentionally excluded from the first implementation:

- true amino-acid identity as encoder input;
- SASA;
- B-factor;
- DSSP or external secondary-structure labels.

Reason: the first replacement should test whether stronger geometry conditioning improves DPLM. Extra annotations can be added later only if the geometry-only arm establishes a meaningful signal.

### Coordinate Update Policy

`update_coors` is a required ablation knob, not a conceptual prohibition.

MapDiff enables coordinate update inside EGNN even though it is also an inverse-folding model. This means coordinate update should be interpreted as an internal latent geometric state, not as changing the target backbone.

Supported modes:

- `update_coors=false`: conservative graph feature passing with fixed coordinates.
- `update_coors=true`: EGNN updates a latent coordinate channel during message passing.

Interface rule:

- The final task condition remains the original backbone.
- IPA should always keep the original `N/CA/C` rigid frames as the anchor.
- If `update_coors=true`, the updated CA coordinates may be used as an additional pair-bias source, for example original pair RBF plus updated-CA pair RBF, but should not replace the original backbone frame.

If `update_coors=true` improves recovery, the interpretation is that latent coordinate relaxation helps construct a sequence-compatible geometric representation. It is not evidence that the model is solving a different structure-generation task.

### IPA Refinement And Output Contract

The IPA block should consume:

- dense EGNN residue hidden states `[B, L, H]`;
- original rigid frames from `N/CA/C`;
- pair features from original atom distances, relative position, and optional updated-coordinate pair bias.

The output contract must match the current DPLM encoder contract:

- `encoder_out["feats"]`: `[B, L, 512]`;
- `encoder_attention_mask`: valid residue mask compatible with DPLM adapter cross-attention;
- optional encoder draft logits when `output_logits=True`.

If draft logits are required for `use_draft_seq=True`, they should come from a small auxiliary AA head over the encoder hidden state. This head is a training and compatibility head; it is not the final sampler.

### Adapter Usage

The current final-layer-only adapter is likely too shallow for a new geometry encoder.

Replace or extend it with late-layer gated adapters:

- default target: last 4 DPLM layers;
- keep the DPLM backbone frozen;
- train only the GeoEGNN-IPA encoder, projection layers, gates, adapter cross-attention, and adapter FFN parameters;
- initialize gates at zero or a small value so the initial model is close to the frozen DPLM baseline;
- preserve the original one-layer adapter as an ablation.

This keeps the core claim clean: same DPLM prior and decoder, stronger structure conditioning and deeper structure injection.

### Training Objective

Main loss:

- original DPLM inverse-folding CE under the existing masked-diffusion training path.

Auxiliary loss:

- add a low-weight amino-acid prediction head from EGNN hidden or post-IPA hidden;
- train it with native AA labels on valid residues;
- do not feed native AA identity into the encoder input;
- drop the auxiliary head at inference except when draft logits are explicitly requested.

Default combined objective:

```text
loss = loss_dplm_if + lambda_aux * loss_aux_aa
```

Recommended starting values:

- `lambda_aux=0.05` and `lambda_aux=0.10`;
- EGNN dropout 0.1;
- IPA dropout 0.2;
- 6 EGNN layers, hidden dimension 128;
- 6 IPA layers, hidden dimension 128, 4 attention heads;
- last-layer adapter and last-4-layer gated adapter comparison.

### Implementation Notes

Candidate model interfaces:

- New encoder wrapper should return the same keys expected by `DPLMInvFold.forward_encoder()`:
  - `feats`: `[batch, length, d_model]`.
  - optional `logits`: draft sequence logits when `output_logits=True`.
  - `coord_mask` or compatible attention mask.
- Keep `encoder_attention_mask` semantics identical to existing DPLM code.
- The graph builder should be derived from the DPLM batch `coords` at runtime or in a cache keyed by protein/sequence, not from MapDiff's pre-generated `.pt` graph corpus.
- The encoder must not require DSSP, SASA, B-factor, or external structure annotation for the first replacement arm.
- The implementation should expose `update_coors`, `adapter_layers`, `lambda_aux`, and feature-set selection as config fields.

MapDiff source files to inspect during implementation:

- `MapDiff/model/egnn_pytorch/egnn_net.py`
- `MapDiff/model/egnn_pytorch/egnn_pyg.py`
- `MapDiff/model/egnn_pytorch/utils.py`
- `MapDiff/dataloader/cath_dataset.py`
- `MapDiff/dataloader/collator.py`
- `MapDiff/dataloader/utils.py`
- `MapDiff/model/ipa/ipa_net.py`
- `MapDiff/model/ipa/ipa_attn.py`
- `MapDiff/model/ipa/rigid_utils.py`

Current DPLM files likely affected:

- `inverse_folding/dplm/src/byprot/models/dplm/modules/gvp_transformer_encoder.py`
- `inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`
- `inverse_folding/dplm/src/byprot/models/dplm/modules/dplm_adapter.py`
- `inverse_folding/dplm/configs/experiment/dplm/cond_dplm_650m.yaml`

### Ablations

Minimum ablation set:

- GVP baseline.
- GeoEGNN-IPA replacement with `update_coors=false`.
- GeoEGNN-IPA replacement with `update_coors=true` and original-frame IPA anchoring.
- Last-layer adapter only.
- Last-4-layer gated adapters.
- Auxiliary AA head off.
- Auxiliary AA head on with `lambda_aux in {0.05, 0.10}`.
- Minimal geometry features only.
- Minimal geometry plus local-frame orientation and `mu_r_norm`.

Metrics:

- Same CATH recovery/scTM metrics as baseline.
- Draft sequence recovery from the encoder path.
- Downstream DPLM recovery after iterative denoising.
- Encoder-only vs final DPLM improvement gap.
- Auxiliary head accuracy, reported separately from final DPLM recovery.
- Adapter gate magnitudes and adapter-attention entropy as diagnostics.

Expected positive signal:

- DPLM final recovery improves by at least 2 absolute percentage points over the current CATH baseline before treating the replacement as mechanistically meaningful.
- A 3-5 point recovery improvement is the target range for a convincing paper narrative.
- Final DPLM improvement is not explained only by draft copying.
- Last-4-layer gated adapters outperform the last-layer-only adapter if adapter depth is the current bottleneck.
- scTM pass rates remain at least baseline-level.

Failure interpretation:

- If auxiliary or draft recovery improves but final DPLM does not, adapter integration is still the bottleneck.
- If neither draft nor final improves, CATH preprocessing or graph feature construction is likely mismatched.
- If final recovery improves but scTM drops, the model is improving native identity without preserving robust foldability.
- If `update_coors=true` helps, latent coordinate relaxation is useful for structure conditioning.
- If `update_coors=true` hurts, keep fixed-coordinate EGNN or restrict updated-coordinate usage to a weak pair-bias term.

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

1. Preserve the current CATH baseline and the completed sidecar/refiner readouts as context arms.
2. Implement a DPLM-compatible GeoEGNN-IPA encoder that returns `encoder_out["feats"]` with shape `[B, L, 512]`.
3. Implement runtime or cached kNN graph construction from DPLM `coords`, with no dependency on DSSP, SASA, or B-factor.
4. Add configurable late-layer gated adapters while keeping the DPLM backbone frozen.
5. Add the auxiliary AA head and combined loss with configurable `lambda_aux`.
6. Train the primary GeoEGNN-IPA replacement on CATH 4.3.
7. Run the minimum ablations: `update_coors`, adapter depth, auxiliary head, and feature-set selection.
8. Evaluate the best CATH arm with the same recovery/scTM/pLDDT/scRMSD schema as the Module K baseline.
9. Move to project IF-ready immune-design evaluation only after the CATH arm exceeds the replacement success threshold.
10. Fill Phase C only after the encoder replacement readout clarifies whether any further sampling-time control is still necessary.

## Completion Criteria

Phase A is considered successful if:

- recovery median improves over the Module K baseline,
- scTM mean and scTM > 0.8 do not materially degrade,
- improvement is concentrated at high-entropy positions or flexible regions.

Phase B is considered successful if:

- GeoEGNN-IPA replacement improves final DPLM recovery by at least 2 absolute percentage points over the current CATH baseline,
- final DPLM improvement is not merely draft-head improvement copied through `use_draft_seq`,
- late-layer gated adapters show a measurable benefit over the last-layer-only adapter or the last-layer-only result already exceeds the recovery threshold,
- foldability remains at least baseline-level.

The overall improvement program is successful if it produces a DPLM-IF variant that improves CATH recovery without losing the structural-quality profile that made the current baseline usable.
