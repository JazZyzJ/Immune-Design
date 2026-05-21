# GeoEGNN-IPA DPLM Encoder Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DPLM-IF's frozen GVP structure encoder with a MapDiff-inspired GeoEGNN-IPA encoder that returns DPLM-compatible `encoder_out["feats"]`, while keeping DPLM as the frozen sequence prior and denoising decoder.

**Architecture:** Build a new geometry encoder path: DPLM `[N, CA, C, O]` backbone coordinates -> kNN graph -> EGNN message passing -> dense residue hidden -> IPA rigid-frame refinement -> `[B, L, 512]` structure memory -> late-layer gated DPLM adapters. Do not copy MapDiff's diffusion loop or AA-logit predictor as the final model; use MapDiff EGNN/IPA ideas only to improve DPLM structure conditioning.

**Tech Stack:** Python 3.12, PyTorch, torch-geometric/torch-scatter, DPLM/byprot, existing `inverse_folding.dplm_refiner` IPA utilities, pytest, YAML, SLURM.

---

## Source Anchors

Use these exact source points while implementing. Do not replace them with memory-based assumptions.

- DPLM encoder contract:
  - `inverse_folding/dplm/src/byprot/models/dplm/modules/gvp_transformer_encoder.py`
  - `GVPTransformerEncoderWrapper.forward()` returns `encoder_out["feats"] = encoder_out["encoder_out"][0].transpose(0, 1)` and, when `output_logits=True`, returns `(logits, encoder_out)`.
- DPLM training path:
  - `inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`
  - `DPLMInvFold.forward()` repeats encoder features for coupled diffusion loss, but currently calls `encoder_out["feats"].repeat(...).detach()`. GeoEGNN-IPA training must explicitly remove or gate this detach for trainable encoders.
- DPLM generation path:
  - `DPLMInvFold.forward_encoder()` supplies `encoder_out["init_pred"]` and `encoder_out["logits"]` when `use_draft_seq=True`.
  - `DPLMInvFold.forward_encoder()` currently adds `encoder_out["coord_mask"]` only on the `use_draft_seq=False` branch. The new encoder path must make `coord_mask` available in both draft and non-draft modes.
  - `DPLMInvFold.generate()` calls `forward_encoder()` once before iterative decoding.
- Adapter path:
  - `inverse_folding/dplm/src/byprot/models/dplm/modules/dplm_adapter.py`
  - `DPLMWithConditionalAdatper.from_pretrained()` currently replaces only the last ESM layer with `AdapterLayer`.
  - `DPLMWithConditionalAdatper.forward()` consumes only `encoder_out["feats"]` and `encoder_attention_mask`.
- Existing reusable geometry/IPA utilities:
  - `inverse_folding/dplm_refiner/geometry.py::dplm_coords_to_ipa_positions()`
  - `inverse_folding/dplm_refiner/ipa/refiner.py::_IPATrunk`, `_EdgePairEncoder`, and related rigid-frame utilities.
- Existing sidecar code to avoid copying semantically:
  - `inverse_folding/dplm_refiner/encoder_wrapper.py::SidecarAttachedEncoder` is residual GVP augmentation, not encoder replacement.
  - `scripts/train_if_imp_sidecar.py` temporarily wraps GVP per batch and trains a residual sidecar under a full-mask training schedule; this should not be the new encoder training pattern.
- MapDiff implementation anchors:
- `MapDiff/model/egnn_pytorch/egnn_pyg.py::EGNN_Sparse`
  - `MapDiff/model/egnn_pytorch/egnn_net.py::EGNN_NET`
  - `MapDiff/dataloader/cath_dataset.py` graph construction and local-frame edge features.
  - `MapDiff/model/ipa/ipa_net.py::IPANetPredictor`

## Implementation Decisions

- EGNN implementation decision: preserve the PyG `MessagePassing` style for the first implementation. Do not hand-write a separate scatter kernel in v1.
- Dependency decision: `torch_geometric` and `torch_scatter` are required runtime dependencies for the GeoEGNN path. Root `requirements.txt` already pins them; `inverse_folding/dplm/requirements.txt` has `torch_scatter` commented and must be checked/updated if that requirements file is used on the target environment.
- Graph representation decision: Task E1 returns a PyG `Data`/`Batch`-compatible object plus explicit metadata for `[N_res, H] -> [B, L_token, H]` scatter-back.
- EGNN output decision: local EGNN layers may internally operate on MapDiff-style `cat([coords, hidden], dim=-1)`, but every public GeoEGNN encoder boundary must split this tensor into separate `hidden` and optional `updated_coords` before passing hidden states to IPA.
- Node feature decision: include deterministic residue positional encoding now, before any cluster-side GeoEGNN-IPA checkpoint exists. Default `posenc_dim=16`; with the current base node features `6 + 1 + 5`, this makes `node_feat_dim=28`.
- Densify decision: use a local fail-fast `densify_to_token_grid()` based on explicit graph metadata and DPLM `residue_mask`. Do not require `torch_geometric.utils.to_dense_batch` as the production path, because DPLM token length includes special/pad positions and empty-protein handling should be explicit.

## Non-Negotiable Constraints

- Default GVP DPLM behavior must remain unchanged when the new encoder config is not selected.
- Production code must not import `MapDiff.*` modules. Copy/adapt minimal EGNN logic with attribution comments.
- The new encoder must return the existing DPLM contract:
  - `encoder_out["feats"]`: `[B, L_token, 512]`
  - `encoder_attention_mask`: `[B, L_token]` bool-compatible mask
  - `coord_mask`: `[B, L_token]`, copied from the DPLM batch or returned by the encoder in both draft and non-draft modes
  - optional `(logits, encoder_out)` when `output_logits=True`
- Do not feed native amino-acid identity, SASA, B-factor, or DSSP/external secondary-structure labels into the first encoder implementation.
- Use `coord_mask & ~special_sym_mask` as the residue mask. Special tokens, pads, and invalid coordinates must not become graph nodes.
- Graph edges must never cross proteins.
- `update_coors` is a required ablation knob. When enabled, updated coordinates are an internal latent channel or optional pair-bias source; original backbone frames remain the IPA anchor.
- DPLM backbone parameters remain frozen. Trainable parameters are the GeoEGNN-IPA encoder, encoder projection/draft head, and configured adapter parameters.
- Main DPLM loss must backpropagate into the encoder when `detach_encoder_feats=false`.
- All cluster paths must be CLI args or environment variables. No Python hardcoded `/scratch/...` paths.
- Extend `scripts/submit_if_imp.slurm`; do not create a duplicate SLURM launcher.
- Register new script modes in `doc/SCRIPTS.md`.
- Append a structured `LOG.md` entry after implementation.

## File Structure

Create:

- `inverse_folding/dplm_refiner/geo_encoder/__init__.py`  
  Public exports for the GeoEGNN-IPA encoder package.
- `inverse_folding/dplm_refiner/geo_encoder/config.py`  
  Dataclasses for graph, EGNN, IPA, output, and training/checkpoint config.
- `inverse_folding/dplm_refiner/geo_encoder/graph.py`  
  Torch/PyG graph builder from DPLM `[B, L, 4, 3]` coordinates and residue masks.
- `inverse_folding/dplm_refiner/geo_encoder/egnn.py`  
  Adapted EGNN sparse layers, with `update_edge`, `update_global`, and `update_coors` knobs.
- `inverse_folding/dplm_refiner/geo_encoder/ipa.py`  
  IPA dense refinement wrapper that consumes EGNN hidden states and existing backbone rigid frames.
- `inverse_folding/dplm_refiner/geo_encoder/encoder.py`  
  `GeoEGNNIPAEncoder` returning `feats`, optional draft logits, and diagnostics.
- `inverse_folding/dplm_refiner/geo_encoder/checkpoint.py`  
  Save/load encoder checkpoints with config and metadata.
- `inverse_folding/dplm/src/byprot/models/dplm/modules/geoegnn_ipa_encoder.py`  
  Thin byprot-registered wrapper: `@register_model("geoegnn_ipa_encoder")`.
- `scripts/train_if_imp_encoder.py`  
  Train GeoEGNN-IPA encoder and configured adapters on CATH batches.

Modify:

- `inverse_folding/dplm/src/byprot/models/dplm/modules/dplm_adapter.py`  
  Add last-N adapter replacement and optional gated adapter branch.
- `inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`  
  Add compatibility config for `detach_encoder_feats`; default must preserve current behavior.
- `inverse_folding/dplm/configs/experiment/dplm/cond_dplm_650m.yaml`  
  Keep GVP default; add a separate GeoEGNN-IPA experiment YAML or documented override.
- `inverse_folding/dplm/requirements.txt`  
  Ensure the DPLM environment path documents or enables `torch_scatter` for the PyG EGNN path if this requirements file is used for installation.
- `scripts/run_if_imp_refiner.py`  
  Add optional `--encoder-checkpoint` / `--encoder-kind` path for generation with the replacement encoder, or create a clearly named `generate_encoder` mode only if reuse becomes unmaintainable.
- `scripts/submit_if_imp.slurm`  
  Add `MODE=train_encoder|generate_encoder|diag_encoder`.
- `doc/SCRIPTS.md`  
  Register `scripts/train_if_imp_encoder.py` and the new `submit_if_imp.slurm` modes.
- `LOG.md`  
  Append one implementation record when the work is complete.

Create tests:

- `tests/inverse_folding/dplm_refiner/test_geo_graph.py`
- `tests/inverse_folding/dplm_refiner/test_geo_egnn.py`
- `tests/inverse_folding/dplm_refiner/test_geo_encoder_contract.py`
- `tests/inverse_folding/dplm_refiner/test_geo_encoder_checkpoint.py`
- `tests/inverse_folding/dplm_refiner/test_dplm_adapter_layers.py`
- `tests/inverse_folding/dplm_refiner/test_dplm_encoder_grad.py`
- `tests/scripts/test_if_imp_encoder_scripts.py`

## Task E0: Adapter And Gradient Contract

**Files:**
- Modify: `inverse_folding/dplm/src/byprot/models/dplm/modules/dplm_adapter.py`
- Modify: `inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`
- Test: `tests/inverse_folding/dplm_refiner/test_dplm_adapter_layers.py`
- Test: `tests/inverse_folding/dplm_refiner/test_dplm_encoder_grad.py`

- [ ] Add `adapter_num_layers: int = 1`, `adapter_gated: bool = False`, and `adapter_gate_init: float = 0.0` to `DPLMWithAdapterConfig`.
- [ ] Replace the single-layer adapter insertion in `DPLMWithConditionalAdatper.from_pretrained()` with a loop over the last `adapter_num_layers` ESM layers.
- [ ] Add an `adapter_gate` parameter to `AdapterLayer` when `adapter_gated=True`; compute `output = residual + gate * adapter_delta`. Preserve current ungated behavior when `adapter_gated=False`.
- [ ] Keep the existing trainable-parameter rule working by naming the gate `adapter_gate`.
- [ ] Add `detach_encoder_feats: bool = True` to `DPLMInvFoldConfig`.
- [ ] In `DPLMInvFold.forward()`, only detach repeated encoder feats when `detach_encoder_feats=True`. Default behavior must remain identical to current GVP training.
- [ ] In `DPLMInvFold.forward_encoder()`, ensure `encoder_out["coord_mask"] = batch["coord_mask"]` is set after both `use_draft_seq` branches unless the encoder has already supplied an identical mask.
- [ ] Test that default config replaces exactly one adapter layer and preserves the ungated path.
- [ ] Test that `adapter_num_layers=4` replaces the last four layers and exposes trainable `adapter_*` parameters only.
- [ ] Test that encoder gradients from the main DPLM CE are absent when `detach_encoder_feats=True` and present when `detach_encoder_feats=False`.
- [ ] Test that the draft/encoder AA loss path (`encoder_loss` in the task step when `output_encoder_logits=True`) also produces gradients on the encoder draft head and shared encoder trunk.

Verification commands:

```bash
PYTHONPATH=inverse_folding/dplm/src pytest \
  tests/inverse_folding/dplm_refiner/test_dplm_adapter_layers.py \
  tests/inverse_folding/dplm_refiner/test_dplm_encoder_grad.py -v
```

## Task E1: Graph Builder

**Files:**
- Create: `inverse_folding/dplm_refiner/geo_encoder/__init__.py`
- Create: `inverse_folding/dplm_refiner/geo_encoder/config.py`
- Create: `inverse_folding/dplm_refiner/geo_encoder/graph.py`
- Test: `tests/inverse_folding/dplm_refiner/test_geo_graph.py`

- [ ] Define graph config fields: `k_neighbors=10`, `cutoff=30.0`, `seq_dist_cut=64`, `dist_bins=16`, `dist_bin_width=0.5`, `include_local_frame=True`, `include_mu_r_norm=True`, `posenc_dim=16`.
- [ ] Build residue mask from `coord_mask & ~special_sym_mask`.
- [ ] Convert DPLM `[N, CA, C, O]` coordinates to IPA positions via `dplm_coords_to_ipa_positions()` and reuse virtual CB construction.
- [ ] Build per-protein CA kNN edges using `torch.cdist`, not SciPy.
- [ ] Add sequence-neighbor fallback edges so every valid residue has at least one edge in length > 1 proteins.
- [ ] Build edge features: clipped sequence distance embedding, CA distance RBF/contact bit, local-frame orientation features.
- [ ] Build node features: dihedral sin/cos, deterministic sinusoidal residue positional encoding with `posenc_dim=16`, coordinate-validity flag, optional `mu_r_norm`.
- [ ] Assert the default node feature dimension is 28 when using 6 dihedral features, 1 coordinate-validity feature, 5 `mu_r_norm` features, and 16 positional-encoding features.
- [ ] When adapting `MapDiff/dataloader/cath_dataset.py`, explicitly skip SASA, B-factor, and DSSP/secondary-structure fields rather than copying the source block wholesale.
- [ ] Return a PyG `Batch`-compatible structure plus enough metadata to scatter flat node hidden states back to `[B, L_token, H]`.
- [ ] Fail fast if `node_count_per_graph` does not equal each row's residue-mask count.

Verification command:

```bash
pytest tests/inverse_folding/dplm_refiner/test_geo_graph.py -v
```

## Task E2: EGNN Backbone

**Files:**
- Create: `inverse_folding/dplm_refiner/geo_encoder/egnn.py`
- Test: `tests/inverse_folding/dplm_refiner/test_geo_egnn.py`

- [ ] Adapt the minimum `EGNN_Sparse` logic from `MapDiff/model/egnn_pytorch/egnn_pyg.py` into local code with attribution comments, preserving PyG `MessagePassing` rather than hand-writing a custom scatter implementation.
- [ ] Support `update_edge`, `update_global`, `update_coors`, `norm_coors`, and `dropout`.
- [ ] If an internal layer returns `cat([coords, hidden], dim=-1)`, split by `pos_dim=3` immediately after the layer output. Do not pass the concatenated tensor into IPA.
- [ ] Keep the public EGNN stack output as flat hidden states `[N_res, H]` plus optional updated latent CA coordinates.
- [ ] Do not implement MapDiff time embedding, diffusion objective, or final 20-way predictor inside the EGNN block.
- [ ] Ensure `update_coors=false` keeps coordinates unchanged and `update_coors=true` changes only the internal latent coordinate channel.
- [ ] Test finite outputs, no cross-graph message passing, edge-update shape, and coordinate-update on/off behavior.
- [ ] Add an import smoke that verifies `torch_geometric` and `torch_scatter` are available in the active environment.

Verification command:

```bash
python -c "import torch_geometric; from torch_scatter import scatter_add; print('pyg-ok')"
pytest tests/inverse_folding/dplm_refiner/test_geo_egnn.py -v
```

## Task E3: IPA Dense Refinement And Encoder Contract

**Files:**
- Create: `inverse_folding/dplm_refiner/geo_encoder/ipa.py`
- Create: `inverse_folding/dplm_refiner/geo_encoder/encoder.py`
- Create: `inverse_folding/dplm/src/byprot/models/dplm/modules/geoegnn_ipa_encoder.py`
- Test: `tests/inverse_folding/dplm_refiner/test_geo_encoder_contract.py`

- [ ] Implement local `densify_to_token_grid(h_flat, graph_meta, residue_mask)` that writes flat EGNN hidden states directly into `[B, L_token, H]`.
- [ ] The densify helper must handle rows with zero residues by returning all-zero features for that protein and must fail fast if nonempty graph counts do not match `residue_mask.sum(dim=1)`.
- [ ] Do not require `torch_geometric.utils.to_dense_batch` in the production path. It may be used only in tests as a comparator for nonempty proteins.
- [ ] Feed EGNN hidden states into IPA as the initial node state; use original `N/CA/C` rigid frames as the anchor.
- [ ] When `update_coors=true`, expose updated-CA pair bias as an optional additive pair feature; do not replace original frames.
- [ ] Project refined IPA hidden states to `d_model=512` and zero invalid/special/pad token rows.
- [ ] Add an optional draft AA head returning `[B, L_token, vocab_or_aa]` compatible with `output_logits=True`.
- [ ] Return `encoder_out["coord_mask"]` in both `output_logits=False` and `output_logits=True` paths.
- [ ] Register `geoegnn_ipa_encoder` as a byprot model target.
- [ ] Test that `forward(batch, output_logits=False)` returns a dict with `feats`.
- [ ] Test that `forward(batch, output_logits=True)` returns `(logits, encoder_out)`.
- [ ] Test that `task.model.forward_encoder(batch, use_draft_seq=True)` can consume the new encoder and produce `init_pred`.
- [ ] Test that `task.model.forward_encoder(batch, use_draft_seq=True)` returns `coord_mask` in `encoder_out`.

Verification command:

```bash
PYTHONPATH=inverse_folding/dplm/src pytest \
  tests/inverse_folding/dplm_refiner/test_geo_encoder_contract.py -v
```

## Task E4: Checkpoint And Config Integration

**Files:**
- Create: `inverse_folding/dplm_refiner/geo_encoder/checkpoint.py`
- Modify: `inverse_folding/dplm/configs/experiment/dplm/cond_dplm_650m.yaml`
- Modify: `inverse_folding/dplm/requirements.txt`
- Test: `tests/inverse_folding/dplm_refiner/test_geo_encoder_checkpoint.py`

- [ ] Save checkpoints with keys: `model_state_dict`, `encoder_config`, `adapter_config`, `extra`, and `format_version`.
- [ ] Load checkpoints with strict config validation and explicit missing/unexpected-key reporting.
- [ ] Add a documented Hydra override or a separate config file for `model.encoder._target_=geoegnn_ipa_encoder`.
- [ ] Verify that byprot's registry-backed `_target_` resolution can instantiate `geoegnn_ipa_encoder` from the same mechanism currently used by `gvp_trans_encoder`.
- [ ] Add config fields for `update_coors`, graph feature selection, EGNN depth/hidden size, IPA depth/hidden size, `adapter_num_layers`, `adapter_gated`, `adapter_gate_init`, `detach_encoder_feats`, and `lambda_aux`.
- [ ] Confirm `torch_geometric` and `torch_scatter` are present in the target environment. If the DPLM vendored requirements file is used by cluster setup, uncomment or add `torch_scatter` there; do not rely only on the root requirements file.
- [ ] Preserve the default `cond_dplm_650m.yaml` GVP behavior unless the new encoder override is explicitly selected.
- [ ] Test checkpoint round-trip on a tiny encoder instance.
- [ ] Add a unit-like instantiation test that composes or loads the GeoEGNN-IPA encoder config and verifies `utils.instantiate_from_config(..., group="model")` resolves the registry name without `ImportError`.

Verification command:

```bash
PYTHONPATH=inverse_folding/dplm/src python - <<'PY'
from byprot import utils
from omegaconf import OmegaConf
cfg = OmegaConf.create({"_target_": "geoegnn_ipa_encoder", "d_model": 512})
model = utils.instantiate_from_config(cfg=cfg, group="model")
print(type(model).__name__)
PY
pytest tests/inverse_folding/dplm_refiner/test_geo_encoder_checkpoint.py -v
```

## Task E5: Training Script And Launcher Modes

**Files:**
- Create: `scripts/train_if_imp_encoder.py`
- Modify: `scripts/run_if_imp_refiner.py`
- Modify: `scripts/submit_if_imp.slurm`
- Modify: `doc/SCRIPTS.md`
- Test: `tests/scripts/test_if_imp_encoder_scripts.py`

- [ ] Create `scripts/train_if_imp_encoder.py` because `train_if_imp_sidecar.py` is semantically tied to residual GVP augmentation and a full-mask CE training schedule.
- [ ] Reuse existing CATH loader and `load_if_task()` helpers where possible.
- [ ] Freeze all DPLM backbone parameters.
- [ ] Train GeoEGNN-IPA encoder, draft head, and configured adapter parameters.
- [ ] Compute main loss through DPLM decoder with `detach_encoder_feats=false`.
- [ ] Use existing encoder-logit CE as the auxiliary AA head when `output_encoder_logits=true`; weight it by `lambda_aux` if implementing a custom loop, or document if using the task's existing `diff_loss + encoder_loss` behavior.
- [ ] Save `encoder_last.pt`, `run_config.yaml`, `metrics.jsonl`, and `manifest.json`.
- [ ] Extend `scripts/run_if_imp_refiner.py` with optional encoder replacement loading for generation.
- [ ] Extend `scripts/submit_if_imp.slurm` with `MODE=train_encoder|generate_encoder|diag_encoder`; do not add a new SLURM file.
- [ ] Update `doc/SCRIPTS.md` under the IF-IMP section with the new script and modes.
- [ ] Add script help tests and one import-level smoke test.

Verification commands:

```bash
python scripts/train_if_imp_encoder.py --help
python scripts/run_if_imp_refiner.py --help
pytest tests/scripts/test_if_imp_encoder_scripts.py -v
```

## Task E6: End-To-End Smoke And Required Ablations

**Files:**
- Modify: `scripts/diag_if_imp_arms.py` only if needed to include encoder arms.
- Modify: `scripts/compare_if_imp_ablation.py` only if needed to compare encoder arms.
- Update: `doc/IF_IMP.md` only if the implementation changes a frozen design point.
- Update: `LOG.md` after completion.

- [ ] Run a two-batch train smoke on CPU or one GPU and confirm loss is finite.
- [ ] Confirm trainable parameter names include GeoEGNN-IPA and configured adapters, and exclude DPLM backbone.
- [ ] Generate two CATH examples with `max_iter=2` and `use_draft_seq=true`.
- [ ] Run the minimum CATH ablations:
  - GVP baseline.
  - GeoEGNN-IPA with `update_coors=false`.
  - GeoEGNN-IPA with `update_coors=true`.
  - last-layer adapter only.
  - last-4 gated adapters.
  - auxiliary AA head on/off or `lambda_aux in {0.05, 0.10}`.
- [ ] Treat the replacement as mechanistically meaningful only if final DPLM recovery improves by at least 2 absolute percentage points over the current CATH baseline without foldability degradation.

Verification commands:

```bash
CATH_ROOT="${CATH_ROOT:?set CATH_ROOT to the CATH 4.3 root}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the Module K DPLM checkpoint}"
python scripts/train_if_imp_encoder.py \
  --cath-root "${CATH_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir /tmp/if_encoder_smoke \
  --limit-batches 2 \
  --device cpu

MODE=train_encoder LIMIT_BATCHES=2 DEVICE=cpu sbatch scripts/submit_if_imp.slurm
```

## Completion Criteria

This plan is complete when:

- default GVP DPLM behavior is unchanged without the GeoEGNN-IPA config;
- `geoegnn_ipa_encoder` satisfies the same forward/output contract as `gvp_trans_encoder`;
- main DPLM loss can backpropagate into the encoder when `detach_encoder_feats=false`;
- last-N gated adapters are configurable and backward-compatible;
- no production code imports `MapDiff`;
- no new runtime dependency on SciPy, RDKit, DSSP, SASA, or B-factor exists;
- `update_coors` is configurable and tested on/off;
- script registration is updated in `doc/SCRIPTS.md`;
- tests and smoke commands above pass or failures are documented with exact error messages;
- `LOG.md` records the implementation and validation evidence.

## Execution Notes

- Do not use `SidecarAttachedEncoder` for the final replacement path; it is useful only as a sidecar baseline reference.
- Do not train through `DPLMInvFold.forward()` until the `detach_encoder_feats` contract is implemented and tested.
- Keep all new graph code torch-native. Replace MapDiff's SciPy `cdist` usage with `torch.cdist`.
- Keep `use_draft_seq=true` support because current DPLM configs rely on encoder draft initialization.
- If torch-geometric is unavailable in the active environment, fail fast with an actionable message naming the missing package and the command/environment where it is expected.
