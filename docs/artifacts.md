# Model And Configuration Artifacts

Production Head weights are distributed separately from source. The release
asset `head-weights.tar.gz` contains `weights/` and `head-assets.json`. Verify
the per-file SHA-256 values in `head-assets.json` before using the checkpoints.

| Allele | Checkpoint | Training role |
|---|---|---|
| HLA-DRB1*07:01 | `weights/drb0701/epoch_29.pt` | Fixed-epoch full-data production |
| HLA-DRB1*04:01 | `weights/drb0401/epoch_34.pt` | Fixed-epoch full-data production |
| HLA-DRB1*15:01 | `weights/drb1501/epoch_24.pt` | Fixed-epoch full-data production |

The weights retain their original bytes, including loader metadata. Their monitor
values come from full-data training and must not be reported as held-out metrics.
The corresponding resolved training configs accompany the weights. The common
inference configs are in `epitope_head/configs/` in the source distribution.

## Inverse-Folding Checkpoint

The DPLM inverse-folding checkpoint is packaged as `dplm-inference.tar.part00`
through `part02`, with per-part and reconstructed-archive hashes in
[`dplm-assets.json`](../dplm-assets.json). Reassemble the parts in lexical order
and extract the archive. It contains `dplm/checkpoints/best.ckpt` and
`dplm/.hydra/config.yaml`. Pass the checkpoint path to the generation CLI.

All 868 inference tensors are preserved exactly. Optimizer/training state is
omitted and config filesystem paths are relative. The checkpoint file hash has
therefore changed; new runs must bind the released checkpoint hash. This package
is for inference, not for resuming the original optimizer state.

ESMFold2, ProteinMPNN, Protenix, AlphaFold3, and external executable dependencies
retain their own distribution terms. This source release does not bundle local
installations or automatically grant access to third-party weights. Supply their
paths through the relevant Python CLI. DPLM base weights and source retain their
upstream terms; the bundled source carries its Apache-2.0 notice, and OpenFold
retains its own notice.

Per-run manifests must bind the model, calibration, policy, and actual released
code revision. Moving a model does not change its file hash; changing its contents
or its inference config requires a new identity. Do not disable provenance checks
to load a configuration produced for a different model.
