# Immune-Design

Protein sequence design with a residue-level MHC-II Head, Fusion V2 generation,
Head-guided refinement, and independent immune/structure evaluation.

## Install

See [installation](docs/installation.md) for the CPU Head environment and optional
generation/structure backends. All entry points are Python APIs or parameterized
Python scripts; scheduling is managed by the caller.

## Score A Sequence

Obtain the allele-matched production checkpoint from the release weight bundle
and verify it against [the asset manifest](head-assets.json).

```bash
python scripts/score_head.py \
  --checkpoint weights/drb0701/epoch_29.pt \
  --input-fasta examples/sequence.fasta \
  --device cpu --output-dir run/head
```

The example is a synthetic input for checking installation, not a biological
reference or a WT baseline. Output includes window scores, residue hotspots,
and global risk. The canonical architecture is `a1res03` / `LC1`.

## Design And Evaluate

Follow [official workflows](docs/workflows.md) for configuration binding,
generation, multiroot selection, refinement, and evaluation. The official
candidate count defaults to eight and can be specified explicitly. Head
refinement retains exact per-seed Pareto fronts and does not require NMP.

For direct library use, the pure selection implementation is
`inverse_folding.reference_flow.official_selection`; index-addressable
structure metrics are in `inverse_folding.evaluation.structural_metrics_v2`.

- [Script and API index](docs/scripts.md)
- [Weights and config identities](docs/artifacts.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Validate

```bash
python -m pip install pytest==9.0.2
python -m pytest tests -q
```

Source is distributed under [MIT](LICENSE). Bundled third-party source and
external model weights retain their respective licenses.
