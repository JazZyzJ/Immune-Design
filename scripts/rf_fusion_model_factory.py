"""Shared model preparation for the RF-Fusion entry paths (PLAN V2F7).

PLAN V2F7: "Extract common model preparation and fresh-root capture from the V1 script into a shared
library factory; the V1 path delegates to it byte-identically and V2 consumes the same factory rather
than copying a closure."

**What is genuinely common.**  Both versions need the same frozen DPLM task and alphabet, the same
sampler, the same reference-flow sampler config under the same null-amplification assertion, the
same canonical AA20 token map, the same tokenizer digest, and -- the expensive one -- the same
per-protein prepared backbone and denoiser, cached so a cohort prepares each protein once.

**What is deliberately NOT common.**  Conditioning identity is not: V1 builds a V1 conditioning
record from its own provenance dict and V2 builds a :class:`V2ConditioningIdentity`.  They are
different types describing the same underlying digests, so the factory exposes the DIGESTS and each
version assembles its own identity.  Pretending they were one object would have forced one version's
schema onto the other.

**Verification boundary, stated plainly.**  ``build_entry_oracles`` cannot run locally: it needs
torch, a DPLM checkpoint, a Head checkpoint and PDBs, and no test in this repo exercises it.  What
is verified here is that the factory performs the expected call sequence against injected seams, and
that there is now exactly ONE implementation of that sequence.  Real-model byte identity is NOT
locally provable and is not claimed; it remains a cluster check.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = ["PreparedModel", "ModelSeams", "build_model_factory"]


@dataclass(frozen=True)
class ModelSeams:
    """The runtime entry points the factory calls.

    Injectable so the factory's wiring is testable without torch.  Defaults resolve the real
    implementations lazily, so importing this module costs nothing.
    """

    load_if_task: Callable[..., Any] | None = None
    prepare_backbone: Callable[..., Any] | None = None
    build_dplm_denoiser_context: Callable[..., Any] | None = None
    make_dplm_denoiser: Callable[..., Any] | None = None
    load_reference_flow_config: Callable[..., Any] | None = None
    make_sampler: Callable[..., Any] | None = None
    read_test_rows: Callable[..., Any] | None = None
    load_constraint_manifest: Callable[..., Any] | None = None

    def resolved(self) -> "ModelSeams":
        """Fill every unset seam with its real implementation, importing torch only now."""
        if all(getattr(self, f.name) is not None for f in self.__dataclass_fields__.values()):
            return self
        from inverse_folding.reference_flow.config import load_reference_flow_config
        from inverse_folding.reference_flow.constraints import load_constraint_manifest
        from inverse_folding.reference_flow.runtime import (
            build_dplm_denoiser_context,
            load_if_task,
            make_dplm_denoiser,
            prepare_backbone,
        )
        from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler

        def _read_test_rows(path):
            import pandas as pd

            return pd.read_parquet(path).set_index("protein_id")

        return ModelSeams(
            load_if_task=self.load_if_task or load_if_task,
            prepare_backbone=self.prepare_backbone or prepare_backbone,
            build_dplm_denoiser_context=(
                self.build_dplm_denoiser_context or build_dplm_denoiser_context),
            make_dplm_denoiser=self.make_dplm_denoiser or make_dplm_denoiser,
            load_reference_flow_config=(
                self.load_reference_flow_config or load_reference_flow_config),
            make_sampler=self.make_sampler or (
                lambda *, mask_token_id, vocab_size: PositionDependentDFMSampler(
                    mask_token_id=mask_token_id, vocab_size=vocab_size)),
            read_test_rows=self.read_test_rows or _read_test_rows,
            load_constraint_manifest=self.load_constraint_manifest or load_constraint_manifest,
        )


@dataclass
class PreparedModel:
    """One cohort's model context: everything below the version-specific oracles.

    Deliberately mutable and NOT frozen: it owns two caches whose whole purpose is to accumulate
    across a cohort.  A frozen value that copied itself would prepare the same backbone twice.
    """

    task: Any
    sampler: Any
    rf_config: Any
    id_to_aa: Mapping[int, str]
    aa_token_ids: frozenset
    mask_token_id: int
    vocab_size: int
    tokenizer_digest: str
    fixed_token_policy: str
    device: str
    pdb_root: Any
    _seams: ModelSeams
    _test_rows: Any = None
    _manifest: Any = None
    _prepared_cache: dict = field(default_factory=dict)
    _denoiser_cache: dict = field(default_factory=dict)

    def backbone_and_denoiser(self, protein_id: str):
        """Prepare a protein's backbone and denoiser ONCE per cohort.

        The cache is the reason this is a shared object rather than a function: preparing a
        backbone is the expensive step, and a per-call factory would repeat it for every root,
        every lookahead and every segment of the same protein.
        """
        if protein_id not in self._prepared_cache:
            row = dict(self._test_rows.loc[protein_id])
            row["protein_id"] = protein_id
            prepared = self._seams.prepare_backbone(
                task=self.task, entry=row, pdb_root=self.pdb_root, device=self.device)
            ctx = self._seams.build_dplm_denoiser_context(
                task=self.task, prepared=prepared, use_draft_seq_override=False)
            self._prepared_cache[protein_id] = prepared
            self._denoiser_cache[protein_id] = self._seams.make_dplm_denoiser(ctx)
        return self._prepared_cache[protein_id], self._denoiser_cache[protein_id]

    def sequence_length(self, protein_id: str) -> int:
        prepared, _ = self.backbone_and_denoiser(protein_id)
        return int(prepared.sequence_length)

    def fixed_tokens(self, protein_id: str) -> dict | None:
        """The hard-anchor constraint class for one protein, as sampler ``fixed_tokens``.

        ``None`` rather than ``{}`` for an unconstrained protein: the sampler treats the two
        differently, and an empty dict would declare a constraint set that exists and is empty
        rather than one that was never declared.
        """
        if self._manifest is None:
            return None
        constraint = self._manifest.constraint_for_protein(protein_id)
        fixed = {
            int(anchor.index_0b): int(self.task.alphabet.get_idx(str(anchor.expected_aa)))
            for anchor in constraint.hard_anchors
        }
        return fixed or None

    def coordinate_mask_digest(self, protein_id: str) -> str:
        """PER-PROTEIN, computed after the backbone is prepared.

        A cohort-level constant would claim a binding that does not exist: the coordinate-valid
        mask is a property of one structure, not of the run.
        """
        from scripts.rf_fusion_v1_oracles import coordinate_mask_digest

        prepared, _ = self.backbone_and_denoiser(protein_id)
        return coordinate_mask_digest(prepared)

    def null_h_values(self, length: int):
        from scripts.rf_fusion_v1_oracles import null_h_values

        return null_h_values(length)


def build_model_factory(
    *,
    base_if_checkpoint: Any,
    rf_sampler_config: Any,
    test_set_parquet: Any,
    pdb_root: Any,
    device: str = "cuda",
    constraint_manifest: Any = None,
    fixed_token_policy: str = "unconstrained",
    seams: ModelSeams | None = None,
) -> PreparedModel:
    """Load the frozen task, sampler and sampler config once for a whole cohort.

    The null-amplification assertion runs HERE rather than in each caller: it is the property that
    makes the substrate the frozen one, and a version that forgot to assert it would silently run a
    different kernel than the one every calibration was measured on.
    """
    from scripts.rf_fusion_v1_oracles import (
        _canonical_id_to_aa,
        assert_null_amplification,
        tokenizer_digest,
    )

    resolved = (seams or ModelSeams()).resolved()
    task = resolved.load_if_task(base_if_checkpoint, device=device)
    rf_config = resolved.load_reference_flow_config(rf_sampler_config)
    assert_null_amplification(rf_config)

    id_to_aa = _canonical_id_to_aa(task.alphabet)
    model = PreparedModel(
        task=task,
        sampler=resolved.make_sampler(
            mask_token_id=int(task.alphabet.mask_idx), vocab_size=len(task.alphabet)),
        rf_config=rf_config,
        id_to_aa=id_to_aa,
        aa_token_ids=frozenset(id_to_aa),
        mask_token_id=int(task.alphabet.mask_idx),
        vocab_size=len(task.alphabet),
        tokenizer_digest=tokenizer_digest(task.alphabet),
        fixed_token_policy=fixed_token_policy,
        device=device,
        pdb_root=pdb_root,
        _seams=resolved,
    )
    model._test_rows = resolved.read_test_rows(test_set_parquet)
    if constraint_manifest:
        model._manifest = resolved.load_constraint_manifest(constraint_manifest)
    return model
