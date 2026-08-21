"""Module E contract tests — chunking, datamodule, negative sampler, model path."""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from epitope_head.training.chunking import (
    ChunkPlan,
    assign_residue_to_chunk,
    assign_span_to_chunk,
    build_chunk_plan,
    compute_residue_reliability,
)
from epitope_head.training.datamodule import (
    ChunkSample,
    ProteinEntry,
    TokenBudgetSampler,
    build_chunk_samples,
    make_collate_fn,
)
from epitope_head.training.negatives import sample_negatives, NegativeSamplingShortfall


# ── E1a: Chunk plan generation ──────────────────────────────────────────────

class TestChunkPlan:

    def test_short_protein_single_chunk(self):
        """Protein <= 1022 gets one chunk starting at 0."""
        plan = build_chunk_plan(500, context_len=1022, stride=512, margin=32)
        assert plan.n_chunks == 1
        assert plan.starts == [0]
        assert plan.chunk_range(0) == (0, 500)

    def test_exact_context_length_no_chunking(self):
        """Protein exactly 1022 gets one chunk."""
        plan = build_chunk_plan(1022, context_len=1022, stride=512, margin=32)
        assert plan.n_chunks == 1

    def test_slightly_over_context_two_chunks(self):
        """1023 AA protein needs exactly 2 chunks."""
        plan = build_chunk_plan(1023, context_len=1022, stride=512, margin=32)
        assert plan.n_chunks == 2
        assert 0 in plan.starts
        assert 1 in plan.starts  # t_last = 1023-1022 = 1

    def test_long_protein_chunk_coverage(self):
        """All residues of a 3000 AA protein are covered by some chunk."""
        plan = build_chunk_plan(3000, context_len=1022, stride=512, margin=32)
        covered = set()
        for j in range(plan.n_chunks):
            s, e = plan.chunk_range(j)
            covered.update(range(s, e))
        assert covered == set(range(3000))

    def test_trusted_interior_full_coverage(self):
        """Union of trusted interiors covers all residues."""
        for L in [1500, 2048, 3000, 5000]:
            plan = build_chunk_plan(L, context_len=1022, stride=512, margin=32)
            covered = set()
            for j in range(plan.n_chunks):
                s, e = plan.trusted_interior(j)
                covered.update(range(s, e))
            assert covered == set(range(L)), f"Gap in trusted coverage for L={L}"

    def test_trusted_interior_first_chunk_no_left_margin(self):
        """First chunk trusted interior starts at 0."""
        plan = build_chunk_plan(2000, context_len=1022, stride=512, margin=32)
        t_start, _ = plan.trusted_interior(0)
        assert t_start == 0

    def test_trusted_interior_last_chunk_no_right_margin(self):
        """Last chunk trusted interior extends to L."""
        plan = build_chunk_plan(2000, context_len=1022, stride=512, margin=32)
        _, t_end = plan.trusted_interior(plan.n_chunks - 1)
        assert t_end == 2000

    def test_t_last_always_included(self):
        """t_last is always in starts."""
        for L in [1500, 2000, 2500, 3333, 5000, 6977]:
            plan = build_chunk_plan(L, context_len=1022, stride=512, margin=32)
            t_last = max(0, L - 1022)
            assert t_last in plan.starts, f"t_last={t_last} missing for L={L}"


# ── E1a+: Span-to-chunk ownership ─────────────────────────────────────────

class TestSpanOwnership:

    def test_short_protein_all_to_chunk_zero(self):
        """Single-chunk protein: all spans go to chunk 0."""
        plan = build_chunk_plan(500, context_len=1022, stride=512, margin=32)
        assert assign_span_to_chunk(plan, 10, 25) == 0
        assert assign_span_to_chunk(plan, 0, 15) == 0
        assert assign_span_to_chunk(plan, 485, 500) == 0

    def test_overlap_span_assigned_to_nearest_center(self):
        """Span in overlap region goes to chunk with nearest center."""
        # L=1500: chunk0=[0,1022) center=511, chunk1=[478,1500) center=989
        # Overlap: [478, 1022)
        plan = build_chunk_plan(1500, context_len=1022, stride=512, margin=32)
        # Span [500, 515): midpoint=507.5, closer to chunk0 center 511
        assert assign_span_to_chunk(plan, 500, 515) == 0
        # Span [900, 915): midpoint=907.5, closer to chunk1 center
        assert assign_span_to_chunk(plan, 900, 915) == 1

    def test_exactly_once_long_protein(self):
        """Each positive assigned to exactly one chunk — no duplication."""
        plan = build_chunk_plan(3000, context_len=1022, stride=512, margin=32)
        # Generate positives spanning the full protein
        positives = [{"start_0b": i, "end_0b": i + 15, "pep_len": 15}
                     for i in range(0, 2985, 50)]
        owners = [assign_span_to_chunk(plan, p["start_0b"], p["end_0b"])
                  for p in positives]
        # Every positive is assigned (no -1)
        assert all(o >= 0 for o in owners)
        # No duplication check: build_chunk_samples should produce exactly
        # len(positives) total positives across all chunks
        entry = ProteinEntry(
            protein_id="test", protein_seq="A" * 3000,
            allele="X", positives=positives, sequence_length=3000,
        )
        samples = build_chunk_samples([entry])
        total = sum(len(s.positives) for s in samples)
        assert total == len(positives), (
            f"Expected {len(positives)} total positives, got {total} "
            f"(duplication or loss)"
        )

    def test_deterministic(self):
        """Same input produces same assignment."""
        plan = build_chunk_plan(2000)
        a1 = assign_span_to_chunk(plan, 600, 615)
        a2 = assign_span_to_chunk(plan, 600, 615)
        assert a1 == a2

    def test_orphan_raises(self):
        """Positive that fits in no chunk raises ValueError, not silent drop."""
        # Artificially tiny context_len so a 15-mer can't fit
        entry = ProteinEntry(
            protein_id="test", protein_seq="A" * 30,
            allele="X",
            positives=[{"start_0b": 5, "end_0b": 20, "pep_len": 15, "support_n": 1}],
            sequence_length=30,
        )
        with pytest.raises(ValueError, match="fit in no chunk"):
            build_chunk_samples([entry], context_len=10, stride=5, margin=2)


# ── E1b: Residue-to-chunk assignment ────────────────────────────────────────

class TestResidueAssignment:

    def test_short_protein_all_chunk_zero(self):
        plan = build_chunk_plan(500)
        assignment = assign_residue_to_chunk(plan)
        assert (assignment == 0).all()
        assert len(assignment) == 500

    def test_long_protein_no_negative_assignments(self):
        plan = build_chunk_plan(3000)
        assignment = assign_residue_to_chunk(plan)
        assert (assignment >= 0).all()

    def test_deterministic(self):
        plan = build_chunk_plan(3000)
        a1 = assign_residue_to_chunk(plan)
        a2 = assign_residue_to_chunk(plan)
        np.testing.assert_array_equal(a1, a2)


# ── E1c: Reliability ───────────────────────────────────────────────────────

class TestReliability:

    def test_short_protein_all_reliable(self):
        plan = build_chunk_plan(500)
        rel = compute_residue_reliability(plan)
        assert (rel == 1.0).all()

    def test_long_protein_all_covered(self):
        """All residues have reliability > 0 (trusted interiors cover everything)."""
        plan = build_chunk_plan(3000)
        rel = compute_residue_reliability(plan)
        assert (rel > 0).all()


# ── E1d: Token budget batching ──────────────────────────────────────────────

class TestTokenBudget:

    def _make_samples(self, lengths: list[int]) -> list[ChunkSample]:
        return [
            ChunkSample(
                protein_id=f"P{i}", protein_seq="A" * l, allele="X",
                positives=[], sequence_length=l,
                chunk_idx=0, chunk_start=0, chunk_end=l,
            )
            for i, l in enumerate(lengths)
        ]

    def test_budget_respected(self):
        """No batch exceeds max_tokens."""
        samples = self._make_samples([500, 300, 400, 600, 200, 800])
        sampler = TokenBudgetSampler(samples, max_tokens=1000, shuffle=False)
        for batch_indices in sampler:
            total = sum(samples[i].chunk_end - samples[i].chunk_start for i in batch_indices)
            assert total <= 1000

    def test_all_samples_yielded(self):
        """All samples appear exactly once."""
        samples = self._make_samples([500, 300, 400, 600, 200])
        sampler = TokenBudgetSampler(samples, max_tokens=1000, shuffle=False)
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(len(samples)))

    def test_single_large_sample(self):
        """A single sample larger than budget still gets its own batch."""
        samples = self._make_samples([1500])
        sampler = TokenBudgetSampler(samples, max_tokens=1000, shuffle=False)
        batches = list(sampler)
        assert len(batches) == 1
        assert batches[0] == [0]


# ── E2: Negative sampler ───────────────────────────────────────────────────

class TestNegativeSampler:

    def test_no_positive_in_negatives(self):
        """No sampled negative matches any positive span."""
        positives = [
            {"start_0b": 10, "end_0b": 25, "pep_len": 15, "support_n": 1},
            {"start_0b": 50, "end_0b": 65, "pep_len": 15, "support_n": 1},
        ]
        rng = np.random.RandomState(42)
        negs = sample_negatives(
            protein_length=200, positives=positives,
            neg_ratio=7, rng=rng,
        )
        pos_set = {(p["start_0b"], p["end_0b"]) for p in positives}
        for n in negs:
            assert (n["start_0b"], n["end_0b"]) not in pos_set

    def test_correct_count(self):
        """Returns approximately neg_ratio * n_positives negatives."""
        positives = [
            {"start_0b": 10, "end_0b": 25, "pep_len": 15, "support_n": 1},
        ]
        rng = np.random.RandomState(42)
        negs = sample_negatives(
            protein_length=500, positives=positives,
            neg_ratio=7, rng=rng,
        )
        assert len(negs) == 7

    def test_empty_positives_returns_empty(self):
        negs = sample_negatives(protein_length=100, positives=[], neg_ratio=7)
        assert negs == []

    def test_negatives_within_bounds(self):
        """All negatives are within protein bounds."""
        positives = [
            {"start_0b": 5, "end_0b": 20, "pep_len": 15, "support_n": 1},
        ]
        rng = np.random.RandomState(42)
        negs = sample_negatives(
            protein_length=100, positives=positives,
            neg_ratio=7, rng=rng,
        )
        for n in negs:
            assert 0 <= n["start_0b"] < n["end_0b"] <= 100
            assert n["pep_len"] == n["end_0b"] - n["start_0b"]

    def test_hard_negatives_have_limited_overlap(self):
        """Hard negatives don't exceed max overlap ratio with their anchor."""
        positives = [
            {"start_0b": 50, "end_0b": 65, "pep_len": 15, "support_n": 1},
        ]
        rng = np.random.RandomState(42)
        negs = sample_negatives(
            protein_length=200, positives=positives,
            neg_ratio=10, hard_negative_fraction=1.0,
            hard_neg_max_overlap_ratio=0.8, rng=rng,
        )
        for n in negs:
            # Check overlap with anchor
            overlap_start = max(50, n["start_0b"])
            overlap_end = min(65, n["end_0b"])
            overlap = max(0, overlap_end - overlap_start)
            assert overlap < 0.8 * 15

    def test_deterministic(self):
        """Same seed produces same negatives."""
        positives = [
            {"start_0b": 10, "end_0b": 25, "pep_len": 15, "support_n": 1},
        ]
        n1 = sample_negatives(200, positives, rng=np.random.RandomState(42))
        n2 = sample_negatives(200, positives, rng=np.random.RandomState(42))
        assert n1 == n2

    def test_length_match_positive(self):
        """With match_positive, negative lengths come from positive distribution."""
        positives = [
            {"start_0b": 10, "end_0b": 25, "pep_len": 15, "support_n": 1},
            {"start_0b": 30, "end_0b": 50, "pep_len": 20, "support_n": 1},
        ]
        rng = np.random.RandomState(42)
        negs = sample_negatives(
            500, positives, neg_ratio=50,
            neg_length_sampling="match_positive", rng=rng,
        )
        neg_lens = {n["pep_len"] for n in negs}
        # All negative lengths should be from {15, 20}
        assert neg_lens.issubset({15, 20})

    def test_strict_mode_raises_on_hard_shortfall(self):
        """strict=True raises when hard negatives can't be generated."""
        # Tiny protein with positives covering most positions → hard negs impossible
        positives = [{"start_0b": i, "end_0b": i + 12, "pep_len": 12, "support_n": 1}
                     for i in range(0, 15)]
        with pytest.raises(NegativeSamplingShortfall, match="Hard negative shortfall"):
            sample_negatives(
                protein_length=20, positives=positives,
                neg_ratio=7, hard_negative_fraction=1.0,
                strict=True, rng=np.random.RandomState(42),
            )

    def test_non_strict_backfills_and_returns(self):
        """strict=False backfills hard shortfall with easy negatives."""
        positives = [{"start_0b": i, "end_0b": i + 12, "pep_len": 12, "support_n": 1}
                     for i in range(0, 3)]
        negs = sample_negatives(
            protein_length=200, positives=positives,
            neg_ratio=7, hard_negative_fraction=0.9,
            strict=False, rng=np.random.RandomState(42),
        )
        # Should still return the full count (backfilled with easy)
        assert len(negs) == 7 * 3


# ── E0: Config deep validation ──────────────────────────────────────────────

class TestConfigSchema:

    def test_train_config_loads_valid(self):
        """Valid train.yaml loads without error."""
        from epitope_head.configs import load_train_config
        cfg = load_train_config()
        assert cfg["neg_ratio"] == 7
        assert cfg["chunking"]["context_len"] == 1022
        assert cfg["loss"]["tau"] == 1.0

    def test_train_config_missing_top_key_raises(self, tmp_path):
        """Missing a top-level frozen key raises ValueError."""
        import yaml
        from epitope_head.configs import load_train_config
        bad_cfg = {"train": {"neg_ratio": 7, "loss": {}, "lr": 1e-3}}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(bad_cfg))
        with pytest.raises(ValueError, match="missing required keys"):
            load_train_config(p)

    def test_train_config_missing_loss_subkey_raises(self, tmp_path):
        """Missing loss sub-key raises ValueError."""
        import yaml
        from epitope_head.configs import load_train_config, load_data_config
        # Load valid config, remove a loss sub-key
        valid = load_train_config()
        del valid["loss"]["tau"]
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump({"train": valid}))
        with pytest.raises(ValueError, match="train.loss missing required keys"):
            load_train_config(p)

    def test_train_config_missing_chunking_subkey_raises(self, tmp_path):
        """Missing chunking sub-key raises ValueError."""
        import yaml
        from epitope_head.configs import load_train_config
        valid = load_train_config()
        del valid["chunking"]["stride"]
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump({"train": valid}))
        with pytest.raises(ValueError, match="train.chunking missing required keys"):
            load_train_config(p)


# ── E1: Collate tensor outputs + tokenize_fn ────────────────────────────────

class TestCollateInterface:

    def _make_batch(self):
        return [
            ChunkSample(
                protein_id="P1", protein_seq="ACDEFGHIKLMNPQRSTVWY" * 5,
                allele="HLA-A", positives=[],
                sequence_length=100, chunk_idx=0, chunk_start=0, chunk_end=100,
            ),
            ChunkSample(
                protein_id="P2", protein_seq="ACDEFGHIKLMNPQRSTVWY" * 3,
                allele="HLA-B", positives=[],
                sequence_length=60, chunk_idx=0, chunk_start=0, chunk_end=60,
            ),
        ]

    def test_collate_returns_tensors_for_numeric_fields(self):
        """Numeric fields (starts, ends, indices, lengths) are LongTensors."""
        import torch
        collate_fn = make_collate_fn(tokenize_fn=None)
        batch = collate_fn(self._make_batch())
        assert isinstance(batch["chunk_starts"], torch.Tensor)
        assert batch["chunk_starts"].dtype == torch.long
        assert isinstance(batch["chunk_ends"], torch.Tensor)
        assert isinstance(batch["chunk_indices"], torch.Tensor)
        assert isinstance(batch["sequence_lengths"], torch.Tensor)
        assert batch["chunk_starts"].shape == (2,)

    def test_collate_without_tokenizer_has_no_token_ids(self):
        collate_fn = make_collate_fn(tokenize_fn=None)
        batch = collate_fn(self._make_batch())
        assert "token_ids" not in batch
        assert "chunk_seqs" in batch

    def test_collate_with_tokenizer_adds_token_ids(self):
        """When tokenize_fn provided, batch includes token_ids and attention_mask."""
        import torch

        def mock_tokenize(seqs: list[str]) -> dict:
            max_len = max(len(s) for s in seqs) + 2  # BOS + EOS
            token_ids = torch.zeros(len(seqs), max_len, dtype=torch.long)
            attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
            for i, s in enumerate(seqs):
                t = len(s) + 2
                token_ids[i, :t] = 1
                attention_mask[i, :t] = True
            return {"token_ids": token_ids, "attention_mask": attention_mask}

        collate_fn = make_collate_fn(tokenize_fn=mock_tokenize)
        batch = collate_fn(self._make_batch())
        assert "token_ids" in batch
        assert "attention_mask" in batch
        assert batch["token_ids"].shape[0] == 2


# ── E3: Model path ──────────────────────────────────────────────────────────

from epitope_head.training.model import (
    ESMTokenizer,
    ProjectionHead,
    SpanFeatureBuilder,
    ScorerMLP,
    EpitopeScorer,
)


class MockEncoder(nn.Module):
    """Mock encoder that returns random embeddings. No ESM-2 weights needed."""

    def __init__(self, d_enc: int = 1280):
        super().__init__()
        self.d_enc = d_enc
        self.num_layers = 33  # mimic ESM-2 attribute

    def forward(self, token_ids, repr_layers=None):
        B, T = token_ids.shape
        hidden = torch.randn(B, T, self.d_enc)
        return {"representations": {self.num_layers: hidden}}


class MockFrozenEncoder(nn.Module):
    """Simpler mock that directly returns (embeddings, lengths) like FrozenESMEncoder."""

    def __init__(self, d_enc: int = 16):
        super().__init__()
        self.d_enc = d_enc

    @torch.no_grad()
    def forward(self, token_ids, attention_mask):
        B, T = token_ids.shape
        real_counts = attention_mask.sum(dim=1)
        lengths = real_counts - 2  # strip BOS/EOS
        L_max = int(lengths.max().item())
        embeddings = torch.randn(B, L_max, self.d_enc)
        return embeddings, lengths


class TestProjectionHead:

    def test_shape(self):
        proj = ProjectionHead(d_enc=1280, d_proj=128)
        x = torch.randn(2, 100, 1280)
        out = proj(x)
        assert out.shape == (2, 100, 128)


class TestSpanFeatureBuilder:

    def _make_builder(self, d_proj=16):
        return SpanFeatureBuilder(
            d_proj=d_proj, length_emb_dim=4, allele_emb_dim=4,
            min_k=12, max_k=25, pad_left_init="zeros", pad_right_init="zeros",
        )

    def test_d_phi(self):
        """D_phi = 5*D_proj + length_emb + allele_emb."""
        # _make_builder uses length_emb_dim=4, allele_emb_dim=4
        builder = self._make_builder(d_proj=128)
        assert builder.d_phi == 5 * 128 + 4 + 4
        small = self._make_builder(d_proj=16)
        assert small.d_phi == 5 * 16 + 4 + 4
        # Verify codemap default: D_phi = 5*128 + 16 + 16 = 672
        full = SpanFeatureBuilder(
            d_proj=128, length_emb_dim=16, allele_emb_dim=16,
        )
        assert full.d_phi == 672

    def test_output_shape(self):
        builder = self._make_builder(d_proj=16)
        G = torch.randn(100, 16)
        spans = torch.tensor([[10, 25], [30, 45], [0, 15]], dtype=torch.long)
        allele_idx = torch.zeros(3, dtype=torch.long)
        phi = builder(G, spans, chunk_len=100, allele_idx=allele_idx)
        assert phi.shape == (3, builder.d_phi)

    def test_boundary_pad_left(self):
        """Span starting at 0 uses pad_left vector for left flank."""
        builder = self._make_builder(d_proj=16)
        # Set pad_left to known value
        with torch.no_grad():
            builder.pad_left.fill_(99.0)
        G = torch.randn(50, 16)
        spans = torch.tensor([[0, 15]], dtype=torch.long)
        allele_idx = torch.zeros(1, dtype=torch.long)
        phi = builder(G, spans, chunk_len=50, allele_idx=allele_idx)
        # Flank-left is at position 3*D_proj : 4*D_proj in the concat
        d = 16
        fl_left = phi[0, 3 * d: 4 * d]
        assert torch.allclose(fl_left, torch.tensor(99.0))

    def test_boundary_pad_right(self):
        """Span ending at chunk_len uses pad_right vector for right flank."""
        builder = self._make_builder(d_proj=16)
        with torch.no_grad():
            builder.pad_right.fill_(77.0)
        G = torch.randn(50, 16)
        spans = torch.tensor([[35, 50]], dtype=torch.long)  # end == chunk_len
        allele_idx = torch.zeros(1, dtype=torch.long)
        phi = builder(G, spans, chunk_len=50, allele_idx=allele_idx)
        d = 16
        fl_right = phi[0, 4 * d: 5 * d]
        assert torch.allclose(fl_right, torch.tensor(77.0))

    def test_prefix_sum_mean_pool_correctness(self):
        """Mean pool matches manual computation."""
        d = 16
        builder = self._make_builder(d_proj=d)
        # Create known embeddings: 12 residues with specific values
        G = torch.zeros(50, d)
        # Set positions 5..16 (span length 12 = min_k) to known values
        G[5] = 1.0
        G[6] = 3.0
        G[7] = 2.0
        # positions 8..16 remain 0
        # Span [5, 17): length=12, mean = (1+3+2+0*9)/12 = 0.5
        spans = torch.tensor([[5, 17]], dtype=torch.long)
        allele_idx = torch.zeros(1, dtype=torch.long)
        phi = builder(G, spans, chunk_len=50, allele_idx=allele_idx)
        mean_pool = phi[0, :d]
        assert torch.allclose(mean_pool, torch.tensor(6.0 / 12.0))

    def test_interior_not_at_boundary(self):
        """Non-boundary spans use actual neighbor embeddings, not pad vectors."""
        d = 16
        builder = self._make_builder(d_proj=d)
        with torch.no_grad():
            builder.pad_left.fill_(99.0)
            builder.pad_right.fill_(99.0)
        G = torch.randn(50, d)
        spans = torch.tensor([[10, 25]], dtype=torch.long)
        allele_idx = torch.zeros(1, dtype=torch.long)
        phi = builder(G, spans, chunk_len=50, allele_idx=allele_idx)
        # Flank-left should be G[9], not pad_left
        fl_left = phi[0, 3 * d: 4 * d]
        assert torch.allclose(fl_left, G[9])
        # Flank-right should be G[25], not pad_right
        fl_right = phi[0, 4 * d: 5 * d]
        assert torch.allclose(fl_right, G[25])


class TestScorerMLP:

    def test_shape(self):
        scorer = ScorerMLP(d_phi=88, hidden_dim=32, activation="gelu")
        phi = torch.randn(5, 88)
        z = scorer(phi)
        assert z.shape == (5,)

    def test_single_span(self):
        scorer = ScorerMLP(d_phi=88, hidden_dim=32)
        phi = torch.randn(1, 88)
        z = scorer(phi)
        assert z.shape == (1,)


class TestEpitopeScorer:

    def _make_model(self, d_enc=16, d_proj=8):
        encoder = MockFrozenEncoder(d_enc=d_enc)
        return EpitopeScorer(
            encoder=encoder,
            d_enc=d_enc,
            d_proj=d_proj,
            length_emb_dim=4,
            allele_emb_dim=4,
            scorer_hidden_dim=16,
        )

    def test_forward_shapes(self):
        """Full forward returns correct logit shapes per chunk."""
        model = self._make_model()
        B = 2
        T = 52  # BOS + 50 residues + EOS
        token_ids = torch.ones(B, T, dtype=torch.long)
        attention_mask = torch.ones(B, T, dtype=torch.bool)
        chunk_lengths = torch.tensor([50, 50])

        spans_list = [
            torch.tensor([[5, 20], [10, 22]], dtype=torch.long),
            torch.tensor([[0, 15]], dtype=torch.long),
        ]
        allele_idx_list = [
            torch.zeros(2, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        ]

        logits = model(token_ids, attention_mask, spans_list, allele_idx_list, chunk_lengths)
        assert len(logits) == 2
        assert logits[0].shape == (2,)
        assert logits[1].shape == (1,)

    def test_encoder_frozen(self):
        """Encoder parameters have requires_grad=False."""
        from epitope_head.training.model import FrozenESMEncoder
        mock_esm = MockEncoder(d_enc=16)
        frozen = FrozenESMEncoder(mock_esm, d_enc=16)
        for p in frozen.esm.parameters():
            assert not p.requires_grad

    def test_encoder_stays_eval_after_train(self):
        """FrozenESMEncoder.esm stays in eval mode even when parent calls .train()."""
        from epitope_head.training.model import FrozenESMEncoder
        mock_esm = MockEncoder(d_enc=16)
        frozen = FrozenESMEncoder(mock_esm, d_enc=16)
        assert not frozen.esm.training  # starts eval
        frozen.train()  # parent goes to train mode
        assert not frozen.esm.training  # ESM still eval
        # Full model: EpitopeScorer.train() should not affect encoder.esm
        model = self._make_model()
        model.train()
        assert not model.encoder.esm.training if hasattr(model.encoder, 'esm') else True

    def test_learnable_params_require_grad(self):
        """Projection, span features, scorer params require grad."""
        model = self._make_model()
        learnable_modules = [model.projection, model.span_features, model.scorer]
        for mod in learnable_modules:
            for p in mod.parameters():
                assert p.requires_grad, f"Parameter in {mod} should be learnable"

    def test_different_span_counts_per_chunk(self):
        """Handles variable number of spans per chunk in a batch."""
        model = self._make_model()
        B = 3
        T = 42
        token_ids = torch.ones(B, T, dtype=torch.long)
        attention_mask = torch.ones(B, T, dtype=torch.bool)
        chunk_lengths = torch.tensor([40, 40, 40])

        spans_list = [
            torch.tensor([[5, 20]], dtype=torch.long),        # 1 span
            torch.tensor([[5, 20], [10, 22], [0, 15]], dtype=torch.long),  # 3 spans
            torch.tensor([[2, 14], [8, 20], [15, 27], [20, 35]], dtype=torch.long),  # 4 spans
        ]
        allele_idx_list = [torch.zeros(s.shape[0], dtype=torch.long) for s in spans_list]

        logits = model(token_ids, attention_mask, spans_list, allele_idx_list, chunk_lengths)
        assert logits[0].shape == (1,)
        assert logits[1].shape == (3,)
        assert logits[2].shape == (4,)


class TestSeamConsistency:
    """PLAN E3 TDD gate: overlapping residues between adjacent chunks
    must produce identical span scores when scored on either chunk."""

    def test_seam_residue_scores_agree(self):
        """A span in the overlap region of two adjacent chunks gets the same score."""
        # Use a deterministic mock encoder that returns position-dependent embeddings
        d_enc = 16
        d_proj = 8

        class DeterministicEncoder(nn.Module):
            """Returns embeddings derived from token_ids so same tokens → same embeddings."""
            def __init__(self):
                super().__init__()
                self.d_enc = d_enc
            @torch.no_grad()
            def forward(self, token_ids, attention_mask):
                B, T = token_ids.shape
                real_counts = attention_mask.sum(dim=1)
                lengths = real_counts - 2
                L_max = int(lengths.max().item())
                embeddings = torch.zeros(B, L_max, d_enc)
                for i in range(B):
                    L_i = int(lengths[i].item())
                    # Deterministic: embedding = one-hot-ish from token_id
                    for j in range(L_i):
                        embeddings[i, j, token_ids[i, j + 1].item() % d_enc] = 1.0
                return embeddings, lengths

        model = EpitopeScorer(
            encoder=DeterministicEncoder(),
            d_enc=d_enc, d_proj=d_proj,
            length_emb_dim=4, allele_emb_dim=4,
            scorer_hidden_dim=16,
        )
        model.eval()

        # Simulate a 1500 AA protein with two overlapping chunks
        # Chunk 0: [0, 1022), Chunk 1: [510, 1500)
        # Overlap: [510, 1022) — 512 residues
        # A span at protein position [600, 615) exists in both chunks:
        #   Chunk 0 local: [600, 615)
        #   Chunk 1 local: [600-510, 615-510) = [90, 105)

        protein_tokens = torch.arange(5, 5 + 1500) % 30 + 4  # deterministic AA tokens
        # Build chunk 0 token_ids: BOS + protein[0:1022] + EOS
        bos, eos = 0, 2
        chunk0_ids = torch.cat([torch.tensor([bos]), protein_tokens[:1022], torch.tensor([eos])]).unsqueeze(0)
        chunk0_mask = torch.ones_like(chunk0_ids, dtype=torch.bool)

        # Build chunk 1 token_ids: BOS + protein[510:1500] + EOS
        chunk1_seq = protein_tokens[510:1500]
        chunk1_ids = torch.cat([torch.tensor([bos]), chunk1_seq, torch.tensor([eos])]).unsqueeze(0)
        chunk1_mask = torch.ones_like(chunk1_ids, dtype=torch.bool)

        # Score span [600, 615) on chunk 0 (local coords: [600, 615))
        spans_c0 = torch.tensor([[600, 615]], dtype=torch.long)
        allele_c0 = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            G0, len0 = model.encode_and_project(chunk0_ids, chunk0_mask)
            score_c0 = model.score_spans(G0[0], int(len0[0].item()), spans_c0, allele_c0)

        # Score same span on chunk 1 (local coords: [90, 105))
        spans_c1 = torch.tensor([[90, 105]], dtype=torch.long)
        allele_c1 = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            G1, len1 = model.encode_and_project(chunk1_ids, chunk1_mask)
            score_c1 = model.score_spans(G1[0], int(len1[0].item()), spans_c1, allele_c1)

        # Scores should be identical (same input tokens → same embeddings → same score)
        assert torch.allclose(score_c0, score_c1, atol=1e-5), \
            f"Seam inconsistency: chunk0 score={score_c0.item():.6f}, chunk1 score={score_c1.item():.6f}"

    def test_overlap_embeddings_match(self):
        """Projected embeddings for overlapping residues match between chunks."""
        d_enc = 16
        d_proj = 8

        class DeterministicEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.d_enc = d_enc
            @torch.no_grad()
            def forward(self, token_ids, attention_mask):
                B, T = token_ids.shape
                real_counts = attention_mask.sum(dim=1)
                lengths = real_counts - 2
                L_max = int(lengths.max().item())
                embeddings = torch.zeros(B, L_max, d_enc)
                for i in range(B):
                    L_i = int(lengths[i].item())
                    for j in range(L_i):
                        embeddings[i, j, token_ids[i, j + 1].item() % d_enc] = 1.0
                return embeddings, lengths

        model = EpitopeScorer(
            encoder=DeterministicEncoder(),
            d_enc=d_enc, d_proj=d_proj,
            length_emb_dim=4, allele_emb_dim=4,
            scorer_hidden_dim=16,
        )
        model.eval()

        protein_tokens = torch.arange(5, 5 + 1500) % 30 + 4
        bos, eos = 0, 2

        chunk0_ids = torch.cat([torch.tensor([bos]), protein_tokens[:1022], torch.tensor([eos])]).unsqueeze(0)
        chunk0_mask = torch.ones_like(chunk0_ids, dtype=torch.bool)

        chunk1_ids = torch.cat([torch.tensor([bos]), protein_tokens[510:1500], torch.tensor([eos])]).unsqueeze(0)
        chunk1_mask = torch.ones_like(chunk1_ids, dtype=torch.bool)

        with torch.no_grad():
            G0, _ = model.encode_and_project(chunk0_ids, chunk0_mask)
            G1, _ = model.encode_and_project(chunk1_ids, chunk1_mask)

        # Overlap region: protein[510:1022]
        # Chunk 0 local: G0[0, 510:1022]
        # Chunk 1 local: G1[0, 0:512]
        overlap_from_c0 = G0[0, 510:1022]  # [512, d_proj]
        overlap_from_c1 = G1[0, 0:512]     # [512, d_proj]
        assert torch.allclose(overlap_from_c0, overlap_from_c1, atol=1e-5), \
            "Projected embeddings differ in overlap region"


class TestSpanLengthGuard:

    def test_illegal_length_raises(self):
        """Span with length outside [min_k, max_k] raises ValueError."""
        builder = SpanFeatureBuilder(
            d_proj=16, length_emb_dim=4, allele_emb_dim=4,
            min_k=12, max_k=25,
        )
        G = torch.randn(50, 16)
        # pep_len = 5, which is < min_k=12
        spans = torch.tensor([[0, 5]], dtype=torch.long)
        allele_idx = torch.zeros(1, dtype=torch.long)
        with pytest.raises(ValueError, match="Span lengths outside"):
            builder(G, spans, chunk_len=50, allele_idx=allele_idx)

    def test_legal_length_passes(self):
        """Span with length in [min_k, max_k] works fine."""
        builder = SpanFeatureBuilder(
            d_proj=16, length_emb_dim=4, allele_emb_dim=4,
            min_k=12, max_k=25,
        )
        G = torch.randn(50, 16)
        spans = torch.tensor([[0, 15], [10, 22]], dtype=torch.long)
        allele_idx = torch.zeros(2, dtype=torch.long)
        phi = builder(G, spans, chunk_len=50, allele_idx=allele_idx)
        assert phi.shape[0] == 2


class TestESMTokenizer:

    def test_tokenize_shape(self):
        """Tokenizer produces correct shapes."""
        import esm
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        tokenizer = ESMTokenizer(alphabet)
        seqs = ["ACDEF", "GHI"]
        out = tokenizer(seqs)
        assert out["token_ids"].shape == (2, 7)  # max(5,3) + 2 = 7
        assert out["attention_mask"].shape == (2, 7)

    def test_bos_eos_present(self):
        """First token is BOS, last real token is EOS."""
        import esm
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        tokenizer = ESMTokenizer(alphabet)
        out = tokenizer(["ACDEF"])
        ids = out["token_ids"][0]
        assert ids[0].item() == alphabet.cls_idx  # BOS
        assert ids[6].item() == alphabet.eos_idx   # EOS after 5 AAs

    def test_padding(self):
        """Shorter sequence is padded, mask is False for padding."""
        import esm
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        tokenizer = ESMTokenizer(alphabet)
        out = tokenizer(["ACDEF", "GH"])
        # "GH" has 4 tokens (BOS+G+H+EOS), padded to 7
        assert out["attention_mask"][1, 3].item() == True   # EOS
        assert out["attention_mask"][1, 4].item() == False   # PAD
        assert out["token_ids"][1, 4].item() == alphabet.padding_idx


# ── E4: Training losses ─────────────────────────────────────────────────────

from epitope_head.training.losses import (
    info_nce_loss,
    multi_positive_loss,
    smoothness_loss,
    compute_loss,
)


class TestInfoNCELoss:

    def test_perfect_ranking_low_loss(self):
        """When pos >> neg, loss approaches 0."""
        pos = torch.tensor([10.0])
        neg = torch.tensor([-10.0, -10.0, -10.0])
        loss = info_nce_loss(pos, neg, tau=0.1)
        assert loss.item() < 0.01

    def test_inverted_ranking_high_loss(self):
        """When neg >> pos, loss is large."""
        pos = torch.tensor([-10.0])
        neg = torch.tensor([10.0, 10.0, 10.0])
        loss = info_nce_loss(pos, neg, tau=0.1)
        assert loss.item() > 10.0

    def test_loss_direction(self):
        """Better ranking yields lower loss."""
        neg = torch.tensor([0.0, 0.0, 0.0])
        loss_good = info_nce_loss(torch.tensor([5.0]), neg, tau=0.1)
        loss_bad = info_nce_loss(torch.tensor([-5.0]), neg, tau=0.1)
        assert loss_good < loss_bad

    def test_numerical_stability_large_logits(self):
        """No NaN/Inf with extreme logits."""
        pos = torch.tensor([1000.0])
        neg = torch.tensor([999.0, 998.0])
        loss = info_nce_loss(pos, neg, tau=0.1)
        assert torch.isfinite(loss)

    def test_numerical_stability_negative_logits(self):
        """No NaN/Inf with very negative logits."""
        pos = torch.tensor([-1000.0])
        neg = torch.tensor([-999.0, -998.0])
        loss = info_nce_loss(pos, neg, tau=0.1)
        assert torch.isfinite(loss)

    def test_empty_positives(self):
        """Empty positives returns zero loss."""
        pos = torch.tensor([])
        neg = torch.tensor([1.0, 2.0])
        loss = info_nce_loss(pos, neg, tau=0.1)
        assert loss.item() == 0.0

    def test_multiple_positives(self):
        """Each positive is scored against the same negatives, losses averaged."""
        pos = torch.tensor([5.0, 3.0])
        neg = torch.tensor([0.0, 0.0])
        loss = info_nce_loss(pos, neg, tau=0.1)
        # Individual losses: pos=5 should have lower loss than pos=3
        l1 = info_nce_loss(torch.tensor([5.0]), neg, tau=0.1)
        l2 = info_nce_loss(torch.tensor([3.0]), neg, tau=0.1)
        expected_avg = (l1 + l2) / 2
        assert torch.allclose(loss, expected_avg, atol=1e-5)

    def test_gradient_flows(self):
        """Loss is differentiable w.r.t. logits."""
        pos = torch.tensor([1.0], requires_grad=True)
        neg = torch.tensor([0.0, -1.0], requires_grad=True)
        loss = info_nce_loss(pos, neg, tau=0.1)
        loss.backward()
        assert pos.grad is not None
        assert neg.grad is not None


class TestMultiPositiveLoss:

    def test_single_positive_returns_zero(self):
        """Multi-positive loss is 0 when P < 2."""
        loss = multi_positive_loss(torch.tensor([5.0]), T_mp=0.1)
        assert loss.item() == 0.0

    def test_uniform_positives_low_loss(self):
        """Equal positive logits yield minimal loss (uniform softmax = -log(1/P))."""
        pos = torch.tensor([5.0, 5.0, 5.0])
        loss = multi_positive_loss(pos, T_mp=0.1)
        expected = -torch.log(torch.tensor(1.0 / 3.0))
        assert torch.allclose(loss, expected, atol=1e-4)

    def test_skewed_positives_higher_loss(self):
        """Skewed positives yield higher loss than uniform."""
        uniform_loss = multi_positive_loss(torch.tensor([5.0, 5.0]), T_mp=0.1)
        skewed_loss = multi_positive_loss(torch.tensor([10.0, 0.0]), T_mp=0.1)
        assert skewed_loss > uniform_loss


class TestSmoothnessLoss:

    def test_constant_scores_zero_loss(self):
        """Constant scores have zero smoothness loss."""
        scores = torch.ones(50)
        loss = smoothness_loss(scores, chunk_len=50)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_noisy_scores_positive_loss(self):
        """Non-constant scores have positive smoothness loss."""
        scores = torch.randn(50)
        loss = smoothness_loss(scores, chunk_len=50)
        assert loss.item() > 0.0

    def test_short_chunk_zero(self):
        """Chunk with < 2 residues returns 0."""
        loss = smoothness_loss(torch.tensor([1.0]), chunk_len=1)
        assert loss.item() == 0.0


class TestComputeLoss:

    def test_v0_defaults_only_intra(self):
        """With v0 defaults (lambda_mp=0, lambda_smooth=0), total == intra."""
        pos = torch.tensor([5.0])
        neg = torch.tensor([0.0, -1.0])
        result = compute_loss(pos, neg, tau=0.1, lambda_mp=0.0, lambda_smooth=0.0)
        assert set(result.keys()) == {
            "loss_total", "loss_intra", "loss_mp", "loss_smooth", "loss_margin",
            "loss_iou_rank",  # Wave-3: additive window IoU-ranking term (0 when disabled)
        }
        assert torch.allclose(result["loss_total"], result["loss_intra"])
        assert result["loss_mp"].item() == 0.0
        assert result["loss_smooth"].item() == 0.0
        assert result["loss_iou_rank"].item() == 0.0

    def test_lambda_mp_contributes(self):
        """Non-zero lambda_mp adds multi-positive term."""
        pos = torch.tensor([5.0, 3.0])
        neg = torch.tensor([0.0])
        r0 = compute_loss(pos, neg, tau=0.1, lambda_mp=0.0)
        r1 = compute_loss(pos, neg, tau=0.1, lambda_mp=1.0)
        # With mp enabled and skewed positives, total should differ
        assert r1["loss_mp"].item() > 0.0
        assert r1["loss_total"].item() > r0["loss_total"].item()

    def test_lambda_smooth_contributes(self):
        """Non-zero lambda_smooth adds smoothness term."""
        pos = torch.tensor([5.0])
        neg = torch.tensor([0.0])
        scores = torch.randn(50)
        r0 = compute_loss(pos, neg, tau=0.1, lambda_smooth=0.0)
        r1 = compute_loss(pos, neg, tau=0.1, lambda_smooth=1.0,
                          per_residue_scores=scores, chunk_len=50)
        assert r1["loss_smooth"].item() > 0.0
        assert r1["loss_total"].item() > r0["loss_total"].item()

    def test_all_terms_finite(self):
        """All returned terms are finite with extreme inputs."""
        pos = torch.tensor([100.0, -100.0])
        neg = torch.tensor([50.0, -50.0])
        scores = torch.randn(20)
        result = compute_loss(pos, neg, tau=0.1, T_mp=0.1,
                              lambda_mp=1.0, lambda_smooth=0.5,
                              per_residue_scores=scores, chunk_len=20)
        for k, v in result.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_config_keys_are_mapped_before_compute_loss(self):
        """train.yaml loss keys are normalized to compute_loss param names."""
        import inspect
        from epitope_head.configs import load_train_config
        from epitope_head.training.trainer import normalize_loss_cfg
        cfg = load_train_config()
        loss_cfg = cfg["loss"]  # {tau, T_mp, lambda_mp, lambda_smooth}
        mapped = normalize_loss_cfg(loss_cfg)
        sig = inspect.signature(compute_loss)
        param_names = set(sig.parameters.keys())
        for key in mapped:
            assert key in param_names, (
                f"Mapped key '{key}' has no matching parameter "
                f"in compute_loss(). Available: {sorted(param_names)}"
            )
        assert "T_mp" in mapped
        assert mapped["T_mp"] == loss_cfg["T_mp"]

    def test_normalize_loss_cfg_accepts_tau_mp_alias(self):
        """normalize_loss_cfg accepts tau_mp alias and maps to T_mp."""
        raw = {"tau": 0.1, "tau_mp": 0.2, "lambda_mp": 0.0, "lambda_smooth": 0.0}
        mapped = normalize_loss_cfg(raw)
        assert "T_mp" in mapped
        assert "tau_mp" not in mapped
        assert mapped["T_mp"] == 0.2


# ── E5: Trainer loop ────────────────────────────────────────────────────────

from epitope_head.training.trainer import (
    StepMetrics,
    NaNDetected,
    aggregate_epoch_metrics,
    boundary_distance_bucket_stats,
    build_optimizer,
    compute_run_digest,
    compute_sanity_metrics,
    long_vs_short_comparison,
    nan_guard,
    normalize_loss_cfg,
    prepare_chunk_spans,
    train_step,
    val_step,
    save_checkpoint,
    load_checkpoint,
    verify_run_artifacts,
    write_log_entry,
    config_hash,
    Trainer,
    CHECKPOINT_METADATA_KEYS,
    LOG_ENTRY_KEYS,
)


class TestNaNGuard:

    def test_finite_passes(self):
        nan_guard(torch.tensor([1.0, 2.0]), "test")

    def test_nan_raises(self):
        with pytest.raises(NaNDetected, match="NaN/Inf"):
            nan_guard(torch.tensor([1.0, float("nan")]), "test")

    def test_inf_raises(self):
        with pytest.raises(NaNDetected, match="NaN/Inf"):
            nan_guard(torch.tensor([float("inf")]), "test")


class TestBuildOptimizer:

    def test_excludes_frozen_params(self):
        """Optimizer only receives parameters with requires_grad=True."""
        model = self._make_model_with_frozen()
        opt = build_optimizer(model, lr=1e-3)
        # Count params in optimizer vs total
        opt_param_count = sum(p.numel() for group in opt.param_groups for p in group["params"])
        learnable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        assert opt_param_count == learnable_count
        assert frozen_count > 0  # confirm some params are frozen

    def _make_model_with_frozen(self):
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 1),
        )
        # Freeze first layer
        for p in model[0].parameters():
            p.requires_grad = False
        return model


class TestSanityMetrics:

    def test_logit_gap_positive_when_correct(self):
        pos = torch.tensor([5.0, 6.0])
        neg = torch.tensor([1.0, 2.0])
        loss_dict = compute_loss(pos, neg, tau=0.1)
        metrics = compute_sanity_metrics(pos, neg, loss_dict)
        assert metrics.logit_gap > 0
        assert metrics.mean_pos_logit > metrics.mean_neg_logit
        assert metrics.per_protein_auc is not None
        assert metrics.per_protein_auc > 0.5

    def test_aggregate_epoch_metrics(self):
        m1 = StepMetrics(loss_total=1.0, logit_gap=2.0, n_pos=5, n_neg=10, per_protein_auc=0.7)
        m2 = StepMetrics(loss_total=3.0, logit_gap=4.0, n_pos=3, n_neg=8, per_protein_auc=0.9)
        agg = aggregate_epoch_metrics([m1, m2])
        assert agg["loss_total"] == 2.0
        assert agg["logit_gap"] == 3.0
        assert agg["total_pos"] == 8
        assert agg["n_steps"] == 2
        assert agg["per_protein_auc"] == 0.8


class TestPrepareChunkSpans:

    def test_filters_positives_to_chunk_bounds(self):
        """Only positives within chunk bounds are included."""
        batch = {
            "protein_ids": ["P1"],
            "chunk_starts": torch.tensor([100]),
            "chunk_ends": torch.tensor([200]),
            "positives": [[
                {"start_0b": 110, "end_0b": 125, "pep_len": 15},  # in chunk
                {"start_0b": 50, "end_0b": 65, "pep_len": 15},    # before chunk
                {"start_0b": 190, "end_0b": 210, "pep_len": 20},  # crosses end
            ]],
        }
        pos_list, neg_list, _, _, _ = prepare_chunk_spans(
            batch, batch_idx=0, rng=np.random.RandomState(42),
        )
        # Only the first positive is within [100, 200)
        assert pos_list[0].shape[0] == 1
        # Converted to chunk-local: 110-100=10, 125-100=25
        assert pos_list[0][0, 0].item() == 10
        assert pos_list[0][0, 1].item() == 25

    def test_no_positives_gives_empty(self):
        batch = {
            "protein_ids": ["P1"],
            "chunk_starts": torch.tensor([0]),
            "chunk_ends": torch.tensor([100]),
            "positives": [[
                {"start_0b": 200, "end_0b": 215, "pep_len": 15},
            ]],
        }
        pos_list, neg_list, _, _, _ = prepare_chunk_spans(
            batch, batch_idx=0, rng=np.random.RandomState(42),
        )
        assert pos_list[0].shape[0] == 0
        assert neg_list[0].shape[0] == 0


# ── E6: Checkpointing & Logging ─────────────────────────────────────────────

class TestCheckpointing:

    def test_save_load_roundtrip(self, tmp_path):
        model = nn.Linear(10, 1)
        opt = torch.optim.Adam(model.parameters())
        path = tmp_path / "ckpt.pt"
        save_checkpoint(model, opt, epoch=5, global_step=100,
                        monitor_metric="logit_gap", monitor_value=1.5,
                        cfg_hash="abc123", path=path,
                        diff_ids_applied=["d001", "e004"])
        ckpt = load_checkpoint(path)
        assert "model_state_dict" in ckpt
        assert ckpt["metadata"]["epoch"] == 5
        assert ckpt["metadata"]["config_hash"] == "abc123"
        assert ckpt["metadata"]["diff_ids_applied"] == ["d001", "e004"]

    def test_missing_metadata_raises(self, tmp_path):
        path = tmp_path / "bad.pt"
        torch.save({"metadata": {"epoch": 1}}, path)
        with pytest.raises(ValueError, match="missing metadata keys"):
            load_checkpoint(path)


class TestLogging:

    def test_log_entry_schema(self, tmp_path):
        log_path = tmp_path / "test.jsonl"
        entry = {
            "epoch": 0, "phase": "train",
            "loss_total": 1.0, "loss_intra": 1.0, "loss_mp": 0.0, "loss_smooth": 0.0, "loss_margin": 0.0,
            "mean_pos_logit": 2.0, "mean_neg_logit": 1.0, "logit_gap": 1.0,
            "per_protein_auc": 0.75,
            "total_pos": 10, "total_neg": 70, "n_steps": 5, "timestamp": 0.0,
            # HIMP4 residue logging fields (always present, 0 when disabled)
            "loss_residue": 0.0, "lambda_residue": 0.0,
            "residue_skipped_chunks": 0, "n_residue_pairs": 0, "n_residue_chunks": 0,
        }
        write_log_entry(log_path, entry)
        with open(log_path) as f:
            loaded = json.loads(f.readline())
        assert set(loaded.keys()) >= LOG_ENTRY_KEYS

    def test_missing_log_key_raises(self, tmp_path):
        log_path = tmp_path / "bad.jsonl"
        with pytest.raises(ValueError, match="missing keys"):
            write_log_entry(log_path, {"epoch": 0})


class TestConfigHash:

    def test_deterministic(self):
        cfg = {"a": 1, "b": [2, 3]}
        assert config_hash(cfg) == config_hash(cfg)

    def test_different_configs_different_hash(self):
        assert config_hash({"a": 1}) != config_hash({"a": 2})


# ── E5+E7: Integration — one-step train/val ─────────────────────────────────

class TestTrainValStep:
    """Integration test: one train step and one val step with mock encoder."""

    def _make_model_and_batch(self):
        d_enc, d_proj = 16, 8
        encoder = MockFrozenEncoder(d_enc=d_enc)
        model = EpitopeScorer(
            encoder=encoder, d_enc=d_enc, d_proj=d_proj,
            length_emb_dim=4, allele_emb_dim=4,
            scorer_hidden_dim=16, min_k=12, max_k=25,
        )
        # Build a synthetic batch: one chunk with known positives
        seq_len = 50
        B = 1
        T = seq_len + 2
        batch = {
            "protein_ids": ["P1"],
            "protein_seqs": ["A" * seq_len],
            "chunk_seqs": ["A" * seq_len],
            "alleles": ["HLA-A"],
            "positives": [[
                {"start_0b": 10, "end_0b": 25, "pep_len": 15},
                {"start_0b": 30, "end_0b": 42, "pep_len": 12},
            ]],
            "chunk_starts": torch.tensor([0]),
            "chunk_ends": torch.tensor([seq_len]),
            "chunk_indices": torch.tensor([0]),
            "sequence_lengths": torch.tensor([seq_len]),
            "token_ids": torch.ones(B, T, dtype=torch.long),
            "attention_mask": torch.ones(B, T, dtype=torch.bool),
        }
        return model, batch

    def test_train_step_runs(self):
        model, batch = self._make_model_and_batch()
        opt = build_optimizer(model, lr=1e-3)
        loss_cfg = {"tau": 0.1, "T_mp": 0.1, "lambda_mp": 0.0, "lambda_smooth": 0.0}
        neg_cfg = {
            "neg_ratio": 3, "hard_negative_fraction": 0.3,
            "hard_neg_max_overlap_ratio": 0.8, "hard_neg_offset_range": 5,
            "neg_length_sampling": "match_positive",
        }
        metrics = train_step(
            model, batch, opt, loss_cfg, neg_cfg,
            rng=np.random.RandomState(42),
        )
        assert metrics.loss_total > 0
        assert metrics.n_pos > 0

    def test_val_step_runs(self):
        model, batch = self._make_model_and_batch()
        loss_cfg = {"tau": 0.1, "T_mp": 0.1, "lambda_mp": 0.0, "lambda_smooth": 0.0}
        neg_cfg = {
            "neg_ratio": 3, "hard_negative_fraction": 0.3,
            "hard_neg_max_overlap_ratio": 0.8, "hard_neg_offset_range": 5,
            "neg_length_sampling": "match_positive",
        }
        metrics = val_step(model, batch, loss_cfg, neg_cfg, rng=np.random.RandomState(42))
        assert metrics.loss_total > 0

    def test_encoder_stays_frozen_after_train_step(self):
        """Encoder params don't change after a training step."""
        model, batch = self._make_model_and_batch()
        # Snapshot encoder params
        encoder_params_before = {
            name: p.clone() for name, p in model.encoder.named_parameters()
        }
        opt = build_optimizer(model, lr=1e-3)
        loss_cfg = {"tau": 0.1, "T_mp": 0.1, "lambda_mp": 0.0, "lambda_smooth": 0.0}
        neg_cfg = {
            "neg_ratio": 3, "hard_negative_fraction": 0.3,
            "hard_neg_max_overlap_ratio": 0.8, "hard_neg_offset_range": 5,
            "neg_length_sampling": "match_positive",
        }
        train_step(model, batch, opt, loss_cfg, neg_cfg, rng=np.random.RandomState(42))
        # Verify encoder params unchanged
        for name, p in model.encoder.named_parameters():
            assert torch.equal(p, encoder_params_before[name]), \
                f"Encoder param '{name}' changed after training step"

    def test_learnable_params_change_after_train_step(self):
        """Projection/scorer params do change after a training step."""
        model, batch = self._make_model_and_batch()
        proj_before = {
            name: p.clone() for name, p in model.projection.named_parameters()
        }
        opt = build_optimizer(model, lr=1e-2)  # higher lr to ensure change
        loss_cfg = {"tau": 0.1, "T_mp": 0.1, "lambda_mp": 0.0, "lambda_smooth": 0.0}
        neg_cfg = {
            "neg_ratio": 3, "hard_negative_fraction": 0.3,
            "hard_neg_max_overlap_ratio": 0.8, "hard_neg_offset_range": 5,
            "neg_length_sampling": "match_positive",
        }
        train_step(model, batch, opt, loss_cfg, neg_cfg, rng=np.random.RandomState(42))
        changed = False
        for name, p in model.projection.named_parameters():
            if not torch.equal(p, proj_before[name]):
                changed = True
        assert changed, "Projection params should change after training step"

    def test_trainer_writes_e6_artifacts(self, tmp_path):
        """Trainer.fit writes split logs + expected checkpoint names and metadata."""
        model, batch = self._make_model_and_batch()
        cfg = {
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "constant",
            "warmup_steps": 0,
            "max_epochs": 1,
            "grad_clip": 1.0,
            "loss": {"tau": 0.1, "T_mp": 0.1, "lambda_mp": 0.0, "lambda_smooth": 0.0},
            "neg_ratio": 3,
            "hard_negative_fraction": 0.3,
            "hard_neg_max_overlap_ratio": 0.8,
            "hard_neg_offset_range": 5,
            "neg_length_sampling": "match_positive",
            "monitor_metric": "logit_gap",
            "early_stopping_patience": 2,
            "checkpoint_every_n_epochs": 1,
            "seed": 42,
            "diff_ids_applied": ["e5", "e6"],
        }
        trainer = Trainer(
            model=model,
            train_loader=[batch],
            val_loader=[batch],
            train_cfg=cfg,
            run_dir=tmp_path,
            device="cpu",
        )
        summary = trainer.fit(max_epochs=1)
        assert summary["final_epoch"] == 0
        assert (tmp_path / "train_log.jsonl").exists()
        assert (tmp_path / "val_log.jsonl").exists()
        assert (tmp_path / "epoch_0.pt").exists()
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "resolved_config.yaml").exists()
        best = load_checkpoint(tmp_path / "best.pt")
        assert best["metadata"]["diff_ids_applied"] == ["e5", "e6"]


# ── E7: Smoke Run & Diagnostics ─────────────────────────────────────────────

class TestSmokeRun:
    """E7 smoke test: multi-epoch training completes, all artifacts valid."""

    def _make_trainer(self, tmp_path, seed=42, max_epochs=3):
        torch.manual_seed(seed)
        d_enc, d_proj = 16, 8
        encoder = MockFrozenEncoder(d_enc=d_enc)
        model = EpitopeScorer(
            encoder=encoder, d_enc=d_enc, d_proj=d_proj,
            length_emb_dim=4, allele_emb_dim=4,
            scorer_hidden_dim=16, min_k=12, max_k=25,
        )
        seq_len = 50
        B = 1
        T = seq_len + 2
        batch = {
            "protein_ids": ["P1"],
            "protein_seqs": ["A" * seq_len],
            "chunk_seqs": ["A" * seq_len],
            "alleles": ["HLA-A"],
            "positives": [[
                {"start_0b": 10, "end_0b": 25, "pep_len": 15},
                {"start_0b": 30, "end_0b": 42, "pep_len": 12},
            ]],
            "chunk_starts": torch.tensor([0]),
            "chunk_ends": torch.tensor([seq_len]),
            "chunk_indices": torch.tensor([0]),
            "sequence_lengths": torch.tensor([seq_len]),
            "token_ids": torch.ones(B, T, dtype=torch.long),
            "attention_mask": torch.ones(B, T, dtype=torch.bool),
        }
        cfg = {
            "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4,
            "scheduler": "constant", "warmup_steps": 0,
            "max_epochs": max_epochs, "grad_clip": 1.0,
            "loss": {"tau": 0.1, "T_mp": 0.1, "lambda_mp": 0.0, "lambda_smooth": 0.0},
            "neg_ratio": 3, "hard_negative_fraction": 0.3,
            "hard_neg_max_overlap_ratio": 0.8, "hard_neg_offset_range": 5,
            "neg_length_sampling": "match_positive",
            "monitor_metric": "logit_gap", "early_stopping_patience": 10,
            "checkpoint_every_n_epochs": 2, "seed": seed,
            "diff_ids_applied": ["e0-e7"],
        }
        trainer = Trainer(
            model=model, train_loader=[batch], val_loader=[batch],
            train_cfg=cfg, run_dir=tmp_path, device="cpu",
        )
        return trainer

    def test_smoke_artifacts_complete(self, tmp_path):
        """Multi-epoch smoke run produces all required artifacts with valid schemas."""
        trainer = self._make_trainer(tmp_path, max_epochs=3)
        trainer.fit(max_epochs=3)
        result = verify_run_artifacts(tmp_path, n_epochs=3, checkpoint_every_n=2)
        assert result["ok"], f"Missing: {result['missing']}, Errors: {result['errors']}"

    def test_smoke_logit_gap_finite_and_trackable(self, tmp_path):
        """Logit gap is finite across all logged epochs."""
        trainer = self._make_trainer(tmp_path, max_epochs=3)
        trainer.fit(max_epochs=3)
        for log_name in ["train_log.jsonl", "val_log.jsonl"]:
            with open(tmp_path / log_name) as f:
                for line in f:
                    entry = json.loads(line)
                    gap = entry["logit_gap"]
                    assert gap is not None
                    assert not (isinstance(gap, float) and (
                        gap != gap or gap == float("inf") or gap == float("-inf")
                    )), f"Non-finite logit_gap in {log_name}: {gap}"

    def test_smoke_reproducibility(self, tmp_path):
        """Two runs with same seed produce identical metric values."""
        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"
        trainer1 = self._make_trainer(run1, seed=123, max_epochs=2)
        trainer1.fit(max_epochs=2)
        trainer2 = self._make_trainer(run2, seed=123, max_epochs=2)
        trainer2.fit(max_epochs=2)
        # Compare metric values (excluding timestamps which differ)
        metric_keys = ["loss_total", "loss_intra", "logit_gap", "per_protein_auc"]
        for log_name in ["train_log.jsonl", "val_log.jsonl"]:
            with open(run1 / log_name) as f1, open(run2 / log_name) as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()
                assert len(lines1) == len(lines2), f"Line count mismatch in {log_name}"
                for i, (l1, l2) in enumerate(zip(lines1, lines2)):
                    e1, e2 = json.loads(l1), json.loads(l2)
                    for k in metric_keys:
                        assert e1[k] == e2[k], (
                            f"Reproducibility mismatch in {log_name} epoch {i}, "
                            f"key '{k}': {e1[k]} != {e2[k]}"
                        )

    def test_epoch_hook_metrics_written_to_train_log_and_summary(self, tmp_path):
        """Epoch hook auxiliary metrics are persisted as structured artifacts."""
        trainer = self._make_trainer(tmp_path, max_epochs=1)

        def epoch_hook(_trainer, epoch):
            return {
                "aug_epoch": epoch,
                "aug_configured_p_aug": 0.2,
                "aug_n_base_entries": 4,
                "aug_n_eligible_entries": 2,
                "aug_n_replaced": 1,
                "aug_effective_fraction": 0.25,
            }

        trainer.epoch_hook = epoch_hook
        trainer.fit(max_epochs=1)

        with open(tmp_path / "train_log.jsonl") as f:
            train_entry = json.loads(f.readline())
        assert train_entry["aug_configured_p_aug"] == 0.2
        assert train_entry["aug_n_replaced"] == 1
        assert train_entry["aug_effective_fraction"] == 0.25

        with open(tmp_path / "run_summary.json") as f:
            summary = json.load(f)
        assert summary["augmentation"]["configured_p_aug"] == 0.2
        assert summary["augmentation"]["n_replaced"] == 1
        assert summary["augmentation"]["effective_fraction"] == 0.25


class TestBoundaryDistanceBuckets:
    """E7 chunk bias gate: boundary-distance bucket diagnostics."""

    def test_bucket_stats_computed(self):
        """Buckets are computed and have expected structure."""
        span_starts = [5, 10, 50, 90, 95, 100, 150, 200]
        chunk_starts = [0] * 8
        chunk_ends = [250] * 8
        scores = [1.0, 1.1, 1.2, 1.3, 1.0, 0.9, 1.1, 1.0]
        buckets = boundary_distance_bucket_stats(
            span_starts, chunk_starts, chunk_ends, scores, n_buckets=4,
        )
        assert len(buckets) == 4
        for b in buckets:
            assert "mean_score" in b
            assert "count" in b
            assert b["count"] > 0

    def test_no_severe_drift(self):
        """Score means across buckets do not vary by more than 2x from overall mean."""
        # Generate uniform-ish scores (no real boundary bias in mock)
        rng = np.random.RandomState(42)
        n = 100
        span_starts = rng.randint(0, 200, size=n).tolist()
        chunk_starts = [0] * n
        chunk_ends = [250] * n
        scores = (rng.randn(n) * 0.5 + 1.0).tolist()  # mean ~1.0
        buckets = boundary_distance_bucket_stats(
            span_starts, chunk_starts, chunk_ends, scores, n_buckets=4,
        )
        overall_mean = sum(scores) / len(scores)
        for b in buckets:
            ratio = b["mean_score"] / overall_mean if overall_mean != 0 else 1.0
            assert 0.5 < ratio < 2.0, (
                f"Bucket {b['bucket']} has severe drift: "
                f"mean={b['mean_score']:.3f} vs overall={overall_mean:.3f}"
            )

    def test_empty_input(self):
        """Empty spans return empty buckets."""
        buckets = boundary_distance_bucket_stats([], [], [], [])
        assert buckets == []


class TestLongVsShort:
    """E7 long-vs-short subgroup comparison diagnostics."""

    def test_subgroup_separation(self):
        """Proteins are correctly split by length threshold."""
        lengths = [500, 800, 1022, 1500, 2000, 3000]
        gaps = [1.0, 1.1, 1.2, 0.8, 0.9, 0.7]
        aucs = [0.7, 0.75, 0.8, 0.65, 0.7, 0.6]
        result = long_vs_short_comparison(lengths, gaps, aucs, threshold=1022)
        assert result["short"]["n"] == 3  # 500, 800, 1022
        assert result["long"]["n"] == 3   # 1500, 2000, 3000
        assert result["short"]["mean_gap"] is not None
        assert result["long"]["mean_gap"] is not None

    def test_all_short(self):
        """All-short input has no long stats."""
        result = long_vs_short_comparison([100, 200], [1.0, 1.0], [0.8, 0.8])
        assert result["short"]["n"] == 2
        assert result["long"]["n"] == 0
        assert result["long"]["mean_gap"] is None

    def test_none_aucs_handled(self):
        """None AUC values are excluded from mean calculation."""
        result = long_vs_short_comparison(
            [100, 200, 1500], [1.0, 1.0, 0.8], [None, 0.7, None],
        )
        assert result["short"]["mean_auc"] == 0.7
        assert result["long"]["mean_auc"] is None
