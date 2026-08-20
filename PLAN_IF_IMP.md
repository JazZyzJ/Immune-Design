# DPLM IF Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a MapDiff-grounded improvement path for DPLM inverse folding: an uncertainty-aware IPA refiner first, then a compatible geometry sidecar for DPLM structure conditioning.

**Architecture:** Keep DPLM as the frozen sequence prior and keep the existing GVP adapter path working by default. Port only the MapDiff components that match DPLM's batch contract: entropy-based mask selection, IPA masked refinement, entropy-weighted logit fusion, MC-dropout inference, and an IPA-derived geometry feature stream projected into the DPLM adapter dimension. Treat `MapDiff/` as a reference checkout, not as a runtime dependency.

**Tech Stack:** Python 3.12, PyTorch, DPLM/byprot, ESM-IF featurization, local adapted MapDiff IPA utilities, pytest, pandas/parquet, YAML, SLURM.

---

## Source Anchors

Use these concrete source points while implementing. Do not replace them with memory-based assumptions.

- DPLM generation hook: `inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`
  - `forward_decoder()` computes decoder logits at lines 201-211, bans special tokens (`mask_id`/`x_id`/`pad_id`/`bos_id`/`eos_id`) at lines 213-217, samples at lines 219-225 (argmax line 220; gumbel_argmax lines 221-225), and writes back to `output_tokens`/`output_scores` at lines 227-228.
  - `generate()` lines 399-486; `forward_decoder()` is called once per outer step at line 443; `_reparam_decoding()` runs at lines 462-473.
  - Best refiner insertion point is inside `forward_decoder()` after line 217 (all five special-token bans applied) and before line 219 (sampling branch).
  - `_reparam_decoding()` (lines 283-397) does NOT wipe the refiner's effect: it re-masks low-score positions back to `mask_id` and preserves high-score positions. The refiner's logit modifications survive iterative decoding because (a) high-score processor outputs are kept by the reparam topk, (b) low-score positions get re-masked and routed back into the next `forward_decoder` call where the processor runs again. The refiner is therefore applied per step, not once.
- DPLM batch contract: `inverse_folding/dplm/src/byprot/datamodules/dataset/cath.py`
  - `collate_batch()` emits `coords`, `tokens`, `confidence`, `coord_mask`, `lengths`, `seqs`, `names` at lines 143-187.
  - Atom order is `("N", "CA", "C", "O")` at lines 143-148.
- Reusable DPLM runtime helpers: `inverse_folding/reference_flow/runtime.py`
  - `prepare_backbone()` validates IF-ready sequence length against `load_coords()` output at lines 145-182.
  - `build_dplm_denoiser_context()` builds residue masks as `coord_mask & ~special_sym_mask` at lines 218-247.
  - `make_dplm_denoiser()` strips special-token logits and returns residue-position logits at lines 250-279.
- MapDiff IPA refiner: `MapDiff/model/ipa/ipa_net.py`
  - `IPANetPredictor.forward(x_aa, x_pos, x_aa_mask, seq_mask)` returns `[B, L, 20]` logits at lines 111-134.
  - It builds rigid frames from `[N, CA, C]`, node features from AA one-hot + mask embedding + dihedrals + position encoding, and pair features from first four atoms.
- MapDiff entropy and fusion: `MapDiff/utils.py`
  - `get_entropy()` uses `-mean(p * log p)`, not standard summed entropy, at lines 222-226.
  - `fuse_logits_by_log_probs()` entropy-weights base/refiner logits at lines 229-238.
  - `sin_mask_ratio_adapter()` computes `center + sin(beta_t_bar * pi / 2) * max_deviation` at lines 241-246.
  - `place_missing_cb()` constructs virtual CB at lines 188-196.
- MapDiff denoising loop: `MapDiff/model/prior_diff.py`
  - `sample_p_zs_given_zt()` computes base logits, entropy mask, IPA prior logits, entropy fusion, and sampling at lines 159-232.
  - `mc_ddim_sample()` iterates skipped denoising steps at lines 234-250.

## Non-Negotiable Constraints

- Keep current DPLM behavior bitwise-equivalent when no refiner/sidecar config is supplied.
- Do not import `MapDiff` modules in production code. Copy/adapt minimal IPA utilities into `inverse_folding/` with attribution comments.
- Do not copy `MapDiff/model/prior_diff.py` wholesale. DPLM is already an iterative masked denoiser; MapDiff's 20-class discrete transition posterior is not the DPLM sampling process.
- Do not mutate caller-owned batches in refiner or sidecar code. Clone tensor fields before edits, following `inverse_folding/reference_flow/runtime.py::clone_batch()`.
- Residue operations must use `coord_mask & ~special_sym_mask`; never treat BOS/EOS/pad as residues.
- DPLM coords are `[B, L, 4, 3]` in `[N, CA, C, O]`; IPA expects `[B, L, 5, 3]` in `[N, CA, C, CB, O]`. Construct CB explicitly.
- Special-token logits must remain `-inf` before sampling.
- When `use_draft_seq=True`, the initial token state at step 0 is encoder-supplied `init_pred`, not `mask_id`. The refiner's entropy mask therefore selects low-confidence positions inside an already-decoded draft at step 0; this is intentional. The no-refiner default path must remain bit-equivalent regardless of draft mode.
- All cluster paths must be CLI args or environment variables passed into scripts. No hardcoded `/scratch/...` paths in Python modules.
- Any new script or SLURM file must be registered in `doc/SCRIPTS.md`.
- Append a structured `LOG.md` entry after completing this implementation.

## File Structure

Create this package:

- `inverse_folding/dplm_refiner/__init__.py`  
  Public exports for the refiner package.
- `inverse_folding/dplm_refiner/tokens.py`  
  Canonical amino-acid ordering and DPLM token-id bridge.
- `inverse_folding/dplm_refiner/geometry.py`  
  DPLM-to-IPA coordinate conversion, CB construction, residue masks.
- `inverse_folding/dplm_refiner/entropy.py`  
  MapDiff entropy, sine mask-ratio schedule, entropy top-quantile mask selection, dropout enabling.
- `inverse_folding/dplm_refiner/fusion.py`  
  Entropy-weighted base/refiner logit fusion.
- `inverse_folding/dplm_refiner/ipa/rigid_utils.py`  
  Adapted from `MapDiff/model/ipa/rigid_utils.py`.
- `inverse_folding/dplm_refiner/ipa/ipa_utils.py`  
  Adapted from `MapDiff/model/ipa/ipa_utils.py`; keep only utilities needed by IPA refiner.
- `inverse_folding/dplm_refiner/ipa/ipa_attn.py`  
  Adapted from `MapDiff/model/ipa/ipa_attn.py`.
- `inverse_folding/dplm_refiner/ipa/refiner.py`  
  DPLM-compatible `DPLMIPARefiner`.
- `inverse_folding/dplm_refiner/logit_processor.py`  
  `DPLMRefinerLogitProcessor` callable for decoder-time refinement.
- `inverse_folding/dplm_refiner/sidecar.py`  
  IPA-derived geometry sidecar returning `[B, L, 512]` adapter-compatible features.
- `inverse_folding/dplm_refiner/config.py`  
  Dataclasses and validation for refiner/sidecar settings.
- `inverse_folding/dplm_refiner/checkpoint.py`  
  Load/save refiner and sidecar checkpoints with explicit config payloads.
- `inverse_folding/dplm_refiner/training.py`  
  Training/evaluation helpers for IPA refiner pretraining on CATH.

Modify existing files:

- `inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`  
  Add optional `logit_processor` argument to `forward_decoder()` and `generate()`, default `None`.
- `inverse_folding/dplm/src/byprot/models/dplm/modules/gvp_transformer_encoder.py`  
  Leave default output unchanged. Add sidecar integration only if the sidecar task reaches that milestone.
- `inverse_folding/reference_flow/runtime.py`  
  Reuse existing helpers; add refiner-aware loader helpers only after the refiner processor is tested.
- `doc/SCRIPTS.md`  
  Register new scripts under a new `Inverse Folding -- IF Improvement (PLAN_IF_IMP.md)` section.

Create scripts only after checking whether existing scripts can be extended:

- `scripts/train_if_imp_refiner.py`  
  Train the IPA refiner on CATH with DPLM-compatible batches.
- `scripts/run_if_imp_refiner.py`  
  Generate IF designs with optional refiner checkpoint and optional MC-dropout aggregation.
- `scripts/submit_if_imp.slurm`  
  Parameterized launcher with `MODE={train_refiner,generate_refiner}`.

Create tests:

- `tests/inverse_folding/dplm_refiner/test_tokens.py`
- `tests/inverse_folding/dplm_refiner/test_geometry.py`
- `tests/inverse_folding/dplm_refiner/test_entropy.py`
- `tests/inverse_folding/dplm_refiner/test_fusion.py`
- `tests/inverse_folding/dplm_refiner/test_ipa_refiner.py`
- `tests/inverse_folding/dplm_refiner/test_logit_processor.py`
- `tests/inverse_folding/dplm_refiner/test_dplm_hook.py`
- `tests/inverse_folding/dplm_refiner/test_training.py`
- `tests/inverse_folding/dplm_refiner/test_sidecar.py`
- `tests/scripts/test_if_imp_scripts.py`

## Task 1: Token Bridge

**Files:**
- Create: `inverse_folding/dplm_refiner/__init__.py`
- Create: `inverse_folding/dplm_refiner/tokens.py`
- Test: `tests/inverse_folding/dplm_refiner/test_tokens.py`

- [ ] **Step 1: Write failing tests for canonical AA mapping**

```python
import torch

from inverse_folding.dplm_refiner.tokens import (
    CANONICAL_AA_ORDER,
    DPLMTokenBridge,
)


class DummyAlphabet:
    padding_idx = 1
    cls_idx = 0
    eos_idx = 2
    mask_idx = 32
    unk_idx = 3

    def __init__(self):
        self._tok_to_idx = {aa: i + 4 for i, aa in enumerate(CANONICAL_AA_ORDER)}
        self._idx_to_tok = {idx: tok for tok, idx in self._tok_to_idx.items()}

    def get_idx(self, token):
        return self._tok_to_idx[token]

    def get_tok(self, idx):
        return self._idx_to_tok[idx]


def test_token_bridge_extracts_canonical_logits_in_stable_order():
    alphabet = DummyAlphabet()
    bridge = DPLMTokenBridge.from_alphabet(alphabet)
    logits = torch.full((1, 2, 40), -100.0)
    for aa_i, token_i in enumerate(bridge.aa_token_ids.tolist()):
        logits[..., token_i] = aa_i

    aa_logits = bridge.to_aa_logits(logits)

    assert aa_logits.shape == (1, 2, 20)
    assert aa_logits[0, 0].tolist() == [float(i) for i in range(20)]


def test_token_bridge_scatter_aa_logits_back_to_dplm_vocab_preserves_special_bans():
    alphabet = DummyAlphabet()
    bridge = DPLMTokenBridge.from_alphabet(alphabet)
    base_logits = torch.zeros((1, 2, 40))
    aa_logits = torch.arange(40, dtype=torch.float32).view(1, 2, 20)

    full = bridge.scatter_aa_logits(aa_logits, base_logits)

    assert full.shape == base_logits.shape
    assert torch.isneginf(full[..., alphabet.padding_idx]).all()
    assert torch.isneginf(full[..., alphabet.cls_idx]).all()
    assert torch.isneginf(full[..., alphabet.eos_idx]).all()
    assert torch.isneginf(full[..., alphabet.mask_idx]).all()
    assert torch.isneginf(full[..., alphabet.unk_idx]).all()
    assert full[0, 1, alphabet.get_idx("Y")].item() == 39.0
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_tokens.py -v
```

Expected: fail because `inverse_folding.dplm_refiner.tokens` does not exist.

- [ ] **Step 3: Implement token bridge**

Use these public names:

```python
CANONICAL_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

@dataclass(frozen=True)
class DPLMTokenBridge:
    aa_token_ids: torch.LongTensor
    banned_token_ids: torch.LongTensor

    @classmethod
    def from_alphabet(
        cls,
        alphabet: Any,
        *,
        extra_banned_token_ids: Sequence[int] = (),
    ) -> "DPLMTokenBridge": ...
    def to_aa_logits(self, dplm_logits: torch.Tensor) -> torch.Tensor: ...
    def to_aa_tokens(self, dplm_tokens: torch.Tensor) -> torch.Tensor: ...
    def from_aa_tokens(self, aa_tokens: torch.Tensor) -> torch.Tensor: ...
    def one_hot_from_dplm_tokens(self, dplm_tokens: torch.Tensor) -> torch.Tensor: ...
    def scatter_aa_logits(self, aa_logits: torch.Tensor, template_logits: torch.Tensor) -> torch.Tensor: ...
```

Implementation rules:

- `CANONICAL_AA_ORDER` is exactly `"ACDEFGHIKLMNPQRSTVWY"`.
- `from_alphabet()` must collect `alphabet.get_idx(aa)` for each canonical AA.
- `banned_token_ids` must include available `padding_idx`, `cls_idx`, `eos_idx`, `mask_idx`, `unk_idx`, and DPLM `x_id` if the caller later passes it through config.
- `scatter_aa_logits()` clones `template_logits`, fills canonical AA columns from `aa_logits`, then fills banned columns with `-torch.inf`.

- [ ] **Step 4: Run token tests**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_tokens.py -v
```

Expected: both tests pass.

## Task 2: Geometry Bridge

**Files:**
- Create: `inverse_folding/dplm_refiner/geometry.py`
- Test: `tests/inverse_folding/dplm_refiner/test_geometry.py`

- [ ] **Step 1: Write failing tests for DPLM-to-IPA coordinates**

```python
import torch

from inverse_folding.dplm_refiner.geometry import (
    dplm_coords_to_ipa_positions,
    place_virtual_cb,
)


def test_place_virtual_cb_returns_finite_cb_for_valid_backbone():
    n = torch.tensor([[0.0, 1.0, 0.0]])
    ca = torch.tensor([[0.0, 0.0, 0.0]])
    c = torch.tensor([[1.5, 0.0, 0.0]])

    cb = place_virtual_cb(n=n, ca=ca, c=c)

    assert cb.shape == (1, 3)
    assert torch.isfinite(cb).all()
    assert torch.linalg.norm(cb - ca, dim=-1).item() > 0.5


def test_dplm_coords_to_ipa_positions_reorders_atoms_and_masks_invalid_rows():
    coords = torch.zeros((1, 3, 4, 3))
    coords[0, :, 0] = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [float("nan"), 0.0, 0.0]])
    coords[0, :, 1] = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    coords[0, :, 2] = torch.tensor([[1.5, 0.0, 0.0], [2.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    coords[0, :, 3] = torch.tensor([[2.0, 0.5, 0.0], [3.0, 0.5, 0.0], [2.0, 0.0, 0.0]])
    coord_mask = torch.tensor([[True, True, False]])
    special_mask = torch.tensor([[False, True, False]])

    out = dplm_coords_to_ipa_positions(coords, coord_mask, special_mask)

    assert out.atom_pos.shape == (1, 3, 5, 3)
    assert out.seq_mask.tolist() == [[True, False, False]]
    assert torch.allclose(out.atom_pos[0, 0, 0], coords[0, 0, 0])
    assert torch.allclose(out.atom_pos[0, 0, 2], coords[0, 0, 2])
    assert torch.allclose(out.atom_pos[0, 0, 4], coords[0, 0, 3])
    assert torch.isfinite(out.atom_pos).all()
    assert out.atom_pos[0, 1].abs().sum().item() == 0.0
```

- [ ] **Step 2: Run geometry tests and verify they fail**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_geometry.py -v
```

Expected: fail because `geometry.py` does not exist.

- [ ] **Step 3: Implement geometry bridge**

Use these public names:

```python
@dataclass(frozen=True)
class IPAAtomPositions:
    atom_pos: torch.Tensor  # [B, L, 5, 3], [N, CA, C, CB, O]
    seq_mask: torch.Tensor  # [B, L], true only for real residues


def place_virtual_cb(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor) -> torch.Tensor: ...


def dplm_coords_to_ipa_positions(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    special_sym_mask: torch.Tensor,
) -> IPAAtomPositions: ...
```

Implementation rules:

- Port the CB construction math from `MapDiff/utils.py::place_fourth_atom()` and `place_missing_cb()`. Source anchor: `MapDiff/utils.py:188-196` calls `place_fourth_atom(C, N, CA, 1.522, 1.927, -2.143)`.
- Use constants from MapDiff code: length `1.522`, planar `1.927`, dihedral `-2.143`. Keep them as named module-level constants in `geometry.py`; do not inline literal numbers at the call site.
- DPLM atom order is `[N, CA, C, O]`; output order is `[N, CA, C, CB, O]`.
- `seq_mask = coord_mask & ~special_sym_mask & torch.isfinite(coords).all(dim=(-1, -2))`.
- Replace invalid atom rows with zeros after computing `seq_mask`.
- Do not mutate input `coords`.

- [ ] **Step 4: Run geometry tests**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_geometry.py -v
```

Expected: all tests pass.

## Task 3: Entropy, Mask Selection, Dropout

**Files:**
- Create: `inverse_folding/dplm_refiner/entropy.py`
- Test: `tests/inverse_folding/dplm_refiner/test_entropy.py`

- [ ] **Step 1: Write failing entropy tests**

```python
import torch
from torch import nn

from inverse_folding.dplm_refiner.entropy import (
    enable_dropout_modules,
    mapdiff_entropy_from_log_probs,
    select_entropy_mask,
    sine_mask_ratio,
)


def test_mapdiff_entropy_matches_source_mean_definition():
    log_probs = torch.log_softmax(torch.tensor([[2.0, 0.0], [0.0, 0.0]]), dim=-1)
    probs = log_probs.exp()
    expected = -(probs * log_probs).mean(dim=-1)

    actual = mapdiff_entropy_from_log_probs(log_probs)

    assert torch.allclose(actual, expected)


def test_sine_mask_ratio_uses_mapdiff_formula():
    beta_t_bar = torch.tensor([[0.0], [1.0]])

    ratios = sine_mask_ratio(beta_t_bar, center=0.4, max_deviation=0.2)

    assert torch.allclose(ratios, torch.tensor([0.4, 0.6]), atol=1e-6)


def test_select_entropy_mask_respects_valid_mask_per_batch():
    entropy = torch.tensor([[0.1, 0.4, 0.3, 0.2], [0.9, 0.8, 0.1, 0.0]])
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    ratios = torch.tensor([0.5, 0.5])

    mask = select_entropy_mask(entropy, valid, ratios)

    assert mask.tolist() == [[False, True, True, False], [True, False, False, False]]


def test_enable_dropout_modules_only_switches_dropout_to_train():
    model = nn.Sequential(nn.Linear(3, 3), nn.Dropout(0.5))
    model.eval()

    changed = enable_dropout_modules(model)

    assert changed == 1
    assert model[0].training is False
    assert model[1].training is True
```

- [ ] **Step 2: Run entropy tests and verify they fail**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_entropy.py -v
```

Expected: fail because `entropy.py` does not exist.

- [ ] **Step 3: Implement entropy utilities**

Use these public names:

```python
def mapdiff_entropy_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor: ...

def sine_mask_ratio(
    beta_t_bar: torch.Tensor,
    *,
    center: float = 0.4,
    max_deviation: float = 0.2,
) -> torch.Tensor: ...

def select_entropy_mask(
    entropy: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_ratios: torch.Tensor,
) -> torch.Tensor: ...

def enable_dropout_modules(model: torch.nn.Module) -> int: ...
```

Implementation rules:

- Entropy must match MapDiff source: `-(log_probs.exp() * log_probs).mean(dim=-1)`.
- `sine_mask_ratio()` returns `[B]` from `[B, 1]` or `[B]`.
- Clamp mask ratios to `[0.0, 1.0]`.
- `select_entropy_mask()` must select `ceil(valid_count * ratio)` highest-entropy valid positions per batch.
- If `valid_count == 0` or selected count is zero, return all false for that row.
- `enable_dropout_modules()` switches only modules whose class name starts with `Dropout` and returns the number switched. Note: this is a wider contract than `MapDiff/utils.py::enable_dropout` (which has no return value); the count return is what enables the test in `test_entropy.py` to assert idempotency. Document the divergence in the file's docstring.

- [ ] **Step 4: Run entropy tests**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_entropy.py -v
```

Expected: all tests pass.

## Task 4: Entropy-Weighted Fusion

**Files:**
- Create: `inverse_folding/dplm_refiner/fusion.py`
- Test: `tests/inverse_folding/dplm_refiner/test_fusion.py`

- [ ] **Step 1: Write failing fusion tests**

```python
import torch

from inverse_folding.dplm_refiner.fusion import fuse_logits_by_entropy


def test_fusion_prefers_lower_entropy_logits():
    base_logits = torch.tensor([[[4.0, 0.0], [0.0, 0.0]]])
    refiner_logits = torch.tensor([[[0.0, 0.0], [5.0, -5.0]]])
    valid = torch.tensor([[True, True]])

    fused = fuse_logits_by_entropy(base_logits, refiner_logits, valid_mask=valid, temperature=1.0)

    assert fused.shape == base_logits.shape
    assert fused[0, 0, 0] > fused[0, 0, 1]
    assert fused[0, 1, 0] > fused[0, 1, 1]


def test_fusion_leaves_invalid_positions_at_base_logits():
    base_logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    refiner_logits = torch.tensor([[[9.0, 9.0], [8.0, 8.0]]])
    valid = torch.tensor([[True, False]])

    fused = fuse_logits_by_entropy(base_logits, refiner_logits, valid_mask=valid)

    assert torch.allclose(fused[0, 1], base_logits[0, 1])
```

- [ ] **Step 2: Run fusion tests and verify they fail**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_fusion.py -v
```

Expected: fail because `fusion.py` does not exist.

- [ ] **Step 3: Implement fusion**

Use this public function:

```python
def fuse_logits_by_entropy(
    base_logits: torch.Tensor,
    refiner_logits: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor: ...
```

Implementation rules:

- Compute `base_log_probs = log_softmax(base_logits, dim=-1)` and same for refiner.
- Compute MapDiff entropy for both.
- Stack entropy as `[2, B, L]`; weights are `softmax(-entropy / temperature, dim=0)`.
- Stack logits as `[2, B, L, 20]`; fused is weighted sum.
- For invalid positions, return original `base_logits`.

- [ ] **Step 4: Run fusion tests**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_fusion.py -v
```

Expected: all tests pass.

## Task 5: Port IPA Refiner

**Files:**
- Create: `inverse_folding/dplm_refiner/ipa/rigid_utils.py`
- Create: `inverse_folding/dplm_refiner/ipa/ipa_utils.py`
- Create: `inverse_folding/dplm_refiner/ipa/ipa_attn.py`
- Create: `inverse_folding/dplm_refiner/ipa/refiner.py`
- Test: `tests/inverse_folding/dplm_refiner/test_ipa_refiner.py`

- [ ] **Step 1: Copy minimal MapDiff IPA utilities with attribution**

Copy/adapt only the required components from:

- `MapDiff/model/ipa/rigid_utils.py`
- `MapDiff/model/ipa/ipa_utils.py`
- `MapDiff/model/ipa/ipa_attn.py`
- `MapDiff/model/ipa/ipa_net.py`

Add this header to adapted files:

```python
"""IPA components adapted from MapDiff.

Source: https://github.com/peizhenbai/MapDiff
Local reference checkout: /Users/jerry/Project/MHC-IF/MapDiff
Adaptation: package-local imports, DPLM batch contract, explicit masks.
"""
```

Package-local import rule:

```python
from inverse_folding.dplm_refiner.ipa.rigid_utils import Rigid
from inverse_folding.dplm_refiner.ipa.ipa_utils import cal_dihedrals, cal_pair_rbf
```

- [ ] **Step 2: Write failing IPA refiner tests**

```python
import torch

from inverse_folding.dplm_refiner.ipa.refiner import DPLMIPARefiner


def test_ipa_refiner_forward_shape_and_invalid_mask_zero_grad():
    model = DPLMIPARefiner(hidden_dim=32, ipa_pairwise_dim=32, ipa_heads=4, ipa_depth=2, dropout=0.0)
    x_aa = torch.zeros((2, 5, 20))
    x_aa[..., 0] = 1.0
    x_pos = torch.randn((2, 5, 5, 3))
    x_aa_mask = torch.zeros((2, 5), dtype=torch.long)
    x_aa_mask[:, 1] = 1
    seq_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )

    logits = model(x_aa=x_aa, x_pos=x_pos, x_aa_mask=x_aa_mask, seq_mask=seq_mask)

    assert logits.shape == (2, 5, 20)
    assert torch.isfinite(logits[seq_mask]).all()
    assert torch.allclose(logits[~seq_mask], torch.zeros_like(logits[~seq_mask]), atol=1e-6)


def test_ipa_refiner_rejects_wrong_atom_count():
    model = DPLMIPARefiner(hidden_dim=32, ipa_pairwise_dim=32, ipa_heads=4, ipa_depth=2)
    x_aa = torch.zeros((1, 4, 20))
    x_pos = torch.randn((1, 4, 4, 3))
    x_aa_mask = torch.zeros((1, 4), dtype=torch.long)
    seq_mask = torch.ones((1, 4), dtype=torch.bool)

    try:
        model(x_aa=x_aa, x_pos=x_pos, x_aa_mask=x_aa_mask, seq_mask=seq_mask)
    except ValueError as exc:
        assert "x_pos" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 3: Run IPA tests and verify they fail**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_ipa_refiner.py -v
```

Expected: fail until the adapted IPA modules and `DPLMIPARefiner` exist.

- [ ] **Step 4: Implement `DPLMIPARefiner`**

Public constructor:

```python
class DPLMIPARefiner(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        ipa_pairwise_dim: int = 128,
        ipa_heads: int = 4,
        ipa_depth: int = 6,
        ipa_qk_points: int = 4,
        ipa_v_points: int = 8,
        dropout: float = 0.2,
    ) -> None: ...

    def forward(
        self,
        *,
        x_aa: torch.Tensor,
        x_pos: torch.Tensor,
        x_aa_mask: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor: ...
```

Implementation rules:

- Match MapDiff `IPANetPredictor.forward()` semantics: `x_aa [B,L,20]`, `x_pos [B,L,5,3]`, `x_aa_mask [B,L]`, `seq_mask [B,L]`, output `[B,L,20]`.
- Use `Rigid.from_3_points(x_pos[:, :, 0], x_pos[:, :, 1], x_pos[:, :, 2])`.
- Preserve MapDiff's node and edge encoders: AA projection, mask embedding when `x_aa_mask == 1`, dihedral features, sinusoidal position encoding, pair RBF over the first four atoms.
- Zero logits where `seq_mask` is false.
- Validate shapes explicitly and raise `ValueError` with argument names.

- [ ] **Step 5: Run IPA tests**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_ipa_refiner.py -v
```

Expected: all tests pass.

## Task 6: Decoder-Time Refiner Logit Processor

**Files:**
- Create: `inverse_folding/dplm_refiner/config.py`
- Create: `inverse_folding/dplm_refiner/logit_processor.py`
- Test: `tests/inverse_folding/dplm_refiner/test_logit_processor.py`

- [ ] **Step 1: Write failing processor tests**

```python
import torch

from inverse_folding.dplm_refiner.config import DPLMRefinerConfig
from inverse_folding.dplm_refiner.logit_processor import DPLMRefinerLogitProcessor
from inverse_folding.dplm_refiner.tokens import CANONICAL_AA_ORDER


class DummyAlphabet:
    padding_idx = 1
    cls_idx = 0
    eos_idx = 2
    mask_idx = 32
    unk_idx = 3

    def __init__(self):
        self._tok_to_idx = {aa: i + 4 for i, aa in enumerate(CANONICAL_AA_ORDER)}
        self._idx_to_tok = {idx: tok for tok, idx in self._tok_to_idx.items()}

    def get_idx(self, token):
        return self._tok_to_idx[token]

    def get_tok(self, idx):
        return self._idx_to_tok[idx]


class ConstantRefiner(torch.nn.Module):
    def forward(self, *, x_aa, x_pos, x_aa_mask, seq_mask):
        logits = torch.zeros((*x_aa.shape[:2], 20), device=x_aa.device)
        logits[..., 0] = 10.0
        logits[~seq_mask] = 0.0
        return logits


def test_logit_processor_only_changes_valid_high_entropy_residue_positions():
    alphabet = DummyAlphabet()
    processor = DPLMRefinerLogitProcessor(
        refiner=ConstantRefiner(),
        alphabet=alphabet,
        config=DPLMRefinerConfig(mask_ratio_center=0.5, mask_ratio_deviation=0.0, fusion_temperature=1.0),
    )
    logits = torch.zeros((1, 5, 40))
    tokens = torch.full((1, 5), alphabet.mask_idx)
    tokens[0, 0] = alphabet.cls_idx
    tokens[0, 4] = alphabet.eos_idx
    coords = torch.randn((1, 5, 4, 3))
    coord_mask = torch.tensor([[False, True, True, True, False]])
    batch = {"coords": coords, "coord_mask": coord_mask, "tokens": tokens}

    out = processor(logits=logits, output_tokens=tokens, batch=batch, step=1, max_step=10)

    assert out.shape == logits.shape
    assert torch.allclose(out[0, 0], logits[0, 0])
    assert torch.allclose(out[0, 4], logits[0, 4])
    assert out[0, 1, alphabet.get_idx("A")] > out[0, 1, alphabet.get_idx("C")]
    assert torch.isneginf(out[..., alphabet.mask_idx]).all()
```

- [ ] **Step 2: Run processor tests and verify they fail**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_logit_processor.py -v
```

Expected: fail because config and processor do not exist.

- [ ] **Step 3: Implement refiner config**

Use:

```python
@dataclass(frozen=True)
class DPLMRefinerConfig:
    enabled: bool = True
    mask_ratio_center: float = 0.4
    mask_ratio_deviation: float = 0.2
    fusion_temperature: float = 1.0
    mc_dropout_passes: int = 1

    def validate(self) -> None: ...
```

Validation rules:

- `0.0 <= mask_ratio_center <= 1.0`
- `0.0 <= mask_ratio_deviation <= 1.0`
- `mask_ratio_center + mask_ratio_deviation <= 1.0`
- `fusion_temperature > 0.0`
- `mc_dropout_passes >= 1`

Default rationale (do not silently change without updating this list):

- `mask_ratio_center=0.4` matches MapDiff's effective mask-prior config default (`MapDiff/conf/mask_prior/default.yaml`), even though the utility fallback in `MapDiff/utils.py::sin_mask_ratio_adapter` is `center=0.5`.
- `mask_ratio_deviation=0.2` matches `MapDiff/conf/mask_prior/default.yaml`; note that `Prior_Diff.__init__` has an older Python fallback of `0.1`.
- `fusion_temperature=1.0` matches MapDiff source.
- `mc_dropout_passes=1` (config default); ablation sweep `{5, 10, 50}` is driven by CLI flag, not the dataclass default.

- [ ] **Step 4: Implement `DPLMRefinerLogitProcessor`**

Public call signature:

```python
class DPLMRefinerLogitProcessor:
    def __call__(
        self,
        *,
        logits: torch.Tensor,
        output_tokens: torch.Tensor,
        batch: dict[str, Any],
        step: int,
        max_step: int,
    ) -> torch.Tensor: ...
```

Implementation rules:

- Build `special_sym_mask` from `batch["tokens"]` using bridge banned ids for pad/BOS/EOS.
- Build `IPAAtomPositions` from `batch["coords"]`, `batch["coord_mask"]`, and `special_sym_mask`.
- Convert current `output_tokens` into `[B,L,20]` one-hot. For masked/special positions, fill zeros before passing to IPA.
- Convert `logits` into AA logits with `DPLMTokenBridge.to_aa_logits()`.
- Compute entropy from `log_softmax(aa_logits, dim=-1)`.
- Use `beta_t_bar = torch.full((B, 1), step / max_step, device=logits.device)` for the first DPLM-compatible schedule.
- Select high-entropy valid positions with `sine_mask_ratio()`.
- Set `x_aa_mask = selected_mask.long()`.
- Run the refiner `mc_dropout_passes` times when configured; average refiner AA logits before fusion.
- Fuse full AA logits with `fuse_logits_by_entropy()`, then scatter back to DPLM vocab with special logits banned.
- Positions outside `seq_mask` must keep original logits except for the existing special-token bans.

- [ ] **Step 5: Run processor tests**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_logit_processor.py -v
```

Expected: all tests pass.

## Task 7: Add Optional DPLM Decoder Hook

**Files:**
- Modify: `inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py`
- Test: `tests/inverse_folding/dplm_refiner/test_dplm_hook.py`

- [ ] **Step 1: Write failing hook tests**

```python
import torch


def test_forward_decoder_accepts_logit_processor_without_changing_default_signature():
    from byprot.models.dplm.dplm_invfold import DPLMInvFold

    assert "logit_processor" in DPLMInvFold.forward_decoder.__code__.co_varnames
    assert "logit_processor" in DPLMInvFold.generate.__code__.co_varnames
```

Use this test as an import/signature guard only; avoid instantiating full DPLM in unit tests.

- [ ] **Step 2: Run hook test and verify it fails**

Run:

```bash
PYTHONPATH=inverse_folding/dplm/src pytest tests/inverse_folding/dplm_refiner/test_dplm_hook.py -v
```

Expected: fail because `logit_processor` is not accepted yet.

- [ ] **Step 3: Patch `forward_decoder()`**

Make this minimal change:

```python
def forward_decoder(
    self,
    prev_decoder_out,
    encoder_out,
    need_attn_weights=False,
    partial_masks=None,
    sampling_strategy="gumbel_argmax",
    logit_processor=None,
    batch=None,
):
    ...
    # After line 217 (eos_id = -inf), before line 219 (sampling branch):
    logits[..., self.eos_id] = -math.inf

    if logit_processor is not None:
        if batch is None:
            raise ValueError("batch is required when logit_processor is provided")
        logits = logit_processor(
            logits=logits,
            output_tokens=output_tokens,
            batch=batch,
            step=step + 1,
            max_step=max_step,
        )
```

Keep default behavior unchanged when `logit_processor is None`. The hook receives logits AFTER all five special-token bans (`mask_id`/`x_id`/`pad_id`/`bos_id`/`eos_id`), so the processor never reintroduces them; `DPLMTokenBridge.scatter_aa_logits()` re-bans them after fusion as a defense-in-depth measure.

- [ ] **Step 4: Patch `generate()`**

Add `logit_processor=None` to the signature and pass it through:

```python
decoder_out = self.forward_decoder(
    prev_decoder_out=prev_decoder_out,
    encoder_out=encoder_out,
    partial_masks=partial_masks,
    sampling_strategy=sampling_strategy,
    logit_processor=logit_processor,
    batch=batch,
)
```

- [ ] **Step 5: Run hook test and a no-refiner baseline smoke**

Run:

```bash
PYTHONPATH=inverse_folding/dplm/src pytest tests/inverse_folding/dplm_refiner/test_dplm_hook.py -v
```

Expected: hook test passes.

Run existing targeted tests that cover native generation helpers:

```bash
pytest tests -k "reference_flow or phase_c0 or dplm" -v
```

Expected: no regressions except pre-existing environment import skips/failures already documented in `LOG.md`.

## Task 8: Refiner Checkpoint and Training Helpers

**Files:**
- Create: `inverse_folding/dplm_refiner/checkpoint.py`
- Create: `inverse_folding/dplm_refiner/training.py`
- Test: `tests/inverse_folding/dplm_refiner/test_training.py`

- [ ] **Step 1: Write failing tests for checkpoint round-trip and masked CE**

```python
import torch

from inverse_folding.dplm_refiner.checkpoint import load_refiner_checkpoint, save_refiner_checkpoint
from inverse_folding.dplm_refiner.config import DPLMRefinerConfig
from inverse_folding.dplm_refiner.ipa.refiner import DPLMIPARefiner
from inverse_folding.dplm_refiner.training import masked_refiner_cross_entropy


def test_masked_refiner_cross_entropy_uses_only_selected_valid_positions():
    logits = torch.zeros((1, 4, 20))
    logits[0, 1, 3] = 10.0
    logits[0, 2, 4] = -10.0
    target = torch.tensor([[0, 3, 4, 5]])
    selected = torch.tensor([[False, True, True, False]])
    valid = torch.tensor([[True, True, True, False]])

    loss = masked_refiner_cross_entropy(logits, target, selected_mask=selected, valid_mask=valid)

    assert loss.item() > 0.0


def test_refiner_checkpoint_round_trip(tmp_path):
    model = DPLMIPARefiner(hidden_dim=32, ipa_pairwise_dim=32, ipa_heads=4, ipa_depth=2)
    config = DPLMRefinerConfig()
    path = tmp_path / "refiner.pt"

    save_refiner_checkpoint(path, model=model, config=config, extra={"epoch": 1})
    loaded_model, loaded_config, extra = load_refiner_checkpoint(path, map_location="cpu")

    assert isinstance(loaded_model, DPLMIPARefiner)
    assert loaded_config == config
    assert extra["epoch"] == 1
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_training.py -v
```

Expected: fail until checkpoint/training helpers exist.

- [ ] **Step 3: Implement helper APIs**

Use:

```python
def masked_refiner_cross_entropy(
    logits: torch.Tensor,
    target_aa_tokens: torch.Tensor,
    *,
    selected_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor: ...
```

Rules:

- Loss positions are `selected_mask & valid_mask`.
- If no positions are selected, return `logits.sum() * 0.0` to preserve graph connectivity.
- Target tokens are AA indices `0..19`, not DPLM vocab ids.

Checkpoint payload:

```python
payload = {
    "model_type": "DPLMIPARefiner",
    "model_config": {...},
    "refiner_config": asdict(config),
    "state_dict": model.state_dict(),
    "extra": extra,
}
```

- [ ] **Step 4: Run training helper tests**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_training.py -v
```

Expected: all tests pass.

## Task 9: Train and Generate Scripts

**Files:**
- Create or extend after reuse check: `scripts/train_if_imp_refiner.py`
- Create or extend after reuse check: `scripts/run_if_imp_refiner.py`
- Create: `scripts/submit_if_imp.slurm`
- Modify: `doc/SCRIPTS.md`
- Test: `tests/scripts/test_if_imp_scripts.py`

- [ ] **Step 1: Reuse-first gate**

Read:

```bash
sed -n '1,220p' doc/SCRIPTS.md
sed -n '1,260p' scripts/train_if_v1.py
sed -n '1,260p' scripts/run_if_phase_c0.py
```

Decision:

- Extend existing scripts only if the final diff would share more than 60% of lines and behavior.
- If creating the new scripts above, register them in `doc/SCRIPTS.md` before the task is complete.

- [ ] **Step 2: Write failing CLI smoke tests**

```python
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_help(script):
    return subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_train_if_imp_refiner_help():
    result = run_help("scripts/train_if_imp_refiner.py")
    assert result.returncode == 0
    assert "--cath-root" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--mask-ratio-center" in result.stdout


def test_run_if_imp_refiner_help():
    result = run_help("scripts/run_if_imp_refiner.py")
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout
    assert "--refiner-checkpoint" in result.stdout
    assert "--mc-dropout-passes" in result.stdout


def test_doc_scripts_registers_if_imp_scripts():
    text = (ROOT / "doc/SCRIPTS.md").read_text()
    assert "scripts/train_if_imp_refiner.py" in text
    assert "scripts/run_if_imp_refiner.py" in text
    assert "scripts/submit_if_imp.slurm" in text
```

- [ ] **Step 3: Run script tests and verify they fail**

Run:

```bash
pytest tests/scripts/test_if_imp_scripts.py -v
```

Expected: fail until scripts and doc registration exist.

- [ ] **Step 4: Implement `scripts/train_if_imp_refiner.py`**

Required CLI:

```text
--cath-root PATH
--checkpoint PATH
--output-dir PATH
--split train
--max-length 500
--batch-size 4
--epochs 10
--lr 1e-4
--mask-ratio-center 0.4
--mask-ratio-deviation 0.2
--device cuda
--seed 42
--num-workers 4
--limit-batches INT
```

Behavior:

- Load the same CATH JSONL/splits format used by DPLM.
- Use DPLM featurizer/batch contract; do not introduce MapDiff PyG graph data.
- Train only `DPLMIPARefiner`; DPLM checkpoint is used for alphabet/featurizer and optional base logits if the selected-mask curriculum uses DPLM entropy.
- Save `refiner_last.pt`, `metrics.jsonl`, and `run_config.yaml` under `--output-dir`.
- Fail fast if `--cath-root`, `--checkpoint`, or data files are missing.

- [ ] **Step 5: Implement `scripts/run_if_imp_refiner.py`**

Required CLI:

```text
--checkpoint PATH
--test-set-parquet PATH
--pdb-root PATH
--output-root PATH
--refiner-checkpoint PATH
--allele HLA-DRB1*07:01
--n-designs-per-protein 1
--max-iter 10
--temperature 1.0
--seed 42
--device cuda
--mc-dropout-passes 1
--mask-ratio-center 0.4
--mask-ratio-deviation 0.2
--fusion-temperature 1.0
--limit-proteins INT
```

Behavior:

- Reuse `inverse_folding/reference_flow/runtime.py::load_if_task`, `load_test_entries`, `prepare_backbone`, and output conventions from `scripts/run_if_phase_c0.py`.
- Instantiate `DPLMRefinerLogitProcessor` only when `--refiner-checkpoint` is provided.
- Write `generated.parquet`, `generated.fasta`, `run_config.yaml`, and `manifest.json`.
- Record refiner checkpoint digest and MapDiff reference commit in `manifest.json`.

- [ ] **Step 6: Implement `scripts/submit_if_imp.slurm`**

Required env vars:

```bash
MODE="${MODE:-generate_refiner}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/zc1519/src/Immune-Design}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
REFINER_CHECKPOINT="${REFINER_CHECKPOINT:-}"
CATH_ROOT="${CATH_ROOT:-}"
TEST_SET_PARQUET="${TEST_SET_PARQUET:-}"
PDB_ROOT="${PDB_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
ALLELE="${ALLELE:-HLA-DRB1*07:01}"
DEVICE="${DEVICE:-cuda}"
```

Rules:

- `MODE=train_refiner` calls `scripts/train_if_imp_refiner.py`.
- `MODE=generate_refiner` calls `scripts/run_if_imp_refiner.py`.
- `#SBATCH --output` and `#SBATCH --error` point under a `logs/` path, not `run/`.
- Cluster paths are shell env defaults only; Python modules still receive paths through CLI flags.

- [ ] **Step 7: Register scripts in `doc/SCRIPTS.md`**

Append a new H2 section in topical order (e.g., after the existing Inverse Folding sections); verify the existing section pattern by reading `doc/SCRIPTS.md` once before editing. Section body:

```markdown
## Inverse Folding -- IF Improvement (PLAN_IF_IMP.md)

MapDiff-grounded DPLM inverse-folding improvement path: IPA refiner training and refiner-enabled DPLM generation.

1. `scripts/train_if_imp_refiner.py` -- Train the DPLM-compatible IPA refiner on CATH batches. Uses DPLM/ESM alphabet and `[N, CA, C, O]` coordinates, converts to IPA `[N, CA, C, CB, O]`, and saves a refiner checkpoint with config.
2. `scripts/run_if_imp_refiner.py` -- Generate inverse-folding designs with optional refiner checkpoint, entropy mask selection, entropy-weighted fusion, and MC-dropout passes.

### SLURM

1. `scripts/submit_if_imp.slurm` -- Parameterized IF improvement launcher. `MODE=train_refiner` trains the IPA refiner; `MODE=generate_refiner` runs DPLM generation with the refiner. Paths are supplied via env vars and forwarded as CLI arguments.
```

- [ ] **Step 8: Run script tests**

Run:

```bash
pytest tests/scripts/test_if_imp_scripts.py -v
```

Expected: all tests pass.

## Task 10: End-to-End Local Smoke

**Files:**
- Modify only if smoke exposes bugs in files from Tasks 1-9.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
PYTHONPATH=inverse_folding/dplm/src pytest \
  tests/inverse_folding/dplm_refiner \
  tests/scripts/test_if_imp_scripts.py \
  -v
```

Expected: all newly added tests pass.

- [ ] **Step 2: Run CLI help checks manually**

Run:

```bash
python scripts/train_if_imp_refiner.py --help
python scripts/run_if_imp_refiner.py --help
```

Expected: both return code `0`; required flags from Task 9 appear.

- [ ] **Step 3: Run one CPU shape smoke for the refiner**

Run:

```bash
python - <<'PY'
import torch
from inverse_folding.dplm_refiner.ipa.refiner import DPLMIPARefiner

model = DPLMIPARefiner(hidden_dim=32, ipa_pairwise_dim=32, ipa_heads=4, ipa_depth=2)
x_aa = torch.zeros((1, 8, 20))
x_aa[..., 0] = 1.0
x_pos = torch.randn((1, 8, 5, 3))
x_aa_mask = torch.zeros((1, 8), dtype=torch.long)
x_aa_mask[:, 2:4] = 1
seq_mask = torch.ones((1, 8), dtype=torch.bool)
with torch.no_grad():
    logits = model(x_aa=x_aa, x_pos=x_pos, x_aa_mask=x_aa_mask, seq_mask=seq_mask)
print(tuple(logits.shape), torch.isfinite(logits).all().item())
PY
```

Expected output:

```text
(1, 8, 20) True
```

## Task 11: IPA Geometry Sidecar

**Files:**
- Create: `inverse_folding/dplm_refiner/sidecar.py`
- Modify: `inverse_folding/dplm/src/byprot/models/dplm/modules/gvp_transformer_encoder.py`
- Test: `tests/inverse_folding/dplm_refiner/test_sidecar.py`

- [ ] **Step 1: Write failing sidecar tests**

```python
import torch

from inverse_folding.dplm_refiner.sidecar import DPLMGeometrySidecar


def test_geometry_sidecar_returns_adapter_dim_and_respects_mask():
    sidecar = DPLMGeometrySidecar(hidden_dim=32, ipa_pairwise_dim=32, ipa_heads=4, ipa_depth=2, output_dim=512)
    coords = torch.randn((2, 6, 4, 3))
    coord_mask = torch.tensor(
        [[True, True, True, False, False, False], [True, True, True, True, True, False]]
    )
    tokens = torch.ones((2, 6), dtype=torch.long)
    special_sym_mask = ~coord_mask

    feats = sidecar(coords=coords, coord_mask=coord_mask, special_sym_mask=special_sym_mask, tokens=tokens)

    assert feats.shape == (2, 6, 512)
    assert torch.isfinite(feats).all()
    assert torch.allclose(feats[~coord_mask], torch.zeros_like(feats[~coord_mask]), atol=1e-6)
```

- [ ] **Step 2: Run sidecar test and verify it fails**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_sidecar.py -v
```

Expected: fail because sidecar does not exist.

- [ ] **Step 3: Implement `DPLMGeometrySidecar`**

Public constructor:

```python
class DPLMGeometrySidecar(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        ipa_pairwise_dim: int = 128,
        ipa_heads: int = 4,
        ipa_depth: int = 6,
        output_dim: int = 512,
        dropout: float = 0.2,
    ) -> None: ...

    def forward(
        self,
        *,
        coords: torch.Tensor,
        coord_mask: torch.Tensor,
        special_sym_mask: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor: ...
```

Implementation rules:

- Reuse `dplm_coords_to_ipa_positions()`.
- Build neutral all-zero AA one-hot features; do not infer amino-acid identity from DPLM token ids in this first sidecar.
- Use the same IPA trunk as the refiner but return projected hidden states instead of logits.
- Output shape is `[B, L, 512]`.
- Zero invalid positions.

- [ ] **Step 4: Integrate sidecar as opt-in only**

Integration rule:

- Add config fields to the GVP encoder wrapper or an adjacent wrapper only after checking the existing config pattern.
- Default `sidecar_enabled=False`.
- When enabled, combine features as `feats = feats + sidecar_projection(...)` after matching `[B, L, 512]`.
- Do not change encoder output keys: `feats`, optional `logits`, `coord_mask`, `encoder_attention_mask`.

- [ ] **Step 5: Run sidecar tests and no-sidecar regression tests**

Run:

```bash
pytest tests/inverse_folding/dplm_refiner/test_sidecar.py -v
PYTHONPATH=inverse_folding/dplm/src pytest tests/inverse_folding/dplm_refiner/test_dplm_hook.py -v
```

Expected: sidecar tests pass; no-refiner/no-sidecar DPLM hook still passes.

## Task 12: Final Verification and Log

**Files:**
- Modify: `LOG.md`

- [ ] **Step 1: Run final focused verification**

Run:

```bash
PYTHONPATH=inverse_folding/dplm/src pytest \
  tests/inverse_folding/dplm_refiner \
  tests/scripts/test_if_imp_scripts.py \
  -v
```

Expected: all new tests pass.

- [ ] **Step 2: Run static sanity checks**

Run:

```bash
python -m compileall inverse_folding/dplm_refiner scripts/train_if_imp_refiner.py scripts/run_if_imp_refiner.py
git diff --check
python - <<'PY'
from pathlib import Path

paths = [
    Path("inverse_folding/dplm_refiner"),
    Path("scripts/train_if_imp_refiner.py"),
    Path("scripts/run_if_imp_refiner.py"),
    Path("PLAN_IF_IMP.md"),
]
needles = [
    "TO" + "DO",
    "T" + "BD",
    "implement " + "later",
    "fill in " + "details",
    "/" + "scratch" + "/gpfs",
]
hits = []
for root in paths:
    candidates = root.rglob("*.py") if root.is_dir() else [root]
    for path in candidates:
        text = path.read_text()
        for needle in needles:
            if needle in text:
                hits.append(f"{path}: {needle}")
if hits:
    raise SystemExit("\n".join(hits))
PY
```

Expected:

- `compileall` succeeds.
- `git diff --check` prints no whitespace errors.
- The Python self-scan finds no placeholder text and no hardcoded cluster path in Python modules. The plan text may mention the cluster path rule only in prose.

- [ ] **Step 3: Append `LOG.md` entry**

Use the next `L####` id and this schema:

```markdown
### L####
- timestamp: YYYY-MM-DDTHH:MM:SS+08:00
- type: IMPLEMENTATION
- module: IF_IMP
- trigger: User requested implementation of the single PLAN_IF_IMP.md coding plan for MapDiff-grounded DPLM inverse-folding improvement.
- change_summary: Implemented a DPLM-compatible IPA refiner package, optional decoder-time refiner hook, refiner training/generation scripts, script registration, and an opt-in IPA geometry sidecar.
- rationale: The implementation ports MapDiff's directly compatible ideas while preserving DPLM's existing sequence prior and default generation behavior.
- artifacts:
  - `/Users/jerry/Project/MHC-IF/inverse_folding/dplm_refiner/`
  - `/Users/jerry/Project/MHC-IF/scripts/train_if_imp_refiner.py`
  - `/Users/jerry/Project/MHC-IF/scripts/run_if_imp_refiner.py`
  - `/Users/jerry/Project/MHC-IF/scripts/submit_if_imp.slurm`
  - `/Users/jerry/Project/MHC-IF/doc/SCRIPTS.md`
  - `/Users/jerry/Project/MHC-IF/tests/inverse_folding/dplm_refiner/`
- evidence: Focused pytest suite and compile/static checks from PLAN_IF_IMP.md Task 12 passed locally.
- impact:
  - scope: DPLM inverse-folding improvement path only; default DPLM generation remains unchanged when refiner/sidecar configs are disabled.
  - risk: medium
  - confidence: 0.82
- status: done
- next_action: Train the refiner on CATH with `scripts/train_if_imp_refiner.py`, then run `scripts/run_if_imp_refiner.py` on the fast IF-ready subset before full CATH/test-set evaluation.
- refs:
  - `PLAN_IF_IMP.md`
  - `doc/IF_IMP.md`
```

## Execution Notes

- Implement Tasks 1-10 first. That gives a complete, testable DPLM + MapDiff-style IPA refiner path.
- Implement Task 11 only after the refiner path passes local smoke tests. The sidecar is opt-in and must not affect default DPLM outputs.
- Full MapDiff EGNN replacement is deliberately not in this first coding plan. Its `EGNN_NET.forward(data, time)` depends on PyG `Data` fields (`x`, `extra_x`, `pos`, `edge_index`, `edge_attr`, `ss`, `batch`, `time`) that DPLM does not currently produce. The IPA path already imports MapDiff's geometry-conditioned refinement idea with a much smaller and safer DPLM interface.
- Pre-existing DPLM bug, deliberately NOT fixed in this plan to keep the diff scoped: `forward_decoder` line 223 hardcodes `temperature=0.0` in `stochastic_sample_from_categorical`, ignoring the value plumbed through `prev_decoder_out["temperature"]`. `--temperature` flags on `run_if_imp_refiner.py` / `run_if_phase_c0.py` are no-ops at the decoder. Document in `LOG.md` tech-debt section after Task 12.
- Note for porters: MapDiff's IPA architecture defaults (`ipa_depth=6`, `hidden_dim=128`, `ipa_heads=4`, `dropout=0.2`) live in the `IPANetPredictor` constructor signature, not in a separate YAML. EGCL dropout is `0.1` per `MapDiff/conf/model/egnn.yaml`; we are porting the IPA path, so `dropout=0.2` is the correct refiner default.
