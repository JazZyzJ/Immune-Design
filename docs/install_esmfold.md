# ESMFold Installation Guide (Princeton HPC Cluster)

Quick reference for installing ESMFold dependencies in the `immune-design`
conda environment (Python 3.12 + PyTorch 2.5.1 + CUDA 12.1).

---

## Prerequisites

| Package | Version | Notes |
|---------|---------|-------|
| Python  | 3.12    | conda env `immune-design` |
| PyTorch | 2.5.1   | compiled against CUDA 12.1 |
| fair-esm | 2.0.0  | `pip install fair-esm[esmfold]` |

> **`LD_PRELOAD` requirement**: torch in this environment needs
> `LD_PRELOAD="$PROJECT_ROOT/lib/ijit_stub.so"` to avoid
> `undefined symbol: iJIT_NotifyEvent`. All commands below assume this is set.

---

## Step 1 — Install OpenFold (pure Python, no CUDA extension)

The standard `pip install git+...openfold` fails for two reasons:

1. **Build isolation**: pip creates a temp venv without torch; OpenFold's
   `setup.py` does `import torch` at the top level.
2. **CUDA version mismatch**: the cluster's CUDA toolkit (13.0 default) does
   not match PyTorch's compile-time CUDA (12.1). PyTorch's `_check_cuda_version`
   enforces strict equality.

**Solution**: clone, patch out the CUDA extension, install as pure Python.

```bash
# Clone
git clone --filter=blob:none \
  https://github.com/aqlaboratory/openfold.git /tmp/openfold_install
cd /tmp/openfold_install
git checkout pl_upgrades

# Patch setup.py — replace the entire file with a minimal version
cat > setup.py << 'SETUP_EOF'
from setuptools import setup, find_packages

setup(
    name='openfold',
    version='2.2.0',
    description="A PyTorch reimplementation of DeepMind's AlphaFold 2",
    author='OpenFold Team',
    license='Apache License, Version 2.0',
    url='https://github.com/aqlaboratory/openfold',
    packages=find_packages(exclude=["tests", "scripts"]),
    include_package_data=True,
    package_data={
        "openfold": ['utils/kernel/csrc/*'],
        "": ["resources/stereo_chemical_props.txt"]
    },
)
SETUP_EOF

# Patch attention_core.py — make CUDA import optional
sed -i 's/^attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")/try:\n    attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")\nexcept ImportError:\n    attn_core_inplace_cuda = None/' \
  openfold/utils/kernel/attention_core.py

# Patch structure_module.py — same fix
sed -i 's/^attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")/try:\n    attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")\nexcept ImportError:\n    attn_core_inplace_cuda = None/' \
  openfold/model/structure_module.py

# Install (no build isolation, no deps to avoid pulling unwanted packages)
LD_PRELOAD="$PROJECT_ROOT/lib/ijit_stub.so" \
pip install --no-build-isolation --no-deps .

# Cleanup
rm -rf /tmp/openfold_install
```

> **Performance note**: the skipped CUDA kernel (`attn_core_inplace_cuda`) is a
> custom fused softmax used only when `use_memory_efficient_kernel=True`.
> ESMFold's code path **never** sets this flag — it always uses standard PyTorch
> attention. **Zero performance impact** for ESMFold inference.

---

## Step 2 — Remove incompatible deepspeed

OpenFold's `primitives.py` probes for deepspeed at import time. If an old
version (e.g. 0.5.9) is installed, it will crash with
`ModuleNotFoundError: No module named 'torch._six'` (removed in PyTorch 2.0+).

```bash
pip uninstall deepspeed -y
```

If you later need deepspeed, install `>=0.14.5` which supports PyTorch 2.x.

---

## Step 3 — Install missing Python dependencies

```bash
pip install modelcif ml-collections dm-tree biopython
```

---

## Step 4 — Patch fair-esm for Python 3.12 dataclass compatibility

Python 3.12 enforces that mutable dataclass defaults must use
`default_factory`. Two files in `fair-esm` need patching:

### 4a. `esm/esmfold/v1/trunk.py`

```python
# Before:
from dataclasses import dataclass
# ...
    structure_module: StructureModuleConfig = StructureModuleConfig()

# After:
from dataclasses import dataclass, field
# ...
    structure_module: StructureModuleConfig = field(default_factory=StructureModuleConfig)
```

### 4b. `esm/esmfold/v1/esmfold.py`

```python
# Before:
from dataclasses import dataclass
# ...
    trunk: T.Any = FoldingTrunkConfig()

# After:
from dataclasses import dataclass, field
# ...
    trunk: T.Any = field(default_factory=FoldingTrunkConfig)
```

**One-liner patch** (run from the site-packages root):

```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")

# trunk.py
sed -i 's/from dataclasses import dataclass$/from dataclasses import dataclass, field/' \
  "$SITE/esm/esmfold/v1/trunk.py"
sed -i 's/structure_module: StructureModuleConfig = StructureModuleConfig()/structure_module: StructureModuleConfig = field(default_factory=StructureModuleConfig)/' \
  "$SITE/esm/esmfold/v1/trunk.py"

# esmfold.py
sed -i 's/from dataclasses import dataclass$/from dataclasses import dataclass, field/' \
  "$SITE/esm/esmfold/v1/esmfold.py"
sed -i 's/trunk: T.Any = FoldingTrunkConfig()/trunk: T.Any = field(default_factory=FoldingTrunkConfig)/' \
  "$SITE/esm/esmfold/v1/esmfold.py"
```

> These patches are applied directly to site-packages and will be **overwritten**
> by `pip install --upgrade fair-esm`. Re-apply after any ESM upgrade.

---

## Verification

```bash
LD_PRELOAD="$PROJECT_ROOT/lib/ijit_stub.so" python -c "
import torch
from esm.esmfold.v1.pretrained import esmfold_v1
from esm.esmfold.v1.esmfold import ESMFold, ESMFoldConfig
from openfold.np import residue_constants
from openfold.data.data_transforms import make_atom14_masks
print('torch:', torch.__version__)
print('residue_constants:', len(residue_constants.atom_types), 'atom types')
print('All ESMFold imports OK')
"
```

Expected output:

```
torch: 2.5.1
residue_constants: 37 atom types
All ESMFold imports OK
```

---

---

## DPLM (ByProt) Installation

DPLM was written for Python 3.10 + PyTorch 2.2.0. Running it in Python 3.12 +
PyTorch 2.5.1 requires several patches. **Do NOT run `install.sh` directly** —
it will downgrade torch to 2.2.0.

### Step A — Install ByProt (editable, no deps)

```bash
cd inverse_folding/dplm

# Comment out torchtext in requirements.txt (deprecated since PyTorch 2.5, unused)
sed -i 's/^torchtext/# torchtext/' requirements.txt

pip install --no-deps -e .
```

### Step B — Install runtime dependencies

```bash
# Core deps
pip install lightning hydra-core hydra-colorlog hydra-optuna-sweeper \
  tensorboard pyrootutils python-dotenv einops opt_einsum \
  e3nn lmdb tmtools biotite mdtraj MDAnalysis

# Pin transformers to 4.44.x — 5.x is incompatible with torch 2.5.1
#   - 5.x enforces torch >= 2.6 for torch.load (CVE-2025-32434)
#   - 5.x changed PreTrainedModel internals (all_tied_weights_keys)
#   - 5.x changed __all__ exports breaking "import *"
pip install "transformers>=4.39,<4.45"

# PyG (use matching torch+CUDA wheel)
pip install torch_scatter torch_geometric \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

> Skip `deepspeed`, `peft`, `torchtext`, and dev tools (`black`, `flake8`,
> `pre-commit`, etc.) — they are not needed for inference/training with DPLM1.

### Step C — Patch DPLM source for Python 3.12

All patches are in `inverse_folding/dplm/src/byprot/`:

**1. `import imp` removed in 3.12** (`datamodules/dataset/tokenized_protein.py`):

```python
# Before:
import imp
# After:
import importlib
```

**2. `transformers` star-import may not export internal classes**
(`models/dplm/modules/dplm_modeling_esm.py`):

```python
# Add after "from transformers.models.esm.modeling_esm import *":
from transformers.models.esm.modeling_esm import (
    EsmSelfAttention, EsmSelfOutput, EsmAttention,
    EsmLayer, EsmEncoder, EsmIntermediate, EsmOutput,
    EsmEmbeddings, EsmPooler, EsmContactPredictionHead,
    EsmLMHead, EsmModel, EsmPreTrainedModel, EsmForMaskedLM,
)
```

> Even with transformers 4.44.x the explicit imports are harmless and provide
> future-proofing.

**3. `lightning_fabric` renamed API** (`utils/strategies.py`):

```python
# Before:
from lightning_fabric.strategies.fsdp import (
    _has_meta_device_parameters, ...
)
# After:
try:
    from lightning_fabric.strategies.fsdp import (
        _has_meta_device_parameters, ...
    )
except ImportError:
    from lightning_fabric.strategies.fsdp import (
        _has_meta_device_parameters_or_buffers as _has_meta_device_parameters, ...
    )
```

**4. `attn_core_inplace_cuda` not available**
(`models/structok/modules/folding_utils/structure_module.py`):

```python
# Before:
attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")
# After:
try:
    attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")
except ImportError:
    attn_core_inplace_cuda = None
```

**5. Dataclass mutable defaults** (multiple files — run bulk fixer):

Any `field(default=SomeConfig())` or bare `var: Type = SomeConfig()` inside
`@dataclass` must become `field(default_factory=SomeConfig)`.

**6. biotite 1.x removed `filter_backbone`** (site-packages `esm/inverse_folding/util.py`):

```python
# Before:
from biotite.structure import filter_backbone
# After:
try:
    from biotite.structure import filter_backbone
except ImportError:
    from biotite.structure import filter_peptide_backbone as filter_backbone
```

**7. Exclude dplm2 from auto-import** (`models/__init__.py`):

```python
# Add "dplm2" to excludes:
import_modules(..., excludes=["protein_structure_prediction", "dplm2"])
```

### Step C+ — Create `pyproject.toml` for pyrootutils

DPLM uses `pyrootutils` to locate the project root. Without a `.git` directory
or `pyproject.toml` in `inverse_folding/dplm/`, Hydra will fail with
"Primary config directory not found".

```bash
cat > inverse_folding/dplm/pyproject.toml << 'EOF'
[project]
name = "byprot"
version = "1.0.0"
EOF
```

### Step C++ — Pre-download models for offline compute nodes

Princeton HPC compute nodes have **no internet access**. All models must be
cached on the login node before submitting jobs.

```bash
# 1. ESM-IF1 weights (torch.hub)
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
python -c "
import esm
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
print('ESM-IF1 cached at:', '$TORCH_HOME/hub/checkpoints/')
"

# 2. DPLM-650m weights (HuggingFace)
export HF_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/huggingface"
python -c "
from transformers import AutoConfig, AutoModel
from huggingface_hub import hf_hub_download

# Download config + tokenizer + weights
hf_hub_download('airkingbd/dplm_650m', 'config.json')
hf_hub_download('airkingbd/dplm_650m', 'pytorch_model.bin')
hf_hub_download('airkingbd/dplm_650m', 'tokenizer_config.json')
hf_hub_download('airkingbd/dplm_650m', 'vocab.txt')
hf_hub_download('airkingbd/dplm_650m', 'special_tokens_map.json')
print('DPLM-650m cached at:', '$HF_HOME/hub/models--airkingbd--dplm_650m/')
"
```

> Run the above on the **login node** (which has internet). The SLURM script
> sets `TORCH_HOME`, `HF_HOME`, and `TRANSFORMERS_OFFLINE=1` so compute nodes
> read from these caches.

### Step D — SLURM environment variables

The training script (`scripts/submit_if_train.slurm`) must export:

```bash
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
export HF_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
```

| Variable | Purpose |
|----------|---------|
| `LD_PRELOAD` | Stub for Intel JIT `iJIT_NotifyEvent` symbol |
| `TORCH_HOME` | Offline cache for `torch.hub` (ESM-IF1 weights) |
| `HF_HOME` | Offline cache for HuggingFace (DPLM-650m weights) |
| `TRANSFORMERS_OFFLINE` | Prevents `transformers` from hitting the network |
| `HF_DATASETS_OFFLINE` | Prevents `datasets` from hitting the network |
| `WANDB_MODE` | Offline W&B logging (no network required) |

### Step E — Verification

```bash
LD_PRELOAD="$PROJECT_ROOT/lib/ijit_stub.so" python -c "
import torch; print('torch:', torch.__version__)
import byprot; print('ByProt OK')
from esm.esmfold.v1.pretrained import esmfold_v1; print('ESMFold OK')
from byprot.models.dplm.dplm import DiffusionProteinLanguageModel; print('DPLM1 OK')
"
```

Expected:

```
torch: 2.5.1+cu121
ByProt OK
ESMFold OK
DPLM1 OK
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `undefined symbol: iJIT_NotifyEvent` | torch needs Intel JIT stub | Set `LD_PRELOAD=...lib/ijit_stub.so` |
| `No module named 'torch'` during pip install | pip build isolation | Use `--no-build-isolation` |
| `CUDA version (13.0) mismatches (12.1)` | No CUDA 12.1 toolkit on cluster | Skip CUDA ext (pure Python install) |
| `No module named 'torch._six'` | deepspeed too old for PyTorch 2.x | `pip uninstall deepspeed` |
| `No module named 'modelcif'` | Missing openfold dependency | `pip install modelcif` |
| `mutable default ... not allowed` | Python 3.12 dataclass strictness | Patch with `field(default_factory=...)` |
| `No module named 'attn_core_inplace_cuda'` | CUDA ext not compiled | Patch imports with try/except |
| `No module named 'imp'` | Removed in Python 3.12 | Replace with `import importlib` |
| `NameError: 'EsmSelfAttention'` | transformers `__all__` changed | Add explicit imports (see patch #2) |
| `NameError: 'EsmLMHead'` | Same as above | Add `EsmLMHead, EsmModel, EsmPreTrainedModel, EsmForMaskedLM` to explicit imports |
| `No module named 'lightning'` | Missing lightning package | `pip install lightning` |
| torch downgraded to 2.2.0 | Ran DPLM `install.sh` line 1 | Reinstall: `pip install torch==2.5.1 --index-url .../cu121` |
| `cannot import name 'filter_backbone' from 'biotite.structure'` | biotite 1.x removed `filter_backbone` | Patch `esm/inverse_folding/util.py` (see patch #6) |
| `URLError: Name or service not known` (compute node) | No internet on compute nodes | Pre-download models (see Step C++); set `TRANSFORMERS_OFFLINE=1` |
| `ValueError: ...require users to upgrade torch to at least v2.6` | transformers 5.x CVE-2025-32434 check | Downgrade: `pip install "transformers>=4.39,<4.45"` |
| `'EsmForDPLM' has no attribute 'all_tied_weights_keys'` | transformers 5.x internal API change | Downgrade: `pip install "transformers>=4.39,<4.45"` |
| `airkingbd/dplm_650m does not appear to have pytorch_model.bin` | Model weights not in HF cache | Download on login node (see Step C++) |
| `Primary config directory not found` (Hydra) | `pyrootutils` can't find project root | Create `pyproject.toml` in `inverse_folding/dplm/` (see Step C+) |
| HF cache format error after transformers downgrade | Cache created by 5.x, read by 4.x | Delete `$HF_HOME/hub/models--airkingbd--dplm_650m/` and re-download |
