# PDLM

Source code: [GitHub](https://github.com/xiongyf520/PDLM)

Model repository: [Hugging Face — xiongyf/PDLM](https://huggingface.co/xiongyf/PDLM)
(currently private; final weight upload is being prepared).
See [weight packaging and Hugging Face setup](weights/README.md).

PDLM jointly models protein and DNA sequences with a shared ESM2-initialized
Transformer, separate token embeddings, and separate masked language modeling
heads. This package provides continued pretraining, zero-shot sequence scoring,
and representation extraction.

## Scope and status

Version `0.3.0` contains the core model, training demos, zero-shot scoring,
and representation extraction. The full research training dataset, trained PDLM
weights, and manuscript-specific downstream predictors are not included.
The 100 synthetic training/validation examples and two demo configurations
demonstrate executable workflows; they do not reproduce manuscript performance.
See [release status](docs/RELEASE_STATUS.md).

| Component | Location | Purpose |
| --- | --- | --- |
| Model | `src/pdlm/modeling.py` | Shared encoder, separate embeddings, and two MLM heads |
| Tokenization | `src/pdlm/tokenization.py` | Protein-only and paired input construction |
| Data and masking | `src/pdlm/data.py`, `src/pdlm/masking.py` | JSONL loading, augmentation, uniform and domain-aware masking |
| Training | `src/pdlm/train.py` | Training, validation, best-checkpoint saving, and resume |
| Scoring | `src/pdlm/scoring.py` | Protein mutation scores and DNA PLL / delta PLL |
| Representations | `src/pdlm/embeddings.py` | Pooled and per-token embeddings |
| Training demos | `configs/demo/`, `examples/` | Base / domain strategies on 100 synthetic pairs |
| Verification | `tests/` | Offline checks using a tiny synthetic model |

## Installation

Python 3.9 or later is required; Python 3.11 is recommended. Install the PyTorch
build appropriate for your hardware before installing PDLM. Training the 35M or
150M models normally requires a GPU. The offline tests run on a CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Dependencies are declared in `pyproject.toml`.
Transformers is pinned because PDLM calls the ESM encoder directly.

## ESM2 initialization

Training loads pretrained ESM2 weights through `transformers.EsmModel`.
The demos use `facebook/esm2_t12_35M_UR50D`. To initialize a 150M model, set
`model.esm2_model` to `facebook/esm2_t30_150M_UR50D` in a copy of the YAML and
choose a new output directory. A local Hugging Face-format ESM2 directory can
also be supplied. Training needs its configuration, tokenizer/vocabulary files,
and pretrained weights.

The `fair-esm` package is not required. This implementation uses Hugging Face
ESM2 checkpoints rather than `esm.pretrained`. A complete PDLM checkpoint
already contains the trained encoder. Loading it requires only the matching
ESM2 configuration and tokenizer, not the original ESM2 pretrained weight file.

PDLM uses a custom wrapper. Its weights cannot be loaded directly with
`AutoModel.from_pretrained()`; use this package's loaders and commands.

## Input data

Inputs use JSON Lines with one record per line. DNA can be plain sequence text
(`ATCG`) or space-separated tokens (`dA dT dC dG`).

```json
{"id":"pair_1","protein_seq":"MKTAYIAK","dna_seq":"ATCGATCG"}
```

Domain-aware training accepts a binary mask aligned to the full protein.
A value of 1 denotes an annotated DNA-binding region.

```json
{"id":"pair_2","protein_seq":"MKTAYIAK","dna_seq":"ATCG","binding_mask":[0,0,1,1,1,0,0,0],"has_annotation":true}
```

For your own data, supply your existing train/validation/test partitions;
the training runner does not generate new splits. The demo provides 80 synthetic
training pairs and 20 synthetic validation pairs, generated with seed 42 and
without shared protein or DNA sequences. Pairings and domain annotations are
also artificial. See [example formats and generation](examples/README.md).
Database download and cleaning pipelines are outside this core package.

## Training demo

The two demo YAML files run one epoch with batch size 2, maximum input length 64,
learning rate 0.0002, no warmup, and seed 42. These are demo settings, not the
verified settings of the final research checkpoints. First use may download
pretrained ESM2 weights; runtime depends on hardware.

Run the base or domain strategy from the repository root:

```bash
pdlm-train --config configs/demo/base.yaml
pdlm-train --config configs/demo/domain.yaml
```

To train on your own data, edit a copy of a demo YAML and select your data files,
hyperparameters, and a new output directory.
YAML paths are relative to `paths.root`, itself relative to the configuration
file. The supplied `paths.root: ../..` selects the repository root. A local ESM2
directory follows the same rule; Hugging Face model identifiers are also accepted.

Training writes `config.yaml`, `metrics.json`, and `checkpoint_best.pt`.
The best checkpoint is selected by the lowest validation loss.

To continue a run, increase `training.epochs` in its configuration and resume:

```bash
pdlm-train \
  --config configs/demo/base.yaml \
  --resume outputs/demo_base/checkpoint_best.pt
```

Resume restores the model, optimizer, scheduler, and epoch. It does not restore
all random-generator states or promise bitwise-identical continuation.
This is a single-process runner; distributed training, mixed precision, and
automatic early stopping are not implemented. For a small offline CPU
training/inference check without an ESM2 download, run
`python tests/integration_test.py`.

## Zero-shot scoring

All commands below require trained PDLM weights, which are not bundled.
CLI paths are relative to the current working directory. Single substitutions
use one-based numbering, for example `K2A`.

### Protein mutation scores

Omit `dna_seq` for protein-only scoring. The score is the mutant-minus-reference
log-probability at a single masked amino-acid position.

```bash
pdlm-score \
  --checkpoint weights/PDLM-35M-base.safetensors \
  --esm2-model facebook/esm2_t12_35M_UR50D \
  --input examples/protein_only_variants.jsonl \
  --output outputs/protein_only_scores.jsonl \
  --mode protein-dll
```

Providing `dna_seq` produces a DNA-conditioned protein mutation score instead.
See `examples/protein_variants.jsonl`. Report the input setting explicitly.

### Protein-conditioned DNA scores

Each nucleotide is masked separately while the protein and other DNA positions
remain visible. Output includes the sum and mean DNA pseudo-log-likelihood,
negative mean PLL (`dna_mean_loss`), and the number of scored bases.
Higher PLL and lower loss indicate greater model compatibility, not a calibrated
binding probability or physical affinity.

```bash
pdlm-score \
  --checkpoint weights/PDLM-35M-domain.safetensors \
  --esm2-model facebook/esm2_t12_35M_UR50D \
  --input examples/toy_pairs.jsonl \
  --output outputs/dna_pll.jsonl \
  --mode dna-pll
```

Delta PLL is `variant mean PLL - reference mean PLL`.
The example covers a changed protein acting on the same DNA and a changed DNA
paired with the same protein.

```bash
pdlm-score \
  --checkpoint weights/PDLM-35M-domain.safetensors \
  --esm2-model facebook/esm2_t12_35M_UR50D \
  --input examples/dna_delta_pll.jsonl \
  --output outputs/dna_delta_pll.jsonl \
  --mode dna-dpll
```

The default limit is 800 total tokens (`--max-length`). Long paired-input proteins
are trimmed from both termini; long protein-only inputs are trimmed from the
right. A mutation removed by truncation raises an error. DNA PLL raises an error
if the full DNA cannot be scored; it does not silently score a shortened DNA.

## Representation extraction

```bash
pdlm-embed \
  --checkpoint weights/PDLM-35M-base.safetensors \
  --esm2-model facebook/esm2_t12_35M_UR50D \
  --input examples/toy_pairs.jsonl \
  --output outputs/embeddings.npz \
  --mode concat
```

Let `H` be the hidden size and `Lp` / `Ld` the retained sequence lengths.
Mean pooling excludes special and padding tokens.

| Mode | Output per record | Definition |
| --- | --- | --- |
| `cls` | `H` | First special-token representation |
| `joint_mean` | `H` | Mean over protein and DNA positions |
| `protein_mean` | `H` | Mean over protein positions |
| `dna_mean` | `H` | Mean over DNA positions |
| `concat` | `2H` | Protein mean concatenated with DNA mean |
| `per_token` | `(Lp + Ld, H)` | Protein positions followed by DNA positions |

NPZ outputs contain `ids` and `embeddings`. Variable-length `per_token` output
is an object array and requires `np.load(path, allow_pickle=True)`; load only
trusted files. Protein-only input supports `cls`, `joint_mean`,
`protein_mean`, and `per_token`. `dna_mean` and `concat` require DNA.

## Weights and verification

The loader accepts a `.pt` checkpoint containing `model_state_dict`, a plain
PyTorch state dictionary, or `.safetensors` tensors. Loading is strict by default.
See [weight packaging](weights/README.md) for export and manifest details.

```bash
python tests/smoke_test.py
python tests/integration_test.py
```

These offline checks use a tiny synthetic model. They do not validate the four
final research checkpoints or reproduce scientific performance.

## Release and citation

The code repository contains source code, demo configurations, synthetic examples,
tests, documentation, and weight-manifest templates. Full research datasets,
model files, generated outputs, environments, and credentials are excluded.
See [model hosting](weights/README.md) for weight distribution.
Software and model licenses are not yet assigned; see
[license status](LICENSE_PENDING.md). Manuscript citation details and permanent
release links will be added when available.

PDLM uses ESM2 from *Evolutionary-scale prediction of atomic-level protein
structure with a language model*. See the
[ESM source repository](https://github.com/facebookresearch/esm) and
[official checkpoint](https://huggingface.co/facebook/esm2_t12_35M_UR50D).
