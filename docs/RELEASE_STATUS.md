# Release status

## Version 0.2.0

This core package is prepared for a private code repository. Its documentation,
source code, configuration files, and examples are in English. No Chinese
comments or hard-coded workstation/server paths were found in the upload files.

Included capabilities:

- Joint ESM2-based protein-DNA MLM model and separate vocabularies.
- JSONL input, uniform masking, and domain-aware masking.
- Single-process continued pretraining, validation, best checkpoints, and resume.
- Protein-only and DNA-conditioned single-substitution scoring.
- Protein-conditioned DNA PLL and reference-relative delta PLL.
- Pooled and per-token representation extraction.
- Safetensors export without optimizer state or training configuration.
- Four pretraining templates and small synthetic examples.

## Verification completed on 2026-10-08

The offline smoke and integration tests passed in this CPU environment:

| Dependency | Version |
| --- | --- |
| Python | 3.11.5 |
| PyTorch | 2.0.1 |
| Transformers | 4.44.2 |
| NumPy | 1.25.2 |
| PyYAML | 6.0.1 |
| Safetensors | 0.8.0 |

Editable installation was tested. Checks covered tiny-model forward passes,
uniform/domain masking, a short training run, checkpoint resume, strict loading,
Safetensors export with identical model tensors, both protein-scoring input
settings, DNA PLL / delta PLL, and pooled embeddings. The checkpoint loader was
tested with a local ESM2 configuration/tokenizer folder containing no base
pretrained weights. Test data and temporary weights were synthetic.

This is a tested local CPU environment, not a prescribed CUDA environment.
Large-model training, final-checkpoint loading, and scientific benchmark
reproduction have not been validated by these tests.

## Changes from 0.1.0

- Added protein-only input and examples without changing paired-input behavior.
- Added a complete DNA delta PLL example.
- Checkpoint loading now builds the backbone from its configuration and loads the
  complete PDLM state, avoiding a redundant ESM2 weight download.
- Added restricted PyTorch loading and Safetensors support/export.
- DNA scoring rejects silent DNA truncation.
- New training runs cannot overwrite existing results accidentally.
- Restricted dependency versions and expanded installation/input/output guidance.
- Added repository instructions and a model-card template.

## Before public scientific release

1. Supply the four final research checkpoints and verify strict loading.
2. Compare exported scores/embeddings with the original inference environment.
3. Confirm the exact configuration for each checkpoint; the YAML values remain
   templates. In particular, verify learning rate, batch size, masking behavior,
   augmentation, stopping rule, and checkpoint selection.
4. Complete weight checksums, download URLs, code/model revisions, and manifests.
5. Select software and model licenses; no license is assigned by this package.
6. Add manuscript-specific evaluation scripts, source data, and dataset/split
   identifiers when the corresponding analyses are frozen.

## Implementation notes

PDLM retains the positional construction of the original implementation: explicit
learned position embeddings at the input and rotary attention in the encoder.
When Transformers does not instantiate the learned table, the wrapper creates it
to retain the expected parameter key. Do not remove this table when preparing
existing checkpoints.

Uniform masking chooses a fixed number of positions, with a minimum of one.
Domain masking samples annotated and non-annotated residues independently; its
optional minimum-one fallback is enabled in the templates. This setting still
requires confirmation against the final research runs.

The training runner does not include distributed execution, mixed precision,
automatic early stopping, or full random-state restoration. It is runnable core
code, not a verified reproduction of every original training convenience.
