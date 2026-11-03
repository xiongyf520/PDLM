# Release status

## Version 0.3.0

Release scope: core model, executable training demos, zero-shot scoring, and
representation extraction. The full research training dataset and trained model
weights are not distributed with this code package. Synthetic demo training
demonstrates software operation, not manuscript performance.

Changes prepared on 2026-11-03:

- Replaced four unverified pretraining templates with base/domain demo YAMLs.
- Added 80 training and 20 validation synthetic pairs, with seed 42 and no shared
  protein or DNA sequences. Pairings and domain annotations are artificial.
- Added deterministic demo generation and an example-format guide.
- Removed three redundant script wrappers; installed CLI commands remain.
- Removed the duplicate requirements file; dependencies live in pyproject.toml.
- Merged repository guidance into the README and weight documentation.
- Documented one shared Hugging Face repository for all four variants, with
  shared configuration/tokenizer folders by model size and selective downloads.
- Model hosting target: https://huggingface.co/xiongyf/PDLM (currently private;
  all four research weight files uploaded).
- Preserved model architecture, tokenization, training, masking, scoring,
  embedding extraction, and strict checkpoint/export behavior.
- Extended offline checks to exercise both demo configurations and demo data.

### Local verification on 2026-11-03

Editable installation passed in a temporary CPU test environment using Python
3.11.5, PyTorch 2.0.1, Transformers 4.44.2, NumPy 1.25.2, PyYAML 6.0.1,
and Safetensors 0.8.0.

- Smoke test: passed (49.86 s, including cold process startup).
- Integration test: passed (35.71 s). Checks covered deterministic generation,
  80/20 counts, unique IDs/sequences, synthetic annotation alignment, refusal
  to overwrite different demo data, both demo training strategies, resume,
  strict loading, tensor-identical export, scoring, and embeddings.
- Installed `pdlm-train`, `pdlm-score`, and `pdlm-embed` help commands: passed.
- Python syntax, demo configuration paths, local Markdown links, and manifest
  checks: passed.

The tests use a tiny randomly initialized backbone and do not download ESM2
weights or validate research-model performance.

Current limitations:

- Fixed-input equivalence checks passed for the final 35M-base, 35M-domain,
  and 150M-base checkpoints. The remaining 150M-domain inference validation
  was not completed. These are software checks, not benchmark reproduction.
- The demo YAMLs are not the configurations of the final research weights.
  Recorded scalar settings recovered from the checkpoints are hosted separately
  in the model repository.
- The model repository contains the weight manifest and recorded scalar settings.
  Checkpoint-selection details and software/model licenses remain pending.
- Manuscript-specific downstream predictors and scientific benchmarks are
  outside this package.
- No claim of complete manuscript reproduction is made.

### Private weight upload on 2026-11-03

All four Safetensors files and their configuration/tokenizer files are available
to authorized users at model revision
`4c5bafef7730b2138ee9d53fbc85272251bdd7ce`.
Remote file sizes and SHA256 checksums match the exported files. Optimizer and
scheduler states and the full research training dataset are not included.
The repository remains private; no PDLM license was assigned during upload.

## Version 0.2.0 — historical record

The following records describe the previous release before the demo cleanup;
its four training templates have been replaced in 0.3.0.

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
