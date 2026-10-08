# Repository and model hosting

## Code repository

Upload the contents of `pdlm-release/` as the root of a GitLab or GitHub project.
Include `src/`, `scripts/`, `configs/`, `examples/`, `tests/`, `docs/`,
`pyproject.toml`, `requirements.txt`, `README.md`, `.gitignore`, and the small
weight-manifest templates. Exclude research datasets, model files, generated
outputs, environments, caches, credentials, and training checkpoints.

A private repository is appropriate during preparation. During peer review,
provide editors and reviewers with working access through the journal's accepted
mechanism; a private URL alone does not grant access. Tag the finalized manuscript
version, then enable public visibility when the authors decide to release it.

## Hugging Face models

Use a separate model repository for each variant. Each can initially be private.

```text
PDLM-35M-domain/
├── README.md                    # Completed model card
├── model.safetensors            # PDLM state dictionary
├── checkpoint_manifest.tsv
├── pretraining_config.yaml      # Verified configuration with relative paths
└── esm2/                        # Matching ESM2 configuration and tokenizer
    ├── config.json
    ├── vocab.txt
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

Distribute inference weights without optimizer or scheduler state.
The PDLM state dictionary already contains the trained encoder; do not include
a redundant copy of the original ESM2 pretrained weights. Record the code tag,
model revision, and weight checksum in each model card.

```bash
hf download YOUR_NAMESPACE/PDLM-35M-domain --local-dir weights/PDLM-35M-domain
pdlm-score \
  --checkpoint weights/PDLM-35M-domain/model.safetensors \
  --esm2-model weights/PDLM-35M-domain/esm2 \
  --input examples/toy_pairs.jsonl \
  --output outputs/dna_pll.jsonl \
  --mode dna-pll
```

`hf` is an optional upload/download tool, not a core PDLM dependency.
Private models require an authorized Hugging Face account. See the official
[upload guide](https://huggingface.co/docs/hub/models-uploading) and
[visibility guide](https://huggingface.co/docs/hub/repositories-settings).

## Before public release

- Confirm all four checkpoint identities and actual training settings.
- Compare exported weights with the original inference implementation.
- Select software and model licenses and complete the model cards.
- Publish the exact code tag and corresponding model revisions.
- Add a persistent archive DOI for the manuscript version when available.

Database download and cleaning pipelines are outside this core package.
The manuscript reproducibility release may additionally need the final evaluation
scripts, source-data tables, dataset identifiers, and split assignments.
Add these after the corresponding analyses are finalized; do not upload the
entire research workspace.
