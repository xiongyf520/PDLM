# PDLM model card template

Complete one card for the shared PDLM model repository. Replace this title and
all bracketed fields. Do not present unverified settings as final metadata.

## Model description

PDLM jointly encodes protein and DNA using an ESM2-initialized shared Transformer,
separate token embeddings, and separate MLM prediction heads.

- Model repository: [xiongyf/PDLM](https://huggingface.co/xiongyf/PDLM)
- Code repository and release tag: [URL and tag]
- Software license: [confirmed license]
- Model license: [confirmed license]
- Manuscript: [title, authors, DOI or preprint URL]

| Variant | Weight file | ESM2 configuration/tokenizer | Masking | SHA256 |
| --- | --- | --- | --- | --- |
| PDLM-35M-base | PDLM-35M-base.safetensors | esm2/35M | uniform | [checksum] |
| PDLM-35M-domain | PDLM-35M-domain.safetensors | esm2/35M | domain-aware | [checksum] |
| PDLM-150M-base | PDLM-150M-base.safetensors | esm2/150M | uniform | [checksum] |
| PDLM-150M-domain | PDLM-150M-domain.safetensors | esm2/150M | domain-aware | [checksum] |

The 35M variants use `facebook/esm2_t12_35M_UR50D`; the 150M variants use
`facebook/esm2_t30_150M_UR50D`.

## Use

The model supports zero-shot mutation scoring, protein-conditioned DNA scoring,
and representation extraction. Load it using the linked `pdlm` package, not a
standard ESM2 `AutoModel` loader. Include installation and inference commands
for the selected variant. Document how to download an individual weight file
and the matching configuration/tokenizer from this shared repository.

## Training

- Data sources and versions: [identifiers or citations]
- Dataset sizes and split policy: [confirmed values]
- Configuration: [verified filename]
- Software and hardware: [versions and devices]
- Checkpoint epoch and selection criterion: [confirmed values]

Report variant-specific settings separately when they differ.

## Evaluation

[Describe the evaluated tasks, splits, metrics, and model-specific results.
Distinguish zero-shot scoring, frozen-feature predictors, and fine-tuning.]

## Limitations

Scores are model likelihoods, not calibrated binding probabilities, physical
affinities, or guaranteed editing efficiencies. Predictions depend on sequence
context and truncation. State the evaluated sequence families and assay settings.

## Provenance

The model was initialized from ESM2. Retain upstream attribution and applicable
license notices. Record the code and checkpoint revisions used for the manuscript.
