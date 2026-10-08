# PDLM model card template

Complete one card per variant before uploading weights. Replace this title and
all bracketed fields. Do not present unverified settings as final metadata.

## Model description

PDLM jointly encodes protein and DNA using an ESM2-initialized shared Transformer,
separate token embeddings, and separate MLM prediction heads.

- Variant: [PDLM-35M-base / PDLM-35M-domain / PDLM-150M-base / PDLM-150M-domain]
- ESM2 backbone: [official identifier]
- Masking: [uniform / domain-aware]
- Code repository and release tag: [URL and tag]
- Weight filename and SHA256: [filename and checksum]
- Software license: [confirmed license]
- Model license: [confirmed license]
- Manuscript: [title, authors, DOI or preprint URL]

## Use

The model supports zero-shot mutation scoring, protein-conditioned DNA scoring,
and representation extraction. Load it using the linked `pdlm` package, not a
standard ESM2 `AutoModel` loader. Include installation and inference commands
for this variant.

## Training

- Data sources and versions: [identifiers or citations]
- Dataset sizes and split policy: [confirmed values]
- Configuration: [verified filename]
- Software and hardware: [versions and devices]
- Checkpoint epoch and selection criterion: [confirmed values]

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
