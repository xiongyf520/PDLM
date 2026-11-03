# Synthetic examples

These files demonstrate PDLM input formats and executable workflows.
The full research training dataset is not distributed with this repository.
No example should be used to reproduce or report manuscript performance.

## Training demo

`toy_train.jsonl` contains 80 synthetic pairs; `toy_val.jsonl` contains 20.
Both are generated with seed 42. Protein sequences (20–40 residues), DNA
sequences (8–16 bases), pairings, and binding masks are artificial.
Each partition contains equal numbers of records with and without synthetic
annotations. Annotated records mark a five-residue interval; this interval
does not represent an experimentally supported binding domain.
Neither protein sequences nor DNA sequences overlap between the partitions.
These demo partitions do not change the original research partitions.

Regenerate the files from the repository root:

```bash
python scripts/generate_demo.py
```

Generation is deterministic. Existing files are accepted only when their
contents already match; otherwise, select another directory with
`--output-dir` to avoid overwriting data.

Required training fields are `protein_seq` and `dna_seq`.
Optional fields are `id`, `has_annotation`, and `binding_mask`.
When annotated, the binary mask must align with the complete protein sequence.
DNA can be plain bases or space-separated `dA dT dC dG` tokens.

Run either strategy after installation:

```bash
pdlm-train --config configs/demo/base.yaml
pdlm-train --config configs/demo/domain.yaml
```

The YAML files initialize ESM2-35M and run one epoch with batch size 2,
maximum input length 64, no warmup, and seed 42. They require pretrained ESM2
weights and may download them on first use. Runtime depends on hardware.
Outputs are `config.yaml`, `metrics.json`, and `checkpoint_best.pt`
under the configured output directory. Select a new directory for a fresh run.

For a small, offline CPU training and inference check without downloading ESM2:

```bash
python tests/integration_test.py
```

This check uses a randomly initialized tiny backbone, not a trained PDLM model.

## Application examples

| File | Purpose |
| --- | --- |
| `toy_pairs.jsonl` | Protein-conditioned DNA PLL and joint embeddings |
| `protein_only_variants.jsonl` | Single substitutions without DNA |
| `protein_variants.jsonl` | DNA-conditioned single substitutions |
| `dna_delta_pll.jsonl` | Reference-versus-variant DNA mean PLL |

Application commands and output definitions are in the main README.
Scores from demo-trained or randomly initialized models have no validated
biological interpretation.
