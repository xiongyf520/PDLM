# Model weights

PDLM weights are distributed separately from the code repository.
Model files are excluded by `.gitignore`. Place local weights here or pass
another path to `--checkpoint`.

## Export for distribution

Export only model tensors. Optimizer state, scheduler state, training records,
and machine-specific configuration are unnecessary for inference.

```bash
python scripts/export_weights.py \
  --checkpoint outputs/pdlm_35m_domain/checkpoint_best.pt \
  --output weights/PDLM-35M-domain.safetensors
```

The command prints a SHA256 checksum and leaves the input intact. It supports
trusted checkpoints containing tensors and basic metadata. PyTorch files are
loaded with `weights_only=True`.

| Public name | Matching ESM2 configuration and tokenizer |
| --- | --- |
| PDLM-35M-base | `facebook/esm2_t12_35M_UR50D` |
| PDLM-35M-domain | `facebook/esm2_t12_35M_UR50D` |
| PDLM-150M-base | `facebook/esm2_t30_150M_UR50D` |
| PDLM-150M-domain | `facebook/esm2_t30_150M_UR50D` |

Complete PDLM weights already contain the trained encoder. Inference only needs
the matching ESM2 configuration and tokenizer; it does not need a second ESM2
pretrained weight file. For offline use, supply these small files in an `esm2/`
directory through `--esm2-model`.

These are custom PDLM checkpoints, not standard `AutoModel` checkpoints.

## Manifest and validation

Complete `checkpoint_manifest_template.tsv` and save the confirmed version as
`checkpoint_manifest.tsv`. Add the exact checkpoint filename, model URL,
backbone, masking strategy, epoch, selection metric, SHA256, configuration,
code version, and license. Do not infer final training values from template defaults.

Before publication, strictly load each exported model and compare scores and
embeddings with the original evaluation environment on fixed example inputs.
