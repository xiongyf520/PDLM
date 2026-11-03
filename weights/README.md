# Model weights

PDLM weights are distributed separately from the code repository.
Model files are excluded by `.gitignore`. Place local weights here or pass
another path to `--checkpoint`.

## Export for distribution

Export only model tensors. Optimizer state, scheduler state, training records,
and machine-specific configuration are unnecessary for inference.

```bash
python scripts/export_weights.py \
  --checkpoint PATH_TO_CONFIRMED_RESEARCH_CHECKPOINT.pt \
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
The manifest's `config_file` field is `TBD` until the actual training settings
are confirmed. Never point it to a synthetic demo configuration as though that
configuration produced the research weights.

Before publication, strictly load each exported model and compare scores and
embeddings with the original evaluation environment on fixed example inputs.

## Hugging Face repository

Keep implementation code on GitHub and publish weights on the Hugging Face Hub.
Use [xiongyf/PDLM](https://huggingface.co/xiongyf/PDLM), containing all four
variants. Use explicit weight filenames to select the size and masking strategy.

The repository is currently private while the weights are being prepared.
A model repository is sufficient; a Space is only needed
if you later want to host an interactive application.
Choose software and model licenses with the authors before public release.
Do not substitute an arbitrary license for the pending decision.

Prepare the repository as follows:

```text
PDLM/
├── README.md
├── PDLM-35M-base.safetensors
├── PDLM-35M-domain.safetensors
├── PDLM-150M-base.safetensors
├── PDLM-150M-domain.safetensors
├── checkpoint_manifest.tsv
└── esm2/
    ├── 35M/
    │   ├── config.json
    │   ├── vocab.txt
    │   ├── tokenizer_config.json
    │   └── special_tokens_map.json
    └── 150M/
        ├── config.json
        ├── vocab.txt
        ├── tokenizer_config.json
        └── special_tokens_map.json
```

Use [the model-card template](../docs/MODEL_CARD_TEMPLATE.md) for `README.md`.
Record verified training settings in the card or an accompanying
`pretraining_config_<variant>.yaml` for each variant. The full research training
dataset is not included in this code release; describe its availability accurately.
The two `esm2/` subfolders contain the matching ESM2 configuration and tokenizer
for each size; base and domain variants of the same size share these files.
The PDLM state dictionary already includes the encoder; a second copy of
ESM2 pretrained weights is unnecessary. Export final research weights, not
weights produced by the synthetic demo.

Upload the prepared files through the repository's **Files and versions**
interface or the optional `huggingface_hub` CLI. Install that tool in a separate
environment if its dependencies conflict with the pinned PDLM environment.
Creating a private repository does not publish it; visibility can be changed
in **Settings** after loading and inference checks pass.
Record the model revision and checksum alongside the corresponding GitHub
release tag.

After installing the optional CLI and authenticating, a download example is:

```bash
hf download xiongyf/PDLM --local-dir weights/PDLM
pdlm-score \
  --checkpoint weights/PDLM/PDLM-35M-domain.safetensors \
  --esm2-model weights/PDLM/esm2/35M \
  --input examples/toy_pairs.jsonl \
  --output outputs/dna_pll.jsonl \
  --mode dna-pll
```

The command above downloads the entire repository. To download only the
35M-domain variant and its configuration/tokenizer:

```bash
hf download xiongyf/PDLM --local-dir weights/PDLM \
  --include "PDLM-35M-domain.safetensors" "esm2/35M/*" "README.md" "checkpoint_manifest.tsv"
```

These are custom PDLM weights. Hub hosting does not make them compatible with
`AutoModel.from_pretrained()`; users must install this package.

Official guides: [uploading models](https://huggingface.co/docs/hub/models-uploading),
[release checklist](https://huggingface.co/docs/hub/model-release-checklist),
[selective downloads](https://huggingface.co/docs/huggingface_hub/guides/download),
[visibility settings](https://huggingface.co/docs/hub/repositories-settings).
