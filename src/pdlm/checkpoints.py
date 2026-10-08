from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import EsmConfig

from .modeling import ProteinDNAMLMModel


def _state_dict(payload):
    state = payload.get("model_state_dict", payload)
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_checkpoint(checkpoint_path, esm2_model, device="cpu", strict=True):
    checkpoint_path = Path(checkpoint_path)
    payload = (load_file(str(checkpoint_path), device="cpu")
               if checkpoint_path.suffix == ".safetensors" else
               torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    state = _state_dict(payload)
    dna_vocab_size = state["dna_embeddings.weight"].shape[0]
    config = EsmConfig.from_pretrained(esm2_model)
    model = ProteinDNAMLMModel.from_esm_config(config, dna_vocab_size=dna_vocab_size)
    model.load_state_dict(state, strict=strict)
    model.to(device).eval()
    return model, payload
