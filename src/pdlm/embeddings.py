import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .checkpoints import load_checkpoint
from .tokenization import encode_pair


@torch.no_grad()
def extract_embedding(model, protein_tokenizer, protein_seq, dna_seq,
                      mode="concat", device="cpu", max_length=800):
    encoded = encode_pair(protein_seq, dna_seq, protein_tokenizer, max_length=max_length)
    output = model(
        encoded.input_ids.unsqueeze(0).to(device),
        encoded.segment_ids.unsqueeze(0).to(device),
        encoded.token_type_ids.unsqueeze(0).to(device),
        encoded.attention_mask.unsqueeze(0).to(device),
    )["last_hidden_state"][0].cpu()
    protein = output[encoded.protein_positions]
    dna = output[encoded.dna_positions]
    joint_positions = encoded.protein_positions + encoded.dna_positions
    if mode == "cls":
        return output[0].numpy()
    if mode == "joint_mean":
        return output[joint_positions].mean(0).numpy()
    if mode == "protein_mean":
        return protein.mean(0).numpy()
    if mode == "dna_mean":
        if not encoded.dna_positions:
            raise ValueError("dna_mean requires DNA input")
        return dna.mean(0).numpy()
    if mode == "concat":
        if not encoded.dna_positions:
            raise ValueError("concat requires DNA input; use protein_mean for protein-only representations")
        return torch.cat([protein.mean(0), dna.mean(0)]).numpy()
    if mode == "per_token":
        return output[joint_positions].numpy()
    raise ValueError(f"Unknown embedding mode: {mode}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--esm2-model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["cls", "joint_mean", "protein_mean", "dna_mean", "concat", "per_token"], default="concat")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=800)
    args = parser.parse_args(argv)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.esm2_model)
    model, _ = load_checkpoint(args.checkpoint, args.esm2_model, device)
    records = []
    with Path(args.input).open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    ids, values = [], []
    for index, record in enumerate(records):
        ids.append(record.get("id", f"sample_{index}"))
        values.append(extract_embedding(model, tokenizer, record["protein_seq"],
                                        record.get("dna_seq"), args.mode, device, args.max_length))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "per_token":
        np.savez_compressed(output_path, ids=np.asarray(ids), embeddings=np.asarray(values, dtype=object))
    else:
        np.savez_compressed(output_path, ids=np.asarray(ids), embeddings=np.stack(values))


if __name__ == "__main__":
    main()
