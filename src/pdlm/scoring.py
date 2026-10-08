import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .checkpoints import load_checkpoint
from .tokenization import DNATokenizer, encode_pair, normalize_dna


MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _forward(model, encoded, device, masked_ids=None):
    input_ids = encoded.input_ids if masked_ids is None else masked_ids
    return model(
        input_ids.unsqueeze(0).to(device),
        encoded.segment_ids.unsqueeze(0).to(device),
        encoded.token_type_ids.unsqueeze(0).to(device),
        encoded.attention_mask.unsqueeze(0).to(device),
    )


@torch.no_grad()
def protein_delta_ll(model, protein_tokenizer, protein_seq, dna_seq, mutation,
                     device="cpu", max_length=800):
    match = MUTATION.match(mutation.upper())
    if not match:
        raise ValueError(f"Invalid mutation: {mutation}; expected A123V")
    wildtype, position_text, mutant = match.groups()
    protein_seq = protein_seq.strip().upper()
    source_index = int(position_text) - 1
    if source_index < 0 or source_index >= len(protein_seq) or protein_seq[source_index] != wildtype:
        raise ValueError(f"Mutation {mutation} does not match the supplied protein")
    encoded = encode_pair(protein_seq, dna_seq, protein_tokenizer, max_length=max_length)
    if source_index not in encoded.protein_source_indices:
        raise ValueError(f"Mutation {mutation} was removed by sequence truncation")
    token_position = encoded.protein_positions[encoded.protein_source_indices.index(source_index)]
    masked = encoded.input_ids.clone()
    masked[token_position] = protein_tokenizer.mask_token_id
    logits = _forward(model, encoded, device, masked)["protein_logits"][0, token_position]
    log_probabilities = F.log_softmax(logits, dim=-1)
    wt_id = protein_tokenizer.convert_tokens_to_ids(wildtype)
    mut_id = protein_tokenizer.convert_tokens_to_ids(mutant)
    return float((log_probabilities[mut_id] - log_probabilities[wt_id]).cpu())


@torch.no_grad()
def dna_pll(model, protein_tokenizer, protein_seq, dna_seq, device="cpu", max_length=800):
    if dna_seq is None:
        raise ValueError("DNA PLL requires dna_seq")
    dna_tokenizer = DNATokenizer()
    encoded = encode_pair(protein_seq, dna_seq, protein_tokenizer, dna_tokenizer, max_length)
    if len(encoded.dna_positions) != len(normalize_dna(dna_seq).split()):
        raise ValueError("DNA exceeds the input length budget; increase max_length or shorten the DNA explicitly")
    values = []
    for position in encoded.dna_positions:
        masked = encoded.input_ids.clone()
        target = int(masked[position])
        masked[position] = dna_tokenizer.mask_token_id
        logits = _forward(model, encoded, device, masked)["dna_logits"][0, position]
        values.append(float(F.log_softmax(logits, dim=-1)[target].cpu()))
    return {"dna_pll_sum": sum(values), "dna_pll_mean": sum(values) / len(values),
            "dna_mean_loss": -sum(values) / len(values), "dna_length": len(values)}


def _load_records(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--esm2-model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["protein-dll", "dna-pll", "dna-dpll"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=800)
    args = parser.parse_args(argv)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.esm2_model)
    model, _ = load_checkpoint(args.checkpoint, args.esm2_model, device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        for index, record in enumerate(_load_records(args.input)):
            result = {"id": record.get("id", f"sample_{index}")}
            if args.mode == "protein-dll":
                result["protein_delta_ll"] = protein_delta_ll(
                    model, tokenizer, record["protein_seq"], record.get("dna_seq"),
                    record["mutation"], device, args.max_length
                )
            elif args.mode == "dna-pll":
                result.update(dna_pll(model, tokenizer, record["protein_seq"],
                                      record["dna_seq"], device, args.max_length))
            else:
                reference_protein = record.get("reference_protein_seq", record.get("protein_seq"))
                variant_protein = record.get("variant_protein_seq", record.get("protein_seq"))
                reference_dna = record.get("reference_dna_seq", record.get("dna_seq"))
                variant_dna = record.get("variant_dna_seq", record.get("dna_seq"))
                if any(value is None for value in [reference_protein, variant_protein,
                                                   reference_dna, variant_dna]):
                    raise ValueError("DNA delta PLL requires both reference and variant protein-DNA inputs")
                reference = dna_pll(model, tokenizer, reference_protein, reference_dna,
                                    device, args.max_length)
                variant = dna_pll(model, tokenizer, variant_protein, variant_dna,
                                  device, args.max_length)
                result.update({"reference_dna_pll_mean": reference["dna_pll_mean"],
                               "variant_dna_pll_mean": variant["dna_pll_mean"],
                               "dna_delta_pll": variant["dna_pll_mean"] - reference["dna_pll_mean"]})
            output.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
