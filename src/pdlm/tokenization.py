from dataclasses import dataclass

import torch

from .configuration import DNA_VOCAB, SEGMENT_IDS


def normalize_dna(sequence):
    sequence = sequence.strip()
    if not sequence:
        raise ValueError("DNA sequence is empty")
    if " " in sequence:
        tokens = sequence.split()
        return " ".join(token if token.startswith("d") else f"d{token.upper()}" for token in tokens)
    return " ".join(f"d{base}" for base in sequence.upper())


def reverse_complement(sequence):
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    bases = [token[1:] if token.startswith("d") else token for token in normalize_dna(sequence).split()]
    return " ".join(f"d{complement.get(base, 'N')}" for base in reversed(bases))


class DNATokenizer:
    def __init__(self):
        self.vocab = dict(DNA_VOCAB)
        self.id_to_token = {value: key for key, value in self.vocab.items()}
        self.mask_token_id = self.vocab["[MASK]"]
        self.unk_token_id = self.vocab["[UNK]"]
        self.sep_token_id = self.vocab["[SEP]"]

    def encode(self, sequence):
        return [self.vocab.get(token, self.unk_token_id) for token in normalize_dna(sequence).split()]

    def decode(self, token_ids):
        return " ".join(self.id_to_token.get(int(token_id), "[UNK]") for token_id in token_ids)

    def __len__(self):
        return len(self.vocab)


@dataclass
class EncodedPair:
    input_ids: torch.Tensor
    segment_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    protein_positions: list
    dna_positions: list
    protein_source_indices: list


def encode_pair(protein_seq, dna_seq, protein_tokenizer, dna_tokenizer=None,
                max_length=800, swap_order=False):
    dna_tokenizer = dna_tokenizer or DNATokenizer()
    protein_seq = protein_seq.strip().upper()
    protein_ids = protein_tokenizer.encode(protein_seq, add_special_tokens=False)
    if not protein_ids:
        raise ValueError("Protein sequence is empty")
    if dna_seq is None:
        if max_length < 3:
            raise ValueError("max_length must be at least 3 for protein-only input")
        protein_ids = protein_ids[:max_length - 2]
        input_ids = [protein_tokenizer.cls_token_id] + protein_ids + [protein_tokenizer.eos_token_id]
        return EncodedPair(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            segment_ids=torch.tensor([0] + [1] * len(protein_ids) + [0], dtype=torch.long),
            token_type_ids=torch.zeros(len(input_ids), dtype=torch.long),
            attention_mask=torch.ones(len(input_ids), dtype=torch.long),
            protein_positions=list(range(1, len(protein_ids) + 1)),
            dna_positions=[],
            protein_source_indices=list(range(len(protein_ids))),
        )
    dna_ids = dna_tokenizer.encode(dna_seq)

    available = max_length - 3
    if len(dna_ids) > available - 1:
        dna_ids = dna_ids[:max(1, available // 2)]
    target_protein = min(len(protein_ids), available - len(dna_ids))
    if target_protein < 1:
        raise ValueError("max_length is too small for the supplied pair")
    trim = max(0, (len(protein_ids) - target_protein) // 2)
    protein_ids = protein_ids[trim:trim + target_protein]
    protein_source_indices = list(range(trim, trim + len(protein_ids)))

    cls_id = protein_tokenizer.cls_token_id
    eos_id = protein_tokenizer.eos_token_id
    sep_id = dna_tokenizer.sep_token_id

    if not swap_order:
        input_ids = [cls_id] + protein_ids + [sep_id] + dna_ids + [eos_id]
        segment_ids = ([SEGMENT_IDS["special"]] +
                       [SEGMENT_IDS["protein"]] * len(protein_ids) +
                       [SEGMENT_IDS["special"]] +
                       [SEGMENT_IDS["dna"]] * len(dna_ids) +
                       [SEGMENT_IDS["special"]])
        token_type_ids = [0] + [0] * len(protein_ids) + [1] + [1] * len(dna_ids) + [0]
    else:
        input_ids = [cls_id] + dna_ids + [sep_id] + protein_ids + [eos_id]
        segment_ids = ([SEGMENT_IDS["special"]] +
                       [SEGMENT_IDS["dna"]] * len(dna_ids) +
                       [SEGMENT_IDS["special"]] +
                       [SEGMENT_IDS["protein"]] * len(protein_ids) +
                       [SEGMENT_IDS["special"]])
        token_type_ids = [0] + [1] * len(dna_ids) + [1] + [0] * len(protein_ids) + [0]

    segment_tensor = torch.tensor(segment_ids, dtype=torch.long)
    return EncodedPair(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        segment_ids=segment_tensor,
        token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
        attention_mask=torch.ones(len(input_ids), dtype=torch.long),
        protein_positions=(segment_tensor == SEGMENT_IDS["protein"]).nonzero(as_tuple=True)[0].tolist(),
        dna_positions=(segment_tensor == SEGMENT_IDS["dna"]).nonzero(as_tuple=True)[0].tolist(),
        protein_source_indices=protein_source_indices,
    )


def pad_encoded_pairs(encoded_pairs, pad_token_id, pad_to_multiple=8):
    maximum = max(len(item.input_ids) for item in encoded_pairs)
    if pad_to_multiple:
        maximum = ((maximum + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    batch_size = len(encoded_pairs)
    result = {
        "input_ids": torch.full((batch_size, maximum), pad_token_id, dtype=torch.long),
        "segment_ids": torch.zeros((batch_size, maximum), dtype=torch.long),
        "token_type_ids": torch.zeros((batch_size, maximum), dtype=torch.long),
        "attention_mask": torch.zeros((batch_size, maximum), dtype=torch.long),
    }
    for index, item in enumerate(encoded_pairs):
        length = len(item.input_ids)
        for key in result:
            result[key][index, :length] = getattr(item, key)
    return result
