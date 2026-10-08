import random

import torch

from .configuration import SEGMENT_IDS
from .tokenization import DNATokenizer, encode_pair, pad_encoded_pairs


class ProteinDNACollator:
    def __init__(self, protein_tokenizer, masking, max_length=800, seed=42):
        self.protein_tokenizer = protein_tokenizer
        self.dna_tokenizer = DNATokenizer()
        self.masking = masking
        self.max_length = max_length
        self.random = random.Random(seed)

    def _uniform_positions(self, positions, probability):
        if not positions:
            return []
        count = max(1, int(len(positions) * probability))
        return self.random.sample(positions, min(count, len(positions)))

    def _protein_positions(self, encoded, sample):
        strategy = self.masking.get("strategy", "uniform")
        if strategy == "uniform" or not sample.get("has_annotation", False):
            probability = (self.masking.get("protein_probability", 0.15)
                           if strategy == "uniform" else self.masking.get("fallback_probability", 0.15))
            return self._uniform_positions(encoded.protein_positions, probability)
        binding_mask = sample.get("binding_mask", [])
        chosen = []
        for absolute_position, source_index in zip(encoded.protein_positions, encoded.protein_source_indices):
            is_binding = source_index < len(binding_mask) and int(binding_mask[source_index]) == 1
            probability = (self.masking.get("binding_probability", 0.30)
                           if is_binding else self.masking.get("other_probability", 0.10))
            if self.random.random() < probability:
                chosen.append(absolute_position)
        if not chosen and self.masking.get("ensure_one_mask", True):
            chosen = self.random.sample(encoded.protein_positions, 1)
        return chosen

    def __call__(self, samples):
        encoded = [
            encode_pair(
                sample["protein_seq"], sample["dna_seq"], self.protein_tokenizer,
                self.dna_tokenizer, self.max_length, sample.get("swap_order", False),
            )
            for sample in samples
        ]
        batch = pad_encoded_pairs(encoded, self.protein_tokenizer.pad_token_id)
        protein_labels = torch.full_like(batch["input_ids"], -100)
        dna_labels = torch.full_like(batch["input_ids"], -100)
        for index, (item, sample) in enumerate(zip(encoded, samples)):
            protein_positions = self._protein_positions(item, sample)
            dna_positions = self._uniform_positions(
                item.dna_positions, self.masking.get("dna_probability", 0.20)
            )
            for position in protein_positions:
                protein_labels[index, position] = batch["input_ids"][index, position]
                batch["input_ids"][index, position] = self.protein_tokenizer.mask_token_id
            for position in dna_positions:
                dna_labels[index, position] = batch["input_ids"][index, position]
                batch["input_ids"][index, position] = self.dna_tokenizer.mask_token_id
        batch["protein_labels"] = protein_labels
        batch["dna_labels"] = dna_labels
        return batch

