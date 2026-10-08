import json
import random
from pathlib import Path

from torch.utils.data import Dataset

from .tokenization import normalize_dna, reverse_complement


class ProteinDNADataset(Dataset):
    def __init__(self, path, training=False, reverse_complement_prob=0.0,
                 swap_order_prob=0.0, seed=42):
        self.path = Path(path)
        self.training = training
        self.reverse_complement_prob = reverse_complement_prob
        self.swap_order_prob = swap_order_prob
        self.random = random.Random(seed)
        with self.path.open() as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        raw = self.records[index]
        dna = normalize_dna(raw["dna_seq"])
        if self.training and self.random.random() < self.reverse_complement_prob:
            dna = reverse_complement(dna)
        return {
            "id": raw.get("id", raw.get("protein_id", f"sample_{index}")),
            "protein_seq": raw["protein_seq"],
            "dna_seq": dna,
            "swap_order": self.training and self.random.random() < self.swap_order_prob,
            "binding_mask": raw.get("binding_mask", []),
            "has_annotation": bool(raw.get("has_annotation", False)),
        }

