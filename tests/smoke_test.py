import sys
from pathlib import Path

import torch
from transformers import EsmConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdlm.data import ProteinDNADataset
from pdlm.embeddings import extract_embedding
from pdlm.masking import ProteinDNACollator
from pdlm.modeling import ProteinDNAMLMModel
from pdlm.scoring import dna_pll, protein_delta_ll
from pdlm.tokenization import DNATokenizer, encode_pair


class TinyProteinTokenizer:
    def __init__(self):
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.tokens = {aa: index + 4 for index, aa in enumerate(amino_acids)}
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = 32

    def encode(self, sequence, add_special_tokens=False):
        ids = [self.tokens.get(aa, self.unk_token_id) for aa in sequence]
        return ([self.cls_token_id] + ids + [self.eos_token_id]) if add_special_tokens else ids

    def convert_tokens_to_ids(self, token):
        return self.tokens.get(token, self.unk_token_id)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer = TinyProteinTokenizer()
    config = EsmConfig(
        vocab_size=33,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        pad_token_id=1,
        mask_token_id=32,
        position_embedding_type="rotary",
        token_dropout=False,
    )
    model = ProteinDNAMLMModel.from_esm_config(config, dna_vocab_size=8, dropout=0.0)
    model.eval()
    encoded = encode_pair("MKTAYIAK", "ATCG", tokenizer, DNATokenizer(), max_length=64)
    output = model(encoded.input_ids[None], encoded.segment_ids[None],
                   encoded.token_type_ids[None], encoded.attention_mask[None])
    assert output["protein_logits"].shape == (1, len(encoded.input_ids), 33)
    assert output["dna_logits"].shape == (1, len(encoded.input_ids), 8)

    dataset = ProteinDNADataset(ROOT / "examples" / "toy_train.jsonl", training=False)
    uniform = ProteinDNACollator(
        tokenizer, {"strategy": "uniform", "protein_probability": 0.15,
                    "dna_probability": 0.20}, max_length=64, seed=42
    )
    batch = uniform([dataset[0], dataset[1]])
    assert batch["protein_labels"].ne(-100).sum() >= 2
    assert batch["dna_labels"].ne(-100).sum() >= 2

    domain = ProteinDNACollator(
        tokenizer, {"strategy": "domain", "binding_probability": 0.30,
                    "other_probability": 0.10, "fallback_probability": 0.15,
                    "dna_probability": 0.20, "ensure_one_mask": True}, max_length=64, seed=42
    )
    domain_batch = domain([dataset[0], dataset[1]])
    assert domain_batch["protein_labels"].ne(-100).sum() >= 2

    dll = protein_delta_ll(model, tokenizer, "MKTAYIAK", "ATCG", "K2A", max_length=64)
    pll = dna_pll(model, tokenizer, "MKTAYIAK", "ATCG", max_length=64)
    embedding = extract_embedding(model, tokenizer, "MKTAYIAK", "ATCG", mode="concat", max_length=64)
    assert isinstance(dll, float)
    assert pll["dna_length"] == 4
    assert embedding.shape == (64,)
    protein_only = encode_pair("MKTAYIAK", None, tokenizer, max_length=64)
    assert len(protein_only.input_ids) == 10
    assert protein_only.dna_positions == []
    assert isinstance(protein_delta_ll(model, tokenizer, "MKTAYIAK", None, "K2A", max_length=64), float)
    assert extract_embedding(model, tokenizer, "MKTAYIAK", None, mode="protein_mean", max_length=64).shape == (32,)
    try:
        dna_pll(model, tokenizer, "MKTAYIAK", "A" * 100, max_length=64)
    except ValueError as error:
        assert "length budget" in str(error)
    else:
        raise AssertionError("DNA PLL must reject silent DNA truncation")
    print("PDLM smoke test passed")


if __name__ == "__main__":
    main()
