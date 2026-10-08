import torch
import torch.nn as nn
from transformers import EsmModel


class ProteinDNAMLMModel(nn.Module):
    def __init__(self, esm2, dna_vocab_size=8, dropout=0.1):
        super().__init__()
        self.esm2 = esm2
        self.config = esm2.config
        self.hidden_size = self.config.hidden_size
        if not hasattr(self.esm2.embeddings, "position_embeddings"):
            self.esm2.embeddings.position_embeddings = nn.Embedding(
                self.config.max_position_embeddings,
                self.hidden_size,
                padding_idx=self.config.pad_token_id,
            )
            nn.init.normal_(self.esm2.embeddings.position_embeddings.weight, std=0.02)
            with torch.no_grad():
                self.esm2.embeddings.position_embeddings.weight[self.config.pad_token_id].zero_()
        self.segment_embeddings = nn.Embedding(3, self.hidden_size)
        self.dna_embeddings = nn.Embedding(dna_vocab_size, self.hidden_size)
        self.protein_lm_head = nn.Linear(self.hidden_size, self.config.vocab_size)
        self.dna_lm_head = nn.Linear(self.hidden_size, dna_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._initialize_new_parameters()

    @classmethod
    def from_esm_pretrained(cls, model_name_or_path, dna_vocab_size=8, dropout=0.1):
        return cls(EsmModel.from_pretrained(model_name_or_path), dna_vocab_size, dropout)

    @classmethod
    def from_esm_config(cls, config, dna_vocab_size=8, dropout=0.1):
        return cls(EsmModel(config), dna_vocab_size, dropout)

    def _initialize_new_parameters(self):
        for module in [self.segment_embeddings, self.dna_embeddings,
                       self.protein_lm_head, self.dna_lm_head]:
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def get_embeddings(self, input_ids, segment_ids, token_type_ids):
        batch_size, sequence_length = input_ids.shape
        embeddings = torch.zeros(
            batch_size, sequence_length, self.hidden_size,
            device=input_ids.device,
            dtype=self.esm2.embeddings.word_embeddings.weight.dtype,
        )
        protein_mask = token_type_ids == 0
        if protein_mask.any():
            protein_ids = input_ids.clone()
            protein_ids[~protein_mask] = self.config.pad_token_id
            protein_embeddings = self.esm2.embeddings.word_embeddings(protein_ids)
            embeddings[protein_mask] = protein_embeddings[protein_mask]
        dna_mask = token_type_ids == 1
        if dna_mask.any():
            embeddings[dna_mask] = self.dna_embeddings(input_ids[dna_mask])
        embeddings = embeddings + self.segment_embeddings(segment_ids)
        position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = embeddings + self.esm2.embeddings.position_embeddings(position_ids)
        return self.dropout(embeddings)

    def forward(self, input_ids, segment_ids, token_type_ids, attention_mask=None):
        embeddings = self.get_embeddings(input_ids, segment_ids, token_type_ids)
        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :].to(embeddings.dtype)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None
        output = self.esm2.encoder(embeddings, attention_mask=extended_mask).last_hidden_state
        return {
            "last_hidden_state": output,
            "protein_logits": self.protein_lm_head(output),
            "dna_logits": self.dna_lm_head(output),
        }
