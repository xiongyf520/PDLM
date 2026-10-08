import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from transformers import EsmConfig, EsmModel, EsmTokenizer
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdlm.checkpoints import load_checkpoint
from pdlm.scoring import dna_pll


def run(*args):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with TemporaryDirectory(prefix="pdlm_integration_") as directory:
        work = Path(directory)
        backbone = work / "esm2"
        backbone.mkdir()
        vocabulary = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V",
                      "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y",
                      "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]
        vocab_file = backbone / "vocab.txt"
        vocab_file.write_text("\n".join(vocabulary) + "\n")
        tokenizer = EsmTokenizer(vocab_file=str(vocab_file))
        tokenizer.save_pretrained(backbone)
        config = EsmConfig(vocab_size=33, hidden_size=32, num_hidden_layers=1,
                           num_attention_heads=4, intermediate_size=64,
                           max_position_embeddings=128, pad_token_id=1,
                           mask_token_id=32, position_embedding_type="rotary",
                           token_dropout=False)
        EsmModel(config).save_pretrained(backbone)
        config_only = work / "config_only"
        config.save_pretrained(config_only)
        tokenizer.save_pretrained(config_only)
        for split in ["train", "val"]:
            shutil.copy2(ROOT / "examples" / f"toy_{split}.jsonl", work / f"{split}.jsonl")
        training = {
            "paths": {"root": ".", "train_file": "train.jsonl", "val_file": "val.jsonl", "output_dir": "run"},
            "model": {"esm2_model": "esm2", "dna_vocab_size": 8, "dropout": 0.0, "max_length": 64},
            "masking": {"strategy": "domain", "binding_probability": 0.3,
                        "other_probability": 0.1, "fallback_probability": 0.15, "dna_probability": 0.2},
            "training": {"seed": 42, "epochs": 1, "batch_size": 2, "learning_rate": 0.001,
                         "warmup_steps": 0, "reverse_complement_probability": 0.3, "swap_order_probability": 0.5},
        }
        config_path = work / "train.yaml"
        config_path.write_text(yaml.safe_dump(training))
        run("-m", "pdlm.train", "--config", config_path)
        checkpoint = work / "run" / "checkpoint_best.pt"
        assert checkpoint.exists()
        history = json.loads((work / "run" / "metrics.json").read_text())
        assert np.isfinite(history[0]["val_loss"])
        training["training"]["epochs"] = 2
        config_path.write_text(yaml.safe_dump(training))
        run("-m", "pdlm.train", "--config", config_path, "--resume", checkpoint)
        resumed = json.loads((work / "run" / "metrics.json").read_text())
        assert resumed[0]["epoch"] == 2
        model, _ = load_checkpoint(checkpoint, config_only)
        weights = work / "model.safetensors"
        run(ROOT / "scripts" / "export_weights.py", "--checkpoint", checkpoint, "--output", weights)
        exported, _ = load_checkpoint(weights, config_only)
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, exported.state_dict()[name]), name
        first = dna_pll(model, tokenizer, "MKTAYIAK", "ATCG", max_length=64)
        second = dna_pll(exported, tokenizer, "MKTAYIAK", "ATCG", max_length=64)
        assert first == second

        for mode, input_name in [
            ("protein-dll", "protein_only_variants.jsonl"),
            ("protein-dll", "protein_variants.jsonl"),
            ("dna-pll", "toy_pairs.jsonl"),
            ("dna-dpll", "dna_delta_pll.jsonl"),
        ]:
            output = work / f"{mode}_{input_name}"
            run("-m", "pdlm.scoring", "--checkpoint", weights, "--esm2-model", config_only,
                "--input", ROOT / "examples" / input_name, "--output", output, "--mode", mode)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            assert records and all(np.isfinite(value) for record in records
                                   for value in record.values() if isinstance(value, (int, float)))
        for mode, input_name, width in [
            ("concat", "toy_pairs.jsonl", 64),
            ("protein_mean", "protein_only_variants.jsonl", 32),
        ]:
            output = work / f"{mode}.npz"
            run("-m", "pdlm.embeddings", "--checkpoint", weights, "--esm2-model", config_only,
                "--input", ROOT / "examples" / input_name, "--output", output, "--mode", mode)
            with np.load(output) as data:
                assert data["embeddings"].shape[1] == width
                assert np.isfinite(data["embeddings"]).all()
    print("PDLM offline integration test passed: training, resume, strict checkpoint loading, export, scoring, embeddings")


if __name__ == "__main__":
    main()
