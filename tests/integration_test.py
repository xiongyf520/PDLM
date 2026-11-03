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
        generated = work / "generated"
        run(ROOT / "scripts" / "generate_demo.py", "--output-dir", generated)
        run(ROOT / "scripts" / "generate_demo.py", "--output-dir", generated)
        partitions = []
        for split, count in [("train", 80), ("val", 20)]:
            original = ROOT / "examples" / f"toy_{split}.jsonl"
            assert original.read_bytes() == (generated / original.name).read_bytes()
            records = [json.loads(line) for line in original.read_text().splitlines()]
            assert len(records) == count
            assert sum(record["has_annotation"] for record in records) == count // 2
            for record in records:
                if record["has_annotation"]:
                    assert len(record["binding_mask"]) == len(record["protein_seq"])
                    assert sum(record["binding_mask"]) == 5
            partitions.append(records)
        for field in ["id", "protein_seq", "dna_seq"]:
            assert len({record[field] for records in partitions for record in records}) == 100
        (generated / "toy_val.jsonl").write_text("keep existing data\n")
        refused = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_demo.py"),
             "--output-dir", str(generated)], cwd=ROOT, text=True, capture_output=True
        )
        assert refused.returncode != 0
        assert (generated / "toy_val.jsonl").read_text() == "keep existing data\n"

        config_path = work / "train.yaml"
        for strategy in ["base", "domain"]:
            training = yaml.safe_load((ROOT / "configs" / "demo" / f"{strategy}.yaml").read_text())
            assert training["metadata"]["config_status"] == "synthetic_demo_only"
            training["paths"] = {
                "root": ".", "train_file": "train.jsonl", "val_file": "val.jsonl",
                "output_dir": f"run_{strategy}",
            }
            training["model"]["esm2_model"] = "esm2"
            training["model"]["dropout"] = 0.0
            config_path.write_text(yaml.safe_dump(training))
            run("-m", "pdlm.train", "--config", config_path)
            run_dir = work / f"run_{strategy}"
            checkpoint = run_dir / "checkpoint_best.pt"
            assert checkpoint.exists()
            history = json.loads((run_dir / "metrics.json").read_text())
            assert history[0]["epoch"] == 1
            assert all(np.isfinite(value) for value in history[0].values())
        training["training"]["epochs"] = 2
        config_path.write_text(yaml.safe_dump(training))
        run("-m", "pdlm.train", "--config", config_path, "--resume", checkpoint)
        resumed = json.loads((run_dir / "metrics.json").read_text())
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
    print("PDLM offline integration test passed: deterministic demo data, base/domain training, resume, strict checkpoint loading, export, scoring, embeddings")


if __name__ == "__main__":
    main()
