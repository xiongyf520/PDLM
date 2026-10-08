import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .configuration import load_yaml, repository_root, resolve_path
from .data import ProteinDNADataset
from .masking import ProteinDNACollator
from .modeling import ProteinDNAMLMModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_loss(logits, labels):
    valid = labels != -100
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid])


def metrics_from_batch(outputs, batch, protein_weight, dna_weight):
    protein_loss = masked_loss(outputs["protein_logits"], batch["protein_labels"])
    dna_loss = masked_loss(outputs["dna_logits"], batch["dna_labels"])
    loss = protein_weight * protein_loss + dna_weight * dna_loss
    return loss, protein_loss, dna_loss


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, loader, device, protein_weight, dna_weight):
    model.eval()
    totals = {"loss": 0.0, "protein_loss": 0.0, "dna_loss": 0.0,
              "protein_correct": 0, "protein_n": 0, "dna_correct": 0, "dna_n": 0}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch["input_ids"], batch["segment_ids"],
                            batch["token_type_ids"], batch["attention_mask"])
            loss, protein_loss, dna_loss = metrics_from_batch(
                outputs, batch, protein_weight, dna_weight
            )
            totals["loss"] += loss.item()
            totals["protein_loss"] += protein_loss.item()
            totals["dna_loss"] += dna_loss.item()
            for modality in ["protein", "dna"]:
                labels = batch[f"{modality}_labels"]
                valid = labels != -100
                predictions = outputs[f"{modality}_logits"].argmax(-1)
                totals[f"{modality}_correct"] += ((predictions == labels) & valid).sum().item()
                totals[f"{modality}_n"] += valid.sum().item()
    batches = max(1, len(loader))
    return {
        "val_loss": totals["loss"] / batches,
        "val_protein_loss": totals["protein_loss"] / batches,
        "val_dna_loss": totals["dna_loss"] / batches,
        "val_protein_accuracy": totals["protein_correct"] / max(1, totals["protein_n"]),
        "val_dna_accuracy": totals["dna_correct"] / max(1, totals["dna_n"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)

    config, config_path = load_yaml(args.config)
    root = repository_root(config, config_path)
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_source = config["model"]["esm2_model"]
    relative_model = root / model_source
    if relative_model.exists():
        model_source = str(relative_model)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = ProteinDNAMLMModel.from_esm_pretrained(
        model_source,
        dna_vocab_size=int(config["model"].get("dna_vocab_size", 8)),
        dropout=float(config["model"].get("dropout", 0.1)),
    ).to(device)

    paths = config["paths"]
    train_data = ProteinDNADataset(
        resolve_path(root, paths["train_file"]), training=True,
        reverse_complement_prob=float(config["training"].get("reverse_complement_probability", 0.0)),
        swap_order_prob=float(config["training"].get("swap_order_probability", 0.0)), seed=seed,
    )
    val_data = ProteinDNADataset(resolve_path(root, paths["val_file"]), training=False, seed=seed)
    train_collator = ProteinDNACollator(tokenizer, config["masking"],
                                        int(config["model"].get("max_length", 800)), seed)
    val_collator = ProteinDNACollator(tokenizer, config["masking"],
                                      int(config["model"].get("max_length", 800)), seed + 1)
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=train_collator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            collate_fn=val_collator, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    epochs = int(config["training"]["epochs"])
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config["training"].get("warmup_steps", 0)), total_steps
    )
    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_loss = float(payload.get("metrics", {}).get("val_loss", best_loss))

    output_dir = resolve_path(root, paths["output_dir"])
    if not args.resume and any((output_dir / name).exists() for name in ["metrics.json", "checkpoint_best.pt"]):
        raise FileExistsError("Output directory contains a previous run; choose a new output_dir or use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    history = []
    protein_weight = float(config.get("loss", {}).get("protein_weight", 0.5))
    dna_weight = float(config.get("loss", {}).get("dna_weight", 0.5))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            batch = move_batch(batch, device)
            outputs = model(batch["input_ids"], batch["segment_ids"],
                            batch["token_type_ids"], batch["attention_mask"])
            loss, _, _ = metrics_from_batch(outputs, batch, protein_weight, dna_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(config["training"].get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            running += loss.item()
        metrics = evaluate(model, val_loader, device, protein_weight, dna_weight)
        metrics.update({"epoch": epoch, "train_loss": running / max(1, len(train_loader))})
        history.append(metrics)
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2))
        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": metrics,
                "config": config,
            }, output_dir / "checkpoint_best.pt")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
