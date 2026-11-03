import argparse
import json
from pathlib import Path
import random


def generate_records():
    rng = random.Random(42)
    records = []
    proteins = set()
    dna_sequences = set()
    while len(records) < 100:
        protein = "".join(rng.choices("ACDEFGHIKLMNPQRSTVWY", k=rng.randint(20, 40)))
        dna = "".join(rng.choices("ACGT", k=rng.randint(8, 16)))
        if protein in proteins or dna in dna_sequences:
            continue
        proteins.add(protein)
        dna_sequences.add(dna)
        index = len(records)
        record = {
            "id": f"synthetic_{index + 1:03d}",
            "protein_seq": protein,
            "dna_seq": dna,
            "has_annotation": index % 2 == 0,
        }
        if record["has_annotation"]:
            start = rng.randint(0, len(protein) - 5)
            record["binding_mask"] = [
                int(start <= position < start + 5) for position in range(len(protein))
            ]
        records.append(record)
    return records[:80], records[80:]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "examples")
    args = parser.parse_args(argv)
    paths = [args.output_dir / f"toy_{split}.jsonl" for split in ["train", "val"]]
    contents = [
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
        for records in generate_records()
    ]
    for path, content in zip(paths, contents):
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f"{path} differs from the generated demo; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in zip(paths, contents):
        path.write_text(content)
    print("Generated 80 training and 20 validation records (synthetic, seed 42)")


if __name__ == "__main__":
    main()
