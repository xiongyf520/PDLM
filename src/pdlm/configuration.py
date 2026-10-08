from pathlib import Path

import yaml


DNA_VOCAB = {
    "dA": 0,
    "dT": 1,
    "dC": 2,
    "dG": 3,
    "dN": 4,
    "[MASK]": 5,
    "[UNK]": 6,
    "[SEP]": 7,
}

SEGMENT_IDS = {"special": 0, "protein": 1, "dna": 2}


def load_yaml(path):
    path = Path(path).resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle)
    return config, path


def repository_root(config, config_path):
    root = config.get("paths", {}).get("root", "../..")
    return (config_path.parent / root).resolve()


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()

