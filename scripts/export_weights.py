import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import save_file

from pdlm.checkpoints import _state_dict


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.suffix != ".safetensors":
        raise ValueError("Output must end in .safetensors")
    if output.exists():
        raise FileExistsError("Output already exists; choose a new filename")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = _state_dict(payload)
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Checkpoint must contain a model state dictionary")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().cpu().contiguous() for key, value in state.items()}, str(output))
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    print(f"Exported {len(state)} tensors to {output.name}")
    print(f"SHA256: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
