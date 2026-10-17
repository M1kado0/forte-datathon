"""`make weights`: download every model once into models/weights, then load each one offline."""

import os
import subprocess
import sys

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForObjectDetection,
    AutoProcessor,
    Owlv2ForObjectDetection,
)

from src.config import WEIGHTS

# name -> (HF repo, model class, processor class). All weights are Apache-2.0, see docs/MODELS.md.
WEIGHT_REPOS = {
    "dinov2-small": ("facebook/dinov2-small", AutoModel, AutoImageProcessor),
    "siglip-base": ("google/siglip-base-patch16-224", AutoModel, AutoProcessor),
    "dfine-small": ("ustc-community/dfine-small-coco", AutoModelForObjectDetection, AutoImageProcessor),
    "owlv2-base": ("google/owlv2-base-patch16-ensemble", Owlv2ForObjectDetection, AutoProcessor),
}


def is_complete(name: str) -> bool:
    folder = WEIGHTS / name
    return (folder / "config.json").is_file() and any(folder.glob("*.safetensors"))


def download() -> None:
    from huggingface_hub import snapshot_download

    for name, (repo, _, _) in WEIGHT_REPOS.items():
        if is_complete(name):
            print(f"skip {name}: already in models/weights")
            continue
        print(f"download {repo} -> models/weights/{name}")
        snapshot_download(
            repo, local_dir=WEIGHTS / name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
        )


def check_offline() -> None:
    for name, (_, model_cls, processor_cls) in WEIGHT_REPOS.items():
        model = model_cls.from_pretrained(WEIGHTS / name)
        processor_cls.from_pretrained(WEIGHTS / name)
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"ok  {name}: {n:.1f}M params + processor, loaded offline")


if __name__ == "__main__":
    if "--check" in sys.argv:
        # run as `HF_HUB_OFFLINE=1 python -m scripts.weights --check` (the flag must be set before imports)
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            sys.exit("set HF_HUB_OFFLINE=1 to check offline loading")
        check_offline()
    else:
        download()
        # fresh process, so the offline flag is honoured from the first import
        subprocess.run(
            [sys.executable, "-m", "scripts.weights", "--check"],
            check=True,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
