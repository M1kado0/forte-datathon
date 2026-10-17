"""`make weights`: download every model once into models/weights, then load each one offline."""

import os
import sys

from src.config import WEIGHTS

# name -> (HF repo, loader). All weights are Apache-2.0, see docs/MODELS.md.
MODELS = {
    "dinov2-small": ("facebook/dinov2-small", "AutoModel"),
    "siglip-base": ("google/siglip-base-patch16-224", "AutoModel"),
    "dfine-small": ("ustc-community/dfine-small-coco", "AutoModelForObjectDetection"),
    "owlv2-base": ("google/owlv2-base-patch16-ensemble", "Owlv2ForObjectDetection"),
}


def download() -> None:
    from huggingface_hub import snapshot_download

    for name, (repo, _) in MODELS.items():
        print(f"download {repo} -> models/weights/{name}")
        snapshot_download(
            repo, local_dir=WEIGHTS / name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
        )


def check_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    import transformers

    for name, (_, loader) in MODELS.items():
        model = getattr(transformers, loader).from_pretrained(WEIGHTS / name)
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"ok  {name}: {n:.1f}M params, loaded offline")


if __name__ == "__main__":
    if "--check" not in sys.argv:
        download()
    check_offline()
