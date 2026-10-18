"""Download approved action-model weights once; inference never downloads anything."""

import hashlib
import json

from src.config import WEIGHTS


def main():
    import torch
    from huggingface_hub import snapshot_download
    from torchvision.models.video import R3D_18_Weights

    snapshot_download(
        "microsoft/xclip-base-patch32",
        local_dir=WEIGHTS / "xclip-base-patch32",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "pytorch_model.bin"],
    )
    target = WEIGHTS / "r3d-18"
    target.mkdir(parents=True, exist_ok=True)
    weights = R3D_18_Weights.KINETICS400_V1
    file = target / "r3d_18-b3b3357e.pth"
    if not file.exists():
        torch.hub.download_url_to_file(weights.url, str(file), hash_prefix="b3b3357e")
    if not hashlib.sha256(file.read_bytes()).hexdigest().startswith("b3b3357e"):
        raise ValueError("R3D weight checksum mismatch")
    (target / "categories.json").write_text(json.dumps(weights.meta["categories"]), encoding="utf-8")
    print("Action weights ready locally; licenses in docs/MODELS.md")


if __name__ == "__main__":
    main()
