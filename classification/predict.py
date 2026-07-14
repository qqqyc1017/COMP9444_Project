from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from data import build_transforms
from engine import get_device
from models import create_model


def main():
    p = argparse.ArgumentParser(description="Classify one insect image")
    p.add_argument("--checkpoint", required=True); p.add_argument("--image", required=True)
    p.add_argument("--top-k", type=int, default=5); p.add_argument("--device", default="auto")
    args = p.parse_args(); device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = ckpt["args"]; classes = ckpt["classes"]
    model = create_model(saved["model"], len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    _, transform = build_transforms(saved.get("image_size", 224))
    with Image.open(Path(args.image)) as image:
        x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = model(x).softmax(1)[0]
    values, indices = probs.topk(min(args.top_k, len(classes)))
    for value, index in zip(values.cpu(), indices.cpu()):
        print(f"{classes[index]}\t{value.item():.4%}")


if __name__ == "__main__":
    main()

