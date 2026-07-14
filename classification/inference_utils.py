from __future__ import annotations

from pathlib import Path

import torch

from data import build_transforms
from engine import get_device
from models import create_model


def load_classifier(checkpoint_path: str | Path, requested_device: str = "auto"):
    """Load a trained classifier and the matching validation transform."""
    device = get_device(requested_device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint["args"]
    classes = checkpoint["classes"]
    model = create_model(saved_args["model"], len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    _, transform = build_transforms(saved_args.get("image_size", 224))
    return model, classes, transform, device


@torch.inference_mode()
def classify_tensor_batch(model, batch: torch.Tensor, classes, device, top_k: int = 5):
    probabilities = model(batch.to(device)).softmax(1)
    values, indices = probabilities.topk(min(top_k, len(classes)), dim=1)
    results = []
    for row_values, row_indices in zip(values.cpu(), indices.cpu()):
        results.append([
            {"class": classes[index.item()], "confidence": float(value.item())}
            for value, index in zip(row_values, row_indices)
        ])
    return results

